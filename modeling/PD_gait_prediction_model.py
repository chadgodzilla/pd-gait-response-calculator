# -*- coding: utf-8 -*-
# PD_gait_prediction_model.py
# Part of: PD Gait Response Calculator (https://github.com/chadgodzilla/pd-gait-response-calculator)
# Archived: https://doi.org/10.5281/zenodo.24248978   License: MIT
"""
Prediction-model analysis for gait-speed change after training in Parkinson's disease (n = 44).

Reproduces, from one participant-level spreadsheet:
  Table 1    Demographic and clinical characteristics
  Table S1   Univariate screening (p < 0.05, VIF < 5), multivariable and final parsimonious OLS models
  Table S2   Incremental value of WPD spectral features (hierarchical blocks, F-change test)
  Table S3   Bootstrap stability of LASSO predictor selection
  Table 2    Nested LOOCV (primary) and nested repeated 5-fold CV (secondary) for
             linear, ridge, LASSO and elastic-net regression
  Table S4   Apparent performance, apparent-to-LOOCV R² gap and calibration (incl. random forest)
  Figure 2   Observed vs predicted gait-speed change, final ridge model (apparent and nested LOOCV)
  Figure 3   SHAP predictor contributions, final ridge model
  Sensitivity analysis   post-training gait speed modeled directly (ANCOVA-style), nested LOOCV
  calculator_coefficients.json   final ridge model in raw units (used by index.html)

At the end the script compares its results with the values reported in the article
(section "SELF-CHECK") so that any discrepancy is visible immediately.

Usage:   python PD_gait_prediction_model.py
Requires: numpy, pandas, scipy, statsmodels, scikit-learn, shap, matplotlib, openpyxl
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
DATA_FILE = "PDgait_IMU_WPD_all_TLLee_20260911.xlsx"    # participant-level data (not public)
SHEET     = "Baseline _IMU_WPD_outcome "                 # falls back to the first sheet if absent
OUT_DIR   = "results_prediction_model"

OUTCOME = "diff_Speed"                                   # post - pre comfortable gait speed (m/s)
POST_OUTCOME = "Speed_ST_post"                           # sensitivity analysis: follow-up gait speed (m/s)

# candidate predictors (column name -> label); MMSE excluded (collinear with MoCA)
CANDIDATES = {
    "Age": "Age",
    "PT": "PT exposure",
    "Sex": "Sex",
    "LEDD": "LEDD",
    "UPDRS": "UPDRS",
    "MoCA": "MoCA",
    "Speed_ST_pre": "Speed",
    "Stride_length_ST_pre": "Stride length",
    "Cadence_ST_pre": "Cadence",
    "Stride Time CoV_pre": "Stride time variability",
    "locomotor_ST_pre": "Locomotor energy",
    "HF_ST_pre": "Freeze energy",
    "HF_index_ST_pre": "Freeze index",
}
FINAL = ["PT", "MoCA", "Speed_ST_pre"]                   # final parsimonious predictor set
WPD_FEATURES = ["locomotor_ST_pre", "HF_ST_pre", "HF_index_ST_pre"]
DISPLAY = {"PT": "Physical therapy", "MoCA": "MoCA", "Speed_ST_pre": "Baseline gait speed"}

# Table 1 rows (column name, label, type); rows whose column is absent are skipped
DESCRIPTIVE = [
    ("Age", "Age", "cont"), ("Sex", "Male, n (%)", "binary"),
    ("Duration", "Disease duration (years)", "cont"),
    ("MoCA", "MoCA (score)", "cont"), ("LEDD", "LEDD (mg)", "cont"),
    ("UPDRS", "UPDRS III score", "cont"), ("PT", "PT exposure, n (%)", "binary"),
    ("Speed_ST_pre", "Speed (m/s)", "cont3"), ("Stride_length_ST_pre", "Stride length (m)", "cont3"),
    ("Cadence_ST_pre", "Cadence (steps/min)", "cont"),
    ("Stride Time CoV_pre", "Stride time variability (CoV, %)", "cont2"),
    ("Asymmetry (stride time)_pre", "Stride time asymmetry (%)", "cont2"),
    ("Asymmetry (stride length)_pre", "Stride length asymmetry (%)", "cont2"),
    ("locomotor_ST_pre", "Locomotor energy (a.u.)", "cont2"),
    ("HF_ST_pre", "Freeze energy (a.u.)", "cont2"), ("HF_index_ST_pre", "Freeze index", "cont3"),
]

P_SCREEN = 0.05
VIF_MAX = 5.0
SEED = 42
INNER_CV = 5                                   # inner folds (shuffled, seed 42) for hyperparameter tuning
RIDGE_ALPHAS = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
LASSO_ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1]
ENET_ALPHAS  = [0.001, 0.005, 0.01, 0.05, 0.1]
ENET_L1      = [0.1, 0.3, 0.5, 0.7, 0.9]
N_REPEATS = 10                                 # repeated 5-fold CV
N_BOOT_LASSO = 1000
LASSO_STABILITY_ALPHA = 0.02                   # fixed penalty (standardized predictors) for Table S3
N_BOOT_BAND = 1000                             # bootstrap CI band of the calibration line
RF_PARAMS = dict(n_estimators=300, max_depth=3, min_samples_leaf=3, max_features="sqrt", random_state=SEED)

# values reported in the article, for the self-check
REPORTED = {
    "LOOCV": {"Ridge regression": (0.378, 0.137, 0.107, 1.046), "Linear regression": (0.360, 0.139, 0.109, 0.878),
              "LASSO regression": (0.375, 0.137, 0.107, 0.967), "Elastic net": (0.379, 0.137, 0.107, 0.988)},
    "CV5_median_R2": {"Ridge regression": 0.340, "Linear regression": 0.358, "LASSO regression": 0.352, "Elastic net": 0.343},
    "Apparent_R2": {"Ridge regression": 0.4511, "Linear regression": 0.4576, "LASSO regression": 0.4548,
                    "Elastic net": 0.4506, "Random forest": 0.5672},
    "RF_LOOCV_R2": 0.2705,
    "S1_multivariable_R2": 0.461, "S1_final_R2": 0.458,
    "S2_p_change": {"locomotor_ST_pre": 0.829, "HF_ST_pre": 0.740, "HF_index_ST_pre": 0.451},
    "S3_inclusion": {"MoCA": 99.3, "Speed_ST_pre": 97.0, "PT": 81.8, "Cadence_ST_pre": 58.6},
    "Fig2_apparent_slope": 1.132,
    "Sensitivity_post_speed": (0.453, 1.005),
    "Fig3_mean_abs_shap": {"Speed_ST_pre": 0.0638, "MoCA": 0.0529, "PT": 0.0388},
}
# ====================================================================

os.makedirs(OUT_DIR, exist_ok=True)


def load():
    try:
        df = pd.read_excel(DATA_FILE, sheet_name=SHEET)
    except ValueError:
        df = pd.read_excel(DATA_FILE, sheet_name=0)
    df.columns = df.columns.astype(str).str.strip()
    need = [OUTCOME] + list(CANDIDATES)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found in {DATA_FILE}: {missing}")
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=need).reset_index(drop=True)
    print(f"n = {len(df)}" + (f"  ({n0 - len(df)} row(s) dropped for missing data)" if len(df) < n0 else ""))
    return df


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_b(b, se):
    return f"{b:+.3f} ± {se:.3f}"


def ols(df, xs):
    return sm.OLS(df[OUTCOME], sm.add_constant(df[xs])).fit()


def vif(df, xs):
    X = sm.add_constant(df[xs]).values
    return {x: variance_inflation_factor(X, i + 1) for i, x in enumerate(xs)}


# ----------------------------- Table 1 -----------------------------
def table1(df):
    rows = []
    for col, lab, kind in DESCRIPTIVE:
        if col not in df.columns:
            print(f"  Table 1: column '{col}' not found - row '{lab}' skipped")
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if kind == "binary":
            rows.append((lab, f"{int(v.sum())} ({100 * v.mean():.1f})"))
        else:
            d = {"cont": 1, "cont2": 2, "cont3": 3}[kind]
            rows.append((lab, f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"))
    return pd.DataFrame(rows, columns=["Variable", f"Patients (n = {len(df)})"])


# ----------------------------- Table S1 ----------------------------
def table_s1(df):
    uni = {c: ols(df, [c]) for c in CANDIDATES}
    screened = [c for c in CANDIDATES if uni[c].pvalues[c] < P_SCREEN]
    v = vif(df, screened)
    screened = [c for c in screened if v[c] < VIF_MAX]
    multi = ols(df, screened)
    final = list(screened)                       # backward elimination of non-significant predictors
    while True:
        fit = ols(df, final)
        p = fit.pvalues.drop("const")
        if p.max() < P_SCREEN or len(final) == 1:
            break
        final.remove(p.idxmax())
    fin = ols(df, final)
    rows = []
    for c, lab in CANDIDATES.items():
        r = {"Variable": lab, "Univariate B ± SE": fmt_b(uni[c].params[c], uni[c].bse[c]),
             "Univariate p": fmt_p(uni[c].pvalues[c]), "VIF": f"{v[c]:.2f}" if c in v else "—",
             "Multivariable B ± SE": fmt_b(multi.params[c], multi.bse[c]) if c in screened else "—",
             "Multivariable p": fmt_p(multi.pvalues[c]) if c in screened else "—",
             "Final B ± SE": fmt_b(fin.params[c], fin.bse[c]) if c in final else "—",
             "Final p": fmt_p(fin.pvalues[c]) if c in final else "—"}
        rows.append(r)
    fit_txt = lambda m: f"R² = {m.rsquared:.3f}, adj. R² = {m.rsquared_adj:.3f}, F = {m.fvalue:.2f}, p = {fmt_p(m.f_pvalue)}"
    rows.append({"Variable": "Model fit", "Multivariable B ± SE": fit_txt(multi), "Final B ± SE": fit_txt(fin)})
    if set(final) != set(FINAL):
        print(f"  NOTE: data-driven final set {final} differs from FINAL {FINAL}")
    return pd.DataFrame(rows), multi, fin


# ----------------------------- Table S2 ----------------------------
def table_s2(df):
    b1 = ols(df, FINAL)
    vifs = vif(df, FINAL + WPD_FEATURES)
    rows = [{"Block": "Block 1 (final model)", "R2": b1.rsquared, "adj_R2": b1.rsquared_adj, "F": b1.fvalue,
             "dR2": np.nan, "F_change": np.nan, "p_change": np.nan}]
    coefs = []
    for f in WPD_FEATURES:
        m = ols(df, FINAL + [f])
        f_ch, p_ch, _ = m.compare_f_test(b1)
        rows.append({"Block": f"Block 2: + {CANDIDATES[f]}", "R2": m.rsquared, "adj_R2": m.rsquared_adj, "F": m.fvalue,
                     "dR2": m.rsquared - b1.rsquared, "F_change": f_ch, "p_change": p_ch,
                     "VIF_feature_in_block": vif(df, FINAL + [f])[f]})
        for c in FINAL + [f]:
            coefs.append({"Block": f"+ {CANDIDATES[f]}", "Variable": CANDIDATES[c],
                          "B ± SE": fmt_b(m.params[c], m.bse[c]), "p": fmt_p(m.pvalues[c])})
    vtab = pd.DataFrame([{"Variable": CANDIDATES[k], "VIF (all six together)": round(v, 2)} for k, v in vifs.items()])
    return pd.DataFrame(rows), pd.DataFrame(coefs), vtab


# ----------------------------- Table S3 ----------------------------
def table_s3(df):
    X = df[list(CANDIDATES)].to_numpy(float)
    y = df[OUTCOME].to_numpy(float)
    rng = np.random.default_rng(SEED)
    sel = np.zeros((N_BOOT_LASSO, X.shape[1]), dtype=bool)
    sign = np.zeros((N_BOOT_LASSO, X.shape[1]))
    for b in range(N_BOOT_LASSO):
        i = rng.choice(len(y), len(y), replace=True)
        Xs = StandardScaler().fit_transform(X[i])
        m = Lasso(alpha=LASSO_STABILITY_ALPHA, max_iter=50000).fit(Xs, y[i])
        sel[b] = m.coef_ != 0
        sign[b] = np.sign(m.coef_)
    rows = []
    for j, c in enumerate(CANDIDATES):
        s = sign[sel[:, j], j]
        rows.append({"Variable": CANDIDATES[c], "column": c, "Inclusion frequency (%)": round(100 * sel[:, j].mean(), 1),
                     "Direction of effect": ("Positive" if (s > 0).sum() >= (s < 0).sum() else "Negative") if len(s) else "—"})
    return pd.DataFrame(rows).sort_values("Inclusion frequency (%)", ascending=False).reset_index(drop=True)


# ------------------------ Models and validation ---------------------
def make_model(name):
    """Scaler + estimator. Penalized models are tuned by grid search (inner shuffled 5-fold CV,
    mean squared error) on whatever data the model is fitted to, so that inside an outer
    cross-validation loop the tuning uses the training fold only (nested CV)."""
    pipe = lambda est: Pipeline([("scaler", StandardScaler()), ("reg", est)])
    if name == "Linear regression":
        return pipe(LinearRegression())
    if name == "Random forest":
        return pipe(RandomForestRegressor(**RF_PARAMS))
    est, grid = {"Ridge regression": (Ridge(), {"reg__alpha": RIDGE_ALPHAS}),
                 "LASSO regression": (Lasso(max_iter=50000), {"reg__alpha": LASSO_ALPHAS}),
                 "Elastic net": (ElasticNet(max_iter=50000), {"reg__alpha": ENET_ALPHAS, "reg__l1_ratio": ENET_L1})}[name]
    return GridSearchCV(pipe(est), grid, cv=KFold(INNER_CV, shuffle=True, random_state=SEED),
                        scoring="neg_mean_squared_error")


MODELS = ["Ridge regression", "Linear regression", "LASSO regression", "Elastic net"]


def metrics(y, p):
    cal = sm.OLS(y, sm.add_constant(p)).fit().params          # observed regressed on predicted
    return {"R2": r2_score(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "cal_intercept": cal[0], "cal_slope": cal[1]}


def nested_loocv(name, X, y):
    pred = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        pred[te] = make_model(name).fit(X[tr], y[tr]).predict(X[te])   # tuning inside the training fold
    return pred


def nested_repeated_cv(name, X, y):
    per_rep = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rkf.split(X))
    for r in range(N_REPEATS):
        pred = np.empty(len(y))
        for tr, te in splits[5 * r:5 * r + 5]:
            pred[te] = make_model(name).fit(X[tr], y[tr]).predict(X[te])
        m = metrics(y, pred)
        per_rep.append({"repeat": r + 1, "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"]})
    return pd.DataFrame(per_rep)


def med_iqr(s, d=3):
    return f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}–{s.quantile(.75):.{d}f}]"


# ----------------------------- Figures -----------------------------
def calib_panel(ax, y, p, title, label, color):
    m = metrics(y, p)
    lim = [min(y.min(), p.min()) - 0.07, max(y.max(), p.max()) + 0.07]
    xs = np.linspace(*lim, 200)
    rng = np.random.default_rng(SEED)
    band = []
    for _ in range(N_BOOT_BAND):
        i = rng.integers(0, len(y), len(y))
        c = np.polyfit(p[i], y[i], 1)
        band.append(np.polyval(c, xs))
    band = np.array(band)
    ax.fill_between(xs, np.percentile(band, 2.5, 0), np.percentile(band, 97.5, 0), color=color, alpha=.2,
                    label="95% bootstrap CI")
    ax.plot(lim, lim, "k--", lw=1, label="Perfect calibration\n(slope=1, int=0)")
    ax.plot(xs, m["cal_intercept"] + m["cal_slope"] * xs, color=color, lw=2,
            label=f"Calibration line\n(α={m['cal_intercept']:.3f}, β={m['cal_slope']:.3f})")
    ax.scatter(p, y, s=55, color=color, edgecolor="white", alpha=.85, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted speed change (m/s)"); ax.set_ylabel("Observed speed change (m/s)")
    ax.set_title(title)
    ax.text(.03, .97, f"{label} R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f} m/s\nMAE  = {m['MAE']:.3f} m/s\n"
            f"Cal α={m['cal_intercept']:.3f}  β={m['cal_slope']:.3f}\nn = {len(y)}", transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="lightgrey"))
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return m


def figure2(y, p_app, p_loo):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    m_app = calib_panel(axes[0], y, p_app, "Apparent (full-sample fit)", "Apparent", "#1565C0")
    calib_panel(axes[1], y, p_loo, "Internal validation (nested LOOCV)", "Nested LOOCV", "#E65100")
    fig.suptitle("Observed-vs-Predicted Plot — Ridge Regression (nested LOOCV)\n"
                 "Outcome: speed change (m/s)  |  Predictors: PT, MoCA, Speed_ST_pre")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Figure2_observed_vs_predicted.png"), dpi=300)
    plt.close(fig)
    return m_app


def figure3(final_pipe, X):
    import shap
    Xs = final_pipe.named_steps["scaler"].transform(X)
    sv = shap.LinearExplainer(final_pipe.named_steps["reg"], Xs).shap_values(Xs)
    labels = [DISPLAY[c] for c in FINAL]
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([labels[i] for i in order], mean_abs[order], color="#1565C0", height=.5)
    for k, i in enumerate(order):
        ax.text(mean_abs[i] + .001, k, f"{mean_abs[i]:.4f}", va="center")
    ax.set_xlabel("Mean |SHAP value| (m/s)"); ax.set_title("Feature Importance — Ridge Regression")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "Figure3A_SHAP_importance.png"), dpi=300); plt.close(fig)
    shap.summary_plot(sv, pd.DataFrame(X, columns=labels), show=False, cmap="coolwarm", plot_size=(8, 3.8))
    plt.title("SHAP Values — Ridge Regression"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure3B_SHAP_beeswarm.png"), dpi=300); plt.close("all")
    return dict(zip(FINAL, mean_abs))


# ------------------------------- main ------------------------------
def main():
    df = load()
    y = df[OUTCOME].to_numpy(float)
    X = df[FINAL].to_numpy(float)
    out = {}

    print("Table 1 ..."); out["Table1"] = table1(df)
    print("Table S1 ..."); out["TableS1"], multi, fin = table_s1(df)
    print("Table S2 ..."); out["TableS2_fit"], out["TableS2_coef"], out["TableS2_VIF"] = table_s2(df)
    print(f"Table S3 ({N_BOOT_LASSO} bootstrap LASSO fits) ..."); out["TableS3"] = table_s3(df)

    print("Nested LOOCV and nested repeated 5-fold CV ...")
    loo_pred, t2, s4, cv_detail = {}, [], [], []
    for name in MODELS + ["Random forest"]:
        p = nested_loocv(name, X, y); loo_pred[name] = p
        ml = metrics(y, p)
        app_pipe = make_model(name).fit(X, y)
        ma = metrics(y, app_pipe.predict(X))
        s4.append({"Model": name, "Apparent R2": round(ma["R2"], 4), "LOOCV R2": round(ml["R2"], 4),
                   "R2 gap": round(ma["R2"] - ml["R2"], 4), "Apparent RMSE (m/s)": round(ma["RMSE"], 4),
                   "Calibration slope": "NA" if name == "Random forest" else round(ml["cal_slope"], 3)})
        if name == "Random forest":
            continue
        cv = nested_repeated_cv(name, X, y); cv.insert(0, "Model", name); cv_detail.append(cv)
        t2.append({"Model": name, "LOOCV R2": round(ml["R2"], 3), "LOOCV RMSE": round(ml["RMSE"], 3),
                   "LOOCV MAE": round(ml["MAE"], 3), "Calibration slope": round(ml["cal_slope"], 3),
                   "5-fold R2 median [IQR]": med_iqr(cv["R2"]), "5-fold RMSE median [IQR]": med_iqr(cv["RMSE"]),
                   "5-fold MAE median [IQR]": med_iqr(cv["MAE"]), "_cv5_med_r2": cv["R2"].median()})
        print(f"  {name:18s} LOOCV R2={ml['R2']:.3f}  5-fold median R2={cv['R2'].median():.3f}")
    out["Table2"] = pd.DataFrame(t2).drop(columns="_cv5_med_r2")
    out["TableS4"] = pd.DataFrame(s4)
    out["CV5_each_repeat"] = pd.concat(cv_detail)
    out["LOOCV_predictions"] = pd.DataFrame({"ID": df["ID"] if "ID" in df else np.arange(1, len(y) + 1),
                                             "observed": y, **{k: v for k, v in loo_pred.items()}})

    # final model: ridge fitted on the full sample
    final_pipe = make_model("Ridge regression").fit(X, y).best_estimator_
    sc, reg = final_pipe.named_steps["scaler"], final_pipe.named_steps["reg"]
    coef_raw = reg.coef_ / sc.scale_
    intercept_raw = reg.intercept_ - np.dot(reg.coef_, sc.mean_ / sc.scale_)
    calc = {"model": "Ridge regression (StandardScaler pipeline), fitted on the full sample",
            "alpha": float(reg.alpha), "intercept": float(intercept_raw),
            "PT": float(coef_raw[0]), "MoCA": float(coef_raw[1]), "Speed": float(coef_raw[2]),
            "n": int(len(y))}
    with open(os.path.join(OUT_DIR, "calculator_coefficients.json"), "w") as fh:
        json.dump(calc, fh, indent=2)
    print("Final ridge model (raw units):", json.dumps(calc))

    print("Sensitivity analysis: post-training gait speed ...")
    y_post = pd.to_numeric(df[POST_OUTCOME], errors="coerce").to_numpy(float)
    m_post = metrics(y_post, nested_loocv("Ridge regression", X, y_post))
    out["Sensitivity_post_speed"] = pd.DataFrame([{"Outcome": POST_OUTCOME, "Model": "Ridge regression (nested LOOCV)",
        **{k: round(v, 4) for k, v in m_post.items()}}])

    print("Figures 2 and 3 ...")
    m_app = figure2(y, final_pipe.predict(X), loo_pred["Ridge regression"])
    shap_imp = figure3(final_pipe, X)

    with pd.ExcelWriter(os.path.join(OUT_DIR, "prediction_model_results.xlsx")) as w:
        for k, v in out.items():
            v.to_excel(w, sheet_name=k[:31], index=False)

    # ----------------------------- SELF-CHECK -----------------------------
    chk = []
    def add(item, got, rep, d):
        chk.append({"item": item, "this run": round(got, d), "article": rep,
                    "match": "OK" if round(got, d) == round(rep, d) else "DIFFERENT"})
    T2 = {r["Model"]: r for r in t2}
    S4 = {r["Model"]: r for r in s4}
    for mdl, (r2, rmse, mae, cs) in REPORTED["LOOCV"].items():
        add(f"Table 2 {mdl} LOOCV R²", T2[mdl]["LOOCV R2"], r2, 3)
        add(f"Table 2 {mdl} LOOCV RMSE", T2[mdl]["LOOCV RMSE"], rmse, 3)
        add(f"Table 2 {mdl} LOOCV MAE", T2[mdl]["LOOCV MAE"], mae, 3)
        add(f"Table 2 {mdl} calibration slope", T2[mdl]["Calibration slope"], cs, 3)
        add(f"Table 2 {mdl} 5-fold median R²", T2[mdl]["_cv5_med_r2"], REPORTED["CV5_median_R2"][mdl], 3)
    for mdl, v in REPORTED["Apparent_R2"].items():
        add(f"Table S4 {mdl} apparent R²", S4[mdl]["Apparent R2"], v, 4)
    add("Table S4 random forest LOOCV R²", S4["Random forest"]["LOOCV R2"], REPORTED["RF_LOOCV_R2"], 4)
    add("Table S1 multivariable R²", multi.rsquared, REPORTED["S1_multivariable_R2"], 3)
    add("Table S1 final model R²", fin.rsquared, REPORTED["S1_final_R2"], 3)
    S2 = out["TableS2_fit"]
    for f, v in REPORTED["S2_p_change"].items():
        add(f"Table S2 F-change p (+{CANDIDATES[f]})",
            S2.loc[S2.Block == f"Block 2: + {CANDIDATES[f]}", "p_change"].iloc[0], v, 3)
    S3 = out["TableS3"].set_index("column")
    for c, v in REPORTED["S3_inclusion"].items():
        add(f"Table S3 inclusion % {CANDIDATES[c]}", S3.loc[c, "Inclusion frequency (%)"], v, 1)
    add("Sensitivity post-speed nested LOOCV R²", m_post["R2"], REPORTED["Sensitivity_post_speed"][0], 3)
    add("Sensitivity post-speed calibration slope", m_post["cal_slope"], REPORTED["Sensitivity_post_speed"][1], 3)
    add("Figure 2 apparent calibration slope", m_app["cal_slope"], REPORTED["Fig2_apparent_slope"], 3)
    for c, v in REPORTED["Fig3_mean_abs_shap"].items():
        add(f"Figure 3 mean |SHAP| {DISPLAY[c]}", shap_imp[c], v, 4)
    C = pd.DataFrame(chk)
    C.to_csv(os.path.join(OUT_DIR, "SELF_CHECK_vs_article.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== SELF-CHECK against values reported in the article ===")
        print(C.to_string(index=False))
    print(f"\n{(C.match == 'OK').sum()} / {len(C)} values match.  Results in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
