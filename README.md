# PD Gait Response Calculator

Analysis code and an interactive web calculator accompanying the article:

> Lee TL, Chien CY, Lin TY, Lin CY, Yang JF, Lin CC. *Predicting Gait Rehabilitation Response in Parkinson's Disease Using Clinical and Wearable-Derived Gait Measures: A Preliminary Study.* [Journal, year, DOI — to be added on publication]

**Web calculator:** https://chadgodzilla.github.io/pd-gait-response-calculator/
[![DOI](https://zenodo.org/badge/1441997874.svg)](https://doi.org/10.5281/zenodo.24248978)

**Archived release (Zenodo):** https://doi.org/10.5281/zenodo.24248978

> ⚠️ **Research use only.** This is an exploratory prototype from a preliminary single-centre study (n = 44, National Cheng Kung University Hospital, Taiwan) without external validation. It is **not** a medical device and must not be used for individual clinical decisions or to replace clinical judgement. The difference between physical-therapy-supervised and self-directed training reflects an observed association, not a causal treatment effect (patients were not randomised).

---

## Purpose

The repository makes the gait-feature extraction and prediction-modeling analyses of the article reproducible:

1. **Feature extraction** from a pelvic inertial measurement unit (IMU) during repeated 7-m walks (64 Hz):
   traditional spatiotemporal gait parameters and wavelet packet decomposition (WPD) band energies
   (locomotor 0.5–3 Hz, freeze 3–8 Hz, impact 8–12 Hz).
2. **Model development and internal validation:** univariable screening, collinearity check (VIF), bootstrap-LASSO
   stability selection, and comparison of linear, ridge, LASSO and elastic-net regression by nested leave-one-out
   (LOOCV) and 5-fold cross-validation.
3. **Explainability:** SHAP analysis of the final ridge model.
4. **Web calculator** (`index.html`) implementing the final model.

## Repository contents

| Path | Description |
|---|---|
| `index.html` | Stand-alone web calculator (no external dependencies; served via GitHub Pages) |
| `feature_extraction/IMU_PDgait_parameters_v1.py` | Streamlit app: spatiotemporal gait parameters from Cometa EmgTools IMU exports — gait events from event labels, foot-switch voltage, or foot gyroscope (Salarian 2004); stride length by the Zijlstra & Hof (2003) pendulum model with pelvic-tilt correction; speed, cadence, stride-time CoV, asymmetry, trunk RMS, double support |
| `feature_extraction/PD_WPD_run3_TLLee.py` | Streamlit app: wavelet packet decomposition of pelvic acceleration; locomotor (0.5–3 Hz), freeze (3–8 Hz) and impact (8–12 Hz) band energies, ratios and freeze index |
| `modeling/PD_gait_prediction_model.py` | Prediction-model analysis: Table 1; predictor screening (Table S1); incremental value of WPD features (Table S2); bootstrap LASSO stability (Table S3); nested LOOCV and nested repeated 5-fold CV of linear, ridge, LASSO and elastic-net regression (Table 2); apparent performance and random-forest comparator (Table S4); observed-vs-predicted calibration plot (Figure 2); SHAP analysis (Figure 3); sensitivity analysis modeling post-training gait speed; final ridge coefficients for the calculator |
| `data/data_dictionary.md` | Variable definitions and units |
| `data/example_synthetic.xlsx`, `.csv` | Simulated data with the study's column structure (not real patients) |
| `requirements.txt` | Python dependencies |

## Analysis settings used in the article

**Gait parameters** (`IMU_PDgait_parameters_v1.py`): leg length (L4/L5 posterior surface to floor) entered per participant; up to three trials per session, averaged.

**WPD features** (`PD_WPD_run3_TLLee.py`, default settings): pelvic IMU, Z-axis acceleration; 4th-order Butterworth low-pass filter at 30 Hz; resampled to 64 Hz; gait segment auto-trimmed from the 2nd detected step to the 2nd-from-last step; Daubechies-4 (`db4`) wavelet, decomposition level 6 (0.5-Hz terminal bands); only nodes fully contained within each band are used; values averaged across trials.

## Final model and coefficient source

The calculator implements a **ridge regression** model predicting the change in comfortable gait speed (Δ speed, post − pre, m/s) from three predictors. The penalty (α = 5) was selected by shuffled 5-fold cross-validation (seed 42) from {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100} and the model was fitted on the full cohort (n = 44) within a standardisation (StandardScaler) pipeline. The coefficients below are back-transformed to raw units:

| Term | Coefficient |
|---|---|
| Intercept | 0.18942 |
| Physical therapy–supervised training (1 = yes, 0 = self-directed) | +0.07834 |
| MoCA score (per point) | +0.012145 |
| Baseline gait speed (per m/s) | −0.42965 |

**Predicted Δ speed (m/s) = 0.18942 + 0.07834 × PT + 0.012145 × MoCA − 0.42965 × baseline speed**

Internal validation (nested LOOCV, tuning repeated inside each training fold): R² = 0.378, RMSE = 0.137 m/s, calibration slope 1.046. Predictions are classified using PD-specific minimal clinically important difference thresholds for gait speed (Hass et al., *J Neurol Phys Ther* 2014;38:233–238): small 0.06, moderate 0.14 and large 0.22 m/s. Because the LOOCV RMSE exceeds the inter-tier interval (0.08 m/s), tier assignments carry substantial individual-level uncertainty.

Valid input ranges are those observed in the cohort: MoCA 5–30; baseline gait speed 0.34–1.33 m/s. Predictions outside these ranges are extrapolations.

## How to run

**Web calculator:** open `index.html` in any modern browser, or use the hosted version above.

**Analysis code:**

```bash
pip install -r requirements.txt

# feature extraction (interactive; upload Cometa EmgTools CSV exports)
streamlit run feature_extraction/IMU_PDgait_parameters_v1.py
streamlit run feature_extraction/PD_WPD_run3_TLLee.py

# prediction model: Tables 1, 2, S1–S4 and Figures 2–3 (set DATA_FILE at the top of the script)
python modeling/PD_gait_prediction_model.py
```

The analysis scripts read one participant-level spreadsheet described in `data/data_dictionary.md`. To check that the code runs, point `DATA_FILE` to `data/example_synthetic.xlsx`; results from simulated data will not reproduce the published estimates.

## Data availability

Individual participant data are not publicly available because of patient privacy and the conditions of ethics approval. De-identified data may be made available from the corresponding author on reasonable request, subject to approval by the Institutional Review Board of National Cheng Kung University Hospital.

## How to cite

If you use this code or the calculator, please cite **both** the article and the archived software release:

- **Article:** [full citation — to be added on publication]
- **Software:** Lee TL, Chien CY, Lin TY, Lin CY, Yang JF, Lin CC. *PD Gait Response Calculator*. Zenodo. https://doi.org/10.5281/zenodo.24248978

Citation metadata are also provided in `CITATION.cff` (GitHub shows a "Cite this repository" button).

## Version history

- **v1.1.0** — Full analysis code (feature extraction, prediction modeling, internal validation, SHAP and sensitivity analysis), data dictionary, synthetic example data and citation metadata. Calculator updated to the final ridge model reported in the article (α = 5).
- **v1.0.0** — Initial release of the web calculator.

## License

Code is released under the [MIT License](LICENSE).

## Contact

Tsung-Lin Lee, MD — Department of Neurology, National Cheng Kung University Hospital, Tainan, Taiwan — p88124069@gs.ncku.edu.tw
