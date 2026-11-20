#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PD_WPD_run3_TLLee.py
# Part of: PD Gait Response Calculator (https://github.com/chadgodzilla/pd-gait-response-calculator)
# Archived: https://doi.org/10.5281/zenodo.24248978   License: MIT
# Wavelet packet decomposition band energies from pelvic IMU acceleration (Streamlit app).

import io
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import streamlit as st
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

st.set_page_config(layout="wide")
st.title("Pelvic IMU — WPD Energy Bands (Multi-Trial)")

# Hide the "Drag and drop / Limit 200MB" text inside file uploaders
st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] {display: none;}
[data-testid="stFileUploaderDropzone"] {padding: 0.4rem 0.6rem; min-height: unset;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Fixed band definitions
# =========================================================
LOC_LOW, LOC_HIGH = 0.5, 3.0
FREEZE_LOW, FREEZE_HIGH = 3.0, 8.0
IMPACT_LOW, IMPACT_HIGH = 8.0, 12.0

# =========================================================
# Helper functions (unchanged from original)
# =========================================================
def estimate_fs_from_time(time):
    time = np.asarray(time).astype(float)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size < 5:
        return np.nan
    return float(1.0 / np.median(dt))


def is_uniform_time(time, tol_cv=0.01):
    time = np.asarray(time).astype(float)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size < 5:
        return False
    cv = np.std(dt) / (np.mean(dt) + 1e-12)
    return cv < tol_cv


def butter_lowpass_filter(x, fs, cutoff_hz, order=4):
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        return x.copy()
    wn = cutoff_hz / (fs / 2.0)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def safe_resample_poly(time, signal, fs_in, fs_out):
    if fs_out <= 0 or fs_in <= 0:
        return time.copy(), signal.copy()
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(signal, up, down)
    duration = float(time[-1] - time[0])
    if duration <= 0:
        return time.copy(), signal.copy()
    n_new = len(y)
    t_new = np.arange(n_new, dtype=float) / fs_out + float(time[0])
    return t_new, y


def build_wpd(x, wavelet_name, maxlevel):
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet_name, mode="symmetric", maxlevel=maxlevel)
    nodes = wp.get_level(maxlevel, order="freq")
    return wp, nodes


def terminal_bandwidth(fs, level):
    return fs / (2 ** (level + 1))


def nodes_freq_table(fs, level, nodes):
    df_band = terminal_bandwidth(fs, level)
    rows = []
    for i, node in enumerate(nodes):
        f_low = i * df_band
        f_high = (i + 1) * df_band
        rows.append({"node_index": i, "node_path": node.path,
                     "f_low_hz": f_low, "f_high_hz": f_high})
    return pd.DataFrame(rows)


def pick_nodes_for_band(nodes_df, band_low, band_high, mode="contained"):
    if mode == "contained":
        mask = (nodes_df["f_low_hz"] >= band_low) & (nodes_df["f_high_hz"] <= band_high)
    else:
        mask = (nodes_df["f_high_hz"] > band_low) & (nodes_df["f_low_hz"] < band_high)
    return nodes_df.loc[mask].copy()


def energy(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def wpd_band_energy_from_nodes(nodes, selected_paths):
    e = 0.0
    for node in nodes:
        if node.path in selected_paths:
            e += energy(node.data)
    return e


def reconstruct_band_from_selected_nodes(signal, wavelet_name, level, selected_paths):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet_name, mode="symmetric", maxlevel=level)
    new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet_name, mode="symmetric", maxlevel=level)
    all_nodes = wp.get_level(level, order="freq")
    for node in all_nodes:
        if node.path in selected_paths:
            new_wp[node.path] = node.data
        else:
            new_wp[node.path] = np.zeros_like(node.data)
    rec = new_wp.reconstruct(update=False)
    rec = np.asarray(rec, dtype=float)
    if len(rec) > len(signal):
        rec = rec[:len(signal)]
    elif len(rec) < len(signal):
        rec = np.pad(rec, (0, len(signal) - len(rec)), mode="edge")
    return rec


def detect_step_peaks_auto(time_sec, signal, fs):
    x = np.asarray(signal, dtype=float)
    x = x - np.nanmedian(x)
    env = np.abs(x)
    min_distance = max(1, int(0.35 * fs))
    prom = 0.30 * np.nanstd(env)
    if not np.isfinite(prom) or prom <= 0:
        prom = None
    peaks, _ = find_peaks(env, distance=min_distance, prominence=prom)
    return peaks, env


def auto_trim_by_peak_order(time_sec, signal, fs, start_step=2, end_from_last_step=2):
    peaks, env = detect_step_peaks_auto(time_sec, signal, fs)
    n_peaks = len(peaks)
    if n_peaks == 0:
        return time_sec, signal, peaks, env, False, None, None
    start_idx_in_peaks = start_step - 1
    end_idx_in_peaks = n_peaks - end_from_last_step
    if start_idx_in_peaks < 0 or end_idx_in_peaks < 0:
        return time_sec, signal, peaks, env, False, None, None
    if start_idx_in_peaks >= n_peaks or end_idx_in_peaks >= n_peaks:
        return time_sec, signal, peaks, env, False, None, None
    start_idx = peaks[start_idx_in_peaks]
    end_idx = peaks[end_idx_in_peaks]
    if end_idx <= start_idx:
        return time_sec, signal, peaks, env, False, None, None
    t0 = time_sec[start_idx]
    t1 = time_sec[end_idx]
    mask = (time_sec >= t0) & (time_sec <= t1)
    return time_sec[mask], signal[mask], peaks, env, True, t0, t1


def plot_frequency_tiling(nodes_df, loc_nodes_df, freeze_nodes_df, impact_nodes_df, fs_used, title="Frequency tiling"):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    nyquist = fs_used / 2.0
    for _, row in nodes_df.iterrows():
        width = row["f_high_hz"] - row["f_low_hz"]
        ax.barh(y=0, width=width, left=row["f_low_hz"], height=0.45,
                align="center", edgecolor="black", linewidth=0.5, alpha=0.15, color="gray")

    def overlay_band(df_band, y, label, color):
        if len(df_band) == 0:
            return
        for _, row in df_band.iterrows():
            width = row["f_high_hz"] - row["f_low_hz"]
            ax.barh(y=y, width=width, left=row["f_low_hz"], height=0.42,
                    align="center", edgecolor="black", linewidth=0.7, color=color)
        xmid = 0.5 * (df_band["f_low_hz"].min() + df_band["f_high_hz"].max())
        ax.text(xmid, y + 0.28, label, ha="center", va="bottom", fontsize=9)

    overlay_band(loc_nodes_df, y=1.0, label="Locomotor", color="tab:blue")
    overlay_band(freeze_nodes_df, y=2.0, label="Freeze", color="tab:orange")
    overlay_band(impact_nodes_df, y=3.0, label="Impact", color="tab:green")
    ax.set_xlim(0, nyquist)
    ax.set_ylim(-0.5, 3.7)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["All terminal nodes", "Locomotor", "Freeze", "Impact"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return fig


# =========================================================
# Core processing function — one trial
# =========================================================
def process_trial(df, imu_choice, axis_choice, imu_name_map,
                  use_antialias, lp_cutoff_hz, lp_order,
                  do_resample, fs_target,
                  use_time_crop, crop_start_sec, crop_end_sec,
                  use_auto_trim, auto_start_step, auto_end_from_last_step,
                  wavelet, wpd_level, node_pick_mode, window_sec):
    """Process one trial and return result dict or (None, error_string)."""

    time_col_candidates = [c for c in df.columns if "Time" in c or "time" in c]
    if len(time_col_candidates) == 0:
        return None, "No Time column found"
    time_col = time_col_candidates[0]

    imu_key = imu_name_map[imu_choice]
    axis_key_fragment = f":{axis_choice}(g)"
    matched_cols = [c for c in df.columns if imu_key in c and axis_key_fragment in c]
    if len(matched_cols) == 0:
        return None, f"No column matching '{imu_key}' + '{axis_key_fragment}'"
    sig_col = matched_cols[0]

    time_raw = df[time_col].values.astype(float)
    signal_raw = df[sig_col].values.astype(float)

    mask = np.isfinite(time_raw) & np.isfinite(signal_raw)
    time_raw, signal_raw = time_raw[mask], signal_raw[mask]
    if len(signal_raw) < 20:
        return None, "Too few valid data points (<20)"

    sort_idx = np.argsort(time_raw)
    time_raw, signal_raw = time_raw[sort_idx], signal_raw[sort_idx]

    fs_infer = estimate_fs_from_time(time_raw)
    uniform_flag = is_uniform_time(time_raw)
    if not np.isfinite(fs_infer):
        return None, "Cannot estimate sample rate"

    signal_pre = signal_raw.copy()
    if use_antialias:
        signal_pre = butter_lowpass_filter(signal_pre, fs=fs_infer, cutoff_hz=lp_cutoff_hz, order=lp_order)

    if do_resample:
        time_proc, signal_proc = safe_resample_poly(time_raw, signal_pre, fs_in=fs_infer, fs_out=float(fs_target))
        fs_used = float(fs_target)
    else:
        time_proc, signal_proc = time_raw.copy(), signal_pre.copy()
        fs_used = float(fs_infer)

    time_rel = time_proc - time_proc[0]

    manual_crop_applied = False
    manual_crop_t0, manual_crop_t1 = None, None
    if use_time_crop:
        if crop_end_sec <= crop_start_sec:
            return None, "Crop end must be > crop start"
        abs_start = time_proc[0] + crop_start_sec
        abs_end = time_proc[0] + crop_end_sec
        mask_crop = (time_proc >= abs_start) & (time_proc <= abs_end)
        if np.sum(mask_crop) < 20:
            return None, "Too few points after manual crop"
        time_proc = time_proc[mask_crop]
        signal_proc = signal_proc[mask_crop]
        time_rel = time_proc - time_proc[0]
        manual_crop_applied = True
        manual_crop_t0, manual_crop_t1 = crop_start_sec, crop_end_sec

    trim_applied = False
    trim_t0, trim_t1 = None, None
    peaks = np.array([])
    env = None

    if use_auto_trim:
        time_trim, signal_trim, peaks, env, trim_applied, trim_t0, trim_t1 = auto_trim_by_peak_order(
            time_rel, signal_proc, fs_used,
            start_step=int(auto_start_step),
            end_from_last_step=int(auto_end_from_last_step)
        )
        if trim_applied:
            time_proc = time_trim + time_proc[0]
            signal_proc = signal_trim
            time_rel = time_proc - time_proc[0]

    N = len(signal_proc)

    # WPD
    wp, terminal_nodes = build_wpd(signal_proc, wavelet, wpd_level)
    nodes_df = nodes_freq_table(fs_used, wpd_level, terminal_nodes)
    loc_nodes_df = pick_nodes_for_band(nodes_df, LOC_LOW, LOC_HIGH, mode=node_pick_mode)
    freeze_nodes_df = pick_nodes_for_band(nodes_df, FREEZE_LOW, FREEZE_HIGH, mode=node_pick_mode)
    impact_nodes_df = pick_nodes_for_band(nodes_df, IMPACT_LOW, IMPACT_HIGH, mode=node_pick_mode)

    loc_paths = set(loc_nodes_df["node_path"].tolist())
    freeze_paths = set(freeze_nodes_df["node_path"].tolist())
    impact_paths = set(impact_nodes_df["node_path"].tolist())

    locomotor_energy = wpd_band_energy_from_nodes(terminal_nodes, loc_paths)
    freeze_energy = wpd_band_energy_from_nodes(terminal_nodes, freeze_paths)
    impact_energy = wpd_band_energy_from_nodes(terminal_nodes, impact_paths)
    total_wpd_energy = sum(energy(node.data) for node in terminal_nodes)
    freeze_index = freeze_energy / locomotor_energy if locomotor_energy > 0 else np.nan

    locomotor_ratio = locomotor_energy / total_wpd_energy if total_wpd_energy > 0 else np.nan
    freeze_ratio    = freeze_energy    / total_wpd_energy if total_wpd_energy > 0 else np.nan

    summary_bands_df = pd.DataFrame([{
        "locomotor_energy": locomotor_energy,
        "locomotor_ratio":  locomotor_ratio,
        "freeze_energy":    freeze_energy,
        "freeze_ratio":     freeze_ratio,
        "impact_energy":    impact_energy,
        "total_wpd_energy": total_wpd_energy,
        "freeze_index":     freeze_index,
    }])

    # Window-level
    samples_per_window = int(round(window_sec * fs_used))
    num_windows = N // samples_per_window if samples_per_window > 8 else 0

    window_rows = []
    node_rows = []
    for w in range(num_windows):
        start = w * samples_per_window
        end = start + samples_per_window
        xw = signal_proc[start:end]
        _, nodes_w = build_wpd(xw, wavelet, wpd_level)
        nodes_w_df = nodes_freq_table(fs_used, wpd_level, nodes_w)
        loc_e = wpd_band_energy_from_nodes(nodes_w, loc_paths)
        frz_e = wpd_band_energy_from_nodes(nodes_w, freeze_paths)
        imp_e = wpd_band_energy_from_nodes(nodes_w, impact_paths)
        total_e = sum(energy(node.data) for node in nodes_w)
        fi = frz_e / loc_e if loc_e > 0 else np.nan
        window_rows.append({
            "imu": imu_choice, "axis": axis_choice, "window": w,
            "t_start_sec": start / fs_used, "t_end_sec": end / fs_used,
            "locomotor_energy": loc_e, "freeze_energy": frz_e,
            "impact_energy": imp_e, "total_wpd_energy": total_e, "freeze_index": fi,
        })
        for i, node in enumerate(nodes_w):
            node_rows.append({
                "window": w, "node_index": i, "node_path": node.path,
                "f_low_hz": nodes_w_df.iloc[i]["f_low_hz"],
                "f_high_hz": nodes_w_df.iloc[i]["f_high_hz"],
                "energy": energy(node.data),
            })

    return {
        "sig_col": sig_col, "time_col": time_col,
        "fs_infer": fs_infer, "fs_used": fs_used, "uniform_flag": uniform_flag,
        "N": N,
        "time_raw": time_raw, "signal_raw": signal_raw, "signal_pre": signal_pre,
        "time_proc": time_proc, "signal_proc": signal_proc, "time_rel": time_rel,
        "manual_crop_applied": manual_crop_applied,
        "manual_crop_t0": manual_crop_t0, "manual_crop_t1": manual_crop_t1,
        "trim_applied": trim_applied, "trim_t0": trim_t0, "trim_t1": trim_t1,
        "peaks": peaks, "env": env,
        "nodes_df": nodes_df,
        "loc_nodes_df": loc_nodes_df, "freeze_nodes_df": freeze_nodes_df, "impact_nodes_df": impact_nodes_df,
        "loc_paths": loc_paths, "freeze_paths": freeze_paths, "impact_paths": impact_paths,
        "terminal_nodes": terminal_nodes,
        "summary_bands_df": summary_bands_df,
        "locomotor_energy": locomotor_energy, "locomotor_ratio": locomotor_ratio,
        "freeze_energy": freeze_energy, "freeze_ratio": freeze_ratio,
        "impact_energy": impact_energy, "total_wpd_energy": total_wpd_energy,
        "freeze_index": freeze_index,
        "window_df": pd.DataFrame(window_rows),
        "node_energy_df": pd.DataFrame(node_rows),
        "samples_per_window": samples_per_window, "num_windows": num_windows,
    }, None


# =========================================================
# Render full detailed analysis for one trial
# =========================================================
def render_trial_results(r, trial_label, imu_choice, axis_choice,
                         wavelet, wpd_level, window_sec,
                         use_antialias, lp_cutoff_hz, fs_target,
                         use_auto_trim, use_time_crop,
                         crop_start_sec, crop_end_sec,
                         auto_start_step, auto_end_from_last_step,
                         do_resample):

    st.success(f"Signal: {r['sig_col']}")
    st.info(
        f"Estimated fs: {r['fs_infer']:.2f} Hz | "
        f"Time axis: {'uniform' if r['uniform_flag'] else 'non-uniform/jittered'}"
    )

    if r["manual_crop_applied"]:
        st.write(f"✅ Manual crop: {r['manual_crop_t0']:.3f}–{r['manual_crop_t1']:.3f} sec")

    if use_auto_trim:
        if r["trim_applied"]:
            st.write(
                f"✅ Auto trim: step {int(auto_start_step)} ({r['trim_t0']:.3f} s) "
                f"→ last -{int(auto_end_from_last_step)} ({r['trim_t1']:.3f} s)"
            )
        else:
            st.warning("⚠️ Auto trim failed — using full segment")

    st.write(f"✅ Analysis fs = {r['fs_used']:.2f} Hz, N = {r['N']}")

    # --- Signals ---
    st.subheader("Source / Processed Signal")
    n_subplots = 3 if use_auto_trim else 2
    fig0, axes0 = plt.subplots(n_subplots, 1, figsize=(10, 7 if use_auto_trim else 5), sharex=False)
    if use_auto_trim:
        ax_raw, ax_proc, ax_trim = axes0
    else:
        ax_raw, ax_proc = axes0

    ax_raw.plot(r["time_raw"] - r["time_raw"][0], r["signal_raw"])
    ax_raw.set_title(f"Raw signal ({r['fs_infer']:.2f} Hz)")
    ax_raw.set_xlabel("Time (s)")
    ax_raw.set_ylabel("Acceleration (g)")

    if do_resample:
        time_proc_show, signal_proc_show = safe_resample_poly(
            r["time_raw"], r["signal_pre"], fs_in=r["fs_infer"], fs_out=float(fs_target))
    else:
        time_proc_show = r["time_raw"].copy()
        signal_proc_show = r["signal_pre"].copy()

    if use_time_crop:
        abs_start_show = time_proc_show[0] + crop_start_sec
        abs_end_show = time_proc_show[0] + crop_end_sec
        mask_show = (time_proc_show >= abs_start_show) & (time_proc_show <= abs_end_show)
        if np.sum(mask_show) >= 20:
            time_proc_show = time_proc_show[mask_show]
            signal_proc_show = signal_proc_show[mask_show]

    time_rel_show = time_proc_show - time_proc_show[0]
    ax_proc.plot(time_rel_show, signal_proc_show)
    proc_title = (f"Processed (LPF={lp_cutoff_hz:.1f} Hz, fs={r['fs_used']:.1f} Hz)"
                  if use_antialias else f"Processed (fs={r['fs_used']:.1f} Hz)")
    ax_proc.set_title(proc_title)
    ax_proc.set_xlabel("Time (s)")
    ax_proc.set_ylabel("Acceleration (g)")

    if use_auto_trim:
        x0 = signal_proc_show - np.nanmedian(signal_proc_show)
        env_show = np.abs(x0)
        prom_show = 0.30 * np.nanstd(env_show)
        if not np.isfinite(prom_show) or prom_show <= 0:
            prom_show = None
        peaks_show, _ = find_peaks(env_show, distance=max(1, int(0.35 * r["fs_used"])), prominence=prom_show)
        ax_trim.plot(time_rel_show, signal_proc_show, label="processed")
        if len(peaks_show) > 0:
            ax_trim.plot(time_rel_show[peaks_show], signal_proc_show[peaks_show],
                         "ro", markersize=4, label="detected peaks")
        if r["trim_applied"]:
            ax_trim.axvline(r["trim_t0"], color="green", linestyle="--", label="start trim")
            ax_trim.axvline(r["trim_t1"], color="purple", linestyle="--", label="end trim")
        ax_trim.set_title("Auto-trim peak detection")
        ax_trim.set_xlabel("Time (s)")
        ax_trim.set_ylabel("Acceleration (g)")
        ax_trim.legend(loc="best")

    st.pyplot(fig0)
    plt.close(fig0)

    # --- Full-signal WPD band results ---
    st.subheader("Full-signal WPD Band Results")
    st.dataframe(r["summary_bands_df"], use_container_width=True)

    # --- WPD terminal nodes ---
    st.subheader("WPD Terminal Nodes")
    st.dataframe(r["nodes_df"], use_container_width=True)
    st.markdown("#### Band node groups")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Locomotor ({LOC_LOW:.1f}–{LOC_HIGH:.1f} Hz)**")
        st.dataframe(r["loc_nodes_df"][["node_index", "node_path", "f_low_hz", "f_high_hz"]], use_container_width=True)
    with c2:
        st.write(f"**Freeze ({FREEZE_LOW:.1f}–{FREEZE_HIGH:.1f} Hz)**")
        st.dataframe(r["freeze_nodes_df"][["node_index", "node_path", "f_low_hz", "f_high_hz"]], use_container_width=True)
    with c3:
        st.write(f"**Impact ({IMPACT_LOW:.1f}–{IMPACT_HIGH:.1f} Hz)**")
        st.dataframe(r["impact_nodes_df"][["node_index", "node_path", "f_low_hz", "f_high_hz"]], use_container_width=True)

    # --- Frequency tiling ---
    st.subheader("Frequency Tiling")
    fig_t = plot_frequency_tiling(
        r["nodes_df"], r["loc_nodes_df"], r["freeze_nodes_df"], r["impact_nodes_df"],
        r["fs_used"],
        title=f"Frequency tiling (fs={r['fs_used']:.1f} Hz, WPD level={wpd_level})"
    )
    st.pyplot(fig_t)
    plt.close(fig_t)

    # --- Node energy bar ---
    st.subheader("Terminal Node Energy (Full Signal)")
    node_energy_full_rows = []
    for i, node in enumerate(r["terminal_nodes"]):
        node_energy_full_rows.append({
            "node_index": i, "node_path": node.path,
            "f_low_hz": r["nodes_df"].iloc[i]["f_low_hz"],
            "f_high_hz": r["nodes_df"].iloc[i]["f_high_hz"],
            "energy": energy(node.data),
            "band_group": (
                "Locomotor" if node.path in r["loc_paths"] else
                "Freeze" if node.path in r["freeze_paths"] else
                "Impact" if node.path in r["impact_paths"] else "Other"
            )
        })
    node_energy_full_df = pd.DataFrame(node_energy_full_rows)
    st.dataframe(node_energy_full_df, use_container_width=True)

    fig_n, ax_n = plt.subplots(figsize=(12, 4))
    x_labels = [f"{row['node_index']}\n{row['f_low_hz']:.1f}-{row['f_high_hz']:.1f}"
                for _, row in node_energy_full_df.iterrows()]
    y_vals = node_energy_full_df["energy"].values
    colors = []
    for group in node_energy_full_df["band_group"]:
        colors.append({"Locomotor": "tab:blue", "Freeze": "tab:orange",
                       "Impact": "tab:green"}.get(group, "lightgray"))
    ax_n.bar(range(len(y_vals)), y_vals, color=colors)
    ax_n.set_xticks(range(len(x_labels)))
    ax_n.set_xticklabels(x_labels, rotation=90)
    ax_n.set_xlabel("Node index / Frequency band (Hz)")
    ax_n.set_ylabel("Energy")
    ax_n.set_title("Terminal node energy (full signal)")
    ax_n.legend(handles=[
        Patch(facecolor="tab:blue", label="Locomotor"),
        Patch(facecolor="tab:orange", label="Freeze"),
        Patch(facecolor="tab:green", label="Impact"),
        Patch(facecolor="lightgray", label="Other"),
    ], loc="upper right")
    st.pyplot(fig_n)
    plt.close(fig_n)

    # --- Window-level ---
    st.subheader("Window-level Band Energies")
    if r["samples_per_window"] <= 8:
        st.error("Window too short — increase window_sec or fs_target.")
    else:
        st.write(f"N={r['N']} samples @ {r['fs_used']:.2f} Hz → "
                 f"{r['num_windows']} windows × {window_sec} s each")
        st.dataframe(r["window_df"], use_container_width=True)

        fig2, ax2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        ax2[0].plot(r["window_df"]["window"], r["window_df"]["locomotor_energy"], marker="o")
        ax2[0].set_ylabel("Energy")
        ax2[0].set_title("Locomotor energy")
        ax2[1].plot(r["window_df"]["window"], r["window_df"]["freeze_energy"], marker="o")
        ax2[1].set_ylabel("Energy")
        ax2[1].set_title("Freeze energy")
        ax2[2].plot(r["window_df"]["window"], r["window_df"]["impact_energy"], marker="o")
        ax2[2].set_ylabel("Energy")
        ax2[2].set_title("Impact energy")
        ax2[3].plot(r["window_df"]["window"], r["window_df"]["freeze_index"], marker="o")
        ax2[3].set_ylabel("FI")
        ax2[3].set_xlabel("Window index")
        ax2[3].set_title("Freeze Index")
        st.pyplot(fig2)
        plt.close(fig2)


# =========================================================
# Sidebar — file upload (top) then parameters
# =========================================================
st.sidebar.header("Upload Trial Files")
st.sidebar.caption("Select up to 3 CSV files (one per trial)")

uploaded_files = []
for i in range(1, 4):
    f = st.sidebar.file_uploader(f"Trial {i} — select CSV", type=["csv"], key=f"trial_{i}")
    uploaded_files.append(f)

active_files = [(i + 1, f) for i, f in enumerate(uploaded_files) if f is not None]

st.sidebar.markdown("---")
st.sidebar.header("Parameters")

imu_name_map = {
    "Right Foot": "R.Right.Foot_ImuAcc",
    "Left Foot": "L.Left.Foot_ImuAcc",
    "Right Shank": "R.Right.Shank_ImuAcc",
    "Left Shank": "L.Left.Shank_ImuAcc",
    "Right Thigh": "R.Right.Thigh_ImuAcc",
    "Left Thigh": "L.Left.Thigh_ImuAcc",
    "Pelvis": "Pelvis_ImuAcc",
}

st.sidebar.subheader("Signal Selection")
imu_choice = st.sidebar.selectbox("IMU segment", list(imu_name_map.keys()), index=6)
axis_choice = st.sidebar.selectbox("Axis", ["X", "Y", "Z"], index=2)

st.sidebar.markdown("---")
window_sec = st.sidebar.selectbox("Window length (sec)", [1.0, 0.5, 0.2, 0.1], index=0)
wavelet = st.sidebar.selectbox("Wavelet", ["db4", "sym5", "coif5"], index=0)
wpd_level = st.sidebar.slider("WPD level", min_value=3, max_value=8, value=6)

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing")
use_antialias = st.sidebar.checkbox("Low-pass filter first", value=True)
lp_cutoff_hz = st.sidebar.slider("LPF cutoff (Hz)", min_value=5.0, max_value=100.0, value=30.0, step=1.0)
lp_order = st.sidebar.slider("Butterworth order", min_value=2, max_value=8, value=4)
do_resample = st.sidebar.checkbox("Resample / decimate", value=True)
fs_target = st.sidebar.number_input("Target fs (Hz)", value=64.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Time Segment")
use_time_crop = st.sidebar.checkbox("Analyse specified time range only", value=False)
crop_start_sec = st.sidebar.number_input("Start time (sec)", value=2.0, step=0.1)
crop_end_sec = st.sidebar.number_input("End time (sec)", value=7.0, step=0.1)

st.sidebar.markdown("---")
use_auto_trim = st.sidebar.checkbox("Auto-detect gait segment", value=True)
auto_start_step = st.sidebar.number_input("Start at step #", min_value=1, value=2, step=1)
auto_end_from_last_step = st.sidebar.number_input("End at last step -#", min_value=1, value=2, step=1)

st.sidebar.markdown("---")
node_pick_mode = st.sidebar.selectbox(
    "Node selection mode", ["contained", "overlap"], index=0,
    help="contained = node fully within band; overlap = any overlap included",
)

# =========================================================
# Guard: need at least one file
# =========================================================
if len(active_files) == 0:
    st.info("👈 Upload at least one CSV file in the sidebar to begin analysis.")
    st.stop()

# =========================================================
# Process each uploaded trial
# =========================================================
all_results = {}
all_errors  = {}

for trial_num, uploaded in active_files:
    df = pd.read_csv(uploaded)
    result, err = process_trial(
        df, imu_choice, axis_choice, imu_name_map,
        use_antialias, lp_cutoff_hz, lp_order,
        do_resample, fs_target,
        use_time_crop, crop_start_sec, crop_end_sec,
        use_auto_trim, auto_start_step, auto_end_from_last_step,
        wavelet, wpd_level, node_pick_mode, window_sec
    )
    if err:
        all_errors[trial_num] = err
    else:
        all_results[trial_num] = result

if len(all_results) == 0:
    for trial_num, err in all_errors.items():
        fname = [f.name for t, f in active_files if t == trial_num][0]
        st.error(f"Trial {trial_num} ({fname}): {err}")
    st.stop()

NUMERIC_COLS = ["locomotor_energy", "locomotor_ratio", "freeze_energy",
                "freeze_ratio", "impact_energy", "total_wpd_energy", "freeze_index"]

# =========================================================
# SECTION 1 — Average table (top of main area)
# =========================================================
st.markdown("---")
st.header("📊 Average of All Trials")
st.caption("Use these averaged values as ML model input features")

avg_df = None
if len(all_results) >= 2:
    avg_data = {}
    for col in NUMERIC_COLS:
        vals = [all_results[tn][col] for tn in all_results if np.isfinite(all_results[tn][col])]
        avg_data[col] = np.mean(vals) if vals else np.nan
    avg_df = pd.DataFrame([avg_data], index=["Average"])
    st.dataframe(
        avg_df.style
              .format("{:.6f}")
              .set_properties(**{"background-color": "#ddeeff", "font-weight": "bold"}),
        use_container_width=True,
    )
    avg_csv = avg_df.to_csv()
    st.download_button(
        "⬇️ Download average CSV", avg_csv,
        file_name="wpd_average_features.csv", mime="text/csv",
    )
else:
    st.info("Upload 2 or 3 trials to see the averaged feature table.")

# =========================================================
# SECTION 2 — Per-trial summary tables (side by side)
# =========================================================
st.markdown("---")
st.header("Trial Summary Tables")

trial_nums = sorted(all_results.keys())
summary_cols = st.columns(len(trial_nums))

for col_widget, trial_num in zip(summary_cols, trial_nums):
    r = all_results[trial_num]
    fname = [f.name for t, f in active_files if t == trial_num][0]
    with col_widget:
        st.markdown(f"**Trial {trial_num}**")
        st.caption(fname)
        disp_df = r["summary_bands_df"].copy()
        disp_df.index = [f"Trial {trial_num}"]
        st.dataframe(disp_df.style.format("{:.6f}"), use_container_width=True)

for trial_num, err in all_errors.items():
    fname = [f.name for t, f in active_files if t == trial_num][0]
    st.error(f"Trial {trial_num} ({fname}): {err}")

# =========================================================
# SECTION 3 — Band-Reconstructed Signals for all trials
# =========================================================
st.markdown("---")
st.header("Band-Reconstructed Signals")

for trial_num in trial_nums:
    r = all_results[trial_num]
    fname = [f.name for t, f in active_files if t == trial_num][0]

    st.subheader(f"Trial {trial_num} — {fname}")

    loc_rec    = reconstruct_band_from_selected_nodes(r["signal_proc"], wavelet, wpd_level, r["loc_paths"])
    freeze_rec = reconstruct_band_from_selected_nodes(r["signal_proc"], wavelet, wpd_level, r["freeze_paths"])
    impact_rec = reconstruct_band_from_selected_nodes(r["signal_proc"], wavelet, wpd_level, r["impact_paths"])

    fig_rec, ax_rec = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax_rec[0].plot(r["time_rel"], r["signal_proc"])
    ax_rec[0].set_title("Processed signal")
    ax_rec[0].set_ylabel("g")
    ax_rec[1].plot(r["time_rel"], loc_rec, color="tab:blue")
    ax_rec[1].set_title(f"Locomotor ({LOC_LOW:.1f}–{LOC_HIGH:.1f} Hz)")
    ax_rec[1].set_ylabel("g")
    ax_rec[2].plot(r["time_rel"], freeze_rec, color="tab:orange")
    ax_rec[2].set_title(f"Freeze ({FREEZE_LOW:.1f}–{FREEZE_HIGH:.1f} Hz)")
    ax_rec[2].set_ylabel("g")
    ax_rec[3].plot(r["time_rel"], impact_rec, color="tab:green")
    ax_rec[3].set_title(f"Impact ({IMPACT_LOW:.1f}–{IMPACT_HIGH:.1f} Hz)")
    ax_rec[3].set_ylabel("g")
    ax_rec[3].set_xlabel("Time (s)")
    fig_rec.tight_layout()
    st.pyplot(fig_rec)
    plt.close(fig_rec)

# =========================================================
# SECTION 4 — Full detailed analysis per trial (tabbed)
# =========================================================
st.markdown("---")
st.header("Detailed Analysis per Trial")

tab_labels = [
    f"Trial {tn} — {[f.name for t, f in active_files if t == tn][0]}"
    for tn in trial_nums
]
tabs = st.tabs(tab_labels) if len(trial_nums) > 1 else [st.container()]

for tab, trial_num in zip(tabs, trial_nums):
    r = all_results[trial_num]
    fname = [f.name for t, f in active_files if t == trial_num][0]

    with tab:
        render_trial_results(
            r, f"Trial {trial_num}",
            imu_choice, axis_choice,
            wavelet, wpd_level, window_sec,
            use_antialias, lp_cutoff_hz, fs_target,
            use_auto_trim, use_time_crop,
            crop_start_sec, crop_end_sec,
            auto_start_step, auto_end_from_last_step,
            do_resample
        )

        # Per-trial Excel export
        st.markdown("---")
        st.subheader("Export This Trial")
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            r["nodes_df"].to_excel(writer, index=False, sheet_name="terminal_nodes")
            r["loc_nodes_df"].to_excel(writer, index=False, sheet_name="locomotor_nodes")
            r["freeze_nodes_df"].to_excel(writer, index=False, sheet_name="freeze_nodes")
            r["impact_nodes_df"].to_excel(writer, index=False, sheet_name="impact_nodes")
            r["summary_bands_df"].to_excel(writer, index=False, sheet_name="full_signal_summary")
            r["window_df"].to_excel(writer, index=False, sheet_name="window_band_energy")
            r["node_energy_df"].to_excel(writer, index=False, sheet_name="window_node_energy")
        st.download_button(
            label=f"⬇️ Download Trial {trial_num} Excel",
            data=out.getvalue(),
            file_name=f"wpd_trial{trial_num}_{fname.replace('.csv', '')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_excel_{trial_num}",
        )

# =========================================================
# SECTION 5 — Combined Excel export
# =========================================================
st.markdown("---")
st.subheader("Combined Export — All Trials")

comb_out = io.BytesIO()
with pd.ExcelWriter(comb_out, engine="xlsxwriter") as writer:
    if avg_df is not None:
        avg_df.to_excel(writer, sheet_name="AVERAGE_features")

    summary_rows = []
    for tn in trial_nums:
        r = all_results[tn]
        fname = [f.name for t, f in active_files if t == tn][0]
        row = r["summary_bands_df"].iloc[0].to_dict()
        row["trial"] = f"Trial {tn}"
        row["filename"] = fname
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="all_trials_summary")

    for tn in trial_nums:
        r = all_results[tn]
        r["window_df"].to_excel(writer, index=False, sheet_name=f"trial{tn}_windows")
        r["node_energy_df"].to_excel(writer, index=False, sheet_name=f"trial{tn}_node_energy")

st.download_button(
    label="⬇️ Download Combined Excel (All Trials + Average)",
    data=comb_out.getvalue(),
    file_name="wpd_all_trials_combined.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_combined",
)