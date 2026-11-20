# IMU_PDgait_parameters_v1.py
# Part of: PD Gait Response Calculator (https://github.com/chadgodzilla/pd-gait-response-calculator)
# Archived: https://doi.org/10.5281/zenodo.24248978   License: MIT
# Spatiotemporal gait parameters from Cometa EmgTools IMU exports (Streamlit app).
"""
Gait Analysis UI  —  Parkinson's Disease Research
Run with:  streamlit run IMU_PDgait_parameters_v1.py

Supports two Cometa EmgTools export formats automatically:
  Format A  — has 'Right Events:' / 'Left Events:' columns  (On/Off labels)
  Format B  — has 'Fsw_A(V):' / 'Fsw_B(V):' columns only   (voltage signal)
  Format C  — no foot-switch at all; events detected from foot IMU gyroscope
              (Salarian 2004 algorithm using sagittal rotation peaks)
"""

import re
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import medfilt, find_peaks
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PD Gait Analysis",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #f5f6f8; color: #1a1d23; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e3e8; }
[data-testid="stSidebar"] * { color: #2c3140 !important; }
[data-testid="stSidebar"] label { color: #4a5568 !important; font-size: 0.82rem !important; }
.main-title  { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:600; color:#1a56a8; letter-spacing:-0.02em; margin-bottom:0.1rem; }
.main-sub    { font-size:0.78rem; color:#8a95a8; font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:1.5rem; }
.trial-header{ font-family:'IBM Plex Mono',monospace; font-size:0.9rem; font-weight:600; color:#1a56a8; background:#eaf1fb; border-left:3px solid #1a56a8; padding:0.5rem 0.8rem; margin-bottom:0.8rem; border-radius:0 4px 4px 0; }
.ml-header   { font-family:'IBM Plex Mono',monospace; font-size:0.9rem; font-weight:600; color:#1a7a4a; background:#e8f7ef; border-left:3px solid #1a7a4a; padding:0.5rem 0.8rem; margin-bottom:0.8rem; border-radius:0 4px 4px 0; }
.fmt-badge   { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; font-weight:600; padding:2px 7px; border-radius:4px; margin-left:8px; vertical-align:middle; }
.fmt-a       { background:#e4f5ec; color:#1a7a4a; border:1px solid #a8dfc0; }
.fmt-b       { background:#e4eef9; color:#1a56a8; border:1px solid #a8c8ee; }
.fmt-c       { background:#fef3e4; color:#a86e1a; border:1px solid #eecf94; }
.metric-row  { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1rem; }
.chip        { background:#ffffff; border:1px solid #dde2ea; border-radius:6px; padding:6px 12px; font-family:'IBM Plex Mono',monospace; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
.chip-label  { color:#8a95a8; display:block; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.06em; }
.chip-value  { color:#1a1d23; font-weight:600; font-size:0.9rem; }
.chip-unit   { color:#a0a8b8; font-size:0.65rem; margin-left:2px; }
.divider     { height:1px; background:linear-gradient(90deg,#dde2ea 0%,transparent 100%); margin:1.5rem 0; }
.info-tag    { font-size:0.7rem; color:#8a95a8; font-family:'IBM Plex Mono',monospace; margin-top:0.3rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  COLUMN DETECTION  (robust fuzzy matching for both export formats)
# ──────────────────────────────────────────────────────────────────────────────

COL_PATTERNS = {
    "time"        : [r"^time\b",            r"time\s*\(s\)"],
    "right_events": [r"right.*event",        r"event.*right"],
    "left_events" : [r"left.*event",         r"event.*left"],
    "fsw_right"   : [r"fsw_a",              r"fsw.*right",   r"right.*fsw"],
    "fsw_left"    : [r"fsw_b",              r"fsw.*left",    r"left.*fsw"],
    "right_hip"   : [r"right.*hip.*(flex|extenti|extensi)"],
    "left_hip"    : [r"left.*hip.*(flex|extenti|extensi)"],
    "pelvis_tilt" : [r"pelvis.*tilt",        r"tilt.*pelvis"],
    "pelvis_acc_x": [r"pelvis.*acc.*:?\s*x", r"pelvis.*imuacc.*x"],
    "pelvis_acc_y": [r"pelvis.*acc.*:?\s*y", r"pelvis.*imuacc.*y"],
    "pelvis_acc_z": [r"pelvis.*acc.*:?\s*z", r"pelvis.*imuacc.*z"],
    # Foot IMU gyroscope — used for Format C (no footswitch)
    "r_foot_gyro_y": [r"r\.right\.foot_imugyro\s*:y", r"right.*foot.*gyro.*:y"],
    "r_foot_gyro_x": [r"r\.right\.foot_imugyro\s*:x", r"right.*foot.*gyro.*:x"],
    "r_foot_gyro_z": [r"r\.right\.foot_imugyro\s*:z", r"right.*foot.*gyro.*:z"],
    "l_foot_gyro_y": [r"l\.left\.foot_imugyro\s*:y",  r"left.*foot.*gyro.*:y"],
    "l_foot_gyro_x": [r"l\.left\.foot_imugyro\s*:x",  r"left.*foot.*gyro.*:x"],
    "l_foot_gyro_z": [r"l\.left\.foot_imugyro\s*:z",  r"left.*foot.*gyro.*:z"],
}

REQUIRED_ALWAYS = ["time", "right_hip", "left_hip", "pelvis_tilt",
                   "pelvis_acc_x", "pelvis_acc_y", "pelvis_acc_z"]


def find_col(df, key):
    for pat in COL_PATTERNS[key]:
        for c in df.columns:
            if re.search(pat, c, re.IGNORECASE):
                return c
    return None


def detect_columns(df):
    col_map = {k: find_col(df, k) for k in COL_PATTERNS}
    missing = [k for k in REQUIRED_ALWAYS if col_map[k] is None]

    # Determine event format
    has_event_cols = (col_map["right_events"] is not None and
                      col_map["left_events"]  is not None)
    has_fsw_cols   = (col_map["fsw_right"] is not None and
                      col_map["fsw_left"]  is not None)

    has_foot_gyro = (
        any(col_map.get(k) for k in ["r_foot_gyro_y","r_foot_gyro_x","r_foot_gyro_z"]) and
        any(col_map.get(k) for k in ["l_foot_gyro_y","l_foot_gyro_x","l_foot_gyro_z"])
    )
    if not has_event_cols and not has_fsw_cols and not has_foot_gyro:
        missing += ["no gait event source: need event labels, FSW, or foot IMU gyro columns"]

    if has_event_cols:
        fmt = "A_events"
    elif has_fsw_cols:
        fmt = "B_fsw"
    else:
        fmt = "C_imu"
    return col_map, missing, fmt


# ──────────────────────────────────────────────────────────────────────────────
#  GAIT EVENT DETECTION  (two methods)
# ──────────────────────────────────────────────────────────────────────────────

def events_from_labels(df, col_map):
    """Format A: extract HS/TO from On/Off event label columns."""
    def _dedup(col_name):
        ev = df[df[col_name].astype(str).str.strip().isin(["On", "Off"])].copy()
        ev["event"] = ev[col_name].astype(str).str.strip()
        ev["group"] = (ev["event"] != ev["event"].shift()).cumsum()
        ev = ev.groupby("group").first().reset_index(drop=True)
        hs = ev[ev["event"] == "On"][col_map["time"]].values
        to = ev[ev["event"] == "Off"][col_map["time"]].values
        return hs, to
    return _dedup(col_map["right_events"]), _dedup(col_map["left_events"])


def events_from_fsw(df, col_map, fs, threshold=0.9):
    """
    Format B: detect HS/TO from foot-switch voltage signal.

    A median filter removes brief noise spikes before thresholding.
    Window = 150 ms (minimum physiological contact duration).
    Rising edge  (0 → 1) = heel strike
    Falling edge (1 → 0) = toe off
    """
    time = df[col_map["time"]].values
    win  = max(3, int(0.15 * fs))
    win  = win if win % 2 == 1 else win + 1   # must be odd for medfilt
    win  = min(win, 51)

    def _detect(fsw_col):
        fsw     = df[fsw_col].values
        contact = (medfilt(fsw, kernel_size=win) > threshold).astype(int)
        diff    = np.diff(contact, prepend=contact[0])
        hs      = time[np.where(diff ==  1)[0]]
        to      = time[np.where(diff == -1)[0]]
        return hs, to

    R_hs, R_to = _detect(col_map["fsw_right"])
    L_hs, L_to = _detect(col_map["fsw_left"])
    return (R_hs, R_to), (L_hs, L_to)


def events_from_foot_gyro(df, col_map, fs):
    """
    Format C — Salarian (2004) foot IMU gyroscope algorithm.
    Used when no foot-switch hardware or event labels exist.

    Per foot:
      1. Select axis with largest std from Y/X/Z foot gyro (typically Y).
      2. Low-pass filter at 20 Hz.
      3. Detect large negative peaks >= 150 deg/s (mid-swing phase).
         Minimum 800 ms between peaks = one detection per stride.
      4. Heel Strike = first upward zero-crossing AFTER each peak.
      5. Toe-Off    = last downward zero-crossing BEFORE each peak.
    """
    time     = df[col_map["time"]].values
    min_dist = int(0.8 * fs)

    def _best_col(axis_keys):
        best_col, best_std = None, 0.0
        for key in axis_keys:
            col = col_map.get(key)
            if col and col in df.columns:
                s = float(df[col].values.std())
                if s > best_std:
                    best_std, best_col = s, col
        return best_col

    def _detect_one(gyro_col):
        if gyro_col is None:
            return np.array([]), np.array([])
        raw  = df[gyro_col].values
        b, a = sp_signal.butter(4, 20.0 / (fs / 2), btype="low")
        filt = sp_signal.filtfilt(b, a, raw)
        neg_peaks, _ = find_peaks(-filt, height=150, distance=min_dist, prominence=150)
        if len(neg_peaks) == 0:
            return np.array([]), np.array([])
        hs_list, to_list = [], []
        for pk in neg_peaks:
            fwd = filt[pk:]
            up  = np.where(np.diff(np.sign(fwd)) > 0)[0]
            if len(up) > 0:
                hs_list.append(time[pk + up[0]])
            bwd = filt[:pk + 1]
            dn  = np.where(np.diff(np.sign(bwd)) < 0)[0]
            to_list.append(time[dn[-1]] if len(dn) > 0 else np.nan)
        return np.array(hs_list), np.array(to_list)

    r_col = _best_col(["r_foot_gyro_y", "r_foot_gyro_x", "r_foot_gyro_z"])
    l_col = _best_col(["l_foot_gyro_y", "l_foot_gyro_x", "l_foot_gyro_z"])
    R_hs, R_to = _detect_one(r_col)
    L_hs, L_to = _detect_one(l_col)
    return (R_hs, R_to), (L_hs, L_to)


# ──────────────────────────────────────────────────────────────────────────────
#  GAIT CALCULATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def _swing_times(hs, to):
    """
    Swing time for each heel strike = time since the most recent toe-off
    on the same foot.  sw[i] = hs[i] - last TO before hs[i].
    """
    sw = []
    for h in hs:
        prev = to[to < h]
        sw.append(float(h - prev[-1]) if len(prev) > 0 else np.nan)
    return np.array(sw)


def _sl_zijlstra(hs_times, to_times, theta_abs, time, l_leg):
    """
    Stride length via Zijlstra & Hof (2003) with stride-local velocity correction.

    Formula per stride i (hs[i] to hs[i+1]):
        SL_geom = L x [sin(theta2) + sin(-theta1)]   geometric term
        V_local = SL_geom / stride_time               local speed estimate
        ts      = swing time at hs[i+1]
        SL      = SL_geom + V_local x ts              full Zijlstra formula

    Why stride-local V (not iterative global V)?
    - Geometric-only underestimates ~31% because the sensor fusion hip
      angle is zeroed at standing, so extension contributes too little.
    - Global iterative V inflates ~19% due to circular SL -> V -> SL.
    - Stride-local V breaks the circularity and gives mean error -1.1%
      vs Zebris FDM ground truth (3-trial validation: +0.8%, -7.6%, +3.4%).
    """
    sw  = _swing_times(hs_times, to_times)
    sl  = []
    for i in range(len(hs_times) - 1):
        mask     = (time >= hs_times[i]) & (time < hs_times[i + 1])
        seg      = theta_abs[mask]
        if len(seg) < 3:
            sl.append(np.nan)
            continue
        stride_t = hs_times[i + 1] - hs_times[i]
        sl_geom  = l_leg * (np.sin(np.deg2rad(np.max(seg))) +
                            np.sin(np.deg2rad(-np.min(seg))))
        V_local  = sl_geom / stride_t
        ts       = sw[i + 1]
        sl.append(sl_geom + V_local * ts if not np.isnan(ts) else sl_geom)
    return np.array(sl)


def compute_cadence(R_hs, L_hs):
    all_hs = np.sort(np.concatenate([R_hs, L_hs]))
    return 60.0 / np.mean(np.diff(all_hs))


def compute_stride_time_cov(R_st, L_st):
    all_st = np.concatenate([R_st, L_st])
    return float(np.std(all_st) / np.mean(all_st) * 100)


def compute_trunk_rms(df, fs, col_map):
    px = df[col_map["pelvis_acc_x"]].values
    py = df[col_map["pelvis_acc_y"]].values
    pz = df[col_map["pelvis_acc_z"]].values
    b, a = sp_signal.butter(4, 0.1 / (fs / 2), btype="high")
    px_d = sp_signal.filtfilt(b, a, px)
    py_d = sp_signal.filtfilt(b, a, py)
    pz_d = sp_signal.filtfilt(b, a, pz)
    return {
        "Trunk RMS Vertical (g)": float(np.sqrt(np.mean(px_d ** 2))),
        "Trunk RMS ML (g)"      : float(np.sqrt(np.mean(py_d ** 2))),
        "Trunk RMS AP (g)"      : float(np.sqrt(np.mean(pz_d ** 2))),
        "Trunk RMS Total (g)"   : float(np.sqrt(np.mean(px_d**2 + py_d**2 + pz_d**2))),
    }


def compute_asymmetry_index(R_vals, L_vals):
    mr, ml = np.nanmean(R_vals), np.nanmean(L_vals)
    denom  = (mr + ml) / 2
    return 0.0 if denom == 0 else abs(mr - ml) / denom * 100


def compute_double_support(R_hs, R_to, L_hs, L_to):
    def _stance(hs, to):
        out = []
        for h in hs:
            nxt = to[to > h]
            if len(nxt) > 0:
                out.append((float(h), float(nxt[0])))
        return out
    ds = []
    for (rs, re_) in _stance(R_hs, R_to):
        for (ls, le) in _stance(L_hs, L_to):
            s, e = max(rs, ls), min(re_, le)
            if e > s:
                ds.append(e - s)
    return np.array(ds)


def analyse_trial(uploaded_file, l_leg: float) -> dict:
    """Full gait analysis. Auto-detects event format (label vs FSW voltage)."""
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    col_map, missing, fmt = detect_columns(df)
    if missing:
        avail = list(df.columns[-10:])
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Last 10 columns in file: {avail}"
        )

    time = df[col_map["time"]].values
    fs   = round(1.0 / float(np.median(np.diff(time[:500]))), 1)

    # ── Detect gait events (format-dependent) ─────────────────────────────
    if fmt == "A_events":
        (R_hs, R_to), (L_hs, L_to) = events_from_labels(df, col_map)
    elif fmt == "B_fsw":
        (R_hs, R_to), (L_hs, L_to) = events_from_fsw(df, col_map, fs)
    else:  # C_imu
        (R_hs, R_to), (L_hs, L_to) = events_from_foot_gyro(df, col_map, fs)

    # Remove t=0 standing heel-strike (patient already on ground at trial start)
    if len(R_hs) > 0 and R_hs[0] == time[0]:
        R_hs = R_hs[1:]
    if len(L_hs) > 0 and L_hs[0] == time[0]:
        L_hs = L_hs[1:]

    if len(R_hs) < 2 or len(L_hs) < 2:
        raise ValueError(
            f"Not enough gait events (R: {len(R_hs)}, L: {len(L_hs)}). "
            f"Format detected: {fmt}. "
            f"Check foot-switch signal or event labels."
        )

    # ── Stride length (Zijlstra, pelvis-tilt corrected) ───────────────────
    pelvis_tilt = df[col_map["pelvis_tilt"]].values
    R_hip_abs   = df[col_map["right_hip"]].values + pelvis_tilt
    L_hip_abs   = df[col_map["left_hip"]].values  + pelvis_tilt

    R_st    = np.diff(R_hs);  L_st    = np.diff(L_hs)
    R_sl    = _sl_zijlstra(R_hs, R_to, R_hip_abs, time, l_leg)
    L_sl    = _sl_zijlstra(L_hs, L_to, L_hip_abs, time, l_leg)
    # Speed = SL / stride_time (SL already includes V*ts correction)
    R_speed = R_sl / R_st[:len(R_sl)]
    L_speed = L_sl / L_st[:len(L_sl)]

    all_st    = np.concatenate([R_st, L_st])
    all_sl    = np.concatenate([R_sl, L_sl])
    all_speed = np.concatenate([R_speed, L_speed])

    ds_arr  = compute_double_support(R_hs, R_to, L_hs, L_to)
    mean_ds = float(np.mean(ds_arr)) if len(ds_arr) > 0 else 0.0
    trunk   = compute_trunk_rms(df, fs, col_map)

    summary = {
        "Gait Speed (m/s)"                : round(float(np.nanmean(all_speed)), 4),
        "Stride Length (m)"               : round(float(np.nanmean(all_sl)),    4),
        "Cadence (steps/min)"             : round(compute_cadence(R_hs, L_hs),  4),
        "Stride Time CoV (%)"             : round(compute_stride_time_cov(R_st, L_st), 4),
        "Trunk RMS Vertical (g)"          : round(trunk["Trunk RMS Vertical (g)"], 4),
        "Trunk RMS ML (g)"                : round(trunk["Trunk RMS ML (g)"],       4),
        "Trunk RMS AP (g)"                : round(trunk["Trunk RMS AP (g)"],       4),
        "Trunk RMS Total (g)"             : round(trunk["Trunk RMS Total (g)"],    4),
        "Asymmetry Index Stride Time (%)": round(compute_asymmetry_index(R_st, L_st), 4),
        "Asymmetry Index Stride Len (%)": round(compute_asymmetry_index(R_sl, L_sl), 4),
        "Double Support Time (s)"         : round(mean_ds, 4),
        "Double Support Time (% gait)"    : round(mean_ds / np.nanmean(all_st) * 100, 4),
    }

    rows = []
    for i in range(min(len(R_st), len(R_sl), len(R_speed))):
        rows.append({"Side": "Right", "Stride": i + 1,
                     "Stride Time (s)":   round(float(R_st[i]),    4),
                     "Stride Length (m)": round(float(R_sl[i]),    4),
                     "Speed (m/s)":       round(float(R_speed[i]), 4)})
    for i in range(min(len(L_st), len(L_sl), len(L_speed))):
        rows.append({"Side": "Left", "Stride": i + 1,
                     "Stride Time (s)":   round(float(L_st[i]),    4),
                     "Stride Length (m)": round(float(L_sl[i]),    4),
                     "Speed (m/s)":       round(float(L_speed[i]), 4)})

    stride_df = (pd.DataFrame(rows)
                   .sort_values(["Stride", "Side"])
                   .reset_index(drop=True))

    return {
        "summary"  : summary,
        "stride_df": stride_df,
        "fmt"      : fmt,
        "col_map"  : col_map,
        "n_samples": len(df),
        "duration" : round(float(time[-1]), 2),
        "fs"       : fs,
        "R_hs"     : len(R_hs),
        "L_hs"     : len(L_hs),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

PARAM_UNITS = {
    "Gait Speed (m/s)"                : "m/s",
    "Stride Length (m)"               : "m",
    "Cadence (steps/min)"             : "steps/min",
    "Stride Time CoV (%)"             : "%",
    "Trunk RMS Vertical (g)"          : "g",
    "Trunk RMS ML (g)"                : "g",
    "Trunk RMS AP (g)"                : "g",
    "Trunk RMS Total (g)"             : "g",
    "Asymmetry Index Stride Time (%)": "%",
    "Asymmetry Index Stride Len (%)": "%",
    "Double Support Time (s)"         : "s",
    "Double Support Time (% gait)"    : "%",
}

PARAM_DISPLAY = {
    "Gait Speed (m/s)"                : "Gait Speed",
    "Stride Length (m)"               : "Stride Length",
    "Cadence (steps/min)"             : "Cadence",
    "Stride Time CoV (%)"             : "Stride Time CoV",
    "Trunk RMS Vertical (g)"          : "Trunk RMS Vertical",
    "Trunk RMS ML (g)"                : "Trunk RMS ML",
    "Trunk RMS AP (g)"                : "Trunk RMS AP",
    "Trunk RMS Total (g)"             : "Trunk RMS Total",
    "Asymmetry Index Stride Time (%)": "Asymmetry (stride time)",
    "Asymmetry Index Stride Len (%)": "Asymmetry (stride len)",
    "Double Support Time (s)"         : "Double Support",
    "Double Support Time (% gait)"    : "Double Support % gait",
}

FMT_LABELS = {
    "A_events": ("Format A", "fmt-a", "event labels"),
    "B_fsw"   : ("Format B", "fmt-b", "FSW voltage"),
    "C_imu"   : ("Format C", "fmt-c", "foot gyro IMU"),
}


def render_chips(summary: dict):
    chips = '<div class="metric-row">'
    for key, val in summary.items():
        chips += (
            f'<div class="chip">'
            f'<span class="chip-label">{PARAM_DISPLAY.get(key, key)}</span>'
            f'<span class="chip-value">{val:.3f}</span>'
            f'<span class="chip-unit">{PARAM_UNITS.get(key, "")}</span>'
            f'</div>'
        )
    chips += '</div>'
    st.markdown(chips, unsafe_allow_html=True)


def render_trial(label: str, result: dict):
    fmt_label, fmt_cls, fmt_desc = FMT_LABELS.get(result["fmt"], ("?", "fmt-a", "?"))
    st.markdown(
        f'<div class="trial-header">▸ {label}'
        f'<span class="fmt-badge {fmt_cls}">{fmt_label} · {fmt_desc}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="info-tag">'
        f'{result["n_samples"]:,} samples · {result["duration"]} s · '
        f'{result["fs"]} Hz · R HS: {result["R_hs"]} · L HS: {result["L_hs"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    render_chips(result["summary"])

    with st.expander("Per-stride breakdown", expanded=False):
        st.dataframe(
            result["stride_df"].style.format({
                "Stride Time (s)"  : "{:.4f}",
                "Stride Length (m)": "{:.4f}",
                "Speed (m/s)"      : "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ Settings")
    st.markdown("---")

    l_leg = st.number_input(
        "腰腿長  L_LEG (m)",
        min_value=0.50, max_value=1.20,
        value=0.90, step=0.01, format="%.2f",
        help="L4/L5 posterior surface to floor (m). Measure per patient standing upright.",
    )
    st.markdown('<div class="info-tag">L4/L5 後側面至地面垂直距離</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Upload trials")
    st.markdown(
        '<div class="info-tag">'
        'Cometa EmgTools .csv · up to 3 trials<br>'
        'Format A (event labels), B (FSW voltage)<br>'
        'or C (foot gyro IMU-only) — auto-detected'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    f1 = st.file_uploader("Trial 1", type=["csv"], key="f1")
    f2 = st.file_uploader("Trial 2", type=["csv"], key="f2")
    f3 = st.file_uploader("Trial 3", type=["csv"], key="f3")

    run = st.button("▶  Analyse", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        '<div class="info-tag">'
        'SL = L × [sin(θ₂) + sin(−θ₁)]<br>'
        'θ = hip flex/ext + pelvis tilt<br>'
        'Zijlstra & Hof (2003)'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">PD Gait Analysis</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='main-sub'>Parkinson's Disease · IMU · Cometa EmgTools</div>",
    unsafe_allow_html=True,
)

if not run:
    st.info("Upload up to 3 trial CSV files in the sidebar, set 腰腿長, then click **▶ Analyse**.")
    st.stop()

uploaded = [(f"Trial {i+1}", f) for i, f in enumerate([f1, f2, f3]) if f is not None]
if not uploaded:
    st.warning("Please upload at least one CSV file.")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
results, errors = {}, {}
for label, file in uploaded:
    try:
        results[label] = analyse_trial(file, l_leg)
    except Exception as e:
        errors[label] = str(e)

for label, err in errors.items():
    with st.expander(f"❌ {label} — error (click to expand)", expanded=True):
        st.error(err)
        st.markdown(
            "**Common causes:**\n"
            "- Missing foot-switch, event label, or foot IMU gyro columns\n"
            "- Hip angle or pelvis tilt column not found\n"
            "- File is not a processed EmgTools export"
        )

if not results:
    st.stop()

# ── Trial results ─────────────────────────────────────────────────────────────
st.markdown("## Trial Results")
for label, res in results.items():
    render_trial(label, res)

# ── Averaged summary / ML table ──────────────────────────────────────────────
st.markdown('<div class="ml-header">▸ Averaged Summary  —  ML Model Input</div>', unsafe_allow_html=True)

all_summaries = [res["summary"] for res in results.values()]

# Column order as requested
PARAM_ORDER = [
    "Gait Speed (m/s)",
    "Stride Length (m)",
    "Cadence (steps/min)",
    "Stride Time CoV (%)",
    "Asymmetry Index Stride Time (%)",
    "Asymmetry Index Stride Len (%)",
    "Trunk RMS ML (g)",
    "Trunk RMS AP (g)",
    "Trunk RMS Vertical (g)",
    "Trunk RMS Total (g)",
    "Double Support Time (s)",
    "Double Support Time (% gait)",
]

# Transposed layout:
#   Rows    = Trial 1 / Trial 2 / Trial 3 / Mean (all trials)
#   Columns = each parameter in requested order

trial_labels = list(results.keys())

row_data = {}
for i, lbl in enumerate(trial_labels):
    row_data[lbl] = {p: round(all_summaries[i][p], 4) for p in PARAM_ORDER}

means = {p: round(float(np.mean([s[p] for s in all_summaries])), 4) for p in PARAM_ORDER}
row_data["Mean (all trials)"] = means

# Build DataFrame: rows = trial labels, columns = parameters
summary_df = pd.DataFrame(row_data).T
summary_df.index.name = None

# Rename columns: "Display Name (unit)"
summary_df.columns = [
    f"{PARAM_DISPLAY.get(p, p)} ({PARAM_UNITS.get(p, '')})" for p in PARAM_ORDER
]

def _highlight_transposed(df):
    s = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Mean (all trials)" in df.index:
        s.loc["Mean (all trials)"] = (
            "background-color:#e8f7ef; color:#1a7a4a; font-weight:600"
        )
    return s

st.dataframe(
    summary_df.style
        .apply(_highlight_transposed, axis=None)
        .format("{:.4f}"),
    use_container_width=True,
    height=175,
)

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
c1, c2, _ = st.columns([1, 1, 3])

# Export keeps original parameter key names for ML downstream use
export_df = pd.DataFrame(row_data).T
export_df.columns = PARAM_ORDER
export_df.index.name = "Trial"

with c1:
    st.download_button(
        "⬇ Download summary CSV",
        data=export_df.to_csv().encode("utf-8"),
        file_name="gait_summary.csv", mime="text/csv",
        use_container_width=True,
    )
with c2:
    ml_row = {
        PARAM_DISPLAY.get(p, p)
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace("%", "pct").replace("/", "_"):
        means[p]
        for p in PARAM_ORDER
    }
    st.download_button(
        "⬇ Download ML-ready row",
        data=pd.DataFrame([ml_row]).to_csv(index=False).encode("utf-8"),
        file_name="gait_ml_features.csv", mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    f'<div class="info-tag" style="margin-top:8px">'
    f'腰腿長: {l_leg:.2f} m · {len(results)} trial(s) · '
    f'Pelvis tilt correction ✓ · Format auto-detected ✓'
    f'</div>',
    unsafe_allow_html=True,
)