# Data dictionary

The study dataset (one row per participant, n = 44) is not public (see README, *Data availability*).
`example_synthetic.xlsx` / `.csv` contain **simulated** values with the same column names so that the scripts can be run; they do not reproduce the published estimates.

Suffixes: `_pre` = before training, `_post` = after training, `diff_` = post − pre.
`ST` = single-task (usual) walking.

| Column | Description | Unit |
|---|---|---|
| `ID` | Study participant number (not a hospital identifier) | – |
| `Age` | Age at baseline | years |
| `Sex` | Sex (1 = male, 0 = female) | – |
| `Duration` | Disease duration (Table 1 only) | years |
| `LEDD` | Levodopa equivalent daily dose | mg |
| `UPDRS` | MDS-UPDRS Part III score | points |
| `MMSE` | Mini-Mental State Examination | points |
| `MoCA` | Montreal Cognitive Assessment | points |
| `PT` | Physical therapy–supervised training (1) vs self-directed exercise only (0) | – |
| `Speed_ST_pre` | Baseline comfortable gait speed | m/s |
| `Stride_length_ST_pre` | Baseline stride length | m |
| `Cadence_ST_pre` | Baseline cadence | steps/min |
| `Stride Time CoV_pre` | Baseline stride-time coefficient of variation | % |
| `Asymmetry (stride time)_pre` | Baseline stride-time asymmetry index | % |
| `Asymmetry (stride length)_pre` | Baseline stride-length asymmetry index | % |
| `locomotor_ST_pre` | Baseline locomotor-band (0.5–3 Hz) WPD energy | a.u. |
| `HF_ST_pre` | Baseline freeze-band (3–8 Hz) WPD energy | a.u. |
| `HF_index_ST_pre` | Baseline freeze index (freeze / locomotor energy) | ratio |
| `Speed_ST_post` | Comfortable gait speed after training (sensitivity analysis) | m/s |
| `diff_Speed` | Change in comfortable gait speed (primary outcome) | m/s |
| `diff_Stride_length` | Change in stride length | m |

Gait parameters are produced by `feature_extraction/IMU_PDgait_parameters_v1.py` and WPD features by `feature_extraction/PD_WPD_run3_TLLee.py`; per-participant values are the mean of up to three 7-m walking trials.
