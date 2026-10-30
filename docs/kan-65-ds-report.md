# KAN-65 report: can a +24 h model beat persistence? EDA, features, walk-forward backtest

`[Data Scientist - Sonnet 5.5]` - branch `kan-65/ds-eda-features` - DRAFT, offline analysis only.

## สรุปสำหรับ Boss (ภาษาไทย)

ข้อมูลชุดนี้มีรอยต่อสำคัญที่ KAN-15/64 มองข้าม: ตั้งแต่ 2026-07-31 17:00 UTC แหล่งข้อมูลเปลี่ยนจาก openaq (เรียบเหมือนค่าเฉลี่ยเคลื่อนที่ ไม่มีวัฏจักรรายวัน ค่าเฉลี่ย 24 µg/m³) เป็น air4thai (รายชั่วโมงจริง ค่าแกว่งมากกว่า 4 เท่า ค่าเฉลี่ย 13 µg/m³) ดังนั้น test span ของ KAN-15 เกือบทั้งหมดคือการทดสอบข้ามแหล่งข้อมูล (train บน openaq, test บน air4thai) และโมเดลที่มีแต่ `pm25_latest` จึงเสมอกับ persistence เป็นเรื่องที่คาดได้ เมื่อเพิ่มฟีเจอร์ (rolling mean/max/std, lag 24/48/72/168 ชม., ปฏิทิน, สถานีข้างเคียง) และจูนบน validation slice ภายในช่วง train เท่านั้น ผู้สมัครที่เลือกด้วย validation (val-selected) ดีกว่า persistence บน holdout ของ KAN-15 ที่ +24 ชม.: MAE ต่างกัน -0.151 µg/m³ (CI 95% day-block [-0.217, -0.089], หลังแก้ Bonferroni 99 การเปรียบเทียบ [-0.272, -0.044]) และดีกว่าใน 4 จาก 4 fold ที่ทดสอบบนข้อมูล air4thai (รวม -0.436, [-0.713, -0.173]) แต่กฎ promote ที่ตั้งไว้ล่วงหน้า **ไม่ผ่าน** 2 ข้อ: (b) CI หลังแก้ Bonferroni ของ fold รวมคร่อมศูนย์เล็กน้อย (+0.015) และ (d) ในชั่วโมงมลพิษสูง (>37.5) ที่เลือกตาม target จริง โมเดลแย่กว่า persistence (+5.15 µg/m³, n=76 ชั่วโมง คือช่วงเริ่มต้นของ episode ที่โมเดลทำนายต่ำเกินไป) แม้เมื่อเลือกตามค่า latest >37.5 จะดีกว่ามาก (-13.5) ข้อสรุป: **ยังไม่ promote** แนวโน้มการได้เปรียบเป็นจริงและสม่ำเสมอ แต่ข้อมูล air4thai มีเพียง ~90 วัน ไม่มีฤดูฝุ่นสูง จึงต้องใช้ข้อมูลมากขึ้น (โดยเฉพาะ backfill air4thai รายชั่วโมงย้อนหลัง) และรัน challenger แบบ shadow ก่อน

**โมเดล global (horizon เป็นฟีเจอร์ ใช้ร่วมทุกสถานี):** แย่กว่า persistence บน holdout ทุก horizon (+6: +0.17, +12: +0.16, +24: +0.05 และเมื่อเพิ่ม station id: +0.20, +0.18, +0.09 µg/m³ โดย CI ของ +24 ไม่คร่อมศูนย์: [+0.018, +0.087] และ [+0.050, +0.133]) แต่ดีกว่าโมเดล per-horizon ปัจจุบันบนข้อมูล air4thai ที่ทุก horizon (+6: -0.147, +12: -0.385, +24: -0.310 µg/m³) เทียบกับตัวที่จูนแล้ว (val-selected, per-horizon) global ไม่ชนะ การนำไปใช้ต้องแก้ serving loader (ปัจจุบันรับเฉพาะ bundle แบบหนึ่ง booster ต่อ horizon หรือ single-horizon และ feature_list ต้องมาจาก feature builder ซึ่งไม่มี `horizon`)

## Verdict

**More data needed; do not promote now.** Pre-registered rule (docs/kan-65-ds-plan.md, Amendment 3 item 6 and Amendment 4) for the single primary candidate (`val_best`) at +24 h:

| Condition | Result |
|---|---|
| (a) KAN-15 holdout, Bonferroni (99 claims) CI entirely below zero | PASS: -0.151, CI [-0.272, -0.044] |
| (b) pooled regime-B folds, Bonferroni CI entirely below zero | FAIL by 0.015: -0.436, CI [-0.953, +0.015] (95% CI [-0.713, -0.173]; Holm p 0.034) |
| (c) beats persistence on point MAE in >= 3 of 4 regime-B folds | PASS: 4/4 |
| (d) no deterioration in high-pollution hours (point diff <= 0), target > 37.5 and latest > 37.5 | FAIL in regime-B pool by target: +5.15 [+3.2, +7.0], n = 76 h. Passes by latest (-13.5), in holdout (-0.43 / -1.76) and in regime-A pool (-1.31 / -1.66) |

The verdict is not rescued by post hoc candidate picking: LightGBM direct (not val-selected) and the one-parameter smoother pass (a)-(c) but also fail (d) by target. Point of honesty on (d): conditioning on the realised target > 37.5 mechanically favours persistence at episode onsets (persistence bias there is already -23 ug/m3), so some of this is unavoidable; but the primary also under-predicts those onsets more (bias -28 vs -23 in regime B), and alerting is exactly the use case, so I do not treat (d) as an artifact. It is only 76 hours, a handful of episodes.

## 1. Data, quality, exclusions

- File `data/history.duckdb`, SHA-256 `fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09`: **identical to the KAN-15 snapshot**. Tables `observations` (64,986 rows) and `forecast_vintages` (1,032, unused). Five stations, 2024-10-28 00:00 to 2026-10-28 21:00 UTC, no negative or non-finite values, all timestamps on the hour.
- Grid coverage per station is only 74-75% of hours: gaps >= 48 h: 2025-04-25..05-07, 2025-06-30..08-25 (1,331 h), 2025-10-10..11-18 (939 h), **2026-07-02..07-29 (664 h, all stations)**, bkp100t/bkp101t 2026-05-08..06-09 (744 h; contains the known 05-19..21 gap), a few shorter ones. Gaps are never imputed; a sample needs a target and a latest reading in [origin-3h, origin-1h] (KAN-15/64 rule). The harness asserts that no sample targets the known gap.
- **Regime break (new finding):** `source` is `openaq` until 2026-07-31 16:00 UTC and `air4thai` from 17:00 at all five stations.

| | openaq (regime A) | air4thai (regime B) |
|---|---|---|
| Span | 2024-10-28 .. 2026-07-02 | 2026-07-29 .. 2026-10-28 |
| Mean PM2.5 (5 stations) | 19-31 (24 at 02t) | 12-17 (13 at 02t) |
| Mean absolute 1 h change | 0.27-0.41 | 1.3 (02t, 59t, 61t), 3.2-4.3 (bkp stations) |
| Identical consecutive values | 10-15% | 0-1.4% |
| Hour-of-day profile (5-station mean, local) | flat 24.1-25.0 | real cycle (12-16 by UTC hour) |
| Share of hours > 37.5 | 6-26% by station | 0.05-2.3% |

  Regime A behaves like a moving-average product, not raw hourly readings (hypothesis; not verified, the export does not say). Consequence: KAN-15 trains on regime A (52,563 rows) and tests on regime B (all but ~100 pairs). It is a cross-source transfer test.
- Test set (KAN-15 rule, lag 1 h): 10,729 pairs at +24 h, 91 target days; persistence MAE 4.4976 reproduces KAN-64 exactly.

## 2. EDA findings

Design decisions used only data with target before the holdout (A-heavy); holdout shown for description.

**Autocorrelation (pooled stations):**

| Lag h | 1 | 6 | 12 | 24 | 48 | 168 |
|---|---:|---:|---:|---:|---:|---:|
| Train span (mostly regime A) | 0.997 | 0.979 | 0.946 | 0.859 | 0.717 | 0.556 |
| Holdout span (regime B) | 0.872 | 0.627 | 0.525 | 0.463 | 0.323 | 0.225 |

Regime B is far less predictable from its own past; that is where a better level estimate (smoothing, mean reversion) can pay off.

**Persistence MAE at +24 h by group (ug/m3):**

| Group | Train (A) | Holdout (B) |
|---|---:|---:|
| Overall | 4.85 | 4.50 |
| Hour of day, local 00-05 / 06-11 / 12-17 / 18-23 | 4.62 / 5.00 / 5.00 / 4.74 | 4.64 / 4.68 / 4.14 / 4.53 |
| Station 02t / 59t / 61t / bkp100t / bkp101t | 4.56 / 3.81 / 4.07 / 6.35 / 5.41 | 3.65 / 3.44 / 3.60 / 5.95 / 5.85 |
| Latest reading <=15 / 15-25 / 25-37.5 / >37.5 | 2.43 / 3.82 / 5.77 / 9.51 | 3.59 / 5.24 / 9.84 / 24.3 (n = 75) |
| Month (all data) | Jan 2025: 10.1, Jun 2025: 2.2, Jan 2026: 7.3, Jun 2026: 1.7 | Aug 3.0, Sep 3.9, Oct 6.8 |

- Error scales with level and season (haze season Dec-Feb: 6-10; rainy season: ~2). Hour of day matters little in regime A (flat profile), more in regime B.
- Persistence is biased after movements: if PM2.5 fell > 3 ug/m3 over the last 6 h the +24 h bias is +5.1 (over-forecast); after a rise it is -1.2 (n = 3,432 / 3,305). Mean reversion is real.
- OLS slope of y(+24) on latest is 0.86 (train): shrinkage toward the mean helps.
- **Cross-station:** concurrent correlation 0.91-0.97; lead-lag correlation is symmetric around lag 0 (0.939 at 0, 0.936 at +/-3 h). No neighbour leads another; a neighbour's latest reading is a noise-reducing duplicate, not an early warning. Simple probe (train, +24 h MAE): latest 4.85, mean24 5.68, 0.5 latest + 0.5 mean24 5.14, neighbour mean 6.41, t-48 6.76, t-168 8.36.
- Weekday means 23.6-25.7 (Mon..Sun), no strong weekly effect.

## 3. Method (pre-registered; see plan and Amendments 1-4)

- Walk-forward, chronological, expanding window, always training on targets strictly before the boundary (gap = one horizon). Runs: F1-F3 (regime A to A), F4 (A to B), R1-R3 (regime-B-native expanding folds, training variant chosen on validation: B-only or A+B with source flag), HOLDOUT (KAN-15 split, reported separately). F4/R1-R3 test windows lie inside the holdout span, so folds and holdout are not independent.
- Features (all as-of origin-1 h, `src/pm25_forecast/experiments/dataset.py`): latest; cyclic target hour/day-of-week/month, station id; readings at target-24 h (NaN at +24 by rule), -48, -72, -168 h; rolling mean/max/std over 3/6/24 h; trend (latest minus reading 3/6 h before); neighbours' mean latest and mean 24 h average. Weather, wind, boundary layer: **needs data**, not used.
- Targets: direct, residual over persistence, log-ratio. Models: LightGBM (objective l2/l1 x leaves 7/31, early stopping on a validation slice inside the training period: last 45 days, last 10 days for regime-B folds), ridge (closed form, alpha 0.1-100), one-parameter smoothers, ensemble (ridge + LGBM-residual), shrink toward persistence, and the global multi-horizon LightGBM (horizon as feature; with and without station id).
- Primary candidate `val_best`: family with the lowest validation MAE among LightGBM direct / LightGBM residual / ridge, per run and horizon.
- Statistics: paired MAE difference vs persistence on identical station-hours, day-block bootstrap on target UTC date (seed 20261030; 10k draws per run, 50k for pooled groups). 99 claims (11 candidates x 3 horizons x 3 groups: holdout, pooled regime B, pooled regime A): Bonferroni CIs and Holm p-values. 231 further per-run comparisons are descriptive.
- Variants tried (from script counters): **3,344 tuned configurations**: 1,760 early-stopped LightGBM fits (1,584 per-horizon + 176 global) and 1,584 ridge fits (8 runs x 3 horizons x 4 feature sets x 3 targets x 4 alphas, doubled training-data variants for the 3 regime-B folds, etc.), plus 13 smoother variants and 113 refits of selected models (89 LightGBM, 24 ridge; the counters in `results.json` are 1,760 / 1,608 / 89). Total comparisons vs persistence 330 (99 corrected claims). This is more than the 1,536 estimate in Amendment 2, which was superseded by Amendment 3; selection used validation MAE only.

## 4. Results

### 4a. KAN-15 holdout, +24 h (persistence MAE 4.498, n = 10,729, 91 days)

| Candidate | MAE | Paired diff vs persistence [95% CI] | Bonferroni CI (99 claims) |
|---|---:|---|---|
| **val-selected best (primary; LightGBM log-ratio, set E)** | 4.347 | -0.151 [-0.217, -0.089] | [-0.272, -0.044] |
| current LightGBM (latest + t-24h), KAN-64 | 4.503 | +0.005 [-0.096, +0.103] | [-0.173, +0.175] |
| seasonal naive (t-48 h, the servable form) | 5.033 | +0.518 [+0.144, +0.939] | [-0.106, +1.332] |
| smoother (shrink toward 24 h mean, beta 0.3) | 4.212 | -0.285 [-0.368, -0.200] | [-0.435, -0.137] |
| ridge | 4.551 | +0.053 [-0.064, +0.161] | [-0.160, +0.235] |
| LightGBM direct (not the val-selected one) | 4.036 | -0.462 [-0.615, -0.318] | [-0.748, -0.215] |
| LightGBM residual / log-ratio | 4.347 | -0.151 [-0.217, -0.089] | [-0.272, -0.044] |
| ensemble ridge + LGBM-residual | 4.402 | -0.096 [-0.184, -0.018] | [-0.262, +0.031] |
| shrink LGBM-residual toward persistence | 4.376 | -0.121 [-0.171, -0.075] | [-0.213, -0.041] |
| global LightGBM (no id) | 4.551 | +0.053 [+0.018, +0.087] | [-0.013, +0.109] |
| global LightGBM (+ station id) | 4.590 | +0.092 [+0.050, +0.133] | [+0.016, +0.162] |

Same candidates at +6 h and +12 h on the holdout: nothing beats persistence (primary +0.156 [+0.094, +0.214] at +6 h; +0.056 [-0.067, +0.159] at +12 h). Full tables for every horizon and group: `out/kan65/tables.md`.

### 4b. Walk-forward, +24 h

Persistence MAE and paired difference of the primary candidate per run (95% day-block CI):

| Run | Test span | Persistence MAE | Primary diff [95% CI] |
|---|---|---:|---|
| F1 (A, haze) | Jan-Feb 2026 | 6.97 | -0.649 [-1.538, +0.258] |
| F2 (A) | Apr-May 2026 | 3.37 | -0.236 [-0.627, +0.176] |
| F3 (A) | Jun 2026 | 1.72 | -0.013 [-0.196, +0.164] |
| F4 (A to B) | Aug-Sep 11 2026 | 3.28 | -0.162 [-0.264, -0.066] |
| R1 (B native) | Sep 1-21 | 3.94 | -0.434 [-1.045, +0.161] |
| R2 (B native) | Sep 22-Oct 12 | 6.26 | -0.534 [-1.236, +0.200] |
| R3 (B native) | Oct 13-27 | 6.20 | -1.103 [-2.226, -0.162] |
| HOLDOUT | Jul 10-Oct 27 | 4.50 | -0.151 [-0.216, -0.090] |

Pooled (primary / LightGBM direct / one-parameter smoother, difference vs persistence with 95% CI and Bonferroni CI):

| Group, +24 h | Primary | LightGBM direct | Smoother |
|---|---|---|---|
| Regime B folds (n = 11,794) | -0.436 [-0.713, -0.173], B [-0.953, +0.015], 4/4 folds | -0.566 [-0.828, -0.320], B [-1.060, -0.132], 4/4 | -0.398 [-0.560, -0.235], B [-0.689, -0.105], 4/4 |
| Regime A folds (n = 12,776) | -0.341 [-0.733, +0.038], 3/3 | -0.225 [-0.637, +0.182], 2/3 | 0.000 (always chooses latest) |

Regime A (smoothed history) ridge and ensemble also gain (-0.52 / -0.51, 3/3 folds), mostly in the haze-season fold F1. Horizons +6 / +12 in regime B pooled: primary -0.230 [-0.347, -0.113] / -0.314 [-0.521, -0.118]; stability 3/4 at both.

### 4c. Global multi-horizon model (Boss question)

Per-horizon difference, global minus current per-horizon boosters (paired, 95% CI):

| Group | +6 h | +12 h | +24 h |
|---|---|---|---|
| Holdout, global (no id) | -0.012 [-0.061, +0.034] | -0.232 [-0.304, -0.160] | +0.048 [-0.042, +0.139] |
| Holdout, global (+ id) | +0.018 [-0.032, +0.068] | -0.206 [-0.278, -0.135] | +0.087 [-0.000, +0.175] |
| Regime B pooled, global (no id) | -0.147 [-0.266, -0.025] | -0.385 [-0.539, -0.234] | -0.310 [-0.523, -0.105] |
| Regime B pooled, global (+ id) | -0.161 [-0.278, -0.042] | -0.381 [-0.532, -0.234] | -0.338 [-0.563, -0.121] |

Against persistence the global models are worse on the holdout at every horizon (+6 h +0.17 / +0.20; +12 h +0.16 / +0.18; +24 h +0.05 / +0.09, no id / with id) and better in regime B pooled (+24: -0.334 / -0.362, 3/4 folds) but never better than the tuned per-horizon `val_best` (-0.436). Hypothesis check: structure sharing helps relative to the untuned current boosters (they are weak), the penalty against persistence appears on the holdout at all horizons (largest at +6 h) but not in regime A (+6 h: -0.20). Station id adds nothing consistent (holdout +0.04 worse, regime B -0.03 better).

**Serving constraint (not changed):** `models/serving.py` accepts a multi-horizon bundle with one booster per horizon or a single-horizon bundle (`_parse_bundle`; exactly one of `horizons` or `horizon_hours`+`booster`). `ServingModel.predict(horizon, features)` indexes a booster by horizon and builds the input row from `feature_list` only; every booster's `feature_name()` must equal the manifest `feature_list`, and every name must exist in `features.FEATURE_NAMES` (currently `pm25_latest`, `pm25_same_hour_24h`, `cams_pm25`). A global model therefore **would need a loader change**: a bundle shape with one booster plus injection of a `horizon` value (and, if used, a station id) into the row. Independently, any of the new features (rolling windows, lags beyond 24 h, calendar, neighbours) requires extending `features/build_features` (shared with training) and the live as-of query for 7 days of history and all five stations; and a residual or log-ratio target needs the loader/predict path to add back persistence (it returns the booster output directly). A direct-target per-horizon LightGBM with the new features needs only the feature-builder extension, no loader change.

### 4d. Bias, bands, high pollution (+24 h, primary vs persistence)

| Group | Band | n | Persistence MAE / bias | Primary MAE / bias |
|---|---|---:|---|---|
| Holdout | <= 15 | 7,285 | 3.48 / +1.49 | 3.40 / +1.70 |
| Holdout | 15-37.5 | 3,369 | 6.28 / -2.82 | 6.00 / -2.83 |
| Holdout | > 37.5 | 75 | 22.9 / -22.8 | 22.5 / -22.5 |
| Regime B pooled | <= 15 | 7,999 | 3.52 / +1.50 | 2.86 / +1.03 |
| Regime B pooled | 15-37.5 | 3,719 | 6.18 / -2.94 | 6.11 / -5.29 |
| Regime B pooled | > 37.5 | 76 | 23.1 / -23.0 | 28.2 / -28.2 |
| Regime A pooled | > 37.5 | 1,621 | 8.8 / -4.0 | 7.5 / -3.2 |

Overall pooled bias of the primary is -1.15 in regime B (under-prediction when the series rises) and +0.11 on the holdout. Where PM2.5 is low, it gains (-0.66 in the <= 15 band, regime B); where PM2.5 is rising from moderate to high it does not. A quantile or asymmetric loss check beyond the l1 objective (the l1 fits are conditional-median) was not run (see Skipped).

### 4e. Breakdowns, +24 h (MAE persistence / primary), holdout

By station: 02t 3.65/3.49, 59t 3.44/3.29, 61t 3.60/3.44, bkp100t 5.95/5.80, bkp101t 5.85/5.71 (the two stations with the noisiest regime-B series gain most with LightGBM direct: 5.26 and 5.15). By local hour group 00-05 / 06-11 / 12-17 / 18-23: 4.64/4.52, 4.68/4.50, 4.14/3.99, 4.53/4.38 (no hour-specific story). By source of the target: air4thai targets (n = 10,629) 4.51/4.36; openaq targets (n = 100) 2.92/3.04. By month in regime B: Aug 3.00/2.90, Sep 3.97/3.60, Oct 6.85/5.94 (gains grow as the level and volatility rise in October). All in `out/kan65/tables.md`.

### 4f. Leakage audit

| Feature | As-of rule (reference time r = origin - 1 h) |
|---|---|
| pm25_latest | newest reading with ts in [origin-3h, origin-1h] |
| hour/dow/month sin-cos | from the target timestamp (known at issue time) |
| station_id, src_air4thai | static / source of the latest reading |
| pm25_t24 | reading at target-24 h only if <= r; at +24 h that is the origin hour, so always NaN (asserted) |
| pm25_t48, t72, t168 | readings at target-48/72/168 h, always <= r at all three horizons |
| mean/max/std 3, 6, 24 | readings in (r-w, r] |
| trend3, trend6 | latest minus reading at origin-1h-3 / -6 |
| nbr_latest_mean, nbr_mean24 | other stations' latest / 24 h mean, same windows |

- Independent scalar re-derivation of every feature from a grid truncated to indices <= r for 300 random rows per horizon: **0 mismatching cells** at +6, +12, +24. A unit test (`tests/test_kan65_experiments.py`) corrupts every value at or after the origin and checks the features are unchanged.
- Deliberately leaky control (set D plus the neighbours' reading at the target hour, never a candidate): MAE 3.008 vs persistence 4.498, paired difference -1.490 [-1.854, -1.154]. The harness would expose leakage.
- Training rows have targets strictly before each boundary; validation slices are inside training; the test span is never used for tuning or early stopping.

## 5. Hypotheses

- H1 (hour-of-day lowers MAE): not supported as a standalone effect; no hour-specific pattern of gain. The selected models mostly use level features (rolling, lags, neighbours); in regime B the mean-reversion structure matters more than the diurnal cycle.
- H2 (residual target better than direct): mixed. Validation preferred residual / log-ratio in 5 of 8 runs at +24 h, but on the test spans direct LightGBM often scored lower MAE (holdout -0.462 vs -0.151). I would not claim either.
- H3 (neighbours help): weak and not isolated. The val-selected primary used set E (with neighbours) in 4 of 8 runs; no ablation was run, and lead-lag EDA says the neighbour acts as a noise reducer.
- H4 (smoothing is a better level estimate): supported in regime B. A single parameter (shrink latest toward the 24 h mean) gives -0.285 on the holdout and -0.398 in regime B folds (4/4) without machine learning, and it chooses plain latest in regime A.

## 6. Limitations and caveats

- Regime B has about 90 days: no haze season, only ~76 high hours, ~3 months of training for the B-native folds, validation slices of 10 days. Day-block bootstrap does not model multi-day dependence; the 7-day-block sensitivity promised in the plan was **not run**.
- Folds F4, R1-R3 lie within the holdout span; holdout and regime-B pool are one evidence source seen twice.
- A first full run was completed and inspected before Amendment 4 added the `val_best` candidate (seed-deterministic; the rerun reproduces the same model numbers). Disclosed in the plan.
- Whether regime A is a moving-average product is a hypothesis from its statistics; the provenance of the `openaq` series should be confirmed with the owner.
- The persistence reference uses the KAN-15 rule (latest within 3 h). Seasonal naive is evaluated on its own non-missing subset. The pair set is "target and persistence present", not the strict all-model intersection of `backtest.py`.
- Calendar features use Bangkok time (UTC+7).

## 7. What surprised me

1. The openaq/air4thai break: the headline history is a different measurement product, so most of the "2 years of data" does not describe the live feed.
2. On the smoothed regime-A series the one-parameter smoother always keeps plain `latest`, yet on the real hourly feed (regime B) a single shrink-toward-the-24-h-mean parameter beats persistence by 6-9% at +24 h.
3. Models trained only on the smoothed history still transfer a gain to the real hourly feed (holdout -0.15 to -0.46), i.e. the structure (mean reversion, rolling level) is shared.
4. The gain concentrates where persistence is worst by level (low levels, noisy bkp stations) and is absent at episode onsets.

## 8. What I would do with two more weeks

1. Backfill raw hourly Air4Thai history (regime B definition) for the five stations to remove the break; rerun this harness unchanged.
2. Collect weather features (data proposal below) and test them; wind and boundary-layer height are the strongest candidate drivers of onsets, which is where the model fails.
3. Add a 7-day-block bootstrap, a pinball-loss / quantile (q 0.5, 0.9) check, a station-neighbour ablation, per-station residual models, and conformal intervals for alerting.
4. Run the 1-parameter smoother and the direct LightGBM as shadow challengers in `predict` (needs the feature builder extension) and score live.
5. Define the onset metric formally (recall of hours > 37.5 24 h ahead) since MAE and rule (d) disagree about what matters.

**Data proposals (not fetched):** (i) hourly weather: wind speed/direction, boundary-layer height, rain, temperature (Open-Meteo or ERA5, hourly archive, keyed on station lat/lon; expected value: highest for onsets and washout); (ii) fire hotspots (NASA FIRMS) for the dry season; (iii) CAMS previous-run archive (the vintages table has 1,032 rows only); (iv) traffic proxies are low-value/high-cost. Each needs ingestion into the observation store with `available_at` stamps to respect the as-of rule.

## 9. Skipped, and why

Per-station and per-hour-bucket models, ExtraTrees/CatBoost/XGBoost (not installed except LightGBM; stations have ~2k regime-B pairs each), seasonal decomposition, pinball-loss quantile models, 7-day-block bootstrap, neighbour ablation. All were lower expected information than the regime-break and walk-forward questions.

## 10. Recommendation and confidence

- **More data needed; do not promote.** No model registration, no loader or feature-builder change in this PR.
- Next step with the best expected value: fix the data break (backfill) and rerun; then add a shadow challenger with the smoothing and rolling-level features.
- Confidence: high that the KAN-64 tie is explained by near-univariate inputs plus the regime mismatch; moderate-high that a real gain of roughly 0.15-0.45 ug/m3 (3-10%) exists in regime B at +24 h; low on production readiness because of the onset-hour deterioration and 90 days of regime-B data.

## Reproduce

```sh
PYTHONPATH=src python -m pm25_forecast.experiments.run_all \
  --database data/history.duckdb --out out/kan65      # about 13 minutes, 8 cores
PYTHONPATH=src python -m pm25_forecast.experiments.make_tables \
  out/kan65/results.json out/kan65/tables.md
```

Seeds: model 20261029, bootstrap 20261030. Library versions, snapshot SHA-256 and runtime are in `out/kan65/results.json` and `tables.md`. Code: `src/pm25_forecast/experiments/` (common, dataset, fitters, eda, run_all, make_tables), tests `tests/test_kan65_experiments.py`. Nothing under `src/pm25_forecast/` outside `experiments/`, no dashboard file and no production module was touched.
