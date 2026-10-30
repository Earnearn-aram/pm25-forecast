# KAN-65 report v2: can a +24 h model beat persistence? EDA, features, walk-forward backtest

`[Data Scientist - Sonnet 5.5]` - branch `kan-65/ds-eda-features` - DRAFT, offline analysis only. This is v2 after independent review; v1 numbers are kept in the appendix for transparency.

## สรุปสำหรับ Boss (ภาษาไทย)

หลังรีวิว ผมแก้การออกแบบ (Amendment 5, เขียนหลังเห็นผล v1 และระบุไว้ตรง ๆ): fold ของ regime B ไม่ซ้อนกันแล้ว (G1-G3, ทดสอบ 7,954 station-hour ไม่ซ้ำ, 67 วัน), ใช้ bootstrap แบบ block 7 วันควบคู่ block 1 วัน, ใช้โมเดลคอนฟิกคงที่ (ไม่จูน) และให้ smoother พารามิเตอร์เดียว (latest + beta x (mean24 - latest)) เป็นตัวหลัก ผล +24 ชม.: smoother ดีกว่า persistence บน holdout KAN-15 -0.383 µg/m³ (CI 7 วัน [-0.465, -0.297], หลัง Bonferroni 30 การเปรียบเทียบ [-0.507, -0.243]) และบน fold regime B ที่ไม่ซ้อนกัน -0.391 ([-0.557, -0.195], Bonferroni [-0.629, -0.057], ดีกว่าใน 3/3 fold; หมายเหตุ: ถ้าใช้ block 1 วัน CI แบบ Bonferroni ของ smoother ใน pool นี้คร่อมศูนย์เล็กน้อย +0.016) LightGBM direct ที่ตั้งคอนฟิกคงที่ได้ -0.408 และ -0.603 ใกล้เคียงกัน แต่ไม่ดีกว่า persistence ที่ +6/+12 บน holdout (train บน regime A เท่านั้น) กฎความปลอดภัยชั่วโมงมลพิษสูงแบบใหม่ (ใช้ค่า latest >37.5 ที่รู้ ณ เวลาพยากรณ์) **ทดสอบไม่ได้** ใน regime B (มีเพียง 74 station-hour, 18 วัน; เกณฑ์ขั้นต่ำ 200 และ 10 วัน) จึงไม่ถือว่าผ่านหรือไม่ผ่าน และการจับ spike (เพิ่ม >=15 µg/m³ ใน 24 ชม.) ทำได้น้อยมาก (จับได้ <=10% ของ 199 เหตุการณ์) ข้อสรุป: **แนะนำ shadow-run (ยังไม่ promote)** smoother ก่อน แล้ว LightGBM direct เป็น challenger ตัวที่สอง ส่วนข้อสงสัย regime break: ระดับต่างกัน (24 vs 13) เป็นผลของฤดูกาล (openaq ส.ค.-ต.ค. 2025 เฉลี่ย 13.85 เทียบ air4thai ส.ค.-ต.ค. 2026 เฉลี่ย 13.61) แต่ความเรียบต่างกัน 8.6 เท่า (|Δ1ชม.| 0.27 เทียบ 2.29) สมมติฐาน H-MA (openaq = ค่าเฉลี่ย 24 ชม.ย้อนหลัง) **ยังสรุปไม่ได้** (ใกล้เคียงบางสถิติแต่ไม่ตรงทั้งหมด) แต่หมายความว่า regime A ทำนายปริมาณที่เรียบกว่า ค่ารายชั่วโมงจริง

**โมเดล global (horizon เป็นฟีเจอร์):** ที่คอนฟิกคงที่ ไม่ต่างจาก persistence บน holdout ที่ +24 (-0.022, CI 7 วัน [-0.080, +0.035]) และดีกว่า persistence บน regime B (-0.49 ที่ +24) แต่ไม่ดีกว่า LightGBM direct ต่อ horizon (-0.60) การนำไปใช้ต้องแก้ serving loader

## What changed after review, and why

An independent reviewer reproduced every v1 number and found design problems. All changes are in `docs/kan-65-ds-plan.md` Amendment 5 / 5b, **written after seeing v1 results** (flagged in the amendment where a choice was informed by v1).

| Review finding | v1 | v2 fix |
|---|---|---|
| Overlapping test windows (F4 and R1 overlapped; folds inside the holdout span called independent) | F4, R1-R3 pooled | Consecutive non-overlapping regime-B folds G1-G3 (2026-08-22..09-11, 09-12..10-02, 10-03..10-27); the script asserts no (station, origin) is tested twice in a pool. Unique counts below. HOLDOUT still overlaps G1-G3 in time and is **not independent** of the regime-B pool |
| 7-day-block bootstrap promised but not run | 1-day blocks only | 1-day and 7-day blocks for every headline CI; decisions use 7-day |
| Rule (d) conditioned on the realised target | withdrawn | Forecast-time subset (latest reading > 37.5), a testability threshold (>= 200 station-hours and >= 10 days), and a separate onset-event metric |
| Selection on validation data from another regime | A-heavy validation for B tests | ML models use fixed pre-declared configurations (no tuning); only the smoother beta is selected, on same-regime validation; for the holdout no regime-B validation exists before its start, so beta is fixed at 0.5 |
| Large search, noisy selector | 3,344 configs, 99 claims | 5 candidates, fixed config, 30-claim family (5 candidates x 3 horizons x 2 headline groups), Holm and Bonferroni |
| Smoother under-exposed | one candidate among many | Headline candidate named in advance; beta procedure, stability and curve reported |
| Source of rows 07-29..07-31 | ambiguous | See below |

**Source of the rows between the gap and the switch.** The 664 h hole is 2026-07-02 01:00..07-29 18:00. The rows from 2026-07-29 18:00 to 2026-07-31 16:00 carry source `openaq` (regime A by label); `air4thai` starts at 2026-07-31 17:00. The regime boundary used everywhere is the **source label of the target row** (validation and B-only training use `source == air4thai`); the G folds start on 08-22 so every G test target is air4thai, and the holdout contains 100 openaq-labelled and 10,629 air4thai-labelled +24 h targets.

## Verdict

**Shadow-run, do not promote.** Pre-registered v2 rule (Amendment 5 item 7), headline candidate the one-parameter smoother (named a priori), +24 h:

| Condition | Smoother | LightGBM direct (fixed config) |
|---|---|---|
| (a) HOLDOUT 7-day-block Bonferroni(30) CI entirely < 0 | PASS: -0.383, [-0.507, -0.243] | PASS: -0.408, [-0.652, -0.208] |
| (b) unique regime-B pool, same | PASS (borderline: with 1-day blocks the Bonferroni CI is [-0.760, +0.016]): -0.391, [-0.629, -0.057] | PASS: -0.603, [-0.868, -0.317] |
| (c) point MAE below persistence in >= 2 of 3 regime-B folds | PASS 3/3 | PASS 3/3 |
| (d) safeguard (latest > 37.5), regime A pool | PASS (identical to persistence: beta = 0 chosen in every A fold) | PASS: -1.44 [-2.44, -0.41] |
| (d) safeguard, regime B pool | **untestable** (74 station-hours, 18 days) | **untestable** |
| Verdict | SHADOW | SHADOW (+24 h only) |

Rule text: PROMOTE needs a-d with a testable regime-B safeguard; a-c passing with an untestable regime-B safeguard means SHADOW. The safeguard cannot be verified on the live feed's regime because it contains almost no high-pollution hours (74 station-hours above 37.5, 18 days, 0.9% of the pool), no haze season. Not passing and not failing is the honest status.

## 1. Data, quality, exclusions

- File `data/history.duckdb`, SHA-256 `fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09`: **identical to the KAN-15 snapshot** (64,986 observation rows, 1,032 unused vintages, five stations, 2024-10-28 00:00..2026-10-28 21:00 UTC; no negative or non-finite values).
- Grid coverage is 74-75% of hours per station. Gaps >= 48 h: 2025-04-25..05-07, 2025-06-30..08-25 (1,331 h), 2025-10-10..11-18 (939 h), 2026-07-02..07-29 (664 h, all stations), bkp100t/bkp101t 2026-05-08..06-09 (744 h, contains the known 05-19..21 gap) and a few shorter ones. Gaps are never imputed; a sample needs a target and a latest reading in [origin-3h, origin-1h]; the harness asserts the known gap is never sampled.
- Pair counts (+24 h, unique station-hours / unique target days): HOLDOUT 10,729 / 91 (persistence MAE 4.4976, reproduces KAN-64); unique regime-B pool G1-G3 7,954 / 67; regime-A pool F1-F3 12,776 / 115. At +12 h and +6 h the regime-B pool has 7,999 and 8,024 station-hours.
- Regime table (by source label): openaq 2024-10-28..2026-07-31 16:00, air4thai from 2026-07-31 17:00.

| | openaq | air4thai |
|---|---|---|
| Mean PM2.5 across stations (all period) | 19-31 | 12-17 |
| Mean abs 1 h change | 0.27-0.41 | 1.3 (02t, 59t, 61t), 3.2-4.3 (bkp stations) |
| Identical consecutive values | 10-15% | 0-1.4% |
| Share of hours > 37.5 | 6-26% by station | 0.05-2.3% |

### H-MA: is the openaq series a trailing 24 h average? (Amendment 5b, rule declared before the run)

| Series (pooled stations) | mean | mean abs 1 h change | identical consecutive (0.1) | ACF lag 1 | ACF lag 24 | diurnal flatness |
|---|---:|---:|---:|---:|---:|---:|
| air4thai raw hourly (Aug-Oct 2026) | 13.61 | 2.292 | 0.005 | 0.872 | 0.464 | 0.287 |
| air4thai trailing 24 h mean (computed) | 13.61 | 0.187 | 0.208 | 0.998 | 0.684 | 0.005 |
| openaq, Aug-Oct 2025 (season control) | 13.85 | 0.267 | 0.169 | 0.956 | 0.350 | 0.049 |
| openaq, all | 24.68 | 0.358 | 0.128 | 0.997 | 0.860 | 0.067 |

- **Level:** season-matched, the openaq level (13.85) equals the air4thai level (13.61). The "24 vs 13" gap is seasonal (the air4thai era is Aug-Oct only), **not** a source effect.
- **Smoothness:** the openaq series is 8.6 times smoother than raw hourly values in mean |1 h change| (0.267 vs 2.292) and close to a computed trailing 24 h mean (0.187; ratio 0.70). But its lag-1 ACF (0.956) is below that of an exact trailing mean (0.998), its diurnal flatness is 10 times larger (0.049 vs 0.005) and the lag-24 ACF is 0.35 vs 0.68. By the declared rule (supported needs flatness within 1.5x and both ACF1 > 0.99; rejected needs |1 h change| ratio outside 0.5-2 or a raw-like diurnal amplitude) the verdict is **undecided**: heavily smoothed, plausibly a moving average of some other window or definition, not exactly a trailing 24 h mean of the hourly values.
- Supporting indirect evidence: in the openaq-era folds any shrinkage toward the 24 h mean makes things worse (beta = 0.5: +0.222 [+0.094, +0.356]), as expected if the series is already a smoothed quantity; in the air4thai era the same operation gains 8%.
- The source-switch hour equals the earliest hour the Air4Thai history endpoint served (docs/kan-9-history-findings.md), which fits an assembly artifact (source precedence) but does not by itself show the openaq values are averages.
- **Meaning for the target:** regime A is a smoother quantity than the live hourly value (8.6x less 1 h variability). Models trained on it predict a different target from the live feed; the training target for production should be the regime-B definition (a backfill of raw hourly Air4Thai history would make the full two years usable).

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


## 3. Method (v2; see plan Amendments 1-5b)

- Chronological; training targets strictly before each boundary (gap = one horizon). Runs: F1-F3 (regime A to A, descriptive, used for the safeguard), G1-G3 (regime-B-native, non-overlapping, headline pool), HOLDOUT (KAN-15 split, second headline group; trained on regime A only, no regime-B validation exists).
- Features (`experiments/dataset.py`, as-of origin-1 h): latest, cyclic target hour/day-of-week/month, station id, readings at target-24 (NaN at +24 by rule), -48, -72, -168 h, rolling mean/max/std over 3/6/24 h, trend 3/6 h. Weather and wind: **needs data**, not used. Set D (no neighbour features) in the fixed ML models.
- Candidates (fixed, no search): persistence; smoother latest + beta*(mean24 - latest), beta in {0, 0.1, ..., 0.9} chosen on same-regime validation (last 10 days of regime-B training for G folds, last 45 days for F folds), fixed 0.5 for the HOLDOUT; LightGBM direct and log-ratio (l1, 7 leaves, lr 0.05, 150 rounds, seed 20261029; informed by v1 which favoured small trees and l1); ridge on the residual (alpha 10); current KAN-64 boosters. For G folds ML rows are regime A+B with a `src_air4thai` flag (flag = 1 as live); B-only rows are a descriptive sensitivity. Descriptive extras outside the claim family: smoother with beta fixed 0.5, the beta curve, global multi-horizon LightGBM with and without station id.
- Statistics: paired MAE difference vs persistence on identical station-hours; bootstrap over target days with 1-day and 7-day blocks (seed 20261030, 40,000 draws). Family = 5 candidates x 3 horizons x 2 headline groups = 30 claims, Holm and Bonferroni (0.05/30) separately for the two block sizes. Per-run CIs (3-6 weeks) use 1-day blocks and are descriptive.
- Fits: no hyper-parameter search; in total 74 LightGBM refits and 30 ridge fits (counters in `results.json`), plus 10 beta values per run and horizon. The v1 search (3,344 configurations) is superseded.

## 4. Results

### 4a. Headline, +24 h (paired MAE difference vs persistence, 95% CI)

| Group (persistence MAE, n unique) | Candidate | MAE | 1-day blocks | 7-day blocks | Bonferroni(30), 7-day | Holm p (1d / 7d) | folds beating |
|---|---|---:|---|---|---|---|---:|
| HOLDOUT (4.498, 10,729; 91 days) | smoother (beta 0.5) | 4.114 | -0.383 [-0.526, -0.235] | [-0.465, -0.297] | [-0.507, -0.243] | 0.001 / 0.001 | n/a |
| | LightGBM direct | 4.090 | -0.408 [-0.542, -0.280] | [-0.556, -0.271] | [-0.652, -0.208] | 0.001 / 0.001 | n/a |
| | LightGBM log-ratio | 4.331 | -0.167 [-0.276, -0.065] | [-0.296, -0.053] | [-0.380, -0.000] | 0.007 / 0.012 | n/a |
| | ridge | 5.209 | +0.711 [+0.552, +0.867] | [+0.544, +0.865] | [+0.443, +0.946] | 0.001 / 0.001 | n/a |
| | current (KAN-64) | 4.503 | +0.005 [-0.096, +0.104] | [-0.060, +0.079] | [-0.089, +0.127] | 1.000 / 1.000 | n/a |
| Regime-B pool G1-G3 (4.993, 7,954; 67 days) | smoother (val beta) | 4.602 | -0.391 [-0.633, -0.139] | [-0.557, -0.195] | [-0.629, -0.057] | 0.015 / 0.005 | 3/3 |
| | LightGBM direct | 4.391 | -0.603 [-0.815, -0.400] | [-0.778, -0.421] | [-0.868, -0.317] | 0.001 / 0.001 | 3/3 |
| | LightGBM log-ratio | 4.487 | -0.507 [-0.680, -0.341] | [-0.659, -0.348] | [-0.741, -0.251] | 0.001 / 0.001 | 3/3 |
| | ridge | 5.038 | +0.044 [-0.145, +0.221] | [-0.227, +0.304] | [-0.389, +0.434] | 1.000 / 1.000 | 1/3 |
| | current (KAN-64) | 4.922 | -0.072 [-0.177, +0.033] | [-0.154, +0.005] | [-0.205, +0.049] | 0.532 / 0.208 | 3/3 |

Block size matters less than expected: 7-day-block CIs are not systematically wider than 1-day ones (13 blocks in the holdout, 10 in the regime-B pool; the percentile bootstrap is itself approximate with so few blocks). **Fragile point:** the smoother's Bonferroni(30) CI in the regime-B pool at +24 h is [-0.629, -0.057] with 7-day blocks (the decision block, pass) but [-0.760, +0.016] with 1-day blocks, which crosses zero (Holm p 0.015 / 0.005 is below 0.05 in both). So condition (b) is borderline for the smoother and robust for LightGBM direct (Bonferroni upper -0.286 / -0.317 for 1-day / 7-day). HOLDOUT and the pool are two views of the same ~90 days (the holdout trains on regime A only; the G folds train on A+B), so agreement between them is reassurance about transfer, not independent replication.

Other horizons (7-day-block Bonferroni(30) upper bound in brackets):

| Horizon | Group | smoother | LightGBM direct | LightGBM log-ratio |
|---|---|---|---|---|
| +12 h | HOLDOUT | -0.535 [-0.334] | +0.051 [+0.136] (not better) | +0.233 [+0.344] (worse) |
| +12 h | Regime-B pool | -0.588 [-0.257] | -0.539 [-0.378] | -0.603 [-0.425] |
| +6 h | HOLDOUT | -0.435 [-0.219] | +0.233 [+0.351] (worse) | +0.304 [+0.416] (worse) |
| +6 h | Regime-B pool | -0.455 [-0.146] | -0.313 [-0.213] | -0.421 [-0.324] |

ML models trained on regime A only do not beat persistence at +6/+12 on the holdout (they are worse); the smoother does at every horizon, and ML trained with regime-B rows (G folds) does too. Per run (+24 h, 1-day blocks, descriptive): smoother G1 -0.528 [-0.803, -0.303], G2 -0.369 [-0.764, +0.045], G3 -0.292 [-0.804, +0.250]; LightGBM direct G1 -0.429, G2 -0.498, G3 -0.844 (all CIs below zero). Persistence MAE by fold: G1 3.46, G2 4.45, G3 6.79 (October is the worst month).

### 4b. The one-parameter smoother (prominent evaluation)

- Form: forecast = latest + beta*(mean of the last 24 h - latest). It uses only the 24 h trailing mean (as-of origin-1 h), no fitted model, and beta = 0 is persistence.
- Beta procedure: grid {0, 0.1, ..., 0.9}, minimum validation MAE on same-regime data (ties to the smaller beta). Chosen beta (+24 h): G1 0.7, G2 0.8, G3 0.6; test-optimal (descriptive) 0.8, 0.6, 0.5. At +6/+12 chosen 0.5-0.9 vs oracle 0.4-0.9. HOLDOUT: fixed 0.5 (oracle 0.7). Regime-A folds choose 0 in all 9 fold-horizon cases (the series is already smooth).
- Beta curve, +24 h, difference vs persistence (7-day CI): flat optimum. Holdout: beta 0.3 -0.285, 0.5 -0.383, 0.6 -0.403, 0.7 -0.404 [-0.513, -0.282], 0.9 -0.349. Regime-B pool: 0.3 -0.287, 0.5 -0.377 [-0.505, -0.227], 0.6 -0.392, 0.7 -0.386, 0.9 -0.316 [-0.542, -0.041]. Any beta in 0.3-0.9 beats persistence in both groups; beta 0.5-0.7 is within 0.03 of the best. The loss from using a mis-set beta is small, which is why a fixed beta is low-risk.
- Stability: 3/3 regime-B folds, 1/1 holdout, all horizons. Beta fixed at 0.5 versus the validated beta: pool -0.377 vs -0.391.
- Where it does not help: regime A (any beta > 0 is worse: beta 0.5 +0.222 [+0.094, +0.356]); it is a statement about the live air4thai-type series, so beta must be estimated from air4thai-era data only.
- Safeguard and onsets: see 4c.

### 4c. High pollution (forecast-time) and spike onsets

Subset: hours whose latest reading (= persistence forecast) exceeds 37.5, +24 h (paired MAE diff vs persistence, 7-day 95% CI; PASS/FAIL/untestable as pre-declared):

| Candidate | HOLDOUT | Regime-B pool | Regime-A pool |
|---|---|---|---|
| smoother | untestable: -7.18 [-9.14, -5.17], n=75, 19 d | untestable: -8.62 [-12.42, -5.87], n=74, 18 d | PASS: 0.00 (beta = 0), n=1,633, 38 d |
| LightGBM direct | untestable: -7.29 [-9.42, -4.92] | untestable: -8.25 [-10.46, -6.34] | PASS: -1.44 [-2.44, -0.41] |
| LightGBM log-ratio | untestable: -6.88 [-8.52, -4.81] | untestable: -8.63 [-10.92, -6.91] | PASS: -1.20 [-2.04, -0.36] |
| ridge | untestable: -0.19 [-2.08, +1.97] | untestable: -3.51 [-4.17, -2.72] | PASS: -1.24 [-2.69, -0.00] |
| smoother, beta 0.5 in regime A | | | PASS: +0.20 [-0.53, +0.63] |

Where persistence forecasts a high level, the candidates are on point much better in regime B (as expected: a 24 h-later fall is common after a high reading), but with n = 74 hours in 18 days the day criterion (>= 10) is met but the hour criterion (>= 200) is not, so the pre-declared rule says **untestable**; the point estimates are shown, not used for pass or fail. The safeguard is therefore passed only where testable (regime A) and unverified for the live regime.

Spike onsets: target >= latest + 15 µg/m³ (defined in advance); a forecast catches an onset if it is >= latest + 7.5. +24 h:

| Group | n onsets (days) | Candidate | caught share [7-day CI] | false alarms among non-onsets | mean error at onset | onset MAE (persistence 21.0) | onset MAE diff [7-day CI] |
|---|---|---|---|---|---|---|---|
| HOLDOUT | 204 (35) | smoother | 0.039 [0.000, 0.086] | 0.002 | -19.2 | 19.2 | -1.79 [-2.34, -0.96] |
| | | LightGBM direct | 0.010 [0.000, 0.015] | 0.001 | -19.3 | 19.3 | -1.71 [-2.11, -1.17] |
| | | LightGBM log-ratio | 0.000 | 0.000 | -20.6 | 20.6 | -0.40 [-0.81, +0.23] |
| Regime-B pool | 199 (31) | smoother | 0.101 [0.039, 0.192] | 0.010 | -18.7 | 18.7 | -2.37 [-3.37, -1.20] |
| | | LightGBM direct | 0.040 [0.012, 0.073] | 0.003 | -18.7 | 18.7 | -2.32 [-2.83, -1.63] |
| | | LightGBM log-ratio | 0.000 | 0.000 | -20.2 | 20.2 | -0.86 [-1.23, -0.29] |

The onset metric is testable (>= 30 onsets) and the result is clear: **no candidate anticipates spikes** (at best 10% caught; persistence 0%). The mean error at onset stays near -19 to -21 µg/m³; candidates shave 2 µg/m³ off the onset MAE by moving slightly upward, at almost no false alarms. Spikes need information the series does not contain (weather, fires); see data proposals.

### 4d. Global multi-horizon model (Boss question, descriptive, fixed config)

Difference vs persistence (7-day-block 95% CI), global LightGBM with horizon as a feature, trained on stacked +6/+12/+24 rows, l1, 7 leaves, 150 rounds: HOLDOUT +24 h -0.022 [-0.080, +0.035] (no station id) and -0.021 [-0.079, +0.033] (+ id); regime-B pool +24 h -0.492 [-0.619, -0.361] and -0.490 [-0.612, -0.357], versus -0.603 for the per-horizon LightGBM direct. Station id adds nothing. Hypothesis check: sharing across horizons and stations costs a little relative to per-horizon boosters in regime B, and does not beat persistence when trained only on regime A. v1 comparison with the untuned current boosters (global better in regime B at all horizons) is in the appendix.

**Serving constraint (not changed):** `models/serving.py` accepts a multi-horizon bundle with one booster per horizon or a single-horizon bundle (`_parse_bundle`; exactly one of `horizons` or `horizon_hours`+`booster`). `ServingModel.predict(horizon, features)` indexes a booster by horizon and builds the input row from `feature_list` only; every booster's `feature_name()` must equal the manifest `feature_list`, and every name must exist in `features.FEATURE_NAMES` (currently `pm25_latest`, `pm25_same_hour_24h`, `cams_pm25`). A global model therefore **would need a loader change**: a bundle shape with one booster plus injection of a `horizon` value (and, if used, a station id) into the row. Independently, any of the new features (rolling windows, lags beyond 24 h, calendar, neighbours) requires extending `features/build_features` (shared with training) and the live as-of query for 7 days of history and all five stations; and a residual or log-ratio target needs the loader/predict path to add back persistence (it returns the booster output directly). A direct-target per-horizon LightGBM with the new features needs only the feature-builder extension, no loader change.


### 4e. Leakage audit

The audit and the leaky control were produced by the v1 harness (`run_all.py`, outputs in `out/kan65/`); v2 reuses the identical feature table (`dataset.py`), so they apply unchanged.


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

- H1 (hour-of-day lowers MAE): not supported as a standalone effect. v2 uses calendar features inside fixed models; the gain is already available from a calendar-free smoother.
- H2 (residual target better than direct): no. At +24 h direct beats log-ratio in both headline groups (-0.408 vs -0.167 holdout, -0.603 vs -0.507 pool); at +12 h log-ratio is better in the pool.
- H3 (neighbours help): not isolated in v2 (set D has no neighbour features and still beats persistence); EDA says neighbours are a noise-reducing duplicate.
- H4 (smoothing is a better level estimate): supported in the live regime; this is the main finding of the study (-0.38 µg/m³ at +24 h with one parameter).
- H-MA: undecided (section 1).

## 6. Limitations

- The regime-B evidence is ~67 test days from one season (Aug-Oct 2026); no haze season, 74 hours above 37.5; the safeguard is untestable there.
- HOLDOUT and the regime-B pool are not independent; the folds G1-G3 are consecutive windows of the same period.
- Fixed ML configurations were chosen with knowledge of v1 results (small trees, l1, set D); beta fixed 0.5 for the holdout was also informed by v1. The smoother being the a-priori headline candidate is informed by v1 as well. None of this used a test span for tuning in v2, but the garden of forking paths includes v1.
- Seasonal naive and the v1 val-selected family are dropped from v2; they are in the appendix numbers.
- 7-day blocks give 9-13 blocks per group; CIs are approximate for that reason.
- Calendar features use Bangkok time (UTC+7). Weather, wind and boundary-layer data were not used (needs data).

## 7. What surprised me

1. A single parameter and no model gives most of the gain; the regime-A history, 80% of the data, teaches nothing about it (the optimal beta there is 0).
2. The "24 vs 13 µg/m³" regime difference is seasonal; the real difference is smoothness (8.6x), and the openaq series is not exactly a trailing 24 h mean either.
3. ML models trained only on the smooth history transfer at +24 h (direct, -0.41) but not at +6/+12 (worse than persistence), while the smoother transfers at all horizons.
4. Spikes are essentially unforecastable from the series itself (<= 10% caught).

## 8. What I would do with two more weeks

1. Backfill raw hourly Air4Thai history (or confirm the openaq definition) to put all of the two years in one definition; rerun this harness unchanged.
2. Weather features (wind, boundary-layer height, rain; hourly archive keyed on station coordinates), fire hotspots (FIRMS) for the dry season, CAMS previous-run archive; they are the only plausible route to onset skill. Each needs ingestion with `available_at`.
3. Shadow-run the challengers below and score them live for 4-8 weeks, including a dry-season check as soon as it arrives.
4. Pinball-loss quantile models (q 0.5/0.9) for alerting, per-station beta, neighbour ablation, conformal intervals.

## 9. Recommendation and confidence

- **Shadow-run now, do not promote.** (1) Smoother (beta about 0.5-0.7, to be re-estimated on live air4thai-era data; the whole curve is flat within 0.03 over that range) at +6, +12 and +24 h, as a new baseline-style predictor next to persistence; it needs a `mean24` input from the feature builder (shared module), no loader or model change. (2) LightGBM direct at +24 h with set D features, trained on regime A+B rows with the source flag (flag = 1 live), as the second challenger; it needs the feature-builder extension (calendar, 48/72/168 h lags, rolling windows) and no loader change; the global model and the log-ratio target would need loader changes. Nothing is implemented in this PR.
- Promotion criterion for a later ticket: re-evaluate once >= 200 regime-B hours above 37.5 over >= 10 days exist (safeguard testable) and the live shadow scores confirm the sign.
- Confidence: high that persistence is beatable on the live-type feed by roughly 0.3-0.6 µg/m³ MAE at +24 h (8-12%) during Aug-Oct; moderate that this holds in the dry season; low for onset detection.

## Reproduce

```sh
PYTHONPATH=src python -m pm25_forecast.experiments.run_review \
  --database data/history.duckdb --out out/kan65_v2          # about 35 seconds
PYTHONPATH=src python -m pm25_forecast.experiments.make_tables_v2 \
  out/kan65_v2/results.json out/kan65_v2/tables.md
```

Seeds: model 20261029, bootstrap 20261030; versions, snapshot SHA-256 and runtime are in `out/kan65_v2/results.json`. Code: `src/pm25_forecast/experiments/` (`run_review.py`, `make_tables_v2.py`, shared `dataset.py`, `fitters.py`, `common.py`), tests `tests/test_kan65_experiments.py`. Full tables: `out/kan65_v2/tables.md`. v1 (superseded) harness `run_all.py`, `out/kan65/`. No production module, dashboard file, model registry or workflow was touched.

## Appendix: v1 headline numbers (superseded, kept for transparency)

v1 (overlapping folds, selection on cross-regime validation, 3,344 configurations, 99 claims, 1-day blocks): val-selected best at +24 h: HOLDOUT -0.151 [-0.217, -0.089] (Bonferroni [-0.272, -0.044]); regime-B pool (F4, R1-R3, overlapping) -0.436 [-0.713, -0.173] (Bonferroni [-0.953, +0.015]); v1 rule failed on (b) and on a target-conditioned (d) (+5.15 in 76 hours). v1 global vs the untuned current boosters: regime-B pooled -0.147 (+6), -0.385 (+12), -0.310 (+24). The v1 target-conditioned high-pollution result is withdrawn as a rule because it conditions on the outcome; v1 numbers are reproducible with `run_all.py`.
