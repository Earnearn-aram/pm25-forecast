# E05 smoothing experiments — KAN-83

Verdict: **NO WIN**. No variant beats the smoother once fragility and the 26-comparison check are applied (dev only).

Development only, frozen snapshot `a00cfc9edd4c3fb5f7b946e8871be9e5a88fe2bb8ef56b91d895101da41c0208`. All 26 preregistered variants produced valid, clean tracked runs; one moving-average variant was rerun after an optimization (27 original-grid runs total), followed by one tracked Bonferroni sensitivity run (28 total). They share 4,200 +24 h pairs across five stations and five weekly folds. Observations assume `available_at = ts + 1h`; this is not measured arrival time. The holdout was not accessed. These are unadjusted exploratory CIs, and selection among 26 variants makes them optimistic. No candidate has an interval entirely below zero after the 26-comparison Bonferroni sensitivity check described below.

Lowest +24 h dev MAE in each family (none establishes an advantage after correction):

- Moving average: 24 h, MAE 3.926; CI vs smoother [-0.099, 0.154], run `3a4d8f968d104fe4818703cc29b28adb`.
- EWMA: 6 h half-life, MAE 3.874; CI [-0.061, 0.060], run `d2e50e44ba8343598ab50e5e75fcada4`.
- Beta smoother: beta 0.7, MAE 3.829; CI [-0.109, 0.008], run `3f8a628c51014da4ac62ffbbe58245ee`.
- Same-hour climatology: 28 days blended equally with persistence, MAE 3.791; CI [-0.339, 0.205], run `444eecf8cc5d4dc492e157ee0fa51104`.

Beta 0.6 has MAE 3.837 and a 72 h CI [-0.066, -0.007] below zero against the fixed smoother, but it is **fragile**: the 24 h sensitivity CI [-0.070, 0.002] crosses zero (168 h: [-0.051, -0.007]). Its dev result is not evidence of holdout performance or promotion. The lowest-MAE beta 0.7 and climatology blend have intervals crossing zero. No variant beats the smoother after the 26-comparison correction. Spike metrics are undefined: 15 observed spikes, below the minimum of 50. Pure climatology and the 50/50 blend use observations from the last N days at the target UTC hour, respecting the serving lag. EWMA weights all available history by elapsed hours. Beta 0.5 is the fixed smoother and beta 1.0 matches the 24 h moving average on this snapshot.


## Review of provenance and uncertainty

The 3 h moving average was rerun after restricting the feature scan to recent observations to reduce runtime. Its original run `ce4e9829b0114e62a9bcff34c0c957b6` and rerun `2dcd664e144e4f37a984a623126a2f37` have identical full score and counter objects in `evaluation.json`; the rerun is the single row counted on the leaderboard. The shared fixed-smoother reference is E04 run `da9ad074c138430199b42f1c2ef23d21` (MAE 3.867). Beta 0.5 is a separate candidate that reproduces that baseline; the report chart uses the E04 run.

Protocol section 6 specifies 72-hour blocks as primary, with 24- and 168-hour sensitivities. The individual preregistrations did not explicitly name a block length, so all 72-hour results here are exploratory, not a candidate-specific preregistered claim.

Protocol section 8 does not require a multiple-comparison adjustment for dev exploration: dev CIs are descriptive and its Bonferroni rule is for up to two registered holdout finalists. As an additional post-hoc conservative sensitivity, we used a two-sided Bonferroni level of 1 - 0.05/26 = 99.8077% on the 72-hour CI. Beta 0.6 was the only candidate with an unadjusted 95% upper bound below zero; at the wider level its CI is [-0.081, 0.009] (tracked diagnostic run `e3fb0fa2e6dc4d4e9050a71208fa56ae`), so it no longer excludes zero. All other finite 95% intervals already included zero and a nested wider percentile interval cannot exclude it; the identical smoother has no CI. Thus no candidate beats the smoother under this sensitivity. This is not the protocol's holdout gate and does not unlock holdout scoring.

The post-run feature safeguards choose one deterministic value per observed UTC hour if duplicates occur and log a count of rows excluded for missing availability. The evaluated frozen snapshot has one observation per station-hour and no missing availability, so these safeguards do not change the recorded runs.

## E05 leaderboard rows

These rows use the same snapshot and are comparable on the common intersection. The generated full leaderboard also contains an unrelated fixture snapshot, which must not be compared with these results.

| experiment | run | mae_24h | rmse_24h | skill vs persistence | n_pairs | git_sha | snapshot hash | dirty | prereg |
|---|---|---|---|---|---|---|---|---|---|
| kan78/e05-clim-28d-blend | climatology-28d-blend-dev (444eecf8) | 3.791 | 5.288 | 0.121 | 4200 | ecd4bd3e5c | a00cfc9edd4c |  | e05-clim-28d-blend |
| kan78/e05-beta-07 | beta-0.7-dev (3f8a628c) | 3.829 | 5.126 | 0.112 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-07 |
| kan78/e05-beta-06 | beta-0.6-dev (1cc8391c) | 3.837 | 5.151 | 0.110 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-06 |
| kan78/e05-beta-08 | beta-0.8-dev (8e76cb0d) | 3.840 | 5.133 | 0.109 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-08 |
| kan78/e05-beta-05 | beta-0.5-dev (18ab9d50) | 3.867 | 5.206 | 0.103 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-05 |
| kan78/e05-beta-09 | beta-0.9-dev (bd8b8f1d) | 3.873 | 5.170 | 0.102 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-09 |
| kan78/e05-ewma-6h | ewma-6h-dev (d2e50e44) | 3.874 | 5.166 | 0.101 | 4200 | 8c8ce99f82 | a00cfc9edd4c |  | e05-ewma-6h |
| kan78/e05-clim-14d-blend | climatology-14d-blend-dev (90e2b7ff) | 3.874 | 5.393 | 0.101 | 4200 | ecd4bd3e5c | a00cfc9edd4c |  | e05-clim-14d-blend |
| kan78/e05-ewma-12h | ewma-12h-dev (1f8cb1e5) | 3.901 | 5.204 | 0.095 | 4200 | 8c8ce99f82 | a00cfc9edd4c |  | e05-ewma-12h |
| kan78/e05-clim-7d-blend | climatology-7d-blend-dev (0abc6815) | 3.902 | 5.405 | 0.095 | 4200 | ecd4bd3e5c | a00cfc9edd4c |  | e05-clim-7d-blend |
| kan78/e05-beta-04 | beta-0.4-dev (bd2be67d) | 3.918 | 5.291 | 0.091 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-04 |
| kan78/e05-beta-10 | beta-1.0-dev (72e99bfd) | 3.926 | 5.238 | 0.089 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-10 |
| kan78/e05-ma-24h | ma-24h-dev (3a4d8f96) | 3.926 | 5.238 | 0.089 | 4200 | 3b16e011b9 | a00cfc9edd4c |  | e05-ma-24h |
| kan78/e05-ewma-2h | ewma-2h-dev (208d1c5e) | 3.941 | 5.323 | 0.086 | 4200 | 8c8ce99f82 | a00cfc9edd4c |  | e05-ewma-2h |
| kan78/e05-beta-03 | beta-0.3-dev (666ec63c) | 3.989 | 5.404 | 0.075 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-03 |
| kan78/e05-ma-12h | ma-12h-dev (b5fd1892) | 3.991 | 5.312 | 0.074 | 4200 | 3b16e011b9 | a00cfc9edd4c |  | e05-ma-12h |
| kan78/e05-ewma-24h | ewma-24h-dev (f740dfd6) | 4.016 | 5.387 | 0.068 | 4200 | 8c8ce99f82 | a00cfc9edd4c |  | e05-ewma-24h |
| kan78/e05-ma-6h | ma-6h-dev (25e86502) | 4.046 | 5.450 | 0.062 | 4200 | 3b16e011b9 | a00cfc9edd4c |  | e05-ma-6h |
| kan78/e05-ma-48h | ma-48h-dev (6deab3b4) | 4.064 | 5.397 | 0.057 | 4200 | 3b16e011b9 | a00cfc9edd4c |  | e05-ma-48h |
| kan78/e05-beta-02 | beta-0.2-dev (c4df6500) | 4.080 | 5.544 | 0.054 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-02 |
| kan78/e05-ma-3h | ma-3h-dev (2dcd664e) | 4.098 | 5.554 | 0.050 | 4200 | 3b16e011b9 | a00cfc9edd4c |  | e05-ma-3h |
| kan78/e05-beta-01 | beta-0.1-dev (be67b794) | 4.187 | 5.709 | 0.029 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-01 |
| kan78/e05-beta-00 | beta-0.0-dev (98317dd7) | 4.311 | 5.896 | 0.000 | 4200 | 6cebb58acf | a00cfc9edd4c |  | e05-beta-00 |
| kan78/e05-clim-28d-pure | climatology-28d-pure-dev (52a84acc) | 4.506 | 6.177 | -0.045 | 4200 | ecd4bd3e5c | a00cfc9edd4c |  | e05-clim-28d-pure |
| kan78/e05-clim-14d-pure | climatology-14d-pure-dev (79a142e2) | 4.665 | 6.407 | -0.082 | 4200 | ecd4bd3e5c | a00cfc9edd4c |  | e05-clim-14d-pure |
| kan78/e05-clim-7d-pure | climatology-7d-pure-dev (8a0fa166) | 4.698 | 6.391 | -0.090 | 4200 | ecd4bd3e5c | a00cfc9edd4c |  | e05-clim-7d-pure |
