## Run info

- snapshot SHA-256 `fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09` matches KAN-15: True
- python 3.12.12, numpy 2.5.4, scipy 1.18.1, lightgbm 4.7.0, duckdb 2.0.0; bootstrap seed 20261030, model seed 20261029
- fits: {'lgbm': 1760, 'ridge': 1608, 'lgbm_refit': 89}; claims (Bonferroni family): 99; runtime 12.6 min

### KAN-15 holdout (headline), +24 h (persistence MAE 4.498, n=10729)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 4.347 | 6.299 | +0.11 | -0.151 [-0.217, -0.089] | [-0.272, -0.044] | 0.002 | 1/1 |
| seasonal naive | 5.033 | 7.142 | -0.08 | +0.518 [+0.144, +0.939] | [-0.106, +1.332] | 0.170 | 0/1 |
| current LightGBM (latest + t-24h) | 4.503 | 6.413 | +1.07 | +0.005 [-0.096, +0.103] | [-0.173, +0.175] | 1.000 | 0/1 |
| smoother (1-param shrink / rolling) | 4.212 | 6.061 | -0.04 | -0.285 [-0.368, -0.200] | [-0.435, -0.137] | 0.002 | 1/1 |
| ridge | 4.551 | 6.572 | -0.42 | +0.053 [-0.064, +0.161] | [-0.160, +0.235] | 1.000 | 0/1 |
| LightGBM direct | 4.036 | 5.715 | +0.07 | -0.462 [-0.615, -0.318] | [-0.748, -0.215] | 0.002 | 1/1 |
| LightGBM residual/log-ratio | 4.347 | 6.299 | +0.11 | -0.151 [-0.217, -0.089] | [-0.272, -0.044] | 0.002 | 1/1 |
| ensemble ridge+LGBM-resid | 4.402 | 6.372 | -0.15 | -0.096 [-0.184, -0.018] | [-0.262, +0.031] | 0.364 | 1/1 |
| shrink LGBM-resid to persistence | 4.376 | 6.353 | +0.08 | -0.121 [-0.171, -0.075] | [-0.213, -0.041] | 0.002 | 1/1 |
| global LGBM (no id) | 4.551 | 6.573 | +0.05 | +0.053 [+0.018, +0.087] | [-0.013, +0.109] | 0.129 | 0/1 |
| global LGBM (+station id) | 4.590 | 6.618 | +0.15 | +0.092 [+0.050, +0.133] | [+0.016, +0.162] | 0.004 | 0/1 |

### KAN-15 holdout (headline), +12 h (persistence MAE 4.394, n=10774)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 4.450 | 6.129 | -0.05 | +0.056 [-0.067, +0.159] | [-0.176, +0.220] | 1.000 | 0/1 |
| seasonal naive | 4.457 | 6.487 | -0.03 | +0.047 [-0.217, +0.320] | [-0.411, +0.531] | 1.000 | 0/1 |
| current LightGBM (latest + t-24h) | 4.783 | 6.680 | +0.94 | +0.389 [+0.299, +0.475] | [+0.225, +0.533] | 0.002 | 0/1 |
| smoother (1-param shrink / rolling) | 4.394 | 6.220 | -0.02 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | 1.000 | 0/1 |
| ridge | 4.809 | 6.947 | -0.18 | +0.416 [+0.318, +0.523] | [+0.253, +0.618] | 0.002 | 0/1 |
| LightGBM direct | 4.448 | 6.236 | +0.50 | +0.054 [-0.015, +0.118] | [-0.072, +0.162] | 1.000 | 0/1 |
| LightGBM residual/log-ratio | 4.450 | 6.129 | -0.05 | +0.056 [-0.067, +0.159] | [-0.176, +0.220] | 1.000 | 0/1 |
| ensemble ridge+LGBM-resid | 4.582 | 6.426 | -0.11 | +0.189 [+0.103, +0.271] | [+0.037, +0.329] | 0.004 | 0/1 |
| shrink LGBM-resid to persistence | 4.399 | 6.088 | -0.04 | +0.005 [-0.090, +0.084] | [-0.177, +0.129] | 1.000 | 0/1 |
| global LGBM (no id) | 4.551 | 6.414 | +0.08 | +0.157 [+0.124, +0.189] | [+0.097, +0.213] | 0.002 | 0/1 |
| global LGBM (+station id) | 4.576 | 6.435 | +0.09 | +0.183 [+0.144, +0.220] | [+0.111, +0.247] | 0.002 | 0/1 |

### KAN-15 holdout (headline), +6 h (persistence MAE 3.908, n=10799)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 4.064 | 5.716 | -0.04 | +0.156 [+0.094, +0.214] | [+0.038, +0.259] | 0.002 | 0/1 |
| seasonal naive | 4.458 | 6.488 | -0.03 | +0.525 [+0.206, +0.869] | [-0.033, +1.171] | 0.034 | 0/1 |
| current LightGBM (latest + t-24h) | 4.094 | 5.907 | +0.44 | +0.186 [+0.136, +0.239] | [+0.100, +0.282] | 0.002 | 0/1 |
| smoother (1-param shrink / rolling) | 3.908 | 5.649 | -0.01 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | 1.000 | 0/1 |
| ridge | 4.103 | 6.000 | -0.25 | +0.195 [+0.136, +0.259] | [+0.093, +0.318] | 0.002 | 0/1 |
| LightGBM direct | 4.026 | 5.782 | +0.39 | +0.118 [+0.060, +0.181] | [+0.020, +0.237] | 0.002 | 0/1 |
| LightGBM residual/log-ratio | 4.064 | 5.716 | -0.04 | +0.156 [+0.094, +0.214] | [+0.038, +0.259] | 0.002 | 0/1 |
| ensemble ridge+LGBM-resid | 4.050 | 5.782 | -0.15 | +0.141 [+0.090, +0.191] | [+0.046, +0.231] | 0.002 | 0/1 |
| shrink LGBM-resid to persistence | 3.996 | 5.652 | -0.03 | +0.088 [+0.040, +0.132] | [-0.001, +0.165] | 0.028 | 0/1 |
| global LGBM (no id) | 4.082 | 5.853 | +0.08 | +0.174 [+0.144, +0.204] | [+0.122, +0.227] | 0.002 | 0/1 |
| global LGBM (+station id) | 4.113 | 5.875 | +0.09 | +0.204 [+0.170, +0.240] | [+0.145, +0.267] | 0.002 | 0/1 |

### Pooled regime-B folds (F4, R1, R2, R3), +24 h (persistence MAE 4.482, n=11794)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 4.046 | 5.816 | -1.15 | -0.436 [-0.713, -0.173] | [-0.953, +0.015] | 0.034 | 4/4 |
| seasonal naive | 4.912 | 6.979 | -0.08 | +0.428 [+0.063, +0.841] | [-0.191, +1.222] | 0.578 | 0/4 |
| current LightGBM (latest + t-24h) | 4.458 | 6.334 | +0.94 | -0.024 [-0.121, +0.071] | [-0.204, +0.144] | 1.000 | 2/4 |
| smoother (1-param shrink / rolling) | 4.085 | 5.877 | -0.09 | -0.398 [-0.560, -0.235] | [-0.689, -0.105] | 0.004 | 4/4 |
| ridge | 4.175 | 5.851 | -0.96 | -0.307 [-0.589, -0.042] | [-0.822, +0.140] | 0.578 | 3/4 |
| LightGBM direct | 3.916 | 5.543 | -0.62 | -0.566 [-0.828, -0.320] | [-1.060, -0.132] | 0.002 | 4/4 |
| LightGBM residual/log-ratio | 4.042 | 5.792 | -0.78 | -0.440 [-0.697, -0.188] | [-0.915, -0.008] | 0.023 | 4/4 |
| ensemble ridge+LGBM-resid | 4.055 | 5.752 | -0.87 | -0.427 [-0.693, -0.173] | [-0.923, +0.002] | 0.028 | 4/4 |
| shrink LGBM-resid to persistence | 4.041 | 5.796 | -0.60 | -0.441 [-0.640, -0.248] | [-0.811, -0.113] | 0.002 | 4/4 |
| global LGBM (no id) | 4.148 | 5.736 | +0.00 | -0.334 [-0.544, -0.136] | [-0.744, +0.020] | 0.041 | 3/4 |
| global LGBM (+station id) | 4.120 | 5.710 | -0.23 | -0.362 [-0.579, -0.157] | [-0.782, -0.003] | 0.024 | 3/4 |

### Pooled regime-B folds (F4, R1, R2, R3), +12 h (persistence MAE 4.436, n=11839)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 4.122 | 5.794 | -0.85 | -0.314 [-0.521, -0.118] | [-0.702, +0.013] | 0.045 | 3/4 |
| seasonal naive | 4.437 | 6.405 | -0.05 | -0.000 [-0.276, +0.285] | [-0.475, +0.503] | 1.000 | 2/4 |
| current LightGBM (latest + t-24h) | 4.521 | 6.315 | +0.57 | +0.085 [-0.013, +0.182] | [-0.095, +0.253] | 1.000 | 2/4 |
| smoother (1-param shrink / rolling) | 4.108 | 5.676 | -0.06 | -0.328 [-0.472, -0.174] | [-0.582, -0.048] | 0.002 | 3/4 |
| ridge | 4.177 | 5.838 | -0.73 | -0.259 [-0.489, -0.038] | [-0.669, +0.111] | 0.578 | 3/4 |
| LightGBM direct | 3.980 | 5.519 | -0.35 | -0.456 [-0.643, -0.283] | [-0.807, -0.169] | 0.002 | 4/4 |
| LightGBM residual/log-ratio | 4.160 | 5.835 | -0.64 | -0.276 [-0.442, -0.111] | [-0.580, +0.009] | 0.037 | 3/4 |
| ensemble ridge+LGBM-resid | 4.104 | 5.749 | -0.68 | -0.332 [-0.531, -0.142] | [-0.686, -0.011] | 0.015 | 3/4 |
| shrink LGBM-resid to persistence | 4.134 | 5.793 | -0.49 | -0.302 [-0.429, -0.176] | [-0.530, -0.088] | 0.002 | 3/4 |
| global LGBM (no id) | 4.135 | 5.614 | -0.00 | -0.301 [-0.480, -0.123] | [-0.622, +0.016] | 0.039 | 3/4 |
| global LGBM (+station id) | 4.139 | 5.633 | -0.22 | -0.297 [-0.472, -0.124] | [-0.610, +0.006] | 0.027 | 3/4 |

### Pooled regime-B folds (F4, R1, R2, R3), +6 h (persistence MAE 3.944, n=11864)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 3.714 | 5.228 | -0.32 | -0.230 [-0.347, -0.113] | [-0.445, -0.030] | 0.002 | 3/4 |
| seasonal naive | 4.423 | 6.394 | -0.04 | +0.478 [+0.153, +0.828] | [-0.061, +1.146] | 0.104 | 0/4 |
| current LightGBM (latest + t-24h) | 3.948 | 5.657 | +0.22 | +0.004 [-0.059, +0.067] | [-0.112, +0.114] | 1.000 | 2/4 |
| smoother (1-param shrink / rolling) | 3.748 | 5.247 | -0.06 | -0.196 [-0.316, -0.072] | [-0.406, +0.036] | 0.104 | 2/4 |
| ridge | 3.829 | 5.428 | -0.79 | -0.114 [-0.264, +0.034] | [-0.382, +0.136] | 1.000 | 2/4 |
| LightGBM direct | 3.764 | 5.285 | -0.37 | -0.180 [-0.311, -0.050] | [-0.416, +0.047] | 0.205 | 3/4 |
| LightGBM residual/log-ratio | 3.694 | 5.231 | -0.36 | -0.250 [-0.368, -0.133] | [-0.461, -0.052] | 0.004 | 3/4 |
| ensemble ridge+LGBM-resid | 3.707 | 5.255 | -0.57 | -0.237 [-0.369, -0.106] | [-0.472, -0.017] | 0.008 | 3/4 |
| shrink LGBM-resid to persistence | 3.675 | 5.207 | -0.27 | -0.269 [-0.362, -0.177] | [-0.435, -0.113] | 0.002 | 3/4 |
| global LGBM (no id) | 3.801 | 5.223 | -0.00 | -0.143 [-0.280, -0.005] | [-0.390, +0.104] | 1.000 | 3/4 |
| global LGBM (+station id) | 3.787 | 5.219 | -0.23 | -0.157 [-0.294, -0.019] | [-0.404, +0.081] | 0.617 | 3/4 |

### Pooled regime-A folds (F1, F2, F3), +24 h (persistence MAE 4.357, n=12776)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 4.016 | 5.825 | +0.48 | -0.341 [-0.733, +0.038] | [-1.041, +0.315] | 1.000 | 3/3 |
| seasonal naive | 5.944 | 8.446 | +0.16 | +1.552 [+0.878, +2.289] | [+0.391, +2.928] | 0.002 | 0/3 |
| current LightGBM (latest + t-24h) | 4.363 | 6.169 | +0.54 | +0.006 [-0.174, +0.179] | [-0.332, +0.305] | 1.000 | 1/3 |
| smoother (1-param shrink / rolling) | 4.357 | 6.395 | +0.15 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | 1.000 | 0/3 |
| ridge | 3.842 | 5.660 | -0.05 | -0.515 [-0.795, -0.251] | [-1.035, -0.059] | 0.008 | 3/3 |
| LightGBM direct | 4.132 | 6.009 | +1.05 | -0.225 [-0.637, +0.182] | [-0.952, +0.502] | 1.000 | 2/3 |
| LightGBM residual/log-ratio | 4.060 | 5.904 | +0.09 | -0.297 [-0.701, +0.094] | [-1.020, +0.389] | 1.000 | 3/3 |
| ensemble ridge+LGBM-resid | 3.851 | 5.640 | +0.02 | -0.506 [-0.834, -0.189] | [-1.091, +0.047] | 0.058 | 3/3 |
| shrink LGBM-resid to persistence | 3.913 | 5.768 | +0.11 | -0.444 [-0.754, -0.146] | [-1.001, +0.080] | 0.129 | 3/3 |
| global LGBM (no id) | 3.956 | 5.915 | +0.20 | -0.401 [-0.601, -0.207] | [-0.771, -0.052] | 0.004 | 3/3 |
| global LGBM (+station id) | 3.957 | 5.933 | +0.00 | -0.400 [-0.598, -0.211] | [-0.779, -0.064] | 0.004 | 3/3 |

### Pooled regime-A folds (F1, F2, F3), +12 h (persistence MAE 2.730, n=12865)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 2.410 | 3.546 | -0.00 | -0.320 [-0.513, -0.136] | [-0.669, +0.003] | 0.027 | 2/3 |
| seasonal naive | 4.259 | 6.272 | +0.10 | +1.518 [+1.210, +1.861] | [+1.008, +2.171] | 0.002 | 0/3 |
| current LightGBM (latest + t-24h) | 2.627 | 3.937 | +0.30 | -0.103 [-0.264, +0.049] | [-0.402, +0.168] | 1.000 | 1/3 |
| smoother (1-param shrink / rolling) | 2.730 | 4.102 | +0.08 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | 1.000 | 0/3 |
| ridge | 2.376 | 3.450 | -0.12 | -0.353 [-0.547, -0.176] | [-0.714, -0.044] | 0.004 | 2/3 |
| LightGBM direct | 2.377 | 3.491 | +0.11 | -0.353 [-0.547, -0.170] | [-0.706, -0.032] | 0.009 | 2/3 |
| LightGBM residual/log-ratio | 2.406 | 3.542 | +0.04 | -0.324 [-0.517, -0.142] | [-0.667, -0.005] | 0.024 | 3/3 |
| ensemble ridge+LGBM-resid | 2.316 | 3.369 | -0.04 | -0.413 [-0.613, -0.231] | [-0.777, -0.095] | 0.002 | 3/3 |
| shrink LGBM-resid to persistence | 2.361 | 3.531 | +0.05 | -0.369 [-0.523, -0.223] | [-0.639, -0.110] | 0.002 | 3/3 |
| global LGBM (no id) | 2.388 | 3.633 | +0.16 | -0.342 [-0.493, -0.198] | [-0.611, -0.079] | 0.002 | 3/3 |
| global LGBM (+station id) | 2.430 | 3.666 | -0.12 | -0.300 [-0.450, -0.152] | [-0.565, -0.033] | 0.002 | 3/3 |

### Pooled regime-A folds (F1, F2, F3), +6 h (persistence MAE 1.657, n=12922)

| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | Bonferroni CI | Holm p | runs beating persistence |
|---|---:|---:|---:|---|---|---:|---:|
| val-selected best (PRIMARY) | 1.397 | 2.071 | -0.02 | -0.260 [-0.354, -0.172] | [-0.433, -0.110] | 0.002 | 3/3 |
| seasonal naive | 4.292 | 6.337 | +0.05 | +2.627 [+2.155, +3.152] | [+1.840, +3.615] | 0.002 | 0/3 |
| current LightGBM (latest + t-24h) | 1.608 | 2.409 | +0.18 | -0.049 [-0.136, +0.034] | [-0.205, +0.103] | 1.000 | 2/3 |
| smoother (1-param shrink / rolling) | 1.657 | 2.527 | +0.05 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | 1.000 | 0/3 |
| ridge | 1.445 | 2.152 | -0.02 | -0.212 [-0.294, -0.135] | [-0.365, -0.082] | 0.002 | 3/3 |
| LightGBM direct | 1.450 | 2.159 | +0.14 | -0.207 [-0.299, -0.122] | [-0.374, -0.056] | 0.002 | 2/3 |
| LightGBM residual/log-ratio | 1.394 | 2.070 | +0.02 | -0.263 [-0.356, -0.175] | [-0.435, -0.113] | 0.002 | 3/3 |
| ensemble ridge+LGBM-resid | 1.390 | 2.071 | +0.00 | -0.267 [-0.354, -0.186] | [-0.434, -0.129] | 0.002 | 3/3 |
| shrink LGBM-resid to persistence | 1.400 | 2.112 | +0.03 | -0.257 [-0.331, -0.186] | [-0.392, -0.135] | 0.002 | 3/3 |
| global LGBM (no id) | 1.454 | 2.189 | +0.07 | -0.203 [-0.300, -0.113] | [-0.378, -0.047] | 0.002 | 2/3 |
| global LGBM (+station id) | 1.504 | 2.222 | -0.09 | -0.153 [-0.262, -0.049] | [-0.353, +0.029] | 0.136 | 1/3 |

### Per run, +24 h, paired diff vs persistence (95% day-block CI)

| Candidate | F1 | F2 | F3 | F4 | R1 | R2 | R3 | HOLDOUT |
|---|---|---|---|---|---|---|---|---|
| persistence MAE | 6.97 | 3.37 | 1.72 | 3.28 | 3.94 | 6.26 | 6.20 | 4.50 |
| test pairs | 4922 | 4765 | 3089 | 5040 | 2520 | 2520 | 1714 | 10729 |
| val-selected best (PRIMARY) | -0.649 [-1.538, +0.258] | -0.236 [-0.627, +0.176] | -0.013 [-0.196, +0.164] | -0.162 [-0.264, -0.066] | -0.434 [-1.045, +0.161] | -0.534 [-1.236, +0.200] | -1.103 [-2.226, -0.162] | -0.151 [-0.216, -0.090] |
| seasonal naive | +2.739 [+1.093, +4.482] | +1.107 [+0.600, +1.641] | +0.323 [-0.011, +0.691] | +0.035 [-0.217, +0.280] | +0.245 [-0.304, +0.851] | +0.661 [-0.303, +1.629] | +1.508 [+0.086, +3.333] | +0.518 [+0.143, +0.939] |
| current LightGBM (latest + t-24h) | -0.310 [-0.681, +0.037] | +0.113 [-0.092, +0.313] | +0.347 [+0.075, +0.609] | +0.033 [-0.095, +0.161] | +0.022 [-0.149, +0.188] | -0.093 [-0.330, +0.131] | -0.159 [-0.306, -0.007] | +0.005 [-0.095, +0.103] |
| smoother (1-param shrink / rolling) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.387 [-0.491, -0.296] | -0.649 [-0.945, -0.369] | -0.385 [-0.914, +0.128] | -0.076 [-0.475, +0.370] | -0.285 [-0.368, -0.201] |
| ridge | -0.900 [-1.518, -0.301] | -0.442 [-0.747, -0.144] | -0.013 [-0.196, +0.164] | +0.140 [+0.041, +0.232] | -0.434 [-1.045, +0.161] | -0.712 [-1.452, +0.056] | -0.842 [-1.949, +0.078] | +0.053 [-0.060, +0.162] |
| LightGBM direct | -0.431 [-1.389, +0.542] | -0.236 [-0.627, +0.176] | +0.120 [-0.103, +0.341] | -0.307 [-0.457, -0.169] | -0.530 [-1.230, +0.156] | -0.755 [-1.167, -0.342] | -1.103 [-2.226, -0.162] | -0.462 [-0.615, -0.319] |
| LightGBM residual/log-ratio | -0.649 [-1.538, +0.258] | -0.100 [-0.569, +0.373] | -0.041 [-0.218, +0.136] | -0.162 [-0.264, -0.066] | -0.532 [-1.101, +0.037] | -0.534 [-1.236, +0.200] | -0.984 [-1.924, -0.134] | -0.151 [-0.216, -0.090] |
| ensemble ridge+LGBM-resid | -0.937 [-1.650, -0.219] | -0.345 [-0.730, +0.038] | -0.065 [-0.233, +0.102] | -0.049 [-0.130, +0.021] | -0.514 [-1.107, +0.068] | -0.713 [-1.425, +0.033] | -0.991 [-2.006, -0.123] | -0.096 [-0.182, -0.017] |
| shrink LGBM-resid to persistence | -0.822 [-1.491, -0.134] | -0.296 [-0.662, +0.071] | -0.069 [-0.201, +0.066] | -0.138 [-0.214, -0.067] | -0.608 [-1.031, -0.180] | -0.549 [-1.095, +0.011] | -0.931 [-1.636, -0.292] | -0.121 [-0.171, -0.076] |
| global LGBM (no id) | -0.684 [-1.098, -0.275] | -0.356 [-0.631, -0.069] | -0.019 [-0.142, +0.105] | +0.036 [-0.068, +0.129] | -0.768 [-1.115, -0.421] | -0.397 [-0.899, +0.106] | -0.695 [-1.644, +0.161] | +0.053 [+0.018, +0.087] |
| global LGBM (+station id) | -0.669 [-1.038, -0.315] | -0.381 [-0.718, -0.062] | -0.000 [-0.136, +0.136] | +0.038 [-0.047, +0.115] | -0.745 [-1.161, -0.324] | -0.455 [-1.020, +0.083] | -0.840 [-1.753, -0.037] | +0.092 [+0.050, +0.133] |

### High-pollution safeguard (rule d), +24 h: paired MAE diff vs persistence (point, n)

| Candidate | HOLDOUT target>37.5 | HOLDOUT latest>37.5 | REG-B target>37.5 | REG-B latest>37.5 | REG-A target>37.5 | REG-A latest>37.5 |
|---|---|---|---|---|---|---|
| val-selected best (PRIMARY) | -0.43 [-1.1, +0.2] n=75 | -1.76 [-2.2, -1.2] n=75 | +5.15 [+3.2, +7.0] n=76 | -13.51 [-16.9, -8.6] n=75 | -1.31 [-2.5, -0.0] n=1621 | -1.66 [-3.2, -0.2] n=1633 |
| seasonal naive | +0.88 [-4.8, +6.5] n=75 | -12.87 [-19.8, -6.3] n=75 | +0.92 [-4.6, +6.5] n=76 | -12.87 [-19.8, -6.3] n=75 | +3.08 [+1.0, +5.0] n=1604 | +1.95 [-0.8, +4.5] n=1619 |
| current LightGBM (latest + t-24h) | -0.65 [-0.9, -0.4] n=75 | -2.67 [-3.1, -2.1] n=75 | -0.47 [-0.7, -0.2] n=76 | -2.78 [-3.2, -2.2] n=75 | +0.41 [-0.2, +0.9] n=1621 | -0.67 [-1.7, +0.3] n=1633 |
| smoother (1-param shrink / rolling) | +0.46 [-0.6, +1.5] n=75 | -4.43 [-5.7, -3.4] n=75 | +1.23 [-0.7, +3.2] n=76 | -7.74 [-11.8, -4.5] n=75 | +0.00 [+0.0, +0.0] n=1621 | +0.00 [+0.0, +0.0] n=1633 |
| ridge | -0.57 [-1.8, +0.9] n=75 | -3.17 [-6.2, -0.7] n=75 | +4.80 [+2.6, +6.9] n=76 | -15.08 [-19.1, -9.5] n=75 | -0.59 [-1.4, +0.1] n=1621 | -1.21 [-2.4, -0.1] n=1633 |
| LightGBM direct | +0.46 [-1.2, +2.1] n=75 | -9.71 [-11.6, -7.2] n=75 | +2.84 [+0.2, +5.9] n=76 | -11.26 [-15.2, -6.4] n=75 | -1.53 [-2.9, -0.1] n=1621 | -1.22 [-2.6, +0.1] n=1633 |
| LightGBM residual/log-ratio | -0.43 [-1.1, +0.2] n=75 | -1.76 [-2.2, -1.2] n=75 | +3.80 [+1.1, +5.7] n=76 | -12.66 [-15.4, -8.0] n=75 | -1.30 [-2.5, +0.0] n=1621 | -1.68 [-3.2, -0.2] n=1633 |
| ensemble ridge+LGBM-resid | -0.51 [-1.2, +0.4] n=75 | -2.55 [-4.3, -1.2] n=75 | +4.30 [+2.1, +6.2] n=76 | -14.00 [-17.3, -8.9] n=75 | -1.14 [-2.1, -0.2] n=1621 | -1.59 [-2.9, -0.4] n=1633 |
| shrink LGBM-resid to persistence | -0.33 [-0.8, +0.1] n=75 | -1.33 [-1.6, -0.9] n=75 | +2.84 [+0.9, +4.2] n=76 | -10.13 [-12.0, -7.1] n=75 | -1.22 [-2.2, -0.2] n=1621 | -1.47 [-2.6, -0.3] n=1633 |
| global LGBM (no id) | -0.08 [-0.4, +0.3] n=75 | -0.57 [-1.0, -0.3] n=75 | +2.30 [+0.4, +4.5] n=76 | -12.96 [-16.3, -8.4] n=75 | -0.67 [-1.3, +0.0] n=1621 | -0.69 [-1.4, -0.0] n=1633 |
| global LGBM (+station id) | -0.29 [-0.7, +0.1] n=75 | -0.08 [-0.6, +0.2] n=75 | +2.27 [+0.4, +4.4] n=76 | -13.06 [-16.5, -8.5] n=75 | -0.61 [-1.2, -0.0] n=1621 | -0.66 [-1.3, -0.1] n=1633 |

### Bands and bias, +24 h, primary candidate vs persistence (MAE / bias, n)

| Group | band | n | persistence MAE | primary MAE | persistence bias | primary bias |
|---|---|---:|---:|---:|---:|---:|
| HOLDOUT | le15 | 7285 | 3.48 | 3.40 | +1.49 | +1.70 |
| HOLDOUT | 15_37.5 | 3369 | 6.28 | 6.00 | -2.82 | -2.83 |
| HOLDOUT | gt37.5 | 75 | 22.88 | 22.45 | -22.83 | -22.45 |
| REGIME_B | le15 | 7999 | 3.52 | 2.86 | +1.50 | +1.03 |
| REGIME_B | 15_37.5 | 3719 | 6.18 | 6.11 | -2.94 | -5.29 |
| REGIME_B | gt37.5 | 76 | 23.06 | 28.21 | -23.02 | -28.21 |
| REGIME_A | le15 | 3269 | 1.88 | 1.99 | +0.58 | +0.86 |
| REGIME_A | 15_37.5 | 7886 | 4.47 | 4.14 | +0.82 | +1.07 |
| REGIME_A | gt37.5 | 1621 | 8.82 | 7.51 | -3.97 | -3.20 |

### Breakdown at +24 h (MAE; persistence vs primary vs smoother)

**HOLDOUT by station**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 02t | 2145 | 3.65 | 3.49 | 3.51 | 3.34 |
| 59t | 2146 | 3.44 | 3.29 | 3.29 | 3.14 |
| 61t | 2146 | 3.60 | 3.44 | 3.46 | 3.29 |
| bkp100t | 2146 | 5.95 | 5.80 | 5.49 | 5.26 |
| bkp101t | 2146 | 5.85 | 5.71 | 5.31 | 5.15 |

**HOLDOUT by hour_group_local**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 0 | 2700 | 4.64 | 4.52 | 4.33 | 4.13 |
| 1 | 2689 | 4.68 | 4.50 | 4.36 | 4.15 |
| 2 | 2675 | 4.14 | 3.99 | 3.87 | 3.72 |
| 3 | 2665 | 4.53 | 4.38 | 4.28 | 4.14 |

**HOLDOUT by month**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 2026-07 | 100 | 2.92 | 3.04 | 2.79 | 3.12 |
| 2026-08 | 3720 | 2.99 | 2.93 | 2.74 | 2.74 |
| 2026-09 | 3600 | 3.94 | 3.80 | 3.64 | 3.58 |
| 2026-10 | 3309 | 6.85 | 6.57 | 6.54 | 6.01 |

**HOLDOUT by source**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 2.92 | 3.04 | 2.79 | 3.12 |
| 2 | 10629 | 4.51 | 4.36 | 4.23 | 4.04 |

**REGIME_B by station**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 02t | 2358 | 3.64 | 3.27 | 3.44 | 3.20 |
| 59t | 2359 | 3.43 | 3.12 | 3.24 | 3.08 |
| 61t | 2359 | 3.59 | 3.26 | 3.40 | 3.18 |
| bkp100t | 2359 | 5.86 | 5.33 | 5.26 | 5.10 |
| bkp101t | 2359 | 5.88 | 5.25 | 5.09 | 5.02 |

**REGIME_B by hour_group_local**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 0 | 2950 | 4.53 | 3.98 | 4.19 | 3.90 |
| 1 | 2954 | 4.62 | 4.22 | 4.23 | 4.01 |
| 2 | 2945 | 4.21 | 3.76 | 3.77 | 3.71 |
| 3 | 2945 | 4.57 | 4.22 | 4.15 | 4.05 |

**REGIME_B by month**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 2026-08 | 3565 | 3.00 | 2.90 | 2.67 | 2.76 |
| 2026-09 | 4920 | 3.97 | 3.60 | 3.49 | 3.53 |
| 2026-10 | 3309 | 6.85 | 5.94 | 6.49 | 5.73 |

**REGIME_B by source**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 2 | 11794 | 4.48 | 4.05 | 4.08 | 3.92 |

**REGIME_A by station**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 02t | 2619 | 3.80 | 3.73 | 3.80 | 3.83 |
| 59t | 2661 | 3.46 | 3.24 | 3.46 | 3.27 |
| 61t | 2656 | 3.59 | 3.39 | 3.59 | 3.39 |
| bkp100t | 2421 | 5.81 | 5.26 | 5.81 | 5.47 |
| bkp101t | 2419 | 5.34 | 4.63 | 5.34 | 4.88 |

**REGIME_A by hour_group_local**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 0 | 3153 | 4.42 | 3.92 | 4.42 | 4.03 |
| 1 | 3225 | 4.39 | 4.11 | 4.39 | 4.22 |
| 2 | 3224 | 4.36 | 4.04 | 4.36 | 4.18 |
| 3 | 3174 | 4.27 | 3.99 | 4.27 | 4.09 |

**REGIME_A by month**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 3497 | 7.05 | 6.72 | 7.05 | 6.88 |
| 2026-02 | 1425 | 6.78 | 5.35 | 6.78 | 5.70 |
| 2026-04 | 3421 | 3.39 | 3.19 | 3.39 | 3.19 |
| 2026-05 | 1344 | 3.31 | 2.99 | 3.31 | 2.99 |
| 2026-06 | 2949 | 1.72 | 1.72 | 1.72 | 1.84 |
| 2026-07 | 140 | 1.54 | 1.38 | 1.54 | 1.68 |

**REGIME_A by source**

| key | n | persistence | primary | smoother | LGBM direct |
|---|---:|---:|---:|---:|---:|
| 1 | 12776 | 4.36 | 4.02 | 4.36 | 4.13 |

### Global vs current per-horizon boosters (paired diff, global minus current)

| Group | horizon | global (no id) | global (+id) |
|---|---|---|---|
| HOLDOUT | +6 | -0.012 [-0.061, +0.034] | +0.018 [-0.032, +0.068] |
| HOLDOUT | +12 | -0.232 [-0.304, -0.160] | -0.206 [-0.278, -0.135] |
| HOLDOUT | +24 | +0.048 [-0.042, +0.139] | +0.087 [-0.000, +0.175] |
| REGIME_B | +6 | -0.147 [-0.266, -0.025] | -0.161 [-0.278, -0.042] |
| REGIME_B | +12 | -0.385 [-0.539, -0.234] | -0.381 [-0.532, -0.234] |
| REGIME_B | +24 | -0.310 [-0.523, -0.105] | -0.338 [-0.563, -0.121] |
| REGIME_A | +6 | -0.154 [-0.231, -0.080] | -0.104 [-0.194, -0.011] |
| REGIME_A | +12 | -0.239 [-0.355, -0.121] | -0.197 [-0.329, -0.059] |
| REGIME_A | +24 | -0.407 [-0.652, -0.163] | -0.407 [-0.659, -0.157] |

### Selected configurations at +24 h (validation-selected)

- F1: primary = lgbm_resid; lgbm_direct: val MAE 4.674 {'var': 'all', 'set': 'D', 'kind': 'direct', 'obj': 'l2', 'leaves': 7}; ridge: val MAE 4.665 {'var': 'all', 'set': 'E', 'kind': 'resid', 'alpha': 100.0}; lgbm_resid: val MAE 4.485 {'var': 'all', 'set': 'D', 'kind': 'logratio', 'obj': 'l2', 'leaves': 31}; smooth: val MAE 4.951 {'name': 'latest'}
- F2: primary = lgbm_direct; lgbm_direct: val MAE 4.116 {'var': 'all', 'set': 'D', 'kind': 'direct', 'obj': 'l1', 'leaves': 31}; ridge: val MAE 4.240 {'var': 'all', 'set': 'E', 'kind': 'direct', 'alpha': 100.0}; lgbm_resid: val MAE 4.159 {'var': 'all', 'set': 'D', 'kind': 'logratio', 'obj': 'l1', 'leaves': 7}; smooth: val MAE 4.759 {'name': 'latest'}
- F3: primary = ridge; lgbm_direct: val MAE 2.401 {'var': 'all', 'set': 'E', 'kind': 'direct', 'obj': 'l2', 'leaves': 7}; ridge: val MAE 2.360 {'var': 'all', 'set': 'E', 'kind': 'direct', 'alpha': 100.0}; lgbm_resid: val MAE 2.400 {'var': 'all', 'set': 'D', 'kind': 'logratio', 'obj': 'l1', 'leaves': 7}; smooth: val MAE 2.818 {'name': 'latest'}
- F4: primary = lgbm_resid; lgbm_direct: val MAE 1.679 {'var': 'all', 'set': 'E', 'kind': 'direct', 'obj': 'l1', 'leaves': 7}; ridge: val MAE 1.764 {'var': 'all', 'set': 'E', 'kind': 'logratio', 'alpha': 0.1}; lgbm_resid: val MAE 1.627 {'var': 'all', 'set': 'E', 'kind': 'logratio', 'obj': 'l1', 'leaves': 31}; smooth: val MAE 1.814 {'name': 'shrink_mean24_0.4'}
- R1: primary = ridge; lgbm_direct: val MAE 2.303 {'var': 'bonly', 'set': 'B', 'kind': 'direct', 'obj': 'l2', 'leaves': 7}; ridge: val MAE 2.301 {'var': 'bonly', 'set': 'B', 'kind': 'resid', 'alpha': 100.0}; lgbm_resid: val MAE 2.335 {'var': 'bonly', 'set': 'E', 'kind': 'logratio', 'obj': 'l2', 'leaves': 7}; smooth: val MAE 2.488 {'name': 'shrink_mean24_0.7'}
- R2: primary = lgbm_resid; lgbm_direct: val MAE 3.300 {'var': 'ab', 'set': 'B', 'kind': 'direct', 'obj': 'l1', 'leaves': 7}; ridge: val MAE 3.331 {'var': 'bonly', 'set': 'E', 'kind': 'resid', 'alpha': 0.1}; lgbm_resid: val MAE 3.201 {'var': 'bonly', 'set': 'E', 'kind': 'logratio', 'obj': 'l2', 'leaves': 7}; smooth: val MAE 3.336 {'name': 'shrink_mean24_0.7'}
- R3: primary = lgbm_direct; lgbm_direct: val MAE 5.696 {'var': 'ab', 'set': 'A', 'kind': 'direct', 'obj': 'l1', 'leaves': 31}; ridge: val MAE 5.782 {'var': 'bonly', 'set': 'B', 'kind': 'direct', 'alpha': 100.0}; lgbm_resid: val MAE 5.913 {'var': 'ab', 'set': 'E', 'kind': 'logratio', 'obj': 'l1', 'leaves': 31}; smooth: val MAE 6.789 {'name': 'mean6'}
- HOLDOUT: primary = lgbm_resid; lgbm_direct: val MAE 1.687 {'var': 'all', 'set': 'E', 'kind': 'direct', 'obj': 'l1', 'leaves': 7}; ridge: val MAE 1.696 {'var': 'all', 'set': 'E', 'kind': 'logratio', 'alpha': 0.1}; lgbm_resid: val MAE 1.640 {'var': 'all', 'set': 'E', 'kind': 'logratio', 'obj': 'l1', 'leaves': 31}; smooth: val MAE 1.705 {'name': 'shrink_mean24_0.3'}

### Leakage audit

- +6: 300 rows re-derived from truncated data, mismatching cells 0, pm25_t24 always NaN at +24: True
- +12: 300 rows re-derived from truncated data, mismatching cells 0, pm25_t24 always NaN at +24: True
- +24: 300 rows re-derived from truncated data, mismatching cells 0, pm25_t24 always NaN at +24: True
- leaky control (set D + neighbours' mean reading AT the target hour (future info)): MAE 3.008 vs persistence 4.498, paired diff -1.490 [-1.854, -1.154]

