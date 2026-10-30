"""Render markdown tables for the v2 (post-review) run from out/kan65_v2/results.json."""

# ruff: noqa: E501
from __future__ import annotations

import json
import sys
from pathlib import Path

FAM = ["smoother", "lgbm_direct", "lgbm_logratio", "ridge", "current"]
EXTRA = [
    "smoother_b05",
    "global",
    "global_id",
    "lgbm_direct_bonly",
    "lgbm_logratio_bonly",
    "ridge_bonly",
]
NAMES = {
    "smoother": "smoother (latest + beta*(mean24-latest))",
    "lgbm_direct": "LightGBM direct",
    "lgbm_logratio": "LightGBM log-ratio",
    "ridge": "ridge (residual)",
    "current": "current LightGBM (KAN-64)",
    "smoother_b05": "smoother, beta fixed 0.5",
    "global": "global LGBM (no id)",
    "global_id": "global LGBM (+id)",
    "lgbm_direct_bonly": "LightGBM direct, B-only rows",
    "lgbm_logratio_bonly": "LightGBM log-ratio, B-only rows",
    "ridge_bonly": "ridge, B-only rows",
}


def f(x, d=3, s=False):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if s else f"{x:.{d}f}"


def cis(c):
    return f"{f(c['diff'], 3, True)} [{f(c['lo'], 3, True)}, {f(c['hi'], 3, True)}]"


def main(path: Path, out: Path):
    r = json.loads(path.read_text())
    L = []
    w = L.append
    inf = r["info"]
    w("## Run info (v2)\n")
    w(
        f"- snapshot `{inf['snapshot_sha256']}` matches KAN-15: {inf['matches_kan15_snapshot']}; "
        f"python {inf['python']}, numpy {inf['numpy']}, lightgbm {inf['lightgbm']}; bootstrap seed "
        f"{inf['boot_seed']} ({inf['n_boot']} draws), model seed {inf['model_seed']}; "
        f"runtime {r['total_runtime_s']:.0f} s; family claims {r['n_claims']}"
    )
    w(f"- unique station-hours / target days (+24 h): {r['unique_counts']}\n")
    w("### H-MA test (series statistics, pooled over stations)\n")
    w(
        "| Series | n values | mean | mean abs 1 h change | identical consecutive (0.1) | ACF lag 1 | "
        "ACF lag 24 | diurnal flatness |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in (
        "air4thai_raw_hourly",
        "air4thai_trailing_24h_mean",
        "openaq_AugOct2025",
        "openaq_all",
    ):
        v = r["hma"][k]
        w(
            f"| {k} | {v['n_values']} | {f(v['mean'], 2)} | {f(v['mean_abs_1h_change'], 3)} | "
            f"{f(v['identical_consecutive_share'], 3)} | {f(v['acf_lag1'], 4)} | "
            f"{f(v['acf_lag24'], 3)} | {f(v['diurnal_flatness'], 4)} |"
        )
    w(
        f"\nRatios {json.dumps({k: round(v, 3) for k, v in r['hma']['ratios'].items()})}; "
        f"verdict by the pre-declared rule: **{r['hma']['verdict']}**\n"
    )
    for g, title in (
        ("HOLDOUT", "KAN-15 HOLDOUT"),
        ("REGIME_B", "Unique regime-B pool (G1-G3)"),
        ("REGIME_A", "Regime-A pool (F1-F3), descriptive"),
    ):
        for h in (24, 12, 6):
            G = r["groups"][f"{g}/+{h}"]
            w(
                f"### {title}, +{h} h (persistence MAE {f(G['persistence']['mae'])}, "
                f"n = {G['persistence']['n']})\n"
            )
            w(
                "| Candidate | MAE | bias | diff vs persistence, 1-day blocks [95% CI] | "
                "7-day blocks [95% CI] | Bonferroni(30) 7-day CI | Holm p (1d / 7d) | folds beating |"
            )
            w("|---|---:|---:|---|---|---|---|---:|")
            for c in FAM + EXTRA:
                if c not in G:
                    continue
                v = G[c]
                a, b = v["vs_persistence_1d"], v["vs_persistence_7d"]
                fam = "lo_bonf" in b
                bon = f"[{f(b['lo_bonf'], 3, True)}, {f(b['hi_bonf'], 3, True)}]" if fam else "-"
                hp = f"{f(a['p_holm'], 3)} / {f(b['p_holm'], 3)}" if fam else "-"
                w(
                    f"| {NAMES[c]} | {f(v['mae'])} | {f(v['bias'], 2, True)} | {cis(a)} | "
                    f"[{f(b['lo'], 3, True)}, {f(b['hi'], 3, True)}] | {bon} | {hp} | "
                    f"{G['runs_beating_persistence'][c]} |"
                )
            w("")
    w("### Smoother beta: chosen vs test-optimal (descriptive) per fold\n")
    w("| Fold / horizon | chosen beta | how | test-optimal beta |")
    w("|---|---:|---|---:|")
    for k, v in r["beta"].items():
        w(f"| {k} | {v['chosen']} | {v['how']} | {v['test_oracle']} |")
    w("\n### Smoother beta curve, +24 h: diff vs persistence (7-day-block 95% CI)\n")
    w("| beta | HOLDOUT | Regime B pool | Regime A pool |")
    w("|---|---|---|---|")
    for b in [round(0.1 * i, 1) for i in range(1, 10)]:
        cells = []
        for g in ("HOLDOUT", "REGIME_B", "REGIME_A"):
            c = r["groups"][f"{g}/+24"][f"sm_{b}"]["vs_persistence_7d"]
            cells.append(cis(c))
        w(f"| {b} | " + " | ".join(cells) + " |")
    w("\n### Per run, +24 h: paired diff vs persistence (1-day blocks, descriptive)\n")
    runs = list(r["runs"])
    w("| Candidate | " + " | ".join(runs) + " |")
    w("|---|" + "---|" * len(runs))
    w(
        "| persistence MAE (n) | "
        + " | ".join(
            f"{f(r['per_run'][f'{x}/+24']['persistence_mae'], 2)} ({r['per_run'][f'{x}/+24']['n']})"
            for x in runs
        )
        + " |"
    )
    for c in [*FAM, "smoother_b05"]:
        w(
            f"| {NAMES[c]} | "
            + " | ".join(cis(r["per_run"][f"{x}/+24"][c]["ci_1d"]) for x in runs)
            + " |"
        )
    w("\n### High-pollution safeguard (hours with latest reading > 37.5), +24 h\n")
    w("| Candidate | HOLDOUT | Regime B pool | Regime A pool |")
    w("|---|---|---|---|")
    for c in [*FAM, "smoother_b05", "global"]:
        cells = []
        for g in ("HOLDOUT", "REGIME_B", "REGIME_A"):
            v = r["groups"][f"{g}/+24"][c]["high_latest_gt_37.5"]
            if "ci_7d" in v:
                t = (
                    f"{v['status']}: {f(v['ci_7d']['diff'], 2, True)} "
                    f"[{f(v['ci_7d']['lo'], 2, True)}, {f(v['ci_7d']['hi'], 2, True)}] "
                    f"n={v['n']}, {v['days']} d"
                )
            else:
                t = f"untestable n={v['n']}"
            cells.append(t)
        w(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
    for h in (24, 12, 6):
        w(
            f"\n### Spike onsets (target >= latest + 15), +{h} h; caught = forecast >= latest + 7.5\n"
        )
        w(
            "| Group | candidate | onsets (days) | caught share [7-day CI] | false alarms among "
            "non-onsets | mean error at onset | onset MAE (persistence) | onset MAE diff [7-day CI] |"
        )
        w("|---|---|---|---|---|---|---|---|")
        for g in ("HOLDOUT", "REGIME_B", "REGIME_A"):
            G = r["groups"][f"{g}/+{h}"]
            pe = G["persistence_events"]
            w(
                f"| {g} | persistence | {pe['onsets']} ({pe['onset_days']}) | 0 | 0 | "
                f"{f(pe['mean_error_at_onset'], 2, True)} | {f(pe['mae_at_onset'], 2)} | - |"
            )
            for c in [*FAM[:4], "smoother_b05"]:
                e = G[c]["events"]
                if "caught_share" not in e:
                    continue
                od = e.get("onset_mae_diff_7d")
                w(
                    f"| {g} | {NAMES[c]} | {e['onsets']} ({e['onset_days']}) | "
                    f"{f(e['caught_share'], 3)} [{f(e['caught_lo7'], 3)}, {f(e['caught_hi7'], 3)}] | "
                    f"{f(e['false_alarm_share'], 3)} | {f(e['mean_error_at_onset'], 2, True)} | "
                    f"{f(e['mae_at_onset'], 2)} ({f(e['persistence_mae_at_onset'], 2)}) | "
                    f"{cis(od) if od else 'n/a'} |"
                )
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
