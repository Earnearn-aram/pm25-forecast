"""Render markdown tables from out/kan65/results.json and eda.json (numbers only, no prose)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = {
    "val_best": "val-selected best (PRIMARY)",
    "seasonal": "seasonal naive",
    "current": "current LightGBM (latest + t-24h)",
    "smooth": "smoother (1-param shrink / rolling)",
    "ridge": "ridge",
    "lgbm_direct": "LightGBM direct",
    "lgbm_resid": "LightGBM residual/log-ratio",
    "ens": "ensemble ridge+LGBM-resid",
    "shrink": "shrink LGBM-resid to persistence",
    "global": "global LGBM (no id)",
    "global_id": "global LGBM (+station id)",
}
ORDER = list(NAMES)


def f(x, d=3, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(p):
    return f"{f(p['diff'], 3, True)} [{f(p['lo'], 3, True)}, {f(p['hi'], 3, True)}]"


def main(res_path: Path, out: Path):
    r = json.loads(res_path.read_text())
    L = []
    w = L.append
    inf = r["info"]
    w("## Run info\n")
    w(
        f"- snapshot SHA-256 `{inf['snapshot_sha256']}` matches KAN-15: "
        f"{inf['matches_kan15_snapshot']}"
    )
    w(
        f"- python {inf['python']}, numpy {inf['numpy']}, scipy {inf['scipy']}, lightgbm "
        f"{inf['lightgbm']}, duckdb {inf['duckdb']}; bootstrap seed {inf['boot_seed']}, "
        "model seed 20261029"
    )
    w(
        f"- fits: {r['fit_counts']}; claims (Bonferroni family): {r['n_claims']}; "
        f"runtime {r['total_runtime_s'] / 60:.1f} min\n"
    )
    for g, title in (
        ("HOLDOUT", "KAN-15 holdout (headline)"),
        ("REGIME_B", "Pooled regime-B folds (F4, R1, R2, R3)"),
        ("REGIME_A", "Pooled regime-A folds (F1, F2, F3)"),
    ):
        for h in (24, 12, 6):
            G = r["groups"][f"{g}/+{h}"]
            w(
                f"### {title}, +{h} h (persistence MAE {f(G['persistence']['mae'])}, "
                f"n={G['persistence']['n']})\n"
            )
            w(
                "| Candidate | MAE | RMSE | bias | paired MAE diff vs persistence [95% CI] | "
                "Bonferroni CI | Holm p | runs beating persistence |"
            )
            w("|---|---:|---:|---:|---|---|---:|---:|")
            for c in ORDER:
                if c not in G:
                    continue
                v = G[c]
                p = v["vs_persistence"]
                w(
                    f"| {NAMES[c]} | {f(v['mae'])} | {f(v['rmse'])} | {f(v['bias'], 2, True)} | "
                    f"{ci(p)} | [{f(p['lo_bonf'], 3, True)}, {f(p['hi_bonf'], 3, True)}] | "
                    f"{f(p['p_holm'], 3)} | {G['stability_runs_beating_persistence'][c]} |"
                )
            w("")
    w("### Per run, +24 h, paired diff vs persistence (95% day-block CI)\n")
    runs = list(r["runs"])
    w("| Candidate | " + " | ".join(runs) + " |")
    w("|---|" + "---|" * len(runs))
    w(
        "| persistence MAE | "
        + " | ".join(f(r["per_run"][f"{x}/+24"]["persistence"]["mae"], 2) for x in runs)
        + " |"
    )
    w(
        "| test pairs | "
        + " | ".join(str(r["per_run"][f"{x}/+24"]["persistence"]["n"]) for x in runs)
        + " |"
    )
    for c in ORDER:
        w(
            f"| {NAMES[c]} | "
            + " | ".join(ci(r["per_run"][f"{x}/+24"][c]["vs_persistence"]) for x in runs)
            + " |"
        )
    w("\n### High-pollution safeguard (rule d), +24 h: paired MAE diff vs persistence (point, n)\n")
    w(
        "| Candidate | HOLDOUT target>37.5 | HOLDOUT latest>37.5 | REG-B target>37.5 | "
        "REG-B latest>37.5 | REG-A target>37.5 | REG-A latest>37.5 |"
    )
    w("|---|---|---|---|---|---|---|")
    for c in ORDER:
        cells = []
        for g in ("HOLDOUT", "REGIME_B", "REGIME_A"):
            for k in ("high_by_target", "high_by_latest"):
                v = r["groups"][f"{g}/+24"][c][f"vs_persistence_{k}"]
                cells.append(
                    "n/a"
                    if v is None
                    else f"{f(v['diff'], 2, True)} [{f(v['lo'], 1, True)}, "
                    f"{f(v['hi'], 1, True)}] n={v['n']}"
                )
        w(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
    w("\n### Bands and bias, +24 h, primary candidate vs persistence (MAE / bias, n)\n")
    w("| Group | band | n | persistence MAE | primary MAE | persistence bias | primary bias |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for g in ("HOLDOUT", "REGIME_B", "REGIME_A"):
        G = r["groups"][f"{g}/+24"]
        for b in ("le15", "15_37.5", "gt37.5"):
            w(
                f"| {g} | {b} | {G['persistence'][f'n_band_{b}']} | "
                f"{f(G['persistence'][f'mae_band_{b}'], 2)} | "
                f"{f(G['val_best'][f'mae_band_{b}'], 2)} | "
                f"{f(G['persistence'][f'bias_band_{b}'], 2, True)} | "
                f"{f(G['val_best'][f'bias_band_{b}'], 2, True)} |"
            )
    w("\n### Breakdown at +24 h (MAE; persistence vs primary vs smoother)\n")
    for g, B in r["breakdown_plus24"].items():
        for view, rows in B.items():
            w(f"**{g} by {view}**\n")
            w("| key | n | persistence | primary | smoother | LGBM direct |")
            w("|---|---:|---:|---:|---:|---:|")
            for k, v in rows.items():
                w(
                    f"| {k} | {v['n']} | {f(v['persistence'], 2)} | {f(v['val_best'], 2)} | "
                    f"{f(v['smooth'], 2)} | {f(v['lgbm_direct'], 2)} |"
                )
            w("")
    w("### Global vs current per-horizon boosters (paired diff, global minus current)\n")
    w("| Group | horizon | global (no id) | global (+id) |")
    w("|---|---|---|---|")
    for g in ("HOLDOUT", "REGIME_B", "REGIME_A"):
        for h in (6, 12, 24):
            G = r["groups"][f"{g}/+{h}"]
            w(
                f"| {g} | +{h} | {ci(G['global']['vs_current'])} | "
                f"{ci(G['global_id']['vs_current'])} |"
            )
    w("\n### Selected configurations at +24 h (validation-selected)\n")
    for run in runs:
        s = r["selections"][f"{run}/+24"]
        w(
            f"- {run}: primary = {s['val_best_family']}; "
            + "; ".join(
                f"{k}: val MAE {v['val_mae']:.3f} "
                + str(
                    {
                        a: b
                        for a, b in v.items()
                        if a in ("var", "set", "kind", "obj", "leaves", "alpha", "name")
                    }
                )
                for k, v in s.items()
                if k in ("lgbm_direct", "lgbm_resid", "ridge", "smooth")
            )
        )
    w("\n### Leakage audit\n")
    for k, v in r["audit"].items():
        w(
            f"- {k}: {v['rows']} rows re-derived from truncated data, mismatching cells "
            f"{v['mismatching_cells']}, pm25_t24 always NaN at +24: {v['t24_all_nan_at_h24']}"
        )
    lc = r["leaky_control"]
    w(
        f"- leaky control ({lc['description']}): MAE {f(lc['mae'])} vs persistence "
        f"{f(lc['persistence_mae'])}, paired diff {ci(lc['paired'])}\n"
    )
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
