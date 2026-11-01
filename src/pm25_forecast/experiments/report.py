# ruff: noqa: E501
"""Self-contained HTML experiment reports (KAN-97).

``python -m pm25_forecast.experiments.report --ticket KAN-83`` writes one HTML file per experiment
ticket from the shared MLflow store and each run's ``evaluation.json`` artifact; ``... report index``
writes ``index.html`` (leaderboard plus one row per ticket report). Plots are matplotlib figures
embedded as base64 PNG, so a file opens offline. Only aggregates are used (scores, CIs, per-station
and per-hour MAE); there is no raw data. Dev split only: holdout runs are never read.

``matplotlib`` is not a declared dependency. It arrives through ``mlflow`` in the ``experiments``
group, and is imported lazily, like ``mlflow`` in ``tracking``.
"""

import argparse
import base64
import html
import io
import json
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pm25_forecast.experiments import tracking

DOCS_DIR = tracking.REPO_ROOT / "docs" / "experiments"
DEFAULT_OUT = DOCS_DIR / "reports"
EXPERIMENT_FILTER = f"name LIKE '{tracking.EXPERIMENT_PREFIX}%'"
HORIZON = "24"
MAX_ROWS = 30  # plots with one row per variant show at most this many (best by MAE)
PRIMARY_CI_LEVEL = 0.95

# Okabe-Ito palette (colour-blind safe); markers/hatching differ too, never colour alone.
BLUE, ORANGE, VERMILLION, GREY, BLACK, SKY, GREEN = (
    "#0072B2",
    "#E69F00",
    "#D55E00",
    "#767676",
    "#000000",
    "#56B4E9",
    "#009E73",
)


@dataclass
class Variant:
    """One finished dev run (the best run of its experiment) with its +24 h aggregates."""

    name: str  # experiment id without the kan78/ prefix
    run_id: str
    run_name: str
    prereg: str
    kind: str  # primary | ablation | sensitivity
    mae: float
    rmse: float | None
    skill: float | None
    n_pairs: int | None
    dirty: bool
    snapshot: str
    git_sha: str
    ci_level: float
    diff: float | None = None
    ci: tuple[float, float] | None = None
    ci_sens: dict[str, tuple[float, float]] = field(default_factory=dict)
    fragile: bool = False
    folds: list[float | None] = field(default_factory=list)
    by_station: dict[str, dict[str, float]] = field(default_factory=dict)
    by_hour: dict[str, dict[str, float]] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    fold_starts: list[str] = field(default_factory=list)
    has_eval: bool = False
    block_h: int = 72
    sens_blocks: list[int] = field(default_factory=list)
    invalid_reasons: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """clean | fragile | none | reference (no CI: the smoother itself)."""
        if self.ci is None:
            return "reference"
        if self.ci[1] < 0:
            return "fragile" if self.fragile else "clean"
        return "none"


# ---- loading --------------------------------------------------------------------------------


def _pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2 and None not in value:
        return float(value[0]), float(value[1])
    return None


def _kind(name: str, ci_level: float) -> str:
    if abs(ci_level - PRIMARY_CI_LEVEL) > 1e-6:
        return "sensitivity"
    return "ablation" if re.search(r"(^|-)abl(-|$)", name) else "primary"


def _read_eval(client: Any, run_id: str, v: Variant) -> None:
    try:
        path = client.download_artifacts(run_id, "evaluation.json")
        data = json.loads(Path(path).read_text())
        score = data["scores"][HORIZON]
    except Exception:  # missing or malformed artifact: the variant stays table-only
        return
    v.has_eval = True
    boot = data.get("bootstrap") or {}
    v.block_h = int(boot.get("primary_block_h", v.block_h))
    v.sens_blocks = [int(b) for b in boot.get("sensitivity_blocks_h") or []]
    if "ci_level" in data:
        v.ci_level = float(data["ci_level"])
    v.invalid_reasons = [str(r) for r in data.get("invalid_reasons") or []]
    v.fold_starts = [str(x) for x in (data.get("plan") or {}).get("fold_starts") or []]
    vs = score.get("vs_smoother") or {}
    v.diff = vs.get("diff")
    v.ci = _pair(vs.get("ci"))
    v.fragile = bool(vs.get("fragile"))
    v.ci_sens = {
        str(k): p for k, val in (vs.get("ci_sensitivity") or {}).items() if (p := _pair(val))
    }
    by_fold = score.get("by_fold") or {}
    v.folds = [by_fold[k].get("diff_vs_smoother") for k in sorted(by_fold, key=int)]
    v.by_station = score.get("mae_by_station") or {}
    v.by_hour = score.get("mae_by_hour_utc") or {}
    v.baselines = {
        k: float(b["mae"]) for k, b in (score.get("baselines") or {}).items() if "mae" in b
    }


def collect_variants(ticket: str) -> list[Variant]:
    """Best valid dev run of every ``kan78/*`` experiment tagged with ``ticket``."""
    client = tracking._client()
    experiments = client.search_experiments(filter_string=EXPERIMENT_FILTER)
    names = {e.experiment_id: e.name for e in experiments}
    if not names:
        return []
    runs = client.search_runs(
        list(names),
        filter_string=f"tags.ticket = '{ticket}' and tags.split = 'dev' and metrics.mae_24h >= 0",
        order_by=["metrics.mae_24h ASC"],
        max_results=5000,
    )
    best: dict[str, Variant] = {}
    for run in runs:
        t, m = run.data.tags, run.data.metrics
        if (
            t.get("invalid") == "true"
            or run.info.lifecycle_stage != "active"
            or run.info.status != "FINISHED"
        ):
            continue
        name = names[run.info.experiment_id].removeprefix(tracking.EXPERIMENT_PREFIX)
        if name in best:
            continue  # sorted by MAE: the first valid run of an experiment is its best
        try:
            ci_level = float(run.data.params.get("ci_level", PRIMARY_CI_LEVEL))
        except ValueError:
            ci_level = PRIMARY_CI_LEVEL
        n_pairs = m.get("n_pairs_24h")
        v = Variant(
            name=name,
            run_id=run.info.run_id,
            run_name=run.info.run_name or name,
            prereg=t.get("prereg", ""),
            kind=_kind(name, ci_level),
            mae=m["mae_24h"],
            rmse=m.get("rmse_24h"),
            skill=m.get("skill_vs_persistence_24h"),
            n_pairs=None if n_pairs is None else int(n_pairs),
            dirty=t.get("git_dirty") == "true",
            snapshot=t.get("snapshot_sha256", ""),
            git_sha=t.get("git_sha", "")[:10],
            ci_level=ci_level,
        )
        _read_eval(client, run.info.run_id, v)
        best[name] = v
    return sorted(best.values(), key=lambda v: v.mae)


# ---- text helpers ---------------------------------------------------------------------------


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _num(x: float | None, digits: int = 3, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _ci_text(ci: tuple[float, float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _inline(text: str) -> str:
    """Escape, then apply code spans, bold and links (the only inline markup we support)."""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(f"<code>{match.group(1)}</code>")
        return f"\x00{len(spans) - 1}\x00"

    out = re.sub(r"`([^`]+)`", stash, html.escape(text, quote=False))
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(
        r"\[([^\]]+)\]\(((?:https?://|#|\.{0,2}/)[^)\s]*)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        out,
    )
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], out)


def markdown_to_html(text: str, *, heading_shift: int = 1) -> str:
    """Tiny markdown subset: headings, paragraphs, lists, tables, code blocks, bold, code, links."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0

    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
        elif stripped.startswith("```") or (line.startswith("    ") and stripped[0] not in "-*"):
            fenced = stripped.startswith("```")
            block: list[str] = []
            i += 1 if fenced else 0
            while i < len(lines) and (
                (fenced and not lines[i].strip().startswith("```"))
                or (not fenced and (lines[i].startswith("    ") or not lines[i].strip()))
            ):
                block.append(lines[i] if fenced else lines[i][4:])
                i += 1
            i += 1 if fenced else 0
            out.append(f"<pre><code>{html.escape(chr(10).join(block).rstrip())}</code></pre>")
        elif m := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            level = min(6, len(m.group(1)) + heading_shift)
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= set("|-: ")
        ):
            head = cells(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            body = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows
            )
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            out.append(
                f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'
            )
        elif re.match(r"^(\s*)([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items: list[str] = []
            while i < len(lines) and lines[i].strip():
                mm = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)$", lines[i])
                if mm:
                    items.append(mm.group(1))
                elif items and lines[i].startswith(" "):
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
        else:
            para = [stripped]
            i += 1
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"^(#{1,6}\s|```|\||\s*([-*]|\d+\.)\s)", lines[i])
            ):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


@dataclass
class Narrative:
    path: Path | None = None
    title: str = ""
    verdict: str = ""
    body_html: str = ""


def find_narrative(ticket: str, docs_dir: Path) -> Narrative:
    """The ``e*.md`` whose title line names the ticket; it stays the written source."""
    for path in sorted(docs_dir.glob("e*.md")):
        text = path.read_text()
        first = next((ln for ln in text.splitlines() if ln.startswith("# ")), "")
        if re.search(rf"\b{re.escape(ticket)}\b", first):
            return Narrative(path, first[2:].strip(), _verdict(text), markdown_to_html(text))
    return Narrative()


def _verdict(text: str) -> str:
    """The 'Verdict: **X**' line of the write-up; empty when there is none (never guessed)."""
    m = re.search(r"^Verdict:\s*\*\*(.+?)\*\*", text, flags=re.M)
    return m.group(1).strip() if m else ""


def load_preregs(variants: Sequence[Variant], docs_dir: Path) -> list[dict[str, Any]]:
    """Hypothesis and decision rule from each variant's prereg YAML (missing files skipped)."""
    out = []
    for v in variants:
        path = docs_dir / "prereg" / f"{v.prereg}.yaml"
        if not path.is_file():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text())
            out.append(
                {
                    "id": v.prereg,
                    "hypothesis": " ".join(str(data["hypothesis"]).split()),
                    "decision_rule": " ".join(str(data["decision_rule"]).split()),
                }
            )
        except Exception:  # unreadable prereg: the report says it is missing rather than failing
            continue
    return out


# ---- plots ----------------------------------------------------------------------------------


def _fig(width: float, height: float) -> Any:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=100)
    return fig


def _png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax: Any) -> None:
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _wrap(text: str, width: int = 78) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


@dataclass
class Plot:
    key: str
    title: str  # states the finding
    png: str
    alt: str
    caption: str
    how: str  # how to read this plot


def _shown(variants: Sequence[Variant]) -> list[Variant]:
    return [v for v in variants if v.kind == "primary" and v.has_eval][:MAX_ROWS]


def _ref(variants: Sequence[Variant], key: str) -> float | None:
    values = [v.baselines[key] for v in variants if key in v.baselines]
    return statistics.median(values) if values else None


def plot_mae(variants: Sequence[Variant]) -> Plot | None:
    rows = _shown(variants)
    if not rows:
        return None
    pers, smooth = _ref(rows, "persistence"), _ref(rows, "smoother")
    best = rows[0]
    title = f"Lowest +24 h error is {best.mae:.2f} ({best.name})"
    if smooth is not None:
        title += f"; smoother {smooth:.2f}"
    if pers is not None:
        title += f"; persistence {pers:.2f}"
    fig = _fig(9, 0.38 * len(rows) + 1.8)
    ax = fig.add_subplot()
    ys = list(range(len(rows)))[::-1]
    ax.scatter([v.mae for v in rows], ys, s=55, color=BLUE, zorder=3, label="variant")
    ax.scatter(
        [rows[0].mae], [ys[0]], s=110, color=BLUE, edgecolor=BLACK, zorder=4, label="lowest MAE"
    )
    if pers is not None:
        ax.axvline(
            pers, color=VERMILLION, linestyle="--", linewidth=2, label=f"persistence {pers:.2f}"
        )
    if smooth is not None:
        ax.axvline(smooth, color=BLACK, linestyle="-", linewidth=2, label=f"smoother {smooth:.2f}")
    ax.set_yticks(ys, [v.name for v in rows])
    ax.set_xlabel("Mean absolute error at +24 h (ug/m3, lower is better; axis does not start at 0)")
    ax.set_title(_wrap(title), fontsize=13, loc="left")
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    _style(ax)
    caption = "Variants are sorted by MAE (best on top)."
    if smooth is not None:
        gap = best.mae - smooth
        caption += (
            f" The best variant is {best.mae:.3f}, {abs(gap):.3f} "
            f"{'below' if gap < 0 else 'above'} the smoother."
        )
    caption += (
        " A gap this size is only meaningful if the paired interval in the next plot is "
        "entirely below zero; the lowest MAE of many variants is partly luck."
    )
    how = (
        "Each dot is one variant's average +24 h error. Left is better. The dashed line is "
        "persistence (tomorrow equals now); the solid line is the smoother baseline. A variant "
        "must be clearly left of the solid line to matter, and 'clearly' is decided in plot 2, "
        "not here."
    )
    return Plot("mae", title, _png(fig), "Dot plot of +24 h MAE per variant", caption, how)


def plot_paired(variants: Sequence[Variant]) -> Plot | None:
    rows = [v for v in _shown(variants) if v.diff is not None]
    if not rows:
        return None
    k = sum(1 for v in rows if v.ci is not None)
    lvl, blk, sens = _cfg(variants)
    clean = [v for v in rows if v.status == "clean"]
    frag = [v for v in rows if v.status == "fragile"]
    if clean:
        title = (
            f"{len(clean)} of {k} variants have a CI entirely below zero; dev only and "
            f"unadjusted for {k} comparisons"
        )
    elif frag:
        title = f"No variant clearly beats the smoother: {len(frag)} CI below zero but fragile"
    else:
        title = f"No variant beats the smoother: all {k} of the {lvl} CIs reach zero"
    fig = _fig(9, 0.38 * len(rows) + 1.9)
    ax = fig.add_subplot()
    ys = list(range(len(rows)))[::-1]
    style = {
        "clean": (BLUE, "o", "CI entirely below 0 (and stable)"),
        "fragile": (ORANGE, "D", "CI below 0 but FRAGILE"),
        "none": (GREY, "s", "CI includes 0 (no evidence)"),
        "reference": (BLACK, "*", "is the smoother (no CI)"),
    }
    seen = set()
    for v, y in zip(rows, ys, strict=True):
        colour, marker, label = style[v.status]
        if v.ci is not None:
            ax.hlines(y, v.ci[0], v.ci[1], color=colour, linewidth=2.5, zorder=2)
        ax.scatter(
            [v.diff],
            [y],
            color=colour,
            marker=marker,
            s=70,
            zorder=3,
            label=None if v.status in seen else label,
        )
        seen.add(v.status)
    ax.axvline(0, color=BLACK, linewidth=1.8, zorder=1)
    ax.set_yticks(ys, [v.name + (" (fragile)" if v.status == "fragile" else "") for v in rows])
    ax.set_xlabel("MAE difference vs smoother, ug/m3 (negative = better than the smoother)")
    ax.set_title(_wrap(title), fontsize=13, loc="left")
    ax.legend(loc="best", fontsize=10, frameon=True)
    _style(ax)
    caption = (
        f"Bars are {lvl} paired block-bootstrap CIs ({blk} h blocks) on the same {rows[0].n_pairs or ''} "
        f"station-hours. {k} variants were compared and the intervals are unadjusted for that "
        f"(exploratory), so some would cross zero by chance alone. "
        "Selecting the best of many also makes the winner look better than it is."
    )
    if frag:
        names = ", ".join(v.name for v in frag)
        caption += (
            f" Fragile: {names} (CI below zero with {blk} h blocks but not with {sens} blocks), "
            "so do not read it as a win."
        )
    how = (
        "Each bar is the plausible range for how much better (left of zero) or worse (right) a "
        "variant is than the smoother on identical cases. Only a bar entirely left of the black "
        "zero line is evidence of an improvement; a bar touching or crossing zero cannot be "
        "told apart from no difference. Orange diamonds are fragile: they cross zero when the "
        "block length is changed."
    )
    return Plot(
        "paired",
        title,
        _png(fig),
        "Forest plot of paired MAE differences vs smoother",
        caption,
        how,
    )


def plot_folds(variants: Sequence[Variant]) -> Plot | None:
    rows = [v for v in _shown(variants) if v.folds and v.ci is not None]
    if not rows:
        return None
    n_folds = max(len(v.folds) for v in rows)
    below = {v.name: sum(1 for f in v.folds if f is not None and f < 0) for v in rows}
    most = max(rows, key=lambda v: below[v.name])
    title = (
        f"Most consistent variant ({most.name}) is below the smoother's MAE in "
        f"{below[most.name]} of {n_folds} weekly folds"
    )
    import numpy as np

    data = np.array([[np.nan if f is None else f for f in v.folds] for v in rows], dtype=float)
    fig = _fig(8, 0.34 * len(rows) + 2.0)
    ax = fig.add_subplot()
    lim = min(0.6, max(0.05, float(np.nanmax(np.abs(data)))))  # clip: outliers saturate
    im = ax.imshow(data, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val == val:
                ax.text(
                    j,
                    i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(val) > 0.6 * lim else BLACK,
                )
    labels = [d[:10] for d in rows[0].fold_starts] or [str(j + 1) for j in range(n_folds)]
    ax.set_xticks(range(n_folds), labels[:n_folds], fontsize=10)
    ax.set_yticks(range(len(rows)), [f"{v.name}  ({below[v.name]}/{n_folds})" for v in rows])
    ax.set_xlabel("Fold start date (UTC; one week of forecasts per fold)")
    ax.set_title(_wrap(title), fontsize=13, loc="left")
    ax.tick_params(labelsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cb.set_label("MAE vs smoother, ug/m3 (purple = better, orange = worse)", fontsize=10)
    caption = (
        "Number in brackets: folds where the variant's MAE is below the smoother's. A real "
        "improvement should show up in most weeks, not in one lucky week. Fold values have no "
        "interval of their own (one week is a small sample) and are descriptive only."
    )
    how = (
        "Each row is a variant, each column a week of test forecasts. Purple cells: the variant "
        "had lower error than the smoother that week. Orange cells: higher error. Look for rows "
        "that are purple across the whole row."
    )
    return Plot(
        "folds", title, _png(fig), "Heatmap of per-fold MAE difference vs smoother", caption, how
    )


def _cfg(variants: Sequence[Variant]) -> tuple[str, int, str]:
    """(CI level text, primary block hours, sensitivity blocks text) from the run data."""
    v = next((x for x in variants if x.kind == "primary" and x.has_eval), None)
    level = v.ci_level if v else PRIMARY_CI_LEVEL
    sens = v.sens_blocks if v and v.sens_blocks else [24, 168]
    return f"{level * 100:g}%", (v.block_h if v else 72), " or ".join(f"{b} h" for b in sens)


def best_variant(variants: Sequence[Variant]) -> Variant | None:
    rows = [v for v in variants if v.kind == "primary" and v.has_eval]
    return rows[0] if rows else None


def plot_breakdown(variants: Sequence[Variant]) -> Plot | None:
    best = best_variant(variants)
    if best is None or not best.by_station or not best.by_hour:
        return None
    import numpy as np

    stations = sorted(best.by_station)
    hours = sorted(best.by_hour)
    n_low_st = sum(
        1 for s in stations if best.by_station[s]["model"] < best.by_station[s]["smoother"]
    )
    n_low_hr = sum(1 for h in hours if best.by_hour[h]["model"] < best.by_hour[h]["smoother"])
    title = (
        f"{best.name} is below the smoother at {n_low_st} of {len(stations)} stations and "
        f"{n_low_hr} of {len(hours)} hours"
    )
    fig = _fig(11, 4.6)
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(stations))
    w = 0.27
    for off, key, colour, label, hatch in (
        (-w, "persistence", VERMILLION, "persistence", "//"),
        (0, "smoother", GREY, "smoother", ""),
        (w, "model", BLUE, best.name, ""),
    ):
        ax1.bar(
            x + off,
            [best.by_station[s][key] for s in stations],
            w,
            color=colour,
            label=label,
            hatch=hatch,
        )
    ax1.set_xticks(x, stations, rotation=30, ha="right")
    ax1.set_ylabel("+24 h MAE (ug/m3)")
    ax1.set_xlabel("Station")
    ax1.set_title("By station", fontsize=12, loc="left")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", color="#DDDDDD")
    ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=10)
    hx = [int(h) for h in hours]
    for key, colour, label, style in (
        ("persistence", VERMILLION, "persistence", "--"),
        ("smoother", BLACK, "smoother", "-"),
        ("model", BLUE, best.name, "-"),
    ):
        ax2.plot(
            hx,
            [best.by_hour[h][key] for h in hours],
            style,
            color=colour,
            linewidth=2.2,
            marker="o" if key == "model" else None,
            markersize=4,
            label=label,
        )
    ax2.set_xlabel("Hour of day of the forecast target (UTC; Bangkok = UTC+7)")
    ax2.set_ylabel("+24 h MAE (ug/m3)")
    ax2.set_title("By hour of day", fontsize=12, loc="left")
    ax2.set_xticks(range(0, 24, 3))
    ax2.legend(fontsize=9)
    ax2.grid(color="#DDDDDD")
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=10)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(_wrap(title, 110), fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    caption = (
        f"The best variant by dev MAE ({best.name}, MAE {best.mae:.3f}) against both baselines. "
        "Per-station and per-hour values come from few cases each (840 per station, about 175 per "
        "hour), carry no CI and are descriptive: they show where a difference comes from, not "
        "whether it is real."
    )
    how = (
        "Left: average error per station; shorter bars are better. Right: average error by "
        "hour of day of the target. If the blue line only dips under the black line at a few "
        "hours or stations, the overall gain is concentrated there rather than general."
    )
    return Plot("breakdown", title, _png(fig), "MAE by station and by hour of day", caption, how)


def plot_ablation(variants: Sequence[Variant]) -> Plot | None:
    abl = [v for v in variants if v.kind == "ablation"]
    ref = best_variant(variants)
    if not abl or ref is None:
        return None
    rows = sorted(abl, key=lambda v: v.mae - ref.mae)
    delta = {v.name: v.mae - ref.mae for v in rows}
    top, low = rows[-1], rows[0]
    gname = lambda v: re.sub(r"^.*?abl-(no-)?", "", v.name)  # noqa: E731
    title = (
        f"Dropping '{gname(top)}' raises MAE most ({delta[top.name]:+.2f}); "
        f"dropping '{gname(low)}' changes it by {delta[low.name]:+.3f}"
    )
    fig = _fig(9, 0.55 * len(rows) + 1.8)
    ax = fig.add_subplot()
    ys = list(range(len(rows)))
    ax.barh(
        ys,
        [delta[v.name] for v in rows],
        color=[BLUE if delta[v.name] > 0 else ORANGE for v in rows],
        height=0.6,
    )
    for y, v in zip(ys, rows, strict=True):
        ax.text(
            max(delta[v.name], 0), y, f" {delta[v.name]:+.3f}", va="center", ha="left", fontsize=10
        )
    ax.axvline(0, color=BLACK, linewidth=1.5)
    ax.set_yticks(ys, [f"without {gname(v)}" for v in rows])
    ax.set_xlabel(
        f"Change in +24 h MAE vs the full model {ref.name} (ug/m3; positive = group helped)"
    )
    ax.set_title(_wrap(title), fontsize=13, loc="left")
    _style(ax)
    caption = (
        f"Reference: {ref.name} (MAE {ref.mae:.3f}). Each bar re-trains the model without one "
        "feature group. This is a single dev run per bar with no CI, so small differences are "
        "indistinguishable from noise. The size of change that counts is set in the prereg decision rule."
    )
    how = (
        "A bar to the right means the model got worse without that group, so the group carries "
        "information. A bar near zero means the group was not needed. Orange (left) means the "
        "model was slightly better without it."
    )
    return Plot(
        "ablation",
        title,
        _png(fig),
        "Bar chart of MAE change when a feature group is removed",
        caption,
        how,
    )


# ---- HTML -----------------------------------------------------------------------------------

CSS = """
:root{--ink:#1a1a1a;--muted:#555;--line:#d0d0d0;--bg:#fff;--soft:#f4f6f8;--blue:#0072B2;--warn:#FFF4DB;--warnb:#B87800}
*{box-sizing:border-box}
body{font:17px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);background:var(--bg);margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:1.8rem;line-height:1.25;margin:.2em 0}
h2{font-size:1.4rem;margin-top:2.2em;border-bottom:2px solid var(--line);padding-bottom:.2em}
h3{font-size:1.2rem;margin:1.6em 0 .5em}
.meta{color:var(--muted);font-size:.95rem}
.chip{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 10px;margin:2px 4px 2px 0;font-size:.9rem;background:var(--soft)}
.box{background:var(--soft);border-left:5px solid var(--blue);padding:10px 16px;border-radius:4px;margin:14px 0}
.box h4{margin:.2em 0 .4em;font-size:1rem}
.warn{background:var(--warn);border-left-color:var(--warnb)}
.plot-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(220px,1fr);gap:18px;align-items:start}
@media (max-width:860px){.plot-grid{grid-template-columns:1fr}}
figure{margin:0}
figure img{max-width:100%;height:auto;border:1px solid var(--line)}
figcaption{font-size:.95rem;color:var(--muted);margin-top:6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:.92rem;margin:10px 0}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{background:var(--soft)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
code{background:#eef0f2;padding:0 4px;border-radius:3px;font-size:.9em}
pre{background:#eef0f2;padding:10px;overflow-x:auto}
.tag{font-weight:600}
.narr{border-top:1px solid var(--line);margin-top:8px}
dl dt{font-weight:600;margin-top:.6em}dl dd{margin:0 0 0 1em}
"""

GLOSSARY = (
    (
        "MAE (mean absolute error)",
        "The average size of the forecast miss, in ug/m3 of PM2.5, 24 hours ahead. Lower is better. All numbers here are +24 h unless a label says otherwise.",
    ),
    (
        "Persistence and the smoother",
        "Persistence predicts 'tomorrow equals now'. The smoother is the current simple baseline: a blend of the latest reading and the 24 h average (KAN-76). A model has to beat both to be worth building on.",
    ),
    (
        "Skill vs persistence",
        "The percentage by which a model's MAE is below persistence's. 0% means no better than persistence. The smoother already earns about 10%, so a model must beat that, not only persistence.",
    ),
    (
        "Paired CI ({lvl}, {blk} h blocks)",
        "For the same station-hours we subtract the smoother's error from the model's error and average. Negative means the model is better. The {lvl} confidence interval is the plausible range for that average, from a block bootstrap ({blk} h blocks, because neighbouring hours are correlated).",
    ),
    (
        "Why the CI must be entirely below zero",
        "If the interval includes zero, the data cannot tell the model from the smoother: the gap may be noise. Only an interval completely below zero is evidence of an improvement. A negative point estimate alone is not.",
    ),
    (
        "Dev vs holdout",
        "Dev is the period used to build and compare models; picking the best of many variants on it flatters the winner. The holdout is a locked later period scored once for the final candidate. These reports show dev only, so nothing here is evidence of future or holdout performance.",
    ),
    (
        "Fragile",
        "The {blk} h CI is below zero but the interval flips to include zero when the block length is changed to {sens}. The conclusion depends on an analysis choice, so it is not trusted.",
    ),
    (
        "Exploratory / unadjusted",
        "When many variants are tried, some look good by chance (at {lvl}, about 1 in {odds}). Intervals here are not corrected for that, so a lone 'success' among many is a lead to test on the holdout, not a result.",
    ),
)


def _glossary_html(variants: Sequence[Variant]) -> str:
    lvl, blk, sens = _cfg(variants)
    odds = round(1 / max(1e-9, 1 - float(lvl.rstrip("%")) / 100))
    fill = {"lvl": lvl, "blk": blk, "sens": sens, "odds": odds}
    items = "".join(
        f"<dt>{_e(t.format(**fill))}</dt><dd>{_e(d.format(**fill))}</dd>" for t, d in GLOSSARY
    )
    return f'<div class="box"><h4>How to read this report</h4><dl>{items}</dl></div>'


def outcome(variants: Sequence[Variant]) -> str:
    """win (some CI entirely below zero, not fragile) | fragile | none, from the numbers."""
    rows = [v for v in variants if v.kind == "primary" and v.has_eval and v.ci is not None]
    if any(v.status == "clean" for v in rows):
        return "win"
    return "fragile" if any(v.status == "fragile" for v in rows) else "none"


def verdict_claims_win(verdict: str) -> bool:
    """Does the written verdict claim a win? Negations (no, not, nothing, stop, pivot) cancel it."""
    v = verdict.lower()
    negated = re.search(r"\b(no|not|nothing|none|never|stop|pivot|fails?)\b", v)
    return bool(re.search(r"\b(win|go|beats?|improves?)\b", v)) and not negated


def headline(variants: Sequence[Variant]) -> str:
    """Plain finding computed from the numbers; never states a win without a clean CI."""
    rows = [v for v in variants if v.kind == "primary" and v.has_eval and v.ci is not None]
    if not rows:
        return "No evaluated variants."
    clean = [v for v in rows if v.status == "clean"]
    frag = [v for v in rows if v.status == "fragile"]
    k = len(rows)
    lvl = _cfg(variants)[0]
    if clean:
        return (
            f"{len(clean)} of {k} variants have a {lvl} CI vs the smoother entirely below zero on dev "
            f"({', '.join(v.name for v in clean)}). Exploratory and unadjusted for {k} comparisons; "
            "dev only, so not evidence of holdout performance."
        )
    if frag:
        return (
            f"No variant clearly beats the smoother. {len(frag)} of {k} have a CI below zero but "
            f"are fragile ({', '.join(v.name for v in frag)})."
        )
    return f"No variant beats the smoother: all {k} paired {lvl} CIs include zero."


def _variant_table(variants: Sequence[Variant]) -> str:
    head = (
        "variant",
        "MAE",
        "skill vs persistence",
        "diff vs smoother",
        f"{_cfg(variants)[0]} CI ({_cfg(variants)[1]} h)",
        "result",
        "run",
        "dirty",
    )
    labels = {
        "clean": "below zero",
        "fragile": "FRAGILE",
        "none": "includes 0",
        "reference": "smoother",
    }
    rows = []
    for v in variants:
        res = labels[v.status] if v.has_eval else "no evaluation.json"
        if v.kind != "primary":
            res += (
                f" ({v.kind}, {v.ci_level * 100:.2f}% CI)"
                if v.kind == "sensitivity"
                else f" ({v.kind})"
            )
        rows.append(
            "<tr>"
            + "".join(
                f"<td{' class=n' if i in (1, 2, 3) else ''}>{c}</td>"
                for i, c in enumerate(
                    (
                        _e(v.name),
                        _num(v.mae),
                        _num(v.skill),
                        _num(v.diff, sign=True),
                        _e(_ci_text(v.ci)),
                        _e(res),
                        f"<code>{_e(v.run_id[:8])}</code>",
                        "DIRTY" if v.dirty else "",
                    )
                )
            )
            + "</tr>"
        )
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _prereg_html(preregs: Sequence[dict[str, Any]]) -> str:
    if not preregs:
        return "<p>No preregistration files were found for these runs.</p>"
    groups: dict[str, list[str]] = {}
    for p in preregs:
        groups.setdefault(p["decision_rule"], []).append(p["id"])
    hyps: dict[str, list[str]] = {}
    for p in preregs:
        hyps.setdefault(p["hypothesis"], []).append(p["id"])

    def ids(xs: list[str]) -> str:
        return ", ".join(xs[:4]) + (f" and {len(xs) - 4} more" if len(xs) > 4 else "")

    out = ["<h4>Hypotheses (fixed before the runs)</h4><ul>"]
    for text, xs in list(hyps.items())[:6]:
        out.append(f"<li>{_e(text)} <span class='meta'>[{_e(ids(xs))}]</span></li>")
    if len(hyps) > 6:
        out.append(f"<li class='meta'>{len(hyps) - 6} more hypotheses in the prereg files.</li>")
    out.append("</ul><h4>Decision rule</h4><ul>")
    for text, xs in groups.items():
        out.append(f"<li>{_e(text)} <span class='meta'>[{_e(ids(xs))}]</span></li>")
    out.append("</ul>")
    return "".join(out)


def _warnings(variants: Sequence[Variant]) -> list[str]:
    out = []
    dirty = [v.name for v in variants if v.dirty]
    if dirty:
        out.append(
            f"DIRTY: {', '.join(dirty)} ran from uncommitted code and cannot be trusted for a decision."
        )
    snaps = sorted({v.snapshot[:12] for v in variants if v.snapshot})
    if len(snaps) > 1:
        out.append(
            f"WARNING: runs have different snapshot_sha256 values ({', '.join(snaps)}); they are not directly comparable."
        )
    missing = [v.name for v in variants if not v.has_eval]
    if missing:
        out.append(f"No evaluation.json for {', '.join(missing)}; they appear in the table only.")
    bad = [v.name for v in variants if v.invalid_reasons]
    if bad:
        out.append(f"Runs with invalid_reasons in evaluation.json: {', '.join(bad)}.")
    sens = [v.name for v in variants if v.kind == "sensitivity"]
    if sens:
        out.append(
            f"{', '.join(sens)} use a different CI level (sensitivity check) and are left out of the plots."
        )
    return out


def build_report(
    ticket: str,
    variants: Sequence[Variant],
    narrative: Narrative,
    preregs: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """The HTML page and the sidecar summary used by the index."""
    plots = [
        p
        for p in (
            plot_mae(variants),
            plot_paired(variants),
            plot_folds(variants),
            plot_breakdown(variants),
            plot_ablation(variants),
        )
        if p is not None
    ]
    title = narrative.title or f"Experiment report {ticket}"
    status = "written up" if narrative.path else "no write-up yet"
    best = best_variant(variants)
    primary = [v for v in variants if v.kind == "primary"]
    snaps = sorted({v.snapshot[:12] for v in variants if v.snapshot})
    pairs = sorted({v.n_pairs for v in primary if v.n_pairs})
    chips = [ticket, f"status: {status}", "split: dev only", f"{len(primary)} variants"]
    if snaps:
        chips.append("snapshot " + ", ".join(snaps))
    if pairs:
        chips.append(f"{pairs[-1]} +24 h pairs per variant")
    chips.append("aggregates only, no raw data")
    parts = [
        f"<h1>{_e(title)}</h1>",
        "<p>" + "".join(f'<span class="chip">{_e(c)}</span>' for c in chips) + "</p>",
        f'<div class="box"><h4>Finding (computed from the runs)</h4><p class="tag">{_e(headline(variants))}</p>'
        + (
            f"<p>Written verdict ({_e(narrative.path.name)}): <strong>{_e(narrative.verdict)}</strong></p>"
            if narrative.verdict and narrative.path
            else ""
        )
        + "</div>",
    ]
    for w in _warnings(variants):
        parts.append(f'<div class="box warn"><strong>{_e(w)}</strong></div>')
    parts.append(_glossary_html(variants))
    parts.append("<h2>Hypothesis and decision rule</h2>" + _prereg_html(preregs))
    parts.append("<h2>Plots</h2>")
    for i, p in enumerate(plots, 1):
        parts.append(
            f'<section><h3>Plot {i}: {_e(p.title)}</h3><div class="plot-grid"><figure>'
            f'<img alt="{_e(p.alt)}" src="data:image/png;base64,{p.png}">'
            f"<figcaption>{_e(p.caption)}</figcaption></figure>"
            f'<aside class="box"><h4>How to read this</h4><p>{_e(p.how)}</p></aside></div></section>'
        )
    if not plots:
        parts.append(
            "<p>No variant has an evaluation.json artifact, so no plots could be drawn.</p>"
        )
    parts.append("<h2>All variants</h2>" + _variant_table(variants))
    if narrative.path:
        parts.append(
            f'<h2>Written analysis</h2><p class="meta">From <code>docs/experiments/{_e(narrative.path.name)}</code>, which stays the written source of record.</p><div class="narr">{narrative.body_html}</div>'
        )
    else:
        parts.append(
            "<h2>Written analysis</h2><p>No <code>docs/experiments/e*.md</code> names this ticket yet.</p>"
        )
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>"
        '<p class="meta"><a href="index.html">All experiments</a></p>'
        + "\n".join(parts)
        + "</main></body></html>\n"
    )
    side = {
        "ticket": ticket,
        "title": title,
        "status": status,
        "verdict": narrative.verdict,
        "outcome": outcome(variants),
        "finding": headline(variants),
        "n_variants": len(primary),
        "best": None if best is None else {"name": best.name, "mae": best.mae},
        "dirty": any(v.dirty for v in variants),
        "snapshots": snaps,
        "plots": [p.key for p in plots],
    }
    return doc, side


# ---- index ----------------------------------------------------------------------------------


def _verdict_cell(s: dict[str, Any]) -> str:
    """Written verdict next to the computed finding; flags a disagreement."""
    written = s.get("verdict") or ""
    out = (
        f"<strong>{_e(written)}</strong>"
        if written
        else "<em>no explicit verdict in the write-up</em>"
    )
    out += f"<br><span class='meta'>Computed: {_e(s.get('finding', ''))}</span>"
    claims = verdict_claims_win(written)
    if written and (
        (claims and s.get("outcome") != "win") or (not claims and s.get("outcome") == "win")
    ):
        out += "<br><strong>MISMATCH: the written verdict and the computed finding disagree; check.</strong>"
    return out


def build_index(out_dir: Path) -> str:
    rows = tracking.collect_leaderboard()
    sidecars = []
    for path in sorted(out_dir.glob("KAN-*.json")):
        try:
            sidecars.append(json.loads(path.read_text()))
        except ValueError:
            continue
    sidecars.sort(key=lambda s: int(re.sub(r"\D", "", s["ticket"]) or 0))
    have = {s["ticket"] for s in sidecars}
    missing = sorted({r["ticket"] for r in rows if r.get("ticket")} - have)

    tk_rows = []
    for s in sidecars:
        best = s.get("best")
        tk_rows.append(
            "<tr>"
            f'<td><a href="{_e(s["ticket"])}.html">{_e(s["ticket"])}</a></td>'
            f"<td>{_e(s['title'])}</td><td>{_e(s['status'])}</td><td>{_verdict_cell(s)}</td>"
            f"<td class='n'>{s['n_variants']}</td>"
            f"<td>{'' if not best else _e(best['name']) + ' (' + _num(best['mae']) + ')'}</td>"
            f"<td>{'DIRTY' if s.get('dirty') else ''}</td></tr>"
        )
    for t in missing:
        tk_rows.append(
            f"<tr><td>{_e(t)}</td><td colspan=6 class='meta'>no report yet: run "
            f"<code>python -m pm25_forecast.experiments.report --ticket {_e(t)}</code></td></tr>"
        )
    lb_head = (
        "experiment",
        "ticket",
        "run",
        "mae_24h",
        "rmse_24h",
        "skill vs persistence",
        "n_pairs",
        "git_sha",
        "snapshot",
        "dirty",
        "prereg",
    )
    lb_rows = []
    for r in rows:
        link = (
            f'<a href="{_e(r["ticket"])}.html">{_e(r["ticket"])}</a>'
            if r.get("ticket") in have
            else _e(r.get("ticket", ""))
        )
        cells = (
            _e(r["experiment"]),
            link,
            _e(r["run"]),
            _num(r["mae"]),
            _num(r["rmse"]),
            _num(r["skill"]),
            "" if r["n_pairs"] is None else str(int(r["n_pairs"])),
            _e(r["git_sha"]),
            _e(r["snapshot"]),
            "DIRTY" if r["dirty"] else "",
            _e(r.get("prereg", "")),
        )
        lb_rows.append(
            "<tr>"
            + "".join(
                f"<td{' class=n' if i in (3, 4, 5, 6) else ''}>{c}</td>"
                for i, c in enumerate(cells)
            )
            + "</tr>"
        )
    snaps = sorted({str(r.get("snapshot", "")) for r in rows})
    warn_html = ""
    if len(snaps) > 1:
        warn_html = (
            '<div class="box warn"><strong>WARNING: ranked runs have different snapshot_sha256 '
            f"values ({_e(', '.join(snaps))}); they are not directly comparable.</strong></div>"
        )
    th_t = "".join(
        f"<th>{h}</th>"
        for h in (
            "ticket (report)",
            "title",
            "write-up",
            "verdict (written, then computed)",
            "variants",
            "best +24 h MAE",
            "dirty",
        )
    )
    th_l = "".join(f"<th>{_e(h)}</th>" for h in lb_head)
    body = (
        "<h1>Experiment reports</h1>"
        '<p class="meta">One report per experiment ticket, generated from the shared MLflow store '
        "(dev split only, aggregates only). Written analysis stays in <code>docs/experiments/e*.md</code>. "
        "Regenerate with <code>python -m pm25_forecast.experiments.report --ticket KAN-n</code> then "
        "<code>... report index</code>.</p>"
        '<div class="box"><h4>How to read the verdicts</h4><p>A variant only counts as better than the '
        "smoother when its paired CI is entirely below zero and not fragile. Everything here is dev "
        "split and exploratory; nothing is holdout evidence.</p></div>"
        f"<h2>Tickets</h2><div class='scroll'><table><thead><tr>{th_t}</tr></thead><tbody>{''.join(tk_rows)}</tbody></table></div>"
        f"<h2>Leaderboard (dev, best run per experiment, by mae_24h)</h2>{warn_html}"
        "<p class='meta'>DIRTY marks runs made from uncommitted code. Unrelated snapshots (for example the "
        "e00 demo) are not comparable with the rest.</p>"
        f"<div class='scroll'><table><thead><tr>{th_l}</tr></thead><tbody>{''.join(lb_rows)}</tbody></table></div>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Experiment reports</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n"
    )


# ---- CLI ------------------------------------------------------------------------------------


def write_ticket_report(ticket: str, out_dir: Path, docs_dir: Path = DOCS_DIR) -> Path:
    if not re.fullmatch(r"KAN-\d+", ticket):
        raise tracking.TrackingError(f"ticket must look like KAN-<n>, got {ticket!r}")
    variants = collect_variants(ticket)
    if not variants:
        raise tracking.TrackingError(f"no valid dev runs tagged ticket={ticket} in the store")
    narrative = find_narrative(ticket, docs_dir)
    doc, side = build_report(ticket, variants, narrative, load_preregs(variants, docs_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticket}.html"
    path.write_text(doc, encoding="utf-8")
    (out_dir / f"{ticket}.json").write_text(json.dumps(side, indent=1, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pm25_forecast.experiments.report")
    parser.add_argument("command", nargs="?", choices=("index",), help="'index' builds index.html")
    parser.add_argument("--ticket", help="ticket to report on, e.g. KAN-83")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help="docs/experiments dir to read e*.md and prereg/ from",
    )
    args = parser.parse_args(argv)
    if args.command == "index":
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "index.html"
        path.write_text(build_index(args.out), encoding="utf-8")
    elif args.ticket:
        path = write_ticket_report(args.ticket, args.out, args.docs_dir)
    else:
        parser.error("give --ticket KAN-<n> or the command 'index'")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
