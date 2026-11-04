"""KAN-73: Thai / English switch on the static dashboard (data-en/data-th + setLanguage).

Offline, FakeDb fixtures only. Ported from the useful parts of PR #48 and adapted to the
mechanism main and #54 actually use: inline ``data-en``/``data-th`` attributes, a client-side
``setLanguage`` and the ``pm25-dashboard-language`` localStorage key.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

from pm25_forecast import dashboard
from test_dashboard import Fake, build, seed

THAI = re.compile("[฀-๿]")
ASSETS = Path(dashboard.__file__).parent / "dashboard_assets"
# Technical terms that may stay identical in both languages.
SAME_IN_THAI = {"PSI", "MAE", "LightGBM", "Challenger", "challenger"}


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag, attrs):
        self.found.append((tag, dict(attrs)))


def parse(page: str) -> list[tuple[str, dict[str, str | None]]]:
    parser = Elements()
    parser.feed(page)
    return parser.found


def pages(tmp_path: Path) -> dict[str, str]:
    scored = Fake()
    seed(scored)
    out = {}
    for name, fake, kw in (
        ("scored", scored, {}),
        ("empty", Fake(), {}),
        ("no_challenger", scored, {"challenger_model_version": None}),
    ):
        build(fake, tmp_path / name, **kw)
        out[name] = (tmp_path / name / "index.html").read_text()
    broken = Fake()
    seed(broken)
    broken.con.execute("DROP TABLE drift_checks")
    build(broken, tmp_path / "broken")
    out["broken"] = (tmp_path / "broken" / "index.html").read_text()
    return out


def test_every_translated_element_has_both_languages(tmp_path):
    for name, page in pages(tmp_path).items():
        translated = [a for _, a in parse(page) if "data-en" in a or "data-th" in a]
        assert translated, name
        for attrs in translated:
            assert attrs.get("data-en", "").strip(), (name, attrs)
            assert attrs.get("data-th", "").strip(), (name, attrs)
            thai, english = attrs["data-th"], attrs["data-en"]
            assert THAI.search(thai) or english in SAME_IN_THAI or thai == english, (name, attrs)


def test_toggle_markup_and_default_language(tmp_path):
    page = pages(tmp_path)["scored"]
    assert page.count('<html lang="en">') == 1
    assert 'id="language-switch"' in page and 'role="group"' in page
    buttons = [a for tag, a in parse(page) if tag == "button" and "data-language" in a]
    assert sorted(b["data-language"] for b in buttons) == ["en", "th"]
    by_lang = {b["data-language"]: b for b in buttons}
    assert all(b["type"] == "button" for b in buttons)
    assert by_lang["en"]["aria-pressed"] == "true" and by_lang["th"]["aria-pressed"] == "false"
    assert by_lang["en"]["lang"] == "en" and by_lang["th"]["lang"] == "th"
    assert re.search(r'data-language="th"[^>]*>ไทย</button>', page)
    assert re.search(r'data-language="en"[^>]*>English</button>', page)
    # The switch is revealed by JS only: without JavaScript the page stays plain English.
    switch = next(a for tag, a in parse(page) if a.get("id") == "language-switch")
    assert "hidden" in switch
    assert re.search(r'<script src="dashboard\.js\?v=[^" ]+" defer>', page)


def test_script_wires_setlanguage_and_storage_key(tmp_path):
    build(Fake(), tmp_path)
    js = (tmp_path / "dashboard.js").read_text()
    assert 'var storageKey = "pm25-dashboard-language"' in js
    assert "function setLanguage(next)" in js
    assert 'language = next === "th" ? "th" : "en"' in js  # unknown values fall back to English
    assert "document.documentElement.lang = language" in js
    assert "localStorage.setItem(storageKey, language)" in js
    assert "localStorage.getItem(storageKey)" in js
    assert js.count("catch (_)") >= 2  # blocked storage never breaks the page
    assert 'getAttribute(language === "th" ? "data-th" : "data-en")' in js
    assert 'setAttribute("aria-pressed"' in js
    assert "switcher.hidden = false" in js
    assert (ASSETS / "dashboard.js").read_text() == js


def test_thai_dictionary_covers_future_forecast_labels(tmp_path):
    js = (ASSETS / "dashboard.js").read_text()
    for english in (
        "Explore the map",
        "Upcoming forecasts",
        "Waiting for future forecasts.",
        "Persistence forecast",
        "Challenger forecast",
    ):
        match = re.search(rf'"{english}": "([^"]+)"', js)
        assert match and THAI.search(match[1]), english
        if "forecast" in english.lower():
            assert "พยากรณ์" in match[1], english


def test_chart_v2_legend_caption_and_labels_have_thai(tmp_path):
    js = (ASSETS / "dashboard.js").read_text()
    for english in (
        "Measured",
        "Baseline forecast (tomorrow = today)",
        "Model forecast",
        "Accuracy of past forecasts: see Typical error and Skill by horizon above.",
    ):
        match = re.search(rf'"{re.escape(english)}": "([^"]+)"', js)
        assert match and THAI.search(match[1]), english
    assert "PM2\\.5 \u00b7 last 24 h and next 24 h" in js
    build(Fake(), tmp_path)
    page = (tmp_path / "index.html").read_text()
    for english in ("Measured", "Baseline forecast (tomorrow = today)"):
        assert f"</i>{english}</span>" in page
    assert 'data-en="Model forecast"' in page and "configured shadow" in page


# --- KAN-103: KPI card tooltips --------------------------------------------------------------

TIP_KEYS = ("skill", "mae", "fresh", "psi")


def _card_html(page: str, title: str) -> str:
    start = page.index(f">{title}</p>")
    start = page.rindex('<div class="card">', 0, start)
    return page[start : page.index("</div></div>", page.index('class="kpi-tip"', start))]


def test_each_kpi_card_has_an_accessible_bilingual_tooltip(tmp_path):
    page = pages(tmp_path)["scored"]
    els = parse(page)
    for key in TIP_KEYS:
        btn = next(a for t, a in els if t == "button" and a.get("data-tip") == key)
        assert btn["type"] == "button" and btn["aria-expanded"] == "false"
        assert btn["aria-controls"] == f"tip-{key}" and "hidden" in btn  # JS reveals it
        panel = next(a for t, a in els if a.get("id") == f"tip-{key}")
        assert panel["role"] == "note" and "hidden" in panel
    titles = (
        "Beats the baseline by",
        "Typical error (MAE, µg/m³)",
        "Data freshness",
        "Input drift (PSI)",
    )
    for key, title in zip(TIP_KEYS, titles, strict=True):
        card = _card_html(page, title)
        assert f'aria-controls="tip-{key}"' in card and f'id="tip-{key}"' in card
        panel_els = [a for _, a in parse(card.split('class="kpi-tip"', 1)[1]) if "data-en" in a]
        assert len(panel_els) >= 3
        for a in panel_els:
            assert THAI.search(a["data-th"]) and a["data-th"] != a["data-en"]


def test_kpi_tooltip_content_matches_the_code(tmp_path):
    page = pages(tmp_path)["scored"]
    skill = _card_html(page, "Beats the baseline by")
    assert "persistence (tomorrow equals now)" in skill
    assert "(MAE of persistence \u2212 MAE of the model) \u00f7 MAE of persistence" in skill
    assert "matched station-hours" in skill and "last 7 days at +24 h" in skill
    assert "Right now n = 96." in skill  # 2 stations x 48 matched issue times
    assert "not a significance test" in skill
    assert "Model: " in skill and "(the shadow challenger)" in skill
    assert "not squared" in _card_html(page, "Typical error (MAE, µg/m³)")
    fresh = _card_html(page, "Data freshness")
    for text in ("60\u201390 minutes", "crosses 3 hours", "Over 6 hours", "at most 3 hours old"):
        assert text in fresh
    psi = _card_html(page, "Input drift (PSI)")
    for text in ("5 quantile bins", "below 0.1", "0.1 to 0.2", "above 0.2", "2 consecutive"):
        assert text in psi
    assert "not that the model got worse" in psi
    assert "Waiting means not comparable yet." in psi
    no_chal = _card_html(pages(tmp_path)["no_challenger"], "Beats the baseline by")
    assert "No challenger is configured" in no_chal


def test_kpi_tooltip_reuses_drift_reason_when_waiting():
    data = {
        "challenger_version": None,
        "kpi": {"worst_psi": None},
        "drift": {
            "features": [
                {"name": "cams_pm25", "psi": None, "status": "insufficient"},
                {"name": "target_pm25", "psi": None, "status": "insufficient"},
            ]
        },
    }
    texts = dashboard._kpi_tip_texts("psi", data)
    assert dashboard.DRIFT_CAMS in texts and dashboard.DRIFT_ERA in texts


def test_kpi_tooltip_escapes_text_and_script_toggles_it(tmp_path):
    texts = dashboard._kpi_tip_texts("skill", {"challenger_version": "<b>x</b>", "scores": {}})
    panel = dashboard._kpi_tip_panel(
        "skill", {"challenger_version": "<b>x</b>", "scores": {}, "kpi": {}}
    )
    assert any("<b>x</b>" in en for en, _ in texts)
    assert "<b>x</b>" not in panel and "&lt;b&gt;x&lt;/b&gt;" in panel
    build(Fake(), tmp_path)
    js = (tmp_path / "dashboard.js").read_text()
    assert 'document.querySelectorAll(".tip-btn")' in js
    assert 'e.key !== "Escape"' in js and "aria-expanded" in js
    css = (tmp_path / "dashboard.css").read_text()
    assert ".kpi-tip" in css and "overflow-wrap: anywhere" in css


def test_kpi_tooltip_js_handlers_registered(tmp_path):
    js = (ASSETS / "dashboard.js").read_text()
    assert re.search(
        r'document\.addEventListener\("keydown",\s*function \(e\)\s*\{\s*'
        r'if \(e\.key !== "Escape"\) return;',
        js,
    )
    assert re.search(
        r'document\.addEventListener\("click",\s*function \(e\)\s*\{\s*tipButtons\.forEach', js
    )
    assert "card.contains(e.target)" in js and "button.hidden = false" in js
    assert re.search(r'button\.addEventListener\("click",.*setTip\(button', js)
