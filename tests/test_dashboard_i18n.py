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
