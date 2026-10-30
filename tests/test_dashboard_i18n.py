"""KAN-73: Thai / English switch on the static dashboard. Offline, FakeDb fixtures only."""

import ast
import re
import string
from pathlib import Path

import pytest

from pm25_forecast import dashboard, dashboard_i18n
from pm25_forecast.dashboard_i18n import TH
from test_dashboard import CHALLENGER, Fake, build, seed

THAI = re.compile("[฀-๿]")
FIELDS = string.Formatter()
ASSETS = Path(dashboard.__file__).parent / "dashboard_assets"
# Technical terms that stay in English on purpose (the Thai page glosses them nearby).
SAME_IN_THAI = {"Challenger", "challenger", "Drift", "drift", "challenger · persistence {mae}"}


def fields(text: str) -> set[str]:
    return {name for _, name, _, _ in FIELDS.parse(text) if name}


def string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def called_keys() -> set[str]:
    """First string literal passed to t(), tr(), attr() in dashboard.py (language arg skipped)."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(Path(dashboard.__file__).read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in {"t", "tr", "attr"}:
            continue
        skip = {"en", "th", "aria-label"}  # language argument, attribute name
        literals = [
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        literals = [v for v in literals if v not in skip]
        if literals:
            keys.add(literals[0])
    return keys


def pages(tmp_path: Path) -> dict[str, str]:
    out = {}
    scored = Fake()
    seed(scored)
    for name, fake, kw in (
        ("scored", scored, {}),
        ("empty", Fake(), {}),
        ("no_challenger", scored, {"challenger_model_version": None}),
    ):
        build(fake, tmp_path / name, **kw)
        out[name] = (tmp_path / name / "index.html").read_text()
    broken = Fake()
    seed(broken)
    for table in ("drift_checks", "scores"):
        broken.con.execute(f"DROP TABLE {table}")
    build(broken, tmp_path / "broken")
    out["broken"] = (tmp_path / "broken" / "index.html").read_text()
    return out


def test_every_key_has_a_thai_text_with_the_same_fields():
    assert TH
    for key, thai in TH.items():
        assert thai.strip(), key
        assert fields(key) == fields(thai), key
        assert THAI.search(thai) or (thai == key and key in SAME_IN_THAI), key


def test_no_missing_and_no_dead_keys():
    keys = called_keys()
    assert keys, "no t()/tr()/attr() calls found"
    assert keys <= TH.keys(), sorted(keys - TH.keys())
    # Keys fed in from tables (steps, labels, pipeline and section names) are plain constants.
    constants = string_constants(Path(dashboard.__file__))
    assert TH.keys() <= constants, sorted(TH.keys() - constants)
    for title, body in dashboard.STEPS:
        assert title in TH and body in TH
    assert all(v in TH for v in dashboard.STATUS_LABELS.values())
    assert all(s in TH for s in dashboard.SECTIONS)


def test_tr_raises_on_untranslated_key():
    with pytest.raises(KeyError):
        dashboard_i18n.tr("th", "This text has no Thai yet")
    assert dashboard_i18n.tr("en", "Hello {x}", x="<b>") == "Hello &lt;b&gt;"


def test_every_rendered_string_is_emitted_in_both_languages(tmp_path):
    for name, page in pages(tmp_path).items():
        langs = re.findall(r'<span class="i" lang="(en|th)">', page)
        assert langs and len(langs) % 2 == 0, name
        assert langs[0::2] == ["en"] * (len(langs) // 2), name
        assert langs[1::2] == ["th"] * (len(langs) // 2), name
        assert "data-th-aria-label" in page and "data-en-aria-label" in page


def test_dynamic_python_strings_are_translated(tmp_path):
    p = pages(tmp_path)
    assert "ยังไม่มีคะแนน" in p["empty"] and "Not scored yet" in p["empty"]
    assert "รอข้อมูล" in p["empty"]  # waiting
    assert "ทำงานได้ไม่เต็มที่" in p["scored"] and "Degraded" in p["scored"]
    assert "อัปเดต 12:20 UTC · 19:20 เวลาไทย" in p["scored"]
    assert "updated 12:20 UTC · 19:20 Bangkok" in p["scored"]
    assert "ชนะ 50.0%" in p["scored"] and "Yes, by 50.0%" in p["scored"]
    assert "เตือน" in p["scored"]  # Alert / Warn badges
    assert "ต.ค." in p["scored"]  # Thai month on the chart axis
    assert "ยังไม่ได้ตั้งค่าโมเดล challenger" in p["no_challenger"]
    assert "ใช้ไม่ได้ในตอนนี้" in p["broken"] and "ข้อมูลบางส่วนใช้ไม่ได้" in p["broken"]
    assert CHALLENGER in p["scored"]


def test_toggle_markup_and_default_language(tmp_path):
    page = pages(tmp_path)["scored"]
    assert page.count('<html lang="en">') == 1
    assert re.search(r'role="group" aria-label="Language" data-en-aria-label="Language"', page)
    th = re.search(r'<button[^>]*data-lang="th"[^>]*>TH</button>', page)
    en = re.search(r'<button[^>]*data-lang="en"[^>]*>EN</button>', page)
    assert th and en and 'type="button"' in th[0] and 'type="button"' in en[0]
    assert 'aria-pressed="false"' in th[0] and 'aria-pressed="true"' in en[0]
    assert 'aria-label="TH' in th[0] and 'aria-label="EN' in en[0]
    assert "localStorage" in page and "try{" in page  # first-paint language, storage guarded
    # Preserved controls: station pills, chart switching hooks, pause button.
    assert 'class="pill" data-station="59t"' in page
    assert 'id="pause" aria-pressed="false"' in page and 'id="arch"' in page
    assert 'class="chart-station"' in page


def test_script_and_css_support_the_switch():
    js = (ASSETS / "dashboard.js").read_text()
    assert "localStorage.setItem" in js and "catch" in js
    assert "root.lang = lang" in js and "aria-pressed" in js
    css = (ASSETS / "dashboard.css").read_text()
    assert 'html[lang="en"] .i[lang="th"]' in css and 'html[lang="th"] .i[lang="en"]' in css
    assert ".langbtn" in css and "focus-visible" in css
    assert 'html[lang="th"] body' in css and "line-height: 1.7" in css
    assert css.count("Noto Sans Thai") >= 2 and "Leelawadee UI" in css and "Thonburi" in css
    assert sorted(p.name for p in (ASSETS / "fonts").iterdir()) == [
        "OFL-Archivo.txt",
        "OFL-JetBrainsMono.txt",
        "archivo-latin.woff2",
        "jetbrains-mono-latin.woff2",
    ]
