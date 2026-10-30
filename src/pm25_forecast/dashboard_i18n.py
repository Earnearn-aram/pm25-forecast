# ruff: noqa: E501, RUF001
"""Thai / English text for the static dashboard (KAN-73).

The page stays fully static: every translated string is written into the HTML twice, as
``<span class="i" lang="en">..</span><span class="i" lang="th">..</span>``, and CSS shows the
span that matches ``<html lang>``. English is the source text and the key; ``TH`` maps each key to
its Thai text. A key missing from ``TH`` raises ``KeyError`` while building, so a new English
string cannot ship untranslated. ``{name}`` fields are filled by ``str.format`` after escaping.
"""

import html
from collections.abc import Callable
from datetime import datetime

LANGS = ("en", "th")
MONTHS_TH = ("ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.")  # fmt: skip


class Markup(str):
    """Already-safe HTML: not escaped again when used as a format value."""


def _esc(value: object) -> str:
    return str(value) if isinstance(value, Markup) else html.escape(str(value))


TH: dict[str, str] = {
    # page, header
    "Bangkok PM2.5 forecast · live MLOps dashboard": "พยากรณ์ PM2.5 กรุงเทพฯ · แดชบอร์ด MLOps สด",
    "Bangkok PM2.5 · 24 h forecast": "PM2.5 กรุงเทพฯ · พยากรณ์ล่วงหน้า 24 ชม.",
    "A live MLOps loop on free tiers: collect → predict → score → monitor → retrain": "ระบบ MLOps ที่ทำงานจริงบนบริการฟรี: เก็บข้อมูล → พยากรณ์ → ให้คะแนน → เฝ้าดู → ฝึกใหม่",
    "Is it beating the naive guess “tomorrow = today”?": "โมเดลชนะวิธีเดาง่าย ๆ “พรุ่งนี้ = วันนี้” หรือยัง?",
    "Not scored yet": "ยังไม่มีคะแนน",
    "Yes, by {pct}%": "ชนะ {pct}%",
    "Not yet ({sign}{pct}%)": "ยังไม่ชนะ ({sign}{pct}%)",
    "Healthy": "ปกติดี",
    "Degraded": "ทำงานได้ไม่เต็มที่",
    "Failing": "ขัดข้อง",
    "Waiting for data": "รอข้อมูล",
    "{label} · updated {when}": "{label} · อัปเดต {when}",
    "{utc} UTC · {bkk} Bangkok": "{utc} UTC · {bkk} เวลาไทย",
    "Source on GitHub": "ซอร์สโค้ดบน GitHub",
    "Language": "ภาษา",
    # notes
    "The challenger’s first +24 h forecasts are scored {when}. Until then, its numbers below read “waiting”.": "คำพยากรณ์ +24 ชม. ชุดแรกของ challenger (โมเดลที่กำลังทดสอบ) จะถูกให้คะแนน{when} ระหว่างนี้ตัวเลขด้านล่างจะแสดงว่า “รอข้อมูล”",
    "from {dt}": "ตั้งแต่ {dt}",
    "once the first forecasts have matured": "เมื่อคำพยากรณ์ชุดแรกครบกำหนด",
    "No challenger model is configured, so only the baseline runs.": "ยังไม่ได้ตั้งค่าโมเดล challenger จึงรันเฉพาะ baseline (วิธีเปรียบเทียบ)",
    "Some data is unavailable right now: {names}.": "ข้อมูลบางส่วนใช้ไม่ได้ในตอนนี้: {names}",
    "scores": "คะแนน",
    "observations": "ค่าที่วัดได้",
    "chart": "กราฟ",
    "quality": "คุณภาพข้อมูล",
    "drift": "drift",
    "models": "โมเดล",
    "waiting": "รอข้อมูล",
    "n/a": "ไม่มี",
    "{what} unavailable right now.": "{what} ใช้ไม่ได้ในตอนนี้",
    # key numbers
    "Key numbers": "ตัวเลขสำคัญ",
    "Beats the baseline by": "ชนะ baseline (วิธีเปรียบเทียบ) อยู่",
    "challenger vs persistence · +24 h · 7 days": "challenger เทียบ persistence · +24 ชม. · 7 วัน",
    "Typical error (MAE, µg/m³)": "ค่าคลาดเคลื่อนเฉลี่ย (MAE, µg/m³)",
    "challenger": "challenger",
    "challenger · persistence {mae}": "challenger · persistence {mae}",
    "persistence only · challenger not scored yet": "persistence เท่านั้น · challenger ยังไม่มีคะแนน",
    "Data freshness": "ความสดของข้อมูล",
    "{n} min": "{n} นาที",
    "no readings yet": "ยังไม่มีค่าที่วัดได้",
    "newest reading · {up} of {total} stations reporting": "ค่าล่าสุด · {up} จาก {total} สถานีส่งข้อมูล",
    "Input drift (PSI)": "ข้อมูลเข้าเปลี่ยนไป (PSI)",
    "alert above 0.2": "เตือนเมื่อเกิน 0.2",
    "worst: {feature} · alert above 0.2": "แย่สุด: {feature} · เตือนเมื่อเกิน 0.2",
    # chart
    "Forecast vs actual · +24 h": "พยากรณ์เทียบค่าจริง · +24 ชม.",
    "Forecast vs actual · +24 h · {station}": "พยากรณ์เทียบค่าจริง · +24 ชม. · {station}",
    "Station": "สถานี",
    "Actual": "ค่าจริง",
    "Challenger": "Challenger",
    "Challenger (not scored yet)": "Challenger (ยังไม่มีคะแนน)",
    "Persistence (tomorrow = today)": "Persistence (พรุ่งนี้ = วันนี้)",
    "The chart": "กราฟ",
    "Waiting for the first observations.": "รอค่าที่วัดได้ชุดแรก",
    "No hourly data for this station yet.": "สถานีนี้ยังไม่มีข้อมูลรายชั่วโมง",
    "Station {label}, last 7 days in UTC: actual PM2.5 in µg/m³ against persistence and challenger forecasts made 24 h earlier": "สถานี {label} ย้อนหลัง 7 วัน (UTC): PM2.5 จริง หน่วย µg/m³ เทียบกับคำพยากรณ์ persistence และ challenger ที่ทำไว้ล่วงหน้า 24 ชม.",
    "UTC, hourly": "UTC รายชั่วโมง",
    "Each point: the forecast made 24 h before that hour. Last 7 days, hourly, UTC.": "แต่ละจุดคือคำพยากรณ์ที่ทำไว้ก่อนชั่วโมงนั้น 24 ชม. ย้อนหลัง 7 วัน รายชั่วโมง เวลา UTC",
    # skill by horizon
    "Skill by horizon": "ความแม่นตามช่วงเวลาล่วงหน้า",
    "1 − MAE challenger ÷ MAE persistence, on the same station-hours. Above 0 means the model beats “tomorrow = today”.": "1 − MAE ของ challenger ÷ MAE ของ persistence ในสถานีและชั่วโมงเดียวกัน ค่าเกิน 0 แปลว่าโมเดลชนะ “พรุ่งนี้ = วันนี้”",
    "+{h} h": "+{h} ชม.",
    # pipeline
    "Pipeline history": "ประวัติการทำงานของระบบ",
    "Waiting for the first hourly quality run.": "รอรอบตรวจคุณภาพรายชั่วโมงรอบแรก",
    "Collect": "เก็บข้อมูล",
    "Predict": "พยากรณ์",
    "Quality": "ตรวจคุณภาพ",
    "{name}: {ok} of {total} hours ok, {warn} warning, {fail} failed, {missed} missed": "{name}: ปกติ {ok} จาก {total} ชม. เตือน {warn} ล้มเหลว {fail} ไม่มีบันทึก {missed}",
    "{ok}/{total} ok": "{ok}/{total} ปกติ",
    "ran ok": "ทำงานปกติ",
    "warning": "เตือน",
    "failed": "ล้มเหลว",
    "no run recorded (half height)": "ไม่มีบันทึกการทำงาน (แท่งเตี้ย)",
    "Pipeline · last 24 hours": "ระบบ (pipeline) · 24 ชม. ล่าสุด",
    "one cell per UTC hour · newest on the right": "หนึ่งช่องต่อหนึ่งชั่วโมง UTC · ใหม่สุดอยู่ขวา",
    # data quality
    "Data quality": "คุณภาพข้อมูล",
    "Waiting for the first station readings.": "รอค่าที่วัดได้ชุดแรกจากสถานี",
    "Data quality by station, scrollable": "คุณภาพข้อมูลรายสถานี เลื่อนดูได้",
    "Latest hourly check per station": "ผลตรวจรายชั่วโมงล่าสุดของแต่ละสถานี",
    "Newest reading": "ค่าล่าสุด",
    "Missing, 24 h": "ขาดหาย 24 ชม.",
    "Status": "สถานะ",
    "{n} h": "{n} ชม.",
    "OK": "ปกติ",
    "Warn": "เตือน",
    "Fail": "ล้มเหลว",
    "Waiting": "รอข้อมูล",
    "Data quality by station": "คุณภาพข้อมูลรายสถานี",
    # drift
    "Drift": "Drift",
    "Waiting for the first daily drift check.": "รอผลตรวจ drift รายวันครั้งแรก",
    "Alert": "เตือน",
    "Drift · last 7 days vs training data": "Drift (ข้อมูลเปลี่ยน) · 7 วันล่าสุดเทียบกับข้อมูลฝึก",
    "PSI measures how far recent inputs have moved from what the model was trained on. Tick = 0.2 alert line; bars above it turn amber and say Alert.": "PSI วัดว่าข้อมูลเข้าช่วงหลังเปลี่ยนไปจากข้อมูลที่ใช้ฝึกโมเดลมากแค่ไหน ขีด = เส้นเตือนที่ 0.2 แท่งที่เกินเส้นจะเป็นสีเหลืองอำพันและมีคำว่าเตือน",
    # models
    "Model registry": "ทะเบียนโมเดล",
    "none yet": "ยังไม่มี",
    "none configured": "ยังไม่ได้ตั้งค่า",
    "Champion (serves)": "Champion (ตัวที่ใช้งานจริง)",
    "Challenger (shadow)": "Challenger (รันคู่ขนานเพื่อทดสอบ)",
    "Last retrain": "ฝึกใหม่ล่าสุด",
    "Promotion": "การเลื่อนขั้น",
    "Manual, after 1–2 weeks of scores": "ทำด้วยมือ หลังมีคะแนน 1–2 สัปดาห์",
    "Models": "โมเดล",
    # how it works
    "How it works": "ระบบทำงานอย่างไร",
    "every 20 minutes, unattended · $0/month": "ทุก 20 นาที ไม่ต้องมีคนดูแล · $0/เดือน",
    "Pause animation": "หยุดภาพเคลื่อนไหว",
    "Trigger": "เริ่มงาน",
    "cron-job.org starts the run at :07, :27, :47": "cron-job.org สั่งรันที่นาที :07, :27, :47",
    "Air4Thai, OpenAQ and CAMS into Postgres": "ดึง Air4Thai, OpenAQ และ CAMS เข้า Postgres",
    "persistence + challenger (shadow), +6/+12/+24 h": "persistence + challenger (รันคู่ขนาน) ล่วงหน้า +6/+12/+24 ชม.",
    "Score": "ให้คะแนน",
    "forecasts vs actuals once they arrive": "เทียบคำพยากรณ์กับค่าจริงเมื่อค่าจริงมาถึง",
    "Monitor": "เฝ้าดู",
    "data quality, drift (PSI), alerts": "คุณภาพข้อมูล drift (PSI) และการแจ้งเตือน",
    "Retrain": "ฝึกใหม่",
    "weekly; promote only if it beats the champion": "ทุกสัปดาห์ เลื่อนขั้นเฉพาะเมื่อชนะ champion",
    "Storage: Neon Postgres (observations, predictions, scores, models)": "ที่เก็บข้อมูล: Neon Postgres (ค่าที่วัดได้ คำพยากรณ์ คะแนน โมเดล)",
    "Alerts: Discord · daily heartbeat": "แจ้งเตือน: Discord · สัญญาณชีพรายวัน",
    "Weekly retrain feeds a new challenger back into Predict": "การฝึกใหม่รายสัปดาห์ส่ง challenger ตัวใหม่กลับเข้าขั้นพยากรณ์",
    # footer
    "GitHub Actions · Neon Postgres · cron-job.org · $0/month. Data: Air4Thai (Pollution Control Department, Thailand), OpenAQ, CAMS via Open-Meteo.": "GitHub Actions · Neon Postgres · cron-job.org · $0/เดือน ข้อมูลจาก Air4Thai (กรมควบคุมมลพิษ), OpenAQ และ CAMS ผ่าน Open-Meteo",
}  # fmt: skip


def tr(lang: str, key: str, **values: object) -> Markup:
    """One language's text for ``key``, escaped, with ``{fields}`` filled in."""
    text = key if lang == "en" else TH[key]
    return Markup(text.format(**{k: _esc(v) for k, v in values.items()}))


def pair(build: Callable[[str], str]) -> Markup:
    """Both languages of whatever ``build(lang)`` returns (already-safe HTML)."""
    return Markup("".join(f'<span class="i" lang="{lang}">{build(lang)}</span>' for lang in LANGS))


def t(key: str, **values: object) -> Markup:
    return pair(lambda lang: tr(lang, key, **values))


def attr_of(name: str, build: Callable[[str], str]) -> Markup:
    """An attribute in English plus ``data-th-<name>`` / ``data-en-<name>`` for the switch."""
    en, th = (html.escape(build(lang), quote=True) for lang in LANGS)
    return Markup(f'{name}="{en}" data-en-{name}="{en}" data-th-{name}="{th}"')


def attr(name: str, key: str, **values: object) -> Markup:
    return attr_of(name, lambda lang: tr(lang, key, **values))


def date_hm(value: datetime, lang: str, *, comma: bool = False) -> str:
    """``30 Oct 02:48`` (or ``30 Oct, 02:48``); Thai month abbreviations in Thai."""
    month = MONTHS_TH[value.month - 1] if lang == "th" else f"{value:%b}"
    return f"{value:%d} {month}{',' if comma else ''} {value:%H:%M}"
