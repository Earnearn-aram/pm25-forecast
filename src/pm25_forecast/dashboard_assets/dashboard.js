// Offline dashboard controls and Thai/English interface copy. No dependencies or network.
(function () {
  "use strict";
  var language = "en";
  var storageKey = "pm25-dashboard-language";
  var thai = {
    "Bangkok PM2.5 forecast · live MLOps dashboard": "พยากรณ์ PM2.5 กรุงเทพฯ · แดชบอร์ด MLOps",
    "Bangkok PM2.5 · 24 h forecast": "PM2.5 กรุงเทพฯ · พยากรณ์ล่วงหน้า 24 ชั่วโมง",
    "A live MLOps loop on free tiers: collect → predict → score → monitor → retrain": "ระบบ MLOps ที่ทำงานจริงบนบริการฟรี: เก็บข้อมูล → พยากรณ์ → ประเมินผล → เฝ้าระวัง → ฝึกใหม่",
    "Is it beating the naive guess “tomorrow = today”?": "โมเดลแม่นกว่าการเดาว่า “พรุ่งนี้ = วันนี้” หรือยัง?",
    "Not scored yet": "ยังไม่มีผลประเมิน",
    "Source on GitHub": "ดูโค้ดบน GitHub",
    "Key numbers": "ตัวเลขสำคัญ",
    "Dashboard language": "ภาษาแดชบอร์ด",
    "Beats the baseline by": "แม่นกว่าวิธีพื้นฐาน",
    "challenger vs persistence · +24 h · 7 days": "โมเดลท้าชิงเทียบกับค่าคงเดิม · ล่วงหน้า 24 ชม. · 7 วัน",
    "Typical error (MAE, µg/m³)": "ความคลาดเคลื่อนเฉลี่ย (MAE, µg/m³)",
    "challenger": "โมเดลท้าชิง",
    "persistence only · challenger not scored yet": "ใช้ค่าคงเดิมเท่านั้น · โมเดลท้าชิงยังไม่มีผลประเมิน",
    "Data freshness": "ความใหม่ของข้อมูล",
    "no readings yet": "ยังไม่มีค่าที่ตรวจวัด",
    "Input drift (PSI)": "การเปลี่ยนแปลงของข้อมูล (PSI)",
    "alert above 0.2": "แจ้งเตือนเมื่อเกิน 0.2",
    "waiting": "รอข้อมูล",
    "Actual": "ค่าจริง",
    "Challenger": "โมเดลท้าชิง",
    "Challenger (not scored yet)": "โมเดลท้าชิง (ยังไม่มีผลประเมิน)",
    "Persistence (tomorrow = today)": "ค่าคงเดิม (พรุ่งนี้ = วันนี้)",
    "Station": "สถานี",
    "UTC, hourly": "รายชั่วโมง (UTC)",
    "No hourly data for this station yet.": "ยังไม่มีข้อมูลรายชั่วโมงของสถานีนี้",
    "Waiting for the first observations.": "กำลังรอข้อมูลตรวจวัดชุดแรก",
    "Each point: the forecast made 24 h before that hour. Last 7 days, hourly, UTC.": "แต่ละจุดคือค่าพยากรณ์ที่ทำไว้ล่วงหน้า 24 ชั่วโมง แสดงข้อมูลรายชั่วโมงย้อนหลัง 7 วัน ตามเวลา UTC",
    "Skill by horizon": "ความแม่นยำตามระยะพยากรณ์",
    "1 − MAE challenger ÷ MAE persistence, on the same station-hours. Above 0 means the model beats “tomorrow = today”.": "1 − MAE ของโมเดลท้าชิง ÷ MAE ของวิธีค่าคงเดิม โดยเทียบสถานีและชั่วโมงเดียวกัน ค่ามากกว่า 0 หมายถึงแม่นกว่า “พรุ่งนี้ = วันนี้”",
    "Pipeline · last 24 hours": "การทำงานของระบบ · 24 ชั่วโมงล่าสุด",
    "one cell per UTC hour · newest on the right": "หนึ่งช่องต่อชั่วโมง UTC · ล่าสุดอยู่ด้านขวา",
    "Waiting for the first hourly quality run.": "กำลังรอการตรวจคุณภาพรายชั่วโมงครั้งแรก",
    "ran ok": "ทำงานสำเร็จ",
    "warning": "มีคำเตือน",
    "failed": "ล้มเหลว",
    "no run recorded (half height)": "ไม่พบการทำงาน (ช่องครึ่งความสูง)",
    "Data quality by station": "คุณภาพข้อมูลรายสถานี",
    "Data quality by station, scrollable": "คุณภาพข้อมูลรายสถานี เลื่อนดูได้",
    "Latest hourly check per station": "ผลตรวจรายชั่วโมงล่าสุดของแต่ละสถานี",
    "Newest reading": "อายุข้อมูลล่าสุด",
    "Missing, 24 h": "ชั่วโมงที่ขาดใน 24 ชม.",
    "Status": "สถานะ",
    "OK": "ปกติ",
    "Warn": "เตือน",
    "Fail": "ผิดพลาด",
    "Waiting": "รอข้อมูล",
    "Waiting for the first station readings.": "กำลังรอค่าตรวจวัดจากสถานีชุดแรก",
    "Drift · last 7 days vs training data": "การเปลี่ยนแปลง · 7 วันล่าสุดเทียบกับข้อมูลฝึก",
    "Waiting for the first daily drift check.": "กำลังรอการตรวจการเปลี่ยนแปลงของข้อมูลรายวันครั้งแรก",
    "PSI measures how far recent inputs have moved from what the model was trained on. Tick = 0.2 alert line; bars above it turn amber and say Alert.": "PSI วัดว่าข้อมูลล่าสุดเปลี่ยนไปจากข้อมูลที่ใช้ฝึกโมเดลมากเพียงใด ขีดที่ 0.2 คือเกณฑ์แจ้งเตือน แถบที่เกินเกณฑ์จะเป็นสีเหลืองพร้อมข้อความแจ้งเตือน",
    "n/a": "ไม่มีข้อมูล",
    "Models": "โมเดล",
    "Champion (serves)": "โมเดลหลัก (ใช้งานจริง)",
    "Challenger (shadow)": "โมเดลท้าชิง (ทดลองคู่ขนาน)",
    "Last retrain": "ฝึกใหม่ล่าสุด",
    "Promotion": "การเลื่อนเป็นโมเดลหลัก",
    "none yet": "ยังไม่มี",
    "none configured": "ยังไม่ได้กำหนด",
    "Manual, after 1–2 weeks of scores": "อนุมัติด้วยตนเอง หลังเก็บผลประเมิน 1–2 สัปดาห์",
    "How it works": "ระบบทำงานอย่างไร",
    "every 20 minutes, unattended · $0/month": "อัตโนมัติทุก 20 นาที · $0/เดือน",
    "Pause animation": "หยุดภาพเคลื่อนไหว",
    "Resume animation": "เล่นภาพเคลื่อนไหวต่อ",
    "Trigger": "เริ่มทำงาน",
    "cron-job.org starts the run at :07, :27, :47": "cron-job.org เริ่มงานที่นาที :07, :27, :47",
    "Collect": "เก็บข้อมูล",
    "Air4Thai, OpenAQ and CAMS into Postgres": "เก็บ Air4Thai, OpenAQ และ CAMS ลง Postgres",
    "Predict": "พยากรณ์",
    "persistence + challenger (shadow), +6/+12/+24 h": "ค่าคงเดิม + โมเดลท้าชิง (ทดลองคู่ขนาน), +6/+12/+24 ชม.",
    "Score": "ประเมินผล",
    "forecasts vs actuals once they arrive": "เทียบค่าพยากรณ์กับค่าจริงเมื่อได้รับข้อมูล",
    "Monitor": "เฝ้าระวัง",
    "data quality, drift (PSI), alerts": "คุณภาพข้อมูล การเปลี่ยนแปลง (PSI) และการแจ้งเตือน",
    "Retrain": "ฝึกใหม่",
    "weekly; promote only if it beats the champion": "ทุกสัปดาห์ เลื่อนเป็นโมเดลหลักเมื่อแม่นกว่าตัวเดิมเท่านั้น",
    "Storage: Neon Postgres (observations, predictions, scores, models)": "จัดเก็บ: Neon Postgres (ค่าตรวจวัด ค่าพยากรณ์ ผลประเมิน และโมเดล)",
    "Alerts: Discord · daily heartbeat": "แจ้งเตือน: Discord · รายงานสถานะรายวัน",
    "Weekly retrain feeds a new challenger back into Predict": "การฝึกใหม่รายสัปดาห์ส่งโมเดลท้าชิงตัวใหม่กลับไปพยากรณ์",
    "GitHub Actions · Neon Postgres · cron-job.org · $0/month. Data: Air4Thai (Pollution Control Department, Thailand), OpenAQ, CAMS via Open-Meteo.": "GitHub Actions · Neon Postgres · cron-job.org · $0/เดือน ข้อมูล: Air4Thai (กรมควบคุมมลพิษ ประเทศไทย), OpenAQ, CAMS ผ่าน Open-Meteo",
    "No challenger model is configured, so only the baseline runs.": "ยังไม่ได้กำหนดโมเดลท้าชิง จึงใช้วิธีพื้นฐานเท่านั้น",
    "The challenger’s first +24 h forecasts are scored from": "ค่าพยากรณ์ล่วงหน้า 24 ชั่วโมงชุดแรกของโมเดลท้าชิงจะเริ่มประเมินตั้งแต่",
    "The challenger’s first +24 h forecasts are scored once the first forecasts have matured. Until then, its numbers below read “waiting”.": "ค่าพยากรณ์ล่วงหน้า 24 ชั่วโมงชุดแรกของโมเดลท้าชิงจะเริ่มประเมินเมื่อถึงเวลาของค่าพยากรณ์ชุดแรก ระหว่างนี้ตัวเลขด้านล่างจะแสดงว่า “รอข้อมูล”",
    ". Until then, its numbers below read “waiting”.": " ระหว่างนี้ตัวเลขด้านล่างจะแสดงว่า “รอข้อมูล”",
    "once the first forecasts have matured": "เมื่อถึงเวลาของค่าพยากรณ์ชุดแรก",
    "once the first forecasts have matured. Until then, its numbers below read “waiting”.": "เมื่อถึงเวลาของค่าพยากรณ์ชุดแรก ระหว่างนี้ตัวเลขด้านล่างจะแสดงว่า “รอข้อมูล”"
  };
  var sections = {
    "The chart": "กราฟ", "Skill by horizon": "ความแม่นยำตามระยะพยากรณ์",
    "Pipeline history": "ประวัติการทำงาน", "Data quality": "คุณภาพข้อมูล",
    "Drift": "การเปลี่ยนแปลงของข้อมูล", "Model registry": "ทะเบียนโมเดล",
    "scores": "ผลประเมิน", "observations": "ค่าตรวจวัด", "chart": "กราฟ",
    "quality": "คุณภาพข้อมูล", "drift": "การเปลี่ยนแปลงของข้อมูล", "models": "โมเดล"
  };
  var statuses = {
    "Healthy": "ปกติ", "Degraded": "ทำงานได้บางส่วน",
    "Failing": "มีข้อผิดพลาด", "Waiting for data": "รอข้อมูล"
  };
  var patterns = [
    [/^Yes, by (.+)$/, function (m) { return "แม่นกว่า " + m[1]; }],
    [/^Not yet \((.+)\)$/, function (m) { return "ยังไม่แม่นกว่า (" + m[1] + ")"; }],
    [/^Forecast vs actual · \+24 h(.*)$/, function (m) { return "ค่าพยากรณ์เทียบค่าจริง · +24 ชม." + m[1]; }],
    [/^(Healthy|Degraded|Failing|Waiting for data) · updated (.+) Bangkok$/, function (m) { return statuses[m[1]] + " · อัปเดต " + m[2] + " กรุงเทพฯ"; }],
    [/^challenger · persistence (.+)$/, function (m) { return "โมเดลท้าชิง · ค่าคงเดิม " + m[1]; }],
    [/^newest reading · (\d+) of (\d+) stations reporting$/, function (m) { return "ค่าล่าสุด · รายงาน " + m[1] + " จาก " + m[2] + " สถานี"; }],
    [/^worst: (.+) · alert above 0.2$/, function (m) { return "สูงสุด: " + m[1] + " · แจ้งเตือนเมื่อเกิน 0.2"; }],
    [/^(\d+) min$/, function (m) { return m[1] + " นาที"; }],
    [/^(\+?\d+) h$/, function (m) { return m[1] + " ชม."; }],
    [/^(\d+)\/24 ok$/, function (m) { return m[1] + "/24 สำเร็จ"; }],
    [/^([\d.]+) Alert$/, function (m) { return m[1] + " แจ้งเตือน"; }],
    [/^(.+) unavailable right now\.$/, function (m) { return (sections[m[1]] || m[1]) + "ไม่พร้อมใช้งานในขณะนี้"; }],
    [/^Some data is unavailable right now: (.+)\.$/, function (m) { return "ข้อมูลบางส่วนไม่พร้อมใช้งาน: " + m[1].split(", ").map(function (s) { return sections[s] || s; }).join(", "); }],
    [/^from$/, function () { return "ตั้งแต่"; }],
    [/^Station (.+), last 7 days in UTC: actual PM2.5 in µg\/m³ against persistence and challenger forecasts made 24 h earlier$/, function (m) { return "สถานี " + m[1] + " ย้อนหลัง 7 วัน (UTC): PM2.5 จริง หน่วย µg/m³ เทียบกับวิธีค่าคงเดิมและโมเดลท้าชิงที่พยากรณ์ล่วงหน้า 24 ชั่วโมง"; }],
    [/^(Collect|Predict|Quality): (\d+) of 24 hours ok, (\d+) warning, (\d+) failed, (\d+) missed$/, function (m) { return (thai[m[1]] || "คุณภาพ") + ": สำเร็จ " + m[2] + " จาก 24 ชั่วโมง เตือน " + m[3] + " ล้มเหลว " + m[4] + " ไม่พบการทำงาน " + m[5]; }]
  ];
  thai.Quality = "คุณภาพ";

  function translate(text) {
    if (language === "en") return text;
    var word = text.trim();
    var translated = Object.prototype.hasOwnProperty.call(thai, word) ? thai[word] : null;
    if (translated === null) {
      for (var i = 0; i < patterns.length; i++) {
        var match = word.match(patterns[i][0]);
        if (match) { translated = patterns[i][1](match); break; }
      }
    }
    return translated === null ? text : text.replace(word, function () { return translated; });
  }

  // Keep original text nodes: switching back is lossless and never rewrites HTML or charts.
  var texts = [];
  var walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    var node = walker.currentNode;
    if (!node.parentElement.closest("script, style, [data-language], .pill, .drow .name") && node.textContent.trim()) {
      texts.push({ node: node, english: node.textContent });
    }
  }
  var labels = [];
  document.querySelectorAll("[aria-label]").forEach(function (el) {
    if (!el.matches(".pill")) labels.push({ node: el, english: el.getAttribute("aria-label") });
  });
  var buttons = document.querySelectorAll("[data-language]");
  var pills = document.querySelectorAll(".pill[data-station]");
  var charts = document.querySelectorAll(".chart[data-station]");
  var title = document.getElementById("chart-title");
  var pause = document.getElementById("pause");
  var arch = document.getElementById("arch");

  function updateControls() {
    var selected = document.querySelector('.pill[aria-pressed="true"]');
    if (title && selected) title.textContent = translate("Forecast vs actual · +24 h · " + selected.getAttribute("data-label"));
    if (pause) pause.textContent = translate(pause.getAttribute("aria-pressed") === "true" ? "Resume animation" : "Pause animation");
  }
  function setLanguage(next) {
    language = next === "th" ? "th" : "en";
    document.documentElement.lang = language;
    texts.forEach(function (entry) { entry.node.textContent = translate(entry.english); });
    labels.forEach(function (entry) { entry.node.setAttribute("aria-label", translate(entry.english)); });
    buttons.forEach(function (button) { button.setAttribute("aria-pressed", button.getAttribute("data-language") === language ? "true" : "false"); });
    updateControls();
  }
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      setLanguage(button.getAttribute("data-language"));
      try { localStorage.setItem(storageKey, language); } catch (_) { /* Storage is optional. */ }
    });
  });
  pills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      var id = pill.getAttribute("data-station");
      pills.forEach(function (p) { p.setAttribute("aria-pressed", p === pill ? "true" : "false"); });
      charts.forEach(function (c) { c.hidden = c.getAttribute("data-station") !== id; });
      updateControls();
    });
  });
  if (pause && arch) {
    pause.addEventListener("click", function () {
      var on = pause.getAttribute("aria-pressed") !== "true";
      pause.setAttribute("aria-pressed", on ? "true" : "false");
      arch.classList.toggle("paused", on);
      updateControls();
    });
  }
  var saved;
  try { saved = localStorage.getItem(storageKey); } catch (_) { /* Keep English if blocked. */ }
  setLanguage(saved);
  var switcher = document.getElementById("language-switch");
  if (switcher) switcher.hidden = false;
})();
