// Offline dashboard controls and Thai/English interface copy. No dependencies or network.
(function () {
  "use strict";
  var language = "en";
  var storageKey = "pm25-dashboard-language";
  var thai = {
    "Bangkok PM2.5 forecast · live MLOps dashboard": "พยากรณ์ PM2.5 กรุงเทพฯ · แดชบอร์ด MLOps",
    "Bangkok PM2.5 · 24 h forecast": "PM2.5 กรุงเทพฯ · พยากรณ์ล่วงหน้า 24 ชั่วโมง",
    "A live MLOps loop on free tiers: collect → predict → score → monitor → retrain": "ระบบ MLOps ที่ทำงานจริงบนบริการฟรี: เก็บข้อมูล → พยากรณ์ → ประเมินผล → เฝ้าระวัง → ฝึกใหม่",
    "Is it beating the baseline of keeping the latest reading?": "โมเดลแม่นกว่าวิธีใช้ค่าล่าสุดหรือยัง?",
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
    "Stations": "สถานีทั้งหมด",
    "Now": "ตอนนี้",
    "Search stations": "ค้นหาสถานี",
    "Status": "สถานะ",
    "Needs attention": "ต้องตรวจสอบ",
    "All": "ทั้งหมด",
    "Previous": "ก่อนหน้า",
    "Next": "ถัดไป",
    "Warning": "คำเตือน",
    "PSI measures how far recent inputs have moved from what the model was trained on. The threshold is 0.2; an alert requires two consecutive daily breaches.": "PSI วัดความเปลี่ยนแปลงของข้อมูลเทียบกับข้อมูลที่ใช้ฝึก เกณฑ์คือ 0.2 และจะแจ้งเตือนเมื่อเกินเกณฑ์สองวันติดกัน",
    "Air4Thai quality checks; Python and Neon Postgres. These checks do not prove workflow completion.": "ผลตรวจคุณภาพ Air4Thai; Python และ Neon Postgres ผลตรวจไม่ได้ยืนยันว่า workflow ทำงานสำเร็จ",
    "Persistence prediction quality checks; Python and Neon Postgres. These checks do not prove workflow completion.": "ผลตรวจคุณภาพค่าพยากรณ์วิธีค่าคงเดิม; Python และ Neon Postgres ผลตรวจไม่ได้ยืนยันว่า workflow ทำงานสำเร็จ",
    "Worst recorded quality-check status across sources; Python and Neon Postgres.": "สถานะผลตรวจคุณภาพที่แย่ที่สุดจากทุกแหล่งข้อมูล; Python และ Neon Postgres",
    "Explore the map": "สำรวจบนแผนที่",
    "Select a station · latest PM2.5 (µg/m³)": "เลือกสถานี · PM2.5 ล่าสุด (µg/m³)",
    "Numbers are PM2.5 concentrations, not AQI. Use the station list if the map is unavailable.": "ตัวเลขคือความเข้มข้น PM2.5 ไม่ใช่ AQI เลือกสถานีจากรายการหากแผนที่ไม่พร้อมใช้งาน",
    "Upcoming forecasts": "ค่าพยากรณ์ล่วงหน้า",
    "Waiting for future forecasts.": "กำลังรอค่าพยากรณ์ล่วงหน้า",
    "Persistence": "วิธีค่าคงเดิม",
    "Persistence forecast": "พยากรณ์วิธีค่าคงเดิม",
    "Challenger forecast": "พยากรณ์โมเดลท้าชิง",
    "Quality checks · last 24 hours": "ผลตรวจคุณภาพ · 24 ชั่วโมงล่าสุด",
    "checks OK": "ผลตรวจปกติ",
    "no check recorded (half height)": "ไม่พบผลตรวจ (ช่องครึ่งความสูง)",
    "Registered champion": "โมเดลหลักในทะเบียน",
    "Configured challenger (shadow)": "โมเดลท้าชิงที่กำหนด (ทดลองคู่ขนาน)",
    "Air4Thai and CAMS; OpenAQ backfill is manual": "Air4Thai และ CAMS; เติมข้อมูลย้อนหลัง OpenAQ ด้วยตนเอง",
    "A candidate is served only after configuration and review": "โมเดลใหม่จะใช้งานหลังพิจารณาและกำหนดค่า",
    "warning above 0.2": "เตือนเมื่อเกิน 0.2",
    "Input drift (PSI)": "การเปลี่ยนแปลงของข้อมูล (PSI)",
    "alert above 0.2": "แจ้งเตือนเมื่อเกิน 0.2",
    "waiting": "รอข้อมูล",
    "Measured": "ค่าที่วัดได้",
    "Baseline forecast (tomorrow = today)": "พยากรณ์พื้นฐาน (พรุ่งนี้ = วันนี้)",
    "Model forecast": "พยากรณ์จากโมเดล",
    "Accuracy of past forecasts: see Typical error and Skill by horizon above.": "ความแม่นยำของค่าพยากรณ์ในอดีต: ดู ความคลาดเคลื่อนเฉลี่ย และ ความแม่นยำตามระยะพยากรณ์ ด้านบน",
    "Times in Bangkok time (UTC+7). Forecast lines start at the latest measured value and join only the horizons available.": "เวลาเป็นเวลากรุงเทพฯ (UTC+7) เส้นพยากรณ์เริ่มจากค่าที่วัดได้ล่าสุดและเชื่อมเฉพาะระยะพยากรณ์ที่มี",
    "Challenger": "โมเดลท้าชิง",
    "Station": "สถานี",
    "UTC, hourly": "รายชั่วโมง (UTC)",
    "No hourly data for this station yet.": "ยังไม่มีข้อมูลรายชั่วโมงของสถานีนี้",
    "Waiting for the first observations.": "กำลังรอข้อมูลตรวจวัดชุดแรก",
    "Each point: the forecast made 24 h before that hour. Last 7 days, hourly, UTC.": "แต่ละจุดคือค่าพยากรณ์ที่ทำไว้ล่วงหน้า 24 ชั่วโมง แสดงข้อมูลรายชั่วโมงย้อนหลัง 7 วัน ตามเวลา UTC",
    "Skill by horizon": "ความแม่นยำตามระยะพยากรณ์",
    "1 − MAE challenger ÷ MAE persistence, on the same station-hours. Above 0 means the model beats the latest-reading baseline.": "1 − MAE โมเดลท้าชิง ÷ MAE วิธีค่าคงเดิม เทียบสถานีและเวลาเดียวกัน ค่ามากกว่า 0 หมายถึงแม่นกว่าวิธีใช้ค่าล่าสุด",
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
    "none registered": "ยังไม่มีโมเดลหลักในทะเบียน",
    "none configured": "ยังไม่ได้กำหนด",
    "Review unless auto-promotion is enabled": "พิจารณาก่อนเลื่อน เว้นแต่เปิดการเลื่อนอัตโนมัติ",
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
    "weekly; evaluate candidate before promotion": "ฝึกทุกสัปดาห์และประเมินก่อนเลื่อนโมเดล",
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
  var stationThai = {
    "Government Public Relations Department": "กรมประชาสัมพันธ์",
    "Bodindecha School": "โรงเรียนบดินทรเดชา",
    "Bueng Kum District Office": "สำนักงานเขตบึงกุ่ม",
    "Khlong Sam Wa District Office": "สำนักงานเขตคลองสามวา",
    "Bansomdejchaopraya Rajabhat University": "มหาวิทยาลัยราชภัฏบ้านสมเด็จเจ้าพระยา"
  };
  Object.keys(stationThai).forEach(function (name) { thai[name] = stationThai[name]; });
  var patterns = [
    [/^(Persistence|Challenger) forecast: (.+) µg\/m³, valid (.+); issued (.+)$/, function (m) { return (m[1] === "Persistence" ? "พยากรณ์วิธีค่าคงเดิม" : "พยากรณ์โมเดลท้าชิง") + ": " + m[2] + " µg/m³ ใช้ได้ " + m[3] + " ออกเมื่อ " + m[4]; }],
    [/^\+(\d+) h · (.+) µg\/m³$/, function (m) { return "+" + m[1] + " ชม. · " + m[2] + " µg/m³"; }],
    [/^Now: (.+) UTC$/, function (m) { return "ตอนนี้: " + m[1] + " UTC"; }],
    [/^\+24 h: (.+) UTC$/, function (m) { return "+24 ชม.: " + m[1] + " UTC"; }],
    [/^worst: (.+) · warning above 0.2$/, function (m) { return "สูงสุด: " + m[1] + " · เตือนเมื่อเกิน 0.2"; }],
    [/^([\d.]+) Warning$/, function (m) { return m[1] + " เตือน"; }],
    [/^Yes, by (.+)$/, function (m) { return "แม่นกว่า " + m[1]; }],
    [/^Not yet \((.+)\)$/, function (m) { return "ยังไม่แม่นกว่า (" + m[1] + ")"; }],
    [/^PM2\.5 · last 24 h and next 24 h(.*)$/, function (m) { return "PM2.5 · 24 ชม.ที่ผ่านมาและ 24 ชม.ข้างหน้า" + m[1]; }],
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
    [/^Station (.+): PM2\.5 in µg\/m³, measured over the last 24 hours and forecast for the next 24 hours, Bangkok time$/, function (m) { return "สถานี " + m[1] + ": PM2.5 หน่วย µg/m³ ค่าที่วัดได้ 24 ชั่วโมงที่ผ่านมาและค่าพยากรณ์ 24 ชั่วโมงข้างหน้า เวลากรุงเทพฯ"; }],
    [/^Latest ([\d.]+)$/, function (m) { return "ล่าสุด " + m[1]; }],
    [/^\+(\d+)h ([\d.]+)$/, function (m) { return "+" + m[1] + " ชม. " + m[2]; }],
    [/^Measured: (.+) µg\/m³ at (.+)$/, function (m) { return "วัดได้: " + m[1] + " µg/m³ เวลา " + m[2]; }],
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
    if (!node.parentElement.closest("script, style, [data-language], [data-en][data-th], .pill, .drow .name") && node.textContent.trim()) {
      texts.push({ node: node, english: node.textContent });
    }
  }
  var labels = [];
  document.querySelectorAll("[aria-label]").forEach(function (el) {
    if (!el.matches(".pill")) labels.push({ node: el, english: el.getAttribute("aria-label") });
  });
  var checkLabels = document.querySelectorAll(".pipe abbr[title]");
  checkLabels.forEach(function (el) { el.dataset.englishTitle = el.title; });
  var buttons = document.querySelectorAll("[data-language]");
  var pills = document.querySelectorAll(".pill[data-station]");
  var charts = document.querySelectorAll(".chart[data-station]");
  var mapPins = document.querySelectorAll(".map-pin");
  var map = document.getElementById("station-map");
  function mercator(lat, lon, zoom) {
    var scale = 256 * Math.pow(2, zoom), sin = Math.sin(lat * Math.PI / 180);
    return { x: (lon + 180) / 360 * scale,
      y: (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * scale };
  }
  function renderMap() {
    if (!map) return;
    var zoom = 11, w = map.clientWidth, h = map.clientHeight;
    var centerLat = 13.79, centerLon = 100.59;
    if (mapPins.length) {
      var lats = Array.prototype.map.call(mapPins, function (p) { return Number(p.dataset.lat); });
      var lons = Array.prototype.map.call(mapPins, function (p) { return Number(p.dataset.lon); });
      centerLat = (Math.min.apply(null, lats) + Math.max.apply(null, lats)) / 2;
      centerLon = (Math.min.apply(null, lons) + Math.max.apply(null, lons)) / 2;
      while (zoom > 8) {
        var nw = mercator(Math.max.apply(null, lats), Math.min.apply(null, lons), zoom);
        var se = mercator(Math.min.apply(null, lats), Math.max.apply(null, lons), zoom);
        if (se.x - nw.x <= w - 60 && se.y - nw.y <= h - 60) break;
        zoom--;
      }
    }
    var center = mercator(centerLat, centerLon, zoom);
    var tiles = map.querySelector(".map-tiles"); tiles.replaceChildren();
    var note = map.querySelector(".map-fallback");
    map.classList.remove("tiles-failed"); if (note) note.hidden = true;
    var left = center.x - w / 2, top = center.y - h / 2;
    // OSM tile policy: one zoom level, a handful of tiles (at most 4 x 3), no API key.
    var x0 = Math.floor(left / 256), x1 = Math.floor((left + w) / 256);
    var y0 = Math.floor(top / 256), y1 = Math.floor((top + h) / 256);
    if (x1 - x0 > 3) { x0 += Math.floor((x1 - x0 - 3) / 2); x1 = x0 + 3; }
    if (y1 - y0 > 2) { y0 += Math.floor((y1 - y0 - 2) / 2); y1 = y0 + 2; }
    var total = (x1 - x0 + 1) * (y1 - y0 + 1), failed = 0;
    function tileFailed(event) {
      // A failed tile never leaves a blank map: drop it; with every tile gone, show the plain
      // background and a note. Pins and the station list keep working either way.
      if (event.target.parentNode) event.target.remove();
      failed++;
      if (failed >= total) { map.classList.add("tiles-failed"); if (note) note.hidden = false; }
    }
    for (var x = x0; x <= x1; x++) {
      for (var y = y0; y <= y1; y++) {
        var img = document.createElement("img"); img.alt = ""; img.loading = "lazy";
        img.referrerPolicy = "strict-origin-when-cross-origin";
        img.onerror = tileFailed;
        img.src = "https://tile.openstreetmap.org/" + zoom + "/" + x + "/" + y + ".png";
        img.style.left = (x * 256 - left) + "px"; img.style.top = (y * 256 - top) + "px";
        tiles.appendChild(img);
      }
    }
    mapPins.forEach(function (pin) {
      var xy = mercator(Number(pin.dataset.lat), Number(pin.dataset.lon), zoom);
      pin.style.left = (xy.x - left) + "px"; pin.style.top = (xy.y - top) + "px";
    });
  }
  mapPins.forEach(function (pin) { pin.addEventListener("click", function () {
    var target = Array.prototype.find.call(pills, function (p) { return p.dataset.station === pin.dataset.station; });
    if (target) target.click();
  }); });
  window.addEventListener("resize", renderMap);
  renderMap();
  var title = document.getElementById("chart-title");
  var pause = document.getElementById("pause");
  var arch = document.getElementById("arch");

  function updateControls() {
    var selected = document.querySelector('.pill[aria-pressed="true"]');
    mapPins.forEach(function (pin) {
      pin.setAttribute("aria-pressed", selected && pin.dataset.station === selected.dataset.station ? "true" : "false");
      var timestamp = Date.parse(pin.dataset.time);
      if (Number.isFinite(timestamp)) {
        var currentAge = Math.max(0, Math.floor((Date.now() - timestamp) / 60000));
        pin.dataset.age = String(currentAge); pin.dataset.stale = currentAge > 180 ? "true" : "false";
        pin.classList.toggle("stale", currentAge > 180);
      }
      var sourcePill = Array.prototype.find.call(pills, function (p) { return p.dataset.station === pin.dataset.station; });
      if (sourcePill) { var age = pin.dataset.age, isStale = pin.dataset.stale === "true";
        var label = translate(sourcePill.dataset.label) + ": " + pin.textContent + " µg/m³ · " + age + (language === "th" ? " นาที" : " min") + (isStale ? (language === "th" ? " · ข้อมูลเก่า" : " · stale") : ""); pin.setAttribute("aria-label", label); pin.title = label; }
    });
    var selectedPin = Array.prototype.find.call(mapPins, function (p) { return selected && p.dataset.station === selected.dataset.station; });
    var selectedReading = document.getElementById("selected-map-reading");
    if (selectedReading) selectedReading.textContent = selectedPin ? selectedPin.getAttribute("aria-label") : "";
    if (title && selected) title.textContent = translate("PM2.5 · last 24 h and next 24 h") + " · " + translate(selected.getAttribute("data-label"));
    pills.forEach(function (pill) { pill.textContent = translate(pill.getAttribute("data-label")); });
    checkLabels.forEach(function (el) { el.title = translate(el.dataset.englishTitle); });
    if (pause) pause.textContent = translate(pause.getAttribute("aria-pressed") === "true" ? "Resume animation" : "Pause animation");
  }
  function setLanguage(next) {
    language = next === "th" ? "th" : "en";
    document.documentElement.lang = language;
    document.querySelectorAll("[data-en][data-th]").forEach(function (el) {
      el.textContent = el.getAttribute(language === "th" ? "data-th" : "data-en");
    });
    texts.forEach(function (entry) { if (entry.node.parentElement && !entry.node.parentElement.closest("[data-en][data-th]")) entry.node.textContent = translate(entry.english); });
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
  var qualityTools = document.querySelector(".quality-tools"), qualityPagination = document.querySelector(".quality-pagination");
  if (qualityTools && qualityPagination) {
    var search = document.getElementById("quality-search"), filter = document.getElementById("quality-filter");
    var qualityRows = Array.prototype.slice.call(document.querySelectorAll("tbody tr[data-quality-status]"));
    var page = 0, pageSize = 10;
    var previous = document.getElementById("quality-prev"), next = document.getElementById("quality-next");
    function showQuality() {
      var term = search.value.trim().toLowerCase(), chosen = filter.value;
      var matches = qualityRows.filter(function (row) {
        var status = row.dataset.qualityStatus;
        return row.textContent.toLowerCase().indexOf(term) !== -1 &&
          (chosen === "all" || (chosen === "attention" ? status === "warn" || status === "fail" : status === chosen));
      });
      page = Math.max(0, Math.min(page, Math.ceil(matches.length / pageSize) - 1));
      qualityRows.forEach(function (row) { row.hidden = true; });
      matches.slice(page * pageSize, (page + 1) * pageSize).forEach(function (row) { row.hidden = false; });
      document.getElementById("quality-count").textContent = matches.length ?
        (page * pageSize + 1) + "–" + Math.min((page + 1) * pageSize, matches.length) + " / " + matches.length : "0 / 0";
      previous.disabled = page === 0; next.disabled = (page + 1) * pageSize >= matches.length;
    }
    search.addEventListener("input", function () { page = 0; showQuality(); });
    filter.addEventListener("change", function () { page = 0; showQuality(); });
    previous.addEventListener("click", function () { page--; showQuality(); });
    next.addEventListener("click", function () { page++; showQuality(); });
    qualityTools.hidden = false; qualityPagination.hidden = false; showQuality();
  }
  var stepTooltip = document.getElementById("step-tooltip");
  var stepInfo = [
    {en: ["Trigger", "cron-job.org dispatches collection at :07, :27 and :47; GitHub cron is the backup."], th: ["เริ่มทำงาน", "cron-job.org เริ่มเก็บข้อมูลที่นาที :07, :27 และ :47 โดยมี GitHub cron สำรอง"]},
    {en: ["Collect", "Python collects Air4Thai and CAMS to Neon Postgres. OpenAQ backfill is manual."], th: ["เก็บข้อมูล", "Python เก็บข้อมูล Air4Thai และ CAMS ลง Neon Postgres ส่วน OpenAQ เติมย้อนหลังด้วยตนเอง"]},
    {en: ["Predict", "After collection, Python runs persistence and the configured LightGBM challenger. Issue times are hourly."], th: ["พยากรณ์", "หลังเก็บข้อมูล Python ใช้วิธีค่าคงเดิมและโมเดล LightGBM ที่กำหนด เวลาออกพยากรณ์เป็นรายชั่วโมง"]},
    {en: ["Score", "Python compares matured forecasts with observations; MAE and skill use matched station-hours."], th: ["ประเมินผล", "Python เทียบค่าพยากรณ์ที่ถึงเวลากับค่าจริง โดย MAE และ skill ใช้ข้อมูลสถานีและเวลาเดียวกัน"]},
    {en: ["Monitor", "Quality checks run with prediction; PSI drift is checked daily; Discord sends alerts."], th: ["เฝ้าระวัง", "ตรวจคุณภาพพร้อมการพยากรณ์ ตรวจ drift แบบ PSI ทุกวัน และแจ้งเตือนผ่าน Discord"]},
    {en: ["Retrain", "Weekly LightGBM candidate training and evaluation; promotion and serving depend on configuration."], th: ["ฝึกใหม่", "ฝึกและประเมิน LightGBM รายสัปดาห์ การเลื่อนโมเดลและการใช้งานขึ้นอยู่กับการกำหนดค่า"]}
  ];
  if (stepTooltip && arch) {
    var stepPinned = null;
    function placeStep(x, y) {
      var width = stepTooltip.offsetWidth, height = stepTooltip.offsetHeight;
      stepTooltip.style.left = Math.max(12, Math.min(x + 12, innerWidth - width - 12)) + "px";
      stepTooltip.style.top = Math.max(12, Math.min(y + 14, innerHeight - height - 12)) + "px";
    }
    function showStep(button, x, y) {
      var info = stepInfo[Number(button.dataset.step)][language];
      stepTooltip.replaceChildren();
      var titleNode = document.createElement("strong"), body = document.createElement("p");
      titleNode.textContent = info[0]; body.textContent = info[1];
      stepTooltip.appendChild(titleNode); stepTooltip.appendChild(body); stepTooltip.hidden = false;
      var rect = button.getBoundingClientRect(); placeStep(x == null ? rect.left : x, y == null ? rect.bottom : y);
      arch.querySelectorAll(".arch-node").forEach(function (b) { b.setAttribute("aria-expanded", b === button ? "true" : "false"); });
    }
    function hideStep() { stepTooltip.hidden = true; arch.querySelectorAll(".arch-node").forEach(function (b) { b.setAttribute("aria-expanded", "false"); }); }
    arch.querySelectorAll(".arch-node").forEach(function (button) {
      button.addEventListener("mouseenter", function (event) { if (!stepPinned) showStep(button, event.clientX, event.clientY); });
      button.addEventListener("mousemove", function (event) { if (!stepPinned && !stepTooltip.hidden) placeStep(event.clientX, event.clientY); });
      button.addEventListener("mouseleave", function () { if (!stepPinned && document.activeElement !== button) hideStep(); });
      button.addEventListener("focus", function () { if (!stepPinned) showStep(button); });
      button.addEventListener("blur", function () { if (!stepPinned) hideStep(); });
      button.addEventListener("click", function () { stepPinned = stepPinned === button ? null : button; if (stepPinned) showStep(button); else hideStep(); });
    });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") { stepPinned = null; hideStep(); } });
    document.addEventListener("click", function (e) { if (!e.target.closest(".arch-node")) { stepPinned = null; hideStep(); } });
  }
  window.setInterval(updateControls, 60000);
  document.querySelectorAll(".source-list details").forEach(function (detail) {
    detail.addEventListener("mouseenter", function () { detail.open = true; });
    detail.addEventListener("mouseleave", function () { if (!detail.contains(document.activeElement)) detail.open = false; });
  });
  var saved;
  try { saved = localStorage.getItem(storageKey); } catch (_) { /* Keep English if blocked. */ }
  setLanguage(saved);
  var switcher = document.getElementById("language-switch");
  if (switcher) switcher.hidden = false;
})();
