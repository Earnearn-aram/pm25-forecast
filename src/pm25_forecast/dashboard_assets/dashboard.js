// KAN-25 dashboard: station pills and the animation pause. KAN-73: Thai / English switch.
// No dependencies, no network. The page is static: both languages are in the HTML and CSS shows
// the one matching <html lang>; the tiny inline script in <head> sets the first language.
(function () {
  var KEY = "pm25-lang";
  var root = document.documentElement;
  var langButtons = document.querySelectorAll(".langbtn[data-lang]");

  function applyLang(lang) {
    root.lang = lang;
    langButtons.forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-lang") === lang ? "true" : "false");
    });
    document.querySelectorAll("[data-" + lang + "-aria-label]").forEach(function (el) {
      el.setAttribute("aria-label", el.getAttribute("data-" + lang + "-aria-label"));
    });
    var title = document.querySelector("title[data-" + lang + "]");
    if (title) {
      document.title = title.getAttribute("data-" + lang);
    }
  }
  langButtons.forEach(function (b) {
    b.addEventListener("click", function () {
      var lang = b.getAttribute("data-lang");
      applyLang(lang);
      try {
        localStorage.setItem(KEY, lang);
      } catch (err) {
        // storage blocked: the choice just is not remembered
      }
    });
  });
  applyLang(root.lang === "th" ? "th" : "en");

  var pills = document.querySelectorAll(".pill[data-station]");
  var charts = document.querySelectorAll(".chart[data-station]");
  var names = document.querySelectorAll("#chart-title .chart-station");
  pills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      var id = pill.getAttribute("data-station");
      pills.forEach(function (p) {
        p.setAttribute("aria-pressed", p === pill ? "true" : "false");
      });
      charts.forEach(function (c) {
        c.hidden = c.getAttribute("data-station") !== id;
      });
      names.forEach(function (n) {
        n.textContent = pill.getAttribute("data-label");
      });
    });
  });
  var pause = document.getElementById("pause");
  var arch = document.getElementById("arch");
  if (pause && arch) {
    pause.addEventListener("click", function () {
      var on = pause.getAttribute("aria-pressed") !== "true";
      pause.setAttribute("aria-pressed", on ? "true" : "false");
      arch.classList.toggle("paused", on);
    });
  }
})();
