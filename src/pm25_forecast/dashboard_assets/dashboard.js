// KAN-25 dashboard: station pills and the animation pause. No dependencies, no network.
(function () {
  var pills = document.querySelectorAll(".pill[data-station]");
  var charts = document.querySelectorAll(".chart[data-station]");
  var title = document.getElementById("chart-title");
  pills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      var id = pill.getAttribute("data-station");
      pills.forEach(function (p) {
        p.setAttribute("aria-pressed", p === pill ? "true" : "false");
      });
      charts.forEach(function (c) {
        c.hidden = c.getAttribute("data-station") !== id;
      });
      if (title) {
        title.textContent = "Forecast vs actual · +24 h · " + pill.getAttribute("data-label");
      }
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
