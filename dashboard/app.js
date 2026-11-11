// app.js - asks the Python backend for fresh numbers every 2 s and draws them.
// No calculations here: every number comes from monitor.py.

const C = { blue: "#2563eb", orange: "#f59e0b", red: "#dc2626", green: "#16a34a", grey: "#cbd5e1" };
Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
Chart.defaults.color = "#64748b";
Chart.defaults.animation = false;                       // instant updates, no lag
Chart.defaults.plugins.legend.labels.boxWidth = 12;

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);

// ------------------------------------------------------------------ charts
const todayChart = new Chart($("ch-today"), {
  data: {
    labels: [],
    datasets: [
      { type: "bar", label: "Actual (normal use)", data: [], backgroundColor: C.blue, stack: "a", borderRadius: 3, order: 2 },
      { type: "bar", label: "Actual (banned appliance)", data: [], backgroundColor: C.red, stack: "a", borderRadius: 3, order: 2 },
      { type: "bar", label: "This hour so far", data: [], backgroundColor: "rgba(37,99,235,.28)", stack: "a", borderRadius: 3, order: 2 },
      { type: "line", label: "AI forecast", data: [], borderColor: C.orange, backgroundColor: C.orange,
        borderWidth: 3, pointRadius: 3, tension: 0.3, order: 1 },
    ],
  },
  options: {
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: { x: { stacked: true, title: { display: true, text: "Hour of the day" }, grid: { display: false } },
              y: { stacked: true, beginAtZero: true, title: { display: true, text: "Energy (Wh)" } } },
    plugins: { tooltip: { callbacks: { label: (c) => c.raw == null ? null : `${c.dataset.label}: ${fmt(c.raw, 1)} Wh` } } },
  },
});

const liveChart = new Chart($("ch-live"), {
  type: "line",
  data: { labels: [], datasets: [
    { label: "Power (W)", data: [], borderColor: C.blue, backgroundColor: "rgba(37,99,235,.10)", fill: true,
      borderWidth: 2, pointRadius: 0, tension: 0.15, spanGaps: false },
    { label: "Trip limit", data: [], borderColor: C.red, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
    { label: "30 s limit", data: [], borderColor: C.orange, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
  ] },
  options: {
    maintainAspectRatio: false,
    scales: { x: { ticks: { maxTicksLimit: 6 }, grid: { display: false } },
              y: { beginAtZero: true, title: { display: true, text: "W" } } },
    plugins: { legend: { display: false } },
  },
});

const daysChart = new Chart($("ch-days"), {
  type: "bar",
  data: { labels: [], datasets: [
    { label: "AI forecast", data: [], backgroundColor: C.orange, borderRadius: 4 },
    { label: "Actual", data: [], backgroundColor: C.blue, borderRadius: 4 },
  ] },
  options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true }, x: { grid: { display: false } } } },
});

// ------------------------------------------------------------------ update the page
function render(s) {
  if (!s.ready) {
    $("pill").textContent = "WAITING FOR DATA";
    $("pill").className = "pill off";
    $("last").textContent = `no rows yet in ${s.csv}`;
    return;
  }
  // header
  $("pill").textContent = s.online ? "LIVE" : "NO DATA";
  $("pill").className = "pill " + (s.online ? "live" : "off");
  $("last").textContent = `last reading ${s.last_time} (${fmt(s.age_seconds)} s ago)`;

  // trip banner
  const trip = s.trip;
  $("trip").classList.toggle("hidden", !(trip && trip.active));
  if (trip) $("trip-text").textContent = `${trip.time} — ${trip.reason}. Relay command printed by the monitor.`;

  // KPIs
  const r = s.reading, L = s.limits;
  const fault = r.status !== "OK";
  $("k-power").textContent = fault ? "0" : fmt(r.power, r.power < 100 ? 1 : 0);
  let state = "normal", note = "Normal load";
  if (fault) { state = "high"; note = "Sensor fault / supply off"; }
  else if (r.power > L.instant) { state = "over"; note = `Above ${L.instant} W trip limit`; }
  else if (r.power > L.sustained) { state = "over"; note = `Above ${L.sustained} W — trips after ${L.sustained_s} s`; }
  else if (r.power > 300) { state = "high"; note = "High load"; }
  $("k-power-card").className = `card kpi big state-${state}`;
  $("k-power-note").textContent = note;
  $("k-volt").textContent = fmt(r.voltage, 1);
  $("k-volt-note").textContent = r.voltage > 253 ? "Over-voltage (> 253 V)" : r.voltage < 195 && !fault ? "Under-voltage" : "Normal range";
  $("k-amp").textContent = fmt(r.current, 3);
  $("k-pf").textContent = fmt(r.pf, 2);
  $("k-pf-note").textContent = fault || r.pf <= 0 ? "" : r.pf >= 0.98 ? "Resistive load (heater-like)"
    : r.pf >= 0.85 ? "Good power factor" : "Low PF: electronic loads (SMPS)";
  const e = s.energy;
  $("k-today").textContent = fmt(e.today_wh);
  $("k-today-note").textContent = `₹${fmt(e.today_cost, 2)} at ₹${e.rate}/unit`;
  $("k-bill").textContent = fmt(e.month_est_cost, 0);
  $("k-bill-note").textContent = `${fmt(e.month_est_kwh, 1)} kWh expected · so far ₹${fmt(e.month_cost, 1)}`;

  const ai = s.appliance;
  const aiCard = $("k-ai-card");
  if (ai.kind === "none") {
    $("k-ai").textContent = "Normal appliances"; $("k-ai-note").textContent = "No heating appliance"; aiCard.className = "card kpi ai";
  } else {
    const banned = ["iron", "induction", "heater/rod"].includes(ai.kind);
    $("k-ai").textContent = ai.kind.toUpperCase() + " detected";
    $("k-ai-note").textContent = `${fmt(ai.prob * 100)}% sure` + (banned ? " · not allowed" : "");
    aiCard.className = "card kpi ai " + (banned ? "alert" : "info");
  }

  // today: forecast vs actual
  const t = s.today;
  const sofar = t.hours.map((_, i) => (i === t.current_hour ? t.running_wh : null));
  todayChart.data.labels = t.hours.map((h) => h + ":00");
  todayChart.data.datasets[0].data = t.actual;
  todayChart.data.datasets[1].data = t.banned;
  todayChart.data.datasets[2].data = sofar;
  todayChart.data.datasets[3].data = t.forecast;
  todayChart.update();
  $("c-total").textContent = t.forecast_total_wh == null ? "–" : `${fmt(t.forecast_total_wh)} Wh`;
  $("c-peak").textContent = t.peak_hour == null ? "–" : `${String(t.peak_hour).padStart(2, "0")}:00`;
  $("c-acc").textContent = s.week_accuracy == null ? "after 1st full day" : `${fmt(s.week_accuracy)}%`;
  $("c-fsofar").textContent = `${fmt(t.forecast_sofar_wh)} Wh`;
  $("c-asofar").textContent = `${fmt(t.actual_sofar_wh)} Wh`;

  // live power
  liveChart.data.labels = s.live.t;
  liveChart.data.datasets[0].data = s.live.p;
  liveChart.data.datasets[1].data = s.live.t.map(() => L.instant);
  liveChart.data.datasets[2].data = s.live.t.map(() => L.sustained);
  liveChart.update();

  // last 7 days
  daysChart.data.labels = s.history.map((d) => d.date);
  daysChart.data.datasets[0].data = s.history.map((d) => d.forecast);
  daysChart.data.datasets[1].data = s.history.map((d) => d.actual);
  daysChart.update();

  // events
  $("active-count").textContent = s.active_alerts.length ? `Active now: ${s.active_alerts.join(" · ")}` : "Nothing active right now";
  $("events").innerHTML = s.events.length ? s.events.map((ev) =>
    `<li><span class="t">${ev.time.slice(11, 19)}</span><span><span class="sev ${ev.severity}">${ev.severity}</span>${ev.message}</span></li>`
  ).join("") : `<li><span></span><span class="muted">No events yet</span></li>`;

  // model scores
  const m = s.models;
  if (m) {
    $("m-trained").textContent = `Tested on days the models never saw · trained ${m.trained_at}`;
    const metric = (name, v, cmp) => `<div class="metric"><span class="name">${name}</span><span class="num">${fmt(v, 1)}%</span>
      <div class="bar"><span style="width:${Math.max(0, Math.min(100, v))}%"></span></div>${cmp ? `<span class="cmp">${cmp}</span>` : ""}</div>`;
    $("models").innerHTML =
      `<h3>Forecast (Random Forest)</h3>` +
      metric("Hostel total, per day", m.forecast_hostel_daily, `simple guess "same as yesterday": ${fmt(m.baseline_hostel_daily, 1)}%`) +
      metric("Hostel total, per hour", m.forecast_hostel_hourly) +
      metric("Single room, per day", m.forecast_room_daily, `simple guess "7-day average": ${fmt(m.baseline_room_daily, 1)}%`) +
      `<h3>Anomaly detection (Random Forest)</h3>` +
      metric("Precision (alarms that were real)", m.anomaly_precision * 100) +
      metric("Recall (real ones caught)", m.anomaly_recall * 100) +
      metric("Correct appliance type", m.anomaly_type_acc * 100,
             `Real recorded day: kettle uses caught ${m.real_events}, false alarms ${m.real_false}`);
  }
  $("footer").textContent = `Reading ${s.csv} · ${s.rows.toLocaleString()} rows · supervisory software, not a replacement for MCB / RCCB protection`;
}

async function refresh() {
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    render(await res.json());
  } catch (err) {
    $("pill").textContent = "SERVER OFF";
    $("pill").className = "pill off";
  }
}
refresh();
setInterval(refresh, 2000);
