"use strict";

const $ = selector => document.querySelector(selector);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escapeText = value => String(value ?? "").replace(
  /[&<>"']/g,
  character => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character],
);

const MODELS = {
  claude: {label: "Claude Opus 5.5", harness: "Claude Code", color: "#b65434", logo: "logos/claude-code.svg"},
  codex: {label: "GPT-6 Sol", harness: "Codex", color: "#366fc0", logo: "logos/codex.svg"},
  opencode: {label: "GLM 5.3 Flash", harness: "OpenCode", color: "#bb3c68", logo: "logos/opencode.svg"},
};

// Best agent per scenario at 2 hours in the published InferenceBench leaderboard (mean over three seed pairs).
const SCENARIOS = {
  A: {name: "Scenario A · time to first token", published: "3.69× (Claude Opus 4.5)"},
  B: {name: "Scenario B · time per output token", published: "12.03× (Claude Sonnet 4.6)"},
  C: {name: "Scenario C · request throughput", published: "33.93× (Claude Sonnet 4.6)"},
  D: {name: "Scenario D · geometric mean of A to C", published: "3.66× (GLM-5)"},
};

const STATES = {
  preparing: "preparing",
  running: "running",
  restoring: "restoring from checkpoint",
  scoring: "scoring",
  scored: "scored",
  errored: "errored",
};

const charts = {};
let latest;

function clock(seconds) {
  const minutes = Math.max(0, Math.round(seconds / 60));
  const hours = Math.floor(minutes / 60);
  return hours ? hours + "h " + String(minutes % 60).padStart(2, "0") + "m" : minutes + "m";
}

function utc(iso) {
  if (!iso) return "n/a";
  const date = new Date(iso);
  const time = date.toISOString().slice(11, 16) + " UTC";
  const today = new Date().toISOString().slice(0, 10);
  return iso.slice(0, 10) === today ? time : date.toLocaleDateString("en-GB", {day: "numeric", month: "short", timeZone: "UTC"}) + " " + time;
}

function money(value) {
  if (value === null || value === undefined) return "n/a";
  if (value < 1) return "$" + value.toFixed(2);
  return "$" + value.toLocaleString("en", {maximumFractionDigits: value < 100 ? 2 : 0, minimumFractionDigits: value < 100 ? 2 : 0});
}

function tokens(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("en", {notation: "compact", maximumSignificantDigits: 3}).format(value);
}

function agentUsed(run) {
  // Between refreshes a running clock keeps moving: the deadline already accounts for restores.
  if (run.status === "running" && run.deadline) {
    const left = (Date.parse(run.deadline) - Date.now()) / 1000;
    return Math.max(0, Math.min(run.agent_seconds, run.agent_seconds - left));
  }
  return run.agent_seconds_used || 0;
}

function renderTotals(data) {
  const spend = data.spend;
  const counts = {};
  for (const run of data.runs) counts[run.status] = (counts[run.status] || 0) + 1;
  const statusLine = Object.keys(STATES).filter(state => counts[state])
    .map(state => counts[state] + " " + STATES[state]).join(" · ");
  let spendHtml = '<div class="stat-label">Spend so far</div><div class="stat-value">n/a</div>';
  if (spend) {
    const share = spend.total_usd / data.budget_usd;
    spendHtml = (
      '<div class="stat-label">Spend so far against the $' + data.budget_usd.toLocaleString("en") + ' cap</div>' +
      '<div class="stat-value">' + money(spend.total_usd) + ' <small>of $' + data.budget_usd.toLocaleString("en") + '</small></div>' +
      '<div class="meter' + (share > 1 ? " is-over" : share > .8 ? " is-warning" : "") + '" role="img" aria-label="' +
        Math.round(share * 100) + '% of the cap"><span style="width:' + Math.min(100, share * 100).toFixed(2) + '%"></span></div>' +
      '<div class="stat-detail">OpenRouter ' + money(spend.openrouter_usd) + ' · RunPod ' + money(spend.runpod_usd) +
        ' (now ' + money(spend.runpod_rate_per_hour) + '/h) · since ' + utc(spend.since) +
        (data.phase === "smoke" ? " (smoke preflight)" : " (campaign start)") + '</div>'
    );
  }
  $("#totals").innerHTML = (
    '<div>' + spendHtml + '</div>' +
    '<div><div class="stat-label">Runs</div><div class="stat-value">' + data.runs.length + '</div>' +
    '<div class="stat-detail">' + escapeText(statusLine || "none yet") + '</div></div>'
  );
}

const budgetLine = {
  id: "budgetLine",
  afterDatasetsDraw(chart) {
    const x = chart.scales.x;
    if (2 > x.max) return;
    const position = x.getPixelForValue(2);
    const {top, bottom} = chart.chartArea;
    const context = chart.ctx;
    context.save();
    context.strokeStyle = css("--muted");
    context.globalAlpha = .7;
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(position, top);
    context.lineTo(position, bottom);
    context.stroke();
    context.setLineDash([]);
    context.globalAlpha = 1;
    context.fillStyle = css("--muted");
    context.font = "10px -apple-system, Arial, sans-serif";
    context.fillText("2 h published budget", position + 4, top + 10);
    context.restore();
  },
};

const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(chart) {
    const context = chart.ctx;
    context.save();
    context.font = "600 11px -apple-system, Arial, sans-serif";
    context.textBaseline = "middle";
    context.lineWidth = 3;
    context.strokeStyle = "#fff";
    context.lineJoin = "round";
    chart.data.datasets.forEach((dataset, index) => {
      const points = chart.getDatasetMeta(index).data;
      const point = points[points.length - 1];
      if (!point) return;
      const value = dataset.data[dataset.data.length - 1].y.toFixed(2) + "×";
      const width = context.measureText(value).width;
      const x = Math.min(point.x + 6, chart.chartArea.right - width);
      context.strokeText(value, x, point.y - 8);
      context.fillStyle = css("--ink");
      context.fillText(value, x, point.y - 8);
    });
    context.restore();
  },
};

function renderCharts(data) {
  $("#legend").innerHTML = Object.values(MODELS).map(model => (
    '<span class="legend-row"><span class="swatch" style="--color:' + model.color + '"></span>' +
    escapeText(model.label + " · " + model.harness) + '</span>'
  )).join("");
  const box = $("#charts");
  for (const [scenario, info] of Object.entries(SCENARIOS)) {
    const runs = data.runs.filter(run => run.scenario === scenario);
    let card = document.getElementById("chart-" + scenario);
    if (!card) {
      card = document.createElement("div");
      card.className = "chart-card";
      card.id = "chart-" + scenario;
      card.innerHTML = '<h3></h3><p class="sub"></p><div class="chartbox"><canvas></canvas><div class="chart-empty" hidden></div></div>';
      box.append(card);
    }
    card.querySelector("h3").textContent = info.name;
    card.querySelector(".sub").textContent = "Best published agent at 2 h: " + info.published;
    const series = runs.filter(run => run.speedup_history?.length);
    const empty = card.querySelector(".chart-empty");
    empty.hidden = series.length > 0;
    if (!runs.length) {
      empty.textContent = data.phase === "smoke" ? "No smoke run in this scenario." : "No run launched yet.";
    } else if (!series.length) {
      const reasons = runs.map(run => run.speedup_note).filter(Boolean);
      empty.textContent = reasons.length ? "Waiting: " + reasons[0] + "."
        : "No evaluate.py result recorded yet. Results arrive when an attempt's log is written (after a pod failure or at the end).";
    }
    const maxHours = Math.max(...(runs.length ? runs.map(run => run.agent_seconds) : [57600])) / 3600;
    const peak = Math.max(2, ...series.flatMap(run => run.speedup_history.map(point => point[1])));
    const datasets = series.map(run => ({
      label: MODELS[run.key].label,
      data: run.speedup_history.map(([x, y]) => ({x, y})),
      showLine: true,
      stepped: "after",
      borderColor: MODELS[run.key].color,
      backgroundColor: MODELS[run.key].color,
      borderWidth: 2,
      pointRadius: 2.5,
      pointHoverRadius: 5,
    }));
    const options = {
      animation: false,
      maintainAspectRatio: false,
      layout: {padding: {right: 30, top: 4}},
      interaction: {mode: "nearest", axis: "x", intersect: false},
      scales: {
        x: {
          type: "linear", min: 0, max: maxHours,
          title: {display: true, text: "agent hours", color: css("--muted"), font: {size: 11}},
          grid: {display: false}, border: {color: css("--rule")},
          ticks: {color: css("--muted"), font: {size: 10}, maxTicksLimit: 9,
            callback: value => (maxHours < 1 ? Math.round(value * 60) + "m" : value + "h")},
        },
        y: {
          min: 0, suggestedMax: Math.ceil(peak * 1.15 * 2) / 2,
          title: {display: true, text: "speedup (×)", color: css("--muted"), font: {size: 11}},
          grid: {color: css("--grid")}, border: {display: false},
          ticks: {color: css("--muted"), font: {size: 10}, callback: value => value + "×"},
        },
      },
      plugins: {
        legend: {display: false},
        tooltip: {
          backgroundColor: "rgba(255,255,255,.97)", borderColor: css("--rule"), borderWidth: 1,
          titleColor: css("--ink"), bodyColor: css("--ink"), padding: 8, displayColors: true, boxWidth: 10, boxHeight: 2,
          callbacks: {
            title: items => items.length ? (maxHours < 1 ? Math.round(items[0].parsed.x * 60) + " agent minutes"
              : items[0].parsed.x.toFixed(2) + " agent hours") : "",
            label: item => item.parsed.y.toFixed(2) + "× · " + item.dataset.label,
          },
        },
      },
    };
    if (charts[scenario]) {
      charts[scenario].data.datasets = datasets;
      charts[scenario].options = options;
      charts[scenario].update();
    } else {
      charts[scenario] = new Chart(card.querySelector("canvas"), {
        type: "scatter", data: {datasets}, options, plugins: [budgetLine, endLabels],
      });
    }
  }
}

function fact(value, label) {
  return '<div><strong>' + escapeText(value) + '</strong><span>' + escapeText(label) + '</span></div>';
}

function card(run) {
  const model = MODELS[run.key];
  const used = agentUsed(run);
  const share = run.agent_seconds ? used / run.agent_seconds : 0;
  const left = Math.max(0, run.agent_seconds - used);
  const leftText = run.status === "running" ? clock(left) + " left"
    : run.status === "restoring" ? "paused at last checkpoint"
    : run.status === "preparing" ? "starts after setup" : "budget closed";
  const tokensNote = run.tokens && !run.tokens_live
    ? "Tokens and cost as recorded by the previous attempt's log (" + utc(run.tokens_as_of) + "); the live attempt adds to them when its log is written."
    : !run.tokens ? "Tokens and cost appear when an attempt's log is written." : "";
  let final = "";
  if (run.final) {
    const speedup = run.final.speedup === null ? "unscored" : run.final.speedup.toFixed(2) + "×";
    const gate = run.final.quality_pass === true ? "quality gate passed"
      : run.final.quality_pass === false ? "quality gate failed" : "quality gate not reached";
    final = '<div class="final">Final <strong>' + escapeText(speedup) + '</strong> · ' + escapeText(gate) +
      (run.final.note ? " · " + escapeText(run.final.note) : "") + '</div>';
  } else if (run.status === "errored") {
    final = '<div class="final">Ended with an error' + (run.error ? ": " + escapeText(run.error) : "") + '</div>';
  }
  const restoreNote = run.status === "restoring" && run.error
    ? "Attempt " + (run.attempt - 1) + " ended with " + run.error + "; Hawk is restoring the last checkpoint on a new pod." : "";
  const notes = [restoreNote, tokensNote, run.trace_note, run.stale_since ? "Hawk data unavailable since " + utc(run.stale_since) + "." : ""]
    .filter(Boolean).map(escapeText).join(" ");
  return (
    '<article class="card" style="--color:' + model.color + ';--logo:url(\'' + model.logo + '\')" data-run="' + escapeText(run.eval_set_id) + '">' +
      '<div class="card-head"><div class="who"><span class="logo" aria-hidden="true"></span><div><strong>' + escapeText(model.label) +
        '</strong><span>' + escapeText(model.harness + " · scenario " + run.scenario) + '</span></div></div>' +
        '<span class="pill" data-state="' + escapeText(run.status) + '">' + escapeText(STATES[run.status] || run.status) + '</span></div>' +
      '<div class="clock"><span><strong>' + clock(used) + '</strong> of ' + clock(run.agent_seconds) + '</span><span>' + escapeText(leftText) + '</span></div>' +
      '<div class="meter" role="img" aria-label="' + Math.round(share * 100) + '% of the agent budget used"><span style="width:' + (share * 100).toFixed(2) + '%"></span></div>' +
      '<div class="facts">' +
        fact(run.attempt ?? "n/a", "attempt") +
        fact(run.checkpoints ?? 0, run.last_checkpoint_at ? "checkpoints, last " + utc(run.last_checkpoint_at) : "checkpoints") +
        fact(run.restores ?? 0, "restores") +
        fact(tokens(run.tokens?.input), "input tokens") +
        fact(tokens(run.tokens?.cache_read), "cache read") +
        fact(tokens(run.tokens?.output), "output tokens") +
        fact(money(run.cost_usd), "model cost") +
        fact(run.grader_cost_usd === null || run.grader_cost_usd === undefined ? "n/a" : money(run.grader_cost_usd), "grader cost") +
        fact(run.model_calls ?? "n/a", "model calls") +
      '</div>' + final +
      (notes ? '<div class="note">' + notes + '</div>' : "") +
      '<div class="links">' +
        (run.log_url ? '<a href="' + escapeText(run.log_url) + '" target="_blank" rel="noopener">log, attempt ' + escapeText(run.attempt) + ' (may lag while a retry runs)</a>' : "") +
        '<a href="' + escapeText(run.eval_set_url) + '" target="_blank" rel="noopener">eval set</a>' +
      '</div>' +
    '</article>'
  );
}

function renderRuns(data) {
  const scenarios = data.phase === "smoke" ? [...new Set(data.runs.map(run => run.scenario))] : Object.keys(SCENARIOS);
  $("#runs").innerHTML = scenarios.map(scenario => {
    const runs = data.runs.filter(run => run.scenario === scenario);
    const cards = data.phase === "smoke" ? runs.map(card) : Object.keys(MODELS).map(key => {
      const run = runs.find(item => item.key === key);
      const model = MODELS[key];
      return run ? card(run) : '<article class="card empty-card" style="--color:' + model.color + '">' +
        escapeText(model.label + " · " + model.harness) + ': not launched yet</article>';
    });
    return '<section class="scenario-row"><h3>' + escapeText(SCENARIOS[scenario].name) + '</h3><div class="run-grid">' + cards.join("") + '</div></section>';
  }).join("");
}

function render(data) {
  latest = data;
  const age = (Date.now() - Date.parse(data.updated_at)) / 60000;
  $("#updated").textContent = "Live · data from " + utc(data.updated_at) + " · the collector refreshes it every " +
    Math.round(data.refresh_seconds / 60) + " minutes" + (age > 25 ? " · last refresh " + Math.round(age) + " minutes ago, check the collector" : "") +
    (data.hawk_ok ? "" : " · some Hawk data is stale this pass");
  const banner = $("#phase");
  banner.hidden = data.phase !== "smoke";
  banner.textContent = "The campaign has not launched yet. This page shows the two smoke runs (20-minute agent budgets) as test data.";
  renderTotals(data);
  renderCharts(data);
  renderRuns(data);
  const baselineNotes = Object.entries(data.baselines || {}).map(([scenario, source]) => scenario + ": " + source);
  $("#notes").textContent = baselineNotes.length
    ? "Chart baselines (scenarios without one wait for the first scored run): " + baselineNotes.join(" · ") : "";
}

async function refresh() {
  try {
    const response = await fetch(new URL("data.json?t=" + Date.now(), import.meta.url), {cache: "no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    render(await response.json());
  } catch (error) {
    if (!latest) $("#updated").textContent = "Data could not load. Please reload the page.";
  }
}

refresh();
setInterval(refresh, 120000);
setInterval(() => latest && renderRuns(latest), 30000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
