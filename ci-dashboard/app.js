/* FlyDSL CI Performance Dashboard — regression-first. See data/schema.md.
 * Landing = "is main regressing?"; PR Check = "does this PR regress vs main?";
 * Trends = per-kernel series with a run-to-run noise band; Board / Local are secondary.
 */
"use strict";

const CFG = {
  repo: "ROCm/FlyDSL",
  dataBranch: "https://raw.githubusercontent.com/ROCm/FlyDSL/ci-dashboard-data/",
  bundled: "./data/",
  api: "https://api.github.com/repos/ROCm/FlyDSL",
  regressionPct: -3.0,   // fixed gate
  warnPct: -1.0,         // surfaced as "watch"
  noiseK: 2.0,           // a drop must exceed K * (run-to-run relative std) to count as real
  minSamples: 4,         // need at least this many main runs to trust a noise estimate
  archOrder: ["gfx950", "gfx942", "gfx1201"],
  archColor: { gfx950: "#4aa8ff", gfx942: "#b389e6", gfx1201: "#e0934a" },
};

const S = {
  records: [], runs: [], updated: null, runMeta: new Map(),
  view: "health", noiseAware: true, boardFilter: "all",
  pr: { sel: null },
  trend: { key: null, arch: "all", metric: null, q: "" },
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const kkey = r => `${r.op} ${r.shape} ${r.dtype}`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const VIEWS = ["health", "prcheck", "trends", "board"];

function relTime(iso) {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 60) return `${m | 0}m ago`;
  if (h < 24) return `${h | 0}h ago`;
  if (d < 30) return `${d | 0}d ago`;
  return new Date(iso).toISOString().slice(0, 10);
}
function fmtVal(v, metric) {
  if (v == null) return "—";
  if (metric === "speedup") return v.toFixed(2) + "×";
  if (metric === "TB/s") return v.toFixed(3);
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}
function fmtPct(d) { return (d > 0 ? "+" : "") + d.toFixed(1) + "%"; }
function toast(msg) {
  let t = $(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show"); clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2400);
}

/* ----------------------------------------------------------------- loading -- */
async function getJSON(url, timeout = 8000) {
  const c = new AbortController(); const id = setTimeout(() => c.abort(), timeout);
  try { const r = await fetch(url, { signal: c.signal, cache: "no-store" }); if (!r.ok) throw 0; return await r.json(); }
  catch { return null; } finally { clearTimeout(id); }
}
async function loadAll() {
  const [hb, hs] = await Promise.all([getJSON(CFG.dataBranch + "history.json"), getJSON(CFG.bundled + "history.json")]);
  const [rb, rs] = await Promise.all([getJSON(CFG.dataBranch + "runs.json"), getJSON(CFG.bundled + "runs.json")]);
  const newer = (a, b) => (a && (!b || (a.updated || "") >= (b.updated || ""))) ? a : b;
  const hist = newer(hb, hs) || { records: [] };
  const runs = newer(rb, rs) || { runs: [] };
  S.records = hist.records || [];
  S.runs = runs.runs || [];
  S.runMeta = new Map(S.runs.map(r => [r.run_id, r]));
  S.updated = hist.updated || runs.updated || null;
  $("#updated").textContent = S.updated ? `snapshot ${relTime(S.updated)}` : "no data";
  renderAll();
  enhanceLiveBoard();
}
async function enhanceLiveBoard() {
  const live = await getJSON(`${CFG.api}/actions/runs?per_page=25`);
  if (!live || !live.workflow_runs) return;
  const wf = live.workflow_runs.filter(r => /fly\s*dsl\s*test/i.test(r.name || ""));
  const byId = new Map(S.runs.map(r => [r.run_id, r]));
  for (const r of wf) {
    const cur = byId.get(r.id);
    byId.set(r.id, {
      run_id: r.id, pr: r.pull_requests?.[0]?.number ?? cur?.pr ?? null, commit: r.head_sha,
      branch: r.head_branch, event: r.event, title: r.display_title, status: r.status, conclusion: r.conclusion,
      url: r.html_url, created_at: r.created_at, updated_at: r.updated_at, actor: r.actor?.login, jobs: cur?.jobs || [],
    });
  }
  S.runs = [...byId.values()].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  S.runMeta = new Map(S.runs.map(r => [r.run_id, r]));
  const active = S.runs.filter(r => r.status !== "completed" || !r.jobs?.length).slice(0, 5);
  await Promise.all(active.map(async r => {
    const j = await getJSON(`${CFG.api}/actions/runs/${r.run_id}/jobs?per_page=100`);
    if (!j || !j.jobs) return;
    r.jobs = j.jobs.filter(x => /linux-flydsl-(mi355|mi325|navi)/.test(x.name)).map(x => ({
      runner: (x.name.match(/\((linux-flydsl-[^)]+)\)/) || [])[1], arch: archOf(x.name),
      status: x.status, conclusion: x.conclusion, url: x.html_url,
    }));
  }));
  if (S.view === "board") renderBoard();
}
function archOf(n) { return /mi355/.test(n) ? "gfx950" : /mi325/.test(n) ? "gfx942" : /navi/.test(n) ? "gfx1201" : "?"; }

/* ----------------------------------------------------------- noise model --- */
function isMainRec(r) { const m = S.runMeta.get(r.run_id); return m ? m.branch === "main" : r.pr == null; }

// chronological main-branch series for one kernel/arch/metric
function mainSeries(op, shape, dtype, arch, metric) {
  return S.records
    .filter(r => r.source === "ci" && r.op === op && r.shape === shape && r.dtype === dtype &&
      r.arch === arch && r.metric === metric && r.value != null && isMainRec(r))
    .sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
}
function noiseOf(values) {
  const n = values.length;
  if (n < 2) return { n, mean: values[0] ?? null, std: 0, relStd: null, lo: null, hi: null };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  const relStd = mean ? (std / mean) * 100 : null;
  return { n, mean, std, relStd, lo: mean - CFG.noiseK * std, hi: mean + CFG.noiseK * std };
}
// "real" regression: drop beyond max(|gate|, K*relStd). With too few samples, just the gate.
function realRegression(deltaPct, noise) {
  const thr = (S.noiseAware && noise.n >= CFG.minSamples && noise.relStd != null)
    ? Math.max(Math.abs(CFG.regressionPct), CFG.noiseK * noise.relStd) : Math.abs(CFG.regressionPct);
  return deltaPct <= -thr;
}
function sev(deltaPct, real) { return real ? "bad" : deltaPct <= CFG.warnPct ? "warn" : deltaPct > 0 ? "ok" : "flat"; }

function sparkline(values, noise, lastReal) {
  const W = 116, H = 30, pad = 3;
  if (values.length < 2) return `<svg class="spark" width="${W}" height="${H}"></svg>`;
  const lo = Math.min(...values, noise.lo ?? Infinity), hi = Math.max(...values, noise.hi ?? -Infinity);
  const span = hi - lo || 1;
  const x = i => pad + (i / (values.length - 1)) * (W - 2 * pad);
  const y = v => H - pad - ((v - lo) / span) * (H - 2 * pad);
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  let band = "";
  if (noise.lo != null && noise.relStd != null && noise.n >= CFG.minSamples) {
    const yh = y(noise.hi), yl = y(noise.lo);
    band = `<rect x="0" y="${yh.toFixed(1)}" width="${W}" height="${Math.max(1, (yl - yh)).toFixed(1)}" fill="var(--band)"/>`;
  }
  const lx = x(values.length - 1), ly = y(values[values.length - 1]);
  const dot = `<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="2.6" fill="${lastReal ? "var(--bad)" : "var(--ink-2)"}"/>`;
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${band}` +
    `<polyline points="${pts}" fill="none" stroke="${lastReal ? "var(--bad)" : "var(--ink-2)"}" stroke-width="1.3"/>${dot}</svg>`;
}

/* latest record per (kernel,arch) over main runs, with vs_main */
function latestMainByKernelArch() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.metric === "speedup" || !r.vs_main || r.value == null || !isMainRec(r)) continue;
    const k = `${r.arch}|${kkey(r)}`; const ex = m.get(k);
    if (!ex || (r.ts || "") > (ex.ts || "")) m.set(k, r);
  }
  return m;
}

/* ----------------------------------------------------------- 1 · HEALTH --- */
function renderHealth() {
  const latest = [...latestMainByKernelArch().values()];
  const rows = latest.map(r => {
    const series = mainSeries(r.op, r.shape, r.dtype, r.arch, r.metric);
    const vals = series.map(s => s.value);
    const noise = noiseOf(vals);
    const real = realRegression(r.vs_main.delta_pct, noise);
    return { r, vals, noise, real, d: r.vs_main.delta_pct, sev: sev(r.vs_main.delta_pct, real) };
  });
  const reals = rows.filter(x => x.real);
  const watch = rows.filter(x => !x.real && x.d <= CFG.warnPct);
  const list = [...reals, ...watch].sort((a, b) => a.d - b.d);

  // hero
  const card = $("#heroCard");
  const n = reals.length;
  card.className = "hero " + (n ? "alert" : "clear");
  $("#heroNum").textContent = n;
  $("#heroLabel").textContent = n ? `kernel regression${n > 1 ? "s" : ""} on main` : "main is clean";
  const lastRun = latest.reduce((a, r) => (r.ts || "") > a ? r.ts : a, "");
  const label = (latest.find(r => r.vs_main.label) || {}).vs_main?.label || "main";
  $("#heroNote").innerHTML =
    `gate <b>${CFG.regressionPct}%</b>${S.noiseAware ? ` or <b>${CFG.noiseK}×</b> run-to-run noise` : ""} · baseline <b>${esc(label)}</b>`;
  $("#heroStats").innerHTML =
    `<div class="stat"><span class="v">${latest.length}</span><span class="l">kernel × arch</span></div>` +
    `<div class="stat warn"><span class="v">${watch.length}</span><span class="l">within noise</span></div>` +
    `<div class="stat"><span class="v">${new Set(latest.map(r => r.arch)).size}</span><span class="l">arches</span></div>` +
    `<div class="stat"><span class="v" style="font-size:15px">${relTime(lastRun)}</span><span class="l">last run</span></div>`;
  const badge = $("#healthBadge"); badge.hidden = false; badge.textContent = n; badge.classList.toggle("zero", n === 0);
  $("#regHeadTitle").textContent = list.length ? `${reals.length} regressed · ${watch.length} watch` : "Regressions on main";

  // list
  if (!list.length) {
    $("#regList").innerHTML = `<div class="reg-list-empty"><div class="big">✓ all kernels within budget</div>` +
      `no kernel on main is slower than the gate or its noise band.</div>`;
    return;
  }
  $("#regList").innerHTML = list.map(({ r, vals, noise, real, d, sev }) => {
    const run = S.runMeta.get(r.run_id);
    const sha = (r.commit || "").slice(0, 7);
    const href = `https://github.com/${CFG.repo}/actions/runs/${r.run_id}`;
    return `<div class="reg-row" data-k="${esc(kkey(r))}" data-arch="${r.arch}">
      <span class="op">${esc(r.op)} <span class="metric-tag">${r.metric}</span></span>
      <span class="shape">${esc(r.shape)} · ${esc(r.dtype)}</span>
      <span class="reg-arch" style="color:${CFG.archColor[r.arch]}">${esc(r.arch)}</span>
      ${sparkline(vals, noise, real)}
      <span class="reg-delta ${sev}">${fmtPct(d)}</span>
      <span class="commit"><a href="${href}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${run?.branch === "main" ? "main" : "#" + (r.pr ?? "?")}·${sha}</a></span>
    </div>`;
  }).join("");
}

/* ---------------------------------------------------------- 2 · PR CHECK --- */
function prsWithData() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || !r.pr || r.metric === "speedup" || !r.vs_main) continue;
    if (!m.has(r.pr)) m.set(r.pr, { pr: r.pr, ts: r.ts, title: S.runMeta.get(r.run_id)?.title || "" });
    const e = m.get(r.pr); if ((r.ts || "") > (e.ts || "")) e.ts = r.ts;
  }
  return [...m.values()].sort((a, b) => b.pr - a.pr);
}
function renderPRSelect() {
  const prs = prsWithData();
  if (S.pr.sel == null && prs.length) S.pr.sel = prs[0].pr;
  $("#prSelect").innerHTML = prs.map(p =>
    `<option value="${p.pr}" ${p.pr === S.pr.sel ? "selected" : ""}>#${p.pr} — ${esc((p.title || "").slice(0, 60))}</option>`).join("")
    || `<option>no PR data</option>`;
}
function renderPRCheck() {
  renderPRSelect();
  const pr = S.pr.sel;
  const pane = $("#prPane");
  if (pr == null) { pane.innerHTML = `<div class="empty">no PR benchmark data in the snapshot yet</div>`; return; }
  // latest run of this PR
  const recs = S.records.filter(r => r.source === "ci" && r.pr === pr && r.vs_main && r.metric !== "speedup" && r.value != null);
  const latestRun = recs.reduce((a, r) => (r.ts || "") > (a.ts || "") ? r : a, { ts: "" }).run_id;
  const cur = recs.filter(r => r.run_id === latestRun);
  const rows = cur.map(r => {
    const noise = noiseOf(mainSeries(r.op, r.shape, r.dtype, r.arch, r.metric).map(s => s.value));
    const real = realRegression(r.vs_main.delta_pct, noise);
    return { r, real, d: r.vs_main.delta_pct, sev: sev(r.vs_main.delta_pct, real) };
  }).sort((a, b) => a.d - b.d);
  const nbad = rows.filter(x => x.real).length;
  const nwatch = rows.filter(x => !x.real && x.d <= CFG.warnPct).length;
  $("#prSummary").innerHTML = `${rows.length} kernels · <b style="color:${nbad ? "var(--bad)" : "var(--good)"}">${nbad} real regression${nbad === 1 ? "" : "s"}</b> · ${nwatch} watch`;
  const runUrl = S.runMeta.get(latestRun)?.url || `https://github.com/${CFG.repo}/actions/runs/${latestRun}`;
  pane.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr>
    <th>kernel</th><th>shape</th><th>dtype</th><th>arch</th><th class="num">PR</th><th class="num">main</th><th class="num">Δ vs main</th><th>baseline</th>
    </tr></thead><tbody>${rows.map(({ r, d, sev }) => `<tr class="${sev === "bad" ? "row-bad" : ""}">
      <td>${esc(r.op)} <span class="metric-tag">${r.metric}</span></td><td class="k-dim">${esc(r.shape)}</td><td>${esc(r.dtype)}</td>
      <td style="color:${CFG.archColor[r.arch]}">${esc(r.arch)}</td>
      <td class="num">${fmtVal(r.value, r.metric)}</td>
      <td class="num k-dim">${fmtVal(r.vs_main.baseline, r.metric)}</td>
      <td class="num delta ${sev}">${fmtPct(d)}</td>
      <td class="k-dim">${esc(r.vs_main.label || "main")}</td></tr>`).join("")
    || `<tr><td colspan="8" class="empty">no vs-main rows for this PR run</td></tr>`}</tbody></table>
    <div class="noise-note" style="padding:10px 12px">latest run of #${pr} · <a href="${esc(runUrl)}" target="_blank" rel="noopener">view on GitHub</a> · “real” = beyond the kernel’s run-to-run noise on main</div></div>`;
}

/* ------------------------------------------------------------ 3 · TRENDS --- */
function kernelIndex() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.value == null) continue;
    const k = kkey(r);
    if (!m.has(k)) m.set(k, { op: r.op, shape: r.shape, dtype: r.dtype, metrics: new Set(), reg: false });
    const e = m.get(k); e.metrics.add(r.metric);
    if (r.regression && isMainRec(r)) e.reg = true;
  }
  return m;
}
function renderKernelRail() {
  const idx = kernelIndex();
  let keys = [...idx.keys()];
  if (S.trend.q) { const q = S.trend.q.toLowerCase(); keys = keys.filter(k => k.toLowerCase().includes(q)); }
  keys.sort();
  if (!S.trend.key && keys.length) selectKernel(keys.find(k => idx.get(k).reg) || keys[0], false);
  $("#kernelList").innerHTML = keys.map(k => {
    const e = idx.get(k);
    return `<button class="kitem ${k === S.trend.key ? "is-active" : ""} ${e.reg ? "has-reg" : ""}" data-k="${esc(k)}">
      ${esc(e.op)}<span class="ks">${esc(e.shape)} · ${esc(e.dtype)}</span></button>`;
  }).join("") || `<div class="empty" style="padding:20px">no match</div>`;
}
let trendChart = null;
function selectKernel(k, rerail = true) {
  S.trend.key = k;
  const idx = kernelIndex(); const e = idx.get(k); if (!e) return;
  const metrics = [...e.metrics];
  if (!metrics.includes(S.trend.metric)) S.trend.metric = metrics.find(m => m !== "speedup") || metrics[0];
  $("#metricSel").innerHTML = metrics.map(m => `<button data-m="${m}" class="${m === S.trend.metric ? "is-active" : ""}">${m}</button>`).join("");
  $("#trendArch").innerHTML = ["all", ...CFG.archOrder].map(a =>
    `<button data-a="${a}" class="${a === S.trend.arch ? "is-active" : ""}">${a}</button>`).join("");
  $("#trendTitle").innerHTML = `${esc(e.op)} <small>${esc(e.shape)} · ${esc(e.dtype)} · ${S.trend.metric}</small>`;
  if (rerail) $$("#kernelList .kitem").forEach(b => b.classList.toggle("is-active", b.dataset.k === k));
  drawTrend(e);
}
function drawTrend(e) {
  const metric = S.trend.metric, op = e.op, shape = e.shape, dtype = e.dtype;
  const recs = S.records.filter(r => r.source === "ci" && r.op === op && r.shape === shape && r.dtype === dtype && r.metric === metric);
  const runIds = [...new Set(recs.map(r => r.run_id))].map(id => {
    const any = recs.find(r => r.run_id === id);
    return { id, ts: any.ts, commit: any.commit, pr: any.pr, main: isMainRec(any) };
  }).sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  const labels = runIds.map(r => (r.commit || "").slice(0, 7));
  const single = S.trend.arch !== "all";
  const archs = single ? [S.trend.arch] : CFG.archOrder;

  const datasets = [];
  let note = "";
  if (single) {
    const noise = noiseOf(mainSeries(op, shape, dtype, S.trend.arch, metric).map(s => s.value));
    if (noise.lo != null && noise.relStd != null && noise.n >= CFG.minSamples) {
      datasets.push({ label: "+2σ", data: labels.map(() => noise.hi), borderColor: "transparent", pointRadius: 0, fill: "+1", backgroundColor: "rgba(150,162,178,.13)", order: 20 });
      datasets.push({ label: "-2σ", data: labels.map(() => noise.lo), borderColor: "transparent", pointRadius: 0, fill: false, order: 20 });
      datasets.push({ label: "main mean", data: labels.map(() => noise.mean), borderColor: "#626c79", borderDash: [4, 4], borderWidth: 1, pointRadius: 0, order: 19 });
      note = `noise band = main mean ${fmtVal(noise.mean, metric)} ± ${CFG.noiseK}σ (σ≈<b>${noise.relStd.toFixed(1)}%</b>, n=${noise.n}). Points below the band are real regressions.`;
    } else {
      note = `n=${noise.n} main runs — too few for a noise band; using the fixed <b>${CFG.regressionPct}%</b> gate.`;
    }
  } else {
    note = `one line per arch. Red points = drop beyond the kernel’s run-to-run noise on main (or the ${CFG.regressionPct}% gate).`;
  }
  for (const arch of archs) {
    const noise = noiseOf(mainSeries(op, shape, dtype, arch, metric).map(s => s.value));
    const data = runIds.map(ri => { const r = recs.find(x => x.run_id === ri.id && x.arch === arch); return r ? r.value : null; });
    if (data.every(v => v == null)) continue;
    const ptColor = runIds.map(ri => {
      const r = recs.find(x => x.run_id === ri.id && x.arch === arch);
      return (r && r.vs_main && realRegression(r.vs_main.delta_pct, noise)) ? "#f0616d" : CFG.archColor[arch];
    });
    datasets.push({
      label: arch, data, borderColor: CFG.archColor[arch], backgroundColor: CFG.archColor[arch] + "20",
      pointBackgroundColor: ptColor, pointBorderColor: ptColor, pointRadius: 3, pointHoverRadius: 5,
      borderWidth: 2, tension: .2, spanGaps: true, order: 1, fill: false,
    });
  }
  $("#noiseNote").innerHTML = note;

  if (trendChart) trendChart.destroy();
  trendChart = new Chart($("#trendChart"), {
    type: "line", data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 220 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#98a1ae", font: { family: "IBM Plex Mono", size: 11 }, boxWidth: 10, usePointStyle: true, filter: i => !/σ|mean/.test(i.text) } },
        tooltip: {
          backgroundColor: "#11161d", borderColor: "#2b333e", borderWidth: 1, titleColor: "#e8ebf0", bodyColor: "#98a1ae",
          titleFont: { family: "IBM Plex Mono" }, bodyFont: { family: "IBM Plex Mono" },
          filter: i => !/σ|mean/.test(i.dataset.label),
          callbacks: {
            title: items => { const ri = runIds[items[0].dataIndex]; return `${(ri.commit || "").slice(0, 7)} · ${ri.main ? "main" : "#" + ri.pr}`; },
            label: i => ` ${i.dataset.label}: ${fmtVal(i.parsed.y, metric)} ${metric}`,
          },
        },
      },
      scales: {
        x: { grid: { color: "#1a2029" }, ticks: { color: "#626c79", font: { family: "IBM Plex Mono", size: 10 } } },
        y: { grid: { color: "#1a2029" }, ticks: { color: "#626c79", font: { family: "IBM Plex Mono", size: 10 } },
          title: { display: true, text: metric, color: "#626c79", font: { family: "IBM Plex Mono", size: 10 } } },
      },
    },
  });
  // status-aware table
  $("#trendBody").innerHTML = runIds.slice().reverse().map(ri => {
    const cell = arch => {
      const r = recs.find(x => x.run_id === ri.id && x.arch === arch)
        || S.records.find(x => x.run_id === ri.id && x.arch === arch && x.op === op && x.shape === shape && x.dtype === dtype);
      if (!r) return `<td class="num st-na">—</td>`;
      if (r.value == null) return `<td class="num cell-status ${r.status === "skip" ? "st-skip" : "st-missing"}">${r.status}</td>`;
      const real = r.vs_main && realRegression(r.vs_main.delta_pct, noiseOf(mainSeries(op, shape, dtype, arch, metric).map(s => s.value)));
      return `<td class="num" style="color:${real ? "var(--bad)" : CFG.archColor[arch]}">${fmtVal(r.value, metric)}</td>`;
    };
    return `<tr><td>${(ri.commit || "").slice(0, 7)}</td><td class="k-dim">${(ri.ts || "").slice(0, 10)}</td>
      <td class="k-dim">${ri.main ? "main" : "#" + ri.pr}</td>${cell("gfx950")}${cell("gfx942")}${cell("gfx1201")}</tr>`;
  }).join("");
}

/* ------------------------------------------------------------- 4 · BOARD --- */
function chipState(j) { if (!j) return "none"; if (j.status && j.status !== "completed") return "running"; return j.conclusion || "none"; }
function worstDeltaForRun(runId) {
  let worst = null, kernel = null;
  for (const r of S.records) {
    if (r.run_id !== runId || !r.vs_main || r.metric === "speedup") continue;
    if (worst == null || r.vs_main.delta_pct < worst) { worst = r.vs_main.delta_pct; kernel = r.op; }
  }
  return worst == null ? null : { worst, kernel };
}
function renderBoard() {
  const grid = $("#boardGrid");
  const groups = new Map();
  for (const r of S.runs) {
    const key = r.pr ? `pr:${r.pr}` : `br:${r.branch}:${r.commit}`;
    const ex = groups.get(key);
    if (!ex || (r.created_at || "") > (ex.created_at || "")) groups.set(key, r);
  }
  let list = [...groups.values()];
  const f = S.boardFilter;
  if (f === "pr") list = list.filter(r => r.pr);
  else if (f === "main") list = list.filter(r => !r.pr && r.branch === "main");
  else if (f === "active") list = list.filter(r => r.status !== "completed" || (r.jobs || []).some(j => j.status !== "completed"));
  list.sort((a, b) => (a.status !== "completed" ? 0 : 1) - (b.status !== "completed" ? 0 : 1) || (b.created_at || "").localeCompare(a.created_at || ""));

  const jobs = list.flatMap(r => r.jobs || []);
  $("#boardCounts").innerHTML =
    `<span class="c-warn"><b>${jobs.filter(j => j.status !== "completed").length}</b>running</span>` +
    `<span class="c-good"><b>${jobs.filter(j => j.conclusion === "success").length}</b>passed</span>` +
    `<span class="c-bad"><b>${jobs.filter(j => j.conclusion === "failure").length}</b>failed</span>` +
    `<span><b>${list.length}</b><i>runs</i></span>`;

  if (!list.length) { grid.innerHTML = `<div class="empty">no recent runs in snapshot</div>`; return; }
  grid.innerHTML = list.map(r => {
    const byRunner = {}; for (const j of (r.jobs || [])) byRunner[j.arch] = j;
    const overall = (r.status !== "completed") ? "running" : (r.conclusion || "none");
    const chips = CFG.archOrder.map(arch => {
      const j = byRunner[arch]; const st = chipState(j);
      const label = st === "running" ? "run" : st === "success" ? "pass" : st === "failure" ? "fail" : st === "none" ? "—" : st.slice(0, 4);
      const link = j?.url ? `<a href="${esc(j.url)}" target="_blank" rel="noopener" title="${arch} · ${st}"></a>` : "";
      return `<div class="chip" data-c="${esc(st)}"><span class="arch">${esc(arch)}</span><span class="st">${esc(label)}</span>${link}</div>`;
    }).join("");
    const wd = worstDeltaForRun(r.run_id);
    const wsev = wd == null ? "none" : wd.worst <= CFG.regressionPct ? "bad" : wd.worst <= CFG.warnPct ? "warn" : "ok";
    const perf = `<div class="pr-perf"><span class="lab">worst Δ vs main</span>` +
      (wd == null ? `<span class="worst none">no perf data</span>`
        : `<span class="worst ${wsev}">${fmtPct(wd.worst)}</span><span class="lab">${esc(wd.kernel)}</span>`) + `</div>`;
    const who = r.pr ? `#${r.pr}` : esc(r.branch || "—");
    return `<div class="pr-card ${overall}">
      <div class="pr-top"><span class="pr-num">${who}</span><span class="pr-event">${esc(r.event || "")}</span>
        <span class="spacer"></span><span class="pr-meta" style="margin:0">${relTime(r.created_at)}</span></div>
      <a class="pr-title" href="${esc(r.url || "#")}" target="_blank" rel="noopener" style="color:inherit">${esc(r.title || r.branch || "")}</a>
      <div class="pr-meta"><span>@${esc(r.actor || "?")}</span><span>${esc((r.commit || "").slice(0, 7))}</span></div>
      ${perf}<div class="chips">${chips}</div></div>`;
  }).join("");
}

/* ------------------------------------------------------------------ shell -- */
function renderAll() { renderHealth(); renderPRCheck(); renderKernelRail(); renderBoard(); }
function showView(v) {
  if (!VIEWS.includes(v)) v = "health";
  S.view = v;
  $$(".tab").forEach(t => { const on = t.dataset.view === v; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", on ? "true" : "false"); });
  $$(".view").forEach(s => s.classList.toggle("is-active", s.dataset.view === v));
  if (location.hash.slice(1) !== v) history.replaceState(null, "", "#" + v);
  if (v === "trends" && trendChart) trendChart.resize();
}
function goTrend(k, arch) { S.trend.key = null; S.trend.arch = arch || "all"; showView("trends"); selectKernel(k); }

function wire() {
  $("#tabs").addEventListener("click", e => { const t = e.target.closest(".tab"); if (t) showView(t.dataset.view); });
  document.addEventListener("keydown", e => {
    if (e.target.matches("input,select")) return;
    const map = { 1: "health", 2: "prcheck", 3: "trends", 4: "board" };
    if (map[e.key]) showView(map[e.key]);
    if (e.key.toLowerCase() === "r") doRefresh();
  });
  $("#refresh").addEventListener("click", doRefresh);
  $("#noiseAware").addEventListener("change", e => { S.noiseAware = e.target.checked; renderHealth(); if (S.trend.key) drawTrend(kernelIndex().get(S.trend.key)); });
  $("#regList").addEventListener("click", e => { const row = e.target.closest(".reg-row"); if (row) goTrend(row.dataset.k, row.dataset.arch); });
  $("#prSelect").addEventListener("change", e => { S.pr.sel = +e.target.value; renderPRCheck(); });
  $("#boardFilter").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.boardFilter = b.dataset.f; $$("#boardFilter button").forEach(x => x.classList.toggle("is-active", x === b)); renderBoard(); });
  $("#kernelSearch").addEventListener("input", e => { S.trend.q = e.target.value; renderKernelRail(); });
  $("#kernelList").addEventListener("click", e => { const b = e.target.closest(".kitem"); if (b) selectKernel(b.dataset.k); });
  $("#metricSel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.metric = b.dataset.m; selectKernel(S.trend.key); });
  $("#trendArch").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.arch = b.dataset.a; selectKernel(S.trend.key); });
}
let refreshing = false;
async function doRefresh() {
  if (refreshing) return; refreshing = true; $("#refresh").classList.add("spin");
  await loadAll(); $("#refresh").classList.remove("spin"); refreshing = false; toast("data reloaded");
}

window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
document.addEventListener("DOMContentLoaded", () => {
  wire();
  if (VIEWS.includes(location.hash.slice(1))) showView(location.hash.slice(1));
  loadAll();
  setInterval(enhanceLiveBoard, 90000);
});
