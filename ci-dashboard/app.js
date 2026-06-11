/* FlyDSL CI Performance Dashboard — data wiring + rendering (vanilla JS).
 * Data model: see ci-dashboard/data/schema.md
 *   history.json -> { records: [ {ts,commit,pr,run_id,source,runner,arch,op,shape,dtype,
 *                                  metric,value,status,vs_main,vs_tag,regression,extra} ] }
 *   runs.json    -> { runs:    [ {run_id,pr,commit,branch,event,title,status,conclusion,url,
 *                                  created_at,updated_at,actor,jobs:[{runner,arch,status,conclusion,url}]} ] }
 */
"use strict";

const CFG = {
  repo: "ROCm/FlyDSL",
  // Fresh data is published here by ci-dashboard/ingest (the ci-dashboard-data branch).
  dataBranch: "https://raw.githubusercontent.com/ROCm/FlyDSL/ci-dashboard-data/",
  bundled: "./data/",
  api: "https://api.github.com/repos/ROCm/FlyDSL",
  regressionPct: -3.0,
  warnPct: -1.0,
  localTolPct: 5.0,           // |local - CI| beyond this flags a cross-check disagreement
  archOrder: ["gfx950", "gfx942", "gfx1201"],
  archColor: { gfx950: "#2fe4cf", gfx942: "#6ea8ff", gfx1201: "#c08cf0" },
  archBox:   { gfx950: "MI350X", gfx942: "MI325", gfx1201: "Navi4" },
};

const S = {              // app state
  records: [], runs: [], local: [], updated: null, live: false,
  view: "board", boardFilter: "all",
  reg: { baseline: "vs_main", runner: "all", only: true, q: "", sort: "delta", asc: true },
  trend: { key: null, metric: null, q: "" },
};

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const kkey = r => `${r.op} ${r.shape} ${r.dtype}`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

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
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);   // TFLOPS
}
function toast(msg) {
  let t = $(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show"); clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ----------------------------------------------------------------- loading -- */
async function getJSON(url, timeout = 8000) {
  const c = new AbortController(); const id = setTimeout(() => c.abort(), timeout);
  try { const r = await fetch(url, { signal: c.signal, cache: "no-store" }); if (!r.ok) throw 0; return await r.json(); }
  catch { return null; } finally { clearTimeout(id); }
}

async function loadAll() {
  // Prefer the live data branch; fall back to the bundled seed. Newest `updated` wins.
  const [hb, hs] = await Promise.all([getJSON(CFG.dataBranch + "history.json"), getJSON(CFG.bundled + "history.json")]);
  const [rb, rs] = await Promise.all([getJSON(CFG.dataBranch + "runs.json"),    getJSON(CFG.bundled + "runs.json")]);
  const [lb, ls] = await Promise.all([getJSON(CFG.dataBranch + "local.json"),   getJSON(CFG.bundled + "local.json")]);
  const newer = (a, b) => (a && (!b || (a.updated || "") >= (b.updated || ""))) ? a : b;
  const hist = newer(hb, hs) || { records: [] };
  const runs = newer(rb, rs) || { runs: [] };
  const local = newer(lb, ls) || { records: [] };
  S.records = hist.records || [];
  S.runs = runs.runs || [];
  S.local = local.records || [];
  S.updated = hist.updated || runs.updated || null;
  renderUpdated();
  renderAll();
  enhanceLiveBoard();   // best-effort live overlay
}

async function enhanceLiveBoard() {
  const live = await getJSON(`${CFG.api}/actions/runs?per_page=25`);
  if (!live || !live.workflow_runs) { setLive(false); return; }
  const wf = live.workflow_runs.filter(r => /fly\s*dsl\s*test/i.test(r.name || "") || /flydsl\.ya?ml/i.test(r.path || ""));
  const byId = new Map(S.runs.map(r => [r.run_id, r]));
  for (const r of wf) {
    const cur = byId.get(r.id);
    const merged = {
      run_id: r.id, pr: r.pull_requests?.[0]?.number ?? cur?.pr ?? null,
      commit: r.head_sha, branch: r.head_branch, event: r.event, title: r.display_title,
      status: r.status, conclusion: r.conclusion, url: r.html_url,
      created_at: r.created_at, updated_at: r.updated_at, actor: r.actor?.login,
      jobs: cur?.jobs || [],
    };
    byId.set(r.id, merged);
  }
  S.runs = [...byId.values()].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  // For up to 5 most-recent active runs lacking job chips, fetch jobs (rate-limit-bounded).
  const active = S.runs.filter(r => r.status !== "completed" || !r.jobs?.length).slice(0, 5);
  await Promise.all(active.map(async r => {
    const j = await getJSON(`${CFG.api}/actions/runs/${r.run_id}/jobs?per_page=100`);
    if (!j || !j.jobs) return;
    r.jobs = j.jobs.filter(x => /linux-flydsl-(mi355|mi325|navi)/.test(x.name))
      .map(x => ({ runner: (x.name.match(/\((linux-flydsl-[^)]+)\)/) || [])[1],
        arch: archOf(x.name), status: x.status, conclusion: x.conclusion, url: x.html_url }));
  }));
  setLive(true);
  if (S.view === "board") renderBoard();
}
function archOf(name) {
  if (/mi355/.test(name)) return "gfx950"; if (/mi325/.test(name)) return "gfx942";
  if (/navi/.test(name)) return "gfx1201"; return "unknown";
}
function setLive(on) {
  S.live = on; const el = $("#liveDot");
  el.classList.toggle("stale", !on);
  el.title = on ? "Live status from the GitHub Actions API" : "Live API unavailable — showing last snapshot";
}
function renderUpdated() { $("#updated").textContent = S.updated ? `snapshot ${relTime(S.updated)}` : "no data"; }

/* -------------------------------------------------------------- 1 · BOARD -- */
function chipState(j) {
  if (!j) return "none";
  if (j.status && j.status !== "completed") return "running";
  return j.conclusion || "none";
}
function renderBoard() {
  const grid = $("#boardGrid");
  // newest run per PR (or per main branch commit)
  const groups = new Map();
  for (const r of S.runs) {
    const key = r.pr ? `pr:${r.pr}` : `br:${r.branch}:${r.commit}`;  // one card per PR, or per branch-commit
    const ex = groups.get(key);
    if (!ex || (r.created_at || "") > (ex.created_at || "")) groups.set(key, r);
  }
  let list = [...groups.values()];
  const f = S.boardFilter;
  if (f === "pr") list = list.filter(r => r.pr);
  else if (f === "main") list = list.filter(r => !r.pr && r.branch === "main");
  else if (f === "active") list = list.filter(r => r.status !== "completed" || (r.jobs || []).some(j => j.status !== "completed"));
  list.sort((a, b) => {
    const act = r => (r.status !== "completed" ? 0 : 1);
    return act(a) - act(b) || (b.created_at || "").localeCompare(a.created_at || "");
  });

  // counts
  const jobs = list.flatMap(r => r.jobs || []);
  const running = jobs.filter(j => j.status !== "completed").length;
  const pass = jobs.filter(j => j.conclusion === "success").length;
  const fail = jobs.filter(j => j.conclusion === "failure").length;
  $("#boardCounts").innerHTML =
    `<span class="c-run"><b>${running}</b>running</span>` +
    `<span class="c-pass"><b>${pass}</b>passed</span>` +
    `<span class="c-fail"><b>${fail}</b>failed</span>` +
    `<span><b>${list.length}</b><i>runs</i></span>`;

  if (!list.length) { grid.innerHTML = `<div class="empty">no recent runs in snapshot</div>`; return; }
  grid.innerHTML = list.map(r => {
    const byRunner = {}; for (const j of (r.jobs || [])) byRunner[j.arch] = j;
    const overall = (r.status !== "completed") ? "running" : (r.conclusion || "none");
    const chips = CFG.archOrder.map(arch => {
      const j = byRunner[arch]; const st = chipState(j);
      const label = st === "running" ? "run" : st === "success" ? "pass" : st === "failure" ? "fail"
        : st === "cancelled" ? "cncl" : st === "none" ? "—" : st.slice(0, 4);
      const link = j?.url ? `<a href="${esc(j.url)}" target="_blank" rel="noopener" title="${arch} · ${st}"></a>` : "";
      return `<div class="chip" data-c="${esc(st)}"><span class="arch">${esc(arch)}</span><span class="st">${esc(label)}</span>${link}</div>`;
    }).join("");
    const who = r.pr ? `#${r.pr}` : esc(r.branch || "—");
    return `<div class="pr-card ${overall}">
      <div class="pr-top"><span class="pr-num">${who}</span><span class="pr-event">${esc(r.event || "")}</span>
        <span class="spacer"></span><span class="pr-meta" style="margin:0">${relTime(r.created_at)}</span></div>
      <a class="pr-title" href="${esc(r.url || "#")}" target="_blank" rel="noopener" style="text-decoration:none;color:inherit;display:block">${esc(r.title || r.branch || "")}</a>
      <div class="pr-meta"><span><b>@${esc(r.actor || "?")}</b></span><span>${esc((r.commit || "").slice(0, 7))}</span><span>${esc(r.branch || "")}</span></div>
      <div class="chips">${chips}</div></div>`;
  }).join("");
}

/* --------------------------------------------------------- 2 · REGRESSIONS -- */
function regRows() {
  const base = S.reg.baseline;
  let rows = S.records.filter(r => r.source === "ci" && r[base] && r.metric !== "speedup" && r.value != null);
  // keep only the latest run per (runner, kernel) so the table is "current state"
  const latest = new Map();
  for (const r of rows) {
    const k = `${r.runner}|${kkey(r)}`; const ex = latest.get(k);
    if (!ex || (r.ts || "") > (ex.ts || "")) latest.set(k, r);
  }
  rows = [...latest.values()];
  if (S.reg.runner !== "all") rows = rows.filter(r => r.arch === S.reg.runner);
  if (S.reg.only) rows = rows.filter(r => r[base].delta_pct <= CFG.warnPct);
  if (S.reg.q) { const q = S.reg.q.toLowerCase(); rows = rows.filter(r => kkey(r).toLowerCase().includes(q)); }
  const get = r => ({ op: r.op, shape: r.shape, dtype: r.dtype, runner: r.arch, metric: r.metric,
    value: r.value, baseline: r[base].baseline, delta: r[base].delta_pct, ts: r.ts });
  const s = S.reg.sort, dir = S.reg.asc ? 1 : -1;
  rows.sort((a, b) => {
    const x = get(a)[s], y = get(b)[s];
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * dir;
  });
  return rows;
}
function sevClass(d) { return d <= CFG.regressionPct ? "reg" : d < 0 ? "neg" : "pos"; }
function renderRegress() {
  const base = S.reg.baseline;
  const rows = regRows();
  // badge counts regressions for the *selected* baseline, not just vs_main
  const all = S.records.filter(r => r.source === "ci" && r[base] && r.metric !== "speedup");
  const nReg = new Set(
    all.filter(r => r[base].delta_pct <= CFG.regressionPct).map(r => `${r.runner}|${kkey(r)}|${r.run_id}`)
  ).size;
  $("#regBadge").hidden = !nReg; $("#regBadge").textContent = nReg;
  $("#regSummary").textContent = `${rows.length} rows · gate ${CFG.regressionPct}%`;
  const maxAbs = Math.max(8, ...rows.map(r => Math.abs(r[base].delta_pct)));
  $("#regBody").innerHTML = rows.map(r => {
    const d = r[base].delta_pct, sc = sevClass(d);
    const w = Math.min(46, Math.abs(d) / maxAbs * 46);
    const col = sc === "reg" ? "var(--bad)" : sc === "neg" ? "var(--warn)" : "var(--good)";
    const sha = (r.commit || "").slice(0, 7);
    return `<tr class="${sc === "reg" ? "row-reg" : ""}">
      <td class="k-op">${esc(r.op)}</td><td class="k-dim">${esc(r.shape)}</td><td class="k-acc">${esc(r.dtype)}</td>
      <td style="color:${CFG.archColor[r.arch]}">${esc(r.arch)}</td>
      <td class="metric-tag">${r.metric}</td>
      <td class="num">${fmtVal(r.value, r.metric)}</td>
      <td class="num k-dim">${fmtVal(r[base].baseline, r.metric)}</td>
      <td class="num delta ${sc}">${d > 0 ? "+" : ""}${d.toFixed(1)}<span class="delta-bar" style="width:${w}px;background:${col}"></span></td>
      <td><a class="commit-link" href="https://github.com/${CFG.repo}/actions/runs/${r.run_id}" target="_blank" rel="noopener">${r.pr ? "#" + r.pr : "main"}·${sha}</a></td>
    </tr>`;
  }).join("") || `<tr><td colspan="9" class="empty">no rows — try turning off “regressions only”</td></tr>`;
}

/* -------------------------------------------------------------- 3 · TRENDS -- */
function kernelIndex() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.value == null) continue;
    const k = kkey(r); if (!m.has(k)) m.set(k, { op: r.op, shape: r.shape, dtype: r.dtype, metrics: new Set(), recs: [], reg: false });
    const e = m.get(k); e.metrics.add(r.metric); e.recs.push(r); if (r.regression) e.reg = true;
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
  }).join("") || `<div class="empty" style="padding:24px">no match</div>`;
}
let trendChart = null;
function selectKernel(k, rerail = true) {
  S.trend.key = k;
  const idx = kernelIndex(); const e = idx.get(k); if (!e) return;
  const metrics = [...e.metrics];
  if (!metrics.includes(S.trend.metric)) S.trend.metric = metrics.find(m => m !== "speedup") || metrics[0];
  $("#metricSel").innerHTML = metrics.map(m => `<button data-m="${m}" class="${m === S.trend.metric ? "is-active" : ""}">${m}</button>`).join("");
  $("#trendTitle").innerHTML = `${esc(e.op)} <small>${esc(e.shape)} · ${esc(e.dtype)} · ${S.trend.metric}</small>`;
  if (rerail) $$("#kernelList .kitem").forEach(b => b.classList.toggle("is-active", b.dataset.k === k));
  drawTrend(e);
}
function drawTrend(e) {
  const metric = S.trend.metric;
  const recs = e.recs.filter(r => r.metric === metric);
  const runIds = [...new Set(recs.map(r => r.run_id))]
    .map(id => ({ id, ts: recs.find(r => r.run_id === id).ts, commit: recs.find(r => r.run_id === id).commit, pr: recs.find(r => r.run_id === id).pr }))
    .sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  const labels = runIds.map(r => (r.commit || "").slice(0, 7));
  const datasets = CFG.archOrder.map(arch => {
    const col = CFG.archColor[arch];
    const data = runIds.map(ri => { const r = recs.find(x => x.run_id === ri.id && x.arch === arch); return r ? r.value : null; });
    if (data.every(v => v == null)) return null;
    return { label: arch, data, borderColor: col, backgroundColor: col + "22", pointBackgroundColor: col,
      pointRadius: 3, pointHoverRadius: 5, borderWidth: 2, tension: .25, spanGaps: true,
      segment: { borderColor: ctx => regSeg(recs, runIds, arch, ctx) ? "#ff5d6c" : col } };
  }).filter(Boolean);

  const ctx = $("#trendChart");
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 280 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#93a0b4", font: { family: "IBM Plex Mono", size: 11 }, boxWidth: 10, usePointStyle: true } },
        tooltip: { backgroundColor: "#0b0f17", borderColor: "#283242", borderWidth: 1, titleColor: "#e8eef6",
          bodyColor: "#93a0b4", titleFont: { family: "IBM Plex Mono" }, bodyFont: { family: "IBM Plex Mono" },
          callbacks: { title: items => { const ri = runIds[items[0].dataIndex]; return `${(ri.commit || "").slice(0, 7)} · ${ri.pr ? "#" + ri.pr : "main"}`; },
            label: i => ` ${i.dataset.label}: ${fmtVal(i.parsed.y, metric)} ${metric}` } },
      },
      scales: {
        x: { grid: { color: "#1b2230" }, ticks: { color: "#5a6678", font: { family: "IBM Plex Mono", size: 10 } } },
        y: { grid: { color: "#1b2230" }, ticks: { color: "#5a6678", font: { family: "IBM Plex Mono", size: 10 } },
          title: { display: true, text: metric, color: "#5a6678", font: { family: "IBM Plex Mono", size: 10 } } },
      },
    },
  });

  $("#trendBody").innerHTML = runIds.slice().reverse().map(ri => {
    const cell = arch => { const r = recs.find(x => x.run_id === ri.id && x.arch === arch);
      return r ? `<td class="num" style="color:${CFG.archColor[arch]}">${fmtVal(r.value, metric)}</td>` : `<td class="num k-dim">—</td>`; };
    return `<tr><td class="k-acc">${(ri.commit || "").slice(0, 7)}</td><td class="k-dim">${(ri.ts || "").slice(0, 10)}</td>
      ${cell("gfx950")}${cell("gfx942")}${cell("gfx1201")}</tr>`;
  }).join("");
}
function regSeg(recs, runIds, arch, ctx) {
  const ri = runIds[ctx.p1DataIndex]; if (!ri) return false;
  const r = recs.find(x => x.run_id === ri.id && x.arch === arch);
  return r && r.regression;
}

/* --------------------------------------------------------------- 4 · LOCAL -- */
function renderLocal() {
  const pane = $("#localPane");
  if (!S.local.length) {
    $("#localCounts").innerHTML = `<span><b>0</b><i>local samples</i></span>`;
    pane.innerHTML = `<div class="local-note">
      <h3>No local gfx950 data yet</h3>
      <p>Cross-check upstream against your own MI350X box. From a built FlyDSL tree, run the helper —
      it benchmarks locally, parses with the same parser CI uses, and pushes results to the
      <code>ci-dashboard-data</code> branch tagged <code>source=local-gfx950</code>:</p>
      <pre>bash scripts/ci_dashboard_local.sh --push</pre>
      <p style="margin-top:14px">Once pushed, this tab shows your local numbers beside the latest CI gfx950 result and
      flags any kernel where they disagree by more than ${CFG.localTolPct}%.</p></div>`;
    return;
  }
  // latest local per kernel
  const loc = new Map();
  for (const r of S.local) { const k = kkey(r); const ex = loc.get(k); if (!ex || (r.ts || "") > (ex.ts || "")) loc.set(k, r); }
  // latest CI per (arch, kernel) — match each local record against CI of the same arch
  const ci = new Map();
  for (const r of S.records) { if (r.source !== "ci" || r.value == null) continue;
    const k = `${r.arch}|${kkey(r)}`; const ex = ci.get(k); if (!ex || (r.ts || "") > (ex.ts || "")) ci.set(k, r); }
  let disagree = 0;
  const rows = [...loc.values()].map(l => {
    const c = ci.get(`${l.arch}|${kkey(l)}`); const cv = c?.value ?? null;
    const d = (cv && l.value) ? (l.value - cv) / cv * 100 : null;
    const bad = d != null && Math.abs(d) > CFG.localTolPct; if (bad) disagree++;
    return { l, cv, d, bad };
  }).sort((a, b) => (b.d == null ? -1 : Math.abs(b.d)) - (a.d == null ? -1 : Math.abs(a.d)));
  $("#localCounts").innerHTML =
    `<span><b>${rows.length}</b><i>kernels</i></span><span class="c-fail"><b>${disagree}</b>disagree &gt;${CFG.localTolPct}%</span>`;
  pane.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr>
    <th>kernel</th><th>shape</th><th>dtype</th><th>arch</th><th class="num">local</th><th class="num">CI</th><th class="num">Δ%</th><th>local run</th>
    </tr></thead><tbody>${rows.map(({ l, cv, d, bad }) => `<tr class="${bad ? "row-reg" : ""}">
      <td class="k-op">${esc(l.op)}</td><td class="k-dim">${esc(l.shape)}</td><td class="k-acc">${esc(l.dtype)}</td>
      <td style="color:${CFG.archColor[l.arch] || "var(--ink-2)"}">${esc(l.arch || "")}</td>
      <td class="num">${fmtVal(l.value, l.metric)} <span class="metric-tag">${l.metric}</span></td>
      <td class="num k-dim">${cv == null ? "—" : fmtVal(cv, l.metric)}</td>
      <td class="num ${bad ? "disagree" : "agree"}">${d == null ? "—" : (d > 0 ? "+" : "") + d.toFixed(1)}</td>
      <td class="k-dim">${esc((l.commit || "").slice(0, 7))} ${relTime(l.ts)}</td></tr>`).join("")}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ render -- */
function renderAll() { renderBoard(); renderRegress(); renderKernelRail(); renderLocal(); }
const VIEWS = ["board", "regress", "trends", "local"];
function showView(v) {
  if (!VIEWS.includes(v)) v = "board";
  S.view = v;
  $$(".tab").forEach(t => {
    const on = t.dataset.view === v;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  $$(".view").forEach(s => s.classList.toggle("is-active", s.dataset.view === v));
  if (location.hash.slice(1) !== v) history.replaceState(null, "", "#" + v);
  if (v === "trends" && trendChart) trendChart.resize();
}

/* ------------------------------------------------------------------ events -- */
function wire() {
  $("#tabs").addEventListener("click", e => { const t = e.target.closest(".tab"); if (t) showView(t.dataset.view); });
  document.addEventListener("keydown", e => {
    if (e.target.matches("input")) return;
    const map = { 1: "board", 2: "regress", 3: "trends", 4: "local" };
    if (map[e.key]) showView(map[e.key]);
    if (e.key.toLowerCase() === "r") doRefresh();
  });
  $("#refresh").addEventListener("click", doRefresh);

  $("#boardFilter").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
    S.boardFilter = b.dataset.f; $$("#boardFilter button").forEach(x => x.classList.toggle("is-active", x === b)); renderBoard(); });

  $("#baselineSel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
    S.reg.baseline = b.dataset.b; $$("#baselineSel button").forEach(x => x.classList.toggle("is-active", x === b)); renderRegress(); });
  $("#runnerSel").innerHTML = `<button data-r="all" class="is-active">all</button>` +
    CFG.archOrder.map(a => `<button data-r="${a}">${a}</button>`).join("");
  $("#runnerSel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
    S.reg.runner = b.dataset.r; $$("#runnerSel button").forEach(x => x.classList.toggle("is-active", x === b)); renderRegress(); });
  $("#regOnly").addEventListener("change", e => { S.reg.only = e.target.checked; renderRegress(); });
  $("#regSearch").addEventListener("input", e => { S.reg.q = e.target.value; renderRegress(); });
  $("#regTable thead").addEventListener("click", e => { const th = e.target.closest("th"); if (!th) return;
    const s = th.dataset.sort; if (S.reg.sort === s) S.reg.asc = !S.reg.asc; else { S.reg.sort = s; S.reg.asc = (s !== "delta"); }
    $$("#regTable thead th").forEach(x => { x.classList.toggle("is-sorted", x === th); x.classList.toggle("asc", x === th && S.reg.asc); });
    renderRegress(); });

  $("#kernelSearch").addEventListener("input", e => { S.trend.q = e.target.value; renderKernelRail(); });
  $("#kernelList").addEventListener("click", e => { const b = e.target.closest(".kitem"); if (b) selectKernel(b.dataset.k); });
  $("#metricSel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
    S.trend.metric = b.dataset.m; selectKernel(S.trend.key); });
}
let refreshing = false;
async function doRefresh() {
  if (refreshing) return; refreshing = true;
  $("#refresh").classList.add("spin");
  await loadAll();
  $("#refresh").classList.remove("spin"); refreshing = false;
  toast("data reloaded");
}

window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
document.addEventListener("DOMContentLoaded", () => {
  wire();
  if (VIEWS.includes(location.hash.slice(1))) showView(location.hash.slice(1));
  loadAll();
  setInterval(enhanceLiveBoard, 90000);
});
