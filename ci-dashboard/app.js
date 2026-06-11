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
  minSamples: 3,         // prior main runs needed to size a noise band (else: low confidence)
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
  const up = $("#updated"); up.classList.remove("syncing");
  up.textContent = S.updated ? `snapshot ${relTime(S.updated)}` : "no data";
  renderAll();
  enhanceLiveBoard();
}
async function enhanceLiveBoard() {
  const live = await getJSON(`${CFG.api}/actions/runs?per_page=25`);
  if (!live || !live.workflow_runs) return;
  const wf = live.workflow_runs.filter(r => /fly\s*dsl\s*test/i.test(r.name || ""));
  const byId = new Map(S.runs.map(r => [r.run_id, r]));
  // The live API often returns pull_requests:[] for fork PRs; reuse the PR the ingest
  // snapshot already resolved for this head branch so new runs aren't shown as branch cards.
  const branchToPr = new Map();
  for (const r of S.runs) if (r.pr && r.branch) branchToPr.set(r.branch, r.pr);
  for (const r of wf) {
    const cur = byId.get(r.id);
    const pr = r.pull_requests?.[0]?.number ?? cur?.pr ?? branchToPr.get(r.head_branch) ?? null;
    byId.set(r.id, {
      run_id: r.id, pr, commit: r.head_sha,
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

// Baseline noise for "did main regress?": the main history EXCLUDING the latest main run.
// On push-to-main, flydsl.yaml rebuilds origin/main in the same job, so a main run's own
// vs_main is current-vs-itself (re-run variance) — useless as a historical signal. We instead
// compare the latest main value against prior main runs.
function mainBaseline(op, shape, dtype, arch, metric) {
  const series = mainSeries(op, shape, dtype, arch, metric);
  if (!series.length) return noiseOf([]);
  const latestRun = series[series.length - 1].run_id;
  return noiseOf(series.filter(s => s.run_id !== latestRun).map(s => s.value));
}
// Δ% and whether it is a real regression, given a prior-main baseline.
//  - main run:  value vs the prior-main mean (a true historical comparison)
//  - PR run:    vs_main is a real PR-commit-vs-main-commit diff, so use it directly
function regOf(r, base) {
  if (!r || r.value == null || r.metric === "speedup") return { d: null, real: false };
  if (isMainRec(r)) {
    if (base.mean == null) return { d: null, real: false, lowConf: true };
    const d = (r.value - base.mean) / base.mean * 100;
    const haveBand = base.n >= CFG.minSamples && base.relStd != null;
    // confident only when enough prior history sizes the band; otherwise it's a
    // raw-gate guess that can't be told apart from run-to-run noise.
    const real = S.noiseAware
      ? (haveBand && d <= -Math.max(Math.abs(CFG.regressionPct), CFG.noiseK * base.relStd))
      : d <= CFG.regressionPct;
    return { d, real, lowConf: !haveBand };
  }
  // PR run: vs_main is a real PR-commit-vs-main-commit diff in the same job — valid as-is.
  if (!r.vs_main) return { d: null, real: false };
  return { d: r.vs_main.delta_pct, real: realRegression(r.vs_main.delta_pct, base) };
}

let _sparkN = 0;
function sparkline(values, noise, lastReal) {
  const W = 128, H = 34, pad = 4;
  if (values.length < 2) return `<svg class="spark" width="${W}" height="${H}"></svg>`;
  const lo = Math.min(...values, noise.lo ?? Infinity), hi = Math.max(...values, noise.hi ?? -Infinity);
  const span = hi - lo || 1;
  const x = i => pad + (i / (values.length - 1)) * (W - 2 * pad);
  const y = v => H - pad - ((v - lo) / span) * (H - 2 * pad);
  const pts = values.map((v, i) => [x(i), y(v)]);
  const col = lastReal ? "#f65f6c" : "#9aa6b4";
  const gid = "sg" + (_sparkN++);
  const line = "M" + pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L");
  const area = `M${pts[0][0].toFixed(1)},${(H - pad).toFixed(1)} ` +
    pts.map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ") +
    ` L${pts[pts.length - 1][0].toFixed(1)},${(H - pad).toFixed(1)} Z`;
  let band = "";
  if (noise.lo != null && noise.relStd != null && noise.n >= CFG.minSamples) {
    const yh = y(noise.hi), yl = y(noise.lo);
    band = `<rect x="0" y="${yh.toFixed(1)}" width="${W}" height="${Math.max(1, yl - yh).toFixed(1)}" fill="rgba(154,166,180,.14)"/>`;
  }
  const [lx, ly] = pts[pts.length - 1];
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
    `<defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">` +
    `<stop offset="0" stop-color="${col}" stop-opacity="0.20"/><stop offset="1" stop-color="${col}" stop-opacity="0"/></linearGradient></defs>` +
    `${band}<path d="${area}" fill="url(#${gid})"/><path d="${line}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round"/>` +
    `<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="2.8" fill="${col}"/></svg>`;
}

/* latest main-run record per (kernel,arch) */
function latestMainByKernelArch() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.metric === "speedup" || r.value == null || !isMainRec(r)) continue;
    const k = `${r.arch}|${kkey(r)}`; const ex = m.get(k);
    if (!ex || (r.ts || "") > (ex.ts || "")) m.set(k, r);
  }
  return m;
}

/* one row per latest-main kernel/arch: latest value vs PRIOR main history */
function healthRows() {
  return [...latestMainByKernelArch().values()].map(r => {
    const noise = mainBaseline(r.op, r.shape, r.dtype, r.arch, r.metric);
    const vals = mainSeries(r.op, r.shape, r.dtype, r.arch, r.metric).map(s => s.value);
    const { d, real } = regOf(r, noise);
    return { r, vals, noise, d, real, sev: d == null ? "flat" : sev(d, real) };
  });
}

/* ----------------------------------------------------------- 1 · HEALTH --- */
function renderHealth() {
  const rows = healthRows();
  const reals = rows.filter(x => x.real);
  // "watch" = dropped past the gate but not confident (thin main history can't size the noise band)
  const watch = rows.filter(x => !x.real && x.d != null && x.d <= CFG.regressionPct);
  const list = [...reals, ...watch].sort((a, b) => a.d - b.d);

  // hero
  const card = $("#heroCard");
  const n = reals.length;
  card.className = "hero " + (n ? "alert" : "clear");
  $("#heroNum").textContent = n;
  const glyph = n
    ? `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2l6 11H2z"/><path d="M8 6.4v3.2M8 11.5v.01"/></svg>`
    : `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3.2L13 5"/></svg>`;
  $("#heroLabel").innerHTML = glyph + (n ? `confirmed regression${n > 1 ? "s" : ""} on main` : watch.length ? "no confirmed regressions" : "main is clean");
  const lastRun = rows.reduce((a, x) => (x.r.ts || "") > a ? x.r.ts : a, "");
  $("#heroNote").innerHTML =
    `latest main vs <b>prior main history</b> · confirmed = below the ${CFG.noiseK}σ noise band ` +
    `(needs ≥${CFG.minSamples} prior main runs)`;
  $("#heroStats").innerHTML =
    `<div class="stat"><span class="v">${rows.length}</span><span class="l">kernel × arch</span></div>` +
    `<div class="stat warn"><span class="v">${watch.length}</span><span class="l">to check</span></div>` +
    `<div class="stat"><span class="v">${new Set(rows.map(x => x.r.arch)).size}</span><span class="l">arches</span></div>` +
    `<div class="stat"><span class="v" style="font-size:15px">${relTime(lastRun)}</span><span class="l">last run</span></div>`;
  const badge = $("#healthBadge"); badge.hidden = false; badge.textContent = n; badge.classList.toggle("zero", n === 0);
  $("#regHeadTitle").textContent = list.length ? `${reals.length} confirmed · ${watch.length} to check` : "Regressions on main";

  // list
  if (!list.length) {
    $("#regList").innerHTML = `<div class="reg-list-empty"><div class="big">` +
      `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3.2L13 5"/></svg>` +
      `all kernels within budget</div>no kernel on main is slower than the gate or its noise band.</div>`;
    return;
  }
  const maxAbs = Math.max(6, ...list.map(x => Math.abs(x.d)));
  $("#regList").innerHTML = list.map(({ r, vals, noise, real, d, sev }, i) => {
    const run = S.runMeta.get(r.run_id);
    const sha = (r.commit || "").slice(0, 7);
    const href = `https://github.com/${CFG.repo}/actions/runs/${r.run_id}`;
    const w = Math.max(4, Math.min(46, Math.abs(d) / maxAbs * 46));
    const bc = sev === "bad" ? "var(--bad)" : sev === "warn" ? "var(--warn)" : "var(--good)";
    return `<div class="reg-row s-${sev}" style="--i:${i}" data-k="${esc(kkey(r))}" data-arch="${r.arch}">
      <span class="op">${esc(r.op)} <span class="metric-tag">${r.metric}</span></span>
      <span class="shape">${esc(r.shape)} · ${esc(r.dtype)}</span>
      <span class="reg-arch" style="color:${CFG.archColor[r.arch]}">${esc(r.arch)}</span>
      ${sparkline(vals, noise, real)}
      <span class="reg-delta ${sev}"><span class="dbar" style="width:${w}px;background:${bc}"></span>${fmtPct(d)}</span>
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
    const { d, real } = regOf(r, mainBaseline(r.op, r.shape, r.dtype, r.arch, r.metric));
    return { r, real, d, sev: sev(d, real) };
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
  const regKeys = new Set(healthRows().filter(x => x.real).map(x => kkey(x.r)));
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.value == null) continue;
    const k = kkey(r);
    if (!m.has(k)) m.set(k, { op: r.op, shape: r.shape, dtype: r.dtype, metrics: new Set(), reg: false });
    m.get(k).metrics.add(r.metric);
  }
  for (const [k, e] of m) e.reg = regKeys.has(k);
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
    const noise = mainBaseline(op, shape, dtype, S.trend.arch, metric);
    if (noise.lo != null && noise.relStd != null && noise.n >= CFG.minSamples) {
      datasets.push({ label: "+2σ", data: labels.map(() => noise.hi), borderColor: "transparent", pointRadius: 0, fill: "+1", backgroundColor: "rgba(150,162,178,.13)", order: 20 });
      datasets.push({ label: "-2σ", data: labels.map(() => noise.lo), borderColor: "transparent", pointRadius: 0, fill: false, order: 20 });
      datasets.push({ label: "main mean", data: labels.map(() => noise.mean), borderColor: "#626c79", borderDash: [4, 4], borderWidth: 1, pointRadius: 0, order: 19 });
      note = `band = prior-main mean ${fmtVal(noise.mean, metric)} ± ${CFG.noiseK}σ (σ≈<b>${noise.relStd.toFixed(1)}%</b>, n=${noise.n}). A main point below the band, or a PR below its main baseline, is a real regression.`;
    } else {
      note = `n=${noise.n} prior main runs — too few for a noise band; using the fixed <b>${CFG.regressionPct}%</b> gate.`;
    }
  } else {
    note = `one line per arch. Red = a main run below its prior-main noise band, or a PR run slower than main (beyond the ${CFG.regressionPct}% gate).`;
  }
  for (const arch of archs) {
    const noise = mainBaseline(op, shape, dtype, arch, metric);
    const data = runIds.map(ri => { const r = recs.find(x => x.run_id === ri.id && x.arch === arch); return r ? r.value : null; });
    if (data.every(v => v == null)) continue;
    const ptColor = runIds.map(ri => {
      const r = recs.find(x => x.run_id === ri.id && x.arch === arch);
      return (r && regOf(r, noise).real) ? "#f65f6c" : CFG.archColor[arch];
    });
    datasets.push({
      label: arch, data, borderColor: CFG.archColor[arch], backgroundColor: CFG.archColor[arch] + "14",
      pointBackgroundColor: ptColor, pointBorderColor: ptColor, pointRadius: 3, pointHoverRadius: 5,
      borderWidth: 2.2, tension: .25, spanGaps: true, order: 1, fill: single ? "origin" : false,
    });
  }
  $("#noiseNote").innerHTML =
    `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.3v4M8 5v.01" stroke-linecap="round"/></svg>` + note;

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
      const real = regOf(r, mainBaseline(op, shape, dtype, arch, metric)).real;
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
    // vs_main is only a real diff for PR runs (a main run rebuilds its own commit as baseline).
    const wd = r.pr ? worstDeltaForRun(r.run_id) : null;
    const wsev = wd == null ? "none" : wd.worst <= CFG.regressionPct ? "bad" : wd.worst <= CFG.warnPct ? "warn" : "ok";
    const perf = `<div class="pr-perf"><span class="lab">${r.pr ? "worst Δ vs main" : "branch"}</span>` +
      (!r.pr ? `<span class="worst none">${esc(r.branch || "—")} commit</span>`
        : wd == null ? `<span class="worst none">no perf data</span>`
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
