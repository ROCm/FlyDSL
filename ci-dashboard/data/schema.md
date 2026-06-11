# CI dashboard data schema

These files are produced by `ci-dashboard/ingest/` and consumed by `app.js`.
The committed copies under `ci-dashboard/data/` are a **seed** for first paint; the
live, continuously-updated copies live on the orphan `ci-dashboard-data` branch and
are fetched at runtime (see `CFG.dataBranch` in `app.js`).

All metrics are bigger-is-better. `metric` is `"TB/s"` (memory-bound ops),
`"TFLOPS"` (compute-bound ops), or `"speedup"` (FlyDSL-vs-AIter ratio).

## `history.json` — per-kernel benchmark records

```jsonc
{
  "schema": 1,
  "updated": "2026-06-11T07:00:00Z",
  "repo": "ROCm/FlyDSL",
  "records": [
    {
      "ts": "2026-06-09T10:00:11Z",   // job completed_at
      "commit": "e33eca2e…",
      "pr": 669,                        // null for main/push runs
      "run_id": 27196355973,
      "source": "ci",
      "runner": "linux-flydsl-mi325-1",
      "arch": "gfx942",                 // gfx950 | gfx942 | gfx1201
      "op": "layernorm",
      "shape": "32768x8192",
      "dtype": "bf16",
      "metric": "TB/s",
      "value": 3.971,                   // current value (null if skipped)
      "status": "ok",                   // ok | skip | missing
      "vs_main": { "label": "main", "baseline": 4.270, "ratio": 0.930, "delta_pct": -7.0 },
      "vs_tag":  { "tag": "v0.2.0", "baseline": 2.326, "ratio": 1.707, "delta_pct": 70.7 },
      "regression": true,               // vs_main.delta_pct <= -3.0
      "extra": null                     // {"flydsl_us","aiter_us","baseline"} for speedup rows
    }
  ]
}
```

`vs_main` / `vs_tag` are `null` when the run had no comparable baseline. The
regression gate (`-3%`) is `parse_bench.DEFAULT_REGRESSION_PCT`.

## `runs.json` — CI run + per-job status (live board)

```jsonc
{
  "schema": 1, "updated": "…", "repo": "ROCm/FlyDSL",
  "runs": [
    {
      "run_id": 27196355973, "pr": 669, "commit": "e33eca2e…",
      "branch": "…", "event": "pull_request", "title": "…",
      "status": "completed", "conclusion": "success",
      "url": "https://github.com/ROCm/FlyDSL/actions/runs/27196355973",
      "created_at": "…", "updated_at": "…", "actor": "jhinpan",
      "jobs": [
        { "runner": "linux-flydsl-mi355-1", "arch": "gfx950",
          "status": "completed", "conclusion": "success", "url": "…" }
      ]
    }
  ]
}
```
