#!/usr/bin/env python3
"""Ingest FlyDSL CI benchmark runs into the dashboard data files.

For each recent run of the benchmark workflow (``flydsl.yaml`` / "Fly DSL test") this:

1. finds the three single-GPU benchmark jobs (``test (linux-flydsl-<box>)``),
2. downloads each job's log and parses it with :mod:`parse_bench`,
3. merges the resulting records into ``history.json`` (idempotent per run+runner),
4. snapshots recent runs + per-job status into ``runs.json`` for the live CI board.

All GitHub access goes through the ``gh`` CLI so the same script runs unchanged in CI
(``GH_TOKEN``/``GITHUB_TOKEN`` from the workflow) and on a laptop (``gh auth login``).
The token only needs ``actions:read`` (logs) + ``contents`` for the caller to commit.

Usage::

    python ingest.py --repo ROCm/FlyDSL --out-dir ../data --history-days 90 --max-runs 40
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import parse_bench

JOB_RUNNER = re.compile(r"\(\s*(linux-flydsl-[A-Za-z0-9-]+)\s*\)")
BENCH_RUNNERS = set(parse_bench.RUNNER_ARCH)  # only the single-GPU benchmark boxes


def gh(path: str, paginate: bool = False) -> object:
    """Call ``gh api <path>`` and return parsed JSON (dict/list)."""
    cmd = ["gh", "api", path]
    if paginate:
        cmd += ["--paginate", "--slurp"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {out.stderr.strip()}")
    # With --paginate --slurp, gh emits a JSON array whose elements are the per-page
    # response bodies (each a dict like {"jobs": [...]}); callers iterate those pages.
    return json.loads(out.stdout or "null")


def gh_text(path: str) -> str:
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {out.stderr.strip()}")
    return out.stdout


def list_runs(repo: str, workflow: str, max_runs: int) -> list[dict]:
    """Most recent runs of the benchmark workflow, newest first."""
    per_page = min(max_runs, 100)
    data = gh(f"repos/{repo}/actions/workflows/{workflow}/runs?per_page={per_page}")
    return data.get("workflow_runs", [])[:max_runs]


def run_jobs(repo: str, run_id: int) -> list[dict]:
    pages = gh(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", paginate=True)
    jobs: list[dict] = []
    for page in (pages if isinstance(pages, list) else [pages]):
        jobs += page.get("jobs", []) if isinstance(page, dict) else []
    return jobs


def runner_of(job_name: str) -> str | None:
    m = JOB_RUNNER.search(job_name or "")
    return m.group(1) if m and m.group(1) in BENCH_RUNNERS else None


def ingest_run(repo: str, run: dict, regression_pct: float) -> tuple[list[dict], dict]:
    """Return (records, run_summary) for one workflow run."""
    run_id = run["id"]
    commit = run.get("head_sha")
    pr = run["pull_requests"][0]["number"] if run.get("pull_requests") else None
    jobs = run_jobs(repo, run_id)
    records: list[dict] = []
    job_status: list[dict] = []
    for job in jobs:
        runner = runner_of(job.get("name", ""))
        if not runner:
            continue
        js = {
            "runner": runner,
            "arch": parse_bench.RUNNER_ARCH[runner],
            "status": job.get("status"),
            "conclusion": job.get("conclusion"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "url": job.get("html_url"),
        }
        job_status.append(js)
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            continue  # only completed-successful jobs have parseable benchmark output
        try:
            text = gh_text(f"repos/{repo}/actions/jobs/{job['id']}/logs")
        except RuntimeError as e:
            # Record the gap so a successful job with unparsed numbers is not mistaken
            # for a job that genuinely produced no benchmark rows.
            js["log_fetch_failed"] = True
            print(f"  ! log fetch failed for job {job['id']}: {e}", file=sys.stderr)
            continue
        recs = parse_bench.parse_log(text, regression_pct=regression_pct)
        recs += parse_bench.parse_aiter_compare(text)
        ts = job.get("completed_at") or run.get("updated_at")
        records += parse_bench.attach_meta(recs, runner=runner, commit=commit, pr=pr, run_id=run_id, ts=ts, source="ci")
    summary = {
        "run_id": run_id,
        "pr": pr,
        "commit": commit,
        "branch": run.get("head_branch"),
        "event": run.get("event"),
        "title": run.get("display_title"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "url": run.get("html_url"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "actor": (run.get("actor") or {}).get("login"),
        "jobs": job_status,
    }
    return records, summary


def _rec_key(r: dict) -> tuple:
    return (r["run_id"], r["runner"], r["op"], r["shape"], r["dtype"], r["metric"])


def merge_history(existing: list[dict], fresh: list[dict], history_days: int) -> list[dict]:
    by_key = {_rec_key(r): r for r in existing}
    for r in fresh:
        by_key[_rec_key(r)] = r  # re-ingest is idempotent; newest parse wins
    cutoff = datetime.now(timezone.utc) - timedelta(days=history_days)
    kept = []
    for r in by_key.values():
        ts = r.get("ts")
        if ts:
            try:
                if datetime.fromisoformat(ts.replace("Z", "+00:00")) < cutoff:
                    continue
            except ValueError:
                pass
        kept.append(r)
    kept.sort(key=lambda r: (r.get("ts") or "", r["runner"], r["op"], r["shape"]))
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="ROCm/FlyDSL")
    ap.add_argument("--workflow", default="flydsl.yaml")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--history-file", default=None, help="default <out-dir>/history.json")
    ap.add_argument("--runs-file", default=None, help="default <out-dir>/runs.json")
    ap.add_argument("--max-runs", type=int, default=40, help="runs to scan for live board")
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--regression-pct", type=float, default=parse_bench.DEFAULT_REGRESSION_PCT)
    args = ap.parse_args(argv)

    import os

    hist_path = args.history_file or os.path.join(args.out_dir, "history.json")
    runs_path = args.runs_file or os.path.join(args.out_dir, "runs.json")

    runs = list_runs(args.repo, args.workflow, args.max_runs)
    print(f"scanning {len(runs)} runs of {args.workflow} in {args.repo}", file=sys.stderr)

    all_records: list[dict] = []
    summaries: list[dict] = []
    for run in runs:
        recs, summary = ingest_run(args.repo, run, args.regression_pct)
        all_records += recs
        summaries.append(summary)
        tag = f"#{summary['pr']}" if summary["pr"] else summary["branch"]
        print(
            f"  run {run['id']} {tag}: {len(recs)} records " f"({len([j for j in summary['jobs']])} bench jobs)",
            file=sys.stderr,
        )

    existing = []
    if os.path.exists(hist_path):
        existing = json.load(open(hist_path)).get("records", [])
    merged = merge_history(existing, all_records, args.history_days)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump({"schema": 1, "updated": now, "repo": args.repo, "records": merged}, open(hist_path, "w"), indent=1)
    json.dump({"schema": 1, "updated": now, "repo": args.repo, "runs": summaries}, open(runs_path, "w"), indent=1)
    print(f"history: {len(merged)} records -> {hist_path}", file=sys.stderr)
    print(f"runs: {len(summaries)} -> {runs_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
