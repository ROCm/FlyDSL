#!/usr/bin/env python3
"""Pure-Python tests for ingest.py (no GPU, no network).

These cover the two riskiest pure-logic functions in the ingester:
``resolve_pr`` (PR-number resolution, incl. the fork fallback path) and
``merge_history`` (idempotent dedup, day cutoff, malformed-timestamp handling).
All GitHub access goes through the module-level ``gh()``; tests monkeypatch it
so nothing touches the network.

Run:  python3 -m pytest .github/dashboard/ingest/test_ingest.py
  or:  python3 .github/dashboard/ingest/test_ingest.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # importable from repo root

import ingest  # noqa: E402


# --------------------------------------------------------------------------- #
# resolve_pr
# --------------------------------------------------------------------------- #
def _clear_cache():
    ingest._PR_CACHE.clear()


def test_resolve_pr_prefers_inline_pull_requests(monkeypatch):
    # When the run already carries pull_requests, gh() must NOT be called.
    monkeypatch.setattr(ingest, "gh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gh called")))
    _clear_cache()
    run = {"pull_requests": [{"number": 42}], "event": "pull_request"}
    assert ingest.resolve_pr("ROCm/FlyDSL", run) == 42


def test_resolve_pr_non_pr_event_returns_none(monkeypatch):
    monkeypatch.setattr(ingest, "gh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gh called")))
    _clear_cache()
    run = {"pull_requests": [], "event": "push", "head_branch": "main"}
    assert ingest.resolve_pr("ROCm/FlyDSL", run) is None


def test_resolve_pr_missing_owner_or_branch_returns_none(monkeypatch):
    monkeypatch.setattr(ingest, "gh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gh called")))
    _clear_cache()
    run = {"event": "pull_request", "head_repository": {"owner": {}}, "head_branch": None}
    assert ingest.resolve_pr("ROCm/FlyDSL", run) is None


def test_resolve_pr_matches_candidate_by_sha(monkeypatch):
    cands = [
        {"number": 100, "head": {"sha": "aaa"}},
        {"number": 101, "head": {"sha": "bbb"}},
    ]
    monkeypatch.setattr(ingest, "gh", lambda *a, **k: cands)
    _clear_cache()
    run = {
        "event": "pull_request",
        "head_repository": {"owner": {"login": "forkuser"}},
        "head_branch": "feature",
        "head_sha": "bbb",
    }
    assert ingest.resolve_pr("ROCm/FlyDSL", run) == 101


def test_resolve_pr_no_sha_match_returns_none_not_guess(monkeypatch):
    # The key regression guard: when no candidate's head SHA matches, we must
    # NOT fall back to cands[0] (which would mislabel records with a wrong PR).
    cands = [{"number": 100, "head": {"sha": "aaa"}}]
    monkeypatch.setattr(ingest, "gh", lambda *a, **k: cands)
    _clear_cache()
    run = {
        "event": "pull_request",
        "head_repository": {"owner": {"login": "forkuser"}},
        "head_branch": "feature",
        "head_sha": "zzz",  # matches nothing
    }
    assert ingest.resolve_pr("ROCm/FlyDSL", run) is None


def test_resolve_pr_empty_candidates_returns_none(monkeypatch):
    monkeypatch.setattr(ingest, "gh", lambda *a, **k: [])
    _clear_cache()
    run = {
        "event": "pull_request",
        "head_repository": {"owner": {"login": "forkuser"}},
        "head_branch": "feature",
        "head_sha": "aaa",
    }
    assert ingest.resolve_pr("ROCm/FlyDSL", run) is None


def test_resolve_pr_gh_error_is_swallowed_to_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("gh api failed")

    monkeypatch.setattr(ingest, "gh", boom)
    _clear_cache()
    run = {
        "event": "pull_request",
        "head_repository": {"owner": {"login": "forkuser"}},
        "head_branch": "feature",
        "head_sha": "aaa",
    }
    assert ingest.resolve_pr("ROCm/FlyDSL", run) is None


def test_resolve_pr_cache_avoids_second_gh_call(monkeypatch):
    calls = {"n": 0}

    def counting_gh(*a, **k):
        calls["n"] += 1
        return [{"number": 7, "head": {"sha": "aaa"}}]

    monkeypatch.setattr(ingest, "gh", counting_gh)
    _clear_cache()
    run = {
        "event": "pull_request",
        "head_repository": {"owner": {"login": "forkuser"}},
        "head_branch": "feature",
        "head_sha": "aaa",
    }
    assert ingest.resolve_pr("ROCm/FlyDSL", run) == 7
    assert ingest.resolve_pr("ROCm/FlyDSL", run) == 7
    assert calls["n"] == 1  # second call served from _PR_CACHE


# --------------------------------------------------------------------------- #
# merge_history
# --------------------------------------------------------------------------- #
def _rec(run_id, runner, op, shape, dtype, metric, value, ts):
    return {
        "run_id": run_id,
        "runner": runner,
        "op": op,
        "shape": shape,
        "dtype": dtype,
        "metric": metric,
        "value": value,
        "ts": ts,
    }


def _now():
    return datetime.now(timezone.utc)


def test_merge_history_idempotent_newest_wins():
    ts = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    old = _rec(1, "linux-flydsl-mi355-1", "gemm", "5120", "fp8", "TFLOPS", 1000.0, ts)
    new = _rec(1, "linux-flydsl-mi355-1", "gemm", "5120", "fp8", "TFLOPS", 1361.4, ts)
    merged = ingest.merge_history([old], [new], history_days=30)
    assert len(merged) == 1  # same _rec_key -> deduped
    assert merged[0]["value"] == 1361.4  # fresh parse wins


def test_merge_history_drops_records_older_than_cutoff():
    fresh_ts = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    old_ts = (_now() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    keep = _rec(1, "r", "op", "s", "d", "TB/s", 1.0, fresh_ts)
    drop = _rec(2, "r", "op", "s", "d", "TB/s", 2.0, old_ts)
    merged = ingest.merge_history([drop], [keep], history_days=30)
    assert len(merged) == 1
    assert merged[0]["run_id"] == 1


def test_merge_history_keeps_records_with_unparseable_or_missing_ts():
    # A None ts and a garbage ts must both be retained (the except: pass / no-ts
    # paths fall through to kept.append), not silently dropped.
    none_ts = _rec(1, "r", "op", "s", "d", "TB/s", 1.0, None)
    bad_ts = _rec(2, "r", "op", "s", "d", "TB/s", 2.0, "not-a-date")
    merged = ingest.merge_history([], [none_ts, bad_ts], history_days=30)
    assert len(merged) == 2


def test_merge_history_sorted_by_ts_then_runner_op_shape():
    t1 = "2026-01-01T00:00:00Z"
    t2 = "2026-02-01T00:00:00Z"
    a = _rec(1, "b-runner", "z-op", "s", "d", "TB/s", 1.0, t2)
    b = _rec(2, "a-runner", "a-op", "s", "d", "TB/s", 2.0, t1)
    merged = ingest.merge_history([], [a, b], history_days=3650)
    assert [r["run_id"] for r in merged] == [2, 1]  # t1 sorts before t2


# --------------------------------------------------------------------------- #
# runner_of
# --------------------------------------------------------------------------- #
def test_runner_of_matches_known_box_and_rejects_unknown():
    assert ingest.runner_of("test (linux-flydsl-mi355-1)") == "linux-flydsl-mi355-1"
    assert ingest.runner_of("test (linux-flydsl-unknown-9)") is None  # not in RUNNER_ARCH
    assert ingest.runner_of("build") is None
    assert ingest.runner_of("") is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
