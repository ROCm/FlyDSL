# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import flydsl.utils.parallel as parallel
from flydsl.utils.env import aot
from flydsl.utils.parallel import run_parallel_jobs

pytestmark = [pytest.mark.l0_backend_agnostic]


def _success_worker(kernel_name, index, delay=0.0):
    time.sleep(delay)
    return {
        "kernel_name": kernel_name,
        "index": index,
        "compile_time": 0.01,
    }


def _tracked_worker(kernel_name, index, state_path, lock_path, delay):
    state_file = Path(state_path)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        state_file.write_text(json.dumps(state), encoding="utf-8")
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    try:
        time.sleep(delay)
        return {
            "kernel_name": kernel_name,
            "index": index,
            "compile_time": 0.01,
        }
    finally:
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            state = json.loads(state_file.read_text(encoding="utf-8"))
            state["active"] -= 1
            state_file.write_text(json.dumps(state), encoding="utf-8")
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _crash_then_succeed(kernel_name, attempt_path, crashes, order_path=None):
    if order_path is not None:
        with open(order_path, "a", encoding="utf-8") as order_file:
            order_file.write(f"{kernel_name}\n")
    path = Path(attempt_path)
    attempt = int(path.read_text(encoding="utf-8")) + 1
    path.write_text(str(attempt), encoding="utf-8")
    if attempt <= crashes:
        os._exit(17)
    return {"kernel_name": kernel_name, "compile_time": 0.01}


def _deterministic_failure(kernel_name, attempt_path):
    path = Path(attempt_path)
    attempt = int(path.read_text(encoding="utf-8")) + 1
    path.write_text(str(attempt), encoding="utf-8")
    return {"kernel_name": kernel_name, "compile_time": None}


def _sleep_worker(kernel_name, delay):
    time.sleep(delay)
    return {"kernel_name": kernel_name, "compile_time": 0.01}


def _mixed_outcome_worker(kernel_name, outcome):
    if outcome == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
    if outcome == "exit-137":
        os._exit(137)
    if outcome == "type-error":
        raise TypeError("synthetic worker type error")
    if outcome == "large-error":
        raise RuntimeError("X" * (parallel._MAX_TRACEBACK_CHARS * 4))
    if outcome == "compile-error":
        return {
            "kernel_name": kernel_name,
            "compile_time": None,
            "failure": {
                "kind": "compile_error",
                "reason": "synthetic codegen failure",
            },
        }
    raise ValueError(f"unknown synthetic outcome: {outcome}")


def _exception_marker_then_exit_zero(worker, kwargs, out_path):
    parallel._write_json_file(
        out_path,
        {
            parallel._WORKER_EXCEPTION_KEY: {
                "kind": "worker_exception",
                "reason": "TypeError: preserved reason",
                "traceback": "Traceback (most recent call last): preserved frame",
            }
        },
    )


def _malformed_exception_marker_then_exit_zero(worker, kwargs, out_path):
    parallel._write_json_file(
        out_path,
        {parallel._WORKER_EXCEPTION_KEY: "not-a-dictionary"},
    )


def _exit_without_result(worker, kwargs, out_path):
    return None


def _non_dict_worker(kernel_name):
    return ["not", "a", "dictionary"]


def _always_oom_worker(kernel_name, attempt_path):
    path = Path(attempt_path)
    attempt = int(path.read_text(encoding="utf-8")) + 1
    path.write_text(str(attempt), encoding="utf-8")
    os.kill(os.getpid(), signal.SIGKILL)


@pytest.fixture(autouse=True)
def _parallel_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "2")
    monkeypatch.setenv("FLYDSL_AOT_MEM_PER_WORKER_GB", "0")
    monkeypatch.setenv("FLYDSL_AOT_TIMEOUT", "5")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "0")


@pytest.fixture
def log_messages(monkeypatch):
    messages = []

    def record(message, *args):
        messages.append(message % args if args else message)

    logger = SimpleNamespace(info=record, warning=record)
    monkeypatch.setattr(parallel, "log", lambda: logger)
    return messages


def test_workers_option_whitespace_uses_automatic_default(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "   ")
    monkeypatch.setattr(parallel, "_affinity_aware_cpu_count", lambda: 8)
    monkeypatch.setattr(parallel, "_memory_worker_cap", lambda workers: workers)

    assert aot.workers == 0
    assert parallel._get_max_workers(num_jobs=100) == 8


def test_results_follow_input_order():
    jobs = [
        {"kernel_name": "slow", "index": 0, "delay": 0.1},
        {"kernel_name": "fast", "index": 1, "delay": 0.0},
        {"kernel_name": "last", "index": 2, "delay": 0.01},
    ]

    results = run_parallel_jobs(_success_worker, jobs)

    assert [result["index"] for result in results] == [0, 1, 2]


def test_committed_result_wins_over_teardown_sigkill(monkeypatch):
    original_write = parallel._write_json_file

    def write_then_sigkill(out_path, payload):
        original_write(out_path, payload)
        os.kill(os.getpid(), signal.SIGKILL)

    monkeypatch.setattr(parallel, "_write_json_file", write_then_sigkill)

    results = run_parallel_jobs(
        _success_worker,
        [{"kernel_name": "committed", "index": 0}],
    )

    assert results == [
        {
            "kernel_name": "committed",
            "index": 0,
            "compile_time": 0.01,
        }
    ]


def test_committed_result_wins_over_teardown_timeout(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_TIMEOUT", "0.05")
    original_write = parallel._write_json_file

    def write_then_wait(out_path, payload):
        original_write(out_path, payload)
        time.sleep(60)

    monkeypatch.setattr(parallel, "_write_json_file", write_then_wait)

    results = run_parallel_jobs(
        _success_worker,
        [{"kernel_name": "committed", "index": 0}],
    )

    assert results == [
        {
            "kernel_name": "committed",
            "index": 0,
            "compile_time": 0.01,
        }
    ]


def test_worker_count_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "2")
    state_path = tmp_path / "state.json"
    lock_path = tmp_path / "state.lock"
    state_path.write_text(json.dumps({"active": 0, "peak": 0}), encoding="utf-8")
    jobs = [
        {
            "kernel_name": f"job-{index}",
            "index": index,
            "state_path": str(state_path),
            "lock_path": str(lock_path),
            "delay": 0.1,
        }
        for index in range(6)
    ]

    results = run_parallel_jobs(_tracked_worker, jobs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == {"active": 0, "peak": 2}
    assert all(result["compile_time"] is not None for result in results)


def test_crashed_worker_is_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "2")
    attempt_path = tmp_path / "attempt.txt"
    attempt_path.write_text("0", encoding="utf-8")

    results = run_parallel_jobs(
        _crash_then_succeed,
        [
            {
                "kernel_name": "retry-job",
                "attempt_path": str(attempt_path),
                "crashes": 2,
            }
        ],
    )

    assert attempt_path.read_text(encoding="utf-8") == "3"
    assert results[0]["compile_time"] is not None


def test_retries_wait_behind_pending_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "1")
    order_path = tmp_path / "order.txt"
    order_path.write_text("", encoding="utf-8")
    attempt_paths = [tmp_path / f"attempt-{index}.txt" for index in range(3)]
    for attempt_path in attempt_paths:
        attempt_path.write_text("0", encoding="utf-8")
    jobs = [
        {
            "kernel_name": "retry",
            "attempt_path": str(attempt_paths[0]),
            "crashes": 1,
            "order_path": str(order_path),
        },
        {
            "kernel_name": "pending-0",
            "attempt_path": str(attempt_paths[1]),
            "crashes": 0,
            "order_path": str(order_path),
        },
        {
            "kernel_name": "pending-1",
            "attempt_path": str(attempt_paths[2]),
            "crashes": 0,
            "order_path": str(order_path),
        },
    ]

    results = run_parallel_jobs(_crash_then_succeed, jobs)

    assert all(result["compile_time"] is not None for result in results)
    assert order_path.read_text(encoding="utf-8").splitlines() == [
        "retry",
        "pending-0",
        "pending-1",
        "retry",
    ]


def test_oom_retry_exhaustion_returns_possible_oom(monkeypatch, tmp_path, log_messages):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "2")
    attempt_path = tmp_path / "attempt.txt"
    attempt_path.write_text("0", encoding="utf-8")

    result = run_parallel_jobs(
        _always_oom_worker,
        [{"kernel_name": "always-oom", "attempt_path": str(attempt_path)}],
    )[0]
    output = "\n".join(log_messages)

    assert attempt_path.read_text(encoding="utf-8") == "3"
    assert result["failure"] == {
        "kind": "possible_oom",
        "reason": "worker killed by SIGKILL (possible OOM)",
        "attempts": 3,
        "exitcode": -signal.SIGKILL,
    }
    assert "retry 1/2" in output
    assert "retry 2/2" in output
    assert "not retrying" in output


def test_retry_exhaustion_returns_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "1")
    attempt_path = tmp_path / "attempt.txt"
    attempt_path.write_text("0", encoding="utf-8")

    results = run_parallel_jobs(
        _crash_then_succeed,
        [
            {
                "kernel_name": "dead-job",
                "attempt_path": str(attempt_path),
                "crashes": 3,
            }
        ],
    )

    assert attempt_path.read_text(encoding="utf-8") == "2"
    assert results[0]["compile_time"] is None
    assert results[0]["failure"] == {
        "kind": "worker_crash",
        "reason": "worker crashed (exitcode=17)",
        "attempts": 2,
        "exitcode": 17,
    }


def test_incomplete_pool_results_raise_before_summary(log_messages):
    with pytest.raises(RuntimeError, match="0/1 jobs completed"):
        parallel._finalize_pool_results(
            [None],
            num_jobs=1,
            completed=0,
            succeeded_jobs=0,
            failed_jobs=0,
            retries_used=0,
        )

    assert not any("succeeded" in message for message in log_messages)


def test_failure_results_preserve_distinct_causes():
    jobs = [
        {"kernel_name": "signal", "outcome": "signal"},
        {"kernel_name": "exit-137", "outcome": "exit-137"},
        {"kernel_name": "type-error", "outcome": "type-error"},
        {"kernel_name": "compile-error", "outcome": "compile-error"},
    ]

    results = run_parallel_jobs(_mixed_outcome_worker, jobs)
    failures = [result["failure"] for result in results]

    assert failures[0] == {
        "kind": "worker_crash",
        "reason": f"worker crashed (exitcode={-signal.SIGTERM})",
        "attempts": 1,
        "exitcode": -signal.SIGTERM,
    }
    assert failures[1] == {
        "kind": "worker_crash",
        "reason": "worker crashed (exitcode=137)",
        "attempts": 1,
        "exitcode": 137,
    }
    assert failures[2]["kind"] == "worker_exception"
    assert failures[2]["reason"] == "TypeError: synthetic worker type error"
    assert failures[2]["attempts"] == 1
    assert failures[2]["exitcode"] == 1
    assert failures[2]["traceback"].startswith("Traceback (most recent call last):")
    assert "test_parallel_jobs.py" in failures[2]["traceback"]
    assert failures[2]["traceback"].endswith("TypeError: synthetic worker type error\n")
    assert failures[3] == {
        "kind": "compile_error",
        "reason": "synthetic codegen failure",
        "attempts": 1,
    }
    assert len({json.dumps(failure, sort_keys=True) for failure in failures}) == 4


def test_python_exception_is_not_retried_or_duplicated_to_stderr(monkeypatch, capfd):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "3")

    result = run_parallel_jobs(
        _mixed_outcome_worker,
        [{"kernel_name": "type-error", "outcome": "type-error"}],
    )[0]

    assert result["failure"]["kind"] == "worker_exception"
    assert result["failure"]["attempts"] == 1
    assert result["failure"]["traceback"].startswith("Traceback (most recent call last):")
    assert capfd.readouterr().err == ""


def test_large_exception_diagnostics_are_bounded():
    result = run_parallel_jobs(
        _mixed_outcome_worker,
        [{"kernel_name": "large-error", "outcome": "large-error"}],
    )[0]
    failure = result["failure"]

    assert len(failure["reason"]) <= parallel._MAX_FAILURE_REASON_CHARS
    assert "... failure reason truncated ..." in failure["reason"]
    assert len(failure["traceback"]) <= parallel._MAX_TRACEBACK_CHARS
    assert failure["traceback"].startswith("Traceback (most recent call last):")
    assert "... traceback truncated ..." in failure["traceback"]
    assert failure["traceback"].endswith("X" * 32 + "\n")


def test_exception_marker_is_preserved_when_process_exits_zero(monkeypatch):
    monkeypatch.setattr(
        parallel,
        "_run_one_to_file",
        _exception_marker_then_exit_zero,
    )

    result = run_parallel_jobs(
        _success_worker,
        [{"kernel_name": "marker", "index": 0}],
    )[0]

    assert result["failure"] == {
        "kind": "worker_exception",
        "reason": "TypeError: preserved reason",
        "attempts": 1,
        "exitcode": 0,
        "traceback": "Traceback (most recent call last): preserved frame",
    }


def test_malformed_exception_marker_is_localized_to_invalid_result(monkeypatch):
    monkeypatch.setattr(
        parallel,
        "_run_one_to_file",
        _malformed_exception_marker_then_exit_zero,
    )

    result = run_parallel_jobs(
        _success_worker,
        [{"kernel_name": "malformed-marker", "index": 0}],
    )[0]

    assert result["failure"] == {
        "kind": "invalid_result",
        "reason": ("worker exception marker must contain a dictionary, got str"),
        "attempts": 1,
        "exitcode": 0,
    }


def test_invalid_result_reasons_distinguish_missing_and_non_dict(monkeypatch):
    non_dict = run_parallel_jobs(
        _non_dict_worker,
        [{"kernel_name": "non-dict"}],
    )[0]

    monkeypatch.setattr(parallel, "_run_one_to_file", _exit_without_result)
    missing = run_parallel_jobs(
        _success_worker,
        [{"kernel_name": "missing", "index": 0}],
    )[0]

    assert non_dict["failure"]["kind"] == "invalid_result"
    assert non_dict["failure"]["reason"] == "worker returned list, expected a dictionary"
    assert missing["failure"]["kind"] == "invalid_result"
    assert missing["failure"]["reason"] == "worker produced no result file"


def test_deterministic_failure_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "3")
    attempt_path = tmp_path / "attempt.txt"
    attempt_path.write_text("0", encoding="utf-8")

    results = run_parallel_jobs(
        _deterministic_failure,
        [
            {
                "kernel_name": "compile-error",
                "attempt_path": str(attempt_path),
            }
        ],
    )

    assert attempt_path.read_text(encoding="utf-8") == "1"
    assert results[0]["compile_time"] is None
    assert results[0]["failure"] == {
        "kind": "compile_error",
        "reason": "worker returned compile_time=None",
        "attempts": 1,
    }


def test_timed_out_worker_is_killed(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_TIMEOUT", "0.05")

    started = time.monotonic()
    results = run_parallel_jobs(
        _sleep_worker,
        [{"kernel_name": "hung-job", "delay": 60}],
    )

    assert time.monotonic() - started < 5
    assert results[0]["compile_time"] is None
    assert results[0]["failure"]["kind"] == "timeout"
    assert results[0]["failure"]["exitcode"] == -signal.SIGKILL
    assert results[0]["failure"]["attempts"] == 1
    assert results[0]["failure"]["reason"] == "exceeded per-job timeout (0.05s); killed"


def test_mem_per_worker_env_caps_automatic_workers(monkeypatch):
    gib = 1024**3
    fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=8 * gib))
    monkeypatch.delenv("FLYDSL_AOT_WORKERS")
    monkeypatch.setenv("FLYDSL_AOT_MEM_PER_WORKER_GB", "2")
    monkeypatch.setattr(parallel, "_affinity_aware_cpu_count", lambda: 16)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert parallel._get_max_workers(num_jobs=100) == 4


def test_missing_psutil_warns_and_uses_conservative_limit(monkeypatch, log_messages):
    monkeypatch.delenv("FLYDSL_AOT_WORKERS")
    monkeypatch.setenv("FLYDSL_AOT_MEM_PER_WORKER_GB", "2")
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(parallel, "_affinity_aware_cpu_count", lambda: 16)

    with pytest.warns(RuntimeWarning, match="limiting AOT concurrency to 4 workers"):
        assert parallel._get_max_workers(num_jobs=100) == 4
    assert "psutil is not installed; automatic AOT memory limiting is unavailable" in "\n".join(log_messages)
