# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import fcntl
import json
import os
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


def _crash_then_succeed(kernel_name, attempt_path, crashes):
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
    if outcome == "crash":
        os._exit(17)
    return {
        "kernel_name": kernel_name,
        "compile_time": None if outcome == "compile-error" else 0.01,
    }


@pytest.fixture(autouse=True)
def _parallel_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "2")
    monkeypatch.setenv("FLYDSL_AOT_MEM_PER_WORKER_GB", "0")
    monkeypatch.setenv("FLYDSL_AOT_TIMEOUT", "5")
    monkeypatch.setenv("FLYDSL_AOT_MAX_RETRIES", "0")


def test_empty_jobs_do_not_invoke_worker(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "invalid")
    assert run_parallel_jobs(_success_worker, []) == []


def test_results_follow_input_order():
    jobs = [
        {"kernel_name": "slow", "index": 0, "delay": 0.1},
        {"kernel_name": "fast", "index": 1, "delay": 0.0},
        {"kernel_name": "last", "index": 2, "delay": 0.01},
    ]

    results = run_parallel_jobs(_success_worker, jobs)

    assert [result["index"] for result in results] == [0, 1, 2]


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
    assert results == [{"kernel_name": "dead-job", "compile_time": None}]


def test_failed_jobs_do_not_stop_remaining_jobs():
    jobs = [
        {"kernel_name": "first", "outcome": "success"},
        {"kernel_name": "crashed", "outcome": "crash"},
        {"kernel_name": "compile-error", "outcome": "compile-error"},
        {"kernel_name": "last", "outcome": "success"},
    ]

    results = run_parallel_jobs(_mixed_outcome_worker, jobs)

    assert [result["kernel_name"] for result in results] == [
        "first",
        "crashed",
        "compile-error",
        "last",
    ]
    assert [result["compile_time"] for result in results] == [
        0.01,
        None,
        None,
        0.01,
    ]


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


def test_timed_out_worker_is_killed(monkeypatch):
    monkeypatch.setenv("FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("FLYDSL_AOT_TIMEOUT", "0.05")

    started = time.monotonic()
    results = run_parallel_jobs(
        _sleep_worker,
        [{"kernel_name": "hung-job", "delay": 60}],
    )

    assert time.monotonic() - started < 5
    assert results == [{"kernel_name": "hung-job", "compile_time": None}]


def test_temporary_result_directory_is_removed(monkeypatch, tmp_path):
    result_dir = tmp_path / "results"

    def make_result_dir(prefix):
        assert prefix == "flydsl_aot_results_"
        result_dir.mkdir()
        return str(result_dir)

    monkeypatch.setattr(parallel.tempfile, "mkdtemp", make_result_dir)

    run_parallel_jobs(
        _success_worker,
        [{"kernel_name": "one", "index": 0}],
    )

    assert not result_dir.exists()


def test_mem_per_worker_env_caps_automatic_workers(monkeypatch):
    gib = 1024**3
    fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=8 * gib))
    monkeypatch.delenv("FLYDSL_AOT_WORKERS")
    monkeypatch.setenv("FLYDSL_AOT_MEM_PER_WORKER_GB", "2")
    monkeypatch.setattr(parallel, "_affinity_aware_cpu_count", lambda: 16)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert parallel._get_max_workers(num_jobs=100) == 4


@pytest.mark.parametrize(
    ("variable", "accessor"),
    [
        ("FLYDSL_AOT_WORKERS", lambda: aot.workers),
        ("FLYDSL_AOT_MEM_PER_WORKER_GB", lambda: aot.mem_per_worker_gb),
        ("FLYDSL_AOT_TIMEOUT", lambda: aot.timeout),
        ("FLYDSL_AOT_MAX_RETRIES", lambda: aot.max_retries),
    ],
)
def test_invalid_environment_value_raises(monkeypatch, variable, accessor):
    monkeypatch.setenv(variable, "invalid")
    with pytest.raises(ValueError, match=variable):
        accessor()
