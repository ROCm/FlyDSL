# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
import traceback
from collections import deque
from collections.abc import Callable
from multiprocessing.connection import wait as wait_for_sentinels
from pathlib import Path
from typing import Any, cast

from .env import aot
from .file import atomic_write
from .logger import log

_DEFAULT_MAX_WORKERS = 64
_MAX_FAILURE_REASON_CHARS = 2 * 1024
_MAX_TRACEBACK_CHARS = 16 * 1024
_OOM_EXITCODE = -signal.SIGKILL
_WORKER_EXCEPTION_KEY = "__flydsl_worker_exception__"


def _write_json_file(out_path: str, payload: Any) -> None:
    with atomic_write(Path(out_path), mode="w", encoding="utf-8") as output:
        json.dump(payload, output)


def _truncate_text(text: str, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def _truncate_failure_reason(reason: str) -> str:
    return _truncate_text(
        reason,
        _MAX_FAILURE_REASON_CHARS,
        "\n... failure reason truncated ...\n",
    )


def _truncate_traceback(traceback_text: str) -> str:
    return _truncate_text(
        traceback_text,
        _MAX_TRACEBACK_CHARS,
        "\n... traceback truncated ...\n",
    )


def _exception_exitcode(error: BaseException) -> int:
    if isinstance(error, SystemExit):
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code & 0xFF
    return 1


def _run_one_to_file(
    worker: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    out_path: str,
) -> None:
    try:
        result = worker(**kwargs)
        _write_json_file(out_path, result)
    except BaseException as error:
        try:
            _write_json_file(
                out_path,
                {
                    _WORKER_EXCEPTION_KEY: {
                        "kind": "worker_exception",
                        "reason": _truncate_failure_reason(f"{type(error).__name__}: {error}"),
                        "traceback": _truncate_traceback(traceback.format_exc()),
                    }
                },
            )
        finally:
            os._exit(_exception_exitcode(error))


def _affinity_aware_cpu_count() -> int:
    """Return the CPU count available to this process."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 0
    return max(count, 1)


def _get_kernel_timeout() -> float:
    return max(aot.timeout, 0.0)


def _get_max_retries() -> int:
    return max(aot.max_retries, 0)


def _memory_worker_cap(default_workers: int) -> int:
    per_worker_gb = aot.mem_per_worker_gb
    if per_worker_gb <= 0:
        return default_workers

    try:
        import psutil
    except ImportError:
        log().warning(
            "psutil is not installed; AOT memory limiting is disabled and the CPU-based worker limit will be used"
        )
        return default_workers

    try:
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception as error:
        log().warning(
            "failed to query available memory for AOT worker limiting (%s); the CPU-based worker limit will be used",
            error,
        )
        return default_workers
    return min(default_workers, max(1, int(available_gb / per_worker_gb)))


def _get_max_workers(num_jobs: int) -> int:
    workers_raw = os.environ.get("FLYDSL_AOT_WORKERS", "").strip()
    configured_workers = aot.workers if workers_raw else 0
    if configured_workers > 0:
        max_workers = configured_workers
    else:
        max_workers = min(_affinity_aware_cpu_count(), _DEFAULT_MAX_WORKERS)
        max_workers = _memory_worker_cap(max_workers)
    return min(max_workers, num_jobs)


def _job_label(job: dict[str, Any]) -> str:
    return str(job.get("kernel_name", "?"))


def _failure_result(
    job: dict[str, Any],
    *,
    kind: str,
    reason: str,
    attempts: int | None = None,
    exitcode: int | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "kind": kind,
        "reason": reason,
    }
    if attempts is not None:
        failure["attempts"] = attempts
    if exitcode is not None:
        failure["exitcode"] = exitcode
    if traceback_text is not None:
        failure["traceback"] = traceback_text
    return {
        "kernel_name": _job_label(job),
        "compile_time": None,
        "failure": failure,
    }


def _run_file_pool(
    worker: Callable[..., dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    max_workers: int,
    kernel_timeout: float,
    max_retries: int,
    result_dir: str,
) -> list[dict[str, Any]]:
    """Run jobs through a Linux fork pool using files for result transport."""
    ctx = multiprocessing.get_context("fork")
    num_jobs = len(jobs)
    results: list[dict[str, Any] | None] = [None] * num_jobs
    attempts = [0] * num_jobs
    retries_used = 0
    completed = 0
    failed_jobs = 0
    progress_stride = max(1, num_jobs // 20)
    initial_max_workers = max_workers
    launch_epoch = 0
    successful_since_backoff = 0

    queue = deque(range(num_jobs))
    running: dict[Any, tuple[int, float | None, int]] = {}

    def launch() -> None:
        while queue and len(running) < max_workers:
            index = queue.popleft()
            out_path = os.path.join(result_dir, f"k{index}.json")
            try:
                os.remove(out_path)
            except OSError:
                pass

            process = ctx.Process(
                target=_run_one_to_file,
                args=(worker, jobs[index], out_path),
            )
            process.start()
            deadline = time.monotonic() + kernel_timeout if kernel_timeout > 0 else None
            running[process] = (index, deadline, launch_epoch)

    def note_done(*, is_failure: bool = False) -> None:
        nonlocal completed, failed_jobs
        completed += 1
        failed_jobs += int(is_failure)
        if completed % progress_stride == 0 or completed == num_jobs:
            log().info(
                "... %d/%d jobs finished (%d failed)",
                completed,
                num_jobs,
                failed_jobs,
            )

    def note_success() -> None:
        nonlocal max_workers, successful_since_backoff
        if max_workers >= initial_max_workers:
            return
        successful_since_backoff += 1
        if successful_since_backoff < max_workers:
            return
        previous_max_workers = max_workers
        max_workers += 1
        successful_since_backoff = 0
        log().info(
            "AOT worker limit recovered %d->%d after healthy completions",
            previous_max_workers,
            max_workers,
        )

    def backoff_worker_limit(worker_epoch: int, reason: str) -> str:
        nonlocal launch_epoch, max_workers, successful_since_backoff
        if worker_epoch != launch_epoch:
            return f"{reason}; worker limit already reduced to {max_workers} for this failure wave"
        previous_max_workers = max_workers
        max_workers = max(1, max_workers // 2)
        launch_epoch += 1
        successful_since_backoff = 0
        return f"{reason}; reduced worker limit {previous_max_workers}->{max_workers}"

    def finish_failure(
        index: int,
        *,
        kind: str,
        reason: str,
        exitcode: int | None = None,
        traceback_text: str | None = None,
    ) -> None:
        results[index] = _failure_result(
            jobs[index],
            kind=kind,
            reason=reason,
            attempts=attempts[index] + 1,
            exitcode=exitcode,
            traceback_text=traceback_text,
        )
        log().warning(
            "AOT job %s %s; not retrying",
            _job_label(jobs[index]),
            reason,
        )
        note_done(is_failure=True)

    def retry_or_drop(
        index: int,
        *,
        kind: str,
        reason: str,
        exitcode: int | None = None,
        traceback_text: str | None = None,
    ) -> None:
        nonlocal retries_used
        if attempts[index] < max_retries:
            attempts[index] += 1
            retries_used += 1
            queue.append(index)
            log().warning(
                "AOT job %s %s; retry %d/%d",
                _job_label(jobs[index]),
                reason,
                attempts[index],
                max_retries,
            )
        else:
            finish_failure(
                index,
                kind=kind,
                reason=reason,
                exitcode=exitcode,
                traceback_text=traceback_text,
            )

    def reap(process: Any, *, timeout_reason: str | None = None) -> None:
        nonlocal launch_epoch, max_workers, successful_since_backoff
        index, _, worker_epoch = running.pop(process)
        out_path = os.path.join(result_dir, f"k{index}.json")
        try:
            loaded: Any = None
            load_error: str | None = None
            if os.path.isfile(out_path):
                try:
                    with open(out_path, encoding="utf-8") as result_file:
                        loaded = json.load(result_file)
                except Exception as error:  # noqa: BLE001
                    load_error = f"failed to read worker result: {type(error).__name__}: {error}"
            else:
                load_error = "worker produced no result file"

            worker_exception = loaded.get(_WORKER_EXCEPTION_KEY) if isinstance(loaded, dict) else None

            # The atomic result file is the worker's commit point. Once a valid
            # result is present, teardown-time signals must not discard it.
            if isinstance(loaded, dict) and worker_exception is None:
                result = loaded
                if result.get("compile_time") is None:
                    failure = result.get("failure")
                    if not isinstance(failure, dict):
                        failure = {}
                        result["failure"] = failure
                    failure.setdefault("kind", "compile_error")
                    failure.setdefault("reason", "worker returned compile_time=None")
                    failure.setdefault("attempts", attempts[index] + 1)
                results[index] = result
                is_failure = result.get("compile_time") is None
                note_done(is_failure=is_failure)
                if not is_failure:
                    note_success()
                return

            # A structured Python exception is deterministic for the same job.
            # Preserve it even if teardown later changes the process exit code.
            if isinstance(worker_exception, dict):
                reason = str(
                    worker_exception.get(
                        "reason",
                        f"worker raised an exception (exitcode={process.exitcode})",
                    )
                )
                traceback_text = worker_exception.get("traceback")
                if not isinstance(traceback_text, str):
                    traceback_text = None
                finish_failure(
                    index,
                    kind="worker_exception",
                    reason=reason,
                    exitcode=process.exitcode,
                    traceback_text=traceback_text,
                )
                return

            if timeout_reason is not None:
                retry_or_drop(
                    index,
                    kind="timeout",
                    reason=timeout_reason,
                    exitcode=process.exitcode,
                )
                return

            if process.exitcode == _OOM_EXITCODE:
                reason = "worker killed by SIGKILL (possible OOM)"
                can_retry = attempts[index] < max_retries
                if not can_retry:
                    finish_failure(
                        index,
                        kind="possible_oom",
                        reason=reason,
                        exitcode=process.exitcode,
                    )
                elif worker_epoch != launch_epoch:
                    retry_or_drop(
                        index,
                        kind="possible_oom",
                        reason=(f"{reason}; worker limit already reduced to {max_workers} for this failure wave"),
                        exitcode=process.exitcode,
                    )
                elif max_workers <= 1:
                    finish_failure(
                        index,
                        kind="possible_oom",
                        reason=f"{reason} at the minimum worker limit",
                        exitcode=process.exitcode,
                    )
                else:
                    reason = backoff_worker_limit(worker_epoch, reason)
                    retry_or_drop(
                        index,
                        kind="possible_oom",
                        reason=reason,
                        exitcode=process.exitcode,
                    )
                return

            if process.exitcode != 0:
                retry_or_drop(
                    index,
                    kind="worker_crash",
                    reason=f"worker crashed (exitcode={process.exitcode})",
                    exitcode=process.exitcode,
                )
                return

            if load_error is not None:
                finish_failure(
                    index,
                    kind="invalid_result",
                    reason=load_error,
                    exitcode=process.exitcode,
                )
                return
            if not isinstance(loaded, dict):
                finish_failure(
                    index,
                    kind="invalid_result",
                    reason=(f"worker returned {type(loaded).__name__}, expected a dictionary"),
                    exitcode=process.exitcode,
                )
                return
        finally:
            process.close()

    try:
        launch()
        while running:
            if kernel_timeout > 0:
                nearest_deadline = min(deadline for _, deadline, _ in running.values() if deadline is not None)
                wait_timeout: float | None = max(0.0, nearest_deadline - time.monotonic())
            else:
                wait_timeout = None

            wait_for_sentinels(
                [process.sentinel for process in running],
                timeout=wait_timeout,
            )

            for process in list(running):
                if not process.is_alive():
                    process.join()
                    reap(process)

            if kernel_timeout > 0:
                now = time.monotonic()
                for process in list(running):
                    _, deadline, _ = running[process]
                    if deadline is not None and now > deadline and process.is_alive():
                        timeout_reason = f"exceeded per-job timeout ({kernel_timeout:g}s); killed"
                        process.kill()
                        process.join()
                        reap(process, timeout_reason=timeout_reason)

            launch()
    finally:
        for process in list(running):
            try:
                if process.is_alive():
                    process.kill()
                process.join()
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    process.close()
                except Exception:  # noqa: BLE001
                    pass
        running.clear()

    retry_label = "retry" if retries_used == 1 else "retries"
    log().info(
        "AOT: %d succeeded, %d failed; %d %s after abnormal worker exits",
        num_jobs - failed_jobs,
        failed_jobs,
        retries_used,
        retry_label,
    )
    if any(result is None for result in results):
        raise RuntimeError("internal AOT scheduler error: unfinished jobs remain")
    return cast(list[dict[str, Any]], results)


def run_parallel_jobs(
    worker: Callable[..., dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run independent AOT jobs in parallel and return results in input order.

    Each job is expanded into ``worker(**job)``. Workers must return a
    JSON-serializable dictionary containing ``compile_time``; deterministic
    compile errors should be represented by ``compile_time=None`` rather than
    escaping as an exception. Every failed result keeps ``compile_time=None``
    for compatibility and adds a ``failure`` dictionary with machine-readable
    ``kind``, ``reason``, and ``attempts`` fields. Process failures also include
    ``exitcode``; uncaught Python exceptions include a bounded ``traceback`` and
    are not retried.

    This executor uses the Linux ``fork`` multiprocessing context because it is
    intended for compile-only workers that inherit the initialized compiler.
    An OOM-like ``-SIGKILL`` halves the concurrency limit once per launch wave;
    healthy completions then restore it additively. OOM exits are not retried
    once the limit reaches one.
    """
    if not jobs:
        return []

    max_workers = _get_max_workers(len(jobs))
    log().info(
        "AOT: %d jobs, %d worker processes",
        len(jobs),
        max_workers,
    )

    result_dir = tempfile.mkdtemp(prefix="flydsl_aot_results_")
    try:
        results = _run_file_pool(
            worker,
            jobs,
            max_workers=max_workers,
            kernel_timeout=_get_kernel_timeout(),
            max_retries=_get_max_retries(),
            result_dir=result_dir,
        )
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)
    return results


__all__ = ["run_parallel_jobs"]
