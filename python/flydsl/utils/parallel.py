# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from multiprocessing.connection import wait as wait_for_sentinels
from pathlib import Path
from typing import Any

from .env import aot
from .file import atomic_write

_DEFAULT_MAX_WORKERS = 64


def _run_one_to_file(
    worker: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    out_path: str,
) -> None:
    result = worker(**kwargs)
    with atomic_write(Path(out_path), mode="w", encoding="utf-8") as output:
        json.dump(result, output)


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

        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:  # noqa: BLE001
        return default_workers
    return min(default_workers, max(1, int(available_gb / per_worker_gb)))


def _get_max_workers(num_jobs: int) -> int:
    if "FLYDSL_AOT_WORKERS" in os.environ:
        max_workers = max(aot.workers, 1)
    else:
        max_workers = min(_affinity_aware_cpu_count(), _DEFAULT_MAX_WORKERS)
        max_workers = _memory_worker_cap(max_workers)
    return min(max_workers, num_jobs)


def _job_label(job: dict[str, Any]) -> str:
    return str(job.get("kernel_name", "?"))


def _run_file_pool(
    worker: Callable[..., dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    max_workers: int,
    kernel_timeout: float,
    max_retries: int,
    result_dir: str,
) -> list[dict[str, Any] | None]:
    """Run jobs through a Linux fork pool using files for result transport."""
    ctx = multiprocessing.get_context("fork")
    num_jobs = len(jobs)
    results: list[dict[str, Any] | None] = [None] * num_jobs
    attempts = [0] * num_jobs
    retries_used = 0
    completed = 0
    progress_stride = max(1, num_jobs // 20)

    queue = list(range(num_jobs))
    queue.reverse()
    running: dict[Any, tuple[int, float | None]] = {}

    def launch() -> None:
        while queue and len(running) < max_workers:
            index = queue.pop()
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
            running[process] = (index, deadline)

    def note_done() -> None:
        nonlocal completed
        completed += 1
        if completed % progress_stride == 0 or completed == num_jobs:
            print(f"  ... {completed}/{num_jobs} jobs done", flush=True)

    def retry_or_drop(index: int, reason: str) -> None:
        nonlocal retries_used
        if attempts[index] < max_retries:
            attempts[index] += 1
            retries_used += 1
            queue.append(index)
            print(
                f"[flydsl] AOT job {_job_label(jobs[index])} {reason}; retry {attempts[index]}/{max_retries}",
                flush=True,
            )
        else:
            note_done()

    def reap(process: Any) -> None:
        index, _ = running.pop(process)
        out_path = os.path.join(result_dir, f"k{index}.json")
        try:
            if process.exitcode != 0:
                retry_or_drop(
                    index,
                    f"worker crashed (exitcode={process.exitcode})",
                )
                return

            result: dict[str, Any] | None = None
            if os.path.isfile(out_path):
                try:
                    with open(out_path, encoding="utf-8") as result_file:
                        loaded = json.load(result_file)
                    if isinstance(loaded, dict):
                        result = loaded
                except Exception:  # noqa: BLE001
                    result = None
            if result is None:
                result = {
                    "kernel_name": _job_label(jobs[index]),
                    "compile_time": None,
                }
            results[index] = result
            note_done()
        finally:
            process.close()

    try:
        launch()
        while running:
            if kernel_timeout > 0:
                nearest_deadline = min(deadline for _, deadline in running.values() if deadline is not None)
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
                    index, deadline = running[process]
                    if deadline is not None and now > deadline and process.is_alive():
                        process.kill()
                        process.join()
                        running.pop(process)
                        process.close()
                        retry_or_drop(
                            index,
                            f"exceeded per-job timeout ({kernel_timeout:.0f}s); killed",
                        )

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

    if retries_used:
        print(
            f"[flydsl] AOT: {retries_used} retr{'y' if retries_used == 1 else 'ies'} after abnormal worker exits",
            flush=True,
        )
    return results


def run_jobs_parallel(
    worker: Callable[..., dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run independent AOT jobs in parallel and return results in input order.

    Each job is expanded into ``worker(**job)``. Workers must return a
    JSON-serializable dictionary containing ``compile_time``; deterministic
    compile errors should be represented by ``compile_time=None`` rather than
    escaping as an exception. Abnormal exits and timeouts are retried before
    being normalized to the same failure shape.

    This executor uses the Linux ``fork`` multiprocessing context because it is
    intended for compile-only workers that inherit the initialized compiler.
    """
    if not jobs:
        return []

    max_workers = _get_max_workers(len(jobs))
    print(
        f"[flydsl] AOT: {len(jobs)} jobs, {max_workers} worker processes",
        flush=True,
    )

    result_dir = tempfile.mkdtemp(prefix="flydsl_aot_results_")
    try:
        raw_results = _run_file_pool(
            worker,
            jobs,
            max_workers=max_workers,
            kernel_timeout=_get_kernel_timeout(),
            max_retries=_get_max_retries(),
            result_dir=result_dir,
        )
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)

    results: list[dict[str, Any]] = []
    for job, result in zip(jobs, raw_results):
        if result is None:
            results.append(
                {
                    "kernel_name": _job_label(job),
                    "compile_time": None,
                }
            )
        else:
            results.append(result)
    return results


__all__ = ["run_jobs_parallel"]
