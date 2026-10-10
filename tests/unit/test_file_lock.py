# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import multiprocessing
import os

from flydsl.compiler.jit_function import FileLock


def _try_lock(path, exclusive, timeout, results):
    try:
        with FileLock(path, exclusive=exclusive, timeout=timeout):
            results.put(True)
    except RuntimeError:
        results.put(False)


def _run_process(path, exclusive, timeout):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=_try_lock, args=(path, exclusive, timeout, results))
    process.start()
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join()
        raise AssertionError("lock worker did not finish")
    assert process.exitcode == 0
    return results.get(timeout=1)


def test_file_lock_process_timeout_and_release(tmp_path):
    lock_path = str(tmp_path / "cache.lock")
    with FileLock(lock_path):
        assert not _run_process(lock_path, True, 3)
    assert _run_process(lock_path, True, 3)


def test_file_lock_shared_mode(tmp_path):
    lock_path = str(tmp_path / "cache.lock")
    with FileLock(lock_path, exclusive=False):
        acquired = _run_process(lock_path, False, 3)
    assert acquired is (os.name != "nt")
