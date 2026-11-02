# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""``flyc.observe_launches``: the cache identity of each ``@flyc.jit`` launch, and stopping a launch there."""

import pickle
import threading

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_function import JitCacheManager


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    return tmp_path


def _launchers():
    # A JitFunction binds its disk cache on its first launch, so each test makes its own.
    @flyc.jit
    def inner(a: fx.Tensor):
        pass

    @flyc.jit
    def launch(a: fx.Tensor, n: fx.Constexpr[int]):
        inner(a)

    return launch, inner


def _cached(cache_dir, record):
    path = cache_dir / f"{record.function}_{record.manager_key}"
    return JitCacheManager(path).get(record.cache_key) is not None


def test_records_each_launch_as_the_cache_stores_it(cache_dir):
    launch, _ = _launchers()
    a = torch.zeros(8)
    with flyc.observe_launches() as launches:
        launch(a, 1)
        launch(a, 1)
        flyc.compile(launch, a, 2)
    # inner is inlined into the trace of launch, so it is not a launch of its own.
    assert [r.function for r in launches] == ["launch"] * 3
    assert launches[0] == launches[1] != launches[2]
    assert all(_cached(cache_dir, r) for r in launches)


def test_stop_ends_a_launch_at_its_identity(cache_dir):
    launch, _ = _launchers()
    a = torch.zeros(8)
    with flyc.observe_launches(stop=True) as stopped:
        assert launch(a, 1) is None
        assert flyc.compile(launch, a, 1) is None
    assert not _cached(cache_dir, stopped[0])

    with flyc.observe_launches() as launched:
        launch(a, 1)
    assert stopped == launched * 2
    assert _cached(cache_dir, launched[0])


def test_stop_reads_no_artifact_of_a_populated_cache(cache_dir, monkeypatch):
    first, first_inner = _launchers()
    a = torch.zeros(8)
    first_inner(a)
    with flyc.observe_launches() as populated:
        first(a, 1)

    loaded = []
    real_load = pickle.load

    def load(file, *args, **kwargs):
        loaded.append(file.name)
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(pickle, "load", load)
    # A fresh launcher of the same source, and its nested inner, bind the populated directories.
    launch, _ = _launchers()
    with flyc.observe_launches(stop=True) as stopped:
        launch(a, 1)
    assert stopped == populated
    assert loaded == []

    launch(a, 1)
    assert loaded


def test_observers_nest_and_see_only_their_thread(cache_dir):
    launch, _ = _launchers()
    a = torch.zeros(8)
    seen_by_thread = []

    def launch_in_thread():
        with flyc.observe_launches(stop=True) as records:
            launch(a, 2)
        seen_by_thread.extend(records)

    with flyc.observe_launches() as outer:
        with flyc.observe_launches(stop=True) as inner:
            launch(a, 1)
            thread = threading.Thread(target=launch_in_thread)
            thread.start()
            thread.join()
        launch(a, 3)
    launch(a, 4)

    assert len(inner) == 1 and not _cached(cache_dir, inner[0])
    assert outer == [inner[0], outer[1]] and outer[1] != inner[0]
    assert len(seen_by_thread) == 1 and seen_by_thread[0] not in outer
