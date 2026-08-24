#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The specialization machinery every block collective shares.

``coop/block/_spec.py`` parses ``[dtype, block_size, algorithm]``, resolves the
default policy, and caches the class it builds. What is checked here is the
contract an algorithm has to hold up when it plugs into that: the default is
answered per target and never read off a constant, an algorithm that answers
nothing is refused rather than defaulted, a policy the enum names but does not
implement is refused by name, and the cache keeps two targets' answers apart.

None of it reaches a device. A specialization is Python, so these run wherever
a backend can name its target.
"""

from __future__ import annotations

import enum

import pytest

import flydsl.expr as fx
from flydsl.compiler.backends import GPUTarget, current_target
from flydsl.extension.coop.block._spec import BlockAlgorithmMeta

CDNA = GPUTarget(backend="rocm", arch="gfx942", warp_size=64)
RDNA = GPUTarget(backend="rocm", arch="gfx1201", warp_size=32)


class _Policy(enum.Enum):
    ALPHA = "alpha"
    BETA = "beta"


def _storage(dtype, block_threads, warp_threads):
    return fx.Struct["slots" : fx.Array[dtype, 1]]


class _PolicyMeta(BlockAlgorithmMeta):
    """What every algorithm supplies; the answer itself is left to a subclass."""

    _algorithms = _Policy
    _shared_storage = {_Policy.ALPHA: _storage, _Policy.BETA: _storage}


class _PinnedMeta(_PolicyMeta):
    """An algorithm whose answer happens not to vary — still stated, not assumed."""

    def _default_algorithm_for(cls, target):
        return _Policy.ALPHA


class _ByWaveMeta(_PolicyMeta):
    """An algorithm whose answer follows the target's wave size."""

    def _default_algorithm_for(cls, target):
        return _Policy.BETA if target.warp_size == 32 else _Policy.ALPHA


class Pinned(metaclass=_PinnedMeta):
    block_threads = None


class ByWave(metaclass=_ByWaveMeta):
    block_threads = None


# ── the hook decides the default ──────────────────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
def test_an_unspecified_algorithm_comes_from_the_hook():
    """Leaving the policy out is what asks the hook, and its answer is what sticks."""
    assert Pinned[fx.Float32, 64].algorithm is _Policy.ALPHA
    # The same call on an algorithm that answers from the target: this machine's wave.
    expected = _Policy.BETA if current_target().warp_size == 32 else _Policy.ALPHA
    assert ByWave[fx.Float32, 64].algorithm is expected


@pytest.mark.l1a_compile_no_target_dialect
def test_an_algorithm_that_does_not_answer_is_refused():
    """There is no target-independent fallback, so silence is an error, not a default."""

    class _SilentMeta(_PolicyMeta):
        pass

    class Silent(metaclass=_SilentMeta):
        block_threads = None

    with pytest.raises(NotImplementedError, match="_default_algorithm_for"):
        Silent[fx.Float32, 64]


@pytest.mark.l1a_compile_no_target_dialect
def test_the_hook_is_handed_the_target_being_compiled_for():
    """``__getitem__`` resolves against ``current_target()``, not a cached one."""
    seen = []

    class _RecordingMeta(_PinnedMeta):
        def _default_algorithm_for(cls, target):
            seen.append(target)
            return _Policy.ALPHA

    class Recording(metaclass=_RecordingMeta):
        block_threads = None

    Recording[fx.Float32, 64]
    assert seen == [current_target()]


@pytest.mark.l1a_compile_no_target_dialect
def test_an_override_answers_per_target():
    """The hook is a pure function of the target, so both answers are reachable."""
    assert _ByWaveMeta._default_algorithm_for(ByWave, CDNA) is _Policy.ALPHA
    assert _ByWaveMeta._default_algorithm_for(ByWave, RDNA) is _Policy.BETA


@pytest.mark.l1a_compile_no_target_dialect
def test_a_named_algorithm_bypasses_the_hook():
    """An explicit policy is the caller's, whatever the target would have picked."""

    class _NeverMeta(_PinnedMeta):
        def _default_algorithm_for(cls, target):
            raise AssertionError("the hook must not run when the caller names a policy")

    class Never(metaclass=_NeverMeta):
        block_threads = None

    assert Never[fx.Float32, 64, _Policy.BETA].algorithm is _Policy.BETA


# ── a policy the enum names but does not implement ────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
def test_a_policy_that_is_not_implemented_is_refused_by_name():
    """The enum names planned variants too, and asking for one says so.

    Only a caller who went looking past the default can reach this — the
    default is always a policy that is implemented.
    """
    with pytest.raises(NotImplementedError, match="BlockScanAlgorithm.RAKING_MEMOIZE is not implemented"):
        fx.coop.BlockScan[fx.Float32, 64, fx.coop.BlockScanAlgorithm.RAKING_MEMOIZE]


# ── the cache keeps the answers apart ─────────────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
def test_two_defaults_do_not_share_a_cache_entry():
    """The resolved policy is part of the key, so a per-target default separates.

    This is what makes the hook safe to override: were the default resolved
    after the key was built, the first target through would pin the class for
    every later one.
    """
    alpha = Pinned[fx.Float32, 64, _Policy.ALPHA]
    beta = Pinned[fx.Float32, 64, _Policy.BETA]
    assert alpha is not beta
    assert alpha is Pinned[fx.Float32, 64]  # the default, resolved to ALPHA


@pytest.mark.l1a_compile_no_target_dialect
def test_an_identical_specialization_is_reused():
    """Same class, same parameters, same target: the cached specialization."""
    assert Pinned[fx.Float32, 64] is Pinned[fx.Float32, 64]


# ── what the shipped collectives currently choose ─────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("target", (CDNA, RDNA), ids=lambda t: t.arch)
def test_the_shipped_defaults_are_target_independent(target):
    """Neither collective overrides the hook yet, and this is what says so.

    ``BlockReduce`` measured ``WARP_REDUCTIONS`` ahead of ``RAKING`` on both a
    wave64 and a wave32 target, and ``BlockScan`` has only one policy
    implemented. A target that inverts either is what would make an override
    the right change — and would land here first.
    """
    assert (
        type(fx.coop.BlockReduce)._default_algorithm_for(fx.coop.BlockReduce, target)
        is fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS
    )
    assert (
        type(fx.coop.BlockScan)._default_algorithm_for(fx.coop.BlockScan, target)
        is fx.coop.BlockScanAlgorithm.WARP_SCANS
    )


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("target", (CDNA, RDNA), ids=lambda t: t.arch)
def test_the_portable_collectives_inherit_the_same_hook(target):
    """``fx.coop.universal`` subclasses the dispatched pair, hook included.

    It overrides ``warp_ops`` and nothing else, so a default that started
    answering per target would reach the portable spelling too — which is what
    keeps the two from drifting into different policies.
    """
    for portable, dispatched in (
        (fx.coop.universal.BlockReduce, fx.coop.BlockReduce),
        (fx.coop.universal.BlockScan, fx.coop.BlockScan),
    ):
        assert type(portable)._default_algorithm_for(portable, target) is type(dispatched)._default_algorithm_for(
            dispatched, target
        )
