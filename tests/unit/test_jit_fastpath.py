# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Semantics tests for the ``JitFunction.__call__`` cache-hit fast path.

The fast path maps a cheap per-call key straight to the ``CallState`` that the
slow path (``_compile_or_execute``) already resolved. These tests pin its
contract so the implementation can be refactored freely:

  * warm calls take the fast path (the slow path runs once per new specialization);
  * the fast key is never coarser than the full cache key (dtype, layout,
    Constexpr values, kwargs/defaults);
  * anything else the slow path depends on -- cache-key env vars, compile hints
    (thread-local and persistent, including in-place edits), captured globals
    (rebind and in-place mutation, incl. dict subclasses), device-runtime
    registration, cache resets -- sends the call back to the unchanged slow path,
    which produces the same results / errors as before;
  * ``FLYDSL_JIT_FASTPATH=0`` (``_FASTPATH_ENABLED``) turns it off.
"""
import collections
import gc

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.compiler.jit_function as jf  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

SIZE = 1024 * 64
BLOCK = 256


@flyc.kernel
def addKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width
    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))
    copy_bits = vec_width * 32
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)
    rA = fx.make_rmem_tensor(vec_width, fx.Float32)
    rB = fx.make_rmem_tensor(vec_width, fx.Float32)
    rC = fx.make_rmem_tensor(vec_width, fx.Float32)
    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)
    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


# Globals captured by the launchers below (drift tests mutate them).
GRID_DIV = 1
CFG = {"div": 1}
OCFG = collections.OrderedDict(div=1)  # dict *subclass* global (review regression)


@flyc.jit
def vecAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    addKernel(A, B, C, block_dim, vec_width).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


@flyc.jit
def vecAddG(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width * GRID_DIV
    grid_x = (n + tile_elems - 1) // tile_elems
    addKernel(A, B, C, block_dim, vec_width).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


@flyc.jit
def vecAddD(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width * CFG["div"]
    grid_x = (n + tile_elems - 1) // tile_elems
    addKernel(A, B, C, block_dim, vec_width).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


@flyc.jit
def vecAddO(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width * OCFG["div"]
    grid_x = (n + tile_elems - 1) // tile_elems
    addKernel(A, B, C, block_dim, vec_width).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


class Counter:
    """Counts executions that went through the slow path."""

    def __init__(self, monkeypatch):
        self.slow = 0
        orig = jf.JitFunction._compile_or_execute

        def wrapped(this, args, kwargs, *, execute, **kw):
            if execute:
                self.slow += 1
            return orig(this, args, kwargs, execute=execute, **kw)

        monkeypatch.setattr(jf.JitFunction, "_compile_or_execute", wrapped)


@pytest.fixture
def cnt(monkeypatch):
    monkeypatch.setattr(jf, "_FASTPATH_ENABLED", True)
    return Counter(monkeypatch)


def _inputs(size=SIZE):
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.zeros_like(a)
    return a, b, c


def _check(a, b, c):
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)


def test_warm_calls_take_fast_path(cnt):
    a, b, c = _inputs()
    for i in range(6):
        c.zero_()
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        _check(a, b, c)
    assert cnt.slow == 1
    assert len(vecAdd._fast_cache) >= 1


def test_new_buffers_and_sizes_hit_fast_path(cnt):
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    base = cnt.slow
    for size in (SIZE // 2, SIZE * 2, SIZE):
        a, b, c = _inputs(size)
        vecAdd(a, b, c, size, BLOCK, 4)
        _check(a, b, c)
    assert cnt.slow == base  # dynamic layout: shape/ptr changes reuse the same kernel


def test_constexpr_change_is_a_different_key(cnt):
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    s0 = cnt.slow
    c.zero_()
    vecAdd(a, b, c, SIZE, BLOCK, 2)  # new Constexpr -> slow path (new kernel)
    _check(a, b, c)
    assert cnt.slow == s0 + 1
    c.zero_()
    vecAdd(a, b, c, SIZE, BLOCK, 2)
    _check(a, b, c)
    assert cnt.slow == s0 + 1
    # bool vs int Constexpr values are distinct keys in the slow path too
    k1 = vecAdd._fast_key(None, [a, b, c, SIZE, BLOCK, 2, vecAdd._fast_spec[2][-1]], resolve=True)
    k2 = vecAdd._fast_key(None, [a, b, c, SIZE, BLOCK, True, vecAdd._fast_spec[2][-1]], resolve=True)
    assert k1 != k2


def test_kwargs_and_defaults_match_bind(cnt):
    a, b, c = _inputs()
    s = torch.cuda.current_stream()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    s0 = cnt.slow
    for call in (
        lambda: vecAdd(a, b, c, SIZE, BLOCK, 4, s),
        lambda: vecAdd(a, b, c, SIZE, block_dim=BLOCK, vec_width=4),
        lambda: vecAdd(A=a, B=b, C=c, n=SIZE, block_dim=BLOCK, vec_width=4, stream=s),
        lambda: vecAdd(a, b, c, n=SIZE, block_dim=BLOCK, vec_width=4, stream=None),
    ):
        c.zero_()
        call()
        _check(a, b, c)
    # stream=None / torch stream / default Stream map to stream-type signatures;
    # torch.cuda.Stream is a new arg type for that param -> at most one extra slow call each
    assert cnt.slow - s0 <= 3
    # malformed calls still raise the slow path's TypeError
    with pytest.raises(TypeError):
        vecAdd(a, b, c, SIZE, BLOCK)
    with pytest.raises(TypeError):
        vecAdd(a, b, c, SIZE, BLOCK, 4, None, 7)
    with pytest.raises(TypeError):
        vecAdd(a, b, c, SIZE, BLOCK, 4, bogus=1)


def test_dtype_and_layout_are_part_of_key(cnt):
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    k_f32 = vecAdd._fast_key(None, [a, b, c, SIZE, BLOCK, 4, None], resolve=True)
    k_f16 = vecAdd._fast_key(None, [a.half(), b, c, SIZE, BLOCK, 4, None], resolve=True)
    a2 = torch.randn(4, SIZE // 4, device="cuda")
    k_2d = vecAdd._fast_key(None, [a2, b, c, SIZE, BLOCK, 4, None], resolve=True)
    a2t = a2.t()  # unit stride on dim 0 now
    k_2dt = vecAdd._fast_key(None, [a2t, b, c, SIZE, BLOCK, 4, None], resolve=True)
    assert len({k_f32, k_f16, k_2d, k_2dt}) == 4
    # no unit stride -> fast key refuses; slow path raises the same error as before
    strided = torch.randn(SIZE * 2, device="cuda")[::2]
    assert vecAdd._fast_key(None, [strided, b, c, SIZE, BLOCK, 4, None], resolve=True) is None
    with pytest.raises(Exception):
        vecAdd(strided, b, c, SIZE, BLOCK, 4)


def test_slow_and_fast_keys_agree_on_identity(cnt):
    """Same fast key <=> same full cache key, over a mix of arguments."""
    a, b, c = _inputs()
    cases = []
    for t in (a, a.half(), torch.randn(8, 16, device="cuda"), torch.randn(16, 8, device="cuda").t()):
        for n in (SIZE, 7, True):
            for vw in (4, 2, True):
                cases.append([t, b, c, n, BLOCK, vw, None])
    vecAdd._ensure_sig()
    if vecAdd._fast_spec is jf._MISSING:
        vecAdd._fast_spec = jf._fast_param_spec(vecAdd._sig)
        vecAdd._fast_handlers = tuple({} for _ in vecAdd._fast_spec[0])
    vecAdd._used_global_vals.setdefault(None, jf._snapshot_refs(vecAdd._get_global_refs(None), stable=False))
    fast_to_full = {}
    for vals in cases:
        fk = vecAdd._fast_key(None, list(vals), resolve=True)
        bound = vecAdd._sig.bind(*vals)
        bound.apply_defaults()
        full = vecAdd._build_full_cache_key(bound.arguments)
        if fk is None:
            continue
        prev = fast_to_full.setdefault(fk, full)
        assert prev == full, (fk, prev, full)


def test_global_rebind_still_raises(cnt):
    global GRID_DIV
    a, b, c = _inputs()
    vecAddG(a, b, c, SIZE, BLOCK, 4)
    vecAddG(a, b, c, SIZE, BLOCK, 4)
    old = GRID_DIV
    try:
        GRID_DIV = 2
        with pytest.raises(RuntimeError, match="changed since first compile"):
            vecAddG(a, b, c, SIZE, BLOCK, 4)
    finally:
        GRID_DIV = old
    c.zero_()
    vecAddG(a, b, c, SIZE, BLOCK, 4)  # restored -> runs again
    _check(a, b, c)


def test_global_inplace_mutation_still_raises(cnt):
    a, b, c = _inputs()
    vecAddD(a, b, c, SIZE, BLOCK, 4)
    vecAddD(a, b, c, SIZE, BLOCK, 4)
    try:
        CFG["div"] = 2
        with pytest.raises(RuntimeError, match="changed since first compile"):
            vecAddD(a, b, c, SIZE, BLOCK, 4)
    finally:
        CFG["div"] = 1
    c.zero_()
    vecAddD(a, b, c, SIZE, BLOCK, 4)
    _check(a, b, c)


def test_env_change_goes_through_slow_path(cnt, monkeypatch):
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    s0 = cnt.slow
    n_states = len(vecAdd._call_state_cache)
    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "1")
    c.zero_()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    _check(a, b, c)
    assert cnt.slow == s0 + 1
    assert len(vecAdd._call_state_cache) == n_states + 1  # env is in the full key
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    assert cnt.slow == s0 + 1  # new epoch is fast again
    monkeypatch.delenv("FLYDSL_COMPILE_OPT_LEVEL")
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    assert cnt.slow == s0 + 2
    # COMPILE_ONLY flips the hit path to "return None" -- must not be skipped
    monkeypatch.setenv("COMPILE_ONLY", "1")
    c.zero_()
    assert vecAdd(a, b, c, SIZE, BLOCK, 4) is None
    torch.cuda.synchronize()
    assert torch.count_nonzero(c).item() == 0
    monkeypatch.delenv("COMPILE_ONLY")


def test_compile_hints_bypass_fast_path(cnt):
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    s0 = cnt.slow
    with CompilationContext.compile_hints({"waves_per_eu": 2}):
        c.zero_()
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        _check(a, b, c)
    assert cnt.slow == s0 + 1
    # persistent hints: mutate the function's hint dict -> new epoch via slow path
    old = vecAdd.compile_hints
    try:
        vecAdd.compile_hints = {"waves_per_eu": 2}
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        assert cnt.slow == s0 + 2
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        assert cnt.slow == s0 + 2
        vecAdd.compile_hints["waves_per_eu"] = 1  # in-place
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        assert cnt.slow == s0 + 3
    finally:
        vecAdd.compile_hints = old


def test_disable_switch(cnt, monkeypatch):
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    monkeypatch.setattr(jf, "_FASTPATH_ENABLED", False)
    s0 = cnt.slow
    for _ in range(3):
        vecAdd(a, b, c, SIZE, BLOCK, 4)
    assert cnt.slow == s0 + 3


def test_compile_api_still_works():
    a, b, c = _inputs()
    stream = torch.cuda.current_stream()
    compiled = flyc.compile(vecAdd, a, b, c, SIZE, BLOCK, 4, stream)
    c.zero_()
    compiled(a, b, c, SIZE, BLOCK, 4, stream)
    _check(a, b, c)


def test_cache_reset_invalidates_fast_path(cnt):
    """A CallState does not own its code object: clearing the JIT caches must not
    leave a dangling fast-path entry (regression for FlyDSL test_compile_hints
    test_cache_disabled_run_perftest_does_not_crash)."""
    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    s0 = cnt.slow
    vecAdd._call_state_cache.clear()
    vecAdd._mem_cache.clear()
    vecAdd._last_compiled = None
    gc.collect()
    c.zero_()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    _check(a, b, c)
    assert cnt.slow == s0 + 1
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    assert cnt.slow == s0 + 1  # re-registered against the fresh CallState
    # replacing the cache with a plain dict: fast path stays off, results still correct
    old = vecAdd._call_state_cache
    try:
        vecAdd._call_state_cache = {}
        for _ in range(3):
            c.zero_()
            vecAdd(a, b, c, SIZE, BLOCK, 4)
            _check(a, b, c)
        assert cnt.slow == s0 + 4
    finally:
        vecAdd._call_state_cache = old
        old.clear()


# ---- regressions found in review ----


def test_dict_subclass_global_inplace_mutation_still_raises(cnt):
    """OrderedDict/defaultdict/namedtuple globals are snapshotted by value by the
    slow path; the fast path must not treat them as opaque (identity-only)."""
    a, b, c = _inputs()
    vecAddO(a, b, c, SIZE, BLOCK, 4)
    vecAddO(a, b, c, SIZE, BLOCK, 4)
    try:
        OCFG["div"] = 2
        with pytest.raises(RuntimeError, match="changed since first compile"):
            vecAddO(a, b, c, SIZE, BLOCK, 4)
    finally:
        OCFG["div"] = 1
    c.zero_()
    vecAddO(a, b, c, SIZE, BLOCK, 4)
    _check(a, b, c)


def test_hints_cleared_in_place_goes_slow(cnt):
    a, b, c = _inputs()
    old = vecAdd.compile_hints
    try:
        vecAdd.compile_hints = {"waves_per_eu": 2}
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        s0 = cnt.slow
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        assert cnt.slow == s0  # warm -> fast
        vecAdd.compile_hints.clear()  # non-empty -> empty, in place: different full key
        n_states = len(vecAdd._call_state_cache)
        c.zero_()
        vecAdd(a, b, c, SIZE, BLOCK, 4)
        _check(a, b, c)
        assert cnt.slow == s0 + 1
        # the no-hint key exists from earlier tests or is compiled now; either way the
        # slow path (not a stale hinted CallState) served it
        assert len(vecAdd._call_state_cache) >= n_states
    finally:
        vecAdd.compile_hints = old


def test_runtime_map_inplace_change_goes_slow(cnt):
    from flydsl.runtime import device_runtime as dr

    a, b, c = _inputs()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    m = dr.COMPILE_BACKEND_TO_RUNTIME_KIND
    old = m["rocm"]
    try:
        m["rocm"] = "bogus"  # same dict, same len: slow path's pairing check raises
        with pytest.raises(RuntimeError, match="requires device runtime kind"):
            vecAdd(a, b, c, SIZE, BLOCK, 4)
    finally:
        m["rocm"] = old
    c.zero_()
    vecAdd(a, b, c, SIZE, BLOCK, 4)
    _check(a, b, c)
