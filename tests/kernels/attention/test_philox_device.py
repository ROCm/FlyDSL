# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device half of the `kernels/attention/philox.py` tests (PHX-03, PHX-05, PHX-06).

PHX-04 (the word order of a forward dropout build) lands with the forward.
Also exercises the Philox seed/offset plumbing of `kernels/attention/common.py`
(null-pointer handling, the report write-back).

Builds are memoised per (width, rounds): the seed/offset pairs are runtime data.
"""

import functools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import gpu, range_constexpr  # noqa: E402
from kernels.attention import abi, common  # noqa: E402
from kernels.attention.philox import (  # noqa: E402
    DEFAULT_ROUNDS,
    PHILOX_WIDTHS,
    Philox,
    dropout_threshold,
    keep_mask,
    philox_4x,
    philox_u32,
    randoms_per_offset,
)
from tests.kernels.attention.philox_oracle import KAT_4X32_10, ref_u32  # noqa: E402

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU"),
]

_WIDTHS = pytest.mark.parametrize("width", PHILOX_WIDTHS, ids=["u32", "u64"])

_CASES = [
    (0, 0),
    (1, 0),
    (0, 1),
    (12345, 67890),
    (0xDEADBEEF, 0xCAFEBABE),
    # Above 2**32 on each side independently. A build that truncates either to 32 bits still
    # produces a perfectly random-looking stream, so only an exact comparison catches it.
    (1 << 32, 7),
    (7, 1 << 32),
    ((1 << 40) + 12345, (1 << 35) + 999),
    ((1 << 63) - 1, (1 << 63) - 1),
]


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


@functools.lru_cache(maxsize=None)
def _stream_launcher(width, n_rounds):
    """One thread block per (seed, offset) pair; the pair count is a launch argument."""
    rn = randoms_per_offset(width)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(SEED: fx.Pointer, OFF: fx.Pointer, OUT: fx.Pointer):
        i64p = fx.PointerType.get(elem_ty=fx.Int64.ir_type, address_space=fx.AddressSpace.Global, alignment=8)
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        z = fx.Int32(fx.Index(gpu.block_idx.x))
        seed = fx.ptr_load(fx.recast_iter(i64p, SEED) + fx.Int64(z))
        off = fx.ptr_load(fx.recast_iter(i64p, OFF) + fx.Int64(z))
        vals = philox_u32(seed, off, width, n_rounds)
        out = fx.recast_iter(i32p, OUT)
        # `range_constexpr`, not `range`: inside a traced kernel the latter is FlyDSL's runtime loop
        # and its induction variable cannot index a Python list.
        for j in range_constexpr(rn):
            fx.ptr_store(fx.Int32(vals[j]), out + fx.Int64(z * fx.Int32(rn) + fx.Int32(j)))

    @flyc.jit
    def launch(SEED: fx.Pointer, OFF: fx.Pointer, OUT: fx.Pointer, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        k(SEED, OFF, OUT).launch(grid=(fx.Index(n), 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def _run_device(seeds, offsets, width, n_rounds=DEFAULT_ROUNDS):
    """An (n, RN) uint32 array, one row per (seed, offset) pair."""
    rn, n = randoms_per_offset(width), len(seeds)
    ts = torch.tensor(seeds, dtype=torch.int64, device="cuda")
    to = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    out = torch.zeros(n * rn, dtype=torch.int32, device="cuda")
    _stream_launcher(width, n_rounds)(_ptr(ts), _ptr(to), _ptr(out), n, fx.Stream(None))
    torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.uint32).reshape(n, rn)


@_WIDTHS
def test_device_philox_matches_oracle(width):
    """PHX-03: bit-exact against the oracle, with 64-bit seeds and offsets."""
    seeds, offs = [s for s, _ in _CASES], [o for _, o in _CASES]
    got = _run_device(seeds, offs, width)
    for i, (s, o) in enumerate(_CASES):
        want = ref_u32(s, o, width)
        assert list(got[i]) == want, f"seed={s:#x} offset={o:#x}\n  got  {list(got[i])}\n  want {want}"
    # Truncating either to 32 bits must change the stream.
    pairs = [(1 << 32, 5), (0, 5), (5, 1 << 32), (5, 0)]
    g = _run_device([s for s, _ in pairs], [o for _, o in pairs], width)
    assert list(g[0]) != list(g[1]), "seed above 2**32 was truncated"
    assert list(g[2]) != list(g[3]), "offset above 2**32 was truncated"


@_WIDTHS
def test_rounds_are_applied(width):
    """`n_rounds` must reach the loop rather than being ignored."""
    a = _run_device([12345], [678], width, n_rounds=DEFAULT_ROUNDS)
    b = _run_device([12345], [678], width, n_rounds=DEFAULT_ROUNDS - 3)
    assert list(a[0]) != list(b[0])
    assert list(b[0]) == ref_u32(12345, 678, width, DEFAULT_ROUNDS - 3)


@_WIDTHS
def test_configured_object_matches_free_functions(width):
    """`Philox` is a wrapper, not a second implementation: same stream as the oracle."""
    rng = Philox(width=width)
    assert rng.randoms_per_offset == randoms_per_offset(width)
    seeds, offs = [s for s, _ in _CASES], [o for _, o in _CASES]
    direct = _run_device(seeds, offs, width)
    for i, (s, o) in enumerate(_CASES):
        assert list(direct[i]) == ref_u32(s, o, width)


def test_matches_random123_known_answers():
    """The round function against the algorithm's own published vectors (via `philox_4x`, 32-bit)."""
    n = len(KAT_4X32_10)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(IN: fx.Pointer, OUT: fx.Pointer):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        z = fx.Int32(fx.Index(gpu.block_idx.x))
        src = fx.recast_iter(i32p, IN)
        w = [fx.Int32(fx.ptr_load(src + fx.Int64(z * fx.Int32(6) + fx.Int32(j)))) for j in range_constexpr(6)]
        r = philox_4x(w[0], w[1], w[2], w[3], w[4], w[5], 32, DEFAULT_ROUNDS)
        out = fx.recast_iter(i32p, OUT)
        for j in range_constexpr(4):
            fx.ptr_store(fx.Int32(r[j]), out + fx.Int64(z * fx.Int32(4) + fx.Int32(j)))

    @flyc.jit
    def launch(IN: fx.Pointer, OUT: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUT).launch(grid=(fx.Index(n), 1, 1), block=(64, 1, 1), stream=stream)

    flat = [x for ctr, key, _ in KAT_4X32_10 for x in ctr + key]
    ti = torch.tensor(np.array(flat, dtype=np.uint32).astype(np.int32), dtype=torch.int32, device="cuda")
    out = torch.zeros(n * 4, dtype=torch.int32, device="cuda")
    launch(_ptr(ti), _ptr(out), fx.Stream(None))
    torch.cuda.synchronize()
    got = out.cpu().numpy().astype(np.uint32).reshape(n, 4)
    for i, (_, _, want) in enumerate(KAT_4X32_10):
        assert list(got[i]) == want, f"KAT {i}: got {[hex(x) for x in got[i]]} want {[hex(x) for x in want]}"


def test_keep_mask_compares_signed():
    """PHX-05: the unsigned-compare trap.

    `dropout_threshold(p)` is negative for every `p < 0.5`; comparing unsigned keeps everything,
    which still looks like attention.
    """
    thr = dropout_threshold(0.25)
    assert thr < 0
    vals_i32 = [-(2**31), thr - 1, thr, thr + 1, 2**31 - 1]
    want = [v > thr for v in vals_i32]
    assert want == [False, False, False, True, True], "test's own model is wrong"
    n = len(vals_i32)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(IN: fx.Pointer, OUT: fx.Pointer, threshold: fx.Int32):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        src = fx.recast_iter(i32p, IN)
        dst = fx.recast_iter(i32p, OUT)
        z = fx.Int32(fx.Index(gpu.block_idx.x))
        v = fx.Int32(fx.ptr_load(src + fx.Int64(z)))
        keep = keep_mask([v], threshold)[0]
        fx.ptr_store(fx.Int32(keep.select(fx.Int32(1), fx.Int32(0))), dst + fx.Int64(z))

    @flyc.jit
    def launch(IN: fx.Pointer, OUT: fx.Pointer, threshold: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUT, threshold).launch(grid=(fx.Index(n), 1, 1), block=(64, 1, 1), stream=stream)

    ti = torch.tensor(vals_i32, dtype=torch.int32, device="cuda")
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    launch(_ptr(ti), _ptr(out), thr, fx.Stream(None))
    torch.cuda.synchronize()
    assert [bool(x) for x in out.cpu().tolist()] == want


@_WIDTHS
def test_span_is_the_stream_read_consecutively(width):
    """PHX-06: `span_u32` equals the individual calls it stands for (absolute offset, no counter)."""
    rng = Philox(width=width)
    rn = rng.randoms_per_offset
    seed, off0, count = 0xFEED_FACE_1234, (1 << 33) + 17, 3 * rn
    want = [w for j in range(3) for w in ref_u32(seed, off0 + j, width)]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(OUT: fx.Pointer, s: fx.Int64, o: fx.Int64):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        vals = rng.span_u32(s, o, count)
        dst = fx.recast_iter(i32p, OUT)
        for j in range_constexpr(count):
            fx.ptr_store(fx.Int32(vals[j]), dst + fx.Int64(j))

    @flyc.jit
    def launch(OUT: fx.Pointer, s: fx.Int64, o: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        k(OUT, s, o).launch(grid=(fx.Index(1), 1, 1), block=(64, 1, 1), stream=stream)

    out = torch.zeros(count, dtype=torch.int32, device="cuda")
    launch(_ptr(out), seed, off0, fx.Stream(None))
    torch.cuda.synchronize()
    assert list(out.cpu().numpy().astype(np.uint32)) == want


@_WIDTHS
def test_distribution_is_plausible(width):
    """Weak by construction, kept because it catches a stuck word."""
    n = 4096
    got = _run_device(list(range(n)), [0] * n, width).astype(np.float64) / 2**32
    assert 0.45 < got.mean() < 0.55, f"mean {got.mean():.4f}"
    for j in range(got.shape[1]):
        assert got[:, j].std() > 0.2, f"word {j} looks stuck"


@functools.lru_cache(maxsize=None)
def _plumbing_launcher():
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(
        SEED_PTR: fx.Pointer,
        OFF1_PTR: fx.Pointer,
        OUT: fx.Pointer,
        SEED_OUT: fx.Pointer,
        OFF_OUT: fx.Pointer,
        offset2: fx.Int64,
    ):
        i64p = fx.PointerType.get(elem_ty=fx.Int64.ir_type, address_space=fx.AddressSpace.Global, alignment=8)
        seed = common.philox_seed_value(SEED_PTR)
        base = common.philox_offset_base(OFF1_PTR, offset2)
        out = fx.recast_iter(i64p, OUT)
        fx.ptr_store(seed, out)
        fx.ptr_store(base, out + fx.Int64(1))
        common.philox_report(SEED_OUT, OFF_OUT, seed, base)

    @flyc.jit
    def launch(
        SEED_PTR: fx.Pointer,
        OFF1_PTR: fx.Pointer,
        OUT: fx.Pointer,
        SEED_OUT: fx.Pointer,
        OFF_OUT: fx.Pointer,
        offset2: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        k(SEED_PTR, OFF1_PTR, OUT, SEED_OUT, OFF_OUT, offset2).launch(grid=(3, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("null_seed", [False, True], ids=["seed", "null_seed"])
@pytest.mark.parametrize("null_off1", [False, True], ids=["off1", "null_off1"])
@pytest.mark.parametrize("null_reports", [False, True], ids=["report", "null_report"])
def test_seed_and_offset_plumbing(null_seed, null_off1, null_reports):
    """Null seed reads as 0, a null offset pointer adds nothing (the load is skipped, not selected),
    and the report is written exactly when its pointer is non-null."""
    dev = "cuda"
    seed_t = torch.tensor([(1 << 40) + 5], dtype=torch.int64, device=dev)
    off1_t = torch.tensor([(1 << 36) + 3], dtype=torch.int64, device=dev)
    out = torch.zeros(2, dtype=torch.int64, device=dev)
    seed_out = torch.full((1,), -1, dtype=torch.int64, device=dev)
    off_out = torch.full((1,), -1, dtype=torch.int64, device=dev)
    offset2 = 77
    _plumbing_launcher()(
        abi.NULL_PTR if null_seed else _ptr(seed_t),
        abi.NULL_PTR if null_off1 else _ptr(off1_t),
        _ptr(out),
        abi.NULL_PTR if null_reports else _ptr(seed_out),
        abi.NULL_PTR if null_reports else _ptr(off_out),
        offset2,
        fx.Stream(None),
    )
    torch.cuda.synchronize()
    want_seed = 0 if null_seed else (1 << 40) + 5
    want_off = offset2 + (0 if null_off1 else (1 << 36) + 3)
    assert out.tolist() == [want_seed, want_off]
    assert seed_out.item() == (-1 if null_reports else want_seed)
    assert off_out.item() == (-1 if null_reports else want_off)
