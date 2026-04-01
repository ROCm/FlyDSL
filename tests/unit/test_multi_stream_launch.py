# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Test multi-stream kernel launch patterns.

Covers:
  1. Independent kernels on different streams (single @jit)
  2. Same stream for both kernels (sequential)
  3. Default streams
  4. Graph capture with multi-stream
  5. Cross-stream data dependency with same-stream ordering
  6. Cross-stream dependency with external event sync (diamond fork-join)
  7. Cross-stream dependency without sync (race condition)
"""
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_DIM = 256
VEC_WIDTH = 4
TILE = BLOCK_DIM * VEC_WIDTH
SIZE = TILE * 100


@flyc.kernel
def _add_kernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
):
    """C = A + B"""
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile = block_dim * vec_width
    tA = fx.slice(fx.logical_divide(A, fx.make_layout(tile, 1)), (None, bid))
    tB = fx.slice(fx.logical_divide(B, fx.make_layout(tile, 1)), (None, bid))
    tC = fx.slice(fx.logical_divide(C, fx.make_layout(tile, 1)), (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))
    copy_bits = vec_width * 32
    reg_ty = fx.MemRefType.get(
        fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register)
    atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)
    rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    fx.copy_atom_call(atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(atom, fx.slice(tB, (None, tid)), rB)
    fx.memref_store_vec(
        fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB)), rC)
    fx.copy_atom_call(atom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def _add_jit(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, n: fx.Int32,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile = block_dim * vec_width
    grid_x = (n + tile - 1) // tile
    _add_kernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


@flyc.jit
def _two_stream_launch(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    D: fx.Tensor, E: fx.Tensor, F: fx.Tensor,
    n: fx.Int32,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
    stream1: fx.Stream = fx.Stream(None),
    stream2: fx.Stream = fx.Stream(None),
):
    """C = A + B on stream1, F = D + E on stream2."""
    tile = block_dim * vec_width
    grid_x = (n + tile - 1) // tile
    _add_kernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream1)
    _add_kernel(D, E, F, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream2)


# ─────────────────────────────────────────────────────────────
# Independent multi-stream launch
# ─────────────────────────────────────────────────────────────

class TestMultiStreamLaunch:

    def test_two_streams_independent(self):
        A, B, C = [torch.randn(SIZE, device="cuda") for _ in range(3)]
        D, E, F = [torch.randn(SIZE, device="cuda") for _ in range(3)]
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

        _two_stream_launch(A, B, C, D, E, F, SIZE, BLOCK_DIM, VEC_WIDTH,
                           stream1=s1, stream2=s2)
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        torch.cuda.synchronize()

        assert torch.allclose(C, A + B, atol=1e-5)
        assert torch.allclose(F, D + E, atol=1e-5)

    def test_same_stream_both_kernels(self):
        A, B, C = [torch.randn(SIZE, device="cuda") for _ in range(3)]
        D, E, F = [torch.randn(SIZE, device="cuda") for _ in range(3)]
        s = torch.cuda.Stream()

        _two_stream_launch(A, B, C, D, E, F, SIZE, BLOCK_DIM, VEC_WIDTH,
                           stream1=s, stream2=s)
        torch.cuda.synchronize()

        assert torch.allclose(C, A + B, atol=1e-5)
        assert torch.allclose(F, D + E, atol=1e-5)

    def test_default_streams(self):
        A, B, C = [torch.randn(SIZE, device="cuda") for _ in range(3)]
        D, E, F = [torch.randn(SIZE, device="cuda") for _ in range(3)]

        _two_stream_launch(A, B, C, D, E, F, SIZE, BLOCK_DIM, VEC_WIDTH)
        torch.cuda.synchronize()

        assert torch.allclose(C, A + B, atol=1e-5)
        assert torch.allclose(F, D + E, atol=1e-5)

    def test_graph_capture_multi_stream(self):
        A, B, C = [torch.randn(SIZE, device="cuda") for _ in range(3)]
        D, E, F = [torch.randn(SIZE, device="cuda") for _ in range(3)]

        # Warmup
        s = torch.cuda.Stream()
        _two_stream_launch(A, B, C, D, E, F, SIZE, BLOCK_DIM, VEC_WIDTH,
                           stream1=s, stream2=s)
        torch.cuda.synchronize()

        C.zero_(); F.zero_()
        graph = torch.cuda.CUDAGraph()
        cs = torch.cuda.Stream()
        cs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cs):
            with torch.cuda.graph(graph, stream=cs):
                _two_stream_launch(A, B, C, D, E, F, SIZE, BLOCK_DIM, VEC_WIDTH,
                                   stream1=cs, stream2=cs)
        C.zero_(); F.zero_()
        graph.replay()
        torch.cuda.synchronize()

        assert torch.allclose(C, A + B, atol=1e-5)
        assert torch.allclose(F, D + E, atol=1e-5)


# ─────────────────────────────────────────────────────────────
# Cross-stream dependencies
# ─────────────────────────────────────────────────────────────

class TestCrossStreamDependency:

    def test_same_stream_dependent_correct(self):
        """Dependent kernels on same stream — ordering guaranteed."""
        A = torch.randn(SIZE, device="cuda")
        B = torch.randn(SIZE, device="cuda")
        C = torch.empty_like(A)
        D = torch.randn(SIZE, device="cuda")
        E = torch.empty_like(A)
        s = torch.cuda.Stream()

        # C = A + B, then E = C + D (same stream → sequential)
        _add_jit(A, B, C, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s)
        _add_jit(C, D, E, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s)
        torch.cuda.synchronize()

        assert torch.allclose(C, A + B, atol=1e-5)
        assert torch.allclose(E, (A + B) + D, atol=1e-5)

    def test_diamond_pipeline_with_event_sync(self):
        """Diamond fork-join: 3 streams with event-based sync.

            Stream1: C = A + B   ─┐
                                   ├─ event sync ─> Stream3: F = C + E
            Stream2: E = Z + D   ─┘
        """
        A = torch.arange(SIZE, dtype=torch.float32, device="cuda")
        B = torch.ones(SIZE, device="cuda")
        C = torch.zeros(SIZE, device="cuda")
        Z = torch.zeros(SIZE, device="cuda")
        D = torch.ones(SIZE, device="cuda") * 10
        E = torch.zeros(SIZE, device="cuda")
        F = torch.zeros(SIZE, device="cuda")

        s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()

        _add_jit(A, B, C, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s1)
        _add_jit(Z, D, E, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s2)

        ev1 = s1.record_event()
        ev2 = s2.record_event()
        s3.wait_event(ev1)
        s3.wait_event(ev2)

        _add_jit(C, E, F, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s3)

        torch.cuda.current_stream().wait_stream(s3)
        torch.cuda.synchronize()

        assert torch.allclose(C, A + B, atol=1e-5)
        assert torch.allclose(E, Z + D, atol=1e-5)
        assert torch.allclose(F, (A + B) + (Z + D), atol=1e-5)

    def test_no_sync_may_race(self):
        """Without sync, cross-stream dependency may produce wrong results."""
        race_detected = False
        for trial in range(30):
            A = torch.ones(SIZE, device="cuda")
            B = torch.ones(SIZE, device="cuda")
            C = torch.zeros(SIZE, device="cuda")
            D = torch.ones(SIZE, device="cuda") * 5
            E = torch.zeros(SIZE, device="cuda")
            F = torch.zeros(SIZE, device="cuda")

            s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()

            _add_jit(A, B, C, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s1)
            _add_jit(torch.zeros_like(D), D, E, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s2)
            _add_jit(C, E, F, SIZE, BLOCK_DIM, VEC_WIDTH, stream=s3)

            torch.cuda.synchronize()

            expected = (A + B) + (torch.zeros_like(D) + D)
            if not torch.allclose(F, expected, atol=1e-5):
                race_detected = True
                wrong = (torch.abs(F - expected) > 1e-5).sum().item()
                print(f"\n  Trial {trial}: Race! {wrong}/{SIZE} wrong. "
                      f"F[0]={F[0].item():.1f} expected={expected[0].item():.1f}")
                break

        if race_detected:
            print("  Race confirmed — cross-stream sync is required.")
        else:
            print(f"\n  No race in 30 trials (GPU may serialize streams).")
