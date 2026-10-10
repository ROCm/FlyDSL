#!/usr/bin/env python3
"""Verify gfx1250 bf16 WMMA mma_atom_call correctness with different modC values.

Single-wave (32 threads), single 16x16x32 WMMA: D = A @ B^T + modC(C).
  modC=0 (none):    D = A*B + C
  modC=1 (neg):     D = A*B - C
  modC=2 (abs):     D = A*B + |C|
  modC=3 (neg_abs): D = A*B - |C|

Each test compares GPU output against torch reference.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.runtime.device import get_rocm_arch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_arch = get_rocm_arch() or ""
_skip_not_gfx1250 = pytest.mark.skipif(not _arch.startswith("gfx1250"), reason=f"requires gfx1250, got {_arch}")

WAVE_SIZE = 32
M, N, K = 16, 16, 32


def _make_wmma_kernel(mod_c):
    """Create a WMMA kernel with a specific modC value (compile-time constant)."""

    @flyc.kernel(known_block_size=[WAVE_SIZE, 1, 1])
    def wmma_kernel(A_frag: fx.Tensor, B_frag: fx.Tensor, C_frag: fx.Tensor, D_frag: fx.Tensor):
        tid = fx.thread_idx.x

        a_rmem = fx.make_rmem_tensor(16, fx.BFloat16)
        b_rmem = fx.make_rmem_tensor(16, fx.BFloat16)
        c_rmem = fx.make_rmem_tensor(8, fx.Float32)

        for i in fx.range_constexpr(16):
            a_rmem[i] = A_frag[tid, i]
        for i in fx.range_constexpr(16):
            b_rmem[i] = B_frag[tid, i]
        for i in fx.range_constexpr(8):
            c_rmem[i] = C_frag[tid, i]

        atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, fx.BFloat16, fx.Float32, mod_c=const_expr(mod_c)))
        fx.mma_atom_call(atom, c_rmem, a_rmem, b_rmem, c_rmem)

        for i in fx.range_constexpr(8):
            D_frag[tid, i] = c_rmem[i]

    @flyc.jit
    def launch(
        A_frag: fx.Tensor,
        B_frag: fx.Tensor,
        C_frag: fx.Tensor,
        D_frag: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        wmma_kernel(A_frag, B_frag, C_frag, D_frag).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


def _build_ab_fragments(A, B):
    """Build per-thread A/B fragments matching gfx1250 WMMA bf16 register layout.

    Layout ((16,2),(8,2)):((1,128),(16,256)):
      m = l % 16,  k = (v // 8) * 16 + (l // 16) * 8 + (v % 8)
    """
    A_frag = torch.zeros(WAVE_SIZE, 16, dtype=torch.bfloat16, device=A.device)
    B_frag = torch.zeros(WAVE_SIZE, 16, dtype=torch.bfloat16, device=B.device)
    for lane in range(WAVE_SIZE):
        g = lane // 16
        m = lane % 16
        for v in range(16):
            k = (v // 8) * 16 + g * 8 + (v % 8)
            A_frag[lane, v] = A[m, k]
            B_frag[lane, v] = B[m, k]
    return A_frag, B_frag


def _pack_c_fragments(C):
    """Pack M=16 x N=16 f32 matrix into per-thread C fragments.

    Layout ((16,2),(8)):((16,8),(1)):
      m = (l // 16) * 8 + v,  n = l % 16
    """
    C_frag = torch.zeros(WAVE_SIZE, 8, dtype=torch.float32, device=C.device)
    for lane in range(WAVE_SIZE):
        g = lane // 16
        n = lane % 16
        for v in range(8):
            m = g * 8 + v
            C_frag[lane, v] = C[m, n]
    return C_frag


def _unpack_c_fragments(D_frag):
    """Unpack per-thread D fragments back to M=16 x N=16 matrix."""
    D = torch.zeros(M, N, dtype=torch.float32, device=D_frag.device)
    for lane in range(WAVE_SIZE):
        g = lane // 16
        n = lane % 16
        for v in range(8):
            m = g * 8 + v
            D[m, n] = D_frag[lane, v]
    return D


def _run_wmma(A, B, C_init, mod_c):
    """Run WMMA with given modC and return the unpacked D matrix."""
    A_frag, B_frag = _build_ab_fragments(A, B)
    C_frag = _pack_c_fragments(C_init)
    D_frag = torch.zeros(WAVE_SIZE, 8, dtype=torch.float32, device=A.device)

    launch_fn = _make_wmma_kernel(mod_c)
    launch_fn(A_frag, B_frag, C_frag, D_frag, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    return _unpack_c_fragments(D_frag)


# modC=0 (none): D = A*B + C
@_skip_not_gfx1250
def test_wmma_modc_none():
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    D_gpu = _run_wmma(A, B, C, mod_c=0)
    D_ref = A.float() @ B.float().T + C

    max_diff = (D_gpu - D_ref).abs().max().item()
    print(f"[modC=none] Max abs diff: {max_diff:.6e}")
    assert torch.allclose(D_gpu, D_ref, atol=0.05, rtol=1e-3), f"modC=none mismatch, max diff = {max_diff}"


# modC=1 (neg): D = A*B + (-C) = A*B - C
@_skip_not_gfx1250
def test_wmma_modc_neg():
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    D_gpu = _run_wmma(A, B, C, mod_c=1)
    D_ref = A.float() @ B.float().T + (-C)

    max_diff = (D_gpu - D_ref).abs().max().item()
    print(f"[modC=neg] Max abs diff: {max_diff:.6e}")
    assert torch.allclose(D_gpu, D_ref, atol=0.05, rtol=1e-3), f"modC=neg mismatch, max diff = {max_diff}"


# modC=2 (abs): D = A*B + |C|
@_skip_not_gfx1250
def test_wmma_modc_abs():
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    D_gpu = _run_wmma(A, B, C, mod_c=2)
    D_ref = A.float() @ B.float().T + C.abs()

    max_diff = (D_gpu - D_ref).abs().max().item()
    print(f"[modC=abs] Max abs diff: {max_diff:.6e}")
    assert torch.allclose(D_gpu, D_ref, atol=0.05, rtol=1e-3), f"modC=abs mismatch, max diff = {max_diff}"


# modC=3 (neg_abs): D = A*B + (-(|C|)) = A*B - |C|
@_skip_not_gfx1250
def test_wmma_modc_neg_abs():
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    D_gpu = _run_wmma(A, B, C, mod_c=3)
    D_ref = A.float() @ B.float().T - C.abs()

    max_diff = (D_gpu - D_ref).abs().max().item()
    print(f"[modC=neg_abs] Max abs diff: {max_diff:.6e}")
    assert torch.allclose(D_gpu, D_ref, atol=0.05, rtol=1e-3), f"modC=neg_abs mismatch, max diff = {max_diff}"


if __name__ == "__main__":
    test_wmma_modc_none()
    test_wmma_modc_neg()
    test_wmma_modc_abs()
    test_wmma_modc_neg_abs()
    print("ALL PASS")
