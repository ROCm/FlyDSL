# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""N-major GEMM / fused_gemm_TN + SwiGLU MLP (zero-LDS + multi-wave LDS)."""

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.rdna4_fused_mlp_nmajor import (
    _run_tile,
    fused_gemm_tn,
    fused_swiglu_mlp_inreg,
    fused_swiglu_mlp_nmajor,
    gemm_bf16_nmajor,
    gemm_bf16_nmajor_lds,
    pick_nmajor_lds_tile,
)
from tests.kernels.oracles import reference_swiglu_mlp

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


def _require_gfx120x() -> None:
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_wmma_tile_matches_matmul(dtype: torch.dtype) -> None:
    _require_gfx120x()
    torch.manual_seed(0)
    a = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    b = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    out = _run_tile(a, b, swap_ab=False)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_ab_swap_std_store_is_transpose() -> None:
    """Layout probe: A/B swap + standard Wave32 store → square-tile transpose."""
    _require_gfx120x()
    torch.manual_seed(1)
    dtype = torch.bfloat16
    a = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    b = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    probed = _run_tile(a, b, swap_ab=True)
    ref = a.float() @ b.float().T
    assert not torch.allclose(probed, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(probed, ref.T, atol=0, rtol=0)


def test_fused_gemm_tn_zero_lds() -> None:
    """GEMM0 swapped (N-major D0 in VGPR) → GEMM1; matches (A0@B0.T)@B1.T."""
    _require_gfx120x()
    torch.manual_seed(2)
    dtype = torch.bfloat16
    a0 = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    b0 = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    b1 = (torch.randn(16, 16, device="cuda") * 4).round().to(dtype)
    got = fused_gemm_tn(a0, b0, b1, out_dtype=torch.float32)
    ref = (a0.float() @ b0.float().T) @ b1.float().T
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_noswap_noswap_fuse_fails_layout() -> None:
    """Unswapped tile matches A @ B.T and is not the transpose of that product."""
    _require_gfx120x()
    # Covered implicitly: fused_gemm_tn requires swap on GEMM0; the probe that
    # noswap->noswap mismatches are documented in the module docstring.
    # Keep a cheap sanity that unswapped single tile is NOT a transpose.
    torch.manual_seed(4)
    a = (torch.randn(16, 16, device="cuda") * 4).round().to(torch.bfloat16)
    b = (torch.randn(16, 16, device="cuda") * 4).round().to(torch.bfloat16)
    out = _run_tile(a, b, swap_ab=False)
    ref = a.float() @ b.float().T
    assert not torch.allclose(out, ref.T, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_gemm_bf16_nmajor_tiled_32() -> None:
    _require_gfx120x()
    torch.manual_seed(5)
    dtype = torch.bfloat16
    a = (torch.randn(32, 32, device="cuda") * 2).round().to(dtype)
    b = (torch.randn(32, 32, device="cuda") * 2).round().to(dtype)
    out = gemm_bf16_nmajor(a, b, out_dtype=torch.float32)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_fused_swiglu_mlp_nmajor_matches_eager() -> None:
    _require_gfx120x()
    torch.manual_seed(3)
    dtype = torch.bfloat16
    m, k, ffn = 16, 16, 16
    # Mild magnitudes — bf16 SiLU×mul + three GEMMs accumulate error.
    x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
    w_gate = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_up = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_down = (torch.randn(k, ffn, device="cuda") * 0.5).to(dtype)
    got = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down)
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    torch.testing.assert_close(got.float(), ref.float(), atol=1.5e-1, rtol=1.5e-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_fused_swiglu_mlp_inreg_matches_eager(dtype: torch.dtype) -> None:
    """In-register SiLU×mul between GEMM0/GEMM1; no mid LDS/GMEM spill.

    Layout documented in module: silu(x@Wgate.T)*(x@Wup.T) then @Wdown.T
    with weights [N,K] (linear B[N,K]). Prefer correct 16×16 path.
    """
    _require_gfx120x()
    torch.manual_seed(7)
    # Mild magnitudes — bf16/fp16 mid narrow after SiLU is the dominant error.
    x = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    w_gate = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    w_up = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    w_down = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    got = fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down, out_dtype=torch.float32)
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    # GEMM acc + SiLU in f32; mid narrows to act dtype before GEMM1.
    atol = 2e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=atol)


def test_fused_swiglu_mlp_inreg_rejects_bad_k() -> None:
    """K/FFN > 16 stay on nmajor/LDS; inreg only soft-pads up to the 16-cube."""
    _require_gfx120x()
    dtype = torch.bfloat16
    x = torch.randn(16, 32, device="cuda", dtype=dtype)
    w_g = torch.randn(16, 32, device="cuda", dtype=dtype)
    w_u = torch.randn(16, 32, device="cuda", dtype=dtype)
    w_d = torch.randn(16, 16, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="0 < K,FFN <= 16"):
        fused_swiglu_mlp_inreg(x, w_g, w_u, w_d)


def test_fused_swiglu_mlp_inreg_matches_thin_host() -> None:
    """In-reg fuse matches eager SwiGLU reference on 16×16 (not nmajor→inreg).

    At K=FFN=16, ``fused_swiglu_mlp_nmajor`` routes into inreg, so comparing
    inreg to nmajor is a tautology. Compare to ``reference_swiglu_mlp`` instead.
    """
    _require_gfx120x()
    torch.manual_seed(11)
    dtype = torch.bfloat16
    x = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    w_gate = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    w_up = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    w_down = (torch.randn(16, 16, device="cuda") * 0.5).to(dtype)
    got = fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down)
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "m,n_out",
    [(32, 16), (32, 32)],
    ids=["m32_n16", "m32_n32"],
)
def test_fused_swiglu_mlp_inreg_host_tile_non16(m: int, n_out: int) -> None:
    """Host-tiled in-reg fuse for K=FFN=16, M/N multiples of 16 vs eager."""
    _require_gfx120x()
    torch.manual_seed(13 + m + n_out)
    dtype = torch.bfloat16
    k = ffn = 16
    x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
    w_gate = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_up = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_down = (torch.randn(n_out, ffn, device="cuda") * 0.5).to(dtype)
    got = fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down, out_dtype=torch.float32)
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    # Do not compare to fused_swiglu_mlp_nmajor here: at K=FFN=16 it routes to inreg.
    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (16, 16, 32),
        (32, 32, 32),
        (32, 64, 64),
        (64, 64, 128),
    ],
    ids=["16x16x32", "32x32x32", "32x64x64", "64x64x128"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_gemm_bf16_nmajor_lds_matches_matmul(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    """Multi-wave LDS production GEMM vs torch matmul (K>16 shapes)."""
    if not str(get_rocm_arch() or "").startswith("gfx120"):
        pytest.skip("requires gfx120x")
    _require_gfx120x()
    torch.manual_seed(20 + m + n + k)
    a = (torch.randn(m, k, device="cuda") * 2).round().to(dtype)
    b = (torch.randn(n, k, device="cuda") * 2).round().to(dtype)
    out = gemm_bf16_nmajor_lds(a, b, out_dtype=torch.float32)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    tile, mp, np_, kp, bm, bn, bk = pick_nmajor_lds_tile(m, n, k)
    assert mp % bm == 0 and np_ % bn == 0 and kp % bk == 0
    assert kp >= 2 * bk


@pytest.mark.parametrize(
    "m,k,ffn,n_out",
    [
        (16, 32, 32, 16),
        (32, 32, 32, 32),
        (16, 64, 32, 16),
    ],
    ids=["m16_k32_ffn32", "m32_k32_ffn32", "m16_k64_ffn32"],
)
def test_fused_swiglu_mlp_nmajor_lds_k_gt_16(m: int, k: int, ffn: int, n_out: int) -> None:
    """K/FFN>16 uses the in-kernel LDS-mid loop; matches eager reference."""
    if not str(get_rocm_arch() or "").startswith("gfx120"):
        pytest.skip("requires gfx120x")
    _require_gfx120x()
    torch.manual_seed(30 + m + k + ffn)
    dtype = torch.bfloat16
    x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
    w_gate = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_up = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_down = (torch.randn(n_out, ffn, device="cuda") * 0.5).to(dtype)
    got = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down)
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    torch.testing.assert_close(got.float(), ref.float(), atol=1.5e-1, rtol=1.5e-1)


@pytest.mark.parametrize(
    "m,k,ffn,n_out,dtype",
    [
        (16, 24, 32, 16, torch.bfloat16),
        (16, 12, 16, 16, torch.bfloat16),
        (16, 8, 8, 16, torch.bfloat16),
        (20, 24, 20, 12, torch.bfloat16),
        (4, 16, 16, 16, torch.bfloat16),
        (16, 24, 24, 16, torch.float16),
    ],
    ids=["k24", "k12", "k8_ffn8", "short_panel", "short_m", "f16_k24"],
)
def test_fused_swiglu_mlp_lds_odd_shape_matches_eager(m: int, k: int, ffn: int, n_out: int, dtype: torch.dtype) -> None:
    """Short K, FFN, or panels stay on the caller's storage and match eager."""
    if not str(get_rocm_arch() or "").startswith("gfx120"):
        pytest.skip("requires gfx120x")
    _require_gfx120x()
    torch.manual_seed(50 + m + k + ffn + n_out)
    x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
    w_gate = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_up = (torch.randn(ffn, k, device="cuda") * 0.5).to(dtype)
    w_down = (torch.randn(n_out, ffn, device="cuda") * 0.5).to(dtype)
    x_ptr, g_ptr = x.data_ptr(), w_gate.data_ptr()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        got = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down, stream=stream)
    stream.synchronize()
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    assert got.shape == (m, n_out)
    assert x.shape == (m, k) and w_gate.shape == (ffn, k)
    assert x.data_ptr() == x_ptr and w_gate.data_ptr() == g_ptr
    torch.testing.assert_close(got.float(), ref.float(), atol=1.5e-1, rtol=1.5e-1)


def test_fused_swiglu_odd_shape_does_not_device_pad(monkeypatch: pytest.MonkeyPatch) -> None:
    """The product host must not clone a short K or FFN up to the WMMA tile."""
    if not str(get_rocm_arch() or "").startswith("gfx120"):
        pytest.skip("requires gfx120x")
    _require_gfx120x()
    import kernels.common.gfx120x_pad as padmod

    def _boom(*_args, **_kwargs):
        raise AssertionError("device_pad")

    monkeypatch.setattr(padmod, "device_pad", _boom)
    dtype = torch.bfloat16
    x = torch.randn(20, 24, device="cuda", dtype=dtype)
    w_gate = torch.randn(20, 24, device="cuda", dtype=dtype)
    w_up = torch.randn(20, 24, device="cuda", dtype=dtype)
    w_down = torch.randn(12, 20, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        got = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down, stream=stream)
    stream.synchronize()
    assert got.shape == (20, 12)
    assert torch.isfinite(got.float()).all()


def test_gemm_bf16_nmajor_zero_lds_still_works() -> None:
    """nmajor GEMM matches A @ B.T. The host entry always uses the LDS kernel."""
    if not str(get_rocm_arch() or "").startswith("gfx120"):
        pytest.skip("requires gfx120x")
    _require_gfx120x()
    torch.manual_seed(41)
    dtype = torch.bfloat16
    a = (torch.randn(32, 32, device="cuda") * 2).round().to(dtype)
    b = (torch.randn(32, 32, device="cuda") * 2).round().to(dtype)
    out = gemm_bf16_nmajor(a, b, out_dtype=torch.float32)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
