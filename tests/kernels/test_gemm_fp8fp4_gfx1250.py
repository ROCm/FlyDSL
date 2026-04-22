#!/usr/bin/env python3
"""Unified MXFP4/MXFP8/A8W4 GEMM correctness tests for gfx1250.

Kernel implementation: kernels/gemm_fp8fp4_gfx1250.py
"""

import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

# workaround for simulator
import flydsl  # noqa: E402,F401 -- preload system comgr before torch/HIP loads LLVM

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
from tests.kernels.utils import fp4_utils


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


SCALE_BLOCK = 32


def preshuffle_e8m0_scale(scale: torch.Tensor, warp_tile: int,
                          scale_k_per_tile: int = 4,
                          WMMA_DIM: int = 16) -> torch.Tensor:
    """Preshuffle E8M0 scale: optional byte swap + interleave for WMMA access."""
    _, K_scale = scale.shape
    assert K_scale % 4 == 0, f"K_scale must be divisible by 4, got {K_scale}"
    SCALES_PER_WMMA = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def random_fp8_data(rows: int, cols: int, *, device="cpu") -> torch.Tensor:
    """Generate random FP8/E4M3 data as uint8. Avoids NaN (0x7F/0xFF)."""
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8, device=device)


def _reference_scaled_gemm(a, b, a_scale, b_scale, M, N, K,
                           convert_fn, convert_fn_b=None):
    """Reference scaled GEMM: D = (A * A_scale) @ (B * B_scale)^T."""
    a_f32 = convert_fn(a.view(torch.uint8))[:M, :K]
    b_f32 = (convert_fn_b or convert_fn)(b.view(torch.uint8))[:N, :K]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


def reference_mxfp4_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K):
    return _reference_scaled_gemm(a_packed, b_packed, a_scale, b_scale,
                                  M, N, K, fp4_utils.mxfp4_to_f32)


def reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K):
    """Standard FP8 reference with SCALE_BLOCK=32."""
    return _reference_scaled_gemm(a, b, a_scale, b_scale,
                                  M, N, K, fp4_utils.fp8_e4m3_to_f32)


def reference_a8w4_gemm(a_fp8, b_fp4, a_scale, b_scale, M, N, K):
    """Standard A8W4 reference: FP8 activation + FP4 weight, SCALE_BLOCK=32."""
    return _reference_scaled_gemm(a_fp8, b_fp4, a_scale, b_scale, M, N, K,
                                  fp4_utils.fp8_e4m3_to_f32,
                                  convert_fn_b=fp4_utils.mxfp4_to_f32)


def _e8m0_exp_range(scale: torch.Tensor) -> tuple[int, int]:
    """Return unbiased exponent range for an E8M0 tensor."""
    scale_u8 = scale.view(torch.uint8).to(torch.int16)
    return int(scale_u8.min().item()) - 127, int(scale_u8.max().item()) - 127


def _a8w4_tolerances(a_scale: torch.Tensor, b_scale: torch.Tensor,
                     K: int, out_dtype: str) -> tuple[float, float, str]:
    """Scale-range-aware tolerance for mixed FP8xFP4 WMMA scale GEMM.

    A8W4 accumulates FP8 activations with FP4 weights and applies independent
    block scales on both operands. The mixed-precision path exhibits a larger
    numeric floor than pure FP8 or pure FP4, and that floor grows with the
    peak product of the two scale ranges.
    """
    a_min_exp, a_max_exp = _e8m0_exp_range(a_scale)
    b_min_exp, b_max_exp = _e8m0_exp_range(b_scale)
    peak_prod_exp = max(0, a_max_exp) + max(0, b_max_exp)
    peak_prod_scale = float(2 ** peak_prod_exp)

    if out_dtype in ("bf16", "f16"):
        rtol = min(5e-2, 1e-2 + 3e-3 * peak_prod_exp)
        atol = max(5e-2, K * (0.6 + 1.5 * peak_prod_exp))
    else:
        rtol = min(2e-2, 1e-3 + 2e-3 * peak_prod_exp)
        atol = max(1e-2, K * (0.6 + 0.55 * peak_prod_exp))

    diag = (
        f"A8W4 scale-aware tolerance: "
        f"A_exp=[{a_min_exp},{a_max_exp}], "
        f"B_exp=[{b_min_exp},{b_max_exp}], "
        f"peak_prod_scale=2^{peak_prod_exp}={peak_prod_scale:.1f}, "
        f"rtol={rtol:.4f}, atol={atol:.4f}"
    )
    return rtol, atol, diag


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _mxscale_pack_factors(data_format: str) -> tuple[int, int]:
    if data_format == "fp4":
        return 2, 2
    if data_format == "a8w4":
        return 1, 2
    if data_format == "fp8":
        return 1, 1
    raise ValueError(f"unsupported data_format={data_format!r}")


def _get_padded_problem_shape(
    data_format: str,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
) -> dict[str, int]:
    """Pad runtime problem to tile-aligned kernel dimensions."""
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")

    pack_a, pack_b = _mxscale_pack_factors(data_format)
    padded_k = _align_up(K, tile_k * split_k)
    return {
        "M": _align_up(M, tile_m),
        "N": _align_up(N, tile_n),
        "K": padded_k,
        "K_scale": padded_k // SCALE_BLOCK,
        "pack_a": pack_a,
        "pack_b": pack_b,
    }


def _pad_2d_tensor(tensor: torch.Tensor, rows: int, cols: int, fill_value: int) -> torch.Tensor:
    if tensor.shape == (rows, cols):
        return tensor
    padded = torch.full((rows, cols), fill_value, dtype=tensor.dtype, device=tensor.device)
    padded[:tensor.shape[0], :tensor.shape[1]] = tensor
    return padded


def _pad_mxscale_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    padded_shape: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad data/scale tensors so the kernel can run full tiles safely."""
    a = _pad_2d_tensor(a, padded_shape["M"], padded_shape["K"] // padded_shape["pack_a"], fill_value=0)
    b = _pad_2d_tensor(b, padded_shape["N"], padded_shape["K"] // padded_shape["pack_b"], fill_value=0)
    a_scale = _pad_2d_tensor(a_scale, padded_shape["M"], padded_shape["K_scale"], fill_value=127)
    b_scale = _pad_2d_tensor(b_scale, padded_shape["N"], padded_shape["K_scale"], fill_value=127)
    return a, b, a_scale, b_scale


def _format_kernel_pad(M: int, N: int, K: int, padded_shape: dict[str, int]) -> str:
    padded_dims = (padded_shape["M"], padded_shape["N"], padded_shape["K"])
    if padded_dims == (M, N, K):
        return ""
    return f", kernel_pad={padded_dims}"


def _run_mxscale_gemm_test(
    data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
    num_buffers, use_tdm_store, out_dtype,
    wave_specialized_tdm=False, use_scale_opsel=False,
    l2_prefetch_distance=0, cluster_m=1, cluster_n=1,
    inst_prefetch=False, waves_per_eu=None,
    expert_sched_mode=True, split_k=1,
    return_launch_fn=False,
):
    """Unified test body for FP4 and FP8."""
    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"

    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA_SCALE requires gfx1250, got {arch}")

    if use_scale_opsel and is_fp4:
        pytest.skip("FP4 32x16 WMMA scaleBType op_sel ignored by AM simulator")

    if K % SCALE_BLOCK != 0:
        pytest.skip(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")

    padded_shape = _get_padded_problem_shape(
        data_format, M, N, K, tile_m, tile_n, tile_k, split_k)
    padded_m = padded_shape["M"]
    padded_n = padded_shape["N"]
    padded_k = padded_shape["K"]
    local_k = padded_k // split_k

    num_k_tiles = local_k // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    # FP8 256x256 + f32 + TDM store exceeds LDS
    if not is_fp4 and tile_m == 256 and tile_n == 256 \
       and out_dtype == "f32" and use_tdm_store:
        pytest.skip("256x256 tile with f32 TDM store exceeds LDS limit")

    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]

    torch.manual_seed(0)

    fmt_name = "A8W4" if is_a8w4 else ("MXFP4" if is_fp4 else "MXFP8")
    mcast_str = f", cluster=({cluster_m},{cluster_n})" \
        if cluster_m > 1 or cluster_n > 1 else ""
    tdm_str = ", tdm_store" if use_tdm_store else ", buffer_store"
    pad_str = _format_kernel_pad(M, N, K, padded_shape)
    print(f"\nRunning {fmt_name} GEMM: M={M}, N={N}, K={K}{pad_str}, "
          f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}"
          f"{mcast_str}{tdm_str}, preshuffle, out={out_dtype}")

    # Generate data
    if is_a8w4:
        a = random_fp8_data(M, K)       # FP8 activation
        b = fp4_utils.random_fp4_packed(N, K)  # FP4 weight
    elif is_fp4:
        a = fp4_utils.random_fp4_packed(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    else:
        a = random_fp8_data(M, K)
        b = random_fp8_data(N, K)
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)
    a_scale_raw = a_scale.clone()
    b_scale_raw = b_scale.clone()

    # Reference
    if is_a8w4:
        ref = reference_a8w4_gemm(a, b, a_scale, b_scale, M, N, K)
    elif is_fp4:
        ref = reference_mxfp4_gemm(a, b, a_scale, b_scale, M, N, K)
    else:
        ref = reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K)

    print(f"Ref stats: min={ref.min():.2f}, max={ref.max():.2f}, "
          f"mean={ref.mean():.2f}, std={ref.std():.2f}")

    a, b, a_scale, b_scale = _pad_mxscale_inputs(
        a, b, a_scale, b_scale, padded_shape)

    # Preshuffle scales
    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)

    # Preshuffle B data
    K_packed = padded_k // padded_shape["pack_b"]
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    # Upload & launch
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_out_dtype, device="cuda")

    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        M=padded_m, N=padded_n, K=padded_k,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m, cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=out_dtype,
        inst_prefetch=inst_prefetch,
        wave_specialized_tdm=wave_specialized_tdm,
        split_k=split_k,
        use_scale_opsel=use_scale_opsel,
        expert_sched_mode=expert_sched_mode,
    )
    launch_fn(
        c_gpu.contiguous().view(-1),
        a_gpu.contiguous().view(-1),
        b_gpu.contiguous().view(-1),
        as_gpu.contiguous().view(-1),
        bs_gpu.contiguous().view(-1),
        padded_m, padded_n, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    c_out = c_gpu[:M, :N].cpu()

    print(f"Out stats: min={c_out.float().min():.2f}, max={c_out.float().max():.2f}, "
          f"mean={c_out.float().mean():.2f}, std={c_out.float().std():.2f}")

    if c_out.float().abs().max() < 1e-10:
        print("WARNING: kernel output is all zeros!")

    if out_dtype in ("bf16", "f16"):
        ref_cmp = ref.to(torch_out_dtype)
        c_out_f = c_out.float()
        ref_f = ref_cmp.float()
    else:
        c_out_f = c_out.float()
        ref_f = ref.float()

    diff = (c_out_f - ref_f).abs()
    print(f"Abs diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

    cos_sim = torch.nn.functional.cosine_similarity(
        c_out_f.flatten().unsqueeze(0), ref_f.flatten().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    # Tolerances: FP4 is exact; FP8/A8W4 have FP accumulation error
    if is_fp4:
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=1e-2)
        else:
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-5, atol=1e-8)
    elif is_a8w4:
        rtol, atol, tol_diag = _a8w4_tolerances(
            a_scale_raw, b_scale_raw, K, out_dtype)
        print(tol_diag)
        torch.testing.assert_close(c_out_f, ref_f, rtol=rtol, atol=atol)
    else:
        # FP8: standard SCALE_BLOCK=32 reference
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-2, atol=5e-2)
        else:
            atol = max(1e-2, K * 0.6)
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=atol)
    print("PASSED")
    if return_launch_fn:
        return launch_fn


def _get_latest_artifact(launch_fn):
    """Return the most recent CompiledArtifact produced by a JIT launch."""
    last_compiled = getattr(launch_fn, "_last_compiled", None)
    if last_compiled is not None:
        return last_compiled[1]

    mem_cache = getattr(launch_fn, "_mem_cache", None)
    if mem_cache:
        newest_key = next(reversed(mem_cache))
        return mem_cache[newest_key]

    raise AssertionError("expected launch_fn to have a compiled artifact")


def _extract_i64_metadata(compiled_ir: str, key: str) -> int:
    match = re.search(rf"{key}\s*=\s*(\d+)\s*:\s*i64", compiled_ir)
    assert match is not None, f"{key} not found in compiled IR:\n{compiled_ir}"
    return int(match.group(1))


# ── pytest parametrized tests ──

@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 512, 7168, 128, 128, 256, 2, 2),
        (128, 7168, 256, 128, 256, 128, 2, 2),
        (128, 4096, 7168, 128, 256, 256, 2, 2),
        (128, 7168, 2048, 128, 256, 256, 2, 2),
        (1024, 1024, 1024, 256, 256, 256, 2, 2),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("wave_specialized_tdm", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp4_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                     num_buffers, use_tdm_store, out_dtype,
                     wave_specialized_tdm, use_scale_opsel):
    _run_mxscale_gemm_test(
        "fp4", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        wave_specialized_tdm=wave_specialized_tdm,
        use_scale_opsel=use_scale_opsel)


@pytest.mark.parametrize("out_dtype", ["bf16", "f16"])
def test_mxfp4_metadata_and_spill_regression(out_dtype):
    launch_fn = _run_mxscale_gemm_test(
        "fp4",
        1024, 1024, 1024,
        256, 256, 256,
        2, 2,
        num_buffers=4,
        use_tdm_store=True,
        out_dtype=out_dtype,
        return_launch_fn=True,
    )
    artifact = _get_latest_artifact(launch_fn)

    assert "known_block_size = array<i32: 128, 1, 1>" in artifact.source_ir, (
        f"expected known_block_size metadata in source IR:\n{artifact.source_ir}"
    )

    compiled_ir = artifact.ir
    assert _extract_i64_metadata(compiled_ir, "max_flat_workgroup_size") == 128
    assert _extract_i64_metadata(compiled_ir, "vgpr_spill_count") == 0


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 256, 256, 128, 256, 128, 2, 4),
        (256, 256, 256, 256, 256, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp8_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                     num_buffers, use_tdm_store, out_dtype, use_scale_opsel):
    _run_mxscale_gemm_test(
        "fp8", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        l2_prefetch_distance=2,
        use_scale_opsel=use_scale_opsel)


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 5760, 2880, 128, 256, 256, 2, 2),
        (128, 2880, 2880, 128, 256, 256, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_a8w4_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                    num_buffers, use_tdm_store, out_dtype, use_scale_opsel):
    _run_mxscale_gemm_test(
        "a8w4", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        l2_prefetch_distance=2,
        use_scale_opsel=use_scale_opsel)


@pytest.mark.parametrize(
    "M, N, K, use_tdm_store",
    [
        (13, 2880, 2880, True),
        (33, 5760, 2880, False),
    ],
)
def test_a8w4_gemm_irregular_m_tile16(M, N, K, use_tdm_store):
    # Small-M path: pad M to 16 and dedicate one wave to the M dimension.
    _run_mxscale_gemm_test(
        "a8w4",
        M, N, K,
        16, 256, 256,
        1, 4,
        num_buffers=2,
        use_tdm_store=use_tdm_store,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        use_scale_opsel=False,
    )


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n",
    [
        (256, 256, 256, 128, 128, 128, 2, 2, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4, 2, 2),
        (128, 256, 256, 128, 128, 128, 2, 2, 1, 2),
        (256, 128, 256, 128, 128, 128, 2, 2, 2, 1),
        (512, 512, 256, 128, 128, 128, 2, 2, 4, 4),
        (1024, 1024, 1024, 128, 256, 128, 2, 4, 4, 4),
        (512, 512, 512, 128, 128, 128, 2, 2, 2, 4),
        (512, 512, 512, 128, 128, 128, 2, 2, 4, 2),
    ],
)
@pytest.mark.parametrize("num_buffers", [2])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp4_gemm_mcast(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                            cluster_m, cluster_n, num_buffers, use_tdm_store,
                            out_dtype):
    _run_mxscale_gemm_test(
        "fp4", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        l2_prefetch_distance=2,
        cluster_m=cluster_m, cluster_n=cluster_n)


def _bench_kernel_us(run_fn, warmup=10, iters=50, flush_l2=True, prep_fn=None):
    """Per-iteration CUDA events timer with L2 flush, IQR outlier removal, median.

    Follows the golden practices from benchmarks/gemm_gfx1250_benchmark.py:
    - L2 flush between iterations via zero-fill of 2× L2 buffer
    - Per-iteration event pairs (no inter-iteration sync on kernel)
    - IQR outlier removal (if n >= 8), median latency
    """
    flush_buf = None
    if flush_l2:
        l2_bytes = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "L2_cache_size", 4 * 1024 * 1024)
        alloc_bytes = max(l2_bytes * 2, 8 * 1024 * 1024)
        flush_buf = torch.empty(alloc_bytes, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        run_fn()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        start_ev[i].record()
        run_fn()
        end_ev[i].record()

    torch.cuda.synchronize()

    latencies = sorted(
        start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters))

    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in latencies if lo <= x <= hi]
        if filtered:
            latencies = filtered

    del flush_buf
    return latencies[len(latencies) // 2]


def _run_benchmark(args):
    """Benchmark mode: compile once, time kernel execution with proper methodology."""
    import time

    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"

    data_format = args.data_format
    M, N, K = args.M, args.N, args.K
    tile_m, tile_n, tile_k = args.tile_m, args.tile_n, args.tile_k
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")

    padded_shape = _get_padded_problem_shape(
        data_format, M, N, K, tile_m, tile_n, tile_k, args.split_k)
    padded_m = padded_shape["M"]
    padded_n = padded_shape["N"]
    padded_k = padded_shape["K"]
    PACK_A = padded_shape["pack_a"]
    PACK_B = padded_shape["pack_b"]

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[args.out_dtype]
    elem_bytes_d = 2 if args.out_dtype in ("bf16", "f16") else 4
    fmt_name = "A8W4" if is_a8w4 else ("MXFP4" if is_fp4 else "MXFP8")

    print("=" * 72)
    print(f"  {fmt_name} GEMM Benchmark on gfx1250")
    print(f"  PyTorch {torch.__version__}, Device: {torch.cuda.get_device_name(0)}")
    needs_pad = (padded_m, padded_n, padded_k) != (M, N, K)
    print(f"  Shape: M={M}, N={N}, K={K}")
    if needs_pad:
        print(f"  Kernel pad: M={padded_m}, N={padded_n}, K={padded_k}")
    print(f"  Tile: ({tile_m}, {tile_n}, {tile_k}), warps=({args.m_warp}x{args.n_warp})")
    print(f"  Buffers={args.num_buffers}, out={args.out_dtype}, "
          f"opsel={args.use_scale_opsel}, inst_prefetch={args.inst_prefetch}")
    if args.split_k > 1:
        print(f"  Split-K={args.split_k} (atomic accumulate, buffer-store epilogue)")
    print(f"  Warmup={args.warmup}, Iters={args.iters}, "
          f"L2 flush={'ON' if not args.no_flush_l2 else 'OFF'}")
    print("  Zero fill: ON (outside timing)")
    print("=" * 72)

    torch.manual_seed(0)

    if is_a8w4:
        a = random_fp8_data(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    elif is_fp4:
        a = fp4_utils.random_fp4_packed(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    else:
        a = random_fp8_data(M, K)
        b = random_fp8_data(N, K)

    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

    a, b, a_scale, b_scale = _pad_mxscale_inputs(
        a, b, a_scale, b_scale, padded_shape)

    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // args.m_warp
    warp_tile_n = tile_n // args.n_warp
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)

    K_packed = padded_k // PACK_B
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_out_dtype, device="cuda")

    print(f"\n[1/3] Compiling kernel...")
    t0 = time.perf_counter()
    use_tdm_store = not args.no_tdm_store
    if args.split_k > 1 and use_tdm_store:
        print("      Note: split-K forces buffer-store atomic epilogue; disabling TDM store.")
        use_tdm_store = False
    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        M=padded_m, N=padded_n, K=padded_k,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=args.m_warp, n_warp=args.n_warp,
        num_buffers=args.num_buffers,
        waves_per_eu=args.waves_per_eu,
        l2_prefetch_distance=args.l2_prefetch_distance,
        cluster_m=args.cluster_m, cluster_n=args.cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=args.out_dtype,
        inst_prefetch=args.inst_prefetch,
        wave_specialized_tdm=args.wave_spec_tdm,
        split_k=args.split_k,
        use_scale_opsel=args.use_scale_opsel,
        expert_sched_mode=args.expert_sched_mode,
        atomic_barrier_enable=args.atomic_barrier_enable,
    )

    stream = torch.cuda.current_stream()
    c_flat = c_gpu.view(-1)
    a_flat = a_gpu.view(-1)
    b_flat = b_gpu.view(-1)
    as_flat = as_gpu.view(-1)
    bs_flat = bs_gpu.view(-1)

    def prep_kernel():
        c_gpu.zero_()

    def run_kernel():
        launch_fn(
            c_flat, a_flat, b_flat,
            as_flat, bs_flat,
            padded_m, padded_n, stream,
        )

    prep_kernel()
    run_kernel()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"      Compile + first launch: {compile_ms:.0f} ms")

    print(f"[2/3] Warming up ({args.warmup} iters) + benchmarking ({args.iters} iters)...")
    us = _bench_kernel_us(run_kernel, warmup=args.warmup, iters=args.iters,
                          flush_l2=not args.no_flush_l2, prep_fn=prep_kernel)

    logical_flops = 2.0 * M * N * K
    kernel_flops = 2.0 * padded_m * padded_n * padded_k
    time_s = us / 1e6
    logical_tflops = logical_flops / time_s / 1e12 if time_s > 0 else 0.0
    kernel_tflops = kernel_flops / time_s / 1e12 if time_s > 0 else 0.0

    bytes_a = padded_m * padded_k // PACK_A
    bytes_b = padded_n * padded_k // PACK_B
    bytes_scale = (padded_m + padded_n) * padded_shape["K_scale"]
    bytes_d = padded_m * padded_n * elem_bytes_d
    read_bytes = bytes_a + bytes_b + bytes_scale
    write_bytes = bytes_d
    bytes_moved = read_bytes + write_bytes
    bw_gbs = bytes_moved / 1e9 / time_s if time_s > 0 else 0.0
    read_bw_gbs = read_bytes / 1e9 / time_s if time_s > 0 else 0.0
    write_bw_gbs = write_bytes / 1e9 / time_s if time_s > 0 else 0.0

    WMMA_K = 128
    WMMA_N_EFF = 32 if is_fp4 else 16
    wmma_m_rep = warp_tile_m // 16
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    k_wmma_steps = tile_k // WMMA_K
    wmma_per_tile = wmma_m_rep * wmma_n_rep * k_wmma_steps
    m_tiles = padded_m // tile_m
    n_tiles = padded_n // tile_n
    k_tiles = padded_k // tile_k
    k_tiles_local = (padded_k // args.split_k) // tile_k
    total_wmma = m_tiles * n_tiles * k_tiles * wmma_per_tile
    # Sequential WMMAs per workgroup (all k_tiles execute sequentially)
    seq_wmma = k_tiles_local * wmma_per_tile
    us_per_wmma = us / seq_wmma if seq_wmma > 0 else 0

    print(f"\n[3/3] Results:")
    print(f"      Kernel time:  {us:.1f} us ({us / 1e3:.4f} ms)")
    if not needs_pad:
        print(f"      TFLOPS:       {kernel_tflops:.4f}")
    else:
        print(f"      TFLOPS:       {logical_tflops:.4f} (logical), {kernel_tflops:.4f} (kernel)")
    print(f"      Bandwidth:    {bw_gbs:.1f} GB/s  "
          f"(read: {read_bw_gbs:.1f} + write: {write_bw_gbs:.1f})")
    print(f"      Bytes moved:  {bytes_moved / 1e6:.1f} MB  "
          f"(A={bytes_a / 1e6:.1f} B={bytes_b / 1e6:.1f} "
          f"scale={bytes_scale / 1e6:.1f} D={bytes_d / 1e6:.1f})")
    print(f"      ---")
    print(f"      WMMA/tile:    {wmma_per_tile} "
          f"({wmma_m_rep}m × {wmma_n_rep}n × {k_wmma_steps}k)")
    if args.split_k > 1:
        print(f"      Total tiles:  {m_tiles}×{n_tiles} spatial × "
              f"{args.split_k} split-K × {k_tiles_local} local K-iters")
    else:
        print(f"      Total tiles:  {m_tiles}×{n_tiles} spatial × {k_tiles} K-iters")
    print(f"      Seq WMMA/WG:  {seq_wmma}")
    print(f"      us/WMMA:      {us_per_wmma:.1f}")
    if us_per_wmma > 1000:
        print(f"      WARNING: {us_per_wmma/1000:.1f} ms/WMMA indicates "
              f"WMMA_SCALE trap-handler emulation")
    print("=" * 72)

    reported_tflops = kernel_tflops if not needs_pad else logical_tflops
    return us, reported_tflops, bw_gbs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-format", type=str, default="fp4",
                        choices=["fp4", "fp8", "a8w4"])
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=256)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--l2-prefetch-distance", type=int, default=2)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--out-dtype", type=str, default="bf16",
                        choices=["f32", "bf16", "f16"])
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--wave-spec-tdm", action="store_true", default=False)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--use-scale-opsel", action="store_true", default=False)
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode",
                        action="store_false", default=True)
    parser.add_argument("--atomic-barrier-enable", action="store_true", default=False,
                        help="Enable TDM atomic_barrier_enable (hardware auto-barrier)")

    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Run benchmark mode (timing only, no correctness check)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-flush-l2", action="store_true", default=False)
    args = parser.parse_args()

    if args.benchmark:
        _run_benchmark(args)
    else:
        use_tdm_store = not args.no_tdm_store and args.split_k == 1
        _run_mxscale_gemm_test(
            args.data_format,
            args.M, args.N, args.K,
            args.tile_m, args.tile_n, args.tile_k,
            args.m_warp, args.n_warp,
            num_buffers=args.num_buffers,
            use_tdm_store=use_tdm_store,
            out_dtype=args.out_dtype,
            wave_specialized_tdm=args.wave_spec_tdm,
            split_k=args.split_k,
            use_scale_opsel=args.use_scale_opsel,
            l2_prefetch_distance=args.l2_prefetch_distance,
            cluster_m=args.cluster_m,
            cluster_n=args.cluster_n,
            inst_prefetch=args.inst_prefetch,
            waves_per_eu=args.waves_per_eu,
            expert_sched_mode=args.expert_sched_mode,
        )
