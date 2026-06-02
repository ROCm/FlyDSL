#!/usr/bin/env python3
"""Unified MXFP4/MXFP8/A8W4 GEMM correctness tests for gfx1250.

Kernel implementation: kernels/gemm_fp8fp4_gfx1250.py
"""

import math
import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc  # noqa: E402,I001

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


SCALE_BLOCK = 32


def preshuffle_e8m0_scale_coalesced(scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Lane-major scale layout for direct buffer_load->VGPR.

    Per (M_block=128, K_tile): [group(2), lane16(16), 4 i32], so a buffer_load_b128's
    16 lanes read 256 contiguous bytes. M = mb*128 + (group*4 + j)*16 + lane16.
    """
    M, Ks = scale.shape
    assert M % block == 0 and Ks % 4 == 0, f"M={M} Ks={Ks} block={block}"
    assert block == 128, "coalesced scale layout assumes warp_tile=128 (8 subtiles)"
    Kt = Ks // 4
    g = scale.view(M // block, 2, 4, 16, Kt, 4)  # [mb, group, j, lane16, kt, spw]
    g = g.permute(0, 4, 1, 3, 2, 5).contiguous()  # [mb, kt, group, lane16, j, spw]
    return g.view(M, Ks)


def preshuffle_e8m0_scale(
    scale: torch.Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
    WMMA_DIM: int = 16,
    coalesced: bool = False,
) -> torch.Tensor:
    """Preshuffle E8M0 scale: optional byte swap + interleave for WMMA access.

    ``coalesced=True`` produces the lane-major layout the scale_load_path
    "vgpr"/"vgpr_ab_split" buffer_load->VGPR path expects.
    """
    if coalesced:
        return preshuffle_e8m0_scale_coalesced(scale, block=warp_tile)
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


def _fp8_e4m3fn_byte(value: float) -> int:
    """Return torch's FP8 E4M3FN byte encoding for a finite scalar."""
    t = torch.tensor([float(value)], dtype=torch.float8_e4m3fn)
    byte = int(t.view(torch.uint8).item())
    if (byte & 0x7F) == 0x7F:
        raise SystemExit(f"--fill-mode constant {value:g} is outside the finite FP8 E4M3FN range")
    return byte


def _parse_fill_mode(arg: str):
    """Parse --fill-mode as ('random',) or ('const', value)."""
    if arg == "random":
        return ("random",)
    if arg == "zero":
        return ("const", 0.0)
    try:
        value = float(arg)
    except ValueError as e:
        raise SystemExit(f"--fill-mode must be 'random' or a finite float constant, got {arg!r}") from e
    if not math.isfinite(value):
        raise SystemExit(f"--fill-mode constant must be finite, got {arg!r}")
    return ("const", value)


def _fp4_e2m1_packed_fill(rows: int, cols: int, value: float) -> torch.Tensor:
    dense = torch.full((rows, cols), float(value), dtype=torch.float32)
    return fp4_utils.f32_to_mxfp4(dense).view(torch.uint8)


def _random_mxscale_inputs(M: int, N: int, K: int, data_format: str):
    if data_format == "a8w4":
        a = random_fp8_data(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    elif data_format == "fp4":
        a = fp4_utils.random_fp4_packed(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    elif data_format == "fp8":
        a = random_fp8_data(M, K)
        b = random_fp8_data(N, K)
    else:
        raise ValueError(f"unsupported data_format={data_format!r}")
    return a, b, fp4_utils.random_e8m0(M, K // SCALE_BLOCK), fp4_utils.random_e8m0(N, K // SCALE_BLOCK)


def _const_fill_inputs(M, N, K, data_format: str, value: float):
    """Build constant A/B tensors with neutral E8M0 scales for CLI runs."""
    if data_format == "fp4":
        a = _fp4_e2m1_packed_fill(M, K, value)
        b = _fp4_e2m1_packed_fill(N, K, value)
    elif data_format == "a8w4":
        fp8_byte = _fp8_e4m3fn_byte(value)
        a = torch.full((M, K), fp8_byte, dtype=torch.uint8)
        b = _fp4_e2m1_packed_fill(N, K, value)
    elif data_format == "fp8":
        fp8_byte = _fp8_e4m3fn_byte(value)
        a = torch.full((M, K), fp8_byte, dtype=torch.uint8)
        b = torch.full((N, K), fp8_byte, dtype=torch.uint8)
    else:
        raise ValueError(f"unsupported data_format={data_format!r}")
    a_scale = torch.full((M, K // SCALE_BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((N, K // SCALE_BLOCK), 127, dtype=torch.uint8)
    return a, b, a_scale, b_scale


def _fill_mode_inputs(M: int, N: int, K: int, data_format: str, fill_mode: str):
    fill_spec = _parse_fill_mode(fill_mode)
    if fill_spec[0] == "const":
        a, b, a_scale, b_scale = _const_fill_inputs(M, N, K, data_format, fill_spec[1])
    else:
        a, b, a_scale, b_scale = _random_mxscale_inputs(M, N, K, data_format)
    return a, b, a_scale, b_scale, fill_spec


def _fill_mode_label(fill_spec, data_format: str) -> str:
    if fill_spec[0] == "random":
        return "random (seed=0)"
    label = f"const={fill_spec[1]:g}, E8M0 byte=127"
    if data_format in ("fp8", "a8w4"):
        label += f", FP8 byte=0x{_fp8_e4m3fn_byte(fill_spec[1]):02x}"
    return label


def _has_nonzero_quantized_values(tensor: torch.Tensor, data_format: str) -> bool:
    convert = fp4_utils.mxfp4_to_f32 if data_format == "fp4" else fp4_utils.fp8_e4m3_to_f32
    return bool(convert(tensor.view(torch.uint8)).abs().max().item() > 0)


def _expect_nonzero_graph_output(a: torch.Tensor, b: torch.Tensor, data_format: str, fill_spec) -> bool:
    if fill_spec[0] == "random":
        return True
    a_format = "fp4" if data_format == "fp4" else "fp8"
    b_format = "fp8" if data_format == "fp8" else "fp4"
    return _has_nonzero_quantized_values(a, a_format) and _has_nonzero_quantized_values(b, b_format)


def _reference_scaled_gemm(a, b, a_scale, b_scale, M, N, K, convert_fn, convert_fn_b=None):
    """Reference scaled GEMM: D = (A * A_scale) @ (B * B_scale)^T."""
    a_f32 = convert_fn(a.view(torch.uint8))[:M, :K]
    b_f32 = (convert_fn_b or convert_fn)(b.view(torch.uint8))[:N, :K]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


def reference_mxfp4_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K):
    return _reference_scaled_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K, fp4_utils.mxfp4_to_f32)


def reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K):
    """Standard FP8 reference with SCALE_BLOCK=32."""
    return _reference_scaled_gemm(a, b, a_scale, b_scale, M, N, K, fp4_utils.fp8_e4m3_to_f32)


def reference_a8w4_gemm(a_fp8, b_fp4, a_scale, b_scale, M, N, K):
    """Standard A8W4 reference: FP8 activation + FP4 weight, SCALE_BLOCK=32."""
    return _reference_scaled_gemm(
        a_fp8, b_fp4, a_scale, b_scale, M, N, K, fp4_utils.fp8_e4m3_to_f32, convert_fn_b=fp4_utils.mxfp4_to_f32
    )


def _e8m0_exp_range(scale: torch.Tensor) -> tuple[int, int]:
    """Return unbiased exponent range for an E8M0 tensor."""
    scale_u8 = scale.view(torch.uint8).to(torch.int16)
    return int(scale_u8.min().item()) - 127, int(scale_u8.max().item()) - 127


def _a8w4_tolerances(a_scale: torch.Tensor, b_scale: torch.Tensor, K: int, out_dtype: str) -> tuple[float, float, str]:
    """Scale-range-aware tolerance for mixed FP8xFP4 WMMA scale GEMM.

    A8W4 accumulates FP8 activations with FP4 weights and applies independent
    block scales on both operands. The mixed-precision path exhibits a larger
    numeric floor than pure FP8 or pure FP4, and that floor grows with the
    peak product of the two scale ranges.
    """
    a_min_exp, a_max_exp = _e8m0_exp_range(a_scale)
    b_min_exp, b_max_exp = _e8m0_exp_range(b_scale)
    peak_prod_exp = max(0, a_max_exp) + max(0, b_max_exp)
    peak_prod_scale = float(2**peak_prod_exp)

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
    padded[: tensor.shape[0], : tensor.shape[1]] = tensor
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
    data_format,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    use_tdm_store,
    out_dtype,
    wave_specialized_tdm=False,
    use_scale_opsel=False,
    l2_prefetch_distance=0,
    cluster_m=1,
    cluster_n=1,
    inst_prefetch=False,
    waves_per_eu=None,
    expert_sched_mode=True,
    split_k=1,
    b_streaming=False,
    scale_load_path="tdm",
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

    padded_shape = _get_padded_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, split_k)
    padded_m = padded_shape["M"]
    padded_n = padded_shape["N"]
    padded_k = padded_shape["K"]
    local_k = padded_k // split_k

    num_k_tiles = local_k // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    # FP8 256x256 + f32 + TDM store exceeds LDS
    if not is_fp4 and tile_m == 256 and tile_n == 256 and out_dtype == "f32" and use_tdm_store:
        pytest.skip("256x256 tile with f32 TDM store exceeds LDS limit")

    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]

    torch.manual_seed(0)

    fmt_name = "A8W4" if is_a8w4 else ("MXFP4" if is_fp4 else "MXFP8")
    mcast_str = f", cluster=({cluster_m},{cluster_n})" if cluster_m > 1 or cluster_n > 1 else ""
    tdm_str = ", tdm_store" if use_tdm_store else ", buffer_store"
    scale_load_str = "" if scale_load_path == "tdm" else f", scale_load={scale_load_path}"
    pad_str = _format_kernel_pad(M, N, K, padded_shape)
    print(
        f"\nRunning {fmt_name} GEMM: M={M}, N={N}, K={K}{pad_str}, "
        f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}"
        f"{mcast_str}{tdm_str}{scale_load_str}, preshuffle, out={out_dtype}"
    )

    # Generate data
    if is_a8w4:
        a = random_fp8_data(M, K)  # FP8 activation
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

    print(f"Ref stats: min={ref.min():.2f}, max={ref.max():.2f}, " f"mean={ref.mean():.2f}, std={ref.std():.2f}")

    a, b, a_scale, b_scale = _pad_mxscale_inputs(a, b, a_scale, b_scale, padded_shape)

    # Preshuffle scales
    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    _coalesced_scale = scale_load_path in ("vgpr", "vgpr_ab_split")
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt, coalesced=_coalesced_scale)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt, coalesced=_coalesced_scale)

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
        M=padded_m,
        N=padded_n,
        K=padded_k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=out_dtype,
        inst_prefetch=inst_prefetch,
        wave_specialized_tdm=wave_specialized_tdm,
        split_k=split_k,
        use_scale_opsel=use_scale_opsel,
        expert_sched_mode=expert_sched_mode,
        b_streaming=b_streaming,
        scale_load_path=scale_load_path,
    )

    # Keep 2D — dynamic_layout=True packs shape as i32; flattening overflows for M*K >= 2^31.
    c_flat = c_gpu.contiguous()
    a_flat = a_gpu.contiguous()
    b_flat = b_gpu.contiguous()
    as_flat = as_gpu.contiguous()
    bs_flat = bs_gpu.contiguous()

    flyc.compile(
        launch_fn,
        c_flat,
        a_flat,
        b_flat,
        as_flat,
        bs_flat,
        padded_m,
        padded_n,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    c_out = c_gpu[:M, :N].cpu()

    print(
        f"Out stats: min={c_out.float().min():.2f}, max={c_out.float().max():.2f}, "
        f"mean={c_out.float().mean():.2f}, std={c_out.float().std():.2f}"
    )

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

    # Compute cosine in float64: for large M/N/K with large E8M0 scales the values
    # (and their squares) overflow float32's accurate-summation range, so an fp32
    # cosine reduction saturates and can print values outside [-1, 1]. fp64 keeps
    # the diagnostic meaningful. (Pass/fail is gated by assert_close below, not this.)
    cos_sim = torch.nn.functional.cosine_similarity(
        c_out_f.flatten().unsqueeze(0).double(), ref_f.flatten().unsqueeze(0).double()
    ).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    # Tolerances: FP4 is exact; FP8/A8W4 have FP accumulation error
    if is_fp4:
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=1e-2)
        else:
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-5, atol=1e-8)
    elif is_a8w4:
        rtol, atol, tol_diag = _a8w4_tolerances(a_scale_raw, b_scale_raw, K, out_dtype)
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
def test_mxfp4_gemm(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    use_tdm_store,
    out_dtype,
    wave_specialized_tdm,
    use_scale_opsel,
):
    _run_mxscale_gemm_test(
        "fp4",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        use_tdm_store,
        out_dtype,
        wave_specialized_tdm=wave_specialized_tdm,
        use_scale_opsel=use_scale_opsel,
    )


@pytest.mark.parametrize("out_dtype", ["bf16", "f16"])
def test_mxfp4_metadata_and_spill_regression(out_dtype):
    launch_fn = _run_mxscale_gemm_test(
        "fp4",
        1024,
        1024,
        1024,
        256,
        256,
        256,
        2,
        2,
        num_buffers=4,
        use_tdm_store=True,
        out_dtype=out_dtype,
        return_launch_fn=True,
    )
    artifact = _get_latest_artifact(launch_fn)

    assert (
        "known_block_size = array<i32: 128, 1, 1>" in artifact.source_ir
    ), f"expected known_block_size metadata in source IR:\n{artifact.source_ir}"

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
@pytest.mark.parametrize("scale_load_path", ["tdm"])
def test_mxfp8_gemm(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    use_tdm_store,
    out_dtype,
    use_scale_opsel,
    scale_load_path,
):
    _run_mxscale_gemm_test(
        "fp8",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        use_tdm_store,
        out_dtype,
        l2_prefetch_distance=2,
        use_scale_opsel=use_scale_opsel,
        scale_load_path=scale_load_path,
    )


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
def test_a8w4_gemm(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, use_tdm_store, out_dtype, use_scale_opsel
):
    _run_mxscale_gemm_test(
        "a8w4",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        use_tdm_store,
        out_dtype,
        l2_prefetch_distance=2,
        use_scale_opsel=use_scale_opsel,
    )


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
        M,
        N,
        K,
        16,
        256,
        256,
        1,
        4,
        num_buffers=2,
        use_tdm_store=use_tdm_store,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        use_scale_opsel=False,
    )


@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        ("fp4", 128, 512, 7168, 128, 128, 256, 2, 2),
        ("fp8", 128, 256, 256, 128, 256, 128, 2, 4),
        ("a8w4", 128, 256, 256, 128, 256, 128, 2, 4),
    ],
)
def test_b_streaming_correctness(data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp):
    _run_mxscale_gemm_test(
        data_format,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers=2,
        use_tdm_store=True,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        b_streaming=True,
    )


@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        ("fp4", 128, 256, 512, 128, 128, 256, 2, 2),
        ("fp8", 128, 256, 256, 128, 256, 128, 2, 2),
        ("a8w4", 128, 256, 256, 128, 256, 128, 2, 2),
    ],
)
def test_b_streaming_with_wave_spec_tdm(data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp):
    _run_mxscale_gemm_test(
        data_format,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers=2,
        use_tdm_store=True,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        b_streaming=True,
        wave_specialized_tdm=True,
    )


@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [False, True])
def test_mxfp8_wave_spec_scale_load_paths(num_buffers, use_tdm_store, use_scale_opsel):
    _run_mxscale_gemm_test(
        "fp8",
        128,
        256,
        384,
        128,
        256,
        128,
        2,
        2,
        num_buffers=num_buffers,
        use_tdm_store=use_tdm_store,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        wave_specialized_tdm=True,
        use_scale_opsel=use_scale_opsel,
        scale_load_path="tdm",
    )


@pytest.mark.parametrize("scale_load_path", ["vgpr", "vgpr_ab_split"])
@pytest.mark.parametrize("cluster_m, cluster_n", [(1, 1), (2, 2)])
def test_mxfp8_vgpr_scale_load(scale_load_path, cluster_m, cluster_n):
    _run_mxscale_gemm_test(
        "fp8",
        256 * cluster_m,
        256 * cluster_n,
        512,
        256,
        256,
        128,
        2,
        2,
        num_buffers=4,
        use_tdm_store=True,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        wave_specialized_tdm=True,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        scale_load_path=scale_load_path,
    )


@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n",
    [
        ("fp4", 256, 512, 256, 128, 256, 128, 2, 2, 2, 2),
        ("fp8", 256, 512, 256, 128, 256, 128, 2, 2, 2, 2),
    ],
)
def test_b_streaming_with_cluster_mcast(
    data_format,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    cluster_m,
    cluster_n,
):
    if str(get_rocm_arch()) != "gfx1250":
        pytest.skip("requires gfx1250")
    if "FFMLITE_TOPOLOGY" in os.environ or "AM_TOPOLOGY" in os.environ:
        pytest.skip("cluster multicast not supported on simulator")
    _run_mxscale_gemm_test(
        data_format,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers=2,
        use_tdm_store=True,
        out_dtype="bf16",
        l2_prefetch_distance=2,
        b_streaming=True,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
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
def test_mxfp4_gemm_mcast(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n, num_buffers, use_tdm_store, out_dtype
):
    _run_mxscale_gemm_test(
        "fp4",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        use_tdm_store,
        out_dtype,
        l2_prefetch_distance=2,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )


@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        ("fp8", 128, 256, 256, 128, 256, 128, 2, 2),
        ("fp4", 128, 256, 256, 128, 256, 128, 2, 2),
    ],
    ids=["fp8-128x256x256", "fp4-128x256x256"],
)
def test_mxscale_gemm_cudagraph(data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp):
    """Verify that the gfx1250 MX-scale GEMM kernel works inside a hipGraph.

    Captures one launch, replays once, and checks the replay output is
    bit-equivalent to an eager launch with the same inputs. Catches kernel
    regressions that would break graph capture / replay (accidental host
    syncs, allocator allocations on the kernel path, stream-event API misuse).
    """
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA_SCALE requires gfx1250, got {arch}")
    if "FFMLITE_TOPOLOGY" in os.environ or "AM_TOPOLOGY" in os.environ:
        pytest.skip("hipGraph capture/replay not supported on simulator")

    is_fp4 = data_format == "fp4"

    # Build inputs (mirrors _run_mxscale_gemm_test, but no padding needed
    # because we pick a clean shape).
    torch.manual_seed(0)
    if is_fp4:
        a = fp4_utils.random_fp4_packed(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    else:
        a = random_fp8_data(M, K)
        b = random_fp8_data(N, K)
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    a_scale_ps = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt)
    b_scale_ps = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)
    pack_b = 2 if is_fp4 else 1
    b_ps = fp4_utils.preshuffle_b_16x16(b, N, K // pack_b)

    a_gpu = a.cuda()
    b_gpu = b_ps.cuda()
    as_gpu = a_scale_ps.cuda()
    bs_gpu = b_scale_ps.cuda()
    c_gpu = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=2,
        use_tdm_store=True,
        out_dtype="bf16",
        wave_specialized_tdm=False,
        split_k=1,
    )

    c_flat = c_gpu.contiguous()
    a_flat = a_gpu.contiguous()
    b_flat = b_gpu.contiguous()
    as_flat = as_gpu.contiguous()
    bs_flat = bs_gpu.contiguous()
    compiled_exe = flyc.compile(
        launch_fn,
        c_flat,
        a_flat,
        b_flat,
        as_flat,
        bs_flat,
        M,
        N,
        torch.cuda.current_stream(),
    )

    # Resolve stream lazily inside the launch closure so graph capture sees
    # the active capture stream rather than a stream bound before capture.
    def launch():
        compiled_exe(c_flat, a_flat, b_flat, as_flat, bs_flat, M, N, torch.cuda.current_stream())

    # ── Eager run (reference) ──
    c_gpu.zero_()
    launch()
    torch.cuda.synchronize()
    eager_result = c_gpu.clone()
    assert eager_result.abs().max().item() > 0, "Eager run produced all zeros — kernel did not execute properly."

    # ── hipGraph capture ──
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    # Warmup on the capture stream so allocator state is stable
    with torch.cuda.stream(s):
        launch()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    c_gpu.zero_()
    with torch.cuda.graph(g, stream=s):
        launch()
    torch.cuda.synchronize()

    # ── Replay ──
    c_gpu.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_result = c_gpu.clone()

    # ── Verify ──
    assert graph_result.abs().max().item() > 0, "hipGraph replay produced all zeros — kernel was NOT captured."
    # Same inputs + same kernel + same stream-order = bit-exact equality
    assert torch.equal(eager_result, graph_result), (
        f"Eager vs hipGraph result mismatch: max abs diff = "
        f"{(eager_result.float() - graph_result.float()).abs().max().item():.6f}"
    )


def _bench_kernel_us_cudagraph(run_fn, warmup=10, iters=100, prep_fn=None, n_per_graph=20):
    """Per-launch timer via hipGraph: capture n_per_graph launches, replay iters times, single event pair around the whole replay loop."""
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            if prep_fn is not None:
                prep_fn()
            run_fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    if prep_fn is not None:
        prep_fn()
    with torch.cuda.graph(g, stream=capture_stream):
        for _ in range(n_per_graph):
            run_fn()
    torch.cuda.synchronize()

    # Sanity guard against empty graph capture.
    ref_start = torch.cuda.Event(enable_timing=True)
    ref_end = torch.cuda.Event(enable_timing=True)
    ref_start.record()
    for _ in range(n_per_graph):
        run_fn()
    ref_end.record()
    torch.cuda.synchronize()
    ref_per_launch_us = ref_start.elapsed_time(ref_end) * 1e3 / n_per_graph

    rep_start = torch.cuda.Event(enable_timing=True)
    rep_end = torch.cuda.Event(enable_timing=True)
    rep_start.record()
    g.replay()
    rep_end.record()
    torch.cuda.synchronize()
    first_replay_per_launch_us = rep_start.elapsed_time(rep_end) * 1e3 / n_per_graph

    print(
        f"SANITY_GRAPH,n_per_graph={n_per_graph},"
        f"ref_per_launch_us={ref_per_launch_us:.3f},"
        f"first_replay_per_launch_us={first_replay_per_launch_us:.3f}",
        file=sys.stderr,
        flush=True,
    )
    if first_replay_per_launch_us < 1.0 and ref_per_launch_us > 2.0:
        raise RuntimeError(
            f"hipGraph replay per-launch={first_replay_per_launch_us:.3f}us "
            f"<< ref direct-launch={ref_per_launch_us:.3f}us. "
            f"Graph capture likely empty (stream mismatch?)."
        )

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        g.replay()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) * 1e3 / (iters * n_per_graph)


def _bench_kernel_us(run_fn, warmup=10, iters=50, flush_l2=True, prep_fn=None):
    """Per-iter CUDA events with L2 flush + IQR-trimmed median; fast path uses a single event pair when no flush/prep is requested (preserves back-to-back launch pipelining)."""
    flush_buf = None
    if flush_l2:
        l2_bytes = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()), "L2_cache_size", 4 * 1024 * 1024
        )
        alloc_bytes = max(l2_bytes * 2, 8 * 1024 * 1024)
        flush_buf = torch.empty(alloc_bytes, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        run_fn()
    torch.cuda.synchronize()

    if flush_buf is None and prep_fn is None:
        # Single event pair preserves back-to-back launch pipelining.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / iters

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

    latencies = sorted(start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters))

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

    padded_shape = _get_padded_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, args.split_k)
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
    print(
        f"  Buffers={args.num_buffers}, out={args.out_dtype}, "
        f"opsel={args.use_scale_opsel}, inst_prefetch={args.inst_prefetch}, "
        f"scale_load={args.scale_load_path}"
    )
    if args.split_k > 1:
        print(f"  Split-K={args.split_k} (atomic accumulate, buffer-store epilogue)")
    l2_flush_label = "OFF (graph)" if getattr(args, "use_graph", False) else ("OFF" if args.no_flush_l2 else "ON")
    print(f"  Warmup={args.warmup}, Iters={args.iters}, L2 flush={l2_flush_label}")
    print("  Output init: zero before warmup")
    print("=" * 72)

    torch.manual_seed(0)
    a, b, a_scale, b_scale, fill_spec = _fill_mode_inputs(M, N, K, data_format, getattr(args, "fill_mode", "random"))
    print(f"  Fill mode: {_fill_mode_label(fill_spec, data_format)}")

    a, b, a_scale, b_scale = _pad_mxscale_inputs(a, b, a_scale, b_scale, padded_shape)

    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // args.m_warp
    warp_tile_n = tile_n // args.n_warp
    _coalesced_scale = args.scale_load_path in ("vgpr", "vgpr_ab_split")
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt, coalesced=_coalesced_scale)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt, coalesced=_coalesced_scale)

    K_packed = padded_k // PACK_B
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_out_dtype, device="cuda")

    print("\n[1/3] Compiling kernel...")
    t0 = time.perf_counter()
    use_tdm_store = not args.no_tdm_store
    if args.split_k > 1 and use_tdm_store:
        print("      Note: split-K forces buffer-store atomic epilogue; disabling TDM store.")
        use_tdm_store = False
    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        M=padded_m,
        N=padded_n,
        K=padded_k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        num_buffers=args.num_buffers,
        waves_per_eu=args.waves_per_eu,
        l2_prefetch_distance=args.l2_prefetch_distance,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=args.out_dtype,
        inst_prefetch=args.inst_prefetch,
        wave_specialized_tdm=args.wave_spec_tdm,
        split_k=args.split_k,
        use_scale_opsel=args.use_scale_opsel,
        expert_sched_mode=args.expert_sched_mode,
        atomic_barrier_enable=args.atomic_barrier_enable,
        b_streaming=args.b_streaming,
        scale_load_path=args.scale_load_path,
    )

    compiled_exe = flyc.compile(
        launch_fn,
        c_gpu,
        a_gpu,
        b_gpu,
        as_gpu,
        bs_gpu,
        padded_m,
        padded_n,
        torch.cuda.current_stream(),
    )

    def prep_kernel():
        c_gpu.zero_()

    def run_kernel():
        compiled_exe(
            c_gpu,
            a_gpu,
            b_gpu,
            as_gpu,
            bs_gpu,
            padded_m,
            padded_n,
            torch.cuda.current_stream(),
        )

    prep_kernel()
    run_kernel()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"      Compile + first launch: {compile_ms:.0f} ms")

    use_graph = getattr(args, "use_graph", False)
    if use_graph:
        print(f"[2/3] Warming up ({args.warmup} iters) + bench via hipGraph " f"({args.iters} replays)...")
        us = _bench_kernel_us_cudagraph(run_kernel, warmup=args.warmup, iters=args.iters)
    else:
        print(f"[2/3] Warming up ({args.warmup} iters) + benchmarking ({args.iters} iters)...")
        us = _bench_kernel_us(
            run_kernel, warmup=args.warmup, iters=args.iters, flush_l2=not args.no_flush_l2, prep_fn=prep_kernel
        )

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
    # Sequential WMMAs per workgroup (all k_tiles execute sequentially)
    seq_wmma = k_tiles_local * wmma_per_tile
    us_per_wmma = us / seq_wmma if seq_wmma > 0 else 0

    print("\n[3/3] Results:")
    print(f"      Kernel time:  {us:.1f} us ({us / 1e3:.4f} ms)")
    if not needs_pad:
        print(f"      TFLOPS:       {kernel_tflops:.4f}")
    else:
        print(f"      TFLOPS:       {logical_tflops:.4f} (logical), {kernel_tflops:.4f} (kernel)")
    print(f"      Bandwidth:    {bw_gbs:.1f} GB/s  " f"(read: {read_bw_gbs:.1f} + write: {write_bw_gbs:.1f})")
    print(
        f"      Bytes moved:  {bytes_moved / 1e6:.1f} MB  "
        f"(A={bytes_a / 1e6:.1f} B={bytes_b / 1e6:.1f} "
        f"scale={bytes_scale / 1e6:.1f} D={bytes_d / 1e6:.1f})"
    )
    print("      ---")
    print(f"      WMMA/tile:    {wmma_per_tile} " f"({wmma_m_rep}m × {wmma_n_rep}n × {k_wmma_steps}k)")
    if args.split_k > 1:
        print(
            f"      Total tiles:  {m_tiles}×{n_tiles} spatial × "
            f"{args.split_k} split-K × {k_tiles_local} local K-iters"
        )
    else:
        print(f"      Total tiles:  {m_tiles}×{n_tiles} spatial × {k_tiles} K-iters")
    print(f"      Seq WMMA/WG:  {seq_wmma}")
    print(f"      us/WMMA:      {us_per_wmma:.1f}")
    if us_per_wmma > 1000:
        print(f"      WARNING: {us_per_wmma/1000:.1f} ms/WMMA indicates " f"WMMA_SCALE trap-handler emulation")
    print("=" * 72)

    reported_tflops = kernel_tflops if not needs_pad else logical_tflops
    return us, reported_tflops, bw_gbs


def _run_graph_verify(args):
    """Compare eager launch and hipGraph replay for the CLI-selected shape."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        raise SystemExit(f"WMMA_SCALE requires gfx1250, got {arch}")

    data_format = args.data_format
    M, N, K = args.M, args.N, args.K
    tile_m, tile_n, tile_k = args.tile_m, args.tile_n, args.tile_k
    if K % SCALE_BLOCK != 0:
        raise SystemExit(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")

    padded_shape = _get_padded_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, args.split_k)
    padded_m = padded_shape["M"]
    padded_n = padded_shape["N"]
    padded_k = padded_shape["K"]

    print("=" * 72)
    print(f"  Graph functional verification ({data_format}) on gfx1250")
    print(f"  Shape: M={M}, N={N}, K={K}  (padded {padded_m}x{padded_n}x{padded_k})")
    print(
        f"  Tile: ({tile_m},{tile_n},{tile_k}) warps=({args.m_warp}x{args.n_warp}) "
        f"nb={args.num_buffers} sk={args.split_k} "
        f"cluster=({args.cluster_m},{args.cluster_n})"
    )
    print("=" * 72)

    torch.manual_seed(0)
    a, b, a_scale, b_scale, fill_spec = _fill_mode_inputs(M, N, K, data_format, getattr(args, "fill_mode", "random"))
    expect_nonzero_output = _expect_nonzero_graph_output(a, b, data_format, fill_spec)
    print(f"  Fill: {_fill_mode_label(fill_spec, data_format)}")

    a, b, a_scale, b_scale = _pad_mxscale_inputs(a, b, a_scale, b_scale, padded_shape)

    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // args.m_warp
    warp_tile_n = tile_n // args.n_warp
    _coalesced_scale = args.scale_load_path in ("vgpr", "vgpr_ab_split")
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt, coalesced=_coalesced_scale)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt, coalesced=_coalesced_scale)
    K_packed = padded_k // padded_shape["pack_b"]
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    torch_out_dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[args.out_dtype]
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_out_dtype, device="cuda")

    use_tdm_store = not args.no_tdm_store and args.split_k == 1
    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        M=padded_m,
        N=padded_n,
        K=padded_k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        num_buffers=args.num_buffers,
        waves_per_eu=args.waves_per_eu,
        l2_prefetch_distance=args.l2_prefetch_distance,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=args.out_dtype,
        inst_prefetch=args.inst_prefetch,
        wave_specialized_tdm=args.wave_spec_tdm,
        split_k=args.split_k,
        use_scale_opsel=args.use_scale_opsel,
        expert_sched_mode=args.expert_sched_mode,
        atomic_barrier_enable=args.atomic_barrier_enable,
        b_streaming=args.b_streaming,
        scale_load_path=args.scale_load_path,
    )

    c_flat = c_gpu.contiguous()
    a_flat = a_gpu.contiguous()
    b_flat = b_gpu.contiguous()
    as_flat = as_gpu.contiguous()
    bs_flat = bs_gpu.contiguous()
    compiled_exe = flyc.compile(
        launch_fn,
        c_flat,
        a_flat,
        b_flat,
        as_flat,
        bs_flat,
        padded_m,
        padded_n,
        torch.cuda.current_stream(),
    )

    def launch():
        compiled_exe(c_flat, a_flat, b_flat, as_flat, bs_flat, padded_m, padded_n, torch.cuda.current_stream())

    c_gpu.zero_()
    launch()
    torch.cuda.synchronize()
    eager_result = c_gpu.clone()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        launch()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    c_gpu.zero_()
    with torch.cuda.graph(g, stream=s):
        launch()
    torch.cuda.synchronize()

    c_gpu.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_result = c_gpu.clone()

    if expect_nonzero_output:
        if eager_result.abs().max().item() == 0:
            raise SystemExit(
                "FAIL: eager run produced all zeros -- kernel did not execute (unexpected for non-zero fill)."
            )
        if graph_result.abs().max().item() == 0:
            raise SystemExit(
                "FAIL: hipGraph replay produced all zeros -- kernel was NOT captured (stream mismatch suspected)."
            )
    if not torch.equal(eager_result, graph_result):
        diff = (eager_result.float() - graph_result.float()).abs().max().item()
        raise SystemExit(f"FAIL: eager vs hipGraph result mismatch, max abs diff = {diff:.6f}")

    sample_max = eager_result.abs().max().item()
    print(
        f"  Eager output |max| = {sample_max:.6g}"
        + ("" if expect_nonzero_output else "  (zero is expected for this fill)")
    )
    print("  PASS: eager == hipGraph replay (bit-exact)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-format", type=str, default="fp8", choices=["fp4", "fp8", "a8w4"])
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--l2-prefetch-distance", type=int, default=2)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--out-dtype", type=str, default="bf16", choices=["f32", "bf16", "f16"])
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--no-wave-spec-tdm", dest="wave_spec_tdm", action="store_false", default=True)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--use-scale-opsel", action="store_true", default=False)
    parser.add_argument(
        "--scale-load-path",
        type=str,
        default="tdm",
        choices=["tdm", "vgpr", "vgpr_ab_split"],
    )
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode", action="store_false", default=True)
    parser.add_argument("--b-streaming", action="store_true", default=False)
    parser.add_argument(
        "--atomic-barrier-enable",
        action="store_true",
        default=False,
        help="Enable TDM atomic_barrier_enable (hardware auto-barrier)",
    )

    parser.add_argument(
        "--benchmark", action="store_true", default=False, help="Run benchmark mode (timing only, no correctness check)"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-flush-l2", action="store_true", default=False)
    parser.add_argument(
        "--use-graph",
        action="store_true",
        default=False,
        help="Time via hipGraph capture+replay to strip "
        "host launch overhead from per-launch latency. "
        "Implicitly disables L2 flush (graph replays "
        "are back-to-back, hot-cache).",
    )
    parser.add_argument(
        "--verify-graph",
        action="store_true",
        default=False,
        help="Functional verification: capture the kernel in a hipGraph, "
        "replay once, assert bit-exact match against an eager launch. ",
    )
    parser.add_argument(
        "--fill-mode",
        type=str,
        default="random",
        help="Input fill mode: 'random', 'zero', or a finite float. Constant "
        "mode uses FP8/FP4 encodings for A/B and neutral E8M0 scales.",
    )
    args = parser.parse_args()

    if args.verify_graph:
        _run_graph_verify(args)
        if not args.benchmark:
            sys.exit(0)
    if args.benchmark:
        _run_benchmark(args)
    else:
        use_tdm_store = not args.no_tdm_store and args.split_k == 1
        _run_mxscale_gemm_test(
            args.data_format,
            args.M,
            args.N,
            args.K,
            args.tile_m,
            args.tile_n,
            args.tile_k,
            args.m_warp,
            args.n_warp,
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
            b_streaming=args.b_streaming,
            scale_load_path=args.scale_load_path,
        )
