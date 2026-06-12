#!/usr/bin/env python3
"""MXFP4/MXFP8/A8W4 and PTPC-FP8 GEMM correctness tests for gfx1250.

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
from kernels.gemm_fp8fp4_gfx1250 import (  # noqa: E402
    compile_mxscale_gemm,
    compile_ptpc_gemm,
    is_ref_segmented_lds_layout,
)
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


def preshuffle_e8m0_scale_coalesced_general(
    scale: torch.Tensor,
    warp_tile: int,
    scale_k_per_tile: int,
    kgrp_shift: int,
    WMMA_DIM: int = 16,
    row_align: int = None,
    b128: bool = False,
) -> torch.Tensor:
    """General lane-major scale layout for the buffer_load->VGPR path.

    Generalizes :func:`preshuffle_e8m0_scale_coalesced` (which is locked to
    warp_tile=128, scale_k_per_tile=4, no lane_kgrp) to arbitrary ``warp_tile``,
    ``scale_k_per_tile`` (=> ``k_wmma_steps``) and the a8w4/fp4 ``lane_kgrp``
    scale shift. The byte delivered to every *physical* 32-lane index
    ``lane32 = lane_kgrp*16 + lane16`` is baked here so the kernel's load address
    is lane_kgrp-agnostic and fully coalesced (32 lanes -> 512 contiguous bytes
    per (k_wmma_step, rep_group)).

    Per (row_block of ``warp_tile`` rows, K-tile): layout is
    ``[k_wmma_step, rep_group(ceil(rep/4)), lane32(32), j(4), spw(4)]`` and the
    value at ``(lane32=(G*16+L), rep=grp*4+j)`` is the original e8m0 quadruple of
    row ``block*warp_tile + (rep + G*kgrp_shift)*16 + L``; slots whose shifted rep
    reaches ``rep_count`` are filled with E8M0 127 (=1.0) guards.

    Mirrors the TDM/LDS path value-for-value (see the offline parity check in
    ``flydsl_fp8_perf/verify_vgpr_scale_layout.py``), so it is correct by
    construction against the working ``scale_load_path='tdm'`` path.
    """
    rows, K_scale = scale.shape
    assert K_scale % scale_k_per_tile == 0, f"K_scale={K_scale} % spt={scale_k_per_tile}"
    assert scale_k_per_tile % 4 == 0, f"scale_k_per_tile={scale_k_per_tile} must be a multiple of 4"
    align = row_align if row_align is not None else warp_tile
    if rows % align != 0:
        pad = _align_up(rows, align) - rows
        scale = torch.cat([scale, torch.full((pad, K_scale), 127, dtype=scale.dtype, device=scale.device)], dim=0)
        rows = scale.shape[0]
    R = warp_tile // WMMA_DIM  # rep_count (wmma reps per warp tile)
    KS = scale_k_per_tile // 4  # k_wmma_steps
    KG = K_scale // scale_k_per_tile  # K-tiles
    NG = (R + 3) // 4  # rep groups (vec4 b128 each)
    num_mb = rows // warp_tile

    # Index grids over [mb, kt, ks, grp, lane32, j]; spw is the trailing dim.
    dev = scale.device
    mb = torch.arange(num_mb, device=dev).view(num_mb, 1, 1, 1, 1, 1)
    kt = torch.arange(KG, device=dev).view(1, KG, 1, 1, 1, 1)
    ks = torch.arange(KS, device=dev).view(1, 1, KS, 1, 1, 1)
    grp = torch.arange(NG, device=dev).view(1, 1, 1, NG, 1, 1)
    l32 = torch.arange(32, device=dev).view(1, 1, 1, 1, 32, 1)
    j = torch.arange(4, device=dev).view(1, 1, 1, 1, 1, 4)
    G = l32 // 16
    L = l32 % 16
    rep = grp * 4 + j
    orig_rep = rep + G * kgrp_shift
    valid = orig_rep < R
    orig_row = mb * warp_tile + orig_rep * WMMA_DIM + L
    # Clamp out-of-range (guard) rows so the gather stays in bounds; masked below.
    orig_row = torch.where(valid, orig_row, torch.zeros_like(orig_row))
    colg = kt * KS + ks  # group-of-4 column index into the K dimension
    # Gather the 4 spw bytes for each (row, colg): scale viewed as [rows, KG*KS, 4].
    scale_g = scale.view(rows, KG * KS, 4)
    row_idx, colg_idx = torch.broadcast_tensors(orig_row, colg)
    out = scale_g[row_idx, colg_idx]  # [mb, KG, KS, NG, 32, 4, 4]
    out = torch.where(valid.unsqueeze(-1), out, torch.full_like(out, 127))
    if b128:
        # b128 variant: move ks to the innermost (next to spw) so a lane's
        # KS scale-words are contiguous -> one buffer_load_b128 reads a whole
        # tile's ks per (rep,lane). Output order [mb, kt, grp, j, lane32, ks, spw].
        out = out.permute(0, 1, 3, 5, 4, 2, 6).contiguous()
    return out.reshape(num_mb, -1).contiguous()


def preshuffle_e8m0_scale(
    scale: torch.Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
    WMMA_DIM: int = 16,
    coalesced: bool = False,
    row_align: int = None,
) -> torch.Tensor:
    """Preshuffle E8M0 scale: optional byte swap + interleave for WMMA access.

    ``coalesced=True`` produces the lane-major layout the scale_load_path
    "vgpr"/"vgpr_ab_split" buffer_load->VGPR path expects.
    """
    if coalesced:
        return preshuffle_e8m0_scale_coalesced(scale, block=warp_tile)
    rows, K_scale = scale.shape
    assert K_scale % 4 == 0, f"K_scale must be divisible by 4, got {K_scale}"
    # Accept an unpadded row count (M for a_scale / N for b_scale): pad rows to
    # row_align (the GEMM reads tile_m-granular tiles, so callers pass row_align=tile_m)
    # with E8M0 127 (=1.0). Padding rows feed only discarded output rows. No-op when
    # already aligned. Defaults to warp_tile (the minimum the reshape needs).
    align = row_align if row_align is not None else warp_tile
    if rows % align != 0:
        pad = _align_up(rows, align) - rows
        scale = torch.cat([scale, torch.full((pad, K_scale), 127, dtype=scale.dtype, device=scale.device)], dim=0)
    SCALES_PER_WMMA = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def preshuffle_scale_for_load_path(
    scale, warp_tile, skt, *, scale_load_path, data_format, ref_segmented, row_align=None
):
    """Host scale preshuffle matching the kernel's selected scale_load_path.

    - 'tdm': interleaved TDM/LDS layout.
    - 'vgpr'/'vgpr_ab_split' on the ref-segmented deep-pipeline config: legacy
      lane-major coalesced layout.
    - 'vgpr' on any other (general) config: general coalesced layout, with the
      a8w4/fp4 lane_kgrp scale shift.
    """
    if scale_load_path in ("vgpr", "vgpr_ab_split"):
        if ref_segmented:
            return preshuffle_e8m0_scale(scale, warp_tile, scale_k_per_tile=skt, coalesced=True)
        kgrp_shift = 1 if data_format in ("a8w4", "fp4") else 0
        # FLYDSL_BUFFER_VGPR_SCALE_PRELOAD=1 switches the general vgpr path to the
        # b128 layout (ks innermost) so the kernel reads scales with wide
        # buffer_load_b128 instead of many b32. Must match the kernel flag of the
        # same name (kernels/gemm_fp8fp4_gfx1250.py::_bvs_b128).
        b128 = bool(int(os.environ.get("FLYDSL_BUFFER_VGPR_SCALE_PRELOAD", "0")))
        return preshuffle_e8m0_scale_coalesced_general(
            scale, warp_tile, skt, kgrp_shift, row_align=row_align, b128=b128
        )
    return preshuffle_e8m0_scale(scale, warp_tile, scale_k_per_tile=skt, row_align=row_align)


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


_MXFP4_MAGS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nearest_mxfp4_value(value: float) -> float:
    """Nearest E2M1-representable value to `value`, never zero unless value == 0."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    mag = abs(float(value))
    return sign * min(_MXFP4_MAGS, key=lambda m: abs(m - mag))


def _fp4_e2m1_packed_fill(rows: int, cols: int, value: float) -> torch.Tensor:
    # Snap to the nearest nonzero E2M1 value: a raw round of a small fill (0.1)
    # would land on 0 and make the whole weight tensor vanish.
    snapped = _nearest_mxfp4_value(value)
    dense = torch.full((rows, cols), float(snapped), dtype=torch.float32)
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
    if data_format in ("fp4", "a8w4"):
        eff = _nearest_mxfp4_value(fill_spec[1])
        label += f", FP4={eff:g}"
        if eff != fill_spec[1]:
            label += f" (snapped from {fill_spec[1]:g})"
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
    """Validate tile alignment and return the (unpadded) kernel dimensions.

    N/K must divide their tiles; M is ragged (hardware OOB). Fail loudly instead
    of silently host-padding.
    """
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n} (no silent pad)")
    if K % (tile_k * split_k) != 0:
        raise ValueError(f"K={K} must be divisible by tile_k*split_k={tile_k * split_k} (no silent pad)")

    pack_a, pack_b = _mxscale_pack_factors(data_format)
    return {
        "M": M,
        "N": N,
        "K": K,
        "K_scale": K // SCALE_BLOCK,
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
    a_load_path="tdm",
    b_split_load=False,
    return_launch_fn=False,
    use_graph=False,
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

    # Split-K accumulates at the output precision.
    kernel_out_dtype = out_dtype
    torch_kernel_dtype = _dtype_map[kernel_out_dtype]

    torch.manual_seed(0)

    fmt_name = "A8W4" if is_a8w4 else ("MXFP4" if is_fp4 else "MXFP8")
    mcast_str = f", cluster=({cluster_m},{cluster_n})" if cluster_m > 1 or cluster_n > 1 else ""
    tdm_str = ", tdm_store" if use_tdm_store else ", buffer_store"
    scale_load_str = "" if scale_load_path == "tdm" else f", scale_load={scale_load_path}"
    a_load_str = "" if a_load_path == "tdm" else f", a_load={a_load_path}"
    b_split_str = ", b_split_load" if b_split_load else ""
    graph_str = ", graph" if use_graph else ""
    pad_str = _format_kernel_pad(M, N, K, padded_shape)
    print(
        f"\nRunning {fmt_name} GEMM: M={M}, N={N}, K={K}{pad_str}, "
        f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}"
        f"{mcast_str}{tdm_str}{scale_load_str}{a_load_str}{b_split_str}{graph_str}, preshuffle, out={out_dtype}"
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
    _ref_seg = is_ref_segmented_lds_layout(
        data_format=data_format,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        split_k=split_k,
        wave_specialized_tdm=wave_specialized_tdm,
        use_scale_opsel=use_scale_opsel,
    )
    a_scale = preshuffle_scale_for_load_path(
        a_scale,
        warp_tile_m,
        skt,
        scale_load_path=scale_load_path,
        data_format=data_format,
        ref_segmented=_ref_seg,
        row_align=tile_m,
    )
    b_scale = preshuffle_scale_for_load_path(
        b_scale,
        warp_tile_n,
        skt,
        scale_load_path=scale_load_path,
        data_format=data_format,
        ref_segmented=_ref_seg,
        row_align=tile_n,
    )

    # Preshuffle B data
    K_packed = padded_k // padded_shape["pack_b"]
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    # Upload & launch
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_kernel_dtype, device="cuda")

    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
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
        out_dtype=kernel_out_dtype,
        inst_prefetch=inst_prefetch,
        wave_specialized_tdm=wave_specialized_tdm,
        split_k=split_k,
        use_scale_opsel=use_scale_opsel,
        expert_sched_mode=expert_sched_mode,
        b_streaming=b_streaming,
        scale_load_path=scale_load_path,
        a_load_path=a_load_path,
        b_split_load=b_split_load,
    )

    # Keep 2D — dynamic_layout=True packs shape as i32; flattening overflows for M*K >= 2^31.
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
        padded_k,
        padded_n,
        torch.cuda.current_stream(),
    )

    if use_graph:

        def _launch():
            compiled_exe(
                c_flat,
                a_flat,
                b_flat,
                as_flat,
                bs_flat,
                padded_m,
                padded_n,
                padded_k,
                padded_n,
                torch.cuda.current_stream(),
            )

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _launch()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        c_gpu.zero_()
        with torch.cuda.graph(g, stream=s):
            _launch()
        c_gpu.zero_()
        g.replay()
    torch.cuda.synchronize()

    c_out = c_gpu[:M, :N].to(torch_out_dtype).cpu()

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
            # split-k atomic-adds at output precision; peak-scale tolerance to
            # absorb the compounded bf16/f16 rounding on large-magnitude outputs.
            if split_k > 1:
                peak = float(ref_f.abs().max())
                torch.testing.assert_close(c_out_f, ref_f, rtol=2e-2, atol=max(5e-2, 2e-2 * peak))
            else:
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


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp8_gemm_splitk(split_k, out_dtype):
    """FP8 split-K: split_k workgroups accumulate partial K-sums into C via atomic add.

    Exercises the atomic epilogue path (use_tdm_store=False). K=2048/tile_k=128 gives
    every split_k value >= 2 local K-tiles (needed for double buffering).
    """
    _run_mxscale_gemm_test(
        "fp8",
        128,
        256,
        2048,
        128,
        256,
        128,
        2,
        4,
        num_buffers=2,
        use_tdm_store=False,
        out_dtype=out_dtype,
        l2_prefetch_distance=2,
        split_k=split_k,
    )


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 5632, 2816, 128, 256, 256, 2, 2),
        (128, 2816, 2816, 128, 256, 256, 2, 2),
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
        (13, 2816, 2816, True),
        (33, 5632, 2816, False),
    ],
)
def test_a8w4_gemm_irregular_m_tile16(M, N, K, use_tdm_store):
    # Small-M path: ragged M via OOB, one wave dedicated to the M dimension.
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
    "scale_load_path, a_load_path, b_split_load",
    [
        ("tdm", "vgpr", False),
        ("tdm", "vgpr", True),
        ("tdm", "tdm", True),
        ("vgpr", "tdm", False),
    ],
)
def test_a8w4_small_m_load_path_variants(scale_load_path, a_load_path, b_split_load):
    _run_mxscale_gemm_test(
        "a8w4",
        1,
        256,
        2048,
        16,
        64,
        512,
        1,
        4,
        num_buffers=4,
        use_tdm_store=True,
        out_dtype="bf16",
        wave_specialized_tdm=True,
        l2_prefetch_distance=0,
        use_scale_opsel=False,
        scale_load_path=scale_load_path,
        a_load_path=a_load_path,
        b_split_load=b_split_load,
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
def test_mxfp8_wave_spec_scale_load_tdm(num_buffers, use_tdm_store, use_scale_opsel):
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
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n",
    [
        ("fp8", 128, 256, 256, 128, 256, 128, 2, 2, 1, 1),
        ("fp4", 128, 256, 256, 128, 256, 128, 2, 2, 1, 1),
        ("fp8", 256, 512, 256, 128, 256, 128, 2, 2, 2, 2),
        ("fp4", 256, 512, 256, 128, 256, 128, 2, 2, 2, 2),
        ("a8w4", 256, 512, 256, 128, 256, 128, 2, 4, 2, 2),
    ],
    ids=[
        "fp8-128x256x256",
        "fp4-128x256x256",
        "fp8-256x512x256-cluster2x2",
        "fp4-256x512x256-cluster2x2",
        "a8w4-256x512x256-cluster2x2",
    ],
)
def test_mxscale_gemm_cudagraph(data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n):
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
    is_a8w4 = data_format == "a8w4"

    # Build inputs (mirrors _run_mxscale_gemm_test, but no padding needed
    # because we pick a clean shape).
    torch.manual_seed(0)
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

    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    a_scale_ps = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt)
    b_scale_ps = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)
    pack_b = 2 if (is_fp4 or is_a8w4) else 1
    b_ps = fp4_utils.preshuffle_b_16x16(b, N, K // pack_b)

    a_gpu = a.cuda()
    b_gpu = b_ps.cuda()
    as_gpu = a_scale_ps.cuda()
    bs_gpu = b_scale_ps.cuda()
    c_gpu = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
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
        cluster_m=cluster_m,
        cluster_n=cluster_n,
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
        K,
        N,
        torch.cuda.current_stream(),
    )

    # Resolve stream lazily inside the launch closure so graph capture sees
    # the active capture stream rather than a stream bound before capture.
    def launch():
        compiled_exe(c_flat, a_flat, b_flat, as_flat, bs_flat, M, N, K, N, torch.cuda.current_stream())

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


def _l2_cache_bytes() -> int:
    """Reported L2 size (gfx1250 under-reports the effective LLC, so callers floor this)."""
    return getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "L2_cache_size", 4 * 1024 * 1024)


def _make_l2_flush_buffer(flush_l2: bool, flush_mb: int) -> torch.Tensor | None:
    """Allocate a scratch buffer used only to evict data from L2."""
    if not flush_l2 or flush_mb <= 0:
        return None
    nbytes = int(flush_mb) * 1024 * 1024
    if nbytes <= 0:
        return None
    nelem = max(1, nbytes // torch.empty((), dtype=torch.int32).element_size())
    cache = torch.empty(nelem, dtype=torch.int32, device="cuda")
    cache.zero_()
    torch.cuda.synchronize()
    return cache


def _graph_rotate_slot_count(working_set_bytes: int, target_bytes: int = 0, cap: int = 512) -> int:
    """Number of graph-captured buffer slots for cold-L2 graph replay."""
    target = max(_l2_cache_bytes() * 5, int(target_bytes), 1)
    needed = 1 + math.ceil(target / max(working_set_bytes, 1))
    return max(2, min(needed, cap))


def _flush_l2_cache(cache: torch.Tensor | None):
    if cache is not None:
        cache.zero_()


def _iqr_trimmed_median_us(latencies_us: list[float]) -> float:
    latencies = sorted(latencies_us)
    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in latencies if lo <= x <= hi]
        if filtered:
            latencies = filtered
    return latencies[len(latencies) // 2]


def _bench_kernel_us_cudagraph(
    run_slot,
    num_slots=1,
    warmup=10,
    iters=100,
    n_per_graph=20,
    post_run_slot=None,
):
    """Per-launch timer via hipGraph."""
    cold_rotate = num_slots > 1
    n_per_graph = num_slots if cold_rotate else (1 if post_run_slot is not None else max(1, n_per_graph))
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    def post_run_all_slots():
        if post_run_slot is not None:
            for slot in range(num_slots):
                post_run_slot(slot)

    def run_direct_graph_body():
        if cold_rotate:
            for slot in range(num_slots):
                run_slot(slot)
        else:
            for _ in range(n_per_graph):
                run_slot(0)

    pre_capture_warmup = max(warmup, num_slots if cold_rotate else warmup)
    with torch.cuda.stream(capture_stream):
        post_run_all_slots()
        for i in range(pre_capture_warmup):
            slot = i % num_slots
            run_slot(slot)
            if post_run_slot is not None:
                post_run_slot(slot)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graphs = []
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g, stream=capture_stream):
            run_direct_graph_body()
    graphs.append(g)
    torch.cuda.synchronize()

    def replay_graph_body():
        graphs[0].replay()

    ref_start = torch.cuda.Event(enable_timing=True)
    ref_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(capture_stream):
        run_direct_graph_body()
        post_run_all_slots()
        ref_start.record()
        run_direct_graph_body()
        ref_end.record()
        post_run_all_slots()
    torch.cuda.synchronize()
    ref_per_launch_us = ref_start.elapsed_time(ref_end) * 1e3 / n_per_graph

    rep_start = torch.cuda.Event(enable_timing=True)
    rep_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(capture_stream):
        replay_graph_body()
        post_run_all_slots()
        rep_start.record()
        replay_graph_body()
        rep_end.record()
        post_run_all_slots()
    torch.cuda.synchronize()
    first_replay_per_launch_us = rep_start.elapsed_time(rep_end) * 1e3 / n_per_graph

    print(
        f"SANITY_GRAPH,n_per_graph={n_per_graph},"
        f"ref_per_launch_us={ref_per_launch_us:.3f},"
        f"first_replay_per_launch_us={first_replay_per_launch_us:.3f},"
        f"cold_rotate_slots={num_slots if cold_rotate else 0}",
        file=sys.stderr,
        flush=True,
    )
    if (
        ref_per_launch_us > 2.0
        and first_replay_per_launch_us < 0.25 * ref_per_launch_us
        and first_replay_per_launch_us < 1.0
    ):
        raise RuntimeError(
            f"hipGraph replay per-launch={first_replay_per_launch_us:.3f}us "
            f"<< ref direct-launch={ref_per_launch_us:.3f}us. "
            f"Graph capture likely empty (uncaptured cluster launch or stream mismatch?)."
        )

    # Stabilize graph replay before collecting samples.
    with torch.cuda.stream(capture_stream):
        replay_graph_body()
        post_run_all_slots()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.cuda.stream(capture_stream):
        for i in range(iters):
            start_ev[i].record()
            replay_graph_body()
            end_ev[i].record()
            post_run_all_slots()
    torch.cuda.synchronize()

    latencies_us = [start_ev[i].elapsed_time(end_ev[i]) * 1e3 / n_per_graph for i in range(iters)]
    return _iqr_trimmed_median_us(latencies_us)


def _bench_kernel_us(run_once, flush_cache=None, warmup=10, iters=50, post_run=None):
    """Per-iter CUDA-event timer with optional pre-launch L2 flush + IQR-trimmed median."""
    if post_run is not None:
        post_run()
    for _ in range(warmup):
        _flush_l2_cache(flush_cache)
        run_once()
        if post_run is not None:
            post_run()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        _flush_l2_cache(flush_cache)
        start_ev[i].record()
        run_once()
        end_ev[i].record()
        if post_run is not None:
            post_run()

    torch.cuda.synchronize()

    latencies_us = [start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters)]
    return _iqr_trimmed_median_us(latencies_us)


def reference_ptpc_gemm(data_format, a, b, sa, sb, M, N, K):
    """PTPC reference: D = (A @ B^T) * sa[:,None] * sb[None,:].

    data_format="fp8": FP8 activation + FP8 weight.
    data_format="a8w4": FP8 activation + FP4 (E2M1) weight.
    """
    a_f32 = fp4_utils.fp8_e4m3_to_f32(a.view(torch.uint8))[:M, :K]
    convert_b = fp4_utils.mxfp4_to_f32 if data_format == "a8w4" else fp4_utils.fp8_e4m3_to_f32
    b_f32 = convert_b(b.view(torch.uint8))[:N, :K]
    raw = torch.matmul(a_f32, b_f32.T)
    return raw * sa[:M].view(M, 1) * sb[:N].view(1, N)


def _run_ptpc_gemm_test(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    out_dtype,
    *,
    data_format="fp8",
    l2_prefetch_distance=2,
    cluster_m=1,
    cluster_n=1,
    split_k=1,
    lda_pad=0,
    ldc_pad=0,
):
    """Correctness body for PTPC (per-token per-channel) GEMM.

    A scale sa[M] (per-token) and B scale sb[N] (per-channel) are fp32, constant
    along K. The K-loop runs the WMMA unscaled (fp8) or with an identity scale
    (a8w4); sa*sb is applied in the epilogue. data_format: "fp8" or "a8w4".
    """
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"PTPC requires gfx1250, got {arch}")

    padded_shape = _get_padded_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, split_k)
    padded_m, padded_n, padded_k = padded_shape["M"], padded_shape["N"], padded_shape["K"]
    local_k = padded_k // split_k
    num_k_tiles = local_k // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]
    kernel_out_dtype = out_dtype  # split-k atomic-adds at output precision
    torch_kernel_dtype = _dtype_map[kernel_out_dtype]

    torch.manual_seed(0)
    a = random_fp8_data(M, K)  # FP8 activation for both fp8 and a8w4
    b = fp4_utils.random_fp4_packed(N, K) if data_format == "a8w4" else random_fp8_data(N, K)
    # Per-token / per-channel fp32 scales in a benign range to avoid degeneracy.
    sa = (0.5 + torch.rand(M, dtype=torch.float32)).contiguous()
    sb = (0.5 + torch.rand(N, dtype=torch.float32)).contiguous()

    ref = reference_ptpc_gemm(data_format, a, b, sa, sb, M, N, K)
    print(
        f"\nRunning PTPC {data_format.upper()} GEMM: M={M}, N={N}, K={K}, tiles=({tile_m},{tile_n},{tile_k}), "
        f"bufs={num_buffers}, split_k={split_k}, out={out_dtype}"
    )
    print(f"Ref stats: min={ref.min():.2f}, max={ref.max():.2f}, mean={ref.mean():.2f}, std={ref.std():.2f}")

    # Pad data to tile-aligned shapes; B is preshuffled like the mxscale path.
    # A8W4 packs the FP4 weight 2-per-byte, so B's column count is K/pack_b.
    K_packed_b = padded_k // padded_shape["pack_b"]
    a = _pad_2d_tensor(a, padded_m, padded_k, fill_value=0)
    b = _pad_2d_tensor(b, padded_n, K_packed_b, fill_value=0)
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed_b)
    # Pad scales (pad region is discarded in the [:M,:N] slice).
    sa_p = torch.zeros(padded_m, dtype=torch.float32)
    sa_p[:M] = sa
    sb_p = torch.zeros(padded_n, dtype=torch.float32)
    sb_p[:N] = sb

    # Optional strided A/C: back data with a wider leading dim (lda/ldc), exercising
    # the runtime-stride descriptor path. lda/ldc are logical leading dims (elements).
    pack_a = padded_shape["pack_a"]
    lda = padded_k + lda_pad
    ldc = padded_n + ldc_pad
    if lda_pad:
        a_full = torch.zeros(padded_m, lda // pack_a, dtype=a.dtype)
        a_full[:, : padded_k // pack_a] = a
        a = a_full

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    sa_gpu = sa_p.cuda()
    sb_gpu = sb_p.cuda()
    c_gpu = torch.zeros(padded_m, ldc, dtype=torch_kernel_dtype, device="cuda")

    launch_fn = compile_ptpc_gemm(
        N=padded_n,
        K=padded_k,
        data_format=data_format,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        out_dtype=kernel_out_dtype,
        split_k=split_k,
    )

    flyc.compile(
        launch_fn,
        c_gpu.contiguous(),
        a_gpu.contiguous(),
        b_gpu.contiguous(),
        sa_gpu.contiguous(),
        sb_gpu.contiguous(),
        padded_m,
        padded_n,
        lda,
        ldc,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    c_out = c_gpu[:M, :N].to(torch_out_dtype).cpu()
    print(
        f"Out stats: min={c_out.float().min():.2f}, max={c_out.float().max():.2f}, "
        f"mean={c_out.float().mean():.2f}, std={c_out.float().std():.2f}"
    )
    if c_out.float().abs().max() < 1e-10:
        print("WARNING: kernel output is all zeros!")

    c_out_f = c_out.float()
    ref_f = ref.to(torch_out_dtype).float() if out_dtype in ("bf16", "f16") else ref.float()
    diff = (c_out_f - ref_f).abs()
    print(f"Abs diff: max={diff.max():.4f}, mean={diff.mean():.4f}")
    cos_sim = torch.nn.functional.cosine_similarity(
        c_out_f.flatten().unsqueeze(0).double(), ref_f.flatten().unsqueeze(0).double()
    ).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    peak = float(ref_f.abs().max())
    if out_dtype in ("bf16", "f16"):
        torch.testing.assert_close(c_out_f, ref_f, rtol=2e-2, atol=max(5e-2, 2e-2 * peak))
    else:
        torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=max(1e-2, K * 0.6))
    print("PASSED")


@pytest.mark.parametrize("out_dtype", ["bf16", "f32"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers",
    [
        (256, 256, 512, 256, 256, 128, 2, 2, 4),  # deep-pipeline eligible
        (128, 256, 512, 128, 256, 128, 2, 2, 4),  # quadrant fallback
    ],
)
def test_ptpc_fp8_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype):
    _run_ptpc_gemm_test(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype)


@pytest.mark.parametrize("lda_pad, ldc_pad", [(128, 0), (0, 256), (128, 256)])
def test_ptpc_fp8_gemm_strided(lda_pad, ldc_pad):
    """Strided A/C: data backed by a wider leading dim, passed via runtime lda/ldc."""
    _run_ptpc_gemm_test(
        128, 256, 512, 128, 256, 128, 2, 2, num_buffers=4, out_dtype="bf16", lda_pad=lda_pad, ldc_pad=ldc_pad
    )


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("out_dtype", ["bf16", "f32"])
def test_ptpc_fp8_gemm_splitk(split_k, out_dtype):
    """PTPC split-K: each chunk applies sa*sb then atomic-adds; sum stays correct."""
    _run_ptpc_gemm_test(128, 256, 2048, 128, 256, 128, 2, 4, num_buffers=2, out_dtype=out_dtype, split_k=split_k)


@pytest.mark.parametrize("out_dtype", ["bf16", "f32"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers",
    [
        (128, 256, 512, 128, 256, 128, 2, 4, 2),  # row-major (a8w4) + wave-spec TDM
        (128, 256, 1024, 128, 256, 256, 2, 4, 3),
    ],
)
def test_ptpc_a8w4_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype):
    """PTPC A8W4 (FP8 act + FP4 weight): K-loop uses identity-scale f8f6f4 WMMA;
    real per-token/per-channel sa*sb is applied in the epilogue."""
    _run_ptpc_gemm_test(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, data_format="a8w4")


@pytest.mark.parametrize("split_k", [2, 4])
def test_ptpc_a8w4_gemm_splitk(split_k):
    """PTPC A8W4 split-K: identity-scale K-loop + epilogue sa*sb + atomic add."""
    _run_ptpc_gemm_test(
        128, 256, 2048, 128, 256, 128, 2, 4, num_buffers=2, out_dtype="bf16", split_k=split_k, data_format="a8w4"
    )


# ---------------------------------------------------------------------------
# Non-tile-aligned M (the default, no host M-padding): A/C (and ptpc sa) are
# allocated at the real M. A-load TDM skips rows>=M, sa buffer_load OOB->0, C
# buffer_store clips via num_records. N,K stay tile-aligned.
# ---------------------------------------------------------------------------
_DT = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
_MPAD_MS = [1, 16, 31, 64, 65, 100, 127, 128, 129, 130, 192, 255, 256, 257, 384, 500, 1000, 2048]


def _assert_mpad(c_real, ref, out_dtype):
    c = c_real.float()
    ref_f = ref.to(_DT[out_dtype]).float()
    peak = float(ref_f.abs().max())
    if out_dtype in ("bf16", "f16"):
        torch.testing.assert_close(c, ref_f, rtol=2e-2, atol=max(5e-2, 2e-2 * peak))
    else:
        torch.testing.assert_close(c, ref_f, rtol=1e-3, atol=max(1e-2, ref.shape[-1] * 0.6))


def _run_ptpc_mpad(
    M,
    N,
    K,
    *,
    data_format="fp8",
    out_dtype="bf16",
    split_k=1,
    tile_m=128,
    tile_n=128,
    tile_k=128,
    m_warp=2,
    n_warp=2,
    num_buffers=4,
    cluster_m=1,
    cluster_n=1,
):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")
    assert N % tile_n == 0 and K % tile_k == 0, "M-pad test keeps N,K tile-aligned"
    # split_k atomic-adds at output precision (per-lane predicate on row < M).
    kernel_out_dtype = out_dtype
    torch.manual_seed(0)
    a = random_fp8_data(M, K)
    b = fp4_utils.random_fp4_packed(N, K) if data_format == "a8w4" else random_fp8_data(N, K)
    sa = (0.5 + torch.rand(M, dtype=torch.float32)).contiguous()
    sb = (0.5 + torch.rand(N, dtype=torch.float32)).contiguous()
    ref = reference_ptpc_gemm(data_format, a, b, sa, sb, M, N, K)
    pack_b = 2 if data_format == "a8w4" else 1
    b_ps = fp4_utils.preshuffle_b_16x16(b, N, K // pack_b)
    c_gpu = torch.zeros(M, N, dtype=_DT[kernel_out_dtype], device="cuda")  # real M; zero for atomic
    launch = compile_ptpc_gemm(
        N=N,
        K=K,
        data_format=data_format,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        out_dtype=kernel_out_dtype,
        split_k=split_k,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )
    launch(c_gpu, a.cuda(), b_ps.cuda(), sa.cuda(), sb.cuda(), M, N, K, N, torch.cuda.current_stream())
    torch.cuda.synchronize()
    _assert_mpad(c_gpu[:M].cpu(), ref, kernel_out_dtype)


def _run_mxscale_mpad(
    M,
    N,
    K,
    *,
    out_dtype="bf16",
    use_tdm_store=True,
    tile_m=128,
    tile_n=128,
    tile_k=128,
    m_warp=2,
    n_warp=2,
    num_buffers=4,
    cluster_m=1,
    cluster_n=1,
):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")
    assert N % tile_n == 0 and K % tile_k == 0, "M-pad test keeps N,K tile-aligned"
    torch.manual_seed(0)
    a = random_fp8_data(M, K)
    b = random_fp8_data(N, K)
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)  # real M, unpadded
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)
    ref = reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K)
    skt = tile_k // SCALE_BLOCK
    # a_scale stays UNPADDED host-side; preshuffle pads rows to tile_m (the GEMM
    # reads tile_m-granular scale tiles for the partial last M-tile). N is aligned.
    as_ps = preshuffle_e8m0_scale(a_scale, tile_m // m_warp, scale_k_per_tile=skt, row_align=tile_m)
    bs_ps = preshuffle_e8m0_scale(b_scale, tile_n // n_warp, scale_k_per_tile=skt)
    b_ps = fp4_utils.preshuffle_b_16x16(b, N, K)
    c_gpu = torch.zeros(M, N, dtype=_DT[out_dtype], device="cuda")  # real M
    launch = compile_mxscale_gemm(
        data_format="fp8",
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        out_dtype=out_dtype,
        use_tdm_store=use_tdm_store,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )
    launch(c_gpu, a.cuda(), b_ps.cuda(), as_ps.cuda(), bs_ps.cuda(), M, N, K, N, torch.cuda.current_stream())
    torch.cuda.synchronize()
    _assert_mpad(c_gpu[:M].cpu(), ref, out_dtype)


@pytest.mark.parametrize("out_dtype", ["bf16", "f32"])
@pytest.mark.parametrize("M", _MPAD_MS)
def test_ptpc_fp8_gemm_mpad(M, out_dtype):
    _run_ptpc_mpad(M, 256, 512, out_dtype=out_dtype)


@pytest.mark.parametrize("M", _MPAD_MS)
def test_ptpc_a8w4_gemm_mpad(M):
    _run_ptpc_mpad(M, 256, 512, data_format="a8w4", m_warp=2, n_warp=4, num_buffers=2)


@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("out_dtype", ["bf16", "f32"])
@pytest.mark.parametrize("M", _MPAD_MS)
def test_mxfp8_gemm_mpad(M, out_dtype, use_tdm_store):
    _run_mxscale_mpad(M, 256, 512, out_dtype=out_dtype, use_tdm_store=use_tdm_store)


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("M", [1, 64, 129, 192, 257, 500])
def test_ptpc_fp8_gemm_splitk_mpad(M, split_k):
    # split_k atomic output predicated per-lane on row < M (auto buffer/atomic path).
    _run_ptpc_mpad(M, 256, 2048, m_warp=2, n_warp=4, num_buffers=2, split_k=split_k)


# Tile/warp-config diversity: the per-warp partial-tile clip uses
# warp_tile_m = tile_m // m_warp, so M must be exercised against different warp
# boundaries. Existing mpad tests are all m_warp=2 (warp_tile_m=64); these add
# warp_tile_m in {128 (single M-warp / tile_m=256), 32 (fine 4-way split)}.
_MPAD_WARP_CFGS = [
    # (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    (128, 128, 128, 1, 4, 4),  # warp_tile_m=128: single M-warp, no M split
    (128, 128, 128, 4, 2, 2),  # warp_tile_m=32: fine-grained M warps
    (256, 128, 128, 2, 2, 2),  # tile_m=256, warp_tile_m=128
]
# Boundary-diverse M for warp_tile_m in {32, 128}: partial/full/OOB warps + aligned.
_MPAD_WARP_MS = [1, 33, 64, 100, 129, 200, 256, 333]


@pytest.mark.parametrize("tile_m,tile_n,tile_k,m_warp,n_warp,num_buffers", _MPAD_WARP_CFGS)
@pytest.mark.parametrize("M", _MPAD_WARP_MS)
def test_ptpc_fp8_gemm_mpad_warps(M, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
    _run_ptpc_mpad(
        M,
        256,
        512,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
    )


#   M=100 -> grid_m 1->2, tile1 fully OOB (rows>=100) under M-multicast
#   M=129,200,450 -> partial last M-tile, grid divisible
#   M=256,512 -> tile-aligned
#   M=257,300 -> grid_m 3->4 (rounded); M=300 also makes tile3 fully OOB
_MPAD_CLUSTER_MS = [100, 129, 200, 256, 257, 300, 450, 512]
_MPAD_CLUSTERS = [(2, 2), (2, 4)]


@pytest.mark.parametrize("cluster_m,cluster_n", _MPAD_CLUSTERS)
@pytest.mark.parametrize("M", _MPAD_CLUSTER_MS)
def test_ptpc_fp8_gemm_mpad_cluster(M, cluster_m, cluster_n):
    _run_ptpc_mpad(M, 512, 512, m_warp=2, n_warp=2, num_buffers=2, cluster_m=cluster_m, cluster_n=cluster_n)


@pytest.mark.parametrize("cluster_m,cluster_n", _MPAD_CLUSTERS)
@pytest.mark.parametrize("M", _MPAD_CLUSTER_MS)
def test_ptpc_a8w4_gemm_mpad_cluster(M, cluster_m, cluster_n):
    _run_ptpc_mpad(
        M, 512, 512, data_format="a8w4", m_warp=2, n_warp=4, num_buffers=2, cluster_m=cluster_m, cluster_n=cluster_n
    )


@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("cluster_m,cluster_n", _MPAD_CLUSTERS)
@pytest.mark.parametrize("M", _MPAD_CLUSTER_MS)
def test_mxfp8_gemm_mpad_cluster(M, cluster_m, cluster_n, use_tdm_store):
    _run_mxscale_mpad(
        M,
        512,
        512,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
    )


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("M", [100, 129, 256, 300, 450])
def test_ptpc_fp8_gemm_splitk_mpad_cluster(M, split_k):
    # split_k atomic output (per-lane row<M predicate) combined with cluster>1.
    _run_ptpc_mpad(M, 512, 2048, m_warp=2, n_warp=2, num_buffers=2, split_k=split_k, cluster_m=2, cluster_n=2)


@pytest.mark.parametrize("cluster_m,cluster_n", [(2, 2), (2, 4)])
@pytest.mark.parametrize("M", [100, 300, 512, 600, 700, 1024])
def test_ptpc_fp8_gemm_mpad_cluster_tm256(M, cluster_m, cluster_n):
    _run_ptpc_mpad(
        M,
        1024,
        512,
        tile_m=256,
        tile_n=256,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )


@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("cluster_m,cluster_n", [(2, 2), (2, 4)])
@pytest.mark.parametrize("M", [100, 300, 512, 600, 700, 1024])
def test_mxfp8_gemm_mpad_cluster_tm256(M, cluster_m, cluster_n, use_tdm_store):
    _run_mxscale_mpad(
        M,
        1024,
        512,
        tile_m=256,
        tile_n=256,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
    )


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
    is_ptpc = getattr(args, "scale_mode", "mxscale") == "ptpc"
    if is_ptpc and data_format not in ("fp8", "a8w4"):
        raise ValueError(f"scale_mode='ptpc' only supports data_format='fp8' or 'a8w4', got {data_format!r}")
    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    # split_k atomic-adds at output precision (bf16/f16).
    kernel_out_dtype = args.out_dtype
    torch_kernel_dtype = _dtype_map[kernel_out_dtype]
    elem_bytes_d = 2 if kernel_out_dtype in ("bf16", "f16") else 4
    if is_ptpc:
        fmt_name = "PTPC-A8W4" if is_a8w4 else "PTPC-FP8"
    else:
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
        f"scale_load={args.scale_load_path}, a_load={args.a_load_path}, "
        f"b_split_load={args.b_split_load}"
    )
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    if args.iters <= 0:
        raise ValueError(f"--iters must be > 0, got {args.iters}")
    if args.l2_flush_mb < 0:
        raise ValueError(f"--l2-flush-mb must be >= 0, got {args.l2_flush_mb}")
    if args.split_k > 1:
        print(f"  Split-K={args.split_k} (atomic accumulate, buffer-store epilogue)")
        print("  Split-K timing excludes the required C reset from the reported kernel time")
    if args.no_flush_l2:
        l2_flush_label = "OFF (hot L2, --no-flush-l2)"
    elif args.l2_flush_mb == 0:
        l2_flush_label = "OFF (hot L2, --l2-flush-mb=0)"
    elif getattr(args, "use_graph", False):
        l2_flush_label = "ON (graph rotating buffers; compare against --no-flush-l2)"
    else:
        l2_flush_label = f"ON ({args.l2_flush_mb} MiB scratch clear before timed launches)"
    print(f"  Warmup={args.warmup}, Iters={args.iters}, L2 defeat={l2_flush_label}")
    if is_ptpc:
        # compile_ptpc_gemm forces these internally; flag the ones the user set off-default.
        _ptpc_ignored = []
        if args.no_tdm_store:
            _ptpc_ignored.append("--no-tdm-store")
        if not args.wave_spec_tdm:
            _ptpc_ignored.append("--no-wave-spec-tdm")
        if args.use_scale_opsel:
            _ptpc_ignored.append("--use-scale-opsel")
        if args.scale_load_path != "tdm":
            _ptpc_ignored.append(f"--scale-load-path {args.scale_load_path}")
        if args.b_streaming:
            _ptpc_ignored.append("--b-streaming")
        if args.b_split_load:
            _ptpc_ignored.append("--b-split-load")
        if _ptpc_ignored:
            print(f"  Note: PTPC ignores (forced internally): {', '.join(_ptpc_ignored)}")
    print("=" * 72)

    torch.manual_seed(0)
    warp_tile_m = tile_m // args.m_warp
    warp_tile_n = tile_n // args.n_warp
    if is_ptpc:
        # PTPC: fp8 A with fp32 per-token (sa[M]) / per-channel (sb[N]) scales, no scale preshuffle.
        # B is fp8 (data_format="fp8") or FP4-packed 2-per-byte (data_format="a8w4").
        K_packed_b = padded_k // PACK_B
        b_kind = "fp4 (a8w4)" if is_a8w4 else "fp8"
        fill_spec = _parse_fill_mode(getattr(args, "fill_mode", "random"))
        if fill_spec[0] == "const":
            value = fill_spec[1]
            fp8_byte = _fp8_e4m3fn_byte(value)
            a_raw = torch.full((M, K), fp8_byte, dtype=torch.uint8)
            b_raw = _fp4_e2m1_packed_fill(N, K, value) if is_a8w4 else torch.full((N, K), fp8_byte, dtype=torch.uint8)
            # Neutral per-token/per-channel scales so the const output stays predictable.
            a_scale = torch.zeros(padded_m, dtype=torch.float32)
            a_scale[:M] = 1.0
            b_scale = torch.zeros(padded_n, dtype=torch.float32)
            b_scale[:N] = 1.0
            if is_a8w4:
                eff_b = _nearest_mxfp4_value(value)
                b_note = f"fp4 B={eff_b:g}" + (f" (snapped from {value:g})" if eff_b != value else "")
            else:
                b_note = "fp8 B"
            print(f"  Fill mode: const={value:g} (FP8 byte=0x{fp8_byte:02x}), {b_note}, sa=sb=1.0")
        else:
            a_raw = random_fp8_data(M, K)
            b_raw = fp4_utils.random_fp4_packed(N, K) if is_a8w4 else random_fp8_data(N, K)
            a_scale = torch.zeros(padded_m, dtype=torch.float32)
            a_scale[:M] = 0.5 + torch.rand(M, dtype=torch.float32)
            b_scale = torch.zeros(padded_n, dtype=torch.float32)
            b_scale[:N] = 0.5 + torch.rand(N, dtype=torch.float32)
            print(f"  Fill mode: random fp8 A / {b_kind} B, fp32 per-token/per-channel scales")
        a = _pad_2d_tensor(a_raw, padded_m, padded_k, fill_value=0)
        b = _pad_2d_tensor(b_raw, padded_n, K_packed_b, fill_value=0)
        b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed_b)
    else:
        a, b, a_scale, b_scale, fill_spec = _fill_mode_inputs(
            M, N, K, data_format, getattr(args, "fill_mode", "random")
        )
        print(f"  Fill mode: {_fill_mode_label(fill_spec, data_format)}")

        a, b, a_scale, b_scale = _pad_mxscale_inputs(a, b, a_scale, b_scale, padded_shape)

        skt = tile_k // SCALE_BLOCK
        _ref_seg = is_ref_segmented_lds_layout(
            data_format=data_format,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=args.m_warp,
            n_warp=args.n_warp,
            num_buffers=args.num_buffers,
            split_k=args.split_k,
            wave_specialized_tdm=args.wave_spec_tdm,
            use_scale_opsel=args.use_scale_opsel,
        )
        a_scale = preshuffle_scale_for_load_path(
            a_scale,
            warp_tile_m,
            skt,
            scale_load_path=args.scale_load_path,
            data_format=data_format,
            ref_segmented=_ref_seg,
            row_align=tile_m,
        )
        b_scale = preshuffle_scale_for_load_path(
            b_scale,
            warp_tile_n,
            skt,
            scale_load_path=args.scale_load_path,
            data_format=data_format,
            ref_segmented=_ref_seg,
            row_align=tile_n,
        )

        K_packed = padded_k // PACK_B
        b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_kernel_dtype, device="cuda")

    print("\n[1/3] Compiling kernel...")
    t0 = time.perf_counter()
    use_tdm_store = not args.no_tdm_store
    if args.split_k > 1 and use_tdm_store:
        print("      Note: split-K forces buffer-store atomic epilogue; disabling TDM store.")
        use_tdm_store = False
    if is_ptpc:
        # compile_ptpc_gemm fixes scale_mode/wave_spec/use_tdm_store internally.
        launch_fn = compile_ptpc_gemm(
            N=padded_n,
            K=padded_k,
            data_format=data_format,
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
            out_dtype=kernel_out_dtype,
            inst_prefetch=args.inst_prefetch,
            expert_sched_mode=args.expert_sched_mode,
            atomic_barrier_enable=args.atomic_barrier_enable,
            split_k=args.split_k,
        )
    else:
        launch_fn = compile_mxscale_gemm(
            data_format=data_format,
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
            out_dtype=kernel_out_dtype,
            inst_prefetch=args.inst_prefetch,
            wave_specialized_tdm=args.wave_spec_tdm,
            split_k=args.split_k,
            use_scale_opsel=args.use_scale_opsel,
            expert_sched_mode=args.expert_sched_mode,
            atomic_barrier_enable=args.atomic_barrier_enable,
            b_streaming=args.b_streaming,
            scale_load_path=args.scale_load_path,
            a_load_path=args.a_load_path,
            b_split_load=args.b_split_load,
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
        padded_k,
        padded_n,
        torch.cuda.current_stream(),
    )

    def run_one(c_, a_, b_, as_, bs_):
        compiled_exe(
            c_,
            a_,
            b_,
            as_,
            bs_,
            padded_m,
            padded_n,
            padded_k,
            padded_n,
            torch.cuda.current_stream(),
        )

    c_gpu.zero_()
    run_one(c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"      Compile + first launch: {compile_ms:.0f} ms")

    use_graph = getattr(args, "use_graph", False)
    flush_l2 = not args.no_flush_l2 and args.l2_flush_mb > 0
    working_set = sum(t.numel() * t.element_size() for t in (a_gpu, b_gpu, as_gpu, bs_gpu, c_gpu))
    flush_cache = None if use_graph else _make_l2_flush_buffer(flush_l2, args.l2_flush_mb)
    graph_num_slots = 1
    if use_graph and flush_l2:
        graph_rotate_target = max(_l2_cache_bytes() * 5, int(args.l2_flush_mb) * 1024 * 1024)
        graph_num_slots = _graph_rotate_slot_count(working_set, graph_rotate_target)
        graph_eviction_bytes = max(0, graph_num_slots - 1) * working_set
        cap_note = "  [WARNING: capped below target]" if graph_eviction_bytes < graph_rotate_target else ""
        print(
            f"      L2 defeat: graph rotating buffers, slots={graph_num_slots}, "
            f"pool={working_set * graph_num_slots / 1e6:.1f} MB "
            f"(evict distance={graph_eviction_bytes / 1e6:.1f} MB, "
            f"target={graph_rotate_target / 1e6:.1f} MB, "
            f"reported L2={_l2_cache_bytes() / 1e6:.1f} MB, "
            f"working set {working_set / 1e6:.1f} MB){cap_note}"
        )
    elif flush_cache is None:
        print(f"      L2 defeat: OFF (hot-cache timing), working set {working_set / 1e6:.1f} MB")
    else:
        print(
            f"      L2 defeat: ON, scratch={flush_cache.numel() * flush_cache.element_size() / 1e6:.1f} MB "
            f"(reported L2={_l2_cache_bytes() / 1e6:.1f} MB, working set {working_set / 1e6:.1f} MB)"
        )

    clear_output_each_run = args.split_k > 1

    def run_bench_once():
        run_one(c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu)

    def reset_bench_output():
        c_gpu.zero_()

    if use_graph:
        if graph_num_slots == 1:
            print(f"[2/3] Warming up ({args.warmup} iters) + bench via hot-cache hipGraph ({args.iters} replays)...")
            us = _bench_kernel_us_cudagraph(
                lambda _slot: run_bench_once(),
                num_slots=1,
                warmup=args.warmup,
                iters=args.iters,
                post_run_slot=(lambda _slot: reset_bench_output()) if clear_output_each_run else None,
            )
        else:
            a_pool = [a_gpu] + [a_gpu.clone() for _ in range(graph_num_slots - 1)]
            b_pool = [b_gpu] + [b_gpu.clone() for _ in range(graph_num_slots - 1)]
            as_pool = [as_gpu] + [as_gpu.clone() for _ in range(graph_num_slots - 1)]
            bs_pool = [bs_gpu] + [bs_gpu.clone() for _ in range(graph_num_slots - 1)]
            c_pool = [c_gpu] + [torch.zeros_like(c_gpu) for _ in range(graph_num_slots - 1)]

            def run_graph_slot(slot):
                s = slot % graph_num_slots
                run_one(c_pool[s], a_pool[s], b_pool[s], as_pool[s], bs_pool[s])

            def reset_graph_slot(slot):
                c_pool[slot % graph_num_slots].zero_()

            print(
                f"[2/3] Warming up ({args.warmup} iters) + bench via rotating-buffer hipGraph "
                f"({args.iters} replays × {graph_num_slots} launches/replay, "
                f"rotating graph-captured buffer slots)..."
            )
            us = _bench_kernel_us_cudagraph(
                run_graph_slot,
                num_slots=graph_num_slots,
                warmup=args.warmup,
                iters=args.iters,
                post_run_slot=reset_graph_slot if clear_output_each_run else None,
            )
    else:
        print(f"[2/3] Warming up ({args.warmup} iters) + benchmarking ({args.iters} iters)...")
        us = _bench_kernel_us(
            run_bench_once,
            flush_cache,
            warmup=args.warmup,
            iters=args.iters,
            post_run=reset_bench_output if clear_output_each_run else None,
        )

    WMMA_K = 128
    WMMA_N_EFF = 32 if is_fp4 else 16
    wmma_m_rep = warp_tile_m // 16
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    k_wmma_steps = tile_k // WMMA_K
    wmma_per_tile = wmma_m_rep * wmma_n_rep * k_wmma_steps
    m_tiles = (padded_m + tile_m - 1) // tile_m
    n_tiles = (padded_n + tile_n - 1) // tile_n
    k_tiles = padded_k // tile_k
    k_tiles_local = (padded_k // args.split_k) // tile_k
    # Sequential WMMAs per workgroup (all k_tiles execute sequentially)
    seq_wmma = k_tiles_local * wmma_per_tile
    us_per_wmma = us / seq_wmma if seq_wmma > 0 else 0

    logical_flops = 2.0 * M * N * K
    tile_m_covered = m_tiles * tile_m
    tile_n_covered = n_tiles * tile_n
    tile_flops = 2.0 * tile_m_covered * tile_n_covered * padded_k
    time_s = us / 1e6
    logical_tflops = logical_flops / time_s / 1e12 if time_s > 0 else 0.0
    tile_tflops = tile_flops / time_s / 1e12 if time_s > 0 else 0.0

    bytes_a = padded_m * padded_k // PACK_A
    bytes_b = padded_n * padded_k // PACK_B
    bytes_scale = (padded_m + padded_n) * (4 if is_ptpc else padded_shape["K_scale"])
    bytes_d = padded_m * padded_n * elem_bytes_d
    read_bytes = bytes_a + bytes_b + bytes_scale
    write_bytes = bytes_d
    bytes_moved = read_bytes + write_bytes
    bw_gbs = bytes_moved / 1e9 / time_s if time_s > 0 else 0.0
    read_bw_gbs = read_bytes / 1e9 / time_s if time_s > 0 else 0.0
    write_bw_gbs = write_bytes / 1e9 / time_s if time_s > 0 else 0.0

    print("\n[3/3] Results:")
    print(f"      Kernel time:  {us:.1f} us ({us / 1e3:.4f} ms)")
    if tile_flops == logical_flops:
        print(f"      TFLOPS:       {logical_tflops:.4f}")
    else:
        print(f"      TFLOPS:       {logical_tflops:.4f} (logical), {tile_tflops:.4f} (tile-covered)")
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

    return us, logical_tflops, bw_gbs


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
    _ref_seg = is_ref_segmented_lds_layout(
        data_format=data_format,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        num_buffers=args.num_buffers,
        split_k=args.split_k,
        wave_specialized_tdm=args.wave_spec_tdm,
        use_scale_opsel=args.use_scale_opsel,
    )
    a_scale = preshuffle_scale_for_load_path(
        a_scale,
        warp_tile_m,
        skt,
        scale_load_path=args.scale_load_path,
        data_format=data_format,
        ref_segmented=_ref_seg,
        row_align=tile_m,
    )
    b_scale = preshuffle_scale_for_load_path(
        b_scale,
        warp_tile_n,
        skt,
        scale_load_path=args.scale_load_path,
        data_format=data_format,
        ref_segmented=_ref_seg,
        row_align=tile_n,
    )
    K_packed = padded_k // padded_shape["pack_b"]
    b = fp4_utils.preshuffle_b_16x16(b, padded_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    # split_k atomic-adds at output precision (bf16/f16).
    kernel_out_dtype = args.out_dtype
    c_gpu = torch.zeros(padded_m, padded_n, dtype=_dtype_map[kernel_out_dtype], device="cuda")

    use_tdm_store = not args.no_tdm_store and args.split_k == 1
    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
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
        out_dtype=kernel_out_dtype,
        inst_prefetch=args.inst_prefetch,
        wave_specialized_tdm=args.wave_spec_tdm,
        split_k=args.split_k,
        use_scale_opsel=args.use_scale_opsel,
        expert_sched_mode=args.expert_sched_mode,
        atomic_barrier_enable=args.atomic_barrier_enable,
        b_streaming=args.b_streaming,
        scale_load_path=args.scale_load_path,
        a_load_path=args.a_load_path,
        b_split_load=args.b_split_load,
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
        padded_k,
        padded_n,
        torch.cuda.current_stream(),
    )

    def launch():
        compiled_exe(
            c_flat,
            a_flat,
            b_flat,
            as_flat,
            bs_flat,
            padded_m,
            padded_n,
            padded_k,
            padded_n,
            torch.cuda.current_stream(),
        )

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
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="mxscale",
        choices=["mxscale", "ptpc"],
        help="Scale organization: 'mxscale' (E8M0 block scale) or 'ptpc' "
        "(per-token/per-channel fp32; supports --data-format fp8 or a8w4).",
    )
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=4, choices=[2, 3, 4, 5, 6])
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
    parser.add_argument(
        "--a-load-path",
        type=str,
        default="tdm",
        choices=["tdm", "vgpr", "vgpr_ascale"],
    )
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode", action="store_false", default=True)
    parser.add_argument("--b-streaming", action="store_true", default=False)
    parser.add_argument(
        "--b-split-load",
        action="store_true",
        default=False,
        help="Split the B TDM load by N groups. With --a-load-path vgpr this reuses "
        "wave0 for B half 0; with TDM A/scale it uses wave0=A, wave1/2=B halves, "
        "wave3=A_scale+B_scale.",
    )
    parser.add_argument(
        "--atomic-barrier-enable",
        action="store_true",
        default=False,
        help="Enable TDM atomic_barrier_enable (hardware auto-barrier)",
    )

    parser.add_argument(
        "--benchmark", action="store_true", default=False, help="Run benchmark mode (timing only, no correctness check)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="With --benchmark, also run the correctness check before timing. "
        "Without --benchmark, runs always verify and this flag is a no-op.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--no-flush-l2",
        action="store_true",
        default=False,
        help="Disable L2 defeat for a hot-cache measurement. Applies to both eager " "and --use-graph modes.",
    )
    parser.add_argument(
        "--l2-flush-mb",
        type=int,
        default=256,
        help="Scratch buffer size in MiB for eager cold-cache timing, and the "
        "minimum address-rotation target for --use-graph rotating-buffer timing.",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        default=False,
        help="Time via hipGraph capture+replay to strip host launch overhead from "
        "per-launch latency. By default this captures a rotating-buffer graph to "
        "avoid replaying the same tensor addresses; compare with --no-flush-l2 to "
        "separate address-reuse/cache effects from launch overhead.",
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

    if args.scale_mode == "ptpc" and args.verify_graph:
        raise SystemExit("--scale-mode ptpc does not support --verify-graph")
    if args.scale_mode == "ptpc" and args.use_graph and not args.benchmark:
        raise SystemExit("--scale-mode ptpc does not support --use-graph for functional tests (use --benchmark)")

    def _run_correctness_test():
        """Run the functional test (computes a reference and asserts correctness)."""
        if args.scale_mode == "ptpc":
            _run_ptpc_gemm_test(
                args.M,
                args.N,
                args.K,
                args.tile_m,
                args.tile_n,
                args.tile_k,
                args.m_warp,
                args.n_warp,
                num_buffers=args.num_buffers,
                out_dtype=args.out_dtype,
                data_format=args.data_format,
                l2_prefetch_distance=args.l2_prefetch_distance,
                cluster_m=args.cluster_m,
                cluster_n=args.cluster_n,
                split_k=args.split_k,
            )
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
                a_load_path=args.a_load_path,
                b_split_load=args.b_split_load,
                use_graph=args.use_graph,
            )

    if args.verify_graph:
        _run_graph_verify(args)
        if not args.benchmark:
            sys.exit(0)
    if args.benchmark:
        # Benchmark defaults to timing-only; --verify opts into a correctness check first.
        if args.verify:
            print("Verifying correctness before benchmark (--verify)...")
            _run_correctness_test()
        _run_benchmark(args)
    else:
        # Non-benchmark runs always verify.
        _run_correctness_test()
