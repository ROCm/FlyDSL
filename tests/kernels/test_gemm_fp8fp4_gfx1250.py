#!/usr/bin/env python3
"""MXFP4/MXFP8/A8W4 and PTPC-FP8 GEMM correctness tests for gfx1250.

Kernel implementation: kernels/gemm_fp8fp4_gfx1250.py
"""

import math
import os
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
)
from tests.kernels.utils import fp4_utils  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


SCALE_BLOCK = 32
_DT = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}


def preshuffle_scale(scale: torch.Tensor, *, inactive_fill: int = 0) -> torch.Tensor:
    """32x4 scale layout (A or B): [R, Ks] -> [ceil(R/32), K] (Ks = K//32).

    out[r_o, k_o, r_i, k_i] = scale[r_o*32 + r_i, k_o*4 + k_i]
    """
    R, Ks = scale.shape
    assert Ks % 4 == 0, f"preshuffle_scale needs Ks%4==0; got R={R} Ks={Ks}"
    R_blocks = (R + 31) // 32
    if R_blocks * 32 != R:
        storage = torch.full((R_blocks * 32, Ks), inactive_fill, dtype=scale.dtype, device=scale.device)
        storage[:R, :] = scale
        scale = storage
        R = R_blocks * 32
    x = scale.view(R // 32, 32, Ks // 4, 4).permute(0, 2, 1, 3).contiguous()  # [R//32, Ks//4, 32, 4]
    return x.reshape(R // 32, -1)  # [R//32, K]


def _select_ascale_load_path(M: int) -> str:
    return "vgpr" if M < 32 else "shuffled_tdm"


def _prepare_a_scale_for_path(a_scale: torch.Tensor, ascale_load_path: str) -> torch.Tensor:
    if ascale_load_path == "vgpr":
        return a_scale
    if ascale_load_path == "shuffled_tdm":
        return preshuffle_scale(a_scale)
    raise ValueError(f"unsupported ascale_load_path={ascale_load_path!r}")


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


def _random_ab_inputs(M: int, N: int, K: int, data_format: str):
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
    return a, b


def _random_mxscale_inputs(M: int, N: int, K: int, data_format: str):
    a, b = _random_ab_inputs(M, N, K, data_format)
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


def _reference_gemm(scale_mode: str, data_format: str, a, b, a_scale, b_scale, M, N, K):
    if scale_mode == "ptpc":
        return reference_ptpc_gemm(data_format, a, b, a_scale, b_scale, M, N, K)
    if data_format == "a8w4":
        return _reference_scaled_gemm(
            a, b, a_scale, b_scale, M, N, K, fp4_utils.fp8_e4m3_to_f32, convert_fn_b=fp4_utils.mxfp4_to_f32
        )
    if data_format == "fp4":
        return _reference_scaled_gemm(a, b, a_scale, b_scale, M, N, K, fp4_utils.mxfp4_to_f32)
    if data_format == "fp8":
        return _reference_scaled_gemm(a, b, a_scale, b_scale, M, N, K, fp4_utils.fp8_e4m3_to_f32)
    raise ValueError(f"unsupported data_format={data_format!r}")


def _format_gemm_name(scale_mode: str, data_format: str) -> str:
    if scale_mode == "ptpc":
        return "PTPC-A8W4" if data_format == "a8w4" else "PTPC-FP8"
    return "A8W4" if data_format == "a8w4" else ("MXFP4" if data_format == "fp4" else "MXFP8")


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


def _pack_factors(data_format: str) -> tuple[int, int]:
    if data_format == "fp4":
        return 2, 2
    if data_format == "a8w4":
        return 1, 2
    if data_format == "fp8":
        return 1, 1
    raise ValueError(f"unsupported data_format={data_format!r}")


def _get_problem_shape(
    data_format: str,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
) -> dict[str, int]:
    """Validate tile alignment and return the actual kernel dimensions.

    N/K must divide their tiles; M is ragged (hardware OOB). Fail loudly instead
    of silently host-padding.
    """
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n} (no silent pad)")
    if K % (tile_k * split_k) != 0:
        raise ValueError(f"K={K} must be divisible by tile_k*split_k={tile_k * split_k} (no silent pad)")

    pack_a, pack_b = _pack_factors(data_format)
    return {
        "M": M,
        "N": N,
        "K": K,
        "K_scale": K // SCALE_BLOCK,
        "pack_a": pack_a,
        "pack_b": pack_b,
    }


def _expect_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]):
    assert tensor.shape == shape, f"{name}.shape={tuple(tensor.shape)} expected {shape}"


def _validate_mxscale_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    problem_shape: dict[str, int],
):
    """Validate the no-host-padding mxscale input contract."""
    _expect_shape("A", a, (problem_shape["M"], problem_shape["K"] // problem_shape["pack_a"]))
    _expect_shape("B", b, (problem_shape["N"], problem_shape["K"] // problem_shape["pack_b"]))
    _expect_shape("A scale", a_scale, (problem_shape["M"], problem_shape["K_scale"]))
    _expect_shape("B scale", b_scale, (problem_shape["N"], problem_shape["K_scale"]))


def _validate_ab_inputs(a: torch.Tensor, b: torch.Tensor, problem_shape: dict[str, int]):
    _expect_shape("A", a, (problem_shape["M"], problem_shape["K"] // problem_shape["pack_a"]))
    _expect_shape("B", b, (problem_shape["N"], problem_shape["K"] // problem_shape["pack_b"]))


def _with_strided_a(a: torch.Tensor, problem_shape: dict[str, int], lda: int) -> torch.Tensor:
    """Return A backed by runtime lda when lda exceeds logical K."""
    pack_a = problem_shape["pack_a"]
    kernel_k = problem_shape["K"]
    if lda % pack_a != 0:
        raise ValueError(f"lda={lda} must be divisible by A pack factor {pack_a}")
    _expect_shape("A", a, (problem_shape["M"], kernel_k // pack_a))
    if lda == kernel_k:
        return a
    a_strided = torch.zeros(problem_shape["M"], lda // pack_a, dtype=a.dtype, device=a.device)
    a_strided[:, : kernel_k // pack_a] = a
    return a_strided


def _run_gemm_test(
    scale_mode,
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
    out_dtype,
    *,
    l2_prefetch_distance=0,
    cluster_m=1,
    cluster_n=1,
    inst_prefetch=False,
    waves_per_eu=None,
    expert_sched_mode=True,
    split_k=1,
    ascale_load_path=None,
    lda_extra=0,
    ldc_extra=0,
    return_launch_fn=False,
):
    """Shared correctness body for mxscale and PTPC GEMM variants."""
    if scale_mode not in ("mxscale", "ptpc"):
        raise ValueError(f"unsupported scale_mode={scale_mode!r}")
    if scale_mode == "ptpc" and data_format not in ("fp8", "a8w4"):
        raise ValueError(f"scale_mode='ptpc' only supports data_format='fp8' or 'a8w4', got {data_format!r}")

    is_mxscale = scale_mode == "mxscale"
    is_ptpc = scale_mode == "ptpc"
    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"

    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"{scale_mode} GEMM requires gfx1250, got {arch}")

    if K % SCALE_BLOCK != 0:
        pytest.skip(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")

    problem_shape = _get_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, split_k)
    kernel_m = problem_shape["M"]
    kernel_n = problem_shape["N"]
    kernel_k = problem_shape["K"]
    pack_b = problem_shape["pack_b"]
    local_k = kernel_k // split_k

    if is_mxscale and ascale_load_path is None:
        ascale_load_path = _select_ascale_load_path(M)
    tdm_store_enabled = split_k == 1

    num_k_tiles = local_k // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    # FP8/A8W4 256x256 + f32 + TDM store exceeds LDS.
    if is_mxscale and not is_fp4 and tile_m == 256 and tile_n == 256 and out_dtype == "f32" and tdm_store_enabled:
        pytest.skip("256x256 tile with f32 TDM store exceeds LDS limit")

    torch_out_dtype = _DT[out_dtype]

    # Split-K accumulates at the output precision.
    kernel_out_dtype = out_dtype
    torch_kernel_dtype = _DT[kernel_out_dtype]

    torch.manual_seed(0)
    if is_mxscale:
        a, b, a_scale, b_scale = _random_mxscale_inputs(M, N, K, data_format)
        a_scale_raw = a_scale.clone()
        b_scale_raw = b_scale.clone()
    else:
        a, b = _random_ab_inputs(M, N, K, data_format)
        a_scale = (0.5 + torch.rand(M, dtype=torch.float32)).contiguous()
        b_scale = (0.5 + torch.rand(N, dtype=torch.float32)).contiguous()
        a_scale_raw = None
        b_scale_raw = None

    ref = _reference_gemm(scale_mode, data_format, a, b, a_scale, b_scale, M, N, K)

    fmt_name = _format_gemm_name(scale_mode, data_format)
    run_attrs = []
    if cluster_m > 1 or cluster_n > 1:
        run_attrs.append(f"cluster=({cluster_m},{cluster_n})")
    if split_k > 1:
        run_attrs.append(f"split_k={split_k}")
    if is_mxscale:
        run_attrs.append("tdm_store" if tdm_store_enabled else "buffer_store")
        run_attrs.append(f"ascale={ascale_load_path}")
    if lda_extra:
        run_attrs.append(f"lda={kernel_k + lda_extra}")
    if ldc_extra:
        run_attrs.append(f"ldc={kernel_n + ldc_extra}")
    run_attrs.append("preshuffle")
    attr_str = ", " + ", ".join(run_attrs) if run_attrs else ""
    print(
        f"\nRunning {fmt_name} GEMM: M={M}, N={N}, K={K}, "
        f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}{attr_str}, out={out_dtype}"
    )
    print(f"Ref stats: min={ref.min():.2f}, max={ref.max():.2f}, mean={ref.mean():.2f}, std={ref.std():.2f}")

    lda = kernel_k + lda_extra
    ldc = kernel_n + ldc_extra

    if is_mxscale:
        _validate_mxscale_inputs(a, b, a_scale, b_scale, problem_shape)
        a = _with_strided_a(a, problem_shape, lda)
        a_scale = _prepare_a_scale_for_path(a_scale, ascale_load_path)
        b_scale = preshuffle_scale(b_scale)
    else:
        _validate_ab_inputs(a, b, problem_shape)
        _expect_shape("A scale", a_scale, (kernel_m,))
        _expect_shape("B scale", b_scale, (kernel_n,))
        a = _with_strided_a(a, problem_shape, lda)

    b = fp4_utils.preshuffle_b_16x16(b, kernel_n, kernel_k // pack_b)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(kernel_m, ldc, dtype=torch_kernel_dtype, device="cuda")

    compile_kwargs = {
        "data_format": data_format,
        "N": kernel_n,
        "K": kernel_k,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "m_warp": m_warp,
        "n_warp": n_warp,
        "num_buffers": num_buffers,
        "waves_per_eu": waves_per_eu,
        "l2_prefetch_distance": l2_prefetch_distance,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "out_dtype": kernel_out_dtype,
        "inst_prefetch": inst_prefetch,
        "split_k": split_k,
        "expert_sched_mode": expert_sched_mode,
    }
    if is_mxscale:
        compile_kwargs["ascale_load_path"] = ascale_load_path
        launch_fn = compile_mxscale_gemm(**compile_kwargs)
    else:
        launch_fn = compile_ptpc_gemm(**compile_kwargs)

    # Keep 2D: dynamic_layout=True packs shape as i32; flattening overflows for M*K >= 2^31.
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
        kernel_m,
        kernel_n,
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

    if out_dtype in ("bf16", "f16"):
        c_out_f = c_out.float()
        ref_f = ref.to(torch_out_dtype).float()
    else:
        c_out_f = c_out.float()
        ref_f = ref.float()

    diff = (c_out_f - ref_f).abs()
    print(f"Abs diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

    # Compute cosine in float64: large scaled outputs can overflow float32's
    # accurate-summation range, while pass/fail is gated by assert_close below.
    cos_sim = torch.nn.functional.cosine_similarity(
        c_out_f.flatten().unsqueeze(0).double(), ref_f.flatten().unsqueeze(0).double()
    ).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    if is_ptpc:
        peak = float(ref_f.abs().max())
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=2e-2, atol=max(5e-2, 2e-2 * peak))
        else:
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=max(1e-2, K * 0.6))
    elif is_fp4:
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=1e-2)
        else:
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-5, atol=1e-8)
    elif is_a8w4:
        rtol, atol, tol_diag = _a8w4_tolerances(a_scale_raw, b_scale_raw, K, out_dtype)
        print(tol_diag)
        torch.testing.assert_close(c_out_f, ref_f, rtol=rtol, atol=atol)
    elif out_dtype in ("bf16", "f16"):
        # Split-K atomic-adds at output precision; peak-scale tolerance absorbs
        # compounded bf16/f16 rounding on large-magnitude outputs.
        if split_k > 1:
            peak = float(ref_f.abs().max())
            torch.testing.assert_close(c_out_f, ref_f, rtol=2e-2, atol=max(5e-2, 2e-2 * peak))
        else:
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-2, atol=5e-2)
    else:
        torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=max(1e-2, K * 0.6))

    print("PASSED")
    if return_launch_fn:
        return launch_fn


# ── pytest parametrized tests ──


def _gen_mxfp4_gemm_configs():
    # (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    base = [
        (128, 512, 7168, 128, 128, 256, 2, 2),
        (128, 7168, 256, 128, 256, 128, 2, 2),
        (128, 4096, 7168, 128, 256, 256, 2, 2),
        (128, 7168, 2048, 128, 256, 256, 2, 2),
        (1024, 1024, 1024, 256, 256, 256, 2, 2),
    ]
    return [(*shape, num_buffers) for shape in base for num_buffers in (2, 3, 4)]


def _gen_mxfp8_gemm_configs():
    # (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    base = [
        (128, 256, 256, 128, 256, 128, 2, 4),
        (256, 256, 256, 256, 256, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ]
    cfgs = [(*shape, num_buffers) for shape in base for num_buffers in (2, 3)]
    cfgs.append((256, 256, 512, 256, 256, 128, 2, 2, 4))
    return cfgs


def _gen_a8w4_gemm_configs():
    # (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    base = [
        (128, 5632, 2816, 128, 256, 256, 2, 2),
        (128, 2816, 2816, 128, 256, 256, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ]
    cfgs = [(*shape, num_buffers) for shape in base for num_buffers in (2, 3)]
    cfgs.append((256, 256, 512, 256, 256, 128, 2, 2, 4))
    return cfgs


def _gen_mxscale_gemm_configs():
    cfgs = []
    for data_format, gen in (
        ("fp4", _gen_mxfp4_gemm_configs),
        ("fp8", _gen_mxfp8_gemm_configs),
        ("a8w4", _gen_a8w4_gemm_configs),
    ):
        cfgs += [(data_format, *cfg) for cfg in gen()]
    return cfgs


def test_mxscale_compile_auto_selects_splitk_store_path():
    """Direct compile API should not require a store-path override for split-K."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA_SCALE requires gfx1250, got {arch}")

    launch_fn = compile_mxscale_gemm(
        data_format="fp8",
        N=256,
        K=2048,
        tile_m=128,
        tile_n=256,
        tile_k=128,
        m_warp=2,
        n_warp=4,
        num_buffers=2,
        l2_prefetch_distance=2,
        out_dtype="bf16",
        split_k=2,
        ascale_load_path="shuffled_tdm",
    )
    assert callable(launch_fn)


@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers",
    _gen_mxscale_gemm_configs(),
)
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxscale_gemm(
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
    out_dtype,
):
    _run_gemm_test(
        "mxscale",
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
        out_dtype,
        l2_prefetch_distance=2 if data_format in ("fp8", "a8w4") else 0,
    )


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp8_gemm_splitk(split_k, out_dtype):
    """FP8 split-K: split_k workgroups accumulate partial K-sums into C via atomic add.

    Exercises the auto-selected atomic epilogue path. K=2048/tile_k=128 gives
    every split_k value >= 2 local K-tiles (needed for double buffering).
    """
    _run_gemm_test(
        "mxscale",
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
        out_dtype=out_dtype,
        l2_prefetch_distance=2,
        split_k=split_k,
    )


# ── Tile-independent 32x4 B-scale coverage ──
# tile_m=16, m_warp=1 -> wmma_m_rep=1 (odd) -> the default row-major streaming
# schedule, exercising the 32x4 B-scale path. The sweep covers every
# tile_n/n_warp that maps to a distinct read shape (b32/b64/b128 per_load and
# group counts 1/2/4 and the non-power-of-2 group count 3 that exercises the
# TDM warp-distribution power-of-two padding), both data formats, k_wmma_steps
# 1/2/4, wave-spec on/off, f32/bf16, multi-buffer, and ragged/decode M.
_BS32_N_FOR_TN = {32: 128, 64: 128, 128: 256, 192: 384, 256: 512}
_BS32_TN_NW = [
    (32, 2),
    (64, 2),
    (64, 4),
    (128, 2),
    (128, 4),
    (192, 2),
    (192, 4),
    (256, 2),
    (256, 4),
]  # fmt: skip  (n_warp>=2: wave-specialized TDM requires >=2 waves)


def _gen_bs32_configs():
    cfgs, seen = [], set()

    def add(fmt, M, tile_n, n_warp, tile_k, nbuf, od):
        N = _BS32_N_FOR_TN[tile_n]
        K = tile_k * max(nbuf, 2)  # >= nbuf K-tiles for double/triple buffering
        key = (fmt, M, N, K, tile_n, tile_k, n_warp, nbuf, od)
        if key not in seen:
            seen.add(key)
            cfgs.append(key)

    for fmt in ("fp8", "a8w4"):
        # 1) full tile_n x n_warp shape sweep (all rep/group/per_load cases).
        for tn, nw in _BS32_TN_NW:
            add(fmt, 16, tn, nw, 256, 2, "bf16")
        # 2) M=1 decode-like. The real decode shape (tile_n=64) uses deep K + 4 buffers.
        add(fmt, 1, 64, 4, 512, 4, "bf16")
        for tn in (128, 192, 256):
            add(fmt, 1, tn, 4, 256, 2, "bf16")
        # 3) k_wmma_steps 1/2/4 on the next_pow2 (192) and clean (256/64) shapes.
        for tn, nw in [(192, 4), (256, 4), (64, 4)]:
            for tk in (128, 512):
                add(fmt, 16, tn, nw, tk, 2, "bf16")
        # 4) f32 + triple buffering on a few shapes.
        for tn, nw in [(192, 4), (128, 2), (32, 2)]:
            add(fmt, 16, tn, nw, 256, 3, "f32")
        # 5) ragged / decode / OOB M.
        for M in (1, 13, 33):
            add(fmt, M, 256, 4, 256, 2, "bf16")
    return cfgs


@pytest.mark.parametrize("data_format, M, N, K, tile_n, tile_k, n_warp, num_buffers, out_dtype", _gen_bs32_configs())
def test_mxscale_bscale_32x4(data_format, M, N, K, tile_n, tile_k, n_warp, num_buffers, out_dtype):
    _run_gemm_test(
        "mxscale",
        data_format,
        M,
        N,
        K,
        16,
        tile_n,
        tile_k,
        1,
        n_warp,
        num_buffers,
        out_dtype=out_dtype,
        l2_prefetch_distance=0,
    )


def _gen_ascale_32x4_configs():
    # (fmt, M, tile_m, tile_n, tile_k, m_warp, n_warp, nbuf) for the A-scale
    # 32x4 TDM path. Covers wave
    # counts 2/3/4, A-scale M op_sel via tile_m (rep 1/2/4/8/16), tile_k (k_steps
    # 1/2/4), multi-64 tile_n, and ragged M. tile_k kept small at large tile_m so
    # LDS fits. All cases use M>=32; small-M coverage stays on the VGPR path.
    cfgs = []
    for fmt in ("fp8", "a8w4"):
        # 4-wave (n_warp=4, tile_n=64 -> rep_n=1 row-major): rep_m sweep via tile_m.
        cfgs += [
            (fmt, 32, 16, 64, 512, 1, 4, 2),  # rep1, k_steps=4
            (fmt, 32, 32, 64, 512, 1, 4, 2),  # rep2 (op_sel)
            (fmt, 64, 64, 64, 256, 1, 4, 2),  # rep4 (op_sel)
            (fmt, 128, 128, 64, 256, 1, 4, 2),  # rep8 (op_sel)
            (fmt, 256, 256, 64, 128, 1, 4, 2),  # rep16 (op_sel)
            (fmt, 32, 16, 64, 128, 1, 4, 2),  # k_steps=1
            (fmt, 32, 16, 64, 256, 1, 4, 2),  # k_steps=2
            (fmt, 32, 16, 128, 256, 1, 4, 2),  # tile_n=128
            (fmt, 32, 16, 192, 256, 1, 4, 2),  # tile_n=192 (next_pow2)
            (fmt, 32, 16, 256, 256, 1, 4, 2),  # tile_n=256
        ]
        # 2-wave (wave0 issues A-data + B-scale, wave1 issues B-data + A-scale).
        cfgs += [(fmt, 32, 16, 64, 512, 1, 2, 2), (fmt, 32, 32, 64, 512, 1, 2, 2)]
        # 3-wave keeps B-scale as wave0 secondary while wave2 issues A-scale.
        cfgs += [(fmt, 32, 16, 192, 256, 1, 3, 2)]
        # ragged / OOB M.
        cfgs += [(fmt, 33, 16, 64, 512, 1, 4, 2), (fmt, 65, 64, 64, 256, 1, 4, 2)]
    return cfgs


@pytest.mark.parametrize("data_format, M, tile_m, tile_n, tile_k, m_warp, n_warp, nbuf", _gen_ascale_32x4_configs())
def test_mxscale_ascale_32x4(data_format, M, tile_m, tile_n, tile_k, m_warp, n_warp, nbuf):
    N = 2 * tile_n
    K = tile_k * nbuf
    _run_gemm_test(
        "mxscale",
        data_format,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        nbuf,
        out_dtype="bf16",
        l2_prefetch_distance=0,
        ascale_load_path="shuffled_tdm",
    )


@pytest.mark.parametrize("data_format", ["fp8", "a8w4", "fp4"])
@pytest.mark.parametrize("M", [1, 13, 31])
def test_mxscale_ascale_vgpr_small_m(data_format, M):
    _run_gemm_test(
        "mxscale",
        data_format,
        M,
        128,
        512,
        16,
        64,
        256,
        1,
        2,
        2,
        out_dtype="bf16",
        l2_prefetch_distance=0,
        ascale_load_path="vgpr",
    )


@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, nbuf",
    [
        ("fp8", 32, 128, 512, 32, 64, 256, 1, 2, 2),  # row-major, M>=32
        ("fp8", 33, 128, 512, 64, 64, 256, 1, 2, 2),  # row-major ragged M>=32
        ("fp8", 128, 512, 512, 128, 256, 256, 2, 2, 2),  # quadrant
        ("a8w4", 128, 512, 512, 128, 256, 256, 2, 2, 2),  # quadrant
        ("fp8", 256, 256, 512, 256, 256, 128, 2, 2, 4),  # deep-pipeline
        ("fp4", 128, 256, 512, 128, 128, 256, 2, 2, 2),  # FP4 quadrant
    ],
)
def test_mxscale_ascale_vgpr_general(data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, nbuf):
    _run_gemm_test(
        "mxscale",
        data_format,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        nbuf,
        out_dtype="bf16",
        l2_prefetch_distance=0,
        ascale_load_path="vgpr",
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
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp4_gemm_mcast(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n, num_buffers, out_dtype
):
    _run_gemm_test(
        "mxscale",
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

    # Build inputs (mirrors _run_gemm_test("mxscale", ...), but no padding needed
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

    ascale_load_path = _select_ascale_load_path(M)
    a_scale_ps = _prepare_a_scale_for_path(a_scale, ascale_load_path)
    b_scale_ps = preshuffle_scale(b_scale)
    pack_b = 2 if is_fp4 else 1
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
        out_dtype="bf16",
        split_k=1,
        ascale_load_path=ascale_load_path,
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


def _gen_ptpc_gemm_configs():
    # (data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    return [
        ("fp8", 256, 256, 512, 256, 256, 128, 2, 2, 4),  # deep-pipeline eligible
        ("fp8", 128, 256, 512, 128, 256, 128, 2, 2, 4),  # quadrant fallback
        ("a8w4", 128, 256, 512, 128, 256, 128, 2, 4, 2),  # row-major + wave-spec TDM
        ("a8w4", 128, 256, 1024, 128, 256, 256, 2, 4, 3),
    ]


@pytest.mark.parametrize("out_dtype", ["bf16", "f32"])
@pytest.mark.parametrize(
    "data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers",
    _gen_ptpc_gemm_configs(),
)
def test_ptpc_gemm(data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype):
    _run_gemm_test(
        "ptpc",
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
        out_dtype,
    )


@pytest.mark.parametrize("scale_mode, data_format", [("ptpc", "fp8"), ("mxscale", "fp8"), ("mxscale", "fp4")])
@pytest.mark.parametrize("lda_extra, ldc_extra", [(128, 0), (0, 256), (128, 256)])
def test_gemm_strided(scale_mode, data_format, lda_extra, ldc_extra):
    """Strided A/C: data backed by a wider leading dim, passed via runtime lda/ldc."""
    _run_gemm_test(
        scale_mode,
        data_format,
        128,
        256,
        512,
        128,
        256,
        128,
        2,
        2,
        num_buffers=4,
        out_dtype="bf16",
        lda_extra=lda_extra,
        ldc_extra=ldc_extra,
    )


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("data_format, out_dtype", [("fp8", "bf16"), ("fp8", "f32"), ("a8w4", "bf16")])
def test_ptpc_gemm_splitk(data_format, split_k, out_dtype):
    """PTPC split-K: each chunk applies sa*sb then atomic-adds; sum stays correct."""
    _run_gemm_test(
        "ptpc",
        data_format,
        128,
        256,
        2048,
        128,
        256,
        128,
        2,
        4,
        num_buffers=2,
        out_dtype=out_dtype,
        split_k=split_k,
    )


# ---------------------------------------------------------------------------
# Non-tile-aligned M (the default, no host M-padding): A/C (and ptpc sa) are
# allocated at the real M. A-load TDM skips rows>=M, sa buffer_load OOB->0, C
# buffer_store clips via num_records. N,K stay tile-aligned.
# ---------------------------------------------------------------------------
_RAGGED_M_VALUES = [
    1,
    16,
    31,
    64,
    65,
    100,
    127,
    128,
    129,
    130,
    192,
    255,
    256,
    257,
    384,
    500,
    1000,
    2048,
]
_RAGGED_M_BASE_CONFIGS = [
    ("ptpc", "fp8", "bf16"),
    ("ptpc", "fp8", "f32"),
    ("ptpc", "a8w4", "bf16"),
    ("mxscale", "fp8", "bf16"),
    ("mxscale", "fp8", "f32"),
]


@pytest.mark.parametrize("scale_mode, data_format, out_dtype", _RAGGED_M_BASE_CONFIGS)
@pytest.mark.parametrize("M", _RAGGED_M_VALUES)
def test_gemm_ragged_m(M, scale_mode, data_format, out_dtype):
    n_warp = 4 if data_format == "a8w4" else 2
    num_buffers = 2 if data_format == "a8w4" else 4
    _run_gemm_test(
        scale_mode,
        data_format,
        M,
        256,
        512,
        128,
        128,
        128,
        2,
        n_warp,
        num_buffers,
        out_dtype,
    )


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("M", [1, 64, 129, 192, 257, 500])
def test_ptpc_fp8_gemm_splitk_ragged_m(M, split_k):
    # split_k atomic output predicated per-lane on row < M (auto buffer/atomic path).
    _run_gemm_test(
        "ptpc",
        "fp8",
        M,
        256,
        2048,
        128,
        256,
        128,
        2,
        4,
        num_buffers=2,
        out_dtype="bf16",
        split_k=split_k,
    )


# Tile/warp-config diversity: the per-warp partial-tile clip uses
# warp_tile_m = tile_m // m_warp, so M must be exercised against different warp
# boundaries. Existing ragged-M tests are all m_warp=2 (warp_tile_m=64); these add
# warp_tile_m in {128 (single M-warp / tile_m=256), 32 (fine 4-way split)}.
_RAGGED_M_WARP_CONFIGS = [
    # (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    (128, 128, 128, 1, 4, 4),  # warp_tile_m=128: single M-warp, no M split
    (128, 128, 128, 4, 2, 2),  # warp_tile_m=32: fine-grained M warps
    (256, 128, 128, 2, 2, 2),  # tile_m=256, warp_tile_m=128
]
# Boundary-diverse M for warp_tile_m in {32, 128}: partial/full/OOB warps + aligned.
_RAGGED_M_WARP_VALUES = [1, 33, 64, 100, 129, 200, 256, 333]


@pytest.mark.parametrize("tile_m,tile_n,tile_k,m_warp,n_warp,num_buffers", _RAGGED_M_WARP_CONFIGS)
@pytest.mark.parametrize("M", _RAGGED_M_WARP_VALUES)
def test_ptpc_fp8_gemm_ragged_m_warps(M, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
    _run_gemm_test(
        "ptpc",
        "fp8",
        M,
        256,
        512,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers=num_buffers,
        out_dtype="bf16",
    )


#   M=100 -> grid_m 1->2, tile1 fully OOB (rows>=100) under M-multicast
#   M=129,200,450 -> partial last M-tile, grid divisible
#   M=256,512 -> tile-aligned
#   M=257,300 -> grid_m 3->4 (rounded); M=300 also makes tile3 fully OOB
_RAGGED_M_CLUSTER_VALUES = [100, 129, 200, 256, 257, 300, 450, 512]
_RAGGED_M_CLUSTERS = [(2, 2), (2, 4)]
_RAGGED_M_CLUSTER_CONFIGS = [
    ("ptpc", "fp8"),
    ("ptpc", "a8w4"),
    ("mxscale", "fp8"),
]
_RAGGED_M_CLUSTER_TM256_VALUES = [100, 300, 512, 600, 700, 1024]


@pytest.mark.parametrize("scale_mode, data_format", _RAGGED_M_CLUSTER_CONFIGS)
@pytest.mark.parametrize("cluster_m,cluster_n", _RAGGED_M_CLUSTERS)
@pytest.mark.parametrize("M", _RAGGED_M_CLUSTER_VALUES)
def test_gemm_ragged_m_cluster(M, cluster_m, cluster_n, scale_mode, data_format):
    n_warp = 4 if data_format == "a8w4" else 2
    _run_gemm_test(
        scale_mode,
        data_format,
        M,
        512,
        512,
        128,
        128,
        128,
        2,
        n_warp,
        num_buffers=2,
        out_dtype="bf16",
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )


@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("M", [100, 129, 256, 300, 450])
def test_ptpc_fp8_gemm_splitk_ragged_m_cluster(M, split_k):
    # split_k atomic output (per-lane row<M predicate) combined with cluster>1.
    _run_gemm_test(
        "ptpc",
        "fp8",
        M,
        512,
        2048,
        128,
        128,
        128,
        2,
        2,
        num_buffers=2,
        out_dtype="bf16",
        split_k=split_k,
        cluster_m=2,
        cluster_n=2,
    )


@pytest.mark.parametrize("scale_mode", ["ptpc", "mxscale"])
@pytest.mark.parametrize("cluster_m,cluster_n", _RAGGED_M_CLUSTERS)
@pytest.mark.parametrize("M", _RAGGED_M_CLUSTER_TM256_VALUES)
def test_gemm_ragged_m_cluster_tm256(M, cluster_m, cluster_n, scale_mode):
    _run_gemm_test(
        scale_mode,
        "fp8",
        M,
        1024,
        512,
        256,
        256,
        128,
        2,
        2,
        num_buffers=2,
        out_dtype="bf16",
        cluster_m=cluster_m,
        cluster_n=cluster_n,
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

    problem_shape = _get_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, args.split_k)
    kernel_m = problem_shape["M"]
    kernel_n = problem_shape["N"]
    kernel_k = problem_shape["K"]
    PACK_A = problem_shape["pack_a"]
    PACK_B = problem_shape["pack_b"]

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    is_ptpc = getattr(args, "scale_mode", "mxscale") == "ptpc"
    if is_ptpc and data_format not in ("fp8", "a8w4"):
        raise ValueError(f"scale_mode='ptpc' only supports data_format='fp8' or 'a8w4', got {data_format!r}")
    # split_k atomic-adds at output precision (bf16/f16).
    kernel_out_dtype = args.out_dtype
    torch_kernel_dtype = _DT[kernel_out_dtype]
    elem_bytes_d = 2 if kernel_out_dtype in ("bf16", "f16") else 4
    if is_ptpc:
        fmt_name = "PTPC-A8W4" if is_a8w4 else "PTPC-FP8"
    else:
        fmt_name = "A8W4" if is_a8w4 else ("MXFP4" if is_fp4 else "MXFP8")

    print("=" * 72)
    print(f"  {fmt_name} GEMM Benchmark on gfx1250")
    print(f"  PyTorch {torch.__version__}, Device: {torch.cuda.get_device_name(0)}")
    print(f"  Shape: M={M}, N={N}, K={K}")
    print(f"  Tile: ({tile_m}, {tile_n}, {tile_k}), warps=({args.m_warp}x{args.n_warp})")
    print(f"  Buffers={args.num_buffers}, out={args.out_dtype}, inst_prefetch={args.inst_prefetch}")
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
    print("=" * 72)

    torch.manual_seed(0)
    warp_tile_m = tile_m // args.m_warp
    warp_tile_n = tile_n // args.n_warp
    ascale_load_path = _select_ascale_load_path(M)
    if is_ptpc:
        # PTPC: fp8 A with fp32 per-token (sa[M]) / per-channel (sb[N]) scales, no scale preshuffle.
        # B is fp8 (data_format="fp8") or FP4-packed 2-per-byte (data_format="a8w4").
        K_packed_b = kernel_k // PACK_B
        b_kind = "fp4 (a8w4)" if is_a8w4 else "fp8"
        fill_spec = _parse_fill_mode(getattr(args, "fill_mode", "random"))
        if fill_spec[0] == "const":
            value = fill_spec[1]
            fp8_byte = _fp8_e4m3fn_byte(value)
            a_raw = torch.full((M, K), fp8_byte, dtype=torch.uint8)
            b_raw = _fp4_e2m1_packed_fill(N, K, value) if is_a8w4 else torch.full((N, K), fp8_byte, dtype=torch.uint8)
            # Neutral per-token/per-channel scales so the const output stays predictable.
            a_scale = torch.ones(M, dtype=torch.float32)
            b_scale = torch.ones(N, dtype=torch.float32)
            if is_a8w4:
                eff_b = _nearest_mxfp4_value(value)
                b_note = f"fp4 B={eff_b:g}" + (f" (snapped from {value:g})" if eff_b != value else "")
            else:
                b_note = "fp8 B"
            print(f"  Fill mode: const={value:g} (FP8 byte=0x{fp8_byte:02x}), {b_note}, sa=sb=1.0")
        else:
            a_raw = random_fp8_data(M, K)
            b_raw = fp4_utils.random_fp4_packed(N, K) if is_a8w4 else random_fp8_data(N, K)
            a_scale = (0.5 + torch.rand(M, dtype=torch.float32)).contiguous()
            b_scale = (0.5 + torch.rand(N, dtype=torch.float32)).contiguous()
            print(f"  Fill mode: random fp8 A / {b_kind} B, fp32 per-token/per-channel scales")
        a = a_raw
        b = b_raw
        _validate_ab_inputs(a, b, problem_shape)
        _expect_shape("A scale", a_scale, (kernel_m,))
        _expect_shape("B scale", b_scale, (kernel_n,))
        b = fp4_utils.preshuffle_b_16x16(b, kernel_n, K_packed_b)
    else:
        a, b, a_scale, b_scale, fill_spec = _fill_mode_inputs(
            M, N, K, data_format, getattr(args, "fill_mode", "random")
        )
        print(f"  Fill mode: {_fill_mode_label(fill_spec, data_format)}")

        _validate_mxscale_inputs(a, b, a_scale, b_scale, problem_shape)
        a_scale = _prepare_a_scale_for_path(a_scale, ascale_load_path)
        b_scale = preshuffle_scale(b_scale)

        K_packed = kernel_k // PACK_B
        b = fp4_utils.preshuffle_b_16x16(b, kernel_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(kernel_m, kernel_n, dtype=torch_kernel_dtype, device="cuda")

    print("\n[1/3] Compiling kernel...")
    t0 = time.perf_counter()
    if is_ptpc:
        launch_fn = compile_ptpc_gemm(
            N=kernel_n,
            K=kernel_k,
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
            N=kernel_n,
            K=kernel_k,
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
            split_k=args.split_k,
            expert_sched_mode=args.expert_sched_mode,
            atomic_barrier_enable=args.atomic_barrier_enable,
            ascale_load_path=ascale_load_path,
        )

    compiled_exe = flyc.compile(
        launch_fn,
        c_gpu,
        a_gpu,
        b_gpu,
        as_gpu,
        bs_gpu,
        kernel_m,
        kernel_n,
        kernel_k,
        kernel_n,
        torch.cuda.current_stream(),
    )

    def run_one(c_, a_, b_, as_, bs_):
        compiled_exe(
            c_,
            a_,
            b_,
            as_,
            bs_,
            kernel_m,
            kernel_n,
            kernel_k,
            kernel_n,
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
    m_tiles = (kernel_m + tile_m - 1) // tile_m
    n_tiles = (kernel_n + tile_n - 1) // tile_n
    k_tiles = kernel_k // tile_k
    k_tiles_local = (kernel_k // args.split_k) // tile_k
    # Sequential WMMAs per workgroup (all k_tiles execute sequentially)
    seq_wmma = k_tiles_local * wmma_per_tile
    us_per_wmma = us / seq_wmma if seq_wmma > 0 else 0

    logical_flops = 2.0 * M * N * K
    tile_m_covered = m_tiles * tile_m
    tile_n_covered = n_tiles * tile_n
    tile_flops = 2.0 * tile_m_covered * tile_n_covered * kernel_k
    time_s = us / 1e6
    logical_tflops = logical_flops / time_s / 1e12 if time_s > 0 else 0.0
    tile_tflops = tile_flops / time_s / 1e12 if time_s > 0 else 0.0

    bytes_a = kernel_m * kernel_k // PACK_A
    bytes_b = kernel_n * kernel_k // PACK_B
    bytes_scale = (kernel_m + kernel_n) * (4 if is_ptpc else problem_shape["K_scale"])
    bytes_d = kernel_m * kernel_n * elem_bytes_d
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
    print(f"      Bandwidth:    {bw_gbs:.1f} GB/s  (read: {read_bw_gbs:.1f} + write: {write_bw_gbs:.1f})")
    print(
        f"      Bytes moved:  {bytes_moved / 1e6:.1f} MB  "
        f"(A={bytes_a / 1e6:.1f} B={bytes_b / 1e6:.1f} "
        f"scale={bytes_scale / 1e6:.1f} D={bytes_d / 1e6:.1f})"
    )
    print("      ---")
    print(f"      WMMA/tile:    {wmma_per_tile} ({wmma_m_rep}m × {wmma_n_rep}n × {k_wmma_steps}k)")
    if args.split_k > 1:
        print(
            f"      Total tiles:  {m_tiles}×{n_tiles} spatial × {args.split_k} split-K × {k_tiles_local} local K-iters"
        )
    else:
        print(f"      Total tiles:  {m_tiles}×{n_tiles} spatial × {k_tiles} K-iters")
    print(f"      Seq WMMA/WG:  {seq_wmma}")
    print(f"      us/WMMA:      {us_per_wmma:.1f}")
    if us_per_wmma > 1000:
        print(f"      WARNING: {us_per_wmma / 1000:.1f} ms/WMMA indicates WMMA_SCALE trap-handler emulation")
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

    problem_shape = _get_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, args.split_k)
    kernel_m = problem_shape["M"]
    kernel_n = problem_shape["N"]
    kernel_k = problem_shape["K"]

    print("=" * 72)
    print(f"  Graph functional verification ({data_format}) on gfx1250")
    print(f"  Shape: M={M}, N={N}, K={K}")
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

    _validate_mxscale_inputs(a, b, a_scale, b_scale, problem_shape)
    ascale_load_path = _select_ascale_load_path(M)
    a_scale = _prepare_a_scale_for_path(a_scale, ascale_load_path)
    b_scale = preshuffle_scale(b_scale)
    K_packed = kernel_k // problem_shape["pack_b"]
    b = fp4_utils.preshuffle_b_16x16(b, kernel_n, K_packed)

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    # split_k atomic-adds at output precision (bf16/f16).
    kernel_out_dtype = args.out_dtype
    c_gpu = torch.zeros(kernel_m, kernel_n, dtype=_DT[kernel_out_dtype], device="cuda")

    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        N=kernel_n,
        K=kernel_k,
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
        split_k=args.split_k,
        expert_sched_mode=args.expert_sched_mode,
        atomic_barrier_enable=args.atomic_barrier_enable,
        ascale_load_path=ascale_load_path,
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
        kernel_m,
        kernel_n,
        kernel_k,
        kernel_n,
        torch.cuda.current_stream(),
    )

    def launch():
        compiled_exe(
            c_flat,
            a_flat,
            b_flat,
            as_flat,
            bs_flat,
            kernel_m,
            kernel_n,
            kernel_k,
            kernel_n,
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
    parser.add_argument("--out-dtype", type=str, default="bf16", choices=["f32", "bf16", "f16"])
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode", action="store_false", default=True)
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
        help="Disable L2 defeat for a hot-cache measurement. Applies to both eager and --use-graph modes.",
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

    def _run_correctness_test():
        """Run the functional test (computes a reference and asserts correctness)."""
        if args.scale_mode == "ptpc":
            _run_gemm_test(
                "ptpc",
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
                out_dtype=args.out_dtype,
                l2_prefetch_distance=args.l2_prefetch_distance,
                cluster_m=args.cluster_m,
                cluster_n=args.cluster_n,
                split_k=args.split_k,
            )
        else:
            _run_gemm_test(
                "mxscale",
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
                out_dtype=args.out_dtype,
                split_k=args.split_k,
                l2_prefetch_distance=args.l2_prefetch_distance,
                cluster_m=args.cluster_m,
                cluster_n=args.cluster_n,
                inst_prefetch=args.inst_prefetch,
                waves_per_eu=args.waves_per_eu,
                expert_sched_mode=args.expert_sched_mode,
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
