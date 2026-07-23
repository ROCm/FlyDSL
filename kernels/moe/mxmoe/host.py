# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side launch glue for the fused a4w4 mxmoe kernels.

Ported (self-contained) from aiter's ``mxfp4_gemm1_kernels.py`` /
``mxfp4_gemm2_kernels.py``. Kernel launch args are raw device pointers
(``fx.Int64``); tensors are passed as ``.data_ptr()``.
"""

import functools

import torch

import flydsl.compiler as flyc
from kernels.moe.mxmoe.gemm1 import compile_gemm1_a4w4_port, gemm1_grid
from kernels.moe.mxmoe.gemm2 import compile_gemm2_a4w4_port

# gemm1 (BM, use_nt, inline_quant, a_dtype) variants the kernel supports.
# a_dtype="fp4" is a4w4 (mxfp4 A); "fp8" is a8w4 (fp8 e4m3 A x mxfp4 W1).
_G1_SUPPORTED = {
    (32, True, False, "fp4"),
    (32, False, False, "fp4"),
    (64, False, False, "fp4"),
    (128, False, False, "fp4"),
    (16, True, True, "fp4"),
    (32, True, False, "fp8"),
    (32, False, False, "fp8"),
    (64, False, False, "fp8"),
    (128, False, False, "fp8"),
    (16, True, True, "fp8"),
}

# gemm2 (BM, use_nt, epilog) variants the kernel supports.
_G2_SUPPORTED = {
    (16, False, "atomic"),
    (16, True, "atomic"),
    (32, False, "atomic"),
    (32, True, "atomic"),
    (64, False, "atomic"),
    (64, True, "atomic"),
    (128, False, "nonatomic"),
    (128, False, "nonatomic_mxfp4"),
    (32, False, "nonatomic_cshuffle"),
    (64, False, "nonatomic_cshuffle"),
    (128, False, "nonatomic_cshuffle"),
}


def _run_compiled(exe, args):
    """JIT-compile on first call, then dispatch via the cached CompiledFunction."""
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        try:
            from flydsl._mlir import ir

            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise


@functools.cache
def _get_compiled_gemm1(
    BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, BN, BK, interleave, xcd_swizzle, a_dtype
):
    return compile_gemm1_a4w4_port(
        BM,
        use_nt,
        inline_quant,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        BN=BN,
        BK=BK,
        interleave=interleave,
        xcd_swizzle=xcd_swizzle,
        a_dtype=a_dtype,
    )


@functools.cache
def _get_compiled_gemm2(BM, use_nt, NE, N_OUT, epilog, D_INTER, D_INTER_REAL, BN, BK, xcd_swizzle):
    return compile_gemm2_a4w4_port(
        BM=BM,
        use_nt=use_nt,
        NE=NE,
        N_OUT=N_OUT,
        epilog=epilog,
        D_INTER=D_INTER,
        D_INTER_REAL=D_INTER_REAL,
        BN=BN,
        BK=BK,
        xcd_swizzle=xcd_swizzle,
    )


def flydsl_mxfp4_gemm1(
    *,
    a_quant,
    a_scale_sorted_shuffled,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    hidden_states,
    n_tokens,
    BM,
    use_nt,
    inline_quant,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BN=256,
    BK=256,
    interleave=False,
    xcd_swizzle=0,
    a_dtype="fp4",
    stream=None,
):
    """Fused stage1: gate+up GEMM + SiLU + fp4 re-quant.

    ``a_dtype`` selects the activation format: "fp4" (a4w4, mxfp4 A) or "fp8"
    (a8w4, fp8 e4m3 A x mxfp4 W1). Writes the sorted fp4 intermediate into
    ``inter_sorted_quant`` / ``inter_sorted_shuffled_scale`` (both pre-allocated).
    """
    if D_HIDDEN % BK != 0:
        raise NotImplementedError(f"mxmoe gemm1 requires D_HIDDEN (K) % {BK} == 0, got H={D_HIDDEN}")
    if (2 * D_INTER) % BN != 0:
        raise NotImplementedError(f"mxmoe gemm1 requires 2*D_INTER (N_OUT) % {BN} == 0, got D_INTER={D_INTER}")
    if (BM, use_nt, inline_quant, a_dtype) not in _G1_SUPPORTED:
        raise NotImplementedError(
            f"mxmoe gemm1 unsupported variant "
            f"(BM={BM}, use_nt={use_nt}, inline_quant={inline_quant}, a_dtype={a_dtype})"
        )

    launch = _get_compiled_gemm1(
        BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, BN, BK, interleave, xcd_swizzle, a_dtype
    )
    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER, BN=BN)
    _run_compiled(
        launch,
        (
            a_quant.data_ptr(),
            a_scale_sorted_shuffled.data_ptr(),
            w1_u8.data_ptr(),
            w1_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            m_indices.data_ptr(),
            int(n_tokens),
            int(grid),
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            hidden_states.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return inter_sorted_quant, inter_sorted_shuffled_scale


def flydsl_mxfp4_gemm2(
    *,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    BM,
    use_nt,
    epilog,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    flat_out_scale=None,
    D_INTER_REAL=None,
    BN=256,
    BK=256,
    xcd_swizzle=0,
    stream=None,
):
    """Down-projection stage2. Consumes the stage1 sorted fp4 intermediate."""
    if D_INTER % BK != 0:
        raise NotImplementedError(
            f"mxmoe gemm2 contraction D_INTER (inter_dim) must be a multiple of " f"{BK}, got D_INTER={D_INTER}"
        )
    if D_HIDDEN % BN != 0:
        raise NotImplementedError(f"mxmoe gemm2 requires D_HIDDEN (N_OUT=model_dim) % {BN} == 0, " f"got H={D_HIDDEN}")
    if (BM, use_nt, epilog) not in _G2_SUPPORTED:
        raise NotImplementedError(f"mxmoe gemm2 unsupported variant (BM={BM}, use_nt={use_nt}, epilog={epilog})")

    launch = _get_compiled_gemm2(BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, BN, BK, xcd_swizzle)
    max_m_blocks = (max_sorted + BM - 1) // BM
    if flat_out_scale is None:
        flat_out_scale = torch.empty(1, dtype=torch.uint8, device=flat_out.device)

    _run_compiled(
        launch,
        (
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            w2_u8.data_ptr(),
            w2_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            int(M_logical),
            int(max_m_blocks),
            flat_out.data_ptr(),
            flat_out_scale.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return flat_out
