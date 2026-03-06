#!/usr/bin/env python3
"""
Test FlyDSL MoE Stage2 only (standalone).

This test file focuses exclusively on stage2 kernel testing, following
the same code style as test_moe_gemm.py. It mirrors the structure of
run_moe_stage2() but is designed for standalone testing without
requiring stage1 execution.

Features:
- Standalone stage2 testing (no stage1 dependency)
- Synthetic A2 generation for testing
- Multiple input dtypes: fp8, fp16, int8, int8smooth, int4 (W4A8)
- Atomic and reduce accumulation modes
- Correctness verification against torch reference
- Performance benchmarking with TFLOPS/TB/s metrics
- Optional comparison with aiter CK implementation

Usage Examples:
    # Quick test with small dimensions
    python test_moe_stage2_only.py -t 32 -dim 128,128 -e 4 -k 2 --tile_k 64

    # Test all dtypes with realistic dimensions
    python test_moe_stage2_only.py -t 128 -dim 6144,4096 -e 8 -k 2 --in_dtype all

    # Test reduce mode
    python test_moe_stage2_only.py --use_reduce

    # Benchmark mode (skip correctness check, more iterations)
    python test_moe_stage2_only.py --skip_ref --num_iters 100 --num_warmup 10

    # Compare with aiter CK (fp8 only)
    python test_moe_stage2_only.py --in_dtype fp8 --compare_aiter_ck

    # Large-scale benchmark
    python test_moe_stage2_only.py -t 2048 -dim 6144,4096 -e 8 -k 2
"""

import argparse
import logging
import math
import os
import sys
from typing import Tuple, Optional

import torch

# Ensure we use the repo-local FlyDSL
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Claim the local `kernels` namespace before aiter's editable install can
# shadow it with its own `kernels/` subpackage.
if "kernels" not in sys.modules:
    import importlib.util as _ilu
    _local_ki = os.path.join(_REPO_ROOT, "kernels", "__init__.py")
    if os.path.isfile(_local_ki):
        _spec = _ilu.spec_from_file_location(
            "kernels", _local_ki,
            submodule_search_locations=[os.path.join(_REPO_ROOT, "kernels")],
        )
        _km = _ilu.module_from_spec(_spec)
        sys.modules["kernels"] = _km
        _spec.loader.exec_module(_km)

from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.utils import pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils

# Reuse routing and utility functions from test_moe_gemm
from tests.kernels.test_moe_gemm import (
    build_routing_buffers,
    _make_reduce_mode_compile_fn,
    _pack_shuffled_int8_to_packed_int4_no_perm,
    RoutingBuffers,
    HAS_AITER,
)


def get_topk_valid_mask(topk_ids: torch.Tensor, expert_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build valid_mask [tokens, topk] for (optional) EP-style masking.

    If expert_mask is None: all slots are valid (all ones)
    Else: valid_mask[t, k] = expert_mask[topk_ids[t, k]] (cast to int8)
    """
    if expert_mask is None:
        return torch.ones(topk_ids.shape, dtype=torch.int8, device=topk_ids.device)
    return expert_mask[topk_ids].to(torch.int8)

from kernels.moe_gemm_2stage import (
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
    compile_moe_reduction,
    MoeGemm2Mode,
)

logging.basicConfig(level=logging.INFO)

ARCH = get_rocm_arch()
# GFX950 (MI350) and newer use OCP standard float8_e4m3fn
# GFX940/941/942 (MI300) use float8_e4m3fnuz
if "gfx95" in ARCH:
    DTYPE_FP8 = torch.float8_e4m3fn
else:
    DTYPE_FP8 = torch.float8_e4m3fnuz

# FP4x2 dtype
DTYPE_FP4 = torch.float4_e2m1fn_x2


def run_moe_stage2_only(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    compare_aiter_ck: Optional[bool] = None,
    moe_sort_mode: Optional[str] = None,
    # Stage2 input override (pre-generated A2 from stage1)
    a2_fp8_in: Optional[torch.Tensor] = None,
    a2_scale_in: Optional[torch.Tensor] = None,
    # Weight override
    w2_fp32_in: Optional[torch.Tensor] = None,
    # Routing override
    topk_ids_in: Optional[torch.Tensor] = None,
    topk_weights_in: Optional[torch.Tensor] = None,
    routing_in: Optional[RoutingBuffers] = None,
    return_outputs: bool = False,
    skip_ref: bool = False,
    init_scale: float = 0.2,
    # Custom compile function for kernel comparison
    compile_fn=None,
    kernel_name: str = "moe_stage2_only",
    # Use reduce mode (accumulate=False) instead of atomic mode
    use_reduce: bool = False,
    # Use valid mask optimization
    use_valid_mask: bool = False,
    # graph mode
    test_graph: bool = False,
):
    """
    MoE stage2 only test (standalone, no stage1 dependency).

    This function mirrors the structure of run_moe_stage2() from test_moe_gemm.py
    but focuses exclusively on testing stage2 kernel.
    """

    # Parameter sanity checks
    if model_dim % tile_n != 0:
        raise ValueError(
            f"Invalid stage2 tiling: model_dim ({model_dim}) must be divisible by tile_n ({tile_n})."
        )
    if inter_dim % tile_k != 0:
        raise ValueError(
            f"Invalid stage2 tiling: inter_dim ({inter_dim}) must be divisible by tile_k ({tile_k})."
        )
    if (tile_m * tile_k) % 256 != 0:
        raise ValueError(
            f"Invalid stage2 tiling: tile_m*tile_k must be divisible by 256. "
            f"Got tile_m={tile_m}, tile_k={tile_k} -> tile_m*tile_k={tile_m * tile_k}."
        )
    bytes_per_thread_x = (tile_m * tile_k) // 256
    if bytes_per_thread_x % 4 != 0:
        raise ValueError(
            f"Invalid stage2 tiling: bytes_per_thread_x must be divisible by 4. "
            f"Got bytes_per_thread_x={bytes_per_thread_x}."
        )

    # Default compile function
    if compile_fn is None:
        if use_reduce:
            compile_fn = _make_reduce_mode_compile_fn(
                use_flydsl_reduce=True
            )
        else:
            compile_fn = compile_moe_gemm2

    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    s = float(init_scale)

    # Validate dtype
    if in_dtype not in ("fp8", "fp16", "int8", "int8smooth", "int4", "fp4"):
        raise ValueError(
            f"in_dtype must be one of ('fp8','fp16','int8','int8smooth','int4','fp4'), got {in_dtype!r}"
        )
    is_int4 = in_dtype == "int4"
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"
    is_fp4 = in_dtype == "fp4"  # A4W4 mode

    # Generate W2 if not provided
    w2_fp32 = (
        w2_fp32_in
        if w2_fp32_in is not None
        else torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)
        * (s / math.sqrt(inter_dim))
    )

    # Routing: deterministic torch topk + softmax
    if topk_ids_in is None or topk_weights_in is None:
        # Generate random routing
        score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    else:
        topk_ids = topk_ids_in
        topk_weights = topk_weights_in

    routing = (
        routing_in
        if routing_in is not None
        else build_routing_buffers(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
            moe_sort_mode=moe_sort_mode,
        )
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    ) = routing

    # Quantize W2
    if in_dtype == "fp8":
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=DTYPE_FP8)
    elif in_dtype == "fp16":
        w2_q = w2_fp32.to(torch.float16)
        scale_w2 = None
    elif in_dtype == "fp4":
        # A4W4: use fp4_utils quantization (no aiter dependency)
        w2_q, scale_w2, _ = fp4_utils.per_1x32_f4_quant(w2_fp32)
        # w2_q: [E, model_dim, inter_dim//2] fp4x2
        # scale_w2: [E*model_dim, inter_dim//32] fp8_e8m0
    elif in_dtype in ("int8", "int8smooth"):
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    else:  # int4
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

    # Preshuffle W2
    if is_fp4:
        # A4W4: use regular shuffle (fp4 weights are already packed)
        w2_shuffled = shuffle_weight(w2_q, layout=(16, 16))
    else:
        w2_shuffled = shuffle_weight(w2_q)

    # Generate or use provided A2 (stage1 output)
    a2_scale_original = None  # For fp4 reference comparison
    if a2_fp8_in is not None and (a2_scale_in is not None or in_dtype in ("fp16", "fp4")):
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
    else:
        # Generate synthetic A2 for standalone testing
        a2_fp32 = (
            torch.randn((tokens, topk, inter_dim), device=device, dtype=torch.float32) * s
        )
        if in_dtype == "fp8":
            a2_q, a2_scale = pertoken_quant(a2_fp32, quant_dtype=DTYPE_FP8)
        elif in_dtype == "fp16":
            a2_q = a2_fp32.to(torch.float16)
            a2_scale = None
        elif in_dtype == "fp4":
            # A4W4: quantize to fp4x2 using fp4_utils (no aiter dependency)
            a2_q_flat, a2_scale_raw, _ = fp4_utils.per_1x32_f4_quant(
                a2_fp32.view(tokens * topk, inter_dim)
            )
            # a2_q_flat: [tokens*topk, inter_dim//2] fp4x2
            # a2_scale_raw: [tokens*topk, inter_dim//32] fp8_e8m0
            a2_q = a2_q_flat.view(tokens, topk, inter_dim // 2)
            a2_scale_original = a2_scale_raw.view(tokens, topk, -1)
            a2_scale = fp4_utils.moe_mxfp4_sort(
                a2_scale_raw.view(tokens, topk, -1),
                sorted_ids=sorted_token_ids,
                num_valid_ids=num_valid_ids,
                token_num=tokens,
                block_size=tile_m,
            )
        elif in_dtype in ("int8", "int8smooth"):
            if is_int8smooth:
                # Apply per-expert smooth scale
                smooth_scale2 = 0.75 + 0.5 * torch.rand(
                    (experts, inter_dim), device=device, dtype=torch.float32
                )
                a2_fp32 = a2_fp32 * smooth_scale2[topk_ids.to(torch.int64)]
            a2_q, a2_scale = pertoken_quant(a2_fp32, quant_dtype=torch.int8)
        else:  # int4
            a2_q, a2_scale = pertoken_quant(a2_fp32, quant_dtype=torch.int8)

    # Flatten weights/scales for kernel
    if is_fp4:
        # A4W4: already packed to inter_dim//2
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim // 2)
        # For fp4, use e8m0_shuffle for scales (same as aiter)
        scale_w2_shuffled = fp4_utils.e8m0_shuffle(scale_w2.view(experts, -1))
        scale_w2_flat = scale_w2_shuffled  # Already in the right shape
    else:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
        scale_w2_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, 1)

    # For W4A8, pack preshuffled int8 weights into packed int4 bytes
    w2_kernel = w2_shuffled_flat
    if is_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4_no_perm(w2_shuffled_flat)

    w2_flat = w2_kernel.contiguous().view(-1)
    w2_kernel = w2_flat
    if is_fp4:
        # A4W4: keep packed shape
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim // 2)
    elif not is_int4:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim)

    # Flatten scales to 1D memrefs
    if a2_scale is None:
        a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        if is_fp4:
            # A4W4: a2_scale is already sorted, shape is from moe_mxfp4_sort
            a2_scale_1d = a2_scale.view(-1).contiguous()
        else:
            a2_scale_1d = a2_scale.view(-1).contiguous()  # [tokens*topk]

    if is_fp4:
        # A4W4: use shuffled scales
        w2_scale_1d = scale_w2_shuffled.view(-1).contiguous()
    elif scale_w2_flat is None:
        w2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        w2_scale_1d = scale_w2_flat.view(-1).contiguous()  # [experts*model_dim]
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    # Output dtype
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half"):
        raise ValueError(f"out_dtype must be 'f16', got {out_dtype!r}")
    out_torch_dtype = torch.float16

    out = torch.zeros((tokens, model_dim), device=device, dtype=out_torch_dtype)
    out_perf = torch.zeros_like(out)

    # Stage2 applies weight when stage1 did not
    doweight_stage2 = True  # For standalone testing, always apply weights in stage2

    # Compile kernel
    fp4_reduce_exe = None  # Separate reduce kernel for fp4+reduce mode
    fp4_intermediate = None  # Intermediate buffer for fp4+reduce mode
    if is_fp4:
        # A4W4: use mixed kernel
        from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        # Determine actual inter_dim for compilation (fp4 uses packed dim)
        compile_inter_dim = inter_dim  # For mixed kernel, use original (unpacked) dim

        exe = compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=compile_inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=bool(doweight_stage2),
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=out_dtype,
            accumulate=not use_reduce,
            enable_bias=False,
        )
        if use_reduce:
            # Compile a separate reduce kernel for fp4+reduce mode
            _rdtype = "f16" if out_dtype in ("f16", "fp16") else ("bf16" if out_dtype in ("bf16", "bfloat16") else "f32")
            fp4_reduce_exe = compile_moe_reduction(
                topk=topk,
                model_dim=model_dim,
                dtype_str=_rdtype,
            )
            fp4_intermediate = torch.empty(
                tokens * topk, model_dim,
                device=device,
                dtype=out_torch_dtype,
            )
    elif use_reduce:
        # For reduce mode, compile_fn is already wrapped (_make_reduce_mode_compile_fn)
        exe = compile_fn(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            in_dtype=in_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=bool(doweight_stage2),
        )
    else:
        # Atomic mode needs accumulate=True
        exe = compile_fn(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            in_dtype=in_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=bool(doweight_stage2),
            accumulate=True,  # Atomic mode
        )
    is_reduce_exe = (getattr(exe, "mode", None) == MoeGemm2Mode.REDUCE) or bool(use_reduce)

    # Prepare bias (fp4 kernels need it)
    if is_fp4:
        bias2 = torch.empty(0, dtype=torch.float32, device=device)
    else:
        bias2 = None

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        stream = torch.cuda.current_stream()
        valid_mask = None
        if is_reduce_exe and bool(use_valid_mask):
            valid_mask = get_topk_valid_mask(topk_ids, expert_mask=None).contiguous()

        if is_fp4 and fp4_reduce_exe is not None:
            # A4W4 + reduce: write to intermediate, then reduce
            fp4_intermediate.zero_()
            exe(
                fp4_intermediate.view(-1),
                x,
                w,
                sx,
                sw,
                st,
                eids,
                sw_sorted,
                num_valid_ids,
                bias2,
                tokens,
                model_dim,
                inter_dim,
                int(blocks),
                stream,
            )
            # Reduce: [tokens*topk, model_dim] -> [tokens, model_dim]
            X_reduce = fp4_intermediate.view(tokens, topk, model_dim)
            Y_reduce = o.view(tokens, model_dim)
            if valid_mask is None:
                valid_mask = torch.empty((0, topk), device=o.device, dtype=torch.uint8)
            fp4_reduce_exe(X_reduce, Y_reduce, valid_mask, tokens, stream)
        elif is_fp4:
            # A4W4: atomic mode
            exe(
                o,
                x,
                w,
                sx,
                sw,
                st,
                eids,
                sw_sorted,
                num_valid_ids,
                bias2,
                tokens,
                model_dim,
                inter_dim,
                int(blocks),
                stream,
            )
        elif is_reduce_exe:
            exe(
                o,
                x,
                w,
                sx,
                sw,
                st,
                eids,
                sw_sorted,
                num_valid_ids,  # Tensor[1] int32
                tokens,
                model_dim,
                inter_dim,
                int(blocks),
                stream,
            )
        else:
            exe(
                o,
                x,
                w,
                sx,
                sw,
                st,
                eids,
                sw_sorted,
                num_valid_ids,  # Tensor[1] int32
                tokens,
                model_dim,
                inter_dim,
                int(blocks),
                stream,
            )

    # Benchmark
    _, us = run_perftest(
        launch,
        out_perf,
        a2_q.view(-1),
        w2_kernel.view(-1),
        a2_scale_1d,
        w2_scale_1d,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_iters=int(num_iters),
        num_warmup=int(num_warmup),
        testGraph=test_graph,
    )
    torch.cuda.synchronize()

    # Correctness run
    out.zero_()
    launch(
        out,
        a2_q.view(-1),
        w2_kernel.view(-1),
        a2_scale_1d,
        w2_scale_1d,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
    )
    torch.cuda.synchronize()

    # Compare with reference
    if not bool(skip_ref):
        if is_fp4:
            # A4W4: pure torch reference using fp4_utils dequantization
            a2_u8 = a2_q.view(torch.uint8).reshape(tokens * topk, inter_dim // 2)
            a2_s_u8 = a2_scale_original.view(torch.uint8).reshape(tokens * topk, -1)
            a2_deq = fp4_utils.mxfp4_to_f32(a2_u8)
            a2_s_f32 = fp4_utils.e8m0_to_f32(a2_s_u8)
            a2_deq = (a2_deq * a2_s_f32.unsqueeze(-1).expand(-1, -1, 32).reshape(tokens * topk, inter_dim))
            a2_deq = a2_deq.view(tokens, topk, inter_dim).to(torch.float16)

            w2_u8 = w2_q.view(torch.uint8).reshape(experts * model_dim, inter_dim // 2)
            w2_s_u8 = scale_w2.view(torch.uint8).reshape(experts * model_dim, -1)
            w2_deq = fp4_utils.mxfp4_to_f32(w2_u8)
            w2_s_f32 = fp4_utils.e8m0_to_f32(w2_s_u8)
            w2_deq = (w2_deq * w2_s_f32.unsqueeze(-1).expand(-1, -1, 32).reshape(experts * model_dim, inter_dim))
            w2_deq = w2_deq.view(experts, model_dim, inter_dim).to(torch.float16)

            ref2 = torch.zeros((tokens, model_dim), device=device, dtype=torch.float32)
            for si in range(topk):
                eidx = topk_ids[:, si].to(torch.int64)
                w_sel = w2_deq.index_select(0, eidx)
                y = torch.bmm(
                    a2_deq[:, si, :].unsqueeze(1).to(torch.float32),
                    w_sel.transpose(1, 2).to(torch.float32),
                ).squeeze(1)
                if doweight_stage2:
                    y = y * topk_weights[:, si : si + 1].to(torch.float32)
                ref2 += y
            ref2 = ref2.to(torch.float16)
            assert verify_output(out.to(torch.float32), ref2.to(torch.float32), rtol=1.0, atol=1.0)
        else:
            ref2 = torch_moe_gemm2(
                a2_q,
                w2_q,
                a2_scale,
                scale_w2,
                topk_ids.to(torch.int64),
                topk_weights,
                model_dim=model_dim,
                doweight_stage2=doweight_stage2,
            )
            assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    # Compute performance metrics
    flops = 2 * tokens * topk * model_dim * inter_dim
    tflops = flops / (us / 1e6) / 1e12

    bytes_moved = 0
    if is_fp4:
        # A4W4: 4-bit packed (0.5 bytes per element)
        bytes_moved += tokens * topk * inter_dim // 2  # a2 (fp4, packed)
        bytes_moved += (experts * model_dim * inter_dim) // 2  # w2 (fp4, packed)
    else:
        bytes_moved += tokens * topk * inter_dim * 1  # a2 (fp8/int8)
        bytes_moved += (experts * model_dim * inter_dim) // (2 if is_int4 else 1)  # w2
    bytes_moved += tokens * model_dim * 2  # out (f16)
    bytes_moved += tokens * topk * 4  # a2_scale f32
    bytes_moved += experts * model_dim * 4  # w2_scale f32
    bytes_moved += int(sorted_weights.numel()) * 4
    bytes_moved += int(sorted_token_ids.numel()) * 4
    bytes_moved += int(sorted_expert_ids.numel()) * 4
    tbps = bytes_moved / 1e12 / (us / 1e6)

    print(
        f"FLIR MoE stage2 [{kernel_name}] {in_dtype} {'reduce' if use_reduce else 'atomic'} | "
        f"{model_dim}x{inter_dim}, E={experts}, K={topk}, M_eff={tokens*topk} | "
        f"{us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s"
    )

    # Optional: compare with aiter CK
    if compare_aiter_ck is None:
        compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    else:
        compare_ck = bool(compare_aiter_ck)
    compare_ck = compare_ck and (in_dtype == "fp8")

    if compare_ck:
        if not HAS_AITER:
            logging.warning("aiter not available; skipping CK comparison")
        else:
            try:
                from aiter.ops.moe_op import ck_moe_stage2_fwd
                from aiter.ops.enum import QuantType, ActivationType

                # Preshuffle W1 for CK (even though it's not used in stage2)
                w1_fp32 = torch.randn(
                    (experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32
                ) * s
                if in_dtype == "fp8":
                    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)
                else:
                    w1_q = w1_fp32.to(torch.float16)
                    scale_w1 = None
                w1_shuffled = shuffle_weight(w1_q)

                out_ck = torch.zeros((tokens, model_dim), device=device, dtype=torch.float16)
                out_ck_perf = torch.zeros_like(out_ck)

                def launch_ck(
                    o, a2_, w1_, w2_, sorted_ids_, sorted_eids_, num_valid_, w2_scale_, a2_scale_, sorted_w_
                ):
                    ck_moe_stage2_fwd(
                        inter_states=a2_,
                        w1=w1_,
                        w2=w2_,
                        sorted_token_ids=sorted_ids_,
                        sorted_expert_ids=sorted_eids_,
                        num_valid_ids=num_valid_,
                        out=o,
                        topk=topk,
                        kernelName="",
                        w2_scale=w2_scale_,
                        a2_scale=a2_scale_,
                        block_m=tile_m,
                        sorted_weights=sorted_w_ if doweight_stage2 else None,
                        quant_type=QuantType.per_Token,
                        activation=ActivationType.Silu,
                    )

                _, us_ck = run_perftest(
                    launch_ck,
                    out_ck_perf,
                    a2_q,
                    w1_shuffled,
                    w2_shuffled,
                    sorted_token_ids,
                    sorted_expert_ids,
                    num_valid_ids,
                    scale_w2.contiguous(),
                    a2_scale.contiguous(),
                    sorted_weights,
                    num_iters=int(num_iters),
                    num_warmup=int(num_warmup),
                    testGraph=test_graph,
                )

                tflops_ck = flops / (us_ck / 1e6) / 1e12
                print(
                    f"[aiter CK] stage2: {us_ck:.1f} us, "
                    f"{tflops_ck:.2f} TFLOPS, speedup: {tflops / tflops_ck:.2f}x"
                )

                # Correctness
                out_ck.zero_()
                launch_ck(
                    out_ck,
                    a2_q,
                    w1_shuffled,
                    w2_shuffled,
                    sorted_token_ids,
                    sorted_expert_ids,
                    num_valid_ids,
                    scale_w2.contiguous(),
                    a2_scale.contiguous(),
                    sorted_weights,
                )
                torch.cuda.synchronize()
                if not verify_output(
                    out.to(torch.float32), out_ck.to(torch.float32), rtol=0.5, atol=0.5, msg="[aiter CK]:"
                ):
                    logging.warning("[aiter CK] correctness mismatch vs FLIR")
            except Exception as e:
                logging.warning(f"Skipping aiter CK comparison: {e}")

    if return_outputs:
        return out, us
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test FlyDSL MoE Stage2 only",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Dimensions
    parser.add_argument("-t", "--tokens", type=int, default=32, help="Number of tokens")
    parser.add_argument(
        "-dim",
        type=lambda v: tuple(int(x) for x in v.split(",")),
        default=(6144, 4096),
        help="Dimensions: model_dim,inter_dim (e.g., 6144,4096)",
    )
    parser.add_argument("-e", "--experts", type=int, default=8, help="Number of experts")
    parser.add_argument("-k", "--topk", type=int, default=2, help="Top-k")

    # Tiling
    parser.add_argument("--tile_m", type=int, default=32, help="Tile M (routing block)")
    parser.add_argument("--tile_n", type=int, default=128, help="Tile N (model_dim tile)")
    parser.add_argument("--tile_k", type=int, default=256, help="Tile K (inter_dim tile)")

    # Kernel config
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp8",
        choices=["fp8", "fp16", "int8", "int8smooth", "int4", "fp4", "all"],
        help="Input dtype (fp4 = A4W4 mixed precision)",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="f16",
        choices=["f16"],
        help="Output dtype (currently only f16 supported)",
    )
    parser.add_argument(
        "--use_reduce",
        action="store_true",
        help="Use reduce mode instead of atomic mode",
    )
    parser.add_argument(
        "--use_valid_mask",
        action="store_true",
        help="Use valid mask optimization (reduce mode only)",
    )

    # Benchmark
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--num_warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--skip_ref", action="store_true", help="Skip reference check")
    parser.add_argument(
        "--compare_aiter_ck",
        action="store_true",
        help="Compare with aiter CK implementation",
    )
    parser.add_argument(
        "--moe_sort_mode",
        type=str,
        default="torch",
        choices=["torch", "aiter"],
        help="MoE sorting mode",
    )

    args = parser.parse_args()

    torch.set_default_device("cuda")

    model_dim, inter_dim = args.dim

    # Determine which dtypes to test
    if args.in_dtype == "all":
        in_dtypes = ["fp8", "fp16", "int8", "int4", "fp4"]
    else:
        in_dtypes = [args.in_dtype]

    # Run tests
    for in_dtype in in_dtypes:
        try:
            run_moe_stage2_only(
                tokens=args.tokens,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=args.experts,
                topk=args.topk,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                tile_k=args.tile_k,
                in_dtype=in_dtype,
                out_dtype=args.out_dtype,
                seed=args.seed,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
                skip_ref=args.skip_ref,
                use_reduce=args.use_reduce,
                use_valid_mask=args.use_valid_mask,
                compare_aiter_ck=args.compare_aiter_ck,
                moe_sort_mode=args.moe_sort_mode,
                kernel_name=f"moe_stage2_{in_dtype}",
            )
        except Exception as e:
            logging.error(f"Test failed for {in_dtype}: {e}")
            raise


if __name__ == "__main__":
    main()
