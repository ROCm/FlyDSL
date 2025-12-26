#!/usr/bin/env python3
"""
MoE GEMM stage1 unit test (flir, MFMA FP8, B preshuffle).

Reference: `../aiter/op_tests/test_moe_2stage.py` (stage1 semantics)
- Per-token topk routing: `topk_ids` and `topk_weights`
- Stage1: y2 = x @ W1[expert]^T -> split into gate/up (2*inter_dim)
         out1 = activation(gate) * up
         (optional) apply topk_weights (doweight_stage1)

This test constructs CK-style routing buffers:
- `sorted_token_ids`: packed (token_id | (topk_slot << 24)), padded to tile_m
- `expert_ids`: expert id per M tile (tile_m rows)

Kernel writes output as E[t, topk_slot, inter_dim] (fp16).
"""

import logging
import os
import sys
from typing import Tuple, Optional

import pytest
import torch
import torch.nn.functional as F

# This test is FLIR/pyflir-specific. In environments where the active embedded MLIR
# runtime comes from RocDSL (or FLIR extensions weren't built), importing FLIR
# python extensions like `_flirPasses` will fail and the test cannot run.
#
# `run_tests.sh` in this repo executes files directly (not via pytest), so
# raising `pytest.skip` would be treated as a failure. Handle both modes.
try:
    from _mlir._mlir_libs import _flirPasses as _FLIR_PASSES  # type: ignore
except Exception:
    if __name__ == "__main__":
        print("[test_moe_gemm] FLIR python extensions not available; skipping.")
        sys.exit(0)
    pytest.skip("FLIR python extensions not available; skipping.", allow_module_level=True)

import pyflir
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.runtime.device import get_rocm_arch as get_hip_arch
from pyflir.utils import SmemAllocator
from tests.utils import pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest

from _mlir import ir
from _mlir.dialects import vector
from pyflir.dialects.ext import arith, gpu, buffer_ops, math as mlir_math
from _mlir.dialects import arith as _arith_mlir
import _mlir.dialects.rocdl as rocdl
import _mlir.extras.types as T

# Optional: use aiter's exact routing/sorting implementation (matches `aiter/op_tests/test_moe_2stage.py`).
# Some environments ship aiter python but miss required JIT .so dependencies; we fall back gracefully.
try:
    import aiter
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting

    HAS_AITER = True
except Exception:
    HAS_AITER = False

from tests.python.gpu.mfma_fp8_preshuffle_pipeline import (
    compute_split_load_lens,
    load_fp8_tile_split_dwordx4,
)

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def unwrap(v):
    if isinstance(v, int):
        return arith.constant(v, index=True).value
    while hasattr(v, "value") or hasattr(v, "_value"):
        v = getattr(v, "_value", getattr(v, "value", v))
    return v


def build_sorted_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    tile_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build CK-style routing buffers:
    - `sorted_token_ids[int32]` (fused token_idx[23:0] + topk_slot[31:24])
    - `sorted_weights[fp32]` aligned with sorted_token_ids
    - `sorted_expert_ids[int32]` per M-block

    Prefer aiter's `moe_sorting` (same path as `aiter/op_tests/test_moe_2stage.py`),
    then pad to `len(sorted_expert_ids) * tile_m` so our kernel never OOB-loads.
    """
    assert topk_ids.is_cuda and topk_weights.is_cuda
    tokens, topk = topk_ids.shape

    if HAS_AITER:
        try:
            topk_ids_i32 = topk_ids.to(torch.int32)
            topk_w_f32 = topk_weights.to(torch.float32)
            sorted_ids, sorted_w, sorted_expert_ids, _num_valid_ids, _moe_buf = aiter_moe_sorting(
                topk_ids_i32,
                topk_w_f32,
                num_experts,
                model_dim,
                torch.float16,
                tile_m,  # block_size must match our kernel tile_m
            )

            # Pad to full blocks so the kernel can safely index [bx*tile_m + lane_row]
            mblocks = int(sorted_expert_ids.numel())
            padded_len = mblocks * tile_m
            if int(sorted_ids.numel()) < padded_len:
                pad_ids = torch.empty((padded_len,), device="cuda", dtype=torch.int32)
                pad_w = torch.empty((padded_len,), device="cuda", dtype=torch.float32)
                pad_ids.fill_(tokens)  # sentinel fused token (token=tokens, slot=0)
                pad_w.zero_()
                pad_ids[: sorted_ids.numel()] = sorted_ids
                pad_w[: sorted_w.numel()] = sorted_w
                sorted_ids = pad_ids
                sorted_w = pad_w
            else:
                sorted_ids = sorted_ids[:padded_len]
                sorted_w = sorted_w[:padded_len]

            return sorted_ids, sorted_w, sorted_expert_ids
        except Exception:
            # Fall back below.
            pass

    # Fallback (no working aiter): build the same CK-style buffers using torch,
    # with a deterministic ordering: sort by (expert, token, slot).
    topk_ids_i64 = topk_ids.to(torch.int64)
    topk_w_f32 = topk_weights.to(torch.float32)

    token_idx = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1).expand(tokens, topk)
    slot_idx = torch.arange(topk, device="cuda", dtype=torch.int64).unsqueeze(0).expand(tokens, topk)

    expert_flat = topk_ids_i64.reshape(-1)
    token_flat = token_idx.reshape(-1)
    slot_flat = slot_idx.reshape(-1)
    fused_flat = (token_flat & 0xFFFFFF) | ((slot_flat & 0xFF) << 24)
    weight_flat = topk_w_f32.reshape(-1)

    # Sort by (expert, token, slot) to make ordering stable/deterministic.
    linear = token_flat * topk + slot_flat
    key = expert_flat * (tokens * topk) + linear
    order = torch.argsort(key)

    expert_sorted = expert_flat[order]
    fused_sorted = fused_flat[order].to(torch.int32)
    w_sorted = weight_flat[order]

    counts = torch.bincount(expert_flat.clamp(min=0, max=num_experts - 1), minlength=num_experts).to("cpu").tolist()
    out_ids = []
    out_w = []
    out_expert_ids: list[int] = []
    off = 0
    sentinel = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    sentinel_w = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    for e in range(num_experts):
        cnt = int(counts[e])
        seg_ids = fused_sorted[off : off + cnt]
        seg_w = w_sorted[off : off + cnt]
        off += cnt

        # pad to tile_m
        pad = (-cnt) % tile_m
        if pad:
            seg_ids = torch.cat([seg_ids, sentinel.expand(pad)])
            seg_w = torch.cat([seg_w, sentinel_w.expand(pad)])

        out_ids.append(seg_ids)
        out_w.append(seg_w)
        for _ in range(int(seg_ids.numel() // tile_m)):
            out_expert_ids.append(e)

    sorted_token_ids = torch.cat(out_ids, dim=0)
    sorted_weights = torch.cat(out_w, dim=0)
    sorted_expert_ids = torch.tensor(out_expert_ids, device="cuda", dtype=torch.int32)
    return sorted_token_ids, sorted_weights, sorted_expert_ids


def torch_stage1_ref(
    x_fp8: torch.Tensor,
    w1_fp8_flat: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w1_flat: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter_dim: int,
    doweight_stage1: bool,
) -> torch.Tensor:
    """Return [tokens, topk, inter_dim] fp32."""
    tokens, model_dim = x_fp8.shape
    topk = topk_ids.shape[1]
    experts = int(topk_ids.max().item()) + 1

    x = x_fp8.to(torch.float32) * scale_x  # [tokens, model_dim] (scale_x [tokens,1])
    w1 = w1_fp8_flat.to(torch.float32) * scale_w1_flat  # [experts*2*inter_dim, model_dim] (scale [rows,1])
    w1 = w1.view(experts, 2 * inter_dim, model_dim)

    out = torch.zeros((tokens, topk, inter_dim), device="cuda", dtype=torch.float32)
    for e in range(experts):
        # routes assigned to expert e
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)  # [num, 2] (t, slot)
        if idx.numel() == 0:
            continue
        t_idx = idx[:, 0]
        s_idx = idx[:, 1]
        y2 = F.linear(x[t_idx, :], w1[e, :, :])  # [num, 2*inter_dim]
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
        if doweight_stage1:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out[t_idx, s_idx, :] = y
    return out


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk,doweight_stage1",
    [
        (256, 1024, 256, 4, 2, False),
        (256, 1024, 256, 4, 2, True),
    ],
)
def _maybe_aiter_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    block_m: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids) or None."""
    if not HAS_AITER:
        return None
    try:
        # aiter expects i32 ids and fp32 weights
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_w_f32 = topk_weights.to(torch.float32)
        sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = aiter_moe_sorting(
            topk_ids_i32,
            topk_w_f32,
            num_experts,
            model_dim,
            torch.float16,
            block_m,
        )
        # `num_valid_ids` is documented as [1]; some builds allocate [2]. Keep the first element.
        if num_valid_ids.numel() > 1:
            num_valid_ids = num_valid_ids[:1].contiguous()
        return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids
    except Exception:
        return None


def _pad_sorted_buffers_to_full_blocks(
    *,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    tokens: int,
    block_m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad (sorted_ids, sorted_weights) to len(sorted_expert_ids)*block_m for safe OOB-free indexing."""
    assert sorted_ids.dtype == torch.int32
    assert sorted_weights.dtype == torch.float32
    blocks = int(sorted_expert_ids.numel())
    padded_len = blocks * block_m
    if int(sorted_ids.numel()) >= padded_len:
        return sorted_ids[:padded_len], sorted_weights[:padded_len]
    pad_ids = torch.empty((padded_len,), device="cuda", dtype=torch.int32)
    pad_w = torch.empty((padded_len,), device="cuda", dtype=torch.float32)
    pad_ids.fill_(tokens)  # sentinel fused token (token=tokens, slot=0)
    pad_w.zero_()
    pad_ids[: sorted_ids.numel()] = sorted_ids
    pad_w[: sorted_weights.numel()] = sorted_weights
    return pad_ids, pad_w


@pytest.mark.parametrize("tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, doweight_stage1", [
    (256, 4096, 2048, 17, 9, 64, 256, 128, False),
])
def test_moe_stage1(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
):
    assert model_dim % 64 == 0
    assert model_dim % tile_k == 0
    assert inter_dim % tile_n == 0

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Data: input and weights (aiter shapes)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32)
    # w2 is required by aiter CK API even for stage1; keep it allocated to avoid null ptr.
    # Stage1 kernels should not touch it, but we allocate a correct-shape tensor for safety.
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)

    # Routing: aiter uses fused_topk; we use torch topk+softmax for portability/determinism.
    score = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    # Prefer aiter moe_sorting buffers (exact CK routing format), fall back to torch sorting.
    sort_mode = os.environ.get("pyflir_MOE_SORT_MODE", "aiter" if HAS_AITER else "torch").lower()
    sorted_token_ids = sorted_weights = sorted_expert_ids = num_valid_ids = None
    if sort_mode == "aiter":
        res = _maybe_aiter_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts=experts,
            model_dim=model_dim,
            block_m=tile_m,
        )
        if res is None:
            pytest.skip("aiter moe_sorting unavailable in this environment; set pyflir_MOE_SORT_MODE=torch to run.", allow_module_level=False)
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res
    else:
        sorted_token_ids, sorted_weights, sorted_expert_ids = build_sorted_routing(
            topk_ids,
            topk_weights,
            num_experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
        )
        # num_valid_ids is total_tokens_post_pad; for our builder that's the full buffer length.
        num_valid_ids = torch.tensor([int(sorted_token_ids.numel())], device=device, dtype=torch.int32)

    # Pad to full blocks for safe indexing in our flir kernel (which uses grid_x = num_blocks).
    sorted_token_ids, sorted_weights = _pad_sorted_buffers_to_full_blocks(
        sorted_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        tokens=tokens,
        block_m=tile_m,
    )
    sorted_size = int(sorted_token_ids.numel())

    # FP8 per-token quantize (aiter config)
    x_fp8, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.float8_e4m3fnuz)  # [tokens, K], [tokens,1]
    w1_fp8, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.float8_e4m3fnuz)  # [E, 2*inter, K], [E,2*inter,1]
    # w2 is not used by our kernel, but required by CK stage1 API
    w2_fp8, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.float8_e4m3fnuz)

    # Preshuffle weights (aiter/CK layout)
    w1_shuffled = shuffle_weight(w1_fp8)
    w2_shuffled = shuffle_weight(w2_fp8)

    # Flatten W1 for our flir kernel (treat expert dim as part of N)
    w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim)
    w1_fp8_flat = w1_fp8.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)

    # Pad storage for forced global dwordx4 loads (same trick as existing GEMM tests)
    PAD_ELEMS = 256
    x_flat = x_fp8.contiguous().view(-1)
    x_storage = torch.empty(x_flat.numel() + PAD_ELEMS, device=device, dtype=x_fp8.dtype)
    x_storage[: x_flat.numel()] = x_flat
    x_fp8 = x_storage[: x_flat.numel()].view(tokens, model_dim)

    w_flat = w1_shuffled_flat.contiguous().view(-1)
    w_storage = torch.empty(w_flat.numel() + PAD_ELEMS, device=device, dtype=w1_shuffled.dtype)
    w_storage[: w_flat.numel()] = w_flat
    w1_shuffled_flat = w_storage[: w_flat.numel()].view(experts * (2 * inter_dim), model_dim)

    # Flatten scales to 1D memrefs
    scale_x_1d = scale_x.view(-1).contiguous()  # [tokens]
    scale_w1_1d = scale_w1_flat.view(-1).contiguous()  # [rows]
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.zeros((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    size_out = tokens * topk * inter_dim
    size_x = tokens * model_dim
    size_w = experts * (2 * inter_dim) * model_dim
    size_sorted = sorted_size
    size_expert_ids = int(sorted_expert_ids.numel())

    total_threads = 256
    elems_x_per_tile = tile_m * tile_k
    elems_per_thread_x = elems_x_per_tile // total_threads
    bytes_per_thread_x = elems_per_thread_x  # fp8
    x_split_lens = compute_split_load_lens(bytes_per_thread_x, max_bytes_per_load=16)
    num_x_loads = len(x_split_lens)

    pad_k = 8
    lds_stride = tile_k + pad_k

    class _MOE1(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_moe1"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_x_decl"] = allocator.allocate_array(ir.Float8E4M3FNType.get(), tile_m * lds_stride)
            allocator.finalize()

        @flir.kernel
        def kernel(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, ir.Float8E4M3FNType.get()),
            arg_w: lambda: T.memref(size_w, ir.Float8E4M3FNType.get()),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            f8 = ir.Float8E4M3FNType.get()
            f32 = ir.F32Type.get()
            i32 = ir.IntegerType.get_signless(32)
            vec4_f32 = ir.VectorType.get([4], f32)
            vec8_f8 = ir.VectorType.get([8], f8)
            vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))

            c0 = arith.constant(0, index=True)
            c4 = arith.constant(4, index=True)
            c1 = arith.constant(1, index=True)
            c16 = arith.constant(16, index=True)
            c64 = arith.constant(64, index=True)
            c256 = arith.constant(256, index=True)
            c1024 = arith.constant(1024, index=True)
            c_tile_k = arith.constant(tile_k, index=True)

            c0f = arith.constant(0.0, type=f32).value
            c1f = arith.constant(1.0, type=f32).value
            c_log2e = arith.constant(1.4426950408889634, type=f32).value  # log2(e)

            def silu(x):
                # sigmoid(x) = 1 / (1 + exp(-x)) using exp2
                neg = flir.arith.SubFOp(unwrap(c0f), unwrap(x)).result
                scaled = flir.arith.MulFOp(unwrap(neg), unwrap(c_log2e)).result
                exp_neg = mlir_math.exp2(unwrap(scaled))
                den = flir.arith.AddFOp(unwrap(c1f), unwrap(exp_neg)).result
                sig = flir.arith.DivFOp(unwrap(c1f), unwrap(den)).result
                return flir.arith.MulFOp(unwrap(x), unwrap(sig)).result

            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result

            # Layouts
            layout_x = flir.make_layout((tokens_in, k_in), stride=(k_in, 1))

            # B preshuffle layout (match GEMM preshuffle layout):
            # Shape: (N0, K0, KLane, NLane, KPack) = (N/16, K/64, 4, 16, 16)
            c_n_total = unwrap(arith.constant(experts * (2 * inter_dim), index=True))
            c_k0 = _arith_mlir.DivUIOp(unwrap(k_in), unwrap(c64)).result
            c_n0 = _arith_mlir.DivUIOp(unwrap(c_n_total), unwrap(c16)).result
            stride_n0 = _arith_mlir.MulIOp(unwrap(c_k0), unwrap(c1024)).result

            stride_b = (
                stride_n0,  # n0
                c1024,      # k0
                c256,       # k1 (KLane)
                c16,        # n1
                c1,         # k2
            )
            layout_b = flir.make_layout(
                (c_n0, c_k0, arith.constant(4, index=True), c16, c16),
                stride=stride_b,
            )

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # tile along sorted M
            by = gpu.block_id("y")  # tile along inter_dim

            base_ptr = allocator.get_base()
            lds_x = _state["lds_x_decl"](base_ptr).get()

            x_rsrc = buffer_ops.create_buffer_resource(arg_x)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out)
            sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights)

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = _arith_mlir.IndexCastOp(ir.IndexType.get(), unwrap(expert_i32)).result
            inter2_idx = arith.constant(2 * inter_dim, index=True)
            expert_off_idx = _arith_mlir.MulIOp(unwrap(expert_idx), unwrap(inter2_idx)).result  # index

            # Thread -> (row_a_local, col_a_local)
            tx_idx = unwrap(tx)
            vec_len_val = arith.constant(bytes_per_thread_x, index=True)
            linear_id = _arith_mlir.MulIOp(unwrap(tx_idx), unwrap(vec_len_val)).result
            row_a_local = _arith_mlir.DivUIOp(unwrap(linear_id), unwrap(c_tile_k)).result
            col_a_local = _arith_mlir.RemUIOp(unwrap(linear_id), unwrap(c_tile_k)).result

            bx_m = _arith_mlir.MulIOp(unwrap(bx), unwrap(arith.constant(tile_m, index=True))).result
            sorted_row = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_a_local)).result

            # Load fused token id and decode token (t) and topk slot (s)
            fused = buffer_ops.buffer_load(sorted_rsrc, sorted_row, vec_width=1, dtype=i32)
            mask24 = arith.i32(0xFFFFFF)._value
            t_i32 = _arith_mlir.AndIOp(unwrap(fused), unwrap(mask24)).result
            s_i32 = _arith_mlir.ShRUIOp(unwrap(fused), unwrap(arith.i32(24)._value)).result

            # valid token?
            tokens_i32 = arith.i32(tokens)._value
            topk_i32 = arith.i32(topk)._value
            valid_t = _arith_mlir.CmpIOp(_arith_mlir.CmpIPredicate.ult, unwrap(t_i32), unwrap(tokens_i32)).result
            t_idx = _arith_mlir.IndexCastOp(ir.IndexType.get(), unwrap(t_i32)).result

            # X base index (token-major)
            coord_x = flir.make_coord(unwrap(t_idx), unwrap(col_a_local))
            idx_x = flir.crd2idx(coord_x, layout_x)
            idx_x_div4 = _arith_mlir.DivUIOp(unwrap(idx_x), unwrap(c4)).result

            # A loads split (global dwordx4), masked by valid_t
            def load_x_split(idx_div4):
                return load_fp8_tile_split_dwordx4(
                    buffer_ops, flir, _arith_mlir, rsrc=x_rsrc, idx_div4=idx_div4, lens=x_split_lens, i32_type=i32, mask=valid_t
                )

            vec_x_parts = load_x_split(unwrap(idx_x_div4))

            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16

            row_x_lds = lane_mod_16

            # Dynamic N tiling within block (same as existing kernels)
            by_n = _arith_mlir.MulIOp(unwrap(by), unwrap(arith.constant(tile_n, index=True))).result
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = _arith_mlir.RemUIOp(unwrap(wave_id), unwrap(c4)).result
            n_tile_base = _arith_mlir.MulIOp(unwrap(wave_mod_4), unwrap(c_n_per_wave)).result

            # Precompute n_blk/n_intra for gate and up rows (index computations)
            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            inter_idx = arith.constant(inter_dim, index=True)
            for ni in range_constexpr(num_acc_n):
                offset = arith.constant(ni * 16, index=True)
                col_g = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                col_g = _arith_mlir.AddIOp(unwrap(col_g), unwrap(offset)).result
                col_g = _arith_mlir.AddIOp(unwrap(col_g), unwrap(lane_mod_16)).result
                col_g_list.append(col_g)

                row_gate = _arith_mlir.AddIOp(unwrap(expert_off_idx), unwrap(col_g)).result
                row_up = _arith_mlir.AddIOp(unwrap(row_gate), unwrap(inter_idx)).result

                n_intra_gate.append(_arith_mlir.RemUIOp(unwrap(row_gate), unwrap(c16)).result)
                n_blk_gate.append(_arith_mlir.DivUIOp(unwrap(row_gate), unwrap(c16)).result)
                n_intra_up.append(_arith_mlir.RemUIOp(unwrap(row_up), unwrap(c16)).result)
                n_blk_up.append(_arith_mlir.DivUIOp(unwrap(row_up), unwrap(c16)).result)

            m_repeat = tile_m // 16
            k_unroll = tile_k // 32

            # CK-style xor-with-modulo swizzle on K (16B granularity)
            def swizzle_idx(row_idx, col_idx):
                k_blocks16 = arith.constant(tile_k // 16, index=True)
                row_mod = _arith_mlir.RemUIOp(unwrap(row_idx), unwrap(k_blocks16)).result
                xor_mask = _arith_mlir.MulIOp(unwrap(row_mod), unwrap(c16)).result
                return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

            # Load one 8B (i64) B pack for one MFMA(x32) step (inline, GEMM-style).
            def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                k0_base = _arith_mlir.DivUIOp(unwrap(base_k), unwrap(c64)).result
                k0 = _arith_mlir.AddIOp(unwrap(k0_base), unwrap(arith.constant(ki_step // 2, index=True))).result
                k1 = lane_div_16  # 0..3
                half = ki_step % 2
                k2_base = arith.constant(half * 8, index=True)

                coord_b = flir.make_coord(
                    unwrap(blk_list[ni]),
                    unwrap(k0),
                    unwrap(k1),
                    unwrap(intra_list[ni]),
                    unwrap(k2_base),
                )
                idx_bytes = flir.crd2idx(coord_b, layout_b)
                idx_i32 = _arith_mlir.DivUIOp(unwrap(idx_bytes), unwrap(c4)).result
                b8 = buffer_ops.buffer_load(w_rsrc, idx_i32, vec_width=2, dtype=i32)
                b_vec64 = vector.BitCastOp(vec1_i64, b8).result
                return vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result

            acc_gate = [acc_init] * (num_acc_n * m_repeat)
            acc_up = [acc_init] * (num_acc_n * m_repeat)

            def emit_tile(k_iv, acc_gate_in, acc_up_in, vec_x_in_parts, is_last=False):
                # Store X to LDS
                store_off = 0
                for pi in range_constexpr(num_x_loads):
                    val_vec = vec_x_in_parts[pi]
                    curr_bytes = x_split_lens[pi]
                    curr_i32 = curr_bytes // 4

                    col0 = _arith_mlir.AddIOp(unwrap(col_a_local), unwrap(arith.constant(store_off, index=True))).result
                    col_sw = swizzle_idx(row_a_local, col0)
                    coord_s = flir.make_coord(unwrap(row_a_local), unwrap(col_sw))
                    idx_s = flir.crd2idx(coord_s, layout_lds)

                    if curr_i32 == 4:
                        vec16_f8 = ir.VectorType.get([16], f8)
                        v16 = vector.BitCastOp(vec16_f8, val_vec).result
                        vector.StoreOp(v16, lds_x, [unwrap(idx_s)])
                    else:
                        vec8 = ir.VectorType.get([8], f8)
                        v2 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                        v8 = vector.BitCastOp(vec8, v2).result
                        vector.StoreOp(v8, lds_x, [unwrap(idx_s)])
                    store_off += curr_bytes

                gpu.barrier()

                vec_x_next = vec_x_in_parts
                if not is_last:
                    next_k = _arith_mlir.AddIOp(unwrap(k_iv), unwrap(c_tile_k)).result
                    next_k_div4 = _arith_mlir.DivUIOp(unwrap(next_k), unwrap(c4)).result
                    next_idx = _arith_mlir.AddIOp(unwrap(idx_x_div4), unwrap(next_k_div4)).result
                    vec_x_next = load_x_split(next_idx)

                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)

                # Inline MFMA hot loop (match fast GEMM K32 pattern: ki_step outer -> mi inner).
                col_offset_base = _arith_mlir.MulIOp(unwrap(lane_div_16), unwrap(c16)).result
                c0_i32 = arith.i32(0)._value

                for ki_step in range_constexpr(k_unroll):
                    # Load B packs (gate+up) once per ki_step, reuse across MI.
                    b_gate_packs = []
                    b_up_packs = []
                    for ni in range_constexpr(num_acc_n):
                        b_gate_packs.append(load_b_pack(k_iv, ki_step, ni, n_blk_gate, n_intra_gate))
                        b_up_packs.append(load_b_pack(k_iv, ki_step, ni, n_blk_up, n_intra_up))

                    ki64 = (ki_step // 2) * 64
                    ki64_val = arith.constant(ki64, index=True)
                    half = ki_step % 2
                    half_off = arith.constant(half * 8, index=True)
                    col_base = _arith_mlir.AddIOp(unwrap(col_offset_base), unwrap(ki64_val)).result

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row = _arith_mlir.AddIOp(unwrap(row_x_lds), unwrap(mi_val)).result

                        col_base_sw = swizzle_idx(curr_row, col_base)
                        col_sw = _arith_mlir.AddIOp(unwrap(col_base_sw), unwrap(half_off)).result
                        coord_a = flir.make_coord(unwrap(curr_row), unwrap(col_sw))
                        idx_a = flir.crd2idx(coord_a, layout_lds)

                        loaded_a8 = vector.LoadOp(vec8_f8, lds_x, [unwrap(idx_a)]).result
                        a_vec64 = vector.BitCastOp(vec1_i64, loaded_a8).result
                        a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            gate_list[acc_idx] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a_pack),
                                    unwrap(b_gate_packs[ni]),
                                    unwrap(gate_list[acc_idx]),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            up_list[acc_idx] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a_pack),
                                    unwrap(b_up_packs[ni]),
                                    unwrap(up_list[acc_idx]),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result

                gpu.barrier()
                return gate_list, up_list, vec_x_next

            c_k_main = _arith_mlir.SubIOp(unwrap(k_in), unwrap(c_tile_k)).result
            for k_iv in range(c0, c_k_main, c_tile_k):
                acc_gate, acc_up, vec_x_parts = emit_tile(k_iv, acc_gate, acc_up, vec_x_parts, is_last=False)
            acc_gate, acc_up, _ = emit_tile(c_k_main, acc_gate, acc_up, vec_x_parts, is_last=True)

            # Store epilogue to out[t, slot, inter]
            # Recompute token/slot for each output row this lane writes.
            for mi in range_constexpr(m_repeat):
                mi_base = arith.constant(mi * 16, index=True)
                for ii in range_constexpr(4):
                    row_off = _arith_mlir.AddIOp(unwrap(_arith_mlir.MulIOp(unwrap(lane_div_16), unwrap(c4)).result), unwrap(arith.constant(ii, index=True))).result
                    row_in_tile = _arith_mlir.AddIOp(unwrap(mi_base), unwrap(row_off)).result
                    sorted_row2 = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_in_tile)).result

                    fused2 = buffer_ops.buffer_load(sorted_rsrc, sorted_row2, vec_width=1, dtype=i32)
                    t2 = _arith_mlir.AndIOp(unwrap(fused2), unwrap(mask24)).result
                    s2 = _arith_mlir.ShRUIOp(unwrap(fused2), unwrap(arith.i32(24)._value)).result
                    valid2 = _arith_mlir.CmpIOp(_arith_mlir.CmpIPredicate.ult, unwrap(t2), unwrap(tokens_i32)).result

                    t2_idx = _arith_mlir.IndexCastOp(ir.IndexType.get(), unwrap(t2)).result
                    sx = buffer_ops.buffer_load(sx_rsrc, t2_idx, vec_width=1, dtype=f32, mask=valid2)

                    # Sorted weight aligned with `sorted_row2` (matches aiter moe_sorting output)
                    tw = buffer_ops.buffer_load(sorted_w_rsrc, sorted_row2, vec_width=1, dtype=f32, mask=valid2)

                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        # bounds: col_g < inter_dim
                        valid_col = _arith_mlir.CmpIOp(_arith_mlir.CmpIPredicate.ult, unwrap(col_g), unwrap(inter_idx)).result
                        valid_store = _arith_mlir.AndIOp(unwrap(valid2), unwrap(valid_col)).result

                        col_i32 = _arith_mlir.IndexCastOp(i32, unwrap(col_g)).result
                        # For scale lookups we want linear row indices in [0, experts*2*inter_dim),
                        # and `buffer_load` will cast index->i32 if needed.
                        row_gate_idx = _arith_mlir.AddIOp(unwrap(expert_off_idx), unwrap(col_g)).result
                        row_up_idx = _arith_mlir.AddIOp(unwrap(row_gate_idx), unwrap(inter_idx)).result
                        row_gate = row_gate_idx
                        row_up = row_up_idx
                        sw_gate = buffer_ops.buffer_load(sw_rsrc, row_gate, vec_width=1, dtype=f32, mask=valid_store)
                        sw_up = buffer_ops.buffer_load(sw_rsrc, row_up, vec_width=1, dtype=f32, mask=valid_store)

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.ExtractOp(acc_gate[acc_idx], [], [ii]).result
                        vu = vector.ExtractOp(acc_up[acc_idx], [], [ii]).result

                        vg = flir.arith.MulFOp(unwrap(vg), unwrap(sx)).result
                        vg = flir.arith.MulFOp(unwrap(vg), unwrap(sw_gate)).result
                        vu = flir.arith.MulFOp(unwrap(vu), unwrap(sx)).result
                        vu = flir.arith.MulFOp(unwrap(vu), unwrap(sw_up)).result

                        y = flir.arith.MulFOp(unwrap(silu(vg)), unwrap(vu)).result
                        if doweight_stage1:
                            y = flir.arith.MulFOp(unwrap(y), unwrap(tw)).result

                        y_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(y)).result

                        # out linear index = ((t*topk + s)*inter_dim + col)
                        inter_i32_local = arith.i32(inter_dim)._value
                        idx0 = _arith_mlir.MulIOp(unwrap(t2), unwrap(topk_i32)).result
                        idx0 = _arith_mlir.AddIOp(unwrap(idx0), unwrap(s2)).result
                        idx0 = _arith_mlir.MulIOp(unwrap(idx0), unwrap(inter_i32_local)).result
                        idx_out = _arith_mlir.AddIOp(unwrap(idx0), unwrap(col_i32)).result

                        buffer_ops.buffer_store(y_f16, out_rsrc, idx_out, mask=valid_store)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, ir.Float8E4M3FNType.get()),
            arg_w: lambda: T.memref(size_w, ir.Float8E4M3FNType.get()),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(256, index=True).value
            gx = arith.constant(size_expert_ids, index=True).value
            gy = arith.constant(inter_dim // tile_n, index=True).value
            flir.gpu_ext.LaunchFuncOp(
                ["mfma_moe1", "kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    unwrap(arg_out),
                    unwrap(arg_x),
                    unwrap(arg_w),
                    unwrap(arg_scale_x),
                    unwrap(arg_scale_w),
                    unwrap(arg_sorted_token_ids),
                    unwrap(arg_expert_ids),
                    unwrap(arg_sorted_weights),
                    unwrap(tokens_in),
                    unwrap(inter_in),
                    unwrap(k_in),
                ],
            )

    m = _MOE1()
    exe = pyflir.compile(m)

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        exe(o, x, w, sx, sw, st, eids, sw_sorted, tokens, inter_dim, model_dim)

    _, us = run_perftest(
        launch,
        out,
        x_fp8,
        w1_shuffled_flat,
        scale_x_1d,
        scale_w1_1d,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_iters=5,
        num_warmup=2,
    )
    torch.cuda.synchronize()

    ref = torch_stage1_ref(
        x_fp8,
        w1_fp8_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )

    assert verify_output(out.to(torch.float32), ref, rtol=0.25, atol=0.25)

    # Compare + benchmark vs aiter CK stage1 (optional; enabled by default when aiter is runnable).
    compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    if compare_ck:
        if not HAS_AITER:
            # `run_tests.sh` executes this file directly (not under pytest), so
            # raising pytest.skip would be treated as a failure. Treat this as an
            # optional comparison and simply disable it when aiter isn't present.
            print("[aiter CK] not available; skipping CK comparison for moe stage1.")
            compare_ck = False
        try:
            from aiter.ops.moe_op import ck_moe_stage1_fwd
            from aiter.ops.enum import QuantType, ActivationType
            import inspect

            out_ck = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

            # aiter CK expects w1/w2 with expert dimension preserved.
            w1_ck = w1_shuffled
            w2_ck = w2_shuffled
            w1_scale_ck = scale_w1.contiguous()

            _sig = None
            try:
                _sig = inspect.signature(ck_moe_stage1_fwd)
            except Exception:
                _sig = None
            _supports_splitk = bool(_sig and ("splitk" in _sig.parameters))

            def launch_ck(o, x, w1_, w2_, sorted_ids_, sorted_eids_, num_valid_, w1_scale_, a1_scale_, sorted_w_):
                kwargs = dict(
                    hidden_states=x,
                    w1=w1_,
                    w2=w2_,
                    sorted_token_ids=sorted_ids_,
                    sorted_expert_ids=sorted_eids_,
                    num_valid_ids=num_valid_,
                    out=o,
                    topk=topk,
                    kernelName="",
                    w1_scale=w1_scale_,
                    a1_scale=a1_scale_,
                    block_m=tile_m,
                    sorted_weights=sorted_w_ if doweight_stage1 else None,
                    quant_type=QuantType.per_Token,
                    activation=ActivationType.Silu,
                    dst_type=torch.float16,
                )
                # Older/newer aiter builds may not support `splitk`. Gate it by signature.
                if _supports_splitk:
                    kwargs["splitk"] = 1
                ck_moe_stage1_fwd(**kwargs)

            # Benchmark CK stage1
            _, us_ck = run_perftest(
                launch_ck,
                out_ck,
                x_fp8,
                w1_ck,
                w2_ck,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                w1_scale_ck,
                scale_x,
                sorted_weights,
                num_iters=5,
                num_warmup=2,
            )

            # Correctness: flir vs CK
            assert verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.25, atol=0.25)

            # Perf print: use the same flop model for both
            flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(f"[aiter CK] stage1: {us_ck:.1f} us, {tflops_ck:.2f} TFLOPS")
        except Exception as e:
            # Optional comparison: don't fail the main FLIR correctness/perf test
            # if the external reference is not compatible with this environment.
            print(f"[aiter CK] skipping CK comparison for moe stage1: {e}")
            compare_ck = False
    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    bytes_moved += tokens * model_dim * 1  # x fp8
    bytes_moved += experts * (2 * inter_dim) * model_dim * 1  # w fp8
    bytes_moved += tokens * topk * inter_dim * 2  # out fp16
    bytes_moved += tokens * 4  # scale_x f32 (1D)
    bytes_moved += experts * (2 * inter_dim) * 4  # scale_w f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4  # sorted_weights f32
    bytes_moved += int(sorted_token_ids.numel()) * 4  # sorted_token_ids i32
    bytes_moved += int(sorted_expert_ids.numel()) * 4  # sorted_expert_ids i32
    tbps = bytes_moved / 1e12 / (us / 1e6)

    print(
        f"MoE stage1: {us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s (doweight_stage1={doweight_stage1})"
    )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    # Run the aiter-aligned FP8 per-token stage1 test.
    test_moe_stage1(tokens=2048, model_dim=4096, inter_dim=2048, experts=17, topk=9, tile_m=64, tile_n=256, tile_k=128, doweight_stage1=False)



