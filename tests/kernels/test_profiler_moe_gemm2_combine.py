# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end acceptance test for MoE GEMM2 + combine.

Used to validate the fused_gemm2_combine operator's accuracy and performance.

Pipeline (dispatch is simplified away):

    [setup, one-shot]  disp_op.dispatch(inp, wts, idx)
                       (populates shmem_disp_out_idx / out_wts / total_recv;
                        these shmem tables are combine's inputs, so once
                        seeded in setup the chain can read them directly
                        and we don't re-run dispatch every iteration.)

    [chain, captured]
        baseline:  moe_gemm2  ->  combine
        fused:     fused_gemm2_combine

Measurement x execution modes (4 orthogonal combos; only profile+cudagraph
is implemented here):
  --mode       profile (torch.profiler) | bench (CUDA Event) | verify
  --cudagraph  off=eager / on=CUDAGraph capture+replay

  Implemented:
    profile + cudagraph    (the core acceptance path)
    verify                  (skeleton; full diff logic TBD)
  The remaining 3 combos raise NotImplementedError.

Invocation:
  # Baseline only (when the fused op is unwired)
  torchrun --nproc_per_node=8 tests/kernels/test_profiler_moe_gemm2_combine.py \
      --mode profile --cudagraph --bench-op baseline \
      --max-tokens 512 --hidden-dim 7168 --inter-dim 4096 --k 8

  # Baseline + fused comparison
  torchrun --nproc_per_node=8 tests/kernels/test_profiler_moe_gemm2_combine.py \
      --mode profile --cudagraph --bench-op both

CUDAGraph notes:
  - Once dispatch has run in setup, total_recv / out_idx / etc. live in shmem
    buffers; GEMM2 / combine in the chain read these static views directly,
    which captures cleanly.
  - GEMM2 captures use the max_recv upper bound + num_valid_ids for early exit.
  - **No dist.barrier() between replays** -- ROCTracer instrumentation and
    P2P shmem ops would otherwise interfere and inflate kernel time massively.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "fp8_ocp": torch.float8_e4m3fn,
    "fp8_fnuz": torch.float8_e4m3fnuz,
    "fp4": torch.float4_e2m1fn_x2,
}

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in [_ROOT, "/home/yashao/FlyDSL/python", "/home/yashao/mori/python"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms

from kernels.dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

# Reduction kernel for GEMM2 reduce (non-atomic) mode: sums the expanded
# [max_recv*topk, model_dim] plain-store scratch over the topk dim back into
# the token-level [max_recv, model_dim] layout that combine expects. This is
# the same kernel the production aiter path uses (compile_moe_gemm2_ex /
# flydsl_moe_stage2 reduce mode), so the harness can benchmark reduce-mode
# GEMM2 -> combine without forcing --gemm2-accumulate.
try:
    from kernels.moe_gemm_2stage import compile_moe_reduction  # type: ignore

    HAS_MOE_REDUCTION = True
    _MOE_REDUCTION_ERR = ""
except Exception as _e:  # noqa: BLE001
    compile_moe_reduction = None  # type: ignore[assignment]
    HAS_MOE_REDUCTION = False
    _MOE_REDUCTION_ERR = repr(_e)

# Preshuffle helpers — required so the GEMM2 kernel reads W2 in the layout
# it was compiled to expect.  Without this both fp4 and fp8 weights are
# interpreted as garbage and the dot-product collapses to ~0, masking any
# real numerical mismatch in downstream stages (the verify path then sees
# 0 == 0 and falsely reports PASS).
try:
    from tests.utils import shuffle_weight  # type: ignore
except Exception:  # noqa: BLE001
    shuffle_weight = None  # type: ignore[assignment]
try:
    from tests.kernels.utils import fp4_utils  # type: ignore
except Exception:  # noqa: BLE001
    fp4_utils = None  # type: ignore[assignment]

try:
    from kernels.mega_moe import (  # type: ignore
        MegaMoeStage2,
    )
    # Module-level READY=False means the file is in place but the kernel
    # implementation isn't wired up; the test gracefully skips the fused
    # path instead of erroring.
    HAS_FUSED_OP = bool(getattr(MegaMoeStage2, "READY", False))
    _FUSED_IMPORT_ERR = "" if HAS_FUSED_OP else "MegaMoeStage2.READY = False (kernel not yet wired)"
except Exception as _e:  # noqa: BLE001
    MegaMoeStage2 = None
    HAS_FUSED_OP = False
    _FUSED_IMPORT_ERR = repr(_e)

# moe_sorting: low-level in-place API, cudagraph-friendly.
# Uses the in-tree FlyDSL kernel (kernels/moe_sorting_kernel.py); the API is
# sorted_token_ids encodes (j_global<<24)|t.
try:
    from kernels.moe_sorting_kernel import moe_sorting_flydsl as _flydsl_moe_sorting_fwd  # type: ignore

    def _moe_sorting_fwd(topk_ids, topk_weights,
                         sorted_token_ids, sorted_weights, sorted_expert_ids,
                         num_valid_ids, moe_buf,
                         num_experts, unit_size, expert_mask,
                         num_local_tokens, _unused=0):
        # Passing num_local_tokens=None makes the kernel use the static
        # topk_ids.shape[0] for M; this avoids a host-side .item() call and
        # keeps the launch cudagraph-capturable. The actual valid-token
        # count is recovered downstream from num_valid_ids written by the
        # device kernel.
        _flydsl_moe_sorting_fwd(
            topk_ids, topk_weights,
            sorted_token_ids, sorted_weights, sorted_expert_ids,
            num_valid_ids, moe_buf,
            num_experts, unit_size, expert_mask,
            None,
        )
    HAS_MOE_SORTING = True
    _MOE_SORTING_ERR = ""
except Exception as _e:  # noqa: BLE001
    _moe_sorting_fwd = None  # type: ignore[assignment]
    HAS_MOE_SORTING = False
    _MOE_SORTING_ERR = repr(_e)


# --- Distributed init ---------------------------------------------------
def setup_distributed(rank, world_size, master_port=29700):
    if "LOCAL_RANK" not in os.environ:
        os.environ.update({
            "LOCAL_RANK": str(rank), "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        })
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank, world_size=world_size, device_id=dev,
    )
    import torch._C._distributed_c10d as c10d
    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return local_rank, world_size


def cleanup():
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


# --- Data / routing construction ----------------------------------------
def _build_dispatch_inputs(rank, world_size, dev, args, cfg):
    """Build dispatch inputs (input/weights/indices) with a fixed seed for
    cross-rank reproducibility."""
    cur_tok = args.max_tokens
    k       = args.k
    n_exp   = world_size * args.num_experts_per_rank
    epr     = args.num_experts_per_rank

    torch.manual_seed(42 + rank)
    if cfg.data_type == torch.float4_e2m1fn_x2:
        inp = torch.randint(
            0, 256, (cur_tok, cfg.hidden_dim // 2),
            dtype=torch.uint8, device=dev,
        ).view(torch.float4_e2m1fn_x2)
    elif cfg.data_type in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        inp = torch.randn(
            cur_tok, cfg.hidden_dim, dtype=torch.bfloat16, device=dev,
        ).to(cfg.data_type)
    else:
        inp = torch.randn(
            cur_tok, cfg.hidden_dim, dtype=cfg.data_type, device=dev,
        )

    wts = torch.rand(cur_tok, k, dtype=torch.float32, device=dev)
    wts = wts / wts.sum(-1, keepdim=True)

    idx = torch.zeros(cur_tok, k, dtype=torch.int32, device=dev)
    routing = getattr(args, "routing", "random")
    if routing == "atomic1_8pe":
        # Deterministic uniform routing (equivalent to a balanced EP under
        # production; atomic_per_pe=1, num_pe=k=8):
        #   idx[t, j] = dest_pe * epr + local_eid
        #   dest_pe   = (rank + t + j) % world_size      -> each token's k
        #                                                   experts hit k
        #                                                   distinct PEs;
        #                                                   the per-rank/per-
        #                                                   token start is
        #                                                   staggered to
        #                                                   avoid hotspots.
        #   local_eid = ((rank*cur_tok + t)*k + j) % epr -> the rank/t/j
        #                                                   triple-loop
        #                                                   covers every
        #                                                   (PE, local_e).
        # The cur_tok*k (t,j) pairs of one rank are spread uniformly over
        # [0, ws*epr) -> per-expert hit count ~= cur_tok*k / (ws*epr).
        # With cur_tok=32, ws=8, epr=32, k=8 this hits 1.0 -- perfectly
        # balanced. The same token's k experts always land on k distinct
        # PEs, so dispatch dedup never fires (dup_ballot == 0) and every
        # (T, j_global) makes it into sorted_token_ids.
        # Name: atomic1_8pe = a given (t, *) output row on each dest_pe
        # takes exactly 1 atomic_fadd (atomic_per_pe=1) and the token is
        # spread across 8 distinct PEs.
        for t in range(cur_tok):
            for j in range(k):
                dest_pe   = (rank + t + j) % world_size
                local_eid = ((rank * cur_tok + t) * k + j) % epr
                idx[t, j] = dest_pe * epr + local_eid
        return inp, wts, idx

    if routing == "atomic8_1pe":
        # Scenario 2: all k experts of one token land on the same dest_pe,
        # creating the GEMM2 atomic-contention worst case (k=8 atomics on the
        # same (t, *) row) while keeping inter-/intra-rank load balanced:
        #   - global token order g = rank * cur_tok + t in [0, world_size*cur_tok)
        #   - dest_pe = g % world_size           -> inter-rank balance (each
        #                                            PE receives cur_tok tokens)
        #   - eid_base = (g // world_size) % epr -> the token's relative
        #     position on *this dp* once sorted by g (0..cur_tok*npes-1) mod
        #     epr; this lets each of the 8 ranks contribute cur_tok/npes
        #     tokens that together cover all epr local_eids on the dp,
        #     giving true intra-rank balance (each local_eid is hit
        #     cur_tok*npes*k/(npes*epr) = cur_tok*k/epr times).
        # local_eid[j] = (eid_base + j) % epr -> each token uses k contiguous
        # experts.
        # Name: atomic8_1pe = a token's k=8 experts all land on one dest_pe,
        # so the same (t, *) output row on that PE is hit by k=8 atomic_fadds
        # (atomic-k worst case).
        if epr % k != 0:
            raise ValueError(
                f"routing=atomic8_1pe requires epr ({epr}) divisible by k ({k}) "
                "to keep per-expert load balanced; got mismatch."
            )
        for t in range(cur_tok):
            g          = rank * cur_tok + t
            dest_pe    = g % world_size
            eid_base   = (g // world_size) % epr
            for j in range(k):
                local_eid = (eid_base + j) % epr
                idx[t, j] = dest_pe * epr + local_eid
        return inp, wts, idx

    if routing == "atomic2_4pe":
        # Scenario 3: each token's k experts are spread across *4 distinct*
        # PEs, hitting each PE atomic_per_pe = k/4 times (k=8 -> 2). That is,
        # the same GEMM2 output row sees only 2 atomic_fadds -- a medium-
        # contention scenario between atomic1_8pe (atomic=1) and atomic8_1pe
        # (atomic=k=8 worst case).
        #
        # Inter-/intra-rank balance construction:
        #   - global token order g = rank * cur_tok + t in [0, world_size*cur_tok)
        #   - PE: base_pe = g % world_size;
        #     dest_pe[j] = (base_pe + j_group) % world_size, j_group =
        #     j // atomic_per_pe in [0, 4). A token's 4 PEs are
        #     {base, base+1, base+2, base+3} (mod ws), always 4 distinct
        #     (requires world_size >= 4). base_pe round-robins across
        #     [0, ws), each PE receives num_pe * cur_tok (rank, t, j_group)
        #     triples (= 4*cur_tok per rank), total atomics per PE per rank
        #     = atomic_per_pe * num_pe * cur_tok = k*cur_tok, globally
        #     ws * cur_tok * k / ws = cur_tok*k -> perfect inter-rank balance.
        #   - local_eid = (g // world_size + j) % epr
        #     Key: use g // ws (not g % ws) as eid_base to avoid local_eid
        #     "collapse" -- under the dest_pe lattice (g%ws), if the r
        #     coefficient in local_eid shares a factor with epr (e.g.
        #     coeff=ws=8, epr=32, gcd=8) we'd cover only epr/gcd=4 distinct
        #     local_eids. Using g//ws (which sweeps [0, cur_tok)) combined
        #     with the j offset spreads cur_tok*k = 256 atomics uniformly
        #     across all epr=32 local_eids on each PE -- every (PE, local_eid)
        #     cell hits cur_tok*k/epr = 8 times (perfectly balanced under
        #     default cur_tok=32, k=8, epr=32).
        #
        # Constraints: k % 4 == 0 (default 8/4=2 OK); world_size >= 4
        # (default 8 OK).
        num_pe = 4
        if k % num_pe != 0:
            raise ValueError(
                f"routing=atomic2_4pe requires k ({k}) divisible by {num_pe}; "
                "got mismatch (atomic_per_pe = k/4 must be integer)."
            )
        if world_size < num_pe:
            raise ValueError(
                f"routing=atomic2_4pe requires world_size >= {num_pe}; "
                f"got world_size={world_size}."
            )
        atomic_per_pe = k // num_pe  # k=8 -> 2
        for t in range(cur_tok):
            g        = rank * cur_tok + t
            base_pe  = g %  world_size
            eid_base = g // world_size
            for j in range(k):
                j_group   = j // atomic_per_pe              # 0..num_pe-1 (4 PE slots)
                dest_pe   = (base_pe + j_group) % world_size
                local_eid = (eid_base + j) % epr
                idx[t, j] = dest_pe * epr + local_eid
        return inp, wts, idx

    # routing == "random" (legacy behaviour)
    if k <= world_size:
        for t in range(cur_tok):
            pes = torch.randperm(world_size, device=dev)[:k]
            for j in range(k):
                idx[t, j] = pes[j] * epr + torch.randint(
                    0, epr, (1,), device=dev,
                )
    else:
        for t in range(cur_tok):
            idx[t] = torch.randperm(n_exp, device=dev)[:k]
    return inp, wts, idx


def _build_gemm2_static_inputs(rank, world_size, dev, args, cfg, *, disp_op,
                               force_zero_copy_out=None):
    """Build the *static* inputs for GEMM2 / combine (unchanged during capture).

    Layout (production-aligned, topk=k):
      - kernel sees ``tokens_in`` = max_recv = world_size * bs (post-dedup
        upper bound).
      - A2 (input) capacity = [max_recv * topk, inter_dim] -- each (t, s)
        occupies its own row; the kernel addresses A2 via ``t * topk + s``.
      - a2_scale (e8m0): [max_recv * topk, inter_dim/32], sized for worst case.
      - W2 (weights): [num_experts_per_rank, model_dim, inter_dim], per-expert FP4.
      - sorted_token_ids is produced by the FlyDSL moe_sorting kernel from the
        actual dispatch routing (the call is captured into the chain and
        rerun each replay); entries follow the (s<<24)|t convention,
        s = j_global in [0, k).
      - accumulate=True -> output buffer = [max_recv, model_dim] (same-t
        slots atomic_fadd onto one row); accumulate=False -> output expands
        to [max_recv*topk, ...].

    Parameters
    ----------
    disp_op
        Required. The first sorting in setup pulls the real routing table
        from ``disp_op.shmem_disp_out_idx / shmem_disp_out_wts``, so one
        dispatch must already have run before calling this function.
    """
    epr        = args.num_experts_per_rank
    max_recv   = world_size * args.max_num_inp_token_per_rank
    model_dim  = args.hidden_dim   # GEMM2 output dim = combine's hidden_dim
    inter_dim  = args.inter_dim
    tile_m     = args.tile_m2
    a_dtype    = args.gemm2_a_dtype
    b_dtype    = args.gemm2_b_dtype
    gemm2_topk = max(1, int(args.k))   # GEMM2 compile-time topk
    # A2 capacity: worst case where every recv token is hit by topk local experts.
    a2_rows    = max_recv * gemm2_topk

    # A2 micro-scale capacity: size to moe_sorting's padded token upper bound
    # (not a2_rows) so the GEMM2 scale-buffer read (num_valid_ids[0]*K/32) can
    # never go OOB and decode garbage e8m0 into a GEMM2 overflow.
    _num_experts_global = world_size * epr
    _scale_capacity_rows = max_recv * gemm2_topk + _num_experts_global * tile_m

    torch.manual_seed(123 + rank)

    # e8m0 micro-scale = 127 means 2^0 = 1.0; headroom>0 lowers to 127-h
    # (= 2^-h) and scales both a2 and w2 -- the GEMM2 output then scales
    # by 2^-2h, avoiding fp4*fp4 random overflow past fp8 e4m3 max=+/-448
    # (causes NaN). headroom is set automatically in run_acceptance (4 for
    # verify+fp8_direct_cast, else 0); headroom=0 preserves legacy behaviour.
    # Clamp [0, 127] so the e8m0 encoding never becomes
    # 0 (= 2^-127 ~= 0), which would collapse entire columns.
    _e8m0_headroom = max(0, min(127, int(getattr(args, "gemm2_scale_headroom", 0))))
    _e8m0_val = 127 - _e8m0_headroom

    # A2: [max_recv * topk, inter_dim] (fp8 / bf16 / fp4)
    if a_dtype == "fp8":
        a2_view = (
            torch.randn(a2_rows, inter_dim, dtype=torch.bfloat16, device=dev)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        a2_storage = a2_view.view(-1)  # 1D for kernel
        # GEMM2 uses mfma_scale_f32_16x16x128_f8f6f4; both fp4 and fp8 paths
        # consume per-32-element e8m0 micro-scales (1 byte each), and the
        # kernel's sx_rsrc `num_records_bytes` is sized as
        # `num_valid_ids * (k_in/32)`. Feeding f32 here had two problems:
        #   1. the buffer is ~128x smaller than the descriptor, so OOB checks
        #      no longer protect us — the LLVM scale layout walks past the
        #      end of the actual allocation and faults.
        #   2. mfma_scale_* would interpret the f32 bits as 4 packed e8m0
        #      bytes anyway, silently producing garbage results.
        a2_scale_1d = torch.full(
            (_scale_capacity_rows * (inter_dim // 32),), _e8m0_val,
            dtype=torch.uint8, device=dev,
        )
    elif a_dtype == "fp4":
        # FP4 placeholder: 2 fp4 elements / byte.
        a2_view = torch.randint(
            0, 256, (a2_rows, inter_dim // 2), dtype=torch.uint8, device=dev,
        )
        a2_storage = a2_view.view(-1)
        # 1x32 group scale (e8m0: 127 = 2^0 = 1.0; using 0 zeroes GEMM2 output).
        # Sized to the padded sort total (see _scale_capacity_rows) so the
        # kernel's sx_rsrc descriptor never reads past the tensor end.
        a2_scale_1d = torch.full(
            (_scale_capacity_rows * (inter_dim // 32),), _e8m0_val, dtype=torch.uint8, device=dev,
        )
    elif a_dtype in ("bf16", "fp16"):
        torch_a = torch.bfloat16 if a_dtype == "bf16" else torch.float16
        a2_view = torch.randn(a2_rows, inter_dim, dtype=torch_a, device=dev)
        a2_storage = a2_view.view(-1)
        a2_scale_1d = torch.empty((0,), dtype=torch.float32, device=dev)
    else:
        raise ValueError(f"unsupported gemm2_a_dtype={a_dtype!r}")

    # W2: [epr, model_dim, inter_dim]
    if b_dtype == "fp4":
        # Important: the FP4 GEMM2 kernel expects W2 already preshuffled into
        # an MFMA-friendly layout, with scales already e8m0_shuffle'd. Feeding
        # raw random uint8 in here would make every MFMA dot product see
        # mis-laid-out 4-bit "logical fp4 elements"; many bit patterns get
        # treated as 0 by the hardware/runtime (subnormal flush), collapsing
        # entire output columns to 0 -- the original cause of the verify
        # path seeing base_out_tok all zeros.
        w2_raw = torch.randint(
            0, 256, (epr, model_dim, inter_dim // 2),
            dtype=torch.uint8, device=dev,
        )
        if shuffle_weight is not None:
            w2_storage = (
                shuffle_weight(w2_raw.view(torch.float4_e2m1fn_x2))
                .view(torch.uint8)
                .contiguous()
                .view(-1)
            )
        else:
            w2_storage = w2_raw.view(-1)

        # 1x32 e8m0 scale, shape: [epr, model_dim, inter_dim/32].
        # Note: e8m0 = 127 means 2^0 = 1.0; using 0 (= 2^-127 ~= 0) would
        # collapse entire GEMM2 output columns to 0 and produce false-PASS
        # 0=0 verify results.
        w2_scale_2d = torch.full(
            (epr * model_dim, inter_dim // 32), _e8m0_val,
            dtype=torch.uint8, device=dev,
        )
        if fp4_utils is not None:
            w2_scale_1d = (
                fp4_utils.e8m0_shuffle(w2_scale_2d)
                .view(torch.uint8)
                .contiguous()
                .view(-1)
            )
        else:
            w2_scale_1d = w2_scale_2d.view(-1)
    elif b_dtype == "fp8":
        w2_storage = (
            torch.randn(epr, model_dim, inter_dim, dtype=torch.bfloat16, device=dev)
            .to(torch.float8_e4m3fn)
            .contiguous()
            .view(-1)
        )
        # Same e8m0 layout as fp4 (see the fp8 a2_scale comment): GEMM2 sizes
        # sw_rsrc as `experts*model_dim * (k_in/32)` 1-byte e8m0 micro-scales,
        # so we have to allocate exactly that and (optionally) push it through
        # the same e8m0_shuffle the fp4 path uses.
        w2_scale_2d_f8 = torch.full(
            (epr * model_dim, inter_dim // 32), _e8m0_val,
            dtype=torch.uint8, device=dev,
        )
        if fp4_utils is not None:
            w2_scale_1d = (
                fp4_utils.e8m0_shuffle(w2_scale_2d_f8)
                .view(torch.uint8)
                .contiguous()
                .view(-1)
            )
        else:
            w2_scale_1d = w2_scale_2d_f8.view(-1)
    elif b_dtype in ("bf16", "fp16"):
        torch_b = torch.bfloat16 if b_dtype == "bf16" else torch.float16
        w2_storage = (
            torch.randn(epr, model_dim, inter_dim, dtype=torch_b, device=dev)
            .contiguous()
            .view(-1)
        )
        w2_scale_1d = torch.empty((0,), dtype=torch.float32, device=dev)
    else:
        raise ValueError(f"unsupported gemm2_b_dtype={b_dtype!r}")

    # FlyDSL moe_sorting with real routing.
    # Inputs:  disp_op.shmem_disp_out_idx [mr, k] i32 (global expert ids)
    #          disp_op.shmem_disp_out_wts [mr, k] f32
    # Outputs: sorted_token_ids[max_padded] i32 = (j_global<<24)|t
    #          sorted_expert_ids[max_blocks] i32 = local_expert_id_per_block
    #          sorted_weights[max_padded] f32
    #          num_valid_ids[2] i32 = [padding-after-total, num_input_tokens]
    if not HAS_MOE_SORTING:
        raise RuntimeError(
            f"moe_sorting kernel is required (real routing only); "
            f"import failed: {_MOE_SORTING_ERR}"
        )
    if disp_op is None:
        raise RuntimeError(
            "moe_sorting requires disp_op (must dispatch once before build)"
        )
    num_experts_global = world_size * epr
    # Sorting upper bound (op_tests/test_moe_sorting.py::moe_sorting_native):
    #   max_padded = topk_ids.numel() + num_experts * block_size - topk
    # bs=32, k=8, ws=8, epr=32, tile_m=32: 256*8 + 256*32 - 8 = 10232.
    max_padded = max_recv * gemm2_topk + num_experts_global * tile_m - gemm2_topk
    max_blocks = (max_padded + tile_m - 1) // tile_m

    sorted_token_ids  = torch.empty((max_padded,), dtype=torch.int32, device=dev)
    sorted_weights    = torch.empty((max_padded,), dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    # num_valid_ids[2]: [0] = padded-total, [1] = num_input; GEMM2 reads [0].
    num_valid_ids     = torch.empty((2,), dtype=torch.int32, device=dev)

    # local_expert_mask: 1 on this rank's [rank*epr, (rank+1)*epr) slice, 0 elsewhere.
    expert_mask = torch.zeros(num_experts_global, dtype=torch.int32, device=dev)
    expert_mask[rank * epr:(rank + 1) * epr] = 1
    # moe_buf API placeholder (contents unused; shape-only allocation).
    moe_buf = torch.empty((max_recv, model_dim), dtype=torch.bfloat16, device=dev)

    # First sorting against real dispatch output to fill the sorted_* buffers.
    # This snapshot may be overwritten during cudagraph capture; the in-chain
    # _run_moe_sorting refills these buffers in-place every replay.
    _moe_sorting_fwd(  # type: ignore[misc]
        disp_op.shmem_disp_out_idx.view(max_recv, gemm2_topk),
        disp_op.shmem_disp_out_wts.view(max_recv, gemm2_topk),
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts_global,
        int(tile_m),
        expert_mask,
        disp_op.total_recv,
        0,
    )

    sorted_size = max_padded
    blocks      = max_blocks
    effective_valid = -1  # decided on-device by num_valid_ids[0].
    sort_state = dict(
        expert_mask=expert_mask,
        moe_buf=moe_buf,
        num_experts_global=num_experts_global,
        tile_m=int(tile_m),
        topk=int(gemm2_topk),
        mr=int(max_recv),
    )

    bias_dummy = torch.empty((0,), dtype=torch.float32, device=dev)

    # GEMM2 / combine-input buffers. `gemm2_out` is ALWAYS the token-level
    # [max_recv, model_dim] buffer that combine reads, regardless of GEMM2
    # epilogue mode:
    #   accumulate=True  -> GEMM2 atomic_fadd writes directly into gemm2_out
    #                       (same-t slots accumulate onto one row).
    #   accumulate=False -> GEMM2 plain-stores into `gemm2_intermediate`
    #                       [max_recv*topk, model_dim] (each (t, s) its own
    #                       row); a reduction kernel then sums over topk into
    #                       gemm2_out. This mirrors the production aiter path
    #                       (flydsl_moe_stage2 / compile_moe_gemm2_ex reduce
    #                       mode), where the topk fold-down is an internal
    #                       step of the GEMM2 op and combine never sees the
    #                       expanded layout.
    out_dtype = cfg.data_type if cfg.data_type in (torch.bfloat16, torch.float16) else torch.bfloat16
    _zc_out = (force_zero_copy_out if force_zero_copy_out is not None
               else (cfg.zero_copy and args.bench_op == "baseline"))
    if _zc_out:
        # zero-copy mode: the combine side skips Stage 1 and reads
        # shmem_comb_inp_tok directly. The caller must write the token-level
        # GEMM2/reduction result into the view returned by
        # op.get_registered_combine_input_buffer so the downstream combine
        # sees the right data. The fused path has its own P2P scatter (via
        # _fx_p2p_comb_inp) and bypasses this hook, so only baseline +
        # zero_copy needs to swap. The registered buffer is
        # [max_recv, model_dim]; in reduce mode the reduction kernel (not the
        # GEMM2 kernel) writes here, so the shapes always match.
        gemm2_out = disp_op.get_registered_combine_input_buffer(out_dtype)
        gemm2_out.zero_()
    else:
        gemm2_out = torch.zeros(max_recv, model_dim, dtype=out_dtype, device=dev)

    # reduce mode needs an extra [max_recv*topk, model_dim] scratch for the
    # non-atomic GEMM2 store; the reduction kernel folds it into gemm2_out.
    gemm2_intermediate = None
    if not args.gemm2_accumulate:
        if not HAS_MOE_REDUCTION:
            raise RuntimeError(
                "reduce mode (--no-gemm2-accumulate) needs the FlyDSL moe "
                f"reduction kernel, but its import failed: {_MOE_REDUCTION_ERR}"
            )
        gemm2_intermediate = torch.zeros(
            a2_rows, model_dim, dtype=out_dtype, device=dev
        )

    return dict(
        a2_storage=a2_storage,
        a2_scale_1d=a2_scale_1d,
        w2_storage=w2_storage,
        w2_scale_1d=w2_scale_1d,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        bias=bias_dummy,
        gemm2_out=gemm2_out,
        gemm2_intermediate=gemm2_intermediate,
        max_recv=max_recv,
        model_dim=model_dim,
        inter_dim=inter_dim,
        epr=epr,
        blocks=blocks,
        sorted_size=sorted_size,
        out_dtype=out_dtype,
        gemm2_topk=gemm2_topk,
        a2_rows=a2_rows,
        effective_valid=effective_valid,
        sort_state=sort_state,
    )


def _run_moe_sorting(gemm2_in, disp_op):
    """Call once after every dispatch in the chain: refills sorted_* /
    num_valid_ids in-place from the actual dispatch routing
    (shmem_disp_out_idx).

    Must be capture-friendly: invokes only the FlyDSL moe_sorting
    kernel; all buffers are pre-allocated by
    ``_build_gemm2_static_inputs`` during setup. Every replay reuses the
    same buffers and only refreshes their contents.
    """
    st = gemm2_in["sort_state"]
    _moe_sorting_fwd(  # type: ignore[misc]
        disp_op.shmem_disp_out_idx.view(st["mr"], st["topk"]),
        disp_op.shmem_disp_out_wts.view(st["mr"], st["topk"]),
        gemm2_in["sorted_token_ids"],
        gemm2_in["sorted_weights"],
        gemm2_in["sorted_expert_ids"],
        gemm2_in["num_valid_ids"],
        st["moe_buf"],
        st["num_experts_global"],
        st["tile_m"],
        st["expert_mask"],
        disp_op.total_recv,
        0,
    )


def _dump_gemm2_inputs(args, gemm2_in, out_dtype, *, prefix="[gemm2-dump]"):
    """Dump every compile-time / runtime GEMM2 argument for debugging and
    production-config diffing.

    Covers the launch signature:
        compiled(o, x, w, sx, sw, st, eids, sw_sorted,
                 num_valid_ids, bias,
                 tokens_in, n_in, k_in, size_expert_ids,
                 stream)
    """
    import torch as _torch

    def _tinfo(t):
        if not isinstance(t, _torch.Tensor):
            return f"(scalar) {t!r}"
        try:
            nbytes = t.numel() * t.element_size()
        except Exception:
            nbytes = -1
        return (f"shape={list(t.shape)} dtype={t.dtype} dev={t.device} "
                f"contig={t.is_contiguous()} bytes={nbytes}")

    out_s = "bf16" if out_dtype == _torch.bfloat16 else (
        "f16" if out_dtype == _torch.float16 else "f32"
    )

    print(f"\n{prefix} === COMPILE-TIME (compile_mixed_moe_gemm2 kwargs) ===")
    print(f"{prefix}   model_dim       = {args.hidden_dim}")
    print(f"{prefix}   inter_dim       = {args.inter_dim}")
    print(f"{prefix}   experts         = {args.num_experts_per_rank}")
    print(f"{prefix}   topk            = {gemm2_in['gemm2_topk']}  "
          f"(A2 row addr: row = t * topk + s)")
    print(f"{prefix}   tile_m / n / k  = {args.tile_m2} / {args.tile_n2} / {args.tile_k2}")
    print(f"{prefix}   a_dtype         = {args.gemm2_a_dtype}")
    print(f"{prefix}   b_dtype         = {args.gemm2_b_dtype}")
    print(f"{prefix}   out_dtype       = {out_s}")
    print(f"{prefix}   accumulate      = {args.gemm2_accumulate}  "
          f"({'atomic_fadd accumulates s onto output row t' if args.gemm2_accumulate else 'plain store, (t,s) owns its row'})")
    print(f"{prefix}   persist_m       = {args.persist_m}")
    print(f"{prefix}   xcd_swizzle     = {args.xcd_swizzle}")
    print(f"{prefix}   doweight_stage2 = True")

    print(f"\n{prefix} === RUNTIME TENSORS (launch args; dtype/shape/dev) ===")
    print(f"{prefix}  [pos  0] o   = arg_out (gemm2_out)         "
          f"{_tinfo(gemm2_in['gemm2_out'])}")
    print(f"{prefix}  [pos  1] x   = arg_x (A2 storage 1D)        "
          f"{_tinfo(gemm2_in['a2_storage'])}")
    print(f"{prefix}  [pos  2] w   = arg_w (W2 storage 1D)        "
          f"{_tinfo(gemm2_in['w2_storage'])}")
    print(f"{prefix}  [pos  3] sx  = arg_scale_x (A2 e8m0 scale)  "
          f"{_tinfo(gemm2_in['a2_scale_1d'])}")
    print(f"{prefix}  [pos  4] sw  = arg_scale_w (W2 e8m0 scale)  "
          f"{_tinfo(gemm2_in['w2_scale_1d'])}")
    print(f"{prefix}  [pos  5] st  = sorted_token_ids (fused i32) "
          f"{_tinfo(gemm2_in['sorted_token_ids'])}")
    print(f"{prefix}  [pos  6] eids= sorted_expert_ids            "
          f"{_tinfo(gemm2_in['sorted_expert_ids'])}")
    print(f"{prefix}  [pos  7] sws = sorted_weights               "
          f"{_tinfo(gemm2_in['sorted_weights'])}")
    print(f"{prefix}  [pos  8] num_valid_ids (i32 device scalar)  "
          f"{_tinfo(gemm2_in['num_valid_ids'])}")
    print(f"{prefix}  [pos  9] bias (empty placeholder)           "
          f"{_tinfo(gemm2_in['bias'])}")

    print(f"\n{prefix} === RUNTIME SCALARS (4 trailing i32) ===")
    print(f"{prefix}  [pos 10] tokens_in       = {gemm2_in['max_recv']}  "
          f"(= world_size * bs; A2 rows = tokens_in * topk = {gemm2_in['a2_rows']})")
    print(f"{prefix}  [pos 11] n_in (=model_dim) = {args.hidden_dim}")
    print(f"{prefix}  [pos 12] k_in (=inter_dim) = {args.inter_dim}")
    print(f"{prefix}  [pos 13] size_expert_ids = {gemm2_in['blocks']}  "
          f"(= epr * ceil(per_e / tile_m), # expert-grouped m_blocks)")
    print(f"{prefix}  [pos 14] stream          = torch.cuda.current_stream()")

    print(f"\n{prefix} === DERIVED / SEMANTIC ===")
    print(f"{prefix}  num_valid_ids value     = {int(gemm2_in['num_valid_ids'][0].item())}  "
          f"(kernel iterates row in [0, this); row-level t/s sentinels DCE internally)")
    print(f"{prefix}  effective_valid (real)  = {gemm2_in['effective_valid']}  "
          f"(-1 = real-routing sorting, decided on-device by num_valid_ids[0])")
    print(f"{prefix}  sorted_size             = {gemm2_in['sorted_size']}")

    # Decode sorted_token_ids and report atomic contention.
    try:
        all_raw = gemm2_in["sorted_token_ids"].tolist()
        max_recv_sentinel = int(gemm2_in["max_recv"])
        real = [(v & 0x00FFFFFF, (v >> 24) & 0xFF) for v in all_raw
                if (v & 0x00FFFFFF) < max_recv_sentinel]

        from collections import Counter
        t_counter: Counter = Counter(t for (t, _s) in real)
        s_counter: Counter = Counter(s for (_t, s) in real)
        cont_hist: Counter = Counter(t_counter.values())  # how many t have d s entries

        # Show the first 8 entries of experts 0 / 1 / 7 to demonstrate that
        # the same t lands on multiple experts.
        tile_m = int(args.tile_m2)
        st_t = gemm2_in["sorted_token_ids"]
        for e in (0, 1, 7):
            block_start = e * tile_m
            head_e = st_t[block_start:block_start + 8].tolist()
            decoded_e = [(v & 0x00FFFFFF, (v >> 24) & 0xFF) for v in head_e]
            print(f"{prefix}  expert={e:2d} block[:8] decoded (t, s) = {decoded_e}")

        print(f"{prefix}  sorted_expert_ids[:16] = "
              f"{gemm2_in['sorted_expert_ids'][:16].tolist()}")
        print(f"{prefix}  real (t, s) pair count          = {len(real)}")
        print(f"{prefix}  unique t count                  = {len(t_counter)}  "
              f"(each unique t -> one output row, the atomic_fadd target)")
        print(f"{prefix}  s value distribution            = "
              f"{dict(sorted(s_counter.items()))}")
        print(f"{prefix}  atomic contention histogram (#s-per-t -> #t) = "
              f"{dict(sorted(cont_hist.items()))}")
        print(f"{prefix}    -> read as: how often the same output row is hit by atomic_fadd")
    except Exception as exc:
        print(f"{prefix}  (decode failed: {exc!r})")

    print()


def _build_reduction_callable(gemm2_in, out_dtype):
    """Compile + warm up the topk reduction used by GEMM2 reduce mode.

    Folds the expanded plain-store scratch X[max_recv, topk, model_dim] into
    the token-level combine input Y[max_recv, model_dim] via Y[t] = sum_s X[t,s].
    The scratch is zeroed before every GEMM2 launch (see launch_gemm2) so the
    unmasked sum is exact: invalid (t, s) slots stay 0 and contribute nothing,
    matching the atomic path where only routed slots atomic_fadd onto row t.

    Returns a no-arg launcher (capturable after the warm-up JIT compile).
    """
    mr = int(gemm2_in["max_recv"])
    topk = int(gemm2_in["gemm2_topk"])
    md = int(gemm2_in["model_dim"])
    reduce_dtype = (
        "bf16" if out_dtype == torch.bfloat16
        else ("f16" if out_dtype == torch.float16 else "f32")
    )
    reduce_exe = compile_moe_reduction(
        topk=topk, model_dim=md, dtype_str=reduce_dtype, use_mask=False,
    )
    _X = gemm2_in["gemm2_intermediate"].view(mr, topk, md)
    _Y = gemm2_in["gemm2_out"].view(mr, md)
    _empty_mask = torch.empty(
        (0, topk), dtype=torch.uint8, device=gemm2_in["gemm2_out"].device,
    )

    def launch_reduction():
        reduce_exe(_X, _Y, _empty_mask, mr, torch.cuda.current_stream())

    # Warm up (triggers JIT compile) outside any cudagraph capture.
    launch_reduction()
    return launch_reduction


def _build_gemm2_callable(args, gemm2_in, out_dtype):
    """Compile mixed_moe_gemm2 and return a no-arg launcher.

    atomic mode (accumulate=True): the GEMM2 kernel atomic_fadds directly into
        gemm2_out [max_recv, model_dim].
    reduce mode (accumulate=False): the GEMM2 kernel plain-stores into
        gemm2_intermediate [max_recv*topk, model_dim], then a reduction kernel
        sums over topk into gemm2_out [max_recv, model_dim]. Either way combine
        reads the same token-level gemm2_out, so the GEMM2 epilogue mode stays
        an internal detail (mirrors the production aiter flydsl_moe_stage2 path).
    """
    out_s = "bf16" if out_dtype == torch.bfloat16 else (
        "f16" if out_dtype == torch.float16 else "f32"
    )

    # accumulate=True -> atomic path; accumulate=False -> reduce path.
    # Note: historically fp4 b_dtype was force-set to accumulate=False
    # (suspected silent-drop of atomic_fadd under mixed fp4 lowering);
    # we no longer force it so the kernel can be validated empirically.
    # If gemm2_out ends up all-zero, file a follow-up kernel fix PR.
    accumulate = args.gemm2_accumulate

    # In reduce mode the GEMM2 kernel writes the expanded per-(t, s) scratch;
    # the reduction kernel produces the token-level combine input. In atomic
    # mode the GEMM2 kernel writes gemm2_out directly.
    _kernel_out = (gemm2_in["gemm2_out"] if accumulate
                   else gemm2_in["gemm2_intermediate"])

    # GEMM2 topk matches the dispatch routing topk (args.k): each recv
    # token can be hit by at most topk local experts; the A2 buffer is
    # addressed as [tokens_in * topk, inter_dim].
    exe = compile_mixed_moe_gemm2(
        model_dim=args.hidden_dim,
        inter_dim=args.inter_dim,
        experts=args.num_experts_per_rank,
        xcd_swizzle=args.xcd_swizzle,
        topk=gemm2_in["gemm2_topk"],
        tile_m=args.tile_m2,
        tile_n=args.tile_n2,
        tile_k=args.tile_k2,
        doweight_stage2=True,
        a_dtype=args.gemm2_a_dtype,
        b_dtype=args.gemm2_b_dtype,
        out_dtype=out_s,
        accumulate=accumulate,
        persist_m=args.persist_m,
        sort_block_m=args.sort_block_m,
        b_nt=args.b_nt,
    )

    def _args(o, x, w, sx, sw, st, eids, sw_sorted):
        # tokens=max_recv (cudagraph upper bound), size_expert_ids=blocks.
        return (
            o,
            x,
            w,
            sx,
            sw,
            st,
            eids,
            sw_sorted,
            gemm2_in["num_valid_ids"],
            gemm2_in["bias"],
            gemm2_in["max_recv"],
            args.hidden_dim,
            args.inter_dim,
            int(gemm2_in["blocks"]),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(
        exe,
        *_args(
            _kernel_out,
            gemm2_in["a2_storage"],
            gemm2_in["w2_storage"],
            gemm2_in["a2_scale_1d"],
            gemm2_in["w2_scale_1d"],
            gemm2_in["sorted_token_ids"],
            gemm2_in["sorted_expert_ids"],
            gemm2_in["sorted_weights"],
        ),
    )

    # reduce mode: compile + warm up the topk reduction that folds the
    # expanded scratch back into the token-level gemm2_out.
    launch_reduction = None
    if not accumulate:
        launch_reduction = _build_reduction_callable(gemm2_in, out_dtype)

    def launch_gemm2():
        if not accumulate:
            # The plain-store scratch must start zeroed each launch so invalid
            # (t, s) slots contribute 0 to the unmasked topk sum (the GEMM2
            # kernel only writes routed slots).
            gemm2_in["gemm2_intermediate"].zero_()
        compiled(
            *_args(
                _kernel_out,
                gemm2_in["a2_storage"],
                gemm2_in["w2_storage"],
                gemm2_in["a2_scale_1d"],
                gemm2_in["w2_scale_1d"],
                gemm2_in["sorted_token_ids"],
                gemm2_in["sorted_expert_ids"],
                gemm2_in["sorted_weights"],
            )
        )
        if launch_reduction is not None:
            launch_reduction()

    return launch_gemm2


# --- Chain abstractions: the baseline and fused end-to-end pipelines ---
# NOTE: callers decide whether dispatch is in the chain (dispatch_inputs=None
# skips it). Including dispatch is the mori best practice (validated in
# test_profiler_dispatch_combine.py): it works around the combine kernel
# zeroing total_recv at the end of Stage 2, which otherwise causes every
# subsequent cudagraph replay to run combine as an empty "for tok_i in
# range(0, 0)" kernel. Without dispatch, measured combine GPU time is just
# prologue + barrier + launch overhead (~17us at bs=256) and does not
# reflect real Stage 1 P2P + Stage 3 work.
_MORI_SUPPORTED_DTYPES = (
    torch.bfloat16, torch.float16,
    torch.float8_e4m3fn, torch.float8_e4m3fnuz,
)


def _build_mori_op(rank, world_size, cfg, block_num=None, warp_per_block=None):
    """Build a mori EpDispatchCombineOp (mirrors
    test_profiler_dispatch_combine.build_mori_ref).

    Reuses this test's FlyDSLDispatchCombineConfig ``cfg``, mapping its shape /
    dtype / launch-geometry fields onto mori's EpDispatchCombineConfig.
    """
    if cfg.data_type not in _MORI_SUPPORTED_DTYPES:
        raise RuntimeError(
            f"mori baseline needs dispatch dtype in {_MORI_SUPPORTED_DTYPES}, "
            f"got cfg.data_type={cfg.data_type}; rerun with --dtype bf16/fp16/fp8_*."
        )
    from mori.ops.dispatch_combine import EpDispatchCombineConfig, EpDispatchCombineOp

    elem = torch.tensor([], dtype=cfg.data_type).element_size()
    mcfg = EpDispatchCombineConfig(
        data_type=cfg.data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=cfg.hidden_dim,
        scale_dim=cfg.num_experts_per_token,
        scale_type_size=4,
        max_token_type_size=(
            cfg.max_token_type_size if cfg.max_token_type_size > 0 else elem
        ),
        max_num_inp_token_per_rank=cfg.max_num_inp_token_per_rank,
        num_experts_per_rank=cfg.num_experts_per_rank,
        num_experts_per_token=cfg.num_experts_per_token,
        warp_num_per_block=(
            warp_per_block if warp_per_block is not None
            else cfg.dispatch_warp_num_per_block_eff
        ),
        block_num=(
            block_num if block_num is not None
            else cfg.dispatch_block_num_eff
        ),
        gpu_per_node=world_size,
        use_external_inp_buf=not cfg.zero_copy,
        quant_type=cfg.quant_type,
        # Only forward the cap when unset (0) or equal to the worst-case ws*M;
        # otherwise mori treats it as a hard contract and a routing overflow
        # device-asserts, permanently poisoning the HIP context.
        max_total_recv_tokens=(
            cfg.max_total_recv_tokens
            if cfg.max_total_recv_tokens == 0
            or cfg.max_total_recv_tokens
            == cfg.world_size * cfg.max_num_inp_token_per_rank
            else 0
        ),
    )
    return EpDispatchCombineOp(mcfg)


class _MoriDispOpAdapter:
    """Adapt a mori EpDispatchCombineOp to the minimal FlyDSL disp_op surface
    this test uses: dispatch / combine / get_registered_combine_input_buffer,
    reset() / barrier() (for _capture_chain), shmem_disp_out_idx /
    shmem_disp_out_wts / total_recv (for _run_moe_sorting, filled in-place by
    dispatch return values), and shmem_comb_inp_tok (setup zero-copy ptr check).
    """

    def __init__(self, mori_op, out_dtype):
        self._op = mori_op
        self._out_dtype = out_dtype
        self.shmem_disp_out_idx = None
        self.shmem_disp_out_wts = None
        self.total_recv = None
        # mori combine Stage1 reads this registered buffer directly under
        # zero-copy; gemm2_out must land here.
        self.shmem_comb_inp_tok = mori_op.get_registered_combine_input_buffer(
            out_dtype
        )

    def dispatch(self, inp, wts, scales, idx):
        ret = self._op.dispatch(inp, wts, scales, idx)
        _out, out_weights, _out_scales, out_indices, total_recv = ret
        self.shmem_disp_out_idx = out_indices
        self.shmem_disp_out_wts = out_weights
        self.total_recv = total_recv
        return ret

    def combine(self, inp_for_kernel, weights, combine_idx, **kwargs):
        return self._op.combine(inp_for_kernel, weights, combine_idx, **kwargs)

    def get_registered_combine_input_buffer(self, dtype):
        return self._op.get_registered_combine_input_buffer(dtype)

    def reset(self):
        r = getattr(self._op, "reset", None)
        if callable(r):
            r()
        ms.shmem_barrier_all()

    def barrier(self):
        ms.shmem_barrier_all()


def _baseline_chain(disp_op, gemm2_launch, gemm2_in, combine_idx,
                    dispatch_inputs=None):
    """baseline: [dispatch ->] [moe_sorting ->] moe_gemm2 -> combine.

    dispatch_inputs=None              : legacy path; launches only GEMM2 +
                                        combine (combine runs empty).
    dispatch_inputs=(inp, wts, scales, idx)
                                      : run dispatch every chain so the
                                        combine-side total_recv / routing
                                        tables stay fresh.

    moe_sorting is inserted after dispatch: every replay must rerun
    sorting because dispatch's dest_tok_all allocation (atomic_add order)
    is non-deterministic. Each replay yields different shmem_disp_out_idx
    row contents, and the t indices in `sorted` must match the
    dispatch-side addr_tis of the current replay.
    """
    if dispatch_inputs is not None:
        inp, wts, scales, idx = dispatch_inputs
        # Re-dispatch every chain so routing / total_recv stay fresh, then
        # combine the routing index produced by THIS dispatch (mirrors
        # test_profiler_dispatch_combine._run_combine, which combines ret[3]).
        # flydsl returns a stable shmem-view handle; mori returns its internal
        # index buffer -- either way combine matches the current routing.
        _disp_ret = disp_op.dispatch(inp, wts, scales, idx)
        combine_idx = _disp_ret[3]
        # capture moe_sorting to refresh sorted_* tables in-place
        _run_moe_sorting(gemm2_in, disp_op)
    if os.environ.get("FLYDSL_DEBUG_CHAIN_DISPATCH_ONLY", "0") == "1":
        return None
    if os.environ.get("FLYDSL_DEBUG_CHAIN_SKIP_GEMM2", "0") != "1":
        gemm2_launch()
    if os.environ.get("FLYDSL_DEBUG_CHAIN_NO_COMBINE", "0") == "1":
        return None
    out = disp_op.combine(gemm2_in["gemm2_out"], None, combine_idx)
    return out


def _fused_chain(disp_op, fused_op, gemm2_in, combine_idx, dispatch_total_recv,
                 dispatch_inputs=None):
    """fused: [dispatch ->] [moe_sorting ->] fused_gemm2_combine [-> combine_no_stage1].

    See _baseline_chain for why moe_sorting must be captured into the chain.

    dispatch_inputs: see _baseline_chain. Without dispatch, the fused path's
    combine_no_stage1 (Stage 1 weight P2P + Stage 3 read+accum) also runs
    empty and the fused gemm2_combine timing is grossly underestimated.
    """
    if fused_op is None:
        raise RuntimeError(
            "fused op not available; build kernels/mixed_moe_gemm2_combine_fused_op.py first"
        )
    if dispatch_inputs is not None:
        inp, wts, scales, idx = dispatch_inputs
        disp_op.dispatch(inp, wts, scales, idx)
        _run_moe_sorting(gemm2_in, disp_op)
    out = fused_op.run(
        a2=gemm2_in["a2_storage"],
        w2=gemm2_in["w2_storage"],
        a2_scale=gemm2_in["a2_scale_1d"],
        w2_scale=gemm2_in["w2_scale_1d"],
        sorted_token_ids=gemm2_in["sorted_token_ids"],
        sorted_expert_ids=gemm2_in["sorted_expert_ids"],
        sorted_weights=gemm2_in["sorted_weights"],
        num_valid_ids=gemm2_in["num_valid_ids"],
    )
    return out


# --- Profiler helpers ---------------------------------------------------
def _make_profiler(active_iters: int = None, prof_warmup: int = 5):
    kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )
    if active_iters is not None and active_iters > 0:
        kwargs["schedule"] = torch.profiler.schedule(
            wait=1, warmup=prof_warmup, active=active_iters, repeat=1,
        )
    return profile(**kwargs)


def _save_profile_json(prof, out_path: str, rank: int, op_tag: str, meta: dict):
    rows = []
    for evt in prof.key_averages():
        rows.append({
            "name":             evt.key,
            "calls":            evt.count,
            "cuda_time_avg_us": round(evt.device_time, 2),
            "cuda_time_total_us": round(evt.device_time * evt.count, 2),
            "cpu_time_avg_us":  round(evt.cpu_time, 2),
            "cpu_time_total_us": round(evt.cpu_time * evt.count, 2),
        })
    rows.sort(key=lambda r: r["cuda_time_total_us"], reverse=True)
    payload = {
        "meta":         {**meta, "op": op_tag, "rank": rank},
        "kernel_stats": rows,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    trace_path = out_path.replace(".json", "_trace.json")
    prof.export_chrome_trace(trace_path)
    return trace_path


# Kernel-name matching rules:
#   - FlyDSL combine defaults to ep_combine_intranode_0 after flyc.compile.
#   - moe_gemm2 compiles to names of the form moe_gemm2*.
#   - fused goes through fused_gemm2_combine.
# Substring matching below avoids per-build suffix differences.
_KERNEL_PATTERNS = {
    "gemm2":          ["moe_gemm2"],
    "combine":        ["ep_combine_intranode"],
    "fused":          ["fused_gemm2_combine", "mixed_moe_gemm2_combine"],
}


def _stats_from_trace(trace_path: str, op_tag: str,
                      rank: int, world_size: int, dev: torch.device,
                      active_iters: int, skip_first: int):
    """Extract gemm2 / combine / fused / replay times from the chrome trace
    and reduce them across ranks (avg/min/max).

    Returns dict[metric_name -> {avg/min/max}] for:
      - gemm2_gpu, combine_gpu, fused_gpu
      - replay_cuda_e2e, replay_cpu_e2e
    """
    with open(trace_path) as f:
        tr = json.load(f)

    cg_label = f"{op_tag}::cudagraph_replay"
    kernel_events = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]

    def _last_active(events_filter, n_take, n_skip):
        sel = sorted(
            [e for e in kernel_events if events_filter(e.get("name", ""))],
            key=lambda e: e["ts"],
        )
        active = [e["dur"] for e in sel[-n_take:]]
        valid = active[n_skip:]
        return valid

    g_valid = _last_active(
        lambda n: any(p in n for p in _KERNEL_PATTERNS["gemm2"]),
        active_iters, skip_first,
    )
    c_valid = _last_active(
        lambda n: any(p in n for p in _KERNEL_PATTERNS["combine"]),
        active_iters, skip_first,
    )
    f_valid = _last_active(
        lambda n: any(p in n for p in _KERNEL_PATTERNS["fused"]),
        active_iters, skip_first,
    )

    cg_all = sorted(
        [e for e in tr["traceEvents"]
         if e.get("cat") == "gpu_user_annotation" and cg_label in e.get("name", "")],
        key=lambda e: e["ts"],
    )
    cg_active = [e["dur"] for e in cg_all[-active_iters:]]
    cg_valid = cg_active[skip_first:]

    cg_cpu_all = sorted(
        [e for e in tr["traceEvents"]
         if e.get("cat") == "user_annotation" and cg_label in e.get("name", "")],
        key=lambda e: e["ts"],
    )
    cg_cpu_active = [e["dur"] for e in cg_cpu_all[-active_iters:]]
    cg_cpu_valid = cg_cpu_active[skip_first:]

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    if rank == 0:
        print(f"[trace-stats] {op_tag}: gemm2={len(g_valid)} "
              f"combine={len(c_valid)} fused={len(f_valid)} replay={len(cg_valid)} "
              f"(active={active_iters}, skip={skip_first})")

    local = torch.tensor([
        _avg(g_valid), _avg(c_valid), _avg(f_valid),
        _avg(cg_valid), _avg(cg_cpu_valid),
    ], dtype=torch.float64, device=dev)

    s  = local.clone(); dist.all_reduce(s,  op=dist.ReduceOp.SUM)
    mx = local.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = ["gemm2_gpu", "combine_gpu", "fused_gpu",
            "replay_cuda_e2e", "replay_cpu_e2e"]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()}
            for i, k in enumerate(keys)}


def _print_aggregated(stats: dict, op_tag: str, world_size: int, meta: dict,
                      active_iters: int):
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {op_tag.upper()} [CUDAGraph+Profiler]  "
          f"EP={world_size}  bs={meta['max_tokens']}  "
          f"h={meta['hidden_dim']}  inter={meta['inter_dim']}  k={meta['k']}  "
          f"({active_iters} valid iters)")
    print(f"  avg / min / max across all {world_size} ranks (us/call)")
    print(sep)
    hdr = f"  {'metric':<38}  {'avg':>8}  {'min':>8}  {'max':>8}"
    print(hdr)
    print(f"  {'-'*64}")

    rows = [
        ("[Device] moe_gemm2 kernel GPU time",        "gemm2_gpu"),
        ("[Device] combine kernel GPU time",          "combine_gpu"),
        ("[Device] fused_gemm2_combine GPU time",     "fused_gpu"),
        ("[E2E]    replay CUDA time (incl. sync)",    "replay_cuda_e2e"),
        ("[Host]   replay CPU  time",                 "replay_cpu_e2e"),
    ]
    for label, key in rows:
        v = stats[key]
        print(f"  {label:<38}  {v['avg']:>8.1f}  {v['min']:>8.1f}  {v['max']:>8.1f}")
    print()


# --- profile + cudagraph driver -----------------------------------------
def _capture_chain(chain_fn, capture_stream, eager_warmup=1, disp_op=None,
                   mori_capture=False):
    """eager warmup (triggers JIT compile) -> CUDAGraph capture.

    If the chain contains dispatch (option A), the sequence must mirror
    test_profiler_dispatch_combine.py:_cudagraph_capture_flydsl exactly:
        op.reset()                -> ms.shmem_barrier_all()
        chain_fn()                -> 1 eager warmup, triggers jit compile
        op.barrier()              -> ms.shmem_barrier_all()
        graph.capture(chain_fn)   -> 1 capture
    Any deviation (extra sync, extra warmups, mid-sequence reset, ...)
    drifts mori shmem's internal cross-device counter and the very first
    chain_fn() during capture trips hipErrorIllegalAddress.

    mori_capture=True (the --baseline-comm mori baseline chain): mirror the
    single-op _cudagraph_capture_mori -- reset() but NO eager warmup, then
    capture directly. mori dispatch/combine are precompiled hsaco (no JIT
    warmup); an extra eager dispatch would advance mori's cross-device
    monotonic flag, so the first captured launch would hit an inconsistent
    counter (symmetric_memory illegal access).
    """
    if mori_capture:
        # Mirror test_profiler_dispatch_combine._cudagraph_capture_mori: NO
        # reset and NO eager warmup -- just a global barrier, then capture
        # dispatch+combine directly. mori dispatch/combine flag usage is
        # self-consistent across replays; an extra reset() shifts the flag
        # baseline so the recorded dispatch first-replay mismatches and trips a
        # symmetric_memory illegal access.
        ms.shmem_barrier_all()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            chain_fn()
        return g

    if disp_op is not None:
        disp_op.reset()    # ms.shmem_barrier_all
    for _ in range(eager_warmup):
        chain_fn()
    if disp_op is not None:
        disp_op.barrier()  # ms.shmem_barrier_all
    else:
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        chain_fn()
    return g


def profile_cudagraph_chain(chain_fn, op_tag: str,
                            rank: int, world_size: int, dev: torch.device,
                            iters: int, out_dir: str, meta: dict,
                            disp_op=None, mori_capture=False):
    """Use torch.profiler to capture per-kernel timings across CUDAGraph replays.

    When the chain includes dispatch (option A), disp_op must be passed so
    _capture_chain can run reset + cross-device barrier between the eager
    warmups.
    """
    ms.shmem_barrier_all()

    capture_stream = torch.cuda.Stream()
    g = _capture_chain(chain_fn, capture_stream, eager_warmup=1, disp_op=disp_op,
                       mori_capture=mori_capture)
    if rank == 0:
        print(f"\n[profile+cudagraph] {op_tag} capture done")

    replay_warmup = 10
    for _ in range(replay_warmup):
        g.replay()
    torch.cuda.synchronize()

    prof_warmup = 5
    skip_first = 5
    valid_iters = max(iters - skip_first, 1)
    total_steps = 1 + prof_warmup + iters
    if rank == 0:
        print(f"[profile+cudagraph] {op_tag} scheduled profiler: "
              f"warmup={prof_warmup}, active={iters}, "
              f"skip first {skip_first}, effective {valid_iters} iters (no-reset)...")

    with _make_profiler(active_iters=iters, prof_warmup=prof_warmup) as prof:
        for _ in range(total_steps):
            with record_function(f"{op_tag}::cudagraph_replay"):
                g.replay()
            prof.step()

    out_path = os.path.join(out_dir, f"{op_tag}_cudagraph_rank{rank}.json")
    trace_path = _save_profile_json(prof, out_path, rank, op_tag, meta)
    if rank == 0:
        print(f"[profile+cudagraph] {op_tag} trace → {trace_path}")

    agg = _stats_from_trace(
        trace_path, op_tag, rank, world_size, dev,
        active_iters=iters, skip_first=skip_first,
    )
    if rank == 0:
        _print_aggregated(agg, op_tag, world_size, meta, active_iters=valid_iters)
    return agg


def _print_speedup(baseline_stats: dict, fused_stats: dict, world_size: int):
    """Print baseline vs fused GPU-kernel speedup."""
    bk = (baseline_stats["gemm2_gpu"]["avg"]
          + baseline_stats["combine_gpu"]["avg"])
    fk = fused_stats["fused_gpu"]["avg"]
    if fk <= 0:
        return
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  Speedup [baseline vs fused]   GPU kernel-only sum")
    print(sep)
    print(f"  baseline (gemm2 + combine):         {bk:>8.1f} μs")
    print(f"  fused    (gemm2_combine fused):     {fk:>8.1f} μs")
    print(f"  speedup:                            {bk / fk:>8.3f}x")
    print()


# --- verify mode: baseline vs fused numerical consistency -------------------
def _reset_combine_shmem(disp_op):
    """Clear combine-path shmem buffers so baseline / fused do not pollute each other.

    Clears:
      - shmem_comb_inp_{tok,wts}: buffers written by stage1 P2P scatter. Both
        paths overwrite fully; zeroing here is defensive (avoids leftover
        bytes from the previous run leaking into bytes not covered this round).
      - shmem_comb_out_{tok,wts}: final outputs written by stage3. Zeroing
        makes diff-checking easy: any byte missed by fused shows up directly.
    Do *not* clear ``addr_xdb_flag`` / ``shmem_xdev_bar_mem`` -- combine
    increments/writes them monotonically and they must stay consistent
    across calls.
    """
    disp_op.shmem_comb_inp_tok.zero_()
    disp_op.shmem_comb_inp_wts.zero_()
    disp_op.shmem_comb_out_tok.zero_()
    disp_op.shmem_comb_out_wts.zero_()


def _snapshot_combine_out(disp_op):
    """Clone current combine output (to avoid being overwritten next round).

    Returns ``(out_tok_clone, out_wts_clone)`` with the same dtype as the
    op-exposed views (bf16/fp16 token + f32 weights).
    """
    cfg = disp_op.cfg
    mt  = cfg.max_num_inp_token_per_rank
    k   = cfg.num_experts_per_token
    # Combine output is always the external dtype (fp8_direct_cast uses fp8
    # only on the wire; Stage 3 casts back to bf16 inline).
    out_tok = (
        disp_op.shmem_comb_out_tok.view(torch.int8)[:mt * cfg.token_bytes]
        .view(cfg.data_type).view(mt, cfg.token_view_dim)
    )
    out_wts = disp_op.shmem_comb_out_wts.view(mt, k)
    return out_tok.detach().clone(), out_wts.detach().clone()


def _compare_two(name_a, ta, name_b, tb, *, rank, atol_abs=2e-2, atol_rel=5e-2,
                 atol_ulp_k=16, max_print=8):
    """Per-token compare of two tensors; prints stats + a few element diffs. Returns pass:bool."""
    assert ta.shape == tb.shape, (
        f"shape mismatch: {name_a}={tuple(ta.shape)} vs {name_b}={tuple(tb.shape)}"
    )
    a_f = ta.detach().float()
    b_f = tb.detach().float()
    nan_a  = torch.isnan(a_f) | torch.isinf(a_f)
    nan_b  = torch.isnan(b_f) | torch.isinf(b_f)
    n_nan_a  = int(nan_a.sum().item())
    n_nan_b  = int(nan_b.sum().item())
    nan_match = bool(torch.equal(nan_a, nan_b))

    # Mask out NaN/Inf positions on both sides before computing abs/rel stats:
    # under fp8_direct_cast, random fp4 data can produce fp8 NaN encodings via
    # cvt_pk_fp8_f32 (baseline and fused share the same input, so NaNs land
    # in identical positions and do not break numerical equivalence).
    # FAIL only when: (1) NaN positions mismatch between a and b; or
    #                 (2) abs/rel over threshold on the finite subset.
    finite_mask = ~(nan_a | nan_b)
    a_finite = torch.where(finite_mask, a_f, torch.zeros_like(a_f))
    b_finite = torch.where(finite_mask, b_f, torch.zeros_like(b_f))
    diff   = (a_finite - b_finite).abs()
    # Use max(|a|,|b|) as denominator so rel does not blow up when a=0
    # (baseline can output 0 at dup-sentinel + zero-fragment boundaries while
    # fused produces ~1 ULP non-zero).
    a_norm = torch.maximum(a_finite.abs(), b_finite.abs()).clamp_min(1e-6)
    rel    = diff / a_norm

    abs_max  = diff.max().item() if diff.numel() > 0 else 0.0
    abs_mean = diff.mean().item() if diff.numel() > 0 else 0.0
    rel_max  = rel.max().item() if rel.numel() > 0 else 0.0
    rel_mean = rel.mean().item() if rel.numel() > 0 else 0.0

    # Element-wise pass: any one of the four thresholds qualifies:
    #   (1) |diff| <= atol_abs                              -- absolute (small-value noise)
    #   (2) |diff| / max(|a|,|b|) <= atol_rel               -- relative (large-value round)
    #   (3) |diff| <= atol_ulp_k * max(|a|,|b|) * 2^-mantissa -- dtype-aware
    #       k-ULP accumulation budget (bf16: 2^-7, atol_ulp_k=8 -> ~6.25% round budget)
    #   (4) max(|a|,|b|) <= near_zero_floor -- near-zero noise zone; sign-flip here
    #       is just the real non-associativity between atomic_fadd ordering and
    #       fp32 single-accum ordering (unavoidable under atomic8_1pe / atomic-k stress)
    if ta.dtype == torch.bfloat16:
        _mantissa_bits = 7
    elif ta.dtype == torch.float16:
        _mantissa_bits = 10
    elif ta.dtype == torch.float32:
        _mantissa_bits = 23
    else:
        _mantissa_bits = 7
    _ulp_thr = (a_norm * (atol_ulp_k * (2.0 ** -_mantissa_bits)))
    # near-zero floor: (k * 2^-mantissa) times the tensor abs max; for bf16,
    # k=8 ~= 6.25% of output magnitude -- any sign-flip below this is
    # accumulation noise.
    _out_max = a_finite.abs().max().clamp_min(1e-6)
    _near_zero_floor = _out_max * (atol_ulp_k * (2.0 ** -_mantissa_bits))
    _is_near_zero = (a_finite.abs() < _near_zero_floor) & (b_finite.abs() < _near_zero_floor)
    fail_mask = ((diff > atol_abs) & (rel > atol_rel) & (diff > _ulp_thr)
                 & (~_is_near_zero))
    n_diff_per_tok = fail_mask.reshape(fail_mask.shape[0], -1).any(dim=-1).sum().item()
    n_fail = int(fail_mask.sum().item())
    pass_ok  = (n_fail == 0) and nan_match

    # Every rank prints (hang debugging / cross-device locating); rank 0 used
    # to be the dominant printer, other ranks only emit a brief report on FAIL.
    if rank == 0 or not pass_ok:
        status = "PASS" if pass_ok else "FAIL"
        print(f"  [rank {rank}] [{status}] {name_a} vs {name_b}: "
              f"shape={tuple(ta.shape)} dtype={ta.dtype}/{tb.dtype}")
        print(f"  [rank {rank}]         abs: max={abs_max:.4e} mean={abs_mean:.4e}  "
              f"(thr={atol_abs:.2e})")
        print(f"  [rank {rank}]         rel: max={rel_max:.4e} mean={rel_mean:.4e}  "
              f"(thr={atol_rel:.2e})")
        print(f"  [rank {rank}]         nan/inf: a={n_nan_a} b={n_nan_b} "
              f"match={'yes' if nan_match else 'NO'}  "
              f"fail_elems={n_fail} tokens_with_fail={n_diff_per_tok}")
        if not pass_ok and ta.numel() > 0:
            flat = diff.reshape(-1)
            top  = torch.topk(flat, k=min(max_print, flat.numel()))
            for off, v in zip(top.indices.tolist(), top.values.tolist()):
                idx = off
                rows = ta.shape[1] if ta.ndim >= 2 else 1
                tok_id = idx // rows
                col_id = idx % rows
                print(f"  [rank {rank}]         diff[{tok_id},{col_id}]: "
                      f"{name_a}={a_f.reshape(-1)[idx].item():.6e} "
                      f"{name_b}={b_f.reshape(-1)[idx].item():.6e} "
                      f"|delta|={v:.6e}")
    return pass_ok


def _run_verify(disp_op, fused_op,
                gemm2_in, gemm2_launch, combine_idx, dispatch_total_recv,
                rank, world_size, dev, args):
    """Baseline vs fused numerical consistency check.

    Flow
    ----
    1. Run baseline once (moe_gemm2 + combine); clone outputs.
    2. Zero ``shmem_comb_inp_*`` / ``shmem_comb_out_*`` + global barrier so
       fused starts clean and is not polluted by baseline leftovers.
    3. Run fused once (fused_gemm2 + combine_no_stage1); clone outputs.
    4. Compare ``out_tok`` and ``out_wts``.

    Tolerances
    ----------
    GEMM compute and P2P scatter share an identical floating-point path; the
    only difference is "local store + remote vec4 copy" vs "direct remote
    store", so fragment data should be bit-exact. The threshold here is
    intentionally loose to tolerate possible future accumulation reordering.
    """
    if rank == 0:
        sep = "=" * 78
        print(f"\n{sep}\n  VERIFY  baseline vs fused  EP={world_size}\n{sep}")

    # ── Step 1: baseline ──────────────────────────────────────────────────────
    if rank == 0:
        print("[verify] step 1: running baseline (moe_gemm2 → combine)")
    # Reset once to start from a clean state.
    _reset_combine_shmem(disp_op)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    # The combine kernel zeros total_recv at the end of Stage 2; if fused is
    # run right after baseline (combine_no_stage1 still uses total_recv to
    # drive the Stage 1 weight loop), the loop runs 0 iterations -> fused_out_wts
    # is all zero. Snapshot the true value before baseline and restore after.
    _saved_total_recv = disp_op.total_recv.detach().clone()
    # accumulate=True takes the atomic_fadd path; gemm2_out MUST be zeroed
    # before every launch. Otherwise _build_gemm2_callable's flyc.compile()
    # in setup has already run GEMM2 once, and this launch accumulates frag
    # onto leftover -> gemm2_out = 2*frag, which is the root cause of
    # base_out_tok = 2*fused_out_tok (fused uses plain store, unaffected).
    gemm2_in["gemm2_out"].zero_()
    torch.cuda.synchronize()
    # Snapshot gemm2_out immediately after the GEMM2 launch inside baseline_chain
    # to diagnose where base_out_tok=0 originates (GEMM2 itself vs the stage1/3 P2P path).
    gemm2_launch()
    torch.cuda.synchronize()
    # Debug: with FLYDSL_VERIFY_FORCE_GEMM2_OUT_PATTERN=1, overwrite gemm2_out
    # with a known non-zero pattern (rank+1 scalar) to isolate GEMM2 vs combine.
    # If base_out_tok is still all zero, the problem is in combine stage 1/3
    # (independent of GEMM2).
    if os.environ.get("FLYDSL_VERIFY_FORCE_GEMM2_OUT_PATTERN", "0") == "1":
        gemm2_in["gemm2_out"].fill_(float(rank) + 1.0)
        torch.cuda.synchronize()
    _gemm2_out_snap = gemm2_in["gemm2_out"].detach().clone()
    if rank == 0:
        _g32 = _gemm2_out_snap.float()
        _g_isnan = torch.isnan(_g32)
        _g_isinf = torch.isinf(_g32)
        _g_finite = ~(_g_isnan | _g_isinf)
        _g_finite_vals = _g32[_g_finite]
        if _g_finite_vals.numel() == 0:
            _gmin = _gmax = _gabs = float("nan")
        else:
            _gmin = _g_finite_vals.min().item()
            _gmax = _g_finite_vals.max().item()
            _gabs = _g_finite_vals.abs().mean().item()
        print(f"  [rank {rank}] post-GEMM2 gemm2_out: "
              f"shape={tuple(_gemm2_out_snap.shape)} "
              f"finite_min={_gmin:.4e} finite_max={_gmax:.4e} "
              f"finite_abs_mean={_gabs:.4e} "
              f"nz={int((_gemm2_out_snap != 0).sum().item())} "
              f"nan_count={int(_g_isnan.sum().item())} "
              f"inf_count={int(_g_isinf.sum().item())}", flush=True)
    base_tok, base_wts = disp_op.combine(
        gemm2_in["gemm2_out"], None, combine_idx,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    base_tok_s, base_wts_s = _snapshot_combine_out(disp_op)
    # Snapshot inp_wts immediately (written by baseline stage 1; read by stage 3 but not cleared).
    base_inp_wts_step1 = disp_op.shmem_comb_inp_wts.detach().clone()
    base_inp_tok_step1 = disp_op.shmem_comb_inp_tok.detach().clone()
    if rank == 0:
        print(f"  [rank {rank}] step1 baseline inp_wts: "
              f"nz={int((base_inp_wts_step1 != 0).sum().item())} "
              f"max={base_inp_wts_step1.float().abs().max().item():.4e}")
        print(f"  [rank {rank}] step1 baseline inp_tok: "
              f"nz={int((base_inp_tok_step1 != 0).sum().item())} "
              f"max={base_inp_tok_step1.float().abs().max().item():.4e} "
              f"abs_mean={base_inp_tok_step1.float().abs().mean().item():.4e}",
              flush=True)

    # --- Step 2: reset combine-related shmem ----------------------------------
    if rank == 0:
        print("[verify] step 2: zeroing shmem_comb_{inp,out}_* before fused run")
    _reset_combine_shmem(disp_op)
    # Restore total_recv (baseline combine wrote it to 0).
    disp_op.total_recv.copy_(_saved_total_recv)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # ── Step 3: fused ─────────────────────────────────────────────────────────
    if rank == 0:
        print("[verify] step 3: running fused (fused_gemm2 → combine_no_stage1)")

    # Debug: with FLYDSL_VERIFY_DUMP_INP_TOK=1, compare shmem_comb_inp_tok
    # after fused_gemm2's P2P scatter vs the baseline stage1 writes, to verify
    # that token P2P inside the GEMM2 epilogue lands completely (bypasses the
    # stage2/3 accumulation in combine).
    # Note: fused_gemm2 only scatters tokens (weights handled by
    # combine_no_stage1), so inp_wts is no longer compared here.
    if os.environ.get("FLYDSL_VERIFY_DUMP_INP_TOK", "0") == "1":
        os.environ["FLYDSL_FUSED_SKIP_COMBINE"] = "1"
        try:
            _ = _fused_chain(disp_op, fused_op, gemm2_in, combine_idx,
                             dispatch_total_recv)
        finally:
            os.environ.pop("FLYDSL_FUSED_SKIP_COMBINE", None)
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        fused_inp_tok_only = disp_op.shmem_comb_inp_tok.detach().clone()
        cfg_t = disp_op.cfg
        npes_t = cfg_t.world_size
        mt_t   = cfg_t.max_num_inp_token_per_rank
        hd_t   = cfg_t.hidden_dim
        b_tok = base_inp_tok_step1.view(npes_t, mt_t, hd_t)
        f_tok = fused_inp_tok_only.view(npes_t, mt_t, hd_t)
        tok_diff_per_pe = []
        for src_pe in range(npes_t):
            row_diff = ((b_tok[src_pe].to(torch.float32) -
                         f_tok[src_pe].to(torch.float32)).abs().sum(dim=-1) > 0)
            tok_diff_per_pe.append(int(row_diff.sum().item()))
        print(f"  [rank {rank}] inp_tok rows differ per src_pe: {tok_diff_per_pe}",
              flush=True)
        if rank == 0:
            f_stats_b = base_inp_tok_step1.float()
            f_stats_f = fused_inp_tok_only.float()
            print(f"  [rank {rank}] base_inp_tok stats:  "
                  f"min={f_stats_b.min().item():.4e} max={f_stats_b.max().item():.4e} "
                  f"abs_mean={f_stats_b.abs().mean().item():.4e} "
                  f"nz={int((base_inp_tok_step1 != 0).sum().item())}",
                  flush=True)
            print(f"  [rank {rank}] fused_inp_tok stats: "
                  f"min={f_stats_f.min().item():.4e} max={f_stats_f.max().item():.4e} "
                  f"abs_mean={f_stats_f.abs().mean().item():.4e} "
                  f"nz={int((fused_inp_tok_only != 0).sum().item())}",
                  flush=True)
            # Debug: print addr_tis contents (each t -> expected (dest_pe, dest_lid)).
            # baseline and fused both decode from the same addr_tis; if fused
            # writes to a (dest_pe, dest_lid) different from the decoded one,
            # the fused kernel has decoding/indexing problems. Only the first
            # 8 valid t are printed as a sanity check.
            tis_view = disp_op.shmem_tok_id_to_src.detach().clone().cpu()
            log2_mt = int(mt_t).bit_length() - 1
            mask_mt = mt_t - 1
            tot_recv = int(_saved_total_recv.item())
            # Find t values that PE 0 routes intra-rank (decoded dest_pe == rank);
            # these correspond to PE 0 writing to its own b_tok[rank, dest_lid] slot.
            self_pe = rank
            intra_t_list = []
            for t in range(tot_recv):
                enc = int(tis_view[t].item())
                if (enc >> log2_mt) == self_pe:
                    intra_t_list.append((t, enc, enc & mask_mt))
            print(f"  [rank {rank}] intra-rank t->(dest_lid) count: "
                  f"{len(intra_t_list)} of total_recv={tot_recv}", flush=True)
            print(f"  [rank {rank}] intra-rank routes (t, enc, dest_lid):", flush=True)
            for entry in intra_t_list[:12]:
                t, enc, dlid = entry
                # Check whether PE 0's local b_tok[self_pe=rank, dlid] has data.
                b_nz = int((b_tok[self_pe, dlid] != 0).sum().item())
                f_nz = int((f_tok[self_pe, dlid] != 0).sum().item())
                print(f"    t={t} enc={enc} dest_lid={dlid}  "
                      f"b_tok[{self_pe},{dlid}] nz={b_nz}  "
                      f"f_tok[{self_pe},{dlid}] nz={f_nz}",
                      flush=True)
            # Also check which lid fused actually wrote to (scan all intra-rank lid).
            f_intra_active = []
            b_intra_active = []
            for lid in range(mt_t):
                if int((f_tok[self_pe, lid] != 0).sum().item()) > 0:
                    f_intra_active.append(lid)
                if int((b_tok[self_pe, lid] != 0).sum().item()) > 0:
                    b_intra_active.append(lid)
            print(f"  [rank {rank}] base intra-rank lid with writes: "
                  f"{b_intra_active[:32]}", flush=True)
            print(f"  [rank {rank}] fused intra-rank lid with writes: "
                  f"{f_intra_active[:32]}", flush=True)
            for src_pe in range(min(npes_t, 2)):
                for lid in range(min(mt_t, 4)):
                    b_row = b_tok[src_pe, lid].float()
                    f_row = f_tok[src_pe, lid].float()
                    bnz = int((b_row != 0).sum().item())
                    fnz = int((f_row != 0).sum().item())
                    if bnz == 0 and fnz == 0:
                        continue
                    print(f"  [rank {rank}] src_pe={src_pe} lid={lid}: "
                          f"base nz={bnz} max={b_row.abs().max().item():.4e} "
                          f"sample={b_row[:4].tolist()} | "
                          f"fused nz={fnz} max={f_row.abs().max().item():.4e} "
                          f"sample={f_row[:4].tolist()}", flush=True)
            # Compare baseline single slot (atomic-8 accumulated value) against
            # fused 8 slots summed (Plan B expects them equal):
            #   baseline (intra-rank) writes b_tok[0, src_lid] = sum of 8 frag (atomic).
            #   fused writes raw slot (src_lid * k + s) for s in 0..7 -> the
            #   sum of these 8 slots should equal baseline's single slot.
            # Note: shmem_comb_inp_tok is an int16 1D buffer but the real data
            # is bf16, so we must view it back to bf16 before comparing.
            try:
                b_bf16 = base_inp_tok_step1.view(torch.bfloat16)
                f_bf16 = fused_inp_tok_only.view(torch.bfloat16)
                k_t = cfg_t.num_experts_per_token
                tot_recv = int(_saved_total_recv.item())
                log2_mt2 = int(mt_t).bit_length() - 1
                mask_mt2 = mt_t - 1
                npes_t2  = cfg_t.world_size
                # Pick a few intra-rank t where dest_pe == self_pe (rank 0 P2P to itself).
                cnt = 0
                for t in range(tot_recv):
                    enc = int(tis_view[t].item())
                    if (enc >> log2_mt2) != self_pe:
                        continue
                    src_lid = enc & mask_mt2
                    # baseline writes view[dest_pe=self_pe, src_lid] = 1 slot.
                    b_row = b_bf16.view(npes_t2, mt_t, hd_t)[self_pe, src_lid].float()
                    # fused writes raw slot (src_lid * k + s) for s in 0..k-1 in the
                    # 1D buffer, equivalent to view (max_recv, hd) row src_lid*k+s.
                    # Here we slice via view (max_recv, hd).
                    mr = npes_t2 * mt_t
                    f_2d = f_bf16.view(mr, hd_t)
                    slot_vals = [f_2d[src_lid * k_t + s].float() for s in range(k_t)]
                    f_sum = sum(slot_vals)
                    diff  = (b_row - f_sum).abs()
                    print(f"  [rank {rank}] CMP_SLOT t={t} src_lid={src_lid}: "
                          f"base[0,{src_lid}] max={b_row.abs().max().item():.4e}  "
                          f"fused_sum(slot[{src_lid*k_t}..{src_lid*k_t+k_t-1}]) "
                          f"max={f_sum.abs().max().item():.4e}  "
                          f"diff_max={diff.max().item():.4e}  diff_mean={diff.mean().item():.4e}",
                          flush=True)
                    cnt += 1
                    if cnt >= 4:
                        break
            except Exception as _e:
                print(f"  [rank {rank}] CMP_SLOT err: {_e}", flush=True)
        # After reset, run the full fused chain (stage1+stage2+stage3) for the out comparison.
        _reset_combine_shmem(disp_op)
        disp_op.total_recv.copy_(_saved_total_recv)
        torch.cuda.synchronize(); ms.shmem_barrier_all()

    _ = _fused_chain(
        disp_op, fused_op, gemm2_in, combine_idx, dispatch_total_recv,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    fused_tok_s, fused_wts_s = _snapshot_combine_out(disp_op)

    # --- Step 4: compare ------------------------------------------------------
    if rank == 0:
        print("[verify] step 4: comparing baseline vs fused outputs")
    # Print raw values per rank so 0=0 cannot fake a PASS.
    with torch.no_grad():
        print(f"  [rank {rank}] base_out_tok stats: "
              f"min={base_tok_s.float().min().item():.4e} "
              f"max={base_tok_s.float().max().item():.4e} "
              f"abs_mean={base_tok_s.float().abs().mean().item():.4e} "
              f"nz={int((base_tok_s != 0).sum().item())}", flush=True)
        print(f"  [rank {rank}] fused_out_tok stats: "
              f"min={fused_tok_s.float().min().item():.4e} "
              f"max={fused_tok_s.float().max().item():.4e} "
              f"abs_mean={fused_tok_s.float().abs().mean().item():.4e} "
              f"nz={int((fused_tok_s != 0).sum().item())}", flush=True)
        # P0 diagnostic: after fused_gemm2's P2P scatter, dump non-zero status
        # of mt*k=256 slots inside shmem_comb_inp_tok to locate the race source.
        #   slot_id = src_lid * k + j.
        # All 256 slots are expected non-zero; on failure some slot=0 means
        # the (src_lid, j) fragment did not land. Combined with the j-axis
        # stats this points to a dest_pe whose fragments all miss.
        _cfg_d = disp_op.cfg
        _mt_d2  = _cfg_d.max_num_inp_token_per_rank
        _k_d2   = _cfg_d.num_experts_per_token
        _hd_d2  = _cfg_d.hidden_dim
        _post_fused_inp_tok = disp_op.shmem_comb_inp_tok.detach().clone()
        _slot_view = _post_fused_inp_tok.view(torch.bfloat16).view(_mt_d2 * 8, -1)[:_mt_d2 * _k_d2]
        _slot_any = (_slot_view != 0).any(dim=-1).cpu().tolist()
        _slot_nz_cnt = sum(_slot_any)
        # Stats along the j axis: nz slot count at j = sum_{src_lid} _slot_any[src_lid*k+j].
        _per_j_nz = [sum(_slot_any[src*_k_d2 + j] for src in range(_mt_d2))
                     for j in range(_k_d2)]
        # List whether each src_lid received all 8 j fragments: sum_{j} slot_any[src*k+j].
        _per_src_nz = [sum(_slot_any[src*_k_d2 + j] for j in range(_k_d2))
                       for src in range(_mt_d2)]
        _full_src = [s for s, v in enumerate(_per_src_nz) if v == _k_d2]
        _empty_src = [s for s, v in enumerate(_per_src_nz) if v == 0]
        # P0 key diagnostic: dump whether sorted_token_ids contains all 32*8=256
        # (t, j) pairs. If valid rows in sorted are fewer than 256, upstream
        # moe_sorting dropped rows -> fused_gemm2 skipped those rows ->
        # the corresponding slots are zero.
        # num_valid_ids[0] is sorting's padded-total (includes padding);
        # num_valid_ids[1] is the actual input token count (= total_recv).
        _nv0 = int(gemm2_in["num_valid_ids"][0].item())
        _nv1 = int(gemm2_in["num_valid_ids"][1].item()) if gemm2_in["num_valid_ids"].numel() >= 2 else -1
        _sti = gemm2_in["sorted_token_ids"].detach().cpu()
        _sti_valid = (_sti[:_nv0] >> 24 < _k_d2)  # s_ok = (s < k)
        _sti_ok = int(_sti_valid.sum().item())
        print(f"  [rank {rank}] fused stage3 inp_tok: "
              f"slots_with_data={_slot_nz_cnt}/{_mt_d2 * _k_d2} (mt*k); "
              f"per-j nz: {_per_j_nz}; "
              f"full_src({len(_full_src)})={_full_src[:8]}; "
              f"empty_src({len(_empty_src)})={_empty_src[:8]}; "
              f"sort num_valid=[{_nv0},{_nv1}] sti_s_ok={_sti_ok}",
              flush=True)
        print(f"  [rank {rank}] base_out_wts stats: "
              f"min={base_wts_s.float().min().item():.4e} "
              f"max={base_wts_s.float().max().item():.4e} "
              f"abs_mean={base_wts_s.float().abs().mean().item():.4e} "
              f"nz={int((base_wts_s != 0).sum().item())}", flush=True)
        print(f"  [rank {rank}] fused_out_wts stats: "
              f"min={fused_wts_s.float().min().item():.4e} "
              f"max={fused_wts_s.float().max().item():.4e} "
              f"abs_mean={fused_wts_s.float().abs().mean().item():.4e} "
              f"nz={int((fused_wts_s != 0).sum().item())}", flush=True)
        print(f"  [rank {rank}] base_out_wts[0,:] = "
              f"{base_wts_s[0, :].float().tolist()}", flush=True)
        print(f"  [rank {rank}] fused_out_wts[0,:] = "
              f"{fused_wts_s[0, :].float().tolist()}", flush=True)
    cfg = disp_op.cfg
    # combine output shape = [cur_rank_num_token, hidden_dim]; the "valid"
    # range is the number of tokens this PE originally held (input rows
    # before dispatch), i.e. the cur_tok passed when calling combine
    # (default = cfg.max_num_inp_token_per_rank).
    #
    # Note: do NOT use dispatch_total_recv -- it is the number of tokens this
    # PE *received*, and combine clears total_recv at the end of stage 2 (see
    # `buffer_store(0, _r_trecv, 0)` in the kernel), so reading it after the
    # baseline run returns 0.
    valid_tok = max(0, min(int(cfg.max_num_inp_token_per_rank), base_tok_s.shape[0]))
    if rank == 0:
        print(f"        valid token range: [0, {valid_tok}) of shape {tuple(base_tok_s.shape)}")
    pass_tok = _compare_two(
        "base_out_tok", base_tok_s[:valid_tok],
        "fused_out_tok", fused_tok_s[:valid_tok],
        rank=rank, atol_abs=2e-2, atol_rel=5e-2,
    )
    pass_wts = _compare_two(
        "base_out_wts", base_wts_s[:valid_tok],
        "fused_out_wts", fused_wts_s[:valid_tok],
        rank=rank, atol_abs=1e-5, atol_rel=1e-5,
    )

    # Aggregate pass/fail across ranks (SUM over rank axis to locate failing ranks).
    local = torch.tensor([1 if pass_tok else 0, 1 if pass_wts else 0],
                         dtype=torch.int32, device=dev)
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    all_pass_tok = bool(local[0].item())
    all_pass_wts = bool(local[1].item())

    pass_tok_vec = torch.tensor([1 if pass_tok else 0] * world_size,
                                dtype=torch.int32, device=dev)
    pass_tok_vec.zero_(); pass_tok_vec[rank] = (1 if pass_tok else 0)
    dist.all_reduce(pass_tok_vec, op=dist.ReduceOp.SUM)
    if rank == 0:
        fail_ranks = [i for i in range(world_size) if pass_tok_vec[i].item() == 0]
        if fail_ranks:
            print(f"  [verify] out_tok FAIL on ranks: {fail_ranks}")

    if rank == 0:
        print(f"\n  RESULT (all-reduce min): "
              f"out_tok={'PASS' if all_pass_tok else 'FAIL'}, "
              f"out_wts={'PASS' if all_pass_wts else 'FAIL'}")
        print("=" * 78)


# --- Mode entry points (profile+cudagraph and bench+eager) -----------------
def _not_impl(name: str):
    raise NotImplementedError(
        f"mode '{name}' not yet implemented in this acceptance script. "
        "Only `--mode profile --cudagraph` and `--mode bench --no-cudagraph` "
        "are wired."
    )


def _run_bench_eager(chain_fn, op_tag: str,
                     rank: int, world_size: int, dev: torch.device,
                     warmup: int, iters: int):
    """Eager (no-cudagraph, no torch.profiler) bench loop.

    rocprofv3-friendly: every call goes through the real hipLaunchKernel path
    so rocprofv3 / ATT trace can see every internal kernel name in the chain.
    Uses torch.cuda.Event for end-to-end timing; all-reduce min/max/avg across ranks.
    """
    ms.shmem_barrier_all()
    if rank == 0:
        print(f"\n[bench+eager] {op_tag} warmup×{warmup} iters×{iters} "
              f"(no cudagraph / no torch.profiler — rocprofv3-friendly)")
    for _ in range(warmup):
        chain_fn()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        chain_fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    local_avg = sum(times_ms) / len(times_ms)
    local_min = min(times_ms)
    local_max = max(times_ms)

    t = torch.tensor([local_avg, local_min, local_max],
                     dtype=torch.float64, device=dev)
    t_max = t.clone()
    t_min = t.clone()
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"[bench+eager] {op_tag} per-iter (us):  "
              f"avg={local_avg*1000:.1f}  min={local_min*1000:.1f}  "
              f"max={local_max*1000:.1f}  (local rank 0)")
        print(f"[bench+eager] {op_tag} all-rank avg us: "
              f"min={t_min[0].item()*1000:.1f}  "
              f"max={t_max[0].item()*1000:.1f}")
    return {"avg_us": local_avg * 1000,
            "min_us": local_min * 1000,
            "max_us": local_max * 1000}


def run_acceptance(rank, world_size, args):
    dev = torch.device("cuda", rank)
    cur_tok = args.max_tokens
    k       = args.k
    # gemm2_scale_headroom is fully script-managed (no CLI knob). The A2/W2
    # e8m0 micro-scale only matters numerically in verify, and only when combine
    # casts the (large, fp4-fed) GEMM2 output through fp8 (fp8_direct_cast),
    # where it would saturate to NaN; headroom=4 (256x) brings it inside the
    # fp8e4m3 +/-448 safe range. profile/bench is value- and perf-independent,
    # so it stays 0. Stored on args so downstream readers see the value.
    args.gemm2_scale_headroom = 4 if (
        args.mode == "verify" and args.combine_quant_type == "fp8_direct_cast"
    ) else 0
    if rank == 0:
        print(f"[setup] gemm2_scale_headroom={args.gemm2_scale_headroom} "
              f"(auto; mode={args.mode}, combine_quant_type={args.combine_quant_type})")
    # recv buffer capacity per rank (M). With per-(token,destPE) dedup in both
    # flydsl and mori dispatch, each source rank contributes a given token at
    # most once to any dest PE, so the recv upper bound is exactly ws*M with
    # M == bs (cur_tok) -- no headroom factor needed (verified across the full
    # aiter-config matrix: 0 overflow at M==bs for both comm backends).
    cap_tok = cur_tok

    # ── cfg / dispatch op ───────────────────────────────────────────────────
    _dtype = DTYPE_MAP.get(args.dispatch_dtype, torch.bfloat16)
    cfg = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world_size,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=cap_tok,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=k,
        data_type=_dtype,
        dispatch_warp_num_per_block=args.dispatch_warp_per_block,
        dispatch_block_num=args.dispatch_block_num,
        combine_warp_num_per_block=args.combine_warp_per_block,
        combine_block_num=args.combine_block_num,
        chip=args.chip,
        zero_copy=args.zero_copy,
        enable_std_moe=args.enable_std_moe,
        scale_dim=args.scale_dim,
        scale_type_size=args.scale_type_size,
        quant_type=args.combine_quant_type,
        use_token_flag_sync=args.token_flag_sync,
    )
    args.max_num_inp_token_per_rank = cap_tok  # consumed by _build_gemm2_static_inputs

    # ── FP4 hard constraints ────────────────────────────────────────────────
    # FlyDSL FP4 mfma_scale_x128 path implicit assumptions:
    #   * tile_k >= 256 (one scale layout cell covers 128 fp4 elements; <256 silently zeros).
    #   * inter_dim / model_dim >= 256 and divisible by 256.
    # Violating these does not raise; GEMM2 silently outputs zero. To avoid
    # stepping on this again, the verify/bench entry raises explicitly.
    # accumulate=True/False is no longer hard-constrained; it is controlled by --gemm2-accumulate.
    if args.gemm2_b_dtype == "fp4" or args.gemm2_a_dtype == "fp4":
        if args.tile_k2 < 256:
            raise ValueError(
                f"FP4 GEMM2 requires --tile-k2 >= 256 (got {args.tile_k2}); "
                "smaller tile_k will silently produce gemm2_out=0."
            )
        if args.inter_dim < 256 or args.inter_dim % 256 != 0:
            raise ValueError(
                f"FP4 GEMM2 requires --inter-dim multiple of 256 (got {args.inter_dim})."
            )
        if args.hidden_dim < 256 or args.hidden_dim % 256 != 0:
            raise ValueError(
                f"FP4 GEMM2 requires --hidden-dim multiple of 256 (got {args.hidden_dim})."
            )

    if rank == 0:
        _max_recv = world_size * cur_tok
        _a2_rows  = _max_recv * max(1, k)
        print(f"\n{'='*78}")
        print(f"[acceptance] EP={world_size}, bs={cur_tok}, "
              f"h={cfg.hidden_dim}, inter={args.inter_dim}, k={k}, "
              f"epr={cfg.num_experts_per_rank}")
        print(f"  GEMM2: a={args.gemm2_a_dtype}/b={args.gemm2_b_dtype}, "
              f"tile_m2={args.tile_m2}, tile_n2={args.tile_n2}, "
              f"tile_k2={args.tile_k2}, persist_m={args.persist_m}, "
              f"accumulate={args.gemm2_accumulate}")
        print(f"  GEMM2 layout: tokens_in=max_recv={_max_recv}, topk={k}, "
              f"A2 rows={_a2_rows} (= bs*ep*topk)")
        print(f"  bench-op={args.bench_op}, "
              f"fused_op_available={HAS_FUSED_OP}")
        if not HAS_FUSED_OP:
            print(f"  [warn] fused op import failed: {_FUSED_IMPORT_ERR}")
        print(f"{'='*78}")

    disp_op = FlyDSLDispatchCombineIntraNodeOp(cfg)

    fused_op = None
    if HAS_FUSED_OP and args.bench_op in ("fused", "both"):
        fused_op = MegaMoeStage2(
            comb_cfg=cfg,
            comb_op=disp_op,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m2, tile_n=args.tile_n2, tile_k=args.tile_k2,
            persist_m=args.persist_m,
            sort_block_m=args.sort_block_m,
            b_nt=args.b_nt,
            a_dtype=args.gemm2_a_dtype, b_dtype=args.gemm2_b_dtype,
            xcd_swizzle=args.xcd_swizzle,
            use_token_flag_sync=args.token_flag_sync,
        )

    ms.shmem_barrier_all()

    # --- inputs ---------------------------------------------------------------
    inp, wts, idx = _build_dispatch_inputs(rank, world_size, dev, args, cfg)

    # scales / packed_recv_x (kept to align with the existing dispatch op interface; unused here by default).
    scales = None
    if cfg.scale_dim > 0 and cfg.scale_type_size > 0:
        _sc_bytes = cfg.scale_dim * cfg.scale_type_size
        scales = torch.randn(cur_tok, _sc_bytes // 4,
                             dtype=torch.float32, device=dev).contiguous()
        scales = scales.view(torch.uint8).view(cur_tok, _sc_bytes)

    # --- One-shot dispatch (to obtain total_recv used for GEMM2 input setup) -
    # combine depends on disp_op's internal shmem_disp_out_* tables, which
    # are populated by dispatch; after one run these tables stay stable, so
    # combine in the chain just reads the static views.
    if rank == 0:
        _rt = getattr(args, "routing", "random")
        _hr = int(getattr(args, "gemm2_scale_headroom", 0))
        print(f"[setup] routing={_rt} sorting=flydsl; "
              f"moe_sorting available = {HAS_MOE_SORTING}; "
              f"gemm2_scale_headroom={_hr} (e8m0 scale = {127 - _hr}, "
              f"GEMM2 magnitude × 2^-{2*_hr})")
        if not HAS_MOE_SORTING:
            print(f"[setup] ERROR: moe_sorting unavailable: "
                  f"{_MOE_SORTING_ERR}; will fail in _build_gemm2_static_inputs.")
        print(f"[setup] running one-shot dispatch to populate routing tables…")
    disp_ret = disp_op.dispatch(inp, wts, scales, idx)
    combine_idx = disp_ret[3]                # shmem_disp_out_idx view
    dispatch_total_recv = disp_ret[4]        # shmem total_recv scalar
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    if rank == 0 and args.bench_op in ("fused", "both") and HAS_FUSED_OP:
        print(f"[setup] dispatch total_recv (rank0) = "
              f"{int(dispatch_total_recv.item())}")

    # Note (moe_sorting ordering): build MUST run *before* hard-reset.
    # _build_gemm2_static_inputs calls moe_sorting once (first JIT + buffer
    # ownership), which requires disp_op.total_recv to still hold the real
    # value written by the setup dispatch. After hard-reset clears
    # total_recv=0, num_local_tokens reads 0 and sorted ends up empty (only
    # buffer shapes valid). At chain time each replay reruns dispatch, after
    # which _run_moe_sorting refills with a fresh total_recv.
    # sorted_*。
    gemm2_in = _build_gemm2_static_inputs(
        rank, world_size, dev, args, cfg, disp_op=disp_op,
    )

    # End-to-end semantic self-check: on the zero-copy path GEMM2 MUST write
    # outputs directly into ``shmem_comb_inp_tok``; otherwise combine Stage1
    # is skipped and the peer-side P2P reads stale shmem data. Print a
    # banner before the op-level raise so rank 0 can immediately see which
    # tensor owns gemm2_out.
    if rank == 0 and args.bench_op in ("baseline", "both"):
        _g_ptr = gemm2_in["gemm2_out"].data_ptr()
        _shm_ptr = disp_op.shmem_comb_inp_tok.data_ptr()
        _is_shmem = _g_ptr == _shm_ptr
        print(f"[setup] gemm2_out.data_ptr() = 0x{_g_ptr:x}; "
              f"shmem_comb_inp_tok.data_ptr() = 0x{_shm_ptr:x}; "
              f"is_zero_copy_buffer={_is_shmem}; cfg.zero_copy={cfg.zero_copy}")
        if cfg.zero_copy and not _is_shmem:
            raise RuntimeError(
                "zero-copy on but gemm2_out is NOT the registered shmem buffer; "
                "combine Stage1 will skip the staging copy and peer reads will "
                "see stale data."
            )

    # --- Option A hard-reset: only clear the *local* counters left by setup dispatch.
    # (dest_pe_ctr / disp_bar / comb_bar / total_recv / disp_grid_bar),
    # Do NOT clear cross-device shmem buffers (shmem_xdev_bar_mem uses the
    # monotonic cur_flag pattern managed by mori; shmem_comb_inp_* is
    # naturally overwritten by the next chain). Zeroing any shmem_* causes
    # mori shmem's internal cur_flag to diverge from the actual buffer
    # contents, surfacing as hipErrorInvalidValue at the first fused gemm2
    # launch during capture.
    #
    # Only needed in profile/cudagraph mode (dispatch in the chain refills
    # these counters). verify mode does not use cudagraph and does not rerun
    # dispatch, so hard-reset would clear total_recv to 0 -> combine Stage 1
    # runs 0 iterations -> all-zero output.
    if args.chain_include_dispatch and args.mode != "verify":
        if rank == 0:
            print(f"[setup] hard-reset disp_op local counters before capture "
                  f"(chain_include_dispatch=True)…")
        ms.shmem_barrier_all()
        torch.cuda.synchronize()
        disp_op.dest_pe_ctr.zero_()
        disp_op.disp_bar.zero_()
        disp_op.comb_bar.zero_()
        disp_op.total_recv.zero_()
        disp_op.disp_grid_bar.zero_()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
    if rank == 0:
        _dump_gemm2_inputs(args, gemm2_in, gemm2_in["out_dtype"])
    gemm2_launch = _build_gemm2_callable(args, gemm2_in, gemm2_in["out_dtype"])

    meta = dict(
        world_size=world_size,
        max_tokens=cur_tok,
        hidden_dim=cfg.hidden_dim,
        inter_dim=args.inter_dim,
        k=k,
        num_experts_per_rank=args.num_experts_per_rank,
        warmup=args.warmup, iters=args.iters,
        dispatch_block_num=cfg.dispatch_block_num,
        dispatch_warp_per_block=cfg.dispatch_warp_num_per_block,
        combine_block_num=cfg.combine_block_num,
        combine_warp_per_block=cfg.combine_warp_num_per_block,
        gemm2_a_dtype=args.gemm2_a_dtype,
        gemm2_b_dtype=args.gemm2_b_dtype,
        tile_m2=args.tile_m2, tile_n2=args.tile_n2, tile_k2=args.tile_k2,
        persist_m=args.persist_m,
        bench_op=args.bench_op,
    )

    # Path switch (both verify and profile entry points read this).
    test_baseline = args.bench_op in ("baseline", "both")
    test_fused    = args.bench_op in ("fused",    "both") and fused_op is not None

    # --- Baseline comm backend (flydsl native / mori reference) -------------
    # Default: baseline reuses the flydsl disp_op + shared gemm2_in /
    # gemm2_launch / combine_idx. With --baseline-comm mori we build a separate
    # mori adapter + its own baseline_gemm2_in (gemm2_out lands in mori's
    # registered combine buffer under zero-copy), so the baseline chain becomes
    # "mori dispatch + flydsl gemm2 + mori combine". The fused path always keeps
    # the flydsl disp_op + shared gemm2_in (unaffected).
    baseline_disp_op      = disp_op
    baseline_gemm2_in     = gemm2_in
    baseline_gemm2_launch = gemm2_launch
    baseline_combine_idx  = combine_idx
    if args.baseline_comm == "mori" and test_baseline:
        if args.mode == "verify":
            raise ValueError(
                "--baseline-comm mori does not support verify mode (numeric "
                "self-check relies on flydsl shmem buffers); use profile/bench."
            )
        # mori combine reads the gemm2 output cross-PE (peers P2P-read it),
        # so it must live in mori's registered symmetric buffer; a plain
        # external buffer faults once dispatch re-runs in the chain. Force
        # zero-copy so gemm2 writes straight into mori's registered combine
        # input buffer (gemm2_out == shmem_comb_inp_tok).
        if not cfg.zero_copy and rank == 0:
            print("[setup] baseline-comm=mori: forcing zero-copy (combine "
                  "reads the registered symmetric gemm2 output buffer).")
        cfg.zero_copy = True
        _mori_zc = True
        if rank == 0:
            print(f"[setup] baseline-comm=mori: building mori "
                  f"EpDispatchCombineOp as baseline dispatch/combine "
                  f"(gemm2 still flydsl, zero_copy={_mori_zc})...")
        _mori_op = _build_mori_op(rank, world_size, cfg)
        mori_adapter = _MoriDispOpAdapter(_mori_op, gemm2_in["out_dtype"])
        # One mori dispatch to populate routing tables; the first moe_sorting
        # inside _build_gemm2_static_inputs needs a real total_recv.
        _mori_ret = mori_adapter.dispatch(inp, wts, scales, idx)
        baseline_combine_idx = _mori_ret[3]
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        baseline_gemm2_in = _build_gemm2_static_inputs(
            rank, world_size, dev, args, cfg, disp_op=mori_adapter,
            force_zero_copy_out=_mori_zc,
        )
        baseline_gemm2_launch = _build_gemm2_callable(
            args, baseline_gemm2_in, baseline_gemm2_in["out_dtype"],
        )
        baseline_disp_op = mori_adapter
        if rank == 0:
            _g = baseline_gemm2_in["gemm2_out"].data_ptr()
            _sp = mori_adapter.shmem_comb_inp_tok.data_ptr()
            print(f"[setup] mori baseline: gemm2_out=0x{_g:x} "
                  f"mori_comb_inp=0x{_sp:x} is_zero_copy_buf={_g == _sp} "
                  f"zero_copy={_mori_zc} "
                  f"total_recv(rank0)={int(mori_adapter.total_recv.item())}")
            if _mori_zc and _g != _sp:
                raise RuntimeError(
                    "mori baseline zero-copy on but gemm2_out is not mori's "
                    "registered combine input buffer; combine Stage1 reads "
                    "stale data."
                )

    # --- Mode dispatch --------------------------------------------------------
    if args.mode == "verify":
        # verify only makes sense with --bench-op both (compares baseline vs fused).
        if args.bench_op != "both":
            if rank == 0:
                print(f"[verify] requires --bench-op both (got {args.bench_op!r}); "
                      "skipping verify and falling through")
            return
        if not test_fused:
            if rank == 0:
                print("[verify] fused op unavailable; skipping verify")
            return
        _run_verify(
            disp_op, fused_op,
            gemm2_in, gemm2_launch, combine_idx, dispatch_total_recv,
            rank, world_size, dev, args,
        )
        return
    if args.mode == "bench":
        if args.cudagraph:
            _not_impl("bench+cudagraph")
        # Eager bench: rocprofv3 / ATT trace friendly (no cudagraph, no torch.profiler).
        # Whether the chain includes dispatch is controlled by --chain-include-dispatch;
        # without dispatch the fused-path combine_no_stage1 also runs empty (see _fused_chain doc).
        chain_disp_inputs = (
            (inp, wts, scales, idx) if args.chain_include_dispatch else None
        )
        bench_results = {}
        if test_baseline:
            def _bl():
                return _baseline_chain(
                    baseline_disp_op, baseline_gemm2_launch, baseline_gemm2_in,
                    baseline_combine_idx,
                    dispatch_inputs=chain_disp_inputs,
                )
            bench_results["baseline"] = _run_bench_eager(
                _bl, "baseline", rank, world_size, dev,
                warmup=args.warmup, iters=args.iters,
            )
        if test_fused:
            def _fu():
                return _fused_chain(
                    disp_op, fused_op, gemm2_in, combine_idx,
                    dispatch_total_recv,
                    dispatch_inputs=chain_disp_inputs,
                )
            bench_results["fused"] = _run_bench_eager(
                _fu, "fused", rank, world_size, dev,
                warmup=args.warmup, iters=args.iters,
            )
        if rank == 0 and "baseline" in bench_results and "fused" in bench_results:
            b = bench_results["baseline"]["avg_us"]
            f = bench_results["fused"]["avg_us"]
            print(f"\n[bench+eager] speedup baseline/fused = {b/f:.3f}x  "
                  f"(baseline {b:.1f} us → fused {f:.1f} us)")
        return
    if args.mode == "profile" and not args.cudagraph:
        _not_impl("profile+eager")

    # Only supported path: profile + cudagraph.
    assert args.mode == "profile" and args.cudagraph, \
        f"unsupported (mode={args.mode}, cudagraph={args.cudagraph})"

    # --output-dir not given -> don't persist anything. The profiler trace JSON
    # is still needed to compute stats, so write it to a scratch temp dir and
    # remove it at the end (_persist_out gates the final "saved to" message).
    # Only the profile path reaches here, so verify/bench never create a dir.
    _persist_out = args.output_dir is not None
    if _persist_out:
        out_dir = os.path.join(args.output_dir, f"ep{world_size}_bs{cur_tok}")
    else:
        out_dir = tempfile.mkdtemp(prefix=f"moe_gemm2_ep{world_size}_bs{cur_tok}_")
    os.makedirs(out_dir, exist_ok=True)

    base_stats = None
    fused_stats = None

    # Default: rerun dispatch inside the chain (option A, mori best practice).
    # Each replay lets dispatch rewrite routing tables / total_recv so combine
    # does not run empty. The old behavior (combine empty) can be reproduced
    # with --no-chain-include-dispatch.
    chain_disp_inputs = (inp, wts, scales, idx) if args.chain_include_dispatch else None
    if rank == 0 and args.chain_include_dispatch:
        print(f"[chain] including dispatch in cudagraph (mori best-practice; "
              f"avoids total_recv being zeroed by combine)")

    # Only pass disp_op to capture (so it runs reset) when the chain includes dispatch (option A).
    _capture_disp_op = disp_op if args.chain_include_dispatch else None
    # Pass disp_op to capture (so the non-mori path runs reset) only when the
    # chain includes dispatch. The mori_capture path ignores reset and just
    # barriers, mirroring test_profiler_dispatch_combine._cudagraph_capture_mori.
    _baseline_capture_disp_op = (
        baseline_disp_op if args.chain_include_dispatch else None
    )

    if test_baseline:
        def _baseline():
            return _baseline_chain(baseline_disp_op, baseline_gemm2_launch,
                                   baseline_gemm2_in, baseline_combine_idx,
                                   dispatch_inputs=chain_disp_inputs)

        base_stats = profile_cudagraph_chain(
            _baseline, "baseline",
            rank, world_size, dev,
            iters=args.iters, out_dir=out_dir, meta=meta,
            disp_op=_baseline_capture_disp_op,
            mori_capture=(args.baseline_comm == "mori"),
        )

    if test_fused:
        def _fused():
            return _fused_chain(
                disp_op, fused_op, gemm2_in, combine_idx, dispatch_total_recv,
                dispatch_inputs=chain_disp_inputs,
            )

        fused_stats = profile_cudagraph_chain(
            _fused, "fused",
            rank, world_size, dev,
            iters=args.iters, out_dir=out_dir, meta=meta,
            disp_op=_capture_disp_op,
        )

    if rank == 0 and base_stats is not None and fused_stats is not None:
        _print_speedup(base_stats, fused_stats, world_size)

    if _persist_out:
        if rank == 0:
            print(f"\n[acceptance] All trace/JSON saved to: {out_dir}/")
    else:
        # Scratch dir: drop it (no --output-dir was given, so nothing persists).
        shutil.rmtree(out_dir, ignore_errors=True)
        if rank == 0:
            print("\n[acceptance] no --output-dir given; traces not saved.")


# ─── Worker / CLI ─────────────────────────────────────────────────────────────
def _worker(rank, world_size, args, master_port):
    setup_distributed(rank, world_size, master_port)
    try:
        run_acceptance(rank, world_size, args)
    except Exception as e:
        import traceback as tb
        print(f"[rank {rank}] ERROR: {e}")
        tb.print_exc()
    finally:
        cleanup()


def _parse_args():
    p = argparse.ArgumentParser(
        description="moe_gemm2 + combine end-to-end acceptance script (dispatch runs once at setup)"
    )
    # Shape / routing
    p.add_argument("--world-size",           type=int, default=8)
    p.add_argument("--max-tokens",           type=int, default=32,
                   help="per-rank dispatch input token count (bs); max_recv = world_size * bs. "
                        "Default 32 matches the production balanced case (bs=32, ep=8, topk=8).")
    p.add_argument("--hidden-dim",           type=int, default=7168,
                   help="GEMM2 output dim (== combine token dim == dispatch token dim)")
    p.add_argument("--inter-dim",            type=int, default=2048,
                   help="GEMM2 input dim (GEMM1 output dim; this script does not run GEMM1). "
                        "Default 2048 matches production a4w4 (ut_per1x32.py: "
                        "model_dim=7168, inter_dim=2048).")
    p.add_argument("--num-experts-per-rank", type=int, default=32)
    p.add_argument("--k",                    type=int, default=8,
                   help="MoE top-k: drives both dispatch routing topk and GEMM2 compile-time "
                        "topk (A2 row addressing t*topk+s, output atomic accumulate over s).")
    p.add_argument("--routing",
                   choices=["random", "atomic1_8pe", "atomic8_1pe", "atomic2_4pe"],
                   default="random",
                   help="How dispatch input idx is constructed (atomicN_Mpe naming: each "
                        "token's k experts spread over M PEs, with N=atomic_per_pe=k/M "
                        "hits per PE -> the same GEMM2 output row faces N atomic_fadd contenders). "
                        "random: each token picks random experts across k PEs (default, legacy). "
                        "atomic1_8pe: deterministic round-robin; k experts across k PEs, "
                        "atomic_per_pe=1, dispatch dedup not triggered; matches production "
                        "balanced EP routing (recommended for accuracy verify). "
                        "atomic8_1pe: k experts are k consecutive local_eid on a single "
                        "dest_pe (atomic-k accumulation worst case), balanced across/within ranks. "
                        "Requires epr%%k==0 (32/8=4). "
                        "atomic2_4pe: k experts spread across *4* PEs with k/4=2 atomics each "
                        "(moderate atomic-2 contention); dest_pe = (g%%ws + j_group) mod ws "
                        "over 4 contiguous PEs, local_eid uses g//ws+j to avoid lattice "
                        "collapse -> inter-rank hits = cur_tok*k perfectly balanced, "
                        "per (PE, local_eid) cell hit count = cur_tok*k/epr. "
                        "Requires k%%4==0 (8/4=2) and ws>=4 (8).")
    # ── dispatch / combine ─────────────────────────────────────────────────
    p.add_argument("--dispatch-dtype",         dest="dispatch_dtype",
                   type=str, default="bf16", choices=list(DTYPE_MAP.keys()),
                   help="dispatch token dtype (combine dtype follows the gemm2 output)")
    # dispatch / combine each get their own grid geometry; empirically combine
    # likes more warps per block than dispatch (the XDB barrier is
    # wave-cooperative). Defaults mirror the op's per-phase defaults.
    p.add_argument("--dispatch-block-num",     dest="dispatch_block_num",
                   type=int, default=128, help="dispatch kernel grid block count")
    p.add_argument("--dispatch-warp-per-block", dest="dispatch_warp_per_block",
                   type=int, default=4, help="dispatch kernel warps per block")
    p.add_argument("--combine-block-num",      dest="combine_block_num",
                   type=int, default=128, help="combine kernel grid block count")
    p.add_argument("--combine-warp-per-block", dest="combine_warp_per_block",
                   type=int, default=8, help="combine kernel warps per block")
    p.add_argument("--chip",                 type=str, default="gfx950")
    p.add_argument("--zero-copy", dest="zero_copy",
                   action="store_true", default=False)
    p.add_argument("--enable-std-moe", action="store_true", default=False)
    p.add_argument("--scale-dim",       type=int, default=0)
    p.add_argument("--scale-type-size", type=int, default=0)
    p.add_argument("--combine-quant-type",   dest="combine_quant_type",
                   type=str, default="none", choices=["none", "fp8_direct_cast"])
    p.add_argument("--baseline-comm", choices=["flydsl", "mori"], default="flydsl",
                   help="Whether the baseline chain dispatch/combine use "
                        "flydsl's own kernels or mori. mori: swap dispatch+"
                        "combine to mori (gemm2 stays flydsl) for a 'mori "
                        "dispatch + flydsl gemm2 + mori combine' baseline; "
                        "needs profile/bench mode + bf16/fp16/fp8 dtype.")
    # ── fused (gemm2 + combine) ─────────────────────────────────────────────
    # token-level-sync: per-token flag sync switch. When on, dispatch / combine
    # kernels enable reset + spin-wait and the fused gemm2 epilogue adds
    # cross-device atomic_add. When off, the whole block is const_expr DCEd
    # and behaviour matches baseline exactly. Only observable under
    # --bench-op fused / both (the fused path).
    p.add_argument("--token-flag-sync", dest="token_flag_sync",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Enable the token-level-sync per-token flag cross-device sync "
                        "path; --no-token-flag-sync (default) disables it")
    # GEMM2
    # Default a=fp4/b=fp4 (production a4w4: GEMM1 output is SiLU + per-1x32
    # quantized to fp4 before feeding GEMM2; see ut_per1x32.py). The early
    # default was a=fp8 and its perf numbers no longer reflect production.
    p.add_argument("--gemm2-a-dtype",        type=str, default="fp4",
                   choices=["fp8", "fp4", "fp16", "int8"])
    p.add_argument("--gemm2-b-dtype",        type=str, default="fp4",
                   choices=["fp8", "fp4", "fp16", "int8", "int4"])
    p.add_argument("--tile-m2",              type=int, default=32)
    p.add_argument("--tile-n2",              type=int, default=128)
    p.add_argument("--tile-k2",              type=int, default=256,
                   help="GEMM2 tile_k; the FP4 path requires >=256")
    p.add_argument("--persist-m",            type=int, default=-1,
                   help="moe_gemm2 persistent block count along M")
    p.add_argument("--b-nt",                 dest="b_nt", type=int, default=2,
                   choices=[0, 2],
                   help="GEMM2 B(weight) load cache modifier: 2=streaming/non-temporal, 0=normal L2")
    p.add_argument("--sort-block-m",         dest="sort_block_m", type=int, default=0,
                   help="MoE sorting block_size for GEMM2 stage1; 0 means tile_m (must be a multiple of tile_m)")
    p.add_argument("--xcd-swizzle",          type=int, default=0,
                   help="moe_gemm2 cross-XCD swizzle factor (0 disables; recommend 8 on MI300)")
    p.add_argument("--gemm2-accumulate",     dest="gemm2_accumulate",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="GEMM2 epilogue reduce mode: atomic-add (default) vs "
                        "--no-gemm2-accumulate for plain-store (per-row dedicated slot)")
    # Mode
    p.add_argument("--mode", choices=["profile", "bench", "verify"], default="profile",
                   help="Only profile mode is supported in this skeleton")
    p.add_argument("--cudagraph", dest="cudagraph",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Use CUDAGraph capture/replay (default); --no-cudagraph runs eager")
    # --- Option A: capture dispatch into the chain too (mori best-practice).
    # Enabled by default; reproduce the old (broken) behaviour with --no-chain-include-dispatch.
    p.add_argument("--chain-include-dispatch", dest="chain_include_dispatch",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Place dispatch inside the cudagraph chain so each replay "
                        "rewrites routing tables / total_recv, avoiding empty replays "
                        "caused by combine clearing total_recv to 0 (default). "
                        "--no-chain-include-dispatch falls back to the old path "
                        "(dispatch runs once at setup; only the 1st chain does real "
                        "combine work, the rest run empty -- diagnostic only).")
    p.add_argument("--bench-op", choices=["baseline", "fused", "both"], default="baseline",
                   help="Which path to run")
    # Profile / output
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=20,
                   help="Profiler active iterations (effective = iters - skip(5))")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Dir to persist profiler trace/JSON. Omit to not save "
                        "anything (traces go to a temp dir, used for stats, then removed).")
    p.add_argument("--port",       type=int, default=29800)
    return p.parse_args()


def main():
    args = _parse_args()
    if "LOCAL_RANK" in os.environ:
        rank       = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        _worker(rank, world_size, args, master_port=args.port)
    else:
        ws = min(args.world_size, torch.cuda.device_count())
        if ws < args.world_size:
            print(f"[warn] available GPUs={torch.cuda.device_count()}, "
                  f"world_size adjusted: {args.world_size} -> {ws}")
        torch.multiprocessing.spawn(
            _worker, args=(ws, args, args.port),
            nprocs=ws, join=True,
        )


if __name__ == "__main__":
    main()
