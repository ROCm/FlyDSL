# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
端到端验收脚本：MoE GEMM2 + combine。

用于验证 fused_gemm2_combine 算子的精度与性能。

链路（dispatch 已被简化掉）：

    [setup, 一次性]   disp_op.dispatch(inp, wts, idx)
                      └─ 写好 shmem_disp_out_idx / out_wts / total_recv
                         （这些 shmem 表是 combine 的依赖；放进 setup 后
                          combine 直接读，无需在 chain 里重复跑 dispatch）

    [chain, 被 capture]
        baseline:  moe_gemm2  →  combine
        fused:     fused_gemm2_combine    （待实现）

测量方式 × 执行方式（共 4 种正交组合，本次仅实现 profile+cudagraph）：
  --mode       profile (torch.profiler) | bench (CUDA Event) | verify
  --cudagraph  不带=eager / 带=CUDAGraph capture+replay

  当前已实现：
    profile + cudagraph    （核心验收路径）
    verify                  （骨架预留，未实现完整对齐逻辑）
  其余 3 种组合抛 NotImplementedError，下次扩展时再补。

启动方式：
  # 仅跑 baseline（fused op 未实现时）
  torchrun --nproc_per_node=8 tests/kernels/test_profiler_moe_gemm2_combine.py \
      --mode profile --cudagraph --bench-op baseline \
      --max-tokens 512 --hidden-dim 7168 --inter-dim 4096 --k 8

  # baseline + fused 对比
  torchrun --nproc_per_node=8 tests/kernels/test_profiler_moe_gemm2_combine.py \
      --mode profile --cudagraph --bench-op both --fuse-mode auto

CUDAGraph 注意：
  - dispatch 在 setup 里跑过一次后，total_recv / out_idx 等都已落入 shmem
    buffer。chain 内 GEMM2 / combine 直接读取这些静态视图，可被合法 capture。
  - GEMM2 在 capture 时使用 max_recv 上界 + num_valid_ids 作早退依据。
  - replay 之间 **不做 dist.barrier()**，避免 ROCTracer 插桩与 P2P shmem
    操作互相干扰，导致 kernel 时间膨胀数十倍。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
    from kernels.mixed_moe_gemm2_combine_fused_op import (  # type: ignore
        FlyDSLMoeGemm2CombineOp,
    )
    # 模块级 READY=False 表示文件已就位但 kernel 实现还没接通；
    # 测试脚本据此 graceful skip fused 路径，不报错。
    HAS_FUSED_OP = bool(getattr(FlyDSLMoeGemm2CombineOp, "READY", False))
    _FUSED_IMPORT_ERR = "" if HAS_FUSED_OP else "FlyDSLMoeGemm2CombineOp.READY = False (kernel not yet wired)"
except Exception as _e:  # noqa: BLE001
    FlyDSLMoeGemm2CombineOp = None
    HAS_FUSED_OP = False
    _FUSED_IMPORT_ERR = repr(_e)


# ─── 分布式初始化 ─────────────────────────────────────────────────────────────
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


# ─── 数据/路由构造 ────────────────────────────────────────────────────────────
def _build_dispatch_inputs(rank, world_size, dev, args, cfg):
    """构造 dispatch 输入（input/weights/indices）。固定种子，便于多卡对齐。"""
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


def _build_gemm2_static_inputs(rank, world_size, dev, args, cfg, *, valid_recv=None):
    """为 GEMM2 / combine 构造 *静态* 输入（capture 期间不变）。

    关键点：
      - GEMM2 输入 A2 形状 [max_recv, inter_dim]，dtype=fp8（与 fp4 W2 配对）。
        实际 EP 场景下 A2 应由 GEMM1+SiLU 产生；本骨架为简化 perf 测量，直接
        随机初始化（accuracy 验证另走 verify 路径，本次未实现）。
      - W2 形状 [num_experts_per_rank, model_dim, inter_dim]，per-expert FP4。
      - 路由 sorted buffer 按 max_recv 上界构造，num_valid_ids 设为最大，
        让 GEMM2 在 cudagraph 中跑满负载（worst-case latency）。

    Parameters
    ----------
    valid_recv
        若给出，则 sorted_token_ids 只取 [0, valid_recv) 的 local-recv 槽位
        作为有效 token，其余仍以 max_recv 哨兵 padding。fused 路径必须传入
        实际 dispatch 后的 total_recv，否则 GEMM2 epilogue 会读到未被 dispatch
        写入的 addr_tis[t]（zeros / 残留），把结果 P2P scatter 到错误位置，
        进一步污染 combine 的 stage 3 输出。
    """
    epr        = args.num_experts_per_rank
    max_recv   = world_size * args.max_num_inp_token_per_rank
    model_dim  = args.hidden_dim   # GEMM2 输出维度 = combine 的 hidden_dim
    inter_dim  = args.inter_dim
    tile_m     = args.tile_m2
    a_dtype    = args.gemm2_a_dtype
    b_dtype    = args.gemm2_b_dtype

    torch.manual_seed(123 + rank)

    # ── A2: [max_recv, inter_dim] (fp8 / bf16 / fp4) ──
    if a_dtype == "fp8":
        a2_view = (
            torch.randn(max_recv, inter_dim, dtype=torch.bfloat16, device=dev)
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
            (max_recv * (inter_dim // 32),), 127,
            dtype=torch.uint8, device=dev,
        )
    elif a_dtype == "fp4":
        # FP4 占位：2 个 fp4 element / 字节
        a2_view = torch.randint(
            0, 256, (max_recv, inter_dim // 2), dtype=torch.uint8, device=dev,
        )
        a2_storage = a2_view.view(-1)
        # 1x32 group scale (e8m0: 127 = 2^0 = 1.0；用 0 会让 GEMM2 输出全 0)
        a2_scale_1d = torch.full(
            (max_recv * (inter_dim // 32),), 127, dtype=torch.uint8, device=dev,
        )
    elif a_dtype in ("bf16", "fp16"):
        torch_a = torch.bfloat16 if a_dtype == "bf16" else torch.float16
        a2_view = torch.randn(max_recv, inter_dim, dtype=torch_a, device=dev)
        a2_storage = a2_view.view(-1)
        a2_scale_1d = torch.empty((0,), dtype=torch.float32, device=dev)
    else:
        raise ValueError(f"unsupported gemm2_a_dtype={a_dtype!r}")

    # ── W2: [epr, model_dim, inter_dim] ──
    if b_dtype == "fp4":
        # 关键：FP4 GEMM2 kernel 期望 W2 已经 preshuffle 到 MFMA 友好布局,
        # 且 scale 已经 e8m0_shuffle.  直接把随机 uint8 当 W2 喂进去时，每个
        # MFMA dot product 实际读到的"逻辑 fp4 元素"是 layout 错乱后的随机
        # 4-bit 值；其中很多模式会被 hardware/runtime 当作 0 处理（subnormal
        # flush），导致整列输出坍缩成 0 — 这就是之前 verify 看到 base_out_tok
        # 全 0 的根因。
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

        # 1x32 e8m0 scale, shape: [epr, model_dim, inter_dim/32]
        # 注意：e8m0 = 127 表示 2^0 = 1.0；用 0 (= 2^-127 ≈ 0) 会让 GEMM2
        # 整列输出坍缩为 0，导致 verify 出现 0=0 假阳性 PASS。
        w2_scale_2d = torch.full(
            (epr * model_dim, inter_dim // 32), 127,
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
            (epr * model_dim, inter_dim // 32), 127,
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

    # ── Sorted buffer (aiter / torch fallback) ──
    # 简化：直接构造 sorted_token_ids = arange(0..effective_recv) 均匀分给 epr 个 expert。
    # 每个 expert 拿到约 effective_recv/epr 个 token，padding 到 tile_m。
    # effective_recv 默认 = max_recv（cudagraph 上界，baseline 路径用），
    # fused 路径必须传入 valid_recv = total_recv 让 GEMM2 只覆盖
    # 真正被 dispatch 写入了 addr_tis 的 [0, total_recv) 槽位。
    effective_recv = int(valid_recv) if valid_recv is not None else max_recv
    per_e = (effective_recv + epr - 1) // epr
    per_e_pad = ((per_e + tile_m - 1) // tile_m) * tile_m
    blocks = epr * (per_e_pad // tile_m)
    sorted_size = blocks * tile_m

    sorted_token_ids = torch.full(
        (sorted_size,), max_recv, dtype=torch.int32, device=dev,  # max_recv 作 padding 哨兵
    )
    sorted_weights = torch.zeros(sorted_size, dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.zeros(blocks, dtype=torch.int32, device=dev)
    for e in range(epr):
        e_start = e * (per_e_pad // tile_m)
        e_end = (e + 1) * (per_e_pad // tile_m)
        sorted_expert_ids[e_start:e_end] = e
        # 把 effective_recv 个 token 平均分到 epr 个 expert
        valid_n = min(per_e, effective_recv - e * per_e)
        if valid_n > 0:
            row_start = e * per_e_pad
            sorted_token_ids[row_start:row_start + valid_n] = (
                torch.arange(e * per_e, e * per_e + valid_n,
                             dtype=torch.int32, device=dev)
                .clamp(max=effective_recv - 1)
            )
            sorted_weights[row_start:row_start + valid_n] = 1.0

    # 让 GEMM2 跑满负载：所有 block 都视作有效。
    num_valid_ids = torch.tensor([sorted_size], dtype=torch.int32, device=dev)

    bias_dummy = torch.empty((0,), dtype=torch.float32, device=dev)

    # GEMM2 输出缓冲：[max_recv, model_dim] of bf16，与 combine.input dtype 对齐
    out_dtype = cfg.data_type if cfg.data_type in (torch.bfloat16, torch.float16) else torch.bfloat16
    gemm2_out = torch.zeros(max_recv, model_dim, dtype=out_dtype, device=dev)

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
        max_recv=max_recv,
        model_dim=model_dim,
        inter_dim=inter_dim,
        epr=epr,
        blocks=blocks,
        sorted_size=sorted_size,
        out_dtype=out_dtype,
    )


def _build_gemm2_callable(args, gemm2_in, out_dtype):
    """编译 mixed_moe_gemm2，返回 launch(o, x, w, sx, sw, st, eids, sw_sorted) 形式。"""
    out_s = "bf16" if out_dtype == torch.bfloat16 else (
        "f16" if out_dtype == torch.float16 else "f32"
    )

    # accumulate=True 走 atomic; accumulate=False 走 reduce 模式（fp4 时常用）
    # FP4 weight 必须用 reduce 模式 (accumulate=False)，否则 atomic_fadd 路径
    # 在某些 (a_dtype, b_dtype) 组合下会被 lowering 优化掉/silent-drop，
    # 导致 gemm2_out 全 0（典型症状：base_inp_tok nz=0 == fused_inp_tok nz=0）。
    if args.gemm2_b_dtype == "fp4":
        accumulate = False
    else:
        accumulate = args.gemm2_accumulate

    exe = compile_mixed_moe_gemm2(
        model_dim=args.hidden_dim,
        inter_dim=args.inter_dim,
        experts=args.num_experts_per_rank,
        xcd_swizzle=args.xcd_swizzle,
        topk=1,                     # EP 场景 reinterpret topk=1
        tile_m=args.tile_m2,
        tile_n=args.tile_n2,
        tile_k=args.tile_k2,
        doweight_stage2=False,
        a_dtype=args.gemm2_a_dtype,
        b_dtype=args.gemm2_b_dtype,
        out_dtype=out_s,
        accumulate=accumulate,
        persist_m=args.persist_m,
    )

    def _args(o, x, w, sx, sw, st, eids, sw_sorted):
        # tokens=max_recv (cudagraph 上界)，size_expert_ids=blocks
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
            gemm2_in["gemm2_out"],
            gemm2_in["a2_storage"],
            gemm2_in["w2_storage"],
            gemm2_in["a2_scale_1d"],
            gemm2_in["w2_scale_1d"],
            gemm2_in["sorted_token_ids"],
            gemm2_in["sorted_expert_ids"],
            gemm2_in["sorted_weights"],
        ),
    )

    def launch_gemm2():
        compiled(
            *_args(
                gemm2_in["gemm2_out"],
                gemm2_in["a2_storage"],
                gemm2_in["w2_storage"],
                gemm2_in["a2_scale_1d"],
                gemm2_in["w2_scale_1d"],
                gemm2_in["sorted_token_ids"],
                gemm2_in["sorted_expert_ids"],
                gemm2_in["sorted_weights"],
            )
        )

    return launch_gemm2


# ─── chain 抽象：baseline / fused 两条端到端链路 ─────────────────────────────
# NOTE: chain 内是否包含 dispatch 由调用方决定 (dispatch_inputs=None 表示不跑).
# 把 dispatch 放进 chain 是 mori best-practice (test_profiler_dispatch_combine.py
# 已验证), 解决 combine kernel 在 Stage 2 末尾把 total_recv 清 0、cudagraph
# 后续 replay 全部 combine 跑成 “for tok_i in range(0, 0)” 空 kernel 的问题.
# 若不跑 dispatch, 测出来的 combine GPU time 只是 prologue + barrier + launch
# 固定开销 (实测 bs=256 时仅 ~17us), 不反映真实 Stage 1 P2P + Stage 3 工作.
def _baseline_chain(disp_op, gemm2_launch, gemm2_in, combine_idx,
                    dispatch_inputs=None):
    """baseline: [dispatch ->] moe_gemm2 -> combine.

    dispatch_inputs=None  : 旧路径, 仅 launch GEMM2 + combine (combine 跑空).
    dispatch_inputs=(inp, wts, scales, idx) : 每次 chain 都重跑 dispatch,
        让 combine 内部的 total_recv / routing tables 永远 fresh.
    """
    if dispatch_inputs is not None:
        inp, wts, scales, idx = dispatch_inputs
        # disp_op.dispatch 内部已经 cache 编译好的 jit kernel, capture 时直接
        # launch; routing tables / total_recv 在 disp_op 内部 shmem buffer 上
        # 原地更新, combine_idx (= shmem_disp_out_idx 视图) 引用不变.
        disp_op.dispatch(inp, wts, scales, idx)
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
    """fused: [dispatch ->] fused_gemm2_combine [-> combine_no_stage1].

    dispatch_inputs 见 _baseline_chain 注释. 不跑 dispatch 时, fused 路径里的
    combine_no_stage1 (Stage 1 weight P2P + Stage 3 read+accum) 也会跑空,
    fused gemm2_combine 测出来的时间会被严重低估.
    """
    if fused_op is None:
        raise RuntimeError(
            "fused op not available; build kernels/mixed_moe_gemm2_combine_fused_op.py first"
        )
    if dispatch_inputs is not None:
        inp, wts, scales, idx = dispatch_inputs
        disp_op.dispatch(inp, wts, scales, idx)
    out = fused_op.run(
        a2=gemm2_in["a2_storage"],
        w2=gemm2_in["w2_storage"],
        a2_scale=gemm2_in["a2_scale_1d"],
        w2_scale=gemm2_in["w2_scale_1d"],
        sorted_token_ids=gemm2_in["sorted_token_ids"],
        sorted_expert_ids=gemm2_in["sorted_expert_ids"],
        sorted_weights=gemm2_in["sorted_weights"],
        num_valid_ids=gemm2_in["num_valid_ids"],
        dispatch_total_recv=dispatch_total_recv,
    )
    return out


# ─── profiler 工具 ───────────────────────────────────────────────────────────
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


# Kernel name 匹配规则：
#   FlyDSL combine 在 flyc.compile 后默认名为 ep_combine_intranode_0；
#   moe_gemm2 编译后名为 moe_gemm2 系列；fused 走 fused_gemm2_combine。
#   下面用子串匹配，避免不同编译版本的后缀差异。
_KERNEL_PATTERNS = {
    "gemm2":          ["moe_gemm2"],
    "combine":        ["ep_combine_intranode"],
    "fused":          ["fused_gemm2_combine", "mixed_moe_gemm2_combine"],
}


def _stats_from_trace(trace_path: str, op_tag: str,
                      rank: int, world_size: int, dev: torch.device,
                      active_iters: int, skip_first: int):
    """从 chrome trace 抽 gemm2 / combine / fused / replay 时间，
    跨卡 all_reduce 出 avg/min/max。

    返回 dict[metric_name -> {avg/min/max}]，metric_name 含：
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
    print(f"  所有 {world_size} 张卡的 avg / min / max（μs/call）")
    print(sep)
    hdr = f"  {'指标':<38}  {'avg':>8}  {'min':>8}  {'max':>8}"
    print(hdr)
    print(f"  {'-'*64}")

    rows = [
        ("[Device] moe_gemm2 kernel GPU time",        "gemm2_gpu"),
        ("[Device] combine kernel GPU time",          "combine_gpu"),
        ("[Device] fused_gemm2_combine GPU time",     "fused_gpu"),
        ("[E2E]    replay CUDA time (含sync)",        "replay_cuda_e2e"),
        ("[Host]   replay CPU  time",                 "replay_cpu_e2e"),
    ]
    for label, key in rows:
        v = stats[key]
        print(f"  {label:<38}  {v['avg']:>8.1f}  {v['min']:>8.1f}  {v['max']:>8.1f}")
    print()


# ─── profile + cudagraph 主流程 ───────────────────────────────────────────────
def _capture_chain(chain_fn, capture_stream, eager_warmup=1, disp_op=None):
    """eager warmup（触发 JIT 编译） → CUDAGraph capture.

    若 chain 内含 dispatch (方案 A), 流程严格对齐
    test_profiler_dispatch_combine.py:_cudagraph_capture_flydsl :
        op.reset()      → ms.shmem_barrier_all()
        chain_fn()      → eager warmup 1 次, 触发 jit compile
        op.barrier()    → ms.shmem_barrier_all()
        graph.capture(chain_fn)   → capture 1 次
    其他形式 (额外 sync / 多次 warmup / 中途 reset) 都会让 mori shmem 内部
    cross-device counter drift, capture 时第一次 chain_fn 就 hipErrorIllegalAddress.
    """
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
                            disp_op=None):
    """torch.profiler 采集 CUDAGraph replay 中的各 kernel 时间.

    若 chain 内含 dispatch (方案 A), 必须传 disp_op 以便 _capture_chain
    在 eager warmup 之间做 reset + cross-device barrier.
    """
    ms.shmem_barrier_all()

    capture_stream = torch.cuda.Stream()
    g = _capture_chain(chain_fn, capture_stream, eager_warmup=1, disp_op=disp_op)
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
              f"丢弃前 {skip_first}，有效 {valid_iters} 次（no-reset）...")

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
    """打印 baseline vs fused 的 GPU 端 kernel 时间加速比。"""
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


# ─── verify 模式：baseline vs fused 数值一致性 ──────────────────────────────
def _reset_combine_shmem(disp_op):
    """清空 combine 路径相关 shmem buffer，确保 baseline / fused 之间互不污染。

    需要清空：
      - shmem_comb_inp_{tok,wts}: stage1 P2P scatter 写入的 buffer。两条
        路径都覆盖性写入，这里清零是为了保险（避免上一轮残留落到新一轮
        没覆盖到的 byte）。
      - shmem_comb_out_{tok,wts}: stage3 写入的最终输出。清零方便对比，
        若 fused 漏写某些 byte，对比时能直接暴露。
    *不要* 清零 ``addr_xdb_flag`` / ``shmem_xdev_bar_mem`` —— 它们由 combine
    自己单调递增/写入，跨调用必须保持连续。
    """
    disp_op.shmem_comb_inp_tok.zero_()
    disp_op.shmem_comb_inp_wts.zero_()
    disp_op.shmem_comb_out_tok.zero_()
    disp_op.shmem_comb_out_wts.zero_()


def _snapshot_combine_out(disp_op):
    """clone 当前 combine 输出（避免下一轮覆盖）。

    返回 ``(out_tok_clone, out_wts_clone)``，dtype 与 op 暴露的视图一致
    （bf16/fp16 token + f32 weights）。
    """
    cfg = disp_op.cfg
    mt  = cfg.max_num_inp_token_per_rank
    k   = cfg.num_experts_per_token
    if disp_op._use_fp8_cast:
        fp8_bytes = mt * cfg.hidden_dim
        out_tok = (
            disp_op.shmem_comb_out_tok.view(torch.int8)[:fp8_bytes]
            .view(torch.float8_e4m3fn).view(mt, cfg.hidden_dim)
            .to(torch.bfloat16)
        )
    else:
        out_tok = (
            disp_op.shmem_comb_out_tok.view(torch.int8)[:mt * cfg.token_bytes]
            .view(cfg.data_type).view(mt, cfg.token_view_dim)
        )
    out_wts = disp_op.shmem_comb_out_wts.view(mt, k)
    return out_tok.detach().clone(), out_wts.detach().clone()


def _compare_two(name_a, ta, name_b, tb, *, rank, atol_abs=2e-2, atol_rel=5e-2,
                 max_print=8):
    """逐 token 对比两个张量，输出统计 + 部分逐元素差异。返回 pass:bool。"""
    assert ta.shape == tb.shape, (
        f"shape mismatch: {name_a}={tuple(ta.shape)} vs {name_b}={tuple(tb.shape)}"
    )
    a_f = ta.detach().float()
    b_f = tb.detach().float()
    diff   = (a_f - b_f).abs()
    a_norm = a_f.abs().clamp_min(1e-6)
    rel    = diff / a_norm
    nan_a  = torch.isnan(a_f)
    nan_b  = torch.isnan(b_f)

    abs_max  = diff.max().item() if diff.numel() > 0 else 0.0
    abs_mean = diff.mean().item() if diff.numel() > 0 else 0.0
    rel_max  = rel.max().item() if rel.numel() > 0 else 0.0
    rel_mean = rel.mean().item() if rel.numel() > 0 else 0.0
    n_nan_a  = int(nan_a.sum().item())
    n_nan_b  = int(nan_b.sum().item())

    n_diff_per_tok = (diff > atol_abs).reshape(diff.shape[0], -1).any(dim=-1).sum().item()
    fail_abs = abs_max > atol_abs
    fail_rel = rel_max > atol_rel
    pass_ok  = (not fail_abs) and (not fail_rel) and (n_nan_a == 0) and (n_nan_b == 0)

    # 所有 rank 都打印（hang 调试 / 跨卡定位）—— rank 0 同步打印过去主导，
    # 其他 rank 仅在 FAIL 时简短报告
    if rank == 0 or not pass_ok:
        status = "PASS" if pass_ok else "FAIL"
        print(f"  [rank {rank}] [{status}] {name_a} vs {name_b}: "
              f"shape={tuple(ta.shape)} dtype={ta.dtype}/{tb.dtype}")
        print(f"  [rank {rank}]         abs: max={abs_max:.4e} mean={abs_mean:.4e}  "
              f"(thr={atol_abs:.2e})")
        print(f"  [rank {rank}]         rel: max={rel_max:.4e} mean={rel_mean:.4e}  "
              f"(thr={atol_rel:.2e})")
        print(f"  [rank {rank}]         nan: a={n_nan_a} b={n_nan_b}  "
              f"tokens_with_diff={n_diff_per_tok}")
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
    """baseline vs fused 数值一致性校验。

    流程
    ----
    1. 跑一次 baseline (moe_gemm2 + combine)，clone 输出。
    2. 清零 ``shmem_comb_inp_*`` / ``shmem_comb_out_*`` + 全局 barrier，
       确保 fused 从干净状态开始且不被 baseline 残留污染。
    3. 跑一次 fused (fused_gemm2 + combine_no_stage1)，clone 输出。
    4. 比较 ``out_tok`` 与 ``out_wts``。

    精度阈值
    --------
    GEMM compute 与 P2P scatter 的浮点路径完全一致；不同点只是
    "本地 store + 远程 vec4 拷贝" vs "直接远程 store"，理论上 frag 数据
    bit-exact 相同。这里给出宽松上限以容忍未来累加顺序的微小差异。
    """
    if rank == 0:
        sep = "=" * 78
        print(f"\n{sep}\n  VERIFY  baseline vs fused  EP={world_size}\n{sep}")

    # ── Step 1: baseline ──────────────────────────────────────────────────────
    if rank == 0:
        print("[verify] step 1: running baseline (moe_gemm2 → combine)")
    # 先 reset 一次确保从干净状态开始
    _reset_combine_shmem(disp_op)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    # combine kernel 在 Stage 2 末尾会把 total_recv 写 0；如果 baseline 跑完
    # 之后再跑 fused（其内部 combine_no_stage1 仍用 total_recv 决定 Stage 1
    # weight 循环），不恢复就会变成 0 次迭代 → fused_out_wts 全 0。snapshot
    # 一下 baseline 之前的真值，跑完再 restore。
    _saved_total_recv = disp_op.total_recv.detach().clone()
    # 在 baseline_chain 内部 GEMM2 launch 完成后立即 snapshot gemm2_out,
    # 用于排查 base_out_tok=0 的来源 (是 GEMM2 出 0 还是 stage1/3 的 P2P 路径出 0).
    gemm2_launch()
    torch.cuda.synchronize()
    # 调试: FLYDSL_VERIFY_FORCE_GEMM2_OUT_PATTERN=1 时, 把 gemm2_out 强制覆盖为
    # 已知非零 pattern (rank+1 标量), 用于隔离 GEMM2 与 combine 的责任. 如果
    # base_out_tok 仍然全 0, 说明问题在 combine 的 stage 1/3 (与 GEMM2 无关).
    if os.environ.get("FLYDSL_VERIFY_FORCE_GEMM2_OUT_PATTERN", "0") == "1":
        gemm2_in["gemm2_out"].fill_(float(rank) + 1.0)
        torch.cuda.synchronize()
    _gemm2_out_snap = gemm2_in["gemm2_out"].detach().clone()
    if rank == 0:
        print(f"  [rank {rank}] post-GEMM2 gemm2_out: "
              f"shape={tuple(_gemm2_out_snap.shape)} "
              f"min={_gemm2_out_snap.float().min().item():.4e} "
              f"max={_gemm2_out_snap.float().max().item():.4e} "
              f"abs_mean={_gemm2_out_snap.float().abs().mean().item():.4e} "
              f"nz={int((_gemm2_out_snap != 0).sum().item())}", flush=True)
    base_tok, base_wts = disp_op.combine(
        gemm2_in["gemm2_out"], None, combine_idx,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    base_tok_s, base_wts_s = _snapshot_combine_out(disp_op)
    # 立刻 snapshot inp_wts (baseline stage 1 写入；stage 3 读但不清)
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

    # ── Step 2: 重置 combine 相关 shmem ────────────────────────────────────────
    if rank == 0:
        print("[verify] step 2: zeroing shmem_comb_{inp,out}_* before fused run")
    _reset_combine_shmem(disp_op)
    # 恢复 total_recv（baseline combine 把它写成 0 了）
    disp_op.total_recv.copy_(_saved_total_recv)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # ── Step 3: fused ─────────────────────────────────────────────────────────
    if rank == 0:
        print("[verify] step 3: running fused (fused_gemm2 → combine_no_stage1)")

    # 调试: FLYDSL_VERIFY_DUMP_INP_TOK=1 时, 对比 fused_gemm2 P2P scatter 结束
    # 后的 shmem_comb_inp_tok vs baseline stage1 写入, 用来定位 GEMM2 epilogue
    # 内的 token P2P 是否落盘完整 (绕过 combine 的 stage2/3 累加).
    # 注意: 由于 fused_gemm2 只 scatter token (weights 由 combine_no_stage1 处理),
    # 这里不再对比 inp_wts.
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
            # 调试: 打印 addr_tis 内容(每 t -> 期望写入的 (dest_pe, dest_lid)).
            # baseline 与 fused 都是用同一个 addr_tis 解码; 如果 fused 写到的
            # (dest_pe, dest_lid) 与解码出的不一致, 说明 fused kernel 内的
            # 解码 / 索引有问题. 这里只打前 8 个 valid t 做 sanity check.
            tis_view = disp_op.shmem_tok_id_to_src.detach().clone().cpu()
            log2_mt = int(mt_t).bit_length() - 1
            mask_mt = mt_t - 1
            tot_recv = int(_saved_total_recv.item())
            # 找 PE 0 自己 intra-rank 路由的 t (decoded dest_pe == rank).
            # 这些 t 对应 PE 0 写到 PE 0 自己 b_tok[rank, dest_lid] 的槽位.
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
                # 看 PE 0 本地的 b_tok[self_pe=rank, dlid] 是否有数据
                b_nz = int((b_tok[self_pe, dlid] != 0).sum().item())
                f_nz = int((f_tok[self_pe, dlid] != 0).sum().item())
                print(f"    t={t} enc={enc} dest_lid={dlid}  "
                      f"b_tok[{self_pe},{dlid}] nz={b_nz}  "
                      f"f_tok[{self_pe},{dlid}] nz={f_nz}",
                      flush=True)
            # 也看 fused 实际写到哪些 lid (intra-rank 全部 lid 扫一遍)
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
        # 重置后跑完整的 fused (stage1+stage2+stage3) 用于后续 out 对比
        _reset_combine_shmem(disp_op)
        disp_op.total_recv.copy_(_saved_total_recv)
        torch.cuda.synchronize(); ms.shmem_barrier_all()

    _ = _fused_chain(
        disp_op, fused_op, gemm2_in, combine_idx, dispatch_total_recv,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    fused_tok_s, fused_wts_s = _snapshot_combine_out(disp_op)

    # ── Step 4: 对比 ──────────────────────────────────────────────────────────
    if rank == 0:
        print("[verify] step 4: comparing baseline vs fused outputs")
    # 打印每个 rank 的 raw value，避免 0=0 假 PASS
    with torch.no_grad():
        print(f"  [rank {rank}] base_out_tok stats: "
              f"min={base_tok_s.float().min().item():.4e} "
              f"max={base_tok_s.float().max().item():.4e} "
              f"abs_mean={base_tok_s.float().abs().mean().item():.4e} "
              f"nz={int((base_tok_s != 0).sum().item())}", flush=True)
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
    # combine 输出 shape = [cur_rank_num_token, hidden_dim]，"valid" 区间是
    # 本 PE 原始持有的 token 数（dispatch 之前的输入行数），即调用 combine
    # 时传入的 cur_tok（默认 = cfg.max_num_inp_token_per_rank）。
    #
    # 注意：不要用 dispatch_total_recv —— 它是本 PE *接收* 到的 token 数，
    # 而且 combine 内部 stage 2 末尾会把 total_recv 写 0（见 kernel 中
    # `buffer_store(0, _r_trecv, 0)`），所以在 baseline 跑完后再读它会拿到 0。
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

    # 跨卡聚合 pass/fail
    local = torch.tensor([1 if pass_tok else 0, 1 if pass_wts else 0],
                         dtype=torch.int32, device=dev)
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    all_pass_tok = bool(local[0].item())
    all_pass_wts = bool(local[1].item())

    if rank == 0:
        print(f"\n  RESULT (all-reduce min): "
              f"out_tok={'PASS' if all_pass_tok else 'FAIL'}, "
              f"out_wts={'PASS' if all_pass_wts else 'FAIL'}")
        print("=" * 78)


# ─── 各模式入口（仅 profile+cudagraph 实现）─────────────────────────────────
def _not_impl(name: str):
    raise NotImplementedError(
        f"mode '{name}' not yet implemented in this acceptance script. "
        "Only `--mode profile --cudagraph` is wired in this skeleton; "
        "extend after the fused kernel is built."
    )


def run_acceptance(rank, world_size, args):
    dev = torch.device("cuda", rank)
    cur_tok = args.max_tokens
    k       = args.k

    # ── cfg / dispatch op ───────────────────────────────────────────────────
    _dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    cfg = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world_size,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=cur_tok,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=k,
        data_type=_dtype,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        chip=args.chip,
        use_external_inp_buf=args.use_external_inp_buf,
        enable_std_moe=args.enable_std_moe,
        scale_dim=args.scale_dim,
        scale_type_size=args.scale_type_size,
        quant_type=args.quant_type,
    )
    args.max_num_inp_token_per_rank = cur_tok  # 给 _build_gemm2_static_inputs 使用

    # ── FP4 hard constraints ────────────────────────────────────────────────
    # FlyDSL FP4 mfma_scale_x128 path 隐式假设：
    #   * tile_k >= 256 （scale layout 一格覆盖 128 fp4 elements，<256 直接全 0）
    #   * inter_dim/model_dim >= 256 且能整除 256
    #   * accumulate=False（reduce 模式；atomic_fadd 在 mixed fp8/fp4 下被
    #     lowering 优化掉，gemm2_out 全 0）
    # 不满足这些约束时不抛错，只是 GEMM2 sile-output 0；为了不再让下次踩坑，
    # 在 verify/bench 入口直接报错。
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
        print(f"\n{'='*78}")
        print(f"[acceptance] EP={world_size}, bs={cur_tok}, "
              f"h={cfg.hidden_dim}, inter={args.inter_dim}, k={k}, "
              f"epr={cfg.num_experts_per_rank}")
        print(f"  GEMM2: a={args.gemm2_a_dtype}/b={args.gemm2_b_dtype}, "
              f"tile_m2={args.tile_m2}, tile_n2={args.tile_n2}, "
              f"tile_k2={args.tile_k2}, persist_m={args.persist_m}")
        print(f"  bench-op={args.bench_op}, fuse-mode={args.fuse_mode}, "
              f"fused_op_available={HAS_FUSED_OP}")
        if not HAS_FUSED_OP:
            print(f"  [warn] fused op import failed: {_FUSED_IMPORT_ERR}")
        print(f"{'='*78}")

    disp_op = FlyDSLDispatchCombineIntraNodeOp(cfg)

    fused_op = None
    if HAS_FUSED_OP and args.bench_op in ("fused", "both"):
        fused_op = FlyDSLMoeGemm2CombineOp(
            cfg=cfg,
            disp_op=disp_op,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m2, tile_n=args.tile_n2, tile_k=args.tile_k2,
            persist_m=args.persist_m,
            a_dtype=args.gemm2_a_dtype, b_dtype=args.gemm2_b_dtype,
            force_mode=args.fuse_mode,
            xcd_swizzle=args.xcd_swizzle,
        )

    ms.shmem_barrier_all()

    # ── 输入 ────────────────────────────────────────────────────────────────
    inp, wts, idx = _build_dispatch_inputs(rank, world_size, dev, args, cfg)

    # scales / packed_recv_x（与已有 dispatch op 接口对齐，本骨架默认不用）
    scales = None
    if cfg.scale_dim > 0 and cfg.scale_type_size > 0:
        _sc_bytes = cfg.scale_dim * cfg.scale_type_size
        scales = torch.randn(cur_tok, _sc_bytes // 4,
                             dtype=torch.float32, device=dev).contiguous()
        scales = scales.view(torch.uint8).view(cur_tok, _sc_bytes)

    # ── 一次性 dispatch（拿到 total_recv，给 GEMM2 输入构造用）──────────────
    # combine 依赖 disp_op 内部的 shmem_disp_out_* 表, 由 dispatch 写入;
    # 跑一次后这些表保持不变, chain 内 combine 直接读取静态视图.
    if rank == 0:
        print(f"[setup] running one-shot dispatch to populate routing tables…")
    disp_ret = disp_op.dispatch(inp, wts, scales, idx)
    combine_idx = disp_ret[3]                # shmem_disp_out_idx 视图
    dispatch_total_recv = disp_ret[4]        # shmem total_recv 标量
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # 对 fused 路径来说，GEMM2 epilogue 会读 addr_tis[t] 来解码
    # (dest_pe, dest_lid)，但 dispatch 只写了 [0, total_recv) 槽位。
    # 所以 sorted_token_ids 的 t 必须落在 [0, total_recv)；否则会 P2P
    # scatter 到错误位置（写到 PE 0 / slot 0），污染 combine stage 3 输入。
    # 注意: 必须在 hard-reset *之前* 读 .item(), 否则 total_recv 会被清 0,
    # sorted_token_ids 全是 padding 哨兵 → fused gemm2 launch 立即 fail.
    valid_recv_for_gemm2 = None
    if args.bench_op in ("fused", "both") and HAS_FUSED_OP:
        valid_recv_for_gemm2 = int(dispatch_total_recv.item())
        if rank == 0:
            print(f"[setup] dispatch total_recv (rank0) = {valid_recv_for_gemm2}; "
                  "will constrain sorted_token_ids to [0, total_recv) for fused path")

    # ── 方案 A 的 hard-reset: 只清 setup dispatch 留下的 *local* counter
    # (dest_pe_ctr / disp_bar / comb_bar / total_recv / disp_grid_bar),
    # **不要** 清 cross-device shmem buffer (shmem_xdev_bar_mem 用 monotonic
    # cur_flag 模式 mori 自己管理; shmem_comb_inp_* 下次 chain 自然覆盖).
    # 任意 zero shmem_* 会让 mori shmem 内部 cur_flag 跟实际 buffer 值脱节,
    # 表现为 capture 阶段第 1 次 fused gemm2 launch 时 hipErrorInvalidValue.
    if args.chain_include_dispatch:
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

    gemm2_in = _build_gemm2_static_inputs(
        rank, world_size, dev, args, cfg, valid_recv=valid_recv_for_gemm2,
    )
    gemm2_launch = _build_gemm2_callable(args, gemm2_in, gemm2_in["out_dtype"])

    meta = dict(
        world_size=world_size,
        max_tokens=cur_tok,
        hidden_dim=cfg.hidden_dim,
        inter_dim=args.inter_dim,
        k=k,
        num_experts_per_rank=args.num_experts_per_rank,
        warmup=args.warmup, iters=args.iters,
        block_num=cfg.block_num,
        warp_per_block=cfg.warp_num_per_block,
        gemm2_a_dtype=args.gemm2_a_dtype,
        gemm2_b_dtype=args.gemm2_b_dtype,
        tile_m2=args.tile_m2, tile_n2=args.tile_n2, tile_k2=args.tile_k2,
        persist_m=args.persist_m,
        bench_op=args.bench_op, fuse_mode=args.fuse_mode,
    )

    out_dir = os.path.join(args.output_dir, f"ep{world_size}_bs{cur_tok}")
    os.makedirs(out_dir, exist_ok=True)

    # 路径开关（verify / profile 两条入口都要看）
    test_baseline = args.bench_op in ("baseline", "both")
    test_fused    = args.bench_op in ("fused",    "both") and fused_op is not None

    # ── 模式分发 ────────────────────────────────────────────────────────────
    if args.mode == "verify":
        # verify 只对 --bench-op both 有意义（baseline 与 fused 互比对）
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
        _not_impl("bench" + ("+cudagraph" if args.cudagraph else "+eager"))
    if args.mode == "profile" and not args.cudagraph:
        _not_impl("profile+eager")

    # 唯一实现的路径：profile + cudagraph
    assert args.mode == "profile" and args.cudagraph, \
        f"unsupported (mode={args.mode}, cudagraph={args.cudagraph})"

    base_stats = None
    fused_stats = None

    # 默认 chain 内重跑 dispatch (方案 A): mori best-practice, 每次 replay 都
    # 让 dispatch 重写 routing tables / total_recv, 避免 combine 跑空.
    # 旧行为 (combine 跑空) 通过 --no-chain-include-dispatch 复现.
    chain_disp_inputs = (inp, wts, scales, idx) if args.chain_include_dispatch else None
    if rank == 0 and args.chain_include_dispatch:
        print(f"[chain] including dispatch in cudagraph (mori best-practice; "
              f"avoids total_recv being zeroed by combine)")

    # 仅当 chain 内含 dispatch (方案 A) 时才传 disp_op 给 capture, 让它做 reset.
    _capture_disp_op = disp_op if args.chain_include_dispatch else None

    if test_baseline:
        def _baseline():
            return _baseline_chain(disp_op, gemm2_launch, gemm2_in, combine_idx,
                                   dispatch_inputs=chain_disp_inputs)

        base_stats = profile_cudagraph_chain(
            _baseline, "baseline",
            rank, world_size, dev,
            iters=args.iters, out_dir=out_dir, meta=meta,
            disp_op=_capture_disp_op,
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

    if rank == 0:
        print(f"\n[acceptance] 全部 trace/JSON 保存至: {out_dir}/")


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
        description="moe_gemm2 + combine 端到端验收脚本（dispatch 在 setup 一次性完成）"
    )
    # 形状 / 路由
    p.add_argument("--world-size",           type=int, default=8)
    p.add_argument("--max-tokens",           type=int, default=512,
                   help="单卡 dispatch 输入 token 数；max_recv = world_size * max_tokens")
    p.add_argument("--hidden-dim",           type=int, default=7168,
                   help="GEMM2 输出维度（== combine token 维度 == dispatch token 维度）")
    p.add_argument("--inter-dim",            type=int, default=4096,
                   help="GEMM2 输入维度（GEMM1 输出维度，本骨架不跑 GEMM1）")
    p.add_argument("--num-experts-per-rank", type=int, default=32)
    p.add_argument("--k",                    type=int, default=8)
    # dispatch / combine
    # NOTE: empirically bn=256 is ~30% faster than bn=80 on combine
    # (27→19us @ bs=256/h=7168/k=8/EP=8) — the small grid was leaving
    # CUs idle.  wpb=4 vs 16 shows no meaningful difference (the XDB
    # barrier is wave-cooperative; more waves just adds protocol cost).
    # Updated default to bn=256/wpb=4 to reflect the real optimum.
    p.add_argument("--block-num",            type=int, default=256)
    p.add_argument("--warp-per-block",       type=int, default=4)
    p.add_argument("--chip",                 type=str, default="gfx950")
    p.add_argument("--dtype",                type=str, default="bf16",
                   choices=list(DTYPE_MAP.keys()),
                   help="dispatch / combine token dtype")
    # GEMM2
    p.add_argument("--gemm2-a-dtype",        type=str, default="fp8",
                   choices=["fp8", "fp4", "fp16", "int8"])
    p.add_argument("--gemm2-b-dtype",        type=str, default="fp4",
                   choices=["fp8", "fp4", "fp16", "int8", "int4"])
    p.add_argument("--tile-m2",              type=int, default=32)
    p.add_argument("--tile-n2",              type=int, default=128)
    p.add_argument("--tile-k2",              type=int, default=256,
                   help="GEMM2 tile_k；FP4 路径必须 >=256")
    p.add_argument("--persist-m",            type=int, default=4,
                   help="moe_gemm2 沿 M 持久化 block 数")
    p.add_argument("--xcd-swizzle",          type=int, default=0,
                   help="moe_gemm2 跨 XCD swizzle 因子（=0 关闭；MI300 推荐 8）")
    p.add_argument("--gemm2-accumulate",     action="store_true", default=True,
                   help="GEMM2 epilogue 走 atomic-add（默认 True；fp4 reduce 模式时关掉）")
    # 模式
    p.add_argument("--mode", choices=["profile", "bench", "verify"], default="profile",
                   help="本骨架仅 profile 模式有效")
    p.add_argument("--cudagraph", action="store_true", default=True,
                   help="本骨架强制 CUDAGraph；--no-cudagraph 仅占位")
    p.add_argument("--no-cudagraph", dest="cudagraph", action="store_false")
    # ── 方案 A：把 dispatch 也 capture 进 chain (mori best-practice)。
    # 默认开启；旧错误行为 (combine 跑空) 用 --no-chain-include-dispatch 复现.
    p.add_argument("--chain-include-dispatch", action="store_true", default=True,
                   help="把 dispatch 放进 cudagraph chain 内，每次 replay 都重写 "
                        "routing tables / total_recv，避免 combine kernel 内部因 "
                        "total_recv 被清 0 而后续 replay 全部跑空。")
    p.add_argument("--no-chain-include-dispatch", dest="chain_include_dispatch",
                   action="store_false",
                   help="退回旧路径：dispatch 只在 setup 阶段跑一次。combine 仅在第 "
                        "1 次 chain 真实工作，后续全部跑空 — 仅用于排查 / 对照。")
    p.add_argument("--bench-op", choices=["baseline", "fused", "both"], default="baseline",
                   help="跑哪条链路")
    p.add_argument("--fuse-mode", choices=["auto", "full", "fallback"], default="auto",
                   help="fused op 内部模式选择（auto=按 occupancy 自适应）")
    # profile / 输出
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=20,
                   help="profiler active 轮次（最终有效 = iters - skip(5)）")
    p.add_argument("--output-dir", type=str, default="acceptance_profile")
    p.add_argument("--port",       type=int, default=29800)
    # 功能开关（与 dispatch op 对齐）
    p.add_argument("--no-external-inp-buf", dest="use_external_inp_buf",
                   action="store_false", default=True)
    p.add_argument("--enable-std-moe", action="store_true", default=False)
    p.add_argument("--scale-dim",       type=int, default=0)
    p.add_argument("--scale-type-size", type=int, default=0)
    p.add_argument("--quant-type", type=str, default="none",
                   choices=["none", "fp8_direct_cast"])
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
            print(f"[warn] 可用 GPU={torch.cuda.device_count()}, "
                  f"world_size 调整: {args.world_size} → {ws}")
        torch.multiprocessing.spawn(
            _worker, args=(ws, args, args.port),
            nprocs=ws, join=True,
        )


if __name__ == "__main__":
    main()
