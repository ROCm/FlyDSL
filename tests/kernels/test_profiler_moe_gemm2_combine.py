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

# aiter moe_sorting：低层 in-place 接口，cudagraph-friendly
# (高层 fused_moe.moe_sorting 每次会重新 alloc 5 个 tensor，无法 capture)。
#
# Backend 优先级 (env `FLYDSL_AITER_SORTING_BACKEND` 覆盖；默认 opus)：
#   * opus  → aiter.moe_sorting_opus_fwd   (raw CUDA, self-contained,
#              不依赖 composable_kernel submodule)
#   * cktile→ aiter.moe_sorting_fwd         (ck_tile, 依赖
#              3rdparty/composable_kernel/example/ck_tile/13_moe_sorting/)
# 两者完全相同接口签名，sorted_token_ids 都按 (j_global<<24)|t 编码。
_AITER_SORTING_BACKEND = os.environ.get("FLYDSL_AITER_SORTING_BACKEND", "opus")
try:
    if _AITER_SORTING_BACKEND == "cktile":
        from aiter import moe_sorting_fwd as _aiter_moe_sorting_fwd  # type: ignore
    else:
        from aiter import moe_sorting_opus_fwd as _aiter_moe_sorting_fwd  # type: ignore
    HAS_AITER_SORTING = True
    _AITER_SORTING_ERR = ""
except Exception as _e:  # noqa: BLE001
    _aiter_moe_sorting_fwd = None  # type: ignore[assignment]
    HAS_AITER_SORTING = False
    _AITER_SORTING_ERR = repr(_e)


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
    routing = getattr(args, "routing", "random")
    if routing == "atomic1_8pe":
        # 确定性均匀路由（生产 balanced EP 等价；atomic_per_pe=1, num_pe=k=8）：
        #   idx[t, j] = dest_pe * epr + local_eid
        #   dest_pe   = (rank + t + j) % world_size          ← 每 token 的 k 个
        #                                                       expert 散到 k 个不同
        #                                                       PE，且不同 rank/token
        #                                                       的起点交错，避免热点
        #   local_eid = ((rank*cur_tok + t)*k + j) % epr     ← rank、t、j 三维循环
        #                                                       填遍所有 (PE, local_e)
        # 单卡 cur_tok*k 个 (t,j) 对均匀洒满 [0, ws*epr) → 每 expert 命中次数 ≈
        #   cur_tok*k / (ws*epr)。在 cur_tok=32/ws=8/epr=32/k=8 默认值下 = 1.0，
        #   完美均衡。同 token 的 k 个 expert 必定跨 k 个不同 PE → dispatch dedup
        #   不会触发 (dup_ballot == 0 always)，每条 (T, j_global) 都进 sorted_token_ids。
        # 命名: atomic1_8pe = 每张 dest_pe 上同一 (t, *) output row 仅 1 次
        #       atomic_fadd（atomic_per_pe=1），token 散到 8 个不同 PE。
        for t in range(cur_tok):
            for j in range(k):
                dest_pe   = (rank + t + j) % world_size
                local_eid = ((rank * cur_tok + t) * k + j) % epr
                idx[t, j] = dest_pe * epr + local_eid
        return inp, wts, idx

    if routing == "atomic8_1pe":
        # 场景 2：单 token 的 k 个 expert 全压到同 1 个 dest_pe，构造
        # GEMM2 atomic-contention worst case (atomic k=8 同 (t, *) row)，
        # 同时保证卡间 / 卡内均衡：
        #   - 全局 token 顺序 g = rank * cur_tok + t  ∈ [0, world_size * cur_tok)
        #   - dest_pe = g % world_size           ← 卡间均衡 (每 PE 收 cur_tok 个 token)
        #   - eid_base = (g // world_size) % epr ← 按 g 排序后该 token 在
        #     *本 dp* 上的相对序号 (0..cur_tok*npes-1) mod epr，让 8 个 rank
        #     各 cur_tok/npes 个 token 共同覆盖 dp 内全部 epr 个 local_eid，
        #     **真正卡内均衡** (每 local_eid 命中 cur_tok*npes*k/(npes*epr)
        #     = cur_tok*k/epr 次)。
        # local_eid[j] = (eid_base + j) % epr → 每 token 连续 k 个 expert。
        # 命名: atomic8_1pe = token 的 k=8 个 expert 全落同 1 个 dest_pe，
        #       该 PE 上同一 (t, *) output row 被 atomic_per_pe=k=8 次
        #       atomic_fadd 命中（atomic-k worst case）。
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
        # 场景 3：每 token 的 k 个 expert 分布到 *4 个不同* PE 上，每 PE 上
        # 落 atomic_per_pe = k/4 个命中 (k=8 → atomic_per_pe=2)，即 GEMM2 同
        # 一 output row 上仅 2 次 atomic_fadd 竞争 — 介于 atomic1_8pe (atomic=1)
        # 和 atomic8_1pe (atomic=k=8 worst case) 之间的中等 contention 场景。
        #
        # 卡间 / 卡内均衡构造：
        #   - 全局 token 顺序 g = rank * cur_tok + t  ∈ [0, world_size * cur_tok)
        #   - 选 PE: base_pe = g % world_size；
        #     dest_pe[j] = (base_pe + j_group) % world_size, j_group = j // atomic_per_pe
        #     ∈ [0, 4)。同一 token 的 4 个 PE = {base, base+1, base+2, base+3} (mod ws),
        #     必然 4 个不同 PE（要求 world_size >= 4）。base_pe 在 g 上 round-robin
        #     取遍 [0, ws)，每张 PE 接收的 (rank,t,j_group) 三元组数 = num_pe * cur_tok
        #     (= 4*cur_tok per rank)，atomic 总数 = atomic_per_pe * num_pe * cur_tok =
        #     k * cur_tok per PE per rank, 全局 ws * cur_tok * k / ws = cur_tok * k →
        #     **卡间完美均衡**。
        #   - 选 local_eid: local_eid = (g // world_size + j) % epr
        #     关键: 用 g // ws (而非 g % ws) 当 eid_base，避开 dest_pe 的 g%ws lattice
        #     和 local_eid 的"塌缩" — PE 限制下 g 在 r 维步长为 ws，若 local_eid 公式
        #     里 r 的系数与 epr 不互质 (例如系数=ws=8, epr=32, gcd=8) 会让 32 个 r
        #     mod 后只覆盖 epr/gcd = 4 个 local_eid。改用 g//ws 取 [0, cur_tok)，配合
        #     j 的 [0, k) offset，在每个 PE 上把 cur_tok * k = 256 个 atomic 均匀洒到
        #     epr=32 个 local_eid → 每 (PE, local_eid) cell 命中 cur_tok*k/epr = 8 次
        #     (默认 cur_tok=32, k=8, epr=32 下完美均衡)。
        #
        # 约束: k % 4 == 0 (默认 8/4=2 ✓); world_size >= 4 (默认 8 ✓)。
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
        atomic_per_pe = k // num_pe  # k=8 → 2
        for t in range(cur_tok):
            g        = rank * cur_tok + t
            base_pe  = g %  world_size
            eid_base = g // world_size
            for j in range(k):
                j_group   = j // atomic_per_pe              # 0..num_pe-1 (4 个 PE 槽)
                dest_pe   = (base_pe + j_group) % world_size
                local_eid = (eid_base + j) % epr
                idx[t, j] = dest_pe * epr + local_eid
        return inp, wts, idx

    # routing == "random" (legacy 行为)
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


def _build_gemm2_static_inputs(rank, world_size, dev, args, cfg, *, disp_op):
    """为 GEMM2 / combine 构造 *静态* 输入（capture 期间不变）。

    Layout（生产对齐, topk=k）：
      - kernel 看到的 `tokens_in` = max_recv = world_size * bs (去重后上界)。
      - A2 (input) capacity = [max_recv * topk, inter_dim] —— 每个 (t, s) 占独立一行，
        kernel 内 A2 行寻址公式为 `t * topk + s`。
      - a2_scale (e8m0): [max_recv * topk, inter_dim/32]，覆盖 worst-case 大小。
      - W2 (weights): [num_experts_per_rank, model_dim, inter_dim]，per-expert FP4。
      - sorted_token_ids 通过 ``aiter.moe_sorting_fwd`` 跟随 dispatch 真实路由
        生成（capture 进 chain，每次 replay 重跑），按生产约定 (s<<24)|t 编码
        entry，s = j_global ∈ [0, k)。
      - accumulate=True 时 output buffer = [max_recv, model_dim]（同一 t 的多个 s 走
        atomic_fadd 累加到同一行）；accumulate=False 时 output 扩到 [max_recv*topk, ...]。

    Parameters
    ----------
    disp_op
        必传。setup 阶段第一次 sorting 需要从
        ``disp_op.shmem_disp_out_idx / shmem_disp_out_wts`` 拿真实路由表
        (须在 build 之前先跑过一次 dispatch)。
    """
    epr        = args.num_experts_per_rank
    max_recv   = world_size * args.max_num_inp_token_per_rank
    model_dim  = args.hidden_dim   # GEMM2 输出维度 = combine 的 hidden_dim
    inter_dim  = args.inter_dim
    tile_m     = args.tile_m2
    a_dtype    = args.gemm2_a_dtype
    b_dtype    = args.gemm2_b_dtype
    gemm2_topk = max(1, int(args.k))   # GEMM2 compile-time topk
    # A2 容量：worst-case 每个 recv token 都被 topk 个本地 expert 命中
    a2_rows    = max_recv * gemm2_topk

    torch.manual_seed(123 + rank)

    # e8m0 micro-scale = 127 表示 2^0=1.0；headroom>0 时降到 127-h 即 2^-h，
    # 同时缩 a2 和 w2 → GEMM2 输出乘 2^-2h，避免 random fp4×fp4 over-flow 到
    # fp8 e4m3 max=±448 引发 NaN（见 --gemm2-scale-headroom 帮助）。headroom=0
    # 保持历史行为。clamp 到 [0, 127] 防止 e8m0 编码变成 0 (=2^-127 ≈ 0) 整列坍缩。
    _e8m0_headroom = max(0, min(127, int(getattr(args, "gemm2_scale_headroom", 0))))
    _e8m0_val = 127 - _e8m0_headroom

    # ── A2: [max_recv * topk, inter_dim] (fp8 / bf16 / fp4) ──
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
            (a2_rows * (inter_dim // 32),), _e8m0_val,
            dtype=torch.uint8, device=dev,
        )
    elif a_dtype == "fp4":
        # FP4 占位：2 个 fp4 element / 字节
        a2_view = torch.randint(
            0, 256, (a2_rows, inter_dim // 2), dtype=torch.uint8, device=dev,
        )
        a2_storage = a2_view.view(-1)
        # 1x32 group scale (e8m0: 127 = 2^0 = 1.0；用 0 会让 GEMM2 输出全 0)
        a2_scale_1d = torch.full(
            (a2_rows * (inter_dim // 32),), _e8m0_val, dtype=torch.uint8, device=dev,
        )
    elif a_dtype in ("bf16", "fp16"):
        torch_a = torch.bfloat16 if a_dtype == "bf16" else torch.float16
        a2_view = torch.randn(a2_rows, inter_dim, dtype=torch_a, device=dev)
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

    # ── aiter moe_sorting_fwd 真实路由 sorting ────────────────────────
    # 输入: disp_op.shmem_disp_out_idx [mr, k] i32 (全 global expert id)
    #       disp_op.shmem_disp_out_wts [mr, k] f32
    # 输出: sorted_token_ids[max_padded] i32 = (j_global<<24)|t
    #       sorted_expert_ids[max_blocks] i32 = local_expert_id_per_block
    #       sorted_weights[max_padded] f32
    #       num_valid_ids[2] i32 = [padding-after-total, num_input_tokens]
    if not HAS_AITER_SORTING:
        raise RuntimeError(
            f"aiter.moe_sorting_fwd is required (real routing only); "
            f"import failed: {_AITER_SORTING_ERR}"
        )
    if disp_op is None:
        raise RuntimeError(
            "aiter sorting requires disp_op (must dispatch once before build)"
        )
    num_experts_global = world_size * epr
    # aiter 公式上界 (op_tests/test_moe_sorting.py::moe_sorting_native):
    #   max_padded = topk_ids.numel() + num_experts * block_size - topk
    # bs=32,k=8,ws=8,epr=32,tile_m=32: 256*8 + 256*32 - 8 = 10232
    max_padded = max_recv * gemm2_topk + num_experts_global * tile_m - gemm2_topk
    max_blocks = (max_padded + tile_m - 1) // tile_m

    sorted_token_ids  = torch.empty((max_padded,), dtype=torch.int32, device=dev)
    sorted_weights    = torch.empty((max_padded,), dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    # aiter 输出 [2]: [0]=padded-total, [1]=num_input；GEMM2 只读 [0]
    num_valid_ids     = torch.empty((2,), dtype=torch.int32, device=dev)

    # local_expert_mask: 本卡 [rank*epr, (rank+1)*epr) 段为 1，其他为 0
    expert_mask = torch.zeros(num_experts_global, dtype=torch.int32, device=dev)
    expert_mask[rank * epr:(rank + 1) * epr] = 1
    # moe_buf API 占位 (内容 unused，仅 alloc 形状对齐)
    moe_buf = torch.empty((max_recv, model_dim), dtype=torch.bfloat16, device=dev)

    # 用 dispatch 真实输出跑首次 sorting，把 sorted_* buffer fill 满 ──
    # 这一次跑出来的 sorted 可能在 cudagraph capture 时被覆盖；chain
    # 内 _run_aiter_sorting 每次 replay 会重填这些 buffer (in-place)。
    _aiter_moe_sorting_fwd(  # type: ignore[misc]
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
    effective_valid = -1  # 由 num_valid_ids[0] 在设备端动态决定
    aiter_state = dict(
        expert_mask=expert_mask,
        moe_buf=moe_buf,
        num_experts_global=num_experts_global,
        tile_m=int(tile_m),
        topk=int(gemm2_topk),
        mr=int(max_recv),
    )

    bias_dummy = torch.empty((0,), dtype=torch.float32, device=dev)

    # GEMM2 输出缓冲：
    #   accumulate=True  → [max_recv, model_dim]      (atomic_fadd 累加同 t 的多个 s)
    #   accumulate=False → [max_recv * topk, model_dim] (每个 (t,s) 独立 slot)
    out_dtype = cfg.data_type if cfg.data_type in (torch.bfloat16, torch.float16) else torch.bfloat16
    out_rows = max_recv if args.gemm2_accumulate else a2_rows
    if cfg.zero_copy and args.bench_op == "baseline":
        # zero-copy 模式：combine 端跳过 Stage 1, 直接读 shmem_comb_inp_tok。
        # 调用者必须把 GEMM2 输出写到 op.get_registered_combine_input_buffer
        # 返回的 view 上, 这样下游 combine 才能拿到正确数据。
        # fused 路径有自己的 P2P scatter (走 _fx_p2p_comb_inp), 不走这条 hook,
        # 所以仅 baseline + zero_copy 的组合需要 swap buffer。
        # NOTE: 当 accumulate=True 且 out_rows == max_recv 时直接 view 即可,
        # 否则形状不匹配 (registered buffer 是 [max_recv, model_dim]).
        if out_rows != max_recv:
            raise ValueError(
                "zero-copy + baseline 需要 --gemm2-accumulate (默认), 否则 GEMM2 "
                "输出 [max_recv*topk, model_dim] 与 shmem_comb_inp_tok "
                "[max_recv, model_dim] 形状不匹配。"
            )
        gemm2_out = disp_op.get_registered_combine_input_buffer(out_dtype)
        gemm2_out.zero_()
    else:
        gemm2_out = torch.zeros(out_rows, model_dim, dtype=out_dtype, device=dev)

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
        gemm2_topk=gemm2_topk,
        a2_rows=a2_rows,
        effective_valid=effective_valid,
        aiter_state=aiter_state,
    )


def _run_aiter_sorting(gemm2_in, disp_op):
    """Chain 内每次 dispatch 之后调一次：in-place 把 sorted_* / num_valid_ids
    重填为基于本次 dispatch 真实路由结果 (shmem_disp_out_idx) 的 sorting 输出。

    必须 capture-friendly：仅调 `aiter.moe_sorting_fwd` 一个 hipKernel，所有
    buffer 都是预分配的（``_build_gemm2_static_inputs`` 在 setup 阶段已分配）。
    每次 replay 用同一份 buffer，只刷新内容。
    """
    st = gemm2_in["aiter_state"]
    _aiter_moe_sorting_fwd(  # type: ignore[misc]
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
    """打印 GEMM2 的全部 compile-time / runtime 入参，方便调试与对齐生产。

    覆盖 launch 签名：
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

    print(f"\n{prefix} ═══ COMPILE-TIME (compile_mixed_moe_gemm2 kwargs) ═══")
    print(f"{prefix}   model_dim       = {args.hidden_dim}")
    print(f"{prefix}   inter_dim       = {args.inter_dim}")
    print(f"{prefix}   experts         = {args.num_experts_per_rank}")
    print(f"{prefix}   topk            = {gemm2_in['gemm2_topk']}  "
          f"(A2 行寻址: row = t * topk + s)")
    print(f"{prefix}   tile_m / n / k  = {args.tile_m2} / {args.tile_n2} / {args.tile_k2}")
    print(f"{prefix}   a_dtype         = {args.gemm2_a_dtype}")
    print(f"{prefix}   b_dtype         = {args.gemm2_b_dtype}")
    print(f"{prefix}   out_dtype       = {out_s}")
    print(f"{prefix}   accumulate      = {args.gemm2_accumulate}  "
          f"({'atomic_fadd 跨 s 累加到 output row t' if args.gemm2_accumulate else 'plain store, (t,s) 独占行'})")
    print(f"{prefix}   persist_m       = {args.persist_m}")
    print(f"{prefix}   xcd_swizzle     = {args.xcd_swizzle}")
    print(f"{prefix}   doweight_stage2 = False")

    print(f"\n{prefix} ═══ RUNTIME TENSORS (launch 参数, dtype/shape/dev) ═══")
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

    print(f"\n{prefix} ═══ RUNTIME SCALARS (launch 后 4 个 i32) ═══")
    print(f"{prefix}  [pos 10] tokens_in       = {gemm2_in['max_recv']}  "
          f"(= world_size * bs, A2 行 = tokens_in * topk = {gemm2_in['a2_rows']})")
    print(f"{prefix}  [pos 11] n_in (=model_dim) = {args.hidden_dim}")
    print(f"{prefix}  [pos 12] k_in (=inter_dim) = {args.inter_dim}")
    print(f"{prefix}  [pos 13] size_expert_ids = {gemm2_in['blocks']}  "
          f"(= epr * ceil(per_e / tile_m), expert-grouped m_block 数)")
    print(f"{prefix}  [pos 14] stream          = torch.cuda.current_stream()")

    print(f"\n{prefix} ═══ DERIVED / SEMANTIC ═══")
    print(f"{prefix}  num_valid_ids value     = {int(gemm2_in['num_valid_ids'][0].item())}  "
          f"(kernel 走 row ∈ [0, this); row-level t/s 哨兵在内部 DCE)")
    print(f"{prefix}  effective_valid (real)  = {gemm2_in['effective_valid']}  "
          f"(-1 = aiter 真实路由 sorting，由 num_valid_ids[0] 在设备端动态决定)")
    print(f"{prefix}  sorted_size             = {gemm2_in['sorted_size']}")

    # 解码 sorted_token_ids，统计 atomic contention
    try:
        all_raw = gemm2_in["sorted_token_ids"].tolist()
        max_recv_sentinel = int(gemm2_in["max_recv"])
        real = [(v & 0x00FFFFFF, (v >> 24) & 0xFF) for v in all_raw
                if (v & 0x00FFFFFF) < max_recv_sentinel]

        from collections import Counter
        t_counter: Counter = Counter(t for (t, _s) in real)
        s_counter: Counter = Counter(s for (_t, s) in real)
        cont_hist: Counter = Counter(t_counter.values())  # 多少个 t 有 d 个 s

        # 显示 expert 0 / 1 / 7 各自的前 8 条，证明同一 t 落到不同 expert
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
              f"(每个唯一 t 对应一个 output row, atomic_fadd target)")
        print(f"{prefix}  s value distribution            = "
              f"{dict(sorted(s_counter.items()))}")
        print(f"{prefix}  atomic contention histogram (#s-per-t -> #t) = "
              f"{dict(sorted(cont_hist.items()))}")
        print(f"{prefix}    -> 含义: 同一 output row 被几次 atomic_fadd 命中的统计")
    except Exception as exc:
        print(f"{prefix}  (decode failed: {exc!r})")

    print()


def _build_gemm2_callable(args, gemm2_in, out_dtype):
    """编译 mixed_moe_gemm2，返回 launch(o, x, w, sx, sw, st, eids, sw_sorted) 形式。"""
    out_s = "bf16" if out_dtype == torch.bfloat16 else (
        "f16" if out_dtype == torch.float16 else "f32"
    )

    # accumulate=True 走 atomic; accumulate=False 走 reduce 模式
    # 注：历史上 fp4 b_dtype 曾被强制 accumulate=False（因怀疑 atomic_fadd 在
    # mixed fp4 lowering 下 silent-drop），现解除该强制以 empirically 验证。
    # 若发现 gemm2_out 全 0，再单独 PR 修 kernel。
    accumulate = args.gemm2_accumulate

    # GEMM2 topk 与 dispatch 路由 topk (args.k) 对齐：每个 recv token 在本地最多
    # 被 topk 个 expert 命中, A2 buffer 按 [tokens_in * topk, inter_dim] 寻址。
    exe = compile_mixed_moe_gemm2(
        model_dim=args.hidden_dim,
        inter_dim=args.inter_dim,
        experts=args.num_experts_per_rank,
        xcd_swizzle=args.xcd_swizzle,
        topk=gemm2_in["gemm2_topk"],
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
    """baseline: [dispatch ->] [aiter_sorting ->] moe_gemm2 -> combine.

    dispatch_inputs=None  : 旧路径, 仅 launch GEMM2 + combine (combine 跑空).
    dispatch_inputs=(inp, wts, scales, idx) : 每次 chain 都重跑 dispatch,
        让 combine 内部的 total_recv / routing tables 永远 fresh.

    aiter sorting 在 dispatch 之后插入：必须每次 replay 都重跑 sorting，
    因为 dispatch 的 dest_tok_all 分配 (atomic_add 顺序) 不是 deterministic，
    每次 replay shmem_disp_out_idx 的行内容会变；sorted 里的 t 索引必须
    跟着 dispatch 当前 replay 的 addr_tis 一致。
    """
    if dispatch_inputs is not None:
        inp, wts, scales, idx = dispatch_inputs
        # disp_op.dispatch 内部已经 cache 编译好的 jit kernel, capture 时直接
        # launch; routing tables / total_recv 在 disp_op 内部 shmem buffer 上
        # 原地更新, combine_idx (= shmem_disp_out_idx 视图) 引用不变.
        disp_op.dispatch(inp, wts, scales, idx)
        # capture aiter.moe_sorting_fwd to refresh sorted_* tables in-place
        _run_aiter_sorting(gemm2_in, disp_op)
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
    """fused: [dispatch ->] [aiter_sorting ->] fused_gemm2_combine [-> combine_no_stage1].

    见 _baseline_chain 关于 aiter sorting 必须 capture 进 chain 的说明.

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
        _run_aiter_sorting(gemm2_in, disp_op)
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
                 atol_ulp_k=16, max_print=8):
    """逐 token 对比两个张量，输出统计 + 部分逐元素差异。返回 pass:bool。"""
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

    # 用 finite 掩码裁掉两边都 NaN/Inf 的位置再算 abs/rel 统计：fp8_direct_cast
    # 下 random fp4 数据偶尔触发 cvt_pk_fp8_f32 输出 fp8 NaN 编码（baseline 与
    # fused 同源数据 → 同一组位置出 NaN，本不影响数值一致性比较）。
    # 失败判定：(1) NaN 位置 mismatch (a 路径有 NaN，b 路径没有 / 反之) 才算 FAIL；
    #         (2) finite 部分内 abs/rel 超 thr 才算 FAIL。
    finite_mask = ~(nan_a | nan_b)
    a_finite = torch.where(finite_mask, a_f, torch.zeros_like(a_f))
    b_finite = torch.where(finite_mask, b_f, torch.zeros_like(b_f))
    diff   = (a_finite - b_finite).abs()
    # 用 max(|a|,|b|) 当分母，避免 a=0 时 rel 爆炸（baseline 在 dup 哨兵 +
    # zero-fragment 边界上偶有 0 输出而 fused 有 ~1 ULP non-zero）。
    a_norm = torch.maximum(a_finite.abs(), b_finite.abs()).clamp_min(1e-6)
    rel    = diff / a_norm

    abs_max  = diff.max().item() if diff.numel() > 0 else 0.0
    abs_mean = diff.mean().item() if diff.numel() > 0 else 0.0
    rel_max  = rel.max().item() if rel.numel() > 0 else 0.0
    rel_mean = rel.mean().item() if rel.numel() > 0 else 0.0

    # Element-wise pass: 四阈值任一满足即通过：
    #   (1) |diff| <= atol_abs  → 绝对阈值（覆盖小值噪声）
    #   (2) |diff| / max(|a|,|b|) <= atol_rel → 相对阈值（覆盖大值 round）
    #   (3) |diff| <= atol_ulp_k * max(|a|,|b|) * 2^-mantissa → dtype-aware
    #       k-ULP 累加预算（bf16: 2^-7, atol_ulp_k=8 → 容忍 6.25% 的 round 累加）
    #   (4) max(|a|,|b|) <= near_zero_floor → near-zero 噪声地带，sign-flip 视
    #       为 atomic_fadd 顺序与 fp32 single-accum 顺序的真实非结合性差异
    #       （在 atomic8_1pe / atomic-k stress 场景下不可避免）
    if ta.dtype == torch.bfloat16:
        _mantissa_bits = 7
    elif ta.dtype == torch.float16:
        _mantissa_bits = 10
    elif ta.dtype == torch.float32:
        _mantissa_bits = 23
    else:
        _mantissa_bits = 7
    _ulp_thr = (a_norm * (atol_ulp_k * (2.0 ** -_mantissa_bits)))
    # near-zero floor: 取 tensor abs max 的 (k * 2^-mantissa) 倍，对 bf16 k=8 ≈
    # 输出量级的 6.25% — 该量级下任何 sign-flip 都属于 accumulation noise。
    _out_max = a_finite.abs().max().clamp_min(1e-6)
    _near_zero_floor = _out_max * (atol_ulp_k * (2.0 ** -_mantissa_bits))
    _is_near_zero = (a_finite.abs() < _near_zero_floor) & (b_finite.abs() < _near_zero_floor)
    fail_mask = ((diff > atol_abs) & (rel > atol_rel) & (diff > _ulp_thr)
                 & (~_is_near_zero))
    n_diff_per_tok = fail_mask.reshape(fail_mask.shape[0], -1).any(dim=-1).sum().item()
    n_fail = int(fail_mask.sum().item())
    pass_ok  = (n_fail == 0) and nan_match

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
    # accumulate=True 走 atomic_fadd，gemm2_out 必须在每次 launch 前清零；
    # 否则 setup 阶段 _build_gemm2_callable 的 flyc.compile() 已经跑过一次
    # GEMM2，本次 launch 会把 frag 累加到上次残留值 → gemm2_out = 2*frag →
    # base_out_tok = 2*fused_out_tok 的根因（fused 路径走 plain store 不受影响）。
    gemm2_in["gemm2_out"].zero_()
    torch.cuda.synchronize()
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
            # 直接比 base 单 slot (atomic-8 累加值) 与 fused 8 slots 之和 (Plan B 期望相等):
            # baseline (intra-rank) 写到 b_tok[0, src_lid] = sum 8 frag (atomic).
            # fused 写到 raw slot (src_lid * k + s) for s in 0..7 → 这 8 个 slot 的和应等于 baseline 1 slot.
            # 注意: shmem_comb_inp_tok 是 int16 1D buffer, 真实数据是 bf16, 必须 view 回 bf16 再比较.
            try:
                b_bf16 = base_inp_tok_step1.view(torch.bfloat16)
                f_bf16 = fused_inp_tok_only.view(torch.bfloat16)
                k_t = cfg_t.num_experts_per_token
                tot_recv = int(_saved_total_recv.item())
                log2_mt2 = int(mt_t).bit_length() - 1
                mask_mt2 = mt_t - 1
                npes_t2  = cfg_t.world_size
                # 找几个 dest_pe=self_pe 的 intra-rank t (rank 0 自己 P2P self)
                cnt = 0
                for t in range(tot_recv):
                    enc = int(tis_view[t].item())
                    if (enc >> log2_mt2) != self_pe:
                        continue
                    src_lid = enc & mask_mt2
                    # baseline 写 view[dest_pe=self_pe, src_lid] = 1 slot
                    b_row = b_bf16.view(npes_t2, mt_t, hd_t)[self_pe, src_lid].float()
                    # fused 写 raw slot (src_lid * k + s) for s in 0..k-1, 在 1D buffer 里
                    # 也即 view (max_recv, hd) → 行 src_lid*k+s. 此处用 view (max_recv, hd) 切.
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
        print(f"  [rank {rank}] fused_out_tok stats: "
              f"min={fused_tok_s.float().min().item():.4e} "
              f"max={fused_tok_s.float().max().item():.4e} "
              f"abs_mean={fused_tok_s.float().abs().mean().item():.4e} "
              f"nz={int((fused_tok_s != 0).sum().item())}", flush=True)
        # P0 诊断：dump fused_gemm2 P2P scatter 后 shmem_comb_inp_tok 内
        # mt*k=256 slot 各自 nz，找 race 源头。slot_id = src_lid * k + j；
        # 期望全 256 slot 非零；fail 时部分 slot=0 → 那些 (src_lid, j) 的
        # fragment 没 land。配合 j 维统计推断是某些 dest_pe 全 miss。
        _cfg_d = disp_op.cfg
        _mt_d2  = _cfg_d.max_num_inp_token_per_rank
        _k_d2   = _cfg_d.num_experts_per_token
        _hd_d2  = _cfg_d.hidden_dim
        _post_fused_inp_tok = disp_op.shmem_comb_inp_tok.detach().clone()
        _slot_view = _post_fused_inp_tok.view(torch.bfloat16).view(_mt_d2 * 8, -1)[:_mt_d2 * _k_d2]
        _slot_any = (_slot_view != 0).any(dim=-1).cpu().tolist()
        _slot_nz_cnt = sum(_slot_any)
        # 按 j 维统计：第 j 维 nz slot 数 = sum_{src_lid} _slot_any[src_lid*k+j]
        _per_j_nz = [sum(_slot_any[src*_k_d2 + j] for src in range(_mt_d2))
                     for j in range(_k_d2)]
        # 列出 src_lid 是否完整接收 8 个 j fragment：sum_{j} slot_any[src*k+j]
        _per_src_nz = [sum(_slot_any[src*_k_d2 + j] for j in range(_k_d2))
                       for src in range(_mt_d2)]
        _full_src = [s for s, v in enumerate(_per_src_nz) if v == _k_d2]
        _empty_src = [s for s, v in enumerate(_per_src_nz) if v == 0]
        # P0 关键诊断：dump sorted_token_ids 是否包含所有 32*8=256 (t, j) 对。
        # 若 sorted 内 valid row 不足 256，说明上游 aiter_sorting 漏了 row → 
        # fused_gemm2 没处理那些 row → 对应 slot=0。
        # num_valid_ids[0] 是 aiter 给的 padded-total，包含 padding；
        # num_valid_ids[1] 是 actual input tokens (= total_recv)。
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
              f"aiter num_valid=[{_nv0},{_nv1}] sti_s_ok={_sti_ok}",
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

    # 跨卡聚合 pass/fail（rank 维 SUM 出 fail rank 列表帮定位）
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


# ─── 各模式入口（profile+cudagraph + bench+eager 实现）──────────────────────
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

    rocprofv3-friendly：每次调用都走真实的 hipLaunchKernel 路径，让
    rocprofv3 / ATT trace 能看到 chain 里所有内部 kernel name。
    用 torch.cuda.Event 量端到端时间；all-reduce min/max/avg 跨卡聚合。
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

    # ── cfg / dispatch op ───────────────────────────────────────────────────
    _dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    cfg = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world_size,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=cur_tok,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=k,
        data_type=_dtype,
        combine_warp_num_per_block=args.warp_per_block,
        combine_block_num=args.block_num,
        chip=args.chip,
        use_external_inp_buf=args.use_external_inp_buf,
        enable_std_moe=args.enable_std_moe,
        scale_dim=args.scale_dim,
        scale_type_size=args.scale_type_size,
        quant_type=args.quant_type,
        use_token_flag_sync=args.token_flag_sync,
    )
    args.max_num_inp_token_per_rank = cur_tok  # 给 _build_gemm2_static_inputs 使用

    # ── FP4 hard constraints ────────────────────────────────────────────────
    # FlyDSL FP4 mfma_scale_x128 path 隐式假设：
    #   * tile_k >= 256 （scale layout 一格覆盖 128 fp4 elements，<256 直接全 0）
    #   * inter_dim/model_dim >= 256 且能整除 256
    # 不满足这些约束时不抛错，只是 GEMM2 silent-output 0；为了不再让下次踩坑，
    # 在 verify/bench 入口直接报错。
    # accumulate=True/False 不再做硬约束，由 --gemm2-accumulate 控制。
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
        print(f"  bench-op={args.bench_op}, fuse-mode={args.fuse_mode}, "
              f"fused_op_available={HAS_FUSED_OP}")
        if not HAS_FUSED_OP:
            print(f"  [warn] fused op import failed: {_FUSED_IMPORT_ERR}")
        print(f"{'='*78}")

    disp_op = FlyDSLDispatchCombineIntraNodeOp(cfg)

    fused_op = None
    if HAS_FUSED_OP and args.bench_op in ("fused", "both"):
        fused_op = FlyDSLMoeGemm2CombineOp(
            comb_cfg=cfg,
            comb_op=disp_op,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m2, tile_n=args.tile_n2, tile_k=args.tile_k2,
            persist_m=args.persist_m,
            a_dtype=args.gemm2_a_dtype, b_dtype=args.gemm2_b_dtype,
            force_mode=args.fuse_mode,
            xcd_swizzle=args.xcd_swizzle,
            use_token_flag_sync=args.token_flag_sync,
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
        _rt = getattr(args, "routing", "random")
        _hr = int(getattr(args, "gemm2_scale_headroom", 0))
        print(f"[setup] routing={_rt} sorting=aiter "
              f"aiter_backend={_AITER_SORTING_BACKEND}; "
              f"aiter sorting available = {HAS_AITER_SORTING}; "
              f"gemm2_scale_headroom={_hr} (e8m0 scale = {127 - _hr}, "
              f"GEMM2 magnitude × 2^-{2*_hr})")
        if not HAS_AITER_SORTING:
            print(f"[setup] ERROR: aiter.moe_sorting_fwd unavailable: "
                  f"{_AITER_SORTING_ERR}; will fail in _build_gemm2_static_inputs.")
        print(f"[setup] running one-shot dispatch to populate routing tables…")
    disp_ret = disp_op.dispatch(inp, wts, scales, idx)
    combine_idx = disp_ret[3]                # shmem_disp_out_idx 视图
    dispatch_total_recv = disp_ret[4]        # shmem total_recv 标量
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    if rank == 0 and args.bench_op in ("fused", "both") and HAS_FUSED_OP:
        print(f"[setup] dispatch total_recv (rank0) = "
              f"{int(dispatch_total_recv.item())}")

    # 注意 (aiter sorting 时序)：build 必须在 hard-reset *之前* 跑。
    # _build_gemm2_static_inputs 会调一次 moe_sorting_fwd (做首次 JIT + 占
    # 住 buffer)，需要 disp_op.total_recv 仍是 setup dispatch 写入的真实
    # 值；hard-reset 清 total_recv=0 之后跑 aiter，num_local_tokens 会被
    # 读成 0，sorted 全空（仅 buffer 形状有效）。实际 chain 内每次 replay
    # 重跑 dispatch 后 _run_aiter_sorting 会用 fresh total_recv 重填
    # sorted_*。
    gemm2_in = _build_gemm2_static_inputs(
        rank, world_size, dev, args, cfg, disp_op=disp_op,
    )

    # 端到端语义自检：zero-copy on 路径 GEMM2 必须直接把输出写到
    # ``shmem_comb_inp_tok``；否则 combine Stage1 跳过后 peer 端 P2P 读到
    # 的就是 stale shmem 数据。这里在 op 内部 raise 之前先打 banner 让
    # rank0 一眼看出 gemm2_out 的归属。
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

    # ── 方案 A 的 hard-reset: 只清 setup dispatch 留下的 *local* counter
    # (dest_pe_ctr / disp_bar / comb_bar / total_recv / disp_grid_bar),
    # **不要** 清 cross-device shmem buffer (shmem_xdev_bar_mem 用 monotonic
    # cur_flag 模式 mori 自己管理; shmem_comb_inp_* 下次 chain 自然覆盖).
    # 任意 zero shmem_* 会让 mori shmem 内部 cur_flag 跟实际 buffer 值脱节,
    # 表现为 capture 阶段第 1 次 fused gemm2 launch 时 hipErrorInvalidValue.
    #
    # 仅 profile/cudagraph 模式需要（chain 内 dispatch 会重新填这些 counter）。
    # verify 模式不走 cudagraph、也不会再跑一次 dispatch，hard-reset 会把
    # total_recv 清成 0 → 后续 combine Stage 1 跑 0 次循环 → 全 0 输出。
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
        block_num=cfg.combine_block_num,
        warp_per_block=cfg.combine_warp_num_per_block,
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
        if args.cudagraph:
            _not_impl("bench+cudagraph")
        # eager bench：rocprofv3 / ATT trace 友好（不走 cudagraph，不包 torch.profiler）。
        # chain 是否带 dispatch 由 --chain-include-dispatch 控制；对 fused 路径不带
        # dispatch 时 combine_no_stage1 也会跑空，参考 _fused_chain doc。
        chain_disp_inputs = (
            (inp, wts, scales, idx) if args.chain_include_dispatch else None
        )
        bench_results = {}
        if test_baseline:
            def _bl():
                return _baseline_chain(
                    disp_op, gemm2_launch, gemm2_in, combine_idx,
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
    p.add_argument("--max-tokens",           type=int, default=32,
                   help="单卡 dispatch 输入 token 数 (bs)；max_recv = world_size * bs。"
                        "默认 32 对齐生产均衡场景 (bs=32, ep=8, topk=8)。")
    p.add_argument("--hidden-dim",           type=int, default=7168,
                   help="GEMM2 输出维度（== combine token 维度 == dispatch token 维度）")
    p.add_argument("--inter-dim",            type=int, default=2048,
                   help="GEMM2 输入维度（GEMM1 输出维度，本骨架不跑 GEMM1）。"
                        "默认 2048 对齐 production a4w4（参考 ut_per1x32.py: "
                        "model_dim=7168, inter_dim=2048）。")
    p.add_argument("--num-experts-per-rank", type=int, default=32)
    p.add_argument("--k",                    type=int, default=8,
                   help="MoE top-k：同时驱动 dispatch 路由 topk 和 GEMM2 compile-time "
                        "topk（A2 行寻址 t*topk+s, output atomic accumulate 跨 s）。")
    p.add_argument("--routing",
                   choices=["random", "atomic1_8pe", "atomic8_1pe", "atomic2_4pe"],
                   default="random",
                   help="dispatch 输入 idx 的构造方式 (atomicN_Mpe 命名约定: 同 token 的 "
                        "k 个 expert 分布到 M 个不同 PE 上，每 PE 落 N=atomic_per_pe=k/M "
                        "个命中 → GEMM2 同一 output row 被 N 次 atomic_fadd 竞争)："
                        "random=每 token 在 k 个 PE 随机 expert（默认，保持历史行为）；"
                        "atomic1_8pe=确定性循环填充，同 token 的 k 个 expert 跨 k 个 PE，"
                        "atomic_per_pe=1，dispatch dedup 不触发，对应 production balanced "
                        "EP routing（精度 verify 推荐用此模式）；"
                        "atomic8_1pe=同 token 的 k 个 expert 是连续 k 个 local_eid，"
                        "全压同 1 个 dest_pe (atomic-k 累加 worst case)，卡间/卡内均衡。"
                        "require epr%%k==0 (32/8=4 ✓)；"
                        "atomic2_4pe=同 token 的 k 个 expert 分布在 *4 个* 不同 PE 上，"
                        "每 PE 上 k/4=2 个 atomic（atomic-2 中等 contention），dest_pe "
                        "为 (g%%ws + j_group) mod ws 四连 PE，local_eid 用 g//ws+j 避免"
                        " lattice 塌缩 → 卡间命中数 = cur_tok*k 完美均衡，卡内每 "
                        "(PE,local_eid) cell 命中 cur_tok*k/epr 次。"
                        "require k%%4==0 (8/4=2 ✓) 且 ws>=4 (8 ✓)。")
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
    # 默认 a=fp4/b=fp4（production a4w4：GEMM1 输出经 SiLU + per-1x32 量化为 fp4
    # 后再喂 GEMM2，参考 ut_per1x32.py）。早期默认是 a=fp8，性能数字偏离 production。
    p.add_argument("--gemm2-a-dtype",        type=str, default="fp4",
                   choices=["fp8", "fp4", "fp16", "int8"])
    p.add_argument("--gemm2-b-dtype",        type=str, default="fp4",
                   choices=["fp8", "fp4", "fp16", "int8", "int4"])
    p.add_argument("--tile-m2",              type=int, default=32)
    p.add_argument("--tile-n2",              type=int, default=128)
    p.add_argument("--tile-k2",              type=int, default=256,
                   help="GEMM2 tile_k；FP4 路径必须 >=256")
    p.add_argument("--persist-m",            type=int, default=-1,
                   help="moe_gemm2 沿 M 持久化 block 数")
    p.add_argument("--xcd-swizzle",          type=int, default=0,
                   help="moe_gemm2 跨 XCD swizzle 因子（=0 关闭；MI300 推荐 8）")
    p.add_argument("--gemm2-accumulate",     dest="gemm2_accumulate",
                   action="store_true", default=True,
                   help="GEMM2 epilogue 走 atomic-add（默认 True）")
    p.add_argument("--no-gemm2-accumulate",  dest="gemm2_accumulate",
                   action="store_false",
                   help="GEMM2 epilogue 走 plain store 的 reduce 模式（每行独立 slot）")
    p.add_argument("--gemm2-scale-headroom", type=int, default=0,
                   help="把 A2/W2 的 e8m0 micro-scale 从 127 降到 127-headroom "
                        "(每降 1 缩小 2×)。fp4×fp4 random 输入在 inter_dim=2048 "
                        "下 GEMM2 输出 ~±4000，经 fp8_direct_cast 的 bf16→fp8 cast "
                        "会 saturate 到 ±inf → 后续 cast 回 bf16 出 NaN，污染 verify "
                        "数值对比。verify 时建议 headroom=4 (256× 缩放) → GEMM2 "
                        "输出 ~±15，完全落在 fp8e4m3 max=±448 安全范围。profile/"
                        "bench 默认 0 不影响性能数据。")
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
    # D-flag C-1: per-token flag 同步开关。打开后 dispatch / combine kernel
    # 启用 reset + spin-wait，fused gemm2 epilogue 增加跨卡 atomic_add；关闭
    # 时整段 const_expr DCE，行为与 baseline 完全一致。仅在 --bench-op fused
    # / both（fused 路径）下有可观察效果。
    p.add_argument("--token-flag-sync", dest="token_flag_sync",
                   action="store_true", default=False,
                   help="启用 D-flag C-1 per-token flag 跨卡同步路径")
    p.add_argument("--no-token-flag-sync", dest="token_flag_sync",
                   action="store_false",
                   help="禁用 D-flag C-1 路径（默认行为）")
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
