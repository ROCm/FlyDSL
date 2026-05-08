# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
单机 2 卡（推荐 CUDA_VISIBLE_DEVICES=6,7）验证 internode v1 LL FlyDSL vs mori：

**精度（默认 --check both）**
  - ``--check dispatch``：只比对 dispatch 输出（tok/wts/idx，行序无关排序后比较）
  - ``--check combine``：只比对 combine 后的 token 张量（仍要求 ``total_recv`` 一致）
  - ``--check both``：dispatch + combine（可用 ``--no-combine`` 关闭 combine 段）

**性能（--bench，与 mori 同口径分段）**
  - 专用入口（优化时少敲参数、末尾汇总表）：``python tests/kernels/bench_internode_v1ll_flydsl.py``（见 ``--help``）
  - ``--bench-scope dispatch``：与 mori 一致，计时 **每轮仅** ``copy+main``；默认 **融合单 JIT**（``FLYDSL_INTERNODE_V1LL_SPLIT_DISPATCH=1`` 强制分拆）。**对比顺序**：先 **FlyDSL bench** 再 **mori bench**（避免高强度 mori 后 ROCm 上 FlyDSL 首帧 fault）；打印的 mori/FlyDSL 数值仍可比。
  - ``--bench-scope combine``：每轮先 redispatch（不计时）再只测 combine 段
  - ``--bench-scope e2e``：dispatch+combine 整段计时

用法::

  CUDA_VISIBLE_DEVICES=6,7 MORI_SHMEM_HEAP_SIZE=8G \\
    torchrun --standalone --nproc_per_node=2 \\
    tests/kernels/test_internode_v1ll_dispatch_flydsl.py

  （ROCm 多卡只暴露子集时，脚本会在导入 torch 前自动将 ``HIP_VISIBLE_DEVICES`` 设为与
  ``CUDA_VISIBLE_DEVICES`` 相同；否则 HIP 仍见全机 GPU，易 P2P 错卡、Memory access fault
  或假死。）

一键串联多场景精度 + dispatch/combine/e2e 性能（默认设备对 6,7）::

  cd FlyDSL && ./tests/kernels/run_internode_v1ll_8gpu_targeted.sh

  或::

  CUDA_VISIBLE_DEVICES=6,7 HIP_VISIBLE_DEVICES=6,7 MORI_SHMEM_HEAP_SIZE=8G \\
    python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py --by-scope

  （``--by-scope``：每个 scope 先精度再性能；``--quick`` 子集；``--dry-run`` 列命令；
  ``--skip-perf`` / ``--skip-correctness`` 可选。形状等可附加 ``--hidden 4096 --n-tok 32``。）

可选: --skip-mori 仅跑 FlyDSL（不做 golden 对比）
  --mori-only     只跑 mori EpDispatchCombineOp（用于确认参考路径是否挂）
  --flydsl-only   只跑 FlyDSL（跳过 mori）

排障: 默认开启 FlyDSL 主机侧步进日志（``[flydsl trace r=N]``，每个 JIT 前后 ``cuda.synchronize``）。
  关闭: ``export FLYDSL_TRACE_INTERNODE=0``
  **性能**: ``--bench`` 时（除非 ``FLYDSL_TRACE_INTERNODE_BENCH=1``）会自动抑制上述 trace：
  ``_trace_flydsl_ordered`` 每条消息会对 ``world_size`` 次 ``dist.barrier``，combine 段数百轮会像卡死。
  精确定位 fault 的 kernel: ``export FLYDSL_TRACE_SYNC_LAUNCH=1``（在 import torch 前生效，会设 ``HIP_LAUNCH_BLOCKING``）。
  若 fault 紧接在 ``jits ready`` 之后: 脚本在 **全部 JIT 返回后** 会先 ``gc.collect``/``empty_cache``、多轮 ``cuda.synchronize``、再 ``dist.barrier`` + ``shmem_barrier_all`` + 第二轮 barrier/sync；**在 rank0 打印 ``jits ready`` 之后必须再 barrier**（否则 rank1 会抢先进入 warmup，ROCm 上易在 6/7 等卡对上报 ``0x11000000000`` 类 fault）；**小 ``zeros`` 预热** 默认（2 卡）亦 **按 rank 串行**（恢复并行 warmup: ``FLYDSL_PARALLEL_POST_JIT_WARMUP=1``）；随后默认 **串行** ``randn``/``idx``（``FLYDSL_PARALLEL_POST_JIT_ALLOC=1`` 恢复并行首分配）；**dispatch 路径大块 mori shmem**（``dispatch_inp``…``shmem_tis``）默认 **按 rank 串行** ``zero_``（``FLYDSL_PARALLEL_SHMEM_ZERO=1`` 恢复两卡并行清零）；combine 路径 ``combine_inp``/``combine_out``/``xdb`` 仍 **逐张量 + barrier**。开启 ``FLYDSL_TRACE_INTERNODE=1``（默认）可看到 ``post-jit: …`` 步进行；``FLYDSL_TRACE_MILESTONE=1`` 可见 ``milestone``。
  **MORI_SHMEM_HEAP_SIZE**: 对称堆为单块 ``hipMalloc``；若 **空闲显存 < 堆大小 + ~1.5GiB** 余量，会在 ``shmem`` 初始化前 ``RuntimeError``。在满卡/碎片严重下 **5G** 仍可能在首帧 kernel 上 **Memory access fault**（低地址），建议 **6–8G** 且保证该卡有足够 **连续** 空闲显存。
  怀疑命中旧 FlyDSL 磁盘 JIT: 删 ``~/.flydsl/cache`` 或设 ``FLYDSL_RUNTIME_CACHE_DIR`` 到空目录后再跑（``smem`` 符号 / copy tag 版本 bump 后仍会重编，但清缓存最稳）。
  combine 段另有 ``ordered`` 行（barrier 串行打印）与 ``combine pre-main diag`` 主机读回关键缓冲。
  更细 combine 定位: ``FLYDSL_TRACE_COMBINE_DETAIL=1``（main_ll 后 chunk_flag/cfc/inter_bar；all 后 comb 张量统计；mori golden 亦打印；可与 ``FLYDSL_TRACE_INTERNODE=0`` 联用以减少其它日志）。
  **密钉一步定位**: 默认打印 ``[flydsl pin] P001 …``（``FLYDSL_TRACE_PINPOINT=0`` 关闭）。约定: P001–P010 编译后准备与清零；**P019** = ``copy_jit`` 前跨 rank fence；P020–P024 dispatch（默认 **融合** ``copy+main`` 时仅 **P021** / **P024**，中间 **P022/P023** 不出现；``FLYDSL_INTERNODE_V1LL_SPLIT_DISPATCH=1`` 恢复分拆）；**P025** = mori-like bench 下每轮 dispatch 后 ``dist`` + ``shmem_barrier``（防多进程回合重叠）；P030–P040 combine；**P033b** = ``combine_bar_jit`` 前 ``cuda.sync`` + ``dist`` + ``shmem_barrier``；**P035b** = ``combine_main_jit`` 前同构 fence（关闭 internode trace 时 ``_trace_flydsl_ordered`` 不再 barrier，此处仍保证跨 PE 序）；**P037** = 紧挨 ``combine_main_jit`` 的 Python launch 前；若 fault 后无 **P038**，多半是 main LL 内核或更早。两 rank 对照 **最后一条 pin**。
  **长 bench 钉节流**: ``FLYDSL_BENCH_PIN_STRIDE=N``（可选）时仅 ``n<=FLYDSL_BENCH_PIN_HEAD``（默认 4）或 ``n%N==0`` 打印 P019–P025；默认不节流（每轮全打）。
  **设备侧 printf**（ROCm 上需同步后才易见）: ``FLYDSL_DEVICE_TRACE_DISPATCH=1`` → main 入口 / DispatchSync+quiet 后各打一行（与 host ``seq`` 对齐）。多 rank 并发 printf 会交错乱码；``FLYDSL_DEVICE_TRACE_DISPATCH_RANK0_ONLY=1`` 时仅 rank0 传非零 trace（rank1 仍跑同一内核，只是不打设备 printf）。
  **dispatch 后主机读回**: ``FLYDSL_TRACE_DISPATCH_POSTSYNC_READBACK=1`` → 每轮 ``cuda.sync`` 后打印 ``total_recv``、``recv_tok_num`` 前 8 元、``disp_grid_bar``、``dest_pe_ctr``（遵守 ``PIN_STRIDE`` 节流）。
  **一键排障横幅**: ``FLYDSL_TRACE_DISPATCH_DEBUG_BANNER=1`` → rank0 打印关键环境变量取值。
  **Bench / P019 / P025 / ``FLYDSL_BENCH_DISPATCH_HOST_FENCE_EVERY_ITER``**（简称「每轮主机 fence」）:
  仅当 ``--bench --bench-scope dispatch`` 且 **kernel-only**（计时段只跑 ``copy+main``、不含每轮 Python reset）时生效。
  - **未设为 0/false/no**（**默认**）: 每次 timed/warmup 调用 ``_launch_copy_main`` **前**都做 P019（``cuda.sync`` + ``dist.barrier`` + ``shmem_barrier_all``）→ **稳**，但与 mori ``dispatch()`` 的「计时环内无主机栅栏」**不完全同口径**。
  - **设为 0 / false / no**（mori-like）: **仅第 1 次** launch 前做 P019；**每轮 dispatch 结束后**自动做 **P025**（``dist.barrier`` + ``shmem_barrier_all``）。原因：``torchrun`` 多进程下 ``bench_gpu_us_torch`` 在 rank 之间**没有**隐式回合同步，若既不 P019 也不 P025，则某一 rank 可能在本 rank 上 enqueue **下一轮** internode 时，对端仍在 **上一轮** 的 GPU 协议里，RDMA/DispatchSync 回合重叠 → **Memory access fault / SIGABRT**。P025 把「一轮 internode dispatch」封成全局原子回合，**不计入** mori ``dispatch()`` 里「launch 前无 fence」的狭义口径，但保证多进程压测可跑通。
  设备侧 FlyDSL main 已与 mori ``v1::DispatchSync`` 对齐（``total_recv`` 后、清 recv 信号前有 ``threadfence_system``）。``FLYDSL_BENCH_DISPATCH_INCLUDE_RESET=1`` 时 ``_reset_state`` 会清零 launch 计数。Bench 要打 P019–P025 的 pin：``FLYDSL_TRACE_INTERNODE_BENCH=1``（否则 ``--bench`` 会关掉 pinpoint）。
  barrier 卡点: ``FLYDSL_TRACE_TS=1`` 给 ``[flydsl trace]`` / combine-detail 行加 ``time.monotonic()`` 前缀。
  ROCm+jits ready 后立刻 fault（无 ``copy_jit`` 行）: 默认已不再用 ``_foreach_zero_`` 清零对称缓冲；
    若要恢复批量清零以省 launch: ``FLYDSL_FOREACH_ZERO_BATCH=1``。
  --std-moe       FlyDSL main_jit 启用 Phase4 ConvertDispatchOutput（expert-major packed 缓冲）
  --bench / --bench-scope / --bench-warmup / --bench-iters  见 ``--help``

与 mori ``tests/python/ops/test_dispatch_combine_internode_v1.py`` 中
``internode_v1_ll`` 的 ``_KERNEL_CONFIGS`` 对齐：默认 block_num=256、rdma_block_num=128。
``recv_tok_num`` 对称区大小须与 ``dispatch_combine.cpp`` 的
``InitializeTokenNumSignalBuf`` 一致：``worldSize * sizeof(index_t) * 2 * numQpPerPe``，
仅分配 ``world_size`` 个 int32 会短一半，存在越界风险。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Callable
import gc
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON = os.path.join(_ROOT, "python")
_TESTS_KERNELS = os.path.dirname(os.path.abspath(__file__))
# torch may append _PYTHON after site-packages; a partial flydsl on PYTHONPATH then wins.
# Remove and prepend so this repo's ``python/`` package shadows site-packages.
for _p in reversed((_TESTS_KERNELS, _ROOT, _PYTHON)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

# ROCm: set before ``import torch`` / HIP init.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _cvd and not os.environ.get("HIP_VISIBLE_DEVICES", "").strip():
    os.environ["HIP_VISIBLE_DEVICES"] = _cvd

# Optional: FLYDSL_TRACE_SYNC_LAUNCH=1 → HIP_LAUNCH_BLOCKING so the last printed line is the faulting kernel.
if os.environ.get("FLYDSL_TRACE_INTERNODE", "1") not in ("0", "false", "False"):
    if os.environ.get("FLYDSL_TRACE_SYNC_LAUNCH", "0") not in ("0", "false", "False"):
        os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")

import torch
import torch.distributed as dist

# 与 mori test_dispatch_combine_internode_v1._KERNEL_CONFIGS["internode_v1_ll"] 一致
MORI_INTERNODE_V1_LL_BLOCK_NUM = 256
MORI_INTERNODE_V1_LL_RDMA_BLOCKS = 128

import flydsl.expr as fx
import mori.shmem as ms
from mori.ir.flydsl.runtime import install_jit_hook
from mori.shmem import mori_shmem_create_tensor

from benchmark_common import bench_gpu_us_torch

# Set True during ``run_flydsl_dispatch(..., bench=True)`` unless FLYDSL_TRACE_INTERNODE_BENCH=1.
_SUPPRESS_INTERNODE_TRACE_FOR_BENCH = False

from kernels.dispatch_combine_internode_v1ll_kernel import (
    make_copy_staging_jit,
    make_dispatch_internode_v1ll_copy_main_fused_jit,
    make_dispatch_internode_v1ll_main_jit,
)
from kernels.dispatch_combine_internode_v1ll_combine_kernel import (
    make_combine_internode_v1ll_all_jit,
    make_combine_internode_v1ll_main_ll_jit,
    make_combine_internode_v1ll_sync_barrier_jit,
    make_combine_internode_v1ll_sync_jit,
)


def _parse_mori_shmem_heap_bytes() -> int | None:
    """Parse ``MORI_SHMEM_HEAP_SIZE`` (e.g. ``5G``, ``512M``). Returns None if unset/unparseable."""
    raw = os.environ.get("MORI_SHMEM_HEAP_SIZE", "").strip()
    if not raw:
        return None
    u = raw.upper().replace(" ", "")
    try:
        if u.endswith("GIB"):
            return int(u[:-3]) * 1024**3
        if u.endswith("GB"):
            return int(u[:-2]) * 1024**3
        if u.endswith("G"):
            return int(u[:-1]) * 1024**3
        if u.endswith("MIB"):
            return int(u[:-3]) * 1024**2
        if u.endswith("MB"):
            return int(u[:-2]) * 1024**2
        if u.endswith("M"):
            return int(u[:-1]) * 1024**2
    except ValueError:
        return None
    return None


# Headroom beyond the single ``hipMalloc`` symmetric heap: PyTorch, NCCL, JIT, mori kernels.
_MORI_HEAP_VRAM_SLACK_BYTES = int(1.5 * 1024**3)


def _preflight_vram_before_mori_shmem_init(dev: torch.device, rank: int) -> None:
    """Fail fast if this GPU cannot satisfy symmetric heap + slack (avoids opaque GPU page faults)."""
    need = _parse_mori_shmem_heap_bytes()
    if need is None:
        return
    free_b, total_b = torch.cuda.mem_get_info(dev)
    if free_b >= need + _MORI_HEAP_VRAM_SLACK_BYTES:
        return
    raise RuntimeError(
        f"[rank{rank}] GPU free VRAM too low for MORI_SHMEM_HEAP_SIZE: "
        f"free={free_b / 1024**3:.2f} GiB, symmetric_heap={need / 1024**3:.1f} GiB, "
        f"required_slack≥{_MORI_HEAP_VRAM_SLACK_BYTES / 1024**3:.1f} GiB. "
        f"Either use a less congested GPU, lower MORI_SHMEM_HEAP_SIZE so heap+slack fits in free memory, "
        f"or free VRAM from other processes. Sub-6G heaps on nearly-full GPUs often fault in combine."
    )


def bench_gpu_us_prep_body(
    prep: Callable[[], None],
    body: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    """Mean microseconds per iteration for ``body`` only; ``prep`` runs before each (un-timed) pair in warmup too."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        prep()
        start.record()
        body()
        end.record()
        torch.cuda.synchronize(device)
    total_ms = 0.0
    for _ in range(iters):
        prep()
        start.record()
        body()
        end.record()
        torch.cuda.synchronize(device)
        total_ms += start.elapsed_time(end)
    return total_ms * 1e3 / iters


def setup_distributed(rank: int, world_size: int, master_port: int = 29611) -> torch.device:
    if "LOCAL_RANK" not in os.environ:
        os.environ.update(
            {
                "LOCAL_RANK": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
            }
        )
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=dev,
    )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    _preflight_vram_before_mori_shmem_init(dev, rank)
    ms.shmem_torch_process_group_init("default")
    return dev


def cleanup():
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _xfer_bytes(hidden_dim: int, elem_sz: int, k: int, scale_b: int) -> int:
    return hidden_dim * elem_sz + k * 4 + k * 4 + 4 + scale_b


def _flydsl_trace_on() -> bool:
    """Host-side step tracing for internode v1ll FlyDSL (set FLYDSL_TRACE_INTERNODE=0 to disable)."""
    if _SUPPRESS_INTERNODE_TRACE_FOR_BENCH:
        return False
    return os.environ.get("FLYDSL_TRACE_INTERNODE", "1") not in ("0", "false", "False")


def _flydsl_trace_ts_on() -> bool:
    return os.environ.get("FLYDSL_TRACE_TS", "0") not in ("0", "false", "False")


def _flydsl_trace_combine_detail_on() -> bool:
    if _SUPPRESS_INTERNODE_TRACE_FOR_BENCH:
        return False
    return os.environ.get("FLYDSL_TRACE_COMBINE_DETAIL", "0") not in ("0", "false", "False")


def _flydsl_pinpoint_on() -> bool:
    """Dense step codes [flydsl pin] for one-run fault localization (default on when not bench)."""
    if _SUPPRESS_INTERNODE_TRACE_FOR_BENCH:
        return False
    return os.environ.get("FLYDSL_TRACE_PINPOINT", "1") not in ("0", "false", "False")


def _bench_pin_stride() -> int | None:
    """If set (positive int), kernel-only bench pins only print for n<=HEAD or n%%STRIDE==0."""
    raw = os.environ.get("FLYDSL_BENCH_PIN_STRIDE", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _bench_pin_head() -> int:
    raw = os.environ.get("FLYDSL_BENCH_PIN_HEAD", "4").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 4


def _bench_emit_pin_for_iter(n: int) -> bool:
    k = _bench_pin_stride()
    if k is None:
        return True
    h = _bench_pin_head()
    return n <= h or (n % k == 0)


def _trace_pin(
    rank: int, code: str, note: str = "", *, bench_iter: int | None = None
) -> None:
    if not _flydsl_pinpoint_on():
        return
    if bench_iter is not None and not _bench_emit_pin_for_iter(bench_iter):
        return
    ts = f"t={time.monotonic():.3f}s " if _flydsl_trace_ts_on() else ""
    tail = f" {note}" if note else ""
    print(f"[flydsl pin r={rank}] {ts}{code}{tail}", flush=True)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def _device_trace_dispatch_level(*, rank: int) -> int:
    if not _env_flag("FLYDSL_DEVICE_TRACE_DISPATCH", "0"):
        return 0
    if _env_flag("FLYDSL_DEVICE_TRACE_DISPATCH_RANK0_ONLY", "0") and rank != 0:
        return 0
    return 1


def _trace_dispatch_postsync_readback(
    rank: int,
    dev: torch.device,
    tag: str,
    *,
    bench_iter: int | None,
    total_recv: torch.Tensor,
    cross_dev_flag: torch.Tensor,
    recv_tok_num: torch.Tensor,
    disp_grid_bar: torch.Tensor,
    dest_pe_ctr: torch.Tensor,
) -> None:
    if not _env_flag("FLYDSL_TRACE_DISPATCH_POSTSYNC_READBACK", "0"):
        return
    if bench_iter is not None and not _bench_emit_pin_for_iter(bench_iter):
        return
    torch.cuda.synchronize(dev)
    tr = int(total_recv.detach().cpu().item())
    cdf = int(cross_dev_flag.detach().cpu().item())
    dgb = disp_grid_bar.detach().cpu().to(torch.int32).tolist()
    dpc = dest_pe_ctr.detach().cpu().to(torch.int32).tolist()
    ne = min(8, recv_tok_num.numel())
    rts = recv_tok_num.detach().cpu().to(torch.int32).numpy().ravel()[:ne].tolist()
    ts = f"t={time.monotonic():.3f}s " if _flydsl_trace_ts_on() else ""
    bi = f" bench_iter={bench_iter}" if bench_iter is not None else ""
    print(
        f"[flydsl dispatch-readback r={rank}] {ts}{tag}{bi} "
        f"total_recv={tr} cross_dev_flag={cdf} "
        f"recv_tok_num[:{ne}]={rts} disp_grid_bar={dgb} dest_pe_ctr={dpc}",
        flush=True,
    )


def _print_dispatch_debug_banner(rank: int, world_size: int) -> None:
    if not _env_flag("FLYDSL_TRACE_DISPATCH_DEBUG_BANNER", "0"):
        return
    keys = (
        "FLYDSL_TRACE_SYNC_LAUNCH",
        "HIP_LAUNCH_BLOCKING",
        "FLYDSL_TRACE_INTERNODE_BENCH",
        "FLYDSL_BENCH_DISPATCH_HOST_FENCE_EVERY_ITER",
        "FLYDSL_DEVICE_TRACE_DISPATCH",
        "FLYDSL_DEVICE_TRACE_DISPATCH_RANK0_ONLY",
        "FLYDSL_TRACE_DISPATCH_POSTSYNC_READBACK",
        "FLYDSL_BENCH_PIN_STRIDE",
        "FLYDSL_BENCH_PIN_HEAD",
        "FLYDSL_TRACE_PINPOINT",
        "FLYDSL_TRACE_TS",
        "FLYDSL_INTERNODE_V1LL_SPLIT_DISPATCH",
    )
    lines = [f"[flydsl dispatch-debug-banner r={rank}/{world_size}]"]
    for k in keys:
        lines.append(f"  {k}={os.environ.get(k, '')!r}")
    print("\n".join(lines), flush=True)


def _trace_combine_detail(rank: int, msg: str) -> None:
    """Verbose combine diagnostics (independent of FLYDSL_TRACE_INTERNODE)."""
    if not _flydsl_trace_combine_detail_on():
        return
    ts = f"t={time.monotonic():.3f}s " if _flydsl_trace_ts_on() else ""
    print(f"[flydsl combine-detail r={rank}] {ts}{msg}", flush=True)


def _trace_internode_or_detail_line(rank: int, msg: str) -> None:
    """Pre-main combine diag: show if either internode trace or COMBINE_DETAIL is on."""
    ts = f"t={time.monotonic():.3f}s " if _flydsl_trace_ts_on() else ""
    if _flydsl_trace_on():
        print(f"[flydsl trace r={rank}] {ts}{msg}", flush=True)
    elif _flydsl_trace_combine_detail_on():
        print(f"[flydsl combine-detail r={rank}] {ts}{msg}", flush=True)


def _trace_flydsl(rank: int, msg: str) -> None:
    if not _flydsl_trace_on():
        return
    ts = f"t={time.monotonic():.3f}s " if _flydsl_trace_ts_on() else ""
    print(f"[flydsl trace r={rank}] {ts}{msg}", flush=True)


def _trace_p2p_self_row(rank: int, **named_tables) -> None:
    if not _flydsl_trace_on():
        return
    parts = [f"{n}[r]==0x{int(t[rank].item()):x}" for n, t in named_tables.items()]
    print(
        f"[flydsl trace r={rank}] P2P table self row (0 expected for some entries; kernels use local sym): "
        + " ".join(parts),
        flush=True,
    )


def _trace_flydsl_ordered(rank: int, world_size: int, msg: str) -> None:
    """Print one rank at a time so mixed logs show a clear global order."""
    if not _flydsl_trace_on():
        return
    ts = f"t={time.monotonic():.3f}s " if _flydsl_trace_ts_on() else ""
    if not dist.is_initialized() or world_size <= 1:
        print(f"[flydsl trace r={rank}] {ts}{msg}", flush=True)
        return
    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(f"[flydsl trace r={rank} ordered] {ts}{msg}", flush=True)
    dist.barrier()


def _trace_combine_pre_main_diag(
    rank: int,
    dev: torch.device,
    world_size: int,
    *,
    total_recv: torch.Tensor,
    node_recv: torch.Tensor,
    cross_dev_flag: torch.Tensor,
    inter_bar: torch.Tensor,
    chunk_flag_combine: torch.Tensor,
    inter_dest: torch.Tensor,
    disp_dest_map: torch.Tensor,
    staging: torch.Tensor,
    xdb_sym: torch.Tensor,
    p2p_ci: torch.Tensor,
) -> None:
    if not (_flydsl_trace_on() or _flydsl_trace_combine_detail_on()):
        return
    torch.cuda.synchronize(dev)
    tr = int(total_recv.detach().cpu().item())
    cdf = int(cross_dev_flag.detach().cpu().item())
    ib = inter_bar.detach().cpu().numpy().tolist()
    nr_u64 = node_recv.view(torch.uint64).detach().cpu().numpy().ravel()
    ide = inter_dest.detach().cpu().numpy().ravel()[: min(16, inter_dest.numel())]
    ddm = disp_dest_map.detach().cpu().numpy().ravel()[: min(16, disp_dest_map.numel())]
    cfc = chunk_flag_combine.detach().cpu().numpy().ravel()[
        : min(32, chunk_flag_combine.numel())
    ]
    xdb = xdb_sym.detach().cpu().numpy().ravel()
    pci = [int(p2p_ci[pe].item()) for pe in range(world_size)]
    _trace_internode_or_detail_line(
        rank,
        f"combine pre-main diag: total_recv={tr} cross_dev_flag={cdf} inter_bar={ib}",
    )
    _trace_internode_or_detail_line(rank, f"combine pre-main diag: node_recv_u64={nr_u64.tolist()}")
    _trace_internode_or_detail_line(rank, f"combine pre-main diag: inter_dest[:16]={ide.tolist()}")
    _trace_internode_or_detail_line(rank, f"combine pre-main diag: disp_dest_map[:16]={ddm.tolist()}")
    _trace_internode_or_detail_line(rank, f"combine pre-main diag: chunk_flag_combine[:32]={cfc.tolist()}")
    _trace_internode_or_detail_line(
        rank, f"combine pre-main diag: xdb_sym={xdb.tolist()} p2p_ci(hex)={[hex(x) for x in pci]}"
    )
    _trace_internode_or_detail_line(
        rank,
        f"combine pre-main diag: staging=0x{staging.data_ptr():x} combine_inp ptr via table[rank]=0x{pci[rank]:x}",
    )


def _trace_after_combine_main_ll(
    rank: int,
    dev: torch.device,
    *,
    inter_bar: torch.Tensor,
    chunk_flag: torch.Tensor,
    chunk_flag_combine: torch.Tensor,
) -> None:
    if not _flydsl_trace_combine_detail_on():
        return
    torch.cuda.synchronize(dev)
    ib = inter_bar.detach().cpu().numpy().ravel().tolist()
    cf_i64 = chunk_flag.view(torch.int64).detach().cpu().numpy().ravel()
    n_cf = min(16, cf_i64.size)
    cfc = chunk_flag_combine.detach().cpu().numpy().ravel()
    n_cfc = min(24, cfc.size)
    _trace_combine_detail(rank, f"post-main_ll: inter_bar={ib}")
    _trace_combine_detail(rank, f"post-main_ll: chunk_flag_i64[:{n_cf}]={cf_i64[:n_cf].tolist()}")
    _trace_combine_detail(rank, f"post-main_ll: chunk_flag_combine_i32[:{n_cfc}]={cfc[:n_cfc].tolist()}")


def _trace_after_combine_all(
    rank: int,
    dev: torch.device,
    *,
    combine_out_shmem: torch.Tensor,
    dt: torch.dtype,
    max_tok: int,
    hidden_dim: int,
    n_tok: int,
) -> None:
    if not _flydsl_trace_combine_detail_on():
        return
    torch.cuda.synchronize(dev)
    comb = combine_out_shmem.view(dt).view(max_tok, hidden_dim)[:n_tok].float()
    finite = bool(torch.isfinite(comb).all().item())
    _trace_combine_detail(
        rank,
        "post-combine_all: "
        f"shape=({n_tok},{hidden_dim}) finite_all={finite} "
        f"mean={comb.mean().item():.6g} std={comb.std().item():.6g} "
        f"min={comb.min().item():.6g} max={comb.max().item():.6g}",
    )
    h8 = min(8, hidden_dim)
    _trace_combine_detail(rank, f"post-combine_all: row0[:{h8}]={comb[0, :h8].tolist()}")
    if n_tok > 1:
        _trace_combine_detail(rank, f"post-combine_all: row1[:{h8}]={comb[1, :h8].tolist()}")


def _trace_mori_comb_snapshot(rank: int, comb_cpu: torch.Tensor, *, label: str = "mori golden comb") -> None:
    if not _flydsl_trace_combine_detail_on():
        return
    t = comb_cpu.float()
    _trace_combine_detail(
        rank,
        f"{label}: shape={tuple(t.shape)} mean={t.mean().item():.6g} std={t.std().item():.6g} "
        f"min={t.min().item():.6g} max={t.max().item():.6g}",
    )
    if t.dim() >= 2:
        h8 = min(8, t.shape[1])
        _trace_combine_detail(rank, f"{label}: row0[:{h8}]={t[0, :h8].tolist()}")
        if t.shape[0] > 1:
            _trace_combine_detail(rank, f"{label}: row1[:{h8}]={t[1, :h8].tolist()}")


def _print_combine_mismatch(rank: int, mori_cpu: torch.Tensor, fly_cpu: torch.Tensor) -> None:
    """Always print numeric hints when allclose fails (not gated on COMBINE_DETAIL)."""
    g = mori_cpu.float()
    f = fly_cpu.float()
    if g.shape != f.shape:
        print(
            f"[rank{rank}] combine mismatch: shape mori={tuple(g.shape)} fly={tuple(f.shape)}",
            flush=True,
        )
        return
    d = (g - f).abs()
    flat = d.reshape(-1)
    mx = flat.max().item()
    j = int(flat.argmax().item())
    hd = g.shape[1] if g.dim() == 2 else 1
    ti = j // hd if g.dim() == 2 else 0
    hi = j % hd if g.dim() == 2 else j
    print(
        f"[rank{rank}] combine mismatch: max_abs_diff={mx:.6g} at tok={ti} h={hi} "
        f"mori={g.reshape(-1)[j].item():.6g} fly={f.reshape(-1)[j].item():.6g}",
        flush=True,
    )
    if g.dim() == 2:
        h8 = min(8, hd)
        print(
            f"[rank{rank}] combine mismatch: row{ti} mori[:{h8}]={g[ti, :h8].tolist()}",
            flush=True,
        )
        print(
            f"[rank{rank}] combine mismatch: row{ti} fly[:{h8}]={f[ti, :h8].tolist()}",
            flush=True,
        )


def run_mori_reference(
    rank: int,
    world_size: int,
    dev: torch.device,
    *,
    hidden_dim: int,
    max_tok: int,
    k: int,
    epr: int,
    n_tok: int,
    block_num: int,
    rdma_block_num: int,
    wpb: int,
    include_combine: bool = False,
):
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    dt = torch.bfloat16
    # mori 官方 internode v1 单测对 bf16 使用 max_token_type_size=2（见 _make_internode_v1_config）
    mcfg = EpDispatchCombineConfig(
        data_type=dt,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=epr,
        num_experts_per_token=k,
        warp_num_per_block=wpb,
        block_num=block_num,
        gpu_per_node=1,
        kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=1,
        use_external_inp_buf=True,
        quant_type="none",
    )
    op = EpDispatchCombineOp(mcfg)
    torch.manual_seed(1234 + rank)
    inp = torch.randn(n_tok, hidden_dim, dtype=dt, device=dev)
    wts = torch.ones(n_tok, k, dtype=torch.float32, device=dev)
    # index_t 为 int32；与 mori dispatch_combine_test_utils 一致用 int32 路由下标
    # 强制跨 rank 路由: rank0 → expert1, rank1 → expert0
    if rank == 0:
        idx = torch.tensor([[1, 1]] * n_tok, dtype=torch.int32, device=dev)
    else:
        idx = torch.tensor([[0, 0]] * n_tok, dtype=torch.int32, device=dev)

    out_t, out_w, out_s, out_i, total = op.dispatch(inp, wts, None, idx)
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    nrecv = int(total.cpu().item())
    # mori EpCombine may update dispatch out buffers in place. Clone on device then H2D
    # so combine cannot race with the golden copy (views can alias shmem).
    out_tok_cpu = out_t[:nrecv].clone().cpu()
    out_wts_cpu = out_w[:nrecv].clone().cpu()
    out_idx_cpu = out_i[:nrecv].clone().cpu()
    torch.cuda.synchronize()
    comb_out_cpu = None
    if include_combine:
        comb_bf16, _comb_w = op.combine(out_t, out_w, out_i, call_reset=False)
        torch.cuda.synchronize(dev)
        ms.shmem_barrier_all()
        comb_out_cpu = comb_bf16[:n_tok].detach().cpu()
        _trace_mori_comb_snapshot(rank, comb_out_cpu, label="mori golden comb")
    # Symmetric teardown: ShmemFree / handle dtor must not race across PEs, or the peer's
    # next FlyDSL mori_shmem alloc + combine_main can fault (low VA) on ROCm.
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    if dist.is_initialized() and world_size > 1:
        dist.barrier()
    del op
    gc.collect()
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    if dist.is_initialized() and world_size > 1:
        dist.barrier()
    return {
        "out_tok": out_tok_cpu,
        "out_wts": out_wts_cpu,
        "out_idx": out_idx_cpu,
        "total_recv": nrecv,
        "inp": inp.detach().cpu(),
        "idx": idx.detach().cpu(),
        "wts": wts.detach().cpu(),
        "comb_out": comb_out_cpu,
    }


def run_mori_bench(
    rank: int,
    world_size: int,
    dev: torch.device,
    *,
    hidden_dim: int,
    max_tok: int,
    k: int,
    epr: int,
    n_tok: int,
    block_num: int,
    rdma_block_num: int,
    wpb: int,
    bench_warmup: int,
    bench_iters: int,
) -> float:
    """CUDA-event mean time (µs/iter) for ``EpDispatchCombineOp.dispatch`` on this rank's GPU.

    Matches ``run_mori_reference`` config: InterNodeV1LL = ``EpDispatchCopyToStaging`` +
    ``EpDispatchInterNodeV1KernelLowLatency`` (same launch params as FlyDSL copy+main).
    """
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    dt = torch.bfloat16
    mcfg = EpDispatchCombineConfig(
        data_type=dt,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=epr,
        num_experts_per_token=k,
        warp_num_per_block=wpb,
        block_num=block_num,
        gpu_per_node=1,
        kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=1,
        use_external_inp_buf=True,
        quant_type="none",
    )
    op = EpDispatchCombineOp(mcfg)
    torch.manual_seed(1234 + rank)
    inp = torch.randn(n_tok, hidden_dim, dtype=dt, device=dev)
    wts = torch.ones(n_tok, k, dtype=torch.float32, device=dev)
    if rank == 0:
        idx = torch.tensor([[1, 1]] * n_tok, dtype=torch.int32, device=dev)
    else:
        idx = torch.tensor([[0, 0]] * n_tok, dtype=torch.int32, device=dev)

    def _step():
        op.dispatch(
            inp,
            wts,
            None,
            idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=wpb,
        )

    _step()
    torch.cuda.synchronize(dev)
    dist.barrier()
    gpu_us = bench_gpu_us_torch(_step, warmup=bench_warmup, iters=bench_iters)
    torch.cuda.synchronize(dev)
    dist.barrier()
    del op
    gc.collect()
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    return gpu_us


def run_mori_bench_combine(
    rank: int,
    world_size: int,
    dev: torch.device,
    *,
    hidden_dim: int,
    max_tok: int,
    k: int,
    epr: int,
    n_tok: int,
    block_num: int,
    rdma_block_num: int,
    wpb: int,
    bench_warmup: int,
    bench_iters: int,
) -> float:
    """Mean µs/iter for ``EpDispatchCombineOp.combine`` only; each iter re-dispatches first (un-timed prep)."""
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    dt = torch.bfloat16
    mcfg = EpDispatchCombineConfig(
        data_type=dt,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=epr,
        num_experts_per_token=k,
        warp_num_per_block=wpb,
        block_num=block_num,
        gpu_per_node=1,
        kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=1,
        use_external_inp_buf=True,
        quant_type="none",
    )
    op = EpDispatchCombineOp(mcfg)
    torch.manual_seed(1234 + rank)
    inp = torch.randn(n_tok, hidden_dim, dtype=dt, device=dev)
    wts = torch.ones(n_tok, k, dtype=torch.float32, device=dev)
    if rank == 0:
        idx = torch.tensor([[1, 1]] * n_tok, dtype=torch.int32, device=dev)
    else:
        idx = torch.tensor([[0, 0]] * n_tok, dtype=torch.int32, device=dev)

    out_t = out_w = out_i = None

    def _prep():
        nonlocal out_t, out_w, out_i
        out_t, out_w, _s, out_i, _total = op.dispatch(
            inp,
            wts,
            None,
            idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=wpb,
        )
        torch.cuda.synchronize(dev)
        dist.barrier()
        ms.shmem_barrier_all()

    def _body():
        op.combine(out_t, out_w, out_i, call_reset=False)
        torch.cuda.synchronize(dev)

    _prep()
    _body()
    torch.cuda.synchronize(dev)
    dist.barrier()
    gpu_us = bench_gpu_us_prep_body(
        _prep, _body, warmup=bench_warmup, iters=bench_iters, device=dev
    )
    torch.cuda.synchronize(dev)
    dist.barrier()
    del op
    gc.collect()
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    return gpu_us


def run_mori_bench_e2e(
    rank: int,
    world_size: int,
    dev: torch.device,
    *,
    hidden_dim: int,
    max_tok: int,
    k: int,
    epr: int,
    n_tok: int,
    block_num: int,
    rdma_block_num: int,
    wpb: int,
    bench_warmup: int,
    bench_iters: int,
) -> float:
    """Mean µs/iter for mori ``dispatch`` + ``combine`` in one step (per-iter redispatch)."""
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    dt = torch.bfloat16
    mcfg = EpDispatchCombineConfig(
        data_type=dt,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=epr,
        num_experts_per_token=k,
        warp_num_per_block=wpb,
        block_num=block_num,
        gpu_per_node=1,
        kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=1,
        use_external_inp_buf=True,
        quant_type="none",
    )
    op = EpDispatchCombineOp(mcfg)
    torch.manual_seed(1234 + rank)
    inp = torch.randn(n_tok, hidden_dim, dtype=dt, device=dev)
    wts = torch.ones(n_tok, k, dtype=torch.float32, device=dev)
    if rank == 0:
        idx = torch.tensor([[1, 1]] * n_tok, dtype=torch.int32, device=dev)
    else:
        idx = torch.tensor([[0, 0]] * n_tok, dtype=torch.int32, device=dev)

    def _step():
        out_t, out_w, _s, out_i, _total = op.dispatch(
            inp,
            wts,
            None,
            idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=wpb,
        )
        op.combine(out_t, out_w, out_i, call_reset=False)
        torch.cuda.synchronize(dev)

    _step()
    torch.cuda.synchronize(dev)
    dist.barrier()
    gpu_us = bench_gpu_us_torch(_step, warmup=bench_warmup, iters=bench_iters)
    torch.cuda.synchronize(dev)
    dist.barrier()
    del op
    gc.collect()
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    return gpu_us


def _zero_tensor(t: torch.Tensor):
    t.zero_()


def _foreach_zero_cuda(tensors: list[torch.Tensor]) -> None:
    """Zero GPU tensors used in internode v1ll setup / reset.

    Default: one ``tensor.zero_()`` per buffer. On ROCm, ``torch._foreach_zero_`` on mori
    symmetric allocations has been observed to fault immediately after JIT init (illegal
    addresses including ``0x3f800000``).

    Opt-in batch path (fewer fill launches): ``export FLYDSL_FOREACH_ZERO_BATCH=1``.
    """
    if os.environ.get("FLYDSL_FOREACH_ZERO_BATCH", "0").lower() not in ("1", "true", "yes"):
        for t in tensors:
            if t.numel():
                t.zero_()
        return
    groups: dict[tuple, list[torch.Tensor]] = defaultdict(list)
    for t in tensors:
        if t.numel() == 0:
            continue
        groups[(t.device, t.dtype)].append(t)
    for ts in groups.values():
        if ts[0].is_cuda:
            try:
                torch._foreach_zero_(ts)
            except (NotImplementedError, RuntimeError):
                for x in ts:
                    x.zero_()
        else:
            for x in ts:
                x.zero_()


def _effective_copy_grid_blocks(mp_count: int, n_tok: int) -> int:
    """rocprof: copy_jit is negligible vs main; still avoid 80 blocks for n_tok=2 (launch + idle warps)."""
    return min(mp_count, max(1, int(n_tok)))


def _perm_sort_dispatch_rows(tok: torch.Tensor, wts: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Permutation that lex-sorts rows so FlyDSL vs mori can be compared row-wise (order may differ)."""
    n = idx.shape[0]
    keys = []
    for i in range(n):
        keys.append(
            (
                tuple(int(x) for x in idx[i].tolist()),
                tuple(float(x) for x in wts[i].tolist()),
                tuple(float(x) for x in tok[i].flatten().tolist()),
            )
        )
    order = sorted(range(n), key=lambda j: keys[j])
    return torch.tensor(order, dtype=torch.long, device=tok.device)


def _aligned_dispatch_tensors(golden: dict, fly: dict, k: int):
    """Return permuted (g_tok, f_tok, g_wts, f_wts, g_idx, f_idx) for allclose / idx check."""
    n = golden["out_idx"].shape[0]
    g_idx = golden["out_idx"].view(n, k)
    f_idx = fly["out_idx"].view(n, k)
    g_wts = golden["out_wts"].view(n, k)
    f_wts = fly["out_wts"].view(n, k)
    g_tok = golden["out_tok"].view(n, -1)
    f_tok = fly["out_tok"].view(n, -1)
    pg = _perm_sort_dispatch_rows(g_tok, g_wts, g_idx)
    pf = _perm_sort_dispatch_rows(f_tok, f_wts, f_idx)
    return (
        golden["out_tok"][pg],
        fly["out_tok"][pf],
        golden["out_wts"][pg],
        fly["out_wts"][pf],
        g_idx[pg],
        f_idx[pf],
    )


def run_flydsl_dispatch(
    rank: int,
    world_size: int,
    dev: torch.device,
    *,
    hidden_dim: int,
    max_tok: int,
    k: int,
    epr: int,
    n_tok: int,
    block_num: int,
    rdma_block_num: int,
    wpb: int,
    mp_count: int,
    num_qp_per_pe: int = 1,
    bench: bool = False,
    bench_warmup: int = 20,
    bench_iters: int = 200,
    bench_scope: str = "dispatch",
    enable_std_moe: bool = False,
    run_combine: bool = False,
    force_split_dispatch: bool = False,  # True → separate copy/main jits (see SPLIT_DISPATCH env)
):
    _trace_flydsl(rank, f"run_flydsl_dispatch start dev={dev} combine={run_combine}")
    install_jit_hook()
    _trace_flydsl(rank, "install_jit_hook() done")
    if rank == 0:
        _print_dispatch_debug_banner(0, world_size)
    if dist.is_initialized() and world_size > 1:
        dist.barrier()
    if _flydsl_trace_on() and rank == 0:
        hb = os.environ.get("HIP_LAUNCH_BLOCKING", "")
        print(
            "[flydsl trace r=0] tips: ordered lines use dist.barrier; for exact faulting kernel on ROCm "
            "set FLYDSL_TRACE_SYNC_LAUNCH=1 (before python) to enable HIP_LAUNCH_BLOCKING; "
            f"now HIP_LAUNCH_BLOCKING={hb!r}",
            flush=True,
        )
    if rank == 0:
        print(
            "[rank0] FlyDSL: compiling internode v1ll jits "
            "(cold compile often 1–3 min — not a hang)...",
            flush=True,
        )
    if run_combine and enable_std_moe:
        raise RuntimeError("FlyDSL internode v1ll combine is not wired with --std-moe")
    if bench and bench_scope in ("combine", "e2e") and not run_combine:
        raise RuntimeError("bench_scope combine/e2e requires run_combine=True")
    dt = torch.bfloat16
    elem = torch.tensor([], dtype=dt).element_size()
    n_nodes = world_size  # gpu_per_node=1
    gpu_per_node = 1
    xb = _xfer_bytes(hidden_dim, elem, k, 0)
    max_toks_send = world_size * max_tok
    max_recv = world_size * max_tok
    # Must not equal any valid route flat = dest_pe * max_toks_send + slot (slot < max_tok).
    # null=max_toks_send collides with PE1 slot0 when max_toks_send=16 (2*8).
    null_flat = (world_size - 1) * max_toks_send + max_tok
    max_chunk_num = (max_tok + 63) // 64

    # --- shmem buffers (sizes from mori dispatch_combine.cpp InterNodeV1) ---
    torch.cuda.set_device(dev.index)
    _trace_flydsl(rank, "alloc dispatch_inp / staging / shmem outputs …")
    dispatch_inp = mori_shmem_create_tensor((n_nodes * max_tok * xb,), torch.uint8)
    staging = mori_shmem_create_tensor((2 * n_nodes * max_tok * xb,), torch.uint8)
    tok_i16 = (max_recv * hidden_dim * elem + 1) // 2
    shmem_out_tok = mori_shmem_create_tensor((tok_i16,), torch.int16)
    shmem_out_idx = mori_shmem_create_tensor((max_recv * k,), torch.int32)
    shmem_out_wts = mori_shmem_create_tensor((max_recv * k,), torch.float32)
    chunk_flag = mori_shmem_create_tensor((n_nodes * max_tok * 8,), torch.uint8)
    node_recv = mori_shmem_create_tensor((n_nodes * 8,), torch.uint8)
    # EpDispatchCombineHandle::InitializeTokenNumSignalBuf:
    #   tokenNumSignalSize = worldSize * sizeof(index_t) * 2 * numQpPerPe
    recv_signal_elems = world_size * 2 * num_qp_per_pe
    recv_tok_num = mori_shmem_create_tensor((recv_signal_elems,), torch.int32)
    shmem_tok_off = mori_shmem_create_tensor((1,), torch.int32)
    max_out_tok = max_toks_send * epr
    shmem_tis = mori_shmem_create_tensor((max_out_tok,), torch.int32)

    # device-local
    block_flag = torch.zeros(n_nodes, dtype=torch.int32, device=dev)
    inter_bar = torch.zeros(4, dtype=torch.int32, device=dev)
    inter_dest = torch.zeros(n_nodes * max_tok * k, dtype=torch.int32, device=dev)
    inter_send = torch.zeros(n_nodes * max_tok, dtype=torch.int32, device=dev)
    disp_dest_map = torch.zeros(max_toks_send * k, dtype=torch.int32, device=dev)
    dest_pe_ctr = torch.zeros(world_size, dtype=torch.int32, device=dev)
    total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
    disp_grid_bar = torch.zeros(world_size, dtype=torch.int32, device=dev)
    comb_grid_bar = torch.zeros(world_size, dtype=torch.int32, device=dev)
    cross_dev_flag = torch.zeros(1, dtype=torch.int64, device=dev)

    max_tpe = world_size * max_tok
    if enable_std_moe:
        packed_recv_x = torch.zeros((epr * max_tpe, hidden_dim), dtype=dt, device=dev)
        packed_recv_count = torch.zeros(epr, dtype=torch.int32, device=dev)
        packed_recv_src_info = torch.zeros(epr * max_tpe, dtype=torch.int32, device=dev)
        disp_tok_map = torch.zeros(max_recv * k, dtype=torch.int64, device=dev)
    else:
        packed_recv_x = packed_recv_count = packed_recv_src_info = disp_tok_map = None

    # P2P pointer tables (i64[npes])
    def p2p_table(get_ptr_callable, npes: int) -> torch.Tensor:
        t = torch.zeros(npes, dtype=torch.int64, device=dev)
        for pe in range(npes):
            t[pe] = get_ptr_callable(pe)
        return t

    r = rank
    npes = world_size
    p2p_tok_off = p2p_table(lambda pe: ms.shmem_ptr_p2p(shmem_tok_off.data_ptr(), r, pe), npes)
    p2p_tis = p2p_table(lambda pe: ms.shmem_ptr_p2p(shmem_tis.data_ptr(), r, pe), npes)
    p2p_out_tok = p2p_table(lambda pe: ms.shmem_ptr_p2p(shmem_out_tok.data_ptr(), r, pe), npes)
    p2p_out_idx = p2p_table(lambda pe: ms.shmem_ptr_p2p(shmem_out_idx.data_ptr(), r, pe), npes)
    p2p_out_wts = p2p_table(lambda pe: ms.shmem_ptr_p2p(shmem_out_wts.data_ptr(), r, pe), npes)
    p2p_recv = p2p_table(lambda pe: ms.shmem_ptr_p2p(recv_tok_num.data_ptr(), r, pe), npes)

    combine_inp = combine_out_shmem = chunk_flag_combine = xdb_sym = None
    p2p_ci = p2p_xdb = None
    if run_combine:
        combine_inp = mori_shmem_create_tensor((max_recv * hidden_dim * elem,), torch.uint8)
        combine_out_shmem = mori_shmem_create_tensor((max_tok * hidden_dim * elem,), torch.uint8)
        chunk_flag_combine = torch.zeros(n_nodes * max_chunk_num, dtype=torch.int32, device=dev)
        # mori InitializeBarrier: crossDeviceBarrierMemObj = barrierSize*2*sizeof(uint64),
        # barrierSize = worldSize * sizeof(uint32_t) → 8*worldSize uint64 elements (bytes / 8).
        _xdb_u64 = world_size * 8
        xdb_sym = mori_shmem_create_tensor((_xdb_u64,), torch.uint64)
        p2p_ci = p2p_table(lambda pe: ms.shmem_ptr_p2p(combine_inp.data_ptr(), r, pe), npes)
        p2p_xdb = p2p_table(lambda pe: ms.shmem_ptr_p2p(xdb_sym.data_ptr(), r, pe), npes)
        # mori shmem_ptr_p2p(sym, rank, rank) can be 0; combine main LL loads pe row — need local VA for self
        p2p_ci[r] = combine_inp.data_ptr()
        p2p_xdb[r] = xdb_sym.data_ptr()

    _trace_p2p_self_row(
        rank,
        tok_off=p2p_tok_off,
        tis=p2p_tis,
        out_tok=p2p_out_tok,
        out_idx=p2p_out_idx,
        out_wts=p2p_out_wts,
        recv=p2p_recv,
        **({"xdb": p2p_xdb, "comb_inp": p2p_ci} if run_combine and p2p_xdb is not None else {}),
    )
    _trace_flydsl(rank, "symmetric buffers + P2P tables built, entering shmem_barrier_all (pre-audit)")

    ms.shmem_barrier_all()
    torch.cuda.synchronize(dev)
    _trace_flydsl(rank, "post shmem_barrier + cuda.sync (pre p2p peer audit)")

    # Raw buffer paths (XGMI scatter, DispatchSync remote stores) use shmem_ptr_p2p VAs.
    # mori addr-based putmem/uint64_atomic expect local symmetric VAs (see kernel module doc).
    # shmem_ptr_p2p may return 0 for non-P2P transports; audited peer entries must be non-zero.
    _p2p_audit_tables = {
        "tok_off": p2p_tok_off,
        "tis": p2p_tis,
        "out_tok": p2p_out_tok,
        "out_idx": p2p_out_idx,
        "out_wts": p2p_out_wts,
        "recv_tok_num": p2p_recv,
    }
    for pe in range(npes):
        if pe == r:
            continue
        parts = []
        for name, tbl in _p2p_audit_tables.items():
            v = int(tbl[pe].item())
            parts.append(f"{name}=0x{v:x}")
            if v == 0:
                raise RuntimeError(
                    f"[internode_v1ll_dispatch_flydsl] rank={r}: shmem_ptr_p2p(..., dest_pe={pe}) "
                    f"returned 0 for table {name!r}. Peer P2P is required for raw remote buffer paths; "
                    f"see mori.shmem.shmem_ptr_p2p docs (0 => RDMA path / invalid for device direct)."
                )
        print(f"[p2p audit] rank={r} -> pe={pe}: " + " ".join(parts), flush=True)

    if run_combine:
        for pe in range(npes):
            if pe == r:
                continue
            for name, tbl in (("combine_inp", p2p_ci), ("xdb", p2p_xdb)):
                v = int(tbl[pe].item())
                if v == 0:
                    raise RuntimeError(
                        f"[internode_v1ll_dispatch_flydsl] combine P2P rank={r} pe={pe} {name}=0"
                    )

    copy_grid_blocks = _effective_copy_grid_blocks(mp_count, n_tok)
    _split_dispatch_jit = (
        force_split_dispatch
        or os.environ.get("FLYDSL_INTERNODE_V1LL_SPLIT_DISPATCH", "0").lower()
        in ("1", "true", "yes")
    )
    copy_jit = main_jit = None
    dispatch_fused_jit = None
    if _split_dispatch_jit:
        _trace_flydsl(rank, "p2p peer audit passed, building copy_jit + main_jit (split) …")
        copy_jit = make_copy_staging_jit(
            rank=rank,
            npes=npes,
            experts_per_token=k,
            hidden_dim=hidden_dim,
            max_tok_per_rank=max_tok,
            copy_grid_blocks=copy_grid_blocks,
            warp_num_per_block=wpb,
            data_type=dt,
        )
        _trace_flydsl(rank, "make_copy_staging_jit returned")
        main_jit = make_dispatch_internode_v1ll_main_jit(
            rank=rank,
            npes=npes,
            gpu_per_node=gpu_per_node,
            experts_per_rank=epr,
            experts_per_token=k,
            hidden_dim=hidden_dim,
            max_tok_per_rank=max_tok,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_num_per_block=wpb,
            num_qp_per_pe=1,
            enable_std_moe=enable_std_moe,
            data_type=dt,
        )
        _trace_flydsl(rank, "make_dispatch_internode_v1ll_main_jit returned")
    else:
        _trace_flydsl(
            rank,
            "p2p peer audit passed, building fused copy+main jit (may compile) …",
        )
        dispatch_fused_jit = make_dispatch_internode_v1ll_copy_main_fused_jit(
            rank=rank,
            npes=npes,
            gpu_per_node=gpu_per_node,
            experts_per_rank=epr,
            experts_per_token=k,
            hidden_dim=hidden_dim,
            max_tok_per_rank=max_tok,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            copy_grid_blocks=copy_grid_blocks,
            warp_num_per_block=wpb,
            num_qp_per_pe=1,
            enable_std_moe=enable_std_moe,
            data_type=dt,
        )
        _trace_flydsl(rank, "make_dispatch_internode_v1ll_copy_main_fused_jit returned")

    combine_sync_jit = combine_bar_jit = combine_main_jit = combine_all_jit = None
    if run_combine:
        combine_sync_jit = make_combine_internode_v1ll_sync_jit(
            mp_count=mp_count,
            warp_num_per_block=wpb,
            hidden_dim=hidden_dim,
            data_type=dt,
        )
        combine_bar_jit = make_combine_internode_v1ll_sync_barrier_jit(
            rank=rank, gpu_per_node=gpu_per_node
        )
        combine_main_jit = make_combine_internode_v1ll_main_ll_jit(
            rank=rank,
            npes=npes,
            gpu_per_node=gpu_per_node,
            experts_per_token=k,
            hidden_dim=hidden_dim,
            max_tok_per_rank=max_tok,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_num_per_block=wpb,
            num_qp_per_pe=num_qp_per_pe,
            staging_slot_bytes=xb,
            data_type=dt,
        )
        combine_all_jit = make_combine_internode_v1ll_all_jit(
            rank=rank,
            npes=npes,
            gpu_per_node=gpu_per_node,
            experts_per_token=k,
            hidden_dim=hidden_dim,
            max_tok_per_rank=max_tok,
            mp_count=mp_count,
            warp_num_per_block=wpb,
            staging_slot_bytes=xb,
            data_type=dt,
        )
        _trace_flydsl(rank, "combine jits (sync / barrier / main / all) built")

    # Cold JIT can finish minutes apart per rank. Letting the fast rank run randn / shmem zero_
    # while the slow rank's compile path still touches the GPU has produced immediate access faults
    # on ROCm (symptom: fault right after "jits ready", no copy_jit line).
    # Python may return from make_*_jit before all device-side compile work is retired; dist.barrier
    # only aligns CPUs. Flush allocator state + repeat sync so each rank's HIP queue is idle before
    # the cross-rank barrier.
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(3):
        torch.cuda.synchronize(dev)
    _trace_pin(rank, "P001", "post-jit gc/empty_cache/3x cuda.sync done")
    _trace_flydsl(
        rank,
        "post-jit: cuda.sync + dist/shmem barriers (align ranks before randn and buffer zero)",
    )
    torch.cuda.synchronize(dev)
    dist.barrier()
    ms.shmem_barrier_all()
    torch.cuda.synchronize(dev)
    dist.barrier()
    torch.cuda.synchronize(dev)
    _trace_pin(rank, "P002", "post-jit dist/shmem/barrier round done (before jits ready print)")

    if rank == 0:
        print(
            "[rank0] FlyDSL: jits ready, executing "
            + ("dispatch+combine" if run_combine else "dispatch")
            + "...",
            flush=True,
        )
    # ROCm (e.g. CUDA_VISIBLE_DEVICES=6,7): rank0-only print is slow; without a barrier, rank1
    # runs ahead into warmup/randn/shmem zero while rank0 has not reached P003 → dual-GPU fault
    # right after this line (e.g. VA 0x11000000000) with no P001/P004 pins if pinpoint is off.
    dist.barrier()
    torch.cuda.synchronize(dev)
    ms.shmem_barrier_all()
    _trace_pin(rank, "P003", "past FlyDSL jits ready banner (all ranks)")

    def _milestone(msg: str) -> None:
        if os.environ.get("FLYDSL_TRACE_MILESTONE", "0") not in ("0", "false", "False"):
            print(f"[rank{rank}] milestone: {msg}", flush=True)

    # Tiny alloc + sync primes the allocator after cold JIT. Parallel warmup on both ranks has
    # still faulted on ROCm right after "jits ready"; serialize like randn when world_size==2.
    _par_warm = os.environ.get("FLYDSL_PARALLEL_POST_JIT_WARMUP", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if _par_warm or world_size != 2:
        _warm = torch.zeros(512, dtype=torch.float32, device=dev)
        torch.cuda.synchronize(dev)
        del _warm
    else:
        for _wr in range(world_size):
            if rank == _wr:
                _warm = torch.zeros(512, dtype=torch.float32, device=dev)
                torch.cuda.synchronize(dev)
                del _warm
            dist.barrier()
    _trace_flydsl(rank, "post-jit: warmup (512 fp32) done")
    _trace_pin(rank, "P004", "post-jit warmup tensor done")

    scales_ptr = fx.Int64(0)
    par_first = os.environ.get("FLYDSL_PARALLEL_POST_JIT_ALLOC", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if par_first or world_size != 2:
        torch.manual_seed(1234 + rank)
        inp = torch.randn(n_tok, hidden_dim, dtype=dt, device=dev)
        wts = torch.ones(n_tok, k, dtype=torch.float32, device=dev)
        if rank == 0:
            idx = torch.tensor([[1, 1]] * n_tok, dtype=torch.int32, device=dev)
        else:
            idx = torch.tensor([[0, 0]] * n_tok, dtype=torch.int32, device=dev)
    else:
        # Default: one rank allocates inp/idx/wts at a time. Parallel first randn right after JIT
        # has produced dual-GPU Memory access faults on ROCm (no copy_jit line yet).
        inp = wts = idx = None  # type: ignore[assignment]
        for _alloc_rank in range(world_size):
            if rank == _alloc_rank:
                torch.manual_seed(1234 + rank)
                inp = torch.randn(n_tok, hidden_dim, dtype=dt, device=dev)
                wts = torch.ones(n_tok, k, dtype=torch.float32, device=dev)
                if rank == 0:
                    idx = torch.tensor([[1, 1]] * n_tok, dtype=torch.int32, device=dev)
                else:
                    idx = torch.tensor([[0, 0]] * n_tok, dtype=torch.int32, device=dev)
                torch.cuda.synchronize(dev)
            dist.barrier()
    _milestone("after randn/idx/wts on device")
    _trace_flydsl(rank, "post-jit: randn/idx/wts done")
    _trace_pin(rank, "P005", "randn/idx/wts tensors ready")

    # Both ranks must reach here before batching zeros into symmetric shmem; otherwise ROCm can
    # fault on early parallel touches (seen right after "jits ready" with no copy_jit line yet).
    dist.barrier()
    ms.shmem_barrier_all()
    torch.cuda.synchronize(dev)
    _trace_pin(rank, "P006", "barrier+shmem_barrier before dispatch-path mori zero")

    # reset state. Parallel zero_ on huge mori shmem from both ranks has faulted post-JIT on ROCm;
    # optional: serialize per-rank clears (each rank clears only its local symmetric view).
    _dispatch_sym_tensors = [
        dispatch_inp,
        staging,
        shmem_out_tok,
        shmem_out_idx,
        shmem_out_wts,
        chunk_flag,
        node_recv,
        recv_tok_num,
        shmem_tok_off,
        shmem_tis,
    ]
    _par_shmem_zero = os.environ.get("FLYDSL_PARALLEL_SHMEM_ZERO", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    _trace_flydsl(rank, "post-jit: zero dispatch-path mori shmem (dispatch_inp…shmem_tis) …")
    _trace_pin(rank, "P007", "about to zero dispatch-path mori shmem")
    if not _par_shmem_zero and world_size == 2:
        for _zr in range(world_size):
            if rank == _zr:
                _foreach_zero_cuda(_dispatch_sym_tensors)
                torch.cuda.synchronize(dev)
            dist.barrier()
            ms.shmem_barrier_all()
    else:
        _foreach_zero_cuda(_dispatch_sym_tensors)
    _trace_flydsl(rank, "post-jit: dispatch-path mori shmem zero done")
    _trace_pin(rank, "P008", "dispatch-path mori shmem zero done")
    block_flag.zero_()
    inter_bar.zero_()
    inter_dest.zero_()
    inter_send.zero_()
    disp_dest_map.fill_(null_flat)
    dest_pe_ctr.zero_()
    total_recv.zero_()
    disp_grid_bar.zero_()
    comb_grid_bar.zero_()
    cross_dev_flag.zero_()
    if enable_std_moe:
        packed_recv_x.zero_()
        packed_recv_count.zero_()
        packed_recv_src_info.zero_()
        disp_tok_map.zero_()
    if run_combine:
        # Parallel zero_ on large mori symmetric buffers has faulted (VA …e03000); clear each
        # buffer with barriers between ranks, then chunk_flag (device malloc, not shmem).
        for _t in (combine_inp, combine_out_shmem, xdb_sym):
            if _t is not None and _t.numel():
                _t.zero_()
                torch.cuda.synchronize(dev)
                dist.barrier()
                ms.shmem_barrier_all()
        chunk_flag_combine.zero_()
        torch.cuda.synchronize(dev)

    _trace_pin(
        rank,
        "P009",
        (
            "combine shmem + chunk_flag_combine zero done"
            if run_combine
            else "combine path skipped (no-combine mode)"
        ),
    )
    _milestone("after buffer reset (zeros); next: copy_jit")
    _trace_pin(rank, "P010", "all buffer reset done → about to copy_jit / dispatch")
    stream = torch.cuda.current_stream()
    cur_tok = n_tok

    # Dispatch bench: default matches mori ``dispatch()`` — timed loop is copy+main only; one
    # reset primes state before warmup (not amortized into per-iter µs). When
    # FLYDSL_BENCH_DISPATCH_INCLUDE_RESET=1, timed body is full _one_step_dispatch (heavy Python
    # reset every iter, for A/B vs old scripts).
    _bench_dispatch_kernel_only = (
        bench
        and bench_scope == "dispatch"
        and not run_combine
        and os.environ.get("FLYDSL_BENCH_DISPATCH_INCLUDE_RESET", "0").lower()
        not in ("1", "true", "yes")
    )
    # Bench kernel-only: P019 on first _launch_copy_main after reset, then skip (mori does not host-fence
    # between back-to-back dispatches). Full test uses FlyDSL bench *before* mori bench (see main()).
    _dispatch_launch_count = [0]
    _host_dispatch_seq = [0]
    _fast_parallel_dispatch_bench_reset = (
        bench
        and bench_scope == "dispatch"
        and not run_combine
        and os.environ.get("FLYDSL_BENCH_DISPATCH_CONSERVATIVE_RESET", "0").lower()
        not in ("1", "true", "yes")
        and os.environ.get("FLYDSL_BENCH_DISPATCH_INCLUDE_RESET", "0").lower()
        not in ("1", "true", "yes")
    )

    def _reset_state():
        # After a full buffer reset, the next dispatch must see P019 again (``INCLUDE_RESET`` bench).
        _dispatch_launch_count[0] = 0
        _rs_dispatch = [
            dispatch_inp,
            staging,
            shmem_out_tok,
            shmem_out_idx,
            shmem_out_wts,
            chunk_flag,
            node_recv,
            recv_tok_num,
            shmem_tok_off,
            shmem_tis,
        ]
        if _fast_parallel_dispatch_bench_reset:
            torch.cuda.synchronize(dev)
            dist.barrier()
            ms.shmem_barrier_all()
            _foreach_zero_cuda(_rs_dispatch)
            block_flag.zero_()
            inter_bar.zero_()
            inter_dest.zero_()
            inter_send.zero_()
            disp_dest_map.fill_(null_flat)
            dest_pe_ctr.zero_()
            total_recv.zero_()
            disp_grid_bar.zero_()
            comb_grid_bar.zero_()
            cross_dev_flag.zero_()
            if enable_std_moe:
                packed_recv_x.zero_()
                packed_recv_count.zero_()
                packed_recv_src_info.zero_()
                disp_tok_map.zero_()
            torch.cuda.synchronize(dev)
            dist.barrier()
            ms.shmem_barrier_all()
            return

        dist.barrier()
        ms.shmem_barrier_all()
        torch.cuda.synchronize(dev)
        _rs_par = os.environ.get("FLYDSL_PARALLEL_SHMEM_ZERO", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if not _rs_par and world_size == 2:
            for _zr in range(world_size):
                if rank == _zr:
                    _foreach_zero_cuda(_rs_dispatch)
                    torch.cuda.synchronize(dev)
                dist.barrier()
                ms.shmem_barrier_all()
        else:
            _foreach_zero_cuda(_rs_dispatch)
        block_flag.zero_()
        inter_bar.zero_()
        inter_dest.zero_()
        inter_send.zero_()
        disp_dest_map.fill_(null_flat)
        dest_pe_ctr.zero_()
        total_recv.zero_()
        disp_grid_bar.zero_()
        comb_grid_bar.zero_()
        cross_dev_flag.zero_()
        if enable_std_moe:
            packed_recv_x.zero_()
            packed_recv_count.zero_()
            packed_recv_src_info.zero_()
            disp_tok_map.zero_()
        if run_combine:
            for _t in (combine_inp, combine_out_shmem, xdb_sym):
                if _t is not None and _t.numel():
                    _t.zero_()
                    torch.cuda.synchronize(dev)
                    dist.barrier()
                    ms.shmem_barrier_all()
            chunk_flag_combine.zero_()
            torch.cuda.synchronize(dev)

    def _launch_copy_main():
        # ROCm: fault at ~copy_jit launch (after P010) when prior async work / shmem visibility
        # is not ordered across ranks (seen VA e.g. 0x8800000000).
        _need_pre_copy_fence = True
        _bench_n_sfx = ""
        _bench_every_pre_fence = True
        if _bench_dispatch_kernel_only:
            _dispatch_launch_count[0] += 1
            _bench_n_sfx = f" [bench n={_dispatch_launch_count[0]}]"
            # Mori timed ``dispatch()`` has no host fence *before* each launch; optional P019-off mode
            # adds P025 *after* each round so torchrun peers cannot overlap internode rounds (see docstring).
            _bench_every_pre_fence = os.environ.get(
                "FLYDSL_BENCH_DISPATCH_HOST_FENCE_EVERY_ITER", "1"
            ).lower() not in ("0", "false", "no")
            if not _bench_every_pre_fence:
                _need_pre_copy_fence = _dispatch_launch_count[0] <= 1
        else:
            _host_dispatch_seq[0] += 1
        _pin_n = (
            _dispatch_launch_count[0] if _bench_dispatch_kernel_only else None
        )
        _dev_seq = (
            _dispatch_launch_count[0]
            if _bench_dispatch_kernel_only
            else _host_dispatch_seq[0]
        )
        _dev_tr = _device_trace_dispatch_level(rank=rank)

        def _mori_like_bench_post_round_fence() -> None:
            if not _bench_dispatch_kernel_only or npes <= 1:
                return
            if _bench_every_pre_fence:
                return
            dist.barrier()
            ms.shmem_barrier_all()
            _trace_pin(
                rank,
                "P025",
                "post-round dist+shmem_barrier (mori-like: no cross-rank round overlap)"
                + _bench_n_sfx,
                bench_iter=_pin_n,
            )

        if _need_pre_copy_fence:
            torch.cuda.synchronize(dev)
            dist.barrier()
            ms.shmem_barrier_all()
            _trace_pin(
                rank,
                "P019",
                "pre-copy_jit cuda.sync + dist + shmem_barrier fence" + _bench_n_sfx,
                bench_iter=_pin_n,
            )
        else:
            _trace_pin(
                rank,
                "P019",
                "skipped (no host fence this launch; mori-like bench)"
                + _bench_n_sfx,
                bench_iter=_pin_n,
            )
        _trace_pin(
            rank,
            "P020",
            "_launch_copy_main entered" + _bench_n_sfx,
            bench_iter=_pin_n,
        )
        _z = fx.Int64(0)
        if enable_std_moe:
            std_moe_args = (
                fx.Int64(shmem_out_tok.data_ptr()),
                fx.Int64(shmem_out_idx.data_ptr()),
                fx.Int64(shmem_tis.data_ptr()),
                fx.Int64(packed_recv_x.data_ptr()),
                fx.Int64(packed_recv_count.data_ptr()),
                fx.Int64(packed_recv_src_info.data_ptr()),
                fx.Int64(disp_tok_map.data_ptr()),
            )
        else:
            # Local symmetric bases for dest_pe==rank (P2P self entries may be 0); Phase 4 is off.
            std_moe_args = (
                fx.Int64(shmem_out_tok.data_ptr()),
                fx.Int64(shmem_out_idx.data_ptr()),
                fx.Int64(shmem_tis.data_ptr()),
                _z,
                _z,
                _z,
                _z,
            )

        if dispatch_fused_jit is not None:
            _trace_flydsl_ordered(
                rank, npes, "dispatch: >>> fused_jit (copy_to_staging + internode_v1ll_main)"
            )
            _trace_pin(
                rank,
                "P021",
                "immediately before fused copy+main jit.launch" + _bench_n_sfx,
                bench_iter=_pin_n,
            )
            _trace_flydsl(
                rank,
                "dispatch: fused ptrs "
                f"inp=0x{inp.data_ptr():x} idx=0x{idx.data_ptr():x} wts=0x{wts.data_ptr():x} "
                f"staging=0x{staging.data_ptr():x} dispatch_inp=0x{dispatch_inp.data_ptr():x} "
                f"chunk_flag=0x{chunk_flag.data_ptr():x} node_recv=0x{node_recv.data_ptr():x} "
                f"cur_tok={int(cur_tok)}",
            )
            dispatch_fused_jit(
                fx.Int64(inp.data_ptr()),
                fx.Int64(idx.data_ptr()),
                fx.Int64(wts.data_ptr()),
                scales_ptr,
                fx.Int64(staging.data_ptr()),
                fx.Int64(dispatch_inp.data_ptr()),
                fx.Int64(chunk_flag.data_ptr()),
                fx.Int64(node_recv.data_ptr()),
                fx.Int64(block_flag.data_ptr()),
                fx.Int64(inter_bar.data_ptr()),
                fx.Int64(inter_dest.data_ptr()),
                fx.Int64(inter_send.data_ptr()),
                fx.Int64(disp_dest_map.data_ptr()),
                fx.Int64(dest_pe_ctr.data_ptr()),
                fx.Int64(recv_tok_num.data_ptr()),
                fx.Int64(p2p_recv.data_ptr()),
                fx.Int64(total_recv.data_ptr()),
                fx.Int64(disp_grid_bar.data_ptr()),
                fx.Int64(comb_grid_bar.data_ptr()),
                fx.Int64(cross_dev_flag.data_ptr()),
                fx.Int64(shmem_tok_off.data_ptr()),
                fx.Int64(p2p_tok_off.data_ptr()),
                fx.Int64(p2p_tis.data_ptr()),
                fx.Int64(p2p_out_tok.data_ptr()),
                fx.Int64(p2p_out_idx.data_ptr()),
                fx.Int64(p2p_out_wts.data_ptr()),
                fx.Int64(0),
                *std_moe_args,
                fx.Int32(cur_tok),
                fx.Int32(_dev_tr),
                fx.Int32(_dev_seq),
                stream=stream,
            )
            torch.cuda.synchronize(dev)
            _trace_pin(
                rank,
                "P024",
                "fused copy+main + cuda.sync done" + _bench_n_sfx,
                bench_iter=_pin_n,
            )
            _trace_dispatch_postsync_readback(
                rank,
                dev,
                "post-fused-dispatch",
                bench_iter=_pin_n,
                total_recv=total_recv,
                cross_dev_flag=cross_dev_flag,
                recv_tok_num=recv_tok_num,
                disp_grid_bar=disp_grid_bar,
                dest_pe_ctr=dest_pe_ctr,
            )
            _trace_flydsl_ordered(rank, npes, "dispatch: <<< fused_jit + cuda.sync (host ok)")
            _mori_like_bench_post_round_fence()
            return

        assert copy_jit is not None and main_jit is not None
        _trace_flydsl_ordered(rank, npes, "dispatch: >>> copy_jit (pack staging) launch")
        _trace_pin(
            rank,
            "P021",
            "immediately before copy_jit.launch" + _bench_n_sfx,
            bench_iter=_pin_n,
        )
        copy_jit(
            fx.Int64(inp.data_ptr()),
            fx.Int64(idx.data_ptr()),
            fx.Int64(wts.data_ptr()),
            scales_ptr,
            fx.Int64(staging.data_ptr()),
            fx.Int32(cur_tok),
            stream=stream,
        )
        # Same stream orders copy → main; extra sync here matched mori back-to-back launches and
        # added a full round-trip per dispatch when trace/pins were off (large bench gap).
        if _flydsl_trace_on() or _flydsl_pinpoint_on():
            torch.cuda.synchronize(dev)
        _trace_pin(
            rank,
            "P022",
            "copy_jit returned + cuda.sync done" + _bench_n_sfx,
            bench_iter=_pin_n,
        )
        _trace_flydsl_ordered(rank, npes, "dispatch: <<< copy_jit + cuda.sync (host ok)")
        _trace_flydsl(
            rank,
            "dispatch: main_jit ptrs "
            f"idx=0x{idx.data_ptr():x} staging=0x{staging.data_ptr():x} "
            f"dispatch_inp=0x{dispatch_inp.data_ptr():x} chunk_flag=0x{chunk_flag.data_ptr():x} "
            f"node_recv=0x{node_recv.data_ptr():x} block_flag=0x{block_flag.data_ptr():x} "
            f"inter_bar=0x{inter_bar.data_ptr():x} inter_dest=0x{inter_dest.data_ptr():x} "
            f"recv_tok=0x{recv_tok_num.data_ptr():x} total_recv=0x{total_recv.data_ptr():x} "
            f"cross_dev=0x{cross_dev_flag.data_ptr():x} cur_tok={int(cur_tok)}",
        )
        _trace_flydsl_ordered(rank, npes, "dispatch: >>> main_jit (ep_dispatch_internode_v1ll_main) launch")
        _trace_pin(
            rank,
            "P023",
            "immediately before main_jit.launch" + _bench_n_sfx,
            bench_iter=_pin_n,
        )
        main_jit(
            fx.Int64(idx.data_ptr()),
            fx.Int64(staging.data_ptr()),
            fx.Int64(dispatch_inp.data_ptr()),
            fx.Int64(chunk_flag.data_ptr()),
            fx.Int64(node_recv.data_ptr()),
            fx.Int64(block_flag.data_ptr()),
            fx.Int64(inter_bar.data_ptr()),
            fx.Int64(inter_dest.data_ptr()),
            fx.Int64(inter_send.data_ptr()),
            fx.Int64(disp_dest_map.data_ptr()),
            fx.Int64(dest_pe_ctr.data_ptr()),
            fx.Int64(recv_tok_num.data_ptr()),
            fx.Int64(p2p_recv.data_ptr()),
            fx.Int64(total_recv.data_ptr()),
            fx.Int64(disp_grid_bar.data_ptr()),
            fx.Int64(comb_grid_bar.data_ptr()),
            fx.Int64(cross_dev_flag.data_ptr()),
            fx.Int64(shmem_tok_off.data_ptr()),
            fx.Int64(p2p_tok_off.data_ptr()),
            fx.Int64(p2p_tis.data_ptr()),
            fx.Int64(p2p_out_tok.data_ptr()),
            fx.Int64(p2p_out_idx.data_ptr()),
            fx.Int64(p2p_out_wts.data_ptr()),
            fx.Int64(0),
            *std_moe_args,
            fx.Int32(cur_tok),
            fx.Int32(_dev_tr),
            fx.Int32(_dev_seq),
            stream=stream,
        )
        torch.cuda.synchronize(dev)
        _trace_pin(
            rank,
            "P024",
            "main_jit returned + cuda.sync done" + _bench_n_sfx,
            bench_iter=_pin_n,
        )
        _trace_dispatch_postsync_readback(
            rank,
            dev,
            "post-split-main-dispatch",
            bench_iter=_pin_n,
            total_recv=total_recv,
            cross_dev_flag=cross_dev_flag,
            recv_tok_num=recv_tok_num,
            disp_grid_bar=disp_grid_bar,
            dest_pe_ctr=dest_pe_ctr,
        )
        _trace_flydsl_ordered(rank, npes, "dispatch: <<< main_jit + cuda.sync (host ok)")
        _mori_like_bench_post_round_fence()

    def _launch_combine():
        if not run_combine:
            raise RuntimeError("_launch_combine without run_combine")
        _trace_flydsl_ordered(rank, npes, "combine: enter _launch_combine")
        _trace_pin(rank, "P030", "_launch_combine entered")
        _trace_flydsl(rank, "combine: prep zeros (inter_bar, chunk_flag_combine, xdb_sym)")
        inter_bar.zero_()
        chunk_flag_combine.zero_()
        _zero_tensor(xdb_sym)
        _trace_pin(rank, "P031", "combine prep local zero (inter_bar/cfc/xdb) done")
        _trace_flydsl_ordered(rank, npes, "combine: >>> combine_sync_jit (EpCombineSync)")
        _trace_pin(rank, "P032", "immediately before combine_sync_jit.launch")
        combine_sync_jit(
            fx.Int64(out_bf16.data_ptr()),
            fx.Int64(combine_inp.data_ptr()),
            fx.Int64(total_recv.data_ptr()),
            stream=stream,
        )
        torch.cuda.synchronize(dev)
        _trace_pin(rank, "P033", "combine_sync_jit + cuda.sync done")
        _trace_flydsl_ordered(rank, npes, "combine: <<< combine_sync_jit + cuda.sync")
        # Ordered trace barriers disappear when FLYDSL_TRACE_INTERNODE=0; shmem/P2P still needs
        # a collective fence before combine_bar (ROCm: intermittent fault at combine_main / P037).
        torch.cuda.synchronize(dev)
        dist.barrier()
        ms.shmem_barrier_all()
        torch.cuda.synchronize(dev)
        _trace_pin(rank, "P033b", "pre-combine_bar cuda.sync + dist + shmem_barrier fence")
        _trace_flydsl_ordered(rank, npes, "combine: >>> combine_bar_jit (EpCombineSyncBarrier)")
        _trace_pin(rank, "P034", "immediately before combine_bar_jit.launch")
        combine_bar_jit(
            fx.Int64(cross_dev_flag.data_ptr()),
            fx.Int64(xdb_sym.data_ptr()),
            fx.Int64(p2p_xdb.data_ptr()),
            stream=stream,
        )
        torch.cuda.synchronize(dev)
        _trace_pin(rank, "P035", "combine_bar_jit + cuda.sync done")
        _trace_flydsl_ordered(rank, npes, "combine: <<< combine_bar_jit + cuda.sync")
        torch.cuda.synchronize(dev)
        dist.barrier()
        ms.shmem_barrier_all()
        torch.cuda.synchronize(dev)
        _trace_pin(rank, "P035b", "pre-combine_main cuda.sync + dist + shmem_barrier fence")
        _trace_combine_pre_main_diag(
            rank,
            dev,
            npes,
            total_recv=total_recv,
            node_recv=node_recv,
            cross_dev_flag=cross_dev_flag,
            inter_bar=inter_bar,
            chunk_flag_combine=chunk_flag_combine,
            inter_dest=inter_dest,
            disp_dest_map=disp_dest_map,
            staging=staging,
            xdb_sym=xdb_sym,
            p2p_ci=p2p_ci,
        )
        torch.cuda.synchronize(dev)
        nrecv_for_main = min(int(total_recv.detach().cpu().item()), max_tok)
        _trace_pin(rank, "P036", "combine pre-main diag + nrecv read done")
        _trace_flydsl(
            rank,
            f"combine: main_ll cur_rank_num_token=total_recv={nrecv_for_main} (not n_tok={int(cur_tok)})",
        )
        _trace_flydsl_ordered(rank, npes, "combine: >>> combine_main_jit (main LL) launch")
        _trace_flydsl(
            rank,
            "combine: main_jit ptrs "
            f"staging=0x{staging.data_ptr():x} chunk_flag=0x{chunk_flag.data_ptr():x} "
            f"node_recv=0x{node_recv.data_ptr():x} inter_dest(RDMA combine map)=0x{inter_dest.data_ptr():x} "
            f"disp_dest(XGMI combine map)=0x{disp_dest_map.data_ptr():x} p2p_ci=0x{p2p_ci.data_ptr():x} "
            f"comb_inp_local=0x{combine_inp.data_ptr():x} inter_bar=0x{inter_bar.data_ptr():x} "
            f"cfc=0x{chunk_flag_combine.data_ptr():x} xdb=0x{xdb_sym.data_ptr():x} "
            f"p2p_xdb=0x{p2p_xdb.data_ptr():x} cur_rank_num_token={nrecv_for_main}",
        )
        _trace_pin(rank, "P037", "immediately before combine_main_jit.launch (fault often here on ROCm)")
        combine_main_jit(
            fx.Int64(staging.data_ptr()),
            fx.Int64(chunk_flag.data_ptr()),
            fx.Int64(node_recv.data_ptr()),
            fx.Int64(inter_dest.data_ptr()),
            fx.Int64(disp_dest_map.data_ptr()),
            fx.Int64(p2p_ci.data_ptr()),
            fx.Int64(combine_inp.data_ptr()),
            fx.Int64(inter_bar.data_ptr()),
            fx.Int64(chunk_flag_combine.data_ptr()),
            fx.Int64(cross_dev_flag.data_ptr()),
            fx.Int64(xdb_sym.data_ptr()),
            fx.Int64(p2p_xdb.data_ptr()),
            fx.Int32(nrecv_for_main),
            stream=stream,
        )
        torch.cuda.synchronize(dev)
        _trace_pin(rank, "P038", "combine_main_jit + cuda.sync done (host survived)")
        _trace_after_combine_main_ll(
            rank,
            dev,
            inter_bar=inter_bar,
            chunk_flag=chunk_flag,
            chunk_flag_combine=chunk_flag_combine,
        )
        _trace_flydsl_ordered(rank, npes, "combine: <<< combine_main_jit + cuda.sync (host survived)")
        _trace_flydsl_ordered(rank, npes, "combine: >>> combine_all_jit (all)")
        _trace_pin(rank, "P039", "immediately before combine_all_jit.launch")
        combine_all_jit(
            fx.Int64(staging.data_ptr()),
            fx.Int64(inter_send.data_ptr()),
            fx.Int64(idx.data_ptr()),
            fx.Int64(combine_out_shmem.data_ptr()),
            fx.Int64(total_recv.data_ptr()),
            fx.Int64(block_flag.data_ptr()),
            fx.Int32(nrecv_for_main),
            stream=stream,
        )
        torch.cuda.synchronize(dev)
        _trace_pin(rank, "P040", "combine_all_jit + cuda.sync done")
        _trace_after_combine_all(
            rank,
            dev,
            combine_out_shmem=combine_out_shmem,
            dt=dt,
            max_tok=max_tok,
            hidden_dim=hidden_dim,
            n_tok=int(cur_tok),
        )
        _trace_flydsl_ordered(rank, npes, "combine: <<< combine_all_jit + cuda.sync")

    # view used by combine path (dispatch output in shmem)
    out_bf16 = shmem_out_tok.view(dt).view(max_recv, hidden_dim)

    if bench:
        global _SUPPRESS_INTERNODE_TRACE_FOR_BENCH
        if os.environ.get("FLYDSL_TRACE_INTERNODE_BENCH", "0") not in (
            "1",
            "true",
            "True",
        ):
            _SUPPRESS_INTERNODE_TRACE_FOR_BENCH = True
        try:

            def _one_step_dispatch():
                _reset_state()
                _launch_copy_main()

            def _prep_dispatch_for_combine():
                _one_step_dispatch()
                torch.cuda.synchronize(dev)
                dist.barrier()
                ms.shmem_barrier_all()

            def _one_step_e2e():
                _one_step_dispatch()
                _launch_combine()
                torch.cuda.synchronize(dev)

            if bench_scope == "dispatch":
                if _bench_dispatch_kernel_only:
                    # Align with mori ``dispatch()`` timing: kernels only per timed iter (warmup+timed).
                    _reset_state()
                    torch.cuda.synchronize(dev)
                    dist.barrier()
                    gpu_us = bench_gpu_us_torch(
                        _launch_copy_main, warmup=bench_warmup, iters=bench_iters
                    )
                else:
                    _one_step_dispatch()
                    torch.cuda.synchronize(dev)
                    dist.barrier()
                    gpu_us = bench_gpu_us_torch(
                        _one_step_dispatch, warmup=bench_warmup, iters=bench_iters
                    )
            elif bench_scope == "combine":
                _prep_dispatch_for_combine()
                _launch_combine()
                torch.cuda.synchronize(dev)
                dist.barrier()
                if rank == 0:
                    print(
                        "[rank0] FlyDSL: combine bench timing "
                        f"(warmup={bench_warmup}, iters={bench_iters}; "
                        "each iter: untimed redispatch+barriers then timed combine only — can take minutes)",
                        flush=True,
                    )
                gpu_us = bench_gpu_us_prep_body(
                    _prep_dispatch_for_combine,
                    _launch_combine,
                    warmup=bench_warmup,
                    iters=bench_iters,
                    device=dev,
                )
            elif bench_scope == "e2e":
                _one_step_e2e()
                torch.cuda.synchronize(dev)
                dist.barrier()
                if rank == 0:
                    print(
                        "[rank0] FlyDSL: e2e bench timing "
                        f"(warmup={bench_warmup}, iters={bench_iters})...",
                        flush=True,
                    )
                gpu_us = bench_gpu_us_torch(
                    _one_step_e2e, warmup=bench_warmup, iters=bench_iters
                )
            else:
                raise ValueError(f"unknown bench_scope {bench_scope!r}")

            torch.cuda.synchronize(dev)
            ms.shmem_barrier_all()
            return {
                "out_tok": torch.empty(0),
                "out_wts": torch.empty(0),
                "out_idx": torch.empty(0, dtype=torch.int32),
                "total_recv": -1,
                "max_chunk_num": max_chunk_num,
                "xfer_bytes": xb,
                "bench_gpu_us": gpu_us,
                "bench_scope": bench_scope,
                "bench_tokens_per_rank": n_tok,
            }
        finally:
            _SUPPRESS_INTERNODE_TRACE_FOR_BENCH = False

    _trace_flydsl(rank, "non-bench: calling _launch_copy_main()")
    _launch_copy_main()
    torch.cuda.synchronize(dev)
    # Gloo process group + mori shmem are separate; without dist.barrier, ranks can enter
    # combine with divergent dispatch visibility (seen total_recv 0 vs 2) and fault in combine_main.
    if dist.is_initialized() and world_size > 1:
        dist.barrier()
    ms.shmem_barrier_all()
    _trace_flydsl(rank, "non-bench: dispatch done, total_recv (device) pending host read")

    nrecv = int(total_recv.cpu().item())
    _trace_flydsl(rank, f"non-bench: total_recv={nrecv} (host)")
    comb_out_cpu = None
    if run_combine:
        _trace_flydsl(rank, "non-bench: calling _launch_combine()")
        _launch_combine()
        torch.cuda.synchronize(dev)
        ms.shmem_barrier_all()
        _trace_flydsl(rank, "non-bench: combine done + barrier")
        comb_view = combine_out_shmem.view(dt).view(max_tok, hidden_dim)
        comb_out_cpu = comb_view[:n_tok].detach().cpu()

    return {
        "out_tok": out_bf16[:nrecv].detach().cpu(),
        "out_wts": shmem_out_wts[: nrecv * k].view(nrecv, k).detach().cpu(),
        "out_idx": shmem_out_idx[: nrecv * k].view(nrecv, k).detach().cpu(),
        "total_recv": nrecv,
        "max_chunk_num": max_chunk_num,
        "xfer_bytes": xb,
        "comb_out": comb_out_cpu,
    }


def _print_bottleneck_hints(bench_scope: str) -> None:
    print(
        "\n[bench hints] Internode v1 LL — FlyDSL vs mori（优化时对照）:\n"
        "  - dispatch: 默认与 mori 一致只计 copy+main/iter（一次 reset 在 warmup 前）；"
        "``FLYDSL_BENCH_DISPATCH_INCLUDE_RESET=1`` 恢复每轮 Python 全量清零计时。\n"
        "  - combine: FlyDSL 四段 launch + 设备同步；mori 为单算子路径，"
        "差异常来自 launch 与跨 PE 同步（barrier/XDB）。\n"
        "  - e2e: 两段叠加；结合 ``xfer_bytes`` 与 hidden 判断是否为带宽/延迟主导。\n"
        f"  - 当前 bench_scope={bench_scope!r}；与 mori 对齐调 ``--block-num`` / ``--rdma-blocks`` / ``--wpb`` 后再对比。\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mori", action="store_true")
    parser.add_argument(
        "--mori-only",
        action="store_true",
        help="只跑 mori 参考（用于与 FlyDSL 解耦排障）",
    )
    parser.add_argument(
        "--flydsl-only",
        action="store_true",
        help="只跑 FlyDSL（等价于 --skip-mori，语义更明确）",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-tok", type=int, default=8)
    parser.add_argument("--n-tok", type=int, default=2)
    parser.add_argument(
        "--block-num",
        type=int,
        default=MORI_INTERNODE_V1_LL_BLOCK_NUM,
        help="与 mori internode_v1_ll _KERNEL_CONFIGS 默认一致",
    )
    parser.add_argument(
        "--rdma-blocks",
        type=int,
        default=MORI_INTERNODE_V1_LL_RDMA_BLOCKS,
        help="与 mori internode_v1_ll _KERNEL_CONFIGS 默认一致",
    )
    parser.add_argument("--wpb", type=int, default=8)
    parser.add_argument(
        "--bench",
        action="store_true",
        help=(
            "性能：与 mori 同场景对比（CUDA event）。范围由 --bench-scope 指定："
            "dispatch / combine / e2e；见 --bench-warmup / --bench-iters"
        ),
    )
    parser.add_argument(
        "--bench-scope",
        choices=("dispatch", "combine", "e2e"),
        default="dispatch",
        help=(
            "dispatch: mori dispatch vs FlyDSL copy+main（默认，与 mori 同口径）；"
            "combine: 每轮先 redispatch（不计时）再只测 combine；"
            "e2e: dispatch+combine 整段计时"
        ),
    )
    parser.add_argument(
        "--check",
        choices=("both", "dispatch", "combine"),
        default="both",
        help="精度：both=dispatch+combine；dispatch=仅 dispatch；combine=仅 combine 输出（仍要求 total_recv 一致）",
    )
    parser.add_argument("--bench-warmup", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument(
        "--std-moe",
        action="store_true",
        help="FlyDSL: compile main kernel with enable_std_moe and pass packed recv / slot-map buffers",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="仅对比 dispatch；不跑 mori/FlyDSL EpCombine（InternodeV1LL）四段 launch",
    )
    args = parser.parse_args()
    if args.mori_only and (args.skip_mori or args.flydsl_only):
        print("不能同时指定 --mori-only 与 --skip-mori/--flydsl-only", file=sys.stderr)
        sys.exit(2)
    if args.flydsl_only:
        args.skip_mori = True
    if args.check == "combine" and args.no_combine:
        print("--check combine 与 --no-combine 互斥", file=sys.stderr)
        sys.exit(2)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 2:
        print("This test expects torchrun with --nproc_per_node=2", file=sys.stderr)
        sys.exit(1)

    if rank == 0:
        print(
            "[rank0] device filter: "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')!r}",
            flush=True,
        )

    if args.bench:
        do_combine = args.bench_scope in ("combine", "e2e")
    elif args.check == "dispatch":
        do_combine = False
    elif args.check == "combine":
        do_combine = True
    else:
        do_combine = not args.no_combine

    dev = setup_distributed(rank, world_size)
    props = torch.cuda.get_device_properties(dev)
    mp_count = props.multi_processor_count

    k = 2
    epr = 1
    try:
        golden = None
        mori_bench_us = None
        mb_common = None
        if args.bench and not args.flydsl_only:
            mb_common = dict(
                rank=rank,
                world_size=world_size,
                dev=dev,
                hidden_dim=args.hidden,
                max_tok=args.max_tok,
                k=k,
                epr=epr,
                n_tok=args.n_tok,
                block_num=args.block_num,
                rdma_block_num=args.rdma_blocks,
                wpb=args.wpb,
                bench_warmup=args.bench_warmup,
                bench_iters=args.bench_iters,
            )
        elif not args.skip_mori:
            golden = run_mori_reference(
                rank,
                world_size,
                dev,
                hidden_dim=args.hidden,
                max_tok=args.max_tok,
                k=k,
                epr=epr,
                n_tok=args.n_tok,
                block_num=args.block_num,
                rdma_block_num=args.rdma_blocks,
                wpb=args.wpb,
                include_combine=do_combine,
            )
            if rank == 0:
                tag = "mori-only" if args.mori_only else "mori golden"
                print(
                    f"[rank0] {tag}: total_recv={golden['total_recv']} "
                    f"(block={args.block_num}, rdma={args.rdma_blocks})"
                )
        if args.mori_only:
            dist.barrier()
            if rank == 0:
                print("internode_v1ll_dispatch_flydsl: mori-only OK")
            return

        # Align ranks before FlyDSL: mori reference + combine use global shmem; avoid
        # rank1 finishing FlyDSL while rank0 mori/combine still visible to the heap.
        _trace_flydsl(
            rank,
            "main: entering dist.barrier + shmem_barrier_all (after mori golden; hang→stuck peer rank)",
        )
        dist.barrier()
        ms.shmem_barrier_all()
        _trace_flydsl(rank, "main: mori phase done, entering run_flydsl_dispatch")

        # FlyDSL-only: gc/empty_cache trims PyTorch pool before big shmem alloc (see docstring).
        # After mori golden: skip empty_cache — it has correlated with combine_main faults when
        # the heap was just torn down; rely on explicit barriers + sync instead.
        if do_combine and args.skip_mori:
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        if not args.skip_mori:
            dist.barrier()
            ms.shmem_barrier_all()
            torch.cuda.synchronize(dev)
            dist.barrier()

        fly = run_flydsl_dispatch(
            rank,
            world_size,
            dev,
            hidden_dim=args.hidden,
            max_tok=args.max_tok,
            k=k,
            epr=epr,
            n_tok=args.n_tok,
            block_num=args.block_num,
            rdma_block_num=args.rdma_blocks,
            wpb=args.wpb,
            mp_count=mp_count,
            num_qp_per_pe=1,
            bench=args.bench,
            bench_warmup=args.bench_warmup,
            bench_iters=args.bench_iters,
            bench_scope=args.bench_scope if args.bench else "dispatch",
            enable_std_moe=args.std_moe,
            run_combine=do_combine,
            force_split_dispatch=os.environ.get("FLYDSL_INTERNODE_V1LL_SPLIT_DISPATCH", "0").lower()
            in ("1", "true", "yes"),
        )

        dist.barrier()
        ms.shmem_barrier_all()

        if mb_common is not None:
            torch.cuda.synchronize(dev)
            dist.barrier()
            ms.shmem_barrier_all()
            if args.bench_scope == "dispatch":
                mori_bench_us = run_mori_bench(**mb_common)
            elif args.bench_scope == "combine":
                mori_bench_us = run_mori_bench_combine(**mb_common)
            else:
                mori_bench_us = run_mori_bench_e2e(**mb_common)
            torch.cuda.synchronize(dev)
            dist.barrier()
            ms.shmem_barrier_all()

        if fly.get("bench_gpu_us") is not None:
            dist.barrier()
            if rank == 0:
                mus = mori_bench_us
                us = fly["bench_gpu_us"]
                tpr = fly["bench_tokens_per_rank"]
                xfer = fly["xfer_bytes"]
                sc = args.bench_scope
                mori_lbl = {
                    "dispatch": "mori EpDispatchCombineOp.dispatch (CopyToStaging+LowLatency)",
                    "combine": "mori EpDispatchCombineOp.combine (timed segment only)",
                    "e2e": "mori dispatch+combine (single timed step per iter)",
                }[sc]
                fly_lbl = {
                    "dispatch": "FlyDSL copy+main (per iter; default fused single-jit)",
                    "combine": "FlyDSL combine 4 launches (prep=redispatch+barrier, untimed)",
                    "e2e": "FlyDSL reset+copy+main+combine (per iter)",
                }[sc]
                if mus is not None:
                    print(
                        f"[bench scope={sc}] {mori_lbl}: {mus:,.2f} us/iter per GPU "
                        f"(warmup={args.bench_warmup} iters={args.bench_iters})"
                    )
                print(f"[bench scope={sc}] {fly_lbl}: {us:,.2f} us/iter per GPU")
                if mus is not None and mus > 0:
                    print(
                        f"[bench scope={sc}] ratio FlyDSL/mori: {us / mus:,.3f}x "
                        f"({'FlyDSL slower' if us > mus else 'FlyDSL faster or equal'})"
                    )
                print(
                    f"[bench] shape: hidden={args.hidden} max_tok={args.max_tok} n_tok={args.n_tok} "
                    f"k={k} block={args.block_num} rdma_blocks={args.rdma_blocks} wpb={args.wpb} "
                    f"xfer={xfer} B/token mp={mp_count}"
                )
                print(
                    f"[bench] FlyDSL throughput (approx): {tpr * world_size * 1e6 / us:,.2f} tokens/s "
                    f"(world {world_size} ranks × {tpr} tok/iter / {us} us)"
                )
                if mus is not None and mus > 0:
                    print(
                        f"[bench] mori throughput (approx): {tpr * world_size * 1e6 / mus:,.2f} tokens/s"
                    )
                if sc == "combine":
                    print(
                        "[bench] note: combine 段计时下 tokens/s 仅为 n_tok 换算的标量，"
                        "不代表 combine 算子语义吞吐；对比以 us/iter 为主。"
                    )
                _print_bottleneck_hints(sc)
            dist.barrier()
            return

        if rank == 0:
            print(
                f"[rank0] FlyDSL total_recv={fly['total_recv']} "
                f"(mp={mp_count}, xfer={fly['xfer_bytes']} B)"
            )
        if golden is not None:
            ok_n = golden["total_recv"] == fly["total_recv"]
            if not ok_n:
                raise RuntimeError(
                    f"rank {rank}: total_recv mismatch mori={golden['total_recv']} fly={fly['total_recv']}"
                )
            ok_t = ok_w = ok_i = False
            if golden["out_idx"].shape[0] == fly["out_idx"].shape[0]:
                g_tok, f_tok, g_wts, f_wts, g_i_s, f_i_s = _aligned_dispatch_tensors(
                    golden, fly, k
                )
                ok_t = torch.allclose(g_tok.float(), f_tok.float(), rtol=1e-2, atol=1e-2)
                ok_w = torch.allclose(g_wts, f_wts, rtol=1e-5, atol=1e-5)
                ok_i = torch.equal(g_i_s.cpu(), f_i_s.cpu())
            else:
                raise RuntimeError(
                    f"rank {rank}: total_recv match but shape mismatch "
                    f"{golden['out_idx'].shape} vs {fly['out_idx'].shape}"
                )
            if args.check in ("both", "dispatch"):
                print(
                    f"[rank{rank}] vs mori [dispatch]: total_match={ok_n} tok={ok_t} wts={ok_w} idx={ok_i} "
                    f"(mori_recv={golden['total_recv']})",
                    flush=True,
                )
                if not (ok_t and ok_w and ok_i and ok_n):
                    raise RuntimeError(f"rank {rank}: FlyDSL vs mori dispatch mismatch")
            if (
                do_combine
                and args.check in ("both", "combine")
                and golden.get("comb_out") is not None
                and fly.get("comb_out") is not None
            ):
                ok_c = torch.allclose(
                    golden["comb_out"].float(),
                    fly["comb_out"].float(),
                    rtol=1e-2,
                    atol=1e-2,
                )
                print(f"[rank{rank}] vs mori [combine]: tok={ok_c}", flush=True)
                if not ok_c:
                    _print_combine_mismatch(rank, golden["comb_out"], fly["comb_out"])
                    raise RuntimeError(f"rank {rank}: FlyDSL vs mori combine mismatch")
            elif args.check == "combine" and (
                golden.get("comb_out") is None or fly.get("comb_out") is None
            ):
                raise RuntimeError(f"rank {rank}: --check combine requires combine outputs")
        dist.barrier()
        if rank == 0:
            print("internode_v1ll_dispatch_flydsl: OK")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
