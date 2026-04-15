"""
FlyDSL v2 和 mori ref 的 dispatch/combine kernel 性能测试。

两个正交维度可自由组合：
  --mode       测量方式：profile（torch.profiler 采集）| bench（CUDA Event 计时）
  --cudagraph  执行方式：不带此标志 = eager 模式 | 带 = CUDAGraph capture+replay

四种组合：
  1. profile + eager    : torch.profiler 采集 eager 执行的 kernel + E2E + CPU 时间
  2. bench   + eager    : CUDA Event 计时 eager dispatch/combine（无 profiler 开销）
  3. profile + cudagraph: torch.profiler 采集 CUDAGraph replay 中的 kernel 时间
  4. bench   + cudagraph: CUDA Event 计时 CUDAGraph replay（零 Python launch 开销）

启动方式（支持 torchrun 或直接 python）：
  # profile + eager（默认）
  python tests/kernels/test_profiler_dispatch_combine.py --max-tokens 512

  # bench + eager
  python tests/kernels/test_profiler_dispatch_combine.py --mode bench

  # bench + cudagraph
  python tests/kernels/test_profiler_dispatch_combine.py --mode bench --cudagraph

  # profile + cudagraph
  python tests/kernels/test_profiler_dispatch_combine.py --mode profile --cudagraph

  # 只测 FlyDSL
  python tests/kernels/test_profiler_dispatch_combine.py --bench-op flydsl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in [_ROOT, "/home/yashao/FlyDSL/python", "/home/yashao/mori/python"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mori.shmem as ms
from kernels.dispatch_combine_intranode_op_v2 import (
    FlyDSLDispatchCombineConfigV2,
    FlyDSLDispatchCombineIntraNodeOpV2,
)


# ─── 分布式初始化 ─────────────────────────────────────────────────────────────
def setup_distributed(rank, world_size, master_port=29600):
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


def build_mori_ref(rank, world_size, cfg,
                   block_num: int = None, warp_per_block: int = None):
    from mori.ops.dispatch_combine import EpDispatchCombineConfig, EpDispatchCombineOp
    elem = torch.tensor([], dtype=cfg.data_type).element_size()
    mcfg = EpDispatchCombineConfig(
        data_type=cfg.data_type,
        rank=rank, world_size=world_size,
        hidden_dim=cfg.hidden_dim,
        scale_dim=cfg.top_k, scale_type_size=4,
        max_token_type_size=elem,
        max_num_inp_token_per_rank=cfg.max_num_inp_token_per_rank,
        num_experts_per_rank=cfg.num_experts_per_rank,
        num_experts_per_token=cfg.top_k,
        warp_num_per_block=warp_per_block if warp_per_block is not None else cfg.warp_num_per_block,
        block_num=block_num if block_num is not None else cfg.block_num,
        gpu_per_node=world_size,
    )
    return EpDispatchCombineOp(mcfg)


def _save_profile_json(prof, out_path: str, rank: int, op_tag: str, meta: dict):
    """将 profiler 结果序列化为 JSON 文件。

    JSON 结构：
      {
        "meta": {op_tag, rank, max_tokens, hidden_dim, k, world_size, ...},
        "kernel_stats": [ {name, calls, cuda_time_avg_us, cpu_time_avg_us}, ... ]
      }
    """
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
    # 按 GPU time 降序
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


def _allreduce_stats(prof, op_tag: str, rank: int, world_size: int,
                     dev: torch.device) -> dict:
    """从本卡 profiler 提取关键指标，跨卡 all_reduce 后返回 avg/min/max 字典。

    采集 6 项指标（顺序固定，打包成 float64 tensor 做 all_reduce）：
      0: dispatch GPU kernel time (μs/call)
      1: combine  GPU kernel time (μs/call)
      2: dispatch record_function CUDA time (μs/call)
      3: combine  record_function CUDA time (μs/call)
      4: dispatch record_function CPU  time (μs/call)
      5: combine  record_function CPU  time (μs/call)
    """
    # kernel 名称映射
    if op_tag == "flydsl":
        d_kernel = "ep_dispatch_intranode_0"
        c_kernel = "ep_combine_intranode_0"
    else:
        d_kernel = "EpDispatchIntraNodeKernel_bf16"
        c_kernel = "EpCombineIntraNodeKernel_bf16_nop2p"
    d_label = f"{op_tag}::dispatch"
    c_label  = f"{op_tag}::combine"

    ev = {e.key: e for e in prof.key_averages()}

    def gpu_us(key):
        e = ev.get(key)
        return e.device_time if (e and e.count) else 0.0

    def cpu_us(key):
        e = ev.get(key)
        return e.cpu_time if (e and e.count) else 0.0

    local = torch.tensor([
        gpu_us(d_kernel), gpu_us(c_kernel),
        gpu_us(d_label),  gpu_us(c_label),
        cpu_us(d_label),  cpu_us(c_label),
    ], dtype=torch.float64, device=dev)

    s = local.clone(); dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = ["dispatch_gpu", "combine_gpu",
            "dispatch_cuda_e2e", "combine_cuda_e2e",
            "dispatch_cpu_e2e", "combine_cpu_e2e"]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()}
            for i, k in enumerate(keys)}


def _print_aggregated(stats: dict, op_tag: str, world_size: int, meta: dict):
    """rank 0 打印全卡聚合统计。"""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {op_tag.upper()}  EP={world_size}  bs={meta['max_tokens']}  "
          f"h={meta['hidden_dim']}  k={meta['k']}  ({meta['iters']} iters)")
    print(f"  所有 {world_size} 张卡的 avg / min / max（μs/call）")
    print(sep)
    hdr = f"  {'指标':<36}  {'avg':>8}  {'min':>8}  {'max':>8}"
    print(hdr)
    print(f"  {'-'*60}")

    rows = [
        ("[Device] dispatch kernel GPU time",    "dispatch_gpu"),
        ("[Device] combine  kernel GPU time",    "combine_gpu"),
        ("[E2E]    dispatch CUDA time (含sync)", "dispatch_cuda_e2e"),
        ("[E2E]    combine  CUDA time (含sync)", "combine_cuda_e2e"),
        ("[Host]   dispatch CPU  time",           "dispatch_cpu_e2e"),
        ("[Host]   combine  CPU  time",           "combine_cpu_e2e"),
    ]
    for label, key in rows:
        v = stats[key]
        print(f"  {label:<36}  {v['avg']:>8.1f}  {v['min']:>8.1f}  {v['max']:>8.1f}")
    print()


def _allreduce_cudagraph_stats_from_key_averages(
        prof, op_tag: str, rank: int, world_size: int,
        dev: torch.device) -> dict:
    """从 key_averages() 提取指标（仅含 active 阶段数据），跨卡 all_reduce。

    采集 4 项：
      0: dispatch kernel GPU time
      1: combine  kernel GPU time
      2: cudagraph_replay CUDA E2E time
      3: cudagraph_replay CPU  E2E time
    """
    if op_tag == "flydsl":
        d_kernel = "ep_dispatch_intranode_0"
        c_kernel = "ep_combine_intranode_0"
    else:
        d_kernel = "EpDispatchIntraNodeKernel_bf16"
        c_kernel = "EpCombineIntraNodeKernel_bf16_nop2p"
    cg_label = f"{op_tag}::cudagraph_replay"

    ev = {e.key: e for e in prof.key_averages()}

    def gpu_us(key):
        e = ev.get(key)
        return e.device_time if (e and e.count) else 0.0

    def cpu_us(key):
        e = ev.get(key)
        return e.cpu_time if (e and e.count) else 0.0

    local = torch.tensor([
        gpu_us(d_kernel), gpu_us(c_kernel),
        gpu_us(cg_label), cpu_us(cg_label),
    ], dtype=torch.float64, device=dev)

    s = local.clone(); dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = ["dispatch_gpu", "combine_gpu", "replay_cuda_e2e", "replay_cpu_e2e"]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()}
            for i, k in enumerate(keys)}


def _cudagraph_stats_from_trace(trace_path: str, op_tag: str,
                                rank: int, world_size: int,
                                dev: torch.device,
                                active_iters: int, skip_first: int = 5) -> dict:
    """从 chrome trace JSON 手动统计 kernel 性能，跳过前 skip_first 次 active 调用。

    流程：解析 trace → 按时间排序取最后 active_iters 个事件 → 丢弃前 skip_first 个 → 跨卡聚合。
    """
    with open(trace_path) as f:
        tr = json.load(f)

    if op_tag == "flydsl":
        d_name, c_name = "ep_dispatch_intranode_0", "ep_combine_intranode_0"
    else:
        d_name = "EpDispatchIntraNodeKernel_bf16"
        c_name = "EpCombineIntraNodeKernel_bf16_nop2p"
    cg_name = f"{op_tag}::cudagraph_replay"

    kernel_events = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
    d_all = sorted([e for e in kernel_events if d_name in e.get("name", "")],
                   key=lambda e: e["ts"])
    c_all = sorted([e for e in kernel_events if c_name in e.get("name", "")],
                   key=lambda e: e["ts"])
    cg_all = sorted([e for e in tr["traceEvents"]
                     if e.get("cat") == "gpu_user_annotation"
                     and cg_name in e.get("name", "")],
                    key=lambda e: e["ts"])

    d_active = [e["dur"] for e in d_all[-active_iters:]]
    c_active = [e["dur"] for e in c_all[-active_iters:]]
    cg_active = [e["dur"] for e in cg_all[-active_iters:]]

    d_valid = d_active[skip_first:]
    c_valid = c_active[skip_first:]
    cg_valid = cg_active[skip_first:]

    valid_n = len(d_valid)
    if rank == 0:
        print(f"[trace-stats] {op_tag}: trace 中 dispatch={len(d_all)} combine={len(c_all)} 个事件，"
              f"取最后 {active_iters} 个 active，跳过前 {skip_first}，有效 {valid_n} 个")

    d_avg = sum(d_valid) / valid_n if valid_n else 0.0
    c_avg = sum(c_valid) / valid_n if valid_n else 0.0
    cg_avg = sum(cg_valid) / len(cg_valid) if cg_valid else 0.0

    local = torch.tensor([d_avg, c_avg, cg_avg, 0.0],
                         dtype=torch.float64, device=dev)
    s  = local.clone(); dist.all_reduce(s,  op=dist.ReduceOp.SUM)
    mx = local.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = ["dispatch_gpu", "combine_gpu", "replay_cuda_e2e", "replay_cpu_e2e"]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()}
            for i, k in enumerate(keys)}


def _print_cudagraph_aggregated(stats: dict, op_tag: str, world_size: int, meta: dict,
                                active_iters: int = None):
    """rank 0 打印 cudagraph profiler 全卡聚合统计。"""
    n = active_iters if active_iters is not None else meta['iters']
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {op_tag.upper()} [CUDAGraph+Profiler]  EP={world_size}  bs={meta['max_tokens']}  "
          f"h={meta['hidden_dim']}  k={meta['k']}  ({n} iters)")
    print(f"  所有 {world_size} 张卡的 avg / min / max（μs/call）")
    print(sep)
    hdr = f"  {'指标':<36}  {'avg':>8}  {'min':>8}  {'max':>8}"
    print(hdr)
    print(f"  {'-'*60}")

    rows = [
        ("[Device] dispatch kernel GPU time",  "dispatch_gpu"),
        ("[Device] combine  kernel GPU time",  "combine_gpu"),
        ("[E2E]   replay CUDA time (含sync)", "replay_cuda_e2e"),
        ("[Host]  replay CPU  time",           "replay_cpu_e2e"),
    ]
    for label, key in rows:
        v = stats[key]
        print(f"  {label:<36}  {v['avg']:>8.1f}  {v['min']:>8.1f}  {v['max']:>8.1f}")
    print()


def _make_profiler(active_iters: int = None, prof_warmup: int = 10):
    """创建 profiler。

    使用 schedule 让前 (1 + prof_warmup) 步不做/轻量追踪，
    减少 ROCTracer 在多 GPU P2P shmem 场景下的累积压力。
    """
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


# ─── bench 模式：不用 profiler，用 CUDA Event 计时 ────────────────────────────
def bench_op(op, op_tag: str, inp, wts, idx, wc_buf, k,
             rank: int, world_size: int, dev: torch.device,
             warmup: int, iters: int, meta: dict):
    """无 profiler 的纯计时模式，输出 dispatch / combine 的 GPU 耗时（avg/min/max）。"""
    ms.shmem_barrier_all()
    if rank == 0:
        print(f"\n[bench] {op_tag} 预热 {warmup} 轮...")
    for _ in range(warmup):
        op.reset()
        ret = op.dispatch(inp, wts, None, idx)
        op.combine(ret[0], None, ret[3])
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"[bench] {op_tag} 计时 {iters} 轮...")

    d_events = [(torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    c_events = [(torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True)) for _ in range(iters)]

    for i in range(iters):
        # op.reset()
        dist.barrier()

        d_events[i][0].record()
        ret = op.dispatch(inp, wts, None, idx)
        d_events[i][1].record()

        dist.barrier()

        c_events[i][0].record()
        op.combine(ret[0], None, ret[3])
        c_events[i][1].record()

    torch.cuda.synchronize()
    d_list = [d_events[i][0].elapsed_time(d_events[i][1]) * 1000 for i in range(iters)]
    c_list = [c_events[i][0].elapsed_time(c_events[i][1]) * 1000 for i in range(iters)]

    # 全卡聚合 avg / min / max
    local = torch.tensor([
        sum(d_list) / len(d_list), min(d_list), max(d_list),
        sum(c_list) / len(c_list), min(c_list), max(c_list),
    ], dtype=torch.float64, device=dev)
    s  = local.clone(); dist.all_reduce(s,  op=dist.ReduceOp.SUM)
    mx = local.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg_d = (s[0] / world_size).item(); mn_d = mn[0].item(); mx_d = mx[2].item()
    avg_c = (s[3] / world_size).item(); mn_c = mn[3].item(); mx_c = mx[5].item()

    if rank == 0:
        sep = "=" * 68
        tag = (f"{op_tag.upper()}  EP={meta['world_size']}  bs={meta['max_tokens']}  "
               f"h={meta['hidden_dim']}  k={meta['k']}  ({iters} iters)")
        print(f"\n{sep}\n  {tag}\n  所有 {world_size} 张卡的 avg / min / max（μs/call）\n{sep}")
        print(f"  {'指标':<36}  {'avg':>8}  {'min':>8}  {'max':>8}")
        print(f"  {'-'*58}")
        print(f"  {'[E2E]  dispatch CUDA time':<36}  {avg_d:>8.1f}  {mn_d:>8.1f}  {mx_d:>8.1f}")
        print(f"  {'[E2E]  combine  CUDA time':<36}  {avg_c:>8.1f}  {mn_c:>8.1f}  {mx_c:>8.1f}")
        print()


# ─── cudagraph 模式：CUDA Graph capture + replay 计时 ─────────────────────────
def _cudagraph_capture_flydsl(op, inp, wts, idx, wc_buf, capture_stream):
    """FlyDSL：录制 dispatch+combine 到 CUDA Graph。

    dispatch/combine 均返回全尺寸 tensor（无 .item()、无动态切片）。
    需要先 eager 调用一次触发 flyc.compile() JIT 编译（编译过程使用
    default stream，不能在 capture 期间执行），之后 capture 中仅录制
    已编译的 kernel launch。
    """
    op.reset()
    ret = op.dispatch(inp, wts, None, idx)
    op.combine(ret[0], None, ret[3])

    op.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        ret = op.dispatch(inp, wts, None, idx)
        op.combine(ret[0], None, ret[3])
    return g, capture_stream


def _cudagraph_capture_mori(op, inp, wts, idx, wc_buf, capture_stream):
    """Mori 专用：直接在 graph capture 中录制 dispatch+combine。

    Mori 的 dispatch 在 capture 模式下返回真实 tensor，combine kernel
    从 HBM 读取 totalRecvTokenNum，无需 pre-capture eager call。
    参考 mori/tests/python/ops/bench_dispatch_combine.py stress_graph 写法。
    """
    ms.shmem_barrier_all()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        ret = op.dispatch(inp, wts, None, idx)
        op.combine(ret[0], None, ret[3])
    return g, capture_stream


def cudagraph_op(op, op_tag: str, inp, wts, idx, wc_buf, k,
                 rank: int, world_size: int, dev: torch.device,
                 warmup: int, iters: int, meta: dict):
    """CUDA Graph 模式：capture dispatch+combine kernel，replay 计时。"""
    capture_stream = torch.cuda.Stream()
    if op_tag == "flydsl":
        g, cs = _cudagraph_capture_flydsl(
            op, inp, wts, idx, wc_buf, capture_stream)
    else:
        g, cs = _cudagraph_capture_mori(
            op, inp, wts, idx, wc_buf, capture_stream)

    if rank == 0:
        print(f"\n[cudagraph] {op_tag} capture done")

    # replay warmup（HIP graph 冷启动 + GPU 缓存预热）
    replay_warmup = 10
    if rank == 0:
        print(f"[cudagraph] replay warmup {replay_warmup} 轮 + 计时 {iters} 轮（no-reset）...")
    for _ in range(replay_warmup):
        g.replay()
    torch.cuda.synchronize()

    # 计时：预分配 event pairs，循环结束后统一 sync
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(iters)]

    for i in range(iters):
        events[i][0].record()
        g.replay()
        events[i][1].record()

    torch.cuda.synchronize()
    gpu_times = [events[i][0].elapsed_time(events[i][1]) * 1000 for i in range(iters)]

    # per-replay 诊断
    per_replay_t = torch.tensor(gpu_times, dtype=torch.float64, device=dev)
    all_per_replay = [torch.zeros_like(per_replay_t) for _ in range(world_size)]
    dist.all_gather(all_per_replay, per_replay_t)

    local = torch.tensor([
        sum(gpu_times) / len(gpu_times), min(gpu_times), max(gpu_times),
    ], dtype=torch.float64, device=dev)
    s  = local.clone(); dist.all_reduce(s,  op=dist.ReduceOp.SUM)
    mx = local.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg_g = (s[0] / world_size).item(); mn_g = mn[0].item(); mx_g = mx[2].item()

    if rank == 0:
        sep = "=" * 68
        tag = (f"{op_tag.upper()} [CUDAGraph]  EP={meta['world_size']}  "
               f"bs={meta['max_tokens']}  h={meta['hidden_dim']}  k={meta['k']}  "
               f"({iters} replays)")
        print(f"\n{sep}\n  {tag}\n  所有 {world_size} 张卡的 avg / min / max（μs/call）\n{sep}")
        print(f"  {'指标':<36}  {'avg':>8}  {'min':>8}  {'max':>8}")
        print(f"  {'-'*58}")
        print(f"  {'[GPU]  dispatch+combine (event)':<36}  {avg_g:>8.1f}  {mn_g:>8.1f}  {mx_g:>8.1f}")

        print(f"\n  Per-replay GPU time (μs) — all {world_size} ranks:")
        hdr = f"  {'replay':>6}" + "".join(f"  {'R'+str(r):>8}" for r in range(world_size)) + f"  {'max':>8}"
        print(hdr)
        mat = torch.stack(all_per_replay)
        for i in range(iters):
            vals = [mat[r, i].item() for r in range(world_size)]
            mx_i = max(vals)
            row = f"  {i:>6}" + "".join(f"  {v:>8.1f}" for v in vals) + f"  {mx_i:>8.1f}"
            if mx_i > avg_g * 3:
                row += "  ← SPIKE"
            print(row)
        print()


# ─── 单算子 profiler 采集 ──────────────────────────────────────────────────────
def profile_op(op, op_tag: str, inp, wts, idx, wc_buf, k,
               rank: int, world_size: int, dev: torch.device,
               iters: int, out_dir: str, meta: dict):
    """对单个算子（FlyDSL 或 mori）独立 profiling，保存 JSON 并打印全卡聚合统计。

    使用 schedule(wait=1, warmup=10, active=iters) 让 ROCTracer 在前 11 步
    不做/轻量追踪，减少与多 GPU P2P shmem 操作的冲突。
    """
    ms.shmem_barrier_all()
    prof_warmup = 10
    total_steps = iters + 1 + prof_warmup  # wait=1 + warmup=prof_warmup + active=iters
    if rank == 0:
        print(f"\n[profiler] {op_tag} 开始采集（{iters} 轮 active + {1 + prof_warmup} 轮 ramp-up）...")

    with _make_profiler(active_iters=iters, prof_warmup=prof_warmup) as prof:
        for step in range(total_steps):
            # with record_function(f"{op_tag}::reset"):
            #     op.reset()
            dist.barrier()

            with record_function(f"{op_tag}::dispatch"):
                ret = op.dispatch(inp, wts, None, idx)

            dist.barrier()

            with record_function(f"{op_tag}::combine"):
                op.combine(ret[0], None, ret[3])
            
            # dist.barrier()

            prof.step()

    # 保存 JSON：每张卡各自保存，文件名含 op_tag 和 rank
    out_path = os.path.join(out_dir, f"{op_tag}_rank{rank}.json")
    _save_profile_json(prof, out_path, rank, op_tag, meta)
    if rank == 0:
        print(f"[profiler] {op_tag} trace → {out_path}")

    # 跨卡聚合统计（all_reduce），rank 0 打印
    agg_stats = _allreduce_stats(prof, op_tag, rank, world_size, dev)
    if rank == 0:
        _print_aggregated(agg_stats, op_tag, world_size, meta)
    return prof


# ─── profile + cudagraph 模式 ─────────────────────────────────────────────────
def profile_cudagraph_op(op, op_tag: str, inp, wts, idx, wc_buf, k,
                         rank: int, world_size: int, dev: torch.device,
                         warmup: int, iters: int, out_dir: str, meta: dict):
    """torch.profiler 采集 CUDAGraph replay，保存 JSON 并打印全卡聚合统计。

    流程：eager warmup → graph capture → replay warmup → profiler 包裹的 replay。
    """
    ms.shmem_barrier_all()

    capture_stream = torch.cuda.Stream()
    if op_tag == "flydsl":
        g, cs = _cudagraph_capture_flydsl(
            op, inp, wts, idx, wc_buf, capture_stream)
    else:
        g, cs = _cudagraph_capture_mori(
            op, inp, wts, idx, wc_buf, capture_stream)

    if rank == 0:
        print(f"\n[profile+cudagraph] {op_tag} capture done")

    # replay warmup（HIP graph 冷启动 + GPU 缓存预热）
    replay_warmup = 10
    for _ in range(replay_warmup):
        g.replay()
    torch.cuda.synchronize()

    prof_warmup = 5
    active_iters = iters
    skip_first = 5
    valid_iters = max(active_iters - skip_first, 1)
    total_steps = 1 + prof_warmup + active_iters  # wait=1 + warmup + active
    if rank == 0:
        print(f"[profile+cudagraph] {op_tag} scheduled profiler: "
              f"warmup={prof_warmup}, active={active_iters}, "
              f"丢弃前 {skip_first} 次，有效 {valid_iters} 次（no-reset）...")

    with _make_profiler(active_iters=active_iters, prof_warmup=prof_warmup) as prof:
        for step in range(total_steps):
            with record_function(f"{op_tag}::cudagraph_replay"):
                g.replay()
            prof.step()

    out_path = os.path.join(out_dir, f"{op_tag}_cudagraph_rank{rank}.json")
    _save_profile_json(prof, out_path, rank, op_tag, meta)
    trace_path = out_path.replace(".json", "_trace.json")
    if rank == 0:
        print(f"[profile+cudagraph] {op_tag} trace → {trace_path}")

    agg_stats = _cudagraph_stats_from_trace(
        trace_path, op_tag, rank, world_size, dev,
        active_iters=active_iters, skip_first=skip_first)
    if rank == 0:
        _print_cudagraph_aggregated(agg_stats, op_tag, world_size, meta,
                                    active_iters=valid_iters)
    return prof


# ─── 主逻辑 ───────────────────────────────────────────────────────────────────
def run_profiler(rank, world_size, args):
    dev     = torch.device("cuda", rank)
    k       = args.k
    cur_tok = args.max_tokens
    n_exp   = world_size * args.num_experts_per_rank

    _combine_wpb = args.combine_warp_per_block if args.combine_warp_per_block > 0 else None
    cfg = FlyDSLDispatchCombineConfigV2(
        rank=rank, world_size=world_size,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=cur_tok,
        num_experts_per_rank=args.num_experts_per_rank,
        top_k=k,
        data_type=torch.bfloat16,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        chip=args.chip,
        combine_warp_num_per_block=_combine_wpb,
    )

    mori_bn  = args.mori_block_num      if args.mori_block_num      > 0 else cfg.block_num
    mori_wpb = args.mori_warp_per_block if args.mori_warp_per_block > 0 else cfg.warp_num_per_block
    meta = dict(
        world_size=world_size, max_tokens=cur_tok,
        hidden_dim=cfg.hidden_dim, k=k,
        num_experts_per_rank=args.num_experts_per_rank,
        warmup=args.warmup, iters=args.iters,
        flydsl_block_num=cfg.block_num,
        flydsl_warp_per_block=cfg.warp_num_per_block,
        mori_block_num=mori_bn,
        mori_warp_per_block=mori_wpb,
    )

    # 输出目录：/tmp/ep{ws}_bs{cur_tok}/
    out_dir = os.path.join(args.output_dir, f"ep{world_size}_bs{cur_tok}")
    os.makedirs(out_dir, exist_ok=True)

    # ── 构建算子 ───────────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"[profiler] EP={world_size}, bs={cur_tok}, h={cfg.hidden_dim}, k={k}")
        print(f"{'='*65}")
        print("[profiler] 构建 FlyDSL v2...")
    op_v2 = FlyDSLDispatchCombineIntraNodeOpV2(cfg)

    op_ref = None
    if args.compare:
        mori_bn  = args.mori_block_num  if args.mori_block_num  > 0 else None
        mori_wpb = args.mori_warp_per_block if args.mori_warp_per_block > 0 else None
        bn_str  = mori_bn  if mori_bn  else cfg.block_num
        wpb_str = mori_wpb if mori_wpb else cfg.warp_num_per_block
        if rank == 0:
            print(f"[profiler] 构建 mori ref (block_num={bn_str}, warp_per_block={wpb_str})...")
        try:
            op_ref = build_mori_ref(rank, world_size, cfg,
                                    block_num=mori_bn, warp_per_block=mori_wpb)
        except Exception as e:
            if rank == 0:
                print(f"[warn] mori ref 不可用: {e}")
    ms.shmem_barrier_all()

    # ── 准备输入（固定 seed，FlyDSL 和 mori 使用完全相同的输入）────────────────
    torch.manual_seed(42 + rank)
    inp = torch.randn(cur_tok, cfg.hidden_dim, dtype=cfg.data_type, device=dev)
    wts = torch.rand(cur_tok, k, dtype=torch.float32, device=dev)
    wts = wts / wts.sum(-1, keepdim=True)
    idx = torch.zeros(cur_tok, k, dtype=torch.int32, device=dev)
    for t in range(cur_tok):
        idx[t] = torch.randperm(n_exp, device=dev)[:k]

    # 预分配 combine 权重 buffer（FlyDSL 和 mori 共用，避免计时窗口内额外 GPU 核）
    max_recv = world_size * cur_tok
    wc_buf = torch.full((max_recv, k), 1.0 / k, dtype=torch.float32, device=dev)

    # profile+eager 模式需要外部预热；其他 3 种组合由各自函数内部处理
    do_warmup = (args.mode == "profile" and not args.cudagraph)

    if do_warmup:
        if rank == 0:
            print(f"[setup] 预热 FlyDSL {args.warmup} 轮...")
        for _ in range(args.warmup):
            op_v2.reset()
            ret = op_v2.dispatch(inp, wts, None, idx)
            op_v2.combine(ret[0], None, ret[3])
            torch.cuda.synchronize()

        if op_ref is not None:
            if rank == 0:
                print(f"[setup] 预热 mori ref {args.warmup} 轮...")
            for _ in range(args.warmup):
                op_ref.reset()
                ret_r = op_ref.dispatch(inp, wts, None, idx)
                op_ref.combine(ret_r[0], None, ret_r[3])
                torch.cuda.synchronize()

    ms.shmem_barrier_all()

    # ── 根据 mode × cudagraph 分发执行 ─────────────────────────────────────
    test_flydsl = args.bench_op in ("flydsl", "both")
    test_mori   = args.bench_op in ("mori",   "both") and op_ref is not None

    if args.mode == "bench" and not args.cudagraph:
        # bench + eager：CUDA Event 计时
        if test_flydsl:
            bench_op(op_v2, "flydsl", inp, wts, idx, wc_buf, k,
                     rank, world_size, dev, args.warmup, args.iters, meta)
        if test_mori:
            ms.shmem_barrier_all()
            bench_op(op_ref, "mori", inp, wts, idx, wc_buf, k,
                     rank, world_size, dev, args.warmup, args.iters, meta)

    elif args.mode == "bench" and args.cudagraph:
        # bench + cudagraph：CUDAGraph replay + CUDA Event 计时
        if test_flydsl:
            cudagraph_op(op_v2, "flydsl", inp, wts, idx, wc_buf, k,
                         rank, world_size, dev, args.warmup, args.iters, meta)
        if test_mori:
            ms.shmem_barrier_all()
            cudagraph_op(op_ref, "mori", inp, wts, idx, wc_buf, k,
                         rank, world_size, dev, args.warmup, args.iters, meta)

    elif args.mode == "profile" and not args.cudagraph:
        # profile + eager：torch.profiler 采集
        if test_flydsl:
            profile_op(op_v2, "flydsl", inp, wts, idx, wc_buf, k,
                       rank, world_size, dev, args.iters, out_dir, meta)
        if test_mori:
            ms.shmem_barrier_all()
            profile_op(op_ref, "mori", inp, wts, idx, wc_buf, k,
                       rank, world_size, dev, args.iters, out_dir, meta)
        if rank == 0:
            print(f"\n[profiler] 全部结果已保存到: {out_dir}/")

    elif args.mode == "profile" and args.cudagraph:
        # profile + cudagraph：torch.profiler 采集 CUDAGraph replay
        if test_flydsl:
            profile_cudagraph_op(op_v2, "flydsl", inp, wts, idx, wc_buf, k,
                                 rank, world_size, dev, args.warmup, args.iters,
                                 out_dir, meta)
        if test_mori:
            ms.shmem_barrier_all()
            profile_cudagraph_op(op_ref, "mori", inp, wts, idx, wc_buf, k,
                                 rank, world_size, dev, args.warmup, args.iters,
                                 out_dir, meta)
        if rank == 0:
            print(f"\n[profiler] 全部结果已保存到: {out_dir}/")


# ─── Worker / 命令行入口 ──────────────────────────────────────────────────────
def _worker(rank, world_size, args, master_port):
    setup_distributed(rank, world_size, master_port)
    try:
        run_profiler(rank, world_size, args)
    except Exception as e:
        import traceback as tb
        print(f"[rank {rank}] ERROR: {e}")
        tb.print_exc()
    finally:
        cleanup()


def _parse_args():
    p = argparse.ArgumentParser(description="torch.profiler 分析 dispatch/combine")
    p.add_argument("--world-size",           type=int, default=8)
    p.add_argument("--max-tokens",           type=int, default=512)
    p.add_argument("--hidden-dim",           type=int, default=7168)
    p.add_argument("--num-experts-per-rank", type=int, default=32)
    p.add_argument("--k",                    type=int, default=8)
    p.add_argument("--block-num",            type=int, default=80)
    p.add_argument("--warp-per-block",       type=int, default=4)
    p.add_argument("--combine-warp-per-block", type=int, default=0,
                   help="combine 内核专用 warp_per_block（0=与 --warp-per-block 相同）")
    p.add_argument("--mori-block-num",       type=int, default=0,
                   help="mori 专用 block_num（0=与FlyDSL相同，mori默认最优=80）")
    p.add_argument("--mori-warp-per-block",  type=int, default=0,
                   help="mori 专用 warp_per_block（0=与FlyDSL相同，mori默认最优=8）")
    p.add_argument("--chip",                 type=str, default="gfx942")
    p.add_argument("--warmup",               type=int, default=5,
                   help="预热轮次（不进 profiler，确保 JIT 编译完成）")
    p.add_argument("--iters",                type=int, default=5,
                   help="profiler 采集轮次")
    p.add_argument("--output-dir",           type=str, default="dispatch_profile",
                   help="JSON 输出根目录（相对当前目录），子目录按 ep{ws}_bs{tok} 命名")
    p.add_argument("--port",                 type=int, default=29800)
    p.add_argument("--no-compare",           dest="compare", action="store_false")
    # ── 模式选择 ──────────────────────────────────────────────────────────────
    p.add_argument("--mode", choices=["profile", "bench"], default="profile",
                   help="测量方式：profile=torch.profiler 采集（默认）; bench=CUDA Event 计时")
    p.add_argument("--cudagraph", action="store_true",
                   help="使用 CUDAGraph capture+replay 执行（默认 eager）")
    p.add_argument("--bench-op", choices=["flydsl", "mori", "both"], default="both",
                   help="测哪个算子（默认 both）")
    p.set_defaults(compare=True)
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
