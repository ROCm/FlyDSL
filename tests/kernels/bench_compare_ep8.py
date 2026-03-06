"""
EP=8 FlyDSL vs mori 性能对比基准测试
测试多组 max_tokens 配置，比较 dispatch/combine 延迟和带宽。

Usage (Python spawn mode):
    python tests/kernels/bench_compare_ep8.py

Usage (torchrun mode):
    torchrun --nproc_per_node=8 tests/kernels/bench_compare_ep8.py
"""
import os
import sys
import time

import torch
import torch.distributed as dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")

_FLYDSL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

from kernels.dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

# ============================================================
# 测试配置（多组 max_tokens）
# ============================================================
WORLD_SIZE = 8
WARMUP     = 2
ITERS      = 8
HIDDEN_DIM = 7168
K          = 8                  # top-k 专家数
NUM_EXPERTS = 256               # 总专家数
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE  # = 32

# (max_tokens_per_rank, block_num, warp_per_block)
TEST_CONFIGS = [
    (256,  80, 16, "低负载"),
    (512,  80, 16, "中低负载"),
    (1024, 80, 16, "中负载"),
    (2048, 80, 16, "中高负载"),
    (4096, 80, 16, "高负载"),
]


# ============================================================
# 分布式初始化
# ============================================================
def setup(rank, ws, port):
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(port),
        "RANK":        str(rank),
        "LOCAL_RANK":  str(rank),
        "WORLD_SIZE":  str(ws),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=ws,
        device_id=torch.device("cuda", rank),
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    import mori.shmem as ms
    ms.shmem_torch_process_group_init("default")


def cleanup():
    import mori.shmem as ms
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# 数据生成
# ============================================================
def gen_data(rank, max_tok, dev, rng):
    inp = torch.randn(max_tok, HIDDEN_DIM, dtype=torch.bfloat16,
                      generator=rng, device=dev)
    idx = torch.empty(max_tok, K, dtype=torch.int32)
    for i in range(max_tok):
        perm = torch.randperm(NUM_EXPERTS, generator=rng, device=dev)
        idx[i] = perm[:K].to(torch.int32)
    idx = idx.to(dev)
    wts = torch.rand(max_tok, K, dtype=torch.float32, generator=rng, device=dev)
    return inp, wts, idx


# ============================================================
# 单次 benchmark 函数
# ============================================================
def bench_once(op, inp, wts, idx, bn, wpb):
    """运行一次 dispatch+combine，返回 (disp_us, comb_us, n_recv)。"""
    op.reset()
    dist.barrier()

    se_d = torch.cuda.Event(enable_timing=True)
    ee_d = torch.cuda.Event(enable_timing=True)
    se_c = torch.cuda.Event(enable_timing=True)
    ee_c = torch.cuda.Event(enable_timing=True)

    # dispatch
    dist.barrier()
    se_d.record()
    disp_out_tok, disp_out_wts, _, disp_out_idx, recv_num = op.dispatch(
        inp, wts, None, idx, block_num=bn, warp_per_block=wpb)
    ee_d.record()
    torch.cuda.synchronize()
    disp_ms = se_d.elapsed_time(ee_d)

    n_recv = int(recv_num[0].item())
    comb_in = disp_out_tok[:n_recv].to(torch.bfloat16)

    # combine
    dist.barrier()
    se_c.record()
    op.combine(comb_in, None, idx, block_num=bn, warp_per_block=wpb)
    ee_c.record()
    torch.cuda.synchronize()
    comb_ms = se_c.elapsed_time(ee_c)

    return disp_ms * 1000, comb_ms * 1000, n_recv


# ============================================================
# 单组配置的 benchmark
# ============================================================
def run_one_config(rank, max_tok, bn, wpb, label, port_offset):
    dev = f"cuda:{rank}"
    rng = torch.Generator(device=dev)
    rng.manual_seed(42 + rank)
    inp, wts, idx = gen_data(rank, max_tok, dev, rng)

    # 构建 FlyDSL op
    fly_cfg = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=WORLD_SIZE,
        hidden_dim=HIDDEN_DIM,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=EXPERTS_PER_RANK,
        num_experts_per_token=K,
        block_num=bn, warp_num_per_block=wpb,
    )
    fly_op = FlyDSLDispatchCombineIntraNodeOp(fly_cfg)

    # 构建 mori reference op
    import mori.ops as mori_ops
    ref_cfg = mori_ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank, world_size=WORLD_SIZE,
        hidden_dim=HIDDEN_DIM,
        scale_dim=0, scale_type_size=0, max_token_type_size=2,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=EXPERTS_PER_RANK,
        num_experts_per_token=K,
        warp_num_per_block=wpb, block_num=bn,
        use_external_inp_buf=True, gpu_per_node=WORLD_SIZE,
    )
    ref_op = mori_ops.EpDispatchCombineOp(ref_cfg)

    def run_iters(op, is_fly):
        results = []
        for it in range(WARMUP + ITERS):
            dm, cm, nr = bench_once(op, inp, wts, idx, bn, wpb)
            if it >= WARMUP:
                results.append((dm, cm, nr))
        return results

    fly_results = run_iters(fly_op, True)
    ref_results = run_iters(ref_op, False)

    # 汇聚各 rank 的结果，计算 max latency（所有 rank 最慢的那个）
    def gather_stats(results, tag):
        disp_us_list, comb_us_list = [], []
        for dm, cm, nr in results:
            # all-gather latencies from all ranks
            d_buf = [torch.zeros(1) for _ in range(WORLD_SIZE)]
            c_buf = [torch.zeros(1) for _ in range(WORLD_SIZE)]
            dist.all_gather(d_buf, torch.tensor([dm]))
            dist.all_gather(c_buf, torch.tensor([cm]))
            max_d = max(t.item() for t in d_buf)
            max_c = max(t.item() for t in c_buf)
            disp_us_list.append(max_d)
            comb_us_list.append(max_c)

        best_d = min(disp_us_list)
        best_c = min(comb_us_list)
        avg_d  = sum(disp_us_list) / len(disp_us_list)
        avg_c  = sum(comb_us_list) / len(comb_us_list)

        # bandwidth: bytes moved per rank / time
        n_recv = results[0][2]
        bps_per_rank = n_recv * HIDDEN_DIM * 2  # bf16 = 2 bytes
        bw_d = bps_per_rank / 1e9 / (best_d / 1e6)
        bw_c = bps_per_rank / 1e9 / (best_c / 1e6)

        return best_d, avg_d, bw_d, best_c, avg_c, bw_c, n_recv

    fly_stats = gather_stats(fly_results, "FlyDSL")
    ref_stats = gather_stats(ref_results, "mori")

    if rank == 0:
        fd, fa, fbd, fc, fca, fbc, fn = fly_stats
        rd, ra, rbd, rc, rca, rbc, rn = ref_stats
        print(f"\n[EP={WORLD_SIZE} | {label} | max_tok={max_tok} | h={HIDDEN_DIM} | k={K}]")
        print(f"  {'':8s}  {'Dispatch(best)':>16s}  {'Dispatch(avg)':>13s}  "
              f"{'Combine(best)':>13s}  {'Combine(avg)':>12s}  {'D+C(best)':>10s}  n_recv")
        print(f"  {'FlyDSL':8s}  {fd:10.0f}μs ({fbd:4.1f}GB/s)  "
              f"{fa:7.0f}μs    {fc:7.0f}μs ({fbc:4.1f}GB/s)  "
              f"{fca:6.0f}μs    {fd+fc:7.0f}μs    {fn}")
        print(f"  {'mori':8s}  {rd:10.0f}μs ({rbd:4.1f}GB/s)  "
              f"{ra:7.0f}μs    {rc:7.0f}μs ({rbc:4.1f}GB/s)  "
              f"{rca:6.0f}μs    {rd+rc:7.0f}μs    {rn}")
        # speedup/slowdown
        sd = rd / fd if fd > 0 else 0
        sc = rc / fc if fc > 0 else 0
        print(f"  {'ratio':8s}  {'mori/fly:':>10s} {sd:.2f}x            "
              f"{'mori/fly:':>10s} {sc:.2f}x")
        print(f"  {'':8s}  (>1.0 = FlyDSL faster, <1.0 = mori faster)")

    del fly_op, ref_op
    dist.barrier()


# ============================================================
# 主 worker 函数
# ============================================================
def worker(rank, ws, port):
    setup(rank, ws, port)

    if rank == 0:
        print("=" * 70)
        print(f"EP={ws} 性能对比: FlyDSL vs mori IntraNode Dispatch+Combine")
        print(f"warmup={WARMUP} iters={ITERS} hidden={HIDDEN_DIM} k={K}")
        print("=" * 70)

    try:
        for cfg_idx, (max_tok, bn, wpb, label) in enumerate(TEST_CONFIGS):
            run_one_config(rank, max_tok, bn, wpb, label, port + cfg_idx)
            dist.barrier()

        if rank == 0:
            print("\n" + "=" * 70)
            print("所有配置测试完成")
            print("=" * 70)
    finally:
        cleanup()


# ============================================================
# 入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29550)
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        # torchrun 模式
        rank = int(os.environ["LOCAL_RANK"])
        ws   = int(os.environ.get("WORLD_SIZE", WORLD_SIZE))
        worker(rank, ws, args.port)
    else:
        # python 直接运行（spawn 模式）
        print(f"[*] Spawning {WORLD_SIZE} processes...")
        torch.multiprocessing.spawn(
            worker, args=(WORLD_SIZE, args.port),
            nprocs=WORLD_SIZE, join=True,
        )


if __name__ == "__main__":
    main()
