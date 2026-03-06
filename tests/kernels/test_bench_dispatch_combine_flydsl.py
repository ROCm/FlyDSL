"""
FlyDSL Dispatch/Combine IntraNode — Accuracy Test + Performance Benchmark.

支持两种启动方式：

  方式1: python（内部 spawn，推荐用于开发调试）
    python tests/kernels/test_bench_dispatch_combine_flydsl.py --mode test
    python tests/kernels/test_bench_dispatch_combine_flydsl.py --mode bench

  方式2: torchrun（外部 fork，推荐用于生产环境）
    torchrun --nproc_per_node=8 tests/kernels/test_bench_dispatch_combine_flydsl.py --mode test

注意：dispatch/combine 本质是多 GPU 通信算子，无论哪种方式都需要启动
  world_size 个进程（默认 8），每个进程对应一块 GPU。

Reference: mori/tests/python/ops/bench_dispatch_combine.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

# FlyDSL imports — support running from any cwd
_FLYDSL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

from kernels.dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)


# ============================================================
# Distributed setup / teardown
# ============================================================
def setup_distributed(rank: int, world_size: int, master_port: int = 29500):
    """Initialize distributed process group and mori shmem.

    Works for both torchrun (env vars already set) and spawn (manual init).
    """
    # torchrun already sets LOCAL_RANK / RANK / WORLD_SIZE;
    # when called from spawn(), we set them explicitly.
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            device_id=torch.device("cuda", local_rank),
        )

    world_group = dist.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)

    import mori.shmem as ms
    ms.shmem_torch_process_group_init("default")

    return dist.get_rank(), dist.get_world_size()


def cleanup():
    import mori.shmem as ms
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# Data generation helpers
# ============================================================
def generate_test_data(cfg: FlyDSLDispatchCombineConfig,
                       rng: torch.Generator,
                       use_max_tokens: bool = False):
    """Generate random test data consistent across all ranks via shared seed."""
    device = torch.device("cuda", cfg.rank)
    n = cfg.world_size
    k = cfg.num_experts_per_token
    total_experts = cfg.num_experts_per_rank * n

    if use_max_tokens:
        num_tok_all = [cfg.max_num_inp_token_per_rank] * n
    else:
        num_tok_all = torch.randint(
            0, cfg.max_num_inp_token_per_rank + 1, [n],
            generator=rng, device=device
        ).tolist()

    all_rank_input, all_rank_indices, all_rank_weights = [], [], []

    for r in range(n):
        nt = num_tok_all[r]
        inp = torch.randn(nt, cfg.hidden_dim, dtype=cfg.data_type,
                          generator=rng, device=device)
        all_rank_input.append(inp)

        idx = torch.empty(nt, k, dtype=torch.int32)
        for i in range(nt):
            perm = torch.randperm(total_experts, generator=rng, device=device)
            idx[i] = perm[:k].to(torch.int32)
        all_rank_indices.append(idx.to(device))

        wts = torch.rand(nt, k, dtype=torch.float32,
                          generator=rng, device=device)
        all_rank_weights.append(wts)

    return num_tok_all, all_rank_indices, all_rank_input, all_rank_weights


# ============================================================
# Reference implementation (mori C++ binding)
# ============================================================
def build_mori_reference_op(cfg: FlyDSLDispatchCombineConfig):
    """Build mori reference op for accuracy comparison."""
    try:
        import mori.ops as mori_ops
        ref_cfg = mori_ops.EpDispatchCombineConfig(
            data_type=cfg.data_type,
            rank=cfg.rank,
            world_size=cfg.world_size,
            hidden_dim=cfg.hidden_dim,
            scale_dim=0,
            scale_type_size=0,
            max_token_type_size=cfg.elem_size,
            max_num_inp_token_per_rank=cfg.max_num_inp_token_per_rank,
            num_experts_per_rank=cfg.num_experts_per_rank,
            num_experts_per_token=cfg.num_experts_per_token,
            warp_num_per_block=cfg.warp_num_per_block,
            block_num=cfg.block_num,
            use_external_inp_buf=True,
            gpu_per_node=cfg.world_size,
        )
        return mori_ops.EpDispatchCombineOp(ref_cfg)
    except Exception as e:
        if cfg.rank == 0:
            print(f"[WARNING] Could not build mori reference op: {e}")
        return None


# ============================================================
# Accuracy test
# ============================================================
def run_accuracy_test(cfg: FlyDSLDispatchCombineConfig, n_rounds: int = 3):
    """Compare FlyDSL op against mori reference op on identical inputs."""
    rank = cfg.rank
    device = torch.device("cuda", rank)
    rng = torch.Generator(device=device)
    # Same seed across all ranks ensures consistent token routing
    rng.manual_seed(42)

    print(f"[PE {rank}] Building FlyDSL op ...")
    flydsl_op = FlyDSLDispatchCombineIntraNodeOp(cfg)

    print(f"[PE {rank}] Building mori reference op ...")
    ref_op = build_mori_reference_op(cfg)
    if ref_op is None:
        if rank == 0:
            print("[WARNING] No reference op, skipping accuracy test")
        return

    dist.barrier()

    for round_idx in range(n_rounds):
        rng.manual_seed(100 + round_idx)
        (num_tok_all, all_indices, all_input, all_weights) = \
            generate_test_data(cfg, rng)

        my_input   = all_input[rank]
        my_indices = all_indices[rank]
        my_weights = all_weights[rank]

        # ---- FlyDSL dispatch ----
        flydsl_op.reset()
        (fly_disp_tok, fly_disp_wts, _, fly_disp_idx,
         fly_recv_num) = flydsl_op.dispatch(
            my_input, my_weights, None, my_indices,
            block_num=cfg.block_num, warp_per_block=cfg.warp_num_per_block)

        # ---- Reference dispatch ----
        ref_op.reset()
        (ref_disp_tok, ref_disp_wts, _, ref_disp_idx,
         ref_recv_num) = ref_op.dispatch(
            my_input, my_weights, None, my_indices,
            block_num=cfg.block_num, warp_per_block=cfg.warp_num_per_block)

        fly_n = fly_recv_num[0].item()
        ref_n = ref_recv_num[0].item()
        assert fly_n == ref_n, \
            f"[PE {rank}] recv count {fly_n} != ref {ref_n}"

        # Verify src_token_pos (sorted, since order may differ)
        fly_src = flydsl_op.get_dispatch_src_token_pos()
        ref_src = ref_op.get_dispatch_src_token_pos()
        src_match = torch.equal(
            torch.sort(fly_src).values, torch.sort(ref_src).values)
        if not src_match and rank == 0:
            print(f"[WARN] round {round_idx}: src_token_pos mismatch")

        # Verify dispatch token content via src_token_pos
        max_tok = cfg.max_num_inp_token_per_rank
        n_recv = int(fly_n)
        fail_tok = 0
        for i in range(n_recv):
            pos = fly_src[i].item()
            src_pe  = pos // max_tok
            src_id  = pos % max_tok
            ref_tok = all_input[src_pe][src_id].to(cfg.data_type)
            got_tok = fly_disp_tok[i]
            if not torch.allclose(ref_tok.float(), got_tok.float(),
                                   atol=1e-2, rtol=1e-2):
                fail_tok += 1
        if fail_tok > 0:
            print(f"[PE {rank}] round {round_idx}: {fail_tok}/{n_recv} "
                  f"dispatch tokens mismatch")
        elif rank == 0:
            print(f"[OK] round {round_idx} dispatch: {n_recv} tokens, "
                  f"content OK, src_pos {'OK' if src_match else 'WARN'}")

        # ---- FlyDSL combine ----
        combine_input = fly_disp_tok[:n_recv].to(cfg.data_type)
        (fly_comb_tok, fly_comb_wts) = flydsl_op.combine(
            combine_input, fly_disp_wts[:n_recv], my_indices,
            block_num=cfg.block_num, warp_per_block=cfg.warp_num_per_block)

        # ---- Reference combine ----
        ref_comb_in = ref_disp_tok[:n_recv].to(cfg.data_type)
        (ref_comb_tok, ref_comb_wts) = ref_op.combine(
            ref_comb_in, ref_disp_wts[:n_recv], my_indices,
            block_num=cfg.block_num, warp_per_block=cfg.warp_num_per_block)

        # Verify combine output
        n_my = len(my_input)
        if n_my > 0:
            max_diff = torch.abs(
                fly_comb_tok[:n_my].float() - ref_comb_tok[:n_my].float()
            ).max().item()
            ok = max_diff < 0.1
            if rank == 0:
                status = "OK" if ok else "FAIL"
                print(f"[{status}] round {round_idx} combine: "
                      f"max_diff={max_diff:.4f} (tol=0.1)")
            if not ok:
                print(f"[PE {rank}] combine output mismatch: max_diff={max_diff}")

        dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Accuracy test completed ({n_rounds} rounds)")
        print(f"{'='*60}")


# ============================================================
# Performance benchmark
# ============================================================
def run_benchmark(cfg: FlyDSLDispatchCombineConfig,
                  warmup: int = 1, iters: int = 10):
    """Benchmark dispatch and combine kernels.

    Methodology follows mori/tests/python/ops/bench_dispatch_combine.py.
    """
    rank = cfg.rank
    device = torch.device("cuda", rank)
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    op = FlyDSLDispatchCombineIntraNodeOp(cfg)

    # Max token load for benchmark
    (_, all_indices, all_input, all_weights) = generate_test_data(
        cfg, rng, use_max_tokens=True)
    my_input   = all_input[rank]
    my_indices = all_indices[rank]
    my_weights = all_weights[rank]

    def run_once():
        op.reset()
        s_disp = torch.cuda.Event(enable_timing=True)
        e_disp = torch.cuda.Event(enable_timing=True)
        s_comb = torch.cuda.Event(enable_timing=True)
        e_comb = torch.cuda.Event(enable_timing=True)

        dist.barrier()
        s_disp.record()
        (disp_tok, disp_wts, _, disp_idx, recv_num) = op.dispatch(
            my_input, my_weights, None, my_indices,
            block_num=cfg.block_num, warp_per_block=cfg.warp_num_per_block)
        e_disp.record()
        torch.cuda.synchronize()

        n_recv = recv_num[0].item()
        comb_in = disp_tok[:n_recv].to(cfg.data_type)

        dist.barrier()
        s_comb.record()
        (comb_tok, _) = op.combine(
            comb_in, None, my_indices,
            block_num=cfg.block_num, warp_per_block=cfg.warp_num_per_block)
        e_comb.record()
        torch.cuda.synchronize()

        return (s_disp.elapsed_time(e_disp),
                s_comb.elapsed_time(e_comb),
                n_recv)

    # Warmup
    for _ in range(warmup):
        run_once()

    # Measurement
    disp_lats, comb_lats = [], []
    for it in range(iters):
        disp_ms, comb_ms, n_recv = run_once()
        disp_us = disp_ms * 1000
        comb_us = comb_ms * 1000

        # Gather all ranks' latencies
        d_list = [torch.zeros(1) for _ in range(cfg.world_size)]
        c_list = [torch.zeros(1) for _ in range(cfg.world_size)]
        dist.all_gather(d_list, torch.tensor([disp_us]))
        dist.all_gather(c_list, torch.tensor([comb_us]))
        disp_lats.append([t.item() for t in d_list])
        comb_lats.append([t.item() for t in c_list])

    if rank == 0:
        esz = cfg.elem_size
        bytes_per_rank = n_recv * cfg.hidden_dim * esz

        print(f"\n{'='*60}")
        print(f"FlyDSL IntraNode Dispatch+Combine Benchmark")
        print(f"  EP={cfg.world_size}, max_tok={cfg.max_num_inp_token_per_rank}, "
              f"hidden={cfg.hidden_dim}, k={cfg.num_experts_per_token}")
        print(f"  block_num={cfg.block_num}, wpb={cfg.warp_num_per_block}")
        print(f"{'='*60}")

        def summarize(label, lats_all):
            min_max_lat = min(max(lats) for lats in lats_all)
            avg_bw = sum(
                bytes_per_rank / 1e9 / (max(lats) / 1e6)
                for lats in lats_all
            ) / len(lats_all)
            print(f"\n{label}:")
            for i, lats in enumerate(lats_all):
                ml = max(lats)
                bw = bytes_per_rank / 1e9 / (ml / 1e6)
                print(f"  iter {i:2d}: max={ml:.0f} μs  "
                      f"bw={bw:.1f} GB/s  "
                      f"all=[{', '.join(f'{l:.0f}' for l in lats)}]")
            print(f"  Best latency: {min_max_lat:.0f} μs  "
                  f"Avg bw: {avg_bw:.1f} GB/s")
            return min_max_lat, avg_bw

        d_lat, d_bw = summarize("Dispatch", disp_lats)
        c_lat, c_bw = summarize("Combine",  comb_lats)
        print(f"\n{'='*60}")
        print(f"Summary: dispatch={d_lat:.0f} μs ({d_bw:.1f} GB/s)  "
              f"combine={c_lat:.0f} μs ({c_bw:.1f} GB/s)  "
              f"total={d_lat+c_lat:.0f} μs")
        print(f"{'='*60}")


# ============================================================
# Worker function (used by both torchrun and spawn)
# ============================================================
def _worker(rank: int, world_size: int, args, master_port: int):
    """Single-process worker. Called by spawn() or directly by torchrun."""
    actual_rank, actual_world = setup_distributed(rank, world_size, master_port)

    total_experts = args.num_experts if args.num_experts else 256
    experts_per_rank = total_experts // actual_world

    cfg = FlyDSLDispatchCombineConfig(
        rank=actual_rank,
        world_size=actual_world,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=args.max_tokens,
        num_experts_per_rank=experts_per_rank,
        num_experts_per_token=args.k,
        data_type=torch.bfloat16,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        chip=args.chip,
    )

    if actual_rank == 0:
        print(f"\n[FlyDSL Dispatch/Combine IntraNode]")
        print(f"  mode={args.mode}, EP={actual_world}, "
              f"max_tok={args.max_tokens}, hidden={args.hidden_dim}, k={args.k}")
        print(f"  experts_per_rank={experts_per_rank}, chip={args.chip}")

    try:
        if args.mode == "test":
            run_accuracy_test(cfg, n_rounds=args.n_rounds)
        else:
            run_benchmark(cfg, warmup=args.warmup, iters=args.iters)
    finally:
        cleanup()


# ============================================================
# Entry point — supports both `python test.py` and `torchrun`
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="FlyDSL Dispatch/Combine IntraNode Test & Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（两种方式等效）:
  # 方式1: python 直接运行（内部自动 spawn N 个进程）
  python tests/kernels/test_bench_dispatch_combine_flydsl.py --mode test
  python tests/kernels/test_bench_dispatch_combine_flydsl.py --mode bench --world-size 4

  # 方式2: torchrun（外部 fork）
  torchrun --nproc_per_node=8 tests/kernels/test_bench_dispatch_combine_flydsl.py --mode test
""")
    parser.add_argument("--mode", choices=["test", "bench"], default="test")
    parser.add_argument("--world-size", type=int, default=8,
                        help="GPU 数量（EP 度），仅 python 直接运行时生效；"
                             "torchrun 模式下由 --nproc_per_node 决定")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=None,
                        help="专家总数（默认 256）")
    parser.add_argument("--k", type=int, default=8, help="top-k 专家数")
    parser.add_argument("--block-num", type=int, default=80)
    parser.add_argument("--warp-per-block", type=int, default=16)
    parser.add_argument("--chip", type=str, default="gfx942")
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="精度测试轮数")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--port", type=int, default=29500,
                        help="分布式通信端口（仅 python 直接运行时使用）")
    return parser.parse_args()


def main():
    args = _parse_args()

    # 判断启动方式：
    # - torchrun 会设置 LOCAL_RANK 环境变量，直接调 _worker
    # - python 直接运行则用 spawn 启动多进程
    if "LOCAL_RANK" in os.environ:
        # ---- torchrun 模式：当前进程就是 worker ----
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        _worker(rank, world_size, args, master_port=args.port)
    else:
        # ---- python 直接运行模式：内部 spawn ----
        world_size = args.world_size
        if world_size > torch.cuda.device_count():
            print(f"[WARNING] --world-size {world_size} > "
                  f"available GPUs {torch.cuda.device_count()}, "
                  f"调整为 {torch.cuda.device_count()}")
            world_size = torch.cuda.device_count()
        print(f"[*] 启动 {world_size} 个进程（python spawn 模式）")
        torch.multiprocessing.spawn(
            _worker,
            args=(world_size, args, args.port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
