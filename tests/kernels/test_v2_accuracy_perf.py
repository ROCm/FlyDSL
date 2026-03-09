"""
FlyDSL v2 Dispatch/Combine IntraNode — 精度 & 性能测试。

测试内容：
  1. 精度测试：与 mori 参考实现对比（dispatch + combine 两个阶段）
  2. 性能测试：测量 dispatch / combine 耗时，与 mori 参考对比

启动方式（两种等效）：
  # spawn 模式（推荐开发调试）
  python tests/kernels/test_v2_accuracy_perf.py --world-size 2

  # torchrun 模式
  torchrun --nproc_per_node=2 tests/kernels/test_v2_accuracy_perf.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

# 路径设置
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in [_ROOT, "/home/yashao/FlyDSL/python", "/home/yashao/mori/python"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mori.shmem as ms
from kernels.dispatch_combine_intranode_op_v2 import (
    FlyDSLDispatchCombineConfigV2,
    FlyDSLDispatchCombineIntraNodeOpV2,
)


# ============================================================
# 分布式初始化
# ============================================================
def setup_distributed(rank, world_size, master_port=29600):
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    # 注册 process group 供 mori shmem 使用
    import torch._C._distributed_c10d as _c10d
    world_group = dist.group.WORLD
    _c10d._register_process_group("default", world_group)
    # 用 torch process group 初始化 mori shmem
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


# ============================================================
# 测试数据生成
# ============================================================
def generate_test_data(rank, world_size, cur_tok, experts_per_token,
                        num_experts, hidden_dim, dtype, device):
    """生成可复现的测试数据（所有 rank 共享相同随机种子）。"""
    torch.manual_seed(42)
    n_exp_total = num_experts
    inp   = torch.randn(cur_tok, hidden_dim, dtype=dtype, device=device)
    wts   = torch.rand(cur_tok, experts_per_token, dtype=torch.float32, device=device)
    # 对权重归一化（sum == 1）
    wts   = wts / wts.sum(dim=-1, keepdim=True)
    # 均匀随机选专家（确保覆盖所有 rank）
    idx   = torch.zeros(cur_tok, experts_per_token, dtype=torch.int32, device=device)
    for tok in range(cur_tok):
        perm = torch.randperm(n_exp_total)[:experts_per_token]
        idx[tok] = perm
    return inp, wts, idx


# ============================================================
# mori 参考算子
# ============================================================
def build_mori_ref(rank, world_size, cfg):
    """构建 mori 参考算子（用于精度对比）。"""
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )
    # bf16 elem_size = 2 bytes
    elem_size = torch.tensor([], dtype=cfg.data_type).element_size()
    mcfg = EpDispatchCombineConfig(
        data_type=cfg.data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=cfg.hidden_dim,
        scale_dim=cfg.num_experts_per_token,
        scale_type_size=4,           # float32 for weights
        max_token_type_size=elem_size,
        max_num_inp_token_per_rank=cfg.max_num_inp_token_per_rank,
        num_experts_per_rank=cfg.num_experts_per_rank,
        num_experts_per_token=cfg.num_experts_per_token,
        warp_num_per_block=cfg.warp_num_per_block,
        block_num=cfg.block_num,
        gpu_per_node=world_size,   # must divide world_size
    )
    return EpDispatchCombineOp(mcfg)


# ============================================================
# 精度测试
# ============================================================
def run_accuracy_test_with_op(cfg, args, op_v2):
    """Wrapper that uses a pre-created op."""
    run_accuracy_test(cfg, args, op_v2=op_v2)


def run_benchmark_with_op(cfg, args, op_v2):
    """Wrapper that uses a pre-created op."""
    run_benchmark(cfg, args, op_v2=op_v2)


def run_accuracy_test(cfg, args, op_v2=None):
    rank       = cfg.rank
    world_size = cfg.world_size
    device     = torch.device("cuda", rank)

    # 构建 v2 算子（若未传入则新建）
    if op_v2 is None:
        op_v2 = FlyDSLDispatchCombineIntraNodeOpV2(cfg)
        ms.shmem_barrier_all()  # ensure all ranks ready

    # 构建 mori 参考
    try:
        op_ref = build_mori_ref(rank, world_size, cfg)
        has_ref = True
    except Exception as e:
        if rank == 0:
            print(f"[warn] mori ref not available: {e}")
        has_ref = False

    if rank == 0:
        print(f"\n[精度测试] world={world_size}, max_tok={cfg.max_num_inp_token_per_rank}, "
              f"hidden={cfg.hidden_dim}, k={cfg.num_experts_per_token}")

    for round_idx in range(args.n_rounds):
        cur_tok = min(
            torch.randint(1, cfg.max_num_inp_token_per_rank + 1, (1,)).item(),
            cfg.max_num_inp_token_per_rank)
        num_experts = cfg.world_size * cfg.num_experts_per_rank

        inp, wts, idx = generate_test_data(
            rank, world_size, cur_tok,
            cfg.num_experts_per_token, num_experts,
            cfg.hidden_dim, cfg.data_type, device)

        # ── v2 dispatch ──────────────────────────────────────────────────────
        op_v2.reset()
        try:
            tok_v2, wts_v2, _, idx_v2, trecv_v2 = op_v2.dispatch(
                inp, wts, None, idx)
        except Exception as e:
            print(f'[rank {rank}] Round {round_idx+1}: dispatch ERROR: {e}', flush=True)
            import traceback; traceback.print_exc()
            break
        n_recv_v2 = int(trecv_v2[0].item())
        src_pos_v2 = op_v2.get_dispatch_src_token_pos()

        # ── mori ref dispatch ────────────────────────────────────────────────
        if has_ref:
            try:
                op_ref.reset()  # reset between rounds
            except Exception:
                pass
            tok_ref, wts_ref, _, idx_ref, trecv_ref = op_ref.dispatch(
                inp, wts, None, idx)
            n_recv_ref = int(trecv_ref[0].item())

            # 比较 total_recv
            dist.all_reduce(trecv_v2, op=dist.ReduceOp.SUM)
            dist.all_reduce(trecv_ref, op=dist.ReduceOp.SUM)
            if rank == 0:
                rv2 = int(trecv_v2[0].item())
                rref = int(trecv_ref[0].item())
                status = "✓" if rv2 == rref else "✗"
                print(f"  Round {round_idx+1}: cur_tok={cur_tok}, "
                      f"total_recv v2={rv2}, ref={rref} {status}")
                if rv2 != rref:
                    print(f"    [WARN] total_recv mismatch: v2={rv2} ref={rref}")

            # 比较 dispatch 输出 token（按 src_token_pos 排序后对比）
            n_cmp = min(n_recv_v2, n_recv_ref)
            if n_cmp > 0:
                # Get source positions for sorting
                src_v2  = op_v2.get_dispatch_src_token_pos()[:n_cmp]
                try:
                    src_ref = op_ref.get_dispatch_src_token_pos()[:n_cmp]
                    # Sort by src position for canonical comparison
                    _, sv2 = src_v2.sort(); _, sr = src_ref.sort()
                    tv2s = tok_v2[:n_cmp][sv2]
                    trs  = tok_ref[:n_cmp][sr]
                    max_diff = (tv2s.float() - trs.float()).abs().max().item()
                except Exception:
                    max_diff = (tok_v2[:n_cmp].float() - tok_ref[:n_cmp].float()).abs().max().item()
                if rank == 0:
                    status = "✓" if max_diff < 1e-1 else "✗"
                    print(f"    dispatch tok max_diff={max_diff:.4f} {status}")
        else:
            if rank == 0:
                print(f"  Round {round_idx+1}: cur_tok={cur_tok}, "
                      f"n_recv={n_recv_v2} (no ref)")

        # ── v2 combine ───────────────────────────────────────────────────────
        # 模拟 expert 处理（简单 identity）
        fake_exp_out = tok_v2  # [n_recv, hidden_dim]
        try:
            out_tok_v2, _ = op_v2.combine(fake_exp_out, None, idx)
        except Exception as e:
            print(f'[rank {rank}] Round {round_idx+1}: combine ERROR: {e}', flush=True)
            import traceback; traceback.print_exc()
            break

        # ── mori ref combine ─────────────────────────────────────────────────
            if has_ref:
                # Create uniform weights (1/k) for fair comparison
                k = cfg.num_experts_per_token
                n_ref = tok_ref.shape[0]
                wts_ref = torch.ones(n_ref, k, dtype=torch.float32, device=device) / k
                out_tok_ref, _ = op_ref.combine(tok_ref, wts_ref, idx_ref)
                n_cmp = min(out_tok_v2.shape[0], out_tok_ref.shape[0])
                if n_cmp > 0:
                    # v2 accumulates raw sum; ref uses weights (1/k each)
                    # Normalize v2 output by 1/k for comparison
                    v2_normalized = out_tok_v2[:n_cmp].float() / k
                    max_diff_c = (v2_normalized - out_tok_ref[:n_cmp].float()).abs().max().item()
                    if rank == 0:
                        status = "✓" if max_diff_c < 0.5 else "✗"
                        print(f"    combine  out max_diff(v2/k vs ref)={max_diff_c:.4f} {status}")

    if rank == 0:
        print("[精度测试] 完成")


# ============================================================
# 性能测试
# ============================================================
def run_benchmark(cfg, args, op_v2=None):
    rank   = cfg.rank
    device = torch.device("cuda", rank)

    if op_v2 is None:
        op_v2 = FlyDSLDispatchCombineIntraNodeOpV2(cfg)
        ms.shmem_barrier_all()
    try:
        op_ref = build_mori_ref(rank, cfg.world_size, cfg)
        has_ref = True
    except Exception:
        has_ref = False

    cur_tok = cfg.max_num_inp_token_per_rank
    num_experts = cfg.world_size * cfg.num_experts_per_rank
    inp, wts, idx = generate_test_data(
        rank, cfg.world_size, cur_tok,
        cfg.num_experts_per_token, num_experts,
        cfg.hidden_dim, cfg.data_type, device)

    def timed_run(op, n_iter=args.iters, warmup=args.warmup):
        for _ in range(warmup):
            op.reset()
            tok, w, _, i, tr = op.dispatch(inp, wts, None, idx)
            out, _ = op.combine(tok, None, idx)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)

        disp_times = []
        comb_times = []
        for _ in range(n_iter):
            op.reset()
            t0.record()
            tok, w, _, i, tr = op.dispatch(inp, wts, None, idx)
            t1.record()
            out, _ = op.combine(tok, None, idx)
            t2.record()
            torch.cuda.synchronize()
            disp_times.append(t0.elapsed_time(t1))
            comb_times.append(t1.elapsed_time(t2))

        avg_d = sum(disp_times) / len(disp_times)
        avg_c = sum(comb_times) / len(comb_times)
        return avg_d, avg_c

    ms.shmem_barrier_all()   # wait for all ranks to finish compilation
    if rank == 0:
        print(f"\n[性能测试] world={cfg.world_size}, max_tok={cur_tok}, "
              f"hidden={cfg.hidden_dim}, k={cfg.num_experts_per_token}, "
              f"block_num={cfg.block_num}, wpb={cfg.warp_num_per_block}")

    # v2 benchmark
    d_v2, c_v2 = timed_run(op_v2)

    # mori ref benchmark
    if has_ref:
        d_ref, c_ref = timed_run(op_ref)
    else:
        d_ref = c_ref = float("nan")

    # 汇总各 rank 结果
    results = torch.tensor([d_v2, c_v2, d_ref, c_ref], device=device)
    dist.all_reduce(results, op=dist.ReduceOp.MAX)   # 取最慢 rank

    if rank == 0:
        d_v2_r, c_v2_r, d_ref_r, c_ref_r = results.tolist()
        print(f"  dispatch  v2={d_v2_r:.3f} ms  ref={d_ref_r:.3f} ms  "
              f"(speedup {d_ref_r/d_v2_r:.2f}x)" if not (d_ref_r != d_ref_r) else
              f"  dispatch  v2={d_v2_r:.3f} ms  (no ref)")
        print(f"  combine   v2={c_v2_r:.3f} ms  ref={c_ref_r:.3f} ms  "
              f"(speedup {c_ref_r/c_v2_r:.2f}x)" if not (c_ref_r != c_ref_r) else
              f"  combine   v2={c_v2_r:.3f} ms  (no ref)")
        print("[性能测试] 完成")

    del op_v2


# ============================================================
# Worker
# ============================================================
def _worker(rank, world_size, args, master_port):
    actual_rank, actual_world = setup_distributed(rank, world_size, master_port)
    device = torch.device("cuda", actual_rank)

    n_experts = args.num_experts or (256 if actual_world <= 8 else actual_world * 32)
    experts_per_rank = n_experts // actual_world

    cfg = FlyDSLDispatchCombineConfigV2(
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

    try:
        # Create op ONCE and reuse for both test and bench
        op_v2 = FlyDSLDispatchCombineIntraNodeOpV2(cfg)
        # Barrier: ensure ALL ranks finished compilation before starting tests
        ms.shmem_barrier_all()

        if args.mode in ("test", "both"):
            run_accuracy_test_with_op(cfg, args, op_v2)
        if args.mode in ("bench", "both"):
            run_benchmark_with_op(cfg, args, op_v2)
    except Exception as e:
        print(f"[rank {actual_rank}] ERROR: {e}")
        traceback.print_exc()
    finally:
        cleanup()


# ============================================================
# 入口
# ============================================================
def _parse_args():
    p = argparse.ArgumentParser(description="FlyDSL v2 dispatch/combine 精度 & 性能测试")
    p.add_argument("--mode", choices=["test", "bench", "both"], default="test")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--k", type=int, default=2, help="top-k 专家数")
    p.add_argument("--block-num", type=int, default=16)
    p.add_argument("--warp-per-block", type=int, default=4)
    p.add_argument("--chip", type=str, default="gfx942")
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--port", type=int, default=29600)
    return p.parse_args()


def main():
    args = _parse_args()

    if "LOCAL_RANK" in os.environ:
        rank       = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        _worker(rank, world_size, args, master_port=args.port)
    else:
        world_size = min(args.world_size, torch.cuda.device_count())
        if world_size < args.world_size:
            print(f"[warn] 可用 GPU={torch.cuda.device_count()}, "
                  f"调整 world_size: {args.world_size} → {world_size}")
        print(f"[*] 启动 {world_size} 个进程（spawn 模式）")
        torch.multiprocessing.spawn(
            _worker,
            args=(world_size, args, args.port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
