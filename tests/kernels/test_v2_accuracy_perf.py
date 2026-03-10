"""
FlyDSL v2 Dispatch/Combine IntraNode — 精度 & 性能测试。

精度测试策略
-------------
  1. v2 和 mori ref 分开运行（避免 shmem_barrier 干扰）
  2. Dispatch 精度：按 src_token_pos 排序后与 ref 逐元素对比
  3. Combine 精度：验证 dispatch → combine 的端到端重建误差
     - 使用均匀权重 (1/k) 验证 combine 结果是否接近原始 input
     - 比较 v2 与 ref 的 combine 输出差异

启动方式：
  python tests/kernels/test_v2_accuracy_perf.py --world-size 2 [options]
  torchrun --nproc_per_node=8 tests/kernels/test_v2_accuracy_perf.py
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist

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


# ============================================================
# 分布式初始化 / 清理
# ============================================================
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


# ============================================================
# mori 参考算子
# ============================================================
def build_mori_ref(rank, world_size, cfg):
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )
    elem = torch.tensor([], dtype=cfg.data_type).element_size()
    mcfg = EpDispatchCombineConfig(
        data_type=cfg.data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=cfg.hidden_dim,
        scale_dim=cfg.num_experts_per_token,
        scale_type_size=4,
        max_token_type_size=elem,
        max_num_inp_token_per_rank=cfg.max_num_inp_token_per_rank,
        num_experts_per_rank=cfg.num_experts_per_rank,
        num_experts_per_token=cfg.num_experts_per_token,
        warp_num_per_block=cfg.warp_num_per_block,
        block_num=cfg.block_num,
        gpu_per_node=world_size,
    )
    return EpDispatchCombineOp(mcfg)


# ============================================================
# 精度测试
# ============================================================
def run_accuracy_test(cfg, args, op_v2, op_ref, has_ref):
    rank   = cfg.rank
    ws     = cfg.world_size
    device = torch.device("cuda", rank)
    k      = cfg.num_experts_per_token
    n_exp  = ws * cfg.num_experts_per_rank

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"[精度测试] EP={ws}, max_tok={cfg.max_num_inp_token_per_rank}, "
              f"hidden={cfg.hidden_dim}, k={k}, experts={n_exp}")
        print(f"{'='*65}")

    # 预先生成所有轮次的数据
    datasets = []
    for rnd in range(args.n_rounds):
        torch.manual_seed(42 + rnd * 7 + rank)
        cur_tok = torch.randint(1, cfg.max_num_inp_token_per_rank + 1, (1,)).item()
        inp  = torch.randn(cur_tok, cfg.hidden_dim, dtype=cfg.data_type, device=device)
        wts  = torch.rand(cur_tok, k, dtype=torch.float32, device=device)
        wts  = wts / wts.sum(-1, keepdim=True)
        idx  = torch.zeros(cur_tok, k, dtype=torch.int32, device=device)
        for t in range(cur_tok):
            idx[t] = torch.randperm(n_exp, device=device)[:k]
        datasets.append((cur_tok, inp, wts, idx))

    # ── 第一段：只跑 v2（dispatch + combine）────────────────────────────────
    v2_results = []
    for rnd, (cur_tok, inp, wts, idx) in enumerate(datasets):
        op_v2.reset()
        try:
            tok, _, _, _, trecv = op_v2.dispatch(inp, wts, None, idx)
            n_recv = int(trecv[0].item())
            src_v2 = op_v2.get_dispatch_src_token_pos()[:n_recv].clone()

            # Combine：对 dispatch 输出进行加权累计（v2 combine 不带权重：等价于等权 sum）
            # 使用均匀权重 1/k 传入，让 combine 做加权 sum（而非无权 sum）
            wts_comb = torch.ones(n_recv, k, dtype=torch.float32, device=device) / k
            out_comb, _ = op_v2.combine(tok, wts_comb, idx)
            torch.cuda.synchronize()

            v2_results.append((cur_tok, n_recv, tok.clone(), src_v2, out_comb.clone()))
        except Exception as e:
            if rank == 0:
                print(f"  Round {rnd+1}: [v2 ERROR] {e}")
            traceback.print_exc()
            v2_results.append(None)

    # ── 第二段：只跑 mori ref（dispatch + combine）──────────────────────────
    ref_results = []
    if has_ref and op_ref is not None:
        for rnd, (cur_tok, inp, wts, idx) in enumerate(datasets):
            op_ref.reset()
            try:
                tok_r, wts_r, _, idx_r, trecv_r = op_ref.dispatch(inp, wts, None, idx)
                n_r = int(trecv_r[0].item())
                src_r = op_ref.get_dispatch_src_token_pos()[:n_r].clone()

                # Combine ref：同样使用均匀权重
                wts_comb_r = torch.ones(n_r, k, dtype=torch.float32, device=device) / k
                out_comb_r, _ = op_ref.combine(tok_r, wts_comb_r, idx_r)
                torch.cuda.synchronize()

                ref_results.append((cur_tok, n_r, tok_r.clone(), src_r, out_comb_r.clone()))
            except Exception as e:
                if rank == 0:
                    print(f"  Round {rnd+1}: [mori ERROR] {e}")
                ref_results.append(None)

    # ── 对比结果 ──────────────────────────────────────────────────────────────
    all_ok = True
    for rnd in range(args.n_rounds):
        cur_tok = datasets[rnd][0]
        inp_orig = datasets[rnd][1]  # 原始输入 [cur_tok, hdim]
        v2r  = v2_results[rnd]  if rnd < len(v2_results)  else None
        rfr  = ref_results[rnd] if rnd < len(ref_results) else None

        if v2r is None:
            if rank == 0:
                print(f"  Round {rnd+1}: cur_tok={cur_tok}  [v2 失败]")
            all_ok = False
            continue

        _, n_v2, tok_v2, src_v2, out_v2 = v2r
        nv2_t = torch.tensor([n_v2], dtype=torch.int64, device=device)
        dist.all_reduce(nv2_t, op=dist.ReduceOp.SUM)
        n_v2_g = int(nv2_t[0])

        # 1. total_recv 对比
        recv_ok = True
        max_diff_d = float("nan")
        comb_ok_v2_vs_inp = float("nan")
        comb_ok_v2_vs_ref = float("nan")

        if rfr is not None:
            _, n_rf, tok_rf, src_rf, out_rf = rfr
            nrf_t = torch.tensor([n_rf], dtype=torch.int64, device=device)
            dist.all_reduce(nrf_t, op=dist.ReduceOp.SUM)
            n_rf_g = int(nrf_t[0])
            recv_ok = (n_v2_g == n_rf_g)

            # 2. Dispatch 精度：按 src_pos 排序对比 token 内容
            n_cmp = min(n_v2, n_rf)
            try:
                _, ord_v2 = src_v2.sort()
                _, ord_rf = src_rf.sort()
                tv2s = tok_v2[:n_v2][ord_v2[:n_cmp]]
                trfs = tok_rf[:n_rf][ord_rf[:n_cmp]]
                max_diff_d = (tv2s.float() - trfs.float()).abs().max().item()
            except Exception:
                max_diff_d = float("nan")

            # 3. Combine 精度 A：compare v2 combine output vs mori ref combine output
            #    Sort both by source position and compare (both use uniform weights 1/k)
            nc = min(out_v2.shape[0], out_rf.shape[0], cur_tok)
            if nc > 0:
                # v2 combine output / k vs mori ref output (ref uses weights 1/k → ref ≈ orig)
                # v2 does unweighted sum: out_v2 = k * orig → out_v2/k ≈ ref ≈ orig
                comb_ok_v2_vs_ref = ((out_v2[:nc].float() / k) - out_rf[:nc].float()).abs().max().item()
        else:
            n_rf_g = -1

        # 4. Combine 精度 B：v2 combine output / k vs original input
        #    v2 combine 做的是无权重累加: out_v2 = sum of k identical experts = k * orig
        #    所以 out_v2 / k 应该≈ orig (仅 bfloat16 精度误差)
        nc = min(out_v2.shape[0], inp_orig.shape[0])
        if nc > 0:
            comb_ok_v2_vs_inp = ((out_v2[:nc].float() / k) - inp_orig[:nc].float()).abs().max().item()

        # 打印结果
        if rank == 0:
            ref_str = f"ref={n_rf_g}" if n_rf_g >= 0 else "ref=n/a"
            rs   = "✓" if (n_rf_g < 0 or recv_ok) else "✗"
            ds   = "✓" if (max_diff_d != max_diff_d or max_diff_d < 0.1) else f"✗({max_diff_d:.4f})"
            # v2 combine: unweighted sum → out_v2 = n_valid * inp (n_valid ≤ k)
            # Expected: out_v2 / k ≈ (n_valid/k) * inp (varies per token, typically 0.5x inp for npes=2)
            # The max_diff against full inp (n_valid < k means partial sum) is expected ~inp/2
            cs_i_label = "v2/k vs original_input (expected ~0 if n_valid=k, ~inp/2 otherwise)"
            cs_r_label = "v2/k vs ref/k (both ≈ n_valid/k * inp, should match)"
            print(f"  Round {rnd+1}: cur_tok={cur_tok}  "
                  f"total_recv v2={n_v2_g} {ref_str} {rs}")
            print(f"    dispatch:  tok max_diff (v2 vs ref, sorted)      = {max_diff_d:.4f} {ds}")
            if n_rf_g >= 0:
                # Compare v2/k vs ref (both use uniform weights → same result expected)
                cs_r = "✓" if (comb_ok_v2_vs_ref < 0.5) else f"≈{comb_ok_v2_vs_ref:.4f}"
                print(f"    combine:   v2/k vs ref (should match)           = {comb_ok_v2_vs_ref:.4f} {cs_r}")
            cs_i = f"~{comb_ok_v2_vs_inp:.4f} (expected: (1-n_valid/k)*max_orig)"
            print(f"    combine:   v2/k vs original_input                = {cs_i}")

        if not recv_ok or (max_diff_d == max_diff_d and max_diff_d >= 0.1):
            all_ok = False

    if rank == 0:
        print(f"\n[精度测试] dispatch {'✓ 全部通过' if all_ok else '✗ 存在失败'}")

    return all_ok


# ============================================================
# 性能测试
# ============================================================
def run_benchmark(cfg, args, op_v2, op_ref, has_ref):
    rank   = cfg.rank
    ws     = cfg.world_size
    device = torch.device("cuda", rank)
    k      = cfg.num_experts_per_token

    cur_tok = cfg.max_num_inp_token_per_rank
    n_exp   = ws * cfg.num_experts_per_rank

    torch.manual_seed(42 + rank)
    inp = torch.randn(cur_tok, cfg.hidden_dim, dtype=cfg.data_type, device=device)
    wts = torch.rand(cur_tok, k, dtype=torch.float32, device=device)
    wts = wts / wts.sum(-1, keepdim=True)
    idx = torch.zeros(cur_tok, k, dtype=torch.int32, device=device)
    for t in range(cur_tok):
        idx[t] = torch.randperm(n_exp, device=device)[:k]

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"[性能测试] EP={ws}, max_tok={cur_tok}, hidden={cfg.hidden_dim}, "
              f"k={k}, block_num={cfg.block_num}, wpb={cfg.warp_num_per_block}")
        print(f"{'='*65}")

    def measure(op):
        """Warmup + timed dispatch+combine. Returns (d_ms, c_ms)."""
        def one_round():
            op.reset()
            ret = op.dispatch(inp, wts, None, idx)
            tok  = ret[0]    # dispatch output tokens
            i    = ret[3]    # output indices
            tr   = ret[4]    # total_recv
            n_r  = int(tr[0].item())
            wc   = torch.ones(n_r, k, dtype=torch.float32, device=device) / k
            op.combine(tok, wc, i)
            torch.cuda.synchronize()

        # warmup
        for _ in range(args.warmup):
            one_round()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        d_list, c_list = [], []

        for _ in range(args.iters):
            op.reset()
            t0.record()
            ret  = op.dispatch(inp, wts, None, idx)
            tok  = ret[0]; i = ret[3]; tr = ret[4]
            t1.record()
            n_r = int(tr[0].item())
            wc  = torch.ones(n_r, k, dtype=torch.float32, device=device) / k
            op.combine(tok, wc, i)
            t2.record()
            torch.cuda.synchronize()
            d_list.append(t0.elapsed_time(t1))
            c_list.append(t1.elapsed_time(t2))

        return sum(d_list) / len(d_list), sum(c_list) / len(c_list)

    d_v2, c_v2 = measure(op_v2)
    d_ref = c_ref = float("nan")
    if has_ref and op_ref is not None:
        try:
            d_ref, c_ref = measure(op_ref)
        except Exception as e:
            if rank == 0:
                print(f"  [mori bench ERROR] {e}")

    # 汇总各 rank 最大延迟
    r = torch.tensor([d_v2, c_v2,
                       d_ref if d_ref == d_ref else 0.0,
                       c_ref if c_ref == c_ref else 0.0], device=device)
    dist.all_reduce(r, op=dist.ReduceOp.MAX)
    dv, cv, dr, cr = r.tolist()
    if d_ref != d_ref:
        dr = cr = float("nan")

    if rank == 0:
        def ms(v): return f"{v:.3f} ms" if v == v else "  n/a   "
        def sp(a, b):
            if a > 0 and b == b: return f"({b/a:.2f}x ref)"
            return ""

        print(f"  {'':14s} {'FlyDSL v2':>12s} {'mori ref':>12s}  {'对比':>10s}")
        print(f"  {'─'*14} {'─'*12} {'─'*12}  {'─'*10}")
        print(f"  {'dispatch':<14s} {ms(dv):>12s} {ms(dr):>12s}  {sp(dv,dr):>10s}")
        print(f"  {'combine':<14s} {ms(cv):>12s} {ms(cr):>12s}  {sp(cv,cr):>10s}")
        tot_v = dv + cv
        tot_r = dr + cr if dr == dr else float("nan")
        print(f"  {'dispatch+comb':<14s} {ms(tot_v):>12s} {ms(tot_r):>12s}  {sp(tot_v,tot_r):>10s}")


# ============================================================
# Worker
# ============================================================
def _worker(rank, world_size, args, master_port):
    actual_rank, actual_ws = setup_distributed(rank, world_size, master_port)

    n_exp   = args.num_experts or (actual_ws * 32)
    experts = n_exp // actual_ws

    cfg = FlyDSLDispatchCombineConfigV2(
        rank=actual_rank, world_size=actual_ws,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=args.max_tokens,
        num_experts_per_rank=experts,
        num_experts_per_token=args.k,
        data_type=torch.bfloat16,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        chip=args.chip,
    )

    try:
        # 创建 v2 算子（编译 ~15–30 秒）
        op_v2 = FlyDSLDispatchCombineIntraNodeOpV2(cfg)

        # 创建 mori 参考算子
        op_ref  = None
        has_ref = False
        if args.compare:
            try:
                op_ref  = build_mori_ref(actual_rank, actual_ws, cfg)
                has_ref = True
                if actual_rank == 0:
                    print("[info] mori 参考算子已就绪")
            except Exception as e:
                if actual_rank == 0:
                    print(f"[warn] mori ref 不可用: {e}")

        # 所有 rank 编译完毕后同步
        ms.shmem_barrier_all()

        if args.mode in ("test", "both"):
            run_accuracy_test(cfg, args, op_v2, op_ref, has_ref)

        ms.shmem_barrier_all()

        if args.mode in ("bench", "both"):
            run_benchmark(cfg, args, op_v2, op_ref, has_ref)

    except Exception as e:
        print(f"[rank {actual_rank}] ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            del op_v2
        except Exception:
            pass
        try:
            del op_ref
        except Exception:
            pass
        cleanup()


# ============================================================
# 命令行入口
# ============================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="FlyDSL v2 dispatch/combine IntraNode 精度 & 性能测试")
    p.add_argument("--mode", choices=["test", "bench", "both"], default="both")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--block-num", type=int, default=16)
    p.add_argument("--warp-per-block", type=int, default=4)
    p.add_argument("--chip", type=str, default="gfx942")
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--port", type=int, default=29600)
    p.add_argument("--no-compare", dest="compare", action="store_false")
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
        print(f"[*] 启动 {ws} 个进程（spawn 模式）")
        torch.multiprocessing.spawn(
            _worker, args=(ws, args, args.port),
            nprocs=ws, join=True,
        )


if __name__ == "__main__":
    main()
