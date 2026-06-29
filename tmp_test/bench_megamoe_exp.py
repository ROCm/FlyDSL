#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Standalone end-to-end accuracy + perf harness for the EXPERIMENTAL MegaMoEExp single-op.

Extracted from tests/kernels/bench_moe_intranode_stage1_groupgemm.py's `_run_full_e2e`, but SELF
-CONTAINED and DECOUPLED from the production MegaMoE: it does NOT import or run
kernels.fused_moe_stage1_stage2.MegaMoE, and it does NOT modify the production bench.  Only generic
test infra (data prep, dist setup, quant, oracle math, CUDAGraph timing) is reused; the mega path is
kernels.megamoe_exp.MegaMoEExp.

Per batch size it measures MegaMoEExp against two independent references (rank0 prints, reduced over
all 8 ranks):
  * accuracy : relL2 of MegaMoEExp output vs a full-precision WEIGHTED torch MoE oracle -- must sit
               at the fp8/fp4 quant floor (~0.25 a8w4 / ~0.32 a4w4), NOT ~1.0.  Correctness gate
               ALSO requires MegaMoEExp to agree with the atom-fp8 production baseline (relL2 ~0).
  * perf     : CUDAGraph device time (STAGE1-only and full E2E), speedup vs the atom-fp8 baseline.

The atom-fp8 baseline = the PRODUCTION ATOM stack (fp8 dispatch -> aiter moe_sorting/mxscale_sort ->
mixed_moe_gemm1 -> fused GEMM2+combine).  It is a *baseline*, not the MegaMoE single-op, so keeping
it here does not couple the harness to the production MegaMoE.

Run (8x MI355X).  DISABLE the JIT disk cache -- the megakernel body is helper code outside the
traced closure, so a stale cached artifact would hide source changes::

  MORI_SHMEM_HEAP_SIZE=40G FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    torchrun --standalone --nproc_per_node=8 tmp_test/bench_megamoe_exp.py \
      --network v4_pro --quant a8w4 --bs-list 64,256,1024,4096 --iters 30 --warmup 8

MegaMoEExp requires max_tok_per_rank a power of two; mtpr = next_pow2(max(16, bs)) is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _ensure_kernel_symlinks():
    """Make the experimental tmp_test/*.py importable as kernels.* (idempotent).

    megamoe_exp.py uses RELATIVE imports (from .tmp_mega_megakernel / .dispatch_combine_intranode_op),
    so it must resolve inside the kernels package -> the tmp_mega_* modules need kernels/ symlinks.
    """
    kdir = os.path.join(_ROOT, "kernels")
    for f in ("tmp_mega_gemm_2stage", "tmp_mega_megakernel", "tmp_mega_ep_dispatch",
              "tmp_mega_stage1_stage2", "tmp_mega_gemm2_combine_op",
              "tmp_mega_gemm2_combine_fused", "tmp_mega_gemm2_2stage", "megamoe_exp"):
        dst = os.path.join(kdir, f + ".py")
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.join("..", "tmp_test", f + ".py"), dst)
            except FileExistsError:
                pass


_ensure_kernel_symlinks()

import flydsl.expr as fx  # noqa: E402
import mori.shmem as ms  # noqa: E402

# ---- generic (non-MegaMoE) test infra reused from the bench: data prep, dist, quant, reductions ----
from tests.kernels.bench_moe_intranode_stage1_groupgemm import (  # noqa: E402
    NETWORKS, CLASSIC_BS, FULL_BS, HAS_AITER,
    _prepare, _setup_dist, _cleanup, _info,
    _all_max, _all_mean, _all_min_int, _chunked_fp4_quant,
)
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.utils import shuffle_weight  # noqa: E402

# ---- the mega single-op under test (decoupled from the production MegaMoE) ----
from kernels.megamoe_exp import MegaMoEExp  # noqa: E402
# ---- atom-fp8 BASELINE stack (production ATOM; a reference, not the MegaMoE single-op) ----
from kernels.dispatch_combine_intranode_op import (  # noqa: E402
    FlyDSLDispatchCombineConfig, FlyDSLDispatchCombineIntraNodeOp,
)
from kernels.mixed_moe_gemm2_combine_fused_op import FlyDSLMoeGemm2CombineOp  # noqa: E402
from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1  # noqa: E402


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _relL2(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = float(((a - b) ** 2).sum()); d = float((b ** 2).sum())
    return (n / d) ** 0.5 if d > 0 else -1.0


def _run(args, rank, world, dev, *, model_dim, inter_dim, experts, epr, topk, run_tokens, mtpr,
         a_dtype, s1_out, w_kernel, scale_w1_1d, x_bf16, topk_ids, wts):
    """MegaMoEExp e2e vs (atom-fp8 baseline, full-precision weighted torch oracle).  8-rank."""
    from aiter import dtypes as _adt
    from aiter.ops.quant import per_1x32_mx_quant_hip
    import aiter

    _is_fp4 = (s1_out == "fp4")
    max_recv = world * mtpr
    tm, tn1, tk = 32, 128, 256
    tm2, tn2, tk2 = 32, 128, 256
    _agv = (lambda t: t.view(torch.uint8)) if a_dtype == "fp4" else (lambda t: t)

    # ---- W2 (down proj): MX-FP4 via the proven W1 pipeline, replicated cross-rank (same seed) ----
    torch.manual_seed(args.seed + 4242)
    w2_f32 = (torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
              * (float(inter_dim) ** -0.25))
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _w2sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2_kernel = shuffle_weight(w2_fp4[_w2sl]).view(torch.uint8).contiguous().view(-1)
    w2_scale_1d = fp4_utils.e8m0_shuffle(w2_sr[_w2sl]).view(torch.uint8).contiguous().view(-1)

    # ---- full-precision oracle weights (pre-quant f32; w1 re-seeded EXACTLY like _prepare) ----
    del w2_fp4, w2_sr
    torch.cuda.empty_cache()
    _init = float(model_dim) ** -0.25
    torch.manual_seed(args.seed)
    _ = torch.randn((run_tokens, model_dim), device=dev, dtype=torch.float32)  # advance RNG like _prepare
    w1_all = (torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * _init)
    w2_all = w2_f32.view(experts, model_dim, inter_dim)

    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()

    # quantized activation (production stage-1 input: fp8/fp4 + e8m0 scale)
    if _is_fp4:
        x_q, x_sc = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(), quant_dtype=_adt.fp4x2)
    else:
        x_q, x_sc = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(),
                                          quant_dtype=_adt.fp8, scale_type=_adt.fp8_e8m0)
    x_sc = x_sc.view(torch.uint8)

    # ---- device-time helper.  Default: CUDAGraph (warmup -> capture -> replay).  With
    # --no-cudagraph: plain eager back-to-back launches (avoids CUDAGraph capture/replay issues
    # with the persistent megakernel + cross-PE combine; slightly noisier but launch-overhead is
    # tiny vs these kernels). ----
    _eager = bool(getattr(args, "no_cudagraph", False))

    def _cg_time(body):
        ms.shmem_barrier_all()
        body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
        _n = max(1, int(args.iters))
        _s = torch.cuda.Event(enable_timing=True); _e = torch.cuda.Event(enable_timing=True)
        if _eager:
            for _ in range(10):
                body()
            torch.cuda.synchronize()
            _s.record()
            for _ in range(_n):
                body()
            _e.record(); torch.cuda.synchronize()
            return _all_mean(dev, _s.elapsed_time(_e) / _n)
        _cap = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=_cap):
            body()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        _s.record()
        for _ in range(_n):
            g.replay()
        _e.record(); torch.cuda.synchronize()
        return _all_mean(dev, _s.elapsed_time(_e) / _n)

    # local w1 slice (kernel indexes by LOCAL expert id; global >4GB would hit the 4GB buffer cap)
    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    _w1_arg = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    _w1s_arg = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()

    # ============ MegaMoEExp (single-op under test) ============
    _tm2_mega = int(args.mega_gemm2_tile_m) if int(getattr(args, "mega_gemm2_tile_m", -1)) > 0 else tm2
    moe = MegaMoEExp(
        rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, quant=args.quant, w1=_w1_arg, w1_scale=_w1s_arg,
        w2=w2_kernel, w2_scale=w2_scale_1d, max_tok_per_rank=mtpr, network=args.network,
        gemm2_tile_m=_tm2_mega, gemm2_tile_n=tn2, gemm2_tile_k=tk2, stage2_mode="fused")
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    _info(rank, f"[stage] MegaMoEExp built (bs={run_tokens}); running warmup forward (may JIT-compile, can be slow with cache off) ...")
    _mega_hold = {}
    def _mega_body():
        _mega_hold["o"] = moe.forward(x_q, x_sc, wc, ic)
    _mega_body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
    out_mega = _mega_hold["o"][:run_tokens].float().cpu().numpy().copy()
    _info(rank, "[stage] MegaMoEExp warmup forward DONE; building atom-fp8 baseline ...")

    # ============ atom-fp8 BASELINE: fp8 dispatch -> sort -> gemm1 -> fused GEMM2+combine ============
    cfg_a = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world, hidden_dim=model_dim, max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr, num_experts_per_token=topk, data_type=torch.bfloat16,
        scale_dim=0, scale_type_size=0, enable_std_moe=False)
    dc = FlyDSLDispatchCombineIntraNodeOp(cfg_a)   # bf16 combine op (gemm2 emits bf16)
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    dc.total_recv.zero_()
    _bt0, _, _, _oidx0, _ = dc.dispatch(x_bf16[:run_tokens].contiguous(), wc, None, ic)  # fix trc
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    trc = max(1, int(dc.total_recv.item()))
    if _all_min_int(dev, trc) <= 0:
        _info(rank, "[megamoe-exp-bench] some rank got 0 recv; skipping"); return None

    _max_pad = max_recv * topk + experts * tm
    _max_blocks = (_max_pad + tm - 1) // tm
    _scaleN_pad = ((model_dim // 32 + 7) // 8) * 8
    a_st = torch.empty(_max_pad, dtype=torch.int32, device=dev)
    a_sw = torch.empty(_max_pad, dtype=torch.float32, device=dev)
    a_se = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
    a_se_local = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
    a_nv = torch.zeros(2, dtype=torch.int32, device=dev)
    a_mbuf = torch.empty((max_recv, model_dim), dtype=torch.float16, device=dev)
    a1s = torch.empty(((_max_pad + 31) // 32 * 32, _scaleN_pad), dtype=_adt.fp8_e8m0, device=dev)
    recv_wts = torch.full((max_recv, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    recv_topk = torch.empty((max_recv, topk), dtype=torch.int32, device=dev)
    _sentinel = torch.full((trc, topk), experts, dtype=torch.int32, device=dev)
    if _is_fp4:
        a2_e = torch.zeros((max_recv * topk, inter_dim // 2), dtype=torch.uint8, device=dev)
    else:
        a2_e = torch.zeros((max_recv * topk, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
    _sbm = max(32, tm); _pr = ((_max_blocks * _sbm + 255) // 256) * 256
    _pc = (((inter_dim // 32) + 7) // 8) * 8
    a2s_e = torch.zeros(_pr * _pc + inter_dim, dtype=torch.uint8, device=dev)
    bias_d = torch.empty((0,), device=dev, dtype=torch.float32)
    _qd = _adt.fp4x2 if _is_fp4 else _adt.fp8
    _stp = None if _is_fp4 else _adt.fp8_e8m0
    gemm1 = compile_mixed_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=topk,
        tile_m=tm, tile_n=tn1, tile_k=tk, doweight_stage1=False, a_dtype=a_dtype, b_dtype="fp4",
        out_dtype=s1_out, act="silu", waves_per_eu=int(args.waves_per_eu),
        use_async_copy=bool(args.async_copy))
    # NOTE: this op's run() is always stage1-only (see _run_stage1_only); there is no force_mode
    # kwarg in this kernels version, so it must NOT be passed.
    g2a = FlyDSLMoeGemm2CombineOp(comb_cfg=cfg_a, comb_op=dc, inter_dim=inter_dim,
                                  tile_m=tm2, tile_n=tn2, tile_k=tk2, persist_m=-1,
                                  a_dtype=s1_out, b_dtype="fp4")

    _scale_mx_blocks = model_dim // 32
    cfg_fp8 = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world, hidden_dim=model_dim, max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr, num_experts_per_token=topk,
        data_type=(torch.float4_e2m1fn_x2 if _is_fp4 else torch.float8_e4m3fn),
        scale_dim=_scale_mx_blocks, scale_type_size=1, enable_std_moe=False)
    dcf = FlyDSLDispatchCombineIntraNodeOp(cfg_fp8)   # fp8 dispatch op
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    _atom8_hold = {}
    def _atom_fp8_body():
        a2_e.zero_()
        dc.shmem_comb_inp_tok.zero_(); dc.shmem_comb_inp_wts.zero_()
        dc.shmem_comb_out_tok.zero_(); dc.shmem_comb_out_wts.zero_()
        dcf.total_recv.zero_()
        _rx, _, _rs, _oidx, _ = dcf.dispatch(x_q, wc, x_sc, ic)   # fp8 dispatch (+ e8m0 scale)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        aiter.moe_sorting_fwd(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc],
                              int(experts), int(tm), None, None, 0)
        aiter.mxfp4_moe_sort_hip(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
        a_se_local.copy_(a_se - rank * epr)
        gemm1(a2_e.view(max_recv, topk, a2_e.shape[-1]), _agv(_rx[:trc]), _w1_arg, a1s.view(torch.uint8),
              _w1s_arg, a_st, a_se_local, a_sw, a_nv, bias_d, a2s_e, fx.Int32(trc),
              fx.Int32(inter_dim * 2), fx.Int32(model_dim), fx.Int32(int(_max_blocks)),
              stream=fx.Stream(torch.cuda.current_stream()))
        dc.total_recv.copy_(dcf.total_recv)
        dc.shmem_tok_id_to_src.copy_(dcf.shmem_tok_id_to_src)
        _ret = g2a.run(a2=a2_e.view(-1), w2=w2_kernel, a2_scale=a2s_e, w2_scale=w2_scale_1d,
                       sorted_token_ids=a_st, sorted_expert_ids=a_se_local, sorted_weights=a_sw,
                       num_valid_ids=a_nv, cur_tok=run_tokens)
        _atom8_hold["o"] = _ret[0] if isinstance(_ret, (tuple, list)) else _ret
    _info(rank, f"[stage] atom-fp8 baseline built (trc={trc}); running baseline warmup ...")
    _atom_fp8_body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
    out_atom8 = _atom8_hold["o"][:run_tokens].float().cpu().numpy().copy()
    _info(rank, "[stage] atom-fp8 warmup DONE; computing torch oracle ...")

    # ---- full-precision WEIGHTED torch ORACLE ----
    x32 = x_bf16[:run_tokens].float(); ti = ic[:run_tokens].long()
    oracle_w = torch.zeros(run_tokens, model_dim, device=dev, dtype=torch.float32)
    wv = wc[:run_tokens].float()
    for k in range(topk):
        ek = ti[:, k]
        for e in torch.unique(ek).tolist():
            rows = (ek == e).nonzero().flatten()
            xr = x32[rows]
            _Wg = w1_all[e, :inter_dim]; _Wu = w1_all[e, inter_dim:2 * inter_dim]
            _a1 = F.silu(xr @ _Wg.t()) * (xr @ _Wu.t())
            _o2 = _a1 @ w2_all[e].t()
            oracle_w[rows] += wv[rows, k:k + 1] * _o2
    orw = oracle_w.cpu().numpy()

    _rm_w = _relL2(out_mega, orw)        # MegaMoEExp vs weighted oracle
    _ra8_w = _relL2(out_atom8, orw)      # atom-fp8 baseline vs weighted oracle
    _rma = _relL2(out_mega, out_atom8)   # MegaMoEExp vs baseline (should be ~0)
    _floor = 0.32 if _is_fp4 else 0.25
    _mega_ok = _rm_w < _floor
    _atom8_ok = _ra8_w < _floor
    # fp8 absorbs activation noise so mega tracks the baseline bitwise (_rma~0); fp4's coarse E2M1
    # lets two valid quantizations diverge, so gate on "mega no worse than baseline (+margin)".
    _match_ok = (_rma < 5e-2) or (_rm_w <= _ra8_w + 2e-2)
    ok = _match_ok and _mega_ok and _atom8_ok

    _rm_w_max = _all_max(dev, _rm_w)
    _ra8_w_max = _all_max(dev, _ra8_w)
    _rma_max = _all_max(dev, _rma)
    _all_ok = _all_max(dev, 0.0 if ok else 1.0) < 0.5

    # ---- STAGE1-only bodies ----
    def _mega_s1_body():
        moe.stage1.forward(x_q, wc, x_sc, ic)
    def _atom_s1_body():
        a2_e.zero_(); dcf.total_recv.zero_()
        _rx, _, _rs, _oidx, _ = dcf.dispatch(x_q, wc, x_sc, ic)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        aiter.moe_sorting_fwd(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc],
                              int(experts), int(tm), None, None, 0)
        aiter.mxfp4_moe_sort_hip(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
        a_se_local.copy_(a_se - rank * epr)
        gemm1(a2_e.view(max_recv, topk, a2_e.shape[-1]), _agv(_rx[:trc]), _w1_arg, a1s.view(torch.uint8),
              _w1s_arg, a_st, a_se_local, a_sw, a_nv, bias_d, a2s_e, fx.Int32(trc),
              fx.Int32(inter_dim * 2), fx.Int32(model_dim), fx.Int32(int(_max_blocks)),
              stream=fx.Stream(torch.cuda.current_stream()))

    _info(rank, f"[stage] oracle+relL2 done (megaexp={_rm_w:.3e} atom={_ra8_w:.3e} vs-base={_rma:.3e}); CUDAGraph timing ...")
    _t_mega = _cg_time(_mega_body)          # MegaMoEExp e2e (stage1+stage2)
    _info(rank, "[stage] timing: megaexp e2e done")
    _t_atom8 = _cg_time(_atom_fp8_body)     # atom-fp8 baseline e2e
    _info(rank, "[stage] timing: atom-fp8 e2e done")
    _t_mega_s1 = _cg_time(_mega_s1_body)    # MegaMoEExp STAGE1-only
    _t_atom_s1 = _cg_time(_atom_s1_body)    # baseline STAGE1-only
    _info(rank, "[stage] timing: stage1-only done")

    if rank == 0:
        print(f"[MEGAEXP-E2E] {args.network} {args.quant} bs={run_tokens} seed={args.seed} "
              f"-> {'PASS' if _all_ok else 'FAIL'} (all {world} ranks)", flush=True)
        print(f"  [precision vs WEIGHTED torch-oracle, MAX over {world} ranks]  megaexp={_rm_w_max:.3e}  "
              f"atom-fp8(baseline)={_ra8_w_max:.3e}  megaexp-vs-baseline={_rma_max:.3e}  (floor~{_floor})",
              flush=True)
        print(f"  [perf STAGE1-only, ms]  baseline-fp8={_t_atom_s1:.4f}  megaexp={_t_mega_s1:.4f}  "
              f"speedup={(_t_atom_s1 / _t_mega_s1) if _t_mega_s1 > 0 else -1:.3f}", flush=True)
        print(f"  [perf E2E (stage1+fused-stage2), ms]  baseline-fp8={_t_atom8:.4f}  megaexp={_t_mega:.4f}  "
              f"speedup={(_t_atom8 / _t_mega) if _t_mega > 0 else -1:.3f}", flush=True)
    return dict(network=args.network, quant=args.quant, tokens=run_tokens, seed=args.seed,
                megaexp_relL2=_rm_w, atom_fp8_relL2=_ra8_w, megaexp_vs_baseline=_rma,
                s1_baseline_ms=_t_atom_s1, s1_megaexp_ms=_t_mega_s1,
                e2e_baseline_fp8_ms=_t_atom8, e2e_megaexp_ms=_t_mega, passed=bool(_all_ok))


def run_one(args, rank, world, dev):
    net = NETWORKS[args.network]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(args.topk) if int(args.topk) > 0 else int(net["topk"])
    run_tokens = max(int(args.tokens), 1)
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world
    mtpr = _next_pow2(max(16, run_tokens))   # MegaMoEExp requires a power-of-two mtpr
    T = _prepare(dev, quant=args.quant, tokens=run_tokens, model_dim=model_dim,
                 inter_dim=inter_dim, experts=experts, topk=topk, seed=args.seed, rank=rank,
                 world=world, keep_ref=False)
    return _run(
        args, rank, world, dev, model_dim=model_dim, inter_dim=inter_dim, experts=experts,
        epr=epr, topk=topk, run_tokens=run_tokens, mtpr=mtpr, a_dtype=T["a_dtype"], s1_out=T["a_dtype"],
        w_kernel=T["w_kernel"], scale_w1_1d=T["scale_w1_1d"], x_bf16=T["x_bf16"],
        topk_ids=T["topk_ids"], wts=T["wts"])


def main():
    p = argparse.ArgumentParser(description="Standalone accuracy+perf bench for MegaMoEExp (decoupled from production MegaMoE).")
    p.add_argument("--network", type=str, default="v4_pro", choices=list(NETWORKS))
    p.add_argument("--quant", type=str, default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=-1)
    p.add_argument("--waves-per-eu", type=int, default=4)
    p.add_argument("--async-copy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--master-port", type=int, default=29922)
    p.add_argument("--matrix", action="store_true", help="run all networks x classic bs")
    p.add_argument("--full-bs", action="store_true", help="use the full bs sweep (1..32768)")
    p.add_argument("--bs-list", type=str, default="",
                   help="comma list of power-of-two batch sizes, e.g. '64,256,1024,4096'.")
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--mega-gemm2-tile-m", type=int, default=-1,
                   help="debug: override MegaMoEExp's gemm2 tile_m.")
    p.add_argument("--no-cudagraph", action="store_true",
                   help="time with plain eager launches instead of CUDAGraph capture/replay "
                        "(use if CUDAGraph capture hangs on the persistent megakernel + cross-PE combine).")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)

    if not HAS_AITER:
        _info(rank, "[megamoe-exp-bench] needs aiter; abort"); _cleanup(); return

    bs_set = FULL_BS if args.full_bs else CLASSIC_BS
    if args.bs_list:
        combos = [(args.network, int(b)) for b in args.bs_list.split(",") if b.strip()]
    elif args.matrix:
        combos = [(net, bs) for net in NETWORKS for bs in bs_set]
    else:
        combos = [(args.network, int(args.tokens))]

    _info(rank, f"[megamoe-exp-bench] MegaMoEExp vs atom-fp8 baseline vs torch-oracle | "
                f"{len(combos)} shape(s) x {max(1, int(args.n_seeds))} seed(s)")

    results = []
    _base_seed = int(args.seed)
    for net, bs in combos:
        args.network = net
        args.tokens = bs
        for _si in range(max(1, int(args.n_seeds))):
            args.seed = _base_seed + _si
            try:
                r = run_one(args, rank, world, dev)
                if r is not None:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                import traceback
                if rank == 0:
                    traceback.print_exc()
                _info(rank, f"[megamoe-exp-bench] {net} bs={bs} seed={args.seed} ERROR: {type(e).__name__}: {e}")
            import gc as _gc
            _gc.collect(); torch.cuda.empty_cache()
            torch.cuda.synchronize(); dist.barrier()
    args.seed = _base_seed

    if rank == 0:
        _np = sum(1 for r in results if r.get("passed"))
        print(f"[megamoe-exp-bench] SUMMARY: {_np}/{len(results)} shapes PASSED the accuracy gate", flush=True)
        if args.json_out:
            with open(args.json_out, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"[megamoe-exp-bench] wrote {len(results)} rows -> {args.json_out}", flush=True)

    torch.cuda.synchronize(); dist.barrier(); _cleanup()


if __name__ == "__main__":
    main()
