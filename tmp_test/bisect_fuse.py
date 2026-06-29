#!/usr/bin/env python3
"""Fast hang/no-hang bisection for the MegaMoE stage-1 scheduler megakernel deadlock.

An SDMA host-peek of the scheduler buffers is UNUSABLE for this deadlock: while the
persistent megakernel is wedged, even a D2H copy on an independent stream never drains
(verified: all buffers "did not drain" on all 8 ranks, on separate compute/copy streams).
So we cannot read the scheduler handshake buffers at all.

Instead this bisects the deadlock along the ONE axis we can still toggle: whether the
merged GEMM2 down-proj phase is fused into the stage-1 megakernel (fuse_gemm2).  The
dynamic claim scheduler itself is hardcoded on (tmp_mega_gemm_2stage.py: `sched = True`),
so it cannot be turned off.

  --fuse    (default) build FusedMoEMegaStage1(fuse_gemm2=True)  -> GEMM1 ⊕ merged GEMM2+combine
  --no-fuse           build FusedMoEMegaStage1(fuse_gemm2=False) -> dispatch + GEMM1 + scheduler ONLY

Detection: launch stage1.forward() on a dedicated NON-default compute_stream, then poll
compute_stream.query() for up to --probe-s seconds.  query()==True -> the megakernel
completed (NO deadlock).  Still False after the timeout -> DEADLOCK-HANG.

  --no-fuse COMPLETES but --fuse HANGS  => deadlock is in the merged GEMM2 handshake
                                           (l2_ready / g2_claim / P2P scatter), NOT GEMM1/dispatch.
  --no-fuse also HANGS                  => deadlock is in GEMM1 claim scheduler or dispatch,
                                           independent of GEMM2.

Run on the 8-GPU node:
    python tmp_test/bisect_fuse.py --no-fuse --network v4_pro --bs 64 --world 8
    python tmp_test/bisect_fuse.py --fuse    --network v4_pro --bs 64 --world 8
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

def _ensure_kernel_symlinks():
    """Make the experimental tmp_test/*.py importable as kernels.* (idempotent).

    megamoe_exp.py uses RELATIVE imports (from .tmp_mega_megakernel / .dispatch_combine_intranode_op),
    so it must resolve inside the kernels package -> the tmp_mega_* modules need kernels/ symlinks.
    (This used to live in the now-removed run_sched_verify.sh / tracecompile_sched.sh.)
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

import mori.shmem as ms  # noqa: E402
from tests.kernels.bench_moe_intranode_stage1_groupgemm import (  # noqa: E402
    NETWORKS, _prepare, _setup_dist, _chunked_fp4_quant,
)
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.utils import shuffle_weight  # noqa: E402

try:
    from aiter.ops.quant import per_1x32_mx_quant_hip as _PQ  # noqa: F401
    HAS_AITER = True
except Exception:  # noqa: BLE001
    HAS_AITER = False


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_w2_local(args, rank, world, dev, *, model_dim, inter_dim, experts, epr):
    """This rank's local W2 (mxfp4 shuffled + e8m0 scale), for the --fuse GEMM2 phase."""
    torch.manual_seed(args.seed + 4242)
    w2_f32 = (torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
              * (float(inter_dim) ** -0.25))
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2k = shuffle_weight(w2_fp4[_sl]).view(torch.uint8).contiguous().view(-1)
    w2s = fp4_utils.e8m0_shuffle(w2_sr[_sl]).view(torch.uint8).contiguous().view(-1)
    del w2_f32, w2_fp4, w2_sr
    torch.cuda.empty_cache()
    return w2k, w2s


def _quantize_a8w4(x_bf16):
    """MXFP8 activation quant, byte-identical to MegaMoEExp.quantize for quant='a8w4'."""
    from aiter import dtypes as _adt
    from aiter.ops.quant import per_1x32_mx_quant_hip
    mq, msq = per_1x32_mx_quant_hip(x_bf16.contiguous(), quant_dtype=_adt.fp8,
                                    scale_type=_adt.fp8_e8m0)
    return mq, msq.view(torch.uint8)


def _worker(rank, world, args):
    try:
        _setup_dist(rank, world, args.master_port)
        dev = torch.device("cuda", rank)
        if not HAS_AITER:
            if rank == 0:
                print("[bisect] needs aiter (per_1x32_mx_quant_hip); abort", flush=True)
            return
        from kernels.tmp_mega_megakernel import FusedMoEMegaStage1

        net = NETWORKS[args.network]
        model_dim, inter_dim = net["model_dim"], net["inter_dim"]
        experts, topk = net["experts"], net["topk"]
        epr = experts // world
        bs = int(args.bs)
        mtpr = _next_pow2(max(16, bs))

        os.environ["FLYDSL_TMP_FORCE_COMPACT"] = "1"  # match prod path

        T = _prepare(dev, quant="a8w4", tokens=bs, model_dim=model_dim, inter_dim=inter_dim,
                     experts=experts, topk=topk, seed=args.seed, rank=rank, world=world)
        w_kernel, scale_w1 = T["w_kernel"], T["scale_w1_1d"]
        _wpe = w_kernel.numel() // experts
        _spe = scale_w1.numel() // experts
        w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
        w1s = scale_w1.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()

        # For --fuse we still need w2/comb_op; the whole point of --no-fuse is to skip them, so
        # only the fused branch pulls in the full MegaMoEExp wiring.
        if args.fuse:
            from kernels.megamoe_exp import MegaMoEExp
            w2k, w2s = _build_w2_local(args, rank, world, dev, model_dim=model_dim,
                                       inter_dim=inter_dim, experts=experts, epr=epr)
            moe = MegaMoEExp(
                rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, quant="a8w4", w1=w1, w1_scale=w1s,
                w2=w2k, w2_scale=w2s, max_tok_per_rank=mtpr, network=args.network,
                stage2_mode="fused")
            stage1 = moe.stage1
            quant = moe.quantize
        else:
            stage1 = FusedMoEMegaStage1(
                rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, quant="a8w4", w1=w1, w1_scale=w1s,
                max_tok_per_rank=mtpr, network=args.network, fuse_gemm2=False)

            def quant(xb):
                return _quantize_a8w4(xb)

        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        xb = T["x_bf16"][:bs].contiguous()
        wc = T["wts"][:bs].contiguous()
        ic = T["topk_ids"][:bs].to(torch.int32).contiguous()
        x_q, scales = quant(xb)

        compute_stream = torch.cuda.Stream(device=dev)

        mode = "FUSE(gemm1+gemm2)" if args.fuse else "NO-FUSE(gemm1 only)"
        if rank == 0:
            print(f"[bisect] mode={mode} net={args.network} bs={bs} world={world} "
                  f"mtpr={mtpr} epr={epr} probe={args.probe_s}s", flush=True)

        # Drain default-stream setup (quantize etc.) before handing off; nothing is wedged yet.
        torch.cuda.synchronize()

        cg_reps = int(getattr(args, "cg_reps", 0))
        if cg_reps > 0:
            # CUDAGraph BACK-TO-BACK probe: the deadlock only shows up under replay (back-to-back
            # launches with NO inter-replay shmem_barrier_all -- cross-PE epoch barriers drift).  We
            # capture ONE graph, then fire cg_reps replays on compute_stream and poll query() (no full
            # sync -> a wedge is observable as query() staying False).  --no-fuse vs --fuse then
            # localizes the deadlock to dispatch/GEMM1 vs the merged GEMM2/combine-Stage1.
            if rank == 0:
                print(f"[bisect] CUDAGraph capture ({mode}); then {cg_reps} back-to-back replays ...",
                      flush=True)
            # single warmup launch (matches the earlier per-op warmup that PASSES on one launch)
            stage1.forward(x_q, wc, scales, ic)
            torch.cuda.synchronize(); ms.shmem_barrier_all()
            _cap = torch.cuda.Stream(device=dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=_cap):
                stage1.forward(x_q, wc, scales, ic)
            ms.shmem_barrier_all()
            if rank == 0:
                print(f"[bisect] captured; firing {cg_reps} back-to-back replays on compute_stream ...",
                      flush=True)
            with torch.cuda.stream(compute_stream):
                for _ in range(cg_reps):
                    g.replay()
        else:
            if rank == 0:
                print(f"[bisect] launching stage1.forward() ({mode}) on compute_stream ...", flush=True)
            with torch.cuda.stream(compute_stream):
                stage1.forward(x_q, wc, scales, ic)
        if rank == 0:
            print("[bisect] forward() returned (async); polling completion ...", flush=True)

        # Poll for completion WITHOUT a full-device sync (which would hang on a deadlock).
        t0 = time.time()
        done = False
        while time.time() - t0 < args.probe_s:
            if compute_stream.query():
                done = True
                break
            time.sleep(0.1)
        dt = time.time() - t0

        verdict = "COMPLETED (no deadlock)" if done else f"DEADLOCK-HANG (still running after {args.probe_s}s)"
        # serialize per-rank prints
        for r in range(world):
            if r == rank:
                print(f"[bisect][rank{rank}] {mode}: {verdict}  (elapsed {dt:.2f}s)", flush=True)
            time.sleep(0.1)
        if rank == 0:
            print(f"\n================= BISECT RESULT ({mode}) =================", flush=True)
            print(f"  COMPLETED => this configuration does NOT deadlock", flush=True)
            print(f"  DEADLOCK-HANG => this configuration deadlocks", flush=True)
            print(f"  compare --no-fuse vs --fuse to localize the deadlock to the GEMM2 merge.", flush=True)
    except Exception as ex:  # noqa: BLE001
        import traceback
        print(f"[bisect][rank{rank}] ERROR: {type(ex).__name__}: {ex}", flush=True)
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # If it deadlocked, the persistent kernel is unrecoverable -> hard-exit (a clean
        # cleanup/ctx destroy would hang).  If it completed, this is still the safe exit.
        os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=str, default="v4_pro", choices=list(NETWORKS))
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world", type=int, default=8)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--fuse", dest="fuse", action="store_true", help="fuse_gemm2=True (default)")
    grp.add_argument("--no-fuse", dest="fuse", action="store_false", help="fuse_gemm2=False (GEMM1 only)")
    p.set_defaults(fuse=True)
    p.add_argument("--cg-reps", type=int, default=0,
                   help="if >0, probe with a CUDAGraph captured once then replayed back-to-back "
                        "cg-reps times (NO inter-replay barrier) -- reproduces the replay deadlock; "
                        "0 (default) keeps the single-launch stream probe.")
    p.add_argument("--probe-s", type=float, default=20.0, help="seconds to wait for completion before declaring a hang")
    p.add_argument("--master-port", type=int, default=29594)
    args = p.parse_args()
    world = int(args.world)
    if world > torch.cuda.device_count():
        raise SystemExit(f"need {world} GPUs, have {torch.cuda.device_count()}")
    mp.spawn(_worker, args=(world, args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
