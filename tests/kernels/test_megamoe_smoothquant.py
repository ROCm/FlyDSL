"""MegaMoE a8w4smooth (W4A8 LQQ + smoothquant) DECODE perf + accuracy, EP8.

For each decode batch (per-rank tokens 8..1024) it builds MegaMoE at that max_tok_per_rank (so the
tune-config lookup picks the per-bucket tile), then reports BOTH:
  * accuracy : fused output vs a torch oracle over the dequantized int8 weights, max relL2 over the
               8 ranks, under CUDA-graph replay (the production decode path) -- so each perf number
               is only reported when the kernel is numerically correct.
  * perf     : per-kernel DEVICE time (test_mega_moe yardstick) split into the MoE phases
               (front-quant / dispatch+gemm1 / requant / gemm2+scatter / combine) + CUDA-graph
               replay wall-clock (max across ranks).

Shape defaults to the target decode config; override via env (MEGAMOE_MD / _ID / _E / _TOPK).

  cd <FlyDSL repo root> && MORI_SHMEM_HEAP_SIZE=16G \
      torchrun --standalone --nproc_per_node=8 tests/kernels/test_megamoe_smoothquant.py
  MEGAMOE_BS=8,32,128 torchrun ... tests/kernels/test_megamoe_smoothquant.py    # custom bs sweep
"""
import os
import sys

os.environ.setdefault("FLIR_A8W4SMOOTH_QPARAM_FORMAT", "packed4")
os.environ.setdefault("FLIR_A8W4SMOOTH_INTERLEAVE_K64", "1")
# Repo-relative paths (this file lives in <repo>/tests/kernels/): built flydsl + repo root, so the
# test runs from a checkout without a hand-set PYTHONPATH. aiter (device quant kernels + mori) comes
# from AITER_ROOT (default /home/ghu/aiter_universe/aiter) or an existing PYTHONPATH/install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "build-fly", "python_packages"), _REPO_ROOT,
           os.environ.get("AITER_ROOT", "/home/ghu/aiter_universe/aiter")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.distributed as dist
import mori.shmem as ms
from torch.profiler import ProfilerActivity, profile
from kernels.moe.mega_moe import MegaMoE
from tests.utils import shuffle_weight   # original repo util: base(16,16) + K64-interleave preshuffle

_ENVI = lambda k, d: int(os.environ.get(k, str(d)))
MD = _ENVI("MEGAMOE_MD", 3584)          # model_dim (hidden)
ID = _ENVI("MEGAMOE_ID", 1280)          # moe intermediate
E = _ENVI("MEGAMOE_E", 384)             # global experts
TOPK = _ENVI("MEGAMOE_TOPK", 8)
SW = 1e-3                                # per-output-row weight scale magnitude
ITERS = _ENVI("MEGAMOE_ITERS", 50)
BS_LIST = [int(b) for b in os.environ.get("MEGAMOE_BS", "8,16,32,64,128,256,512,1024").split(",") if b.strip()]


# ---- a8w4smooth weight generation (inlined so this script has no dependency on the removed
# standalone a8w4 kernel/test; only original repo files -- tests.utils.shuffle_weight + MegaMoE --
# plus aiter are imported). The packed-int4 nibble pairing here is K64-interleave-specific
# (b0=(v1<<4)|v0), which differs from tests.kernels.test_moe_gemm's _pack, so it is NOT reused. ----
def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8):
    """Pack a K64-interleaved int8 tensor into packed int4 bytes: each contiguous 8-value block
    [v0..v7] -> 4 bytes with adjacent pairing b0=(v1<<4)|v0, b1=(v3<<4)|v2, b2=(v5<<4)|v4,
    b3=(v7<<4)|v6 (so each byte holds a (lo,hi) nibble pair for the kernel's even/odd split)."""
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 1] << 4)
    out[:, 1] = u[:, 2] | (u[:, 3] << 4)
    out[:, 2] = u[:, 4] | (u[:, 5] << 4)
    out[:, 3] = u[:, 6] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)


def build_a8w4smooth_moe_weight(*, experts, rows_per_expert, K, device, seed=0, interleave_k64=True):
    """Generate a8w4smooth weights + qparams in the agreed physical layouts. Returns
    (w_packed_i8, qscale_u8, qzero_u8, qscale_i32, qzero_i32, w_int8_unshuffled_flat); this script
    uses w_packed_i8 / qscale_i32 / qzero_i32 (fed to MegaMoE) + w_int8_unshuffled_flat (oracle)."""
    if int(experts) <= 0 or int(rows_per_expert) <= 0:
        raise ValueError(f"invalid experts/rows_per_expert: {experts=}, {rows_per_expert=}")
    if K % 256 != 0:
        raise ValueError(f"requires K%256==0, got K={K}")
    if interleave_k64 and (K % 128 != 0):
        raise ValueError(f"interleave_k64 requires K%128==0, got K={K}")
    if rows_per_expert % 16 != 0:
        raise ValueError(f"requires rows_per_expert%16==0, got rows_per_expert={rows_per_expert}")

    torch.manual_seed(int(seed))
    nb, g256 = rows_per_expert // 16, K // 256

    u4_unshuf = torch.randint(0, 16, (experts, rows_per_expert, K), device=device, dtype=torch.uint8)
    u4_shuf_i8 = shuffle_weight(u4_unshuf.view(torch.int8), use_int4=True, interleave_k64=bool(interleave_k64))
    u4_shuf = (u4_shuf_i8.view(torch.uint8) & 0xF).contiguous()

    qparam_shape = (experts, nb, g256, 16, 4)
    qs_i32 = torch.randint(1, 3, qparam_shape, device=device, dtype=torch.int32)
    qz_i32 = torch.randint(0, 16, qparam_shape, device=device, dtype=torch.int32)
    qscale_u8, qzero_u8 = qs_i32.to(torch.uint8), qz_i32.to(torch.uint8)

    # logical dequant (int8 = clamp(u4*scale + zero, 0, 255) ^ 0x80) per 64-K group, unshuffled.
    u4_logical = u4_unshuf.view(experts, nb, 16, g256, 4, 64).to(torch.int32).permute(0, 1, 3, 2, 4, 5)
    u8_logical = torch.clamp((u4_logical * qs_i32.unsqueeze(-1)) + qz_i32.unsqueeze(-1), 0, 255).to(torch.uint8)
    u8_unshuf = u8_logical.permute(0, 1, 3, 2, 4, 5).reshape(experts, rows_per_expert, K)

    w_packed = _pack_shuffled_int8_to_packed_int4_no_perm(u4_shuf.reshape(-1, K).to(torch.int8))

    def _pack_qparam_i32(q_u8):  # [E, nb, g256, 16, 4] u8 -> [E, nb, g256, 16] i32 (4 K64 bytes LE)
        return (q_u8[..., 0].to(torch.int32) | (q_u8[..., 1].to(torch.int32) << 8)
                | (q_u8[..., 2].to(torch.int32) << 16) | (q_u8[..., 3].to(torch.int32) << 24))

    w_i8_unshuffled_flat = (u8_unshuf.to(torch.int32) ^ 0x80).to(torch.int8).reshape(-1, K).contiguous()
    return (w_packed, qscale_u8, qzero_u8,
            _pack_qparam_i32(qscale_u8), _pack_qparam_i32(qzero_u8), w_i8_unshuffled_flat)


def _setup():
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=dev)
    import torch._C._distributed_c10d as c10d
    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return dev, rank, world


def _silu(x):
    return x * torch.sigmoid(x)


def _phase(name):
    n = name.lower()
    if "smooth_per_token_scaled_quant_kernel_v2" in n or "moe_smooth_per_token" in n:
        return "requant"
    if "smooth_per_token_scaled_quant" in n or "per_token_scaled_quant" in n:
        return "front-quant"
    if "moe_gemm1" in n:
        return "dispatch+gemm1"
    if "mfma_moe2" in n:
        return "gemm2+scatter"
    if "ep_combine" in n:
        return "combine"
    return "misc"


def _oracle(x, ids, wts, w1r, w2r, fc1, fc2, tok, dev):
    """Torch reference over dequantized int8 weights, batched per expert (fast for large bs).
    Uses aiter's device smoothquant kernel so the rounding matches the fused path."""
    from aiter.ops.quant import smooth_per_token_scaled_quant as _sptsq
    _A1 = torch.zeros(tok, TOPK, MD, dtype=torch.int8, device=dev)
    _A1s = torch.zeros(tok, TOPK, 1, dtype=torch.float32, device=dev)
    _sptsq(_A1, x.view(tok, 1, MD).expand(tok, TOPK, MD), _A1s, fc1, ids, smooth_scale_map_hash=None, enable_ps=True)
    _a2 = torch.zeros(tok, TOPK, ID, device=dev, dtype=torch.float16)
    _ts = ids.view(-1)                                   # [tok*topk] expert per (t,s)
    _a1f = (_A1.view(-1, MD).float() * _A1s.view(-1, 1))  # [tok*topk, MD] (pre-weight-scale)
    _g = torch.empty(tok * TOPK, 2 * ID, device=dev, dtype=torch.float32)
    for e in range(E):
        m = (_ts == e)
        if m.any():
            _g[m] = (_a1f[m] @ w1r[e].T) * SW
    _g = _g.view(tok, TOPK, 2 * ID)
    _a2 = (_silu(_g[..., :ID]) * _g[..., ID:]).to(torch.float16)
    _A2 = torch.zeros(tok, TOPK, ID, dtype=torch.int8, device=dev)
    _A2s = torch.zeros(tok, TOPK, 1, dtype=torch.float32, device=dev)
    _sptsq(_A2, _a2, _A2s, fc2, ids, smooth_scale_map_hash=None, enable_ps=True)
    _a2f = (_A2.view(-1, ID).float() * _A2s.view(-1, 1))
    _o = torch.empty(tok * TOPK, MD, device=dev, dtype=torch.float32)
    for e in range(E):
        m = (_ts == e)
        if m.any():
            _o[m] = (_a2f[m] @ w2r[e].T) * SW
    return (_o.view(tok, TOPK, MD) * wts.unsqueeze(-1)).sum(1)


def main():
    dev, rank, world = _setup()
    epr = E // world
    lo, hi = rank * epr, (rank + 1) * epr

    (w1p, _1, _2, w1qs, w1qz, w1u) = build_a8w4smooth_moe_weight(
        experts=E, rows_per_expert=2 * ID, K=MD, device=dev, seed=3, interleave_k64=True)
    (w2p, _3, _4, w2qs, w2qz, w2u) = build_a8w4smooth_moe_weight(
        experts=E, rows_per_expert=MD, K=ID, device=dev, seed=25, interleave_k64=True)
    torch.manual_seed(1234)
    fc1 = (0.75 + 0.5 * torch.rand((E, MD), device=dev, dtype=torch.float32))
    fc2 = (0.75 + 0.5 * torch.rand((E, ID), device=dev, dtype=torch.float32))
    w1_loc = w1p.view(E, -1)[lo:hi].reshape(-1).contiguous()
    w2_loc = w2p.view(E, -1)[lo:hi].reshape(-1).contiguous()
    _w1qs, _w1qz = w1qs[lo:hi].contiguous(), w1qz[lo:hi].contiguous()
    _w2qs, _w2qz = w2qs[lo:hi].contiguous(), w2qz[lo:hi].contiguous()
    w1_scale = torch.full((epr * 2 * ID,), SW, device=dev, dtype=torch.float32)
    w2_scale = torch.full((epr * MD,), SW, device=dev, dtype=torch.float32)
    w1r = w1u.view(E, 2 * ID, MD).float()
    w2r = w2u.view(E, MD, ID).float()

    def _run_bucket(bs, *, measure):
        """One decode bucket: build MegaMoE(mtpr=bs) -> CUDA-graph capture/replay -> accuracy vs
        oracle + per-kernel/wall timing (printed on rank 0). ``measure=False`` is a throwaway warmup
        (no print) used once before the sweep to pay process-global first-time costs."""
        mtpr = bs
        torch.manual_seed(100 + rank)
        x = torch.randn(bs, MD, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(bs)]).to(torch.int32)
        wts = torch.rand(bs, TOPK, device=dev, dtype=torch.float32)

        moe = MegaMoE(rank=rank, world_size=world, model_dim=MD, inter_dim=ID, experts=E, topk=TOPK,
                      quant="a8w4smooth", w1=w1_loc, w1_scale=w1_scale, w2=w2_loc, w2_scale=w2_scale,
                      w1_lqq_scale=_w1qs, w1_lqq_zero=_w1qz, w2_lqq_scale=_w2qs, w2_lqq_zero=_w2qz,
                      fc1_smooth_scale=fc1, fc2_smooth_scale=fc2,
                      max_tok_per_rank=mtpr, enable_fused_stage1=True, enable_fused_stage2=True)
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        ref = _oracle(x, ids, wts, w1r, w2r, fc1, fc2, bs, dev)

        # CUDA-graph capture/replay = production decode path; accuracy checked under replay.
        for _ in range(3):
            moe.forward(x, wts, ids)
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_g = moe.forward(x, wts, ids)
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize(); dist.barrier()
        og = out_g.view(bs, MD).float()
        rel = (((og - ref) ** 2).sum() / (ref ** 2).sum().clamp_min(1e-12)).sqrt()
        fin = torch.isfinite(og).all()

        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(ITERS):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        wall = e0.elapsed_time(e1) / ITERS * 1e3

        reps = max(20, ITERS // 2)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                g.replay()
            torch.cuda.synchronize()
        ph = {}
        for ev in prof.key_averages():
            if ev.self_device_time_total > 0:
                ph[_phase(ev.key)] = ph.get(_phase(ev.key), 0.0) + ev.self_device_time_total / reps

        # 8-card accuracy: EACH rank validated its OWN tokens through the full EP path (cross-rank
        # dispatch to all 384 experts + combine) vs its OWN torch oracle. Gather ALL ranks' relL2
        # (not just an aggregate) so every card is individually visible + checked; ok requires ALL
        # 8 ranks < 0.2 AND finite (relL2 column is the worst-of-8).
        rr = torch.zeros(world, device=dev); rr[rank] = rel
        ff = torch.zeros(world, device=dev); ff[rank] = 1.0 if fin else 0.0
        dist.all_reduce(rr, op=dist.ReduceOp.SUM); dist.all_reduce(ff, op=dist.ReduceOp.SUM)
        t_wall = torch.tensor([wall], device=dev); dist.all_reduce(t_wall, op=dist.ReduceOp.MAX)
        relmax = rr.max().item()
        ok = relmax < 0.2 and bool((ff > 0.5).all().item())
        if measure and rank == 0:
            print(f"  {bs:>6} {relmax:>10.2e} {('Y' if ok else 'N'):>4} {t_wall.item():>10.1f} "
                  f"{ph.get('front-quant',0):>8.1f} {ph.get('dispatch+gemm1',0):>8.1f} "
                  f"{ph.get('requant',0):>8.1f} {ph.get('gemm2+scatter',0):>8.1f} {ph.get('combine',0):>8.1f}")
            print("         8-rank relL2: " + " ".join(f"r{r}={rr[r].item():.2e}" for r in range(world)))
        del moe, g
        torch.cuda.empty_cache(); dist.barrier()

    # GPU clock warmup, then a throwaway full-pipeline warmup on the smallest bucket (build +
    # CUDA-graph capture/replay + timed loop, DISCARDED). This pays the process-global first-time
    # costs -- CUDA-graph subsystem init, caching allocator pools, shmem P2P setup, first profiler
    # cycle -- that otherwise inflate the FIRST measured row's wall-clock (device time is unaffected).
    _wu = torch.randn(4096, MD, device=dev, dtype=torch.bfloat16)
    for _ in range(20):
        _ = _wu @ _wu.t()[:MD, :MD]
    torch.cuda.synchronize(); dist.barrier()
    _run_bucket(BS_LIST[0], measure=False)

    if rank == 0:
        print(f"\n=== MegaMoE a8w4smooth DECODE (md={MD} inter={ID} experts={E} topk={TOPK}, EP{world}) ===")
        print(f"  {'bs':>6} {'relL2max':>10} {'ok':>4} {'graph us':>10} "
              f"{'front-q':>8} {'disp+g1':>8} {'requant':>8} {'gemm2':>8} {'combine':>8}"
              f"   (relL2max/ok are over ALL {world} ranks; per-rank relL2 printed under each row)")

    for bs in BS_LIST:
        _run_bucket(bs, measure=True)

    dist.barrier()
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
