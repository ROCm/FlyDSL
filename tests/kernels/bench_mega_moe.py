# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Self-contained **accuracy + performance** bench for the fused ``mega_moe``
operator (``FlyDSLMegaMoEIntraNodeOp.fp8_fp4_mega_moe``), merging the former
``test_mega_moe_accuracy_multigpu.py`` (correctness) and
``test_mega_moe_op_perf.py`` (timing) into ONE file with NO same-directory
imports -- only ``torch`` + the op + ``mori`` (and ``aiter`` for the serial
baseline) need be importable.  Ship/run this single file standalone.

Each run reports, per (mode x cudagraph) cell:
  * ACCURACY (fused): relL2 of ``y`` vs a full-precision f32 torch oracle
    (bench_moe_intranode_stage1_groupgemm.py methodology -- TRUE pre-quant bf16
    activations + TRUE f32 weights, NO quant noise reproduced), reduced with
    MAX across ranks; PASS = relL2 <= ``--acc-floor`` (default 0.30, the a8w4
    mxfp4-weight quant floor).  Computed only for ``--phase full``.
  * PERF: E2E (and kernel, in profile mode) device time for the fused op and
    the serial baseline, plus the serial/fused speedup.

Timing methodology (ported from the production benchmark
``bench_moe_intranode_stage1_groupgemm.py`` ``_cg_time``): each measured cell
times ``iters`` **back-to-back** calls under a SINGLE CUDA-event pair and
divides by ``iters`` (steady-state **mean** us/iter, no per-iter host/sync
bubble), then reduces across ranks with the **average** (``_all_mean``, the
official EP reporting口径; min/max also reported for the tail).  The
production-matching cell is ``--mode bench --cudagraph`` (CUDAGraph device
time); use it for the headline prefill/decode numbers.

Two orthogonal axes can be freely combined (4 cases), exactly as in the skill's
reference ``test_profiler_dispatch_combine.py``:

  --mode       measurement: ``profile`` (torch.profiler -> host trace + pure
               kernel GPU time) | ``bench`` (CUDA-event E2E timing, no profiler
               overhead, rocprofv3-friendly)
  --cudagraph  execution:   absent = eager | present = CUDAGraph capture+replay
               (zero Python launch overhead; production methodology)

The operator under test (``--op``) is a *whole* MoE op, treated like the
dispatch/combine op in the reference:

  serial : ``mori.dispatch -> aiter.fused_moe -> mori.combine``
  fused  : ``FlyDSLMegaMoEIntraNodeOp.fp8_fp4_mega_moe``

Serial (aiter) and fused (FlyDSL) cannot co-init mori shmem in one process, so
each runs as a **separate** ``torch.multiprocessing.spawn`` batch; ``--op both``
runs them sequentially.

CUDAGraph notes (per skill "no-reset" rule):
  * The fused op calls a host-side ``ms.shmem_barrier_all()`` *inside* the timed
    call.  Host collectives are not capturable, so during capture+replay we
    stub ``ms.shmem_barrier_all`` to a no-op (one real barrier is issued just
    before capture; replays add no barrier).
  * The serial mori dispatch/combine can fail to capture on data-dependent
    shapes; capture is wrapped in try/except and reported, never fatal.

Launch (inside the ROCm container, >= 2x gfx950)::

    export PYTHONPATH=/home/ywx/mori/python:.:tests:\\
/home/ywx/megamoe/FlyDSL/build-fly/python_packages:/home/ywx/aiter
    export AITER_USE_SYSTEM_TRITON=1
    # fused-kernel tuning knobs go in the env (see kernel/op):
    export MEGA_MOE_L1_WS=1 MEGA_MOE_L2_WS=1 MEGA_MOE_FUSE_SWIGLU=1 \\
           MEGA_MOE_FUSE_COMBINE=1 MEGA_MOE_DISPATCH_OVERLAP=1 \\
           MEGA_MOE_NUM_SMS=512

    # bench + eager (accuracy + perf, closest to production E2E)
    HIP_VISIBLE_DEVICES=0,1 python3 tests/kernels/bench_mega_moe.py \\
        --op both --mode bench

    # bench + cudagraph
    ... --op both --mode bench --cudagraph

    # profile + eager  (host trace + pure kernel GPU time)
    ... --op fused --mode profile --trace-dir ./traces

    # profile + cudagraph
    ... --op fused --mode profile --cudagraph --trace-dir ./traces

Geometry knobs are CLI flags (defaults mirror
``bench_mega_moe_serial_aiter_mori.py``): ``--world-size`` (2), ``--epr`` (8 ->
E=epr*ws), ``--topk`` (8), ``--hidden`` (2048, %256), ``--ih`` (1024),
``--tokens`` per rank (2048), ``--iters`` (100), ``--warmup`` (30), ``--quant``
(``per_1x32`` | ``a8w4`` | ``per_Token`` | ``per_128x128``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)


# ===========================================================================
# Inlined helpers -- this file is SELF-CONTAINED so it can be shipped/run on
# its own (only ``torch`` + the op-under-test ``kernels.mega_moe_intranode_op``
# + mori/aiter need be importable).  These are copied verbatim from the sibling
# test modules; keep in sync with the originals if those ever change:
#   * fp4 / e8m0 MX quant utils  <- tests/kernels/utils/fp4_utils.py
#   * per_1x32_mxfp4_quant       <- tests/kernels/test_mega_moe_l1_gemm.py
#   * per_token_cast_to_fp8      <- tests/kernels/test_mega_moe_dispatch.py
#   * shuffle_weight             <- tests/utils.py
# (The serial worker deliberately keeps using aiter's OWN fp4_utils /
# shuffle_weight; only the fused worker consumes these inlined copies.)
# ===========================================================================
_FP8_E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
_FP4X2 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
_EBITS_F32, _MBITS_F32 = 8, 23
_F32_EXP_BIAS = (1 << (_EBITS_F32 - 1)) - 1


def _n_ones(n: int) -> int:
    return (1 << n) - 1


def _down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def _pack_uint4(uint8_data) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(_down_size(shape))


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """FP32 -> sub-byte float (encoding in the LSBs), no NaN/inf support.
    Adapted from torchao custom_fp_utils; used here for FP4 (ebits=2, mbits=1)."""
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(_MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (_MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp
    return x.to(torch.uint8)


def f32_to_mxfp4(x):
    x = _f32_to_floatx_unpacked(x.float(), 2, 1)
    x = _pack_uint4(x)
    return x.view(_FP4X2)


def f32_to_e8m0(x):
    u32 = x.view(torch.int32)
    exponent = ((u32 >> 23) & 0xFF).view(torch.uint32).to(torch.uint8)
    nan_case = exponent == 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0))
    exponent[round_case] += 1
    exponent[nan_case] = 0xFF
    return exponent.view(_FP8_E8M0)


def e8m0_to_f32(scale_e8m0_biased):
    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    return scale_f32.view(torch.float32)


def e8m0_shuffle(scale):
    if scale is None:
        return scale
    if scale.dtype == torch.float32:
        return scale
    assert scale.ndim == 2, "scale must be a 2D tensor"
    m, n = scale.shape
    scale_padded = torch.empty(
        (m + 255) // 256 * 256, (n + 7) // 8 * 8,
        dtype=scale.dtype, device=scale.device)
    scale_padded[:m, :n] = scale
    scale = scale_padded
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    return scale.view(sm, sn)


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _pack_ue8m0_to_int(sf: torch.Tensor) -> torch.Tensor:
    assert sf.size(-1) % 4 == 0
    u8 = (sf.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    return u8.view(torch.int32)


def per_token_cast_to_fp8(x: torch.Tensor, gran_k: int = 32):
    """[m, n] -> (fp8 [m, n], packed-ue8m0 int32 SF [m, n // 128]), K-major."""
    m, n = x.shape
    assert n % gran_k == 0
    xv = x.view(m, n // gran_k, gran_k)
    amax = xv.abs().float().amax(dim=2).clamp(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)
    x_fp8 = (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n).contiguous()
    return x_fp8, _pack_ue8m0_to_int(sf)


def per_1x32_mxfp4_quant(x: torch.Tensor):
    """[..., K] f32 -> (x_fp4 packed [..., K//2], scale_e8m0 uint8 [..., K//32])."""
    F4E2M1_MAX = 6.0
    dtype_max = 2.0 ** int(torch.log2(torch.tensor(F4E2M1_MAX)).item())
    shape = x.shape
    xb = x.contiguous().view(-1, 32).float()
    amax = torch.amax(torch.abs(xb), dim=-1)
    scale_e8m0 = f32_to_e8m0(amax / dtype_max)
    scale_f32 = e8m0_to_f32(scale_e8m0)
    y_fp4 = f32_to_mxfp4(xb / scale_f32.view(-1, 1))
    y_fp4 = y_fp4.view(*shape[:-1], -1)
    scale = scale_e8m0.view(*shape[:-1], shape[-1] // 32).view(torch.uint8)
    return y_fp4, scale


def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN}"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK}"
    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_


# ===========================================================================
# Accuracy helpers: deterministic per-GLOBAL-expert weights (so every rank can
# rebuild any expert's f32 weights for its oracle), a full-precision f32 torch
# oracle, and the host-side float64 relL2 -- all mirroring
# bench_moe_intranode_stage1_groupgemm.py + the accuracy test.
# ===========================================================================
def _gen_w1(global_e, N, hidden, device):
    g = torch.Generator(device=device).manual_seed(10000 + global_e)
    return torch.randn((N, hidden), generator=g, device=device, dtype=torch.float32) * 0.3


def _gen_w2(global_e, hidden, ih, device):
    g = torch.Generator(device=device).manual_seed(20000 + global_e)
    return torch.randn((hidden, ih), generator=g, device=device, dtype=torch.float32) * 0.3


def _relL2(a, b):
    """relative L2 on host in float64: ||a-b||_2 / ||b||_2 (bench _relL2)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = float(((a - b) ** 2).sum())
    d = float((b ** 2).sum())
    return (n / d) ** 0.5 if d > 0 else -1.0


def _fused_relL2(y, x_bf16, ti, tw, *, N, hidden, ih, topk, device):
    """relL2 of the kernel output ``y`` vs a full-precision f32 torch oracle
    built from TRUE pre-quant bf16 activations + TRUE f32 weights (per-global-
    expert), routing-WEIGHTED reduce -- the kernel's total quant floor."""
    import torch.nn.functional as F
    x32 = x_bf16.float()
    n_tok = x32.shape[0]
    oracle = torch.zeros(n_tok, hidden, device=device, dtype=torch.float32)
    til = ti.long()
    twf = tw.float()
    w1c, w2c = {}, {}
    for k in range(topk):
        ek = til[:, k]
        for e in torch.unique(ek).tolist():
            if e < 0:
                continue
            rows = (ek == e).nonzero().flatten()
            xr = x32[rows]
            if e not in w1c:
                w1c[e] = _gen_w1(e, N, hidden, device)
                w2c[e] = _gen_w2(e, hidden, ih, device)
            w1e = w1c[e]
            a1 = F.silu(xr @ w1e[:ih].t()) * (xr @ w1e[ih:].t())
            oracle[rows] += twf[rows, k:k + 1] * (a1 @ w2c[e].t())
    y_np = y.float().cpu().numpy()
    orc_np = oracle.cpu().numpy()
    return _relL2(y_np, orc_np)


# ===========================================================================
# Small shared helpers (timing + cross-rank aggregation), per skill section 5.
# ===========================================================================
def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _skewed_scores(tokens: int, E: int, skew: float, device):
    """Gating scores with a controllable expert-load imbalance.

    ``skew == 0`` -> plain ``randn`` (balanced random routing, the original
    behaviour).  ``skew > 0`` -> add a Zipf-like per-expert popularity bias
    ``skew * log(1 / rank)`` (a random permutation of experts decides which are
    "hot"), so topk concentrates on a subset and a few experts capture most
    tokens.  Larger ``skew`` => heavier tail.
    """
    scores = torch.randn(tokens, E, device=device)
    if skew and skew > 0.0:
        perm = torch.randperm(E, device=device)
        rank = torch.empty(E, device=device)
        rank[perm] = torch.arange(1, E + 1, device=device, dtype=rank.dtype)
        bias = skew * torch.log(1.0 / rank)  # Zipf-ish, hottest expert ~ +0
        scores = scores + bias[None, :]
    return scores


def _expert_load_stats(topk_ids, E: int):
    """Return (max/mean ratio, per-expert token counts) for a topk_ids tensor."""
    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=E).float()
    mean = counts.mean().clamp_min(1e-9)
    return float((counts.max() / mean).item()), counts


def _allreduce_avg_min_max(local_val: float, world_size: int):
    """Each rank contributes one local scalar; return (avg, mn, mx) across ranks
    (skill 5: SUM/MAX/MIN all_reduce, avg = sum / world)."""
    t = torch.tensor([local_val], dtype=torch.float64, device="cuda")
    s = t.clone(); dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = t.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = t.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    return float(s.item()) / world_size, float(mn.item()), float(mx.item())


def _bench_backtoback(run_fn, iters: int) -> float:
    """Back-to-back device-time timing of ``run_fn`` (one CUDA-event pair around
    ``iters`` *uninterrupted* calls, divided by ``iters`` -> **mean** us/iter).

    This is the production benchmark methodology from
    ``bench_moe_intranode_stage1_groupgemm.py`` (``_cg_time``): the iterations
    run back-to-back on the stream with a SINGLE ``synchronize`` at the end, so
    the measured span is pure GPU steady-state throughput with no per-iter
    host/sync bubble.  (The previous per-iter event-pair + ``end.synchronize()``
    + median injected a host round-trip between every iter, inflating and adding
    variance to short ops.)  ``run_fn`` is ``graph.replay`` for the CUDAGraph
    path or the eager step for the eager path; caller does warmup first."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / max(1, iters)


def _bench_eager(step_fn, iters: int):
    """Back-to-back mean device time of eager ``step_fn`` (us/iter)."""
    return _bench_backtoback(step_fn, iters)


def _bench_graph(graph, iters: int):
    """Back-to-back mean device time of CUDAGraph replay (us/iter). No barrier
    between replays (skill no-reset rule)."""
    return _bench_backtoback(graph.replay, iters)


def _profiler_kernel_us_per_iter(prof, active_iters: int) -> float:
    """Pure-kernel GPU time per iter (us) from a torch.profiler run: sum of
    every real CUDA kernel's ``self_device_time_total`` (a TOTAL, not a mean --
    see skill "key_averages() device_time is mean" pitfall) over the active
    window, divided by ``active_iters``.

    Two skill pitfalls cause over-counting if summed naively; both are device
    time that is NOT a distinct GPU kernel and must be excluded:

    1. ``record_function`` tags project onto the GPU timeline as
       ``gpu_user_annotation`` events that ALSO carry a CUDA ``device_type`` and
       a ``self_device_time_total`` equal to the span of the kernels they wrap
       (torch does not subtract it from the children) -> filter
       ``is_user_annotation``.
    2. A CPU-side launcher op (e.g. ``aten::mm``) carries a
       ``self_device_time_total`` equal to the GPU time of the kernel IT
       launched, while that kernel ALSO appears as its own CUDA event -> a
       ``self_device_time_total > 0`` test would count the same GPU time twice
       (once on the CPU launcher, once on the device kernel).  So require the
       event to be a real DEVICE event (``device_type`` CUDA/PrivateUse1); do
       NOT fall back to ``self_device_time_total > 0``.

    The result matches the trace ``cat=="kernel"`` dur sum the skill recommends
    (verified: serial prefill-4096 -> 0.85 ms, == trace busy time, < E2E)."""
    total_self = 0.0
    for evt in prof.key_averages():
        if getattr(evt, "is_user_annotation", False):
            continue  # gpu_user_annotation (record_function) is not a kernel
        dt = str(getattr(evt, "device_type", None))
        if dt.endswith("CUDA") or dt.endswith("PrivateUse1"):
            total_self += float(getattr(evt, "self_device_time_total", 0.0))
    return total_self / max(active_iters, 1)


def _make_profiler(active_iters: int, prof_warmup: int = 5):
    from torch.profiler import ProfilerActivity, profile, schedule
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=prof_warmup, active=active_iters, repeat=1),
        record_shapes=False, with_stack=False,
    )


def _profile_eager(step_fn, active_iters: int, prof_warmup: int = 5):
    """torch.profiler over eager ``step_fn`` (skill section 2). step_fn should
    contain its own ``record_function`` tags. Returns (kernel_us_per_iter,
    e2e_us_per_iter, prof)."""
    n_steps = 1 + prof_warmup + active_iters
    e2e = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with _make_profiler(active_iters, prof_warmup) as prof:
        for s in range(n_steps):
            start.record()
            step_fn()
            end.record()
            end.synchronize()
            if s >= 1 + prof_warmup:
                e2e.append(start.elapsed_time(end) * 1000.0)
            prof.step()
    kern = _profiler_kernel_us_per_iter(prof, active_iters)
    return kern, (_median(e2e) if e2e else float("nan")), prof


def _profile_graph(graph, active_iters: int, prof_warmup: int = 5):
    """torch.profiler over CUDAGraph replay (skill section 3). No barrier between
    replays (no-reset). Returns (kernel_us_per_iter, e2e_us_per_iter, prof)."""
    n_steps = 1 + prof_warmup + active_iters
    e2e = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with _make_profiler(active_iters, prof_warmup) as prof:
        for s in range(n_steps):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if s >= 1 + prof_warmup:
                e2e.append(start.elapsed_time(end) * 1000.0)
            prof.step()
    kern = _profiler_kernel_us_per_iter(prof, active_iters)
    return kern, (_median(e2e) if e2e else float("nan")), prof


def _save_trace(prof, trace_dir: str, tag: str, rank: int):
    if not trace_dir:
        return None
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"{tag}_rank{rank}.json")
    prof.export_chrome_trace(path)
    return path


# ===========================================================================
# Serial baseline: aiter mori dispatch -> aiter.fused_moe -> mori combine.
# ===========================================================================
def _serial_worker(rank, world_size, args, out_path):
    import aiter
    from aiter import ActivationType, dtypes, get_hip_quant, get_torch_quant
    from aiter.fused_moe import fused_moe, fused_topk
    from aiter.ops.shuffle import (
        shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4,
    )
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.utility import fp4_utils
    from aiter.dist.utils import get_distributed_init_method
    from torch.profiler import record_function

    os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank)

    quant = args.quant
    is_a8w4 = quant == "a8w4"
    quant_type = aiter.QuantType.per_1x32 if is_a8w4 else getattr(aiter.QuantType, quant)
    E = world_size * args.epr
    N = 2 * args.ih

    init_method = get_distributed_init_method("127.0.0.1", args.port)
    aiter.init_dist_env(
        tensor_model_parallel_size=1, rankID=0, backend="nccl",
        distributed_init_method=init_method,
        data_parallel_size=world_size, data_parallel_rank=rank,
    )
    from aiter.dist.parallel_state import get_ep_group
    ep_group = get_ep_group()

    result = {}
    try:
        tokens = torch.randn((args.tokens, args.hidden), dtype=dtypes.bf16, device=device)
        score = _skewed_scores(args.tokens, E, getattr(args, "skew", 0.0), device).to(dtypes.bf16)
        topk_weights, topk_ids = fused_topk(tokens, score, args.topk, True)

        w1 = torch.randn((args.epr, N, args.hidden), dtype=dtypes.bf16, device=device) * 0.3
        w2 = torch.randn((args.epr, args.hidden, args.ih), dtype=dtypes.bf16, device=device) * 0.3
        is_fp4 = quant_type == aiter.QuantType.per_1x32
        gate_mode = GateMode.SEPARATED.value

        if is_a8w4:
            wq = get_torch_quant(aiter.QuantType.per_1x32)
            w1_qt, w1_scale = wq(w1, quant_dtype=dtypes.fp4x2)
            w2_qt, w2_scale = wq(w2, quant_dtype=dtypes.fp4x2)
            w1_qt = w1_qt.view(args.epr, N, args.hidden // 2)
            w2_qt = w2_qt.view(args.epr, args.hidden, args.ih // 2)
            w1_qt = shuffle_weight_a16w4(w1_qt, 16, True)
            w1_scale = shuffle_scale_a16w4(w1_scale, args.epr, True)
            w2_qt = shuffle_weight_a16w4(w2_qt, 16, False)
            w2_scale = shuffle_scale_a16w4(w2_scale, args.epr, False)
            gate_mode = GateMode.INTERLEAVE.value
            tokens_qt, scale = get_hip_quant(aiter.QuantType.per_1x32)(
                tokens, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
        elif is_fp4:
            wq = get_torch_quant(aiter.QuantType.per_1x32)
            w1_qt, w1_scale = wq(w1, quant_dtype=dtypes.fp4x2)
            w2_qt, w2_scale = wq(w2, quant_dtype=dtypes.fp4x2)
            w1_qt = shuffle_weight(w1_qt.view(args.epr, N, args.hidden // 2), layout=(16, 16))
            w2_qt = shuffle_weight(w2_qt.view(args.epr, args.hidden, args.ih // 2), layout=(16, 16))
            w1_scale = fp4_utils.e8m0_shuffle(w1_scale)
            w2_scale = fp4_utils.e8m0_shuffle(w2_scale)
            w1_qt.is_shuffled = True
            w2_qt.is_shuffled = True
            tokens_qt, scale = get_hip_quant(aiter.QuantType.per_1x32)(
                tokens, quant_dtype=dtypes.fp4x2)
        else:
            wq = get_torch_quant(
                quant_type if quant_type != aiter.QuantType.per_128x128
                else aiter.QuantType.per_1x128)
            w1_qt, w1_scale = wq(w1, quant_dtype=dtypes.fp8)
            w2_qt, w2_scale = wq(w2, quant_dtype=dtypes.fp8)
            w1_qt = shuffle_weight(w1_qt)
            w2_qt = shuffle_weight(w2_qt)
            qfunc = get_hip_quant(
                quant_type if quant_type != aiter.QuantType.per_128x128
                else aiter.QuantType.per_1x128)
            tokens_qt, scale = qfunc(tokens, quant_dtype=dtypes.fp8)

        mori_manager = ep_group.device_communicator.all2all_manager
        assert mori_manager is not None, f"rank {rank}: all2all_manager is None"
        handle_kwargs = dict(
            rank=rank, num_ep_ranks=world_size,
            input_dtype=tokens.dtype, quant_dtype=tokens_qt.dtype,
            token_hidden_size=args.hidden,
            scale_dim=scale.shape[-1] if scale is not None else 0,
            scale_type_size=scale.dtype.itemsize if scale is not None else 0,
            max_num_tokens_per_dp_rank=2 * 8192 * 1024 // tokens_qt.dtype.itemsize // args.hidden * 2,
            num_local_experts=args.epr, num_experts_per_token=args.topk,
            gpu_per_node=1,
        )
        mori_op = mori_manager.get_handle(handle_kwargs)

        expert_mask = torch.zeros((E,), dtype=dtypes.i32, device=device)
        expert_mask[args.epr * rank: args.epr * (rank + 1)] = 1

        def _chain_tagged():
            with record_function("serial::dispatch"):
                (d_out, d_w, d_s, d_ids, d_recv) = mori_op.dispatch(
                    tokens_qt, topk_weights, scale, topk_ids)
            with record_function("serial::fused_moe"):
                out = fused_moe(
                    d_out, w1_qt, w2_qt, d_w, d_ids, expert_mask,
                    num_local_tokens=d_recv, activation=ActivationType.Silu,
                    gate_mode=gate_mode, w1_scale=w1_scale, w2_scale=w2_scale,
                    a1_scale=d_s, quant_type=quant_type, dtype=tokens.dtype)
            with record_function("serial::combine"):
                mori_op.combine(out, topk_weights, topk_ids)

        def _chain():
            (d_out, d_w, d_s, d_ids, d_recv) = mori_op.dispatch(
                tokens_qt, topk_weights, scale, topk_ids)
            out = fused_moe(
                d_out, w1_qt, w2_qt, d_w, d_ids, expert_mask,
                num_local_tokens=d_recv, activation=ActivationType.Silu,
                gate_mode=gate_mode, w1_scale=w1_scale, w2_scale=w2_scale,
                a1_scale=d_s, quant_type=quant_type, dtype=tokens.dtype)
            mori_op.combine(out, topk_weights, topk_ids)

        result = _measure(
            rank, world_size, args, op_tag="serial",
            eager_step=_chain, profile_step=_chain_tagged,
            capture_step=_chain, host_barrier=None)
    finally:
        if rank == 0 and result:
            with open(out_path, "w") as f:
                json.dump(result, f)
        try:
            dist.barrier()
        except Exception:
            pass
        aiter.destroy_dist_env()


# ===========================================================================
# Fused kernel: FlyDSLMegaMoEIntraNodeOp.fp8_fp4_mega_moe.
# ===========================================================================
def _fused_worker(rank, world_size, args, out_path):
    import mori.shmem as ms
    from kernels.mega_moe_intranode_op import (
        MegaMoEIntraNodeConfig, FlyDSLMegaMoEIntraNodeOp,
    )
    from torch.profiler import record_function
    # per_token_cast_to_fp8 / per_1x32_mxfp4_quant / e8m0_shuffle / shuffle_weight
    # are inlined at module scope above (this file is self-contained).

    os.environ.update(dict(
        LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost", MASTER_PORT=str(args.port + 1),
    ))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    import torch._C._distributed_c10d as c10d
    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")

    def _prep_weight(w_f32):
        w_fp4, w_scale_u8 = per_1x32_mxfp4_quant(w_f32)
        w_sh = shuffle_weight(w_fp4.view(torch.uint8), layout=(16, 16)).contiguous()
        w_ck = e8m0_shuffle(w_scale_u8).contiguous().view(torch.int32)
        return (w_sh, w_ck)

    result = {}
    try:
        E = world_size * args.epr
        N = 2 * args.ih
        cfg = MegaMoEIntraNodeConfig(
            rank=rank, world_size=world_size, num_experts=E,
            num_topk=args.topk, hidden=args.hidden, intermediate_hidden=args.ih,
            num_max_tokens_per_rank=args.tokens, block_m=getattr(args, "block_m", 128),
        )
        op = FlyDSLMegaMoEIntraNodeOp(cfg)

        x = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
        x_fp8, x_sf = per_token_cast_to_fp8(x, gran_k=32)
        scores = _skewed_scores(args.tokens, E, getattr(args, "skew", 0.0), device)
        tw, ti = torch.topk(scores, args.topk, dim=-1)
        if rank == 0:
            ratio, _ = _expert_load_stats(ti, E)
            print(f"[fused rank0] skew={getattr(args, 'skew', 0.0)} "
                  f"expert-load max/mean={ratio:.2f}x", flush=True)
        op.x[:args.tokens].copy_(x_fp8)
        op.x_sf[:args.tokens].copy_(x_sf)
        op.topk_idx[:args.tokens].copy_(ti.to(torch.int64))
        op.topk_weights[:args.tokens].copy_(tw.to(torch.float32))

        # Phase-incremental staging (Tier-0 attribution): the op enables the
        # L1 / L2 compute phases based on whether real preshuffled weights are
        # supplied (mega_moe_intranode_op.py:580/590).  A 1-elem int8 placeholder
        # -> that phase is skipped (dispatch parity pattern, test_mega_moe_dispatch.py:405).
        #   dispatch : dispatch (Phase A-E) + its grid/nvlink barriers only
        #   l1       : + L1 GEMM + SwiGLU (no L2, no combine)
        #   full     : + L2 GEMM + combine (writes y)
        phase = getattr(args, "phase", "full")
        _plc = torch.zeros((1,), dtype=torch.int8, device=device)
        # Deterministic per-GLOBAL-expert weights (this rank owns experts
        # [rank*epr, (rank+1)*epr)); keyed by global id so the f32 oracle can
        # rebuild ANY expert's weights for the accuracy check.
        def _l1_real():
            w = torch.cat([_gen_w1(rank * args.epr + e, N, args.hidden, device)
                           for e in range(args.epr)], dim=0)
            return _prep_weight(w)

        def _l2_real():
            w = torch.cat([_gen_w2(rank * args.epr + e, args.hidden, args.ih, device)
                           for e in range(args.epr)], dim=0)
            return _prep_weight(w)
        if phase == "dispatch":
            l1, l2 = (_plc, _plc), (_plc, _plc)
        elif phase == "l1":
            l1, l2 = _l1_real(), (_plc, _plc)
        else:
            l1, l2 = _l1_real(), _l2_real()
        stats = torch.zeros((args.epr,), dtype=torch.int32, device=device)
        y = torch.zeros(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)

        def _one():
            op.fp8_fp4_mega_moe(y, l1, l2,
                                cumulative_local_expert_recv_stats=stats,
                                recipe=(1, 1, 32), activation="swiglu")

        def _one_tagged():
            with record_function("fused::mega_moe"):
                _one()

        # The fused op issues a host-side ms.shmem_barrier_all() inside the
        # timed call; expose a stub-swap so the cudagraph path can no-op it
        # during capture+replay (skill no-reset rule).
        def _host_barrier_ctx():
            return _ShmemBarrierStub(ms)

        result = _measure(
            rank, world_size, args, op_tag="fused",
            eager_step=_one, profile_step=_one_tagged,
            capture_step=_one, host_barrier=_host_barrier_ctx)

        # ---- accuracy: relL2 vs full-precision f32 oracle (full phase only;
        #      `y` holds the last timed run's output) ----
        if phase == "full":
            try:
                rl_local = _fused_relL2(
                    y[:args.tokens], x[:args.tokens], ti, tw,
                    N=N, hidden=args.hidden, ih=args.ih, topk=args.topk, device=device)
            except Exception as _e:  # noqa: BLE001
                rl_local = -1.0
            per = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(per, torch.tensor([float(rl_local)], device=device))
            rls = [float(t.item()) for t in per]
            if rank == 0 and result:
                result["acc_rell2_per_rank"] = rls
                result["acc_floor"] = float(args.acc_floor)
                result["acc_pass"] = all(0 <= v <= args.acc_floor for v in rls)
        op.destroy()
    finally:
        if rank == 0 and result:
            with open(out_path, "w") as f:
                json.dump(result, f)
        try:
            ms.shmem_finalize()
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass


class _ShmemBarrierStub:
    """Context manager that swaps ``ms.shmem_barrier_all`` for a no-op so the
    host-side collective inside the fused op does not abort CUDAGraph capture.
    A single real barrier is issued on ``__enter__`` to leave all ranks in sync
    before capture begins."""

    def __init__(self, ms):
        self._ms = ms
        self._orig = ms.shmem_barrier_all

    def __enter__(self):
        self._orig()
        self._ms.shmem_barrier_all = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        self._ms.shmem_barrier_all = self._orig
        self._orig()
        return False


# ===========================================================================
# The 4-way measurement core (profile|bench x eager|cudagraph), per skill.
# ===========================================================================
def _measure(rank, world_size, args, *, op_tag,
             eager_step, profile_step, capture_step, host_barrier):
    """Run the requested (mode x cudagraph) cell for one operator and return a
    rank-0 result dict.  Warmup is mode-matched (skill: eager warmup for eager
    runs, replay warmup for cudagraph runs -- no cross warmup)."""
    iters, warmup = args.iters, args.warmup

    def _sync():
        torch.cuda.synchronize()
        try:
            dist.barrier()
        except Exception:
            pass

    res = {"op": op_tag, "mode": args.mode, "cudagraph": bool(args.cudagraph)}

    # ----------------------------------------------------------------- eager
    if not args.cudagraph:
        for _ in range(warmup):           # eager warmup (JIT + GPU cache)
            eager_step()
        _sync()
        if args.mode == "bench":
            t_local = _bench_eager(eager_step, iters)
            avg, mn, mx = _allreduce_avg_min_max(t_local, world_size)
            res.update(e2e_us=avg, e2e_us_min=mn, e2e_us_max=mx)
        else:  # profile
            kern, e2e, prof = _profile_eager(profile_step, iters)
            tp = _save_trace(prof, args.trace_dir, f"{op_tag}_eager", rank)
            k_avg, _, k_mx = _allreduce_avg_min_max(kern, world_size)
            e_avg, _, e_mx = _allreduce_avg_min_max(e2e, world_size)
            res.update(kernel_us=k_avg, kernel_us_max=k_mx,
                       e2e_us=e_avg, e2e_us_max=e_mx, trace=tp)
        return res if rank == 0 else {}

    # ------------------------------------------------------------- cudagraph
    # eager warmup first: populate JIT/compiled caches BEFORE capture.
    for _ in range(warmup):
        eager_step()
    _sync()

    graph = torch.cuda.CUDAGraph()
    cap_stream = torch.cuda.Stream()
    barrier_cm = host_barrier() if host_barrier is not None else _NullCtx()
    capture_ok, capture_err = False, ""
    try:
        with barrier_cm:
            # warm the capture stream once (skill: side-stream warmup) then capture
            with torch.cuda.stream(cap_stream):
                capture_step()
            cap_stream.synchronize()
            with torch.cuda.graph(graph, stream=cap_stream):
                capture_step()
            capture_ok = True
    except Exception as e:  # noqa: BLE001
        capture_err = f"{type(e).__name__}: {e}"

    if not capture_ok:
        res.update(capture_ok=False, capture_err=capture_err)
        # Fall back to eager numbers so the cell still reports something useful.
        for _ in range(warmup):
            eager_step()
        _sync()
        t_local = _bench_eager(eager_step, iters)
        avg, _, mx = _allreduce_avg_min_max(t_local, world_size)
        res.update(e2e_us=avg, e2e_us_max=mx, note="eager fallback (capture failed)")
        return res if rank == 0 else {}

    # cudagraph warmup: replay (HIP graph cold-start + cache warm), no barrier.
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()

    if args.mode == "bench":
        t_local = _bench_graph(graph, iters)
        avg, mn, mx = _allreduce_avg_min_max(t_local, world_size)
        res.update(capture_ok=True, e2e_us=avg, e2e_us_min=mn, e2e_us_max=mx)
    else:  # profile
        kern, e2e, prof = _profile_graph(graph, iters)
        tp = _save_trace(prof, args.trace_dir, f"{op_tag}_cudagraph", rank)
        k_avg, _, k_mx = _allreduce_avg_min_max(kern, world_size)
        e_avg, _, e_mx = _allreduce_avg_min_max(e2e, world_size)
        res.update(capture_ok=True, kernel_us=k_avg, kernel_us_max=k_mx,
                   e2e_us=e_avg, e2e_us_max=e_mx, trace=tp)
    return res if rank == 0 else {}


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ===========================================================================
# Spawn / orchestration.
# ===========================================================================
def _spawn(worker, world_size, args, out_path):
    torch.multiprocessing.spawn(worker, args=(world_size, args, out_path), nprocs=world_size)
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {}


def _print_cell(label, r):
    if not r:
        print(f"  {label:8s}: (no result)")
        return
    if r.get("capture_ok") is False:
        print(f"  {label:8s}: CUDAGraph capture FAILED -> {r.get('capture_err')}")
        if r.get("e2e_us") is not None:
            print(f"            (eager fallback E2E: {r['e2e_us']/1000:.4f} ms)")
        return
    parts = []
    if r.get("e2e_us") is not None:
        parts.append(f"E2E={r['e2e_us']/1000:.4f} ms")
    if r.get("kernel_us") is not None:
        parts.append(f"kernel={r['kernel_us']/1000:.4f} ms")
    if r.get("e2e_us_max") is not None:
        parts.append(f"(E2E max-rank={r['e2e_us_max']/1000:.4f})")
    print(f"  {label:8s}: " + "  ".join(parts))
    if r.get("trace"):
        print(f"            trace: {r['trace']}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", choices=["serial", "fused", "both"], default="both")
    p.add_argument("--mode", choices=["profile", "bench"], default="bench")
    p.add_argument("--cudagraph", action="store_true",
                   help="CUDAGraph capture+replay (default: eager)")
    p.add_argument("--world-size", type=int, default=int(os.environ.get("BENCH_WORLD_SIZE", "2")))
    p.add_argument("--epr", type=int, default=8, help="local experts/rank (E = epr*world)")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048, help="must be %%256 for fused L1")
    p.add_argument("--ih", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=2048, help="tokens per rank")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--quant", default="per_1x32",
                   choices=["per_1x32", "a8w4", "per_Token", "per_128x128"])
    p.add_argument("--port", type=int, default=int(os.environ.get("BENCH_PORT", "29955")))
    p.add_argument("--trace-dir", default="", help="export chrome traces here (profile mode)")
    p.add_argument("--phase", choices=["dispatch", "l1", "full"], default="full",
                   help="fused-op phase staging for Tier-0 attribution (placeholder weights)")
    p.add_argument("--block-m", type=int, default=64,
                   help="per-expert pool BLOCK_M for the fused op; must be a "
                        "candidate value {8,16,32,64,96,128,192} (default 64: "
                        "measured fastest on gfx950)")
    p.add_argument("--skew", type=float, default=0.0,
                   help="expert-load imbalance: 0 = balanced random routing (default); "
                        ">0 adds a Zipf-like per-expert popularity bias to the gating "
                        "scores so a few experts capture most tokens (tail-effect test)")
    p.add_argument("--acc-floor", type=float,
                   default=float(os.environ.get("MEGA_MOE_ACC_FLOOR", "0.30")),
                   help="fused accuracy gate: PASS if relL2 vs the f32 oracle <= this "
                        "(a8w4 mxfp4-weight quant floor; default 0.30). Only --phase full.")
    return p.parse_args()


def main():
    args = _parse_args()
    world_size = args.world_size
    if torch.cuda.device_count() < world_size:
        print(f"[skip] need >= {world_size} GPUs, have {torch.cuda.device_count()}")
        return

    exec_tag = "cudagraph" if args.cudagraph else "eager"
    geom = (f"world={world_size} E={world_size * args.epr} (epr={args.epr}) "
            f"topk={args.topk} hidden={args.hidden} ih={args.ih} "
            f"tokens/rank={args.tokens} quant={args.quant} phase={args.phase} "
            f"skew={args.skew} block_m={args.block_m} "
            f"mode={args.mode}+{exec_tag} (warmup={args.warmup} iters={args.iters})")

    serial = fused = {}
    tmpdir = tempfile.mkdtemp(prefix="bench_mega_moe_")
    if args.op in ("serial", "both"):
        serial = _spawn(_serial_worker, world_size, args, os.path.join(tmpdir, "serial.json"))
    if args.op in ("fused", "both"):
        fused = _spawn(_fused_worker, world_size, args, os.path.join(tmpdir, "fused.json"))

    print(f"\n=== mega_moe accuracy + perf: {args.mode}+{exec_tag} ===")
    print(f"  {geom}")
    # ---- accuracy (fused, relL2 vs f32 oracle) ----
    if fused.get("acc_rell2_per_rank") is not None:
        rls = fused["acc_rell2_per_rank"]
        floor = fused.get("acc_floor", args.acc_floor)
        npass = sum(0 <= v <= floor for v in rls)
        verdict = "PASS" if fused.get("acc_pass") else "FAIL"
        per = "  ".join(f"r{r}={v:.4e}" for r, v in enumerate(rls))
        print(f"  [accuracy fused]  relL2 max={max(rls):.4e} floor={floor:.2f}  "
              f"{verdict}  ({npass}/{len(rls)} cards)")
        print(f"      per-rank relL2: {per}")
    elif args.op in ("fused", "both") and args.phase != "full":
        print(f"  [accuracy fused]  skipped (--phase {args.phase}; needs full)")
    # ---- perf ----
    _print_cell("serial", serial)
    _print_cell("fused", fused)
    ts = serial.get("e2e_us")
    tf = fused.get("e2e_us")
    if ts and tf:
        print(f"  --> speedup (T_serial / T_fused) : {ts / tf:.2f}x")
    print()


if __name__ == "__main__":
    main()
