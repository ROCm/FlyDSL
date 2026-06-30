# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Whole-operator performance harness for the **serial** (aiter mori) MoE chain
and the **fused** ``mega_moe`` kernel, written to the ``comm-op-bench`` skill
(``testscript-SKILL.md``).

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

    # bench + eager (closest to production E2E)
    HIP_VISIBLE_DEVICES=0,1 python3 tests/kernels/test_mega_moe_op_perf.py \\
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

import torch
import torch.distributed as dist

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)


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
    from tests.kernels.test_mega_moe_dispatch import per_token_cast_to_fp8
    from tests.kernels.test_mega_moe_l1_gemm import per_1x32_mxfp4_quant
    from tests.kernels.utils import fp4_utils
    from tests.utils import shuffle_weight
    from torch.profiler import record_function

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
        w_ck = fp4_utils.e8m0_shuffle(w_scale_u8).contiguous().view(torch.int32)
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
        _l1_real = lambda: _prep_weight(torch.randn(args.epr * N, args.hidden, device=device) * 0.3)
        _l2_real = lambda: _prep_weight(torch.randn(args.epr * args.hidden, args.ih, device=device) * 0.3)
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
    tmpdir = tempfile.mkdtemp(prefix="test_mega_moe_op_perf_")
    if args.op in ("serial", "both"):
        serial = _spawn(_serial_worker, world_size, args, os.path.join(tmpdir, "serial.json"))
    if args.op in ("fused", "both"):
        fused = _spawn(_fused_worker, world_size, args, os.path.join(tmpdir, "fused.json"))

    print(f"\n=== mega_moe op perf: {args.mode}+{exec_tag} ===")
    print(f"  {geom}")
    _print_cell("serial", serial)
    _print_cell("fused", fused)
    ts = serial.get("e2e_us")
    tf = fused.get("e2e_us")
    if ts and tf:
        print(f"  --> speedup (T_serial / T_fused) : {ts / tf:.2f}x")
    print()


if __name__ == "__main__":
    main()
