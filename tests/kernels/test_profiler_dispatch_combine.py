# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Performance harness for FlyDSL and mori-ref dispatch/combine kernels.

Two orthogonal axes can be freely combined:
  --mode       measurement: ``profile`` (torch.profiler) | ``bench``
               (CUDA event timing) | ``verify`` (correctness check)
  --cudagraph  execution:   absent = eager mode | present =
               CUDAGraph capture+replay

Four combinations:
  1. profile + eager    : torch.profiler over eager kernels + E2E +
                          CPU timing
  2. bench   + eager    : CUDA event timing of eager dispatch/combine
                          (no profiler overhead)
  3. profile + cudagraph: torch.profiler over CUDAGraph replay kernels
  4. bench   + cudagraph: CUDA event timing of CUDAGraph replay
                          (zero Python launch overhead)

Launching (works under torchrun or plain python):
  # profile + eager (default)
  python tests/kernels/test_profiler_dispatch_combine.py --max-tokens 512

  # bench + eager
  python tests/kernels/test_profiler_dispatch_combine.py --mode bench

  # bench + cudagraph
  python tests/kernels/test_profiler_dispatch_combine.py --mode bench --cudagraph

  # profile + cudagraph
  python tests/kernels/test_profiler_dispatch_combine.py --mode profile --cudagraph

  # FlyDSL + mori head-to-head perf comparison
  python tests/kernels/test_profiler_dispatch_combine.py --compare-mori
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

# dtype mapping
# Keys mirror the ``torch.<dtype>`` repr suffix where applicable
# (``float8_e4m3fn`` / ``float8_e4m3fnuz``) for PyTorch parity.  ``bf16``
# / ``f32`` / ``fp4`` are kept as widely-used short aliases.
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "fp4": torch.float4_e2m1fn_x2,
}

# Values are mori's internal kernel-symbol suffix (private ABI of
# ``mori/python/mori/ops/dispatch_combine.py::_DTYPE_SUFFIX``); keep them
# verbatim so we can still locate the right ``ep_intranode_*.hsaco``
# symbol at run time.
MORI_KERNEL_SUFFIX = {
    "bf16": "bf16",
    "f32": "f32",
    "float8_e4m3fn": "fp8_ocp",
    "float8_e4m3fnuz": "fp8_fnuz",
    "fp4": "fp4",
}


# -----------------------------------------------------------------------------
# Mori-parity launch-time dtype helpers
# -----------------------------------------------------------------------------
def _resolve_combine_dtype(args, dispatch_dtype):
    """Resolve the launch-time combine dtype.  Empty / missing
    ``--combine-dtype`` reverts to the dispatch dtype (mori parity:
    same caller dtype throughout).  Used by the perf sweep to mix
    ``fp4`` / ``fp8_ocp`` dispatch with ``bf16`` combine in the same
    op."""
    cstr = getattr(args, "combine_dtype", "") or ""
    if not cstr:
        return dispatch_dtype
    return DTYPE_MAP[cstr]


_FP4_LUT_F32_CACHE: dict = {}


def _fp4_to_f32(t):
    """Unpack ``float4_e2m1fn_x2`` to ``float32``.

    Mirrors ``utils.fp4_utils.mxfp4_to_f32`` but caches the 16-entry LUT
    on each device so the repeated cast does NOT allocate fresh tensors
    under CUDA graph capture (``torch.tensor(...)`` inside a captured
    region raises "operation not permitted when stream is capturing").
    """
    x = t.view(torch.uint8) if t.dtype == torch.float4_e2m1fn_x2 else t
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    dev = x.device
    lut = _FP4_LUT_F32_CACHE.get(dev)
    if lut is None:
        # mxfp4 e2m1 nibble decode table; entries 0..7 are positive,
        # 8..15 are the sign-bit-set negatives.  Identical to
        # ``utils.fp4_utils.mxfp4_to_f32``'s in-line list.
        _vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        lut = torch.tensor(_vals, dtype=torch.float32, device=dev)
        _FP4_LUT_F32_CACHE[dev] = lut
    return lut[x.long()]


def _cast_dispatch_to_combine(t, combine_dtype):
    """Cast a dispatch-out tensor to ``combine_dtype``.

    fp4 dispatch outputs are stored in ``float4_e2m1fn_x2`` packed format
    (2 fp4 elements per byte, last-dim halved); PyTorch has no native
    cast kernel today, so we route through the project-local
    ``_fp4_to_f32`` unpacker (cudagraph-safe, see :func:`_fp4_to_f32`)
    which expands the last dim by 2 and decodes each nibble via a
    pre-cached 16-entry LUT.  Same-dtype is a no-op (returns the input
    unchanged so we keep zero-copy on the matched path)."""
    if t.dtype == combine_dtype:
        return t
    if t.dtype == torch.float4_e2m1fn_x2:
        return _fp4_to_f32(t).to(combine_dtype)
    return t.to(combine_dtype)


def _op_zero_copy(op):
    """Return the op's ``zero_copy`` switch across mori/FlyDSL configs."""
    cfg = getattr(op, "cfg", None) or getattr(op, "config", None)
    if hasattr(cfg, "zero_copy"):
        return bool(getattr(cfg, "zero_copy"))
    return not bool(getattr(cfg, "use_external_inp_buf", True))


def _run_combine(op, ret, combine_dtype, **kwargs):
    """Mori-parity combine launcher.

    Centralises the zero-copy caller contract so every test-side combine
    invocation goes through the same staging path:

      - ``zero_copy=False`` -> just dtype-cast ``ret[0]`` and
        let the kernel do its own Stage 1 staging copy.
      - ``zero_copy=True`` -> caller MUST write
        the combine input into the buffer returned by
        ``op.get_registered_combine_input_buffer(combine_dtype)`` before
        invoking ``combine``; the kernel runs ``skip_stage1=True`` and
        peers read that buffer on Stage 3.
    """
    src = _cast_dispatch_to_combine(ret[0], combine_dtype)
    if _op_zero_copy(op):
        # Mori-parity zero-copy caller contract: caller MUST pre-stage
        # token bytes into the symmetric ``shmem_comb_inp_tok`` buffer
        # returned by :meth:`get_registered_combine_input_buffer`. The
        # kernel-side Stage 1 token copy is compile-time eliminated and
        # peer PEs read this buffer directly during Stage 3.
        cb = op.get_registered_combine_input_buffer(combine_dtype)
        n = src.shape[0]
        if n > 0:
            cb[:n].copy_(src)
        inp_for_kernel = cb
    else:
        inp_for_kernel = src
    return op.combine(inp_for_kernel, None, ret[3], **kwargs)


# ============================================================================
# CI sweep cases
# ----------------------------------------------------------------------------
# Curated subset of mori's tuning configs
# (mori/python/mori/ops/tuning_configs/gfx950_mi355x_IntraNode_ep8_*.json),
# augmented with StdMoE cases (mori has no StdMoE entries in its table).
#
# Each entry is a self-contained ``cfg`` dict that overrides ``args``
# fields when the runner is invoked with ``--ci-sweep``.  For every case,
# ``_run_ci_sweep`` does two things in order:
#   1) ``--mode verify``              — accuracy gate (PASS / FAIL)
#   2) ``--mode profile --cudagraph`` — perf gate (writes JSON trace)
#
# Coverage axes (matches mori tuning table + StdMoE):
#   - dtype          : bf16 (combine), fp8_ocp (dispatch), fp4 (dispatch)
#   - quant_type     : none, fp8_direct_cast
#   - zero_copy      : True (zero-copy), False (non-zero-copy)
#   - enable_std_moe : True, False
#   - token bucket   : 128 (small), 4096 (large)
#
# Default shape for every case (unless overridden):
#   world_size=8, k=8, num_experts_per_rank=32,
#   block_num=80, warp_per_block=4 (FlyDSL defaults).
# ============================================================================
CI_CASES = [
    {
        "name": "bf16_baseline",
        "dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
    },
    {
        "name": "bf16_fp8_direct_cast",
        "dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "fp8_direct_cast",
        "zero_copy": False,
        "enable_std_moe": False,
    },
    {
        "name": "bf16_zero_copy",
        "dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": True,
        "enable_std_moe": False,
        # Verified via shmem dumps that FlyDSL's zero-copy combine path is
        # correct (Stage 1 fills shmem_comb_inp_tok and Stage 3 writes
        # ``k*inp`` into shmem_comb_out_tok); the previous "zero output"
        # symptom was actually mori's zero-copy reference returning all
        # zeros.  Accuracy is now validated by the FlyDSL self-check
        # (``fly == k*inp``) so this case can participate in the full
        # sweep (accuracy + profile).
    },
    {
        "name": "bf16_std_moe",
        "dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": True,
    },
    {
        "name": "bf16_baseline_large",
        "dtype": "bf16",
        "max_tokens": 4096,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
    },
    {
        "name": "bf16_fp8_direct_cast_large",
        "dtype": "bf16",
        "max_tokens": 4096,
        "hidden_dim": 7168,
        "quant_type": "fp8_direct_cast",
        "zero_copy": False,
        "enable_std_moe": False,
    },
    # Two fp8 dispatch flavours that share the kernel's ``hidden_elem_size==1``
    # codepath (``cvt_pk_f32_fp8`` + ×0.5/×2.0 scaling for FNUZ).  Both should
    # run on whatever arch the kernel JIT actually supports; we don't gate on
    # arch here because the kernel is gfx942 native (verified by inspecting
    # the emitted ISA: ``v_cvt_pk_f32_fp8_e32`` / ``_sdwa``).  If CI fails on
    # a specific arch, fix the underlying codegen / runtime issue rather than
    # silently skipping coverage.
    {
        "name": "float8_e4m3fn_dispatch",
        "dtype": "float8_e4m3fn",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
        # OCP fp8 (``float8_e4m3fn``) is a gfx950+ device-side feature:
        # ROCm's ``amd_hip_fp8.h`` defines ``HIP_FP8_TYPE_OCP=0`` for
        # ``__gfx942__`` device compile, so mori's ``MORI_FP8_OCP(...)``
        # macros are dropped at JIT-compile time and
        # ``EpDispatchIntraNodeKernel_fp8_ocp`` is never generated.  The
        # host-side mori launcher unconditionally dispatches to that
        # symbol on ``torch.float8_e4m3fn`` and the resulting
        # ``hipModuleGetFunction`` returns ``hipErrorNotFound``,
        # poisoning the HIP context.  Gate on gfx950 (parity with the
        # fp4 cases below) instead of pretending this works on gfx942.
        "requires_arch": ("gfx950",),
    },
    {
        "name": "float8_e4m3fnuz_dispatch",
        "dtype": "float8_e4m3fnuz",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
    },
    {
        "name": "fp4_dispatch",
        "dtype": "fp4",
        "max_tokens": 128,
        "hidden_dim": 3584,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
        # fp4 dispatch emits ``v_cvt_scalef32_pk_f32_fp4`` which only
        # exists on gfx950+ (MI355x); MI300/MI325 (gfx942) cannot compile it.
        "requires_arch": ("gfx950",),
    },
    {
        "name": "bf16_std_moe_large",
        "dtype": "bf16",
        "max_tokens": 4096,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": True,
    },
    # ---- Mori-parity coverage (intranode) ----
    # max_total_recv_tokens cap + max_token_type_size override (mori
    # parity for memory-bound serving).
    #
    # cap == worst-case max_recv == ``world_size * max_num_inp_token_per_rank``
    # (8 * 128 = 1024) so this case exercises the recv-cap codepath
    # without forcing the dispatch kernel to drop slots: every randomly-
    # routed token still fits into the per-dest receive buffer, mori (which
    # also runs without a cap in build_mori_ref) sees the same workload,
    # and combine byte-equality holds.  An "overflow" case where ``cap <
    # actual receive`` is tracked separately: the dispatch-side fix in
    # ``make_dispatch_kernel`` now drops the overflowing slots gracefully
    # (instead of hipErrorIllegalAddress), but mori asserts on the same
    # configuration ("caller must guarantee actual routing stays within
    # the cap" -- see ``EpDispatchCombineConfig.max_total_recv_tokens``),
    # so cross-impl byte parity is undefined and would need a
    # ``verify_overflow_drop`` mode in ``verify_self`` (out of scope here).
    {
        "name": "bf16_recv_cap_token_size",
        "dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
        "max_total_recv_tokens": 1024,
        "max_token_type_size": 2,
    },
    # bf16 + cap=512 (half of the worst-case ws*M=1024).  Exercises the
    # mori-aligned ceil-division: per-rank slot count drops to
    # ceil(512/8)=64 < M=128, so symmetric token / metadata buffers
    # shrink linearly to 64*8 = 512 slots.  ``build_mori_ref`` now also
    # forwards the cap (see :func:`build_mori_ref` for the rationale),
    # so mori and FlyDSL run with the same per-rank slot count and
    # ``verify_self`` keeps byte-exact equality as the hard gate.
    {
        "name": "bf16_recv_cap_half",
        "dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
        "max_total_recv_tokens": 512,
    },
    # ---- Mixed-dtype coverage (mori-parity launch-time dtype) ----
    # Each op now JIT-compiles a kernel specialized to ``input.dtype``
    # at every dispatch / combine call (mori parity), so a single op
    # can route fp4 / fp8_ocp inputs through dispatch and have combine
    # write back bf16.  These cases exercise that decoupled-dtype path
    # in both buffer modes (zero-copy and non-zero-copy).
    #
    # ``max_token_type_size`` is auto-derived inside ``run_profiler``
    # when ``--combine-dtype != --dtype`` (max of the dispatch and
    # combine element sizes), so we don't set it here.
    {
        "name": "mixed_float8_e4m3fn_dispatch_bf16_combine_zero_copy",
        "dtype": "float8_e4m3fn",
        "combine_dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": True,
        "enable_std_moe": False,
        # ``float8_e4m3fn`` (OCP) dispatch requires gfx950+ -- see the
        # case-level comment on ``float8_e4m3fn_dispatch`` above for the
        # full root-cause analysis (mori's fp8_ocp kernel is not
        # generated for gfx942 because ROCm sets
        # ``HIP_FP8_TYPE_OCP=0`` on device).
        "requires_arch": ("gfx950",),
    },
    {
        "name": "mixed_fp4_dispatch_bf16_combine_zero_copy",
        "dtype": "fp4",
        "combine_dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 3584,
        "quant_type": "none",
        "zero_copy": True,
        "enable_std_moe": False,
        # fp4 dispatch emits ``v_cvt_scalef32_pk_f32_fp4`` (gfx950+).
        "requires_arch": ("gfx950",),
    },
    {
        "name": "mixed_float8_e4m3fn_dispatch_bf16_combine_no_zero_copy",
        "dtype": "float8_e4m3fn",
        "combine_dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
        # OCP fp8 requires gfx950+; see ``float8_e4m3fn_dispatch`` above.
        "requires_arch": ("gfx950",),
    },
    # ---- bs sweep: fp8_ocp dispatch -> bf16 combine, randomized zero_copy ----
    # Three cases at distinct M (4 / 32 / 8192) to cover the small-batch,
    # moderate-batch and large-batch regimes of the mixed-dtype path.  In
    # each case ``zero_copy`` is picked at random (False or True) per
    # spawn, so over enough CI runs both buffer modes get exercised at
    # every M without doubling the case count.  ``random.choice`` runs
    # once per case in the parent before spawn, so all 8 ranks see the
    # same resolved value and the kernel launch geometry stays consistent.
    {
        "name": "mixed_float8_e4m3fn_dispatch_bf16_combine_bs4_random_zc",
        "dtype": "float8_e4m3fn",
        "combine_dtype": "bf16",
        "max_tokens": 4,
        "hidden_dim": 7168,
        "quant_type": "none",
        "enable_std_moe": False,
        "_random_fields": {
            "zero_copy": [False, True],
        },
        # OCP fp8 requires gfx950+; see ``float8_e4m3fn_dispatch`` above.
        "requires_arch": ("gfx950",),
    },
    {
        "name": "mixed_float8_e4m3fn_dispatch_bf16_combine_bs32_random_zc",
        "dtype": "float8_e4m3fn",
        "combine_dtype": "bf16",
        "max_tokens": 32,
        "hidden_dim": 7168,
        "quant_type": "none",
        "enable_std_moe": False,
        "_random_fields": {
            "zero_copy": [False, True],
        },
        # OCP fp8 requires gfx950+; see ``float8_e4m3fn_dispatch`` above.
        "requires_arch": ("gfx950",),
    },
    {
        "name": "mixed_float8_e4m3fn_dispatch_bf16_combine_bs8K_random_zc",
        "dtype": "float8_e4m3fn",
        "combine_dtype": "bf16",
        "max_tokens": 8192,
        "hidden_dim": 7168,
        "quant_type": "none",
        "enable_std_moe": False,
        "_random_fields": {
            "zero_copy": [False, True],
        },
        # OCP fp8 requires gfx950+; see ``float8_e4m3fn_dispatch`` above.
        "requires_arch": ("gfx950",),
    },
    # ---- FNUZ fp8 mixed-dtype coverage (gfx942 native) ----
    # MI300/MI325 (gfx942) only support NANOO fp8 (fnuz) on device; ROCm's
    # ``amd_hip_fp8.h`` exposes ``HIP_FP8_TYPE_FNUZ=1`` for ``__gfx942__``
    # and mori's JIT emits ``EpDispatchIntraNodeKernel_fp8_fnuz``.  This
    # mirrors the OCP ``mixed_float8_e4m3fn_dispatch_bf16_combine_no_zero_copy``
    # case (gfx950-only above) so that the mixed-dtype dispatch + bf16
    # combine path is exercised on gfx942 too -- FlyDSL self-check is
    # the primary gate and mori's ``fp8_fnuz`` intranode kernel
    # participates as the byte-level oracle.
    {
        "name": "mixed_float8_e4m3fnuz_dispatch_bf16_combine_no_zero_copy",
        "dtype": "float8_e4m3fnuz",
        "combine_dtype": "bf16",
        "max_tokens": 128,
        "hidden_dim": 7168,
        "quant_type": "none",
        "zero_copy": False,
        "enable_std_moe": False,
    },
]


# Fields in CI_CASES entries that are sweep-runner-only metadata and must
# NOT be forwarded as ``args`` overrides (else argparse Namespace gains
# bogus attrs that downstream code may consult).
_CI_META_FIELDS = {"name", "skip_profile", "requires_arch", "known_failure", "skip_ci", "_random_fields"}


def _current_gpu_arch_prefix() -> str:
    """Return e.g. 'gfx942' or 'gfx950' (strips feature flags).

    Returns empty string if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return ""
    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, "gcnArchName", "") or ""
    # gcnArchName looks like 'gfx942:sramecc+:xnack-'.
    return arch.split(":", 1)[0]


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Module-level skip when mori is unavailable AND we are being imported by
# pytest collection.  This file is a torchrun standalone script, but pytest
# still picks it up because the name matches ``test_*.py`` -- single-GPU CI
# runners don't install mori, so unconditional ``import mori`` would crash
# pytest collection.  We only trigger ``pytest.importorskip`` when pytest
# is the orchestrator (``"pytest" in sys.modules``), so direct
# ``python``/``torchrun`` invocations still surface a normal ImportError
# if mori is missing (instead of an opaque pytest Skipped exception).
if "pytest" in sys.modules:
    sys.modules["pytest"].importorskip(
        "mori",
        reason="dispatch/combine intranode test requires mori shmem (8-GPU multi-gpu CI only)",
    )

import mori.shmem as ms  # noqa: E402

from kernels.dispatch_combine_intranode_op import (  # noqa: E402
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)


# --- Distributed init ---
def setup_distributed(rank, world_size, master_port=29600):
    if "LOCAL_RANK" not in os.environ:
        os.environ.update(
            {
                "LOCAL_RANK": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(master_port),
            }
        )
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    # Torch 2.8's eager multi-backend path (``backend=cpu:gloo,cuda:nccl``
    # + ``device_id=...``) can fail during NCCL eager-connect on MI325/MI355
    # runners with ``Failed to CUDA calloc 6291456 bytes`` before the first
    # test case starts.  Use plain NCCL init here to match our actual usage
    # (all collectives in this script are CUDA-side) and avoid the eager
    # device allocation path.
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
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


# Dtypes for which mori's precompiled hsaco actually ships the
# ``EpDispatchIntraNodeKernel_<sfx>`` / ``EpCombineIntraNodeKernel_<sfx>_(no)p2p``
# kernel symbols (verified against ``~/.mori/jit/.../ep_intranode.hsaco``).
# Both fp8_ocp and fp4_e2m1fn_x2 are now present, so we keep them in the
# allowlist; if a future container drops a symbol the worker will raise
# at JIT lookup and ``build_mori_ref`` will catch and fall back gracefully.
_MORI_SUPPORTED_DTYPES = {
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.float4_e2m1fn_x2,
}


def build_mori_ref(rank, world_size, cfg, block_num: int = None, warp_per_block: int = None):
    if cfg.data_type not in _MORI_SUPPORTED_DTYPES:
        raise RuntimeError(
            f"mori ref kernel for dtype {cfg.data_type} is not available in this "
            f"container; will fall back to FlyDSL self-check"
        )
    from mori.ops.dispatch_combine import EpDispatchCombineConfig, EpDispatchCombineOp

    elem = torch.tensor([], dtype=cfg.data_type).element_size()
    mcfg = EpDispatchCombineConfig(
        data_type=cfg.data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=cfg.hidden_dim,
        scale_dim=cfg.num_experts_per_token,
        scale_type_size=4,
        max_token_type_size=cfg.max_token_type_size if cfg.max_token_type_size > 0 else elem,
        max_num_inp_token_per_rank=cfg.max_num_inp_token_per_rank,
        num_experts_per_rank=cfg.num_experts_per_rank,
        num_experts_per_token=cfg.num_experts_per_token,
        warp_num_per_block=warp_per_block if warp_per_block is not None else cfg.dispatch_warp_num_per_block,
        block_num=block_num if block_num is not None else cfg.dispatch_block_num,
        gpu_per_node=world_size,
        use_external_inp_buf=not cfg.zero_copy,
        quant_type=cfg.quant_type,
        # ``max_total_recv_tokens`` is intentionally NOT forwarded to
        # mori unless the cap equals the worst-case ws*M (effectively
        # uncapped).  mori treats the cap as a hard contract -- when
        # actual routing exceeds the per-rank slot share it aborts via
        # device-side assert (``intranode.hpp:133:
        # destTokId < MaxNumTokensToRecv()``), which permanently
        # corrupts the HIP context and breaks every subsequent kernel
        # in the run.  FlyDSL treats the same cap as a budget and
        # gracefully drops overflow into the dup-sentinel codepath, so
        # the two implementations are byte-comparable only when the
        # routing happens to fit within mori's contract.  Random-
        # routing CI case ``bf16_recv_cap_half`` cannot guarantee that,
        # so we keep mori uncapped and verify_self falls back to its
        # ``cap_shrinks`` liveness gate.  The ``bf16_recv_cap_token_size``
        # case (cap == ws*M) DOES fit and is exercised by the worst-
        # case path below.
        max_total_recv_tokens=(
            cfg.max_total_recv_tokens
            if cfg.max_total_recv_tokens == 0
            or cfg.max_total_recv_tokens == cfg.world_size * cfg.max_num_inp_token_per_rank
            else 0
        ),
    )
    return EpDispatchCombineOp(mcfg)


def _safe_op_reset(op):
    """Implementation-agnostic pre-launch sync.

    The FlyDSL intranode op no longer exposes a ``reset()`` method (the
    counters it would have reset are kernel-managed across replays); the
    mori reference op still has ``reset()`` and DOES need it because mori
    clears ``destPeTokenCounter`` etc. on the host side.  This helper
    calls ``reset()`` if the op has one, then issues a global shmem
    barrier (which both implementations need so every rank enters the
    next launch with matching state).
    """
    reset = getattr(op, "reset", None)
    if callable(reset):
        reset()
    ms.shmem_barrier_all()


def _save_profile_json(prof, out_path: str, rank: int, op_tag: str, meta: dict):
    """Serialize profiler results to a JSON file.

    JSON layout::

      {
        "meta": {op_tag, rank, max_tokens, hidden_dim, k, world_size, ...},
        "kernel_stats": [ {name, calls, cuda_time_avg_us, cpu_time_avg_us}, ... ]
      }
    """
    rows = []
    for evt in prof.key_averages():
        rows.append(
            {
                "name": evt.key,
                "calls": evt.count,
                "cuda_time_avg_us": round(evt.device_time, 2),
                "cuda_time_total_us": round(evt.device_time * evt.count, 2),
                "cpu_time_avg_us": round(evt.cpu_time, 2),
                "cpu_time_total_us": round(evt.cpu_time * evt.count, 2),
            }
        )
    rows.sort(key=lambda r: r["cuda_time_total_us"], reverse=True)

    payload = {
        "meta": {**meta, "op": op_tag, "rank": rank},
        "kernel_stats": rows,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    trace_path = out_path.replace(".json", "_trace.json")
    prof.export_chrome_trace(trace_path)


def _allreduce_stats(
    prof,
    op_tag: str,
    rank: int,
    world_size: int,
    dev: torch.device,
    dtype_key: str = "bf16",
    quant_type: str = "none",
    zero_copy: bool = False,
) -> dict:
    """Pull per-rank profiler metrics, all-reduce them across ranks, and
    return an avg/min/max dict.

    Six metrics, packed into a float64 tensor in this fixed order for
    the all-reduce:
      0: dispatch GPU kernel time (us/call)
      1: combine  GPU kernel time (us/call)
      2: dispatch record_function CUDA time (us/call)
      3: combine  record_function CUDA time (us/call)
      4: dispatch record_function CPU  time (us/call)
      5: combine  record_function CPU  time (us/call)
    """
    msuf = MORI_KERNEL_SUFFIX.get(dtype_key, "bf16")
    _cast_suf = "_fp8cast" if (quant_type == "fp8_direct_cast" and not zero_copy) else ""
    _zc_suf = "_p2p" if zero_copy else "_nop2p"
    if op_tag == "flydsl":
        d_kernel = "ep_dispatch_intranode_0"
        c_kernel = "ep_combine_intranode_0"
    else:
        d_kernel = f"EpDispatchIntraNodeKernel_{msuf}"
        c_kernel = f"EpCombineIntraNodeKernel_{msuf}{_zc_suf}{_cast_suf}"
    d_label = f"{op_tag}::dispatch"
    c_label = f"{op_tag}::combine"

    ev = {e.key: e for e in prof.key_averages()}

    def gpu_us(key):
        e = ev.get(key)
        return e.device_time if (e and e.count) else 0.0

    def cpu_us(key):
        e = ev.get(key)
        return e.cpu_time if (e and e.count) else 0.0

    local = torch.tensor(
        [
            gpu_us(d_kernel),
            gpu_us(c_kernel),
            gpu_us(d_label),
            gpu_us(c_label),
            cpu_us(d_label),
            cpu_us(c_label),
        ],
        dtype=torch.float64,
        device=dev,
    )

    s = local.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = [
        "dispatch_gpu",
        "combine_gpu",
        "dispatch_cuda_e2e",
        "combine_cuda_e2e",
        "dispatch_cpu_e2e",
        "combine_cpu_e2e",
    ]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()} for i, k in enumerate(keys)}


def _algo_bw_GBs(total_recv: int, token_bytes_per_tok: int, duration_us: float) -> float:
    """Per-rank single-direction bandwidth in decimal GB/s.

    Mirrors mori ``bench_dispatch_combine.py`` L403-404:

        bytes  = total_recv * token_bytes_per_tok
        bw_GBs = bytes / 1000**3 / (duration_us / 1e6)

    ``token_bytes_per_tok`` is ``cfg.token_bytes`` which equals
    ``hidden_dim * element_size`` for non-fp4 dtypes and
    ``hidden_dim // 2`` for fp4 (one packed byte per pair of fp4
    lanes), so the formula matches mori's ``hidden_dim * element_size``
    product for bf16/f32 and remains the natural per-token byte count
    for packed fp4.

    Returns 0.0 for degenerate inputs (no recv tokens / zero duration)
    so the column stays printable.
    """
    if duration_us <= 0 or total_recv <= 0 or token_bytes_per_tok <= 0:
        return 0.0
    return total_recv * token_bytes_per_tok / (1000.0**3) / (duration_us / 1e6)


def _print_aggregated(stats: dict, op_tag: str, world_size: int, meta: dict):
    """Print the cross-rank aggregated stats on rank 0."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(
        f"  {op_tag.upper()}  EP={world_size}  bs={meta['max_tokens']}  "
        f"h={meta['hidden_dim']}  k={meta['k']}  ({meta['iters']} iters)"
    )
    print(f"  avg / min / max across all {world_size} ranks (us/call)")
    print(sep)
    hdr = f"  {'metric':<36}  {'avg':>8}  {'min':>8}  {'max':>8}  {'bw GB/s':>9}"
    print(hdr)
    print(f"  {'-'*70}")

    _tr = int(meta.get("total_recv", 0))
    _tb = int(meta.get("token_bytes_per_tok", 0))
    # CPU-time rows leave the bw column blank: host timing doesn't
    # measure GPU-side data transfer.
    rows = [
        ("[Device] dispatch kernel GPU time", "dispatch_gpu", True),
        ("[Device] combine  kernel GPU time", "combine_gpu", True),
        ("[E2E]    dispatch CUDA time (w/sync)", "dispatch_cuda_e2e", True),
        ("[E2E]    combine  CUDA time (w/sync)", "combine_cuda_e2e", True),
        ("[Host]   dispatch CPU  time", "dispatch_cpu_e2e", False),
        ("[Host]   combine  CPU  time", "combine_cpu_e2e", False),
    ]
    for label, key, show_bw in rows:
        v = stats[key]
        if show_bw:
            bw = _algo_bw_GBs(_tr, _tb, v["avg"])
            bw_str = f"  {bw:>9.1f}"
        else:
            bw_str = f"  {'-':>9}"
        print(f"  {label:<36}  {v['avg']:>8.1f}  {v['min']:>8.1f}  {v['max']:>8.1f}{bw_str}")
    print()


def _allreduce_cudagraph_stats_from_key_averages(
    prof,
    op_tag: str,
    rank: int,
    world_size: int,
    dev: torch.device,
    dtype_key: str = "bf16",
    combine_dtype_key: str | None = None,
    quant_type: str = "none",
    zero_copy: bool = False,
) -> dict:
    """Pull metrics from ``prof.key_averages()`` (active phase only) and
    all-reduce them across ranks.

    Four metrics:
      0: dispatch kernel GPU time
      1: combine  kernel GPU time
      2: cudagraph_replay CUDA E2E time
      3: cudagraph_replay CPU  E2E time

    Mori-parity mixed-dtype: ``dtype_key`` -> dispatch kernel suffix,
    ``combine_dtype_key`` -> combine kernel suffix (default = dtype_key).
    """
    if combine_dtype_key is None:
        combine_dtype_key = dtype_key
    msuf_d = MORI_KERNEL_SUFFIX.get(dtype_key, "bf16")
    msuf_c = MORI_KERNEL_SUFFIX.get(combine_dtype_key, "bf16")
    _cast_suf = "_fp8cast" if (quant_type == "fp8_direct_cast" and not zero_copy) else ""
    _zc_suf = "_p2p" if zero_copy else "_nop2p"
    if op_tag == "flydsl":
        d_kernel = "ep_dispatch_intranode_0"
        c_kernel = "ep_combine_intranode_0"
    else:
        d_kernel = f"EpDispatchIntraNodeKernel_{msuf_d}"
        c_kernel = f"EpCombineIntraNodeKernel_{msuf_c}{_zc_suf}{_cast_suf}"
    cg_label = f"{op_tag}::cudagraph_replay"

    ev = {e.key: e for e in prof.key_averages()}

    def gpu_us(key):
        e = ev.get(key)
        return e.device_time if (e and e.count) else 0.0

    def cpu_us(key):
        e = ev.get(key)
        return e.cpu_time if (e and e.count) else 0.0

    local = torch.tensor(
        [
            gpu_us(d_kernel),
            gpu_us(c_kernel),
            gpu_us(cg_label),
            cpu_us(cg_label),
        ],
        dtype=torch.float64,
        device=dev,
    )

    s = local.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = ["dispatch_gpu", "combine_gpu", "replay_cuda_e2e", "replay_cpu_e2e"]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()} for i, k in enumerate(keys)}


def _cudagraph_stats_from_trace(
    trace_path: str,
    op_tag: str,
    rank: int,
    world_size: int,
    dev: torch.device,
    active_iters: int,
    skip_first: int = 5,
    dtype_key: str = "bf16",
    combine_dtype_key: str | None = None,
    quant_type: str = "none",
    zero_copy: bool = False,
) -> dict:
    """Compute kernel stats by parsing the chrome trace JSON, dropping
    the first ``skip_first`` active iterations.

    Pipeline: parse trace -> sort by ts and keep the last
    ``active_iters`` events -> drop the first ``skip_first`` ->
    all-reduce across ranks.

    Mori-parity mixed-dtype: ``dtype_key`` names the **dispatch** kernel
    suffix while ``combine_dtype_key`` names the **combine** kernel
    suffix.  When ``combine_dtype_key`` is ``None`` we fall back to
    ``dtype_key`` (same-dtype dispatch + combine).
    """
    with open(trace_path) as f:
        tr = json.load(f)

    if combine_dtype_key is None:
        combine_dtype_key = dtype_key
    msuf_d = MORI_KERNEL_SUFFIX.get(dtype_key, "bf16")
    msuf_c = MORI_KERNEL_SUFFIX.get(combine_dtype_key, "bf16")
    _cast_suf = "_fp8cast" if (quant_type == "fp8_direct_cast" and not zero_copy) else ""
    _zc_suf = "_p2p" if zero_copy else "_nop2p"
    if op_tag == "flydsl":
        d_name, c_name = "ep_dispatch_intranode_0", "ep_combine_intranode_0"
    else:
        d_name = f"EpDispatchIntraNodeKernel_{msuf_d}"
        c_name = f"EpCombineIntraNodeKernel_{msuf_c}{_zc_suf}{_cast_suf}"
    cg_name = f"{op_tag}::cudagraph_replay"

    kernel_events = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
    d_all = sorted([e for e in kernel_events if d_name in e.get("name", "")], key=lambda e: e["ts"])
    c_all = sorted([e for e in kernel_events if c_name in e.get("name", "")], key=lambda e: e["ts"])
    cg_all = sorted(
        [e for e in tr["traceEvents"] if e.get("cat") == "gpu_user_annotation" and cg_name in e.get("name", "")],
        key=lambda e: e["ts"],
    )

    d_active = [e["dur"] for e in d_all[-active_iters:]]
    c_active = [e["dur"] for e in c_all[-active_iters:]]
    cg_active = [e["dur"] for e in cg_all[-active_iters:]]

    d_valid = d_active[skip_first:]
    c_valid = c_active[skip_first:]
    cg_valid = cg_active[skip_first:]

    valid_n = len(d_valid)
    if rank == 0:
        print(
            f"[trace-stats] {op_tag}: trace has dispatch={len(d_all)} combine={len(c_all)} events; "
            f"keeping last {active_iters} active, skipping first {skip_first}, {valid_n} valid"
        )

    d_avg = sum(d_valid) / valid_n if valid_n else 0.0
    c_avg = sum(c_valid) / valid_n if valid_n else 0.0
    cg_avg = sum(cg_valid) / len(cg_valid) if cg_valid else 0.0

    local = torch.tensor([d_avg, c_avg, cg_avg, 0.0], dtype=torch.float64, device=dev)
    s = local.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg = s / world_size

    keys = ["dispatch_gpu", "combine_gpu", "replay_cuda_e2e", "replay_cpu_e2e"]
    return {k: {"avg": avg[i].item(), "min": mn[i].item(), "max": mx[i].item()} for i, k in enumerate(keys)}


def _print_cudagraph_aggregated(stats: dict, op_tag: str, world_size: int, meta: dict, active_iters: int = None):
    """Print the cudagraph+profiler aggregated stats on rank 0."""
    n = active_iters if active_iters is not None else meta["iters"]
    sep = "=" * 72
    print(f"\n{sep}")
    print(
        f"  {op_tag.upper()} [CUDAGraph+Profiler]  EP={world_size}  bs={meta['max_tokens']}  "
        f"h={meta['hidden_dim']}  k={meta['k']}  ({n} iters)"
    )
    print(f"  avg / min / max across all {world_size} ranks (us/call)")
    print(sep)
    hdr = f"  {'metric':<36}  {'avg':>8}  {'min':>8}  {'max':>8}  {'bw GB/s':>9}"
    print(hdr)
    print(f"  {'-'*70}")

    _tr = int(meta.get("total_recv", 0))
    _tb = int(meta.get("token_bytes_per_tok", 0))
    # dispatch/combine rows use single-phase bytes; replay covers both
    # so it gets 2x bytes; CPU row gets no bw.
    rows = [
        ("[Device] dispatch kernel GPU time", "dispatch_gpu", _tb),
        ("[Device] combine  kernel GPU time", "combine_gpu", _tb),
        ("[E2E]   replay CUDA time (w/sync)", "replay_cuda_e2e", _tb * 2),
        ("[Host]  replay CPU  time", "replay_cpu_e2e", 0),
    ]
    for label, key, bytes_per_tok in rows:
        v = stats[key]
        if bytes_per_tok > 0:
            bw = _algo_bw_GBs(_tr, bytes_per_tok, v["avg"])
            bw_str = f"  {bw:>9.1f}"
        else:
            bw_str = f"  {'-':>9}"
        print(f"  {label:<36}  {v['avg']:>8.1f}  {v['min']:>8.1f}  {v['max']:>8.1f}{bw_str}")
    print()


def _make_profiler(active_iters: int = None, prof_warmup: int = 10):
    """Build a torch.profiler.

    The schedule keeps the first (1 + prof_warmup) steps in wait/warmup
    so ROCTracer doesn't accumulate state under heavy multi-GPU P2P
    shmem traffic.
    """
    kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )
    if active_iters is not None and active_iters > 0:
        kwargs["schedule"] = torch.profiler.schedule(
            wait=1,
            warmup=prof_warmup,
            active=active_iters,
            repeat=1,
        )
    return profile(**kwargs)


# --- bench mode: profiler-free CUDA-event timing ---
def bench_op(
    op,
    op_tag: str,
    inp,
    wts,
    idx,
    wc_buf,
    k,
    rank: int,
    world_size: int,
    dev: torch.device,
    warmup: int,
    iters: int,
    meta: dict,
    scales=None,
    packed_recv_x=None,
    combine_dtype=None,
):
    """Profiler-free CUDA-event timing of dispatch/combine; reports GPU
    time avg/min/max.  ``combine_dtype`` (mori parity) defaults to the
    dispatch input dtype; pass a different dtype to exercise mixed-dtype
    dispatch + combine (e.g. fp4 dispatch + bf16 combine)."""
    _dkw = dict(packed_recv_x=packed_recv_x) if packed_recv_x is not None else {}
    _ckw = dict(packed_recv_x=packed_recv_x) if packed_recv_x is not None else {}
    if combine_dtype is None:
        combine_dtype = inp.dtype
    ms.shmem_barrier_all()
    if rank == 0:
        print(f"\n[bench] {op_tag} warmup {warmup} iters...")
    for _ in range(warmup):
        _safe_op_reset(op)
        ret = op.dispatch(inp, wts, scales, idx, **_dkw)
        _run_combine(op, ret, combine_dtype, **_ckw)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"[bench] {op_tag} timing {iters} iters...")

    d_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    c_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]

    for i in range(iters):
        # op.reset()
        dist.barrier()

        d_events[i][0].record()
        ret = op.dispatch(inp, wts, scales, idx, **_dkw)
        d_events[i][1].record()

        dist.barrier()

        c_events[i][0].record()
        _run_combine(op, ret, combine_dtype, **_ckw)
        c_events[i][1].record()

    torch.cuda.synchronize()
    d_list = [d_events[i][0].elapsed_time(d_events[i][1]) * 1000 for i in range(iters)]
    c_list = [c_events[i][0].elapsed_time(c_events[i][1]) * 1000 for i in range(iters)]

    # Aggregate avg / min / max across ranks.
    local = torch.tensor(
        [
            sum(d_list) / len(d_list),
            min(d_list),
            max(d_list),
            sum(c_list) / len(c_list),
            min(c_list),
            max(c_list),
        ],
        dtype=torch.float64,
        device=dev,
    )
    s = local.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg_d = (s[0] / world_size).item()
    mn_d = mn[0].item()
    mx_d = mx[2].item()
    avg_c = (s[3] / world_size).item()
    mn_c = mn[3].item()
    mx_c = mx[5].item()

    # Bandwidth (mori IntraNode formula) — uses per-rank total_recv
    # captured once in run_profiler; avg_d/avg_c are the cross-rank
    # algorithm-bandwidth latencies, so the column shows the algo bw.
    _tr = int(meta.get("total_recv", 0))
    _tb = int(meta.get("token_bytes_per_tok", 0))
    bw_d = _algo_bw_GBs(_tr, _tb, avg_d)
    bw_c = _algo_bw_GBs(_tr, _tb, avg_c)

    if rank == 0:
        sep = "=" * 78
        tag = (
            f"{op_tag.upper()}  EP={meta['world_size']}  bs={meta['max_tokens']}  "
            f"h={meta['hidden_dim']}  k={meta['k']}  ({iters} iters)"
        )
        print(f"\n{sep}\n  {tag}\n  avg / min / max across all {world_size} ranks (us/call)\n{sep}")
        print(f"  {'metric':<36}  {'avg':>8}  {'min':>8}  {'max':>8}  {'bw GB/s':>9}")
        print(f"  {'-'*68}")
        print(f"  {'[E2E]  dispatch CUDA time':<36}  {avg_d:>8.1f}  {mn_d:>8.1f}  {mx_d:>8.1f}  {bw_d:>9.1f}")
        print(f"  {'[E2E]  combine  CUDA time':<36}  {avg_c:>8.1f}  {mn_c:>8.1f}  {mx_c:>8.1f}  {bw_c:>9.1f}")
        print()


# --- cudagraph mode: CUDA Graph capture + replay timing ---
def _cudagraph_capture_flydsl(
    op, inp, wts, idx, wc_buf, capture_stream, scales=None, packed_recv_x=None, combine_dtype=None
):
    """Capture FlyDSL dispatch+combine into a CUDA Graph.

    Both dispatch and combine return full-sized tensors (no ``.item()``,
    no dynamic slicing).  We must first run them eagerly once to trigger
    the ``flyc.compile()`` JIT (which uses the default stream and can't
    run during capture); the capture then records only the already-
    compiled kernel launches.

    ``combine_dtype`` (mori parity) defaults to ``inp.dtype``; pass a
    different dtype for mixed-dtype dispatch + combine sweeps.
    """
    _dkw = dict(packed_recv_x=packed_recv_x) if packed_recv_x is not None else {}
    _ckw = dict(packed_recv_x=packed_recv_x) if packed_recv_x is not None else {}
    if combine_dtype is None:
        combine_dtype = inp.dtype
    ms.shmem_barrier_all()
    ret = op.dispatch(inp, wts, scales, idx, **_dkw)
    _run_combine(op, ret, combine_dtype, **_ckw)

    ms.shmem_barrier_all()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        ret = op.dispatch(inp, wts, scales, idx, **_dkw)
        _run_combine(op, ret, combine_dtype, **_ckw)
    return g, capture_stream


def _cudagraph_capture_mori(
    op, inp, wts, idx, wc_buf, capture_stream, scales=None, packed_recv_x=None, combine_dtype=None
):
    """Capture mori dispatch+combine into a CUDA Graph.

    Mori's dispatch returns a real tensor under capture and the combine
    kernel reads ``totalRecvTokenNum`` from HBM, so no pre-capture eager
    call is needed.  Pattern follows mori's ``stress_graph`` in
    ``mori/tests/python/ops/bench_dispatch_combine.py``.
    """
    if combine_dtype is None:
        combine_dtype = inp.dtype
    ms.shmem_barrier_all()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        ret = op.dispatch(inp, wts, None, idx)
        _run_combine(op, ret, combine_dtype)
    return g, capture_stream


def cudagraph_op(
    op,
    op_tag: str,
    inp,
    wts,
    idx,
    wc_buf,
    k,
    rank: int,
    world_size: int,
    dev: torch.device,
    warmup: int,
    iters: int,
    meta: dict,
    scales=None,
    packed_recv_x=None,
    combine_dtype=None,
):
    """CUDA Graph mode: capture dispatch+combine, then time replays."""
    if combine_dtype is None:
        combine_dtype = inp.dtype
    capture_stream = torch.cuda.Stream()
    if op_tag == "flydsl":
        g, cs = _cudagraph_capture_flydsl(
            op,
            inp,
            wts,
            idx,
            wc_buf,
            capture_stream,
            scales=scales,
            packed_recv_x=packed_recv_x,
            combine_dtype=combine_dtype,
        )
    else:
        g, cs = _cudagraph_capture_mori(
            op,
            inp,
            wts,
            idx,
            wc_buf,
            capture_stream,
            scales=scales,
            packed_recv_x=packed_recv_x,
            combine_dtype=combine_dtype,
        )

    if rank == 0:
        print(f"\n[cudagraph] {op_tag} capture done")

    # Replay warmup (HIP graph cold start + GPU cache warmup).
    replay_warmup = 10
    if rank == 0:
        print(f"[cudagraph] replay warmup {replay_warmup} + timing {iters} iters (no-reset)...")
    for _ in range(replay_warmup):
        g.replay()
    torch.cuda.synchronize()

    # Timing: pre-allocate event pairs, sync once after the loop.
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]

    for i in range(iters):
        events[i][0].record()
        g.replay()
        events[i][1].record()

    torch.cuda.synchronize()
    gpu_times = [events[i][0].elapsed_time(events[i][1]) * 1000 for i in range(iters)]

    # Per-replay diagnostics.
    per_replay_t = torch.tensor(gpu_times, dtype=torch.float64, device=dev)
    all_per_replay = [torch.zeros_like(per_replay_t) for _ in range(world_size)]
    dist.all_gather(all_per_replay, per_replay_t)

    local = torch.tensor(
        [
            sum(gpu_times) / len(gpu_times),
            min(gpu_times),
            max(gpu_times),
        ],
        dtype=torch.float64,
        device=dev,
    )
    s = local.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = local.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = local.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    avg_g = (s[0] / world_size).item()
    mn_g = mn[0].item()
    mx_g = mx[2].item()

    # Combined dispatch+combine bytes: count one set per phase, divide
    # by the single replay duration that already covers both kernels.
    _tr = int(meta.get("total_recv", 0))
    _tb = int(meta.get("token_bytes_per_tok", 0))
    bw_g = _algo_bw_GBs(_tr, _tb * 2, avg_g)

    if rank == 0:
        sep = "=" * 78
        tag = (
            f"{op_tag.upper()} [CUDAGraph]  EP={meta['world_size']}  "
            f"bs={meta['max_tokens']}  h={meta['hidden_dim']}  k={meta['k']}  "
            f"({iters} replays)"
        )
        print(f"\n{sep}\n  {tag}\n  avg / min / max across all {world_size} ranks (us/call)\n{sep}")
        print(f"  {'metric':<36}  {'avg':>8}  {'min':>8}  {'max':>8}  {'bw GB/s':>9}")
        print(f"  {'-'*68}")
        print(f"  {'[GPU]  dispatch+combine (event)':<36}  {avg_g:>8.1f}  {mn_g:>8.1f}  {mx_g:>8.1f}  {bw_g:>9.1f}")

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


# --- Per-op profiler capture ---
def profile_op(
    op,
    op_tag: str,
    inp,
    wts,
    idx,
    wc_buf,
    k,
    rank: int,
    world_size: int,
    dev: torch.device,
    iters: int,
    out_dir: str,
    meta: dict,
    scales=None,
    packed_recv_x=None,
    dtype_key: str = "bf16",
    quant_type: str = "none",
    zero_copy: bool = False,
    combine_dtype=None,
):
    """Profile a single op (FlyDSL or mori) standalone; save the JSON and
    print cross-rank aggregated stats.

    Uses ``schedule(wait=1, warmup=10, active=iters)`` so ROCTracer
    skips / light-traces the first 11 steps and avoids races with
    multi-GPU P2P shmem.
    """
    ms.shmem_barrier_all()
    prof_warmup = 10
    total_steps = iters + 1 + prof_warmup  # wait=1 + warmup=prof_warmup + active=iters
    if rank == 0:
        print(f"\n[profiler] {op_tag} capturing ({iters} active + {1 + prof_warmup} ramp-up)...")

    _dkw = dict(packed_recv_x=packed_recv_x) if packed_recv_x is not None else {}
    _ckw = dict(packed_recv_x=packed_recv_x) if packed_recv_x is not None else {}
    if combine_dtype is None:
        combine_dtype = inp.dtype
    with _make_profiler(active_iters=iters, prof_warmup=prof_warmup) as prof:
        for step in range(total_steps):
            # with record_function(f"{op_tag}::reset"):
            #     op.reset()
            dist.barrier()

            with record_function(f"{op_tag}::dispatch"):
                ret = op.dispatch(inp, wts, scales, idx, **_dkw)

            dist.barrier()

            with record_function(f"{op_tag}::combine"):
                _run_combine(op, ret, combine_dtype, **_ckw)

            # dist.barrier()

            prof.step()

    # Save JSON: one file per rank, named by op_tag and rank.
    out_path = os.path.join(out_dir, f"{op_tag}_rank{rank}.json")
    _save_profile_json(prof, out_path, rank, op_tag, meta)
    if rank == 0:
        print(f"[profiler] {op_tag} trace -> {out_path}")

    # Cross-rank aggregation via all_reduce; rank 0 prints.
    agg_stats = _allreduce_stats(
        prof, op_tag, rank, world_size, dev, dtype_key=dtype_key, quant_type=quant_type, zero_copy=zero_copy
    )
    if rank == 0:
        _print_aggregated(agg_stats, op_tag, world_size, meta)
    return prof


# --- profile + cudagraph mode ---
def profile_cudagraph_op(
    op,
    op_tag: str,
    inp,
    wts,
    idx,
    wc_buf,
    k,
    rank: int,
    world_size: int,
    dev: torch.device,
    warmup: int,
    iters: int,
    out_dir: str,
    meta: dict,
    scales=None,
    packed_recv_x=None,
    dtype_key: str = "bf16",
    quant_type: str = "none",
    zero_copy: bool = False,
    combine_dtype=None,
):
    """Profile CUDAGraph replays with torch.profiler; save JSON and
    print cross-rank aggregated stats.

    Pipeline: eager warmup -> graph capture -> replay warmup ->
    profiled replay loop.
    """
    if combine_dtype is None:
        combine_dtype = inp.dtype
    ms.shmem_barrier_all()

    capture_stream = torch.cuda.Stream()
    if op_tag == "flydsl":
        g, cs = _cudagraph_capture_flydsl(
            op,
            inp,
            wts,
            idx,
            wc_buf,
            capture_stream,
            scales=scales,
            packed_recv_x=packed_recv_x,
            combine_dtype=combine_dtype,
        )
    else:
        g, cs = _cudagraph_capture_mori(
            op,
            inp,
            wts,
            idx,
            wc_buf,
            capture_stream,
            scales=scales,
            packed_recv_x=packed_recv_x,
            combine_dtype=combine_dtype,
        )

    if rank == 0:
        print(f"\n[profile+cudagraph] {op_tag} capture done")

    # Replay warmup (HIP graph cold start + GPU cache warmup).
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
        print(
            f"[profile+cudagraph] {op_tag} scheduled profiler: "
            f"warmup={prof_warmup}, active={active_iters}, "
            f"dropping first {skip_first}, {valid_iters} valid (no-reset)..."
        )

    with _make_profiler(active_iters=active_iters, prof_warmup=prof_warmup) as prof:
        for step in range(total_steps):
            with record_function(f"{op_tag}::cudagraph_replay"):
                g.replay()
            prof.step()

    out_path = os.path.join(out_dir, f"{op_tag}_cudagraph_rank{rank}.json")
    _save_profile_json(prof, out_path, rank, op_tag, meta)
    trace_path = out_path.replace(".json", "_trace.json")
    if rank == 0:
        print(f"[profile+cudagraph] {op_tag} trace -> {trace_path}")

    _combine_dtype_key = None
    if combine_dtype is not None and combine_dtype != inp.dtype:
        for _k, _v in DTYPE_MAP.items():
            if _v == combine_dtype:
                _combine_dtype_key = _k
                break
    agg_stats = _cudagraph_stats_from_trace(
        trace_path,
        op_tag,
        rank,
        world_size,
        dev,
        active_iters=active_iters,
        skip_first=skip_first,
        dtype_key=dtype_key,
        combine_dtype_key=_combine_dtype_key,
        quant_type=quant_type,
        zero_copy=zero_copy,
    )
    if rank == 0:
        _print_cudagraph_aggregated(agg_stats, op_tag, world_size, meta, active_iters=valid_iters)
    return prof


# --- verify mode: correctness check ---
VERIFY_TOL = {
    "f32": {"atol": 1e-5, "rtol": 1e-4},
    "bf16": {"atol": 1e-2, "rtol": 1e-2},
    "float8_e4m3fn": {"atol": 1e-1, "rtol": 5e-2},
    "float8_e4m3fnuz": {"atol": 1e-1, "rtol": 5e-2},
    "fp4": {"atol": 5e-1, "rtol": 1e-1},
}


def _check_close(name, a, b, atol, rtol, rank, cast_to=None):
    """Compare two tensors and print PASS/FAIL."""
    if cast_to is not None:
        a, b = a.to(cast_to), b.to(cast_to)
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    max_diff = (a.float() - b.float()).abs().max().item()
    status = "PASS" if ok else "FAIL"
    if rank == 0:
        print(f"  [{status}] {name:40s}  max_diff={max_diff:.6g}  atol={atol} rtol={rtol}")
    return ok


def _check_exact(name, a, b, rank):
    """Compare two tensors for exact equality."""
    ok = torch.equal(a, b)
    if not ok:
        diff_count = (a != b).sum().item()
        status = "FAIL"
    else:
        diff_count = 0
        status = "PASS"
    if rank == 0:
        print(f"  [{status}] {name:40s}  diff_elements={diff_count}")
    return ok


def _global_reduce_all_pass(all_pass: bool, rank: int) -> bool:
    """Cross-rank AND reduction so a single failing rank fails the whole job.

    Without this every rank only sees its own ``all_pass`` and CI would
    falsely pass when e.g. rank0 succeeds but rank3 fails (problem 2).
    """
    if not dist.is_available() or not dist.is_initialized():
        return all_pass
    t = torch.tensor([1 if all_pass else 0], dtype=torch.int32, device=torch.device("cuda", rank))
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _decode_tok_id_to_src(tis, total_recv, max_tok_per_rank):
    """Decode ``tok_id_to_src[:total_recv]`` -> (src_pe, src_lid) tensors.

    The kernel encodes each recv slot as ``src_pe * max_tok_per_rank + src_lid``
    (see dispatch_combine_intranode_kernel.py).  Only the first
    ``total_recv`` entries are valid; tail entries carry leftover bytes.
    """
    enc = tis[:total_recv].to(torch.int64)
    src_pe = enc // max_tok_per_rank
    src_lid = enc % max_tok_per_rank
    return src_pe, src_lid


def _allgather_rows(local_t, world_size):
    """All-gather ``local_t`` (shape [N, ...]) along a new leading PE axis.

    Returns ``[world_size, N, ...]`` so callers can index ``[src_pe, src_lid]``
    to recover the original sender-side row for every recv slot.

    NCCL/RCCL does not support some packed/sub-byte dtypes (notably
    ``Float4_e2m1fn_x2``).  We transparently view the tensor as ``uint8``
    bytes for transport and view it back to the original dtype on the
    gathered side so that callers (and downstream ``view(uint8)`` byte
    compares) get a tensor that round-trips losslessly.
    """
    if not (dist.is_available() and dist.is_initialized() and world_size > 1):
        return local_t.unsqueeze(0)
    src = local_t.contiguous()
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    needs_view = fp4 is not None and src.dtype == fp4
    if needs_view:
        orig_dtype = src.dtype
        src = src.view(torch.uint8)
    gather = [torch.empty_like(src) for _ in range(world_size)]
    dist.all_gather(gather, src)
    if needs_view:
        gather = [g.view(orig_dtype) for g in gather]
    return torch.stack(gather, dim=0)


def _verify_dispatch_self_consistency(ret_f, op_fly, inp, wts, idx, scales, cfg, world_size, rank):
    """Byte-level semantic check of dispatch outputs (mori parity, no mori dep).

    Recv-row ordering is governed by an atomic race, so a cross-impl
    raw-tensor compare is fragile.  We instead verify the *invariant*
    that every recv row truly originates from the sender row that the
    kernel claims via ``shmem_tok_id_to_src``:

      out_tok[i]    == sender_input  [src_pe, src_lid]
      out_idx[i]    == sender_indices[src_pe, src_lid]
      out_wts[i]    == sender_weights[src_pe, src_lid]
      out_scales[i] == sender_scales [src_pe, src_lid]   (when configured)

    This is the FlyDSL-internal equivalent of mori's dispatch byte
    verify in bench_dispatch_combine.py and covers all four output
    fields end-to-end without referencing any mori implementation.
    """
    all_pass = True
    total_recv = int(ret_f[4].item())
    mt = cfg.max_num_inp_token_per_rank

    # IMPORTANT: do NOT early-return on ``total_recv == 0``.  The
    # ``_allgather_rows`` calls below are collective operations and must
    # be reached by every rank in lock-step; an early return on a single
    # rank desynchronizes the NCCL group and the surviving ranks' next
    # ``dist.all_gather`` blocks until the watchdog timeout (30 min) --
    # this is exactly the symptom L2's ``L2_bf16_bs1_random`` case
    # surfaced (rank 0 had ``total_recv = 0`` while peers had
    # ``total_recv > 0``).  We still skip the byte-compare body when
    # ``total_recv == 0`` because there's nothing to slice, but the
    # all_gather of the per-rank input rows happens unconditionally so
    # peer ranks can still resolve their ``g_inp[src_pe, src_lid]``
    # references back to this rank's input row.
    skip_compare = total_recv == 0
    if rank == 0:
        if skip_compare:
            print(
                "\n  [SKIP] dispatch byte verify: total_recv == 0 on rank 0 "
                "(peer ranks still gather this rank's input rows)"
            )
        else:
            print("\n  -- Dispatch byte verify (rows resolved via tok_id_to_src) --")

    # fp4 has no element-wise ``index_cuda`` kernel; we must view-as-uint8
    # before all-gather + index.  fp8 supports indexing but we still byte
    # compare below.
    is_fp4 = cfg.data_type == torch.float4_e2m1fn_x2
    inp_for_gather = inp.view(torch.uint8) if is_fp4 else inp

    g_inp = _allgather_rows(inp_for_gather, world_size)
    g_wts = _allgather_rows(wts, world_size)
    g_idx = _allgather_rows(idx.to(torch.int32), world_size)
    g_sc = _allgather_rows(scales, world_size) if scales is not None else None

    if skip_compare:
        return all_pass

    src_pe, src_lid = _decode_tok_id_to_src(op_fly.shmem_tok_id_to_src, total_recv, mt)

    expected_tok = g_inp[src_pe, src_lid]  # uint8-shape when fp4
    expected_idx = g_idx[src_pe, src_lid]
    expected_wts = g_wts[src_pe, src_lid]

    f_tok = ret_f[0][:total_recv]
    if is_fp4:
        all_pass &= _check_exact(
            "dispatch out_tok == g_inp (fp4 bytes)",
            f_tok.view(torch.uint8),
            expected_tok,
            rank,
        )
    elif cfg.data_type in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        all_pass &= _check_exact(
            "dispatch out_tok == g_inp (fp8 bytes)",
            f_tok.view(torch.uint8),
            expected_tok.view(torch.uint8),
            rank,
        )
    else:
        all_pass &= _check_exact("dispatch out_tok == g_inp", f_tok, expected_tok, rank)

    f_idx = ret_f[3][:total_recv]
    all_pass &= _check_exact("dispatch out_idx == g_idx", f_idx, expected_idx, rank)

    f_wts = ret_f[1][:total_recv]
    all_pass &= _check_exact("dispatch out_wts == g_wts", f_wts, expected_wts, rank)

    if g_sc is not None and ret_f[2] is not None and cfg.scale_bytes > 0:
        expected_sc = g_sc[src_pe, src_lid]
        f_sc = ret_f[2][:total_recv].contiguous().view(torch.uint8)
        e_sc = expected_sc.contiguous().view(torch.uint8)
        all_pass &= _check_exact("dispatch out_scales == g_scales (bytes)", f_sc, e_sc, rank)

    return all_pass


def verify_self(op_fly, inp, wts, idx, k, rank, world_size, dev, dtype_key, cfg, op_mori=None, combine_dtype=None):
    """FlyDSL verify, mori-byte parity when available.

    Four invariants are checked, mirroring mori's bench_dispatch_combine.py:

      1. dispatch byte verify       : out_tok / out_wts / out_idx / out_scales
         each match the all-gathered sender row resolved via
         ``shmem_tok_id_to_src``.
      2. recv-slot dedup            : ``unique(src_token_pos).numel() == total_recv``.
      3. combine token round-trip   : when ``op_mori`` is provided this is
         a byte-level equality (``fly_tok == mori_tok``) — the strongest
         correctness gate, independent of dispatch / combine algebraic
         semantics.  When ``op_mori`` is ``None`` we fall back to a
         best-effort sanity check (NaN/Inf gate plus a magnitude DIAG
         against ``k * inp``; the DIAG does **not** fail the test because
         the actual combine semantics are weight-folded and the closed-
         form expected requires mori as oracle).
      4. combine weight round-trip  : when ``op_mori`` is provided,
         byte verify against ``mori_out_wts``; otherwise a sanity NaN/Inf
         gate (the previous ``wts * k`` closed-form was a misconception
         and is left as DIAG only).

    Bug-A history: prior to this revision the self-check asserted
    ``fly_tok ≈ k * inp`` / ``out_wts ≈ k * wts`` unconditionally.  Both
    expectations were closed-form approximations that do **not** match
    the actual intra-node dispatch+combine algebra produced by either
    FlyDSL or mori (probed: ``|fly - mori| == 0`` element-wise across
    all 4 ranks, but ``|fly - k*inp|.max ≈ 25``).  Falling back to
    mori-byte verify when mori is available restores the strict
    semantic that the original ``verify_op`` provided before 977df719.
    """
    all_pass = True
    if combine_dtype is None:
        combine_dtype = inp.dtype

    if rank == 0:
        _cd_lbl = "" if combine_dtype == inp.dtype else f"  combine_dtype={combine_dtype}"
        print(f"\n{'='*65}")
        print(
            f"  VERIFY (self-check)  dtype={dtype_key}{_cd_lbl}  "
            f"EP={world_size}  bs={inp.shape[0]}  h={cfg.hidden_dim}  k={k}"
        )
        print(f"{'='*65}")

    ms.shmem_barrier_all()

    packed_recv_x = None
    if cfg.enable_std_moe:
        epr = cfg.num_experts_per_rank
        mr = cfg.max_recv
        _prx_nbytes = epr * mr * cfg.token_bytes
        packed_recv_x = (
            torch.zeros(_prx_nbytes, dtype=torch.uint8, device=dev)
            .view(cfg.data_type)
            .view(epr * mr, cfg.token_view_dim)
        )

    scales = None
    if cfg.scale_dim > 0 and cfg.scale_type_size > 0:
        _sc_bytes = cfg.scale_dim * cfg.scale_type_size
        scales = torch.randn(inp.shape[0], _sc_bytes // 4, dtype=torch.float32, device=dev).contiguous()
        scales = scales.view(torch.uint8).view(inp.shape[0], _sc_bytes)

    ret_f = op_fly.dispatch(inp, wts, scales, idx, packed_recv_x=packed_recv_x)
    torch.cuda.synchronize()
    dist.barrier()

    total_recv = int(ret_f[4].item())
    if rank == 0:
        print(f"\n  total_recv = {total_recv}")

    # === (1) dispatch byte verify (mori-parity, no mori dep) ===
    all_pass &= _verify_dispatch_self_consistency(
        ret_f,
        op_fly,
        inp,
        wts,
        idx,
        scales,
        cfg,
        world_size,
        rank,
    )

    # === (2) recv-slot dedup ===
    # No two recv slots may share the same (src_pe, src_lid); a
    # collision would mean two distinct senders claimed the same
    # token origin, which would corrupt the round-trip combine
    # equations below.
    if total_recv > 0:
        mt_dd = cfg.max_num_inp_token_per_rank
        sp_dd, sl_dd = _decode_tok_id_to_src(op_fly.shmem_tok_id_to_src, total_recv, mt_dd)
        src_pos = sp_dd * mt_dd + sl_dd
        n_unique = int(torch.unique(src_pos).numel())
        ok_uniq = n_unique == total_recv
        if rank == 0:
            status = "PASS" if ok_uniq else "FAIL"
            print(f"  [{status}] recv-slot dedup  unique={n_unique}  total_recv={total_recv}")
        all_pass &= ok_uniq

    cout_f = _run_combine(op_fly, ret_f, combine_dtype, packed_recv_x=packed_recv_x)
    torch.cuda.synchronize()
    dist.barrier()

    mt = cfg.max_num_inp_token_per_rank
    f_tok = cout_f[0][:mt]

    # Mori-byte oracle: when a mori reference op is available, run an
    # identical dispatch+combine on it and use its outputs as the strict
    # byte-level expected for the combine token / weight checks below.
    # This is the same contract the original (pre-977df719) verify_op
    # enforced.
    mori_tok = None
    mori_out_wts = None
    if op_mori is not None and not cfg.enable_std_moe and cfg.quant_type == "none":
        try:
            op_mori.reset()
            ms.shmem_barrier_all()
            mret = op_mori.dispatch(inp, wts, scales, idx)
            torch.cuda.synchronize()
            dist.barrier()
            mout = _run_combine(op_mori, mret, combine_dtype)
            torch.cuda.synchronize()
            dist.barrier()
            mori_tok = mout[0][:mt] if mout[0] is not None else None
            mori_out_wts = mout[1][:mt] if (len(mout) > 1 and mout[1] is not None) else None
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(f"  [INFO] mori oracle disabled (dispatch/combine raised): {e}")
            mori_tok = None
            mori_out_wts = None

    if cfg.enable_std_moe:
        scale_factor = 1
        check_label = "out_tok vs inp (StdMoE weighted)"
    else:
        scale_factor = k
        check_label = "out_tok vs k*inp"

    # Symmetric I/O contract: ``f_tok`` is in ``cfg.data_type`` (matches
    # dispatch input dtype).  mori parity -- caller dtype is symmetric;
    # the only wire-format divergence is ``fp8_direct_cast`` (bf16
    # caller / fp8 wire), which still writes back bf16 to ``f_tok``.

    if rank == 0:
        print(f"\n  ── Self-check: combine output vs {'inp' if scale_factor == 1 else 'k*input'} ──")
        if combine_dtype == torch.float4_e2m1fn_x2:
            if k == 1 and not cfg.enable_std_moe and inp.dtype == combine_dtype:
                ok = torch.equal(f_tok.view(torch.uint8), inp.view(torch.uint8))
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] out_tok vs inp (byte-level, k=1)")
                all_pass &= ok
            else:
                # k>1 / std-MoE: combine accumulates k contributions in
                # f32 and saturates back to fp4, so we cannot do an
                # exact byte compare in PyTorch (no fp4 arithmetic).
                # Run a *liveness* check instead: with N(0,1) inputs
                # essentially all fp4 lanes encode non-zero codes, so
                # a combine that ran to completion must leave the
                # output buffer >>50% non-zero bytes.  This catches
                # the kernel-deadlock / all-zero-output failure modes
                # the bf16 zero-copy bug surfaced earlier on mori, and
                # is the strongest sanity check we can do without
                # actual fp4 PyTorch arithmetic.
                f_u8 = f_tok.view(torch.uint8)
                nz_ratio = (f_u8 != 0).float().mean().item()
                # Lower nibble (0xF0) + upper nibble (0x0F) -- both
                # carry an fp4 lane.  An all-zero combine would put
                # nz_ratio == 0; an all-NaN combine would actually be
                # impossible in fp4 (no NaN encoding), but byte 0xFF
                # is fp4's most negative pair so we also flag that.
                allff_ratio = (f_u8 == 0xFF).float().mean().item()
                ok_nz = nz_ratio > 0.5
                ok_no_sat = allff_ratio < 0.9
                status = "PASS" if (ok_nz and ok_no_sat) else "FAIL"
                print(
                    f"  [{status}] fp4 out_tok liveness  "
                    f"non-zero={nz_ratio:.3f} (>0.5)  "
                    f"all-saturated={allff_ratio:.3f} (<0.9)  "
                    f"(k={k}, std_moe={cfg.enable_std_moe})"
                )
                all_pass &= ok_nz and ok_no_sat
        else:
            cast_to = torch.float32 if combine_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) else None
            # recv-cap shrink path: when FlyDSL is given a non-trivial
            # ``max_total_recv_tokens`` (< worst-case ws*M), mori's
            # equivalent cap is a hard device-side assert -- random
            # routing would abort mori on the very first batch and
            # poison the HIP context for the rest of the run, so
            # ``build_mori_ref`` keeps mori uncapped on those cases
            # (see :func:`build_mori_ref` docstring).  Cross-impl byte
            # parity is therefore undefined here; fall back to a
            # NaN/Inf liveness gate and report the magnitude diff vs
            # the uncapped mori reference as a diagnostic.
            cap_shrinks = cfg.max_total_recv_tokens > 0 and cfg.effective_max_recv < cfg.max_recv
            if cap_shrinks and mori_tok is not None:
                try:
                    diff_to_mori = (f_tok.float() - mori_tok.float()).abs().max().item()
                except Exception:
                    diff_to_mori = float("inf")
                has_nan = torch.isnan(f_tok.float()).any().item()
                has_inf = torch.isinf(f_tok.float()).any().item()
                ok_finite = (not has_nan) and (not has_inf)
                status = "PASS" if ok_finite else "FAIL"
                print(
                    f"  [{status}] cap-shrinks liveness  "
                    f"NaN={has_nan} Inf={has_inf}  "
                    f"|fly - mori (uncapped)|.max={diff_to_mori:.4f}  "
                    f"(cap={cfg.max_total_recv_tokens}, worst={cfg.max_recv}; "
                    "mori not capped under this config -- see "
                    "build_mori_ref docstring)"
                )
                all_pass &= ok_finite
                mori_tok = None  # skip the byte-exact branches below
            if mori_tok is not None:
                # Strongest gate: byte-level equality vs mori output.
                # No tolerance — the two kernels operate on the same
                # symmetric shmem layout and must agree element-wise.
                #
                # Exception: in zero-copy combine mode
                # (``cfg.zero_copy=True``) mori's reference path
                # has historically diverged from FlyDSL: shmem dumps
                # confirmed FlyDSL writes ``k*inp`` into ``shmem_comb_out_tok``
                # while mori's zero-copy kernel folds weights (or used to
                # return zeros).  When the mori byte gate fails *and*
                # ``fly == k*inp`` byte-exactly, we honour the case-level
                # comment in ``bf16_zero_copy`` and treat the
                # ``fly==k*inp`` invariant as the authoritative gate.  In
                # every other (non-zero-copy) configuration mori byte
                # equality is still hard-required.
                # ``inp.float()`` would HSAIL-crash for fp4 inp; route
                # through the project-local fp4 unpacker (see the
                # mori-unavailable branch below).
                inp_for_diag = _cast_dispatch_to_combine(inp, torch.float32)
                try:
                    diff_to_kinp = (f_tok.float() - inp_for_diag * scale_factor).abs().max().item()
                except Exception:
                    diff_to_kinp = float("inf")
                zero_copy_mode = cfg.zero_copy and cfg.quant_type == "none"

                if combine_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                    ok_b = torch.equal(f_tok.view(torch.uint8), mori_tok.view(torch.uint8))
                    accept = ok_b
                    if ok_b:
                        print("  [PASS] out_tok vs mori (fp8 bytes)")
                    else:
                        # Bytes differ.  fp8 has multiple bit patterns that
                        # decode to the same float (e.g. +0 = 0x00 vs
                        # -0 = 0x80 in OCP e4m3) and several NaN encodings.
                        # When ``|fly - mori|.max(f32) == 0`` AND there are
                        # no NaNs on either side, the difference is purely
                        # representational and we treat it as PASS with a
                        # diagnostic note (mori-parity at value level).
                        f_f32 = f_tok.float()
                        m_f32 = mori_tok.float()
                        has_nan = bool(torch.isnan(f_f32).any() or torch.isnan(m_f32).any())
                        d_max = (f_f32 - m_f32).abs().max().item()
                        n_byte_diff = (f_tok.view(torch.uint8) != mori_tok.view(torch.uint8)).sum().item()
                        if (not has_nan) and d_max == 0.0:
                            print(
                                f"  [PASS] out_tok vs mori (fp8 value-equiv; "
                                f"{n_byte_diff} bytes differ by +0/-0 sign encoding)"
                            )
                            accept = True
                        else:
                            print("  [FAIL] out_tok vs mori (fp8 bytes)")
                            print(
                                f"          |fly - mori|.max(f32) = {d_max:.6f}  "
                                f"(NaN={has_nan}, byte_diffs={n_byte_diff})"
                            )
                    all_pass &= accept
                else:
                    ok_mori = _check_close(
                        "out_tok vs mori",
                        f_tok,
                        mori_tok,
                        atol=0.0,
                        rtol=0.0,
                        rank=rank,
                        cast_to=cast_to,
                    )
                    if (not ok_mori) and zero_copy_mode and diff_to_kinp == 0.0:
                        print(
                            "  [PASS] out_tok vs k*inp (zero_copy mode; mori ref historically "
                            "broken on this path, see case comment)"
                        )
                    else:
                        all_pass &= ok_mori
                # Magnitude DIAG against k*inp -- informational only, never
                # gates correctness.  The combine algebra is weight-folded,
                # so ``k * inp`` is **not** the expected value (Bug-A).
                try:
                    print(f"  [DIAG] {check_label}                 |fly - k*inp|.max={diff_to_kinp:.4f}")
                except Exception:
                    pass
            else:
                # Mori unavailable: fall back to sanity-only gate.
                # NaN/Inf is hard failure; the ``k*inp`` magnitude check
                # is purely diagnostic.  ``inp.float()`` is unsafe when
                # ``inp.dtype`` is ``float4_e2m1fn_x2``: PyTorch's CUDA
                # path schedules ``direct_copy_kernel`` for it which
                # raises an HSAIL hardware exception (poisoning the HIP
                # context).  Route through ``_cast_dispatch_to_combine``
                # so fp4 -> f32 unpacks via the project-local LUT
                # instead.
                inp_for_diag = _cast_dispatch_to_combine(inp, torch.float32)
                try:
                    diag = (f_tok.float() - inp_for_diag * scale_factor).abs().max().item()
                    print(
                        f"  [DIAG] {check_label}                 |fly - k*inp|.max={diag:.4f}  "
                        "(no mori oracle; treated as informational)"
                    )
                except Exception:
                    pass
                has_nan = torch.isnan(f_tok.float()).any().item()
                has_inf = torch.isinf(f_tok.float()).any().item()
                ok_finite = (not has_nan) and (not has_inf)
                status = "PASS" if ok_finite else "FAIL"
                print(f"  [{status}] combine out_tok finite (NaN={has_nan}, Inf={has_inf})")
                all_pass &= ok_finite

    # === (4) combine output weight round-trip (mori parity) ===
    # Same Bug-A discipline as the token check: prefer mori byte
    # equality when available; otherwise downgrade to NaN/Inf sanity
    # only.  The legacy ``wts * k`` closed-form was a misconception (the
    # combine kernel folds weights through the partial-sum slot, not a
    # plain ``k *`` scale).  Skipped under std-MoE since the weighted-
    # sum kernel folds wts into the token path; the token-side check
    # above already gates std-MoE correctness.
    if not cfg.enable_std_moe and cout_f[1] is not None and rank == 0:
        # Reuse the cap-shrinks gate from the token check above: when
        # the cap drops slots, weights diverge between FlyDSL (capped)
        # and mori (uncapped) the same way tokens do, so demote to a
        # NaN/Inf liveness check.
        cap_shrinks_wts = cfg.max_total_recv_tokens > 0 and cfg.effective_max_recv < cfg.max_recv
        try:
            f_out_wts = cout_f[1][: cfg.max_num_inp_token_per_rank].float()
            if cap_shrinks_wts and mori_out_wts is not None:
                has_nan = torch.isnan(f_out_wts).any().item()
                has_inf = torch.isinf(f_out_wts).any().item()
                ok_finite = (not has_nan) and (not has_inf)
                status = "PASS" if ok_finite else "FAIL"
                diff_w = (f_out_wts - mori_out_wts.float()).abs().max().item()
                print(
                    f"  [{status}] cap-overflow combine out_wts liveness  "
                    f"NaN={has_nan} Inf={has_inf}  "
                    f"|fly - mori (uncapped)|.max={diff_w:.6f}"
                )
                all_pass &= ok_finite
                mori_out_wts = None
            if mori_out_wts is not None:
                all_pass &= _check_close(
                    "combine out_wts vs mori",
                    f_out_wts,
                    mori_out_wts.float(),
                    atol=0.0,
                    rtol=0.0,
                    rank=rank,
                )
                try:
                    diag = (f_out_wts - wts.float() * k).abs().max().item()
                    print(f"  [DIAG] combine out_wts                 |fly - k*wts|.max={diag:.6f}")
                except Exception:
                    pass
            else:
                has_nan = torch.isnan(f_out_wts).any().item()
                has_inf = torch.isinf(f_out_wts).any().item()
                ok_finite = (not has_nan) and (not has_inf)
                status = "PASS" if ok_finite else "FAIL"
                print(f"  [{status}] combine out_wts finite (NaN={has_nan}, Inf={has_inf})")
                all_pass &= ok_finite
        except Exception as e:
            print(f"  [INFO] combine-wts check exception: {e}")

    # Cross-rank AND reduction (problem 2): a failure on any rank must
    # fail the whole job, otherwise rank 0 alone could falsely PASS.
    all_pass = _global_reduce_all_pass(all_pass, rank)

    if rank == 0:
        result = "ALL PASS" if all_pass else "SOME FAILED"
        print(f"\n  >>> {result} (global across {world_size} ranks) <<<\n")
    return all_pass


# --- Main entry ---
def run_profiler(rank, world_size, args):
    dev = torch.device("cuda", rank)
    k = args.k
    cur_tok = args.max_tokens
    n_exp = world_size * args.num_experts_per_rank

    _dtype = DTYPE_MAP.get(args.dtype, torch.bfloat16)
    _comb_dtype = _resolve_combine_dtype(args, _dtype)

    # Mori-parity buffer-capacity hint for mixed dtypes: when dispatch
    # and combine use different dtypes, ``cfg.max_token_type_size`` must
    # cover whichever dtype writes the largest per-element record (else
    # the shmem ``comb_inp_tok`` view-cast would OOB the per-row budget).
    _user_mtt = getattr(args, "max_token_type_size", 0)
    if _comb_dtype != _dtype:
        _disp_es = 1 if _dtype == torch.float4_e2m1fn_x2 else torch.tensor([], dtype=_dtype).element_size()
        _comb_es = 1 if _comb_dtype == torch.float4_e2m1fn_x2 else torch.tensor([], dtype=_comb_dtype).element_size()
        _auto_mtt = max(_disp_es, _comb_es)
        if _user_mtt <= 0 or _user_mtt < _auto_mtt:
            _user_mtt = _auto_mtt

    cfg = FlyDSLDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden_dim,
        max_num_inp_token_per_rank=cur_tok,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=k,
        data_type=_dtype,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        dispatch_warp_num_per_block=args.dispatch_warp_per_block,
        dispatch_block_num=args.dispatch_block_num,
        combine_warp_num_per_block=args.combine_warp_per_block,
        combine_block_num=args.combine_block_num,
        use_external_inp_buf=not args.zero_copy,
        enable_std_moe=args.enable_std_moe,
        scale_dim=args.scale_dim,
        scale_type_size=args.scale_type_size,
        quant_type=args.quant_type,
        max_total_recv_tokens=getattr(args, "max_total_recv_tokens", 0),
        max_token_type_size=_user_mtt,
    )

    mori_bn = args.mori_block_num if args.mori_block_num > 0 else cfg.dispatch_block_num
    mori_wpb = args.mori_warp_per_block if args.mori_warp_per_block > 0 else cfg.dispatch_warp_num_per_block
    meta = dict(
        world_size=world_size,
        max_tokens=cur_tok,
        hidden_dim=cfg.hidden_dim,
        k=k,
        num_experts_per_rank=args.num_experts_per_rank,
        warmup=args.warmup,
        iters=args.iters,
        flydsl_dispatch_block_num=cfg.dispatch_block_num,
        flydsl_dispatch_warp_per_block=cfg.dispatch_warp_num_per_block,
        flydsl_combine_block_num=cfg.combine_block_num,
        flydsl_combine_warp_per_block=cfg.combine_warp_num_per_block,
        mori_block_num=mori_bn,
        mori_warp_per_block=mori_wpb,
        zero_copy=cfg.zero_copy,
        enable_std_moe=cfg.enable_std_moe,
        scale_dim=cfg.scale_dim,
        scale_type_size=cfg.scale_type_size,
        quant_type=cfg.quant_type,
    )

    # Output dir layout: <output_dir>/ep{ws}_bs{cur_tok}/
    out_dir = os.path.join(args.output_dir, f"ep{world_size}_bs{cur_tok}")
    os.makedirs(out_dir, exist_ok=True)

    # Build ops.
    if rank == 0:
        print(f"\n{'='*65}", flush=True)
        print(f"[profiler] EP={world_size}, bs={cur_tok}, h={cfg.hidden_dim}, k={k}", flush=True)
        print(f"{'='*65}", flush=True)
        print("[profiler] building FlyDSL...", flush=True)
    op_fly = FlyDSLDispatchCombineIntraNodeOp(cfg)

    # Mori reference op.  Constructed whenever it can act as a verify
    # oracle (dtype supported, not std-MoE) — independent of
    # ``--compare-mori`` which only governs whether the *timing phase*
    # also benches mori for perf comparison.  Without this oracle the
    # verify path can only do a NaN/Inf liveness gate on combine
    # outputs (see ``verify_self`` for details — Bug-A history).
    op_ref = None
    _want_mori = not cfg.enable_std_moe and cfg.quant_type == "none"
    if _want_mori:
        mori_bn = args.mori_block_num if args.mori_block_num > 0 else None
        mori_wpb = args.mori_warp_per_block if args.mori_warp_per_block > 0 else None
        bn_str = mori_bn if mori_bn else cfg.dispatch_block_num
        wpb_str = mori_wpb if mori_wpb else cfg.dispatch_warp_num_per_block
        if rank == 0:
            print(
                f"[profiler] building mori ref (block_num={bn_str}, warp_per_block={wpb_str}) "
                f"-- used as verify oracle{' + perf bench' if args.compare_mori else ''}..."
            )
        try:
            op_ref = build_mori_ref(rank, world_size, cfg, block_num=mori_bn, warp_per_block=mori_wpb)
        except Exception as e:
            if rank == 0:
                print(f"[warn] mori ref unavailable: {e}")
    elif cfg.enable_std_moe and rank == 0:
        print("[info] StdMoE mode: skipping mori ref, using self-check")
    ms.shmem_barrier_all()

    # Prepare inputs (fixed seed so FlyDSL and mori see identical data).
    torch.manual_seed(42 + rank)
    if cfg.data_type == torch.float4_e2m1fn_x2:
        inp = torch.randint(0, 256, (cur_tok, cfg.hidden_dim // 2), dtype=torch.uint8, device=dev).view(
            torch.float4_e2m1fn_x2
        )
    elif cfg.data_type in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        inp = torch.randn(cur_tok, cfg.hidden_dim, dtype=torch.bfloat16, device=dev).to(cfg.data_type)
    else:
        inp = torch.randn(cur_tok, cfg.hidden_dim, dtype=cfg.data_type, device=dev)
    wts = torch.rand(cur_tok, k, dtype=torch.float32, device=dev)
    wts = wts / wts.sum(-1, keepdim=True)
    epr = args.num_experts_per_rank
    idx = torch.zeros(cur_tok, k, dtype=torch.int32, device=dev)
    routing_mode = getattr(args, "routing", "random")
    if routing_mode == "forced_hot_spot":
        # Deterministic worst-case routing: every token's k experts land
        # on PE 0's local experts only.  Drives every dispatched slot to
        # rank 0, so the per-dest receive counter saturates at
        # ``ws * M`` ( == the worst case) while every other PE sees zero
        # tokens.  Exercises the dispatch overflow / sentinel codepath
        # under deterministic (vs random) routing.  Note: k must be
        # <= ``num_experts_per_rank`` so the k experts on PE 0 can stay
        # distinct (else the dedup branch would compress them, which
        # defeats the test); enforced by the assert below.
        assert k <= epr, f"forced_hot_spot routing requires k ({k}) <= num_experts_per_rank ({epr})"
        for t in range(cur_tok):
            # k distinct experts all on PE 0 => expert id = 0 * epr + j.
            for j in range(k):
                idx[t, j] = j
    elif routing_mode == "soft_hot_pe":
        # "Soft" hot-spot variant of ``forced_hot_spot``: every token has
        # exactly ONE of its k experts pinned to PE 0 (a random local
        # expert there) and the remaining ``k-1`` experts uniformly
        # spread over the other ``world_size - 1`` PEs (each on a
        # distinct PE).  The PE-0 atomic_add load drops from
        # ``cur_tok * k`` (forced_hot_spot's all-on-PE-0 pattern, which
        # livelocks ROCm IPC fabric on the single ``shmem_tok_off[0]``
        # slot) to ``cur_tok``, while PE 0 still sees ~k× the average
        # token rate of the other PEs -- a realistic MoE hot-expert
        # imbalance pattern.  Used as the L1 perf-matrix imbalance
        # routing because ``forced_hot_spot`` is not exercised in CI
        # (ROCm IPC fabric livelock; available via CLI only).
        if not (k - 1 <= world_size - 1):
            raise ValueError(
                f"soft_hot_pe routing requires k-1 ({k - 1}) <= "
                f"world_size-1 ({world_size - 1}) so the non-PE-0 "
                f"experts can stay on distinct PEs"
            )
        for t in range(cur_tok):
            # Slot 0 -> PE 0, random local expert.
            idx[t, 0] = 0 * epr + torch.randint(0, epr, (1,), device=dev)
            # Slots 1..k-1 -> distinct PEs from {1..world_size-1}.
            other_pes = torch.randperm(world_size - 1, device=dev)[: k - 1] + 1
            for j in range(k - 1):
                idx[t, j + 1] = other_pes[j] * epr + torch.randint(0, epr, (1,), device=dev)
    elif k <= world_size:
        # Every run now embeds a FlyDSL self-check at startup.  That
        # self-check (and mori's IntraNode bench too) assumes each
        # token's k experts land on k DISTINCT PEs: FlyDSL dispatch
        # deduplicates same-PE assignments while mori does not, so a
        # collision would let the self-check disagree with ``k*inp``.
        for t in range(cur_tok):
            pes = torch.randperm(world_size, device=dev)[:k]
            for j in range(k):
                idx[t, j] = pes[j] * epr + torch.randint(0, epr, (1,), device=dev)
    else:
        # k > world_size: distinct PEs are impossible, fall back to
        # plain random expert ids.  Used only for stress configs that
        # intentionally exercise the dedup path.
        for t in range(cur_tok):
            idx[t] = torch.randperm(n_exp, device=dev)[:k]

    # Pre-allocate the combine weight buffer (shared by FlyDSL and mori
    # so no extra GPU kernel sneaks into the timing window).
    max_recv = world_size * cur_tok
    wc_buf = torch.full((max_recv, k), 1.0 / k, dtype=torch.float32, device=dev)

    # Build scales / packed_recv_x (shared across modes).
    packed_recv_x = None
    if cfg.enable_std_moe:
        _prx_nbytes = cfg.num_experts_per_rank * cfg.max_recv * cfg.token_bytes
        packed_recv_x = (
            torch.zeros(_prx_nbytes, dtype=torch.uint8, device=dev)
            .view(cfg.data_type)
            .view(cfg.num_experts_per_rank * cfg.max_recv, cfg.token_view_dim)
        )

    scales = None
    if cfg.scale_dim > 0 and cfg.scale_type_size > 0:
        _sc_bytes = cfg.scale_dim * cfg.scale_type_size
        scales = torch.randn(cur_tok, _sc_bytes // 4, dtype=torch.float32, device=dev).contiguous()
        scales = scales.view(torch.uint8).view(cur_tok, _sc_bytes)

    # ------------------------------------------------------------------
    # Capture per-rank ``total_recv`` so every timing helper can stamp
    # a GB/s column on its table (mirrors mori
    # ``bench_dispatch_combine.py`` L236 which also runs one eager
    # dispatch to learn this number).  total_recv depends only on
    # ``idx`` (fixed for the entire run), so one read suffices.
    # ------------------------------------------------------------------
    # mori shmem ops (and FlyDSL's dispatch which builds on them) are
    # inherently collective: every rank must enter dispatch together
    # or peers may read uninitialised symmetric heap pages, surfacing
    # as HIP "illegal memory access".  Bracket the probe with explicit
    # barriers, same pattern verify_self uses.
    # ------------------------------------------------------------------
    # Capture per-rank ``total_recv`` so every timing helper can stamp
    # a GB/s column on its table (mirrors mori ``bench_dispatch_combine.py``
    # L236 which also runs one eager round-trip to learn this).  We do
    # a *paired* dispatch + combine here -- a lone dispatch leaves the
    # shmem heap half-written by peers, which surfaces as a HIP illegal
    # memory access on the next kernel launch.  total_recv depends only
    # on ``idx`` (fixed for the whole run) so a single read suffices.
    # ------------------------------------------------------------------
    # NOTE: combine() zeroes ``self.total_recv`` as part of its
    # teardown for the next round, so we MUST read ``ret[4]`` *after
    # dispatch + sync* but *before* combine.  combine still runs so the
    # shmem heap is fully drained (a lone dispatch leaves peer-written
    # output / index buffers half-filled, which surfaced as a HIP
    # "illegal memory access" on the next kernel launch).
    ms.shmem_barrier_all()
    _ret_for_tr = op_fly.dispatch(inp, wts, scales, idx, packed_recv_x=packed_recv_x)
    torch.cuda.synchronize()
    total_recv_per_rank = int(_ret_for_tr[4].item())
    _run_combine(op_fly, _ret_for_tr, _comb_dtype, packed_recv_x=packed_recv_x)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    meta["total_recv"] = total_recv_per_rank
    meta["token_bytes_per_tok"] = cfg.token_bytes
    meta["combine_dtype"] = str(_comb_dtype)
    if rank == 0:
        print(f"[setup] per-rank total_recv = {total_recv_per_rank} tokens; " f"token bytes = {cfg.token_bytes}")

    # profile+eager needs an external warmup; the other three combos
    # warm up inside their own functions.
    do_warmup = args.mode == "profile" and not args.cudagraph

    if do_warmup:
        if rank == 0:
            print(f"[setup] warming up FlyDSL for {args.warmup} iters...")
        for _ in range(args.warmup):
            ms.shmem_barrier_all()
            ret = op_fly.dispatch(inp, wts, scales, idx, packed_recv_x=packed_recv_x)
            _run_combine(op_fly, ret, _comb_dtype, packed_recv_x=packed_recv_x)
            torch.cuda.synchronize()

        if op_ref is not None:
            if rank == 0:
                print(f"[setup] warming up mori ref for {args.warmup} iters...")
            for _ in range(args.warmup):
                op_ref.reset()
                ret_r = op_ref.dispatch(inp, wts, None, idx)
                _run_combine(op_ref, ret_r, _comb_dtype)
                torch.cuda.synchronize()

    ms.shmem_barrier_all()

    # ------------------------------------------------------------------
    # Embedded accuracy gate.
    #
    # Every bench / profile run starts with a FlyDSL self-check so the
    # timing numbers we report never come from a silently-broken kernel.
    # Accuracy is independent of --compare-mori, which only governs
    # whether the timing phase also runs a mori reference for perf
    # comparison.
    # ------------------------------------------------------------------
    ok = verify_self(
        op_fly, inp, wts, idx, k, rank, world_size, dev, args.dtype, cfg, op_mori=op_ref, combine_dtype=_comb_dtype
    )
    if not ok:
        if rank == 0:
            print("[run_profiler] embedded verify FAILED -- skipping timing")
        return ok
    # verify_self leaves op_fly resynchronised via its own internal
    # barriers; one more collective barrier so every rank enters the
    # timing loop together.
    ms.shmem_barrier_all()
    if op_ref is not None:
        try:
            op_ref.reset()
        except Exception:
            pass
    ms.shmem_barrier_all()

    # Timing target selection. FlyDSL is always benched; mori is added
    # iff the user passed --compare-mori AND a mori reference op was
    # successfully built (e.g. mori kernels for the requested dtype are
    # available in the container).
    test_flydsl = True
    test_mori = args.compare_mori and op_ref is not None

    if args.mode == "bench" and not args.cudagraph:
        if test_flydsl:
            bench_op(
                op_fly,
                "flydsl",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.warmup,
                args.iters,
                meta,
                scales=scales,
                packed_recv_x=packed_recv_x,
                combine_dtype=_comb_dtype,
            )
        if test_mori:
            ms.shmem_barrier_all()
            bench_op(
                op_ref,
                "mori",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.warmup,
                args.iters,
                meta,
                combine_dtype=_comb_dtype,
            )

    elif args.mode == "bench" and args.cudagraph:
        if test_flydsl:
            cudagraph_op(
                op_fly,
                "flydsl",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.warmup,
                args.iters,
                meta,
                scales=scales,
                packed_recv_x=packed_recv_x,
                combine_dtype=_comb_dtype,
            )
        if test_mori:
            ms.shmem_barrier_all()
            cudagraph_op(
                op_ref,
                "mori",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.warmup,
                args.iters,
                meta,
                combine_dtype=_comb_dtype,
            )

    elif args.mode == "profile" and not args.cudagraph:
        _zero_copy = args.zero_copy
        if test_flydsl:
            profile_op(
                op_fly,
                "flydsl",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.iters,
                out_dir,
                meta,
                scales=scales,
                packed_recv_x=packed_recv_x,
                dtype_key=args.dtype,
                quant_type=args.quant_type,
                zero_copy=_zero_copy,
                combine_dtype=_comb_dtype,
            )
        if test_mori:
            ms.shmem_barrier_all()
            profile_op(
                op_ref,
                "mori",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.iters,
                out_dir,
                meta,
                dtype_key=args.dtype,
                quant_type=args.quant_type,
                zero_copy=_zero_copy,
                combine_dtype=_comb_dtype,
            )
        if rank == 0:
            print(f"\n[profiler] all results saved to: {out_dir}/")

    elif args.mode == "profile" and args.cudagraph:
        _zero_copy = args.zero_copy
        if test_flydsl:
            profile_cudagraph_op(
                op_fly,
                "flydsl",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.warmup,
                args.iters,
                out_dir,
                meta,
                scales=scales,
                packed_recv_x=packed_recv_x,
                dtype_key=args.dtype,
                quant_type=args.quant_type,
                zero_copy=_zero_copy,
                combine_dtype=_comb_dtype,
            )
        if test_mori:
            ms.shmem_barrier_all()
            profile_cudagraph_op(
                op_ref,
                "mori",
                inp,
                wts,
                idx,
                wc_buf,
                k,
                rank,
                world_size,
                dev,
                args.warmup,
                args.iters,
                out_dir,
                meta,
                dtype_key=args.dtype,
                quant_type=args.quant_type,
                zero_copy=_zero_copy,
                combine_dtype=_comb_dtype,
            )
        if rank == 0:
            print(f"\n[profiler] all results saved to: {out_dir}/")


# --- CI sweep runner (worker-side) ---
def _apply_ci_case(args, case, *, phase, output_dir):
    """Mutate ``args`` in place to match a CI case (used by ``_worker``)."""
    for fk, fv in case.items():
        if fk in _CI_META_FIELDS:
            continue
        setattr(args, fk, fv)
    if phase == "verify":
        # The "verify" phase reuses the same execution path as the
        # perf phase (``profile + cudagraph``) so the entire test
        # plan only ever touches one timing/launch codepath -- this
        # matches the merge-test plan that explicitly disables
        # ``--mode bench``.  ``run_profiler`` always runs the embedded
        # ``verify_self`` accuracy gate before timing (see the
        # ``embedded verify`` call site), so a minimal-iter profile
        # launch is sufficient to exercise the accuracy invariants
        # without paying the full perf-iters overhead.  ``warmup=0``
        # because the cudagraph capture itself plus the embedded
        # verify already trigger JIT compilation; an explicit warmup
        # phase would just cost extra time without changing the
        # accuracy result.
        args.mode = "profile"
        args.cudagraph = True
        args.warmup = 0
        args.iters = 1
        # Accuracy is always FlyDSL self-check (the embedded verify
        # inside run_profiler).  --compare-mori only affects the
        # timing phase: it builds a mori ref op and times it for
        # head-to-head comparison.
    elif phase == "profile":
        args.mode = "profile"
        args.cudagraph = True
        # profile_cudagraph_op uses a torch.profiler scheduler that skips
        # the first 5 warmup samples and keeps the last 3 ``active`` ones;
        # iters < 8 leaves the trace empty (0us across the board).  Use a
        # generous default so every case yields a usable measurement.
        args.warmup = max(getattr(args, "warmup", 0), 5)
        args.iters = max(getattr(args, "iters", 0), 10)
        # Profile defaults to FlyDSL-only timing. Passing --compare-mori
        # on the sweep command line builds a mori ref AND times it for
        # head-to-head comparison; mori kernels missing for fp8_ocp /
        # fp4 naturally fall back to FlyDSL-only inside build_mori_ref.
        args.output_dir = os.path.join(output_dir, f"ci_sweep/{case['name']}")
    else:
        raise ValueError(f"unknown sweep phase: {phase!r}")


# --- Worker / CLI entry ---
def _worker(rank, world_size, args, master_port):
    """Worker process entry.

    Translates any error or verify failure into a non-zero exit code so the
    parent process (and CI) actually observes the failure. Previously
    exceptions were merely printed and the worker exited 0, which let
    ``--mode verify`` silently "pass" on real failures (problem 1).
    """
    setup_distributed(rank, world_size, master_port)
    exit_code = 0
    try:
        # CI sweep dispatch: one spawn cycle per (case, phase) pair, fed in
        # by the parent ``main`` as ``args._ci_case`` + ``args._ci_phase``
        # attributes. Each phase reconfigures ``args`` in place and then
        # falls through to the regular single-case path, so the worker
        # never juggles multiple ops in one process (which previously
        # caused symmetric shmem heap exhaustion and hangs).
        case = getattr(args, "_ci_case", None)
        if case is not None:
            phase = getattr(args, "_ci_phase", "verify")
            base_output_dir = getattr(args, "_ci_base_output_dir", args.output_dir)
            _apply_ci_case(args, case, phase=phase, output_dir=base_output_dir)
        ret = run_profiler(rank, world_size, args)
        # run_profiler returns ``True``/``False`` only in verify mode; for
        # other modes it returns ``None`` and we treat it as success.
        if ret is False:
            exit_code = 1
            print(f"[rank {rank}] verify FAILED")
    except Exception as e:
        import traceback as tb

        print(f"[rank {rank}] ERROR: {e}")
        tb.print_exc()
        exit_code = 2
    finally:
        cleanup()
    if exit_code != 0:
        # ``sys.exit`` makes ``torch.multiprocessing.spawn`` raise
        # ``ProcessRaisedException`` on the parent so the outer ``main``
        # can re-raise with a non-zero exit code.
        sys.exit(exit_code)


def _parse_args():
    p = argparse.ArgumentParser(description="torch.profiler analysis of dispatch/combine")
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=7168)
    p.add_argument("--num-experts-per-rank", type=int, default=32)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--block-num", type=int, default=128)
    p.add_argument("--warp-per-block", type=int, default=8)
    p.add_argument("--dispatch-block-num", type=int, default=128, help="FlyDSL dispatch-only block_num")
    p.add_argument("--dispatch-warp-per-block", type=int, default=4, help="FlyDSL dispatch-only warp_per_block")
    p.add_argument("--combine-block-num", type=int, default=128, help="FlyDSL combine-only block_num")
    p.add_argument("--combine-warp-per-block", type=int, default=8, help="FlyDSL combine-only warp_per_block")
    p.add_argument(
        "--mori-block-num",
        type=int,
        default=0,
        help="mori-only block_num (0 = same as FlyDSL dispatch geometry)",
    )
    p.add_argument(
        "--mori-warp-per-block",
        type=int,
        default=0,
        help="mori-only warp_per_block (0 = same as FlyDSL dispatch geometry)",
    )
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=list(DTYPE_MAP.keys()), help="data type (default: bf16)"
    )
    p.add_argument(
        "--combine-dtype",
        type=str,
        default="",
        choices=["", *DTYPE_MAP.keys()],
        help=(
            "combine launch-time dtype (mori parity).  Empty (default) "
            "reuses --dtype on both dispatch and combine; setting it to "
            "a different dtype (e.g. --dtype float8_e4m3fn --combine-dtype bf16) "
            "exercises the mixed-dtype path: fp4 / float8_e4m3fn dispatch + "
            "bf16 combine, with caller-side staging into the registered "
            "combine input buffer when zero_copy=True."
        ),
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warmup iters outside the profiler (ensures JIT compilation completes)",
    )
    p.add_argument("--iters", type=int, default=5, help="profiler active iters")
    p.add_argument(
        "--output-dir",
        type=str,
        default="dispatch_profile",
        help="JSON output root (relative to cwd); per-shape subdir is named ep{ws}_bs{tok}",
    )
    p.add_argument("--port", type=int, default=29800)
    p.add_argument(
        "--compare-mori",
        action="store_true",
        default=False,
        help=(
            "build a mori reference op AND time it alongside FlyDSL "
            "during bench / profile so the two implementations can be "
            "compared head-to-head (two latency / bw tables printed). "
            "Accuracy is unaffected -- the embedded verify step always "
            "runs the FlyDSL self-check regardless of this flag. "
            "Default off: FlyDSL-only timing, no mori ref constructed."
        ),
    )
    # Mode selection. Accuracy is always checked in-line at the start
    # of the run via the FlyDSL self-check; timing only runs once that
    # embedded verify passes.  --compare-mori does NOT change accuracy
    # behaviour, only adds a mori reference to the timing tables.
    p.add_argument(
        "--mode",
        choices=["profile", "bench"],
        default="profile",
        help=(
            "timing measurement: profile=torch.profiler (default); "
            "bench=CUDA event timing. Both modes embed an accuracy "
            "check at the start of every run."
        ),
    )
    p.add_argument("--cudagraph", action="store_true", help="use CUDAGraph capture+replay (default: eager)")
    # Feature switches
    p.add_argument(
        "--zero-copy",
        action="store_true",
        default=False,
        help="enable zero-copy combine variant (default: disabled)",
    )
    p.add_argument("--enable-std-moe", action="store_true", default=False, help="enable Standard MoE adapt mode")
    p.add_argument(
        "--ci-sweep",
        action="store_true",
        default=False,
        help=(
            "ignore single-case args and run the curated CI_CASES table: "
            "each case is gated by --mode verify (accuracy) followed by "
            "--mode profile --cudagraph (perf). Used by .github/workflows/flydsl.yaml."
        ),
    )
    p.add_argument(
        "--cases-file",
        type=str,
        default="",
        help=(
            "OPTIONAL JSON file with a list of case dicts that replaces "
            "the in-tree CI_CASES table; each dict has the same shape as "
            "a CI_CASES entry (``name`` plus argparse fields, optional "
            "``skip_profile`` / ``skip_ci`` / ``requires_arch`` metadata). "
            "Used by bench/dispatch_combine_merge/run_full.sh to drive "
            "the L1 main matrix without polluting the workflow CI sweep."
        ),
    )
    p.add_argument(
        "--skip-verify-spawn",
        action="store_true",
        default=False,
        help=(
            "Skip the standalone verify spawn in --ci-sweep and rely on "
            "the embedded verify_self that runs at the top of every "
            "profile spawn. Halves spawn count when nearly all cases pass "
            "(L1 main matrix); a verify failure still surfaces because "
            "run_profiler returns False before timing on a failed gate."
        ),
    )
    p.add_argument("--scale-dim", type=int, default=0, help="scale tensor dim (0 = disable scales)")
    p.add_argument("--scale-type-size", type=int, default=0, help="scale element size in bytes (0 = disable scales)")
    p.add_argument(
        "--routing",
        type=str,
        default="random",
        choices=["random", "forced_hot_spot", "soft_hot_pe"],
        help=(
            "indices generation policy: 'random' (default) routes each "
            "token to k distinct PEs uniformly; 'forced_hot_spot' routes "
            "every token's k experts onto PE 0's local experts only "
            "(deterministic worst-case, IPC-livelocks on ROCm fabric so "
            "not exercised in CI -- exposed via CLI for manual IPC-atomic "
            "regression hunting); 'soft_hot_pe' pins ONE of the k experts "
            "per token to PE 0 and spreads the rest over distinct other "
            "PEs, giving a realistic ~k× hot-spot imbalance without "
            "the slot-0 atomic-add livelock."
        ),
    )
    p.add_argument(
        "--quant-type",
        type=str,
        default="none",
        choices=["none", "fp8_direct_cast"],
        help="quantization type (none = default; fp8_direct_cast = inline fp8 cast in combine)",
    )
    p.add_argument(
        "--max-total-recv-tokens",
        type=int,
        default=0,
        help=(
            "explicit cap on total received tokens across all ranks "
            "(0 = use worst-case world_size * max_num_inp_token_per_rank). "
            "Per-rank slot count becomes ceil(cap / world_size).  Shrinks "
            "symmetric shmem token/metadata buffers linearly (mori parity)."
        ),
    )
    p.add_argument(
        "--max-token-type-size",
        type=int,
        default=0,
        help=(
            "explicit upper bound (in bytes) on the token element size "
            "(0 = derive from --dtype).  Lets the op stay alive across "
            "dtype changes without re-allocating shmem (mori parity)."
        ),
    )
    return p.parse_args()


def _spawn_one(ws, args, master_port):
    """Spawn ``ws`` workers, raise non-zero exit if any worker fails."""
    import copy

    try:
        torch.multiprocessing.spawn(
            _worker,
            args=(ws, copy.copy(args), master_port),
            nprocs=ws,
            join=True,
        )
        return True
    except torch.multiprocessing.ProcessRaisedException as e:
        print(f"[main] worker raised: {e}")
        return False
    except torch.multiprocessing.ProcessExitedException as e:
        print(f"[main] worker exited non-zero: {e}")
        return False


def _load_cases_file(path: str) -> list:
    """Load a JSON list of case dicts from ``path``.

    Each dict must at minimum have a ``name`` field (used for
    output-dir naming and the summary table); the remaining keys must
    correspond to argparse names (``--max-tokens`` -> ``max_tokens``)
    plus optional sweep metadata (``skip_profile``, ``skip_ci``,
    ``requires_arch``, ``known_failure``).  Used by
    bench/dispatch_combine_merge/run_full.sh to feed the L1 main
    matrix without polluting the in-tree CI_CASES table.
    """
    with open(path) as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError(f"cases-file {path!r} must contain a JSON list of case dicts, got {type(cases)}")
    for i, c in enumerate(cases):
        if not isinstance(c, dict) or "name" not in c:
            raise ValueError(f"cases-file {path!r} entry [{i}] missing 'name' field: {c!r}")
    return cases


def _resolve_random_fields(case):
    """Materialise any ``_random_fields`` entries into concrete values.

    A case may carry an optional ``"_random_fields"`` dict mapping a field
    name to a non-empty list of choices, e.g.::

        {"zero_copy": [False, True]}

    Each invocation picks one value per random field via ``random.choice``
    and returns a NEW case dict with the random fields collapsed to
    concrete scalars (the resolved dict is what flows into the spawn so
    all ranks see the same value).  Picks are also returned for logging.
    """
    rfs = case.get("_random_fields")
    if not rfs:
        return case, {}
    resolved = {k: v for k, v in case.items() if k != "_random_fields"}
    picks = {}
    for fk, fv_choices in rfs.items():
        if not isinstance(fv_choices, (list, tuple)) or not fv_choices:
            raise ValueError(f"_random_fields[{fk!r}] must be a non-empty list/tuple of choices, got {fv_choices!r}")
        picked = random.choice(list(fv_choices))
        resolved[fk] = picked
        picks[fk] = picked
    return resolved, picks


def _run_ci_sweep_main(ws, args):
    """Orchestrate CI sweep: one ``spawn`` per (case, phase).

    Each spawn cycle creates a fresh distributed group, fresh CUDA
    contexts and a fresh mori symmetric shmem heap, so cross-case
    resource contention (which previously hung the sweep) cannot occur.
    Accuracy failures stop only that case's perf phase, not the sweep.

    With ``--cases-file <path>`` the in-tree :data:`CI_CASES` is
    replaced by the JSON-loaded case list; with
    ``--skip-verify-spawn`` the standalone verify spawn is collapsed
    into the profile spawn (the embedded ``verify_self`` inside
    :func:`run_profiler` still gates timing, so a verify failure still
    surfaces -- this just halves spawn count on the L1 main matrix
    where nearly all cases pass).
    """
    base_output_dir = args.output_dir
    base_port = args.port
    per_case_status = []  # [(name, verify_label, profile_label)]
    overall_ok = True
    cur_arch = _current_gpu_arch_prefix()

    cases = CI_CASES
    cases_source = "CI_CASES (in-tree)"
    if getattr(args, "cases_file", ""):
        cases = _load_cases_file(args.cases_file)
        cases_source = f"cases-file {args.cases_file!r}"

    skip_verify_spawn = bool(getattr(args, "skip_verify_spawn", False))

    print(f"\n{'#'*70}")
    print(f"# CI sweep: {len(cases)} cases  (world_size={ws}, arch={cur_arch or 'unknown'})")
    print(f"# source: {cases_source}")
    print(f"# skip-verify-spawn: {skip_verify_spawn}  (verify_self still runs inside profile spawn)")
    print(f"# base output dir: {base_output_dir}")
    print(f"{'#'*70}")

    for idx, case in enumerate(cases):
        case, rand_picks = _resolve_random_fields(case)
        print(f"\n{'='*70}")
        print(f"# [case {idx + 1}/{len(cases)}] {case['name']}")
        for fk, fv in case.items():
            if fk != "name":
                tag = "  (random pick)" if fk in rand_picks else ""
                print(f"#   {fk:>22} = {fv}{tag}")
        print(f"{'='*70}")

        # Per-case world_size override: L2 accuracy edge cases run on
        # ws=2/4 to exercise the small-EP topology.  ``mp.spawn``'s
        # ``nprocs`` is driven by this value, so it must come from the
        # case dict (not the global --world-size) when present.  Cap to
        # the number of CUDA devices (consistent with main()'s default
        # ws derivation).
        case_ws = int(case.get("world_size", ws))
        case_ws = min(case_ws, torch.cuda.device_count())
        if case_ws != ws:
            print(f"# [case {case['name']}] using world_size={case_ws} (override)")

        # -- skip_ci gate --
        if case.get("skip_ci", False):
            skip_msg = "skipped (case marked skip_ci=True)"
            print(f"\n[case {case['name']}] !! {skip_msg}")
            per_case_status.append((case["name"], skip_msg, skip_msg))
            continue

        # -- arch gate --
        req_arch = case.get("requires_arch")
        if req_arch and cur_arch and cur_arch not in req_arch:
            arch_msg = f"skipped (need {'/'.join(req_arch)}, have {cur_arch})"
            print(f"\n[case {case['name']}] !! {arch_msg}")
            per_case_status.append((case["name"], arch_msg, arch_msg))
            continue

        # -- accuracy gate --
        # The embedded verify_self runs at the top of every profile
        # spawn (gates timing on accuracy, returns False on FAIL),
        # so --skip-verify-spawn collapses verify+profile into a
        # single spawn.  When that flag is on we skip this dedicated
        # verify spawn and let the profile spawn double as the
        # accuracy gate.
        if skip_verify_spawn:
            verify_ok = True  # tentative; will be overwritten by profile spawn outcome
            verify_label_tag = "(via embedded verify_self in profile spawn)"
        else:
            v_args = type(args)(**vars(args))
            v_args._ci_case = case
            v_args._ci_phase = "verify"
            v_args._ci_base_output_dir = base_output_dir
            verify_port = base_port + 100 + idx * 2
            print(f"\n[case {case['name']}] >> verify (port={verify_port})")
            verify_ok = _spawn_one(case_ws, v_args, verify_port)
            verify_label_tag = ""

        profile_ok = True
        profile_label = None
        if not verify_ok:
            profile_label = "skipped (verify FAILED)"
        elif case.get("skip_profile", False):
            profile_label = "skipped (case opt-out)"
        else:
            p_args = type(args)(**vars(args))
            p_args._ci_case = case
            p_args._ci_phase = "profile"
            p_args._ci_base_output_dir = base_output_dir
            profile_port = base_port + 101 + idx * 2
            print(f"\n[case {case['name']}] >> profile + cudagraph (port={profile_port})")
            profile_ok = _spawn_one(case_ws, p_args, profile_port)
            profile_label = "ok" if profile_ok else "warn"
            # When the standalone verify spawn was skipped, the
            # profile spawn outcome IS the accuracy outcome (the
            # embedded verify_self ran at the top of run_profiler and
            # would have returned False before any timing started).
            if skip_verify_spawn:
                verify_ok = profile_ok

        if profile_label and profile_label.startswith("skipped"):
            print(f"\n[case {case['name']}] !! {profile_label}")

        # Sweep failure semantics:
        #   - verify PASS         -> case is healthy
        #   - verify FAIL + known -> downgrade to "xfail" (warning, doesn't
        #                            fail the sweep) so a regression on the
        #                            OTHER cases still gets surfaced clearly
        #   - verify FAIL         -> fail the sweep
        #   - profile fails       -> warn only
        known_fail_tag = case.get("known_failure")
        if verify_ok:
            verify_label = "PASS"
            if skip_verify_spawn:
                verify_label = f"PASS {verify_label_tag}"
        elif known_fail_tag:
            verify_label = f"xfail ({known_fail_tag})"
        else:
            verify_label = "FAIL"
        per_case_status.append((case["name"], verify_label, profile_label or ("ok" if profile_ok else "warn")))
        if not verify_ok and not known_fail_tag:
            overall_ok = False  # only unknown failures trip the sweep

    print(f"\n{'#'*70}")
    print("# CI sweep summary")
    print(f"{'#'*70}")
    print(f"# {'case':<35} {'verify':<42} {'profile':<42}")
    print(f"# {'-' * 122}")
    for name, vlabel, plabel in per_case_status:
        print(f"# {name:<35} {vlabel:<42} {plabel:<42}")
    print(f"{'#'*70}")
    result = "ALL PASS" if overall_ok else "SOME FAILED"
    print(f"# >>> {result} (accuracy across {len(cases)} cases) <<<\n")
    return overall_ok


def main():
    args = _parse_args()
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        _worker(rank, world_size, args, master_port=args.port)
        return

    ws = min(args.world_size, torch.cuda.device_count())
    if ws < args.world_size:
        print(f"[warn] available GPUs={torch.cuda.device_count()}, world_size adjusted: {args.world_size} -> {ws}")

    if args.ci_sweep:
        ok = _run_ci_sweep_main(ws, args)
        if not ok:
            sys.exit(1)
        return

    if not _spawn_one(ws, args, args.port):
        sys.exit(1)


if __name__ == "__main__":
    main()
