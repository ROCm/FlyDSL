"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8)."""

import math as _math

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import arith, gpu, rocdl, tdm_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T


def _raw_lds_ptr(lds_base_idx, byte_offset):
    """Materialize an LLVM LDS pointer from a pre-extracted byte base."""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.arith import ArithValue as _AV

    if not isinstance(_raw(byte_offset).type, ir.IndexType):
        byte_offset = arith.index_cast(T.index, byte_offset)
    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    total_byte = _AV(lds_base_idx) + byte_offset
    addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
    return _llvm.inttoptr(lds_ptr_ty, addr_i32)


def lds_load_b128_raw(lds_base_idx, byte_offset):
    """Load 16 bytes from LDS using a pre-extracted base index (raw LLVM).

    Args:
        lds_base_idx: LDS byte-base index value.
        byte_offset: Byte offset (index-type) relative to the base.
    """
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    return llvm_dialect.load(ir.VectorType.get([4], ir.IntegerType.get_signless(32)), ptr_val)


def lds_load_b32_raw(lds_base_idx, byte_offset):
    """Load 4 bytes (one i32) from LDS using a pre-extracted base index (raw LLVM).

    Unlike :func:`lds_load_b128_raw`, this only requires 4-byte alignment, so it
    suits scale layouts where consumed words sit at 4-byte (not 16-byte) granular
    offsets (e.g. the 32x4 B-scale layout's one-i32-per-atom reads).
    """
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    return llvm_dialect.load(ir.IntegerType.get_signless(32), ptr_val)


def lds_store_b128_raw(lds_base_idx, byte_offset, data):
    """Store 16 bytes to LDS using a pre-extracted base index (raw LLVM)."""
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    llvm_dialect.store(_raw(data), ptr_val)


def lds_store_b64_raw(lds_base_idx, byte_offset, data):
    """Store 8 bytes to LDS using a pre-extracted base index (vector<2xi32>)."""
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    llvm_dialect.store(_raw(data), ptr_val)


def workgroup_barrier(use_cluster=False):
    """Issue the appropriate barrier for LDS visibility.

    Cluster mode layers an inter-workgroup barrier on top of the regular
    workgroup barrier protocol, so call sites can treat it as a single
    "LDS is now readable" fence.
    """
    if use_cluster:
        cluster.cluster_barrier()
    else:
        gpu.barrier()


def pipeline_fence(outstanding=0, use_cluster=False):
    """Fused READY+REUSE fence for gfx1250 multi-buffer pipeline.

    Issues ``s_wait_tensorcnt`` followed by the appropriate barrier.
    """
    tdm_ops.tensor_wait(outstanding)
    workgroup_barrier(use_cluster=use_cluster)


WGP_BARRIER_ID = -1


def pipeline_fence_signal(outstanding=0, use_cluster=False):
    """Signal half of a split barrier fence.

    Issues ``s_wait_tensorcnt`` then ``s_barrier_signal -1``.
    The matching ``pipeline_fence_wait`` must be called later
    (typically mid-compute) before reading the LDS data.

    When *use_cluster* is True the intra-WG barrier is still required
    so that all waves' TDM loads are visible before any wave reads LDS.
    The cluster barrier is layered on top for inter-WG synchronisation.
    """
    tdm_ops.tensor_wait(outstanding)
    rocdl.s_barrier_signal(WGP_BARRIER_ID)
    if use_cluster:
        cluster.cluster_signal_once_per_wg()


def pipeline_fence_wait(use_cluster=False):
    """Wait half of a split barrier fence.

    Issues ``s_barrier_wait -1``.  Must be preceded by a matching
    ``pipeline_fence_signal`` from all waves in the workgroup.
    """
    rocdl.s_barrier_wait(WGP_BARRIER_ID)
    if use_cluster:
        cluster.cluster_wait()


LOG2E = _math.log2(_math.e)


def fmin_f32(a, b):
    """Scalar f32 min (select-based, no NaN handling)."""
    import flydsl.expr as _fx

    return _fx.Float32((a < b).select(a, b))


def fmax_f32(a, b):
    """Scalar f32 max (select-based, no NaN handling)."""
    import flydsl.expr as _fx

    return _fx.Float32((a > b).select(a, b))


def fused_silu_swiglu_elem(g, u, *, swiglu, limit_f32, neg_limit_f32):
    """One (gate, up) pair -> fused silu or swiglu scalar (gpt-oss clamp)."""
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    g = fmin_f32(g, limit_f32)
    u = fmin_f32(fmax_f32(u, neg_limit_f32), limit_f32)
    if swiglu:
        nlog2e = _fx.Float32(-1.702 * LOG2E)
        sig = _fx.Float32(rocdl.rcp(T.f32, _one + (g * nlog2e).exp2()))
        return g * sig * (u + _one)
    nlog2e = _fx.Float32(-LOG2E)
    sig = _fx.Float32(rocdl.rcp(T.f32, _one + (g * nlog2e).exp2()))
    return g * sig * u


__all__ = [
    # LDS helpers
    # Raw LLVM path
    "lds_load_b128_raw",
    # Pipeline
    "workgroup_barrier",
    "pipeline_fence",
    "pipeline_fence_signal",
    "pipeline_fence_wait",
    # Scalar math
    "LOG2E",
    "fmin_f32",
    "fmax_f32",
    "fused_silu_swiglu_elem",
]
