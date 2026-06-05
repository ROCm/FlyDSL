# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Smoke tests for flydsl.expr.pinned_vgpr.

Pure-Python checks for ``PinnedRange`` / ``PinnedLayout`` semantics, plus
a single trace-compiled kernel that exercises every helper so the wrappers
actually emit MLIR.  No device launch -- we run under ``COMPILE_ONLY=1``.
"""
import os

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, rocdl
from flydsl.expr import pinned_vgpr as pv
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


# ===========================================================================
# Pure-Python: PinnedRange / PinnedLayout semantics
# ===========================================================================
class TestPinnedRange:
    def test_basic(self):
        r = pv.PinnedRange(72, 103, name="q_vgpr")
        assert r.size == 32
        assert r.lo == 72 and r.hi == 103
        assert r.asm_name == "v[72:103]"

    def test_singleton_asm(self):
        r = pv.PinnedRange(120, 120, name="x")
        assert r.size == 1
        assert r.asm_name == "v[120]"

    def test_index_int(self):
        r = pv.PinnedRange(72, 103)
        sub = r[5]
        assert sub.lo == 77 and sub.hi == 77 and sub.size == 1

    def test_index_slice(self):
        r = pv.PinnedRange(72, 103)
        sub = r[4:8]
        assert sub.lo == 76 and sub.hi == 79 and sub.size == 4

    def test_split(self):
        r = pv.PinnedRange(72, 103)  # 32 vgprs
        tiles = r.split(4)
        assert len(tiles) == 8
        assert tiles[0].lo == 72 and tiles[0].hi == 75
        assert tiles[7].lo == 100 and tiles[7].hi == 103

    def test_split_bad_size(self):
        r = pv.PinnedRange(72, 75)  # 4 vgprs
        with pytest.raises(ValueError):
            r.split(3)

    def test_out_of_bounds_rejects(self):
        with pytest.raises(ValueError):
            pv.PinnedRange(200, 256)


class TestPinnedLayout:
    def test_declare_and_union_disjoint(self):
        pl = pv.PinnedLayout()
        pl.declare(72, 103, name="q_vgpr")
        pl.declare(128, 255, name="oaccu")
        assert pl.union_intervals == ((72, 103), (128, 255))

    def test_declare_overlap_merges(self):
        pl = pv.PinnedLayout()
        pl.declare(120, 127, name="p_comp")
        pl.declare(120, 123, name="p_mfma")  # overlay
        assert pl.union_intervals == ((120, 127),)

    def test_declare_adjacent_merges(self):
        pl = pv.PinnedLayout()
        pl.declare(64, 71, name="q_lds")
        pl.declare(72, 103, name="q_vgpr")
        assert pl.union_intervals == ((64, 103),)

    def test_clobbers_count(self):
        pl = pv.PinnedLayout()
        pl.declare(120, 127)
        pl.declare(120, 123)  # overlap
        # Union [120,127] = 8 vgprs total
        assert len(pl.clobbers) == 8
        assert pl.clobbers[0] == "~{v120}"
        assert pl.clobbers[-1] == "~{v127}"

    def test_full_v40_layout(self):
        """V40 layout: q_lds, q_vgpr, pv_v_aux, kv, p_comp + p_mfma overlay, oaccu."""
        pl = pv.PinnedLayout()
        pl.declare(64,  71,  name="q_lds")
        pl.declare(72,  103, name="q_vgpr")
        pl.declare(104, 111, name="pv_v_aux")
        pl.declare(112, 119, name="kv")
        pl.declare(120, 127, name="p_comp")
        pl.declare(120, 123, name="p_mfma")  # overlay
        pl.declare(128, 255, name="oaccu")
        # All adjacent or overlapping -> single union interval.
        assert pl.union_intervals == ((64, 255),)
        assert len(pl.clobbers) == 192  # 255 - 64 + 1


# ===========================================================================
# Trace-compile a tiny kernel that touches every helper.
# Only validates: helpers parse, type-check, and produce MLIR without errors.
# No device launch -- this runs under COMPILE_ONLY=1.
# ===========================================================================
NUM_THREADS = 64


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_pinned_vgpr_smoke(
    a_buf: fx.Tensor,       # i64 SSA input source
    b_buf: fx.Tensor,       # i64 SSA input source
    acc_buf: fx.Tensor,     # f32x4 SSA input source
    out_buf: fx.Tensor,     # output (unused, just to force a store)
):
    # Layout exercising overlap + the helpers under test.
    pl_local = pv.PinnedLayout()
    # fp8 mfma: A/B = 2 vgprs each
    a_pin = pl_local.declare(112, 113, name="a_fp8")    # size 2
    b_pin = pl_local.declare(114, 115, name="b_fp8")    # size 2
    d_pin = pl_local.declare(116, 119, name="d_acc")    # size 4
    # f16 mfma: A/B = 4 vgprs each
    a4_pin = pl_local.declare(64, 67, name="a_f16")     # size 4
    b4_pin = pl_local.declare(68, 71, name="b_f16")     # size 4
    p_comp = pl_local.declare(120, 127, name="p_comp")  # size 8
    p_mfma = pl_local.declare(120, 123, name="p_mfma")  # overlay (low half)
    cvt_dst = pl_local.declare(124, 124, name="cvt_dst")  # size 1
    pair = pl_local.declare(126, 127, name="pair")      # size 2

    pl_local.emit_clobber()

    # ---- write_pinned / read_pinned trampolines ----
    from flydsl.expr import buffer_ops, gpu
    a_rsrc = buffer_ops.create_buffer_resource(a_buf)
    b_rsrc = buffer_ops.create_buffer_resource(b_buf)
    acc_rsrc = buffer_ops.create_buffer_resource(acc_buf)
    out_rsrc = buffer_ops.create_buffer_resource(out_buf)

    tid = gpu.thread_id("x")
    a_src = buffer_ops.buffer_load(a_rsrc, tid, vec_width=1, dtype=T.i64)
    b_src = buffer_ops.buffer_load(b_rsrc, tid, vec_width=1, dtype=T.i64)
    acc_src = buffer_ops.buffer_load(acc_rsrc, tid, vec_width=4, dtype=T.f32)
    # f16 mfma needs 4-vgpr A/B (8 bf16 elems = 4 dwords / lane)
    a4_src = buffer_ops.buffer_load(a_rsrc, tid, vec_width=4, dtype=T.i32)
    b4_src = buffer_ops.buffer_load(b_rsrc, tid, vec_width=4, dtype=T.i32)

    # ---- pinned_mfma_fp8_fp8 (accum form, returns f32x4) ----
    d1 = pv.pinned_mfma_fp8_fp8(a_pin, b_pin, d_pin, a_src, b_src, acc_src)

    # ---- pinned_mfma_fp8_fp8_init (3-arg form) ----
    d2 = pv.pinned_mfma_fp8_fp8_init(a_pin, b_pin, d_pin, a_src, b_src)

    # ---- pinned_mfma_bf16 (bf16 PV variant) ----
    d3 = pv.pinned_mfma_bf16(a4_pin, b4_pin, d_pin, a4_src, b4_src, acc_src)

    # ---- pinned_cvt_scalef32_pk_bf16_fp8 (low + high opsel) ----
    src_dw = buffer_ops.buffer_load(a_rsrc, tid, vec_width=1, dtype=T.i32)
    scale_f = fx.Float32(2.0)
    pv.pinned_cvt_scalef32_pk_bf16_fp8(cvt_dst, src_dw, scale_f, opsel=False)
    pv.pinned_cvt_scalef32_pk_bf16_fp8(cvt_dst, src_dw, scale_f, opsel=True)

    # ---- pinned_cvt_pk_bf16_f32 (overlay pack) ----
    f_a = fx.Float32(1.0)
    f_b = fx.Float32(3.0)
    pv.pinned_cvt_pk_bf16_f32(cvt_dst, f_a, f_b)

    # ---- pinned_v_mul_f32_pair (needs packed-fp32 pair) ----
    factor_pair = _raw(fx.Int64(0))  # dummy {0.0, 0.0}
    pv.pinned_v_mul_f32_pair(pair, factor_pair)

    # ---- pinned_softmax_exp_block (both inputs are packed pairs) ----
    neg_max_pair = _raw(fx.Int64(0))
    log2e_pk = _raw(fx.Int64(0))
    pv.pinned_softmax_exp_block(p_comp, neg_max_pair, log2e_pk)

    # ---- pinned_inline_asm escape hatch (no-op s_nop) ----
    pv.pinned_inline_asm("s_nop 0")

    # ---- Store something so the kernel isn't dead-stripped ----
    # Single store sums all three results -- keeps every helper's value live.
    sum_vec = arith.addf(d1, arith.addf(d2, d3))
    buffer_ops.buffer_store(sum_vec, out_rsrc, tid)


@flyc.jit
def launch_smoke(
    a_buf: fx.Tensor, b_buf: fx.Tensor, acc_buf: fx.Tensor, out_buf: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    kn_pinned_vgpr_smoke(a_buf, b_buf, acc_buf, out_buf).launch(
        grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), smem=0, stream=stream,
    )


def test_kernel_trace_compiles():
    """Verify the kernel containing every pinned_vgpr helper produces MLIR."""
    if os.environ.get("COMPILE_ONLY") != "1":
        pytest.skip("requires COMPILE_ONLY=1")
    a = torch.zeros(64, dtype=torch.int64)
    b = torch.zeros(64, dtype=torch.int64)
    acc = torch.zeros(64, 4, dtype=torch.float32)
    out = torch.zeros(64, 4, dtype=torch.float32)
    launch_smoke(a, b, acc, out)
