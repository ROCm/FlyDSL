"""Shared reduction helpers for FLIR example kernels.

These helpers build MLIR ops (flir/gpu/scf/vector/etc). They are extracted from
softmax/layernorm/rmsnorm kernels to de-duplicate code without changing codegen.
"""
from __future__ import annotations


def unwrap(v):
    if hasattr(v, "value"):
        return v.value
    if hasattr(v, "_value"):
        return v._value
    if hasattr(v, "result"):
        return v.result
    return v


def reduce_vec_max(vec_val, *, VEC_WIDTH, compute_type, vector):
    if VEC_WIDTH == 1:
        return vector.extract(vec_val, static_position=[0], dynamic_position=[])
    # Avoid fastmath on bf16 max reduction; some backends can fail to select.
    return vector.reduction(compute_type, "maxnumf", unwrap(vec_val))


def reduce_vec_sum(vec_val, *, VEC_WIDTH, compute_type, vector, fm_fast):
    if VEC_WIDTH == 1:
        return vector.extract(vec_val, static_position=[0], dynamic_position=[])
    return vector.reduction(compute_type, "add", unwrap(vec_val), fastmath=fm_fast)


def make_block_reduce(*, tid, BLOCK_SIZE, compute_type, arith, gpu, flir, s_red_tv, T, ir, c_zero, c_neg_inf, c_zero_idx, fm_fast):
    """Return a `block_reduce(val, reduce_op_name)` function (softmax-style)."""

    def block_reduce(val, reduce_op_name):
        # AMD wavefront size is 64 on gfx9+/gfx10+/gfx11.
        WARP_SIZE = 64
        NUM_WAVES = (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE  # python int
        # Use Flir layout algebra to compute LDS indices for the reduction scratch.
        c_num_waves = flir.const_index(NUM_WAVES)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(unwrap(c_num_waves))
        stride_red = flir.make_stride(unwrap(c1))
        layout_red = flir.make_layout(shape_red, stride_red)

        tid_i32 = flir.arith.IndexCastOp(T.i32(), unwrap(tid)).result
        c_warp_i32 = arith.constant(WARP_SIZE, type=T.i32()).value
        lane_i32 = flir.arith.RemUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
        wave_i32 = flir.arith.DivUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result

        width_i32 = arith.constant(WARP_SIZE, type=T.i32()).value
        w = unwrap(val)

        # Intra-wave reduction via xor shuffle
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.constant(sh, type=T.i32()).value
            peer = gpu.ShuffleOp(unwrap(w), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
            if reduce_op_name == "max":
                w = flir.arith.MaximumFOp(unwrap(w), unwrap(peer)).result
            else:
                w = flir.arith.AddFOp(unwrap(w), unwrap(peer), fastmath=fm_fast).result

        # lane0 writes per-wave partial into LDS s_red[wave_id]
        is_lane0 = flir.arith.CmpIOp(
            flir.arith.CmpIPredicate.eq,
            unwrap(lane_i32),
            unwrap(arith.constant(0, type=T.i32()).value),
        ).result
        with flir.scf_ext.if_(is_lane0) as then_blk:
            with ir.InsertionPoint(then_blk):
                wave_idx = flir.arith.IndexCastOp(T.index(), unwrap(wave_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(wave_idx)), layout_red)
                s_red_tv[unwrap(red_idx)] = unwrap(w)
        gpu.barrier()

        # wave0 reduces NUM_WAVES partials (still using shuffle)
        is_wave0 = flir.arith.CmpIOp(
            flir.arith.CmpIPredicate.eq,
            unwrap(wave_i32),
            unwrap(arith.constant(0, type=T.i32()).value),
        ).result
        with flir.scf_ext.if_(is_wave0) as then_blk:
            with ir.InsertionPoint(then_blk):
                in_range = flir.arith.CmpIOp(
                    flir.arith.CmpIPredicate.ult,
                    unwrap(lane_i32),
                    unwrap(arith.constant(NUM_WAVES, type=T.i32()).value),
                ).result

                # Predicated load: clamp lane index to 0 when out-of-range, then select.
                c0_i32 = arith.constant(0, type=T.i32()).value
                lane_safe_i32 = flir.arith.SelectOp(unwrap(in_range), unwrap(lane_i32), unwrap(c0_i32)).result
                lane_safe_idx = flir.arith.IndexCastOp(T.index(), unwrap(lane_safe_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(lane_safe_idx)), layout_red)
                v = s_red_tv[unwrap(red_idx)]
                neutral = c_neg_inf if reduce_op_name == "max" else c_zero
                ww = flir.arith.SelectOp(unwrap(in_range), unwrap(v), unwrap(neutral)).result

                for sh in [32, 16, 8, 4, 2, 1]:
                    off = arith.constant(sh, type=T.i32()).value
                    peer = gpu.ShuffleOp(unwrap(ww), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                    if reduce_op_name == "max":
                        ww = flir.arith.MaximumFOp(unwrap(ww), unwrap(peer)).result
                    else:
                        ww = flir.arith.AddFOp(unwrap(ww), unwrap(peer), fastmath=fm_fast).result

                # lane0 writes final to s_red[0]
                is_lane0_2 = flir.arith.CmpIOp(
                    flir.arith.CmpIPredicate.eq,
                    unwrap(lane_i32),
                    unwrap(arith.constant(0, type=T.i32()).value),
                ).result
                with flir.scf_ext.if_(is_lane0_2) as then2:
                    with ir.InsertionPoint(then2):
                        red_idx0 = flir.crd2idx(flir.make_coord(unwrap(c_zero_idx)), layout_red)
                        s_red_tv[unwrap(red_idx0)] = unwrap(ww)
        gpu.barrier()

        red_idx0 = flir.crd2idx(flir.make_coord(unwrap(c_zero_idx)), layout_red)
        return s_red_tv[unwrap(red_idx0)]

    return block_reduce


def make_block_reduce_add(*, tid, fm_fast, WARP_SIZE, RED_SLOTS, gpu, arith, arith_ops, flir, T, ir, zero_idx, scratch_tv_shape_stride=(None, None)):
    """Return a `block_reduce_add(val_f32, scratch_memref)` function (norm-style)."""
    shape_unused, stride_unused = scratch_tv_shape_stride
    _ = shape_unused
    _ = stride_unused

    def block_reduce_add(val_f32, scratch_memref):
        # Fast path: single-wave block (RED_SLOTS==1) needs no LDS and no barrier.
        # After xor-shuffle reduction, all lanes hold the same reduced value.
        if RED_SLOTS == 1:
            width_i32 = arith.constant(T.i32(), WARP_SIZE)
            w = unwrap(val_f32)
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.constant(T.i32(), sh)
                peer = gpu.ShuffleOp(unwrap(w), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                w = arith_ops.AddFOp(unwrap(w), unwrap(peer), fastmath=fm_fast).result
            return w

        scratch_tv = flir.make_tensor(scratch_memref, shape=(RED_SLOTS,), strides=(1,))
        tid_v = tid.value if hasattr(tid, "value") else unwrap(tid)
        tid_i32 = arith_ops.IndexCastOp(T.i32(), tid_v).result
        c_warp_i32 = arith.constant(T.i32(), WARP_SIZE)
        lane_i32 = arith_ops.RemUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
        wave_i32 = arith_ops.DivUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
        width_i32 = arith.constant(T.i32(), WARP_SIZE)
        # Use Flir layout algebra to compute LDS indices for the reduction scratch.
        c_num_waves = flir.const_index(RED_SLOTS)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(unwrap(c_num_waves))
        stride_red = flir.make_stride(unwrap(c1))
        layout_red = flir.make_layout(shape_red, stride_red)

        w = unwrap(val_f32)
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.constant(T.i32(), sh)
            peer = gpu.ShuffleOp(unwrap(w), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
            w = arith_ops.AddFOp(unwrap(w), unwrap(peer), fastmath=fm_fast).result

        is_lane0 = arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            unwrap(lane_i32),
            unwrap(arith.constant(T.i32(), 0)),
        ).result
        with flir.scf_ext.if_(is_lane0) as then_blk:
            with ir.InsertionPoint(then_blk):
                wave_idx = arith_ops.IndexCastOp(T.index(), unwrap(wave_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(wave_idx)), layout_red)
                scratch_tv[unwrap(red_idx)] = unwrap(w)
        gpu.barrier()

        NUM_WAVES = RED_SLOTS
        is_wave0 = arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            unwrap(wave_i32),
            unwrap(arith.constant(T.i32(), 0)),
        ).result
        # Only wave0 does final reduction and writes scratch[0].
        with flir.scf_ext.if_(is_wave0) as then_blk:
            with ir.InsertionPoint(then_blk):
                in_range = arith_ops.CmpIOp(
                    arith_ops.CmpIPredicate.ult,
                    unwrap(lane_i32),
                    unwrap(arith.constant(T.i32(), NUM_WAVES)),
                ).result

                c0_i32 = arith.constant(T.i32(), 0)
                lane_safe_i32 = flir.arith.SelectOp(unwrap(in_range), unwrap(lane_i32), unwrap(c0_i32)).result
                lane_safe_idx = arith_ops.IndexCastOp(T.index(), unwrap(lane_safe_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(lane_safe_idx)), layout_red)
                v = scratch_tv[unwrap(red_idx)]
                z = arith.constant(T.f32(), 0.0).value
                ww = flir.arith.SelectOp(unwrap(in_range), unwrap(v), unwrap(z)).result

                for sh in [32, 16, 8, 4, 2, 1]:
                    off = arith.constant(T.i32(), sh)
                    peer = gpu.ShuffleOp(unwrap(ww), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                    ww = arith_ops.AddFOp(unwrap(ww), unwrap(peer), fastmath=fm_fast).result

                is_lane0_2 = arith_ops.CmpIOp(
                    arith_ops.CmpIPredicate.eq,
                    unwrap(lane_i32),
                    unwrap(arith.constant(T.i32(), 0)),
                ).result
                with flir.scf_ext.if_(is_lane0_2) as then2:
                    with ir.InsertionPoint(then2):
                        red_idx0 = flir.crd2idx(flir.make_coord(unwrap(zero_idx)), layout_red)
                        scratch_tv[unwrap(red_idx0)] = unwrap(ww)

        gpu.barrier()
        red_idx0 = flir.crd2idx(flir.make_coord(unwrap(zero_idx)), layout_red)
        return scratch_tv[unwrap(red_idx0)]

    return block_reduce_add


def make_block_reduce_add2(*, tid, fm_fast, WARP_SIZE, RED_SLOTS, gpu, arith, arith_ops, flir, T, ir, zero_idx):
    """Return a `block_reduce_add2(a_f32, b_f32, scratch_a, scratch_b)` function.

    This is NOT pair-reduce: it reduces two independent scalars but shares the same
    cross-wave synchronization so we only pay the barriers once.
    """

    def _wave_reduce_add(x):
        width_i32 = arith.constant(T.i32(), WARP_SIZE)
        w = unwrap(x)
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.constant(T.i32(), sh)
            peer = gpu.ShuffleOp(unwrap(w), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
            w = arith_ops.AddFOp(unwrap(w), unwrap(peer), fastmath=fm_fast).result
        return w

    def block_reduce_add2(val0_f32, val1_f32, scratch0_memref, scratch1_memref):
        # Single-wave block: no LDS/no barrier, just two wave reductions.
        if RED_SLOTS == 1:
            return _wave_reduce_add(val0_f32), _wave_reduce_add(val1_f32)

        scratch0_tv = flir.make_tensor(scratch0_memref, shape=(RED_SLOTS,), strides=(1,))
        scratch1_tv = flir.make_tensor(scratch1_memref, shape=(RED_SLOTS,), strides=(1,))

        tid_v = tid.value if hasattr(tid, "value") else unwrap(tid)
        tid_i32 = arith_ops.IndexCastOp(T.i32(), tid_v).result
        c_warp_i32 = arith.constant(T.i32(), WARP_SIZE)
        lane_i32 = arith_ops.RemUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
        wave_i32 = arith_ops.DivUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result

        # Layout for LDS scratch.
        c_num_waves = flir.const_index(RED_SLOTS)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(unwrap(c_num_waves))
        stride_red = flir.make_stride(unwrap(c1))
        layout_red = flir.make_layout(shape_red, stride_red)

        # Intra-wave reduce both values independently.
        w0 = _wave_reduce_add(val0_f32)
        w1 = _wave_reduce_add(val1_f32)

        # lane0 writes per-wave partials into LDS for both sums.
        is_lane0 = arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            unwrap(lane_i32),
            unwrap(arith.constant(T.i32(), 0)),
        ).result
        with flir.scf_ext.if_(is_lane0) as then_blk:
            with ir.InsertionPoint(then_blk):
                wave_idx = arith_ops.IndexCastOp(T.index(), unwrap(wave_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(wave_idx)), layout_red)
                scratch0_tv[unwrap(red_idx)] = unwrap(w0)
                scratch1_tv[unwrap(red_idx)] = unwrap(w1)
        gpu.barrier()

        # wave0 loads NUM_WAVES partials for both, reduces each with shuffle, writes scratch[0].
        is_wave0 = arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            unwrap(wave_i32),
            unwrap(arith.constant(T.i32(), 0)),
        ).result
        with flir.scf_ext.if_(is_wave0) as then_blk:
            with ir.InsertionPoint(then_blk):
                in_range = arith_ops.CmpIOp(
                    arith_ops.CmpIPredicate.ult,
                    unwrap(lane_i32),
                    unwrap(arith.constant(T.i32(), RED_SLOTS)),
                ).result

                c0_i32 = arith.constant(T.i32(), 0)
                lane_safe_i32 = flir.arith.SelectOp(unwrap(in_range), unwrap(lane_i32), unwrap(c0_i32)).result
                lane_safe_idx = arith_ops.IndexCastOp(T.index(), unwrap(lane_safe_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(lane_safe_idx)), layout_red)
                v0 = scratch0_tv[unwrap(red_idx)]
                v1 = scratch1_tv[unwrap(red_idx)]
                z = arith.constant(T.f32(), 0.0).value
                ww0 = flir.arith.SelectOp(unwrap(in_range), unwrap(v0), unwrap(z)).result
                ww1 = flir.arith.SelectOp(unwrap(in_range), unwrap(v1), unwrap(z)).result

            ww0 = _wave_reduce_add(ww0)
            ww1 = _wave_reduce_add(ww1)

            is_lane0_2 = arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.eq,
                unwrap(lane_i32),
                unwrap(arith.constant(T.i32(), 0)),
            ).result
            with flir.scf_ext.if_(is_lane0_2) as then2:
                with ir.InsertionPoint(then2):
                    red_idx0 = flir.crd2idx(flir.make_coord(unwrap(zero_idx)), layout_red)
                    scratch0_tv[unwrap(red_idx0)] = unwrap(ww0)
                    scratch1_tv[unwrap(red_idx0)] = unwrap(ww1)

        gpu.barrier()
        red_idx0 = flir.crd2idx(flir.make_coord(unwrap(zero_idx)), layout_red)
        return scratch0_tv[unwrap(red_idx0)], scratch1_tv[unwrap(red_idx0)]

    return block_reduce_add2
