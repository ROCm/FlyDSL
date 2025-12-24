"""Shared reduction helpers for FLIR example kernels.

These helpers build MLIR ops (flir/gpu/scf/vector/etc). They are extracted from
softmax/layernorm/rmsnorm kernels to de-duplicate code without changing codegen.
"""

from pyflir.dialects.ext.python_control_flow import lower_range_for_loops as _lower_range_for_loops


def unwrap(v):
    if hasattr(v, "value"):
        return v.value
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
        if is_lane0:
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
        if is_wave0:
            in_range = flir.arith.CmpIOp(
                flir.arith.CmpIPredicate.ult,
                unwrap(lane_i32),
                unwrap(arith.constant(NUM_WAVES, type=T.i32()).value),
            ).result
            neutral = c_neg_inf if reduce_op_name == "max" else c_zero
            ww = unwrap(neutral)
            if in_range:
                lane_idx = flir.arith.IndexCastOp(T.index(), unwrap(lane_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(lane_idx)), layout_red)
                v = s_red_tv[unwrap(red_idx)]
                ww = unwrap(v)
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
            if is_lane0_2:
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
        scratch_tv = flir.make_tensor(scratch_memref, shape=(RED_SLOTS,), strides=(1,))
        tid_i32 = arith_ops.IndexCastOp(T.i32(), tid.value).result
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
        if is_lane0:
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
        if is_wave0:
            in_range = arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.ult,
                unwrap(lane_i32),
                unwrap(arith.constant(T.i32(), NUM_WAVES)),
            ).result
            ww = unwrap(arith.constant(T.f32(), 0.0).value)
            if in_range:
                lane_idx = arith_ops.IndexCastOp(T.index(), unwrap(lane_i32)).result
                red_idx = flir.crd2idx(flir.make_coord(unwrap(lane_idx)), layout_red)
                v = scratch_tv[unwrap(red_idx)]
                ww = unwrap(v)
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.constant(T.i32(), sh)
                peer = gpu.ShuffleOp(unwrap(ww), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                ww = arith_ops.AddFOp(unwrap(ww), unwrap(peer), fastmath=fm_fast).result

            is_lane0_2 = arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.eq,
                unwrap(lane_i32),
                unwrap(arith.constant(T.i32(), 0)),
            ).result
            if is_lane0_2:
                red_idx0 = flir.crd2idx(flir.make_coord(unwrap(zero_idx)), layout_red)
                scratch_tv[unwrap(red_idx0)] = unwrap(ww)

        gpu.barrier()
        red_idx0 = flir.crd2idx(flir.make_coord(unwrap(zero_idx)), layout_red)
        return scratch_tv[unwrap(red_idx0)]

    return block_reduce_add


# Apply Python control-flow lowering to these helpers so the examples can use
# plain Python `if` with MLIR `Value` predicates.
make_block_reduce = _lower_range_for_loops(make_block_reduce)
make_block_reduce_add = _lower_range_for_loops(make_block_reduce_add)

