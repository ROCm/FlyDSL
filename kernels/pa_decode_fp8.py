"""PA Decode FP8 — FlyDSL, matching Triton Gluon LLIR structure.

Supports kv_block_size=16 (original) and kv_block_size=1024 (trans_v required).
"""
from __future__ import annotations
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

QUERY_GROUP_SIZE = 16
HEAD_SIZE        = 128
KV_BLOCK_SIZE    = 16
KV_COMPUTE_BLOCK = 256
NUM_WARPS        = 4
WARP_SIZE        = 64
BLOCK_THREADS    = NUM_WARPS * WARP_SIZE
MFMA_M = MFMA_N  = 16
MFMA_K           = 32
QK_N_TILES_WARP  = (KV_COMPUTE_BLOCK // NUM_WARPS) // MFMA_N  # 4
PV_K_STEPS       = KV_COMPUTE_BLOCK // MFMA_K                  # 8
PV_N_TILES_WARP  = (HEAD_SIZE // NUM_WARPS) // MFMA_N          # 2

Q_LDS_BYTES      = BLOCK_THREADS * 8
PROB_LDS_BYTES   = BLOCK_THREADS * 16
BT_LDS_BYTES     = NUM_WARPS * 16
RED_SLOTS        = NUM_WARPS
FP8_MAX          = 240.0
LOG2E            = 1.4426950408889634

def _i32_const(val):
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects._arith_ops_gen import ConstantOp
    i32 = _ir.IntegerType.get_signless(32)
    return ConstantOp(_ir.IntegerAttr.get(i32, int(val))).result

def _vsplat_mul(vec, scalar):
    return vec * vector.broadcast(T.f32x4, scalar)

def _raw_global_load_2xi64(tensor_val, byte_off_i32):
    """Load <2xi64> via ptr<1> → generates global_load_dwordx4."""
    from flydsl._mlir.dialects import fly as _fly_d, llvm as _llvm_d
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects import arith as _a
    from flydsl.expr.buffer_ops import _unwrap_value
    raw = _unwrap_value(tensor_val)
    ptr_generic = _ir.Type.parse('!llvm.ptr')
    ptr1_type   = _ir.Type.parse('!llvm.ptr<1>')
    base_generic = _fly_d.extract_aligned_pointer_as_index(ptr_generic, raw)
    base_ptr1 = _llvm_d.AddrSpaceCastOp(ptr1_type, base_generic).result
    i64t = _ir.IntegerType.get_signless(64)
    i8t  = _ir.IntegerType.get_signless(8)
    off64 = _a.extsi(i64t, arith.unwrap(byte_off_i32))
    gep = _llvm_d.GEPOp(ptr1_type, base_ptr1, [off64], [-2147483648], i8t, 0)
    vec2i64 = _ir.VectorType.get([2], i64t)
    return _llvm_d.LoadOp(vec2i64, gep.result, alignment=16).result

def _ex(v2i64, k):
    """Extract element k from <2xi64> LLVM value."""
    from flydsl._mlir.dialects import llvm as _llvm_d
    return _llvm_d.ExtractElementOp(v2i64, _i32_const(k)).result

allocator = None

def build_pa_decode_module(
    num_seqs, num_kv_heads, num_partitions,
    max_blocks_per_seq=256,
    softmax_scale=None, query_scale=1.0, key_scale=1.0, value_scale=1.0,
    kv_block_size=16, trans_v=False,
    one_shot=False, ps_num_splits=0,
):
    global allocator
    arch = get_hip_arch()
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _qk_scale   = float(softmax_scale * query_scale * key_scale)
    _prob_scale = float(value_scale / FP8_MAX)

    _bs = kv_block_size
    _num_heads       = num_kv_heads * QUERY_GROUP_SIZE
    _stride_q_seq    = _num_heads * HEAD_SIZE
    _stride_q_head   = HEAD_SIZE
    _stride_k_block  = num_kv_heads * (HEAD_SIZE // 16) * _bs * 16
    _stride_k_head   = (HEAD_SIZE // 16) * _bs * 16
    _stride_bt_seq   = max_blocks_per_seq

    if trans_v:
        _stride_v_block = num_kv_heads * (_bs // 16) * HEAD_SIZE * 16
        _stride_v_head  = (_bs // 16) * HEAD_SIZE * 16
    else:
        _stride_v_block = num_kv_heads * HEAD_SIZE * _bs
        _stride_v_head  = HEAD_SIZE * _bs

    _stride_out_part = QUERY_GROUP_SIZE * HEAD_SIZE
    _stride_out_head = num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE
    _stride_out_seq  = num_kv_heads * num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE
    _stride_es_seq   = num_kv_heads * num_partitions * QUERY_GROUP_SIZE
    _stride_ml_seq   = num_kv_heads * num_partitions * QUERY_GROUP_SIZE

    _use_large_block = (_bs > KV_BLOCK_SIZE)
    _partitions_per_block = _bs // KV_COMPUTE_BLOCK if _use_large_block else 1
    _blocks_per_partition = KV_COMPUTE_BLOCK // _bs if not _use_large_block else 1

    import math as _math
    _ps_mode = ps_num_splits > 0
    _max_pps = _math.ceil(num_partitions / ps_num_splits) if _ps_mode else 1

    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_smem")
    q_off    = 0;            allocator.ptr = Q_LDS_BYTES
    prob_off = Q_LDS_BYTES;  allocator.ptr += PROB_LDS_BYTES
    bt_off   = prob_off + PROB_LDS_BYTES; allocator.ptr += BT_LDS_BYTES
    rmax_off = bt_off + BT_LDS_BYTES;    allocator.ptr += RED_SLOTS * 4
    rsum_off = rmax_off + RED_SLOTS * 4; allocator.ptr += RED_SLOTS * 4

    @flyc.kernel
    def pa_decode_dot_kernel(
        out_ptr: fx.Tensor, exp_sums_ptr: fx.Tensor, max_logits_ptr: fx.Tensor,
        query_ptr: fx.Tensor, key_cache_ptr: fx.Tensor, value_cache_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        context_length_i32: Int32,
    ):
        tid  = arith.index_cast(T.i32, gpu.thread_idx.x)
        seq  = arith.index_cast(T.i32, gpu.block_idx.x)
        kv_h = arith.index_cast(T.i32, gpu.block_idx.y)
        part = arith.index_cast(T.i32, gpu.block_idx.z)

        mfma_row    = tid & _i32_const(15)
        lane_hi4    = (tid & _i32_const(0xF0)) >> _i32_const(4)
        warp_id     = tid >> _i32_const(6)
        kv_col_bits = tid & _i32_const(48)
        lane_iw     = tid % _i32_const(WARP_SIZE)
        c8    = _i32_const(8)
        c112  = _i32_const(112)
        c_w   = _i32_const(WARP_SIZE)

        q_rsrc  = buffer_ops.create_buffer_resource(query_ptr,        max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr,         max_size=True)
        es_rsrc  = buffer_ops.create_buffer_resource(exp_sums_ptr,    max_size=True)
        ml_rsrc  = buffer_ops.create_buffer_resource(max_logits_ptr,  max_size=True)

        base       = allocator.get_base()
        q_lds_i32  = SmemPtr(base, q_off,    T.i32, shape=(Q_LDS_BYTES    // 4,)).get()
        q_lds_i64  = SmemPtr(base, q_off,    T.i64, shape=(Q_LDS_BYTES    // 8,)).get()
        p_lds_i32  = SmemPtr(base, prob_off, T.i32, shape=(PROB_LDS_BYTES // 4,)).get()
        bt_lds_i64 = SmemPtr(base, bt_off,   T.i64, shape=(BT_LDS_BYTES   // 8,)).get()
        s_max_p    = SmemPtr(base, rmax_off, T.f32, shape=(RED_SLOTS,))
        s_sum_p    = SmemPtr(base, rsum_off, T.f32, shape=(RED_SLOTS,))

        c_kb = arith.constant(_stride_k_block, type=T.i32)
        c_kh = arith.constant(_stride_k_head,  type=T.i32)
        c_vb = arith.constant(_stride_v_block, type=T.i32)
        c_vh = arith.constant(_stride_v_head,  type=T.i32)
        c_sq = arith.constant(_stride_q_seq,   type=T.i32)
        c_qh = arith.constant(_stride_q_head,  type=T.i32)
        c_bt = arith.constant(_stride_bt_seq,  type=T.i32)
        wave_idx = arith.index_cast(T.index, arith.unwrap(warp_id))

        _q_cta_base = seq * c_sq + kv_h * _i32_const(QUERY_GROUP_SIZE) * c_qh
        _k_head_off  = kv_h * c_kh
        _v_head_off  = kv_h * c_vh

        part_z = arith.index_cast(T.i32, gpu.block_idx.z)

        # ── STEP 1: Q → LDS ────────────────────────────────────
        q_off_g = _q_cta_base + mfma_row * c_qh + lane_hi4 * c8
        q_vec   = buffer_ops.buffer_load(q_rsrc, q_off_g // _i32_const(4), vec_width=2, dtype=T.i32)
        swiz    = (tid * c8) ^ (tid & c112)
        vector.store(q_vec, q_lds_i32, [arith.index_cast(T.index, swiz // _i32_const(4))])

        # ── STEP 4: barrier for Q LDS ──────────────────────────
        gpu.barrier()

        # ── STEP 5: Q from LDS ─────────────────────────────────
        _q_col = ((tid * _i32_const(16)) & c112) ^ kv_col_bits
        _q_b0  = (mfma_row * _i32_const(HEAD_SIZE)) | _q_col
        _q_b1  = _q_b0 ^ _i32_const(64)
        q_v0 = vector.load_op(T.vec(2, T.i64), q_lds_i64, [arith.index_cast(T.index, _q_b0 // c8)])
        q_v1 = vector.load_op(T.vec(2, T.i64), q_lds_i64, [arith.index_cast(T.index, _q_b1 // c8)])

        q_a0 = vector.extract(q_v0, static_position=[0], dynamic_position=[])
        q_a1 = vector.extract(q_v0, static_position=[1], dynamic_position=[])
        q_a2 = vector.extract(q_v1, static_position=[0], dynamic_position=[])
        q_a3 = vector.extract(q_v1, static_position=[1], dynamic_position=[])

        NEG_INF  = arith.constant(float("-inf"), type=T.f32)
        ZERO_F   = arith.constant(0.0, type=T.f32)
        LOG2E_C  = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)
        F240     = arith.constant(FP8_MAX, type=T.f32)
        PROB_SCALE_C = arith.constant(_prob_scale, type=T.f32)
        warp_head_base = warp_id * _i32_const(32)

        from flydsl._mlir.dialects import arith as _a, gpu as _g
        from flydsl._mlir.dialects import arith as _arith_bt
        from flydsl._mlir import ir as _ir_bt
        _i64bt = _ir_bt.IntegerType.get_signless(64)
        _i32bt = _ir_bt.IntegerType.get_signless(32)

        def _wave_max(x):
            w = x
            for sh in [32,16,8,4,2,1]:
                peer = _g.ShuffleOp(arith.unwrap(w), arith.constant(sh, type=T.i32),
                                    arith.unwrap(c_w), _g.ShuffleMode.XOR).shuffleResult
                w = w.maximumf(peer)
            return w
        def _wave_add(x):
            w = x
            for sh in [32,16,8,4,2,1]:
                peer = _g.ShuffleOp(arith.unwrap(w), arith.constant(sh, type=T.i32),
                                    arith.unwrap(c_w), _g.ShuffleMode.XOR).shuffleResult
                w = w + peer
            return w

        _mi0 = arith.index_cast(T.index, _i32_const(0))
        _mi1 = arith.index_cast(T.index, _i32_const(1))
        _mi2 = arith.index_cast(T.index, _i32_const(2))
        _mi3 = arith.index_cast(T.index, _i32_const(3))

        # ── Partition loop ──────────────────────────────────────
        # PS: scf.for over _max_pps iterations per split
        # Non-PS: scf.for with 1 iteration (no overhead)
        for _pi in range(int(_max_pps)):
            if _ps_mode:
                _pi_i32 = arith.index_cast(T.i32, _pi)
                part = part_z * _i32_const(int(_max_pps)) + _pi_i32
            else:
                part = part_z

            if _use_large_block:
                bt_idx = part // _i32_const(_partitions_per_block)
                page_off = (part % _i32_const(_partitions_per_block)) * _i32_const(KV_COMPUTE_BLOCK)
                partition_start = part * _i32_const(KV_COMPUTE_BLOCK)
                _bt_seq_base = seq * c_bt + bt_idx
            else:
                bt_start = part * _i32_const(_blocks_per_partition)
                partition_start = part * _i32_const(KV_COMPUTE_BLOCK)
                _bt_seq_base = seq * c_bt + bt_start

            # ── STEP 2: BT loads ───────────────────────────────
            if _use_large_block:
                phys_block = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base, vec_width=1, dtype=T.i32)
                phys_list = [phys_block, phys_block, phys_block, phys_block]
            else:
                phys_0 = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base+warp_id,              vec_width=1, dtype=T.i32)
                phys_1 = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base+warp_id+_i32_const(4),  vec_width=1, dtype=T.i32)
                phys_2 = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base+warp_id+_i32_const(8),  vec_width=1, dtype=T.i32)
                phys_3 = buffer_ops.buffer_load(bt_rsrc, _bt_seq_base+warp_id+_i32_const(12), vec_width=1, dtype=T.i32)
                phys_list = [phys_0, phys_1, phys_2, phys_3]

            # ── STEP 3: K batch loads ──────────────────────────
            kv = []
            for n_tile in [0, 1, 2, 3]:
                pb = phys_list[n_tile]
                _k_blk_base = pb * c_kb + _k_head_off
                if _use_large_block:
                    tok_in_blk = page_off + warp_id * _i32_const(64) + _i32_const(n_tile * 16) + mfma_row
                    kb0 = _k_blk_base + tok_in_blk * _i32_const(16)
                    kb1 = _k_blk_base + _i32_const(2 * _bs * 16) + tok_in_blk * _i32_const(16)
                else:
                    kb0 = _k_blk_base + mfma_row * _i32_const(16)
                    kb1 = _k_blk_base + _i32_const(2*KV_BLOCK_SIZE*16) + mfma_row * _i32_const(16)
                kv.append([_raw_global_load_2xi64(key_cache_ptr, kb0),
                           _raw_global_load_2xi64(key_cache_ptr, kb1)])

            k00 = _ex(kv[0][0], 0); k01 = _ex(kv[0][0], 1)
            k02 = _ex(kv[0][1], 0); k03 = _ex(kv[0][1], 1)
            k10 = _ex(kv[1][0], 0); k11 = _ex(kv[1][0], 1)
            k12 = _ex(kv[1][1], 0); k13 = _ex(kv[1][1], 1)
            k20 = _ex(kv[2][0], 0); k21 = _ex(kv[2][0], 1)
            k22 = _ex(kv[2][1], 0); k23 = _ex(kv[2][1], 1)
            k30 = _ex(kv[3][0], 0); k31 = _ex(kv[3][0], 1)
            k32 = _ex(kv[3][1], 0); k33 = _ex(kv[3][1], 1)

            # ── STEP 6: QK MFMAs ──────────────────────────────
            rocdl.sched_barrier(0)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(16)
            rocdl.sched_barrier(0)

            zero = arith.constant_vector(0.0, T.f32x4)
            rocdl.s_waitcnt(0x0F77)
            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k00, q_a0, zero, 0, 0, 0])
            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k01, q_a1, acc0, 0, 0, 0])
            rocdl.s_waitcnt(0x0F76)
            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k02, q_a2, acc0, 0, 0, 0])
            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k03, q_a3, acc0, 0, 0, 0])
            rocdl.s_waitcnt(0x0F75)
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k10, q_a0, zero, 0, 0, 0])
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k11, q_a1, acc1, 0, 0, 0])
            rocdl.s_waitcnt(0x0F74)
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k12, q_a2, acc1, 0, 0, 0])
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k13, q_a3, acc1, 0, 0, 0])
            rocdl.s_waitcnt(0x0F73)
            acc2 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k20, q_a0, zero, 0, 0, 0])
            acc2 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k21, q_a1, acc2, 0, 0, 0])
            rocdl.s_waitcnt(0x0F72)
            acc2 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k22, q_a2, acc2, 0, 0, 0])
            acc2 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k23, q_a3, acc2, 0, 0, 0])
            rocdl.s_waitcnt(0x0F71)
            acc3 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k30, q_a0, zero, 0, 0, 0])
            acc3 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k31, q_a1, acc3, 0, 0, 0])
            rocdl.s_waitcnt(0x0F70)
            acc3 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k32, q_a2, acc3, 0, 0, 0])
            acc3 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k33, q_a3, acc3, 0, 0, 0])
            acc_qk = [acc0, acc1, acc2, acc3]

            ctx_len = context_length_i32
            for n_tile in [0, 1, 2, 3]:
                acc_qk[n_tile] = _vsplat_mul(acc_qk[n_tile], QK_SCALE)
                for elem in [0, 1, 2, 3]:
                    kv_tok = partition_start + warp_id * _i32_const(64) + _i32_const(n_tile*16+elem)
                    in_b = kv_tok < ctx_len
                    v = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    acc_qk[n_tile] = vector.insert(
                        arith.select(in_b, v, NEG_INF),
                        acc_qk[n_tile], static_position=[elem], dynamic_position=[])

            # ── STEP 7: BT LDS staging (for V loads) ──────────
            if _use_large_block:
                token_page_base = page_off // _i32_const(16)
                tp0 = token_page_base + warp_id
                tp1 = token_page_base + warp_id + _i32_const(4)
                tp0_i64 = _arith_bt.extsi(_i64bt, arith.unwrap(tp0))
                tp1_i64 = _arith_bt.extsi(_i64bt, arith.unwrap(tp1))
                bt_si  = arith.index_cast(T.index, warp_id * _i32_const(2))
                bt_vec = vector.from_elements(T.vec(2, T.i64), [tp0_i64, tp1_i64])
                vector.store(bt_vec, bt_lds_i64, [bt_si])
                gpu.barrier()
                bt_li   = arith.index_cast(T.index, kv_col_bits // _i32_const(8))
                bt_load = vector.load_op(T.vec(2, T.i64), bt_lds_i64, [bt_li])
                phys_pv_0 = _arith_bt.trunci(_i32bt, vector.extract(bt_load, static_position=[0], dynamic_position=[]))
                phys_pv_1 = _arith_bt.trunci(_i32bt, vector.extract(bt_load, static_position=[1], dynamic_position=[]))
            else:
                gpu.barrier()
                p0_i64 = _arith_bt.extsi(_i64bt, arith.unwrap(phys_0))
                p1_i64 = _arith_bt.extsi(_i64bt, arith.unwrap(phys_1))
                bt_si  = arith.index_cast(T.index, warp_id * _i32_const(2))
                bt_vec = vector.from_elements(T.vec(2, T.i64), [p0_i64, p1_i64])
                vector.store(bt_vec, bt_lds_i64, [bt_si])
                gpu.barrier()
                bt_li   = arith.index_cast(T.index, kv_col_bits // _i32_const(8))
                bt_load = vector.load_op(T.vec(2, T.i64), bt_lds_i64, [bt_li])
                phys_pv_0 = _arith_bt.trunci(_i32bt, vector.extract(bt_load, static_position=[0], dynamic_position=[]))
                phys_pv_1 = _arith_bt.trunci(_i32bt, vector.extract(bt_load, static_position=[1], dynamic_position=[]))

            # ── STEP 8: V batch loads ──────────────────────────
            vv = []
            for n_tile in [0, 1]:
                h_py  = n_tile * MFMA_N
                pv_pb = phys_pv_0 if n_tile == 0 else phys_pv_1
                if _use_large_block and trans_v:
                    _v_blk_base = arith.unwrap(phys_block) * c_vb + _v_head_off \
                        + pv_pb * _i32_const(HEAD_SIZE * 16) \
                        + _i32_const(h_py * 16)
                elif _use_large_block:
                    _v_blk_base = pv_pb * c_vb + _v_head_off + _i32_const(h_py * _bs) + page_off
                else:
                    _v_blk_base = pv_pb * c_vb + _v_head_off + _i32_const(h_py * KV_BLOCK_SIZE)
                nt_loads = []
                for load_i in [0, 1, 2, 3]:
                    v_off = _v_blk_base + _i32_const(load_i * 32)
                    nt_loads.append(_raw_global_load_2xi64(value_cache_ptr, v_off))
                vv.append(nt_loads)

            # ── STEP 9: Softmax ────────────────────────────────
            local_max = NEG_INF
            for n_tile in [0,1,2,3]:
                local_max = local_max.maximumf(vector.reduction(T.f32, "maxnumf", acc_qk[n_tile]))
            wmax = _wave_max(local_max)
            s_max_p.store(wmax, [wave_idx])
            gpu.barrier()
            global_max = s_max_p.load([_mi0]).maximumf(s_max_p.load([_mi1])) \
                         .maximumf(s_max_p.load([_mi2])).maximumf(s_max_p.load([_mi3]))

            acc_pv_init = [arith.constant_vector(0.0, T.f32x4) for _ in [0, 1]]
            acc_scale = ((NEG_INF - global_max) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
            acc_pv = [_vsplat_mul(t, acc_scale) for t in acc_pv_init]

            local_sum = ZERO_F
            for n_tile in [0,1,2,3]:
                for elem in [0,1,2,3]:
                    s = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    p = ((s - global_max) * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    local_sum = local_sum + p
                    acc_qk[n_tile] = vector.insert(p, acc_qk[n_tile],
                                                   static_position=[elem], dynamic_position=[])
            wsum = _wave_add(local_sum)
            s_sum_p.store(wsum, [wave_idx])
            gpu.barrier()
            global_sum = (s_sum_p.load([_mi0]) + s_sum_p.load([_mi1])
                          + s_sum_p.load([_mi2]) + s_sum_p.load([_mi3]))

            # ── STEP 10: FP8 pack + prob → LDS ────────────────
            probs = []
            for n_tile in [0,1,2,3]:
                for elem in [0,1,2,3]:
                    pf = vector.extract(acc_qk[n_tile], static_position=[elem], dynamic_position=[])
                    probs.append(pf * F240)

            fp8_i32 = []
            for i in [0,1,2,3]:
                lo = rocdl.cvt_pk_fp8_f32(T.i32, probs[i*4],   probs[i*4+1], _i32_const(0), False)
                wd = rocdl.cvt_pk_fp8_f32(T.i32, probs[i*4+2], probs[i*4+3], lo,            True)
                fp8_i32.append(wd)

            gpu.barrier()
            prob_vec4 = vector.from_elements(T.vec(4, T.i32), fp8_i32)
            vector.store(prob_vec4, p_lds_i32, [arith.index_cast(T.index, tid * _i32_const(4))])
            gpu.barrier()

            # ── STEP 11: P from LDS → 8 i64 ──────────────────
            _prob_base = kv_col_bits * _i32_const(64) + mfma_row * _i32_const(16)
            p_lds_i32b = SmemPtr(base, prob_off, T.i32, shape=(PROB_LDS_BYTES//4,)).get()

            def _load_p4i32(byte_off):
                idx = arith.index_cast(T.index, (_prob_base + _i32_const(byte_off)) // _i32_const(4))
                return vector.load_op(T.vec(4, T.i32), p_lds_i32b, [idx])
            _pa = _load_p4i32(0)
            _pb = _load_p4i32(256)
            _pc = _load_p4i32(512)
            _pd = _load_p4i32(768)

            from flydsl._mlir.dialects import llvm as _llvm_p
            from flydsl._mlir import ir as _ir_p
            _i64p    = _ir_p.IntegerType.get_signless(64)
            _vec2i32 = _ir_p.VectorType.get([2], _ir_p.IntegerType.get_signless(32))

            def _pack(vec_a, vec_b, k):
                a = _llvm_p.ExtractElementOp(vec_a, _i32_const(k)).result
                b = _llvm_p.ExtractElementOp(vec_b, _i32_const(k)).result
                undef = _llvm_p.mlir_undef(_vec2i32)
                v = _llvm_p.InsertElementOp(undef, a, _i32_const(0)).result
                v = _llvm_p.InsertElementOp(v,    b, _i32_const(1)).result
                return _llvm_p.BitcastOp(_i64p, v).result

            p_ops = [_pack(_pa, _pb, k) for k in [0,1,2,3]] + [_pack(_pc, _pd, k) for k in [0,1,2,3]]

            # ── STEP 12: V extracts ────────────────────────────
            v00=_ex(vv[0][0],0); v01=_ex(vv[0][0],1)
            v02=_ex(vv[0][1],0); v03=_ex(vv[0][1],1)
            v04=_ex(vv[0][2],0); v05=_ex(vv[0][2],1)
            v06=_ex(vv[0][3],0); v07=_ex(vv[0][3],1)
            v10=_ex(vv[1][0],0); v11=_ex(vv[1][0],1)
            v12=_ex(vv[1][1],0); v13=_ex(vv[1][1],1)
            v14=_ex(vv[1][2],0); v15=_ex(vv[1][2],1)
            v16=_ex(vv[1][3],0); v17=_ex(vv[1][3],1)

            # ── STEP 13: PV MFMAs ─────────────────────────────
            rocdl.sched_barrier(0)
            rocdl.sched_dsrd(4)
            rocdl.sched_mfma(16)
            rocdl.sched_barrier(0)

            rocdl.s_waitcnt(0x0F77)
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v00, p_ops[0], acc_pv[0], 0, 0, 0])
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v01, p_ops[1], pv0,        0, 0, 0])
            rocdl.s_waitcnt(0x0F76)
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v02, p_ops[2], pv0,        0, 0, 0])
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v03, p_ops[3], pv0,        0, 0, 0])
            rocdl.s_waitcnt(0x0F75)
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v04, p_ops[4], pv0,        0, 0, 0])
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v05, p_ops[5], pv0,        0, 0, 0])
            rocdl.s_waitcnt(0x0F74)
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v06, p_ops[6], pv0,        0, 0, 0])
            pv0 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v07, p_ops[7], pv0,        0, 0, 0])
            rocdl.s_waitcnt(0x0F73)
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v10, p_ops[0], acc_pv[1], 0, 0, 0])
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v11, p_ops[1], pv1,        0, 0, 0])
            rocdl.s_waitcnt(0x0F72)
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v12, p_ops[2], pv1,        0, 0, 0])
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v13, p_ops[3], pv1,        0, 0, 0])
            rocdl.s_waitcnt(0x0F71)
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v14, p_ops[4], pv1,        0, 0, 0])
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v15, p_ops[5], pv1,        0, 0, 0])
            rocdl.s_waitcnt(0x0F70)
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v16, p_ops[6], pv1,        0, 0, 0])
            pv1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v17, p_ops[7], pv1,        0, 0, 0])

            # ── STEP 14: Output ────────────────────────────────
            rcp = arith.constant(1.0, type=T.f32) / global_sum
            pv_out = [_vsplat_mul(_vsplat_mul(pv0, PROB_SCALE_C), rcp),
                      _vsplat_mul(_vsplat_mul(pv1, PROB_SCALE_C), rcp)]

            if not one_shot:
                c_np_qg = arith.constant(num_partitions * QUERY_GROUP_SIZE, type=T.i32)
                c_qg    = arith.constant(QUERY_GROUP_SIZE, type=T.i32)
                ml_off  = seq * arith.constant(_stride_ml_seq, type=T.i32) + kv_h*c_np_qg + part*c_qg + mfma_row
                es_off  = seq * arith.constant(_stride_es_seq, type=T.i32) + kv_h*c_np_qg + part*c_qg + mfma_row
                buffer_ops.buffer_store(global_max, ml_rsrc, ml_off)
                buffer_ops.buffer_store(global_sum, es_rsrc, es_off)

            c_os = arith.constant(_stride_out_seq,  type=T.i32)
            c_oh = arith.constant(_stride_out_head, type=T.i32)
            c_op = arith.constant(_stride_out_part, type=T.i32)
            for n_tile in [0, 1]:
                h_py = n_tile * MFMA_N
                out_off = (seq*c_os + kv_h*c_oh + part*c_op
                           + mfma_row*_i32_const(HEAD_SIZE)
                           + warp_head_base + _i32_const(h_py))
                out_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_out[n_tile])
                out_i32  = vector.bitcast(T.vec(2, T.i32), out_bf16)
                buffer_ops.buffer_store(out_i32, out_rsrc, out_off * _i32_const(2), offset_is_bytes=True)

    return pa_decode_dot_kernel
