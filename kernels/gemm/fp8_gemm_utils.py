# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops as _buffer_ops

# ceildiv is the canonical cdiv from the shared layer; re-exported here for the
# gemm kernels that historically imported it from this module.
from kernels.common.utils import cdiv as ceildiv  # noqa: F401


def divmod(a, b):
    """Integer divmod that works on DSL values (e.g. ``Int32``).

    The builtin ``divmod`` rejects DSL scalar types, so this uses the overloaded
    ``//`` / ``%`` operators to emit the corresponding ops.
    """
    return (a // b, a % b)


def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` for ``b_preshuffled=True``."""
    n, k = b_t.shape[-2:]
    assert n % 16 == 0 and k % 64 == 0, f"need N%16==0 and K%64==0, got N={n} K={k}"
    return b_t.reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16) + (row % 16) * 16 + (col // 64) * 1024 + ((col % 64) // 16) * 256 + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)


class StoreC:
    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        # Exact byte counts from compile-time shape (BF16 C output, FP32 scales).
        # ``num_records_bytes`` is required when ``max_size=False`` -- see
        # ``make_buffer_tensor`` docstring for the silent-OOB rationale.
        c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
        sa_nbytes = c_rows * 4  # Float32 row-wise scale
        sb_nbytes = c_cols * 4  # Float32 col-wise scale
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=sa_nbytes)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=sb_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _load_scale_vec4(self, row):
        fx.copy(self.scale_atom_4, fx.slice(self.sa_div, (None, fx.Int32(row))), self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _load_scale_scalar(self, col):
        fx.copy(self.scale_atom_1, fx.slice(self.sb_div, (None, fx.Int32(col))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        a_scales = [
            self._load_scale_vec4(base_row + i * 16 + (self.lane_id // 16) * 4) for i in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_scale_scalar(base_col + i * 16 + self.lane_id % 16) for i in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(scaled, col_valid.select(c_index, oob))


def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


class Mfma16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _do_mma(self, a, b, c):
        a_frag = self._make_operand_frag(a)
        b_frag = self._make_operand_frag(b)
        c_frag = self._make_accum_frag(c)
        fx.gemm(self.atom, c_frag, a_frag, b_frag, c_frag)
        return c_frag.load().ir_value()

    def call(self, a, b, c, *, set_prio=True):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [self._make_operand_frag(a[idx]) for idx in range_constexpr(self.n_tiles_a)]
        b_frags = [self._make_operand_frag(b[idx]) for idx in range_constexpr(self.n_tiles_b)]
        c_frags = [self._make_accum_frag(c[idx]) for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(self.atom, cf, a_frags[i], b_frags[j], cf)
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [c_frags[idx].load().ir_value() for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)]

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b

        return self._do_mma(a[i], b[j], c[self.idx(i, j)])


# ═══════════════════════════════════════════════════════════════════════════
# MXFP8 (per-1x32 E8M0 block-scaled) dense-GEMM primitives.
# ═══════════════════════════════════════════════════════════════════════════


def _readfirstlane_i64(v):
    """Force a wave-uniform-in-value i64 into SGPRs via s_readfirstlane."""
    raw = _raw(v)
    r = rocdl.readfirstlane(res=raw.type, src=raw)
    rv = r.result if hasattr(r, "result") else r
    return fx.Int64(rv)


def make_fp8_buffer_tensor_rebased(arg_i8, fp8_ir_t, base_elems, num_records_bytes):
    base = fx.Int64(_buffer_ops.extract_base_index(arg_i8))
    # Pin the wave-uniform shifted base + num_records to SGPRs: the group-scan base reads
    # as VGPR -> VGPR SRD -> readfirstlane waterfall per K-loop load. Pin keeps it scalar.
    base = _readfirstlane_i64(base + base_elems)
    nr = fx.min(num_records_bytes, fx.Int64(0xFFFFFFFF))
    nrec = _readfirstlane_i64(nr)
    flags = _buffer_ops._get_buffer_flags()
    # global int8 ptr at the shifted addr -> int8 BufferDesc fat ptr -> recast fp8.
    base_ptr = fx.inttoptr(fx.PointerType.get(elem_ty=T.i8, address_space=1, alignment=16), base)
    i8_buf_ty = fx.PointerType.get(elem_ty=T.i8, address_space=TargetAddressSpace.BufferDesc, alignment=16)
    buf_ptr = fx.make_ptr(i8_buf_ty, [base_ptr, fx.Int16(0).ir_value(), nrec.ir_value(), fx.Int32(flags).ir_value()])
    lay = fx.make_layout(0x40000000, 1)  # 1D flat; HW bounds via num_records
    iter_i8 = fx.get_iter(fx.make_view(buf_ptr, lay))
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, lay))


class MfmaScale16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b, cbsz=0, blgp=0):
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.opsel = 0
        a_ty = fx.Float8E5M2 if cbsz == 1 else fx.Float8E4M3FN
        b_ty = fx.Float8E5M2 if blgp == 1 else fx.Float8E4M3FN
        self.atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, a_ty, b_ty, opsel_a=self.opsel, opsel_b=self.opsel)
        )

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c, sa, sb):
        # fx.gemm scaled-MMA: VGPR operand/accum frags + scale_a/scale_b (1xi32 each).
        a_frag = fx.make_rmem_tensor(8, fx.Int32)
        a_frag.store(Vec(a))
        b_frag = fx.make_rmem_tensor(8, fx.Int32)
        b_frag.store(Vec(b))
        c_frag = fx.make_rmem_tensor(4, fx.Float32)
        c_frag.store(Vec(c))
        fx.gemm(
            self.atom,
            c_frag,
            a_frag,
            b_frag,
            c_frag,
            scale_a=Vec.from_elements([fx.Int32(sa)], fx.Int32),
            scale_b=Vec.from_elements([fx.Int32(sb)], fx.Int32),
        )
        return c_frag.load().ir_value()

    def call(self, a, b, c, sa, sb):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        assert len(sa) == self.n_tiles_a
        assert len(sb) == self.n_tiles_b

        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])
        return c


class ScaleBComb:
    def __init__(self, sp_tensor, dim, K):
        self.K128 = K // 128  # number of K-groups (one i32 per K-iter)
        self.lane = fx.thread_idx.x % 64
        # grp = (col//256)*4 + wn is block-strided, so the buffer holds cdiv(dim,256)*4
        # groups. A partial last 256-block reads only its valid wn groups; OOB-col reads
        # clamp to 0 and StoreC drops them. dim%256==0 -> cdiv(dim,256)*4 == dim//64.
        nbytes = ((dim + 255) // 256) * 4 * self.K128 * 64 * 4 * 4  # int32 records
        self.rsrc = _buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        """base: sb_base0 (b0 region col base). Returns 4 i32 (b0:0,1  b1:2,3)."""
        grp = (base // 256) * 4 + (base % 256) // 32
        idx = ((grp * self.K128 + k) * 64 + self.lane) * 4
        v = Vec(_buffer_ops.buffer_load(self.rsrc, idx, vec_width=4, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(4)]


class ScaleS2R:
    """Per-lane E8M0 scale loader for v_mfma_scale_f32_16x16x128 (preshuffled)."""

    def __init__(self, sp_tensor, dim, K, n_tiles):
        self.K128 = K // 128  # number of K-groups (one i32 per K-iter)
        self.n_tiles = n_tiles
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64  # == (lane//16)*16 + lane%16
        nbytes = ceildiv(dim, self.group_span) * self.K128 * 64 * n_tiles * 4  # int32 records
        self.rsrc = _buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        grp = base // self.group_span
        idx = ((grp * self.K128 + k) * 64 + self.lane) * self.n_tiles
        v = Vec(_buffer_ops.buffer_load(self.rsrc, idx, vec_width=self.n_tiles, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(self.n_tiles)]


def make_row_band_resource(c_base, base_row, c_rows, c_cols, elem_bytes):
    elem = fx.Int64(elem_bytes)
    cols_i = fx.Int64(c_cols)
    row_i = fx.Int64(base_row)
    rows_i = fx.Int64(c_rows)
    row_c = fx.min(row_i, rows_i)
    band_base = fx.Int64(c_base) + row_c * cols_i * elem
    # cap at 0x7FFFFFFF so a masked-out buffer_store (voffset=0x7FFFFFFF) is always OOB
    nrec = fx.min((rows_i - row_c) * cols_i * elem, fx.Int64(0x7FFFFFFF))
    band_base_i64 = _readfirstlane_i64(band_base)
    nrec_pinned = _readfirstlane_i64(nrec)
    return _buffer_ops.create_buffer_resource_from_addr(band_base_i64, num_records_bytes=nrec_pinned)


class StoreCPerTensor:
    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, out_ty):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.out_ty = out_ty
        self.scaled = A_scale is not None
        self.c_base = _buffer_ops.extract_base_index(C)  # index = byte base address
        if self.scaled:
            gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=4)  # 1 fp32
            gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=4)  # 1 fp32
            self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
            self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
            self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    def _load_scalar(self, div):
        fx.copy(self.scale_atom_1, fx.slice(div, (None, fx.Int32(0))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def store(self, c_frag, base_row, base_col):
        scale = self._load_scalar(self.sa_div) * self._load_scalar(self.sb_div) if self.scaled else None
        # buffer_store row-band path (int64-safe); the band SRD is pinned to SGPRs inside.
        rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
        for ti in range_constexpr(self.n_tiles_a):
            row_local = ti * 16 + (self.lane_id // 16) * 4  # relative to base_row
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    val = (vec_f32[i] * scale if self.scaled else vec_f32[i]).to(self.out_ty)
                    off = ((row_local + i) * self.c_cols + col) * 2  # i32-small within band
                    _buffer_ops.buffer_store(val, rsrc, off, mask=col_valid, offset_is_bytes=True)


def make_value_attrs(waves_per_eu, agpr_alloc, fwg):
    """Kernel value_attrs. agpr_alloc: 0 = compiler default; N>0 = force exactly
    N AGPRs ("N,N"); -N = allow up to N ("0,N")."""
    d = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": fwg}
    if agpr_alloc != 0:
        if agpr_alloc < 0:
            alloc = f"0,{-agpr_alloc}"
        else:
            alloc = f"{agpr_alloc},{agpr_alloc}"
        d["passthrough"] = [
            ["amdgpu-agpr-alloc", alloc],
            ["amdgpu-mfma-vgpr-form", "false"],
        ]
    return d


def xcd_remap_pid(pid, total_pids, num_xcd):
    if num_xcd <= 1:
        return pid
    per_xcd = total_pids // num_xcd  # floor
    rem = total_pids - per_xcd * num_xcd
    xcd = pid % num_xcd
    local = pid // num_xcd
    offset = xcd * per_xcd + (xcd < rem).select(xcd, rem)
    return offset + local


def block_mn(pid, num_pid_m, n_blocks, GM, GN):
    """Tile-id -> (block_m, block_n), resolved at trace time. GN==0: 1D GROUP_M
    super-row swizzle (block_m inner). GN>0: 2D band — N split into width-GN bands
    with GROUP_M inside each, keeping both A and B slabs L2-resident. Bijection."""
    if GN > 0:
        band_tiles = num_pid_m * GN
        band = pid // band_tiles
        pid_in_band = pid % band_tiles
        band_n0 = band * GN
        rem_n = n_blocks - band_n0
        band_w = (rem_n < GN).select(rem_n, fx.Int32(GN))
        nig = GM * band_w
        gid = pid_in_band // nig
        pig = pid_in_band % nig
        fpm = gid * GM
        rem_m = num_pid_m - fpm
        gsm = (rem_m < GM).select(rem_m, fx.Int32(GM))
        return fpm + (pig % gsm), band_n0 + (pig // gsm)
    nig = GM * n_blocks
    gid = pid // nig
    pig = pid % nig
    fpm = gid * GM
    rem_m = num_pid_m - fpm
    gsm = (rem_m < GM).select(rem_m, fx.Int32(GM))
    return fpm + (pig % gsm), pig // gsm


_PRESHUF_KT = 16  # k-tile (rows*KT dwords staged in LDS per workgroup)


def _lds_barrier():
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


def _emit_lds_repack(is_a, grp, k0, tile, rin, rout, dim, K128, KT, tid, BLK):
    NT = 4
    TILE = 64 * KT
    NOUT = KT * 64
    assert TILE % BLK == 0 and NOUT % BLK == 0
    for i in range_constexpr(TILE // BLK):
        idx = tid + i * BLK
        rr = idx // KT
        kk = idx % KT
        gk = k0 + kk
        if is_a:
            grow = grp * 64 + rr  # A: rows grp*64 + (s*16+r)
        else:
            s = rr // 16  # B-comb: row = nblk*256 + wn*32 + OFF[s] + rinner
            off = (s % 2) * fx.Int32(16) + (s // 2) * fx.Int32(128)
            grow = (grp // 4) * 256 + (grp % 4) * 32 + off + (rr % 16)
        dw = _buffer_ops.buffer_load(rin, grow * K128 + gk, vec_width=1, dtype=T.i32, mask=(gk < K128) & (grow < dim))
        fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)).store(
            Vec.from_elements([fx.Int32(dw)], fx.Int32)
        )
    _lds_barrier()
    for j in range_constexpr(NOUT // BLK):
        ol = tid + j * BLK
        kk = ol // 64
        lane = ol % 64
        r = lane % 16
        sh = (lane // 16) * fx.Int32(8)
        gk = k0 + kk
        elems = []
        for s in range_constexpr(NT):
            so = (s * 16 + r) * KT + kk
            val = Vec(fx.make_view(fx.add_offset(tile.ptr, fx.make_int_tuple(so)), fx.make_layout(1, 1)).load())
            b = (fx.Int32(val[0]) >> sh) & fx.Int32(0xFF)
            elems.append(b | (b << fx.Int32(8)) | (b << fx.Int32(16)) | (b << fx.Int32(24)))
        vec = Vec.from_elements(elems, fx.Int32)
        _buffer_ops.buffer_store(vec.ir_value(), rout, ((grp * K128 + gk) * 64 + lane) * 4, mask=gk < K128)


def build_preshuffle_ab_kernel(K128: int, KT: int = _PRESHUF_KT, BLK: int = 256):
    """Build the fused A (layout 1) + B-comb (layout 3) scale-preshuffle @flyc.kernel."""
    TILE = 64 * KT
    n_kt = ceildiv(K128, KT)

    @fx.struct
    class Smem:
        tile: fx.Array[fx.Int32, TILE, 16]

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        m: fx.Int32,
        n: fx.Int32,
        a_blocks: fx.Int32,
        a_ngrp: fx.Int32,
        b_ngrp: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile = fx.SharedAllocator().allocate(Smem).peek().tile
        rin_a = _buffer_ops.create_buffer_resource(a_raw, max_size=False, num_records_bytes=m * K128 * 4)
        rin_b = _buffer_ops.create_buffer_resource(b_raw, max_size=False, num_records_bytes=n * K128 * 4)
        rout_a = _buffer_ops.create_buffer_resource(a_sp, max_size=False, num_records_bytes=a_ngrp * K128 * 256 * 4)
        rout_b = _buffer_ops.create_buffer_resource(b_sp, max_size=False, num_records_bytes=b_ngrp * K128 * 256 * 4)
        if bid < a_blocks:
            _emit_lds_repack(True, bid // n_kt, (bid % n_kt) * KT, tile, rin_a, rout_a, m, K128, KT, tid, BLK)
        if bid >= a_blocks:
            bb = bid - a_blocks
            _emit_lds_repack(False, bb // n_kt, (bb % n_kt) * KT, tile, rin_b, rout_b, n, K128, KT, tid, BLK)

    return kern, n_kt
