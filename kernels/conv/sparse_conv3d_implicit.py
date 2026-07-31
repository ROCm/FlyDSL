import functools
import os
import weakref
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from kernels.common.mem_ops import buffer_atomic_add
from kernels.common.tensor_shim import GTensor, _run_compiled


@functools.lru_cache(maxsize=4)
def _compile_presence_scatter_kernel(block: int):
    BLOCK = block

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def scatter_kernel(
        out_xyz: fx.Tensor,
        presence: fx.Tensor,
        sy: fx.Int32,
        sz: fx.Int32,
        n_pts: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        blk = fx.Int32(gpu.block_id("x"))
        i = blk * fx.Int32(BLOCK) + tid

        xyz_ = GTensor(out_xyz, dtype=T.i32, shape=(-1,))
        pr_ = GTensor(presence, dtype=T.i32, shape=(-1,))

        i_ok = (i < n_pts).ir_value()
        iif = scf.IfOp(i_ok, results_=[], has_else=False)
        with ir.InsertionPoint(iif.then_block):
            base = i * fx.Int32(3)
            x = fx.Int32(xyz_.load(base))
            y = fx.Int32(xyz_.load(base + fx.Int32(1)))
            z = fx.Int32(xyz_.load(base + fx.Int32(2)))
            pr_.store((x * sy + y) * sz + z, i)
            scf.YieldOp([])

    @flyc.jit
    def launch(
        out_xyz,
        presence,
        sy,
        sz,
        n_pts,
        grid,
        stream: fx.Stream = fx.Stream(None),
    ):
        scatter_kernel(out_xyz, presence, sy, sz, n_pts).launch(grid=(grid,), block=(BLOCK,), stream=stream)

    return launch


@functools.lru_cache(maxsize=8)
def _compile_fused_tiled_kernel(block_m: int, kv_total: int):
    BLOCK_M = block_m
    KV = kv_total
    K = round(kv_total ** (1.0 / 3.0))
    assert K**3 == kv_total, f"kv_total={kv_total} is not a cube"
    R = K // 2

    @flyc.kernel(known_block_size=[BLOCK_M, 1, 1])
    def fused_tiled_kernel(
        presence: fx.Tensor,
        out_xyz: fx.Tensor,
        sx: fx.Int32,
        sy: fx.Int32,
        sz: fx.Int32,
        inp_row_lut: fx.Tensor,
        mask: fx.Tensor,
        active_kv_ids: fx.Tensor,
        active_count: fx.Tensor,
        n_pts: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        tile = fx.Int32(gpu.block_idx.x)

        pr_ = GTensor(presence, dtype=T.i32, shape=(-1,))
        xyz_ = GTensor(out_xyz, dtype=T.i32, shape=(-1,))
        lut_ = GTensor(inp_row_lut, dtype=T.i32, shape=(-1,))
        msk_ = GTensor(mask, dtype=T.i32, shape=(-1,))
        akv_ = GTensor(active_kv_ids, dtype=T.i32, shape=(-1,))
        cnt_ = GTensor(active_count, dtype=T.i32, shape=(-1,))

        zero_i = fx.Int32(0)
        one_c = fx.Int32(1)

        o = tile * fx.Int32(BLOCK_M) + tid
        row = tid

        neg1_c = fx.Int32(-1)
        for kv in range_constexpr(KV):
            lut_.store(tile * fx.Int32(KV * BLOCK_M) + fx.Int32(kv * BLOCK_M) + row, neg1_c)

        oif = scf.IfOp((o < n_pts).ir_value(), results_=[], has_else=False)
        with ir.InsertionPoint(oif.then_block):
            o3 = o * fx.Int32(3)
            ox = fx.Int32(xyz_.load(o3))
            oy = fx.Int32(xyz_.load(o3 + fx.Int32(1)))
            oz = fx.Int32(xyz_.load(o3 + fx.Int32(2)))

            for kv in range_constexpr(KV):
                nx = ox + fx.Int32((kv % K) - R)
                ny = oy + fx.Int32(((kv // K) % K) - R)
                nz = oz + fx.Int32((kv // (K * K)) - R)
                in_range = (nx >= zero_i) & (nx < sx) & (ny >= zero_i) & (ny < sy) & (nz >= zero_i) & (nz < sz)
                rif = scf.IfOp(in_range.ir_value(), results_=[], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    inp = fx.Int32(pr_.load((nx * sy + ny) * sz + nz))
                    mif = scf.IfOp((inp >= zero_i).ir_value(), results_=[], has_else=False)
                    with ir.InsertionPoint(mif.then_block):
                        lut_.store(tile * fx.Int32(KV * BLOCK_M) + fx.Int32(kv * BLOCK_M) + row, inp)
                        msk_.store(tile * fx.Int32(KV) + fx.Int32(kv), one_c)
                        scf.YieldOp([])
                    scf.YieldOp([])

            scf.YieldOp([])

        gpu.barrier()

        t0_if = scf.IfOp((tid == fx.Int32(0)).ir_value(), results_=[], has_else=False)
        with ir.InsertionPoint(t0_if.then_block):
            base = tile * fx.Int32(KV)
            cur_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            cur = fx.memref_alloca(cur_ty, fx.make_layout(1, 1))
            fx.memref_store_vec(fx.Vector.filled(1, 0, fx.Int32), cur)

            for k in range_constexpr(KV):
                m = fx.Int32(msk_.load(base + fx.Int32(k)))
                act_if = scf.IfOp((m != fx.Int32(0)).ir_value(), results_=[], has_else=False)
                with ir.InsertionPoint(act_if.then_block):
                    c = fx.Vector(fx.memref_load_vec(cur))[0]
                    akv_.store(base + c, fx.Int32(k))
                    fx.memref_store_vec(fx.Vector.from_elements([c + fx.Int32(1)]), cur)
                    scf.YieldOp([])

            cnt_.store(tile, fx.Vector(fx.memref_load_vec(cur))[0])
            scf.YieldOp([])

    @flyc.jit
    def launch(
        presence,
        out_xyz,
        sx,
        sy,
        sz,
        inp_row_lut,
        mask,
        active_kv_ids,
        active_count,
        n_pts,
        num_tiles,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_tiled_kernel(
            presence,
            out_xyz,
            sx,
            sy,
            sz,
            inp_row_lut,
            mask,
            active_kv_ids,
            active_count,
            n_pts,
        ).launch(grid=(num_tiles,), block=(BLOCK_M,), stream=stream)

    return launch


def _fold_batch_column(coords, spatial_shape, n_batch, R):
    batch = coords[:, 0]
    xyz = coords[:, 1:]
    if spatial_shape is not None:
        sx_raw = int(spatial_shape[0])
        nb = int(n_batch) if n_batch is not None else int(batch.max().item()) + 1
        x_shift = xyz[:, 0]
    else:
        xyz_min, xyz_max = torch.aminmax(xyz, dim=0)
        sx_raw = int((xyz_max[0] - xyz_min[0] + 1).item())
        nb = int(batch.max().item()) + 1
        x_shift = xyz[:, 0] - xyz_min[0]
    coords = torch.stack([x_shift + batch * (sx_raw + 2 * R), xyz[:, 1], xyz[:, 2]], dim=1)
    if spatial_shape is not None:
        spatial_shape = (nb * (sx_raw + 2 * R), int(spatial_shape[1]), int(spatial_shape[2]))
    return coords, spatial_shape


def build_lut_dense(coords: torch.Tensor, block_m: int = 16, spatial_shape=None, n_batch=None, kernel_size: int = 3):
    KV = kernel_size**3
    R = kernel_size // 2
    device = coords.device
    N = int(coords.shape[0])
    num_tiles = (N + block_m - 1) // block_m

    zbuf = torch.zeros(num_tiles * (2 * KV + 1), dtype=torch.int32, device=device)
    mask = zbuf[: num_tiles * KV].view(num_tiles, KV)
    active_kv_ids = zbuf[num_tiles * KV : 2 * num_tiles * KV].view(num_tiles, KV)
    active_count = zbuf[2 * num_tiles * KV :].view(num_tiles)
    if N == 0:
        inp_row_lut = torch.full((num_tiles, KV, block_m), -1, dtype=torch.int32, device=device)
        return inp_row_lut, mask, active_kv_ids, active_count, num_tiles, 0
    inp_row_lut = torch.empty((num_tiles, KV, block_m), dtype=torch.int32, device=device)

    if coords.shape[1] == 4:
        coords, spatial_shape = _fold_batch_column(coords, spatial_shape, n_batch, R)

    if spatial_shape is not None:
        sx, sy, sz = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
        out_xyz = coords.to(torch.int32).contiguous()
    else:
        cmin, cmax = torch.aminmax(coords, dim=0)
        sx, sy, sz = (cmax - cmin + 1).tolist()
        out_xyz = (coords - cmin.unsqueeze(0)).to(torch.int32).contiguous()

    grid_size = sx * sy * sz
    presence = torch.empty(grid_size, dtype=torch.int32, device=device)

    stream = torch.cuda.current_stream()

    presence.fill_(-1)
    scatter_block = 256
    scatter_grid = (N + scatter_block - 1) // scatter_block
    _run_compiled(
        _compile_presence_scatter_kernel(scatter_block),
        out_xyz.reshape(-1),
        presence,
        sy,
        sz,
        N,
        scatter_grid,
        stream,
    )

    _run_compiled(
        _compile_fused_tiled_kernel(block_m, KV),
        presence,
        out_xyz.reshape(-1),
        sx,
        sy,
        sz,
        inp_row_lut.reshape(-1),
        mask.reshape(-1),
        active_kv_ids.reshape(-1),
        active_count.reshape(-1),
        N,
        num_tiles,
        stream,
    )
    return inp_row_lut, mask, active_kv_ids, active_count, num_tiles, N


# NOTE: this kernel stays on raw arith/vector ops. Its keys are KEY_T (i32 or i64
# depending on the grid extent), and GTensor.load returns a raw value whose type
# follows that dtype rather than an fx wrapper, so operator-based compares and
# fx.Int32 offsets do not apply here -- forcing them changes the buffer_load
# offset path and fails to compile. The map kernels above, whose tensors are all
# i32, use the fx operator surface.
@functools.lru_cache(maxsize=16)
def _compile_zdelta_kernel(block_m: int, kernel_size: int, n_bits: int, key32: bool = False):
    BLOCK_M = block_m
    K = kernel_size
    KV = K**3
    N_GROUPS = K * K
    KEY32 = key32

    GROUP_KV = tuple(tuple(((j) * K * K + ((g // K)) * K + (g % K)) for j in range(K)) for g in range(N_GROUPS))

    @flyc.kernel(known_block_size=[BLOCK_M, 1, 1])
    def zdelta_kernel(
        sorted_keys: fx.Tensor,
        sorted_order: fx.Tensor,
        out_packed: fx.Tensor,
        delta_packed: fx.Tensor,
        inp_row_lut: fx.Tensor,
        mask: fx.Tensor,
        active_kv_ids: fx.Tensor,
        active_count: fx.Tensor,
        n_pts: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        tile = fx.Int32(gpu.block_idx.x)

        key_t = T.i32 if const_expr(KEY32) else T.i64
        sk_ = GTensor(sorted_keys, dtype=key_t, shape=(-1,))
        so_ = GTensor(sorted_order, dtype=T.i32, shape=(-1,))
        op_ = GTensor(out_packed, dtype=key_t, shape=(-1,))
        dp_ = GTensor(delta_packed, dtype=key_t, shape=(-1,))
        lut_ = GTensor(inp_row_lut, dtype=T.i32, shape=(-1,))
        msk_ = GTensor(mask, dtype=T.i32, shape=(-1,))
        akv_ = GTensor(active_kv_ids, dtype=T.i32, shape=(-1,))
        cnt_ = GTensor(active_count, dtype=T.i32, shape=(-1,))

        one_i32 = fx.Int32(arith.constant(1, type=T.i32))
        zero_i32 = fx.Int32(arith.constant(0, type=T.i32))

        o = tile * fx.Int32(const_expr(BLOCK_M)) + tid
        row = fx.Index(tid)

        neg1 = fx.Int32(arith.constant(-1, type=T.i32))
        for kv in range_constexpr(KV):
            lut_.store(
                fx.Index(tile) * fx.Index(const_expr(KV * BLOCK_M)) + fx.Index(const_expr(kv * BLOCK_M)) + row,
                neg1,
            )

        o_ok = arith.cmpi(arith.CmpIPredicate.slt, o, n_pts)
        oif = scf.IfOp(o_ok, results_=[], has_else=False)
        with ir.InsertionPoint(oif.then_block):
            q_packed = op_.load(fx.Index(o))

            i32_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(const_expr(1), 1), fx.AddressSpace.Register)

            for g in range_constexpr(N_GROUPS):
                anchor_kv = const_expr(GROUP_KV[g][0])
                qkey_a = arith.addi(q_packed, dp_.load(fx.Index(anchor_kv)))

                lo_m = fx.memref_alloca(i32_ty, fx.make_layout(const_expr(1), 1))
                hi_m = fx.memref_alloca(i32_ty, fx.make_layout(const_expr(1), 1))
                fx.memref_store_vec(vector.from_elements(T.vec(1, T.i32), [zero_i32]), lo_m)
                fx.memref_store_vec(vector.from_elements(T.vec(1, T.i32), [n_pts]), hi_m)
                for _it in range_constexpr(n_bits):
                    lo = vector.extract(fx.memref_load_vec(lo_m), static_position=[const_expr(0)], dynamic_position=[])
                    hi = vector.extract(fx.memref_load_vec(hi_m), static_position=[const_expr(0)], dynamic_position=[])
                    still = arith.cmpi(arith.CmpIPredicate.slt, lo, hi)
                    sif = scf.IfOp(still, results_=[], has_else=False)
                    with ir.InsertionPoint(sif.then_block):
                        mid = arith.addi(lo, arith.divsi(arith.subi(hi, lo), arith.constant(2, type=T.i32)))
                        lt = arith.cmpi(arith.CmpIPredicate.slt, sk_.load(fx.Index(fx.Int32(mid))), qkey_a)
                        fx.memref_store_vec(
                            vector.from_elements(
                                T.vec(1, T.i32),
                                [arith.select(lt, arith.addi(mid, arith.constant(1, type=T.i32)), lo)],
                            ),
                            lo_m,
                        )
                        fx.memref_store_vec(vector.from_elements(T.vec(1, T.i32), [arith.select(lt, hi, mid)]), hi_m)
                        scf.YieldOp([])
                pos = vector.extract(fx.memref_load_vec(lo_m), static_position=[const_expr(0)], dynamic_position=[])

                for d in range_constexpr(K):
                    p = arith.addi(pos, arith.constant(d, type=T.i32))
                    p_ok = arith.cmpi(arith.CmpIPredicate.slt, p, n_pts)
                    pif = scf.IfOp(p_ok, results_=[], has_else=False)
                    with ir.InsertionPoint(pif.then_block):
                        sk_val = sk_.load(fx.Index(fx.Int32(p)))
                        inp_row_here = so_.load(fx.Index(fx.Int32(p)))
                        for j in range_constexpr(K):
                            kv_c = const_expr(GROUP_KV[g][j])
                            qk = arith.addi(q_packed, dp_.load(fx.Index(kv_c)))
                            hit = arith.cmpi(arith.CmpIPredicate.eq, sk_val, qk)
                            hif = scf.IfOp(hit, results_=[], has_else=False)
                            with ir.InsertionPoint(hif.then_block):
                                lut_.store(
                                    fx.Index(tile) * fx.Index(const_expr(KV * BLOCK_M))
                                    + fx.Index(const_expr(kv_c * BLOCK_M))
                                    + row,
                                    inp_row_here,
                                )
                                msk_.store(
                                    fx.Index(tile) * fx.Index(const_expr(KV)) + fx.Index(const_expr(kv_c)), one_i32
                                )
                                scf.YieldOp([])
                        scf.YieldOp([])
            scf.YieldOp([])

        gpu.barrier()

        is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, zero_i32)
        t0if = scf.IfOp(is_t0, results_=[], has_else=False)
        with ir.InsertionPoint(t0if.then_block):
            base = fx.Index(tile) * fx.Index(const_expr(KV))
            cur_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(const_expr(1), 1), fx.AddressSpace.Register)
            cur = fx.memref_alloca(cur_ty, fx.make_layout(const_expr(1), 1))
            fx.memref_store_vec(arith.constant_vector(0, T.vec(1, T.i32)), cur)
            for k in range_constexpr(KV):
                m = msk_.load(base + fx.Index(const_expr(k)))
                act_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.ne, m, zero_i32), results_=[], has_else=False)
                with ir.InsertionPoint(act_if.then_block):
                    c = vector.extract(fx.memref_load_vec(cur), static_position=[const_expr(0)], dynamic_position=[])
                    akv_.store(base + fx.Index(fx.Int32(c)), fx.Int32(arith.constant(k, type=T.i32)))
                    fx.memref_store_vec(
                        vector.from_elements(T.vec(1, T.i32), [arith.addi(c, arith.constant(1, type=T.i32))]),
                        cur,
                    )
                    scf.YieldOp([])
            cnt_.store(
                fx.Index(tile),
                vector.extract(fx.memref_load_vec(cur), static_position=[const_expr(0)], dynamic_position=[]),
            )
            scf.YieldOp([])

    @flyc.jit
    def launch(
        sorted_keys,
        sorted_order,
        out_packed,
        delta_packed,
        inp_row_lut,
        mask,
        active_kv_ids,
        active_count,
        n_pts,
        num_tiles,
        stream: fx.Stream = fx.Stream(None),
    ):
        zdelta_kernel(
            sorted_keys,
            sorted_order,
            out_packed,
            delta_packed,
            inp_row_lut,
            mask,
            active_kv_ids,
            active_count,
            n_pts,
        ).launch(grid=(num_tiles,), block=(BLOCK_M,), stream=stream)

    return launch


def build_lut_zdelta(coords: torch.Tensor, block_m: int = 16, spatial_shape=None, n_batch=None, kernel_size: int = 3):
    K = kernel_size
    KV = K**3
    R = K // 2
    device = coords.device
    N = int(coords.shape[0])
    num_tiles = (N + block_m - 1) // block_m

    zbuf = torch.zeros(num_tiles * (2 * KV + 1), dtype=torch.int32, device=device)
    mask = zbuf[: num_tiles * KV].view(num_tiles, KV)
    active_kv_ids = zbuf[num_tiles * KV : 2 * num_tiles * KV].view(num_tiles, KV)
    active_count = zbuf[2 * num_tiles * KV :].view(num_tiles)
    if N == 0:
        inp_row_lut = torch.full((num_tiles, KV, block_m), -1, dtype=torch.int32, device=device)
        return inp_row_lut, mask, active_kv_ids, active_count, num_tiles, 0
    inp_row_lut = torch.empty((num_tiles, KV, block_m), dtype=torch.int32, device=device)

    if coords.shape[1] == 4:
        coords, spatial_shape = _fold_batch_column(coords, spatial_shape, n_batch, R)

    c64 = coords.to(torch.int64)
    cmin = torch.aminmax(c64, dim=0)[0] if spatial_shape is None else None
    shifted = c64 - cmin.unsqueeze(0) if cmin is not None else c64

    if spatial_shape is not None:
        sx, sy, sz = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
    else:
        ext = (torch.aminmax(c64, dim=0)[1] - cmin + 1).tolist()
        sx, sy, sz = int(ext[0]), int(ext[1]), int(ext[2])
    SY = sy + 2 * R
    SZ = sz + 2 * R

    key32 = (sx + 2 * R) * SY * SZ < (1 << 31)
    key_dtype = torch.int32 if key32 else torch.int64

    shifted = shifted + R
    packed = ((shifted[:, 0] * SY + shifted[:, 1]) * SZ + shifted[:, 2]).to(key_dtype)

    sorted_keys, perm = torch.sort(packed)

    kvs = torch.arange(KV, device=device, dtype=torch.int64)
    dx = (kvs % K) - R
    dy = ((kvs // K) % K) - R
    dz = (kvs // (K * K)) - R
    delta_packed = ((dx * SY + dy) * SZ + dz).to(key_dtype)

    n_bits = max(1, int(N - 1).bit_length() + 1)
    _run_compiled(
        _compile_zdelta_kernel(block_m, K, n_bits, key32),
        sorted_keys.contiguous(),
        perm.to(torch.int32).contiguous(),
        packed.contiguous(),
        delta_packed.contiguous(),
        inp_row_lut.reshape(-1),
        mask.reshape(-1),
        active_kv_ids.reshape(-1),
        active_count.reshape(-1),
        N,
        num_tiles,
        torch.cuda.current_stream(),
    )
    return inp_row_lut, mask, active_kv_ids, active_count, num_tiles, N


ZDELTA_CELL_THRESHOLD = int(os.environ.get("FLYDSL_SPCONV_ZDELTA_CELLS", str(150_000_000)))


def build_lut_auto(coords, block_m=16, spatial_shape=None, n_batch=None, kernel_size=3):
    if spatial_shape is not None:
        sx, sy, sz = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
        if coords.shape[1] == 4:
            nb = int(n_batch) if n_batch is not None else int(coords[:, 0].max().item()) + 1
            sx = nb * (sx + 2 * (kernel_size // 2))
        cells = sx * sy * sz
    else:
        cmin, cmax = torch.aminmax(coords[:, -3:], dim=0)
        cells = int(torch.prod((cmax - cmin + 1).to(torch.int64)).item())
    fn = build_lut_zdelta if cells > ZDELTA_CELL_THRESHOLD else build_lut_dense
    return fn(coords, block_m, spatial_shape=spatial_shape, n_batch=n_batch, kernel_size=kernel_size)


def build_compacted_pairs(inp_row_lut, num_tiles, num_act, block_m=16, kernel_size=3):
    KV = kernel_size**3
    device = inp_row_lut.device
    lv = inp_row_lut.reshape(num_tiles, KV, block_m)

    out_all = torch.arange(num_tiles * block_m, dtype=torch.int32, device=device).reshape(num_tiles, 1, block_m)
    valid = (lv >= 0) & (out_all < num_act)

    counts = valid.sum(dim=(0, 2))
    tiles_per_kv = (counts + block_m - 1) // block_m
    n_tiles_c = int(tiles_per_kv.sum().item())
    if n_tiles_c == 0:
        empty_i = torch.zeros(0, dtype=torch.int32, device=device)
        return empty_i, empty_i, empty_i, 0

    tile_start = torch.cumsum(tiles_per_kv, 0) - tiles_per_kv
    slot_start = (tile_start * block_m).to(torch.int64)

    vt = valid.permute(1, 0, 2).reshape(KV, -1)
    idx = vt.reshape(-1).nonzero(as_tuple=True)[0]
    kv_of = torch.div(idx, vt.shape[1], rounding_mode="floor")
    first_of_kv = torch.cumsum(counts, 0) - counts
    rank = torch.arange(idx.numel(), device=device) - first_of_kv[kv_of]
    dst = slot_start[kv_of] + rank

    total_slots = n_tiles_c * block_m
    in_rows = torch.zeros(total_slots, dtype=torch.int32, device=device)
    out_rows = torch.full((total_slots,), -1, dtype=torch.int32, device=device)

    src_in = lv.permute(1, 0, 2).reshape(-1)
    src_out = out_all.expand(num_tiles, KV, block_m).permute(1, 0, 2).reshape(-1)
    in_rows.scatter_(0, dst, src_in[idx])
    out_rows.scatter_(0, dst, src_out[idx])

    tile_kv = torch.repeat_interleave(torch.arange(KV, dtype=torch.int32, device=device), tiles_per_kv).contiguous()
    return in_rows, out_rows, tile_kv, n_tiles_c


F32_BLOCK_M = 16
F32_C_OUT_TILE = 16
F32_BLOCK_THREADS = 64
VEC_BF16 = 8

_CT_ENV = os.environ.get("FLYDSL_SPCONV_CT_PER_BLOCK")
_KU_ENV = os.environ.get("FLYDSL_SPCONV_K_UNROLL")


def _pick_ct_per_block(c_out: int) -> int:
    if _CT_ENV is not None:
        return int(_CT_ENV)
    return 4 if c_out <= 256 else 8


def _pick_k_unroll(c_out: int) -> int:
    if _KU_ENV is not None:
        return int(_KU_ENV)
    return 2 if c_out >= 512 else 1


_LUT_CACHE: dict = {}


_CMP_CACHE: dict = {}


def clear_lut_cache():
    _LUT_CACHE.clear()
    _CMP_CACHE.clear()


_WPACK_CACHE: dict = {}


def clear_weight_cache():
    _WPACK_CACHE.clear()


def _prepare_weight_bf16(weight: torch.Tensor, kv_order: str, c_out_tile: int, k_step: int = 32):
    key = (id(weight), weight._version, kv_order, c_out_tile, "bf16")
    hit = _WPACK_CACHE.get(key)
    if hit is not None:
        return hit

    c_out, c_in, k = weight.shape[0], weight.shape[1], weight.shape[2]
    if kv_order == "spconv":
        weight = weight.permute(0, 1, 4, 3, 2)
    elif kv_order != "zyx":
        raise ValueError(f"kv_order must be 'zyx' or 'spconv', got {kv_order!r}")

    cip = (c_in + k_step - 1) // k_step * k_step
    cop = (c_out + c_out_tile - 1) // c_out_tile * c_out_tile
    if cip != c_in or cop != c_out:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 0, 0, cip - c_in, 0, cop - c_out))

    n_tiles = cop // c_out_tile
    kk = k**3
    packed = (
        weight.reshape(n_tiles, c_out_tile, cip // VEC_BF16, VEC_BF16, kk)
        .permute(4, 0, 2, 1, 3)
        .contiguous()
        .to(torch.bfloat16)
        .reshape(-1)
    )

    entry = (packed, cip, cop, k**3)
    _WPACK_CACHE[key] = entry
    try:
        weakref.finalize(weight, _WPACK_CACHE.pop, key, None)
    except TypeError:
        pass
    return entry


@functools.lru_cache(maxsize=32)
def _compile_bf16_compacted(c_in: int, c_out: int, ct_per_block: int = 1, k_unroll: int = 1):
    C_IN = c_in
    C_OUT = c_out
    BLOCK_M = F32_BLOCK_M
    C_OUT_TILE = F32_C_OUT_TILE
    N_C_OUT_TILES = C_OUT // C_OUT_TILE
    BLOCK_THREADS = F32_BLOCK_THREADS
    K_STEP = 32
    VEC = 8

    CT_PER_BLOCK = max(1, min(ct_per_block, N_C_OUT_TILES))
    while N_C_OUT_TILES % CT_PER_BLOCK != 0:
        CT_PER_BLOCK -= 1
    CT_GROUPS = N_C_OUT_TILES // CT_PER_BLOCK

    W_TILE_ELEMS = C_IN * C_OUT_TILE

    K_UNROLL = max(1, k_unroll)
    while (C_IN // K_STEP) % K_UNROLL != 0:
        K_UNROLL -= 1

    allocator = SmemAllocator(None, arch="gfx950", global_sym_name=f"smem_spconv_bf16_ci{c_in}_co{c_out}_k{K_UNROLL}")

    _GTensor = GTensor
    _atomic_add = buffer_atomic_add

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel_bf16_cmp(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        in_rows: fx.Tensor,
        out_rows: fx.Tensor,
        tile_kv: fx.Tensor,
    ):
        from flydsl.expr import gpu

        tid = fx.Int32(gpu.thread_idx.x)
        bid = fx.Int32(gpu.block_idx.x)

        feat_ = _GTensor(features, dtype=T.bf16, shape=(-1,))
        wp_ = _GTensor(weights_packed, dtype=T.bf16, shape=(-1,))
        out_ = _GTensor(output, dtype=T.f32, shape=(-1,))
        ir_ = _GTensor(in_rows, dtype=T.i32, shape=(-1,))
        or_ = _GTensor(out_rows, dtype=T.i32, shape=(-1,))
        tkv_ = _GTensor(tile_kv, dtype=T.i32, shape=(-1,))

        cg = bid % fx.Int32(CT_GROUPS)
        p_tile = bid // fx.Int32(CT_GROUPS)
        c_out_offset = cg * fx.Int32(C_OUT_TILE * CT_PER_BLOCK)

        mfma_row = tid % fx.Int32(16)
        mfma_col = mfma_row
        mfma_k_lane = tid // fx.Int32(16)
        c_row_vec_base = mfma_k_lane * fx.Int32(4)
        k_lane_off = mfma_k_lane * fx.Int32(VEC)

        acc_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
        acc_regs = []
        for _ctl in range_constexpr(CT_PER_BLOCK):
            _r = fx.memref_alloca(acc_reg_ty, fx.make_layout(4, 1))
            fx.memref_store_vec(fx.Vector.filled(4, 0.0, fx.Float32), _r)
            acc_regs.append(_r)

        k = fx.Int32(tkv_.load(p_tile))
        w_base = k * fx.Int32(N_C_OUT_TILES * W_TILE_ELEMS) + cg * fx.Int32(W_TILE_ELEMS * CT_PER_BLOCK)

        inp_row = fx.Int32(ir_.load(p_tile * fx.Int32(BLOCK_M) + mfma_row))
        feat_base = inp_row * fx.Int32(C_IN)

        for c_blk in range_constexpr(0, C_IN, K_STEP * K_UNROLL):
            a_vecs = []
            b_vecs = []
            for ku in range_constexpr(K_UNROLL):
                c0 = c_blk + ku * K_STEP
                a_vecs.append(feat_.vec_load((feat_base + fx.Int32(c0) + k_lane_off,), VEC))
                for ctl in range_constexpr(CT_PER_BLOCK):
                    b_off = (
                        fx.Int32(ctl * W_TILE_ELEMS + (c0 // VEC) * C_OUT_TILE * VEC)
                        + mfma_k_lane * fx.Int32(C_OUT_TILE * VEC)
                        + mfma_col * fx.Int32(VEC)
                    )
                    b_vecs.append(wp_.vec_load((w_base + b_off,), VEC))
            for ku in range_constexpr(K_UNROLL):
                for ctl in range_constexpr(CT_PER_BLOCK):
                    cur_acc = fx.memref_load_vec(acc_regs[ctl])
                    new_acc = rocdl.mfma_f32_16x16x32_bf16(
                        T.vec(4, T.f32), [a_vecs[ku], b_vecs[ku * CT_PER_BLOCK + ctl], cur_acc, 0, 0, 0]
                    )
                    fx.memref_store_vec(new_acc, acc_regs[ctl])

        # The epilogue stays on fx.Index / vector.extract on purpose: the fx-operator
        # form is op-count-identical but measured 185 us vs 170 us at C_OUT=512
        # (median-of-7, ~9% slower), an instruction-scheduling effect around the
        # predicated atomics. The hot loop above took the migration with no cost.
        neg1 = fx.Int32(-1)
        z0 = fx.Int32(0)
        for ctl in range_constexpr(CT_PER_BLOCK):
            final_acc = fx.memref_load_vec(acc_regs[ctl])
            ctl_col = fx.Index(const_expr(ctl * C_OUT_TILE)) + fx.Index(mfma_col)
            for ri in range_constexpr(4):
                slot_r = (
                    fx.Index(p_tile) * fx.Index(const_expr(BLOCK_M))
                    + fx.Index(c_row_vec_base)
                    + fx.Index(const_expr(ri))
                )
                o_row = or_.load(slot_r)
                store_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.sgt, o_row, neg1), results_=[], has_else=False)
                with ir.InsertionPoint(store_if.then_block):
                    val = vector.extract(final_acc, static_position=[const_expr(ri)], dynamic_position=[])
                    off_elems = fx.Index(o_row) * fx.Index(const_expr(C_OUT)) + c_out_offset + ctl_col
                    off_b = fx.Int32(arith.index_cast(T.i32, off_elems)) * fx.Int32(4)
                    _atomic_add(val, out_.rsrc, off_b, z0, z0)
                    scf.YieldOp([])

    @flyc.jit
    def launch_fn(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        in_rows: fx.Tensor,
        out_rows: fx.Tensor,
        tile_kv: fx.Tensor,
        n_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel_bf16_cmp(features, weights_packed, output, in_rows, out_rows, tile_kv).launch(
            grid=(n_tiles * CT_GROUPS,), block=(BLOCK_THREADS,), stream=stream
        )

    return launch_fn


def sparse_conv3d_from_coords_f32(
    coords: torch.Tensor,
    features: torch.Tensor,
    weight: torch.Tensor,
    indice_key: Optional[str] = None,
    spatial_shape=None,
    n_batch: Optional[int] = None,
    kv_order: str = "zyx",
) -> torch.Tensor:
    assert features.dtype == torch.float32, "fp32 path: features must be fp32"
    assert weight.dtype == torch.float32, "fp32 path: weight must be fp32"
    k = weight.shape[2]
    assert weight.shape[2:] == (k, k, k), f"kernel must be cubic, got {tuple(weight.shape[2:])}"
    assert k % 2 == 1, f"kernel size must be odd, got {k}"

    cache_key = (indice_key, F32_BLOCK_M, k) if indice_key is not None else None
    if cache_key is not None and cache_key in _LUT_CACHE:
        inp_row_lut, mask, _akv, _cnt, num_tiles, num_act = _LUT_CACHE[cache_key]
        if num_act != coords.shape[0]:
            raise ValueError(
                f"indice_key={indice_key!r} was cached for N={num_act} but got "
                f"N={coords.shape[0]}; use a distinct key per set of coordinates"
            )
    else:
        inp_row_lut, mask, _akv, _cnt, num_tiles, num_act = build_lut_auto(
            coords, F32_BLOCK_M, spatial_shape=spatial_shape, n_batch=n_batch, kernel_size=k
        )
        if cache_key is not None:
            _LUT_CACHE[cache_key] = (inp_row_lut, mask, _akv, _cnt, num_tiles, num_act)

    packed_w, c_in, c_out, _kv = _prepare_weight_bf16(weight, kv_order, F32_C_OUT_TILE)
    c_in_real = features.shape[1]
    c_out_real = weight.shape[0]
    if c_in != c_in_real:
        features = torch.nn.functional.pad(features, (0, c_in - c_in_real))

    out = torch.zeros(num_act, c_out, dtype=torch.float32, device=features.device)
    if num_tiles == 0:
        return out[:, :c_out_real]

    cmp_entry = _CMP_CACHE.get(cache_key) if cache_key is not None else None
    if cmp_entry is None:
        cmp_entry = build_compacted_pairs(inp_row_lut, num_tiles, num_act, F32_BLOCK_M, k)
        if cache_key is not None:
            _CMP_CACHE[cache_key] = cmp_entry
    in_rows, out_rows, tile_kv, n_tiles_c = cmp_entry
    if n_tiles_c == 0:
        return out[:, :c_out_real]

    _run_compiled(
        _compile_bf16_compacted(c_in, c_out, _pick_ct_per_block(c_out), _pick_k_unroll(c_out)),
        features.to(torch.bfloat16).contiguous(),
        packed_w,
        out,
        in_rows,
        out_rows,
        tile_kv,
        n_tiles_c,
        torch.cuda.current_stream(),
    )
    return out if c_out == c_out_real else out[:, :c_out_real]
