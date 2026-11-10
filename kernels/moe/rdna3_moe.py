# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""bf16 / f16 MoE for RDNA3 / RDNA3.5 (gfx11*, wave32 WMMA).

One routed layer is five launches:

    gating   logits    -> topk_ids, topk_weights      optional; build_topk_gating_softmax_module
    routing  topk_ids  -> sorted_ids, expert_ids       counting sort in one workgroup
    gemm1    x         -> a2[tokens, topk, inter]      silu(gate) * up, scattered by (token, slot)
    gemm2    a2        -> y[tokens, topk, model]       down, routing weight folded into the epilogue
    reduce   y         -> out[tokens, model]           compile_moe_reduction, unweighted sum over topk

The topk sum is its own launch because gfx11 has no bf16 atomic, so gemm2 cannot accumulate into a
``[tokens, model]`` output the way the CDNA kernels do; it weights each row instead and leaves a plain sum.
The gating and the reduce are the CDNA package's kernels, which are wave-size agnostic and run here as built.

With few rows per expert two launches drop out: up to ``TOKEN_DOWN_ROWS_PER_EXPERT`` rows an expert a GEMV
(``compile_token_down``) does gemm2 and the reduce together, and up to ``DIRECT_ROWS_PER_EXPERT`` gemm1 also
takes one tile per slot straight from ``topk_ids`` (``direct``), so nothing builds a routing.

The routing never tells the host how many tiles it made: reading that back costs more than the routing kernel.
The grid is ``max_blocks_for``'s bound and each GEMM workgroup compares its tile index against the count left
in device memory (``bounded_blocks``).

Everything is this rank's: ``w1``/``w2`` hold this rank's experts. Only tensor parallelism is wired up.
``w1`` is read in ``wmma_order`` (as the CDNA kernels read a pre-shuffled weight): rearranged once at load,
each 16-lane WMMA operand is one contiguous 256 B run rather than 32 B from each of 16 rows.

The tile tables and thresholds were measured on a W7900 (gfx1100).
"""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector
from flydsl.expr import as_ir_value, const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
from kernels.gemm.rdna3_f16_gemm import K_PAD, WAVE_SIZE, WMMA_K, WMMA_M, WMMA_N
from kernels.moe.moe_gemm_2stage import compile_moe_reduction
from kernels.moe.topk_gating_softmax_kernel import build_topk_gating_softmax_module

__all__ = [
    "Routing",
    "SUPPORTED_TILE_M",
    "build_routing_device",
    "compile_grouped_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_routing",
    "compile_token_down",
    "create_grouped_gemm_module",
    "max_blocks_for",
    "moe_forward",
    "moe_forward_from_logits",
    "moe_gating",
    "moe_reduce",
    "tile_bucket_for",
    "tile_m_for",
    "wmma_order",
]

# ── Routing ──────────────────────────────────────────────────────────────────

# Packed routing id (slot << 24) | token, as in the CDNA MoE kernels. A padded row carries the sentinel
# (topk << 24) | tokens, whose row lands one past the end of every buffer it indexes.
TOKEN_MASK = 0xFFFFFF
SLOT_SHIFT = 24


class Routing(NamedTuple):
    """One padded routing order: rows grouped by expert, each expert padded up to ``tile_m``.

    ``sorted_ids`` has ``num_blocks * tile_m`` entries and ``expert_ids`` one per tile. ``exact`` says whether
    ``num_blocks`` is the tile count or an upper bound on it; with a bound the real count is in
    ``num_blocks_device`` and the GEMMs must be built with ``bounded_blocks=True``.
    """

    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_blocks: int
    tile_m: int
    tokens: int
    topk: int
    exact: bool = True
    num_blocks_device: torch.Tensor | None = None

    @property
    def rows(self) -> int:
        return self.num_blocks * self.tile_m


def _on_device_of_first_arg(fn: Callable) -> Callable:
    """Run ``fn`` with its first tensor's device current: a kernel loads into the current device, and a launch
    there with another device's pointers is a memory fault rather than an error."""

    @functools.wraps(fn)
    def wrapper(t, *args, **kwargs):
        if t.device.type != "cuda" or t.device.index == torch.cuda.current_device():
            return fn(t, *args, **kwargs)
        with torch.cuda.device(t.device):
            return fn(t, *args, **kwargs)

    return wrapper


_ROUTING_BLOCK = 256
# The one workgroup walks every row twice (histogram, scatter), so prefill row counts take a wider one.
_WIDE_ROUTING_BLOCK = 1024
_ROWS_FOR_WIDE_ROUTING = 2048
# One LDS counter per expert next to the block's scan scratch.
MAX_EXPERTS = 1024


def _routing_block_for(rows: int) -> int:
    return _ROUTING_BLOCK if rows < _ROWS_FOR_WIDE_ROUTING else _WIDE_ROUTING_BLOCK


def _scan_offsets(width: int) -> tuple[int, ...]:
    """Hillis-Steele step offsets 1, 2, 4, ... covering ``width``."""
    offs, off = [], 1
    while off < int(width):
        offs.append(off)
        off *= 2
    return tuple(offs)


def _lds_ld(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_st(ptr, val, idx):
    fx.ptr_store(val, ptr + fx.Int64(idx))


def _lds_atomic_add(ptr, idx, val):
    """``ds_add_rtn_u32`` at workgroup scope; in the scatter the returned old value is the row's slot."""
    llvm_ptr = fx.to_llvm_ptr(fx.add_offset(ptr, fx.make_int_tuple(idx)))
    raw = val.ir_value() if hasattr(val, "ir_value") else val
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add, llvm_ptr, raw, _llvm.AtomicOrdering.monotonic, syncscope="workgroup", alignment=4
        ).result
    )


def _create_routing_module(*, experts: int, topk: int, tile_m: int, block: int = _ROUTING_BLOCK) -> Callable:
    """The routing kernel: histogram, scan over experts, scatter, then the padding.

    One workgroup, because the prefix sum runs across experts and the whole job is a few thousand ints. Rows
    are claimed with an LDS atomic, so the order inside an expert is not fixed; no output depends on it.
    Returns ``launch(topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks, stream)``; ``aux[0]`` gets
    the tile count.
    """
    E, TOPK, TILE = int(experts), int(topk), int(tile_m)
    BLOCK = int(block)
    if not str(get_rocm_arch() or "").startswith("gfx11"):
        raise RuntimeError(f"rdna3_moe requires gfx11*, got {get_rocm_arch()!r}")
    if not 1 <= E <= MAX_EXPERTS:
        raise ValueError(f"experts must be in [1, {MAX_EXPERTS}], got {E}")
    if not 1 <= TOPK <= 0xFF:
        raise ValueError(f"the packed id has 8 bits of slot, so topk must be in [1, 255], got {TOPK}")
    if TILE < 1:
        raise ValueError(f"tile_m must be positive, got {TILE}")

    CHUNKS = -(-E // BLOCK)

    @fx.struct
    class _Shared:
        # Per expert: the row count, then the slot its rows start at, then the scatter's cursor.
        cnt: fx.Array[fx.Int32, E, 16]
        scan: fx.Array[fx.Int32, BLOCK, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def routing_kernel(
        TopkIds: fx.Pointer,
        SortedIds: fx.Pointer,
        ExpertIds: fx.Pointer,
        Aux: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_blocks: fx.Int32,
    ):
        i32pt = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        ids = fx.inttoptr(i32pt, fx.Int64(ptrtoint(TopkIds)))
        out = fx.inttoptr(i32pt, fx.Int64(ptrtoint(SortedIds)))
        eout = fx.inttoptr(i32pt, fx.Int64(ptrtoint(ExpertIds)))
        aux = fx.inttoptr(i32pt, fx.Int64(ptrtoint(Aux)))

        lds = fx.SharedAllocator().allocate(_Shared).peek()
        cnt_mr, scan_mr = lds.cnt.ptr, lds.scan.ptr

        tid = gpu.thread_idx.x
        c0, c1 = fx.Int32(0), fx.Int32(1)
        c_tile = fx.Int32(TILE)
        c_e = fx.Int32(E)

        def _tiles(count):
            # An expert with no rows owns no tile: an all-padding tile still streams its expert's weights.
            return (count + c_tile - c1) // c_tile

        rows = i32_tokens * fx.Int32(TOPK)
        sentinel = (fx.Int32(TOPK) << fx.Int32(SLOT_SHIFT)) | i32_tokens
        steps = ArithValue((rows + fx.Int32(BLOCK - 1)) // fx.Int32(BLOCK)).index_cast(T.index)

        def _my_row(_i):
            row = tid + fx.Int32(_i) * fx.Int32(BLOCK)
            return row, row < rows

        def _expert_of(row, inb):
            # A row naming an expert this rank does not hold is dropped.
            eid = _lds_ld(ids, inb.select(row, c0))
            return eid, inb & (eid >= c0) & (eid < c_e)

        # Histogram.
        for _c in range_constexpr(CHUNKS):
            e = fx.Int32(_c * BLOCK) + tid
            if e < c_e:
                _lds_st(cnt_mr, c0, e)
        gpu.barrier()

        for _i in range(fx.Index(0), steps, fx.Index(1)):
            row, inb = _my_row(_i)
            eid, keep = _expert_of(row, inb)
            if keep:
                _lds_atomic_add(cnt_mr, eid, c1)
        gpu.barrier()

        # Scan the tile counts chunk by chunk with a running carry, and name the tiles.
        nblocks = c0
        for _c in range_constexpr(CHUNKS):
            e = fx.Int32(_c * BLOCK) + tid
            inr = e < c_e
            count = inr.select(_lds_ld(cnt_mr, inr.select(e, c0)), c0)
            mine = _tiles(count)
            _lds_st(scan_mr, mine, tid)
            gpu.barrier()

            cur = mine
            for off in _scan_offsets(BLOCK):
                src = tid - fx.Int32(off)
                prev = _lds_ld(scan_mr, (src >= c0).select(src, c0))
                # The next step reads this step's values out of the same array.
                gpu.barrier()
                cur = (src >= c0).select(cur + prev, cur)
                _lds_st(scan_mr, cur, tid)
                gpu.barrier()

            first = nblocks + cur - mine
            row0 = first * c_tile
            if inr:
                _lds_st(cnt_mr, row0, e)

            named = ArithValue(mine).index_cast(T.index)
            for _b in range(fx.Index(0), named, fx.Index(1)):
                _lds_st(eout, e, first + fx.Int32(_b))
            # Pad slots of this expert's last tile; disjoint from every slot the scatter claims.
            pad = ArithValue(mine * c_tile - count).index_cast(T.index)
            for _p in range(fx.Index(0), pad, fx.Index(1)):
                _lds_st(out, sentinel, row0 + count + fx.Int32(_p))

            nblocks = nblocks + _lds_ld(scan_mr, fx.Int32(BLOCK - 1))
            gpu.barrier()

        # Scatter.
        for _i in range(fx.Index(0), steps, fx.Index(1)):
            row, inb = _my_row(_i)
            eid, keep = _expert_of(row, inb)
            safe = inb.select(row, c0)
            packed = ((safe % fx.Int32(TOPK)) << fx.Int32(SLOT_SHIFT)) | (safe // fx.Int32(TOPK))
            if keep:
                _lds_st(out, packed, _lds_atomic_add(cnt_mr, eid, c1))

        if tid == c0:
            _lds_st(aux, nblocks, c0)

        # The tiles between the real count and the caller's bound: expert 0, all padding.
        tail_b = ArithValue((i32_max_blocks - nblocks - tid + fx.Int32(BLOCK - 1)) // fx.Int32(BLOCK)).index_cast(
            T.index
        )
        for _t in range(fx.Index(0), tail_b, fx.Index(1)):
            blk = nblocks + tid + fx.Int32(_t) * fx.Int32(BLOCK)
            if blk < i32_max_blocks:
                _lds_st(eout, c0, blk)
        tail_r = ArithValue(
            ((i32_max_blocks - nblocks) * c_tile - tid + fx.Int32(BLOCK - 1)) // fx.Int32(BLOCK)
        ).index_cast(T.index)
        for _t in range(fx.Index(0), tail_r, fx.Index(1)):
            slot = nblocks * c_tile + tid + fx.Int32(_t) * fx.Int32(BLOCK)
            if slot < i32_max_blocks * c_tile:
                _lds_st(out, sentinel, slot)

    @flyc.jit
    def launch_routing(
        TopkIds: fx.Pointer,
        SortedIds: fx.Pointer,
        ExpertIds: fx.Pointer,
        Aux: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        routing_kernel(TopkIds, SortedIds, ExpertIds, Aux, i32_tokens, i32_max_blocks).launch(
            grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    def launch(topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks, stream):
        ptrs = [flyc.from_c_void_p(fx.Uint8, t.data_ptr()) for t in (topk_ids, sorted_ids, expert_ids, aux)]
        return _run_compiled(launch_routing, *ptrs, fx.Int32(int(tokens)), fx.Int32(int(max_blocks)), stream)

    launch.experts = E
    launch.topk = TOPK
    launch.tile_m = TILE
    return launch


@functools.lru_cache(maxsize=256)
def compile_routing(*, experts: int, topk: int, tile_m: int) -> Callable:
    """Build (and cache) the routing kernel for one shape; each launch picks its workgroup width by row count."""
    variants = {_ROUTING_BLOCK: _create_routing_module(experts=experts, topk=topk, tile_m=tile_m)}

    def launch(topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks, stream):
        block = _routing_block_for(int(tokens) * int(topk))
        if block not in variants:
            variants[block] = _create_routing_module(experts=experts, topk=topk, tile_m=tile_m, block=block)
        return variants[block](topk_ids, sorted_ids, expert_ids, aux, tokens, max_blocks, stream)

    launch.experts, launch.topk, launch.tile_m = int(experts), int(topk), int(tile_m)
    return launch


def max_blocks_for(*, tokens: int, topk: int, experts: int, tile_m: int) -> int:
    """An upper bound on the tile count from the row count alone.

    A non-empty expert with ``c`` rows owns ``1 + (c-1)//t`` tiles, so ``n`` of them own at most
    ``n + rows//t``, with ``n <= min(E, rows)``.
    """
    rows = int(tokens) * int(topk)
    return max(1, min(int(experts), rows) + rows // int(tile_m))


class _Workspace(NamedTuple):
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    aux: torch.Tensor


_WORKSPACES: dict[tuple, _Workspace] = {}


def _new_workspace(max_blocks: int, tile_m: int, device) -> _Workspace:
    return _Workspace(
        sorted_ids=torch.empty(max_blocks * tile_m, dtype=torch.int32, device=device),
        expert_ids=torch.empty(max_blocks, dtype=torch.int32, device=device),
        aux=torch.empty(1, dtype=torch.int32, device=device),
    )


@_on_device_of_first_arg
def build_routing_device(
    topk_ids: torch.Tensor, *, experts: int, tile_m: int, exact: bool = False, reuse: bool = True, stream=None
) -> Routing:
    """The padded routing order of ``topk_ids`` (``[tokens, topk]``), built on the device.

    ``exact=True`` reads the tile count back and slices the buffers to it. The default does not wait: it
    returns the bound as ``num_blocks`` and leaves the count in ``num_blocks_device``.

    ``reuse`` keeps one set of buffers per shape. That is safe for one layer on one stream, because the next
    routing launch queues behind the GEMMs reading this one; ``reuse=False`` gives a routing its own buffers.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, topk], got {tuple(topk_ids.shape)}")
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"the routing kernel reads int32 expert ids, got {topk_ids.dtype}")
    tokens, topk = topk_ids.shape
    if tokens > TOKEN_MASK:
        raise ValueError(f"the packed id has 24 bits of token, so tokens must be <= {TOKEN_MASK}, got {tokens}")

    launch = compile_routing(experts=int(experts), topk=int(topk), tile_m=int(tile_m))
    max_blocks = max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)
    dev = topk_ids.device
    if reuse:
        key = (int(max_blocks), int(tile_m), str(dev))
        ws = _WORKSPACES.get(key)
        if ws is None:
            ws = _WORKSPACES[key] = _new_workspace(max_blocks, int(tile_m), dev)
    else:
        ws = _new_workspace(max_blocks, int(tile_m), dev)

    st = stream if stream is not None else torch.cuda.current_stream()
    launch(topk_ids.contiguous(), ws.sorted_ids, ws.expert_ids, ws.aux, int(tokens), max_blocks, st)
    common = dict(tile_m=int(tile_m), tokens=int(tokens), topk=int(topk), num_blocks_device=ws.aux)
    if not exact:
        return Routing(ws.sorted_ids, ws.expert_ids, max_blocks, exact=False, **common)
    st.synchronize()
    num_blocks = int(ws.aux[0])
    return Routing(ws.sorted_ids[: num_blocks * tile_m], ws.expert_ids[:num_blocks], num_blocks, exact=True, **common)


# ── Grouped GEMM ─────────────────────────────────────────────────────────────

# LDS budget per workgroup on gfx11.
LDS_BYTES = 64 * 1024
# 8 bf16/f16 = one 128-bit GMEM/LDS access.
LOAD_VEC = 8
IN_DTYPES = {"bf16": fx.BFloat16, "f16": fx.Float16}
OUT_DTYPES = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}
STAGES = ("gate_up", "down")
# silu(x) = x * rcp(1 + exp2(x * -log2 e)).
NEG_LOG2E = -1.4426950408889634


def validate_request(*, stage: str, in_dtype: str, out_dtype: str, doweight: bool, topk: int) -> None:
    """Reject what no block tile can fix, so the tile search only has to skip tiles that do not fit."""
    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx11"):
        raise RuntimeError(
            f"rdna3_moe requires gfx11* (RDNA3 / RDNA3.5); current arch is {gpu_arch!r}. "
            "The CDNA MoE kernels live in kernels/moe/moe_gemm_2stage."
        )
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if in_dtype not in IN_DTYPES:
        raise ValueError(f"in_dtype must be one of {sorted(IN_DTYPES)}, got {in_dtype!r}")
    if out_dtype not in OUT_DTYPES:
        raise ValueError(f"out_dtype must be one of {sorted(OUT_DTYPES)}, got {out_dtype!r}")
    if doweight and stage != "down":
        # The weight is per row; before the LDS staging a lane's eight values are eight different rows.
        raise ValueError(f"doweight is only implemented for stage='down', not {stage!r}")
    if int(topk) < 1:
        raise ValueError(f"stage={stage!r} needs the routing topk to address its output rows, got {topk}")


def create_grouped_gemm_module(
    *,
    k_dim: int,
    n_out: int,
    experts: int,
    stage: str,
    topk: int = 1,
    doweight: bool = False,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    reg_m: int = 1,
    reg_n: int = 2,
    reg_k: int = 4,
    waves_m: int = 1,
    waves_n: int = 2,
    bounded_blocks: bool = False,
    m_major: bool = False,
    direct: bool = False,
    g2r: bool = False,
    b_wmma: bool = False,
) -> tuple[Callable, int, int, int]:
    """Build one MoE stage. Returns ``(launch, BLOCK_M, BLOCK_N, BLOCK_K)``.

    ``gate_up``: weight ``[experts, 2*n_out, k_dim]`` gate half first, output ``silu(gate) * up`` as
    ``[tokens, topk, n_out]``; one workgroup runs both B streams over one A tile so a channel's gate and up
    meet in one accumulator. ``down``: A is ``[tokens, topk, k_dim]`` (row ``token*topk + slot``), weight
    ``[experts, n_out, k_dim]``, output ``[tokens, topk, n_out]``, and ``doweight`` multiplies by the routing
    weight.

    A tile belongs to the expert ``expert_ids[tile]`` and gathers its A rows through ``sorted_ids``; the
    result is scattered back by (token, slot). ``launch(out, a, w, sorted_ids, expert_ids, topk_weights,
    tokens, num_blocks, stream, num_blocks_dev=None)``.

    ``bounded_blocks``: ``num_blocks`` may be any upper bound; a workgroup returns if its tile index is not
    below ``num_blocks_dev[0]``. ``m_major``: M tiles innermost in the grid, so an expert's weight slab is
    reused from cache across its M tiles. ``direct``: tile ``t`` is slot ``t`` of the flat ``[tokens, topk]``
    in row 0, and ``expert_ids`` is ``topk_ids`` itself. ``g2r``: a one-wave tile loads its operands from
    global memory straight into the WMMA registers; nothing would be shared through LDS, and the LDS freed is
    what limits workgroups per CU. ``b_wmma``: the weight is in ``wmma_order``, so the 16 lanes of a WMMA
    operand load one contiguous 256 B run instead of 16 rows.

    Buffer offsets are 32-bit, so one A buffer and one output buffer are each limited to 2 GiB. The weight
    is offset to its expert in 64 bits first, so only one expert's slab has to fit.
    """
    validate_request(stage=stage, in_dtype=in_dtype, out_dtype=out_dtype, doweight=doweight, topk=topk)
    if direct and bounded_blocks:
        raise ValueError("a direct build has its exact grid from topk_ids, so direct and bounded_blocks are exclusive")
    if g2r and waves_m * waves_n != 1:
        raise ValueError("g2r needs a tile of one wave")

    K = int(k_dim)
    N = int(n_out)
    TOPK = int(topk)
    ld_a = K
    ld_b = K
    n_b_tiles = 2 if stage == "gate_up" else 1
    w_slab_rows = n_b_tiles * N
    a_rows_per_token = TOPK if stage == "down" else 1

    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    THREADS_PER_BLOCK = waves_m * waves_n * WAVE_SIZE

    if n_b_tiles * reg_m * reg_n < 2:
        # A K loop carrying one value hands it back bare rather than as a sequence of one.
        raise ValueError(
            f"a tile of one accumulator (reg_m={reg_m}, reg_n={reg_n} on {n_b_tiles} B stream(s)) is not supported"
        )
    # A tile may overhang N and K: B is bounded at the expert's n_out rows, A columns past k_dim point past the
    # end of A, and the scatter drops channels past n_out, so the tail reads zero and writes nothing.
    TAIL = bool(N % BLOCK_N or K % BLOCK_K)
    if ld_a % LOAD_VEC or ld_b % LOAD_VEC:
        raise ValueError(f"k_dim={K} must be a multiple of {LOAD_VEC} for the 128-bit copy")
    if b_wmma and N % WMMA_N:
        raise ValueError(f"b_wmma groups the weight by {WMMA_N} rows, so n_out={N} must be a multiple of it")

    num_k_tiles = -(-K // BLOCK_K)
    if num_k_tiles < 2:
        raise ValueError(
            f"the prefetch pipeline needs at least 2 K-tiles; k_dim={K} over BLOCK_K={BLOCK_K} gives {num_k_tiles}"
        )

    # G2S geometry: thread (tk, tm) moves the 128-bit chunk at columns [tk*LOAD_VEC, +LOAD_VEC) of row tm.
    THRS_K = BLOCK_K // LOAD_VEC
    THRS_M = THREADS_PER_BLOCK // THRS_K
    if THRS_K * THRS_M != THREADS_PER_BLOCK:
        raise ValueError(f"BLOCK_K={BLOCK_K} does not divide {THREADS_PER_BLOCK} threads into whole 128-bit chunks")
    if BLOCK_M % THRS_M or BLOCK_N % THRS_M:
        raise ValueError(f"BLOCK_M={BLOCK_M} and BLOCK_N={BLOCK_N} must both be multiples of THRS_M={THRS_M}")
    REPS_M_A = BLOCK_M // THRS_M

    ROW_STRIDE_A = BLOCK_K + K_PAD
    ROW_STRIDE_B = BLOCK_K + K_PAD
    LDS_A_SIZE = BLOCK_M * ROW_STRIDE_A
    LDS_B_SIZE = BLOCK_N * ROW_STRIDE_B
    LDS_ONE_BUF = LDS_A_SIZE + n_b_tiles * LDS_B_SIZE
    if ROW_STRIDE_A % LOAD_VEC:
        raise ValueError(
            f"K_PAD={K_PAD} leaves an LDS row of {ROW_STRIDE_A} elements, "
            f"not a multiple of the {LOAD_VEC}-element vector store"
        )

    elem_dtype = IN_DTYPES[in_dtype]
    out_elem_cls = OUT_DTYPES[out_dtype]
    elem_bytes = elem_dtype.width // 8
    out_bytes = out_elem_cls.width // 8
    acc_size = 8 * reg_m * reg_n

    # The scatter goes through LDS: a gfx11 lane holds D[2*si + lane/16][lane%16], eight rows of one channel,
    # so storing from the accumulator would be eight unrelated 2-byte stores. The staged tile is read back
    # one 128-bit chunk per row; the row pad keeps those reads on different banks.
    SVEC = 128 // out_elem_cls.width
    C_ROW_STRIDE = BLOCK_N + LOAD_VEC
    LANES_N = BLOCK_N // SVEC
    ROWS_PER_PASS = THREADS_PER_BLOCK // LANES_N if LANES_N else 0
    if BLOCK_N % SVEC or THREADS_PER_BLOCK % LANES_N:
        raise ValueError(f"BLOCK_N={BLOCK_N} does not split into {SVEC}-element chunks over the block's threads")
    if ROWS_PER_PASS > BLOCK_M or BLOCK_M % ROWS_PER_PASS:
        raise ValueError(
            f"the CShuffle readback covers {ROWS_PER_PASS} rows per pass, which does not tile BLOCK_M={BLOCK_M}"
        )
    C_REPS_M = BLOCK_M // ROWS_PER_PASS

    # The pipeline buffers and the staging tile are never live at once, so they share one allocation.
    cshuf_elems = -(-BLOCK_M * C_ROW_STRIDE * out_bytes // elem_bytes)
    LDS_TOTAL = cshuf_elems if g2r else max(2 * LDS_ONE_BUF, cshuf_elems)
    if LDS_TOTAL * elem_bytes > LDS_BYTES:
        raise ValueError(
            f"tile needs {LDS_TOTAL * elem_bytes} B of LDS, over the {LDS_BYTES} B a gfx11 workgroup may allocate"
        )

    grid_n = -(-N // BLOCK_N)
    is_bf16 = in_dtype == "bf16"

    def _wmma_op(a_vec, b_vec, acc):
        # gfx11 WMMA takes v16 operands and a v8 accumulator, and spells bf16 as i16.
        if is_bf16:
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_vec.bitcast(fx.Int16), b_vec.bitcast(fx.Int16), acc).result
        return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result

    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_dtype, LDS_TOTAL, 16]

    @flyc.kernel
    def grouped_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_w: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_topk_weights: fx.Tensor,
        arg_num_blocks: fx.Tensor,
        i32_tokens: fx.Int32,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        lds_storage = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = lds_storage.lds.ptr

        def _v8_load(v8_idx):
            elem_off = fx.Int32(v8_idx * 8)
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            return fx.make_view(fx.recast_iter(elem_dtype, ptr_off), fx.make_layout(8, 1)).load()

        tid = gpu.thread_id("x")
        if const_expr(m_major):
            bid_m = fx.Int32(gpu.block_id("x"))
            bid_n = fx.Int32(gpu.block_id("y"))
        else:
            bid_n = fx.Int32(gpu.block_id("x"))
            bid_m = fx.Int32(gpu.block_id("y"))

        tid_i = fx.Int32(tid)
        wave_id = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        # In the gfx11 v16 ABI lanes 16-31 mirror lanes 0-15: a lane's row is lane % 16, with all 16 K values.
        lane16 = lane % 16
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_mma = tiled_mma.thr_slice(tid)

        def _load_scalar(base_iter, idx):
            # Offset the pointer rather than index a view, whose layout would need the runtime length.
            return fx.make_view(base_iter + idx, fx.make_layout(1, 1))[0]

        sid_iter = fx.recast_iter(fx.Int32, fx.get_iter(arg_sorted_ids))
        eid_iter = fx.recast_iter(fx.Int32, fx.get_iter(arg_expert_ids))
        expert_id = _load_scalar(eid_iter, bid_m)
        row_base = bid_m * fx.Int32(BLOCK_M)

        def _routing_row(packed):
            # token*topk + slot: the down stage's A row and both stages' output row.
            token = packed & fx.Int32(TOKEN_MASK)
            slot = packed >> fx.Int32(SLOT_SHIFT)
            return token * fx.Int32(TOPK) + slot

        def _routing_id(row):
            if const_expr(direct):
                own = ((bid_m % fx.Int32(TOPK)) << fx.Int32(SLOT_SHIFT)) | (bid_m // fx.Int32(TOPK))
                pad = (fx.Int32(TOPK) << fx.Int32(SLOT_SHIFT)) | fx.Int32(i32_tokens)
                return (row == fx.Int32(0)).select(own, pad)
            return _load_scalar(sid_iter, row_base + row)

        # A and the output are bounded at their real row count (num_records_bytes makes the descriptor check
        # it), so a padded row's sentinel address is out of range: its loads read zero, its stores are dropped.
        a_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_a), fx.make_layout(1, 1)),
            num_records_bytes=fx.Int64(i32_tokens) * fx.Int64(a_rows_per_token * ld_a * elem_bytes),
        )
        a_iter = fx.get_iter(a_buf)
        a_end = fx.Int32(i32_tokens) * fx.Int32(a_rows_per_token * ld_a)

        tm = tid_i // fx.Int32(THRS_K)
        tk = tid_i % fx.Int32(THRS_K)
        a_row_off = []
        for r in range_constexpr(REPS_M_A):
            packed = _routing_id(fx.Int32(r * THRS_M) + tm)
            if const_expr(stage == "down"):
                a_row = _routing_row(packed)
            else:
                a_row = packed & fx.Int32(TOKEN_MASK)
            a_row_off.append(a_row * fx.Int32(ld_a) + tk * fx.Int32(LOAD_VEC))

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

        w_base = fx.get_iter(arg_w) + fx.Int64(expert_id) * fx.Int64(w_slab_rows * ld_b)

        def _b_stream(row_offset):
            shape = (grid_n * BLOCK_N, num_k_tiles * BLOCK_K)
            view = fx.make_view(w_base + fx.Int64(row_offset), fx.make_layout(shape, (ld_b, 1)))
            bound = {"num_records_bytes": fx.Int64(N * ld_b * elem_bytes)} if TAIL else {}
            return thr_g2s.partition_S(
                fx.flat_divide(fx.rocdl.make_buffer_tensor(view, **bound), fx.make_tile(BLOCK_N, BLOCK_K))[
                    None, None, bid_n, None
                ]
            )

        pB_g = [_b_stream(h * N * ld_b) for h in range_constexpr(n_b_tiles)]

        def _lds_dst(buf_offset, base, rows, row_stride):
            ptr = fx.add_offset(lds_ptr, fx.make_int_tuple(buf_offset + base))
            layout = fx.make_layout((rows, BLOCK_K), (row_stride, 1))
            return thr_g2s.partition_D(fx.make_view(fx.recast_iter(elem_dtype, ptr), layout))[None, None, None]

        def _pA_s(buf_offset):
            return _lds_dst(buf_offset, 0, BLOCK_M, ROW_STRIDE_A)

        def _pB_s(buf_offset, half):
            return _lds_dst(buf_offset, LDS_A_SIZE + half * LDS_B_SIZE, BLOCK_N, ROW_STRIDE_B)

        frag_copy_A = fx.make_fragment_like(_pA_s(0))
        frag_copy_B = [fx.make_fragment_like(_pB_s(0, h)) for h in range_constexpr(n_b_tiles)]

        # One flat view per gathered row: a copy into the sliced rank-2 fragment survives register promotion as
        # a leftover register pointer.
        frag_a_slots = [
            fx.make_view(fx.get_iter(frag_copy_A[None, r, 0]), fx.make_layout(LOAD_VEC, 1))
            for r in range_constexpr(REPS_M_A)
        ]

        # In wmma_order, element (n, c) of a slab is at (n//16)*16*K + (n%16)*8 + c*16 for c a multiple of 8.
        def _wmma_row(n):
            return (n // fx.Int32(WMMA_N)) * fx.Int32(WMMA_N * ld_b) + (n % fx.Int32(WMMA_N)) * fx.Int32(LOAD_VEC)

        w_slab = fx.rocdl.make_buffer_tensor(
            fx.make_view(w_base, fx.make_layout(1, 1)), num_records_bytes=fx.Int64(w_slab_rows * ld_b * elem_bytes)
        )
        reps_n_b = BLOCK_N // THRS_M
        b_row_off = [_wmma_row(bid_n * fx.Int32(BLOCK_N) + fx.Int32(r * THRS_M) + tm) for r in range(reps_n_b)]
        frag_b_slots = [
            [
                fx.make_view(fx.get_iter(frag_copy_B[h][None, r, 0]), fx.make_layout(LOAD_VEC, 1))
                for r in range(reps_n_b)
            ]
            for h in range(n_b_tiles)
        ]

        def _gmem_load(k_tile):
            k_off = fx.Int32(k_tile) * fx.Int32(BLOCK_K)
            for r in range_constexpr(REPS_M_A):
                off = a_row_off[r] + k_off
                if const_expr(K % BLOCK_K):
                    off = (k_off + tk * fx.Int32(LOAD_VEC) < fx.Int32(K)).select(off, a_end)
                mem = fx.make_view(a_iter + off, fx.make_layout(LOAD_VEC, 1))
                fx.copy(buf_copy, mem, frag_a_slots[r])
            for h in range_constexpr(n_b_tiles):
                if const_expr(b_wmma):
                    col = (k_off + tk * fx.Int32(LOAD_VEC)) * fx.Int32(WMMA_N)
                    for r in range_constexpr(reps_n_b):
                        off = fx.Int32(h * N * ld_b) + b_row_off[r] + col
                        fx.copy(
                            buf_copy,
                            fx.make_view(fx.get_iter(w_slab) + off, fx.make_layout(LOAD_VEC, 1)),
                            frag_b_slots[h][r],
                        )
                else:
                    fx.copy(buf_copy, pB_g[h][None, None, None, k_tile], frag_copy_B[h])

        def _lds_store(buf_offset):
            fx.copy(uni_copy, frag_copy_A, _pA_s(buf_offset))
            for h in range_constexpr(n_b_tiles):
                fx.copy(uni_copy, frag_copy_B[h], _pB_s(buf_offset, h))

        _concat16_mask = list(range(16))

        def _load_b_from_lds(rk, buf_offset, half):
            vecs = []
            base = buf_offset + LDS_A_SIZE + half * LDS_B_SIZE
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + 16 * rn + lane16
                lo = _v8_load((base + (row * ROW_STRIDE_B + 16 * rk)) // 8)
                hi = _v8_load((base + (row * ROW_STRIDE_B + (16 * rk + 8))) // 8)
                vecs.append(lo.shuffle(hi, _concat16_mask))
            return vecs

        def _load_a_from_lds(rk, rm_val, buf_offset):
            row = wave_m * (reg_m * WMMA_M) + 16 * rm_val + lane16
            lo = _v8_load((buf_offset + (row * ROW_STRIDE_A + 16 * rk)) // 8)
            hi = _v8_load((buf_offset + (row * ROW_STRIDE_A + (16 * rk + 8))) // 8)
            return lo.shuffle(hi, _concat16_mask)

        def _barrier():
            # gfx11 barrier: the split signal/wait and s_wait_dscnt are gfx12+.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _compute_k_tile(accs_in, buf_offset):
            # One A fragment feeds every B stream; the next A read is issued ahead of its use.
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                b_vecs = [_load_b_from_lds(rk, buf_offset, h) for h in range_constexpr(n_b_tiles)]
                a_next = _load_a_from_lds(rk, 0, buf_offset)
                for rm in range_constexpr(reg_m):
                    a_vec = a_next
                    if const_expr(rm + 1 < reg_m):
                        a_next = _load_a_from_lds(rk, rm + 1, buf_offset)
                    for h in range_constexpr(n_b_tiles):
                        for rn in range_constexpr(reg_n):
                            idx = h * (reg_m * reg_n) + rm * reg_n + rn
                            new_accs[idx] = _wmma_op(a_vec, b_vecs[h][rn], new_accs[idx])
            return new_accs

        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_acc = n_b_tiles * reg_m * reg_n
        c_lds_buf_stride = LDS_ONE_BUF

        # g2r: a lane's operand is 16 contiguous K values of one row, two 128-bit loads. Rows and columns past
        # the shape read zero: A's past its bound, B's past the end of the expert's slab.
        def _g2r_rows():
            lane_row = tid_i % fx.Int32(16)
            b_end = fx.Int32(w_slab_rows * ld_b)
            b_buf = fx.rocdl.make_buffer_tensor(
                fx.make_view(w_base, fx.make_layout(1, 1)), num_records_bytes=fx.Int64(w_slab_rows * ld_b * elem_bytes)
            )
            a_rows = []
            for rm in range_constexpr(reg_m):
                packed = _routing_id(fx.Int32(16 * rm) + lane_row)
                a_row = _routing_row(packed) if const_expr(stage == "down") else packed & fx.Int32(TOKEN_MASK)
                a_rows.append(a_row * fx.Int32(ld_a))
            b_rows = []
            for h in range_constexpr(n_b_tiles):
                for rn in range_constexpr(reg_n):
                    n = bid_n * fx.Int32(BLOCK_N) + fx.Int32(16 * rn) + lane_row
                    off = fx.Int32(h * N * ld_b) + (_wmma_row(n) if const_expr(b_wmma) else n * fx.Int32(ld_b))
                    b_rows.append((n < fx.Int32(N)).select(off, b_end) if const_expr(N % BLOCK_N) else off)
            return (a_iter, a_end, a_rows, 1), (fx.get_iter(b_buf), b_end, b_rows, WMMA_N if b_wmma else 1)

        def _g2r_v16(operand, i, col):
            it, end, rows, col_step = operand
            halves = []
            for half in range_constexpr(2):
                c = col + fx.Int32(8 * half)
                off = rows[i] + c * fx.Int32(col_step)
                if const_expr(K % BLOCK_K):
                    off = (c < fx.Int32(K)).select(off, end)
                frag = fx.make_rmem_tensor(fx.make_layout(LOAD_VEC, 1), elem_dtype)
                fx.copy(buf_copy, fx.make_view(it + off, fx.make_layout(LOAD_VEC, 1)), frag)
                halves.append(frag.load())
            return halves[0].shuffle(halves[1], _concat16_mask)

        def _g2r_accumulate():
            a_op, b_op = _g2r_rows()
            init_state = [zero_acc for _ in range_constexpr(n_acc)]
            for iv, state in range(0, num_k_tiles, 1, init=init_state):
                accs = list(state[:n_acc])
                for rk in range_constexpr(reg_k):
                    col = fx.Int32(iv) * fx.Int32(BLOCK_K) + fx.Int32(16 * rk)
                    b_vecs = [_g2r_v16(b_op, i, col) for i in range_constexpr(n_b_tiles * reg_n)]
                    for rm in range_constexpr(reg_m):
                        a_vec = _g2r_v16(a_op, rm, col)
                        for h in range_constexpr(n_b_tiles):
                            for rn in range_constexpr(reg_n):
                                idx = h * (reg_m * reg_n) + rm * reg_n + rn
                                accs[idx] = _wmma_op(a_vec, b_vecs[h * reg_n + rn], accs[idx])
                results = yield accs
            return list(results[:n_acc])

        def _accumulate():
            if const_expr(g2r):
                return _g2r_accumulate()
            _gmem_load(fx.Int32(0))
            _lds_store(0)
            _barrier()

            init_state = [zero_acc for _ in range_constexpr(n_acc)]
            for iv, state in range(0, num_k_tiles - 1, 1, init=init_state):
                s_accs = list(state[:n_acc])
                _gmem_load(iv + 1)
                s_accs = _compute_k_tile(s_accs, iv % 2 * c_lds_buf_stride)
                _lds_store((1 - iv % 2) * c_lds_buf_stride)
                _barrier()
                results = yield list(s_accs)

            return _compute_k_tile(list(results[:n_acc]), ((num_k_tiles - 1) % 2) * c_lds_buf_stride)

        # frag_C flattens as si + 8*(rm + reg_m*rn), the order this walk over the accumulators produces.
        def _acc_order(accs, half):
            off = half * (reg_m * reg_n)
            return [accs[off + rm * reg_n + rn] for rn in range_constexpr(reg_n) for rm in range_constexpr(reg_m)]

        def _to_out(vals):
            if const_expr(out_elem_cls is fx.Float32):
                return vals
            return [v.to(out_elem_cls) for v in vals]

        def _fill_out_frag(frag_C, out_elems):
            if const_expr(out_elem_cls is fx.Float32):
                frag_out = frag_C
            else:
                frag_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_out.store(
                vector.from_elements(T.vec(acc_size, out_elem_cls.ir_type), [as_ir_value(e) for e in out_elems])
            )
            return frag_out

        def _epilogue_values(accs):
            if const_expr(stage == "gate_up"):
                # Raw exp2: the arguments are well inside the range where the bare hardware op is exact.
                out = []
                for g_acc, u_acc in zip(_acc_order(accs, 0), _acc_order(accs, 1)):
                    for si in range_constexpr(8):
                        g = fx.Float32(g_acc[si])
                        u = fx.Float32(u_acc[si])
                        e = fx.Float32(rocdl.exp2(T.f32, as_mlir_value(g * NEG_LOG2E)))
                        out.append(g * fx.Float32(rocdl.rcp(T.f32, as_mlir_value(fx.Float32(1.0) + e))) * u)
                return _to_out(out)
            return _to_out([acc[si] for acc in _acc_order(accs, 0) for si in range_constexpr(8)])

        def _store_scatter(accs):
            lds_c = fx.make_view(
                fx.recast_iter(out_elem_cls, lds_ptr), fx.make_layout((BLOCK_M, BLOCK_N), (C_ROW_STRIDE, 1))
            )
            cshuf_atom = fx.make_copy_atom(fx.UniversalCopy(out_elem_cls.width), out_elem_cls)
            thr_cshuf = fx.make_tiled_copy_C(cshuf_atom, tiled_mma).get_slice(tid)
            frag_out = _fill_out_frag(thr_mma.make_fragment_C(lds_c), _epilogue_values(accs))

            # The last _compute_k_tile may still be reading the buffer this overwrites.
            _barrier()
            fx.copy(cshuf_atom, thr_cshuf.retile(frag_out), thr_cshuf.partition_D(lds_c))
            _barrier()

            out_buf = fx.rocdl.make_buffer_tensor(
                fx.make_view(fx.get_iter(arg_c), fx.make_layout(1, 1)),
                num_records_bytes=fx.Int64(i32_tokens) * fx.Int64(TOPK * N * out_bytes),
            )
            out_iter = fx.get_iter(out_buf)
            lds_c_ptr = fx.recast_iter(out_elem_cls, lds_ptr)
            uni_copy_out = fx.make_copy_atom(fx.UniversalCopy128b(), out_elem_cls)
            buf_copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), out_elem_cls)
            if const_expr(doweight):
                wt_iter = fx.recast_iter(fx.Float32, fx.get_iter(arg_topk_weights))
                n_rows = fx.Int32(i32_tokens) * fx.Int32(TOPK)

            chan = (tid_i % fx.Int32(LANES_N)) * fx.Int32(SVEC)
            row_in_pass = tid_i // fx.Int32(LANES_N)
            for rep in range_constexpr(C_REPS_M):
                row = row_in_pass + fx.Int32(rep * ROWS_PER_PASS)
                out_row = _routing_row(_routing_id(row))
                if const_expr(N % BLOCK_N):
                    col_ok = bid_n * fx.Int32(BLOCK_N) + chan < fx.Int32(N)
                    out_row = col_ok.select(out_row, fx.Int32(i32_tokens) * fx.Int32(TOPK))

                src = fx.make_view(
                    fx.add_offset(lds_c_ptr, fx.make_int_tuple(row * fx.Int32(C_ROW_STRIDE) + chan)),
                    fx.make_layout(SVEC, 1),
                )
                staged = fx.make_fragment_like(src)
                fx.copy(uni_copy_out, src, staged)

                if const_expr(doweight):
                    # gfx11 has no bf16 atomic, so the topk sum is a separate unweighted pass and the routing
                    # weight lands here, once per row. A padded row reads weight 0's slot; its store is dropped.
                    safe_row = (out_row < n_rows).select(out_row, fx.Int32(0))
                    w = fx.Float32(_load_scalar(wt_iter, safe_row))
                    cur = staged.load()
                    scaled = _to_out([fx.Float32(cur[i]) * w for i in range_constexpr(SVEC)])
                    staged.store(
                        vector.from_elements(T.vec(SVEC, out_elem_cls.ir_type), [as_ir_value(v) for v in scaled])
                    )

                dst = fx.make_view(
                    out_iter + (out_row * fx.Int32(N) + bid_n * fx.Int32(BLOCK_N) + chan), fx.make_layout(SVEC, 1)
                )
                fx.copy(buf_copy_out, staged, dst)

        if const_expr(bounded_blocks):
            # The block index is uniform across the workgroup, so the barriers inside stay matched.
            nb_iter = fx.recast_iter(fx.Int32, fx.get_iter(arg_num_blocks))
            if bid_m < _load_scalar(nb_iter, fx.Int32(0)):
                _store_scatter(_accumulate())
        else:
            _store_scatter(_accumulate())

    @flyc.jit
    def launch_grouped_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_w: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_topk_weights: fx.Tensor,
        arg_num_blocks: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_num_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        # The permutation keeps wave wm on the contiguous row band [wm*reg_m*16, +reg_m*16).
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32)),
            fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)),
            permutation=(
                fx.make_layout((WMMA_M, waves_m, reg_m), (1, WMMA_M * reg_m, WMMA_M)),
                fx.make_layout((WMMA_N, waves_n, reg_n), (1, WMMA_N * reg_n, WMMA_N)),
                WMMA_K,
            ),
        )
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype),
            fx.make_layout(((THRS_K, THRS_M), (1, LOAD_VEC)), ((THRS_M * LOAD_VEC, 1), (1, THRS_M))),
            fx.make_tile(THRS_M, BLOCK_K),
        )
        grouped_gemm_kernel(
            arg_c,
            arg_a,
            arg_w,
            arg_sorted_ids,
            arg_expert_ids,
            arg_topk_weights,
            arg_num_blocks,
            i32_tokens,
            tiled_mma,
            tiled_copy_g2s,
        ).launch(
            grid=(fx.Int64(i32_num_blocks), grid_n, 1) if m_major else (grid_n, fx.Int64(i32_num_blocks), 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    def launch(
        arg_c,
        arg_a,
        arg_w,
        arg_sorted_ids,
        arg_expert_ids,
        arg_topk_weights,
        tokens,
        num_blocks,
        stream,
        num_blocks_dev=None,
    ):
        if int(num_blocks) <= 0:
            return None
        if bounded_blocks and num_blocks_dev is None:
            raise ValueError("this build reads its tile count from the device, so num_blocks_dev is required")
        nb_dev = num_blocks_dev if num_blocks_dev is not None else arg_expert_ids
        # Through the CompiledFunction: the jit launcher's own call path costs more than a decode-sized kernel.
        return _run_compiled(
            launch_grouped_gemm,
            arg_c,
            arg_a,
            arg_w,
            arg_sorted_ids,
            arg_expert_ids,
            arg_topk_weights,
            nb_dev,
            fx.Int32(int(tokens)),
            fx.Int32(int(num_blocks)),
            stream,
        )

    launch.threads = THREADS_PER_BLOCK
    launch.m_major = m_major
    launch.direct = direct
    launch.g2r = g2r
    launch.b_wmma = b_wmma
    return launch, BLOCK_M, BLOCK_N, BLOCK_K


def wmma_order(w: torch.Tensor) -> torch.Tensor:
    """``[experts, rows, k]`` -> the order a ``b_wmma`` build reads: each 16 rows x 8 columns contiguous."""
    e, n, k = w.shape
    return w.view(e, n // WMMA_N, WMMA_N, k // LOAD_VEC, LOAD_VEC).transpose(2, 3).contiguous().view(e, n, k)


# ── token_down: the down projection and the topk sum in one GEMV ─────────────

_TD_THREADS = 256
# Lanes per output channel: a power of two no wider than a wave, so the butterfly stays in one wave.
_TD_LANES = 16
_TD_CHANNELS = _TD_THREADS // _TD_LANES
_INFINITY_CACHE_BYTES = 96 << 20


def token_down_fits(model_dim: int, inter_dim: int) -> bool:
    return int(inter_dim) % LOAD_VEC == 0 and int(model_dim) % _TD_CHANNELS == 0


@functools.lru_cache(maxsize=64)
def compile_token_down(*, model_dim: int, inter_dim: int, experts: int, topk: int, in_dtype: str = "bf16") -> Callable:
    """``out[t] = sum_k topk_weights[t, k] * w2[topk_ids[t, k]] @ a2[t, k]`` in f32, rounded once.

    Workgroup ``(token, bid_n)`` owns ``_TD_CHANNELS`` output channels; ``_TD_LANES`` lanes split a channel's
    ``inter`` in 128-bit chunks slot by slot and meet in a butterfly, so the sum order is fixed. Tokens go
    innermost when ``w2`` overflows the Infinity Cache, so two tokens on one expert share its rows in L2.
    ``launch(out, a2, w2, topk_ids, topk_weights, tokens, stream)``.
    """
    N, K, TOPK = int(model_dim), int(inter_dim), int(topk)
    if not token_down_fits(N, K):
        raise ValueError(
            f"token_down needs inter_dim % {LOAD_VEC} == 0 and model_dim % {_TD_CHANNELS} == 0, got {K}, {N}"
        )
    LANES, CHANNELS = _TD_LANES, _TD_CHANNELS
    CHUNKS = K // LOAD_VEC
    STEPS = -(-CHUNKS // LANES)
    elem = IN_DTYPES[in_dtype]
    tokens_inner = int(experts) * N * K * (elem.width // 8) > _INFINITY_CACHE_BYTES

    def _global(ir_type, ptr, align):
        pt = fx.PointerType.get(ir_type, address_space=fx.AddressSpace.Global, alignment=align)
        return fx.inttoptr(pt, fx.Int64(ptrtoint(ptr)))

    @flyc.kernel(known_block_size=[_TD_THREADS, 1, 1])
    def token_down_kernel(Out: fx.Pointer, A: fx.Pointer, W: fx.Pointer, Ids: fx.Pointer, Wts: fx.Pointer):
        vec_in, vec_f32 = T.vec(LOAD_VEC, elem.ir_type), T.vec(LOAD_VEC, T.f32)
        tid = fx.Int32(gpu.thread_id("x"))
        token = fx.Int32(gpu.block_id("x" if tokens_inner else "y"))
        lane = tid % fx.Int32(LANES)
        n = fx.Int32(gpu.block_id("y" if tokens_inner else "x")) * fx.Int32(CHANNELS) + tid // fx.Int32(LANES)
        a_ptr, w_ptr = _global(elem.ir_type, A, 16), _global(elem.ir_type, W, 16)
        ids, wts = _global(T.i32, Ids, 4), _global(T.f32, Wts, 4)
        acc = fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            slot = token * fx.Int32(TOPK) + fx.Int32(k)
            w_row = (fx.Int64(fx.Int32(ids[slot])) * fx.Int64(N) + fx.Int64(n)) * fx.Int64(K)
            part = fx.full(LOAD_VEC, 0.0, fx.Float32)
            for s in range_constexpr(STEPS):
                c = lane + fx.Int32(s * LANES)
                if const_expr((s + 1) * LANES > CHUNKS):
                    live = c < fx.Int32(CHUNKS)
                    c = live.select(c, fx.Int32(0))
                a_off = fx.Int64(slot * fx.Int32(K) + c * fx.Int32(LOAD_VEC))
                a = fx.Vector(fx.ptr_load(a_ptr + a_off, vec_in)).extf(vec_f32)
                b = fx.Vector(fx.ptr_load(w_ptr + (w_row + fx.Int64(c * fx.Int32(LOAD_VEC))), vec_in)).extf(vec_f32)
                if const_expr((s + 1) * LANES > CHUNKS):
                    part = part + live.select(a * b, fx.full(LOAD_VEC, 0.0, fx.Float32))
                else:
                    part = part + a * b
            acc = acc + part.reduce("add") * fx.Float32(wts[slot])
        for off in (LANES // 2, LANES // 4, LANES // 8, LANES // 16):
            acc = acc + acc.bitcast(fx.Int32).shuffle_xor(fx.Int32(off), fx.Int32(32)).bitcast(fx.Float32)
        if lane == fx.Int32(0):
            fx.ptr_store(acc.to(elem), _global(elem.ir_type, Out, 2) + fx.Int64(token * fx.Int32(N) + n))

    @flyc.jit
    def launch_token_down(
        Out: fx.Pointer,
        A: fx.Pointer,
        W: fx.Pointer,
        Ids: fx.Pointer,
        Wts: fx.Pointer,
        i32_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        tokens = fx.Int64(i32_tokens)
        token_down_kernel(Out, A, W, Ids, Wts).launch(
            grid=(tokens, N // CHANNELS, 1) if tokens_inner else (N // CHANNELS, tokens, 1),
            block=(_TD_THREADS, 1, 1),
            stream=stream,
        )

    def launch(out, a2, w2, topk_ids, topk_weights, tokens, stream):
        ptrs = [flyc.from_c_void_p(fx.Uint8, t.data_ptr()) for t in (out, a2, w2, topk_ids, topk_weights)]
        return _run_compiled(launch_token_down, *ptrs, fx.Int32(int(tokens)), stream)

    launch.tokens_inner = tokens_inner
    return launch


# ── Tile choice ──────────────────────────────────────────────────────────────

Tile = tuple[int, int, int, int, int]  # (reg_m, reg_n, reg_k, waves_m, waves_n), optionally followed by "g2r"

# Block tiles, most specific key first: (stage, tile_m, k_dim, n_out, bucket), (stage, tile_m, k_dim, n_out),
# (stage, tile_m, bucket), (stage, tile_m); then every tile of the height (``_narrowings``). For gate_up,
# ``n_out`` is inter_dim. gate_up carries an accumulator set per B stream, so its N width comes from waves
# rather than registers (gfx11 caps a thread at 256 VGPRs). BLOCK_K=32 lets two workgroups share a CU's LDS.
_TILES: dict[tuple, tuple[Tile, ...]] = {
    ("gate_up", 16): ((1, 2, 4, 1, 2), (1, 1, 4, 1, 2), (1, 2, 2, 1, 2), (1, 1, 2, 1, 2)),  # 16x64x64
    ("gate_up", 32): ((2, 1, 4, 1, 2), (2, 1, 2, 1, 2), (2, 1, 2, 1, 1)),  # 32x32x64
    ("gate_up", 64): ((2, 1, 4, 2, 2), (2, 1, 2, 2, 2), (2, 1, 2, 2, 1)),  # 64x32x64
    ("gate_up", 128): ((4, 1, 2, 2, 2), (2, 1, 2, 4, 2), (4, 1, 2, 2, 1)),  # 128x32x32
    ("down", 16): ((1, 2, 2, 1, 2), (1, 2, 2, 1, 1)),  # 16x64x32
    ("down", 32): ((2, 2, 2, 1, 4), (2, 2, 2, 1, 2), (2, 2, 2, 1, 1)),  # 32x128x32
    ("down", 64): ((2, 2, 2, 2, 4), (2, 2, 2, 2, 2), (2, 2, 2, 2, 1)),  # 64x128x32
    ("down", 128): ((2, 2, 2, 4, 2), (2, 2, 2, 4, 1)),  # 128x64x32
    ("down", 16, "partial"): ((1, 2, 4, 1, 2),),  # 16x64x64
    ("down", 16, "few"): ((1, 2, 4, 1, 2),),  # 16x64x64
    ("down", 32, "partial"): ((2, 2, 4, 1, 4),),  # 32x128x64
    ("down", 32, "few"): ((2, 2, 4, 1, 4),),  # 32x128x64
    ("down", 64, "partial"): ((2, 2, 4, 2, 4),),  # 64x128x64
    ("down", 64, "few"): ((2, 2, 4, 2, 4),),  # 64x128x64
    ("gate_up", 128, 4096, 14336): ((4, 2, 2, 2, 2),),  # 128x64x32
    ("down", 128, 14336, 4096): ((4, 4, 2, 2, 2),),  # 128x128x32
    ("gate_up", 16, 2560, 80): ((1, 1, 8, 1, 1),),  # 16x16x128
    ("down", 16, 80, 2560): ((1, 4, 1, 1, 1),),  # 16x64x16
    ("gate_up", 16, 4096, 1792): ((1, 1, 4, 1, 1, "g2r"),),  # 16x16x64
    ("gate_up", 32, 4096, 1792): ((2, 1, 8, 1, 1, "g2r"),),  # 32x16x128
    ("gate_up", 128, 4096, 1792): ((2, 2, 2, 4, 4),),  # 128x128x32
    ("down", 16, 1792, 4096): ((1, 2, 4, 1, 1),),  # 16x32x64
    ("down", 32, 1792, 4096): ((2, 1, 8, 1, 1),),  # 32x16x128
    ("down", 128, 1792, 4096): ((4, 4, 2, 2, 2),),  # 128x128x32
    ("gate_up", 16, 4096, 176): ((1, 1, 8, 1, 1),),  # 16x16x128
    ("gate_up", 64, 4096, 176): ((1, 1, 4, 4, 1),),  # 64x16x64
    ("down", 16, 176, 4096): ((1, 2, 2, 1, 1),),  # 16x32x32, K tail of 16
    ("down", 64, 176, 4096): ((2, 4, 1, 2, 2),),  # 64x128x16
    ("gate_up", 16, 2048, 96): ((1, 1, 8, 1, 1),),  # 16x16x128
    ("gate_up", 64, 2560, 80): ((2, 1, 4, 2, 2),),  # 64x32x64, N tail of 16
    ("gate_up", 16, 2048, 96, "partial"): ((1, 1, 4, 1, 1, "g2r"),),  # 16x16x64
    ("gate_up", 16, 4096, 176, "few"): ((1, 2, 4, 1, 1, "g2r"),),  # 16x32x64, N tail of 16
    ("gate_up", 64, 4096, 176, "partial"): ((2, 1, 4, 2, 2),),  # 64x32x64, N tail of 16
    ("gate_up", 64, 4096, 176, "few"): ((2, 1, 4, 2, 2),),  # 64x32x64, N tail of 16
    ("gate_up", 64, 4096, 176, "many"): ((4, 1, 2, 1, 4),),  # 64x64x32, N tail of 48
    ("down", 16, 80, 2560, "few"): ((1, 2, 1, 1, 1, "g2r"),),  # 16x32x16
    ("down", 16, 96, 2048, "few"): ((1, 2, 2, 1, 1, "g2r"),),  # 16x32x32
    ("down", 64, 1792, 4096, "few"): ((2, 2, 2, 2, 4),),  # 64x128x32
}

# Shapes whose N sweep is long enough to evict an expert's B slab before its next M tile asks for it.
_M_MAJOR = {("gate_up", 4096, 14336), ("gate_up", 4096, 1792)}

SUPPORTED_TILE_M = (16, 32, 64, 128)
_BUCKETS = ("partial", "few", "many")


def tile_bucket_for(rows_per_expert: float, tile_m: int) -> str:
    """How many M tiles the average expert owns: ``partial`` (at most one), ``few`` (up to four), ``many``."""
    tiles = float(rows_per_expert) / float(tile_m)
    if tiles <= 1.0:
        return "partial"
    if tiles <= 4.0:
        return "few"
    return "many"


# One expert's bf16 gate, up and down weights against the 6 MB L2. Where they fit, an expert's second M tile
# re-reads them from L2; where they do not, it streams them again, so taller tiles pay off from far fewer rows.
_L2_BYTES = 6 << 20
# Rows per expert from which the 32, 64 and 128 tiles take over.
_NARROW_EXPERT_ROWS = (48, 48, math.inf)
_WIDE_EXPERT_ROWS = (2, 33, 512)


def tile_m_for(*, tokens: int, topk: int, experts: int, inter: int = 0, hidden: int = 0) -> int:
    """The routing tile height for a layer; ``inter``/``hidden`` left at 0 take the expert to fit in L2."""
    rows_per_expert = tokens * topk / max(int(experts), 1)
    wide = 3 * inter * hidden * 2 > _L2_BYTES
    t32, t64, t128 = _WIDE_EXPERT_ROWS if wide else _NARROW_EXPERT_ROWS
    if rows_per_expert < t32:
        return 16
    if rows_per_expert < t64:
        return 32
    if rows_per_expert < t128:
        return 64
    return 128


@functools.lru_cache(maxsize=64)
def _narrowings(tile_m: int) -> tuple[Tile, ...]:
    """Every block tile of this height, widest BLOCK_N first, then deepest BLOCK_K, then most waves in N."""
    out = []
    for waves_m in (1, 2, 4, 8):
        if tile_m % (16 * waves_m) or tile_m // (16 * waves_m) not in (8, 4, 2, 1):
            continue
        reg_m = tile_m // (16 * waves_m)
        for waves_n in (1, 2, 4, 8):
            if waves_m * waves_n > 8:
                continue
            for reg_n in (8, 4, 2, 1):
                for reg_k in (4, 2, 1):
                    out.append((reg_m, reg_n, reg_k, waves_m, waves_n))
    out.sort(key=lambda c: (-(c[1] * c[4]), -c[2], -c[4]))
    return tuple(dict.fromkeys(out))


def _candidates(stage: str, k_dim: int, n_out: int, tile_m: int, bucket: str | None) -> list[Tile]:
    ranked = [
        _TILES.get((stage, tile_m, k_dim, n_out, bucket)),
        _TILES.get((stage, tile_m, k_dim, n_out)),
        _TILES.get((stage, tile_m, bucket)) if bucket else None,
        _TILES[(stage, tile_m)],
        _narrowings(tile_m),
    ]
    return list(dict.fromkeys(cfg for table in ranked for cfg in table or ()))


@functools.lru_cache(maxsize=256)
def _build(
    *,
    k_dim: int,
    n_out: int,
    experts: int,
    stage: str,
    topk: int,
    doweight: bool,
    tile_m: int,
    in_dtype: str,
    out_dtype: str,
    bounded_blocks: bool,
    bucket: str | None,
    direct: bool,
) -> tuple[Callable, int, int, int]:
    validate_request(stage=stage, in_dtype=in_dtype, out_dtype=out_dtype, doweight=doweight, topk=topk)
    tried = []
    measured = _TILES.get((stage, tile_m, k_dim, n_out), ()) + _TILES.get((stage, tile_m, k_dim, n_out, bucket), ())
    for cfg in _candidates(stage, k_dim, n_out, tile_m, bucket):
        reg_m, reg_n, reg_k, waves_m, waves_n, *flags = cfg
        # Only a measured shape entry may overhang the shape; the rest of a tile's limits are the builder's.
        if cfg not in measured and (n_out % (16 * reg_n * waves_n) or k_dim % (16 * reg_k)):
            continue
        try:
            launch, block_m, block_n, block_k = create_grouped_gemm_module(
                k_dim=k_dim,
                n_out=n_out,
                experts=experts,
                stage=stage,
                topk=topk,
                doweight=doweight,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                reg_m=reg_m,
                reg_n=reg_n,
                reg_k=reg_k,
                waves_m=waves_m,
                waves_n=waves_n,
                bounded_blocks=bounded_blocks,
                m_major=(stage, k_dim, n_out) in _M_MAJOR,
                direct=direct,
                g2r="g2r" in flags,
                b_wmma=stage == "gate_up",
            )
        except ValueError as exc:
            tried.append(f"{(reg_m, reg_n, reg_k, waves_m, waves_n)}: {exc}")
            continue
        assert block_m == tile_m, f"tile table disagrees with the kernel: {block_m} != {tile_m}"
        return launch, block_m, block_n, block_k

    detail = "\n  ".join(tried) if tried else "no candidate divided the shape"
    raise ValueError(f"no tile for stage={stage!r} at k_dim={k_dim}, n_out={n_out}, tile_m={tile_m}:\n  {detail}")


def compile_grouped_gemm(
    *,
    k_dim: int,
    n_out: int,
    experts: int,
    stage: str,
    topk: int = 1,
    doweight: bool = False,
    tile_m: int = 16,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    bounded_blocks: bool = False,
    rows_per_expert: float | None = None,
    direct: bool = False,
) -> tuple[Callable, int, int, int]:
    """Build (and cache) one stage at the routing's ``tile_m``; returns ``(launch, BLOCK_M, BLOCK_N, BLOCK_K)``.

    ``rows_per_expert`` (the average) picks a bucket before the cache key is formed; ``direct`` tiles skip the
    bucket entries, which are measured on routed tiles.
    """
    if tile_m not in SUPPORTED_TILE_M:
        raise ValueError(f"tile_m must be one of {SUPPORTED_TILE_M}, got {tile_m}")
    return _build(
        k_dim=int(k_dim),
        n_out=int(n_out),
        experts=int(experts),
        stage=stage,
        topk=int(topk),
        doweight=bool(doweight),
        tile_m=int(tile_m),
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        bounded_blocks=bool(bounded_blocks),
        bucket=None if rows_per_expert is None or direct else tile_bucket_for(rows_per_expert, tile_m),
        direct=bool(direct),
    )


def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    bounded_blocks: bool = False,
    rows_per_expert: float | None = None,
    direct: bool = False,
) -> tuple[Callable, int, int, int]:
    """Stage1, ``silu(gate) * up`` into ``[tokens, topk, inter_dim]``; ``w1`` is ``wmma_order`` of
    ``[experts, 2*inter, model]``."""
    return compile_grouped_gemm(
        k_dim=model_dim,
        n_out=inter_dim,
        experts=experts,
        stage="gate_up",
        topk=topk,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        bounded_blocks=bounded_blocks,
        rows_per_expert=rows_per_expert,
        direct=direct,
    )


def compile_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    doweight: bool = True,
    tile_m: int = 16,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    bounded_blocks: bool = False,
    rows_per_expert: float | None = None,
) -> tuple[Callable, int, int, int]:
    """Stage2 into ``[tokens, topk, model_dim]``; ``w2`` is ``[experts, model, inter]``."""
    return compile_grouped_gemm(
        k_dim=inter_dim,
        n_out=model_dim,
        experts=experts,
        stage="down",
        topk=topk,
        doweight=doweight,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        bounded_blocks=bounded_blocks,
        rows_per_expert=rows_per_expert,
    )


# ── The layer ────────────────────────────────────────────────────────────────

_DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
# token_down reads w2 once per slot and still beats gemm2 + reduce up to two rows an expert; the direct gemm1
# also reads w1 once per slot, which stops paying for the routing launch it saves past 0.6 rows an expert.
TOKEN_DOWN_ROWS_PER_EXPERT = 2.0
DIRECT_ROWS_PER_EXPERT = 0.6


@functools.lru_cache(maxsize=64)
def _reduce_exe(topk: int, model_dim: int, dtype_str: str):
    return compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)


@functools.lru_cache(maxsize=64)
def _gating_launch(experts: int, topk: int, dtype_str: str, renormalize: bool):
    return build_topk_gating_softmax_module(experts, topk, dtype_str, renormalize)


def _ptr(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


@_on_device_of_first_arg
def moe_gating(
    gating_logits: torch.Tensor, *, topk: int, renormalize: bool = True, stream=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[tokens, experts]`` logits -> ``(topk_ids int32, topk_weights f32)``.

    The fused softmax + top-K reduces in a lane group of ``experts // VPT`` lanes, which has to be a power of
    two no wider than a wave; on wave32 that rules out some expert counts.
    """
    if gating_logits.dim() != 2:
        raise ValueError(f"gating_logits must be [tokens, experts], got {tuple(gating_logits.shape)}")
    tokens, experts = gating_logits.shape
    dtype_str = _DTYPES.get(gating_logits.dtype)
    if dtype_str is None:
        raise ValueError(f"gating_logits dtype must be bf16/f16/f32, got {gating_logits.dtype}")

    launch = _gating_launch(int(experts), int(topk), dtype_str, bool(renormalize))
    dev = gating_logits.device
    weights = torch.empty(tokens, topk, dtype=torch.float32, device=dev)
    ids = torch.empty(tokens, topk, dtype=torch.int32, device=dev)
    token_expert_indices = torch.empty(tokens, topk, dtype=torch.int32, device=dev)
    st = stream if stream is not None else torch.cuda.current_stream()
    _run_compiled(launch, gating_logits.contiguous(), weights, ids, token_expert_indices, fx.Int32(int(tokens)), st)
    return ids, weights


@_on_device_of_first_arg
def moe_reduce(y: torch.Tensor, *, out: torch.Tensor | None = None, stream=None) -> torch.Tensor:
    """Unweighted sum ``[tokens, topk, model_dim] -> [tokens, model_dim]``."""
    if y.dim() != 3:
        raise ValueError(f"y must be [tokens, topk, model_dim], got {tuple(y.shape)}")
    tokens, topk, model_dim = y.shape
    dtype_str = _DTYPES.get(y.dtype)
    if dtype_str is None:
        raise ValueError(f"y dtype must be bf16/f16/f32, got {y.dtype}")

    if out is None:
        out = torch.empty(tokens, model_dim, dtype=y.dtype, device=y.device)
    exe = _reduce_exe(int(topk), int(model_dim), dtype_str)
    # The expert mask and routing ids are for EP, where a slot can be invalid; neither is read here.
    unused = torch.empty(0, dtype=torch.int32, device=y.device)
    st = stream if stream is not None else torch.cuda.current_stream()
    _run_compiled(exe, _ptr(y), _ptr(out), _ptr(unused), _ptr(unused), int(tokens), st)
    return out


@_on_device_of_first_arg
def moe_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tile_m: int | None = None,
    routing: Routing | None = None,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """One MoE layer from a routing that is already decided.

    ``x`` is ``[tokens, model_dim]``, ``w1`` ``[experts, 2*inter_dim, model_dim]`` gate half first and put
    through ``wmma_order`` once at load, ``w2`` ``[experts, model_dim, inter_dim]``, all contiguous and of one
    dtype, and ``topk_weights`` f32.
    ``topk_ids`` must lie in ``[0, experts)``: the kernels do not check them. ``tile_m`` defaults to
    ``tile_m_for`` on these shapes. ``routing`` reuses one already built for this ``topk_ids`` and ``tile_m``;
    otherwise ``build_routing_device`` builds one without waiting for its tile count.
    """
    if x.dim() != 2 or topk_ids.dim() != 2:
        raise ValueError(f"x and topk_ids must be 2-D, got x={tuple(x.shape)} topk_ids={tuple(topk_ids.shape)}")
    if w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(f"weights must be 3-D, got w1={tuple(w1.shape)} w2={tuple(w2.shape)}")
    in_dtype = _DTYPES.get(x.dtype)
    if in_dtype not in ("bf16", "f16") or w1.dtype != x.dtype or w2.dtype != x.dtype:
        raise ValueError(f"x, w1 and w2 must all be bf16 or all f16, got {x.dtype}, {w1.dtype}, {w2.dtype}")
    if not (x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()):
        raise ValueError("x, w1 and w2 must be contiguous: the kernels address them with their logical strides")

    tokens, model_dim = x.shape
    experts, two_inter, w1_k = w1.shape
    inter_dim = two_inter // 2
    topk = topk_ids.shape[1]

    if two_inter != 2 * inter_dim:
        raise ValueError(f"w1's row count must be 2*inter_dim, got {two_inter}")
    if w1_k != model_dim:
        raise ValueError(f"w1 contracts {w1_k}, but x has model_dim={model_dim}")
    if tuple(w2.shape) != (experts, model_dim, inter_dim):
        raise ValueError(f"w2 must be {(experts, model_dim, inter_dim)}, got {tuple(w2.shape)}")
    if topk_ids.shape[0] != tokens or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"routing must be [tokens={tokens}, topk], got ids={tuple(topk_ids.shape)} "
            f"weights={tuple(topk_weights.shape)}"
        )
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"topk_weights must be f32, got {topk_weights.dtype}")
    if tile_m is None:
        tile_m = (
            routing.tile_m
            if routing is not None
            else tile_m_for(tokens=tokens, topk=topk, experts=experts, inter=inter_dim, hidden=model_dim)
        )
    if tile_m not in SUPPORTED_TILE_M:
        raise ValueError(f"tile_m must be one of {SUPPORTED_TILE_M}, got {tile_m}")

    rows_per_expert = tokens * topk / max(experts, 1)
    down_in_registers = rows_per_expert <= TOKEN_DOWN_ROWS_PER_EXPERT and token_down_fits(model_dim, inter_dim)
    direct = routing is None and down_in_registers and rows_per_expert <= DIRECT_ROWS_PER_EXPERT
    ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    if routing is None and not direct:
        routing = build_routing_device(ids, experts=experts, tile_m=tile_m, stream=stream)
    elif routing is not None and (routing.tile_m, routing.tokens, routing.topk) != (tile_m, tokens, topk):
        raise ValueError(
            f"the routing was built for tile_m={routing.tile_m}, tokens={routing.tokens}, "
            f"topk={routing.topk}, not tile_m={tile_m}, tokens={tokens}, topk={topk}"
        )

    bounded = not direct and not routing.exact
    gemm1, _, _, _ = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=16 if direct else tile_m,
        in_dtype=in_dtype,
        out_dtype=in_dtype,
        bounded_blocks=bounded,
        rows_per_expert=rows_per_expert,
        direct=direct,
    )
    if stream is None:
        stream = torch.cuda.current_stream()
    # Every (token, slot) is in the routing exactly once, so a2 and y are fully written.
    a2 = torch.empty(tokens, topk, inter_dim, dtype=x.dtype, device=x.device)
    unused = torch.empty(0, dtype=torch.float32, device=x.device)
    if direct:
        gemm1(a2, x, w1, ids, ids, unused, tokens, tokens * topk, stream)
    else:
        nb_dev = routing.num_blocks_device if bounded else None
        gemm1(a2, x, w1, routing.sorted_ids, routing.expert_ids, unused, tokens, routing.num_blocks, stream, nb_dev)
    if down_in_registers:
        down = compile_token_down(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, in_dtype=in_dtype
        )
        if out is None:
            out = torch.empty(tokens, model_dim, dtype=x.dtype, device=x.device)
        down(out, a2, w2, ids, topk_weights, tokens, stream)
        return out
    gemm2, _, _, _ = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        doweight=True,
        tile_m=tile_m,
        in_dtype=in_dtype,
        out_dtype=in_dtype,
        bounded_blocks=bounded,
        rows_per_expert=rows_per_expert,
    )
    y = torch.empty(tokens, topk, model_dim, dtype=x.dtype, device=x.device)
    gemm2(y, a2, w2, routing.sorted_ids, routing.expert_ids, topk_weights, tokens, routing.num_blocks, stream, nb_dev)
    return moe_reduce(y, out=out, stream=stream)


def moe_forward_from_logits(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_logits: torch.Tensor,
    *,
    topk: int,
    tile_m: int | None = None,
    renormalize: bool = True,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """``moe_forward`` with the gating in front of it."""
    topk_ids, topk_weights = moe_gating(gating_logits, topk=topk, renormalize=renormalize, stream=stream)
    return moe_forward(x, w1, w2, topk_ids, topk_weights, tile_m=tile_m, out=out, stream=stream)
