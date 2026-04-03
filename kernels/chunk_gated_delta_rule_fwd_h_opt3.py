"""Specialized K5 opt3 implementation for FlyDSL.

This module keeps three layers side by side:

1. TTGIR-derived thread/layout mapping helpers used to validate the recovered
   CTA decomposition.
2. A Python/Torch reference implementation that mirrors the specialized Triton
   `opt3` semantics exactly.
3. A FlyDSL kernel path that preserves the same specialized host contract while
   expressing the computation through `@flyc.kernel` / `@flyc.jit`.

The FlyDSL path is intentionally scoped to the cached TTGIR specialization:
- `B = 1`
- `H = 8`
- `Hg = 2`
- `K = 128`
- `V = 128`
- `BT = 64`
- `BV = 16`
- `wu_contiguous = True`
- variable-length batching is enabled
- `g` and `initial_state` are required
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Iterable

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


WARP_SIZE = 64
BLOCK_THREADS = 256

H = 8
HG = 2
K = 128
V = 128
BT = 64
BV = 16


@dataclass(frozen=True)
class ThreadCoord:
    row: int
    col: int


@dataclass(frozen=True)
class BlockedLayoutSpec:
    size_per_thread: tuple[int, int]
    threads_per_warp: tuple[int, int]
    warps_per_cta: tuple[int, int]
    order: tuple[int, int]

    @property
    def shape_per_warp(self) -> tuple[int, int]:
        return (
            self.size_per_thread[0] * self.threads_per_warp[0],
            self.size_per_thread[1] * self.threads_per_warp[1],
        )

    @property
    def shape_per_cta(self) -> tuple[int, int]:
        return (
            self.shape_per_warp[0] * self.warps_per_cta[0],
            self.shape_per_warp[1] * self.warps_per_cta[1],
        )


BLOCKED_K = BlockedLayoutSpec(
    size_per_thread=(8, 2),
    threads_per_warp=(8, 8),
    warps_per_cta=(1, 4),
    order=(0, 1),
)
BLOCKED_H = BlockedLayoutSpec(
    size_per_thread=(1, 4),
    threads_per_warp=(16, 4),
    warps_per_cta=(4, 1),
    order=(1, 0),
)
BLOCKED_W = BlockedLayoutSpec(
    size_per_thread=(1, 8),
    threads_per_warp=(8, 8),
    warps_per_cta=(4, 1),
    order=(1, 0),
)


def _split_bits(value: int, count: int) -> list[int]:
    return [(value >> i) & 1 for i in range(count)]


def blocked_h_coords_python(tid: int) -> list[ThreadCoord]:
    """Coords for `#blocked1` on a logical `64x16` tile.

    `sizePerThread=[1,4] threadsPerWarp=[16,4] warpsPerCTA=[4,1] order=[1,0]`
    """

    warp_id = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    lane_row = lane // 4
    lane_col_group = lane % 4
    row = warp_id * 16 + lane_row
    col_base = lane_col_group * 4
    return [ThreadCoord(row=row, col=col_base + reg_col) for reg_col in range(4)]


def blocked_k_coords_python(tid: int) -> list[ThreadCoord]:
    """Coords for `#blocked` on a logical `64x64` tile.

    `sizePerThread=[8,2] threadsPerWarp=[8,8] warpsPerCTA=[1,4] order=[0,1]`
    """

    warp_id = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    lane_row = lane % 8
    lane_col = lane // 8
    row_base = lane_row * 8
    col_base = warp_id * 16 + lane_col * 2
    coords = []
    for reg_row in range(8):
        for reg_col in range(2):
            coords.append(ThreadCoord(row=row_base + reg_row, col=col_base + reg_col))
    return coords


def blocked_w_coords_python(tid: int) -> list[ThreadCoord]:
    """Coords for `#blocked2` on a logical `64x64` tile.

    `sizePerThread=[1,8] threadsPerWarp=[8,8] warpsPerCTA=[4,1] order=[1,0]`

    `shapePerCTA` is `32x64`, so the `64x64` logical tensor carries one extra
    row repeat in registers. The TTGIR uses this layout for the `w` tile.
    """

    warp_id = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    lane_row = lane // 8
    lane_col_group = lane % 8
    row_base = warp_id * 8 + lane_row
    col_base = lane_col_group * 8
    coords = []
    for row_repeat in range(2):
        row = row_base + row_repeat * 32
        for reg_col in range(8):
            coords.append(ThreadCoord(row=row, col=col_base + reg_col))
    return coords


def linear_k_coords_python(tid: int) -> list[ThreadCoord]:
    """Coords for `#linear` after `amdgpu.in_thread_transpose`.

    The TTGIR encodes:
    `register = [[0,1], [1,0], [2,0], [4,0]]`
    `lane     = [[8,0], [16,0], [32,0], [0,2], [0,4], [0,8]]`
    `warp     = [[0,16], [0,32]]`
    """

    warp_id = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    lane_bits = _split_bits(lane, 6)
    warp_bits = _split_bits(warp_id, 2)
    coords = []
    for reg in range(16):
        reg_bits = _split_bits(reg, 4)
        row = (
            reg_bits[1] * 1
            + reg_bits[2] * 2
            + reg_bits[3] * 4
            + lane_bits[0] * 8
            + lane_bits[1] * 16
            + lane_bits[2] * 32
        )
        col = (
            reg_bits[0] * 1
            + lane_bits[3] * 2
            + lane_bits[4] * 4
            + lane_bits[5] * 8
            + warp_bits[0] * 16
            + warp_bits[1] * 32
        )
        coords.append(ThreadCoord(row=row, col=col))
    return coords


def _coords_to_set(coords: Iterable[ThreadCoord]) -> set[tuple[int, int]]:
    return {(coord.row, coord.col) for coord in coords}


def validate_blocked_h_mapping() -> bool:
    all_coords = set()
    for tid in range(BLOCK_THREADS):
        all_coords |= _coords_to_set(blocked_h_coords_python(tid))
    return len(all_coords) == 64 * 16


def validate_blocked_k_mapping() -> bool:
    all_coords = set()
    for tid in range(BLOCK_THREADS):
        all_coords |= _coords_to_set(blocked_k_coords_python(tid))
    return len(all_coords) == 64 * 64


def validate_blocked_w_mapping() -> bool:
    all_coords = set()
    for tid in range(BLOCK_THREADS):
        all_coords |= _coords_to_set(blocked_w_coords_python(tid))
    return len(all_coords) == 64 * 64


def validate_linear_k_mapping() -> bool:
    all_coords = set()
    for tid in range(BLOCK_THREADS):
        all_coords |= _coords_to_set(linear_k_coords_python(tid))
    return len(all_coords) == 64 * 64


def _prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_counts = []
    total = 0
    for seq in range(cu_seqlens.numel() - 1):
        bos = int(cu_seqlens[seq].item())
        eos = int(cu_seqlens[seq + 1].item())
        count = (eos - bos + chunk_size - 1) // chunk_size
        chunk_counts.append(total)
        total += count
    return torch.tensor(chunk_counts, dtype=torch.int32, device=cu_seqlens.device)


def _unwrap_ir(value):
    if hasattr(value, "ir_value"):
        return value.ir_value()
    if hasattr(value, "value"):
        return value.value
    return value


def _normalize_specialized_g(g: torch.Tensor) -> torch.Tensor:
    if g.dim() == 3:
        if g.shape[0] != 1:
            raise ValueError(f"Expected `g.shape[0] == 1`, got {g.shape[0]}.")
        g = g[0]
    if g.dim() != 2:
        raise ValueError(f"Expected specialized `g` to be 2D or [1,T,H], got shape={tuple(g.shape)}.")
    return g.contiguous()


def _validate_specialized_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    wu_contiguous: bool,
) -> torch.Tensor:
    if not wu_contiguous:
        raise ValueError("The FlyDSL opt3 path is specialized for `wu_contiguous=True`.")
    if chunk_size != BT:
        raise ValueError(f"The FlyDSL opt3 path is specialized for `chunk_size == {BT}`.")
    if cu_seqlens is None:
        raise ValueError("The FlyDSL opt3 path requires variable-length batching (`cu_seqlens`).")
    if g is None or initial_state is None:
        raise ValueError("The FlyDSL opt3 path requires both `g` and `initial_state`.")

    if k.ndim != 4:
        raise ValueError(f"Expected `k` to have 4 dims, got {k.ndim}.")
    if w.ndim != 4 or u.ndim != 4:
        raise ValueError(
            "The FlyDSL opt3 path expects `w`/`u` in `[B,H,T_flat,K/V]` contiguous layout."
        )

    batch, total_t, num_hg, head_k = k.shape
    if batch != 1:
        raise ValueError(f"Expected `B == 1`, got {batch}.")
    if num_hg != HG:
        raise ValueError(f"Expected `Hg == {HG}`, got {num_hg}.")
    if head_k != K:
        raise ValueError(f"Expected `K == {K}`, got {head_k}.")
    if w.shape[0] != 1 or u.shape[0] != 1:
        raise ValueError("Expected specialized `w`/`u` batch dimension to be 1.")
    if w.shape[1] != H or u.shape[1] != H:
        raise ValueError(f"Expected `H == {H}`, got `w.shape[1]={w.shape[1]}` and `u.shape[1]={u.shape[1]}`.")
    if w.shape[-1] != K:
        raise ValueError(f"Expected `w.shape[-1] == {K}`, got {w.shape[-1]}.")
    if u.shape[-1] != V:
        raise ValueError(f"Expected `u.shape[-1] == {V}`, got {u.shape[-1]}.")
    if initial_state.shape != (cu_seqlens.numel() - 1, H, K, V):
        raise ValueError(
            "Expected `initial_state.shape == (num_seq, H, K, V)` for the specialized path, "
            f"got {tuple(initial_state.shape)}."
        )
    if int(cu_seqlens[-1].item()) != total_t:
        raise ValueError(
            "Expected `cu_seqlens[-1]` to match the flattened token dimension of `k`, "
            f"got {int(cu_seqlens[-1].item())} vs {total_t}."
        )

    return _normalize_specialized_g(g)


def chunk_gated_delta_rule_fwd_h_opt3_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    output_final_state: bool = True,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = BT,
    wu_contiguous: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Specialized opt3 reference matching the cached TTGIR configuration.

    This is intentionally specialized to the captured kernel:
    - variable-length batching is enabled
    - `wu_contiguous=True`
    - `g` and `initial_state` are present
    - `K=V=128`, `BT=64`, `BV=16`
    """

    g = _validate_specialized_inputs(
        k,
        w,
        u,
        g,
        initial_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        wu_contiguous=wu_contiguous,
    )

    batch, total_t, num_hg, _ = k.shape
    num_h = w.shape[1]
    t_flat = w.shape[2]
    num_seq = cu_seqlens.numel() - 1
    chunk_offsets = _prepare_chunk_offsets(cu_seqlens, chunk_size)

    total_chunks = 0
    for seq in range(num_seq):
        bos = int(cu_seqlens[seq].item())
        eos = int(cu_seqlens[seq + 1].item())
        total_chunks += (eos - bos + chunk_size - 1) // chunk_size

    h = torch.empty((batch, total_chunks, num_h, K, V), dtype=k.dtype, device=k.device)
    final_state = (
        torch.empty((num_seq, num_h, K, V), dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )
    v_new = (
        torch.empty((batch, num_h, t_flat, V), dtype=u.dtype, device=u.device)
        if save_new_value
        else None
    )

    head_group = max(num_h // num_hg, 1)

    for seq in range(num_seq):
        bos = int(cu_seqlens[seq].item())
        eos = int(cu_seqlens[seq + 1].item())
        seq_t = eos - bos
        seq_chunks = (seq_t + chunk_size - 1) // chunk_size
        chunk_base = int(chunk_offsets[seq].item())

        for h_idx in range(num_h):
            k_head_idx = h_idx // head_group
            h_state = initial_state[seq, h_idx].to(torch.float32).clone()

            for chunk_id in range(seq_chunks):
                t0 = chunk_id * chunk_size
                t1 = min(t0 + chunk_size, seq_t)
                chunk_len = t1 - t0

                h[0, chunk_base + chunk_id, h_idx] = h_state.to(h.dtype)

                w_chunk = w[0, h_idx, bos + t0 : bos + t1, :].to(torch.bfloat16)
                u_chunk = u[0, h_idx, bos + t0 : bos + t1, :].to(torch.bfloat16)
                g_chunk = g[bos + t0 : bos + t1, h_idx].to(torch.float32)
                k_chunk = k[0, bos + t0 : bos + t1, k_head_idx, :].to(torch.bfloat16)

                correction = w_chunk.to(torch.float32) @ h_state
                v_chunk = u_chunk.to(torch.float32) - correction

                if save_new_value:
                    v_new[0, h_idx, bos + t0 : bos + t1] = v_chunk.to(v_new.dtype)

                g_last = g_chunk[-1].exp()
                decay = torch.exp(g_chunk[-1:] - g_chunk).unsqueeze(-1)
                v_chunk = v_chunk * decay
                h_state = h_state * g_last
                h_state = h_state + k_chunk.transpose(0, 1).to(torch.float32) @ v_chunk

                if chunk_len < chunk_size and save_new_value:
                    pad_begin = bos + t1
                    pad_end = bos + t0 + chunk_size
                    if pad_begin < pad_end:
                        v_new[0, h_idx, pad_begin:pad_end].zero_()

            if output_final_state:
                final_state[seq, h_idx] = h_state

    return h, v_new, final_state


@functools.lru_cache(maxsize=8)
def build_chunk_gated_delta_rule_fwd_h_opt3_step(num_seq: int):
    """Build the specialized FlyDSL single-chunk step kernel."""

    arch = str(get_hip_arch())
    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"gdn_opt3_step_smem_{num_seq}")
    v_tile_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = v_tile_offset + (BT * BV * 4)

    @flyc.kernel
    def chunk_gated_delta_rule_fwd_kernel_h_opt3(
        k: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        g: fx.Tensor,
        state_in: fx.Tensor,
        state_out: fx.Tensor,
        h_out: fx.Tensor,
        v_out: fx.Tensor,
        cu_seqlens: fx.Tensor,
        chunk_offsets: fx.Tensor,
        chunk_id: fx.Int32,
        save_new_value: fx.Int32,
    ):
        tid = gpu.thread_id("x")
        bid_x = gpu.block_id("x")
        bid_y = gpu.block_id("y")

        seq_idx = bid_y // H
        head_idx = bid_y % H
        v_tile_idx = bid_x
        v_base = v_tile_idx * BV
        k_head_idx = head_idx // (H // HG)

        seq_idx_i = arith.index_cast(T.index, seq_idx)
        head_idx_i = arith.index_cast(T.index, head_idx)
        v_base_i = arith.index_cast(T.index, v_base)
        k_head_idx_i = arith.index_cast(T.index, k_head_idx)

        c0_idx = arith.constant(0, index=True)
        c1_idx = arith.constant(1, index=True)
        c_bt_idx = arith.constant(BT, index=True)
        c_k_idx = arith.constant(K, index=True)
        c_zero_f = arith.constant(0.0, type=T.f32)
        c_log2e = arith.constant(math.log2(math.e), type=T.f32)
        fm_fast = arith.FastMathFlags.fast

        base_ptr = allocator.get_base()
        s_v = SmemPtr(base_ptr, v_tile_offset, T.f32, shape=(BT, BV))
        s_v.get()

        bos = cu_seqlens[seq_idx_i]
        eos = cu_seqlens[seq_idx_i + c1_idx]
        seq_t = eos - bos
        seq_nt = (seq_t + fx.Int32(BT - 1)) // fx.Int32(BT)
        chunk_base = chunk_offsets[seq_idx_i]
        chunk_start = chunk_id * fx.Int32(BT)
        chunk_valid = arith.cmpi(arith.CmpIPredicate.slt, chunk_id, seq_nt)

        if chunk_valid:
            remaining = seq_t - chunk_start
            chunk_len = arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, remaining, fx.Int32(BT)),
                remaining,
                fx.Int32(BT),
            )
            chunk_len_i = arith.index_cast(T.index, chunk_len)
            chunk_base_token_i = arith.index_cast(T.index, bos + chunk_start)
            out_chunk_idx_i = arith.index_cast(T.index, chunk_base + chunk_id)
            last_token_i = arith.index_cast(T.index, bos + chunk_start + chunk_len - fx.Int32(1))
            g_last = g[last_token_i, head_idx_i]
            g_last_exp = (g_last * c_log2e).exp2(fastmath=fm_fast)

            for rep in range_constexpr((BT * BV) // BLOCK_THREADS):
                linear = tid + fx.Int32(rep * BLOCK_THREADS)
                t_rel = linear // BV
                v_rel = linear % BV
                t_rel_i = arith.index_cast(T.index, t_rel)
                v_rel_i = arith.index_cast(T.index, v_rel)
                token_valid = arith.cmpi(arith.CmpIPredicate.slt, t_rel, chunk_len)
                gated_value = c_zero_f

                if token_valid:
                    token_i = arith.index_cast(T.index, bos + chunk_start + t_rel)
                    v_idx_i = v_base_i + v_rel_i
                    dot_init = [_unwrap_ir(c_zero_f)]
                    dot_result = dot_init
                    for kk, acc_state in range(c0_idx, c_k_idx, c1_idx, init=dot_init):
                        acc_prev = acc_state[0]
                        w_val = w[0, head_idx_i, token_i, kk].extf(T.f32)
                        h_prev = state_in[seq_idx_i, head_idx_i, kk, v_idx_i]
                        acc_next = acc_prev + (w_val * h_prev)
                        dot_result = yield [_unwrap_ir(acc_next)]

                    correction = dot_result[0]
                    raw_v = u[0, head_idx_i, token_i, v_idx_i].extf(T.f32) - correction

                    if arith.cmpi(arith.CmpIPredicate.ne, save_new_value, fx.Int32(0)):
                        v_out[0, head_idx_i, token_i, v_idx_i] = arith.trunc_f(T.bf16, raw_v)

                    g_cur = g[token_i, head_idx_i]
                    decay = ((g_last - g_cur) * c_log2e).exp2(fastmath=fm_fast)
                    gated_value = raw_v * decay

                s_v.store(gated_value, [t_rel_i, v_rel_i])

            gpu.barrier()

            for rep in range_constexpr((K * BV) // BLOCK_THREADS):
                linear = tid + fx.Int32(rep * BLOCK_THREADS)
                k_rel = linear // BV
                v_rel = linear % BV
                k_rel_i = arith.index_cast(T.index, k_rel)
                v_rel_i = arith.index_cast(T.index, v_rel)
                v_idx_i = v_base_i + v_rel_i
                old_state = state_in[seq_idx_i, head_idx_i, k_rel_i, v_idx_i]
                h_out[0, out_chunk_idx_i, head_idx_i, k_rel_i, v_idx_i] = arith.trunc_f(T.bf16, old_state)

                update_init = [_unwrap_ir(c_zero_f)]
                update_result = update_init
                for t_idx, acc_state in range(c0_idx, chunk_len_i, c1_idx, init=update_init):
                    acc_prev = acc_state[0]
                    token_i = chunk_base_token_i + t_idx
                    k_val = k[0, token_i, k_head_idx_i, k_rel_i].extf(T.f32)
                    v_gated = s_v.load([t_idx, v_rel_i])
                    acc_next = acc_prev + (k_val * v_gated)
                    update_result = yield [_unwrap_ir(acc_next)]

                state_out[seq_idx_i, head_idx_i, k_rel_i, v_idx_i] = (old_state * g_last_exp) + update_result[0]
        else:
            for rep in range_constexpr((K * BV) // BLOCK_THREADS):
                linear = tid + fx.Int32(rep * BLOCK_THREADS)
                k_rel = linear // BV
                v_rel = linear % BV
                k_rel_i = arith.index_cast(T.index, k_rel)
                v_rel_i = arith.index_cast(T.index, v_rel)
                v_idx_i = v_base_i + v_rel_i
                state_out[seq_idx_i, head_idx_i, k_rel_i, v_idx_i] = state_in[
                    seq_idx_i, head_idx_i, k_rel_i, v_idx_i
                ]

    @flyc.jit
    def launch_chunk_step(
        k: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        g: fx.Tensor,
        state_in: fx.Tensor,
        state_out: fx.Tensor,
        h_out: fx.Tensor,
        v_out: fx.Tensor,
        cu_seqlens: fx.Tensor,
        chunk_offsets: fx.Tensor,
        chunk_id: fx.Int32,
        save_new_value: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        chunk_gated_delta_rule_fwd_kernel_h_opt3(
            k,
            w,
            u,
            g,
            state_in,
            state_out,
            h_out,
            v_out,
            cu_seqlens,
            chunk_offsets,
            chunk_id,
            save_new_value,
        ).launch(
            grid=(V // BV, num_seq * H, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_chunk_step


def chunk_gated_delta_rule_fwd_h_opt3(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = BT,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    wu_contiguous: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """FlyDSL specialized host wrapper for the cached opt3 configuration."""

    if gk is not None:
        raise NotImplementedError("The FlyDSL opt3 path does not yet support `gk`.")

    g = _validate_specialized_inputs(
        k,
        w,
        u,
        g,
        initial_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        wu_contiguous=wu_contiguous,
    )

    num_seq = cu_seqlens.numel() - 1
    t_flat = w.shape[2]
    chunk_offsets = _prepare_chunk_offsets(cu_seqlens, chunk_size).contiguous()
    total_chunks = 0
    max_chunks = 0
    for seq in range(num_seq):
        bos = int(cu_seqlens[seq].item())
        eos = int(cu_seqlens[seq + 1].item())
        seq_chunks = (eos - bos + chunk_size - 1) // chunk_size
        total_chunks += seq_chunks
        max_chunks = max(max_chunks, seq_chunks)

    h = torch.empty((1, total_chunks, H, K, V), dtype=k.dtype, device=k.device)
    v_out = (
        torch.empty((1, H, t_flat, V), dtype=u.dtype, device=u.device)
        if save_new_value
        else torch.empty((1, 1, 1, 1), dtype=u.dtype, device=u.device)
    )

    state_a = initial_state.to(torch.float32).contiguous()
    state_b = torch.empty_like(state_a)

    cu_kernel = cu_seqlens.to(torch.int32).contiguous()
    chunk_offsets_kernel = chunk_offsets.to(torch.int32).contiguous()
    stream = torch.cuda.current_stream(device=k.device)
    launch_chunk_step = build_chunk_gated_delta_rule_fwd_h_opt3_step(num_seq)
    compiled_step = flyc.compile(
        launch_chunk_step,
        k,
        w,
        u,
        g,
        state_a,
        state_b,
        h,
        v_out,
        cu_kernel,
        chunk_offsets_kernel,
        0,
        int(save_new_value),
        stream,
    )

    for chunk_id in range(max_chunks):
        compiled_step(
            k,
            w,
            u,
            g,
            state_a,
            state_b,
            h,
            v_out,
            cu_kernel,
            chunk_offsets_kernel,
            chunk_id,
            int(save_new_value),
            stream,
        )
        state_a, state_b = state_b, state_a

    final_state = state_a if output_final_state else None
    return h, (v_out if save_new_value else None), final_state


__all__ = [
    "BT",
    "BV",
    "BLOCK_THREADS",
    "H",
    "HG",
    "K",
    "V",
    "BLOCKED_H",
    "BLOCKED_K",
    "BLOCKED_W",
    "ThreadCoord",
    "blocked_h_coords_python",
    "blocked_k_coords_python",
    "blocked_w_coords_python",
    "linear_k_coords_python",
    "validate_blocked_h_mapping",
    "validate_blocked_k_mapping",
    "validate_blocked_w_mapping",
    "validate_linear_k_mapping",
    "build_chunk_gated_delta_rule_fwd_h_opt3_step",
    "chunk_gated_delta_rule_fwd_h_opt3",
    "chunk_gated_delta_rule_fwd_h_opt3_reference",
]
