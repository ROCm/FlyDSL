"""Reusable epilogue helpers for MFMA 16x16-based kernels.

This module provides two high-level epilogue building blocks:

- `default_epilog(...)`
  A lightweight row-iterator for the common MFMA accumulator-to-output mapping
  (mi in [0,m_repeat), ii in [0,4), row = bx_m + mi*16 + lane_div_16*4 + ii).
  The caller supplies `body_row(...)` that performs the per-row epilogue work
  (e.g. loads scales once, loops over ni, stores).

- `c_shuffle_epilog(...)`
  A CK-style LDS CShuffle epilogue skeleton:
    1) call `write_row_to_lds(...)` for each MFMA output row to populate `lds_out`
       in row-major [tile_m, tile_n] order
    2) barrier
    3) remap threads into (MLane, NLane) = (8,32) and read half2 (EVec=2) from LDS,
       then call `store_pair(...)` to emit the final global store/atomic.

These helpers are intentionally *dialect-agnostic*: callers pass the dialect
modules (`arith`, `vector`, `gpu`) and the `range_constexpr` iterator.
"""

from __future__ import annotations

from typing import Callable

from _mlir import ir


def default_epilog(
    *,
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    body_row: Callable,
):
    """Iterate the standard MFMA 16x16 row mapping and call `body_row(...)`.

    The mapping matches the common MFMA fragment layout used across kernels in this repo.

    Args:
      arith: flydsl arith ext module.
      range_constexpr: compile-time unrolled range helper.
      m_repeat: tile_m // 16 (python int).
      lane_div_16: index Value (0..3).
      bx_m: base row (index Value). For MoE, this is the base sorted-row for the tile.
      body_row: callback invoked as:
        body_row(mi=<int>, ii=<int>, row_in_tile=<index>, row=<index>)
    """
    bx_m_v = arith.ArithValue(bx_m)
    lane_div_16_mul4 = arith.ArithValue(lane_div_16) * 4
    ii_idx_list = [arith.constant(ii, index=True) for ii in range(4)]

    for mi in range_constexpr(m_repeat):
        mi_base = arith.constant(mi * 16, index=True)
        for ii in range_constexpr(4):
            row_off = lane_div_16_mul4 + ii_idx_list[ii]
            row_in_tile = arith.ArithValue(mi_base) + row_off
            row = bx_m_v + row_in_tile
            body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)


def c_shuffle_epilog(
    *,
    arith,
    vector,
    gpu,
    range_constexpr,
    # Tile params
    tile_m: int,
    tile_n: int,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    m_repeat: int,
    num_acc_n: int,
    # Thread mapping inputs
    tx,
    lane_div_16,
    lane_mod_16,
    bx_m,
    by_n,
    n_tile_base,
    # LDS buffer (f16 view, row-major [tile_m, tile_n] flattened)
    lds_out,
    # Callbacks
    write_row_to_lds: Callable,
    precompute_row: Callable | None = None,
    store_pair: Callable,
):
    """CK-style LDS CShuffle epilogue skeleton.

    Call pattern:
      - `write_row_to_lds(...)` is called once per MFMA row produced by this thread.
        It is responsible for writing all ni columns for that row into `lds_out`.
      - `store_pair(...)` is called for each (row_local, col_pair0) half2 after shuffle.

    `store_pair` can implement either global stores or atomics.
    """
    if int(block_size) <= 0 or (int(block_size) % int(cshuffle_nlane)) != 0:
        raise ValueError(
            f"block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuffle_nlane})"
        )
    cshuffle_mlane = int(block_size) // int(cshuffle_nlane)
    if (int(tile_m) % cshuffle_mlane) != 0:
        raise ValueError(
            f"tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile_m={tile_m}"
        )
    if int(e_vec) <= 0:
        raise ValueError(f"e_vec must be positive, got {e_vec}")
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )

    # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
    tile_n_idx = arith.constant(int(tile_n), index=True)
    n_tile_base_v = arith.ArithValue(n_tile_base)
    col_base_local = n_tile_base_v + arith.ArithValue(lane_mod_16)  # index within [0,tile_n)

    def _write_row(mi: int, ii: int, row_in_tile, row):
        # row_base_lds = row_in_tile * tile_n
        row_base_lds = arith.ArithValue(row_in_tile) * tile_n_idx
        write_row_to_lds(
            mi=mi,
            ii=ii,
            row_in_tile=row_in_tile,
            row=row,
            row_base_lds=row_base_lds,
            col_base_local=col_base_local,
            num_acc_n=num_acc_n,
            lds_out=lds_out,
        )

    default_epilog(
        arith=arith,
        range_constexpr=range_constexpr,
        m_repeat=m_repeat,
        lane_div_16=lane_div_16,
        bx_m=bx_m,
        body_row=_write_row,
    )

    # Ensure all LDS writes are visible before the shuffle-read.
    gpu.barrier()

    # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)

    m_reps_shuffle = int(tile_m) // CShuffleMLane
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    c_nlane = arith.constant(CShuffleNLane, index=True)
    m_lane = arith.ArithValue(tx) / c_nlane
    n_lane = arith.ArithValue(tx) % c_nlane
    c_evec = arith.constant(EVec, index=True)

    vec_f16 = ir.VectorType.get([EVec], ir.F16Type.get())
    bx_m_v = arith.ArithValue(bx_m)
    by_n_v = arith.ArithValue(by_n)

    for mr in range_constexpr(m_reps_shuffle):
        row_base_m = arith.constant(mr * CShuffleMLane, index=True)
        row_local = arith.ArithValue(row_base_m) + m_lane
        row = bx_m_v + row_local

        row_ctx = (
            precompute_row(row_local=row_local, row=row) if precompute_row is not None else None
        )

        row_base_lds = row_local * tile_n_idx
        for nr in range_constexpr(n_reps_shuffle):
            col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
            col_pair0 = arith.ArithValue(col_base_nr) + (n_lane * c_evec)  # even col within tile

            lds_idx_pair = row_base_lds + col_pair0
            frag = vector.load_op(vec_f16, lds_out, [lds_idx_pair])

            store_pair(
                row_local=row_local,
                row=row,
                row_ctx=row_ctx,
                col_pair0=col_pair0,
                col_g0=by_n_v + col_pair0,
                frag=frag,
            )

