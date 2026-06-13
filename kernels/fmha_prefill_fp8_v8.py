# SPDX-License-Identifier: Apache-2.0
"""Per-shape dispatch for FP8 paged FMHA prefill.

Evidence from v7 diagonal-pair benchmarking:

* sq1024: baseline ``fmha_prefill_fp8_8wave`` is still faster.
* sq2048 and larger customer shapes: ``fmha_prefill_fp8_v7`` is faster.

This wrapper keeps both kernels intact and chooses at host launch time. The
``grid_blocks`` argument is accepted for API compatibility but recomputed for
the selected implementation because the two kernels intentionally export
different ``BM`` values.
"""

import flydsl.expr as fx

import fmha_prefill_fp8_8wave as _base
import fmha_prefill_fp8_v7 as _diag

HD = _base.HD
KSTEPS = _base.KSTEPS
DT = _base.DT
NWAVES = _base.NWAVES
NTHREADS = _base.NTHREADS
WAVE_ROWS = _base.WAVE_ROWS
TILE_BM = _diag.TILE_BM
BM = _diag.BM
BN = _base.BN
NSLOT = _base.NSLOT
NPASS = _base.NPASS
LOG2E = _base.LOG2E


def _as_int(value):
    try:
        return int(value)
    except TypeError:
        return int(value.item())


def _numel(value):
    if hasattr(value, "numel"):
        return int(value.numel())
    return int(value.shape.num_elements())


def run_attn(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    Qd: fx.Tensor,
    Kd: fx.Tensor,
    Vd: fx.Tensor,
    LTD: fx.Tensor,
    LTP: fx.Tensor,
    Ps: fx.Tensor,
    O: fx.Tensor,
    sq: fx.Int32,
    sk: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    page_size: fx.Constexpr[int],
    k_page_stride: fx.Int32,
    v_page_stride: fx.Int32,
    sm_scale: fx.Constexpr[float],
    causal: fx.Constexpr[int],
    grid_blocks: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    del grid_blocks
    sq_host = _as_int(sq)
    if sq_host <= 1024:
        selected = _base
    else:
        selected = _diag

    nq_host = int(nq)
    batch = _numel(Ps) // nq_host
    grid = batch * nq_host * ((sq_host + selected.BM - 1) // selected.BM)
    selected.run_attn(
        Q,
        K,
        V,
        Qd,
        Kd,
        Vd,
        LTD,
        LTP,
        Ps,
        O,
        sq,
        sk,
        nq,
        nk,
        page_size,
        k_page_stride,
        v_page_stride,
        sm_scale,
        causal,
        grid,
        stream,
    )
