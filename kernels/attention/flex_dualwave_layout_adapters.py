# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Layout-API implementations of flash dualwave helper interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath

from kernels.attention.pipeline import InfraContext


class FlexLayoutDualwaveTraits:
    PAGED = False
    SPLITK = False
    CROSS_SEQLEN = False
    DUALWAVE_SWP_SETPRIO = False
    DUALWAVE_SWP_LAZY_RESCALE = False
    LGKMCNT_0_ONLY = 0xC07F
    SCHED_MFMA_MASK = 0x008
    SCHED_VALU_MASK = 0x002
    SCHED_EXP_MASK = 0x400

    def __init__(self, head_dim: int, *, enable_stagger: bool = False, causal: bool = False):
        self.HEAD_DIM = int(head_dim)
        self.D_CHUNKS = 1
        self.CAUSAL = bool(causal)
        self.DUALWAVE_SWP_ENABLE_STAGGER = bool(enable_stagger)

    def summary_lines(self, *, prefix: str = "    ") -> List[str]:
        return [
            f"{prefix}FlexLayoutDualwaveTraits:",
            f"{prefix}  HEAD_DIM={self.HEAD_DIM} D_CHUNKS={self.D_CHUNKS} "
            f"PAGED={self.PAGED} SPLITK={self.SPLITK} CROSS_SEQLEN={self.CROSS_SEQLEN}",
            f"{prefix}  CAUSAL={self.CAUSAL} DUALWAVE_SWP_ENABLE_STAGGER={self.DUALWAVE_SWP_ENABLE_STAGGER} "
            f"SETPRIO={self.DUALWAVE_SWP_SETPRIO} LAZY_RESCALE={self.DUALWAVE_SWP_LAZY_RESCALE}",
            f"{prefix}  LGKMCNT_0_ONLY=0x{self.LGKMCNT_0_ONLY:X} "
            f"SCHED_MFMA=0x{self.SCHED_MFMA_MASK:03X} "
            f"SCHED_VALU=0x{self.SCHED_VALU_MASK:03X} "
            f"SCHED_EXP=0x{self.SCHED_EXP_MASK:03X}",
        ]


class FlexLayoutDualwaveCtx:
    def __init__(self, num_dma_k: int, num_dma_v: int):
        self.NUM_DMA_K = num_dma_k
        self.NUM_DMA_V = num_dma_v
        self.q_row = 0

    def split_tile(self, offset):
        return offset


class _LayoutVP:
    __slots__ = ()


def layout_v_pair_to_vec32(v_p: _LayoutVP):
    return v_p


def layout_v_vec32_to_pair(v) -> _LayoutVP:
    return _LayoutVP()


class _NullPageIds:
    def load_block_table_to_lds(self):
        pass

    def async_load_split_page(self, _n):
        return fx.Index(0)

    def load_page_id_lds(self, _t):
        return fx.Index(0)

    def finish_page_id(self, _lds):
        return fx.Index(0)

    def split_tile(self, off):
        return off


class _NullQLoader:
    def load_all(self):
        return None

    def scale_all(self, _q):
        return None


class _NullOutputStore:
    def store_final_o(self, *_a, **_k):
        pass

    def store_splitk_partial_o(self, *_a, **_k):
        pass


@dataclass
class FlexLayoutDualwaveAdapters:
    traits: FlexLayoutDualwaveTraits
    ctx: FlexLayoutDualwaveCtx
    infra: InfraContext
    shared_regs: dict
    kv_gmem_to_lds: Any
    kv_lds_to_regs: Any
    q_loader: Any
    gemm_helper: Any
    softmax_helper: Any
    page_ids: Any
    output_store: Any


def build_flex_layout_dualwave_adapters(
    *,
    traits: FlexLayoutDualwaveTraits,
    ctx: FlexLayoutDualwaveCtx,
    infra: InfraContext,
    shared_regs: dict,
    load_k_tile: Callable[[int, int], None],
    load_v_tile: Callable[[int, int], None],
    read_k: Callable,
    read_v: Callable,
    gemm1: Callable,
    softmax_finish: Callable,
    softmax_start: Callable,
    gemm2_write_p: Callable,
    gemm2_pv: Callable,
    row_reduce_max: Callable,
    npair: int,
    row_slots: List[List[int]],
    n_o: int,
    scale_log2e: fx.Float32,
    _FM,
) -> FlexLayoutDualwaveAdapters:
    def _apply(fn, **extra):
        out = fn(infra, **shared_regs, **extra)
        if out:
            shared_regs.update(out)

    class KvGmem:
        def load_k_split(self, tile_off, buf_id, page_id=None):
            load_k_tile(tile_off, buf_id)

        def load_k_tile(self, tile_idx, buf_id, page_id=None):
            load_k_tile(tile_idx, buf_id)

        def load_v_split(self, tile_off, buf_id, page_id=None):
            load_v_tile(tile_off, buf_id)

        def load_v_tile(self, tile_idx, buf_id, page_id=None):
            load_v_tile(tile_idx, buf_id)

    class KvLds:
        def load_k(self, buf_id):
            infra.buf_slot = buf_id
            _apply(read_k)
            return buf_id

        def load_v(self, buf_id):
            infra.buf_slot = buf_id
            _apply(read_v)
            return buf_id

    class Gemm:
        def __init__(self):
            self._pv_done = False

        def qk(self, _v_k, _q_scaled):
            _apply(gemm1)
            return _LayoutVP()

        def pv_step_k(self, step, v_p, v_v, v_o):
            if step == 0:
                self._pv_done = False
            elif step == 1 and not self._pv_done:
                _apply(gemm2_write_p)
                _apply(gemm2_pv)
                self._pv_done = True
            return v_o

        def pv(self, v_p, v_v, v_o):
            if not self._pv_done:
                _apply(gemm2_write_p)
                _apply(gemm2_pv)
                self._pv_done = True
            return v_o

    class Softmax:
        def split_tile(self, off):
            return off

        def v_s_vec_to_lists(self, v_s):
            return v_s

        def causal_mask_split_prologue_if_needed(self, v_s):
            return v_s

        def causal_mask_prologue_if_needed(self, v_s, *_a, **_k):
            return v_s

        def seq_pad_mask_if_needed(self, v_s, _tile):
            return v_s

        def floor_masked_max(self, m_row):
            return m_row

        def _row_max_from_frag_s(self):
            frag_S = shared_regs["frag_S"]
            m_vals = []
            for r in range_constexpr(npair):
                slots = row_slots[r]
                row_max = frag_S[slots[0]]
                for si in range_constexpr(1, len(slots)):
                    row_max = row_max.maximumf(frag_S[slots[si]])
                m_vals.append(row_reduce_max(row_max, "max"))
            return m_vals

        def reduce_max(self, _v_s):
            m_vals = self._row_max_from_frag_s()
            for r in range_constexpr(npair):
                shared_regs["m_i"][r] = m_vals[r]
            return m_vals[0]

        def sub_m(self, _v_s, m_row):
            frag_S = shared_regs["frag_S"]
            for r in range_constexpr(npair):
                m_val = shared_regs["m_i"][r]
                for si in range_constexpr(len(row_slots[r])):
                    s = row_slots[r][si]
                    frag_S[s] = frag_S[s] - m_val
            return _LayoutVP()

        def exp2(self, _v_s, start, length):
            if start == 16 and length == 16:
                _apply(softmax_finish)
                return _LayoutVP()
            frag_S = shared_regs["frag_S"]
            v_pp = shared_regs["v_p_partial"]
            m_i = shared_regs["m_i"]
            if start == 0 and length == 16:
                for r in range_constexpr(npair):
                    for si in range_constexpr(len(row_slots[r])):
                        s = row_slots[r][si]
                        v_pp[s] = fmath.exp2(
                            (frag_S[s] - m_i[r]) * scale_log2e, fastmath=_FM,
                        )
            return _LayoutVP()

        def reduce_sum(self, l_row, _v_p):
            _apply(softmax_finish)
            return shared_regs["l_i"][0]

        def cast_p(self, v_p):
            _apply(gemm2_write_p)
            return v_p

        def rescale_from_tile_max(self, m_row, m_tile_max):
            m_new = m_row.maximumf(m_tile_max) if hasattr(m_row, "maximumf") else m_tile_max
            corr = fmath.exp2(m_row - m_new, fastmath=_FM)
            return m_new, corr

        def apply_l_rescale(self, l_row, rescale):
            return l_row * rescale

        def safe_l_inv(self, l_row):
            denom = l_row + fx.Float32(1e-6)
            return fx.Float32(1.0) / denom

        def scale_o(self, v_o, scale_scalar):
            frag_O = v_o[0]
            for i in range_constexpr(n_o):
                frag_O[i] = frag_O[i] * scale_scalar

        def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
            _apply(softmax_start)
            return v_o, shared_regs["m_i"][0], shared_regs["l_i"][0], v_p

        def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
            return self.rescale_o(v_o, m_row, l_row, m_tile_max, v_p)

    gemm = Gemm()
    return FlexLayoutDualwaveAdapters(
        traits=traits,
        ctx=ctx,
        infra=infra,
        shared_regs=shared_regs,
        kv_gmem_to_lds=KvGmem(),
        kv_lds_to_regs=KvLds(),
        q_loader=_NullQLoader(),
        gemm_helper=gemm,
        softmax_helper=Softmax(),
        page_ids=_NullPageIds(),
        output_store=_NullOutputStore(),
    )
