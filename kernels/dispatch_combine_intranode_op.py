# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL DispatchCombine IntraNode 算子包装器。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

from .dispatch_combine_intranode_kernel import (
    make_dispatch_jit,
    make_combine_jit,
)


@dataclass
class FlyDSLDispatchCombineConfig:
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    data_type: torch.dtype = torch.bfloat16
    warp_num_per_block: int = 16
    block_num: int = 80
    chip: str = "gfx950"
    scale_dim: int = 0
    scale_type_size: int = 0
    enable_std_moe: bool = False
    use_external_inp_buf: bool = True
    quant_type: str = "none"

    @property
    def is_fp4(self):
        return self.data_type == torch.float4_e2m1fn_x2

    @property
    def elem_size(self):
        return torch.tensor([], dtype=self.data_type).element_size()

    @property
    def token_bytes(self):
        if self.is_fp4:
            return self.hidden_dim // 2
        return self.hidden_dim * self.elem_size

    @property
    def token_view_dim(self):
        if self.is_fp4:
            return self.hidden_dim // 2
        return self.hidden_dim

    @property
    def block_dim(self):
        return self.warp_num_per_block * 64

    @property
    def max_recv(self):
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def scale_bytes(self):
        return self.scale_dim * self.scale_type_size


class FlyDSLDispatchCombineIntraNodeOp:

    def __init__(self, config):
        self.cfg = config
        self._dev = torch.device("cuda", config.rank)
        r = config.rank

        self._alloc_buffers()
        ms.shmem_barrier_all()

        npes = config.world_size
        self._p2p_tok_off  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_tis      = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_wts  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_idx  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_tok  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_recv_num = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_scales = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_tok_off[pe]  = ms.shmem_ptr_p2p(self.shmem_tok_off.data_ptr(), r, pe)
            self._p2p_tis[pe]      = ms.shmem_ptr_p2p(self.shmem_tok_id_to_src.data_ptr(), r, pe)
            self._p2p_out_wts[pe]  = ms.shmem_ptr_p2p(self.shmem_disp_out_wts.data_ptr(), r, pe)
            self._p2p_out_idx[pe]  = ms.shmem_ptr_p2p(self.shmem_disp_out_idx.data_ptr(), r, pe)
            self._p2p_out_tok[pe]  = ms.shmem_ptr_p2p(self.shmem_disp_out_tok.data_ptr(), r, pe)
            self._p2p_recv_num[pe] = ms.shmem_ptr_p2p(self.shmem_recv_tok_num.data_ptr(), r, pe)
            self._p2p_out_scales[pe] = ms.shmem_ptr_p2p(self.shmem_out_scales.data_ptr(), r, pe)

        self._p2p_comb_inp     = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_comb_inp_wts = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_xdb_mem      = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_comb_inp[pe]     = ms.shmem_ptr_p2p(self.shmem_comb_inp_tok.data_ptr(), r, pe)
            self._p2p_comb_inp_wts[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_wts.data_ptr(), r, pe)
            self._p2p_xdb_mem[pe]      = ms.shmem_ptr_p2p(self.shmem_xdev_bar_mem.data_ptr(), r, pe)

        _disp_wpb = config.warp_num_per_block
        self._disp_fn = make_dispatch_jit(
            rank=r, npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=_disp_wpb,
            data_type=config.data_type,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            enable_std_moe=config.enable_std_moe,
        )

        _use_fp8_cast = (config.quant_type == "fp8_direct_cast" and config.data_type == torch.bfloat16)
        _comb_dtype = torch.float8_e4m3fn if _use_fp8_cast else config.data_type
        self._comb_fn = make_combine_jit(
            rank=r, npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=_disp_wpb,
            data_type=_comb_dtype,
            enable_weights=True,
            enable_std_moe=config.enable_std_moe,
            use_p2p_read=not config.use_external_inp_buf,
        )
        self._use_fp8_cast = _use_fp8_cast

        # barrier flag 初始值必须为 1, 否则首次 wait_until_equals(slot, 0) 立即满足
        self._xdev_flag = torch.ones(1, dtype=torch.int64, device=self._dev)

        self._fx_out_tok   = fx.Int64(self.shmem_disp_out_tok.data_ptr())
        self._fx_out_wts   = fx.Int64(self.shmem_disp_out_wts.data_ptr())
        self._fx_out_idx   = fx.Int64(self.shmem_disp_out_idx.data_ptr())
        self._fx_tok_off   = fx.Int64(self.shmem_tok_off.data_ptr())
        self._fx_recv_num  = fx.Int64(self.shmem_recv_tok_num.data_ptr())
        self._fx_dest_ctr  = fx.Int64(self.dest_pe_ctr.data_ptr())
        self._fx_disp_bar  = fx.Int64(self.disp_bar.data_ptr())
        self._fx_tok_map   = fx.Int64(self.dest_tok_map.data_ptr())
        self._fx_tis       = fx.Int64(self.shmem_tok_id_to_src.data_ptr())
        self._fx_total_rv  = fx.Int64(self.total_recv.data_ptr())
        # combine 固定地址
        self._fx_comb_inp  = fx.Int64(self.shmem_comb_inp_tok.data_ptr())
        self._fx_comb_out  = fx.Int64(self.shmem_comb_out_tok.data_ptr())
        self._fx_xdb_mem   = fx.Int64(self.shmem_xdev_bar_mem.data_ptr())
        self._fx_xdev_flag = fx.Int64(self._xdev_flag.data_ptr())
        self._fx_comb_bar  = fx.Int64(self.comb_bar.data_ptr())
        self._fx_trecv     = fx.Int64(self.total_recv.data_ptr())
        self._fx_p2p_tok_off  = fx.Int64(self._p2p_tok_off.data_ptr())
        self._fx_p2p_tis      = fx.Int64(self._p2p_tis.data_ptr())
        self._fx_p2p_out_wts  = fx.Int64(self._p2p_out_wts.data_ptr())
        self._fx_p2p_out_idx  = fx.Int64(self._p2p_out_idx.data_ptr())
        self._fx_p2p_out_tok  = fx.Int64(self._p2p_out_tok.data_ptr())
        self._fx_p2p_recv_num = fx.Int64(self._p2p_recv_num.data_ptr())
        self._fx_p2p_out_scales = fx.Int64(self._p2p_out_scales.data_ptr())
        self._fx_out_scales   = fx.Int64(self.shmem_out_scales.data_ptr())
        self._fx_p2p_comb_inp = fx.Int64(self._p2p_comb_inp.data_ptr())
        self._fx_p2p_comb_inp_wts = fx.Int64(self._p2p_comb_inp_wts.data_ptr())
        self._fx_p2p_xdb_mem  = fx.Int64(self._p2p_xdb_mem.data_ptr())
        self._fx_comb_inp_wts = fx.Int64(self.shmem_comb_inp_wts.data_ptr())
        self._fx_comb_out_wts = fx.Int64(self.shmem_comb_out_wts.data_ptr())
        self._fx_packed_recv_count = fx.Int64(self.packed_recv_count.data_ptr())
        self._fx_packed_recv_src_info = fx.Int64(self.packed_recv_src_info.data_ptr())
        self._fx_disp_tok_map = fx.Int64(self.disp_tok_to_ep_slot_map.data_ptr())
        self._fx_disp_grid_bar = fx.Int64(self.disp_grid_bar.data_ptr())
        self._fx_disp_out_wts = fx.Int64(self.shmem_disp_out_wts.data_ptr())

        self._disp_compiled = None
        self._comb_compiled = None

    def _alloc_buffers(self):
        cfg  = self.cfg
        npes = cfg.world_size
        k    = cfg.num_experts_per_token
        mt   = cfg.max_num_inp_token_per_rank
        mr   = cfg.max_recv   # npes * mt
        hdim = cfg.hidden_dim
        esz  = cfg.elem_size  # bytes per element

        tb = cfg.token_bytes
        tok_i16_mr = (mr * tb + 1) // 2
        tok_i16_mt = (mt * tb + 1) // 2

        # Symmetric shmem buffers
        self.shmem_disp_out_tok  = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self.shmem_disp_out_wts  = mori_shmem_create_tensor((mr * k,),     torch.float32)
        self.shmem_disp_out_idx  = mori_shmem_create_tensor((mr * k,),     torch.int32)
        scale_total = mr * cfg.scale_bytes if cfg.scale_bytes > 0 else 1
        self.shmem_out_scales    = mori_shmem_create_tensor((scale_total,), torch.int8)
        self.shmem_tok_off       = mori_shmem_create_tensor((1,),           torch.int32)
        self.shmem_recv_tok_num  = mori_shmem_create_tensor((npes,),        torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,),          torch.int32)
        self.shmem_comb_inp_tok  = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self.shmem_comb_out_tok  = mori_shmem_create_tensor((tok_i16_mt,), torch.int16)
        self.shmem_comb_inp_wts  = mori_shmem_create_tensor((mr * k,),     torch.float32)
        self.shmem_comb_out_wts  = mori_shmem_create_tensor((mt * k,),     torch.float32)
        self.shmem_xdev_bar_mem  = mori_shmem_create_tensor((npes,),        torch.int64)

        # Local device buffers
        self.dest_pe_ctr  = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar     = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        self.comb_bar     = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        self.total_recv   = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full(
            (mt * k,), sentinel, dtype=torch.int32, device=self._dev)

        # StdMoE buffers
        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            max_tok_per_expert = mr  # world_size * max_num_inp_token_per_rank
            self.packed_recv_count = torch.zeros(
                epr, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(
                epr * max_tok_per_expert, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.full(
                (mr * k,), -1, dtype=torch.int64, device=self._dev)
            self.disp_grid_bar = torch.zeros(
                1, dtype=torch.int32, device=self._dev)
        else:
            self.packed_recv_count = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.zeros(1, dtype=torch.int64, device=self._dev)
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)

    def barrier(self):
        ms.shmem_barrier_all()

    def reset(self):
        self.barrier()

    def dispatch(self, input, weights, scales, indices,
                 packed_recv_x=None,
                 block_num=-1, rdma_block_num=-1, warp_per_block=-1):
        cfg     = self.cfg
        cur_tok = input.shape[0]
        stream  = torch.cuda.current_stream()
        inp_c = input if input.is_contiguous() else input.contiguous()
        wts_c = weights if weights.is_contiguous() else weights.contiguous()
        idx_c = indices if (indices.dtype == torch.int32 and indices.is_contiguous()) \
            else indices.to(torch.int32).contiguous()

        sc_ptr = scales.data_ptr() if scales is not None else 0
        prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        if cfg.enable_std_moe:
            self.packed_recv_count.zero_()

        _std_args = (
            self._fx_packed_recv_count if cfg.enable_std_moe else fx.Int64(0),
            self._fx_packed_recv_src_info,
            self._fx_disp_tok_map,
            self._fx_disp_grid_bar,
        )

        if self._disp_compiled is None:
            args = (
                fx.Int64(inp_c.data_ptr()),
                fx.Int64(idx_c.data_ptr()),
                fx.Int64(wts_c.data_ptr()),
                self._fx_out_tok,
                self._fx_out_wts,
                self._fx_out_idx,
                self._fx_tok_off,
                self._fx_recv_num,
                self._fx_dest_ctr,
                self._fx_disp_bar,
                self._fx_tok_map,
                self._fx_tis,
                self._fx_total_rv,
                self._fx_p2p_tok_off,
                self._fx_p2p_tis,
                self._fx_p2p_out_wts,
                self._fx_p2p_out_idx,
                self._fx_p2p_out_tok,
                self._fx_p2p_recv_num,
                fx.Int64(sc_ptr),
                self._fx_p2p_out_scales,
                fx.Int64(prx_ptr),
                *_std_args,
                cur_tok,
                stream,
            )
            self._disp_compiled = flyc.compile(self._disp_fn, *args)
        else:
            self._disp_compiled(
                inp_c.data_ptr(),
                idx_c.data_ptr(),
                wts_c.data_ptr(),
                self._fx_out_tok,
                self._fx_out_wts,
                self._fx_out_idx,
                self._fx_tok_off,
                self._fx_recv_num,
                self._fx_dest_ctr,
                self._fx_disp_bar,
                self._fx_tok_map,
                self._fx_tis,
                self._fx_total_rv,
                self._fx_p2p_tok_off,
                self._fx_p2p_tis,
                self._fx_p2p_out_wts,
                self._fx_p2p_out_idx,
                self._fx_p2p_out_tok,
                self._fx_p2p_recv_num,
                sc_ptr,
                self._fx_p2p_out_scales,
                prx_ptr,
                *_std_args,
                cur_tok,
                stream,
            )

        mr   = cfg.max_recv
        hdim = cfg.hidden_dim
        k    = cfg.num_experts_per_token

        out_tok = self.shmem_disp_out_tok.view(torch.int8)[
            :mr * cfg.token_bytes].view(cfg.data_type).view(mr, cfg.token_view_dim)
        out_wts = self.shmem_disp_out_wts.view(mr, k)
        out_idx = self.shmem_disp_out_idx.view(mr, k)
        out_scales = None
        if cfg.scale_bytes > 0:
            out_scales = self.shmem_out_scales[:mr * cfg.scale_bytes].view(
                mr, cfg.scale_dim * cfg.scale_type_size)

        result = (out_tok, out_wts, out_scales, out_idx, self.total_recv)
        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            result = result + (
                self.packed_recv_count[:epr],
                self.packed_recv_src_info,
            )
        return result

    def combine(self, input, weights, indices,
                packed_recv_x=None, cur_tok=None,
                block_num=-1, rdma_block_num=-1, warp_per_block=-1,
                use_external_inp_buf=-1, call_reset=False):
        cfg    = self.cfg
        stream = torch.cuda.current_stream()

        if self._use_fp8_cast:
            inp_c = input.to(torch.float8_e4m3fn).contiguous()
        else:
            inp_c = input if input.is_contiguous() else input.contiguous()
        _cur_tok = cur_tok if cur_tok is not None else cfg.max_num_inp_token_per_rank

        wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if self._use_fp8_cast and packed_recv_x is not None:
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        _std_args_comb = (
            fx.Int64(prx_ptr),
            self._fx_disp_tok_map,
            self._fx_disp_out_wts,
        )

        if self._comb_compiled is None:
            args = (
                fx.Int64(inp_c.data_ptr()),
                self._fx_comb_inp,
                self._fx_comb_out,
                self._fx_xdb_mem,
                self._fx_xdev_flag,
                self._fx_tok_map,
                self._fx_comb_bar,
                self._fx_trecv,
                self._fx_tis,
                self._fx_p2p_comb_inp,
                self._fx_p2p_xdb_mem,
                fx.Int64(wts_ptr),
                self._fx_comb_inp_wts,
                self._fx_comb_out_wts,
                self._fx_p2p_comb_inp_wts,
                *_std_args_comb,
                _cur_tok,
                stream,
            )
            self._comb_compiled = flyc.compile(self._comb_fn, *args)
        else:
            self._comb_compiled(
                inp_c.data_ptr(),
                self._fx_comb_inp,
                self._fx_comb_out,
                self._fx_xdb_mem,
                self._fx_xdev_flag,
                self._fx_tok_map,
                self._fx_comb_bar,
                self._fx_trecv,
                self._fx_tis,
                self._fx_p2p_comb_inp,
                self._fx_p2p_xdb_mem,
                wts_ptr,
                self._fx_comb_inp_wts,
                self._fx_comb_out_wts,
                self._fx_p2p_comb_inp_wts,
                prx_ptr,
                self._fx_disp_tok_map,
                self._fx_disp_out_wts,
                _cur_tok,
                stream,
            )

        mt   = cfg.max_num_inp_token_per_rank
        hdim = cfg.hidden_dim
        k    = cfg.num_experts_per_token

        if self._use_fp8_cast:
            fp8_bytes = mt * hdim  # 1 byte per fp8 element
            out_tok = self.shmem_comb_out_tok.view(torch.int8)[
                :fp8_bytes].view(torch.float8_e4m3fn).view(mt, hdim).to(torch.bfloat16)
        else:
            out_tok = self.shmem_comb_out_tok.view(torch.int8)[
                :mt * cfg.token_bytes].view(cfg.data_type).view(mt, cfg.token_view_dim)
        out_wts = self.shmem_comb_out_wts.view(mt, k)

        if call_reset:
            self.reset()
        return out_tok, out_wts

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()
        n = int(self.total_recv[0].item())
        return self.shmem_tok_id_to_src[:n].clone()

    def get_registered_combine_input_buffer(self, dtype, hidden_dim=-1):
        h = hidden_dim if hidden_dim > 0 else self.cfg.token_view_dim
        dt = dtype if dtype is not None else self.cfg.data_type
        return self.shmem_comb_inp_tok.view(torch.int8).view(dt).view(-1, h)
