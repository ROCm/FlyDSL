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
    # D-flag C-1: 启用 per-token flag 跨卡同步。开启后 dispatch / combine kernel
    # 在 const_expr 路径上启用 reset / spin-wait，并由 fused gemm2 epilogue
    # 跨卡 atomic_add 远端 flag；关闭则整段 DCE，行为等同 baseline。
    use_token_flag_sync: bool = False

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
        # D-flag C-1：per-token flag 的 P2P base 表。fused gemm2 epilogue
        # 用 dest_pe -> base 查表 + flag offset 跨卡 atomic_add_global_at。
        self._p2p_comb_flag    = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_comb_inp[pe]     = ms.shmem_ptr_p2p(self.shmem_comb_inp_tok.data_ptr(), r, pe)
            self._p2p_comb_inp_wts[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_wts.data_ptr(), r, pe)
            self._p2p_xdb_mem[pe]      = ms.shmem_ptr_p2p(self.shmem_xdev_bar_mem.data_ptr(), r, pe)
            self._p2p_comb_flag[pe]    = ms.shmem_ptr_p2p(self.shmem_comb_token_flag.data_ptr(), r, pe)

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
            use_token_flag_sync=config.use_token_flag_sync,
        )

        _use_fp8_cast = (config.quant_type == "fp8_direct_cast" and config.data_type == torch.bfloat16)
        _comb_dtype = torch.float8_e4m3fn if _use_fp8_cast else config.data_type
        # Mixed-dtype Stage 1 (mori UseFp8DirectCast equivalent): when
        # _use_fp8_cast is on, the user feeds bf16 input to ``combine()`` and
        # the kernel performs an inline bf16 → fp8 cast in Stage 1 before P2P
        # scatter.  This avoids an extra ~12μs ``input.to(fp8).contiguous()``
        # PyTorch elementwise kernel that would otherwise sit on the cudagraph
        # critical path.  Wrapper-side allocation/views remain fp8-stride.
        _comb_inp_dt = torch.bfloat16 if _use_fp8_cast else None
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
            inp_data_type=_comb_inp_dt,
            use_token_flag_sync=config.use_token_flag_sync,
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
        # D-flag C-1 fx wrappers
        self._fx_comb_flag    = fx.Int64(self.shmem_comb_token_flag.data_ptr())
        self._fx_p2p_comb_flag = fx.Int64(self._p2p_comb_flag.data_ptr())
        self._fx_local_counter = fx.Int64(self.device_local_counter.data_ptr())
        self._fx_comb_inp_wts = fx.Int64(self.shmem_comb_inp_wts.data_ptr())
        self._fx_comb_out_wts = fx.Int64(self.shmem_comb_out_wts.data_ptr())
        self._fx_packed_recv_count = fx.Int64(self.packed_recv_count.data_ptr())
        self._fx_packed_recv_src_info = fx.Int64(self.packed_recv_src_info.data_ptr())
        self._fx_disp_tok_map = fx.Int64(self.disp_tok_to_ep_slot_map.data_ptr())
        self._fx_disp_grid_bar = fx.Int64(self.disp_grid_bar.data_ptr())
        self._fx_disp_out_wts = fx.Int64(self.shmem_disp_out_wts.data_ptr())

        self._disp_compiled = None
        self._comb_compiled = None
        # combine kernel 的 skip_stage1 变体：给 fused_gemm2_combine 算子使用，
        # 此时 fused kernel 已经把 token / 权重 P2P 写入 shmem_comb_inp[_wts]，
        # combine 只跑 Stage 2 (CrossDeviceBarrier) + Stage 3 (本地 weighted-accum)。
        self._comb_no_s1_fn = None
        self._comb_no_s1_compiled = None
        # D-flag C-1: dispatch() 调用时填入 raw input weights data_ptr，供
        # combine_no_stage1 在 use_token_flag_sync 路径下直接本地读取。
        self._raw_input_wts_ptr = 0

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

        # D-flag C-1 per-token flag buffer：fused gemm2 epilogue 完成 (token, j)
        # partial sum 后，本卡 cross-N-tile reduce 完成态下由代表 thread 跨卡
        # atomic_add(remote_flag[token_id], 1)。combine kernel stage 3 入口
        # per-warp spin-wait `flag[tok_id] >= topk`。长度按 mr 上限分配（实际
        # bs <= mr，dispatch kernel 入口仅 reset 前 mr 个）。fallback 路径
        # (use_token_flag_sync=False) 不用此 buffer，分配 ~4KB 开销可忽略。
        self.shmem_comb_token_flag = mori_shmem_create_tensor((mr,), torch.int32)

        # mori_shmem_create_tensor 走 shmem_malloc，分配的是未初始化的 raw memory。
        # 对 fused MoE-GEMM2 + EP-Combine 路径，GEMM2 需要在 epilogue 用
        # shmem_tok_id_to_src 解码 dest_pe / dest_lid，越界 garbage 会触发 LDS
        # OOB → 写到任意全局地址 → 破坏 control state。这里把所有 combine 路径
        # 直接读写的 symmetric buffer 显式清零，保证：
        #   - shmem_tok_id_to_src[t] 对未被 dispatch 写入的 t 解码为 (pe=0, lid=0)，
        #     P2P scatter 退化成"安全无副作用"（多写同一槽位）
        #   - shmem_xdev_bar_mem 起始 0，CrossDeviceBarrier 第一次 wait 不会读到
        #     残留值（依赖 cur_flag 单调递增）
        #   - shmem_comb_inp_{tok,wts} 起始 0，combine_no_stage1 在 stage 3 累加
        #     时不会读到 garbage
        self.shmem_tok_id_to_src.zero_()
        self.shmem_comb_inp_tok.zero_()
        self.shmem_comb_inp_wts.zero_()
        self.shmem_xdev_bar_mem.zero_()
        # flag buffer 入口先 zero_ 一次。运行期 use_token_flag_sync=True 时
        # 由 dispatch kernel 入口 grid-stride memset 重置（每 chain 一次）。
        self.shmem_comb_token_flag.zero_()

        # Local device buffers
        self.dest_pe_ctr  = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar     = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        self.comb_bar     = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        self.total_recv   = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full(
            (mt * k,), sentinel, dtype=torch.int32, device=self._dev)

        # D-flag C-1 device-local counter（仅本卡可见，无需 symmetric）：
        # GEMM2 epilogue 用 atomicrmw add (device-scope) 在 row_i32（即
        # sorted_token_ids 的全局 row 索引）维度累计 N-tile 完成数。
        # 当 old == num_n_tiles - 1 时由代表 thread 跨卡 system-scope
        # atomic_add(remote_flag[dest_lid], 1)，并 atomicrmw xchg 把 counter
        # 复位 0 供下一 chain 复用。fallback 路径不用。
        #
        # **size 必须覆盖 row_i32 上界 ≤ num_valid_ids**。aiter sorting 下
        # num_valid_ids 上界 = max_padded = mr*k + npes*epr*tile_m - k
        # （tile_m 是 GEMM2 编译时 m-tile，最大常用 128）。早期版本误用
        # mr*k=npes*mt*k 作为 size，在 BS≤8 + aiter sorting 下 row_i32 可
        # 达 epr*tile_m=1024 (epr=32,tile_m=32) >> mr*k=256，触发 OOB
        # atomic ++ → 跨卡 ++ 永不触发 → combine spin 永远等不到 → hang。
        # 这里按 tile_m_max=128 取保守上界，多 ~16KB/rank 开销可忽略。
        epr = cfg.num_experts_per_rank
        tile_m_max = 128
        local_counter_size = mr * k + npes * epr * tile_m_max
        self.device_local_counter = torch.zeros(
            local_counter_size, dtype=torch.int32, device=self._dev)

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
        # D-flag C-1: 缓存 raw input weights pointer 给 combine_no_stage1 用。
        # me 的 raw input weights layout = [max_tok_per_rank, topk] f32，索引
        # (src_tok, lane)，正好是 stage 3b weight 累加需要的源（baseline 绕了
        # scatter + P2P read 一圈得到的也是同一个值，只不过是 topk 个副本求和
        # = 8 × weights[src_tok, lane]）。直接本地读，无 P2P 无跨卡 sync。
        self._raw_input_wts_ptr = wts_c.data_ptr()

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
                self._fx_comb_flag,
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
                self._fx_comb_flag,
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

        # In _use_fp8_cast mode, the combine kernel does the bf16 → fp8 cast
        # inline in Stage 1 (mori UseFp8DirectCast equivalent), so the wrapper
        # passes bf16 input straight through.  Skipping the PyTorch-level
        # ``.to(fp8).contiguous()`` saves ~12μs per iter on the cudagraph
        # critical path.
        inp_c = input if input.is_contiguous() else input.contiguous()
        _cur_tok = cur_tok if cur_tok is not None else cfg.max_num_inp_token_per_rank

        wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if self._use_fp8_cast and packed_recv_x is not None:
            # std-MoE expert-major buffer (`packed_recv_x`) is produced in bf16
            # by the upstream pipeline; downstream Stage 1 reads it in fp8
            # dtype, so we still cast here.  This branch is independent from
            # the regular combine input path above.
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
                self._fx_comb_flag,
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
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )

        mt   = cfg.max_num_inp_token_per_rank
        hdim = cfg.hidden_dim
        k    = cfg.num_experts_per_token

        # fp8_direct_cast contract: external dtype is bf16 on both ends; the
        # combine kernel itself writes bf16 to ``shmem_comb_out_tok`` (Stage 3
        # _from_accum casts v4f32 → v4bf16 inline), so we view the buffer as
        # bf16 directly with no extra PyTorch-level cast on the critical path.
        out_tok = self.shmem_comb_out_tok.view(torch.int8)[
            :mt * cfg.token_bytes].view(cfg.data_type).view(mt, cfg.token_view_dim)
        out_wts = self.shmem_comb_out_wts.view(mt, k)

        if call_reset:
            self.reset()
        return out_tok, out_wts

    def combine_no_stage1(self, input, weights, indices,
                          packed_recv_x=None, cur_tok=None,
                          call_reset=False,
                          enable_weights: bool = True):
        """combine 的 stage1-skipped 变体。

        语义：跳过 P2P scatter（外部 fused kernel 已把数据写入 shmem_comb_inp[_wts]），
              只执行 Stage 2 (CrossDeviceBarrier) + Stage 3 (本地 weighted-accum)。

        Parameters
        ----------
        enable_weights
            ``True`` (默认) 兼容当前 fused-with-weight 链路：保留 Stage 1
            的 weight scatter (``skip_stage1_keep_wts=True``) + Stage 3b
            的 weight accumulate。
            ``False`` 给 weight-free fused 路径（fused MoE 上游已经把
            weight 处理掉了，combine 端不需要 out_wts）：完全 DCE 掉
            weight scatter + Stage 3b，省 ~3-5 μs。
            两种变体走不同的 JIT 缓存，互不污染。

        约定：调用前 fused kernel 必须保证：
              - shmem_comb_inp_tok 已写入本 PE 应接收的所有 token（按 max_tok_per_rank 槽位）
              - shmem_comb_inp_wts 已写入对应权重（仅 enable_weights=True 时需要）
              - total_recv 已被 dispatch 设置完毕（Stage 3 用于读 cur_rank_num_token）
        """
        cfg    = self.cfg
        stream = torch.cuda.current_stream()

        # When skip_stage1=True (the only mode this method ever compiles for),
        # the combine kernel does NOT read inp_c — Stage 1 is bypassed and the
        # kernel reads from shmem_comb_inp_tok directly (already populated by
        # the upstream fused GEMM2 epilogue P2P scatter).  So skip the
        # potentially-expensive Python-level fp8 cast (.to(fp8) + .contiguous())
        # if the caller gave us a fp8 input or even a placeholder bf16: the
        # cast is a ~12us elementwise kernel that gets captured by cudagraph
        # and ends up serially on the chain critical path for nothing.
        # Caller (fused op wrapper) already CV-casted in the GEMM2 epilogue.
        if self._use_fp8_cast and input.dtype != torch.float8_e4m3fn:
            inp_c = input.to(torch.float8_e4m3fn).contiguous()
        else:
            inp_c = input if input.is_contiguous() else input.contiguous()
        _cur_tok = cur_tok if cur_tok is not None else cfg.max_num_inp_token_per_rank

        # D-flag C-1: use_token_flag_sync 时，combine stage 3b 不再走 P2P read
        # disp_out_wts/comb_inp_wts，而是直接读本地 raw input weights
        # ([max_tok_per_rank, topk] f32 layout，索引 = (src_tok, lane))。
        # caller 传 raw_input_wts_ptr 作为 wts_ptr，kernel stage 3b 在
        # use_token_flag_sync 路径下用 addr_wts_buf + (wt_tok*topk+lane)*4
        # offset。这样 stage 1 weight scatter / stage 2 barrier 全部裁掉。
        if cfg.use_token_flag_sync and weights is None:
            assert self._raw_input_wts_ptr != 0, (
                "use_token_flag_sync requires dispatch() to be called first "
                "so _raw_input_wts_ptr is populated.")
            wts_ptr = self._raw_input_wts_ptr
        else:
            wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if self._use_fp8_cast and packed_recv_x is not None:
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        # JIT 缓存按 enable_weights 区分（两份编译产物）。
        # 历史 self._comb_no_s1_fn / _compiled 升级为 dict[bool, fn]。
        if not isinstance(self._comb_no_s1_fn, dict):
            self._comb_no_s1_fn = {}
            self._comb_no_s1_compiled = {}

        if enable_weights not in self._comb_no_s1_fn:
            from .dispatch_combine_intranode_kernel import make_combine_jit
            _use_fp8_cast = self._use_fp8_cast
            _comb_dtype = torch.float8_e4m3fn if _use_fp8_cast else cfg.data_type
            # Mixed-dtype contract for fp8_direct_cast: external dtype = bf16,
            # transport dtype = fp8.  Stage 3 _from_accum will cast f32 → bf16
            # inline so kernel writes bf16 directly to shmem_comb_out_tok and
            # the wrapper does NOT need a post .to(bf16) cast.
            _comb_inp_dt = torch.bfloat16 if _use_fp8_cast else None
            # enable_weights=False 路径（fused MoE 不需要 out_wts）：
            # weight scatter + Stage 3b weight accumulate 都在 const_expr
            # 处被 DCE 掉，省 ~3-5μs。
            # enable_weights=True 路径（兼容 fused-with-weight）：
            # 保留 Stage 1 weight scatter（skip_stage1_keep_wts=True），
            # 因为同 fabric 上与 token P2P 并发的 16B 小写会被静默丢，必须
            # 放在静态 fabric 上由 combine kernel 完成。
            self._comb_no_s1_fn[enable_weights] = make_combine_jit(
                rank=cfg.rank, npes=cfg.world_size,
                experts_per_rank=cfg.num_experts_per_rank,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.block_num,
                warp_num_per_block=cfg.warp_num_per_block,
                data_type=_comb_dtype,
                enable_weights=bool(enable_weights),
                enable_std_moe=cfg.enable_std_moe,
                use_p2p_read=not cfg.use_external_inp_buf,
                skip_stage1=True,
                inp_data_type=_comb_inp_dt,
                skip_stage1_keep_wts=bool(enable_weights),
                use_token_flag_sync=cfg.use_token_flag_sync,
            )

        if enable_weights not in self._comb_no_s1_compiled:
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
                fx.Int64(prx_ptr),
                self._fx_disp_tok_map,
                self._fx_disp_out_wts,
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )
            self._comb_no_s1_compiled[enable_weights] = flyc.compile(
                self._comb_no_s1_fn[enable_weights], *args
            )
        else:
            self._comb_no_s1_compiled[enable_weights](
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
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )

        mt   = cfg.max_num_inp_token_per_rank
        hdim = cfg.hidden_dim
        k    = cfg.num_experts_per_token

        # fp8_direct_cast contract: combine kernel writes bf16 to
        # ``shmem_comb_out_tok`` directly (see ``combine`` above for details).
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
