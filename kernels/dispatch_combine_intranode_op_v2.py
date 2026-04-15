"""
FlyDSL v2 DispatchCombine IntraNode 算子包装器。

与 v1 (`dispatch_combine_intranode_op.py`) 的区别：
- kernel 采用 Python FlyDSL 语法（`@flyc.kernel` + `mori_shmem.*`）
- 所有 buffer 地址以 fx.Int64 传入内核（避免 fly.memref → LLVM ptr 时序问题）
- 外部 API 与 v1 完全兼容（可直接替换）
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

from .dispatch_combine_intranode_kernel_v2 import (
    make_dispatch_jit,
    make_combine_jit,
)


@dataclass
class FlyDSLDispatchCombineConfigV2:
    """与 FlyDSLDispatchCombineConfig (v1) 相同字段，可互换使用。"""
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    top_k: int
    data_type: torch.dtype = torch.bfloat16
    warp_num_per_block: int = 16
    block_num: int = 80
    chip: str = "gfx942"
    # combine 内核可独立设置 warp_num_per_block。None 表示与 warp_num_per_block 相同。
    combine_warp_num_per_block: int = None

    @property
    def elem_size(self):
        return torch.tensor([], dtype=self.data_type).element_size()

    @property
    def block_dim(self):
        return self.warp_num_per_block * 64

    @property
    def max_recv(self):
        return self.world_size * self.max_num_inp_token_per_rank


class FlyDSLDispatchCombineIntraNodeOpV2:
    """FlyDSL v2 IntraNode Dispatch+Combine 算子。

    使用 Python FlyDSL 语法编写的 kernel（v2），功能与 v1 相同。
    接口与 FlyDSLDispatchCombineIntraNodeOp (v1) 完全兼容。
    """

    def __init__(self, config):
        self.cfg = config
        self._dev = torch.device("cuda", config.rank)
        r = config.rank

        # 先分配 symmetric buffer（顺序：alloc → barrier → compile）
        self._alloc_buffers()
        ms.shmem_barrier_all()

        # 预计算 dispatch P2P 地址表（消除内核中 ptr_p2p extern 调用开销）
        npes = config.world_size
        self._p2p_tok_off  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_tis      = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_wts  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_idx  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_tok  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_recv_num = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_tok_off[pe]  = ms.shmem_ptr_p2p(self.shmem_tok_off.data_ptr(), r, pe)
            self._p2p_tis[pe]      = ms.shmem_ptr_p2p(self.shmem_tok_id_to_src.data_ptr(), r, pe)
            self._p2p_out_wts[pe]  = ms.shmem_ptr_p2p(self.shmem_disp_out_wts.data_ptr(), r, pe)
            self._p2p_out_idx[pe]  = ms.shmem_ptr_p2p(self.shmem_disp_out_idx.data_ptr(), r, pe)
            self._p2p_out_tok[pe]  = ms.shmem_ptr_p2p(self.shmem_disp_out_tok.data_ptr(), r, pe)
            self._p2p_recv_num[pe] = ms.shmem_ptr_p2p(self.shmem_recv_tok_num.data_ptr(), r, pe)

        # 预计算 combine P2P 地址表（消除内核中 ptr_p2p extern 调用开销）
        self._p2p_comb_inp = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_xdb_mem  = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_comb_inp[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_tok.data_ptr(), r, pe)
            self._p2p_xdb_mem[pe]  = ms.shmem_ptr_p2p(self.shmem_xdev_bar_mem.data_ptr(), r, pe)

        # 创建 @flyc.jit launcher（首次调用时自动编译 + shmem_module_init）
        _disp_wpb = config.warp_num_per_block
        print(f"[v2] Rank {r}: creating v2 dispatch jit (warp_per_block={_disp_wpb})...")
        self._disp_fn = make_dispatch_jit(
            rank=r, npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.top_k,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=_disp_wpb,
            data_type=config.data_type,
        )

        _comb_wpb = (config.combine_warp_num_per_block
                     if config.combine_warp_num_per_block is not None
                     else config.warp_num_per_block)
        print(f"[v2] Rank {r}: creating v2 combine jit (warp_per_block={_comb_wpb})...")
        self._comb_fn = make_combine_jit(
            rank=r, npes=config.world_size,
            experts_per_token=config.top_k,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=_comb_wpb,
            data_type=config.data_type,
        )

        # combine 用的单调递增 barrier flag。
        # 初始值必须为 1（而非 0）：reset() 会把 shmem_xdev_bar_mem 清零，
        # 若 flag=0 则第一次 combine 的 wait_until_equals(slot, 0) 立即满足，
        # 跳过跨 GPU 屏障。与 mori 的 crossDeviceBarrierFlag[0]=1 对齐。
        self._xdev_flag = torch.ones(1, dtype=torch.int64, device=self._dev)

        # 预缓存固定 shmem buffer 地址（地址在 _alloc_buffers 后不变，避免每次重建）
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
        self._fx_trecv     = fx.Int64(self.total_recv.data_ptr())  # alias of _fx_total_rv
        # dispatch P2P 地址数组（预计算，消除内核中 ptr_p2p extern 调用）
        self._fx_p2p_tok_off  = fx.Int64(self._p2p_tok_off.data_ptr())
        self._fx_p2p_tis      = fx.Int64(self._p2p_tis.data_ptr())
        self._fx_p2p_out_wts  = fx.Int64(self._p2p_out_wts.data_ptr())
        self._fx_p2p_out_idx  = fx.Int64(self._p2p_out_idx.data_ptr())
        self._fx_p2p_out_tok  = fx.Int64(self._p2p_out_tok.data_ptr())
        self._fx_p2p_recv_num = fx.Int64(self._p2p_recv_num.data_ptr())
        # combine P2P 地址数组
        self._fx_p2p_comb_inp = fx.Int64(self._p2p_comb_inp.data_ptr())
        self._fx_p2p_xdb_mem  = fx.Int64(self._p2p_xdb_mem.data_ptr())

        self._disp_compiled = None
        self._comb_compiled = None

    def _alloc_buffers(self):
        cfg  = self.cfg
        npes = cfg.world_size
        k    = cfg.top_k
        mt   = cfg.max_num_inp_token_per_rank
        mr   = cfg.max_recv   # npes * mt
        hdim = cfg.hidden_dim

        # ── Symmetric shmem buffers（mori.shmem Python API 分配）
        self.shmem_disp_out_tok  = mori_shmem_create_tensor((mr * hdim,), torch.int16)
        self.shmem_disp_out_wts  = mori_shmem_create_tensor((mr * k,),    torch.float32)
        self.shmem_disp_out_idx  = mori_shmem_create_tensor((mr * k,),    torch.int32)
        self.shmem_tok_off       = mori_shmem_create_tensor((1,),          torch.int32)  # slot cnt
        self.shmem_recv_tok_num  = mori_shmem_create_tensor((npes,),       torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,),         torch.int32)  # src token id
        self.shmem_comb_inp_tok  = mori_shmem_create_tensor((mr * hdim,), torch.int16)
        self.shmem_comb_out_tok  = mori_shmem_create_tensor((mt * hdim,), torch.int16)
        self.shmem_xdev_bar_mem  = mori_shmem_create_tensor((npes,),       torch.int64)

        # ── 本地普通 device buffer
        self.dest_pe_ctr  = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar     = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        self.comb_bar     = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        self.total_recv   = torch.zeros(1,    dtype=torch.int32, device=self._dev)
        # sentinel = npes * max_recv（= npes² * max_tok_per_rank）
        # 保证 sentinel // max_recv = npes → dest_pe_j >= npes → 无效
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full(
            (mt * k,), sentinel, dtype=torch.int32, device=self._dev)

    def barrier(self):
        """跨 rank 同步。kernel 内部已实现自清理，无需清零缓冲区。"""
        ms.shmem_barrier_all()

    def reset(self):
        """等同于 barrier()。kernel 自清理，无需显式清零。"""
        self.barrier()

    def dispatch(self, input, weights, scales, indices,
                 block_num=-1, rdma_block_num=-1, warp_per_block=-1):
        """Dispatch tokens → remote experts via shmem P2P。

        返回 max_recv 全尺寸 tensor（不做 .item() 和动态切片），
        eager / CUDA Graph capture 均可直接调用，无特判。
        """
        cfg     = self.cfg
        cur_tok = input.shape[0]
        stream  = torch.cuda.current_stream()
        inp_c = input if input.is_contiguous() else input.contiguous()
        wts_c = weights if weights.is_contiguous() else weights.contiguous()
        idx_c = indices if (indices.dtype == torch.int32 and indices.is_contiguous()) \
            else indices.to(torch.int32).contiguous()

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
                cur_tok,
                stream,
            )

        mr   = cfg.max_recv
        hdim = cfg.hidden_dim
        k    = cfg.top_k

        out_tok = (self.shmem_disp_out_tok.view(torch.bfloat16)
                   .view(mr, hdim).to(cfg.data_type))
        out_wts = self.shmem_disp_out_wts.view(mr, k)
        out_idx = self.shmem_disp_out_idx.view(mr, k)
        return out_tok, out_wts, None, out_idx, self.total_recv

    def combine(self, input, weights, indices,
                block_num=-1, rdma_block_num=-1, warp_per_block=-1,
                use_external_inp_buf=-1, call_reset=False):
        """Combine expert outputs via P2P read + weighted accumulate。

        返回 max_tok 全尺寸 tensor（kernel 从 HBM 读取 total_recv），
        eager / CUDA Graph capture 均可直接调用，无特判。
        """
        cfg    = self.cfg
        stream = torch.cuda.current_stream()

        inp_c = input if (input.dtype == cfg.data_type and input.is_contiguous()) \
            else input.to(cfg.data_type).contiguous()

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
                stream,
            )

        mt   = cfg.max_num_inp_token_per_rank
        hdim = cfg.hidden_dim

        out_tok = (self.shmem_comb_out_tok.view(torch.bfloat16)
                   .view(mt, hdim).to(cfg.data_type))
        out_wts = None

        if call_reset:
            self.reset()
        return out_tok, out_wts

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()
        n = int(self.total_recv[0].item())
        return self.shmem_tok_id_to_src[:n].clone()

    def get_registered_combine_input_buffer(self, dtype, hidden_dim=-1):
        h = hidden_dim if hidden_dim > 0 else self.cfg.hidden_dim
        return self.shmem_comb_inp_tok.view(torch.bfloat16).view(-1, h)
