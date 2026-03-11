"""
FlyDSL DispatchCombine IntraNode Python Wrapper — v1（Legacy）。

注意：v1 使用手写 LLVM IR 字符串作为 kernel body；新项目请使用 v2：
  dispatch_combine_intranode_op_v2.py   # Python FlyDSL 语法，ExternFunction shmem

关键 buffer 尺寸说明：
  max_recv = npes * max_tok_per_rank
  接收端 shmem buffer 需为 max_recv × hidden_dim（而非仅 max_tok × hidden_dim），
  因为任意一个 PE 最多可从所有 npes 个 rank 各接收 max_tok 个 token。
"""
from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

from .dispatch_combine_intranode_kernel import (
    _find_mori_shmem_bc,
    build_and_compile_dispatch,
    build_and_compile_combine,
)

_hip = ctypes.CDLL("libamdhip64.so")


def _hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


def hip_module_load(path):
    mod = ctypes.c_void_p()
    _hip_check(_hip.hipModuleLoad(ctypes.byref(mod), path.encode()),
               f"hipModuleLoad({path})")
    return mod


def hip_get_function(mod, name):
    func = ctypes.c_void_p()
    _hip_check(_hip.hipModuleGetFunction(ctypes.byref(func), mod, name.encode()), name)
    return func


def hip_launch_kernel(func, grid_x, block_x, params):
    """Launch GPU kernel. params = list of ctypes scalars/pointers."""
    objs = list(params)
    ptrs = (ctypes.c_void_p * len(objs))()
    for i, p in enumerate(objs):
        ptrs[i] = ctypes.cast(ctypes.byref(p), ctypes.c_void_p)
    _hip_check(_hip.hipModuleLaunchKernel(
        func, grid_x, 1, 1, block_x, 1, 1, 0, None, ptrs, None
    ), "hipModuleLaunchKernel")


def _p(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _pnull():
    return ctypes.c_void_p(0)


def _i32(val):
    return ctypes.c_int32(int(val))


@dataclass
class FlyDSLDispatchCombineConfig:
    """对齐 mori EpDispatchCombineConfig。"""
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

    @property
    def elem_size(self):
        return torch.tensor([], dtype=self.data_type).element_size()

    @property
    def block_dim(self):
        return self.warp_num_per_block * 64

    @property
    def max_recv(self):
        """每 PE 可接收的最大 token 数 = npes * max_tok_per_rank。"""
        return self.world_size * self.max_num_inp_token_per_rank


class FlyDSLDispatchCombineIntraNodeOp:
    """FlyDSL IntraNode Dispatch+Combine 算子。

    所有 shmem buffer 通过 mori.shmem Python API 分配（host 侧使用 shmem bitcode）。
    """

    def __init__(self, config):
        self.cfg = config
        self._device = torch.device("cuda", config.rank)
        self._mori_bc = _find_mori_shmem_bc()
        self._disp_hsaco, self._comb_hsaco = self._compile_kernels()
        self._disp_mod = hip_module_load(self._disp_hsaco)
        self._comb_mod = hip_module_load(self._comb_hsaco)
        self._disp_func = hip_get_function(self._disp_mod, "ep_dispatch_intranode")
        self._comb_func = hip_get_function(self._comb_mod, "ep_combine_intranode")
        # 先分配所有对称 buffer，再初始化 shmem module（确保堆布局一致）
        self._alloc_buffers()
        # barrier：确保所有 PE 均完成对称 buffer 分配后再初始化 module
        ms.shmem_barrier_all()
        ms.shmem_module_init(self._disp_mod.value)
        ms.shmem_module_init(self._comb_mod.value)
        torch.cuda.synchronize()
        # 单调递增的 barrier flag（combine() 内每轮 +1，首轮变为 1）
        self._xdev_flag = torch.zeros(1, dtype=torch.int64, device=self._device)

    def _compile_kernels(self):
        tmp = tempfile.gettempdir()
        chip = self.cfg.chip
        r = self.cfg.rank
        d = os.path.join(tmp, f"ep_dispatch_{chip}_r{r}.hsaco")
        c = os.path.join(tmp, f"ep_combine_{chip}_r{r}.hsaco")
        build_and_compile_dispatch(chip, d, self._mori_bc)
        build_and_compile_combine(chip, c, self._mori_bc)
        return d, c

    def _alloc_buffers(self):
        cfg = self.cfg
        npes = cfg.world_size
        k    = cfg.top_k
        mt   = cfg.max_num_inp_token_per_rank   # max_tok_per_rank
        mr   = cfg.max_recv                      # npes * mt
        hdim = cfg.hidden_dim

        # ── Symmetric buffers（mori.shmem Python API = host 侧 shmem bitcode）
        # 接收端 buffer 尺寸 = max_recv（npes*mt）确保最坏情况下不越界
        # 使用 torch.int16 作为 bf16 代理（相同 2 字节大小）
        self.shmem_disp_out_tok  = mori_shmem_create_tensor((mr * hdim,), torch.int16)
        self.shmem_disp_out_wts  = mori_shmem_create_tensor((mr * k,),    torch.float32)
        self.shmem_disp_out_idx  = mori_shmem_create_tensor((mr * k,),    torch.int32)
        self.shmem_tok_off       = mori_shmem_create_tensor((1,),          torch.int32)
        self.shmem_recv_tok_num  = mori_shmem_create_tensor((npes,),       torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,),         torch.int32)
        self.shmem_comb_inp_tok  = mori_shmem_create_tensor((mr * hdim,), torch.int16)
        self.shmem_comb_inp_wts  = mori_shmem_create_tensor((mr * k,),    torch.float32)
        # 输出 buffer 尺寸 = mt（本 rank 的 token 数 ≤ max_tok_per_rank）
        self.shmem_comb_out_tok  = mori_shmem_create_tensor((mt * hdim,), torch.int16)
        self.shmem_comb_out_wts  = mori_shmem_create_tensor((mt * k,),    torch.float32)
        self.shmem_xdev_bar_mem  = mori_shmem_create_tensor((npes,),       torch.int64)

        # ── 本地 GPU buffer（普通 device tensor）
        self.dest_pe_ctr  = torch.zeros(npes, dtype=torch.int32, device=self._device)
        self.disp_bar     = torch.zeros(1,    dtype=torch.int32, device=self._device)
        self.comb_bar     = torch.zeros(1,    dtype=torch.int32, device=self._device)
        self.total_recv   = torch.zeros(1,    dtype=torch.int32, device=self._device)
        sentinel = npes * mr  # npes^2 * mt, large enough for max_recv multiplier
        self.dest_tok_map = torch.full((mt * k,), sentinel,
                                        dtype=torch.int32, device=self._device)

    def reset(self):
        """清零所有计数器和信号 buffer，供下一轮使用。"""
        self.shmem_tok_off.fill_(0)
        self.shmem_recv_tok_num.fill_(0)
        self.shmem_xdev_bar_mem.fill_(0)
        # 清零来源映射（避免旧轮次残留值干扰下一轮精度检测）
        self.shmem_tok_id_to_src.fill_(0)
        self.dest_pe_ctr.fill_(0)
        self.disp_bar.fill_(0)
        self.comb_bar.fill_(0)
        self.total_recv.fill_(0)
        max_recv = self.cfg.world_size * self.cfg.max_num_inp_token_per_rank
        sentinel = self.cfg.world_size * max_recv  # npes^2 * max_tok
        self.dest_tok_map.fill_(sentinel)
        torch.cuda.synchronize()
        # 跨 PE 屏障：确保所有 PE 的 symmetric buffer 状态一致
        ms.shmem_barrier_all()

    def dispatch(self, input, weights, scales, indices,
                 block_num=-1, rdma_block_num=-1, warp_per_block=-1):
        """Dispatch tokens → remote experts via shmem P2P.

        Returns (out_tok, out_wts, None, out_idx, total_recv_num)
        """
        cfg    = self.cfg
        cur_tok = input.shape[0]
        bn     = block_num     if block_num     > 0 else cfg.block_num
        wpb    = warp_per_block if warp_per_block > 0 else cfg.warp_num_per_block

        self.shmem_tok_off.fill_(0)
        self.dest_pe_ctr.fill_(0)
        self.disp_bar.fill_(0)
        self.total_recv.fill_(0)
        max_recv_d = cfg.world_size * cfg.max_num_inp_token_per_rank
        sentinel = cfg.world_size * max_recv_d  # npes^2 * max_tok
        self.dest_tok_map.fill_(sentinel)

        inp_c = input.contiguous()
        wts_c = weights.contiguous()
        idx_c = indices.to(torch.int32).contiguous()

        params = [
            _p(inp_c),                    # inp_tok
            _p(idx_c),                    # token_indices
            _p(wts_c),                    # weights_buf
            _p(self.shmem_disp_out_tok),  # shmem_out_tok
            _p(self.shmem_disp_out_wts),  # shmem_out_wts
            _p(self.shmem_disp_out_idx),  # shmem_out_idx
            _p(self.shmem_tok_off),       # shmem_tok_off
            _p(self.shmem_recv_tok_num),  # recv_tok_num
            _p(self.dest_pe_ctr),         # dest_pe_ctr
            _p(self.disp_bar),            # dispatch_bar
            _p(self.dest_tok_map),        # dest_tok_map
            _p(self.shmem_tok_id_to_src), # tok_id_to_src
            _p(self.total_recv),          # total_recv
            _i32(cfg.rank),
            _i32(cfg.world_size),
            _i32(cur_tok),
            _i32(cfg.num_experts_per_rank),
            _i32(cfg.top_k),
            _i32(cfg.hidden_dim),
            _i32(cfg.elem_size),
            _i32(cfg.max_num_inp_token_per_rank),
            _i32(bn),
            _i32(wpb),
        ]
        hip_launch_kernel(self._disp_func, bn, wpb * 64, params)
        torch.cuda.synchronize()

        n     = int(self.total_recv[0].item())
        mr    = cfg.max_recv
        hdim  = cfg.hidden_dim
        k     = cfg.top_k

        out_tok = (self.shmem_disp_out_tok
                   .view(torch.bfloat16).view(mr, hdim)[:n]
                   .to(cfg.data_type))
        out_wts = self.shmem_disp_out_wts.view(mr, k)[:n]
        out_idx = self.shmem_disp_out_idx.view(mr, k)[:n]
        return out_tok, out_wts, None, out_idx, self.total_recv.clone()

    def combine(self, input, weights, indices,
                block_num=-1, rdma_block_num=-1, warp_per_block=-1,
                use_external_inp_buf=-1, call_reset=False):
        """Combine expert outputs via P2P read + weighted accumulate.

        Returns (out_tok, out_wts)
        """
        cfg           = self.cfg
        cur_tok       = indices.shape[0]
        total_recv_val = int(self.total_recv[0].item())
        bn  = block_num     if block_num     > 0 else cfg.block_num
        wpb = warp_per_block if warp_per_block > 0 else cfg.warp_num_per_block

        self.comb_bar.fill_(0)
        # barrier flag 单调递增，旧轮次的残留值不会误匹配新 flag
        self._xdev_flag += 1

        inp_c   = input.to(cfg.data_type).contiguous()
        has_wts = (weights is not None) and (weights.numel() > 0)
        wts_c   = weights.contiguous() if has_wts else \
                  torch.zeros(1, dtype=torch.float32, device=self._device)

        params = [
            _p(inp_c),                    # inp_tok
            _p(wts_c) if has_wts else _pnull(),  # weights_buf
            _p(self.shmem_comb_inp_tok),  # shmem_comb_inp
            _p(self.shmem_comb_out_tok),  # shmem_comb_out
            _p(self.shmem_comb_inp_wts),  # shmem_inp_wts
            _p(self.shmem_comb_out_wts),  # shmem_out_wts
            _p(self.shmem_xdev_bar_mem),  # xdev_bar_mem
            _p(self._xdev_flag),          # xdev_bar_flag
            _p(self.dest_tok_map),        # dest_tok_map
            _p(self.comb_bar),            # combine_bar
            _p(self.total_recv),          # total_recv_ptr
            _i32(cfg.rank),
            _i32(cfg.world_size),
            _i32(cur_tok),
            _i32(total_recv_val),
            _i32(cfg.top_k),
            _i32(cfg.hidden_dim),
            _i32(cfg.elem_size),
            _i32(cfg.max_num_inp_token_per_rank),
            _i32(bn),
            _i32(wpb),
        ]
        hip_launch_kernel(self._comb_func, bn, wpb * 64, params)
        torch.cuda.synchronize()

        mt   = cfg.max_num_inp_token_per_rank
        hdim = cfg.hidden_dim
        k    = cfg.top_k

        out_tok = (self.shmem_comb_out_tok
                   .view(torch.bfloat16).view(mt, hdim)[:cur_tok]
                   .to(cfg.data_type))
        out_wts = None
        if has_wts:
            out_wts = self.shmem_comb_out_wts.view(mt, k)[:cur_tok]

        if call_reset:
            self.reset()
        return out_tok, out_wts

    def get_dispatch_src_token_pos(self):
        """返回本轮接收到的 token 来源 (srcPe * max_tok + srcTokId)。"""
        torch.cuda.synchronize()
        n = int(self.total_recv[0].item())
        return self.shmem_tok_id_to_src[:n].clone()

    def get_registered_combine_input_buffer(self, dtype, hidden_dim=-1):
        h = hidden_dim if hidden_dim > 0 else self.cfg.hidden_dim
        return self.shmem_comb_inp_tok.view(torch.bfloat16).view(-1, h)
