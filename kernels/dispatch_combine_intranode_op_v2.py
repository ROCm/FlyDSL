"""
FlyDSL v2 DispatchCombine IntraNode 算子包装器。

与 v1 (`dispatch_combine_intranode_op.py`) 的区别：
- kernel 采用 Python FlyDSL 语法（`@flyc.kernel` + `mori_shmem.*`）
- 所有 buffer 地址以 fx.Int64 传入内核（避免 fly.memref → LLVM ptr 时序问题）
- 外部 API 与 v1 完全兼容（可直接替换）
"""
from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as dist
import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

from .dispatch_combine_intranode_v2 import (
    _find_mori_shmem_bc,
    build_and_compile_dispatch,
    build_and_compile_combine,
)

_hip = ctypes.CDLL("libamdhip64.so")


def _hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


# ── ctypes 帮助函数 ─────────────────────────────────────────────────────────

def _i64(val) -> ctypes.c_int64:
    """int64 标量（用于传 buffer 基地址）"""
    return ctypes.c_int64(int(val))


def _i32(val) -> ctypes.c_int32:
    return ctypes.c_int32(int(val))


def _ptr(tensor: torch.Tensor) -> ctypes.c_int64:
    """将 tensor 的 data_ptr() 作为 i64 传入内核。"""
    return ctypes.c_int64(tensor.data_ptr())


@dataclass
class FlyDSLDispatchCombineConfigV2:
    """与 FlyDSLDispatchCombineConfig (v1) 相同字段，可互换使用。"""
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
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
        return self.world_size * self.max_num_inp_token_per_rank


class FlyDSLDispatchCombineIntraNodeOpV2:
    """FlyDSL v2 IntraNode Dispatch+Combine 算子。

    使用 Python FlyDSL 语法编写的 kernel（v2），功能与 v1 相同。
    接口与 FlyDSLDispatchCombineIntraNodeOp (v1) 完全兼容。
    """

    def __init__(self, config):
        self.cfg    = config
        self._dev   = torch.device("cuda", config.rank)
        self._mori_bc = _find_mori_shmem_bc()
        tmp  = tempfile.gettempdir()
        chip = config.chip
        r    = config.rank

        # HSACO 文件名包含关键编译参数，避免不同配置间的缓存污染
        mt  = config.max_num_inp_token_per_rank
        epr = config.num_experts_per_rank
        ept = config.num_experts_per_token
        hdim = config.hidden_dim

        print(f"[v2] Rank {r}: compiling v2 dispatch kernel...")
        self._disp_kernel = build_and_compile_dispatch(
            rank=r, npes=config.world_size,
            experts_per_rank=epr,
            experts_per_token=ept,
            hidden_dim=hdim,
            max_tok_per_rank=mt,
            block_num=config.block_num,
            warp_num_per_block=config.warp_num_per_block,
            data_type=config.data_type,
            chip=chip,
            out_path=os.path.join(
                tmp, f"v2_disp_{chip}_ep{config.world_size}"
                     f"_mt{mt}_h{hdim}_k{ept}_r{r}.hsaco"),
            shmem_bc=self._mori_bc,
        )

        print(f"[v2] Rank {r}: compiling v2 combine kernel...")
        self._comb_kernel = build_and_compile_combine(
            rank=r, npes=config.world_size,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=config.warp_num_per_block,
            data_type=config.data_type,
            chip=chip,
            out_path=os.path.join(
                tmp, f"v2_comb_{chip}_ep{config.world_size}"
                     f"_mt{mt}_h{hdim}_k{ept}_r{r}.hsaco"),
            shmem_bc=self._mori_bc,
        )

        # 先分配 symmetric buffer，再 shmem_module_init（顺序不能颠倒）
        self._alloc_buffers()
        ms.shmem_barrier_all()
        # 手动加载 HSACO，注册 shmem_module_init hook，然后调用
        self._disp_kernel._ensure_loaded()
        ms.shmem_module_init(self._disp_kernel._mod.value)
        self._comb_kernel._ensure_loaded()
        ms.shmem_module_init(self._comb_kernel._mod.value)
        torch.cuda.synchronize()
        # 再次 barrier 确保所有 rank 均完成 shmem_module_init
        ms.shmem_barrier_all()

        # combine 用的单调递增 barrier flag
        self._xdev_flag = torch.zeros(1, dtype=torch.int64, device=self._dev)

    def _alloc_buffers(self):
        cfg  = self.cfg
        npes = cfg.world_size
        k    = cfg.num_experts_per_token
        mt   = cfg.max_num_inp_token_per_rank
        mr   = cfg.max_recv   # npes * mt
        hdim = cfg.hidden_dim

        # ── Symmetric shmem buffers（mori.shmem Python API 分配）
        self.shmem_disp_out_tok  = mori_shmem_create_tensor((mr * hdim,), torch.int16)
        self.shmem_disp_out_wts  = mori_shmem_create_tensor((mr * k,),    torch.float32)
        self.shmem_disp_out_idx  = mori_shmem_create_tensor((mr * k,),    torch.int32)
        self.shmem_tok_off       = mori_shmem_create_tensor((1,),          torch.int32)
        self.shmem_recv_tok_num  = mori_shmem_create_tensor((npes,),       torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,),         torch.int32)
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

    def reset(self):
        """清零所有计数器和信号 buffer（供下一轮使用）。

        shmem_barrier_all 确保 NIC 操作（prev round Phase 2/3 的远端写）
        在下一轮 Phase 1 开始前完全完成。
        调用方需确保 BOTH ranks 在同一时间点调用 reset()，
        否则 barrier 可能与其他操作配对，导致竞争。
        """
        self.shmem_tok_off.fill_(0)
        self.shmem_recv_tok_num.fill_(0)
        self.shmem_xdev_bar_mem.fill_(0)
        self.shmem_tok_id_to_src.fill_(0)
        self.dest_pe_ctr.fill_(0)
        self.disp_bar.fill_(0)
        self.comb_bar.fill_(0)
        self.total_recv.fill_(0)
        max_recv = self.cfg.world_size * self.cfg.max_num_inp_token_per_rank
        sentinel = self.cfg.world_size * max_recv
        self.dest_tok_map.fill_(sentinel)
        torch.cuda.synchronize()
        # dist.barrier ensures both ranks reach this point before shmem_barrier_all
        # This prevents wrong pairing of shmem barriers when ranks have different timing
        if dist.is_initialized():
            dist.barrier()
        ms.shmem_barrier_all()

    def dispatch(self, input, weights, scales, indices,
                 block_num=-1, rdma_block_num=-1, warp_per_block=-1):
        """Dispatch tokens → remote experts via shmem P2P。"""
        cfg     = self.cfg
        cur_tok = input.shape[0]
        bn      = block_num    if block_num    > 0 else cfg.block_num
        wpb     = warp_per_block if warp_per_block > 0 else cfg.warp_num_per_block

        # 初始化 per-round 计数器
        self.shmem_tok_off.fill_(0)
        self.dest_pe_ctr.fill_(0)
        self.disp_bar.fill_(0)
        self.total_recv.fill_(0)
        sentinel = cfg.world_size * cfg.max_recv
        self.dest_tok_map.fill_(sentinel)

        inp_c = input.contiguous()
        wts_c = weights.contiguous()
        idx_c = indices.to(torch.int32).contiguous()

        # v2 kernel：所有参数均为 i64 地址 + i32 标量
        args = [
            _ptr(inp_c),                      # addr_inp_tok
            _ptr(idx_c),                      # addr_idx
            _ptr(wts_c),                      # addr_wts
            _ptr(self.shmem_disp_out_tok),    # addr_out_tok
            _ptr(self.shmem_disp_out_wts),    # addr_out_wts
            _ptr(self.shmem_disp_out_idx),    # addr_out_idx
            _ptr(self.shmem_tok_off),         # addr_tok_off
            _ptr(self.shmem_recv_tok_num),    # addr_recv_num
            _ptr(self.dest_pe_ctr),           # addr_dest_ctr
            _ptr(self.disp_bar),              # addr_disp_bar
            _ptr(self.dest_tok_map),          # addr_tok_map
            _ptr(self.shmem_tok_id_to_src),   # addr_tis
            _ptr(self.total_recv),            # addr_total_rv
            _i32(cur_tok),                    # cur_tok
        ]
        self._disp_kernel.launch(
            grid=(bn, 1, 1), block=(wpb * 64, 1, 1), args=args)
        torch.cuda.synchronize()

        n    = int(self.total_recv[0].item())
        mr   = cfg.max_recv
        hdim = cfg.hidden_dim
        k    = cfg.num_experts_per_token

        out_tok = (self.shmem_disp_out_tok.view(torch.bfloat16)
                   .view(mr, hdim)[:n].to(cfg.data_type))
        out_wts = self.shmem_disp_out_wts.view(mr, k)[:n]
        out_idx = self.shmem_disp_out_idx.view(mr, k)[:n]
        return out_tok, out_wts, None, out_idx, self.total_recv.clone()

    def combine(self, input, weights, indices,
                block_num=-1, rdma_block_num=-1, warp_per_block=-1,
                use_external_inp_buf=-1, call_reset=False):
        """Combine expert outputs via P2P read + weighted accumulate。"""
        cfg            = self.cfg
        cur_tok        = indices.shape[0]
        total_recv_val = int(self.total_recv[0].item())
        bn  = block_num    if block_num    > 0 else cfg.block_num
        wpb = warp_per_block if warp_per_block > 0 else cfg.warp_num_per_block

        self.comb_bar.fill_(0)
        self._xdev_flag += 1

        inp_c = input.to(cfg.data_type).contiguous()

        args = [
            _ptr(inp_c),                      # addr_inp_tok
            _ptr(self.shmem_comb_inp_tok),    # addr_comb_inp
            _ptr(self.shmem_comb_out_tok),    # addr_comb_out
            _ptr(self.shmem_xdev_bar_mem),    # addr_xdb_mem
            _ptr(self._xdev_flag),            # addr_xdb_flag
            _ptr(self.dest_tok_map),          # addr_tok_map
            _ptr(self.comb_bar),              # addr_comb_bar
            _ptr(self.total_recv),            # addr_trecv
            _i32(cur_tok),                    # cur_tok
            _i32(total_recv_val),             # total_recv_val
        ]
        self._comb_kernel.launch(
            grid=(bn, 1, 1), block=(wpb * 64, 1, 1), args=args)
        torch.cuda.synchronize()

        mt   = cfg.max_num_inp_token_per_rank
        hdim = cfg.hidden_dim

        out_tok = (self.shmem_comb_out_tok.view(torch.bfloat16)
                   .view(mt, hdim)[:cur_tok].to(cfg.data_type))
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
