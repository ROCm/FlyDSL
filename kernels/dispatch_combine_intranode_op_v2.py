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

        # 已知 BUG：dispatch kernel 在 warp_num_per_block >= 5 时出现 XGMI P2P 死锁。
        # 根本原因：Phase 2 的 store_i32_system(ptr_p2p_addr) 对某些 rank-PE 组合
        # 返回无效地址（p2pPeerPtrs 为 0），写操作访问非法地址或信号不可见。
        # 安全配置：warp_num_per_block <= 4。使用大 block_num（如 bn=80）时保持 wpb=4。
        if config.warp_num_per_block > 4 and r == 0:
            import warnings
            warnings.warn(
                f"[FlyDSL] warp_num_per_block={config.warp_num_per_block} > 4 会导致 "
                f"dispatch kernel 死锁（已知 XGMI P2P bug）。请使用 warp_num_per_block=4。",
                stacklevel=2
            )

        # 先分配 symmetric buffer（顺序：alloc → barrier → compile）
        self._alloc_buffers()
        ms.shmem_barrier_all()

        # 创建 @flyc.jit launcher（首次调用时自动编译 + shmem_module_init）
        print(f"[v2] Rank {r}: creating v2 dispatch jit...")
        self._disp_fn = make_dispatch_jit(
            rank=r, npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.top_k,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=config.warp_num_per_block,
            data_type=config.data_type,
        )

        print(f"[v2] Rank {r}: creating v2 combine jit...")
        self._comb_fn = make_combine_jit(
            rank=r, npes=config.world_size,
            experts_per_token=config.top_k,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=config.warp_num_per_block,
            data_type=config.data_type,
        )

        # combine 用的单调递增 barrier flag。
        # 初始值必须为 1（而非 0）：reset() 会把 shmem_xdev_bar_mem 清零，
        # 若 flag=0 则第一次 combine 的 wait_until_equals(slot, 0) 立即满足，
        # 跳过跨 GPU 屏障。与 mori 的 crossDeviceBarrierFlag[0]=1 对齐。
        self._xdev_flag = torch.ones(1, dtype=torch.int64, device=self._dev)

        # Fix3: 预缓存固定 shmem buffer 地址的 fx.Int64 包装（地址在 _alloc_buffers 后不变）。
        # 每次 dispatch/combine 调用避免重复创建，节省约 5×3μs ≈ 15μs/call。
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
        self._fx_comb_bar  = fx.Int64(self.comb_bar.data_ptr())
        self._fx_trecv     = fx.Int64(self.total_recv.data_ptr())

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
        """Dispatch tokens → remote experts via shmem P2P。

        Fix1: 移除 5 个冗余 fill_()，前提是调用方已调用 reset()。
          reset() 已清零：shmem_tok_off, dest_pe_ctr, disp_bar, total_recv, dest_tok_map。
          dispatch kernel Phase2/3 末尾会自行重置；stale dest_tok_map 条目不被读取。

        Fix3: 固定 shmem 地址使用预缓存的 fx.Int64 对象，避免每次重建（省 ~5×3μs）。
        """
        cfg     = self.cfg
        cur_tok = input.shape[0]
        inp_c = input.contiguous()
        wts_c = weights.contiguous()
        idx_c = indices.to(torch.int32).contiguous()

        # 通过 @flyc.jit 调用 kernel（Fix3：固定地址使用预缓存对象）
        self._disp_fn(
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
            cur_tok,  # 裸 Python int，不要用 fx.Int32（会传指针而非值）
        )
        torch.cuda.synchronize()

        n    = int(self.total_recv[0].item())
        mr   = cfg.max_recv
        hdim = cfg.hidden_dim
        k    = cfg.top_k

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

        self.comb_bar.fill_(0)
        # flag 递增已移至 GPU kernel 内（gwtid==0 的线程在 barrier 完成后 atomic_add_i64_at）

        inp_c = input.to(cfg.data_type).contiguous()

        # Fix3：固定地址使用预缓存的 fx.Int64 对象
        # 注意：cur_tok/total_recv_val 保持裸 Python int（不要包装成 fx.Int32！）
        # fly_pointers(fx.Int32(n)) 传递的是指针地址而非整数值，会导致 kernel 收到错误的
        # cur_tok（约为 0x7f... 的大地址），使 Stage 3 循环次数异常 → combine 无限等待。
        self._comb_fn(
            fx.Int64(inp_c.data_ptr()),
            self._fx_comb_inp,
            self._fx_comb_out,
            self._fx_xdb_mem,
            fx.Int64(self._xdev_flag.data_ptr()),
            self._fx_tok_map,
            self._fx_comb_bar,
            self._fx_trecv,
            cur_tok,
            total_recv_val,
        )
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
