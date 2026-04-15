"""
FlyDSL dispatch/combine intranode kernels — v2 (Python FlyDSL syntax).

所有 buffer 基地址以 fx.Int64 传入（避免 fly.memref → LLVM pointer 时序问题）。
算子逻辑与 EpDispatchIntraNodeKernel / EpCombineIntraNodeKernel 一致。

FlyDSL 编码规则（来自实际调试经验）
-------------------------------------
1. 编译期常量 → 闭包变量，不在 kernel 参数列表中
2. 动态 if 条件必须是函数调用形式（icmp_eq_i32 等）
3. for 循环不能嵌套在 scf.if 闭包内（SSA 作用域限制）
4. scf.ForOp 归纳变量是 index 类型 → 用 idx_to_i32() 转回 i32
5. 所有 tensor 地址以 fx.Int64 传入，内部用 load_i32_at / store_i32_at 等访问
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_HERE, "../python"), "/home/yashao/FlyDSL/python",
           "/home/yashao/mori/python"]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr   # Python-level loop (no scf.for transform)
from flydsl.expr import arith
from flydsl.expr.typing import Stream
import torch  # 用于 element_size

import mori.ir.flydsl as mori_shmem

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr import T

from flydsl.expr.lowlevel import (
    ballot_i64,
    readlane,
    fence_one_as_seq_cst,
    fence_one_as_release,
    load_v4i32,
    load_v4i32_global,
    load_v4i32_global_nt,
    store_v4i32_global,
    store_v4i32_global_nt,
    store_v4i32_shmem,
    load_i32_at,
    load_f32_at,
    load_i64_at,
    store_i32_at,
    load_i32_global_at,
    load_i32_global_nt_at,
    load_f32_global_at,
    load_i64_global_at,
    store_i32_global_at,
    store_i32_global_nt_at,
    atomic_add_i32_global_at,
    store_i32_global,
    store_i32_system,
    store_i32_shmem,
    store_i64_system,
    store_i64_global_system,
    atomic_add_i64_global_at,
    zext_i32_to_i64,
    add_i64,
    const_i32,
    const_i64,
    as_index,
    idx_to_i32,
    atomic_add_i32_at,
    atomic_add_i64_at,
    atomic_fetch_add_i32_global,
    bitcast_i32_to_v2bf16,
    bitcast_v2bf16_to_i32,
    divui,
    remui,
)
from flydsl.expr.lowlevel import _unwrap as _lv_unwrap


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _sel_pe(rem_list, dest_pe):
    """用 arith.select 链实现运行时动态索引（不能用 Python list[ArithValue]）。"""
    result = rem_list[-1]
    for pe in reversed(range(len(rem_list) - 1)):
        result = arith.select(arith.cmpi(arith.CmpIPredicate.eq, dest_pe, const_i32(pe)), rem_list[pe], result)
    return result


# ============================================================
# Dispatch Kernel Factory
# ============================================================

def make_dispatch_kernel(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
):
    """创建 dispatch intranode @flyc.kernel（编译期常量作为闭包）。

    所有 buffer 基地址以 fx.Int64 传入，内部用 load/store_*_at 辅助函数访问。
    """
    max_recv = npes * max_tok_per_rank
    n_i32    = hidden_dim >> 1          # hidden_dim/2（bf16 用 i32 对存储）

    @flyc.kernel
    def ep_dispatch_intranode(
        addr_inp_tok:  fx.Int64,  # [cur_tok, hidden_dim]  bf16
        addr_idx:      fx.Int64,  # [cur_tok, k]           i32  (token_indices)
        addr_wts:      fx.Int64,  # [cur_tok, k]           f32  (weights_buf)
        addr_out_tok:  fx.Int64,  # shmem_out_tok
        addr_out_wts:  fx.Int64,  # shmem_out_wts
        addr_out_idx:  fx.Int64,  # shmem_out_idx
        addr_tok_off:  fx.Int64,  # shmem_tok_off (i32[1])
        addr_recv_num: fx.Int64,  # recv_tok_num  (i32[npes])
        addr_dest_ctr: fx.Int64,  # dest_pe_ctr   (i32[npes])
        addr_disp_bar: fx.Int64,  # dispatch_bar  (i32[1])
        addr_tok_map:  fx.Int64,  # dest_tok_map  (i32[cur_tok*k])
        addr_tis:      fx.Int64,  # tok_id_to_src (i32[max_recv])
        addr_total_rv: fx.Int64,  # total_recv    (i32[1])
        # P2P 地址数组：预计算的各 shmem buffer 对所有 PE 的远程地址 (i64[npes])
        addr_p2p_tok_off:  fx.Int64,
        addr_p2p_tis:      fx.Int64,
        addr_p2p_out_wts:  fx.Int64,
        addr_p2p_out_idx:  fx.Int64,
        addr_p2p_out_tok:  fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        cur_tok:       fx.Int32,  # 动态：本轮实际 token 数
    ):
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63                           # lane ID within warp，范围 [0, 63]
        warp   = tid >> 6                           # warp ID within block
        gw_id  = bid * warp_num_per_block + warp    # global warp ID（跨所有 block）
        gw_num = block_num * warp_num_per_block     # global warp 总数（grid-stride 步长）
        limit  = cur_tok * experts_per_token        # 外层循环总迭代数

        # ── Phase 1: 发送 token ───────────────────────────────────────────────
        # 使用预计算 P2P 地址数组（消除 ptr_p2p extern 调用开销）：
        # load_i64_at(addr_p2p_xxx, dest_pe) 替代 mori_shmem.ptr_p2p(addr_xxx, rank, dest_pe)
        # 等价于 mori 的 GetAs<T*>(destPe)（从设备内存数组读单个指针）。
        for i in range(as_index(gw_id), as_index(limit), as_index(gw_num)):
            i = idx_to_i32(i)
            src_tok  = divui(i, experts_per_token)
            j        = remui(i, experts_per_token)
            # 两个 idx load 并行发射（消除不必要的 s_waitcnt vmcnt(0)）
            dest_exp = load_i32_global_at(addr_idx, i)
            safe_lane    = arith.select(arith.cmpi(arith.CmpIPredicate.ult, lane, j), lane, const_i32(0))
            lane_exp     = load_i32_global_at(addr_idx, src_tok * experts_per_token + safe_lane)
            # divui 延后到两个 load 都发射之后
            dest_pe      = divui(dest_exp, experts_per_rank)
            lane_pe      = divui(lane_exp, experts_per_rank)
            dup_per_lane = arith.select(
                arith.cmpi(arith.CmpIPredicate.eq, lane_pe, dest_pe),
                arith.select(arith.cmpi(arith.CmpIPredicate.ult, lane, j), lane, const_i32(64)),
                const_i32(64))
            dup_ballot   = ballot_i64(arith.cmpi(arith.CmpIPredicate.ult, dup_per_lane, const_i32(64)))
            is_dup       = arith.cmpi(arith.CmpIPredicate.ne, dup_ballot, const_i64(0))

            # 原子分配 destTokId：
            #   - lane0 + non-dup：XGMI 硬件原子 fetch+add（单条 global_atomic_add_ret 指令）
            #   - lane0 + dup：跳过 AMO，返回 0
            #   - lanes 1-63：不执行原子，readlane 广播 lane0 的结果
            from flydsl._mlir.dialects import scf as _scf_d
            from flydsl._mlir.ir import InsertionPoint as _IP
            _i32_ty = T.i32()
            _if_lane0 = _scf_d.IfOp(_lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, lane, const_i32(0))),
                                     [_i32_ty], has_else=True)
            with _IP(_if_lane0.then_block):
                _if_nodup = _scf_d.IfOp(
                    _lv_unwrap(arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, const_i64(0))),
                    [_i32_ty], has_else=True)
                with _IP(_if_nodup.then_block):
                    _old_tok = atomic_fetch_add_i32_global(
                        load_i64_global_at(addr_p2p_tok_off, dest_pe),
                        const_i32(1))
                    _scf_d.YieldOp([_lv_unwrap(_old_tok)])
                with _IP(_if_nodup.else_block):
                    _scf_d.YieldOp([_lv_unwrap(const_i32(0))])
                _scf_d.YieldOp([_if_nodup.result])
            with _IP(_if_lane0.else_block):
                _scf_d.YieldOp([_lv_unwrap(const_i32(0))])
            dest_tok_all = readlane(_if_lane0.result, 0)

            # 写入 dest_tok_map[i]（lane0）
            # sentinel = npes * max_recv，保证解码后 dest_pe_j >= npes → 视为无效
            sentinel_val = npes * max_recv
            dtm_val = arith.select(is_dup, const_i32(sentinel_val),
                                  dest_pe * max_recv + dest_tok_all)
            if arith.cmpi(arith.CmpIPredicate.eq, lane, const_i32(0)):
                store_i32_global_at(addr_tok_map, i, dtm_val)

            # 仅非重复时写入 tok_id_to_src 和更新 dest_pe_ctr（lane0）
            # P2P global store（与 mori GetAs<T*>(destPe)[offset] = val 一致）
            if arith.cmpi(arith.CmpIPredicate.eq, lane, const_i32(0)):
                if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, const_i64(0)):
                    src_enc  = rank * max_tok_per_rank + src_tok
                    store_i32_global(
                        load_i64_global_at(addr_p2p_tis, dest_pe),
                        dest_tok_all, src_enc)
                    ctr_addr = addr_dest_ctr + zext_i32_to_i64(dest_pe) * 4
                    atomic_add_i32_global_at(ctr_addr, const_i32(1))

            # 写入权重 + 索引（lanes 0..k-1，仅非重复路径）
            # P2P global store（与 mori 一致）
            if arith.cmpi(arith.CmpIPredicate.ult, lane, const_i32(experts_per_token)):
                if arith.cmpi(arith.CmpIPredicate.eq, dup_ballot, const_i64(0)):
                    wt_src   = src_tok * experts_per_token + lane
                    wt_val   = load_f32_global_at(addr_wts, wt_src)
                    ix_val   = load_i32_global_at(addr_idx, wt_src)
                    dst_slot = dest_tok_all * experts_per_token + lane
                    store_i32_global(
                        load_i64_global_at(addr_p2p_out_wts, dest_pe),
                        dst_slot, wt_val.bitcast(T.i32()))
                    store_i32_global(
                        load_i64_global_at(addr_p2p_out_idx, dest_pe),
                        dst_slot, ix_val)

            # 写入 token embedding（inp_tok 不在 shmem heap，用 XGMI 直接写）
            # 优化 1: is_dup 时 copy_end == lane4，循环零迭代（等价于 mori 的 continue）
            # 优化 2: stride=512，每次先发 2 个 global_load_dwordx4 再 2 个 store，隐藏内存延迟
            tok_remote = load_i64_global_at(addr_p2p_out_tok, dest_pe) + \
                zext_i32_to_i64(dest_tok_all * hidden_dim * hidden_elem_size)
            inp_src_b  = addr_inp_tok + zext_i32_to_i64(
                src_tok * hidden_dim * hidden_elem_size)
            lane4    = lane * 4
            copy_end = arith.select(is_dup, lane4, const_i32(n_i32))
            for ec4 in range(as_index(lane4), as_index(copy_end), 512):
                ec4      = idx_to_i32(ec4)
                ec4_byt0 = zext_i32_to_i64(ec4) * 4
                ec4_byt1 = zext_i32_to_i64(ec4 + 256) * 4
                vec4_0   = load_v4i32_global(inp_src_b + ec4_byt0)
                vec4_1   = load_v4i32_global(inp_src_b + ec4_byt1)
                store_v4i32_global(vec4_0, tok_remote + ec4_byt0)
                store_v4i32_global(vec4_1, tok_remote + ec4_byt1)

        # ── Phase 2: 栅栏 + 发送 token 数量信号 ──────────────────────────────
        # Phase 1 全部使用 XGMI P2P store（非 NIC put），无需 quiet。
        # sync_threads + atomicAdd barrier 确保本 block 所有 warp 的 P2P store 已完成。
        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, const_i32(0)):
            atomic_add_i32_global_at(addr_disp_bar, const_i32(1))

        rtn_local_off = zext_i32_to_i64(const_i32(rank)) * 4
        for dest_pe in range(as_index(lane), as_index(npes), 64):
            dest_pe = idx_to_i32(dest_pe)
            if arith.cmpi(arith.CmpIPredicate.eq, gw_id, const_i32(0)):
                mori_shmem.int32_wait_until_equals(addr_disp_bar, block_num)
                store_i32_global_at(addr_disp_bar, const_i32(0), const_i32(0))
                nsig       = load_i32_global_at(addr_dest_ctr, dest_pe) + 1
                rtn_remote = load_i64_global_at(addr_p2p_recv_num, dest_pe) + rtn_local_off
                mori_shmem.int32_wait_until_equals(rtn_remote, 0)
                # fence 已移除：barrier wait 已保证 Phase 1 P2P store 可见性
                # mori 同样只用 AtomicStoreRelaxedSystem，无额外 fence
                store_i32_system(rtn_remote, const_i32(0), nsig)

        # ── Phase 3: 接收信号，累计 total_recv ───────────────────────────────
        for src_pe in range(as_index(lane), as_index(npes), 64):
            src_pe = idx_to_i32(src_pe)
            if arith.cmpi(arith.CmpIPredicate.eq, gw_id, const_i32(0)):
                rtn_src  = addr_recv_num + zext_i32_to_i64(src_pe) * 4
                sig_val  = mori_shmem.int32_wait_until_greater_than(rtn_src, 0)
                recv_cnt = sig_val - 1
                store_i32_system(rtn_src, const_i32(0), const_i32(0))
                atomic_add_i32_global_at(addr_total_rv, recv_cnt)
                store_i32_global_at(addr_dest_ctr, src_pe, const_i32(0))

        if arith.cmpi(arith.CmpIPredicate.eq, gw_id, const_i32(0)):
            if arith.cmpi(arith.CmpIPredicate.eq, lane, const_i32(0)):
                store_i32_global_at(addr_tok_off, const_i32(0), const_i32(0))

    return ep_dispatch_intranode


# ============================================================
# Combine Kernel Factory
# ============================================================

def make_combine_kernel(
    *,
    rank: int,
    npes: int,
    experts_per_token: int,
    hidden_dim: int,
    hidden_elem_size: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
):
    """创建 combine intranode @flyc.kernel（nop2p 模式）。

    Stage 1: P2P scatter — 将 expert 处理后的 token 写到远端 shmem_comb_inp
    Stage 2: CrossDeviceBarrier（所有 PE 就绪后互通知）
    Stage 3: 本地读 + WarpAccum → shmem_comb_out（从本地 shmem_comb_inp 读取）
    """
    max_recv   = npes * max_tok_per_rank
    n_i32      = hidden_dim >> 1
    nbytes     = hidden_dim * hidden_elem_size   # bytes per token (Python int)
    tok_stride = n_i32 * 4                       # bytes per token in i32-addressed buffer

    # 预计算 power-of-2 移位量，用于替换 divui/remui
    def _log2_if_pow2(v):
        """如果 v 是 2 的幂，返回 log2(v)；否则返回 None。"""
        if v > 0 and (v & (v - 1)) == 0:
            return v.bit_length() - 1
        return None
    _log2_max_tok = _log2_if_pow2(max_tok_per_rank)
    _log2_max_recv = _log2_if_pow2(max_recv)
    _mask_max_tok = max_tok_per_rank - 1 if _log2_max_tok is not None else None

    # ── LDS 分配：Stage 1 需要 P2P 基地址表（写入远端）──
    allocator = SmemAllocator(None, arch="gfx942")
    p2p_base_offset = allocator._align(allocator.ptr, 8)
    p2p_base_size = npes * 8
    allocator.ptr = p2p_base_offset + p2p_base_size


    @flyc.kernel
    def ep_combine_intranode(
        addr_inp_tok:  fx.Int64,   # inp_tok  基地址（expert 处理后的 token）
        addr_comb_inp: fx.Int64,   # shmem_comb_inp 基地址（symmetric）
        addr_comb_out: fx.Int64,   # shmem_comb_out 基地址（symmetric）
        addr_xdb_mem:  fx.Int64,   # xdev_bar_mem   基地址（u64[npes]）
        addr_xdb_flag: fx.Int64,   # xdev_bar_flag  基地址（u64[1]）
        addr_tok_map:  fx.Int64,   # dest_tok_map   基地址（i32[cur_tok*k]）
        addr_comb_bar: fx.Int64,   # combine_bar    基地址（i32[1]）
        addr_trecv:    fx.Int64,   # total_recv_ptr 基地址（i32[1]）
        addr_tis:      fx.Int64,   # tok_id_to_src  基地址（i32[max_recv]，symmetric）
        addr_p2p_comb_inp: fx.Int64,  # 预计算 P2P 地址数组 i64[npes]
        addr_p2p_xdb_mem:  fx.Int64,  # 预计算 P2P 地址数组 i64[npes]
    ):
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63
        warp   = tid >> 6
        gw_id  = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        gwtid  = bid * (warp_num_per_block * 64) + tid

        total_recv_val = load_i32_global_at(addr_trecv, const_i32(0))
        cur_flag = load_i64_at(addr_xdb_flag, const_i32(0))

        # ── LDS P2P 基地址表（用于 Stage 1 P2P write）──
        base_ptr = allocator.get_base()
        _lds_p2p_bases = SmemPtr(base_ptr, p2p_base_offset, T.i64(),
                                 shape=(npes,))
        _lds_p2p_bases.get()

        # 从预计算 P2P 地址数组加载到 LDS（消除 ptr_p2p extern 调用）
        if arith.cmpi(arith.CmpIPredicate.ult, lane, const_i32(npes)):
            _p2p_base = load_i64_global_at(addr_p2p_comb_inp, lane)
            _lds_p2p_bases.store(_p2p_base, [as_index(lane)])
        fx.barrier()

        # ── Stage 1: P2P scatter（nop2p 模式）──────────────────────────────────
        # 对齐 mori UseP2PRead=false (intranode.hpp:268-289):
        #   destTokId = tok_id_to_src[tokenIdx]
        #   destPe = destTokId / MaxRecvPerRank
        #   destLocalTokId = destTokId % MaxRecvPerRank
        #   P2P write → shmem_comb_inp[destPe] @ (myPe * MaxRecvPerRank + destLocalTokId)
        # Stage 3 再从本地 shmem_comb_inp 读取（全部本地内存访问）。
        n_chunks = nbytes // 16
        for tok_i in range(as_index(gw_id), as_index(total_recv_val), as_index(gw_num)):
            tok_i    = idx_to_i32(tok_i)
            # 解码目标 PE 和本地 token ID
            dest_enc = load_i32_global_at(addr_tis, tok_i)
            if _log2_max_tok is not None:
                dest_pe  = dest_enc >> const_i32(_log2_max_tok)
                dest_lid = dest_enc & const_i32(_mask_max_tok)
            else:
                dest_pe  = divui(dest_enc, max_tok_per_rank)
                dest_lid = remui(dest_enc, max_tok_per_rank)
            # 远端 shmem_comb_inp 上的目标偏移：(myPe * max_tok + destLocalTokId) * nbytes
            _pe_base   = _lds_p2p_bases.load([as_index(dest_pe)])
            _dest_off  = zext_i32_to_i64(const_i32(rank) * max_tok_per_rank + dest_lid) * nbytes
            _dest_base = _lv_unwrap(_pe_base) + _dest_off
            # 本地 inp_tok 偏移
            _src_base  = addr_inp_tok + zext_i32_to_i64(tok_i) * nbytes
            for cj in range(as_index(lane), as_index(n_chunks), as_index(128)):
                cj      = idx_to_i32(cj)
                cj_off  = zext_i32_to_i64(cj) * 16
                vec4_a  = load_v4i32_global(_src_base + cj_off)
                cj2     = cj + const_i32(64)
                cj2_off = zext_i32_to_i64(cj2) * 16
                vec4_b  = load_v4i32_global(_src_base + cj2_off)
                store_v4i32_global(vec4_a, _dest_base + cj_off)
                store_v4i32_global(vec4_b, _dest_base + cj2_off)

        # ── Stage 2: CrossDeviceBarrier ───────────────────────────────────────
        # 与 mori CrossDeviceBarrierIntraNodeKernel 完全一致的结构
        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, const_i32(0)):
            atomic_add_i32_global_at(addr_comb_bar, const_i32(1))

        if arith.cmpi(arith.CmpIPredicate.ult, gwtid, const_i32(npes)):
            mori_shmem.int32_wait_until_equals(addr_comb_bar, block_num)
            store_i32_global_at(addr_comb_bar, const_i32(0), const_i32(0))
            xdb_remote = load_i64_global_at(addr_p2p_xdb_mem, gwtid) + \
                zext_i32_to_i64(const_i32(rank)) * 8
            store_i64_global_system(xdb_remote, cur_flag)

        if arith.cmpi(arith.CmpIPredicate.eq, gwtid, const_i32(0)):
            atomic_add_i64_global_at(addr_xdb_flag, const_i64(1))

        if arith.cmpi(arith.CmpIPredicate.ult, tid, const_i32(npes)):
            peer_slot = addr_xdb_mem + zext_i32_to_i64(tid) * 8
            mori_shmem.uint64_wait_until_equals(peer_slot, cur_flag)

        fx.barrier()
        if arith.cmpi(arith.CmpIPredicate.eq, tid, const_i32(0)):
            store_i32_global_at(addr_trecv, const_i32(0), const_i32(0))

        # ── Stage 3: 本地读 + WarpAccum ──────────────────────────────────────
        # 从本地 shmem_comb_inp 读取数据（Stage 1 已通过 P2P 写入）。

        from flydsl._mlir.dialects import llvm as _llvm_d
        from flydsl._mlir.dialects import scf as _scf_d
        from flydsl._mlir.ir import InsertionPoint as _IP
        _v2bf16   = T.VectorType.get([2], T.bf16())
        _v2f32    = T.VectorType.get([2], T.f32())
        _i32t     = T.i32()
        _ptr_g    = _llvm_d.PointerType.get(address_space=1)

        # VecBytes=4 + exec mask + tok_map 提前加载
        n_elems = n_i32   # hidden_dim / 2
        _safe_recv = arith.select(
            arith.cmpi(arith.CmpIPredicate.eq, total_recv_val, const_i32(0)), const_i32(1), total_recv_val)
        wpt_v2    = (gw_num + _safe_recv - 1) // _safe_recv
        hpw_v2    = (n_elems + wpt_v2 - 1) // wpt_v2
        s3_lim2   = total_recv_val * wpt_v2

        for si in range(as_index(gw_id), as_index(s3_lim2), as_index(gw_num)):
            si      = idx_to_i32(si)
            tok_id  = divui(si, wpt_v2)
            part_id = remui(si, wpt_v2)
            h_off   = part_id * hpw_v2

            # 提前加载 tok_map（向量化 2×dwordx4 = 8 个 i32，匹配 mori 的模式）
            # tok_map 编码 = dest_pe * max_recv + dest_tok_id
            _tm_base_off = zext_i32_to_i64(tok_id * experts_per_token) * 4
            _tm_addr0    = addr_tok_map + _tm_base_off
            _tm_vec0     = load_v4i32_global(_tm_addr0)               # entries [0..3]
            _tm_vec1     = load_v4i32_global(_tm_addr0 + const_i64(16))  # entries [4..7]

            _expert_base = []
            _expert_vld  = []
            for j_py in range_constexpr(experts_per_token):
                # 从向量化结果中提取单个 i32
                if j_py < 4:
                    enc_j = _llvm_d.ExtractElementOp(
                        _tm_vec0, const_i32(j_py)).res
                else:
                    enc_j = _llvm_d.ExtractElementOp(
                        _tm_vec1, const_i32(j_py - 4)).res
                if _log2_max_recv is not None:
                    dest_pe_j = enc_j >> const_i32(_log2_max_recv)
                else:
                    dest_pe_j = divui(enc_j, max_recv)
                vld_j     = arith.cmpi(arith.CmpIPredicate.ult, dest_pe_j, const_i32(npes))
                safe_pe   = arith.select(vld_j, dest_pe_j, const_i32(rank))
                # 本地读：shmem_comb_inp + (srcPe * max_tok + tok_id) * nbytes
                _tok_off  = zext_i32_to_i64(safe_pe * max_tok_per_rank + tok_id) * nbytes
                _ebase    = _lv_unwrap(addr_comb_inp + _tok_off)
                _expert_base.append(_ebase)
                _expert_vld.append(vld_j)

            # WarpAccum — Fix A+B+D + Fix F(消除glob_ec<n_elems guard) +
            #             Fix G(预计算slot B基地址) + Fix H(分离双槽主循环和单槽尾部)
            _nt_attr  = arith.BoolAttr.get(True)
            _all_vld  = (npes >= experts_per_token)   # Fix B: 编译期判断，EP=8 k=8 恒True

            _use_wide = arith.cmpi(arith.CmpIPredicate.ult, const_i32(895), hpw_v2)   # Fix D: hpw_v2 > 895
            _if_wide  = _scf_d.IfOp(_lv_unwrap(_use_wide), [], has_else=True)

            # ── step=128 路径（hpw_v2 > 895）────────────────────────────────────────
            with _IP(_if_wide.then_block):
                # Fix F: 安全上界 = min(hpw_v2, n_elems - h_off)
                _n_rem_128   = const_i32(n_elems) - h_off
                _adj_end_128 = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, _n_rem_128, hpw_v2), _n_rem_128, hpw_v2)
                # Fix G+I: 预计算 slot B/C/D 基地址（+256/+512/+768 bytes）
                _const_256 = const_i64(256)
                _const_512 = const_i64(512)
                _const_768 = const_i64(768)
                _expert_base_b, _expert_base_c, _expert_base_d = [], [], []
                for _j_py in range_constexpr(experts_per_token):
                    _expert_base_b.append(add_i64(_expert_base[_j_py], _const_256))
                    _expert_base_c.append(add_i64(_expert_base[_j_py], _const_512))
                    _expert_base_d.append(add_i64(_expert_base[_j_py], _const_768))

                if n_i32 % 256 == 0 and warp_num_per_block < 16:
                    # Fix I+K: 4-slot step=256，仅在 wpb<16 时使用
                    # SIMD VMEM 队列分析：
                    #   wpb=8:  2波/SIMD × 32 loads = 64/64 ★最优
                    #   wpb=16: 4波/SIMD × 32 loads = 128/64 !! 溢出→stall
                    # → wpb≥16 时走 2-slot，4波×16 loads=64/64 且有更多 wavefront 隐藏 ALU 延迟
                    # Bug fix: hpw_v2 = ceil(n_i32/wpt_v2)，wpt_v2>1 时 hpw_v2 可能不整除256
                    # （例如 BS=512 wpb=16 时 wpt_v2=3, hpw_v2=1195, 1195%256=171≠0）
                    # 需要运行期检查，不对齐时回退 2-slot
                    _hpw_rem256  = remui(hpw_v2, const_i32(256))
                    _is_256aln   = arith.cmpi(arith.CmpIPredicate.ult, _hpw_rem256, const_i32(1))  # hpw_v2%256==0
                    _if_256aln   = _scf_d.IfOp(_lv_unwrap(_is_256aln), [], has_else=True)
                    with _IP(_if_256aln.then_block):
                      _quad_end_128 = _adj_end_128 - const_i32(192)
                      for ec in range(as_index(lane), as_index(_quad_end_128), as_index(256)):
                        ec       = idx_to_i32(ec)
                        glob_ec  = h_off + ec
                        elem_byte_off = zext_i32_to_i64(glob_ec) * 4
                        # 先发射全部 32 个 NT load（最大化 XGMI 并发）
                        _va, _vb, _vc, _vd = [], [], [], []
                        for _j_py in range_constexpr(experts_per_token):
                            _bj  = _expert_base[_j_py]
                            _aa  = add_i64(_bj, _lv_unwrap(elem_byte_off))
                            _va.append(_llvm_d.LoadOp(
                                _i32t, _llvm_d.IntToPtrOp(_ptr_g, _aa).result,
                                alignment=4, nontemporal=_nt_attr).result)
                            _ab  = add_i64(_expert_base_b[_j_py], _lv_unwrap(elem_byte_off))
                            _vb.append(_llvm_d.LoadOp(
                                _i32t, _llvm_d.IntToPtrOp(_ptr_g, _ab).result,
                                alignment=4, nontemporal=_nt_attr).result)
                            _ac  = add_i64(_expert_base_c[_j_py], _lv_unwrap(elem_byte_off))
                            _vc.append(_llvm_d.LoadOp(
                                _i32t, _llvm_d.IntToPtrOp(_ptr_g, _ac).result,
                                alignment=4, nontemporal=_nt_attr).result)
                            _ad  = add_i64(_expert_base_d[_j_py], _lv_unwrap(elem_byte_off))
                            _vd.append(_llvm_d.LoadOp(
                                _i32t, _llvm_d.IntToPtrOp(_ptr_g, _ad).result,
                                alignment=4, nontemporal=_nt_attr).result)
                        # 累加 4 个 slot（Fix J: 消除 ZeroOp，用首个 expert 值初始化）
                        if _all_vld:
                            # EP=8 k=8: 全部有效，直接用 va[0]/vb[0]/vc[0]/vd[0] 初始化
                            _acca = bitcast_i32_to_v2bf16(_va[0]).extf(_v2f32)
                            _accb = bitcast_i32_to_v2bf16(_vb[0]).extf(_v2f32)
                            _accc = bitcast_i32_to_v2bf16(_vc[0]).extf(_v2f32)
                            _accd = bitcast_i32_to_v2bf16(_vd[0]).extf(_v2f32)
                            for _j_py in range_constexpr(1, experts_per_token):
                                _acca = _acca + bitcast_i32_to_v2bf16(_va[_j_py]).extf(_v2f32)
                                _accb = _accb + bitcast_i32_to_v2bf16(_vb[_j_py]).extf(_v2f32)
                                _accc = _accc + bitcast_i32_to_v2bf16(_vc[_j_py]).extf(_v2f32)
                                _accd = _accd + bitcast_i32_to_v2bf16(_vd[_j_py]).extf(_v2f32)
                        else:
                            _acca = arith.constant_vector(0.0, _v2f32)
                            _accb = arith.constant_vector(0.0, _v2f32)
                            _accc = arith.constant_vector(0.0, _v2f32)
                            _accd = arith.constant_vector(0.0, _v2f32)
                            for _j_py in range_constexpr(experts_per_token):
                                _fa  = bitcast_i32_to_v2bf16(_va[_j_py]).extf(_v2f32)
                                _fb  = bitcast_i32_to_v2bf16(_vb[_j_py]).extf(_v2f32)
                                _fc  = bitcast_i32_to_v2bf16(_vc[_j_py]).extf(_v2f32)
                                _fd  = bitcast_i32_to_v2bf16(_vd[_j_py]).extf(_v2f32)
                                _vld = _lv_unwrap(_expert_vld[_j_py])
                                _z   = arith.constant_vector(0.0, _v2f32)
                                _acca = _acca + arith.select(_vld, _fa, _z)
                                _accb = _accb + arith.select(_vld, _fb, _z)
                                _accc = _accc + arith.select(_vld, _fc, _z)
                                _accd = _accd + arith.select(_vld, _fd, _z)
                        _i32a = bitcast_v2bf16_to_i32(_acca.truncf(_v2bf16))
                        _i32b = bitcast_v2bf16_to_i32(_accb.truncf(_v2bf16))
                        _i32c = bitcast_v2bf16_to_i32(_accc.truncf(_v2bf16))
                        _i32d = bitcast_v2bf16_to_i32(_accd.truncf(_v2bf16))
                        # Fix J: 预计算输出基地址，4 次 store 共用 base + 常量偏移
                        _out_off = zext_i32_to_i64(tok_id * n_i32 + glob_ec) * 4
                        _out_base_i64 = _lv_unwrap(addr_comb_out + _out_off)
                        _llvm_d.StoreOp(_i32a, _llvm_d.IntToPtrOp(
                            _ptr_g, _out_base_i64).result,
                            alignment=4, nontemporal=_nt_attr)
                        _llvm_d.StoreOp(_i32b, _llvm_d.IntToPtrOp(
                            _ptr_g, add_i64(_out_base_i64, _const_256)).result,
                            alignment=4, nontemporal=_nt_attr)
                        _llvm_d.StoreOp(_i32c, _llvm_d.IntToPtrOp(
                            _ptr_g, add_i64(_out_base_i64, _const_512)).result,
                            alignment=4, nontemporal=_nt_attr)
                        _llvm_d.StoreOp(_i32d, _llvm_d.IntToPtrOp(
                            _ptr_g, add_i64(_out_base_i64, _const_768)).result,
                            alignment=4, nontemporal=_nt_attr)
                      _scf_d.YieldOp([])
                    with _IP(_if_256aln.else_block):
                      # 2-slot fallback: hpw_v2 不整除 256（如 wpt_v2=3 时 hpw_v2=1195）
                      _dual_end_256fb = _adj_end_128 - const_i32(64)
                      for ec in range(as_index(lane), as_index(_dual_end_256fb), as_index(128)):
                        ec       = idx_to_i32(ec)
                        glob_ec  = h_off + ec
                        elem_byte_off = zext_i32_to_i64(glob_ec) * 4
                        _va, _vb = [], []
                        for _j_py in range_constexpr(experts_per_token):
                            _bj  = _expert_base[_j_py]
                            _aa  = add_i64(_bj, _lv_unwrap(elem_byte_off))
                            _va.append(_llvm_d.LoadOp(
                                _i32t, _llvm_d.IntToPtrOp(_ptr_g, _aa).result,
                                alignment=4, nontemporal=_nt_attr).result)
                            _ab  = add_i64(_expert_base_b[_j_py], _lv_unwrap(elem_byte_off))
                            _vb.append(_llvm_d.LoadOp(
                                _i32t, _llvm_d.IntToPtrOp(_ptr_g, _ab).result,
                                alignment=4, nontemporal=_nt_attr).result)
                        if _all_vld:
                            _acca = bitcast_i32_to_v2bf16(_va[0]).extf(_v2f32)
                            _accb = bitcast_i32_to_v2bf16(_vb[0]).extf(_v2f32)
                            for _j_py in range_constexpr(1, experts_per_token):
                                _acca = _acca + bitcast_i32_to_v2bf16(_va[_j_py]).extf(_v2f32)
                                _accb = _accb + bitcast_i32_to_v2bf16(_vb[_j_py]).extf(_v2f32)
                        else:
                            _acca = arith.constant_vector(0.0, _v2f32)
                            _accb = arith.constant_vector(0.0, _v2f32)
                            for _j_py in range_constexpr(experts_per_token):
                                _fa  = bitcast_i32_to_v2bf16(_va[_j_py]).extf(_v2f32)
                                _fb  = bitcast_i32_to_v2bf16(_vb[_j_py]).extf(_v2f32)
                                _vld = _lv_unwrap(_expert_vld[_j_py])
                                _z   = arith.constant_vector(0.0, _v2f32)
                                _acca = _acca + arith.select(_vld, _fa, _z)
                                _accb = _accb + arith.select(_vld, _fb, _z)
                        _i32a = bitcast_v2bf16_to_i32(_acca.truncf(_v2bf16))
                        _i32b = bitcast_v2bf16_to_i32(_accb.truncf(_v2bf16))
                        _out_off = zext_i32_to_i64(tok_id * n_i32 + glob_ec) * 4
                        _out_base_i64 = _lv_unwrap(addr_comb_out + _out_off)
                        _llvm_d.StoreOp(_i32a, _llvm_d.IntToPtrOp(
                            _ptr_g, _out_base_i64).result,
                            alignment=4, nontemporal=_nt_attr)
                        _llvm_d.StoreOp(_i32b, _llvm_d.IntToPtrOp(
                            _ptr_g, add_i64(_out_base_i64, _const_256)).result,
                            alignment=4, nontemporal=_nt_attr)
                      _scf_d.YieldOp([])
                _scf_d.YieldOp([])

            # ── step=64 路径（hpw_v2 <= 895）────────────────────────────────────────
            with _IP(_if_wide.else_block):
                # Fix F: 安全上界，消除 if glob_ec < n_elems guard
                _n_rem_64   = const_i32(n_elems) - h_off
                _adj_end_64 = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, _n_rem_64, hpw_v2), _n_rem_64, hpw_v2)
                for ec in range(as_index(lane), as_index(_adj_end_64), as_index(64)):
                    ec      = idx_to_i32(ec)
                    glob_ec = h_off + ec
                    elem_byte_off = zext_i32_to_i64(glob_ec) * 4
                    _vals = []
                    for _j_py in range_constexpr(experts_per_token):
                        _aa = add_i64(_expert_base[_j_py], _lv_unwrap(elem_byte_off))
                        _vals.append(_llvm_d.LoadOp(  # Fix A: NT load
                            _i32t, _llvm_d.IntToPtrOp(_ptr_g, _aa).result,
                            alignment=4, nontemporal=_nt_attr).result)
                    acc = arith.constant_vector(0.0, _v2f32)
                    for _j_py in range_constexpr(experts_per_token):
                        _fa  = bitcast_i32_to_v2bf16(_vals[_j_py]).extf(_v2f32)
                        if _all_vld:  # Fix B
                            acc = acc + _fa
                        else:
                            _z   = arith.constant_vector(0.0, _v2f32)
                            acc  = acc + arith.select(
                                    _lv_unwrap(_expert_vld[_j_py]), _fa, _z)
                    _i32v = bitcast_v2bf16_to_i32(acc.truncf(_v2bf16))
                    _ob   = zext_i32_to_i64(tok_id * n_i32 + glob_ec) * 4
                    _op   = _llvm_d.IntToPtrOp(
                        _ptr_g, _lv_unwrap(addr_comb_out + _ob)).result
                    _llvm_d.StoreOp(_i32v, _op, alignment=4, nontemporal=_nt_attr)
                _scf_d.YieldOp([])

    ep_combine_intranode._allocator = allocator
    return ep_combine_intranode


# ============================================================
# @flyc.jit launcher factories（公开 API）
# ============================================================

def make_dispatch_jit(*, rank, npes, experts_per_rank, experts_per_token,
                      hidden_dim, max_tok_per_rank, block_num,
                      warp_num_per_block, data_type):
    """创建 dispatch kernel 的 @flyc.jit launcher。"""
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_dispatch_kernel(
        rank=rank, npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )

    _rank_id   = rank             # expose rank as simple-type closure var → enters cache key
    _npes_id   = npes             # expose npes as simple-type closure var → enters cache key
    _block_num = block_num        # must include in cache key (baked into wait_until_equals)
    _wpb       = warp_num_per_block  # must include in cache key (baked into block_dim)
    _max_tok   = max_tok_per_rank # must include in cache key (baked into P2P offset formula)

    @flyc.jit
    def dispatch_launch(
        addr_inp_tok: fx.Int64, addr_idx: fx.Int64, addr_wts: fx.Int64,
        addr_out_tok: fx.Int64, addr_out_wts: fx.Int64, addr_out_idx: fx.Int64,
        addr_tok_off: fx.Int64, addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64, addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64, addr_tis: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_p2p_tok_off: fx.Int64, addr_p2p_tis: fx.Int64,
        addr_p2p_out_wts: fx.Int64, addr_p2p_out_idx: fx.Int64,
        addr_p2p_out_tok: fx.Int64, addr_p2p_recv_num: fx.Int64,
        cur_tok: fx.Int32,
        stream: Stream = Stream(None),
    ):
        _ = (_rank_id, _npes_id, _block_num, _wpb, _max_tok)  # referenced to include in cache key
        kernel(addr_inp_tok, addr_idx, addr_wts,
               addr_out_tok, addr_out_wts, addr_out_idx,
               addr_tok_off, addr_recv_num, addr_dest_ctr,
               addr_disp_bar, addr_tok_map, addr_tis,
               addr_total_rv,
               addr_p2p_tok_off, addr_p2p_tis,
               addr_p2p_out_wts, addr_p2p_out_idx,
               addr_p2p_out_tok, addr_p2p_recv_num,
               cur_tok).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return dispatch_launch


def make_combine_jit(*, rank, npes, experts_per_token, hidden_dim,
                     max_tok_per_rank, block_num, warp_num_per_block,
                     data_type):
    """创建 combine kernel 的 @flyc.jit launcher。"""
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    kernel = make_combine_kernel(
        rank=rank, npes=npes,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )

    _rank_id   = rank             # expose rank as simple-type closure var → enters cache key
    _npes_id   = npes             # expose npes as simple-type closure var → enters cache key
    _block_num = block_num        # must include in cache key (baked into wait_until_equals)
    _wpb       = warp_num_per_block  # must include in cache key (baked into block_dim)
    _max_tok   = max_tok_per_rank # must include in cache key (baked into P2P offset formula)

    # 获取 kernel 内部闭包引用的 allocator（用于 finalize）
    _allocator = kernel._allocator

    @flyc.jit
    def combine_launch(
        addr_inp_tok: fx.Int64, addr_comb_inp: fx.Int64,
        addr_comb_out: fx.Int64, addr_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64, addr_tok_map: fx.Int64,
        addr_comb_bar: fx.Int64, addr_trecv: fx.Int64,
        addr_tis: fx.Int64,
        addr_p2p_comb_inp: fx.Int64, addr_p2p_xdb_mem: fx.Int64,
        stream: Stream = Stream(None),
    ):
        _ = (_rank_id, _npes_id, _block_num, _wpb, _max_tok)  # referenced to include in cache key

        # 在 gpu_module_body 中插入 LDS global 声明（对齐 rmsnorm_kernel 模式）
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        _allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            _allocator.finalize()

        kernel(addr_inp_tok, addr_comb_inp, addr_comb_out,
               addr_xdb_mem, addr_xdb_flag, addr_tok_map,
               addr_comb_bar, addr_trecv, addr_tis,
               addr_p2p_comb_inp, addr_p2p_xdb_mem).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return combine_launch
