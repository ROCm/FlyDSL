"""
FlyDSL dispatch/combine intranode kernels — v2 (Python FlyDSL syntax).

所有 buffer 基地址以 fx.Int64 传入（避免 fly.memref → LLVM pointer 时序问题）。
算子逻辑与 EpDispatchIntraNodeKernel / EpCombineIntraNodeKernel 一致。

FlyDSL 编码规则（来自实际调试经验）
-------------------------------------
1. 编译期常量 → 闭包变量，不在 kernel 参数列表中
2. 动态 if 条件必须是函数调用形式（icmp_eq_i32 等）
3. for 循环必须在 kernel 顶层（不能嵌套在 scf.if 闭包内）
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
from flydsl.compiler.shmem_compile import compile_shmem_kernel, ShmemKernel

import mori.ir.flydsl as mori_shmem

from flydsl.expr.lowlevel import (
    ballot_i64,
    readlane,
    fence_one_as_seq_cst,
    load_v4i32,
    store_v4i32_global,
    sync_threads,
    load_i32_at,
    load_f32_at,
    load_i32_global,   # P2P read from XGMI-mapped (addrspace 1) address
    load_i64_at,
    store_i32_at,
    zext_i32_to_i64,
    const_i32,
    const_i64,
    select_i32,
    select_i64,
    icmp_eq_i32,
    icmp_ult_i32,
    as_index,
    idx_to_i32,
    atomic_add_i32_at,             # GPU local atomic add (not shmem)
    atomic_fetch_add_i32_global,   # XGMI native fetch-add, single instruction
    divui,              # unsigned integer divide (faster than Python //)
    remui,              # unsigned integer remainder (faster than Python %)
)
from flydsl.expr.lowlevel import _unwrap as _lv_unwrap


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _icmp_ne_i64(a, b):
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.lowlevel import _unwrap
    return llvm.ICmpOp(llvm.ICmpPredicate.ne, _unwrap(a), _unwrap(b)).res


def _icmp_eq_i64(a, b):
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.lowlevel import _unwrap
    return llvm.ICmpOp(llvm.ICmpPredicate.eq, _unwrap(a), _unwrap(b)).res


def _sel_pe(rem_list, dest_pe):
    """用 select_i64 链实现运行时动态索引（不能用 Python list[ArithValue]）。"""
    result = rem_list[-1]
    for pe in reversed(range(len(rem_list) - 1)):
        result = select_i64(icmp_eq_i32(dest_pe, const_i32(pe)), rem_list[pe], result)
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
        # 输入 buffer（i64 基地址）
        addr_inp_tok:  fx.Int64,  # [cur_tok, hidden_dim]  bf16
        addr_idx:      fx.Int64,  # [cur_tok, k]           i32  (token_indices)
        addr_wts:      fx.Int64,  # [cur_tok, k]           f32  (weights_buf)
        # Shmem buffer（symmetric，i64 基地址）
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
        cur_tok:       fx.Int32,  # 动态：本轮实际 token 数
    ):
        # Thread / warp 坐标
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63
        warp   = tid >> 6
        gw_id  = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        limit  = cur_tok * experts_per_token

        # 预计算各 PE 的 P2P 基地址（常量展开，避免循环内重复调用 ptr_p2p）
        rem_tok = [mori_shmem.ptr_p2p(addr_out_tok, rank, pe) for pe in range(npes)]
        rem_wts = [mori_shmem.ptr_p2p(addr_out_wts, rank, pe) for pe in range(npes)]
        rem_idx = [mori_shmem.ptr_p2p(addr_out_idx, rank, pe) for pe in range(npes)]
        rem_tis = [mori_shmem.ptr_p2p(addr_tis,     rank, pe) for pe in range(npes)]

        # 预计算 tok_off 的 XGMI 远端地址（每 PE 一个，避免循环内重复调用）
        rem_tok_off = [mori_shmem.ptr_p2p(addr_tok_off, rank, pe) for pe in range(npes)]

        # ── Phase 1: 发送 token ───────────────────────────────────────────────
        # 每个 warp 处理一组 (srcTok, expertSlot) 对（步长 gw_num）
        for i in range(as_index(gw_id), as_index(limit), as_index(gw_num)):
            i = idx_to_i32(i)
            # 使用 divui/remui（unsigned）替代 Python 的 ///%（→ sdiv/srem）
            # 这些值在运行时始终为非负，udiv 比 sdiv 在 AMD GPU 上快约 3×
            src_tok  = divui(i, experts_per_token)
            j        = remui(i, experts_per_token)
            dest_exp = load_i32_at(addr_idx, i)
            dest_pe  = divui(dest_exp, experts_per_rank)

            # 去重检测：ballot 判断当前 (srcTok, destPe) 是否被更早的 slot 发过
            safe_lane    = select_i32(icmp_ult_i32(lane, j), lane, const_i32(0))
            lane_exp     = load_i32_at(addr_idx, src_tok * experts_per_token + safe_lane)
            lane_pe      = divui(lane_exp, experts_per_rank)
            dup_per_lane = select_i32(
                icmp_eq_i32(lane_pe, dest_pe),
                select_i32(icmp_ult_i32(lane, j), lane, const_i32(64)),
                const_i32(64))
            dup_ballot   = ballot_i64(icmp_ult_i32(dup_per_lane, const_i32(64)))
            is_dup       = _icmp_ne_i64(dup_ballot, const_i64(0))

            # ── 原子分配 destTokId（仅 lane0 非重复路径执行 XGMI 原子） ───────
            # Fix 1（前次）：scf.IfOp 将 exec 收缩到 lane0 → 消除 63 个无效 CAS 调用。
            # Fix 2（本次）：lane0 block 内再嵌套 IfOp 区分 dup/non-dup：
            #   - non-dup：发 atomic_fetch_add_i32_global(+1)（单条 XGMI 硬件原子）
            #   - dup：直接返回 0，完全不发 AMO 指令（mori 用 continue 实现同等效果）
            from flydsl._mlir.dialects import scf as _scf_d
            from flydsl._mlir.ir import InsertionPoint as _IP, IntegerType as _IT_mlir
            _i32_ty = _IT_mlir.get_signless(32)
            _lane0_cond = _lv_unwrap(icmp_eq_i32(lane, const_i32(0)))
            _if_lane0 = _scf_d.IfOp(_lane0_cond, [_i32_ty], has_else=True)
            with _IP(_if_lane0.then_block):
                # lane 0：non-dup → fetch+add 1；dup → return 0（跳过 XGMI AMO）
                _nodup_cond = _lv_unwrap(_icmp_eq_i64(dup_ballot, const_i64(0)))
                _if_nodup = _scf_d.IfOp(_nodup_cond, [_i32_ty], has_else=True)
                with _IP(_if_nodup.then_block):
                    _tok_off_xgmi = _sel_pe(rem_tok_off, dest_pe)
                    _old_tok = atomic_fetch_add_i32_global(_tok_off_xgmi, const_i32(1))
                    _scf_d.YieldOp([_lv_unwrap(_old_tok)])
                with _IP(_if_nodup.else_block):
                    # dup：返回 0；dest_tok_all 在 dup 路径的所有使用均受 is_dup/dup_ballot 保护
                    _scf_d.YieldOp([_lv_unwrap(const_i32(0))])
                _scf_d.YieldOp([_if_nodup.result])
            with _IP(_if_lane0.else_block):
                # lanes 1-63：不执行原子，返回 0（readlane 只用 lane0 的值）
                _scf_d.YieldOp([_lv_unwrap(const_i32(0))])
            dest_tok     = _if_lane0.result  # single result → scalar ArithValue
            dest_tok_all = readlane(dest_tok, 0)  # 广播 lane0 的结果给整个 warp

            # 写入 dest_tok_map[i]（lane0）
            # sentinel = npes * max_recv（保证解码后 dest_pe_j >= npes，视为无效）
            sentinel_val = npes * max_recv   # npes² × max_tok_per_rank
            dtm_nodup = dest_pe * max_recv + dest_tok_all
            dtm_val   = select_i32(is_dup, const_i32(sentinel_val), dtm_nodup)
            if icmp_eq_i32(lane, const_i32(0)):
                store_i32_at(addr_tok_map, i, dtm_val)

            # 仅非重复时写入 tok_id_to_src 和更新 dest_pe_ctr（lane0）
            # 注意：int32_p 使用 LOCAL symmetric 地址（不是 ptr_p2p 地址）
            if icmp_eq_i32(lane, const_i32(0)):
                if _icmp_eq_i64(dup_ballot, const_i64(0)):
                    src_enc  = rank * max_tok_per_rank + src_tok
                    # LOCAL symmetric addr: addr_tis + dest_tok_all * 4
                    tis_local = addr_tis + zext_i32_to_i64(dest_tok_all) * 4
                    mori_shmem.int32_p(tis_local, src_enc, dest_pe, 0)
                    ctr_addr = addr_dest_ctr + zext_i32_to_i64(dest_pe) * 4
                    atomic_add_i32_at(ctr_addr, const_i32(1))

            # 写入权重 + 索引（lanes 0..k-1，仅非重复路径）
            # 使用 LOCAL symmetric 地址 + int32_p（NIC transport）
            # 关键：int32_p 第一个参数是 LOCAL symmetric 地址，非 ptr_p2p 地址
            if icmp_ult_i32(lane, const_i32(experts_per_token)):
                if _icmp_eq_i64(dup_ballot, const_i64(0)):
                    wt_src   = src_tok * experts_per_token + lane
                    wt_val   = load_f32_at(addr_wts, wt_src)
                    ix_val   = load_i32_at(addr_idx, wt_src)
                    dst_slot = dest_tok_all * experts_per_token + lane
                    off      = zext_i32_to_i64(dst_slot) * 4
                    # LOCAL symmetric addresses for NIC-based writes
                    mori_shmem.float_p(addr_out_wts + off, wt_val, dest_pe, 0)
                    mori_shmem.int32_p(addr_out_idx + off, ix_val, dest_pe, 0)

            # 写入 token embedding（ptr_p2p + store_v4i32_global XGMI 直接写）
            # inp_tok 不在 shmem heap，不能用 NIC；改用 XGMI 直接写（已验证可行）
            tok_boff   = zext_i32_to_i64(dest_tok_all * hidden_dim * hidden_elem_size)
            tok_remote = _sel_pe(rem_tok, dest_pe) + tok_boff
            inp_boff   = zext_i32_to_i64(src_tok * hidden_dim * hidden_elem_size)
            inp_src_b  = addr_inp_tok + inp_boff
            lane4      = lane * 4

            for ec4 in range(as_index(lane4), as_index(n_i32), 256):
                ec4      = idx_to_i32(ec4)
                ec4_byt  = zext_i32_to_i64(ec4) * 4
                vec4     = load_v4i32(inp_src_b + ec4_byt)
                if _icmp_eq_i64(dup_ballot, const_i64(0)):
                    store_v4i32_global(vec4, tok_remote + ec4_byt)

        # ── Phase 2: 栅栏 + 发送 token 数量信号 ──────────────────────────────
        # 注意：必须用 GPU 原子操作累加 dispatch_bar，否则多块竞争导致计数错误
        sync_threads()
        if icmp_eq_i32(tid, const_i32(0)):
            atomic_add_i32_at(addr_disp_bar, const_i32(1))

        rtn_local_off = zext_i32_to_i64(const_i32(rank)) * 4
        for dest_pe in range(as_index(lane), as_index(npes), 64):
            dest_pe = idx_to_i32(dest_pe)
            if icmp_eq_i32(gw_id, const_i32(0)):
                mori_shmem.int32_wait_until_equals(addr_disp_bar, block_num)
                store_i32_at(addr_disp_bar, const_i32(0), const_i32(0))
                nsig       = load_i32_at(addr_dest_ctr, dest_pe) + 1
                # rtn_local = LOCAL symmetric addr of recv_tok_num[rank]
                rtn_local  = addr_recv_num + rtn_local_off
                # rtn_remote = ptr_p2p addr of dest_pe's recv_tok_num[rank] (for spin-read)
                rtn_remote = mori_shmem.ptr_p2p(addr_recv_num, rank, dest_pe) + rtn_local_off
                # Spin-wait on remote addr (XGMI read, ptr_p2p addr OK for wait_until)
                mori_shmem.int32_wait_until_equals(rtn_remote, 0)
                fence_one_as_seq_cst()
                # Fix B: uint32_p → P2P AtomicStoreRelaxedSystem（单条 XGMI store 指令）
                # 替代原来的 uint32_atomic_add_thread（软件 CAS 循环）。
                # 语义正确：等到 rtn_remote==0 后值已知为 0，直接 store nsig 即可，无需 RMW。
                # 对应 mori: core::AtomicStoreRelaxedSystem(signal, numTokenSignal)
                mori_shmem.uint32_p(rtn_local, nsig, dest_pe, 0)

        # ── Phase 3: 接收信号，累计 total_recv ───────────────────────────────
        for src_pe in range(as_index(lane), as_index(npes), 64):
            src_pe = idx_to_i32(src_pe)
            if icmp_eq_i32(gw_id, const_i32(0)):
                # rtn_src is LOCAL symmetric addr of recv_tok_num[src_pe]
                # int32_wait_until_greater_than and uint32_atomic_add_thread both use LOCAL sym addr
                rtn_src  = addr_recv_num + zext_i32_to_i64(src_pe) * 4
                sig_val  = mori_shmem.int32_wait_until_greater_than(rtn_src, 0)
                recv_cnt = sig_val - 1
                # Fix C: 直接写 0 重置信号，替代原子减法（CAS 循环）。
                # 语义正确：sig_val 刚读出，此时信号值已知为 sig_val，无并发写入者，
                # 直接 store 0 安全。对应 mori: core::AtomicStoreRelaxedSystem(signal, 0)。
                # 使用 uint32_p(pe=rank) 保证 system-scope ordering（同 mori AtomicStore）。
                mori_shmem.uint32_p(rtn_src, const_i32(0), rank, 0)
                # total_recv 累加（原子，多 lane 可能并发）
                atomic_add_i32_at(addr_total_rv, recv_cnt)
                store_i32_at(addr_dest_ctr, src_pe, const_i32(0))

        if icmp_eq_i32(gw_id, const_i32(0)):
            if icmp_eq_i32(lane, const_i32(0)):
                store_i32_at(addr_tok_off, const_i32(0), const_i32(0))

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
    """创建 combine intranode @flyc.kernel（UseP2PRead=True）。

    Stage 1: 将本地 expert 输出复制到 shmem_comb_inp（本 PE 对称堆，P2P 可读）
    Stage 2: CrossDeviceBarrier（所有 PE 就绪后互通知）
    Stage 3: P2P 读 + 加权累计 → shmem_comb_out
    """
    max_recv = npes * max_tok_per_rank
    n_i32    = hidden_dim >> 1

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
        cur_tok:       fx.Int32,   # 本 rank token 数
        total_recv_val:fx.Int32,   # dispatch 阶段接收到的总 token 数
    ):
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63
        warp   = tid >> 6
        gw_id  = bid * warp_num_per_block + warp
        gw_num = block_num * warp_num_per_block
        gwtid  = bid * (warp_num_per_block * 64) + tid

        cur_flag = load_i64_at(addr_xdb_flag, const_i32(0))
        nbytes   = hidden_dim * hidden_elem_size

        # Stage 1: inp_tok → shmem_comb_inp（putmem_nbi_warp 批量写）
        for ci in range(as_index(gw_id), as_index(total_recv_val), as_index(gw_num)):
            ci = idx_to_i32(ci)
            ci_off = zext_i32_to_i64(ci) * nbytes
            mori_shmem.putmem_nbi_warp(
                addr_comb_inp + ci_off, addr_inp_tok + ci_off,
                const_i64(nbytes), rank, 0)
            mori_shmem.quiet_thread_pe(rank)

        # Stage 2: CrossDeviceBarrier（原子累加，避免多块竞争）
        sync_threads()
        if icmp_eq_i32(tid, const_i32(0)):
            atomic_add_i32_at(addr_comb_bar, const_i32(1))

        if icmp_ult_i32(gwtid, const_i32(npes)):
            mori_shmem.int32_wait_until_equals(addr_comb_bar, block_num)
            store_i32_at(addr_comb_bar, const_i32(0), const_i32(0))
            fence_one_as_seq_cst()
            xdb_sym = addr_xdb_mem + zext_i32_to_i64(const_i32(rank)) * 8
            mori_shmem.uint64_p(xdb_sym, cur_flag, gwtid, 0)

        if icmp_ult_i32(tid, const_i32(npes)):
            peer_slot = addr_xdb_mem + zext_i32_to_i64(tid) * 8
            mori_shmem.uint64_wait_until_equals(peer_slot, cur_flag)

        sync_threads()
        if icmp_eq_i32(tid, const_i32(0)):
            store_i32_at(addr_trecv, const_i32(0), const_i32(0))

        # Stage 3: P2P 读 + 累计
        rem_sci = [mori_shmem.ptr_p2p(addr_comb_inp, rank, pe) for pe in range(npes)]

        # ceil(gw_num / cur_tok): 每个 token 分配的 warp 数
        wpt    = (gw_num + cur_tok - 1) // cur_tok
        hpw    = (n_i32 + wpt - 1) // wpt
        s3_lim = cur_tok * wpt

        for si in range(as_index(gw_id), as_index(s3_lim), as_index(gw_num)):
            si      = idx_to_i32(si)
            # 使用 divui/remui（unsigned，比 sdiv/srem 快）
            tok_id  = divui(si, wpt)
            part_id = remui(si, wpt)
            h_i32   = part_id * hpw

            # 每个 lane 处理 1 个 i32 (= 2 bf16) 元素，步长 = warp_size = 64
            # dispatch 写入用 lane*4 + stride=256 (每次写 <4xi32>)
            # combine 读取用 lane + stride=64 (每次处理 1 i32 累加)
            for ec4 in range(as_index(lane), as_index(hpw), 64):
                ec4        = idx_to_i32(ec4)
                global_ec4 = h_i32 + ec4
                in_bounds  = icmp_ult_i32(global_ec4, const_i32(n_i32))
                out_base   = zext_i32_to_i64(tok_id * n_i32 + global_ec4) * 4

                # j-loop UNROLLED (Python compile-time, not scf.for):
                # This avoids the SSA domination issue when carrying `acc` out of scf.for.
                # Each iteration generates MLIR ops inline; acc is an SSA chain.
                from flydsl._mlir.dialects import llvm as _llvm_d, arith as _arith_d
                from flydsl._mlir.ir import (VectorType, BF16Type, F32Type,
                                             IntegerType as _IT, IntegerAttr as _IA,
                                             FloatAttr as _FA, IntegerType as _IT2)
                _v2bf16 = VectorType.get([2], BF16Type.get())
                _v2f32  = VectorType.get([2], F32Type.get())
                _i32t   = _IT.get_signless(32)
                _i1t    = _IT2.get_signless(1)
                _f32t   = F32Type.get()
                # Accumulator: starts at zero
                acc = _llvm_d.ZeroOp(_v2f32).res

                # Use range_constexpr to force Python-level loop (no scf.ForOp transform).
                # This avoids the SSA domination issue with acc across scf.for iterations.
                for j_py in range_constexpr(experts_per_token):
                    enc_j     = load_i32_at(addr_tok_map,
                                    tok_id * experts_per_token + j_py)
                    # 使用 unsigned 除法（enc_j 值域 [0, npes*max_recv]，均为非负）
                    dest_pe_j = divui(enc_j, max_recv)
                    local_tok = remui(enc_j, max_recv)
                    valid_pe  = icmp_ult_i32(dest_pe_j, const_i32(npes))

                    elem_off  = zext_i32_to_i64(local_tok * n_i32 + global_ec4) * 4
                    src_addr  = _sel_pe(rem_sci, dest_pe_j) + elem_off
                    # P2P read via XGMI addrspace(1) — verified working
                    raw_i32   = load_i32_global(src_addr)
                    # Unpack: i32 → <2xbf16> → <2xf32>
                    as_bf16   = _llvm_d.BitcastOp(_v2bf16, raw_i32).res
                    as_v2f32  = _arith_d.ExtFOp(_v2f32, as_bf16).result
                    # Gate: only add if in_bounds AND valid_pe
                    gate_i32  = select_i32(in_bounds,
                                    select_i32(valid_pe, const_i32(1), const_i32(0)),
                                    const_i32(0))
                    gate_i1b  = _arith_d.TruncIOp(_i1t, gate_i32).result
                    one_f32   = _arith_d.ConstantOp(_f32t, _FA.get(_f32t, 1.0)).result
                    zero_f32  = _arith_d.ConstantOp(_f32t, _FA.get(_f32t, 0.0)).result
                    gate_f32  = _arith_d.SelectOp(gate_i1b, one_f32, zero_f32).result
                    # Broadcast gate_f32 to <2xf32>
                    c0 = _llvm_d.ConstantOp(_i32t, _IA.get(_i32t, 0)).result
                    c1 = _llvm_d.ConstantOp(_i32t, _IA.get(_i32t, 1)).result
                    gv = _llvm_d.ZeroOp(_v2f32).res
                    gv = _llvm_d.InsertElementOp(gv, gate_f32, c0).res
                    gv = _llvm_d.InsertElementOp(gv, gate_f32, c1).res
                    # Accumulate: acc += as_v2f32 * gv
                    acc = _arith_d.AddFOp(
                        acc, _arith_d.MulFOp(as_v2f32, gv).result).result

                # Store accumulated result → shmem_comb_out[tok_id, global_ec4]
                if in_bounds:
                    acc_bf16 = _arith_d.TruncFOp(_v2bf16, acc).result
                    acc_i32  = _llvm_d.BitcastOp(_i32t, acc_bf16).res
                    store_i32_at(addr_comb_out, tok_id * n_i32 + global_ec4, acc_i32)

    return ep_combine_intranode


# ============================================================
# Build + compile helpers（公开 API）
# ============================================================

def _find_mori_shmem_bc() -> str:
    from mori.ir.bitcode import find_bitcode
    return find_bitcode()


def build_and_compile_dispatch(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
    data_type,
    chip: str = "gfx942",
    shmem_bc: str = None,
) -> ShmemKernel:
    """编译 v2 dispatch kernel，返回 ShmemKernel callable。

    编译产物缓存在 ~/.flydsl/cache/shmem/，基于内容 hash 自动管理。
    """
    import torch
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    if shmem_bc is None:
        shmem_bc = _find_mori_shmem_bc()

    kernel_fn = make_dispatch_kernel(
        rank=rank, npes=npes,
        experts_per_rank=experts_per_rank, experts_per_token=experts_per_token,
        hidden_dim=hidden_dim, hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num, warp_num_per_block=warp_num_per_block,
    )
    dummy = [fx.Int64(0)] * 13 + [fx.Int32(1)]
    return compile_shmem_kernel(
        kernel_fn, dummy_args=dummy,
        chip=chip, shmem_bc=shmem_bc,
    )


def build_and_compile_combine(
    *,
    rank: int,
    npes: int,
    experts_per_token: int,
    hidden_dim: int,
    max_tok_per_rank: int,
    block_num: int,
    warp_num_per_block: int,
    data_type,
    chip: str = "gfx942",
    shmem_bc: str = None,
) -> ShmemKernel:
    """编译 v2 combine kernel，返回 ShmemKernel callable。

    编译产物缓存在 ~/.flydsl/cache/shmem/，基于内容 hash 自动管理。
    """
    import torch
    hidden_elem_size = torch.tensor([], dtype=data_type).element_size()
    if shmem_bc is None:
        shmem_bc = _find_mori_shmem_bc()

    kernel_fn = make_combine_kernel(
        rank=rank, npes=npes,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim, hidden_elem_size=hidden_elem_size,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num, warp_num_per_block=warp_num_per_block,
    )
    dummy = [fx.Int64(0)] * 8 + [fx.Int32(1), fx.Int32(1)]
    return compile_shmem_kernel(
        kernel_fn, dummy_args=dummy,
        chip=chip, shmem_bc=shmem_bc,
    )
