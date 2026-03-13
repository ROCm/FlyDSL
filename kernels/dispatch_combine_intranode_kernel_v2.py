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
    load_i32_global,
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
    atomic_add_i32_at,
    atomic_fetch_add_i32_global,
    divui,
    remui,
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
        cur_tok:       fx.Int32,  # 动态：本轮实际 token 数
    ):
        tid    = fx.thread_idx.x
        bid    = fx.block_idx.x
        lane   = tid & 63                           # lane ID within warp，范围 [0, 63]
        warp   = tid >> 6                           # warp ID within block
        gw_id  = bid * warp_num_per_block + warp    # global warp ID（跨所有 block）
        gw_num = block_num * warp_num_per_block     # global warp 总数（grid-stride 步长）
        limit  = cur_tok * experts_per_token        # 外层循环总迭代数

        # 预计算各 PE 的 XGMI 基地址（编译期展开，避免循环内重复调用 ptr_p2p）
        rem_tok     = [mori_shmem.ptr_p2p(addr_out_tok, rank, pe) for pe in range(npes)]
        rem_wts     = [mori_shmem.ptr_p2p(addr_out_wts, rank, pe) for pe in range(npes)]
        rem_idx     = [mori_shmem.ptr_p2p(addr_out_idx, rank, pe) for pe in range(npes)]
        rem_tis     = [mori_shmem.ptr_p2p(addr_tis,     rank, pe) for pe in range(npes)]
        rem_tok_off = [mori_shmem.ptr_p2p(addr_tok_off, rank, pe) for pe in range(npes)]

        # ── Phase 1: 发送 token ───────────────────────────────────────────────
        for i in range(as_index(gw_id), as_index(limit), as_index(gw_num)):
            i = idx_to_i32(i)
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

            # 原子分配 destTokId：
            #   - lane0 + non-dup：XGMI 硬件原子 fetch+add（单条 global_atomic_add_ret 指令）
            #   - lane0 + dup：跳过 AMO，返回 0
            #   - lanes 1-63：不执行原子，readlane 广播 lane0 的结果
            from flydsl._mlir.dialects import scf as _scf_d
            from flydsl._mlir.ir import InsertionPoint as _IP, IntegerType as _IT_mlir
            _i32_ty = _IT_mlir.get_signless(32)
            _if_lane0 = _scf_d.IfOp(_lv_unwrap(icmp_eq_i32(lane, const_i32(0))),
                                     [_i32_ty], has_else=True)
            with _IP(_if_lane0.then_block):
                _if_nodup = _scf_d.IfOp(
                    _lv_unwrap(_icmp_eq_i64(dup_ballot, const_i64(0))),
                    [_i32_ty], has_else=True)
                with _IP(_if_nodup.then_block):
                    _old_tok = atomic_fetch_add_i32_global(
                        _sel_pe(rem_tok_off, dest_pe), const_i32(1))
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
            dtm_val = select_i32(is_dup, const_i32(sentinel_val),
                                  dest_pe * max_recv + dest_tok_all)
            if icmp_eq_i32(lane, const_i32(0)):
                store_i32_at(addr_tok_map, i, dtm_val)

            # 仅非重复时写入 tok_id_to_src 和更新 dest_pe_ctr（lane0）
            # int32_p 使用 LOCAL symmetric 地址（非 ptr_p2p 地址）
            if icmp_eq_i32(lane, const_i32(0)):
                if _icmp_eq_i64(dup_ballot, const_i64(0)):
                    src_enc  = rank * max_tok_per_rank + src_tok
                    tis_local = addr_tis + zext_i32_to_i64(dest_tok_all) * 4
                    mori_shmem.int32_p(tis_local, src_enc, dest_pe, 0)
                    ctr_addr = addr_dest_ctr + zext_i32_to_i64(dest_pe) * 4
                    atomic_add_i32_at(ctr_addr, const_i32(1))

            # 写入权重 + 索引（lanes 0..k-1，仅非重复路径）
            # int32_p/float_p 使用 LOCAL symmetric 地址（非 ptr_p2p 地址）
            if icmp_ult_i32(lane, const_i32(experts_per_token)):
                if _icmp_eq_i64(dup_ballot, const_i64(0)):
                    wt_src   = src_tok * experts_per_token + lane
                    wt_val   = load_f32_at(addr_wts, wt_src)
                    ix_val   = load_i32_at(addr_idx, wt_src)
                    dst_slot = dest_tok_all * experts_per_token + lane
                    off      = zext_i32_to_i64(dst_slot) * 4
                    mori_shmem.float_p(addr_out_wts + off, wt_val, dest_pe, 0)
                    mori_shmem.int32_p(addr_out_idx + off, ix_val, dest_pe, 0)

            # 写入 token embedding（inp_tok 不在 shmem heap，用 XGMI 直接写）
            tok_remote = _sel_pe(rem_tok, dest_pe) + zext_i32_to_i64(
                dest_tok_all * hidden_dim * hidden_elem_size)
            inp_src_b  = addr_inp_tok + zext_i32_to_i64(
                src_tok * hidden_dim * hidden_elem_size)
            lane4 = lane * 4
            for ec4 in range(as_index(lane4), as_index(n_i32), 256):
                ec4     = idx_to_i32(ec4)
                ec4_byt = zext_i32_to_i64(ec4) * 4
                vec4    = load_v4i32(inp_src_b + ec4_byt)
                if _icmp_eq_i64(dup_ballot, const_i64(0)):
                    store_v4i32_global(vec4, tok_remote + ec4_byt)

        # ── Phase 2: 栅栏 + 发送 token 数量信号 ──────────────────────────────
        # 用 GPU 原子累加 dispatch_bar，确保所有 block 的 Phase 1 都完成后再发信号
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
                rtn_local  = addr_recv_num + rtn_local_off
                rtn_remote = mori_shmem.ptr_p2p(addr_recv_num, rank, dest_pe) + rtn_local_off
                mori_shmem.int32_wait_until_equals(rtn_remote, 0)
                fence_one_as_seq_cst()
                # uint32_p → P2P AtomicStoreRelaxedSystem（单条 XGMI store）
                # 等到 rtn_remote==0 后直接 store nsig，无需 RMW
                mori_shmem.uint32_p(rtn_local, nsig, dest_pe, 0)

        # ── Phase 3: 接收信号，累计 total_recv ───────────────────────────────
        for src_pe in range(as_index(lane), as_index(npes), 64):
            src_pe = idx_to_i32(src_pe)
            if icmp_eq_i32(gw_id, const_i32(0)):
                rtn_src  = addr_recv_num + zext_i32_to_i64(src_pe) * 4
                sig_val  = mori_shmem.int32_wait_until_greater_than(rtn_src, 0)
                recv_cnt = sig_val - 1
                # 直接写 0 重置信号（·sig_val 已知，无并发写入者）
                mori_shmem.uint32_p(rtn_src, const_i32(0), rank, 0)
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

    Stage 1: inp_tok → shmem_comb_inp
             向量化 warp-stride 直接 load/store（WarpCopy 风格），消除 per-token quiet fence。
    Stage 2: CrossDeviceBarrier（所有 PE 就绪后互通知）
    Stage 3: P2P 读 + 加权累计 → shmem_comb_out
             关键优化：tok_map 查找 + _sel_pe 地址计算提升到 ec4 循环外，
             每个 token 只计算一次（原来每个 element 都重复计算）。
    """
    max_recv   = npes * max_tok_per_rank
    n_i32      = hidden_dim >> 1
    nbytes     = hidden_dim * hidden_elem_size   # bytes per token (Python int)
    # n_i32 * 4 = nbytes（bf16: hidden_elem_size=2），用于 Stage 3 地址计算
    tok_stride = n_i32 * 4                       # bytes per token in i32-addressed buffer

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

        # ── Stage 1: inp_tok → shmem_comb_inp ────────────────────────────────
        # NBI put 批量发送所有 token，循环结束后统一 quiet（单次 quiet 替代 per-token quiet）。
        # putmem_nbi_warp 使用 XGMI RDMA 路径，quiet_thread_pe 确保数据对远端 GPU 可见，
        # 这是保证 Stage 3 P2P read 正确性的必要条件，不能用普通 store_v4i32_global 替代。
        for ci in range(as_index(gw_id), as_index(total_recv_val), as_index(gw_num)):
            ci     = idx_to_i32(ci)
            ci_off = zext_i32_to_i64(ci) * nbytes
            mori_shmem.putmem_nbi_warp(
                addr_comb_inp + ci_off, addr_inp_tok + ci_off,
                const_i64(nbytes), rank, 0)
        # 所有 token 的 NBI put 批量提交后，统一 quiet（原来每 token 一次 quiet，现在只一次）
        mori_shmem.quiet_thread_pe(rank)

        # ── Stage 2: CrossDeviceBarrier ───────────────────────────────────────
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

        # ── Stage 3: P2P 读 + 累计 ───────────────────────────────────────────
        rem_sci = [mori_shmem.ptr_p2p(addr_comb_inp, rank, pe) for pe in range(npes)]

        # 每个 token 分配 wpt 个 warp，每个 warp 负责 hpw 个 i32
        wpt    = (gw_num + cur_tok - 1) // cur_tok
        hpw    = (n_i32 + wpt - 1) // wpt
        s3_lim = cur_tok * wpt

        from flydsl._mlir.dialects import llvm as _llvm_d, arith as _arith_d
        from flydsl._mlir.ir import (VectorType, BF16Type, F32Type,
                                     IntegerType as _IT, IntegerAttr as _IA,
                                     FloatAttr as _FA, IntegerType as _IT2)
        _v2bf16 = VectorType.get([2], BF16Type.get())
        _v2f32  = VectorType.get([2], F32Type.get())
        _i32t   = _IT.get_signless(32)
        _i1t    = _IT2.get_signless(1)
        _f32t   = F32Type.get()

        for si in range(as_index(gw_id), as_index(s3_lim), as_index(gw_num)):
            si      = idx_to_i32(si)
            tok_id  = divui(si, wpt)
            part_id = remui(si, wpt)
            h_i32   = part_id * hpw

            # ── 关键优化：将 tok_map 查找、div/rem、_sel_pe 地址计算提升到 ec4 外 ──
            # 这些值只依赖 tok_id 和 j，与 ec4（element 维度）无关。
            # 原实现：每次 ec4 迭代（共 hpw/64 次）都重复执行一遍 k 个 expert 的全部计算。
            # 新实现：在 for si 体内、for ec4 前执行一次，存入 Python list 供内层引用。
            # MLIR SSA 合法：for si 体 dominates for ec4 体，外层 Value 可被内层直接引用。
            src_base_j = []   # rem_sci[dest_pe_j] + local_tok_j * tok_stride（i64 地址）
            valid_j    = []   # icmp_ult(dest_pe_j, npes)（i1-equivalent i32）
            for j_py in range_constexpr(experts_per_token):
                enc_j       = load_i32_at(addr_tok_map,
                                tok_id * experts_per_token + j_py)
                dest_pe_j   = divui(enc_j, max_recv)
                local_tok_j = remui(enc_j, max_recv)
                valid_pe_j  = icmp_ult_i32(dest_pe_j, const_i32(npes))
                # 远端 token 基地址：rem_sci[dest_pe_j][local_tok_j * n_i32]（字节偏移）
                tok_base    = (_sel_pe(rem_sci, dest_pe_j)
                               + zext_i32_to_i64(local_tok_j) * tok_stride)
                src_base_j.append(tok_base)
                valid_j.append(valid_pe_j)

            # 每个 lane 处理 1 个 i32（= 2 bf16），步长 64
            for ec4 in range(as_index(lane), as_index(hpw), 64):
                ec4        = idx_to_i32(ec4)
                global_ec4 = h_i32 + ec4
                in_bounds  = icmp_ult_i32(global_ec4, const_i32(n_i32))
                out_base   = zext_i32_to_i64(tok_id * n_i32 + global_ec4) * 4

                # ec4 字节偏移（加到各 expert 的 tok_base 上得到元素地址）
                ec4_byt = zext_i32_to_i64(global_ec4) * 4
                acc = _llvm_d.ZeroOp(_v2f32).res

                # j-loop Python-level unroll（range_constexpr）：
                # 内层 j 循环仅执行 P2P read + bf16→f32 转换 + 累加，
                # tok_map 查找和 _sel_pe 已在外层完成。
                for j_py in range_constexpr(experts_per_token):
                    src_addr  = src_base_j[j_py] + ec4_byt
                    raw_i32   = load_i32_global(src_addr)
                    # i32 → <2xbf16> → <2xf32>
                    as_bf16   = _llvm_d.BitcastOp(_v2bf16, raw_i32).res
                    as_v2f32  = _arith_d.ExtFOp(_v2f32, as_bf16).result
                    # gate：in_bounds AND valid_pe
                    gate_i32  = select_i32(in_bounds,
                                    select_i32(valid_j[j_py], const_i32(1), const_i32(0)),
                                    const_i32(0))
                    gate_i1b  = _arith_d.TruncIOp(_i1t, gate_i32).result
                    one_f32   = _arith_d.ConstantOp(_f32t, _FA.get(_f32t, 1.0)).result
                    zero_f32  = _arith_d.ConstantOp(_f32t, _FA.get(_f32t, 0.0)).result
                    gate_f32  = _arith_d.SelectOp(gate_i1b, one_f32, zero_f32).result
                    # gate_f32 broadcast to <2xf32>
                    c0 = _llvm_d.ConstantOp(_i32t, _IA.get(_i32t, 0)).result
                    c1 = _llvm_d.ConstantOp(_i32t, _IA.get(_i32t, 1)).result
                    gv = _llvm_d.ZeroOp(_v2f32).res
                    gv = _llvm_d.InsertElementOp(gv, gate_f32, c0).res
                    gv = _llvm_d.InsertElementOp(gv, gate_f32, c1).res
                    acc = _arith_d.AddFOp(
                        acc, _arith_d.MulFOp(as_v2f32, gv).result).result

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
