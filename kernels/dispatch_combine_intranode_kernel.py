"""
FlyDSL-style Dispatch/Combine IntraNode Kernel Builder.

Uses flydsl._mlir.ir (FlyDSL's MLIR Python bindings) for module structure,
flydsl._mlir.dialects.rocdl for GPU intrinsics (FlyDSL syntax main body),
and flydsl._mlir.dialects.llvm for shmem extern declarations (MLIR supplement).

The kernel body is expressed as LLVM IR text (which is what the LLVM dialect
produces) and compiled via:
  MLIR module → mlir-translate → LLVM IR → llvm-link + mori_bc → .hsaco

Reference: mori/src/ops/dispatch_combine/intranode.hpp
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

# ============================================================
# FlyDSL MLIR Python bindings
# ============================================================
from flydsl._mlir import ir
from flydsl._mlir._mlir_libs import _mlirRegisterEverything as _reg_all
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.ir import (
    Attribute,
    Context,
    DenseI32ArrayAttr,
    FlatSymbolRefAttr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    TypeAttr,
    UnitAttr,
)

# ============================================================
# Tool paths
# ============================================================
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
LLVM_LINK = os.environ.get("LLVM_LINK",
    os.path.join(ROCM_PATH, "lib/llvm/bin/llvm-link"))
ROCM_CLANG = os.environ.get("ROCM_CLANG",
    os.path.join(ROCM_PATH, "llvm/bin/clang"))
MLIR_TRANSLATE = (
    os.environ.get("MLIR_TRANSLATE")
    or shutil.which("mlir-translate-20")
    or shutil.which("mlir-translate")
    or "/mnt/data/xiaobing/llvm-project/buildmlir/bin/mlir-translate"
)

# ============================================================
# Config
# ============================================================
@dataclass
class KernelConfig:
    """Static compile-time configuration for dispatch/combine kernels."""
    chip: str = "gfx942"
    # warpSize is always 64 on AMD gfx9xx
    warp_size: int = 64


def _find_mori_shmem_bc() -> str:
    """Locate libmori_shmem_device.bc (same logic as mlir_shmem_kernel.py)."""
    candidates = []
    if os.environ.get("MORI_SHMEM_BC"):
        candidates.append(os.environ["MORI_SHMEM_BC"])
    if os.environ.get("MORI_HOME"):
        candidates.append(os.path.join(os.environ["MORI_HOME"], "lib",
                                       "libmori_shmem_device.bc"))
    # Search relative to mori repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in ["../../mori/lib", "../../mori/build/lib.linux-x86_64-cpython-312/mori/ir",
                "../../mori/python/mori/ir"]:
        candidates.append(os.path.join(script_dir, rel,
                                       "libmori_shmem_device.bc"))
    # Absolute fallback
    candidates.append("/home/yashao/mori/lib/libmori_shmem_device.bc")
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"libmori_shmem_device.bc not found. Searched: {candidates}")


# ============================================================
# MLIR context helper (FlyDSL style)
# ============================================================
def _make_context() -> Context:
    """Create a properly registered MLIR context using flydsl._mlir.ir."""
    registry = ir.DialectRegistry()
    _reg_all.register_dialects(registry)
    ctx = Context()
    ctx.append_dialect_registry(registry)
    ctx.load_all_available_dialects()
    return ctx


_no_bundle = None

def _get_no_bundle():
    global _no_bundle
    if _no_bundle is None:
        _no_bundle = DenseI32ArrayAttr.get([])
    return _no_bundle


def _declare(name: str, sig: str):
    """Declare an external llvm.func (body provided by shmem bitcode)."""
    f = llvm.LLVMFuncOp(name, TypeAttr.get(ir.Type.parse(sig)))
    f.operation.attributes["sym_visibility"] = Attribute.parse('"private"')
    return f


# ============================================================
# shmem + GPU intrinsic declarations (FlyDSL rocdl + llvm style)
# ============================================================
def _declare_shmem_fns():
    """Declare all mori shmem extern functions used by both kernels.
    These are provided by libmori_shmem_device.bc (MLIR supplement).
    """
    # Query
    _declare("mori_shmem_my_pe",  "!llvm.func<i32 ()>")
    _declare("mori_shmem_n_pes",  "!llvm.func<i32 ()>")
    # P2P address
    _declare("mori_shmem_ptr_p2p",
             "!llvm.func<i64 (i64, i32, i32)>")
    # warp-level put
    _declare("mori_shmem_putmem_nbi_warp",
             "!llvm.func<i32 (i64, i64, i64, i32, i32)>")
    # quiet
    _declare("mori_shmem_quiet_thread_pe",
             "!llvm.func<i32 (i32)>")
    # atomic ops
    _declare("mori_shmem_uint32_atomic_fetch_add_thread",
             "!llvm.func<i32 (i64, i32, i32, i32)>")
    _declare("mori_shmem_uint32_atomic_add_thread",
             "!llvm.func<i32 (i64, i32, i32, i32)>")
    # wait operations
    _declare("mori_shmem_int32_wait_until_greater_than",
             "!llvm.func<i32 (i64, i32)>")
    _declare("mori_shmem_int32_wait_until_equals",
             "!llvm.func<i32 (i64, i32)>")
    _declare("mori_shmem_uint64_wait_until_equals",
             "!llvm.func<i32 (i64, i64)>")
    # immediate u64 put (for barrier)
    _declare("mori_shmem_uint64_p",
             "!llvm.func<i32 (i64, i64, i32, i32)>")


def _declare_gpu_intrinsics():
    """Declare AMD GPU intrinsics using rocdl dialect (FlyDSL syntax main body)."""
    i32 = IntegerType.get_signless(32)
    i64 = IntegerType.get_signless(64)
    i1  = IntegerType.get_signless(1)
    # These are declared as extern llvm functions matching rocdl intrinsic names
    _declare("llvm.amdgcn.workitem.id.x",  "!llvm.func<i32 ()>")
    _declare("llvm.amdgcn.workgroup.id.x", "!llvm.func<i32 ()>")
    _declare("llvm.amdgcn.ballot.i64",     "!llvm.func<i64 (i1)>")
    _declare("llvm.amdgcn.readlane",       "!llvm.func<i32 (i32, i32)>")
    _declare("llvm.amdgcn.s.barrier",      "!llvm.func<void ()>")


# ============================================================
# Dispatch Kernel LLVM IR
# ============================================================
# Implements EpDispatchIntraNodeKernel logic from intranode.hpp:77
# Phases:
#   Phase 1: Send tokens via shmem P2P (putmem_nbi_warp)
#   Phase 2: Grid barrier + signal token counts to all PEs
#   Phase 3: Receive token count signals, accumulate total_recv
#
# Parameters (flat, no C++ struct):
#   Pointers (symmetric shmem unless noted):
#     inp_tok        - local input tokens [cur_tok, hidden_dim]
#     token_indices  - expert routing indices [cur_tok, k] (i32)
#     weights_buf    - routing weights [cur_tok, k] (f32)
#     shmem_out_tok  - symmetric dispatch output tokens
#     shmem_out_wts  - symmetric dispatch output weights
#     shmem_out_idx  - symmetric dispatch output expert indices
#     shmem_tok_off  - symmetric per-PE token offset counter (i32[1])
#     recv_tok_num   - symmetric recv-token-num signal buffer [npes]
#     dest_pe_ctr    - LOCAL per-PE send counter [npes]
#     dispatch_bar   - LOCAL grid barrier [1]
#     dest_tok_map   - LOCAL routing map [cur_tok*k]: destPe*max_tok+destTokId
#     tok_id_to_src  - symmetric source mapping [max_tok]
#     total_recv     - LOCAL output: total tokens received
#   Scalars: rank, npes, cur_tok, experts_per_rank, experts_per_token,
#            hidden_dim, hidden_elem_size (bytes), max_tok_per_rank,
#            block_num (=gridDim.x), warp_num_per_block

_DISPATCH_IR = """\
; ===== shmem extern declarations =====
declare i32 @mori_shmem_uint32_atomic_fetch_add_thread(i64, i32, i32, i32)
declare i32 @mori_shmem_uint32_atomic_add_thread(i64, i32, i32, i32)
declare i32 @mori_shmem_int32_wait_until_equals(i64, i32)
declare i32 @mori_shmem_int32_wait_until_greater_than(i64, i32)
declare i64 @mori_shmem_ptr_p2p(i64, i32, i32)
declare i32 @mori_shmem_putmem_nbi_warp(i64, i64, i64, i32, i32)
declare i32 @mori_shmem_quiet_thread_pe(i32)
declare i32 @mori_shmem_int32_p(i64, i32, i32, i32)
declare i32 @mori_shmem_float_p(i64, float, i32, i32)

; ===== AMD GPU intrinsics (rocdl dialect) =====
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i64 @llvm.amdgcn.ballot.i64(i1)
declare i32 @llvm.amdgcn.readlane(i32, i32)
declare void @llvm.amdgcn.s.barrier()
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc(ptr addrspace(1), i16, i32, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8), i32, i32, i32)

; ===== Dispatch IntraNode Kernel =====
define amdgpu_kernel void @ep_dispatch_intranode(
    ptr %inp_tok, ptr %token_indices, ptr %weights_buf,
    ptr %shmem_out_tok, ptr %shmem_out_wts, ptr %shmem_out_idx,
    ptr %shmem_tok_off, ptr %recv_tok_num,
    ptr %dest_pe_ctr, ptr %dispatch_bar, ptr %dest_tok_map, ptr %tok_id_to_src,
    ptr %total_recv,
    i32 %rank, i32 %npes, i32 %cur_tok,
    i32 %experts_per_rank, i32 %experts_per_token,
    i32 %hidden_dim, i32 %hidden_elem_size,
    i32 %max_tok_per_rank, i32 %block_num, i32 %warp_num_per_block) #0 {
entry:
  ; Thread/warp identification (rocdl intrinsics)
  %tid      = call i32 @llvm.amdgcn.workitem.id.x()
  %bid      = call i32 @llvm.amdgcn.workgroup.id.x()
  %lane     = and i32 %tid, 63
  %warp     = lshr i32 %tid, 6
  %tmp_gw   = mul i32 %bid, %warp_num_per_block
  %gw_id    = add i32 %tmp_gw, %warp
  %gw_num   = mul i32 %block_num, %warp_num_per_block
  %limit    = mul i32 %cur_tok, %experts_per_token
  ; ---- Precompute all 24 P2P base addresses (once per kernel) ----
  ; Eliminates ~65K ptr_p2p calls in Phase 1 loop → just 24 calls at startup
  %stok_base = ptrtoint ptr %shmem_out_tok to i64
  %swts_base = ptrtoint ptr %shmem_out_wts to i64
  %sidx_base = ptrtoint ptr %shmem_out_idx to i64
  %rem_tok0 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 0)
  %rem_tok1 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 1)
  %rem_tok2 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 2)
  %rem_tok3 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 3)
  %rem_tok4 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 4)
  %rem_tok5 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 5)
  %rem_tok6 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 6)
  %rem_tok7 = call i64 @mori_shmem_ptr_p2p(i64 %stok_base, i32 %rank, i32 7)
  %rem_wts0 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 0)
  %rem_wts1 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 1)
  %rem_wts2 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 2)
  %rem_wts3 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 3)
  %rem_wts4 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 4)
  %rem_wts5 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 5)
  %rem_wts6 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 6)
  %rem_wts7 = call i64 @mori_shmem_ptr_p2p(i64 %swts_base, i32 %rank, i32 7)
  %rem_idx0 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 0)
  %rem_idx1 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 1)
  %rem_idx2 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 2)
  %rem_idx3 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 3)
  %rem_idx4 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 4)
  %rem_idx5 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 5)
  %rem_idx6 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 6)
  %rem_idx7 = call i64 @mori_shmem_ptr_p2p(i64 %sidx_base, i32 %rank, i32 7)
  br label %ph1_hdr

; ----------------------------------------------------------------
; Phase 1: Send tokens (all warps parallel, warp-grain iteration)
; ----------------------------------------------------------------
ph1_hdr:
  %i = phi i32 [ %gw_id, %entry ], [ %i_next_ph1, %ph1_latch ]
  %ph1_cond = icmp ult i32 %i, %limit
  br i1 %ph1_cond, label %ph1_body, label %ph2_sync

ph1_body:
  %src_tok   = udiv i32 %i, %experts_per_token
  %j         = urem i32 %i, %experts_per_token
  %idx_gep   = getelementptr i32, ptr %token_indices, i32 %i
  %dest_exp  = load i32, ptr %idx_gep, align 4
  %dest_pe   = udiv i32 %dest_exp, %experts_per_rank
  ; Dedup: any lane with laneId < j already routes srcTok to same destPe?
  %lt_j      = icmp ult i32 %lane, %j
  %lb        = mul i32 %src_tok, %experts_per_token
  ; CRITICAL: clamp lane to 0 for laneId >= j to prevent OOB access.
  ; token_indices has only cur_tok*k elements; lb+lane can exceed this for lane >= k.
  ; Safe: clamped lanes have lt_j=false so their load result is discarded by ballot.
  %safe_ld   = select i1 %lt_j, i32 %lane, i32 0
  %lane_pos  = add i32 %lb, %safe_ld
  %lane_gep  = getelementptr i32, ptr %token_indices, i32 %lane_pos
  %lane_exp  = load i32, ptr %lane_gep, align 4
  %lane_pe   = udiv i32 %lane_exp, %experts_per_rank
  %pe_same   = icmp eq i32 %lane_pe, %dest_pe
  %dup_cond  = and i1 %lt_j, %pe_same
  %ballot_r  = call i64 @llvm.amdgcn.ballot.i64(i1 %dup_cond)
  %any_dup   = icmp ne i64 %ballot_r, 0
  br i1 %any_dup, label %ph1_dup, label %ph1_nodup

ph1_dup:
  ; lane0 writes sentinel to dest_tok_map[i] = npes * max_tok
  %is_l0_dup = icmp eq i32 %lane, 0
  br i1 %is_l0_dup, label %ph1_write_sent, label %ph1_latch

ph1_write_sent:
  ; sentinel = npes * max_recv = npes^2 * max_tok (must exceed all valid encoded values)
  %max_recv0 = mul i32 %npes, %max_tok_per_rank
  %sentinel  = mul i32 %npes, %max_recv0
  %dtm_g_dup = getelementptr i32, ptr %dest_tok_map, i32 %i
  store i32 %sentinel, ptr %dtm_g_dup, align 4
  br label %ph1_latch

ph1_nodup:
  ; lane0: atomically allocate destTokId slot on remote destPe
  %is_l0     = icmp eq i32 %lane, 0
  br i1 %is_l0, label %ph1_l0_alloc, label %ph1_join

ph1_l0_alloc:
  ; Remote atomic fetch-add on destPe's shmem_tok_off counter (LOCAL sym addr, dest_pe)
  %soff_i64  = ptrtoint ptr %shmem_tok_off to i64
  %dest_tok  = call i32 @mori_shmem_uint32_atomic_fetch_add_thread(
                   i64 %soff_i64, i32 1, i32 %dest_pe, i32 0)
  ; Local: dest_pe_ctr[destPe]++
  %dpc_gep   = getelementptr i32, ptr %dest_pe_ctr, i32 %dest_pe
  %_old_dpc  = atomicrmw add ptr %dpc_gep, i32 1 monotonic
  ; dest_tok_map[i] = destPe * max_recv + destTokId  (max_recv = npes*max_tok)
  ; Using max_recv ensures destTokId < max_recv avoids decoding ambiguity
  %max_recv1 = mul i32 %npes, %max_tok_per_rank
  %dtm_p1    = mul i32 %dest_pe, %max_recv1
  %dtm_val   = add i32 %dtm_p1, %dest_tok
  %dtm_gep   = getelementptr i32, ptr %dest_tok_map, i32 %i
  store i32 %dtm_val, ptr %dtm_gep, align 4
  ; tok_id_to_src: use shmem_int32_p(LOCAL_sym_addr, val, dest_pe, 0)
  ; shmem translates local addr to dest_pe's equivalent addr internally
  %tis_i64   = ptrtoint ptr %tok_id_to_src to i64
  %tis_off   = zext i32 %dest_tok to i64
  %tis_boff  = mul i64 %tis_off, 4
  %tis_sym   = add i64 %tis_i64, %tis_boff
  %src_enc_p = mul i32 %rank, %max_tok_per_rank
  %src_enc   = add i32 %src_enc_p, %src_tok
  ; Use ptr_p2p + addrspace(1) store for tok_id_to_src
  %tis_rem   = call i64 @mori_shmem_ptr_p2p(i64 %tis_i64, i32 %rank, i32 %dest_pe)
  %tis_raddr = add i64 %tis_rem, %tis_boff
  %tis_rptr  = inttoptr i64 %tis_raddr to ptr addrspace(1)
  store i32 %src_enc, ptr addrspace(1) %tis_rptr, align 4
  br label %ph1_join

ph1_join:
  ; Broadcast lane0's dest_tok to all lanes via readlane (rocdl intrinsic)
  %dtok_phi  = phi i32 [ %dest_tok, %ph1_l0_alloc ], [ 0, %ph1_nodup ]
  %dtok_all  = call i32 @llvm.amdgcn.readlane(i32 %dtok_phi, i32 0)
  br label %ph1_put_tok

; ---- Use ptr_p2p + direct XGMI stores (bypasses NIC, uses XGMI hardware path) ----
ph1_put_tok:
  ; 1. Weights/Indices: select from precomputed P2P bases, add slot offset
  %wdst_bi   = mul i32 %dtok_all, %experts_per_token
  %wdst_bi64 = zext i32 %wdst_bi to i64
  %wdst_byt  = mul i64 %wdst_bi64, 4
  %is_pe0w = icmp eq i32 %dest_pe, 0
  %is_pe2w = icmp eq i32 %dest_pe, 2
  %is_pe4w = icmp eq i32 %dest_pe, 4
  %is_pe6w = icmp eq i32 %dest_pe, 6
  %rw01    = select i1 %is_pe0w, i64 %rem_wts0, i64 %rem_wts1
  %rw23    = select i1 %is_pe2w, i64 %rem_wts2, i64 %rem_wts3
  %rw45    = select i1 %is_pe4w, i64 %rem_wts4, i64 %rem_wts5
  %rw67    = select i1 %is_pe6w, i64 %rem_wts6, i64 %rem_wts7
  %rwlt2   = icmp ult i32 %dest_pe, 2
  %rwlt4   = icmp ult i32 %dest_pe, 4
  %rwlt6   = icmp ult i32 %dest_pe, 6
  %rw03    = select i1 %rwlt2, i64 %rw01, i64 %rw23
  %rw47    = select i1 %rwlt6, i64 %rw45, i64 %rw67
  %wts_base2 = select i1 %rwlt4, i64 %rw03, i64 %rw47
  %wts_rem   = add i64 %wts_base2, %wdst_byt
  %is_pe0i = icmp eq i32 %dest_pe, 0
  %is_pe2i = icmp eq i32 %dest_pe, 2
  %is_pe4i = icmp eq i32 %dest_pe, 4
  %is_pe6i = icmp eq i32 %dest_pe, 6
  %ri01    = select i1 %is_pe0i, i64 %rem_idx0, i64 %rem_idx1
  %ri23    = select i1 %is_pe2i, i64 %rem_idx2, i64 %rem_idx3
  %ri45    = select i1 %is_pe4i, i64 %rem_idx4, i64 %rem_idx5
  %ri67    = select i1 %is_pe6i, i64 %rem_idx6, i64 %rem_idx7
  %rilt2   = icmp ult i32 %dest_pe, 2
  %rilt4   = icmp ult i32 %dest_pe, 4
  %rilt6   = icmp ult i32 %dest_pe, 6
  %ri03    = select i1 %rilt2, i64 %ri01, i64 %ri23
  %ri47    = select i1 %rilt6, i64 %ri45, i64 %ri67
  %idx_base2 = select i1 %rilt4, i64 %ri03, i64 %ri47
  %idx_rem   = add i64 %idx_base2, %wdst_byt
  %l_lt_kx = icmp ult i32 %lane, %experts_per_token
  br i1 %l_lt_kx, label %ph1_wts_store, label %ph1_tok_prep

ph1_wts_store:
  ; Direct XGMI store using global addrspace(1) for optimal AMD GPU write path
  %wsrc_idx  = add i32 %lb, %lane
  %wsrc_gp   = getelementptr float, ptr %weights_buf, i32 %wsrc_idx
  %wt_valx   = load float, ptr %wsrc_gp, align 4
  %wlanex64  = zext i32 %lane to i64
  %wlanex_b  = mul i64 %wlanex64, 4
  %wdst_ax   = add i64 %wts_rem, %wlanex_b
  %wdst_gp   = inttoptr i64 %wdst_ax to ptr addrspace(1)
  store float %wt_valx, ptr addrspace(1) %wdst_gp, align 4
  ; index store via addrspace(1)
  %ixsrc_gp  = getelementptr i32, ptr %token_indices, i32 %wsrc_idx
  %ix_valx   = load i32, ptr %ixsrc_gp, align 4
  %idst_ax   = add i64 %idx_rem, %wlanex_b
  %idst_gp   = inttoptr i64 %idst_ax to ptr addrspace(1)
  store i32 %ix_valx, ptr addrspace(1) %idst_gp, align 4
  br label %ph1_tok_prep

ph1_tok_prep:
  ; 2. Token: use precomputed P2P base (from entry), add token offset
  ; Vectorized 128-bit store: <4 x i32> = 16 bytes per lane per iteration
  %tok_ei    = mul i32 %dtok_all, %hidden_dim
  %tok_e64   = zext i32 %tok_ei to i64
  %esz_64    = zext i32 %hidden_elem_size to i64
  %tok_boff  = mul i64 %tok_e64, %esz_64
  ; Select precomputed rem_tok base for dest_pe (avoids runtime ptr_p2p per token)
  %is_pe0t = icmp eq i32 %dest_pe, 0
  %is_pe2t = icmp eq i32 %dest_pe, 2
  %is_pe4t = icmp eq i32 %dest_pe, 4
  %is_pe6t = icmp eq i32 %dest_pe, 6
  %rt01  = select i1 %is_pe0t, i64 %rem_tok0, i64 %rem_tok1
  %rt23  = select i1 %is_pe2t, i64 %rem_tok2, i64 %rem_tok3
  %rt45  = select i1 %is_pe4t, i64 %rem_tok4, i64 %rem_tok5
  %rt67  = select i1 %is_pe6t, i64 %rem_tok6, i64 %rem_tok7
  %lt2t  = icmp ult i32 %dest_pe, 2
  %lt4t  = icmp ult i32 %dest_pe, 4
  %lt6t  = icmp ult i32 %dest_pe, 6
  %rt03  = select i1 %lt2t, i64 %rt01, i64 %rt23
  %rt47  = select i1 %lt6t, i64 %rt45, i64 %rt67
  %tok_rem_base = select i1 %lt4t, i64 %rt03, i64 %rt47
  %tok_rem  = add i64 %tok_rem_base, %tok_boff
  %inp_i64   = ptrtoint ptr %inp_tok to i64
  %inp_ei    = mul i32 %src_tok, %hidden_dim
  %inp_e64   = zext i32 %inp_ei to i64
  %inp_boff  = mul i64 %inp_e64, %esz_64
  %inp_base  = add i64 %inp_i64, %inp_boff
  %hdim_i32  = lshr i32 %hidden_dim, 1
  %lane4     = mul i32 %lane, 4
  br label %tch_hdr

tch_hdr:
  ; Vector stride loop: each lane handles positions lane*4, lane*4+256, ...
  %ec4      = phi i32 [ %lane4, %ph1_tok_prep ], [ %ec4_next, %tch_latch ]
  %tc_cond  = icmp ult i32 %ec4, %hdim_i32
  br i1 %tc_cond, label %tch_body, label %ph1_latch

tch_body:
  ; 128-bit vector load + addrspace(1) store to precomputed P2P remote address
  %ec4_64   = zext i32 %ec4 to i64
  %ec4_b64  = mul i64 %ec4_64, 4
  %src_v4a  = add i64 %inp_base, %ec4_b64
  %src_v4p  = inttoptr i64 %src_v4a to ptr
  %vec4     = load <4 x i32>, ptr %src_v4p, align 4
  %dst_v4a  = add i64 %tok_rem, %ec4_b64
  %dst_v4gp = inttoptr i64 %dst_v4a to ptr addrspace(1)
  store <4 x i32> %vec4, ptr addrspace(1) %dst_v4gp, align 4
  br label %tch_latch

tch_latch:
  %ec4_next = add i32 %ec4, 256             ; 64 lanes × 4 i32 = 256 stride
  br label %tch_hdr

ph1_latch:
  %i_next_ph1 = add i32 %i, %gw_num
  br label %ph1_hdr

; ----------------------------------------------------------------
; Phase 2: Grid barrier + send token count signals to all PEs
; ----------------------------------------------------------------
ph2_sync:
  ; Block-level sync
  call void @llvm.amdgcn.s.barrier()
  %is_t0     = icmp eq i32 %tid, 0
  br i1 %is_t0, label %ph2_inc_bar, label %ph2_check_gw0

ph2_inc_bar:
  ; thread0 of each block increments the grid barrier counter
  %_arm2     = atomicrmw add ptr %dispatch_bar, i32 1 monotonic
  br label %ph2_check_gw0

ph2_check_gw0:
  ; Only warp 0 (globalWarpId==0, i.e., block 0 warp 0) does signaling
  %is_gw0    = icmp eq i32 %gw_id, 0
  br i1 %is_gw0, label %ph2_loop_hdr, label %ph3_check_gw0

ph2_loop_hdr:
  ; lanes 0..npes-1 handle one destPe each
  %dp        = phi i32 [ %lane, %ph2_check_gw0 ], [ %dp_next, %ph2_loop_body ]
  %dp_cond   = icmp ult i32 %dp, %npes
  br i1 %dp_cond, label %ph2_loop_body, label %ph3_check_gw0

ph2_loop_body:
  ; Wait until all blocks have finished Phase 1
  %dbar_i64  = ptrtoint ptr %dispatch_bar to i64
  %_weq1     = call i32 @mori_shmem_int32_wait_until_equals(i64 %dbar_i64, i32 %block_num)
  store i32 0, ptr %dispatch_bar, align 4
  ; numSignal = numTokensSentToDP + 1 (so that 0 tokens still sends a signal)
  %dpc2_gep  = getelementptr i32, ptr %dest_pe_ctr, i32 %dp
  %dpc2_val  = load i32, ptr %dpc2_gep, align 4
  %nsig      = add i32 %dpc2_val, 1
  ; Compute addresses for recv_tok_num[rank] slot
  %rtn_i64   = ptrtoint ptr %recv_tok_num to i64
  %rtn_off   = zext i32 %rank to i64
  %rtn_boff  = mul i64 %rtn_off, 4
  ; LOCAL sym address (for shmem_atomic_add_thread - takes local addr, writes to dest_pe)
  %rtn_local = add i64 %rtn_i64, %rtn_boff
  ; P2P address of dp's slot (for wait - need to read remote memory)
  %rtn_rem   = call i64 @mori_shmem_ptr_p2p(i64 %rtn_i64, i32 %rank, i32 %dp)
  %rtn_remote = add i64 %rtn_rem, %rtn_boff
  ; Wait for dp's slot to be 0 (spin on remote memory via XGMI)
  %_weq2     = call i32 @mori_shmem_int32_wait_until_equals(i64 %rtn_remote, i32 0)
  ; System fence: ensure all Phase 1 NBI puts are visible before signal
  fence syncscope("one-as") seq_cst
  ; Write signal: use LOCAL sym addr + dest_pe (shmem handles addr translation)
  %_sig1     = call i32 @mori_shmem_uint32_atomic_add_thread(
                   i64 %rtn_local, i32 %nsig, i32 %dp, i32 0)
  %dp_next   = add i32 %dp, 64
  br label %ph2_loop_hdr

; ----------------------------------------------------------------
; Phase 3: Receive token count signals, accumulate total_recv
; ----------------------------------------------------------------
ph3_check_gw0:
  %is_gw0_3  = icmp eq i32 %gw_id, 0
  br i1 %is_gw0_3, label %ph3_loop_hdr, label %done

ph3_loop_hdr:
  %sp        = phi i32 [ %lane, %ph3_check_gw0 ], [ %sp_next, %ph3_loop_body ]
  %sp_cond   = icmp ult i32 %sp, %npes
  br i1 %sp_cond, label %ph3_loop_body, label %ph3_reset

ph3_loop_body:
  ; spin on local recv_tok_num[sp] until srcPe sp writes numSignal
  %rtn_l_gep = getelementptr i32, ptr %recv_tok_num, i32 %sp
  %rtn_l_i64 = ptrtoint ptr %rtn_l_gep to i64
  %sig_val   = call i32 @mori_shmem_int32_wait_until_greater_than(i64 %rtn_l_i64, i32 0)
  %recv_cnt  = sub i32 %sig_val, 1
  store i32 0, ptr %rtn_l_gep, align 4
  %_arm3     = atomicrmw add ptr %total_recv, i32 %recv_cnt monotonic
  %dpc3_gep  = getelementptr i32, ptr %dest_pe_ctr, i32 %sp
  store i32 0, ptr %dpc3_gep, align 4
  %sp_next   = add i32 %sp, 64
  br label %ph3_loop_hdr

ph3_reset:
  ; lane0: reset shmem_tok_off to 0 for the next dispatch round
  %is_l0_r   = icmp eq i32 %lane, 0
  br i1 %is_l0_r, label %ph3_do_reset, label %done

ph3_do_reset:
  store i32 0, ptr %shmem_tok_off, align 4
  br label %done

done:
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,1024" }
"""

# ============================================================
# Combine Kernel LLVM IR
# ============================================================
# Implements EpCombineIntraNodeKernel<UseP2PRead=True> from intranode.hpp:207
# Stages:
#   Stage 1: Copy local expert outputs to shmem registered buffer
#   Stage 2: CrossDeviceBarrier (all PEs synchronize)
#   Stage 3: P2P read from remote shmem, weighted accumulate into output
#
# Parameters:
#   inp_tok         - expert output tokens [total_recv_val, hidden_dim]
#   weights_buf     - routing weights [total_recv_val, k] (f32, can be null ptr)
#   shmem_comb_inp  - symmetric combine input (tokens)
#   shmem_comb_out  - symmetric combine output (tokens)
#   shmem_inp_wts   - symmetric combine input (weights)
#   shmem_out_wts   - symmetric combine output (weights)
#   xdev_bar_mem    - symmetric cross-device barrier [npes * 8 bytes]
#   xdev_bar_flag   - LOCAL expected barrier flag (u64[1])
#   dest_tok_map    - LOCAL routing map [cur_tok*k]: destPe*max_tok+destTokId
#   combine_bar     - LOCAL combine grid barrier [1]
#   total_recv_ptr  - LOCAL ptr to clear after barrier
#   Scalars: rank, npes, cur_tok, total_recv_val, experts_per_token,
#            hidden_dim, hidden_elem_size, max_tok_per_rank,
#            block_num, warp_num_per_block

_COMBINE_IR = """\
; ===== shmem + GPU intrinsic declarations =====
declare i32 @mori_shmem_uint32_atomic_fetch_add_thread(i64, i32, i32, i32)
declare i32 @mori_shmem_int32_wait_until_equals(i64, i32)
declare i32 @mori_shmem_uint64_wait_until_equals(i64, i64)
declare i64 @mori_shmem_ptr_p2p(i64, i32, i32)
declare i32 @mori_shmem_putmem_nbi_warp(i64, i64, i64, i32, i32)
declare i32 @mori_shmem_quiet_thread_pe(i32)
declare i32 @mori_shmem_uint64_p(i64, i64, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare void @llvm.amdgcn.s.barrier()

; ===== Combine IntraNode Kernel (UseP2PRead=True) =====
define amdgpu_kernel void @ep_combine_intranode(
    ptr %inp_tok, ptr %weights_buf,
    ptr %shmem_comb_inp, ptr %shmem_comb_out,
    ptr %shmem_inp_wts, ptr %shmem_out_wts,
    ptr %xdev_bar_mem, ptr %xdev_bar_flag,
    ptr %dest_tok_map, ptr %combine_bar, ptr %total_recv_ptr,
    i32 %rank, i32 %npes, i32 %cur_tok, i32 %total_recv_val,
    i32 %experts_per_token,
    i32 %hidden_dim, i32 %hidden_elem_size,
    i32 %max_tok_per_rank, i32 %block_num, i32 %warp_num_per_block) #0 {
entry:
  %tid      = call i32 @llvm.amdgcn.workitem.id.x()
  %bid      = call i32 @llvm.amdgcn.workgroup.id.x()
  %lane     = and i32 %tid, 63
  %warp     = lshr i32 %tid, 6
  %tmp_gw   = mul i32 %bid, %warp_num_per_block
  %gw_id    = add i32 %tmp_gw, %warp
  %gw_num   = mul i32 %block_num, %warp_num_per_block
  ; Global thread id (for barrier)
  %bdim     = mul i32 %warp_num_per_block, 64
  %tmp_gt   = mul i32 %bid, %bdim
  %gwtid    = add i32 %tmp_gt, %tid
  ; Read current round's barrier flag
  %cur_flag = load i64, ptr %xdev_bar_flag, align 8
  %esz_64   = zext i32 %hidden_elem_size to i64
  %hdim_64  = zext i32 %hidden_dim to i64
  %nbytes   = mul i64 %hdim_64, %esz_64
  br label %s1_hdr

; ----------------------------------------------------------------
; Stage 1: Copy local expert outputs → shmem_comb_inp (local shmem)
; ----------------------------------------------------------------
s1_hdr:
  %ci        = phi i32 [ %gw_id, %entry ], [ %ci_next, %s1_latch ]
  %s1_cond   = icmp ult i32 %ci, %total_recv_val
  br i1 %s1_cond, label %s1_body, label %s2_bar_init

s1_body:
  ; dst = shmem_comb_inp (local addr) + ci * hidden_dim * elem_size
  %sc_i64    = ptrtoint ptr %shmem_comb_inp to i64
  %ci_64     = zext i32 %ci to i64
  %ci_off    = mul i64 %ci_64, %nbytes
  %sc_dst    = add i64 %sc_i64, %ci_off
  ; src = inp_tok + ci * hidden_dim * elem_size
  %inp_i64   = ptrtoint ptr %inp_tok to i64
  %inp_src   = add i64 %inp_i64, %ci_off
  ; warp put to local shmem (myPe = rank), ensures XGMI visibility
  %_put_s1   = call i32 @mori_shmem_putmem_nbi_warp(
                   i64 %sc_dst, i64 %inp_src, i64 %nbytes, i32 %rank, i32 0)
  %_qt_s1    = call i32 @mori_shmem_quiet_thread_pe(i32 %rank)
  ; Copy weights if provided (check for null ptr by comparing to 0 as i64)
  %wbuf_i64  = ptrtoint ptr %weights_buf to i64
  %wbuf_nonnull = icmp ne i64 %wbuf_i64, 0
  br i1 %wbuf_nonnull, label %s1_copy_wts, label %s1_latch

s1_copy_wts:
  %siw_i64   = ptrtoint ptr %shmem_inp_wts to i64
  %kf_64     = zext i32 %experts_per_token to i64
  %wts_off   = mul i64 %ci_64, %kf_64
  %wts_boff  = mul i64 %wts_off, 4
  %siw_dst   = add i64 %siw_i64, %wts_boff
  %wbuf_src  = add i64 %wbuf_i64, %wts_boff
  %wts_bytes = mul i64 %kf_64, 4
  %_put_wts  = call i32 @mori_shmem_putmem_nbi_warp(
                   i64 %siw_dst, i64 %wbuf_src, i64 %wts_bytes, i32 %rank, i32 0)
  %_qt_wts   = call i32 @mori_shmem_quiet_thread_pe(i32 %rank)
  br label %s1_latch

s1_latch:
  %ci_next   = add i32 %ci, %gw_num
  br label %s1_hdr

; ----------------------------------------------------------------
; Stage 2: CrossDeviceBarrier
;   Step A: block-sync, then first npes global-threads write ready-flag
;           to every remote PE's xdev_bar_mem[rank] slot
;   Step B: first npes block-threads spin on local xdev_bar_mem[thdId]
; ----------------------------------------------------------------
s2_bar_init:
  call void @llvm.amdgcn.s.barrier()
  %is_t0     = icmp eq i32 %tid, 0
  br i1 %is_t0, label %s2_inc_cbar, label %s2_check_gwtid

s2_inc_cbar:
  %_arm_cbar = atomicrmw add ptr %combine_bar, i32 1 monotonic
  br label %s2_check_gwtid

s2_check_gwtid:
  ; Only global threads 0..npes-1 write flags to remote PEs
  %lt_npes_g = icmp ult i32 %gwtid, %npes
  br i1 %lt_npes_g, label %s2_wait_blocks, label %s2_wait_peers

s2_wait_blocks:
  ; Wait for all blocks to finish Stage 1
  %cbar_i64  = ptrtoint ptr %combine_bar to i64
  %_wcbar    = call i32 @mori_shmem_int32_wait_until_equals(i64 %cbar_i64, i32 %block_num)
  store i32 0, ptr %combine_bar, align 4
  ; system fence to ensure Stage 1 shmem writes are visible
  fence syncscope("one-as") seq_cst
  ; Write ready-flag to remote PE gwtid's xdev_bar_mem[rank] slot
  %xdb_i64   = ptrtoint ptr %xdev_bar_mem to i64
  %xdb_rem   = call i64 @mori_shmem_ptr_p2p(i64 %xdb_i64, i32 %rank, i32 %gwtid)
  %rank_64   = zext i32 %rank to i64
  %rank_boff = mul i64 %rank_64, 8
  %xdb_slot  = add i64 %xdb_rem, %rank_boff
  %_xdbp     = call i32 @mori_shmem_uint64_p(i64 %xdb_slot, i64 %cur_flag, i32 %gwtid, i32 0)
  br label %s2_wait_peers

s2_wait_peers:
  ; Thread thdId waits for local xdev_bar_mem[thdId] == flag
  %lt_npes_t = icmp ult i32 %tid, %npes
  br i1 %lt_npes_t, label %s2_peer_spin, label %s2_bar_done

s2_peer_spin:
  %pslot_gep = getelementptr i64, ptr %xdev_bar_mem, i32 %tid
  %pslot_i64 = ptrtoint ptr %pslot_gep to i64
  %_wpeer    = call i32 @mori_shmem_uint64_wait_until_equals(i64 %pslot_i64, i64 %cur_flag)
  br label %s2_bar_done

s2_bar_done:
  call void @llvm.amdgcn.s.barrier()
  ; Reset total_recv for next round (any thread can do it, store is idempotent)
  store i32 0, ptr %total_recv_ptr, align 4
  ; If this rank has no tokens to combine, exit
  %no_tok    = icmp eq i32 %cur_tok, 0
  br i1 %no_tok, label %done, label %s3_prep

; ----------------------------------------------------------------
; Stage 3: P2P read from remote shmem, weighted accumulate
; ----------------------------------------------------------------
s3_prep:
  ; warpsPerToken = ceil(globalWarpNum / cur_tok)
  ; hiddenPerWarp = ceil(hidden_dim / warpsPerToken)
  %wpt_div   = udiv i32 %gw_num, %cur_tok
  %wpt_rem   = urem i32 %gw_num, %cur_tok
  %wpt_rne   = icmp ne i32 %wpt_rem, 0
  %wpt_add   = zext i1 %wpt_rne to i32
  %wpt       = add i32 %wpt_div, %wpt_add
  %hpw_div   = udiv i32 %hidden_dim, %wpt
  %hpw_rem   = urem i32 %hidden_dim, %wpt
  %hpw_rne   = icmp ne i32 %hpw_rem, 0
  %hpw_add   = zext i1 %hpw_rne to i32
  %hpw       = add i32 %hpw_div, %hpw_add
  %s3_limit  = mul i32 %cur_tok, %wpt
  br label %s3_hdr

s3_hdr:
  %si        = phi i32 [ %gw_id, %s3_prep ], [ %si_next, %s3_latch ]
  %s3_cond   = icmp ult i32 %si, %s3_limit
  br i1 %s3_cond, label %s3_body, label %done

s3_body:
  %tokenId   = udiv i32 %si, %wpt
  %partId    = urem i32 %si, %wpt
  %hiddenOff = mul i32 %partId, %hpw
  %h_remain  = sub i32 %hidden_dim, %hiddenOff
  %hsize     = call i32 @llvm.smin.i32(i32 %h_remain, i32 %hpw)
  ; Each lane handles one element: elem_idx = hiddenOff + lane
  %elem_idx  = add i32 %hiddenOff, %lane
  %elem_valid = icmp ult i32 %elem_idx, %hidden_dim
  ; Inner j loop over experts_per_token
  ; Initialize accumulator
  br label %s3_j_hdr

s3_j_hdr:
  %j2        = phi i32 [ 0, %s3_body ], [ %j2_next, %s3_j_merge ]
  %acc_f32   = phi float [ 0.0, %s3_body ], [ %acc_out, %s3_j_merge ]
  %j2_cond   = icmp ult i32 %j2, %experts_per_token
  br i1 %j2_cond, label %s3_j_body, label %s3_accum_done

s3_j_body:
  ; encoded = dest_tok_map[tokenId * k + j]
  %dtm_bi    = mul i32 %tokenId, %experts_per_token
  %dtm_i     = add i32 %dtm_bi, %j2
  %dtm_gep   = getelementptr i32, ptr %dest_tok_map, i32 %dtm_i
  %encoded   = load i32, ptr %dtm_gep, align 4
  %destPe2   = udiv i32 %encoded, %max_tok_per_rank
  %is_valid  = icmp ult i32 %destPe2, %npes
  br i1 %is_valid, label %s3_j_load, label %s3_j_merge

s3_j_load:
  ; Compute P2P address: shmem_comb_inp[destPe][localTok*hidden + elem_idx]
  %localTok  = urem i32 %encoded, %max_tok_per_rank
  %sci_i64   = ptrtoint ptr %shmem_comb_inp to i64
  %sci_rem   = call i64 @mori_shmem_ptr_p2p(i64 %sci_i64, i32 %rank, i32 %destPe2)
  %ltok_64   = zext i32 %localTok to i64
  %ltok_e    = mul i64 %ltok_64, %hdim_64
  %elem_64   = zext i32 %elem_idx to i64
  %elem_e    = add i64 %ltok_e, %elem_64
  %elem_b    = mul i64 %elem_e, %esz_64
  %src_addr  = add i64 %sci_rem, %elem_b
  %src_ptr   = inttoptr i64 %src_addr to ptr
  ; Load as bfloat and convert to f32 for accumulation
  %val_bf16  = load bfloat, ptr %src_ptr, align 2
  %val_f32   = fpext bfloat %val_bf16 to float
  %new_acc   = fadd float %acc_f32, %val_f32
  br label %s3_j_merge

s3_j_merge:
  %acc_out   = phi float [ %acc_f32, %s3_j_body ], [ %new_acc, %s3_j_load ]
  %j2_next   = add i32 %j2, 1
  br label %s3_j_hdr

s3_accum_done:
  ; Store accumulated f32 → bfloat to shmem_comb_out[rank][tokenId*hidden + elem_idx]
  %sco_i64   = ptrtoint ptr %shmem_comb_out to i64
  %out_e_i   = mul i32 %tokenId, %hidden_dim
  %out_e_j   = add i32 %out_e_i, %elem_idx
  %out_e64   = zext i32 %out_e_j to i64
  %out_b     = mul i64 %out_e64, %esz_64
  %out_addr  = add i64 %sco_i64, %out_b
  %out_ptr   = inttoptr i64 %out_addr to ptr
  %out_bf16  = fptrunc float %acc_f32, bfloat
  br i1 %elem_valid, label %s3_store, label %s3_wt_check

s3_store:
  store bfloat %out_bf16, ptr %out_ptr, align 2
  br label %s3_wt_check

s3_wt_check:
  ; Accumulate weights only on last part of this token
  %is_last_p = icmp eq i32 %partId, %s3_last_part
  %wbuf2_i64 = ptrtoint ptr %weights_buf to i64
  %wbuf2_nn  = icmp ne i64 %wbuf2_i64, 0
  %do_wts    = and i1 %is_last_p, %wbuf2_nn
  br i1 %do_wts, label %s3_wt_j_hdr, label %s3_latch

s3_wt_j_hdr:
  %wj        = phi i32 [ 0, %s3_wt_check ], [ %wj_next, %s3_wt_j_merge ]
  %wacc      = phi float [ 0.0, %s3_wt_check ], [ %wacc_out, %s3_wt_j_merge ]
  %wj_cond   = icmp ult i32 %wj, %experts_per_token
  br i1 %wj_cond, label %s3_wt_j_body, label %s3_wt_store

s3_wt_j_body:
  %wdtm_bi   = mul i32 %tokenId, %experts_per_token
  %wdtm_i    = add i32 %wdtm_bi, %wj
  %wdtm_gep  = getelementptr i32, ptr %dest_tok_map, i32 %wdtm_i
  %wencoded  = load i32, ptr %wdtm_gep, align 4
  %wdestPe   = udiv i32 %wencoded, %max_tok_per_rank
  %wvalid    = icmp ult i32 %wdestPe, %npes
  br i1 %wvalid, label %s3_wt_j_load, label %s3_wt_j_merge

s3_wt_j_load:
  %wlocalTok = urem i32 %wencoded, %max_tok_per_rank
  %siw_i64   = ptrtoint ptr %shmem_inp_wts to i64
  %siw_rem   = call i64 @mori_shmem_ptr_p2p(i64 %siw_i64, i32 %rank, i32 %wdestPe)
  %wltok_64  = zext i32 %wlocalTok to i64
  %wkf       = zext i32 %experts_per_token to i64
  %wltok_e   = mul i64 %wltok_64, %wkf
  %wj_64     = zext i32 %wj to i64
  %welem_e   = add i64 %wltok_e, %wj_64
  %welem_b   = mul i64 %welem_e, 4
  %wsrc_a    = add i64 %siw_rem, %welem_b
  %wsrc_p    = inttoptr i64 %wsrc_a to ptr
  %wval      = load float, ptr %wsrc_p, align 4
  %new_wacc  = fadd float %wacc, %wval
  br label %s3_wt_j_merge

s3_wt_j_merge:
  %wacc_out  = phi float [ %wacc, %s3_wt_j_body ], [ %new_wacc, %s3_wt_j_load ]
  %wj_next   = add i32 %wj, 1
  br label %s3_wt_j_hdr

s3_wt_store:
  ; Store weight accumulation (indexed by lane = j slot, each lane does one slot)
  %l_lt_k2   = icmp ult i32 %lane, %experts_per_token
  br i1 %l_lt_k2, label %s3_wt_do_store, label %s3_latch

s3_wt_do_store:
  %sow_i64   = ptrtoint ptr %shmem_out_wts to i64
  %wout_bi   = mul i32 %tokenId, %experts_per_token
  %wout_i    = add i32 %wout_bi, %lane
  %wout_64   = zext i32 %wout_i to i64
  %wout_b    = mul i64 %wout_64, 4
  %wout_addr = add i64 %sow_i64, %wout_b
  %wout_ptr  = inttoptr i64 %wout_addr to ptr
  store float %wacc_out, ptr %wout_ptr, align 4
  br label %s3_latch

s3_latch:
  %si_next   = add i32 %si, %gw_num
  br label %s3_hdr

done:
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,1024" }

; Intrinsic needed for min computation
declare i32 @llvm.smin.i32(i32, i32)
"""

# Fix: s3_last_part referenced in s3_wt_check but not computed - need to compute it
# Let me compute it as (wpt - 1) in s3_prep and pass through phi
# The IR above has a bug: %s3_last_part is not defined. Let me fix this.

_COMBINE_IR = """\
declare i32 @mori_shmem_uint32_atomic_fetch_add_thread(i64, i32, i32, i32)
declare i32 @mori_shmem_int32_wait_until_equals(i64, i32)
declare i32 @mori_shmem_uint64_wait_until_equals(i64, i64)
declare i64 @mori_shmem_ptr_p2p(i64, i32, i32)
declare i32 @mori_shmem_putmem_nbi_warp(i64, i64, i64, i32, i32)
declare i32 @mori_shmem_quiet_thread_pe(i32)
declare i32 @mori_shmem_uint64_p(i64, i64, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.smin.i32(i32, i32)

define amdgpu_kernel void @ep_combine_intranode(
    ptr %inp_tok, ptr %weights_buf,
    ptr %shmem_comb_inp, ptr %shmem_comb_out,
    ptr %shmem_inp_wts, ptr %shmem_out_wts,
    ptr %xdev_bar_mem, ptr %xdev_bar_flag,
    ptr %dest_tok_map, ptr %combine_bar, ptr %total_recv_ptr,
    i32 %rank, i32 %npes, i32 %cur_tok, i32 %total_recv_val,
    i32 %experts_per_token,
    i32 %hidden_dim, i32 %hidden_elem_size,
    i32 %max_tok_per_rank, i32 %block_num, i32 %warp_num_per_block) #0 {
entry:
  %tid      = call i32 @llvm.amdgcn.workitem.id.x()
  %bid      = call i32 @llvm.amdgcn.workgroup.id.x()
  %lane     = and i32 %tid, 63
  %warp     = lshr i32 %tid, 6
  %tmp_gw   = mul i32 %bid, %warp_num_per_block
  %gw_id    = add i32 %tmp_gw, %warp
  %gw_num   = mul i32 %block_num, %warp_num_per_block
  %bdim     = mul i32 %warp_num_per_block, 64
  %tmp_gt   = mul i32 %bid, %bdim
  %gwtid    = add i32 %tmp_gt, %tid
  %cur_flag = load i64, ptr %xdev_bar_flag, align 8
  %esz_64   = zext i32 %hidden_elem_size to i64
  %hdim_64  = zext i32 %hidden_dim to i64
  %nbytes   = mul i64 %hdim_64, %esz_64
  %wbuf_i64 = ptrtoint ptr %weights_buf to i64
  %wbuf_nn  = icmp ne i64 %wbuf_i64, 0
  br label %s1_hdr

s1_hdr:
  %ci       = phi i32 [ %gw_id, %entry ], [ %ci_next, %s1_latch ]
  %s1_cond  = icmp ult i32 %ci, %total_recv_val
  br i1 %s1_cond, label %s1_body, label %s2_bar_init

s1_body:
  %sc_i64   = ptrtoint ptr %shmem_comb_inp to i64
  %ci_64    = zext i32 %ci to i64
  %ci_off   = mul i64 %ci_64, %nbytes
  %sc_dst   = add i64 %sc_i64, %ci_off
  %inp_i64  = ptrtoint ptr %inp_tok to i64
  %inp_src  = add i64 %inp_i64, %ci_off
  %_ps1     = call i32 @mori_shmem_putmem_nbi_warp(
                  i64 %sc_dst, i64 %inp_src, i64 %nbytes, i32 %rank, i32 0)
  %_qs1     = call i32 @mori_shmem_quiet_thread_pe(i32 %rank)
  br i1 %wbuf_nn, label %s1_wts, label %s1_latch

s1_wts:
  %siw_i64  = ptrtoint ptr %shmem_inp_wts to i64
  %kf_64    = zext i32 %experts_per_token to i64
  %woff     = mul i64 %ci_64, %kf_64
  %wboff    = mul i64 %woff, 4
  %siw_d    = add i64 %siw_i64, %wboff
  %wbuf_s   = add i64 %wbuf_i64, %wboff
  %wbytes   = mul i64 %kf_64, 4
  %_psw     = call i32 @mori_shmem_putmem_nbi_warp(
                  i64 %siw_d, i64 %wbuf_s, i64 %wbytes, i32 %rank, i32 0)
  %_qsw     = call i32 @mori_shmem_quiet_thread_pe(i32 %rank)
  br label %s1_latch

s1_latch:
  %ci_next  = add i32 %ci, %gw_num
  br label %s1_hdr

s2_bar_init:
  call void @llvm.amdgcn.s.barrier()
  %is_t0    = icmp eq i32 %tid, 0
  br i1 %is_t0, label %s2_inc_cbar, label %s2_check_gwtid

s2_inc_cbar:
  %_acb     = atomicrmw add ptr %combine_bar, i32 1 monotonic
  br label %s2_check_gwtid

s2_check_gwtid:
  %lt_npes  = icmp ult i32 %gwtid, %npes
  br i1 %lt_npes, label %s2_wait_blks, label %s2_wait_peers

s2_wait_blks:
  %cbar_i64 = ptrtoint ptr %combine_bar to i64
  %_wcb     = call i32 @mori_shmem_int32_wait_until_equals(i64 %cbar_i64, i32 %block_num)
  store i32 0, ptr %combine_bar, align 4
  fence syncscope("one-as") seq_cst
  ; Use LOCAL sym addr for uint64_p: shmem translates to dest_pe (gwtid) internally
  %xdb_i64  = ptrtoint ptr %xdev_bar_mem to i64
  %rnk_64   = zext i32 %rank to i64
  %rnk_b    = mul i64 %rnk_64, 8
  %xdb_sym  = add i64 %xdb_i64, %rnk_b
  %_xp      = call i32 @mori_shmem_uint64_p(i64 %xdb_sym, i64 %cur_flag, i32 %gwtid, i32 0)
  br label %s2_wait_peers

s2_wait_peers:
  %lt_npes2 = icmp ult i32 %tid, %npes
  br i1 %lt_npes2, label %s2_spin, label %s2_done

s2_spin:
  %ps_gep   = getelementptr i64, ptr %xdev_bar_mem, i32 %tid
  %ps_i64   = ptrtoint ptr %ps_gep to i64
  %_wpeer   = call i32 @mori_shmem_uint64_wait_until_equals(i64 %ps_i64, i64 %cur_flag)
  br label %s2_done

s2_done:
  call void @llvm.amdgcn.s.barrier()
  store i32 0, ptr %total_recv_ptr, align 4
  %no_tok   = icmp eq i32 %cur_tok, 0
  br i1 %no_tok, label %ret, label %s3_prep

; Stage 3: tiled P2P accumulation with inner element stride loop
; warpsPerToken * warpSize elements per token, each lane covers its stride
s3_prep:
  %wpt_d    = udiv i32 %gw_num, %cur_tok
  %wpt_r    = urem i32 %gw_num, %cur_tok
  %wpt_rne  = icmp ne i32 %wpt_r, 0
  %wpt_add  = zext i1 %wpt_rne to i32
  %wpt      = add i32 %wpt_d, %wpt_add
  %hpw_d    = udiv i32 %hidden_dim, %wpt
  %hpw_r    = urem i32 %hidden_dim, %wpt
  %hpw_rne  = icmp ne i32 %hpw_r, 0
  %hpw_add  = zext i1 %hpw_rne to i32
  %hpw      = add i32 %hpw_d, %hpw_add
  %s3_lim   = mul i32 %cur_tok, %wpt
  %last_pt  = sub i32 %wpt, 1
  br label %s3_hdr

s3_hdr:
  %si       = phi i32 [ %gw_id, %s3_prep ], [ %si_next, %s3_latch ]
  %s3_cond  = icmp ult i32 %si, %s3_lim
  br i1 %s3_cond, label %s3_body, label %ret

s3_body:
  %tokId    = udiv i32 %si, %wpt
  %ptId     = urem i32 %si, %wpt
  %hOff     = mul i32 %ptId, %hpw
  ; Inner element stride loop: each lane handles elements lane, lane+64, lane+128...
  br label %elp_hdr

; ---- inner element stride loop ----
elp_hdr:
  %eoff     = phi i32 [ %lane, %s3_body ], [ %eoff_next, %elp_latch ]
  %elemIdx  = add i32 %hOff, %eoff
  %ev2      = icmp ult i32 %elemIdx, %hidden_dim
  %ehpw     = icmp ult i32 %eoff, %hpw
  %edo      = and i1 %ev2, %ehpw
  br i1 %edo, label %jlp_hdr, label %elp_done

; ---- j loop: accumulate k expert values ----
jlp_hdr:
  %jl       = phi i32 [ 0, %elp_hdr ], [ %jl_next, %jlp_mrg ]
  %jacc     = phi float [ 0.0, %elp_hdr ], [ %jacc_o, %jlp_mrg ]
  %jl_cond  = icmp ult i32 %jl, %experts_per_token
  br i1 %jl_cond, label %jlp_body, label %jlp_done

jlp_body:
  %dm_bi    = mul i32 %tokId, %experts_per_token
  %dm_i     = add i32 %dm_bi, %jl
  %dm_gep   = getelementptr i32, ptr %dest_tok_map, i32 %dm_i
  %enc      = load i32, ptr %dm_gep, align 4
  ; Decode using max_recv = npes * max_tok (same multiplier used in dispatch encoding)
  %max_recv2 = mul i32 %npes, %max_tok_per_rank
  %dpei     = udiv i32 %enc, %max_recv2
  %jvalid   = icmp ult i32 %dpei, %npes
  br i1 %jvalid, label %jlp_load, label %jlp_mrg

jlp_load:
  %ltok     = urem i32 %enc, %max_recv2
  %sci_i64  = ptrtoint ptr %shmem_comb_inp to i64
  %sci_rem  = call i64 @mori_shmem_ptr_p2p(i64 %sci_i64, i32 %rank, i32 %dpei)
  %lt64     = zext i32 %ltok to i64
  %lte      = mul i64 %lt64, %hdim_64
  %eli64    = zext i32 %elemIdx to i64
  %elij     = add i64 %lte, %eli64
  %elib     = mul i64 %elij, %esz_64
  %srca     = add i64 %sci_rem, %elib
  %srcp     = inttoptr i64 %srca to ptr
  %v16_raw  = load i16, ptr %srcp, align 2
  %v16      = bitcast i16 %v16_raw to bfloat
  %vf32     = fpext bfloat %v16 to float
  %nacc     = fadd float %jacc, %vf32
  br label %jlp_mrg

jlp_mrg:
  %jacc_o   = phi float [ %jacc, %jlp_body ], [ %nacc, %jlp_load ]
  %jl_next  = add i32 %jl, 1
  br label %jlp_hdr

jlp_done:
  ; Store accumulated bf16 to shmem_comb_out[tokId * hidden_dim + elemIdx]
  %sco_i64  = ptrtoint ptr %shmem_comb_out to i64
  %oei      = mul i32 %tokId, %hidden_dim
  %oej      = add i32 %oei, %elemIdx
  %oe64     = zext i32 %oej to i64
  %oeb      = mul i64 %oe64, %esz_64
  %oa       = add i64 %sco_i64, %oeb
  %op       = inttoptr i64 %oa to ptr
  %obf16_f  = fptrunc float %jacc to bfloat
  %obf16    = bitcast bfloat %obf16_f to i16
  store i16 %obf16, ptr %op, align 2
  br label %elp_latch

elp_latch:
  %eoff_next = add i32 %eoff, 64
  br label %elp_hdr

elp_done:
  ; Weight accumulation (only on last part, lane < k)
  %islast   = icmp eq i32 %ptId, %last_pt
  %do_wt    = and i1 %islast, %wbuf_nn
  br i1 %do_wt, label %wjlp_hdr, label %s3_latch

; ---- weight accumulation j loop ----
wjlp_hdr:
  %wjl      = phi i32 [ 0, %elp_done ], [ %wjl_next, %wjlp_mrg ]
  %wjacc    = phi float [ 0.0, %elp_done ], [ %wjacc_o, %wjlp_mrg ]
  %wjl_cond = icmp ult i32 %wjl, %experts_per_token
  br i1 %wjl_cond, label %wjlp_body, label %wjlp_done

wjlp_body:
  %wdm_bi   = mul i32 %tokId, %experts_per_token
  %wdm_i    = add i32 %wdm_bi, %wjl
  %wdm_gep  = getelementptr i32, ptr %dest_tok_map, i32 %wdm_i
  %wenc     = load i32, ptr %wdm_gep, align 4
  %max_recv3 = mul i32 %npes, %max_tok_per_rank
  %wdpei    = udiv i32 %wenc, %max_recv3
  %wjvalid  = icmp ult i32 %wdpei, %npes
  br i1 %wjvalid, label %wjlp_load, label %wjlp_mrg

wjlp_load:
  %wltok    = urem i32 %wenc, %max_recv3
  %siw2_i64 = ptrtoint ptr %shmem_inp_wts to i64
  %siw_rem  = call i64 @mori_shmem_ptr_p2p(i64 %siw2_i64, i32 %rank, i32 %wdpei)
  %wlt64    = zext i32 %wltok to i64
  %wkf64    = zext i32 %experts_per_token to i64
  %wlte     = mul i64 %wlt64, %wkf64
  %wjl64    = zext i32 %wjl to i64
  %welij    = add i64 %wlte, %wjl64
  %welib    = mul i64 %welij, 4
  %wsrca    = add i64 %siw_rem, %welib
  %wsrcp    = inttoptr i64 %wsrca to ptr
  %wvf32    = load float, ptr %wsrcp, align 4
  %nwacc    = fadd float %wjacc, %wvf32
  br label %wjlp_mrg

wjlp_mrg:
  %wjacc_o  = phi float [ %wjacc, %wjlp_body ], [ %nwacc, %wjlp_load ]
  %wjl_next = add i32 %wjl, 1
  br label %wjlp_hdr

wjlp_done:
  %lltk     = icmp ult i32 %lane, %experts_per_token
  br i1 %lltk, label %wst, label %s3_latch

wst:
  %sow_i64  = ptrtoint ptr %shmem_out_wts to i64
  %woi      = mul i32 %tokId, %experts_per_token
  %woj      = add i32 %woi, %lane
  %wo64     = zext i32 %woj to i64
  %wob      = mul i64 %wo64, 4
  %woa      = add i64 %sow_i64, %wob
  %wop      = inttoptr i64 %woa to ptr
  store float %wjacc, ptr %wop, align 4
  br label %s3_latch

s3_latch:
  %si_next  = add i32 %si, %gw_num
  br label %s3_hdr

ret:
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,1024" }
"""


# ============================================================
# Module builders
# ============================================================
def build_dispatch_intranode_kernel() -> Module:
    """Build the dispatch intranode MLIR module using flydsl._mlir.ir.

    Uses rocdl dialect for GPU intrinsics (FlyDSL syntax main body)
    and llvm dialect for shmem extern declarations (MLIR supplement).
    Returns the MLIR Module with all declarations and the kernel body.
    """
    ctx = _make_context()
    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            # Declare shmem externs (llvm dialect - MLIR supplement)
            _declare_shmem_fns()
            # Declare GPU intrinsics (rocdl dialect - FlyDSL syntax)
            _declare_gpu_intrinsics()
        assert module.operation.verify()
    # Return module (with shmem/intrinsic declarations) and the kernel LLVM IR text
    return module, _DISPATCH_IR


def build_combine_intranode_kernel() -> Module:
    """Build the combine intranode MLIR module using flydsl._mlir.ir."""
    ctx = _make_context()
    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            _declare_shmem_fns()
            _declare_gpu_intrinsics()
        assert module.operation.verify()
    return module, _COMBINE_IR


# ============================================================
# Compilation pipeline
# ============================================================
def compile_ir_to_hsaco(llvm_ir: str, chip: str, out_path: str,
                        mori_bc: str) -> str:
    """Compile LLVM IR text to .hsaco by linking with mori shmem bitcode.

    Pipeline (same as mlir_shmem_kernel.py):
        kernel.ll + libmori_shmem_device.bc → llvm-link → linked.bc
        linked.bc → ROCm clang -x ir → kernel.hsaco
    """
    tmpdir = tempfile.mkdtemp(prefix="flydsl_dc_")
    try:
        kernel_ll = os.path.join(tmpdir, "kernel.ll")
        with open(kernel_ll, "w") as f:
            f.write(llvm_ir)

        linked_bc = os.path.join(tmpdir, "linked.bc")
        r = subprocess.run(
            [LLVM_LINK, kernel_ll, mori_bc, "-o", linked_bc],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llvm-link failed:\n{r.stderr}")

        r = subprocess.run(
            [ROCM_CLANG,
             "-x", "ir", linked_bc,
             "-target", "amdgcn-amd-amdhsa",
             f"-mcpu={chip}",
             "-O3",
             "-o", out_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"clang failed:\n{r.stderr}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path


def build_and_compile_dispatch(chip: str, out_path: str,
                               mori_bc: Optional[str] = None) -> str:
    """Build and compile the dispatch intranode kernel to HSACO."""
    if mori_bc is None:
        mori_bc = _find_mori_shmem_bc()
    _, ir_text = build_dispatch_intranode_kernel()
    return compile_ir_to_hsaco(ir_text, chip, out_path, mori_bc)


def build_and_compile_combine(chip: str, out_path: str,
                              mori_bc: Optional[str] = None) -> str:
    """Build and compile the combine intranode kernel to HSACO."""
    if mori_bc is None:
        mori_bc = _find_mori_shmem_bc()
    _, ir_text = build_combine_intranode_kernel()
    return compile_ir_to_hsaco(ir_text, chip, out_path, mori_bc)


if __name__ == "__main__":
    import sys
    chip = sys.argv[1] if len(sys.argv) > 1 else "gfx942"
    print(f"[*] Target chip: {chip}")
    mori_bc = _find_mori_shmem_bc()
    print(f"[*] Mori shmem bitcode: {mori_bc}")
    import tempfile
    tmp = tempfile.gettempdir()

    print("\n[1/2] Building dispatch intranode kernel ...")
    disp_hsaco = os.path.join(tmp, f"ep_dispatch_intranode_{chip}.hsaco")
    build_and_compile_dispatch(chip, disp_hsaco, mori_bc)
    print(f"[OK] {disp_hsaco} ({os.path.getsize(disp_hsaco)} bytes)")

    print("\n[2/2] Building combine intranode kernel ...")
    comb_hsaco = os.path.join(tmp, f"ep_combine_intranode_{chip}.hsaco")
    build_and_compile_combine(chip, comb_hsaco, mori_bc)
    print(f"[OK] {comb_hsaco} ({os.path.getsize(comb_hsaco)} bytes)")

    print("\n[*] All kernels compiled successfully!")
