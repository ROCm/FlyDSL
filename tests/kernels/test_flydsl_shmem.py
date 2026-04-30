#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
FlyDSL + mori shmem integration test — cross-PE basic operations.

Validates the full pipeline:
  ffi → link_extern → bitcode linking → post-load module init

Usage:
    torchrun --nproc_per_node=2 tests/kernels/test_flydsl_shmem.py
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("mori", reason="mori package required for shmem tests")

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLYDSL_PY = os.path.join(_HERE, "../../python")
if os.path.isdir(_FLYDSL_PY) and _FLYDSL_PY not in sys.path:
    sys.path.insert(0, _FLYDSL_PY)

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.extern import ffi
from flydsl.compiler.extern_link import link_extern

import mori.shmem as ms
from mori.ir.flydsl.runtime import get_bitcode_path

from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.ir import IntegerAttr as _IntAttr, IntegerType as _IntTy


def _mori_shmem_module_init(hip_module: int) -> None:
    ms.shmem_module_init(hip_module)


_MORI_SHMEM_BITCODE = get_bitcode_path()


def _mori_extern(symbol, args, ret):
    return link_extern(
        ffi(symbol, args, ret),
        bitcode_path=_MORI_SHMEM_BITCODE,
        module_init_fn=_mori_shmem_module_init,
    )


class mori_shmem:
    my_pe = _mori_extern("mori_shmem_my_pe", [], "int32")
    n_pes = _mori_extern("mori_shmem_n_pes", [], "int32")
    int32_p = _mori_extern("mori_shmem_int32_p", ["uint64", "int32", "int32", "int32"], "int32")
    quiet_thread = _mori_extern("mori_shmem_quiet_thread", [], "int32")


# ===================================================================
# Pointer helpers: pass a 64-bit address as two Int32 halves
# ===================================================================


def _split_ptr(ptr: int):
    """Split a 64-bit pointer into (lo, hi) 32-bit halves."""
    return ptr & 0xFFFFFFFF, (ptr >> 32) & 0xFFFFFFFF


def _reconstruct_i64(lo, hi):
    """Reconstruct i64 from two i32 halves inside a kernel."""
    _i64 = _IntTy.get_signless(64)
    _i32 = _IntTy.get_signless(32)

    def _lv(v):
        if isinstance(v, _ir.Value):
            return v
        if hasattr(v, "__fly_values__"):
            vals = v.__fly_values__()
            if len(vals) == 1:
                return vals[0]
        if isinstance(v, int):
            return _llvm_d.ConstantOp(_i32, _IntAttr.get(_i32, v)).result
        raise TypeError(f"Cannot convert {type(v).__name__} to ir.Value")

    lo_v = _llvm_d.ZExtOp(_i64, _lv(lo)).res
    hi_v = _llvm_d.ZExtOp(_i64, _lv(hi)).res
    _nuw = _ir.Attribute.parse("#llvm.overflow<none>")
    hi_shifted = _llvm_d.ShlOp(
        hi_v,
        _llvm_d.ConstantOp(_i64, _IntAttr.get(_i64, 32)).result,
        _nuw,
    ).result
    return _llvm_d.OrOp(hi_shifted, lo_v).result


def _store_i32_at(addr_i64, offset_i32, val_i32):
    """Store i32 *val* at addr_i64 + offset*4 (global, monotonic)."""
    _i64 = _IntTy.get_signless(64)
    _i32 = _IntTy.get_signless(32)
    _nuw = _ir.Attribute.parse("#llvm.overflow<none>")

    def _lv(v):
        if isinstance(v, _ir.Value):
            return v
        if hasattr(v, "__fly_values__"):
            vals = v.__fly_values__()
            if len(vals) == 1:
                return vals[0]
        if isinstance(v, int):
            return _llvm_d.ConstantOp(_i32, _IntAttr.get(_i32, v)).result
        raise TypeError(f"Cannot convert {type(v).__name__}")

    off = _lv(offset_i32)
    val = _lv(val_i32)
    off64 = _llvm_d.ZExtOp(_i64, off).res if off.type == _i32 else off
    byte_off = _llvm_d.MulOp(
        off64,
        _llvm_d.ConstantOp(_i64, _IntAttr.get(_i64, 4)).result,
        _nuw,
    ).result
    addr = _llvm_d.AddOp(addr_i64, byte_off, _nuw).result
    gptr = _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1),
        addr,
    ).result
    _llvm_d.StoreOp(
        val,
        gptr,
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


# ===================================================================
# 1. Kernels
# ===================================================================


@flyc.kernel
def shmem_basic_kernel(out_lo: fx.Int32, out_hi: fx.Int32):
    """Write my_pe and n_pes to output buffer."""
    addr = _reconstruct_i64(out_lo, out_hi)
    pe = mori_shmem.my_pe()
    npe = mori_shmem.n_pes()
    _store_i32_at(addr, 0, pe)
    _store_i32_at(addr, 1, npe)


@flyc.kernel
def shmem_put_kernel(symm_lo: fx.Int32, symm_hi: fx.Int32, value: fx.Int32):
    """Put *value* into the peer PE's symmetric buffer via int32_p."""
    symm_addr = _reconstruct_i64(symm_lo, symm_hi)
    pe = mori_shmem.my_pe()
    npe = mori_shmem.n_pes()
    dest_pe = arith.remui(arith.addi(pe, arith.constant(1)), npe)
    mori_shmem.int32_p(symm_addr, value, dest_pe, arith.constant(0))
    mori_shmem.quiet_thread()


# ===================================================================
# 2. JIT launchers
# ===================================================================


@flyc.jit
def launch_basic(out_lo: fx.Int32, out_hi: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    shmem_basic_kernel(out_lo, out_hi).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@flyc.jit
def launch_put(symm_lo: fx.Int32, symm_hi: fx.Int32, value: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    shmem_put_kernel(symm_lo, symm_hi, value).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


# ===================================================================
# 3. Distributed setup
# ===================================================================


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    ms.shmem_torch_process_group_init("default")
    mype, npes = ms.shmem_mype(), ms.shmem_npes()
    print(f"[PE {mype}/{npes}] initialized on GPU {local_rank}")
    return mype, npes


def cleanup():
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


# ===================================================================
# 4. Tests
# ===================================================================


def run_basic(mype, npes):
    """Verify my_pe() and n_pes() return correct values."""
    print(f"\n[PE {mype}] === FlyDSL shmem_basic_kernel ===")
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    lo, hi = _split_ptr(out.data_ptr())
    stream = torch.cuda.current_stream()
    launch_basic(lo, hi, stream=stream)
    torch.cuda.synchronize()

    got_pe, got_npe = out[0].item(), out[1].item()
    print(f"[PE {mype}] my_pe={got_pe}, n_pes={got_npe}")
    assert got_pe == mype, f"PE {mype}: expected my_pe={mype}, got {got_pe}"
    assert got_npe == npes, f"PE {mype}: expected n_pes={npes}, got {got_npe}"
    print(f"[PE {mype}] [FlyDSL] basic  PASS")


def run_put(mype, npes):
    """Send an int32 to peer PE via shmem_put, verify receipt."""
    print(f"\n[PE {mype}] === FlyDSL shmem_put_kernel ===")
    buf = ms.mori_shmem_create_tensor((1,), torch.int32)
    buf.fill_(-1)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    value = mype * 100 + 42
    lo, hi = _split_ptr(buf.data_ptr())
    stream = torch.cuda.current_stream()
    launch_put(lo, hi, value, stream=stream)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    src_pe = (mype - 1 + npes) % npes
    expected = src_pe * 100 + 42
    got = buf.item()
    print(f"[PE {mype}] buf={got}, expected={expected} (from PE {src_pe})")
    assert got == expected, f"PE {mype}: expected {expected}, got {got}"
    print(f"[PE {mype}] [FlyDSL] put    PASS")


# ===================================================================
# main
# ===================================================================


def main():
    mype, npes = setup_distributed()
    try:
        run_basic(mype, npes)
        run_put(mype, npes)
        if mype == 0:
            print(f"\n{'=' * 60}")
            print(f"  All tests PASSED on {npes} PEs (FlyDSL + mori shmem)")
            print(f"{'=' * 60}")
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
