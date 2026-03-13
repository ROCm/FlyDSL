#!/usr/bin/env python3
"""
Intra-node allreduce (bf16 sum) using FlyDSL @flyc.jit + mori shmem device API.

Port of mori/examples/shmem/ir/test_triton_allreduce.py (Kernel A: P2P load)
to FlyDSL syntax.  Each PE reads from all PEs via P2P pointers and accumulates
locally — every PE gets the same result.

Usage:
    torchrun --nproc_per_node=2 tests/test_flydsl_allreduce.py
    torchrun --nproc_per_node=8 tests/test_flydsl_allreduce.py

Requirements:
    - Multiple AMD GPUs (MI300X / MI350)
    - mori installed with shmem support (editable install)
    - FlyDSL built with shmem JIT support (mgpuSetModuleLoadHook)
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.lowlevel import (
    const_i32,
    store_i32_at,
)

import mori.ir.flydsl as mori_shmem


# ===================================================================
# 1. Kernel: device-side P2P load allreduce (bf16 → f32 accumulation)
# ===================================================================
#
# FlyDSL encoding rules (from dispatch_combine_intranode_v2.py):
#   - Compile-time constants → closure variables, not kernel params
#   - Dynamic if conditions must use function-call form (icmp_eq_i32 etc.)
#   - For loops at kernel top level, induction variable is index → idx_to_i32()
#   - All tensor addresses passed as fx.Int64, accessed via lowlevel load/store
#
# This kernel is equivalent to allreduce_p2p_kernel in Triton:
#   - Each block handles BLOCK_SIZE elements
#   - For each PE, if it's self → local load; else → ptr_p2p + P2P load
#   - Accumulate in f32, store result as i32 (bf16 bit pattern)


def make_simple_allreduce_kernel(*, npes: int, BLOCK_SIZE: int = 256):
    """Simplified allreduce kernel: each thread copies one i32 from all PEs.

    This kernel demonstrates the core shmem P2P pattern:
      1. mori_shmem.my_pe() — get local PE id
      2. mori_shmem.ptr_p2p(addr, mype, target_pe) — get remote address
      3. load_i32_global(remote_addr) — P2P load via XGMI

    For correctness validation, each PE writes its PE id to its buffer,
    then allreduce sums all PE ids → expected = 0 + 1 + ... + (npes-1).
    """
    @flyc.kernel
    def simple_allreduce_kernel(
        data_ptr: fx.Int64,        # symmetric buffer (i32 per element)
        result_ptr: fx.Int64,      # result buffer (i32 per element)
        n_elems: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        global_tid = bid * BLOCK_SIZE + tid

        from flydsl.expr.lowlevel import _unwrap
        from flydsl._mlir.dialects import arith as _arith

        mype = mori_shmem.my_pe()

        # Accumulate sum across all PEs (range_constexpr = Python-level unroll)
        acc = const_i32(0)

        for pe in range_constexpr(npes):
            # Get P2P address for PE `pe` (works for self too — returns local addr)
            pe_const = const_i32(pe)
            src_addr = mori_shmem.ptr_p2p(data_ptr, mype, pe_const)

            # P2P load via load_i32_global (addrspace 1) with element offset
            # load_i32_at computes base + offset * 4 internally
            from flydsl.expr.lowlevel import load_i32_at as _load_i32_at
            val = _load_i32_at(src_addr, global_tid)

            # Accumulate: acc += val
            acc_v = _unwrap(acc)
            val_v = _unwrap(val)
            acc = _arith.AddIOp(acc_v, val_v).result

        # Store result
        store_i32_at(result_ptr, global_tid, acc)

    return simple_allreduce_kernel


# ===================================================================
# 2. JIT wrapper
# ===================================================================
def make_allreduce_jit(npes: int, n_elems: int, BLOCK_SIZE: int = 256):
    """Create @flyc.jit launcher for the allreduce kernel.

    Note: Python int → JitArgument always maps to Int32 (32-bit).
    For Int64 parameters (64-bit addresses), callers must wrap values
    in fx.Int64(value) before calling the JIT function.
    """
    kernel = make_simple_allreduce_kernel(npes=npes, BLOCK_SIZE=BLOCK_SIZE)
    grid_x = (n_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

    @flyc.jit
    def allreduce_launch(
        data_ptr: fx.Int64,
        result_ptr: fx.Int64,
        n_elems_param: fx.Int32,
    ):
        kernel(data_ptr, result_ptr, n_elems_param).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
        )

    return allreduce_launch


# ===================================================================
# 3. Distributed setup (same as Triton version)
# ===================================================================
def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    import mori.shmem as ms
    ms.shmem_torch_process_group_init("default")
    mype, npes = ms.shmem_mype(), ms.shmem_npes()
    print(f"[PE {mype}/{npes}] initialized on GPU {local_rank}")
    return mype, npes


def cleanup():
    import mori.shmem as ms
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


# ===================================================================
# 4. Test
# ===================================================================
def test_allreduce(mype, npes):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    N = 256  # number of i32 elements
    BLOCK_SIZE = 256

    print(f"\n[PE {mype}] === FlyDSL allreduce (i32 PE-sum, N={N}) ===")

    # Each PE fills its symmetric buffer with its PE id
    symm_buf = mori_shmem_create_tensor((N,), torch.int32)
    symm_buf.fill_(mype)

    result = torch.zeros(N, dtype=torch.int32, device="cuda")

    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # Expected: sum of all PE ids = 0 + 1 + ... + (npes-1) = npes*(npes-1)/2
    expected_val = npes * (npes - 1) // 2
    print(f"[PE {mype}] Expected allreduce result: {expected_val}")

    # Create and run the JIT kernel
    # Wrap addresses in fx.Int64 (Python int → Int32 by default in JIT registry)
    allreduce_fn = make_allreduce_jit(npes=npes, n_elems=N, BLOCK_SIZE=BLOCK_SIZE)
    allreduce_fn(fx.Int64(symm_buf.data_ptr()), fx.Int64(result.data_ptr()), N)
    torch.cuda.synchronize()

    # Verify
    max_err = (result - expected_val).abs().max().item()
    print(f"[PE {mype}] max_err={max_err}")
    assert max_err == 0, f"PE {mype}: allreduce mismatch, max_err={max_err}"
    print(f"[PE {mype}] PASS")


# ===================================================================
# 5. Compile-only test (single GPU, no torchrun needed)
# ===================================================================
def test_compile_only():
    """Verify the allreduce kernel compiles through the full MLIR pipeline.

    Uses COMPILE_ONLY=1 — no kernel launch, no shmem init required.
    Can run with: python tests/test_flydsl_allreduce.py --compile-only
    """
    os.environ["COMPILE_ONLY"] = "1"

    N = 256
    BLOCK_SIZE = 256
    npes = 2  # compile-time constant

    print(f"[compile-only] Compiling allreduce kernel (npes={npes}, N={N})...")

    allreduce_fn = make_allreduce_jit(npes=npes, n_elems=N, BLOCK_SIZE=BLOCK_SIZE)

    data = torch.zeros(N, dtype=torch.int32, device="cuda")
    result = torch.zeros(N, dtype=torch.int32, device="cuda")

    # COMPILE_ONLY=1: should return None on success
    # Wrap addresses in fx.Int64 (Python int → Int32 by default in JIT registry)
    ret = allreduce_fn(fx.Int64(data.data_ptr()), fx.Int64(result.data_ptr()), N)
    assert ret is None, f"Expected None from COMPILE_ONLY, got {ret}"
    print("[compile-only] PASS — allreduce kernel compiled successfully")


# ===================================================================
# main
# ===================================================================
def main():
    if "--compile-only" in sys.argv:
        test_compile_only()
        return

    mype, npes = setup_distributed()
    try:
        test_allreduce(mype, npes)
        if mype == 0:
            print(f"\n{'=' * 60}")
            print(f"  FlyDSL allreduce test PASSED on {npes} PEs")
            print(f"  (@flyc.jit + mori shmem device API)")
            print(f"{'=' * 60}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
