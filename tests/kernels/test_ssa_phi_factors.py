#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Factor analysis: which condition causes the SSA phi explosion?

2x2 factorial design:

  +-------------------+---------------------+---------------------+
  |                   | Identical branches  | Different branches  |
  +-------------------+---------------------+---------------------+
  | Small phi (1 vec) | test_ssa_phi_old.py | Factor B (here)     |
  |                   | -> NO explosion     | -> ???               |
  +-------------------+---------------------+---------------------+
  | Large phi (32 vec)| Factor A (here)     | test_ssa_phi_diverge.|
  |                   | -> ???              | -> EXPLOSION         |
  +-------------------+---------------------+---------------------+

Factor A: large phi + identical branches
  - 32 x f32x4 yielded through phi
  - Both branches do the same 2-MFMA chain with different data (A vs B buffer)
  - Tests whether phi WIDTH alone causes the explosion
  - Note: LLVM may merge identical branches via v_cndmask pointer selection,
    eliminating phi entirely. If so, this proves structural difference is
    required to *preserve* phi, not to cause the explosion itself.

Factor B: small phi + different branches
  - 1 x f32x4 yielded through phi
  - Branch A: 3-MFMA chain, Branch B: 2-MFMA + MulF
  - Tests whether branch ASYMMETRY alone causes the explosion
  - Even with different branches, only 4 VGPRs cross phi — expect no explosion.
"""

import os
import sys

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.vector import full
from flydsl.expr.numeric import Float16, Float32
from flydsl.expr import buffer_ops

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import vector as _vd

from flydsl.runtime.device import get_rocm_arch

from kernels.kernels_common import get_warp_size

WARP_SIZE = get_warp_size()
BLOCK_THREADS = 64
NUM_ACCU_VECS = 32  # for large phi


def _raw(v):
    if hasattr(v, "ir_value"):
        return v.ir_value()
    if hasattr(v, "result"):
        return v.result
    return v


def _i32_const(val):
    return _std_arith.ConstantOp(T.i32, val).result


# ─────────────────────────────────────────────────────────────────────
# Factor A: Large phi (32 x f32x4) + identical branches
# ─────────────────────────────────────────────────────────────────────

def build_factor_a(N: int):
    """Large phi, identical branches.

    Both branches: load from different buffers, 2-MFMA chain, yield 32 x f32x4.
    Branch computation is IDENTICAL in structure (only data source differs).
    """

    @flyc.kernel
    def factor_a_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32x4_type = T.f32x4

        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)

        tid_i32 = _raw(ArithValue(tid))
        bid_i32 = _raw(ArithValue(bid))
        row_off = _std_arith.MulIOp(bid_i32, _i32_const(N)).result
        tid_off = _std_arith.MulIOp(tid_i32, _i32_const(4)).result
        base_off = _std_arith.AddIOp(row_off, tid_off).result

        b_frag = _raw(full(4, Float16(1.0), Float16))
        c_zero = _raw(full(4, Float32(0.0), Float32))

        def _branch_body_identical(rsrc):
            """Same computation in both branches: 2-MFMA chain per vec."""
            results = []
            for vec_idx in range_constexpr(NUM_ACCU_VECS):
                elem_off = _std_arith.AddIOp(
                    base_off,
                    _i32_const(vec_idx * BLOCK_THREADS * 4),
                ).result
                a_frag = buffer_ops.buffer_load(
                    rsrc, elem_off, vec_width=4, dtype=T.f16
                )
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, c_zero, 0, 0, 0]
                ))
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
                ))
                results.append(accu)
            return results

        warp_id = ArithValue(tid) // WARP_SIZE
        is_first_warp = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.eq,
            _raw(warp_id),
            _i32_const(0),
        ).result

        result_types = [f32x4_type] * NUM_ACCU_VECS
        if_op = scf.IfOp(is_first_warp, result_types, has_else=True)

        # Both branches: same structure, different buffer
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            scf.YieldOp(_branch_body_identical(rsrc_a))
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            scf.YieldOp(_branch_body_identical(rsrc_b))

        merged = [if_op.results[i] for i in range(NUM_ACCU_VECS)]
        total = merged[0]
        for i in range_constexpr(1, NUM_ACCU_VECS):
            total = _std_arith.AddFOp(total, merged[i]).result

        for elem in range_constexpr(4):
            val = _vd.extract(total, [], [elem])
            store_idx = _std_arith.AddIOp(
                _std_arith.AddIOp(row_off, tid_off).result,
                _i32_const(elem),
            ).result
            buffer_ops.buffer_store(val, rsrc_c, store_idx)

    @flyc.jit
    def launch_a(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = factor_a_kernel(A, B, C)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_a


# ─────────────────────────────────────────────────────────────────────
# Factor B: Small phi (1 x f32x4) + different branches
# ─────────────────────────────────────────────────────────────────────

def build_factor_b(N: int):
    """Small phi, different branches.

    Branch A: 3-MFMA chain -> yield 1 x f32x4
    Branch B: 2-MFMA + MulF -> yield 1 x f32x4
    Only 1 x f32x4 (= 4 VGPRs) crosses phi.
    """

    @flyc.kernel
    def factor_b_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32x4_type = T.f32x4

        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)

        tid_i32 = _raw(ArithValue(tid))
        bid_i32 = _raw(ArithValue(bid))
        row_off = _std_arith.MulIOp(bid_i32, _i32_const(N)).result
        tid_off = _std_arith.MulIOp(tid_i32, _i32_const(4)).result
        base_off = _std_arith.AddIOp(row_off, tid_off).result

        b_frag = _raw(full(4, Float16(1.0), Float16))
        c_zero = _raw(full(4, Float32(0.0), Float32))
        scale_vec = _raw(full(4, Float32(0.5), Float32))

        warp_id = ArithValue(tid) // WARP_SIZE
        is_first_warp = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.eq,
            _raw(warp_id),
            _i32_const(0),
        ).result

        result_types = [f32x4_type]  # only 1 x f32x4 through phi
        if_op = scf.IfOp(is_first_warp, result_types, has_else=True)

        # Branch A: 3-MFMA chain
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            a_frag = buffer_ops.buffer_load(
                rsrc_a, base_off, vec_width=4, dtype=T.f16
            )
            accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                f32x4_type, [a_frag, b_frag, c_zero, 0, 0, 0]
            ))
            accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
            ))
            accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
            ))
            scf.YieldOp([accu])

        # Branch B: 2-MFMA + MulF (structurally different)
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            a_frag = buffer_ops.buffer_load(
                rsrc_b, base_off, vec_width=4, dtype=T.f16
            )
            accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                f32x4_type, [a_frag, b_frag, c_zero, 0, 0, 0]
            ))
            accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
            ))
            accu = _std_arith.MulFOp(accu, scale_vec).result
            scf.YieldOp([accu])

        # After merge: 1 x f32x4 through phi
        result = if_op.results[0]

        for elem in range_constexpr(4):
            val = _vd.extract(result, [], [elem])
            store_idx = _std_arith.AddIOp(
                _std_arith.AddIOp(row_off, tid_off).result,
                _i32_const(elem),
            ).result
            buffer_ops.buffer_store(val, rsrc_c, store_idx)

    @flyc.jit
    def launch_b(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = factor_b_kernel(A, B, C)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_b


# ─────────────────────────────────────────────────────────────────────
# Shared analysis
# ─────────────────────────────────────────────────────────────────────

def _parse_isa_stats(isa_path):
    stats = {
        "vgpr_spill_count": 0,
        "vgpr_count": 0,
        "total_mfma_count": 0,
        "unique_mfma_dsts": set(),
    }
    try:
        with open(isa_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    for line in lines:
        stripped = line.strip()
        if "v_mfma" in stripped:
            stats["total_mfma_count"] += 1
            for p in stripped.split():
                if p.startswith("v[") and ":" in p:
                    stats["unique_mfma_dsts"].add(p.rstrip(","))
        if "vgpr_spill_count" in stripped:
            try:
                stats["vgpr_spill_count"] = int(stripped.split(":")[-1].strip())
            except ValueError:
                pass
        if ".vgpr_count" in stripped:
            try:
                stats["vgpr_count"] = int(stripped.split(":")[-1].strip())
            except ValueError:
                pass
    return stats


def _count_phi(llvm_ir_path):
    try:
        with open(llvm_ir_path) as f:
            ir_text = f.read()
        return ir_text.count(" phi "), ir_text.count("phi <4 x float>")
    except FileNotFoundError:
        return None, None


def _run_test(name, build_fn, N, kernel_name):
    M = 4

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"  M={M}, N={N}, BLOCK_THREADS={BLOCK_THREADS}")
    print(f"  arch={get_rocm_arch()}")
    print(f"{'='*70}")

    launch_fn = build_fn(N)

    a_dev = torch.randn((M, N), device="cuda", dtype=torch.float16)
    b_dev = torch.randn((M, N), device="cuda", dtype=torch.float16)
    c_dev = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    launch_fn(a_dev, b_dev, c_dev, M, stream=stream)
    torch.cuda.synchronize()

    nonzero = (c_dev[:, :BLOCK_THREADS * 4] != 0).sum().item()
    print(f"  Non-zero outputs: {nonzero}/{BLOCK_THREADS * 4 * M}")
    assert nonzero > 0

    isa_path = os.path.expanduser(f"~/.flydsl/debug/{kernel_name}_0/17_final_isa.s")
    stats = _parse_isa_stats(isa_path)
    phi_total, phi_f32x4 = _count_phi(
        os.path.expanduser(f"~/.flydsl/debug/{kernel_name}_0/16_llvm_ir.ll")
    )

    result = {}
    if stats:
        result = {
            "mfma_count": stats["total_mfma_count"],
            "unique_dsts": len(stats["unique_mfma_dsts"]),
            "vgpr_count": stats["vgpr_count"],
            "vgpr_spills": stats["vgpr_spill_count"],
            "phi_total": phi_total or 0,
            "phi_f32x4": phi_f32x4 or 0,
        }
        print(f"\n  Results:")
        print(f"    MFMA instructions: {result['mfma_count']}")
        print(f"    Unique MFMA dst groups: {result['unique_dsts']}")
        print(f"    VGPR count: {result['vgpr_count']}")
        print(f"    VGPR spill count: {result['vgpr_spills']}")
        print(f"    LLVM phi nodes: {result['phi_total']} total, {result['phi_f32x4']} x <4xf32>")

    print(f"  PASSED")
    return result


def test_factor_a_large_phi_identical():
    """Factor A: large phi + identical branches."""
    N = NUM_ACCU_VECS * BLOCK_THREADS * 4
    return _run_test(
        "Factor A: Large phi (32 x f32x4) + IDENTICAL branches",
        build_factor_a, N, "factor_a_kernel",
    )


def test_factor_b_small_phi_different():
    """Factor B: small phi + different branches."""
    N = BLOCK_THREADS * 8
    return _run_test(
        "Factor B: Small phi (1 x f32x4) + DIFFERENT branches",
        build_factor_b, N, "factor_b_kernel",
    )


if __name__ == "__main__":
    if "FLYDSL_DUMP_IR" not in os.environ:
        os.environ["FLYDSL_DUMP_IR"] = "1"

    results = {}
    results["factor_a"] = test_factor_a_large_phi_identical()
    results["factor_b"] = test_factor_b_small_phi_different()

    print(f"\n{'='*70}")
    print(f"FACTOR ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test':<45} {'Phi f32x4':>10} {'Uniq Dsts':>10} {'VGPRs':>6} {'Spills':>7}")
    print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*6} {'-'*7}")

    for name, r in results.items():
        if r:
            print(f"{name:<45} {r['phi_f32x4']:>10} {r['unique_dsts']:>10} "
                  f"{r['vgpr_count']:>6} {r['vgpr_spills']:>7}")

    print(f"\nCompare with:")
    print(f"  Old test (small phi + identical):    0 phi f32x4, few unique dsts")
    print(f"  Current test (large phi + different): 32 phi f32x4, 76 unique dsts, 134 VGPRs")
    print(f"{'='*70}")

    sys.exit(0)
