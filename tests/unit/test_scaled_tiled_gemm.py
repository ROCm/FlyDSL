# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Scaled tiled GEMM lowering: scale indexing, accumulation, and byte selectors."""

import itertools

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu
from flydsl._mlir.passmanager import PassManager

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

PIPELINE = (
    "builtin.module(fly-layout-lowering,canonicalize,fly-convert-atom-call-to-ssa-form,"
    "fly-promote-regmem-to-vectorssa,canonicalize,convert-fly-to-rocdl,canonicalize)"
)
MFMA = "rocdl.mfma.scale.f32.16x16x128.f8f6f4"


def _walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _build_gemm(rank, traversal=None, invalid_scale=None, scale_mode="tensor"):
    m, n, k = (1, 1, 1) if rank == 1 else (2, 3, 2 if rank == 3 else 1)
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        device = gpu.GPUModuleOp("test")
        block = device.operation.regions[0].blocks.append()
        with ir.InsertionPoint(block):
            function = gpu.GPUFuncOp(
                ir.FunctionType.get([], [ir.VectorType.get([4 * m * n], fx.Float32.ir_type)]), sym_name="gemm"
            )
            with ir.InsertionPoint(function.add_entry_block()):
                atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
                if invalid_scale == "stateless":
                    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
                mma = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (1, 1, 1))) if rank > 1 else atom
                a_shape = (32,) if rank == 1 else ((32, m) if rank == 2 else (32, m, k))
                b_shape = (32,) if rank == 1 else ((32, n) if rank == 2 else (32, n, k))
                a = fx.make_rmem_tensor(32 if rank == 1 else a_shape, fx.Float8E4M3FN)
                b = fx.make_rmem_tensor(32 if rank == 1 else b_shape, fx.Float8E4M3FN)
                c = fx.make_rmem_tensor(4 if rank == 1 else (4, m, n), fx.Float32)
                d = fx.make_fragment_like(c)
                a.fill(1.0)
                b.fill(1.0)
                c.fill(2.0)
                sa_shape = (1,) + a_shape[1:]
                if invalid_scale == "rank":
                    sa_shape = (1,)
                elif invalid_scale == "mode0":
                    sa_shape = (2,) + a_shape[1:]
                elif invalid_scale == "tiles":
                    sa_shape = (1, m + 1) + a_shape[2:]
                sa = fx.make_rmem_tensor(sa_shape, fx.Float32 if invalid_scale == "dtype" else fx.Int32)
                sb = fx.make_rmem_tensor((1,) + b_shape[1:], fx.Int32)
                if invalid_scale is None:
                    for kt, mt in itertools.product(range(k), range(m)):
                        sa[(0, mt, kt)[:rank]] = fx.Int32(117 + mt + m * kt)
                    for kt, nt in itertools.product(range(k), range(n)):
                        sb[(0, nt, kt)[:rank]] = fx.Int32(120 + nt + n * kt)
                kwargs = {}
                if traversal == "layout":
                    kwargs["traversal_layout"] = fx.make_layout((m, n, k), (n * k, k, 1))
                elif traversal is not None:
                    kwargs["traversal_order"] = getattr(fx.GemmTraversalOrder, traversal)
                if scale_mode == "a_only":
                    kwargs["scale_a"] = sa
                elif scale_mode == "scalar_a":
                    kwargs.update(scale_a=117, scale_b=sb)
                elif scale_mode == "scalar_b":
                    kwargs.update(scale_a=sa, scale_b=fx.Int32(120))
                else:
                    kwargs.update(scale_a=sa, scale_b=sb)
                fx.gemm(mma, d, a, b, c, **kwargs)
                gpu.ReturnOp([d.load().ir_value()])
    return module, (m, n, k)


@pytest.mark.parametrize(
    "rank,traversal", [(1, None)] + list(itertools.product([2, 3], [None, "KNM", "KMN_Serpentine", "layout"]))
)
def test_scaled_gemm_expands_all_tiles(rank, traversal):
    with ir.Context(), ir.Location.unknown():
        module, (m, n, k) = _build_gemm(rank, traversal)
        assert module.operation.verify()
        PassManager.parse(PIPELINE).run(module.operation)
        ops = list(_walk(module.operation))
        assert not any(op.name in ("fly.gemm", "scf.for", "scf.while") for op in ops)
        calls = [op for op in ops if op.name == MFMA]
        assert len(calls) == m * n * k
        seen = set()
        last = {}
        for op in calls:
            sa, sb = [ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:]]
            ka, mt = divmod(sa - 117, m)
            kb, nt = divmod(sb - 120, n)
            assert ka == kb
            seen.add((mt, nt, ka))
            key = mt, nt
            if key in last:
                assert op.operands[2] == last[key]
            else:
                assert op.operands[2].owner.name == "arith.constant"
            last[key] = op.results[0]
        assert seen == set(itertools.product(range(m), range(n), range(k)))


@pytest.mark.parametrize("mode", ["a_only", "scalar_a", "scalar_b"])
def test_tiled_gemm_scalar_and_optional_scales(mode):
    with ir.Context(), ir.Location.unknown():
        module, (m, n, _) = _build_gemm(2, scale_mode=mode)
        PassManager.parse(PIPELINE).run(module.operation)
        scales = [
            tuple(ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:])
            for op in _walk(module.operation)
            if op.name == MFMA
        ]
        expected_a = [117] * m if mode == "scalar_a" else list(range(117, 117 + m))
        expected_b = [0] * n if mode == "a_only" else ([120] * n if mode == "scalar_b" else list(range(120, 120 + n)))
        assert sorted(scales) == sorted(itertools.product(expected_a, expected_b))


@pytest.mark.parametrize(
    "invalid,diagnostic",
    [
        ("dtype", "i32 elements"),
        ("rank", "operand rank"),
        ("mode0", "mode-0 size 1"),
        ("tiles", "tile dimensions"),
        ("stateless", "stateful MMA atom"),
    ],
)
def test_scaled_gemm_rejects_invalid_fragments(invalid, diagnostic):
    with ir.Context(), ir.Location.unknown():
        module, _ = _build_gemm(3, invalid_scale=invalid)
        with pytest.raises(ir.MLIRError, match=diagnostic):
            module.operation.verify()


@pytest.mark.parametrize(
    "shift,opsel,folded",
    [
        (8, 0, 1),
        (16, 0, 2),
        (24, 0, 3),
        (8, 2, 3),
        (8, 3, None),
        (24, 1, None),
        (7, 0, None),
        (-8, 0, None),
        (32, 0, None),
    ],
)
def test_scale_byte_shift_uses_mfma_selector(shift, opsel, folded):
    atom = f"!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = {opsel}, opselB = {opsel}>>"
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(f"""
        func.func @test(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {{
          %amount = arith.constant {shift} : i32
          %as = arith.shrsi %sa, %amount : i32
          %bs = arith.shrui %sb, %amount : i32
          %atom = fly.make_mma_atom : {atom}
          %atom_a = fly.atom.set_value(%atom, "scale_a", %as) : ({atom}, i32) -> {atom}
          %atom_ab = fly.atom.set_value(%atom_a, "scale_b", %bs) : ({atom}, i32) -> {atom}
          %r = fly.mma_atom_call_ssa(%atom_ab, %a, %b, %c) : ({atom}, vector<8xi32>, vector<8xi32>, vector<4xf32>) -> vector<4xf32>
          return %r : vector<4xf32>
        }}""")
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        function = module.body.operations[0]
        call = next(op for op in _walk(function.operation) if op.name == MFMA)
        assert ir.IntegerAttr(call.attributes["opselA"]).value == (opsel if folded is None else folded)
        assert ir.IntegerAttr(call.attributes["opselB"]).value == (opsel if folded is None else folded)
        if folded is not None:
            assert call.operands[3] == function.body.blocks[0].arguments[3]
            assert call.operands[4] == function.body.blocks[0].arguments[4]
        else:
            assert call.operands[3].owner.name == "arith.shrsi"
            assert call.operands[4].owner.name == "arith.shrui"
