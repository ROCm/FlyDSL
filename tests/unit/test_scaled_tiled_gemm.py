# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Variadic tiled GEMM operands, scale indexing, and hardware byte selectors."""

import itertools

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, gpu
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


def _build_gemm(rank, traversal=None, invalid_scale=None, scale_mode="tensor", packed=False):
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
                v = 8 if packed else 32
                a_shape = (v,) if rank == 1 else ((v, m) if rank == 2 else (v, m, k))
                b_shape = (v,) if rank == 1 else ((v, n) if rank == 2 else (v, n, k))
                dtype = fx.Int32 if packed else fx.Float8E4M3FN
                a = fx.make_rmem_tensor(v if rank == 1 else a_shape, dtype)
                b = fx.make_rmem_tensor(v if rank == 1 else b_shape, dtype)
                c = fx.make_rmem_tensor(4 if rank == 1 else (4, m, n), fx.Float32)
                d = fx.make_fragment_like(c)
                a.fill(1 if packed else 1.0)
                b.fill(1 if packed else 1.0)
                c.fill(2.0)
                sa_shape = (1,) + a_shape[1:]
                if invalid_scale == "rank":
                    sa_shape = (1,)
                elif invalid_scale == "mode0":
                    sa_shape = (2,) + a_shape[1:]
                elif invalid_scale == "tiles":
                    sa_shape = (1, m + 1) + a_shape[2:]
                sa_layout = fx.make_layout(sa_shape, (0,) * rank) if scale_mode == "broadcast_a" else sa_shape
                sb_shape = (1,) + b_shape[1:]
                sb_layout = fx.make_layout(sb_shape, (0,) * rank) if scale_mode == "broadcast_b" else sb_shape
                sa = fx.make_rmem_tensor(sa_layout, fx.Float32 if invalid_scale == "dtype" else fx.Int32)
                sb = fx.make_rmem_tensor(sb_layout, fx.Int32)
                if invalid_scale is None:
                    for kt, mt in itertools.product(range(k), range(m)):
                        sa[(0, mt, kt)[:rank]] = fx.Int32(117 if scale_mode == "broadcast_a" else 117 + mt + m * kt)
                    for kt, nt in itertools.product(range(k), range(n)):
                        sb[(0, nt, kt)[:rank]] = fx.Int32(120 if scale_mode == "broadcast_b" else 120 + nt + n * kt)
                kwargs = {}
                if traversal == "layout":
                    kwargs["traversal_layout"] = fx.make_layout((m, n, k), (n * k, k, 1))
                elif traversal is not None:
                    kwargs["traversal_order"] = getattr(fx.GemmTraversalOrder, traversal)
                a_group, b_group = [a, sa], [b, sb]
                if scale_mode in ("a_only", "none", "state"):
                    b_group = b
                if scale_mode in ("b_only", "none", "state"):
                    a_group = a
                if scale_mode == "state":
                    kwargs.update(scale_a=117, scale_b=fx.Int32(120))
                if scale_mode == "tuple":
                    a_group, b_group = tuple(a_group), tuple(b_group)
                for operand, group, shape, tiles in (("a", a_group, a_shape, m), ("b", b_group, b_shape, n)):
                    if scale_mode in (f"triple_{operand}", "triple_both"):
                        metadata = fx.make_rmem_tensor(2 if rank == 1 else (2,) + shape[1:], fx.Int16)
                        for kt, tile, element in itertools.product(range(k), range(tiles), range(2)):
                            metadata[(element, tile, kt)[:rank]] = fx.Int16(200 + element + 2 * (tile + tiles * kt))
                        group.append(metadata)
                call = fx.mma_atom_call if scale_mode == "direct" else fx.gemm
                call(mma, d, a_group, b_group, c, **kwargs)
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


@pytest.mark.parametrize("mode", ["a_only", "b_only", "broadcast_a", "broadcast_b", "tuple", "none"])
def test_tiled_gemm_broadcast_and_optional_scales(mode):
    with ir.Context(), ir.Location.unknown():
        module, (m, n, _) = _build_gemm(2, scale_mode=mode)
        PassManager.parse(PIPELINE).run(module.operation)
        scales = [
            tuple(ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:])
            for op in _walk(module.operation)
            if op.name == MFMA
        ]
        expected_a = (
            [0] * m
            if mode in ("b_only", "none")
            else ([117] * m if mode == "broadcast_a" else list(range(117, 117 + m)))
        )
        expected_b = (
            [0] * n
            if mode in ("a_only", "none")
            else ([120] * n if mode == "broadcast_b" else list(range(120, 120 + n)))
        )
        assert sorted(scales) == sorted(itertools.product(expected_a, expected_b))


@pytest.mark.parametrize(
    "invalid,diagnostic",
    [
        ("dtype", "i32.*elements"),
        ("rank", "operand rank"),
        ("mode0", "mode-0 size 1"),
        ("tiles", "tile dimensions"),
        ("stateless", "does not support auxiliary operands"),
    ],
)
def test_scaled_gemm_rejects_invalid_fragments(invalid, diagnostic):
    with ir.Context(), ir.Location.unknown():
        module, _ = _build_gemm(3, invalid_scale=invalid)
        with pytest.raises(ir.MLIRError, match=diagnostic):
            module.operation.verify()
            PassManager.parse(PIPELINE).run(module.operation)


@pytest.mark.parametrize("mode", ["tensor", "direct", "state"])
@pytest.mark.parametrize("promote", [False, True])
def test_atom_operand_groups_and_legacy_scalar_state(mode, promote):
    with ir.Context(), ir.Location.unknown():
        module, _ = _build_gemm(1, scale_mode=mode, packed=True)
        pipeline = (
            PIPELINE
            if promote
            else "builtin.module(fly-layout-lowering,canonicalize,convert-fly-to-rocdl,canonicalize)"
        )
        PassManager.parse(pipeline).run(module.operation)
        calls = [op for op in _walk(module.operation) if op.name == MFMA]
        assert len(calls) == 1
        if promote or mode == "state":
            assert [ir.IntegerAttr(v.owner.attributes["value"]).value for v in calls[0].operands[3:]] == [117, 120]
        else:
            assert all(value.owner.name == "llvm.load" for value in calls[0].operands[3:])


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("mode", ["triple_a", "triple_b", "triple_both"])
def test_three_tensor_operand_groups_survive_expansion_and_ssa(rank, mode):
    with ir.Context(), ir.Location.unknown():
        module, (m, n, k) = _build_gemm(rank, scale_mode=mode)
        # Check the public builder and textual IR preserve the group boundaries.
        module = ir.Module.parse(str(module))
        assert module.operation.verify()
        PassManager.parse(PIPELINE.replace(",convert-fly-to-rocdl,canonicalize", "")).run(module.operation)
        calls = [op.opview for op in _walk(module.operation) if op.name == "fly.mma_atom_call_ssa"]
        assert len(calls) == m * n * k
        assert not any(op.name in ("fly.gemm", "scf.for", "scf.while") for op in _walk(module.operation))
        for call in calls:
            for operand, group, tiles, base in (("a", call.a, m, 117), ("b", call.b, n, 120)):
                has_metadata = mode in (f"triple_{operand}", "triple_both")
                assert len(group) == (3 if has_metadata else 2)
                if has_metadata:
                    scale = ir.IntegerAttr(group[1].owner.attributes["value"]).value
                    kt, tile = divmod(scale - base, tiles)
                    metadata = list(ir.DenseIntElementsAttr(group[2].owner.attributes["value"]))
                    assert metadata == [200 + element + 2 * (tile + tiles * kt) for element in range(2)]
        # The generic path carries three inputs; this particular hardware atom
        # consumes only data and scales and must diagnose unsupported metadata.
        with pytest.raises(ir.MLIRError, match="scaled MMA expects"):
            PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)


@pytest.mark.parametrize("call", [fx.gemm, fx.mma_atom_call])
@pytest.mark.parametrize("operand", ["a", "b"])
@pytest.mark.parametrize("invalid,error", [(None, TypeError), ([], ValueError), ([1], TypeError), ((), ValueError)])
def test_mma_operand_group_validation(call, operand, invalid, error):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
            tensor = fx.make_rmem_tensor(1, fx.Int32)
            args = dict(a=tensor, b=tensor)
            args[operand] = invalid
            with pytest.raises(error, match=f"'{operand}'"):
                call(atom, tensor, c=tensor, **args)


@pytest.mark.parametrize("call", [fly.gemm, fly.mma_atom_call, fly.mma_atom_call_ssa])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_ir_rejects_empty_operand_groups(call, operand):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
            tensor = fx.make_rmem_tensor(1, fx.Int32)
            args = dict(a=[tensor], b=[tensor])
            args[operand] = []
            if call == fly.mma_atom_call_ssa:
                op = call([], atom, c=tensor, d=tensor, **args)
            else:
                op = call(atom, tensor, c=tensor, **args)
            with pytest.raises(ir.MLIRError, match="at least one tensor"):
                op.operation.verify()


@pytest.mark.parametrize("layout", ["(1,?):(1,1)", "(1,2):(1,?)"])
def test_gemm_rejects_dynamic_auxiliary_layouts(layout):
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(ir.MLIRError, match="auxiliary operands must have static layouts"):
            module = ir.Module.parse(
                f"""
            !atom = !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>
            !data = !fly.memref<i32, register, (8,2):(1,8)>
            !acc = !fly.memref<f32, register, (4,2,2):(1,4,8)>
            !scale = !fly.memref<i32, register, {layout}>
            func.func @test(%atom: !atom, %a: !data, %b: !data, %c: !acc, %s: !scale) {{
              fly.gemm(%atom, %c, [%a, %s], [%b], %c) : (!atom, !acc, !data, !scale, !data, !acc) -> ()
              return
            }}""",
            )
            module.operation.verify()


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("representation", ["scalar", "vector", "memref"])
def test_gfx1250_scale_operand_types(block_size, representation):
    scale_type = "i64" if block_size == 16 else "i32"
    atom = f"!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 0, reuseA = false, reuseB = false, blockSize = {block_size}>>"
    if representation == "memref":
        data = "!fly.memref<i32, register, 16:1>"
        acc = "!fly.memref<f32, register, 8:1>"
        scale = f"!fly.memref<{scale_type}, register, 1:1>"
        call = f"fly.mma_atom_call(%atom, %c, [%a, %sa], [%b, %sb], %c) : ({atom}, {acc}, {data}, {scale}, {data}, {scale}, {acc}) -> ()"
    else:
        data, acc = "vector<16xi32>", "vector<8xf32>"
        scale = f"vector<1x{scale_type}>" if representation == "vector" else scale_type
        call = f"%r = fly.mma_atom_call_ssa(%atom, [%a, %sa], [%b, %sb], %c) : ({atom}, {data}, {scale}, {data}, {scale}, {acc}) -> {acc}"
    result = "" if representation == "memref" else f"-> {acc}"
    ret = "return" if representation == "memref" else f"return %r : {acc}"
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(f"""
        func.func @test(%a: {data}, %b: {data}, %c: {acc}, %sa: {scale}, %sb: {scale}) {result} {{
          %atom = fly.make_mma_atom : {atom}
          {call}
          {ret}
        }}""")
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        name = f"rocdl.wmma.scale{'16' if block_size == 16 else ''}.f32.16x16x128.f8f6f4"
        call = next(op for op in _walk(module.operation) if op.name == name)
        function = module.body.operations[0]
        for value, argument in zip(call.operands[3:], function.body.blocks[0].arguments[3:]):
            assert str(value.type) == scale_type
            if representation == "scalar":
                assert value == argument
            else:
                assert value.owner.name == ("llvm.load" if representation == "memref" else "vector.extract")
                assert value.owner.operands[0] == argument


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
          %r = fly.mma_atom_call_ssa(%atom_ab, [%a], [%b], %c) : ({atom}, vector<8xi32>, vector<8xi32>, vector<4xf32>) -> vector<4xf32>
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
