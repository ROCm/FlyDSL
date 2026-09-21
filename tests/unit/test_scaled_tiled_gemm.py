# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Variadic tiled GEMM operands, scale indexing, and hardware byte selectors."""

import itertools

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, func, gpu
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


def _build_gemm(
    rank,
    traversal=None,
    scale_mode="tensor",
    packed=False,
    *,
    call=fx.gemm,
    atom_callback=None,
):
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
                sa_layout = fx.make_layout(sa_shape, (0,) * rank) if scale_mode == "broadcast_a" else sa_shape
                sb_shape = (1,) + b_shape[1:]
                sb_layout = fx.make_layout(sb_shape, (0,) * rank) if scale_mode == "broadcast_b" else sb_shape
                sa = fx.make_rmem_tensor(sa_layout, fx.Int32)
                sb = fx.make_rmem_tensor(sb_layout, fx.Int32)
                for kt, mt in itertools.product(range(k), range(m)):
                    sa[(0, mt, kt)[:rank]] = fx.Int32(117 if scale_mode == "broadcast_a" else 117 + mt + m * kt)
                for kt, nt in itertools.product(range(k), range(n)):
                    sb[(0, nt, kt)[:rank]] = fx.Int32(120 if scale_mode == "broadcast_b" else 120 + nt + n * kt)
                kwargs = {}
                if atom_callback is not None:
                    kwargs["atom_callback"] = atom_callback
                if traversal == "layout":
                    kwargs["traversal_layout"] = fx.make_layout((m, n, k), (n * k, k, 1))
                elif traversal == "nested_layout":
                    kwargs["traversal_layout"] = fx.make_layout(((m, n), k), ((n * k, k), 1))
                elif traversal is not None:
                    kwargs["traversal_order"] = getattr(fx.GemmTraversalOrder, traversal)
                a_group, b_group = [a, sa], [b, sb]
                if scale_mode in ("a_only", "none", "state"):
                    b_group = b
                if scale_mode in ("b_only", "none", "state"):
                    a_group = a
                if scale_mode == "state":
                    kwargs.update(scale_a=117, scale_b=fx.Int32(120))
                elif scale_mode == "override_state":
                    kwargs.update(scale_a=110, scale_b=fx.Int32(111))
                if scale_mode == "tuple":
                    a_group, b_group = tuple(a_group), tuple(b_group)
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


@pytest.mark.parametrize("call", [fx.gemm, fx.mma_atom_call])
@pytest.mark.parametrize("mode", ["tensor", "state", "override_state"])
@pytest.mark.parametrize("promote", [False, True])
def test_atom_operand_groups_and_scalar_state(call, mode, promote):
    with ir.Context(), ir.Location.unknown():
        module, _ = _build_gemm(1, scale_mode=mode, packed=True, call=call)
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


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("mode", ["state", "override_state"])
def test_tiled_gemm_forwards_atom_state(rank, mode):
    with ir.Context(), ir.Location.unknown():
        module, (m, n, k) = _build_gemm(rank, scale_mode=mode)
        PassManager.parse(PIPELINE).run(module.operation)
        scales = [
            tuple(ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:])
            for op in _walk(module.operation)
            if op.name == MFMA
        ]
        expected = (
            [(117, 120)] * (m * n * k)
            if mode == "state"
            else [
                (117 + mt + m * kt, 120 + nt + n * kt) for mt, nt, kt in itertools.product(range(m), range(n), range(k))
            ]
        )
        assert sorted(scales) == sorted(expected)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("setter", ["field", "dict", "primitive"])
@pytest.mark.parametrize("numeric", [False, True])
def test_mma_set_value_preserves_type_and_runtime_atom(tiled, setter, numeric):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
            mma = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 4)), (32, 32, 128)) if tiled else atom
            data_type = ir.VectorType.get([8], fx.Int32.ir_type)
            acc_type = ir.VectorType.get([4], fx.Float32.ir_type)
            function = func.FuncOp("update", ([mma.type, fx.Int32.ir_type, data_type, acc_type], [acc_type]))
            with ir.InsertionPoint(function.add_entry_block()):
                original, scale, data, acc = function.arguments
                value = fx.Int32(scale) if numeric else scale
                if setter == "field":
                    updated = original.set_value("scale_a", value)
                elif setter == "dict":
                    updated = original.set_value({"scale_a": value})
                else:
                    updated = fx.atom_set_value(original, "scale_a", value)
                assert updated.type == original.type
                assert updated.owner.name == "fly.atom.set_value"
                assert updated.owner.operands[0] == original
                updated_atom = fly.get_mma_atom(updated) if tiled else updated
                result = fly.mma_atom_call_ssa([acc_type], updated_atom, [data], [data], acc)
                func.ReturnOp([result])
        assert module.operation.verify()
        PassManager.parse("builtin.module(canonicalize,convert-fly-to-rocdl,canonicalize)").run(module.operation)
        function = module.body.operations[0]
        call = next(op for op in _walk(function.operation) if op.name == MFMA)
        assert call.operands[3] == function.arguments[1]
        assert call.operands[4].owner.name == "llvm.extractvalue"
        assert call.operands[4].owner.operands[0] == function.arguments[0]


@pytest.mark.parametrize("call", [fly.mma_atom_call, fly.mma_atom_call_ssa])
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


@pytest.mark.parametrize("call", [fly.gemm, fly.mma_atom_call])
def test_low_level_memref_builders_accept_singleton_groups(call):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
            data = fx.make_rmem_tensor(8, fx.Int32)
            acc = fx.make_rmem_tensor(4, fx.Float32)
            op = call(atom, acc, [data], [data], acc)
        assert module.operation.verify()
        assert "[" not in str(op).split(":", 1)[0]


def test_low_level_ssa_builder_accepts_singleton_groups():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
            data = fx.Vector.filled(8, 1, fx.Int32).ir_value()
            acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
            result = fly.mma_atom_call_ssa([acc.type], atom, [data], [data], acc)
        assert module.operation.verify()
        assert "[" not in str(result.owner).split(":", 1)[0]


def test_textual_ir_accepts_legacy_single_operand_syntax():
    atom = "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>"
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(f"""
        func.func @test(%a: vector<8xi8>, %b: vector<8xi8>, %c: vector<4xf32>) -> vector<4xf32> {{
          %atom = fly.make_mma_atom : {atom}
          %result = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : ({atom}, vector<8xi8>, vector<8xi8>, vector<4xf32>) -> vector<4xf32>
          return %result : vector<4xf32>
        }}""")
        assert module.operation.verify()
        text = str(module)
        assert "fly.mma_atom_call_ssa(%0, %arg0, %arg1, %arg2)" in text


def test_textual_ir_prints_multi_operand_groups_with_brackets():
    atom = "!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>"
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(f"""
        func.func @test(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {{
          %atom = fly.make_mma_atom : {atom}
          %result = fly.mma_atom_call_ssa(%atom, [%a, %sa], [%b, %sb], %c) : ({atom}, vector<8xi32>, i32, vector<8xi32>, i32, vector<4xf32>) -> vector<4xf32>
          return %result : vector<4xf32>
        }}""")
        assert module.operation.verify()
        text = str(module)
        assert "fly.mma_atom_call_ssa(%0, [%arg0, %arg3], [%arg1, %arg4], %arg2)" in text


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
                assert value.owner.name == ("llvm.load" if representation == "memref" else "llvm.bitcast")
                assert value.owner.operands[0] == argument


@pytest.mark.parametrize("scale_source", ["operand", "state"])
@pytest.mark.parametrize("shift,opsel", [(8, 0), (16, 1), (24, 0), (8, 3)])
def test_scale_byte_shift_preserves_explicit_selector(shift, opsel, scale_source):
    atom = f"!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = {opsel}, opselB = {opsel}>>"
    call = (
        f"fly.mma_atom_call_ssa(%atom, [%a, %as], [%b, %bs], %c) : ({atom}, vector<8xi32>, i32, vector<8xi32>, i32, vector<4xf32>) -> vector<4xf32>"
        if scale_source == "operand"
        else f"fly.mma_atom_call_ssa(%atom_ab, [%a], [%b], %c) : ({atom}, vector<8xi32>, vector<8xi32>, vector<4xf32>) -> vector<4xf32>"
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(f"""
        func.func @test(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {{
          %amount = arith.constant {shift} : i32
          %as = arith.shrsi %sa, %amount : i32
          %bs = arith.shrui %sb, %amount : i32
          %atom = fly.make_mma_atom : {atom}
          %atom_a = fly.atom.set_value(%atom, "scale_a", %as) : ({atom}, i32) -> {atom}
          %atom_ab = fly.atom.set_value(%atom_a, "scale_b", %bs) : ({atom}, i32) -> {atom}
          %r = {call}
          return %r : vector<4xf32>
        }}""")
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        function = module.body.operations[0]
        call = next(op for op in _walk(function.operation) if op.name == MFMA)
        assert ir.IntegerAttr(call.attributes["opselA"]).value == opsel
        assert ir.IntegerAttr(call.attributes["opselB"]).value == opsel
        assert call.operands[3].owner.name == "arith.shrsi"
        assert call.operands[4].owner.name == "arith.shrui"


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize(
    "traversal", [None, "layout", "nested_layout"] + [order.name for order in fx.GemmTraversalOrder]
)
def test_atom_callback_matches_compiler_traversal_and_accumulation(rank, traversal):
    with ir.Context(), ir.Location.unknown():
        reference, (m, n, k) = _build_gemm(rank, traversal)
        visits = []

        def callback(atom, mnk):
            assert isinstance(atom.type, fx.MmaAtomType)
            assert all(isinstance(index, int) for index in mnk)
            visits.append(mnk)
            return fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN, opsel_a=mnk[0] % 2, opsel_b=mnk[1] % 4)
            )

        expanded, _ = _build_gemm(rank, traversal, atom_callback=callback)
        assert not any(op.name == "fly.gemm" for op in _walk(expanded.operation))
        assert len(visits) == m * n * k
        assert set(visits) == set(itertools.product(range(m), range(n), range(k)))

        signatures = []
        for module in (reference, expanded):
            assert module.operation.verify()
            PassManager.parse(PIPELINE).run(module.operation)
            calls = [op for op in _walk(module.operation) if op.name == MFMA]
            last = {}
            signature = []
            for op in calls:
                sa, sb = [ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:]]
                kt, mt = divmod(sa - 117, m)
                kb, nt = divmod(sb - 120, n)
                assert kt == kb
                signature.append((mt, nt, kt))
                if (mt, nt) in last:
                    assert op.operands[2] == last[mt, nt]
                else:
                    assert op.operands[2].owner.name == "arith.constant"
                    assert set(ir.DenseFPElementsAttr(op.operands[2].owner.attributes["value"])) == {2.0}
                last[mt, nt] = op.results[0]
                if module is expanded:
                    assert ir.IntegerAttr(op.attributes["opselA"]).value == mt % 2
                    assert ir.IntegerAttr(op.attributes["opselB"]).value == nt % 4
            signatures.append(signature)
        assert signatures[0] == signatures[1] == visits


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("mode", ["state", "override_state"])
@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("promote", [False, True])
def test_gemm_preserves_callback_state(rank, mode, callback, promote):
    with ir.Context(), ir.Location.unknown():
        module, (m, n, k) = _build_gemm(
            rank, scale_mode=mode, packed=True, atom_callback=(lambda atom, mnk: atom) if callback else None
        )
        pipeline = (
            PIPELINE
            if promote
            else "builtin.module(fly-layout-lowering,canonicalize,convert-fly-to-rocdl,canonicalize)"
        )
        PassManager.parse(pipeline).run(module.operation)
        calls = [op for op in _walk(module.operation) if op.name == MFMA]
        assert len(calls) == m * n * k
        if mode != "override_state":
            assert all(
                [ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:]] == [117, 120]
                for op in calls
            )
        elif promote:
            assert sorted(
                tuple(ir.IntegerAttr(v.owner.attributes["value"]).value for v in op.operands[3:]) for op in calls
            ) == sorted(
                (117 + mt + m * kt, 120 + nt + n * kt) for mt, nt, kt in itertools.product(range(m), range(n), range(k))
            )


def test_atom_callback_requires_atom_result():
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(TypeError, match="atom_callback must return an MmaAtom"):
            _build_gemm(1, atom_callback=lambda atom, mnk: None)


@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize(
    "operand,shape,diagnostic",
    [
        ("a", (32, 2), "rank-1 fragments or rank-2/3"),
        ("d", (4, 2), "rank-1 fragments or rank-2/3"),
        ("a", (32, 4, 2), "M/N tile dimensions must match"),
        ("b", (32, 4, 2), "M/N tile dimensions must match"),
        ("c", (4, 4, 3), "M/N tile dimensions must match"),
        ("c", (4, 2, 4), "M/N tile dimensions must match"),
        ("b", (32, 3, 4), "K tile dimensions must match"),
    ],
)
def test_gemm_validates_shapes_before_callback_dispatch(callback, operand, shape, diagnostic):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        visits = []

        def atom_callback(atom, mnk):
            visits.append(mnk)
            return atom

        with ir.InsertionPoint(module.body):
            atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
            shapes = dict(d=(4, 2, 3), a=(32, 2, 2), b=(32, 3, 2), c=(4, 2, 3))
            shapes[operand] = shape
            tensors = {
                key: fx.make_rmem_tensor(value, fx.Float8E4M3FN if key in ("a", "b") else fx.Float32)
                for key, value in shapes.items()
            }
            with pytest.raises(ValueError, match=diagnostic):
                fx.gemm(atom, **tensors, atom_callback=atom_callback if callback else None)
        assert not visits
        assert not any(op.name in ("fly.gemm", "fly.mma_atom_call") for op in _walk(module.operation))


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("preshuffled", [False, True])
def test_mxfp8_callback_kernel_compiles(monkeypatch, default_device, preshuffled):
    import torch

    import flydsl.compiler as flyc
    from kernels.gemm.mxfp8_gemm_8wave import compile_mxfp8_gemm_8w

    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    with torch.device(default_device):
        launch = compile_mxfp8_gemm_8w(K=256, b_preshuffled=preshuffled)
        a = torch.empty(256 * 256, dtype=torch.int8, device="cpu")
        b = torch.empty_like(a, device="cpu")
        c = torch.empty(256 * 256, dtype=torch.bfloat16, device="cpu")
        sa = torch.empty(256 * 8, dtype=torch.uint8, device="cpu")
        sb = torch.empty_like(sa, device="cpu")
        flyc.compile(launch, a, b, c, sa, sb, 256, 256, fx.Stream(None))
        assert launch._last_compiled is not None
