#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for flydsl.expr.math DSL wrappers.

Verifies that:
1. DSL wrappers override raw MLIR star-imports (dsl_loc_tracing + _to_raw).
2. Each wrapper generates the correct math dialect op in IR.
3. Wrappers accept DSL Numeric types (Float32, Int32) and auto-unwrap them.
4. fastmath= attribute propagates to the generated ops.
"""

import sys

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, func
from flydsl._mlir.dialects import math as _raw_math
from flydsl.expr import math as fly_math
from flydsl.expr.numeric import Boolean, Float32, Int32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_module(build_fn, arg_types=None):
    """Build an MLIR module with a function that calls build_fn(args...).

    *arg_types* is a list of callables ``() -> ir.Type`` (to defer type
    creation until a Context is live).  Pass ``None`` for a single f32 arg.

    Returns the IR text string.
    """
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            if arg_types is None:
                types = [ir.F32Type.get()]
            else:
                types = [t() if callable(t) else t for t in arg_types]
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get(types, [])
                f = func.FuncOp("test", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    args = list(f.entry_block.arguments)
                    build_fn(*args)
                    func.ReturnOp([])
            module.operation.verify()
            return str(module)


# ---------------------------------------------------------------------------
# 1. Wrapper identity — DSL wrappers override raw star-imports
# ---------------------------------------------------------------------------

_WRAPPED_NAMES = [
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "rsqrt",
    "cbrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "absf",
    "floor",
    "ceil",
    "round",
    "roundeven",
    "trunc",
    "absi",
    "ctlz",
    "cttz",
    "ctpop",
    "powf",
    "fpowi",
    "ipowi",
    "atan2",
    "copysign",
    "fma",
    "clampf",
    "isnan",
    "isinf",
    "isfinite",
    "isnormal",
    "sincos",
]


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("name", _WRAPPED_NAMES)
def test_wrapper_overrides_raw(name):
    """Our @dsl_loc_tracing wrapper must not be the same object as the raw MLIR binding."""
    ours = getattr(fly_math, name)
    raw = getattr(_raw_math, name)
    assert ours is not raw, f"fly_math.{name} is still the raw MLIR function"
    assert ours.__closure__ is not None, f"fly_math.{name} has no closure (not wrapped)"


# ---------------------------------------------------------------------------
# 2. Unary float ops — correct MLIR op in IR
# ---------------------------------------------------------------------------

_UNARY_FLOAT_OPS = [
    ("exp", "math.exp"),
    ("exp2", "math.exp2"),
    ("log", "math.log"),
    ("log2", "math.log2"),
    ("log10", "math.log10"),
    ("sqrt", "math.sqrt"),
    ("rsqrt", "math.rsqrt"),
    ("cbrt", "math.cbrt"),
    ("sin", "math.sin"),
    ("cos", "math.cos"),
    ("tan", "math.tan"),
    ("asin", "math.asin"),
    ("acos", "math.acos"),
    ("atan", "math.atan"),
    ("sinh", "math.sinh"),
    ("cosh", "math.cosh"),
    ("tanh", "math.tanh"),
    ("asinh", "math.asinh"),
    ("acosh", "math.acosh"),
    ("atanh", "math.atanh"),
    ("erf", "math.erf"),
    ("erfc", "math.erfc"),
    ("absf", "math.absf"),
    ("floor", "math.floor"),
    ("ceil", "math.ceil"),
    ("round", "math.round"),
    ("roundeven", "math.roundeven"),
    ("trunc", "math.trunc"),
]


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("fn_name,expected_op", _UNARY_FLOAT_OPS)
def test_unary_float_op(fn_name, expected_op):
    fn = getattr(fly_math, fn_name)

    def build(x):
        fn(x)

    ir_text = _build_module(build)
    assert expected_op in ir_text, f"{expected_op} not found in IR:\n{ir_text}"


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "type_fn,log_op",
    [
        (ir.F32Type.get, "rocdl.log %"),
        (ir.F16Type.get, "rocdl.log %"),
        (ir.BF16Type.get, "rocdl.log %"),
        (ir.F64Type.get, "math.log %"),
    ],
    ids=["f32", "f16", "bf16", "f64"],
)
def test_log1p_is_compensated(type_fn, log_op):
    """log1p emits log(w) + (x - (w - 1)) with w = 1 + x, not math.log1p (lowered to log(1 + x), 0 for tiny x);
    the log is the hardware log2 (rocdl.log) times ln 2, math.log for f64, and there is no division."""

    ir_text = _build_module(lambda x: fly_math.log1p(x), arg_types=[type_fn])
    assert "math.log1p" not in ir_text
    assert "arith.divf" not in ir_text
    for op in ("arith.addf", "arith.subf", "arith.cmpf olt", "arith.select", log_op):
        assert op in ir_text, f"{op} not found in IR:\n{ir_text}"


@pytest.mark.l0_backend_agnostic
def test_log1p_vector_logs_per_lane():
    """A vector<2x2xf32> log1p is four rocdl.log calls, reassembled into vector<2x2xf32>."""

    def build(x):
        y = fly_math.log1p(x)
        assert y.type == x.type

    ir_text = _build_module(build, arg_types=[lambda: ir.VectorType.get([2, 2], ir.F32Type.get())])
    assert ir_text.count("rocdl.log %") == 4


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "type_fn,callee",
    [
        (ir.F32Type.get, "__ocml_expm1_f32"),
        (ir.F64Type.get, "__ocml_expm1_f64"),
        (ir.F16Type.get, "__ocml_expm1_f16"),
        (ir.BF16Type.get, "__ocml_expm1_f32"),
    ],
    ids=["f32", "f64", "f16", "bf16"],
)
def test_expm1_calls_ocml(type_fn, callee):
    """expm1 calls OCML's expm1, not math.expm1 (lowered to exp(x) - 1, 0 for tiny x); bf16 goes through f32."""

    ir_text = _build_module(lambda x: fly_math.expm1(x), arg_types=[type_fn])
    assert "math.expm1" not in ir_text
    assert f"llvm.func @{callee}(" in ir_text
    assert f"llvm.call @{callee}" in ir_text


@pytest.mark.l0_backend_agnostic
def test_expm1_vector_calls_ocml_per_lane():
    """A vector<2x2xf32> expm1 is four calls to one declaration, reassembled into vector<2x2xf32>."""

    def build(x):
        y = fly_math.expm1(x)
        assert y.type == x.type
        fly_math.expm1(x)

    ir_text = _build_module(build, arg_types=[lambda: ir.VectorType.get([2, 2], ir.F32Type.get())])
    assert ir_text.count("llvm.func @__ocml_expm1_f32(") == 1
    assert ir_text.count("llvm.call @__ocml_expm1_f32") == 8


# ---------------------------------------------------------------------------
# 3. Unary integer ops
# ---------------------------------------------------------------------------

_UNARY_INT_OPS = [
    ("absi", "math.absi"),
    ("ctlz", "math.ctlz"),
    ("cttz", "math.cttz"),
    ("ctpop", "math.ctpop"),
]


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("fn_name,expected_op", _UNARY_INT_OPS)
def test_unary_int_op(fn_name, expected_op):
    fn = getattr(fly_math, fn_name)

    def build(xi):
        fn(xi)

    ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
    assert expected_op in ir_text, f"{expected_op} not found in IR:\n{ir_text}"


# ---------------------------------------------------------------------------
# 4. Binary ops
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_powf():
    def build(x):
        fly_math.powf(x, x)

    ir_text = _build_module(build)
    assert "math.powf" in ir_text


@pytest.mark.l0_backend_agnostic
def test_fpowi():
    def build(x, xi):
        fly_math.fpowi(x, xi)

    ir_text = _build_module(
        build,
        arg_types=[ir.F32Type.get, lambda: ir.IntegerType.get_signless(32)],
    )
    assert "math.fpowi" in ir_text


@pytest.mark.l0_backend_agnostic
def test_ipowi():
    def build(xi):
        fly_math.ipowi(xi, xi)

    ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
    assert "math.ipowi" in ir_text


@pytest.mark.l0_backend_agnostic
def test_atan2():
    def build(x):
        fly_math.atan2(x, x)

    ir_text = _build_module(build)
    assert "math.atan2" in ir_text


@pytest.mark.l0_backend_agnostic
def test_copysign():
    def build(x):
        fly_math.copysign(x, x)

    ir_text = _build_module(build)
    assert "math.copysign" in ir_text


# ---------------------------------------------------------------------------
# 5. Ternary ops
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_fma():
    def build(x):
        fly_math.fma(x, x, x)

    ir_text = _build_module(build)
    assert "math.fma" in ir_text


@pytest.mark.l0_backend_agnostic
def test_clampf():
    def build(x):
        lo = arith.ConstantOp(ir.F32Type.get(), 0.0).result
        hi = arith.ConstantOp(ir.F32Type.get(), 1.0).result
        fly_math.clampf(x, lo, hi)

    ir_text = _build_module(build)
    assert "math.clampf" in ir_text


# ---------------------------------------------------------------------------
# 6. Predicate ops (return i1)
# ---------------------------------------------------------------------------

_PREDICATE_OPS = [
    ("isnan", "math.isnan"),
    ("isinf", "math.isinf"),
    ("isfinite", "math.isfinite"),
    ("isnormal", "math.isnormal"),
]


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("fn_name,expected_op", _PREDICATE_OPS)
def test_predicate_op(fn_name, expected_op):
    fn = getattr(fly_math, fn_name)

    def build(x):
        fn(x)

    ir_text = _build_module(build)
    assert expected_op in ir_text


# ---------------------------------------------------------------------------
# 6b. Multi-result ops
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_sincos():
    """sincos returns two results (sin, cos)."""

    def build(x):
        results = fly_math.sincos(x)
        assert len(results) == 2

    ir_text = _build_module(build)
    assert "math.sincos" in ir_text


# ---------------------------------------------------------------------------
# 7. DSL Numeric type auto-unwrap (Float32, Int32)
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_numeric_float32_unwrap():
    """fly_math should accept Float32 DSL type, not just raw ir.Value."""

    def build(x_raw):
        x = Float32(x_raw)
        fly_math.exp(x)
        fly_math.sqrt(x)
        fly_math.fma(x, x, x)

    ir_text = _build_module(build)
    assert "math.exp" in ir_text
    assert "math.sqrt" in ir_text
    assert "math.fma" in ir_text


@pytest.mark.l0_backend_agnostic
def test_numeric_int32_unwrap():
    """fly_math should accept Int32 DSL type for integer ops."""

    def build(xi_raw):
        xi = Int32(xi_raw)
        fly_math.absi(xi)
        fly_math.ctlz(xi)

    ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
    assert "math.absi" in ir_text
    assert "math.ctlz" in ir_text


@pytest.mark.l0_backend_agnostic
def test_numeric_mixed_unwrap():
    """fpowi accepts Float32 base and Int32 exponent."""

    def build(x_raw, xi_raw):
        x = Float32(x_raw)
        xi = Int32(xi_raw)
        fly_math.fpowi(x, xi)

    ir_text = _build_module(
        build,
        arg_types=[ir.F32Type.get, lambda: ir.IntegerType.get_signless(32)],
    )
    assert "math.fpowi" in ir_text


# ---------------------------------------------------------------------------
# 8. fastmath attribute propagation
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_fastmath_propagates():
    """fastmath= kwarg should appear in the generated op attributes."""

    def build(x):
        fly_math.exp(x, fastmath="fast")
        fly_math.sqrt(x, fastmath="fast")

    ir_text = _build_module(build)
    assert "fastmath = #arith.fastmath<fast>" in ir_text or "fastmath" in ir_text


# ---------------------------------------------------------------------------
# 9. Vector type support
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_vector_type_ops():
    """fly_math should work on vector<4xf32> inputs."""

    def build(x):
        vtype = ir.VectorType.get([4], ir.F32Type.get())
        splat = arith.ConstantOp(
            vtype,
            ir.DenseElementsAttr.get_splat(vtype, ir.FloatAttr.get(ir.F32Type.get(), 1.0)),
        ).result
        fly_math.exp(splat)
        fly_math.sqrt(splat)
        fly_math.fma(splat, splat, splat)

    ir_text = _build_module(build)
    assert "vector<4xf32>" in ir_text
    assert "math.exp" in ir_text
    assert "math.sqrt" in ir_text
    assert "math.fma" in ir_text


# ---------------------------------------------------------------------------
# 10. Class invariance — Numeric type preservation
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_float32_class_invariance():
    """Unary float ops: Float32 in → Float32 out."""

    def build(x_raw):
        x = Float32(x_raw)
        y = fly_math.exp(x)
        assert isinstance(y, Float32), f"exp: expected Float32, got {type(y).__name__}"
        y = fly_math.sqrt(x)
        assert isinstance(y, Float32), f"sqrt: expected Float32, got {type(y).__name__}"
        y = fly_math.floor(x)
        assert isinstance(y, Float32), f"floor: expected Float32, got {type(y).__name__}"

    _build_module(build)


@pytest.mark.l0_backend_agnostic
def test_int32_class_invariance():
    """Unary int ops: Int32 in → Int32 out."""

    def build(xi_raw):
        xi = Int32(xi_raw)
        y = fly_math.absi(xi)
        assert isinstance(y, Int32), f"absi: expected Int32, got {type(y).__name__}"
        y = fly_math.ctlz(xi)
        assert isinstance(y, Int32), f"ctlz: expected Int32, got {type(y).__name__}"

    _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])


@pytest.mark.l0_backend_agnostic
def test_predicate_returns_boolean():
    """Predicate ops on Float32 → Boolean."""

    def build(x_raw):
        x = Float32(x_raw)
        y = fly_math.isnan(x)
        assert isinstance(y, Boolean), f"isnan: expected Boolean, got {type(y).__name__}"
        y = fly_math.isinf(x)
        assert isinstance(y, Boolean), f"isinf: expected Boolean, got {type(y).__name__}"

    _build_module(build)


@pytest.mark.l0_backend_agnostic
def test_binary_class_invariance():
    """Binary float ops: Float32 in → Float32 out."""

    def build(x_raw):
        x = Float32(x_raw)
        y = fly_math.powf(x, x)
        assert isinstance(y, Float32), f"powf: expected Float32, got {type(y).__name__}"
        y = fly_math.atan2(x, x)
        assert isinstance(y, Float32), f"atan2: expected Float32, got {type(y).__name__}"

    _build_module(build)


@pytest.mark.l0_backend_agnostic
def test_ternary_class_invariance():
    """Ternary float ops: Float32 in → Float32 out."""

    def build(x_raw):
        x = Float32(x_raw)
        y = fly_math.fma(x, x, x)
        assert isinstance(y, Float32), f"fma: expected Float32, got {type(y).__name__}"

    _build_module(build)


@pytest.mark.l0_backend_agnostic
def test_sincos_class_invariance():
    """sincos(Float32) → tuple of Float32."""

    def build(x_raw):
        x = Float32(x_raw)
        results = fly_math.sincos(x)
        assert isinstance(results, tuple), f"sincos: expected tuple, got {type(results).__name__}"
        assert len(results) == 2
        assert isinstance(results[0], Float32), f"sincos[0]: expected Float32, got {type(results[0]).__name__}"
        assert isinstance(results[1], Float32), f"sincos[1]: expected Float32, got {type(results[1]).__name__}"

    _build_module(build)


@pytest.mark.l0_backend_agnostic
def test_raw_value_passthrough():
    """Raw ir.Value input should NOT be wrapped in Numeric."""

    def build(x_raw):
        y = fly_math.exp(x_raw)
        assert not isinstance(y, Float32), f"raw input should not produce Float32, got {type(y).__name__}"

    _build_module(build)


# ---------------------------------------------------------------------------
# 11. End-to-end GPU tests
# ---------------------------------------------------------------------------

try:
    import torch as _torch

    _HAS_GPU = _torch.cuda.is_available()
except ImportError:
    _torch = None
    _HAS_GPU = False

_gpu_skip = pytest.mark.skipif(not _HAS_GPU, reason="CUDA/ROCm not available")


@_gpu_skip
@pytest.mark.l2_device
@pytest.mark.rocm_lower
class TestMathOpsGPU:
    """End-to-end GPU correctness for fly_math wrappers.

    Uses a single kernel that chains multiple math ops to verify that
    fly_math DSL wrappers compile, lower to GPU, and produce correct results.

    Pipeline: C = floor(sqrt(exp(abs(A))))

    This exercises absf, exp, sqrt, floor in a single JIT compilation.
    """

    def test_math_chain_gpu(self):
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.expr import math as _mops

        VEC_WIDTH = 4
        BLOCK_DIM = 256
        TILE_ELEMS = BLOCK_DIM * VEC_WIDTH
        N = TILE_ELEMS * 64

        @flyc.kernel
        def math_chain_kernel(
            A: fx.Tensor,
            C: fx.Tensor,
            block_dim: fx.Constexpr[int],
            vec_width: fx.Constexpr[int],
        ):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            tile_elems = block_dim * vec_width

            tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
            tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
            tA = fx.slice(tA, (None, bid))
            tC = fx.slice(tC, (None, bid))
            tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
            tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

            copy_bits = vec_width * 32
            copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

            rA = fx.make_rmem_tensor(vec_width, fx.Float32)
            rC = fx.make_rmem_tensor(vec_width, fx.Float32)

            fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)

            vA = fx.memref_load_vec(rA)
            # Chain: floor(sqrt(exp(abs(x))))
            v1 = _mops.absf(vA)
            v2 = _mops.exp(v1)
            v3 = _mops.sqrt(v2)
            v4 = _mops.floor(v3)
            fx.memref_store_vec(v4, rC)

            fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

        @flyc.jit
        def launch(
            A: fx.Tensor,
            C,
            n: fx.Int32,
            const_n: fx.Constexpr[int],
            block_dim: fx.Constexpr[int],
            vec_width: fx.Constexpr[int],
            stream: fx.Stream = fx.Stream(None),
        ):
            tile = block_dim * vec_width
            grid_x = (n + tile - 1) // tile
            math_chain_kernel(A, C, block_dim, vec_width).launch(
                grid=(grid_x, 1, 1),
                block=(block_dim, 1, 1),
                stream=stream,
            )

        # Use small input range to keep exp() in a reasonable range
        a_host = _torch.empty(N, dtype=_torch.float32).uniform_(-2.0, 2.0)
        a_dev = a_host.cuda()
        c_dev = _torch.empty_like(a_dev)

        tA = flyc.from_torch_tensor(a_dev).mark_layout_dynamic(
            leading_dim=0,
            divisibility=VEC_WIDTH,
        )

        stream = _torch.cuda.Stream()
        launch(tA, c_dev, N, N, BLOCK_DIM, VEC_WIDTH, stream=stream)
        _torch.cuda.synchronize()

        c_ref = _torch.floor(_torch.sqrt(_torch.exp(_torch.abs(a_host)))).cuda()
        assert _torch.allclose(
            c_dev, c_ref, atol=1e-5, rtol=1e-4
        ), f"fly_math chain GPU mismatch: max_diff={(_torch.abs(c_dev - c_ref)).max().item():.6e}"

    def test_math_trig_chain_gpu(self):
        """Chain: C = cos(sin(A))  — exercises trig math ops on GPU."""
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.expr import math as _mops

        VEC_WIDTH = 4
        BLOCK_DIM = 256
        TILE_ELEMS = BLOCK_DIM * VEC_WIDTH
        N = TILE_ELEMS * 64

        @flyc.kernel
        def trig_kernel(
            A: fx.Tensor,
            C: fx.Tensor,
            block_dim: fx.Constexpr[int],
            vec_width: fx.Constexpr[int],
        ):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            tile_elems = block_dim * vec_width

            tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
            tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
            tA = fx.slice(tA, (None, bid))
            tC = fx.slice(tC, (None, bid))
            tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
            tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

            copy_bits = vec_width * 32
            copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

            rA = fx.make_rmem_tensor(vec_width, fx.Float32)
            rC = fx.make_rmem_tensor(vec_width, fx.Float32)

            fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)

            vA = fx.memref_load_vec(rA)
            v1 = _mops.sin(vA)
            v2 = _mops.cos(v1)
            fx.memref_store_vec(v2, rC)

            fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

        @flyc.jit
        def launch(
            A: fx.Tensor,
            C,
            n: fx.Int32,
            const_n: fx.Constexpr[int],
            block_dim: fx.Constexpr[int],
            vec_width: fx.Constexpr[int],
            stream: fx.Stream = fx.Stream(None),
        ):
            tile = block_dim * vec_width
            grid_x = (n + tile - 1) // tile
            trig_kernel(A, C, block_dim, vec_width).launch(
                grid=(grid_x, 1, 1),
                block=(block_dim, 1, 1),
                stream=stream,
            )

        a_host = _torch.empty(N, dtype=_torch.float32).uniform_(-3.14, 3.14)
        a_dev = a_host.cuda()
        c_dev = _torch.empty_like(a_dev)

        tA = flyc.from_torch_tensor(a_dev).mark_layout_dynamic(
            leading_dim=0,
            divisibility=VEC_WIDTH,
        )

        stream = _torch.cuda.Stream()
        launch(tA, c_dev, N, N, BLOCK_DIM, VEC_WIDTH, stream=stream)
        _torch.cuda.synchronize()

        c_ref = _torch.cos(_torch.sin(a_host)).cuda()
        assert _torch.allclose(
            c_dev, c_ref, atol=1e-4, rtol=1e-3
        ), f"trig chain GPU mismatch: max_diff={(_torch.abs(c_dev - c_ref)).max().item():.6e}"


@_gpu_skip
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("name", ["log1p", "expm1"])
def test_log1p_expm1_accurate_gpu(name):
    """log1p and expm1 within a few f32 ulps of torch (float64) from 1e-30 to 10 in magnitude, and exact at the
    edges. ``log(1 + x)`` / ``exp(x) - 1`` return 0 for |x| below 6e-8."""
    import math

    import flydsl.compiler as flyc
    import flydsl.expr as fx

    VEC, BLOCK = 4, 64
    fn = getattr(fly_math, name)

    @flyc.kernel
    def kernel(A: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        tA = fx.logical_divide(A, fx.make_layout(VEC, 1))
        tC = fx.logical_divide(C, fx.make_layout(VEC, 1))
        atom = fx.make_copy_atom(fx.UniversalCopy(VEC * 32), fx.Float32)
        rA = fx.make_rmem_tensor(VEC, fx.Float32)
        rC = fx.make_rmem_tensor(VEC, fx.Float32)
        fx.copy_atom_call(atom, fx.slice(tA, (None, tid)), rA)
        fx.memref_store_vec(fn(fx.memref_load_vec(rA)), rC)
        fx.copy_atom_call(atom, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def launch(A: fx.Tensor, C, stream: fx.Stream = fx.Stream(None)):
        kernel(A, C).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    n = VEC * BLOCK
    edges = [math.inf, -math.inf, math.nan, 0.0, -0.0, -1.0, -2.0, 100.0, -100.0, 3e38]
    tiny = n // 2 - len(edges)
    x = _torch.cat(
        [
            _torch.logspace(-30, 1, tiny, dtype=_torch.float64),
            -_torch.logspace(-30, -0.01, n // 2, dtype=_torch.float64),
            _torch.tensor(edges, dtype=_torch.float64),
        ]
    ).float()
    c = _torch.empty_like(x, device="cuda")
    launch(x.cuda(), c)
    _torch.cuda.synchronize()

    got, want = c.cpu().double(), getattr(_torch, name)(x.double()).float().double()
    finite = slice(0, n - len(edges))
    rel = ((got[finite] - want[finite]) / want[finite]).abs()
    assert rel.max().item() <= 1e-6, f"{name}: max rel err {rel.max().item():.3e} at x={x[finite][rel.argmax()].item()}"
    _torch.testing.assert_close(got[finite.stop :], want[finite.stop :], rtol=1e-6, atol=0, equal_nan=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
