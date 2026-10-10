#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""AOT compile + C object export: ``flyc.compile(...).export_to_c(...)``.

Compile-tier tests check the ABI model, symbol namespacing and file publishing.
Device-tier tests link exported objects into shared libraries with the system
HIP compiler, load them with ``ctypes`` and launch the kernels through the
packed C ABI.
"""

import ctypes
import gc
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir._mlir_libs._mlirDialectsFly import _emit_host_object
from flydsl._mlir.dialects import gpu, llvm
from flydsl.compiler import _aot as aot
from flydsl.compiler.backends import _get_backend_class
from flydsl.runtime._libraries import _find_aot_runtime_archive
from flydsl.utils._elf import _defined_global_symbols, _dynamic_info

try:
    import torch
except ImportError:
    torch = None

HAS_GPU = torch is not None and torch.cuda.is_available()
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
HIPCC = shutil.which("hipcc")
if HIPCC is None:
    adjacent_hipcc = Path(sys.executable).parent / "hipcc"
    HIPCC = str(adjacent_hipcc) if adjacent_hipcc.is_file() else None
ROCM_PATH = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
ROCM_CLANG = ROCM_PATH / "llvm" / "bin" / "clang"

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="requires a ROCm GPU")
needs_cc = pytest.mark.skipif(CC is None, reason="requires a C compiler")


# Launchers


@flyc.kernel
def _axpy_kernel(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        Y[idx] = alpha * X[idx] + Y[idx]


@flyc.jit
def axpy(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32, stream: fx.Stream = fx.Stream(None)):
    _axpy_kernel(X, Y, n, alpha).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.kernel
def _add_one_kernel(X: fx.Pointer, n: fx.Int32):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        X[idx] = X[idx] + 1.0


@flyc.kernel
def _scale_kernel(X: fx.Pointer, n: fx.Int32, factor: fx.Constexpr[float]):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        X[idx] = X[idx] * factor


@flyc.jit
def add_one_then_scale(X: fx.Pointer, n: fx.Int32, factor: fx.Constexpr[float] = 2.0):
    """Two kernel launches, implicit stream."""
    grid = ((n + 63) // 64, 1, 1)
    _add_one_kernel(X, n).launch(grid=grid, block=(64, 1, 1))
    _scale_kernel(X, n, factor).launch(grid=grid, block=(64, 1, 1))


@flyc.kernel
def _div_kernel(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        X[idx] = X[idx] / Y[idx]


@flyc.kernel
def _tensor_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, tiled_copy: fx.TiledCopy):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y
    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
    TileMN = tiled_copy.tile_mn
    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]
    thr_copy = tiled_copy.get_slice(tid)
    thr_gA = thr_copy.partition_S(gA)
    thr_gB = thr_copy.partition_S(gB)
    thr_gC = thr_copy.partition_D(gC)
    thr_cC = thr_copy.partition_S(cC)
    thr_rA = fx.make_fragment_like(thr_gA)
    thr_rB = fx.make_fragment_like(thr_gB)
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)
    for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
        thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))
    copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)
    fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)
    thr_rC.store(thr_rA.load() + thr_rB.load())
    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)


@flyc.jit
def tensor_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((8, 16), order=(1, 0)),
        fx.make_ordered_layout((1, 1), order=(0, 1)),
    )
    tile_m, tile_n = tiled_copy.tile_mn.unpack()
    M, N = A.shape.unpack()
    grid = ((M + tile_m - 1) // tile_m, (N + tile_n - 1) // tile_n, 1)
    _tensor_add_kernel(A, B, C, tiled_copy).launch(grid=grid, block=(128, 1, 1), stream=stream)


_SCALAR_TYPES = [
    fx.Int8,
    fx.Int16,
    fx.Int32,
    fx.Int64,
    fx.Uint8,
    fx.Uint16,
    fx.Uint32,
    fx.Uint64,
    fx.Boolean,
    fx.Float16,
    fx.BFloat16,
    fx.Float32,
    fx.Float64,
]


@flyc.kernel
def _scalar_echo_kernel(
    out: fx.Pointer,
    a: fx.Int8,
    b: fx.Int16,
    c: fx.Int32,
    d: fx.Int64,
    e: fx.Uint8,
    f: fx.Uint16,
    g: fx.Uint32,
    h: fx.Uint64,
    i: fx.Boolean,
    j: fx.Float16,
    k: fx.BFloat16,
    m: fx.Float32,
    p: fx.Float64,
):
    if fx.thread_idx.x == 0:
        out[0] = fx.Float32(a)
        out[1] = fx.Float32(b)
        out[2] = fx.Float32(c)
        out[3] = fx.Float32(d)
        out[4] = fx.Float32(e)
        out[5] = fx.Float32(f)
        out[6] = fx.Float32(g)
        out[7] = fx.Float32(h)
        out[8] = fx.Float32(fx.Int32(i))
        out[9] = fx.Float32(j)
        out[10] = fx.Float32(k)
        out[11] = m
        out[12] = fx.Float32(p)


@flyc.jit
def scalar_echo(
    out: fx.Pointer,
    a: fx.Int8,
    b: fx.Int16,
    c: fx.Int32,
    d: fx.Int64,
    e: fx.Uint8,
    f: fx.Uint16,
    g: fx.Uint32,
    h: fx.Uint64,
    i: fx.Boolean,
    j: fx.Float16,
    k: fx.BFloat16,
    m: fx.Float32,
    p: fx.Float64,
):
    _scalar_echo_kernel(out, a, b, c, d, e, f, g, h, i, j, k, m, p).launch(grid=(1, 1, 1), block=(64, 1, 1))


@fx.struct
class _AxpyParams:
    n: fx.Int32
    alpha: fx.Float32
    block: fx.Constexpr[int]


@flyc.kernel
def _axpy_struct_kernel(X: fx.Pointer, Y: fx.Pointer, p: _AxpyParams):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < p.n:
        Y[idx] = p.alpha * X[idx] + Y[idx]


@flyc.jit
def axpy_struct(X: fx.Pointer, Y: fx.Pointer, p: _AxpyParams):
    _axpy_struct_kernel(X, Y, p).launch(grid=((p.n + p.block - 1) // p.block, 1, 1), block=(p.block, 1, 1))


class _Scaler:
    @flyc.jit
    def launch(self, X: fx.Pointer, n: fx.Int32):
        _add_one_kernel(X, n).launch(grid=((n + 63) // 64, 1, 1), block=(64, 1, 1))


def _ptr(value=0):
    return flyc.from_c_void_p(fx.Float32, value)


def _aot_view(launcher, *args, **kwargs):
    """Compile through the public API and expose its internal export view."""
    compiled = flyc.compile(launcher, *args, **kwargs)
    if compiled._aot_error is not None:
        error_type, message = compiled._aot_error
        raise error_type(message)
    assert compiled._aot_compiled is not None
    return compiled._aot_compiled


# C-caller helpers (metadata-driven, independent of FlyDSL's own dispatch)

_SCALAR_CTYPES = {
    "bool": ctypes.c_bool,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    **{f"{s}int{w}": getattr(ctypes, f"c_{s}int{w}") for s in ("", "u") for w in (8, 16, 32, 64)},
}


def _half_bits(value, encoding):
    if encoding == "float16_bits":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def _pack_args(abi, values):
    """Build ``void **args`` from one Python value per ABI slot.

    ``tensor_layout`` values are ``(shape_values, stride_values)``.
    """
    assert len(values) == len(abi)
    storages = []
    for slot, value in zip(abi, values):
        if slot.kind == "tensor_layout":
            shape_vals, stride_vals = value
            fmt = "<" + {32: "i", 64: "q"}[slot.shape_bits] * len(shape_vals)
            if stride_vals:
                fmt += {32: "i", 64: "q"}[slot.stride_bits] * len(stride_vals)
            data = struct.pack(fmt, *shape_vals, *stride_vals)
            assert len(data) == slot.size
            storages.append(ctypes.create_string_buffer(data, slot.size))
        elif slot.ctype == "pointer":
            storages.append(ctypes.c_void_p(value))
        elif slot.encoding is not None:
            storages.append(ctypes.c_uint16(_half_bits(value, slot.encoding)))
        else:
            storages.append(_SCALAR_CTYPES[slot.ctype](value))
        assert ctypes.sizeof(storages[-1]) == slot.size
    packed = (ctypes.c_void_p * max(len(storages), 1))(*[ctypes.addressof(s) for s in storages])
    packed._keepalive = storages
    return packed


def _link_shared(tmp_path, objects, name, extra=()):
    out = tmp_path / name
    linker = HIPCC or CC
    flags = list(_get_backend_class().aot_runtime_config().linker_flags)
    subprocess.run([linker, "-shared", "-o", str(out), *map(str, objects), *flags, *extra], check=True)
    return out


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@needs_cc
def test_cpu_only_compile_export_and_link(tmp_path):
    """AOT compilation must not require a visible or initialized GPU."""
    script = r"""\
import os
import sys
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.runtime.device_runtime as device_runtime
from tests.unit.test_aot_export import _ptr, axpy

def unexpected_device_runtime():
    raise AssertionError("AOT compilation must not query the device runtime")

assert flyc.get_backend().target.arch == "gfx950"
assert "HIP_VISIBLE_DEVICES" not in os.environ
assert "ROCR_VISIBLE_DEVICES" not in os.environ
device_runtime.get_device_runtime = unexpected_device_runtime

out = Path(sys.argv[1])
compiled = flyc.compile(axpy, _ptr(), _ptr(), 256, 1.0)
assert compiled.export_to_c(out, "cpu_only", "cpu_only_kernel") is None
assert (out / "cpu_only.o").stat().st_size > 0
assert (out / "cpu_only.h").stat().st_size > 0
"""
    child_env = os.environ.copy()
    child_env.update(
        {
            "ARCH": "gfx950",
            "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        }
    )
    child_env.pop("HIP_VISIBLE_DEVICES", None)
    child_env.pop("ROCR_VISIBLE_DEVICES", None)
    child_env.pop("FLYDSL_GPU_ARCH", None)
    child_env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=child_env, check=True)

    shared = _link_shared(tmp_path, [tmp_path / "cpu_only.o"], "libcpu_only.so")
    assert shared.stat().st_size > 0
    needed = _dynamic_info(shared).needed
    assert any(name.startswith("libamdhip64.so") for name in needed)
    assert "libfly_jit_runtime.so" not in needed


class _Loaded:
    """A linked FlyDSL export loaded with ctypes."""

    def __init__(self, lib, compiled, symbol):
        self.compiled = compiled
        self.entry = getattr(lib, symbol)
        self.entry.argtypes = [ctypes.c_void_p]
        self.entry.restype = ctypes.c_int32
        self.init = getattr(lib, f"{symbol}__module_init")
        self.init.restype = ctypes.c_int32
        self.load = getattr(lib, f"{symbol}__module_load")
        self.load.argtypes = [ctypes.c_int32]
        self.load.restype = ctypes.c_int32
        self.unload = getattr(lib, f"{symbol}__module_unload")
        self.unload.restype = ctypes.c_int32
        # The metadata symbol is a char array, not a char pointer.
        self.metadata = ctypes.string_at(ctypes.addressof(ctypes.c_char.in_dll(lib, f"{symbol}__metadata"))).decode()

    def __call__(self, *values):
        return self.entry(_pack_args(self.compiled._abi, values))


# Internal export compilation: ABI model


@pytest.mark.l0_backend_agnostic
def test_export_compile_rejects_non_jit():
    with pytest.raises(TypeError, match="@flyc.jit"):
        flyc.compile(lambda _x: None, 1)


@pytest.mark.l0_backend_agnostic
def test_compiled_function_constructor_remains_compatible():
    def state(args):
        return args

    compiled = flyc.CompiledFunction(state, object())
    assert compiled(1, 2) == (1, 2)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
class TestExportCompile:
    def test_pointer_scalar_explicit_stream(self):
        compiled = _aot_view(axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None))
        assert [(s.arg_index, s.arg_name, s.kind, s.ctype) for s in compiled._abi] == [
            (0, "X", "pointer", "pointer"),
            (1, "Y", "pointer", "pointer"),
            (2, "n", "scalar", "int32"),
            (3, "alpha", "scalar", "float"),
            (4, "stream", "stream", "pointer"),
        ]
        assert [(s.size, s.alignment) for s in compiled._abi] == [(8, 8), (8, 8), (4, 4), (4, 4), (8, 8)]

    def test_implicit_stream_and_constexpr_removed(self):
        compiled = _aot_view(add_one_then_scale, _ptr(), 16, factor=3.0)
        assert [(s.arg_index, s.arg_name, s.kind) for s in compiled._abi] == [
            (0, "X", "pointer"),
            (1, "n", "scalar"),
            (None, None, "stream"),
        ]

    def test_multiple_kernel_launches_traced(self):
        compiled = _aot_view(add_one_then_scale, _ptr(), 16)
        assert compiled._ir_text.count("gpu.launch_func") == 2

    def test_all_scalar_types(self):
        values = [t(1) for t in _SCALAR_TYPES]
        compiled = _aot_view(scalar_echo, _ptr(), *values)
        scalars = compiled._abi[1:-1]
        assert [(s.ctype, s.size, s.encoding) for s in scalars] == [
            ("int8", 1, None),
            ("int16", 2, None),
            ("int32", 4, None),
            ("int64", 8, None),
            ("uint8", 1, None),
            ("uint16", 2, None),
            ("uint32", 4, None),
            ("uint64", 8, None),
            ("bool", 1, None),
            ("uint16", 2, "float16_bits"),
            ("uint16", 2, "bfloat16_bits"),
            ("float", 4, None),
            ("double", 8, None),
        ]
        assert compiled._abi[-1].kind == "stream" and compiled._abi[-1].arg_index is None

    def test_dynamic_tensor_layout_widths_from_lowered_abi(self):
        t = torch.empty_strided((2, 3), (5, 1), device="meta")
        compiled = _aot_view(tensor_add, t, t, t)
        data, layout = compiled._abi[0], compiled._abi[1]
        assert (data.kind, data.ctype, data.size) == ("tensor_data", "pointer", 8)
        assert layout.kind == "tensor_layout" and layout.ctype == "bytes"
        assert layout.shape_dims == (0, 1) and layout.stride_dims == (0,)
        assert (layout.shape_bits, layout.stride_bits) == (32, 64)
        assert (layout.size, layout.alignment) == (2 * 4 + 8, 1)

    def test_32bit_stride_layout(self):
        t = torch.empty(4, 8, device="meta")
        a = flyc.from_torch_tensor(t, use_32bit_stride=True)
        # from_torch_tensor wraps a static layout; mark it dynamic explicitly.
        a.mark_layout_dynamic()
        compiled = _aot_view(tensor_add, a, t, t)
        assert compiled._abi[1].stride_bits == 32 and compiled._abi[1].size == 3 * 4

    def test_static_tensor_layout_has_no_layout_slot(self):
        t = torch.empty(4, 8, device="meta")
        s = flyc.from_torch_tensor(t)
        compiled = _aot_view(tensor_add, s, s, s)
        assert [slot.kind for slot in compiled._abi] == ["tensor_data"] * 3 + ["stream"]

    def test_kwargs_defaults_and_methods(self):
        by_kw = _aot_view(axpy, _ptr(), Y=_ptr(), n=4, alpha=2.0)
        positional = _aot_view(axpy, _ptr(), _ptr(), 4, 2.0, fx.Stream(None))
        # The defaulted explicit stream keeps its index and name.
        assert by_kw._abi == positional._abi
        assert (by_kw._abi[-1].arg_index, by_kw._abi[-1].arg_name) == (4, "stream")

        obj = _Scaler()
        hinted = flyc.compile[{"fast_fp_math": True}](obj.launch)
        assert isinstance(hinted, partial)
        assert _Scaler.launch.compile_hints == {"fast_fp_math": True}

        bound = _aot_view(hinted, _ptr(), 4)
        unbound = _aot_view(_Scaler.launch, obj, _ptr(), 4)
        assert [s.arg_name for s in bound._abi] == ["X", "n", None]
        assert bound._abi == unbound._abi
        # arg_index is relative to the launcher signature without ``self``.
        assert [s.arg_index for s in bound._abi] == [0, 1, None]

    def test_struct_argument_expands_runtime_fields(self):
        compiled = _aot_view(axpy_struct, _ptr(), _ptr(), _AxpyParams(n=8, alpha=1.0, block=128))
        # The constexpr field packs nothing; runtime fields keep declaration order.
        assert [(s.arg_index, s.arg_name, s.field, s.ctype) for s in compiled._abi] == [
            (0, "X", None, "pointer"),
            (1, "Y", None, "pointer"),
            (2, "p", "n", "int32"),
            (2, "p", "alpha", "float"),
            (None, None, None, "pointer"),
        ]
        assert "field" not in compiled._abi[0]._to_dict() and compiled._abi[2]._to_dict()["field"] == "n"

    def test_does_not_dispatch_or_create_engine(self, monkeypatch):
        from flydsl.compiler import jit_executor, jit_function

        def boom(*_a, **_k):
            raise AssertionError("export compilation must not execute anything")

        monkeypatch.setattr(jit_function.JitFunction, "__call__", boom)
        monkeypatch.setattr(jit_executor, "ExecutionEngine", boom)
        monkeypatch.setattr(jit_executor.CompiledArtifact, "_ensure_engine", boom)
        monkeypatch.setattr(jit_function, "_build_call_state", boom)
        _aot_view(axpy, _ptr(), _ptr(), 8, 1.0)

        class CpuDlpackSource:
            def __init__(self):
                self.tensor = torch.empty(4, 8, device="cpu")

            def __dlpack__(self, stream=None):
                return self.tensor.__dlpack__()

        tensor = flyc.from_dlpack(CpuDlpackSource())
        _aot_view(tensor_add, tensor, tensor, tensor)

    def test_placeholder_detection_does_not_duck_type_user_arguments(self):
        from flydsl.compiler.jit_function import _is_compile_placeholder

        class UserArgument:
            is_meta = True
            device = type("Device", (), {"type": "cpu"})()

        assert not _is_compile_placeholder(UserArgument())

    def test_matches_jit_lowering_including_compile_hints(self, monkeypatch):
        monkeypatch.setenv("COMPILE_ONLY", "1")
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

        def make_launcher():
            @flyc.jit
            def divide(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32):
                _div_kernel(X, Y, n).launch(grid=(1, 1, 1), block=(64, 1, 1))

            return divide

        plain, hinted = make_launcher(), make_launcher()
        hinted.compile_hints = {"fast_fp_math": True}
        hinted(_ptr(), _ptr(), 4)
        jit_ir = hinted._last_compiled[1].ir
        monkeypatch.delenv("COMPILE_ONLY")
        aot_ir = _aot_view(hinted, _ptr(), _ptr(), 4)._ir_text
        # The hint changes the GPU binary, so the comparison below is meaningful.
        assert _aot_view(plain, _ptr(), _ptr(), 4)._ir_text != aot_ir
        assert aot_ir == jit_ir

    def test_unsupported_argument_reports_name_and_type(self):
        class Opaque:
            def __get_ir_types__(self):
                return [ir.IntegerType.get_signless(32)]

            def __cache_signature__(self):
                return (Opaque,)

            def __c_abi_spec__(self):
                return [(ctypes.c_int32, lambda a, s: None)]

            @classmethod
            def __construct_from_ir_values__(cls, values, exemplar=None):
                return fx.Int32(values[0])

            def __extract_to_ir_values__(self):
                return []

        @flyc.jit
        def takes_opaque(x: Opaque):
            pass

        with pytest.raises(NotImplementedError, match=r"'x'.*Opaque"):
            _aot_view(takes_opaque, Opaque())

    def test_post_load_processors_rejected(self, monkeypatch):
        from flydsl.compiler.jit_function import JitFunction

        @flyc.jit
        def with_post_load(X: fx.Pointer, n: fx.Int32):
            _add_one_kernel(X, n).launch(grid=(1, 1, 1), block=(64, 1, 1))

        original = JitFunction._trace_and_compile

        def with_processor(self, *args, **kwargs):
            traced = original(self, *args, **kwargs)
            traced.post_load_processors.append(print)
            return traced

        monkeypatch.setattr(JitFunction, "_trace_and_compile", with_processor)
        with pytest.raises(NotImplementedError, match="post-load"):
            _aot_view(with_post_load, _ptr(), 8)

    def test_artifact_keeps_no_example_arguments(self):
        t = torch.empty(4, 8)
        ref = weakref.ref(t)
        compiled = _aot_view(tensor_add, t, t, t)
        del t
        gc.collect()
        assert ref() is None
        assert compiled._abi[0].kind == "tensor_data"


@pytest.mark.l0_backend_agnostic
def test_mixed_width_layout_rejected():
    assert aot._group_width([32, 32], "shape", "A") == 32
    assert aot._group_width([], "stride", "A") is None
    with pytest.raises(NotImplementedError, match="mixed-width"):
        aot._group_width([32, 64], "stride", "A")


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize(
    "semantic, type_asm, ok",
    [
        ({"ctype": "pointer"}, "!llvm.ptr<1>", True),
        ({"ctype": "pointer"}, "i64", False),
        ({"ctype": "bytes"}, "!llvm.struct<packed (i32, i64)>", True),
        ({"ctype": "bool"}, "i1", True),
        ({"ctype": "int32"}, "i32", True),
        ({"ctype": "uint64"}, "i64", True),
        ({"ctype": "int32"}, "f32", False),
        ({"ctype": "float"}, "i32", False),
        ({"ctype": "float"}, "f32", True),
        ({"ctype": "double"}, "f64", True),
        ({"ctype": "uint16", "encoding": "float16_bits"}, "f16", True),
        ({"ctype": "uint16", "encoding": "bfloat16_bits"}, "f16", False),
        ({"ctype": "uint16", "encoding": "bfloat16_bits"}, "bf16", True),
        ({"ctype": "uint16"}, "i16", True),
    ],
)
def test_lowered_type_must_match_slot_ctype(semantic, type_asm, ok):
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        assert aot._lowered_type_matches(semantic, ir.Type.parse(type_asm)) is ok


# Symbol namespacing (IR level)

_NAMESPACE_IR = """
module attributes {gpu.container_module} {
  llvm.mlir.global internal constant @table(42 : i32) : i32
  llvm.mlir.global external @extern_global() : i32
  llvm.func @extern_fn(i32) -> i32
  llvm.func @helper(%a: i32) -> i32 {
    %r = llvm.call @extern_fn(%a) : (i32) -> i32
    llvm.return %r : i32
  }
  llvm.func @ctor() {
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@ctor], priorities = [0 : i32], data = [#llvm.zero]
  llvm.func @entry() -> i32 {
    %p = llvm.mlir.addressof @table : !llvm.ptr
    %q = llvm.mlir.addressof @extern_global : !llvm.ptr
    %v = llvm.load %p : !llvm.ptr -> i32
    %r = llvm.call @helper(%v) : (i32) -> i32
    llvm.return %r : i32
  }
}
"""


@pytest.mark.l1a_compile_no_target_dialect
def test_namespace_renames_definitions_only():
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_NAMESPACE_IR)
        renamed = aot._namespace_symbols(module, "k0")
        assert renamed == {"table": "k0__table", "helper": "k0__helper", "ctor": "k0__ctor", "entry": "k0__entry"}
        module.operation.verify()
        text = str(module)
    # Definitions renamed and internalized; references follow.
    assert "llvm.func internal @k0__helper" in text
    assert "llvm.call @k0__helper" in text
    assert "llvm.mlir.addressof @k0__table" in text
    assert "ctors = [@k0__ctor]" in text
    # External declarations and their uses keep their names.
    assert "llvm.func @extern_fn(i32) -> i32" in text
    assert "llvm.call @extern_fn" in text
    assert "@extern_global()" in text and "llvm.mlir.addressof @extern_global" in text


@pytest.mark.l1a_compile_no_target_dialect
def test_is_definition_checks_body_not_region():
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_NAMESPACE_IR)
        by_name = {
            ir.StringAttr(op.attributes["sym_name"]).value: op
            for op in module.body.operations
            if "sym_name" in op.attributes
        }
        # A declaration still owns an (empty) region.
        assert len(by_name["extern_fn"].regions) == 1
        assert not aot._is_definition(by_name["extern_fn"])
        assert not aot._is_definition(by_name["extern_global"])
        assert aot._is_definition(by_name["helper"])
        assert aot._is_definition(by_name["table"])


@pytest.mark.l1a_compile_no_target_dialect
@needs_cc
def test_external_declaration_resolves_at_link_time(tmp_path):
    ext_src = tmp_path / "ext.c"
    ext_src.write_text("int extern_fn(int x) { return x * 3; }\nint extern_global = 5;\n")
    ext_lib = tmp_path / "libext.so"
    subprocess.run([CC, "-shared", "-fPIC", "-o", str(ext_lib), str(ext_src)], check=True)

    with ir.Context() as ctx, ir.Location.unknown():
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_NAMESPACE_IR)
        aot._namespace_symbols(module, "k0")
        # An externally visible probe calling the (now internal) entry.
        i32 = ir.IntegerType.get_signless(32)
        with ir.InsertionPoint(module.body):
            probe = llvm.LLVMFuncOp("probe", ir.TypeAttr.get(llvm.FunctionType.get(i32, [])))
            with ir.InsertionPoint(probe.body.blocks.append()):
                llvm.ReturnOp(arg=llvm.CallOp(i32, [], [], [], callee="k0__entry").result)
        module.operation.verify()
        data = _emit_host_object(module.operation, 2)
    # Only the probe is exported; extern_fn / extern_global stay undefined references.
    assert set(_defined_global_symbols(data)) == {"probe"}
    obj = tmp_path / "probe.o"
    obj.write_bytes(data)
    so = tmp_path / "libprobe.so"
    subprocess.run([CC, "-shared", "-o", str(so), str(obj), str(ext_lib), f"-Wl,-rpath,{tmp_path}"], check=True)
    lib = ctypes.CDLL(str(so))
    lib.probe.restype = ctypes.c_int32
    assert lib.probe() == 42 * 3


# Embedded runtime archive and ELF


@pytest.mark.l0_backend_agnostic
def test_find_aot_runtime_archive():
    runtime = _get_backend_class().aot_runtime_config()
    archive = _find_aot_runtime_archive()
    assert archive.is_absolute() and archive.is_file()
    assert archive.name == runtime.archive_basename == "libfly_rocm_aot_runtime.a"
    assert archive.read_bytes().startswith(b"!<arch>\n")
    assert runtime.runtime_libraries == ("libamdhip64.so",)
    assert runtime.linker_flags == ("-lamdhip64", "-pthread", "-ldl")

    runtime_object, symbols = aot._materialize_aot_runtime(archive)
    assert runtime_object.startswith(b"\x7fELF")
    assert {"flydslRuntimeTakeError", "hipModuleLoadData", "malloc"} <= symbols


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_backend_selects_the_first_object_by_default():
    backend_cls = _get_backend_class("rocm")
    assert backend_cls.aot_object_index("gfx950") == 0
    # ROCm AOT object index is arch-agnostic (single object); gfx120x family included.
    for arch in ("gfx1200", "gfx1201", "gfx1202", "gfx120x"):
        assert backend_cls.aot_object_index(arch) == 0


@pytest.mark.l1a_compile_no_target_dialect
def test_aot_module_attribute_rejects_invalid_selection():
    with aot._create_mlir_context():
        for text, message in (
            ('#fly.aot_module<"", 0>', "symbol prefix"),
            ('#fly.aot_module<"kernel", -1>', "non-negative"),
            ('#fly.aot_module<"kernel", "not a target">', "TargetAttrInterface"),
        ):
            with pytest.raises(ir.MLIRError, match=message):
                ir.Attribute.parse(text)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_translation_selects_target_instead_of_first_object():
    compiled = _aot_view(axpy, _ptr(), _ptr(), 8, 1.0)
    with aot._create_mlir_context():
        module = ir.Module.parse(compiled._ir_text)
        for op in module.body.operations:
            if op.operation.name != "gpu.binary":
                continue
            original = ir.ArrayAttr(op.attributes["objects"])[0]
            wrong_target = ir.Attribute.parse('#rocdl.target<chip = "gfx942">')
            wrong = gpu.ObjectAttr.get(wrong_target, gpu.CompilationTarget.Binary, b"WRONG_GPU_OBJECT")
            op.attributes["objects"] = ir.ArrayAttr.get([wrong, original])
            break
        else:
            raise AssertionError("compiled launcher has no gpu.binary")
        ir_text = str(module)

    class SelectSecondObject(_get_backend_class("rocm")):
        @classmethod
        def aot_object_index(cls, arch):
            assert arch == "gfx950"
            return 1

    host_object = aot._emit_export_object(
        ir_text,
        compiled._entry,
        compiled._abi,
        "target_selection",
        "target_selection__metadata",
        b"{}",
        SelectSecondObject,
        "gfx950",
    )
    assert b"WRONG_GPU_OBJECT" not in host_object
    assert b"\x7fELF" in host_object


@pytest.mark.l0_backend_agnostic
def test_elf_reader_rejects_malformed_input():
    library = Path(sys.executable).read_bytes()
    # Program headers follow the 64-byte ELF header; section headers sit at the end.
    for bad in (b"", b"not an elf file", library[:64]):
        with pytest.raises(ValueError):
            _dynamic_info(bad)
    for bad in (b"", b"not an elf file", library[: len(library) - 1]):
        with pytest.raises(ValueError):
            _defined_global_symbols(bad)


@pytest.mark.l0_backend_agnostic
def test_output_directory_lock_serializes_threads(tmp_path):
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first():
        with aot._output_directory_lock(tmp_path):
            first_entered.set()
            assert release_first.wait(timeout=5)

    def second():
        assert first_entered.wait(timeout=5)
        with aot._output_directory_lock(tmp_path):
            second_entered.set()

    with ThreadPoolExecutor(2) as pool:
        first_future = pool.submit(first)
        second_future = pool.submit(second)
        assert first_entered.wait(timeout=5)
        assert not second_entered.wait(timeout=0.1)
        release_first.set()
        first_future.result(timeout=5)
        second_future.result(timeout=5)
    assert second_entered.is_set()


# export_to_c


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
class TestExport:
    @pytest.fixture(scope="class")
    @classmethod
    def compiled(cls):
        return _aot_view(axpy, _ptr(), _ptr(), 8, 1.0)

    def test_argument_validation(self, compiled, tmp_path):
        for bad in ("not-a-c-symbol", "1abc", "a b", "int", "class", "__private", "_Reserved"):
            with pytest.raises(ValueError, match="C identifier"):
                compiled._export_to_c(tmp_path, "k", bad)
        for runtime_name in ("flydslRuntimeTakeError", "hipModuleLoadData", "malloc"):
            with pytest.raises(ValueError, match="collide.*backend AOT runtime"):
                compiled._export_to_c(tmp_path, "k", runtime_name)
        with pytest.raises(FileNotFoundError):
            compiled._export_to_c(tmp_path / "missing", "k")
        for bad in ("../escape", "nested/k", "/tmp/k", ".", "..", ""):
            with pytest.raises(ValueError, match="base name"):
                compiled._export_to_c(tmp_path, bad)
        assert list(tmp_path.iterdir()) == []

    def test_export_usage(self, compiled, tmp_path):
        out = tmp_path / "artifacts"
        out.mkdir()
        assert compiled._export_to_c(out, "axpy_file", "aoti_axpy") is None
        assert (out / "axpy_file.o").is_file()
        assert (out / "axpy_file.h").is_file()
        assert sorted(path.name for path in out.iterdir()) == ["axpy_file.h", "axpy_file.o"]
        assert "aoti_axpy" in _defined_global_symbols(out / "axpy_file.o")
        assert b'"schema_version":2' in (out / "axpy_file.o").read_bytes()
        assert b'"runtime_libraries":[{"link_name":"libamdhip64.so"}]' in (out / "axpy_file.o").read_bytes()
        assert b'"linker_flags":["-lamdhip64","-pthread","-ldl"]' in (out / "axpy_file.o").read_bytes()

        assert compiled._export_to_c(out, "default_name") is None
        assert "default_name" in _defined_global_symbols(out / "default_name.o")
        assert compiled._export_to_c(out, "empty_prefix", "") is None
        assert "empty_prefix" in _defined_global_symbols(out / "empty_prefix.o")
        assert compiled._export_to_c(out, "keyword_prefix", function_prefix="keyword_symbol") is None
        assert "keyword_symbol" in _defined_global_symbols(out / "keyword_prefix.o")

    def test_metadata_and_symbols(self, compiled, tmp_path):
        out = tmp_path / "with space"
        out.mkdir()
        compiled._export_to_c(out, "axpy", "aoti_axpy")

        exported = set(_defined_global_symbols(out / "axpy.o"))
        lifecycle = {"aoti_axpy__module_init", "aoti_axpy__module_load", "aoti_axpy__module_unload"}
        assert exported == {"aoti_axpy", "aoti_axpy__metadata", *lifecycle}
        assert (out / "axpy.o").stat().st_mode & 0o044 == 0o044  # not left 0600

    def test_reexport_is_deterministic_and_leaves_artifact_unchanged(self, compiled, tmp_path):
        before = compiled._ir_text
        compiled._export_to_c(tmp_path, "a", "first_name")
        compiled._export_to_c(tmp_path, "b", "second_name")
        compiled._export_to_c(tmp_path, "c", "first_name")
        assert compiled._ir_text == before
        assert (tmp_path / "a.o").read_bytes() == (tmp_path / "c.o").read_bytes()
        assert "first_name" not in (tmp_path / "b.o").read_bytes().decode("latin-1")

    @needs_cc
    def test_header_matches_abi(self, tmp_path):
        compiled = _aot_view(scalar_echo, _ptr(), *[t(1) for t in _SCALAR_TYPES])
        compiled._export_to_c(tmp_path, "echo")
        header = (tmp_path / "echo.h").read_text()
        assert "#define ECHO_NUM_ARGS 15" in header
        assert "int32_t echo(void **args);" in header
        assert "flydsl_status_ = echo__module_init();" in header
        assert "flydsl_status_ = echo__module_load(-1);" in header
        assert "static inline int32_t echo_call_auto(" in header
        assert (
            "static inline int32_t echo_call(void *out, int8_t a, int16_t b, int32_t c, int64_t d, "
            "uint8_t e, uint16_t f, uint32_t g, uint64_t h, bool i, uint16_t j, uint16_t k, float m, "
            "double p, void *stream)"
        ) in header
        # The header compiles as C and C++ (static size assertions included).
        for lang, std in (("c", "-std=c11"), ("c++", "-std=c++17")):
            subprocess.run(
                [CC, "-x", lang, std, "-fsyntax-only", "-Wall", "-Werror", str(tmp_path / "echo.h")], check=True
            )

        t = torch.empty_strided((2, 3), (5, 1), device="meta")
        tensor = _aot_view(tensor_add, t, t, t)
        tensor._export_to_c(tmp_path, "t", "tadd")
        theader = (tmp_path / "t.h").read_text()
        assert (
            "typedef struct __attribute__((packed)) { int32_t shape[2]; int64_t stride[1]; } tadd_A_layout_t;"
            in theader
        )
        assert f"sizeof(tadd_A_layout_t) == {tensor._abi[1].size}" in theader
        subprocess.run([CC, "-x", "c", "-fsyntax-only", "-Wall", "-Werror", str(tmp_path / "t.h")], check=True)

    @needs_cc
    def test_header_parameter_names_avoid_c_keywords(self, tmp_path):
        @flyc.jit
        def keywords(new: fx.Pointer, args: fx.Int32, default: fx.Float32, _Alignas: fx.Int32, __private: fx.Int32):
            _axpy_kernel(new, new, args, default).launch(grid=(1, 1, 1), block=(64, 1, 1))

        compiled = _aot_view(keywords, _ptr(), 4, 1.0, 0, 0)
        compiled._export_to_c(tmp_path, "kw")
        assert (
            "kw_call(void *new_, int32_t args, float default_, int32_t flydsl_Alignas_, "
            "int32_t flydsl__private, void *stream)"
        ) in (tmp_path / "kw.h").read_text()
        for lang in ("c", "c++"):
            subprocess.run([CC, "-x", lang, "-fsyntax-only", "-Wall", "-Werror", str(tmp_path / "kw.h")], check=True)

    @needs_cc
    def test_header_layout_type_names_are_unique(self, tmp_path):
        slot = dict(
            arg_index=0,
            arg_name="p",
            kind="tensor_layout",
            ctype="bytes",
            size=4,
            alignment=1,
            shape_dims=(0,),
            stride_dims=(),
            shape_bits=32,
        )
        metadata = aot._ExportMetadata(
            schema_version=1,
            symbol="k",
            module_init_symbol="k__module_init",
            module_load_symbol="k__module_load",
            module_unload_symbol="k__module_unload",
            metadata_symbol="k__metadata",
            runtime_libraries=(),
            linker_flags=(),
            abi=(aot._AbiSlot(field="a.b", **slot), aot._AbiSlot(field="a_b", **slot)),
            return_abi=aot._ReturnAbi(),
            backend="rocm",
            arch="gfx950",
            host_triple="x86_64-unknown-linux-gnu",
            flydsl_version="test",
        )
        header = aot._render_header(metadata)
        assert "k_p_a_b_layout_t;" in header
        assert "k_p_a_b_layout_t_;" in header
        path = tmp_path / "unique.h"
        path.write_text(header)
        for lang in ("c", "c++"):
            subprocess.run([CC, "-x", lang, "-fsyntax-only", "-Wall", "-Werror", str(path)], check=True)


# GPU end to end


def _export_and_load(tmp_path, compiled, name, lib_name=None):
    compiled._export_to_c(tmp_path, name)
    so = _link_shared(tmp_path, [tmp_path / f"{name}.o"], lib_name or f"lib{name}.so")
    return _Loaded(ctypes.CDLL(str(so)), compiled, name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@needs_gpu
@needs_cc
class TestEndToEnd:
    def test_flyc_compile_result_can_export(self, tmp_path, monkeypatch):
        n = 256
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        expected = 1.5 * x + y
        compiled = flyc.compile(
            axpy,
            _ptr(x.data_ptr()),
            Y=_ptr(y.data_ptr()),
            n=n,
            alpha=1.5,
            stream=fx.Stream(None),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(y, expected)

        y_fast = torch.randn(n, device="cuda")
        expected_fast = 2.0 * x + y_fast
        compiled(
            _ptr(x.data_ptr()),
            _ptr(y_fast.data_ptr()),
            n,
            2.0,
            fx.Stream(None),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(y_fast, expected_fast)

        from flydsl.compiler import jit_function

        def unexpected_compile(*_args, **_kwargs):
            raise AssertionError("export_to_c must reuse the existing compiled artifact")

        monkeypatch.setattr(jit_function.JitFunction, "_trace_and_compile", unexpected_compile)

        assert compiled.export_to_c(tmp_path, "compiled_axpy", "compiled_axpy_entry") is None
        so = _link_shared(tmp_path, [tmp_path / "compiled_axpy.o"], "libcompiled_axpy.so")
        loaded = _Loaded(ctypes.CDLL(str(so)), compiled._aot_compiled, "compiled_axpy_entry")
        assert loaded.init() == 0 and loaded.load(-1) == 0

        y2 = torch.randn(n, device="cuda")
        expected2 = 2.0 * x + y2
        assert loaded(x.data_ptr(), y2.data_ptr(), n, 2.0, None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(y2, expected2)
        assert loaded.unload() == 0

    def test_compile_placeholders_delay_first_launch(self):
        n = 256
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        original_y = y.clone()

        compiled = flyc.compile(axpy, _ptr(), _ptr(), n, 1.5, fx.Stream(None))
        torch.cuda.synchronize()
        torch.testing.assert_close(y, original_y)

        expected = 1.5 * x + y
        compiled(_ptr(x.data_ptr()), _ptr(y.data_ptr()), n, 1.5, fx.Stream(None))
        torch.cuda.synchronize()
        torch.testing.assert_close(y, expected)

    def test_pointer_scalar_explicit_stream(self, tmp_path):
        k = _export_and_load(
            tmp_path,
            _aot_view(axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None)),
            "e2e_axpy",
        )
        assert json.loads(k.metadata)["symbol"] == "e2e_axpy"
        assert k.init() == 0 and k.load(-1) == 0
        n = 1000
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        expected = 2.5 * x + y
        stream = torch.cuda.Stream()
        assert k(x.data_ptr(), y.data_ptr(), n, 2.5, stream.cuda_stream) == 0
        stream.synchronize()
        torch.testing.assert_close(y, expected)
        assert k.unload() == 0 and k.unload() == 0

    def test_convenience_call_initializes_and_loads_lazily(self, tmp_path):
        compiled = _aot_view(add_one_then_scale, _ptr(), 8)
        compiled._export_to_c(tmp_path, "lazy")
        wrapper = tmp_path / "lazy_wrapper.c"
        wrapper.write_text(
            '#include "lazy.h"\n'
            "int32_t invoke_loaded(void *x, int32_t n, void *stream) {\n"
            "  return lazy_call(x, n, stream);\n"
            "}\n"
            "int32_t invoke_lazy(void *x, int32_t n, void *stream) {\n"
            "  return lazy_call_auto(x, n, stream);\n"
            "}\n"
        )
        so = _link_shared(
            tmp_path,
            [tmp_path / "lazy.o", wrapper],
            "liblazy.so",
            extra=[f"-I{tmp_path}"],
        )
        lib = ctypes.CDLL(str(so))
        invoke_loaded = lib.invoke_loaded
        invoke_loaded.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        invoke_loaded.restype = ctypes.c_int32
        invoke = lib.invoke_lazy
        invoke.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        invoke.restype = ctypes.c_int32
        x = torch.zeros(8, device="cuda")
        assert invoke_loaded(x.data_ptr(), x.numel(), None) == -1
        assert invoke(x.data_ptr(), x.numel(), None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(x, torch.full((8,), 2.0, device="cuda"))
        loaded = _Loaded(lib, compiled, "lazy")
        assert loaded.unload() == 0

    def test_struct_argument(self, tmp_path):
        compiled = _aot_view(axpy_struct, _ptr(), _ptr(), _AxpyParams(n=8, alpha=1.0, block=128))
        k = _export_and_load(tmp_path, compiled, "e2e_struct")
        assert k.init() == 0 and k.load(-1) == 0
        n = 300
        x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        expected = 0.5 * x + y
        assert k(x.data_ptr(), y.data_ptr(), n, 0.5, None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(y, expected)
        k.unload()

    def test_implicit_stream_multi_kernel(self, tmp_path):
        k = _export_and_load(
            tmp_path,
            _aot_view(add_one_then_scale, _ptr(), 8, factor=3.0),
            "e2e_two",
        )
        assert k.init() == 0 and k.load(-1) == 0
        x = torch.arange(100, device="cuda", dtype=torch.float32)
        expected = (x + 1) * 3
        assert k(x.data_ptr(), 100, None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(x, expected)
        k.unload()

    def test_dynamic_non_contiguous_tensor(self, tmp_path):
        t = torch.empty_strided((2, 3), (5, 1), device="meta")
        k = _export_and_load(tmp_path, _aot_view(tensor_add, t, t, t), "e2e_tensor")
        assert k.init() == 0 and k.load(-1) == 0

        def strided(fill):
            base = torch.full((10,), fill, device="cuda")
            return base, base.as_strided((2, 3), (5, 1))

        _, a = strided(0.0)
        a.copy_(torch.randn(2, 3))
        _, b = strided(0.0)
        b.copy_(torch.randn(2, 3))
        c_base, c = strided(-7.0)
        layout = ((2, 3), (5,))
        args = [a.data_ptr(), layout, b.data_ptr(), layout, c.data_ptr(), layout, None]
        assert k(*args) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a + b)
        padding = c_base.view(2, 5)[:, 3:]
        assert torch.all(padding == -7.0), "padding between rows must not be written"

        # Same object, a different dynamic shape.
        a2, b2 = torch.randn(33, 70, device="cuda"), torch.randn(33, 70, device="cuda")
        c2 = torch.zeros(33, 70, device="cuda")
        layout2 = ((33, 70), (70,))
        assert k(a2.data_ptr(), layout2, b2.data_ptr(), layout2, c2.data_ptr(), layout2, None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c2, a2 + b2)
        k.unload()

    def test_scalar_abi_values(self, tmp_path):
        k = _export_and_load(tmp_path, _aot_view(scalar_echo, _ptr(), *[t(1) for t in _SCALAR_TYPES]), "e2e_echo")
        assert k.init() == 0 and k.load(-1) == 0
        out = torch.zeros(13, device="cuda")
        values = [-5, -300, 70000, 2**40, 250, 65000, 4_000_000_000, 2**50, True, 1.5, -2.5, 3.25, 6.125]
        assert k(out.data_ptr(), *values, None) == 0
        torch.cuda.synchronize()
        expected = torch.tensor([float(v) for v in values], dtype=torch.float32, device="cpu")
        torch.testing.assert_close(out.cpu(), expected)
        k.unload()

    def test_status_propagation(self, tmp_path):
        k = _export_and_load(tmp_path, _aot_view(add_one_then_scale, _ptr(), 8), "e2e_status")
        x = torch.zeros(8, device="cuda")
        assert k(x.data_ptr(), 8, None) == -1  # not initialized
        assert k.load(-1) == -1
        assert k.init() == 0 and k.init() == 0
        assert k(x.data_ptr(), 8, None) == -2  # not loaded on this device
        assert k.load(10_000) > 0
        assert k.load(-1) == 0 and k.load(-1) == 0
        assert k(x.data_ptr(), 8, None) == 0
        torch.cuda.synchronize()
        assert k.unload() == 0
        assert k(x.data_ptr(), 8, None) == -1
        torch.testing.assert_close(x, torch.full((8,), 2.0, device="cuda"))

    def test_two_artifacts_in_one_library(self, tmp_path):
        first = _aot_view(add_one_then_scale, _ptr(), 8, factor=2.0)
        second = _aot_view(add_one_then_scale, _ptr(), 8, factor=10.0)
        first._export_to_c(tmp_path, "a", "twin_a")
        second._export_to_c(tmp_path, "b", "twin_b")
        so = _link_shared(tmp_path, [tmp_path / "a.o", tmp_path / "b.o"], "libtwins.so", extra=["-Wl,--no-undefined"])
        lib = ctypes.CDLL(str(so))
        ka, kb = _Loaded(lib, first, "twin_a"), _Loaded(lib, second, "twin_b")
        for k in (ka, kb):
            assert k.init() == 0 and k.load(-1) == 0
        x = torch.ones(8, device="cuda")
        assert ka(x.data_ptr(), 8, None) == 0
        assert kb(x.data_ptr(), 8, None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(x, torch.full((8,), 50.0, device="cuda"))  # ((1+1)*2+1)*10
        assert ka.unload() == 0
        assert kb(x.data_ptr(), 8, None) == 0  # independent module state
        assert kb.unload() == 0

    def test_concurrent_launches(self, tmp_path):
        k = _export_and_load(
            tmp_path,
            _aot_view(axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None)),
            "e2e_mt",
        )
        assert k.init() == 0
        n, threads, iters = 4096, 8, 25
        x = torch.ones(n, device="cuda")
        ys = [torch.zeros(n, device="cuda") for _ in range(threads)]
        streams = [torch.cuda.Stream() for _ in range(threads)]
        barrier = threading.Barrier(threads)

        def work(i):
            barrier.wait()
            status = [k.load(-1)]
            status += [k(x.data_ptr(), ys[i].data_ptr(), n, 1.0, streams[i].cuda_stream) for _ in range(iters)]
            return status

        with ThreadPoolExecutor(threads) as pool:
            statuses = [s for result in pool.map(work, range(threads)) for s in result]
        torch.cuda.synchronize()
        assert statuses == [0] * (threads * (iters + 1))
        for y in ys:
            torch.testing.assert_close(y, torch.full((n,), float(iters), device="cuda"))
        k.unload()

    def test_unload_racing_with_launches(self, tmp_path):
        """Launches during unload/reload cycles fail cleanly instead of using freed state."""
        k = _export_and_load(tmp_path, _aot_view(add_one_then_scale, _ptr(), 8), "e2e_race")
        x = torch.zeros(64, device="cuda")
        stop = threading.Event()
        device = torch.cuda.current_device()

        def launch():
            torch.cuda.set_device(device)
            seen = set()
            while not stop.is_set():
                seen.add(k(x.data_ptr(), 64, None))
            return seen

        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(launch) for _ in range(4)]
            for _ in range(50):
                assert k.init() == 0 and k.load(-1) == 0
                assert k.unload() == 0
            stop.set()
            seen = set().union(*(f.result() for f in futures))
        torch.cuda.synchronize()
        assert seen <= {0, -1, -2}

    @pytest.mark.skipif(not ROCM_CLANG.exists(), reason="requires ROCm clang")
    def test_extern_bitcode_launcher(self, tmp_path):
        from flydsl.compiler.extern_link import link_extern
        from flydsl.expr.extern import ffi

        src = tmp_path / "triple.c"
        src.write_text("float aot_test_triple(float x) { return 3.0f * x; }\n")
        bitcode = tmp_path / "triple.bc"
        arch = flyc.get_backend().target.arch
        subprocess.run(
            [str(ROCM_CLANG), "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch}", "-nogpulib", "-O2"]
            + ["-emit-llvm", "-c", str(src), "-o", str(bitcode)],
            check=True,
        )
        triple = link_extern(ffi("aot_test_triple", ["float32"], "float32"), bitcode_path=str(bitcode))

        @flyc.kernel
        def triple_kernel(X: fx.Pointer, n: fx.Int32):
            idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
            if idx < n:
                X[idx] = fx.Float32(triple(X[idx]))

        @flyc.jit
        def triple_launcher(X: fx.Pointer, n: fx.Int32):
            triple_kernel(X, n).launch(grid=(1, 1, 1), block=(64, 1, 1))

        k = _export_and_load(tmp_path, _aot_view(triple_launcher, _ptr(), 8), "e2e_extern")
        assert k.init() == 0 and k.load(-1) == 0
        x = torch.arange(8, dtype=torch.float32, device="cuda")
        assert k(x.data_ptr(), 8, None) == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(x, torch.arange(8, dtype=torch.float32, device="cuda") * 3)
        k.unload()

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(not HAS_GPU or torch.cuda.device_count() < 2, reason="requires 2 GPUs")
    def test_two_devices(self, tmp_path):
        k = _export_and_load(tmp_path, _aot_view(add_one_then_scale, _ptr(), 8), "e2e_devs")
        assert k.init() == 0
        prev = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            assert k.load(-1) == 0
            x1 = torch.zeros(8, device="cuda:1")
            torch.cuda.set_device(1)
            assert k(x1.data_ptr(), 8, None) == -2  # device 1 not loaded yet
            assert k.load(1) == 0
            assert k(x1.data_ptr(), 8, None) == 0
            torch.cuda.set_device(0)
            x0 = torch.zeros(8, device="cuda:0")
            assert k(x0.data_ptr(), 8, None) == 0
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            torch.testing.assert_close(x0.cpu(), torch.full((8,), 2.0))
            torch.testing.assert_close(x1.cpu(), torch.full((8,), 2.0))
        finally:
            torch.cuda.set_device(prev)
        assert k.unload() == 0
