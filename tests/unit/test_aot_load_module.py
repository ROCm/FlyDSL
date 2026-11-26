#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``flyc.load_module``: load an ``export_to_c`` object back into the process.

The object is JIT-linked in memory; launches go through the returned
``CompiledFunction`` and must match the JIT-compiled launcher.
"""

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

HAS_GPU = torch is not None and torch.cuda.is_available()

pytestmark = [
    pytest.mark.rocm_lower,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="flyc.load_module supports Linux only"),
]
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="requires a ROCm GPU")


@flyc.kernel
def _axpy_kernel(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        Y[idx] = alpha * X[idx] + Y[idx]


@flyc.jit
def axpy(
    X: fx.Pointer,
    Y: fx.Pointer,
    n: fx.Int32 = fx.Int32(8),
    alpha: fx.Float32 = 2.5,
    stream: fx.Stream = fx.Stream(None),
):
    _axpy_kernel(X, Y, n, alpha).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.kernel
def _scale_kernel(X: fx.Pointer, n: fx.Int32, factor: fx.Constexpr[float]):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        X[idx] = X[idx] * factor


@flyc.jit
def scale(X: fx.Pointer, factor: fx.Constexpr[float], n: fx.Int32):
    _scale_kernel(X, n, factor).launch(grid=((n + 63) // 64, 1, 1), block=(64, 1, 1))


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


@flyc.kernel
def _add_one_kernel(X: fx.Pointer):
    X[fx.thread_idx.x] = X[fx.thread_idx.x] + 1.0


@flyc.jit
def add_one_twice(X: fx.Pointer, first_block: fx.Int32, second_block: fx.Int32):
    _add_one_kernel(X).launch(grid=(1, 1, 1), block=(first_block, 1, 1))
    _add_one_kernel(X).launch(grid=(1, 1, 1), block=(second_block, 1, 1))


@flyc.jit
def add_one(X: fx.Pointer, block: fx.Int32):
    _add_one_kernel(X).launch(grid=(1, 1, 1), block=(block, 1, 1))


def _ptr(value=0):
    return flyc.from_c_void_p(fx.Float32, value)


def _export(tmp_path, name, launcher, *args):
    compiled = flyc.compile(launcher, *args, compile_only=True)
    compiled.export_to_c(tmp_path, name)
    return tmp_path / f"{name}.o"


@pytest.mark.l1b_target_dialect
class TestMetadata:
    def test_signature_recorded(self, tmp_path):
        from flydsl.utils._elf import _defined_global_symbols

        path = _export(tmp_path, "axpy", axpy, _ptr(), _ptr())
        assert "axpy__metadata" in _defined_global_symbols(path)
        data = path.read_bytes()
        start = data.index(b'{"abi"')
        metadata = json.loads(data[start : data.index(b"\0", start)])
        assert metadata["schema_version"] == 3
        assert [p["name"] for p in metadata["params"]] == ["X", "Y", "n", "alpha", "stream"]
        assert all(p["kind"] == "positional_or_keyword" for p in metadata["params"])
        assert [p.get("default", "<none>") for p in metadata["params"]] == ["<none>", "<none>", 8, 2.5, None]
        assert "Pointer" in metadata["params"][0]["annotation"]

    def test_rejects_other_objects(self, tmp_path):
        bogus = tmp_path / "bogus.o"
        bogus.write_bytes(b"not an object")
        with pytest.raises(Exception):
            flyc.load_module(bogus)

    def test_embedded_runtime_needs_no_platform_runtime(self):
        # The loader neither runs static constructors nor services thread-local storage.
        from flydsl.compiler._aot import _materialize_aot_runtime
        from flydsl.runtime._libraries import _find_aot_runtime_archive
        from flydsl.utils._elf import _sections

        runtime_object, _symbols = _materialize_aot_runtime(_find_aot_runtime_archive())
        for _name, sh_type, sh_flags, *_rest in _sections(runtime_object):
            assert sh_type not in (14, 15, 16)  # SHT_INIT_ARRAY, SHT_FINI_ARRAY, SHT_PREINIT_ARRAY
            assert not sh_flags & 0x400  # SHF_TLS

    @pytest.mark.skipif(shutil.which("cc") is None, reason="requires a C compiler")
    @pytest.mark.parametrize(
        "source, message",
        [
            ("__attribute__((constructor)) static void init(void) {}", "static constructors"),
            ("__thread int value; int get(void) { return value; }", "thread-local storage"),
            ("extern __thread int value; int get(void) { return value; }", "thread-local storage"),
        ],
    )
    def test_loader_rejects_platform_runtime_sections(self, tmp_path, source, message):
        from flydsl._mlir._mlir_libs._mlirDialectsFly import _HostObject

        (tmp_path / "probe.c").write_text(source + "\n")
        subprocess.run(["cc", "-fPIC", "-c", "probe.c", "-o", "probe.o"], cwd=tmp_path, check=True)
        with pytest.raises(RuntimeError, match=message):
            _HostObject((tmp_path / "probe.o").read_bytes(), [])


@pytest.mark.l2_device
@needs_gpu
class TestLoadModule:
    def test_pointer_scalars_and_defaults(self, tmp_path):
        fn = flyc.load_module(_export(tmp_path, "axpy", axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None)))
        assert isinstance(fn, flyc.CompiledFunction)
        n = 1000
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        expected = 1.5 * x + y
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        fn(x.data_ptr(), y.data_ptr(), n, 1.5, stream)
        stream.synchronize()
        torch.testing.assert_close(y, expected)

        # Omitted trailing arguments take the launcher's defaults: n=8, alpha=2.5, default stream.
        before = y.clone()
        fn(x.data_ptr(), y.data_ptr())
        torch.cuda.synchronize()
        torch.testing.assert_close(y[:8], 2.5 * x[:8] + before[:8])
        torch.testing.assert_close(y[8:], before[8:])

        with pytest.raises(TypeError, match="'Y'"):
            fn(x.data_ptr())

    def test_matches_jit_and_ignores_constexpr(self, tmp_path):
        fn = flyc.load_module(_export(tmp_path, "scale", scale, _ptr(), 3.0, 4))
        x = torch.arange(100, dtype=torch.float32, device="cuda")
        y = x.clone()
        fn(x.data_ptr(), None, 100)
        scale(_ptr(y.data_ptr()), 3.0, 100)
        torch.cuda.synchronize()
        torch.testing.assert_close(x, y)

    def test_dynamic_tensor_layout(self, tmp_path):
        a, b, c = (torch.empty(4, 4) for _ in range(3))
        fn = flyc.load_module(_export(tmp_path, "tensor_add", tensor_add, a, b, c))
        for shape in ((33, 70), (128, 16)):
            a = torch.randn(*shape, device="cuda")
            b = torch.randn(*shape, device="cuda")
            c = torch.zeros(*shape, device="cuda")
            fn(a, b, c)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, a + b)

    def test_struct_argument(self, tmp_path):
        fn = flyc.load_module(
            _export(tmp_path, "axpy_struct", axpy_struct, _ptr(), _ptr(), _AxpyParams(n=8, alpha=1.0, block=64))
        )
        x = torch.randn(300, device="cuda")
        y = torch.randn(300, device="cuda")
        expected = 0.5 * x + y
        fn(x.data_ptr(), y.data_ptr(), _AxpyParams(n=300, alpha=0.5, block=64))
        torch.cuda.synchronize()
        torch.testing.assert_close(y, expected)

    def test_concurrent_threads(self, tmp_path):
        # Each call reports through its own status argument, so concurrent calls do not share one.
        fn = flyc.load_module(_export(tmp_path, "axpy", axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None)))
        xs = [torch.randn(512, device="cuda") for _ in range(8)]
        ys = [torch.randn(512, device="cuda") for _ in range(8)]
        expected = [2.0 * x + y for x, y in zip(xs, ys)]

        def run(i):
            for _ in range(10):
                fn(xs[i].data_ptr(), ys[i].data_ptr(), 512, 2.0)
                fn(xs[i].data_ptr(), ys[i].data_ptr(), 512, -2.0)
            fn(xs[i].data_ptr(), ys[i].data_ptr(), 512, 2.0)

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(run, range(8)))
        torch.cuda.synchronize()
        for y, e in zip(ys, expected):
            torch.testing.assert_close(y, e)

    def test_load_twice_and_release(self, tmp_path):
        path = _export(tmp_path, "axpy", axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None))
        first = flyc.load_module(path)
        second = flyc.load_module(path)
        x = torch.ones(8, device="cuda")
        y = torch.zeros(8, device="cuda")
        first(x.data_ptr(), y.data_ptr(), 8, 1.0)
        del first
        second(x.data_ptr(), y.data_ptr(), 8, 1.0)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, torch.full_like(y, 2.0))
        with pytest.raises(RuntimeError, match="cannot be exported again"):
            second.export_to_c(tmp_path, "again")

    def test_failed_launch_ends_the_call(self, tmp_path):
        fn = flyc.load_module(_export(tmp_path, "add_one_twice", add_one_twice, _ptr(), 8, 8))
        x = torch.zeros(8, device="cuda")
        # More threads per block than any device allows: the first launch fails and the second is skipped.
        with pytest.raises(RuntimeError, match="failed with status [1-9]"):
            fn(x.data_ptr(), 1 << 16, 8)
        torch.cuda.synchronize()
        torch.testing.assert_close(x, torch.zeros_like(x))
        # The failure belonged to that call alone.
        fn(x.data_ptr(), 8, 8)
        torch.cuda.synchronize()
        torch.testing.assert_close(x, torch.full_like(x, 2.0))

    @pytest.mark.parametrize("hint", [None, 512])
    def test_runtime_block_of_512(self, tmp_path, monkeypatch, hint):
        # flyc.compile[hints] stores its hints on the launcher; restore them after the test.
        monkeypatch.setattr(add_one, "compile_hints", {})
        compile_ = flyc.compile if hint is None else flyc.compile[{"max_flat_work_group_size": hint}]
        compile_(add_one, _ptr(), 8, compile_only=True).export_to_c(tmp_path, "add_one")
        fn = flyc.load_module(tmp_path / "add_one.o")
        x = torch.zeros(1024, device="cuda")

        if hint is None:
            # The object keeps the default limit of 256, so HIP refuses 512 threads.
            with pytest.raises(RuntimeError, match="failed with status [1-9]"):
                fn(x.data_ptr(), 512)
            torch.cuda.synchronize()
            assert int(torch.count_nonzero(x)) == 0
        else:
            fn(x.data_ptr(), 512)
            torch.cuda.synchronize()
            torch.testing.assert_close(x, torch.cat([torch.ones(512), torch.zeros(512)]).cuda())
            # The hint is the object's limit: a larger block is still refused.
            with pytest.raises(RuntimeError, match="failed with status [1-9]"):
                fn(x.data_ptr(), 1024)

    def test_many_loads_stay_usable(self, tmp_path):
        # A load holds no per-process resource, so more loads than there are pthread keys still work.
        path = _export(tmp_path, "axpy", axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None))
        loaded = [flyc.load_module(path) for _ in range(1100)]
        x = torch.ones(8, device="cuda")
        y = torch.zeros(8, device="cuda")
        loaded[-1](x.data_ptr(), y.data_ptr(), 8, 1.0)
        loaded[0](x.data_ptr(), y.data_ptr(), 8, 1.0)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, torch.full_like(y, 2.0))

    @pytest.mark.skipif(not HAS_GPU or torch.cuda.device_count() < 2, reason="requires two GPUs")
    def test_loads_lazily_on_other_devices(self, tmp_path):
        # Loaded on the current device only; another device loads on its first call there.
        fn = flyc.load_module(_export(tmp_path, "axpy", axpy, _ptr(), _ptr(), 8, 1.0, fx.Stream(None)))
        for device in (1, 0, 1):
            with torch.cuda.device(device):
                x = torch.ones(8, device="cuda")
                y = torch.zeros(8, device="cuda")
                fn(x.data_ptr(), y.data_ptr(), 8, 3.0)
                torch.cuda.synchronize()
                torch.testing.assert_close(y, torch.full_like(y, 3.0))
