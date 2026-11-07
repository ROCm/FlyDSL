# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for ``flydsl.extension.hadamard`` (``fx.hadamard``)."""

import itertools
import math
import os
from contextlib import contextmanager

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import func as _mlir_func
from flydsl.compiler.protocol import dsl_size_of

_ALL_ARCHES = ("gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250")


def _full_matrix_enabled() -> bool:
    return os.environ.get("FLYDSL_HADAMARD_TESTS_FULL") == "1" or os.environ.get("RUN_TESTS_FULL") == "1"


def _matrix_cases(*axes):
    """Return a diagonal matrix visiting every axis value, or the full product on request."""
    axes = tuple(tuple(axis) for axis in axes)
    if _full_matrix_enabled():
        return tuple(itertools.product(*axes))
    count = max(map(len, axes))
    return tuple(tuple(axis[index % len(axis)] for axis in axes) for index in range(count))


@contextmanager
def _target_arch(arch: str):
    prev_arch = os.environ.get("ARCH")
    prev_gpu = os.environ.get("FLYDSL_GPU_ARCH")
    os.environ["ARCH"] = arch
    os.environ["FLYDSL_GPU_ARCH"] = arch
    try:
        yield
    finally:
        if prev_arch is None:
            os.environ.pop("ARCH", None)
        else:
            os.environ["ARCH"] = prev_arch
        if prev_gpu is None:
            os.environ.pop("FLYDSL_GPU_ARCH", None)
        else:
            os.environ["FLYDSL_GPU_ARCH"] = prev_gpu


def _ref_fwht(x: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
    """Exact float64 Sylvester Fast Walsh-Hadamard Transform along the last dimension."""
    h = x.to(torch.float64).clone()
    n = h.shape[-1]
    stride = 1
    while stride < n:
        h = h.view(*h.shape[:-1], n // (2 * stride), 2, stride)
        a = h[..., 0, :] + h[..., 1, :]
        b = h[..., 0, :] - h[..., 1, :]
        h = torch.stack((a, b), dim=-2).reshape(*x.shape[:-1], n)
        stride <<= 1
    if normalize:
        h = h / math.sqrt(n)
    return h


def _compile_to_isa(monkeypatch, tmp_path, arch: str, build_fn) -> str:
    dump_dir = tmp_path / f"isa_{arch}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    with _target_arch(arch):
        build_fn()
    isa_files = list(dump_dir.glob("*/*_final_isa.s"))
    assert isa_files, f"expected dumped final ISA under {dump_dir}"
    return "\n".join(p.read_text() for p in isa_files)


# ---------------------------------------------------------------------------
# L1a: Portable target-independent IR & compile-time validation
# ---------------------------------------------------------------------------


@pytest.mark.l1a_compile_no_target_dialect
def test_universal_fwht_emits_no_rocdl_ops(ctx, insert_point):
    i32 = fx.Int32.ir_type
    f32 = fx.Float32.ir_type

    @_mlir_func.FuncOp.from_py_func(f32, i32)
    def _universal_warp_body(val, _tag):
        y = fx.hadamard.universal.warp_fwht(fx.Float32(val), width=16, normalize=True)
        return y.ir_value()

    ir_text = str(ctx.module)
    assert "rocdl." not in ir_text
    assert "gpu.shuffle" in ir_text


@pytest.mark.l1a_compile_no_target_dialect
def test_validation_and_storage_contracts():
    with pytest.raises(ValueError, match="power-of-two"):
        with _target_arch("gfx1201"):
            fx.hadamard.BlockFWHT[fx.Float32, 48]

    with pytest.raises(ValueError, match="1024"):
        with _target_arch("gfx1201"):
            fx.hadamard.BlockFWHT[fx.Float32, 2048]

    with pytest.raises(ValueError, match="power-of-two"):
        with _target_arch("gfx1201"):
            fx.hadamard.BlockFWHT[fx.Float32, 64, 3]

    with pytest.raises(TypeError, match="Float32, Float16, BFloat16"):
        with _target_arch("gfx1201"):
            fx.hadamard.BlockFWHT[fx.Int32, 64]

    with pytest.raises(TypeError, match="specialize"):
        fx.hadamard.BlockFWHT.fwht(None)

    with _target_arch("gfx1201"):
        single = fx.hadamard.BlockFWHT[fx.Float16, 32, 4]
        assert single.SharedStorage is fx.Empty
        assert single.num_warps == 1
        assert single.transform_size == 128

        with pytest.raises(TypeError, match="already specialized"):
            _ = single[fx.Float16, 32]

        for block_threads in (64, 128, 256, 512, 1024):
            multi = fx.hadamard.BlockFWHT[fx.Float16, block_threads, 2]
            expected_n = block_threads * 2
            assert dsl_size_of(multi.SharedStorage) == (expected_n + 32) * 4

    with _target_arch("gfx942"):
        single_w64 = fx.hadamard.BlockFWHT[fx.Float32, 64, 2]
        assert single_w64.SharedStorage is fx.Empty
        assert single_w64.num_warps == 1

        multi_w64 = fx.hadamard.BlockFWHT[fx.Float32, 256, 4]
        assert multi_w64.num_warps == 4
        assert dsl_size_of(multi_w64.SharedStorage) == (256 * 4 + 32) * 4


# ---------------------------------------------------------------------------
# L1b: 5-architecture ISA verification (gfx942, gfx950, gfx1100, gfx1201, gfx1250)
# ---------------------------------------------------------------------------


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", _ALL_ARCHES)
def test_warp_fwht_isa_across_architectures(monkeypatch, tmp_path, arch):
    def _build():
        w = 32 if arch.startswith(("gfx11", "gfx12")) else 64

        @flyc.kernel
        def _k(inp: fx.Pointer, out: fx.Pointer):
            tid = fx.thread_idx.x
            v = (inp + tid * 2).load(fx.Float16x2)
            y = fx.hadamard.warp_fwht(v, width=w, normalize=True)
            (out + tid * 2).store(y)

        @flyc.jit
        def _launch(inp: fx.Pointer, out: fx.Pointer):
            _k(inp, out).launch(grid=(1, 1, 1), block=(w, 1, 1))

        ptr = flyc.from_c_void_p(fx.Float16, 0x1000)
        _launch(ptr, ptr)

    isa = _compile_to_isa(monkeypatch, tmp_path, arch, _build)
    assert "_fake16" not in isa

    if arch.startswith(("gfx11", "gfx12")):
        # Wave32 RDNA3 / RDNA4 / CDNA5: 100% pure VALU DPP + permlanex16, zero LDS!
        assert "row_xmask" in isa
        assert "v_permlanex16_b32" in isa
        assert "ds_swizzle_b32" not in isa
        assert "ds_bpermute_b32" not in isa
        assert "row_ror" not in isa
    elif arch.startswith("gfx95"):
        # Wave64 CDNA4: quad_perm + row_ror:8 + SWAP,4 swizzle + VALU permlane16/32_swap, zero bpermute!
        assert "quad_perm" in isa
        assert "row_ror:8" in isa
        assert "ds_swizzle_b32" in isa
        assert "swizzle(SWAP,16)" not in isa
        assert "v_permlane16_swap_b32" in isa
        assert "v_permlane32_swap_b32" in isa
        assert "ds_bpermute_b32" not in isa
        assert "v_permlanex16_b32" not in isa
        assert "row_xmask" not in isa
    else:
        # Wave64 CDNA3: quad_perm + row_ror:8 + SWAP,4/16 ds_swizzle_b32 + cross-half-wave bpermute.
        assert "quad_perm" in isa
        assert "row_ror:8" in isa
        assert "swizzle(SWAP,4)" in isa
        assert "swizzle(SWAP,16)" in isa
        assert "ds_bpermute_b32" in isa
        assert "v_permlanex16_b32" not in isa
        assert "row_xmask" not in isa


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", _ALL_ARCHES)
def test_block_fwht_multiwarp_isa_across_architectures(monkeypatch, tmp_path, arch):
    def _build():
        block = 128 if arch.startswith(("gfx11", "gfx12")) else 256

        @flyc.kernel
        def _k(inp: fx.Pointer, out: fx.Pointer):
            fwht = fx.hadamard.BlockFWHT[fx.BFloat16, fx.known_block_size(), 2]
            storage = fx.SharedAllocator().allocate(fwht.SharedStorage).peek()
            tid = fx.thread_idx.x
            v = (inp + tid * 2).load(fx.BFloat16x2)
            y = fwht(v, storage=storage, normalize=True)
            (out + tid * 2).store(y)

        @flyc.jit
        def _launch(inp: fx.Pointer, out: fx.Pointer):
            _k(inp, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

        ptr = flyc.from_c_void_p(fx.BFloat16, 0x1000)
        _launch(ptr, ptr)

    isa = _compile_to_isa(monkeypatch, tmp_path, arch, _build)
    assert "_fake16" not in isa
    assert "ds_write" in isa or "ds_store" in isa
    if arch.startswith("gfx12"):
        assert "s_wait_dscnt" in isa
    else:
        assert "s_waitcnt" in isa


# ---------------------------------------------------------------------------
# L2: Live GPU numerical & universal-equivalence verification
# ---------------------------------------------------------------------------


_DTYPE_SPECS = (
    (fx.Float32, torch.float32, 1e-5, 1e-5),
    (fx.Float16, torch.float16, 2e-2, 2e-2),
    (fx.BFloat16, torch.bfloat16, 1.5e-1, 1.5e-1),
)

_BLOCK_SHAPE_ITEMS = (
    ((16, 1, 1), 1),
    ((32, 1, 1), 1),
    ((32, 1, 1), 4),
    ((64, 1, 1), 1),
    ((128, 1, 1), 2),
    ((32, 4, 1), 2),
    ((256, 1, 1), 4),
    ((512, 1, 1), 4),
    ((1024, 1, 1), 2),
)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "width,items,normalize",
    _matrix_cases((1, 2, 4, 8, 16, 32), (1, 2, 4), (False, True)),
)
def test_warp_fwht_gpu_accuracy_and_universal_equivalence(width, items, normalize):
    if not torch.cuda.is_available():
        pytest.skip("GPU device required")

    block = 32
    total = block * items
    group_n = width * items

    @flyc.kernel
    def _k_rocdl(inp: fx.Tensor, out: fx.Tensor):
        tid = fx.thread_idx.x
        if fx.const_expr(items == 1):
            y = fx.hadamard.warp_fwht(inp[tid], width=width, normalize=normalize)
            out[tid] = y
        else:
            vals = [inp[tid * items + j] for j in fx.range_constexpr(items)]
            ys = fx.hadamard.warp_fwht(vals, width=width, normalize=normalize)
            for j in fx.range_constexpr(items):
                out[tid * items + j] = ys[j]

    @flyc.kernel
    def _k_univ(inp: fx.Tensor, out: fx.Tensor):
        tid = fx.thread_idx.x
        if fx.const_expr(items == 1):
            y = fx.hadamard.universal.warp_fwht(inp[tid], width=width, normalize=normalize)
            out[tid] = y
        else:
            vals = tuple(inp[tid * items + j] for j in fx.range_constexpr(items))
            ys = fx.hadamard.universal.warp_fwht(vals, width=width, normalize=normalize)
            for j in fx.range_constexpr(items):
                out[tid * items + j] = ys[j]

    @flyc.jit
    def _run(inp: fx.Tensor, out_rocdl: fx.Tensor, out_univ: fx.Tensor):
        _k_rocdl(inp, out_rocdl).launch(grid=(1, 1, 1), block=(block, 1, 1))
        _k_univ(inp, out_univ).launch(grid=(1, 1, 1), block=(block, 1, 1))

    torch.manual_seed(100 + width * 7 + items)
    x = torch.randn(total, dtype=torch.float32, device="cuda")
    out_r = torch.empty_like(x)
    out_u = torch.empty_like(x)
    _run(x, out_r, out_u)
    torch.cuda.synchronize()

    ref = _ref_fwht(x.view(-1, group_n), normalize=normalize).reshape(total).to(torch.float32)
    torch.testing.assert_close(out_r, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_u, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "dtype_spec,shape_spec",
    _matrix_cases(_DTYPE_SPECS, _BLOCK_SHAPE_ITEMS),
)
def test_block_fwht_gpu_accuracy_dtypes_and_involution(dtype_spec, shape_spec):
    if not torch.cuda.is_available():
        pytest.skip("GPU device required")

    fx_dtype, torch_dtype, rtol, atol = dtype_spec
    block_shape, items = shape_spec
    block_threads = block_shape[0] * block_shape[1] * block_shape[2]
    n = block_threads * items

    vec_map = {
        (fx.Float32, 2): fx.Float32x2,
        (fx.Float32, 4): fx.Float32x4,
        (fx.Float16, 2): fx.Float16x2,
        (fx.Float16, 4): fx.Float16x4,
        (fx.BFloat16, 2): fx.BFloat16x2,
        (fx.BFloat16, 4): fx.BFloat16x4,
    }

    @flyc.kernel
    def _k_forward(inp: fx.Pointer, out: fx.Pointer):
        fwht = fx.hadamard.BlockFWHT[fx_dtype, fx.known_block_size(), items]
        storage = fx.SharedAllocator().allocate(fwht.SharedStorage).peek()
        tid = fx.thread_idx.x + fx.thread_idx.y * block_shape[0] + fx.thread_idx.z * (block_shape[0] * block_shape[1])
        if fx.const_expr(items == 1):
            val = (inp + tid).load(fx_dtype)
            res = fwht(val, storage=storage, normalize=True)
            (out + tid).store(res)
        else:
            vec_t = vec_map[(fx_dtype, items)]
            val = (inp + tid * items).load(vec_t)
            res = fwht(val, storage=storage, normalize=True)
            (out + tid * items).store(res)

    @flyc.kernel
    def _k_univ(inp: fx.Pointer, out: fx.Pointer):
        fwht = fx.hadamard.universal.BlockFWHT[fx_dtype, fx.known_block_size(), items]
        storage = fx.SharedAllocator().allocate(fwht.SharedStorage).peek()
        tid = fx.thread_idx.x + fx.thread_idx.y * block_shape[0] + fx.thread_idx.z * (block_shape[0] * block_shape[1])
        if fx.const_expr(items == 1):
            val = (inp + tid).load(fx_dtype)
            res = fwht(val, storage=storage, normalize=True)
            (out + tid).store(res)
        else:
            vec_t = vec_map[(fx_dtype, items)]
            val = (inp + tid * items).load(vec_t)
            res = fwht(val, storage=storage, normalize=True)
            (out + tid * items).store(res)

    @flyc.jit
    def _run_fwd(inp: fx.Pointer, out: fx.Pointer):
        _k_forward(inp, out).launch(grid=(1, 1, 1), block=block_shape)

    @flyc.jit
    def _run_univ(inp: fx.Pointer, out: fx.Pointer):
        _k_univ(inp, out).launch(grid=(1, 1, 1), block=block_shape)

    torch.manual_seed(42 + n)
    x = (torch.randn(n, dtype=torch.float32, device="cuda") * 0.5).to(torch_dtype)
    y = torch.empty_like(x)
    y_univ = torch.empty_like(x)
    x_rec = torch.empty_like(x)

    x_ptr = flyc.from_c_void_p(fx_dtype, x.data_ptr())
    y_ptr = flyc.from_c_void_p(fx_dtype, y.data_ptr())
    yu_ptr = flyc.from_c_void_p(fx_dtype, y_univ.data_ptr())
    xr_ptr = flyc.from_c_void_p(fx_dtype, x_rec.data_ptr())

    _run_fwd(x_ptr, y_ptr)
    _run_univ(x_ptr, yu_ptr)
    _run_fwd(y_ptr, xr_ptr)
    torch.cuda.synchronize()

    ref = _ref_fwht(x, normalize=True).to(torch_dtype)
    torch.testing.assert_close(y, ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(y_univ, ref, rtol=rtol, atol=atol)
    # Self-inverse involution check: H_norm(H_norm(x)) == x.
    torch.testing.assert_close(x_rec, x, rtol=rtol * 2, atol=atol * 2)
