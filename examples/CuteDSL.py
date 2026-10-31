"""
CuTeDSL Python GPU-kernel syntax demo for static + dynamic algebra checks.

This script:
1) Runs static algebra cases in GPU kernel and prints results.
2) Runs dynamic one-value-changed cases one by one.
3) Never stops early: every case is attempted.

Error mode:
- ALGEBRA_CASE_ERROR_MODE=suppress (default): print failures and continue.
- ALGEBRA_CASE_ERROR_MODE=raise: run all cases, then raise if any failed.
"""

import os

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


def _cu_ok(ret):
    if ret[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {ret[0]}")
    if len(ret) == 1:
        return None
    if len(ret) == 2:
        return ret[1]
    return ret[1:]


def _launch_1x1(kernel_call, stream: cuda.CUstream):
    kernel_call.launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)


@cute.kernel
def algebra_static_kernel():
    cute.printf("=== Static Algebra Cases ===")
    ld_in_static = cute.make_layout((2, 3, 4), stride=(1, 2, 6))
    ld_tiler_static = cute.make_layout((2, 2, 2), stride=(1, 2, 4))
    cute.printf("logical_divide/static out  : {}", cute.logical_divide(ld_in_static, ld_tiler_static))

    lp_in_static = cute.make_layout((4, 2), stride=(1, 4))
    lp_tiler_static = cute.make_layout((2, 2), stride=(1, 2))
    cute.printf("logical_product/static out : {}", cute.logical_product(lp_in_static, lp_tiler_static))

    cp_lhs_static = cute.make_layout(16, stride=1)
    cp_rhs_static = cute.make_layout((2, 8), stride=(8, 1))
    cute.printf("composition/static out     : {}", cute.composition(cp_lhs_static, cp_rhs_static))

    cm_in_static = cute.make_layout((2, 4), stride=(1, 2))
    cm_ct_static = cute.make_layout(16, stride=1)
    cute.printf("complement/static out      : {}", cute.complement(cm_in_static, cm_ct_static))

    sz_static_layout = cute.make_layout((2, 3, 4), stride=(1, 2, 6))
    cute.printf("size/static value          : {}", cute.size(sz_static_layout))
    cute.printf("cosize/static value        : {}", cute.cosize(sz_static_layout))
    cute.printf("")


@cute.kernel
def ld_dyn_in_shape_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 3, v), stride=(1, 2, 6))
    tiler = cute.make_layout((2, 2, 2), stride=(1, 2, 4))
    cute.printf("logical_divide/dyn(in.shape)      : {}", cute.logical_divide(inp, tiler))


@cute.kernel
def ld_dyn_in_stride_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 3, 4), stride=(1, 2, v))
    tiler = cute.make_layout((2, 2, 2), stride=(1, 2, 4))
    cute.printf("logical_divide/dyn(in.stride)     : {}", cute.logical_divide(inp, tiler))


@cute.kernel
def ld_dyn_tiler_shape_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 3, 4), stride=(1, 2, 6))
    tiler = cute.make_layout((2, 2, v), stride=(1, 2, 4))
    cute.printf("logical_divide/dyn(tiler.shape)   : {}", cute.logical_divide(inp, tiler))


@cute.kernel
def ld_dyn_tiler_stride_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 3, 4), stride=(1, 2, 6))
    tiler = cute.make_layout((2, 2, 2), stride=(1, 2, v))
    cute.printf("logical_divide/dyn(tiler.stride)  : {}", cute.logical_divide(inp, tiler))


@cute.kernel
def lp_dyn_in_shape_kernel(v: cute.Int32):
    inp = cute.make_layout((4, v), stride=(1, 4))
    tiler = cute.make_layout((2, 2), stride=(1, 2))
    cute.printf("logical_product/dyn(in.shape)     : {}", cute.logical_product(inp, tiler))


@cute.kernel
def lp_dyn_in_stride_kernel(v: cute.Int32):
    inp = cute.make_layout((4, 2), stride=(1, v))
    tiler = cute.make_layout((2, 2), stride=(1, 2))
    cute.printf("logical_product/dyn(in.stride)    : {}", cute.logical_product(inp, tiler))


@cute.kernel
def lp_dyn_tiler_shape_kernel(v: cute.Int32):
    inp = cute.make_layout((4, 2), stride=(1, 4))
    tiler = cute.make_layout((2, v), stride=(1, 2))
    cute.printf("logical_product/dyn(tiler.shape)  : {}", cute.logical_product(inp, tiler))


@cute.kernel
def lp_dyn_tiler_stride_kernel(v: cute.Int32):
    inp = cute.make_layout((4, 2), stride=(1, 4))
    tiler = cute.make_layout((2, 2), stride=(1, v))
    cute.printf("logical_product/dyn(tiler.stride) : {}", cute.logical_product(inp, tiler))


@cute.kernel
def cp_dyn_lhs_shape_kernel(v: cute.Int32):
    lhs = cute.make_layout(v, stride=1)
    rhs = cute.make_layout((2, 8), stride=(8, 1))
    cute.printf("composition/dyn(lhs.shape)        : {}", cute.composition(lhs, rhs))


@cute.kernel
def cp_dyn_lhs_stride_kernel(v: cute.Int32):
    lhs = cute.make_layout(16, stride=v)
    rhs = cute.make_layout((2, 8), stride=(8, 1))
    cute.printf("composition/dyn(lhs.stride)       : {}", cute.composition(lhs, rhs))


@cute.kernel
def cp_dyn_rhs_shape_kernel(v: cute.Int32):
    lhs = cute.make_layout(16, stride=1)
    rhs = cute.make_layout((2, v), stride=(8, 1))
    cute.printf("composition/dyn(rhs.shape)        : {}", cute.composition(lhs, rhs))


@cute.kernel
def cp_dyn_rhs_stride_kernel(v: cute.Int32):
    lhs = cute.make_layout(16, stride=1)
    rhs = cute.make_layout((2, 8), stride=(v, 1))
    cute.printf("composition/dyn(rhs.stride)       : {}", cute.composition(lhs, rhs))


@cute.kernel
def cm_dyn_in_shape_kernel(v: cute.Int32):
    inp = cute.make_layout((2, v), stride=(1, 2))
    cotarget = cute.make_layout(16, stride=1)
    cute.printf("complement/dyn(input.shape)       : {}", cute.complement(inp, cotarget))


@cute.kernel
def cm_dyn_in_stride_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 4), stride=(1, v))
    cotarget = cute.make_layout(16, stride=1)
    cute.printf("complement/dyn(input.stride)      : {}", cute.complement(inp, cotarget))


@cute.kernel
def cm_dyn_ct_shape_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 4), stride=(1, 2))
    cotarget = cute.make_layout(v, stride=1)
    cute.printf("complement/dyn(cotarget.shape)    : {}", cute.complement(inp, cotarget))


@cute.kernel
def cm_dyn_ct_stride_kernel(v: cute.Int32):
    inp = cute.make_layout((2, 4), stride=(1, 2))
    cotarget = cute.make_layout(16, stride=v)
    cute.printf("complement/dyn(cotarget.stride)   : {}", cute.complement(inp, cotarget))


@cute.kernel
def sz_dyn_shape_kernel(v: cute.Int32):
    layout = cute.make_layout((2, 3, v), stride=(1, 2, 6))
    cute.printf("size/dyn(shape)={}, cosize/dyn(shape)={}", cute.size(layout), cute.cosize(layout))


@cute.kernel
def sz_dyn_stride_kernel(v: cute.Int32):
    layout = cute.make_layout((2, 3, 4), stride=(1, 2, v))
    cute.printf("size/dyn(stride)={}, cosize/dyn(stride)={}", cute.size(layout), cute.cosize(layout))


@cute.jit
def launch_static_case(stream: cuda.CUstream):
    _launch_1x1(algebra_static_kernel(), stream)


@cute.jit
def launch_ld_dyn_in_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(ld_dyn_in_shape_kernel(v), stream)


@cute.jit
def launch_ld_dyn_in_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(ld_dyn_in_stride_kernel(v), stream)


@cute.jit
def launch_ld_dyn_tiler_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(ld_dyn_tiler_shape_kernel(v), stream)


@cute.jit
def launch_ld_dyn_tiler_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(ld_dyn_tiler_stride_kernel(v), stream)


@cute.jit
def launch_lp_dyn_in_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(lp_dyn_in_shape_kernel(v), stream)


@cute.jit
def launch_lp_dyn_in_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(lp_dyn_in_stride_kernel(v), stream)


@cute.jit
def launch_lp_dyn_tiler_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(lp_dyn_tiler_shape_kernel(v), stream)


@cute.jit
def launch_lp_dyn_tiler_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(lp_dyn_tiler_stride_kernel(v), stream)


@cute.jit
def launch_cp_dyn_lhs_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cp_dyn_lhs_shape_kernel(v), stream)


@cute.jit
def launch_cp_dyn_lhs_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cp_dyn_lhs_stride_kernel(v), stream)


@cute.jit
def launch_cp_dyn_rhs_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cp_dyn_rhs_shape_kernel(v), stream)


@cute.jit
def launch_cp_dyn_rhs_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cp_dyn_rhs_stride_kernel(v), stream)


@cute.jit
def launch_cm_dyn_in_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cm_dyn_in_shape_kernel(v), stream)


@cute.jit
def launch_cm_dyn_in_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cm_dyn_in_stride_kernel(v), stream)


@cute.jit
def launch_cm_dyn_ct_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cm_dyn_ct_shape_kernel(v), stream)


@cute.jit
def launch_cm_dyn_ct_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(cm_dyn_ct_stride_kernel(v), stream)


@cute.jit
def launch_sz_dyn_shape(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(sz_dyn_shape_kernel(v), stream)


@cute.jit
def launch_sz_dyn_stride(stream: cuda.CUstream, *, v: cute.Int32):
    _launch_1x1(sz_dyn_stride_kernel(v), stream)


def _run_case(name: str, run_fn, failures: list):
    print(f"[RUN ] {name}")
    try:
        run_fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        failures.append((name, str(e)))


if __name__ == "__main__":
    _cu_ok(cuda.cuInit(0))
    dev = _cu_ok(cuda.cuDeviceGet(0))
    primary_ctx = _cu_ok(cuda.cuDevicePrimaryCtxRetain(dev))
    _cu_ok(cuda.cuCtxSetCurrent(primary_ctx))

    stream = cuda.CUstream(0)
    failures = []

    values = {
        "ld_in_shape": cutlass.Int32(4),
        "ld_in_stride": cutlass.Int32(6),
        "ld_tiler_shape": cutlass.Int32(2),
        "ld_tiler_stride": cutlass.Int32(4),
        "lp_in_shape": cutlass.Int32(2),
        "lp_in_stride": cutlass.Int32(4),
        "lp_tiler_shape": cutlass.Int32(2),
        "lp_tiler_stride": cutlass.Int32(2),
        "cp_lhs_shape": cutlass.Int32(16),
        "cp_lhs_stride": cutlass.Int32(1),
        "cp_rhs_shape": cutlass.Int32(8),
        "cp_rhs_stride": cutlass.Int32(8),
        "cm_in_shape": cutlass.Int32(4),
        "cm_in_stride": cutlass.Int32(2),
        "cm_ct_shape": cutlass.Int32(16),
        "cm_ct_stride": cutlass.Int32(1),
        "sz_shape": cutlass.Int32(4),
        "sz_stride": cutlass.Int32(6),
    }

    _run_case("static_all", lambda: launch_static_case(stream), failures)
    _run_case("logical_divide_in_shape", lambda: launch_ld_dyn_in_shape(stream, v=values["ld_in_shape"]), failures)
    _run_case("logical_divide_in_stride", lambda: launch_ld_dyn_in_stride(stream, v=values["ld_in_stride"]), failures)
    _run_case("logical_divide_tiler_shape", lambda: launch_ld_dyn_tiler_shape(stream, v=values["ld_tiler_shape"]), failures)
    _run_case("logical_divide_tiler_stride", lambda: launch_ld_dyn_tiler_stride(stream, v=values["ld_tiler_stride"]), failures)
    _run_case("logical_product_in_shape", lambda: launch_lp_dyn_in_shape(stream, v=values["lp_in_shape"]), failures)
    _run_case("logical_product_in_stride", lambda: launch_lp_dyn_in_stride(stream, v=values["lp_in_stride"]), failures)
    _run_case("logical_product_tiler_shape", lambda: launch_lp_dyn_tiler_shape(stream, v=values["lp_tiler_shape"]), failures)
    _run_case("logical_product_tiler_stride", lambda: launch_lp_dyn_tiler_stride(stream, v=values["lp_tiler_stride"]), failures)
    _run_case("composition_lhs_shape", lambda: launch_cp_dyn_lhs_shape(stream, v=values["cp_lhs_shape"]), failures)
    _run_case("composition_lhs_stride", lambda: launch_cp_dyn_lhs_stride(stream, v=values["cp_lhs_stride"]), failures)
    _run_case("composition_rhs_shape", lambda: launch_cp_dyn_rhs_shape(stream, v=values["cp_rhs_shape"]), failures)
    _run_case("composition_rhs_stride", lambda: launch_cp_dyn_rhs_stride(stream, v=values["cp_rhs_stride"]), failures)
    _run_case("complement_in_shape", lambda: launch_cm_dyn_in_shape(stream, v=values["cm_in_shape"]), failures)
    _run_case("complement_in_stride", lambda: launch_cm_dyn_in_stride(stream, v=values["cm_in_stride"]), failures)
    _run_case("complement_cotarget_shape", lambda: launch_cm_dyn_ct_shape(stream, v=values["cm_ct_shape"]), failures)
    _run_case("complement_cotarget_stride", lambda: launch_cm_dyn_ct_stride(stream, v=values["cm_ct_stride"]), failures)
    _run_case("size_cosize_shape", lambda: launch_sz_dyn_shape(stream, v=values["sz_shape"]), failures)
    _run_case("size_cosize_stride", lambda: launch_sz_dyn_stride(stream, v=values["sz_stride"]), failures)

    print("")
    print(f"case summary: total={19}, passed={19 - len(failures)}, failed={len(failures)}")
    if failures:
        print("failed cases:")
        for name, err in failures:
            print(f"- {name}: {err}")

    if failures and os.getenv("ALGEBRA_CASE_ERROR_MODE", "suppress").lower() == "raise":
        raise RuntimeError(f"{len(failures)} algebra case(s) failed")
