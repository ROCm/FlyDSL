"""
FlyDSL GPU-kernel algebra demo for static + dynamic cases.

This script mirrors CuteDSL.py case settings:
1) one static-all case
2) dynamic one-value-changed cases
3) never stops early: every case is attempted

Error mode:
- ALGEBRA_CASE_ERROR_MODE=suppress (default): print failures and continue.
- ALGEBRA_CASE_ERROR_MODE=raise: run all cases, then raise if any failed.
"""

import multiprocessing as mp
import os

import flydsl.compiler as fmc
import flydsl.expr as fx


def _launch_1x1(kernel_call, stream: fx.Stream):
    kernel_call.launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


@fmc.kernel
def algebra_static_kernel():
    fx.printf("=== Static Algebra Cases ===")
    ld_in_static = fx.make_layout((2, 3, 4), (1, 2, 6))
    ld_tiler_static = fx.make_layout((2, 2, 2), (1, 2, 4))
    fx.printf("logical_divide/static out  : {}", fx.logical_divide(ld_in_static, ld_tiler_static))

    lp_in_static = fx.make_layout((4, 2), (1, 4))
    lp_tiler_static = fx.make_layout((2, 2), (1, 2))
    fx.printf("logical_product/static out : {}", fx.logical_product(lp_in_static, lp_tiler_static))

    cp_lhs_static = fx.make_layout(16, 1)
    cp_rhs_static = fx.make_layout((2, 8), (8, 1))
    fx.printf("composition/static out     : {}", fx.composition(cp_lhs_static, cp_rhs_static))

    cm_in_static = fx.make_layout((2, 4), (1, 2))
    fx.printf("complement/static out      : {}", fx.complement(cm_in_static, 16))

    sz_static_layout = fx.make_layout((2, 3, 4), (1, 2, 6))
    fx.printf("size/static value          : {}", fx.size(sz_static_layout))
    fx.printf("cosize/static value        : {}", fx.cosize(sz_static_layout))
    fx.printf("")


@fmc.kernel
def ld_dyn_in_shape_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 3, v), (1, 2, 6))
    tiler = fx.make_layout((2, 2, 2), (1, 2, 4))
    fx.printf("logical_divide/dyn(in.shape)      : {}", fx.logical_divide(inp, tiler))


@fmc.kernel
def ld_dyn_in_stride_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 3, 4), (1, 2, v))
    tiler = fx.make_layout((2, 2, 2), (1, 2, 4))
    fx.printf("logical_divide/dyn(in.stride)     : {}", fx.logical_divide(inp, tiler))


@fmc.kernel
def ld_dyn_tiler_shape_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 3, 4), (1, 2, 6))
    tiler = fx.make_layout((2, 2, v), (1, 2, 4))
    fx.printf("logical_divide/dyn(tiler.shape)   : {}", fx.logical_divide(inp, tiler))


@fmc.kernel
def ld_dyn_tiler_stride_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 3, 4), (1, 2, 6))
    tiler = fx.make_layout((2, 2, 2), (1, 2, v))
    fx.printf("logical_divide/dyn(tiler.stride)  : {}", fx.logical_divide(inp, tiler))


@fmc.kernel
def lp_dyn_in_shape_kernel(v: fx.Int32):
    inp = fx.make_layout((4, v), (1, 4))
    tiler = fx.make_layout((2, 2), (1, 2))
    fx.printf("logical_product/dyn(in.shape)     : {}", fx.logical_product(inp, tiler))


@fmc.kernel
def lp_dyn_in_stride_kernel(v: fx.Int32):
    inp = fx.make_layout((4, 2), (1, v))
    tiler = fx.make_layout((2, 2), (1, 2))
    fx.printf("logical_product/dyn(in.stride)    : {}", fx.logical_product(inp, tiler))


@fmc.kernel
def lp_dyn_tiler_shape_kernel(v: fx.Int32):
    inp = fx.make_layout((4, 2), (1, 4))
    tiler = fx.make_layout((2, v), (1, 2))
    fx.printf("logical_product/dyn(tiler.shape)  : {}", fx.logical_product(inp, tiler))


@fmc.kernel
def lp_dyn_tiler_stride_kernel(v: fx.Int32):
    inp = fx.make_layout((4, 2), (1, 4))
    tiler = fx.make_layout((2, 2), (1, v))
    fx.printf("logical_product/dyn(tiler.stride) : {}", fx.logical_product(inp, tiler))


@fmc.kernel
def cp_dyn_lhs_shape_kernel(v: fx.Int32):
    lhs = fx.make_layout(v, 1)
    rhs = fx.make_layout((2, 8), (8, 1))
    fx.printf("composition/dyn(lhs.shape)        : {}", fx.composition(lhs, rhs))


@fmc.kernel
def cp_dyn_lhs_stride_kernel(v: fx.Int32):
    lhs = fx.make_layout(16, v)
    rhs = fx.make_layout((2, 8), (8, 1))
    fx.printf("composition/dyn(lhs.stride)       : {}", fx.composition(lhs, rhs))


@fmc.kernel
def cp_dyn_rhs_shape_kernel(v: fx.Int32):
    lhs = fx.make_layout(16, 1)
    rhs = fx.make_layout((2, v), (8, 1))
    fx.printf("composition/dyn(rhs.shape)        : {}", fx.composition(lhs, rhs))


@fmc.kernel
def cp_dyn_rhs_stride_kernel(v: fx.Int32):
    lhs = fx.make_layout(16, 1)
    rhs = fx.make_layout((2, 8), (v, 1))
    fx.printf("composition/dyn(rhs.stride)       : {}", fx.composition(lhs, rhs))


@fmc.kernel
def cm_dyn_in_shape_kernel(v: fx.Int32):
    inp = fx.make_layout((2, v), (1, 2))
    cotarget = fx.make_layout(16, 1)
    fx.printf("complement/dyn(input.shape)       : {}", fx.complement(inp, cotarget))


@fmc.kernel
def cm_dyn_in_stride_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 4), (1, v))
    cotarget = fx.make_layout(16, 1)
    fx.printf("complement/dyn(input.stride)      : {}", fx.complement(inp, cotarget))


@fmc.kernel
def cm_dyn_ct_shape_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 4), (1, 2))
    cotarget = fx.make_layout(v, 1)
    fx.printf("complement/dyn(cotarget.shape)    : {}", fx.complement(inp, cotarget))


@fmc.kernel
def cm_dyn_ct_stride_kernel(v: fx.Int32):
    inp = fx.make_layout((2, 4), (1, 2))
    cotarget = fx.make_layout(16, v)
    fx.printf("complement/dyn(cotarget.stride)   : {}", fx.complement(inp, cotarget))


@fmc.kernel
def sz_dyn_shape_kernel(v: fx.Int32):
    layout = fx.make_layout((2, 3, v), (1, 2, 6))
    fx.printf("size/dyn(shape)={}, cosize/dyn(shape)={}", fx.size(layout), fx.cosize(layout))


@fmc.kernel
def sz_dyn_stride_kernel(v: fx.Int32):
    layout = fx.make_layout((2, 3, 4), (1, 2, v))
    fx.printf("size/dyn(stride)={}, cosize/dyn(stride)={}", fx.size(layout), fx.cosize(layout))


@fmc.jit
def launch_static_case(stream: fx.Stream):
    _launch_1x1(algebra_static_kernel(), stream)


@fmc.jit
def launch_ld_dyn_in_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(ld_dyn_in_shape_kernel(v), stream)


@fmc.jit
def launch_ld_dyn_in_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(ld_dyn_in_stride_kernel(v), stream)


@fmc.jit
def launch_ld_dyn_tiler_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(ld_dyn_tiler_shape_kernel(v), stream)


@fmc.jit
def launch_ld_dyn_tiler_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(ld_dyn_tiler_stride_kernel(v), stream)


@fmc.jit
def launch_lp_dyn_in_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(lp_dyn_in_shape_kernel(v), stream)


@fmc.jit
def launch_lp_dyn_in_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(lp_dyn_in_stride_kernel(v), stream)


@fmc.jit
def launch_lp_dyn_tiler_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(lp_dyn_tiler_shape_kernel(v), stream)


@fmc.jit
def launch_lp_dyn_tiler_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(lp_dyn_tiler_stride_kernel(v), stream)


@fmc.jit
def launch_cp_dyn_lhs_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cp_dyn_lhs_shape_kernel(v), stream)


@fmc.jit
def launch_cp_dyn_lhs_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cp_dyn_lhs_stride_kernel(v), stream)


@fmc.jit
def launch_cp_dyn_rhs_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cp_dyn_rhs_shape_kernel(v), stream)


@fmc.jit
def launch_cp_dyn_rhs_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cp_dyn_rhs_stride_kernel(v), stream)


@fmc.jit
def launch_cm_dyn_in_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cm_dyn_in_shape_kernel(v), stream)


@fmc.jit
def launch_cm_dyn_in_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cm_dyn_in_stride_kernel(v), stream)


@fmc.jit
def launch_cm_dyn_ct_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cm_dyn_ct_shape_kernel(v), stream)


@fmc.jit
def launch_cm_dyn_ct_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(cm_dyn_ct_stride_kernel(v), stream)


@fmc.jit
def launch_sz_dyn_shape(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(sz_dyn_shape_kernel(v), stream)


@fmc.jit
def launch_sz_dyn_stride(stream: fx.Stream, v: fx.Int32):
    _launch_1x1(sz_dyn_stride_kernel(v), stream)


VALUES = {
    "ld_in_shape": 4,
    "ld_in_stride": 6,
    "ld_tiler_shape": 2,
    "ld_tiler_stride": 4,
    "lp_in_shape": 2,
    "lp_in_stride": 4,
    "lp_tiler_shape": 2,
    "lp_tiler_stride": 2,
    "cp_lhs_shape": 16,
    "cp_lhs_stride": 1,
    "cp_rhs_shape": 8,
    "cp_rhs_stride": 8,
    "cm_in_shape": 4,
    "cm_in_stride": 2,
    "cm_ct_shape": 16,
    "cm_ct_stride": 1,
    "sz_shape": 4,
    "sz_stride": 6,
}


def _run_single_case(stream: fx.Stream, case_name: str, value: int | None):
    if case_name == "static_all":
        launch_static_case(stream)
        return
    if case_name == "logical_divide_in_shape":
        launch_ld_dyn_in_shape(stream, value)
        return
    if case_name == "logical_divide_in_stride":
        launch_ld_dyn_in_stride(stream, value)
        return
    if case_name == "logical_divide_tiler_shape":
        launch_ld_dyn_tiler_shape(stream, value)
        return
    if case_name == "logical_divide_tiler_stride":
        launch_ld_dyn_tiler_stride(stream, value)
        return
    if case_name == "logical_product_in_shape":
        launch_lp_dyn_in_shape(stream, value)
        return
    if case_name == "logical_product_in_stride":
        launch_lp_dyn_in_stride(stream, value)
        return
    if case_name == "logical_product_tiler_shape":
        launch_lp_dyn_tiler_shape(stream, value)
        return
    if case_name == "logical_product_tiler_stride":
        launch_lp_dyn_tiler_stride(stream, value)
        return
    if case_name == "composition_lhs_shape":
        launch_cp_dyn_lhs_shape(stream, value)
        return
    if case_name == "composition_lhs_stride":
        launch_cp_dyn_lhs_stride(stream, value)
        return
    if case_name == "composition_rhs_shape":
        launch_cp_dyn_rhs_shape(stream, value)
        return
    if case_name == "composition_rhs_stride":
        launch_cp_dyn_rhs_stride(stream, value)
        return
    if case_name == "complement_in_shape":
        launch_cm_dyn_in_shape(stream, value)
        return
    if case_name == "complement_in_stride":
        launch_cm_dyn_in_stride(stream, value)
        return
    if case_name == "complement_cotarget_shape":
        launch_cm_dyn_ct_shape(stream, value)
        return
    if case_name == "complement_cotarget_stride":
        launch_cm_dyn_ct_stride(stream, value)
        return
    if case_name == "size_cosize_shape":
        launch_sz_dyn_shape(stream, value)
        return
    if case_name == "size_cosize_stride":
        launch_sz_dyn_stride(stream, value)
        return
    raise ValueError(f"unknown case: {case_name}")


def _run_case(name: str, stream: fx.Stream, value: int | None):
    print(f"[RUN ] {name}")
    try:
        _run_single_case(stream, name, value)
        print(f"[PASS] {name}")
        return True, ""
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False, str(e)


def _run_case_child(name: str, value: int | None, result_queue):
    stream = fx.Stream(None)
    ok, err = _run_case(name, stream, value)
    result_queue.put((ok, err))


def _run_case_isolated(name: str, value: int | None, failures: list):
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_run_case_child, args=(name, value, result_queue))
    proc.start()
    proc.join()

    if proc.exitcode == 0:
        if not result_queue.empty():
            ok, err = result_queue.get()
            if ok:
                return
            failures.append((name, err))
            return
        failures.append((name, "no child result"))
        print(f"[FAIL] {name}: no child result")
        return

    err = f"child exited abnormally: exitcode={proc.exitcode}"
    failures.append((name, err))
    print(f"[FAIL] {name}: {err}")


if __name__ == "__main__":
    failures = []
    cases = [
        ("static_all", None),
        ("logical_divide_in_shape", VALUES["ld_in_shape"]),
        ("logical_divide_in_stride", VALUES["ld_in_stride"]),
        ("logical_divide_tiler_shape", VALUES["ld_tiler_shape"]),
        ("logical_divide_tiler_stride", VALUES["ld_tiler_stride"]),
        ("logical_product_in_shape", VALUES["lp_in_shape"]),
        ("logical_product_in_stride", VALUES["lp_in_stride"]),
        ("logical_product_tiler_shape", VALUES["lp_tiler_shape"]),
        ("logical_product_tiler_stride", VALUES["lp_tiler_stride"]),
        ("composition_lhs_shape", VALUES["cp_lhs_shape"]),
        ("composition_lhs_stride", VALUES["cp_lhs_stride"]),
        ("composition_rhs_shape", VALUES["cp_rhs_shape"]),
        ("composition_rhs_stride", VALUES["cp_rhs_stride"]),
        ("complement_in_shape", VALUES["cm_in_shape"]),
        ("complement_in_stride", VALUES["cm_in_stride"]),
        ("complement_cotarget_shape", VALUES["cm_ct_shape"]),
        ("complement_cotarget_stride", VALUES["cm_ct_stride"]),
        ("size_cosize_shape", VALUES["sz_shape"]),
        ("size_cosize_stride", VALUES["sz_stride"]),
    ]

    for case_name, case_value in cases:
        _run_case_isolated(case_name, case_value, failures)

    print("")
    total = len(cases)
    print(f"case summary: total={total}, passed={total - len(failures)}, failed={len(failures)}")
    if failures:
        print("failed cases:")
        for name, err in failures:
            print(f"- {name}: {err}")

    if failures and os.getenv("ALGEBRA_CASE_ERROR_MODE", "suppress").lower() == "raise":
        raise RuntimeError(f"{len(failures)} algebra case(s) failed")
