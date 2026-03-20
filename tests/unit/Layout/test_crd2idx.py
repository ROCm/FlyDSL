import os
import subprocess
import sys

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx


# --- jit kernels (not collected by pytest due to _run_ prefix) ---


@flyc.jit
def _run_crd2idx_col_major():
    """(4,8) col-major: idx = r + 4*c"""
    layout = fx.make_layout((4, 8), (1, 4))
    for r in range_constexpr(4):
        for c in range_constexpr(8):
            idx = fx.crd2idx(fx.make_coord((r, c)), layout)
            fx.printf("{}", idx)


@flyc.jit
def _run_crd2idx_row_major():
    """(4,8) row-major: idx = r*8 + c"""
    layout = fx.make_layout((4, 8), (8, 1))
    for r in range_constexpr(4):
        for c in range_constexpr(8):
            idx = fx.crd2idx(fx.make_coord((r, c)), layout)
            fx.printf("{}", idx)


@flyc.jit
def _run_crd2idx_1d():
    """1D layout: shape=(8,), stride=(2,)"""
    layout = fx.make_layout((8,), (2,))
    for c in range_constexpr(8):
        idx = fx.crd2idx(fx.make_coord((c,)), layout)
        fx.printf("{}", idx)


@flyc.jit
def _run_crd2idx_3d():
    """3D layout: (2,3,4) row-major strides (12,4,1)"""
    layout = fx.make_layout((2, 3, 4), (12, 4, 1))
    for i in range_constexpr(2):
        for j in range_constexpr(3):
            for k in range_constexpr(4):
                idx = fx.crd2idx(fx.make_coord((i, j, k)), layout)
                fx.printf("{}", idx)


# --- subprocess helper to capture C-level printf ---

JIT_KERNELS = {
    "col_major": _run_crd2idx_col_major,
    "row_major": _run_crd2idx_row_major,
    "1d": _run_crd2idx_1d,
    "3d": _run_crd2idx_3d,
}

EXPECTED = {
    "col_major": [r + 4 * c for r in range(4) for c in range(8)],
    "row_major": [r * 8 + c for r in range(4) for c in range(8)],
    "1d": [c * 2 for c in range(8)],
    "3d": [i * 12 + j * 4 + k for i in range(2) for j in range(3) for k in range(4)],
}


def _run_jit_and_capture(test_name):
    """Run a jit kernel in a subprocess and return parsed int output."""
    env = os.environ.copy()
    env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    result = subprocess.run(
        [sys.executable, __file__, "--run", test_name],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    return [int(x) for x in lines]


def _run_error_test(snippet_name):
    """Run an error-test snippet in a subprocess, return (returncode, stderr)."""
    env = os.environ.copy()
    env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    result = subprocess.run(
        [sys.executable, __file__, "--error", snippet_name],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# --- pytest test cases: correctness ---


def test_crd2idx_col_major():
    """(4,8) col-major layout: idx = r + 4*c"""
    actual = _run_jit_and_capture("col_major")
    assert actual == EXPECTED["col_major"]


def test_crd2idx_row_major():
    """(4,8) row-major layout: idx = r*8 + c"""
    actual = _run_jit_and_capture("row_major")
    assert actual == EXPECTED["row_major"]


def test_crd2idx_1d():
    """1D layout: shape=(8,), stride=(2,)"""
    actual = _run_jit_and_capture("1d")
    assert actual == EXPECTED["1d"]


def test_crd2idx_3d():
    """3D layout: (2,3,4) row-major strides (12,4,1)"""
    actual = _run_jit_and_capture("3d")
    assert actual == EXPECTED["3d"]


# --- pytest test cases: error handling (via subprocess to avoid conftest path issues) ---


def test_make_coord_rejects_varargs():
    """make_coord(r, c) must raise TypeError."""
    rc, stdout, stderr = _run_error_test("varargs")
    assert rc != 0, f"Expected failure but succeeded.\nstdout: {stdout}"
    assert "make_coord expects a tuple" in stderr, f"Wrong error message:\n{stderr}"


def test_make_coord_rejects_int():
    """make_coord(42) must raise TypeError."""
    rc, stdout, stderr = _run_error_test("int_arg")
    assert rc != 0, f"Expected failure but succeeded.\nstdout: {stdout}"
    assert "make_coord expects a tuple" in stderr, f"Wrong error message:\n{stderr}"


# --- subprocess entry point ---

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        JIT_KERNELS[sys.argv[2]]()

    elif len(sys.argv) >= 3 and sys.argv[1] == "--error":
        # These are intentionally broken snippets that should raise TypeError.
        # Import fresh (no conftest interference).
        import flydsl.compiler as flyc_fresh
        import flydsl.expr as fx_fresh

        if sys.argv[2] == "varargs":

            @flyc_fresh.jit
            def _bad():
                layout = fx_fresh.make_layout((4, 8), (1, 4))
                idx = fx_fresh.crd2idx(fx_fresh.make_coord(0, 1), layout)

            _bad()

        elif sys.argv[2] == "int_arg":

            @flyc_fresh.jit
            def _bad():
                layout = fx_fresh.make_layout((4,), (1,))
                idx = fx_fresh.crd2idx(fx_fresh.make_coord(0), layout)

            _bad()

    else:
        pytest.main([__file__, "-v"])
