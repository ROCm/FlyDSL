#!/usr/bin/env python3
"""Regression tests for dynamic shared memory argument typing."""

import argparse
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def smem_probe_kernel():
    # Keep the kernel minimal: this test focuses on launch-time smem typing.
    fx.printf("[smem_probe] tid={}", fx.thread_idx.x)


@flyc.jit
def run_with_fx_int32_smem(smem: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    smem_probe_kernel().launch(grid=(1, 1, 1), block=[1, 1, 1], smem=smem, stream=stream)


@flyc.jit
def run_with_constexpr_smem(smem: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    smem_probe_kernel().launch(grid=(1, 1, 1), block=[1, 1, 1], smem=smem, stream=stream)


@flyc.jit
def run_with_python_int_smem(smem: int, stream: fx.Stream = fx.Stream(None)):
    smem_probe_kernel().launch(grid=(1, 1, 1), block=[1, 1, 1], smem=smem, stream=stream)


@pytest.mark.parametrize("smem_size", [0, 64, 128])
def test_dynamic_smem_fx_int32(smem_size):
    try:
        run_with_fx_int32_smem(smem_size)
    except Exception as e:
        pytest.fail(f"fx.Int32 smem failed for smem={smem_size}: {type(e).__name__}: {e}")


@pytest.mark.parametrize("smem_size", [0, 64, 128])
def test_dynamic_smem_constexpr_int(smem_size):
    try:
        run_with_constexpr_smem(smem_size)
    except Exception as e:
        pytest.fail(f"fx.Constexpr[int] smem failed for smem={smem_size}: {type(e).__name__}: {e}")


@pytest.mark.parametrize("smem_size", [0, 64, 128])
def test_dynamic_smem_python_int(smem_size):
    try:
        run_with_python_int_smem(smem_size)
    except Exception as e:
        pytest.fail(f"python int smem failed for smem={smem_size}: {type(e).__name__}: {e}")


def _run_one(smem_type: str, smem_size: int):
    if smem_type == "fx_int32":
        run_with_fx_int32_smem(smem_size)
    elif smem_type == "constexpr_int":
        run_with_constexpr_smem(smem_size)
    elif smem_type == "python_int":
        run_with_python_int_smem(smem_size)
    else:
        raise ValueError(f"Unsupported smem_type: {smem_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Run dynamic shared memory typing checks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--smem_type",
        type=str,
        default="all",
        choices=["fx_int32", "constexpr_int", "python_int", "all"],
        help="Which smem annotation case to run",
    )
    parser.add_argument(
        "--smem_size",
        type=int,
        action="append",
        default=None,
        help="smem size in bytes; pass multiple times to run multiple values",
    )
    args = parser.parse_args()

    smem_sizes = args.smem_size if args.smem_size else [0, 64, 128]
    smem_types = (
        ["fx_int32", "constexpr_int", "python_int"]
        if args.smem_type == "all"
        else [args.smem_type]
    )

    print("=" * 72)
    print("FlyDSL Dynamic Shared Memory Typing Check")
    print("=" * 72)
    print(f"  smem_types: {smem_types}")
    print(f"  smem_sizes: {smem_sizes}")
    print("=" * 72)

    has_error = False
    for st in smem_types:
        for sz in smem_sizes:
            try:
                _run_one(st, sz)
                print(f"[OK] smem_type={st}, smem_size={sz}")
            except Exception as e:
                has_error = True
                print(f"[FAIL] smem_type={st}, smem_size={sz}: {type(e).__name__}: {e}")

    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()

