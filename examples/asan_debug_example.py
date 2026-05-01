#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
FlyDSL AddressSanitizer (ASan) Example
=====================================

This example demonstrates how to use ROCm ASan with FlyDSL for GPU memory
debugging. ASan can detect:
- Out-of-bounds memory accesses
- Use-after-free errors
- Memory leaks

Requirements:
- AMD GPU with xnack+ support (e.g., gfx90a, gfx942)
- ROCm 6.0+ with ASan support
- Set HSA_XNACK=1 to enable xnack mode

Usage:
    # Run with ASan disabled (default)
    python asan_debug_example.py

    # Run with ASan enabled
    FLYDSL_DEBUG_ENABLE_ASAN=1 python asan_debug_example.py

    # Run with ASan and debug info
    FLYDSL_DEBUG_ENABLE_ASAN=1 FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 python asan_debug_example.py

Environment Variables:
    FLYDSL_DEBUG_ENABLE_ASAN=1     - Enable AddressSanitizer
    FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 - Generate debug info for better ASan reports
    HSA_XNACK=1                    - Required: Enable GPU xnack mode
    LD_LIBRARY_PATH                - Should include path to libasan.so

Note:
    ASan adds significant overhead (~3x slower execution, ~2x memory).
    Only use for debugging, not production.
"""

import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.utils import env


def print_asan_status():
    """Print current ASan configuration."""
    print("=" * 60)
    print("AddressSanitizer Configuration")
    print("=" * 60)
    print(f"FLYDSL_DEBUG_ENABLE_ASAN: {env.debug.enable_asan}")
    print(f"FLYDSL_DEBUG_ENABLE_DEBUG_INFO: {env.debug.enable_debug_info}")
    print(f"HSA_XNACK: {os.environ.get('HSA_XNACK', 'not set')}")
    print(f"LD_LIBRARY_PATH contains asan: {'asan' in os.environ.get('LD_LIBRARY_PATH', '').lower()}")
    print("=" * 60)


@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
):
    """Simple vector addition kernel."""
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    idx = bid * 256 + tid

    if idx < N:
        # Normal access - should be safe
        C[idx] = A[idx] + B[idx]


@flyc.jit
def vector_add_safe(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    """Safe vector addition with proper bounds checking."""
    vector_add_kernel(A, B, C, N).launch(
        grid=((N + 255) // 256,),
        block=(256,),
        stream=stream,
    )


@flyc.kernel
def oob_access_kernel(data: fx.Tensor, size: fx.Constexpr[int]):
    """Kernel with intentional out-of-bounds access for ASan testing.

    WARNING: This kernel deliberately accesses memory beyond the allocated
    buffer to test ASan detection. Without ASan, this may silently corrupt
    memory or crash. With ASan, it will report an error.
    """
    tid = gpu.thread_idx.x

    # This is an intentional out-of-bounds access for testing
    # In real code, this would be a bug!
    if tid == 0:
        # Access beyond the allocated size
        # ASan should detect this if enabled
        oob_index = size + 100  # Deliberately out of bounds
        # Note: This will only be caught by ASan if the access
        # actually goes beyond the allocation boundary
        data[oob_index] = 999.0


@flyc.jit
def test_oob_access(
    data: fx.Tensor,
    size: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    """Test out-of-bounds access detection."""
    oob_access_kernel(data, size).launch(
        grid=(1,),
        block=(64,),
        stream=stream,
    )


def run_safe_example():
    """Run the safe vector addition example."""
    print("\nRunning safe vector addition example...")
    print("-" * 60)

    N = 1024
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty(N, device="cuda", dtype=torch.float32)

    vector_add_safe(A, B, C, N)

    # Verify results
    expected = A + B
    max_error = torch.max(torch.abs(C - expected)).item()
    print(f"Vector addition completed. Max error: {max_error:.6e}")

    if max_error < 1e-5:
        print("✓ Results are correct!")
    else:
        print("✗ Results are incorrect!")

    return max_error < 1e-5


def run_oob_test():
    """Run the out-of-bounds access test."""
    print("\nRunning out-of-bounds access test...")
    print("-" * 60)

    if not env.debug.enable_asan:
        print("WARNING: ASan is not enabled (FLYDSL_DEBUG_ENABLE_ASAN=1)")
        print("This test may crash or corrupt memory without ASan!")
        print("Skipping OOB test for safety.")
        return None

    size = 256
    data = torch.zeros(size, device="cuda", dtype=torch.float32)

    print(f"Allocated buffer of {size} floats")
    print("Attempting out-of-bounds access...")
    print("(ASan should detect and report this)")

    try:
        test_oob_access(data, size)
        print("Test completed - check ASan output above for errors")
        return True
    except Exception as e:
        print(f"Exception caught: {e}")
        return False


def check_xnack_support():
    """Check if xnack is supported on current GPU."""
    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = get_rocm_arch()
        # xnack is supported on gfx90a+, gfx94x, gfx95x
        supported_archs = ["gfx90a", "gfx942", "gfx950", "gfx951", "gfx952"]
        is_supported = any(a in arch for a in supported_archs)

        print(f"Detected GPU architecture: {arch}")
        print(f"xnack support: {'Yes' if is_supported else 'Unknown/No'}")

        return is_supported
    except Exception as e:
        print(f"Could not detect GPU architecture: {e}")
        return False


def main():
    """Main entry point."""
    print("FlyDSL AddressSanitizer Example")
    print("=" * 60)

    # Print configuration
    print_asan_status()

    # Check GPU support
    print("\nChecking GPU support...")
    check_xnack_support()

    # Check environment
    if env.debug.enable_asan and os.environ.get("HSA_XNACK") != "1":
        print("\nWARNING: FLYDSL_DEBUG_ENABLE_ASAN is set but HSA_XNACK=1 is not set!")
        print("ASan requires xnack mode to be enabled. Set HSA_XNACK=1")

    # Run examples
    safe_result = run_safe_example()

    oob_result = run_oob_test()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Safe example: {'PASSED' if safe_result else 'FAILED'}")
    if oob_result is not None:
        print(f"OOB test: {'COMPLETED' if oob_result else 'ERROR'}")
    else:
        print("OOB test: SKIPPED (ASan not enabled)")

    return 0 if safe_result else 1


if __name__ == "__main__":
    sys.exit(main())
