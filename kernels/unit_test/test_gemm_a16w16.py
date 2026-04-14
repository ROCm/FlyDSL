# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Unit tests for A16W16 GEMM kernels.

Kernel implementation lives in `kernels/gemm_a16w16_gfx1250.py`.
This file is the correctness harness.

pytest -k filter:
    pytest ... -k "activation"      # only activation tests
    pytest ... -k "bias"            # only bias tests
    pytest ... -k "layout"          # only layout tests
    pytest ... -k "not activation"  # skip activation tests

Default: runs all tests where available.
Tests are automatically skipped if gfx1250 is not detected.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# workaround for simulator
import flydsl  # noqa: E402,F401 -- preload system comgr before torch/HIP loads LLVM

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm_a16w16_gfx1250 import gemm_a16w16
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _check_gfx1250():
    arch = str(get_rocm_arch(timeout_s=300))
    if not arch.startswith("gfx1250"):
        pytest.skip(f"gemm_a16w16 requires gfx1250, got {arch}")


def _check_min_k(K, tile_k=32, num_buffers=2):
    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    num_k_tiles = K_padded // tile_k
    if num_k_tiles < num_buffers - 1:
        pytest.skip(
            f"{num_buffers}-stage pipeline requires num_k_tiles >= {num_buffers - 1}, "
            f"got K={K} (K_padded={K_padded}, num_k_tiles={num_k_tiles})"
        )


def get_x_vals():
    return [
        (1, 1, 1),
        (1, 16, 16),
        (16, 1, 16),
        (16, 16, 1),
        # Irregular shapes (masking & OOB)
        (3, 5, 7),
        (17, 33, 65),
        (63, 127, 255),
        (65, 129, 257),
        #
        (64, 64, 64),
        (128, 128, 128),
        # Multiple blocks
        (128, 256, 512),
        (256, 512, 256),
        # Asymmetric shapes
        (32, 256, 128),
        (256, 32, 128),
        (128, 128, 1024),
        (1024, 128, 128),
        (1536, 512, 768),
    ]


def get_fewer_x_vals():
    return [
        (64, 64, 64),
        (128, 256, 512),
        (256, 512, 256),
        (128, 128, 1024),
        (1024, 128, 128),
        (1536, 512, 768),
    ]


def get_x_vals_large():
    return [
        (1024, 1024, 1024),
        (2048, 2048, 1024),
    ]


def _generate_inputs(M, N, K, dtype, layout="TN"):
    """Generate random input tensors for GEMM: Y = X @ W^T + bias.

    Layout convention (matches F.linear / triton tests):
        First letter  = X layout:  T = (M,K) contiguous, N = (K,M).T view
        Second letter = W layout:  N = (N,K) contiguous, T = (K,N).T view
    """
    torch.manual_seed(0)

    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "N":
        w = torch.randn((N, K), dtype=dtype, device="cuda")
    else:
        w = torch.randn((K, N), dtype=dtype, device="cuda").T

    return x, w


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16(M, N, K, dtype):
    """Basic GEMM correctness: Y = X @ W^T."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype)
    ref = torch.mm(x.float(), w.float().T).to(torch.float32)

    out = gemm_a16w16(x, w, dtype=dtype)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_bias(M, N, K, dtype):
    """GEMM with bias: Y = X @ W^T + bias."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype)
    bias = torch.randn((N,), dtype=dtype, device="cuda")
    ref = (torch.mm(x.float(), w.float().T) + bias.float()).to(torch.float32)

    out = gemm_a16w16(x, w, bias=bias, dtype=dtype)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_preallocated_output(M, N, K, dtype):
    """GEMM writing into a pre-allocated output tensor."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype)
    y = torch.empty((M, N), dtype=dtype, device="cuda")
    ref = torch.mm(x.float(), w.float().T).to(torch.float32)

    out = gemm_a16w16(x, w, dtype=dtype, y=y)

    if N % 128 == 0:
        assert out.data_ptr() == y.data_ptr(), "Output should reuse pre-allocated tensor"
    else:
        assert torch.equal(out, y), "Pre-allocated tensor should contain the result"
    assert verify_output(out.cpu().to(torch.float32), ref.cpu(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", "silu_exp2", "relu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_activation(M, N, K, dtype, activation):
    """GEMM with fused activation."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype)
    torch_out = torch.mm(x.float(), w.float().T)

    if activation == "gelu":
        ref = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        ref = F.gelu(torch_out, approximate="tanh")
    elif activation in ("silu", "silu_exp2"):
        ref = F.silu(torch_out)
    elif activation == "relu":
        ref = F.relu(torch_out)

    out = gemm_a16w16(x, w, dtype=dtype, activation=activation)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu().to(torch.float32), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("activation", ["gelu", "silu"])
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 512, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_bias_activation(M, N, K, dtype, activation):
    """GEMM with both bias and fused activation."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype)
    bias = torch.randn((N,), dtype=dtype, device="cuda")
    torch_out = torch.mm(x.float(), w.float().T) + bias.float()

    if activation == "gelu":
        ref = F.gelu(torch_out)
    elif activation == "silu":
        ref = F.silu(torch_out)

    out = gemm_a16w16(x, w, bias=bias, dtype=dtype, activation=activation)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu().to(torch.float32), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_layout(M, N, K, layout, dtype):
    """GEMM correctness across all input memory layouts."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype, layout=layout)
    ref = torch.mm(x.float(), w.float().T).to(torch.float32)

    out = gemm_a16w16(x, w, dtype=dtype)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_num_buffers(M, N, K, num_buffers, dtype):
    """GEMM with different pipeline buffer counts."""
    _check_gfx1250()
    torch.cuda.empty_cache()

    tile_k = 32
    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    num_k_tiles = K_padded // tile_k
    if num_k_tiles < num_buffers - 1:
        pytest.skip(f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers - 1}, got {num_k_tiles}")

    x, w = _generate_inputs(M, N, K, dtype)
    ref = torch.mm(x.float(), w.float().T).to(torch.float32)

    out = gemm_a16w16(x, w, dtype=dtype, num_buffers=num_buffers)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals_large())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_a16w16_large(M, N, K, dtype):
    """GEMM correctness on large shapes."""
    _check_gfx1250()
    _check_min_k(K)
    torch.cuda.empty_cache()

    x, w = _generate_inputs(M, N, K, dtype)
    ref = torch.mm(x.float(), w.float().T).to(torch.float32)

    out = gemm_a16w16(x, w, dtype=dtype)

    assert verify_output(out.cpu().to(torch.float32), ref.cpu(), rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-N", type=int, default=256)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--activation", type=str, default=None,
                        choices=["relu", "silu", "silu_exp2", "gelu", "gelu_tanh"])
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--layout", type=str, default="TN", choices=["TN", "TT", "NN", "NT"])
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    _check_gfx1250()
    _check_min_k(args.K, num_buffers=args.num_buffers)

    x, w = _generate_inputs(args.M, args.N, args.K, dtype, layout=args.layout)
    bias = torch.randn((args.N,), dtype=dtype, device="cuda") if args.bias else None

    ref = torch.mm(x.float(), w.float().T)
    if bias is not None:
        ref = ref + bias.float()
    if activation := args.activation:
        if activation == "gelu":
            ref = F.gelu(ref)
        elif activation == "gelu_tanh":
            ref = F.gelu(ref, approximate="tanh")
        elif activation in ("silu", "silu_exp2"):
            ref = F.silu(ref)
        elif activation == "relu":
            ref = F.relu(ref)

    out = gemm_a16w16(x, w, bias=bias, dtype=dtype,
                      activation=args.activation, num_buffers=args.num_buffers)

    torch.cuda.synchronize()
    passed = verify_output(out.cpu().to(torch.float32), ref.cpu().to(torch.float32),
                           rtol=3e-2, atol=3e-2)
    print("PASSED" if passed else "FAILED")
