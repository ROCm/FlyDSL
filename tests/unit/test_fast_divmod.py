#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""fx.FastDivmodDivisor: division by a runtime divisor whose magic numbers
fly.make_fast_divmod computes once, on the host in a @flyc.jit function or on
the GPU in a kernel. Checked against torch for i32 and i64 divisors, as a
kernel argument, built in a kernel, and as a shape leaf of the layout algebra.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping fast divmod tests.", allow_module_level=True)

BLOCK = 256
SIZE = 4096
I32_MAX = 2**31 - 1
I64_MAX = 2**63 - 1


@flyc.kernel
def _divmod_kernel(
    N: fx.Pointer,
    Q: fx.Pointer,
    R: fx.Pointer,
    D: fx.Pointer,
    divisor: fx.FastDivmodDivisor,
    n: fx.Int32,
):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        q, r = divmod(N[idx], divisor)
        Q[idx] = q
        R[idx] = r
        D[idx] = divisor.divisor


@flyc.jit
def _host_divmod_i32(N: fx.Pointer, Q: fx.Pointer, R: fx.Pointer, D: fx.Pointer, d: fx.Int32, n: fx.Int32):
    # The host computes the magic numbers; the kernel gets the divisor as one argument.
    divisor = fx.FastDivmodDivisor(d)
    _divmod_kernel(N, Q, R, D, divisor, n).launch(grid=((n + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1))


@flyc.jit
def _host_divmod_i64(N: fx.Pointer, Q: fx.Pointer, R: fx.Pointer, D: fx.Pointer, d: fx.Int64, n: fx.Int32):
    divisor = fx.FastDivmodDivisor(d)
    _divmod_kernel(N, Q, R, D, divisor, n).launch(grid=((n + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1))


@flyc.kernel
def _device_divmod_kernel(N: fx.Pointer, Q: fx.Pointer, R: fx.Pointer, d: fx.Int32, n: fx.Int32):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        # The GPU computes the magic numbers.
        divisor = fx.FastDivmodDivisor(d)
        Q[idx] = N[idx] // divisor
        R[idx] = N[idx] % divisor


@flyc.jit
def _device_divmod(N: fx.Pointer, Q: fx.Pointer, R: fx.Pointer, d: fx.Int32, n: fx.Int32):
    _device_divmod_kernel(N, Q, R, d, n).launch(grid=((n + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1))


@flyc.kernel
def _crd2idx_kernel(
    N: fx.Pointer,
    OUT: fx.Pointer,
    m: fx.FastDivmodDivisor,
    k: fx.FastDivmodDivisor,
    n: fx.Int32,
):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        # Decomposing a coordinate by the shape (m, k, 8) divides by m and k.
        layout = fx.make_layout(fx.make_shape(m, k, 8), fx.make_stride(1, 128, 4096))
        OUT[idx] = fx.get_scalar(fx.crd2idx(N[idx], layout))


@flyc.jit
def _crd2idx(N: fx.Pointer, OUT: fx.Pointer, m: fx.Int32, k: fx.Int32, n: fx.Int32):
    _crd2idx_kernel(N, OUT, fx.FastDivmodDivisor(m), fx.FastDivmodDivisor(k), n).launch(
        grid=((n + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1)
    )


def _ptr(dtype, tensor):
    return flyc.from_c_void_p(dtype, tensor.data_ptr())


def _dividends(d, top, dtype):
    edges = [n for n in (0, 1, d - 1, d, d + 1, 2 * d - 1, top - top % d, top - 1, top) if 0 <= n <= top]
    rest = torch.randint(0, top, (SIZE - len(edges),), dtype=dtype)
    return torch.cat([torch.tensor(edges, dtype=dtype), rest]).cuda()


@pytest.mark.parametrize(
    "dtype, fx_dtype, top, host_divmod, divisors",
    [
        (
            torch.int32,
            fx.Int32,
            I32_MAX,
            _host_divmod_i32,
            [1, 2, 3, 7, 13, 64, 1000, 65537, 2**20 + 1, 2**30, I32_MAX - 1, I32_MAX],
        ),
        (
            torch.int64,
            fx.Int64,
            I64_MAX,
            _host_divmod_i64,
            [1, 3, 1000, 2**31 - 1, 2**32 + 1, 2**40 + 7, 2**62, 2**62 + 1, I64_MAX],
        ),
    ],
    ids=["i32", "i64"],
)
def test_divisor_from_the_host(dtype, fx_dtype, top, host_divmod, divisors):
    torch.manual_seed(0)
    for d in divisors:
        n = _dividends(d, top, dtype)
        q, r, div = (torch.empty_like(n) for _ in range(3))
        host_divmod(_ptr(fx_dtype, n), _ptr(fx_dtype, q), _ptr(fx_dtype, r), _ptr(fx_dtype, div), d, SIZE)
        torch.cuda.synchronize()
        n, q, r, div = n.cpu(), q.cpu(), r.cpu(), div.cpu()
        assert torch.equal(q, n // d), f"quotient by {d}"
        assert torch.equal(r, n % d), f"remainder by {d}"
        assert torch.equal(div, torch.full_like(n, d)), f"divisor {d}"


def test_divisor_built_in_the_kernel():
    torch.manual_seed(1)
    for d in [1, 5, 4096, 99991, I32_MAX]:
        n = _dividends(d, I32_MAX, torch.int32)
        q, r = torch.empty_like(n), torch.empty_like(n)
        _device_divmod(_ptr(fx.Int32, n), _ptr(fx.Int32, q), _ptr(fx.Int32, r), d, SIZE)
        torch.cuda.synchronize()
        n, q, r = n.cpu(), q.cpu(), r.cpu()
        assert torch.equal(q, n // d), f"quotient by {d}"
        assert torch.equal(r, n % d), f"remainder by {d}"


def test_divisor_as_a_shape_leaf():
    torch.manual_seed(2)
    for m, k in [(3, 5), (7, 1), (128, 32), (100, 17)]:
        n = torch.randint(0, 1 << 20, (SIZE,), dtype=torch.int32).cuda()
        out = torch.empty_like(n)
        _crd2idx(_ptr(fx.Int32, n), _ptr(fx.Int32, out), m, k, SIZE)
        torch.cuda.synchronize()
        n, out = n.cpu(), out.cpu()
        expected = n % m + (n // m % k) * 128 + (n // (m * k)) * 4096
        assert torch.equal(out, expected), f"shape ({m}, {k}, 8)"
