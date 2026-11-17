# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Activation outlier smoothing with ``fx.hadamard.BlockFWHT``.

Low-bit quantization formats (FP4/INT4/FP8) lose precision when a few activation
channels carry large outlier spikes. Applying a normalized Sylvester Walsh-Hadamard
rotation ``y = (1 / sqrt(N)) * H_N @ x`` spreads outlier energy evenly across all
``N`` channels in ``O(N log N)`` additions without materializing an ``N x N`` matrix,
and applying the exact same kernel a second time reconstructs ``x`` because the
normalized Hadamard matrix is symmetric and orthogonal (``H @ H == I``).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK = 128
ITEMS = 4
N = BLOCK * ITEMS


@flyc.kernel
def fwht_fp16_kernel(inp: fx.Pointer, out: fx.Pointer):
    fwht = fx.hadamard.BlockFWHT[fx.Float16, fx.known_block_size(), ITEMS]
    storage = fx.SharedAllocator().allocate(fwht.SharedStorage).peek()

    offset = fx.thread_idx.x * ITEMS
    items = (inp + offset).load(fx.Float16x4)
    rotated = fwht(items, storage=storage, normalize=True)
    (out + offset).store(rotated)


@flyc.jit
def run_fwht_fp16(
    inp: fx.Pointer,
    out: fx.Pointer,
    stream: fx.Stream = fx.Stream(None),
):
    fwht_fp16_kernel(inp, out).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)


torch.manual_seed(0)
x = torch.randn(N, dtype=torch.float16, device="cuda") * 0.25
x[17] = 32.0  # Inject a sharp activation outlier spike.
y = torch.empty_like(x)
x_rec = torch.empty_like(x)

run_fwht_fp16(flyc.from_c_void_p(fx.Float16, x.data_ptr()), flyc.from_c_void_p(fx.Float16, y.data_ptr()))
run_fwht_fp16(flyc.from_c_void_p(fx.Float16, y.data_ptr()), flyc.from_c_void_p(fx.Float16, x_rec.data_ptr()))
torch.cuda.synchronize()

peak_before = float(x.abs().max().item())
peak_after = float(y.abs().max().item())
max_err = float((x_rec.float() - x.float()).abs().max().item())

if peak_after < peak_before * 0.25 and max_err < 5e-2:
    print(f"PASS (N={N}, peak {peak_before:.2f} -> {peak_after:.2f}, roundtrip max_err={max_err:.4e})")
else:
    print(f"FAIL (peak_before={peak_before}, peak_after={peak_after}, max_err={max_err})")
    raise SystemExit(1)
