<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Fast Walsh-Hadamard transform

`flydsl.extension.hadamard` (`fx.hadamard`) provides warp- and block-scope Sylvester Fast
Walsh-Hadamard Transforms (FWHT) for activation rotation, outlier smoothing prior to low-bit
quantization, and structured orthogonal projections.

## Public API

| API | Return type | Description |
|---|---|---|
| `fx.hadamard.warp_fwht(value, *, width=None, normalize=False)` | Same dtype and container shape as `value` | Compute the FWHT across `width` lanes and the caller's blocked per-lane items |
| `fx.hadamard.BlockFWHT[dtype, block_size, items_per_thread=1]` | Specialized block primitive class | Cooperative block-wide FWHT over `N = block_threads * items_per_thread` elements |

### Supported types and shapes

- **Element dtypes:** `fx.Float32`, `fx.Float16`, and `fx.BFloat16`.
- **Per-thread items:** a single scalar (`items_per_thread == 1`) or a power-of-two `fx.Vector`,
  `tuple`, or `list` owned by each lane/thread in **blocked** order (lane/thread `t` owns contiguous
  transform indices `t * M .. t * M + M - 1`).
- **Normalization:** when `normalize=False` (default), computes the unnormalized Sylvester matrix
  product `y = H_N @ x` where `H_1 = [[1]]` and `H_{2k} = [[H_k, H_k], [H_k, -H_k]]`. When
  `normalize=True`, scales the output by `1 / sqrt(N)` so the transform is orthonormal and
  self-inverse (`H(H(x)) == x`).
- **Precision and `true16` safety:** internal butterfly stages and cross-lane exchanges execute in
  `fx.Float32` before converting back to `dtype`, preventing `fx.Float16` intermediate overflow and
  avoiding `+real-true16` 16-bit subregister hazards on RDNA4 targets.

### Shared storage for `BlockFWHT`

Specialize `fx.hadamard.BlockFWHT` with `[dtype, block_size]` (for one scalar per thread) or
`[dtype, block_size, items_per_thread]`:

- **Single-warp blocks (`block_threads <= num_warp_threads()`):** `SharedStorage` is `fx.Empty`
  (`0` bytes of LDS and `0` barriers). `storage` may be omitted or passed from
  `fx.SharedAllocator().allocate(H.SharedStorage).peek()`.
- **Multi-warp blocks (`num_warps >= 2`, up to `1024` threads):** `SharedStorage` stages a single
  bank-conflict-free padded LDS transpose (`(N + 32) * 4` bytes total, two `fx.barrier()`
  synchronizations) factoring `H_N = kron(H_num_warps, H_warp_tile)`. Allocate
  `H.SharedStorage` once per block and synchronize with `fx.barrier()` before reusing that shared
  memory.

## Usage

The following complete example smooths activation outliers across a 512-element `fx.Float16` vector
(`BLOCK = 128`, `ITEMS = 4`) and verifies self-inverse reconstruction:

```python
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK = 128
ITEMS = 4
N = BLOCK * ITEMS


@flyc.kernel
def rotate_fp16(x_ptr: fx.Pointer, y_ptr: fx.Pointer):
    fwht = fx.hadamard.BlockFWHT[fx.Float16, fx.known_block_size(), ITEMS]
    storage = fx.SharedAllocator().allocate(fwht.SharedStorage).peek()

    offset = fx.thread_idx.x * ITEMS
    items = (x_ptr + offset).load(fx.Float16x4)
    rotated = fwht(items, storage=storage, normalize=True)
    (y_ptr + offset).store(rotated)


@flyc.jit
def launch_rotate_fp16(
    x_ptr: fx.Pointer,
    y_ptr: fx.Pointer,
    stream: fx.Stream = fx.Stream(None),
):
    rotate_fp16(x_ptr, y_ptr).launch(
        grid=(1, 1, 1),
        block=(BLOCK, 1, 1),
        stream=stream,
    )
```

See `examples/extension/hadamard/01-block_fwht.py` for the runnable script.

## Target dispatch

`fx.hadamard` selects an architecture-specialized implementation for the active compilation target:

- On wave32 GFX10+/RDNA/CDNA5 targets (`gfx10*`, `gfx11*`, `gfx12*`), intra-warp XOR butterflies
  use `row_xmask` DPP stages for distances `1, 2, 4, 8` and `v_permlanex16_b32` for distance `16`
  with zero LDS instructions.
- On wave64 CDNA targets (`gfx9*`), distances `1, 2, 8` use VALU DPP (`quad_perm` and `row_ror:8`),
  distance `4` uses `ds_swizzle_b32` (along with distance `16` on `gfx90*`/`gfx94*` and
  `ds_bpermute_b32` for distance `32`), while CDNA4 (`gfx95*`) lowers distances `16` and `32` to
  VALU `v_permlane16_swap_b32` and `v_permlane32_swap_b32`.
- `fx.hadamard.universal` always uses portable `fx.gpu` shuffles for target-independent compilation
  and equivalence testing.
