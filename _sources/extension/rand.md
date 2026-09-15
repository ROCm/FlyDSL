<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Random number generation

`flydsl.extension.random` provides stateless, counter-based random-number generation for FlyDSL
kernels.

## Public API

| API | Return type | Description |
|---|---|---|
| `fx.random.randint(seed, offset, n_rounds=10)` | `fx.Uint32` | First word from one Philox 4x32 draw |
| `fx.random.randint4x(seed, offset, n_rounds=10)` | 4-tuple of `fx.Uint32` | Four words from one Philox 4x32 draw |
| `fx.random.rand(seed, offset, n_rounds=10)` | `fx.Float32` | Uniform sample in `[0, 1)` |
| `fx.random.rand4x(seed, offset, n_rounds=10)` | 4-tuple of `fx.Float32` | Four uniform samples in `[0, 1)` |
| `fx.random.randn(seed, offset, n_rounds=10)` | `fx.Float32` | Standard normal sample |
| `fx.random.randn4x(seed, offset, n_rounds=10)` | 4-tuple of `fx.Float32` | Four standard normal samples |

All functions use the same parameters:

| Parameter | Description |
|---|---|
| `seed` | Integer value selecting the deterministic random stream |
| `offset` | Integer counter identifying a draw within the stream |
| `n_rounds` | Number of Philox rounds; defaults to `10` |

The same `(seed, offset, n_rounds)` produces the same result. Use a distinct offset for every
logically independent draw. Prefer a `4x` API when four values are needed from one offset.

## Usage

The following complete example generates four uniform values per thread. The output tensor has one
contiguous four-value segment for each thread:

```python
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK = 256


@flyc.kernel
def fill_uniform4(out: fx.Tensor, seed: fx.Uint64):
    tid = fx.thread_idx.x
    values = fx.random.rand4x(seed, fx.Uint64(tid))
    base = tid * 4
    for i in fx.range_constexpr(4):
        out[base + i] = values[i]


@flyc.jit
def launch_fill_uniform4(
    out: fx.Tensor,
    seed: fx.Uint64,
    stream: fx.Stream = fx.Stream(None),
):
    fill_uniform4(out, seed).launch(
        grid=(1, 1, 1),
        block=(BLOCK, 1, 1),
        stream=stream,
    )


out = torch.empty(BLOCK * 4, dtype=torch.float32, device="cuda")
launch_fill_uniform4(out, 1234, stream=torch.cuda.current_stream())
torch.cuda.synchronize()
```

Transform a standard normal sample to a distribution with a chosen mean and standard deviation:

```python
x = mean + stddev * fx.random.randn(seed, offset)
```

`fx.random` selects an implementation for the active compilation target. Use
`fx.random.universal` only when the portable implementation is explicitly required for testing or
comparison. These generators are not cryptographically secure.
