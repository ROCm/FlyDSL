<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Quantization codecs

`flydsl.extension.quant` (`fx.quant`) provides register-level micro-scaled and sub-byte
quantization codecs for FlyDSL kernels, with portable bit-exact implementations in
`fx.quant.universal` and target-specific hardware overrides in `fx.quant.rocdl`.

## Public API

### Block-scale codecs

| API | Return type | Description |
|---|---|---|
| `fx.quant.f32_to_e8m0(x, *, round_up=True)` | `fx.Uint8` | Encode non-negative `Float32` scale as OCP MX `E8M0` (`2^(code - 127)`) |
| `fx.quant.e8m0_to_f32(code)` | `fx.Float32` | Decode OCP MX `E8M0` scale byte (`0x00 -> 2^-127`, `0xFF -> NaN`) |
| `fx.quant.f32_to_ue4m3(x)` | `fx.Uint8` | Encode non-negative `Float32` scale as `UE4M3` (`sign=0`, bias `7`, max `448.0`) |
| `fx.quant.ue4m3_to_f32(code)` | `fx.Float32` | Decode `UE4M3` micro-block scale byte (`[0.0, 448.0]` or `NaN`) |

### Packed `BFloat16` and OCP `FP8` (`E4M3FN` / `E5M2`) codecs

| API | Return type | Description |
|---|---|---|
| `fx.quant.pack_bf16x2(v0, v1)` | `fx.Uint32` | Round two `Float32` values to `BFloat16` (RNE) and pack into `Uint32` |
| `fx.quant.unpack_bf16x2(packed)` | 2-tuple of `fx.Float32` | Unpack a `bf16x2` word into two exact `Float32` values |
| `fx.quant.pack_e4m3x2(v0, v1, *, scale_inv=None, saturate=True)` | `fx.Uint32` | Pack two `Float32` values as OCP `E4M3FN` in bits `[15:0]` |
| `fx.quant.pack_e4m3x4(v0, v1, v2, v3, *, scale_inv=None, saturate=True)` | `fx.Uint32` | Pack four `Float32` values as four OCP `E4M3FN` bytes |
| `fx.quant.unpack_e4m3x4(packed, *, scale=None)` | 4-tuple of `fx.Float32` | Unpack four OCP `E4M3FN` bytes to `Float32` with optional block scale |
| `fx.quant.unpack_e4m3x4_bf16x2(packed)` | 2-tuple of `fx.Uint32` | Losslessly unpack four OCP `E4M3FN` bytes into two `bf16x2` words |
| `fx.quant.stoch_round_e4m3x4(v0, v1, v2, v3, entropy, *, scale_inv=None, saturate=True)` | `fx.Uint32` | Stochastically round four `Float32` values to packed OCP `E4M3FN` |
| `fx.quant.pack_e5m2x2(v0, v1, *, scale_inv=None, saturate=True)` | `fx.Uint32` | Pack two `Float32` values as OCP `E5M2` in bits `[15:0]` |
| `fx.quant.pack_e5m2x4(v0, v1, v2, v3, *, scale_inv=None, saturate=True)` | `fx.Uint32` | Pack four `Float32` values as four OCP `E5M2` bytes |
| `fx.quant.unpack_e5m2x4(packed, *, scale=None)` | 4-tuple of `fx.Float32` | Unpack four OCP `E5M2` bytes to `Float32` with optional block scale |
| `fx.quant.unpack_e5m2x4_bf16x2(packed)` | 2-tuple of `fx.Uint32` | Losslessly unpack four OCP `E5M2` bytes into two `bf16x2` words |
| `fx.quant.stoch_round_e5m2x4(v0, v1, v2, v3, entropy, *, scale_inv=None, saturate=True)` | `fx.Uint32` | Stochastically round four `Float32` values to packed OCP `E5M2` |

### Sub-byte `FP4` (`E2M1`) and `INT4` / `UINT4` codecs

| API | Return type | Description |
|---|---|---|
| `fx.quant.pack_e2m1x8(values, *, scale_inv=None)` | `fx.Uint32` | Pack eight `Float32` values into eight 4-bit `E2M1` nibbles (RNE) |
| `fx.quant.unpack_e2m1x4(packed, *, half=0, scale=None)` | 4-tuple of `fx.Float32` | Unpack low (`half=0`) or high (`half=1`) four `E2M1` nibbles to `Float32` |
| `fx.quant.unpack_e2m1x8(packed, *, scale=None)` | 8-tuple of `fx.Float32` | Unpack all eight `E2M1` nibbles to `Float32` |
| `fx.quant.unpack_e2m1x4_bf16x2(packed, *, half=0)` | 2-tuple of `fx.Uint32` | Losslessly unpack four `E2M1` nibbles into two `bf16x2` words |
| `fx.quant.unpack_e2m1x4_e4m3x4(packed, *, half=0)` | `fx.Uint32` | Losslessly unpack four `E2M1` nibbles into four OCP `E4M3FN` bytes |
| `fx.quant.unpack_e2m1x8_int4_pair(packed)` | 2-tuple of `fx.Uint32` | Decompose eight `E2M1` nibbles into signed `INT4` words `(p, q)` with `p + q == 2 * w` |
| `fx.quant.pack_int4x8(values)` | `fx.Uint32` | Pack eight 4-bit integers into one `Uint32` |
| `fx.quant.unpack_int4x8(packed)` | 8-tuple of `fx.Int32` | Unpack eight signed two's-complement `INT4` values in `[-8, 7]` |
| `fx.quant.unpack_uint4x8(packed)` | 8-tuple of `fx.Uint32` | Unpack eight unsigned `UINT4` values in `[0, 15]` |

## Architecture dispatch and True16 notes

`fx.quant` preserves identical numerical semantics across architectures while emitting native
hardware instructions when available:

- **`gfx950` (CDNA4)**: Uses native OCP `FP8` (`v_cvt_pk_fp8_f32`, `v_cvt_pk_f32_fp8`,
  `v_cvt_sr_fp8_f32`) and native `FP4` (`v_cvt_scalef32_pk_fp4_f32`, `v_cvt_scalef32_pk_f32_fp4`,
  `v_cvt_scalef32_pk_bf16_fp4`), plus 1-cycle `v_perm_b32` nibble LUTs.
- **`gfx120x` (RDNA4) and `gfx1250` (CDNA5)**: Uses native OCP `FP8` (`v_cvt_pk_fp8_f32`,
  `v_cvt_f32_fp8`, `v_cvt_sr_fp8_f32`) and `v_perm_b32` register table lookups for `E2M1`.
  On True16 `HasCvtFP8ByteSel` targets, FlyDSL's pinned LLVM currently omits `UseRealTrue16Insts`
  (`_t16`) ISel patterns for packed `v_cvt_pk_f32_fp8` / `v_cvt_pk_f32_bf8` (tracked in
  [ROCm/FlyDSL#1376](https://github.com/ROCm/FlyDSL/issues/1376) and upstream LLVM PR
  [llvm/llvm-project#231721](https://github.com/llvm/llvm-project/pull/231721)), so `fx.quant`
  currently dispatches scalar byte-indexed `v_cvt_f32_fp8` / `v_cvt_f32_bf8` (`byte_sel=0..3`).
  The 2-instruction True16 `v_cvt_pk_f32_{fp8,bf8}` unpack path and paired `ue4m3x2_to_f32` scale
  decoder are already built in (commented out in `quant/universal.py` and `quant/rocdl.py`) and
  ready to enable once FlyDSL's LLVM pin includes `llvm/llvm-project#231721`.
- **`gfx942` (CDNA3) and `gfx11xx` (RDNA3)**: `gfx942` hardware FP8 instructions implement FNUZ
  (`E4M3FNUZ` / `E5M2FNUZ`) rather than OCP `E4M3FN` / `E5M2`, and `gfx11xx` has no hardware FP8
  conversion ISA; both use the bit-exact VALU OCP FP8 paths and 1-cycle `v_perm_b32` `E2M1` LUTs.

## Usage

Combine `fx.coop` reductions, `fx.random` Philox draws, and `fx.quant` codecs to implement
per-block scaling or stochastic rounding in registers:

```python
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK = 64


@flyc.kernel
def quantize_fp8_sr(Src: fx.Tensor, Packed: fx.Tensor, seed: fx.Uint32):
    tid = fx.thread_idx.x
    base = tid * 4
    entropy = fx.random.randint4x(seed, fx.Uint32(tid))
    word = fx.quant.stoch_round_e4m3x4(
        Src[base + 0],
        Src[base + 1],
        Src[base + 2],
        Src[base + 3],
        entropy,
    )
    Packed[tid] = word.bitcast(fx.Int32)


@flyc.jit
def launch_quantize_fp8_sr(
    Src: fx.Tensor,
    Packed: fx.Tensor,
    seed: fx.Uint32,
    stream: fx.Stream = fx.Stream(None),
):
    quantize_fp8_sr(Src, Packed, seed).launch(
        grid=(1, 1, 1),
        block=(BLOCK, 1, 1),
        stream=stream,
    )


src = torch.randn(BLOCK * 4, dtype=torch.float32, device="cuda")
packed = torch.empty(BLOCK, dtype=torch.int32, device="cuda")
launch_quantize_fp8_sr(src, packed, 42, stream=torch.cuda.current_stream())
torch.cuda.synchronize()
```

See `examples/extension/quant/01-microscaled_quant.py` for a complete runnable `MXFP4` and `NVFP4`
round-trip example.
