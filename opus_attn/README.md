# opus_attn — GQA Flash Attention Kernels for gfx950

Hand-written Grouped-Query Attention (GQA) kernels using the [OPUS](https://github.com/ROCm/aiter) template library on AMD gfx950 (MI355).

## Features

- Flash Attention with online softmax (no materialized NxN attention matrix)
- D=128 and D=512 head dimensions selected by `-d`
- Causal and non-causal variants for each head dimension (parallel compilation via `make -j`)
- Double-buffered K/V tiles in shared memory
- Software-pipelined global→shared→register data movement
- Fine-grained scheduling barriers for MFMA/VALU/EXP interleaving
- `__HIP_DEVICE_COMPILE__` guard for fast host pass (~580ms saved)
- CPU reference implementation for validation

## Files

```
opus_attn/
├── Makefile                              # Parallel build (make -j)
├── rebuild.sh                            # Build + benchmark d128/d512 variants
├── gqa_defs.h                            # Shared types: bf16_t, opus_gqa_kargs, opus_gqa_traits
├── gqa_d128_kernel_template.hpp          # D=128 kernel implementation
├── gqa_d128_kernel_causal.cc             # D=128 causal kernel instantiation
├── gqa_d128_kernel_noncausal.cc          # D=128 non-causal kernel instantiation
├── gqa_d512_kernel_template.hpp          # D=512 kernel implementation
├── gqa_d512_kernel_causal.cc             # D=512 causal kernel instantiation
├── gqa_d512_kernel_noncausal.cc          # D=512 non-causal kernel instantiation
└── gqa_host.cc                           # Host launcher, benchmark, validation, main()
```

## Prerequisites

- ROCm with hipcc (tested with ROCm 7.1.1)
- gfx950 GPU target
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP support (for CPU reference and random init)

## Build

```bash
cd opus_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include   # default: /home/carhuang/repo/aiter/csrc/include
make -j        # parallel build: d128/d512 causal + non-causal + host + link
```

Or use the convenience script (builds + runs benchmarks):

```bash
./rebuild.sh
```

## Run

```bash
./build/gqa_attn.exe -b 16 -d 128 -h_q 64 -h_kv 8 -n 1024 --no-causal
./build/gqa_attn.exe -b 1 -d 512 -h_q 128 -h_kv 1 -n 512 --causal --verify
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-b` | Batch size | 16 |
| `-h_q` | Number of query heads | 64 |
| `-h_kv` | Number of KV heads | 8 |
| `-n` | Sequence length | 1024 |
| `-d` | Head dimension (128 or 512) | 128 |
| `--causal` | Enable causal masking | (default) |
| `--no-causal` | Disable causal masking | |
| `--verify` | Enable CPU reference verification | off |


## Kernel configuration

The kernel is parameterized via `opus_gqa_traits<Q_TILE, KV_TILE, D_TILE, NUM_WARPS, CAUSAL>`. Four kernel instantiations are compiled: D=128/D=512 crossed with causal/non-causal.

### D=128 configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 32 | Query tile size per warp |
| KV_TILE_SIZE | 64 | KV tile size in shared memory |
| D_TILE_SIZE | 128 | Head dimension |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| CAUSAL | `true`/`false` | Causal masking (two separate kernel binaries) |
| MFMA | 32x32x16 bf16 | Matrix multiply instruction |

### D=512 configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 16 | Query tile size per warp |
| KV_TILE_SIZE | 32 | KV tile size in shared memory |
| D_TILE_SIZE | 512 | Head dimension (iterated in 32-element slices) |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| CAUSAL | `true`/`false` | Causal masking (two separate kernel binaries) |
| MFMA | 16x16x32 bf16 | Matrix multiply instruction |

## Compile time

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701).

The timing numbers below were collected with the kernel resource-report/temp-file flags removed from the compile command: `-Rpass-analysis=kernel-resource-usage -save-temps=obj`.

| Build mode | Time | Notes |
|------------|------|-------|
| `make -j` (parallel) | **1.42s** | All four kernel variants + host compiled in parallel |
| `make` (sequential) | 6.09s | All kernels (D=128 causal/non-causal, D=512 causal/non-causal) + host sequentially |

### Compile-time techniques applied

- **`__HIP_DEVICE_COMPILE__` guard**: kernel .cc files skip the full kernel body on the host pass, providing only an empty stub for `__device_stub__` generation (~580ms saved per kernel file)
- **`-D__HIPCC_RTC__`**: applied to kernel .cc files to skip the implicit `__clang_hip_runtime_wrapper.h` on the host pass (~250ms saved per kernel file)
- **Parallel build**: four kernel translation units and the host compile simultaneously via `make -j`

### Per-file breakdown

| File | Time | VGPRs | SGPRs | Spill | Occ | LDS (Byte) |
|------|:----:|:-----:|:-----:|:-----:|:---:|:----------:|
| `gqa_d128_kernel_causal.cc` | 1.21s | 237 | 50 | 0 | 2 | 68096 |
| `gqa_d128_kernel_noncausal.cc` | 1.12s | 232 | 44 | 0 | 2 | 68096 |
| `gqa_d512_kernel_causal.cc` | 1.34s | 248 | 56 | 0 | 2 | 135168 |
| `gqa_d512_kernel_noncausal.cc` | 1.38s | 248 | 52 | 0 | 2 | 135168 |
| `gqa_host.cc` | 0.94s | — | — | — | — | — |
| Link | 0.02s | — | — | — | — | — |

## Performance

Measured on MI355X with ROCm 7.1.1.

### D=128 (B=16, H_Q=64, H_KV=8)

| N | Causal Time (ms) | Causal TFlops | Non-causal Time (ms) | Non-causal TFlops |
|---:|---:|---:|---:|---:|
| 1024 | 0.384 | 716.20 | 0.534 | 1029.60 |
| 2048 | 1.163 | 945.50 | 1.872 | 1174.84 |
| 4096 | 3.967 | 1108.79 | 7.016 | 1253.64 |
| 8192 | 14.637 | 1201.90 | 27.173 | 1294.82 |
| 16384 | 56.100 | 1254.33 | 107.061 | 1314.55 |

### D=512 (B=1, H_Q=128, H_KV=1)

| N | Causal Time (ms) | Causal TFlops | Non-causal Time (ms) | Non-causal TFlops |
|---:|---:|---:|---:|---:|
| 1024 | 0.192 | 714.95 | 0.281 | 978.26 |
| 2048 | 0.583 | 942.64 | 0.998 | 1101.30 |
| 4096 | 2.073 | 1060.69 | 3.807 | 1155.34 |
| 8192 | 7.733 | 1137.54 | 14.835 | 1185.85 |
| 16384 | 29.785 | 1181.29 | 58.690 | 1198.99 |
