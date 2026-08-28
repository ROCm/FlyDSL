# M13 A8W4 SmoothQuant EP8 optimized result on MI355X

- Date: 2026-08-13 UTC
- Host: `mi355-gpu-47`, container `jun_950`
- GPU: 8 x MI355X (`gfx950`)
- Source branch/base: `ghu/mega_moe_v1_copy` / `77fa2e4e9bcc1d72aeaa4c7b66e53136836961dc`
- AITER: commit `05e85c8a`
- Mori: `1.2.3.dev46+g3784f3a7d`
- Shape: model dim 3584, intermediate dim 1280, 384 experts, top-k 8, EP8
- Quantization: A8W4 SmoothQuant with LQQ 1x64 weights
- Timing: 1000 CUDA Graph replays; maximum elapsed time over all 8 ranks
- Accuracy: distributed torch/AITER oracle, reduced over all 8 ranks

Performance command:

```bash
MORI_SHMEM_HEAP_SIZE=16G \
PYTHONPATH=/tmp/mori_latest_site_20260813:/tmp/FlyDSL_universe_77fa2e4e:/tmp/aiter_universe_commit_05e85c8a \
torchrun --standalone --nproc_per_node=8 \
  tests/kernels/test_mega_moe_int8.py \
  --mode a8w4smooth \
  --bs-list 8,16,32,64,128,256,512,1024 \
  --iters 1000 --measure-perf --skip-acc --strict
```

## Performance versus FlyDSL_bk

| tokens/rank | FlyDSL_bk us | optimized us | speedup |
|---:|---:|---:|---:|
| 8 | 133.9 | 127.5 | 4.8% |
| 16 | 143.0 | 132.5 | 7.3% |
| 32 | 152.8 | 135.3 | 11.5% |
| 64 | 157.8 | 146.8 | 7.0% |
| 128 | 242.1 | 164.9 | 31.9% |
| 256 | 272.2 | 229.1 | 15.8% |
| 512 | 350.7 | 342.0 | 2.5% |
| 1024 | 582.7 | 538.5 | 7.6% |

The FlyDSL_bk numbers are preserved in
`m13_a8w4smooth_ep8_mi355x_20260811.md` and were measured with 100 graph
replays.  The optimized numbers above use 1000 replays.

## Eight-rank accuracy

| tokens/rank | output relL2 max | Stage1 relL2 max | graph replay relL2 max | result |
|---:|---:|---:|---:|:---:|
| 8 | 1.9770e-3 | 2.2877e-5 | 0 | PASS |
| 16 | 2.0252e-3 | 2.1482e-5 | 0 | PASS |
| 32 | 2.0632e-3 | 2.2134e-5 | 0 | PASS |
| 64 | 1.9538e-3 | 1.9947e-5 | 0 | PASS |
| 128 | 2.0044e-3 | 1.9354e-5 | 0 | PASS |
| 256 | 1.9350e-3 | 1.6466e-5 | 0 | PASS |
| 512 | 1.8241e-3 | 2.1372e-5 | 0 | PASS |
| 1024 | 1.8403e-3 | 2.7437e-5 | 0 | PASS |

The fixed-slot 256 and 512 configurations were additionally checked with
2000 consecutive graph replays.  Both retained zero output drift on all eight
ranks.  The 512 stress result was 342.0 us end-to-end.

## Derived fixed-slot configurations

- 256: Stage1 `SBM64/N256/W8/grid3/DCU224/WPE2/work_shards8`, Stage2
  `BM32/N128/persist_cu128/use_nt=true`.
- 512: Stage1 `SBM64/N256/W8/grid1/DCU208/WPE1/work_shards4`, Stage2
  `BM32/N128/persist_cu240/use_nt=false`.

The selector derives these from route pressure, rows per expert, matrix shape,
fixed/compact mode, and CU budget.  It does not key the tuned path on EP count.
