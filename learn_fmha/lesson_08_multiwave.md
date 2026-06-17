# Lesson 08 — Multi-wave workgroups: occupancy & CU-starvation

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_08_multiwave.py 1024 16384`
(drives the production `fmha_prefill_fp8_8wave` kernel at three wave counts).

## The trick (the whole diff)
Tile **height** = waves-per-workgroup × 32 q-rows. The single knob `FMHA_NWAVES`:
`8 → BM=256`, `4 → BM=128`, `2 → BM=64`. Lessons 00–07 used **one** wave (BM=16/32) on **one** of
80 CUs — which is why their throughput was meaningless. Real throughput needs a *grid* of workgroups
that fills the machine.

## Measured (bs=1, nq8, nk1, causal, TFLOPS)
| seq | BM=256 (8w) | **BM=128 (4w)** | BM=64 (2w) |
|---|---|---|---|
| 1024 | 4.3 | **5.0** | 4.8 |
| 4096 | 18.8 | **23.2** | 22.3 |
| 16384 | 39.7 | **46.6** | 38.3 |
| 32768 | ~45 | **52.8** | 41.9 |

**BM=128 / 4-wave is uniformly best (+17–23%).** A live run reproduces 39 / 46 / 39 at sq16384.

## Why — the occupancy reasoning (the profiling habit)
Two competing effects, and you read both from the profile:

1. **Grid vs CUs (fills the machine?).** `grid = ceil(seqlen/BM) · nheads`. With 80 CUs you want
   grid ≫ 80. Bigger BM → *fewer* workgroups → at small seqlen the grid can't even cover 80 CUs
   (CU-starvation). Smaller BM helps here. *But note:* at sq1024 the BM=256→128 gain was only +16%
   even though grid doubled 32→64 — so starvation was a minor factor; the bigger issue is (2).

2. **Per-workgroup latency hiding.** A workgroup must have enough independent waves in flight to hide
   MFMA/LDS latency. **VGPR caps this:** a SIMD has 512 VGPR-banks, so `waves/SIMD = ⌊512 /
   vgpr_count⌋`. The BM=128 kernel uses 164 VGPR → 3 waves/SIMD resident. BM=256 has a longer
   per-wave dependency chain that the same occupancy hides *worse*; BM=64 shrinks per-wg work so much
   that large-seq throughput drops. BM=128 is the sweet spot between the two.

**How to read it yourself:**
- grid size: `ceil(seqlen/BM)·nheads` vs 80 → starved?
- `vgpr_count` from `19_gpu_module_to_binary.mlir` → `⌊512/vgpr⌋` waves/SIMD → occupancy.
- sweep the knob and measure — the optimum is shape-dependent, so *measure on your shapes*.

## A/B method note
The "before/after" here is one compile-time constant (waves → BM). Because BM is baked at module
load, each NWAVES runs in its own subprocess (the driver shells out per value). This is the pattern
for all the *structural* lessons (08, 15, 16, 17): drive the real validated kernel and read the
bench, rather than re-deriving a simplified multi-wave kernel that might mislead.

## Next
Lesson 10: cooperative K/V staging into LDS (load the shared tile once for all waves) — and the
barrier cost that comes with it.
