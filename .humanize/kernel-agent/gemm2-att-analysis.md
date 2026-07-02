# gemm2 ATT stall attribution: ours-best vs aiter, and the B-prefetch-hoist fix

rocprofv3 **Advanced Thread Trace (ATT)** instruction-timeline analysis of the
isolated GPT-OSS MoE gemm2 (down-proj) kernel, decomposing where OUR gemm2 loses
time vs aiter's cktile a16w4 stage2, and the fix that closes part of the gap.

- GPU gfx950 (MI350X), `HIP_VISIBLE_DEVICES=7`, cold (`FLYDSL_RUNTIME_ENABLE_CACHE=0`).
- Shape: GPT-OSS M=128, model=inter=2880->pad 3072, E=128, topk=4, a4w4 (primary)
  / a8w4, mxfp4 w2, per_1x32 E8M0 scale. Identical fixed-seed inputs + routing on
  both sides (`same_input_parity.build_shared_inputs`), per-expert histogram
  identical -> identical 554.5 MB weight+scale traffic (established, not re-derived).
- Tool: **rocprofv3 1.1.0 ATT** (`advanced_thread_trace: true`, `att_target_cu: 1`,
  full SE/SIMD mask), `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` for source mapping (ours
  99% src-mapped; aiter's prebuilt CK binary has no debug info -> ASM-level only).
  Analyzer: `.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py`.

Configs traced:
- (i)  ours BEST = BM32 + reduce + G (`MXFP4_G2_KSTAGES=2` B-prefetch, NT w2).
       kernel `gemm2_a4w4_port_h3072_imax8192_bm32_nt_reduce_tk4_pad_g2ks2_v2`.
- (ii) ours BM16 + atomic (aiter-tile-aligned). kernel `..._bm16_nt_atomic_pad_v2`.
- (iii) aiter BM16 flatmm (`.../aiter_gemm2_isolated.py`), `ck_tile::MoeFlatmmKernel`.

Isolated device-kernel times (rocprofv3 kernel-trace, median over ~50 in-run
dispatches, this GPU/env):

| config | gemm2 median us | eff BW | ratio vs aiter |
|---|---:|---:|---:|
| aiter BM16 flatmm | **93.0** | 5.96 TB/s | 1.00x |
| ours BEST (BM32+reduce+G) | 97.8 | 5.67 TB/s | 1.055x |
| ours BM16 + atomic | **169.5** | 3.27 TB/s | 1.82x (dead end) |

(The ~1.055x here is tighter than the doc's 1.10-1.13x because ours-BEST already
carries G's B-prefetch and this GPU runs slightly hotter; the *diagnosis* is the
same.)

---

## 1. Stall breakdown (% of total stall cycles, steady-state dispatch, single CU)

| stall class | ours BEST | aiter | ours BM16-atomic |
|---|---:|---:|---:|
| **VMEM-wait** (`s_waitcnt vmcnt`) | 52.6% | 57.7% | 43.5% |
| **VMEM-load** (buffer_load issue/back-pressure) | 30.4% | 21.8% | 45.2% |
| **barrier** (`s_barrier`) | **11.3%** | 6.6% | 7.8% |
| MFMA | 1.8% | 0.5% | 0.1% |
| LDS (ds_read/ds_write) | 0.4% | 1.3% | 0.1% |
| LDS/SMEM-wait (`lgkmcnt`) | 1.5% | 1.8% | 0.6% |
| other | 2.1% | 10.2% | 2.6% |
| **total-stall / total-cycles** | **94.1%** | **84.3%** | ~96% |

Occupancy on the traced (busy) CU: **both ours and aiter run 4 waves/SIMD,
VGPR-bound** (ours VGPR 128, aiter 104; both hit 4 waves; avg waves-in-flight ~3.9
each). So at the instruction level on a hot CU the LDS-blocks/CU story is NOT the
binding limiter — both are VGPR-capped at 4 waves and **VMEM(weight)-bound**.

### The dominant differences

1. **Both kernels are weight-VMEM-bound** — ours 83% (VMEM-wait+load), aiter
   79.5%. HBM is near-saturated; there is no LDS-wait, atomic-store, or MFMA
   bottleneck. The reduce epilog contributes <2% (LDS-wait + a handful of store
   waitcnts) — **not epilog-bound, not LDS-bound, not atomic-bound.**

2. **Ours-BEST's distinguishing excess is barrier idle: 11.3% vs aiter's 6.6%**
   (ours 3 barriers costing 1.18M stall cycles, ~394K each; aiter 6 barriers
   costing 631K, ~105K each). Ours' single per-K-iteration `gpu.barrier()`
   (guarding the A-LDS triple-buffer ring, `mxmoe_gemm_v2.py:1385`) is a full-block
   sync that **absorbs the per-wave VMEM-latency imbalance**: waves whose weight
   loads returned late drag all 4 waves at the barrier. aiter processes 2 K-blocks
   per iteration with finer 16x16x32 MFMAs, so its barriers are both less frequent
   relative to compute and cheaper per hit.

3. **BM16-atomic confirms the pipeline-underfeed root cause** (supporting
   evidence, not the target). Halving BM to 16 halves our wide-MFMA density (8 vs
   16 16x16x128 MFMAs / K-tile) -> **VMEM-load back-pressure explodes to 45.2%**
   (queue-full: the weight loads can't be drained because there isn't enough
   compute to hide them) and the kernel balloons to 169us. This is why BM16 is a
   dtype dead-end for our wide scaled-fp4 MFMA: fewer/bigger MFMAs under-feed the
   weight stream.

**Dominant class = pipeline / VMEM-wait (weight stream not back in time), with
the ours-specific residual concentrated in barrier idle coupled to that VMEM
imbalance. NOT LDS, NOT atomic, NOT epilog.**

---

## 2. The fix: hoist the B-weight prefetch above the mainloop barrier (`g2_bhoist`)

The G 2-stage B pipeline (`g2_kstages==2`) already prefetches the next K-tile's B
weight+scale one tile ahead, but it issued that prefetch **after** the
per-iteration `gpu.barrier()`. B is a GMEM->register stream with **no LDS
dependency** — the barrier only orders the A-LDS ring, so it does not need to gate
the weight loads. Moving the prefetch **above** the barrier puts the long-latency
weight loads into the VMEM queue *before* the block synchronizes, so the weight
fetch overlaps the barrier wait instead of starting after it.

- Opt-in `g2_bhoist` param on `gemm2_body_v2` (default `False` = byte-identical),
  env `MXFP4_G2_BHOIST=1`, threaded through the dispatcher cache key + kernel-name
  tag (`_bhoist`). No-op unless `g2_kstages==2`. gemm1 untouched.
- The prefetch body was factored into a `prefetch_next_b(kt_rt)` closure called
  either before (hoist) or after (default) the barrier via `const_expr`.

### Effect (ATT, same dispatch index, ours BEST vs BEST+bhoist)

| stall class | BEST | BEST + bhoist |
|---|---:|---:|
| VMEM-wait | 52.6% | **34.9%** |
| VMEM-load | 30.4% | 46.8% |
| barrier | 11.3% | 12.4% |
| total stall (M cycles) | 10.46M | **10.17M** |

The hoist does exactly what the diagnosis predicted: **VMEM-wait drops
52.6% -> 34.9%** (the `s_waitcnt vmcnt` on the weights waits less because the
loads were issued earlier), shifting into VMEM-load queue-occupancy (loads now
in flight during the barrier). Total stall cycles drop; the profile moves toward
aiter's "loads always flowing" shape.

### Measured gemm2 (median device-kernel us, ~50 dispatches, cold)

| config | a4w4 median us | eff BW | ratio vs aiter (93.0us) |
|---|---:|---:|---:|
| ours BEST | 97.8 | 5.67 TB/s | 1.055x |
| **ours BEST + bhoist** | **95.6** | **5.80 TB/s** | **1.028x** |

a4w4: **97.8 -> 95.6 us (-2.2%)**, min 96.2 -> 93.5 us; ratio **1.055x -> 1.028x**.

a8w4: **100.5 -> 100.8 us (neutral, within noise)** — a8w4's heavier A path
(`KH_TILE_A=256`, 2x A-LDS DMA) is not as purely weight-VMEM-wait-bound, so the
hoist has nothing to hide there. Neutral is acceptable since it is opt-in.

---

## 3. Correctness + byte-identical + gemm1-untouched proofs

- **Correctness (cold, real 2880 dims):** a4w4 cos = **0.9910** (>0.85),
  a8w4 cos = **0.9996** (>0.95) — bitwise the same cos as without bhoist (the
  reorder does not change the math, only issue order).
- **Byte-identical default (AC-3):** the shipped default (`MXFP4_G2_BHOIST` unset
  -> `g2_bhoist=False`, and the shipped `g2_kstages=1` default) emits no `_bhoist`
  tag and `prefetch_next_b` stays in its original position. ISA md5 of the default
  `g2ks2` kernel is **identical** across the edited tree vs the pre-change source:
  `18b5e4d7ed041e45e2705f333a3e2f53` (both). (g2_kstages=1, the true shipped
  default, is untouched code.)
- **gemm1 untouched:** `git diff` touches only the gemm2 body's `g2_kstages==2`
  loop and the gemm2 dispatcher path; no gemm1 line changed.

---

## 4. Verdict on the residual

After bhoist, ours-BEST is **1.028x** aiter (95.6 vs 93.0 us, 5.80 vs 5.96 TB/s).
The ATT shows the kernel is now **VMEM-load-queue-bound (46.8%)** — i.e. HBM is
essentially saturated, loads are always in flight, and the remaining ~2.8% gap is
the last sliver of aiter's finer 2-K-block/16x16x32 interleave keeping the queue
marginally fuller. That residual is small and hard to close without adopting
aiter's bf16-A / finer-MFMA structure (a dtype regression for us, per the
deepdiff) or a deeper (kstages 3) B pipe that would add VGPR pressure and risk
dropping below 4 waves/SIMD. The dominant, actionable stall (weight-VMEM-wait +
barrier-coupled imbalance) has been attacked; further gains are in diminishing-
returns territory against a near-saturated HBM.

## Artifacts

- ATT dirs: `/tmp/att_ours_best/`, `/tmp/att_bhoist/`, `/tmp/att_bm16/`,
  `/tmp/att_aiter/` (ui_output_agent_*_dispatch_* with code.json + per-wave traces).
- Code: `kernels/mxmoe_gemm_v2.py` (`g2_bhoist` param + `prefetch_next_b` closure),
  `kernels/moe_dispatcher.py` (env `MXFP4_G2_BHOIST`, cache key, `_bhoist` tag).
- Drivers: `.humanize/kernel-agent/same_input_parity.py` (ours),
  `.humanize/kernel-agent/aiter_gemm2_isolated.py` (aiter).
