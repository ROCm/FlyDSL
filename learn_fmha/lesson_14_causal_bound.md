# Lesson 14 — Skip fully-masked kv tiles (and why less work ≠ less time)

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_14_causal_bound.py` (prints, per q-tile, how
many kv-tiles are computed vs skipped).

## The trick
Under causal masking a q-tile at rows `[q0, q0+BM)` attends to kv only up to `q0+BM-1+(sk-sq)`. Every
kv-tile beyond that is 100% masked → computing it is pure waste. Cap the kv loop at runtime:
```
n_kv = min(full_tiles, ceil((q_max + (sk-sq) + 1) / BN))
```
so early q-tiles iterate far fewer kv-tiles. Also **fold the two per-element mask compares**
(`kv < sk` AND `kv <= causal_bound`) into ONE compare against a loop-invariant `eff_bound` hoisted
out of the kv loop — halving the mask VALU. Both are in `fmha_prefill_fp8_8wave.py`.

Summed over all q-tiles, the causal bound roughly **halves** total kv work.

## The subtle lesson: less work ≠ less time on an under-occupied machine
PMC measured this cutting total work ~45% (busy cycles 2805M → 1547M) — yet **wall-clock barely
moved** at the benchmark's small grid. Why: at `grid=64` on 80 CUs the GPU is *under-utilized*, so
doing less work per workgroup just makes CUs go idle *sooner*; it doesn't shorten the critical path
(which is gated by per-workgroup latency, not total work).

The bound helps most when the grid is **large** (many workgroups competing for CUs) — i.e. it
**compounds with diagonal-pair** (Lesson 16), which fills the machine and also rebalances the
triangle. Optimizations interact: a work-cut that's invisible alone becomes real once occupancy is
fixed.

## The habit
Separate two questions: *"did I reduce work?"* (count iterations/instructions) and *"did I reduce
time?"* (wall clock). They diverge whenever the machine isn't saturated. Check grid-vs-CU occupancy
before concluding a work-cut "didn't help."

## Next
Lesson 15: ping-pong double-buffer + prefetch to hide load latency.
