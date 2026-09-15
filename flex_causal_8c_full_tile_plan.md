# Implemented: 8c on full KV tiles, causal mask only in the tail

Status: implemented and measured 15 Sep 2026. See
`flex_gfx950_optimizations.md`, step 9, for correctness, latency, and PMC.

Goal: `_flash_deferred_step_8c` never sees causal mask. Interior tiles
are fully valid (`kv_tile_end <= q_min`). The diagonal band is a **masked
4c tail**, then the existing **PV drain** epilogue. Mask is not applied
inside `_flash_deferred_epilogue`.

Kernel: `kernels/attention/flex_attention_gfx950.py`.
Do not change `_LONG_SEQ_8C_SKV_*` unless a later bench shows a ≥20 MHz
or clear µs win.

## Why

Causal live range is already `[_kv_lo, _kv_hi)`. Inside that, only a thin
band (~4 tiles/WG) needs element-wise mask. Today 8c still runs
`tile_needs_mask` (`kv*64+63 > q`) in C1 and C5 on **every** pair (~4.5
INT32/wave/tile). Packed cmp is already skipped when the flag is false;
the predicate is not.

Pinned N=128: MFMA/LDS/VMEM match flash; ~66 µs follows `SQ_BUSY_CYCLES`.
This split will not by itself close that gap. It does make 8c a straight
full-tile machine and confines causal VALU to the tail.

## Definitions (WG-uniform, same `q_min` as today)

```
q_min  = q_tile * 256 + (Skv - Sq)     # _causal_wg_q_min
kv_hi  = min((q_max + block_n) // block_n, n_kv_tiles)   # existing
kv_full_hi = first tile with (tile * 64 + 63) > q_min
           = (q_min - 63) <= 0 ? 0 : (q_min - 63) // 64 + 1
full   = [_kv_lo, min(kv_full_hi, _kv_hi))   # no mask
diag   = [min(kv_full_hi, _kv_hi), _kv_hi)   # must mask S before softmax
```

`kv_full_hi` and `_kv_hi` are loop-invariant SGPRs (q_tile, seq lens).
Compute once next to `_kv_lo, _kv_hi`.

Examples (Sq=3072, block_n=64, 256 rows/WG):

| WG | Skv | full tiles | diag tiles |
| --- | --- | --- | --- |
| last (q_min=7936) | 8192 | 0–123 | 124–127 |
| first (q_min=5120) | 8192 | 0–78 | 79–84 |
| first (q_min=0) | 3072 | none | 0–3 |

When `full` is empty, skip 8c entirely (today’s short causal WGs).

## Pipeline (do not drain between phases)

Keep deferred softmax: tile n’s `p_mixed` is consumed as PV while tile
n+1 does QK.

```
1. Prologue on _kv_lo
   - If _kv_lo < kv_full_hi: no mask (full tile).
   - Else: mask (prologue is already in the diagonal).
2. 8c steady pairs only while both tiles of the pair are in full.
   Pair still starts at kv_odd = _kv_lo+1 + 2*pair_i (unchanged slotting).
   Last full tile may be odd → one 4c _flash_deferred_step with mask OFF,
   still in the full region (same as today’s _tail_count, but cut at
   kv_full_hi not _kv_hi).
3. Masked 4c: for kv in [kv_full_hi, _kv_hi), _flash_deferred_step
   with mask ON. First of these consumes p_mixed from the last full tile.
   If full was empty, prologue already produced p_mixed under mask.
4. Existing _flash_deferred_epilogue: finish last P@V only. No mask.
```

Implementation note: the original plan proposed capping every 8c K
prefetch at `kv_full_hi`. That is wrong at the phase boundary. The final
8c C2/C6 must stage the first diagonal K tile for the following 4c step.
Only the pre-loop K2 prime is capped at `kv_full_hi`, because when no 8c
pair exists the first 4c step stages K2 itself.

## Code changes (single kernel, compile-time causal)

1. **Range helpers** next to `_kv_lo/_kv_hi` (~1915):
   `kv_full_hi` from `q_min` and `block_n`. Clamp to `_kv_hi`.
   `_full_range = kv_full_hi - _kv_lo` (max 0).

2. **Gated `apply_mods` in 8c / full 4c:**
   Add `_mask_scores_this_tile` constexpr or a runtime SGPR
   `_in_full_region` is wrong for a mixed loop; better: **two call
   sites**.
   - `_flash_deferred_step_8c`: **never** call
     `_flash_apply_mods_and_mask` (causal + 8c only). Dense 8c
     unchanged (already MASK_NONE).
   - `_flash_deferred_prologue` / `_flash_deferred_step`: keep
     `apply_mods`, but prologue/full-tail 4c pass `apply_mask=False`
     when the tile index `< kv_full_hi`.

   Smallest API: `_flash_apply_mods_and_mask(..., force_no_mask=False)`
   with `force_no_mask` constexpr False except 8c always True for
   causal, and full-region 4c True.

3. **Rewrite the deferred-softmax loop** (~2701–2840) for
   `long_seq_8c and MASK_CAUSAL` only. Dense 8c keeps the current
   `_kv_hi` pairing.

   Causal 8c counts:
   ```
   full_remaining = max(full_range - 1, 0)  # prologue ate _kv_lo if full
   if _kv_lo >= kv_full_hi:  # prologue was diagonal
       8c pairs = 0
       masked steps from _kv_lo+1 .. _kv_hi-1
   else:
       8c pairs = full_remaining // 2
       optional full 4c tail if full_remaining odd
       then masked 4c for kv_full_hi .. _kv_hi-1
   ```
   Then current `_tail_count` / `_no_tail_count` PV drain, keyed off
   **last processed slot**, not `_kv_hi` parity of the old loop.

4. **Prefetch caps** in `_flash_deferred_step_8c` C2/C6 and the
   pre-loop `load_k(_kv_lo+2)`: compare against `kv_full_hi` when
   causal 8c.

5. **Do not** add mask to `_flash_deferred_epilogue`.
   **Do not** `scf.if` around C4–C7.
   **Do not** mix 4c/8c by odd/even K index.

Sliding window / prefix-LM: out of scope. They keep today’s 8c+mask
unless we later add a similar “interior band” split.

## Correctness

`exp_causal_packed_mask.py` shapes plus the 8c-heavy cases:

- Sq=Skv=256, 512, 3072 (empty or tiny full prefix)
- Sq=3072 Skv=2048 (grid shrink + mask band)
- Sq=3072 Skv=3072 (N=48, 8c on)
- Sq=3072 Skv=8192 (N=128, long full prefix)
- First and last Q tile covered by B=2 H=32 (all WGs)

Vs bottom-right SDPA: max err / cosine same bar as packed-mask (err <
8e-2, cos > 0.98). Watch NaNs on the first WG (all-masked prefix already
handled by `_kv_hi` / dead-Q grid).

## Performance (same protocol)

HIP 0 = GPU[3]. Occupancy-held B=2 Sq=3072 H=32 D=128 bf16 n64 g8 pd1.

1. Unpinned causal N=32/48/128 vs flash (packed-mask script).
2. Pin 1400 MHz, time N=48 and N=128.
3. `SQ_INSTS_VALU_INT32` / `SQ_INSTS_VALU` / `SQ_BUSY_CYCLES` at N=128.

Expect: INT32 at N=128 drops toward flash’s ~1.0M (interior A→0). Time:
noise at N=48; N=128 pinned gap vs 66 µs — land only if time improves
beyond sample std. If INT32 falls and µs does not, keep the split for
structure unless it regresses.

## Flag / rollback

`_CAUSAL_8C_FULL_TILES_ONLY = True` next to
`_CAUSAL_WG_UNIFORM_PACKED_MASK`. Packed-mask stays on for the **tail**
4c path. The flag disables full-prefix mask removal; the common deferred
loop still contains the new zero-or-more 4c tail-pair region.

## Order of work

1. [done] Add `kv_full_hi` (host-side python unit test of the index formula
   vs brute force over q_min, no GPU).
2. [done] Causal-only: strip `apply_mods` from `_flash_deferred_step_8c`;
   cap 8c prefetch and pair count at `kv_full_hi`.
3. [done] Insert masked 4c loop for `[kv_full_hi, _kv_hi)` before PV drain;
   fix slot/p_mixed handoff.
4. [done] Corr suite, then unpinned + pinned + PMC.
5. [done] Update `flex_gfx950_optimizations.md` / canvas with the decision.

No git commit unless asked.
