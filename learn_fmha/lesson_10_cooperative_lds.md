# Lesson 10 — Cooperative K/V staging into LDS (and the barrier cost)

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_10_cooperative_lds.py` (dumps the real
8wave kernel's ISA). Shows `ds_write` (coop store) + `s_barrier` guarding the shared tile.

## The trick
With multiple waves per workgroup (Lesson 08), all waves need the **same** K/V tile. Either each wave
re-loads it from global (N×redundant traffic), or the waves **cooperate**: each thread loads one slice
into LDS, **one `gpu.barrier()`**, then all waves read the shared tile from LDS. See `load_kv_regs` +
`store_kv_to_lds` + the cooperative pass loop in `kernels/fmha_prefill_fp8_8wave.py`.

## The trade-off
- **Win:** removes redundant per-wave global loads.
- **Cost:** a workgroup `s_barrier` per tile so every wave sees the store before reading. A barrier
  stalls *all* waves until the slowest arrives.

## The hard-won result (don't stage what isn't the bottleneck)
Adding **K→LDS staging alone regressed** the kernel in this project's history: the barriers it
introduced cost more than the redundant K loads they saved, *because K traffic wasn't the
bottleneck* (nk=1, K is small). Cooperative LDS only pays off once (a) the thing you stage is
actually a hotspot (the V byte-gather, Lesson 11) **and** (b) the barrier is hidden by pipelining
(Lesson 15). **Rule: don't move data into LDS unless the profile says that data movement is the
binding cost.**

## FlyDSL mechanics
- LDS allocated via `SmemAllocator`/`SmemPtr`; **LDS indices use `fx.Index`**, not `fx.Int32`.
- `gpu.barrier()` = workgroup sync. Recreate `SmemPtr` views inside `scf.for` (Lesson 07 gotcha).

## Next
Lesson 11: store V *transposed* in LDS so GEMM2 reads are wide (kill the byte-gather).
