# Lesson 09 — Wide global loads: when they help, and when they don't

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_09_wide_loads.py`

```
NARROW (4x32-bit)   err=0.0e+00   100.8 us   333 GB/s
WIDE (128-bit)      err=0.0e+00    96.3 us   348 GB/s
```

## The trick (the whole diff)
One parameter: `vec_width`. NARROW = four 1-element 32-bit loads per lane; WIDE = one 4-element
128-bit load (`buffer_load_dwordx4`). Same bytes, same result. On this memory-bound 16 MB copy, WIDE
is ~5% faster — fewer load instructions, better coalescing into 128-bit memory transactions.

## Why it's an A/B *microkernel*, not a step in the attention kernel
The trick is a single argument; on a tiny copy kernel you see it cleanly. The point isn't "wide loads
are good" (everyone knows that) — it's the **method** around it.

## The real lesson: a good trick only helps if the kernel is bound by what it improves
In the production attention kernel, switching K/V to 128-bit loads was **NEUTRAL** — zero speedup.
Why? Attention here runs at **~12 GB/s of 5300** (measured): it is **not** memory-bound. It's
compute/LDS-bound (Lesson 12's PMC shows LDS-wait ≈ 54% of busy). Widening a load that wasn't the
bottleneck does nothing. The copy kernel *is* memory-bound, so the same change helps.

**Takeaway:** before applying any "generically good" optimization, measure whether the kernel is
bound by the thing that optimization improves:
- **memory-bound?** → check achieved GB/s vs ~5300. Wide loads, coalescing, fewer bytes help.
- **compute-bound?** → check VALU:MFMA, MFMA utilization. Wide loads won't help; reduce ALU / raise
  MFMA density.
- **LDS-bound?** → check LDS-wait/busy. Wide loads won't help; cut LDS traffic / hide it.

This is the single most important habit in the whole tutorial: **classify the bottleneck first.**

## How to measure GB/s yourself
`do_bench` gives median ms; `GB/s = bytes / time`. For this copy, `bytes = 2·N·4` (read + write).
For attention, compute the real K/V/Q/O byte traffic and compare to 5300 — if you're at single-digit
%, stop thinking about load width.

## Next
Lesson 08 (we present it after 09 deliberately): the first *structural* perf win — widening from one
wave to a multi-wave workgroup, and reading occupancy from the profile.
