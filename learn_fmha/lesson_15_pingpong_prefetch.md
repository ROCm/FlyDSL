# Lesson 15 — LDS ping-pong double-buffer + prefetch

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_15_pingpong_prefetch.py` (dumps the 8wave
ISA; shows the prefetch loads and the per-tile barrier).

## The trick (classic software pipeline)
Cooperative LDS (Lesson 10) added a barrier per kv-tile, and the global→LDS load has hundreds of
cycles of latency the GEMMs would stall on. Two combined moves in
`kernels/fmha_prefill_fp8_8wave.py`:
- **Ping-pong:** two LDS buffers. Compute reads buffer `tile&1` while the **next** tile loads into
  `(tile+1)&1` — no waiting for the store before computing.
- **Prefetch (OPT3):** *issue* the next tile's global loads early (during this tile's
  GEMM1/softmax/GEMM2) so the ~300-cycle latency overlaps compute; *defer* the LDS store until after
  the GEMMs so it doesn't stall right before GEMM1.

This is exactly the shape CK Tile's pipeline driver implements. The "write→barrier→read" chain
becomes "load-next while compute-now."

## The result, and the lesson
On *this* kernel the measured gain was small (**+0.3 TF**). Why so little? The bottleneck is the LDS
**transpose** (Lesson 12: LDS-wait ≈ 54%), **not** global-load latency — so hiding the load latency
barely touches the binding cost. Once the transpose is removed (column-V, Lesson 17), latency-hiding
matters more.

**Same recurring habit:** a textbook-correct optimization (software pipelining) gives little when it
doesn't target the binding bottleneck. Pipelining hides *load* latency; this kernel was bound by *LDS
transpose throughput*. Profile to know which.

## Why it's still worth doing
Even at +0.3 TF it's free insurance, and it's a prerequisite for kernels that *are* load-latency
bound (decode, memory-bound GEMM). The technique is general; its payoff is shape/kernel-specific.

## Next
Lesson 16: diagonal-pair tiling — the first big structural win.
