---
name: vgpr-pressure-analysis
description: Analyze AMDGCN VGPR/AGPR register pressure of a FlyDSL kernel from its ISA dump — report the compiler high-water (vgpr/agpr/accum_offset/spill/headroom) AND a per-purpose breakdown of where the arch VGPRs go (A/B fragment ds_read, address ALU, scale loads, epilogue), plus the true peak-pressure region. Use when the user asks "how many VGPRs does X use", "where is the register pressure", "为啥 vgpr 这么高 / 占用统计 / 寄存器压力", is chasing spill, or wants to free registers (deeper prefetch, more occupancy) on CDNA GEMM/attention kernels.
---

# VGPR Pressure Analysis

Find out where a kernel's registers go so you can target what to cut (to kill
spill, deepen a prefetch, or raise occupancy). Works from the ISA dump, using the
compiler's own allocation — not guesses.

## Mental model (CDNA3/4, wave64)

- A SIMD has **512 VGPRs total**, split into **arch VGPRs** (v0..accum_offset-1)
  and the **AGPR file** (accum_offset..511). MFMA accumulators normally live in
  AGPR (pin them there — see [[feedback-4wave-1occ-is-optimal]]).
- So a 256x256 tile that needs **256 AGPR for accumulators** leaves only **256
  arch VGPRs** for everything else (A/B fragments, scale, addresses, temps).
  That ~256 is the budget you actually fight over.
- `accum_offset` in the ISA metadata **IS the compiler's arch-VGPR high-water**
  (the AGPR file starts right after the last arch VGPR). Trust it over hand counts.
- **spill** (`.vgpr_spill_count > 0`) is catastrophic on the hot loop — treat any
  spill as the top priority. Always analyze the BIG shape (small shapes / short K
  loops often don't spill and mislead).

## How to get the ISA dump

```bash
rm -rf /tmp/kdump
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 \
  FLYDSL_DUMP_DIR=/tmp/kdump PYTHONPATH=<repo> HIP_VISIBLE_DEVICES=<n> \
  python3 <bench_script>            # use a realistic/large shape
# -> /tmp/kdump/<kernel>_0/21_final_isa.s
```

## Run the analysis

```bash
python3 .claude/skills/vgpr-pressure-analysis/scripts/vgpr_pressure.py /tmp/kdump
```

Three sections:
1. **high-water** — vgpr/agpr/accum_offset/sgpr, spill flags, headroom (512-total).
2. **arch-VGPR breakdown** — each arch VGPR bucketed by the op that LAST defines it:
   `ds_read` (A/B fragment data), `address/index ALU`, `buffer_load dword` (scale),
   `cvt` (epilogue), `mov/accvgpr`. This tells you the proportion: e.g. fragments
   136, address ALU 89, scale 23.
3. **peak region** — the instructions that touch the very top arch VGPRs
   (those numbers only get allocated at the pressure peak), so you see exactly
   what is co-live at the worst point.

## Reading the result / what to cut

- **`mov/accvgpr` or `MFMA dest` showing many ARCH regs** => accumulators are NOT
  AGPR-pinned (compiler spilled them to arch VGPR + inserted v_accvgpr_mov/read).
  Fix: pin the accumulator in AGPR via inline asm (`=a,...,0`). Biggest single win.
- **`ds_read (A/B fragment)` dominates** => that's real operand data; to cut it you
  must reduce *simultaneously-live* fragments (e.g. don't pre-load all M/N tiles up
  front; load some inside the loop) — costs scheduling latitude, do only if spilling.
- **`address/index ALU` surprisingly high** => many live address/offset registers
  (LDS swizzle bases, per-buffer LDS bases, scale addresses). Often reducible by
  sharing one base + immediate offsets (ds_read/buffer_load have immediate offset
  fields) instead of materializing a full address per access. Watch the 16-bit
  immediate-offset limit (65535) — group accesses so the spread fits.
- **`buffer_load dword (scale)`** => per-block scale; carrying it as loop-carried
  (prefetch) costs regs but hides latency. Trade-off: dropping the carry frees ~8
  regs but exposes scale latency (measure both).

## Caveats

- The breakdown buckets by *last* defining op, so a reg reused across phases is
  attributed to its final use — it's a proportional map of pressure, not exact
  per-instant liveness. For exact peak liveness trust `accum_offset`.
- Re-dump after every change (cache off) and re-check spill on the big shape.
- Related: [[project-fp4-gemm-4wave-status]] has a worked example (arch 252:
  fragment 136 / address 89 / scale 23).
