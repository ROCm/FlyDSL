# Lesson 12 — Register-resident P (ds_bpermute) + the decisive profile reading

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_12_register_p.py` (dumps the real 8wave
kernel's ISA and prints its op mix).

Live op mix (sq1024, per the unrolled hot loop):
```
v_mfma       16
ds_read       8
ds_write     34      <- LDS writes dominate the matrix ops
ds_bpermute  10      <- the "register" P-transpose is itself an LDS-unit op
v_perm       32
```

## The trick
Lesson 07 transposed P through LDS: store P → `barrier` → reload 8-contiguous-kv per lane. The
register-resident version does the transpose **across lanes** with `ds_bpermute` (a lane reads
another lane's register), skipping the explicit P store/reload. `fmha_prefill_fp8_8wave.py` uses
4 `ds_bpermute` per k-step. Open it and read the "register-resident P transpose" block.

## The decisive lesson (bigger than the trick)
Profiling the result is the turning point of the whole optimization effort. **Register-P did *not*
make the kernel fast** — and the PMC says why. Measured on the 8wave kernel (sq16384, `rocprofv3`):

```
SQ_WAIT_INST_LDS / SQ_BUSY_CU_CYCLES  ≈ 54%    # HALF of all cycles wait on the LDS unit
SQ_INSTS_VALU    / SQ_INSTS_MFMA      ≈ 23:1   # ~23 ALU ops per matrix op
```

`ds_bpermute` is **itself an LDS-unit op**. So "register-P" just **moved** LDS traffic from explicit
read/write to bpermute — it didn't remove it. The kernel is **LDS-bound and VALU-heavy**, *not*
memory-bound (Lesson 09: ~12 GB/s of 5300) and *not* compute-bound (~1–2% of fp8 peak). Every
transpose mechanism on gfx942 (LDS scatter, `ds_bpermute`, `ds_swizzle`) hits the same DS unit.

**This reading is what pointed at the real fix (Lesson 17): don't *move* the transpose, *delete* it**
by storing V column-major.

## How to read PMC yourself
`pmc.txt`:
```
pmc: SQ_WAVES SQ_BUSY_CU_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_WAIT_INST_LDS SQ_INSTS_LDS
```
```
rocprofv3 -i pmc.txt -- python3 <run_one_shape>.py
sqlite3 *_results.db \
  "SELECT n.name,SUM(e.value) FROM rocpd_pmc_event e JOIN rocpd_info_pmc n ON e.pmc_id=n.id GROUP BY n.name;"
```
Then compute the two ratios above. **LDS-wait/busy** and **VALU:MFMA** are the two numbers that
classified this kernel and redirected the whole effort.

## The habit
A change that is locally "better" (fewer explicit LDS ops) can be globally neutral if it relocates
the bottleneck instead of removing it. Always re-profile after a change and ask: *did the binding
counter actually drop, or did I just shuffle work between units?*

## Next
Lesson 14: skip fully-masked kv tiles — and why cutting *work* doesn't help an *under-occupied* machine.
