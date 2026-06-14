# Lesson 11 — V transposed in LDS: byte-gather → wide reads

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_11_v_transpose_lds.py` (dumps the 8wave ISA).
Shows `buffer_load_ubyte = 0` (byte-gather gone) and `ds_write_b8 = 32` (the transpose scatter that
now dominates LDS writes).

## The problem
GEMM2 contracts over kv, but V is stored `[kv, d]` row-major. A naive GEMM2 gathers V **one byte at a
time** per (kv, d) — the original kernel had **64 `buffer_load_ubyte` per tile**, completely
uncoalesced. That byte-gather was a measured hotspot.

## The trick
Cooperatively store V into LDS **transposed** (column-major, d-contiguous). Then each GEMM2 V operand
is a single wide LDS read instead of 8+ scalar byte loads. This is the V-side twin of the P-transpose
(Lesson 12) — both pay the transpose on the **DS unit**.

## What the ISA shows (and the catch)
- `buffer_load_ubyte`: **64 → 0** — the uncoalesced byte-gather is gone.
- But `ds_write_b8`: now **32+** per tile — the scalar **scatter that BUILDS the transpose** is now
  the dominant LDS-write op. We traded uncoalesced global byte-loads for LDS byte-scatters.

So we *moved* the cost onto the DS unit, which Lesson 12's PMC then shows is the kernel's bottleneck
(LDS-wait ≈ 54%). Lesson 20 tries to make the scatter wide with `v_perm` and still loses; **Lesson 17
removes it entirely** by storing V column-major in *global* memory so even the LDS transpose
disappears.

## The arc this sets up
11 (transpose via LDS scatter) → 12 (transpose via ds_bpermute, same DS unit) → 20 (transpose via
v_perm wide stores, *still* DS-bound, regresses) → **17 (no transpose: column-V)**. Watching one cost
get shuffled between mechanisms until it's finally *deleted by a layout change* is the central story
of Part D.

## Next
Lesson 12: the register-resident P transpose — and the PMC reading that reframed everything.
