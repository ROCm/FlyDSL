# Optimization Agent Prompt — FP8 Paged FMHA Prefill (FlyDSL, MI308X)

> Paste everything below the line into a fresh agent. It is self-contained but points to the full
> handoff (`FMHA_FP8_OPTIMIZATION_HANDOFF.md`) for depth.

---

## Your mission

You are optimizing an **FP8 paged causal FMHA *prefill* kernel** written in **FlyDSL** (a Python/MLIR
GPU DSL) for **AMD MI308X (gfx942 / CDNA3, 80 CU, 4 SIMD/CU, 256 VGPR/thread, 512 VGPR-banks/SIMD,
64 KB LDS/SIMD, ~5.3 TB/s HBM, ~1.3 PFLOPS fp8)**. Goal: **close the throughput gap to the
reference implementations without breaking correctness.** The customer (AITERKER-112, HunyuanVideo
3.0, TP8) only cares about **batch size = 1**, head config **nq=8, nk=1, head_dim=128, causal**.

**Read `/workspaces/amir/FlyDSL/FMHA_FP8_OPTIMIZATION_HANDOFF.md` end-to-end before touching code.**
It has the file map, every experiment with evidence, hardware-counter data, and exact run commands.
Do not re-derive what it already establishes. (That doc's §0 is the full repository map.)

## Where everything is (orientation)

Workspace root: **`/workspaces/amir/`**.
- **`FlyDSL/`** — **the project you edit.** Python/MLIR GPU-kernel DSL. Our kernels are in
  `FlyDSL/kernels/fmha_prefill_fp8*.py`; tests/bench in `FlyDSL/tests/kernels/`. DSL conventions in
  `FlyDSL/CLAUDE.md`, authoring API in `FlyDSL/docs/kernel_authoring_guide.md`.
- **`kernels/pyisa/gfx942/fmha_prefill/f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv/`** —
  **THE reference kernel you're chasing**, hand-written gfx942 ISA in Python. Read `helpers.py`
  (`core_loop`, `gemm_QK`, `gemm_PV`, `process_current_work`, diagonal-pair tiling) + `constants.py`
  (kTileQ=256, kTileKV=128, kWaves=8) + `kernel.py`. This is the structure behind the 36–83 TF bar.
- **`PyISA/`** — the assembler framework that compiles the above (you won't edit it).
- **`asm/`** — compiled reference: `fwd_fp8` (exe), `fwd_causal.co` (binary, loaded by relative
  path → run from `asm/`), `fwd_causal.s` (disassembly = the real instruction schedule to learn from).
- **`aiter/`** — provides the **CK-Tile fp8** kernel we benchmark against (`op_tests/test_batch_prefill.py`).
- **`mlse-tools-internal/performance/kernel_optimization/`** — runbook + FlyDSL skills + profiling
  toolkit (`source setup_env.sh` first).

All three (PyISA, CK-Tile, ours) compute the same fp8 paged causal FMHA prefill; the torch ref
`FlyDSL/tests/kernels/fmha_prefill_fp8_ref.py` defines correctness.

## Where things stand (do not re-measure to confirm — it's verified)

- **Best kernel:** `kernels/fmha_prefill_fp8_8wave.py` (BM=128, 4 waves, `FMHA_NWAVES=4`). All 9
  pytest cases pass; err ≤ 0.055 (fp8 noise floor).
- **Perf, bs=1 (TFLOPS):** ours 5 @sq1024, 13 @sq2048, 47 @sq16384, 53 @sq32768.
  **Bars to beat:** CK-Tile fp8 = 30 / 62 / 141 / 146; PyISA asm = 36 @sq1024, 83 @sq2048.
- **We are ~3× behind CK at large seq.** This is currently a *structural* gap, not a missing tweak
  (see "hard evidence" below). Your job is to find whatever still moves it.

## Hard evidence (from PMC + compiled binary — treat as ground truth)

- PMC on the best kernel (sq4096): **LDS-wait = 54% of busy cycles, VALU:MFMA = 23.5:1.**
  NOT memory-bound (~12.5 GB/s of 5300), NOT compute-bound (~1–2% fp8 peak), NOT occupancy-starved
  at large grids.
- **VGPR = 164** → occupancy limiter (512/164 = 3 waves/SIMD). LDS = 16 KB (not limiting).
- The GEMM2 transpose (O = Vᵀ@P needs kv contiguous per lane; V is stored kv-strided) is
  **unavoidable for either MFMA orientation**, and on gfx942 **every** transpose mechanism (LDS
  scatter-write, `ds_bpermute`, `ds_swizzle`) uses the DS hardware unit → all count toward the 54%
  LDS-wait. gfx950+ `ds_read_tr` (HW transpose-load) would remove it; gfx942 has nothing. So the
  LDS-wait can only be **hidden via overlap**, not removed.
- **PyISA does NOT wave-specialize QK vs PV** (a tempting but wrong reading of the source — verified
  in `kernels/pyisa/.../helpers.py`). It q-partitions waves exactly like us. Its speed comes from
  (1) a hand-scheduled **2-rep software pipeline**, (2) interleaved **partial `s_waitcnt lgkmcnt(N)`**
  inside the MFMA loops, and (3) the **diagonal-pair tile** trick.

## Proven dead-ends — DO NOT repeat these

| tried | result |
|---|---|
| `waves_per_eu` / `maxnreg` compile-hints | **no-op in flydsl 0.2.0** (only effective via external LLVM codegen) |
| 128-KV-per-softmax grouping (`fmha_prefill_fp8_v5.py`) | regressed (VALU wasn't the wall) |
| 2-rep software pipeline as a plain restructure (`fmha_prefill_fp8_v6.py`) | regressed: VGPR 164→215, compiler won't interleave across the online-softmax dependency |
| wide 128-bit global loads | neutral (not bandwidth-bound) |
| cooperative-K→LDS without pipelining | regressed (added barriers for a non-bottleneck) |
| LDS shrink (removed dead P scratch) | neutral (kept as cleanup; LDS isn't the limiter) |

## Most promising levers (in priority order)

1. **Diagonal-pair tiling (tg_div=2)** — the main PyISA trick we never ported. Each workgroup
   processes q-tile `t` AND its causal mirror `num_tiles-1-t`. Balances the causal triangle (early
   q-tiles do little KV work, late ones do lots) → fuller machine at low grid + 2× work/wg.
   Independent of the LDS/scheduling walls. **Start here.** Watch VGPR/occupancy — doubling per-wg
   work can blow the 3-waves/SIMD budget (that killed earlier wide-tile attempts).
2. **External LLVM codegen** (`FLYDSL_COMPILE_LLVM_DIR`) to unlock `--amdgpu-num-vgpr` /
   `--amdgpu-waves-per-eu`, then trade VGPR for occupancy (3→4 waves/SIMD) to hide the DS-wait.
   Requires building/pointing at an external LLVM; payoff uncertain but a clean controlled test.
3. **Cut VGPR live state** (e.g. recompute `q_i64` instead of holding 32 regs across the kv loop) to
   raise occupancy without external LLVM. Likely small; measure.
4. **Per-shape dispatch**: small seq wants smaller workgroups; large seq wants max MFMA density.
5. **gfx950 path** (only if such hardware is available): `ds_read_tr` removes the transpose entirely
   — the single biggest structural win, but a different target.

## Workflow & rules

- **GPU 2 only** (`HIP_VISIBLE_DEVICES=2`; GPU 0 is broken).
- **Keep `fmha_prefill_fp8_8wave.py` as the untouched baseline.** Do each experiment in a NEW file
  `kernels/fmha_prefill_fp8_v7.py` (and v8, …), each with a UNIQUE `SmemAllocator(global_sym_name=...)`.
- After every change:
  1. **Correctness first** — err < 0.06 vs `tests/kernels/fmha_prefill_fp8_ref.py`, across odd tile
     counts, sk≠sq, non-causal, batch>1, GQA, p_scale. Use ONE shape per process (the module-global
     smem can't re-finalize across shapes in-process). Template runner is in handoff §10.6.
  2. **Extract resources** — `vgpr_count` / `group_segment` / `vgpr_spill_count` from
     `/tmp/isa/attn_kernel_0/19_gpu_module_to_binary.mlir` (dump via `FLYDSL_DUMP_IR=1
     FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0`).
  3. **Benchmark** — `tests/kernels/bench_fmha_compare.py --kernels <yours> --ck --no-pyisa
     --seqs 1024 16384 32768` (PYTHONPATH=/workspaces/amir/aiter). PyISA only for seqs ≤ 2048.
  4. **If perf moves, re-take PMC** (handoff §4) to see whether the 54% LDS-wait / VALU ratio shifted.
- **Log every variant** — hypothesis / correct? / TFLOPS per seq / VGPR / LDS / why it worked or
  didn't. Negative and neutral results are valuable here; the search space is mostly walls.
- Don't claim a speedup without a before/after benchmark on the same shapes. Don't merge anything
  that regresses the customer shapes (sq 1024 / 16384 / 32768, bs=1, nq8 nk1 causal).
- Stop and report if you hit a FlyDSL expressiveness limit (e.g. you need per-instruction waitcnt
  scheduling and the DSL won't emit it) — that's a finding, not a failure.

## Quick commands (full set in handoff §10)

```bash
export HIP_VISIBLE_DEVICES=2
cd /workspaces/amir/FlyDSL

# correctness (all 9, subprocess per shape):
python3 -m pytest tests/kernels/test_fmha_prefill_fp8.py -q

# bench vs CK (+ PyISA on small seqs):
PYTHONPATH=/workspaces/amir/aiter FMHA_NWAVES=4 python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_8wave --ck --seqs 1024 2048
PYTHONPATH=/workspaces/amir/aiter FMHA_NWAVES=4 python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_8wave --ck --no-pyisa --seqs 16384 32768

# PyISA reference (must run from asm/):
cd /workspaces/amir/asm && ./fwd_fp8 causal=1 nheads=8 nheads_k=1 seq_len=2048

# resources of a compiled kernel:
rm -rf /tmp/isa && FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python3 <your_runner>.py
grep -o "vgpr_count = [0-9]*\|group_segment_fixed_size = [0-9]*\|vgpr_spill_count = [0-9]*" \
  /tmp/isa/attn_kernel_0/19_gpu_module_to_binary.mlir
```

## Environment notes

- flydsl **0.2.0** via pip (do NOT build from source). If aiter was installed it DOWNGRADES flydsl
  to 0.1.9 → run `pip install --upgrade flydsl==0.2.0` and re-verify the kernel.
- PyISA `asm/fwd_fp8` always runs a slow CPU reference → only timeable up to ~sq2048; `validate=0`
  skips the whole run (not perf-only).
- CK-Tile via aiter: first `--ck` run JIT-compiles (~5 min); pass `dtype=torch.bfloat16`
  (fp8 quant is internal), `table_layout="sglang"`.
- `nohup` does not survive in this env — use tracked background or foreground with a long timeout.

## Reference target source (study before designing)

`/workspaces/amir/kernels/pyisa/gfx942/fmha_prefill/f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv/`
— `kernel.py` + `helpers.py` (`core_loop`, `gemm_QK`, `gemm_PV`, `process_current_work`,
diagonal-pair tiling) + `constants.py`. This is the 36–83 TF target you're chasing.
