# fp8 FlashAttention Forward (gfx950 / CDNA4) — Design & Optimization Notes

This document describes the fp8 (OCP `float8_e4m3fn`) FlashAttention forward path
added to the gfx950 dual-wave software-pipelined (SWP) kernel
(`kernels/flash_attn_gfx950.py`), its numerics contract, validation status, and a
profiling-driven optimization roadmap toward aiter-asm-fp8 throughput parity.

## Scope

- **In:** dense forward, `head_dim == 128`, causal and non-causal, MHA and GQA,
  per-tensor descales, fp32 accumulation, bf16 output, gfx950 only.
- **Out (rejected with a clear error, not silently skipped):** fp8 split-K
  (`num_kv_splits > 1`) and fp8 packed varlen (`cu_seqlens`).
- **Additive:** the existing bf16/f16 paths are byte-identical when `dtype != fp8`;
  the gfx942 fallback, varlen, split-K, and GQA/MQA for bf16/f16 are untouched.

## Numerics contract

- Inputs Q/K/V arrive **pre-quantized** to e4m3fn (no in-kernel quantization).
- Per-tensor shape-`[1]` fp32 descales `q_descale`, `k_descale`, `v_descale`
  (launch kwargs), mirroring aiter's per-tensor fp8 ABI.
- QK uses the native `mfma_f32_32x32x16_fp8_fp8` atom; `q_descale*k_descale*sm_scale`
  is applied to the fp32 logits.
- Online softmax running max/sum and the PV accumulator stay **fp32**.
- PV applies `v_descale`; output is **bf16**.

## PV operand precision — two modes

The CDNA4 `32x32x16` MFMA inventory has only 8-bit×8-bit fp8 variants (no mixed
bf16×fp8 atom), so a fp8 V operand forces an 8-bit P operand. Two PV modes exist,
selected by mutually-exclusive env flags (invalid combinations fail fast):

1. **High-precision-P (default, shipping).** fp8 V is dequantized to bf16
   in-kernel (`* v_descale`) and PV runs as a bf16 MMA with bf16 P. Passes the
   fixed fp8 gate at very high cosine similarity. This is the default.
2. **Packed-fp8-P (FROMBF16, opt-in correctness path).** Reuses the proven bf16 V
   transpose layout, then quantizes both the V operand and the softmax
   probabilities P to fp8 e4m3 in-register and runs a genuine `fp8×fp8` PV MMA.
   This proves on-device that 8-bit P is sufficient for the fixed gate when P is
   packed as the unnormalized `exp(S - m)` (not normalized probabilities, which
   land in e4m3 subnormals and collapse accuracy).

A key finding during bring-up: an earlier hypothesis that 8-bit P imposed a
fundamental precision wall was **false**. A host-side numerics check of the exact
`fp8×fp8` PV MMA showed the unnormalized-exp packing the kernel emits reaches a
cosine far above the gate; the on-device shortfall of an experimental
key-contiguous fp8-V staging path is a layout/indexing defect, not a precision
limit. The FROMBF16 mode realizes the achievable accuracy by reusing the proven
element order.

## Validation

Validated on MI350X (gfx950) against a dequantized-input PyTorch SDPA reference:

- **fp8 correctness gate (fixed, not relaxed):** `max_err < 5e-2` AND
  `min_cos > 0.98`, no FAIL/ERROR rows.
- Default high-precision-P fp8: full dense sweep passes (causal/non-causal,
  MHA + GQA Hkv∈{8,16,32,64}, S up to 8192), `min_cos ≈ 0.99999`.
- FROMBF16 packed-fp8-P: full dense sweep passes, `min_cos ≈ 0.9986`.
- bf16/f16 non-regression: unchanged (`min_cos ≈ 0.99999` / `1.00000`).
- Full repo gate (`RUN_TESTS_FULL=1 scripts/run_tests.sh`): pytest + examples +
  MLIR FileCheck all green.

Reproduce:

```bash
# fp8 correctness (default high-precision-P)
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --warmup 3 --iters 5
# fp8 vs aiter asm fp8 / aiter ck fp8 / FlyDSL bf16
python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare --warmup 10 --iters 50
```

## Performance status (honest)

Throughput parity with aiter's native fp8 ASM kernel is **not yet reached**. On
MI350X at B=1, S=2048, H=64, D=128, non-causal (warmup 10, iters 50; aiter native
ASM via `fmha_v3_fwd`, `how_v3_bf16_cvt=0`):

| Path | TFLOPS | % of aiter asm | min_cos |
|---|---|---|---|
| aiter asm fp8 (parity target) | ~1306 | 100% | 0.9993 |
| aiter ck fp8 | ~919 | 70% | 0.9993 |
| **FlyDSL fp8 (high-precision-P, default)** | ~863 | **66%** | 0.99999 |
| FlyDSL fp8 (FROMBF16 packed-P) | ~620 | 47% | 0.9986 |

The FROMBF16 path is intentionally a correctness vehicle, not the perf path: fp8
and bf16 `32x32x16` MMA have equal CDNA4 throughput, so packing P/V to fp8 only
helps once the bf16 round-trip and per-MMA conversion are removed.

## Profiling evidence (rocprofv3 ATT thread trace)

ATT thread traces of the default high-precision-P kernel (same shape) show it is
**stall-bound, not compute-bound** — MFMA issue is a small fraction of cycles:

| Stall class | High-precision-P | FROMBF16 |
|---|---|---|
| total stall | 64.7% | 56.4% |
| barrier | 33.6% | 42.0% |
| vmcnt (global-load wait) | 20.5% | 13.9% |
| valu | 13.9% | 23.1% |
| lds | 12.6% | 7.9% |
| **mfma** | **7.6%** | **4.6%** |

Two readings:
- The dominant bubbles are cross-wave synchronization (`s_barrier`) and global-load
  waits, with MFMA units idle most of the time. Closing the gap to aiter asm is a
  scheduling/synchronization problem, not an arithmetic-throughput one.
- FROMBF16's `valu` share jumps (13.9% → 23.1%), directly attributable to the
  per-MMA bf16↔fp8 conversion — confirming why it is slower than the default path.

Ranked recommendations from the trace (auto-generated, with source lines):
1. **Relax redundant synchronization** (barrier ≈ 33.6%): drop `s_barrier`s that do
   not guard a real cross-wave LDS dependency; in ping-pong, barrier only the
   swapped LDS region; prefer shuffle/DPP reductions over LDS cross-wave reduces.
2. **Raise occupancy** (≈ 4 waves/CU): move staging loads through LDS via async copy
   to free staging VGPRs, hiding the exposed memory/LDS latency with more resident
   waves.
3. **Deepen VMEM prefetch** (vmcnt ≈ 20.5%): increase prefetch distance on the K/V
   global loads so `s_waitcnt vmcnt(...)` is less exposed.

## What is not done / next steps

1. **True-fp8 no-roundtrip V path (primary).** Stage and read fp8 V directly in the
   proven element order — no bf16 vt round-trip, no per-MMA conversion — halving the
   `ds_read` bandwidth for V and removing the `valu` conversion overhead. The open
   problem is that the 8-bit transpose load permutes lanes differently from the
   bf16 transpose-load + shuffle, so the fp8 B-operand element order must be matched
   to the proven layout (via an operand-dump oracle), then re-benchmarked vs aiter
   asm fp8.
2. **Barrier relaxation and prefetch deepening** per the profiling recommendations.
3. **fp8 split-K and packed varlen** (currently rejected) once dense parity lands.

The default high-precision-P path is correct, additive, and merge-ready as a
functional fp8 forward; the throughput-parity work above is the follow-up.
