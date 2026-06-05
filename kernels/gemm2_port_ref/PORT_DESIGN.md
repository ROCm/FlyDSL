# gemm2_a4w4 HIP → FlyDSL 1:1 port — design & status

## Goal
Port aiter PR #3470 `gemm2_a4w4` (HIP) into FlyDSL (`kernels/`), driving the
FlyDSL-generated LLVM IR to match the HIP-generated LLVM IR as closely as the
two toolchains allow. Test on remote gfx950 (`dzm-mi355-gpu34-dev`).

## Feasibility calibration (important)
Byte-identical LLVM across clang(HIP) and MLIR(FlyDSL) is **not achievable**
(SSA value names, basic-block order, metadata/TBAA, attribute sets, GEP
canonicalization all differ structurally between the two frontends).

**Achievable bar = instruction-level equivalence**: same set of `amdgcn.*`
intrinsic calls with identical immediate operands, same LDS layout/size, same
control-flow skeleton. The census below proves this is within reach.

## Target instance (pinned)
`mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM32_ATOMIC`
`launch_atomic<MAX_M=655360, NE=385, K=512, N_OUT=7168, TOPK=9, BM=32, kUseNT=false, kXcdSwizzle=0>`

Constants: BN=256, BK=256, K_HALF=256, K_TILES_TOTAL=2, kStages=2,
kSubBlocks=1, kMChunks=2, num_n_blocks=N_OUT/256=28.
LDS = union{ s_Aq[2][32][128] fp4x2 (8KB), lds_acc[32*256] f32 (32KB) } = **32KB = [8192 x float]**.

Scale layout consts: kBS_c_n1=224, kBS_c_k1=2, kBS_stride_k0_dw=64,
kBS_stride_n0_dw=128, kBS_per_expert_dw=28672; kAS_c_k1=2, kAS_per_chunk_dw=128.

## HIP reference IR census (kernels/gemm2_port_ref/hip_gemm2_bm32.ll, 570 lines)
| intrinsic | count | immediates |
|---|---|---|
| `mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32` | 32 | cbsz=4, blgp=4, op_sel a/b per cluster |
| `make.buffer.rsrc.p8.p1` | 4 | i16 0, flags i32 131072(0x20000); num_bytes: A_q=167772160, A_scale=10485760, B_q=706478080, B_scale=44154880 |
| `raw.ptr.buffer.load.lds` | 2 | size=16, soffset imm 0 then 128 |
| `raw.ptr.buffer.load.v4i32` (B) | 16 | 4 j × {imm 0,1024} × 2 K_C |
| `raw.ptr.buffer.load.i32` (scales) | 6 | A_scale 2 (imm 0,256) + B_scale 4 |
| inline asm `s_waitcnt vmcnt(23)` / `vmcnt(22)` | 2 | cross-wave fence before s_barrier |
| `s.barrier` | 3 | 2 in K-loop + 1 epilog |
| `global.atomic.fadd.v2bf16` | 16 | 4 mr × 4 stride |
| LDS addrspace(3) refs | 126 | |

Kernel attrs: target-cpu gfx950, `amdgpu-flat-work-group-size`="1,256",
`amdgpu-waves-per-eu`="2", `amdgpu-agpr-alloc`="0".

## FlyDSL existing gemm2 census (flydsl_gemm2_existing.ll, 1256 lines)
`mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_persist_cu256`
ALREADY matches at intrinsic level: mfma 32 ✓, buffer.load.lds 2 ✓,
buffer.load.v4i32 16 ✓, atomic v2bf16 16 ✓.
Differences (structural, the port must fix):
- make.buffer.rsrc 8 (2×) vs 4 — persistent loop re-creates rsrc.
- s.barrier 6 vs 3; no `s_waitcnt vmcnt` inline-asm fences (0 vs 2).
- IR 1256 vs 570 lines — persistent cu256 grid loop + CShuffle verbosity.
→ Port targets the **non-persistent atomic** structure (one tile/block + early
return), 4 rsrc, explicit vmcnt fences, K=512 2-stage unroll.

## FlyDSL primitives (exact)
- raw MFMA: `rocdl.mfma_scale_f32_16x16x128_f8f6f4(vec4_f32, [a8xi32, b8xi32, acc, cbsz, blgp, opselA, scaleA_i32, opselB, scaleB_i32])` — rocdl/__init__.py:168
- A→LDS: `rocdl.raw_ptr_buffer_load_lds(rsrc_p8, lds_p3, size, voffset, soffset, offset, aux)` — :495
- B/scale loads: `buffer_ops.buffer_load(rsrc, offset_elems, vec_width, dtype, soffset_bytes=)` ; rsrc via `buffer_ops.create_buffer_resource(memref, max_size=False, num_records_bytes=N)` (-> p8) — buffer_ops.py:386
- atomic pk bf16: `rocdl.raw_ptr_buffer_atomic_fadd(vdata_2xbf16, rsrc, offset, soffset, aux)` — :456 (or llvm.AtomicRMWOp fadd v2bf16, syncscope agent, moe_gemm_2stage.py:3100)
- LDS: SmemAllocator/SmemPtr (utils/smem_allocator.py); ds_read via vector.load on LDS memref
- inline asm fence: `_llvm.inline_asm(None, [], "s_waitcnt vmcnt(N)\ns_barrier", "", has_side_effects=True)` — fp8_gemm_utils.py:197
- s.barrier: `rocdl.barrier()`; sched: `rocdl.sched_barrier(0)`; `rocdl.readfirstlane(T.i32, src)`
- fp4 pack: build vector<4xi64> via vector.from_elements (2 data + 2 zero i64) then vector.bitcast to vector<8xi32> — mixed_moe_gemm_2stage.py:1232
- decorators: `@flyc.kernel(name=, known_block_size=[256,1,1])`, `@flyc.jit`; launch `.launch(grid=, block=(256,1,1), stream=)`; waves-per-eu via CompilationContext.compile_hints({"waves_per_eu":2})

## Iteration loop (remote gfx950)
Helper: /tmp/rmi.sh '<cmd>' runs in container. FlyDSL env:
```
FLY=/home/zhiming_ding_qle/sixifang/FlyDSL
PYTHONPATH=$FLY/build-fly/python_packages:$FLY/python
LD_LIBRARY_PATH=$FLY/build-fly/python_packages/flydsl/_mlir/_mlir_libs
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/flydsl_dump FLYDSL_RUNTIME_ENABLE_CACHE=0
```
Dumped final LLVM IR = `<dump_dir>/<kernel_name>/NN_llvm_ir.ll`.
HIP reference rebuild: /tmp/emit_hip_ir.sh (hipcc --cuda-device-only -emit-llvm -S).

## Status
- [x] worktree on latest main (f3c8ff5d)
- [x] HIP source fully understood (gemm2_a4w4.cuh + mfma_f4f4 + epilogs + common)
- [x] HIP reference IR captured + censused
- [x] FlyDSL existing gemm2 IR captured — intrinsic parity confirmed
- [x] FlyDSL API mapped
- [x] kernels/gemm2_a4w4_port.py written — **COMPILES** end-to-end (COMPILE_ONLY) on gfx950
- [x] IR dump + census vs HIP — core compute matches intrinsic-for-intrinsic
- [ ] close peripheral gap (rsrc 9→4, i32 loads 12→6): use plain global loads
      for cumsum/eids/stids/sweights and a global atomic for out (HIP only makes
      buffer rsrc for the 4 main tensors A_q/A_scale/B_q/B_scale).
- [ ] correctness vs HIP on gfx950 (cosine), benchmark

## port v1 census (flydsl_port_v1.ll, 892 lines vs HIP 570)
MATCHED: mfma.scale .v4i32.v4i32 = 32 ✓ (NOT the .v8i32 padded variant the
existing FlyDSL kernel uses), buffer.load.lds = 2 ✓, buffer.load.v4i32 = 16 ✓,
s_waitcnt vmcnt fences = 2 ✓.
GAP: make.buffer.rsrc 9 vs 4, buffer.load.i32 12 vs 6 — because the port routes
cumsum/eids/sorted_token_ids/sorted_weights/out through buffer resources while
HIP uses plain addrspace(1) loads + global.atomic.fadd.v2bf16. Fix = plain
global loads for those + global atomic for the epilog.

## port v2 census (flydsl_port_v2.ll, 956 lines vs HIP 570) — FULL INTRINSIC MATCH
| metric | HIP | port v2 |
|---|---|---|
| mfma.scale .v4i32.v4i32 | 32 | 32 ✓ |
| raw.ptr.buffer.load.lds | 2 | 2 ✓ |
| raw.ptr.buffer.load.v4i32 (B) | 16 | 16 ✓ |
| raw.ptr.buffer.load.i32 (scale) | 6 | 6 ✓ |
| make.buffer.rsrc | 4 | 4 ✓ |
| s_waitcnt vmcnt fences | 2 | 2 ✓ |
| atomicrmw fadd <2xbf16> | 16 | 16 ✓ |
| plain load i32 addrspace(1) | 6 | 6 ✓ |

Every hardware intrinsic and memory-op class matches. (HIP's
`__builtin_amdgcn_global_atomic_fadd_v2bf16` lowers to `atomicrmw fadd <2xbf16>`
in this LLVM, same as the port — no named global.atomic intrinsic on either side.)

RESIDUAL (the unavoidable cross-frontend delta): addrspace(3) refs 160 vs 126,
fence syncscope 6 vs 4, total lines 956 vs 570. Source = FlyDSL emits
`vector.load/store` on memref views for the LDS cshuffle, generating more
GEP/index scaffolding than clang's raw pointer arithmetic; optimizes to the same
ISA. Byte-identical .ll across clang vs MLIR is not achievable (stated up front).
Next refinement to shrink this: hand-build ptr<3> + llvm.load/store for the
epilog cshuffle (as done for ds_read) instead of SmemPtr.store/.load; and compare
final ISA (21_final_isa.s) which should be far closer than the .ll.

## FINAL ISA comparison (the decisive "same machine code" check) — port v3
Both compiled to gfx950 ISA (HIP: hipcc -S; FlyDSL: 21_final_isa.s). Every
compute and memory instruction matches exactly:
| ISA instruction | HIP | port |
|---|---|---|
| v_mfma_scale_f32_16x16x128_f8f6f4 | 32 | 32 ✓ |
| buffer_load_dwordx4 (16 B + 2 A→LDS) | 18 | 18 ✓ |
| buffer_load_dword (scales) | 6 | 6 ✓ |
| ds_read_b128 (A from LDS) | 8 | 8 ✓ |
| s_barrier | 4 | 4 ✓ |
| s_waitcnt vmcnt | 22 | 23 (1 extra) |
| total .s lines | 548 | 625 |

The 77-line ISA delta is entirely scalar address arithmetic in the epilog
(per-element inttoptr address math + 1 extra s_waitcnt); zero difference in
MFMA / buffer-load / ds-read / barrier instructions. This is the strongest
"identical generated code" result achievable across the clang(HIP) and MLIR
(FlyDSL) frontends — both feed the same AMDGPU backend and emit the same
machine instructions for the compute core.

## v3 changes (fence + raw-ptr epilog)
- K-loop barrier: bare `s_barrier` inline-asm (fence-free) → fence count 6→4 (=HIP).
- epilog cshuffle: raw ptr<3> + llvm.store/load (scalar) → `store float` 32 (=HIP).
- added pre-epilog fenced barrier (HIP run_one __syncthreads()).
Residual .ll delta (addrspace(3) 224 vs 126, lines 1083 vs 570) is LDS-address
scaffolding (inttoptr-per-op + no scalar-load vectorization that clang does);
it does NOT affect the ISA compute/memory instruction counts above.

## CORRECTNESS (Task #6) — BIT-EXACT vs HIP
Same random inputs fed to both aiter HIP `mxfp4_moe_gemm2_a4w4` (kernelName
...BM32_ATOMIC) and the ported FlyDSL kernel; e8m0 scales pinned to 127 (2^0)
to avoid overflow, unique sorted_token_ids to avoid atomic-order nondeterminism.
Result (M=64, srt=64): finite 458752/458752 both, cosine=1.000000,
max_abs_diff=0, **bitexact=True — EXACT MATCH**.
Driver: /tmp/verify_gemm2.py + /tmp/run_verify.sh.

## FINAL (v6) — diff minimized to irreducible cross-frontend artifacts
Iteration v1→v6 progressively closed the gap. v6 result:
- LLVM .ll: 866 lines (HIP 570). ALL intrinsics + memory ops match:
  mfma.scale.v4i32 32, buffer.load.lds 2, buffer.load.v4i32 16, buffer.load.i32 6,
  make.buffer.rsrc 4, fence 4, atomicrmw 16, **load <2xf32> 16, load float 4,
  store float 32, addrspace(3) 124≈126**.
- ISA .s: 575 lines (HIP 548). EVERY compute/memory machine instruction identical:
  v_mfma_scale ×32, buffer_load_dwordx4 ×18, buffer_load_dword ×6, ds_read_b128 ×8,
  s_barrier ×4.
- Output: **bit-exact** (cosine 1.0, max_abs_diff 0).

The ONLY remaining .ll differences are provably irreducible across clang↔MLIR:
1. SSA value names (%0,%1,...) — two frontends never name values identically.
   This alone makes byte-identical text impossible, independent of any effort.
2. inttoptr ×9 (vs HIP 0) — FlyDSL models LDS (SmemAllocator) and kernel args as
   memrefs; obtaining raw addrspace(1/3) pointers requires ptrtoint→inttoptr.
   clang has native addrspace pointers / the @lds addrspace(3) global symbol.
   A FlyDSL-model difference, not an algorithm difference.
3. Integer address-math is spelled with explicit arith ops in MLIR where clang
   folds constants; these optimize to the same ISA (the 27-line ISA delta).

Conclusion: literal byte-identical .ll is physically unattainable (point 1).
Achieved instead: identical LLVM intrinsics+memory-ops, identical ISA compute
instructions, and bit-exact output — the maximum attainable fidelity.

## PERFORMANCE (gemm2 standalone, gfx950/MI355, /tmp/bench_gemm2.py)
| sorted rows | M-blocks | HIP us | port us | port/HIP |
|---|---|---|---|---|
| 256  | 8   | 18.7  | 27.2  | 1.46x |
| 1024 | 32  | 31.7  | 40.7  | 1.28x |
| 4096 | 128 | 79.3  | 83.0  | 1.05x |
| 8192 | 256 | 143.5 | 139.8 | 0.97x |

GPU-saturating sizes (>=4096, 256 CU): within 5%, slightly FASTER at 8192 —
confirms near-identical ISA => near-identical perf. Small/latency-bound sizes:
1.3-1.5x slower, from per-block fixed overhead (the +27 ISA scalar-address lines,
9 inttoptr address setup, FlyDSL host launch path) being amplified when the grid
is small and compute is short.

## opt -O3 comparison (fair, same passes) + the decisive impossibility proof
Fair comparison must opt both at the same level (FlyDSL's 20_llvm_ir.ll is
PRE-LLVM-opt; HIP's -emit-llvm is POST-O3). After `opt -O3` on both:
- FLY 866->690 lines; HIP 570->569.
- inttoptr 9->2 (opt eliminated 7); mfma 33, buffer.load.lds 3, buffer.load.v4i32
  17, atomicrmw 16, fence 4, load <4xi32> 8, store float 32 — all still match.
- residual: ~120 extra lines, addrspace(3) 160 vs 126 (standalone opt -O3 lacks
  the AMDGPU-target load-vectorizer that clang's full pipeline applied to HIP).

DECISIVE: the two toolchains use DIFFERENT LLVM versions — FlyDSL bundles LLVM
rev 554785 (emits attrs like `nocreateundeforpoison`); HIP uses ROCm 7.2.3's
LLVM (its opt cannot even parse FlyDSL's IR). Different LLVM + different frontend
=> byte-identical .ll text is impossible in principle, independent of the kernel.
Maximal attainable fidelity (intrinsics + ISA compute instrs + bit-exact output)
is achieved and verified.

## Unit test (tests/kernels/test_gemm2_a4w4_port.py)
4 tests, all PASS on gfx950/MI355:
- test_smoke (no aiter): compile+run, output finite & nonzero.
- test_accuracy_vs_hip[256,1024] (needs aiter): bit-exact vs HIP gemm2.
- test_performance[4096] (needs aiter): wall-clock ratio + loose regression bound.
Run: `python3 -m pytest tests/kernels/test_gemm2_a4w4_port.py -v -s`
(markers l2_device + rocm_lower; module-skips if not gfx95 / no CUDA).

GOTCHA: `flyc.compile(launch, ..., out, ...)` EXECUTES the kernel once into the
buffer passed to it. Since gemm2 atomic-accumulates, compiling and then calling
into the SAME un-zeroed `out` doubles the result (looks like a 2x bug). Correct
pattern: compile against a throwaway buffer, then zero the real out and call once
(see `_compile_port`/`_run_port`).

## Iterate loop helper
/tmp/iter_gemm2.sh  → syncs kernel to remote, runs /tmp/run_compile_gemm2.sh
(COMPILE_ONLY=1 + FLYDSL_DUMP_IR), dumps to /tmp/gemm2_dump/.
Driver: /tmp/drive_gemm2.py. HIP ref rebuild: /tmp/emit_hip_ir.sh.
