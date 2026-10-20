# Rule bodies

Not loaded with the skill. `triage.py expand` writes only the blocks a diff derives into
`$WORK/rules_expanded.txt`; `MAPPING.md` lists which family emits which rule. Edit rule text
here. A rule id is internal: it goes in `verdicts.txt`, never on the card.

Every rule names a FlyDSL PR where the defect shipped, was caught in review, or was fixed. A
rule with no such PR does not belong here.

### A — Cache and recompilation correctness

_The signature FlyDSL failure, silent by construction: the wrong artifact runs and nothing
reports an error._ Keys have two levels: `manager_key` (function source, transitive helper
sources, closure values, `_flydsl_key()`) and a per-call tuple (`_env_`, `_target_`, `_hints_`,
argument signatures). **Every A finding must state its direction**, because the fixes are
opposite: over-keying recompiles or misses a shipped cache; under-keying serves a stale binary.

**A1 — Over-keying: an input already captured elsewhere, or with no effect, enters the key** ⚠️/🔴
The canonical case is an arch env var: AOT precompilation runs on a GPU-less build host with
`ARCH=gfx950` injected, while runtime detects the arch from the device. The resolved target is
identical, but the raw `_env_` segment differs, so every shipped cache misses.
Real example (PR #624): `ARCH`, `FLYDSL_GPU_ARCH` and `HSA_OVERRIDE_GFX_VERSION` were removed
from the invalidating env vars after aiter hit `RuntimeError: AOT cache miss`.
Real example (PR #1153): four LLVM hints never reached codegen, yet each still changed the JIT
cache key, so setting one caused a real recompile that behaved exactly like setting nothing.
FP self-check: is the added value's effect not already represented by `_target_` or by an
argument signature? If it is, adding it is the bug.
→ `⚠️ A1: [value] enters the cache key but its effect is already captured by [X] (or is nil) — build-host and runtime keys diverge, so caches miss`

**A2 — A runtime value leaking into the key** ⚠️
`inspect.signature()` under `from __future__ import annotations` stringifies `fx.Int32`, so the
argument is no longer recognized as a DSL type and its **value** joins the key: one compile per
distinct value.
Real example (PR #556): fixed with `resolve_signature(..., eval_str=True)`; the regression test
pins both directions, `("n", int) in key` and `("n", (int, 1)) not in key`.
→ `⚠️ A2: [arg] enters the key by value, not by type — every distinct value recompiles`

**A3 — A codegen input missing from the key** 🔴
Two places to check. (1) Hand-maintained `cache_tag` tuples, for example the `cache_tag` methods
in `kernels/attention/flash_attn_utils.py`: nothing enforces their completeness, so a new trait,
constexpr or builder kwarg that reaches codegen without a matching tag entry makes two
configurations share one artifact. `$WORK/scan.txt` pairs new traits with tag edits under
`cache_key`. (2) The automatic dependency collector in `python/flydsl/compiler/jit_function.py`.
Real example (PR #1130): captured dtype classes (`fx.BFloat16` vs `fx.Float16`) specialize the
binary but were dropped by the collector, so the second dtype was served the first one's
binary; the `@flyc.jit` nested-kernel walk had the same hole.
Ask for a test that compiles with two values of the new input and asserts two cache entries.
→ `🔴 A3: [input] changes codegen but is absent from [cache_tag / the collector] — [config X] and [config Y] share one artifact`

**A4 — A trait in the key that nothing reads** ⚠️
The dual of A3: count read sites for every name in `cache_tag`; zero reads means the user-facing
flag does nothing.
Real example (PR #1020): `DUALWAVE_SWP_LAZY_RESCALE` was in the fp8 tag while both call sites
used `lazy_correct_o` unconditionally; fixed with a real `const_expr` branch.
→ `⚠️ A4: [trait] is in cache_tag but has no read site — the flag is inert; wire it or remove it`

**A5 — Tuned results outliving the kernel that produced them** ⚠️
When a kernel's numerics or codegen change, autotune winners recorded under the old behavior stay
selected unless the tuning schema version moves.
Real example (PR #1022): the LDS-barrier fix also bumped `TUNING_SCHEMA` for exactly this reason.
→ `⚠️ A5: kernel behavior changed but the tuning schema did not — previously tuned configs stay selected`

**A6 — Device identity confused with architecture identity** ⚠️
HSACO is arch-specific, not device-specific. Folding `device_id` into a disk key stores one
artifact per device; resolving the arch from the first GPU `rocminfo` reports compiles for the
wrong device on a mixed-GPU box.
Real examples: PR #597 (duplicate per-device artifacts); PR #403 (`_get_lds_size_per_cu()` cached
process-wide with no device parameter); PR #1057 (fixed by resolving the arch from `A.device`).
→ `⚠️ A6: [quantity] resolved from the first enumerated GPU / keyed by device — resolve from the tensor's device and key by arch`

---

### B — Gate and flag threading

_The code looks complete; certain inputs silently take a path that drops a feature._

**B1 — A new branch inserted above an existing validation** 🔴
A fast path, early return or recursive split placed before the block that raises on unsupported
combinations. The `raise` still exists but is unreachable on the new path, and the rebuilt `kw`
for the sub-call often omits arguments.
Real example (PR #1020): a split ran before fp8 modifier validation while `kw` omitted `bias`,
`alibi_slopes` and `sink`; the forced path returned output bit-identical to the call with no
modifier, where the unsplit path raised `NotImplementedError`.
Real example (PR #844): `return_lse=True` with the caller omitting `lse` fell back to `Out`, so
fp32 LSE overwrote the output buffer.
→ `🔴 B1: [param] is dropped on [new path], which now runs before its validation — [what silently computes wrong]; assert or forward it`

**B2 — A flag or a result not threaded through every route** 🔴/⚠️
A new kwarg on a public entry point where the file has several builder or launcher routes
(`_build_splitk`, `_build_varlen`, paged vs dense, fp8 vs bf16). The unthreaded routes fall back to
the library default, and tests stay green because they pass the flag explicitly. The same shape
appears on the way out: a wrapper that calls the wrapped function on every branch and returns its
value on none.
Real example (PR #1056 review): `causal_lpt=False` was not honored by `_build_splitk` or the long
`_build_varlen` route.
Real example (PR #899): `_run_with_hints` called `self.fn(*args, **kwargs)` on both branches
without returning it, so every autotuned function that returns its result got `None`.
Check: enumerate every route that reaches the gated code and confirm each accepts *and forwards*.
→ `🔴/⚠️ B2: [flag/result] is not forwarded by [routes] — the caller's value is silently lost on those paths`

**B3 — A default-off trait placed in front of previously unconditional behavior** 🔴
The diff shape is `if const_expr(not A and not B)` becoming `if const_expr(NEW_TRAIT and not A)`,
where `NEW_TRAIT` defaults to `False` and only the builder in the PR's scope passes it.
Real example (PR #1009): `XCD_SWIZZLE` defaulted to `False` and only the fp8 builder passed it, so
bf16 dense attention lost its XCD remap for two weeks; restoring it recovered +17.4% at S=180,180
and +19.4% at S=239,580 with bit-identical output. No test covered the mapping.
→ `🔴 B3: [trait] defaults False and only [builder] passes it — [other builders] silently lose [behavior]`

**B4 — A surviving always-false gate** 🔴
`False and ...`, `if False:` or `and False` left in a condition dead-codes the path.
Real example (PR #650): `if const_expr(False and N >= tile_cols ...)` hid two compile failures and
kept the vectorized softmax fast path dead.
→ `🔴 B4: [line] can never be true — the path is dead; remove the literal or fix the underlying failure`

**B5 — Fail-open where the contract promises fail-closed** ⚠️
A `catch` or `if (failed(...))` that returns to a default path with no diagnostic; `continue` on
an unparseable input; a warning where an error is contracted.
Real examples: PR #470 review (a pass-plugin failure silently skipped an optimization the user
had opted into via `compile_hints`); PR #1015 (a corrupted byte changed a reported count while the
report still read "no problems").
→ `⚠️ B5: [path] falls back silently — the caller opted in and gets no signal; emit a diagnostic or fail`

**B6 — A guard, zero-init or contiguity requirement removed without a reason** ⚠️
A deleted `raise`, `assert`, mask, `torch.zeros`, `max_size=False` or `.contiguous()`, where the
PR does not say what now enforces the invariant. The other recurring shapes here: a removed
`s_waitcnt` or barrier, a NaN or divide-by-zero guard dropped in a refactor, a narrowed dtype or
arch validation, a FileCheck `CHECK` line deleted instead of updated, and a test deleted because it
started failing. For every deleted line, name the invariant it enforced and find where head
re-establishes it; `$WORK/scan.txt` lists them under `guard_removed`. Counter-rule: maintainers reject redundant
validation on both sides of the Python/C++ boundary (PRs #644, #506, #297), so removing a
duplicate is correct when the other side still enforces it.
FP self-check (do this before firing): find where the invariant is still enforced on head. If you
can name the file and line, do not fire.
Real example (aiter PR #4043, same hazard class): the old comment said "trailing pad must read back
as zero for the asm reader"; the new one said "never read", with no spec cited.
→ `⚠️ B6: [guard] was removed and nothing on head enforces [invariant] — [input] now reaches [kernel] unchecked`

---

### C — Architecture capability

_Correct for gfx942 and wave64; silently wrong on gfx1250 or RDNA._

**C1 — A capability derived from a family predicate** 🔴
`is_rdna_arch()` answers "is this RDNA", which is not "what is the wave size". gfx1250 is wave32
and is not matched by RDNA prefixes.
Real example (PR #1024): `32 if is_rdna_arch(arch) else 64` emitted wave64 codegen for gfx1250.
Fixed by one source of truth, `get_warp_size` in `python/flydsl/runtime/device.py`. A kernel that
hard-codes `WAVE_SIZE = 32` while the shared helper reports 64 means the predicate is wrong.
→ `🔴 C1: [capability] derived from [family predicate] — gfx1250 lands on the wrong side; route through get_warp_size / the capability accessor`

**C2 — An arch predicate too wide, too narrow, or missing** 🔴/⚠️
`startswith("gfx12")` also catches gfx1250, which is not a GFX120X WMMA part. A missing gate is
worse: an op the target lacks can pass MLIR verification and abort in LLVM instruction selection,
which kills the Python process with no traceback.
Real examples: PR #544 (restrict a WMMA atom to gfx11 so older RDNA targets do not select it);
PR #943 (narrowed `gfx12` to `gfx120` so it no longer overlaps gfx1250); PR #1122
(`buffer_load ... lds` has no gfx11 hardware path; the fix raises `ValueError` at the kernel line,
reading `ARCH` before `get_rocm_arch()` so cross-compilation gates the right target).
Severity: 🔴 when the target aborts or computes wrong values; ⚠️ for a predicate that selects an
unsupported but failing-loudly path.
→ `🔴/⚠️ C2: [predicate] [also matches / does not gate] [arch], which lacks [feature] — [abort / wrong atom]; narrow or add the gate`

**C3 — A tolerance or heuristic derived on one arch applied to all** ⚠️
Real example (PR #1022): a gfx950-derived tolerance applied everywhere, while `M=32768, N=8192,
bf16` failed four independent Navi runs at 0.523–0.533 scaled relative error against 0.02.
→ `⚠️ C3: [tolerance/heuristic] was derived on [arch] but applies to all — [arch B] behaves differently`

**C4 — A new arch added in fewer than four places** ⚠️
Adding an architecture needs all of: `SMEM_CAPACITY_MAP` and `get_warp_size` in
`python/flydsl/runtime/device.py`, an atom directory under
`lib/Dialect/FlyROCDL/`, and the test predicates in `tests/arch_compat.py` and per-test `skipif`.
A missing LDS entry makes `check_smem_capacity` skip the check for that arch.
FP self-check: an arch string inside an arch-specific file (`*_gfx1250.py`) is normal; fire only
when a shared dispatch or capability path learned a new value.
→ `⚠️ C4: [arch] added to [places] but not [missing] — [consequence]`

**C5 — A path that no PR CI runner executes** ⚠️
The PR GPU matrix is `linux-flydsl-mi325-1` (gfx942), `linux-flydsl-mi35x-1` (gfx950) and
`linux-flydsl-navi-2` (`.github/workflows/flydsl.yaml`); multi-GPU tests run only with the
`multi-gpu` label. A change reachable only on another arch, or behind a module-level skip none of
those runners satisfies, reports green by skipping.
Real example (PR #1136): 33 of 44 tests in `tests/kernels/test_rdna3_int8_gemm.py`, a module that
skips unless the arch is `gfx11*`, were failing at trace time on main.
→ `⚠️ C5: [path] runs only on [arch], which no PR runner provides — CI green is a skip; **Reviewer should ask** for the author's on-hardware log`

---

### D — Index width and the ABI boundary

**D1 — Index × stride with no 64-bit widening** 🔴
A position-shaped value multiplied by an extent-shaped value producing an address, with no
`fx.Int64(...)` on that line or the line above. The repo idiom widens the position operand first.
Real example (PR #1064): `(((phys_row[sub] * n_kv + kv_h) * STEPS_PER_PAGE + step) * head_dim ...)`
became `phys = fx.Int64(phys_row[sub])` and then the same chain — page ids fit in i32, their
element offsets do not once one KV cache exceeds 2 GiB.
`$WORK/scan.txt` lists candidates under `index_width`. Clear each one, or name the production size
at which the product passes 2^31. If the list is empty, say so.
→ `🔴 D1: [expr] multiplies in int32 — widen [operand] to fx.Int64; overflows at [concrete size]`

**D2 — A tensor flattened before a launch** 🔴
The launch ABI packs shapes as int32 regardless of `use_32bit_stride`; only strides honor it. A
flattened tensor makes the packed extent the whole element count.
Real example (PR #1020): the fp8 attention path flattened Q/K/V/O to 1-D and overflowed at
`B*S*H*D >= 2^31`, i.e. `S >= 131072` at `D=128, H=64`; fixed with one launch per batch entry.
The bf16 path passes the natural 4-D shape and is exempt, which is also the FP self-check.
→ `🔴 D2: [tensor] is flattened before launch — the packed int32 extent overflows at [shape]; pass the natural rank`

**D3 — Element extent and descriptor byte span are two separate bounds** ⚠️/🔴
The signed int32 element extent (2^31) and the unsigned descriptor byte span (2^32) are different
limits; a guard on one is not a guard on the other, and a check on `.shape` alone checks neither.
Real examples: PR #960 (`row * stride + column` in signed i32 while validation checked only the
shape); PR #1020 (a `q.numel()` guard that ran only when `B > 1`).
Real example (PR #1136): descriptor offsets and the element-to-byte scaling are 32-bit, so an
operand at or above 4 GiB wraps; the wrapped offset is small, compares in-range against
`num_records`, and the access lands near the buffer start instead of being dropped. `M=65027,
N=16512`, i32 output put the invalid tail row at byte offset exactly 2^32.
→ `⚠️/🔴 D3: the guard covers [one bound] but not [the other] — [input] passes validation and still wraps`

**D4 — A defaulted width on a width-carrying C++ factory** ⚠️
A factory that defaults the width records the wrong one: the former `IntAttr::getDynamic(MLIRContext*,
int32_t width = 32, ...)` silently recorded width 32 for an i64. Core's `Leaf<Trait>::getDInt` takes the
width explicitly, and `Leaf<EmitOp>::getDyn` reads it from the SSA value's type.
Real example (PR #729): fixed by reading the actual `IntegerType` width and rejecting anything but
32 or 64.
→ `⚠️ D4: getDynamic called without an explicit width — an i64 value is recorded as i32`

**D5 — A buffer descriptor with no bound on a runtime-ragged extent** 🔴/⚠️
`fx.rocdl.make_buffer_tensor(t)` and the legacy `create_buffer_resource(t)` default to
`max_size=True`: the descriptor size becomes `0xFFFFFFFF` and RDNA bounds checking stays off (see
the `make_buffer_tensor` docstring in `python/flydsl/expr/rocdl/universal.py`). Hardware OOB
suppression then never fires, so when the bound dimension of `t` is a runtime extent — M, a token
count, `num_valid` — tail-tile loads read past the live data and unmasked stores write past it.
`$WORK/bounds.txt` lists every such call on an added line. It is a candidate list.
FP self-check: full-size weights and caches, compile-time tile-aligned extents, and stores guarded
by an explicit mask (E3 judges the mask) are fine. Name the runtime value at which the access
leaves the allocation.
Real example (PR #1136): `rdna3_int8_gemm` relies on `max_size=not partial_m` to make the hardware
drop tail-tile A loads and C stores; with `max_size=True` those accesses are performed.
→ `🔴 D5: [call] binds [tensor] with no byte bound while [dim] is runtime-ragged — accesses past the allocation at [value]`

---

### E — Data-movement shape math

**E1 — Truncating division in a tile, chunk or loop count** 🔴
Any `//` (or a `x if x < K else K` clamp) deriving a count or width where the numerator is not
provably a multiple of the denominator. Chained truncating divisions are the worst case.
Real example (PR #1064): `head_dim=192` gives `QCHUNK = 12`, so `N_QLOADS = 12 // 8 = 1` left one
third of every query row unstaged in LDS.
Real example (PR #1007): `num_a_loads = bytes_per_thread_a // a_load_bytes` dropped the remainder
for `tile_m=112, tile_k=64, bf16`, while `_TILE_PRELOAD_TABLE` advertised such tiles as tuned.
Real example (PR #1199): `K_TILES = K // tile_k` dropped the last `K % tile_k` columns; with
`K=640, tile_k=256` the output was 60% off the full-K reference. The fix rejects such K when the
kernel is built, and its test fails without the fix.
→ `🔴 E1: [expr] truncates for [shape] — [what is never fetched]; assert the exact divisibility or ceil-div with a masked tail`

**E2 — Copy-atom width ≠ register vector width × dtype width** 🔴
`BufferCopy128b` with a `make_rmem_tensor(K, dtype)` where `K * dtype.width != 128` is a hard LLVM
abort (`CastInst::Create: Invalid cast!`), not a graceful error.
Real examples: PR #650 (`VEC_WIDTH = 8` was right for 16-bit and invalid for f32; fixed to
`vec_width = 128 // elem_bits`); PR #564 (the same defect via a `--vec-width 8` flag).
Check the identity for every dtype the factory accepts.
→ `🔴 E2: [atom] pairs with [K]x[dtype] = [bits] bits — invalid cast at lowering for [dtype]`

**E3 — A store mask guarding fewer dimensions than the address fuses** 🔴
When the address fuses dimensions (`off = col * dhw + row`), "out of range in one dimension" does
not imply "out of bounds in memory", so hardware OOB suppression does not fire and padded rows
alias real elements of the next column.
Real example (PR #969): the fp8 conv3d epilogue masked only the column; 12 of 200 runs failed at
exactly `rel_err 3.075e-01`. The bf16 kernel already guarded rows via `_row_chk`, and the fp8
split-K path already checked `row < npq`.
→ `🔴 E3: mask covers [dims] but the address fuses [dims] — padded rows alias live elements and race real stores`

**E4 — A launch path over 256 threads without `known_block_size`** ⚠️
Without it the AMDGPU backend keeps its default max flat workgroup size of 256 and the launch aborts.
Real example (PR #639): rmsnorm crashed on real DeepSeek-R1 and Qwen3 shapes that the test list did
not contain.
→ `⚠️ E4: [path] can launch [n] threads without known_block_size — launch aborts at [shape]`

**E5 — A raw block-id read after a remap** 🔴
Once a kernel derives remapped block indices, a later raw `gpu.block_id(...)` bypasses the remap.
Real example (PR #782): the `preshuffle_gemm` epilogue re-read `gpu.block_id("x")` and scrambled
the output; the same PR fixed `/` used for `//` in `xcd_remap_bx_by`.
→ `🔴 E5: [line] re-reads the raw block id after the remap at [line] — output goes to the unremapped tile`

---

### F — Numerics and fastmath

**F1 — An `fx.*` math wrapper without `@dsl_math_wrap_result`** 🔴
The decorator fills `fastmath` from the ambient context. A wrapper that takes `**kwargs`, wraps
its own result and calls the raw builder emits `fastmath<none>` whatever the kernel requested.
Real example (PR #1035): `maxnumf` lacked it while `maximumf`/`minimumf` had it; a mechanical
migration had moved 29 call sites onto it, costing 9–11% on the gfx950 bf16 attention path at long
sequence lengths, and nothing failed.
→ `🔴 F1: [wrapper] lacks @dsl_math_wrap_result / a named fastmath param — emits fastmath<none> at every call site`

**F2 — A Python literal where a fastmath enum is required** ⚠️
`fastmath=True` becomes an invalid `#arith.fastmath<True>` attribute (PR #650). Prefer the ambient
compile hint over per-op flags.
→ `⚠️ F2: fastmath=[literal] is not an enum value — use the FastMathFlags enum or the ambient compile hint`

**F3 — A wave-uniform branch whose body assumes the predicate held per lane** 🔴
`ballot` plus `read_exec` turns a per-lane predicate into one decision for the wave, so lanes whose
own predicate was false run the body.
Real example (PR #1033): such a lane had `m_tile_max < m_row`, so `exp2(m_row - m_tile_max)` scaled
the accumulator up to `inf`; with K scaled 32×, 9.2% of the output was NaN, at 64× all of it. The
eager path took `m_new = maxnumf(m_row, m_tile_max)` first and never had the bug.
→ `🔴 F3: the [ballot] body consumes [per-lane value] unguarded — non-triggering lanes produce [inf/NaN] at [input]`

**F4 — A low-precision intermediate underflowing while its normalizer sums wider** 🔴
A softmax-like reduction where `P` is cast to fp8 before the second matmul while the normalizer
accumulates before the cast, with no scale-up into the format's range.
Real example (PR #1020): with `V` all ones the exact answer is 1.0; bf16 gave 0.99996 and fp8 gave
0.637. Fixed with a `log2(448)` headroom term. Random-init q/k did not show it.
→ `🔴 F4: [value] is cast to [fp8] while [normalizer] sums pre-cast — rows underflow at [distribution]`

**F5 — Rounding-mode divergence between a fast path and its generic sibling** ⚠️
A hand-written conversion (`+ 0x8000`, `>> 16`, `cvt_pk_bf16_f32`) on a vector path where the
scalar or tail path converts differently.
Real example (PR #848 review): `+ 0x8000` rounds exact ties away from zero, unlike the generic path
and PyTorch; the bias stays inside bf16 `atol=2e-2`, so tests do not catch it.
→ `⚠️ F5: [fast path] converts with [mode] while [generic path] uses [mode] — systematic bias inside tolerance`

**F6 — Compile-time folding that does not wrap like the hardware** ⚠️
Static folding on Python's arbitrary-precision ints disagrees with the `arith` op it replaces.
Real example (PR #973): `Uint32(0xFFFFFFFF) + Uint32(2)` folded to `0x100000001` instead of `1`.
→ `⚠️ F6: [fold] does not match what the [width]-bit op computes — reduce to the type's width`

---

### G — Layout algebra and compiler internals

**G1 — A `{value, attr}` adaptor pair describing different shapes** 🔴
Binding SSA scalars to a profile of another shape (`core::bind(profile, bindings, ctx)`, formerly the
`IntTupleValueAdaptor` pair) yields wrong coordinates, which can make edge predicates true and permit
out-of-bounds accesses.
Real example (PR #1052): a `(48,48)` tile returned grid coordinate `(0,1)` as logical `(0,1)`
instead of `(0,48)`.
→ `🔴 G1: [adaptor] pairs a [scalar] value with a [nested] attr — coordinates resolve to [wrong value]`

**G2 — Type inference that updates part of a result type** 🔴
Real example (PR #707): `SliceOp` recomputed the layout and left the base unmoved.
→ `🔴 G2: [op]'s inferred type updates [component] but not [component] — addresses stay at the source base`

**G3 — Reading a value off an attribute without its static predicate** 🔴
A static value read off a dynamic leaf is meaningless: the former `IntAttr` carried `value == 0`, so
`getValue()` without `isStatic()` returned 0. Check `isStatic()` before `Leaf::staticValue()`.
Real example (PR #926): `add_offset(reg_ptr, dynamic)` dropped the dynamic index and always
accessed slot 0.
→ `🔴 G3: getValue() without isStatic() — a dynamic attr reads back as 0 and [access] targets slot 0`

**G4 — Reconstructed DSL values losing wrapper-only metadata** ⚠️
Attributes that live only on the Python wrapper (multi-dim shape, signedness) are lost when a value
crosses an `scf` region boundary.
Real example (PR #759): a `Vector` of shape `(4,1)` collapsed to `(4,)` across a loop carry, and the
next `+=` broadcast to `(4,4)`.
→ `⚠️ G4: [attribute] is wrapper-only and is dropped when the value crosses [region] — thread the exemplar`

**G5 — Promotion implemented twice, or signedness re-derived from MLIR** 🔴
MLIR integers are signless, so deriving a DSL dtype from an element type turns `Uint32` into
`Int32`, and later shifts, divides and compares switch to signed semantics.
Real example (PR #816): `Vector` and `Numeric` promoted differently; unified into one lattice.
→ `🔴 G5: [logic] derives a DSL dtype from a signless MLIR type / duplicates the [other] lattice — unsigned data gets signed semantics`

**G6 — Metadata read before a transform that invalidates it** 🔴
Real example (PR #1043): alignment was read from the pointer type and attached to a load built on
the swizzled pointer, over-promising alignment; a non-power-of-two byte count then crashed
`llvm::Align`. Fixed with a FileCheck test in `tests/mlir/Conversion/ptr_alignment.mlir`.
→ `🔴 G6: [metadata] is read before [transform], which weakens it — the op over-promises [property]`

**G7 — A lowering or pass change with no FileCheck test pinning the emitted IR** ⚠️
A change under `lib/` needs a `tests/mlir/` test that asserts the emitted arithmetic or op, not only
the result type. Models: PR #1052 asserts `arith.constant 48`, `muli` and `addi`; PR #792 added the
missing CDNA3 BF16 MFMA case to `tests/mlir/Conversion/mma_atom.mlir`. The lit stage fails closed
only with `FLYDSL_REQUIRE_FILECHECK=1`, which CI sets; locally `scripts/run_tests.sh` reports a skip
(PR #1146).
→ `⚠️ G7: [pass/lowering] changes [emitted op] with no FileCheck test pinning it — a regression stays invisible`

---

### H — Host, stream and tensor hygiene

**H1 — An internal copy or fill racing an explicit stream** 🔴
`.contiguous()`, `torch.cat(...)`, `.to(...)` or a fill run on the ambient stream while the kernel
consumes the result on the caller's `stream=`.
Real examples: PR #1066 review (the side-stream test used contiguous inputs, so every copy was a
no-op and could not catch the race); PR #1020 (a `torch.cat` with no dependency).
Real example (PR #1220): a test filled `Out` on the default stream and launched on a side stream
via `from_c_void_p`; it failed about once in 100,000 launches under cross-stream stress. Fixed by
allocating and filling under the launch stream.
FP self-check: does the launcher accept a `stream=` at all? No explicit stream, no race.
→ `🔴 H1: [copy/fill] runs on the ambient stream while the kernel consumes it on [stream] — add a stream dependency`

**H2 — A reshape that silently materializes the output** 🔴
`reshape()` on an output tensor can produce a temporary; the kernel writes it and the caller's
tensor never changes. Real example (PR #403).
→ `🔴 H2: [out] is reshaped before the launch — the kernel may write a temporary and the caller sees stale data`

**H3 — `assert` as the only guard on a kernel precondition** ⚠️
Under `python -O` the assertion disappears while the kernel still assumes the layout.
Real example (PR #801 review): "it can silently read the wrong row".
→ `⚠️ H3: [precondition] is enforced only by assert — it vanishes under -O; raise, or make the tensor contiguous`

**H4 — Cross-device tensor and stream mixing** ⚠️
Real example (PR #1022): tensors on `cuda:0` were accepted with a stream from `cuda:1`.
→ `⚠️ H4: [tensor] device and [stream] device are not checked — validate they match`

---

### I — Cross-repo coupling with aiter

FlyDSL production code does not import aiter. The coupling is tests and benchmarks that call
aiter, and aiter building its FlyDSL kernels on the `flydsl` wheel.

**I1 — A positional call into aiter** 🔴
CI clones aiter at `main`, so an upstream argument change reddens every FlyDSL PR at once.
Real example (PR #710): aiter reordered `pa_reduce_v1`, so `0` landed in `final_output`.
Real example (PR #680): aiter inserted `num_kv_splits` before `final_output`, failing 8 MI325
tests until the call sites changed.
→ `🔴 I1: [call] passes [n] arguments positionally into aiter — an upstream reorder reddens every PR; pass by keyword`

**I2 — `importorskip` that does not cover symbol imports** ⚠️
It covers a missing module, not a drifted API; a top-level `from aiter... import X` raises during
collection and ends the session. Real example (PR #840).
→ `⚠️ I2: [import] is not guarded — an aiter API change aborts collection for the whole session`

**I3 — A stale intra-repo module path after a refactor** ⚠️
Real example (PR #870): `kernels/gemm/mxfp4_preshuffle.py` still imported `kernels.mma.*` after
the helpers moved to `kernels/common/mma/`. When a PR moves a module, grep the old path tree-wide.
→ `⚠️ I3: [module] moved but [importer] still uses the old path — collection fails for that file`

**I4 — A change aiter consumes, with no downstream signal** ⚠️
The ATOM, SGLang and vLLM integrations run on a schedule or by manual dispatch only
(`.github/workflows/flydsl-atom-integration.yaml` and its siblings). A change to the behavior or
signature of `python/flydsl/` or `lib/` reaches aiter with no pre-merge check.
→ `⚠️ I4: [surface] changes behavior aiter builds on, and downstream integration is nightly-only — **Reviewer should ask** for a manual run before merge`

---

### S — Stable API

**S1 — A stable API changed without the compatibility review** ⚠️/🔴
`docs/api_stability.md` defines which `fx` and `flyc` paths are stable. Removing, renaming or
re-signaturing a stable path, or changing its semantics, needs the deprecation window the policy
describes. Run the **api-stability** skill on the diff: it classifies stability mechanically.
FP self-check: paths containing `experimental` and `_`-prefixed names are unstable by policy.
→ `🔴/⚠️ S1: [stable path] is [removed/renamed/changed] without a deprecation window — callers break at [import/call]`

---

### T — Test and benchmark methodology

_A suite that cannot fail is worse than no suite, because it reports green._

**T1 — A test that cannot fail** 🔴
Shapes seen here: a decorator inserted mid-function that truncates the previous test; a timing stub
returning a constant; a parametrize matrix whose values never satisfy the fast-path predicate; a
reference built from the same objects as the kernel output; a result returned instead of asserted;
a loop, graph replay or repeated launch that writes one buffer and asserts only its final state,
which is one observation, not one per launch.
Real examples: PR #1056 review (a new test inserted mid-function ended the previous one, so four
XCD cases passed without launching a kernel); PR #899 (a stub returning `1.0` made the first config
"win"); PR #643 (the test returned timings instead of asserting).
→ `🔴 T1: [test] cannot fail because [reason] — the regression is unguarded; replace with [independent oracle]`

**T2 — A regression test that passes on the pre-fix commit** 🔴
The local standard, set by reviewers: run the submitted test on the defective parent and show it
fails there. Inputs that hide this repo's numerics bugs are the norm: #1020 needed trained q/k,
#1033 needed K scaled 32×, #1064 needed `head_dim=192`, #969 needed `dhw == npq`.
Real example (PR #1033 review): the submitted test passed unchanged on the defective head.
→ `🔴 T2: [test] would also pass before the fix — it pins nothing; **Author must** show it failing on the parent commit`

**T3 — A benchmark measuring the wrong thing** 🔴
Real examples: PR #654 (the parser kept the last match, so layernorm reported a scalar variant,
1.69 TB/s against a real 5.6); PR #848 review (speedup columns inverted); PR #1200 (one event pair
around a loop of `@flyc.jit` launches measured ~40 µs of host dispatch for a ~3.5 µs kernel; fixed
with `run_perftest`).
When a PR changes a benchmark parser, ask whether the stored baseline came from the same parser.
→ `🔴 T3: [benchmark] reports [wrong quantity] — the published number is [X], not [Y]`

**T4 — A reference the test's own dtype cannot represent** ⚠️
Real example (PR #643): activations built from raw fp8 codes (~448) instead of dequantized values
(~0.2) overflowed f16 in both the kernel and the reference, so a correct kernel read as 100% wrong.
When every independent implementation "fails", the harness is wrong.
→ `⚠️ T4: reference [expr] overflows [dtype] at the magnitudes this test generates — the comparison is invalid`

**T5 — Tolerance widened, or shape rows disabled** ⚠️
Compare head against base. A test-only widening with no numerical justification is the concern.
→ `⚠️ T5: [tolerance] widened from [x] to [y] with no stated justification — the test no longer guards [what]`

**T6 — A test that only runs in one harness, or in none** ⚠️
`scripts/run_tests.sh` runs pytest over `tests/kernels/`, `tests/language/`, `tests/unit/`,
`tests/system/`, `tests/extension/` and `tests/python/examples/` with
`-m "not multi_gpu and not benchmark"`; `multi_gpu` tests run only in the label-gated job;
`scripts/run_benchmark.sh` executes test files as scripts, so pytest-only variants never run there.
`$WORK/ci_coverage.txt` says where each added test runs. Kernel tests belong in `tests/kernels/`
(PR #980 review). Real example (PR #481).
→ `⚠️ T6: [test] runs only under [harness] / no CI job — it never executes in [the other]`

**T7 — A measurement taken with a warm JIT cache** 🔴
The disk cache invalidates on kernel source and closure changes, not on C++ passes or helper code
outside the traced closure. Any number in a PR that edits `lib/` or a non-closure helper, taken
without `FLYDSL_RUNTIME_ENABLE_CACHE=0`, is unverified. PRs #1009, #1033 and #1035 state the cache
setting explicitly; that is the local standard.
→ `🔴 T7: the PR edits [C++ pass / non-closure helper] but the numbers were taken with a warm cache — **Author must** re-run with FLYDSL_RUNTIME_ENABLE_CACHE=0`

**T8 — A skip that reports as a pass** ⚠️
Real examples: PR #1146 (the MLIR lit stage skipped when FileCheck was missing and still printed
"All tests passed"); PR #1187 (`dumpir.sh` succeeded with no command); a module-level
`pytest.skip(..., allow_module_level=True)` whose condition no CI runner meets (see C5).
→ `⚠️ T8: [stage/test] skips and reports success — a run with zero coverage reads as clean; fail or say it did not run`

---

### P — Performance evidence

**P1 — A perf claim with no numbers** ⚠️
`CONTRIBUTING.md` asks kernel PRs for hardware, baseline, optimized and improvement. Screenshots are
not numbers. Exception: PRs that add benchmarks without claiming an improvement.
→ `⚠️ P1: perf claimed with no [hardware / baseline / units]`

**P2 — Toy shapes only** ⚠️
This repo's regressions show at long context (S=180,180 and S=239,580 in #1009 and #1035) and on
production GEMM and MoE rows. `M=1` / `M=16` only is the generated-test signature.
→ `⚠️ P2: benchmark omits [production shape class]`

**P3 — Not reproducible** ⚠️
Missing command, ROCm version, GPU model, arch, or cache state (see T7).
→ `⚠️ P3: perf claim missing [reproduction detail]`

**P4 — A knob that never reaches codegen** ⚠️
An option or attribute accepted in Python and dropped before the emitted IR or ISA.
Real examples: PR #785 (`Config(waves_per_eu=…)` reached codegen only as `gpu-module-to-binary opts=`,
which the AMD path drops); PR #1153 (four such LLVM knobs, including `maxnreg`, which stayed inert
while autotune searched over it); PR #1177 (`daz=True` left the denorm mode bit at 3, IEEE, because
upstream no longer reads the `denormal-fp-math*` passthrough strings).
→ `⚠️ P4: [knob] may not reach codegen — **Reviewer should ask** for an IR/ISA diff across two values`

**P5 — A kernel changed and nobody compared base and head** ⚠️
`scripts/compare_benchmark.py` prints ratios and returns 0, so the CI benchmark step cannot fail,
and it only covers the ops `scripts/run_benchmark.sh` lists. `$WORK/bench_compare.txt` holds the
rows CI measured for this head and names the changed kernels no benchmark exercises.
Fire when the diff changes a kernel's executed path and neither the PR nor CI carries a base-vs-head
number for it. A single CI run on a shared runner is advisory; say so with the number.
Real example (PRs #1009, #1035): #950 and #930 merged with green CI and cost −8.0% and −10.4% on
the gfx950 bf16 attention path; the two fixes found them weeks later by bisection.
→ `⚠️ P5: [kernel] changed and no base-vs-head timing exists — [why CI cannot measure it]; measure [shapes] on both sides, or mark perf findings [inferred]`

---

### Housekeeping (quick scan — mostly 📝)

Report at most one or two, and only when the diff makes them concrete.

| Rule | Trigger | Flag |
|---|---|---|
| HK1 | `ir.IntegerType`, `ArithValue`, `_to_raw` or raw `arith.`/`vector.`/`scf.`/`memref.`/`llvm.` builders on a `+` line in kernel code | `📝 HK1: use the fx.* surface (fx.Int32/fx.Int64, fx.copy, fx.gemm) rather than raw dialects` |
| HK2 | `scf.IfOp` / `_if_then` in a kernel | `📝 HK2: use native Python if/for inside @flyc.jit; a device helper with control flow needs its own @flyc.jit` |
| HK3 | `inline_asm` on a `+` line | `⚠️ HK3: prefer the s_waitcnt / sched_barrier wrappers; has_side_effects=True can create a data hazard (#1073)` |
| HK4 | `SmemAllocator` / `SmemPtr` in new code | `📝 HK4: SmemAllocator is legacy — use SharedAllocator` |
| HK5 | a new helper that copies an existing one | `📝 HK5: shared helpers belong in kernels/common/; reuse the existing one (#795)` |
| HK6 | a tuned config table added inside a kernel file | `📝 HK6: tuning configs belong behind a dispatcher, not in the kernel file (#780)` |
| HK7 | a direct `flyc.compile` in a launch path | `📝 HK7: route through _run_compiled so the CompiledFunction is cached (#776)` |
| HK8 | comment lines outnumbering code in a new file | `📝 HK8: comment density is out of line with the repo` |
| HK9 | a new source file without the SPDX header | `📝 HK9: add the SPDX/copyright header that CONTRIBUTING.md requires` |
| HK10 | a new requirement, `pyproject.toml` dependency or third-party import | `📝 HK10: CONTRIBUTING.md asks for a justification for any new dependency` |
| HK11 | a new environment variable read | `📝 HK11: use the names in python/flydsl/utils/env.py; an env var read at trace time is not in the cache key unless registered there` |
| HK12 | `TODO`, `FIXME`, `raise NotImplementedError` or a lone `pass` in a new branch | `⚠️ HK12: [location] — incomplete implementation in a new code path` |
| HK13 | a new public kernel or launcher | `📝 HK13: add it to docs/prebuilt_kernels_guide.md` |
| HK14 | `sys.path.insert` / `sys.path.append` at module level | `⚠️ HK14: sys.path mutation leaks global state — use package imports` |
| HK15 | `buffer_ops.*` or `make_ptr(` on a `+` line in kernel code outside `kernels/common/buffer_ops.py` | `📝 HK15: raw buffer_ops / manual byte offsets are legacy — prefer fx.rocdl.make_buffer_tensor with layout ops (CLAUDE.md); for make_ptr, check the retyping` |
