---
name: add-target-atom-op
description: >
  Add a new target-specific Mma / Copy Op type to a FlyDSL backend dialect
  (`lib/Dialect/Fly<TARGET>/<SUBTARGET>/` + `include/flydsl/Dialect/Fly<TARGET>/IR/`).
  Covers the MmaOp/CopyOp type design, the stateful-vs-stateless variants, and the
  `emitAtomCall` / `emitAtomCallSSA` lowering contract to the backend dialect
  (LLVM/ROCDL/NVVM/SPIR-V/...). Use when adding a new tensor-core / matrix instruction
  (MFMA, WMMA, HMMA, WGMMA, ...), a new buffer / shared-memory / global copy atom, a new
  stateful copy (per-atom offset or descriptor), or bringing up a new backend dialect
  (`FlyPTX`, `FlyCPU`, ...). Reference implementations live in `FlyROCDL`.
  Usage: /add-target-atom-op
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Add a Target-Specific Mma / Copy Op to a FlyDSL Backend Dialect

Step-by-step recipe for authoring a new `MmaOp*Type` or `CopyOp*Type` in a backend dialect
(`fly_rocdl`, or a future `fly_ptx` / ...), plus the **inherent design contract** every Op author
must understand before writing a single line of code.

The examples throughout this skill draw from the `fly_rocdl` dialect (AMD ROCDL backend). The design
is deliberately backend-agnostic: the generic `!fly.mma_atom` / `!fly.copy_atom` wrappers and the
three type interfaces (`Fly_MmaOpTypeInterface`, `Fly_CopyOpTypeInterface`,
`Fly_StatefulOpTypeInterface`) live in the target-neutral `fly` dialect and know nothing about AMD,
NVIDIA, or other specifics. A new backend follows the exact same recipe — only the payload types,
the final intrinsic emission, and the directory prefix change.

---

## 1. Inherent Design: How FlyDSL Atoms Work

Internalize these five facts before adding any `Op`. They explain *why* the reference `CDNA3`,
`CDNA4`, `GFX1250` implementations look the way they do — and the same structure applies verbatim to
any new backend.

### 1.1 Two-level type design: generic wrapper + target-specific payload

There are **two kinds** of related types, and they live in different dialects:

| Level | Dialect | Type (example) | Role |
|-------|---------|----------------|------|
| generic wrapper | `fly`       | `!fly.mma_atom<...>`, `!fly.copy_atom<..., bits>` | Target-agnostic. Appears everywhere in kernel IR. |
| target payload  | backend dialect (e.g. `fly_rocdl`) | `!fly_rocdl.cdna3.mfma<...>`, `!fly_rocdl.cdna3.buffer_copy<32>` | Knows *which* concrete instruction/intrinsic to emit. |

The generic wrapper always holds a **payload** type as its first parameter.

```mlir
// Using the ROCDL backend (the current reference):
!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>
!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>

// A hypothetical NVIDIA backend would look like:
!fly.mma_atom<!fly_ptx.sm90.wgmma<64x128x16, (f16, f16) -> f32>>
!fly.copy_atom<!fly_ptx.sm80.cp_async<128>, 128>
```

Every method you see on `MmaAtomType` / `CopyAtomType` is a trampoline:

```cpp
// lib/Dialect/Fly/IR/FlyTypeDefs.cpp
Attribute MmaAtomType::getShapeMNK() const {
  return cast<MmaOpTypeInterface>(getMmaOp()).getShapeMNK();
}
LogicalResult MmaAtomType::emitAtomCall(...) const {
  return cast<MmaOpTypeInterface>(getMmaOp()).emitAtomCall(...);
}
```

**Your job** when adding a new Op is to define the *payload* type and implement the interface
methods — the wrapper and the kernel-level ops (`fly.mma_atom_call`, `fly.copy_atom_call`,
`fly.make_mma_atom`, ...) work automatically.

### 1.2 Three interfaces an Op type may implement

`include/flydsl/Dialect/Fly/IR/FlyInterfaces.td`:

| Interface                         | Required for... | Methods — **mandatory** / pipeline-dependent (see §1.3) |
|-----------------------------------|-----------------|----------------------|
| `Fly_MayStaticTypeInterface`      | Stateless atoms (CopyOp or MmaOp with no mutable state) | **`isStatic`**, **`rebuildStaticValue`** |
| `Fly_CopyOpTypeInterface`         | All CopyOps    | **`getThrLayout`**, **`getThrBitLayoutSrc/Dst/Ref`**, **`emitAtomCall`** (mem + pred), `emitAtomCallSSA` when a legal call is SSA-reachable |
| `Fly_MmaOpTypeInterface`          | All MmaOps     | **`getThrLayout`**, **`getShapeMNK`**, **`getValTypeA/B/C/D`**, **`getThrValLayoutA/B/C`**, **`emitAtomCall`**, `emitAtomCallSSA` when a legal call is SSA-reachable |
| `Fly_StatefulOpTypeInterface`    | Atoms that carry mutable per-call state (e.g. `soffset`, `imm_offset`) | **`getConvertedType`**, **`getDefaultState`**, **`setAtomState`** |

Backend dialect could provide four convenience base classes that pre-declare the right interface
combinations. In the reference ROCDL backend (`include/flydsl/Dialect/FlyROCDL/IR/Dialect.td`) they
are:

```tablegen
class FlyROCDL_CopyOp         // stateless CopyOp    : MayStatic + CopyOp
class FlyROCDL_StatefulCopyOp // stateful  CopyOp    : CopyOp + Stateful
class FlyROCDL_MmaOp          // stateless MmaOp     : MayStatic + MmaOp
class FlyROCDL_StatefulMmaOp  // stateful  MmaOp     : MmaOp    + Stateful
```

Mnemonic: **stateful => no `MayStaticTypeInterface`**; the mutable state *is* the dynamic component,
so the type is never "fully static" in the canonical-rebuild sense.

### 1.3 `emitAtomCall` and `emitAtomCallSSA` are one semantic contract

Two kernel-IR ops carry the atom invocation, and they correspond to the two interface methods:

| Kernel Op                | Operand form                                  | Lowered via           | Implementation status |
|--------------------------|-----------------------------------------------|-----------------------|-----------------------|
| `fly.copy_atom_call`     | `src/dst : !fly.memref<...>`                  | `emitAtomCall`        | **Required**          |
| `fly.mma_atom_call`      | `a/b/c/d : !fly.memref<...>`                  | `emitAtomCall`        | **Required**          |
| `fly.copy_atom_call_ssa` | promoted SSA values plus retained non-register memrefs | `emitAtomCallSSA` | Required when FlyROCDL's default SSA pass can rewrite a legal call |
| `fly.mma_atom_call_ssa`  | promoted SSA values plus retained non-register memrefs | `emitAtomCallSSA` | Required when FlyROCDL's default SSA pass can rewrite a legal call |

**Memref path (`emitAtomCall`).** An unpromoted `fly.copy_atom_call` / `fly.mma_atom_call`
lowers through `emitAtomCall`. The Op receives operand pointers/memrefs
(`!fly.memref<..., register, layout>`), is expected to issue `llvm.load` / `llvm.store` itself to
read/write register operands, and emits or delegates to the backend operation.

**SSA path (`emitAtomCallSSA`).** A pipeline may insert the
`fly-convert-atom-call-to-ssa-form` pass (see
`lib/Dialect/Fly/Transforms/ConvertAtomCallToSSAForm.cpp`). That pass inspects every `AtomCall` and,
for operands whose `register`-address-space memref has a **coalescable** layout
(`isEligibleToPromote`: stride-1 or shape-1 after coalesce), rewrites them:

1. `PtrLoadOp` pulls the whole register memref into a single SSA value of type
   `RegMem2SSAType(memref)` — which is `elemTy` when the layout has cosize 1, or 
   `vector<cosize × elemTy>` otherwise (see `RegMem2SSAType` in `Fly/Utils/PointerUtils.cpp`).
2. The `AtomCall` is replaced with `AtomCallSSA`, taking those SSA values in place of pointers.
3. For output-producing cases, a `PtrStoreOp` writes the SSA result back to the original register
   memref.

At lowering time, `AtomCallSSA` dispatches to `emitAtomCallSSA`. Only eligible
register operands become SSA; global/shared/buffer memrefs remain, so this can be
a mixed form. A payload may bitcast SSA vectors and emit the intrinsic directly,
or delegate to memref lowering when the operation still consumes memrefs.

**Concrete differences between the two methods:**

|                          | `emitAtomCall`                                  | `emitAtomCallSSA`                               |
|--------------------------|-------------------------------------------------|-------------------------------------------------|
| Operand kinds            | `Value`s of `!fly.memref<...>` types | promoted scalar/vector values plus retained non-register memrefs |
| What the method does     | Load/store register operands and emit/delegate the operation | Lower the reachable full/mixed form; may bitcast and emit or delegate to memref form |
| Return type              | `LogicalResult`                                 | `FailureOr<Value>` (the result SSA value, or `failure`) |
| Needs layout/cosize info | No — operand type already carries it            | No — caller already packed operands into `vector<N>` |
| Bitcast dance            | Typically unnecessary (load yields the right type) | Often necessary (SSA vector width may not match intrinsic's expected operand width) |
| Backend intrinsic emitted | Same                                           | Same                                            |

FlyROCDL's default pipeline always runs `fly-convert-atom-call-to-ssa-form`,
so each payload must handle every form its legal calls can reach. Sharing may go
either direction: CDNA3 MFMA loads then calls SSA, while async/TDM copies can
delegate their SSA entry back to memref lowering. A different backend may omit
SSA support only when its complete pipeline cannot create that call form.

### 1.4 ThrVal layouts describe the per-thread register footprint

Every MmaOp / CopyOp publishes layouts consumed by `TiledCopy` / `TiledMma`.
Value-granular atoms describe which thread holds each element; whole-tile target
operations may instead publish a documented sentinel layout interpreted by their payload.

| Method (MmaOp)         | What it describes |
|------------------------|-------------------|
| `getThrLayout`         | thread-count layout inside one thread group that issues the instruction (e.g. `(64):(1)` for an AMD wave64 MFMA, `(32):(1)` for AMD wave32 WMMA, `(1):(1)` for a single thread, `(128):(1)` for NVIDIA WGMMA issued by a warpgroup) |
| `getShapeMNK`          | tuple `(M, N, K)` of the instruction tile |
| `getValTypeA/B/C/D`    | per-operand element type |
| `getThrValLayoutA/B/C` | layout mapping `(thr, val)` → element coordinate in the **reference tile** (column-major `(M,K)` for A, `(N,K)` for B, `(M,N)` for C) |

| Method (CopyOp)            | What it describes |
|----------------------------|-------------------|
| `getThrLayout`             | thread count participating in one atom call |
| `getThrBitLayoutSrc/Dst/Ref` | value-granular copies use bit layouts; whole-tile payloads may use a sentinel contract |

The base `CopyAtomType::getThrValLayoutSrc()` then "recasts" the bit layout into a
`valBits`-granularity layout (see `CopyAtomType::getThrValLayout{Src,Dst,Ref}` in
`FlyTypeDefs.cpp`). This is why CopyOp types publish a **bit-layout** and MmaOp types publish a
**value-layout**: copies carry an extra `valBits` parameter on the wrapper, and one CopyOp type can
serve multiple element widths.

Use the `FxLayout / FxShape / FxStride / FxThr / FxVal / FxC` macros from
`flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc` — they're the auxiliary way to build these
`LayoutAttr`s.

### 1.5 Critical checks for ThrVal / ThrBit layouts — read before writing any

A wrong ThrVal/ThrBit layout is the #1 source of silent-wrong-result bugs in FlyDSL: the compiler
accepts it, the kernel runs, and the output is garbage. There are no good runtime diagnostics for
this. Before you commit any new `getThrValLayout*` / `getThrBitLayout*`, verify every applicable rule below
on paper or in a scratch test.

#### 1.5.1 Shape must be a top-level 2-tuple `((thr...), (val...))`

Look at any existing example: `FxLayout(FxShape(FxThr(...), FxVal(...)), FxStride(FxThr(...),
FxVal(...)))`. The top-level shape has exactly **rank 2**: outer mode 0 is the thread axes, outer
mode 1 is the value axes. Each of these modes may itself be a nested tuple.

This is not a style convention — it's load-bearing: `TiledOpUtils.h` unconditionally does
`shape.at(0)` / `shape.at(1)` / `stride.at(0)` / `stride.at(1)` to slice "thr" from "val". For
CopyOps, the same shape is passed to `layoutZippedDivide(tiledLayoutThrVal, atomTile)` where
`atomTile = (atomNumThr, atomNumVal)` is computed as the product of mode-0 and mode-1 (see
`detail::layoutTiledCopyThrValView` in `TiledOpUtils.h`).

If you nest an extra level or flatten it to rank 1, the code compiles but silently reads
`thrShape=firstLeaf`, `valShape=secondLeaf`, and produces wrong tile divisions.

#### 1.5.2 Shape-product invariants

Let `|·|` denote "total number of elements". Then:

| Op kind  | Method                    | Must satisfy |
|----------|---------------------------|--------------|
| MmaOp    | `getThrLayout`            | `\|thr\|` == number of threads that cooperate on one instruction (e.g. 64 for AMD wave64 MFMA, 32 for AMD wave32 WMMA, 128 for NVIDIA SM90 WGMMA warpgroup) |
| MmaOp    | `getThrValLayoutA`        | `\|thr\| * \|val\|` == `M * K` |
| MmaOp    | `getThrValLayoutB`        | `\|thr\| * \|val\|` == `N * K` |
| MmaOp    | `getThrValLayoutC` (and D) | `\|thr\| * \|val\|` == `M * N` |
| MmaOp    | `\|thr\|` of ThrValLayout{A,B,C} | matches `\|thr\|` of `getThrLayout` |
| MmaOp    | `\|val\|` of ThrValLayout | matches the thread's register vector width used in `emitAtomCallSSA` (e.g. `accVecSize` for C; `vecSize` of `abTyA` for A) |
| Value-granular CopyOp | `getThrLayout` | `\|thr\|` == participating threads (e.g. 1 for a per-thread load, 16 for `ds_read_tr16_b64`) |
| Value-granular CopyOp | `getThrBitLayoutSrc/Dst/Ref` | `\|val\|` == the payload's `bitSize`; all three thread modes match `getThrLayout` |
| Whole-tile CopyOp | layout methods | Follow the payload's documented operation-level contract; do not infer a `bitSize` footprint |

Violating an applicable invariant can still compile but yield undefined behavior. Thread-count mismatch is
especially insidious: a wave64 MFMA registered with `FxC(32)` (or a 32-thread NVIDIA warp MMA
registered with `FxC(16)`) will happily emit the intrinsic, but half the threads will compute on stale
registers.

#### 1.5.3 Value-granular reference coordinates are *column-major*

| Op       | Operand | Reference tile | Column-major interpretation |
|----------|---------|----------------|-----------------------------|
| MmaOp    | A       | `(M, K)`       | stride `(1, M)` is baseline |
| MmaOp    | B       | `(N, K)`       | stride `(1, N)` is baseline |
| MmaOp    | C, D    | `(M, N)`       | stride `(1, M)` is baseline |
| Value-granular CopyOp | src/dst | `(M, N)` | stride `(1, M)` is baseline |

Whole-tile CopyOps do not use this `(M,N)` table; follow the payload's own
rank, dimension, state, and coordinate contract.

#### 1.5.4 CopyOp bit-layout vs. value-layout — do not confuse them

For value-granular copies, the interface publishes `getThrBitLayout*` and the
`CopyAtomType` wrapper computes `getThrValLayout*` with
`layoutRecast(bitLayout, /*oldBits=*/1, /*newBits=*/valBits)`. Whole-tile
operations such as GFX1250 TDM can use a `(1,1)` sentinel and carry no payload
`bitSize`; validate the payload consumer instead of applying the recast footprint rules.

Consequences:
- A 32b buffer copy writes `FxShape(FxC(1), FxC(32))` for an f32 → the recast at `valBits=32` trivially
  keeps it as `FxShape(FxC(1), FxC(1))` (1 f32 per thread). The same Op reused for a 16b copy of an
  f16 pair gives `FxShape(FxC(1), FxC(2))` (2 f16 per thread), all automatically.
- If `bitSize` does not divide evenly by the downstream `valBits` (e.g. 96b / 32b f32 = 3 — fine;
  96b / 64b = 1.5 — broken), the `layoutRecast` path silently produces nonsense. It is programmer's
  duty to find this mismatch behavior.

### 1.6 Stateful atoms are lowered to an LLVM struct

A stateful Op's state lives as an `!llvm.struct<(i32, i32, ...)>` at runtime. The three methods you
implement for `StatefulOpTypeInterface` wire this up:

1. `getConvertedType(ctx)` — the concrete `!llvm.struct<...>` layout.
2. `getDefaultState(builder, loc)` — build an initial value (typically zero-initialized).
3. `setAtomState(builder, loc, struct, fieldAttr, fieldValue)` — field write. The `fieldAttr` is a
   `StringAttr` that must be one of the `AtomStateField` enum mnemonics (`"soffset"`,
   `"imm_offset"`).

Then at lowering time the field is read back via `LLVM::ExtractValueOp` with the index returned by
your static `getFieldIndex(AtomStateField)` helper. See the `CopyOpCDNA3BufferCopyType` stateful
methods (`getFieldIndex` / `getConvertedType` / `getDefaultState` / `setAtomState`) in
`CDNA3/CopyAtom.cpp` for the full template.

If your Op needs a new field kind (something other than `Soffset` or `ImmOffset`), extend the
`AtomStateField` enum in your backend's `Atom.td` — for the ROCDL reference backend that lives at
`include/flydsl/Dialect/FlyROCDL/IR/Atom.td` (this regenerates `AtomStateEnums.{h,cpp}.inc`). A new
backend should declare its own `AtomStateField` enum in its IR directory.

---

## 2. The Files You Will Touch

Below, `<BACKEND>` stands for the backend dialect's name (e.g. `FlyROCDL` today; would be `FlyPTX`,
`FlySPIRV`, etc. for a new backend) and `<SUBTARGET>` is the per-chip subdirectory (e.g.
`CDNA3`, `CDNA4`, `GFX1250` for ROCDL; would be `SM80`, `SM90`, `SM100` for a PTX port).

| File | Purpose |
|------|---------|
| `include/flydsl/Dialect/<BACKEND>/IR/MmaAtom.td` or `CopyAtom.td` | TableGen declaration of the new type (`def <BACKEND>_MmaOp<SubTarget>_<Family>` / `def <BACKEND>_CopyOp<SubTarget><Kind>`) |
| `lib/Dialect/<BACKEND>/<SUBTARGET>/MmaAtom.cpp` or `CopyAtom.cpp` | Interface method implementations |
| `lib/Dialect/<BACKEND>/CMakeLists.txt` | Add the new `.cpp` to `MLIR<BACKEND>Dialect` |
| `lib/Bindings/Python/<BACKEND>Extension.cpp` | Expose the type to Python so kernels can construct it |
| `python/flydsl/expr/<backend>/<subtarget>.py` | DSL-level constructor wrappers (e.g. `MFMA(...)`, `WMMA(...)`, `BufferCopy(...)` for ROCDL; would be `WGMMA(...)`, `CpAsyncBulk(...)` etc. for PTX) |
| `include/flydsl/Dialect/<BACKEND>/IR/Atom.td` | (Only if adding a new `AtomStateField`) extend the enum |
| `tests/mlir/Conversion/<something>.mlir` | A FileCheck test exercising the new lowering |

Concrete instantiation for the reference ROCDL backend:
`include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td`, `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`,
`MLIRFlyROCDLDialect`, etc.

---

## 3. Recipe: Add a Stateless MmaOp

Examples use the ROCDL backend with a hypothetical `CDNA5_MFMA`. For other backends substitute
`FlyROCDL_MmaOp` → your backend's base class, `CDNA<n>` → your chip family (`SM80`, `SM90`,
`X86AVX512`, ...), `ROCDL::mfma_*` → your intrinsic ops (`NVVM::WgmmaMmaAsyncOp`, `vector.contract`,
...), `--convert-fly-to-rocdl` → your conversion pass. Everything else is identical.

**Step 1 — TableGen declaration** in `include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td`:

```tablegen
def FlyROCDL_MmaOpCDNA5_MFMA : FlyROCDL_MmaOp<"MmaOpCDNA5_MFMA", "cdna5.mfma", []> {
  let parameters = (ins "int32_t":$m, "int32_t":$n, "int32_t":$k,
                        "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc);
  let assemblyFormat = "`<` custom<MNKDimensionList>($m, $n, $k) `,` "
                       "`(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";
  let builders = [TypeBuilderWithInferredContext<(ins ...), [{
    return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc); }]>];
  let genVerifyDecl = 1;
}
```

The `FlyROCDL_MmaOp` base auto-adds `DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>` +
`<Fly_MmaOpTypeInterface>`. `MNKDimensionList` is the shared parser/printer pair
(`parseMNKDimensionList` / `printMNKDimensionList` in `lib/Dialect/Fly/IR/FlyDialect.cpp`) — reuse
it.

**Step 2 — Interface methods** in `lib/Dialect/FlyROCDL/CDNA5/MmaAtom.cpp` <!-- api-check: ignore -->
(the hypothetical subtarget of this walkthrough; it does not exist yet). Clone
`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp` as the template; the only non-boilerplate methods are:

- `getThrLayout()` → `FxLayout(FxC(<wave-size>), FxC(1))`
- `getShapeMNK()` → `IntTupleAttr` of `(m, n, k)`
- `getValTypeA/B/C/D()` → operand element type (or packed vector type if the intrinsic expects it)
- `getThrValLayoutA/B/C()` → the layout that maps `(thr, val)` into the reference tile (A=(M,K),
  B=(N,K), C=(M,N), all column-major). Satisfy every invariant in §1.5 before trusting it.

**Step 3 — `verify`** (static, from `genVerifyDecl = 1`). Reject any `(m, n, k, elemTy)` tuple you
don't support, with a clear `emitError()` message; otherwise an invalid config silently hits
`return failure()` in `emitAtomCallSSA` with no diagnostic.

**Step 4 — `emitAtomCallSSA`** (required when FlyROCDL can create this form; see §1.3).
For a direct SSA implementation this is the backend intrinsic boundary. Pattern from
`MmaOpCDNA3_MFMAType::emitAtomCallSSA` in `CDNA3/MmaAtom.cpp`: derive the intrinsic's exact operand
types, `LLVM::BitcastOp` each SSA operand to match, then dispatch to lowered dialect ops. Find
intrinsic names in `llvm/include/llvm/IR/IntrinsicsAMDGPU.td` (ROCDL), `NVVMOps.td` (NVVM), or
`SPIRVOps.td` (SPIR-V).

**Step 5 — `emitAtomCall`** (mandatory entry point; see §1.3). If Step 4 exists, this is a ~15-line
shim:

```cpp
LogicalResult MmaOpCDNA5_MFMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
    Type dMemTy, Type aMemTy, Type bMemTy, Type cMemTy,
    Value atomVal, Value dPtr, Value aPtr, Value bPtr, Value cPtr) const {
  // Derive abTyA, abTyB, accTy exactly as in emitAtomCallSSA.
  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, Type{},
                             abTyA, abTyB, accTy, atomVal, Value{}, a, b, c);
  if (failed(res)) return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}
```

If you skipped Step 4, emit the intrinsic directly here.

**Step 6 — CMake.** Add `CDNA5/MmaAtom.cpp` to the source list in
`lib/Dialect/FlyROCDL/CMakeLists.txt`.

**Step 7 — Python bindings**. In `lib/Bindings/Python/FlyROCDLExtension.cpp`, add a
`PyMmaOpCDNA5_MFMAType : PyConcreteType<...>` following the existing `PyMmaOpCDNA3_MFMAType`
template, and register it in the `NB_MODULE(_mlirDialectsFlyROCDL, m)` block. Then add a thin
wrapper like `MFMA_CDNA5(m, n, k, elem, ...)` in `python/flydsl/expr/rocdl/universal.py`.

**Step 8 — FileCheck test.** Clone `tests/mlir/Conversion/mma_atom.mlir` and swap the payload type:

```mlir
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize \
// RUN:   --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s
// CHECK: rocdl.mfma.f32.16x16x32f16
```

**Step 9 — Build and verify.** `bash scripts/build.sh` rebuilds C++, bindings, and stubs. Run
FileCheck, then a 1-wave end-to-end Python kernels before trusting the layout.

---

## 4. Recipe: Add a Stateful CopyOp

Same skeleton as §3 but replacing `MayStaticTypeInterface` with `StatefulOpTypeInterface`. Clone
`CDNA3/CopyAtom.cpp` (`CopyOpCDNA3BufferCopy`) as the template. Stateful CopyOps model backend
concepts with per-call mutable state (AMD buffer descriptors, NVIDIA TMA descriptors, per-atom SM
offsets, ...). State always lowers to `!llvm.struct<...>` in rocdl backend, so only the field set
differs.

**Step 1 — TableGen.** Pick `FlyROCDL_StatefulCopyOp` as the base:

```tablegen
def FlyROCDL_CopyOpCDNA5GlobalCopy
    : FlyROCDL_StatefulCopyOp<"CopyOpCDNA5GlobalCopy", "cdna5.global_copy", []> {
  let parameters = (ins "int32_t":$bitSize);
  let assemblyFormat = "`<` $bitSize `>`";
}
```

**Step 2 — Stateful methods** (`getFieldIndex`, `getConvertedType`, `getDefaultState`,
`setAtomState`). Template directly from the stateful methods of `CopyOpCDNA3BufferCopyType` in
`CDNA3/CopyAtom.cpp`. Key points:

- `getFieldIndex(AtomStateField)` is a static `switch` returning the struct field index.
- `getConvertedType(ctx)` returns the `LLVM::LLVMStructType::getLiteral(ctx, {i32, i32, ...})`
  matching your state.
- `getDefaultState` builds an `UndefOp` then `InsertValueOp`s zero into each field.
- `setAtomState` must return `nullptr` on unrecognized fields (not fail-silently to `success`).

**Step 3 — `getThrLayout` + `getThrBitLayoutSrc/Dst/Ref`** (all in **bit** granularity). For a
simple per-thread copy: `FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)))` for
Src/Dst/Ref. If Src ≠ Dst (e.g. LDS-read-transpose), Ref usually mirrors the register side — see
`CDNA4/CopyAtom.cpp`. All three layouts must satisfy the invariants in §1.5.

**Step 4 — `emitAtomCallSSA`** (required when FlyROCDL can create this form; see §1.3).
Lower the reachable operands directly or delegate to the memref form. For a direct
implementation, extract state fields with `LLVM::ExtractValueOp` and dispatch to
the backend intrinsic. Pattern: the unpredicated
`CopyOpCDNA3BufferCopyType::emitAtomCallSSA` overload in `CDNA3/CopyAtom.cpp`.

**Step 5 — Predicated SSA variant.** Wrap the unpredicated form in `scf::IfOp` — load side yields
`result` in `then` / old dst in `else`; store side uses a single-branch `scf.if`. Template: the
predicated `CopyOpCDNA3BufferCopyType::emitAtomCallSSA` overload (the one taking `Value pred`) in
`CDNA3/CopyAtom.cpp`.

**Step 6 — memref-form `emitAtomCall` (mandatory + predicated).** `LLVM::LoadOp`/`StoreOp` shim
around the SSA form (or intrinsic dispatch directly if you skipped Steps 4-5). Template: the two
`CopyOpCDNA3BufferCopyType::emitAtomCall` overloads in `CDNA3/CopyAtom.cpp`.

**Steps 7-9 — CMake / Python / test.** Identical to §3 Steps 6-8. Python wrapper follows
`CopyOpCDNA3BufferCopyType.get(bit_size)`.

---

## 5. Adding a New `AtomStateField` (rare)

If your stateful Op needs a field kind no existing Op uses (`"voffset"`, `"cpol"`, `"tensor_map"`
…), extend the enum in your backend's `Atom.td` (`include/flydsl/Dialect/FlyROCDL/IR/Atom.td` for
ROCDL):

```tablegen
def FlyROCDL_AtomStateField : I32EnumAttr<"AtomStateField", "", [
  I32EnumAttrCase<"Soffset",   0, "soffset">,
  I32EnumAttrCase<"ImmOffset", 1, "imm_offset">,
  I32EnumAttrCase<"Voffset",   2, "voffset">         // <-- NEW
]> { let genSpecializedAttr = 0; let cppNamespace = FlyROCDL_Dialect.cppNamespace; }
```

Use it in your `getFieldIndex` switch. `fly.atom.set_value(%atom, "voffset", %val)` then works
automatically via `AtomSetValueOp`.

---

## 6. Review and verification

Treat construction, lowering, final instruction selection, and device behavior
as separate evidence boundaries.

- The TableGen verifier's accepted parameter set must equal the configurations
  handled by layout construction and `emitAtomCall{SSA}`. Add a positive case
  for each new dispatch branch and a negative case at each accepted/rejected
  boundary; a constructible atom that falls through during conversion is a bug.
- Derive reachable call forms from the SSA pass's eligibility and each operand's
  address space. Cover unpromoted memref and every reachable full/mixed SSA form;
  do not require an impossible pure-SSA form for whole-tile or async copies.
  Copy changes also cover predicates; stateful atoms cover changed state fields.
  For MMA, exactly one layer must write D in each reachable form.
- Build a matrix from distinct target/ABI, layout, dtype/packing, shape,
  state/modifier, and call-path branches. Use representative and interacting
  pairs rather than a Cartesian product, plus a sibling-target control for
  shared changes.
- FileCheck proves only the checked intermediate IR. Inspect normalized final
  ISA for opcode, operand order, and modifiers; use resource diff only for
  resource claims. Layout, signedness, packing, state, or predicate changes also
  require a numerical test on matching hardware against an independent oracle.

---

### Recommended reading order

Files (4) and (7) are target-neutral; the rest are ROCDL templates a new backend mirrors in its own
tree.

1. `include/flydsl/Dialect/FlyROCDL/IR/Dialect.td` — base classes
2. `include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td` — MmaOp type decls
3. `include/flydsl/Dialect/FlyROCDL/IR/CopyAtom.td` — CopyOp type decls
4. `include/flydsl/Dialect/Fly/IR/FlyInterfaces.td` — **target-neutral** interface contracts
5. `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp` — simplest stateless MmaOp
6. `lib/Dialect/FlyROCDL/CDNA3/CopyAtom.cpp` — all three CopyOp patterns
7. `lib/Dialect/Fly/IR/FlyTypeDefs.cpp` — **target-neutral** wrapper trampolines (see the
   `CopyAtomType::*` and `MmaAtomType::*` method definitions)
8. `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp` — `MakeCopyAtomOpLowering` / `MakeMmaAtomOpLowering` /
   `AtomSetValueOpLowering` / `CopyAtomCallLowering` / `CopyAtomCallSSALowering` /
   `MmaAtomCallLowering` / `MmaAtomCallSSALowering` (callers of your interface methods)

