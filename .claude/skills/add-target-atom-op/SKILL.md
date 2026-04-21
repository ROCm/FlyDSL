---
name: add-target-atom-op
description: >
  Add a new target-specific Mma / Copy Op type to any FlyDSL backend
  dialect (`lib/Dialect/Fly<TARGET>/<SUBTARGET>/` +
  `include/flydsl/Dialect/Fly<TARGET>/IR/`). Explains the `MmaOp`-type /
  `CopyOp`-type design (each type plugs into the generic
  `!fly.mma_atom<...>` / `!fly.copy_atom<...>` wrapper through
  `Fly_MmaOpTypeInterface` / `Fly_CopyOpTypeInterface`), the
  stateful-vs-stateless variants (`Fly_StatefulOpTypeInterface`), and the
  required `emitAtomCall` / `emitAtomCallSSA` lowering contract to the
  backend dialect (LLVM/ROCDL/NVVM/SPIR-V/...). Use when adding a new
  tensor-core / matrix instruction (MFMA, WMMA, HMMA, WGMMA, ...), a new
  buffer / shared-memory / global copy atom, a new stateful copy (e.g.
  per-atom offset or descriptor), or bringing up a new backend dialect
  (`FlyPTX`, `FlyCPU`, etc.). The current reference implementation is
  `FlyROCDL` with `CDNA3` MFMA, `CDNA3` BufferCopy, `CDNA4`
  LDS-read-transpose, treat these as templates, not
  prerequisites. Usage: /add-target-atom-op
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

| Interface                         | Required for... | Methods — **mandatory** / *optional* (see §1.3) |
|-----------------------------------|-----------------|----------------------|
| `Fly_MayStaticTypeInterface`      | Stateless atoms (CopyOp with *no* mutable state; all MmaOps today) | **`isStatic`**, **`rebuildStaticValue`** |
| `Fly_CopyOpTypeInterface`         | All CopyOps    | **`getThrLayout`**, **`getThrBitLayoutSrc/Dst/Ref`**, **`emitAtomCall`** (mem + pred), *`emitAtomCallSSA`* (mem + pred — only if `fly-convert-atom-call-to-ssa-form` is in the pipeline) |
| `Fly_MmaOpTypeInterface`          | All MmaOps     | **`getThrLayout`**, **`getShapeMNK`**, **`getValTypeA/B/C/D`**, **`getThrValLayoutA/B/C`**, **`emitAtomCall`**, *`emitAtomCallSSA`* (only if SSA-promotion pass is active) |
| `Fly_StatefulOpTypeInterface`    | Atoms that carry mutable per-call state (e.g. `soffset`, `imm_offset`) | **`getConvertedType`**, **`getDefaultState`**, **`setAtomState`** |

Backend dialect could provide four convenience base classes that pre-declare the right interface
combinations. In the reference ROCDL backend (`include/flydsl/Dialect/FlyROCDL/IR/Dialect.td`) they
are:

```tablegen
class FlyROCL_CopyOp         // stateless CopyOp    : MayStatic + CopyOp
class FlyROCL_StatefulCopyOp // stateful  CopyOp    : CopyOp + Stateful
class FlyROCL_MmaOp          // stateless MmaOp     : MayStatic + MmaOp
class FlyROCL_StatefulMmaOp  // stateful  MmaOp     : MmaOp    + Stateful
```

Mnemonic: **stateful => no `MayStaticTypeInterface`**; the mutable state *is* the dynamic component,
so the type is never "fully static" in the canonical-rebuild sense.

### 1.3 `emitAtomCall` vs `emitAtomCallSSA` — only `emitAtomCall` is mandatory

Two kernel-IR ops carry the atom invocation, and they correspond to the two interface methods:

| Kernel Op                | Operand form                                  | Lowered via           | Implementation status |
|--------------------------|-----------------------------------------------|-----------------------|-----------------------|
| `fly.copy_atom_call`     | `src/dst : !fly.memref<...>`                  | `emitAtomCall`        | **Required**          |
| `fly.mma_atom_call`      | `a/b/c/d : !fly.memref<...>`                  | `emitAtomCall`        | **Required**          |
| `fly.copy_atom_call_ssa` | `src/dst : SSA value or !fly.memref<..., addressSpace != Register>` | `emitAtomCallSSA` | **Optional** — only needed if `fly-convert-atom-call-to-ssa-form` appears in the pipeline |
| `fly.mma_atom_call_ssa`  | `a/b/c : SSA value or !fly.memref<..., addressSpace != Register>`   | `emitAtomCallSSA`     | **Optional** (same condition) |

**Default path (memref / `emitAtomCall`).** Every `fly.copy_atom_call` / `fly.mma_atom_call` in the
IR lowers through `emitAtomCall`. The Op receives the operand *pointers* into register memory
(`!fly.memref<..., register, layout>`), is expected to issue `llvm.load` / `llvm.store` itself to
read/write threads' registers, and emit the backend intrinsic in between. This is sufficient for the
full compile-to-binary pipeline — no SSA version required.

**Optional path (SSA / `emitAtomCallSSA`).** A pipeline may insert the
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

At lowering time, `AtomCallSSA` dispatches to `emitAtomCallSSA` instead of `emitAtomCall`. The Op's
job there is **just the intrinsic + any required `LLVM::BitcastOp` between the SSA `vector<...>` and
the intrinsic's expected packed type** — no loads or stores because the SSA values already live in
registers.

**Concrete differences between the two methods:**

|                          | `emitAtomCall`                                  | `emitAtomCallSSA`                               |
|--------------------------|-------------------------------------------------|-------------------------------------------------|
| Operand kinds            | `Value`s of type `!fly.memref<..., register>` (lowered to `!llvm.ptr`) | `Value`s of scalar / `vector<Nxelem>` type |
| What the method does     | `LLVM::LoadOp` to fetch operands → intrinsic → `LLVM::StoreOp` to write result | (optional bitcast to intrinsic's packed type) → intrinsic → return `Value` / `failure` |
| Return type              | `LogicalResult`                                 | `FailureOr<Value>` (the result SSA value, or `failure`) |
| Needs layout/cosize info | No — operand type already carries it            | No — caller already packed operands into `vector<N>` |
| Bitcast dance            | Typically unnecessary (load yields the right type) | Often necessary (SSA vector width may not match intrinsic's expected operand width) |
| Backend intrinsic emitted | Same                                           | Same                                            |

In practice every reference Op implements `emitAtomCall` as a thin shim over `emitAtomCallSSA` —
load operands, call `emitAtomCallSSA`, store the result. See `MmaOpCDNA3_MFMAType::emitAtomCall` in
`CDNA3/MmaAtom.cpp` for the canonical shim and `CopyOpCDNA3BufferAtomicType::emitAtomCall` in
`CDNA3/CopyAtom.cpp` for a CopyOp instance. **If your downstream pipeline never runs
`fly-convert-atom-call-to-ssa-form`, you may skip `emitAtomCallSSA` entirely and write a
self-contained `emitAtomCall`** — but the shim pattern is strictly better because it keeps the two
paths in sync for free.

### 1.4 ThrVal layouts describe the per-thread register footprint

Every MmaOp / CopyOp must publish layouts that describe *which thread holds which element* of the
tile. This is consumed by `TiledCopy` / `TiledMma` in the layout-lowering pass.

| Method (MmaOp)         | What it describes |
|------------------------|-------------------|
| `getThrLayout`         | thread-count layout inside one thread group that issues the instruction (e.g. `(64):(1)` for an AMD wave64 MFMA, `(32):(1)` for AMD wave32 WMMA, `(1):(1)` for a single thread, `(128):(1)` for NVIDIA WGMMA issued by a warpgroup) |
| `getShapeMNK`          | tuple `(M, N, K)` of the instruction tile |
| `getValTypeA/B/C/D`    | per-operand element type |
| `getThrValLayoutA/B/C` | layout mapping `(thr, val)` → element coordinate in the **reference tile** (column-major `(M,K)` for A, `(N,K)` for B, `(M,N)` for C) |

| Method (CopyOp)            | What it describes |
|----------------------------|-------------------|
| `getThrLayout`             | thread count participating in one atom call |
| `getThrBitLayoutSrc/Dst/Ref` | layout in **bit-granularity** — shape is `(num_threads, num_bits)` — one bit per leaf |

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
this. Before you commit any new `getThrValLayout*` / `getThrBitLayout*`, verify **every** rule below
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
| CopyOp   | `getThrLayout`            | `\|thr\|` == number of threads participating in one atom call (e.g. 1 for a per-thread load, 16 for AMD `ds_read_tr16_b64`) |
| CopyOp   | `getThrBitLayoutSrc/Dst/Ref` | `\|val\|` == `bitSize` (the Op's `bitSize` parameter or per-atom constant). Shape is always `(|thr|, bitSize)`. |
| CopyOp   | `\|thr\|` of ThrBitLayout{Src,Dst,Ref} | all three equal and equal to `\|thr\|` of `getThrLayout` |

Violating any of these still compiles but yields undefined behavior. Thread-count mismatch is
especially insidious: a wave64 MFMA registered with `FxC(32)` (or a 32-thread NVIDIA warp MMA
registered with `FxC(16)`) will happily emit the intrinsic, but half the threads will compute on stale
registers.

#### 1.5.3 Reference coordinate system is *column-major*, not row-major

| Op       | Operand | Reference tile | Column-major interpretation |
|----------|---------|----------------|-----------------------------|
| MmaOp    | A       | `(M, K)`       | stride `(1, M)` is baseline |
| MmaOp    | B       | `(N, K)`       | stride `(1, N)` is baseline |
| MmaOp    | C, D    | `(M, N)`       | stride `(1, M)` is baseline |
| CopyOp   | src/dst | `(M, N)`       | stride `(1, M)` is baseline |

#### 1.5.4 CopyOp bit-layout vs. value-layout — do not confuse them

The interface publishes `getThrBitLayout*` (bit granularity); the `CopyAtomType` wrapper computes
`getThrValLayout*` by calling `layoutRecast(bitLayout, /*oldBits=*/1, /*newBits=*/valBits)` (see
`CopyAtomType::getThrValLayout{Src,Dst,Ref}` in `FlyTypeDefs.cpp`).

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
`FlyROCL_MmaOp` → your backend's base class, `CDNA<n>` → your chip family (`SM80`, `SM90`,
`X86AVX512`, ...), `ROCDL::mfma_*` → your intrinsic ops (`NVVM::WgmmaMmaAsyncOp`, `vector.contract`,
...), `--convert-fly-to-rocdl` → your conversion pass. Everything else is identical.

**Step 1 — TableGen declaration** in `include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td`:

```tablegen
def FlyROCDL_MmaOpCDNA5_MFMA : FlyROCL_MmaOp<"MmaOpCDNA5_MFMA", "cdna5.mfma", []> {
  let parameters = (ins "int32_t":$m, "int32_t":$n, "int32_t":$k,
                        "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc);
  let assemblyFormat = "`<` custom<MNKDimensionList>($m, $n, $k) `,` "
                       "`(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";
  let builders = [TypeBuilderWithInferredContext<(ins ...), [{
    return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc); }]>];
  let genVerifyDecl = 1;
}
```

The `FlyROCL_MmaOp` base auto-adds `DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>` +
`<Fly_MmaOpTypeInterface>`. `MNKDimensionList` is the shared parser/printer pair
(`parseMNKDimensionList` / `printMNKDimensionList` in `lib/Dialect/Fly/IR/FlyDialect.cpp`) — reuse
it.

**Step 2 — Interface methods** in `lib/Dialect/FlyROCDL/CDNA5/MmaAtom.cpp`. Clone
`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp` as the template; the only non-boilerplate methods are:

- `getThrLayout()` → `FxLayout(FxC(<wave-size>), FxC(1))`
- `getShapeMNK()` → `IntTupleAttr` of `(m, n, k)`
- `getValTypeA/B/C/D()` → operand element type (or packed vector type if the intrinsic expects it)
- `getThrValLayoutA/B/C()` → the layout that maps `(thr, val)` into the reference tile (A=(M,K),
  B=(N,K), C=(M,N), all column-major). Satisfy every invariant in §1.5 before trusting it.

**Step 3 — `verify`** (static, from `genVerifyDecl = 1`). Reject any `(m, n, k, elemTy)` tuple you
don't support, with a clear `emitError()` message; otherwise an invalid config silently hits
`return failure()` in `emitAtomCallSSA` with no diagnostic.

**Step 4 — `emitAtomCallSSA`** (optional, only needed when `fly-convert-atom-call-to-ssa-form` is in
the pipeline; see §1.3). The only place you touch backend intrinsics. Pattern from
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
wrapper like `MFMA_CDNA5(m, n, k, elem, ...)` in `python/flydsl/expr/rocdl.py`.

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

**Step 1 — TableGen.** Pick `FlyROCL_StatefulCopyOp` as the base:

```tablegen
def FlyROCDL_CopyOpCDNA5GlobalCopy
    : FlyROCL_StatefulCopyOp<"CopyOpCDNA5GlobalCopy", "cdna5.global_copy", []> {
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

**Step 4 — `emitAtomCallSSA`** (optional, see §1.3). Extract state fields with
`LLVM::ExtractValueOp`, then dispatch to the backend intrinsic. Pattern: the unpredicated
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

### Recommended reading order

Files (4) and (8) are target-neutral; the rest are ROCDL templates a new backend mirrors in its own
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

