# FlyDSL Codegen Pass Plugin

How FlyDSL inserts its own AMDGPU `MachineFunctionPass`es into the LLVM
codegen pipeline **without modifying LLVM source or applying patches to
the LLVM build**.

## Why this exists

Two AMDGPU MIR-level optimizations are owned by FlyDSL:

| Pass | Effect |
|---|---|
| `fly-amdgpu-prefer-agpr-for-ds-read` | Pin nontemporal DS-load destinations (and their COPY/PHI fan-out) to AGPR, so loop-carried accumulators stay in AGPR across iterations instead of bouncing through VGPR. |
| `fly-amdgpu-mfma-tie-vdst-src2` | On 128-bit-dst MFMA/SMFMAC instructions, tie `vdst` to `src2` at the `MachineInstr` level so the register coalescer reuses the accumulator. Equivalent to flipping the `VOPProfileMAI` `NoDstOverlap` predicate from `> 128` to `>= 128`. |

Both are `MachineFunctionPass`es — they live below LLVM IR, in the
codegen pipeline. Historically they were carried as a fork: pass `.cpp`
files staged into `llvm/lib/Target/AMDGPU/` plus a unified diff that
patched `AMDGPU.h` / `AMDGPUTargetMachine.cpp` / `CMakeLists.txt` to
register and schedule them. That coupled FlyDSL to a specific LLVM
commit and made every LLVM bump a patch-rebase exercise.

The new design keeps the passes **inside FlyDSL's own C++ tree** and
hooks them into AMDGPU codegen at runtime via
`TargetPassConfig::insertPass(...)`. LLVM is built vanilla.

## What changed in the source tree

### Added

```
include/flydsl/
└── Codegen/AMDGPU/
    ├── PreferAGPRForDSRead.h
    ├── MFMATieVDSTToSrc2.h
    └── PassRegistration.h          # registerFlyAMDGPUCodegenPasses(PassRegistry&)

lib/Codegen/AMDGPU/
├── CMakeLists.txt
├── PreferAGPRForDSRead.cpp         # uses public addRegAllocationHint API; no LLVM source edit
├── MFMATieVDSTToSrc2.cpp           # uses public MachineInstr::tieOperands; no LLVM source edit
└── PassRegistration.cpp            # INITIALIZE_PASS_* + registerFlyAMDGPUCodegenPasses

include/flydsl/Dialect/FlyROCDL/Target/
└── FlyAMDGPUTarget.h               # registerFlyAMDGPUTargetInterfaceExternalModels

lib/Dialect/FlyROCDL/Target/
├── CMakeLists.txt
└── FlyAMDGPUTarget.cpp             # external model on #rocdl.target that injects our passes
```

C-API surface (`include/flydsl-c/FlyROCDLDialect.h`,
`lib/CAPI/Dialect/FlyROCDL/...`):

- `mlirRegisterFlyAMDGPUCodegenPasses()` — registers passes with the
  global `PassRegistry` (legacy PM bookkeeping).
- `mlirRegisterFlyAMDGPUTargetInterfaceExternalModels(MlirDialectRegistry)` —
  attaches our `gpu::TargetAttrInterface` external model to
  `#rocdl.target`. Called **after** upstream's
  `registerROCDLTargetInterfaceExternalModels` so ours wins.

### Changed

- `python/mlir_flydsl/FlyRegisterEverything.cpp` — calls the two new C-API
  registrars after `mlirRegisterAllDialects` and `mlirRegisterAllPasses`.
- `lib/CMakeLists.txt`, `include/flydsl/CMakeLists.txt`,
  `lib/Dialect/FlyROCDL/CMakeLists.txt` — add the new subdirectories.
- `scripts/build_llvm.sh` — reverted to a vanilla LLVM checkout. The
  staging step and patch loop are gone. `LLVM_BRANCH` /
  `LLVM_HASH` semantics unchanged.

### Removed

- `thirdparty/llvm-patches/` (entire directory)
- `llvm-passes/` (entire directory)

## How the runtime injection works

```text
                    @flyc.kernel(...) compile path
                                │
                                ▼
                   MLIR pipeline (FlyDSL backend)
                                │
                                ▼
            rocdl-attach-target  →  attaches #rocdl.target<...>
                                │
                                ▼
            gpu-module-to-binary{format=fatbin opts="..."}
                                │
                                ▼
      gpu::TargetAttrInterface::serializeToObject(#rocdl.target, ...)
                                │
                                ▼
       FlyAMDGPUTargetAttrImpl  ◄── our external model (replaces upstream's)
                                │
                                ▼
       Subclass of mlir::ROCDL::SerializeGPUModuleBase
                                │
                                ▼
           Override moduleToObject(llvm::Module &):
              1. Translate to LLVM IR (parent helper)
              2. Build AMDGPU TargetMachine
              3. legacy::PassManager PM;
                 TargetPassConfig *PC = TM->createPassConfig(PM);
                 PM.add(PC);
                 PC->insertPass(&AnchorPassID, &OurPass1ID);  ◄─ injection point
                 PC->insertPass(&AnchorPassID, &OurPass2ID);
                 PC->addISelPasses(); PC->addMachinePasses(); ...
                 PM.run(*module);
              4. Return ELF bytes
```

The two passes are linked into FlyDSL's main `_mlirDialectsFly*.so`
together with the rest of LLVM AMDGPU codegen, so they share the same
`PassRegistry`, `cl::opt` registry, and `OpName` tables. There is no
dlopen/plugin .so — the "decoupling" is purely **compile-time vs
patch-time**: we no longer modify LLVM source, we just add C++ files to
FlyDSL.

## Why the rewrite of Pass 1 was necessary

The previous fork added `AMDGPURI::PreferAGPR = 3` to LLVM's
`SIRegisterInfo.h` and a corresponding `case` in
`SIRegisterInfo::getRegAllocationHints()`. That branch is what turned a
typed hint into actual AGPR physical-register suggestions for the
register allocator.

Without that LLVM-side change, the typed hint is meaningless. The new
implementation drops the typed-hint mechanism entirely:

```cpp
// OLD (required SIRegisterInfo edit):
MRI.setRegAllocationHint(VirtReg, AMDGPURI::PreferAGPR, Register());

// NEW (uses public API, works on stock LLVM):
const TargetRegisterClass *AGPR_RC = TRI.getEquivalentAGPRClass(CurRC);
if (!AGPR_RC) return false;
MRI.constrainRegClass(VirtReg, AGPR_RC);
// AGPR_RC pinning alone is enough — RegAllocGreedy will pick AGPR phys regs.
```

`constrainRegClass(VirtReg, AGPR_RC)` is the public path: it tightens
the virtual register's class so the allocator can only pick AGPRs from
the equivalent class. No new enum, no `getRegAllocationHints` case
needed.

Pass 2 was already free of LLVM source modifications — it only calls
`MachineInstr::tieOperands(...)`, which is a public method.

## How `PC->insertPass` injects into AMDGPU's pipeline

`TargetPassConfig::insertPass(AnalysisID Anchor, IdentifyingPassPtr P)`
records `(Anchor → P)` in a per-`TargetPassConfig` `InsertedPasses`
list. When pipeline construction later calls `addPass(Anchor)` — which
AMDGPU does inside `GCNPassConfig::addPreRegAlloc()` for
`AMDGPUPrepareAGPRAllocLegacyID` — `addPass` consults that list and
splices `P` in immediately after `Anchor`.

The anchor we use is `AMDGPUPrepareAGPRAllocLegacyID`, which is
declared in `llvm/lib/Target/AMDGPU/AMDGPU.h`. That header is internal
to the AMDGPU codegen library and is not installed by `make install`.
FlyDSL's `lib/Codegen/AMDGPU/PassRegistration.cpp` re-declares only
the symbol we need via a forward `extern`:

```cpp
namespace llvm {
extern char &AMDGPUPrepareAGPRAllocLegacyID;
}
```

This avoids dragging in any AMDGPU internal header. The symbol resolves
at link time against the static AMDGPU codegen lib already pulled in by
`MLIRROCDLTarget`.

## Build

```bash
bash scripts/build_llvm.sh   # vanilla LLVM checkout, no patches
bash scripts/build.sh        # builds FlyDSL .so with passes inside
pip install -e .
```

No `thirdparty/llvm-patches/`, no `llvm-passes/` staging, no LLVM source
edit. Bumping `thirdparty/llvm-hash.txt` to a new commit only requires
verifying the two AMDGPU symbols we depend on
(`AMDGPUPrepareAGPRAllocLegacyID`, `SIRegisterInfo::getEquivalentAGPRClass`)
still exist with the same signatures — both have been stable across many
LLVM releases.

## Surfacing the passes from Python

A new compile hint toggles each pass via `cl::opt` (already supported by
`flydsl.compiler.llvm_options`):

```python
@flyc.kernel(compile_hints={
    "prefer_agpr_for_ds_read": True,           # default off
    # "mfma_tie_vdst_src2_for_128": False,     # default on, escape hatch
})
def my_kernel(...): ...
```

`RocmBackend.default_llvm_options(compile_hints=...)` translates these
into `{"fly-amdgpu-prefer-agpr-for-ds-read": True, ...}` which the
`llvm_options` context manager pushes into `cl::opt` storage before
codegen runs.

## Caveats

1. **Not LLVM-version-portable.** "No patches" ≠ "any LLVM works". LLVM
   has no stable C++ ABI; bumping `thirdparty/llvm-hash.txt` may require
   adjusting our pass code for `MachineFunctionPass` API drift,
   `TargetPassConfig::insertPass` signature changes, etc. The win is
   that the change surface is small (a handful of files inside FlyDSL)
   and the diff is auditable in the FlyDSL repo, not as a crusty
   `.patch` against a moving upstream.
2. **AMDGPU codegen is statically linked** into FlyDSL's .so. There is
   exactly one copy of the AMDGPU `PassRegistry` and `cl::opt` registry
   in the process, which is what makes the `cl::opt` toggling and
   `insertPass` injection work. Switching FlyDSL to dynamically link
   against `libLLVMAMDGPUCodeGen.so` would not require code changes,
   but is out of scope here.
3. **No support for "load this pass from a .so at runtime".** The
   passes are compiled into FlyDSL. To add a new codegen pass, drop a
   `.cpp` under `lib/Codegen/AMDGPU/`, register it from
   `PassRegistration.cpp`, and add an `insertPass` call in
   `FlyAMDGPUTarget.cpp`. No external plugin loading mechanism — that
   was investigated and rejected (see "Why we didn't ship a real LLVM
   plugin .so" below).

## Why we didn't ship a real LLVM plugin .so

`opt -load-pass-plugin` only works for new-PM IR-level passes invoked
from `opt`/`llc`. AMDGPU codegen runs inside FlyDSL's process via
upstream MLIR's `gpu-module-to-binary`, which uses the legacy PM and
calls `LLVMTargetMachine::addPassesToEmitFile`. That path has no plugin
hook. Even with the new PM, `MachineFunctionPass`es from a separately
built .so face two killers:

1. AMDGPU internal headers (`AMDGPU.h`, `SIInstrInfo.h`, `GCNSubtarget.h`)
   aren't installed, so the plugin can't be built outside the LLVM
   source tree anyway.
2. AMDGPU codegen would be statically linked into both FlyDSL.so and
   the plugin.so, giving two copies of every codegen-internal global
   (`PassRegistry`, `cl::opt` registry, `OpName` tables). Symbol
   resolution does not silently merge them.

The chosen design — pass `.cpp` files compiled into FlyDSL's own .so —
achieves the same goal (no LLVM patches) without the plugin pain.
