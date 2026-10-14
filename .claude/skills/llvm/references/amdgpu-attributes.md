# AMDGPU function attributes

Load this when you need the full attribute list rather than the handful in
`SKILL.md` §6. Per-arch behaviour lives in `arch-gfx950.md` / `arch-gfx1250.md`.

The canonical upstream table is `llvm/docs/AMDGPUUsage.rst` ("AMDGPU LLVM IR
Attributes"). Everything below was read from the pinned LLVM
(the pin in `thirdparty/llvm-build-info.json`) or measured with `llc`.

## How to set one from FlyDSL

These are **IR function attributes, not command-line flags.** `llc` rejects
`--amdgpu-waves-per-eu=2` outright. Three routes reach them:

```python
# 1. The compile hint (waves_per_eu only) - also lowers to rocdl.waves_per_eu
CompilationContext.compile_hints({"waves_per_eu": 2})

# 2. Per-kernel MLIR attributes - the general route
kernel_attrs = {
    "rocdl.waves_per_eu": 2,
    "rocdl.flat_work_group_size": "256,256",
}

# 3. Raw LLVM passthrough - for anything with no rocdl.* spelling
kernel_attrs = {"llvm.passthrough": [["no-nans-fp-math", "true"]]}
```

⚠ A global `waves_per_eu` compile hint **overwrites** a per-kernel
`rocdl.waves_per_eu` *and* a `passthrough` `amdgpu-waves-per-eu` entry
(`RocmBackend.lower_compile_hints`, pinned by
`test_rocm_lower_wpe_preserves_source_default_and_overrides_kernel_entries`).

## Value syntax

Pair-valued attributes split on `,`. The second value is optional **only** where
`OnlyFirstRequired=true`: that is `amdgpu-waves-per-eu`, `amdgpu-lds-size` and
`amdgpu-agpr-alloc` — but **not** `amdgpu-flat-work-group-size`, which requires
both. A malformed first integer calls `Ctx.emitError`; a malformed *second* one
may not.

## Attributes you set

| Attribute | Value | Notes |
|---|---|---|
| `amdgpu-flat-work-group-size` | `"<min>,<max>"` | **Both required.** Tune this first: its implied waves-per-EU bound **wins** over a conflicting explicit `amdgpu-waves-per-eu` (`AMDGPUSubtarget.cpp:193-201`). ⚠ Invalid input reverts to Default with **no diagnostic** — [measured] `"512,128"` and `"1,99999"` both yielded `.max_flat_workgroup_size 1024`. |
| `amdgpu-waves-per-eu` | `"<min>[,<max>]"` | The preferred occupancy control. Max 8 on CDNA, **16** on gfx1250. ⚠ Out-of-range requests are discarded wholesale, not clamped. |
| `amdgpu-num-vgpr` | `"<N>"` | ⚠ **Deprecated** (`AMDGPUUsage.rst:2534`, "use amdgpu-waves-per-eu instead"). ⚠ **Silently doubled on gfx90a+** (`GCNSubtarget.cpp:624-626`) — see `arch-gfx950.md` §3. |
| `amdgpu-num-sgpr` | `"<N>"` | ⚠ Deprecated (`:2537`). Zeroes the request in three separate conditions. |
| `amdgpu-agpr-alloc` | `"<min>[,<max>]"` | Splits the unified 512 file. Defaults to 256/256 when AGPRs are needed. ⚠ **Inert on gfx1250** (no AGPRs). ⚠ UB if a function needing more AGPRs is reached from one carrying a lower bound. |
| `amdgpu-max-num-workgroups` | `"<x>[,<y>,<z>]"` | Launch-bound hint. |
| `amdgpu-dynamic-vgpr-block-size` | `16` or `32` | ⚠ Anything else is **silently coerced to 0 = disabled**. Unconfirmed on gfx1250. |
| `amdgpu-cluster-dims` | dims | Only meaningful where `FeatureClusters` holds (gfx1250). FlyDSL injects this via its own `fly-rocdl-cluster-attr` pass. |
| `amdgpu-sched-strategy` | strategy name | Attribute form of the flag; ⚠ the **attribute wins over the cl::opt** (`AMDGPUTargetMachine.cpp:602-611`). |
| `amdgpu-expert-scheduling-mode` | bool | gfx12+ only; ⚠ the **flag** overrides this module-wide via `getNumOccurrences()`. |
| `amdgpu-ieee` | bool | ⚠ Gated on `FeatureDX10ClampAndIEEEMode` — a **silent no-op on GFX12+**. [measured] gfx942/gfx1030 emit `.amdhsa_ieee_mode 0`; gfx1200 emits no directive at all. |
| `amdgpu-dx10-clamp` | bool | Same gate, same silent inertness on GFX12+. |

## Floating-point attributes

Reach these through `passthrough`.

⚠ **The string form of `denormal-fp-math-f32` no longer does anything on this
pin.** [measured] on gfx950, `.amdhsa_float_denorm_mode_32` stays at its default
`3` whether or not you set it — yet the attribute still appears in the IR, so a
grep for it reports success. That is a false positive of exactly the kind this
skill exists to prevent. The attention kernels in this tree still carry the old
spelling, so their DAZ flag is not reaching the hardware mode bit.

⚠ **A structured attribute cannot be smuggled through `passthrough`.**
`passthrough` carries string key/value pairs only; putting `denormal_fpenv(...)`
in the *name* slot yields a string attribute of that literal name, which LLVM
ignores.

⚠ **And on the pinned LLVM there is no working `gpu.func` spelling at all.**
Measured on gfx950 with a fly-opt built from the pin (LLVM 24):

| `denormal_fpenv` attribute on | `.amdhsa_float_denorm_mode_32` |
|---|---|
| `llvm.func` | **0** — works |
| `gpu.func` | **3** — dropped by `convert-gpu-to-rocdl` |

Upstream reads it through `LLVMFuncOp`'s *inherent* getter
(`ModuleTranslation.cpp:1741`); a discardable attribute on a `gpu.func` never
reaches that slot. Since FlyDSL's `kernel_attrs` land on `gpu.func`, setting the
denorm mode needs a lowering step that puts the inherent attribute on the
`llvm.func` after conversion — there is no kernel-side spelling that does it
today.

⚠ **Check your tool before measuring this.** `build-fly/bin/fly-opt` may be
older than the pin (ours reports LLVM 23, the pin is 24), and LLVM 23 *does*
propagate the `gpu.func` placement. Compare `fly-opt --version` against
`thirdparty/llvm-build-info.json` first, or the result is about the wrong
compiler.

Note `amdgpu-unsafe-fp-atomics` is
**worse than deprecated**: absent from the documented table, carries a
`// TODO: Remove this.` in `SIISelLowering.cpp`, and has zero emission sites in
clang.

## Attributes the compiler infers — do NOT hand-write

- **The `amdgpu-no-*` implicit-arg family** (21 entries in
  `AMDGPUAttributes.def`): `amdgpu-no-workitem-id-x`, `-no-dispatch-ptr`,
  `-no-implicitarg-ptr`, and so on. ⚠ **Violating one is whole-program undefined
  behaviour** — including when the function is merely *reached through* a call
  site carrying the attribute (`AMDGPUUsage.rst:2555-2560`). Never set these to
  chase SGPR savings.
- `amdgpu-memory-bound`, `amdgpu-wave-limiter` — documented as "set internally by
  backend". They *are* user-overridable in practice, but `AMDGPUPerfHint` skips
  inference only when **both** are present, so setting just one does not suppress
  inference of the other.
- `amdgpu-lds-size`, `amdgpu-uniform-work-group-size`, `amdgpu-agpr-alloc` when
  inferred.

## Names that do not exist

Verified absent in the pin — do not use:

| Wrong | Right |
|---|---|
| `amdgpu-no-agpr` | **deleted**; use `amdgpu-agpr-alloc="0"` |
| `amdgpu-color` | `amdgpu-color-export` |
| `--amdgpu-waves-per-eu` as a flag | it is an attribute |
| `--amdgpu-num-vgpr` as a flag | it is an attribute |

⚠ Watch the near-collision: `-amdgpu-num-vgprs-for-wwm-alloc` (plural, a real
cl::opt in `SILowerSGPRSpills.cpp:52`) is unrelated to the singular
`amdgpu-num-vgpr` attribute.

## Verifying

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/llvm_ir_attrs.py $FLYDSL_DUMP_DIR
```

Or read them back from a standalone `llc` run — this prints occupancy and spills
without any assembly parsing:

```bash
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -pass-remarks-analysis=kernel-resource-usage k.ll -o /dev/null
```

⚠ Always confirm against the **emitted** `.vgpr_count` or the remark, never
against the number you requested — see the doubling trap.
