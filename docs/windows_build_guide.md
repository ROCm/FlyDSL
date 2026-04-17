# FlyDSL on Windows (experimental)

This guide covers building and running FlyDSL natively on Windows. The Windows
port uses the **TheRock ROCm SDK** (installed as a Python package into a venv)
rather than a system-wide ROCm install.

> **Status**: experimental. 301 / 310 unit tests pass (97%). See
> [Known limitations](#known-limitations) for what doesn't yet.

## Prerequisites

1. **Windows 10/11, x64** with an AMD GPU supported by ROCm. Tested on RDNA4 (`gfx1200`, Radeon RX 9060 XT).
2. **Visual Studio 2022 Build Tools** (or full VS 2022) with the *Desktop
   development with C++* workload — provides the MSVC toolchain.
3. **LLVM/Clang** in PATH — `clang-cl`, `llvm-ar`, `llvm-ml`, `lld-link`. The
   official LLVM Windows installer puts these on PATH.
4. **Ninja** in PATH (`choco install ninja` or bundled with VS).
5. **Git** in PATH.
6. **Python 3.12** (the build has only been exercised against 3.12).
7. **PowerShell 5.1+** (ships with Windows).

All commands below assume you're running in a **Developer PowerShell / x64
Native Tools Command Prompt**, or have run `vcvarsall.bat amd64` first so that
`cl.exe`/`link.exe` are on PATH.

## 1. Create a venv and install TheRock ROCm SDK

TheRock ships ROCm as a Python wheel that installs into your venv. FlyDSL on
Windows currently expects this layout (it finds `ld.lld.exe`, OCML bitcode,
and `amdhip64_7.dll` relative to the SDK root).

```powershell
python -m venv C:\path\to\flydsl-venv
C:\path\to\flydsl-venv\Scripts\Activate.ps1

# Install TheRock ROCm SDK for your GPU family and initialize. For RDNA4 / gfx1200:
pip install --pre --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all torch torchaudio torchvision rocm[libraries,devel]
rocm-sdk init
```

After install, locate the SDK development root — typically:
`...\flydsl-venv\Lib\site-packages\_rocm_sdk_devel`

## 2. Set environment variables

```powershell
$env:ROCM_PATH = "C:\path\to\flydsl-venv\Lib\site-packages\_rocm_sdk_devel"
# Optional: force a specific GPU arch when torch-rocm auto-detect isn't right.
# $env:FLYDSL_GPU_ARCH = "gfx1200"
```

## 3. Build LLVM/MLIR

This pins the commit from `thirdparty/llvm-hash.txt`, builds with Ninja +
`clang-cl`, and installs into `..\llvm-project\mlir_install\`.

```powershell
# From the FlyDSL repo root:
.\scripts\build_llvm.ps1 -Arch gfx1200 # substitute your GPU arch
# or rely on FLYDSL_GPU_ARCH env var; default is gfx942.
```

Notes:
- The script passes `/DMLIR_USE_FALLBACK_TYPE_IDS=1` globally — required to
  make MLIR's TypeIDs work across multiple DLLs on Windows.
- Build takes 30–60 min on a typical workstation. Use `-Jobs N` to cap
  parallelism (default is `NUMBER_OF_PROCESSORS / 2`).
- Output: `..\llvm-project\mlir_install\` — keep this around.

## 4. Build FlyDSL

```powershell
$env:MLIR_PATH = "C:\llvm-project\mlir_install" # or wherever step 3 installed
.\scripts\build.ps1
```

This produces `build-fly\python_packages\flydsl\` with the compiled extension
modules, runtime DLLs, and python sources.

## 5. Install FlyDSL into the venv

```powershell
pip install -e .
```

This runs `setup.py` which creates a directory junction from
`python\flydsl\_mlir` → `build-fly\python_packages\flydsl\_mlir` (junction
instead of symlink so no admin / Developer Mode required).

## 6. Run tests

```powershell
$env:PYTHONPATH = "$PWD\build-fly\python_packages;$PWD"
python -m pytest tests\unit\ -q
```

Expected: ~301 passed / ~4 failed / 5 skipped. See
[Known limitations](#known-limitations).

## How the Windows-specific bits work

| Component | Linux behavior | Windows behavior |
|---|---|---|
| TypeID | `SelfOwningTypeID` (pointer identity across `.so`) | `MLIR_USE_FALLBACK_TYPE_IDS=1` — string-based, works across DLLs |
| Symbol export | `-fvisibility=hidden` + version script | `WINDOWS_EXPORT_ALL_SYMBOLS` on `FlyPythonCAPI.dll`, with `obj.MLIRFlyDialect` / `obj.MLIRFlyROCDLDialect` added as direct sources and upstream `MLIRIR.lib` / `MLIRSupport.lib` extracted via `llvm-ar x` so auto-`.def` generation sees them |
| lld for ROCDL | `<toolkit>/llvm/bin/ld.lld` — matches `/opt/rocm` layout | Staging junction at `%LOCALAPPDATA%\flydsl\rocm_toolkit\` unifies TheRock's `lib/llvm/bin/ld.lld.exe` + `lib/llvm/amdgcn/bitcode/` into the layout MLIR expects |
| Runtime DLL search | `RPATH=$ORIGIN` | `os.add_dll_directory` + ctypes pre-load of `_mlir_libs\*.dll` before JIT engine init (LLVM's `LoadLibraryPermanently` doesn't search DLL-local dirs) |
| GPU arch detect | `rocm_agent_enumerator` | Falls back to `torch.cuda.get_device_properties(0).gcnArchName` since TheRock doesn't ship the enumerator |
| `_mlir` package link | Symlink | Directory junction (no admin needed) |

## Known limitations

- **Multi-stream correctness**: 2 `test_multi_stream_launch` tests fail
  (`test_two_streams_independent`, `test_diamond_pipeline_with_event_sync`).
  Single-stream launches work correctly.
- **Disk cache test fragility**: `test_fp_math_reaches_pipeline` passes solo
  but fails in-suite because cached compilation artifacts bypass the monkey-
  patched hook. Not Windows-specific. Run with `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
- **Torch profiler test**: `test_cache_disabled_run_perftest_does_not_crash`
  hits a `DataFrame.host_time_sum` attribute error inside torch.profiler —
  version compat, not a FlyDSL issue.
- **No CI coverage** yet — every build is verified manually.
- **Only gfx1200 exercised** on Windows; other arches should work
  if your TheRock SDK + GPU combination is supported.

## Troubleshooting

- **`MLIR_FOUND=FALSE` or `LLVMNVPTXCodeGen` missing at configure time**:
  the ROCm SDK ships its own `LLVMConfig.cmake`. Ensure `-DLLVM_DIR=...` is
  passed explicitly (the scripts do this). Don't add `_rocm_sdk_devel` to
  `CMAKE_PREFIX_PATH`.
- **`_ITERATOR_DEBUG_LEVEL` link mismatch**: you're building FlyDSL as Debug
  while MLIR was built Release. Keep both at Release (`build.ps1` passes
  `-DCMAKE_BUILD_TYPE=Release`).
- **`ModuleNotFoundError: flydsl._mlir`**: the editable install junction
  didn't get created. `cd` into the repo and run:
  `New-Item -ItemType Junction -Path python\flydsl\_mlir -Target build-fly\python_packages\flydsl\_mlir`.
- **`hipErrorNoBinaryForGpu`**: your kernel was compiled for the wrong arch.
  Set `FLYDSL_GPU_ARCH` to the value printed by
  `python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"`.
- **`rocm amdgcn bitcode path ... does not exist`** during compilation:
  the toolkit staging junction failed. Check
  `%LOCALAPPDATA%\flydsl\rocm_toolkit\amdgcn\bitcode\` and that `ROCM_PATH`
  points at TheRock's `_rocm_sdk_devel`.
