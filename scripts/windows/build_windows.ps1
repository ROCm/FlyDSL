# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Windows counterpart of build.sh: configure and build FlyDSL with clang-cl and
# Ninja against an MLIR install from build_llvm_windows.ps1, taking HIP from the
# ROCm SDK installed in the given Python (TheRock's rocm[libraries,devel]) or
# from ROCM_PATH.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\build_windows.ps1 [-MlirPath <mlir_install>]
#       [-BuildDir <dir>] [-Python <python.exe>] [-ClangCl <clang-cl.exe>] [-Jobs <n>]
#
# MlirPath, BuildDir and Python default to MLIR_PATH, FLY_BUILD_DIR (else
# build-fly) and PYTHON_EXECUTABLE (else python on PATH), like build.sh.
param(
    [string]$MlirPath = $env:MLIR_PATH,
    [string]$BuildDir = $env:FLY_BUILD_DIR,
    [string]$Python = $env:PYTHON_EXECUTABLE,
    [string]$ClangCl,
    [int]$Jobs = [Environment]::ProcessorCount
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\toolchain_windows.ps1"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # scripts\windows -> repo root
if (-not $Python) { $Python = 'python' }
if (-not $BuildDir) { $BuildDir = 'build-fly' }
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not $MlirPath) {
    $base = Split-Path -Parent $repo
    $MlirPath = @('llvm-project-flydsl\build-flydsl\mlir_install', 'llvm-project\build-flydsl\mlir_install',
        'llvm-project\mlir_install') | ForEach-Object { "$base\$_" } |
        Where-Object { Test-Path "$_\lib\cmake\mlir" } | Select-Object -First 1
}
if (-not $MlirPath -or -not (Test-Path "$MlirPath\lib\cmake\mlir")) {
    throw 'No MLIR install found: run scripts\windows\build_llvm_windows.ps1 first, or pass -MlirPath <mlir_install>'
}
if (-not (Test-Path "$repo\thirdparty\dlpack\include\dlpack\dlpack.h")) {
    Invoke-Native 'git submodule update' { git -C $repo submodule update --init --recursive }
}

$vs = Enter-MsvcEnvironment
$ClangCl = Resolve-ClangCl $ClangCl $vs
$Python = Resolve-Python $Python
$nanobindDir = Get-NanobindDir $Python

# lib/Runtime/ROCm/CMakeLists.txt finds HIP through ROCM_PATH or `rocm-sdk path
# --root`; put this interpreter's rocm-sdk first on PATH. Its HIP CMake package
# exists only once the devel archive is expanded, which `rocm-sdk init` does once.
$scripts = @(Invoke-Native 'Locating Python scripts' {
        & $Python -c "import sysconfig; print(sysconfig.get_path('scripts'))" })[-1]
$env:PATH = "$scripts;$env:PATH"
if (-not $env:ROCM_PATH) {
    if (-not (Test-Path "$scripts\rocm-sdk.exe")) {
        throw ("No ROCm SDK in $Python. Install TheRock's rocm[libraries,devel] into it " +
            '(pip install --index-url <TheRock index for your GPU> "rocm[libraries,devel]"), or set ROCM_PATH')
    }
    Invoke-Native 'rocm-sdk init (expands the ROCm devel SDK)' { & "$scripts\rocm-sdk.exe" init }
}

Set-TypeIdFlags $BuildDir
$cmakeArgs = @(
    '-G', 'Ninja', '-S', $repo, '-B', $BuildDir,
    "-DCMAKE_C_COMPILER=$ClangCl", "-DCMAKE_CXX_COMPILER=$ClangCl",
    "-DMLIR_DIR=$MlirPath\lib\cmake\mlir",
    "-DPython3_EXECUTABLE=$Python", "-DPython_EXECUTABLE=$Python", "-Dnanobind_DIR=$nanobindDir",
    '-DHIP_PLATFORM=amd',
    # Release like the LLVM install. CMake defaults clang-cl builds to Debug, whose
    # debug C++ runtime fails lld-link's /failifmismatch against it; RelWithDebInfo
    # (/Ob1) makes FlyPythonCAPI.dll 241 MB instead of 175 MB.
    '-DCMAKE_BUILD_TYPE=Release'
)
Invoke-Native 'FlyDSL configure' { cmake @cmakeArgs }
Invoke-Native 'FlyDSL build' { cmake --build $BuildDir -j $Jobs }

Write-Host ''
Write-Host 'Build complete. Use it without installing:'
Write-Host "  `$env:PYTHONPATH = '$BuildDir\python_packages;$repo'"
Write-Host 'or build a wheel (the build dir must be inside the checkout):'
Write-Host "  `$env:FLY_BUILD_DIR = '$BuildDir'; $Python setup.py bdist_wheel"
