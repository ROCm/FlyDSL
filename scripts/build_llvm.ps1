# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

param(
    [string]$LLVMSourceDir = 'C:\llvmfdsl',
    [string]$LLVMBuildDir = 'C:\lbuild',
    [string]$LLVMInstallDir = 'C:\linstall',
    [string]$Python = 'C:\fdsl-venv\Scripts\python.exe',
    [int]$Jobs = 32
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildInfo = Get-Content (Join-Path $repoRoot 'thirdparty\llvm-build-info.json') -Raw | ConvertFrom-Json
$llvmCommit = $buildInfo.upstream.llvm_hash
$patches = @(
    (Join-Path $repoRoot 'thirdparty\llvm-rocdl-lld-argv0.patch'),
    (Join-Path $repoRoot 'thirdparty\llvm-windows-nanobind-ehrtti.patch')
)

if (-not (Test-Path -LiteralPath $LLVMSourceDir)) {
    New-Item -ItemType Directory -Path $LLVMSourceDir | Out-Null
    git -C $LLVMSourceDir init
    if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
    git -C $LLVMSourceDir remote add origin https://github.com/llvm/llvm-project.git
    if ($LASTEXITCODE -ne 0) { throw 'git remote add failed' }
} elseif (-not (Test-Path -LiteralPath (Join-Path $LLVMSourceDir '.git'))) {
    throw "LLVM source path exists but is not a Git checkout: $LLVMSourceDir"
}

$origin = git -C $LLVMSourceDir remote get-url origin
if ($LASTEXITCODE -ne 0 -or $origin -ne 'https://github.com/llvm/llvm-project.git') {
    throw "Unexpected LLVM remote at $LLVMSourceDir : $origin"
}
git -C $LLVMSourceDir fetch --depth 1 origin $llvmCommit
if ($LASTEXITCODE -ne 0) { throw 'Fetching the pinned LLVM revision failed' }
git -C $LLVMSourceDir checkout --detach $llvmCommit
if ($LASTEXITCODE -ne 0) { throw 'Checking out the pinned LLVM revision failed' }
foreach ($patch in $patches) {
    git -C $LLVMSourceDir apply --unidiff-zero --check $patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        git -C $LLVMSourceDir apply --unidiff-zero $patch
        if ($LASTEXITCODE -ne 0) { throw "Applying the FlyDSL LLVM patch failed: $patch" }
    } else {
        git -C $LLVMSourceDir apply --unidiff-zero --reverse --check $patch 2>$null
        if ($LASTEXITCODE -ne 0) { throw "Pinned LLVM checkout has unexpected changes around patch: $patch" }
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $vsPath) { throw 'Visual Studio C++ build tools were not found' }
$vsDevCmd = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
$envLines = & $env:ComSpec /d /s /c "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
if ($LASTEXITCODE -ne 0) { throw 'VsDevCmd failed to initialize the compiler environment' }
foreach ($line in $envLines) {
    $equals = $line.IndexOf('=')
    if ($equals -gt 0) { [Environment]::SetEnvironmentVariable($line.Substring(0, $equals), $line.Substring($equals + 1), 'Process') }
}

& $Python -m pip install 'nanobind==2.12.0' numpy pybind11
if ($LASTEXITCODE -ne 0) { throw 'Installing isolated LLVM Python build dependencies failed' }
$nanobindDir = (& $Python -c "import nanobind, pathlib; print(pathlib.Path(nanobind.__file__).parent / 'cmake')").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not locate nanobind CMake files' }

cmake -U Python3_EXECUTABLE -U nanobind_DIR -U MLIR_USE_FALLBACK_TYPE_IDS `
    -G Ninja -S (Join-Path $LLVMSourceDir 'llvm') -B $LLVMBuildDir `
    "-DCMAKE_C_COMPILER=cl.exe" `
    "-DCMAKE_CXX_COMPILER=cl.exe" `
    "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld" `
    "-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU" `
    -DLLVM_ENABLE_RUNTIMES= `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_STANDARD=17 `
    "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /DMLIR_USE_FALLBACK_TYPE_IDS=1" `
    -DLLVM_ENABLE_ASSERTIONS=ON `
    -DLLVM_INSTALL_UTILS=ON `
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON `
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir `
    "-DPython3_EXECUTABLE=$Python" `
    "-Dnanobind_DIR:PATH=$nanobindDir" `
    -DBUILD_SHARED_LIBS=OFF `
    -DLLVM_BUILD_LLVM_DYLIB=OFF `
    -DLLVM_LINK_LLVM_DYLIB=OFF `
    -DLLVM_INCLUDE_TESTS=OFF `
    -DMLIR_INCLUDE_TESTS=OFF `
    -DHIP_PLATFORM=amd
if ($LASTEXITCODE -ne 0) { throw 'LLVM/MLIR CMake configuration failed' }

cmake --build $LLVMBuildDir --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'LLVM/MLIR build failed' }
cmake --install $LLVMBuildDir --prefix $LLVMInstallDir --config Release
if ($LASTEXITCODE -ne 0) { throw 'LLVM/MLIR install failed' }
if (-not (Test-Path -LiteralPath (Join-Path $LLVMInstallDir 'lib\cmake\mlir'))) {
    throw "MLIR install tree is missing lib\cmake\mlir: $LLVMInstallDir"
}

Write-Output "LLVM source: $LLVMSourceDir @ $llvmCommit"
Write-Output "MLIR install: $LLVMInstallDir"
