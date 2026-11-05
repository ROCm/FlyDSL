# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Windows counterpart of build_llvm.sh, profile amd-minimal: LLVM/MLIR with the
# Python bindings, built by clang-cl and Ninja. clang-cl because upstream MLIR
# already gives it the EH/RTTI flags its nanobind library needs (cl.exe would
# need another LLVM patch); Ninja because it makes the obj.<lib> object
# libraries FlyDSL's Windows build links from.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\build_llvm_windows.ps1 [-Python <python.exe>]
#       [-SourceDir <dir>] [-BuildDir <dir>] [-InstallDir <dir>] [-ClangCl <clang-cl.exe>] [-Jobs <n>]
#
# Defaults follow build_llvm.sh: ..\llvm-project, its build-flydsl and
# mlir_install. LLVM's tree is deep, so keep these paths short.
param(
    [string]$Python = $env:PYTHON_EXECUTABLE,
    [string]$SourceDir,
    [string]$BuildDir,
    [string]$InstallDir,
    [string]$ClangCl,
    [int]$Jobs = [Environment]::ProcessorCount
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\toolchain_windows.ps1"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # scripts\windows -> repo root
if (-not $Python) { $Python = 'python' }
if (-not $SourceDir) { $SourceDir = Join-Path (Split-Path -Parent $repo) 'llvm-project' }
if (-not $BuildDir) { $BuildDir = Join-Path $SourceDir 'build-flydsl' }
if (-not $InstallDir) { $InstallDir = Join-Path $SourceDir 'mlir_install' }

# The series build_llvm.sh applies (required_patches, then extensions, in listed
# order), then the Windows-only patches, which no Linux build or CI key reads.
$pin = (Get-Content -Raw "$repo\thirdparty\llvm-build-info.json" | ConvertFrom-Json).upstream
$patches = @(@($pin.required_patches) + @($pin.extensions) | Where-Object { $_ } |
    ForEach-Object { "$repo\thirdparty\llvm-extensions\$($_.file)" })
$patches += @(Get-ChildItem "$repo\thirdparty\llvm-windows" -Filter *.patch | Sort-Object Name |
    ForEach-Object { $_.FullName })

$vs = Enter-MsvcEnvironment
$ClangCl = Resolve-ClangCl $ClangCl $vs
$Python = Resolve-Python $Python
Invoke-Native 'Installing nanobind' { & $Python -m pip install 'nanobind==2.12.0' numpy }
$nanobindDir = Get-NanobindDir $Python

Write-Host "LLVM pin $($pin.llvm_hash) in $SourceDir"
if (-not (Test-Path "$SourceDir\.git")) {
    Invoke-Native 'git init' { git init --quiet $SourceDir }
    Invoke-Native 'git remote add' { git -C $SourceDir remote add origin https://github.com/llvm/llvm-project.git }
}
# Git for Windows defaults to autocrlf=true, but the patches and LLVM's files are
# LF. A checkout written as CRLF keeps its bytes when only the setting changes,
# so empty its index once: the forced checkout below then rewrites every file.
$autocrlf = Invoke-Native 'git config' { git -C $SourceDir config --get core.autocrlf } -AllowFailure
if ($autocrlf -ne 'false') {
    Invoke-Native 'git config' { git -C $SourceDir config core.autocrlf false }
    Invoke-Native 'git read-tree' { git -C $SourceDir read-tree --empty }
}
Invoke-Native 'git config' { git -C $SourceDir config core.longpaths true }  # LLVM paths exceed MAX_PATH
if (-not (Test-Native 'git cat-file' { git -C $SourceDir cat-file -e "$($pin.llvm_hash)^{commit}" })) {
    Invoke-Native 'git fetch' { git -C $SourceDir fetch --depth 1 origin $pin.llvm_hash }
}
# Replay the series on a pristine pin every run (see build_llvm.sh): the forced
# checkout also drops files an earlier --index apply created. Never git clean:
# the default build and install dirs live in this checkout.
Invoke-Native 'git checkout' { git -C $SourceDir checkout --quiet --force $pin.llvm_hash }
foreach ($patch in $patches) {
    $name = Split-Path -Leaf $patch
    Write-Host "  applying $name"
    Invoke-Native "Applying LLVM patch $name" { git -C $SourceDir apply --index $patch }
}

Set-TypeIdFlags $BuildDir
$cmakeArgs = @(
    '-G', 'Ninja', '-S', "$SourceDir\llvm", '-B', $BuildDir,
    "-DCMAKE_C_COMPILER=$ClangCl", "-DCMAKE_CXX_COMPILER=$ClangCl",
    '-DLLVM_ENABLE_PROJECTS=mlir', '-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU', '-DLLVM_ENABLE_RUNTIMES=',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_STANDARD=17',
    '-DLLVM_ENABLE_ASSERTIONS=ON', '-DLLVM_INSTALL_UTILS=ON',
    '-DMLIR_ENABLE_BINDINGS_PYTHON=ON', '-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir',
    "-DPython3_EXECUTABLE=$Python", "-DPython_EXECUTABLE=$Python", "-Dnanobind_DIR=$nanobindDir",
    '-DBUILD_SHARED_LIBS=OFF', '-DLLVM_BUILD_LLVM_DYLIB=OFF', '-DLLVM_LINK_LLVM_DYLIB=OFF',
    '-DMLIR_INCLUDE_TESTS=OFF'
)
Invoke-Native 'LLVM configure' { cmake @cmakeArgs }
Invoke-Native 'LLVM build' { cmake --build $BuildDir -j $Jobs }
# Install next to the old prefix and swap only once the new one is complete, so a
# failed run never leaves the previous install deleted. The install is relocatable.
$staging = "$InstallDir.new"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
Invoke-Native 'LLVM install' { cmake --install $BuildDir --prefix $staging }
if (-not (Test-Path "$staging\lib\cmake\mlir")) { throw "Install prefix lacks lib\cmake\mlir: $staging" }
if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir }
Move-Item $staging $InstallDir

Write-Host "LLVM/MLIR installed to $InstallDir"
Write-Host "Build FlyDSL with: powershell -ExecutionPolicy Bypass -File scripts\windows\build_windows.ps1 -MlirPath $InstallDir"
