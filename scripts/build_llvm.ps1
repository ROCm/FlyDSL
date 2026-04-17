# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Windows build script for LLVM/MLIR (PowerShell equivalent of build_llvm.sh)

param(
    [int]$Jobs = 0,
    [switch]$NoInstall,
    # GPU arch for ROCM_TEST_CHIPSET. TheRock SDK doesn't ship
    # rocm_agent_enumerator, so MLIR's ROCM runner config needs an explicit arch.
    # Override via -Arch or the FLYDSL_GPU_ARCH env var (defaults to gfx942).
    [string]$Arch = $(if ($env:FLYDSL_GPU_ARCH) { $env:FLYDSL_GPU_ARCH } else { "gfx942" })
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$REPO_ROOT = (Resolve-Path "$SCRIPT_DIR\..").Path
$BASE_DIR = (Resolve-Path "$REPO_ROOT\..").Path

$LLVM_SRC_DIR = "$BASE_DIR\llvm-project"
$LLVM_BUILD_DIR = "$LLVM_SRC_DIR\build-flydsl"
$LLVM_INSTALL_DIR = if ($env:LLVM_INSTALL_DIR) { $env:LLVM_INSTALL_DIR } else { "$LLVM_SRC_DIR\mlir_install" }
$LLVM_PACKAGE_INSTALL = if ($NoInstall) { "0" } else { "1" }

# Read LLVM commit hash
$LLVM_HASH_FILE = "$REPO_ROOT\thirdparty\llvm-hash.txt"
$LLVM_COMMIT = if ($env:LLVM_COMMIT) { $env:LLVM_COMMIT } else { (Get-Content $LLVM_HASH_FILE -Raw).Trim() }

if ($LLVM_COMMIT.Length -lt 40) {
    Write-Error "LLVM_COMMIT must be a full 40-char SHA (got '$LLVM_COMMIT')"
    exit 1
}

Write-Host "Base directory: $BASE_DIR"
Write-Host "LLVM Source:    $LLVM_SRC_DIR"
Write-Host "LLVM Build:     $LLVM_BUILD_DIR"
Write-Host "LLVM Install:   $LLVM_INSTALL_DIR"
Write-Host "LLVM Commit:    $LLVM_COMMIT"

# 1. Clone LLVM
$LLVM_REMOTE = if ($env:LLVM_REMOTE) { $env:LLVM_REMOTE } else { "https://github.com/llvm/llvm-project.git" }

if (-not (Test-Path $LLVM_SRC_DIR)) {
    Write-Host "Fetching llvm-project commit $LLVM_COMMIT (shallow, single commit)..."
    git init $LLVM_SRC_DIR
    Push-Location $LLVM_SRC_DIR
    git remote add origin $LLVM_REMOTE
} else {
    Push-Location $LLVM_SRC_DIR
}

$commitExists = git cat-file -e "${LLVM_COMMIT}^{commit}" 2>$null; $commitFound = $LASTEXITCODE -eq 0
if (-not $commitFound) {
    Write-Host "Fetching commit $LLVM_COMMIT ..."
    git fetch --depth 1 origin $LLVM_COMMIT
}
git checkout $LLVM_COMMIT
Pop-Location

# 2. Create Build Directory
if (-not (Test-Path $LLVM_BUILD_DIR)) {
    New-Item -ItemType Directory -Path $LLVM_BUILD_DIR -Force | Out-Null
}

# 3. Install Python deps
Write-Host "Installing Python dependencies..."
pip install nanobind numpy pybind11

# 4. Detect tools
$PYTHON = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PYTHON) { $PYTHON = (Get-Command python3 -ErrorAction SilentlyContinue).Source }

$GENERATOR = "NMake Makefiles"
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    $GENERATOR = "Ninja"
    Write-Host "Using Ninja generator."
}

$NANOBIND_DIR = & $PYTHON -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')"

# 5. Determine compiler - prefer clang-cl on Windows for LLVM builds
$CC = $null
$CXX = $null
if (Get-Command clang-cl -ErrorAction SilentlyContinue) {
    $CC = "clang-cl"
    $CXX = "clang-cl"
    Write-Host "Using clang-cl compiler."
} elseif (Get-Command cl -ErrorAction SilentlyContinue) {
    Write-Host "Using MSVC cl compiler."
} else {
    Write-Error "No C/C++ compiler found. Install Visual Studio Build Tools or LLVM/Clang."
    exit 1
}

# 6. Configure CMake
Write-Host "Configuring LLVM..."
$cmake_args = @(
    "-G", $GENERATOR,
    "-S", "$LLVM_SRC_DIR\llvm",
    "-B", $LLVM_BUILD_DIR,
    "-DLLVM_ENABLE_PROJECTS=mlir;clang",
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_STANDARD=17",
    "-DCMAKE_CXX_FLAGS=/DMLIR_USE_FALLBACK_TYPE_IDS=1",
    "-DCMAKE_C_FLAGS=/DMLIR_USE_FALLBACK_TYPE_IDS=1",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DLLVM_INSTALL_UTILS=ON",
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
    "-DMLIR_ENABLE_ROCM_RUNNER=ON",
    "-DROCM_TEST_CHIPSET=$Arch",
    "-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir",
    "-DPython3_EXECUTABLE=$PYTHON",
    "-Dnanobind_DIR=$NANOBIND_DIR",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLVM_BUILD_LLVM_DYLIB=OFF",
    "-DLLVM_LINK_LLVM_DYLIB=OFF",
    "-DMLIR_INCLUDE_TESTS=OFF"
)

# Skip compiler-rt on Windows (not needed and may cause issues)
# Don't set RPATH on Windows (not applicable)

if ($CC) {
    $cmake_args += "-DCMAKE_C_COMPILER=$CC"
    $cmake_args += "-DCMAKE_CXX_COMPILER=$CXX"
}

# When using clang-cl, ml64 (MSVC assembler) is typically not available.
# Use llvm-ml with -m64 for 64-bit MASM assembly (BLAKE3 SSE/AVX).
$LLVM_ML = (Get-Command llvm-ml -ErrorAction SilentlyContinue).Source
if (-not (Get-Command ml64 -ErrorAction SilentlyContinue) -and $LLVM_ML) {
    $cmake_args += "-DCMAKE_ASM_MASM_COMPILER=$LLVM_ML"
    $cmake_args += "-DCMAKE_ASM_MASM_FLAGS=-m64"
    Write-Host "Using llvm-ml as MASM assembler (ml64 not found)."
}

cmake @cmake_args
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

# 7. Build
if ($Jobs -eq 0) {
    $Jobs = [Math]::Max(1, [int]($env:NUMBER_OF_PROCESSORS) / 2)
}
Write-Host "Starting build with $Jobs parallel jobs..."
$buildStart = Get-Date
Write-Host "Build started at: $buildStart"
# Ninja prints [N/total] per step; -v adds full compile commands so progress is always visible.
cmake --build $LLVM_BUILD_DIR -j $Jobs -v
if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

# 8. Install
if ($LLVM_PACKAGE_INSTALL -eq "1") {
    Write-Host "=============================================="
    Write-Host "Installing MLIR/LLVM to a clean prefix..."

    if (Test-Path $LLVM_INSTALL_DIR) {
        Remove-Item -Recurse -Force $LLVM_INSTALL_DIR
    }
    New-Item -ItemType Directory -Path $LLVM_INSTALL_DIR -Force | Out-Null

    cmake --install $LLVM_BUILD_DIR --prefix $LLVM_INSTALL_DIR
    if ($LASTEXITCODE -ne 0) { throw "CMake install failed" }

    if (-not (Test-Path "$LLVM_INSTALL_DIR\lib\cmake\mlir")) {
        Write-Error "Install prefix missing lib\cmake\mlir: $LLVM_INSTALL_DIR"
        exit 1
    }
}

Write-Host "=============================================="
Write-Host "LLVM/MLIR build completed successfully!"
Write-Host ""
Write-Host "To build FlyDSL, set:"
Write-Host "  `$env:MLIR_PATH = '$LLVM_INSTALL_DIR'"
Write-Host "Then run:"
Write-Host "  .\scripts\build.ps1"
Write-Host "=============================================="
