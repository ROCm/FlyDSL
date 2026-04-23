# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Windows build script for FlyDSL (PowerShell equivalent of build.sh)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$REPO_ROOT = (Resolve-Path "$SCRIPT_DIR\..").Path

# ---------------------------------------------------------------------------
# Build directory (default: build-fly\, overridable via FLY_BUILD_DIR)
# ---------------------------------------------------------------------------
$BUILD_DIR = if ($env:FLY_BUILD_DIR) { $env:FLY_BUILD_DIR } else { "$REPO_ROOT\build-fly" }
if (-not [System.IO.Path]::IsPathRooted($BUILD_DIR)) {
    $BUILD_DIR = "$REPO_ROOT\$BUILD_DIR"
}

# ---------------------------------------------------------------------------
# Parallelism: default to processor count, overridable via -jN argument
# ---------------------------------------------------------------------------
$PARALLEL_JOBS = $env:NUMBER_OF_PROCESSORS
foreach ($arg in $args) {
    if ($arg -match '^-j(\d+)$') {
        $PARALLEL_JOBS = $Matches[1]
    }
}

# ---------------------------------------------------------------------------
# Discover MLIR_PATH
# ---------------------------------------------------------------------------
if (-not $env:MLIR_PATH) {
    $BASE_DIR = (Resolve-Path "$REPO_ROOT\..").Path
    $candidates = @(
        "$BASE_DIR\llvm-project-flydsl\build-flydsl\mlir_install",
        "$BASE_DIR\llvm-project\build-flydsl\mlir_install",
        "$BASE_DIR\llvm-project\mlir_install"
    )
    foreach ($p in $candidates) {
        if (Test-Path "$p\lib\cmake\mlir") {
            Write-Host "Auto-detected MLIR_PATH: $p"
            $env:MLIR_PATH = $p
            break
        }
    }
}

if (-not $env:MLIR_PATH) {
    Write-Error "MLIR_PATH not set and could not be auto-detected. Build LLVM/MLIR first or set `$env:MLIR_PATH to your mlir_install path."
    exit 1
}

# ---------------------------------------------------------------------------
# CMake generator: prefer Ninja, fall back to Visual Studio
# ---------------------------------------------------------------------------
$GENERATOR = $null
$CMAKE_CACHE = "$BUILD_DIR\CMakeCache.txt"
if (Test-Path $CMAKE_CACHE) {
    $cached = Select-String -Path $CMAKE_CACHE -Pattern '^CMAKE_GENERATOR:INTERNAL=(.+)$'
    if ($cached) {
        $GENERATOR = $cached.Matches[0].Groups[1].Value
    }
}
if (-not $GENERATOR) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $GENERATOR = "Ninja"
    } else {
        $GENERATOR = "NMake Makefiles"
    }
}

Write-Host "=============================================="
Write-Host "FlyDSL Build (Windows)"
Write-Host "  REPO_ROOT:  $REPO_ROOT"
Write-Host "  BUILD_DIR:  $BUILD_DIR"
Write-Host "  MLIR_PATH:  $($env:MLIR_PATH)"
Write-Host "  PARALLEL:   -j$PARALLEL_JOBS"
Write-Host "  GENERATOR:  $GENERATOR"
Write-Host "=============================================="

# ---------------------------------------------------------------------------
# Initialize git submodules if needed
# ---------------------------------------------------------------------------
if (-not (Test-Path "$REPO_ROOT\thirdparty\dlpack\include\dlpack\dlpack.h")) {
    Write-Host "Initializing git submodules..."
    git -C $REPO_ROOT submodule update --init --recursive
}

# ---------------------------------------------------------------------------
# Remove editable-install symlink/junction that would cause conflicts
# ---------------------------------------------------------------------------
$EDITABLE_MLIR_LINK = "$REPO_ROOT\python\flydsl\_mlir"
if (Test-Path $EDITABLE_MLIR_LINK) {
    $item = Get-Item $EDITABLE_MLIR_LINK -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        Write-Host "Removing editable-install symlink/junction: $EDITABLE_MLIR_LINK"
        cmd /c rmdir $EDITABLE_MLIR_LINK
    }
}

# ---------------------------------------------------------------------------
# Discover Python and nanobind
# ---------------------------------------------------------------------------
$PYTHON = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PYTHON) {
    $PYTHON = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $PYTHON) {
    Write-Error "Python not found in PATH"
    exit 1
}

$NANOBIND_DIR = $null
try {
    $NANOBIND_DIR = & $PYTHON -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')" 2>$null
} catch {}

# ---------------------------------------------------------------------------
# CMake configure
# ---------------------------------------------------------------------------
if (-not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null
}

# Detect compiler
$CC = $null
if (Get-Command clang-cl -ErrorAction SilentlyContinue) {
    $CC = "clang-cl"
} elseif (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Warning "No C/C++ compiler found. Install Visual Studio Build Tools or LLVM/Clang."
}

$cmake_args = @(
    "-G", $GENERATOR,
    $REPO_ROOT,
    "-DMLIR_DIR=$($env:MLIR_PATH)\lib\cmake\mlir",
    "-DLLVM_DIR=$($env:MLIR_PATH)\lib\cmake\llvm",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DPython3_EXECUTABLE=$PYTHON"
)

if ($CC) {
    $cmake_args += "-DCMAKE_C_COMPILER=$CC"
    $cmake_args += "-DCMAKE_CXX_COMPILER=$CC"
}
if ($NANOBIND_DIR) {
    $cmake_args += "-Dnanobind_DIR=$NANOBIND_DIR"
}

# Add HIP/ROCm paths for Windows (TheRock)
if ($env:ROCM_PATH) {
    $cmake_args += "-DCMAKE_PREFIX_PATH=$($env:ROCM_PATH)"
}
if ($env:HIP_PATH) {
    $cmake_args += "-Dhip_DIR=$($env:HIP_PATH)\lib\cmake\hip"
}

Write-Host "Configuring CMake..."
Push-Location $BUILD_DIR
try {
    cmake @cmake_args
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

    # ---------------------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------------------
    Write-Host "Building with -j$PARALLEL_JOBS..."
    cmake --build . -j $PARALLEL_JOBS
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
$PYTHON_PKG_DIR = "$BUILD_DIR\python_packages"

Write-Host ""
Write-Host "=============================================="
Write-Host "Build complete!"
Write-Host ""
Write-Host "Usage (no install):"
Write-Host "  `$env:PYTHONPATH = '$PYTHON_PKG_DIR;' + `$env:PYTHONPATH"
Write-Host ""
Write-Host "Or install as editable package:"
Write-Host "  cd $REPO_ROOT; pip install -e ."
Write-Host "=============================================="
