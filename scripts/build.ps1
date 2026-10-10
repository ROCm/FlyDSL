# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

param(
    [string]$MLIRPath = 'C:\linstall',
    [string]$BuildDir = 'C:\fbuild',
    [string]$Python = 'C:\fdsl-venv\Scripts\python.exe',
    [int]$Jobs = 32
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$requiredSubmodules = @('thirdparty/dlpack', 'thirdparty/tvm-ffi')
git -C $repoRoot submodule update --init -- $requiredSubmodules
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize required pinned submodules' }
$submoduleStatus = git -C $repoRoot submodule status -- $requiredSubmodules
$uninitializedSubmodules = @($submoduleStatus | Where-Object { $_ -match '^[-+U]' })
if ($LASTEXITCODE -ne 0 -or $uninitializedSubmodules.Count -gt 0) {
    throw 'Required pinned submodules are not initialized at their recorded commits'
}
$sitePackages = (& $Python -c "import pathlib, sys; print(next(str(pathlib.Path(p)) for p in sys.path if (pathlib.Path(p) / '_rocm_sdk_devel').is_dir()))").Trim()
if ($LASTEXITCODE -ne 0 -or -not $sitePackages) { throw 'Could not find the active Python ROCm SDK site-packages' }
$sdkDevel = Join-Path $sitePackages '_rocm_sdk_devel'
$sdkCore = Join-Path $sitePackages '_rocm_sdk_core'
if (-not (Test-Path -LiteralPath (Join-Path $MLIRPath 'lib\cmake\mlir\MLIRConfig.cmake'))) {
    throw "MLIRConfig.cmake not found under $MLIRPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $sdkDevel 'lib\cmake\hip\hip-config.cmake'))) {
    throw "HIP CMake package not found in the active Python SDK: $sdkDevel"
}

$env:ROCM_PATH = $sdkDevel
$env:HIP_PATH = $sdkDevel
$env:HIP_PLATFORM = 'amd'
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
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
$nanobindDir = (& $Python -c "import nanobind, pathlib; print(pathlib.Path(nanobind.__file__).parent / 'cmake')").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not locate nanobind CMake files' }

cmake -G 'Visual Studio 17 2022' -A x64 -T ClangCL -S $repoRoot -B $BuildDir `
    "-DMLIR_DIR:PATH=$(Join-Path $MLIRPath 'lib\cmake\mlir')" `
    "-DLLVM_DIR:PATH=$(Join-Path $MLIRPath 'lib\cmake\llvm')" `
    "-DPython3_EXECUTABLE=$Python" `
    "-Dnanobind_DIR:PATH=$nanobindDir" `
    -DHIP_PLATFORM=amd `
    "-DCMAKE_PREFIX_PATH=$MLIRPath;$sdkDevel;$sdkCore"
if ($LASTEXITCODE -ne 0) { throw 'FlyDSL CMake configuration failed' }
cmake --build $BuildDir --parallel $Jobs --config Release
if ($LASTEXITCODE -ne 0) { throw 'FlyDSL build failed' }

Write-Output "Build tree: $BuildDir"
Write-Output "Python package path: $(Join-Path $BuildDir 'python_packages')"
