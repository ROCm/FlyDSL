# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Helpers dot-sourced by build_llvm_windows.ps1 and build_windows.ps1: run
# native tools so that failures stop the build, enter the MSVC x64 environment,
# and pick clang-cl and the Python interpreter. Both scripts must work under
# Windows PowerShell 5.1.

function Invoke-Native([string]$What, [scriptblock]$Command, [switch]$AllowFailure) {
    # Windows PowerShell 5.1 turns redirected native stderr (git and cmake
    # progress) into error records, which 'Stop' would make fatal. Judge native
    # tools by their exit code instead, read right after the call. A command that
    # cannot start (not found, not executable) only reports an error and leaves
    # $LASTEXITCODE as the previous command set it, so clear it first: still
    # unset afterwards means the tool never ran.
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = $null
    & $Command
    if ($null -eq $LASTEXITCODE) { throw "$What could not be run" }
    if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) { throw "$What failed with exit code $LASTEXITCODE" }
}

function Test-Native([string]$What, [scriptblock]$Command) {
    Invoke-Native $What $Command -AllowFailure 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Enter-MsvcEnvironment {
    # clang-cl finds the MSVC and Windows SDK headers and libraries through the
    # environment VsDevCmd sets up. Returns the Visual Studio installation path.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw 'vswhere.exe not found: install Visual Studio 2022 or its Build Tools with "Desktop development with C++"'
    }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'No Visual Studio installation has the MSVC x64 build tools' }
    Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64 -no_logo' | Out-Null
    return $vs
}

function Resolve-ClangCl([string]$ClangCl, [string]$VsPath) {
    # The ClangCL toolset, unless -ClangCl names a standalone LLVM. The compiler
    # is passed to CMake by full path: PATH may hold other compilers (MinGW gcc).
    if (-not $ClangCl) { $ClangCl = "$VsPath\VC\Tools\Llvm\x64\bin\clang-cl.exe" }
    if (-not (Test-Path $ClangCl)) {
        throw ("clang-cl not found at $ClangCl. Install the Visual Studio component " +
            '"C++ Clang tools for Windows" (Microsoft.VisualStudio.Component.VC.Llvm.Clang), ' +
            'or pass -ClangCl with the path of a standalone clang-cl.exe')
    }
    return (Resolve-Path $ClangCl).Path
}

function Resolve-Python([string]$Python) {
    # The full path of the interpreter that will run FlyDSL. MLIR detects Python
    # twice (Python3_ and, for nanobind, Python_), so the callers pass it to both
    # -DPython3_EXECUTABLE and -DPython_EXECUTABLE.
    $exe = Invoke-Native "Python ($Python)" {
        & $Python -c "import sys; assert sys.version_info >= (3, 10) and sys.maxsize > 2**32, '64-bit Python 3.10+ required'; print(sys.executable)"
    }
    return @($exe)[-1]
}

function Get-NanobindDir([string]$Python) {
    $dir = Invoke-Native 'Locating nanobind (pip install nanobind)' {
        & $Python -c 'import nanobind; print(nanobind.cmake_dir())'
    }
    return @($dir)[-1]
}

function Set-TypeIdFlags([string]$BuildDir) {
    # Implicit MLIR TypeIDs are function-local statics, one copy per DLL, so a type
    # made in FlyPythonCAPI.dll would fail isa<> in an extension. With
    # MLIR_USE_FALLBACK_TYPE_IDS=1 they resolve by name through MLIRSupport. It
    # changes inline code, so every LLVM and FlyDSL translation unit needs it.
    # -DCMAKE_CXX_FLAGS would replace clang-cl's defaults (/DWIN32 /D_WINDOWS
    # /EHsc /GR); CXXFLAGS adds to them, but only when the cache is created.
    $cache = "$BuildDir\CMakeCache.txt"
    if ((Test-Path $cache) -and
        -not (Select-String -Path $cache -Pattern '^CMAKE_CXX_FLAGS:STRING=.*MLIR_USE_FALLBACK_TYPE_IDS=1' -Quiet)) {
        throw "$BuildDir was configured without MLIR_USE_FALLBACK_TYPE_IDS=1; delete it and rerun"
    }
    $env:CXXFLAGS = "/DMLIR_USE_FALLBACK_TYPE_IDS=1 $env:CXXFLAGS".Trim()
}
