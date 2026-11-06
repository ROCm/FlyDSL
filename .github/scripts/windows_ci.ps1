# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# The steps of .github/workflows/windows.yaml, kept in one script so that a Windows
# box can replay the job:
#
#   powershell -ExecutionPolicy Bypass -File .github\scripts\windows_ci.ps1 -Step <step>
#
# Steps, in order: venv, mlir, build, test, cleanup. They read these variables,
# which the workflow sets once:
#   FCI_WORK        per-job scratch: venv, wheel, home, JIT cache, logs (RUNNER_TEMP)
#   FCI_SHORT       short per-runner root for the LLVM source and build trees
#   FCI_MLIR_CACHE  MLIR installs kept on this runner, one directory per key
#   GPU_ARCH, ROCM_INDEX, ROCM_VERSION, TORCH_VERSION
# The test step also needs HIP_VISIBLE_DEVICES from windows_gpu_gate.py.
param([Parameter(Mandatory = $true)][ValidateSet('venv', 'mlir', 'build', 'test', 'cleanup')][string]$Step)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # .github\scripts -> repo root
. "$repo\scripts\windows\toolchain_windows.ps1"
foreach ($var in 'FCI_WORK', 'FCI_SHORT', 'FCI_MLIR_CACHE') {
    if (-not [Environment]::GetEnvironmentVariable($var)) { throw "$var is not set" }
}
$work = $env:FCI_WORK
$venv = "$work\venv"
$py = "$venv\Scripts\python.exe"

function Set-JobEnv([string]$Name, [string]$Value) {
    # For later steps (GITHUB_ENV, UTF-8 without BOM) and for the rest of this one.
    Set-Item "env:$Name" $Value
    if ($env:GITHUB_ENV) { [IO.File]::AppendAllText($env:GITHUB_ENV, "$Name=$Value`n") }
}

function Get-ClangCl {
    # TheRock's wheel ships clang-cl; the runners' Visual Studio lacks the C++ Clang tools.
    $root = @(Invoke-Native 'rocm-sdk path' { & "$venv\Scripts\rocm-sdk.exe" path --root })[-1]
    return "$root\lib\llvm\bin\clang-cl.exe"
}

function Get-MlirKey([string]$ClangCl) {
    # Everything the Windows MLIR install depends on: the patch series, the build
    # scripts, the compiler, the Python ABI of the bindings and the MSVC headers.
    $files = @('thirdparty\llvm-build-info.json', 'scripts\windows\build_llvm_windows.ps1',
        'scripts\windows\toolchain_windows.ps1')
    $files += Get-ChildItem -Recurse -File "$repo\thirdparty\llvm-extensions", "$repo\thirdparty\llvm-windows" |
        ForEach-Object { $_.FullName.Substring($repo.Length + 1) }
    $lines = [Collections.Generic.List[string]]::new()
    foreach ($f in $files) { $lines.Add("$($f.Replace('\', '/')) $((Get-FileHash -Algorithm SHA256 "$repo\$f").Hash)") }
    $lines.Sort([StringComparer]::Ordinal)
    $lines.Add("clang-cl $(@(Invoke-Native 'clang-cl --version' { & $ClangCl --version })[0])")
    $lines.Add("python $(@(Invoke-Native 'python version' { & $py -c 'import sys; print(sys.version_info[:2])' })[-1])")
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $lines.Add("vs $(& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion)")
    $text = $lines -join "`n"
    Write-Host "MLIR key inputs:`n$text"
    $sha = [Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))
    return (($hash | ForEach-Object { $_.ToString('x2') }) -join '').Substring(0, 16)
}

switch ($Step) {
    'venv' {
        if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
        Invoke-Native 'python -m venv' { python -m venv $venv }
        $arch = $env:GPU_ARCH
        Invoke-Native 'TheRock ROCm and torch' {
            & $py -m pip install --no-cache-dir --index-url $env:ROCM_INDEX "torch[device-$arch]==$env:TORCH_VERSION" `
                "rocm[libraries,devel,device-$arch]==$env:ROCM_VERSION"
        }
        Invoke-Native 'Build and test tools' {
            & $py -m pip install --no-cache-dir nanobind==2.12.0 numpy cmake ninja setuptools wheel pytest pandas
        }
        Invoke-Native 'rocm-sdk init' { & "$venv\Scripts\rocm-sdk.exe" init }
    }
    'mlir' {
        # Built once per key on each runner and kept beside the checkout. LLVM's
        # source and build trees go under the short root: under the ~70-character
        # workspace prefix, rc.exe fails on a 262-character object path (RC1109).
        $env:PATH = "$venv\Scripts;$env:PATH"  # pip's CMake and Ninja
        $clangCl = Get-ClangCl
        $key = Get-MlirKey $clangCl
        $install = "$env:FCI_MLIR_CACHE\$key"
        if (Test-Path "$install\lib\cmake\mlir") {
            Write-Host "Reusing MLIR install $install"
            (Get-Item $install).LastWriteTime = Get-Date
        } else {
            Write-Host "Building MLIR install $install"
            & "$repo\scripts\windows\build_llvm_windows.ps1" -Python $py -SourceDir "$env:FCI_SHORT\s" `
                -BuildDir "$env:FCI_SHORT\b" -InstallDir $install -ClangCl $clangCl
            Remove-Item -Recurse -Force "$env:FCI_SHORT\s", "$env:FCI_SHORT\b" -ErrorAction Continue
        }
        # Keep this key and the most recently used other one.
        Get-ChildItem -Directory $env:FCI_MLIR_CACHE | Where-Object { $_.Name -ne $key } |
            Sort-Object LastWriteTime -Descending | Select-Object -Skip 1 |
            ForEach-Object { Write-Host "Pruning $($_.FullName)"; Remove-Item -Recurse -Force $_.FullName }
        Set-JobEnv 'FCI_MLIR' $install
    }
    'build' {
        # build-win, not build-fly: tests/conftest.py puts build-fly first on sys.path,
        # and the tests must import the installed wheel.
        $build = "$repo\build-win"
        if (Test-Path $build) { Remove-Item -Recurse -Force $build }
        $clangCl = Get-ClangCl
        & "$repo\scripts\windows\build_windows.ps1" -MlirPath $env:FCI_MLIR -BuildDir $build -Python $py -ClangCl $clangCl
        $env:FLY_BUILD_DIR = $build
        $env:FLY_REBUILD = '0'
        Push-Location $repo
        Invoke-Native 'bdist_wheel' { & $py setup.py -q bdist_wheel -d "$work\dist" }
        Pop-Location
        $wheel = @(Get-ChildItem "$work\dist\flydsl-*.whl")
        if ($wheel.Count -ne 1) { throw "Expected one wheel in $work\dist, found $($wheel.Count)" }
        Write-Host "Wheel: $($wheel[0].Name) ($([math]::Round($wheel[0].Length / 1MB)) MB)"
        Invoke-Native 'pip install wheel' { & $py -m pip install --no-deps $wheel[0].FullName }
        # Outside the checkout, flydsl must resolve to the venv.
        Push-Location $work
        $where = @(Invoke-Native 'import flydsl' {
                & $py -c 'import flydsl, flydsl.compiler, flydsl.expr; print(flydsl.__file__)' })[-1]
        Pop-Location
        if (-not $where.StartsWith("$venv\", [StringComparison]::OrdinalIgnoreCase)) {
            throw "flydsl imported from $where, not from the installed wheel"
        }
        Write-Host "flydsl imported from $where"
    }
    'test' {
        if (-not $env:HIP_VISIBLE_DEVICES) { throw 'HIP_VISIBLE_DEVICES is unset: run windows_gpu_gate.py first' }
        # cl.exe on PATH for the AOT tests (test_aot_export compiles a C host program).
        $null = Enter-MsvcEnvironment
        if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'cl.exe is not on PATH' }
        foreach ($d in 'home', 'tmp', 'jit-cache', 'autotune', 'logs') { New-Item -ItemType Directory -Force "$work\$d" | Out-Null }
        # Keep everything the tests write inside this job's directory.
        $env:USERPROFILE = "$work\home"
        $env:TEMP = "$work\tmp"
        $env:TMP = "$work\tmp"
        $env:FLYDSL_RUNTIME_CACHE_DIR = "$work\jit-cache"
        $env:FLYDSL_AUTOTUNE_CACHE_DIR = "$work\autotune"
        $env:FLYDSL_AUTOTUNE = '0'
        $env:FLYDSL_RUN_QUANT = '1'
        $env:PYTHONPATH = $PSScriptRoot  # windows_known_failures, the plugin that skips known failures
        Push-Location $repo
        try {
            # Standalone examples for this arch, as scripts/run_tests.sh selects them.
            $examples = @(Invoke-Native 'Selecting examples' {
                    & $py -c 'import fnmatch, sys; from tests.arch_compat import EXAMPLE_ARCHITECTURES as E; [print(k) for k, v in E.items() if any(fnmatch.fnmatchcase(sys.argv[1], p) for p in v)]' $env:GPU_ARCH
                })
            foreach ($name in $examples) {
                $out = Invoke-Native "example $name" { & $py "examples\$name" 2>&1 | Out-String }
                if ($out -match 'Result correct: False|All passed: False') { throw "example $name reported a wrong result:`n$out" }
                Write-Host "PASS  $name"
            }
            $junit = "$work\logs\pytest.xml"
            Invoke-Native 'pytest' {
                & $py -m pytest tests/kernels tests/language tests/unit tests/system tests/extension tests/python/examples `
                    -q --no-header --tb=short --durations=20 -rfE -p no:cacheprovider `
                    -m 'not multi_gpu and not benchmark and not large_shape' --junitxml $junit -p windows_known_failures
            }
            # A missing compiler would silently skip the AOT tests; make that an error.
            [xml]$report = Get-Content -Raw $junit
            $noCc = @($report.SelectNodes('//testcase/skipped') | Where-Object { $_.message -like '*requires a C compiler*' })
            if ($noCc.Count) { throw "$($noCc.Count) tests skipped for 'requires a C compiler'" }
        } finally {
            Pop-Location
        }
    }
    'cleanup' {
        # The runner ends the processes the job started; this stops what still runs
        # from the job's own trees, then frees the LLVM and FlyDSL build trees.
        $roots = @($work, $env:FCI_SHORT) | ForEach-Object { $_.TrimEnd('\') + '\' }
        Get-CimInstance Win32_Process | Where-Object {
            $exe = $_.ExecutablePath
            $exe -and @($roots | Where-Object { $exe.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count
        } | ForEach-Object {
            Write-Host "Stopping $($_.Name) ($($_.ProcessId))"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        foreach ($dir in $env:FCI_SHORT, "$repo\build-win") {
            if (Test-Path $dir) { Remove-Item -Recurse -Force $dir -ErrorAction Continue }
        }
    }
}
