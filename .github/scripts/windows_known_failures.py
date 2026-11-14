# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Pytest plugin for the Windows CI job: skip the known Windows test failures.

Each entry gives its reason, with the PR that fixes it where one is pending;
delete it when the fix lands. Keys are paths under tests/ (pytest's rootdir), as
pytest prints node IDs: an exact node ID covers all its parametrizations, and a
module path the whole module. Entries that match nothing are listed at the end of
the run, so a renamed or fixed test does not stay hidden.
"""

PATHS = "Windows path escaping in the location assertion (raw __file__ vs escaped backslashes)"
PROFILER = "torch.profiler records no GPU events without Kineto, so run_perftest gets an empty frame"
DETECTION = "mocks the Linux rocm_agent_enumerator/rocminfo path; Windows detects the GPU through torch and HIP"

KNOWN_FAILURES = {
    "unit/test_capture_user_location.py::test_each_capture_records_its_own_line_and_column": PATHS,
    "unit/test_capture_user_location.py::test_identical_code_in_two_files_keeps_each_file": PATHS,
    "unit/test_capture_user_location.py::test_kernel_frame_is_kept_at_the_depth_limit": PATHS,
    "unit/test_compile_hints.py::TestCacheDisabledRegression::test_cache_disabled_run_perftest_does_not_crash": PROFILER,
    "kernels/test_layernorm.py::test_layernorm": PROFILER,
    "kernels/test_layernorm.py::test_layernorm_dynamicquant": PROFILER,
    "kernels/test_layernorm.py::test_layernorm_smoothquant": PROFILER,
    "kernels/test_layernorm.py::test_fused_add_layernorm": PROFILER,
    "kernels/test_layernorm.py::test_fused_add_layernorm_dynamicquant": PROFILER,
    "kernels/test_layernorm.py::test_fused_add_layernorm_smoothquant": PROFILER,
    "kernels/test_rmsnorm.py::test_rmsnorm": PROFILER,
    "kernels/test_rmsnorm.py::test_rmsnorm_dynamicquant": PROFILER,
    "kernels/test_rmsnorm.py::test_rmsnorm_smoothquant": PROFILER,
    "kernels/test_rmsnorm.py::test_fused_add_rmsnorm": PROFILER,
    "kernels/test_rmsnorm.py::test_fused_add_rmsnorm_dynamicquant": PROFILER,
    "kernels/test_rmsnorm.py::test_fused_add_rmsnorm_smoothquant": PROFILER,
    "kernels/test_rdna_gemm.py::test_f16_gemm_benchmark": PROFILER,
    "kernels/test_softmax.py::test_all": PROFILER,
    "kernels/test_softmax_bwd.py::test_all": PROFILER,
    "kernels/test_topk_gating_softmax.py::test_all": PROFILER,
    "unit/test_device.py::test_arch_detection_ignores_hidden_heterogeneous_gpu": DETECTION,
    "unit/test_device.py::test_hardware_detection_falls_back_to_rocminfo": DETECTION,
    "unit/test_device.py::test_hardware_detection_does_not_default_to_gfx942": DETECTION,
    "unit/test_device.py::test_device_count_counts_gpus_not_supported_isas": DETECTION,
    "unit/test_device.py::test_device_count_remains_best_effort": DETECTION,
}

_used = set()


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for item in items:
        nodeid = item.nodeid
        keys = (nodeid, nodeid.split("[", 1)[0], nodeid.split("::", 1)[0])
        hit = next((k for k in keys if k in KNOWN_FAILURES), None)
        if hit:
            _used.add(hit)
            drop.append(item)
        else:
            keep.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_terminal_summary(terminalreporter):
    unused = sorted(set(KNOWN_FAILURES) - _used)
    terminalreporter.write_line(f"Windows known failures: {len(_used)} entries used, {len(unused)} unused")
    for key in unused:
        terminalreporter.write_line(f"  unused (fixed or renamed?): {key}")
