# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Pytest plugin for the Windows CI job: skip the known Windows test failures.

Each entry gives its reason, with the PR that fixes it where one is pending;
delete it when the fix lands. Keys are paths under tests/ (pytest's rootdir), as
pytest prints node IDs: an exact node ID covers all its parametrizations, and a
module path the whole module. Entries that match nothing are listed at the end of
the run, so a renamed or fixed test does not stay hidden.
"""

TRITON = "imports triton, which has no Windows wheel; fixed by #1379"
STREAMS = "uses inputs from the current stream on a new stream without a wait; fixed by #1378"
PATHS = "Windows path escaping in the location assertion (raw __file__ vs escaped backslashes)"
PROFILER = "torch.profiler records no GPU events without Kineto, so run_perftest gets an empty frame"

# Modules that cannot even be collected.
IGNORED_MODULES = {
    "kernels/test_fp4_gemm_4wave.py": TRITON,
    "kernels/test_gemm_fp8fp4_gfx1250.py": TRITON,
    "kernels/test_mega_moe_tp.py": TRITON,
    "kernels/test_moe_a8w4_mxscale_gfx1250.py": TRITON,
    "kernels/test_mxfp8_gemm_8wave.py": TRITON,
    "kernels/test_pa.py": TRITON,
    "kernels/test_preshuffle_gemm.py": TRITON,
}

KNOWN_FAILURES = {
    # Fails on Windows at random. Listed are the tests that #1378 fixes, as whole
    # modules where the failing tests vary from run to run.
    "unit/test_multi_stream_launch.py": STREAMS,
    "unit/test_jit_stream_param.py": STREAMS,
    "unit/test_universal_atomic.py": STREAMS,
    "unit/test_pointer_argument_vec_add.py": STREAMS,
    "kernels/test_vec_add.py::test_benchmark_vector_add": STREAMS,
    "kernels/test_softmax_autotune.py::test_default_serves_without_searching_on_the_current_stream": STREAMS,
    "kernels/test_softmax_autotune.py::test_compiled_cache_hit_bypasses_resolution_and_accepts_new_buffers_and_stream": STREAMS,
    "extension/random/test_random.py": STREAMS,
    "extension/coop/test_block_reduce.py": STREAMS,
    "extension/coop/test_block_scan.py": STREAMS,
    "extension/coop/test_warp_reduce.py": STREAMS,
    "extension/coop/test_warp_reduce_batched.py": STREAMS,
    "extension/coop/test_warp_rocdl.py": STREAMS,
    "extension/coop/test_warp_scan.py": STREAMS,
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
}

_used = set()


def pytest_ignore_collect(collection_path, config):
    if not collection_path.is_relative_to(config.rootpath):
        return None
    rel = collection_path.relative_to(config.rootpath).as_posix()
    if rel in IGNORED_MODULES:
        _used.add(rel)
        return True
    return None


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
    unused = sorted((set(IGNORED_MODULES) | set(KNOWN_FAILURES)) - _used)
    terminalreporter.write_line(f"Windows known failures: {len(_used)} entries used, {len(unused)} unused")
    for key in unused:
        terminalreporter.write_line(f"  unused (fixed or renamed?): {key}")
