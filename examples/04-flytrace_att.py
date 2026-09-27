# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Collect Example04 flytrace + rocprofv3 ATT, then export a merged Perfetto JSON."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workload(run):
    import importlib.util

    import torch

    import flydsl.compiler as flyc
    from flydsl.extension import flytrace
    from tests.utils import shuffle_weight

    spec = importlib.util.spec_from_file_location("att_gemm", ROOT / "examples/04-flytrace_gemm.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    torch.manual_seed(0)
    a = torch.randn(demo.M, demo.K, dtype=torch.float16, device="cuda")
    b = torch.randn(demo.N, demo.K, dtype=torch.float16, device="cuda")
    c = torch.empty((demo.M, demo.N), dtype=torch.float16, device="cuda")
    shuffled = shuffle_weight(b, layout=(16, 16))
    ta = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=1, divisibility=16)
    tc = flyc.from_dlpack(c).mark_layout_dynamic(leading_dim=1, divisibility=16)
    args = (ta, shuffled, tc, torch.cuda.current_stream())
    # Capture the full grid: the scheduler decides which CTA reaches the ATT CU.
    with flytrace.capture(block=None, hardware=True, exclude=("mainloop", "drain")) as cap:
        fn = flyc.compile(demo.preshuffle_gemm, *args)  # compiles AND launches the warmup
        fn(*args)  # ATT selects matching invocation 2; flytrace keeps this launch
    expected = (a @ b.T).float()
    assert torch.isfinite(c).all()
    assert ((c.float() - expected).abs() <= 1e-3 + 1e-3 * expected.abs()).all()
    print("GEMM correctness: PASS", flush=True)
    print("flytrace:", cap.save(run / "flytrace.raw.json"), flush=True)
    (run / "device.json").write_text(
        json.dumps(
            dict(
                name=torch.cuda.get_device_name(),
                arch=torch.cuda.get_device_properties(0).gcnArchName,
                torch=torch.__version__,
                hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New collection directory (must not already exist)")
    parser.add_argument("--workload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--collect-only", action="store_true", help="Keep raw data without merging")
    parser.add_argument("--max-blocks", type=int, default=1, help="Blocks in the merged view; 0 means all")
    args = parser.parse_args()
    if args.workload:
        workload(args.workload)
        return
    if args.max_blocks < 0:
        parser.error("--max-blocks must be nonnegative")
    if args.output:
        run = args.output.resolve()
        run.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT / "results/flytrace-att"
        parent.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    config = dict(
        jobs=[
            dict(
                output_directory=str(run / "att"),
                output_file="out",
                output_format=["csv"],
                advanced_thread_trace=True,
                kernel_trace=True,
                truncate_kernels=True,
                kernel_include_regex="^gemm_kernel.*$",
                kernel_iteration_range="2",
                att_library_path=[os.environ.get("ATT_LIBRARY_PATH", "/opt/rocm/lib")],
                att_target_cu=1,
                att_shader_engine_mask="0x1",
                att_simd_select="0xf",
                att_buffer_size="0xC000000",
            )
        ]
    )
    (run / "input.json").write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{ROOT}/build-fly/python_packages:{ROOT}:" + env.get("PYTHONPATH", ""),
        FLYDSL_BUILD_DIR=str(ROOT / "build-fly"),
        FLYDSL_RUNTIME_ENABLE_CACHE="0",
        FLYDSL_DEBUG_ENABLE_DEBUG_INFO="1",
        FLYDSL_DUMP_IR="1",
        FLYDSL_DEBUG_PRINT_ORIGIN_IR="0",
        FLYDSL_DUMP_DIR=str(run / "ir"),
        FLYDSL_DEBUG_LOG_TO_FILE=str(run / "flydsl.log"),
    )
    cmd = [
        os.environ.get("ROCPROFV3", "/opt/rocm/bin/rocprofv3"),
        "-i",
        str(run / "input.json"),
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--workload",
        str(run),
    ]
    print("Collecting:", run, flush=True)
    with (run / "profile.log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not args.collect_only:
        from flydsl.extension.flytrace import merge_att

        dispatches = list((run / "att").glob("ui_output_agent_*"))
        if len(dispatches) != 1:
            raise RuntimeError(f"Expected one ATT dispatch, found {len(dispatches)}")
        summary = merge_att(run / "flytrace.raw.json", dispatches[0], run / "merged.json", max_blocks=args.max_blocks)
        print(json.dumps(summary, indent=2))
    print("Result:", run)


if __name__ == "__main__":
    main()
