# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Make the intranode dispatch/combine accuracy gate reachable from pytest.

``tests/kernels/test_profiler_dispatch_combine.py`` carries the real correctness
machinery for ``kernels/comm/flydsl_dispatch_combine_intranode_*`` -- ``verify_self``,
``_verify_dispatch_self_consistency``, the per-rank allclose comparisons against the
mori reference -- and exits non-zero when a case fails. But it defines no
``def test_`` function, so pytest collects zero items from it, and the only thing
that runs it is ``.github/workflows/flydsl.yaml``'s label-gated 8-GPU job, through
``--ci-sweep``.

That leaves the op with no coverage at any smaller scale: ``pytest -m multi_gpu`` on
a 2- or 4-GPU box exercises shmem and allreduce but never dispatch/combine, and vLLM
reaches this path through MegaMoE's all2all.

This wrapper runs the existing runner at world_size=2 on a small shape and asserts it
passes. Both ``--mode`` values embed the accuracy check at the top of every run, so a
numerical regression fails here without duplicating any of the verification logic.
The 8-GPU sweep stays where it is; this is the small-scale gate under it.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parent / "test_profiler_dispatch_combine.py"

# Small enough to stay well under the multi-GPU job's budget while still routing
# every token across both ranks (k=2, 4 experts per rank).
_CASE = [
    "--mode",
    "bench",
    "--world-size",
    "2",
    "--max-tokens",
    "128",
    "--hidden-dim",
    "1024",
    "--num-experts-per-rank",
    "4",
    "--k",
    "2",
    "--port",
    "29571",
]


def _skip_unless_two_gpu_dispatch_combine():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mori.shmem", reason="dispatch/combine runner imports mori.shmem")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"needs 2 GPUs, found {torch.cuda.device_count()}")
    assert _RUNNER.is_file(), f"dispatch/combine runner not found at {_RUNNER}"


@pytest.mark.multi_gpu
def test_dispatch_combine_2gpu_accuracy():
    """Run the intranode dispatch/combine accuracy gate across two ranks."""
    _skip_unless_two_gpu_dispatch_combine()

    env = dict(os.environ)
    # The runner rendezvouses over TCP; loopback keeps it off whatever NIC the
    # host happens to expose first.
    env.setdefault("MORI_SOCKET_IFNAME", "lo")

    completed = subprocess.run(
        [sys.executable, str(_RUNNER), *_CASE],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, f"dispatch/combine runner exited {completed.returncode}\n{output[-4000:]}"
    # Guard against the runner changing shape and exiting 0 without checking
    # anything: the global verdict line must be present.
    assert "ALL PASS" in output, f"runner did not report a global pass verdict\n{output[-4000:]}"
