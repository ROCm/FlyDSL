# SPDX-License-Identifier: Apache-2.0
"""Shared driver for the STRUCTURAL optimization lessons (10, 14, 15, 16, 17, 18-22).

Structural tricks (cooperative LDS, ping-pong, diagonal-pair, column-V, ...) only matter
at real tile/grid sizes, so instead of re-deriving a simplified kernel we DRIVE the real,
validated production kernels in /workspaces/amir/FlyDSL/kernels/ through the unified
benchmark and read the before/after numbers. Each lesson calls compare() with the two
kernel module names that differ by exactly the one trick.

This keeps the "diff" honest: the learner opens the two named kernel files and the change
between them IS the lesson; this script just measures it.
"""

import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCH = os.path.join(_REPO, "tests", "kernels", "bench_fmha_compare.py")


def compare(kernels, seqs=(1024, 16384), ck=False, nwaves=4):
    """Run the unified benchmark over `kernels` (list of module names) at `seqs`."""
    env = dict(os.environ, HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES", "2"),
               FMHA_NWAVES=str(nwaves), PYTHONPATH=os.path.join(_REPO, "..", "aiter"))
    cmd = [sys.executable, _BENCH, "--kernels", *kernels, "--no-pyisa", "--seqs", *map(str, seqs)]
    if ck:
        cmd.append("--ck")
    subprocess.run(cmd, env=env, check=False)
