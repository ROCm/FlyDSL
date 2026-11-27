# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm JIT runtime line-buffers the C stdout stream.

The HIP runtime writes device ``fx.printf`` output to the C ``stdout`` stream,
which is fully buffered when stdout is a pipe (Jupyter, redirected logs). The
output then stays in the buffer until the process exits. Loading the JIT runtime
library switches the stream to line buffering.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.l1b_target_dialect,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(sys.platform == "win32", reason="the MSVC CRT has no line-buffered mode"),
]

# The child leaves through os._exit, which skips the exit-time flush of the C
# streams: the line reaches the pipe only if the stream wrote it out by itself.
_CHILD = textwrap.dedent("""
    import ctypes
    import os

    from flydsl.compiler.jit_executor import _resolve_runtime_libs

    ctypes.CDLL(_resolve_runtime_libs()[0])
    ctypes.CDLL(None).puts(b"line from the C stdout stream")
    os._exit(0)
    """)


def test_piped_c_stdout_is_line_buffered():
    result = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "line from the C stdout stream" in result.stdout
