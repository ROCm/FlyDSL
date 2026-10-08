# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regenerate `fwd_isa_baseline.json`: a fingerprint of the forward's ISA with every optional input folded away.

ABI-07 (`test_folded_features_leave_abi_and_isa`) asserts that a build with ALiBi, sink, split-K, paged and XCD swizzle
all off still compiles to the ISA the forward had before those features existed. This file is that ISA's fingerprint:
the sha256 of the instruction stream (comments and directives stripped, so no temp path leaks in) and the instruction
count and kernarg size, for six builds. It was recorded at the forward-only commit (PR 4 of the port), before any of the
optional inputs was added.

Run from the repository root with the GPU environment of the tests (ROCM_PATH, FLYDSL_GPU_ARCH=gfx950):

    PYTHONPATH=. python3 tests/kernels/attention/data/gen_fwd_isa_baseline.py

Regenerate only when the toolchain moves (the file records the flydsl version it was made with, and the test skips on a
different one) or when a forward change is *meant* to move the folded ISA; say so in the commit message.
"""

import json
import os
import tempfile
from pathlib import Path

import flydsl
from kernels.attention import dispatch
from tests.kernels.attention import isa_tools
from tests.kernels.attention.attn_testlib import compile_inputs, meta_of

CONFIGS = {
    "d64": dict(head_dim=64),
    "d128_window": dict(head_dim=128, window=True),
    "d256": dict(head_dim=256),
    "d384_wide": dict(head_dim=384),
    "d64_bias": dict(head_dim=64, bias=True),
    "d128_dropout": dict(head_dim=128, dropout=True),
}


def compile_fresh(backend, arch, meta, out):
    """Compile a fresh builder with the IR dump on; return its `Dump`."""
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(out), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    fn = backend.build_fwd(meta, backend.fwd_knobs(arch).resolve(meta))
    args, kw = compile_inputs(meta)
    fn.compile(*args, **kw)
    dirs = sorted(p for p in Path(out).iterdir() if p.is_dir())
    return isa_tools.Dump(dirs[-1])


def main():
    arch = dispatch.current_arch()
    backend = dispatch.backend_for(arch)
    golden = {"flydsl": flydsl.__version__, "arch": arch, "configs": CONFIGS, "builds": {}}
    for name, kw in CONFIGS.items():
        with tempfile.TemporaryDirectory() as tmp:
            dump = compile_fresh(backend, arch, meta_of(**kw), tmp)
            sha, n = isa_tools.isa_fingerprint(dump.isa)
            golden["builds"][name] = dict(sha256=sha, instructions=n, kernarg=dump.metadata[".kernarg_segment_size"])
    out = Path(__file__).with_name("fwd_isa_baseline.json")
    out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
