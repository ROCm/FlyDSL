#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Multi-agent orchestrator that ports a HIP kernel to FlyDSL.

Built directly on the Anthropic Messages API (`import anthropic`). Each role is
its own agentic tool-use loop with a tailored system prompt and tool set; the
Python `orchestrate()` loop is the deterministic controller around them.

Pipeline (looped until accepted or max iterations is hit):

    1. Analyzer  - reads the HIP source + relevant FlyDSL docs/examples and
                   writes a concrete, LLVM-1:1-aligned implementation plan.
    2. Implementer - reads the plan + HIP source and writes/edits the FlyDSL
                   kernel so its lowered IR aligns 1:1 with the HIP kernel.
    2b. Test author (only if no --test-file and GPU mode) - generates a pytest
                   that uses the HIP kernel as the ground-truth reference and
                   compares it against the FlyDSL port. Runs once per session.
    3. Evaluator - verifies the port. With a GPU it runs the pytest numerical
                   check and a benchmark vs. the HIP baseline; with no GPU it
                   falls back to compile + LLVM/MLIR structural alignment.
                   Emits a structured JSON verdict.

If the evaluator rejects the attempt, its `feedback` is fed back into the next
analyze -> implement -> evaluate iteration.

    pip install anthropic
    export AMD_LLM_GATEWAY_KEY=...   # AMD LLM gateway key

The whole request is a single natural-language prompt; an agent parses it into the
structured config (HIP source/URL, kernel name, output dir, remote GPU, include
dirs, iterations, tolerances, ...):

    python scripts/port_hip_to_flydsl_agent.py "Port the gemm1_a4w4 kernel from \
        https://github.com/org/repo/blob/<sha>/path/gemm1_a4w4.cuh into \
        ports/gemm1_a4w4, include dirs csrc/kernels/mxfp4_moe,csrc/include, \
        test on remote host gpu01 as user fsx in container flydsl_ctr (FlyDSL at \
        /workspace/FlyDSL), 3 iterations, rtol 1e-3"

All artifacts land under the output dir:
    <output-dir>/<kernel_name>.py   - the FlyDSL kernel
    <output-dir>/plan.md            - the analyzer's plan
    <output-dir>/performance.md     - perf/accuracy/trace/IR records per iteration
    <output-dir>/ir_dumps/iter<N>/  - exported FlyDSL + HIP LLVM IR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("anthropic is required. Install it with:\n    pip install anthropic")


# max_tokens is a hard per-response cap required by the Messages API; effort still
# governs how much of it the model actually spends. Generous so xhigh/max agents
# have room to think + act across tool-use turns.
MAX_TOKENS = 32000


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    hip_source: Path  # the HIP kernel file (after any fetch/clone)
    hip_root: Path  # dependency root: clone root / download dir / source's dir
    kernel_name: str
    output: Path
    test_file: Path | None
    repo_root: Path
    plan_file: Path
    model: str
    max_iterations: int
    max_turns: int  # tool-use rounds per agent
    accuracy_rtol: float
    perf_ratio: float  # FlyDSL latency must be <= perf_ratio * HIP latency
    eval_mode: str  # "gpu" or "compile_ir"
    bash_timeout: int
    compile_arch: str  # ARCH for the local COMPILE_ONLY gate (e.g. gfx950)
    compile_attempts: int  # implementer retries to pass the local compile gate
    perf_doc: Path
    trace_skill: Path
    analysis_skill: Path
    authoring_skill: Path
    ir_dir: Path
    extra_context: str = ""
    hip_include_dirs: list[str] = field(default_factory=list)  # extra -I dirs for HIP compile
    hip_build_cmd: str = ""  # optional explicit command to build HIP IR / reference
    remote_ctx: str = ""  # prompt block telling agents to run GPU work on a remote host
    test_reference: str = ""  # reference harness the TEST AUTHOR builds the unit test from

    analyzer_tools: list[str] = field(
        default_factory=lambda: ["read_file", "list_dir", "grep", "run_bash", "write_file"]
    )
    implementer_tools: list[str] = field(
        default_factory=lambda: ["read_file", "list_dir", "grep", "run_bash", "write_file", "edit_file"]
    )
    test_author_tools: list[str] = field(
        default_factory=lambda: ["read_file", "list_dir", "grep", "run_bash", "write_file", "edit_file"]
    )
    evaluator_tools: list[str] = field(
        default_factory=lambda: ["read_file", "list_dir", "grep", "run_bash", "append_file", "write_file"]
    )


# --------------------------------------------------------------------------- #
# Tool definitions (Anthropic tool-use schema) + local executors
# --------------------------------------------------------------------------- #
MAX_TOOL_OUTPUT = 12000  # chars; keep tool_result from blowing the context window

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "name": "read_file",
        "description": "Read a UTF-8 text file. Returns content with 1-based line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (absolute or repo-relative)."},
                "offset": {"type": "integer", "description": "1-based start line (optional)."},
                "limit": {"type": "integer", "description": "Max lines to read (optional)."},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "append_file": {
        "name": "append_file",
        "description": "Append text to a file, creating it (with parent dirs) if missing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Replace an exact substring in a file. old_string must be unique unless replace_all is true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    "run_bash": {
        "name": "run_bash",
        "description": "Run a bash command from the repo root. Returns combined stdout/stderr and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds (optional)."},
            },
            "required": ["command"],
        },
    },
    "list_dir": {
        "name": "list_dir",
        "description": "List entries of a directory (non-recursive). Defaults to repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    "grep": {
        "name": "grep",
        "description": "Search files for a regex (ripgrep if available, else grep -r). Returns matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "File or dir to search (optional, default repo root)."},
            },
            "required": ["pattern"],
        },
    },
}


def _resolve(cfg: Config, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (cfg.repo_root / p)


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    head = text[: MAX_TOOL_OUTPUT // 2]
    tail = text[-MAX_TOOL_OUTPUT // 2 :]
    return f"{head}\n... [truncated {len(text) - MAX_TOOL_OUTPUT} chars] ...\n{tail}"


def execute_tool(cfg: Config, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool locally. Returns (result_text, is_error)."""
    try:
        if name == "read_file":
            path = _resolve(cfg, args["path"])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, args.get("offset", 1))
            end = start - 1 + args["limit"] if args.get("limit") else len(lines)
            body = "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(lines) if start <= i + 1 <= end)
            return _truncate(body or "(empty / no lines in range)"), False

        if name == "write_file":
            path = _resolve(cfg, args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return f"Wrote {len(args['content'])} bytes to {path}", False

        if name == "append_file":
            path = _resolve(cfg, args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(args["content"])
            return f"Appended {len(args['content'])} bytes to {path}", False

        if name == "edit_file":
            path = _resolve(cfg, args["path"])
            text = path.read_text(encoding="utf-8")
            old, new = args["old_string"], args["new_string"]
            count = text.count(old)
            if count == 0:
                return f"old_string not found in {path}", True
            if count > 1 and not args.get("replace_all"):
                return f"old_string is not unique ({count} matches); set replace_all or add context.", True
            text = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
            path.write_text(text, encoding="utf-8")
            return f"Edited {path} ({count if args.get('replace_all') else 1} replacement(s))", False

        if name == "run_bash":
            timeout = args.get("timeout", cfg.bash_timeout)
            proc = subprocess.run(
                args["command"],
                shell=True,
                cwd=str(cfg.repo_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = f"$ {args['command']}\n[exit {proc.returncode}]\n{proc.stdout}{proc.stderr}"
            return _truncate(out), proc.returncode != 0

        if name == "list_dir":
            path = _resolve(cfg, args.get("path", "."))
            entries = sorted(f"{e.name}/" if e.is_dir() else e.name for e in path.iterdir())
            return _truncate("\n".join(entries) or "(empty)"), False

        if name == "grep":
            target = str(_resolve(cfg, args["path"])) if args.get("path") else str(cfg.repo_root)
            if shutil.which("rg"):
                cmd = ["rg", "-n", "--no-heading", args["pattern"], target]
            else:
                cmd = ["grep", "-rn", args["pattern"], target]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.bash_timeout)
            return _truncate(proc.stdout or "(no matches)"), False

        return f"Unknown tool: {name}", True
    except subprocess.TimeoutExpired:
        return f"Tool '{name}' timed out", True
    except Exception as exc:  # surface the error back to the model
        return f"Tool '{name}' error: {type(exc).__name__}: {exc}", True


# --------------------------------------------------------------------------- #
# Shared project knowledge + role prompts
# --------------------------------------------------------------------------- #
SHARED_CONTEXT = """\
You are working inside the FlyDSL repository (a Python DSL + MLIR compiler stack
that lowers GPU kernels to ROCDL/HSACO for AMD GPUs). Use the provided tools
(read_file, write_file, edit_file, run_bash, list_dir, grep) to inspect and
modify the repo. Key facts:

- New kernels use the layout API: `fx.rocdl.make_buffer_tensor()`, logical layout
  ops, and `fx.copy_atom_call`. Raw byte offsets / `create_buffer_resource()` are
  legacy. Use `@flyc.kernel` for device kernels and `@flyc.jit` for launchers.
- Compile-time unrolled loops: `range_constexpr`. Loop-carried `scf.for`:
  `range(start, stop, step, init=[...])`. Keep a single explicit exit path; do
  not define a value only inside one branch and use it after.
- Arch helpers: `from flydsl.runtime.device import get_rocm_arch, is_rdna_arch`.
  Shared wave-size logic in `kernels/kernels_common.py`. gfx942/gfx950 = wave 64
  MFMA; gfx120*/gfx1250 = wave 32 WMMA.
- IR dump: `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>`. Disable JIT disk cache while
  iterating: `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Compile w/o run: `COMPILE_ONLY=1`.
- Tests live in `tests/kernels/` (pytest, need GPU). Look at `examples/` and
  `docs/kernel_authoring_guide.md` for idioms. Read CLAUDE.md for conventions.

PORTING PRINCIPLE - LLVM 1:1 ALIGNMENT:
The FlyDSL port must lower to LLVM/ROCDL IR that aligns 1:1 with what the HIP
kernel compiles to: same memory access pattern (loads/stores, buffer resources,
vectorization width), same MMA/MFMA/WMMA atom selection, same LDS usage and
barriers, same loop structure and unroll factors, same math intrinsics. Do not
rewrite cleverly - mirror the HIP kernel's instruction-level shape using FlyDSL's
layout algebra. Prefer the construct whose lowering matches the HIP instruction.
"""

ANALYZER_PROMPT = SHARED_CONTEXT + """
ROLE: ANALYZER (read-only except writing the plan file).

Produce a concrete plan to port the HIP kernel to FlyDSL with 1:1 LLVM alignment:
  1. Read the HIP source in full. The kernel may live in a larger repo with
     dependencies: the task gives you the HIP dependency root - FOLLOW the
     kernel's `#include`s within that root and read all transitively-included
     PROJECT files (helpers, headers, traits, config). Skip system/toolchain
     headers. Identify grid/block dims, per-thread work, memory access pattern,
     LDS usage, MMA/MFMA/WMMA usage, vectorization, loop structure/unrolling,
     math intrinsics, data types, target arch.
  2. Read 1-2 closest FlyDSL reference kernels in `kernels/` and relevant
     `examples/` to find the matching FlyDSL constructs.
  3. Map each HIP construct to the specific FlyDSL API that lowers to the SAME
     LLVM/ROCDL instruction(s). Call out hard-to-align spots and how to verify
     them via IR dumps.
  4. Specify exact `@flyc.kernel`/`@flyc.jit` signatures, layouts, copy atoms,
     MMA atoms, and loop forms.

REFINEMENT ITERATIONS (when a previous attempt was evaluated):
  The task gives you the performance doc, the latest captured trace directory,
  and the exported FlyDSL + HIP LLVM IR paths. Before revising the plan, do BOTH:

  IR-DIFF ANALYSIS (correctness/alignment of the lowering):
    i.  Diff the FlyDSL LLVM IR against the HIP LLVM IR (e.g. `diff`, or read both
        and compare). Focus on STRUCTURAL divergence, not SSA names/order: buffer
        load/store presence and access widths, vectorization, MMA/MFMA/WMMA
        intrinsic selection, LDS usage and barriers, loop/unroll structure, math
        intrinsics. Identify each spot where the FlyDSL lowering departs from the
        HIP instruction shape - this is where 1:1 alignment broke.

  TRACE / BOTTLENECK ANALYSIS (runtime cost) using kernel-trace-analysis:
    ii. READ the skill file at the path given in the task and FOLLOW it. In
        practice: run its `scripts/hotspot_analyzer.py` on the latest trace dir
        (the skill also accepts `--dir <ui_output_agent_*_dispatch_*>`) to get the
        top-K stall hotspots (VMEM-load/wait, LDS/SMEM-wait, barrier, MFMA stalls)
        mapped back to source lines, plus occupancy. If memory-bound, also use the
        PMC analyzer path as that skill describes.

  RECONCILE & UPDATE:
    iii. Correlate the two - an IR divergence frequently explains a trace
         hotspot. Translate the findings into concrete plan changes (wider/
         coalesced loads, double-buffer/prefetch, different copy/MMA atom, unroll
         factor, LDS layout to kill bank conflicts, fewer barriers) that BOTH
         restore LLVM 1:1 alignment vs. HIP and remove the bottleneck.
    iv. UPDATE the plan file in place, and add a short "Iteration N analysis"
        section summarizing (1) the key IR divergences and (2) the top trace
        hotspots, with the specific fix chosen for each, so the implementer knows
        exactly what to change.
  Also fold in the evaluator's failure feedback from the task.

Do NOT write the kernel. Write ONLY the plan to the given plan file path, then
stop. The plan must let another engineer implement without re-reading the HIP
source line by line.
"""

IMPLEMENTER_PROMPT = SHARED_CONTEXT + """
ROLE: IMPLEMENTER.

Read the plan file and the HIP source, then write/edit the FlyDSL kernel at the
given output path:
  - FIRST read the flydsl-kernel-authoring skill file at the path given in the
    task and FOLLOW its conventions, idioms, and API guidance when writing the
    kernel (layout API, @flyc.kernel/@flyc.jit structure, copy/MMA atoms, loops).
  - Follow the plan. Mirror the HIP kernel's instruction-level shape (LLVM 1:1
    alignment). Make only the changes needed.
  - Match FlyDSL style (black line length 120; `flydsl` is first-party for isort).
    Use the layout API, not legacy byte offsets.
  - PITFALL (common port bug): do NOT define a value only inside one `if`/`else`
    branch and use it after - with `range_constexpr`/`scf` the branch condition is
    often NOT a plain Python bool, so the body may trace symbolically and the var
    is undefined later (`NameError`). Hoist the definition or merge to one value.
  - MANDATORY LOCAL COMPILE GATE: the kernel MUST trace+compile locally before it
    goes to evaluation. Write a small smoke harness to the smoke path given in the
    task that imports your kernel module and INVOKES the `@flyc.jit` launcher with
    representative dummy torch tensors (correct shapes/dtypes) - importing alone
    does NOT trigger tracing. Run it locally with
    `COMPILE_ONLY=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 ARCH=<arch from task> python3 <smoke>`
    and FIX every trace/compile error until it exits 0. Do not stub/fake to pass.
    The orchestrator re-runs this exact gate; if it fails the kernel comes back to
    you - so make it genuinely pass.

End with a one-paragraph summary of what you implemented and any alignment risks.
"""

TEST_AUTHOR_PROMPT = SHARED_CONTEXT + """
ROLE: TEST AUTHOR.

No test file was provided. Create a pytest numerical-correctness test that uses
the HIP kernel as the ground-truth reference and compares it against the FlyDSL
port. Steps:
  1. Read the HIP source and the FlyDSL kernel to learn the exact entry point
     (function name, launch wrapper, argument shapes/dtypes) of BOTH sides.
  2. Look at existing `tests/kernels/*.py` to copy this repo's conventions for
     pytest markers (e.g. `l2_device`, `rocm_lower`), fixtures, tensor setup,
     and how device kernels are launched. Match them.
  3. Build the HIP kernel as the reference. Prefer the repo's existing mechanism
     if one exists; otherwise compile the HIP source to a shared lib with
     `hipcc -shared -fPIC --offload-arch=<arch>` and call it via ctypes, or use
     torch's HIP cpp_extension. Generate representative random inputs (cover the
     dtype and a couple of shapes), get the reference output from HIP, run the
     FlyDSL kernel on the same inputs, and assert closeness with
     `torch.testing.assert_close` (or numpy allclose) at rtol ~ the target
     tolerance given in the task. Seed the RNG for determinism.
  4. Write the test to the path given in the task. Then RUN it once with
     `FLYDSL_RUNTIME_ENABLE_CACHE=0` to confirm it executes (it is fine if it
     currently FAILS on numerical mismatch - that is the evaluator's job - but
     it must be importable, collectable, and actually exercise both kernels;
     fix any import/compile/setup errors in the test harness itself).

REFERENCE HARNESS (if a test-reference path is given in the task):
  READ that file and BUILD the unit test from it instead of inventing inputs:
    - Reuse its real input/weight/quantization/shuffle-layout construction
      (shapes, dtypes, fp4/e8m0 quant, the mxfp4 preshuffle layouts) so the unit
      test exercises the SAME data layout the kernel sees in production.
    - The golden for the gemm1 unit test is the HIP/aiter gemm1 kernel (the same
      kernel family this port targets): drive it with those reference-derived
      inputs to produce the ground-truth output, then compare the FlyDSL gemm1
      port against it. Isolate gemm1 (do NOT compare the whole fused MoE).
    - Keep it a focused pytest (parametrized over a couple of M/shapes), repo
      conventions as above.

Do not weaken the comparison to force a pass, and do not edit the FlyDSL kernel
or the HIP source - only author the test file. End with a one-line note of the
test path and whether it ran (pass/fail/collected).
"""


def evaluator_prompt(cfg: Config) -> str:
    inc_flags = " ".join(f"-I{d}" for d in cfg.hip_include_dirs)
    hip_build = (
        f"Use this exact build command for the HIP IR: `{cfg.hip_build_cmd}`."
        if cfg.hip_build_cmd
        else (
            "Emit the real DEVICE LLVM IR, not host IR, and from a TU that actually "
            "INSTANTIATES the kernel. Compiling the .cuh header alone yields an EMPTY "
            "device module (the `launch<...>` template is never instantiated). So:\n"
            "       - Compile the test-harness/reference TU that instantiates and calls the "
            "kernel (e.g. the .cu you built for the HIP reference), with device-only emit:\n"
            f"         `hipcc --cuda-device-only -S -emit-llvm --offload-arch=<arch> "
            f"-I{cfg.hip_root} {inc_flags} <ref_or_instantiation.cu> -o <out>/hip.ll`\n"
            "       - If no such TU exists, write a tiny one that `#include`s the kernel and "
            "explicitly instantiates the exact template params used by the FlyDSL port.\n"
            "       - Verify hip.ll is non-empty and contains the device function / mfma / "
            "buffer intrinsics (e.g. grep for `amdgcn` / `mfma` / `llvm.amdgcn.raw.buffer`). "
            "Add repo include dirs/flags as needed. Record the exact command used."
        )
    )
    if cfg.eval_mode == "gpu":
        mode_block = f"""
EVALUATION MODE: GPU (numerical + performance + trace).
  1. Accuracy: run the pytest test file with `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
     Passes if the test passes (relative tolerance ~{cfg.accuracy_rtol:g}).
  2. Performance: benchmark FlyDSL vs. the HIP baseline (test/benchmark harness
     or scripts/run_benchmark.sh). Passes if
     flydsl_latency <= {cfg.perf_ratio:g} * hip_latency.
  3. Trace: capture a GPU kernel trace of the FlyDSL kernel using the
     capture-kernel-trace skill. READ the skill file at `{cfg.trace_skill}` and
     FOLLOW its rocprofv3 workflow exactly: run it on the test/benchmark script
     ({cfg.test_file or "the FlyDSL kernel's bench/test entry"}) with a
     kernel_include_regex derived from `{cfg.kernel_name}`, set
     FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1, and collect the ui_output_agent_* trace +
     out_kernel_trace.csv. Record the trace location, kernel duration, VGPR/SGPR
     counts, instruction count and source-mapping %, per that skill's Output
     section. Trace capture is NOT a gating axis, but it MUST be attempted and
     its result (success / failure reason) recorded.
  4. LLVM IR export (always, even on pass): dump BOTH LLVM IRs to
     `{cfg.ir_dir}/iter<N>/` and record their absolute paths.
       - FlyDSL: `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR={cfg.ir_dir}/iter<N>` while
         compiling/running the kernel; save the LLVM/ROCDL stage as
         `{cfg.ir_dir}/iter<N>/flydsl.ll`.
       - HIP: build to `{cfg.ir_dir}/iter<N>/hip.ll`. {hip_build}
     Note any structural divergence (buffer load/store widths, MMA/MFMA/WMMA
     intrinsics, LDS/barrier, loop/unroll, math intrinsics).
  5. If accuracy/perf fail, use the trace + the IR diff to locate the divergence.
Gating axes: accuracy AND performance (trace and IR export are recorded, not gating).
"""
    else:
        mode_block = f"""
EVALUATION MODE: COMPILE + IR ALIGNMENT (no GPU).
  1. Compile with `COMPILE_ONLY=1 FLYDSL_RUNTIME_ENABLE_CACHE=0`. Must be clean.
  2. Dump BOTH LLVM IRs to `{cfg.ir_dir}/iter<N>/` and record their absolute paths:
       - FlyDSL: `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR={cfg.ir_dir}/iter<N>`; save the
         LLVM/ROCDL stage as `{cfg.ir_dir}/iter<N>/flydsl.ll`.
       - HIP: build to `{cfg.ir_dir}/iter<N>/hip.ll`. {hip_build}
     Compare STRUCTURALLY: buffer loads/stores and widths, MMA/MFMA/WMMA
     intrinsics, LDS/barrier usage, loop/unroll structure, math intrinsics. SSA
     names/order may differ; instruction shape must match.
Gating axes: compile AND ir_alignment (record compile result in ir_alignment.details).
"""
    return SHARED_CONTEXT + mode_block + """
ROLE: EVALUATOR (read-only for source; you MAY write only the performance doc).

Be skeptical and concrete: never report a pass you did not actually observe. If a
step is skipped/unrunnable, say so and treat that axis as NOT passed. Do not edit
the FlyDSL kernel, the HIP source, or the test - the only file you may write is
the performance doc.

RECORD RESULTS: Before emitting the verdict, APPEND a section to the performance
doc at the path given in the task (create it with a top-level title if missing).
Use this structure, filling in the iteration number from the task:

```markdown
## Iteration <N> - <pass|fail>

### Performance
- FlyDSL: <X> us   |   HIP baseline: <Y> us   |   ratio: <X/Y> (target <= {ratio})
- <benchmark command used; shapes/dtype>

### Accuracy
- result: <pass|fail>   max_rel_err: <e>   (rtol target {rtol})
- <test command used>

### Trace (capture-kernel-trace)
- status: <captured|failed: reason>
- location: <path to ui_output_agent_* / trace_data dir>
- kernel: <name>  duration: <us>  VGPR/AGPR/SGPR: <...>
- instructions: <n>  source-mapped: <n> (<pct>%)
- notes: <key stalls / bottleneck hints from the trace, if any>

### LLVM IR
- FlyDSL IR: <abs path to flydsl.ll>
- HIP IR: <abs path to hip.ll>  (compiler: <hipcc ...>)
- diff summary: <key structural divergences vs HIP, or "aligned">
```

Then your FINAL message must end with a single fenced ```json block and nothing
after it, matching exactly:

```json
{
  "passed": true,
  "mode": "gpu",
  "accuracy": {"passed": true, "max_rel_err": 0.0, "details": "..."},
  "performance": {"passed": true, "flydsl_us": 0.0, "hip_us": 0.0, "ratio": 0.0},
  "trace": {"captured": true, "location": "...", "duration_us": 0.0, "summary": "..."},
  "ir_alignment": {"passed": true, "flydsl_ir_path": "...", "hip_ir_path": "...", "details": "..."},
  "perf_doc": "<path to the performance doc you appended>",
  "feedback": "Concrete actionable feedback for the next iteration; empty if passed."
}
```

`passed` is true ONLY if every gating axis for the current mode passed (trace is
NOT gating). `feedback` must point at the specific divergence (file/line, missing
layout, wrong atom, wrong unroll) so the next iteration can fix it directly.
""".replace("{ratio}", f"{cfg.perf_ratio:g}").replace("{rtol}", f"{cfg.accuracy_rtol:g}")


# --------------------------------------------------------------------------- #
# Agent runner: manual tool-use loop on the Messages API
# --------------------------------------------------------------------------- #
def _block_to_input(b: Any) -> dict[str, Any]:
    """Convert a response content block to a minimal API-input dict (drops
    output-only fields like text.parsed_output that the API rejects on input)."""
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if b.type == "thinking":
        return {"type": "thinking", "thinking": b.thinking, "signature": b.signature}
    if b.type == "redacted_thinking":
        return {"type": "redacted_thinking", "data": b.data}
    return b.model_dump()


def run_agent(
    client: anthropic.Anthropic,
    *,
    label: str,
    system_prompt: str,
    user_prompt: str,
    tool_names: list[str],
    cfg: Config,
    effort: str = "high",
) -> str:
    tools = [TOOL_SCHEMAS[n] for n in tool_names]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    print(f"\n{'=' * 72}\n[{label}] running (effort={effort})...\n{'=' * 72}", flush=True)

    last_text = ""
    for turn in range(cfg.max_turns):
        with client.messages.stream(
            model=cfg.model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=messages,
            output_config={"effort": effort},
        ) as stream:
            resp = stream.get_final_message()

        # Echo assistant text; collect tool calls.
        tool_uses = []
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                last_text = block.text
                print(f"[{label}] {block.text}", flush=True)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # Record the assistant turn. Rebuild minimal input blocks - model_dump()
        # includes output-only fields (e.g. text.parsed_output) the API rejects.
        messages.append({"role": "assistant", "content": [_block_to_input(b) for b in resp.content]})

        if resp.stop_reason != "tool_use":
            break

        # Execute each requested tool and feed results back.
        tool_results = []
        for tu in tool_uses:
            result, is_error = execute_tool(cfg, tu.name, tu.input)
            print(
                f"[{label}] -> {tu.name}({_brief(tu.input)}) "
                f"=> {'ERR ' if is_error else ''}{result[:200].splitlines()[0] if result else ''}",
                flush=True,
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"[{label}] hit max_turns={cfg.max_turns}", flush=True)

    return last_text


def _brief(d: dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        s = str(v).replace("\n", " ")
        parts.append(f"{k}={s[:40]}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #
def parse_verdict(text: str) -> dict[str, Any]:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = blocks[-1] if blocks else None
    if candidate is None:
        brace = re.findall(r"(\{.*\})", text, re.DOTALL)
        candidate = brace[-1] if brace else None
    if candidate is None:
        return {"passed": False, "feedback": "Evaluator produced no JSON verdict.", "_parse_error": True}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"passed": False, "feedback": f"Invalid JSON verdict ({exc}).", "_parse_error": True}


# --------------------------------------------------------------------------- #
# Stage prompts
# --------------------------------------------------------------------------- #
def analyzer_task(
    cfg: Config,
    it: int,
    feedback: str,
    trace_dir: str | None,
    flydsl_ir: str | None,
    hip_ir: str | None,
) -> str:
    s = f"""Port this HIP kernel to FlyDSL with 1:1 LLVM alignment.

HIP source:   {cfg.hip_source}
HIP dep root: {cfg.hip_root}  (follow #includes within this root)
Kernel name:  {cfg.kernel_name}
FlyDSL output target: {cfg.output}
Test file:    {cfg.test_file or "(none)"}
Write the plan to: {cfg.plan_file}
Plan the KERNEL port only - do NOT design the unit test (the test author owns that)."""
    if it > 0:
        s += f"""

Iteration {it + 1}: the previous attempt was evaluated and did NOT pass.
Evaluator feedback:
{feedback or "(none)"}

Performance doc (perf/accuracy/trace/IR history): {cfg.perf_doc}
kernel-trace-analysis skill: {cfg.analysis_skill}
Latest captured trace dir: {trace_dir or "(none - read the trace location from the performance doc)"}
FlyDSL LLVM IR: {flydsl_ir or "(read the path from the performance doc's LLVM IR section)"}
HIP LLVM IR: {hip_ir or "(read the path from the performance doc's LLVM IR section)"}

Do BOTH analyses, then UPDATE the plan (see the REFINEMENT ITERATIONS
instructions):
  (a) diff the FlyDSL vs HIP LLVM IR to find where 1:1 alignment broke
      (instruction shape, access widths, atoms, LDS/barriers, unroll), and
  (b) analyze the latest trace with kernel-trace-analysis to find the real
      runtime bottleneck.
Reconcile the two: an IR divergence often explains a trace hotspot. Address the
IR misalignment, the bottleneck, and the evaluator feedback."""
    return s


def implementer_task(cfg: Config, it: int, feedback: str, smoke_path: Path) -> str:
    s = f"""Implement the FlyDSL port following the plan.

Plan file:    {cfg.plan_file}
HIP source:   {cfg.hip_source}
HIP dep root: {cfg.hip_root}  (follow #includes here if you need more detail)
flydsl-kernel-authoring skill: {cfg.authoring_skill}  (READ and follow it)
Write kernel to: {cfg.output}
Write the local compile-gate smoke harness to: {smoke_path}
It MUST pass locally before you finish (the orchestrator re-runs this exact gate):
  COMPILE_ONLY=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 ARCH={cfg.compile_arch} python3 {smoke_path}
Your job is the KERNEL only - do NOT write the unit test (the test author owns that).
Test file:    {cfg.test_file or "(none)"}"""
    if feedback:
        s += f"\n\nFix these issues before finishing:\n{feedback}"
    return s


def _hip_build_context(cfg: Config) -> str:
    inc = ", ".join(cfg.hip_include_dirs) or "(none)"
    bc = cfg.hip_build_cmd or "(default hipcc)"
    return (
        f"HIP dep root: {cfg.hip_root}  (include with -I; the kernel may #include "
        f"files here)\nHIP extra include dirs: {inc}\nHIP build command: {bc}"
    )


def test_author_task(cfg: Config, test_path: Path) -> str:
    return f"""Author a numerical-correctness pytest using the HIP kernel as reference.

HIP source (reference): {cfg.hip_source}
{_hip_build_context(cfg)}
FlyDSL kernel (under test): {cfg.output}
Kernel name:  {cfg.kernel_name}
Write the test to: {test_path}
Accuracy rtol target: {cfg.accuracy_rtol:g}
Test reference harness: {cfg.test_reference or "(none - synthesize inputs yourself)"}

If a test reference harness is given, BUILD the unit test from it (reuse its
input/weight/quant/shuffle-layout construction; golden = HIP/aiter gemm1 kernel
on those inputs; isolate gemm1, not the whole fused MoE) - see REFERENCE HARNESS.
When compiling the HIP reference, add `-I<HIP dep root>` and the extra include
dirs above (or use the HIP build command if given) so its #includes resolve.
Create the test, then run it once with FLYDSL_RUNTIME_ENABLE_CACHE=0 to confirm
it executes and exercises both kernels.
{cfg.remote_ctx}
{cfg.extra_context}"""


def evaluator_task(cfg: Config, it: int) -> str:
    return f"""Evaluate the FlyDSL port of `{cfg.kernel_name}`.

Iteration number: {it + 1}
FlyDSL kernel:  {cfg.output}
HIP source:     {cfg.hip_source}
{_hip_build_context(cfg)}
Test file:      {cfg.test_file or "(none)"}
Accuracy rtol target: {cfg.accuracy_rtol:g}
Performance target:   flydsl_latency <= {cfg.perf_ratio:g} * hip_latency
capture-kernel-trace skill: {cfg.trace_skill}
LLVM IR dump dir (use {cfg.ir_dir}/iter{it + 1}/): {cfg.ir_dir}
Performance doc to append: {cfg.perf_doc}

Run the checks, capture the kernel trace, export both LLVM IRs (FlyDSL + HIP) to
{cfg.ir_dir}/iter{it + 1}/, append the "## Iteration {it + 1}" section to the
performance doc (including the LLVM IR paths), then emit the JSON verdict.
{cfg.remote_ctx}
{cfg.extra_context}"""


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url="https://llm-api.amd.com/Unified",
        api_key=os.environ["AMD_LLM_GATEWAY_KEY"],
        default_headers={"Ocp-Apim-Subscription-Key": os.environ["AMD_LLM_GATEWAY_KEY"]},
    )


def local_compile_gate(cfg: Config, smoke_path: Path) -> tuple[bool, str]:
    """Run the implementer's smoke harness locally under COMPILE_ONLY to verify the
    kernel traces+compiles before the (expensive, remote) evaluator sees it."""
    if not smoke_path.exists():
        return False, (
            f"Smoke harness {smoke_path} was not written. Create it: import the "
            "kernel module and invoke the @flyc.jit launcher with representative "
            "dummy torch tensors (importing alone does not trigger tracing)."
        )
    env = {**os.environ, "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0", "ARCH": cfg.compile_arch}
    print(f"[gate] running local COMPILE_ONLY (ARCH={cfg.compile_arch}): {smoke_path}")
    try:
        proc = subprocess.run(
            ["python3", str(smoke_path)],
            cwd=str(cfg.repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=cfg.bash_timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"local COMPILE_ONLY smoke timed out after {cfg.bash_timeout}s"
    return proc.returncode == 0, _truncate(proc.stdout + proc.stderr)


def orchestrate(cfg: Config, client: anthropic.Anthropic) -> int:
    feedback = ""
    trace_dir: str | None = None
    flydsl_ir: str | None = None
    hip_ir: str | None = None

    for it in range(cfg.max_iterations):
        print(f"\n\n########## ITERATION {it + 1}/{cfg.max_iterations} ##########")

        run_agent(
            client,
            label=f"ANALYZER (it{it + 1})",
            system_prompt=ANALYZER_PROMPT,
            user_prompt=analyzer_task(cfg, it, feedback, trace_dir, flydsl_ir, hip_ir),
            tool_names=cfg.analyzer_tools,
            cfg=cfg,
            effort="max",
        )

        # Implement + local COMPILE_ONLY gate: the kernel must trace+compile
        # locally before it reaches the (expensive, remote) evaluator. On failure,
        # feed the compile error back to the implementer and retry.
        smoke_path = cfg.output.parent / "_smoke_compile.py"
        gate_feedback = feedback
        for attempt in range(cfg.compile_attempts):
            run_agent(
                client,
                label=f"IMPLEMENTER (it{it + 1}.{attempt + 1})",
                system_prompt=IMPLEMENTER_PROMPT,
                user_prompt=implementer_task(cfg, it, gate_feedback, smoke_path),
                tool_names=cfg.implementer_tools,
                cfg=cfg,
                effort="xhigh",
            )
            ok, gate_out = local_compile_gate(cfg, smoke_path)
            if ok:
                print(f"[gate] local COMPILE_ONLY passed (attempt {attempt + 1})")
                break
            print(
                f"[gate] local COMPILE_ONLY FAILED (attempt {attempt + 1}/{cfg.compile_attempts}); "
                "feeding error back to implementer"
            )
            gate_feedback = ((feedback + "\n\n") if feedback else "") + (
                "Your kernel FAILED the local COMPILE_ONLY gate. Fix these errors and make "
                f"the smoke harness ({smoke_path}) exit 0:\n{gate_out}"
            )
        else:
            print(
                f"[gate] local compile still failing after {cfg.compile_attempts} attempts; "
                "proceeding so the evaluator records the failure"
            )

        # No test file: in GPU mode, author one once (HIP kernel as reference) so
        # the accuracy axis is verifiable. The FlyDSL kernel now exists, so its
        # entry point is concrete. compile_ir mode cannot run kernels, so skip.
        if cfg.test_file is None and cfg.eval_mode == "gpu":
            test_path = (cfg.repo_root / f"tests/kernels/test_{cfg.kernel_name}_autogen.py").resolve()
            run_agent(
                client,
                label=f"TEST AUTHOR (it{it + 1})",
                system_prompt=TEST_AUTHOR_PROMPT,
                user_prompt=test_author_task(cfg, test_path),
                tool_names=cfg.test_author_tools,
                cfg=cfg,
            )
            if test_path.exists():
                cfg.test_file = test_path
                print(f"[orchestrate] using autogenerated test: {test_path}")
            else:
                print(
                    f"[orchestrate] WARNING: test author did not produce {test_path}; "
                    "accuracy axis will be unverifiable this run."
                )

        verdict_text = run_agent(
            client,
            label=f"EVALUATOR (it{it + 1})",
            system_prompt=evaluator_prompt(cfg),
            user_prompt=evaluator_task(cfg, it),
            tool_names=cfg.evaluator_tools,
            cfg=cfg,
            effort="medium",
        )
        verdict = parse_verdict(verdict_text)

        print(f"\n----- VERDICT (it{it + 1}) -----")
        print(json.dumps(verdict, indent=2, ensure_ascii=False))

        if verdict.get("passed") is True:
            print(f"\n✅ Port accepted on iteration {it + 1}: {cfg.output}")
            return 0

        # Carry the latest trace location and exported LLVM IR paths forward so
        # the next ANALYZER can run kernel-trace-analysis and diff the IRs.
        new_trace = (verdict.get("trace") or {}).get("location")
        if new_trace:
            trace_dir = new_trace
        ir = verdict.get("ir_alignment") or {}
        if ir.get("flydsl_ir_path"):
            flydsl_ir = ir["flydsl_ir_path"]
        if ir.get("hip_ir_path"):
            hip_ir = ir["hip_ir_path"]
        feedback = verdict.get("feedback") or "No specific feedback; re-examine the port."
        print(f"\n❌ Iteration {it + 1} rejected. Carrying feedback forward.")

    print(f"\n⛔ No accepted port within {cfg.max_iterations} iterations.\nLast feedback:\n{feedback}")
    return 1


# --------------------------------------------------------------------------- #
# GPU detection + CLI
# --------------------------------------------------------------------------- #
def detect_eval_mode(forced: str) -> str:
    if forced in ("gpu", "compile_ir"):
        return forced
    if os.environ.get("COMPILE_ONLY"):
        return "compile_ir"
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return "gpu"
    return "compile_ir"


def build_remote_ctx(host: str, user: str, container: str, remote_root: str, workdir: str) -> str:
    """Build the prompt block that makes an agent run GPU work on a remote host via
    ssh/scp/docker (invoked through its local run_bash). Empty if no host."""
    if not host:
        return ""
    target = f"{user}@{host}" if user else host
    workdir = workdir or "/tmp/flydsl_port"
    if container:
        run_pat = (
            f'ssh {target} "docker exec -e PYTHONPATH={remote_root}/python:{remote_root}/tests '
            f"-e FLYDSL_RUNTIME_ENABLE_CACHE=0 {container} bash -c 'cd {workdir} && <CMD>'\""
        )
        copy_in = f'scp <file> {target}:/tmp/ && ssh {target} "docker cp /tmp/<file> {container}:{workdir}/"'
        copy_out = (
            f'ssh {target} "docker cp {container}:<remote_path> /tmp/" && ' f"scp {target}:/tmp/<file> <local_path>"
        )
    else:
        run_pat = (
            f'ssh {target} "cd {remote_root} && PYTHONPATH={remote_root}/python:{remote_root}/tests '
            f"FLYDSL_RUNTIME_ENABLE_CACHE=0 bash -c 'cd {workdir} && <CMD>'\""
        )
        copy_in = f"scp <file> {target}:{workdir}/"
        copy_out = f"scp {target}:<remote_path> <local_path>"

    return f"""
REMOTE GPU EXECUTION (a remote GPU host is configured):
  Run ALL GPU/compile work on the REMOTE host, not locally - accuracy pytest,
  benchmark, FLYDSL_DUMP_IR, hipcc IR, and the rocprofv3 trace (the
  capture-kernel-trace skill's remote/SSH branch). Use your run_bash tool to
  invoke ssh/scp/docker yourself.
    SSH target:        {target}
    Docker container:  {container or "(none - run directly on the host)"}
    Remote FlyDSL root:{remote_root}
    Remote work dir:   {workdir}
  Run-command pattern (substitute <CMD>):
    {run_pat}
  Copy an input TO remote:
    {copy_in}
  Copy a result BACK to local:
    {copy_out}
  Steps: (1) ensure {workdir} exists on the remote; (2) copy the FlyDSL kernel,
  the test, and the HIP source/deps to the remote; (3) run the checks remotely;
  (4) copy the trace dir, the *.ll IR files, and any result files BACK to the
  LOCAL paths given in this task (ir_dir, perf_doc) so the local performance doc
  and the next ANALYZER can read them. Record the remote commands you used.
"""


def _is_git_url(raw: str) -> bool:
    if raw.startswith("git@"):
        return True
    base = raw.split("#", 1)[0].split("?", 1)[0]
    return base.endswith(".git") or "#" in raw


# https://github.com/<owner>/<repo>/(blob|tree|raw)/<ref>/<path>
_GH_BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/(?:blob|tree|raw)/([^/]+)/(.+)$")


def _clone_repo(url: str, ref: str, clone_dir: Path) -> None:
    """Clone `url` into `clone_dir`, checked out at `ref` (commit/branch/tag) if given."""
    if clone_dir.exists():
        print(f"[fetch] reusing existing clone {clone_dir}")
        return
    if ref:
        # Try a shallow fetch of just the pinned ref (GitHub allows fetch-by-SHA).
        clone_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(clone_dir), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "remote", "add", "origin", url], check=True)
        shallow = subprocess.run(["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", ref])
        if shallow.returncode == 0:
            subprocess.run(["git", "-C", str(clone_dir), "checkout", "-q", "FETCH_HEAD"], check=True)
            print(f"[fetch] cloned {url} @ {ref} (shallow)")
            return
        print("[fetch] shallow fetch of ref failed; falling back to full clone + checkout")
        shutil.rmtree(clone_dir, ignore_errors=True)
    print(f"[fetch] git clone {url} -> {clone_dir}")
    subprocess.run(["git", "clone", url, str(clone_dir)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(clone_dir), "checkout", "-q", ref], check=True)


def fetch_hip_source(raw: str, dest_root: Path, git_ref: str = "") -> tuple[Path, Path]:
    """Resolve --hip-source (local path / single-file URL / git URL / GitHub blob
    URL) to a local file. Returns (hip_source_file, hip_root) where hip_root is the
    dependency root the agents may read for #includes."""
    # GitHub blob/tree/raw URL: owner/repo + embedded ref + subpath -> clone at ref.
    m = _GH_BLOB_RE.match(raw)
    if m:
        owner, repo, ref, subpath = m.groups()
        repo = repo[:-4] if repo.endswith(".git") else repo
        url = f"https://github.com/{owner}/{repo}.git"
        clone_dir = (dest_root / "repo").resolve()
        _clone_repo(url, git_ref or ref, clone_dir)
        src = (clone_dir / subpath).resolve()
        if not src.exists():
            raise SystemExit(f"subpath not found in cloned repo: {src}")
        return src, clone_dir

    # git URL: "<url>" or "<url>#<subpath-to-kernel>"
    if _is_git_url(raw):
        url, _, subpath = raw.partition("#")
        if not subpath:
            raise SystemExit(
                f"git HIP source needs a '#<subpath>' to the kernel file, e.g.\n" f"  {url}#path/to/kernel.hip.cpp"
            )
        clone_dir = (dest_root / "repo").resolve()
        _clone_repo(url, git_ref, clone_dir)
        src = (clone_dir / subpath).resolve()
        if not src.exists():
            raise SystemExit(f"subpath not found in cloned repo: {src}")
        return src, clone_dir

    # single-file http(s) URL
    if raw.startswith(("http://", "https://")):
        dest_root.mkdir(parents=True, exist_ok=True)
        fname = raw.split("?", 1)[0].rstrip("/").split("/")[-1] or "hip_source"
        dest = (dest_root / fname).resolve()
        print(f"[fetch] downloading {raw} -> {dest}")
        urllib.request.urlretrieve(raw, dest)  # noqa: S310 (trusted user input)
        return dest, dest_root.resolve()

    # local path
    src = Path(raw).resolve()
    if not src.exists():
        raise SystemExit(f"HIP source not found: {src}")
    return src, src.parent.resolve()


# --------------------------------------------------------------------------- #
# Prompt -> structured config (parsed by an agent)
# --------------------------------------------------------------------------- #
# Field defaults applied after parsing; also the source of truth for valid keys.
CONFIG_DEFAULTS: dict[str, Any] = {
    "hip_source": "",  # REQUIRED: local path / http(s) URL / git URL#subpath / GitHub blob URL
    "hip_git_ref": "",  # commit/branch/tag (overrides a ref in a blob URL)
    "kernel_name": "",  # REQUIRED
    "output_dir": "",  # REQUIRED: artifacts dir
    "test_file": "",  # optional pytest path; "" -> auto-generate
    "repo_root": "",  # FlyDSL repo root; "" -> cwd
    "model": "claude-opus-4-8",
    "max_iterations": 10,
    "max_turns": 60,
    "accuracy_rtol": 1e-3,
    "perf_ratio": 1.10,
    "eval_mode": "auto",  # auto | gpu | compile_ir
    "bash_timeout": 1800,
    "compile_arch": "gfx950",  # ARCH for the local COMPILE_ONLY gate before remote eval
    "compile_attempts": 3,  # implementer retries to pass the local compile gate
    "hip_include_dirs": "",  # comma-separated -I dirs (relative -> against HIP root)
    "hip_build_cmd": "",  # explicit HIP IR/reference build command
    "ssh_host": "",  # remote GPU host (enables remote execution)
    "ssh_user": "",
    "container": "",  # docker container on the remote (optional)
    "remote_root": "",  # FlyDSL root on the remote host
    "remote_workdir": "",  # remote scratch dir (default /tmp/flydsl_port)
    "test_reference": "",  # path to a reference harness to BUILD the unit test from
    "extra_context": "",  # anything else worth passing to the agents
}

TASK_PARSER_SYSTEM = f"""You convert a natural-language kernel-porting request into
a strict JSON config for an automation script. Output ONE JSON object and nothing
else - no prose, no markdown fence.

Use exactly these keys (types and defaults shown); omit nothing:
{json.dumps(CONFIG_DEFAULTS, indent=2)}

Rules:
- hip_source, kernel_name, output_dir are REQUIRED - extract them from the request.
- If the request gives a GitHub blob/commit URL, put it verbatim in hip_source
  (the script extracts the commit/subpath itself); leave hip_git_ref "" unless a
  separate ref is stated.
- Infer kernel_name from the file/function if not stated (e.g. "gemm1_a4w4.cuh"
  -> "gemm1_a4w4"). Infer output_dir as "ports/<kernel_name>" if not stated.
- Map remote-GPU phrasing ("on host X", "remote gpu", "in container C", "ssh as U")
  to ssh_host/ssh_user/container/remote_root/remote_workdir.
- Map include dirs / build commands to hip_include_dirs (comma-separated) /
  hip_build_cmd. Numbers like iterations, rtol, perf ratio map to their fields.
- If the request names a reference harness/script to BUILD the unit test from
  ("use X as reference", "construct a unit test from X"), put ONLY the path in
  test_reference (distinct from test_file, an already-written test). Do NOT also
  copy any test-construction wording into extra_context - test-building guidance
  must reach the test author alone, not the analyzer/implementer.
- extra_context is general notes for ALL agents; keep it minimal and free of
  test-construction or remote-exec instructions (those have dedicated fields).
- For anything else stated that has no dedicated field, append it to extra_context.
- Use the given defaults for anything not mentioned. Output valid JSON only."""


def parse_task_prompt(client: anthropic.Anthropic, task: str, model: str) -> dict[str, Any]:
    """Run one agent that parses the natural-language request into a config dict."""
    print("Parsing task prompt into config...")
    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=TASK_PARSER_SYSTEM,
        messages=[{"role": "user", "content": task}],
        output_config={"effort": "low"},
    ) as stream:
        resp = stream.get_final_message()
    text = "".join(b.text for b in resp.content if b.type == "text")
    blocks = re.findall(r"\{.*\}", text, re.DOTALL)
    if not blocks:
        raise SystemExit(f"task parser did not return JSON. Raw:\n{text}")
    try:
        parsed = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"task parser returned invalid JSON ({exc}). Raw:\n{text}")
    fields = {**CONFIG_DEFAULTS, **{k: v for k, v in parsed.items() if k in CONFIG_DEFAULTS}}
    print("Parsed config:\n" + json.dumps(fields, indent=2))
    for req in ("hip_source", "kernel_name", "output_dir"):
        if not fields[req]:
            raise SystemExit(f"could not determine required field '{req}' from the request.")
    return fields


def build_config(fields: dict[str, Any]) -> Config:
    """Build a Config from the parsed field dict (fetch source, resolve paths)."""
    repo_root = Path(fields["repo_root"] or Path.cwd()).resolve()
    output_dir = Path(fields["output_dir"])
    output_dir = (output_dir if output_dir.is_absolute() else (repo_root / output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{fields['kernel_name']}.py"
    plan_file = output_dir / "plan.md"
    # Per-run hash so each invocation gets its own performance doc (no stale-append).
    run_hash = hashlib.sha1(f"{time.time()}|{fields['hip_source']}|{fields['kernel_name']}".encode()).hexdigest()[:8]
    perf_doc = output_dir / f"performance_{run_hash}.md"
    print(f"Performance doc for this run: {perf_doc}")
    ir_dir = output_dir / "ir_dumps"
    trace_skill = (repo_root / ".claude/skills/capture-kernel-trace/SKILL.md").resolve()
    analysis_skill = (repo_root / ".claude/skills/kernel-trace-analysis/SKILL.md").resolve()
    authoring_skill = (repo_root / ".claude/skills/flydsl-kernel-authoring/SKILL.md").resolve()

    hip_source, hip_root = fetch_hip_source(fields["hip_source"], output_dir / "hip_src", fields["hip_git_ref"])
    print(f"HIP source: {hip_source}\nHIP root (deps): {hip_root}")

    def _abs_inc(s: str) -> str:
        p = Path(s)
        return str((p if p.is_absolute() else hip_root / p).resolve())

    hip_include_dirs = [_abs_inc(s.strip()) for s in str(fields["hip_include_dirs"]).split(",") if s.strip()]
    remote_ctx = build_remote_ctx(
        fields["ssh_host"],
        fields["ssh_user"],
        fields["container"],
        fields["remote_root"],
        fields["remote_workdir"],
    )
    test_reference = str(Path(fields["test_reference"]).resolve()) if fields["test_reference"] else ""

    eval_mode = detect_eval_mode(fields["eval_mode"])
    if fields["ssh_host"] and fields["eval_mode"] == "auto":
        eval_mode = "gpu"  # a remote GPU is configured; don't fall back to compile_ir
    print(f"Evaluation mode: {eval_mode}{' (remote GPU)' if fields['ssh_host'] else ''}")
    if eval_mode == "gpu" and not trace_skill.exists():
        print(
            f"WARNING: capture-kernel-trace skill not found at {trace_skill}; "
            "the evaluator will record trace as unavailable."
        )

    return Config(
        hip_source=hip_source,
        hip_root=hip_root,
        kernel_name=fields["kernel_name"],
        output=output,
        test_file=Path(fields["test_file"]).resolve() if fields["test_file"] else None,
        repo_root=repo_root,
        plan_file=plan_file,
        model=fields["model"],
        max_iterations=int(fields["max_iterations"]),
        max_turns=int(fields["max_turns"]),
        accuracy_rtol=float(fields["accuracy_rtol"]),
        perf_ratio=float(fields["perf_ratio"]),
        eval_mode=eval_mode,
        bash_timeout=int(fields["bash_timeout"]),
        compile_arch=fields["compile_arch"],
        compile_attempts=int(fields["compile_attempts"]),
        perf_doc=perf_doc,
        trace_skill=trace_skill,
        analysis_skill=analysis_skill,
        authoring_skill=authoring_skill,
        ir_dir=ir_dir,
        hip_include_dirs=hip_include_dirs,
        hip_build_cmd=fields["hip_build_cmd"],
        remote_ctx=remote_ctx,
        test_reference=test_reference,
        extra_context=fields["extra_context"],
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Port a HIP kernel to FlyDSL via a multi-agent loop (Anthropic API). "
        "Describe the whole request in natural language; an agent parses it."
    )
    p.add_argument(
        "task",
        nargs="+",
        help='Natural-language port request, e.g. "Port gemm1_a4w4 from '
        "<github url> into ports/gemm1_a4w4, include dirs csrc/include, "
        'test on remote gpu01 in container flydsl_ctr".',
    )
    a = p.parse_args(sys.argv[1:])

    client = make_client()
    fields = parse_task_prompt(client, " ".join(a.task), CONFIG_DEFAULTS["model"])
    cfg = build_config(fields)
    return orchestrate(cfg, client)


if __name__ == "__main__":
    raise SystemExit(main())
