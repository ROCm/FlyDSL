#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Generate GitHub Actions job summaries for FlyDSL CI.

Usage:
    python3 scripts/generate_summary.py build
    python3 scripts/generate_summary.py test
    python3 scripts/generate_summary.py promote

Each mode reads its inputs from environment variables and appends
Markdown to $GITHUB_STEP_SUMMARY.
"""

import os
import re
import sys
from pathlib import Path


DOMAIN_MAP = {
    "nightlies": "rocm.frameworks-nightlies.amd.com",
    "devreleases": "rocm.frameworks-devreleases.amd.com",
    "prereleases": "rocm.frameworks-prereleases.amd.com",
    "release": "rocm.frameworks.amd.com",
}


class SummaryWriter:
    """Collects lines in memory, flushes once to GITHUB_STEP_SUMMARY."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def line(self, text: str = "") -> None:
        self._lines.append(text)

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        self.line("| " + " | ".join(headers) + " |")
        self.line("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self.line("| " + " | ".join(row) + " |")
        self.line()

    def flush(self, path: Path) -> None:
        with open(path, "a") as f:
            f.write("\n".join(self._lines) + "\n")


# ── Build summary ───────────────────────────────────────────────────────────

def build_summary(summary: Path) -> None:
    docker_image = os.environ.get("SUMMARY_DOCKER_IMAGE", "unknown")
    llvm_commit = os.environ.get("SUMMARY_LLVM_COMMIT", "unknown")
    mlir_cache = os.environ.get("SUMMARY_MLIR_CACHE", "unknown")
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    wheel_dir = os.environ.get("SUMMARY_WHEEL_DIR", "dist")

    w = SummaryWriter()
    w.line("## Build Summary")
    w.line()
    w.table(["Item", "Value"], [
        ["Docker image", f"`{docker_image}`"],
        ["LLVM commit", f"`{llvm_commit}`"],
        ["MLIR cache", mlir_cache],
        ["Release type", f"`{release_type}`"],
    ])

    w.line("### Wheels")
    w.line("```")
    whl_dir = Path(wheel_dir)
    wheels = sorted(whl_dir.glob("*.whl")) if whl_dir.is_dir() else []
    if wheels:
        for wh in wheels:
            size_mb = wh.stat().st_size / (1024 * 1024)
            w.line(f"  {wh.name}  ({size_mb:.1f} MB)")
    else:
        w.line("  No wheels found")
    w.line("```")
    w.flush(summary)


# ── Test summary ────────────────────────────────────────────────────────────

def test_summary(summary: Path) -> None:
    runner = os.environ.get("SUMMARY_RUNNER", "unknown")
    install_outcome = os.environ.get("SUMMARY_INSTALL_OUTCOME", "unknown")
    external_tools_outcome = os.environ.get("SUMMARY_EXTERNAL_TOOLS_OUTCOME", "unknown")
    tests_outcome = os.environ.get("SUMMARY_TESTS_OUTCOME", "unknown")
    bench_outcome = os.environ.get("SUMMARY_BENCHMARKS_OUTCOME", "unknown")
    test_log = os.environ.get("SUMMARY_TEST_LOG", "/tmp/test_output.log")
    bench_log = os.environ.get("SUMMARY_BENCH_LOG", "/tmp/bench_output.log")

    w = SummaryWriter()
    w.line(f"## Test Summary (`{runner}`)")
    w.line()
    w.table(["Step", "Status"], [
        ["Install wheels", f"`{install_outcome}`"],
        ["Smoke test external LLVM tools", f"`{external_tools_outcome}`"],
        ["Run tests", f"`{tests_outcome}`"],
        ["Run benchmarks", f"`{bench_outcome}`"],
    ])

    _write_test_results(w, test_log)
    _write_bench_results(w, bench_log)
    w.flush(summary)


def _write_test_results(w: SummaryWriter, log_path: str) -> None:
    log = Path(log_path)
    if not log.is_file():
        return

    text = log.read_text(errors="replace")
    mlir = _first_match(r"^MLIR Tests:.*", text) or "N/A"
    ir_result = _first_match(r"^IR Tests:.*", text) or "N/A"
    gpu = _first_match(r"^GPU Tests:.*", text) or "N/A"

    w.line("### Test Results")
    w.line()
    w.table(["Suite", "Result"], [
        ["MLIR IR (Lowering)", mlir],
        ["Python IR (Generation)", ir_result],
        ["GPU Execution", gpu],
    ])


def _write_bench_results(w: SummaryWriter, log_path: str) -> None:
    log = Path(log_path)
    if not log.is_file():
        return

    text = log.read_text(errors="replace")

    perf_block = _extract_perf_table(text)
    if perf_block:
        w.line("### Benchmark Results")
        w.line()
        w.line("```")
        for line in perf_block[:30]:
            w.line(line)
        w.line("```")
        w.line()

    for pattern in (r"^Total:.*", r"^Success:.*", r"^Failed:.*"):
        match = _first_match(pattern, text)
        if match:
            w.line(match)
    w.line()


def _extract_perf_table(text: str) -> list[str]:
    """Return lines between the 'op' header and 'Benchmark Summary'."""
    lines: list[str] = []
    capturing = False
    for line in text.splitlines():
        if not capturing and line.startswith("op "):
            capturing = True
        if capturing:
            if "Benchmark Summary" in line:
                break
            lines.append(line)
    return lines


# ── Promote summary ─────────────────────────────────────────────────────────

def promote_summary(summary: Path) -> None:
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    source = os.environ.get("SUMMARY_S3_SOURCE", "unknown")
    dest = os.environ.get("SUMMARY_S3_DEST", "unknown")
    wheel_names = os.environ.get("SUMMARY_WHEEL_NAMES", "").strip()
    llvm_tool_names = os.environ.get("SUMMARY_LLVM_TOOL_NAMES", "").strip()

    w = SummaryWriter()
    w.line("## Promote Summary")
    w.line()
    w.table(["Item", "Value"], [
        ["Release type", f"`{release_type}`"],
        ["Source", f"`{source}`"],
        ["Destination", f"`{dest}`"],
    ])

    if wheel_names:
        w.line("### Promoted Wheels")
        w.line("```")
        for whl in wheel_names.split():
            w.line(f"  {whl}")
        w.line("```")
        w.line()

    if llvm_tool_names:
        w.line("### Promoted LLVM Tools")
        w.line("```")
        for tool in llvm_tool_names.split():
            w.line(f"  {tool}")
        w.line("```")
        w.line()

    domain = DOMAIN_MAP.get(release_type)
    if domain:
        index_url = f"https://{domain}/whl/gfx942-gfx950/"
        llvm_tools_url = f"https://{domain}/llvm-tools/gfx942-gfx950/"
        w.line("### Wheels Available At")
        w.line(f"- {index_url}")
        if llvm_tool_names:
            w.line("### LLVM Tools Available At")
            w.line(f"- {llvm_tools_url}")
        w.line()
        w.line("### Install")
        w.line("```bash")
        w.line(f"pip install --index-url {index_url} flydsl")
        w.line("```")
        w.line()
    w.flush(summary)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _first_match(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(0) if m else None


# ── Main ────────────────────────────────────────────────────────────────────

MODES = {
    "build": build_summary,
    "test": test_summary,
    "promote": promote_summary,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        print(f"Usage: {sys.argv[0]} {{{','.join(MODES)}}}", file=sys.stderr)
        sys.exit(1)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        print("GITHUB_STEP_SUMMARY is not set", file=sys.stderr)
        sys.exit(1)

    MODES[sys.argv[1]](Path(summary_path))


if __name__ == "__main__":
    main()
