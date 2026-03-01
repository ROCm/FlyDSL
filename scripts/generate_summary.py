#!/usr/bin/env python3
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


def _out(path: Path, line: str = "") -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def _table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    _out(path, "| " + " | ".join(headers) + " |")
    _out(path, "| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        _out(path, "| " + " | ".join(row) + " |")
    _out(path)


# ── Build summary ───────────────────────────────────────────────────────────

def build_summary(summary: Path) -> None:
    docker_image = os.environ.get("SUMMARY_DOCKER_IMAGE", "unknown")
    llvm_commit = os.environ.get("SUMMARY_LLVM_COMMIT", "unknown")
    mlir_cache = os.environ.get("SUMMARY_MLIR_CACHE", "unknown")
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    wheel_dir = os.environ.get("SUMMARY_WHEEL_DIR", "dist")

    _out(summary, "## Build Summary")
    _out(summary)
    _table(summary, ["Item", "Value"], [
        ["Docker image", f"`{docker_image}`"],
        ["LLVM commit", f"`{llvm_commit}`"],
        ["MLIR cache", mlir_cache],
        ["Release type", f"`{release_type}`"],
    ])

    _out(summary, "### Wheels")
    _out(summary, "```")
    whl_dir = Path(wheel_dir)
    wheels = sorted(whl_dir.glob("*.whl")) if whl_dir.is_dir() else []
    if wheels:
        for w in wheels:
            size_mb = w.stat().st_size / (1024 * 1024)
            _out(summary, f"  {w.name}  ({size_mb:.1f} MB)")
    else:
        _out(summary, "  No wheels found")
    _out(summary, "```")


# ── Test summary ────────────────────────────────────────────────────────────

def test_summary(summary: Path) -> None:
    runner = os.environ.get("SUMMARY_RUNNER", "unknown")
    install_outcome = os.environ.get("SUMMARY_INSTALL_OUTCOME", "unknown")
    tests_outcome = os.environ.get("SUMMARY_TESTS_OUTCOME", "unknown")
    bench_outcome = os.environ.get("SUMMARY_BENCHMARKS_OUTCOME", "unknown")
    test_log = os.environ.get("SUMMARY_TEST_LOG", "/tmp/test_output.log")
    bench_log = os.environ.get("SUMMARY_BENCH_LOG", "/tmp/bench_output.log")

    _out(summary, f"## Test Summary (`{runner}`)")
    _out(summary)
    _table(summary, ["Step", "Status"], [
        ["Install wheels", f"`{install_outcome}`"],
        ["Run tests", f"`{tests_outcome}`"],
        ["Run benchmarks", f"`{bench_outcome}`"],
    ])

    _write_test_results(summary, test_log)
    _write_bench_results(summary, bench_log)


def _write_test_results(summary: Path, log_path: str) -> None:
    log = Path(log_path)
    if not log.is_file():
        return

    text = log.read_text(errors="replace")
    mlir = _first_match(r"^MLIR Tests:.*", text) or "N/A"
    ir = _first_match(r"^IR Tests:.*", text) or "N/A"
    gpu = _first_match(r"^GPU Tests:.*", text) or "N/A"

    _out(summary, "### Test Results")
    _out(summary)
    _table(summary, ["Suite", "Result"], [
        ["MLIR IR (Lowering)", mlir],
        ["Python IR (Generation)", ir],
        ["GPU Execution", gpu],
    ])


def _write_bench_results(summary: Path, log_path: str) -> None:
    log = Path(log_path)
    if not log.is_file():
        return

    text = log.read_text(errors="replace")

    perf_block = _extract_perf_table(text)
    if perf_block:
        _out(summary, "### Benchmark Results")
        _out(summary)
        _out(summary, "```")
        for line in perf_block[:30]:
            _out(summary, line)
        _out(summary, "```")
        _out(summary)

    for pattern in (r"^Total:.*", r"^Success:.*", r"^Failed:.*"):
        match = _first_match(pattern, text)
        if match:
            _out(summary, match)
    _out(summary)


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

    _out(summary, "## Promote Summary")
    _out(summary)
    _table(summary, ["Item", "Value"], [
        ["Release type", f"`{release_type}`"],
        ["Source", f"`{source}`"],
        ["Destination", f"`{dest}`"],
    ])


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
