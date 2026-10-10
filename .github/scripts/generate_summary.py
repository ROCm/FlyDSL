#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Generate GitHub Actions job summaries for FlyDSL CI.

Usage:
    python3 .github/scripts/generate_summary.py build
    python3 .github/scripts/generate_summary.py test
    python3 .github/scripts/generate_summary.py promote

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
    _table(
        summary,
        ["Item", "Value"],
        [
            ["Docker image", f"`{docker_image}`"],
            ["LLVM commit", f"`{llvm_commit}`"],
            ["MLIR cache", mlir_cache],
            ["Release type", f"`{release_type}`"],
        ],
    )

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
    aiter_outcome = os.environ.get("SUMMARY_AITER_OUTCOME")
    test_log = os.environ.get("SUMMARY_TEST_LOG", "/tmp/test_output.log")
    bench_log = os.environ.get("SUMMARY_BENCH_LOG", "/tmp/bench_output.log")

    _out(summary, f"## Test Summary (`{runner}`)")
    _out(summary)
    step_rows = [
        ["Install wheels", f"`{install_outcome}`"],
        ["Run tests", f"`{tests_outcome}`"],
        ["Run benchmarks", f"`{bench_outcome}`"],
    ]
    aiter_log = os.environ.get("SUMMARY_AITER_LOG", "")
    # A gated-off step reports the literal string "skipped", which is truthy.
    aiter_ran = bool(aiter_outcome) and aiter_outcome != "skipped"
    if aiter_ran:
        step_rows.append(["Aiter CSV MoE / HGEMM", f"`{aiter_outcome}`"])
        # The comparison is informational and cannot fail the step, so put the
        # verdict in the status table rather than only in the block below.
        regress = _aiter_regress_count(aiter_log) if aiter_log else None
        if regress is not None:
            note = "none" if regress == 0 else f"**{regress}** (see table below)"
            step_rows.append(["Aiter perf regressions vs pin", note])
    _table(
        summary,
        ["Step", "Status"],
        step_rows,
    )

    _write_test_results(summary, test_log)
    _write_bench_results(summary, bench_log)
    if aiter_log and aiter_ran:
        _write_aiter_compare(summary, aiter_log)


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
    _table(
        summary,
        ["Suite", "Result"],
        [
            ["MLIR IR (Lowering)", mlir],
            ["Python IR (Generation)", ir],
            ["GPU Execution", gpu],
        ],
    )


def _write_bench_results(summary: Path, log_path: str) -> None:
    log = Path(log_path)
    if not log.is_file():
        return

    text = log.read_text(errors="replace")

    perf_block = _extract_perf_table(text)
    if perf_block:
        _write_block(summary, "### Benchmark Results", perf_block, 30)

    for pattern in (r"^Total:.*", r"^Success:.*", r"^Failed:.*"):
        match = _first_match(pattern, text)
        if match:
            _out(summary, match)
    _out(summary)


def _capture_block(text, start_pred, end_pred, inclusive_end=False) -> list[str]:
    """Return the lines from the first start_pred match to the first end_pred match.

    Without an end_pred the capture would run to the end of the log, so every
    caller must supply one.
    """
    lines: list[str] = []
    capturing = False
    for line in text.splitlines():
        if not capturing and start_pred(line):
            capturing = True
        if capturing:
            if end_pred(line):
                if inclusive_end:
                    lines.append(line)
                break
            lines.append(line)
    return lines


def _trim(lines: list[str], limit: int, tail_pred=None) -> list[str]:
    """Cap lines at limit, marking what was dropped.

    With tail_pred, the block starting at the first match is always kept: the
    aiter sweep is far longer than the cap and its verdict is the last thing in
    the block, so plain head truncation would silently drop it.
    """
    if len(lines) <= limit:
        return lines
    tail: list[str] = []
    if tail_pred is not None:
        idx = next((i for i, line in enumerate(lines) if tail_pred(line)), None)
        if idx is not None and len(lines) - idx < limit - 1:
            tail = lines[idx:]
    head = lines[: max(limit - len(tail) - 1, 0)]
    dropped = len(lines) - len(head) - len(tail)
    return head + [f"... {dropped} line(s) omitted; see the step log ..."] + tail


def _write_block(summary: Path, title: str, lines: list[str], limit: int, tail_pred=None) -> None:
    _out(summary, title)
    _out(summary)
    _out(summary, "```")
    for line in _trim(lines, limit, tail_pred):
        _out(summary, line)
    _out(summary, "```")
    _out(summary)


def _extract_perf_table(text: str) -> list[str]:
    """Return lines between the 'op' header and 'Benchmark Summary'."""
    return _capture_block(
        text,
        lambda line: line.startswith("op "),
        lambda line: "Benchmark Summary" in line,
    )


def _aiter_regress_count(log_path: str) -> int | None:
    """REGRESS count from compare_benchmark.py's summary block, if it ran."""
    log = Path(log_path)
    if not log.is_file():
        return None
    match = _first_match(r"^  REGRESS:\s+\d+", log.read_text(errors="replace"))
    return int(match.split()[-1]) if match else None


def _write_aiter_compare(summary: Path, log_path: str) -> None:
    log = Path(log_path)
    if not log.is_file():
        return
    text = log.read_text(errors="replace")
    # compare_benchmark.py ends its report with the "SKIPPED:" counter; without
    # that sentinel the capture would swallow the trailing pip output too.
    lines = _capture_block(
        text,
        lambda line: line.startswith("=== Tuned op bench:"),
        lambda line: line.startswith("  SKIPPED:"),
        inclusive_end=True,
    )
    title = "### Aiter CSV: wheel vs aiter-main flydsl pin"
    if not lines:
        # The banner comes from aiter's compare_benchmark.py, which is unpinned.
        # Say so rather than dropping the table from an otherwise green summary.
        _out(summary, title)
        _out(summary)
        _out(summary, "No comparison block found in the aiter log (banner may have changed upstream).")
        _out(summary)
        return
    _write_block(summary, title, lines, 80, tail_pred=lambda line: line.startswith("Summary:"))


# ── Promote summary ─────────────────────────────────────────────────────────


def promote_summary(summary: Path) -> None:
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    source = os.environ.get("SUMMARY_S3_SOURCE", "unknown")
    dest = os.environ.get("SUMMARY_S3_DEST", "unknown")
    wheel_names = os.environ.get("SUMMARY_WHEEL_NAMES", "").strip()

    _out(summary, "## Promote Summary")
    _out(summary)
    _table(
        summary,
        ["Item", "Value"],
        [
            ["Release type", f"`{release_type}`"],
            ["Source", f"`{source}`"],
            ["Destination", f"`{dest}`"],
        ],
    )

    if wheel_names:
        _out(summary, "### Promoted Wheels")
        _out(summary, "```")
        for whl in wheel_names.split():
            _out(summary, f"  {whl}")
        _out(summary, "```")
        _out(summary)

    domain = DOMAIN_MAP.get(release_type)
    if domain:
        index_url = f"https://{domain}/whl/gfx942-gfx950/"
        _out(summary, "### Wheels Available At")
        _out(summary, f"- {index_url}")
        _out(summary)
        _out(summary, "### Install")
        _out(summary, "```bash")
        _out(summary, f"pip install --index-url {index_url} flydsl")
        _out(summary, "```")
        _out(summary)


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
