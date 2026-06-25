#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measurement harness for the MXFP4 MoE 2-stage tuning campaign on gfx950.

The harness emits a per-point CSV that is the single reference table every
candidate is compared against.  Two measurement paths feed it:

* **Per-stage kernel-path us** comes from the FlyDSL ``tests/kernels/test_moe_gemm.py``
  benchmark, which prints ``FlyDSL MoE stage1[..]`` / ``FlyDSL MoE stage2 [..]``
  lines with per-stage us.  Combined kernel-path us = stage1 + stage2 + sorting.
* **Strict correctness + full fused-MoE e2e us** comes from the aiter
  ``op_tests/test_moe_2stage.py`` harness (``strict_accuracy``,
  ``logits_diff <= 0.01``, ``fail_on_aot_cache_miss``).  That harness times the
  whole ``fused_moe`` call as the e2e guardrail.

Every row records full provenance (GPU id+model, branch+commit, exact command,
shape, dtype+act, warmup/iters, idle-GPU check) and the resolved metric formula,
under the locked protocol in :mod:`kernels.moe_tuning_spec`.

This module keeps the parsing / metric / provenance / CSV logic as pure
functions so they are unit-testable without a GPU.  The live sweep driver
(:func:`run_point`) shells out to the two harnesses and is intended to run on the
fixed idle gfx950 node.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels import moe_tuning_spec as spec  # noqa: E402

# CSV columns: provenance first, then shape/config, then metrics.
CSV_COLUMNS = [
    # provenance
    "gpu_id",
    "gpu_model",
    "branch",
    "commit",
    "command",
    "warmup",
    "iters",
    "idle_gpu_verified",
    "graph_capture",
    "l2_flush_per_iter",
    "clocks_pinned",
    "metric_formula",
    # shape / config
    "model",
    "model_dim",
    "inter_dim",
    "experts",
    "topk",
    "dtype",
    "act",
    "token",
    "tile_m1",
    "tile_n1",
    "tile_k1",
    "tile_m2",
    "tile_n2",
    "tile_k2",
    # metrics (median + p95 over iters)
    "stage1_us",
    "stage2_us",
    "sorting_us",
    "kernel_path_us",
    "kernel_path_us_p95",
    "effective_tflops",
    "mfu",
    "e2e_us",
    "e2e_us_p95",
    "logits_diff",
    "correctness_pass",
    # failure provenance (auditable for quarantined / failing rows)
    "flydsl_command",
    "strict_error",
    "error_category",
    "aot_status",
]

METRIC_FORMULA = (
    "effective_tflops = token*model_dim*inter_dim*3*topk*2 / combined_us / 1e6; mfu = effective_tflops / 4523"
)

# Print formats from tests/kernels/test_moe_gemm.py (the first us is the median;
# an optional " p95=<v> us" suffix appears when FLYDSL_PERF_DIST is set):
#   "FlyDSL MoE stage1[fp4]: 1163.2 us, p95=1170.0 us 1654.24 TFLOPS(...), 0.377 TB/s (...)"
#   "FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | ... | 1163.2 us, p95=1170.0 us 1654.24 TFLOPS, 0.377 TB/s"
_STAGE1_RE = re.compile(r"FlyDSL MoE stage1\[[^\]]+\]:\s*([0-9.]+)\s*us")
_STAGE2_RE = re.compile(r"FlyDSL MoE stage2 \[[^\]]+\]\s+\S+\s+(atomic|reduce)\b.*?([0-9.]+)\s*us")
# Optional per-stage p95 suffix.
_STAGE1_P95_RE = re.compile(r"FlyDSL MoE stage1\[[^\]]+\]:\s*[0-9.]+\s*us,\s*p95=([0-9.]+)\s*us")
_STAGE2_P95_RE = re.compile(
    r"FlyDSL MoE stage2 \[[^\]]+\]\s+\S+\s+(?:atomic|reduce)\b.*?[0-9.]+\s*us,\s*p95=([0-9.]+)\s*us"
)
# Optional sorting print, if the FlyDSL benchmark emits one.
_SORT_RE = re.compile(r"FlyDSL MoE sort(?:ing)?[^\d]*([0-9.]+)\s*us", re.IGNORECASE)

# aiter op_tests/test_moe_2stage.py full fused_moe e2e print (line 363):
#   "ck_moe_2stages:  123.45 us,  654.00 tflops......(quant:...)"
_AITER_E2E_RE = re.compile(r"ck_moe_2stages:\s*([0-9.]+)\s*us")
# aiter logits_diff warning line (only printed when logits_diff > 1e-3).
_AITER_LOGITS_RE = re.compile(r"logits_diff[:=]\s*([0-9.eE+-]+)")
# aiter summary markdown data row: the final two numeric cells are
# ``... | <e2e us> | <logits_diff> | <model> |``.  This carries logits_diff even
# when it is below the 1e-3 warning threshold (so no warning line is printed).
_AITER_MD_ROW_RE = re.compile(r"\|\s*([0-9][0-9.eE+-]*)\s*\|\s*([0-9][0-9.eE+-]*)\s*\|\s*\w+\s*\|\s*$")
# Real correctness-miss signals: the strict-accuracy assertion or a hard error.
# NOTE: the bare ``checkAllclose ... failed!`` line is the LOOSE elementwise check
# and is EXPECTED for fp4; correctness is gated on logits_diff <= 0.01 per the
# locked contract, not on that line.
_AITER_FAIL_RE = re.compile(r"accuracy check failed|AssertionError|Traceback|RuntimeError", re.IGNORECASE)

# aiter -q quant index -> dtype alias used here (see l_quant in the harness).
DTYPE_ALIAS_TO_AITER_Q = {"a4w4": 4, "a8w4": 7}


@dataclass
class Provenance:
    """Run provenance recorded with every measured point."""

    gpu_id: str = ""
    gpu_model: str = ""
    branch: str = ""
    commit: str = ""
    warmup: int = spec.WARMUP_ITERS
    iters: int = spec.BENCH_ITERS
    idle_gpu_verified: bool = False
    graph_capture: bool = spec.GRAPH_CAPTURE
    l2_flush_per_iter: bool = spec.L2_FLUSH_PER_ITER
    # NOT proof until verified: defaults False so a row never claims pinned clocks
    # unless the driver enabled performance determinism AND verified the state.
    # (spec.CLOCKS_PINNED is the protocol's INTENT, not evidence.)
    clocks_pinned: bool = False
    metric_formula: str = METRIC_FORMULA

    REQUIRED_FIELDS = ("gpu_id", "gpu_model", "branch", "commit", "warmup", "iters")

    def missing_fields(self) -> List[str]:
        """Required provenance fields that are empty/unset (the baseline contract negative gate)."""
        missing = []
        for f in self.REQUIRED_FIELDS:
            v = getattr(self, f)
            if v in ("", None):
                missing.append(f)
        return missing

    def is_complete(self) -> bool:
        return not self.missing_fields()


@dataclass
class PointRow:
    """One per-point measurement row (provenance + shape/config + metrics)."""

    provenance: Provenance
    command: str
    model: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    dtype: str
    act: str
    token: int
    tile_m1: int = 0
    tile_n1: int = 0
    tile_k1: int = 0
    tile_m2: int = 0
    tile_n2: int = 0
    tile_k2: int = 0
    stage1_us: Optional[float] = None
    stage2_us: Optional[float] = None
    sorting_us: Optional[float] = None
    kernel_path_us: Optional[float] = None
    kernel_path_us_p95: Optional[float] = None
    effective_tflops: Optional[float] = None
    mfu: Optional[float] = None
    e2e_us: Optional[float] = None
    e2e_us_p95: Optional[float] = None
    logits_diff: Optional[float] = None
    correctness_pass: Optional[bool] = None
    flydsl_command: str = ""
    strict_error: str = ""
    error_category: str = ""
    aot_status: str = ""

    def to_csv_dict(self) -> dict:
        p = self.provenance
        row = {
            "gpu_id": p.gpu_id,
            "gpu_model": p.gpu_model,
            "branch": p.branch,
            "commit": p.commit,
            "command": self.command,
            "warmup": p.warmup,
            "iters": p.iters,
            "idle_gpu_verified": p.idle_gpu_verified,
            "graph_capture": p.graph_capture,
            "l2_flush_per_iter": p.l2_flush_per_iter,
            "clocks_pinned": p.clocks_pinned,
            "metric_formula": p.metric_formula,
        }
        for k in (
            "model",
            "model_dim",
            "inter_dim",
            "experts",
            "topk",
            "dtype",
            "act",
            "token",
            "tile_m1",
            "tile_n1",
            "tile_k1",
            "tile_m2",
            "tile_n2",
            "tile_k2",
            "stage1_us",
            "stage2_us",
            "sorting_us",
            "kernel_path_us",
            "kernel_path_us_p95",
            "effective_tflops",
            "mfu",
            "e2e_us",
            "e2e_us_p95",
            "logits_diff",
            "correctness_pass",
            "flydsl_command",
            "strict_error",
            "error_category",
            "aot_status",
        ):
            row[k] = getattr(self, k)
        return row


# --- pure parsing / metric helpers (unit-testable, no GPU) -----------------


def parse_flydsl_stage_us(stdout: str) -> dict:
    """Extract stage1 / stage2 median us and optional p95 from FlyDSL stdout.

    Returns ``{"stage1_us", "stage2_us", "stage1_p95", "stage2_p95"}`` using the
    last matching line for each stage (the benchmarked, post-warmup print).  The
    p95 fields are populated only when the FlyDSL benchmark was run with
    FLYDSL_PERF_DIST (true timed-loop distribution); otherwise None.
    """
    s1 = _STAGE1_RE.findall(stdout)
    s2 = _STAGE2_RE.findall(stdout)
    s1p = _STAGE1_P95_RE.findall(stdout)
    s2p = _STAGE2_P95_RE.findall(stdout)
    return {
        "stage1_us": float(s1[-1]) if s1 else None,
        "stage2_us": float(s2[-1][1]) if s2 else None,
        "stage1_p95": float(s1p[-1]) if s1p else None,
        "stage2_p95": float(s2p[-1]) if s2p else None,
    }


def parse_flydsl_sorting_us(stdout: str) -> Optional[float]:
    """Extract sorting us from FlyDSL stdout if present, else None (sorting is 0)."""
    m = _SORT_RE.findall(stdout)
    return float(m[-1]) if m else None


def parse_aiter_output(stdout: str) -> dict:
    """Extract e2e us, logits_diff, and correctness pass/fail from aiter stdout.

    The aiter ``op_tests/test_moe_2stage.py`` harness times the whole fused_moe
    call (the e2e guardrail) and logs ``ck_moe_2stages: <us> us``; the
    per-case ``us`` and ``logits_diff`` also appear in the final summary markdown
    row (which carries logits_diff even when it is below the 1e-3 warning
    threshold).  Correctness is gated on ``logits_diff <= 0.01`` (the locked
    contract) plus the absence of a hard assertion/error; the bare loose
    ``checkAllclose ... failed!`` line is expected for fp4 and is NOT a miss.

    ``correctness_pass`` requires an e2e number, a logits_diff, ``logits_diff <=
    0.01``, and no hard failure.
    """
    md = _AITER_MD_ROW_RE.findall(stdout)
    md_e2e = float(md[-1][0]) if md else None
    md_logits = float(md[-1][1]) if md else None

    e2e_line = _AITER_E2E_RE.findall(stdout)
    logits_line = _AITER_LOGITS_RE.findall(stdout)
    e2e_us = float(e2e_line[-1]) if e2e_line else md_e2e
    # Prefer the markdown logits cell (always present); fall back to the warning line.
    logits_diff = md_logits if md_logits is not None else (float(logits_line[-1]) if logits_line else None)

    failed = bool(_AITER_FAIL_RE.search(stdout))
    correctness_pass = (e2e_us is not None) and (logits_diff is not None) and (logits_diff <= 0.01) and (not failed)
    return {"e2e_us": e2e_us, "logits_diff": logits_diff, "correctness_pass": correctness_pass}


def parse_strict_aiter_output(stdout: str) -> dict:
    """Parse the ``STRICT_RESULT {json}`` line from ``scripts/aiter_strict_point.py``.

    Returns ``{"e2e_us", "logits_diff", "correctness_pass", "error"}``.  The strict
    runner already applies ``strict_accuracy=True`` + ``logits_diff <= 0.01``, so
    ``correctness_pass`` is authoritative; an AOT miss or strict assertion is
    reported as ``error`` with ``correctness_pass=False``.
    """
    line = None
    for ln in stdout.splitlines():
        if ln.startswith("STRICT_RESULT "):
            line = ln[len("STRICT_RESULT ") :]
    empty = {
        "e2e_us": None,
        "e2e_us_p95": None,
        "logits_diff": None,
        "correctness_pass": False,
        "error": "no_strict_result",
        "error_category": "no_result",
        "aot_status": "",
    }
    if line is None:
        return empty
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return {**empty, "error": "bad_strict_json", "error_category": "bad_json"}
    return {
        "e2e_us": d.get("e2e_us"),
        "e2e_us_p95": d.get("e2e_us_p95"),
        "logits_diff": d.get("logits_diff"),
        "correctness_pass": bool(d.get("correctness_pass")),
        "error": d.get("error", ""),
        "error_category": d.get("error_category", ""),
        "aot_status": "checked" if d.get("check_aot_cache") else "no_aot",
    }


def combined_kernel_path_us(stage1_us: float, stage2_us: float, sorting_us: float = 0.0) -> float:
    """Combined kernel-path latency = stage1 + stage2 + sorting (microseconds)."""
    return float(stage1_us) + float(stage2_us) + float(sorting_us)


def summarize(samples: List[float]) -> dict:
    """Median + p95 over a list of per-iter latencies (the locked statistics)."""
    if not samples:
        return {"median": None, "p95": None}
    ordered = sorted(samples)
    median = statistics.median(ordered)
    # Nearest-rank p95.
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return {"median": median, "p95": ordered[idx]}


def compute_metrics(*, token: int, model_dim: int, inter_dim: int, topk: int, combined_us: float) -> dict:
    """Effective TFLOPS + MFU for a combined kernel-path us, via the spec formula."""
    tflops = spec.effective_tflops(token, model_dim, inter_dim, topk, combined_us)
    return {"effective_tflops": tflops, "mfu": spec.mfu(tflops)}


# --- provenance collection (uses the host; safe no-ops when tools absent) ---


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def git_provenance(repo_root: str = _REPO_ROOT) -> dict:
    """Current branch + commit SHA of ``repo_root`` (empty strings on failure)."""
    branch = _run(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "-C", repo_root, "rev-parse", "HEAD"])
    return {"branch": branch, "commit": commit}


def gpu_provenance(gpu_id: str) -> dict:
    """GPU model name from rocm-smi for ``gpu_id`` (empty string on failure)."""
    out = _run(["rocm-smi", "--showproductname"])
    model = ""
    for line in out.splitlines():
        if "Card Series" in line:
            model = line.split(":")[-1].strip()
            break
    return {"gpu_id": str(gpu_id), "gpu_model": model}


def write_csv(rows: List[PointRow], path: str) -> None:
    """Write per-point rows to ``path`` using the fixed CSV schema."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_dict())


def read_csv(path: str) -> List[dict]:
    """Read a per-point CSV back as a list of column dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- workload run list (full the token-grid policy coverage from the spec) ------------------


@dataclass(frozen=True)
class RunPoint:
    """One (model, dtype, act, token) point in the campaign workload."""

    model: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    act: str
    dtype: str  # "a4w4" | "a8w4"
    token: int


def build_run_list() -> List[RunPoint]:
    """Every model x in-scope dtype x the token-grid policy token from ``moe_tuning_spec.MODELS``.

    This is the authoritative campaign workload; the harness sweeps exactly these
    points so coverage is the full the token-grid policy grid (not a partial manual table).
    """
    points: List[RunPoint] = []
    for m in spec.MODELS:
        for dtype in m.dtypes:
            for token in m.token_grid:
                points.append(RunPoint(m.name, m.model_dim, m.inter_dim, m.experts, m.topk, m.act, dtype, token))
    return points


def expected_point_keys() -> set:
    """The set of (model, dtype, act, token) keys the full workload must cover."""
    return {(p.model, p.dtype, p.act, str(p.token)) for p in build_run_list()}


def select_run_points(model=None, dtype=None, tokens=None) -> List[RunPoint]:
    """Filter the full run list by model / dtype / token set (for candidate sweeps).

    ``model`` and ``dtype`` are exact-match strings (None = all); ``tokens`` is an
    iterable of ints (None = the model's full grid).  Lets a reproducible candidate
    sweep target e.g. one model+dtype over chosen tokens instead of the whole grid.
    """
    tok_set = set(int(t) for t in tokens) if tokens else None
    out = []
    for rp in build_run_list():
        if model is not None and rp.model != model:
            continue
        if dtype is not None and rp.dtype != dtype:
            continue
        if tok_set is not None and rp.token not in tok_set:
            continue
        out.append(rp)
    return out


def candidate_tile_for(rp: RunPoint, overrides: dict) -> dict:
    """Tile config for a candidate sweep: the shape's default tiles with explicit
    per-key overrides applied (only keys present in ``overrides`` are changed).

    Raises ValueError if the resulting (stage1, stage2) tiles are illegal for the
    shape under the pre-compile legality filter, so a candidate sweep never spends
    GPU time on a config the kernel would reject.
    """
    from kernels import moe_tuning as _mt

    tile = dict(default_tile_for(rp))
    for k in ("tile_m1", "tile_n1", "tile_k1", "tile_n2", "tile_k2"):
        if overrides.get(k) is not None:
            tile[k] = int(overrides[k])
    a_dtype = spec.DTYPE_ALIAS_TO_A_DTYPE[rp.dtype]
    r1 = _mt.check_tile_config(
        stage=1,
        model_dim=rp.model_dim,
        inter_dim=rp.inter_dim,
        tile_m=tile["tile_m1"],
        tile_n=tile["tile_n1"],
        tile_k=tile["tile_k1"],
        a_dtype=a_dtype,
    )
    r2 = _mt.check_tile_config(
        stage=2,
        model_dim=rp.model_dim,
        inter_dim=rp.inter_dim,
        tile_m=tile["tile_m1"],
        tile_n=tile["tile_n2"],
        tile_k=tile["tile_k2"],
        a_dtype=a_dtype,
    )
    if not (r1.legal and r2.legal):
        raise ValueError(f"illegal candidate tiles for {rp.model}/{rp.dtype}: s1={r1.reason} s2={r2.reason}")
    return tile


def prepare_candidate_run(overrides: dict, model=None, dtype=None, tokens=None, prov=None, command=""):
    """Resolve a fail-closed candidate run: (run_list, per-point tiles).

    Requirements (raises ValueError, recording a machine-readable rejection for
    illegal tiles, so the caller fails closed WITHOUT writing a partial CSV):
    - at least one explicit tile override must be given (no silent default-tile
      fallback for candidate mode);
    - the selection must match at least one point;
    - EVERY selected point's tiles must pass the legality filter — the first
      illegal point aborts the whole run (a candidate run must be all-legal).

    ``prov`` (a ``Provenance``) and ``command`` (the exact top-level invocation)
    supply the run-provenance class carried by every rejected-candidate record so
    a rejection is as auditable as a measured attempt.  When ``prov`` is None the
    git branch/commit are still resolved (host-side path), so the record stays
    complete; GPU identity is then left to the caller's monkeypatch/tests.
    """
    import moe_tuning_ledger as _ledger

    if not any(v is not None for v in overrides.values()):
        raise ValueError("candidate mode requires at least one explicit --tile-* override")
    run_list = select_run_points(model=model, dtype=dtype, tokens=tokens)
    if not run_list:
        raise ValueError("candidate selection matched no points")
    # Provenance shared by every rejection from this run (filled from prov + git).
    git = git_provenance()
    base_prov = {
        "gpu_id": getattr(prov, "gpu_id", "") or "",
        "gpu_model": getattr(prov, "gpu_model", "") or "",
        "branch": getattr(prov, "branch", "") or git.get("branch", ""),
        "commit": getattr(prov, "commit", "") or git.get("commit", ""),
        "warmup": getattr(prov, "warmup", spec.WARMUP_ITERS),
        "iters": getattr(prov, "iters", spec.BENCH_ITERS),
        "command": command,
        "selection": {"model": model, "dtype": dtype, "tokens": list(tokens) if tokens else None},
    }
    tiles = []
    for rp in run_list:
        try:
            tiles.append(candidate_tile_for(rp, overrides))
        except ValueError as e:
            _ledger.append_rejected_candidate(
                {
                    **base_prov,
                    "model": rp.model,
                    "dtype": rp.dtype,
                    "act": rp.act,
                    "token": rp.token,
                    "stage": 0,  # candidate-tile rejection spans both stages; reason names the stage
                    "config": {k: overrides.get(k) for k in overrides},
                    "reason": str(e),
                    # No measured artifact exists for a pre-compile rejection, but
                    # the keys must be present to match a measured attempt's schema.
                    "csv_path": "",
                    "profile_path": "",
                }
            )
            raise ValueError(f"illegal candidate at {rp.model}/{rp.dtype} t={rp.token}: {e}") from e
    return run_list, tiles


# --- baseline validation gate (the baseline contract negative tests) ------------------------

# The locked baseline must come from this exact commit (DEC scope).
LOCKED_BASELINE_COMMIT = "523ca1c7"
# Identity/provenance fields every baseline row must carry beyond the protocol.
ROW_REQUIRED_FIELDS = ("command", "dtype", "act", "model", "token")
# Numeric metric fields every baseline row must carry, parseable as float
# (the baseline contract + the no-regression policy: per-stage, combined kernel-path median+p95, effective TFLOPS,
# MFU, and the e2e guardrail median+p95, plus the correctness logits_diff).
ROW_REQUIRED_METRIC_FIELDS = (
    "stage1_us",
    "stage2_us",
    "sorting_us",
    "kernel_path_us",
    "kernel_path_us_p95",
    "effective_tflops",
    "mfu",
    "e2e_us",
    "e2e_us_p95",
    "logits_diff",
)


def _is_float(v) -> bool:
    if v in (None, "", "None"):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def validate_baseline_row(row: dict) -> List[str]:
    """Return reasons ``row`` is NOT an acceptable locked-baseline row (empty=OK).

    Rejects rows that are not from the locked commit, not idle-GPU verified, miss
    a required provenance/identity field, miss or non-numeric any the baseline contract/the no-regression policy metric
    field (per-stage, kernel-path median+p95, effective TFLOPS, MFU, e2e
    median+p95, logits_diff), are not correctness_pass=True, or use a non-locked
    protocol (warmup/iters/graph/L2/clock).
    """
    reasons: List[str] = []

    commit = str(row.get("commit", ""))
    if not commit:
        reasons.append("missing_commit")
    elif not commit.startswith(LOCKED_BASELINE_COMMIT):
        reasons.append(f"commit_not_{LOCKED_BASELINE_COMMIT}")

    if str(row.get("idle_gpu_verified", "")).lower() not in ("true", "1"):
        reasons.append("idle_gpu_not_verified")

    for f in ("gpu_id", "gpu_model", "branch", *ROW_REQUIRED_FIELDS):
        if str(row.get(f, "")).strip() in ("", "None"):
            reasons.append(f"missing_{f}")

    # Every the baseline contract/the no-regression policy metric must be present AND numeric.
    for f in ROW_REQUIRED_METRIC_FIELDS:
        if not _is_float(row.get(f)):
            reasons.append(f"missing_{f}")

    # Correctness gate must have passed for this point.
    if str(row.get("correctness_pass", "")).lower() not in ("true", "1"):
        reasons.append("correctness_not_passed")

    # Locked protocol (the no-regression policy): warmup=10, iters=100, graph OFF, L2 flush on, clocks pinned.
    if str(row.get("warmup", "")) != str(spec.WARMUP_ITERS):
        reasons.append("warmup_mismatch")
    if str(row.get("iters", "")) != str(spec.BENCH_ITERS):
        reasons.append("iters_mismatch")
    if str(row.get("graph_capture", "")).lower() not in ("false", "0"):
        reasons.append("graph_capture_must_be_off")
    if str(row.get("l2_flush_per_iter", "")).lower() not in ("true", "1"):
        reasons.append("l2_flush_must_be_on")
    if str(row.get("clocks_pinned", "")).lower() not in ("true", "1"):
        reasons.append("clocks_must_be_pinned")
    return reasons


def validate_baseline_csv(path: str, expected_keys: Optional[set] = None) -> dict:
    """Validate every row of a baseline CSV and that coverage equals the workload.

    Returns ``{"valid": bool, "row_errors": {key: [reasons]}, "missing_points":
    [...], "n_rows": int}``.  A baseline is valid only if every row that belongs
    to ``expected_keys`` passes :func:`validate_baseline_row` AND all
    ``expected_keys`` points are present.

    ``expected_keys`` defaults to the full the token-grid policy workload
    (:func:`expected_point_keys`).  Pass a subset (e.g.
    ``moe_tuning_spec.validated_point_keys()``) to validate the correctness-passing
    subset independently of the quarantined a8w4 shapes.  Rows outside
    ``expected_keys`` are ignored (neither required nor cause errors).
    """
    if expected_keys is None:
        expected_keys = expected_point_keys()
    rows = read_csv(path)
    row_errors: Dict[str, list] = {}
    seen = set()
    for row in rows:
        key = (row.get("model"), row.get("dtype"), row.get("act"), row.get("token"))
        if key not in expected_keys:
            continue  # quarantined / out-of-subset row: not validated here.
        seen.add(key)
        errs = validate_baseline_row(row)
        if errs:
            row_errors[str(key)] = errs
    missing = sorted(str(k) for k in (expected_keys - seen))
    valid = not row_errors and not missing
    return {"valid": valid, "row_errors": row_errors, "missing_points": missing, "n_rows": len(rows)}


# --- live measurement (runs on the gfx950 node) ----------------------------


def check_idle_gpu(gpu_id: str, busy_pct_threshold: int = 5) -> bool:
    """True if the GPU's utilization is below ``busy_pct_threshold`` (idle check)."""
    out = _run(["rocm-smi", "-d", str(gpu_id), "--showuse"])
    for line in out.splitlines():
        m = re.search(r"GPU use \(%\)\s*:?\s*([0-9]+)", line)
        if m:
            return int(m.group(1)) < busy_pct_threshold
    # If utilization could not be read, do not claim idle.
    return False


# Locked sclk to pin for the measurement protocol (this node's max, MHz).
PINNED_SCLK_MHZ = 2200


def pin_clocks(gpu_id: str, sclk_mhz: int = PINNED_SCLK_MHZ) -> bool:
    """Enable performance determinism (pin sclk) so the recorded
    ``clocks_pinned`` flag is truthful, not aspirational.

    Returns True if determinism was enabled (rocm-smi reports success), else
    False (e.g. the container forbids it).  DVFS auto-scaling is the dominant
    source of small-token run-to-run jitter; pinning is the in-protocol way to
    reduce it without changing the no-regression band.
    """
    out = _run(["rocm-smi", "-d", str(gpu_id), "--setperfdeterminism", str(sclk_mhz)])
    return "performance determinism" in out.lower() and "successfully" in out.lower()


def clocks_pinned_state(gpu_id: str) -> bool:
    """True if the GPU performance level is a pinned/deterministic mode (not auto)."""
    out = _run(["rocm-smi", "-d", str(gpu_id), "--showperflevel"]).lower()
    # "determinism" or "manual"/"high" indicate a pinned level; "auto" is DVFS.
    return ("determinism" in out) or ("manual" in out) or ("high" in out)


def setup_run_provenance(gpu_id: str, assume_idle: bool = False, repo_ref: str = _REPO_ROOT) -> Provenance:
    """Build the run Provenance with VERIFIED idle + clock-pinned state.

    Enables performance determinism (pins sclk) and verifies it via
    ``clocks_pinned_state``; ``Provenance.clocks_pinned`` reflects only the
    verified state (never the static intent default).  Used by the live sweep so
    every emitted row's clock provenance is trustworthy.
    """
    idle = True if assume_idle else check_idle_gpu(gpu_id)
    pin_clocks(gpu_id)  # best-effort enable
    pinned = clocks_pinned_state(gpu_id)  # verify the actual state
    prov = Provenance(idle_gpu_verified=idle, clocks_pinned=pinned)
    prov.__dict__.update(git_provenance(repo_ref))
    prov.__dict__.update(gpu_provenance(gpu_id))
    return prov


def _flydsl_cmd(rp: RunPoint, gpu_id: str, tile: dict) -> List[str]:
    """FlyDSL per-stage benchmark command for one point under the locked protocol."""
    in_dtype = "fp4" if rp.dtype == "a4w4" else "a8w4"
    return [
        "python3",
        os.path.join(_REPO_ROOT, "tests", "kernels", "test_moe_gemm.py"),
        "--in_dtype",
        in_dtype,
        "-dim",
        f"{rp.model_dim},{rp.inter_dim}",
        "-t",
        str(rp.token),
        "-e",
        str(rp.experts),
        "-k",
        str(rp.topk),
        "--num_warmup",
        str(spec.WARMUP_ITERS),
        "--num_iters",
        str(spec.BENCH_ITERS),
        "--tile_m",
        str(tile["tile_m1"]),
        "--tile_n",
        str(tile["tile_n1"]),
        "--tile_k",
        str(tile["tile_k1"]),
        "--tile_n2",
        str(tile["tile_n2"]),
        "--tile_k2",
        str(tile["tile_k2"]),
        "--skip_ref",
        "true",
        "--compare_aiter_ck",
        "false",
    ]


AITER_REPO = "/sgl-workspace/aiter"
# Default gate mode per quant alias for the strict aiter guardrail.  a4w4 uses
# SEPARATED (validated correct); a8w4 is quarantined (see moe_tuning_spec) so its
# gate choice is recorded but never gates a win.
DTYPE_ALIAS_TO_GATE = {"a4w4": "separated", "a8w4": "interleave"}


def _aiter_cmd(rp: RunPoint, check_aot: bool = True) -> List[str]:
    """Strict, AOT-checked, model-correct single-case aiter guardrail command.

    Invokes ``scripts/aiter_strict_point.py`` which calls aiter ``test_fmoe`` with
    the model's TRUE activation and gate mode, ``strict_accuracy=True``, the
    AOT-cache-wrapped variant (``check_aot`` -> ``fail_on_aot_cache_miss``), and
    the locked e2e protocol (warmup=10/iters=100 injected over aiter's internal
    2/5).  This is NOT the aiter legacy CLI (which is non-strict, non-AOT, and
    hardcodes Swiglu/INTERLEAVE for the fp8xfp4 case).
    """
    aq = spec.DTYPE_ALIAS_TO_A_DTYPE[rp.dtype]  # a4w4->fp4, a8w4->fp8
    gate = DTYPE_ALIAS_TO_GATE[rp.dtype]
    cmd = [
        "python3",
        os.path.join(_REPO_ROOT, "scripts", "aiter_strict_point.py"),
        "--model-dim",
        str(rp.model_dim),
        "--inter-dim",
        str(rp.inter_dim),
        "-e",
        str(rp.experts),
        "-k",
        str(rp.topk),
        "-t",
        str(rp.token),
        "--aq",
        aq,
        "--wq",
        "fp4",
        "--act",
        rp.act,
        "--gate",
        gate,
        "--warmup",
        str(spec.WARMUP_ITERS),
        "--iters",
        str(spec.BENCH_ITERS),
        "--aiter-repo",
        AITER_REPO,
    ]
    if not check_aot:
        cmd.append("--no-aot")
    return cmd


def _exec(cmd: List[str], gpu_id: str, extra_env: Optional[dict] = None) -> str:
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        return (out.stdout or "") + "\n" + (out.stderr or "")
    except Exception as e:  # pragma: no cover - live-run only
        return f"HARNESS_EXEC_ERROR: {e}"


def run_point(
    rp: RunPoint,
    tile: dict,
    gpu_id: str,
    provenance: Provenance,
    measure_e2e: bool = True,
    reps: int = 3,
    check_aot: bool = True,
) -> PointRow:  # pragma: no cover - exercised only on the gfx950 node
    """Measure one workload point: FlyDSL per-stage us + aiter e2e/correctness.

    ``tile`` carries tile_m1/n1/k1 and tile_n2/k2 (stage1 + stage2 tiles).  The
    combined kernel-path us = stage1 + stage2 + sorting; the aiter run supplies
    the e2e guardrail us, logits_diff, and correctness pass/fail.

    Median + p95 come from the TRUE timed loop inside each subprocess: the FlyDSL
    benchmark runs with ``FLYDSL_PERF_DIST=1`` (per-iteration median+p95 over
    ``iters``) and the strict aiter runner times fused_moe per iteration.  ``reps``
    here is just how many independent subprocess samples to take of the median; the
    per-point p95 is the timed-loop p95 (median of the per-rep p95 values), NOT a
    dispersion across reps.  ``flydsl_command``, ``strict_error``,
    ``error_category``, and ``aot_status`` are recorded for auditability.

    ``check_aot`` gates the strict aiter AOT-cache check; when False the e2e still
    runs strict+correct but does not require a pre-populated AOT cache (recorded as
    ``aot_status="no_aot"``).  ``command`` names ONLY the commands actually executed
    for this row: the aiter command is appended only when ``measure_e2e`` is True.
    """
    flydsl_cmd = _flydsl_cmd(rp, gpu_id, tile)
    aiter_cmd = _aiter_cmd(rp, check_aot=check_aot)
    # The FlyDSL benchmark must emit its true per-iteration distribution; the env
    # is part of the reproducible command provenance (a replay must set it too).
    flydsl_env = {"FLYDSL_PERF_DIST": "1"}
    env_prefix = f"HIP_VISIBLE_DEVICES={gpu_id} FLYDSL_PERF_DIST=1 "
    flydsl_command_str = env_prefix + " ".join(flydsl_cmd)
    # Only name commands that actually run for this row (truthful provenance).
    command = flydsl_command_str
    if measure_e2e:
        command += " ; " + f"HIP_VISIBLE_DEVICES={gpu_id} " + " ".join(aiter_cmd)

    s1_samples, s2_samples, sort_samples, combined_samples = [], [], [], []
    s1_p95s, s2_p95s = [], []
    for _ in range(max(1, reps)):
        out = _exec(flydsl_cmd, gpu_id, extra_env=flydsl_env)
        stages = parse_flydsl_stage_us(out)
        if stages["stage1_us"] is None or stages["stage2_us"] is None:
            continue
        srt = parse_flydsl_sorting_us(out) or 0.0
        s1_samples.append(stages["stage1_us"])
        s2_samples.append(stages["stage2_us"])
        sort_samples.append(srt)
        combined_samples.append(combined_kernel_path_us(stages["stage1_us"], stages["stage2_us"], srt))
        if stages["stage1_p95"] is not None:
            s1_p95s.append(stages["stage1_p95"])
        if stages["stage2_p95"] is not None:
            s2_p95s.append(stages["stage2_p95"])

    e2e_samples, e2e_p95s, logits_samples, correctness = [], [], [], None
    strict_error, error_category, aot_status = "", "", ""
    if measure_e2e:
        for _ in range(max(1, reps)):
            res = parse_strict_aiter_output(_exec(aiter_cmd, gpu_id))
            if res["e2e_us"] is not None:
                e2e_samples.append(res["e2e_us"])
            if res.get("e2e_us_p95") is not None:
                e2e_p95s.append(res["e2e_us_p95"])
            if res["logits_diff"] is not None:
                logits_samples.append(res["logits_diff"])
            rep_ok = res["correctness_pass"]
            correctness = rep_ok if correctness is None else (correctness and bool(rep_ok))
            # keep the last rep's failure provenance (representative).
            strict_error = res.get("error", "") or strict_error
            error_category = res.get("error_category", "") or error_category
            aot_status = res.get("aot_status", "") or aot_status

    row = PointRow(
        provenance=provenance,
        command=command,
        model=rp.model,
        model_dim=rp.model_dim,
        inter_dim=rp.inter_dim,
        experts=rp.experts,
        topk=rp.topk,
        dtype=rp.dtype,
        act=rp.act,
        token=rp.token,
        tile_m1=tile["tile_m1"],
        tile_n1=tile["tile_n1"],
        tile_k1=tile["tile_k1"],
        tile_m2=tile["tile_m1"],
        tile_n2=tile["tile_n2"],
        tile_k2=tile["tile_k2"],
        flydsl_command=flydsl_command_str,
        strict_error=strict_error,
        error_category=error_category,
        aot_status=aot_status,
    )
    if combined_samples:
        row.stage1_us = summarize(s1_samples)["median"]
        row.stage2_us = summarize(s2_samples)["median"]
        row.sorting_us = summarize(sort_samples)["median"]
        row.kernel_path_us = summarize(combined_samples)["median"]
        # p95 is the timed-loop p95 (median across the per-rep timed-loop p95s);
        # fall back to the across-rep combined p95 only if the timed-loop p95 is
        # unavailable.
        if s1_p95s and s2_p95s:
            row.kernel_path_us_p95 = (
                summarize(s1_p95s)["median"] + summarize(s2_p95s)["median"] + summarize(sort_samples)["median"]
            )
        else:
            row.kernel_path_us_p95 = summarize(combined_samples)["p95"]
        m = compute_metrics(
            token=rp.token, model_dim=rp.model_dim, inter_dim=rp.inter_dim, topk=rp.topk, combined_us=row.kernel_path_us
        )
        row.effective_tflops = m["effective_tflops"]
        row.mfu = m["mfu"]
    if e2e_samples:
        row.e2e_us = summarize(e2e_samples)["median"]
        row.e2e_us_p95 = summarize(e2e_p95s)["median"] if e2e_p95s else summarize(e2e_samples)["p95"]
    if logits_samples:
        row.logits_diff = max(logits_samples)  # worst-case correctness across reps
    row.correctness_pass = correctness
    return row


def row_missing_kernel_path(row: "PointRow") -> bool:
    """True if a measured row has no parseable kernel-path timing.

    The FlyDSL benchmark emits no stage times for some tile shapes (e.g. the
    tile_k1!=256 / tile_n1=512 harness limitation): the subprocess returns but
    ``parse_flydsl_stage_us`` finds nothing, so the row's stage/kernel-path fields
    stay ``None``.  Such a row is NOT a measurement and must never be recorded as a
    ``loss`` -- candidate mode treats it as a fail-closed rejected measurement.
    """
    return row.stage1_us is None or row.stage2_us is None or row.kernel_path_us is None


# Default (baseline) tile config per shape: matches scripts/run_benchmark.sh.
def default_tile_for(rp: RunPoint) -> dict:  # pragma: no cover - simple table
    if rp.model_dim == 3072:  # GPT-OSS
        return {"tile_m1": 32, "tile_n1": 128, "tile_k1": 256, "tile_n2": 256, "tile_k2": 256}
    return {"tile_m1": 64, "tile_n1": 256, "tile_k1": 256, "tile_n2": 256, "tile_k2": 256}


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI/live
    import argparse

    ap = argparse.ArgumentParser(description="MXFP4 MoE tuning measurement harness (gfx950)")
    ap.add_argument("mode", choices=["baseline", "candidate", "validate", "list"])
    ap.add_argument("--gpu", default=os.environ.get("GPU", "0"), help="GPU id (HIP_VISIBLE_DEVICES)")
    ap.add_argument("--out", default="", help="output CSV path")
    ap.add_argument("--csv", default="", help="CSV to validate (validate mode)")
    ap.add_argument("--no-e2e", action="store_true", help="skip the aiter e2e/correctness run")
    ap.add_argument(
        "--no-aot-check",
        action="store_true",
        help="run e2e strict+correct but do not require a pre-populated AOT cache (records aot_status=no_aot)",
    )
    ap.add_argument("--assume-idle", action="store_true", help="skip the live idle-GPU probe")
    ap.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="proceed (recording clocks_pinned=False) even if clock pinning cannot be verified",
    )
    # Candidate-mode selection + explicit tile overrides (reproducible sweeps).
    ap.add_argument("--model", default=None, help="restrict to one model (candidate mode)")
    ap.add_argument("--dtype", default=None, help="restrict to one dtype alias, e.g. a4w4 (candidate mode)")
    ap.add_argument("--tokens", default=None, help="comma/space-separated token list (candidate mode)")
    ap.add_argument("--reps", type=int, default=3, help="independent subprocess reps per point")
    for _k in ("tile-m1", "tile-n1", "tile-k1", "tile-n2", "tile-k2"):
        ap.add_argument(f"--{_k}", type=int, default=None, help=f"candidate {_k.replace('-', '_')} override")
    args = ap.parse_args(argv)

    if args.mode == "list":
        for rp in build_run_list():
            print(rp)
        return 0

    if args.mode == "validate":
        res = validate_baseline_csv(args.csv)
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    prov = setup_run_provenance(args.gpu, assume_idle=args.assume_idle)
    print(f"clocks_pinned (verified)={prov.clocks_pinned} idle_gpu_verified={prov.idle_gpu_verified}")
    # The locked protocol requires fixed clocks: if verification failed, do not
    # emit a baseline that falsely claims pinned clocks.
    if spec.CLOCKS_PINNED and not prov.clocks_pinned and not args.allow_unpinned:
        print(
            "ERROR: locked protocol requires pinned clocks but verification failed; "
            "the run would be non-comparable. Re-run with the GPU clocks pinnable, "
            "or pass --allow-unpinned to record clocks_pinned=False explicitly.",
            file=sys.stderr,
        )
        return 2

    overrides = {
        "tile_m1": args.tile_m1,
        "tile_n1": args.tile_n1,
        "tile_k1": args.tile_k1,
        "tile_n2": args.tile_n2,
        "tile_k2": args.tile_k2,
    }

    if args.mode == "candidate":
        toks = [int(t) for t in args.tokens.replace(",", " ").split()] if args.tokens else None
        top_command = "python3 " + shlex.join([os.path.relpath(__file__, _REPO_ROOT), *(argv or sys.argv[1:])])
        try:
            run_list, tiles = prepare_candidate_run(
                overrides, model=args.model, dtype=args.dtype, tokens=toks, prov=prov, command=top_command
            )
        except ValueError as e:
            # Fail closed: do not write a partial CSV; rejection already recorded.
            print(f"ERROR: candidate run rejected: {e}", file=sys.stderr)
            return 2
        rows = [
            run_point(
                rp,
                tiles[i],
                args.gpu,
                prov,
                measure_e2e=not args.no_e2e,
                reps=args.reps,
                check_aot=not args.no_aot_check,
            )
            for i, rp in enumerate(run_list)
        ]
        # Fail closed on unmeasured rows: a missing kernel-path row is NOT a loss.
        import moe_tuning_ledger as _ledger

        bad = [(rp, tiles[i], r) for i, (rp, r) in enumerate(zip(run_list, rows)) if row_missing_kernel_path(r)]
        if bad:
            for rp, tile, r in bad:
                _ledger.append_rejected_candidate(
                    {
                        "model": rp.model,
                        "dtype": rp.dtype,
                        "act": rp.act,
                        "token": rp.token,
                        "stage": 1,
                        "config": {k: tile.get(k) for k in ("tile_m1", "tile_n1", "tile_k1", "tile_n2", "tile_k2")},
                        "reason": "no parseable kernel-path stage times emitted (unmeasured shape; e.g. "
                        "tile_k1!=256 / tile_n1=512 harness limitation)",
                        "selection": {"model": args.model, "dtype": args.dtype, "tokens": toks},
                        "gpu_id": prov.gpu_id,
                        "gpu_model": prov.gpu_model,
                        "branch": prov.branch,
                        "commit": prov.commit,
                        "command": top_command,
                        "warmup": prov.warmup,
                        "iters": prov.iters,
                        "csv_path": "",
                        "profile_path": "",
                    }
                )
            print(
                f"ERROR: {len(bad)} candidate point(s) produced no kernel-path measurement; "
                "recorded as rejected measurements, no CSV written.",
                file=sys.stderr,
            )
            return 2
    else:  # baseline: full grid, default tiles
        run_list = build_run_list()
        rows = [
            run_point(
                rp,
                default_tile_for(rp),
                args.gpu,
                prov,
                measure_e2e=not args.no_e2e,
                reps=args.reps,
                check_aot=not args.no_aot_check,
            )
            for rp in run_list
        ]

    out = args.out or f"/tmp/moe_{args.mode}.csv"
    write_csv(rows, out)
    print(f"wrote {len(rows)} rows -> {out}")
    return 0


__all__ = [
    "CSV_COLUMNS",
    "METRIC_FORMULA",
    "LOCKED_BASELINE_COMMIT",
    "Provenance",
    "PointRow",
    "RunPoint",
    "parse_flydsl_stage_us",
    "parse_flydsl_sorting_us",
    "parse_aiter_output",
    "parse_strict_aiter_output",
    "combined_kernel_path_us",
    "summarize",
    "compute_metrics",
    "git_provenance",
    "gpu_provenance",
    "check_idle_gpu",
    "pin_clocks",
    "clocks_pinned_state",
    "setup_run_provenance",
    "build_run_list",
    "expected_point_keys",
    "select_run_points",
    "candidate_tile_for",
    "prepare_candidate_run",
    "default_tile_for",
    "validate_baseline_row",
    "validate_baseline_csv",
    "run_point",
    "row_missing_kernel_path",
    "write_csv",
    "read_csv",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
