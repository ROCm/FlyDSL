#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attempt ledger + Pareto comparison for the MXFP4 MoE tuning campaign.

Every candidate attempt — win or loss — is appended to ``docs/attempts.jsonl``
with full provenance (config, stage, model, dtype, act, GPU id+model,
branch+commit, command, warmup/iters, CSV/profile path, result).  A human-facing
running log lives in ``docs/optimization-ledger.md``.

The Pareto comparison takes a baseline per-point CSV and a candidate per-point
CSV (both emitted by ``scripts/moe_tuning_harness.py``) and reports, per point,
whether the candidate is a win / regression / neutral under the locked the win-margin policy /
the no-regression policy predicates.  A win is only claimable when no point regresses on either the
kernel-path or e2e metric (no Pareto regression) and the re-run-stability rule
holds.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels import moe_tuning_spec as spec  # noqa: E402

ATTEMPTS_JSONL = os.path.join(_REPO_ROOT, "docs", "attempts.jsonl")
LEDGER_MD = os.path.join(_REPO_ROOT, "docs", "optimization-ledger.md")

# Required provenance keys for any ledger attempt (the ledger contract).
REQUIRED_ATTEMPT_FIELDS = (
    "config",
    "stage",
    "model",
    "dtype",
    "act",
    "gpu_id",
    "gpu_model",
    "branch",
    "commit",
    "command",
    "warmup",
    "iters",
    "result",
)

# A rejected search candidate never reaches compile/GPU, so it has no measured
# metrics (csv_path/profile_path stay empty), but it MUST still carry the same
# identity + run-provenance class as a measured attempt so the rejection is
# auditable (the rejected-candidate ledger contract).  ``stage`` is 0 when the
# rejection is at the candidate-tile level spanning both stages; the reason
# string still names the offending stage.  ``selection`` records the run's
# model/dtype/tokens filter so the rejection is reproducible.
REQUIRED_REJECTED_FIELDS = (
    "model",
    "dtype",
    "act",
    "token",
    "stage",
    "config",
    "reason",
    "selection",
    "gpu_id",
    "gpu_model",
    "branch",
    "commit",
    "command",
    "warmup",
    "iters",
)

# Keys that must be PRESENT on a rejected record but may legitimately be empty
# strings: a pre-compile rejection produces no measured CSV/profile artifact, yet
# the keys must exist so the record schema matches a measured attempt.
REQUIRED_REJECTED_PRESENT_KEYS = (
    "csv_path",
    "profile_path",
)


@dataclass
class Attempt:
    """One tuning attempt record (win or loss)."""

    config: dict
    stage: int
    model: str
    dtype: str
    act: str
    gpu_id: str
    gpu_model: str
    branch: str
    commit: str
    command: str
    warmup: int
    iters: int
    result: str  # "win" | "loss" | "rejected" | "neutral"
    csv_path: str = ""
    profile_path: str = ""
    note: str = ""
    timestamp: Optional[float] = None

    def missing_fields(self) -> List[str]:
        return [f for f in REQUIRED_ATTEMPT_FIELDS if getattr(self, f, None) in ("", None)]


def append_attempt(attempt: Attempt, path: str = ATTEMPTS_JSONL, now: Optional[float] = None) -> dict:
    """Append an attempt to the JSONL ledger.

    Raises ``ValueError`` if any required provenance field is missing, so a win
    can never be recorded without complete provenance (the ledger contract negative gate).
    """
    missing = attempt.missing_fields()
    if missing:
        raise ValueError(f"attempt missing required provenance fields: {missing}")
    rec = asdict(attempt)
    rec["timestamp"] = now if now is not None else time.time()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def append_rejected_candidate(record: dict, path: str = ATTEMPTS_JSONL, now: float = None) -> dict:
    """Append a machine-readable rejected-candidate record to the JSONL ledger.

    ``record`` must carry the full provenance class (``REQUIRED_REJECTED_FIELDS``)
    so a rejected search candidate is as auditable as a measured attempt — even
    though it never reached compile/GPU.  The measured-artifact keys
    (``REQUIRED_REJECTED_PRESENT_KEYS``: ``csv_path``/``profile_path``) must be
    present but may be empty strings (no artifact exists pre-compile).  Raises
    ``ValueError`` if any required field is missing, so an incomplete rejection can
    never be recorded (the rejected-candidate contract negative gate).
    """
    # Treat only None / "" as missing — integer 0 (stage, warmup, iters) is valid.
    missing = [k for k in REQUIRED_REJECTED_FIELDS if record.get(k) in (None, "")]
    # Artifact keys must EXIST (empty string allowed); only a truly absent key fails.
    missing += [k for k in REQUIRED_REJECTED_PRESENT_KEYS if k not in record]
    if missing:
        raise ValueError(f"rejected-candidate record missing fields: {missing}")
    # selection must be a non-empty dict so the rejection's run filter is recorded.
    sel = record.get("selection")
    if not isinstance(sel, dict) or not sel:
        raise ValueError("rejected-candidate record 'selection' must be a non-empty dict")
    rec = {"result": "rejected_candidate", **record}
    rec["timestamp"] = now if now is not None else time.time()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def read_point_csv(path: str) -> Dict[Tuple, dict]:
    """Read a per-point harness CSV keyed by (model, dtype, token, stage tiles).

    The key is (model, dtype, act, token) — the comparison axis between baseline
    and candidate at one shape/token point.
    """
    table: Dict[Tuple, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("model"), row.get("dtype"), row.get("act"), row.get("token"))
            table[key] = row
    return table


def _f(row: dict, col: str) -> Optional[float]:
    v = row.get(col)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class PointVerdict:
    key: Tuple
    token: int
    kernel_path_regression: bool = False
    e2e_regression: bool = False
    large_shape_win: bool = False
    small_token_win: bool = False
    note: str = ""


def compare_point(baseline: dict, candidate: dict) -> PointVerdict:
    """Apply the win-margin policy / the no-regression policy predicates to one (baseline, candidate) point pair."""
    token = int(float(candidate.get("token") or baseline.get("token") or 0))
    key = (candidate.get("model"), candidate.get("dtype"), candidate.get("act"), candidate.get("token"))
    v = PointVerdict(key=key, token=token)

    b_kp, c_kp = _f(baseline, "kernel_path_us"), _f(candidate, "kernel_path_us")
    b_e2e, c_e2e = _f(baseline, "e2e_us"), _f(candidate, "e2e_us")
    b_mfu, c_mfu = _f(baseline, "mfu"), _f(candidate, "mfu")

    if b_kp is not None and c_kp is not None:
        v.kernel_path_regression = spec.is_regression(b_kp, c_kp, token=token)
    if b_e2e is not None and c_e2e is not None:
        v.e2e_regression = spec.is_regression(b_e2e, c_e2e, token=token)

    if spec.is_large_token(token) and token in spec.MFU_TARGET_BUCKETS:
        if b_mfu is not None and c_mfu is not None:
            v.large_shape_win = spec.is_large_shape_win(b_mfu, c_mfu)
    if spec.is_small_token(token):
        if b_kp is not None and c_kp is not None:
            v.small_token_win = spec.is_small_token_win(b_kp, c_kp)
    return v


def _required_fields_for_point(token: int) -> Tuple[str, ...]:
    """Comparison fields a candidate row must carry for its token regime.

    Every point needs both latency metrics; large target buckets additionally
    need ``mfu`` (the large-shape win/regression axis).
    """
    fields = ["kernel_path_us", "e2e_us"]
    if spec.is_large_token(token) and token in spec.MFU_TARGET_BUCKETS:
        fields.append("mfu")
    return tuple(fields)


def _row_missing_fields(row: dict, fields: Tuple[str, ...]) -> List[str]:
    return [f for f in fields if _f(row, f) is None]


@dataclass
class CampaignVerdict:
    points: List[PointVerdict] = field(default_factory=list)
    any_regression: bool = False
    large_wins: List[Tuple] = field(default_factory=list)
    small_wins: List[Tuple] = field(default_factory=list)
    missing_candidate_points: List[Tuple] = field(default_factory=list)
    incomplete_points: List[Tuple] = field(default_factory=list)
    # Strict correctness + AOT-cache hard gate over the candidate CSV
    # (``selected_candidate_gate`` output).  Populated by ``compare_csvs``; a
    # candidate that fails this gate (e.g. ``aot_status=no_aot``) can never be a
    # claimable win even if its metrics look winning.
    gate: dict = field(default_factory=lambda: {"passed": False, "n_rows": 0, "violations": []})

    @property
    def coverage_complete(self) -> bool:
        """True only if every baseline point has a candidate row with all the
        regime-required comparison fields present (no cherry-picking)."""
        return not self.missing_candidate_points and not self.incomplete_points

    @property
    def pareto_clean(self) -> bool:
        """True only if coverage is complete AND no point regressed on kernel-path
        or e2e.  Incomplete/cherry-picked candidate CSVs can never be clean."""
        return self.coverage_complete and not self.any_regression

    @property
    def claimable_win(self) -> bool:
        """The SINGLE source of truth for whether a candidate may be promoted to a
        win.  True only when ALL hold:
        - ``pareto_clean`` (full coverage + no kernel-path/e2e regression),
        - at least one target-bucket or small-token win is present, and
        - the strict correctness + AOT-cache hard gate passed
          (``aot_status=checked`` + correctness + ``logits_diff<=0.01`` on every
          row) -- so a ``no_aot`` / failed-correctness candidate is never claimable
          regardless of how good its metrics look.
        Re-run stability is enforced separately by re-running and re-comparing."""
        return self.pareto_clean and bool(self.large_wins or self.small_wins) and bool(self.gate.get("passed"))


def compare_csvs(baseline_csv: str, candidate_csv: str) -> CampaignVerdict:
    """Full per-point Pareto comparison of a candidate vs the locked baseline.

    Iterates the COMPLETE baseline key set so a candidate cannot pass by omitting
    a regressing/uncovered point.  A point with a missing candidate row, or whose
    candidate row lacks a regime-required field (kernel_path_us/e2e_us for every
    point; mfu for large target buckets), makes ``coverage_complete`` False, which
    forces ``pareto_clean`` False.

    The candidate is run through ``selected_candidate_gate`` and the result is
    stored on the verdict.  ``CampaignVerdict.claimable_win`` is the single source
    of truth for promotability: it requires ``pareto_clean`` + at least one win +
    the gate (``aot_status=checked`` + correctness + ``logits_diff<=0.01``).  Do
    NOT promote a candidate from ``pareto_clean`` + win lists alone -- a ``no_aot``
    candidate can be pareto_clean with wins yet must not be claimable.
    """
    base = read_point_csv(baseline_csv)
    cand = read_point_csv(candidate_csv)
    cv = CampaignVerdict()
    cv.gate = selected_candidate_gate(candidate_csv)
    for key, b_row in base.items():
        token = int(float(b_row.get("token") or 0))
        c_row = cand.get(key)
        if c_row is None:
            cv.missing_candidate_points.append(key)
            cv.points.append(PointVerdict(key=key, token=token, note="missing_candidate_point"))
            continue
        missing = _row_missing_fields(c_row, _required_fields_for_point(token))
        if missing:
            cv.incomplete_points.append(key)
            cv.points.append(PointVerdict(key=key, token=token, note="missing_fields:" + ",".join(missing)))
            continue
        pv = compare_point(b_row, c_row)
        cv.points.append(pv)
        if pv.kernel_path_regression or pv.e2e_regression:
            cv.any_regression = True
        if pv.large_shape_win:
            cv.large_wins.append(key)
        if pv.small_token_win:
            cv.small_wins.append(key)
    return cv


def selected_candidate_gate(candidate_csv: str, max_logits_diff: float = 0.01) -> dict:
    """Hard gate a candidate CSV before it can be promoted to a win.

    A selected candidate must clear the strict correctness + AOT-cache hard gate on
    EVERY row: ``aot_status == "checked"`` (the strict aiter run required a
    pre-populated AOT cache, not the ``no_aot`` repeatability/diagnostic bypass),
    ``correctness_pass`` is true, and ``logits_diff <= max_logits_diff``.  Rows
    measured with ``--no-aot-check`` (``aot_status == "no_aot"``) are valid for
    NEUTRAL repeatability/diagnostic artifacts but can never be promoted to a win,
    so they fail this gate.

    Returns ``{"passed": bool, "n_rows": int, "violations": [(key, reason), ...]}``.
    ``passed`` is False if there are zero rows (nothing to promote) or any violation.
    """
    rows = read_point_csv(candidate_csv)
    violations: List[Tuple] = []
    for key, row in rows.items():
        aot = (row.get("aot_status") or "").strip()
        if aot != "checked":
            violations.append((key, f"aot_status={aot or 'missing'} (need 'checked')"))
        cp = (row.get("correctness_pass") or "").strip().lower()
        if cp not in ("true", "1"):
            violations.append((key, f"correctness_pass={row.get('correctness_pass')!r} (need True)"))
        ld = _f(row, "logits_diff")
        if ld is None:
            violations.append((key, "logits_diff missing"))
        elif ld > max_logits_diff:
            violations.append((key, f"logits_diff={ld} > {max_logits_diff}"))
    return {"passed": bool(rows) and not violations, "n_rows": len(rows), "violations": violations}


def repeatability_check(csv_a: str, csv_b: str) -> dict:
    """Compare two independent sweeps of the SAME config under the no-regression policy.

    For each shared (model, dtype, act, token) point, a metric is "stable" if the
    two runs agree within the no-regression noise band (NOT a regression in either
    direction): ``|b - a| <= max(a*REGRESSION_REL, abs_floor_us(token))``, where
    the absolute floor is regime-aware (8 us for tokens <= SMALL_TOKEN_MAX, 2 us
    otherwise).  Returns the set of unstable points per metric; an empty unstable
    set demonstrates the harness is repeatable (the measurement protocol).
    """
    a = read_point_csv(csv_a)
    b = read_point_csv(csv_b)
    shared = sorted(set(a) & set(b))
    unstable = {"kernel_path_us": [], "e2e_us": []}

    def band(x, token):
        return max(abs(x) * spec.REGRESSION_REL, spec.abs_floor_us(token))

    for key in shared:
        token = int(float(a[key].get("token") or 0))
        for metric in ("kernel_path_us", "e2e_us"):
            va, vb = _f(a[key], metric), _f(b[key], metric)
            if va is None or vb is None:
                unstable[metric].append((key, "missing"))
            elif abs(vb - va) > band(va, token):
                unstable[metric].append((key, va, vb))
    return {
        "n_shared": len(shared),
        "unstable": unstable,
        "stable": not unstable["kernel_path_us"] and not unstable["e2e_us"],
    }


def scan_replay_consistency(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find committed attempts whose ``csv_path`` lists files the ``command`` cannot replay.

    A multi-file attempt (``csv_path`` = ``a.csv;b.csv``) must name EVERY listed
    file in its ``command`` string, so the attempt is replayable end-to-end from
    the ledger alone (no brace shorthand like ``run{1,2}.csv``, no required step
    hidden behind a ``#`` comment).  Superseded records are skipped.  Returns a
    list of ``(timestamp, [missing files])`` for offending records (empty == clean).
    """
    if not os.path.exists(path):
        return []
    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if "superseded_by" in rec:
                continue
            csv_path = rec.get("csv_path") or ""
            files = [p for p in csv_path.split(";") if p.strip()]
            if len(files) < 2:
                continue  # single/no file: nothing multi-file to reconcile
            command = rec.get("command") or ""
            # Strip anything after a '#' on each segment: a required step hidden in
            # a comment is not actually replayed by a shell.
            replayable = " ".join(seg.split("#", 1)[0] for seg in command.splitlines())
            missing = [fp for fp in files if fp not in replayable]
            if missing:
                offenders.append((rec.get("timestamp"), missing))
    return offenders


def _rejected_key(rec: dict) -> Tuple:
    """Identity of a rejected probe: model/dtype/token/act + the tile config.
    Used to detect duplicate non-superseded rejection records for the same probe."""
    cfg = rec.get("config") or {}
    cfg_key = tuple(sorted((str(k), str(v)) for k, v in cfg.items()))
    return (rec.get("model"), rec.get("dtype"), rec.get("act"), rec.get("token"), cfg_key)


def scan_duplicate_rejected_candidates(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find probes with more than one ACTIVE (non-superseded) rejected record.

    Two ledger entries that reject the same (model,dtype,act,token,config) probe
    are a provenance defect -- there must be exactly one active reason per probe
    (older duplicates must be marked ``superseded_by``).  Returns a list of
    ``(key, [timestamps])`` for probes with >1 active record (empty == clean).
    """
    if not os.path.exists(path):
        return []
    seen: Dict[Tuple, List] = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") != "rejected_candidate" or "superseded_by" in rec:
                continue
            seen.setdefault(_rejected_key(rec), []).append(rec.get("timestamp"))
    return [(k, ts) for k, ts in seen.items() if len(ts) > 1]


def scan_superseded_rejected_candidates(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find superseded rejected records that do NOT link to a matching successor.

    Every ``rejected_candidate`` carrying ``superseded_by`` must point at the
    timestamp of an EXISTING active (non-superseded) rejected record for the SAME
    rejected key ``(model,dtype,act,token,config)``.  A supersede link to a
    different probe's record (or to no record) is an evidence-integrity defect:
    ``scan_duplicate_rejected_candidates`` only proves one active record per key, it
    does not prove the superseded chain points to the correct successor.  Returns a
    list of ``(timestamp, reason)`` for offending records (empty == clean).
    """
    if not os.path.exists(path):
        return []
    active_ts_by_key: Dict[Tuple, set] = {}
    superseded: List[dict] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") != "rejected_candidate":
                continue
            if "superseded_by" in rec:
                superseded.append(rec)
            else:
                active_ts_by_key.setdefault(_rejected_key(rec), set()).add(rec.get("timestamp"))
    offenders: List[Tuple] = []
    for rec in superseded:
        key = _rejected_key(rec)
        target = rec.get("superseded_by")
        if target not in active_ts_by_key.get(key, set()):
            offenders.append((rec.get("timestamp"), f"superseded_by={target} is not an active record of the same key"))
    return offenders


__all__ = [
    "ATTEMPTS_JSONL",
    "LEDGER_MD",
    "REQUIRED_ATTEMPT_FIELDS",
    "Attempt",
    "append_attempt",
    "read_point_csv",
    "compare_point",
    "compare_csvs",
    "selected_candidate_gate",
    "scan_replay_consistency",
    "scan_duplicate_rejected_candidates",
    "scan_superseded_rejected_candidates",
    "repeatability_check",
    "PointVerdict",
    "CampaignVerdict",
]
