#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The review artifact contract, shared by the runner and publisher."""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re

SCHEMA_VERSION = 2
PER_ANGLE = 6
SWEEP_MAX = 8
MAX_FINDINGS = 12
ANGLES = (
    ("trace-time", "correctness", "Angle A — trace-time vs runtime semantics"),
    ("addressing", "correctness", "Angle B — memory addressing and out-of-bounds"),
    ("sync-lds", "correctness", "Angle C — synchronization, LDS, and value lifetime"),
    ("arch-atom", "correctness", "Angle D — architecture and atom contracts"),
    ("removed", "correctness", "Angle E — removed-behavior auditor"),
    ("cross-layer", "correctness", "Angle F — cross-layer tracer"),
    ("conventions", "convention", "Angle G — repo conventions and API stability"),
    ("reuse", "convention", "Angle H — reuse, simplification, and altitude"),
    ("test-doc", "convention", "Angle I — test and documentation contract"),
)
PREFLIGHTS = (
    ("preflight:conventions", "scan_legacy_spelling.py"),
    ("preflight:test-doc", "scan_unreachable_tests.py"),
)
VERDICTS = ("CONFIRMED", "PLAUSIBLE", "REFUTED")
SEVERITIES = ("P0", "P1", "P2", "P3")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize_path(value: str, snapshot: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("a finding needs a repository-relative file path")
    value = value.replace("\\", "/")
    if snapshot and value.startswith(snapshot.rstrip("/") + "/"):
        value = value[len(snapshot.rstrip("/")) + 1 :]
    value = posixpath.normpath(value)
    if value in (".", "..") or value.startswith(("/", "../")) or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"file is outside the reviewed tree: {value}")
    return value


def validate_output(output: dict, *, candidate_limit: int | None = None, snapshot: str | None = None) -> dict:
    """Validate independently of the model's structured-output schema."""
    if not isinstance(output, dict) or output.get("status") != "COMPLETE":
        raise ValueError(f"agent did not complete: {output!r}")
    if output.get("limitations") != []:
        raise ValueError(f"agent reported unresolved limitations: {output.get('limitations')!r}")
    if candidate_limit is None:
        if output.get("verdict") not in VERDICTS or not isinstance(output.get("evidence"), str):
            raise ValueError("invalid verifier verdict or evidence")
        if not output["evidence"].strip():
            raise ValueError("empty verifier evidence")
    else:
        candidates = output.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > candidate_limit:
            raise ValueError("invalid or over-limit candidate list; candidates must not be silently truncated")
        for c in candidates:
            if not isinstance(c, dict):
                raise ValueError("candidate must be an object")
            for key in ("summary", "mechanism", "failure_scenario"):
                if not isinstance(c.get(key), str) or not c[key].strip():
                    raise ValueError(f"candidate has no {key}")
            c["file"] = normalize_path(c.get("file"), snapshot)
            c["mechanism"] = " ".join(c["mechanism"].casefold().split())
            line = c.get("line")
            if line is not None and (type(line) is not int or line < 1):
                raise ValueError("line must be a positive integer or null")
            if c.get("severity") not in SEVERITIES:
                raise ValueError("candidate needs a P0/P1/P2/P3 severity")
    return output


def stage_output(stages: dict, label: str) -> dict | None:
    stage = stages.get(label, {})
    return stage.get("output") if stage.get("status") == "COMPLETE" else None


def validate_preflight(output: dict) -> dict:
    if not isinstance(output, dict) or type(output.get("exit_code")) is not int or output["exit_code"] not in (0, 1):
        raise ValueError("completed preflight must have scanner exit code 0 or 1")
    if not all(isinstance(output.get(key), str) for key in ("stdout", "stderr")):
        raise ValueError("completed preflight must retain scanner stdout and stderr")
    return output


def collect_candidates(stages: dict) -> list[dict]:
    """Exact location AND mechanism, after every finder has had a chance to finish."""
    unique: dict[str, dict] = {}
    sources = [(f"find:{label}", kind) for label, kind, _ in ANGLES] + [("sweep", "correctness")]
    for label, kind in sources:
        output = stage_output(stages, label)
        if output is None:
            continue
        validate_output(output, candidate_limit=SWEEP_MAX if label == "sweep" else PER_ANGLE)
        for index, raw in enumerate(output["candidates"]):
            identity = [raw["file"], raw.get("line"), raw["mechanism"]]
            cid = digest(identity)
            source = {"stage": label, "index": index}
            if cid in unique:
                unique[cid]["sources"].append(source)
                unique[cid]["severity"] = min(unique[cid]["severity"], raw["severity"])
                if kind == "correctness":
                    unique[cid]["kind"] = kind
                continue
            unique[cid] = {
                **{key: raw[key] for key in ("file", "mechanism", "summary", "severity", "failure_scenario")},
                "line": raw.get("line"),
                "id": cid,
                "kind": kind,
                "sources": [source],
            }
    return sorted(unique.values(), key=candidate_order)


def candidate_order(c: dict) -> tuple:
    return (c["kind"] == "convention", SEVERITIES.index(c["severity"]), c["file"], c["line"] or 0, c["id"])


def judged_candidates(stages: dict) -> list[dict]:
    candidates = collect_candidates(stages)
    for c in candidates:
        verification = stage_output(stages, "verify:" + c["id"])
        challenge = stage_output(stages, "challenge:" + c["id"])
        c.update(verification=verification, challenge=challenge, verdict=None, evidence=None)
        if verification is None:
            continue
        validate_output(verification)
        if verification["verdict"] == "CONFIRMED":
            if challenge is None:
                continue  # An unchallenged CONFIRMED is unresolved, never reportable.
            validate_output(challenge)
            c["verdict"] = challenge["verdict"]
            c["evidence"] = verification["evidence"] + "\n\nChallenger: " + challenge["evidence"]
        else:
            c["verdict"] = verification["verdict"]
            c["evidence"] = verification["evidence"]
    return candidates


def rank_findings(candidates: list[dict]) -> list[dict]:
    surviving = [c for c in candidates if c["verdict"] in ("CONFIRMED", "PLAUSIBLE")]
    return sorted(
        surviving,
        key=lambda c: (c["kind"] == "convention", c["verdict"] == "PLAUSIBLE", *candidate_order(c)[1:]),
    )[:MAX_FINDINGS]


def required_stages(scope: dict | None, candidates: list[dict]) -> list[str]:
    labels = ["scope"]
    if scope and scope.get("files"):
        labels += [label for label, _ in PREFLIGHTS]
        labels += ["find:" + label for label, _, _ in ANGLES]
        labels += ["verify:" + c["id"] for c in candidates]
        labels += [
            "challenge:" + c["id"]
            for c in candidates
            if c.get("verification") and c["verification"]["verdict"] == "CONFIRMED"
        ]
        labels.append("sweep")
    labels.append("synthesize")
    return labels


def usage_metrics(stages: dict, elapsed_seconds: float) -> dict:
    attempts = [a for stage in stages.values() for a in stage.get("attempts", [])]
    costs = []
    tokens: dict[str, int] = {}
    for attempt in attempts:
        usage = attempt.get("usage") or {}
        cost = usage.get("total_cost_usd")
        costs.append(cost if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 else None)
        for key, value in (usage.get("tokens") or {}).items():
            if type(value) is int:
                tokens[key] = tokens.get(key, 0) + value
    return {
        "wall_time_seconds": elapsed_seconds,
        "agent_attempts": len(attempts),
        "tokens": tokens,
        "known_cost_usd": sum(c for c in costs if c is not None),
        "cost_is_complete": all(c is not None for c in costs),
        "attempts_without_cost": sum(c is None for c in costs),
    }


def build_report(state: dict) -> dict:
    stages = state["stages"]
    scope = state.get("scope")
    candidates = judged_candidates(stages)
    required = required_stages(scope, candidates)
    failed = [
        {"stage": label, "reason": stages.get(label, {}).get("error", "stage has not completed")}
        for label in required
        if stage_output(stages, label) is None
    ]
    for label, _ in PREFLIGHTS:
        output = stage_output(stages, label)
        if label in required and output is not None:
            try:
                validate_preflight(output)
            except ValueError as exc:
                failed.append({"stage": label, "reason": str(exc)})
    # Integrity checks and cancellation can fail outside an agent stage.
    failed += [
        {"stage": label, "reason": stage.get("error", "stage failed")}
        for label, stage in stages.items()
        if label not in required and stage.get("status") != "COMPLETE"
    ]
    complete = not failed
    ranked = rank_findings(candidates)
    reported = ranked if complete else []
    confirmed = [c for c in reported if c["verdict"] == "CONFIRMED"]
    risks = [c for c in reported if c["verdict"] == "PLAUSIBLE"]
    if not complete:
        summary = f"Review INCOMPLETE: {len(failed)} required stage(s) failed or have not completed."
    elif not scope["files"]:
        summary = "No changes in the pinned review scope."
    elif not ranked:
        summary = "Review complete. No findings survived verification."
    else:
        summary = f"Review complete. {len(confirmed)} confirmed finding(s); {len(risks)} plausible risk(s)."
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": state["run_id"],
        "implementation_sha256": state["implementation_sha256"],
        "config": state["config"],
        "failure_history": state.get("failure_history", []),
        "elapsed_seconds": state.get("elapsed_seconds", 0),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "summary": summary,
        "scope": scope,
        "stages": stages,
        "stage_failures": failed,
        "unresolved_candidate_ids": [c["id"] for c in candidates if c["verdict"] is None],
        "candidates": candidates,
        "reported_ids": [c["id"] for c in reported],
        "findings": confirmed,
        "risks": risks,
        "partial_findings": [] if complete else ranked,
        "metrics": usage_metrics(stages, state.get("elapsed_seconds", 0)),
        "stats": {
            "finders_completed": sum(stage_output(stages, "find:" + a[0]) is not None for a in ANGLES),
            "candidates": len(candidates),
            "duplicates": sum(len(c["sources"]) - 1 for c in candidates),
            "verified": sum(c["verification"] is not None for c in candidates),
            "challenged": sum(c["challenge"] is not None for c in candidates),
            "challenge_downgraded": sum(
                c["challenge"] is not None and c["challenge"]["verdict"] != "CONFIRMED" for c in candidates
            ),
            "refuted": sum(c["verdict"] == "REFUTED" for c in candidates),
            "reported": len(reported),
        },
    }


def validate_report(report: dict) -> dict:
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("expected a versioned runner result, not a bare findings array")
    if report.get("status") != "COMPLETE":
        raise ValueError("refusing to publish an INCOMPLETE review")
    scope = report.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("missing reviewed scope")
    for key in ("base_oid", "merge_base_oid", "diff_base_oid", "head_oid"):
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", scope.get(key, "")):
            raise ValueError(f"missing or invalid scope.{key}")
    if not re.fullmatch(r"[0-9a-f]{64}", scope.get("diff_sha256", "")):
        raise ValueError("missing diff hash")
    if not isinstance(scope.get("files"), list):
        raise ValueError("missing changed files")
    for file in scope["files"]:
        if normalize_path(file) != file:
            raise ValueError("noncanonical changed-file path")
    expected = build_report(report)
    if expected["status"] != "COMPLETE":
        raise ValueError("required stages are missing or unresolved")
    if stage_output(report["stages"], "scope") != scope:
        raise ValueError("scope does not match the completed scope stage")
    for key in (
        "summary",
        "candidates",
        "reported_ids",
        "findings",
        "risks",
        "partial_findings",
        "stage_failures",
        "unresolved_candidate_ids",
        "metrics",
        "stats",
    ):
        if report.get(key) != expected[key]:
            raise ValueError(f"{key} does not match the verified records")
    return report
