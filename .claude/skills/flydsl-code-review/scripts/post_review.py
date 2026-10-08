#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Publish a COMPLETE runner artifact in one GitHub review request.

    post_review.py --findings /tmp/flydsl-review-<id>/result.json --publish-severity P1 --dry-run

The artifact supplies the repository, PR, reviewed OIDs, candidate IDs, evidence
and metrics. Bare arrays and incomplete or edited findings are rejected. Repeating
the same pinned diff checks its marker before writing, including after an ambiguous
network failure. No POST is retried automatically.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
from pathlib import Path

from review_common import MAX_FINDINGS, SEVERITIES, canonical, digest, finding_order, validate_report

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def gh(*args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def json_documents(raw: str) -> list:
    """Decode the concatenated JSON documents emitted by gh api --paginate."""
    decoder = json.JSONDecoder()
    documents = []
    index = 0
    while index < len(raw):
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index == len(raw):
            break
        document, index = decoder.raw_decode(raw, index)
        documents.append(document)
    return documents


def pages(endpoint: str) -> list[dict]:
    documents = json_documents(gh("api", "--paginate", endpoint))
    if not all(isinstance(page, list) for page in documents):
        raise ValueError("paginated GitHub response must contain JSON arrays")
    return [item for page in documents for item in page]


def commentable_lines(patch: str) -> set[int]:
    """RIGHT-side added and context lines; deleted lines have no RIGHT-side number."""
    lines: set[int] = set()
    cursor = 0
    for raw in patch.splitlines():
        header = HUNK.match(raw)
        if header:
            cursor = int(header.group(1))
            continue
        if cursor == 0 or raw.startswith(("-", "\\")):
            continue
        if raw.startswith(("+", " ")):
            lines.add(cursor)
            cursor += 1
    return lines


def body_for(f: dict) -> str:
    return (
        f"**{f['verdict']} · {f['severity']} · {f['kind']}** — {f['summary'].strip()}\n\n"
        f"_Failure scenario:_ {f['failure_scenario'].strip()}\n\n"
        f"<details><summary>Verifier evidence</summary>\n\n{f['evidence'].strip()}\n\n</details>\n\n"
        f"Candidate: `{f['id']}`\n<!-- flydsl-code-review-finding:{f['id']} -->"
    )


def finding_set_marker(report: dict) -> str:
    scope = report["scope"]
    # One publication owns one pinned diff. Stochastic wording or hidden
    # lower-priority candidates must not create another review on the same tree.
    identity = {k: scope[k] for k in ("repo", "pr", "merge_base_oid", "diff_base_oid", "head_oid", "diff_sha256")}
    return "<!-- flydsl-code-review:" + digest(identity) + " -->"


def publishable_findings(report: dict, publish_severity: str = "P1") -> list[dict]:
    if publish_severity not in SEVERITIES:
        raise ValueError(f"invalid publish severity: {publish_severity!r}")
    cutoff = SEVERITIES.index(publish_severity)
    eligible = [
        candidate
        for candidate in report["candidates"]
        if candidate["verdict"] == "CONFIRMED" and SEVERITIES.index(candidate["severity"]) <= cutoff
    ]
    return sorted(eligible, key=finding_order)[:MAX_FINDINGS]


def severity_range(publish_severity: str) -> str:
    return "P0" if publish_severity == "P0" else f"P0-{publish_severity}"


def payload_for(report: dict, files: list[dict], publish_severity: str = "P1") -> dict:
    scope = report["scope"]
    findings = publishable_findings(report, publish_severity)
    diff_lines = {f["filename"]: commentable_lines(f.get("patch") or "") for f in files}
    inline, deferred = [], []
    for finding in findings:
        path, line = finding["file"], finding["line"]
        if line is not None and line in diff_lines.get(path, set()):
            inline.append({"path": path, "line": line, "side": "RIGHT", "body": body_for(finding)})
        else:
            deferred.append(finding)
    eligible_count = sum(
        candidate["verdict"] == "CONFIRMED"
        and SEVERITIES.index(candidate["severity"]) <= SEVERITIES.index(publish_severity)
        for candidate in report["candidates"]
    )
    omitted_count = sum(candidate["verdict"] in ("CONFIRMED", "PLAUSIBLE") for candidate in report["candidates"]) - len(
        findings
    )
    level = severity_range(publish_severity)
    body = [
        "### FlyDSL code review",
        f"Published {len(findings)} confirmed {level} finding(s); "
        f"{omitted_count} lower-priority or capped record(s) remain in the local artifact.",
        finding_set_marker(report),
    ]
    if deferred:
        body.append("#### Confirmed findings outside the diff")
        for finding in deferred:
            location = finding["file"] + (f":{finding['line']}" if finding["line"] is not None else "")
            body.append(f"`{location}`\n\n" + body_for(finding))
    provenance = {
        "run_id": report["run_id"],
        "status": report["status"],
        "implementation_sha256": report["implementation_sha256"],
        "config": report["config"],
        "paths": scope.get("paths", []),
        "instructions": scope.get("instructions", ""),
        "base_oid": scope["base_oid"],
        "merge_base_oid": scope["merge_base_oid"],
        "diff_base_oid": scope["diff_base_oid"],
        "head_oid": scope["head_oid"],
        "diff_sha256": scope["diff_sha256"],
        "publish_severity": publish_severity,
        "eligible_count": eligible_count,
        "published_ids": [finding["id"] for finding in findings],
        "omitted_count": omitted_count,
        "stage_failures": report["stage_failures"],
        "metrics": report["metrics"],
        "stats": report["stats"],
    }
    body.append(
        "<details><summary>Run provenance and usage</summary>\n\n```json\n"
        + json.dumps(provenance, indent=2)
        + "\n```\n</details>"
    )
    payload = {"commit_id": scope["head_oid"], "event": "COMMENT", "body": "\n\n".join(body), "comments": inline}
    if any(len(text) > 65000 for text in [payload["body"], *(c["body"] for c in inline)]):
        raise ValueError("review body exceeds GitHub's size limit; evidence was not truncated or posted")
    return payload


def check_pr(scope: dict) -> None:
    pr = json.loads(gh("api", f"repos/{scope['repo']}/pulls/{scope['pr']}"))
    if pr["state"] != "open":
        raise ValueError(f"PR is {pr['state']}; refusing to post")
    if pr["head"]["sha"] != scope["head_oid"] or pr["base"]["sha"] != scope["base_oid"]:
        raise ValueError("PR base/head changed since this review; start a new run")


def existing_review(endpoint: str, marker: str, head: str) -> dict | None:
    for review in pages(endpoint):
        if marker in (review.get("body") or ""):
            if review.get("commit_id") != head or review.get("state") == "PENDING":
                raise ValueError(
                    "matching marker is on an unexpected head or pending review; inspect it before retrying"
                )
            return review
    return None


def publish(report: dict, *, dry_run: bool, publish_severity: str = "P1") -> int:
    validate_report(report)
    scope = report["scope"]
    if not isinstance(scope.get("repo"), str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", scope["repo"]):
        raise ValueError("artifact is not a GitHub PR review")
    if type(scope.get("pr")) is not int or scope["pr"] < 1:
        raise ValueError("artifact is not a GitHub PR review")
    findings = publishable_findings(report, publish_severity)
    if not findings:
        print(f"Completed review has no confirmed {severity_range(publish_severity)} findings to post.")
        return 0
    endpoint = f"repos/{scope['repo']}/pulls/{scope['pr']}"
    reviews = endpoint + "/reviews"
    marker = finding_set_marker(report)
    check_pr(scope)
    existing = existing_review(reviews, marker, scope["head_oid"])
    if existing:
        print(f"Already posted: {existing.get('html_url', existing['id'])}")
        return 0
    payload = payload_for(report, pages(endpoint + "/files"), publish_severity)
    # Pins both routing and the live write. commit_id still pins the review if a
    # push races the final GET; GitHub provides no compare-and-swap POST primitive.
    check_pr(scope)
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    try:
        response = json.loads(gh("api", "--method", "POST", reviews, "--input", "-", stdin=canonical(payload)))
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
        # A lost response does not mean the write failed. Reconcile, never repost.
        existing = existing_review(reviews, marker, scope["head_oid"])
        if existing:
            print(f"Posted (response recovered): {existing.get('html_url', existing['id'])}")
            return 0
        raise
    print(f"Posted review: {response.get('html_url', response.get('id'))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--findings", "--result", required=True, type=Path, dest="result")
    parser.add_argument("--pr", type=int, help="optional assertion; must match the artifact")
    parser.add_argument("--repo", help="optional assertion; must match the artifact")
    parser.add_argument("--expected-head", help="optional assertion; the artifact's reviewed head is always required")
    parser.add_argument(
        "--publish-severity",
        choices=SEVERITIES,
        default="P1",
        help="publish confirmed findings from P0 through this severity (default: P1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the exact review payload without posting")
    args = parser.parse_args()
    try:
        report = validate_report(json.loads(args.result.read_text()))
        scope = report["scope"]
        for supplied, key in ((args.pr, "pr"), (args.repo, "repo"), (args.expected_head, "head_oid")):
            if supplied is not None and supplied != scope[key]:
                raise ValueError(f"requested {key} does not match the reviewed artifact")
        # Serialize local invocations sharing an artifact; remote retries use the marker.
        with args.result.with_suffix(".publish.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return publish(report, dry_run=args.dry_run, publish_severity=args.publish_severity)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Review was not published: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
