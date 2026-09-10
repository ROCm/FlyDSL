#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Publish a COMPLETE runner artifact in one GitHub review request.

    post_review.py --findings /tmp/flydsl-review-<id>/result.json --dry-run

The artifact supplies the repository, PR, reviewed OIDs, candidate IDs, evidence
and metrics. Bare arrays and incomplete or edited findings are rejected. Repeating
the same finding set checks its marker before writing, including after an ambiguous
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

from review_common import canonical, digest, validate_report

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def gh(*args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def pages(endpoint: str) -> list[dict]:
    # --slurp keeps pagination valid JSON even when there is more than one page.
    return [item for page in json.loads(gh("api", "--paginate", "--slurp", endpoint)) for item in page]


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
    identity = {
        "scope": {k: scope[k] for k in ("repo", "pr", "base_oid", "merge_base_oid", "head_oid", "diff_sha256")},
        "findings": report["findings"],
        "risks": report["risks"],
    }
    return "<!-- flydsl-code-review:" + digest(identity) + " -->"


def payload_for(report: dict, files: list[dict]) -> dict:
    scope = report["scope"]
    diff_lines = {f["filename"]: commentable_lines(f.get("patch") or "") for f in files}
    inline, deferred = [], []
    for finding in report["findings"]:
        path, line = finding["file"], finding["line"]
        if line is not None and line in diff_lines.get(path, set()):
            inline.append({"path": path, "line": line, "side": "RIGHT", "body": body_for(finding)})
        else:
            deferred.append(finding)
    body = ["### FlyDSL code review", report["summary"], finding_set_marker(report)]
    for title, findings in (
        ("Confirmed findings outside the diff", deferred),
        ("Plausible risks (not merge blockers)", report["risks"]),
    ):
        if findings:
            body.append("#### " + title)
            for f in findings:
                location = f["file"] + (f":{f['line']}" if f["line"] is not None else "")
                body.append(f"`{location}`\n\n" + body_for(f))
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
        "reported_ids": report["reported_ids"],
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


def publish(report: dict, *, dry_run: bool) -> int:
    validate_report(report)
    scope = report["scope"]
    if not isinstance(scope.get("repo"), str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", scope["repo"]):
        raise ValueError("artifact is not a GitHub PR review")
    if type(scope.get("pr")) is not int or scope["pr"] < 1:
        raise ValueError("artifact is not a GitHub PR review")
    if not report["findings"] and not report["risks"]:
        print("Completed review has no findings or risks to post.")
        return 0
    endpoint = f"repos/{scope['repo']}/pulls/{scope['pr']}"
    reviews = endpoint + "/reviews"
    marker = finding_set_marker(report)
    check_pr(scope)
    existing = existing_review(reviews, marker, scope["head_oid"])
    if existing:
        print(f"Already posted: {existing.get('html_url', existing['id'])}")
        return 0
    payload = payload_for(report, pages(endpoint + "/files"))
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
            return publish(report, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Review was not published: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
