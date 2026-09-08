#!/usr/bin/env python3
"""Post flydsl-code-review findings to a GitHub PR as inline comments.

GitHub only accepts an inline review comment on a line that appears in the PR
diff. A finding on any other line is rejected with 422, so the placement
decision has to be made against the actual patch rather than guessed. This
script parses the PR's per-file patches, posts what it can inline, and rolls
everything else into a single summary comment so no finding is silently lost.

Usage:
    post_review.py --pr 1106 --findings findings.json [--repo owner/name] [--dry-run]

The findings file is the `findings` array from the workflow result, or a bare
JSON array of the same objects:

    [{"file": "python/flydsl/expr/llvm.py",
      "line": 82,
      "summary": "...",
      "failure_scenario": "...",
      "verdict": "CONFIRMED"}]

`line` is optional; a finding without one goes to the summary comment.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def gh(*args: str, stdin: str | None = None) -> str:
    """Run a gh command and return stdout, raising with gh's stderr on failure."""
    proc = subprocess.run(
        ["gh", *args],
        input=stdin,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def commentable_lines(patch: str) -> set[int]:
    """RIGHT-side line numbers that appear in *patch*.

    Added and context lines are commentable; deleted lines are not, because they
    have no RIGHT-side number. Tracks the post-image line counter through each
    hunk.
    """
    lines: set[int] = set()
    cursor = 0
    for raw in patch.splitlines():
        header = HUNK.match(raw)
        if header:
            cursor = int(header.group(1))
            continue
        if cursor == 0:  # before the first hunk header
            continue
        if raw.startswith("-"):
            continue  # deleted: no RIGHT-side line
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        # added ("+") or context (" ") — both exist on the RIGHT side
        lines.add(cursor)
        cursor += 1
    return lines


def load_findings(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("findings", [])
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of findings or an object with a 'findings' key")
    return data


def normalize(file_field: str, repo_root: Path) -> str:
    """Reduce a finding's `file` to a repo-relative POSIX path."""
    p = Path(file_field)
    if p.is_absolute():
        try:
            p = p.relative_to(repo_root)
        except ValueError:
            pass
    return p.as_posix()


def body_for(f: dict) -> str:
    verdict = f.get("verdict", "")
    tag = f"**{verdict}** — " if verdict else ""
    parts = [f"{tag}{f.get('summary', '').strip()}"]
    scenario = (f.get("failure_scenario") or "").strip()
    if scenario:
        parts.append(f"\n_Failure scenario:_ {scenario}")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr", required=True, help="PR number")
    ap.add_argument("--findings", required=True, type=Path, help="JSON file of findings")
    ap.add_argument("--repo", help="owner/name; inferred from the checkout when omitted")
    ap.add_argument("--dry-run", action="store_true", help="print what would be posted; post nothing")
    args = ap.parse_args()

    findings = load_findings(args.findings)
    if not findings:
        print("No findings to post.")
        return 0

    repo = args.repo or gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner").strip()
    repo_root = Path(
        subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or "."
    )

    pr = json.loads(gh("api", f"repos/{repo}/pulls/{args.pr}", "--jq", "{sha: .head.sha, state: .state}"))
    if pr["state"] != "open":
        print(f"PR #{args.pr} is {pr['state']}; refusing to comment.", file=sys.stderr)
        return 1
    head_sha = pr["sha"]

    files = json.loads(gh("api", "--paginate", f"repos/{repo}/pulls/{args.pr}/files"))
    diff_lines = {f["filename"]: commentable_lines(f.get("patch") or "") for f in files}

    inline: list[tuple[dict, str, int]] = []
    deferred: list[tuple[dict, str]] = []
    for f in findings:
        path = normalize(f.get("file", ""), repo_root)
        line = f.get("line")
        if path not in diff_lines:
            deferred.append((f, "file not in this PR's diff"))
        elif line is None:
            deferred.append((f, "no line number"))
        elif int(line) not in diff_lines[path]:
            deferred.append((f, f"line {line} is not in the diff"))
        else:
            inline.append((f, path, int(line)))

    print(f"{repo} PR #{args.pr} @ {head_sha[:12]}")
    print(f"  {len(inline)} inline, {len(deferred)} deferred to a summary comment")

    posted = 0
    for f, path, line in inline:
        payload = {
            "body": body_for(f),
            "commit_id": head_sha,
            "path": path,
            "line": line,
            "side": "RIGHT",
        }
        if args.dry_run:
            print(f"  [inline] {path}:{line} — {f.get('summary', '')[:80]}")
            continue
        try:
            gh(
                "api",
                "--method",
                "POST",
                f"repos/{repo}/pulls/{args.pr}/comments",
                "--input",
                "-",
                stdin=json.dumps(payload),
            )
            posted += 1
        except RuntimeError as exc:
            # Placement can still be rejected (e.g. the PR was pushed to between
            # our files call and this post). Don't lose the finding.
            print(f"  ! inline post failed for {path}:{line}: {exc}", file=sys.stderr)
            deferred.append((f, "inline post rejected by GitHub"))

    if deferred:
        rows = "\n".join(
            f"- `{normalize(f.get('file',''), repo_root)}"
            + (f":{f['line']}" if f.get("line") is not None else "")
            + f"` — {f.get('summary','').strip()}  \n  _({why})_"
            for f, why in deferred
        )
        summary = (
            "### Code review — findings outside the diff\n\n"
            "These could not be attached to a line in this PR's diff:\n\n" + rows + "\n"
        )
        if args.dry_run:
            print("\n  [summary comment]\n" + summary)
        else:
            gh(
                "api",
                "--method",
                "POST",
                f"repos/{repo}/issues/{args.pr}/comments",
                "--input",
                "-",
                stdin=json.dumps({"body": summary}),
            )

    if args.dry_run:
        print("\nDry run: nothing was posted.")
    else:
        print(f"Posted {posted} inline comment(s)" + (" and 1 summary comment." if deferred else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
