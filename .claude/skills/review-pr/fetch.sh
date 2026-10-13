#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Step 1 and Step 1b of the review-pr skill: fetch one PR and collect the evidence the later
# steps read. Everything it prints is meant to be read; it ends with WORK=<dir>.
#
# usage: fetch.sh <PR | owner/repo#N> [owner/repo]
set -euo pipefail

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SKILL" rev-parse --show-toplevel)"
TRIAGE="$SKILL/triage.py"

PR="${1:?usage: fetch.sh <PR | owner/repo#N> [owner/repo]}"
REPO="${2:-ROCm/FlyDSL}"
case "$PR" in */*#*) REPO="${PR%#*}"; PR="${PR##*#}" ;; esac
# Fetch from the repository under review by URL, never from `origin`: a clone whose origin is a
# fork resolves refs/pull/N/head against the fork and silently reviews a different PR.
REPO_URL="https://github.com/$REPO"

# One scratch dir per run: two reviews sharing fixed /tmp paths analyse each other's diffs.
WORK=$(mktemp -d "${TMPDIR:-/tmp}/review-pr-XXXXXX")
mkdir -p "$WORK/head"
META="$WORK/pr_meta.json"
meta() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print($1)" "$META"; }

gh pr view "$PR" --repo "$REPO" --json \
  number,title,body,url,state,isDraft,author,labels,files,baseRefName,baseRefOid,headRefName,headRefOid,mergedAt,reviews,comments,statusCheckRollup \
  > "$META"
TITLE=$(meta 'd["title"]')
STATE=$(meta 'd["state"]')
HEAD_SHA=$(meta 'd["headRefOid"]')
BASE_REF=$(meta 'd["baseRefName"]')
echo "== $REPO#$PR: $TITLE"
echo "   state=$STATE draft=$(meta 'd["isDraft"]') author=$(meta 'd["author"]["login"]') head=$HEAD_SHA"

# Pin the head once. Every artifact below is computed from HEAD_SHA and BASE_SHA, never from the
# PR's moving refs or `gh pr diff`, so a push during the review cannot mix two revisions.
git -C "$PROJECT_ROOT" fetch -q "$REPO_URL" "$HEAD_SHA" 2>/dev/null || git -C "$PROJECT_ROOT" fetch -q "$REPO_URL" "refs/pull/$PR/head"
git -C "$PROJECT_ROOT" cat-file -e "$HEAD_SHA^{commit}" 2>/dev/null \
  || { echo "head $HEAD_SHA could not be fetched; the PR moved while fetching -- rerun" >&2; exit 1; }

# The merge target. An open PR merges into the base branch tip, which can be far ahead of the
# recorded baseRefOid. A merged or closed PR is reviewed as it was: against its recorded base.
BASE_SHA=""
if [ "$STATE" = "OPEN" ]; then
  BASE_SHA=$(gh api "repos/$REPO/branches/$BASE_REF" --jq .commit.sha 2>/dev/null || true)
  BASE_NOTE="tip of $BASE_REF"
fi
if [ -z "$BASE_SHA" ]; then
  BASE_SHA=$(meta 'd.get("baseRefOid") or ""')
  BASE_NOTE="recorded base of a $STATE PR (historical, so staleness is not judged)"
fi
[ -n "$BASE_SHA" ] || { echo "no base commit for $REPO#$PR; nothing downstream can be trusted" >&2; exit 1; }
git -C "$PROJECT_ROOT" cat-file -e "$BASE_SHA^{commit}" 2>/dev/null || git -C "$PROJECT_ROOT" fetch -q "$REPO_URL" "$BASE_SHA"
echo "$BASE_SHA" > "$WORK/base_head.txt"
echo "   base=$BASE_SHA ($BASE_NOTE)"

# The diff: the same three-dot diff GitHub shows, computed from the pinned commits. It also has
# no 20000-line cap, which large FlyDSL kernel PRs exceed.
git -C "$PROJECT_ROOT" diff --no-color --no-ext-diff "$BASE_SHA...$HEAD_SHA" > "$WORK/pr.diff"
[ -s "$WORK/pr.diff" ] || { echo "empty diff for $REPO#$PR; reviewing it would report nothing" >&2; exit 1; }
echo "git diff $BASE_SHA...$HEAD_SHA sha256=$(sha256sum < "$WORK/pr.diff" | cut -d' ' -f1)" > "$WORK/diff_source.txt"
echo "   diff: $(grep -c '^diff --git' "$WORK/pr.diff") files, $(cut -d' ' -f4 "$WORK/diff_source.txt")"

# Linked issue, and the human discussion so far. Copilot authors most inline comments here and
# its findings are not maintainer evidence, so bot accounts are filtered out.
ISSUE=$(python3 - "$META" <<'PY'
import json, re, sys
m = re.search(r"(?:fix|close|resolve)[sd]?[: ]*#(\d+)", json.load(open(sys.argv[1])).get("body") or "", re.I)
print(m.group(1) if m else "")
PY
)
if [ -n "$ISSUE" ]; then
  gh issue view "$ISSUE" --repo "$REPO" --json title,body > "$WORK/pr_issue.json" 2>/dev/null || rm -f "$WORK/pr_issue.json"
fi
{
  python3 - "$META" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for kind, items in (("REVIEW", d.get("reviews") or []), ("COMMENT", d.get("comments") or [])):
    for c in items:
        who = (c.get("author") or {}).get("login", "")
        body = (c.get("body") or "").strip()
        if body and "bot" not in who.lower() and "copilot" not in who.lower():
            print(f"[{kind} {who}] {body[:300]}")
PY
  gh api --paginate "repos/$REPO/pulls/$PR/comments" \
    --jq '.[] | {who: .user.login, path, line: (.line // .original_line), body}' 2>/dev/null \
    | python3 -c '
import json, sys
for raw in sys.stdin:
    c = json.loads(raw)
    who, body = c["who"] or "", (c["body"] or "").strip()
    if body and "bot" not in who.lower() and "copilot" not in who.lower():
        print("[INLINE %s] %s:%s  %s" % (who, c["path"], c["line"], body[:300]))
' || true
} > "$WORK/comments.txt"
echo "   human comments: $(grep -c '^\[' "$WORK/comments.txt" || true)"

# The tree the PR merges into, and the PR's own file images. Read base files from merge-target/,
# never from the reviewer's checkout: a finding about the wrong tree is how premises go wrong.
git -C "$PROJECT_ROOT" worktree list --porcelain | sed -n "s|^worktree \(${TMPDIR:-/tmp}/review-pr-.*\)\$|\1|p" \
  | while IFS= read -r w; do
      if [ -d "$w" ] && [ -z "$(find "$w" -maxdepth 0 -mtime -1 2>/dev/null)" ]; then
        git -C "$PROJECT_ROOT" worktree remove --force "$w" >/dev/null 2>&1 || true
      fi
    done || true
git -C "$PROJECT_ROOT" worktree prune
git -C "$PROJECT_ROOT" worktree add -q --detach "$WORK/merge-target" "$BASE_SHA"
echo "MERGE TARGET: $WORK/merge-target at $BASE_SHA ($BASE_NOTE)" > "$WORK/merge_target.txt"
if git -C "$WORK/merge-target" apply --check "$WORK/pr.diff" 2> "$WORK/.apply_err"; then
  echo "APPLIES: the diff applies cleanly to $BASE_SHA" > "$WORK/applies.txt"
else
  { echo "STALE: the diff no longer applies to $BASE_SHA -- the PR needs a rebase, and its CI result"
    echo "  describes a tree that has moved"; sed 's/^/  /' "$WORK/.apply_err"; } > "$WORK/applies.txt"
fi
rm -f "$WORK/.apply_err"
grep '^diff --git' "$WORK/pr.diff" | sed 's|^diff --git a/[^ ]* b/||' | while IFS= read -r f; do
  if git -C "$PROJECT_ROOT" cat-file -e "$HEAD_SHA:$f" 2>/dev/null; then
    mkdir -p "$WORK/head/$(dirname "$f")"
    git -C "$PROJECT_ROOT" show "$HEAD_SHA:$f" > "$WORK/head/$f"
  fi
done

# CI at this head. A green tick is evidence only for what that job runs; see SKILL.md Step 1.
python3 - "$META" > "$WORK/ci_checks.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
labels = sorted(x["name"] for x in d.get("labels") or [])
print(f"head {d['headRefOid'][:12]}  labels: {', '.join(labels) or 'none'}")
rows = d.get("statusCheckRollup") or []
if not rows:
    print("no checks reported for this head")
for c in rows:
    name = c.get("name") or c.get("context") or "?"
    state = c.get("conclusion") or c.get("state") or "?"
    if c.get("status") and c.get("status") != "COMPLETED":
        state = c["status"]
    flow = c.get("workflowName") or ""
    print(f"  {state:<12} {flow + ' / ' if flow else ''}{name}  {c.get('detailsUrl') or c.get('targetUrl') or ''}")
PY

# What the CI benchmark measured at this head, current vs main and vs the latest tag. The step
# never fails CI (compare_benchmark.py returns 0), so nothing else reads these rows.
python3 - "$META" > "$WORK/.bench_jobs" <<'PY'
import json, re, sys
for c in json.load(open(sys.argv[1])).get("statusCheckRollup") or []:
    m = re.search(r"/job/(\d+)", c.get("detailsUrl") or "")
    if (c.get("name") or "").startswith("test (linux-flydsl-") and m and c.get("status") == "COMPLETED":
        print(m.group(1), c["name"])
PY
{
  if [ ! -s "$WORK/.bench_jobs" ]; then
    echo "no completed test (linux-flydsl-*) job at this head, so CI measured nothing"
  fi
  while read -r job name; do
    echo "== $name (job $job)"
    if gh api "repos/$REPO/actions/jobs/$job/logs" > "$WORK/.joblog" 2>/dev/null; then
      sed -E 's/^[0-9T:.-]+Z //' "$WORK/.joblog" \
        | awk '/=== Benchmark: current vs/{on=1} on{print} /Benchmark comparison report completed/{on=0}' \
        | grep -E '===|ratio=|SKIP|No comparable|Skipped' | sed 's/^/  /' || echo "  no comparison rows in the log"
    else
      echo "  job log unavailable (expired, or no access)"
    fi
  done < "$WORK/.bench_jobs"
  python3 "$TRIAGE" benchcov "$WORK/pr.diff" "$PROJECT_ROOT"
} > "$WORK/bench_compare.txt"
rm -f "$WORK/.bench_jobs" "$WORK/.joblog"

# Step 1b: the rules this diff can trigger, their text, and the evidence they ask for.
python3 "$TRIAGE" rules "$WORK/pr.diff" "$TITLE" "$PROJECT_ROOT" > "$WORK/rules.txt"
python3 "$TRIAGE" expand "$WORK/rules.txt" "$SKILL/rules.md" > "$WORK/rules_expanded.txt"
python3 "$TRIAGE" scan "$WORK/pr.diff" > "$WORK/scan.txt"
python3 "$TRIAGE" core "$WORK/pr.diff" "$PROJECT_ROOT" > "$WORK/core_expected.txt"
python3 "$TRIAGE" bounds "$WORK/pr.diff" "$WORK/head" > "$WORK/bounds.txt"
python3 "$TRIAGE" citest "$WORK/pr.diff" "$PROJECT_ROOT" "$WORK/head" > "$WORK/ci_coverage.txt"

echo
echo "--- CI (ci_checks.txt) ---"; cat "$WORK/ci_checks.txt"
echo "--- human comments (comments.txt) ---"; head -40 "$WORK/comments.txt"
echo "--- derived rules (rules.txt) ---"; cat "$WORK/rules.txt"
echo "--- backbone files (core_expected.txt) ---"; cat "$WORK/core_expected.txt"
echo "--- $(head -1 "$WORK/applies.txt")"
echo
echo "WORK=$WORK"
echo "artifacts: pr_meta.json pr.diff diff_source.txt base_head.txt merge_target.txt applies.txt comments.txt" \
  "ci_checks.txt bench_compare.txt rules.txt rules_expanded.txt scan.txt core_expected.txt bounds.txt" \
  "ci_coverage.txt head/ merge-target/${ISSUE:+ pr_issue.json}"
