#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Step 1 fetch for the FlyDSL review-pr skill.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: fetch.sh <PR> [owner/repo] [validation-report.json]" >&2
  exit 1
fi

PR=$1
REPO="${2:-ROCm/FlyDSL}"
VALIDATION_REPORT="${3:-}"
case "$1" in
  */*#*)
    REPO="${1%#*}"
    PR="${1##*#}"
    VALIDATION_REPORT="${2:-}"
    ;;
esac

WORK=$(mktemp -d /tmp/flydsl-review-XXXXXX)
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh pr view "$PR" --repo "$REPO" \
  --json title,body,number,labels,files,author,reviews,comments,baseRefName,headRefOid,statusCheckRollup,isDraft,mergeable \
  >"$WORK/pr_meta.json"
gh pr diff "$PR" --repo "$REPO" >"$WORK/pr.diff"

BASE_REF=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseRefName"])' "$WORK/pr_meta.json")
BASE_REF_PATH=$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$BASE_REF")
gh api "repos/$REPO/branches/$BASE_REF_PATH" --jq .commit.sha >"$WORK/base_head.txt" 2>/dev/null || echo "" >"$WORK/base_head.txt"

echo "=== PR identity ==="
python3 - "$WORK/pr_meta.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"#{d['number']} {d['title']}")
print(f"author={d['author']['login']}  draft={d.get('isDraft')}  mergeable={d.get('mergeable')}")
labels = [x["name"] for x in d.get("labels", [])]
print(f"labels={labels or 'none'}")
files = d.get("files", [])
print(f"files_changed={len(files)}  +{sum(f.get('additions',0) for f in files)}/-{sum(f.get('deletions',0) for f in files)}")
PY

# The single most important CI fact for this repo: a green tick does not mean what a
# reviewer assumes. Print what actually ran so the review can say so precisely.
echo
echo "=== CI checks (read the caveats below before trusting green) ==="
python3 - "$WORK/pr_meta.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rollup = d.get("statusCheckRollup") or []
if not rollup:
    print("no checks reported")
for c in rollup:
    name = c.get("name") or c.get("context") or "?"
    state = c.get("conclusion") or c.get("state") or "?"
    print(f"  {state:<12} {name}")
PY
cat <<'EOF'

  Caveats that hold for every FlyDSL PR:
    - The PR benchmark step is REPORT-ONLY. scripts/compare_benchmark.py prints ratios
      and returns 0 unconditionally, so a performance regression cannot turn CI red.
      Someone must read the numbers out of the job log.
    - multi-gpu is label-gated ('multi-gpu' label); absent the label it never ran.
    - ATOM / vLLM / SGLang integration are nightly cron workflows, not PR checks.
      A change that aiter consumes has ZERO downstream coverage at PR time.
    - Docs-only detection can substitute a green placeholder for the 4-runner GPU matrix.
EOF

echo
echo "=== Human review comments (Copilot and bots filtered out) ==="
gh api "repos/$REPO/pulls/$PR/comments" --paginate 2>/dev/null | python3 -c "
import json,sys
try:
    comments = json.load(sys.stdin)
except Exception:
    comments = []
n = 0
for c in comments:
    author = c.get('user',{}).get('login','')
    low = author.lower()
    if 'copilot' in low or low.endswith('[bot]') or 'bot' == low:
        continue
    body = (c.get('body','') or '').strip()
    if not body:
        continue
    n += 1
    print(f\"[{author}] {c.get('path','')}:{c.get('line') or c.get('original_line','')}\")
    print('  ' + body[:400].replace('\n', '\n  '))
print(f'({n} human inline comments)')
"

python3 - "$WORK/pr_meta.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for kind in ("reviews", "comments"):
    for r in d.get(kind, []):
        login = (r.get("author") or {}).get("login", "")
        if "copilot" in login.lower() or login.lower().endswith("[bot]"):
            continue
        body = (r.get("body") or "").strip()
        if body:
            print(f"[{kind[:-1].upper()} {login}] {body[:400]}")
PY

echo
SCAN="$SKILL_DIR/scan_flydsl_diff.py"
if [ ! -r "$SCAN" ]; then
  echo "required scanner is missing: $SCAN" >&2
  exit 1
fi
if ! python3 "$SCAN" --diff "$WORK/pr.diff"; then
  echo "required diff scan failed; do not report an empty candidate list" >&2
  exit 1
fi

echo
if [ -n "$VALIDATION_REPORT" ] && [ -r "$VALIDATION_REPORT" ]; then
  python3 - "$WORK/pr_meta.json" "$VALIDATION_REPORT" <<'PY'
import json, sys
meta, report = (json.load(open(p)) for p in sys.argv[1:3])
expected = meta["headRefOid"]
actual = (report.get("repo") or {}).get("head")
if actual != expected:
    raise SystemExit(
        f"validation report is for another checkout: expected head {expected}, got {actual}. "
        "Treat this review as static-only."
    )
stages = report.get("stages") or {}
required = {"merge_sim", "gpu_claim", "runtime_compat", "test_policy", "correctness", "perf", "diff_scan"}
missing = required - stages.keys()
if missing:
    raise SystemExit(f"validation report omits required stages: {sorted(missing)}")

print(f"validation report accepted for head {expected}; verdict={report.get('verdict')}")
for name in sorted(required):
    st = stages[name]
    print(f"  {st.get('status'):<5} {name:<16} {st.get('reason','')}")

perf = stages["perf"]
env = report.get("environment") or {}
if perf.get("status") == "pass":
    rows = (perf.get("detail") or {}).get("rows") or []
    print(f"\nPERFORMANCE: no regression across {len(rows)} row(s) beyond the measured noise floor.")
elif perf.get("status") == "fail":
    print("\nPERFORMANCE: REGRESSION -- reproducible, may gate the merge.")
    for r in (perf.get("detail") or {}).get("regressions", []):
        print(f"  {r['label']}: {r['change_pct']:+.1f}% (noise floor {r['noise_floor']*100:.1f}%)")
else:
    print(f"\nPERFORMANCE: NOT MEASURED ({perf.get('reason')}). CI cannot catch this either.")
if env.get("cold_cache_required"):
    print(f"cold cache was forced: {env.get('cold_cache_reason')}")
for f in report.get("findings", []):
    print(f"  [{f['severity']}] {f['stage']}: {f['detail']}")
PY
else
  echo "validation report not supplied: this is a STATIC-ONLY advisory review."
  echo "No finding may assert runtime behaviour (perf, accuracy, launch failure) as fact."
  echo "In particular you may NOT say performance is unaffected: CI's benchmark step"
  echo "returns 0 unconditionally, so nothing has checked it. Run validate-kernel-pr."
fi

echo "---"
echo "review-pr fetch complete"
echo "work_dir=$WORK"
echo "pr_meta=$WORK/pr_meta.json"
echo "pr_diff=$WORK/pr.diff"
echo "base_head=$(cat "$WORK/base_head.txt")"
