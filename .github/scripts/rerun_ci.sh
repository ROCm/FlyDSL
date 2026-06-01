#!/usr/bin/env bash
#
# Re-run FlyDSL CI workflow run(s) for a given PR head commit.
#
# Invoked by .github/workflows/slash-command-rerun-ci.yml. Inputs come from the
# environment so the same script serves both the comment and label entry points:
#
#   REPO         owner/repo                     (e.g. ROCm/FlyDSL)
#   HEAD_SHA     PR head commit SHA to match
#   CMD          rerun-ci | rerun-checks | rerun-all
#   FAILED_ONLY  "true" to re-run only failed jobs, else "false"
#   GH_TOKEN     token with actions:write (the workflow passes GITHUB_TOKEN)
#
# It resolves the most recent run of each target workflow whose headSha equals
# HEAD_SHA and re-runs it with `gh run rerun`. Re-running an existing run (rather
# than dispatching a ref) re-executes against the exact PR commit and works for
# fork PRs.

set -euo pipefail

: "${REPO:?REPO is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"
: "${CMD:?CMD is required}"
FAILED_ONLY="${FAILED_ONLY:-false}"

# Map command -> workflow file(s). These must match files in .github/workflows/.
case "$CMD" in
  rerun-ci)     targets=("flydsl.yaml") ;;
  rerun-checks) targets=("pre-checks.yaml") ;;
  rerun-all)    targets=("flydsl.yaml" "pre-checks.yaml") ;;
  *) echo "::error::unknown CMD '$CMD'"; exit 1 ;;
esac

rerun_one() {
  local wf="$1" run_id
  run_id="$(gh run list \
    --repo "$REPO" \
    --workflow "$wf" \
    --limit 50 \
    --json databaseId,headSha,createdAt \
    --jq "[.[] | select(.headSha == \"$HEAD_SHA\")] | sort_by(.createdAt) | last | .databaseId // empty")"

  if [ -z "$run_id" ]; then
    echo "::warning::No existing '$wf' run found for $HEAD_SHA. " \
         "Push a commit (or use the workflow's manual dispatch) to seed the first run."
    return 0
  fi

  if [ "$FAILED_ONLY" = "true" ]; then
    echo "Re-running FAILED jobs of $wf run $run_id ($HEAD_SHA)"
    gh run rerun "$run_id" --repo "$REPO" --failed
  else
    echo "Re-running ALL jobs of $wf run $run_id ($HEAD_SHA)"
    gh run rerun "$run_id" --repo "$REPO"
  fi
}

for wf in "${targets[@]}"; do
  rerun_one "$wf"
done
