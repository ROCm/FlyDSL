#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Headless review of one PR: fetch -> worker agent -> independent refuter agent -> gates.
# Nothing is posted; the card and the gate report are left in $WORK.
#
# usage: run_one.sh <PR | owner/repo#N> [owner/repo]
#
#   FLYDSL_REVIEW_AGENT           agent CLI, default `claude` (run as `<agent> -p --dangerously-skip-permissions`)
#   FLYDSL_REVIEW_REFUTER_AGENT   a different CLI or model for Step 8.7, default the same agent
#   FLYDSL_REVIEW_SANDBOX_IMAGE   run both agents in this Docker image instead of on the host
#   FLYDSL_AGENT_TIMEOUT          seconds per agent, default 2400
#
# The agents read untrusted PR text with permissions skipped. In the sandbox they see the checkout
# read-only and $WORK read-write, run as the calling user, and get no GitHub credentials and no
# home directory, so they cannot post to GitHub or read the host's tokens. Model access comes from
# ANTHROPIC_BASE_URL, which must be a gateway that holds the credential and ignores the client's
# x-api-key: no real token is passed in.
set -euo pipefail

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(git -C "$SKILL" rev-parse --show-toplevel)"
PR_ARG="${1:?usage: run_one.sh <PR | owner/repo#N> [owner/repo]}"
AGENT="${FLYDSL_REVIEW_AGENT:-claude}"
REFUTER="${FLYDSL_REVIEW_REFUTER_AGENT:-$AGENT}"
TIMEOUT="${FLYDSL_AGENT_TIMEOUT:-2400}"
say() { echo "[run_one $PR_ARG] $*"; }

say "fetch..."
FETCH_LOG="$(mktemp)"
bash "$SKILL/fetch.sh" "$@" | tee "$FETCH_LOG"
W="$(sed -n 's/^WORK=//p' "$FETCH_LOG" | tail -1)"
rm -f "$FETCH_LOG"
[ -n "$W" ] && [ -d "$W" ] || { say "fetch produced no WORK dir"; exit 1; }

render() {  # <template> -> prompt text with the placeholders filled in
  python3 - "$1" "$W" "$SKILL" "$PROJ" <<'PY'
import json, pathlib, sys
tpl, work, skill, proj = sys.argv[1:5]
meta = json.loads(pathlib.Path(work, "pr_meta.json").read_text())
repo = meta["url"].split("github.com/")[1].split("/pull/")[0]
text = pathlib.Path(tpl).read_text()
for key, val in {"PR": str(meta["number"]), "TITLE": meta["title"], "REPO": repo,
                 "WORK": work, "SKILL": skill, "PROJ": proj}.items():
    text = text.replace("{{" + key + "}}", val)
print(text)
PY
}

run_agent() {  # <label> <cli> <prompt-file> <expected-output>
  local label="$1" cli="$2" prompt="$3" out="$4" rc=0
  say "$label..."
  if [ -n "${FLYDSL_REVIEW_SANDBOX_IMAGE:-}" ]; then
    local bin env_args=()
    bin="$(readlink -f "$(command -v "$cli")")"
    mkdir -p "$W/.home"
    for v in ANTHROPIC_BASE_URL ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL \
             ANTHROPIC_SMALL_FAST_MODEL CLAUDE_CODE_EFFORT_LEVEL; do
      [ -n "${!v:-}" ] && env_args+=(-e "$v")
    done
    # The placeholder key only satisfies the CLI's start-up check; the gateway supplies the real one.
    timeout "$TIMEOUT" docker run --rm --network host --user "$(id -u):$(id -g)" \
      -e HOME="$W/.home" -e ANTHROPIC_API_KEY=sandbox-has-no-credential "${env_args[@]}" \
      -v "$PROJ:$PROJ:ro" -v "$W:$W" -v "$bin:/usr/local/bin/review-agent:ro" -w "$PROJ" \
      --entrypoint /usr/local/bin/review-agent "$FLYDSL_REVIEW_SANDBOX_IMAGE" \
      -p --dangerously-skip-permissions "$(cat "$prompt")" > "$W/$label.log" 2>&1 || rc=$?
  else
    (cd "$PROJ" && timeout "$TIMEOUT" "$cli" -p --dangerously-skip-permissions "$(cat "$prompt")") \
      > "$W/$label.log" 2>&1 || rc=$?
  fi
  [ "$rc" -eq 124 ] && say "$label hit the ${TIMEOUT}s timeout"
  [ -s "$out" ] || { say "$label wrote no $(basename "$out") (exit $rc); see $W/$label.log"; return 1; }
  say "$label done (exit $rc)"
}

render "$SKILL/prompts/worker.md" > "$W/.prompt_worker"
run_agent worker "$AGENT" "$W/.prompt_worker" "$W/card.md"

if python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import triage, pathlib; \
sys.exit(0 if triage.card_findings(pathlib.Path(sys.argv[2]).read_text()) else 1)" "$SKILL" "$W/card.md"; then
  render "$SKILL/prompts/refuter.md" > "$W/.prompt_refuter"
  if ! run_agent refuter "$REFUTER" "$W/.prompt_refuter" "$W/independent.txt"; then
    echo "NONE AVAILABLE -- the refuter agent did not finish on this run" > "$W/independent.txt"
    sed -i '0,/^Review (advisory)/s//Review (advisory, not independently refuted)/' "$W/card.md"
  fi
  python3 "$SKILL/triage.py" apply-refutation "$W/card.md" "$W/independent.txt"
else
  echo "NONE AVAILABLE -- the card has no findings to refute" > "$W/independent.txt"
fi

say "gates..."
GATES_RC=0
python3 "$SKILL/triage.py" gates "$W" "$PROJ" || GATES_RC=$?
say "card: $W/card.md  gates: $W/GATES.txt (exit $GATES_RC)"
exit "$GATES_RC"
