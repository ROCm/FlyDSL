#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Headless review of one PR: fetch -> worker agent -> independent refuter agent -> gates.
# Nothing is posted; the card, the gate report and usage.txt (each agent's wall time and token
# use) are left in $WORK.
#
# usage: run_one.sh <PR | owner/repo#N> [owner/repo]
#
#   FLYDSL_REVIEW_AGENT           agent CLI, default `claude` (run as `<agent> -p --dangerously-skip-permissions`)
#   FLYDSL_REVIEW_REFUTER_AGENT   a different CLI or model for Step 8.7, default the same agent
#   FLYDSL_REVIEW_SANDBOX_IMAGE   run both agents in this Docker image instead of on the host
#   FLYDSL_REVIEW_SANDBOX_NETWORK Docker network for that container, default host
#   FLYDSL_AGENT_TIMEOUT          seconds per agent, default 2400
#
# The agents read untrusted PR text with permissions skipped. In the sandbox they see the checkout
# read-only and $WORK read-write, run as the calling user, and get no GitHub credentials and no
# home directory, so they cannot post to GitHub or read the host's tokens. Model access comes from
# ANTHROPIC_BASE_URL, which must be a gateway that holds the credential and ignores the client's
# x-api-key: no real token is passed in. `gateway.sh start` runs one on a network whose only
# reachable peer is the gateway, and prints the two variables that point the sandbox at it.
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
  local label="$1" cli="$2" prompt="$3" out="$4" rc=0 started=$SECONDS
  say "$label..."
  if [ -n "${FLYDSL_REVIEW_SANDBOX_IMAGE:-}" ]; then
    local bin env_args=()
    bin="$(readlink -f "$(command -v "$cli")")"
    mkdir -p "$W/.home"
    for v in ANTHROPIC_BASE_URL ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL \
             ANTHROPIC_DEFAULT_HAIKU_MODEL ANTHROPIC_SMALL_FAST_MODEL CLAUDE_CODE_EFFORT_LEVEL \
             CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC; do
      [ -n "${!v:-}" ] && env_args+=(-e "$v")
    done
    # The placeholder key only satisfies the CLI's start-up check; the gateway supplies the real one.
    timeout "$TIMEOUT" docker run --rm --network "${FLYDSL_REVIEW_SANDBOX_NETWORK:-host}" --user "$(id -u):$(id -g)" \
      -e HOME="$W/.home" -e ANTHROPIC_API_KEY=sandbox-has-no-credential "${env_args[@]}" \
      -v "$PROJ:$PROJ:ro" -v "$W:$W" -v "$bin:/usr/local/bin/review-agent:ro" -w "$PROJ" \
      --entrypoint /usr/local/bin/review-agent "$FLYDSL_REVIEW_SANDBOX_IMAGE" \
      -p --dangerously-skip-permissions --output-format json "$(cat "$prompt")" \
      > "$W/$label.json" 2> "$W/$label.log" || rc=$?
  else
    (cd "$PROJ" && timeout "$TIMEOUT" "$cli" -p --dangerously-skip-permissions --output-format json "$(cat "$prompt")") \
      > "$W/$label.json" 2> "$W/$label.log" || rc=$?
  fi
  usage "$label" "$((SECONDS - started))" >> "$W/usage.txt"
  [ "$rc" -eq 124 ] && say "$label hit the ${TIMEOUT}s timeout"
  [ -s "$out" ] || { say "$label wrote no $(basename "$out") (exit $rc); see $W/$label.log"; return 1; }
  say "$label done (exit $rc): $(tail -1 "$W/usage.txt")"
}

usage() {  # <label> <seconds> -> one line from the CLI's JSON result
  python3 - "$W/$1.json" "$1" "$2" <<'PY'
import json, sys
path, label, secs = sys.argv[1:4]
try:
    data = json.loads(open(path).read() or "null")
except (OSError, ValueError):
    data = None
if isinstance(data, list):  # `verbose: true` in the CLI settings prints the whole transcript
    data = next((x for x in reversed(data) if isinstance(x, dict) and x.get("type") == "result"), None)
data = data if isinstance(data, dict) else {}
u = data.get("usage") or {}
inp = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
cost = data.get("total_cost_usd")
print(", ".join([
    f"{label}: {secs}s wall, {data.get('num_turns', '?')} turns",
    f"{inp} input + {int(u.get('output_tokens') or 0)} output tokens" if u else "no usage record",
    f"${cost:.2f} as priced by the CLI" if isinstance(cost, (int, float)) else "cost not reported",
]))
PY
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
[ -f "$W/usage.txt" ] && sed 's/^/  /' "$W/usage.txt"
say "card: $W/card.md  gates: $W/GATES.txt (exit $GATES_RC)"
exit "$GATES_RC"
