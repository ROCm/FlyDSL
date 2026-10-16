---
name: code-review
description: >
  FlyDSL pull request review for Copilot code review on GitHub. Applies the review-pr skill in
  .claude/skills/review-pr, with rules derived per diff by its triage.py, to catch
  silent-correctness bugs, JIT cache-key mistakes, architecture-capability holes, index-width
  overflow, unbounded buffer descriptors, layout-algebra errors, dead flags, and tests or
  benchmarks that cannot fail.
---

# FlyDSL code review on GitHub

`.claude/skills/review-pr/SKILL.md` is the standard; read it in full first. This file changes only
what a Copilot code review cannot do as written there. Where the two disagree, this file wins.

## Evidence, instead of fetch.sh

Do not run `fetch.sh` or `run_one.sh`: they call `gh`, and this sandbox blocks `api.github.com`.
`.github/workflows/copilot-code-review.yml` fetches the full history before the firewall starts, so
the block below needs no network. Run it as one shell command from the repository root, with the PR
base branch and title filled in. It stops at the first failed command, including an empty diff, and
writes `rules.txt` only if every command succeeded. Use the printed `WORK=` directory as `$W` in
every later step.

```bash
W=$(mktemp -d /tmp/review-XXXXXX); T=.claude/skills/review-pr/triage.py; echo "WORK=$W"
cat > $W/title.txt <<'EOF'
<PR title>
EOF
(
set -euo pipefail
B=origin/<PR base branch>
[ "$(git rev-parse --is-shallow-repository)" = false ] || git fetch -q --unshallow origin
git rev-parse -q --verify "$B" > /dev/null || git fetch -q origin "+refs/heads/${B#origin/}:refs/remotes/$B"
git diff --no-color --no-ext-diff "$B...HEAD" > $W/pr.diff
[ -s $W/pr.diff ]
git worktree add -q --detach $W/merge-target "$B"
python3 $T rules $W/pr.diff "$(cat $W/title.txt)" . > $W/rules.tmp
python3 $T expand $W/rules.tmp .claude/skills/review-pr/rules.md > $W/rules_expanded.txt
python3 $T scan $W/pr.diff > $W/scan.txt
python3 $T core $W/pr.diff . > $W/core_expected.txt
python3 $T bounds $W/pr.diff . > $W/bounds.txt
python3 $T citest $W/pr.diff . . > $W/ci_coverage.txt
python3 $T benchcov $W/pr.diff . > $W/bench_compare.txt
mv $W/rules.tmp $W/rules.txt
)
echo "evidence exit=$?"
```

If it does not print `evidence exit=0`, do not rebuild the evidence by hand: review from the diff you
were given, and the gates will report the missing `rules.txt`. Read CI checks and human review
comments through the GitHub MCP server. If the `test (linux-flydsl-*)` job logs are not readable
there, the Perf line is `NOT MEASURED`, as Step 9 allows.

## Steps 2 to 9

Run Steps 2 to 8.6 as written, with `$W` as `$WORK`. No independent reader exists for Step 8.7:
write `NONE AVAILABLE -- Copilot code review has no independent reader` to `$W/independent.txt`.
Write the card to `$W/card.md` with the review line reading
`Review (advisory, not independently refuted): ...`, then run
`python3 .claude/skills/review-pr/triage.py gates "$W" .`. If a gate is red, fix that step once and
run the gates again; then stop, whatever they report.

## Output

Unlike a `review-pr` card, this review is posted. Post the card's findings as review comments on
their `file:line`, keeping the three parts, severity and `[verified]` / `[inferred]` tag. No rule
ids. If a gate is still red, also post a 📝 comment on the first changed line naming the red gates.
