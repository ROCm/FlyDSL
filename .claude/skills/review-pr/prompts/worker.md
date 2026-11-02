You are the reviewer for {{REPO}} PR #{{PR}} (`{{TITLE}}`). Run **Step 1b through Step 9** of the
`review-pr` skill.

Step 1 has already run. Its artifacts are in `{{WORK}}`; the skill is at `{{SKILL}}`; the project
root is `{{PROJ}}`.

1. Read `{{SKILL}}/SKILL.md` in full first. It is the only standard; this prompt adds only what it
   does not cover.
2. Write each artifact as soon as its step is done — `answers.txt` first, then `core_files.txt`,
   `verdicts.txt`, `ai_diagnostic.txt`, `refutations.txt`, `card.md` — and confirm each with
   `wc -l`. A rough file that exists beats a perfect one a dropped connection erased.
3. Cite changed files by their repo-relative path, with no `head/` or `merge-target/` prefix. Get the
   list with `grep '^+++ b/' {{WORK}}/pr.diff | sed 's|^+++ b/||'`.
4. `merge-target/` and `head/` are read-only. Never run `git checkout`, `switch`, `reset` or `stash`
   in `{{PROJ}}`. For an experiment, copy files into a new directory under `/tmp`.
5. **Step 8.7 is not yours.** A separate reader who has not seen your reasoning writes
   `independent.txt`. Do not write it and do not predict it; the `independent` gate is expected to
   be red when you finish.
6. Do not post anything to GitHub.

Finally, run the Step 9 gates and report, in English, the line count of each artifact, each gate's
result, and the number of findings on the card. Do not restate the card.
