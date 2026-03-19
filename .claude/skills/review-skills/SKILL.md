---
name: review-skills
description: >
  Meta-skill that reviews all existing skills, applies accumulated improvement
  notes from memory, and updates SKILL.md files. Checks each skill for outdated
  steps, missing edge cases, format inconsistencies, and lessons learned from
  actual usage. Run periodically to keep skills accurate and effective.
  Usage: /review-skills
tools: Read,Edit,Write,Bash,Grep,Glob
---

# Review and Improve Skills

Systematically review all skills, apply accumulated feedback, and update them.

## Step 1: Gather Context

### 1.1 List all skills

```bash
ls -d ~/.claude/skills/*/SKILL.md /path/to/project/.claude/skills/*/SKILL.md 2>/dev/null
```

Read each SKILL.md to build an inventory.

### 1.2 Read improvement log

Read the auto memory improvement log:

```
~/.claude/projects/*/memory/skill-improvements.md
```

This file contains dated observations from actual skill usage — things that
worked, things that broke, missing steps, incorrect assumptions, etc.

### 1.3 Read MEMORY.md

Check for any user preferences or conventions that skills should follow.

---

## Step 2: Review Each Skill

For each skill, evaluate the following checklist:

### Accuracy

- [ ] Do the shell commands actually work? (correct flags, paths, tool names)
- [ ] Are file formats and data structures described correctly?
- [ ] Are version-specific details still current? (tool versions, API changes)
- [ ] Do examples match the actual output format?

### Completeness

- [ ] Are all edge cases covered? (empty input, missing files, permission errors)
- [ ] Is error handling documented for common failure modes?
- [ ] Are prerequisites listed? (tools, packages, permissions)
- [ ] Are output formats fully described?

### Consistency

- [ ] Does the YAML frontmatter have correct `name`, `description`, `tools`?
- [ ] Is the description concise but complete for skill discovery?
- [ ] Does the `tools` field list all tools actually needed?
- [ ] Is the writing style consistent across skills? (imperative, structured)

### Lessons from Usage

- [ ] Apply all pending items from `skill-improvements.md`
- [ ] Mark applied items as done: `- [2026-xx-xx] [APPLIED] ...`
- [ ] Remove items that are no longer relevant

---

## Step 3: Apply Updates

For each skill that needs changes:

1. Read the current SKILL.md
2. Apply improvements using the Edit tool (prefer minimal diffs)
3. Summarize what was changed and why

### Update rules

- **DO** fix incorrect commands, formats, or data structures
- **DO** add missing edge cases discovered during actual usage
- **DO** update examples to match real output
- **DO** add new sections for patterns discovered in practice
- **DON'T** bloat skills with hypothetical scenarios never encountered
- **DON'T** change working procedures that have been validated in practice
- **DON'T** add redundant explanations — keep skills concise and actionable

---

## Step 4: Update Improvement Log

After applying changes, update `skill-improvements.md`:

- Mark applied items with `[APPLIED]`
- Remove stale items that no longer apply
- Keep the file clean and actionable

---

## Step 5: Report

Generate a summary of all changes:

```
Skills Review Summary
=====================

Reviewed: N skills
Updated:  M skills
Skipped:  K skills (no changes needed)

Changes:
  - kernel-trace-analysis: Added waitcnt dependency parsing docs, fixed type encoding table
  - bisect-perf-regression: No changes (no usage data yet)
  ...

Pending (need more data):
  - prefetch-data-load: Not yet used in practice, cannot validate
  ...
```

---

## When to Run

- After completing a major task that used multiple skills
- When the user explicitly asks to review skills
- When accumulated items in `skill-improvements.md` exceed 5 pending entries
- After discovering a bug or gap in a skill during usage
