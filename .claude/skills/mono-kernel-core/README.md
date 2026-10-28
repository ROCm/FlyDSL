# MI355X mono-kernel development recipes

Develop low-concurrency inference mono-kernels for AMD Instinct MI355X GPUs
(`gfx950`). Start with the operator contract and a reproducible reference, bring
up standalone stages, and extend fusion through an inspect–change–validate loop.
The recipes cover kernel development, optimization, and unit testing.

| Skill | Use |
|---|---|
| [dev](mi355/dev/SKILL.md) | Define functional and numerical boundaries; implement stages, persistent workgroups, and incremental fusion; plan TP/EP/CP execution |
| [tune](mi355/tune/SKILL.md) | Select an optimization hypothesis, run controlled comparisons, and interpret PMC/ATT evidence using embedded case studies |
| [test](mi355/test/SKILL.md) | Validate intermediate and final results, cache and state updates, task coverage, replay, forward progress, and paired latency |

In this checkout, the three relative `mi355-mono-kernel-*` links in
`.claude/skills` expose the skills to Claude Code. Start with
`/mi355-mono-kernel-dev`; use `/mi355-mono-kernel-tune` or
`/mi355-mono-kernel-test` for the corresponding task.

Load supporting references only when the current step or failure needs them.
Keep all three sibling directories under `mi355` when copying the package so
cross-skill links continue to resolve. Model configuration and runtime tensors
come from the target project; optimization cases are included in the package.
Framework integration and serving validation are outside this suite's scope.

Keep optimization cases concise and self-contained: context, the change,
measurements, and limits. References must resolve within this package or point
to formal documentation; do not reference PRs, commits, or external source
snapshots.
