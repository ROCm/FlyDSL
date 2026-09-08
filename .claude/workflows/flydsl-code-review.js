export const meta = {
  name: 'flydsl-code-review',
  description: 'FlyDSL code review — one finder agent per review angle, an independent verifier for every candidate, then a ranked, capped findings report.',
  whenToUse: 'Launched by the /flydsl-code-review skill. Pass args as an optional review target: a PR number, branch, ref range, path, or free-form review instructions (e.g. "only review kernels/attention", "focus on the LDS changes").',
  phases: [
    { title: 'Scope', detail: 'resolve the diff command and changed files' },
    { title: 'Find', detail: 'one agent per review angle' },
    { title: 'Verify', detail: 'one independent verifier per candidate' },
    { title: 'Sweep', detail: 'a fresh reviewer hunting only for gaps' },
    { title: 'Synthesize', detail: 'merge, rank, cap' },
  ],
}

const PER_ANGLE = 6
// 9 angles x PER_ANGLE overruns a 25-slot budget badly: a trial run produced 61
// candidates, deduped to 42, and dropped 17 unverified. Sized so the finder
// phase is rarely truncated after dedup.
const FINDER_VERIFY_BUDGET = 40
// Held back from the finder phase. Sweep runs last, so a single shared budget
// starves it every time — an earlier revision of this script logged
// "sweep: 0 new candidates verified" for exactly that reason.
const SWEEP_VERIFY_BUDGET = 10
const SWEEP_MAX = 8
const MAX_FINDINGS = 12

const SKILL = '.claude/skills/flydsl-code-review/SKILL.md'
// --comment is handled by the skill after this workflow returns (it posts the
// findings via scripts/post_review.py). Strip it here so a caller that forwards
// the raw argument string does not leave the Scope agent reading "--comment" as
// a free-form review instruction.
const TARGET = (typeof args === 'string' ? args : '')
  .replace(/(^|\s)--comment(?=\s|$)/g, ' ')
  .trim()

// The angle prose lives in SKILL.md, not here. Each finder reads its own section
// so the two never drift; workflow scripts have no filesystem access, but their
// agents do.
const ANGLES = [
  { label: 'trace-time',   kind: 'correctness', section: '## Angle A — trace-time vs runtime semantics' },
  { label: 'addressing',   kind: 'correctness', section: '## Angle B — memory addressing and out-of-bounds' },
  { label: 'sync-lds',     kind: 'correctness', section: '## Angle C — synchronization, LDS, and value lifetime' },
  { label: 'arch-atom',    kind: 'correctness', section: '## Angle D — architecture and atom contracts' },
  { label: 'removed',      kind: 'correctness', section: '## Angle E — removed-behavior auditor' },
  { label: 'cross-layer',  kind: 'correctness', section: '## Angle F — cross-layer tracer' },
  { label: 'conventions',  kind: 'convention',  section: '## Angle G — repo conventions and API stability' },
  { label: 'reuse',        kind: 'convention',  section: '## Angle H — reuse, simplification, and altitude' },
  { label: 'test-doc',     kind: 'convention',  section: '## Angle I — test and documentation contract' },
]

const SCOPE_SCHEMA = {
  type: 'object', required: ['diffCommand', 'files', 'summary'],
  properties: {
    diffCommand: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
    conventions: { type: 'string' },
  },
}
const CANDIDATES_SCHEMA = {
  type: 'object', required: ['candidates'],
  properties: {
    candidates: { type: 'array', items: {
      type: 'object', required: ['file', 'summary', 'failure_scenario'],
      properties: {
        file: { type: 'string' },
        line: { type: 'number' },
        summary: { type: 'string' },
        failure_scenario: { type: 'string' },
      },
    }},
  },
}
const VERDICT_SCHEMA = {
  type: 'object', required: ['verdict', 'evidence'],
  properties: {
    verdict: { enum: ['CONFIRMED', 'PLAUSIBLE', 'REFUTED'] },
    evidence: { type: 'string' },
  },
}
const REPORT_SCHEMA = {
  type: 'object', required: ['summary', 'findings'],
  properties: {
    summary: { type: 'string' },
    findings: { type: 'array', items: {
      type: 'object', required: ['file', 'summary', 'failure_scenario', 'verdict'],
      properties: {
        file: { type: 'string' },
        line: { type: 'number' },
        summary: { type: 'string' },
        failure_scenario: { type: 'string' },
        verdict: { enum: ['CONFIRMED', 'PLAUSIBLE'] },
      },
    }},
  },
}

// The verdict ladder is the one piece of prompt text duplicated from SKILL.md:
// a verifier that has to Read the skill before judging spends a tool call to
// learn three definitions, and the recall bias is too important to risk it
// skipping the read.
const VERDICT_LADDER =
  '- **CONFIRMED** — you can name the inputs, state, or target that trigger it and the\n' +
  '  resulting wrong output, crash, hang, or CI failure. Quote the line.\n' +
  '- **PLAUSIBLE** — the mechanism is real but the trigger is uncertain (timing,\n' +
  '  architecture, config, shape). State what would confirm it.\n' +
  '- **REFUTED** — factually wrong, or already guarded. Quote the line that proves it.\n\n' +
  '**Default to PLAUSIBLE.** Do not refute a candidate for being "speculative" or for\n' +
  '"depending on runtime state" when the state is realistic. On a GPU these are all\n' +
  'PLAUSIBLE, not REFUTED: a race between waves, an OOB on a boundary tile the code\n' +
  'does not exclude, a NaN on an all-masked partition, a divergent barrier on a path\n' +
  'taken only by the last workgroup, a wave32 target the kernel was not tested on, an\n' +
  'i32 overflow at a large shape.\n\n' +
  '**REFUTED only when constructible from the code:** factually wrong (quote the actual\n' +
  'line); provably impossible from a type, constant, or invariant (show it); already\n' +
  'handled in this diff (cite the guard); or pure style with no observable effect.'

// ---------------------------------------------------------------- Scope

phase('Scope')
const scope = await agent(
  'Establish the scope of a FlyDSL code review.\n\n' +
  (TARGET
    ? 'Review target / instructions (passed by the user, verbatim): "' + TARGET + '". If it names a PR number, branch, ref range, or file path, build the matching git diff command for it (use `gh pr diff <n>` for a PR); if it is a free-form instruction, honor any scope restriction when building the diff command and start from the current branch diff (`git diff @{upstream}...HEAD`, falling back to `git diff main...HEAD` or `git diff HEAD~1`) for whatever it does not narrow.\n'
    : 'No explicit target — review the current branch: prefer `git diff @{upstream}...HEAD` (fall back to `git diff main...HEAD` or `git diff HEAD~1`), and if there are uncommitted changes also include `git diff HEAD`.\n') +
  '\n1. Determine the exact diff command(s) for the review and run them to confirm they produce a non-empty diff.\n' +
  '2. List the changed files.\n' +
  '3. Summarize what changed in one paragraph. Note which layers are touched: Python DSL (python/flydsl/), kernels (kernels/), C++ dialect (lib/, include/), MLIR FileCheck tests (tests/mlir/), pytest (tests/).\n' +
  '4. Read the root CLAUDE.md and any CLAUDE.md near the changed files, and note the conventions a reviewer should hold this diff to.\n\n' +
  'Return diffCommand exactly as a reviewer should run it. Structured output only.',
  { label: 'scope', schema: SCOPE_SCHEMA }
)
if (!scope) {
  return { error: 'Scope agent returned no result — cannot establish the review scope.' }
}
if (!scope.files || scope.files.length === 0) {
  return { target: TARGET || undefined, summary: 'No changes found to review.', findings: [], stats: { finders: 0, candidates: 0, verified: 0 } }
}
log('reviewing ' + scope.files.length + ' changed files')

const SCOPE_BLOCK =
  '## Review scope\n' +
  'Diff command: ' + scope.diffCommand + '\n' +
  'Changed files (' + scope.files.length + '):\n' +
  scope.files.map(f => '  - ' + f).join('\n') + '\n\n' +
  '## What changed\n' + scope.summary + '\n\n' +
  '## Conventions\n' + (scope.conventions || '(none noted)') + '\n' +
  // The user's verbatim target rides along to every finder, verifier, and sweep
  // agent so focus areas and skip requests are honored, not just used for diff
  // scoping.
  (TARGET
    ? '\n## User instructions (verbatim)\n' + TARGET + '\nHonor any scope restrictions or focus areas stated above — they take precedence over your angle\'s default breadth. Do not surface findings the instructions ask to skip.\n'
    : '')

// ---------------------------------------------------------------- Prompts

const FINDER_PROMPT = a =>
  '## FlyDSL code-review finder — ' + a.label + '\n\n' + SCOPE_BLOCK + '\n' +
  'Read `' + SKILL + '`, section `' + a.section + '`. That section is your assigned\n' +
  'review angle. Run the diff command above and review ONLY through that lens —\n' +
  'other angles are covered by other agents running in parallel.\n\n' +
  'Read the enclosing function for each hunk: bugs on unchanged lines of a touched\n' +
  'function are in scope.\n\n' +
  (a.kind === 'convention'
    ? 'You are hunting convention violations and cleanup, not crashes. In\n' +
      '`failure_scenario`, state the concrete cost — what fails in CI, what is\n' +
      'duplicated, what becomes arch-fragile — instead of inventing a crash.\n\n'
    : '') +
  'Surface up to ' + PER_ANGLE + ' candidate findings, each with file, line, a one-line\n' +
  'summary, and a concrete failure_scenario. Pass every candidate with a nameable\n' +
  'failure scenario through — do not silently drop half-believed candidates; an\n' +
  'independent verifier judges them next. If nothing qualifies, return an empty list.\n\n' +
  'Structured output only.'

const VERIFIER_PROMPT = c =>
  '## FlyDSL code-review verifier\n\n' + SCOPE_BLOCK + '\n' +
  '## Candidate finding\n' +
  'File: ' + c.file + (c.line != null ? ':' + c.line : '') + '\n' +
  'Summary: ' + c.summary + '\n' +
  'Failure scenario: ' + c.failure_scenario + '\n\n' +
  'Run the diff command above, read the relevant file(s), and return exactly one verdict:\n\n' +
  VERDICT_LADDER + '\n\n' +
  'Structured output only. Evidence must quote or cite the relevant line(s).'

// ---------------------------------------------------------------- Find + Verify
// Dedup state accumulates as finders complete (pipeline has no barrier).

const dedupKey = c => c.file + ':' + (c.line != null ? Math.round(c.line / 5) * 5 : 'x:' + c.summary.toLowerCase().slice(0, 40))
const seen = new Map()
const dupes = []
const budgetDropped = []
let verifySlots = FINDER_VERIFY_BUDGET

function verifyCandidate(c) {
  const short = (c.file || '').split('/').pop()
  return agent(VERIFIER_PROMPT(c), { label: 'verify:' + short, phase: 'Verify', schema: VERDICT_SCHEMA })
    .then(v => (v ? { ...c, verdict: v.verdict, evidence: v.evidence } : null))
}

function admit(candidates, kind) {
  const novel = candidates.filter(c => {
    const key = dedupKey(c)
    if (seen.has(key)) {
      dupes.push(c)
      return false
    }
    if (verifySlots <= 0) {
      budgetDropped.push(c)
      return false
    }
    seen.set(key, true)
    verifySlots--
    return true
  })
  return parallel(novel.map(c => () => verifyCandidate({ ...c, kind })))
}

const finderResults = await pipeline(
  ANGLES,

  a => agent(FINDER_PROMPT(a), { label: a.label, phase: 'Find', schema: CANDIDATES_SCHEMA }).then(r => {
    if (!r) return { angle: a, candidates: [] }
    log(a.label + ': ' + r.candidates.length + ' candidates')
    return { angle: a, candidates: r.candidates.slice(0, PER_ANGLE) }
  }),

  result => admit(result.candidates, result.angle.kind)
)

let verified = finderResults.flat().filter(Boolean)

// ---------------------------------------------------------------- Sweep
// One fresh finder that holds the verified list and hunts only for gaps.

phase('Sweep')
const knownBlock = verified.length > 0
  ? verified.map(c => '- ' + c.file + (c.line != null ? ':' + c.line : '') + ' — ' + c.summary).join('\n')
  : '(none)'
const sweep = await agent(
  '## FlyDSL code-review sweep — gaps only\n\n' + SCOPE_BLOCK + '\n' +
  '## Already-found candidates (do NOT re-derive or re-confirm these)\n' + knownBlock + '\n\n' +
  'Re-read the diff and the enclosing functions looking ONLY for defects not already\n' +
  'listed. Focus on what the first pass tends to miss: a bug in unchanged lines of a\n' +
  'touched function; code that moved between files and lost a guard, mask, or anchor on\n' +
  'the way; setup/teardown asymmetry in tests; a default value flipped; a constant\n' +
  'changed in one place but not its mirror; an interaction between two separately\n' +
  'correct hunks.\n\n' +
  'Surface up to ' + SWEEP_MAX + ' additional candidates. If nothing new, return an empty\n' +
  'list — do not pad.\n\nStructured output only.',
  { label: 'sweep', phase: 'Sweep', schema: CANDIDATES_SCHEMA }
)
if (sweep && sweep.candidates.length > 0) {
  // Top the budget back up so the sweep is judged on its own reserve rather
  // than on whatever the finder phase happened to leave behind.
  verifySlots = Math.max(verifySlots, SWEEP_VERIFY_BUDGET)
  const fresh = sweep.candidates.slice(0, SWEEP_MAX)
  const sweepVerified = await admit(fresh, 'correctness')
  const kept = sweepVerified.filter(Boolean)
  log('sweep: ' + kept.length + ' new candidates verified')
  verified = verified.concat(kept)
}

const surviving = verified.filter(c => c.verdict !== 'REFUTED')
const refuted = verified.filter(c => c.verdict === 'REFUTED')
log('Verify done: ' + verified.length + ' verified — ' + surviving.length + ' kept, ' + refuted.length + ' refuted')

const stats = {
  finders: ANGLES.length,
  candidates: seen.size + dupes.length + budgetDropped.length,
  verified: verified.length,
  refuted: refuted.length,
  dupes: dupes.length,
  budgetDropped: budgetDropped.length,
}
if (budgetDropped.length > 0) {
  log('NOTE: ' + budgetDropped.length + ' candidates were dropped unverified — the verifier budget ran out. Coverage is not complete.')
}

if (surviving.length === 0) {
  return {
    target: TARGET || undefined,
    summary: 'No findings survived verification.',
    findings: [],
    stats,
  }
}

// ---------------------------------------------------------------- Synthesize
// Correctness bugs outrank convention findings when the cap forces a cut;
// CONFIRMED outranks PLAUSIBLE within each group.

phase('Synthesize')
const rank = c => (c.kind === 'convention' ? 2 : 0) + (c.verdict === 'PLAUSIBLE' ? 1 : 0)
const ranked = surviving.slice().sort((a, b) => rank(a) - rank(b))
const block = ranked.map((c, i) =>
  '### [' + i + '] ' + c.file + (c.line != null ? ':' + c.line : '') + ' (' + c.verdict + (c.kind === 'convention' ? ', convention' : '') + ')\n' +
  c.summary + '\nFailure scenario: ' + c.failure_scenario + '\nVerifier evidence: ' + c.evidence + '\n'
).join('\n')

const report = await agent(
  '## Synthesis: final FlyDSL code-review report\n\n' +
  ranked.length + ' findings survived independent verification.\n\n' + block + '\n' +
  '## Instructions\n' +
  '1. Merge findings that describe the same defect (same root cause) — combine their evidence.\n' +
  '2. Rank most-severe first. Correctness bugs always outrank convention findings.\n' +
  '3. Keep at most ' + MAX_FINDINGS + ' findings; drop the least severe beyond the cap.\n' +
  '4. Write a 2-3 sentence summary of the review.\n\nStructured output only.',
  { label: 'synthesize', schema: REPORT_SCHEMA }
)

// Synthesis skipped or errored — salvage the verified findings unmerged rather
// than discarding the run.
const findings = report
  ? report.findings.slice(0, MAX_FINDINGS)
  : ranked.slice(0, MAX_FINDINGS).map(c => ({
      file: c.file, line: c.line, summary: c.summary, failure_scenario: c.failure_scenario, verdict: c.verdict,
    }))

return {
  target: TARGET || undefined,
  summary: report ? report.summary : 'Synthesis step was skipped or failed — returning verified findings unmerged.',
  findings,
  refuted: refuted.map(c => ({ file: c.file, line: c.line, summary: c.summary })),
  stats: { ...stats, reported: findings.length },
}
