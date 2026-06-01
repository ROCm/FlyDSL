# Re-running CI with slash commands or a label

FlyDSL CI runs on a limited pool of self-hosted GPU runners (MI325 / MI355 /
Navi). When a run is preempted, hits a flaky failure, or you just want a clean
re-run on the same commit, you can re-trigger it **without pushing an empty
commit** — either by commenting a slash command, or by applying a label.

## Option 1 — Slash command (preferred, repeatable)

Comment one of these on the PR:

| Command | Effect |
|---|---|
| `/rerun-ci` | Re-run the latest **Fly DSL test** (`flydsl.yaml`) run for the PR's current head commit |
| `/rerun-checks` | Re-run the latest **Checks** (`pre-checks.yaml`) run |
| `/rerun-all` | Re-run both of the above |
| `… failed` | Append `failed` (e.g. `/rerun-ci failed`) to re-run **only the failed jobs** of that run |

## Option 2 — `rerun-ci` label

Apply the **`rerun-ci`** label to the PR to re-run **Fly DSL test**.

> ⚠️ GitHub does **not** re-fire the `labeled` event for a label that is already
> on the PR. To trigger again with the label you must **remove it and re-add it**.
> For repeated reruns, prefer the slash command — it has no such limitation.

The handler reacts to your comment:

- 🚀 — command accepted; the run(s) are being re-triggered (a confirmation comment follows).
- 😕 — you are not allowed to trigger CI on this PR (see permissions below).

> Re-running edits also work: editing an existing comment to contain the command
> re-fires the handler, so you don't need to keep adding new comments.

## Who can use it

- The **PR author** (always, even from a fork), and
- anyone with **write / maintain / admin** permission on the repository.

Other users get the 😕 reaction and no run is triggered. This keeps the limited
GPU runners from being spammed by drive-by comments.

## How it works (and why it's safe)

The handler lives in
[`.github/workflows/slash-command-rerun-ci.yml`](../.github/workflows/slash-command-rerun-ci.yml)
with the shared re-run step in
[`.github/scripts/rerun_ci.sh`](../.github/scripts/rerun_ci.sh). It is triggered
by `issue_comment` (the slash command) and `pull_request_target: labeled` (the
label). Both of these triggers always execute the workflow file from the **base
repo's default branch**, so a fork PR cannot change what the handler does even
though it has `actions: write`. The handler never checks out PR code — it only
re-runs an already-created run — so `pull_request_target` here does not expose
the usual "runs untrusted fork code with secrets" risk.

The handler **re-runs the existing workflow run** for the PR's head SHA
(`gh run rerun <run-id>`), rather than dispatching a ref. This is deliberate:

- The `workflow_dispatch` REST API only accepts a **branch or tag** for `ref`,
  not an arbitrary commit SHA — and a fork PR's head branch does not exist in
  this repository. Re-running the existing run re-executes against the exact PR
  head commit and works for fork PRs too.
- The privileged handler never checks out or runs PR code; it only re-runs a run
  that the normal `pull_request` trigger already created.

### First run on a brand-new PR

A slash command **re-runs an existing run** — so there must already be a run for
the current commit. New PRs already get one because `flydsl.yaml` and
`pre-checks.yaml` trigger on `pull_request`. If for some reason no run exists yet
for the head SHA (e.g. it was never triggered), push a commit or use the
workflow's **Run workflow** button (`workflow_dispatch`) once to seed it; after
that, the slash commands re-run it.

## Permissions used by the handler

```yaml
permissions:
  contents: read
  actions: write        # re-run the target workflow runs
  pull-requests: read
  issues: write         # add a reaction + confirmation comment
```

These are the minimum needed. The handler uses the automatic `GITHUB_TOKEN`; no
PAT or extra secret is required.
