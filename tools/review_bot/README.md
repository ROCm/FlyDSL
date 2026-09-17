# Local FlyDSL review operator

This directory contains the disabled local pilot operator for `ROCm/FlyDSL`.
Each invocation is finite. The user timer supplies the two-minute cadence, and
the watcher uses both a file lock and a SQLite primary key to serialize and
claim `(repository id, PR number, head OID, engine SHA)` exactly once. A base-only
move can stale an active run but never retries a terminal head.

The first successful scan only seeds all eligible open heads. Later scans
review new keys. An interrupted, rejected, stale, or otherwise incomplete key
is terminal and is not retried; a newer head remains eligible.

## Required deployment inputs

The host must provide:

- `/usr/bin/gh`, authenticated through its normal credential source as
  `jhinpan` (immutable user id `47354855`);
- `/usr/bin/docker` and permission to use its normal local daemon socket;
- `/usr/bin/git` and public GitHub access for fetching the exact PR and base
  refs;
- a clean, protected FlyDSL engine checkout at the exact deployed commit;
- the node's normal Claude gateway environment: `ANTHROPIC_AUTH_TOKEN` and
  `ANTHROPIC_BASE_URL`. The watcher checks only presence and passes variable
  names to Docker; it never prints or stores either value. The pilot requires
  the verified local endpoint `http://127.0.0.1:8882`. Listener binding and
  host-firewall policy belong to the existing gateway deployment, not this bot;
- an AppArmor configuration that permits the pinned sandbox runtime to create
  the required `bubblewrap` user namespace, verified with non-value canaries;
- a review image pinned by repository digest, built from a Debian-compatible
  Node/npm base image pinned by digest and exact versions of Claude Code and
  `@anthropic-ai/sandbox-runtime`.

The deployed review engine must support these runner options:

```text
--scope-manifest /review-input/scope-manifest.json
--execution-profile untrusted-container
--model opus
--effort max
--concurrency 9
--agent-timeout 1200
--phase-timeout 3600
--claude-path /usr/local/bin/claude
--run-dir /review-run
```

Its trusted publisher must support
`--expected-implementation-sha256` and `--expected-publisher-id`, in addition
to `--expected-repository-id`, `--expected-author-id`,
`--expected-author-login`, `--expected-head`, and `--publish-severity`.

The Dockerfile pins its linux/amd64 Node base manifest, Claude Code 2.1.274,
sandbox runtime 0.0.76, and both npm integrity values. Re-resolve and review
all four values together when updating. Record the built image as either a
local `sha256:<image-id>` or a repository `name@sha256:<digest>`; mutable tags
are rejected.

## Configuration

Copy `config.example.json` outside the repository, replace every placeholder,
and make the file readable only by the operator account. `state_root` defaults
to `~/.local/state/flydsl-review-bot` when omitted and is forced to mode 0700.
The implementation hash is the exact value the deployed runner writes to
`result.json`; it is not the engine Git SHA.
`publish_enabled` defaults to `false`, causing the trusted publisher itself to
run with `--dry-run`. Changing it to `true` is the explicit live-publication gate.

The review container has no GPU request, GitHub config, GitHub token, or
Docker socket. It uses the host network only so the trusted Claude core can
reach the loopback-only model gateway; the `untrusted-container` sandbox denies network to
PR-influenced tool commands. The root is read-only; capabilities are dropped;
privilege gain is disabled; PID, memory, CPU, and file-descriptor limits are
applied. The engine, source, and manifest mounts are read-only. Only the local
run artifact directory is a persistent writable mount. Claude's subprocess
scrubber and sandbox credential deny rules remove the model token from every
PR-influenced tool command while retaining it in the trusted CLI core.

Run one existing eligible PR as a publication-disabled canary only after the
engine interfaces and pinned image have passed deployment validation. This
also seeds every other currently open eligible head without reviewing it:

```sh
/usr/bin/python3 /absolute/path/to/tools/review_bot/watch.py \
  --config /absolute/path/to/config.json --canary-pr 1137
```

Output is one summary-only JSON line. Full source snapshots, result data,
container logs, publisher logs, and operator diagnostics remain under the
mode-0700 state root. Terminal run directories older than `retention_days`
are removed; the SQLite claims remain, so retention never permits a retry.

## User systemd templates

The checked-in units are templates, not installed or enabled units. Replace
all three placeholders in the service:

- `@PYTHON@` with the trusted absolute Python path;
- `@REVIEW_BOT_ROOT@` with the clean deployed repository path;
- `@CONFIG_PATH@` with the protected config path.

Install the rendered service and the unchanged timer into
`~/.config/systemd/user/`, then run `systemctl --user daemon-reload`. This
still leaves the timer disabled. Inspect the rendered units and complete a
publication-disabled canary before activation.

The service uses `PassEnvironment` for the two model gateway variables. Before
a manual service run or timer activation, the operator must import their
current values into the user manager without printing them:
`systemctl --user import-environment ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL`.
Missing variables fail closed before any review.

Only after the canary passes and `publish_enabled` is explicitly changed to
`true`, pilot approval may enable the timer as a separate operator action:

```sh
systemctl --user enable --now flydsl-review-bot.timer
```

Disable it with `systemctl --user disable --now flydsl-review-bot.timer`.
Neither the repository nor the watcher performs either action.

## Durable outcomes

`SEEDED`, `REJECTED`, `STALE`, `INCOMPLETE`, and `COMPLETE` are terminal.
Startup converts any prior `CLAIMED`, `PREPARING`, `RUNNING`, `VALIDATING`, or
`PUBLISHING` row to `INCOMPLETE` with reason `interrupted`. Before that
transition, it removes any matching orphan container only after all immutable
identity labels match; cleanup failure leaves the claim active for another
cleanup attempt and never reruns the review. Publication is possible only
when the container exited successfully, the bounded result is `COMPLETE` and
identity-bound, a live `gh` lookup still reports the claimed head, and the
authenticated publisher identity is rechecked. The trusted host publisher
performs its own final live check and idempotent GitHub write.
