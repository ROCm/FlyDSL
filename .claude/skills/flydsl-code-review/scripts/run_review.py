#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Run or resume a review against a pinned checkout; stdout is the result JSON.

Examples:
    run_review.py 1106
    run_review.py --base HEAD~3 --head HEAD
    run_review.py --path kernels/attention --instructions 'focus on LDS changes'
    run_review.py --resume /tmp/flydsl-review-<run-id>

Each agent is a separate Claude Code CLI process. No Workflow/inline fallback,
model override, publication, or permission bypass is implicit in this runner.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from review_common import (
    ANGLES,
    PER_ANGLE,
    PREFLIGHTS,
    SCHEMA_VERSION,
    SEVERITIES,
    SWEEP_MAX,
    VERDICTS,
    build_report,
    canonical,
    collect_candidates,
    digest,
    judged_candidates,
    normalize_path,
    stage_output,
    validate_output,
    validate_preflight,
)

SCRIPTS = Path(__file__).resolve().parent
SKILL = SCRIPTS.parent / "SKILL.md"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def command_result(
    *argv: str,
    cwd: Path,
    stdin: str | None = None,
    deadline: float | None = None,
    cancelled: threading.Event | None = None,
) -> subprocess.CompletedProcess:
    deadline = min(deadline or float("inf"), time.monotonic() + 120)
    if (cancelled and cancelled.is_set()) or time.monotonic() >= deadline:
        raise TimeoutError("command cancelled or phase deadline exceeded")
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        first = True
        while True:
            if (cancelled and cancelled.is_set()) or time.monotonic() >= deadline:
                raise TimeoutError("command cancelled or phase deadline exceeded")
            try:
                stdout, stderr = process.communicate(input=stdin if first else None, timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                first = False
    finally:
        stop_process(process)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def command(*argv: str, **options) -> str:
    result = command_result(*argv, **options)
    if result.returncode:
        raise RuntimeError(f"{shlex.join(argv)}: {result.stderr.strip()}")
    return result.stdout


def git(root: Path, *args: str, **options) -> str:
    return command("git", *args, cwd=root, **options)


def revision(root: Path, ref: str, **options) -> str:
    return git(root, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}", **options).strip()


def default_base(root: Path, **options) -> str:
    # A branch's tracking ref often points at its own head, yielding an empty review.
    for ref in ("origin/main", "main", "HEAD^"):
        try:
            return revision(root, ref, **options)
        except RuntimeError:
            pass
    raise ValueError("cannot resolve a review base; pass --base and --head")


def pin_scope(root: Path, run_dir: Path, config: dict, cancelled: threading.Event | None = None) -> dict:
    control = {"deadline": time.monotonic() + config["phase_timeout"], "cancelled": cancelled}

    def scope_git(root, *args, **kwargs):
        return git(root, *args, **kwargs, **control)

    target = config["target"].strip()
    repo = config["repo"]
    pr = config["pr"]
    base_ref, head_ref = config["base"], config["head"]
    paths = list(config["paths"])
    instructions = config["instructions"]
    three_dot = True
    url = re.fullmatch(r"https://github.com/([^/]+/[^/]+)/pull/(\d+)/?", target)
    if url:
        repo, pr = url.group(1), int(url.group(2))
    elif target.isdecimal():
        pr = int(target)
    elif target:
        if base_ref or head_ref or pr:
            raise ValueError("use either a positional target or explicit revision/PR options")
        if (root / target).exists():
            paths.append(str((root / target).resolve().relative_to(root)))
        elif ".." in target and not any(c.isspace() for c in target):
            separator = "..." if "..." in target else ".."
            base_ref, head_ref = target.split(separator, 1)
            base_ref, head_ref = base_ref or "HEAD", head_ref or "HEAD"
            three_dot = separator == "..."
        else:
            try:
                base_ref = revision(root, target, **control)
            except RuntimeError:
                instructions = "\n".join(s for s in (instructions, target) if s)
    paths = sorted({normalize_path(p) for p in paths if p.rstrip("/") not in (".", "./")})
    snapshot = run_dir / "repo"
    snapshot.mkdir(exist_ok=True)
    scope_git(snapshot, "init", "--quiet")
    include_worktree = not (pr or base_ref or head_ref)
    if pr:
        if base_ref or head_ref:
            raise ValueError("--pr cannot be combined with --base or --head")
        repo = (
            repo
            or command(
                "gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner", cwd=root, **control
            ).strip()
        )
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or pr < 1:
            raise ValueError("invalid GitHub repository or PR number")
        metadata = json.loads(command("gh", "api", f"repos/{repo}/pulls/{pr}", cwd=root, **control))
        base_oid, head_oid = metadata["base"]["sha"], metadata["head"]["sha"]
        # Fetch the OIDs, not a moving pull ref. All writes go to the run's own repo.
        for oid in (base_oid, head_oid):
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
                raise ValueError("GitHub returned an invalid commit OID")
            try:
                revision(root, oid, **control)
                remote = str(root)
            except RuntimeError:
                remote = f"https://github.com/{repo}.git"
            scope_git(snapshot, "fetch", "--quiet", "--no-tags", remote, oid)
    else:
        head_oid = revision(root, head_ref or "HEAD", **control)
        base_oid = revision(root, base_ref, **control) if base_ref else default_base(root, **control)
        scope_git(snapshot, "fetch", "--quiet", "--no-tags", str(root), base_oid, head_oid)
    scope_git(snapshot, "checkout", "--quiet", "--detach", head_oid)
    source_head_oid = head_oid
    if include_worktree:
        patch = scope_git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
        if patch:
            scope_git(snapshot, "apply", "--binary", "-", stdin=patch)
        for name in scope_git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
            if not name:
                continue
            source = root / name
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination, follow_symlinks=False)
        scope_git(snapshot, "add", "-A")
        tree = scope_git(snapshot, "write-tree").strip()
        if tree != scope_git(snapshot, "rev-parse", head_oid + "^{tree}").strip():
            head_oid = scope_git(
                snapshot,
                "-c",
                "user.name=FlyDSL review",
                "-c",
                "user.email=review@localhost",
                "commit-tree",
                tree,
                "-p",
                head_oid,
                stdin="Pinned working tree for code review\n",
            ).strip()
            scope_git(snapshot, "checkout", "--quiet", "--detach", head_oid)
    merge_base = scope_git(snapshot, "merge-base", base_oid, head_oid).strip()
    diff_base = merge_base if three_dot else base_oid
    diff_args = ["diff", "--binary", "--no-ext-diff", "--no-renames", diff_base, head_oid, "--", *paths]
    diff = scope_git(snapshot, *diff_args)
    (run_dir / "diff.patch").write_text(diff, encoding="utf-8")
    files = scope_git(snapshot, "diff", "--name-only", "--no-renames", "-z", diff_base, head_oid, "--", *paths)
    return {
        "repo": repo,
        "pr": pr,
        "base_oid": base_oid,
        "merge_base_oid": merge_base,
        "diff_base_oid": diff_base,
        "head_oid": head_oid,
        "source_head_oid": source_head_oid,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "diff_command": shlex.join(["git", *diff_args]),
        "paths": paths,
        "files": [f for f in files.split("\0") if f],
        "instructions": instructions,
    }


def check_snapshot(run_dir: Path, scope: dict, **control) -> None:
    snapshot = run_dir / "repo"
    if revision(snapshot, "HEAD", **control) != scope["head_oid"]:
        raise ValueError("reviewed checkout moved from the pinned head")
    if git(snapshot, "status", "--porcelain", "--untracked-files=all", **control).strip():
        raise ValueError("an agent or another process modified the reviewed checkout")
    diff = git(
        snapshot,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-renames",
        scope["diff_base_oid"],
        scope["head_oid"],
        "--",
        *scope["paths"],
        **control,
    )
    if hashlib.sha256(diff.encode()).hexdigest() != scope["diff_sha256"]:
        raise ValueError("pinned diff hash changed")
    if hashlib.sha256((run_dir / "diff.patch").read_bytes()).hexdigest() != scope["diff_sha256"]:
        raise ValueError("saved diff.patch does not match the pinned diff")


def section(text: str, title: str) -> str:
    marker = "## " + title + "\n"
    if marker not in text:
        raise ValueError(f"missing skill section: {title}")
    return text.split(marker, 1)[1].split("\n## ", 1)[0].strip()


def output_schema(limit: int | None) -> dict:
    properties = {
        "status": {"enum": ["COMPLETE", "INCOMPLETE"]},
        "limitations": {"type": "array", "items": {"type": "string"}},
    }
    if limit is not None:
        fields = {
            "file": {"type": "string"},
            "line": {"type": ["integer", "null"], "minimum": 1},
            "summary": {"type": "string"},
            "mechanism": {"type": "string"},
            "severity": {"enum": list(SEVERITIES)},
            "failure_scenario": {"type": "string"},
        }
        properties["candidates"] = {
            "type": "array",
            "maxItems": limit,
            "items": {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False},
        }
    else:
        properties.update(verdict={"enum": list(VERDICTS)}, evidence={"type": "string"})
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def stop_process(process: subprocess.Popen) -> None:
    """Terminate the whole agent process group, including a running tool."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def cli_agent(
    task: dict, config: dict, snapshot: Path, logs: Path, deadline: float, cancelled: threading.Event
) -> dict:
    started = time.monotonic()
    argv = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--json-schema",
        canonical(task["schema"]),
        "--no-session-persistence",
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Read,Grep,Glob,Bash",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--allowedTools",
        "Read",
        "Grep",
        "Glob",
        "Bash(git diff *)",
        "Bash(git show *)",
        "Bash(git status *)",
        "Bash(python3 -c *)",
    ]
    if config["model"]:
        argv += ["--model", config["model"]]
    if config["effort"]:
        argv += ["--effort", config["effort"]]
    stdout_path, stderr_path = logs.with_suffix(".stdout.json"), logs.with_suffix(".stderr.txt")
    attempt = {"status": "INCOMPLETE", "usage": {}, "stdout": str(stdout_path), "stderr": str(stderr_path)}
    deadline = min(deadline, started + config["agent_timeout"])
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                argv,
                cwd=snapshot,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            try:
                first = True
                while True:
                    if cancelled.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("cancelled" if cancelled.is_set() else "agent/phase deadline exceeded")
                    try:
                        process.communicate(input=task["prompt"] if first else None, timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        first = False
            finally:
                # Also reap tool descendants if the CLI exited without waiting for them.
                stop_process(process)
        envelope = json.loads(stdout_path.read_text())
        # Verbose CLI settings emit a transcript array instead of one result.
        # The last result owns the verdict and usage, including a failed result.
        if isinstance(envelope, list):
            envelope = next(
                (
                    record
                    for record in reversed(envelope)
                    if isinstance(record, dict) and record.get("type") == "result"
                ),
                None,
            )
        if not isinstance(envelope, dict) or envelope.get("type") != "result":
            raise ValueError("CLI returned no terminal result record")
        attempt["usage"] = {
            "total_cost_usd": envelope.get("total_cost_usd"),
            "tokens": envelope.get("usage", {}),
            "model_usage": envelope.get("modelUsage", {}),
            "duration_api_ms": envelope.get("duration_api_ms"),
            "num_turns": envelope.get("num_turns"),
        }
        attempt["session_id"] = envelope.get("session_id")
        attempt["permission_denials"] = envelope.get("permission_denials", [])
        if process.returncode or envelope.get("is_error") or envelope.get("subtype") != "success":
            raise ValueError(f"CLI failed: exit={process.returncode}, subtype={envelope.get('subtype')}")
        if attempt["permission_denials"]:
            raise ValueError("agent encountered permission denials; see the saved CLI result")
        attempt["output"] = validate_output(
            envelope.get("structured_output"),
            candidate_limit=task["limit"],
            snapshot=str(snapshot),
        )
        attempt["status"] = "COMPLETE"
    except (OSError, ValueError, TimeoutError) as exc:
        attempt["error"] = str(exc)
    attempt["wall_time_seconds"] = time.monotonic() - started
    return attempt


class ReviewRun:
    def __init__(self, run_dir: Path, state: dict, backend=cli_agent):
        self.run_dir, self.state, self.backend = run_dir, state, backend
        self.snapshot = run_dir / "repo"
        self.config = state["config"]
        self.cancelled = threading.Event()
        self.started = time.monotonic()
        self.previous_elapsed = state.get("elapsed_seconds", 0)
        self.skill = SKILL.read_text()
        (run_dir / "agents").mkdir(exist_ok=True)

    def save(self):
        self.state["elapsed_seconds"] = self.previous_elapsed + time.monotonic() - self.started
        atomic_json(self.run_dir / "state.json", self.state)

    def context(self) -> str:
        scope = self.state["scope"]
        return (
            "Perform a read-only FlyDSL code review in this pinned checkout. Do not edit repository files, "
            "post comments, spawn agents, inspect later history, or access the repository network. "
            "Run small arithmetic probes with python3 -c when needed.\n"
            f"Reviewed base: {scope['base_oid']}\nReviewed head: {scope['head_oid']}\n"
            f"Diff: {scope['diff_command']}\nDiff SHA256: {scope['diff_sha256']}\n"
            "Read source and policy from this checkout (git show <reviewed-head>:<path> also works), "
            "never from a moving branch or the caller's workspace. Read enclosing functions for touched hunks.\n"
            f"Changed files: {canonical(scope['files'])}\nUser scope/instructions: {scope['instructions']}\n\n"
            + section(self.skill, "Reusing existing skills")
            + "\n\n"
            "The supplied skill excerpts describe the review method. Apply repository rules only where "
            "they exist and apply at the reviewed revision; do not impose later migrations on old code.\n"
            "Resolve relative links in these excerpts from .claude/skills/flydsl-code-review/SKILL.md; "
            "the referenced technical skills live under .claude/skills/.\n"
            "Return status COMPLETE with limitations [] only if you finished the assigned task. "
            "A tool failure, permission denial, missing required evidence, or unresolved stage means "
            "status INCOMPLETE with explicit limitations, never a successful empty list. "
            "An uncertain bug trigger is PLAUSIBLE; that alone is not an execution failure.\n\n"
        )

    def task(self, label: str, prompt: str, limit: int | None = None) -> dict:
        return {"label": label, "prompt": self.context() + prompt, "limit": limit, "schema": output_schema(limit)}

    def preflight(self) -> bool:
        """Run trusted scanners over the pinned data; their matches are only leads."""
        stages = self.state["stages"]
        scope = self.state["scope"]
        deadline = time.monotonic() + self.config["phase_timeout"]
        for label, script in PREFLIGHTS:
            fingerprint = digest(
                {"head": scope["head_oid"], "diff": scope["diff_sha256"], "script": (SCRIPTS / script).read_text()}
            )
            stage = stages.setdefault(label, {"runs": []})
            if stage.get("status") == "COMPLETE":
                validate_preflight(stage.get("output"))
                if stage.get("input_sha256") != fingerprint:
                    raise ValueError(f"cached input changed for {label}; start a new run")
                continue
            for prior in stage["runs"]:
                if prior["status"] == "RUNNING":
                    prior.update(status="INCOMPLETE", error="parent stopped before recording the scanner result")
            stage.update(status="RUNNING", output=None, input_sha256=fingerprint)
            stage.pop("error", None)
            record = {"status": "RUNNING"}
            stage["runs"].append(record)
            self.save()
            argv = [sys.executable, str(SCRIPTS / script), "--diff", str(self.run_dir / "diff.patch")]
            if label == "preflight:test-doc":
                argv += ["--head", str(self.snapshot)]
            print(f"{label}: scanning pinned diff", file=sys.stderr, flush=True)
            started = time.monotonic()
            try:
                result = command_result(*argv, cwd=self.snapshot, deadline=deadline, cancelled=self.cancelled)
                record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode not in (0, 1):
                    raise ValueError(f"scanner exited {result.returncode}: {result.stderr.strip()}")
                record["status"] = "COMPLETE"
                stage["output"] = {key: record[key] for key in ("exit_code", "stdout", "stderr")}
            except (OSError, ValueError, TimeoutError) as exc:
                record.update(status="INCOMPLETE", error=str(exc))
                stage["error"] = str(exc)
            record["wall_time_seconds"] = time.monotonic() - started
            stage["status"] = record["status"]
            self.save()
        return all(stage_output(stages, label) is not None for label, _ in PREFLIGHTS)

    def preflight_context(self, angle: str) -> str:
        output = stage_output(self.state["stages"], "preflight:" + angle)
        if output is None:
            return ""
        return (
            "\nDeterministic preflight observations (unverified leads, not findings):\n"
            + canonical(output)
            + "\n"
            + section(self.skill, "Deterministic preflight")
            + "\n"
        )

    def phase(self, name: str, tasks: list[dict]) -> bool:
        print(f"{name}: {len(tasks)} stage(s)", file=sys.stderr, flush=True)
        stages = self.state["stages"]
        pending = []
        for task in tasks:
            fingerprint = digest({"prompt": task["prompt"], "schema": task["schema"]})
            stage = stages.setdefault(task["label"], {"attempts": []})
            if stage.get("status") == "COMPLETE":
                if stage.get("input_sha256") != fingerprint:
                    raise ValueError(f"cached input changed for {task['label']}; start a new run")
                continue
            for attempt in stage["attempts"]:
                if attempt.get("status") == "RUNNING":
                    attempt.update(status="INCOMPLETE", error="parent stopped before recording this attempt's result")
            stage["input_sha256"] = fingerprint
            pending.append(task)
        deadline = time.monotonic() + self.config["phase_timeout"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["concurrency"]) as pool:
            active = {}
            while pending or active:
                expired = self.cancelled.is_set() or time.monotonic() >= deadline
                while pending and len(active) < self.config["concurrency"] and not expired:
                    task = pending.pop(0)
                    stage = stages[task["label"]]
                    stage.update(status="RUNNING", output=None)
                    stage.pop("error", None)
                    attempt_number = len(stage["attempts"]) + 1
                    logs = self.run_dir / "agents" / f"{task['label'].replace(':', '-')}-{attempt_number}"
                    logs.with_suffix(".prompt.txt").write_text(task["prompt"])
                    stage["attempts"].append(
                        {
                            "status": "RUNNING",
                            "usage": {},
                            "started_at": time.time(),
                            "stdout": str(logs.with_suffix(".stdout.json")),
                            "stderr": str(logs.with_suffix(".stderr.txt")),
                        }
                    )
                    self.save()  # A killed parent leaves an explicit RUNNING stage to retry.
                    future = pool.submit(self.backend, task, self.config, self.snapshot, logs, deadline, self.cancelled)
                    active[future] = task
                if expired:
                    for task in pending:
                        stages[task["label"]].update(
                            status="INCOMPLETE", error="cancelled or phase deadline before start"
                        )
                    pending.clear()
                if active:
                    done, _ = concurrent.futures.wait(
                        active, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        task = active.pop(future)
                        stage = stages[task["label"]]
                        try:
                            attempt = future.result()
                            if attempt.get("status") == "COMPLETE":
                                validate_output(
                                    attempt.get("output"), candidate_limit=task["limit"], snapshot=str(self.snapshot)
                                )
                        except Exception as exc:
                            attempt = {"status": "INCOMPLETE", "error": str(exc), "usage": {}}
                        stage["attempts"][-1].update(attempt)
                        stage["status"] = attempt["status"]
                        stage["output"] = attempt.get("output") if attempt["status"] == "COMPLETE" else None
                        if attempt["status"] != "COMPLETE":
                            stage["error"] = attempt.get("error", "agent returned no result")
                        self.save()
        return all(stage_output(stages, t["label"]) is not None for t in tasks)

    def verify(self, candidates: list[dict], phase_prefix: str = "") -> bool:
        ladder = section(self.skill, "Step 3 — Verify every candidate")
        tasks = []
        for c in candidates:
            observations = [self.state["stages"][s["stage"]]["output"]["candidates"][s["index"]] for s in c["sources"]]
            tasks.append(
                self.task(
                    "verify:" + c["id"],
                    "Independently verify this candidate.\n"
                    + canonical(c)
                    + "\nOriginal observations:\n"
                    + canonical(observations)
                    + "\n\n"
                    + ladder,
                )
            )
        if not self.phase(phase_prefix + "Verify", tasks):
            return False
        challenges = []
        for c in candidates:
            verdict = stage_output(self.state["stages"], "verify:" + c["id"])
            if verdict["verdict"] == "CONFIRMED":
                challenges.append(
                    self.task(
                        "challenge:" + c["id"],
                        "Challenge this CONFIRMED finding. Try to refute it. Independently execute its arithmetic "
                        "and trace the defect to an observable output, checking downstream masks and bounds. "
                        "Keep CONFIRMED only if both checks succeed; otherwise return PLAUSIBLE or REFUTED with evidence.\n"
                        + canonical(c)
                        + "\nPrior verifier:\n"
                        + canonical(verdict)
                        + "\n\n"
                        + ladder,
                    )
                )
        return self.phase(phase_prefix + "Challenge", challenges)

    def review(self) -> None:
        stages = self.state["stages"]
        if not self.state.get("scope"):
            scope = pin_scope(Path(self.state["source_root"]), self.run_dir, self.config, self.cancelled)
            self.state["scope"] = scope
            stages["scope"] = {"status": "COMPLETE", "output": scope}
            self.save()
        check_snapshot(
            self.run_dir,
            self.state["scope"],
            deadline=time.monotonic() + self.config["phase_timeout"],
            cancelled=self.cancelled,
        )
        if self.state["scope"]["files"]:
            if not self.preflight():
                return
            finders = [
                self.task(
                    "find:" + label,
                    "Review only this angle:\n"
                    + section(self.skill, title)
                    + self.preflight_context(label)
                    + f"\nReturn up to {PER_ANGLE} candidates with a specific mechanism/root cause, "
                    "severity (P0 critical, P1 high, P2 normal, P3 low), exact file/line and failure scenario. "
                    "Pass every candidate with a nameable failure scenario to independent verification. "
                    "For conventions, describe the concrete CI or maintenance cost. Do not invent crashes.",
                    PER_ANGLE,
                )
                for label, _, title in ANGLES
            ]
            if not self.phase("Find", finders):
                return
            # No completion-order admission or verification budget: verify every finder candidate.
            candidates = collect_candidates({k: v for k, v in stages.items() if k != "sweep"})
            if not self.verify(candidates):
                return
            known = judged_candidates({k: v for k, v in stages.items() if k != "sweep"})
            sweep = self.task(
                "sweep",
                "Hunt only for correctness defects absent from the known candidates. Check removed guards, "
                "setup/teardown asymmetry, changed defaults, cross-layer interactions, and unchanged lines "
                "of touched functions. Do not re-confirm known candidates.\nKnown candidates:\n"
                + canonical(known)
                + f"\nReturn at most {SWEEP_MAX} new candidates.",
                SWEEP_MAX,
            )
            if not self.phase("Sweep", [sweep]):
                return
            known_ids = {c["id"] for c in candidates}
            fresh = [c for c in collect_candidates(stages) if c["id"] not in known_ids]
            if not self.verify(fresh, "Sweep "):
                return
        check_snapshot(
            self.run_dir,
            self.state["scope"],
            deadline=time.monotonic() + self.config["phase_timeout"],
            cancelled=self.cancelled,
        )
        # Synthesis is code, not a model: it cannot invent findings or change verdicts/evidence.
        stages["synthesize"] = {"status": "COMPLETE", "output": {"method": "deterministic ranking of verified IDs"}}

    def run(self) -> dict:
        self.state["stages"].pop("run", None)
        try:
            self.review()
            if self.cancelled.is_set():
                raise ValueError("review cancelled")
        except Exception as exc:
            self.state["stages"]["run"] = {"status": "INCOMPLETE", "error": str(exc)}
            self.state.setdefault("failure_history", []).append(str(exc))
        self.save()
        report = build_report(self.state)
        # One terminal artifact. Checkpoints and raw attempts are never presented as a completed result.
        atomic_json(self.run_dir / "result.json", report)
        return report


def implementation_hash() -> str:
    paths = [Path(__file__), SCRIPTS / "review_common.py", SKILL, *(SCRIPTS / script for _, script in PREFLIGHTS)]
    return digest([p.read_text() for p in paths])


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", default="")
    parser.add_argument("--pr", type=positive)
    parser.add_argument("--repo")
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", help="omit to use the CLI's configured model")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--concurrency", type=positive, help="concurrent agents (default: 3)")
    parser.add_argument("--agent-timeout", type=positive, help="seconds per agent (default: 600)")
    parser.add_argument(
        "--phase-timeout", type=positive, help="seconds per phase, including queued agents (default: 1800)"
    )
    parser.add_argument("--run-dir", type=Path, help="new empty directory; defaults to a temporary directory")
    parser.add_argument("--resume", type=Path, help="existing run directory; retries only incomplete stages")
    args = parser.parse_args()
    if args.resume:
        if any(
            (
                args.target,
                args.pr,
                args.repo,
                args.base,
                args.head,
                args.path,
                args.instructions,
                args.run_dir,
                args.model,
                args.effort,
                args.concurrency,
                args.agent_timeout,
                args.phase_timeout,
            )
        ):
            parser.error("--resume uses the saved scope, configuration and model; do not supply new ones")
        run_dir = args.resume.resolve()
    else:
        run_dir = args.run_dir.resolve() if args.run_dir else Path(tempfile.mkdtemp(prefix="flydsl-review-"))
        run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this run is active; cancel it before resuming")
        if args.resume:
            state = json.loads((run_dir / "state.json").read_text())
            if state["implementation_sha256"] != implementation_hash():
                parser.error("runner or skill changed since the checkpoint; start a new run")
        else:
            if any(p.name != ".lock" for p in run_dir.iterdir()):
                parser.error("--run-dir must be empty; use --resume for an existing run")
            root = Path(command("git", "rev-parse", "--show-toplevel", cwd=Path.cwd()).strip()).resolve()
            if run_dir == root or root in run_dir.parents:
                parser.error("--run-dir must be outside the source checkout to avoid reviewing its own artifacts")
            state = {
                "schema_version": SCHEMA_VERSION,
                "run_id": str(uuid.uuid4()),
                "source_root": str(root),
                "implementation_sha256": implementation_hash(),
                "stages": {},
                "scope": None,
                "config": {
                    "target": args.target,
                    "pr": args.pr,
                    "repo": args.repo,
                    "base": args.base,
                    "head": args.head,
                    "paths": args.path,
                    "instructions": args.instructions,
                    "model": args.model,
                    "effort": args.effort,
                    "concurrency": args.concurrency or 3,
                    "agent_timeout": args.agent_timeout or 600,
                    "phase_timeout": args.phase_timeout or 1800,
                },
            }
        print(f"Run {state['run_id']}: {run_dir}\nResult: {run_dir / 'result.json'}", file=sys.stderr, flush=True)
        runner = ReviewRun(run_dir, state)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: runner.cancelled.set())
        runner.save()
        result = runner.run()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
