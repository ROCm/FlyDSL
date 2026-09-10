# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Review pipeline regressions; no model calls, GitHub writes or GPU dependencies."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.l0_backend_agnostic
SCRIPTS = Path(__file__).resolve().parents[2] / ".claude/skills/flydsl-code-review/scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_script("review_common")
runner = load_script("run_review")
publisher = load_script("post_review")


def candidate(line=1, mechanism="missing bounds guard", file="kernel.py", severity="P1"):
    return {
        "file": file,
        "line": line,
        "mechanism": mechanism,
        "severity": severity,
        "summary": mechanism,
        "failure_scenario": "tail row reaches an invalid stored output",
    }


def found(*candidates):
    return {"status": "COMPLETE", "limitations": [], "candidates": list(candidates)}


def verdict(value="CONFIRMED", evidence="Executed probe: row 9 stores the incorrect value 17."):
    return {"status": "COMPLETE", "limitations": [], "verdict": value, "evidence": evidence}


def done(output):
    return {"status": "COMPLETE", "output": output}


def configuration(**overrides):
    return {
        "target": "",
        "pr": None,
        "repo": None,
        "base": None,
        "head": None,
        "paths": [],
        "instructions": "",
        "model": None,
        "effort": None,
        "concurrency": 3,
        "agent_timeout": 1,
        "phase_timeout": 10,
        **overrides,
    }


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    runner.git(root, "init", "--quiet", "-b", "main")
    runner.git(root, "config", "user.name", "Test")
    runner.git(root, "config", "user.email", "test@example.invalid")
    (root / "kernel.py").write_text("x = 1\n")
    runner.git(root, "add", ".")
    runner.git(root, "commit", "--quiet", "-m", "base")
    base = runner.revision(root, "HEAD")
    runner.git(root, "checkout", "--quiet", "-b", "feature")
    (root / "kernel.py").write_text("x = 2\n")
    runner.git(root, "commit", "--quiet", "-am", "head")
    return root, base, runner.revision(root, "HEAD")


def new_run(tmp_path, source_repo, backend, **options):
    root, base, head = source_repo
    run_dir = tmp_path / "review"
    run_dir.mkdir()
    state = {
        "run_id": "test-run",
        "implementation_sha256": runner.implementation_hash(),
        "source_root": str(root),
        "scope": None,
        "stages": {},
        "config": configuration(base=base, head=head, **options),
    }
    return runner.ReviewRun(run_dir, state, backend)


class Backend:
    def __init__(self, failure=None, *, fail_once=False, many=False, downgrade=False, duplicate_sweep=False, delay=0):
        self.failure, self.fail_once, self.many = failure, fail_once, many
        self.downgrade, self.duplicate_sweep, self.delay = downgrade, duplicate_sweep, delay
        self.calls, self.active, self.peak = [], 0, 0
        self.lock = threading.Lock()

    def __call__(self, task, config, snapshot, logs, deadline, cancelled):
        label = task["label"]
        with self.lock:
            self.calls.append(label)
            self.active += 1
            self.peak = max(self.peak, self.active)
            fail = self.failure and label.startswith(self.failure)
            if fail and self.fail_once:
                self.failure = None
        try:
            if self.delay:
                time.sleep(self.delay)
            if fail or cancelled.is_set() or time.monotonic() >= deadline:
                return {"status": "INCOMPLETE", "error": "injected stage failure", "usage": {}}
            if label.startswith("find:"):
                if self.many:
                    output = found(*(candidate(i + 1, label + str(i)) for i in range(6)))
                else:
                    output = found(candidate()) if label == "find:trace-time" else found()
            elif label == "sweep":
                output = found(candidate()) if self.duplicate_sweep else found()
            else:
                output = verdict(
                    "PLAUSIBLE" if self.many or (self.downgrade and label.startswith("challenge:")) else "CONFIRMED"
                )
            return {
                "status": "COMPLETE",
                "output": output,
                "usage": {"total_cost_usd": 0.01, "tokens": {"output_tokens": 10}},
            }
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.parametrize("second_line", [99, 101])
def test_nearby_or_same_line_different_mechanisms_survive(second_line):
    stages = {
        "find:trace-time": done(found(candidate(99, "bounds"))),
        "find:addressing": done(found(candidate(second_line, "barrier"))),
    }
    candidates = common.collect_candidates(stages)
    assert len(candidates) == 2
    assert len({c["id"] for c in candidates}) == 2


def test_dedup_normalizes_location_and_mechanism_and_keeps_sources():
    stages = {
        "find:conventions": done(found(candidate(99, " Missing GUARD ", "./kernel.py"))),
        "find:trace-time": done(found(candidate(99, "missing guard", "sub/../kernel.py"))),
    }
    candidates = common.collect_candidates(stages)
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "correctness"
    assert len(candidates[0]["sources"]) == 2


def test_sweep_correctness_classification_is_not_hidden_by_earlier_convention():
    stages = {"find:conventions": done(found(candidate())), "sweep": done(found(candidate()))}
    candidates = common.collect_candidates(stages)
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "correctness"
    assert len(candidates[0]["sources"]) == 2


@pytest.mark.parametrize("line", [1.5, True, 0, -1, "1"])
def test_line_is_integer_not_coerced(line):
    with pytest.raises(ValueError, match="positive integer"):
        common.validate_output(found(candidate(line)), candidate_limit=6)


@pytest.mark.parametrize("failure", ["find:addressing", "verify:", "challenge:", "sweep"])
def test_required_failure_never_returns_clean_review(tmp_path, source_repo, failure):
    review = new_run(tmp_path, source_repo, Backend(failure))
    report = review.run()
    assert report["status"] == "INCOMPLETE"
    assert report["stage_failures"]
    assert report["findings"] == report["risks"] == []
    assert "No findings survived" not in report["summary"]
    assert json.loads((review.run_dir / "result.json").read_text())["status"] == "INCOMPLETE"
    if failure == "challenge:":
        assert report["candidates"][0]["verdict"] is None
        assert report["unresolved_candidate_ids"] == [report["candidates"][0]["id"]]
    with pytest.raises(ValueError, match="INCOMPLETE"):
        publisher.publish(report, dry_run=False)


def test_all_54_candidates_verified_and_correctness_has_priority(tmp_path, source_repo):
    backend = Backend(many=True)
    report = new_run(tmp_path, source_repo, backend).run()
    assert report["status"] == "COMPLETE"
    assert sum(c.startswith("verify:") for c in backend.calls) == 54
    assert report["stats"]["verified"] == 54
    assert len(report["risks"]) == 12
    assert all(c["kind"] == "correctness" for c in report["risks"])
    assert backend.peak <= 3


def test_challenge_downgrade_and_evidence_survive_synthesis(tmp_path, source_repo):
    report = new_run(tmp_path, source_repo, Backend(downgrade=True)).run()
    assert report["status"] == "COMPLETE"
    assert not report["findings"]
    risk = report["risks"][0]
    assert risk["verification"]["verdict"] == "CONFIRMED"
    assert risk["challenge"]["verdict"] == "PLAUSIBLE"
    assert "Challenger:" in risk["evidence"]
    assert report["stats"]["challenge_downgraded"] == 1
    common.validate_report(report)
    risk["verdict"] = "CONFIRMED"
    with pytest.raises(ValueError, match="verified records"):
        common.validate_report(report)


def test_resume_retries_only_failed_stages_and_keeps_prior_usage(tmp_path, source_repo):
    backend = Backend("verify:", fail_once=True, duplicate_sweep=True)
    review = new_run(tmp_path, source_repo, backend)
    first = review.run()
    assert first["status"] == "INCOMPLETE"
    state = json.loads((review.run_dir / "state.json").read_text())
    resumed = runner.ReviewRun(review.run_dir, state, backend).run()
    assert resumed["status"] == "COMPLETE"
    assert all(backend.calls.count("find:" + label) == 1 for label, _, _ in common.ANGLES)
    assert backend.calls.count("verify:" + resumed["candidates"][0]["id"]) == 2
    assert resumed["metrics"]["attempts_without_cost"] == 1
    assert resumed["metrics"]["cost_is_complete"] is False
    calls = len(backend.calls)
    state = json.loads((review.run_dir / "state.json").read_text())
    again = runner.ReviewRun(review.run_dir, state, backend).run()
    assert again["status"] == "COMPLETE"  # A duplicate sweep must not change cached prompt identity.
    assert len(backend.calls) == calls


def test_phase_deadline_and_concurrency_account_for_queued_tasks(tmp_path, source_repo):
    backend = Backend(delay=0.05)
    review = new_run(tmp_path, source_repo, backend, concurrency=2)
    review.state["scope"] = runner.pin_scope(source_repo[0], review.run_dir, review.config)
    review.state["stages"]["scope"] = done(review.state["scope"])
    review.config["phase_timeout"] = 0.02
    tasks = [review.task("find:" + a[0], "test finder", 6) for a in common.ANGLES]
    assert review.phase("Find", tasks) is False
    report = common.build_report(review.state)
    assert report["status"] == "INCOMPLETE"
    assert backend.peak == 2
    assert len(backend.calls) == 2
    assert all(report["stages"]["find:" + a[0]]["status"] == "INCOMPLETE" for a in common.ANGLES)


def test_interrupted_attempt_keeps_its_log_and_unknown_cost_on_resume(tmp_path, source_repo):
    review = new_run(tmp_path, source_repo, Backend("verify:"))
    report = review.run()
    label = "verify:" + report["candidates"][0]["id"]
    state = json.loads((review.run_dir / "state.json").read_text())
    stage = state["stages"][label]
    stage["status"] = stage["attempts"][0]["status"] = "RUNNING"
    previous_log = stage["attempts"][0]["stdout"]
    resumed = runner.ReviewRun(review.run_dir, state, Backend()).run()
    assert resumed["status"] == "COMPLETE"
    attempts = resumed["stages"][label]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["status"] == "INCOMPLETE"
    assert attempts[0]["stdout"] == previous_log != attempts[1]["stdout"]
    assert resumed["metrics"]["attempts_without_cost"] == 1


def test_pinned_scope_survives_source_push(tmp_path, source_repo):
    root, base, head = source_repo
    run_dir = tmp_path / "pin"
    run_dir.mkdir()
    scope = runner.pin_scope(root, run_dir, configuration(base=base, head=head))
    (root / "kernel.py").write_text("x = 3\n")
    runner.git(root, "commit", "--quiet", "-am", "later push")
    runner.check_snapshot(run_dir, scope)
    assert scope["head_oid"] == head != runner.revision(root, "HEAD")
    assert (run_dir / "repo/kernel.py").read_text() == "x = 2\n"
    assert head in scope["diff_command"] and base in scope["diff_command"]


def test_working_tree_gets_own_commit_without_mutating_source(tmp_path, source_repo):
    root, _, head = source_repo
    (root / "kernel.py").write_text("x = 4\n")
    (root / "untracked.py").write_text("y = 5\n")
    before = runner.git(root, "status", "--porcelain")
    run_dir = tmp_path / "dirty"
    run_dir.mkdir()
    scope = runner.pin_scope(root, run_dir, configuration())
    assert scope["source_head_oid"] == head != scope["head_oid"]
    assert runner.git(root, "status", "--porcelain") == before
    assert runner.revision(root, "HEAD") == head
    assert (run_dir / "repo/untracked.py").read_text() == "y = 5\n"
    runner.check_snapshot(run_dir, scope)


def test_modified_snapshot_is_incomplete(tmp_path, source_repo):
    review = new_run(tmp_path, source_repo, Backend())
    assert review.run()["status"] == "COMPLETE"
    (review.snapshot / "kernel.py").write_text("tampered\n")
    state = json.loads((review.run_dir / "state.json").read_text())
    result = runner.ReviewRun(review.run_dir, state, Backend()).run()
    assert result["status"] == "INCOMPLETE"
    assert "modified" in result["stages"]["run"]["error"]


def test_scope_commands_have_a_cancellable_deadline(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="phase deadline"):
        runner.command(sys.executable, "-c", "import time; time.sleep(30)", cwd=tmp_path, deadline=started + 0.1)
    assert time.monotonic() - started < 3


def test_cancellation_before_scope_returns_incomplete(tmp_path, source_repo):
    backend = Backend()
    review = new_run(tmp_path, source_repo, backend)
    review.cancelled.set()
    report = review.run()
    assert report["status"] == "INCOMPLETE"
    assert "cancelled" in report["stages"]["run"]["error"]
    assert backend.calls == []


def install_fake_cli(tmp_path, monkeypatch, body):
    binary = tmp_path / "claude"
    binary.write_text("#!" + sys.executable + "\n" + body)
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


@pytest.mark.parametrize("mode", ["success", "no_footer", "denied", "null_output"])
def test_cli_requires_success_footer_and_no_permission_denials(tmp_path, monkeypatch, mode):
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "structured_output": found(),
        "total_cost_usd": 0.25,
        "usage": {"output_tokens": 12},
        "permission_denials": [],
    }
    if mode == "denied":
        envelope["permission_denials"] = [{"tool_name": "Bash"}]
    if mode == "null_output":
        envelope["structured_output"] = None
    install_fake_cli(
        tmp_path, monkeypatch, "print(" + repr("" if mode == "no_footer" else json.dumps(envelope)) + ")\n"
    )
    task = {"prompt": "test", "schema": runner.output_schema(6), "limit": 6}
    attempt = runner.cli_agent(
        task, configuration(), tmp_path, tmp_path / "attempt", time.monotonic() + 3, threading.Event()
    )
    assert attempt["status"] == ("COMPLETE" if mode == "success" else "INCOMPLETE")
    if mode != "no_footer":
        assert attempt["usage"]["total_cost_usd"] == 0.25


def test_timeout_cancels_agent_and_its_tool_process(tmp_path, monkeypatch):
    child_pid = tmp_path / "child.pid"
    install_fake_cli(
        tmp_path,
        monkeypatch,
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n",
    )
    task = {"prompt": "test", "schema": runner.output_schema(6), "limit": 6}
    started = time.monotonic()
    attempt = runner.cli_agent(
        task, configuration(agent_timeout=0.4), tmp_path, tmp_path / "timeout", started + 5, threading.Event()
    )
    assert time.monotonic() - started < 3
    assert attempt["status"] == "INCOMPLETE"
    assert "deadline" in attempt["error"]
    assert child_pid.exists()
    stat = Path("/proc") / child_pid.read_text() / "stat"
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_command_line_entry_persists_one_result_and_resumes(tmp_path, source_repo, monkeypatch):
    root, base, head = source_repo
    record = candidate()
    install_fake_cli(
        tmp_path,
        monkeypatch,
        "import json, sys\n"
        "schema = json.loads(sys.argv[sys.argv.index('--json-schema') + 1])\n"
        "sys.stdin.read()\n"
        f"output = {found(record)!r} if 'candidates' in schema['properties'] else {verdict()!r}\n"
        "print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, "
        "'structured_output': output, 'total_cost_usd': 0.01, 'usage': {'output_tokens': 10}}))\n",
    )
    run_dir = tmp_path / "cli-run"
    entry = [sys.executable, str(SCRIPTS / "run_review.py")]
    process = subprocess.run(
        [*entry, "--base", base, "--head", head, "--run-dir", str(run_dir)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report == json.loads((run_dir / "result.json").read_text())
    assert report["status"] == "COMPLETE"
    assert report["metrics"]["agent_attempts"] == 12
    assert report["stats"]["verified"] == report["stats"]["challenged"] == 1
    resumed = subprocess.run([*entry, "--resume", str(run_dir)], cwd=root, capture_output=True, text=True, timeout=10)
    assert resumed.returncode == 0, resumed.stderr
    second = json.loads(resumed.stdout)
    assert second["run_id"] == report["run_id"]
    assert second["reported_ids"] == report["reported_ids"]
    assert second["metrics"]["agent_attempts"] == 12


@pytest.fixture
def complete_report():
    scope = {
        "repo": "ROCm/FlyDSL",
        "pr": 1106,
        "base_oid": "a" * 40,
        "merge_base_oid": "a" * 40,
        "diff_base_oid": "a" * 40,
        "head_oid": "b" * 40,
        "diff_sha256": "c" * 64,
        "files": ["kernel.py"],
    }
    stages = {"scope": done(scope), "sweep": done(found()), "synthesize": done({})}
    stages.update({"find:" + label: done(found()) for label, _, _ in common.ANGLES})
    stages["find:trace-time"] = done(
        found(candidate(10), candidate(90, "second defect"), candidate(11, "uncertain race"))
    )
    for c in common.collect_candidates(stages):
        if c["mechanism"] == "uncertain race":
            stages["verify:" + c["id"]] = done(verdict("PLAUSIBLE"))
        else:
            stages["verify:" + c["id"]] = done(verdict())
            stages["challenge:" + c["id"]] = done(verdict())
    state = {"run_id": "test-run", "implementation_sha256": "d" * 64, "config": {}, "scope": scope, "stages": stages}
    return common.build_report(state)


class GitHub:
    def __init__(self, report, *, advance_at=None, lost_response=False):
        self.scope = report["scope"]
        self.advance_at, self.lost_response = advance_at, lost_response
        self.head_reads, self.posts, self.reviews = 0, [], []

    def __call__(self, *args, stdin=None):
        endpoint = next(a for a in args if a.startswith("repos/"))
        if "POST" in args:
            assert endpoint.endswith("/reviews")
            payload = json.loads(stdin)
            self.posts.append(payload)
            self.reviews.append(
                {"id": 1, "body": payload["body"], "commit_id": payload["commit_id"], "state": "COMMENTED"}
            )
            if self.lost_response:
                raise RuntimeError("response lost after server committed the review")
            return '{"id":1}'
        if endpoint.endswith("/reviews"):
            return json.dumps([self.reviews])
        if endpoint.endswith("/files"):
            return json.dumps([[{"filename": "kernel.py", "patch": "@@ -10,2 +10,2 @@\n-old\n+new\n context"}], []])
        self.head_reads += 1
        head = "e" * 40 if self.head_reads == self.advance_at else self.scope["head_oid"]
        return json.dumps({"state": "open", "head": {"sha": head}, "base": {"sha": self.scope["base_oid"]}})


def test_single_review_preserves_deferred_evidence_and_separates_risks(monkeypatch, complete_report):
    api = GitHub(complete_report)
    monkeypatch.setattr(publisher, "gh", api)
    assert publisher.publish(complete_report, dry_run=False) == 0
    assert len(api.posts) == 1
    payload = api.posts[0]
    assert payload["commit_id"] == complete_report["scope"]["head_oid"]
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["line"] == 10
    assert "CONFIRMED" in payload["body"] and "PLAUSIBLE" in payload["body"]
    assert "not merge blockers" in payload["body"]
    assert "tail row reaches" in payload["body"] and "Executed probe" in payload["body"]
    assert "known_cost_usd" in payload["body"] and complete_report["run_id"] in payload["body"]
    assert publisher.publish(complete_report, dry_run=False) == 0
    assert len(api.posts) == 1


@pytest.mark.parametrize("advance_at", [1, 2])
def test_post_rejects_a_changed_head_before_or_during_routing(monkeypatch, complete_report, advance_at):
    api = GitHub(complete_report, advance_at=advance_at)
    monkeypatch.setattr(publisher, "gh", api)
    with pytest.raises(ValueError, match="base/head changed"):
        publisher.publish(complete_report, dry_run=False)
    assert api.posts == []


def test_lost_post_response_is_reconciled_without_reposting(monkeypatch, complete_report):
    api = GitHub(complete_report, lost_response=True)
    monkeypatch.setattr(publisher, "gh", api)
    assert publisher.publish(complete_report, dry_run=False) == 0
    assert publisher.publish(complete_report, dry_run=False) == 0
    assert len(api.posts) == 1


def test_dry_run_does_not_write(monkeypatch, complete_report, capsys):
    api = GitHub(complete_report)
    monkeypatch.setattr(publisher, "gh", api)
    assert publisher.publish(complete_report, dry_run=True) == 0
    assert api.posts == []
    assert json.loads(capsys.readouterr().out)["commit_id"] == complete_report["scope"]["head_oid"]


@pytest.mark.parametrize("field", ["verdict", "evidence", "kind", "id", "line"])
def test_publisher_rejects_changed_provenance(complete_report, field):
    report = copy.deepcopy(complete_report)
    report["findings"][0][field] = "tampered"
    with pytest.raises(ValueError, match="verified records"):
        common.validate_report(report)


def test_missing_stage_is_not_complete_even_with_empty_findings(complete_report):
    del complete_report["stages"]["sweep"]
    with pytest.raises(ValueError, match="required stages"):
        common.validate_report(complete_report)


def test_diff_hunks_exclude_deleted_and_outside_lines():
    patch = "@@ -44,3 +44,3 @@\n before\n-old\n+new\n after\n\\ No newline at end of file\n"
    assert publisher.commentable_lines(patch) == {44, 45, 46}
