# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Local review watcher regressions; no network, model, Docker, GitHub write, or GPU use."""

import copy
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l0_backend_agnostic
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/review_bot/watch.py"
SPEC = importlib.util.spec_from_file_location("flydsl_review_bot_watch", SCRIPT)
watch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watch
SPEC.loader.exec_module(watch)

ENGINE_SHA = "e" * 40
IMPLEMENTATION_SHA256 = "1" * 64


def config(tmp_path):
    engine = tmp_path / "engine-root"
    engine.mkdir()
    value = watch.Config(
        engine_root=engine,
        image="registry.example.invalid/flydsl-review-bot@sha256:" + "d" * 64,
        implementation_sha256=IMPLEMENTATION_SHA256,
        state_root=tmp_path / "state",
        retention_days=30,
    )
    value.validate()
    watch.ensure_private_root(value.state_root)
    return value


def pull(number=7, head="a" * 40, base="b" * 40, base_ref="main"):
    return watch.PullRequest(number, head, base, base_ref)


def api_pull(number=7, head="a" * 40, base="b" * 40):
    repository = {"id": watch.REPOSITORY_ID, "full_name": watch.REPOSITORY}
    return {
        "number": number,
        "state": "open",
        "draft": False,
        "user": {"id": watch.AUTHOR_ID, "login": watch.AUTHOR_LOGIN},
        "head": {"sha": head, "repo": copy.deepcopy(repository)},
        "base": {"sha": base, "ref": "main", "repo": copy.deepcopy(repository)},
    }


class FakeGitHub:
    def __init__(self, pulls):
        self.pulls = list(pulls)
        self.current = {}
        self.publisher_checks = 0
        self.current_reads = 0

    def open_eligible_pulls(self):
        self.publisher_checks += 1
        return list(self.pulls)

    def current_pull(self, number):
        self.current_reads += 1
        if number in self.current:
            return self.current[number]
        return next((item for item in self.pulls if item.number == number), None)

    def verify_publisher(self):
        self.publisher_checks += 1


class FakeSource:
    def __init__(self, engine_sha=ENGINE_SHA):
        self.engine_sha = engine_sha
        self.source_calls = []
        self.engine_calls = []

    def engine_identity(self):
        return self.engine_sha

    def prepare_source(self, destination, item):
        destination.mkdir(mode=0o700)
        self.source_calls.append(item)
        return watch.PreparedSource(destination, item.base_oid, 1, 128)

    def prepare_engine(self, destination, engine_sha):
        destination.mkdir(mode=0o700)
        self.engine_calls.append(engine_sha)
        return destination


class FakeDocker:
    def __init__(self, cfg, *, returncode=0, result_status="COMPLETE", timed_out=False):
        self.config = cfg
        self.returncode = returncode
        self.result_status = result_status
        self.timed_out = timed_out
        self.calls = []
        self.cleanup_calls = []

    def cleanup(self, key):
        self.cleanup_calls.append(key)

    def run(self, argv, name, artifact_root):
        self.calls.append((list(argv), name, artifact_root))
        manifest = json.loads((artifact_root / "input/scope-manifest.json").read_text())
        result = {
            "status": self.result_status,
            "implementation_sha256": self.config.implementation_sha256,
            "scope": manifest,
            "config": {
                "execution_profile": "untrusted-container",
                "model": "opus",
                "effort": "max",
            },
        }
        watch.atomic_json(artifact_root / "output/result.json", result)
        return watch.ContainerOutcome(self.returncode, self.timed_out, 1.0)


class FakePublisher:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def publish(self, argv, artifact_root):
        self.calls.append((list(argv), artifact_root))
        return self.returncode


def make_watcher(tmp_path, pulls, *, docker_options=None, github=None, source=None, publisher=None, now=None):
    cfg = config(tmp_path)
    state = watch.StateStore(cfg.state_root, now=now or (lambda: 1_800_000_000))
    github = github or FakeGitHub(pulls)
    source = source or FakeSource()
    docker = FakeDocker(cfg, **(docker_options or {}))
    publisher = publisher or FakePublisher()
    operator = watch.Watcher(
        cfg,
        state,
        github,
        source,
        docker,
        publisher,
        now=now or (lambda: 1_800_000_000),
    )
    return cfg, state, github, source, docker, publisher, operator


def claim_rows(state):
    return state.connection.execute(
        "SELECT pr_number, head_oid, engine_sha, status, reason FROM claims ORDER BY pr_number, head_oid"
    ).fetchall()


def test_file_lock_serializes_overlapping_scans(tmp_path):
    root = tmp_path / "state"
    watch.ensure_private_root(root)
    with watch.scan_lock(root) as first:
        assert first is True
        with watch.scan_lock(root) as second:
            assert second is False


def test_model_gateway_requires_both_named_variables(monkeypatch):
    for name in watch.MODEL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(watch.ConfigurationError, match="model gateway"):
        watch.require_model_environment()
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "synthetic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8882")
    watch.require_model_environment()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:7201")
    with pytest.raises(watch.ConfigurationError, match="endpoint"):
        watch.require_model_environment()


def test_first_scan_seeds_all_existing_heads_without_review(tmp_path):
    first, second = pull(7, "a" * 40), pull(9, "c" * 40)
    _, state, github, source, docker, publisher, operator = make_watcher(tmp_path, [first, second])
    try:
        summary = operator.run_once()
        assert summary.seeded == 2
        assert summary.claimed == 0
        assert [(row["pr_number"], row["status"]) for row in claim_rows(state)] == [
            (7, "SEEDED"),
            (9, "SEEDED"),
        ]
        assert source.source_calls == docker.calls == publisher.calls == []
        assert github.publisher_checks == 1

        duplicate = operator.run_once()
        assert duplicate.claimed == 0
        assert source.source_calls == docker.calls == publisher.calls == []
    finally:
        state.close()


def test_manual_canary_consumes_one_seed_with_dry_run_only(tmp_path):
    current = pull(head="a" * 40)
    cfg, state, _, _, docker, publisher, operator = make_watcher(tmp_path, [current])
    try:
        summary = operator.run_canary(current.number)
        assert summary.seeded == summary.claimed == summary.complete == 1
        assert len(docker.calls) == len(publisher.calls) == 1
        assert "--dry-run" in publisher.calls[0][0]
        with pytest.raises(watch.PolicyRejection, match="already_terminal"):
            operator.run_canary(current.number)
        operator.config = dataclasses.replace(cfg, publish_enabled=True)
        with pytest.raises(watch.ConfigurationError, match="publish_enabled=false"):
            operator.run_canary(current.number)
    finally:
        state.close()


def test_claim_runs_once_and_new_head_can_run(tmp_path):
    old = pull(head="a" * 40)
    cfg, state, github, source, docker, publisher, operator = make_watcher(tmp_path, [old])
    try:
        assert operator.run_once().seeded == 1
        newer = pull(head="c" * 40)
        github.pulls = [newer]
        first = operator.run_once()
        assert first.claimed == first.complete == 1
        assert len(docker.calls) == len(publisher.calls) == 1
        assert github.current_reads == 1
        assert github.publisher_checks == 3
        assert (
            state.connection.execute("SELECT status FROM claims WHERE head_oid = ?", (newer.head_oid,)).fetchone()[
                "status"
            ]
            == "COMPLETE"
        )

        again = operator.run_once()
        assert again.claimed == 0
        assert len(docker.calls) == len(publisher.calls) == 1

        newest = pull(head="d" * 40)
        github.pulls = [newest]
        assert operator.run_once().complete == 1
        assert len(docker.calls) == len(publisher.calls) == 2
        publisher_argv = publisher.calls[-1][0]
        assert publisher_argv[publisher_argv.index("--expected-head") + 1] == newest.head_oid
        assert publisher_argv[publisher_argv.index("--expected-implementation-sha256") + 1] == (
            cfg.implementation_sha256
        )
    finally:
        state.close()


def test_base_tip_change_does_not_retry_the_same_terminal_head(tmp_path):
    original = pull(head="a" * 40, base="b" * 40)
    _, state, github, _, docker, publisher, operator = make_watcher(tmp_path, [original])
    try:
        assert operator.run_once().seeded == 1
        advanced = pull(head=original.head_oid, base="c" * 40)
        github.pulls = [advanced]
        github.current[advanced.number] = advanced
        summary = operator.run_once()
        assert summary.claimed == summary.complete == 0
        assert docker.calls == publisher.calls == []
        rows = state.connection.execute(
            "SELECT base_oid, status FROM claims WHERE head_oid = ? ORDER BY base_oid",
            (original.head_oid,),
        ).fetchall()
        assert [(row["base_oid"], row["status"]) for row in rows] == [(original.base_oid, "SEEDED")]
    finally:
        state.close()


@pytest.mark.parametrize(
    "docker_options",
    [
        {"returncode": 17},
        {"result_status": "INCOMPLETE"},
        {"timed_out": True},
    ],
)
def test_incomplete_head_is_terminal_and_does_not_auto_retry(tmp_path, docker_options):
    current = pull(head="c" * 40)
    _, state, github, source, docker, publisher, operator = make_watcher(tmp_path, [], docker_options=docker_options)
    try:
        operator.run_once()
        github.pulls = [current]
        assert operator.run_once().incomplete == 1
        assert len(docker.calls) == 1
        assert publisher.calls == []
        assert operator.run_once().claimed == 0
        assert len(docker.calls) == 1

        github.pulls = [pull(head="d" * 40)]
        docker.returncode = 0
        docker.result_status = "COMPLETE"
        docker.timed_out = False
        assert operator.run_once().complete == 1
        assert len(docker.calls) == 2
    finally:
        state.close()


def test_startup_recovers_active_claim_as_incomplete_without_retry(tmp_path):
    current = pull(head="c" * 40)
    _, state, _, _, docker, publisher, operator = make_watcher(tmp_path, [current])
    try:
        state.seed([], ENGINE_SHA)
        key = state.claim(current, ENGINE_SHA)
        state.transition(key, "CLAIMED", "RUNNING")
        summary = operator.run_once()
        assert summary.recovered == 1
        assert summary.claimed == 0
        row = claim_rows(state)[0]
        assert row["status"] == "INCOMPLETE"
        assert row["reason"] == "interrupted"
        assert docker.calls == publisher.calls == []
        assert docker.cleanup_calls == [key]
    finally:
        state.close()


def test_stale_live_head_never_invokes_publisher(tmp_path):
    reviewed = pull(head="c" * 40)
    github = FakeGitHub([])
    cfg, state, github, _, docker, publisher, operator = make_watcher(tmp_path, [], github=github)
    try:
        operator.run_once()
        github.pulls = [reviewed]
        github.current[reviewed.number] = pull(head="d" * 40)
        summary = operator.run_once()
        assert summary.stale == 1
        assert len(docker.calls) == 1
        assert publisher.calls == []
        assert github.current_reads == 1
        assert github.publisher_checks == 2
        assert (
            state.connection.execute("SELECT status FROM claims WHERE head_oid = ?", (reviewed.head_oid,)).fetchone()[
                "status"
            ]
            == "STALE"
        )
        assert cfg.state_root.stat().st_mode & 0o777 == 0o700
    finally:
        state.close()


def test_stale_live_base_never_invokes_publisher(tmp_path):
    reviewed = pull(head="c" * 40, base="b" * 40)
    github = FakeGitHub([])
    _, state, github, _, docker, publisher, operator = make_watcher(tmp_path, [], github=github)
    try:
        operator.run_once()
        github.pulls = [reviewed]
        github.current[reviewed.number] = pull(head=reviewed.head_oid, base="d" * 40)
        assert operator.run_once().stale == 1
        assert len(docker.calls) == 1
        assert publisher.calls == []
    finally:
        state.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(state="closed"),
        lambda value: value.update(draft=True),
        lambda value: value["user"].update(id=1),
        lambda value: value["user"].update(login="renamed"),
        lambda value: value["head"]["repo"].update(id=1),
        lambda value: value["head"]["repo"].update(full_name="fork/FlyDSL"),
        lambda value: value["base"]["repo"].update(id=1),
        lambda value: value["head"].update(sha="not-an-oid"),
    ],
)
def test_immutable_identity_login_and_same_repo_gate(mutation):
    value = api_pull()
    mutation(value)
    assert watch.parse_eligible_pull(value) is None


def test_exact_eligible_gate_accepts_only_expected_pr():
    value = api_pull()
    assert watch.parse_eligible_pull(value) == pull(head="a" * 40)
    repository = {"id": watch.REPOSITORY_ID, "full_name": watch.REPOSITORY}
    assert watch.repository_identity(repository)
    assert not watch.repository_identity({**repository, "id": 1})


class ScriptedProcess:
    def __init__(self, responses):
        self.responses = responses
        self.argv = []

    def run(self, argv, **_):
        self.argv.append(list(argv))
        endpoint = argv[-1]
        return subprocess.CompletedProcess(argv, 0, self.responses[endpoint], "")


def test_gh_245_pagination_and_expected_publisher_identity(tmp_path):
    cfg = config(tmp_path)
    page = json.dumps([api_pull()])
    process = ScriptedProcess(
        {
            "repos/ROCm/FlyDSL": json.dumps({"id": watch.REPOSITORY_ID, "full_name": watch.REPOSITORY}),
            "user": json.dumps({"id": watch.PUBLISHER_ID, "login": watch.PUBLISHER_LOGIN}),
            "repos/ROCm/FlyDSL/pulls?state=open&per_page=100": page + "\n[]\n",
        }
    )
    github = watch.GitHubClient(cfg, process)
    assert github.open_eligible_pulls() == [pull(head="a" * 40)]
    assert all(argv[:4] == [cfg.gh_bin, "api", "--method", "GET"] for argv in process.argv)
    assert any("--paginate" in argv for argv in process.argv)
    assert all("--slurp" not in argv for argv in process.argv)

    process.responses["user"] = json.dumps({"id": 1, "login": watch.PUBLISHER_LOGIN})
    with pytest.raises(watch.WatcherError, match="publisher identity mismatch"):
        github.verify_publisher()


def option_values(argv, option):
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == option]


def test_container_argv_has_fixed_interface_mounts_limits_and_no_host_credentials(tmp_path):
    cfg = config(tmp_path)
    dataclasses.replace(cfg, image="sha256:" + "a" * 64).validate()
    item = pull(head="a" * 40)
    paths = {
        "engine_root": tmp_path / "run/engine",
        "source_root": tmp_path / "run/source",
        "input_root": tmp_path / "run/input",
        "output_root": tmp_path / "run/output",
    }
    argv = watch.build_docker_argv(cfg, item, ENGINE_SHA, **paths)
    assert all(isinstance(value, str) for value in argv)
    assert argv[:2] == [cfg.docker_bin, "run"]
    assert option_values(argv, "--log-driver") == ["none"]
    assert option_values(argv, "--runtime") == ["runc"]
    assert option_values(argv, "--network") == ["host"]
    assert "--add-host" not in argv
    assert "--read-only" in argv
    assert option_values(argv, "--cap-drop") == ["ALL"]
    assert option_values(argv, "--security-opt") == ["no-new-privileges=true"]
    assert option_values(argv, "--pids-limit") == [str(cfg.pids_limit)]
    assert option_values(argv, "--memory") == [cfg.memory_limit]
    assert option_values(argv, "--memory-swap") == [cfg.memory_limit]
    assert option_values(argv, "--cpus") == [cfg.cpu_limit]
    assert option_values(argv, "--stop-timeout") == [str(watch.CONTAINER_STOP_SECONDS)]
    assert option_values(argv, "--user") == [f"{os.getuid()}:{os.getgid()}"]
    mounts = option_values(argv, "--mount")
    assert len(mounts) == 4
    assert all("readonly" in value for value in mounts[:3])
    assert "dst=/review-run" in mounts[3] and "readonly" not in mounts[3]
    joined = "\0".join(argv)
    assert "/var/run/docker.sock" not in joined
    assert ".config/gh" not in joined
    assert "GH_TOKEN" not in joined
    assert "ANTHROPIC_AUTH_TOKEN" in option_values(argv, "--env")
    assert "ANTHROPIC_BASE_URL" in option_values(argv, "--env")
    assert "--device" not in argv and "--gpus" not in argv
    assert option_values(argv, "--env")[-3:] == [
        "ROCR_VISIBLE_DEVICES=-1",
        "HIP_VISIBLE_DEVICES=-1",
        "CUDA_VISIBLE_DEVICES=-1",
    ]
    image_index = argv.index(cfg.image)
    assert argv[image_index + 1 :] == [
        "--scope-manifest",
        "/review-input/scope-manifest.json",
        "--execution-profile",
        "untrusted-container",
        "--model",
        "opus",
        "--effort",
        "max",
        "--claude-path",
        "/usr/local/bin/claude",
        "--run-dir",
        "/review-run",
    ]
    labels = option_values(argv, "--label")
    assert f"com.amd.flydsl.pr-number={item.number}" in labels
    assert f"com.amd.flydsl.head-oid={item.head_oid}" in labels
    assert f"com.amd.flydsl.engine-sha={ENGINE_SHA}" in labels


def test_container_timeout_requests_graceful_stop_at_80_minutes(tmp_path):
    cfg = config(tmp_path)

    class ReviewProcess:
        pid = 999_999_991

        def __init__(self):
            self.returncode = None
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired(["docker", "run"], timeout)
            self.returncode = 143
            return self.returncode

        def poll(self):
            return self.returncode

    class FinishedProcess:
        pid = 999_999_992
        returncode = 0

        def wait(self, _timeout):
            return 0

        def poll(self):
            return 0

    class TimeoutProcessRunner:
        def __init__(self):
            self.review = ReviewProcess()
            self.calls = []

        def popen(self, argv, **_):
            self.calls.append(list(argv))
            return self.review if len(self.calls) == 1 else FinishedProcess()

    process = TimeoutProcessRunner()
    clock = iter((0.0, float(watch.GRACEFUL_REVIEW_SECONDS), float(watch.GRACEFUL_REVIEW_SECONDS + 1)))
    backend = watch.DockerBackend(cfg, process, monotonic=lambda: next(clock))
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outcome = backend.run([cfg.docker_bin, "run"], "fixed-name", artifact_root)
    assert outcome.timed_out is True
    assert process.review.waits == [
        watch.GRACEFUL_REVIEW_SECONDS,
        watch.CONTAINER_STOP_SECONDS,
    ]
    assert process.calls[1] == [
        cfg.docker_bin,
        "stop",
        "--time",
        str(watch.CONTAINER_STOP_SECONDS),
        "fixed-name",
    ]


def test_crash_cleanup_requires_all_container_identity_labels(tmp_path):
    cfg = config(tmp_path)
    key = watch.ClaimKey(watch.REPOSITORY_ID, 7, "a" * 40, ENGINE_SHA)
    name = "flydsl-review-7-" + "a" * 12 + "-" + ENGINE_SHA[:12]

    class CleanupProcess:
        def __init__(self, labels):
            self.labels = labels
            self.calls = []

        def run(self, argv, **_):
            self.calls.append(list(argv))
            if "ls" in argv:
                stdout = name + "\n"
            elif "inspect" in argv:
                stdout = json.dumps(self.labels)
            else:
                stdout = ""
            return subprocess.CompletedProcess(argv, 0, stdout, "")

    expected = watch.container_labels(key.pr_number, key.head_oid, key.engine_sha)
    process = CleanupProcess(expected)
    watch.DockerBackend(cfg, process).cleanup(key)
    assert process.calls[-1][-3:] == ["rm", "--force", name]

    forged = CleanupProcess({**expected, "com.amd.flydsl.head-oid": "f" * 40})
    with pytest.raises(watch.WatcherError, match="identity mismatch"):
        watch.DockerBackend(cfg, forged).cleanup(key)
    assert all("rm" not in argv for argv in forged.calls)


def test_publisher_argv_binds_head_implementation_and_publisher(tmp_path):
    cfg = config(tmp_path)
    item = pull(head="a" * 40)
    argv = watch.build_publisher_argv(
        cfg,
        engine_root=tmp_path / "trusted-engine",
        result_path=tmp_path / "result.json",
        pull=item,
    )
    assert argv[0] == cfg.python_bin
    assert argv[1].endswith("/.claude/skills/flydsl-code-review/scripts/post_review.py")
    assert option_values(argv, "--expected-head") == [item.head_oid]
    assert option_values(argv, "--expected-implementation-sha256") == [cfg.implementation_sha256]
    assert option_values(argv, "--expected-publisher-id") == [str(watch.PUBLISHER_ID)]
    assert option_values(argv, "--expected-repository-id") == [str(watch.REPOSITORY_ID)]
    assert option_values(argv, "--expected-author-id") == [str(watch.AUTHOR_ID)]
    assert option_values(argv, "--expected-author-login") == [watch.AUTHOR_LOGIN]
    assert option_values(argv, "--publish-severity") == ["P1"]
    assert "--dry-run" in argv
    live = dataclasses.replace(cfg, publish_enabled=True)
    assert "--dry-run" not in watch.build_publisher_argv(
        live,
        engine_root=tmp_path / "trusted-engine",
        result_path=tmp_path / "result.json",
        pull=item,
    )


class SyntheticPolicySource(watch.SourceManager):
    def __init__(
        self,
        *,
        paths=("kernel.py",),
        modes=None,
        binary=False,
        patch_bytes=128,
        tree_symlink=False,
    ):
        self.paths = list(paths)
        self.modes = modes or [("100644", "100644")] * len(self.paths)
        self.binary = binary
        self.patch_bytes = patch_bytes
        self.tree_symlink = tree_symlink

    def _bounded_diff_size(self, _source, _diff_base, _head_oid):
        if self.patch_bytes > watch.MAX_DIFF_BYTES:
            raise watch.PolicyRejection("diff_too_large")
        return self.patch_bytes

    def git(self, _root, *args, text=True, **_):
        if args[0] == "merge-base":
            stdout = "f" * 40 + "\n"
        elif args[0] == "diff" and "--raw" in args:
            records = []
            for index, (path, modes) in enumerate(zip(self.paths, self.modes)):
                old_mode, new_mode = modes
                header = f":{old_mode} {new_mode} {'a' * 40} {'b' * 40} M".encode()
                records.extend((header, os.fsencode(path)))
            stdout = b"\0".join(records) + (b"\0" if records else b"")
        elif args[0] == "diff" and "--numstat" in args:
            additions = b"-" if self.binary else b"1"
            records = [additions + b"\t" + additions + b"\t" + os.fsencode(path) for path in self.paths]
            stdout = b"\0".join(records) + (b"\0" if records else b"")
        elif args[0] == "ls-tree":
            mode = b"120000" if self.tree_symlink else b"100644"
            stdout = mode + b" blob " + b"a" * 40 + b"\tkernel.py\0"
        else:
            raise AssertionError(args)
        if text and isinstance(stdout, bytes):
            stdout = stdout.decode()
        return subprocess.CompletedProcess(args, 0, stdout, "" if text else b"")


@pytest.mark.parametrize(
    "path",
    [
        ".github",
        ".github/workflows/review.yml",
        ".claude/skills/override.md",
        "CLAUDE.md",
        "tools/review_bot",
        "tools/review_bot/watch.py",
    ],
)
def test_pilot_control_paths_are_rejected(path):
    source = SyntheticPolicySource(paths=[path])
    with pytest.raises(watch.PolicyRejection, match="control_path_change"):
        source._enforce_policy(Path("/synthetic"), "a" * 40, "b" * 40)


@pytest.mark.parametrize("path", ["../escape.py", "line\nbreak.py", "control\u0085.py"])
def test_unsafe_repository_paths_are_rejected(path):
    source = SyntheticPolicySource(paths=[path])
    with pytest.raises(watch.PolicyRejection, match="unsafe_path"):
        source._enforce_policy(Path("/synthetic"), "a" * 40, "b" * 40)


def test_policy_limits_are_inclusive_and_reject_excess():
    accepted = SyntheticPolicySource(
        paths=[f"file-{index}.py" for index in range(watch.MAX_FILES)],
        patch_bytes=watch.MAX_DIFF_BYTES,
    )
    _, count, size = accepted._enforce_policy(Path("/synthetic"), "a" * 40, "b" * 40)
    assert (count, size) == (watch.MAX_FILES, watch.MAX_DIFF_BYTES)

    too_many = SyntheticPolicySource(paths=[f"file-{index}.py" for index in range(watch.MAX_FILES + 1)])
    with pytest.raises(watch.PolicyRejection, match="too_many_files"):
        too_many._enforce_policy(Path("/synthetic"), "a" * 40, "b" * 40)

    too_large = SyntheticPolicySource(patch_bytes=watch.MAX_DIFF_BYTES + 1)
    with pytest.raises(watch.PolicyRejection, match="diff_too_large"):
        too_large._enforce_policy(Path("/synthetic"), "a" * 40, "b" * 40)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"binary": True}, "binary_change"),
        ({"modes": [("100644", "120000")]}, "symlink_change"),
        ({"modes": [("100644", "160000")]}, "submodule_change"),
        ({"paths": [".gitmodules"]}, "submodule_change"),
        ({"tree_symlink": True}, "symlink_tree"),
    ],
)
def test_binary_symlink_and_submodule_changes_are_rejected(options, reason):
    source = SyntheticPolicySource(**options)
    with pytest.raises(watch.PolicyRejection, match=reason):
        source._enforce_policy(Path("/synthetic"), "a" * 40, "b" * 40)


def test_retention_deletes_only_artifacts_and_preserves_claim(tmp_path):
    clock = [1_000_000.0]
    cfg = config(tmp_path)
    state = watch.StateStore(cfg.state_root, now=lambda: clock[0])
    item = pull(head="c" * 40)
    try:
        state.seed([], ENGINE_SHA)
        key = state.claim(item, ENGINE_SHA)
        run_root = cfg.state_root / "runs/old-run"
        run_root.mkdir()
        (run_root / "artifact").write_text("local")
        state.transition(key, "CLAIMED", "INCOMPLETE", reason="test", run_path=run_root)
        clock[0] += 31 * 24 * 60 * 60
        assert state.prune_artifacts(30) == 1
        assert not run_root.exists()
        row = state.connection.execute(
            "SELECT status, artifacts_pruned_at FROM claims WHERE head_oid = ?",
            (item.head_oid,),
        ).fetchone()
        assert row["status"] == "INCOMPLETE"
        assert row["artifacts_pruned_at"] is not None
        assert state.claim(item, ENGINE_SHA) is None
    finally:
        state.close()


def test_subprocess_seam_always_uses_argv_and_shell_false(monkeypatch):
    observed = []

    def fake_run(argv, **options):
        observed.append((argv, options))
        return subprocess.CompletedProcess(argv, 0, "", "")

    class FakePopen:
        def __init__(self, argv, **options):
            observed.append((argv, options))

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    process = watch.ProcessRunner()
    process.run(["/bin/synthetic", "github controlled; $(ignored)"])
    process.popen(
        ["/bin/synthetic", "another value"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert len(observed) == 2
    assert all(isinstance(argv, list) for argv, _ in observed)
    assert all(options["shell"] is False for _, options in observed)
    with pytest.raises(TypeError, match="argv"):
        process.run("/bin/not-an-argv")


def test_diff_capture_stops_at_fixed_byte_limit(tmp_path):
    process = watch.ProcessRunner()
    result = process.run_bounded(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 128)"],
        cwd=tmp_path,
        max_stdout_bytes=128,
        timeout=2,
    )
    assert len(result.stdout) == 128
    with pytest.raises(watch.OutputLimitExceeded):
        process.run_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 129)"],
            cwd=tmp_path,
            max_stdout_bytes=128,
            timeout=2,
        )


def test_operator_templates_remain_explicit_and_disabled():
    directory = ROOT / "tools/review_bot"
    timer = (directory / "systemd/flydsl-review-bot.timer").read_text()
    service = (directory / "systemd/flydsl-review-bot.service").read_text()
    dockerfile = (directory / "Dockerfile").read_text()
    assert "OnUnitActiveSec=2min" in timer
    assert "WantedBy=timers.target" in timer
    assert "systemctl" not in timer + service
    assert "@REVIEW_BOT_ROOT@" in service and "@CONFIG_PATH@" in service
    assert "PassEnvironment=ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL" in service
    assert "ARG BASE_IMAGE" in dockerfile
    assert "sha256:4d676821dff059fd00d277ee4261ef34ea712317fed0737c03941481b5760c96" in dockerfile
    assert "ARG CLAUDE_CODE_VERSION=2.1.274" in dockerfile
    assert "ARG SANDBOX_RUNTIME_VERSION=0.0.76" in dockerfile
    assert "dist.integrity" in dockerfile
    assert "USER reviewer:reviewer" in dockerfile
    assert "gh" not in [
        line.strip().split()[0]
        for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
