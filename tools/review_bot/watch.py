#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Run one finite FlyDSL pull-request review scan.

The systemd timer owns the two-minute polling cadence. This process owns the
cross-invocation lock, durable claim state, untrusted container boundary, and
the final trusted-host publication gate.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import json
import os
import re
import selectors
import shutil
import signal
import sqlite3
import stat
import subprocess
import time
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Sequence, TextIO
from urllib.parse import urlsplit

REPOSITORY = "ROCm/FlyDSL"
REPOSITORY_ID = 1102472199
AUTHOR_LOGIN = "coderfeli"
AUTHOR_ID = 184409145
PUBLISHER_LOGIN = "jhinpan"
PUBLISHER_ID = 47354855
PUBLIC_GIT_URL = "https://github.com/ROCm/FlyDSL.git"
MODEL_ENV_NAMES = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
MODEL_GATEWAY = ("http", "127.0.0.1", 8882)

MAX_FILES = 100
MAX_DIFF_BYTES = 2 * 1024 * 1024
GRACEFUL_REVIEW_SECONDS = 80 * 60
HARD_REVIEW_SECONDS = 90 * 60
CONTAINER_STOP_SECONDS = HARD_REVIEW_SECONDS - GRACEFUL_REVIEW_SECONDS
MAX_RESULT_BYTES = 64 * 1024 * 1024

CONTROL_PATHS = (".github", ".claude", "CLAUDE.md", "tools/review_bot")
ACTIVE_STATES = ("CLAIMED", "PREPARING", "RUNNING", "VALIDATING", "PUBLISHING")
TERMINAL_STATES = ("SEEDED", "INCOMPLETE", "REJECTED", "STALE", "COMPLETE")
OID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IMAGE_PATTERN = re.compile(r"(?:[^\s@]+(?:/[^\s@]+)*@)?sha256:[0-9a-f]{64}")


class WatcherError(RuntimeError):
    """A fail-closed operator error whose message is safe for local diagnostics."""


class ConfigurationError(WatcherError):
    """Invalid non-secret operator configuration."""


class CommandError(WatcherError):
    """A child process failed without exposing its output in the exception."""

    def __init__(self, program: str, returncode: int):
        super().__init__(f"{Path(program).name} exited with status {returncode}")
        self.program = program
        self.returncode = returncode


class OutputLimitExceeded(WatcherError):
    """A child process exceeded its allowed captured output."""


class PolicyRejection(WatcherError):
    """A stable policy code for a PR that must not enter the review container."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class StaleClaim(WatcherError):
    """The claimed GitHub identity moved before publication."""


@dataclasses.dataclass(frozen=True)
class Config:
    engine_root: Path
    image: str
    implementation_sha256: str
    state_root: Path = Path.home() / ".local/state/flydsl-review-bot"
    retention_days: int = 30
    gh_bin: str = "/usr/bin/gh"
    git_bin: str = "/usr/bin/git"
    docker_bin: str = "/usr/bin/docker"
    python_bin: str = "/usr/bin/python3"
    memory_limit: str = "16g"
    cpu_limit: str = "8"
    pids_limit: int = 512
    publish_enabled: bool = False

    @classmethod
    def load(cls, path: Path) -> Config:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError("cannot load operator config") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("operator config must be a JSON object")
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = set(raw) - allowed
        required = {"engine_root", "image", "implementation_sha256"}
        if unknown or not required.issubset(raw):
            raise ConfigurationError("operator config has missing or unexpected fields")
        for key in ("engine_root", "state_root"):
            if key in raw:
                if not isinstance(raw[key], str):
                    raise ConfigurationError(f"{key} must be a path string")
                raw[key] = Path(raw[key]).expanduser()
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise ConfigurationError("operator config has invalid fields") from exc
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("engine_root", "state_root"):
            path = getattr(self, name)
            if not path.is_absolute() or Path(os.path.normpath(path)) != path or "\0" in str(path) or "," in str(path):
                raise ConfigurationError(f"{name} must be a normalized absolute path without commas")
        protected_paths = (self.engine_root, self.state_root)
        for index, left in enumerate(protected_paths):
            for right in protected_paths[index + 1 :]:
                if left == right or left in right.parents or right in left.parents:
                    raise ConfigurationError("engine and state paths must not overlap")
        if not IMAGE_PATTERN.fullmatch(self.image):
            raise ConfigurationError("image must use an immutable sha256 image id or repository digest")
        if not SHA256_PATTERN.fullmatch(self.implementation_sha256):
            raise ConfigurationError("implementation_sha256 must be 64 lowercase hexadecimal characters")
        if type(self.retention_days) is not int or self.retention_days < 1:
            raise ConfigurationError("retention_days must be a positive integer")
        if type(self.pids_limit) is not int or self.pids_limit < 1:
            raise ConfigurationError("pids_limit must be a positive integer")
        if type(self.publish_enabled) is not bool:
            raise ConfigurationError("publish_enabled must be a boolean")
        if not re.fullmatch(r"[1-9][0-9]*(?:[kKmMgG])?", self.memory_limit):
            raise ConfigurationError("memory_limit must be a positive Docker memory value")
        if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?", self.cpu_limit):
            raise ConfigurationError("cpu_limit must be a positive Docker CPU value")
        for name in ("gh_bin", "git_bin", "docker_bin", "python_bin"):
            value = getattr(self, name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ConfigurationError(f"{name} must be an absolute executable path")


@dataclasses.dataclass(frozen=True)
class PullRequest:
    number: int
    head_oid: str
    base_oid: str
    base_ref: str


@dataclasses.dataclass(frozen=True)
class ClaimKey:
    repo_id: int
    pr_number: int
    head_oid: str
    engine_sha: str


@dataclasses.dataclass(frozen=True)
class PreparedSource:
    root: Path
    merge_base_oid: str
    file_count: int
    diff_bytes: int


@dataclasses.dataclass(frozen=True)
class ContainerOutcome:
    returncode: int
    timed_out: bool
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class ScanSummary:
    seeded: int = 0
    claimed: int = 0
    complete: int = 0
    incomplete: int = 0
    rejected: int = 0
    stale: int = 0
    recovered: int = 0
    retained_deleted: int = 0

    def add(self, **changes: int) -> ScanSummary:
        values = dataclasses.asdict(self)
        for key, value in changes.items():
            values[key] += value
        return ScanSummary(**values)


class ProcessRunner:
    """Subprocess seam. Every command is an argv array and never a shell string."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        input_data: str | bytes | None = None,
        timeout: float = 120,
        text: bool = True,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        if isinstance(argv, (str, bytes)) or not argv:
            raise TypeError("commands must be non-empty argv sequences")
        result = subprocess.run(
            [str(value) for value in argv],
            cwd=cwd,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=text,
            check=False,
            shell=False,
            env=env,
        )
        if check and result.returncode:
            raise CommandError(str(argv[0]), result.returncode)
        return result

    def popen(
        self,
        argv: Sequence[str],
        *,
        stdout: TextIO | BinaryIO | int,
        stderr: TextIO | BinaryIO | int,
    ) -> subprocess.Popen:
        if isinstance(argv, (str, bytes)) or not argv:
            raise TypeError("commands must be non-empty argv sequences")
        return subprocess.Popen(
            [str(value) for value in argv],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
            shell=False,
        )

    def run_bounded(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        max_stdout_bytes: int,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        if isinstance(argv, (str, bytes)) or not argv:
            raise TypeError("commands must be non-empty argv sequences")
        process = subprocess.Popen(
            [str(value) for value in argv],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            shell=False,
            env=env,
        )
        stdout, stderr = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                ready = selector.select(min(remaining, 0.2))
                for key, _ in ready:
                    chunk = os.read(key.fd, 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    destination = key.data
                    if destination is stdout and len(stdout) + len(chunk) > max_stdout_bytes:
                        raise OutputLimitExceeded("command output exceeded its fixed limit")
                    if destination is stdout or len(stderr) < 1024 * 1024:
                        destination.extend(chunk)
            remaining = max(0.0, deadline - time.monotonic())
            process.wait(timeout=remaining)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            raise
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
        result = subprocess.CompletedProcess(list(argv), process.returncode, bytes(stdout), bytes(stderr))
        if result.returncode:
            raise CommandError(str(argv[0]), result.returncode)
        return result


def parse_json_documents(value: str) -> list[object]:
    decoder = json.JSONDecoder()
    documents: list[object] = []
    position = 0
    while position < len(value):
        while position < len(value) and value[position].isspace():
            position += 1
        if position == len(value):
            break
        document, position = decoder.raw_decode(value, position)
        documents.append(document)
    return documents


def valid_oid(value: object) -> bool:
    return isinstance(value, str) and OID_PATTERN.fullmatch(value) is not None


def exact_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def repository_identity(value: object) -> bool:
    return isinstance(value, dict) and value.get("id") == REPOSITORY_ID and value.get("full_name") == REPOSITORY


def parse_eligible_pull(value: object) -> PullRequest | None:
    if not isinstance(value, dict):
        return None
    user, head, base = value.get("user"), value.get("head"), value.get("base")
    if (
        value.get("state") != "open"
        or value.get("draft") is not False
        or not exact_positive_int(value.get("number"))
        or not isinstance(user, dict)
        or user.get("id") != AUTHOR_ID
        or user.get("login") != AUTHOR_LOGIN
        or not isinstance(head, dict)
        or not isinstance(base, dict)
        or not repository_identity(head.get("repo"))
        or not repository_identity(base.get("repo"))
        or not valid_oid(head.get("sha"))
        or not valid_oid(base.get("sha"))
        or not isinstance(base.get("ref"), str)
    ):
        return None
    base_ref = base["ref"]
    if not base_ref or len(base_ref) > 255 or any(ord(character) < 32 for character in base_ref):
        return None
    return PullRequest(value["number"], head["sha"], base["sha"], base_ref)


class GitHubClient:
    def __init__(self, config: Config, process: ProcessRunner):
        self.config = config
        self.process = process

    def _json(self, endpoint: str, *, paginate: bool = False) -> list[object]:
        argv = [self.config.gh_bin, "api", "--method", "GET"]
        if paginate:
            argv.append("--paginate")
        argv.append(endpoint)
        result = self.process.run(argv, timeout=120)
        try:
            documents = parse_json_documents(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise WatcherError("gh returned malformed JSON") from exc
        if not documents:
            raise WatcherError("gh returned no JSON document")
        return documents

    def verify_repository(self) -> None:
        documents = self._json("repos/ROCm/FlyDSL")
        if len(documents) != 1 or not repository_identity(documents[0]):
            raise WatcherError("repository identity mismatch")

    def verify_publisher(self) -> None:
        documents = self._json("user")
        identity = documents[0] if len(documents) == 1 else None
        if (
            not isinstance(identity, dict)
            or identity.get("id") != PUBLISHER_ID
            or identity.get("login") != PUBLISHER_LOGIN
        ):
            raise WatcherError("publisher identity mismatch")

    def open_eligible_pulls(self) -> list[PullRequest]:
        self.verify_repository()
        self.verify_publisher()
        documents = self._json("repos/ROCm/FlyDSL/pulls?state=open&per_page=100", paginate=True)
        pulls: list[PullRequest] = []
        for page in documents:
            if not isinstance(page, list):
                raise WatcherError("gh pull pagination returned a non-list page")
            pulls.extend(candidate for item in page if (candidate := parse_eligible_pull(item)) is not None)
        by_number: dict[int, PullRequest] = {}
        for pull in pulls:
            previous = by_number.get(pull.number)
            if previous and previous != pull:
                raise WatcherError("gh returned conflicting pull metadata")
            by_number[pull.number] = pull
        return [by_number[number] for number in sorted(by_number)]

    def current_pull(self, number: int) -> PullRequest | None:
        documents = self._json(f"repos/ROCm/FlyDSL/pulls/{number}")
        return parse_eligible_pull(documents[0]) if len(documents) == 1 else None


def ensure_private_root(root: Path) -> None:
    if root.is_symlink():
        raise ConfigurationError("state_root cannot be a symlink")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ConfigurationError("state_root must be a directory")
    os.chmod(root, 0o700)
    for child in ("runs",):
        path = root / child
        if path.is_symlink():
            raise ConfigurationError("state directory cannot be a symlink")
        path.mkdir(mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)


@contextlib.contextmanager
def scan_lock(root: Path) -> Iterable[bool]:
    lock_path = root / "watch.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    os.chmod(lock_path, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


class StateStore:
    SCHEMA_VERSION = "1"

    def __init__(self, root: Path, *, now: Callable[[], float] = time.time):
        self.root = root
        self.now = now
        database = root / "state.sqlite3"
        if database.is_symlink():
            raise ConfigurationError("watcher database cannot be a symlink")
        self.connection = sqlite3.connect(database, timeout=30, isolation_level=None)
        os.chmod(database, 0o600)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                repo_id INTEGER NOT NULL,
                pr_number INTEGER NOT NULL,
                head_oid TEXT NOT NULL,
                engine_sha TEXT NOT NULL,
                base_oid TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'CLAIMED', 'PREPARING', 'RUNNING', 'VALIDATING', 'PUBLISHING',
                        'SEEDED', 'INCOMPLETE', 'REJECTED', 'STALE', 'COMPLETE'
                    )
                ),
                reason TEXT,
                run_path TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                terminal_at REAL,
                artifacts_pruned_at REAL,
                PRIMARY KEY (repo_id, pr_number, head_oid, engine_sha)
            )
            """)
        row = self.connection.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)", (self.SCHEMA_VERSION,)
            )
        elif row["value"] != self.SCHEMA_VERSION:
            raise ConfigurationError("unsupported watcher database schema")

    def close(self) -> None:
        self.connection.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterable[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def recover_interrupted(self) -> int:
        now = self.now()
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        with self.transaction():
            cursor = self.connection.execute(
                f"""
                UPDATE claims
                   SET status = 'INCOMPLETE',
                       reason = 'interrupted',
                       updated_at = ?,
                       terminal_at = ?
                 WHERE status IN ({placeholders})
                """,
                (now, now, *ACTIVE_STATES),
            )
        return cursor.rowcount

    def active_claims(self) -> list[ClaimKey]:
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        rows = self.connection.execute(
            f"""
            SELECT repo_id, pr_number, head_oid, engine_sha
              FROM claims
             WHERE status IN ({placeholders})
             ORDER BY created_at, pr_number
            """,
            ACTIVE_STATES,
        ).fetchall()
        return [ClaimKey(row["repo_id"], row["pr_number"], row["head_oid"], row["engine_sha"]) for row in rows]

    def initialized(self) -> bool:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = 'initialized'").fetchone()
        return row is not None and row["value"] == "1"

    def seed(self, pulls: Sequence[PullRequest], engine_sha: str) -> int:
        now = self.now()
        inserted = 0
        with self.transaction():
            if self.connection.execute("SELECT 1 FROM metadata WHERE key = 'initialized'").fetchone():
                return 0
            for pull in pulls:
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO claims(
                        repo_id, pr_number, head_oid, engine_sha, base_oid, status,
                        reason, run_path, created_at, updated_at, terminal_at
                    ) VALUES (?, ?, ?, ?, ?, 'SEEDED', 'initial_seed', NULL, ?, ?, ?)
                    """,
                    (REPOSITORY_ID, pull.number, pull.head_oid, engine_sha, pull.base_oid, now, now, now),
                )
                inserted += cursor.rowcount
            self.connection.execute("INSERT INTO metadata(key, value) VALUES ('initialized', '1')")
        return inserted

    def claim(self, pull: PullRequest, engine_sha: str) -> ClaimKey | None:
        now = self.now()
        key = ClaimKey(REPOSITORY_ID, pull.number, pull.head_oid, engine_sha)
        with self.transaction():
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO claims(
                    repo_id, pr_number, head_oid, engine_sha, base_oid, status,
                    reason, run_path, created_at, updated_at, terminal_at
                ) VALUES (?, ?, ?, ?, ?, 'CLAIMED', NULL, NULL, ?, ?, NULL)
                """,
                (
                    key.repo_id,
                    key.pr_number,
                    key.head_oid,
                    key.engine_sha,
                    pull.base_oid,
                    now,
                    now,
                ),
            )
        return key if cursor.rowcount == 1 else None

    def claim_seeded(self, pull: PullRequest, engine_sha: str) -> ClaimKey | None:
        now = self.now()
        key = ClaimKey(REPOSITORY_ID, pull.number, pull.head_oid, engine_sha)
        with self.transaction():
            cursor = self.connection.execute(
                """
                UPDATE claims
                   SET status = 'CLAIMED', reason = 'manual_canary',
                       updated_at = ?, terminal_at = NULL
                 WHERE repo_id = ? AND pr_number = ?
                   AND head_oid = ? AND engine_sha = ? AND status = 'SEEDED'
                """,
                (now, *dataclasses.astuple(key)),
            )
        return key if cursor.rowcount == 1 else None

    def transition(
        self,
        key: ClaimKey,
        expected: str | Sequence[str],
        status_value: str,
        *,
        reason: str | None = None,
        run_path: Path | None = None,
    ) -> None:
        expected_states = (expected,) if isinstance(expected, str) else tuple(expected)
        if status_value not in ACTIVE_STATES + TERMINAL_STATES:
            raise ValueError("invalid claim state")
        now = self.now()
        terminal = now if status_value in TERMINAL_STATES else None
        placeholders = ",".join("?" for _ in expected_states)
        with self.transaction():
            cursor = self.connection.execute(
                f"""
                UPDATE claims
                   SET status = ?, reason = ?, run_path = COALESCE(?, run_path),
                       updated_at = ?, terminal_at = ?
                 WHERE repo_id = ? AND pr_number = ? AND head_oid = ? AND engine_sha = ?
                   AND status IN ({placeholders})
                """,
                (
                    status_value,
                    reason,
                    str(run_path) if run_path else None,
                    now,
                    terminal,
                    key.repo_id,
                    key.pr_number,
                    key.head_oid,
                    key.engine_sha,
                    *expected_states,
                ),
            )
        if cursor.rowcount != 1:
            raise WatcherError("durable claim transition conflict")

    def terminal_status(self, key: ClaimKey) -> str:
        row = self.connection.execute(
            """
            SELECT status FROM claims
             WHERE repo_id = ? AND pr_number = ? AND head_oid = ? AND engine_sha = ?
            """,
            dataclasses.astuple(key),
        ).fetchone()
        if row is None:
            raise WatcherError("claim disappeared")
        return row["status"]

    def prune_artifacts(self, retention_days: int) -> int:
        cutoff = self.now() - retention_days * 24 * 60 * 60
        rows = self.connection.execute(
            """
            SELECT repo_id, pr_number, head_oid, engine_sha, run_path
              FROM claims
             WHERE terminal_at IS NOT NULL AND terminal_at < ?
               AND run_path IS NOT NULL AND artifacts_pruned_at IS NULL
            """,
            (cutoff,),
        ).fetchall()
        runs_root = self.root / "runs"
        deleted = 0
        for row in rows:
            run_path = Path(row["run_path"])
            if run_path.parent != runs_root or run_path == runs_root:
                raise WatcherError("refusing to prune a path outside the run root")
            if run_path.is_symlink():
                run_path.unlink()
            elif run_path.exists():
                shutil.rmtree(run_path)
            now = self.now()
            with self.transaction():
                self.connection.execute(
                    """
                    UPDATE claims SET artifacts_pruned_at = ?, updated_at = ?
                     WHERE repo_id = ? AND pr_number = ? AND head_oid = ? AND engine_sha = ?
                    """,
                    (
                        now,
                        now,
                        row["repo_id"],
                        row["pr_number"],
                        row["head_oid"],
                        row["engine_sha"],
                    ),
                )
            deleted += 1
        return deleted


def git_environment() -> dict[str, str]:
    allowed = {
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "NO_PROXY",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )
    return environment


def require_model_environment() -> None:
    missing = [name for name in MODEL_ENV_NAMES if not os.environ.get(name)]
    if missing:
        raise ConfigurationError("required model gateway environment is unavailable")
    endpoint = urlsplit(os.environ["ANTHROPIC_BASE_URL"])
    if (
        (endpoint.scheme, endpoint.hostname, endpoint.port) != MODEL_GATEWAY
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise ConfigurationError("model gateway endpoint does not match the local deployment")


class SourceManager:
    def __init__(self, config: Config, process: ProcessRunner):
        self.config = config
        self.process = process
        self.environment = git_environment()

    def git_argv(self, *args: str) -> list[str]:
        return [
            self.config.git_bin,
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-c",
            "submodule.recurse=false",
            *args,
        ]

    def git(
        self,
        root: Path | None,
        *args: str,
        text: bool = True,
        check: bool = True,
        timeout: float = 300,
    ) -> subprocess.CompletedProcess:
        return self.process.run(
            self.git_argv(*args),
            cwd=root,
            timeout=timeout,
            text=text,
            check=check,
            env=self.environment,
        )

    def _bounded_diff_size(self, source: Path, diff_base: str, head_oid: str) -> int:
        try:
            result = self.process.run_bounded(
                self.git_argv(
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-renames",
                    diff_base,
                    head_oid,
                    "--",
                ),
                cwd=source,
                max_stdout_bytes=MAX_DIFF_BYTES,
                timeout=600,
                env=self.environment,
            )
        except OutputLimitExceeded as exc:
            raise PolicyRejection("diff_too_large") from exc
        return len(result.stdout)

    def engine_identity(self) -> str:
        status_result = self.git(
            self.config.engine_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if status_result.stdout:
            raise WatcherError("trusted engine checkout is not clean")
        result = self.git(self.config.engine_root, "rev-parse", "--verify", "HEAD^{commit}")
        engine_sha = result.stdout.strip()
        if not valid_oid(engine_sha):
            raise WatcherError("trusted engine has an invalid commit identity")
        return engine_sha

    def prepare_engine(self, destination: Path, engine_sha: str) -> Path:
        self.git(
            None,
            "clone",
            "--quiet",
            "--no-checkout",
            "--no-local",
            str(self.config.engine_root),
            str(destination),
            timeout=600,
        )
        self.git(destination, "checkout", "--quiet", "--detach", engine_sha, timeout=600)
        observed = self.git(destination, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
        status_result = self.git(destination, "status", "--porcelain=v1", "--untracked-files=all")
        if observed != engine_sha or status_result.stdout:
            raise WatcherError("prepared engine snapshot is not clean at the claimed commit")
        return destination

    def _validate_base_ref(self, base_ref: str) -> None:
        result = self.git(
            None,
            "check-ref-format",
            f"refs/heads/{base_ref}",
            check=False,
        )
        if result.returncode:
            raise PolicyRejection("invalid_base_ref")

    def _raw_diff_records(self, source: Path, diff_base: str, head_oid: str) -> list[tuple[str, str, list[str]]]:
        result = self.git(
            source,
            "diff",
            "--raw",
            "-z",
            "--no-abbrev",
            "--no-renames",
            "--no-textconv",
            diff_base,
            head_oid,
            "--",
            text=False,
        )
        fields = result.stdout.split(b"\0")
        records: list[tuple[str, str, list[str]]] = []
        position = 0
        while position < len(fields) and fields[position]:
            try:
                header = fields[position].decode("ascii").split()
                if len(header) != 5 or not header[0].startswith(":"):
                    raise ValueError
                old_mode, new_mode, change = header[0][1:], header[1], header[4]
                position += 1
                paths = [os.fsdecode(fields[position])]
                position += 1
                if change.startswith(("R", "C")):
                    paths.append(os.fsdecode(fields[position]))
                    position += 1
            except (IndexError, UnicodeDecodeError, ValueError) as exc:
                raise WatcherError("git returned a malformed raw diff") from exc
            records.append((old_mode, new_mode, paths))
        return records

    def _enforce_policy(self, source: Path, base_oid: str, head_oid: str) -> tuple[str, int, int]:
        merge_base = self.git(source, "merge-base", base_oid, head_oid).stdout.strip()
        if not valid_oid(merge_base):
            raise WatcherError("git returned an invalid merge base")
        records = self._raw_diff_records(source, merge_base, head_oid)
        if len(records) > MAX_FILES:
            raise PolicyRejection("too_many_files")
        for old_mode, new_mode, paths in records:
            if "120000" in (old_mode, new_mode):
                raise PolicyRejection("symlink_change")
            if "160000" in (old_mode, new_mode):
                raise PolicyRejection("submodule_change")
            for changed_path in paths:
                if not safe_repo_path(changed_path):
                    raise PolicyRejection("unsafe_path")
                if changed_path == ".gitmodules":
                    raise PolicyRejection("submodule_change")
                if control_path(changed_path):
                    raise PolicyRejection("control_path_change")

        numstat = self.git(
            source,
            "diff",
            "--numstat",
            "-z",
            "--no-renames",
            "--no-textconv",
            merge_base,
            head_oid,
            "--",
            text=False,
        ).stdout
        for record in numstat.split(b"\0"):
            if not record:
                continue
            columns = record.split(b"\t", 2)
            if len(columns) != 3:
                raise WatcherError("git returned malformed numstat data")
            if columns[0] == b"-" or columns[1] == b"-":
                raise PolicyRejection("binary_change")

        tree = self.git(
            source,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            head_oid,
            text=False,
        ).stdout
        for entry in tree.split(b"\0"):
            if entry.startswith(b"120000 "):
                raise PolicyRejection("symlink_tree")

        diff_bytes = self._bounded_diff_size(source, merge_base, head_oid)
        return merge_base, len(records), diff_bytes

    def prepare_source(self, destination: Path, pull: PullRequest) -> PreparedSource:
        self._validate_base_ref(pull.base_ref)
        destination.mkdir(mode=0o700)
        self.git(destination, "init", "--quiet")
        self.git(destination, "remote", "add", "origin", PUBLIC_GIT_URL)
        self.git(
            destination,
            "fetch",
            "--quiet",
            "--no-tags",
            "--filter=blob:none",
            "--force",
            "origin",
            f"+{pull.base_oid}:refs/review-bot/base",
            f"+{pull.head_oid}:refs/review-bot/head",
            timeout=1200,
        )
        observed_base = self.git(destination, "rev-parse", "--verify", "refs/review-bot/base^{commit}").stdout.strip()
        observed_head = self.git(destination, "rev-parse", "--verify", "refs/review-bot/head^{commit}").stdout.strip()
        if observed_base != pull.base_oid or observed_head != pull.head_oid:
            raise StaleClaim("head_or_base_moved_during_fetch")
        merge_base, file_count, diff_bytes = self._enforce_policy(destination, pull.base_oid, pull.head_oid)
        self.git(destination, "checkout", "--quiet", "--detach", pull.head_oid, timeout=1200)
        status_result = self.git(destination, "status", "--porcelain=v1", "--untracked-files=all")
        observed_head = self.git(destination, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
        if status_result.stdout or observed_head != pull.head_oid:
            raise WatcherError("prepared source repository is not clean at the claimed head")
        return PreparedSource(destination, merge_base, file_count, diff_bytes)


def control_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in CONTROL_PATHS)


def safe_repo_path(path: str) -> bool:
    return (
        bool(path)
        and len(path) <= 4096
        and not path.startswith("/")
        and ".." not in Path(path).parts
        and not any(ord(character) < 32 or 0x7F <= ord(character) <= 0x9F for character in path)
    )


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_private(path: Path, value: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def scope_manifest(pull: PullRequest) -> dict:
    return {
        "schema_version": 1,
        "repository_id": REPOSITORY_ID,
        "repo": REPOSITORY,
        "pr": pull.number,
        "author_id": AUTHOR_ID,
        "author_login": AUTHOR_LOGIN,
        "head_repo": REPOSITORY,
        "base_oid": pull.base_oid,
        "head_oid": pull.head_oid,
    }


def mount(source: Path, target: str, *, readonly: bool) -> str:
    options = ["type=bind", f"src={source}", f"dst={target}", "bind-propagation=rprivate"]
    if readonly:
        options.append("readonly")
    return ",".join(options)


def container_name(pull: PullRequest, engine_sha: str) -> str:
    return f"flydsl-review-{pull.number}-{pull.head_oid[:12]}-{engine_sha[:12]}"


def container_labels(pr_number: int, head_oid: str, engine_sha: str) -> dict[str, str]:
    return {
        "com.amd.flydsl.review-bot": "local-pilot",
        "com.amd.flydsl.repository-id": str(REPOSITORY_ID),
        "com.amd.flydsl.pr-number": str(pr_number),
        "com.amd.flydsl.head-oid": head_oid,
        "com.amd.flydsl.engine-sha": engine_sha,
    }


def build_docker_argv(
    config: Config,
    pull: PullRequest,
    engine_sha: str,
    *,
    engine_root: Path,
    source_root: Path,
    input_root: Path,
    output_root: Path,
) -> list[str]:
    name = container_name(pull, engine_sha)
    return [
        config.docker_bin,
        "run",
        "--rm",
        "--init",
        "--log-driver",
        "none",
        "--name",
        name,
        "--hostname",
        "flydsl-review",
        "--runtime",
        "runc",
        "--network",
        "host",
        "--ipc",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        str(config.pids_limit),
        "--memory",
        config.memory_limit,
        "--memory-swap",
        config.memory_limit,
        "--cpus",
        config.cpu_limit,
        "--stop-timeout",
        str(CONTAINER_STOP_SECONDS),
        "--ulimit",
        "nofile=4096:4096",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/home/reviewer",
        "--env",
        "LANG=C.UTF-8",
        "--env",
        "LC_ALL=C.UTF-8",
        "--env",
        "GIT_CONFIG_GLOBAL=/dev/null",
        "--env",
        "GIT_CONFIG_NOSYSTEM=1",
        "--env",
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1",
        *[value for name in MODEL_ENV_NAMES for value in ("--env", name)],
        "--env",
        "ROCR_VISIBLE_DEVICES=-1",
        "--env",
        "HIP_VISIBLE_DEVICES=-1",
        "--env",
        "CUDA_VISIBLE_DEVICES=-1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=2g,mode=1777",
        "--mount",
        mount(engine_root, "/opt/review-engine", readonly=True),
        "--mount",
        mount(source_root, "/workspace/source", readonly=True),
        "--mount",
        mount(input_root, "/review-input", readonly=True),
        "--mount",
        mount(output_root, "/review-run", readonly=False),
        "--workdir",
        "/workspace/source",
        *[
            value
            for key, label_value in container_labels(pull.number, pull.head_oid, engine_sha).items()
            for value in ("--label", f"{key}={label_value}")
        ],
        config.image,
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


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


class DockerBackend:
    def __init__(
        self,
        config: Config,
        process: ProcessRunner,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.process = process
        self.monotonic = monotonic

    def cleanup(self, key: ClaimKey) -> None:
        name = f"flydsl-review-{key.pr_number}-{key.head_oid[:12]}-{key.engine_sha[:12]}"
        listing = self.process.run(
            [
                self.config.docker_bin,
                "container",
                "ls",
                "--all",
                "--filter",
                f"name=^/{name}$",
                "--format",
                "{{.Names}}",
            ],
            timeout=30,
        )
        if name not in listing.stdout.splitlines():
            return
        inspected = self.process.run(
            [
                self.config.docker_bin,
                "container",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                name,
            ],
            timeout=30,
        )
        try:
            labels = json.loads(inspected.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise WatcherError("cannot verify interrupted review container labels") from exc
        expected = container_labels(key.pr_number, key.head_oid, key.engine_sha)
        if not isinstance(labels, dict) or any(labels.get(name) != value for name, value in expected.items()):
            raise WatcherError("interrupted container identity mismatch")
        self.process.run(
            [self.config.docker_bin, "container", "rm", "--force", name],
            timeout=30,
        )

    def run(self, argv: Sequence[str], name: str, artifact_root: Path) -> ContainerOutcome:
        stdout_path, stderr_path = artifact_root / "container.stdout.log", artifact_root / "container.stderr.log"
        started = self.monotonic()
        timed_out = False
        with stdout_path.open("wb", buffering=0) as stdout, stderr_path.open("wb", buffering=0) as stderr:
            os.chmod(stdout_path, 0o600)
            os.chmod(stderr_path, 0o600)
            process = self.process.popen(argv, stdout=stdout, stderr=stderr)
            try:
                try:
                    process.wait(timeout=GRACEFUL_REVIEW_SECONDS)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stopper = self.process.popen(
                        [self.config.docker_bin, "stop", "--time", str(CONTAINER_STOP_SECONDS), name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    remaining = max(0.0, HARD_REVIEW_SECONDS - (self.monotonic() - started))
                    try:
                        process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(CommandError, subprocess.TimeoutExpired):
                            self.process.run(
                                [self.config.docker_bin, "kill", name],
                                timeout=15,
                                check=False,
                            )
                    finally:
                        terminate_process_group(stopper)
            finally:
                terminate_process_group(process)
        return ContainerOutcome(
            process.returncode if process.returncode is not None else -signal.SIGKILL,
            timed_out,
            self.monotonic() - started,
        )


def build_publisher_argv(
    config: Config,
    *,
    engine_root: Path,
    result_path: Path,
    pull: PullRequest,
) -> list[str]:
    publisher = engine_root / ".claude/skills/flydsl-code-review/scripts/post_review.py"
    argv = [
        config.python_bin,
        str(publisher),
        "--result",
        str(result_path),
        "--repo",
        REPOSITORY,
        "--pr",
        str(pull.number),
        "--expected-head",
        pull.head_oid,
        "--expected-implementation-sha256",
        config.implementation_sha256,
        "--expected-publisher-id",
        str(PUBLISHER_ID),
        "--expected-repository-id",
        str(REPOSITORY_ID),
        "--expected-author-id",
        str(AUTHOR_ID),
        "--expected-author-login",
        AUTHOR_LOGIN,
        "--publish-severity",
        "P1",
    ]
    if not config.publish_enabled:
        argv.append("--dry-run")
    return argv


class HostPublisher:
    def __init__(self, process: ProcessRunner):
        self.process = process

    def publish(self, argv: Sequence[str], artifact_root: Path) -> int:
        result = self.process.run(argv, timeout=180, check=False)
        write_private(artifact_root / "publisher.stdout.log", result.stdout)
        write_private(artifact_root / "publisher.stderr.log", result.stderr)
        return result.returncode


def read_result(path: Path, config: Config, pull: PullRequest) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RESULT_BYTES:
            raise WatcherError("result artifact is not a bounded regular file")
        try:
            result = json.load(stream)
        except json.JSONDecodeError as exc:
            raise WatcherError("result artifact is malformed") from exc
    if not isinstance(result, dict) or result.get("status") != "COMPLETE":
        raise WatcherError("review result is not COMPLETE")
    if result.get("implementation_sha256") != config.implementation_sha256:
        raise WatcherError("review implementation hash mismatch")
    scope = result.get("scope")
    expected_scope = {
        "repository_id": REPOSITORY_ID,
        "repo": REPOSITORY,
        "pr": pull.number,
        "author_id": AUTHOR_ID,
        "author_login": AUTHOR_LOGIN,
        "head_repo": REPOSITORY,
        "base_oid": pull.base_oid,
        "head_oid": pull.head_oid,
    }
    if not isinstance(scope, dict) or any(scope.get(key) != value for key, value in expected_scope.items()):
        raise WatcherError("review result scope mismatch")
    run_config = result.get("config")
    if (
        not isinstance(run_config, dict)
        or run_config.get("execution_profile") != "untrusted-container"
        or run_config.get("model") != "opus"
        or run_config.get("effort") != "max"
    ):
        raise WatcherError("review execution profile mismatch")
    return result


class Watcher:
    def __init__(
        self,
        config: Config,
        state: StateStore,
        github: GitHubClient,
        source: SourceManager,
        docker: DockerBackend,
        publisher: HostPublisher,
        *,
        now: Callable[[], float] = time.time,
    ):
        self.config = config
        self.state = state
        self.github = github
        self.source = source
        self.docker = docker
        self.publisher = publisher
        self.now = now

    def _new_run_root(self, pull: PullRequest, engine_sha: str) -> Path:
        stamp = dt.datetime.fromtimestamp(self.now(), tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = (
            self.config.state_root
            / "runs"
            / (f"{stamp}-pr{pull.number}-{pull.base_oid[:8]}-{pull.head_oid[:12]}-{engine_sha[:12]}")
        )
        suffix = 0
        candidate = root
        while candidate.exists():
            suffix += 1
            candidate = root.with_name(root.name + f"-{suffix}")
        candidate.mkdir(mode=0o700)
        for child in ("input", "output"):
            (candidate / child).mkdir(mode=0o700)
        return candidate

    def _record_failure(self, run_root: Path | None, stage: str, exc: BaseException) -> None:
        if run_root is None:
            return
        with contextlib.suppress(OSError, TypeError, ValueError):
            atomic_json(
                run_root / "operator-failure.json",
                {
                    "stage": stage,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    @staticmethod
    def _add_status(summary: ScanSummary, status_value: str) -> ScanSummary:
        field = {
            "COMPLETE": "complete",
            "INCOMPLETE": "incomplete",
            "REJECTED": "rejected",
            "STALE": "stale",
        }.get(status_value)
        if field is None:
            raise WatcherError("review returned an unexpected terminal state")
        return summary.add(**{field: 1})

    def _process(self, key: ClaimKey, pull: PullRequest) -> str:
        run_root: Path | None = None
        stage = "prepare"
        try:
            run_root = self._new_run_root(pull, key.engine_sha)
            self.state.transition(key, "CLAIMED", "PREPARING", run_path=run_root)
            source_root = self.source.prepare_source(run_root / "source", pull)
            engine_root = self.source.prepare_engine(run_root / "engine", key.engine_sha)
            input_root, output_root = run_root / "input", run_root / "output"
            atomic_json(input_root / "scope-manifest.json", scope_manifest(pull))
            atomic_json(
                input_root / "policy.json",
                {
                    "file_count": source_root.file_count,
                    "diff_bytes": source_root.diff_bytes,
                    "merge_base_oid": source_root.merge_base_oid,
                    "limits": {"files": MAX_FILES, "diff_bytes": MAX_DIFF_BYTES},
                },
            )

            stage = "review"
            argv = build_docker_argv(
                self.config,
                pull,
                key.engine_sha,
                engine_root=engine_root,
                source_root=source_root.root,
                input_root=input_root,
                output_root=output_root,
            )
            atomic_json(run_root / "container-command.json", {"argv": argv})
            self.state.transition(key, "PREPARING", "RUNNING")
            outcome = self.docker.run(argv, container_name(pull, key.engine_sha), run_root)
            atomic_json(run_root / "container-outcome.json", dataclasses.asdict(outcome))
            if outcome.timed_out or outcome.returncode != 0:
                raise WatcherError("review container did not complete successfully")

            stage = "validate"
            self.state.transition(key, "RUNNING", "VALIDATING")
            result_path = output_root / "result.json"
            read_result(result_path, self.config, pull)
            current = self.github.current_pull(pull.number)
            if current != pull:
                self.state.transition(key, "VALIDATING", "STALE", reason="pull_identity_moved")
                return "STALE"
            self.github.verify_publisher()

            stage = "publish"
            publisher_argv = build_publisher_argv(
                self.config,
                engine_root=engine_root,
                result_path=result_path,
                pull=pull,
            )
            atomic_json(run_root / "publisher-command.json", {"argv": publisher_argv})
            self.state.transition(key, "VALIDATING", "PUBLISHING")
            if self.publisher.publish(publisher_argv, run_root) != 0:
                raise WatcherError("trusted host publisher failed")
            self.state.transition(key, "PUBLISHING", "COMPLETE")
            return "COMPLETE"
        except PolicyRejection as exc:
            self._record_failure(run_root, stage, exc)
            current = self.state.terminal_status(key)
            if current in ACTIVE_STATES:
                self.state.transition(key, current, "REJECTED", reason=exc.code, run_path=run_root)
            return "REJECTED"
        except StaleClaim as exc:
            self._record_failure(run_root, stage, exc)
            current = self.state.terminal_status(key)
            if current in ACTIVE_STATES:
                self.state.transition(key, current, "STALE", reason="head_or_base_moved", run_path=run_root)
            return "STALE"
        except (OSError, ValueError, TypeError, WatcherError, subprocess.TimeoutExpired) as exc:
            self._record_failure(run_root, stage, exc)
            current = self.state.terminal_status(key)
            if current in ACTIVE_STATES:
                self.state.transition(key, current, "INCOMPLETE", reason=f"{stage}_failed", run_path=run_root)
            return "INCOMPLETE"

    def run_once(self) -> ScanSummary:
        summary = ScanSummary()
        for key in self.state.active_claims():
            self.docker.cleanup(key)
        recovered = self.state.recover_interrupted()
        summary = summary.add(recovered=recovered)
        engine_sha = self.source.engine_identity()
        pulls = self.github.open_eligible_pulls()
        if not self.state.initialized():
            seeded = self.state.seed(pulls, engine_sha)
            deleted = self.state.prune_artifacts(self.config.retention_days)
            return summary.add(seeded=seeded, retained_deleted=deleted)
        for pull in pulls:
            key = self.state.claim(pull, engine_sha)
            if key is None:
                continue
            summary = summary.add(claimed=1)
            status_value = self._process(key, pull)
            summary = self._add_status(summary, status_value)
        deleted = self.state.prune_artifacts(self.config.retention_days)
        return summary.add(retained_deleted=deleted)

    def run_canary(self, pr_number: int) -> ScanSummary:
        if self.config.publish_enabled:
            raise ConfigurationError("canary mode requires publish_enabled=false")
        summary = ScanSummary()
        for key in self.state.active_claims():
            self.docker.cleanup(key)
        summary = summary.add(recovered=self.state.recover_interrupted())
        engine_sha = self.source.engine_identity()
        pulls = self.github.open_eligible_pulls()
        if not self.state.initialized():
            summary = summary.add(seeded=self.state.seed(pulls, engine_sha))
        pull = next((item for item in pulls if item.number == pr_number), None)
        if pull is None:
            raise PolicyRejection("canary_pr_not_eligible")
        key = self.state.claim_seeded(pull, engine_sha) or self.state.claim(pull, engine_sha)
        if key is None:
            raise PolicyRejection("canary_scope_already_terminal")
        summary = summary.add(claimed=1)
        summary = self._add_status(summary, self._process(key, pull))
        deleted = self.state.prune_artifacts(self.config.retention_days)
        return summary.add(retained_deleted=deleted)


def summary_line(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True, separators=(",", ":")), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--canary-pr", type=int)
    args = parser.parse_args(argv)
    previous_umask = os.umask(0o077)
    state: StateStore | None = None
    try:
        config = Config.load(args.config)
        require_model_environment()
        ensure_private_root(config.state_root)
        with scan_lock(config.state_root) as acquired:
            if not acquired:
                summary_line("scan_skipped", reason="lock_busy")
                return 0
            process = ProcessRunner()
            state = StateStore(config.state_root)
            watcher = Watcher(
                config,
                state,
                GitHubClient(config, process),
                SourceManager(config, process),
                DockerBackend(config, process),
                HostPublisher(process),
            )
            if args.canary_pr is not None and args.canary_pr < 1:
                raise ConfigurationError("canary PR must be a positive integer")
            summary = watcher.run_canary(args.canary_pr) if args.canary_pr else watcher.run_once()
            summary_line("scan_complete", **dataclasses.asdict(summary))
            return 0
    except (OSError, ValueError, TypeError, WatcherError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        summary_line("scan_failed", reason=type(exc).__name__)
        return 1
    finally:
        if state is not None:
            state.close()
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
