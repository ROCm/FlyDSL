"""Simple on-disk cache for compiled FLIR modules (inspired by Triton cache).

This is intentionally minimal:
- File-based cache only (no remote backend)
- Per-key directory with atomic writes (tmp dir + os.replace)
- Optional file lock to avoid concurrent writers
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import sysconfig
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in {"", "0", "false", "no", "off"}


def _default_cache_root() -> Path:
    # Roughly mirrors Triton behavior (user cache dir).
    # Allow override:
    #   FLIR_CACHE_DIR=/path/to/cache
    p = os.environ.get("FLIR_CACHE_DIR", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return (Path(xdg) / "flydsl").expanduser().resolve()
    return (Path.home() / ".cache" / "flydsl").expanduser().resolve()


def cache_enabled() -> bool:
    # Default on.
    if _env_truthy("FLIR_NO_CACHE", "0"):
        return False
    if _env_truthy("FLIR_CACHE_DISABLE", "0"):
        return False
    return True


def cache_rebuild_requested() -> bool:
    # User requested env var: "rebuild".
    # Accept both names.
    return _env_truthy("FLIR_REBUILD", "0") or _env_truthy("FLIR_CACHE_REBUILD", "0")


def make_cache_key(payload: dict) -> str:
    """Return a stable hex key."""
    # Make sure payload is JSON-stable.
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _flydsl_version() -> str:
    try:
        import flydsl  # type: ignore
        return str(getattr(flydsl, "__version__", "")) or ""
    except Exception:
        return ""


def _git_commit() -> str:
    """Best-effort git commit hash for the repo (empty if unavailable)."""
    try:
        # Repo root: flydsl/src/flydsl/compiler/cache.py -> .../flydsl/src/flydsl/compiler
        # parents[4] should be repo root containing `.git` in this checkout.
        repo = Path(__file__).resolve().parents[4]
        if not (repo / ".git").exists():
            return ""
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Short-circuit obviously bad values.
        if len(out) < 7:
            return ""
        return out
    except Exception:
        return ""


@dataclass(frozen=True)
class CachePaths:
    root: Path
    key: str

    @property
    def dir(self) -> Path:
        return self.root / self.key

    @property
    def lock(self) -> Path:
        return self.dir / "lock"

    @property
    def module_mlir(self) -> Path:
        return self.dir / "module.mlir"

    @property
    def meta_json(self) -> Path:
        return self.dir / "meta.json"


class FileCache:
    def __init__(self, *, key: str):
        self.paths = CachePaths(root=_default_cache_root(), key=key)
        self.paths.dir.mkdir(parents=True, exist_ok=True)

    def _lock_fd(self):
        # Best-effort: if fcntl isn't available, operate without locking.
        try:
            import fcntl  # type: ignore
        except Exception:
            return None
        fd = os.open(str(self.paths.lock), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            return None
        return fd

    def _unlock_fd(self, fd):
        if fd is None:
            return
        try:
            import fcntl  # type: ignore
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass

    @contextlib.contextmanager
    def lock(self):
        """Process-level lock for this cache key (best-effort)."""
        fd = self._lock_fd()
        try:
            yield fd
        finally:
            self._unlock_fd(fd)

    def get_module_asm(self) -> Optional[str]:
        p = self.paths.module_mlir
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def put_module_asm(self, asm: str, *, meta: Optional[dict] = None, lock_fd=None) -> None:
        # Atomic write pattern from Triton cache: write to temp dir then replace.
        fd = lock_fd if lock_fd is not None else self._lock_fd()
        try:
            rnd_id = str(uuid.uuid4())
            pid = os.getpid()
            tmp_dir = self.paths.dir / f"tmp.pid_{pid}_{rnd_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_mlir = tmp_dir / "module.mlir"
            tmp_mlir.write_text(asm, encoding="utf-8")
            os.replace(str(tmp_mlir), str(self.paths.module_mlir))
            try:
                tmp_dir.rmdir()
            except Exception:
                pass

            if meta is not None:
                tmp_meta = tmp_dir / "meta.json"
                tmp_meta.write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
                os.replace(str(tmp_meta), str(self.paths.meta_json))
        finally:
            if lock_fd is None:
                self._unlock_fd(fd)


def default_key_payload(*, chip: str, pipeline: str, input_asm: str) -> dict:
    """Build a default cache-key payload similar in spirit to Triton."""
    return {
        "chip": str(chip),
        "pipeline": str(pipeline),
        "input_sha256": hashlib.sha256(input_asm.encode("utf-8")).hexdigest(),
        "flydsl_version": _flydsl_version(),
        "git_commit": _git_commit(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "soabi": sysconfig.get_config_var("SOABI") or "",
        "platform": sysconfig.get_platform(),
    }

