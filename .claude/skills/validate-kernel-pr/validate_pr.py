#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Reproducible validation executor for FlyDSL kernel PRs.

``review-pr`` is static: it never builds and never runs, so it cannot see a kernel that
is wrong while its own suite is green, a suite that cannot fail, or a performance
regression. This executor produces the missing evidence and keeps it in a machine
checkable report, separate from the review's advisory judgement.

The performance stage is the reason this exists. FlyDSL's PR benchmark step calls
``scripts/compare_benchmark.py``, which prints ratios and returns 0 unconditionally, so
no performance regression can turn CI red. Nothing else in the pipeline gates on speed.

Stages: merge_sim, gpu_claim, runtime_compat, test_policy, correctness, perf, diff_scan.
A stage that could not run reports ``skip`` with a reason; it never reports ``pass`` for
work it did not do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1

# Editing any of these means the JIT disk cache key does not move with the change, so a
# warm cache would serve the previous kernel. CLAUDE.md: "Disable it when debugging stale
# artifacts, changing C++ passes, or changing helper code that is not part of the traced
# closure."
COLD_CACHE_PATH_PREFIXES = ("lib/", "include/", "python/flydsl/", "kernels/common/", "tools/")


# --------------------------------------------------------------------------------------
# Report primitives
# --------------------------------------------------------------------------------------


@dataclass
class Stage:
    status: str = "skip"  # pass | fail | skip | info
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, **({"detail": self.detail} if self.detail else {})}


@dataclass
class Finding:
    severity: str  # blocker | should-fix | note
    stage: str
    detail: str


class Report:
    def __init__(self, label: str):
        self.label = label
        self.stages: dict[str, Stage] = {
            k: Stage(reason="not reached")
            for k in (
                "merge_sim",
                "gpu_claim",
                "runtime_compat",
                "test_policy",
                "correctness",
                "perf",
                "diff_scan",
            )
        }
        self.findings: list[Finding] = []
        self.repo: dict = {}
        self.environment: dict = {}
        self.selection: dict = {}
        self.degraded_mode: str | None = None

    def finding(self, severity: str, stage: str, detail: str) -> None:
        self.findings.append(Finding(severity, stage, detail))

    def verdict(self) -> str:
        sev = {f.severity for f in self.findings}
        if "blocker" in sev:
            return "BLOCK"
        required = ("merge_sim", "gpu_claim", "runtime_compat", "test_policy", "correctness", "perf")
        complete = all(self.stages[k].status == "pass" for k in required) and self.stages["diff_scan"].status in {
            "info",
            "pass",
        }
        if "should-fix" in sev:
            return "NEEDS_WORK"
        return "PASS" if complete else "INCONCLUSIVE"

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "label": self.label,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repo": self.repo,
            "environment": self.environment,
            "test_selection": self.selection,
            "degraded_mode": self.degraded_mode,
            "stages": {k: v.as_dict() for k, v in self.stages.items()},
            "findings": [{"severity": f.severity, "stage": f.stage, "detail": f.detail} for f in self.findings],
            "verdict": self.verdict(),
        }


# --------------------------------------------------------------------------------------
# Metric parsing (pure)
# --------------------------------------------------------------------------------------


def parse_metrics(text: str, fmt: str = "flydsl-table", regex: str | None = None) -> dict[str, float]:
    """Extract labelled metric values from benchmark stdout.

    Labelled rows are what keeps this honest. FlyDSL PR #654 shipped a parser that looped
    ``for m in re.finditer(...): pass`` and kept the LAST match, so layernorm was reported
    at 1.69 TB/s for months against a real 5.6. A dict keyed by an explicit row label
    cannot silently collapse several measurements into one.
    """
    out: dict[str, float] = {}
    if fmt == "flydsl-table":
        # run_benchmark.sh emits: op shape dtype tbps tflops
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            op, shape, dtype, tbps, tflops = parts
            if op == "op" or set(op) == {"-"}:
                continue
            value = None
            for candidate in (tflops, tbps):
                try:
                    value = float(candidate)
                    break
                except ValueError:
                    continue
            if value is None:
                continue
            out[f"{op}|{shape}|{dtype}"] = value
        return out

    if fmt == "regex":
        if not regex:
            raise ValueError("--metric-regex is required when --metric-format=regex")
        rx = re.compile(regex)
        for i, m in enumerate(rx.finditer(text)):
            gd = m.groupdict()
            label = gd.get("label") or f"match{i}"
            raw = gd.get("value") or (m.group(1) if m.re.groups else None)
            if raw is None:
                continue
            try:
                out[label] = float(raw)
            except ValueError:
                continue
        return out

    raise ValueError(f"unknown metric format: {fmt}")


# --------------------------------------------------------------------------------------
# Performance analysis (pure, and therefore testable without a GPU)
# --------------------------------------------------------------------------------------


def analyze_perf(
    rounds: list[dict[str, dict[str, float]]],
    min_effect: float = 0.03,
    lower_is_better: bool = False,
    safety: float = 1.0,
) -> dict:
    """Compare head against base using an A/B/A design with a measured noise floor.

    Each round runs base, then head, then base again. Two properties follow:

    * Sandwiching head between two base runs cancels monotonic drift (clocks warming,
      another tenant ramping), which a run-all-base-then-all-head design would attribute
      entirely to the patch.
    * The two base runs form a genuine A/A control. Their disagreement is this machine's
      noise floor for this row, measured now, rather than a threshold guessed in advance.
      A head-vs-base delta smaller than the control's own disagreement is not evidence.

    ``run_benchmark.sh`` already documents why this matters: one softmax-backward tier
    "at M=64 fills 0.25 workgroups per CU on a 256-CU gfx950 and swings 24% run to run".
    A fixed 5% threshold would call that a regression on every other run.
    """
    labels: set[str] = set()
    for r in rounds:
        for side in ("base_a", "head", "base_b"):
            labels.update(r.get(side, {}).keys())

    rows = []
    for label in sorted(labels):
        a = [r["base_a"][label] for r in rounds if label in r.get("base_a", {})]
        b = [r["base_b"][label] for r in rounds if label in r.get("base_b", {})]
        h = [r["head"][label] for r in rounds if label in r.get("head", {})]
        if not a or not b or not h:
            rows.append(
                {
                    "label": label,
                    "status": "incomparable",
                    "reason": "row missing from one side",
                    "base_samples": len(a) + len(b),
                    "head_samples": len(h),
                }
            )
            continue

        base_all = a + b
        base_med = statistics.median(base_all)
        head_med = statistics.median(h)
        if base_med <= 0:
            rows.append({"label": label, "status": "incomparable", "reason": "non-positive base median"})
            continue

        # A/A control: how far apart are two base measurements of the same code?
        control_ratio = statistics.median(b) / statistics.median(a)
        control_dev = abs(control_ratio - 1.0)
        # Dispersion of the pooled base samples, as a second noise estimate.
        base_spread = (max(base_all) - min(base_all)) / base_med

        ratio = head_med / base_med
        # Normalise so that >1 always means "head is better".
        gain = (1.0 / ratio) if lower_is_better else ratio
        threshold = max(min_effect, control_dev * safety, base_spread * safety)

        if gain < 1.0 - threshold:
            status = "regression"
        elif gain > 1.0 + threshold:
            status = "improvement"
        else:
            status = "unchanged"

        rows.append(
            {
                "label": label,
                "status": status,
                "base_median": base_med,
                "head_median": head_med,
                "gain": gain,
                "change_pct": (gain - 1.0) * 100.0,
                "noise_floor": threshold,
                "control_deviation": control_dev,
                "base_spread": base_spread,
                "base_samples": len(base_all),
                "head_samples": len(h),
            }
        )

    regressions = [r for r in rows if r.get("status") == "regression"]
    return {
        "rows": rows,
        "regressions": regressions,
        "improvements": [r for r in rows if r.get("status") == "improvement"],
        "incomparable": [r for r in rows if r.get("status") == "incomparable"],
        "min_effect": min_effect,
        "lower_is_better": lower_is_better,
        "rounds": len(rounds),
        "design": "A/B/A interleaved, noise floor from the A/A control",
    }


# --------------------------------------------------------------------------------------
# Shell helpers
# --------------------------------------------------------------------------------------


def run(cmd: list[str] | str, cwd: Path | None = None, env: dict | None = None, timeout: int = 3600):
    shell = isinstance(cmd, str)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=shell,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def git(repo: Path, *args: str) -> str:
    r = run(["git", *args], cwd=repo)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def side_env(base_env: dict, side_dir: Path, cache_dir: Path, hip_index: str | None, cold_cache: bool) -> dict:
    """Per-side environment. Cache isolation is not optional.

    If both sides shared FLYDSL_RUNTIME_CACHE_DIR, the base run could populate an artifact
    that the head run then loads, and the comparison would measure the same kernel twice.
    """
    env = dict(base_env)
    pkg = side_dir / "build-fly" / "python_packages"
    parts = [str(pkg), str(side_dir)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    mlir_libs = pkg / "flydsl" / "_mlir" / "_mlir_libs"
    if mlir_libs.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(mlir_libs), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache_dir)
    env.setdefault("FLYDSL_AUTOTUNE", "0")  # scripts/run_tests.sh does this for determinism
    if hip_index is not None:
        env["HIP_VISIBLE_DEVICES"] = hip_index
    if cold_cache:
        env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    return env


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


def patch_paths(patch_text: str) -> list[str]:
    return sorted({re.sub(r"^b/", "", m) for m in re.findall(r"^\+\+\+ (?!/dev/null)(\S+)", patch_text, re.M)})


def needs_cold_cache(paths: list[str]) -> list[str]:
    return [p for p in paths if p.startswith(COLD_CACHE_PATH_PREFIXES)]


def stage_merge_sim(rep: Report, base_dir: Path, head_dir: Path, patch: Path | None) -> bool:
    st = rep.stages["merge_sim"]
    if patch is None:
        st.status = "skip"
        st.reason = "no patch supplied; base attribution and mergeability cannot be proven"
        return True
    text = patch.read_text(errors="replace")
    rep.repo["patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    rep.repo["patch_paths"] = patch_paths(text)

    dirty = run(["git", "status", "--porcelain"], cwd=head_dir).stdout.strip()
    if dirty:
        st.status = "fail"
        st.reason = "head worktree is not clean before applying the patch"
        rep.finding("blocker", "merge_sim", f"worktree dirty: {dirty.splitlines()[:3]}")
        return False

    check = run(["git", "apply", "--check", "-p1", str(patch)], cwd=head_dir)
    if check.returncode != 0:
        st.status = "fail"
        st.reason = "patch does not apply to the base commit"
        rep.finding(
            "blocker",
            "merge_sim",
            f"patch conflicts with base; no downstream number would describe the merged code: {check.stderr.strip()[:400]}",
        )
        return False

    applied = run(["git", "apply", "-p1", str(patch)], cwd=head_dir)
    if applied.returncode != 0:
        st.status = "fail"
        st.reason = "patch check passed but apply failed"
        rep.finding("blocker", "merge_sim", applied.stderr.strip()[:400])
        return False

    st.status = "pass"
    st.reason = "patch applies cleanly to the base commit"
    st.detail = {"changed_paths": rep.repo["patch_paths"][:50]}
    return True


def stage_gpu_claim(rep: Report, samples: int, interval: float) -> str | None:
    st = rep.stages["gpu_claim"]
    picker = SKILL_DIR / "pick_idle_gpu.py"
    r = run([sys.executable, str(picker), "--samples", str(samples), "--interval", str(interval), "--json"])
    if r.returncode != 0:
        st.status = "skip"
        st.reason = f"no idle GPU claimable: {r.stderr.strip()[:200]}"
        rep.degraded_mode = "NO_GPU"
        return None
    try:
        rec = json.loads(r.stdout)
    except json.JSONDecodeError:
        st.status = "skip"
        st.reason = "picker output was not JSON"
        rep.degraded_mode = "NO_GPU"
        return None

    arch = ""
    ri = run(["rocminfo"])
    if ri.returncode == 0:
        m = re.search(r"Name:\s+(gfx\w+)", ri.stdout)
        arch = m.group(1) if m else ""
    st.status = "pass"
    st.reason = "device idle across the whole sampling window"
    st.detail = {**rec, "arch": arch, "host": os.uname().nodename}
    rep.environment["arch"] = arch
    rep.environment["gpu"] = rec
    return str(rec["smi_index"])


def stage_runtime_compat(rep: Report, base_dir: Path, head_dir: Path) -> bool:
    """Does the checkout's own flydsl import, and does it shadow or get shadowed?

    A prebuilt package that drifts behind the tree raises ImportError that looks exactly
    like a defect in the PR. That is an environment fact and must not be attributed to
    the author, so this returns INCONCLUSIVE rather than a correctness failure.
    """
    st = rep.stages["runtime_compat"]
    details = {}
    for name, d in (("base", base_dir), ("head", head_dir)):
        src = d / "python" / "flydsl" / "__init__.py"
        pkg = d / "build-fly" / "python_packages" / "flydsl" / "__init__.py"

        def ver(p: Path) -> str | None:
            if not p.is_file():
                return None
            m = re.search(r'__version__\s*=\s*"([^"]+)"', p.read_text(errors="replace"))
            return m.group(1) if m else None

        src_v, pkg_v = ver(src), ver(pkg)
        env = side_env(os.environ.copy(), d, Path("/tmp/flydsl-validate-probe") / name, None, False)
        probe = run(
            [
                sys.executable,
                "-c",
                "import flydsl, flydsl.compiler as flyc, flydsl.expr as fx;"
                "from flydsl.runtime.device import get_rocm_arch;"
                "print(flydsl.__version__, flydsl.__file__)",
            ],
            cwd=d,
            env=env,
            timeout=300,
        )
        details[name] = {
            "checkout_version": src_v,
            "prebuilt_version": pkg_v,
            "import_ok": probe.returncode == 0,
            "import_output": (probe.stdout or probe.stderr).strip()[-400:],
        }

    for name, d in details.items():
        if not d["import_ok"]:
            st.status = "skip"
            st.reason = f"{name} checkout cannot import flydsl against the installed runtime"
            rep.finding(
                "note",
                "runtime_compat",
                f"{name}: import failed -- environment fact, not attributable to the PR: {d['import_output'][:200]}",
            )
            st.detail = details
            return False
        if d["checkout_version"] and d["prebuilt_version"] and d["checkout_version"] != d["prebuilt_version"]:
            st.status = "skip"
            st.reason = (
                f"{name} prebuilt runtime {d['prebuilt_version']} is behind the checkout "
                f"{d['checkout_version']}; it would shadow the tree under test"
            )
            rep.finding(
                "note",
                "runtime_compat",
                f"{name}: prebuilt {d['prebuilt_version']} != checkout {d['checkout_version']} -- rebuild before trusting any result",
            )
            st.detail = details
            return False

    st.status = "pass"
    st.reason = "both checkouts import flydsl and match their prebuilt runtime"
    st.detail = details
    return True


TOL_RX = re.compile(r"\b(atol|rtol|tol|_RTOL|eps)\b\s*[=:]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def stage_test_policy(rep: Report, patch_text: str | None) -> bool:
    """A suite that cannot fail is worse than no suite, because it reports green."""
    st = rep.stages["test_policy"]
    if patch_text is None:
        st.status = "skip"
        st.reason = "no patch supplied; head-vs-base test policy cannot be compared"
        return False

    widened, disabled = [], []
    path = None
    removed: dict[str, list[tuple[str, float]]] = {}
    added: dict[str, list[tuple[str, float]]] = {}
    for raw in patch_text.splitlines():
        if raw.startswith("+++ "):
            path = re.sub(r"^b/", "", raw[4:].strip())
            continue
        if not path or raw.startswith(("+++", "---", "@@")):
            continue
        if raw.startswith(("+", "-")):
            body = raw[1:]
            for m in TOL_RX.finditer(body):
                (added if raw[0] == "+" else removed).setdefault(path, []).append((m.group(1), float(m.group(2))))
            if raw[0] == "+" and re.match(r"^\s*#\s*\(?\s*\d", body):
                disabled.append(f"{path}: {body.strip()[:100]}")

    for p, adds in added.items():
        for name, new_val in adds:
            olds = [v for n, v in removed.get(p, []) if n == name]
            if olds and new_val > max(olds):
                widened.append(f"{p}: {name} {max(olds)} -> {new_val}")

    code_changed = any(
        pp.endswith((".py", ".cpp", ".h", ".td")) and not pp.startswith("tests/")
        for pp in (rep.repo.get("patch_paths") or [])
    )
    st.detail = {"tolerances_widened": widened, "rows_disabled": disabled, "kernel_code_also_changed": code_changed}

    if widened and not code_changed:
        st.status = "fail"
        st.reason = "test-only tolerance widening"
        rep.finding(
            "blocker",
            "test_policy",
            f"tolerance widened with no kernel change to justify it: {widened}",
        )
        return False
    if widened:
        st.status = "pass"
        st.reason = "tolerance widened alongside a kernel change; needs numerical justification"
        rep.finding("should-fix", "test_policy", f"tolerance widened: {widened} -- justify numerically")
        return True
    if disabled:
        st.status = "pass"
        st.reason = "shape rows newly disabled"
        rep.finding("should-fix", "test_policy", f"shape rows disabled by this change: {disabled[:5]}")
        return True

    st.status = "pass"
    st.reason = "no tolerance widening and no newly disabled shape rows"
    return True


def detect_runner(target: str, root: Path) -> tuple[str, str]:
    if "::" in target:
        return "pytest", "caller named an explicit pytest node id"
    f = root / target
    if not f.is_file():
        return "none", f"target not present: {target}"
    text = f.read_text(errors="replace")
    if re.search(r"^\s*(?:def\s+test\w*|class\s+Test\w*)", text, re.M):
        return "pytest", "target defines test functions or classes"
    if "__main__" in text:
        return "script", "no test function, but the file has a __main__ entry point"
    return "none", "no runnable entry point"


def stage_correctness(
    rep: Report, base_dir: Path, head_dir: Path, target: str, env_base: dict, env_head: dict, timeout: int
) -> bool:
    st = rep.stages["correctness"]
    if not target:
        st.status = "skip"
        st.reason = "no test target supplied"
        return False

    runner, why = detect_runner(target, head_dir)
    rep.selection["test_target"] = target
    rep.selection["runner"] = runner
    rep.selection["runner_reason"] = why
    if runner == "none":
        st.status = "skip"
        st.reason = why
        rep.finding(
            "note", "correctness", f"{target} has no runnable entry point -- a packaging fact, not a kernel defect"
        )
        return False

    def cmd(root: Path) -> list[str]:
        if runner == "pytest":
            return [sys.executable, "-m", "pytest", target, "-q", "--no-header"]
        return [sys.executable, target]

    results = {}
    for name, root, env in (("base", base_dir, env_base), ("head", head_dir, env_head)):
        try:
            r = run(cmd(root), cwd=root, env=env, timeout=timeout)
            results[name] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}
        except subprocess.TimeoutExpired:
            results[name] = {"rc": -1, "tail": f"timed out after {timeout}s"}
    st.detail = results

    base_ok = results["base"]["rc"] == 0
    head_ok = results["head"]["rc"] == 0
    if head_ok:
        st.status = "pass"
        st.reason = "head passes the selected target"
        if not base_ok:
            st.reason += "; base fails it (pre-existing or PR-added test)"
        return True
    if not base_ok:
        st.status = "skip"
        st.reason = "target fails on base too -- pre-existing failure, not attributable to the PR"
        rep.finding("note", "correctness", "the selected target already fails on base; pick a target base passes")
        return False
    st.status = "fail"
    st.reason = "head fails a target that base passes"
    rep.finding("blocker", "correctness", f"regression on {target}: {results['head']['tail'][-500:]}")
    return False


def stage_perf(
    rep: Report,
    base_dir: Path,
    head_dir: Path,
    bench_cmd: str,
    env_base: dict,
    env_head: dict,
    rounds: int,
    timeout: int,
    metric_format: str,
    metric_regex: str | None,
    min_effect: float,
    lower_is_better: bool,
) -> bool:
    st = rep.stages["perf"]
    if not bench_cmd:
        st.status = "skip"
        st.reason = "no benchmark command supplied"
        return False

    rep.selection["bench_cmd"] = bench_cmd
    rep.selection["perf_rounds"] = rounds
    rep.selection["metric_format"] = metric_format

    samples: list[dict[str, dict[str, float]]] = []
    failures = []
    for i in range(rounds):
        rnd: dict[str, dict[str, float]] = {}
        for side, root, env in (
            ("base_a", base_dir, env_base),
            ("head", head_dir, env_head),
            ("base_b", base_dir, env_base),
        ):
            try:
                r = run(bench_cmd, cwd=root, env=env, timeout=timeout)
            except subprocess.TimeoutExpired:
                failures.append(f"round {i} {side}: timeout")
                rnd[side] = {}
                continue
            if r.returncode != 0:
                failures.append(f"round {i} {side}: rc={r.returncode} {(r.stderr or '')[-200:]}")
            rnd[side] = parse_metrics(r.stdout, metric_format, metric_regex)
        samples.append(rnd)

    analysis = analyze_perf(samples, min_effect=min_effect, lower_is_better=lower_is_better)
    st.detail = {**analysis, "run_failures": failures[:10]}

    if not any(r.get("status") in {"regression", "improvement", "unchanged"} for r in analysis["rows"]):
        st.status = "skip"
        st.reason = "no benchmark row was measurable on both sides"
        rep.finding("note", "perf", f"benchmark produced no comparable rows; failures: {failures[:3]}")
        return False

    if analysis["regressions"]:
        worst = min(analysis["regressions"], key=lambda r: r["gain"])
        st.status = "fail"
        st.reason = f"{len(analysis['regressions'])} row(s) regressed beyond the measured noise floor"
        for r in analysis["regressions"]:
            rep.finding(
                "blocker",
                "perf",
                (
                    f"{r['label']}: {r['change_pct']:+.1f}% "
                    f"({r['base_median']:.3f} -> {r['head_median']:.3f}), "
                    f"noise floor {r['noise_floor'] * 100:.1f}% from an A/A control of "
                    f"{r['control_deviation'] * 100:.1f}%"
                ),
            )
        st.detail["worst"] = worst
        return False

    st.status = "pass"
    st.reason = f"no row regressed beyond the measured noise floor across {rounds} A/B/A rounds"
    return True


def stage_diff_scan(rep: Report, patch: Path | None) -> None:
    st = rep.stages["diff_scan"]
    scanner = SKILL_DIR.parent / "review-pr" / "scan_flydsl_diff.py"
    if patch is None or not scanner.is_file():
        st.status = "skip"
        st.reason = "no patch or scanner unavailable"
        return
    r = run([sys.executable, str(scanner), "--diff", str(patch)])
    counts = dict(re.findall(r"\[(\w+)\] (\d+) candidate", r.stdout))
    st.status = "info"
    st.reason = "static candidates for the reviewer; not verdicts"
    st.detail = {"candidates": {k: int(v) for k, v in counts.items()}, "output": r.stdout[-4000:]}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path, help="clean checkout at the PR base commit")
    ap.add_argument("--patch", type=Path, help="base-to-head patch")
    ap.add_argument("--head-sha", help="exact remote PR head the patch represents")
    ap.add_argument("--head-dir", type=Path, help="reuse an existing head worktree instead of creating one")
    ap.add_argument("--tests", default="", help="correctness target, e.g. tests/kernels/test_softmax.py")
    ap.add_argument("--bench-cmd", default="", help="benchmark command run in each side's worktree")
    ap.add_argument("--metric-format", default="flydsl-table", choices=["flydsl-table", "regex"])
    ap.add_argument("--metric-regex", help="regex with named groups (?P<label>...) and (?P<value>...)")
    ap.add_argument("--perf-rounds", type=int, default=5, help="A/B/A rounds; each round runs base, head, base")
    ap.add_argument("--min-effect", type=float, default=0.03, help="smallest change treated as real, before noise")
    ap.add_argument("--lower-is-better", action="store_true", help="metric is latency, not throughput")
    ap.add_argument("--gpu-samples", type=int, default=10)
    ap.add_argument("--gpu-interval", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", default="flydsl-validate")
    ap.add_argument("--out", type=Path, default=Path("validation_report.json"))
    args = ap.parse_args()

    rep = Report(args.label)
    base_dir = args.repo.resolve()
    rep.repo["base_dir"] = str(base_dir)
    try:
        rep.repo["base"] = git(base_dir, "rev-parse", "HEAD")
    except Exception as exc:
        print(f"cannot read base commit: {exc}", file=sys.stderr)
        return 2
    rep.repo["head"] = args.head_sha
    rep.environment["container"] = False
    rep.environment["isolation"] = "git-worktree + private JIT caches"
    rep.environment["python"] = sys.version.split()[0]

    # Head worktree
    if args.head_dir:
        head_dir = args.head_dir.resolve()
        created = False
    else:
        head_dir = Path(f"/tmp/flydsl-head-{rep.repo['base'][:10]}-{os.getpid()}")
        git(base_dir, "worktree", "add", "--detach", str(head_dir), rep.repo["base"])
        created = True
    rep.repo["head_dir"] = str(head_dir)

    patch_text = args.patch.read_text(errors="replace") if args.patch else None

    try:
        ok = stage_merge_sim(rep, base_dir, head_dir, args.patch)
        if ok:
            hip = stage_gpu_claim(rep, args.gpu_samples, args.gpu_interval)
            stage_test_policy(rep, patch_text)

            cold_paths = needs_cold_cache(rep.repo.get("patch_paths") or [])
            cold = bool(cold_paths)
            rep.environment["cold_cache_required"] = cold
            rep.environment["cold_cache_reason"] = (
                f"patch touches {cold_paths[:5]}; the JIT key does not move with these, "
                "so a warm cache would serve the previous kernel"
                if cold
                else "patch is confined to traced-closure sources; the JIT key moves with it"
            )

            env_base = side_env(os.environ.copy(), base_dir, Path("/tmp/flydsl-cache-base"), hip, cold)
            env_head = side_env(os.environ.copy(), head_dir, Path("/tmp/flydsl-cache-head"), hip, cold)

            if stage_runtime_compat(rep, base_dir, head_dir) and hip is not None:
                stage_correctness(rep, base_dir, head_dir, args.tests, env_base, env_head, args.timeout)
                stage_perf(
                    rep,
                    base_dir,
                    head_dir,
                    args.bench_cmd,
                    env_base,
                    env_head,
                    args.perf_rounds,
                    args.timeout,
                    args.metric_format,
                    args.metric_regex,
                    args.min_effect,
                    args.lower_is_better,
                )
            else:
                for k in ("correctness", "perf"):
                    rep.stages[k].status = "skip"
                    rep.stages[k].reason = (
                        "runtime_compat did not pass" if hip is not None else "no GPU claimed (degraded mode)"
                    )
        stage_diff_scan(rep, args.patch)
    finally:
        if created:
            run(["git", "worktree", "remove", "--force", str(head_dir)], cwd=base_dir)

    args.out.write_text(json.dumps(rep.as_dict(), indent=2) + "\n")
    d = rep.as_dict()
    print(f"verdict: {d['verdict']}")
    for k, v in d["stages"].items():
        print(f"  {v['status']:<5} {k:<16} {v['reason']}")
    for f in d["findings"]:
        print(f"  [{f['severity']}] {f['stage']}: {f['detail']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
