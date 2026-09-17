#!/usr/bin/env python3
"""Extract AMDGPU function attributes from dumped FlyDSL LLVM IR.

This answers "did my knob actually reach LLVM?" without a GPU, a profiler, or a
timing run. It reads the `NN_llvm_ir.ll` artifact that FLYDSL_DUMP_IR=1 already
writes and reports the attributes that were attached to each kernel.

Artifacts are resolved by GLOB SUFFIX, never by stage index: stage numbers are
positional, and FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 inserts a stage that shifts
everything after it.

Usage:
    llvm_ir_attrs.py <dump-dir> [<dump-dir>...]   # tabulate
    llvm_ir_attrs.py <before> <after> --diff      # compare two runs

Produce a dump with:
    FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_DIR=<dir> python k.py
(FLYDSL_RUNTIME_ENABLE_CACHE=0 is mandatory: a cache hit writes no dumps at all.)

Exit codes: 0 attributes found, 1 no IR artifacts or none carried attributes,
2 usage error.
"""

import sys

# Parsed before anything runs, so this guard only protects against *runtime* use
# of newer syntax; PEP 585/604 annotations below are never evaluated on an old
# interpreter. The host python is 3.6 -- run this inside the dev container.
if sys.version_info < (3, 10):  # pragma: no cover - exercised via subprocess
    sys.stderr.write(
        ".claude/skills/llvm/scripts/llvm_ir_attrs.py requires Python 3.10+ (running %s).\n"
        "This repository targets 3.10+ (CONTRIBUTING.md; ruff target-version=py310).\n"
        "The host interpreter is too old; run it in the dev container:\n"
        "  docker exec <dev-container> python3 .claude/skills/llvm/scripts/llvm_ir_attrs.py ...\n"
        % sys.version.split()[0]
    )
    raise SystemExit(2)

import argparse
import re
from pathlib import Path

# The AMDGPU attributes worth reporting for tuning. Anything else in the
# attribute group is noise for this purpose.
INTERESTING = (
    "amdgpu-waves-per-eu",
    "amdgpu-flat-work-group-size",
    "amdgpu-num-vgpr",
    "amdgpu-num-sgpr",
    "amdgpu-agpr-alloc",
    "amdgpu-max-num-workgroups",
    "amdgpu-dynamic-vgpr-block-size",
    "amdgpu-memory-bound",
    "amdgpu-wave-limiter",
    "amdgpu-expert-scheduling-mode",
    "amdgpu-sched-strategy",
    "amdgpu-cluster-dims",
    "amdgpu-ieee",
    # Reported, but inert on this pin -- see references/amdgpu-attributes.md.
    # Their presence is NOT evidence the denorm mode changed.
    "denormal-fp-math-f32",
    "denormal-fp-math",
    "denormal_fpenv",
    "unsafe-fp-math",
    "no-nans-fp-math",
    "no-infs-fp-math",
    "amdgpu-unsafe-fp-atomics",
)

ATTR_GROUP_RE = re.compile(r"^attributes\s+#(\d+)\s*=\s*\{(.*)\}\s*$", re.MULTILINE)
KERNEL_RE = re.compile(
    r"^define\s+(?:protected\s+|hidden\s+|dllexport\s+)*" r"amdgpu_kernel\s+void\s+@([\w.$]+)\s*\(.*?\)\s*(.*?)\{",
    re.MULTILINE | re.DOTALL,
)


def find_ir_files(root: str | Path) -> list[Path]:
    """Every *_llvm_ir.ll under root, resolved by suffix not by stage number."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.glob("**/*llvm_ir.ll"))


def parse_attr_groups(text: str) -> dict[str, dict[str, str]]:
    """{group-id: {attr: value}} for every `attributes #N = { ... }` line."""
    groups = {}
    for gid, body in ATTR_GROUP_RE.findall(text):
        attrs = {}
        for key, val in re.findall(r'"([\w.-]+)"(?:="([^"]*)")?', body):
            attrs[key] = val
        # Structured attributes are unquoted and carry their arguments inline,
        # e.g. `denormal_fpenv(float: preservesign)`. Their *arguments* are the
        # payload -- reporting only the name would hide a mode change.
        for key, args in re.findall(r"(?<![\w\"])([a-z_][\w]*)\(([^()]*)\)", body):
            attrs[key] = args.strip()
        # Bare (non-quoted) flags such as `nounwind` are ignored on purpose.
        groups[gid] = attrs
    return groups


def kernels_with_attrs(text: str) -> list[tuple[str, dict[str, str]]]:
    """[(kernel_name, {attr: value})] for each amdgpu_kernel in the module."""
    groups = parse_attr_groups(text)
    out = []
    for name, tail in KERNEL_RE.findall(text):
        attrs = {}
        # `#N` is not always last: debug IR emits `define ... #0 !dbg !3 {`.
        m = re.search(r"#(\d+)\b(?![\w.])", tail)
        if m:
            attrs = groups.get(m.group(1), {})
        out.append((name, attrs))
    return out


def collect(dump_dir: str | Path) -> dict[str, dict[str, str]]:
    """{kernel: {attr: value}} across every IR artifact in dump_dir."""
    found = {}
    for path in find_ir_files(dump_dir):
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for name, attrs in kernels_with_attrs(text):
            keep = {k: v for k, v in attrs.items() if k in INTERESTING}
            if name in found:
                found[name].update(keep)
            else:
                found[name] = keep
    return found


def report(dump_dir: str | Path) -> int:
    files = find_ir_files(dump_dir)
    if not files:
        print(
            f"no *_llvm_ir.ll under {dump_dir}\n"
            "  Did the run actually compile? A JIT cache hit writes NO dumps.\n"
            "  Re-run with: FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0",
            file=sys.stderr,
        )
        return 1

    found = collect(dump_dir)
    print(f"# {dump_dir}  ({len(files)} IR artifact(s))")
    if not found:
        print("  no amdgpu_kernel definitions found")
        return 1

    any_attr = False
    for kernel in sorted(found):
        attrs = found[kernel]
        print(f"\n  {kernel}")
        if not attrs:
            print("    (no tuning attributes -- the compiler chose everything)")
            continue
        any_attr = True
        for key in sorted(attrs):
            val = attrs[key]
            print(f"    {key:<34} {val if val else '(present)'}")
    return 0 if any_attr else 1


def diff(before: str | Path, after: str | Path) -> int:
    # Check both sides first: an absent dump would otherwise surface as
    # `'4' -> None`, i.e. exactly the false positive this tool exists to catch.
    for side, d in (("before", before), ("after", after)):
        if not find_ir_files(d):
            print(
                f"no *_llvm_ir.ll under the {side} dump ({d})\n"
                "  A JIT cache hit writes NO dumps. Re-run that side with:\n"
                "  FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0",
                file=sys.stderr,
            )
            return 1

    # Each side must independently yield kernels. A present-but-empty dump, or
    # one holding only helper functions, would otherwise surface as
    # `'4' -> None` -- the false positive this tool exists to catch.
    b, a = collect(before), collect(after)
    for side, parsed, d in (("before", b, before), ("after", a, after)):
        if not parsed:
            print(
                f"no amdgpu_kernel found in the {side} dump ({d})\n"
                "  The IR is present but holds no kernel -- a truncated dump, or\n"
                "  the wrong directory. Not comparing.",
                file=sys.stderr,
            )
            return 1

    names = sorted(set(b) | set(a))

    changed = False
    for name in names:
        ba, aa = b.get(name, {}), a.get(name, {})
        keys = sorted(set(ba) | set(aa))
        rows = [(k, ba.get(k), aa.get(k)) for k in keys if ba.get(k) != aa.get(k)]
        if not rows:
            continue
        changed = True
        print(f"\n  {name}")
        for key, bv, av in rows:
            print(f"    {key:<34} {bv!r:>16}  ->  {av!r}")

    if not changed:
        print("no attribute differences")
        print("  If you expected one: an unknown hint key is silently ignored,")
        print("  yet still changes the cache key -- so a real recompile proves nothing.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Show AMDGPU function attributes in dumped FlyDSL LLVM IR.")
    ap.add_argument("dirs", nargs="+", help="dump dir(s) from FLYDSL_DUMP_DIR")
    ap.add_argument("--diff", action="store_true", help="compare exactly two dumps")
    args = ap.parse_args()

    if args.diff:
        if len(args.dirs) != 2:
            ap.error("--diff needs exactly two dump dirs")
        return diff(args.dirs[0], args.dirs[1])

    worst = 1
    for d in args.dirs:
        worst = min(worst, report(d))
    return worst


if __name__ == "__main__":
    sys.exit(main())
