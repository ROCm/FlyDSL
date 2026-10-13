#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Deterministic half of the review-pr skill.

Step 1b uses it to decide which rules a diff can trigger and to collect the evidence those
rules need; Step 9 uses it to refuse a card whose steps left no trace. Whether a finding is
correct is not checked here: that is what the human reading the card is for.

  triage.py scan <diff>                         candidate sites per category
  triage.py rules <diff> [title] [repo-root]    derived families and rule ids
  triage.py expand <rules.txt> <rules.md>       text of exactly the derived rules
  triage.py mapping                             family -> rule table (MAPPING.md)
  triage.py core <diff> <repo-root>             backbone files the diff touches
  triage.py bounds <diff> <head-root>           buffer descriptors with no byte bound
  triage.py citest <diff> <repo-root> [head]    where each added test runs in CI
  triage.py benchcov <diff> <repo-root>         which changed kernels the CI benchmark reaches
  triage.py gates <work-dir> <repo-root>        the seven Step 9 gates
  triage.py apply-refutation <card> <independent>
"""

from __future__ import annotations

import ast
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES_MD = HERE / "rules.md"

# --------------------------------------------------------------------------- diff parsing

CODE_EXT = (".py", ".cpp", ".cc", ".h", ".hpp", ".td")


def parse_diff(diff_text: str) -> OrderedDict:
    """path -> {"add": [...], "del": [...], "new": bool, "deleted": bool, "renamed_from": str|None}."""
    files: OrderedDict = OrderedDict()
    cur = None
    for line in diff_text.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            cur = files.setdefault(
                m.group(2), {"add": [], "del": [], "new": False, "deleted": False, "renamed_from": None}
            )
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur["new"] = True
        elif line.startswith("deleted file mode"):
            cur["deleted"] = True
        elif line.startswith("rename from "):
            cur["renamed_from"] = line[len("rename from ") :].strip()
        elif line.startswith("+") and not line.startswith("+++"):
            cur["add"].append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            cur["del"].append(line[1:])
    return files


def added_lines(diff_text: str):
    """Yield (path, new_line_number, text) for every added line."""
    path, lineno = None, 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            path = None
            continue
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else re.sub(r"^b/", "", target)
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
            continue
        if path is None:
            continue
        if raw.startswith("+"):
            yield path, lineno, raw[1:]
            lineno += 1
        elif raw.startswith(("-", "\\")):
            continue
        else:
            lineno += 1


def deleted_lines(diff_text: str):
    """Yield (path, text) for every deleted line."""
    path = None
    for raw in diff_text.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/", raw)
        if m:
            path = m.group(1)
            continue
        if raw.startswith(("---", "+++")):
            continue
        if path and raw.startswith("-"):
            yield path, raw[1:]


def changed_paths(diff_text: str) -> set:
    return set(re.findall(r"^diff --git a/\S+ b/(\S+)", diff_text, re.M))


def _is_comment(text: str) -> bool:
    s = text.strip()
    return not s or s.startswith(("#", "//", "*", "/*"))


# --------------------------------------------------------------------------- scan

INDEX_SHAPED = re.compile(
    r"\b\w*(?:_id|_idx|_index|idx|_pos)\b"
    r"|\b(?:bid|pid|tid|wid|lane|phys|phys_row|block|blk|row|col|page|token|slot|step|seq|tile)\w*\b",
    re.I,
)
STRIDE_SHAPED = re.compile(
    r"\b\w*(?:stride|pitch|extent|numel|nelem|_dim|_size|_bytes|_width|_len|pages|_elems?)\w*\b"
    r"|\b(?:head_dim|n_kv|num_kv|hidden|dhw|npq|elem_bytes|page_bytes)\b",
    re.I,
)
WIDENED = re.compile(
    r"fx\.Int64|Int64\(|int64_t|torch\.int64|\.to\(\s*(?:tl\.)?int64|static_cast<int64_t>|\bi64\b"
    r"|IntegerType::get\w*\(\s*\w+,\s*64",
    re.I,
)
BUF_CALLS = ("make_buffer_tensor", "create_buffer_resource")


# key -> (rules, what a candidate is). The rule text itself lives in rules.md only.
CATEGORIES = (
    ("index_width", "D1 D3", "index x stride multiply with no 64-bit widening"),
    ("abi_flatten", "D2", "tensor flattened before a launch"),
    ("trunc_div", "E1", "truncating division feeding a tile, chunk or loop count"),
    ("cache_key", "A1 A3 A4", "cache-key surface touched, or a trait added next to a cache_tag"),
    ("arch_capability", "C1 C2 C4", "architecture capability derived inline, or an arch string literal"),
    ("dead_gate", "B3 B4", "always-false gate, or a default-off trait"),
    ("fastmath", "F1 F2", "fastmath literal, or an fx.* wrapper added or edited"),
    ("raw_mlir", "HK1", "raw MLIR builder in kernel code"),
    ("scf_control", "HK2", "explicit scf control flow in a kernel"),
    ("inline_asm", "HK3", "inline assembly"),
    ("smem_legacy", "HK4", "legacy SmemAllocator / SmemPtr"),
    ("stream", "H1 H4", "explicit stream or cross-stream launch"),
    ("skip", "T8 C5", "module-level skip or importorskip"),
    ("ref_dtype", "T4", "dtype conversion in a test reference"),
    ("host_copy", "H1 H2", "host-side copy or reshape near a launch"),
    ("aiter_call", "I1 I2", "call into or import from aiter"),
    ("block_id", "E5", "raw block-id read"),
    ("launch_geometry", "E4", "kernel launch geometry"),
    ("test_shape", "T1 T4 T5", "tolerance, finditer loop, or benchmark harness change"),
    ("cpp_attr", "D4 G3", "C++ attribute width or getValue() read"),
    ("buffer_bounds", "D3 D5", "buffer descriptor construction (see bounds.txt)"),
    ("copy_width", "E2", "copy atom vs register vector width"),
    ("rounding", "F5", "hand-written float conversion"),
    ("online_softmax", "F3 F4", "online-softmax or wave-uniform numerics"),
    ("guard_removed", "B6", "guard, zero-init or contiguity requirement deleted and not re-added"),
    ("env_var", "A1 HK11", "environment variable read outside tests"),
    ("stub", "HK12", "TODO / FIXME / NotImplementedError / pass in a new path"),
    ("config_table", "HK6", "tuned config table in a kernel file"),
    ("direct_compile", "HK7", "direct flyc.compile in a launch path"),
    ("syspath", "HK14", "sys.path mutation"),
    ("assert_guard", "H3", "assert guarding a kernel precondition"),
    ("codegen_knob", "P4", "compile hint, LLVM option or function attribute"),
)


@dataclass
class Category:
    key: str
    rules: str
    title: str
    hits: list = field(default_factory=list)


def build_categories() -> "OrderedDict[str, Category]":
    return OrderedDict((key, Category(key, rules, title)) for key, rules, title in CATEGORIES)


def _is_test(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path or Path(path).name.startswith("test_")


def _scanned(path: str) -> bool:
    # Skill prose quotes these patterns literally; scanning it would light up every category.
    return path.endswith(CODE_EXT) and not path.startswith(".claude/")


def _cache_tag_files(diff_text: str):
    """Files whose diff shows the cache_tag idiom, and those whose cache_tag hunk was edited."""
    using, edited = set(), set()
    path, hunk = None, []

    def flush():
        if path and any("cache_tag" in x for x in hunk):
            using.add(path)
            if any(x.startswith(("+", "-")) and not x.startswith(("+++", "---")) for x in hunk):
                edited.add(path)

    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            flush()
            hunk = []
            target = raw[4:].strip()
            path = None if target == "/dev/null" else re.sub(r"^b/", "", target)
        elif raw.startswith("@@"):
            flush()
            # git appends the enclosing function to the hunk header; "def cache_tag(self):" there
            # means the hunk lands inside the tuple even when no edited line says "cache_tag".
            hunk = [raw] if "cache_tag" in raw else []
        elif path:
            hunk.append(raw)
    flush()
    return using, edited


def scan(diff_text: str) -> "OrderedDict[str, Category]":
    cats = build_categories()
    tag_files, tag_edited = _cache_tag_files(diff_text)
    new_traits = []

    def add(key, path, lineno, text, note=""):
        cats[key].hits.append((path, lineno, text.strip(), note))

    for path, lineno, text in added_lines(diff_text):
        if not _scanned(path) or _is_comment(text):
            continue
        s = text.strip()
        py = path.endswith(".py")
        cpp = not py
        kernel = path.startswith("kernels/")
        test = _is_test(path)

        if "*" in text and "**" not in text and not WIDENED.search(text):
            if INDEX_SHAPED.search(text) and STRIDE_SHAPED.search(text) and re.search(r"[\w\]\)]\s*\*\s*[\w\(]", text):
                add("index_width", path, lineno, s)
        if py and re.search(r"\.(?:view|reshape)\(\s*-1\s*\)|\.flatten\(\)", text):
            add("abi_flatten", path, lineno, s)
        if (
            py
            and re.search(r"[\w\)\]]\s*//\s*[\w\(]", text)
            and re.search(
                r"\b\w*(?:load|chunk|vec|copy|tile|unit|bytes|elems?|iters?|steps?|rounds?|per_thread|per_row|_k|num_)\w*",
                text,
                re.I,
            )
        ):
            add("trunc_div", path, lineno, s)
        if "cache_tag" in text:
            add("cache_key", path, lineno, s, "cache_tag edited")
        if re.search(r"_CACHE_INVALIDATING_ENV_VARS|_flydsl_key|__cache_signature__|manager_key|TUNING_SCHEMA", text):
            add("cache_key", path, lineno, s, "cache-key machinery edited")
        if py and re.match(r"^\s*self\.[A-Z][A-Z0-9_]{2,}\s*[:=]", s):
            new_traits.append((path, lineno, s, "traits field added"))
        elif py and path in tag_files and re.match(r"^[A-Z][A-Z0-9_]{2,}\s*[:=]", s):
            new_traits.append((path, lineno, s, "constant added in a cache_tag file"))
        if py and re.search(r"is_rdna_arch|is_cdna|startswith\(\s*[\"']gfx|wave64|WAVE_SIZE\s*=|get_warp_size\(", text):
            if "def get_warp_size" not in text:
                add("arch_capability", path, lineno, s)
        if re.search(r"[\"']gfx\d{3,4}\w*[\"']", text) and not test:
            add("arch_capability", path, lineno, s, "arch string literal")
        if re.search(r"\bFalse\s+and\b|\band\s+False\b|if\s+False\s*:", text):
            add("dead_gate", path, lineno, s, "always-false gate")
        if py and re.match(r"^\s*(?:self\.)?[A-Z_]*(?:SWIZZLE|ENABLE|USE_|LAZY|OPT_)[A-Z0-9_]*\s*[:=].*\bFalse\b", s):
            add("dead_gate", path, lineno, s, "default-off trait")
        if re.search(r"fastmath\s*=\s*(?:True|False|[\"'])", text):
            add("fastmath", path, lineno, s, "non-enum fastmath value")
        if py and path.startswith("python/flydsl/expr/") and re.match(r"^\s*def\s+\w+", s):
            add("fastmath", path, lineno, s, "expr wrapper added or edited")
        if py and (kernel or path.startswith("examples/")):
            if re.search(
                r"\bir\.(?:IntegerType|F32Type|Type)\b|ArithValue|_to_raw|\b(?:arith|vector|memref|llvm)\.[a-z_]+\(",
                text,
            ):
                add("raw_mlir", path, lineno, s)
            if re.search(r"\bscf\.(?:IfOp|ForOp|WhileOp)\b|\b_if_then\(", text):
                add("scf_control", path, lineno, s)
            if re.search(r"\bflyc\.compile\(", text):
                add("direct_compile", path, lineno, s)
            if re.match(r"^\s*_?[A-Z][A-Z0-9_]*(?:CONFIG|CONFIGS|TABLE|TUNED)[A-Z0-9_]*\s*(?::[^=]+)?=\s*[\[{(]", s):
                add("config_table", path, lineno, s)
            if re.match(r"^\s*assert\b", s):
                add("assert_guard", path, lineno, s)
        if re.search(r"\binline_asm\b", text):
            add("inline_asm", path, lineno, s)
        if re.search(r"\bSmemAllocator\b|\bSmemPtr\b", text):
            add("smem_legacy", path, lineno, s)
        if py and re.search(r"torch\.cuda\.[Ss]tream\(|wait_stream|from_c_void_p|record_event", text):
            add("stream", path, lineno, s)
        if py and re.search(r"importorskip\(|allow_module_level|^pytestmark\b.*skipif", s):
            add("skip", path, lineno, s)
        if (
            py
            and test
            and re.search(r"\.(?:float|half|double|bfloat16)\(\)|\.to\(\s*(?:torch\.)?(?:float|bfloat|half)", text)
        ):
            add("ref_dtype", path, lineno, s)
        if (
            py
            and not test
            and re.search(r"\.contiguous\(\)|torch\.cat\(|\.reshape\(|\.to\(\s*(?:device|torch\.)", text)
        ):
            add("host_copy", path, lineno, s)
        if py and re.search(r"\baiter[\w.]*\(|from\s+aiter\b|import\s+aiter\b", text):
            add("aiter_call", path, lineno, s)
        if re.search(r"gpu\.block_id\(|block_id\(\s*[\"']", text):
            add("block_id", path, lineno, s)
        if re.search(r"known_block_size|block_dim\s*=|BLOCK_THREADS\s*=|threads_per_block", text):
            add("launch_geometry", path, lineno, s)
        if test or "bench" in path:
            if re.search(r"\b(?:atol|rtol|tol|eps)\s*=\s*[0-9.eE-]+", text):
                add("test_shape", path, lineno, s, "tolerance")
            if re.search(r"for\s+\w+\s+in\s+re\.finditer", text):
                add("test_shape", path, lineno, s, "finditer loop: keeps the last match?")
        if cpp and re.search(r"getDynamic\(\s*\w+\s*\)|\.getValue\(\)", text):
            add("cpp_attr", path, lineno, s)
        if py and (any(c + "(" in text for c in BUF_CALLS) or re.search(r"\bmax_size\s*=|num_records", text)):
            add("buffer_bounds", path, lineno, s)
        if re.search(r"BufferCopy\d+b|make_rmem_tensor|\bvec_width\b|\bVEC_WIDTH\b", text):
            add("copy_width", path, lineno, s)
        if re.search(r"\+\s*0x8000\b|>>\s*16\b|cvt_pk_bf16|cvt_pk_", text):
            add("rounding", path, lineno, s)
        if re.search(r"\bballot\b|read_exec|\bexp2\(|\blse\b|rescale|m_row|row_max", text):
            add("online_softmax", path, lineno, s)
        if py and not test and re.search(r"os\.environ|os\.getenv\(|getenv\(", text):
            add("env_var", path, lineno, s)
        if re.search(r"#\s*(?:TODO|FIXME)\b|raise\s+NotImplementedError|^\s*pass\s*$", text):
            add("stub", path, lineno, s)
        if py and re.search(r"sys\.path\.(?:insert|append)\(", text):
            add("syspath", path, lineno, s)
        if re.search(r"compile_hints|llvm_options|waves_per_eu|maxnreg|\bdaz\b|denormal|passthrough", text):
            add("codegen_knob", path, lineno, s)

    # A guard that comes straight back elsewhere in the diff was moved, not removed.
    readded = {re.sub(r"\s+", "", t) for p, _, t in added_lines(diff_text) if _scanned(p)}
    for path, line in deleted_lines(diff_text):
        if not _scanned(path) or _is_comment(line) or re.sub(r"\s+", "", line) in readded:
            continue
        if re.search(
            r"^\s*assert\b|\braise\s+(?:ValueError|RuntimeError|NotImplementedError|TypeError)|torch\.zeros|"
            r"\.contiguous\(\)|max_size\s*=\s*False|num_records_bytes|\bmask\s*=",
            line,
        ):
            add("guard_removed", path, 0, line, "deleted, not re-added anywhere in the diff")

    for path, lineno, text, note in new_traits[:16]:
        if not path.startswith("kernels/"):
            continue
        if path in tag_edited:
            note = "cache_tag region was edited -- confirm THIS trait is in the tuple"
        else:
            note = "no cache_tag edit in this file's diff -- does this reach codegen?"
        add("cache_key", path, lineno, text, note)

    for f in changed_paths(diff_text):
        if f.endswith(("run_benchmark.sh", "compare_benchmark.py", "benchmark_output_to_csv.py")):
            add(
                "test_shape",
                f,
                0,
                "(benchmark harness changed)",
                "was the stored baseline produced by the same parser?",
            )
    return cats


def render_scan(cats, limit: int = 12) -> str:
    out = ["=== FlyDSL diff scan: candidates, not verdicts; each rule's text is in rules_expanded.txt ==="]
    total = sum(len(c.hits) for c in cats.values())
    if total == 0:
        out.append("no candidates in any category -- say so in the review; do not skip the categories silently")
        return "\n".join(out) + "\n"
    for cat in cats.values():
        if not cat.hits:
            continue
        out.append(f"\n[{cat.key}] {len(cat.hits)} candidate(s) -- {cat.title}  (rules: {cat.rules})")
        seen, shown = set(), 0
        for path, lineno, text, note in cat.hits:
            if (path, lineno, text) in seen:
                continue
            seen.add((path, lineno, text))
            if shown >= limit:
                out.append(f"  ... {len(cat.hits) - shown} more suppressed")
                break
            where = f"{path}:{lineno}" if lineno else path
            out.append(f"  {where}: {text[:140]}" + (f"  <- {note}" if note else ""))
            shown += 1
    empty = [c.key for c in cats.values() if not c.hits]
    out.append(f"\n0 candidates: {', '.join(empty)}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- tiers

TIER1 = (
    "python/flydsl/compiler/jit_function.py",
    "python/flydsl/compiler/kernel_function.py",
    "python/flydsl/compiler/ast_rewriter.py",
    "python/flydsl/compiler/backends/rocm.py",
    "python/flydsl/expr/__init__.py",
    "python/flydsl/expr/typing.py",
    "python/flydsl/expr/numeric.py",
    "python/flydsl/expr/primitive.py",
    "lib/Conversion/FlyToROCDL/FlyToROCDL.cpp",
    "lib/Dialect/Fly/Transforms/LayoutLowering.cpp",
    "lib/Dialect/Fly/IR/FlyOps.cpp",
    "lib/Dialect/Fly/Utils/IntTupleUtils.cpp",
    "lib/Dialect/Fly/Utils/NormalForm.cpp",
)
# Files whose defect changes WHICH artifact is served: silent by construction.
TIER1C = (
    "python/flydsl/compiler/jit_argument.py",
    "python/flydsl/compiler/protocol.py",
    "python/flydsl/compiler/_aot.py",
    "python/flydsl/autotune.py",
)
TIER2_PREFIXES = (
    "python/flydsl/runtime/device.py",
    "python/flydsl/expr/rocdl/",
    "lib/Dialect/FlyROCDL/",
    "tests/arch_compat.py",
)
# A kernels/ module this many files import is shared infrastructure, not one kernel. Counted from
# the tree so a helper joins when it becomes widely used: kernels/common/buffer_ops.py (33),
# kernels_common.py (24) and tensor_shim.py (22) are well above it; one-kernel modules sit at 1-3.
SHARED_FANIN = 7


def _module_of(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    mod = path[:-3].replace("/", ".")
    return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod


_IMPORT_INDEX: dict = {}


def import_index(root: Path) -> dict:
    """file -> set of modules it imports (`from a.b import c` records a.b and a.b.c)."""
    key = str(root)
    if key in _IMPORT_INDEX:
        return _IMPORT_INDEX[key]
    index = {}
    for sub in ("kernels", "tests", "examples", "python"):
        base = root / sub
        if not base.is_dir():
            continue
        for f in base.rglob("*.py"):
            rel = str(f.relative_to(root))
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError):
                continue
            own = (_module_of(rel) or "").split(".")
            package = own if rel.endswith("__init__.py") else own[:-1]
            mods = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parts = package[: len(package) - (node.level - 1)] + ([node.module] if node.module else [])
                        module = ".".join(parts)
                    else:
                        module = node.module or ""
                    if module:
                        mods.add(module)
                        mods.update(f"{module}.{a.name}" for a in node.names)
            index[rel] = mods
    _IMPORT_INDEX[key] = index
    return index


def fanin(root: Path, path: str) -> int:
    mod = _module_of(path)
    if not mod:
        return 0
    return sum(1 for f, mods in import_index(root).items() if f != path and mod in mods)


def tier_of(path: str, root: Path | None = None) -> str | None:
    if path in TIER1:
        return "TIER1"
    if path in TIER1C:
        return "TIER1C"
    if path.startswith(TIER2_PREFIXES):
        return "TIER2"
    if root is not None and path.startswith("kernels/") and fanin(root, path) >= SHARED_FANIN:
        return "TIER2"
    return None


def core_files(diff_text: str, root: Path | None = None) -> "OrderedDict[str, str]":
    out = OrderedDict()
    for path, f in parse_diff(diff_text).items():
        tier = tier_of(path, root)
        if tier:
            out[path] = tier
    return out


# --------------------------------------------------------------------------- derivation

# Family -> rules. `derive` emits a family by name, so this table is the whole mapping and
# MAPPING.md is generated from it rather than kept by hand.
PATH_FAMILIES = OrderedDict(
    [
        ("docs-only", ("NONE", "every path is docs, Markdown, images or agent metadata")),
        ("cache-surface", ("A1 A2 A3 A4 A6 STEP4", "a Tier 1c file or jit_function.py")),
        ("compiler-core", ("B5 G4 G5 I4 STEP4", "a Tier 1 compiler or lowering file")),
        (
            "arch-table",
            ("C1 C2 C3 C4 STEP4", "runtime/device.py, expr/rocdl/, lib/Dialect/FlyROCDL/, tests/arch_compat.py"),
        ),
        ("shared-kernel-helper", ("A3 B2 B3 STEP4", f"a kernels/ module imported by >= {SHARED_FANIN} files")),
        ("mlir-cpp", ("G1 G2 G3 G6 G7 D4", "lib/ or include/")),
        ("expr-surface", ("F1 F2 F6 G4 G5 S1", "python/flydsl/expr/")),
        ("public-api", ("S1 B5", "a package __init__ or compiler/protocol.py")),
        ("kernel", ("B1 B2 E1 E3 D1 T2 P5", "any kernels/ Python file")),
        ("new-kernel", ("T1 T6 HK5 HK8 HK13", "a new kernels/ Python file")),
        ("attention", ("F3 F4 B3", "kernels/attention/")),
        ("autotune", ("A5 P4", "autotune.py, or @autotune / Config( added")),
        ("tests", ("T1 T2 T5 T6", "tests/")),
        ("bench-harness", ("T3 T7", "a benchmark script, or a timing primitive added")),
        ("ci-infra", ("T8 B5", ".github/workflows/, .github/scripts/ or scripts/run_tests.sh")),
        ("perf-claim", ("P1 P2 P3 T7", "a performance word in the title")),
        ("uncovered-arch", ("C5", "a path or added line naming gfx1250 or gfx11")),
        ("module-move", ("I3", "a Python file deleted or renamed")),
        ("new-source", ("HK9", "a new source file with no SPDX header")),
        ("dependency", ("HK10", "requirements, pyproject.toml or setup.py")),
    ]
)
SCAN_FAMILIES = OrderedDict((k, (c.rules, c.title)) for k, c in build_categories().items())

DOCS_ONLY = re.compile(
    r"^(docs/|\.claude/|\.cursor/|\.codex/|\.github/ISSUE_TEMPLATE/|\.github/dashboard/)"
    r"|\.(md|rst|png|jpe?g|gif|svg|webp)$|^LICENSE$"
)
PUBLIC_API = (
    "python/flydsl/__init__.py",
    "python/flydsl/compiler/__init__.py",
    "python/flydsl/compiler/protocol.py",
    "python/flydsl/expr/__init__.py",
)
PERF_WORDS = re.compile(
    r"perf|optimi[sz]|faster|speed-?up|latency|throughput|tflops|tb/s|fuse|bandwidth|\d+(\.\d+)?x\b", re.I
)
UNCOVERED_ARCH = re.compile(r"gfx1250|gfx11\d\d|gfx115\d|\brdna3\b", re.I)


def derive(diff_text: str, title: str = "", root: Path | None = None) -> list:
    files = parse_diff(diff_text)
    paths = list(files)
    fams: list = []

    def hit(name):
        if name not in [f for f, _ in fams]:
            fams.append((name, PATH_FAMILIES[name][0]))

    if not paths:
        return []
    if all(DOCS_ONLY.search(p) for p in paths):
        return [("docs-only", "NONE")]
    added = "\n".join(line for f in files.values() for line in f["add"])
    live = [p for p, f in files.items() if not f["deleted"]]

    if any(p in TIER1C or p == "python/flydsl/compiler/jit_function.py" for p in paths):
        hit("cache-surface")
    if any(p in TIER1 for p in paths):
        hit("compiler-core")
    if any(p.startswith(TIER2_PREFIXES) for p in paths):
        hit("arch-table")
    if root is not None and any(tier_of(p, root) == "TIER2" and p.startswith("kernels/") for p in live):
        hit("shared-kernel-helper")
    if any(p.startswith(("lib/", "include/")) for p in paths):
        hit("mlir-cpp")
    if any(p.startswith("python/flydsl/expr/") for p in paths):
        hit("expr-surface")
    if any(p in PUBLIC_API for p in paths):
        hit("public-api")
    if any(p.startswith("kernels/") and p.endswith(".py") for p in live):
        hit("kernel")
    if any(p.startswith("kernels/") and p.endswith(".py") and files[p]["new"] for p in paths):
        hit("new-kernel")
    if any(p.startswith("kernels/attention/") for p in live):
        hit("attention")
    if "python/flydsl/autotune.py" in paths or re.search(r"@autotune\b|\bautotune\(|\bConfig\(", added):
        hit("autotune")
    if any(re.match(r"tests/(.+/)?test_[^/]+\.py$", p) and files[p]["add"] for p in live):
        hit("tests")
    if any(p.endswith(("run_benchmark.sh", "compare_benchmark.py", "benchmark_output_to_csv.py")) for p in paths) or (
        re.search(r"run_perftest|elapsed_time\(|perf_counter|cuda\.Event\(", added)
    ):
        hit("bench-harness")
    if any(p.startswith((".github/workflows/", ".github/scripts/")) or p == "scripts/run_tests.sh" for p in paths):
        hit("ci-infra")
    if PERF_WORDS.search(title or ""):
        hit("perf-claim")
    code_added = "\n".join(
        line for p, f in files.items() if p.endswith(CODE_EXT) for line in f["add"] if not _is_comment(line)
    )
    if any(UNCOVERED_ARCH.search(p) for p in live if p.endswith(CODE_EXT)) or UNCOVERED_ARCH.search(code_added):
        hit("uncovered-arch")
    if any(p.endswith(".py") and (f["deleted"] or f["renamed_from"]) for p, f in files.items()):
        hit("module-move")
    if any(
        f["new"]
        and p.endswith(CODE_EXT + (".sh",))
        and f["add"]
        and not any("SPDX-License-Identifier" in x for x in f["add"][:6])
        for p, f in files.items()
    ):
        hit("new-source")
    if any(re.search(r"(^|/)requirements[^/]*\.txt$|(^|/)pyproject\.toml$|(^|/)setup\.py$", p) for p in paths):
        hit("dependency")

    for key, cat in scan(diff_text).items():
        if cat.hits:
            fams.append((f"scan:{key}", cat.rules))
    return fams


def render_rules(fams: list) -> str:
    if not fams:
        return "[none] NONE\n"
    return "".join(f"[{name}] {rules}\n" for name, rules in fams)


RULE_TOKEN = re.compile(r"^(?:[A-Z]{1,2}\d{1,2}[a-z]?|STEP4)$")


def expected_rules(rules_text: str) -> list:
    """Rule ids `rules` derived, without the STEP4 marker (Step 4 has its own gate)."""
    ids = set()
    for line in rules_text.splitlines():
        m = re.match(r"^\s*\[[\w:-]+\]\s+(.+)$", line)
        if m:
            ids.update(t for t in m.group(1).split() if RULE_TOKEN.match(t) and t != "STEP4")
    return sorted(ids, key=_rule_sort_key)


def _rule_sort_key(rid: str):
    m = re.match(r"([A-Z]+)(\d+)([a-z]?)", rid)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (rid, 0, "")


# --------------------------------------------------------------------------- expand / mapping

RULE_HEAD = re.compile(r"^\*\*([A-Z]{1,2}\d{1,2}[a-z]?) — ")
HK_ROW = re.compile(r"^\|\s*(HK\d+)\s*\|")


def rule_bodies(rules_md: str):
    """(blocks, hk_rows, hk_header, headings): rule id -> text, and each block's category heading."""
    lines = rules_md.splitlines()
    blocks, headings, hk_rows, hk_header = OrderedDict(), {}, OrderedDict(), []
    heading, cur = None, None
    in_hk = False
    for line in lines:
        if line.startswith("### "):
            heading, cur = line, None
            in_hk = line.startswith("### Housekeeping")
            continue
        if in_hk:
            m = HK_ROW.match(line)
            if m:
                hk_rows[m.group(1)] = line
            elif line.startswith("|") and len(hk_header) < 2:
                hk_header.append(line)
            continue
        m = RULE_HEAD.match(line)
        if m:
            cur = m.group(1)
            blocks[cur] = [line]
            headings[cur] = heading
            continue
        if cur and line.strip() == "---":
            cur = None
            continue
        if cur:
            blocks[cur].append(line)
    blocks = OrderedDict((k, "\n".join(v).rstrip()) for k, v in blocks.items())
    return blocks, hk_rows, hk_header, headings


def all_rule_ids(rules_md: str) -> list:
    blocks, hk_rows, _, _ = rule_bodies(rules_md)
    return list(blocks) + list(hk_rows)


def expand_rules(rules_text: str, rules_md: str):
    want = expected_rules(rules_text)
    blocks, hk_rows, hk_header, headings = rule_bodies(rules_md)
    out, last_heading = [], None
    for rid in [r for r in blocks if r in want]:
        if headings.get(rid) != last_heading:
            last_heading = headings.get(rid)
            out += ["", last_heading or "", ""]
        out += [blocks[rid], ""]
    hk = [r for r in hk_rows if r in want]
    if hk:
        out += ["", "### Housekeeping (quick scan)", ""] + hk_header + [hk_rows[r] for r in hk]
    unresolved = [r for r in want if r not in blocks and r not in hk_rows]
    return "\n".join(out).strip() + "\n", unresolved


def render_mapping(rules_md: str) -> str:
    out = [
        "# Family -> rule mapping",
        "",
        "Generated by `triage.py mapping`. Do not edit; regenerate with",
        "`python3 triage.py mapping > MAPPING.md` and diff, which is how a disagreement",
        "between this file and the deriver is caught.",
        "",
        "## Path and title families",
        "",
        "| family | rules | fires on |",
        "|---|---|---|",
    ]
    out += [f"| `{name}` | {rules} | {why} |" for name, (rules, why) in PATH_FAMILIES.items()]
    out += ["", "## Scan families (a category with at least one candidate)", "", "| family | rules | candidates |"]
    out += ["|---|---|---|"]
    out += [f"| `scan:{name}` | {rules} | {why} |" for name, (rules, why) in SCAN_FAMILIES.items()]
    reachable = set()
    for rules, _ in list(PATH_FAMILIES.values()) + list(SCAN_FAMILIES.values()):
        reachable.update(rules.split())
    ids = all_rule_ids(rules_md)
    unreachable = [r for r in ids if r not in reachable]
    unknown = sorted(r for r in reachable - set(ids) - {"NONE", "STEP4"})
    out += ["", f"{len(ids)} rules in rules.md; unreachable from every family: {', '.join(unreachable) or 'none'}."]
    if unknown:
        out.append(f"Derived but missing from rules.md: {', '.join(unknown)}.")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- evidence


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def buffer_bounds(diff_text: str, head_root: Path) -> list:
    """Added buffer-descriptor constructions that bind a tensor with no byte bound.

    `max_size` defaults to True for both constructors, so the literal text is not enough: an omitted
    argument is the common spelling of an unbounded descriptor.
    """
    want = {}
    for path, lineno, _ in added_lines(diff_text):
        if path.endswith(".py") and not path.startswith(".claude/"):
            want.setdefault(path, set()).add(lineno)
    rows = []
    for path, lines in want.items():
        src_path = head_root / path
        if not src_path.is_file():
            rows.append((path, 0, "", "head image unavailable; not checked"))
            continue
        try:
            tree = ast.parse(src_path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            rows.append((path, 0, "", f"does not parse at head ({exc.msg}); not checked"))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in BUF_CALLS:
                continue
            span = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            if not span & lines:
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if "num_records_bytes" in kw:
                continue
            pos = 1 if _call_name(node) == "make_buffer_tensor" else 2
            size = kw.get("max_size", node.args[pos] if len(node.args) > pos else None)
            if size is not None and not (isinstance(size, ast.Constant) and size.value is True):
                continue
            tensor = ast.unparse(node.args[0]) if node.args else "?"
            how = "max_size=True" if size is not None else "max_size defaulted to True"
            rows.append((path, node.lineno, f"{_call_name(node)}({tensor})", how))
    return rows


def render_bounds(rows: list) -> str:
    if not rows:
        return (
            "BUFFER BOUNDS: no added make_buffer_tensor / create_buffer_resource call binds a tensor without a "
            "byte bound.\n"
        )
    out = [
        "BUFFER BOUNDS: added descriptors with no byte bound (size 0xFFFFFFFF, RDNA bounds checking off).",
        "Candidates, not verdicts: fire D5 only when the bound dimension is a runtime extent (M, tokens,",
        "num_valid) and no mask guards the access; full-size weights and caches are fine.",
    ]
    out += [f"  {p}:{n}: {call} -- {how}" if n else f"  {p}: {how}" for p, n, call, how in rows]
    return "\n".join(out) + "\n"


def run_tests_dirs(root: Path) -> list:
    text = (root / "scripts" / "run_tests.sh").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"python3 -m pytest \\\n((?:\s+tests/\S+ \\\n)+)", text)
    return re.findall(r"(tests/\S+?/?)\s+\\", m.group(1)) if m else []


def _head_text(path: str, files: dict, head_root: Path | None) -> str:
    if head_root is not None and (head_root / path).is_file():
        return (head_root / path).read_text(encoding="utf-8", errors="replace")
    return "\n".join(files[path]["add"])


def ci_coverage(diff_text: str, root: Path, head_root: Path | None = None) -> list:
    files = parse_diff(diff_text)
    dirs = run_tests_dirs(root)
    rows = []
    for path, f in files.items():
        name = Path(path).name
        if f["deleted"]:
            continue
        if path.startswith("tests/") and name.startswith("test_") and name.endswith(".py"):
            text = _head_text(path, files, head_root)
            notes = []
            collected = any(path.startswith(d) for d in dirs)
            if not collected:
                notes.append(f"outside the directories scripts/run_tests.sh passes to pytest ({' '.join(dirs)})")
            if re.search(r"mark\.multi_gpu", text):
                notes.append("marked multi_gpu: runs only in the label-gated multi-gpu job")
            if re.search(r"mark\.benchmark", text):
                notes.append("marked benchmark: deselected by scripts/run_tests.sh")
            if re.search(r"large_shape", text):
                notes.append("large_shape rows run in CI (RUN_TESTS_FULL=1) but not in a default local run")
            if not re.search(r"^\s*def test_|^\s*class Test", text, re.M):
                notes.append("defines no test_* function or Test* class: pytest collects nothing")
            for m in re.finditer(r"^(if .+):\s*\n\s+pytest\.skip\([^\n]*allow_module_level\s*=\s*True", text, re.M):
                notes.append(f"module-level skip when `{m.group(1)[3:].strip()}` -- check a PR runner can run it (C5)")
            status = (
                "RUNS in PR CI"
                if collected and not any("only in" in n or "deselected" in n for n in notes)
                else "NOT in the default PR CI run"
            )
            rows.append((path, status, notes))
        elif path.startswith("examples/") and name.endswith(".py") and f["new"]:
            registered = path[len("examples/") :] in _example_registry(root)
            note = (
                "registered in tests/arch_compat.py"
                if registered
                else ("not in tests/arch_compat.py EXAMPLE_ARCHITECTURES: scripts/run_tests.sh looks it up and aborts")
            )
            rows.append((path, "example", [note]))
    return rows


def _example_registry(root: Path) -> set:
    try:
        tree = ast.parse((root / "tests" / "arch_compat.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EXAMPLE_ARCHITECTURES" for t in node.targets):
            if isinstance(node.value, ast.Dict):
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return set()


def render_ci_coverage(rows: list) -> str:
    if not rows:
        return "CI COVERAGE: this diff adds or changes no test file and no example.\n"
    out = ["CI COVERAGE: where each changed test or new example runs (scripts/run_tests.sh is the PR CI suite)."]
    for path, status, notes in rows:
        out.append(f"  {path}: {status}")
        out += [f"    - {n}" for n in notes]
    return "\n".join(out) + "\n"


def bench_coverage(diff_text: str, root: Path) -> list:
    """Changed kernel modules, and the run_benchmark.sh tests that import them within three hops."""
    bench = (root / "scripts" / "run_benchmark.sh").read_text(encoding="utf-8", errors="replace")
    bench_tests = sorted(set(re.findall(r"tests/kernels/test_\w+\.py", bench)))
    index = import_index(root)
    by_module = {_module_of(f): mods for f, mods in index.items() if _module_of(f)}
    rows = []
    for path, f in parse_diff(diff_text).items():
        if f["deleted"] or not (path.startswith("kernels/") and path.endswith(".py")):
            continue
        target = _module_of(path)
        reached = []
        for test in bench_tests:
            frontier, seen = set(index.get(test, ())), set()
            for _ in range(3):
                if target in frontier:
                    reached.append(test)
                    break
                seen |= frontier
                nxt = set()
                for mod in frontier:
                    nxt |= {m for m in by_module.get(mod, ()) if m.startswith("kernels")}
                frontier = nxt - seen
        rows.append((path, reached))
    return rows


def render_bench_coverage(rows: list) -> str:
    if not rows:
        return "BENCHMARK COVERAGE: no kernels/ file changed.\n"
    out = ["BENCHMARK COVERAGE: scripts/run_benchmark.sh tests that import each changed kernel (<= 3 hops)."]
    for path, tests in rows:
        out.append(f"  {path}: " + (", ".join(tests) if tests else "no CI benchmark reaches it"))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- gates

ADJUDICATION = re.compile(r"^\s*([A-Z]{1,2}\d{1,2}[a-z]?)\s+(FIRE|CLEAR|N/A)\s*(?:--|—|:)\s*(.+?)\s*$")
ANSWER_KEY = re.compile(r"^\s*(Q[1-5]|BLIND)\s*[:\-]\s*(.+?)\s*$", re.I)
DIAGNOSTIC_KEY = re.compile(r"^\s*([1-6])\s*[:.\-]\s*(.+?)\s*$")
CORE_LINE = re.compile(r"^\s*(\S+)\s+(TIER1C|TIER1|TIER2)\s+(COVERED|GAP|N/A)\s*(?:--|—|:)\s*(.+?)\s*$")
CORE_NONE = re.compile(r"^\s*NONE\s*(?:--|—|:)\s*(.+?)\s*$")
FINDING = re.compile(r"^\s*[-*]?\s*(\U0001F534|\u26A0\uFE0F?|\U0001F4DD)\s*(.+)$")
REFUTATION = re.compile(r"^\s*(RED|WARN|NOTE)\s+(SURVIVED|KILLED)\s*(?:--|—|:)\s*(.+?)\s*$")
INDEP = re.compile(r"^\s*(SURVIVED|KILLED)\s*(?:--|—|:)\s*(.+?)\s*$")
INDEP_NONE = re.compile(r"^\s*NONE AVAILABLE\s*(?:--|—|:)\s*(.+?)\s*$")
CITATION = re.compile(r"([\w./-]+\.(?:mlir|yaml|toml|cpp|hpp|json|yml|py|td|cc|sh|md|h))(?![\w])")
IDENT = re.compile(r"\b[a-z]+_[a-z0-9_]{2,}\b|\b[A-Z][A-Z0-9_]{3,}\b|`[^`]{3,}`")
VALUE = re.compile(r"\b\d+\b|fp8|fp16|bf16|fp4|int8|int32|int64|i32|i64|e4m3|e5m2|gfx\d+|wave(32|64)", re.I)
SURFACE = re.compile(r"^(n/?a|none|no|yes|ok|nothing|same|unchanged|see above|tbd|clean|fine|-+)\.?$", re.I)
MIN_ANSWER, MIN_EVIDENCE, MIN_CORE_REASON, MIN_REFUTE = 40, 12, 30, 20


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _touches(cited: str, touched: set) -> bool:
    c = cited[2:] if cited.startswith("./") else cited
    return any(t == c or t.endswith("/" + c) for t in touched)


def card_findings(card: str) -> list:
    out = []
    in_fence = False
    for line in card.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        m = FINDING.match(line)
        if m and not in_fence:
            out.append((m.group(1), m.group(2).strip()))
    return out


def gate_answers(work: Path, root: Path, diff: str) -> list:
    text = _read(work / "answers.txt")
    if text is None:
        return ["MISSING answers.txt -- Step 2 and Step 8.5 have not been written"]
    seen = {m.group(1).upper(): m.group(2) for m in map(ANSWER_KEY.match, text.splitlines()) if m}
    bad = [f"UNANSWERED {q}" for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "BLIND") if q not in seen]
    bad += [f"NO-SUBSTANCE {q}: '{seen[q][:40]}'" for q in seen if len(seen[q]) < MIN_ANSWER or SURFACE.match(seen[q])]
    return bad


def gate_corefiles(work: Path, root: Path, diff: str) -> list:
    text = _read(work / "core_files.txt")
    if text is None:
        return ["MISSING core_files.txt -- Step 4 has not been written"]
    want = core_files(diff, root)
    touched = changed_paths(diff)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows, nones, bad = {}, [], []
    for ln in lines:
        m = CORE_LINE.match(ln)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(4))
            continue
        m = CORE_NONE.match(ln)
        if m:
            nones.append(m.group(1))
            continue
        bad.append(f"MALFORMED '{ln[:60]}'")
    for path, (tier, reason) in rows.items():
        if not _touches(path, touched):
            bad.append(f"UNTOUCHED-FILE {path}: the diff does not change it")
    for path, tier in want.items():
        got = next((v for k, v in rows.items() if _touches(k, {path}) or k == path), None)
        if got is None:
            bad.append(f"UNASSESSED {path} ({tier})")
            continue
        if got[0] != tier:
            bad.append(f"TIER-MISMATCH {path}: recorded {got[0]}, table says {tier}")
        if len(got[1]) < MIN_CORE_REASON or SURFACE.match(got[1]):
            bad.append(f"NO-EVIDENCE {path}: the reason must name what THIS PR changed")
    if want and nones:
        bad.append(f"UNDECLARED-CORE NONE written while the diff touches {', '.join(want)}")
    if not want and not rows and not any(len(n) >= MIN_CORE_REASON for n in nones):
        bad.append("NO-EVIDENCE a NONE line must say which Tier 3 files the diff touches")
    return bad


def gate_ledger(work: Path, root: Path, diff: str) -> list:
    rules_text = _read(work / "rules.txt")
    if not rules_text or not rules_text.strip():
        return ["MISSING rules.txt -- Step 1b never ran; nothing can be adjudicated against it"]
    want = expected_rules(rules_text)
    if not want:
        return []
    text = _read(work / "verdicts.txt") or ""
    seen = {}
    for ln in text.splitlines():
        m = ADJUDICATION.match(ln)
        if m:
            seen[m.group(1)] = (m.group(2), m.group(3))
    touched = changed_paths(diff)
    bad = [f"UNADJUDICATED {r}" for r in want if r not in seen]
    for r in want:
        if r in seen:
            verdict, reason = seen[r]
            if len(reason) < MIN_EVIDENCE or SURFACE.match(reason):
                bad.append(f"NO-EVIDENCE {r} {verdict}: '{reason[:40]}'")
            if verdict == "FIRE":
                cited = [c for c in CITATION.findall(reason) if not c.endswith((".txt", ".json"))]
                if cited and not any(_touches(c, touched) for c in cited):
                    bad.append(f"UNTOUCHED-CITATION {r}: cites only {cited[0]}, which this PR does not change")
    return bad


def gate_diagnostic(work: Path, root: Path, diff: str) -> list:
    text = _read(work / "ai_diagnostic.txt")
    if text is None:
        return ["MISSING ai_diagnostic.txt -- Step 7 has not been written"]
    seen = {m.group(1): m.group(2) for m in map(DIAGNOSTIC_KEY.match, text.splitlines()) if m}
    bad = [f"UNANSWERED check {q}" for q in "123456" if q not in seen]
    bad += [f"NO-SUBSTANCE check {q}" for q in seen if len(seen[q]) < MIN_ANSWER or SURFACE.match(seen[q])]
    return bad


def _rule_code_pattern(rules_md: str):
    ids = sorted(all_rule_ids(rules_md), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, ids)) + r")(?![\w-])") if ids else None


def gate_card(work: Path, root: Path, diff: str) -> list:
    card = _read(work / "card.md")
    if card is None or not card.strip():
        return ["MISSING card.md -- write the card before running the gates"]
    bad = []
    for label, pattern in (
        ("Review (advisory", r"^\s*Review \(advisory"),
        ("CI (", r"^\s*CI \("),
        ("Perf (", r"^\s*Perf \("),
    ):
        if not re.search(pattern, card, re.M):
            bad.append(f"NO-HEADER the '{label}...' line is missing")
    findings = card_findings(card)
    if len(findings) > 5:
        bad.append(f"TOO-MANY {len(findings)} findings; keep the 5 most severe")
    touched = changed_paths(diff)
    codes = _rule_code_pattern(_read(RULES_MD) or "")
    for sev, text in findings:
        head = text[:60]
        if "[verified]" not in text and "[inferred]" not in text:
            bad.append(f"UNTAGGED '{head}': tag it [verified] or [inferred]")
        if "Author must" not in text and "Reviewer should ask" not in text:
            bad.append(f"NO-ACTION '{head}': end with **Author must** or **Reviewer should ask**")
        if codes and codes.search(text):
            bad.append(f"RULE-CODE '{head}': rule ids are internal labels ({codes.search(text).group(1)})")
        cited = CITATION.findall(text)
        if cited and not any(_touches(c, touched) for c in cited):
            bad.append(f"UNANCHORED '{head}': cites only {cited[0]} and nothing this PR changes")
        if sev == "\U0001f534" and not (VALUE.search(text) or len(set(IDENT.findall(text))) >= 2):
            bad.append(f"UNPROVEN-RED '{head}': name the shape, dtype, arch or value that makes it fire")
    verdicts = _read(work / "verdicts.txt") or ""
    body = "\n".join(t for _, t in findings)
    for ln in verdicts.splitlines():
        m = ADJUDICATION.match(ln)
        if not m or m.group(2) != "FIRE" or "not reported:" in m.group(3).lower():
            continue
        anchors = [c for c in CITATION.findall(m.group(3)) if _touches(c, touched)]
        anchors = [a.rsplit("/", 1)[-1] for a in anchors] or [t.strip("`") for t in IDENT.findall(m.group(3))]
        if not any(a and a in body for a in anchors):
            bad.append(
                f"UNREPORTED-FIRE {m.group(1)}: adjudicated FIRE and absent from the card -- report it, change the "
                "verdict, or append `-- not reported: <reason>`"
            )
    indep = _read(work / "independent.txt") or ""
    if INDEP_NONE.search(indep) and findings and "not independently refuted" not in card:
        bad.append("UNADMITTED no independent reader ran, and the review line does not say 'not independently refuted'")
    return bad


def gate_refutations(work: Path, root: Path, diff: str) -> list:
    findings = card_findings(_read(work / "card.md") or "")
    if not findings:
        return []
    text = _read(work / "refutations.txt")
    if text is None:
        return ["MISSING refutations.txt -- Step 8.6 has not been written"]
    bad, survived = [], 0
    for ln in (x for x in text.splitlines() if x.strip()):
        m = REFUTATION.match(ln)
        if not m:
            bad.append(f"MALFORMED '{ln[:60]}'")
            continue
        survived += m.group(2) == "SURVIVED"
        reason = m.group(3)
        if len(reason) < MIN_REFUTE or not (CITATION.search(reason) or IDENT.search(reason) or "git " in reason):
            bad.append(f"NO-EVIDENCE '{reason[:40]}': name the file, symbol or command you opened")
    if survived < len(findings):
        bad.append(f"UNATTACKED {len(findings)} findings on the card, {survived} SURVIVED refutation lines")
    return bad


def gate_independent(work: Path, root: Path, diff: str) -> list:
    card = _read(work / "card.md") or ""
    findings = card_findings(card)
    text = _read(work / "independent.txt")
    if not findings:
        return []
    if text is None:
        return ["MISSING independent.txt -- Step 8.7 has not run; write NONE AVAILABLE if no reader exists"]
    if INDEP_NONE.search(text):
        return (
            [] if "not independently refuted" in card else ["UNADMITTED the card must say 'not independently refuted'"]
        )
    rows = [INDEP.match(x) for x in text.splitlines() if x.strip()]
    if not all(rows):
        return ["MALFORMED independent.txt: one 'SURVIVED|KILLED -- what they opened' line per finding"]
    survived = sum(m.group(1) == "SURVIVED" for m in rows)
    bad = []
    if survived != len(findings):
        bad.append(f"COUNT-MISMATCH {len(findings)} findings on the card, {survived} survived the independent reader")
    bad += [f"NO-EVIDENCE '{m.group(2)[:40]}'" for m in rows if len(m.group(2)) < MIN_REFUTE]
    return bad


GATES = (
    ("answers", gate_answers),
    ("corefiles", gate_corefiles),
    ("ledger", gate_ledger),
    ("diagnostic", gate_diagnostic),
    ("card", gate_card),
    ("refutations", gate_refutations),
    ("independent", gate_independent),
)


def run_gates(work: Path, root: Path) -> int:
    diff = _read(work / "pr.diff") or ""
    lines, red = [], 0
    for name, fn in GATES:
        problems = fn(work, root, diff)
        if problems:
            red += 1
            lines.append(f"  ❌ {name}")
            lines += [f"       {p}" for p in problems]
        else:
            lines.append(f"  ✅ {name}")
    report = f"======== Step 9 gates ({work}) ========\n" + "\n".join(lines)
    report += f"\n======== {len(GATES) - red} green / {red} red ========\n"
    print(report, end="")
    (work / "GATES.txt").write_text(report, encoding="utf-8")
    return 1 if red else 0


CORRECTIONS_HEAD = "Independent reader's corrections (not yet applied to the findings above):"
CORRECTION = re.compile(r"\bCorrection:\s*(.+)$", re.I)


def apply_refutation(card_path: Path, indep_path: Path) -> str:
    """Drop the findings the independent reader KILLED and list its corrections to the rest, in card order."""
    card = (_read(card_path) or "").split("\n" + CORRECTIONS_HEAD)[0].rstrip() + "\n"
    text = _read(indep_path) or ""
    findings = card_findings(card)
    rows = [INDEP.match(x) for x in text.splitlines() if x.strip()]
    if INDEP_NONE.search(text) or not findings:
        return "no-op: no independent verdicts to apply"
    if not all(rows) or len(rows) != len(findings):
        return f"no-op: {len(rows)} verdict lines for {len(findings)} findings; fix independent.txt by hand"
    killed = {t for (_, t), m in zip(findings, rows) if m.group(1) == "KILLED"}
    kept = [ln for ln in card.splitlines() if not ((m := FINDING.match(ln)) and m.group(2).strip() in killed)]
    survivors = [m.group(2) for m in rows if m.group(1) == "SURVIVED"]
    notes = [f"- Finding {i}: {c.group(1).strip()}" for i, s in enumerate(survivors, 1) if (c := CORRECTION.search(s))]
    if notes:
        kept += ["", CORRECTIONS_HEAD] + notes
    card_path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
    return f"removed {len(killed)} killed finding(s); listed {len(notes)} correction(s)"


# --------------------------------------------------------------------------- main


def _arg(i: int, default=None):
    return sys.argv[i] if len(sys.argv) > i else default


def read(i: int) -> str:
    return Path(sys.argv[i]).read_text(encoding="utf-8", errors="replace")


def main() -> int:
    mode = _arg(1, "")
    if mode == "scan":
        print(render_scan(scan(read(2))), end="")
    elif mode == "rules":
        root = Path(_arg(4)) if _arg(4) else None
        print(render_rules(derive(read(2), _arg(3, "") or "", root)), end="")
    elif mode == "expand":
        text, unresolved = expand_rules(read(2), read(3))
        print(text, end="")
        if unresolved:
            print(f"rules derived with no body in rules.md: {', '.join(unresolved)}", file=sys.stderr)
            return 1
    elif mode == "mapping":
        print(render_mapping(RULES_MD.read_text(encoding="utf-8")), end="")
    elif mode == "core":
        rows = core_files(read(2), Path(sys.argv[3]))
        print("".join(f"{p} {t}\n" for p, t in rows.items()) or "NONE (no Tier 1 / 1c / 2 file)\n", end="")
    elif mode == "bounds":
        print(render_bounds(buffer_bounds(read(2), Path(sys.argv[3]))), end="")
    elif mode == "citest":
        head = Path(_arg(4)) if _arg(4) else None
        print(render_ci_coverage(ci_coverage(read(2), Path(sys.argv[3]), head)), end="")
    elif mode == "benchcov":
        print(render_bench_coverage(bench_coverage(read(2), Path(sys.argv[3]))), end="")
    elif mode == "gates":
        return run_gates(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "apply-refutation":
        print(apply_refutation(Path(sys.argv[2]), Path(sys.argv[3])))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
