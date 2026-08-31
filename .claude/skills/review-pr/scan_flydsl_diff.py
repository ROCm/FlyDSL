#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Deterministic static scan of a FlyDSL PR diff.

Emits candidate sites for the review rules that can be checked mechanically, so a
reviewer works a fixed list instead of relying on what a language model happened to
notice. Every category is deliberately noisy in the safe direction.

Candidates are not verdicts. A candidate becomes a finding only when the reviewer can
name the concrete shape, dtype, arch, or value that makes it fire.

Usage:
    scan_flydsl_diff.py --diff /tmp/pr.diff
    scan_flydsl_diff.py ROCm/FlyDSL 1064
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

PY_EXT = (".py",)
CPP_EXT = (".cpp", ".h", ".td", ".hpp", ".cc")

# An identifier that names a position: which page, which block, which token.
INDEX_SHAPED = re.compile(
    r"\b\w*(?:_id|_idx|_index|idx|_pos)\b"
    r"|\b(?:bid|pid|tid|wid|lane|phys|phys_row|block|blk|row|col|page|token|slot|step|seq|tile)\w*\b",
    re.I,
)
# An identifier that names an extent: how far apart two positions are.
STRIDE_SHAPED = re.compile(
    r"\b\w*(?:stride|pitch|extent|numel|nelem|_dim|_size|_bytes|_width|_len|pages|_elems?)\w*\b"
    r"|\b(?:head_dim|n_kv|num_kv|hidden|dhw|npq|elem_bytes|page_bytes)\b",
    re.I,
)
WIDENED = re.compile(
    r"fx\.Int64|Int64\(|int64_t|torch\.int64|\.to\(\s*(?:tl\.)?int64|static_cast<int64_t>|\bi64\b|IntegerType::get\w*\(\s*\w+,\s*64",
    re.I,
)


@dataclass
class Hit:
    path: str
    line: int
    text: str
    note: str = ""


@dataclass
class Category:
    key: str
    title: str
    guidance: str
    hits: list[Hit] = field(default_factory=list)


def _iter_added_lines(diff_text: str):
    """Yield (path, new_line_number, added_line_text) for every '+' line in a diff."""
    path = None
    new_lineno = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else re.sub(r"^b/", "", target)
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            new_lineno = int(m.group(1)) if m else 0
            continue
        if path is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            yield path, new_lineno, raw[1:]
            new_lineno += 1
        elif raw.startswith("-"):
            continue
        else:
            new_lineno += 1


def _changed_files(diff_text: str) -> list[str]:
    return sorted(
        {
            re.sub(r"^b/", "", m.strip())
            for m in re.findall(r"^\+\+\+ (.+)$", diff_text, re.M)
            if m.strip() != "/dev/null"
        }
    )


def _is_py(path: str) -> bool:
    return path.endswith(PY_EXT)


def _is_scanned(path: str) -> bool:
    """Skill prose and its test fixtures quote these patterns literally.

    Without this, any PR touching .claude/skills/** lights up every category by matching
    the rules' own example strings, which is noise that would train a reviewer to ignore
    the output.
    """
    return not path.startswith(".claude/")


def _is_test(path: str) -> bool:
    return "/tests/" in f"/{path}" or path.startswith("tests/") or "test_" in path.rsplit("/", 1)[-1]


def _is_kernel(path: str) -> bool:
    return path.startswith("kernels/")


def build_categories() -> dict[str, Category]:
    c = [
        Category(
            "index_width",
            "Index x stride multiply with no 64-bit widening",
            "Clear each site or name the tensor size at which the product passes 2^31 elements. "
            "The repo idiom is to wrap the position operand: fx.Int64(phys_row[sub]) * n_kv * ...",
        ),
        Category(
            "abi_flatten",
            "Tensor flattened before a launch",
            "_LayoutPlan packs every shape entry as int32 regardless of use_32bit_stride, so a flattened "
            "tensor overflows the C ABI at numel >= 2^31. Confirm the flattened extent stays below that.",
        ),
        Category(
            "trunc_div",
            "Truncating division feeding a copy or loop count",
            "Ask whether the numerator is a multiple of the denominator for EVERY shape this factory accepts. "
            "If it depends on a caller-supplied tile dim or head_dim, demand an assert on the exact "
            "divisibility condition, or a ceil-div plus a masked tail.",
        ),
        Category(
            "cache_key",
            "Cache-key surface touched",
            "State the direction. Adding to the key risks AOT misses across build-vs-runtime; omitting from it "
            "risks serving a stale artifact; a trait in the key that nothing reads is a dead flag.",
        ),
        Category(
            "arch_capability",
            "Architecture capability derived inline",
            "A family predicate answers 'is this RDNA', not 'what is the wave size'. gfx1250 is wave32 and is "
            "not matched by RDNA prefixes. Capabilities must come from the accessor, not from an inline branch.",
        ),
        Category(
            "dead_gate",
            "Gate that is always false, or a default-off trait",
            "A surviving 'False and' dead-codes the path entirely. A new default-off trait in front of "
            "previously unconditional behavior orphans every builder that does not pass it.",
        ),
        Category(
            "fastmath",
            "Fastmath / math-wrapper surface",
            "An fx.* math wrapper needs @dsl_math_wrap_result and a named fastmath parameter, or it silently "
            "emits fastmath<none> at every call site. A bare fastmath=True is an invalid attribute.",
        ),
        Category(
            "raw_mlir",
            "Raw MLIR dialect use instead of the fx.* surface",
            "The project's most-repeated house rule. Prefer fx.Int32/fx.Int64/fx.copy/fx.gemm/SharedAllocator "
            "over arith./vector./scf./memref/llvm./ir.* and ArithValue.",
        ),
        Category(
            "stream_race",
            "Host-side copy near an explicit stream",
            "A .contiguous()/.cat()/.to() runs on the ambient current stream while the kernel consumes it on "
            "the caller's stream. Also check the copy does not silently materialize the output tensor.",
        ),
        Category(
            "aiter_positional",
            "Call into aiter with positional arguments",
            "CI clones aiter at main HEAD, so an argument reorder there breaks every FlyDSL PR at once. "
            "Parameter names are the stable contract; pass them by keyword.",
        ),
        Category(
            "block_id",
            "Raw block-id read",
            "Once a kernel derives remapped block indices, any later raw gpu.block_id read bypasses the remap. "
            "Expect exactly one derivation site.",
        ),
        Category(
            "launch_geometry",
            "Kernel launch geometry",
            "Any path that can launch more than 256 threads needs known_block_size, or the AMDGPU default "
            "max_flat_workgroup_size of 256 aborts the launch.",
        ),
        Category(
            "test_shape",
            "Test or benchmark assertion surface",
            "Check the test can actually fail: tolerance not widened, shape rows not disabled, a real assert "
            "rather than a return, and a reference that is not a twin of the kernel.",
        ),
        Category(
            "cpp_attr",
            "C++ attribute width / static read",
            "IntAttr::getDynamic defaults its width to 32. getValue() on an attribute with an isStatic() "
            "predicate needs the predicate first, because a dynamic attr reads back as 0.",
        ),
    ]
    return {x.key: x for x in c}


def _cache_tag_files(diff_text: str) -> tuple[set[str], set[str]]:
    """Per file: does its diff show the cache_tag idiom, and was that region edited?

    Hunk granularity matters. A trait is added to a cache_tag tuple as a bare
    ``self.NEW_TRAIT,`` line that does not itself contain the string "cache_tag", so
    scanning added lines alone would wrongly report the tuple as untouched.
    """
    using: set[str] = set()
    edited: set[str] = set()
    path = None
    hunk: list[str] = []

    def flush():
        # "uses the idiom" is a property of the file; "was edited" is a property of the
        # hunk, so only the latter is hunk-scoped.
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
            # git appends the enclosing function to the hunk header, so "@@ ... def
            # cache_tag(self):" is the strongest available signal that the addition below
            # lands inside the tuple itself.
            hunk = [raw] if "cache_tag" in raw else []
            if path and "cache_tag" in raw:
                using.add(path)
        elif path:
            hunk.append(raw)
            if "cache_tag" in raw:
                using.add(path)
    flush()
    return using, edited


def scan(diff_text: str) -> dict[str, Category]:
    cats = build_categories()
    files = _changed_files(diff_text)
    cache_tag_files, cache_tag_edited = _cache_tag_files(diff_text)

    new_traits: list[Hit] = []

    for path, lineno, text in _iter_added_lines(diff_text):
        if not _is_scanned(path):
            continue
        stripped = text.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        py = _is_py(path)
        cpp = path.endswith(CPP_EXT)

        if (py or cpp) and "*" in text and not WIDENED.search(text):
            if INDEX_SHAPED.search(text) and STRIDE_SHAPED.search(text):
                if re.search(r"[\w\]\)]\s*\*\s*[\w\(]", text) and "**" not in text:
                    cats["index_width"].hits.append(Hit(path, lineno, stripped))

        if py and re.search(r"\.(?:view|reshape)\(\s*-1|\.flatten\(\)", text):
            cats["abi_flatten"].hits.append(Hit(path, lineno, stripped))

        if (
            py
            and "//" in text
            and re.search(
                r"\b\w*(?:load|chunk|vec|copy|tile|unit|bytes|elems?|iters?|steps?|rounds?|per_thread|per_row)\w*\b",
                text,
                re.I,
            )
        ):
            if re.search(r"[\w\)\]]\s*//\s*[\w\(]", text):
                cats["trunc_div"].hits.append(Hit(path, lineno, stripped))

        if "cache_tag" in text:
            cats["cache_key"].hits.append(Hit(path, lineno, stripped, "cache_tag edited"))
        if "_CACHE_INVALIDATING_ENV_VARS" in text or "_flydsl_key" in text or "__cache_signature__" in text:
            cats["cache_key"].hits.append(Hit(path, lineno, stripped, "cache-key machinery edited"))
        # Only a traits field (self.UPPER) or a constant in a file that already uses the
        # cache_tag idiom is a cache-key candidate. Plain module constants in a new kernel
        # file are not, and treating them as such buries the real signal.
        if py and re.match(r"^\s*self\.[A-Z][A-Z0-9_]{2,}\s*[:=]", stripped):
            new_traits.append(Hit(path, lineno, stripped, "traits field added"))
        elif py and path in cache_tag_files and re.match(r"^\s*[A-Z][A-Z0-9_]{2,}\s*[:=]", stripped):
            new_traits.append(Hit(path, lineno, stripped, "constant added in a cache_tag file"))

        if py and re.search(r"is_rdna_arch|is_cdna|startswith\(\s*[\"']gfx|wave64|WAVE_SIZE\s*=|get_warp_size\(", text):
            if "def get_warp_size" not in text:
                cats["arch_capability"].hits.append(Hit(path, lineno, stripped))
        if re.search(r"[\"']gfx\d{3,4}[\"']", text) and not _is_test(path):
            cats["arch_capability"].hits.append(Hit(path, lineno, stripped, "arch string literal"))

        if re.search(r"\bFalse\s+and\b|\band\s+False\b|if\s+False\s*:", text):
            cats["dead_gate"].hits.append(Hit(path, lineno, stripped, "always-false gate"))
        if py and re.search(r"^\s*[A-Z_]*(?:SWIZZLE|ENABLE|USE_|LAZY|OPT_)[A-Z0-9_]*\s*[:=]\s*False", stripped):
            cats["dead_gate"].hits.append(Hit(path, lineno, stripped, "default-off trait"))

        if re.search(r"fastmath\s*=\s*(?:True|False|[\"'])", text):
            cats["fastmath"].hits.append(Hit(path, lineno, stripped, "non-enum fastmath value"))
        if py and "expr/" in path and re.match(r"^\s*def\s+\w+", stripped):
            cats["fastmath"].hits.append(Hit(path, lineno, stripped, "expr wrapper added or edited"))

        if py and re.search(
            r"\bir\.(?:IntegerType|F32Type|Value|Type)|ArithValue|_to_raw|\bSmemAllocator\b|\bmemref_alloca\b"
            r"|\b(?:arith|scf|vector|memref|llvm)\.[a-z_]+\(|inline_asm",
            text,
        ):
            cats["raw_mlir"].hits.append(Hit(path, lineno, stripped))

        if py and re.search(r"\.contiguous\(\)|torch\.cat\(|\.reshape\(|\.to\(\s*(?:device|torch\.)", text):
            cats["stream_race"].hits.append(Hit(path, lineno, stripped))

        if py and re.search(r"\baiter[\w.]*\(|from\s+aiter|import\s+aiter", text):
            cats["aiter_positional"].hits.append(Hit(path, lineno, stripped))

        if re.search(r"gpu\.block_id\(|block_id\(\s*[\"']", text):
            cats["block_id"].hits.append(Hit(path, lineno, stripped))

        if re.search(r"known_block_size|@flyc\.kernel|block_dim\s*=|BLOCK_M\s*\*", text):
            cats["launch_geometry"].hits.append(Hit(path, lineno, stripped))

        if _is_test(path) or "benchmark" in path:
            if re.search(r"\b(?:atol|rtol|tol|eps)\s*=\s*[0-9.eE-]+", text):
                cats["test_shape"].hits.append(Hit(path, lineno, stripped, "tolerance"))
            if re.match(r"^\s*#\s*\(", stripped) or re.match(r"^\s*#\s*[\"']?\w+.*,\s*$", stripped):
                cats["test_shape"].hits.append(Hit(path, lineno, stripped, "possibly disabled row"))
            if re.search(r"for\s+\w+\s+in\s+re\.finditer", text):
                cats["test_shape"].hits.append(Hit(path, lineno, stripped, "finditer loop: keeps last match?"))

        if cpp and re.search(r"getDynamic\(\s*\w+\s*\)|\.getValue\(\)|static_cast<", text):
            cats["cpp_attr"].hits.append(Hit(path, lineno, stripped))

    for h in new_traits[:12]:
        if not _is_kernel(h.path):
            continue
        note = (
            "cache_tag region was edited -- confirm THIS trait is in the tuple"
            if h.path in cache_tag_edited
            else "no cache_tag edit in this file's diff -- does this reach codegen?"
        )
        cats["cache_key"].hits.append(Hit(h.path, h.line, h.text, note))

    for f in files:
        if f.endswith(("run_benchmark.sh", "compare_benchmark.py", "benchmark_output_to_csv.py")):
            cats["test_shape"].hits.append(
                Hit(f, 0, "(benchmark harness changed)", "was the stored baseline produced by the same parser?")
            )
    return cats


def render(cats: dict[str, Category], limit: int) -> str:
    out: list[str] = []
    out.append("=== FlyDSL diff scan: candidates, not verdicts ===")
    total = sum(len(c.hits) for c in cats.values())
    if total == 0:
        out.append("no candidates in any category (say so explicitly; do not skip the categories silently)")
        return "\n".join(out)
    for cat in cats.values():
        if not cat.hits:
            out.append(f"\n[{cat.key}] 0 candidates")
            continue
        out.append(f"\n[{cat.key}] {len(cat.hits)} candidate(s) -- {cat.title}")
        out.append(f"  guidance: {cat.guidance}")
        seen = set()
        shown = 0
        for h in cat.hits:
            sig = (h.path, h.line)
            if sig in seen:
                continue
            seen.add(sig)
            if shown >= limit:
                out.append(f"  ... {len(cat.hits) - shown} more suppressed (raise --limit to see all)")
                break
            note = f"  <- {h.note}" if h.note else ""
            out.append(f"  {h.path}:{h.line}: {h.text[:140]}{note}")
            shown += 1
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", help="owner/repo, e.g. ROCm/FlyDSL")
    ap.add_argument("pr", nargs="?", type=int, help="PR number")
    ap.add_argument("--diff", help="path to a unified diff instead of fetching one")
    ap.add_argument("--limit", type=int, default=12, help="max sites shown per category")
    args = ap.parse_args()

    if args.diff:
        with open(args.diff, encoding="utf-8", errors="replace") as fh:
            diff_text = fh.read()
    elif args.repo and args.pr:
        diff_text = subprocess.run(
            ["gh", "pr", "diff", str(args.pr), "--repo", args.repo],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    else:
        ap.error("supply --diff PATH, or owner/repo and a PR number")

    print(render(scan(diff_text), args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
