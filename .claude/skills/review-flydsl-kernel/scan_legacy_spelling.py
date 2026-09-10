#!/usr/bin/env python3
"""List legacy-spelling review candidates on added kernel consumer lines.

Only kernels/**/*.py is checked, excluding kernels/common/buffer_ops.py. These
spelling matches require source review; in particular, ordinary make_ptr pointer
construction is valid. Comments, strings, and imports visible in each diff hunk
are ignored. A hunk starting inside a string or import whose opening is omitted
does not provide enough lexical context; check the full source in those cases.

Exit status: 0 = no candidates, 1 = candidates, 2 = input or tool failure.
"""

import argparse
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

RULES = [
    (
        "raw ir.* / ArithValue",
        r"(?<![\w.])ir\.[A-Za-z_]|\b_mlir\.|\bArithValue\b|\barith\.unwrap\b|\bas_ir_value\b",
        "check whether typed fx values (fx.Float32 / fx.Int32, expr/numeric.py) can replace raw IR values",
        "coderfeli #202 #250 #300 #326 #426 #850",
    ),
    (
        "scf.* control flow",
        r"(?<![\w.])scf\.(?:If|For|While|Yield)(?:Op)?\b",
        "prefer ordinary Python if/for inside the kernel; scf.* is a lowering detail",
        "coderfeli #33 #433 #540 #582",
    ),
    (
        "buffer_ops.*",
        r"(?<![\w.])buffer_ops\.",
        "check whether fx.copy / a copy atom can express this kernel memory operation",
        "coderfeli #404 #416 #894 #1032",
    ),
    (
        "SmemAllocator",
        r"\bSmemAllocator\b",
        "prefer SharedAllocator for new kernels",
        "sjfeng1999 #549 #567",
    ),
    (
        "make_ptr (check retyping)",
        r"\bmake_ptr\s*\(",
        "if this only retypes an existing pointer, use recast_iter; ordinary pointer construction is valid",
        "sjfeng1999 #288 #745",
    ),
]
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


def get_diff(args):
    if args.diff is not None:
        return args.diff.read_text(encoding="utf-8")
    return subprocess.run(
        ["gh", "pr", "diff", str(args.pr), "--repo", args.repo],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def kernel_hunks(diff):
    """Yield new-side hunk rows as (line number, is added, source)."""
    path = None
    rows = []
    old_left = new_left = 0
    next_line = None
    saw_header = False
    have_new_header = False
    for diff_line, line in enumerate(diff.splitlines(), 1):
        if line == r"\ No newline at end of file":
            continue
        if old_left or new_left:
            prefix = line[:1]
            if prefix not in {" ", "+", "-"}:
                raise ValueError(f"invalid or truncated hunk at diff line {diff_line}")
            if prefix in {" ", "-"}:
                old_left -= 1
            if prefix in {" ", "+"}:
                new_left -= 1
                rows.append((next_line, prefix == "+", line[1:]))
                next_line += 1
            if old_left < 0 or new_left < 0:
                raise ValueError(f"hunk line count mismatch at diff line {diff_line}")
            if not old_left and not new_left:
                if path and path.startswith("kernels/") and path.endswith(".py"):
                    if path != "kernels/common/buffer_ops.py":
                        yield path, rows
                rows = []
            continue
        if line.startswith("diff --git ") or line.startswith("--- "):
            path = None
            saw_header = True
            have_new_header = False
        elif line.startswith("+++ "):
            new_path = line[4:].split("\t", 1)[0]
            if new_path.startswith('"'):
                raise ValueError("quoted git paths are not supported; provide a diff with unquoted paths")
            if not new_path.startswith("b/") and new_path != "/dev/null":
                raise ValueError(f"expected a b/ path or /dev/null at diff line {diff_line}")
            path = new_path[2:] if new_path.startswith("b/") else None
            saw_header = True
            have_new_header = True
        elif line.startswith("@@"):
            match = HUNK.fullmatch(line)
            if match is None or not have_new_header:
                raise ValueError(f"invalid hunk header at diff line {diff_line}")
            old_left = int(match.group(2) or "1")
            next_line = int(match.group(3))
            new_left = int(match.group(4) or "1")
        elif line.startswith("+"):
            raise ValueError(f"added line outside a hunk at diff line {diff_line}")
    if old_left or new_left:
        raise ValueError("truncated diff hunk")
    if diff.strip() and not saw_header:
        raise ValueError("expected a unified git diff")


def code_lines(rows):
    """Keep code tokens; hunk fragments need not be complete Python statements."""
    # Indentation is irrelevant to spelling matches, and a hunk can start midway
    # through a block. Removing it avoids unrelated tokenizer indentation errors.
    source = [row[2].lstrip() for row in rows]
    visible = [[" "] * len(line) for line in source]
    statement_start = True
    importing = False
    try:
        for token in tokenize.generate_tokens(io.StringIO("\n".join(source) + "\n").readline):
            if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.ENDMARKER}:
                continue
            if token.type == tokenize.NEWLINE or (token.type == tokenize.OP and token.string == ";"):
                statement_start = True
                importing = False
                continue
            if token.type == tokenize.COMMENT:
                continue
            if statement_start and token.type == tokenize.NAME and token.string in {"from", "import"}:
                importing = True
            statement_start = False
            if importing or token.type == tokenize.STRING:
                continue
            line, start = token.start
            _, end = token.end
            visible[line - 1][start:end] = token.string
    except tokenize.TokenError:
        # Partial statements and strings are normal at the end of a diff hunk.
        # A string token is emitted only when closed, so its contents stay blank.
        pass
    return ["".join(line) for line in visible]


def scan(diff):
    hits = []
    for path, rows in kernel_hunks(diff):
        for (line, added, original), code in zip(rows, code_lines(rows)):
            if not added:
                continue
            for name, pattern, advice, provenance in RULES:
                if re.search(pattern, code):
                    hits.append((path, line, name, original.strip()[:100], advice, provenance))
    return hits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", help="GitHub OWNER/REPO")
    parser.add_argument("pr", nargs="?", type=int, help="positive pull request number")
    parser.add_argument("--diff", type=Path, metavar="FILE", help="read an offline unified git diff")
    args = parser.parse_args(argv)
    if args.diff is not None:
        if args.repo is not None or args.pr is not None:
            parser.error("--diff FILE cannot be combined with OWNER/REPO or PR")
    elif args.repo is None or args.pr is None:
        parser.error("provide OWNER/REPO and PR, or --diff FILE")
    elif not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo) or args.pr <= 0:
        parser.error("provide a valid OWNER/REPO and a positive PR number")

    try:
        hits = scan(get_diff(args))
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "no diagnostic from gh"
        print(f"error: gh pr diff failed (exit {exc.returncode}): {detail}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not hits:
        print("no legacy spelling candidates on added kernel lines")
        return 0
    print(f"== legacy spelling candidates on added kernel lines: {len(hits)} ==")
    groups = {}
    for hit in hits:
        groups.setdefault((hit[0], hit[2]), []).append(hit)
    for matches in groups.values():
        path, line, name, code, advice, provenance = matches[0]
        print(
            f"  {path}:{line}: {name} x{len(matches)}\n      {advice}\n      e.g. {code}\n      maintainers: {provenance}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
