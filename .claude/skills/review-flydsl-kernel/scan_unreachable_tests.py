#!/usr/bin/env python3
"""Find tests a PR adds that the script entry point cannot reach.

Several test files here run two ways: pytest collects `test_*`, and `run_benchmark.sh` runs the
same file as a script, where only what `__main__` calls happens. Add a pytest test to such a file
and it silently does not run in the script path, so the file reports coverage it does not have.

coderfeli on #481: "These new variant tests are pytest-only today. run_benchmark.sh executes this
file as a script, but __main__ only calls test_all(), so the fused/quant variants are not
exercised in that path."

Only tests ADDED by the diff are reported. Pre-existing unreachable tests are the norm in this
repo -- an earlier version of this script flagged 6 of 7 untouched tests in one file -- and they
are not the PR's debt.

usage: scan_unreachable_tests.py --diff <diff-file> [worktree-root]
"""
import ast
import re
import sys


def calls_in(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def analyse(path):
    """-> ({tests, reached, unreachable}, None) or (None, reason)."""
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError) as e:
        return None, f"cannot parse: {e}"

    funcs = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    tests = {name for name in funcs if name.startswith("test_")}
    if not tests:
        return None, "no test functions"

    main_body = []
    for n in tree.body:
        if isinstance(n, ast.If) and "__main__" in ast.dump(n.test):
            main_body.extend(n.body)
    if not main_body:
        return None, "no __main__ block -- pytest-only file, not dual-entry"

    reached, frontier = set(), set()
    for stmt in main_body:
        frontier |= calls_in(stmt)
    while frontier:
        name = frontier.pop()
        if name in reached:
            continue
        reached.add(name)
        if name in funcs:
            frontier |= calls_in(funcs[name]) - reached

    return {"tests": sorted(tests),
            "reached": sorted(tests & reached),
            "unreachable": sorted(tests - reached)}, None


def added_tests_from_diff(diff_path):
    out, cur = {}, None
    for line in open(diff_path):
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
        elif line.startswith("+") and not line.startswith("+++") and cur and cur.endswith(".py"):
            m = re.match(r"\+\s*(?:async\s+)?def\s+(test_\w+)", line)
            if m:
                out.setdefault(cur, []).append(m.group(1))
    return out


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != "--diff":
        print(__doc__)
        return 2
    root = (argv[2] if len(argv) > 2 else ".").rstrip("/")

    added = added_tests_from_diff(argv[1])
    if not added:
        print("  no test functions added by this diff")
        return 0

    flagged = 0
    for rel, names in added.items():
        res, why = analyse(f"{root}/{rel}")
        if res is None:
            print(f"  {rel}: {why}")
            continue
        bad = [n for n in names if n in res["unreachable"]]
        if not bad:
            print(f"  {rel}: {len(names)} added test(s) reachable from __main__")
            continue
        flagged += 1
        print(f"  {rel}: {len(bad)} of {len(names)} ADDED test(s) unreachable from the script "
              f"entry point")
        for t in bad:
            print(f"      {t}")
        print(f"      __main__ reaches: {', '.join(res['reached']) or '(no tests)'}")
        print("      -> pytest runs these; `python3 <file>` does not. Wire them in, or state "
              "which job executes them.")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
