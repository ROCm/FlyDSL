#!/usr/bin/env python3
"""Find added tests without a statically visible script entry path.

Read the head source and a unified diff; never import or execute reviewed code.
Check module-level test_* definitions and methods in Test* classes whose def
line is added by the diff, using qualified names and line numbers. Signature
edits can therefore produce candidates; edits only to an existing body do not.
Follow direct calls to unambiguous
module functions from an exact ``__name__ == "__main__"`` guard, and recognize
pytest.main([__file__]) (including import aliases and harmless reporting flags).

This is a review aid, not a Python interpreter: dynamic dispatch, pytest
selectors/unknown arguments, runtime branches, decorators, configuration and
plugins need review. A visible path does not guarantee runtime execution.
Exit 0: no candidates; 1: coverage candidates/manual review; 2: invalid input.

usage: scan_unreachable_tests.py --diff <diff-file> --head <worktree-root>
"""

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path

_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_REPORTING_FLAGS = {
    "-q",
    "-qq",
    "--quiet",
    "-v",
    "-vv",
    "--verbose",
    "-s",
    "-ra",
    "-rA",
    "--disable-warnings",
    "--tb=short",
    "--tb=long",
    "--tb=no",
    "--color=yes",
    "--color=no",
    "--color=auto",
    "--capture=no",
}


def added_lines_from_diff(diff):
    """Map Python head paths to (added line numbers, expected head lines)."""
    files = {}
    path = None
    old_left = new_left = 0
    next_line = None
    saw_header = False
    has_head = False
    for line in diff.splitlines():
        if old_left or new_left:
            prefix = line[:1]
            if line == "\\ No newline at end of file":
                continue
            if prefix not in {"+", "-", " "}:
                raise ValueError("incomplete or malformed diff hunk")
            if prefix != "+":
                old_left -= 1
            if prefix != "-":
                new_left -= 1
                if path is not None:
                    added, expected = files[path]
                    expected[next_line] = line[1:]
                    if prefix == "+":
                        added.add(next_line)
                next_line += 1
            if old_left < 0 or new_left < 0:
                raise ValueError("diff hunk exceeds its declared line counts")
            continue
        if line.startswith("diff --git "):
            path = None
            saw_header = True
            has_head = False
        elif line.startswith("+++ "):
            name = line[4:].split("\t", 1)[0]
            saw_header = True
            has_head = True
            if name == "/dev/null":
                path = None
            elif name.startswith("b/"):
                candidate = Path(name[2:])
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise ValueError(f"unsafe head path: {name}")
                path = candidate if candidate.suffix == ".py" else None
                if path is not None:
                    files.setdefault(path, (set(), {}))
            else:
                raise ValueError(f"expected unquoted b/ head path, got: {name}")
        elif line.startswith("@@"):
            match = _HUNK.match(line)
            if not match or not has_head:
                raise ValueError("malformed unified diff hunk header")
            old_left = int(match[2]) if match[2] is not None else 1
            next_line = int(match[3])
            new_left = int(match[4]) if match[4] is not None else 1
        elif line.startswith("--- ") or line == "\\ No newline at end of file":
            continue
        elif line.startswith(("+", "-", " ")):
            raise ValueError("diff content outside a hunk")
    if old_left or new_left:
        raise ValueError("incomplete diff hunk")
    if diff.strip() and not saw_header:
        raise ValueError("expected a unified diff")
    return files


def scope_nodes(statements):
    """Walk one scope, without treating uncalled nested definitions as calls."""
    for node in statements:
        yield node
        if not isinstance(node, (*_FUNCTIONS, ast.ClassDef, ast.Lambda)):
            yield from scope_nodes(ast.iter_child_nodes(node))


def bound_names(nodes, include_functions=True):
    names = set()
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ClassDef) or include_functions and isinstance(node, _FUNCTIONS):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def pytest_aliases(nodes, inherited=None, parameters=()):
    aliases = dict(inherited or {})
    imports = {}
    non_imports = [node for node in nodes if not isinstance(node, (ast.Import, ast.ImportFrom))]
    shadowed = bound_names(non_imports) | set(parameters)
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    imports[alias.asname or "pytest"] = "module"
                else:
                    shadowed.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module == "pytest" and not node.level and alias.name == "main":
                    imports[alias.asname or "main"] = "main"
                else:
                    shadowed.add(alias.asname or alias.name)
    # Assignments/parameters/other imports cannot be assumed to retain an alias.
    for name in bound_names(nodes) | set(parameters):
        aliases.pop(name, None)
    aliases.update({name: kind for name, kind in imports.items() if name not in shadowed})
    return aliases


def is_main_guard(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    return any(
        isinstance(name, ast.Name)
        and name.id == "__name__"
        and isinstance(value, ast.Constant)
        and value.value == "__main__"
        for name, value in ((left, right), (right, left))
    )


def test_definitions(body, prefix=""):
    for node in body:
        if isinstance(node, _FUNCTIONS) and node.name.startswith("test_"):
            yield (prefix + node.name, node.lineno)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            yield from test_definitions(node.body, prefix + node.name + ".")


def full_file_pytest(call):
    """Accept an explicit whole-file target with only known reporting flags."""
    if len(call.args) == 1 and not call.keywords:
        args = call.args[0]
    elif not call.args and len(call.keywords) == 1 and call.keywords[0].arg == "args":
        args = call.keywords[0].value
    else:
        return False
    if not isinstance(args, (ast.List, ast.Tuple)):
        return False
    file_count = 0
    for arg in args.elts:
        if isinstance(arg, ast.Name) and arg.id == "__file__":
            file_count += 1
        elif not isinstance(arg, ast.Constant) or arg.value not in _REPORTING_FLAGS:
            return False
    return file_count == 1


def analyse(tree):
    tests = set(test_definitions(tree.body))
    test_names = Counter(name for name, _ in tests)
    pytest_tests = {test for test in tests if test_names[test[0]] == 1}
    main_body = [stmt for node in tree.body if is_main_guard(node) for stmt in node.body]
    if not main_body:
        return tests, set(), False, False

    module_nodes = list(scope_nodes(tree.body))
    definitions = [node for node in tree.body if isinstance(node, _FUNCTIONS)]
    counts = Counter(node.name for node in definitions)
    rebound = bound_names(module_nodes, include_functions=False)
    funcs = {node.name: node for node in definitions if counts[node.name] == 1 and node.name not in rebound}
    module_aliases = pytest_aliases(module_nodes)
    reached, visited = set(), set()
    unknown_pytest = False
    pending = [(main_body, ())]
    while pending:
        body, parameters = pending.pop()
        nodes = list(scope_nodes(body))
        shadowed = bound_names(nodes) | set(parameters)
        aliases = pytest_aliases(nodes, module_aliases, parameters)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            is_pytest = (
                isinstance(target, ast.Name)
                and aliases.get(target.id) == "main"
                or isinstance(target, ast.Attribute)
                and target.attr == "main"
                and isinstance(target.value, ast.Name)
                and aliases.get(target.value.id) == "module"
            )
            if is_pytest:
                if full_file_pytest(node) and "__file__" not in rebound | shadowed:
                    reached.update(pytest_tests)
                else:
                    unknown_pytest = True
            if not isinstance(target, ast.Name) or target.id in shadowed or target.id not in funcs:
                continue
            func = funcs[target.id]
            # Calling async/generator functions does not execute their bodies.
            # Await/iteration is outside the direct synchronous call graph.
            if (
                isinstance(func, ast.AsyncFunctionDef)
                or func.name in visited
                or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in scope_nodes(func.body))
            ):
                continue
            visited.add(func.name)
            reached.add((func.name, func.lineno))
            args = func.args
            parameters = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            parameters += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
            pending.append((func.body, parameters))
    return tests, reached, True, unknown_pytest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path, metavar="DIR")
    args = parser.parse_args(argv)
    try:
        root = args.head.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"worktree root is not a directory: {root}")
        files = added_lines_from_diff(args.diff.read_text())
        results = []
        for rel, (added, expected) in files.items():
            path = (root / rel).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"head path is outside worktree root: {rel}")
            source = path.read_text()
            lines = source.splitlines()
            for line, content in expected.items():
                if line < 1 or line > len(lines) or lines[line - 1] != content:
                    raise ValueError(f"{rel}:{line}: diff does not match head source")
            tests, reached, dual_entry, unknown = analyse(ast.parse(source, filename=str(rel)))
            new_tests = {test for test in tests if test[1] in added}
            if new_tests:
                results.append((rel, new_tests, reached, dual_entry, unknown))
    except (OSError, UnicodeError, SyntaxError, ValueError) as error:
        print(f"  input error: {error}", file=sys.stderr)
        return 2

    flagged = False
    for rel, new_tests, reached, dual_entry, unknown in results:
        if not dual_entry:
            print(f"  {rel}: no supported __main__ guard; script coverage not assessed")
            continue
        missing = new_tests - reached
        if not missing:
            print(f"  {rel}: {len(new_tests)} added test definition line(s) have a statically visible __main__ path")
            continue
        flagged = True
        print(f"  {rel}: {len(missing)} of {len(new_tests)} ADDED test definition line(s) need manual review")
        for name, line in sorted(missing, key=lambda test: test[1]):
            print(f"      {name}:{line}: no statically supported entry path found")
        if unknown:
            print("      pytest invocation uses selectors or unknown arguments; whole-file coverage is unknown")
        print("      -> Check script wiring or identify the pytest job that exercises these tests.")
    if not results:
        print("  no test definitions added by this diff")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
