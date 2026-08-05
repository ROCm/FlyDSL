#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""List non-deprecated stable API paths under ``docs/api_stability.md``.

The collector is deliberately static: it reads export manifests and the
documentation instead of importing ``flydsl``.  This keeps it usable before
the optional target bindings have been built.  §3 deprecated APIs remain
compatible, but are intentionally excluded because that table is maintained as
a separate retirement-debt list.  The catalog lists declared namespaces and
exports; stable type entries carry their member contracts without expanding
every Python member path.  Equivalent top-level ``flydsl.expr.<name>`` aliases
are omitted in favor of their defining direct-child module paths.

Usage:
    python3 scripts/list_stable_apis.py
    python3 scripts/list_stable_apis.py --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


class StableApiCollectionError(RuntimeError):
    """The source manifests cannot be interpreted by this static collector."""


def _parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise StableApiCollectionError(f"cannot parse {path}: {exc}") from exc


def _assignment_value(tree: ast.Module, name: str, path: Path) -> ast.expr:
    value = None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value

    if value is None:
        raise StableApiCollectionError(f"{path} does not define {name}")
    return value


def _string_list(tree: ast.Module, name: str, path: Path) -> list[str]:
    result = None
    top_level_nodes = set(tree.body)
    for node in ast.walk(tree):
        if node in top_level_nodes:
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            raise StableApiCollectionError(f"{path}:{node.lineno}: {name} must be declared at module scope")

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue

        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if isinstance(node, ast.AugAssign) and not isinstance(node.op, ast.Add):
            raise StableApiCollectionError(f"{path}:{node.lineno}: {name} only supports += with a literal string list")

        value = node.value
        try:
            values = ast.literal_eval(value)
        except (TypeError, ValueError) as exc:
            raise StableApiCollectionError(f"{path}:{node.lineno}: {name} must use a literal list of strings") from exc
        if not isinstance(values, (list, tuple)) or not all(isinstance(item, str) for item in values):
            raise StableApiCollectionError(f"{path}:{node.lineno}: {name} must be a list of strings")

        if isinstance(node, ast.AugAssign):
            if result is None:
                raise StableApiCollectionError(
                    f"{path}:{node.lineno}: {name} is extended before its initial assignment"
                )
            result.extend(values)
        else:
            result = list(values)

    if result is None:
        raise StableApiCollectionError(f"{path} does not define {name}")
    return result


def _literal_value(tree: ast.Module, name: str, path: Path) -> object:
    value = _assignment_value(tree, name, path)
    try:
        return ast.literal_eval(value)
    except (TypeError, ValueError) as exc:
        raise StableApiCollectionError(f"{path}: {name} must be a literal value") from exc


def _string_mapping(tree: ast.Module, name: str, path: Path) -> dict[str, str]:
    result = _literal_value(tree, name, path)

    if not isinstance(result, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in result.items()
    ):
        raise StableApiCollectionError(f"{path}: {name} must be a mapping of strings to strings")
    return result


def _module_file(package_dir: Path, dotted_name: str) -> Path | None:
    path = package_dir.joinpath(*dotted_name.lstrip(".").split("."))
    module = path.with_suffix(".py")
    package = path / "__init__.py"
    if module.is_file():
        return module
    if package.is_file():
        return package
    return None


def _public_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _public_path(path: str) -> bool:
    return all(_public_name(part) for part in path.split("."))


def _add_path(paths: set[str], path: str) -> None:
    if _public_path(path):
        paths.add(path)


def _star_imported_modules(expr_init: Path) -> list[str]:
    modules = []
    for node in _parse_python(expr_init).body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
            continue
        if any(alias.name == "*" for alias in node.names):
            modules.append(node.module)
    return modules


def _collect_backend_exports(
    module_file: Path,
    public_path: str,
    paths: set[str],
    seen: set[tuple[Path, str]],
) -> None:
    key = (module_file.resolve(), public_path)
    if key in seen:
        return
    seen.add(key)

    _add_path(paths, public_path)
    module_dir = module_file.parent
    for name in _string_list(_parse_python(module_file), "__all__", module_file):
        if not _public_name(name):
            continue
        child_public_path = f"{public_path}.{name}"
        _add_path(paths, child_public_path)

        child_module = _module_file(module_dir, name)
        if child_module is not None:
            _collect_backend_exports(child_module, child_public_path, paths, seen)


def _collect_expr_paths(repo_root: Path, paths: set[str]) -> None:
    expr_dir = repo_root / "python" / "flydsl" / "expr"
    expr_init = expr_dir / "__init__.py"
    expr_tree = _parse_python(expr_init)

    _add_path(paths, "flydsl.expr")
    for module_name in _star_imported_modules(expr_init):
        module_file = _module_file(expr_dir, module_name)
        if module_file is None:
            raise StableApiCollectionError(f"{expr_init}: cannot resolve direct child module {module_name!r}")

        module_path = f"flydsl.expr.{module_name}"
        _add_path(paths, module_path)
        for name in _string_list(_parse_python(module_file), "__all__", module_file):
            if not _public_name(name):
                continue
            _add_path(paths, f"{module_path}.{name}")

    for public_name, target in _string_mapping(expr_tree, "_BACKEND_MODULES", expr_init).items():
        if not _public_name(public_name):
            continue
        module_file = _module_file(expr_dir, target)
        if module_file is None:
            raise StableApiCollectionError(f"{expr_init}: cannot resolve lazy backend {target!r}")
        _collect_backend_exports(module_file, f"flydsl.expr.{public_name}", paths, set())


def _collect_compiler_paths(repo_root: Path, paths: set[str]) -> None:
    compiler_dir = repo_root / "python" / "flydsl" / "compiler"
    compiler_init = compiler_dir / "__init__.py"
    protocol = compiler_dir / "protocol.py"

    _add_path(paths, "flydsl.compiler")
    for name in _string_list(_parse_python(compiler_init), "__all__", compiler_init):
        _add_path(paths, f"flydsl.compiler.{name}")

    _add_path(paths, "flydsl.compiler.protocol")
    for name in _string_list(_parse_python(protocol), "__all__", protocol):
        _add_path(paths, f"flydsl.compiler.protocol.{name}")


def _markdown_section(document: str, heading: str) -> str:
    start = document.find(heading)
    if start < 0:
        raise StableApiCollectionError(f"docs/api_stability.md does not contain {heading!r}")
    remainder = document[start + len(heading) :]
    next_heading = re.search(r"^#{1,3} ", remainder, flags=re.MULTILINE)
    return remainder[: next_heading.start()] if next_heading else remainder


def _table_api_paths(
    section: str,
    source_prefix: str | None = None,
    target_prefix: str | None = None,
) -> list[str]:
    paths = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        api_column = line.split("|", 2)[1]
        for path in re.findall(r"`([^`]+)`", api_column):
            if source_prefix is not None:
                if not path.startswith(source_prefix):
                    continue
                path = f"{target_prefix}{path[len(source_prefix) :]}"
            if path.startswith("flydsl."):
                paths.append(path)
    return paths


def _collect_documented_paths(repo_root: Path, paths: set[str]) -> None:
    document = (repo_root / "docs" / "api_stability.md").read_text()

    for path in _table_api_paths(_markdown_section(document, "### 2.3")):
        _add_path(paths, path)


def _deprecated_exclusions(repo_root: Path) -> tuple[set[str], set[str]]:
    expr_dir = repo_root / "python" / "flydsl" / "expr"
    expr_init = expr_dir / "__init__.py"
    expr_tree = _parse_python(expr_init)

    direct_export_aliases: dict[str, set[str]] = {}
    for module_name in _star_imported_modules(expr_init):
        module_file = _module_file(expr_dir, module_name)
        if module_file is None:
            raise StableApiCollectionError(f"{expr_init}: cannot resolve direct child module {module_name!r}")
        for name in _string_list(_parse_python(module_file), "__all__", module_file):
            if _public_name(name):
                direct_export_aliases.setdefault(name, set()).update(
                    {
                        f"flydsl.expr.{name}",
                        f"flydsl.expr.{module_name}.{name}",
                    }
                )

    lazy_backends = set(_string_mapping(expr_tree, "_BACKEND_MODULES", expr_init))
    document = (repo_root / "docs" / "api_stability.md").read_text()
    deprecated_paths = _table_api_paths(
        _markdown_section(document, "## 3."),
        source_prefix="fx.",
        target_prefix="flydsl.expr.",
    )

    exact: set[str] = set()
    prefixes: set[str] = set()
    for path in deprecated_paths:
        parts = path.split(".")
        if parts[:2] != ["flydsl", "expr"] or len(parts) == 2:
            exact.add(path)
            continue

        name, *member_path = parts[2:]
        if not member_path and name in lazy_backends:
            prefixes.add(path)
            continue

        for alias in direct_export_aliases.get(name, {f"flydsl.expr.{name}"}):
            exact.add(".".join((alias, *member_path)))

    return exact, prefixes


def collect_stable_api_paths(repo_root: Path) -> list[str]:
    """Return sorted, non-deprecated stable API paths from the current source.

    Direct-child exports use ``flydsl.expr.<module>.<name>`` as their canonical
    path; their equivalent top-level ``flydsl.expr.<name>`` aliases are omitted.
    §3 paths are deliberately omitted: they remain stable for compatibility but
    belong to the separate deprecated API list.
    """

    paths: set[str] = set()
    _collect_expr_paths(repo_root, paths)
    _collect_compiler_paths(repo_root, paths)
    _collect_documented_paths(repo_root, paths)
    exact_exclusions, prefix_exclusions = _deprecated_exclusions(repo_root)
    paths.difference_update(exact_exclusions)
    paths = {
        path
        for path in paths
        if not any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefix_exclusions)
    }
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="FlyDSL repository root (default: inferred from this script)",
    )
    parser.add_argument("--format", choices=("lines", "json"), default="lines", help="Output format (default: lines)")
    args = parser.parse_args(argv)

    try:
        paths = collect_stable_api_paths(args.repo_root.resolve())
    except StableApiCollectionError as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(json.dumps(paths, indent=2))
    else:
        print("\n".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
