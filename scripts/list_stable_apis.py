#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""List or compare stable API paths under ``docs/api_stability.md``.

The collector is deliberately static: it reads export manifests and the
documentation instead of importing ``flydsl``.  This keeps it usable before
the optional target bindings have been built.  §3 deprecated APIs remain
compatible, but are excluded by default because that table is maintained as
a separate retirement-debt list; use --include-deprecated for release review.
The catalog lists declared namespaces and exports; stable type entries carry
their member contracts without expanding every Python member path.
Equivalent top-level ``flydsl.expr.<name>`` aliases
are omitted in favor of their defining direct-child module paths. Extension
library aliases use their canonical ``flydsl.extension`` paths.

Comparison reads two committed Git snapshots and checks old_paths <= new_paths.
It includes deprecated exports and equivalent public aliases. It does not
check signatures or semantics. Exit codes: 0 for inclusion, 1 for missing old
paths, and 2 for invalid arguments or unreadable source.

Usage:
    python3 scripts/list_stable_apis.py
    python3 scripts/list_stable_apis.py --format json
    python3 scripts/list_stable_apis.py --include-deprecated --format json
    python3 scripts/list_stable_apis.py --old v0.3.3 --new HEAD --format json
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
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


def _export_list_value(tree: ast.Module, value: ast.expr, path: Path, active: frozenset[Path]) -> list[str]:
    """Read literal exports and compositions of relative modules' __all__ lists."""
    if isinstance(value, (ast.List, ast.Tuple)):
        result = []
        for item in value.elts:
            if isinstance(item, ast.Starred):
                result.extend(_export_list_value(tree, item.value, path, active))
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                result.append(item.value)
            else:
                break
        else:
            return result
    elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        return _export_list_value(tree, value.left, path, active) + _export_list_value(tree, value.right, path, active)
    elif (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"list", "tuple"}
        and len(value.args) == 1
        and not value.keywords
    ):
        return _export_list_value(tree, value.args[0], path, active)
    elif isinstance(value, ast.Attribute) and value.attr == "__all__" and isinstance(value.value, ast.Name):
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            for alias in node.names:
                if (alias.asname or alias.name) != value.value.id:
                    continue
                target = "." * node.level + ".".join(filter(None, (node.module, alias.name)))
                module_file = _module_file(path.parent, target)
                if module_file is not None:
                    return _string_list(_parse_python(module_file), "__all__", module_file, active)
    raise StableApiCollectionError(f"{path}:{value.lineno}: unsupported static export list: {ast.unparse(value)}")


def _string_list(tree: ast.Module, name: str, path: Path, active: frozenset[Path] = frozenset()) -> list[str]:
    if path.resolve() in active:
        raise StableApiCollectionError(f"{path}: cyclic {name} reference")
    active = active | {path.resolve()}
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
        if name == "__all__":
            values = _export_list_value(tree, value, path, active)
        else:
            try:
                values = ast.literal_eval(value)
            except (TypeError, ValueError) as exc:
                raise StableApiCollectionError(
                    f"{path}:{node.lineno}: {name} must use a literal list of strings"
                ) from exc
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
    level = len(dotted_name) - len(dotted_name.lstrip("."))
    for _ in range(max(0, level - 1)):
        package_dir = package_dir.parent
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


def _public_path(path: str, *, exclude_experimental: bool = True) -> bool:
    return all(_public_name(part) and (not exclude_experimental or part != "experimental") for part in path.split("."))


def _add_path(paths: set[str], path: str) -> None:
    # Experimental exclusion depends on the policy in the inspected revision.
    if _public_path(path, exclude_experimental=False):
        paths.add(path)


def _star_imported_modules(expr_init: Path) -> list[str]:
    modules = []
    for node in _parse_python(expr_init).body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
            continue
        if any(alias.name == "*" for alias in node.names):
            modules.append(node.module)
    return modules


def _collect_namespace_exports(
    module_file: Path,
    public_path: str,
    paths: set[str],
    seen: set[tuple[Path, str]],
    exclude_experimental: bool = True,
) -> None:
    if not _public_path(public_path, exclude_experimental=exclude_experimental):
        return
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
            _collect_namespace_exports(child_module, child_public_path, paths, seen, exclude_experimental)


def _collect_expr_paths(repo_root: Path, paths: set[str], exclude_experimental: bool) -> None:
    expr_dir = repo_root / "python" / "flydsl" / "expr"
    expr_init = expr_dir / "__init__.py"
    expr_tree = _parse_python(expr_init)

    _add_path(paths, "flydsl.expr")
    for module_name in _star_imported_modules(expr_init):
        module_path = f"flydsl.expr.{module_name}"
        if not _public_path(module_path, exclude_experimental=exclude_experimental):
            continue
        module_file = _module_file(expr_dir, module_name)
        if module_file is None:
            raise StableApiCollectionError(f"{expr_init}: cannot resolve direct child module {module_name!r}")

        _add_path(paths, module_path)
        for name in _string_list(_parse_python(module_file), "__all__", module_file):
            if not _public_name(name):
                continue
            _add_path(paths, f"{module_path}.{name}")

    for public_name, target in _string_mapping(expr_tree, "_BACKEND_MODULES", expr_init).items():
        if not _public_path(f"flydsl.expr.{public_name}", exclude_experimental=exclude_experimental):
            continue
        module_file = _module_file(expr_dir, target)
        if module_file is None:
            raise StableApiCollectionError(f"{expr_init}: cannot resolve lazy backend {target!r}")
        _collect_namespace_exports(module_file, f"flydsl.expr.{public_name}", paths, set(), exclude_experimental)


def _library_modules(repo_root: Path) -> dict[str, tuple[Path, str]]:
    document = (repo_root / "docs" / "api_stability.md").read_text()
    if not re.search(r"^### \d+\.\d+ `flydsl\.extension`\s*$", document, re.MULTILINE):
        return {}  # Releases before the extension stability commitment.
    expr_init = repo_root / "python" / "flydsl" / "expr" / "__init__.py"
    tree = _parse_python(expr_init)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    manifest = "_EXTENSION_MODULES" if "_EXTENSION_MODULES" in names else "_LIBRARY_MODULES"
    libraries = {}
    for alias, target in _string_mapping(tree, manifest, expr_init).items():
        module_file = _module_file(expr_init.parent, target)
        if module_file is None:
            raise StableApiCollectionError(f"{expr_init}: cannot resolve extension library {target!r}")
        module_path = module_file.relative_to(repo_root / "python").with_suffix("")
        if module_path.name == "__init__":
            module_path = module_path.parent
        public_path = ".".join(module_path.parts)
        if not public_path.startswith("flydsl.extension."):
            raise StableApiCollectionError(f"{expr_init}: library {target!r} must be under flydsl.extension")
        libraries[alias] = (module_file, public_path)
    return libraries


def _collect_extension_paths(repo_root: Path, paths: set[str], exclude_experimental: bool) -> None:
    libraries = _library_modules(repo_root)
    _add_path(paths, "flydsl.extension")
    for module_file, public_path in libraries.values():
        _collect_namespace_exports(module_file, public_path, paths, set(), exclude_experimental)


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

    heading = re.search(r"^### \d+\.\d+ Other explicitly stable APIs\s*$", document, re.MULTILINE)
    if heading is None:
        raise StableApiCollectionError("docs/api_stability.md has no supported explicit stable API table")
    for path in _table_api_paths(_markdown_section(document, heading.group())):
        _add_path(paths, path)


def _deprecated_exclusions(repo_root: Path, exclude_experimental: bool) -> tuple[set[str], set[str]]:
    expr_dir = repo_root / "python" / "flydsl" / "expr"
    expr_init = expr_dir / "__init__.py"
    expr_tree = _parse_python(expr_init)

    direct_export_aliases: dict[str, set[str]] = {}
    for module_name in _star_imported_modules(expr_init):
        if not _public_path(f"flydsl.expr.{module_name}", exclude_experimental=exclude_experimental):
            continue
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
    libraries = _library_modules(repo_root)
    document = (repo_root / "docs" / "api_stability.md").read_text()
    deprecated_section = _markdown_section(document, "## 3.")
    deprecated_paths = _table_api_paths(
        deprecated_section,
        source_prefix="fx.",
        target_prefix="flydsl.expr.",
    )
    deprecated_paths.extend(_table_api_paths(deprecated_section))

    exact: set[str] = set()
    prefixes: set[str] = set()
    for path in deprecated_paths:
        if path.startswith("flydsl.extension."):
            prefixes.add(path)
            continue
        parts = path.split(".")
        if parts[:2] != ["flydsl", "expr"] or len(parts) == 2:
            exact.add(path)
            continue

        name, *member_path = parts[2:]
        if name in libraries:
            prefixes.add(".".join((libraries[name][1], *member_path)))
            continue
        if not member_path and name in lazy_backends:
            prefixes.add(path)
            continue

        for alias in direct_export_aliases.get(name, {f"flydsl.expr.{name}"}):
            exact.add(".".join((alias, *member_path)))

    return exact, prefixes


def collect_stable_api_paths(
    repo_root: Path, *, include_deprecated: bool = False, include_aliases: bool = False
) -> list[str]:
    """Return sorted stable API paths from the current source.

    Direct-child exports use ``flydsl.expr.<module>.<name>`` as their canonical
    path; their equivalent top-level ``flydsl.expr.<name>`` aliases are omitted.
    Extension aliases use canonical ``flydsl.extension`` paths. §3 paths are
    omitted by default; include_deprecated retains exported deprecated APIs,
    but does not add documented names absent from the source manifests.
    include_aliases expands equivalent public access paths for comparison.
    """

    document = (repo_root / "docs" / "api_stability.md").read_text()
    exclude_experimental = "`experimental`" in _markdown_section(document, "## 2.")
    paths: set[str] = set()
    _collect_expr_paths(repo_root, paths, exclude_experimental)
    _collect_compiler_paths(repo_root, paths)
    if re.search(r"^### \d+\.\d+ `flydsl\.extension`\s*$", document, re.MULTILINE):
        _collect_extension_paths(repo_root, paths, exclude_experimental)
    _collect_documented_paths(repo_root, paths)
    if not include_deprecated:
        exact_exclusions, prefix_exclusions = _deprecated_exclusions(repo_root, exclude_experimental)
        paths.difference_update(exact_exclusions)
        paths = {
            path
            for path in paths
            if not any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefix_exclusions)
        }
    if include_aliases:
        aliases = {
            f"flydsl.expr.{module}": "flydsl.expr"
            for module in _star_imported_modules(repo_root / "python/flydsl/expr/__init__.py")
        }
        for prefix, alias in aliases.items():
            paths.update(alias + path[len(prefix) :] for path in tuple(paths) if path.startswith(prefix + "."))
        for alias, (_, prefix) in _library_modules(repo_root).items():
            paths.update(
                f"flydsl.expr.{alias}" + path[len(prefix) :]
                for path in tuple(paths)
                if path == prefix or path.startswith(prefix + ".")
            )
    paths = {path for path in paths if _public_path(path, exclude_experimental=exclude_experimental)}
    return sorted(paths)


def _git(repo_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(repo_root), *arguments], check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise StableApiCollectionError(exc.stderr.decode(errors="replace").strip()) from exc
    except OSError as exc:
        raise StableApiCollectionError(f"cannot run git: {exc}") from exc


def _revision_paths(repo_root: Path, commit: str) -> set[str]:
    archive = _git(repo_root, "archive", "--format=tar", commit, "python/flydsl", "docs/api_stability.md")
    # Plain source snapshots, not worktrees: no checkout or Git metadata is changed.
    with tempfile.TemporaryDirectory(prefix="flydsl-api-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for entry in source:
                if entry.isdir():
                    continue
                path = Path(entry.name)
                if not entry.isfile() or path.is_absolute() or ".." in path.parts:
                    raise StableApiCollectionError(f"unsupported source archive entry: {entry.name}")
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(entry) as contents:
                    target.write_bytes(contents.read())
        return set(collect_stable_api_paths(root, include_deprecated=True, include_aliases=True))


def compare_stable_api_paths(repo_root: Path, old: str, new: str) -> dict:
    """Check exact old/new revisions in the caller's direction, including aliases."""
    endpoints = []
    inventories = []
    for label, revision in (("old", old), ("new", new)):
        try:
            commit = _git(repo_root, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
            sha = commit.decode().strip()
            paths = _revision_paths(repo_root, sha)
        except (StableApiCollectionError, OSError, tarfile.TarError) as exc:
            raise StableApiCollectionError(f"{label} revision {revision!r}: {exc}") from exc
        endpoints.append({"ref": revision, "commit": sha, "count": len(paths)})
        inventories.append(paths)
    old_paths, new_paths = inventories
    return {
        "old": endpoints[0],
        "new": endpoints[1],
        "is_superset": old_paths <= new_paths,
        "added": sorted(new_paths - old_paths),
        "removed": sorted(old_paths - new_paths),
        "retained": sorted(old_paths & new_paths),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="FlyDSL repository root (default: inferred from this script)",
    )
    parser.add_argument("--format", choices=("lines", "json"), default="lines", help="Output format (default: lines)")
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated APIs when listing (always on for comparison)",
    )
    parser.add_argument("--old", help="Old version: commit SHA, tag, or other commit reference")
    parser.add_argument("--new", help="New version: commit SHA, tag, or other commit reference; checked against --old")
    args = parser.parse_args(argv)
    if (args.old is None) != (args.new is None):
        parser.error("--old and --new must be specified together")

    try:
        if args.old is not None:
            result = compare_stable_api_paths(args.repo_root.resolve(), args.old, args.new)
            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                for label in ("old", "new"):
                    endpoint = result[label]
                    print(f"{label}: {endpoint['ref']} ({endpoint['commit']}), {endpoint['count']} paths")
                status = "PASS" if result["is_superset"] else "FAIL"
                print(f"{status}: new stable API paths contain all old paths = {result['is_superset']}")
                print(
                    f"Retained: {len(result['retained'])}; added: {len(result['added'])}; removed: {len(result['removed'])}"
                )
                for key, marker in (("removed", "-"), ("added", "+")):
                    for path in result[key]:
                        print(f"{marker} {path}")
                print("Path inclusion only; signatures, semantics, and retirement windows require review.")
            return 0 if result["is_superset"] else 1
        paths = collect_stable_api_paths(args.repo_root.resolve(), include_deprecated=args.include_deprecated)
    except (StableApiCollectionError, OSError) as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(json.dumps(paths, indent=2))
    else:
        print("\n".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
