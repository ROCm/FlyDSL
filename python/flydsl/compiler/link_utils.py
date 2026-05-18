# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Helpers for backend target-attach pass options."""

import os


def _format_link_lib_options(link_libs: list) -> str:
    """Format external bitcode paths for backend attach-target passes.

    The MLIR pass-pipeline option parser treats whitespace, commas, and braces
    as structural syntax. Until FlyDSL grows proper MLIR option escaping here,
    reject such paths loudly instead of producing a malformed pipeline.
    """
    opts = []
    for lib in link_libs:
        path = os.fspath(lib)
        bad_chars = sorted({ch for ch in path if ch.isspace() or ch in ",{}\"'"})
        if not path or bad_chars:
            chars = "empty path" if not path else f"unsupported character(s) {bad_chars!r}"
            raise ValueError(
                f"Cannot pass external bitcode path {path!r} to an attach-target pass: {chars}. "
                "Use a path without whitespace, commas, braces, or quotes, or add MLIR pass-option escaping."
            )
        opts.append(f"l={path}")
    return " ".join(opts)


def _append_link_lib_options_to_attach_targets(fragments: list, link_opt: str):
    """Append link options to backend attach-target fragments."""
    new_fragments = []
    found_attach_target = False
    for fragment in fragments:
        stripped = fragment.rstrip()
        if "-attach-target" not in stripped:
            new_fragments.append(fragment)
            continue

        if "{" in stripped:
            base = stripped[:-1].rstrip() if stripped.endswith("}") else stripped
            suffix = "}" if stripped.endswith("}") else ""
            new_fragments.append(f"{base} {link_opt}{suffix}")
        else:
            new_fragments.append(f"{stripped}{{{link_opt}}}")
        found_attach_target = True
    return new_fragments, found_attach_target
