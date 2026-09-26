# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Configuration file for the Sphinx documentation builder.

import os
import subprocess
import sys
from datetime import datetime, timezone

# -- Path setup --------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "python"))

import flydsl  # noqa: E402

version = release = flydsl.__version__

# Identify the source revision without importing GPU-dependent compiler modules.
try:
    _revision = (
        subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_project_root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
except OSError:
    # Source archive builds may not have Git installed.
    _revision = "unknown"
_built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# -- Project information -----------------------------------------------------
project = "FlyDSL"
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
author = "Advanced Micro Devices, Inc."

# -- General configuration ---------------------------------------------------
extensions = [
    "rocm_docs",
    "sphinx_autodoc_typehints",
]
external_toc_path = "./sphinx/_toc.yml"
# This repository does not use intersphinx references to other ROCm projects.
# Use the rocm-docs-core bundled project metadata and avoid network access so
# local and pull-request builds remain deterministic and do not depend on the
# GitHub API rate limit. FlyDSL is not present in every supported bundled
# registry version, so use its parent documentation family for version context.
external_projects_current_project = "ai-ecosystem"
external_projects_remote_repository = ""
external_projects = []
# Generate llms.txt
rocm_docs_generate_llms = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Options for HTML output -------------------------------------------------
html_title = f"FlyDSL {version}"
html_theme = "rocm_docs_theme"
html_static_path = ["_static"]
html_theme_options = {
    "flavor": "ai-ecosystem",
    "link_main_doc": True,
    "repository_url": "https://github.com/ROCm/FlyDSL",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_source_button": True,
    "use_download_button": True,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google/NumPy docstring support)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "_mlir",
    "hip",
    "torch",
    "numpy",
    "nanobind",
    "pybind11",
]

# MyST parser settings
myst_enable_extensions = {
    "deflist",
    "tasklist",
}
myst_heading_anchors = 3
# For substitutions in MyST Markdown and rST files.
# Usage:
#   ```md              | ```rst
#   {{ ROCM_VERSION }} | |ROCM_VERSION|
#   ```                | ```
myst_substitutions = {"FLYDSL_VERSION": version, "SOURCE_COMMIT": _revision, "BUILD_DATE": _built_at}
rst_prolog = "\n".join(f".. |{key}| replace:: {val}" for key, val in myst_substitutions.items())
