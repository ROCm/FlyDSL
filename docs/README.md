# FlyDSL Documentation

This directory contains the Sphinx documentation source for FlyDSL.

## Building locally

Use Python 3.12, matching GitHub Pages, Read the Docs, and the checked-in
dependency lock. Run these commands from the repository root. Install
dependencies:

```bash
python -m pip install -r docs/requirements.txt
```

Build the HTML documentation:

```bash
make -C docs html
```

The output is in **docs/_build/html/**. Open
**docs/_build/html/index.html** in a browser.

Treat warnings as errors before submitting a documentation change:

```bash
make -C docs clean html SPHINXOPTS="-W --keep-going"
python3 scripts/check_docs_api.py --include-docs --all
```

## Live preview

For a live-reloading preview during editing:

```bash
python -m pip install sphinx-autobuild
make -C docs livehtml
```

## Deployment

Pull requests build the documentation through `.github/workflows/docs.yml`.
Pushes to `main` publish the rendered site to GitHub Pages. The repository also
contains `docs/.readthedocs.yaml` for Read the Docs-compatible builds.
