# FlyDSL Documentation

This directory contains the Sphinx documentation source for FlyDSL.

## Building locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the HTML documentation:

```bash
make html
```

The output will be in `_build/html/`. Open `_build/html/index.html` in a browser.

## Live preview

For a live-reloading preview during editing:

```bash
pip install sphinx-autobuild
make livehtml
```

## Deployment

Documentation is automatically built and deployed to GitHub Pages via the
`.github/workflows/docs.yml` workflow on pushes to `main`.
