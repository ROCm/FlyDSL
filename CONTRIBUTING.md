# Contributing to FlyDSL

Thank you for your interest in contributing to FlyDSL! FlyDSL is a Python DSL and MLIR compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling, optimized for AMD GPUs with ROCm. We welcome contributions of all kinds.

## Ways to Contribute

There are several ways you can contribute to FlyDSL:

* **Report Issues**: Identify and report bugs, performance issues, or unexpected behavior
* **Add Kernels**: Implement new GPU kernels using the `@flyc.kernel` / `@flyc.jit` API
* **Add DSL Features**: Extend the Python DSL or Fly MLIR dialect with new operations
* **Optimize Performance**: Improve existing kernel performance or compiler pipeline
* **Documentation**: Improve docs, add tutorials, or write examples
* **Code Review**: Review pull requests and provide constructive feedback
* **Community Support**: Answer questions and help other users

---

## Getting Started

### Job Board

Not sure where to start? Check out these tasks:

* **Good First Issues**: Simple bugs or small enhancements labeled `good first issue`
* **Help Wanted**: Features or optimizations labeled `help wanted`
* **New Kernel Requests**: Missing kernels needed for new workloads

### Prerequisites

Before contributing, ensure you have:

* **ROCm**: Version 6.x or 7.x installed and configured
* **Build Tools**: `cmake` (≥3.20), C++17 compiler, optionally `ninja`
* **Python**: 3.10+
* **Git**: For version control

---

## Issue Discussion

Please use the [GitHub Issues](https://github.com/ROCm/FlyDSL/issues) tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration (Python version, ROCm version, GPU architecture, etc.).
    This helps reduce the time required to reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

---

## Development Setup

### Build from Source

```bash
# Step 1: Clone the repository
git clone https://github.com/ROCm/FlyDSL.git
cd FlyDSL
git remote add upstream https://github.com/ROCm/FlyDSL.git

# Step 2: Build LLVM/MLIR (one-time, ~30min with -j64)
bash scripts/build_llvm.sh -j64

# Step 3: Build FlyDSL
bash scripts/build.sh -j64

# Step 4: Install in development mode
pip install -e .

# Step 5: Verify
bash scripts/run_tests.sh
```

For more details, see the [README](README.md).

---

## Acceptance Criteria

Contributions should align with FlyDSL's goal of providing a Python DSL and MLIR compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling.

### Add a New Kernel

* Add the kernel implementation under `kernels/` following the existing module conventions.
* Ensure the kernel uses the `@flyc.kernel` / `@flyc.jit` API from `flydsl.compiler` and `flydsl.expr`.
* Add corresponding tests under `tests/` with pytest.

### Add a New DSL Feature or Dialect Op

* For Python DSL extensions, add the implementation under `python/flydsl/expr/` or `python/flydsl/compiler/`.
* For Fly dialect (C++/MLIR) changes, update headers in `include/flydsl/` and implementation in `lib/`.
* Add MLIR lit tests and/or Python-level pytest tests covering the new functionality.

---

## Testing

### Running Tests

For new features or bug fixes, it's mandatory to run the associated tests:

```bash
# Run the full test suite
bash scripts/run_tests.sh

# Run performance benchmarks
bash scripts/run_benchmark.sh

# Run specific test files directly
pytest tests/ -k "test_name" -v
```

Tiered pytest markers, env vars, and examples: **[`tests/README.md`](tests/README.md)**.

### Adding Tests

When adding new features or fixing bugs, include tests that cover the changes:

* Use pytest as the test framework
* Place kernel tests under `tests/kernels/`
* Place unit tests under `tests/unit/`
* Include both correctness and (where applicable) performance checks

### Testing on Different Hardware

If you don't have access to specific AMD GPU models, mention this in your PR. Our CI system will run tests on the supported hardware.

---

## Code Quality

### Python

FlyDSL uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The project configuration is defined in `pyproject.toml`:

* **Line length**: 120 characters
* **Target version**: Python 3.10+
* **Linting rules**: pycodestyle (E/W), pyflakes (F), isort (I)
* **Import sorting**: `flydsl` is treated as first-party

Before submitting, run:

```bash
ruff check python/ kernels/ tests/
ruff format --check python/ kernels/ tests/
```

### C++ (Fly Dialect)

* Tabs should be expanded to spaces. Use 2 spaces indentation (consistent with MLIR/LLVM style).
* Follow MLIR coding conventions for dialect implementation code.
* Use `clang-format` where applicable.

### General Style Guidelines

* Prefer clear, descriptive naming for functions and variables.
* Keep kernel implementations self-contained and well-documented with docstrings.
* Use type hints for Python function signatures.
* `TODO` refers to a note that should be addressed in long-term.
* `FIXME` refers to a short-term bug that needs to be addressed.
* **Minimize external dependencies** — avoid adding new third-party libraries unless absolutely necessary. Prefer using existing dependencies (PyTorch, ROCm/HIP, MLIR). If a new dependency is essential, provide justification in the PR.

---

## Performance Testing

### Benchmarking

Always benchmark your changes for kernel-related PRs:

```bash
# Run the benchmark suite
bash scripts/run_benchmark.sh
```

### Performance Requirements

For kernel PRs, include in the PR description:

* **Hardware**: GPU model (e.g., MI300X)
* **Baseline**: Performance before changes
* **Optimized**: Performance after changes
* **Improvement**: Percentage gain

Example:

```
## Performance Results (MI300X)

| Config | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| BF16, M=1024, N=4096 | 180 μs | 150 μs | 16.7% |
| BF16, M=2048, N=8192 | 720 μs | 600 μs | 16.7% |
```

---

## Pull Request Guidelines

By creating a pull request, you agree to the statements made in the [License](#license) section. Your pull request should target the **main** branch.

### PR Title Format

Use one of these prefixes:

* `[Bugfix]` — Bug fixes
* `[Feature]` — New features or operators
* `[Kernel]` — Kernel additions or optimizations
* `[DSL]` — Python DSL changes
* `[Dialect]` — Fly MLIR dialect changes
* `[Perf]` — Performance optimizations
* `[Doc]` — Documentation improvements
* `[Test]` — Test additions or fixes
* `[CI]` — CI/CD improvements
* `[Misc]` — Miscellaneous changes

Examples:

* `[Kernel][Perf] Optimize RMSNorm kernel using vectorized loads`
* `[Feature] Add PagedAttention decode kernel`
* `[Bugfix] Fix numerical instability in FP16 softmax`
* `[DSL] Add support for async copy expressions`

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Motivation
Why is this change needed?

## Changes
- Detailed list of changes
- Impact on existing code

## Performance (if applicable)
| Configuration | Before | After | Improvement |
|---------------|--------|-------|-------------|
| ...           | ...    | ...   | ...         |

## Testing
- [ ] Unit tests added/updated
- [ ] Performance benchmarks run
- [ ] Tested on MI300X

## Dependencies
- [ ] No new third-party dependencies added
- [ ] If new dependencies added, justification provided

## Breaking Changes
List any breaking changes and migration guide.
```

### Commit Message Guidelines

Follow existing best practice for writing a good Git commit message:

* http://chris.beams.io/posts/git-commit/
* https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message

In particular:

* Use imperative voice, e.g. "Fix this bug", "Refactor the XYZ routine", "Add support for FP8 GEMM".
  Not: "Fixing the bug", "Fixed the bug", "Bug fix", etc.
* Subject should summarize the commit. Do not end subject with a period. Use a blank line
  after the subject.

### Code Review Process

1. **Initial Review**: A maintainer will review within 3–5 business days
2. **Feedback**: Address comments and push updates
3. **Approval**: After approval, CI will run the full test suite
4. **Merge**: Once CI passes, a maintainer will merge

If your PR is urgent or hasn't been reviewed, ping maintainers on the issue or PR.

### CI Pipeline

PRs must pass through the CI checks (see `.github/workflows/flydsl.yaml`) and code review before they can be merged. The CI pipeline will:

1. Build LLVM/MLIR (or restore from cache)
2. Build FlyDSL
3. Run the full test suite
4. Run performance benchmarks

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table to view logs associated with a check if it fails.

To update the code in your PR (e.g. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.

---

## License

FlyDSL is an open source project licensed under the [Apache License 2.0](LICENSE). Because of this, we include the following license header at the top of every new source file. If you create new source files in the repository, please include this text in them as well (replacing "20xx" with the digits for the current year):

**Python / Shell files:**

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 20xx FlyDSL Project Contributors
```

**C++ / TableGen / MLIR files:**

```cpp
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 20xx FlyDSL Project Contributors
```

### Developer Certificate of Origin (DCO)

By contributing to FlyDSL, you certify that your contribution was created in whole or in part by you and that you have the right to submit it under the Apache License 2.0, as specified in this project's [LICENSE](LICENSE).

You can sign off your commits using:

```bash
git commit -s -m "Your commit message"
```

This adds a `Signed-off-by` line to your commit message.

---

## FAQ

**Q: I don't have access to AMD GPU hardware. Can I still contribute?**

A: Yes! You can contribute documentation, Python DSL improvements, and compiler passes. For kernel changes, submit your PR and mention the hardware limitation — our CI will test on supported GPUs.

**Q: Can I add a new third-party library dependency?**

A: We strongly prefer to avoid new dependencies. If necessary, provide justification in your PR explaining why existing dependencies (PyTorch, HIP, MLIR) don't suffice.

**Q: How do I run only the tests related to my change?**

A: Use pytest's `-k` flag:
```bash
pytest tests/ -k "test_rmsnorm" -v
```

**Q: My PR conflicts with the main branch. How do I resolve?**

A:
```bash
git fetch upstream
git rebase upstream/main
# Resolve conflicts, then force-push
git push --force-with-lease
```

---

## Community

* **Issues**: [GitHub Issues](https://github.com/ROCm/FlyDSL/issues)

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and constructive in all interactions.

---

## Thank You!

Thank you for contributing to FlyDSL! Whether you're optimizing a single kernel, adding a new DSL feature, or improving documentation, we appreciate your effort and dedication to the project.
