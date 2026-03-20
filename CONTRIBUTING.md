# Contributing to FlyDSL

We welcome contributions to the FlyDSL project. Please follow these details to help ensure your contributions will be successfully accepted.

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

## Acceptance Criteria

FlyDSL is a Python DSL and MLIR compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling. Contributions should align with this goal, whether they are new features, bug fixes, documentation improvements, or performance optimizations.

### Add a New Kernel

* Add the kernel implementation under `kernels/` following the existing module conventions.
* Ensure the kernel uses the `@flyc.kernel` / `@flyc.jit` API from `flydsl.compiler` and `flydsl.expr`.
* Add corresponding tests under `tests/` with pytest.

### Add a New DSL Feature or Dialect Op

* For Python DSL extensions, add the implementation under `python/flydsl/expr/` or `python/flydsl/compiler/`.
* For Fly dialect (C++/MLIR) changes, update headers in `include/flydsl/` and implementation in `lib/`.
* Add MLIR lit tests and/or Python-level pytest tests covering the new functionality.

### Run Tests

For new features or bug fixes, it's mandatory to run the associated tests:

```bash
# Run the full test suite
bash scripts/run_tests.sh

# Run performance benchmarks
bash scripts/run_benchmark.sh
```

For development iteration, you can also run specific test files directly:

```bash
pytest tests/ -k "test_name" -v
```

## Coding Style

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

### General

* Prefer clear, descriptive naming for functions and variables.
* Keep kernel implementations self-contained and well-documented with docstrings.
* TODO refers to a note that should be addressed in long-term.
* FIXME refers to a short-term bug that needs to be addressed.

## Development Setup

### Prerequisites

* **ROCm**: required for GPU execution (tested on ROCm 6.x, 7.x)
* **Build tools**: `cmake` (≥3.20), C++17 compiler, optionally `ninja`
* **Python**: Python 3.10+ with `pip`

### Build from Source

```bash
# Step 1: Build LLVM/MLIR (one-time, ~30min with -j64)
bash scripts/build_llvm.sh -j64

# Step 2: Build FlyDSL
bash scripts/build.sh -j64

# Step 3: Install in development mode
pip install -e .

# Step 4: Verify
bash scripts/run_tests.sh
```

For more details, see the [README](README.md).

## Pull Request Guidelines

By creating a pull request, you agree to the statements made in the [License](#deliverables) section. Your pull request should target the **main** branch.

Follow existing best practice for writing a good Git commit message.

Some tips:
  * http://chris.beams.io/posts/git-commit/
  * https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message

In particular:

* Use imperative voice, e.g. "Fix this bug", "Refactor the XYZ routine", "Add support for FP8 GEMM".
  Not: "Fixing the bug", "Fixed the bug", "Bug fix", etc.
* Subject should summarize the commit. Do not end subject with a period. Use a blank line
  after the subject.

### Deliverables

FlyDSL is an open source project licensed under the Apache License 2.0. Because of this, we include the following license header at the top of every new source file. If you create new source files in the repository, please include this text in them as well (replacing "xx" with the digits for the current year):

**Python files:**

```python
# Copyright (c) 20xx FlyDSL Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**C++ files:**

```cpp
// Copyright (c) 20xx FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

### Process

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the CI checks (see `.github/workflows/flydsl.yaml`) and code review before they can be merged. The CI pipeline will:

1. Build LLVM/MLIR (or restore from cache)
2. Build FlyDSL
3. Run the full test suite
4. Run performance benchmarks

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table to view logs associated with a check if it fails.

During code reviews, another developer will take a look through your proposed change. If any modifications are requested (or further discussion about anything is needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas. When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (e.g. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.
