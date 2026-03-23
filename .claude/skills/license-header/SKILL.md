---
name: license-header
description: >
  Add or update SPDX license headers on source files before git commit. Handles Python, C/C++,
  CUDA/HIP, and shell scripts. Detects LLVM-derived files and skips them. Updates copyright end
  year on existing headers. Use when the user says "add license header", "update license",
  "check headers", "/license-header", or wants to ensure files have proper SPDX headers before
  committing. Also trigger before git commits when new files are being added.
---

# License Header

Add or update SPDX license headers on changed files before committing.

## Header Format

**Python** (`.py`):
```
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-{year}, Advanced Micro Devices, Inc. All rights reserved.
```

**C/C++/CUDA/HIP** (`.c`, `.cpp`, `.h`, `.cu`, `.hip`, etc.):
```
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-{year}, Advanced Micro Devices, Inc. All rights reserved.
```

**Shell** (`.sh`, `.bash`):
```
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-{year}, Advanced Micro Devices, Inc. All rights reserved.
```

`{year}` is the current calendar year.

## Behavior

1. **New files without header** -> prepend the 2-line header (preserving shebang / encoding lines)
2. **Files with existing AMD copyright** -> update end year to current year if outdated
3. **LLVM-derived files** (contain "Part of the LLVM Project" or "Apache-2.0 WITH LLVM-exception") -> skip
4. **Non-source files** (`.json`, `.md`, `.txt`, etc.) -> skip
5. **Empty files** -> skip

## Steps

### 1. Run the script

Run `update_license_headers.py` from this skill's `scripts/` directory on the current repo:

```bash
python /FlyDSL/.claude/skills/license-header/scripts/update_license_headers.py --repo .
```

To preview changes without writing:

```bash
python /FlyDSL/.claude/skills/license-header/scripts/update_license_headers.py --repo . --dry-run
```

To process specific files only:

```bash
python /FlyDSL/.claude/skills/license-header/scripts/update_license_headers.py file1.py file2.cu
```

### 2. Review output

The script reports:
- Files where a header was **added**
- Files where the copyright year was **updated**
- Files **already correct** (no changes)
- Files **skipped** (LLVM-derived or unsupported type)

### 3. Re-stage if needed

If files were staged before running the script, remind the user to re-stage them since
in-place edits mark them as modified again:

```bash
git add <modified files>
```

## Notes

- The script auto-detects changed files from `git diff --cached`, `git diff`, and `git ls-files --others`.
- Shebang lines (`#!/...`) and encoding declarations (`# -*- coding ...`) are preserved above the header.
- The copyright start year is always `2024` for new headers. For existing headers, only the end year is updated.
- To override the end year: `--year 2027`.
