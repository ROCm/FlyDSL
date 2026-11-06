# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Pytest plugin for the Windows CI job: skip the known Windows test failures.

Each entry gives its reason, with the PR that fixes it where one is pending;
delete it when the fix lands. Keys are paths under tests/ (pytest's rootdir), as
pytest prints node IDs: an exact node ID covers all its parametrizations, and a
module path the whole module. Entries that match nothing are listed at the end of
the run, so a renamed or fixed test does not stay hidden.
"""

# Empty while no Windows-only failure is known.
KNOWN_FAILURES = {}

_used = set()


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for item in items:
        nodeid = item.nodeid
        keys = (nodeid, nodeid.split("[", 1)[0], nodeid.split("::", 1)[0])
        hit = next((k for k in keys if k in KNOWN_FAILURES), None)
        if hit:
            _used.add(hit)
            drop.append(item)
        else:
            keep.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_terminal_summary(terminalreporter):
    unused = sorted(set(KNOWN_FAILURES) - _used)
    terminalreporter.write_line(f"Windows known failures: {len(_used)} entries used, {len(unused)} unused")
    for key in unused:
        terminalreporter.write_line(f"  unused (fixed or renamed?): {key}")
