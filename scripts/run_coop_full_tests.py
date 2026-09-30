#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Run one deterministic shard of the exhaustive cooperative test matrix."""

import argparse
import hashlib
import os

import pytest


class CoopShard:
    def __init__(self, index: int, count: int):
        self.index = index
        self.count = count

    def pytest_collection_modifyitems(self, config, items):
        selected = []
        deselected = []
        for item in items:
            digest = hashlib.sha256(item.nodeid.encode("utf-8")).digest()
            shard = int.from_bytes(digest[:8], "big") % self.count
            (selected if shard == self.index else deselected).append(item)
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")
    if os.environ.get("FLYDSL_COOP_TESTS_FULL") != "1":
        parser.error("FLYDSL_COOP_TESTS_FULL=1 is required")

    pytest_args = ["tests/extension/coop/", "-q", "--no-header", "--tb=short"]
    if args.collect_only:
        pytest_args.append("--collect-only")
    return pytest.main(pytest_args, plugins=[CoopShard(args.shard_index, args.shard_count)])


if __name__ == "__main__":
    raise SystemExit(main())
