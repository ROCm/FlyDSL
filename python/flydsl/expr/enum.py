# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral DSL enums."""


class SyncScope:
    """LLVM target-neutral sync scopes.

    Target-specific scopes (e.g. AMDGPU ``agent`` / ``workgroup``) live in
    ``flydsl.expr.rocdl.enum.SyncScope``.
    """

    System = ""
    SingleThread = "singlethread"
