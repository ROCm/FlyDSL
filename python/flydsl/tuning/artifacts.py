# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Serve a searched winner through FlyDSL's existing offline-config path.

``@autotune(..., artifact_name=...)`` already serves a previously tuned config
from ``FLYDSL_AUTOTUNE_CONFIG_DIR`` without benchmarking
(``docs/autotune_guide.md``). The search engine finds the winner; this module
writes it in that artifact format so nothing else in the serving path changes.

An artifact carries launcher keyword arguments, ``num_warps`` and
``waves_per_eu`` (``flydsl.autotune.Config``). ``llvm:<option>`` and ``env:<NAME>``
search keys have no place in an artifact, so a winner that sets one is refused:
bake the option into the launcher (or its ``compile_hints``) instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["export_winner"]


def export_winner(tuner, config: Mapping[str, object], *args: Any, **kwargs: Any):
    """Write ``config`` as the offline artifact ``tuner`` would serve for a
    call with ``*args, **kwargs``; returns the artifact path.

    ``tuner`` is an ``flydsl.autotune.Autotuner`` built with
    ``artifact_name=`` (its ``key`` axes name the problem), and the call
    arguments are the ones the search ran on, with the same ``key`` values
    and device. ``FLYDSL_AUTOTUNE_CONFIG_DIR`` selects the directory.
    """
    from ..autotune import Config

    outside = sorted(k for k in config if k.startswith(("llvm:", "env:")))
    if outside:
        raise ValueError(
            f"export_winner: config sets {outside}; offline artifacts carry launcher kwargs, "
            "num_warps and waves_per_eu only. Apply the LLVM options and environment variables in the launcher."
        )
    chosen = Config.from_dict({k: v for k, v in config.items() if v is not None})
    ref = tuner._artifact_ref(args, kwargs, required=True)
    tuner._emit_artifact(chosen, ref, args, kwargs)
    return ref[0]
