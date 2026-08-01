# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Config-equivalence gate for the CSV-driven a16w-mix tile config.

Asserts that for EVERY tuned (w_dtype, shape, token, stage) cell the CSV-resolved
tiles (via the shipped resolver) equal the original inline heuristic (the
independent oracle transcribed in scripts/gen_a16wmix_tuned_csv.py). This is a
pure-Python, backend-agnostic check: it does NOT compile or run a kernel.
"""

import importlib.util
import os

import pytest

from kernels.moe.moe_2stage_a16wmix import (
    resolve_a16wmix_gemm1_config,
    resolve_a16wmix_gemm2_config,
)

_GEN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
    "gen_a16wmix_tuned_csv.py",
)


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_a16wmix_tuned_csv", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells():
    gen = _load_gen()
    out = []
    for w_dtype, md, inter, E, topk in gen.SHAPES:
        for tok in gen.TOKENS:
            out.append((gen, w_dtype, md, inter, E, topk, tok))
    return out


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("cell", _cells(), ids=lambda c: f"{c[1]}_{c[2]}x{c[3]}_E{c[4]}k{c[5]}_t{c[6]}")
def test_csv_resolver_matches_heuristic(cell):
    gen, w_dtype, md, inter, E, topk, tok = cell
    bm = gen.base_tile_m(w_dtype, tok, E, topk)

    old_g1 = gen.old_gemm1_tiles(D_HIDDEN=md, D_INTER=inter, n_tokens=tok, w_dtype=w_dtype, tile_m=bm)
    new_g1 = resolve_a16wmix_gemm1_config(
        w_dtype=w_dtype, model_dim=md, inter_dim=inter, experts=E, topk=topk, tokens=tok, tile_m=bm
    )
    assert new_g1 == old_g1, f"gemm1 mismatch {w_dtype} {md}x{inter} E{E}k{topk} t{tok}: {new_g1} != {old_g1}"

    old_g2 = gen.old_gemm2_tiles(D_HIDDEN=md, D_INTER=inter, M_logical=tok, w_dtype=w_dtype, tile_m=bm)
    new_g2 = resolve_a16wmix_gemm2_config(
        w_dtype=w_dtype, model_dim=md, inter_dim=inter, experts=E, topk=topk, tokens=tok, tile_m=bm
    )
    assert new_g2 == old_g2, f"gemm2 mismatch {w_dtype} {md}x{inter} E{E}k{topk} t{tok}: {new_g2} != {old_g2}"
