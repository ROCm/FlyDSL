# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend-agnostic tests for the MoE tile-config legality filter.

These tests exercise pure host-side math in ``kernels/moe_tuning.py`` and do not
require a GPU, the FlyROCDL bindings, or a compile.  They lock in two properties:

1. Every tile config currently used by ``scripts/run_benchmark.sh`` for the
   in-scope MXFP4 / A8W4 MoE shapes is accepted.
2. Each named illegal case is rejected with the expected machine-readable reason.
"""

import pytest

from kernels.moe_tuning import (
    LDS_LIMIT_BYTES,
    check_tile_config,
    enumerate_legal_configs,
)

pytestmark = pytest.mark.l0_backend_agnostic


# (stage, model_dim, inter_dim, tile_m, tile_n, tile_k, a_dtype)
# Derived from run_benchmark.sh MOE_FP4_SHAPES / MOE_A8W4_SHAPES.  Stage1 uses
# (tile_m, tile_n, tile_k); stage2 uses (tile_m, tile_n2, tile_k2).  In the
# benchmark tables tile_n2 == tile_k2 == 256 for all in-scope MoE rows.
_RUN_BENCHMARK_CONFIGS = [
    # MOE_FP4_SHAPES group A: 7168/256/257/9, tile 64/256/256, n2/k2 256/256
    (1, 7168, 256, 64, 256, 256, "fp4"),
    (2, 7168, 256, 64, 256, 256, "fp4"),
    # MOE_FP4_SHAPES group B: 7168/2048/32/8, tile 64/256/256
    (1, 7168, 2048, 64, 256, 256, "fp4"),
    (2, 7168, 2048, 64, 256, 256, "fp4"),
    # MOE_A8W4_SHAPES GPT-OSS: 3072/3072/128/4, stage1 tile 32/128/256
    (1, 3072, 3072, 32, 128, 256, "fp8"),
    # stage2 tile_n2=256, tile_k2=256
    (2, 3072, 3072, 32, 256, 256, "fp8"),
]


@pytest.mark.parametrize("stage,model_dim,inter_dim,tile_m,tile_n,tile_k,a_dtype", _RUN_BENCHMARK_CONFIGS)
def test_accepts_run_benchmark_configs(stage, model_dim, inter_dim, tile_m, tile_n, tile_k, a_dtype):
    res = check_tile_config(
        stage=stage,
        model_dim=model_dim,
        inter_dim=inter_dim,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype=a_dtype,
        gpu_arch="gfx950",
    )
    assert res.legal, f"expected legal, got reason={res.reason!r} ({res.detail})"
    assert res.lds_bytes is not None and res.lds_bytes <= LDS_LIMIT_BYTES["gfx950"]


def test_rejects_tile_k_bytes_not_div_64():
    # fp4 a_elem_bytes=1 -> tile_k_bytes = tile_k; 288 % 64 != 0.  tile_k>=256 ok.
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=288, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "tile_k_bytes_not_div_64"


def test_rejects_splitk_k_per_batch_not_div_tile_k():
    # model_dim=7168, k_batch=56 -> k_per_batch=128; 128 % 256 != 0.
    res = check_tile_config(
        stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4", k_batch=56
    )
    assert not res.legal
    assert res.reason == "k_per_batch_not_div_tile_k"


def test_rejects_splitk_model_dim_not_div_k_batch():
    res = check_tile_config(
        stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4", k_batch=3
    )
    assert not res.legal
    assert res.reason == "model_dim_not_div_k_batch"


def test_rejects_stage2_model_dim_not_div_tile_n():
    # 7168 % 384 != 0
    res = check_tile_config(stage=2, model_dim=7168, inter_dim=256, tile_m=64, tile_n=384, tile_k=256, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "model_dim_not_div_tile_n"


def test_rejects_stage2_inter_dim_not_div_tile_k():
    # inter_dim=2048, tile_k=768 -> 2048 % 768 != 0 (and 768 % 64 == 0, tile_k>=256)
    res = check_tile_config(stage=2, model_dim=7168, inter_dim=2048, tile_m=64, tile_n=256, tile_k=768, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "inter_dim_not_div_tile_k"


def test_rejects_lds_over_limit():
    # A very large tile pushes stage1 LDS past the gfx950 163840-byte limit.
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=512, tile_n=512, tile_k=256, a_dtype="fp8")
    assert not res.legal
    assert res.reason == "lds_over_limit"
    assert res.lds_bytes is not None and res.lds_bytes > LDS_LIMIT_BYTES["gfx950"]


def test_stage1_fp4_lds_mirrors_builder_no_vec_pack_halving():
    # Regression: stage1 sizes _single_x_bytes from the FULL lds_stride for fp4
    # (no a_elem_vec_pack division), matching compile_mixed_moe_gemm1.  These
    # large-tile_k fp4 configs overflow the gfx950 163840-byte limit and MUST be
    # rejected -- an earlier version halved the fp4 stride and wrongly accepted
    # them.  Source-faithful footprints: 230400 and 197632 bytes.
    from kernels.moe_tuning import stage1_lds_bytes

    r1 = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=32, tile_n=32, tile_k=3584, a_dtype="fp4")
    assert not r1.legal and r1.reason == "lds_over_limit"
    assert stage1_lds_bytes(tile_m=32, tile_n=32, tile_k=3584, a_dtype="fp4") == 230400

    r2 = check_tile_config(stage=1, model_dim=3072, inter_dim=3072, tile_m=32, tile_n=32, tile_k=3072, a_dtype="fp4")
    assert not r2.legal and r2.reason == "lds_over_limit"
    assert stage1_lds_bytes(tile_m=32, tile_n=32, tile_k=3072, a_dtype="fp4") == 197632

    # fp4 and fp8 share the same single_x sizing at stage1 (a_elem_bytes==1, no
    # vec-pack division), so equal tiles give equal LDS.
    assert stage1_lds_bytes(tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4") == stage1_lds_bytes(
        tile_m=64, tile_n=256, tile_k=256, a_dtype="fp8"
    )


def test_rejects_fp4_tile_m_too_small():
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=16, tile_n=256, tile_k=256, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "tile_m_lt_32"


def test_rejects_fp4_tile_k_too_small():
    # tile_k=128 is < 256; still tile_k_bytes % 64 == 0, so the MX-FP4 floor must catch it.
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=128, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "tile_k_lt_256"


def test_rejects_bad_stage_and_dtype():
    assert (
        check_tile_config(
            stage=3, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4"
        ).reason
        == "bad_stage"
    )
    assert (
        check_tile_config(
            stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="bogus"
        ).reason
        == "bad_a_dtype"
    )


def test_enumerate_logs_rejections_with_reasons():
    rejected = []
    legal = enumerate_legal_configs(
        stage=1,
        model_dim=7168,
        inter_dim=256,
        a_dtype="fp4",
        tile_m_choices=(16, 32, 64),  # 16 is illegal (tile_m_lt_32)
        tile_n_choices=(256,),
        tile_k_choices=(128, 256),  # 128 is illegal (tile_k_lt_256)
        rejected_log=rejected,
    )
    # At least one legal config (e.g. tile_m in {32,64}, tile_k=256).
    assert legal, "expected some legal configs"
    assert all(r.legal for r in legal)
    # Every rejection carries a machine-readable reason.
    assert rejected, "expected some rejected configs"
    assert all(r["reason"] for r in rejected)
    reasons = {r["reason"] for r in rejected}
    assert "tile_m_lt_32" in reasons
    assert "tile_k_lt_256" in reasons
