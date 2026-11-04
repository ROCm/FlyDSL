# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reject oversized GLM calls before weight packing, allocation or launch."""

from types import SimpleNamespace

import pytest
import torch

from kernels.monokernel.config import HIDDEN, glm5_tp_config
from kernels.monokernel.glm import op as glm_op


class ReachedNextValidation(Exception):
    pass


@pytest.mark.parametrize("samples", [0, 17, 32, 40, 41])
@pytest.mark.parametrize("activation", ["fp8", "fp4"])
@pytest.mark.parametrize("with_indexer", [False, True])
def test_constructor_rejects_total_tokens_before_reading_weights(samples, activation, with_indexer):
    with pytest.raises(ValueError, match="total query tokens must be in \\[1, 16\\]"):
        glm_op.Glm5MonoKernel(
            object(),
            samples,
            npes=4,
            native_fp4_mfma=True,
            moe_activation_dtype=activation,
            with_indexer=with_indexer,
        )


@pytest.mark.parametrize("samples", [1, 15, 16])
def test_native_constructor_accepts_boundary_for_shard_validation(monkeypatch, samples):
    def validate_shard(actual, *args, supported_samples, **kwargs):
        assert actual == samples and actual in supported_samples
        assert max(supported_samples) == 16
        raise ReachedNextValidation

    monkeypatch.setattr(glm_op, "validate_shard", validate_shard)
    weights = SimpleNamespace(config=glm5_tp_config(4), heads=16)
    with pytest.raises(ReachedNextValidation):
        glm_op.Glm5MonoKernel(weights, samples, npes=4, native_fp4_mfma=True)


class PositionSentinel:
    @property
    def shape(self):
        raise ReachedNextValidation


@pytest.mark.parametrize("samples", [0, 17, 32, 40])
def test_call_cannot_bypass_total_limit_with_small_chunks(samples):
    op = object.__new__(glm_op.Glm5MonoKernel)
    op.launches_per_step = 1
    op.S = 8
    hidden = torch.empty((samples, HIDDEN), device="meta")
    with pytest.raises(ValueError, match="total query tokens must be in \\[1, 16\\]"):
        op.forward(hidden, PositionSentinel(), None, None, None, None, None)


@pytest.mark.parametrize("samples", [1, 15, 16])
def test_call_accepts_total_boundary_for_position_validation(samples):
    op = object.__new__(glm_op.Glm5MonoKernel)
    op.launches_per_step = 1
    hidden = torch.empty((samples, HIDDEN), device="meta")
    with pytest.raises(ReachedNextValidation):
        op.forward(hidden, PositionSentinel(), None, None, None, None, None)
