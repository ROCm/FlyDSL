"""No-GPU tests for the dense flash_attn_func routing threshold (gfx950).

Locks the batch-aware dense gate: DUALWAVE_SWP is chosen when seq_len clears a
batch-dependent threshold (256 by default, 192 at large batch), because the
pipeline's fixed cost amortizes over batch*seq_len rather than seq_len alone.
"""

import pytest

from kernels.flash_attn_generic import (
    _DUALWAVE_LARGE_BATCH,
    _DUALWAVE_MIN_DENSE_SEQ,
    _DUALWAVE_MIN_DENSE_SEQ_LARGE_BATCH,
    _routes_dense_to_dualwave,
)

pytestmark = pytest.mark.l0_backend_agnostic


@pytest.mark.parametrize(
    "batch,seq_len,expect_dualwave",
    [
        # Small batch: crossover at _DUALWAVE_MIN_DENSE_SEQ (256).
        (1, 128, False),
        (1, 192, False),
        (1, 255, False),
        (1, 256, True),
        (1, 8192, True),
        # Large batch: crossover drops to 192 (the fix vs the old flat S<256 gate).
        (8, 128, False),
        (8, 191, False),
        (8, 192, True),
        (8, 256, True),
    ],
)
def test_dense_threshold_is_batch_aware(batch, seq_len, expect_dualwave):
    assert _routes_dense_to_dualwave(batch, seq_len) is expect_dualwave


def test_large_batch_192_is_the_regression_fix():
    # The old flat gate routed B>=8, S=192 to the generic kernel; measured data
    # shows DUALWAVE_SWP is ~14-16% faster there. This is the cell the fix targets.
    assert _routes_dense_to_dualwave(8, 192) is True
    assert _routes_dense_to_dualwave(1, 192) is False


def test_non_int_seq_len_routes_to_dualwave():
    # A symbolic / unknown seq_len cannot be gated; DUALWAVE_SWP handles any length.
    assert _routes_dense_to_dualwave(1, None) is True
    assert _routes_dense_to_dualwave(8, "dynamic") is True


def test_unknown_batch_treated_as_small():
    assert _routes_dense_to_dualwave(None, 192) is False
    assert _routes_dense_to_dualwave(None, 256) is True


def test_threshold_constants_consistent():
    # Large-batch threshold must not exceed the default, or the fix would be a no-op.
    assert _DUALWAVE_MIN_DENSE_SEQ_LARGE_BATCH <= _DUALWAVE_MIN_DENSE_SEQ
    assert _DUALWAVE_LARGE_BATCH >= 2
