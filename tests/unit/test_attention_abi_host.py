# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host contract of `kernels/attention/abi.py` and the arch dispatch (no GPU).

Covers CFG-02 (no module-scope torch), CFG-18 (dropout arguments), CFG-19
(varlen arguments) and CFG-20 (the 8xD check) of the attention test plan.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kernels.attention import abi, dispatch
from kernels.attention.philox import dropout_threshold

pytestmark = [pytest.mark.l0_backend_agnostic]

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CFG-02
# ---------------------------------------------------------------------------


def test_abi_module_scope_has_no_torch():
    """No module-scope statement of `abi.py` imports torch (it is imported lazily inside functions)."""
    tree = ast.parse((REPO / "kernels/attention/abi.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "torch" for a in node.names), "module-scope import torch"
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch", "module-scope from torch import"
    # A blocked-torch subprocess import is not possible: `flydsl.compiler` itself imports torch
    # (jit_argument.py), so the AST check above is the part of the contract `abi.py` owns.


def test_dispatch_is_lazy_and_maps_gfx950():
    code = (
        "import sys\n"
        "import kernels.attention.dispatch as d\n"
        "assert 'flydsl' not in sys.modules and 'torch' not in sys.modules\n"
        "assert d.backend_for('gfx950') is not None and d.backend_for('gfx950:sramecc+:xnack-') is not None\n"
        "assert d.backend_for('gfx942') is None and d.backend_for('gfx1201') is None\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True)
    assert dispatch.backend_for("gfx950").arch_prefix == "gfx95"


# ---------------------------------------------------------------------------
# CFG-18
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1e-9, 0.1, 0.5, 0.9, 1 - 1e-9])
def test_dropout_threshold_and_scale(p):
    seed, off1, off2, threshold, scale, keep = abi.dropout_args(True, p, 1234, None, 7, device="cpu")
    want = max(-(2**31), min(2**31 - 1, int((p - 0.5) * 0xFFFFFFFF)))
    assert threshold == want == dropout_threshold(p)
    assert scale == pytest.approx(1.0 / (1.0 - p))
    assert off1 is abi.NULL_PTR and off2 == 7
    assert keep[0] is not None and keep[1] is None


def test_dropout_disabled_is_a_null_abi():
    seed, off1, off2, threshold, scale, keep = abi.dropout_args(False, 0.3, 1, None, 5)
    assert (seed, off1, off2, threshold, scale, keep) == (abi.NULL_PTR, abi.NULL_PTR, 0, 0, 1.0, None)


def test_dropout_p_zero_enabled_keeps_everything():
    _, _, _, threshold, scale, _ = abi.dropout_args(True, 0.0, 1, None, 0, device="cpu")
    assert threshold < -(2**31) + 2
    assert scale == 1.0


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_dropout_rejects_out_of_range_p(p):
    with pytest.raises(ValueError):
        abi.dropout_args(True, p, 1, None, 0, device="cpu")
    with pytest.raises(ValueError, match="requires dropout_p"):
        abi.dropout_args(True, None, 1, None, 0, device="cpu")


# ---------------------------------------------------------------------------
# CFG-19
# ---------------------------------------------------------------------------


def _cu(n, step=8):
    return torch.arange(0, (n + 1) * step, step, dtype=torch.int32)


def _modes(n):
    cu = _cu(n)
    return {
        0x0B0B: abi.varlen_compact(cu, cu, 8, 8),
        0x0202: abi.varlen_padded(cu, cu, 8, 8),
        0x1313: abi.varlen_strided(cu, cu, cu, cu, 8, 8),
        0x150B: abi.varlen_seqused_k(cu, cu, cu, 8, 8),
        0x040B: abi.varlen_seqused_k(cu, None, cu, 8, 8, k_is_cache=True),
    }


@pytest.mark.parametrize("bits", [0x0B0B, 0x0202, 0x1313, 0x150B, 0x040B])
def test_varlen_args_contract(bits):
    n, h, d = 3, 2, 64
    varlen = _modes(n)[bits]
    assert varlen["bits"] == bits
    stacked = bool(bits & abi.VARLEN_STACKED)
    q = torch.empty((1, h, 8 * n, d) if stacked else (n, h, 8, d))
    num_seqlens = n if stacked else 0
    batch = q.shape[0]
    got = abi.varlen_args(varlen, 8, 8, q, batch, num_seqlens)
    assert got[0] == bits and got[-2:] == (8, 8) and len(got) == 7
    # "grid z" = num_seqlens or batch: the packed count when stacked, the real batch otherwise.
    assert (num_seqlens or batch) == (n if stacked else q.shape[0])
    # An underivable count is refused.
    with pytest.raises(ValueError, match="num_seqlens"):
        abi.varlen_args(varlen, 8, 8, q, batch, num_seqlens + 1 if stacked else 1)
    # batch_size is always q.size(0).
    with pytest.raises(ValueError, match="batch_size"):
        abi.varlen_args(varlen, 8, 8, q, batch + 1, num_seqlens)


def test_varlen_args_dense_and_unread_slots():
    q = torch.empty((2, 4, 16, 64))
    got = abi.varlen_args(None, 16, 16, q, 2, 0)
    assert got == (0, abi.NULL_PTR, abi.NULL_PTR, abi.NULL_PTR, abi.NULL_PTR, 16, 16)
    with pytest.raises(ValueError, match="dense call"):
        abi.varlen_args(None, 16, 16, q, 2, 3)
    # Unread slots stay null: varlen_compact reads no position array.
    cu = _cu(2)
    q = torch.empty((1, 4, 16, 64))
    got = abi.varlen_args(abi.varlen_compact(cu, cu, 8, 8), 8, 8, q, 1, 2)
    assert got[2] is abi.NULL_PTR and got[4] is abi.NULL_PTR and got[1] is not abi.NULL_PTR


def test_varlen_bits_rejects_reserved_and_reuse_without_cumulative():
    with pytest.raises(ValueError, match="REUSE"):
        abi.varlen_bits(q_side=abi.VARLEN_POSITION_REUSE)
    with pytest.raises(ValueError, match="reserved"):
        abi.varlen_bits(q_side=3 << 1)
    with pytest.raises(ValueError, match="byte"):
        abi.varlen_bits(q_side=0x100)


# ---------------------------------------------------------------------------
# CFG-20
# ---------------------------------------------------------------------------


def _bhsd_view(b, h, s, d, pad_to=None):
    """A BHSD-shaped `[..., :d]` view of an allocation whose last dim is `pad_to`."""
    return torch.empty(b, h, s, pad_to or d)[..., :d]


def _bshd_compact(b, h, s, d):
    return torch.empty(b, s, h, d).transpose(1, 2)


@pytest.mark.parametrize("d", [20, 73, 100])
def test_8xd_rejects_tight_odd_and_bshd_compact(d):
    for name, t in (("tight", _bhsd_view(2, 3, 5, d)), ("bshd", _bshd_compact(2, 3, 5, d))):
        with pytest.raises(ValueError, match="8"):
            abi.check_8xd(name, t, d)
        with pytest.raises(ValueError, match="8"):
            abi.strides_of(t, name)


@pytest.mark.parametrize("d", [20, 73, 100])
def test_8xd_accepts_ceil8_views(d):
    need = (d + 7) // 8 * 8
    t = _bhsd_view(2, 3, 5, d, pad_to=need)
    abi.check_8xd("padded", t, d)
    assert abi.strides_of(t, "padded") == (3 * 5 * need, 5 * need, need)


def test_8xd_multiple_of_8_needs_nothing():
    abi.check_8xd("q", _bhsd_view(2, 3, 5, 64), 64)
    abi.check_8xd("q", _bshd_compact(2, 3, 5, 64), 64)  # outer strides are multiples of 8


def test_8xd_ignores_size_one_axes():
    # S == 1 and H == 1: only the batch stride matters, and it is ceil8-aligned here.
    abi.check_8xd("q", torch.empty(4, 1, 1, 24)[..., :20], 20)
    # A size-1 batch axis does not constrain the pitch either.
    abi.check_8xd("q", torch.empty(1, 3, 5, 24)[..., :20], 20)


def test_8xd_requires_unit_last_stride():
    with pytest.raises(ValueError, match="contiguous"):
        abi.check_8xd("q", torch.empty(2, 3, 5, 64).transpose(2, 3), 64)


@pytest.mark.parametrize("path", ["fwd", "dq", "dkdv"])
def test_prep_tensors_applies_8xd_to_every_operand(path):
    """The fwd, dQ and dK/dV host paths all go through `prep_tensors` -> `strides_of`."""
    good = {n: _bhsd_view(1, 2, 8, 20, pad_to=24) for n in ("Q", "K", "V", "O", "dO", "dQ", "dK", "dV")}
    named = {
        "fwd": ["Q", "K", "V", "O"],
        "dq": ["Q", "K", "V", "dO", "dQ"],
        "dkdv": ["Q", "K", "V", "dO", "dK", "dV"],
    }[path]
    abi.prep_tensors([(n, good[n]) for n in named])
    for victim in named:
        bad = dict(good)
        bad[victim] = _bshd_compact(1, 2, 8, 20)
        with pytest.raises(ValueError, match=victim):
            abi.prep_tensors([(n, bad[n]) for n in named])


# ---------------------------------------------------------------------------
# run_compiled / dispatch accessors
# ---------------------------------------------------------------------------


def test_run_compiled_compiles_then_launches_every_call(monkeypatch):
    """`flyc.compile` returns the compiled function without launching it, so the first call must launch too."""
    calls = []

    class Compiled:
        def __call__(self, *args):
            calls.append(("launch", args))

    def fake_compile(exe, *args):
        calls.append(("compile", args))
        return Compiled()

    monkeypatch.setattr(abi.flyc, "compile", fake_compile)
    cache, exe = abi.new_compiled_cache(), type("Exe", (), {})()
    abi.run_compiled(cache, exe, 1, 2)
    abi.run_compiled(cache, exe, 3)
    assert [c[0] for c in calls] == ["compile", "launch", "launch"]


def test_dispatch_exposes_traits_and_the_build_identity():
    be = dispatch.backend_for("gfx950")
    for name in ("fwd_traits", "dq_traits", "dkdv_traits", "build_cache_key"):
        assert callable(getattr(be, name))
