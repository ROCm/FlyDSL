# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for SageAttention2 G3 wave32 online softmax helper (spec §9)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.sageattn2_layout import alpha_to_pv_axis
from tests.kernels.sageattn2_ref import softmax_update

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"softmax tests require gfx120x, got {_ARCH}", allow_module_level=True)


def _is_gfx120() -> bool:
    return str(get_rocm_arch()).startswith("gfx120")


def _arch_str() -> str:
    return "gfx120"


def _fx_stream():
    return fx.Stream(torch.cuda.current_stream())


def test_task1_exp2_hw_vs_numpy_error_report():
    """Spec Task 1 (non-blocking measurement): rocdl.exp2 vs np.exp2 on the
    domain _softmax_update_frag actually evaluates it on (S - m_ij in [-30, 0]).
    Confirms the frozen rtol=1e-4/atol=1e-5 covers it. If this fails, STOP and
    reopen the spec — do not loosen the assertion here.
    """
    rng = np.random.default_rng(42)
    n = 4096
    block = 256
    x_np = (-rng.uniform(0.0, 30.0, size=n)).astype(np.float32)
    want = np.exp2(x_np)

    @flyc.kernel
    def exp2_kernel(A: fx.Tensor, Out: fx.Tensor, N: fx.Constexpr[int], BLOCK: fx.Constexpr[int]):
        tid = fx.Int32(gpu.thread_id("x")) + fx.Int32(BLOCK) * fx.Int32(gpu.block_id("x"))
        a = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((N,), (1,)))
        o = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Out)), fx.make_layout((N,), (1,)))
        from kernels.common.tensor_shim import _to_raw as _raw

        o[tid] = fx.Float32(fx.rocdl.exp2(T.f32, _raw(a[tid])))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        Out: fx.Tensor,
        N: fx.Constexpr[int],
        BLOCK: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        exp2_kernel(A, Out, N, BLOCK).launch(grid=(N // BLOCK,), block=(BLOCK,), stream=stream)

    a = torch.from_numpy(x_np.copy()).cuda()
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    launch(a, out, n, block, stream=_fx_stream())
    torch.cuda.synchronize()
    got = out.detach().cpu().numpy()

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


_KERNEL_MODULE_PATH = Path(__file__).resolve().parents[2] / "kernels" / "attention" / "sageattn2_softmax.py"
_FORBIDDEN_CALL_NAMES = {"WMMA", "wmma", "mma", "mma_atom_call", "GenericSoftmaxHelper"}


def test_a1_kernel_module_does_not_import_tests():
    src = _KERNEL_MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] != "tests"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] != "tests"


def _called_names(node: ast.AST):
    """Yield the dotted-tail name of every function/attribute *call*, not every mention."""
    func = node.func
    if isinstance(func, ast.Name):
        yield func.id
    elif isinstance(func, ast.Attribute):
        yield func.attr


def test_a2_kernel_module_never_calls_wmma_or_generic_softmax_helper():
    src = _KERNEL_MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for name in _called_names(node):
                assert name not in _FORBIDDEN_CALL_NAMES, f"forbidden call: {name}"


def test_r2_shuffle_xor_lane_sees_peer():
    """Spec §9.2 R2: lane `lane` sees the value written by `lane^16` after shuffle_xor(x, 16, 32)."""

    @flyc.kernel
    def r2_kernel(Out: fx.Tensor):
        lane = fx.Int32(gpu.thread_id("x"))
        val = lane.to(fx.Float32)
        peer = val.shuffle_xor(fx.Int32(16), 32)
        o = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Out)), fx.make_layout((32,), (1,)))
        o[lane] = peer

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        r2_kernel(Out).launch(grid=(1,), block=(32,), stream=stream)

    out = torch.empty(32, dtype=torch.float32, device="cuda")
    launch(out, stream=_fx_stream())
    torch.cuda.synchronize()
    got = out.detach().cpu().numpy()
    want = (np.arange(32, dtype=np.int64) ^ 16).astype(np.float32)
    np.testing.assert_array_equal(got, want)


def _r1_launch_for_arch(arch: str):
    @flyc.kernel
    def r1_kernel(Out: fx.Tensor, ARCH: fx.Constexpr[str]):
        lane = fx.Int32(gpu.thread_id("x"))
        g = lane // fx.Int32(16)
        l16 = lane % fx.Int32(16)
        alpha_qk = l16.to(fx.Float32)  # both half-waves agree: l16 is symmetric under lane -> lane^16
        src_i32 = alpha_qk.bitcast(fx.Int32)
        out2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Out)), fx.make_layout((32, 8), (8, 1)))
        for v in range_constexpr(8):
            src_lane = g * fx.Int32(8) + fx.Int32(v)
            byte_idx = src_lane * fx.Int32(4)
            peer_i32 = fx.Int32(fx.rocdl.ds_bpermute(T.i32, byte_idx, src_i32))
            out2d[lane, fx.Int32(v)] = peer_i32.bitcast(fx.Float32)

    @flyc.jit
    def launch(Out: fx.Tensor, ARCH: fx.Constexpr[str], stream: fx.Stream = fx.Stream(None)):
        r1_kernel(Out, ARCH).launch(grid=(1,), block=(32,), stream=stream)

    return launch


@pytest.mark.parametrize("arch", ["gfx120"])
def test_r1_reaxis_permutation_only(arch):
    """Spec §9.2 R1: alpha_qk[l16]=float(l16) on both half-waves, run §5, compare to
    G-I alpha_to_pv_axis(arange(16), arch). Isolated from exp2 and running state."""
    launch = _r1_launch_for_arch(arch)
    out = torch.empty(32, 8, dtype=torch.float32, device="cuda")
    launch(out, arch, stream=_fx_stream())
    torch.cuda.synchronize()
    got = out.detach().cpu().numpy()
    want = alpha_to_pv_axis(np.arange(16, dtype=np.float32), arch)
    np.testing.assert_array_equal(got, want)


@pytest.mark.skipif(not _is_gfx120(), reason="H0 is the gfx1201 arch-pinned set")
def test_h0_gfx1201_core_set_smoke():
    """Spec §9.2 H0. This only asserts the arch gate; the other gfx1201 tests
    run independently in the same pytest session."""
    assert _is_gfx120()


def _tile_to_c_frags(tile: np.ndarray, arch: str) -> np.ndarray:
    """Inverse of sageattn2_layout.c_frags_to_tile. tile [16, 16] (q, k) -> C [32, 8].

    Same M(v) formula as the golden, applied in reverse; this is test-local
    fixture construction, not a second golden.
    """
    if tile.shape != (16, 16):
        raise ValueError(f"tile must be [16, 16], got {tile.shape}")
    c = np.empty((32, 8), dtype=tile.dtype)
    for lane in range(32):
        l16, g = lane % 16, lane // 16
        for v in range(8):
            if arch != "gfx120":
                raise ValueError(arch)
            k = 8 * g + v
            c[lane, v] = tile[l16, k]
    return c


def _run_frag(S_tile: np.ndarray, m_q: np.ndarray, l_q: np.ndarray, *, apply_fp8_offset: bool, arch: str):
    """Launch _softmax_update_tile on a [16,16] logical tile + [16] running state.

    m_q / l_q are broadcast symmetric across both half-waves before launch —
    that symmetry is the caller's precondition (spec §2.1), not something the
    device establishes.
    """
    from kernels.attention.sageattn2_softmax import _softmax_update_tile

    c = _tile_to_c_frags(S_tile.astype(np.float32), arch)
    m32 = np.concatenate([m_q, m_q]).astype(np.float32)
    l32 = np.concatenate([l_q, l_q]).astype(np.float32)
    np.testing.assert_array_equal(m32[:16], m32[16:])  # precondition sanity, not a device check
    np.testing.assert_array_equal(l32[:16], l32[16:])

    S_dev = torch.from_numpy(c.copy()).cuda()
    m_dev = torch.from_numpy(m32.copy()).cuda()
    l_dev = torch.from_numpy(l32.copy()).cuda()
    P, m_out, l_out, alpha_qk, alpha_pv = _softmax_update_tile(S_dev, m_dev, l_dev, apply_fp8_offset=apply_fp8_offset)
    torch.cuda.synchronize()
    return (
        P.detach().cpu().numpy(),
        m_out.detach().cpu().numpy(),
        l_out.detach().cpu().numpy(),
        alpha_qk.detach().cpu().numpy(),
        alpha_pv.detach().cpu().numpy(),
    )


def _assert_matches_g0(P_np, m_out_np, l_out_np, alpha_qk_np, alpha_pv_np, S_tile, m_q, l_q, *, apply_fp8_offset, arch):
    m_ij_ref, l_ref, alpha_ref, p_ref = softmax_update(S_tile, m_q, l_q, apply_fp8_offset=apply_fp8_offset)

    # Output symmetry: a real check on §4/§6, since the *input* symmetry was
    # asserted separately in _run_frag before the launch (spec §9 note).
    np.testing.assert_array_equal(m_out_np[:16], m_out_np[16:])
    np.testing.assert_array_equal(l_out_np[:16], l_out_np[16:])
    np.testing.assert_array_equal(alpha_qk_np[:16], alpha_qk_np[16:])

    np.testing.assert_array_equal(m_out_np[:16], m_ij_ref)
    np.testing.assert_allclose(l_out_np[:16], l_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(alpha_qk_np[:16], alpha_ref, rtol=1e-4, atol=1e-5)

    from tests.kernels.sageattn2_layout import c_frags_to_tile

    p_dev_tile = c_frags_to_tile(P_np.astype(np.float32), arch)
    np.testing.assert_allclose(p_dev_tile, p_ref, rtol=1e-4, atol=1e-5)

    want_alpha_pv = alpha_to_pv_axis(alpha_qk_np[:16], arch)
    np.testing.assert_array_equal(alpha_pv_np, want_alpha_pv)


def test_m1_random_finite_first_tile_no_offset():
    """Spec §9.2 M1: random finite S, m=-inf, l=0, apply_fp8_offset=False, one tile vs G0."""
    arch = _arch_str()
    rng = np.random.default_rng(10)
    S_tile = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)
    outs = _run_frag(S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)
    _assert_matches_g0(*outs, S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)


def test_m2_random_finite_first_tile_with_offset():
    """Spec §9.2 M2: same as M1, apply_fp8_offset=True. m_out equals G0 exactly (includes -8.807)."""
    arch = _arch_str()
    rng = np.random.default_rng(11)
    S_tile = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)
    outs = _run_frag(S_tile, m_q, l_q, apply_fp8_offset=True, arch=arch)
    _assert_matches_g0(*outs, S_tile, m_q, l_q, apply_fp8_offset=True, arch=arch)


@pytest.mark.parametrize("apply_fp8_offset", [False, True])
def test_e1_all_masked_first_tile(apply_fp8_offset):
    """Spec §9.2 E1 (+ both offset flags per H0/H1 note): S all -inf, m=-inf, l=0."""
    arch = _arch_str()
    S_tile = np.full((16, 16), -np.inf, dtype=np.float32)
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)
    P_np, m_out_np, l_out_np, alpha_qk_np, alpha_pv_np = _run_frag(
        S_tile, m_q, l_q, apply_fp8_offset=apply_fp8_offset, arch=arch
    )
    np.testing.assert_array_equal(m_out_np, np.full(32, -np.inf, dtype=np.float32))
    np.testing.assert_array_equal(P_np, np.zeros((32, 8), dtype=np.float32))
    np.testing.assert_array_equal(alpha_qk_np, np.zeros(32, dtype=np.float32))
    np.testing.assert_array_equal(l_out_np, np.zeros(32, dtype=np.float32))
    _assert_matches_g0(
        P_np,
        m_out_np,
        l_out_np,
        alpha_qk_np,
        alpha_pv_np,
        S_tile,
        m_q,
        l_q,
        apply_fp8_offset=apply_fp8_offset,
        arch=arch,
    )


@pytest.mark.parametrize("apply_fp8_offset", [False, True])
def test_e2_all_masked_later_tile_running_state_finite(apply_fp8_offset):
    """Spec §9.2 E2: S all -inf, running m/l finite, both offset flags.
    m_out=m_in, alpha_qk=1, P=0, l_out=l_in. Offset is NOT applied again."""
    arch = _arch_str()
    S_tile = np.full((16, 16), -np.inf, dtype=np.float32)
    rng = np.random.default_rng(12)
    m_q = np.full(16, 3.5, dtype=np.float32)
    l_q = rng.uniform(0.1, 5.0, size=16).astype(np.float32)
    P_np, m_out_np, l_out_np, alpha_qk_np, alpha_pv_np = _run_frag(
        S_tile, m_q, l_q, apply_fp8_offset=apply_fp8_offset, arch=arch
    )
    np.testing.assert_array_equal(m_out_np[:16], m_q)
    np.testing.assert_array_equal(alpha_qk_np, np.ones(32, dtype=np.float32))
    np.testing.assert_array_equal(P_np, np.zeros((32, 8), dtype=np.float32))
    np.testing.assert_allclose(l_out_np[:16], l_q, rtol=1e-4, atol=1e-5)
    _assert_matches_g0(
        P_np,
        m_out_np,
        l_out_np,
        alpha_qk_np,
        alpha_pv_np,
        S_tile,
        m_q,
        l_q,
        apply_fp8_offset=apply_fp8_offset,
        arch=arch,
    )


def test_e3_mixed_some_rows_fully_masked():
    """Spec §9.2 E3: some rows fully -inf, some finite. Per-row match; empty rows atol=0."""
    arch = _arch_str()
    rng = np.random.default_rng(13)
    S_tile = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    S_tile[3, :] = -np.inf
    S_tile[9, :] = -np.inf
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)
    outs = _run_frag(S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)
    _assert_matches_g0(*outs, S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)
    m_out_np = outs[1]
    assert m_out_np[3] == -np.inf and m_out_np[9] == -np.inf
    assert m_out_np[3 + 16] == -np.inf and m_out_np[9 + 16] == -np.inf


@pytest.mark.parametrize("apply_fp8_offset", [False, True])
def test_m3_two_successive_tiles_running_state_fed_back(apply_fp8_offset):
    """Spec §9.2 M3: two successive tiles, running (m,l) fed back, vs two G0 calls."""
    arch = _arch_str()
    rng = np.random.default_rng(14)
    S0 = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    S1 = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    m_q0 = np.full(16, -np.inf, dtype=np.float32)
    l_q0 = np.zeros(16, dtype=np.float32)

    m_ij0_ref, l0_ref, _alpha0_ref, _p0_ref = softmax_update(S0, m_q0, l_q0, apply_fp8_offset=apply_fp8_offset)

    outs0 = _run_frag(S0, m_q0, l_q0, apply_fp8_offset=apply_fp8_offset, arch=arch)
    _assert_matches_g0(*outs0, S0, m_q0, l_q0, apply_fp8_offset=apply_fp8_offset, arch=arch)
    m_q1, l_q1 = outs0[1][:16], outs0[2][:16]  # device m_out/l_out feed the second call

    m_ij1_ref, l1_ref, alpha1_ref, p1_ref = softmax_update(S1, m_ij0_ref, l0_ref, apply_fp8_offset=apply_fp8_offset)
    outs1 = _run_frag(S1, m_q1, l_q1, apply_fp8_offset=apply_fp8_offset, arch=arch)
    _assert_matches_g0(*outs1, S1, m_q1, l_q1, apply_fp8_offset=apply_fp8_offset, arch=arch)
    np.testing.assert_array_equal(outs1[1][:16], m_ij1_ref)
    np.testing.assert_allclose(outs1[2][:16], l1_ref, rtol=1e-4, atol=1e-5)


def test_c1_causal_upper_triangle_masked():
    """Spec §9.2 C1: upper triangle -inf in the 16x16 tile, vs G0 on that pattern."""
    arch = _arch_str()
    rng = np.random.default_rng(15)
    S_tile = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    q_idx = np.arange(16)[:, None]
    k_idx = np.arange(16)[None, :]
    S_tile = np.where(k_idx > q_idx, np.float32(-np.inf), S_tile)
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)
    outs = _run_frag(S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)
    _assert_matches_g0(*outs, S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)


def test_c2_short_tile_five_finite_columns():
    """Spec §9.2 C2 / §6.3: columns 5..15 = -inf; oracle is G0 on the 5-col slice.
    GPU always presents 16 columns; the trailing 11 are -inf pad."""
    arch = _arch_str()
    rng = np.random.default_rng(16)
    S_short = rng.uniform(-20.0, 0.0, size=(16, 5)).astype(np.float32)
    S_tile = np.full((16, 16), -np.inf, dtype=np.float32)
    S_tile[:, :5] = S_short
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)

    m_ij_ref, l_ref, alpha_ref, p_ref = softmax_update(S_short, m_q, l_q, apply_fp8_offset=False)

    P_np, m_out_np, l_out_np, alpha_qk_np, alpha_pv_np = _run_frag(S_tile, m_q, l_q, apply_fp8_offset=False, arch=arch)

    np.testing.assert_array_equal(m_out_np[:16], m_ij_ref)
    np.testing.assert_allclose(l_out_np[:16], l_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(alpha_qk_np[:16], alpha_ref, rtol=1e-4, atol=1e-5)

    from tests.kernels.sageattn2_layout import c_frags_to_tile

    p_dev_tile = c_frags_to_tile(P_np.astype(np.float32), arch)
    np.testing.assert_allclose(p_dev_tile[:, :5], p_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(p_dev_tile[:, 5:], np.zeros((16, 11), dtype=np.float32))


def test_o1_alpha_pv_vs_g0_alpha_end_to_end():
    """Spec §9.2 O1: O_out = O_in * alpha_pv vs O_in * alpha_to_pv_axis(G0 alpha).

    Anchored to G0's alpha, not to the device's own alpha_qk — the only place
    alpha_pv is compared against the oracle end to end (see Global Constraints:
    R1 vs O1 vs the §9.1 alpha_pv row). Uses a second tile after feeding a
    finite, non-uniform running state back from a first tile so the alpha
    vector is not degenerate across all 16 rows."""
    arch = _arch_str()
    rng = np.random.default_rng(17)
    S0 = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    S1 = rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32)
    m_q = np.full(16, -np.inf, dtype=np.float32)
    l_q = np.zeros(16, dtype=np.float32)

    m_ij0_ref, l0_ref, _alpha0_ref, _p0_ref = softmax_update(S0, m_q, l_q, apply_fp8_offset=False)
    outs0 = _run_frag(S0, m_q, l_q, apply_fp8_offset=False, arch=arch)
    m_q1, l_q1 = outs0[1][:16], outs0[2][:16]

    _m_ij1_ref, _l1_ref, alpha_ref, _p1_ref = softmax_update(S1, m_ij0_ref, l0_ref, apply_fp8_offset=False)
    _P_np, _m_out_np, _l_out_np, _alpha_qk_np, alpha_pv_np = _run_frag(
        S1, m_q1, l_q1, apply_fp8_offset=False, arch=arch
    )

    O_in = rng.standard_normal((32, 8)).astype(np.float32)
    O_out_dev = O_in * alpha_pv_np
    want_alpha_pv = alpha_to_pv_axis(alpha_ref, arch)
    O_out_ref = O_in * want_alpha_pv
    np.testing.assert_allclose(O_out_dev, O_out_ref, rtol=1e-4, atol=1e-5)


_MW_NWAVE = 4
_MW_BLOCK = 32 * _MW_NWAVE


def _run_frag_multiwave(tiles, m_qs, l_qs, *, apply_fp8_offset: bool, arch: str):
    """Drive _softmax_update_frag from a MULTI-wave block, one 16x16 tile per wave.

    The product callers (G4/G5) invoke the helper inside a multi-wave CTA, so
    the helper must derive its lane wave-relatively. This launch gives each
    wave a different tile and a different running state, so a wave that reads
    another wave's lanes produces a visibly wrong result.
    """
    from kernels.attention.sageattn2_softmax import _softmax_update_frag

    @flyc.kernel
    def mw_kernel(
        S: fx.Tensor,
        M: fx.Tensor,
        L: fx.Tensor,
        AlphaPV: fx.Tensor,
        AlphaQK: fx.Tensor,
        M_out: fx.Tensor,
        APPLY: fx.Constexpr[bool],
        ARCH: fx.Constexpr[str],
        ROWS: fx.Constexpr[int],
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        s2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(S)), fx.make_layout((ROWS, 8), (8, 1)))
        m1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(M)), fx.make_layout((ROWS,), (1,)))
        l1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(L)), fx.make_layout((ROWS,), (1,)))
        apv = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(AlphaPV)), fx.make_layout((ROWS, 8), (8, 1)))
        aqk = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(AlphaQK)), fx.make_layout((ROWS,), (1,)))
        mo = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(M_out)), fx.make_layout((ROWS,), (1,)))

        S_in = [s2d[tid, fx.Int32(v)] for v in range_constexpr(8)]
        _P, m_ij, _l_new, alpha_qk, alpha_pv = _softmax_update_frag(
            S_in, m1d[tid], l1d[tid], apply_fp8_offset=APPLY
        )
        for v in range_constexpr(8):
            apv[tid, fx.Int32(v)] = alpha_pv[v]
        aqk[tid] = alpha_qk
        mo[tid] = m_ij

    @flyc.jit
    def launch(
        S: fx.Tensor,
        M: fx.Tensor,
        L: fx.Tensor,
        AlphaPV: fx.Tensor,
        AlphaQK: fx.Tensor,
        M_out: fx.Tensor,
        APPLY: fx.Constexpr[bool],
        ARCH: fx.Constexpr[str],
        ROWS: fx.Constexpr[int],
        BLK: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        mw_kernel(S, M, L, AlphaPV, AlphaQK, M_out, APPLY, ARCH, ROWS).launch(grid=(1,), block=(BLK,), stream=stream)

    s_rows, m_rows, l_rows = [], [], []
    for tile, m_q, l_q in zip(tiles, m_qs, l_qs):
        s_rows.append(_tile_to_c_frags(tile.astype(np.float32), arch))
        m_rows.append(np.concatenate([m_q, m_q]).astype(np.float32))
        l_rows.append(np.concatenate([l_q, l_q]).astype(np.float32))

    s_dev = torch.from_numpy(np.ascontiguousarray(np.concatenate(s_rows, axis=0))).cuda()
    m_dev = torch.from_numpy(np.ascontiguousarray(np.concatenate(m_rows))).cuda()
    l_dev = torch.from_numpy(np.ascontiguousarray(np.concatenate(l_rows))).cuda()
    apv = torch.empty(_MW_BLOCK, 8, dtype=torch.float32, device="cuda")
    aqk = torch.empty(_MW_BLOCK, dtype=torch.float32, device="cuda")
    m_out = torch.empty(_MW_BLOCK, dtype=torch.float32, device="cuda")

    launch(s_dev, m_dev, l_dev, apv, aqk, m_out, apply_fp8_offset, arch, _MW_BLOCK, _MW_BLOCK, stream=_fx_stream())
    torch.cuda.synchronize()
    return apv.detach().cpu().numpy(), aqk.detach().cpu().numpy(), m_out.detach().cpu().numpy()


def test_mw1_multiwave_block_each_wave_independent():
    """Regression guard: _softmax_update_frag must derive its lane wave-relatively.

    Spec §1.2 freezes this helper as something G4/G5 call in-register inside a
    multi-wave k-loop, but every other test here launches a single wave, where
    gpu.thread_id("x") and gpu.lane_id() coincide. With a block-global lane the
    §5 ds_bpermute re-axis reads the wrong source lane for every wave but the
    first, and only alpha_pv is affected -- m/l/P stay correct, so nothing else
    in this file would notice.

    On gfx120, M(v) = 8*g + v. ds_bpermute wraps its lane index mod 32, and
    lanes l and l^16 hold the same alpha_qk, so a block-global lane can still
    look right. This test checks each wave against its own tile anyway.
    """
    arch = _arch_str()
    rng = np.random.default_rng(23)
    tiles, m_qs, l_qs = [], [], []
    for _ in range(_MW_NWAVE):
        tiles.append(rng.uniform(-20.0, 0.0, size=(16, 16)).astype(np.float32))
        # Finite, non-uniform running m => non-degenerate alpha, so the
        # permutation is actually asserted (a uniform alpha would not be).
        m_qs.append(rng.uniform(-5.0, 5.0, size=16).astype(np.float32))
        l_qs.append(rng.uniform(0.1, 3.0, size=16).astype(np.float32))

    alpha_pv_np, alpha_qk_np, m_out_np = _run_frag_multiwave(tiles, m_qs, l_qs, apply_fp8_offset=False, arch=arch)

    for w in range(_MW_NWAVE):
        sl = slice(w * 32, (w + 1) * 32)
        m_ij_ref, _l_ref, alpha_ref, _p_ref = softmax_update(tiles[w], m_qs[w], l_qs[w], apply_fp8_offset=False)
        assert len(np.unique(np.round(alpha_ref, 6))) > 1, f"wave {w} fixture has degenerate alpha"
        np.testing.assert_array_equal(m_out_np[sl][:16], m_ij_ref)
        np.testing.assert_allclose(alpha_qk_np[sl][:16], alpha_ref, rtol=1e-4, atol=1e-5)
        # The decisive one: this wave's alpha_pv against the host oracle's alpha.
        np.testing.assert_allclose(alpha_pv_np[sl], alpha_to_pv_axis(alpha_ref, arch), rtol=1e-4, atol=1e-5)

