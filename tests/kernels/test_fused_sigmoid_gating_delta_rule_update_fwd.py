#!/usr/bin/env python3
"""Test FlyDSL fused sigmoid-gating delta-rule update-forward kernel (MVP)."""

import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
_embedded = _repo / "build" / "python_packages" / "rocdsl"
if _embedded.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import pytest
import torch

import flydsl
from kernels.fused_sigmoid_gating_delta_rule_update_fwd_kernel import (
    build_fused_sigmoid_gating_delta_rule_update_fwd_module,
)

_sglang_py = Path("/sgl-workspace/sglang/python")
if _sglang_py.exists() and str(_sglang_py) not in sys.path:
    sys.path.insert(0, str(_sglang_py))

try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
        fused_sigmoid_gating_delta_rule_update_kernel,
    )
    HAS_TRITON_KERNEL = True
except ImportError:
    HAS_TRITON_KERNEL = False


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)


def torch_ref_sigmoid_gating_update_fwd(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state_pool,
    state_indices,
    scale,
    use_qk_l2norm_in_kernel=False,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    """Reference for fused sigmoid-gating + recurrent update forward.

    Shapes:
    - A_log, dt_bias: [HV]
    - a, b: [B,T,HV]
    - q, k: [B,T,H,K]
    - v: [B,T,HV,V]
    - state_pool: [N_STATE,HV,K,V]
    - state_indices: [B]
    """
    B, T_seq, H, K_dim = q.shape
    _, _, HV, V_dim = v.shape

    out = torch.empty(B, T_seq, HV, V_dim, device=q.device, dtype=torch.float32)
    state_after = state_pool.clone().float()

    for n in range(B):
        state_idx = int(state_indices[n].item())
        for hv in range(HV):
            h_idx = hv // (HV // H)
            h = state_after[state_idx, hv].clone()  # [K, V]
            for t in range(T_seq):
                q_t = q[n, t, h_idx, :].float()
                k_t = k[n, t, h_idx, :].float()
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                    k_t = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                q_t = q_t * scale
                v_t = v[n, t, hv, :].float().clone()

                x = a[n, t, hv].float() + dt_bias[hv].float()
                beta_x = softplus_beta * x
                if beta_x <= softplus_threshold:
                    softplus_x = (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x))
                else:
                    softplus_x = x
                g_t = -torch.exp(A_log[hv].float()) * softplus_x

                beta_t = torch.sigmoid(b[n, t, hv].float())

                h = h * torch.exp(g_t)
                v_t = v_t - (h * k_t[:, None]).sum(dim=0)
                v_t = v_t * beta_t
                h = h + k_t[:, None] * v_t[None, :]
                out[n, t, hv, :] = (h * q_t[:, None]).sum(dim=0)

            state_after[state_idx, hv] = h

    return out, state_after


def _build_inputs(
    B, T_seq, H, HV, K, V, N_STATE, state_indices, seed=42, dtype=torch.float32
):
    torch.manual_seed(seed)
    A_log = torch.randn(HV, device="cuda", dtype=dtype) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=dtype) * 0.1
    a = torch.randn(B, T_seq, HV, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(B, T_seq, HV, device="cuda", dtype=dtype) * 0.1
    q = torch.randn(B, T_seq, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T_seq, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T_seq, HV, V, device="cuda", dtype=dtype) * 0.1
    state_pool = torch.randn(N_STATE, HV, K, V, device="cuda", dtype=dtype) * 0.1
    state_indices = torch.tensor(state_indices, device="cuda", dtype=torch.int32)
    return A_log, a, dt_bias, q, k, v, b, state_pool, state_indices


def _run_kernel(
    B,
    T_seq,
    H,
    HV,
    K,
    V,
    N_STATE,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state_pool,
    state_indices,
    use_qk_l2norm_in_kernel=False,
    disable_state_update=False,
    disable_output_calculation=False,
    BV=64,
    dtype_str="f32",
):
    m = build_fused_sigmoid_gating_delta_rule_update_fwd_module(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        dtype_str=dtype_str,
        BV=BV,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        disable_output_calculation=disable_output_calculation,
    )
    exe = flydsl.compile(m)
    a_flat = a.reshape(B * T_seq, HV).contiguous()
    b_flat = b.reshape(B * T_seq, HV).contiguous()
    q_flat = q.reshape(B * T_seq * H, K).contiguous()
    k_flat = k.reshape(B * T_seq * H, K).contiguous()
    v_flat = v.reshape(B * T_seq * HV, V).contiguous()
    state_flat = state_pool.reshape(N_STATE * HV, K, V).contiguous()
    if disable_output_calculation:
        o_flat = torch.full(
            (B * T_seq * HV, V),
            float("nan"),
            device="cuda",
            dtype=state_pool.dtype,
        )
    else:
        o_flat = torch.empty(B * T_seq * HV, V, device="cuda", dtype=state_pool.dtype)

    exe(A_log, a_flat, dt_bias, q_flat, k_flat, v_flat, b_flat, state_flat, state_indices, o_flat)
    torch.cuda.synchronize()
    return o_flat.reshape(B, T_seq, HV, V), state_flat.reshape(N_STATE, HV, K, V)


def _run_triton_kernel(
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices,
    scale, softplus_beta=1.0, softplus_threshold=20.0,
    use_qk_l2norm_in_kernel=False,
):
    state_clone = state_pool.clone()
    triton_o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state_clone,
        initial_state_indices=state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    torch.cuda.synchronize()
    return triton_o, state_clone


def _dump_triton_ir_for_fused_sigmoid_gating_delta_rule_update(
    output_dir: str | Path,
    best_only: bool = False,
):
    """Compile Triton kernel and dump available compiler artifacts."""
    if not HAS_TRITON_KERNEL:
        raise RuntimeError("Triton kernel (sglang) is not available.")

    import triton

    # Keep the dump case minimal while preserving the real call path.
    B, T_seq, H, HV, K, V = 1, 1, 1, 1, 16, 16
    N_STATE = 2
    softplus_beta = 1.0
    softplus_threshold = 20.0
    use_qk_l2norm_in_kernel = False

    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        state_indices=[1],
        seed=2026,
        dtype=torch.float32,
    )
    scale = K ** (-0.5)

    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "This helper currently supports NK == 1 only."

    N = B
    o = q.new_empty(NK, *v.shape)
    grid = lambda META: (NK, triton.cdiv(V, META["BV"]), N * HV)

    base_kwargs = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=state_pool,
        h0_indices=state_indices,
        cu_seqlens=None,
        scale=scale,
        T=T_seq,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_INITIAL_STATE=True,
        IS_VARLEN=False,
    )

    autotuner = fused_sigmoid_gating_delta_rule_update_kernel.fn
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compiled_kernels = []
    if best_only:
        # Trigger autotune first so that best_config is selected.
        fused_sigmoid_gating_delta_rule_update(
            **{
                k: v
                for k, v in base_kwargs.items()
                if k not in {"o", "BK", "USE_INITIAL_STATE", "IS_VARLEN"}
            }
        )
        best_config = autotuner.best_config
        if best_config is None:
            raise RuntimeError("Autotuner did not produce best_config.")
        compiled_kernels = [
            autotuner.fn.warmup(
                grid=grid,
                **base_kwargs,
                **best_config.all_kwargs(),
            )
        ]
    else:
        for cfg in autotuner.configs:
            compiled_kernels.append(
                autotuner.fn.warmup(
                    grid=grid,
                    **base_kwargs,
                    **cfg.all_kwargs(),
                )
            )

    dumped = []
    for kernel_idx, compiled in enumerate(compiled_kernels):
        asm_keys = sorted(compiled.asm.keys())
        print(f"[ir-dump] kernel[{kernel_idx}] asm keys: {asm_keys}")
        prefix = "best" if best_only else f"cfg{kernel_idx:02d}"
        for asm_key in asm_keys:
            payload = compiled.asm[asm_key]
            file_path = out_dir / f"{prefix}.{asm_key}"
            if isinstance(payload, bytes):
                file_path.write_bytes(payload)
            else:
                file_path.write_text(payload, encoding="utf-8")
            dumped.append(file_path)

    ttir_files = sorted(out_dir.glob("*.ttir"))
    ttgir_files = sorted(out_dir.glob("*.ttgir"))
    print(
        f"[ir-dump] dumped={len(dumped)} files, "
        f"ttir={len(ttir_files)}, ttgir={len(ttgir_files)}, dir={out_dir}"
    )
    if not ttir_files or not ttgir_files:
        raise AssertionError("TTIR/TTGIR files were not generated.")
    return dumped


def test_dump_triton_ir_for_fused_sigmoid_gating_delta_rule_update(ctx):
    """Optional debug test: dump Triton TTIR/TTGIR artifacts to disk."""
    if not HAS_TRITON_KERNEL:
        pytest.skip("Triton kernel (sglang) not available.")
    if os.environ.get("FLYDSL_DUMP_TRITON_IR", "0") != "1":
        pytest.skip("Set FLYDSL_DUMP_TRITON_IR=1 to enable IR dump test.")

    best_only = os.environ.get("FLYDSL_DUMP_TRITON_IR_BEST_ONLY", "0") == "1"
    default_dir = _repo / "trace" / "triton_ir" / "fused_sigmoid_gating_delta_rule_update"
    dump_dir = Path(os.environ.get("FLYDSL_TRITON_IR_DIR", str(default_dir)))
    dumped = _dump_triton_ir_for_fused_sigmoid_gating_delta_rule_update(
        output_dir=dump_dir,
        best_only=best_only,
    )
    assert dumped


def _print_minmax(name, x):
    x_f = x.float()
    finite_mask = torch.isfinite(x_f)
    if finite_mask.any():
        v = x_f[finite_mask]
        print(f"{name} min={v.min().item():.3e}, max={v.max().item():.3e}")
    else:
        print(f"{name} min=nan, max=nan (all non-finite)")


# @pytest.mark.parametrize(
#     "B,T_seq,H,HV,K,V,N_STATE,state_indices,use_qk_l2norm_in_kernel",
#     [
#         (1, 4, 1, 1, 8, 8, 2, [1], False),  # MVP baseline
#         (1, 4, 2, 4, 8, 8, 3, [2], False),  # HV > H mapping
#         (2, 3, 2, 2, 8, 8, 4, [3, 1], False),  # multi-batch + state-pool indirection
#         (1, 4, 1, 1, 8, 8, 2, [1], True),  # qk l2norm
#     ],
# )
# def test_fused_sigmoid_gating_delta_rule_update_fwd_core(
#     ctx, B, T_seq, H, HV, K, V, N_STATE, state_indices, use_qk_l2norm_in_kernel
# ):
#     A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
#         B, T_seq, H, HV, K, V, N_STATE, state_indices
#     )
#     scale = K ** (-0.5)
#     ref_o, ref_state = torch_ref_sigmoid_gating_update_fwd(
#         A_log=A_log,
#         a=a,
#         dt_bias=dt_bias,
#         q=q,
#         k=k,
#         v=v,
#         b=b,
#         state_pool=state_pool,
#         state_indices=state_indices,
#         scale=scale,
#         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
#     )
#     got_o, got_state = _run_kernel(
#         B=B,
#         T_seq=T_seq,
#         H=H,
#         HV=HV,
#         K=K,
#         V=V,
#         N_STATE=N_STATE,
#         A_log=A_log,
#         a=a,
#         dt_bias=dt_bias,
#         q=q,
#         k=k,
#         v=v,
#         b=b,
#         state_pool=state_pool,
#         state_indices=state_indices,
#         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
#     )
#     _print_minmax("got_o", got_o)
#     _print_minmax("ref_o", ref_o)
#     _print_minmax("got_state", got_state)
#     _print_minmax("ref_state", ref_state)
#     out_err = (got_o - ref_o).abs().max().item()
#     state_err = (got_state - ref_state).abs().max().item()
#     print(f"shape(B,T,H,HV,K,V)=({B},{T_seq},{H},{HV},{K},{V}), qk_l2={use_qk_l2norm_in_kernel}")
#     print(f"max output error: {out_err:.3e}")
#     print(f"max state  error: {state_err:.3e}")
#     assert out_err < 1e-4
#     assert state_err < 1e-4


@pytest.mark.large_shape
def test_fused_sigmoid_gating_delta_rule_update_fwd_large_shape_case(ctx):
    """Large-shape BF16 case: accuracy + performance (FlyDSL vs Triton)."""
    WARMUP = 10
    LOOP = 1000

    batch_size = 64
    seqlen = 1
    num_heads_qk = 4
    num_heads_v = 8
    head_dim = 128
    softplus_beta = 1.0
    softplus_threshold = 20.0
    scale = head_dim ** -0.5
    BV = 64

    B, T_seq, H, HV, K, V = (
        batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, head_dim,
    )
    N_STATE = 64
    state_indices_list = list(range(B))

    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        B=B, T_seq=T_seq, H=H, HV=HV, K=K, V=V,
        N_STATE=N_STATE, state_indices=state_indices_list,
        seed=2026, dtype=torch.bfloat16,
    )

    # ==================== Accuracy ====================

    # --- PyTorch reference ---
    ref_o, ref_state = torch_ref_sigmoid_gating_update_fwd(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        state_pool=state_pool, state_indices=state_indices,
        scale=scale, softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )

    # --- FlyDSL kernel (compile once, reuse for benchmark) ---
    m = build_fused_sigmoid_gating_delta_rule_update_fwd_module(
        B=B, T_seq=T_seq, H=H, HV=HV, K=K, V=V,
        N_STATE=N_STATE, dtype_str="bf16", BV=BV,
    )
    fly_exe = flydsl.compile(m)

    a_flat = a.reshape(B * T_seq, HV).contiguous()
    b_flat = b.reshape(B * T_seq, HV).contiguous()
    q_flat = q.reshape(B * T_seq * H, K).contiguous()
    k_flat = k.reshape(B * T_seq * H, K).contiguous()
    v_flat = v.reshape(B * T_seq * HV, V).contiguous()
    state_pool_flat_orig = state_pool.reshape(N_STATE * HV, K, V).contiguous()
    state_flat_fly = state_pool_flat_orig.clone()
    o_flat_fly = torch.empty(B * T_seq * HV, V, device="cuda", dtype=torch.bfloat16)

    fly_exe(A_log, a_flat, dt_bias, q_flat, k_flat, v_flat, b_flat,
            state_flat_fly, state_indices, o_flat_fly)
    torch.cuda.synchronize()
    got_o = o_flat_fly.reshape(B, T_seq, HV, V)
    got_state = state_flat_fly.reshape(N_STATE, HV, K, V)

    # --- FlyDSL vs Reference ---
    _print_minmax("fly_o", got_o)
    _print_minmax("ref_o", ref_o)
    _print_minmax("fly_state", got_state)
    _print_minmax("ref_state", ref_state)
    fly_out_err = (got_o.float() - ref_o).abs().max().item()
    fly_state_err = (got_state.float() - ref_state).abs().max().item()
    print(f"[FlyDSL vs Ref] max output error: {fly_out_err:.3e}")
    print(f"[FlyDSL vs Ref] max state  error: {fly_state_err:.3e}")
    assert fly_out_err < 1e-3
    assert fly_state_err < 1e-3

    # --- Triton accuracy & performance ---
    if not HAS_TRITON_KERNEL:
        pytest.skip("Triton kernel (sglang) not available for comparison")

    triton_o, triton_state = _run_triton_kernel(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        state_pool=state_pool, state_indices=state_indices,
        scale=scale, softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    _print_minmax("triton_o", triton_o)
    _print_minmax("triton_state", triton_state)

    triton_ref_out_err = (triton_o.float() - ref_o).abs().max().item()
    triton_ref_state_err = (triton_state.float() - ref_state).abs().max().item()
    print(f"[Triton vs Ref]    max output error: {triton_ref_out_err:.3e}")
    print(f"[Triton vs Ref]    max state  error: {triton_ref_state_err:.3e}")

    fly_triton_out_err = (got_o.float() - triton_o.float()).abs().max().item()
    fly_triton_state_err = (got_state.float() - triton_state.float()).abs().max().item()
    print(f"[FlyDSL vs Triton] max output error: {fly_triton_out_err:.3e}")
    print(f"[FlyDSL vs Triton] max state  error: {fly_triton_state_err:.3e}")

    assert triton_ref_out_err < 1e-3
    assert triton_ref_state_err < 1e-3
    assert fly_triton_out_err < 1e-3
    assert fly_triton_state_err < 1e-3

    # ==================== Performance ====================

    # --- Benchmark FlyDSL ---
    for _ in range(WARMUP):
        state_flat_fly.copy_(state_pool_flat_orig)
        fly_exe(A_log, a_flat, dt_bias, q_flat, k_flat, v_flat, b_flat,
                state_flat_fly, state_indices, o_flat_fly)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(LOOP):
        state_flat_fly.copy_(state_pool_flat_orig)
        fly_exe(A_log, a_flat, dt_bias, q_flat, k_flat, v_flat, b_flat,
                state_flat_fly, state_indices, o_flat_fly)
    end_ev.record()
    torch.cuda.synchronize()
    fly_ms = start_ev.elapsed_time(end_ev) / LOOP

    # --- Benchmark Triton ---
    state_triton = state_pool.clone()
    for _ in range(WARMUP):
        state_triton.copy_(state_pool)
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_triton,
            initial_state_indices=state_indices, scale=scale,
        )
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(LOOP):
        state_triton.copy_(state_pool)
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_triton,
            initial_state_indices=state_indices, scale=scale,
        )
    end_ev.record()
    torch.cuda.synchronize()
    triton_ms = start_ev.elapsed_time(end_ev) / LOOP

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  Performance (warmup={WARMUP}, loop={LOOP})")
    print(f"  Shape: B={B}, T={T_seq}, H={H}, HV={HV}, K={K}, V={V}")
    print(f"{'='*60}")
    print(f"  FlyDSL : {fly_ms:.4f} ms/iter")
    print(f"  Triton : {triton_ms:.4f} ms/iter")
    speedup = triton_ms / fly_ms if fly_ms > 0 else float("inf")
    if speedup >= 1.0:
        print(f"  FlyDSL is {speedup:.2f}x faster than Triton")
    else:
        print(f"  Triton is {1.0/speedup:.2f}x faster than FlyDSL")
    print(f"{'='*60}\n")


# def test_fused_sigmoid_gating_delta_rule_update_fwd_softplus_threshold_branch(ctx):
#     """Cover softplus threshold branch with extreme positive inputs."""
#     B, T_seq, H, HV, K, V, N_STATE = 1, 4, 1, 1, 8, 8, 2
#     A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
#         B, T_seq, H, HV, K, V, N_STATE, [1], seed=789
#     )
#     # Force beta*x >> threshold to trigger the linear branch in softplus.
#     a.fill_(35.0)
#     dt_bias.fill_(0.0)

#     scale = K ** (-0.5)
#     ref_o, ref_state = torch_ref_sigmoid_gating_update_fwd(
#         A_log=A_log,
#         a=a,
#         dt_bias=dt_bias,
#         q=q,
#         k=k,
#         v=v,
#         b=b,
#         state_pool=state_pool,
#         state_indices=state_indices,
#         scale=scale,
#     )
#     got_o, got_state = _run_kernel(
#         B=B,
#         T_seq=T_seq,
#         H=H,
#         HV=HV,
#         K=K,
#         V=V,
#         N_STATE=N_STATE,
#         A_log=A_log,
#         a=a,
#         dt_bias=dt_bias,
#         q=q,
#         k=k,
#         v=v,
#         b=b,
#         state_pool=state_pool,
#         state_indices=state_indices,
#     )
#     _print_minmax("got_o", got_o)
#     _print_minmax("ref_o", ref_o)
#     _print_minmax("got_state", got_state)
#     _print_minmax("ref_state", ref_state)
#     out_err = (got_o - ref_o).abs().max().item()
#     state_err = (got_state - ref_state).abs().max().item()
#     print(f"max output error (softplus threshold): {out_err:.3e}")
#     print(f"max state  error (softplus threshold): {state_err:.3e}")
#     assert got_o.isfinite().all()
#     assert got_state.isfinite().all()
#     assert out_err < 1e-4
#     assert state_err < 1e-4
