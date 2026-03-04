"""FlyDSL fused sigmoid-gating delta-rule update-forward kernel.

V3 – thread-parallel over V dimension, no LDS:
- block_size = (BV, 1, 1); each thread owns one V-column of the hidden state
- recurrent state h[K] lives in registers (not shared memory)
- q[K], k[K] loaded directly from global memory (L1 cache broadcast)
- all K-dim reductions are thread-local (no cross-thread communication)
"""

from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
import _mlir.extras.types as T


KERNEL_NAME = "fused_sigmoid_gating_delta_rule_update_fwd_kernel"

SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0


def build_fused_sigmoid_gating_delta_rule_update_fwd_module(
    B: int,
    T_seq: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    N_STATE: int,
    dtype_str: str = "f32",
    BV: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
):
    """Build fused sigmoid-gating + recurrent-delta-rule update-forward module.

    V3 uses BV threads per block; each thread handles one V position.
    q/k loaded directly from global memory (L1 cache broadcast), no LDS.
    """
    if dtype_str not in {"f32", "bf16"}:
        raise NotImplementedError("Supported dtype_str values are 'f32' and 'bf16'.")
    if HV % H != 0:
        raise ValueError("HV must be divisible by H for hv->h mapping.")
    if BV <= 0:
        raise ValueError("BV must be positive.")

    gpu_arch = get_hip_arch()
    compute_type = T.f32()
    elem_type = T.f32() if dtype_str == "f32" else T.bf16()
    scale = K ** (-0.5)

    _state = {
        "elem_type": elem_type,
        "compute_type": compute_type,
    }

    BT = B * T_seq
    BTH = B * T_seq * H
    BTHV = B * T_seq * HV
    STATEHV = N_STATE * HV

    class _FusedSigmoidGatingDeltaRuleUpdateFwd(flir.MlirModule):
        GPU_MODULE_NAME = f"fused_sigmoid_gating_delta_rule_update_fwd_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            pass

        @flir.kernel
        def fused_sigmoid_gating_delta_rule_update_fwd_kernel(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, elem_type),
            a: lambda: T.memref(BT, HV, elem_type),
            dt_bias: lambda: T.memref(HV, elem_type),
            q: lambda: T.memref(BTH, K, elem_type),
            k: lambda: T.memref(BTH, K, elem_type),
            v: lambda: T.memref(BTHV, V, elem_type),
            b: lambda: T.memref(BT, HV, elem_type),
            initial_state_source: lambda: T.memref(STATEHV, K, V, elem_type),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(BTHV, V, elem_type),
        ):
            # ---- block / thread indices ----
            i_v_tile = flir.const_index(flir.block_idx("y"))
            i_nh = flir.const_index(flir.block_idx("z"))
            c_hv = arith.as_value(arith.index(HV))
            i_n = arith.as_value(flir.arith.DivUIOp(arith.as_value(i_nh), c_hv).result)
            i_hv = arith.as_value(flir.arith.RemUIOp(arith.as_value(i_nh), c_hv).result)
            c_hv_per_h = arith.as_value(arith.index(HV // H))
            i_h = arith.as_value(
                flir.arith.DivUIOp(arith.as_value(i_hv), c_hv_per_h).result
            )

            tid = flir.const_index(flir.thread_idx("x"))
            c_bv = arith.as_value(arith.index(BV))
            iv_base = arith.as_value(
                flir.arith.MulIOp(arith.as_value(i_v_tile), c_bv).result
            )
            iv_global = arith.as_value(
                flir.arith.AddIOp(arith.as_value(iv_base), arith.as_value(tid)).result
            )
            c_V = arith.as_value(arith.index(V))
            c_zero_idx = arith.as_value(arith.index(0))
            iv_valid = arith.CmpIOp(
                arith.CmpIPredicate.ult,
                arith.as_value(iv_global),
                arith.as_value(c_V),
            )
            iv_safe = arith.select(
                arith.as_value(iv_valid),
                arith.as_value(iv_global),
                arith.as_value(c_zero_idx),
            )

            comp = compute_type
            c_log2e = arith.constant(1.4426950408889634, type=comp)
            c_ln2 = arith.constant(0.6931471805599453, type=comp)
            c_scale = arith.constant(scale, type=comp)
            c_neg_one = arith.constant(-1.0, type=comp)
            c_one = arith.constant(1.0, type=comp)
            c_softplus_beta = arith.constant(SOFTPLUS_BETA, type=comp)
            c_inv_softplus_beta = arith.constant(1.0 / SOFTPLUS_BETA, type=comp)
            c_softplus_threshold = arith.constant(SOFTPLUS_THRESHOLD, type=comp)
            c_eps = arith.constant(1e-6, type=comp)
            fm_fast = flir.arith.FastMathFlags.fast
            c_zero_f32 = arith.constant(0.0, type=comp)

            c_TH = arith.as_value(arith.index(T_seq * H))
            c_THV = arith.as_value(arith.index(T_seq * HV))
            c_T = arith.as_value(arith.index(T_seq))
            c_H = arith.as_value(arith.index(H))
            c_K = arith.as_value(arith.index(K))

            # ---- state row = state_idx * HV + hv ----
            state_idx_i32 = memref.load(initial_state_indices, [arith.as_value(i_n)])
            c_zero_i32 = arith.constant(0, type=T.i32())
            state_idx_nonneg = arith.CmpIOp(
                arith.CmpIPredicate.sge,
                arith.as_value(state_idx_i32),
                arith.as_value(c_zero_i32),
            )
            state_idx_i32_safe = arith.select(
                arith.as_value(state_idx_nonneg),
                arith.as_value(state_idx_i32),
                arith.as_value(c_zero_i32),
            )
            state_idx = arith.as_value(
                flir.arith.IndexCastOp(T.index(), arith.as_value(state_idx_i32_safe)).result
            )
            state_mul = flir.arith.MulIOp(arith.as_value(state_idx), c_hv).result
            row_state = arith.as_value(
                flir.arith.AddIOp(arith.as_value(state_mul), arith.as_value(i_hv)).result
            )

            # ---- register-based hidden state h[K] per thread ----
            h_reg = []
            for ik in range_constexpr(K):
                h_init = memref.load(
                    initial_state_source,
                    [arith.as_value(row_state), arith.index(ik), arith.as_value(iv_safe)],
                )
                if dtype_str != "f32":
                    h_init = flir.arith.extf(comp, arith.as_value(h_init))
                h_init = arith.select(
                    arith.as_value(state_idx_nonneg),
                    arith.as_value(h_init),
                    arith.as_value(c_zero_f32),
                )
                h_init = arith.select(
                    arith.as_value(iv_valid),
                    arith.as_value(h_init),
                    arith.as_value(c_zero_f32),
                )
                h_reg.append(h_init)

            # ================ time-step loop ================
            for t in range_constexpr(T_seq):
                t_idx = arith.as_value(arith.index(t))

                # row_qt = n * (T*H) + t * H + h
                m1 = flir.arith.MulIOp(arith.as_value(i_n), c_TH).result
                m2 = flir.arith.MulIOp(t_idx, c_H).result
                a1 = flir.arith.AddIOp(arith.as_value(m1), arith.as_value(m2)).result
                row_qt = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(a1), arith.as_value(i_h)).result
                )

                # row_vt = n * (T*HV) + t * HV + hv
                m3 = flir.arith.MulIOp(arith.as_value(i_n), c_THV).result
                m4 = flir.arith.MulIOp(t_idx, c_hv).result
                a2 = flir.arith.AddIOp(arith.as_value(m3), arith.as_value(m4)).result
                row_vt = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(a2), arith.as_value(i_hv)).result
                )

                # row_ab = n * T + t
                m5 = flir.arith.MulIOp(arith.as_value(i_n), c_T).result
                row_ab = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(m5), t_idx).result
                )

                # ---- optional L2 norm (redundantly computed per thread) ----
                if use_qk_l2norm_in_kernel:
                    q_norm_sq = arith.constant(0.0, type=comp)
                    k_norm_sq = arith.constant(0.0, type=comp)
                    for ik in range_constexpr(K):
                        qi = memref.load(q, [arith.as_value(row_qt), arith.index(ik)])
                        if dtype_str != "f32":
                            qi = flir.arith.extf(comp, arith.as_value(qi))
                        ki = memref.load(k, [arith.as_value(row_qt), arith.index(ik)])
                        if dtype_str != "f32":
                            ki = flir.arith.extf(comp, arith.as_value(ki))
                        q_sq = flir.arith.MulFOp(
                            arith.as_value(qi), arith.as_value(qi), fastmath=fm_fast
                        ).result
                        k_sq = flir.arith.MulFOp(
                            arith.as_value(ki), arith.as_value(ki), fastmath=fm_fast
                        ).result
                        q_norm_sq = flir.arith.AddFOp(
                            arith.as_value(q_norm_sq), arith.as_value(q_sq), fastmath=fm_fast
                        ).result
                        k_norm_sq = flir.arith.AddFOp(
                            arith.as_value(k_norm_sq), arith.as_value(k_sq), fastmath=fm_fast
                        ).result
                    q_inv_norm = flir.math.rsqrt(
                        arith.as_value(
                            flir.arith.AddFOp(
                                arith.as_value(q_norm_sq), arith.as_value(c_eps), fastmath=fm_fast
                            ).result
                        ),
                        fastmath=fm_fast,
                    )
                    k_inv_norm = flir.math.rsqrt(
                        arith.as_value(
                            flir.arith.AddFOp(
                                arith.as_value(k_norm_sq), arith.as_value(c_eps), fastmath=fm_fast
                            ).result
                        ),
                        fastmath=fm_fast,
                    )

                # ---- load v (1 scalar per thread, coalesced) ----
                v_val = memref.load(v, [arith.as_value(row_vt), arith.as_value(iv_safe)])
                if dtype_str != "f32":
                    v_val = flir.arith.extf(comp, arith.as_value(v_val))
                v_val = arith.select(
                    arith.as_value(iv_valid),
                    arith.as_value(v_val),
                    arith.as_value(c_zero_f32),
                )

                # ---- g = -exp(A_log) * softplus(a + dt_bias) ----
                a_log_val = memref.load(A_log, [arith.as_value(i_hv)])
                dt_bias_val = memref.load(dt_bias, [arith.as_value(i_hv)])
                a_val = memref.load(a, [arith.as_value(row_ab), arith.as_value(i_hv)])
                if dtype_str != "f32":
                    a_log_val = flir.arith.extf(comp, arith.as_value(a_log_val))
                    dt_bias_val = flir.arith.extf(comp, arith.as_value(dt_bias_val))
                    a_val = flir.arith.extf(comp, arith.as_value(a_val))
                x = flir.arith.AddFOp(
                    arith.as_value(a_val), arith.as_value(dt_bias_val), fastmath=fm_fast
                ).result
                beta_x = flir.arith.MulFOp(
                    arith.as_value(c_softplus_beta), arith.as_value(x), fastmath=fm_fast
                ).result
                x_log2e = flir.arith.MulFOp(
                    arith.as_value(beta_x), arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                exp_x = flir.math.exp2(arith.as_value(x_log2e), fastmath=fm_fast)
                one_plus_exp_x = flir.arith.AddFOp(
                    arith.as_value(c_one), arith.as_value(exp_x), fastmath=fm_fast
                ).result
                log2_one_plus_exp_x = flir.math.log2(
                    arith.as_value(one_plus_exp_x), fastmath=fm_fast
                )
                ln_one_plus_exp_x = flir.arith.MulFOp(
                    arith.as_value(log2_one_plus_exp_x),
                    arith.as_value(c_ln2),
                    fastmath=fm_fast,
                ).result
                softplus_x = flir.arith.MulFOp(
                    arith.as_value(c_inv_softplus_beta),
                    arith.as_value(ln_one_plus_exp_x),
                    fastmath=fm_fast,
                ).result
                use_exp_branch = arith.CmpFOp(
                    arith.CmpFPredicate.OLE,
                    arith.as_value(beta_x),
                    arith.as_value(c_softplus_threshold),
                ).result
                softplus_x = arith.select(
                    use_exp_branch, arith.as_value(softplus_x), arith.as_value(x)
                )

                a_log_log2e = flir.arith.MulFOp(
                    arith.as_value(a_log_val), arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                exp_a_log = flir.math.exp2(arith.as_value(a_log_log2e), fastmath=fm_fast)
                exp_a_log_softplus = flir.arith.MulFOp(
                    arith.as_value(exp_a_log), arith.as_value(softplus_x), fastmath=fm_fast
                ).result
                g_val = flir.arith.MulFOp(
                    arith.as_value(c_neg_one),
                    arith.as_value(exp_a_log_softplus),
                    fastmath=fm_fast,
                ).result

                # ---- beta = sigmoid(b) ----
                b_val = memref.load(b, [arith.as_value(row_ab), arith.as_value(i_hv)])
                if dtype_str != "f32":
                    b_val = flir.arith.extf(comp, arith.as_value(b_val))
                neg_b = flir.arith.MulFOp(
                    arith.as_value(c_neg_one), arith.as_value(b_val), fastmath=fm_fast
                ).result
                neg_b_log2e = flir.arith.MulFOp(
                    arith.as_value(neg_b), arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                exp_neg_b = flir.math.exp2(arith.as_value(neg_b_log2e), fastmath=fm_fast)
                one_plus_exp_neg_b = flir.arith.AddFOp(
                    arith.as_value(c_one), arith.as_value(exp_neg_b), fastmath=fm_fast
                ).result
                beta_val = flir.arith.DivFOp(
                    arith.as_value(c_one),
                    arith.as_value(one_plus_exp_neg_b),
                    fastmath=fm_fast,
                ).result

                # ---- exp(g) ----
                g_log2e = flir.arith.MulFOp(
                    arith.as_value(g_val), arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                exp_g = flir.math.exp2(arith.as_value(g_log2e), fastmath=fm_fast)

                # ---- h *= exp(g) ----
                for ik in range_constexpr(K):
                    h_reg[ik] = flir.arith.MulFOp(
                        arith.as_value(h_reg[ik]), arith.as_value(exp_g), fastmath=fm_fast
                    ).result

                # ---- v -= sum(h * k, dim=0)  [thread-local K reduction] ----
                acc_hk = arith.constant(0.0, type=comp)
                for ik in range_constexpr(K):
                    k_ik = memref.load(k, [arith.as_value(row_qt), arith.index(ik)])
                    if dtype_str != "f32":
                        k_ik = flir.arith.extf(comp, arith.as_value(k_ik))
                    if use_qk_l2norm_in_kernel:
                        k_ik = flir.arith.MulFOp(
                            arith.as_value(k_ik), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                    prod = flir.arith.MulFOp(
                        arith.as_value(h_reg[ik]), arith.as_value(k_ik), fastmath=fm_fast
                    ).result
                    acc_hk = flir.arith.AddFOp(
                        arith.as_value(acc_hk), arith.as_value(prod), fastmath=fm_fast
                    ).result
                v_val = flir.arith.SubFOp(
                    arith.as_value(v_val), arith.as_value(acc_hk)
                ).result

                # ---- v *= beta ----
                v_val = flir.arith.MulFOp(
                    arith.as_value(v_val), arith.as_value(beta_val), fastmath=fm_fast
                ).result

                # ---- h += k ⊗ v ----
                for ik in range_constexpr(K):
                    k_ik = memref.load(k, [arith.as_value(row_qt), arith.index(ik)])
                    if dtype_str != "f32":
                        k_ik = flir.arith.extf(comp, arith.as_value(k_ik))
                    if use_qk_l2norm_in_kernel:
                        k_ik = flir.arith.MulFOp(
                            arith.as_value(k_ik), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                    prod = flir.arith.MulFOp(
                        arith.as_value(k_ik), arith.as_value(v_val), fastmath=fm_fast
                    ).result
                    h_reg[ik] = flir.arith.AddFOp(
                        arith.as_value(h_reg[ik]), arith.as_value(prod), fastmath=fm_fast
                    ).result

                # ---- o = sum(h * q * scale, dim=0)  [thread-local K reduction] ----
                if not disable_output_calculation:
                    o_acc = arith.constant(0.0, type=comp)
                    for ik in range_constexpr(K):
                        q_ik = memref.load(q, [arith.as_value(row_qt), arith.index(ik)])
                        if dtype_str != "f32":
                            q_ik = flir.arith.extf(comp, arith.as_value(q_ik))
                        if use_qk_l2norm_in_kernel:
                            q_ik = flir.arith.MulFOp(
                                arith.as_value(q_ik), arith.as_value(q_inv_norm), fastmath=fm_fast
                            ).result
                        q_scaled = flir.arith.MulFOp(
                            arith.as_value(q_ik),
                            arith.as_value(c_scale),
                            fastmath=fm_fast,
                        ).result
                        prod = flir.arith.MulFOp(
                            arith.as_value(h_reg[ik]), arith.as_value(q_scaled), fastmath=fm_fast
                        ).result
                        o_acc = flir.arith.AddFOp(
                            arith.as_value(o_acc), arith.as_value(prod), fastmath=fm_fast
                        ).result

                    out_elem = (
                        o_acc
                        if dtype_str == "f32"
                        else flir.arith.truncf(elem_type, arith.as_value(o_acc))
                    )
                    old_o = memref.load(o, [arith.as_value(row_vt), arith.as_value(iv_safe)])
                    out_o = arith.select(
                        arith.as_value(iv_valid),
                        arith.as_value(out_elem),
                        arith.as_value(old_o),
                    )
                    memref.store(
                        arith.as_value(out_o),
                        o,
                        [arith.as_value(row_vt), arith.as_value(iv_safe)],
                    )

            # ================ write final state back ================
            if not disable_state_update:
                for ik in range_constexpr(K):
                    h_fin = h_reg[ik]
                    h_fin_elem = (
                        h_fin
                        if dtype_str == "f32"
                        else flir.arith.truncf(elem_type, arith.as_value(h_fin))
                    )
                    old_h = memref.load(
                        initial_state_source,
                        [arith.as_value(row_state), arith.index(ik), arith.as_value(iv_safe)],
                    )
                    new_h = arith.select(
                        arith.as_value(iv_valid),
                        arith.as_value(h_fin_elem),
                        arith.as_value(old_h),
                    )
                    new_h = arith.select(
                        arith.as_value(state_idx_nonneg),
                        arith.as_value(new_h),
                        arith.as_value(old_h),
                    )
                    memref.store(
                        arith.as_value(new_h),
                        initial_state_source,
                        [arith.as_value(row_state), arith.index(ik), arith.as_value(iv_safe)],
                    )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, elem_type),
            a: lambda: T.memref(BT, HV, elem_type),
            dt_bias: lambda: T.memref(HV, elem_type),
            q: lambda: T.memref(BTH, K, elem_type),
            k: lambda: T.memref(BTH, K, elem_type),
            v: lambda: T.memref(BTHV, V, elem_type),
            b: lambda: T.memref(BT, HV, elem_type),
            initial_state_source: lambda: T.memref(STATEHV, K, V, elem_type),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(BTHV, V, elem_type),
        ):
            c1 = flir.const_index(1)
            gx = flir.const_index(1)
            gy = flir.const_index((V + BV - 1) // BV)
            gz = flir.const_index(B * HV)
            bx = flir.const_index(BV)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_sigmoid_gating_delta_rule_update_fwd_kernel"],
                grid_size=(gx, gy, gz),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    A_log,
                    a,
                    dt_bias,
                    q,
                    k,
                    v,
                    b,
                    initial_state_source,
                    initial_state_indices,
                    o,
                ],
            )

    return _FusedSigmoidGatingDeltaRuleUpdateFwd()
