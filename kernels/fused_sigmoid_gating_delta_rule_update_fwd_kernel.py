"""FlyDSL fused sigmoid-gating delta-rule update-forward kernel.

MVP scope:
- fixed-length sequence only
- scalar beta only: b[B*T, HV], beta = sigmoid(b)
- reads/writes recurrent states from/to state pool via state indices
- compile-time output/state write switches
- dtype: f32
"""

from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
import _mlir.extras.types as T


KERNEL_NAME = "fused_sigmoid_gating_delta_rule_update_fwd_kernel"

# Keep these aligned with the Triton reference defaults for MVP.
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
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
):
    """Build fused sigmoid-gating + recurrent-delta-rule update-forward module.

    Runtime memref contracts (flattened):
    - A_log: [HV]
    - a: [B*T_seq, HV]
    - dt_bias: [HV]
    - q: [B*T_seq*H, K]
    - k: [B*T_seq*H, K]
    - v: [B*T_seq*HV, V]
    - b: [B*T_seq, HV]
    - initial_state_source: [N_STATE*HV, K, V]
    - initial_state_indices: [B] (int32)
    - o: [B*T_seq*HV, V]
    """
    if dtype_str != "f32":
        raise NotImplementedError("MVP currently supports dtype_str='f32' only.")
    if HV % H != 0:
        raise ValueError("HV must be divisible by H for hv->h mapping.")

    gpu_arch = get_hip_arch()
    compute_type = T.f32()
    elem_type = T.f32()
    scale = K ** (-0.5)
    h_smem_size = K * V

    allocator = SmemAllocator(None, arch=gpu_arch)
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
            _state["smem_h"] = allocator.allocate_array(compute_type, h_smem_size)
            allocator.finalize()

        @flir.kernel
        def fused_sigmoid_gating_delta_rule_update_fwd_kernel(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, T.f32()),
            a: lambda: T.memref(BT, HV, T.f32()),
            dt_bias: lambda: T.memref(HV, T.f32()),
            q: lambda: T.memref(BTH, K, T.f32()),
            k: lambda: T.memref(BTH, K, T.f32()),
            v: lambda: T.memref(BTHV, V, T.f32()),
            b: lambda: T.memref(BT, HV, T.f32()),
            initial_state_source: lambda: T.memref(STATEHV, K, V, T.f32()),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(BTHV, V, T.f32()),
        ):
            i_nh = flir.const_index(flir.block_idx("x"))
            c_hv = arith.as_value(arith.index(HV))
            i_n = arith.as_value(flir.arith.DivUIOp(arith.as_value(i_nh), c_hv).result)
            i_hv = arith.as_value(flir.arith.RemUIOp(arith.as_value(i_nh), c_hv).result)
            c_hv_per_h = arith.as_value(arith.index(HV // H))
            i_h = arith.as_value(
                flir.arith.DivUIOp(arith.as_value(i_hv), c_hv_per_h).result
            )

            base_ptr = allocator.get_base()
            s_h = _state["smem_h"](base_ptr).get()
            layout_h = flir.make_layout(flir.make_shape(K, V), flir.make_stride(V, 1))

            comp = compute_type
            c_log2e = arith.constant(1.4426950408889634, type=comp)
            c_ln2 = arith.constant(0.6931471805599453, type=comp)
            c_scale = arith.constant(scale, type=comp)
            c_neg_one = arith.constant(-1.0, type=comp)
            c_one = arith.constant(1.0, type=comp)
            c_softplus_beta = arith.constant(SOFTPLUS_BETA, type=comp)
            c_inv_softplus_beta = arith.constant(1.0 / SOFTPLUS_BETA, type=comp)
            c_softplus_threshold = arith.constant(SOFTPLUS_THRESHOLD, type=comp)
            fm_fast = flir.arith.FastMathFlags.fast

            c_TH = arith.as_value(arith.index(T_seq * H))
            c_THV = arith.as_value(arith.index(T_seq * HV))
            c_T = arith.as_value(arith.index(T_seq))
            c_H = arith.as_value(arith.index(H))

            # state row = state_idx * HV + hv
            state_idx_i32 = memref.load(initial_state_indices, [arith.as_value(i_n)])
            state_idx = arith.as_value(
                flir.arith.IndexCastOp(T.index(), arith.as_value(state_idx_i32)).result
            )
            state_mul = flir.arith.MulIOp(arith.as_value(state_idx), c_hv).result
            row_state = arith.as_value(
                flir.arith.AddIOp(arith.as_value(state_mul), arith.as_value(i_hv)).result
            )

            # Load initial recurrent state h from the state pool.
            for ik in range_constexpr(K):
                for iv in range_constexpr(V):
                    h_init = memref.load(
                        initial_state_source,
                        [arith.as_value(row_state), arith.index(ik), arith.index(iv)],
                    )
                    crd = flir.make_coord(arith.index(ik), arith.index(iv))
                    idx = flir.crd2idx(crd, layout_h)
                    memref.store(arith.as_value(h_init), s_h, [arith.as_value(idx)])

            gpu.barrier()

            for t in range_constexpr(T_seq):
                t_idx = arith.as_value(arith.index(t))

                # row_qt = n * (T*H) + t * H + h   for q, k
                m1 = flir.arith.MulIOp(arith.as_value(i_n), c_TH).result
                m2 = flir.arith.MulIOp(t_idx, c_H).result
                a1 = flir.arith.AddIOp(arith.as_value(m1), arith.as_value(m2)).result
                row_qt = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(a1), arith.as_value(i_h)).result
                )

                # row_vt = n * (T*HV) + t * HV + hv   for v, o
                m3 = flir.arith.MulIOp(arith.as_value(i_n), c_THV).result
                m4 = flir.arith.MulIOp(t_idx, c_hv).result
                a2 = flir.arith.AddIOp(arith.as_value(m3), arith.as_value(m4)).result
                row_vt = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(a2), arith.as_value(i_hv)).result
                )

                # row_ab = n * T + t   for a, b
                m5 = flir.arith.MulIOp(arith.as_value(i_n), c_T).result
                row_ab = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(m5), t_idx).result
                )

                # Load q, k, v.
                q_vals = []
                k_vals = []
                for ik in range_constexpr(K):
                    q_vals.append(memref.load(q, [arith.as_value(row_qt), arith.index(ik)]))
                    k_vals.append(memref.load(k, [arith.as_value(row_qt), arith.index(ik)]))

                v_vals = []
                for iv in range_constexpr(V):
                    v_vals.append(memref.load(v, [arith.as_value(row_vt), arith.index(iv)]))

                # g = -exp(A_log) * softplus(a + dt_bias)
                a_log_val = memref.load(A_log, [arith.as_value(i_hv)])
                dt_bias_val = memref.load(dt_bias, [arith.as_value(i_hv)])
                a_val = memref.load(a, [arith.as_value(row_ab), arith.as_value(i_hv)])
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

                # beta = sigmoid(b)
                b_val = memref.load(b, [arith.as_value(row_ab), arith.as_value(i_hv)])
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

                # h *= exp(g)
                g_log2e = flir.arith.MulFOp(
                    arith.as_value(g_val), arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                exp_g = flir.math.exp2(arith.as_value(g_log2e), fastmath=fm_fast)
                for ik in range_constexpr(K):
                    for iv in range_constexpr(V):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        old_h = memref.load(s_h, [arith.as_value(idx)])
                        h_scaled = flir.arith.MulFOp(
                            arith.as_value(old_h), arith.as_value(exp_g), fastmath=fm_fast
                        ).result
                        memref.store(h_scaled, s_h, [arith.as_value(idx)])

                # v -= sum(h * k, dim=0)
                for iv in range_constexpr(V):
                    s = arith.constant(0.0, type=comp)
                    for ik in range_constexpr(K):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        h_ikv = memref.load(s_h, [arith.as_value(idx)])
                        prod = flir.arith.MulFOp(
                            arith.as_value(h_ikv), arith.as_value(k_vals[ik]), fastmath=fm_fast
                        ).result
                        s = flir.arith.AddFOp(
                            arith.as_value(s), arith.as_value(prod), fastmath=fm_fast
                        ).result
                    v_vals[iv] = flir.arith.SubFOp(
                        arith.as_value(v_vals[iv]), arith.as_value(s)
                    ).result

                # v *= beta
                for iv in range_constexpr(V):
                    v_vals[iv] = flir.arith.MulFOp(
                        arith.as_value(v_vals[iv]),
                        arith.as_value(beta_val),
                        fastmath=fm_fast,
                    ).result

                # h += k ⊗ v
                for ik in range_constexpr(K):
                    for iv in range_constexpr(V):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        old_h = memref.load(s_h, [arith.as_value(idx)])
                        prod = flir.arith.MulFOp(
                            arith.as_value(k_vals[ik]),
                            arith.as_value(v_vals[iv]),
                            fastmath=fm_fast,
                        ).result
                        upd = flir.arith.AddFOp(
                            arith.as_value(old_h), arith.as_value(prod), fastmath=fm_fast
                        ).result
                        memref.store(upd, s_h, [arith.as_value(idx)])

                # o = sum(h * q * scale, dim=0)
                if not disable_output_calculation:
                    for iv in range_constexpr(V):
                        s = arith.constant(0.0, type=comp)
                        for ik in range_constexpr(K):
                            crd = flir.make_coord(arith.index(ik), arith.index(iv))
                            idx = flir.crd2idx(crd, layout_h)
                            h_ikv = memref.load(s_h, [arith.as_value(idx)])
                            q_scaled = flir.arith.MulFOp(
                                arith.as_value(q_vals[ik]),
                                arith.as_value(c_scale),
                                fastmath=fm_fast,
                            ).result
                            prod = flir.arith.MulFOp(
                                arith.as_value(h_ikv), arith.as_value(q_scaled), fastmath=fm_fast
                            ).result
                            s = flir.arith.AddFOp(
                                arith.as_value(s), arith.as_value(prod), fastmath=fm_fast
                            ).result
                        memref.store(
                            arith.as_value(s), o, [arith.as_value(row_vt), arith.index(iv)]
                        )

            # Write final recurrent state back to the shared state pool.
            if not disable_state_update:
                for ik in range_constexpr(K):
                    for iv in range_constexpr(V):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        h_fin = memref.load(s_h, [arith.as_value(idx)])
                        memref.store(
                            arith.as_value(h_fin),
                            initial_state_source,
                            [arith.as_value(row_state), arith.index(ik), arith.index(iv)],
                        )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, T.f32()),
            a: lambda: T.memref(BT, HV, T.f32()),
            dt_bias: lambda: T.memref(HV, T.f32()),
            q: lambda: T.memref(BTH, K, T.f32()),
            k: lambda: T.memref(BTH, K, T.f32()),
            v: lambda: T.memref(BTHV, V, T.f32()),
            b: lambda: T.memref(BT, HV, T.f32()),
            initial_state_source: lambda: T.memref(STATEHV, K, V, T.f32()),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(BTHV, V, T.f32()),
        ):
            c1 = flir.const_index(1)
            gx = flir.const_index(B * HV)
            bx = flir.const_index(1)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_sigmoid_gating_delta_rule_update_fwd_kernel"],
                grid_size=(gx, c1, c1),
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
