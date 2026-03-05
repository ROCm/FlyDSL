"""FlyDSL fused split-GDR update-forward kernel (ksplit2, HIP-aligned)."""

from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
import _mlir.extras.types as T


KERNEL_NAME = "fused_split_gdr_update_ksplit2_flyc_kernel"
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0


def build_fused_split_gdr_update_ksplit2_flyc_module(
    B: int,
    T_seq: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    N_STATE: int,
    key_dim: int | None = None,
    value_dim: int | None = None,
    dtype_str: str = "f32",
    BV: int = 64,
    softplus_beta: float = SOFTPLUS_BETA,
    softplus_threshold: float = SOFTPLUS_THRESHOLD,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Build HIP-aligned split-GDR ksplit2 FlyDSL module."""
    if dtype_str not in {"f32", "bf16"}:
        raise NotImplementedError("Supported dtype_str values are 'f32' and 'bf16'.")
    if HV % H != 0:
        raise ValueError("HV must be divisible by H for hv->h mapping.")
    if K % 4 != 0:
        raise ValueError("K must be divisible by 4 for swizzled state layout.")
    if BV <= 0 or BV % 2 != 0:
        raise ValueError("BV must be positive and divisible by 2 for ksplit2 mapping.")

    block_threads = min(BV, 2 * V)
    if block_threads <= 0 or block_threads % 2 != 0:
        raise ValueError("block_threads must be positive and divisible by 2.")

    if key_dim is None:
        key_dim = H * K
    if value_dim is None:
        value_dim = HV * V
    mixed_dim = 2 * key_dim + value_dim

    gpu_arch = get_hip_arch()
    compute_type = T.f32()
    elem_type = T.f32() if dtype_str == "f32" else T.bf16()
    state_elem_type = T.f32()
    scale = K ** (-0.5)
    BT = B * T_seq
    STATEHV = N_STATE * HV

    class _FusedSplitGdrUpdateKSplit2(flir.MlirModule):
        GPU_MODULE_NAME = f"fused_split_gdr_update_ksplit2_flyc_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            pass

        @flir.kernel
        def fused_split_gdr_update_ksplit2_flyc_kernel(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, elem_type),
            a: lambda: T.memref(BT, HV, elem_type),
            dt_bias: lambda: T.memref(HV, elem_type),
            mixed_qkv: lambda: T.memref(B, mixed_dim, T_seq, elem_type),
            b: lambda: T.memref(BT, HV, elem_type),
            initial_state_source: lambda: T.memref(STATEHV, K // 4, V, 4, state_elem_type),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(B, T_seq, HV, V, elem_type),
        ):
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
            c_bv = arith.as_value(arith.index(block_threads // 2))
            iv_base = arith.as_value(
                flir.arith.MulIOp(arith.as_value(i_v_tile), c_bv).result
            )
            lane_i32 = arith.as_value(
                flir.arith.IndexCastOp(T.i32(), arith.as_value(tid)).result
            )
            c_one_i32 = arith.constant(1, type=T.i32())
            v_idx_i32 = flir.arith.ShRUIOp(
                arith.as_value(lane_i32), arith.as_value(c_one_i32)
            ).result
            v_idx = arith.as_value(
                flir.arith.IndexCastOp(T.index(), arith.as_value(v_idx_i32)).result
            )
            iv_global = arith.as_value(
                flir.arith.AddIOp(arith.as_value(iv_base), arith.as_value(v_idx)).result
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
            lane_lsb = arith.andi(arith.as_value(lane_i32), arith.as_value(c_one_i32))
            c_zero_i32 = arith.constant(0, type=T.i32())
            c_two_i32 = arith.constant(2, type=T.i32())
            is_primary_lane = arith.CmpIOp(
                arith.CmpIPredicate.eq,
                arith.as_value(lane_lsb),
                arith.as_value(c_zero_i32),
            )
            partner_lane_i32 = arith.xori(arith.as_value(lane_i32), arith.as_value(c_one_i32))
            partner_lane_bytes = flir.arith.ShLIOp(
                arith.as_value(partner_lane_i32), arith.as_value(c_two_i32)
            ).result

            comp = compute_type
            c_log2e = arith.constant(1.4426950408889634, type=comp)
            c_ln2 = arith.constant(0.6931471805599453, type=comp)
            c_scale = arith.constant(scale, type=comp)
            c_neg_one = arith.constant(-1.0, type=comp)
            c_one = arith.constant(1.0, type=comp)
            c_softplus_beta = arith.constant(softplus_beta, type=comp)
            c_inv_softplus_beta = arith.constant(1.0 / softplus_beta, type=comp)
            c_softplus_threshold = arith.constant(softplus_threshold, type=comp)
            c_eps = arith.constant(1e-6, type=comp)
            fm_fast = flir.arith.FastMathFlags.fast
            c_zero_f32 = arith.constant(0.0, type=comp)
            c_T = arith.as_value(arith.index(T_seq))
            c_K_idx = arith.as_value(arith.index(K))
            c_V_idx = arith.as_value(arith.index(V))
            c_key_dim = arith.as_value(arith.index(key_dim))
            c_v_base = arith.as_value(arith.index(2 * key_dim))
            c_q_dim_off = arith.as_value(
                flir.arith.MulIOp(arith.as_value(i_h), arith.as_value(c_K_idx)).result
            )
            c_k_dim_off = arith.as_value(
                flir.arith.AddIOp(arith.as_value(c_key_dim), arith.as_value(c_q_dim_off)).result
            )
            c_v_h_off = arith.as_value(
                flir.arith.MulIOp(arith.as_value(i_hv), arith.as_value(c_V_idx)).result
            )
            c_v_dim_off = arith.as_value(
                flir.arith.AddIOp(arith.as_value(c_v_base), arith.as_value(c_v_h_off)).result
            )
            c_k_half = arith.as_value(arith.index(K // 2))
            k_base = arith.select(
                arith.as_value(is_primary_lane),
                arith.as_value(arith.index(0)),
                arith.as_value(c_k_half),
            )

            state_idx_i32 = memref.load(initial_state_indices, [arith.as_value(i_n)])
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

            h_reg = []
            for ik_local in range_constexpr(K // 2):
                k_idx = arith.as_value(
                    flir.arith.AddIOp(
                        arith.as_value(k_base), arith.as_value(arith.index(ik_local))
                    ).result
                )
                kg = arith.as_value(
                    flir.arith.DivUIOp(arith.as_value(k_idx), arith.as_value(arith.index(4))).result
                )
                k4 = arith.as_value(
                    flir.arith.RemUIOp(arith.as_value(k_idx), arith.as_value(arith.index(4))).result
                )
                h_init = memref.load(
                    initial_state_source,
                    [
                        arith.as_value(row_state),
                        arith.as_value(kg),
                        arith.as_value(iv_safe),
                        arith.as_value(k4),
                    ],
                )
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

            for t in range_constexpr(T_seq):
                t_idx = arith.as_value(arith.index(t))
                m5 = flir.arith.MulIOp(arith.as_value(i_n), c_T).result
                row_ab = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(m5), t_idx).result
                )

                k_inv_norm = c_one
                v_col = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(c_v_dim_off), arith.as_value(iv_safe)).result
                )
                v_val = memref.load(
                    mixed_qkv,
                    [arith.as_value(i_n), arith.as_value(v_col), arith.as_value(t_idx)],
                )
                if dtype_str != "f32":
                    v_val = flir.arith.extf(comp, arith.as_value(v_val))
                v_val = arith.select(
                    arith.as_value(iv_valid),
                    arith.as_value(v_val),
                    arith.as_value(c_zero_f32),
                )

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

                g_log2e = flir.arith.MulFOp(
                    arith.as_value(g_val), arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                exp_g = flir.math.exp2(arith.as_value(g_log2e), fastmath=fm_fast)

                acc_hk = arith.constant(0.0, type=comp)
                k_sq_local = arith.constant(0.0, type=comp)
                for ik_local in range_constexpr(K // 2):
                    k_idx = arith.as_value(
                        flir.arith.AddIOp(
                            arith.as_value(k_base), arith.as_value(arith.index(ik_local))
                        ).result
                    )
                    k_col = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_k_dim_off), arith.as_value(k_idx)).result
                    )
                    k_ik = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(k_col), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        k_ik = flir.arith.extf(comp, arith.as_value(k_ik))
                    prod = flir.arith.MulFOp(
                        arith.as_value(h_reg[ik_local]), arith.as_value(k_ik), fastmath=fm_fast
                    ).result
                    acc_hk = flir.arith.AddFOp(
                        arith.as_value(acc_hk), arith.as_value(prod), fastmath=fm_fast
                    ).result
                    if use_qk_l2norm_in_kernel:
                        k_sq_i = flir.arith.MulFOp(
                            arith.as_value(k_ik), arith.as_value(k_ik), fastmath=fm_fast
                        ).result
                        k_sq_local = flir.arith.AddFOp(
                            arith.as_value(k_sq_local), arith.as_value(k_sq_i), fastmath=fm_fast
                        ).result
                acc_hk_peer = rocdl.ds_bpermute(
                    T.i32(),
                    arith.as_value(partner_lane_bytes),
                    arith.as_value(flir.arith.bitcast(T.i32(), arith.as_value(acc_hk))),
                )
                acc_hk_peer = flir.arith.bitcast(comp, arith.as_value(acc_hk_peer))
                acc_hk = flir.arith.AddFOp(
                    arith.as_value(acc_hk), arith.as_value(acc_hk_peer), fastmath=fm_fast
                ).result
                if use_qk_l2norm_in_kernel:
                    k_sq_peer = rocdl.ds_bpermute(
                        T.i32(),
                        arith.as_value(partner_lane_bytes),
                        arith.as_value(flir.arith.bitcast(T.i32(), arith.as_value(k_sq_local))),
                    )
                    k_sq_peer = flir.arith.bitcast(comp, arith.as_value(k_sq_peer))
                    k_sq = flir.arith.AddFOp(
                        arith.as_value(k_sq_local), arith.as_value(k_sq_peer), fastmath=fm_fast
                    ).result
                    k_inv_norm = flir.math.rsqrt(
                        arith.as_value(
                            flir.arith.AddFOp(
                                arith.as_value(k_sq), arith.as_value(c_eps), fastmath=fm_fast
                            ).result
                        ),
                        fastmath=fm_fast,
                    )
                decayed_dot = flir.arith.MulFOp(
                    arith.as_value(acc_hk), arith.as_value(exp_g), fastmath=fm_fast
                ).result
                if use_qk_l2norm_in_kernel:
                    decayed_dot = flir.arith.MulFOp(
                        arith.as_value(decayed_dot), arith.as_value(k_inv_norm), fastmath=fm_fast
                    ).result
                v_val = flir.arith.SubFOp(
                    arith.as_value(v_val), arith.as_value(decayed_dot)
                ).result
                v_val = flir.arith.MulFOp(
                    arith.as_value(v_val), arith.as_value(beta_val), fastmath=fm_fast
                ).result

                for ik_local in range_constexpr(K // 2):
                    k_idx = arith.as_value(
                        flir.arith.AddIOp(
                            arith.as_value(k_base), arith.as_value(arith.index(ik_local))
                        ).result
                    )
                    k_col = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_k_dim_off), arith.as_value(k_idx)).result
                    )
                    k_ik = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(k_col), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        k_ik = flir.arith.extf(comp, arith.as_value(k_ik))
                    if use_qk_l2norm_in_kernel:
                        k_ik = flir.arith.MulFOp(
                            arith.as_value(k_ik), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                    prod = flir.arith.MulFOp(
                        arith.as_value(k_ik), arith.as_value(v_val), fastmath=fm_fast
                    ).result
                    h_decay = flir.arith.MulFOp(
                        arith.as_value(h_reg[ik_local]), arith.as_value(exp_g), fastmath=fm_fast
                    ).result
                    h_reg[ik_local] = flir.arith.AddFOp(
                        arith.as_value(h_decay), arith.as_value(prod), fastmath=fm_fast
                    ).result

                o_acc = arith.constant(0.0, type=comp)
                q_sq_local = arith.constant(0.0, type=comp)
                for ik_local in range_constexpr(K // 2):
                    k_idx = arith.as_value(
                        flir.arith.AddIOp(
                            arith.as_value(k_base), arith.as_value(arith.index(ik_local))
                        ).result
                    )
                    q_col = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_q_dim_off), arith.as_value(k_idx)).result
                    )
                    q_ik = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(q_col), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        q_ik = flir.arith.extf(comp, arith.as_value(q_ik))
                    q_scaled = flir.arith.MulFOp(
                        arith.as_value(q_ik), arith.as_value(c_scale), fastmath=fm_fast
                    ).result
                    prod = flir.arith.MulFOp(
                        arith.as_value(h_reg[ik_local]), arith.as_value(q_scaled), fastmath=fm_fast
                    ).result
                    o_acc = flir.arith.AddFOp(
                        arith.as_value(o_acc), arith.as_value(prod), fastmath=fm_fast
                    ).result
                    if use_qk_l2norm_in_kernel:
                        q_sq_i = flir.arith.MulFOp(
                            arith.as_value(q_ik), arith.as_value(q_ik), fastmath=fm_fast
                        ).result
                        q_sq_local = flir.arith.AddFOp(
                            arith.as_value(q_sq_local), arith.as_value(q_sq_i), fastmath=fm_fast
                        ).result
                o_acc_peer = rocdl.ds_bpermute(
                    T.i32(),
                    arith.as_value(partner_lane_bytes),
                    arith.as_value(flir.arith.bitcast(T.i32(), arith.as_value(o_acc))),
                )
                o_acc_peer = flir.arith.bitcast(comp, arith.as_value(o_acc_peer))
                o_acc = flir.arith.AddFOp(
                    arith.as_value(o_acc), arith.as_value(o_acc_peer), fastmath=fm_fast
                ).result
                q_inv_norm = c_one
                if use_qk_l2norm_in_kernel:
                    q_sq_peer = rocdl.ds_bpermute(
                        T.i32(),
                        arith.as_value(partner_lane_bytes),
                        arith.as_value(flir.arith.bitcast(T.i32(), arith.as_value(q_sq_local))),
                    )
                    q_sq_peer = flir.arith.bitcast(comp, arith.as_value(q_sq_peer))
                    q_sq = flir.arith.AddFOp(
                        arith.as_value(q_sq_local), arith.as_value(q_sq_peer), fastmath=fm_fast
                    ).result
                    q_inv_norm = flir.math.rsqrt(
                        arith.as_value(
                            flir.arith.AddFOp(
                                arith.as_value(q_sq), arith.as_value(c_eps), fastmath=fm_fast
                            ).result
                        ),
                        fastmath=fm_fast,
                    )
                o_acc = flir.arith.MulFOp(
                    arith.as_value(o_acc), arith.as_value(q_inv_norm), fastmath=fm_fast
                ).result
                out_elem = (
                    o_acc
                    if dtype_str == "f32"
                    else flir.arith.truncf(elem_type, arith.as_value(o_acc))
                )
                if iv_valid:
                    if is_primary_lane:
                        memref.store(
                            arith.as_value(out_elem),
                            o,
                            [
                                arith.as_value(i_n),
                                arith.as_value(t_idx),
                                arith.as_value(i_hv),
                                arith.as_value(iv_safe),
                            ],
                        )

            for ik_local in range_constexpr(K // 2):
                k_idx = arith.as_value(
                    flir.arith.AddIOp(
                        arith.as_value(k_base), arith.as_value(arith.index(ik_local))
                    ).result
                )
                kg = arith.as_value(
                    flir.arith.DivUIOp(arith.as_value(k_idx), arith.as_value(arith.index(4))).result
                )
                k4 = arith.as_value(
                    flir.arith.RemUIOp(arith.as_value(k_idx), arith.as_value(arith.index(4))).result
                )
                h_fin_elem = h_reg[ik_local]
                old_h = memref.load(
                    initial_state_source,
                    [
                        arith.as_value(row_state),
                        arith.as_value(kg),
                        arith.as_value(iv_safe),
                        arith.as_value(k4),
                    ],
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
                if iv_valid:
                    if state_idx_nonneg:
                        memref.store(
                            arith.as_value(new_h),
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(k4),
                            ],
                        )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, elem_type),
            a: lambda: T.memref(BT, HV, elem_type),
            dt_bias: lambda: T.memref(HV, elem_type),
            mixed_qkv: lambda: T.memref(B, mixed_dim, T_seq, elem_type),
            b: lambda: T.memref(BT, HV, elem_type),
            initial_state_source: lambda: T.memref(STATEHV, K // 4, V, 4, state_elem_type),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(B, T_seq, HV, V, elem_type),
        ):
            c1 = flir.const_index(1)
            gx = flir.const_index(1)
            gy = flir.const_index((V + (block_threads // 2) - 1) // (block_threads // 2))
            gz = flir.const_index(B * HV)
            bx = flir.const_index(block_threads)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME],
                grid_size=(gx, gy, gz),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    A_log,
                    a,
                    dt_bias,
                    mixed_qkv,
                    b,
                    initial_state_source,
                    initial_state_indices,
                    o,
                ],
            )

    return _FusedSplitGdrUpdateKSplit2()
