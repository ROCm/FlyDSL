"""FlyDSL fused split-GDR update-forward kernel (ksplit2, HIP-aligned)."""

from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
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

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _FusedSplitGdrUpdateKSplit2(flir.MlirModule):
        GPU_MODULE_NAME = f"fused_split_gdr_update_ksplit2_flyc_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            _state["smem_kq"] = allocator.allocate_array(compute_type, K)
            allocator.finalize()

        @flir.kernel
        def fused_split_gdr_update_ksplit2_flyc_kernel(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, T.f32()),
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
            kg_start = arith.as_value(
                flir.arith.DivUIOp(arith.as_value(k_base), arith.as_value(arith.index(4))).result
            )

            base_ptr = allocator.get_base()
            smem_kq = _state["smem_kq"](base_ptr)

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

            h_vec_x = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_vec_y = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_vec_z = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_vec_w = [c_zero_f32 for _ in range_constexpr(K // 8)]
            if state_idx_nonneg:
                if iv_valid:
                    for kg_local in range_constexpr(K // 8):
                        kg = arith.as_value(
                            flir.arith.AddIOp(
                                arith.as_value(kg_start), arith.as_value(arith.index(kg_local))
                            ).result
                        )
                        h_vec_x[kg_local] = memref.load(
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(0)),
                            ],
                        )
                        h_vec_y[kg_local] = memref.load(
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(1)),
                            ],
                        )
                        h_vec_z[kg_local] = memref.load(
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(2)),
                            ],
                        )
                        h_vec_w[kg_local] = memref.load(
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(3)),
                            ],
                        )

            a_log_val = memref.load(A_log, [arith.as_value(i_hv)])
            dt_bias_val = memref.load(dt_bias, [arith.as_value(i_hv)])
            if dtype_str != "f32":
                dt_bias_val = flir.arith.extf(comp, arith.as_value(dt_bias_val))
            neg_exp_a = flir.arith.MulFOp(
                arith.as_value(c_neg_one),
                arith.as_value(
                    flir.math.exp2(
                        arith.as_value(
                            flir.arith.MulFOp(
                                arith.as_value(a_log_val), arith.as_value(c_log2e), fastmath=fm_fast
                            ).result
                        ),
                        fastmath=fm_fast,
                    )
                ),
                fastmath=fm_fast,
            ).result

            m5 = flir.arith.MulIOp(arith.as_value(i_n), c_T).result
            for t in range_constexpr(T_seq):
                t_idx = arith.as_value(arith.index(t))
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

                a_val = memref.load(a, [arith.as_value(row_ab), arith.as_value(i_hv)])
                if dtype_str != "f32":
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
                exp_a_log_softplus = flir.arith.MulFOp(
                    arith.as_value(neg_exp_a), arith.as_value(softplus_x), fastmath=fm_fast
                ).result
                g_val = exp_a_log_softplus

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

                # K tiles: cooperative load to LDS, then reuse in phase1/phase2.
                k_pair_base = arith.as_value(
                    flir.arith.MulIOp(arith.as_value(tid), arith.as_value(arith.index(2))).result
                )
                k_pair_valid0 = arith.CmpIOp(
                    arith.CmpIPredicate.ult,
                    arith.as_value(k_pair_base),
                    arith.as_value(c_K_idx),
                )
                if k_pair_valid0:
                    k_col0 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_k_dim_off), arith.as_value(k_pair_base)).result
                    )
                    k0 = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(k_col0), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        k0 = flir.arith.extf(comp, arith.as_value(k0))
                    smem_kq.store(k0, [arith.as_value(k_pair_base)])
                k_pair_base1 = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(k_pair_base), arith.as_value(arith.index(1))).result
                )
                k_pair_valid1 = arith.CmpIOp(
                    arith.CmpIPredicate.ult,
                    arith.as_value(k_pair_base1),
                    arith.as_value(c_K_idx),
                )
                if k_pair_valid1:
                    k_col1 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_k_dim_off), arith.as_value(k_pair_base1)).result
                    )
                    k1 = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(k_col1), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        k1 = flir.arith.extf(comp, arith.as_value(k1))
                    smem_kq.store(k1, [arith.as_value(k_pair_base1)])
                gpu.barrier()

                acc_hk = arith.constant(0.0, type=comp)
                k_sq_local = arith.constant(0.0, type=comp)
                k_vec0 = []
                k_vec1 = []
                k_vec2 = []
                k_vec3 = []
                for kg_local in range_constexpr(K // 8):
                    k_idx0 = arith.as_value(
                        flir.arith.AddIOp(
                            arith.as_value(k_base),
                            arith.as_value(arith.index(kg_local * 4)),
                        ).result
                    )
                    k_idx1 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(k_idx0), arith.as_value(arith.index(1))).result
                    )
                    k_idx2 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(k_idx0), arith.as_value(arith.index(2))).result
                    )
                    k_idx3 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(k_idx0), arith.as_value(arith.index(3))).result
                    )
                    k0 = smem_kq.load([arith.as_value(k_idx0)])
                    k1 = smem_kq.load([arith.as_value(k_idx1)])
                    k2 = smem_kq.load([arith.as_value(k_idx2)])
                    k3 = smem_kq.load([arith.as_value(k_idx3)])
                    k_vec0.append(k0)
                    k_vec1.append(k1)
                    k_vec2.append(k2)
                    k_vec3.append(k3)
                    prod0 = flir.arith.MulFOp(
                        arith.as_value(h_vec_x[kg_local]), arith.as_value(k0), fastmath=fm_fast
                    ).result
                    prod1 = flir.arith.MulFOp(
                        arith.as_value(h_vec_y[kg_local]), arith.as_value(k1), fastmath=fm_fast
                    ).result
                    prod2 = flir.arith.MulFOp(
                        arith.as_value(h_vec_z[kg_local]), arith.as_value(k2), fastmath=fm_fast
                    ).result
                    prod3 = flir.arith.MulFOp(
                        arith.as_value(h_vec_w[kg_local]), arith.as_value(k3), fastmath=fm_fast
                    ).result
                    sum01 = flir.arith.AddFOp(
                        arith.as_value(prod0), arith.as_value(prod1), fastmath=fm_fast
                    ).result
                    sum23 = flir.arith.AddFOp(
                        arith.as_value(prod2), arith.as_value(prod3), fastmath=fm_fast
                    ).result
                    sum0123 = flir.arith.AddFOp(
                        arith.as_value(sum01), arith.as_value(sum23), fastmath=fm_fast
                    ).result
                    acc_hk = flir.arith.AddFOp(
                        arith.as_value(acc_hk), arith.as_value(sum0123), fastmath=fm_fast
                    ).result
                    if use_qk_l2norm_in_kernel:
                        k0_sq = flir.arith.MulFOp(
                            arith.as_value(k0), arith.as_value(k0), fastmath=fm_fast
                        ).result
                        k1_sq = flir.arith.MulFOp(
                            arith.as_value(k1), arith.as_value(k1), fastmath=fm_fast
                        ).result
                        k2_sq = flir.arith.MulFOp(
                            arith.as_value(k2), arith.as_value(k2), fastmath=fm_fast
                        ).result
                        k3_sq = flir.arith.MulFOp(
                            arith.as_value(k3), arith.as_value(k3), fastmath=fm_fast
                        ).result
                        sq01 = flir.arith.AddFOp(
                            arith.as_value(k0_sq), arith.as_value(k1_sq), fastmath=fm_fast
                        ).result
                        sq23 = flir.arith.AddFOp(
                            arith.as_value(k2_sq), arith.as_value(k3_sq), fastmath=fm_fast
                        ).result
                        sq0123 = flir.arith.AddFOp(
                            arith.as_value(sq01), arith.as_value(sq23), fastmath=fm_fast
                        ).result
                        k_sq_local = flir.arith.AddFOp(
                            arith.as_value(k_sq_local), arith.as_value(sq0123), fastmath=fm_fast
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

                for kg_local in range_constexpr(K // 8):
                    k0 = k_vec0[kg_local]
                    k1 = k_vec1[kg_local]
                    k2 = k_vec2[kg_local]
                    k3 = k_vec3[kg_local]
                    if use_qk_l2norm_in_kernel:
                        k0 = flir.arith.MulFOp(
                            arith.as_value(k0), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                        k1 = flir.arith.MulFOp(
                            arith.as_value(k1), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                        k2 = flir.arith.MulFOp(
                            arith.as_value(k2), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                        k3 = flir.arith.MulFOp(
                            arith.as_value(k3), arith.as_value(k_inv_norm), fastmath=fm_fast
                        ).result
                    kv0 = flir.arith.MulFOp(
                        arith.as_value(k0), arith.as_value(v_val), fastmath=fm_fast
                    ).result
                    kv1 = flir.arith.MulFOp(
                        arith.as_value(k1), arith.as_value(v_val), fastmath=fm_fast
                    ).result
                    kv2 = flir.arith.MulFOp(
                        arith.as_value(k2), arith.as_value(v_val), fastmath=fm_fast
                    ).result
                    kv3 = flir.arith.MulFOp(
                        arith.as_value(k3), arith.as_value(v_val), fastmath=fm_fast
                    ).result
                    h_decay0 = flir.arith.MulFOp(
                        arith.as_value(h_vec_x[kg_local]), arith.as_value(exp_g), fastmath=fm_fast
                    ).result
                    h_decay1 = flir.arith.MulFOp(
                        arith.as_value(h_vec_y[kg_local]), arith.as_value(exp_g), fastmath=fm_fast
                    ).result
                    h_decay2 = flir.arith.MulFOp(
                        arith.as_value(h_vec_z[kg_local]), arith.as_value(exp_g), fastmath=fm_fast
                    ).result
                    h_decay3 = flir.arith.MulFOp(
                        arith.as_value(h_vec_w[kg_local]), arith.as_value(exp_g), fastmath=fm_fast
                    ).result
                    h_vec_x[kg_local] = flir.arith.AddFOp(
                        arith.as_value(h_decay0), arith.as_value(kv0), fastmath=fm_fast
                    ).result
                    h_vec_y[kg_local] = flir.arith.AddFOp(
                        arith.as_value(h_decay1), arith.as_value(kv1), fastmath=fm_fast
                    ).result
                    h_vec_z[kg_local] = flir.arith.AddFOp(
                        arith.as_value(h_decay2), arith.as_value(kv2), fastmath=fm_fast
                    ).result
                    h_vec_w[kg_local] = flir.arith.AddFOp(
                        arith.as_value(h_decay3), arith.as_value(kv3), fastmath=fm_fast
                    ).result

                # Q tiles: cooperative load to LDS, then reuse in phase3.
                if k_pair_valid0:
                    q_col0 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_q_dim_off), arith.as_value(k_pair_base)).result
                    )
                    q0 = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(q_col0), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        q0 = flir.arith.extf(comp, arith.as_value(q0))
                    smem_kq.store(q0, [arith.as_value(k_pair_base)])
                if k_pair_valid1:
                    q_col1 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_q_dim_off), arith.as_value(k_pair_base1)).result
                    )
                    q1 = memref.load(
                        mixed_qkv,
                        [arith.as_value(i_n), arith.as_value(q_col1), arith.as_value(t_idx)],
                    )
                    if dtype_str != "f32":
                        q1 = flir.arith.extf(comp, arith.as_value(q1))
                    smem_kq.store(q1, [arith.as_value(k_pair_base1)])
                gpu.barrier()

                o_acc = arith.constant(0.0, type=comp)
                q_sq_local = arith.constant(0.0, type=comp)
                for kg_local in range_constexpr(K // 8):
                    q_idx0 = arith.as_value(
                        flir.arith.AddIOp(
                            arith.as_value(k_base),
                            arith.as_value(arith.index(kg_local * 4)),
                        ).result
                    )
                    q_idx1 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(q_idx0), arith.as_value(arith.index(1))).result
                    )
                    q_idx2 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(q_idx0), arith.as_value(arith.index(2))).result
                    )
                    q_idx3 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(q_idx0), arith.as_value(arith.index(3))).result
                    )
                    q0 = smem_kq.load([arith.as_value(q_idx0)])
                    q1 = smem_kq.load([arith.as_value(q_idx1)])
                    q2 = smem_kq.load([arith.as_value(q_idx2)])
                    q3 = smem_kq.load([arith.as_value(q_idx3)])
                    q0s = flir.arith.MulFOp(
                        arith.as_value(q0), arith.as_value(c_scale), fastmath=fm_fast
                    ).result
                    q1s = flir.arith.MulFOp(
                        arith.as_value(q1), arith.as_value(c_scale), fastmath=fm_fast
                    ).result
                    q2s = flir.arith.MulFOp(
                        arith.as_value(q2), arith.as_value(c_scale), fastmath=fm_fast
                    ).result
                    q3s = flir.arith.MulFOp(
                        arith.as_value(q3), arith.as_value(c_scale), fastmath=fm_fast
                    ).result
                    prod0 = flir.arith.MulFOp(
                        arith.as_value(h_vec_x[kg_local]), arith.as_value(q0s), fastmath=fm_fast
                    ).result
                    prod1 = flir.arith.MulFOp(
                        arith.as_value(h_vec_y[kg_local]), arith.as_value(q1s), fastmath=fm_fast
                    ).result
                    prod2 = flir.arith.MulFOp(
                        arith.as_value(h_vec_z[kg_local]), arith.as_value(q2s), fastmath=fm_fast
                    ).result
                    prod3 = flir.arith.MulFOp(
                        arith.as_value(h_vec_w[kg_local]), arith.as_value(q3s), fastmath=fm_fast
                    ).result
                    sum01 = flir.arith.AddFOp(
                        arith.as_value(prod0), arith.as_value(prod1), fastmath=fm_fast
                    ).result
                    sum23 = flir.arith.AddFOp(
                        arith.as_value(prod2), arith.as_value(prod3), fastmath=fm_fast
                    ).result
                    sum0123 = flir.arith.AddFOp(
                        arith.as_value(sum01), arith.as_value(sum23), fastmath=fm_fast
                    ).result
                    o_acc = flir.arith.AddFOp(
                        arith.as_value(o_acc), arith.as_value(sum0123), fastmath=fm_fast
                    ).result
                    if use_qk_l2norm_in_kernel:
                        q0_sq = flir.arith.MulFOp(
                            arith.as_value(q0), arith.as_value(q0), fastmath=fm_fast
                        ).result
                        q1_sq = flir.arith.MulFOp(
                            arith.as_value(q1), arith.as_value(q1), fastmath=fm_fast
                        ).result
                        q2_sq = flir.arith.MulFOp(
                            arith.as_value(q2), arith.as_value(q2), fastmath=fm_fast
                        ).result
                        q3_sq = flir.arith.MulFOp(
                            arith.as_value(q3), arith.as_value(q3), fastmath=fm_fast
                        ).result
                        sq01 = flir.arith.AddFOp(
                            arith.as_value(q0_sq), arith.as_value(q1_sq), fastmath=fm_fast
                        ).result
                        sq23 = flir.arith.AddFOp(
                            arith.as_value(q2_sq), arith.as_value(q3_sq), fastmath=fm_fast
                        ).result
                        sq0123 = flir.arith.AddFOp(
                            arith.as_value(sq01), arith.as_value(sq23), fastmath=fm_fast
                        ).result
                        q_sq_local = flir.arith.AddFOp(
                            arith.as_value(q_sq_local), arith.as_value(sq0123), fastmath=fm_fast
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

            if iv_valid:
                if state_idx_nonneg:
                    for kg_local in range_constexpr(K // 8):
                        kg = arith.as_value(
                            flir.arith.AddIOp(
                                arith.as_value(kg_start), arith.as_value(arith.index(kg_local))
                            ).result
                        )
                        memref.store(
                            arith.as_value(h_vec_x[kg_local]),
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(0)),
                            ],
                        )
                        memref.store(
                            arith.as_value(h_vec_y[kg_local]),
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(1)),
                            ],
                        )
                        memref.store(
                            arith.as_value(h_vec_z[kg_local]),
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(2)),
                            ],
                        )
                        memref.store(
                            arith.as_value(h_vec_w[kg_local]),
                            initial_state_source,
                            [
                                arith.as_value(row_state),
                                arith.as_value(kg),
                                arith.as_value(iv_safe),
                                arith.as_value(arith.index(3)),
                            ],
                        )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A_log: lambda: T.memref(HV, T.f32()),
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
