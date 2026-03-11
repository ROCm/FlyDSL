"""FlyDSL fused split-GDR update-forward kernel (ksplit2)."""

from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref, rocdl
from flydsl.runtime.device import get_rocm_arch
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
    """Build split-GDR ksplit2 FlyDSL module."""
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

    gpu_arch = get_rocm_arch()
    compute_type = T.f32()
    elem_type = T.f32() if dtype_str == "f32" else T.bf16()
    state_elem_type = T.f32()
    scale = K ** (-0.5)
    BT = B * T_seq
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}
    _asv = arith.as_value
    _extf = flir.arith.extf

    def _load_mixed_qkv_fp32(mixed_qkv, i_n, col, t_idx):
        val = memref.load(mixed_qkv, [_asv(i_n), _asv(col), _asv(t_idx)])
        if dtype_str != "f32":
            val = _extf(compute_type, _asv(val))
        return val

    def _load_state_lane(initial_state_source, state_idx, i_hv, kg, iv_safe, lane_comp):
        return memref.load(
            initial_state_source,
            [
                _asv(state_idx),
                _asv(i_hv),
                _asv(kg),
                _asv(iv_safe),
                _asv(arith.index(lane_comp)),
            ],
        )

    def _store_state_lane(
        initial_state_source, value, state_idx, i_hv, kg, iv_safe, lane_comp
    ):
        memref.store(
            _asv(value),
            initial_state_source,
            [
                _asv(state_idx),
                _asv(i_hv),
                _asv(kg),
                _asv(iv_safe),
                _asv(arith.index(lane_comp)),
            ],
        )

    def _add4(v0, v1, v2, v3, fm_fast):
        sum01 = flir.arith.AddFOp(_asv(v0), _asv(v1), fastmath=fm_fast).result
        sum23 = flir.arith.AddFOp(_asv(v2), _asv(v3), fastmath=fm_fast).result
        return flir.arith.AddFOp(_asv(sum01), _asv(sum23), fastmath=fm_fast).result

    def _dot4(a0, a1, a2, a3, b0, b1, b2, b3, fm_fast):
        prod0 = flir.arith.MulFOp(_asv(a0), _asv(b0), fastmath=fm_fast).result
        prod1 = flir.arith.MulFOp(_asv(a1), _asv(b1), fastmath=fm_fast).result
        prod2 = flir.arith.MulFOp(_asv(a2), _asv(b2), fastmath=fm_fast).result
        prod3 = flir.arith.MulFOp(_asv(a3), _asv(b3), fastmath=fm_fast).result
        return _add4(prod0, prod1, prod2, prod3, fm_fast)

    def _sumsq4(v0, v1, v2, v3, fm_fast):
        sq0 = flir.arith.MulFOp(_asv(v0), _asv(v0), fastmath=fm_fast).result
        sq1 = flir.arith.MulFOp(_asv(v1), _asv(v1), fastmath=fm_fast).result
        sq2 = flir.arith.MulFOp(_asv(v2), _asv(v2), fastmath=fm_fast).result
        sq3 = flir.arith.MulFOp(_asv(v3), _asv(v3), fastmath=fm_fast).result
        return _add4(sq0, sq1, sq2, sq3, fm_fast)

    def _smem_load4(smem_kq, idx0):
        idx1 = arith.as_value(
            flir.arith.AddIOp(arith.as_value(idx0), arith.as_value(arith.index(1))).result
        )
        idx2 = arith.as_value(
            flir.arith.AddIOp(arith.as_value(idx0), arith.as_value(arith.index(2))).result
        )
        idx3 = arith.as_value(
            flir.arith.AddIOp(arith.as_value(idx0), arith.as_value(arith.index(3))).result
        )
        v0 = smem_kq.load([arith.as_value(idx0)])
        v1 = smem_kq.load([arith.as_value(idx1)])
        v2 = smem_kq.load([arith.as_value(idx2)])
        v3 = smem_kq.load([arith.as_value(idx3)])
        return v0, v1, v2, v3

    class _FusedSplitGdrUpdateKSplit2(flir.MlirModule):
        GPU_MODULE_NAME = f"fused_split_gdr_update_ksplit2_flyc_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            _state["smem_kq"] = allocator.allocate_array(compute_type, K)
            allocator.finalize()

        @flir.kernel
        def fused_split_gdr_update_ksplit2_flyc_kernel(
            self: flir.T.i64,
            mixed_qkv: lambda: T.memref(B, mixed_dim, T_seq, elem_type),
            A_log: lambda: T.memref(HV, T.f32()),
            a: lambda: T.memref(BT, HV, elem_type),
            dt_bias: lambda: T.memref(HV, elem_type),
            b_gate: lambda: T.memref(BT, HV, elem_type),
            initial_state_source: lambda: T.memref(N_STATE, HV, K // 4, V, 4, state_elem_type),
            initial_state_indices: lambda: T.memref(B, T.i32()),
            o: lambda: T.memref(B, T_seq, HV, V, elem_type),
        ):
            # Kernel math per timestep:
            # 1) g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b_gate)
            # 2) v <- (v - <h, k> * exp(g)) * beta
            # 3) h <- h * exp(g) + k * v
            # 4) o <- <h, q * scale>
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

            h_vec_x = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_vec_y = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_vec_z = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_vec_w = [c_zero_f32 for _ in range_constexpr(K // 8)]
            h_planes = [h_vec_x, h_vec_y, h_vec_z, h_vec_w]
            if state_idx_nonneg:
                if iv_valid:
                    for kg_local in range_constexpr(K // 8):
                        kg = arith.as_value(
                            flir.arith.AddIOp(
                                arith.as_value(kg_start), arith.as_value(arith.index(kg_local))
                            ).result
                        )
                        for lane_comp in range_constexpr(4):
                            h_planes[lane_comp][kg_local] = _load_state_lane(
                                initial_state_source,
                                state_idx,
                                i_hv,
                                kg,
                                iv_safe,
                                lane_comp,
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

                # Phase 0: load current V lane and compute scalar gates (g, beta).
                k_inv_norm = c_one
                v_col = arith.as_value(
                    flir.arith.AddIOp(arith.as_value(c_v_dim_off), arith.as_value(iv_safe)).result
                )
                v_val = _load_mixed_qkv_fp32(mixed_qkv, i_n, v_col, t_idx)
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

                b_val = memref.load(b_gate, [arith.as_value(row_ab), arith.as_value(i_hv)])
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

                # Phase 1: cooperative K load -> reduce (h·k) and optional K norm.
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
                    k0 = _load_mixed_qkv_fp32(mixed_qkv, i_n, k_col0, t_idx)
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
                    k1 = _load_mixed_qkv_fp32(mixed_qkv, i_n, k_col1, t_idx)
                    smem_kq.store(k1, [arith.as_value(k_pair_base1)])
                gpu.barrier()

                acc_hk = arith.constant(0.0, type=comp)
                k_sq_local = arith.constant(0.0, type=comp)
                k_vecs = [[] for _ in range_constexpr(4)]
                for kg_local in range_constexpr(K // 8):
                    k_idx0 = arith.as_value(
                        flir.arith.AddIOp(
                            arith.as_value(k_base),
                            arith.as_value(arith.index(kg_local * 4)),
                        ).result
                    )
                    k0, k1, k2, k3 = _smem_load4(smem_kq, k_idx0)
                    k_vals = [k0, k1, k2, k3]
                    for lane_comp in range_constexpr(4):
                        k_vecs[lane_comp].append(k_vals[lane_comp])
                    sum0123 = _dot4(
                        h_planes[0][kg_local],
                        h_planes[1][kg_local],
                        h_planes[2][kg_local],
                        h_planes[3][kg_local],
                        k0,
                        k1,
                        k2,
                        k3,
                        fm_fast,
                    )
                    acc_hk = flir.arith.AddFOp(
                        arith.as_value(acc_hk), arith.as_value(sum0123), fastmath=fm_fast
                    ).result
                    if use_qk_l2norm_in_kernel:
                        sq0123 = _sumsq4(k0, k1, k2, k3, fm_fast)
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

                # Phase 2: recurrent state update h = h*exp(g) + k*v.
                for kg_local in range_constexpr(K // 8):
                    k_vals = [
                        k_vecs[0][kg_local],
                        k_vecs[1][kg_local],
                        k_vecs[2][kg_local],
                        k_vecs[3][kg_local],
                    ]
                    if use_qk_l2norm_in_kernel:
                        for lane_comp in range_constexpr(4):
                            k_vals[lane_comp] = flir.arith.MulFOp(
                                arith.as_value(k_vals[lane_comp]),
                                arith.as_value(k_inv_norm),
                                fastmath=fm_fast,
                            ).result
                    for lane_comp in range_constexpr(4):
                        kv = flir.arith.MulFOp(
                            arith.as_value(k_vals[lane_comp]),
                            arith.as_value(v_val),
                            fastmath=fm_fast,
                        ).result
                        h_decay = flir.arith.MulFOp(
                            arith.as_value(h_planes[lane_comp][kg_local]),
                            arith.as_value(exp_g),
                            fastmath=fm_fast,
                        ).result
                        h_planes[lane_comp][kg_local] = flir.arith.AddFOp(
                            arith.as_value(h_decay),
                            arith.as_value(kv),
                            fastmath=fm_fast,
                        ).result

                # Phase 3: cooperative Q load -> reduce (h·q) and optional Q norm.
                if k_pair_valid0:
                    q_col0 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_q_dim_off), arith.as_value(k_pair_base)).result
                    )
                    q0 = _load_mixed_qkv_fp32(mixed_qkv, i_n, q_col0, t_idx)
                    smem_kq.store(q0, [arith.as_value(k_pair_base)])
                if k_pair_valid1:
                    q_col1 = arith.as_value(
                        flir.arith.AddIOp(arith.as_value(c_q_dim_off), arith.as_value(k_pair_base1)).result
                    )
                    q1 = _load_mixed_qkv_fp32(mixed_qkv, i_n, q_col1, t_idx)
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
                    q0, q1, q2, q3 = _smem_load4(smem_kq, q_idx0)
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
                    sum0123 = _dot4(
                        h_planes[0][kg_local],
                        h_planes[1][kg_local],
                        h_planes[2][kg_local],
                        h_planes[3][kg_local],
                        q0s,
                        q1s,
                        q2s,
                        q3s,
                        fm_fast,
                    )
                    o_acc = flir.arith.AddFOp(
                        arith.as_value(o_acc), arith.as_value(sum0123), fastmath=fm_fast
                    ).result
                    if use_qk_l2norm_in_kernel:
                        sq0123 = _sumsq4(q0, q1, q2, q3, fm_fast)
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
                # Phase 4: write output for primary lane only.
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
                        for lane_comp in range_constexpr(4):
                            _store_state_lane(
                                initial_state_source,
                                h_planes[lane_comp][kg_local],
                                state_idx,
                                i_hv,
                                kg,
                                iv_safe,
                                lane_comp,
                            )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            mixed_qkv: lambda: T.memref(B, mixed_dim, T_seq, elem_type),
            A_log: lambda: T.memref(HV, T.f32()),
            a: lambda: T.memref(BT, HV, elem_type),
            dt_bias: lambda: T.memref(HV, elem_type),
            b_gate: lambda: T.memref(BT, HV, elem_type),
            initial_state_source: lambda: T.memref(N_STATE, HV, K // 4, V, 4, state_elem_type),
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
                    mixed_qkv,
                    A_log,
                    a,
                    dt_bias,
                    b_gate,
                    initial_state_source,
                    initial_state_indices,
                    o,
                ],
            )

    return _FusedSplitGdrUpdateKSplit2()
