"""FlyDSL fused recurrent gated delta rule forward kernel.

Phase 2: single-step update logic implemented in the timestep loop:
- h *= exp(g)
- v -= sum(h * k, dim=0)
- v *= beta
- h += k ⊗ v
- o = sum(h * q * scale, dim=0)

Scope:
- supports scalar beta and headwise beta-vector branches
- supports h0 (initial state) and ht (final state) IO
- dtype: f32 main path
"""

from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
import _mlir.extras.types as T


KERNEL_NAME = "fused_recurrent_gated_delta_rule_fwd_kernel"


def build_fused_recurrent_gated_delta_rule_fwd_module(
    B,
    T_seq,
    H,
    HV,
    K,
    V,
    dtype_str="f32",
    is_beta_headwise=False,
):
    """Build the fused recurrent gated delta rule forward kernel module.

    Args:
        B, T_seq, H, HV, K, V: compile-time dimensions
        dtype_str: "f32" or "f16" (MVP: f32 only)
        is_beta_headwise: use beta_headwise[B*T*HV, V] when True,
            otherwise use beta_scalar[B*T, HV]
    """
    gpu_arch = get_hip_arch()
    compute_type = T.f32()
    elem_type = T.f32() if dtype_str == "f32" else T.f16()
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
    BHV = B * HV

    class _FusedRecurrentGatedDeltaRuleFwd(flir.MlirModule):
        GPU_MODULE_NAME = f"fused_recurrent_gated_delta_rule_fwd_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            _state["smem_h"] = allocator.allocate_array(compute_type, h_smem_size)
            allocator.finalize()

        @flir.kernel
        def fused_recurrent_gated_delta_rule_fwd_kernel(
            self: flir.T.i64,
            q: lambda: T.memref(BTH, K, elem_type),
            k: lambda: T.memref(BTH, K, elem_type),
            v: lambda: T.memref(BTHV, V, elem_type),
            g: lambda: T.memref(BT, HV, elem_type),
            beta_scalar: lambda: T.memref(BT, HV, elem_type),
            beta_headwise: lambda: T.memref(BTHV, V, elem_type),
            h0: lambda: T.memref(BHV, K, V, elem_type),
            ht: lambda: T.memref(BHV, K, V, elem_type),
            o: lambda: T.memref(BTHV, V, elem_type),
        ):
            # Block index -> (n, hv)
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
            c_scale = arith.constant(scale, type=comp)
            fm_fast = flir.arith.FastMathFlags.fast

            c_TH = arith.as_value(arith.index(T_seq * H))
            c_THV = arith.as_value(arith.index(T_seq * HV))
            c_T = arith.as_value(arith.index(T_seq))
            c_H = arith.as_value(arith.index(H))

            # row_h = n * HV + hv for h0/ht
            m0 = flir.arith.MulIOp(arith.as_value(i_n), c_hv).result
            row_h = arith.as_value(flir.arith.AddIOp(arith.as_value(m0), arith.as_value(i_hv)).result)

            # Initialize recurrent state h from h0.
            for ik in range_constexpr(K):
                for iv in range_constexpr(V):
                    h_init = memref.load(h0, [arith.as_value(row_h), arith.index(ik), arith.index(iv)])
                    if dtype_str != "f32":
                        h_init = flir.arith.extf(comp, arith.as_value(h_init))
                    crd = flir.make_coord(arith.index(ik), arith.index(iv))
                    idx = flir.crd2idx(crd, layout_h)
                    memref.store(arith.as_value(h_init), s_h, [arith.as_value(idx)])

            gpu.barrier()

            # Time loop: apply update-fwd logic each step.
            for t in range_constexpr(T_seq):
                t_idx = arith.as_value(arith.index(t))

                # row_qt = n * (T*H) + t * H + h   for q, k
                m1 = flir.arith.MulIOp(arith.as_value(i_n), c_TH).result
                m2 = flir.arith.MulIOp(t_idx, c_H).result
                a1 = flir.arith.AddIOp(arith.as_value(m1), arith.as_value(m2)).result
                row_qt = arith.as_value(flir.arith.AddIOp(arith.as_value(a1), arith.as_value(i_h)).result)

                # row_vt = n * (T*HV) + t * HV + hv   for v, o
                m3 = flir.arith.MulIOp(arith.as_value(i_n), c_THV).result
                m4 = flir.arith.MulIOp(t_idx, c_hv).result
                a2 = flir.arith.AddIOp(arith.as_value(m3), arith.as_value(m4)).result
                row_vt = arith.as_value(flir.arith.AddIOp(arith.as_value(a2), arith.as_value(i_hv)).result)

                # row_gb = n * T + t   for g, beta
                m5 = flir.arith.MulIOp(arith.as_value(i_n), c_T).result
                row_gb = arith.as_value(flir.arith.AddIOp(arith.as_value(m5), t_idx).result)

                # Load g and beta (already precomputed).
                g_val = memref.load(g, [arith.as_value(row_gb), arith.as_value(i_hv)])
                if dtype_str != "f32":
                    g_val = flir.arith.extf(comp, arith.as_value(g_val))

                beta_val_scalar = None
                if not is_beta_headwise:
                    beta_val_scalar = memref.load(beta_scalar, [arith.as_value(row_gb), arith.as_value(i_hv)])
                    if dtype_str != "f32":
                        beta_val_scalar = flir.arith.extf(comp, arith.as_value(beta_val_scalar))

                # exp(g) via exp2(g * log2(e))
                g_log2e = flir.arith.MulFOp(arith.as_value(g_val), arith.as_value(c_log2e), fastmath=fm_fast).result
                exp_g = flir.math.exp2(arith.as_value(g_log2e), fastmath=fm_fast)

                # h *= exp(g)
                for ik in range_constexpr(K):
                    for iv in range_constexpr(V):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        old_h = memref.load(s_h, [arith.as_value(idx)])
                        h_scaled = flir.arith.MulFOp(arith.as_value(old_h), arith.as_value(exp_g), fastmath=fm_fast).result
                        memref.store(h_scaled, s_h, [arith.as_value(idx)])

                # Load q, k vectors.
                q_vals = []
                k_vals = []
                for ik in range_constexpr(K):
                    qv = memref.load(q, [arith.as_value(row_qt), arith.index(ik)])
                    kv = memref.load(k, [arith.as_value(row_qt), arith.index(ik)])
                    if dtype_str != "f32":
                        qv = flir.arith.extf(comp, arith.as_value(qv))
                        kv = flir.arith.extf(comp, arith.as_value(kv))
                    q_vals.append(qv)
                    k_vals.append(kv)

                # Load v vector.
                v_vals = []
                for iv in range_constexpr(V):
                    vv = memref.load(v, [arith.as_value(row_vt), arith.index(iv)])
                    if dtype_str != "f32":
                        vv = flir.arith.extf(comp, arith.as_value(vv))
                    v_vals.append(vv)

                # v -= sum(h * k, dim=0)
                for iv in range_constexpr(V):
                    s = arith.constant(0.0, type=comp)
                    for ik in range_constexpr(K):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        h_ikv = memref.load(s_h, [arith.as_value(idx)])
                        prod = flir.arith.MulFOp(arith.as_value(h_ikv), arith.as_value(k_vals[ik]), fastmath=fm_fast).result
                        s = flir.arith.AddFOp(arith.as_value(s), arith.as_value(prod), fastmath=fm_fast).result
                    v_vals[iv] = flir.arith.SubFOp(arith.as_value(v_vals[iv]), arith.as_value(s)).result

                # v *= beta
                for iv in range_constexpr(V):
                    if is_beta_headwise:
                        beta_v = memref.load(beta_headwise, [arith.as_value(row_vt), arith.index(iv)])
                        if dtype_str != "f32":
                            beta_v = flir.arith.extf(comp, arith.as_value(beta_v))
                        v_vals[iv] = flir.arith.MulFOp(
                            arith.as_value(v_vals[iv]), arith.as_value(beta_v), fastmath=fm_fast
                        ).result
                    else:
                        v_vals[iv] = flir.arith.MulFOp(
                            arith.as_value(v_vals[iv]), arith.as_value(beta_val_scalar), fastmath=fm_fast
                        ).result

                # h += k ⊗ v
                for ik in range_constexpr(K):
                    for iv in range_constexpr(V):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        old_h = memref.load(s_h, [arith.as_value(idx)])
                        prod = flir.arith.MulFOp(arith.as_value(k_vals[ik]), arith.as_value(v_vals[iv]), fastmath=fm_fast).result
                        upd = flir.arith.AddFOp(arith.as_value(old_h), arith.as_value(prod), fastmath=fm_fast).result
                        memref.store(upd, s_h, [arith.as_value(idx)])

                # o = sum(h * q * scale, dim=0)
                for iv in range_constexpr(V):
                    s = arith.constant(0.0, type=comp)
                    for ik in range_constexpr(K):
                        crd = flir.make_coord(arith.index(ik), arith.index(iv))
                        idx = flir.crd2idx(crd, layout_h)
                        h_ikv = memref.load(s_h, [arith.as_value(idx)])
                        q_scaled = flir.arith.MulFOp(
                            arith.as_value(q_vals[ik]), arith.as_value(c_scale), fastmath=fm_fast
                        ).result
                        prod = flir.arith.MulFOp(arith.as_value(h_ikv), arith.as_value(q_scaled), fastmath=fm_fast).result
                        s = flir.arith.AddFOp(arith.as_value(s), arith.as_value(prod), fastmath=fm_fast).result
                    out_val = s if dtype_str == "f32" else flir.arith.truncf(elem_type, arith.as_value(s))
                    memref.store(arith.as_value(out_val), o, [arith.as_value(row_vt), arith.index(iv)])

            # Write final recurrent state to ht.
            for ik in range_constexpr(K):
                for iv in range_constexpr(V):
                    crd = flir.make_coord(arith.index(ik), arith.index(iv))
                    idx = flir.crd2idx(crd, layout_h)
                    h_fin = memref.load(s_h, [arith.as_value(idx)])
                    out_h = h_fin if dtype_str == "f32" else flir.arith.truncf(elem_type, arith.as_value(h_fin))
                    memref.store(arith.as_value(out_h), ht, [arith.as_value(row_h), arith.index(ik), arith.index(iv)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            q: lambda: T.memref(BTH, K, elem_type),
            k: lambda: T.memref(BTH, K, elem_type),
            v: lambda: T.memref(BTHV, V, elem_type),
            g: lambda: T.memref(BT, HV, elem_type),
            beta_scalar: lambda: T.memref(BT, HV, elem_type),
            beta_headwise: lambda: T.memref(BTHV, V, elem_type),
            h0: lambda: T.memref(BHV, K, V, elem_type),
            ht: lambda: T.memref(BHV, K, V, elem_type),
            o: lambda: T.memref(BTHV, V, elem_type),
        ):
            c1 = flir.const_index(1)
            gx = flir.const_index(B * HV)
            bx = flir.const_index(1)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_recurrent_gated_delta_rule_fwd_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[q, k, v, g, beta_scalar, beta_headwise, h0, ht, o],
            )

    return _FusedRecurrentGatedDeltaRuleFwd()
