"""FlyDSL Fused Recurrent Gated Delta Rule Forward kernel.

Phase 1: Minimal skeleton - grid/block setup, index computation only.
Target algorithm (SGLang fused_recurrent_gated_delta_rule_fwd):
- g and beta are precomputed inputs
- h *= exp(g)
- v -= sum(h * k, dim=0)   [Delta rule step 1]
- v *= beta
- h += k ⊗ v               [Delta rule step 2]
- o = sum(h * q * scale, dim=0)

Simplified scope: IS_BETA_HEADWISE=False, no h0/ht, dtype=f32.
Layout: q,k [B*T*H,K], v [B*T*HV,V], g,beta [B*T,HV], o [B*T*HV,V]
"""

from _mlir import ir
from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
import _mlir.extras.types as T


KERNEL_NAME = "fused_recurrent_gated_delta_rule_fwd_kernel"


def build_fused_recurrent_gated_delta_rule_fwd_module(B, T_seq, H, HV, K, V, dtype_str="f32"):
    """Build the fused recurrent gated delta rule forward kernel module.

    Phase 1: Minimal skeleton - only index computation, placeholder output.

    Args:
        B, T_seq, H, HV, K, V: compile-time dimensions
        dtype_str: "f32" or "f16" (MVP: f32 only)
    """
    gpu_arch = get_hip_arch()
    elem_type = T.f32() if dtype_str == "f32" else T.f16()
    comp = T.f32()

    BT = B * T_seq
    BTH = B * T_seq * H
    BTHV = B * T_seq * HV

    class _FusedRecurrentGatedDeltaRuleFwd(flir.MlirModule):
        GPU_MODULE_NAME = f"fused_recurrent_gated_delta_rule_fwd_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            pass

        @flir.kernel
        def fused_recurrent_gated_delta_rule_fwd_kernel(
            self: flir.T.i64,
            q: lambda: T.memref(BTH, K, elem_type),
            k: lambda: T.memref(BTH, K, elem_type),
            v: lambda: T.memref(BTHV, V, elem_type),
            g: lambda: T.memref(BT, HV, elem_type),
            beta: lambda: T.memref(BT, HV, elem_type),
            o: lambda: T.memref(BTHV, V, elem_type),
        ):
            # Block index -> (n, hv)
            i_nh = flir.const_index(flir.block_idx("x"))
            c_hv = arith.as_value(arith.index(HV))
            i_n = arith.as_value(flir.arith.DivUIOp(arith.as_value(i_nh), c_hv).result)
            i_hv = arith.as_value(flir.arith.RemUIOp(arith.as_value(i_nh), c_hv).result)
            i_h = i_hv  # H==HV for simplified scope

            c_TH = arith.as_value(arith.index(T_seq * H))
            c_THV = arith.as_value(arith.index(T_seq * HV))
            c_T = arith.as_value(arith.index(T_seq))
            c_H = arith.as_value(arith.index(H))

            # Time loop: index computation only, placeholder write to o
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

                # Placeholder: write 0 to o (verifies indexing + launch)
                zero = arith.constant(0.0, type=comp)
                for iv in range_constexpr(V):
                    val = (
                        zero
                        if dtype_str == "f32"
                        else flir.arith.truncf(elem_type, arith.as_value(zero))
                    )
                    memref.store(arith.as_value(val), o, [arith.as_value(row_vt), arith.index(iv)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            q: lambda: T.memref(BTH, K, elem_type),
            k: lambda: T.memref(BTH, K, elem_type),
            v: lambda: T.memref(BTHV, V, elem_type),
            g: lambda: T.memref(BT, HV, elem_type),
            beta: lambda: T.memref(BT, HV, elem_type),
            o: lambda: T.memref(BTHV, V, elem_type),
        ):
            c1 = flir.const_index(1)
            gx = flir.const_index(B * HV)
            bx = flir.const_index(1)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_recurrent_gated_delta_rule_fwd_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[q, k, v, g, beta, o],
            )

    return _FusedRecurrentGatedDeltaRuleFwd()
