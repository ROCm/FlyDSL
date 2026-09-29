# SPDX-License-Identifier: Apache-2.0
"""Unfused ATOM/AITER MoE operations with the native Pro A8W4 recipe.

This exercises the kernels and layout preparation used by ATOM's MoE and
Expert modules. It is a component baseline, not a full ATOM server benchmark.
"""

import os


class AtomMoeBaseline:
    def __init__(self, weights, shared, config, tp=1, group=None):
        import torch
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import resolve_activation_dtype
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight, shuffle_weight

        if os.environ.get("ATOM_MOE_GU_ITLV") != "1":
            raise ValueError("ATOM baseline requires ATOM_MOE_GU_ITLV=1")
        for n in (1, 2, 3, 4):
            dtype = resolve_activation_dtype(
                QuantType.per_1x32, dtypes.fp4x2, activation=ActivationType.Silu, gate_mode=GateMode.INTERLEAVE, M=n
            )
            if dtype != dtypes.fp8:
                raise ValueError(f"ATOM dispatch chose {dtype} at M={n}; set AITER_BF16_FP8_MOE_BOUND=0")
        if shared is None:
            raise ValueError("ATOM native Pro baseline requires FP8 shared weights")
        self.config, self.tp, self.group = config, tp, group
        self.router, self.bias, up, us, down, ds = weights
        e = config.experts
        self.up = moe_shuffle_weight(up.view(dtypes.fp4x2), experts_cnt=e, is_guinterleave=True, gate_up=True)
        self.down = moe_shuffle_weight(down.view(dtypes.fp4x2), experts_cnt=e, is_guinterleave=True, gate_up=False)
        self.up.is_shuffled = self.down.is_shuffled = True
        self.us = moe_shuffle_scale(us.reshape(-1, us.shape[-1]), e, is_guinterleave=True, gate_up=True)
        self.ds = moe_shuffle_scale(ds.reshape(-1, ds.shape[-1]), e, is_guinterleave=True, gate_up=False)
        su, sus, sd, sds = shared
        self.su, self.sd = shuffle_weight(su, (16, 16)), shuffle_weight(sd, (16, 16))
        self.sus = sus.view(torch.float8_e8m0fnu)
        self.sds = sds.view(torch.float8_e8m0fnu)
        self.selected_dtype = str(dtype)

    def __call__(self, x, hash_ids=None, *, return_parts=False):
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        from aiter import ActivationType, QuantType, silu_and_mul
        from aiter.fused_moe import fused_moe
        from aiter.tuned_gemm import tgemm
        from atom.model_ops.linear import gemm_a8w8_blockscale_preshuffle_impl, per_1x128_e8m0_quant
        from atom.model_ops.moe import FusedMoE

        def shared_linear(x, w, scale):
            q, s = per_1x128_e8m0_quant(x, torch.float8_e4m3fn, True)
            return gemm_a8w8_blockscale_preshuffle_impl(q, w, s, scale)

        gu = shared_linear(x, self.su, self.sus)
        mid = torch.empty(x.shape[0], gu.shape[1] // 2, dtype=x.dtype, device=x.device)
        silu_and_mul(mid, gu, self.config.swiglu_limit)
        shared_out = shared_linear(mid, self.sd, self.sds)
        logits = tgemm.mm(x, self.router, None, otype=torch.bfloat16)
        if hash_ids is None:
            probs, ids = FusedMoE.select_experts(
                x,
                logits,
                self.config.top_k,
                False,
                True,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=self.bias,
                routed_scaling_factor=self.config.route_scale,
            )
        else:
            ids = hash_ids
            selected = F.softplus(logits.float()).sqrt().gather(1, ids.long())
            probs = selected / selected.sum(-1, keepdim=True) * self.config.route_scale
        routed = fused_moe(
            x,
            self.up,
            self.down,
            probs,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=self.us,
            w2_scale=self.ds,
            gate_mode="interleave",
            swiglu_limit=self.config.swiglu_limit,
        )
        result = routed + shared_out
        if self.tp > 1:
            dist.all_reduce(result, group=self.group)
        if return_parts:
            return result, routed, shared_out, logits, ids, probs
        return result
