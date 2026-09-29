# SPDX-License-Identifier: Apache-2.0
"""Locate the first numerical difference in the native ATOM A8W4 MoE path.

This diagnostic records errors without changing acceptance tolerances or
replacing the standard ATOM path. Run the normal compare tool for acceptance.
"""

import argparse
import json
import os
from pathlib import Path

from ..config import validate_shape
from .compare import error_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--tp", type=int, choices=(1, 2, 4, 8), required=True)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--fused-epilogue-reference",
        action="store_true",
        help="also diagnose the native fused FP32-activation-to-FP8 boundary",
    )
    a = p.parse_args()
    for seq in a.seq_lens:
        validate_shape(1, seq)
    if a.repeats < 1 or not 3 <= a.layer <= 60:
        p.error("repeats must be positive and the bias-routing layer must be in 3..60")
    os.environ["ATOM_DSV4_MOE_MONOKERNEL"] = "1"
    os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")
    os.environ.setdefault("ATOM_MOE_GU_ITLV", "1")
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from aiter import silu_and_mul
    from atom.model_ops.linear import per_1x128_e8m0_quant

    from kernels.monokernel.formats import dequantize_mxfp4

    from .. import reference as ref
    from ..atom_module import AtomMoeModule
    from ..checkpoint import load_moe

    rank, world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world != a.tp:
        p.error("torchrun WORLD_SIZE must match --tp")
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
    cfg, weights, shared, table = load_moe(a.checkpoint, a.layer, world, rank, "cuda")
    fixture = AtomMoeModule(a.checkpoint, a.layer, weights, shared, table, world, rank)
    module, parts = fixture.module, {}
    for name, child in (
        ("logits", module.gate),
        ("shared_gu", module.shared_experts.gate_up_proj),
        ("shared", module.shared_experts),
    ):
        child.register_forward_hook(lambda mod, inputs, out, name=name: parts.update({name: out.clone()}))
    combine = module.combine_outputs

    def capture_combine(routed, shared, **kwargs):
        parts["routed"] = routed.clone()
        return combine(routed, shared, **kwargs)

    module.combine_outputs = capture_combine
    select_experts = module.experts.quant_method.select_experts_with_record

    def capture_routing(**kwargs):
        actual_probs, actual_ids = select_experts(**kwargs)
        parts["probs"], parts["ids"] = actual_probs.clone(), actual_ids.clone()
        return actual_probs, actual_ids

    # Record the route actually consumed by the native MoE, including its
    # dispatch policy. Re-selecting from the logits would be a second path.
    module.experts.quant_method.select_experts_with_record = capture_routing

    def dequant_weight(w, s):
        scales = torch.pow(2.0, s.view(torch.uint8).float() - 127)
        return w.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)

    su, sd = dequant_weight(shared[0], shared[1]), dequant_weight(shared[2], shared[3])

    def native_quant(x):
        q, s = per_1x128_e8m0_quant(x, torch.float8_e4m3fn, True)
        # The HIP API allocates [M, K/128] but writes group-major bytes when
        # transpose_scale=True; the GEMM consumes this physical ABI directly.
        scales = s.view(torch.uint8).reshape(x.shape[-1] // 128, x.shape[0]).t()
        return q.float() * torch.pow(2.0, scales.float() - 127).repeat_interleave(128, -1)

    def activation(gu):
        gate, up = gu.float().chunk(2, -1)
        return (F.silu(gate.clamp(max=cfg.swiglu_limit)) * up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)).bfloat16()

    def routed_fused_reference(x, actual_ids, actual_probs):
        # Native FlyDSL *_fp8 stage1 quantizes FP32 activation directly. Its
        # scale exponent rounds amax's mantissa before subtracting eight.
        # Keep this separate from the accepted BF16-boundary oracle so the
        # experiment cannot silently change the normal comparison contract.
        xq = ref.quant_dequant_a8(x)
        results = [torch.zeros_like(x, dtype=torch.float32) for _ in range(2)]
        for token in range(x.shape[0]):
            for slot in range(cfg.top_k):
                expert = int(actual_ids[token, slot])
                wgu = dequantize_mxfp4(weights[2][expert], weights[3][expert])
                wd = dequantize_mxfp4(weights[4][expert], weights[5][expert])
                gate, up = F.linear(xq[token], wgu).chunk(2, -1)
                mid = F.silu(gate.clamp(max=cfg.swiglu_limit)) * up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
                blocks = mid.reshape(-1, 32)
                amax_bits = blocks.abs().amax(-1).contiguous().view(torch.int32)
                exponent = (((amax_bits + 0x400000) & -0x800000) >> 23) - 8
                exponent = exponent.clamp_min(0)
                scale = (exponent << 23).view(torch.float32)
                inverse = ((254 - exponent) << 23).view(torch.float32)
                fused_q = (blocks * inverse[:, None]).to(torch.float8_e4m3fn).float() * scale[:, None]
                for output, quantized in zip(results, (ref.quant_dequant_a8(mid), fused_q.reshape_as(mid))):
                    output[token] += F.linear(quantized, wd) * actual_probs[token, slot]
        return [output.bfloat16() for output in results]

    def tp_sum(x):
        if world == 1:
            return x
        peers = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(peers, x)
        return sum(v.float() for v in peers).bfloat16()

    cases = []
    for seq in a.seq_lens:
        for sample, input_scale in enumerate((0.2, 2.0)):
            torch.manual_seed(a.seed + seq * 1000 + sample)
            x = torch.randn(seq, cfg.hidden, device="cuda", dtype=torch.bfloat16) * input_scale
            expected, ids, probs, rr, rs = ref.moe_reference(x, *weights, cfg, shared=shared, return_parts=True)
            expected = tp_sum(expected)
            expected_logits = F.linear(x.float(), weights[0].float()).bfloat16()
            quantized_x = native_quant(x)
            expected_gu = F.linear(ref.quant_dequant_a8(x, 128), su).bfloat16()
            gu_native_f32 = F.linear(quantized_x, su)
            gu_with_native_quant = gu_native_f32.bfloat16()
            gu_truncated = (gu_native_f32.contiguous().view(torch.int32) & -65536).view(torch.float32).bfloat16()
            previous = None
            for repeat in range(a.repeats):
                parts.clear()
                out = fixture(x).clone()
                logits, gu = parts["logits"], parts["shared_gu"]
                actual_probs, actual_ids = parts["probs"], parts["ids"]
                selected_scores = F.softplus(expected_logits.float()).sqrt().gather(1, actual_ids.long())
                probs_same_ids = selected_scores / selected_scores.sum(-1, keepdim=True) * cfg.route_scale
                original_route = ref.route
                try:
                    ref.route = lambda *args, **kwargs: (actual_ids, actual_probs)
                    _, _, _, rr_fixed, _ = ref.moe_reference(x, *weights, cfg, shared=shared, return_parts=True)
                finally:
                    ref.route = original_route
                mid = torch.empty(seq, gu.shape[-1] // 2, dtype=x.dtype, device=x.device)
                silu_and_mul(mid, gu, cfg.swiglu_limit)
                native_mid = native_quant(mid)
                down_same_input = module.shared_experts.w2(mid)
                expected_down_same_input = F.linear(native_mid, sd).bfloat16()
                item = {
                    "seq_len": seq,
                    "input_scale": input_scale,
                    "repeat": repeat,
                    "whole": error_metrics(out, expected),
                    "router": error_metrics(logits, expected_logits),
                    "router_probs_same_ids": error_metrics(actual_probs, probs_same_ids),
                    "same_expert_set": torch.equal(actual_ids.sort(-1).values, ids.sort(-1).values),
                    "actual_ids": actual_ids.tolist(),
                    "reference_ids": ids.tolist(),
                    "routed": error_metrics(parts["routed"], rr),
                    "routed_fixed_routing": error_metrics(parts["routed"], rr_fixed),
                    "shared": error_metrics(parts["shared"], rs),
                    "shared_input_quant": error_metrics(quantized_x, ref.quant_dequant_a8(x, 128)),
                    "shared_gemm1": error_metrics(gu, expected_gu),
                    "shared_gemm1_native_quant": error_metrics(gu, gu_with_native_quant),
                    "shared_gemm1_truncated_reference": error_metrics(gu, gu_truncated),
                    "shared_activation_same_input": error_metrics(mid, activation(gu)),
                    "shared_mid_quant": error_metrics(native_mid, ref.quant_dequant_a8(mid, 128)),
                    "shared_gemm2_same_input": error_metrics(down_same_input, expected_down_same_input),
                    "native_repeat": (
                        None
                        if previous is None
                        else {
                            k: error_metrics(parts[k], previous[k]) for k in ("logits", "shared_gu", "shared", "routed")
                        }
                    ),
                }
                if a.fused_epilogue_reference:
                    fp32_upward, fp32_native = routed_fused_reference(x, actual_ids, actual_probs)
                    item["routed_fp32_activation_upward_scale"] = error_metrics(parts["routed"], fp32_upward)
                    item["routed_native_fused_epilogue"] = error_metrics(parts["routed"], fp32_native)
                previous = {k: v.clone() for k, v in parts.items()}
                cases.append(item)
                print(f"rank={rank}: {json.dumps(item)}", flush=True)
    result = {
        "model": "DSV4-Pro",
        "layer": a.layer,
        "tp": world,
        "rank": rank,
        "seed": a.seed,
        "repeats": a.repeats,
        "seq_lens": a.seq_lens,
        "input_scales": [0.2, 2.0],
        "selected_activation_dtype": fixture.selected_dtype,
        "routing_observation": "native select_experts_with_record return values",
        "bf16_gemm_config_override": os.environ.get("AITER_CONFIG_GEMM_BF16"),
        "shared_gemm_config_override": os.environ.get("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"),
        "fused_epilogue_reference": a.fused_epilogue_reference,
        "diagnostic_only": True,
        "cases": cases,
    }
    result["routed_mid_contract"] = "FP32 SwiGLU directly to per32 FP8, native fused exponent"
    path = a.output.with_name(a.output.stem + f".rank{rank}" + a.output.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    module.experts.quant_method.select_experts_with_record = select_experts
    module.combine_outputs = combine
    fixture.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
