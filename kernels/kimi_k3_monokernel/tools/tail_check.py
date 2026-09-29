# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Independent checks of the fused tail's communication and residual update."""

import torch
import torch.distributed as dist


def check_tail_reduction(layer, output: torch.Tensor, npes: int) -> dict[str, bool]:
    """Check the reduction from the actual BF16 projection outputs.

    The tail sums shared projections in FP32, adds each output owner's latent
    projection, and then rounds to BF16. Keep that rounding boundary in this
    reference so even one corrupted mailbox value fails the check.
    """

    parts = [None] * npes
    dist.all_gather_object(parts, (layer.shared_partial.cpu(), layer.tail.cpu()))
    shared_sum = parts[0][0].float()
    for shared, _ in parts[1:]:
        shared_sum.add_(shared.float())
    latent = torch.cat([tail for _, tail in parts], dim=-1)
    delta = (shared_sum + latent.float()).to(torch.bfloat16)
    expected = (layer.updated_prefix.cpu().float() + delta.float()).to(torch.bfloat16)
    return {
        "tail_reduce_equal": torch.equal(layer.moe_delta.cpu(), delta),
        "tail_output_equal": torch.equal(output.cpu(), expected),
    }


def check_routed_pipeline(layer, npes: int) -> dict:
    """Check the new expert path from actual BF16 input and selected routes."""
    from kernels.monokernel.reference import situ
    from kernels.monokernel.formats import dequantize_mxfp4

    pipeline = layer.routed_pipeline
    actual_mid = pipeline.mid[..., 0].contiguous().view(torch.bfloat16).reshape(layer.S, 16, 384)
    expected_mid = torch.empty_like(actual_mid)
    expected_out = torch.zeros_like(layer.routed_partial, dtype=torch.float32)
    ids = layer.topk_ids.cpu().tolist()
    for sample in range(layer.S):
        for slot, expert in enumerate(ids[sample]):
            gu = dequantize_mxfp4(layer.t["w_ug"][expert], layer.t["s_ug"][expert]) @ layer.latent[sample].float()
            mid = situ(gu, beta=layer.config.situ_beta, linear_beta=layer.config.situ_linear_beta).to(torch.bfloat16)
            expected_mid[sample, slot] = mid
            down = dequantize_mxfp4(layer.t["w_dn"][expert], layer.t["s_dn"][expert]) @ mid.float()
            expected_out[sample].add_(down * layer.topk_weights[sample, slot])
    parts = [None] * npes
    dist.all_gather_object(parts, layer.routed_partial.cpu())
    total = torch.zeros_like(parts[0], dtype=torch.float32)
    for part in parts:
        total.add_(part.float())

    def relative_l2(actual, expected):
        return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))

    return {
        "pipeline_mid_rel_l2": relative_l2(actual_mid, expected_mid),
        "pipeline_output_rel_l2": relative_l2(layer.routed_partial, expected_out),
        "pipeline_tp_equal": torch.equal(layer.routed_reduced.cpu(), total.to(torch.bfloat16)),
    }
