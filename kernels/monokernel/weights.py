# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared host-side weight container for model-specific MonoKernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.monokernel.config import (
    GLM5_CONFIG,
    LayerConfig,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)


class MonoUnsupported(ValueError):
    """Invalid MonoKernel weight storage."""


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _unshuffle_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """Invert AITER's MI355 16x16 weight preshuffle."""

    if not getattr(weight, "is_shuffled", False):
        return weight.contiguous()
    source_dtype = weight.dtype
    raw = weight if weight.dtype == torch.uint8 else weight.view(torch.uint8)
    *lead, rows, packed_k = raw.shape
    lane_k = 16 // raw.element_size()
    block_k = 32
    shuffled = raw.reshape(
        *lead,
        rows // 16,
        packed_k // block_k,
        block_k // lane_k,
        16,
        lane_k,
    )
    nlead = len(lead)
    order = list(range(nlead)) + [nlead, nlead + 3, nlead + 1, nlead + 2, nlead + 4]
    return shuffled.permute(*order).contiguous().reshape(*lead, rows, packed_k).view(source_dtype)


def _unshuffle_linear_scale(
    scale: torch.Tensor,
    *,
    experts: int,
    rows: int,
    groups: int,
) -> torch.Tensor:
    """Invert the non-GUGU E8M0 scale layout used by FlyDSL linears."""

    flat_rows = experts * rows
    padded_rows = (flat_rows + 255) // 256 * 256
    padded_groups = (groups + 7) // 8 * 8
    _need(
        scale.numel() == padded_rows * padded_groups,
        f"linear scale bytes {scale.numel()} != {padded_rows * padded_groups}",
    )
    packed = scale.reshape(
        padded_rows // 32,
        padded_groups // 8,
        4,
        16,
        2,
        2,
    )
    native = packed.permute(0, 5, 3, 1, 4, 2).contiguous()
    return native.reshape(padded_rows, padded_groups)[:flat_rows, :groups].reshape(experts, rows, groups)


@dataclass
class LayerWeights:
    """One tensor-parallel rank's weights and model geometry."""

    heads: int
    t: dict[str, torch.Tensor]
    config: LayerConfig = GLM5_CONFIG
    rank: int = 0
    npes: int = 1
    mxfp4_weight_layout: Mxfp4WeightLayout = Mxfp4WeightLayout.NATIVE
    mxfp4_scale_layout: Mxfp4ScaleLayout = Mxfp4ScaleLayout.NATIVE
    physical_experts: int | None = None


def atom_mxfp4_storage_view(
    tensor: torch.Tensor,
    *,
    name: str,
    logical_rows: int,
    logical_k: int,
    scale: bool,
) -> torch.Tensor:
    if logical_rows <= 0 or logical_k <= 0 or logical_k % 32:
        raise ValueError(f"{name} invalid logical shape [{logical_rows}, {logical_k}]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} ATOM storage must be contiguous")
    if tensor.element_size() != 1:
        raise ValueError(f"{name} ATOM storage must use one-byte values")
    if scale:
        groups = logical_k // 32
        padded_rows = (logical_rows + 255) // 256 * 256
        padded_groups = (groups + 7) // 8 * 8
        expected_bytes = padded_rows * padded_groups
    else:
        if not getattr(tensor, "is_shuffled", False):
            raise ValueError(f"{name} ATOM weight must carry the preshuffled marker")
        expected_bytes = logical_rows * logical_k // 2
    if tensor.numel() != expected_bytes:
        raise ValueError(f"{name} has {tensor.numel()} bytes, expected {expected_bytes}")
    return tensor.view(torch.uint8).view(-1)


def prepare_mxfp4_expert_storage(weights: LayerWeights, *, canonical: bool = True) -> tuple[torch.Tensor, ...]:
    config = weights.config
    tensors = weights.t
    experts = config.n_experts if weights.physical_experts is None else weights.physical_experts
    expert_hidden = config.hidden if config.routed_hidden is None else config.routed_hidden
    ug_rows = 2 * config.inter
    dn_rows = expert_hidden

    if not canonical:
        if (
            weights.mxfp4_weight_layout is not Mxfp4WeightLayout.ATOM
            or weights.mxfp4_scale_layout is not Mxfp4ScaleLayout.ATOM
        ):
            raise ValueError("zero-copy MXFP4 storage requires ATOM values and scales")
        return (
            atom_mxfp4_storage_view(
                tensors["w_ug"],
                name="w_ug",
                logical_rows=experts * ug_rows,
                logical_k=expert_hidden,
                scale=False,
            ),
            atom_mxfp4_storage_view(
                tensors["s_ug"],
                name="s_ug",
                logical_rows=experts * ug_rows,
                logical_k=expert_hidden,
                scale=True,
            ),
            atom_mxfp4_storage_view(
                tensors["w_dn"],
                name="w_dn",
                logical_rows=experts * dn_rows,
                logical_k=config.inter,
                scale=False,
            ),
            atom_mxfp4_storage_view(
                tensors["s_dn"],
                name="s_dn",
                logical_rows=experts * dn_rows,
                logical_k=config.inter,
                scale=True,
            ),
        )

    from kernels.monokernel.packing import pack_mxfp4

    def values(name: str, rows: int, k: int) -> torch.Tensor:
        tensor = tensors[name]
        if weights.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM:
            packed = atom_mxfp4_storage_view(tensor, name=name, logical_rows=experts * rows, logical_k=k, scale=False)
            shuffled = packed.view(experts, rows, k // 2)
            shuffled.is_shuffled = True
            tensor = _unshuffle_linear_weight(shuffled)
        elif weights.mxfp4_weight_layout is not Mxfp4WeightLayout.NATIVE:
            raise ValueError(f"unsupported MXFP4 weight layout {weights.mxfp4_weight_layout!r}")
        return pack_mxfp4(tensor)

    def scales(name: str, rows: int, k: int) -> torch.Tensor:
        tensor = tensors[name]
        groups = k // 32
        if weights.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM:
            packed = atom_mxfp4_storage_view(tensor, name=name, logical_rows=experts * rows, logical_k=k, scale=True)
            tensor = _unshuffle_linear_scale(packed, experts=experts, rows=rows, groups=groups)
        elif weights.mxfp4_scale_layout is not Mxfp4ScaleLayout.NATIVE:
            raise ValueError(f"unsupported MXFP4 scale layout {weights.mxfp4_scale_layout!r}")
        return tensor.view(torch.uint8).contiguous().view(-1)

    return (
        values("w_ug", ug_rows, expert_hidden),
        scales("s_ug", ug_rows, expert_hidden),
        values("w_dn", dn_rows, config.inter),
        scales("s_dn", dn_rows, config.inter),
    )


__all__ = [
    "LayerWeights",
    "atom_mxfp4_storage_view",
    "prepare_mxfp4_expert_storage",
]
