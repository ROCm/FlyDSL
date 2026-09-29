# SPDX-License-Identifier: Apache-2.0
"""DSV4-Pro per-layer forward, following the GLM/K3 host-wrapper API.

The initial backend binds an already loaded ATOM Block. Its native attention
owns the indexer, paged caches and compressor state; its MoE adapter owns the
packed weights, scratch and TP mailboxes. One forward may launch several GPU
kernels. Preparing the object never duplicates a layer's expert weights.
"""

import weakref

from .config import validate_shape


class Dsv4MonoKernel:
    def __init__(self, block, samples, *, layer_idx, rank, npes=8, group=None):
        validate_shape(1, samples)
        adapter = block.ffn._moe_mono
        if adapter is None or samples not in adapter.ops:
            raise ValueError("prepare the ATOM DSV4-Pro A8W4 MoE adapter before the layer wrapper")
        config = adapter.prepared.config
        config.validate_pro()
        config.validate_moe(samples, npes)
        if block.layer_id != layer_idx or block.ffn.tp_size != npes:
            raise ValueError("layer_idx/npes must match the bound ATOM layer")
        if block.ffn.experts.tp_rank != rank:
            raise ValueError("rank must match the bound ATOM TP rank")
        self._block = weakref.ref(block)
        self.S, self.layer_idx, self.rank, self.npes = samples, layer_idx, rank, npes
        self.group, self.config = group, config
        self.moe = adapter.ops[samples]
        self.closed = False
        from atom.model_ops.dsv4_monokernel import bind_stable_compressors

        bind_stable_compressors(block)

    def forward(self, hc_state, positions, *, unfused=False):
        """Run one complete layer and return the delayed mHC state.

        ATOM's forward context supplies attention metadata and input token IDs.
        Caches stay bound to the ATOM attention module, exactly as for native
        execution. ``unfused=True`` splits only the MoE into five GPU stages;
        indexer, attention, cache updates and mHC follow the same native path.
        Use one stream at a time. All guards run before the first cache write.
        """
        import torch
        from atom.model_ops.dsv4_monokernel import shape_supported
        from atom.utils.forward_context import get_forward_context

        if self.closed or self.moe._closed:
            raise RuntimeError("DSV4 layer wrapper is closed")
        block = self._block()
        if block is None:
            raise RuntimeError("the bound ATOM layer has been released")
        fwd = get_forward_context()
        context = fwd.context
        if not shape_supported(context, self.S) or fwd.ubatch_slices is not None:
            raise ValueError("DSV4 layer forward requires one decode request, seq=1..4, without padding or ubatches")
        residual = hc_state.residual
        if residual.shape != (self.S, self.config.hc_mult, self.config.hidden):
            raise ValueError(f"residual must have shape [{self.S}, 4, 7168]")
        if residual.dtype != torch.bfloat16 or residual.device != self.moe.device or not residual.is_contiguous():
            raise ValueError("residual must be BF16 on the prepared layer's device")
        if positions.shape != (self.S,) or positions.dtype != torch.int64:
            raise ValueError(f"positions must be an int64 tensor of shape [{self.S}]")
        if positions.device != residual.device or not positions.is_contiguous():
            raise ValueError("positions must be contiguous on the layer's device")
        if context.is_dummy_run or fwd.attn_metadata is None:
            raise ValueError("DSV4 layer forward requires real ATOM attention metadata and bound caches")
        if block.attn.swa_window is None:
            raise ValueError("bind the ATOM paged cache before calling the layer forward")
        if block.ffn._moe_mono is None or block.ffn._moe_mono.ops.get(self.S) is not self.moe:
            raise RuntimeError("the bound ATOM MoE adapter has been replaced or disabled")
        if hc_state.x_prev is not None:
            for name, shape, dtype in (
                ("x_prev", (self.S, self.config.hidden), torch.bfloat16),
                ("post_mix", (self.S, self.config.hc_mult), torch.float32),
                ("comb_mix", (self.S, self.config.hc_mult, self.config.hc_mult), torch.float32),
            ):
                value = getattr(hc_state, name)
                if (
                    value is None
                    or value.shape != shape
                    or value.dtype != dtype
                    or value.device != residual.device
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"invalid delayed HCState.{name}; expected contiguous {shape} on {residual.device}"
                    )
        token_ids = context.input_ids if block.ffn.is_hash_layer else None
        if block.ffn.is_hash_layer and (
            token_ids is None
            or token_ids.shape != (self.S,)
            or token_ids.dtype != torch.int64
            or token_ids.device != residual.device
            or not token_ids.is_contiguous()
        ):
            raise ValueError("hash layers require contiguous integer input_ids for every query token")

        state = block.fuse_hc(
            hc_state, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base, block.attn_norm.weight, block.norm_eps
        )
        # _sparse_attention performs indexer.topk before the sparse attention.
        # Do not invoke the indexer separately: that would repeat state writes.
        state.x_prev = block.attn(state.x_prev, positions)
        state = block.fuse_hc(
            state, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base, block.ffn_norm.weight, block.norm_eps
        )
        if unfused:
            state.x_prev = self.moe(state.x_prev, token_ids=token_ids, unfused=True)
        else:
            # Preserve ATOM's opaque MoE custom-op boundary for torch.compile.
            # The validated adapter above selects this bucket's mono launch.
            state.x_prev = block.ffn(state.x_prev)
        return state

    mono_kernel_forward = forward
    __call__ = forward

    def close(self):
        """Collectively release this bucket's MoE IPC mappings before TP teardown."""
        if not self.closed:
            self.moe.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["Dsv4MonoKernel"]
