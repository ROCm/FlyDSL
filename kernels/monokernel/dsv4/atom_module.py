# SPDX-License-Identifier: Apache-2.0
"""Real ATOM MoE module fixture for the component comparison script."""

import os
import tempfile


class AtomMoeModule:
    def __init__(self, checkpoint, layer, weights, shared, table, tp, rank, *, full_layer=False):
        import torch
        from aiter.dist.parallel_state import init_distributed_environment, initialize_model_parallel
        from atom.config import Config, set_current_atom_config
        from atom.model_ops.base_config import QuantizeMethodBase
        from atom.model_ops.moe import FusedMoEMethodBase
        from atom.models.deepseek_v4 import Block, DeepseekV4Args, MoE, make_v4_quant_config

        if not torch.distributed.is_initialized() and "MASTER_ADDR" not in os.environ:
            # A unique FileStore also permits direct --tp 1 invocation without
            # a fixed port that could collide with another validation process.
            self._store_dir = tempfile.TemporaryDirectory(prefix="dsv4-module-")
            init_method = "file://" + self._store_dir.name + "/store"
        else:
            self._store_dir = None
            init_method = "env://"
        init_distributed_environment(world_size=tp, rank=rank, local_rank=rank, distributed_init_method=init_method)
        initialize_model_parallel(tensor_model_parallel_size=tp)
        self.config = Config(
            model=str(checkpoint),
            tensor_parallel_size=tp,
            max_num_batched_tokens=4,
            max_num_seqs=1,
            max_model_len=512,
            enforce_eager=True,
            kv_cache_dtype="fp8",
        )
        set_current_atom_config(self.config)
        args = DeepseekV4Args.from_hf_config(self.config.hf_config)
        args.quant_config = make_v4_quant_config(self.config.hf_config, model_path=str(checkpoint))
        self.config.quant_config = args.quant_config
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.config.torch_dtype)
            with torch.device("cuda", rank):
                if full_layer:
                    self.block = Block(layer, args, prefix=f"layers.{layer}")
                    self.module = self.block.ffn
                else:
                    self.module = MoE(layer, args, prefix=f"model.layers.{layer}.ffn")
        finally:
            torch.set_default_dtype(default_dtype)
        module = self.module
        router, bias, up, us, down, ds = weights
        pairs = [
            (module.gate.weight, router),
            (module.experts.w13_weight, up),
            (module.experts.w13_weight_scale, us),
            (module.experts.w2_weight, down),
            (module.experts.w2_weight_scale, ds),
            (module.shared_experts.gate_up_proj.weight, shared[0]),
            (module.shared_experts.gate_up_proj.weight_scale, shared[1]),
            (module.shared_experts.w2.weight, shared[2]),
            (module.shared_experts.w2.weight_scale, shared[3]),
        ]
        pairs.append(
            (module.gate.tid2eid, table) if module.is_hash_layer else (module.gate.e_score_correction_bias, bias)
        )
        for target, source in pairs:
            if target.shape != source.shape:
                raise ValueError(f"ATOM checkpoint shape mismatch: {target.shape} != {source.shape}")
            if target.element_size() == source.element_size() == 1:
                target.data.view(torch.uint8).copy_(source.view(torch.uint8))
            else:
                target.data.copy_(source)
        if full_layer:
            from .atom_layer import load_layer_non_moe

            load_layer_non_moe(self.block, checkpoint, layer, rank, tp)
        # Match the real loader's parent-first ordering and method type gates.
        for child in (self.block if full_layer else module).modules():
            post = getattr(child, "process_weights_after_loading", None)
            if callable(post):
                post()
            quant = getattr(child, "quant_method", None)
            if isinstance(quant, QuantizeMethodBase):
                quant.process_weights_after_loading(child)
            if isinstance(quant, FusedMoEMethodBase):
                quant.init_prepare_finalize(child)
        self.adapter = module._moe_mono
        if self.adapter is None or set(self.adapter.ops) != {1, 2, 3, 4}:
            raise AssertionError("the real ATOM module did not prepare the mono adapter")
        # Assert that preparation did not keep a duplicate of routed weights.
        assert self.adapter.prepared.weights[2] is module.experts.w13_weight
        assert self.adapter.prepared.weights[4] is module.experts.w2_weight
        self.positions = torch.arange(4, device=router.device)
        self.selected_dtype = "torch.float8_e4m3fn"

    def __call__(self, x, hash_ids=None, *, token_ids=None, mono=False):
        from atom.utils.forward_context import Context, set_forward_context

        seq = x.shape[0]
        context = Context(
            positions=self.positions[:seq],
            running_bs=1,
            running_tokens=seq,
            scheduled_bs=1,
            scheduled_tokens=seq,
            input_ids=token_ids,
        )
        set_forward_context(None, self.config, context)
        self.module._moe_mono = self.adapter if mono else None
        try:
            return self.module(x)
        finally:
            self.module._moe_mono = self.adapter

    def close(self):
        self.adapter.close()
        if self._store_dir is not None:
            self._store_dir.cleanup()
