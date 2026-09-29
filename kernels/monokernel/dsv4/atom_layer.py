# SPDX-License-Identifier: Apache-2.0
"""One real ATOM DSV4-Pro layer with native paged cache/metadata for validation."""

import json
from pathlib import Path
from types import SimpleNamespace


def load_layer_non_moe(block, checkpoint, layer, rank, tp):
    """Use each ATOM linear's loader for TP slicing and merged projections."""
    from safetensors import safe_open

    root = Path(checkpoint)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer}."
    params = {k: v for k, v in block.named_parameters() if not k.startswith("ffn.")}
    loaded = set()
    merged = (
        ("attn.wq_a.", "attn.wqkv_a.", 0),
        ("attn.wkv.", "attn.wqkv_a.", 1),
        ("compressor.wkv.", "compressor.wkv_gate.", 0),
        ("compressor.wgate.", "compressor.wkv_gate.", 1),
    )
    handles = {}
    for key, shard in index.items():
        if not key.startswith(prefix) or key.startswith(prefix + "ffn."):
            continue
        name = key[len(prefix) :]
        if name.endswith(".scale"):
            name = name[:-6] + ".weight_scale"
        shard_id = None
        for src, dest, which in merged:
            if src in name:
                name, shard_id = name.replace(src, dest), which
                break
        if name not in params:
            raise ValueError(f"unmapped non-MoE checkpoint parameter: {key} -> {name}")
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt", device="cpu")
        value = handles[shard].get_tensor(key)
        param = params[name]
        if name == "attn.attn_sink":
            value = value.chunk(tp)[rank]
        value = value.to(param.device)
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            if shard_id is None:
                loader(param, value)
            else:
                loader(param, value, shard_id)
        else:
            if param.shape != value.shape:
                raise ValueError(f"{key}: {value.shape} != {param.shape}")
            param.data.copy_(value)
        loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise ValueError(f"missing non-MoE layer weights: {sorted(missing)}")


class AtomLayerModule:
    def __init__(self, checkpoint, layer, weights, shared, table, tp, rank):
        import torch
        from atom.model_ops.attentions.deepseek_v4_attn import DeepseekV4AttentionMetadataBuilder
        from atom.utils import CpuGpuBuffer

        from .atom_module import AtomMoeModule

        self.fixture = AtomMoeModule(checkpoint, layer, weights, shared, table, tp, rank, full_layer=True)
        self.block = self.fixture.block
        if self.block.attn.indexer is not None and self.block.attn.skip_topk:
            raise ValueError("single-layer fixture requires a CSA index-refresh layer")
        self.config = self.fixture.config
        device = torch.device("cuda", rank)
        self.runner = runner = SimpleNamespace(
            config=self.config,
            model=self.block,
            device=device,
            block_size=256,
            max_num_batched_tokens=4,
            max_bs=1,
            kv_cache_dtype="fp8",
            drafter=SimpleNamespace(mtp_k=3, uses_confidence_schedule=False),
            async_execute_stream=torch.cuda.current_stream(),
            forward_vars={"positions": CpuGpuBuffer(4, dtype=torch.int64, device=device)},
        )
        self.builder = builder = DeepseekV4AttentionMetadataBuilder(runner)
        # The same sizing and allocation contracts used by ModelRunner, with
        # two pages and one private request slot. No full model is allocated.
        from atom.model_engine.kv_block import STATE_SLOT_CLASS
        from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
        from atom.model_engine.state_runtime import StateRuntime

        transfer = builder.state_transfer()
        page = (
            sum(builder.pool_geometry.block_bytes(w) for w in builder._plane_row_widths())
            + builder._indexer_page_bytes()
        )
        slot = sum(builder.pool_geometry.slot_bytes(w) for w in builder._plane_row_widths())
        runner.state_runtime = StateRuntime(
            transfer=transfer,
            checkpoint_spec=PagedStateCheckpointSpec(
                page_unit_bytes=page,
                slot_bytes=slot,
                image_bytes=builder.checkpoint_image_bytes(),
                layout_id=transfer.paged_layout_id,
            ),
        )
        runner.pool_plan = SimpleNamespace(entries={STATE_SLOT_CLASS: 1})
        for name, value in builder.allocate_kv_cache_tensors(blocks=2, buf=None).items():
            setattr(runner, name, value)
        for name, value in builder.allocate_per_req_cache(runner.pool_plan.entries).items():
            setattr(runner, name, value)
        for module in self.block.modules():
            builder.build_kv_cache_tensor(module)
        self.cache_tensors = self._cache_storage()

    def _cache_storage(self):
        """Unique backing storages, including both KV planes and compressor state."""
        import torch

        tensors = {}
        # Pool views overlap: snapshot their backing bytes once each.
        for name in ("v4_unified_kv", "v4_unified_kv_rope", "v4_csa_idx_kv", "v4_csa_idx_kv_scale"):
            value = getattr(self.runner, name, None)
            for tensor in value if isinstance(value, list) else [value]:
                if tensor is None:
                    continue
                storage = tensor.untyped_storage()
                ptr = storage.data_ptr()
                if ptr not in tensors:
                    tensors[ptr] = torch.empty(0, device=tensor.device, dtype=torch.uint8).set_(
                        storage, 0, (storage.nbytes(),), (1,)
                    )
        return list(tensors.values())

    def metadata(self, seq, token_ids):
        from atom.utils.forward_context import set_forward_context

        md, ctx = self.builder.build_for_cudagraph_capture(1, max_q_len=seq)
        ctx.input_ids = token_ids
        set_forward_context(md, self.config, ctx)
        return md, ctx

    def cache_fields(self, md):
        """Named live compressor fields for diagnosing backing-byte mismatches."""
        slot = int(md.state_slot_out_cpu[0])
        fields = {}
        for name, module in self.block.attn.named_modules():
            for attr in ("kv_state", "score_state"):
                tensor = getattr(module, attr, None)
                if tensor is not None:
                    fields[f"{name}.{attr}"] = tensor[slot]
        return fields

    def cache_observation(self, md, ctx):
        """Logical numerical state plus the exact backing regions it occupies.

        Only completed compression rows may change, in addition to the live
        FP32 ring fields. All other backing bytes are checked independently.
        """
        import torch
        from atom.model_ops.sparse_indexer_fp4 import fp4_index_scale_rows
        from atom.model_ops.v4_kernels.v4_quant import dequantize_v4_2buff_to_bf16

        from kernels.monokernel.formats import dequantize_mxfp4

        values = self.cache_fields(md)
        regions = list(values.values())
        attn = self.block.attn
        ratio = attn.compress_ratio
        if not ratio:
            return values, regions
        for position in ctx.positions.tolist():
            if (position + 1) % ratio:
                continue
            logical = (position + 1) // ratio - 1
            block_id = int(md.block_tables[0, position // 256])
            row = block_id * md.envelope_rows + logical % (256 // ratio)
            packed = attn.unified_kv[row]
            rope = attn.unified_kv_rope[row]
            scale_pairs = packed.view(torch.uint8)[448:462].reshape(7, 2)
            if not torch.equal(scale_pairs[:, 0], scale_pairs[:, 1]):
                raise AssertionError("native KV row has inconsistent duplicated E8M0 scales")
            values[f"main_compressed.{logical}"] = dequantize_v4_2buff_to_bf16(packed, rope)
            # The last 50 packed bytes are padding, not numerical payload.
            regions.extend((packed[:462], rope))
            if attn.indexer is not None:
                indexer = attn.indexer
                if not indexer._indexer_fp4:
                    raise ValueError("layer fixture currently requires ATOM's native gfx950 FP4 index cache")
                slot = logical % 64
                # The packed data is row-major within each 32-value group,
                # but gfx950 transposes the scale row axis for dword loads.
                scale_slot = fp4_index_scale_rows(slot, 64)
                data = indexer.kv_cache[block_id, 0, :, slot, :]
                scales = indexer.kv_scale[block_id, 0, :, scale_slot]
                values[f"index_compressed.{logical}"] = dequantize_mxfp4(
                    data.contiguous(), scales.contiguous().reshape(4, 1)
                )
                # gfx950 keeps each group of 32 values as 16 contiguous bytes;
                # groups have a block-row stride and are separate regions.
                for group in range(4):
                    regions.append(indexer.kv_cache[block_id, 0, group, slot])
                    regions.append(indexer.kv_scale[block_id, 0, group, scale_slot : scale_slot + 1])
        return values, regions

    def close(self):
        from atom.model_ops.dsv4_monokernel import close_monokernels

        close_monokernels(self.block)
        self.fixture.close()
