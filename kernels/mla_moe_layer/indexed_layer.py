# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Model-configured wrapper for the extensible indexed MLA + MoE kernel."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.config import (
    GLM5_CONFIG,
    HIDDEN,
    INTER,
    KIMI_K3_CONFIG,
    MAX_LAYERS_PER_STEP,
    MOE_SLOTS,
    N_EXPERTS,
    ExpertActivation,
    LayerConfig,
    MoeMode,
    as_layer_config,
    as_moe_mode,
    moe_format,
    validate_shard,
)
from kernels.mla_moe_layer.indexed_mla_moe_kernel import (
    TL_COLS,
    build_indexed_mla_moe_kernel,
    layout,
    stage_tasks,
)
from kernels.mla_moe_layer.packing import pack_layer_weights
from kernels.mla_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer

__all__ = ["Glm5IndexedMlaMoeBlock", "IndexedMlaMoeBlock", "KimiK3MlaLayer", "MoeMode"]


class IndexedMlaMoeBlock:
    """One rank of a model-configured indexed sparse MLA block.

    The caller supplies sparse-attention indices. This wrapper implements the
    symmetric reuse topology; model-specific wrappers select a compile-time
    geometry and whether the persistent kernel stops after attention.

    The symmetric buffer is a torch allocation exported to every peer through
    HIP IPC; scratch and symmetric buffers may be shared by all layers because
    every launch uses a fresh tag.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        sparse_attention_topk: int = 2048,
        launches_per_step: int = 1,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.W8A8,
        model_config: LayerConfig | str = GLM5_CONFIG,
        attention_only: bool = False,
    ):
        self.config = as_layer_config(model_config)
        if W.config != self.config:
            raise ValueError(f"weight profile {W.config.name!r} does not match {self.config.name!r}")
        if self.config != GLM5_CONFIG and not attention_only:
            raise ValueError(f"{self.config.name} currently supports the attention-only kernel path")
        validate_shard(samples, W.heads, rank, npes, sparse_attention_topk, self.config)
        if not 1 <= launches_per_step <= MAX_LAYERS_PER_STEP:
            raise ValueError(f"launches_per_step must be in [1, {MAX_LAYERS_PER_STEP}], got {launches_per_step}")
        self.moe_mode = as_moe_mode(moe_mode)
        self.attention_only = attention_only
        self.W = W
        self.S = samples
        self.rank = rank
        self.npes = npes
        self.sparse_attention_topk = sparse_attention_topk
        self.launches_per_step = launches_per_step
        self.packed = pack_layer_weights(W.t, self.moe_mode, self.config, attention_only)
        self.scr_layout, self.sym_layout = layout(
            samples,
            W.heads,
            npes,
            sparse_attention_topk,
            self.moe_mode,
            self.config,
            attention_only,
        )
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses
        self.launch = build_indexed_mla_moe_kernel(
            samples,
            W.heads,
            npes,
            sparse_attention_topk,
            launches_per_step=launches_per_step,
            timeline=timeline,
            moe_mode=self.moe_mode,
            model_config=self.config,
            attention_only=attention_only,
        )
        self.stages = stage_tasks(samples, W.heads, sparse_attention_topk, self.config, attention_only)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False) -> torch.Tensor:
        """Read a scratch mailbox, optionally decoding tagged or BF16-pair values."""

        off = self.scr_layout[name]
        n = 1
        for dimension in shape:
            n *= dimension
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = self.scratch[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
            return words.view(torch.bfloat16).float().view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
        return words.view(dtype).view(shape)

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        pe_cache,
        sparse_indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
    ):
        """Launch one block invocation.

        For one invocation per step, keep ``launches_per_step=1`` and
        ``layer=0``. A graph that reuses the object several times must pass
        consecutive layer ordinals and advance the step after the final call.
        """

        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
        tensors = dict(self.W.t, **self.packed)
        if x_out is None:
            x_out = torch.empty(self.S, self.config.hidden, dtype=torch.bfloat16, device=h.device)
        pointer = lambda value: value.data_ptr()  # noqa: E731
        pointer_or_zero = lambda name: pointer(tensors[name]) if name in tensors else 0  # noqa: E731
        self.launch(
            pointer(h),
            pointer(x_out),
            pointer(cur_pos),
            pointer(kv_cache),
            pointer(pe_cache),
            pointer(sparse_indices),
            pointer(cos),
            pointer(sin),
            pointer(tensors["g_in"]),
            pointer(tensors["g_q"]),
            pointer(tensors["g_kv"]),
            pointer(tensors["g_post"]),
            pointer_or_zero("w_qkv_a"),
            pointer_or_zero("s_qkv_a"),
            pointer_or_zero("w_q_b"),
            pointer_or_zero("s_q_b"),
            pointer_or_zero("w_uk"),
            pointer_or_zero("s_uk"),
            pointer_or_zero("w_uv"),
            pointer_or_zero("s_uv"),
            pointer_or_zero("w_o"),
            pointer_or_zero("s_o"),
            pointer_or_zero("w_r"),
            pointer_or_zero("bias"),
            pointer_or_zero("w_ug"),
            pointer_or_zero("s_ug"),
            pointer_or_zero("w_dn"),
            pointer_or_zero("s_dn"),
            pointer(self.scratch),
            self.sym,
            pointer(self.peers),
            0 if self.timeline is None else pointer(self.timeline),
            pointer(self.step),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self):
        self.step.add_(1)

    def close(self):
        """Release this rank's remote HIP IPC mappings."""

        self.peer_buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def timeline_report(self) -> str:
        """Return timing ranges and phase medians for each scheduled stage."""

        if self.timeline is None:
            raise RuntimeError("timeline collection was not enabled")
        timeline = self.timeline[:, :5].cpu().double() / 100.0
        origin = timeline[:, 0].min()
        rows, offset = [], 0
        for name, count in self.stages:
            stage = timeline[offset : offset + count].clone()
            offset += count
            for column in (1, 2, 3):
                stage[:, column] = torch.where(
                    stage[:, column] > 0,
                    stage[:, column],
                    stage[:, column - 1],
                )
            durations = (stage[:, 1:] - stage[:, :-1]).median(0).values
            rows.append(
                f"{name:7s} x{count:4d}  "
                f"[{(stage[:, 0].min() - origin):6.1f} | "
                f"hint {(stage[:, 1].median() - origin):6.1f} | "
                f"end {(stage[:, 4].max() - origin):6.1f}]  "
                f"hint {durations[0]:5.1f}  stage {durations[1]:5.1f}  "
                f"compute {durations[2]:5.1f}  epi {durations[3]:5.1f}"
            )
        return "\n".join(rows)

    def intermediates(self):
        samples, heads = self.S, self.W.heads
        config = self.config
        attention = {
            "q_a": self.debug("q_a", (samples, config.q_lora)),
            "kv_a": self.debug("kv_a", (samples, config.kv_lora + config.pe_dim)),
            "q_nope": self.debug("q_nope", (samples, heads, config.nope_dim), bf2=True),
            "q_pe": self.debug("q_pe", (samples, heads, config.pe_dim), bf2=True),
            "q_lat": self.debug("q_lat", (samples, heads, config.kv_lora), bf2=True),
            "o": self.debug("o", (samples, heads * config.v_dim), bf2=True),
            "a": self.debug("a", (samples, config.hidden), bf2=True).to(torch.bfloat16),
        }
        if config.attention_output_gate:
            attention["gate"] = self.debug("gate", (samples, heads, config.v_dim))
        if self.attention_only:
            return attention

        mid = self.debug("mid", (samples, MOE_SLOTS, INTER))
        if moe_format(self.moe_mode).activation is ExpertActivation.BF16:
            mid = mid.to(torch.bfloat16).float()
        return {
            **attention,
            "scores": self.debug("scores", (samples, N_EXPERTS)),
            "sel": self.debug("sel", (samples, MOE_SLOTS), torch.int32),
            "prob": self.debug("prob", (samples, MOE_SLOTS)),
            "mid": mid,
            "xq": self.debug("xqd", (samples, HIDDEN), pairs=False),
        }


class Glm5IndexedMlaMoeBlock(IndexedMlaMoeBlock):
    """GLM-5 indexed sparse MLA + MoE block."""

    def __init__(self, W: LayerWeights, samples: int, **kwargs):
        kwargs.pop("model_config", None)
        kwargs.pop("attention_only", None)
        super().__init__(W, samples, model_config=GLM5_CONFIG, attention_only=False, **kwargs)


class KimiK3MlaLayer(IndexedMlaMoeBlock):
    """Kimi-K3 TP8 full-attention shard, excluding the latent-MoE tail."""

    def __init__(self, W: LayerWeights, samples: int, **kwargs):
        kwargs.pop("model_config", None)
        kwargs.pop("attention_only", None)
        if "topk" in kwargs:
            kwargs["sparse_attention_topk"] = kwargs.pop("topk")
        kwargs.setdefault("launches_per_step", MAX_LAYERS_PER_STEP)
        super().__init__(
            W,
            samples,
            model_config=KIMI_K3_CONFIG,
            attention_only=True,
            **kwargs,
        )
