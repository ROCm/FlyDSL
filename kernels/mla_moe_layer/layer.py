# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's layer."""

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
from kernels.mla_moe_layer.packing import pack_layer_weights
from kernels.mla_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer
from kernels.mla_moe_layer.shared_reuse_moe_kernel import (
    TL_COLS,
    build_shared_reuse_kernel,
    layout,
    stage_tasks,
)

__all__ = ["KimiK3MlaLayer", "MoeMode", "SharedReuseMlaMoeLayer"]


class SharedReuseMlaMoeLayer:
    """One rank of the TP layer. ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer is a torch allocation exported to every peer through
    HIP IPC; scratch and symmetric buffers may be shared by all
    layers because every launch uses a fresh ``tag``.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        topk: int = 2048,
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
        validate_shard(samples, W.heads, rank, npes, topk, self.config)
        self.moe_mode = as_moe_mode(moe_mode)
        self.attention_only = attention_only
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.packed = pack_layer_weights(W.t, self.moe_mode, self.config, attention_only)
        self.scr_layout, self.sym_layout = layout(
            samples,
            W.heads,
            npes,
            topk,
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
        self.launch = build_shared_reuse_kernel(
            samples,
            W.heads,
            npes,
            topk,
            timeline=timeline,
            moe_mode=self.moe_mode,
            model_config=self.config,
            attention_only=attention_only,
        )
        self.stages = stage_tasks(samples, W.heads, topk, self.config, attention_only)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
        ``bf2``: each pair's value word packs two bf16 elements)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = self.scratch[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
            return words.view(torch.bfloat16).float().view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
        return words.view(dtype).view(shape)

    def forward(self, h, cur_pos, kv_cache, pe_cache, indices, cos, sin, x_out=None, layer=0, advance=True):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < MAX_LAYERS_PER_STEP:
            raise ValueError(f"layer must be in [0, {MAX_LAYERS_PER_STEP}), got {layer}")
        t = dict(self.W.t, **self.packed)
        if x_out is None:
            x_out = torch.empty(self.S, self.config.hidden, dtype=torch.bfloat16, device=h.device)
        p = lambda x: x.data_ptr()  # noqa: E731
        p_or_zero = lambda name: p(t[name]) if name in t else 0  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_cache),
            p(pe_cache),
            p(indices),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p_or_zero("w_qkv_a"),
            p_or_zero("s_qkv_a"),
            p_or_zero("w_q_b"),
            p_or_zero("s_q_b"),
            p_or_zero("w_uk"),
            p_or_zero("s_uk"),
            p_or_zero("w_uv"),
            p_or_zero("s_uv"),
            p_or_zero("w_o"),
            p_or_zero("s_o"),
            p_or_zero("w_r"),
            p_or_zero("bias"),
            p_or_zero("w_ug"),
            p_or_zero("s_ug"),
            p_or_zero("w_dn"),
            p_or_zero("s_dn"),
            p(self.scratch),
            self.sym,
            p(self.peers),
            0 if self.timeline is None else p(self.timeline),
            p(self.step),
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
        """Per stage, in us from launch start: [first start, median hint seen, last end]
        and median per-task phases (hint wait, payload staging, compute, epilogue)."""
        if self.timeline is None:
            raise RuntimeError("timeline collection was not enabled")
        tl = self.timeline[:, :5].cpu().double() / 100.0  # s_memrealtime ticks at 100 MHz
        t0 = tl[:, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n].clone()
            i += n
            for c in (1, 2, 3):  # missing marks inherit the previous one
                st[:, c] = torch.where(st[:, c] > 0, st[:, c], st[:, c - 1])
            d = (st[:, 1:] - st[:, :-1]).median(0).values
            rows.append(
                f"{name:7s} x{n:4d}  [{(st[:, 0].min() - t0):6.1f} | hint {(st[:, 1].median() - t0):6.1f} | "
                f"end {(st[:, 4].max() - t0):6.1f}]  hint {d[0]:5.1f}  stage {d[1]:5.1f}  "
                f"compute {d[2]:5.1f}  epi {d[3]:5.1f}"
            )
        return "\n".join(rows)

    def intermediates(self):
        S, H = self.S, self.W.heads
        config = self.config

        attention = dict(
            q_a=self.debug("q_a", (S, config.q_lora)),
            kv_a=self.debug("kv_a", (S, config.kv_lora + config.pe_dim)),
            q_nope=self.debug("q_nope", (S, H, config.nope_dim), bf2=True),
            q_pe=self.debug("q_pe", (S, H, config.pe_dim), bf2=True),
            q_lat=self.debug("q_lat", (S, H, config.kv_lora), bf2=True),
            o=self.debug("o", (S, H * config.v_dim), bf2=True),
            a=self.debug("a", (S, config.hidden), bf2=True).to(torch.bfloat16),
        )
        if config.attention_output_gate:
            attention["gate"] = self.debug("gate", (S, H, config.v_dim))
        if self.attention_only:
            return attention

        mid = self.debug("mid", (S, MOE_SLOTS, INTER))
        if moe_format(self.moe_mode).activation is ExpertActivation.BF16:
            mid = mid.to(torch.bfloat16).float()
        return dict(
            **attention,
            scores=self.debug("scores", (S, N_EXPERTS)),
            sel=self.debug("sel", (S, MOE_SLOTS), torch.int32),
            prob=self.debug("prob", (S, MOE_SLOTS)),
            mid=mid,
            xq=self.debug("xqd", (S, HIDDEN), pairs=False),
        )


class KimiK3MlaLayer(SharedReuseMlaMoeLayer):
    """Kimi-K3 TP8 full-attention shard, excluding the latent-MoE tail."""

    def __init__(self, W: LayerWeights, samples: int, **kwargs):
        kwargs.pop("model_config", None)
        kwargs.pop("attention_only", None)
        super().__init__(
            W,
            samples,
            model_config=KIMI_K3_CONFIG,
            attention_only=True,
            **kwargs,
        )
