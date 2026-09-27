# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

import torch

from kernels.glm5_monokernel.kernel import build_glm5_monokernel
from kernels.glm5_monokernel.layout import (
    INDEX_DIM,
    layout,
    stage_tasks,
)
from kernels.mla_moe_layer.config import (
    GLM5_CONFIG,
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    validate_shard,
)
from kernels.mla_moe_layer.kernel_layout import TL_COLS
from kernels.mla_moe_layer.packing import pack_bf16, pack_fp8, pack_layer_weights
from kernels.mla_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer

__all__ = ["Glm5MonoKernel"]


class Glm5MonoKernel:
    """One TP rank of GLM-5's indexed decode MonoKernel.

    With ``with_indexer=True``, a single persistent launch covers index K/Q/W
    projection, index K normalization/RoPE/cache update, scoring, exact sparse
    top-k selection, MLA, routing, all expert compute, and both TP reductions.
    ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer uses PyTorch's caching allocator and CUDA/ROCm IPC
    storage sharing. Scratch and symmetric buffers may be shared by all layers
    because every launch uses a fresh ``tag``.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        topk: int = 2048,
        launches_per_step: int = 1,
        with_indexer: bool = False,
        index_max_seq: int = 4096,
        timeline=False,
    ):
        if W.config != GLM5_CONFIG:
            raise ValueError(f"Glm5MonoKernel requires GLM-5 weights, got {W.config.name!r}")
        validate_shard(samples, W.heads, rank, npes, topk, GLM5_CONFIG)
        if not 1 <= launches_per_step <= 128:
            raise ValueError(f"launches_per_step must be in [1, 128], got {launches_per_step}")
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.launches_per_step = launches_per_step
        self.with_indexer = with_indexer
        self.index_max_seq = index_max_seq
        t = W.t
        self.expert_mxfp4 = t["w_ug"].dtype is torch.uint8
        self.packed = pack_layer_weights(
            t,
            MoeMode.A16W4 if self.expert_mxfp4 else MoeMode.W8A8,
            GLM5_CONFIG,
            mxfp4_weight_layout=Mxfp4WeightLayout.NATIVE,
            mxfp4_scale_layout=Mxfp4ScaleLayout.NATIVE,
            router_weight_layout=RouterWeightLayout.NATIVE,
        )
        if with_indexer:
            required = ("w_index_k", "s_index_k", "w_index_w", "w_index_q", "s_index_q", "g_index_k", "b_index_k")
            missing = [name for name in required if name not in t]
            if missing:
                raise ValueError(f"with_indexer=True requires weights: {', '.join(missing)}")
            self.packed["w_index_k"] = pack_fp8(t["w_index_k"])
            self.packed["w_index_q"] = pack_fp8(t["w_index_q"])
            self.packed["w_index_w"] = pack_bf16(t["w_index_w"])
        self.scr_layout, self.sym_layout = layout(samples, W.heads, npes, topk, with_indexer, index_max_seq)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.stages = stage_tasks(samples, W.heads, topk, with_indexer, index_max_seq, self.expert_mxfp4)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        if with_indexer:
            index_tensors = dict(t, **self.packed)
            self.index_params = torch.tensor(
                [
                    index_tensors[name].data_ptr()
                    for name in (
                        "w_index_k",
                        "s_index_k",
                        "w_index_w",
                        "w_index_q",
                        "s_index_q",
                        "g_index_k",
                        "b_index_k",
                    )
                ]
                + [0 if self.timeline is None else self.timeline.data_ptr()],
                dtype=torch.int64,
                device=dev,
            )
        else:
            self.index_params = None
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses
        self.launch = build_glm5_monokernel(
            samples,
            W.heads,
            npes,
            topk,
            launches_per_step=launches_per_step,
            with_indexer=with_indexer,
            index_max_seq=index_max_seq,
            expert_mxfp4=self.expert_mxfp4,
            timeline=timeline,
        )
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

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        pe_cache,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
        index_cache=None,
    ):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
        if self.with_indexer:
            if index_cache is None:
                raise ValueError("index_cache is required when with_indexer=True")
            if index_cache.shape != (self.index_max_seq, INDEX_DIM) or index_cache.dtype is not torch.bfloat16:
                raise ValueError(
                    f"index_cache must be bf16 [{self.index_max_seq}, {INDEX_DIM}], got "
                    f"{tuple(index_cache.shape)} {index_cache.dtype}"
                )
        t = dict(self.W.t, **self.packed)
        if x_out is None:
            x_out = torch.empty(self.S, HIDDEN, dtype=torch.bfloat16, device=h.device)
        p = lambda x: x.data_ptr()  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_cache),
            p(pe_cache),
            p(index_cache) if self.with_indexer else p(indices),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            p(t["w_uk"]),
            p(t["s_uk"]),
            p(t["w_uv"]),
            p(t["s_uv"]),
            p(t["w_o"]),
            p(t["s_o"]),
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
            p(self.scratch),
            self.sym,
            p(self.peers),
            p(self.index_params) if self.with_indexer else (0 if self.timeline is None else p(self.timeline)),
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
            if name == "index_score":
                per_sample = n // self.S
                ready = [(st[s * per_sample : (s + 1) * per_sample, 4].max() - t0).item() for s in range(self.S)]
                rows.append(" " * 10 + "score-ready/sample " + " ".join(f"{v:.1f}" for v in ready))
            elif name == "index_select":
                done = [(st[s, 4] - t0).item() for s in range(self.S)]
                rows.append(" " * 10 + "select-done/sample " + " ".join(f"{v:.1f}" for v in done))
        return "\n".join(rows)

    def intermediates(self):
        S, H = self.S, self.W.heads
        result = dict(
            q_a=self.debug("q_a", (S, Q_LORA)),
            kv_a=self.debug("kv_a", (S, KV_LORA + PE_DIM)),
            q_nope=self.debug("q_nope", (S, H, NOPE_DIM), bf2=True),
            q_pe=self.debug("q_pe", (S, H, PE_DIM), bf2=True),
            q_lat=self.debug("q_lat", (S, H, KV_LORA), bf2=True),
            o=self.debug("o", (S, H * V_DIM), bf2=True),
            a=self.debug("a", (S, HIDDEN), bf2=True).to(torch.bfloat16),
            scores=self.debug("scores", (S, N_EXPERTS)),
            sel=self.debug("sel", (S, MOE_SLOTS), torch.int32),
            prob=self.debug("prob", (S, MOE_SLOTS)),
            mid=self.debug("mid", (S, MOE_SLOTS, INTER)),
            xq=self.debug("xqd", (S, HIDDEN), pairs=False),
        )
        if self.with_indexer:
            result["index_q"] = self.debug("index_q", (S, 32, INDEX_DIM), bf2=True)
            result["index_w"] = self.debug("index_w", (S, 32))
            off = self.scr_layout["indices"]
            result["indices"] = self.scratch[off : off + S * self.topk * 4].view(torch.int32).view(S, self.topk)
        return result
