# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's layer."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.config import (
    HIDDEN,
    INTER,
    MAX_LAYERS_PER_STEP,
    MOE_SLOTS,
    N_EXPERTS,
    ExpertActivation,
    MoeMode,
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

__all__ = ["Glm5IndexedMlaMoeBlock", "MoeMode"]


class Glm5IndexedMlaMoeBlock:
    """One rank of the indexed sparse MLA + MoE block.

    The caller supplies sparse-attention indices. This wrapper implements the
    symmetric eight-head-per-rank reuse topology; the asymmetric refresh
    topology uses the same MoE math but needs separate rank-specific wrappers.
    ``group`` is a torch.distributed group (None for npes=1).

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
        sparse_attention_topk: int = 2048,
        launches_per_step: int = 1,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.W8A8,
    ):
        validate_shard(samples, W.heads, rank, npes, sparse_attention_topk)
        if not 1 <= launches_per_step <= MAX_LAYERS_PER_STEP:
            raise ValueError(f"launches_per_step must be in [1, {MAX_LAYERS_PER_STEP}], got {launches_per_step}")
        self.moe_mode = as_moe_mode(moe_mode)
        self.W = W
        self.S = samples
        self.rank = rank
        self.npes = npes
        self.sparse_attention_topk = sparse_attention_topk
        self.launches_per_step = launches_per_step
        self.packed = pack_layer_weights(W.t, self.moe_mode)
        self.scr_layout, self.sym_layout = layout(
            samples,
            W.heads,
            npes,
            sparse_attention_topk,
            self.moe_mode,
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
        )
        self.stages = stage_tasks(samples, W.heads, sparse_attention_topk)
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

        For the normal one-invocation-per-step path, keep ``launches_per_step=1``
        and ``layer=0``. A benchmark that reuses this object multiple times in
        one graph must set ``launches_per_step`` to that count, pass consecutive
        ``layer`` ordinals, and advance the step once after the final launch.
        """
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
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
            p(sparse_indices),
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
        from kernels.mla_moe_layer.config import KV_LORA, NOPE_DIM, PE_DIM, Q_LORA, V_DIM

        mid = self.debug("mid", (S, MOE_SLOTS, INTER))
        if moe_format(self.moe_mode).activation is ExpertActivation.BF16:
            mid = mid.to(torch.bfloat16).float()
        return dict(
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
            mid=mid,
            xq=self.debug("xqd", (S, HIDDEN), pairs=False),
        )
