# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's layer."""

from __future__ import annotations

import torch

from kernels.comm.custom_all_reduce import FlyDSLAllreduce as _Hip
from kernels.mla_moe_layer.glm5_mla_moe_layer import build_layer, layout, pack_bf16, pack_fp8, stage_tasks
from kernels.mla_moe_layer.reference import HIDDEN, INTER, MOE_SLOTS, N_EXPERTS, LayerWeights

__all__ = ["Glm5MlaMoeLayer"]


class Glm5MlaMoeLayer:
    """One rank of the TP layer. ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer is hipDeviceMallocUncached memory exported to every
    peer through HIP IPC; scratch and symmetric buffers may be shared by all
    layers because every launch uses a fresh ``tag``.
    """

    def __init__(
        self, W: LayerWeights, samples: int, rank: int = 0, npes: int = 1, group=None, topk: int = 2048, timeline=False
    ):
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        # MFMA-native weight packings (see pack_fp8 / pack_bf16)
        t = W.t
        self.packed = {n: pack_fp8(t[n]) for n in ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o", "w_ug", "w_dn")}
        self.packed["w_r"] = pack_bf16(t["w_r"])
        self.scr_layout, self.sym_layout = layout(samples, W.heads, npes, topk)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.sym = _Hip._alloc_uncached(self.sym_layout["_bytes"])
        if npes == 1:
            addrs = [self.sym]
        else:
            import torch.distributed as dist

            # IPC handles name the whole allocation; the buffer may sit at an offset in it
            base = _Hip._get_alloc_base_ptr(self.sym)
            mine = (_Hip._get_mem_handle_bytes(base), self.sym - base)
            peers = [None] * npes
            dist.all_gather_object(peers, mine, group=group)
            addrs = [self.sym if i == rank else _Hip._open_mem_handle(peers[i][0]) + peers[i][1] for i in range(npes)]
            dist.barrier(group=group)
        self.peers = torch.tensor(addrs, dtype=torch.int64, device=dev)
        self.launch = build_layer(samples, W.heads, npes, topk, timeline=timeline)
        self.stages = stage_tasks(samples, W.heads, topk)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, 3, dtype=torch.int64, device=dev) if timeline else None
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
        return words.view(dtype).view(shape)

    def forward(self, h, cur_pos, kv_cache, pe_cache, indices, cos, sin, x_out=None, layer=0, advance=True):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
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
            p(indices),
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

    def timeline_report(self) -> str:
        """Per stage: [first start, last end] and mean wait / work per task, in us from launch start."""
        tl = self.timeline.cpu().double() / 100.0  # s_memrealtime ticks at 100 MHz
        t0 = tl[:, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n]
            i += n
            ready = torch.where(st[:, 1] > 0, st[:, 1], st[:, 0])
            rows.append(
                f"{name:7s} x{n:4d}  span [{(st[:, 0].min() - t0):7.1f}, {(st[:, 2].max() - t0):7.1f}]"
                f"  wait {(ready - st[:, 0]).mean():6.1f}  work {(st[:, 2] - ready).mean():6.1f}"
            )
        return "\n".join(rows)

    def intermediates(self):
        S, H = self.S, self.W.heads
        from kernels.mla_moe_layer.reference import KV_LORA, NOPE_DIM, PE_DIM, Q_LORA, V_DIM

        return dict(
            q_a=self.debug("q_a", (S, Q_LORA)),
            kv_a=self.debug("kv_a", (S, KV_LORA + PE_DIM)),
            q_nope=self.debug("q_nope", (S, H, NOPE_DIM)),
            q_pe=self.debug("q_pe", (S, H, PE_DIM)),
            q_lat=self.debug("q_lat", (S, H, KV_LORA)),
            o=self.debug("o", (S, H * V_DIM)),
            a=self.debug("a", (S, HIDDEN)).to(torch.bfloat16),
            scores=self.debug("scores", (S, N_EXPERTS)),
            sel=self.debug("sel", (S, MOE_SLOTS), torch.int32),
            prob=self.debug("prob", (S, MOE_SLOTS)),
            mid=self.debug("mid", (S, MOE_SLOTS, INTER)),
            xq=self.debug("xqd", (S, HIDDEN), pairs=False),
        )
