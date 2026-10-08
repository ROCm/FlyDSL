# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's V4 layer."""

from __future__ import annotations

import torch

from kernels.monokernel.dsv4.config import (
    BLOCK_TOKENS,
    MAX_LAYERS_PER_STEP,
    ExpertActivation,
    MoeMode,
    as_moe_mode,
    moe_format,
    validate_shard,
)
from kernels.monokernel.dsv4.kernel import build_advance_step, build_dsv4_kernel
from kernels.monokernel.dsv4.kernel.plan import TL_COLS, layout, stage_tasks
from kernels.monokernel.dsv4.packing import pack_layer_weights
from kernels.monokernel.dsv4.reference import LayerWeights, fp4_pool_shapes
from kernels.monokernel.runtime import SymmetricPeerBuffer

__all__ = ["MoeMode", "Dsv4MonoKernel", "Dsv4Variant", "shape_dims"]


def shape_dims(cfg) -> dict:
    """The shape arguments layout(), stage_tasks() and build_dsv4_kernel() must share identically.

    The JIT disk cache does not key on layout()/stage_tasks(): edit them with FLYDSL_RUNTIME_ENABLE_CACHE=0.
    """
    return dict(
        hidden=cfg.hidden,
        q_lora=cfg.q_lora,
        head_dim=cfg.head_dim,
        o_groups=cfg.o_groups,
        o_lora=cfg.o_lora,
        hc_mult=cfg.hc_mult,
        compress_ratio=cfg.compress_ratio,
        n_keys=cfg.n_keys,
        c_coff=cfg.c_coff,
        # 0: no indexer (only CSA has one)
        index_head_dim=cfg.index_head_dim if cfg.indexed else 0,
        index_heads=cfg.index_heads if cfg.indexed else 0,
        max_seq=cfg.max_seq,
        index_topk=cfg.index_topk if cfg.indexed else 0,
        kv_fp8=cfg.kv_fp8,
        indexer_hadamard=cfg.indexer_hadamard,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        inter=cfg.inter,
    )


def _check_i32(name: str, t: torch.Tensor, shape: tuple) -> None:
    """Index tensors are read by raw buffer loads as contiguous int32."""
    if tuple(t.shape) != shape or t.dtype != torch.int32 or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous int32 {shape}, got {tuple(t.shape)} {t.dtype}")


def _variant_key(cfg, samples, npes, moe_mode, timeline, poll_timeout_us, tokens_per_seq=1):
    """Everything the compiled kernel and the scratch layout depend on."""
    return (
        poll_timeout_us,
        tokens_per_seq,
        tuple(sorted(shape_dims(cfg).items())),
        samples,
        cfg.heads,
        npes,
        cfg.window,
        cfg.cache_rows,
        cfg.n_experts,
        cfg.top_k,
        cfg.inter,
        cfg.softmax_scale,
        cfg.swiglu_limit,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        moe_mode,
        timeline,
    )


class Dsv4Variant:
    """The compiled kernel, scratch and symmetric buffer shared by all layers of one attention variant.

    Scratch cannot be shared across variants (different mailbox layouts). Within one, fresh epoch
    tags make sharing safe, so ``step`` lives here: one counter per scratch, advanced once per step.
    """

    _cache: dict = {}
    _adv_cache: dict = {}

    def __init__(
        self,
        cfg,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
        poll_timeout_us: int | None = None,
        tokens_per_seq: int = 1,
    ):
        moe_mode = as_moe_mode(moe_mode)
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if samples % tokens_per_seq:
            raise ValueError(f"{samples} samples are not whole runs of {tokens_per_seq} tokens per sequence")
        self.poll_timeout_us = poll_timeout_us
        self.tokens_per_seq = tokens_per_seq
        self.key = _variant_key(cfg, samples, npes, moe_mode, timeline, poll_timeout_us, tokens_per_seq)
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(samples, cfg.heads, npes, cfg.window, moe_mode, **dims)
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter
        # nonzero once a bounded poll timed out: that launch's and later outputs are garbage
        self.hang = torch.zeros(1, dtype=torch.int32, device=dev)
        built = Dsv4Variant._cache.get(self.key)
        if built is None:
            built = build_dsv4_kernel(
                samples,
                cfg.heads,
                npes,
                cfg.window,
                scale=cfg.softmax_scale,
                timeline=timeline,
                moe_mode=moe_mode,
                poll_timeout_us=poll_timeout_us,
                tokens_per_seq=tokens_per_seq,
                swiglu_limit=cfg.swiglu_limit,
                hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                hc_eps=cfg.hc_eps,
                window_rows=cfg.cache_rows,
                **dims,
            )
            Dsv4Variant._cache[self.key] = built
        self.launch = built
        # the step advance clears mailbox pairs; the plain-f32 "xqd" region sits last and is skipped
        self.scr_pairs = self.scr_layout["xqd"] // 8
        self.sym_pairs = self.sym_layout["_bytes"] // 8
        adv_key = (self.scr_pairs, self.sym_pairs)
        if adv_key not in Dsv4Variant._adv_cache:
            Dsv4Variant._adv_cache[adv_key] = build_advance_step(*adv_key)
        self._advance = Dsv4Variant._adv_cache[adv_key]
        self.stages = stage_tasks(samples, cfg.heads, window=cfg.window, **dims)

    def hang_detected(self) -> bool:
        """Whether a launch timed out a poll (needs ``poll_timeout_us``). Synchronizes."""
        return bool(self.hang.item())

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs so tags can wrap; graph-capturable."""
        self._advance(self.step.data_ptr(), self.scratch.data_ptr(), self.sym, stream=torch.cuda.current_stream())

    def close(self):
        self.peer_buffer.close()


class Dsv4MonoKernel:
    """One rank of one transformer layer: its weights and rolling state.

    Layers sharing a ``Dsv4Variant`` need distinct ``layer`` values and one step advance per decode step.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
        variant: "Dsv4Variant | None" = None,
        packed: dict | None = None,
        tokens_per_seq: int = 1,
        experts: dict | None = None,
        own_state: bool = True,
    ):
        """``packed``: another layer's packed weights to share. ``tokens_per_seq`` > 1: MTP verify runs.
        ``experts``: a packed MXFP4 bank (ATOM's) used in place. ``own_state=False``: the caller passes
        all per-sequence state to ``forward``."""
        cfg = W.cfg
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if cfg.hc_mult > 1 and cfg.hc_mult & (cfg.hc_mult - 1):
            raise ValueError(f"hc_mult must be 1 or a power of two, got {cfg.hc_mult}")
        self.moe_mode = as_moe_mode(moe_mode)
        # the kernel reads N_EXPERTS f32; a bf16 bias would be read past its end
        if W.t["bias"].dtype != torch.float32:
            raise ValueError(f"the router bias must be float32, got {W.t['bias'].dtype}")
        self.W, self.S, self.rank, self.npes = W, samples, rank, npes
        self.window = cfg.window
        self.packed = pack_layer_weights(W.t, self.moe_mode, experts) if packed is None else packed
        # [3 scales | hc_mix bases] per side, as one f32 vector the kernel indexes
        dev0 = torch.device("cuda", torch.cuda.current_device())
        self.hc_sb = {}
        for side in ("attn", "ffn"):
            if f"hc_{side}_scale" in W.t:
                self.hc_sb[side] = torch.cat(
                    [W.t[f"hc_{side}_scale"].float(), W.t[f"hc_{side}_base"].float()]
                ).contiguous()
            else:
                self.hc_sb[side] = torch.zeros(1, dtype=torch.float32, device=dev0)
        dev = torch.device("cuda", torch.cuda.current_device())
        # a variant passed in is shared by other layers: closing it is the caller's job
        self._owns_variant = variant is None
        if variant is None:
            variant = Dsv4Variant(
                cfg,
                samples,
                rank=rank,
                npes=npes,
                group=group,
                timeline=timeline,
                moe_mode=self.moe_mode,
                tokens_per_seq=tokens_per_seq,
            )
        elif variant.key != _variant_key(
            cfg, samples, npes, self.moe_mode, timeline, variant.poll_timeout_us, variant.tokens_per_seq
        ):
            raise ValueError(
                "this layer's shape is not the one the variant was compiled for; a variant is shared "
                "only by layers of the SAME attention variant (compress_ratio, n_keys, max_seq, ...)"
            )
        self.variant = variant
        self.scr_layout, self.sym_layout = variant.scr_layout, variant.sym_layout
        self.scratch, self.sym, self.peers = variant.scratch, variant.sym, variant.peers
        self.launch, self.stages = variant.launch, variant.stages
        tok = self.tokens_per_seq = variant.tokens_per_seq
        n_seq = samples // tok
        c_ring = cfg.c_rows + tok - 1  # the kernel's C_RING
        # rolling compressor state, one per sequence (the S compressor tasks are unordered)
        self.own_state = own_state
        if cfg.indexed and own_state:
            ishape = (n_seq, c_ring, cfg.c_coff * cfg.index_head_dim)
            self.i_kv_state = torch.zeros(*ishape, dtype=torch.float32, device=dev)
            self.i_score_state = torch.full(ishape, float("-inf"), dtype=torch.float32, device=dev)
            # the indexer's key cache in ATOM's paged FP4 pool layout (codes, scales)
            dshape, sshape = fp4_pool_shapes(cfg, n_seq)
            self.i_cache = torch.zeros(*dshape, dtype=torch.uint8, device=dev)
            self.i_cache_s = torch.zeros(*sshape, dtype=torch.uint8, device=dev)
        else:
            self.i_kv_state = self.i_score_state = self.i_cache = self.i_cache_s = torch.zeros(
                1, dtype=torch.float32, device=dev
            )
        if cfg.compress_ratio and own_state:
            shape = (n_seq, c_ring, cfg.c_coff * cfg.head_dim)
            self.kv_state = torch.zeros(*shape, dtype=torch.float32, device=dev)
            # -inf: CSA's overlapping window rows are unwritten before the first emit
            self.score_state = torch.full(shape, float("-inf"), dtype=torch.float32, device=dev)
        else:
            self.kv_state = self.score_state = torch.zeros(1, dtype=torch.float32, device=dev)
        # trivial state pool: one contiguous slot per sequence; a serving pool passes its own
        self.state_slots = torch.arange(samples, dtype=torch.int32, device=dev) // tok
        self.st_kv = self.kv_state[0].numel() if cfg.compress_ratio and own_state else 0
        self.st_i = self.i_kv_state[0].numel() if cfg.indexed and own_state else 0
        self.st_ic = self.i_cache[0].numel() if cfg.indexed and own_state else 0  # bytes per sample
        # trivial block table (block b is b); ATOM passes its own to forward()
        self.k_pb = BLOCK_TOKENS // cfg.compress_ratio if cfg.compress_ratio else 1
        n_blocks = max(1, -(-cfg.n_compressed // self.k_pb)) if cfg.compress_ratio else 1
        self.block_tables = torch.arange(n_blocks, dtype=torch.int32, device=dev).repeat(samples, 1)
        self.env_rows = self.k_pb
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        self.step = variant.step

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False) -> torch.Tensor:
        """A scratch mailbox's values (``bf2``: two bf16 per value word)."""
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
        dest_rows,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
        tokens=None,
        block_tables=None,
        env_rows=None,
        state=None,
    ):
        """One layer; the epoch tag is ``step * MAX_LAYERS_PER_STEP + layer + 1``. Graph-capturable."""
        if not 0 <= layer < MAX_LAYERS_PER_STEP:
            raise ValueError(f"layer must be in [0, {MAX_LAYERS_PER_STEP}), got {layer}")
        cfg = self.W.cfg
        # offsets derive from self.S: a wrong leading dim reads out of bounds
        hshape = (self.S, cfg.hidden) if cfg.hc_mult == 1 else (self.S, cfg.hc_mult, cfg.hidden)
        if tuple(h.shape) != hshape or h.dtype != torch.bfloat16 or not h.is_contiguous():
            raise ValueError(f"h must be contiguous bfloat16 {hshape}, got {tuple(h.shape)} {h.dtype}")
        if cfg.kv_fp8:
            # ATOM's fp8 layout: (NoPE plane uint8 [rows, 512], RoPE plane bf16 [rows, rope_dim])
            kv_nope, kv_rope = kv_cache
            if kv_nope.dtype != torch.uint8 or kv_nope.shape[-1] != 512 or kv_rope.shape[-1] != cfg.rope_dim:
                raise ValueError(
                    f"kv_fp8 wants (uint8 [rows, 512], bf16 [rows, {cfg.rope_dim}]), "
                    f"got {kv_nope.dtype} {tuple(kv_nope.shape)}, {tuple(kv_rope.shape)}"
                )
        else:
            kv_nope, kv_rope = kv_cache, None
            if kv_cache.ndim != 2 or kv_cache.shape[1] != cfg.head_dim:
                raise ValueError(
                    f"kv_cache must be one plane [rows, {cfg.head_dim}], got {tuple(kv_cache.shape)} -- "
                    "which rows a sequence owns is the caller's, supplied through indices and dest_rows"
                )
        _check_i32("dest_rows", dest_rows, (2, self.S))
        _check_i32("indices", indices, (self.S, cfg.n_keys))
        # one position per sample: the samples are independent sequences
        _check_i32("cur_pos", cur_pos, (self.S,))
        if block_tables is not None and (
            block_tables.dtype != torch.int32
            or block_tables.ndim != 2
            or block_tables.shape[0] != self.S
            or block_tables.stride(1) != 1
        ):
            raise ValueError(
                f"block_tables must be int32 [{self.S}, blocks] with unit-stride rows, "
                f"got {tuple(block_tables.shape)} {block_tables.dtype} strides {block_tables.stride()}"
            )
        t = dict(self.W.t, **self.packed)
        bt = self.block_tables if block_tables is None else block_tables
        # `state` overrides any of these with a serving engine's views, slots and strides
        st = {
            k: getattr(self, k)
            for k in (
                "kv_state",
                "score_state",
                "i_kv_state",
                "i_score_state",
                "i_cache",
                "i_cache_s",
                "state_slots",
                "st_kv",
                "st_i",
                "st_ic",
            )
        }
        if state:
            unknown = set(state) - set(st)
            if unknown:
                raise ValueError(f"unknown state keys {sorted(unknown)}")
            st.update(state)
        if not self.own_state:
            need = {"kv_state", "score_state", "state_slots", "st_kv"} if cfg.compress_ratio else set()
            if cfg.indexed:
                need |= {"i_kv_state", "i_score_state", "i_cache", "i_cache_s", "st_i", "st_ic"}
            missing = need - set(state or ())
            if missing:
                raise ValueError(f"own_state=False: forward needs state {sorted(missing)}")
        use_hash = "tid2eid" in t
        if use_hash:
            if tokens is None:
                raise ValueError(f"a hash-routed layer needs tokens: int32, one per sample ({self.S})")
            _check_i32("tokens", tokens, (self.S,))
        if x_out is None:
            x_out = torch.empty(*hshape, dtype=torch.bfloat16, device=h.device)
        elif tuple(x_out.shape) != hshape or x_out.dtype != torch.bfloat16 or not x_out.is_contiguous():
            raise ValueError(f"x_out must be contiguous bfloat16 {hshape}, got {tuple(x_out.shape)} {x_out.dtype}")
        p = lambda x: x.data_ptr()  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_nope),
            p(kv_rope) if kv_rope is not None else 0,
            p(dest_rows),
            p(indices),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["attn_sink"]),
            p(t["ape"]) if "ape" in t else 0,
            p(t["g_ckv"]) if "g_ckv" in t else 0,
            p(st["kv_state"]),
            p(st["score_state"]),
            p(t["i_ape"]) if "i_ape" in t else 0,
            p(t["g_ickv"]) if "g_ickv" in t else 0,
            p(st["i_kv_state"]),
            p(st["i_score_state"]),
            p(st["i_cache"]),
            p(t["hc_attn_fn"]) if "hc_attn_fn" in t else 0,
            p(self.hc_sb["attn"]),
            p(t["hc_ffn_fn"]) if "hc_ffn_fn" in t else 0,
            p(self.hc_sb["ffn"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            p(t["w_qkv_c"]) if "w_qkv_c" in t else 0,
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            p(t["w_i_q_b"]) if "w_i_q_b" in t else 0,
            p(t["s_i_q_b"]) if "s_i_q_b" in t else 0,
            p(t["i_w"]) if "i_w" in t else 0,
            p(t["w_o_a"]),
            p(t["s_o_a"]),
            p(t["w_o_b"]),
            p(t["s_o_b"]),
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
            p(t["w_sug"]) if "w_sug" in t else 0,
            p(t["s_sug"]) if "s_sug" in t else 0,
            p(t["w_sdn"]) if "w_sdn" in t else 0,
            p(t["s_sdn"]) if "s_sdn" in t else 0,
            p(self.scratch),
            self.sym,
            p(self.peers),
            0 if self.timeline is None else p(self.timeline),
            p(self.step),
            p(self.variant.hang),
            p(st["state_slots"]),
            p(tokens) if use_hash else 0,
            p(t["tid2eid"]) if use_hash else 0,
            p(bt),
            p(st["i_cache_s"]),
            self.rank,
            layer,
            st["st_kv"],
            st["st_i"],
            st["st_ic"],
            int(use_hash),
            bt.stride(0),
            self.env_rows if env_rows is None else env_rows,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self):
        self.variant.advance_step()

    def close(self):
        """Close the variant's IPC mappings if this layer created it; a shared variant is the caller's to close."""
        if self._owns_variant:
            self.variant.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def timeline_report(self) -> str:
        """Per stage, in us from launch start: span and median per-task phases."""
        if self.timeline is None:
            raise RuntimeError("timeline collection was not enabled")
        tl = self.timeline[:, :5].cpu().double() / 100.0  # s_memrealtime ticks at 100 MHz
        t0 = tl[:, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            if not n:
                continue
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
        cfg = self.W.cfg
        S, H = self.S, cfg.heads
        mid = self.debug("mid", (S, cfg.top_k + 1, cfg.inter))
        if moe_format(self.moe_mode).activation is ExpertActivation.BF16:
            mid = mid.to(torch.bfloat16).float()
        return dict(
            q_a=self.debug("q_a", (S, cfg.q_lora)),
            kv=self.debug("kv_a", (S, cfg.head_dim)),
            q=self.debug("q", (S, H, cfg.head_dim), bf2=True),
            o=self.debug("o", (S, H, cfg.head_dim), bf2=True),
            o_lora=self.debug("o_lora", (S, cfg.o_groups * cfg.o_lora), bf2=True),
            a=self.debug(
                "a",
                (S, cfg.hidden) if cfg.hc_mult == 1 else (S, cfg.hc_mult, cfg.hidden),
                bf2=True,
            ).to(torch.bfloat16),
            scores=self.debug("scores", (S, cfg.n_experts)),
            sel=self.debug("sel", (S, cfg.top_k + 1), torch.int32),
            prob=self.debug("prob", (S, cfg.top_k + 1)),
            mid=mid,
            xq=self.debug("xqd", (S, cfg.hidden), pairs=False),
        )
