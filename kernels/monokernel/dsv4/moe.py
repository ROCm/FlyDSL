# SPDX-License-Identifier: Apache-2.0
"""Owned, graph-replay-safe DSV4-Pro A8W4 MoE runtime."""

import torch

from kernels.monokernel.packing import pack_bf16
from kernels.monokernel.runtime import SymmetricPeerBuffer

from .config import Dsv4Config, validate_shape
from .moe_kernel import BLOCKS, build_dsv4_moe, scratch_layout
from .packing import pack_experts, pack_shared_fp8


class PreparedMoeWeights:
    """One immutable packed weight set shared by all sequence buckets.

    ``shared=None`` means the synthetic/all-FP4 shared-expert contract.
    Native Pro passes (up, up_scale, down, down_scale) for its FP8 shared
    expert. Scales remain raw E8M0 bytes in [N/128,K/128] order.
    """

    def __init__(
        self,
        router,
        bias,
        up,
        up_scale,
        down,
        down_scale,
        *,
        tp=1,
        shared=None,
        hash_table=None,
        config=Dsv4Config(),
        borrow_aiter_experts=None,
    ):
        config.validate_pro()
        config.validate_moe(1, tp)
        h, inter, experts = config.hidden, config.intermediate // tp, config.experts + (shared is None)
        expected = {
            "router": (router, (config.experts, h), torch.bfloat16),
            "bias": (bias, (config.experts,), torch.float32),
            "up": (up, (experts, inter * 2, h // 2), torch.uint8),
            "up_scale": (up_scale, (experts, inter * 2, h // 32), torch.uint8),
            "down": (down, (experts, h, inter // 2), torch.uint8),
            "down_scale": (down_scale, (experts, h, inter // 32), torch.uint8),
        }
        for name, (tensor, shape, dtype) in expected.items():
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
            if not tensor.is_cuda or tensor.device != router.device:
                raise ValueError("all DSV4 weights must be on the same gfx950 GPU")
        arch = torch.cuda.get_device_properties(router.device).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise ValueError(f"DSV4 A8W4 MonoKernel requires gfx950, got {arch}")
        self.config, self.tp, self.device = config, tp, router.device
        self.hash_table = hash_table
        if hash_table is not None:
            if (
                hash_table.ndim != 2
                or hash_table.shape[1] != config.top_k
                or hash_table.shape[0] < 1
                or hash_table.dtype != torch.int32
                or hash_table.device != self.device
                or not hash_table.is_contiguous()
            ):
                raise ValueError("hash table must be device contiguous int32 [vocab,top_k]")
            if ((hash_table < 0) | (hash_table >= config.experts)).any():
                raise ValueError("hash table contains invalid expert ids")
        self.shared_fp8 = shared is not None
        self.aiter_experts = borrow_aiter_experts is not None
        if self.aiter_experts:
            # Keep the Parameter objects, not views of their current storage:
            # ATOM's child post-load hook replaces .data with shuffled bytes.
            # This avoids duplicating the full routed weight set (roughly
            # 200 GB per rank for Pro TP4). Only the compact raw scales remain.
            packed_up, packed_down = borrow_aiter_experts
            for borrowed, raw in ((packed_up, up), (packed_down, down)):
                if borrowed.data_ptr() != raw.data_ptr() or borrowed.shape != raw.shape:
                    raise ValueError("borrowed AITER Parameters must own the supplied expert weights")
        else:
            packed_up, packed_down = pack_experts(up), pack_experts(down)
        self.weights = (pack_bf16(router), bias, packed_up, up_scale, packed_down, down_scale)
        self.shared = ()
        if shared is not None:
            su, sus, sd, sds = shared
            for name, tensor, shape, dtype in (
                ("shared_up", su, (2 * inter, h), torch.float8_e4m3fn),
                ("shared_up_scale", sus, (2 * inter // 128, h // 128), torch.uint8),
                ("shared_down", sd, (h, inter), torch.float8_e4m3fn),
                ("shared_down_scale", sds, (h // 128, inter // 128), torch.uint8),
            ):
                if (
                    tuple(tensor.shape) != shape
                    or tensor.dtype != dtype
                    or not tensor.is_contiguous()
                    or tensor.device != self.device
                ):
                    raise ValueError(f"{name} must be contiguous {dtype} {shape} on {self.device}")
            self.shared = (pack_shared_fp8(su), sus, pack_shared_fp8(sd), sds)


class Dsv4MoeMonoKernel:
    """One MoE launch including routing, both quantizers and TP reduction.

    Each instance owns its scratch/epochs. Use one instance per stream;
    simultaneous launches sharing an instance are not supported. Hash-layer
    ids are dynamic [seq,6] tensors and may change on every graph replay.
    """

    def __init__(
        self,
        router=None,
        bias=None,
        up=None,
        up_scale=None,
        down=None,
        down_scale=None,
        *,
        seq_len,
        batch_size=1,
        tp=1,
        rank=0,
        group=None,
        hash_routing=False,
        config=Dsv4Config(),
        shared=None,
        prepared=None,
    ):
        validate_shape(batch_size, seq_len)
        config.validate_pro()
        config.validate_moe(seq_len, tp)
        if not 0 <= rank < tp:
            raise ValueError("rank is outside the TP group")
        if prepared is None:
            prepared = PreparedMoeWeights(
                router, bias, up, up_scale, down, down_scale, tp=tp, shared=shared, config=config
            )
        elif prepared.config != config or prepared.tp != tp:
            raise ValueError("prepared weights have a different config or TP size")
        self.prepared = prepared
        self._closed = False
        self.config, self.seq_len, self.tp, self.rank = config, seq_len, tp, rank
        self.hash_routing, self.shared_fp8 = hash_routing, prepared.shared_fp8
        self.device, self.weights = prepared.device, prepared.weights
        h = config.hidden
        self.layout = scratch_layout(config, seq_len, tp, self.shared_fp8)
        self.scratch = torch.zeros(self.layout["_bytes"], dtype=torch.uint8, device=self.device)
        self.epochs = torch.zeros(BLOCKS, dtype=torch.int32, device=self.device)
        with torch.cuda.device(self.device):
            self.peer = SymmetricPeerBuffer(2 * tp * seq_len * h // 2 * 8, rank, tp, group)
        self.hash_vocab = prepared.hash_table.shape[0] if hash_routing and prepared.hash_table is not None else 0
        self.launch = build_dsv4_moe(
            config,
            seq_len,
            tp,
            hash_routing,
            shared_fp8=self.shared_fp8,
            hash_vocab=self.hash_vocab,
            aiter_experts=prepared.aiter_experts,
        )
        self.staged = [
            build_dsv4_moe(
                config, seq_len, tp, hash_routing, stage, self.shared_fp8, self.hash_vocab, prepared.aiter_experts
            )
            for stage in ("router", "route", "quant", "up", "down")
        ]

    def __call__(self, x, *, hash_ids=None, token_ids=None, out=None, unfused=False):
        if self._closed:
            raise RuntimeError("DSV4 MoE runtime has been closed")
        if x.shape != (self.seq_len, self.config.hidden) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError(f"input must be contiguous BF16 [{self.seq_len},{self.config.hidden}]")
        if x.device != self.device:
            raise ValueError("input and weights must be on the same device")
        if self.hash_vocab:
            if (
                token_ids is None
                or token_ids.shape != (self.seq_len,)
                or token_ids.dtype != torch.int64
                or token_ids.device != self.device
                or not token_ids.is_contiguous()
                or hash_ids is not None
            ):
                raise ValueError("table routing needs contiguous device int64 token_ids [seq_len]")
        elif token_ids is not None:
            raise ValueError("token_ids require prepared hash-table routing")
        elif self.hash_routing:
            if (
                hash_ids is None
                or hash_ids.shape != (self.seq_len, self.config.top_k)
                or hash_ids.dtype != torch.int32
                or not hash_ids.is_contiguous()
                or hash_ids.device != x.device
            ):
                raise ValueError("hash routing needs contiguous device int32 [seq_len,top_k] ids")
        elif hash_ids is not None:
            raise ValueError("hash ids were supplied to a bias-routed layer")
        if out is None:
            out = torch.empty_like(x)
        if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
            raise ValueError("out must have the same shape, dtype, device and contiguity as input")
        if torch._C._overlaps(x, out):
            raise ValueError("out must not alias the input read by resident CTAs")
        router, bias, up, us, down, ds = self.weights
        if self.prepared.aiter_experts and not (
            getattr(up, "is_shuffled", False) and getattr(down, "is_shuffled", False)
        ):
            raise RuntimeError("borrowed AITER expert weights must finish their post-load shuffle before launch")
        launches = [self.launch]
        if unfused:
            launches = self.staged
        shared_ptrs = [tensor.data_ptr() for tensor in self.prepared.shared] if self.shared_fp8 else [0] * 4
        routing_input = token_ids if self.hash_vocab else hash_ids
        for launch in launches:
            launch(
                x.data_ptr(),
                router.data_ptr(),
                bias.data_ptr(),
                0 if routing_input is None else routing_input.data_ptr(),
                self.prepared.hash_table.data_ptr() if self.hash_vocab else 0,
                up.data_ptr(),
                us.data_ptr(),
                down.data_ptr(),
                ds.data_ptr(),
                *shared_ptrs,
                out.data_ptr(),
                self.scratch.data_ptr(),
                self.epochs.data_ptr(),
                self.peer.local_address,
                self.peer.addresses.data_ptr(),
                self.rank,
                torch.cuda.current_stream(self.device),
            )
        return out

    def routing(self):
        shape = (self.seq_len, self.config.top_k + 1)
        n = shape[0] * shape[1]

        def read(name, dtype):
            start = self.layout[name]
            return (
                self.scratch[start : start + n * 8]
                .view(torch.int32)
                .view(n, 2)[:, 0]
                .contiguous()
                .view(dtype)
                .view(shape)
            )

        return read("ids", torch.int32)[:, :-1], read("probs", torch.float32)[:, :-1]

    def close(self):
        self.peer.close()
        self._closed = True
