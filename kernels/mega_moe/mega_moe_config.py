# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Workload-derived MegaMoEV2 configuration rules for MI355X.

The selector intentionally separates invariant hardware constraints from the
few measured crossover thresholds.  Configurations are derived from dispatch
mode, route density, matrix shape, and CU budget; token buckets only quantize a
continuous runtime input for compilation-cache reuse.
"""

from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import cache

TOKEN_BUCKETS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
A8W4_DECODE_MTPRS = (1, 4, 8, 16, 32, 64, 128, 256, 512)
P2P_FP8_MIN_MTPR = 1024
FIXED_SLOT_MAX_MTPR = 255
A8W4SMOOTH_FIXED_SLOT_MAX_MTPR = 512
A8W4SMOOTH_DECODE_MTPRS = (1, 4, 8, 16, 32, 64, 128, 256, 512)
MAX_MTPR_CLASS = 32768
REFERENCE_EXPERTS_PER_RANK = 48
EXPERT_CONFIG_GRANULARITY = 64
REFERENCE_WORLD_SIZE = 8
REFERENCE_TOPK = 8
REFERENCE_NUM_CU = 256


@dataclass(frozen=True, slots=True)
class Stage1Config:
    sort_block_m: int
    tile_n: int
    num_waves: int
    grid_mult: int
    num_dispatch_cu: int
    mfma_amajor: bool
    async_a_copy: bool
    use_tile_resource: bool
    b_nt: int
    waves_per_eu_hint: int = 2
    tile_k: int = 256
    pipe_weights: bool = True
    swizzle_a: bool = True
    work_shards: int = 8
    external_grouping: bool = False
    external_counting: bool = False
    payload_chunk_rows: int = 0
    payload_tile_ready: bool = False


@dataclass(frozen=True, slots=True)
class Stage2Config:
    block_m: int
    block_n: int
    persist: bool
    persist_cu: int
    use_nt: bool
    persist_strided: bool = False
    skew_cu: int = 0
    block_k: int = 256
    b_hoist: bool = True
    ascale_prefetch: bool = True
    spatial_partition: int = 402
    bf16_lds: bool = False


@dataclass(frozen=True, slots=True)
class MegaMoEConfig:
    stage1: Stage1Config
    stage2: Stage2Config
    p2p_quant: str

    def __post_init__(self):
        sbm = self.stage1.sort_block_m
        bm = self.stage2.block_m
        if bm > sbm or sbm % bm:
            raise ValueError(f"Stage2 block_m={bm} must divide Stage1 sort_block_m={sbm}")
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != "none" and self.stage2.bf16_lds:
            raise ValueError("FP8 P2P requires Stage2 bf16_lds=False")


def nearest_token_bucket(tokens: int) -> int:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    index = bisect_left(TOKEN_BUCKETS, tokens)
    if index == 0:
        return TOKEN_BUCKETS[0]
    if index == len(TOKEN_BUCKETS):
        return TOKEN_BUCKETS[-1]
    lower, upper = TOKEN_BUCKETS[index - 1], TOKEN_BUCKETS[index]
    return upper if upper - tokens <= tokens - lower else lower


def mtpr_config_class(mtpr: int) -> int:
    return mtpr if mtpr <= P2P_FP8_MIN_MTPR else MAX_MTPR_CLASS


def expert_config_class(experts_per_rank: int) -> int:
    return (experts_per_rank + EXPERT_CONFIG_GRANULARITY - 1) // EXPERT_CONFIG_GRANULARITY * EXPERT_CONFIG_GRANULARITY


def fixed_slot_max_mtpr(
    quant_mode: str,
    experts_per_rank: int,
    model_dim: int,
    inter_dim: int,
    world_size: int = REFERENCE_WORLD_SIZE,
    topk: int = REFERENCE_TOPK,
) -> int:
    """Return the production fixed-slot limit for each quantization mode."""
    if quant_mode == "w8a8smooth":
        return 0
    if quant_mode == "a8w4smooth":
        return A8W4SMOOTH_FIXED_SLOT_MAX_MTPR
    return FIXED_SLOT_MAX_MTPR


@dataclass(frozen=True, slots=True)
class _Workload:
    bucket: int
    mtpr_class: int
    experts_per_rank: int
    model_dim: int
    inter_dim: int
    quant_mode: str
    world_size: int
    topk: int
    num_cu: int
    fixed_slot_limit: int

    @property
    def large_capacity(self) -> bool:
        return self.mtpr_class == MAX_MTPR_CLASS

    @property
    def fixed_slot(self) -> bool:
        return not self.large_capacity and self.mtpr_class <= self.fixed_slot_limit

    @property
    def int_smooth(self) -> bool:
        return self.quant_mode in ("a8w4smooth", "w8a8smooth")

    @property
    def routes_per_rank(self) -> int:
        # With E = world_size * experts_per_rank, world_size cancels from the
        # expected local receive count; this is why config matching need not be
        # keyed by EP size.
        return self.bucket * self.topk

    @property
    def rows_per_expert(self) -> float:
        return self.routes_per_rank / self.experts_per_rank

    @property
    def stage1_tile_n(self) -> int:
        return 512 if self.inter_dim >= 2048 else 256


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _fit_dispatch_cu(requested: int, workload: _Workload) -> int:
    expert_waves = (workload.experts_per_rank + 63) // 64
    reserve = max(32, workload.world_size)
    limit = _align_down(workload.num_cu - reserve, workload.world_size)
    scaled = requested * expert_waves
    return max(
        workload.world_size,
        _align_down(min(limit, scaled), workload.world_size),
    )


def _fixed_dispatch_request(bucket: int) -> int:
    """CTA budget for fixed slots before expert-wave scaling."""
    if bucket <= 1:
        return 64
    if bucket <= 8:
        return 128
    if bucket <= 16:
        return 96
    if bucket <= 32:
        return 128
    # Route-copy parallelism grows logarithmically once each producer owns
    # several routes; leave at least one 32-CU compute wave resident.
    return min(224, 16 * (bucket.bit_length() + 7))


def _bounded_dispatch_request(bucket: int) -> int:
    if bucket <= 1:
        return 224
    if bucket <= 4:
        return 128
    if bucket <= 8:
        return 192
    if bucket <= 16:
        return 64
    if bucket <= 32:
        return 128
    if bucket <= 64:
        return 192
    return 128


def _large_dispatch_request(bucket: int) -> int:
    """Payload producer budget across increasing route-density regimes."""
    if bucket <= 1:
        return 224
    if bucket <= 4:
        return 128
    if bucket <= 8:
        return 192
    if bucket <= 32:
        return 64
    if bucket <= 64:
        return 160
    if bucket <= 128:
        return 192
    if bucket <= 256:
        return 160
    if bucket == 8192:
        return 96
    if bucket >= 16384:
        return 32
    return 64


def _derive_stage1(workload: _Workload) -> Stage1Config:
    bucket = workload.bucket
    if workload.fixed_slot:
        grid_mult = max(1, bucket // 4) if bucket <= 16 else 3
        dispatch_request = _fixed_dispatch_request(bucket)
        if workload.quant_mode == "a8w4":
            # Match the validated production MX-scale A8W4 crossover points.
            # SmoothQuant modes retain their shape-specific overrides below.
            if bucket == 1:
                dispatch_request = 160
            elif bucket == 8:
                dispatch_request = 32
        return Stage1Config(
            sort_block_m=32,
            tile_n=256 if bucket <= 8 else 128,
            num_waves=4,
            grid_mult=grid_mult,
            num_dispatch_cu=_fit_dispatch_cu(dispatch_request, workload),
            mfma_amajor=False,
            async_a_copy=False,
            use_tile_resource=bucket <= 16,
            b_nt=0 if bucket == 1 else 3,
            waves_per_eu_hint=1 if bucket == 16 else 2,
        )

    if bucket <= 4:
        sort_block_m, tile_n, num_waves = 32, 256, 4
        mfma_amajor, async_a_copy = False, False
    elif bucket <= 128:
        sort_block_m, tile_n, num_waves = 32, workload.stage1_tile_n, 8
        mfma_amajor, async_a_copy = True, True
    elif bucket <= 2048:
        sort_block_m, tile_n, num_waves = 64, workload.stage1_tile_n, 8
        mfma_amajor, async_a_copy = True, True
    else:
        sort_block_m, tile_n, num_waves = 128, workload.stage1_tile_n, 8
        mfma_amajor, async_a_copy = True, True

    if workload.large_capacity:
        work_shards = 1 if bucket <= 32 else 4
        if bucket == 2048:
            work_shards = 8
        return Stage1Config(
            sort_block_m=sort_block_m,
            tile_n=tile_n,
            num_waves=num_waves,
            grid_mult=1,
            num_dispatch_cu=_fit_dispatch_cu(_large_dispatch_request(bucket), workload),
            mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy,
            use_tile_resource=True,
            b_nt=3 if 1 < bucket <= 256 else 0,
            work_shards=work_shards,
            external_grouping=bucket == 4 or bucket >= 256,
            external_counting=bucket >= 256,
            payload_chunk_rows=384,
            payload_tile_ready=True,
        )

    if bucket > 1024:
        raise ValueError(f"bounded MTPR does not support token bucket {bucket}")
    grid_mult = 1 if bucket <= 256 else 2
    dispatch_request = _bounded_dispatch_request(bucket) if bucket <= 128 else 160 if bucket == 256 else 128
    if workload.quant_mode == "a8w4" and bucket == 512:
        dispatch_request = 64
    use_tile_resource = bucket == 256
    b_nt = 0 if bucket == 1 or bucket >= 1024 else 3
    if workload.mtpr_class > bucket:
        if bucket == 32:
            dispatch_request = 64
        elif bucket == 64:
            dispatch_request = 160
        elif bucket == 128:
            dispatch_request = 192
        elif bucket == 512:
            grid_mult, dispatch_request, use_tile_resource, b_nt = 1, 64, True, 0
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=grid_mult,
        num_dispatch_cu=_fit_dispatch_cu(dispatch_request, workload),
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=use_tile_resource,
        b_nt=b_nt,
    )


def _large_persist_cu(bucket: int) -> int:
    # These are MI355X residency crossover points rather than model IDs.
    if bucket == 1024:
        return 224
    if bucket == 2048:
        return 256
    if bucket == 16384:
        return 192
    return 240


def _derive_stage2(workload: _Workload, stage1: Stage1Config) -> Stage2Config:
    bucket = workload.bucket
    if workload.large_capacity:
        return Stage2Config(
            block_m=64 if stage1.sort_block_m == 128 else 32,
            block_n=128 if bucket == 256 or workload.model_dim < 4096 else 256,
            persist=True,
            persist_cu=_large_persist_cu(bucket),
            use_nt=bucket <= 128,
            persist_strided=512 <= bucket <= 2048,
            skew_cu=96 if bucket >= 512 else 0,
        )

    if not workload.fixed_slot and workload.mtpr_class > bucket:
        return Stage2Config(
            block_m=64 if stage1.sort_block_m == 128 else 32,
            block_n=128 if bucket == 256 and stage1.sort_block_m == 64 else 256,
            persist=True,
            persist_cu=240,
            use_nt=bucket <= 128,
            persist_strided=512 <= bucket <= 2048,
        )

    block_n = 256 if bucket in (1, 4, 64) or bucket >= 1024 or (not workload.fixed_slot and bucket < 128) else 128
    if workload.model_dim < 4096:
        block_n = 128
    persist = bucket >= 128
    return Stage2Config(
        block_m=64 if bucket >= 4096 else 32,
        block_n=block_n,
        persist=persist,
        persist_cu=128 if bucket == 256 else 240 if persist else 0,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
    )


def _narrow_int_dispatch_request(routes: int) -> int:
    """Measured MI355X sparse-route crossover tiers for INT8 payload copy."""
    if routes <= 128:
        return 96
    if routes <= 256:
        return 192
    if routes <= 512:
        return 128
    return 192


def _apply_narrow_int_fixed_rules(
    workload: _Workload,
    stage1: Stage1Config,
    stage2: Stage2Config,
) -> tuple[Stage1Config, Stage2Config]:
    """Derive the tuned narrow A8W4Smooth fixed-slot configuration."""
    bucket = workload.bucket
    routes = workload.routes_per_rank
    if bucket == 8:
        stage1 = replace(
            stage1,
            sort_block_m=32,
            tile_n=256,
            num_waves=8,
            grid_mult=1,
            num_dispatch_cu=_fit_dispatch_cu(192, workload),
            b_nt=0,
            waves_per_eu_hint=2,
            swizzle_a=False,
        )
    elif bucket <= 128 and bucket >= 16:
        dispatch_request = _narrow_int_dispatch_request(routes)
        low_pressure = dispatch_request <= 128
        stage1 = replace(
            stage1,
            sort_block_m=32,
            tile_n=128,
            num_waves=4,
            grid_mult=2,
            num_dispatch_cu=_fit_dispatch_cu(dispatch_request, workload),
            b_nt=0,
            waves_per_eu_hint=1 if low_pressure else 2,
            swizzle_a=low_pressure,
        )
    elif 128 < bucket <= A8W4SMOOTH_FIXED_SLOT_MAX_MTPR:
        # At 256 the expected expert density leaves one M tile/expert and needs
        # a deep consumer grid; at 512 it becomes two tiles/expert, so one grid
        # wave plus producer retirement supplies enough GEMM workers.
        one_tile_density = workload.rows_per_expert < 48
        consumer_reserve = 32 if one_tile_density else 48
        stage1 = replace(
            stage1,
            sort_block_m=64,
            tile_n=256,
            num_waves=8,
            grid_mult=3 if one_tile_density else 1,
            num_dispatch_cu=_fit_dispatch_cu(workload.num_cu - consumer_reserve, workload),
            b_nt=0,
            waves_per_eu_hint=2 if one_tile_density else 1,
            swizzle_a=True,
        )

    if 128 <= bucket <= A8W4SMOOTH_FIXED_SLOT_MAX_MTPR:
        # One Stage2 CTA consumes roughly 16 routes in this shape class.  Keep
        # a 96-CU latency floor and reserve 16 CUs at saturation for combine.
        persist_cu = min(
            workload.num_cu - 16,
            max(96, routes // 16),
        )
        stage2 = replace(
            stage2,
            persist=True,
            persist_cu=persist_cu,
            persist_strided=False,
            skew_cu=0,
            use_nt=routes <= 2048,
        )
    return stage1, stage2


def _apply_quant_and_shape_rules(
    workload: _Workload,
    stage1: Stage1Config,
    stage2: Stage2Config,
) -> tuple[Stage1Config, Stage2Config]:
    bucket = workload.bucket

    if workload.quant_mode == "a8w4smooth" and workload.fixed_slot and bucket <= 8:
        stage1 = replace(
            stage1,
            tile_n=128,
            grid_mult=1,
            num_dispatch_cu=_fit_dispatch_cu(64, workload),
            b_nt=0 if bucket <= 4 else 3,
        )
        stage2 = replace(stage2, use_nt=False)

    if workload.int_smooth and bucket == 128:
        stage2 = replace(stage2, block_m=32, block_n=128, block_k=256)

    if workload.quant_mode == "a8w4smooth":
        # INT Stage1 keeps separate gate/up B fragments. Carrying the next full
        # B step doubles live state, so retain only A-register/LDS ping-pong.
        stage1 = replace(stage1, pipe_weights=False, async_a_copy=False)
        stage1, stage2 = _apply_narrow_int_fixed_rules(workload, stage1, stage2)
    return stage1, stage2


@cache
def _select_bucket_config(
    bucket: int,
    mtpr_class: int,
    experts_per_rank: int,
    model_dim: int,
    inter_dim: int,
    quant_mode: str,
    world_size: int,
    topk: int,
    num_cu: int,
) -> MegaMoEConfig:
    fixed_slot_limit = fixed_slot_max_mtpr(
        quant_mode,
        experts_per_rank,
        model_dim,
        inter_dim,
        world_size,
        topk,
    )
    workload = _Workload(
        bucket=bucket,
        mtpr_class=mtpr_class,
        experts_per_rank=experts_per_rank,
        model_dim=model_dim,
        inter_dim=inter_dim,
        quant_mode=quant_mode,
        world_size=world_size,
        topk=topk,
        num_cu=num_cu,
        fixed_slot_limit=fixed_slot_limit,
    )
    stage1 = _derive_stage1(workload)
    stage2 = _derive_stage2(workload, stage1)
    stage1, stage2 = _apply_quant_and_shape_rules(workload, stage1, stage2)
    p2p_quant = "fp8_blockwise_1x32" if workload.large_capacity else "none"
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant=p2p_quant)


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
    quant_mode: str = "a8w4",
    dispatch_quant: str | None = None,
    world_size: int = REFERENCE_WORLD_SIZE,
    topk: int = REFERENCE_TOPK,
    num_cu: int = REFERENCE_NUM_CU,
) -> MegaMoEConfig:
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")
    if quant_mode not in ("a8w4", "a8w4smooth", "w8a8smooth"):
        raise ValueError(f"unsupported quant_mode={quant_mode!r}")
    if dispatch_quant not in (None, "mxfp4"):
        raise ValueError(f"unsupported dispatch_quant={dispatch_quant!r}")
    if dispatch_quant == "mxfp4" and quant_mode != "w8a8smooth":
        raise ValueError(
            "dispatch_quant='mxfp4' is only supported by w8a8smooth prefill"
        )
    if quant_mode == "a8w4":
        shape = (model_dim, inter_dim, experts_per_rank, world_size, topk)
        if shape != (3584, 1280, 48, 8, 8):
            raise ValueError(
                "native A8W4 is decode-only for M13 "
                "(D=3584, I=1280, EPR=48, EP=8, topk=8)"
            )
        if tokens != mtpr or mtpr not in A8W4_DECODE_MTPRS:
            raise ValueError(
                "native A8W4 requires tokens=MTPR in "
                f"{A8W4_DECODE_MTPRS}, got tokens={tokens}, MTPR={mtpr}"
            )
    elif quant_mode == "a8w4smooth":
        if tokens != mtpr or mtpr not in A8W4SMOOTH_DECODE_MTPRS:
            raise ValueError(
                "a8w4smooth decode requires tokens=MTPR in "
                "{1,4,8,16,32,64,128,256,512}"
            )
        if (
            experts_per_rank,
            model_dim,
            inter_dim,
            world_size,
            topk,
        ) != (48, 3584, 1280, 8, 8):
            raise ValueError("a8w4smooth decode is specialized for M13 EP8")
    if world_size <= 0 or topk <= 0:
        raise ValueError(f"invalid routing shape world_size={world_size}, topk={topk}")
    if num_cu <= max(32, world_size):
        raise ValueError(f"num_cu={num_cu} leaves no compute CU reserve")

    bucket = nearest_token_bucket(tokens)
    mtpr_class = mtpr_config_class(mtpr)
    fixed_slot_limit = fixed_slot_max_mtpr(
        quant_mode,
        experts_per_rank,
        model_dim,
        inter_dim,
        world_size,
        topk,
    )
    if mtpr_class <= fixed_slot_limit and bucket > fixed_slot_limit:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    if mtpr_class <= fixed_slot_limit and experts_per_rank > 64:
        raise ValueError("fixed-slot supports at most 64 experts per rank")
    return _select_bucket_config(
        bucket,
        mtpr_class,
        expert_config_class(experts_per_rank),
        model_dim,
        inter_dim,
        quant_mode,
        world_size,
        topk,
        num_cu,
    )
