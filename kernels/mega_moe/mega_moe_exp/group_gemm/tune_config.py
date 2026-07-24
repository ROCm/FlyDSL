"""Online MegaMoE stage1 autotune search space."""

from flydsl.autotune import Config

_SHAPES = (
    (32, 128, 2),
    (32, 128, 4),
    (32, 256, 2),
    (32, 256, 4),
    (32, 256, 8),
    (32, 512, 4),
    (32, 512, 8),
    (64, 128, 2),
    (64, 128, 4),
    (64, 256, 4),
    (64, 256, 8),
    (64, 512, 8),
    (128, 256, 4),
    (128, 256, 8),
    (128, 512, 8),
)
_ANCHORS = {
    (32, 128, 4),
    (32, 256, 4),
    (64, 256, 4),
    (128, 256, 4),
    (128, 512, 8),
}
_GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16)
_DISPATCH_CU_VALUES = (8, 16, 24, 32, 48, 64, 96, 128)
_CALIBRATED_VARIANTS = {
    (128, 512, 8): (
        {
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
    ),
}


def _candidate_variants(shape):
    variants = [{}]
    if shape not in _ANCHORS:
        return variants
    variants += [{"grid_mult": value} for value in _GRID_MULT_VALUES if value != 4]
    variants += [{"num_dispatch_cu": value} for value in _DISPATCH_CU_VALUES if value != 64]
    variants += [{"wgm": value} for value in (1, 4, 8)]
    variants += [
        {"sched_nmajor": True},
        {"pipe_weights": False, "mfma_amajor": False},
        {"mfma_amajor": False},
        {"swizzle_a": False},
        {"tune_use_xcd": False, "wgm": 1},
        {"use_tile_resource": False},
        {"waves_per_eu_hint": 1},
    ]
    variants += list(_CALIBRATED_VARIANTS.get(shape, ()))
    return variants


def get_stage1_autotune_configs(dispatch_cu=None, grid_mult=None, tile_m_values=(32,)):
    tile_m_values = {int(value) for value in tile_m_values}
    configs = []
    seen = set()
    for sort_block_m, tile_n, num_waves in _SHAPES:
        if sort_block_m not in tile_m_values:
            continue
        base = dict(
            sort_block_m=sort_block_m,
            tile_n=tile_n,
            tile_k=256,
            num_waves=num_waves,
            wgm=2,
            grid_mult=4,
            sched_nmajor=False,
            pipe_weights=True,
            mfma_amajor=True,
            swizzle_a=True,
            num_dispatch_cu=64,
            tune_use_xcd=True,
            use_tile_resource=True,
            waves_per_eu_hint=2,
        )
        for update in _candidate_variants((sort_block_m, tile_n, num_waves)):
            values = dict(base, **update)
            if dispatch_cu is not None:
                values["num_dispatch_cu"] = int(dispatch_cu)
            if grid_mult is not None:
                values["grid_mult"] = int(grid_mult)
            signature = tuple(sorted(values.items()))
            if signature not in seen:
                configs.append(Config(**values))
                seen.add(signature)
    return configs


def prune_stage1_autotune_configs(configs, sig_args):
    """Prune invalid and batch-irrelevant configs before collective compilation."""
    tokens = int(sig_args["tune_tokens"])
    model_dim = int(sig_args["model_dim"])
    inter_dim = int(sig_args["inter_dim"])
    num_cu = int(sig_args["num_cu"])
    fuse_npes = int(sig_args.get("fuse_npes", 0))
    fuse_topk = int(sig_args.get("fuse_topk", 0))
    fuse_mtpr = int(sig_args.get("fuse_mtpr", 0))
    experts_per_rank = int(sig_args.get("experts_per_rank", 0))
    out = []
    for config in configs:
        values = config.kwargs
        block_m = int(values["sort_block_m"])
        tile_n = int(values["tile_n"])
        tile_k = int(values["tile_k"])
        num_waves = int(values["num_waves"])
        grid_mult = int(values["grid_mult"])
        dispatch_cu = int(values["num_dispatch_cu"])
        use_tile_resource = bool(values["use_tile_resource"])
        num_acc_n = tile_n // num_waves // 16
        m_repeat = block_m // 16
        lds_pool = max(2 * block_m * tile_k, 2 * block_m * tile_n)
        lds_scale = block_m * (model_dim // 32)
        payload_bytes = (
            (fuse_npes * fuse_mtpr * fuse_topk + experts_per_rank * block_m) * model_dim
            if fuse_npes and fuse_topk and fuse_mtpr and experts_per_rank
            else 0
        )
        if (
            model_dim % tile_k
            or (2 * inter_dim) % tile_n
            or tile_n % (num_waves * 32)
            or block_m % 32
            or m_repeat * num_acc_n * 4 > 256
            or lds_pool + lds_scale > 160 * 1024
            or not 0 < dispatch_cu <= num_cu
            or (payload_bytes >= 1 << 32 and not use_tile_resource)
        ):
            continue
        if tokens <= 64:
            keep = block_m == 32 and tile_n <= 256 and num_waves <= 4 and grid_mult >= 3 and dispatch_cu >= 32
        elif tokens <= 1024:
            keep = block_m <= 64 and tile_n <= 512 and grid_mult <= 8 and dispatch_cu >= 24
        else:
            keep = block_m >= 64 and tile_n >= 256 and grid_mult <= 6 and dispatch_cu <= 96
        if keep:
            out.append(config)
    if not out:
        raise ValueError(f"no valid stage1 configs for tokens={tokens}")
    return out
