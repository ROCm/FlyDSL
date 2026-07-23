"""Online MegaMoE stage1 autotune search space."""

from flydsl.autotune import Config


def get_stage1_autotune_configs(dispatch_cu=None, grid_mult=None):
    dispatch_values = (dispatch_cu,) if dispatch_cu is not None else (32, 64)
    grid_values = (grid_mult,) if grid_mult is not None else (2, 4, 8)
    compute_values = ((128, 1), (128, 2), (256, 1), (256, 2))
    configs = []
    for tile_n, wgm in compute_values:
        for gm in grid_values:
            for dc in dispatch_values:
                configs.append(
                    Config(
                        tile_n=tile_n,
                        tile_k=256,
                        num_waves=4,
                        wgm=wgm,
                        grid_mult=gm,
                        sched_nmajor=False,
                        mfma_amajor=True,
                        swizzle_a=True,
                        num_dispatch_cu=dc,
                    )
                )
    configs.append(
        Config(
            tile_n=256,
            tile_k=256,
            num_waves=4,
            wgm=4,
            grid_mult=grid_values[0],
            sched_nmajor=False,
            mfma_amajor=False,
            swizzle_a=False,
            num_dispatch_cu=dispatch_values[0],
        )
    )
    return configs
