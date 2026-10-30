# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regenerate `flash_attn_gfx950_aotriton_golden.json` from AOTriton's vendored gfx950 tuning modules.

    python3 -I tests/unit/data/gen_flash_attn_gfx950_golden.py <aotriton>/modules/flash/flyc \
        tests/unit/data/flash_attn_gfx950_aotriton_golden.json

The golden records, for the P0 matrix (every rung x {bf16, f16} x {dense, window, bias, dropout}), what AOTriton's
`resolve` decides for the forward, dQ and dK/dV, and every trait field it derives. Run it with **no flydsl on the
path**, as AOTriton's own generator does: `gfx950_standalone` then falls back to its flydsl-free traits copy, and the
`fmha_*_gfx950` tuning modules import cleanly. Names are AOTriton's (`block_dmodel`, `num_waves`, ...);
`test_flash_attn_gfx950_config.py` owns the mapping onto this repository's knob and trait names.
"""

import dataclasses
import json
import sys

RUNGS = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)
FEATURES = ("dense", "window", "bias", "dropout")


def main(flyc_dir, out):
    sys.path.insert(0, flyc_dir)
    import fmha_tuning_bwd_dkdv_gfx950 as dkdv
    import fmha_tuning_bwd_dq_gfx950 as dq
    import fmha_tuning_gfx950 as fwd

    def knob_dict(k):
        return {f.name: getattr(k, f.name) for f in dataclasses.fields(k)}

    def trait_dict(t):
        return {f.name: getattr(t, f.name) for f in dataclasses.fields(t)}

    cases = []
    for rung in RUNGS:
        for dtype in ("bf16", "f16"):
            for feat in FEATURES:
                window, bias, dropout = feat == "window", feat == "bias", feat == "dropout"
                meta = fwd.FmhaInputMetadata(
                    num_heads=1,
                    head_dim=rung,
                    head_dim_v=rung,
                    causal=window,
                    window=window,
                    bias=bias,
                    dropout=dropout,
                    dtype_str=dtype,
                )
                dmeta = dkdv.BwdDkDvInputMetadata(
                    num_heads=1,
                    head_dim=rung,
                    head_dim_v=rung,
                    causal=window,
                    window=window,
                    bias=bias,
                    dropout=dropout,
                    dtype_str=dtype,
                )
                entry = dict(rung=rung, dtype=dtype, feature=feat)
                fk = fwd.fmha_knobs("gfx950").resolve(meta)
                entry["fwd"] = dict(knobs=knob_dict(fk), traits=trait_dict(fk.build_traits(meta)))
                qk = dq.bwd_dq_knobs("gfx950", store_db=bias).resolve(meta)
                entry["dq"] = dict(knobs=knob_dict(qk), traits=trait_dict(qk.build_traits(meta)))
                kk = dkdv.bwd_dkdv_knobs("gfx950").resolve(dmeta)
                entry["dkdv"] = dict(knobs=knob_dict(kk), traits=trait_dict(kk.build_traits(dmeta)))
                cases.append(entry)
    # Compact form: per kernel, the trait fields identical in every case are stored once, and the rest as lists
    # aligned with `varying`.
    packed = {}
    for kernel in ("fwd", "dq", "dkdv"):
        names = list(cases[0][kernel]["traits"])
        constant = {
            n: cases[0][kernel]["traits"][n]
            for n in names
            if all(c[kernel]["traits"][n] == cases[0][kernel]["traits"][n] for c in cases)
        }
        varying = [n for n in names if n not in constant]
        packed[kernel] = dict(constant=constant, varying=varying)
        for c in cases:
            c[kernel]["traits"] = [c[kernel]["traits"][n] for n in varying]
    with open(out, "w") as f:
        json.dump(dict(cases=cases, traits=packed), f, sort_keys=True, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
