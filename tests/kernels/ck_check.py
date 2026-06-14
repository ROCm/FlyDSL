# SPDX-License-Identifier: Apache-2.0
"""Standalone single-shape correctness checker for a FlyDSL FMHA fp8 module.

Run ONE shape per process (the module-global smem can't re-finalize across shapes).
Usage:
  HIP_VISIBLE_DEVICES=6 python3 tests/kernels/ck_check.py <module> b sq sk nk gqa causal page_size pscale
Example (customer-ish):
  HIP_VISIBLE_DEVICES=6 python3 tests/kernels/ck_check.py fmha_prefill_fp8_ck 1 256 256 1 8 1 16 1.0
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import importlib

import torch

import fmha_prefill_fp8_ref as R


def main():
    mod = sys.argv[1]
    b, sq, sk, nk, gqa, causal, ps = (int(x) for x in sys.argv[2:9])
    pscale = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
    K = importlib.import_module(mod)
    HD = K.HD
    nq = nk * gqa
    sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD)
    k = torch.randn(b, sk, nk, HD)
    v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q)
    kf, kd = R.quantize_per_token_head(k)
    vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
    args = [
        qf.to("cuda"),
        c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"),
        kd.to("cuda"),
        vd.to("cuda"),
        c.page_ids.to("cuda"),
        c.kv_indptr.to("cuda"),
        torch.full((b * nq,), pscale, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)
    K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, causal, grid)
    torch.cuda.synchronize()
    ref = R.fmha_prefill_reference(qf, kf, vf, qd, kd, vd, sm, causal=bool(causal))
    err = (Og.float().cpu() - ref.float()).abs().max().item()
    tag = f"{mod} b{b} sq{sq} sk{sk} nk{nk} gqa{gqa} c{causal} ps{ps} pscale{pscale}"
    print(f"{tag} -> ERR {err:.4f} {'OK' if err < 6e-2 else 'FAIL'}  (KT={getattr(K,'KT','?')} NBUF={getattr(K,'NBUF','?')} BM={K.BM})")


if __name__ == "__main__":
    main()
