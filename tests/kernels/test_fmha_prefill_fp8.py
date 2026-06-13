# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the FP8 paged causal FMHA prefill kernel (gfx942).

Each parametrized case runs in-process; the kernel is launched once per case.
NOTE: the FlyDSL JIT cannot be re-finalized for different shapes within a single
process (the static smem global goes stale), so each shape is covered by a
separate pytest case (separate process under -p no:cacheprovider / xdist, or
just one assert per invocation here — pytest runs them sequentially but the
smem global is shared, so we keep ONE launch per process via subprocess).

We therefore drive the kernel through a subprocess helper to guarantee a fresh
JIT/smem state per shape, matching how the kernel will be used in production
(one compiled launcher per config).
"""

import subprocess
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]  # /workspaces/amir/FlyDSL

# (b, sq, sk, nk, gqa, causal, page_size, p_scale)
CASES = [
    (1, 32, 32, 1, 1, 1, 16, 1.0),
    (1, 64, 64, 2, 2, 1, 16, 1.0),
    (1, 128, 128, 2, 4, 1, 16, 1.0),
    (2, 64, 128, 2, 2, 1, 16, 1.0),
    (1, 96, 256, 1, 1, 1, 16, 1.0),
    (2, 128, 128, 4, 4, 1, 32, 1.0),
    (1, 64, 64, 1, 1, 0, 16, 1.0),  # non-causal
    (1, 64, 64, 2, 2, 1, 16, 16.0),  # p_scale != 1 (must cancel)
    (1, 64, 64, 2, 2, 1, 16, 64.0),
]

_RUNNER = r'''
import sys
sys.path.insert(0, "tests/kernels")
sys.path.insert(0, "kernels")
import torch
import fmha_prefill_fp8_ref as R
import fmha_prefill_fp8 as K

b, sq, sk, nk, gqa, causal, ps, pscale = (
    int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]),
)
torch.manual_seed(0)
HD = K.HD
nq = nk * gqa
sm = 1.0 / HD**0.5
q = torch.randn(b, sq, nq, HD)
k = torch.randn(b, sk, nk, HD)
v = torch.randn(b, sk, nk, HD)
qf, qd = R.quantize_per_token_head(q)
kf, kd = R.quantize_per_token_head(k)
vf, vd = R.quantize_per_head(v)
cache = R.pack_paged_cache(kf, vf, ps, scatter=True)
LTDg = cache.page_ids.to("cuda")
LTPg = cache.kv_indptr.to("cuda")
Kpool = cache.k_pool.view(torch.float8_e4m3fnuz).to("cuda")
Vpool = cache.v_pool.view(torch.float8_e4m3fnuz).to("cuda")
Psg = torch.full((b * nq,), pscale, device="cuda", dtype=torch.float32)
args = [qf.to("cuda"), Kpool, Vpool, qd.to("cuda"), kd.to("cuda"), vd.to("cuda"), LTDg, LTPg, Psg]
Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
grid = b * nq * ((sq + K.BM - 1) // K.BM)
K.run_attn(*args, Og, sq, sk, nq, nk, ps, cache.k_page_stride, cache.v_page_stride, sm, causal, grid)
torch.cuda.synchronize()
ref = R.fmha_prefill_reference(qf, kf, vf, qd, kd, vd, sm, causal=bool(causal))
err = (Og.float().cpu() - ref.float()).abs().max().item()
print("ERR", err)
sys.exit(0 if err < 6e-2 else 1)
'''


@pytest.mark.parametrize("case", CASES, ids=lambda c: "b{}_sq{}_sk{}_nk{}_g{}_c{}_ps{}_psc{}".format(*c))
def test_fmha_prefill_fp8(case):
    runner = _REPO / "tests" / "kernels" / "_fmha_prefill_runner.py"
    runner.write_text(_RUNNER)
    proc = subprocess.run(
        [sys.executable, str(runner), *[str(x) for x in case]],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        env={"HIP_VISIBLE_DEVICES": "2", "PATH": "/usr/bin:/bin:/opt/rocm/bin"},
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"case {case} failed:\n{out[-2000:]}"
    assert "ERR" in out, f"no result for case {case}:\n{out[-2000:]}"
