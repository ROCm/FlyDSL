#!/usr/bin/env python3
"""Reproducer: the fused mxfp_moe a4w4 / a8w4 MoE kernels are numerically broken.

This is a *tracking* reproducer for a confirmed pre-existing defect in
``kernels/moe/mxfp_moe/`` -- it does NOT fix anything. It demonstrates, with a
strict end-to-end cosine gate, that:

  * a16w4 (bf16 A x mxfp4 W, separate ``flydsl_a16w4_gemm1/2`` kernels) is
    numerically faithful (cos ~0.9999). This is the *control* that proves the
    reference / quant / shuffle / verify machinery in the reproducer is correct.
  * a4w4 (MX-FP4 A) and a8w4 (MX-FP8 A) drive the fused mxfp4 pipeline and are
    badly broken (cos ~0.1 / ~0.07), far below any quantization ceiling
    (aiter's a4w4/a8w4 reach cos ~0.995 on the same math -> a correct impl
    exists; this is a real defect, not a precision limit).

Localization (structural, from the shared-kernel topology + the measured table):
  * a16w4 uses its OWN gemm2 (``flydsl_a16w4_gemm2``) and passes.
  * a4w4 and a8w4 SHARE ``flydsl_mxfp4_gemm2`` (down-proj) and both fail ->
    the shared mxfp4 gemm2 down-proj is broken (see kernels/moe/mxfp_moe/gemm2.py).
  * a8w4 fails *worse* than a4w4 (they share the same broken gemm2, differ only
    in the stage1 activation dtype) -> the fp8 gemm1 A-path is *additionally*
    broken (see kernels/moe/mxfp_moe/gemm1.py:237-243,256-270).

The broken kernels are also memory-unsafe: launching them and unwinding can
raise ``hipErrorIllegalAddress`` that corrupts the HIP module state and cascades
into unrelated work in the same process. The decisive, SAFE evidence here is the
cosine table (all runs below succeed cleanly); the illegal-address behaviour is
demonstrated separately, in a SUBPROCESS, so it cannot crash this harness.

Run (cold cache, deterministic):

    cd /root/FlyDSL-a16w4-verify && source .verify_runenv.sh && \
      HIP_VISIBLE_DEVICES=7 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
      python3 tools/repro/repro_mxfp_moe_a4w4_a8w4_broken.py

Exit code 0 = reproduced as expected (control faithful, a4w4/a8w4 broken).
Exit code 1 = did NOT reproduce (control failed, or a4w4/a8w4 unexpectedly fixed).

See docs/issues/mxfp_moe_a4w4_a8w4_broken.md for the filed-issue draft.
"""

import os
import sys

import torch

# Make `tests.*` / `kernels.*` importable when run from the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse the EXACT test-internal e2e path so the reference/quant/shuffle/verify is
# byte-identical to the committed test. We only monkeypatch verify_output to
# CAPTURE (out, ref2) instead of asserting, so we can compute the real cosine.
import tests.kernels.test_moe_gemm as T  # noqa: E402
from tests.kernels.test_moe_gemm import build_routing_buffers  # noqa: E402

# Control thresholds.
CONTROL_MIN_COS = 0.99  # a16w4 must clear this for the table to be trusted.
BROKEN_MAX_COS = 0.90  # a4w4/a8w4 must be *below* this to confirm the defect.

SHAPE = dict(tokens=128, model_dim=1024, inter_dim=256, experts=8, topk=2, tile_m=32)

_CAP = {}


def _capturing_verify_output(out, ref2, *args, **kwargs):
    _CAP["out"] = out.detach().clone()
    _CAP["ref2"] = ref2.detach().clone()
    return True  # never assert; we compute cosine ourselves


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _make_inputs(cfg):
    dev = torch.device("cuda")
    s = 0.2
    tokens, model_dim, inter_dim = cfg["tokens"], cfg["model_dim"], cfg["inter_dim"]
    experts, topk = cfg["experts"], cfg["topk"]
    torch.manual_seed(0)  # deterministic
    x_fp32 = torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * s
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * s
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=dev, dtype=torch.float32) * (s / (inter_dim**0.5))
    score = torch.rand((tokens, experts), device=dev, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=cfg["tile_m"],
        moe_sort_mode="torch",
    )
    return x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights, routing


def _run_e2e_cos(a_dtype, cfg):
    """Drive the exact committed e2e path; return (cos, max_abs_err, ref_absmax)."""
    _CAP.clear()
    x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights, routing = _make_inputs(cfg)
    T._run_mxfp_moe_e2e(
        tokens=cfg["tokens"],
        model_dim=cfg["model_dim"],
        inter_dim=cfg["inter_dim"],
        experts=cfg["experts"],
        topk=cfg["topk"],
        tile_m=cfg["tile_m"],
        use_reduce=False,
        x_fp32=x_fp32,
        w1_fp32=w1_fp32,
        w2_fp32=w2_fp32,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        routing=routing,
        a_dtype=("a16" if a_dtype == "a16w4" else a_dtype),
        inline_quant=False,
        interleave=False,
        skip_ref=False,
    )
    out, ref2 = _CAP["out"], _CAP["ref2"]
    return _cos(out, ref2), (out - ref2).abs().max().item(), ref2.abs().max().item()


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA/HIP device visible.")
        return 0

    # Patch AFTER import so the exact test code path is used.
    T.verify_output = _capturing_verify_output

    cfg = SHAPE
    print("=" * 78)
    print("mxfp_moe a4w4 / a8w4 breakage reproducer")
    print(f"shape = {cfg}")
    print("=" * 78)

    rows = {}
    # a16w4 first: it is the control and validates the harness fidelity.
    for a_dtype in ["a16w4", "fp4", "fp8"]:
        label = {"a16w4": "a16w4 (control)", "fp4": "a4w4  (fp4)", "fp8": "a8w4  (fp8)"}[a_dtype]
        try:
            c, mae, absmax = _run_e2e_cos(a_dtype, cfg)
            rows[a_dtype] = c
            print(f"  {label:16s} e2e cos = {c:9.6f}   " f"max_abs_err = {mae:8.4f}   ref_absmax = {absmax:.4f}")
        except Exception as e:  # noqa: BLE001
            rows[a_dtype] = None
            print(f"  {label:16s} ERROR: {type(e).__name__}: {e}")

    print("-" * 78)
    print("Localization (shared-kernel topology + measured table):")
    print("  * a16w4 uses its OWN gemm1/gemm2 (flydsl_a16w4_*) and is faithful.")
    print("  * a4w4 and a8w4 SHARE flydsl_mxfp4_gemm2 (down-proj) and BOTH fail")
    print("    -> shared mxfp4 gemm2 down-proj is broken (kernels/moe/mxfp_moe/gemm2.py).")
    print("  * a8w4 fails worse than a4w4 (same broken gemm2, only stage1 A dtype")
    print("    differs) -> fp8 gemm1 A-path also broken (gemm1.py:237-243,256-270).")
    print("-" * 78)

    # --- PASS/FAIL verdict ---------------------------------------------------
    ok_control = rows.get("a16w4") is not None and rows["a16w4"] >= CONTROL_MIN_COS
    ok_fp4_broken = rows.get("fp4") is not None and rows["fp4"] < BROKEN_MAX_COS
    ok_fp8_broken = rows.get("fp8") is not None and rows["fp8"] < BROKEN_MAX_COS

    if not ok_control:
        print(
            "RESULT: FAIL -- a16w4 control did NOT reach the fidelity floor "
            f"({CONTROL_MIN_COS}); the a4w4/a8w4 numbers cannot be trusted."
        )
        return 1
    if ok_fp4_broken and ok_fp8_broken:
        print(
            "RESULT: PASS -- reproduced. Control faithful (a16w4 >= "
            f"{CONTROL_MIN_COS}); a4w4 & a8w4 broken (cos < {BROKEN_MAX_COS})."
        )
        print("        The illegal-address behaviour is demonstrated separately")
        print("        (subprocess) so it cannot crash this harness -- see the")
        print("        module docstring / issue draft for the note.")
        return 0
    print(
        "RESULT: FAIL -- a4w4/a8w4 did NOT reproduce the breakage "
        f"(expected cos < {BROKEN_MAX_COS}). Has the kernel been fixed? "
        "Update the xfail in tests/kernels/test_moe_gemm.py and this repro."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
