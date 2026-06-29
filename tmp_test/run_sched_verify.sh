#!/usr/bin/env bash
# One-click verify of M2 (the dynamic claim GEMM1 scheduler): 8-rank A/B of MegaMoEExp end-to-end
# output with FLYDSL_TMP_SCHED off vs on (both forced compact, so only the claim scheduler differs).
# PASS = relL2(on,off) ~0 (claim only reorders which block computes which tile -> math identical)
# AND sched actually engaged in the ON run.
#
# Usage (from anywhere; defaults to v4_pro small bs):
#     bash tmp_test/run_sched_verify.sh
#   override:
#     NET=v4_pro BS=64,256 ITERS=30 bash tmp_test/run_sched_verify.sh
set -u
cd "$(dirname "$0")/.." || { echo "cannot cd to repo root"; exit 1; }

NET="${NET:-v4_pro}"
BS="${BS:-64,256}"
ITERS="${ITERS:-30}"
WORLD="${WORLD:-8}"

# 1) ensure the experimental tmp_test/*.py are importable as kernels.* (idempotent symlinks).
for f in tmp_mega_gemm_2stage tmp_mega_megakernel tmp_mega_ep_dispatch tmp_mega_stage1_stage2 \
         tmp_mega_gemm2_combine_op tmp_mega_gemm2_combine_fused tmp_mega_gemm2_2stage megamoe_exp; do
  [ -e "kernels/$f.py" ] || ln -s "../tmp_test/$f.py" "kernels/$f.py"
done

# 2) on THIS dev box flydsl is broken by default -> use the complete 0.1.8 overlay if present.
#    on the production GPU box this dir won't exist and the system flydsl is used as-is.
DPKG=/data/docker/overlay2/c89e9c837133f1b365799dfa765953cce58915a923c0b8f813d48e223ae67641/diff/usr/local/lib/python3.12/dist-packages
PP=""
if [ -d "$DPKG/flydsl" ] && ! python3 -c "import mori.ir.flydsl" >/dev/null 2>&1; then
  PP="$DPKG"
  echo "[verify] using docker-overlay flydsl: $DPKG"
fi

echo "[verify] network=$NET bs=$BS iters=$ITERS world=$WORLD"
echo "[verify] A/B: FLYDSL_TMP_SCHED off vs on (both FORCE_COMPACT) -> relL2 must be ~0"
PYTHONPATH="${PP:+$PP:}${PYTHONPATH:-}" python3 tmp_test/test_sched_overlap.py \
    --network "$NET" --bs-list "$BS" --iters "$ITERS" --world "$WORLD" 2>&1 | tee /tmp/sched_verify.log

echo
echo "================= RESULT ================="
grep -E "sched-AB\] bs=|SUMMARY|PASS|FAIL|ERROR|Traceback" /tmp/sched_verify.log | tail -n 30
echo "=========================================="
