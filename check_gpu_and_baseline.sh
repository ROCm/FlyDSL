#!/usr/bin/env bash
# GPU runtime diagnostic + Stage A+B baseline validation for the TDM K-split work.
# Run this in an environment where the GPU is actually reachable.
#
#   bash check_gpu_and_baseline.sh
#
# It (1) probes the ROCm/HSA runtime, (2) if healthy, runs the verified
# Stage A+B K-split baseline (PF_FORCE_DK=2 PF_SPLIT_OP=all, a8w4) which must
# print a correctness ratio of 1.0 before we touch Stage C.

set -uo pipefail

PY=/home/user/.venvs/flydsl-gfx1250/bin/python
ROCMINFO=/home/user/.venvs/flydsl-gfx1250/bin/rocminfo
REPO=/data/zanzhang/FlyDSL-main
TEST="$REPO/tests/kernels/test_gemm_fp8fp4_gfx1250.py"

echo "########################################################################"
echo "# 1. Devices present?"
echo "########################################################################"
ls -l /dev/kfd 2>&1
ls /dev/dri 2>&1 | tr '\n' ' '; echo
echo

echo "########################################################################"
echo "# 2. rocminfo (does HSA enumerate, or segfault?)"
echo "########################################################################"
timeout 30 "$ROCMINFO" 2>&1 | grep -iE 'Agent |Name:|gfx|Marketing' | head -20
echo "  rocminfo rc=$?"
echo

echo "########################################################################"
echo "# 3. torch.cuda enumeration (this is what currently segfaults at 8)"
echo "########################################################################"
"$PY" -X faulthandler -c "
import faulthandler; faulthandler.enable()
import torch
print('torch', torch.__version__, flush=True)
print('device_count', torch.cuda.device_count(), flush=True)
print('is_available', torch.cuda.is_available(), flush=True)
print('name', torch.cuda.get_device_name(0), flush=True)
" 2>&1
echo "  torch probe rc=$?"
echo

echo "########################################################################"
echo "# 4. Same torch probe, pinned to one device (ROCR/HIP_VISIBLE_DEVICES=0)"
echo "########################################################################"
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 "$PY" -c "
import torch
print('device_count', torch.cuda.device_count())
print('name', torch.cuda.get_device_name(0))
" 2>&1
echo "  pinned torch probe rc=$?"
echo

echo "########################################################################"
echo "# 5. If GPU is healthy: Stage A+B baseline (MUST be ratio 1.0)"
echo "#    PF_FORCE_DK=2 PF_SPLIT_OP=all, a8w4, K=3072, tile_n=128"
echo "########################################################################"
cd "$REPO" || exit 1
PF_FORCE_DK=2 PF_SPLIT_OP=all "$PY" "$TEST" \
    --data-format a8w4 -M 1024 -N 1024 -K 3072 \
    --tile-m 128 --tile-n 128 --tile-k 128 \
    --m-warp 2 --n-warp 2 --num-buffers 4 --out-dtype bf16 2>&1 | tail -40
echo "  baseline rc=$?"
echo
echo "Done. Paste the full output back."
