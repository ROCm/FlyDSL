#!/bin/bash
# torchrun --no-python wrapper: apply rocprofv3 ONLY to LOCAL_RANK 0 (avoids 8-way
# output collision). Other ranks run plain python so the collective/P2P still forms.
#   TRACE_YAML=/tmp/x.yaml torchrun --no-python --nproc_per_node=8 megamoeexp/trace_wrap.sh
set -e
cd /home/ghu/FlyDSL
if [ "${LOCAL_RANK:-0}" = "0" ] && [ -n "$TRACE_YAML" ]; then
  exec rocprofv3 -i "$TRACE_YAML" -- python -u megamoeexp/trace_drv.py
else
  exec python -u megamoeexp/trace_drv.py
fi
