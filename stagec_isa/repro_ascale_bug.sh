#!/usr/bin/env bash
# ============================================================================
# Repro: deferred-reuse pipeline a-scale (AS) bug
# ----------------------------------------------------------------------------
# Symptom: PF_PIPELINE=1 (deferred-reuse, pre_loaded=num_buffers, cb load_stage=buf_idx)
#          gives cosine ~0.985 on random data. Per-operand isolation pins it to AS
#          (a-scale) ONLY; spatial map pins it to output warp(1,1) = wave3 ONLY.
#          Deterministic (not a race), independent of TDM-store vs buffer-store,
#          independent of tensor_wait counts (FORCE_TCNT0=1 unchanged).
#
# Standard pipeline (.pf_backup/gemm_fp8fp4_gfx1250.py.pre-defer-reuse) = 1.0 (correct).
#
# Key fact: a-scale READ code (_precompute_scale_lane_bases) is symmetric in warp_m and
# does NOT depend on warp_n -> warp(1,0) and warp(1,1) read the SAME LDS a-scale, yet
# only warp(1,1) is wrong. A uses the same warp_m_base and is correct. So the suspect is
# wave3's a-scale VGPR/carry at the register level (the extra _addr_rem_box carry), not a
# source-level warp condition.
#
# Config: a8w4 32x32x256, m_warp=2 n_warp=2, num_buffers=2, pf_depth_wmma=1, l2pf=0.
# IMPORTANT: rm -rf ~/.flydsl/cache before EVERY run — inline env flags (PF_*, FORCE_*)
# are NOT in the .flydsl cache key, so toggling them without clearing gives stale hits.
# ============================================================================
# NOTE: no `set -u` — the ffmlite_env.sh references unset vars (LD_PRELOAD).
REPO=/data/zanzhang/FlyDSL-main
PY=/home/user/.venvs/flydsl-gfx1250/bin/python
TEST=tests/kernels/test_gemm_fp8fp4_gfx1250.py
CFG="--data-format a8w4 -M 32 -N 32 -K 1024 --tile-m 32 --tile-n 32 --tile-k 256 \
--m-warp 2 --n-warp 2 --num-buffers 2 --out-dtype bf16 --pf-depth-wmma 1 --l2-prefetch-distance 0"

# ---- source the ffmlite/rocdtif-7.13 cmodel env (from the ffm-env-setup skill) ----
FFM_DIR=$(ls -d /data/docker/overlay2/*/diff/home/user/ffm-env/rocdtif-7.13-am+ffmlite-mi400.*-rel-* 2>/dev/null | head -1)
[ -z "$FFM_DIR" ] && { echo "ERROR: ffm env dir not found"; exit 1; }
source "$FFM_DIR/ffmlite_env.sh"
cd "$REPO"
export PYTHONPATH="$REPO"
clean() { rm -rf ~/.flydsl/cache; }

case "${1:-all}" in

# ---------------------------------------------------------------------------
repro)   # 1) reproduce the failure (deferred-reuse) — expect cosine ~0.985
  clean
  echo "### DEFERRED-REUSE (PF_PIPELINE=1) — expect ~0.985 ###"
  PF_PIPELINE=1 $PY $TEST $CFG 2>&1 | grep -E "PASSED|Cosine|Mismatch"
  ;;

# ---------------------------------------------------------------------------
isolate) # 2) per-operand isolation — expect only AS != 1.0
  clean
  PF_PIPELINE=1 $PY - <<'PYEOF'
import torch, torch.nn.functional as F
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (preshuffle_e8m0_scale, _get_padded_problem_shape,
    _pad_mxscale_inputs, reference_a8w4_gemm, random_fp8_data)
from tests.kernels.utils import fp4_utils
import flydsl.compiler as flyc
M,N,K=32,32,1024; TM,TN,TK=32,32,256
ps=_get_padded_problem_shape('a8w4',M,N,K,TM,TN,TK,1); pm,pn,pk=ps['M'],ps['N'],ps['K']
torch.manual_seed(0)
fn=compile_mxscale_gemm(data_format='a8w4',N=pn,K=pk,tile_m=TM,tile_n=TN,tile_k=TK,m_warp=2,n_warp=2,
                        num_buffers=2,out_dtype='f32',wave_specialized_tdm=True,l2_prefetch_distance=0,pf_depth_wmma=1)
def run(tag,a,b,asc,bsc):
    ref=reference_a8w4_gemm(a,b,asc,bsc,M,N,K)
    ap,bp,ascp,bscp=_pad_mxscale_inputs(a.clone(),b.clone(),asc.clone(),bsc.clone(),ps)
    ascp=preshuffle_e8m0_scale(ascp,16,scale_k_per_tile=8); bscp=preshuffle_e8m0_scale(bscp,16,scale_k_per_tile=8)
    bp=fp4_utils.preshuffle_b_16x16_tiled(bp,pn,pk//ps['pack_b'],TN,TK//ps['pack_b'],ksmajor=True)
    cg=torch.zeros(pm,pn,dtype=torch.float32,device='cuda')
    flyc.compile(fn,cg,ap.cuda(),bp.cuda(),ascp.cuda(),bscp.cuda(),pm,pn,pk,pn,torch.cuda.current_stream()); torch.cuda.synchronize()
    out=cg[:M,:N].float().cpu()
    print(f"{tag}: cosine={F.cosine_similarity(out.flatten().float(),ref.flatten().float(),dim=0).item():.6f}")
Ar=random_fp8_data(M,K); Br=fp4_utils.random_fp4_packed(N,K); ASr=fp4_utils.random_e8m0(M,K//32); BSr=fp4_utils.random_e8m0(N,K//32)
Ac=torch.full((M,K),0x38,dtype=torch.uint8); Bc=torch.full((N,K//2),0x22,dtype=torch.uint8)
ASc=torch.full((M,K//32),127,dtype=torch.uint8); BSc=torch.full((N,K//32),127,dtype=torch.uint8)
run("ALL const  ",Ac,Bc,ASc,BSc)
run("A rand     ",Ar,Bc,ASc,BSc)
run("B rand     ",Ac,Br,ASc,BSc)
run("AS rand    ",Ac,Bc,ASr,BSc)   # <-- the only one != 1.0
run("BS rand    ",Ac,Bc,ASc,BSr)
run("ALL rand   ",Ar,Br,ASr,BSr)
PYEOF
  ;;

# ---------------------------------------------------------------------------
map)     # 3) spatial map of the AS error (AS rand, A/B/BS const) — expect only warp(1,1) wrong
  clean
  PF_PIPELINE=1 $PY - <<'PYEOF'
import torch
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (preshuffle_e8m0_scale, _get_padded_problem_shape,
    _pad_mxscale_inputs, reference_a8w4_gemm)
from tests.kernels.utils import fp4_utils
import flydsl.compiler as flyc
M,N,K=32,32,1024; TM,TN,TK=32,32,256
ps=_get_padded_problem_shape('a8w4',M,N,K,TM,TN,TK,1); pm,pn,pk=ps['M'],ps['N'],ps['K']
torch.manual_seed(0)
Ac=torch.full((M,K),0x38,dtype=torch.uint8); Bc=torch.full((N,K//2),0x22,dtype=torch.uint8); BSc=torch.full((N,K//32),127,dtype=torch.uint8)
AS=fp4_utils.random_e8m0(M,K//32)
ref=reference_a8w4_gemm(Ac,Bc,AS,BSc,M,N,K)
fn=compile_mxscale_gemm(data_format='a8w4',N=pn,K=pk,tile_m=TM,tile_n=TN,tile_k=TK,m_warp=2,n_warp=2,
                        num_buffers=2,out_dtype='f32',wave_specialized_tdm=True,l2_prefetch_distance=0,pf_depth_wmma=1)
ap,bp,ascp,bscp=_pad_mxscale_inputs(Ac.clone(),Bc.clone(),AS.clone(),BSc.clone(),ps)
ascp=preshuffle_e8m0_scale(ascp,16,scale_k_per_tile=8); bscp=preshuffle_e8m0_scale(bscp,16,scale_k_per_tile=8)
bp=fp4_utils.preshuffle_b_16x16_tiled(bp,pn,pk//ps['pack_b'],TN,TK//ps['pack_b'],ksmajor=True)
cg=torch.zeros(pm,pn,dtype=torch.float32,device='cuda')
flyc.compile(fn,cg,ap.cuda(),bp.cuda(),ascp.cuda(),bscp.cuda(),pm,pn,pk,pn,torch.cuda.current_stream()); torch.cuda.synchronize()
out=cg[:M,:N].float().cpu()
good=(out.sub(ref).abs() <= 1e-3*ref.abs()+1).int()
print("correct map (1=ok)  rows=M(0-31) cols=N(0-31); wrong block = warp(1,1)=rows16-31,cols16-31:")
for m in range(32): print(''.join(str(x) for x in good[m].tolist()))
PYEOF
  ;;

# ---------------------------------------------------------------------------
standard) # 4) standard pipeline (correct, 1.0) for comparison — swaps in the backup
  cp kernels/gemm_fp8fp4_gfx1250.py /tmp/_cur_kernel.py
  cp .pf_backup/gemm_fp8fp4_gfx1250.py.pre-defer-reuse kernels/gemm_fp8fp4_gfx1250.py
  clean
  echo "### STANDARD pipeline (.pre-defer-reuse) — expect 1.0 ###"
  PF_PIPELINE=1 $PY $TEST $CFG 2>&1 | grep -E "PASSED|Cosine"
  cp /tmp/_cur_kernel.py kernels/gemm_fp8fp4_gfx1250.py
  echo "(restored current kernel)"
  ;;

# ---------------------------------------------------------------------------
isa)     # 5) dump the deferred-reuse ISA (a-scale reads = v69 ds_load_b32)
  clean
  rm -rf /tmp/dr_isa
  FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/dr_isa PF_PIPELINE=1 $PY - <<'PYEOF'
import torch
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
import flydsl.compiler as flyc
N,K=32,1024; TM,TN,TK=32,32,256; pm,pn,pk=32,32,1024
z=lambda *s: torch.zeros(*s,dtype=torch.uint8,device='cuda')
fn=compile_mxscale_gemm(data_format='a8w4',N=pn,K=pk,tile_m=TM,tile_n=TN,tile_k=TK,m_warp=2,n_warp=2,
                        num_buffers=2,out_dtype='f32',wave_specialized_tdm=True,l2_prefetch_distance=0,pf_depth_wmma=1)
flyc.compile(fn,torch.zeros(pm,pn,dtype=torch.float32,device='cuda'),z(32,K),z(N,K//2),z(32,K//32),z(N,K//32),
             pm,pn,pk,pn,torch.cuda.current_stream())
PYEOF
  echo "ISA: /tmp/dr_isa/kernel_mxscale_gemm_0/21_final_isa.s"
  echo "AS reads (v69), BS reads (v70):"
  grep -nE "ds_load_b32 .*v69|ds_load_b32 .*v70" /tmp/dr_isa/kernel_mxscale_gemm_0/21_final_isa.s
  ;;

all)
  "$0" repro; echo; "$0" isolate; echo; "$0" map; echo; "$0" standard
  ;;
*) echo "usage: $0 {repro|isolate|map|standard|isa|all}"; exit 1;;
esac
