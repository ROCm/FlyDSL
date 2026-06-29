#!/usr/bin/env bash
# Trace-compile the experimental scheduler megakernel on THIS dev box (catches MLIR-build errors
# before handing to the GPU box).  Uses the complete 0.1.8 flydsl found in a docker overlay layer
# (the venv flydsl 0.1.4 lacks expr/extern.py; the editable 0.2.2 source lacks compiled _mlir).
#
# Experimental tmp_test/*.py are symlinked into kernels/ so their relative imports resolve.
# Edit tmp_test/*.py (source of truth) -> run this -> read errors.  NOT a correctness/perf test
# (single process, no real dispatch); only validates that the kernel lowers to MLIR.
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root
DPKG=/data/docker/overlay2/c89e9c837133f1b365799dfa765953cce58915a923c0b8f813d48e223ae67641/diff/usr/local/lib/python3.12/dist-packages
[ -d "$DPKG/flydsl" ] || { echo "ERROR: 0.1.8 flydsl overlay not found at $DPKG"; exit 1; }

# ensure experimental files are visible as kernels.* (idempotent symlinks)
for f in tmp_mega_gemm_2stage tmp_mega_megakernel tmp_mega_ep_dispatch tmp_mega_stage1_stage2 \
         tmp_mega_gemm2_combine_op tmp_mega_gemm2_combine_fused tmp_mega_gemm2_2stage megamoe_exp; do
  [ -e "kernels/$f.py" ] || ln -s "../tmp_test/$f.py" "kernels/$f.py"
done

SCHED="${SCHED:-1}"   # FLYDSL_TMP_SCHED for the compile (kernel-arg path)
echo "trace-compiling experimental GEMM kernel (FLYDSL_TMP_SCHED=$SCHED) ..."
FLYDSL_TMP_SCHED="$SCHED" timeout 500 env PYTHONPATH="$DPKG" python3 -c "
from kernels.tmp_mega_gemm_2stage import compile_fused_moe_gemm1
k = compile_fused_moe_gemm1(model_dim=7168, inter_dim=3072, experts=32, topk=1,
    tile_m=128, tile_n=256, tile_k=512, doweight_stage1=False,
    a_dtype='fp8', b_dtype='fp4', out_dtype='fp8', act='silu',
    waves_per_eu=4, use_async_copy=True, use_cshuffle_epilog=None,
    contiguous_io=True, dedup_gather=False, atom_contract=True,
    sparse_tiles=True, persist_m=-1, raw_a_scale=True, xcd_swizzle=0,
    fuse_dispatch='fixedslot', fuse_npes=8, fuse_topk=6,
    fuse_cap=8*16, fuse_mtpr=16, fuse_scale_dim=224, fuse_scale_type_size=1,
    rank=0, experts_per_rank=4, compact_dispatch=False, compact_allgather=True,
    sched=(__import__('os').environ.get('FLYDSL_TMP_SCHED','1')=='1'), sched_disp_base=50)
print('FIXEDSLOT COMPILE OK ->', type(k).__name__)
k2 = compile_fused_moe_gemm1(model_dim=7168, inter_dim=3072, experts=32, topk=1,
    tile_m=128, tile_n=256, tile_k=512, doweight_stage1=False,
    a_dtype='fp8', b_dtype='fp4', out_dtype='fp8', act='silu',
    waves_per_eu=4, use_async_copy=True, use_cshuffle_epilog=None,
    contiguous_io=True, dedup_gather=False, atom_contract=True,
    sparse_tiles=True, persist_m=-1, raw_a_scale=True, xcd_swizzle=0,
    fuse_dispatch='fixedslot', fuse_npes=8, fuse_topk=6,
    fuse_cap=8*16, fuse_mtpr=16, fuse_scale_dim=224, fuse_scale_type_size=1,
    rank=0, experts_per_rank=4, compact_dispatch=True, compact_allgather=True,
    sched=(__import__('os').environ.get('FLYDSL_TMP_SCHED','1')=='1'), sched_disp_base=50)
print('COMPACT   COMPILE OK ->', type(k2).__name__)
" 2>&1 | tail -25
