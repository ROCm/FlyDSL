# SPDX-License-Identifier: Apache-2.0
"""Lesson 15 — LDS ping-pong double-buffer + next-tile prefetch (hide load latency).

Explainer + ISA reading. Cooperative LDS staging (Lesson 10) added a barrier per kv-tile,
and the global->LDS load has ~hundreds of cycles latency that the GEMMs would otherwise
stall on. Two combined tricks (in fmha_prefill_fp8_8wave.py):
  - PING-PONG: two LDS buffers. Compute reads buffer (tile&1) while the NEXT tile is being
    loaded into buffer ((tile+1)&1) -> no wait for the store to finish before computing.
  - PREFETCH (OPT3): ISSUE the next tile's global loads early (during this tile's GEMM1/
    softmax/GEMM2) so the ~300cy latency overlaps compute; DEFER the LDS store until after
    the GEMMs so it doesn't stall right before GEMM1.

This is the classic software-pipeline shape (CK Tile's pipeline does the same). On THIS
kernel the measured gain was small (+0.3 TF) because the bottleneck is the LDS transpose
itself (Lesson 12), not the global-load latency — another "fix the binding bottleneck"
data point. It matters more once the transpose is removed (column-V, Lesson 17).

Read the prologue + the "issue next tile / deferred store" block in
fmha_prefill_fp8_8wave.py. This script dumps the ISA so you can see the 2 LDS buffers and
the loads hoisted above the MFMAs.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_15_pingpong_prefetch.py
"""

import glob
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUN = (
    "import sys; sys.path.insert(0,'tests/kernels'); sys.path.insert(0,'kernels');"
    "import torch, fmha_prefill_fp8_ref as R, fmha_prefill_fp8_8wave as K;"
    "b,sq,sk,nk,gqa,ps=1,1024,1024,1,8,16; nq=nk*gqa; sm=1/128**0.5; torch.manual_seed(0);"
    "q=torch.randn(b,sq,nq,128);k=torch.randn(b,sk,nk,128);v=torch.randn(b,sk,nk,128);"
    "qf,qd=R.quantize_per_token_head(q);kf,kd=R.quantize_per_token_head(k);vf,vd=R.quantize_per_head(v);"
    "c=R.pack_paged_cache(kf,vf,ps,scatter=True);"
    "a=[qf.to('cuda'),c.k_pool.view(torch.float8_e4m3fnuz).to('cuda'),c.v_pool.view(torch.float8_e4m3fnuz).to('cuda'),"
    "qd.to('cuda'),kd.to('cuda'),vd.to('cuda'),c.page_ids.to('cuda'),c.kv_indptr.to('cuda'),torch.full((b*nq,),1.0,device='cuda')];"
    "O=torch.zeros(b,sq,nq,128,device='cuda',dtype=torch.bfloat16);grid=b*nq*((sq+K.BM-1)//K.BM);"
    "K.run_attn(*a,O,sq,sk,nq,nk,ps,c.k_page_stride,c.v_page_stride,sm,1,grid);torch.cuda.synchronize()"
)

if __name__ == "__main__":
    dump = "/tmp/isa_l15"
    os.environ.update(HIP_VISIBLE_DEVICES="2", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=dump,
                      FLYDSL_RUNTIME_ENABLE_CACHE="0", FMHA_NWAVES="4")
    subprocess.run([sys.executable, "-c", _RUN], cwd=_REPO, check=False)
    isa = glob.glob(f"{dump}/*/21_final_isa.s")
    if isa:
        s = open(isa[0]).read()
        print("\n8wave pipeline evidence:")
        print(f"  s_barrier per hot loop : {s.count('s_barrier')}")
        print(f"  buffer_load (prefetch) : {s.count('buffer_load')}")
    print("\nPing-pong (2 LDS buffers) + prefetch (issue next-tile loads during compute) is the")
    print("classic software pipeline. Here it gained only ~+0.3 TF because the LDS TRANSPOSE")
    print("(not load latency) is the bottleneck (Lesson 12). Fix the binding bottleneck first.")
