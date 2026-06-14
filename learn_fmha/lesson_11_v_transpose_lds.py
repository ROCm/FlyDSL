# SPDX-License-Identifier: Apache-2.0
"""Lesson 11 — V transposed in LDS: turn a byte-gather into wide reads.

Explainer + ISA reading. GEMM2 contracts over kv, but V is stored [kv, d] row-major — so
a naive GEMM2 gathers V one byte at a time per (kv,d) (the original kernel had 64
buffer_load_ubyte per tile, totally uncoalesced). The fix: cooperatively store V into LDS
TRANSPOSED (column-major, d-contiguous), so each GEMM2 operand is a single wide ds_read
instead of 8+ scalar byte loads.

This is the V-side counterpart of the P-transpose (Lesson 12). Both pay the transpose on
the DS unit; Lesson 17 (column-V) removes the V transpose entirely by storing V
column-major in GLOBAL memory so even the LDS transpose disappears.

The script dumps the 8wave ISA; note ds_read are now WIDE (b64/b128) and the byte-gather
(buffer_load_ubyte) is gone. Read the .md for the before/after instruction story.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_11_v_transpose_lds.py
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


def main():
    dump = "/tmp/isa_l11"
    os.environ.update(HIP_VISIBLE_DEVICES="2", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=dump,
                      FLYDSL_RUNTIME_ENABLE_CACHE="0", FMHA_NWAVES="4")
    subprocess.run([sys.executable, "-c", _RUN], cwd=_REPO, check=False)
    isa = glob.glob(f"{dump}/*/21_final_isa.s")
    if isa:
        s = open(isa[0]).read()
        print("\n8wave GEMM2 V-read evidence:")
        print(f"  buffer_load_ubyte (V byte-gather, BAD): {s.count('buffer_load_ubyte')}")
        print(f"  ds_read_b64 / b128 (wide LDS reads)   : {s.count('ds_read_b64') + s.count('ds_read_b128')}")
        print(f"  ds_write_b8 (the V-transpose scatter) : {s.count('ds_write_b8')}")
    print("\nV-transpose-in-LDS killed the 64 byte-gathers -> wide ds_read. But note ds_write_b8")
    print("(the scatter that BUILDS the transpose) is now the dominant LDS op -> Lesson 17")
    print("removes it entirely with column-major V in global memory.")


if __name__ == "__main__":
    main()
