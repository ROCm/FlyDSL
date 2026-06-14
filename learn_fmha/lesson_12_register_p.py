# SPDX-License-Identifier: Apache-2.0
"""Lesson 12 — Register-resident P (ds_bpermute) + the DECISIVE profile reading.

This lesson is an ISA/PMC READING exercise on the real kernel, not a new microkernel —
the trick is an internal transform of kernels/fmha_prefill_fp8_8wave.py, so the most
honest way to "see" it is to dump its ISA and read the counters, which this script does.

### The trick
Lesson 07 transposed P through LDS: store P to LDS, barrier, reload 8-contiguous-kv per
lane. That's a full LDS round-trip per kv-tile. The register-resident version does the
transpose ACROSS LANES with ds_bpermute (a lane reads another lane's register), avoiding
the explicit LDS store/reload of P. The 8wave kernel uses 4 ds_bpermute per k-step.

### The decisive lesson (why this matters more than the trick itself)
When you PROFILE the result you discover register-P did NOT make the kernel fast — and
the PMC tells you why, reframing the whole optimization effort. Measured on the 8wave
kernel (sq16384, rocprofv3):
    SQ_WAIT_INST_LDS / SQ_BUSY_CU_CYCLES  ~= 54%   (HALF of all cycles wait on LDS)
    SQ_INSTS_VALU    / SQ_INSTS_MFMA      ~= 23:1  (drowning in ALU per matrix op)
ds_bpermute is ITSELF an LDS-unit op, so "register-P" just MOVED LDS traffic, it didn't
remove it. The kernel is LDS-bound and VALU-heavy, NOT memory- or compute-bound. THIS
reading is what pointed at column-V (Lesson 17) as the real fix: don't move the transpose,
DELETE it.

This script dumps the 8wave ISA and prints the ds-op mix so you can see the LDS pressure.
Then read the .md for the rocprofv3 command + the counter interpretation.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_12_register_p.py
"""

import glob
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    dump = "/tmp/isa_l12"
    os.environ.update(HIP_VISIBLE_DEVICES="2", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=dump,
                      FLYDSL_RUNTIME_ENABLE_CACHE="0", FMHA_NWAVES="4")
    # compile+run the real 8wave kernel once (single shape) to emit its ISA.
    runner = (
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
    subprocess.run([sys.executable, "-c", runner], cwd=_REPO, check=False)
    isa = glob.glob(f"{dump}/*/21_final_isa.s")
    if not isa:
        print("no ISA dumped (did the kernel compile?)")
        return
    s = open(isa[0]).read()
    print(f"\n8wave kernel ISA op mix ({os.path.basename(os.path.dirname(isa[0]))}):")
    for op in ("v_mfma", "ds_read", "ds_write", "ds_bpermute", "v_perm"):
        print(f"  {op:14} {s.count(op)}")
    print("\nNote ds_bpermute > 0 and ds_read/ds_write dominate v_mfma — this kernel is LDS-bound.")
    print("Register-P moved the P transpose into ds_bpermute (an LDS-unit op), so it did NOT")
    print("remove LDS pressure. PMC (see .md): LDS-wait ~54% of busy, VALU:MFMA ~23:1.")
    print("That reading is what motivated column-V (Lesson 17) — DELETE the transpose, don't move it.")


if __name__ == "__main__":
    main()
