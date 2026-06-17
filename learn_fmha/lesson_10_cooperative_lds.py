# SPDX-License-Identifier: Apache-2.0
"""Lesson 10 — Cooperative K/V staging into LDS (and the barrier cost).

Explainer + ISA reading on the real kernel. With multiple waves per workgroup (Lesson 08),
all waves in a workgroup need the SAME K/V tile. Two options:
  (a) every wave loads the tile from global itself  -> N_waves x redundant global traffic.
  (b) the waves COOPERATE: each thread loads a slice once into LDS, one barrier, then all
      waves read the shared tile from LDS. This is what fmha_prefill_fp8_8wave.py does
      (see load_kv_regs + store_kv_to_lds + the cooperative pass loop).

The win is removing redundant global loads; the COST is a workgroup barrier (gpu.barrier)
so every wave sees the stored tile before reading. That barrier is why cooperative LDS only
pays off once it's PIPELINED (Lesson 15) — a bare barrier-per-tile serializes the waves.

This script dumps the 8wave ISA so you can see the cooperative loads land in LDS
(ds_write) and the s_barrier that guards them. Read the .md for the trade-off and the
"don't stage what isn't the bottleneck" rule.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_10_cooperative_lds.py
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
    dump = "/tmp/isa_l10"
    os.environ.update(HIP_VISIBLE_DEVICES="2", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=dump,
                      FLYDSL_RUNTIME_ENABLE_CACHE="0", FMHA_NWAVES="4")
    subprocess.run([sys.executable, "-c", _RUN], cwd=_REPO, check=False)
    isa = glob.glob(f"{dump}/*/21_final_isa.s")
    if isa:
        s = open(isa[0]).read()
        print("\n8wave LDS staging evidence:")
        print(f"  ds_write (coop store to LDS): {s.count('ds_write')}")
        print(f"  ds_read  (waves read shared) : {s.count('ds_read')}")
        print(f"  s_barrier                    : {s.count('s_barrier')}")
        print(f"  buffer_load (global)         : {s.count('buffer_load')}")
    print("\nCooperative LDS removes redundant per-wave global loads; the price is s_barrier")
    print("per tile. PRIOR RESULT: K->LDS staging ALONE regressed (added barriers for a")
    print("non-bottleneck). Only pays off pipelined (Lesson 15). Rule: don't stage to LDS")
    print("unless the thing you're staging is the measured hotspot.")


if __name__ == "__main__":
    main()
