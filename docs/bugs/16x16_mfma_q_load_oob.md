# 16x16 MFMA Kernel: Q Load Out-of-Bounds Memory Fault

## Symptom

The 16x16x16 MFMA flex attention kernel (`flex_attn_fwd_gfx950_16x16_kernel`)
crashed with **"Memory access fault"** when the HIP grid exceeded ~1024
workgroups — for example `B=2, S=8192, H=32, D=128`. The same kernel worked
correctly at smaller grids (e.g. `S=64, H=32` → 128 WGs).

The crash was reproducible across all 8 GPUs on the MI350X node, ruling out
hardware faults. The equivalent 32x32x16 MFMA kernel worked at any grid size.

## Root Cause

The `BufferCopy128b` copy atom used to load Q from global memory reads **8 bf16
elements (128 bits) per load**. The 16x16x16 MFMA B operand layout distributes
64 lanes across 16 N-columns × 4 K-groups, giving each K-group a D-offset of
`group * 4`:

| K-group (lanes) | D start | 8 loads × stride 16 | D range |
|---|---|---|---|
| 0 (tid 0–15) | 0 | 0, 16, 32, … 112 | **[0, 119]** |
| 1 (tid 16–31) | 4 | 4, 20, 36, … 116 | **[4, 123]** |
| 2 (tid 32–47) | 8 | 8, 24, 40, … 120 | **[8, 127]** |
| 3 (tid 48–63) | 12 | 12, 28, 44, … 124 | **[12, 131]** ← OOB |

K-group 3's final 128-bit load reads D positions **124–131**, but `head_dim` is
only 128 (D = 0–127). The last 4 elements (**D = 128–131**) are past the end of
the head dimension.

The Q buffer descriptor used `max_size=True` (`num_records = 0xFFFFFFFF`), which
tells the hardware that *any* 32-bit offset is valid. The hardware therefore does
not clamp the OOB read — it generates a real global memory address. When that
address falls past the Q tensor's allocation and hits an unmapped GPU page, the
MMU raises a memory access fault.

### Why it was grid-size-dependent

The OOB read overshoots by only 8 bytes. At small grids, the last workgroup's Q
tile is near the middle of the tensor, and the 8 extra bytes land inside the
next head's Q data (mapped memory, wrong but silent). At large grids, the last
workgroup processes the very last Q rows, and the overshoot lands past the
tensor allocation — in unmapped VRAM. Whether the fault fires depends on the
exact tensor placement relative to page boundaries, which is why adding a 2 MB
dummy allocation before Q shifted addresses enough to avoid it.

### Why the 32x32 kernel is not affected

The 32x32x16 MFMA has only **2 K-groups** (64 lanes / 32 N-columns) with
D-offsets 0 and 4, and a load stride of 32. The maximum D accessed is
`4 + 3×32 + 7 = 107`, well within `head_dim = 128`.

## Investigation Timeline

The bug was initially misattributed to the softmax O-rescale loop, then to an
LLVM AMDGPU backend codegen issue. A systematic binary-search isolation
established:

1. **DMA only** → pass
2. **DMA + K/V reads** → pass
3. **DMA + reads + QK GEMM** → pass
4. **DMA + reads + QK + softmax** → pass
5. **DMA + reads + QK + PV MFMA (constant B operand)** → pass
6. **DMA + reads + QK + PV MFMA (real P from QK)** → **crash**

This pointed at the QK→PV data flow as the trigger, but the actual fault was in
the **Q load** (executed once before the loop). The PV MFMA merely increased
register pressure enough to change LLVM's instruction scheduling, which moved
the `s_waitcnt vmcnt(0)` — the point where the asynchronous Q load fault is
surfaced — to a position where it was actually reached.

The ROCm debug agent (`librocm-debug-agent.so`) confirmed the faulting PC was
the `s_waitcnt vmcnt(0)` after the 8 Q `buffer_load_dwordx4` instructions.

Three LLVM upgrades (July 23 → Aug 14 → Aug 20) were tested and all exhibited
the same crash, confirming the issue was in the kernel's address computation,
not in the compiler backend.

## Fix

**File:** `kernels/attention/flex_attention_layout_gfx950.py`

Replace the unbounded Q buffer descriptor with one sized to the actual Q tensor:

```python
# Before (crashes):
q_it = _make_buffer_ptr(
    fx.recast_iter(elem_dtype, fx.get_iter(q)) + fx.Int32(q_off)
)

# After (fixed):
_q_total_bytes = num_batches * seqlen_q * fx.Int32(hq * head_dim * param.in_data_bytes)
q_it = _make_buffer_ptr(
    fx.recast_iter(elem_dtype, fx.get_iter(q)),
    num_records_bytes=_q_total_bytes,
)
gQ = fx.make_view(
    q_it + fx.Int32(q_off),
    fx.make_layout((block_m, head_dim), (hq * head_dim, 1)),
)
```

Key changes:

1. **Anchor the buffer descriptor at `q.data_ptr()`** (tensor base) instead of
   `q.data_ptr() + q_off` (per-workgroup base). This keeps the descriptor
   uniform and allows `num_records` to cover the entire tensor.

2. **Set `num_records_bytes`** to the actual Q tensor size
   (`B × S × H × D × sizeof(elem)`). The 4 OOB elements are clamped to zero by
   the hardware instead of generating a real memory access.

3. **Fold `q_off` into the view offset** (`q_it + fx.Int32(q_off)`) so per-lane
   addressing remains correct via the VGPR offset.

4. **Add `num_batches: fx.Int32`** as a kernel parameter (passed from the
   `@flyc.jit` launch wrapper) so the total tensor size can be computed inside
   the kernel.

## Verification

```bash
# Previously crashed:
python3 -c "
import torch; from kernels.attention.flex_attention_layout_gfx950 import *
q=torch.randn(2,8192,32,128,dtype=torch.bfloat16,device='cuda')
k=torch.randn(2,8192,32,128,dtype=torch.bfloat16,device='cuda')
v=torch.randn(2,8192,32,128,dtype=torch.bfloat16,device='cuda')
flydsl_flex_attention_layout(q,k,v,scale=0.088,block_m=16,block_n=16,mma_m=16,mma_k=16,num_groups=2)
torch.cuda.synchronize(); print('OK')
"

# Full test suite (70/71 pass; the 1 failure is a pre-existing causal mask bug):
python3 -m pytest tests/kernels/test_flex_attention_layout.py -v
```
