# Kernel Walkthrough: `fmha_prefill_fp8_ck_hk5`

Source: `kernels/fmha_prefill_fp8_ck_hk5.py`
Target: AMD gfx942 (MI300/MI308X), wave size 64, FP8 MFMA.

---

## 1. High-Level Intent

FP8 **causal flash-attention prefill** for paged KV cache — the canonical "best" FlyDSL FMHA
kernel. It is a structural port of AMD CK-Tile's `BlockFmhaBatchPrefillPipelineQRKSVSAsync`
pipeline, fused with HK's **LDS row-padding** trick to kill bank conflicts.

It computes, per (batch, q-head), the standard online-softmax attention:

```
O = softmax( (sm_scale · Q · Kᵀ) + causal_mask ) · V        (all in FP8 with per-tensor descales)
```

Both GEMMs use `mfma_f32_32x32x16_fp8_fp8`:
- **GEMM1** `S[kv,q] = K · Qᵀ`  (S laid out as `[kv, q]`)
- **GEMM2** `O[d,q] += Vᵀ · P`  (O laid out as `[d, q]`)

Key design choices inherited from CK and why they matter here:

| Choice | Code | Payoff |
|---|---|---|
| **Q-load-once** into registers, reused across whole KV loop | `q_i64` in `process_qtile` | removes redundant Q re-fetch |
| **Column-major V** (`vec_k_col_v`) | `VCOL`, `v_head_off_col` | GEMM2 contraction dim contiguous → **no V transpose**; one 128-bit store/slot vs 16× `ds_write_b8` scatter (LDS-wait 54%→18%) |
| **Diagonal-pair tiling** (causal load-balancer) | `DIAG`, `process_qtile` twice | each CTA does tile `t` AND its causal mirror `N-1-t` → light+heavy tiles share a workgroup (+8-24% at sq≥2048) |
| **Masked/unmasked loop split** | Phase 1 / Phase 2 loops | interior tiles skip per-element causal-mask VALU (VALU:MFMA 24→19) |
| **LDS row padding** (HK5 win) | `_K_PAD`, `_V_PAD` = 8 | row stride coprime-ish with 32-bank period → bank conflicts 68%→15% of busy, +66-73% at large seq |

Measured (MI308X, bs=1 nq8 nk1 causal): **5 / 18 / 101 / 121 TF** @ sq 1024/2048/16384/32768.
Still ~2.1× behind CK-Tile fp8 — remaining gap is VALU/scheduling-bound (softmax VALU not
overlapped with MFMA), not memory.

---

## 2. Tile / Block Hierarchy

```
Constants:  HD=128 (head dim)   BN=32 (kv per MFMA subtile)   KT=32 (outer kv tile)
            NWAVES=4   WAVE_ROWS=32   TILE_BM=128 (q rows per q-tile)   NSUB=KT/BN=1
            KSTEPS=HD/16=8 (GEMM1 K-steps)   DT=HD/32=4 (GEMM2 d-tiles)
```

```
GRID (1-D)                       one CTA = workgroup of NTHREADS=256 (4 waves × 64 lanes)
 └─ block_idx.x  decoded into:
      first_idx  = blk % num_first         (which q-tile pair this CTA owns)
      qhead      = (blk//num_first) % nq
      batch      = (blk//num_first) // nq
      num_first  = ceil(num_q_tiles / 2)   when DIAG (each CTA handles 2 tiles)

CTA (256 threads)
 ├─ wave 0 → q rows  0..31      ┐
 ├─ wave 1 → q rows 32..63      │ TILE_BM = 128 q rows per q-tile
 ├─ wave 2 → q rows 64..95      │ (each wave owns WAVE_ROWS=32)
 └─ wave 3 → q rows 96..127     ┘

Within a wave (64 lanes):
   q_local = lane % 32   (which of 32 q rows in MFMA fragment)
   half    = lane // 32  (upper/lower 32-lane half → head-dim halving for FP8 MFMA)
```

Per CTA the work is processed by `process_qtile()`, called **once** (non-diag) or **twice**
(diag: `first_idx` then `mirror_qtile = num_q_tiles-1-first_idx`).

---

## 3. Outer Loop Structure

Each `process_qtile(qtile)` walks the KV dimension in **KT-sized outer tiles** with a 2-deep
LDS ping-pong, split into an unmasked phase and a masked phase:

```
process_qtile(qtile):
    load Q once into registers (q_i64[KSTEPS])
    compute per-lane causal bound (eff_bound), tile counts:
        n_kt_rt   = #KT-tiles up to causal/seq bound (skip fully-masked tiles)
        n_unmask  = #KT-tiles fully below diagonal & in-bounds (no mask VALU)

    ── PROLOGUE ──────────────────────────────────────────────
    load_kv_regs(0)            # cooperative global load of KT tile 0
    store_kv_to_lds(→ buf 0)   # K straight store; V straight (col-major)
    gpu.barrier()

    ── PHASE 1: unmasked interior tiles  (kt_iv = 0 .. n_unmask) ──
    scf.for carrying [m_run, l_run, o_acc[DT]]:
        loop_body(kt_iv, do_mask=False)

    ── PHASE 2: masked diagonal + OOB tail (kt_iv = n_unmask .. n_kt_rt) ──
    scf.for carrying same state:
        loop_body(kt_iv, do_mask=True)

    ── EPILOGUE ──────────────────────────────────────────────
    O[d,q] *= v_descale / l_run ; cast bf16 ; buffer_store
```

`loop_body(kt_iv)` is one software-pipelined outer step (prefetch-next / compute-cur):

```
loop_body:
    cur_buf = kt_iv % 2   ;  nxt_buf = (kt_iv+1) % 2     # ping-pong LDS
    load_kv_regs(kv0 + KT)             # OPT3: PREFETCH next tile into VGPRs
    s_setprio(1)
    compute_kt_tile(cur_buf, ...)      # GEMM1 → softmax → P-transpose → GEMM2 on CURRENT tile
    s_setprio(0)
    store_kv_to_lds(→ nxt_buf)         # commit prefetched next tile to LDS
    gpu.barrier()
```

`compute_kt_tile` (the math, over `NSUB` 32-kv subtiles, softmax done **once** per KT tile):

```
1. GEMM1:  for sub in NSUB:  S[kv,q] = Σ_ks K_pack[ks] · Q[ks]   (KSTEPS MFMAs)
2. descale + causal mask:    s = S · (q_descale·sm_scale·k_descale); mask if do_mask
3. softmax (once):           m_new = max(m_run, rowmax);  corr = exp2(m_run-m_new)
                             p = exp2(s - m_new + log2_pscale);  l_run = l_run·corr + Σp
                             o_acc[dt] *= corr
4. P transpose:              cvt_pk_fp8 + ds_bpermute  → P as [q,kv] FP8 packs
5. GEMM2:  for sub, dt:      O[d,q] += Σ_s Vᵀ_pack · P_pack   (loads V from LDS, MFMAs)
```

---

## 4. Memory Access Pattern

### Global → register (cooperative load, `load_kv_regs`)
- 256 threads cooperatively load one KT tile across `NPASS = ceil(NSLOT/NTHREADS)` passes.
  `NSLOT = KT·8` 16-byte slots.
- **K** (vec_k layout `[pages, nk, hd/16, ps, 16]`): kv-major, vectorized `buffer_load`
  `vec_width=4` (128-bit). Page table indirection: `kphys = LTD[page0 + kvrow/page_size]`.
- **V column-major** (`[pages, nk, hd, ps]`, default `VCOL=1`): loads **16 contiguous kv**
  for a fixed head-dim `d` — already GEMM2-ready, no transpose.
- Bounds: out-of-range rows clamped to row 0 (`select`), masked later.

### Register → LDS (`store_kv_to_lds`)
- **K**: one 128-bit `Vector.store` per pass into `k_lds` at row stride `_K_LDSW = HD + 8`.
- **V (col)**: one 128-bit straight store per pass into `vt_lds` at row stride `_V_LDSW = KT + 8`.
  (Row-major fallback path does the 16× `ds_write_b8` scatter-transpose — avoided here.)

### LDS layout & the padding trick
```
_K_LDSW = HD + 8 = 136 bytes/row   (K tile = KT × 136)
_V_LDSW = KT + 8 = 40  bytes/row   (V tile = HD × 40)
NBUF = 2  → ping-pong:  [K buf0][K buf1][V buf0][V buf1]
```
Without padding, K rows have stride 128 B = 32 banks × 4 B, so consecutive kv rows alias to the
**same** bank → up to 32-way `ds_read` conflict. Padding each row +8 B makes the stride
coprime-ish with the 128-B/32-bank period, spreading rows across banks. Bank conflicts dropped
68% → 15% of busy cycles. (8/8 is the swept optimum; the response is non-monotonic.)

### LDS → register (in `compute_kt_tile`)
- **K**: `Vector.load` vec(8,i8) → bitcast i64, per `KSTEPS`, at `_K_LDSW` row stride.
- **V**: `Vector.load` vec(8,i8) → i64, per `DT × 2`, at `_V_LDSW` row stride.
- `sched_dsrd` hints group the `ds_read`s; `sched_mfma(1)` paces the MFMA issue.

### P transpose (register-only, no LDS)
- `cvt_pk_fp8_f32` packs softmax probs to FP8, then `ds_bpermute` (lane shuffle) transposes
  `P[kv,q] → P[q,kv]` across the 64-lane wave. `_wait_lds()` after the bpermutes.

### Register → global (epilogue)
- `O[d,q]` rescaled by `v_descale / l_run`, cast to bf16, stored via `buffer_store` of 4-wide
  bf16 vectors to `O[batch, qrow, qhead, d]`. Guarded by `in_b = qrow < sq`.

---

## 5. ASCII Visualization

### Diagonal-pair grid mapping (DIAG=1)
```
q-tiles along the causal diagonal (light work top, heavy work bottom):

  tile 0  ▏░                 ┐ paired:  CTA handles tile 0  AND  tile N-1
  tile 1  ▏░░                │          CTA handles tile 1  AND  tile N-2
  tile 2  ▏░░░               │  ...     balances light(early)+heavy(late)
   ...    ▏░░░░              │
  tile N-2▏░░░░░░░░░░░░░░    │
  tile N-1▏░░░░░░░░░░░░░░░   ┘
          └ kv (causal: tile t attends kv ≤ its diagonal) ┘

num_first = ceil(N/2) CTAs per (batch,qhead)
```

### One CTA, one q-tile: data flow per outer KT step
```
        GLOBAL (paged KV)                         REGISTERS                LDS (padded)
   ┌─────────────────────┐   load_kv_regs    ┌──────────────┐  store   ┌──────────────┐
   │ K [pages,nk,hd/16,..]│ ───(prefetch)───▶ │ kc[NPASS] 128b│ ───────▶ │ k_lds  KT×136│
   │ V [pages,nk,hd,ps]   │ ───(col-major)──▶ │ vc[NPASS] 128b│ ───────▶ │ vt_lds HD×40 │
   └─────────────────────┘                   └──────────────┘          └──────┬───────┘
   ┌─────────────────────┐                                                    │ ds_read
   │ Q [b,sq,nq,hd]       │ ──load ONCE──▶ q_i64[KSTEPS] (reused all KV) ◀─────┘
   └─────────────────────┘                          │
                                                     ▼
   GEMM1  S[kv,q] = K·Qᵀ   (KSTEPS × mfma_32x32x16_fp8)
                                                     │
                                                     ▼
   descale·mask → softmax(once) → m_run,l_run,corr ; o_acc[DT]*=corr
                                                     │
                                                     ▼
   P[kv,q] ──cvt_pk_fp8 + ds_bpermute──▶ P[q,kv] (FP8, register-only transpose)
                                                     │
                                                     ▼
   GEMM2  O[d,q] += Vᵀ·P   (DT×2 × mfma_32x32x16_fp8)   ──▶ o_acc[DT] (f32 accum)
```

### Wave / lane fragment layout (per 64-lane wave)
```
                 q dimension (32 rows owned by wave)
                 q_local = lane % 32  ─────────────►
 head-dim half   ┌────────────────────────────────┐
 half=lane//32   │  FP8 32×32×16 MFMA fragment      │
   half 0 ──────▶│  (lanes  0..31)                  │
   half 1 ──────▶│  (lanes 32..63)                  │
                 └────────────────────────────────┘
  GEMM1: A=K_pack(i64)  B=Q(i64)   → acc f32×16  (S as [kv,q])
  GEMM2: A=V_pack(i64)  B=P(i64)   → o_acc f32×16 (O as [d,q])
```

### Two-phase KV loop (causal)
```
 kt_iv:  0 ─────────── n_unmask ─────────── n_kt_rt
         │  PHASE 1     │      PHASE 2        │
         │  unmasked    │   masked diagonal   │   (tiles > n_kt_rt fully masked: skipped)
         │  (no mask    │   + OOB tail        │
         │   VALU)      │   (per-elem mask)   │
         ▼              ▼                     ▼
   loop_body: prefetch next ▸ compute cur ▸ store next ▸ barrier   (LDS ping-pong buf = kt_iv%2)
```

---

## Tunables (env)

| Env | Default | Effect |
|---|---|---|
| `FMHA_NWAVES` | 4 | waves/wg → `TILE_BM = NWAVES·32` (≠4 regresses) |
| `FMHA_KT` | 32 | outer kv tile (>32 regresses occupancy) |
| `FMHA_VCOL` | 1 | column-major V (no transpose) — key win |
| `FMHA_DIAG` | 1 | diagonal-pair causal load-balancer |
| `FMHA_KPAD` / `FMHA_VPAD` | 8 / 8 | LDS row padding bytes (bank-conflict fix; swept optimum) |
| `FMHA_BUFK` | 0 | async `buffer_load_to_lds` for K — **broken** in flydsl 0.2.0 |

**Resources:** VGPR 165, LDS 16 KB (18944 B at KPAD=VPAD=8), 0 spills, occupancy VGPR-pinned
at 3 waves/SIMD.

**Remaining ~2.1× gap to CK-Tile** is VALU/scheduling-bound: softmax VALU sits between the two
MFMA bursts with no independent MFMA to hide it; FlyDSL 0.2.0's scheduler won't auto-overlap.
Next levers: manual cross-tile pipelining (interleave GEMM1(i+1) into softmax(i)), external-LLVM
VGPR cap for 4 waves/SIMD, a fixed `buffer_load_to_lds`, or gfx950 `ds_read_tr`.
