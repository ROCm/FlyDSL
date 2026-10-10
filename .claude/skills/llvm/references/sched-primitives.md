# Scheduling primitives — `sched_*`, `s_setprio`, `s_waitcnt`

Load this when placing scheduling barriers or wait counters by hand. For the
scheduler *policy* flags (`-amdgpu-sched-strategy` and friends) see
`mllvm-flags.md`.

These are the finest-grained lever FlyDSL has: they constrain the machine
scheduler directly, per instruction group, where a global flag cannot.

## The SchedGroupMask table

⚠ **A typo'd mask does not raise — it silently becomes a full scheduling fence.**
`_mask_to_attr` (`python/flydsl/expr/rocdl/__init__.py:135-137`) builds an empty
parts list from an unrecognised non-zero mask and converts it to
`#rocdl<sched_group_mask none>`. Since `none`/0 is itself a meaningful and
heavily-used value (total fence), your intended *selective* barrier silently
becomes a *total* one. Always verify the resulting schedule changed the way you
intended.

Bit *n* = `1 << n`. Verified against LLVM's `ROCDLEnums.td:120-141`, which matches
FlyDSL's table exactly:

| Value | Keyword | Covers |
|---|---|---|
| `0x000` | `none` | total fence — nothing may cross |
| `0x001` | `non_mem_non_sideeffect` | ALU with no memory or side effects |
| `0x002` | `valu` | vector ALU |
| `0x004` | `salu` | scalar ALU |
| `0x008` | `mfma_wmma` | MFMA / WMMA |
| `0x010` | `all_vmem` | all VMEM (loads **and** stores) |
| `0x020` | `vmem_read` | VMEM loads only |
| `0x040` | `vmem_write` | VMEM stores only |
| `0x080` | `all_ds` | all LDS |
| `0x100` | `ds_read` | LDS reads |
| `0x200` | `ds_write` | LDS writes |
| `0x400` | `transcendental` | transcendental ops |
| `0x800` | `ldsdma` | LDS DMA |

Masks OR together: `0x008 | 0x020` = MFMA plus VMEM reads.

## The two barriers

```python
fx.rocdl.sched_barrier(mask)                        # nothing NOT in mask may cross
fx.rocdl.sched_group_barrier(mask, size, group_id)  # schedule `size` of `mask` as a group
```

`sched_barrier(0)` is the common total fence — used in this tree to pin a
hand-written schedule (`examples/04-preshuffle_gemm.py:132`, and throughout the
gfx950 FMHA kernels).

`sched_group_barrier` is how you interleave: emitting alternating MFMA and
ds_read groups builds a software-pipelined inner loop the scheduler will respect.

### Shorthands

```python
fx.rocdl.sched_mfma(cnt)   # sched_group_barrier(0x008, cnt, 0)
fx.rocdl.sched_vmem(cnt)   # sched_group_barrier(0x020, cnt, 0)
fx.rocdl.sched_dsrd(cnt)   # sched_group_barrier(0x100, cnt, 0)
fx.rocdl.sched_dswr(cnt)   # sched_group_barrier(0x200, cnt, 0)
```

⚠ **`sched_vmem` covers loads only** (`vmem_read` 0x020, not `all_vmem` 0x010).
To schedule `buffer_store_*` you must call `sched_group_barrier(0x040, ...)` or
`0x010` directly.

## `iglp_opt`

⚠ Has **no FlyDSL wrapper** — zero hits repo-wide. It reaches `fx.rocdl` only via
the wildcard re-export of the ODS-generated ROCDL ops, with no arch check, no
test and no in-tree user. `llvm.amdgcn.iglp.opt` applies a canned interleaving
strategy; it and `sched_group_barrier` are mutually exclusive in a region.

## `s_setprio`

Also has no wrapper — kernels call the wildcard-exported builder directly, or the
one-liner `_s_setprio` at `kernels/attention/flash_attn_utils.py:55-56`. Used by
the dualwave FMHA kernels to raise priority around the MFMA block so the other
wave yields. ⚠ No validation of the I16 priority range on the FlyDSL side.

## `s_waitcnt` — arch-dependent, and gfx1250 raises

```python
fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)   # keyword form, preferred
fx.rocdl.s_waitcnt(0)                              # legacy raw bitfield
```

⚠ **One Python call, different bitfield encodings per arch.** The keyword form
dispatches on `get_rocm_arch()`:

| Arch | Encoder | Field widths |
|---|---|---|
| gfx942, gfx950 | `cdna3.py` | vmcnt 6b split, lgkmcnt 4b @ 8, expcnt 3b @ 4 |
| gfx11xx | `rdna3.py` | vmcnt 6b @ 10, lgkmcnt 6b @ 4, expcnt 3b |
| gfx120x | `rdna4.py` | as RDNA4 |
| **gfx1250** | — | ⚠ **raises `ValueError` by design** |

⚠ Note `cdna3.py` serves **both** gfx942 and gfx950 despite gfx950 being CDNA4 —
the file's generation number does not track the target arch.

### Why gfx1250 raises — and why that is correct

[measured] the legacy `llvm.amdgcn.s.waitcnt` intrinsic **aborts the compiler** on
gfx1250 (`LLVM ERROR: Cannot select`, rc=134, core dumped) while the identical IR
compiles cleanly on gfx950. The packed bitfield has no gfx12 encoding.

**Raising a Python error early is strictly better than an ISel crash. Do not
"fix" this.** gfx12 replaced the packed field with split counters:

```python
# gfx12+ : per-class counters
llvm.amdgcn.s.wait.loadcnt / .storecnt / .dscnt / .kmcnt / .expcnt / .samplecnt / .bvhcnt
# gfx1250 only
llvm.amdgcn.s.wait.asynccnt / .tensorcnt
```

`s_wait_xcnt` is **compiler-managed** and has no user intrinsic.

## Async groups

```python
fx.rocdl.asyncmark()          # close the current async group
fx.rocdl.wait_asyncmark(n)    # wait until at most n groups outstanding; 0 drains all
```

Async LDS DMA copies (the `*LoadAsyncLDS` atoms, gfx1250 TDM / async global
loads) are **not tracked by the compiler's automatic wait insertion**. Group and
drain them yourself or you will read garbage.

## Verifying a scheduling change

Scheduling primitives leave no function attribute, so §7's attribute grep does not
apply. Verify by **ISA diff** instead:

```bash
# two dumps, identical except for the barrier
diff <(grep -oE '^\s+[a-z_0-9]+' before/*_final_isa.s) \
     <(grep -oE '^\s+[a-z_0-9]+' after/*_final_isa.s)
```

The instruction *order* must differ. If it does not, the barrier did nothing —
suspect a silently-degraded mask (top of this file) before concluding anything
about the hardware.

⚠ For any ROCDL intrinsic on a new arch, also assert the **mnemonic appears** in
`*_final_isa.s`. FlyDSL's ROCDL layer has no arch guards on most gfx950/gfx1250
entry points, and an unsupported op can be **silently dropped** at ISel rather
than reported — the failure class that got `BufferCopyLDS64b` deprecated.
