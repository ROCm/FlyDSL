	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs,"axG",@progbits,_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs,comdat
	.protected	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs
	.globl	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs
	.p2align	8
	.type	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs,@function
_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs:
; OPUS ISA ANNOTATION
; Source map: FlyDSL/opus_attn/gqa_d128_kernel_template.hpp, kernel body around C++ lines 296-754.
; Notation:
;   ; HPP  = nearby source-level C++ operation lowered into this ISA region.
;   ; NOTE = lowering/scheduling note; LLVM may reorder, combine, or split C++ operations.
; Only comments have been added to this dump. Executable ISA instructions are intentionally unchanged.
;
; Template instantiation in this dump:
;   gqa_d128_kernel<opus_gqa_traits<32, 64, 128, 8, true>>(opus_gqa_kargs)
;   Q_TILE_SIZE=32, KV_TILE_SIZE=64, D_TILE_SIZE=128, NUM_WARPS=8 (BLOCK_SIZE=512), CAUSAL=true.
;   D_ATTN=bf16, D_ACC=fp32; MFMA shape 32x32x16 bf16 (W_M=W_N=32, W_K=16).
;   GEMM0_E_M=1, GEMM0_E_N=2, GEMM0_E_K=8;  GEMM1_E_M=1, GEMM1_E_N=4, GEMM1_E_K=4.
;   Per-wave s_len=32, s_half_len=16, o_len=16, q_len=8.
;   VEC_Q=8, VEC_KV=8, VEC_TR_V=4, VEC_O=4.
;   k_buffer_load_insts = v_buffer_load_insts = 2;  k_ds_read_insts = 16, v_ds_read_insts = 32.
;
; OPUS high-level flow:
;   1. Kernel setup derives workgroup/q-block/batch coordinates, GQA q-head to kv-head mapping
;      (workgroup_x permutation), Q/K/V/O global memory offsets, lane/warp ids, and the SMEM
;      ping-pong addressing for s_k[2] and s_v[2].
;   2. Prologue prefetches K[0] into LDS + s_waitcnt(0) + s_barrier; loads Q from gmem,
;      scales Q by (1/sqrt(D)) * log2(e); prefetches K[1] + V[0] async; runs the first MMA0
;      with causal mask for tile 0, computes m_row + sub_row + exp2 first half;
;      prefetches K[2] async.
;   3. The steady-state main loop `for (j=3; j<max_num_tiles-1; j+=2)` pipelines pairs of KV
;      tiles in 8 clusters per iteration:
;        Cluster 0: async V[j-2] + K[1] LDS read + waitcnt + s_barrier
;        Cluster 1: MMA0 + exp2 second half + attn_sum + cast P + sched fences + s_barrier
;        Cluster 2: async K[j] + V[0] LDS tr_load + waitcnt + s_barrier
;        Cluster 3: setprio 1 + 1st MMA1 step + attn_row_max + lazy-rescale test
;                   + 3 remaining MMA1 steps + sub_row + exp2 first half
;                   + setprio 0 + s_barrier
;        Cluster 4-7: same shape with V/K buffer flipped and causal mask in Cluster 6.
;   4. The epilogue drains the final 3 KV tiles in 14 clusters (Ep C0..C13) without
;      the all-below shortcut (always rescales), then normalizes O by 1/l_row using
;      a (l_row > 0) ? rcp : 0 guard, packs the fp32 accumulators to bf16, and stores to gmem.
;
; LLVM SCHED HINT LEGEND
;   Source `__builtin_amdgcn_sched_group_barrier(mask, size, groupId)` (used directly and via
;   the `sched_barrier_pairs` / `sched_barrier_exp_pairs` helpers at HPP lines 18-30) lowers
;   to the `llvm.amdgcn.sched.group.barrier` intrinsic and then to the AMDGPU
;   `SCHED_GROUP_BARRIER` pseudo. The pseudo is consumed by IGroupLP scheduling and is not
;   emitted as real ISA.
;   Masks used by this kernel:
;     0x008 = MFMA/WMMA, 0x002 = VALU, 0x400 = TRANS/EXP (`v_exp_f32`).
;   Directionality:
;     `SCHED_GROUP_BARRIER` scans upward from the marker to earlier SUnits
;     (`initSchedGroupBarrierPipelineStage` calls `findCandidateSUnits(RIter,
;     SUnits.rend(), ...)`). It groups already-emitted candidate instructions,
;     not future instructions below the marker. Same `groupId` barriers are
;     solved together as one pipeline by `PipelineSolver`.
;   Source `__builtin_amdgcn_sched_barrier(0)` lowers to `SCHED_BARRIER 0`. In this LLVM
;   tree, `SIInstrInfo::isSchedulingBoundary()` returns true for `SCHED_BARRIER` with
;   immediate 0, so no real instruction, including `S_BARRIER`, may be moved across that
;   scheduling boundary. The pseudo is removed before final ISA emission; only the
;   resulting ordering remains.

; Setup: kargs load + block/warp/lane IDs.
; HPP:
;   const int workgroup_x = block_id_x();
;   const int q_block_idx = block_id_y();
;   const int b = block_id_z();
;   const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
;   const int lane_id = thread_id_x() % T::WARP_SIZE;
;   const int stagger = warp_id / 4;
; NOTE:
;   - `s_load_dwordx8 s[16:23] / s[8:15]` and `s_load_dwordx2 s[24:25]` load the
;     `opus_gqa_kargs` struct from the kernel argument segment: stride/B/N/H/H_KV/D
;     fields land in s[16:25] and the four base pointers (ptr_q, ptr_k, ptr_v, ptr_o)
;     land in s[8:15]. `s[0:1]` is the kernarg segment base preserved by the
;     dispatcher; `s2`/`s3`/`s4` carry block_id_x/y/z; `v0` carries thread_id_x.
;   - `v_readfirstlane_b32 s31, v0` then `s_lshr_b32 s30, s31, 6` extracts the
;     wave-id `warp_id` (= thread_id_x() / WARP_SIZE) into s30 (a SGPR because of
;     the `__builtin_amdgcn_readfirstlane` source).
;   - `v_and_b32_e32 v114, 7, v0` extracts `lane_id & 7` (used later for the
;     `lane_id % T::W_M`-style layout coordinate; here W_M = 32 but the per-wave
;     N-coord uses `lane_id & 7` for the bf16 read groups).
;   - `s_waitcnt lgkmcnt(0)` waits for the s_load_dwordx{2,8} to complete before
;     any scalar uses the kargs fields.
	s_load_dwordx8 s[16:23], s[0:1], 0x24
	s_load_dwordx8 s[8:15], s[0:1], 0x0
	s_load_dwordx2 s[24:25], s[0:1], 0x44
	v_readfirstlane_b32 s31, v0
	s_lshr_b32 s30, s31, 6
	v_and_b32_e32 v114, 7, v0
	s_waitcnt lgkmcnt(0)

; Setup: GQA q-head to kv-head mapping (workgroup_x permutation).
; HPP:
;   const int group_size = kargs.H / kargs.H_KV;
;   const int h = (workgroup_x % kargs.H_KV) * group_size + (workgroup_x / kargs.H_KV);
;   const int h_kv = h / group_size;
;   const int q_block_size = T::NUM_WARPS * T::Q_TILE_SIZE;
;   const int q_block_start = q_block_idx * q_block_size;
; NOTE:
;   GQA reorders `workgroup_x` so the H_KV q-heads sharing one kv-head are
;   contiguous in workgroup_x. LLVM materialises the integer divisions with the
;   classic multiply-by-reciprocal sequence (Hacker's Delight ch.10):
;     - `s_abs_i32 s5, s18` then `v_cvt_f32_u32_e32 v1, s5` + `v_rcp_iflag_f32_e32`
;       + `v_mul_f32_e32 v1, 0x4f7ffffe, v1` + `v_cvt_u32_f32_e32` builds the
;       32-bit reciprocal magic number for `H_KV`.
;     - `s_mul_hi_u32` / `s_mul_i32` / `s_sub_i32` / `s_cselect_b32` realises
;       `workgroup_x / H_KV` (quotient s1 = h-within-group) and
;       `workgroup_x % H_KV` (remainder s5 used as the kv-group index).
;     - The same magic-number sequence is repeated for `h / group_size = h_kv`
;       (second divmod producing s2 = h_kv).
;   `s_mul_i32 s27, s30, 0x410` and `s_mov_b32 m0, s27` compute the per-warp
;   SMEM base offset (0x410 = 1040 bytes = `smem_linear_wave + smem_padding_16B`
;   = 64*16 + 16 padding = K wave-stripe per warp) and set m0 for the first
;   `buffer_load_dwordx4 ... lds` issued later in the prologue.
;   `v_and_b32_e32 v113, 31, v0` extracts `lane_id % T::W_M` (W_M = 32) for the
;   per-lane wave-tile column coordinate.
;   `v_bfe_u32 v209, v0, 5, 1` extracts `lane_id / W_M` (single-bit lane group)
;   used as the y-coordinate in the wave-tile partition.
	s_abs_i32 s5, s18
	v_cvt_f32_u32_e32 v1, s5
	s_sub_i32 s6, 0, s5
	s_abs_i32 s1, s17
	s_xor_b32 s0, s17, s18
	v_rcp_iflag_f32_e32 v1, v1
	s_ashr_i32 s0, s0, 31
	s_mul_i32 s27, s30, 0x410
	s_mov_b32 m0, s27
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_and_b32_e32 v113, 31, v0
	v_bfe_u32 v209, v0, 5, 1
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s6, s6, s7
	s_mul_hi_u32 s6, s7, s6
	s_add_i32 s7, s7, s6
	s_mul_hi_u32 s6, s1, s7
	s_mul_i32 s17, s6, s5
	s_sub_i32 s1, s1, s17
	s_add_i32 s26, s6, 1
	s_sub_i32 s17, s1, s5
	s_cmp_ge_u32 s1, s5
	s_cselect_b32 s6, s26, s6
	s_cselect_b32 s1, s17, s1
	s_add_i32 s17, s6, 1
	s_cmp_ge_u32 s1, s5
	s_cselect_b32 s1, s17, s6
	s_abs_i32 s6, s2
	s_mul_hi_u32 s7, s6, s7
	s_xor_b32 s1, s1, s0
	s_mul_i32 s17, s7, s5
	s_sub_i32 s0, s1, s0
	s_xor_b32 s1, s2, s18
	s_sub_i32 s6, s6, s17
	s_ashr_i32 s1, s1, 31
	s_add_i32 s17, s7, 1
	s_sub_i32 s26, s6, s5
	s_cmp_ge_u32 s6, s5
	s_cselect_b32 s7, s17, s7
	s_cselect_b32 s6, s26, s6
	s_add_i32 s17, s7, 1
	s_cmp_ge_u32 s6, s5
	s_cselect_b32 s5, s17, s7
	s_abs_i32 s6, s0
	v_cvt_f32_u32_e32 v1, s6
	s_xor_b32 s5, s5, s1
	s_sub_i32 s1, s5, s1
	s_mul_i32 s5, s1, s18
	v_rcp_iflag_f32_e32 v1, v1
	s_sub_i32 s2, s2, s5
	s_sub_i32 s5, 0, s6
	s_mul_i32 s2, s2, s0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s1, s2, s1
	s_abs_i32 s2, s1
	s_xor_b32 s0, s1, s0
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s5, s5, s7
	s_mul_hi_u32 s5, s7, s5
	s_add_i32 s7, s7, s5
	s_mul_hi_u32 s5, s2, s7
	s_mul_i32 s7, s5, s6
	s_sub_i32 s2, s2, s7
	s_ashr_i32 s0, s0, 31
	s_add_i32 s7, s5, 1
	s_sub_i32 s17, s2, s6
	s_cmp_ge_u32 s2, s6
	s_cselect_b32 s5, s7, s5
	s_cselect_b32 s2, s17, s2
	s_add_i32 s7, s5, 1
	s_cmp_ge_u32 s2, s6
	s_cselect_b32 s2, s7, s5
	s_xor_b32 s2, s2, s0

; Setup: q_block_start + gmem offsets for Q/O and K/V.
; HPP:
;   const int q_block_size = T::NUM_WARPS * T::Q_TILE_SIZE;
;   const int q_block_start = q_block_idx * q_block_size;
;   const int qo_gmem_offset = b * kargs.stride_q_b + q_block_start * kargs.stride_q_n + h * kargs.stride_q_h;
;   const int kv_gmem_offset = b * kargs.stride_kv_b + h_kv * kargs.stride_kv_h;
; NOTE:
;   - `s_lshl_b32 s18, s3, 8` = `q_block_start = q_block_idx * 256` because
;     q_block_size = NUM_WARPS * Q_TILE_SIZE = 8 * 32 = 256.
;   - The chain `s_mul_i32 s0, s20, s4` (= b * stride_q_b), `s_mul_i32 s3, s21,
;     s18` (= stride_q_n * q_block_start), `s_mul_i32 s1, s1, s22` (= h *
;     stride_q_h), and the two `s_add_i32` accumulations form `qo_gmem_offset`
;     in s0. (s4 carries `b` = block_id_z(); see post-divmod register naming.)
;   - `s_mul_i32 s1, s23, s4` (= b * stride_kv_b) + `s_mul_i32 s2, s2, s25`
;     (= h_kv * stride_kv_h) accumulate into `kv_gmem_offset` in s4.
;   `stride_q_h` lives in s22, `stride_q_n` in s21, `stride_kv_h` in s25,
;   `stride_kv_b` in s23, `stride_q_b` in s20 (struct field order from kargs).
	s_lshl_b32 s18, s3, 8
	s_sub_i32 s2, s2, s0
	s_mul_i32 s0, s20, s4
	s_mul_i32 s3, s21, s18
	s_add_i32 s0, s3, s0
	s_mul_i32 s1, s1, s22
	s_add_i32 s0, s0, s1
	s_mul_i32 s1, s23, s4
	s_mul_i32 s2, s2, s25
	s_add_i32 s4, s2, s1

; Setup: build gmem buffer descriptors for Q/O and K/V.
; HPP:
;   auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_q) + qo_gmem_offset);
;   auto g_k = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_k) + kv_gmem_offset);
;   auto g_v = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_v) + kv_gmem_offset);
;   auto g_o = make_gmem(reinterpret_cast<D_ATTN*>(kargs.ptr_o) + qo_gmem_offset);
; NOTE:
;   - `s_ashr_i32 s1, s0, 31` + `s_lshl_b64 s[22:23], s[0:1], 1` sign-extends
;     `qo_gmem_offset` to 64-bit and multiplies by 2 (bf16 element size). The
;     same pattern (sign-extend + shift) is applied to `kv_gmem_offset` for
;     `s[4:5]`.
;   - `s_add_u32 s0, s8, s22` + `s_addc_u32 s1, s9, s23` form
;     `g_q_base = ptr_q + qo_gmem_offset*2` (s[8:9] = ptr_q, hi-32 of ptr_q
;     was loaded along with ptr_k since they form a dwordx8 pack).
;   - `s_add_u32 s8, s10, s4` (= ptr_k + kv_gmem_offset*2) and the symmetric
;     pair for ptr_v form `g_k_base` (s[8:9]) and the later `g_v_base`.
;   - `s_mov_b32 s3, 0x20000` and `s_mov_b32 s2, -1` set the upper two dwords
;     of the AMDGCN BUFFER_RSRC for Q (s2 = -1 = stride/numrecords-MSBs, s3 =
;     format/swizzle bits). The Q/O buffer descriptor is s[0:3]; K/V is
;     s[8:11]; V (after later concat) uses s[4:7].
;   - `v_and_b32_e32 v1, 56, v0` + `v_add_u32_e32 v1, s30, v1` + `v_mul_lo_u32
;     v1, v1, s24` constructs the per-lane Q-row offset:
;       `((lane_id & 0x38) + warp_id) * stride_q_n` where `s30 = warp_id` and
;       `s24 = stride_q_n`. The lane bits 3-5 form the row index within the
;       wave's 32-row Q tile (a Q-row per group of 8 lanes).
;   - `v_lshl_add_u32 v211, v114, 4, v1` then forms the LDS write address for
;     the K async load: `v211 = lane_in_block * 16 + per_lane_q_offset`.
;     (v114 = `lane_id & 7` from earlier; 16 = 16-byte bf16 vec per dwordx4.)
;   - `s_add_i32 s28, s27, 0x2080` derives the SMEM LDS offset for the second
;     buffer_load (split across 2 dwordx4 instructions): 0x2080 = 8320 bytes
;     past the K base = the second half of the K tile in s_k[0].
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[22:23], s[0:1], 1
	s_add_u32 s0, s8, s22
	s_addc_u32 s1, s9, s23
	s_ashr_i32 s5, s4, 31
	s_and_b32 s1, s1, 0xffff
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s8, s10, s4
	v_and_b32_e32 v1, 56, v0
	s_addc_u32 s6, s11, s5
	v_add_u32_e32 v1, s30, v1
	s_and_b32 s9, s6, 0xffff
	v_mul_lo_u32 v1, v1, s24
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	s_add_u32 s4, s12, s4
	v_lshlrev_b32_e32 v1, 1, v1
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_addc_u32 s5, s13, s5
	v_lshl_add_u32 v211, v114, 4, v1
	s_add_i32 s28, s27, 0x2080

; Prologue: prime LDS with the first K tile, then synchronize before Q load.
; HPP:
;   async_load<T::VEC_KV>(g_k, s_k[0].ptr, u_gk, u_sk, kv_tile(0));
;   __builtin_amdgcn_s_waitcnt(0);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
; NOTE:
;   `m0` selects the LDS write base for `buffer_load ... lds` async copies.
;   Two `buffer_load_dwordx4 v211/v212, s[8:11], 0 offen lds` issue the K[0]
;   async copy. The K-tile is 64 (KV_TILE_SIZE) * 128 (D) bf16 = 16KB; with
;   BLOCK_SIZE=512 lanes and VEC_KV=8 (16 bytes/lane), each lane writes
;   `KV_TILE_SIZE * D / (BLOCK_SIZE * VEC_KV) = 64*128/(512*8) = 2` packs into
;   LDS, hence two `buffer_load_dwordx4 ... lds` instructions per lane.
;   `s_mov_b32 m0, s27` set the K[0] LDS base; the second pack uses
;   `m0 = s28 = s27 + 0x2080` (advance by 8320 bytes = second wave-row tile).
;   `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` is the source-level `s_waitcnt(0)`
;   that drains all outstanding memory ops before the synchronizing `s_barrier`
;   below.
	buffer_load_dwordx4 v211, s[8:11], 0 offen lds
	v_add_u32_e32 v212, 0x80, v211
	s_mov_b32 m0, s28
	s_and_b32 s5, s5, 0xffff
	buffer_load_dwordx4 v212, s[8:11], 0 offen lds
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	s_movk_i32 s13, 0x410
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; Prologue: load the 32x128 Q tile from global memory.
; HPP:
;   v_q = load<T::VEC_Q>(g_q, u_q);
; NOTE:
;   - `v_mul_lo_u32 v210, s21, v113` + `v_lshl_add_u32 v1, v209, 3, v210`
;     forms the per-lane Q-element gmem address using the `make_layout_q`
;     descriptor: lane_id % T::W_M (v113) drives stride_q_n; lane_id / T::W_M
;     (v209) drives the column micro-tile; warp_id (folded via s12 =
;     stride_q_n * warp_id * 32) drives the row macro-tile.
;   - `v_add_lshl_u32 v1, v1, s12, 1` adds the warp row offset and scales by
;     bf16 byte size (shift-left 1 = *2).
;   - `s_barrier` matches the prologue `s_barrier()` from HPP line 401, paired
;     with the eventual barrier in the staggered warp group.
;   - Eight `buffer_load_dwordx4 v[X:X+3], v1, s[0:3], 0 offen offset:Y` lines
;     are the unrolled `load<VEC_Q>(g_q, u_q)`: Q_TILE_SIZE = 32 rows * D=128
;     bf16 / (BLOCK_SIZE=512 lanes * VEC_Q=8 elts/lane) = 1 dwordx4 per lane
;     for the wave-row, but the `make_layout_q` shape includes 8 column
;     micro-tiles per lane group → 8 packs spaced by 32 bytes.
	s_mul_i32 s12, s21, s30
	v_mul_lo_u32 v210, s21, v113
	s_lshl_b32 s12, s12, 5
	v_lshl_add_u32 v1, v209, 3, v210
	v_add_lshl_u32 v1, v1, s12, 1
	s_barrier
	buffer_load_dwordx4 v[14:17], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v1, s[0:3], 0 offen offset:32
	buffer_load_dwordx4 v[96:99], v1, s[0:3], 0 offen offset:64
	buffer_load_dwordx4 v[84:87], v1, s[0:3], 0 offen offset:96
	buffer_load_dwordx4 v[72:75], v1, s[0:3], 0 offen offset:128
	buffer_load_dwordx4 v[60:63], v1, s[0:3], 0 offen offset:160
	buffer_load_dwordx4 v[48:51], v1, s[0:3], 0 offen offset:192
	buffer_load_dwordx4 v[18:21], v1, s[0:3], 0 offen offset:224

; Prologue: prefetch K[1] + V[0] async to LDS, read K[0] from LDS into v_k.
; HPP:
;   async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(1));
;   async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(0));
;   v_k = load<T::VEC_KV>(s_k[0], u_rk);
;   __builtin_amdgcn_sched_barrier(0);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
; NOTE:
;   - `s_add_i32 s26, s27, 0x8500` derives the K[1] LDS base
;     (s_k[1] = s_k[0] + smem_buffer_elems * 2 = 0x8500 bytes later); a paired
;     `s_add_i32 s25, s26, 0x2080` builds the second-half offset for the
;     dwordx8 split.
;   - `s_lshl_b32 s29, s24, 7` builds `kv_tile(1) = 1 * KV_TILE_SIZE *
;     stride_kv_n = 64 * stride_kv_n`, used as the gmem byte offset for the
;     two K[1] `buffer_load_dwordx4 ... lds` calls.
;   - `s_add_i32 s21, s17, 0x4100` and `s_add_i32 s20, s21, 0x2200` derive the
;     V[0] LDS base and second-half offset (V[0] starts at smem_k_tile_elems
;     bytes past s_k[0] base; 0x4100/0x2200 are the per-warp wave-row strides
;     for V which uses 64-byte padding instead of 16-byte).
;   - The four `buffer_load_dwordx4` (two for K[1], two for V[0]) use the
;     concatenated `s[4:7]` resource for V (built earlier from ptr_v).
;   - `v_lshlrev_b32_e32 v1, 4, v0` + `v_lshlrev_b32_e32 v208, 4, v209` +
;     `v_or_b32_e32 v215, v208, v218` + `v_mad_u32_u24 v1, v114, s13, v215`
;     builds the LDS read base address per the `make_layout_rk` descriptor
;     (s13 = smem_linear_wave + smem_padding_16B = 0x410 = 1040).
;   - 16 `ds_read_b128 v[X:X+3], v1 offset:Y` form `v_k = load<VEC_KV>(s_k[0],
;     u_rk)`. Per HPP defs k_ds_read_insts = (GEMM0_E_N * GEMM0_E_K * W_N *
;     W_K) / (WARP_SIZE * VEC_KV) = (2*8*32*16)/(64*8) = 16. The offsets cover
;     two `D_128B_SIZE = 64` blocks within s_k[0]: low half (offsets
;     0/32/64/96 and 8320/8352/8384/8416) for GEMM0_E_N=0 wave-tile and
;     +512 (offsets 512/544/576/608 and 8832/8864/8896/8928) for
;     GEMM0_E_N=1 wave-tile.
	s_add_i32 s26, s27, 0x8500
	s_lshl_b32 s29, s24, 7
	s_mov_b32 m0, s26
	s_add_i32 s25, s26, 0x2080
	s_mul_i32 s17, s30, 0x440
	buffer_load_dwordx4 v211, s[8:11], s29 offen lds
	s_mov_b32 m0, s25
	s_add_i32 s21, s17, 0x4100
	buffer_load_dwordx4 v212, s[8:11], s29 offen lds
	s_mov_b32 m0, s21
	s_add_i32 s20, s21, 0x2200
	buffer_load_dwordx4 v211, s[4:7], 0 offen lds
	s_mov_b32 m0, s20
	v_lshlrev_b32_e32 v1, 4, v0
	buffer_load_dwordx4 v212, s[4:7], 0 offen lds
	v_lshlrev_b32_e32 v208, 4, v209
	v_and_b32_e32 v218, 0x180, v1
	v_or_b32_e32 v215, v208, v218
	v_mad_u32_u24 v1, v114, s13, v215
	ds_read_b128 v[10:13], v1
	ds_read_b128 v[2:5], v1 offset:32
	ds_read_b128 v[104:107], v1 offset:64
	ds_read_b128 v[92:95], v1 offset:96
	ds_read_b128 v[80:83], v1 offset:8320
	ds_read_b128 v[68:71], v1 offset:8352
	ds_read_b128 v[56:59], v1 offset:8384
	ds_read_b128 v[22:25], v1 offset:8416
	ds_read_b128 v[6:9], v1 offset:512
	ds_read_b128 v[108:111], v1 offset:544
	ds_read_b128 v[100:103], v1 offset:576
	ds_read_b128 v[88:91], v1 offset:608
	ds_read_b128 v[76:79], v1 offset:8832
	ds_read_b128 v[64:67], v1 offset:8864
	ds_read_b128 v[52:55], v1 offset:8896
	ds_read_b128 v[26:29], v1 offset:8928

; Prologue: stagger barrier (warps 0-3 skip, warps 4-7 wait).
; HPP:
;   if (stagger) {
;       __builtin_amdgcn_sched_barrier(0);
;       __builtin_amdgcn_s_barrier();
;   }
; NOTE:
;   - `s_cmpk_lt_u32 s31, 0x100` tests `thread_id_x < 256`. Recall s31 was
;     `v_readfirstlane_b32 s31, v0` from setup, so s31 = thread_id_x of lane 0.
;     With BLOCK_SIZE=512 split across 8 warps, lane 0 of warps 0-3 has
;     thread_id_x in {0, 64, 128, 192} (all < 256) and warps 4-7 have
;     thread_id_x in {256, 320, 384, 448} (all >= 256). So this implements
;     `stagger = warp_id / 4` and branches when stagger == 1 (warps 4-7) so
;     that those warps execute the extra `s_barrier`.
;   - `s_waitcnt vmcnt(2) lgkmcnt(0)` waits for the V[0] buffer_load to finish
;     (vmcnt(2) leaves K[1]'s 2 outstanding loads) AND for all LDS reads to
;     complete before the conditional barrier.
;   - Warps 0-3 fall through directly to `.LBB0_2`; warps 4-7 execute the
;     `s_barrier` first. This produces the dual-group phase-shift used by the
;     loop's `s_setprio 0/1` time-multiplex strategy in the main loop.
	s_cmpk_lt_u32 s31, 0x100
	s_cselect_b64 s[0:1], -1, 0
	s_and_b64 vcc, exec, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:

; Prologue: scale Q by (1/sqrt(D))*log2(e) and compute the first MMA0 score tile.
; HPP:
;   auto v_q_f32 = opus::cast<float>(v_q);
;   static_for<q_len>([&](auto i) { v_q_f32[i.value] *= temperature_scale; });
;   v_q = opus::cast<D_ATTN>(v_q_f32);
;   v_s[0] = mma0(v_q, v_k);
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - `v_cvt_f32_i32_e32 v1, s19` + `v_rsq_f32_e32 v1, v1` computes
;     `1.0f / sqrtf((float)D)` (D=128 → 1/sqrt(128) = 0.0883883). Then
;     `v_mul_f32_e32 v112, 0x3fb8aa3b, v1` multiplies by
;     `0x3fb8aa3b = 1.44269502 = log2(e)`, giving `temperature_scale` in v112
;     (with v113 implicitly carrying the same value for pk_mul use).
;   - The Q gmem load delivered bf16 packed in v[14:17], v[30:33], v[96:99],
;     v[84:87], v[72:75], v[60:63], v[48:51], v[18:21] (8 dwordx4 = 32 bf16
;     pairs per lane = the lane's slice of the 32-row, 128-col Q tile).
;   - Each Q pack is unpacked to fp32 via the pattern
;     `v_and_b32_e32 vH, 0xffff0000, vX` (mask hi-half = even-index fp32)
;     and `v_lshlrev_b32_e32 vL, 16, vX` (shift lo-half = odd-index fp32),
;     then `v_pk_mul_f32 v[L:H], v[112:113], v[L:H] op_sel_hi:[0,1]` scales
;     both lanes of the pack by temperature_scale, and finally
;     `v_cvt_pk_bf16_f32 v144..v175, vL, vH` packs the scaled pair back to
;     bf16. This produces v_q_bf16 in v[144:175] (32 dwords = 64 bf16).
;   - Interleaved with the Q scaling, 16 `v_mfma_f32_32x32x16_bf16 v[X:X+15],
;     v[A:A+3], v[B:B+3], v[Y:Y+15]` instructions form the GEMM0 chain. With
;     GEMM0_E_M=1, GEMM0_E_N=2, GEMM0_E_K=8 → 1*2*8 = 16 MFMA per tile,
;     accumulating into the two halves of v_s[0]:
;       * v[32:47] (first 16 elements, GEMM0_E_N=0)
;       * v[2:17]  (second 16 elements, GEMM0_E_N=1)
;     The first MFMA of each accumulator uses immediate operand `0` to zero
;     the C register; subsequent ones accumulate.
	v_cvt_f32_i32_e32 v1, s19
	v_and_b32_e32 v117, 0xffff0000, v33
	v_lshlrev_b32_e32 v116, 16, v33
	v_and_b32_e32 v119, 0xffff0000, v32
	v_rsq_f32_e32 v1, v1
	v_lshlrev_b32_e32 v118, 16, v32
	v_and_b32_e32 v33, 0xffff0000, v17
	v_lshlrev_b32_e32 v32, 16, v17
	v_mul_f32_e32 v112, 0x3fb8aa3b, v1
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	v_and_b32_e32 v35, 0xffff0000, v15
	v_lshlrev_b32_e32 v34, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v14
	v_lshlrev_b32_e32 v14, 16, v14
	v_pk_mul_f32 v[14:15], v[112:113], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[112:113], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[112:113], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[112:113], v[32:33] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v147, v32, v33
	v_cvt_pk_bf16_f32 v146, v16, v17
	v_cvt_pk_bf16_f32 v145, v34, v35
	v_cvt_pk_bf16_f32 v144, v14, v15
	v_and_b32_e32 v121, 0xffff0000, v31
	v_lshlrev_b32_e32 v120, 16, v31
	v_mfma_f32_32x32x16_bf16 v[32:47], v[10:13], v[144:147], 0
	v_and_b32_e32 v31, 0xffff0000, v30
	v_lshlrev_b32_e32 v30, 16, v30
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_mul_f32_e64 v10, v112, v120
	v_mul_f32_e64 v11, v112, v121
	v_pk_mul_f32 v[12:13], v[112:113], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[112:113], v[116:117] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v151, v14, v15
	v_cvt_pk_bf16_f32 v150, v12, v13
	v_cvt_pk_bf16_f32 v149, v10, v11
	v_cvt_pk_bf16_f32 v148, v30, v31
	v_and_b32_e32 v31, 0xffff0000, v99
	v_lshlrev_b32_e32 v30, 16, v99
	v_mfma_f32_32x32x16_bf16 v[32:47], v[2:5], v[148:151], v[32:47]
	v_and_b32_e32 v99, 0xffff0000, v98
	v_lshlrev_b32_e32 v98, 16, v98
	v_and_b32_e32 v117, 0xffff0000, v97
	v_lshlrev_b32_e32 v116, 16, v97
	v_and_b32_e32 v97, 0xffff0000, v96
	v_lshlrev_b32_e32 v96, 16, v96
	v_pk_mul_f32 v[98:99], v[112:113], v[98:99] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[6:9], v[144:147], 0
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_cvt_pk_bf16_f32 v154, v98, v99
	v_mul_f32_e64 v98, v112, v116
	v_mul_f32_e64 v99, v112, v117
	v_pk_mul_f32 v[96:97], v[112:113], v[96:97] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v155, v30, v31
	v_cvt_pk_bf16_f32 v153, v98, v99
	v_cvt_pk_bf16_f32 v152, v96, v97
	v_mfma_f32_32x32x16_bf16 v[2:17], v[108:111], v[148:151], v[2:17]
	v_and_b32_e32 v99, 0xffff0000, v87
	v_lshlrev_b32_e32 v98, 16, v87
	v_and_b32_e32 v87, 0xffff0000, v86
	v_lshlrev_b32_e32 v86, 16, v86
	v_and_b32_e32 v97, 0xffff0000, v85
	v_lshlrev_b32_e32 v96, 16, v85
	v_and_b32_e32 v85, 0xffff0000, v84
	v_mfma_f32_32x32x16_bf16 v[32:47], v[104:107], v[152:155], v[32:47]
	v_lshlrev_b32_e32 v84, 16, v84
	v_mul_f32_e64 v84, v112, v84
	v_mul_f32_e64 v85, v112, v85
	v_mul_f32_e64 v96, v112, v96
	v_mul_f32_e64 v97, v112, v97
	v_pk_mul_f32 v[86:87], v[112:113], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[112:113], v[98:99] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v159, v98, v99
	v_cvt_pk_bf16_f32 v158, v86, v87
	v_mfma_f32_32x32x16_bf16 v[2:17], v[100:103], v[152:155], v[2:17]
	v_cvt_pk_bf16_f32 v157, v96, v97
	v_cvt_pk_bf16_f32 v156, v84, v85
	v_and_b32_e32 v105, 0xffff0000, v75
	v_lshlrev_b32_e32 v104, 16, v75
	v_and_b32_e32 v75, 0xffff0000, v74
	v_lshlrev_b32_e32 v74, 16, v74
	v_and_b32_e32 v85, 0xffff0000, v73
	v_mfma_f32_32x32x16_bf16 v[32:47], v[92:95], v[156:159], v[32:47]
	v_lshlrev_b32_e32 v84, 16, v73
	v_and_b32_e32 v73, 0xffff0000, v72
	v_lshlrev_b32_e32 v72, 16, v72
	v_mul_f32_e64 v72, v112, v72
	v_mul_f32_e64 v73, v112, v73
	v_pk_mul_f32 v[84:85], v[112:113], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[112:113], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[112:113], v[104:105] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[88:91], v[156:159], v[2:17]
	v_cvt_pk_bf16_f32 v163, v86, v87
	v_cvt_pk_bf16_f32 v162, v74, v75
	v_cvt_pk_bf16_f32 v161, v84, v85
	v_cvt_pk_bf16_f32 v160, v72, v73
	v_and_b32_e32 v31, 0xffff0000, v63
	v_lshlrev_b32_e32 v30, 16, v63
	v_and_b32_e32 v63, 0xffff0000, v62
	v_mfma_f32_32x32x16_bf16 v[32:47], v[80:83], v[160:163], v[32:47]
	v_lshlrev_b32_e32 v62, 16, v62
	v_and_b32_e32 v73, 0xffff0000, v61
	v_lshlrev_b32_e32 v72, 16, v61
	v_and_b32_e32 v61, 0xffff0000, v60
	v_lshlrev_b32_e32 v60, 16, v60
	v_pk_mul_f32 v[60:61], v[112:113], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[112:113], v[72:73] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[76:79], v[160:163], v[2:17]
	v_mul_f32_e64 v62, v112, v62
	v_mul_f32_e64 v63, v112, v63
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_cvt_pk_bf16_f32 v167, v30, v31
	v_cvt_pk_bf16_f32 v166, v62, v63
	v_cvt_pk_bf16_f32 v165, v72, v73
	v_cvt_pk_bf16_f32 v164, v60, v61
	v_and_b32_e32 v109, 0xffff0000, v51
	v_lshlrev_b32_e32 v108, 16, v51
	v_mfma_f32_32x32x16_bf16 v[32:47], v[68:71], v[164:167], v[32:47]
	v_and_b32_e32 v31, 0xffff0000, v50
	v_lshlrev_b32_e32 v30, 16, v50
	v_and_b32_e32 v51, 0xffff0000, v49
	v_lshlrev_b32_e32 v50, 16, v49
	v_and_b32_e32 v49, 0xffff0000, v48
	v_lshlrev_b32_e32 v48, 16, v48
	v_pk_mul_f32 v[48:49], v[112:113], v[48:49] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[64:67], v[164:167], v[2:17]
	v_mul_f32_e64 v50, v112, v50
	v_mul_f32_e64 v51, v112, v51
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_mul_f32_e64 v60, v112, v108
	v_mul_f32_e64 v61, v112, v109
	v_cvt_pk_bf16_f32 v171, v60, v61
	v_cvt_pk_bf16_f32 v170, v30, v31
	v_cvt_pk_bf16_f32 v169, v50, v51
	v_cvt_pk_bf16_f32 v168, v48, v49
	v_and_b32_e32 v119, 0xffff0000, v21
	v_lshlrev_b32_e32 v118, 16, v21
	v_mfma_f32_32x32x16_bf16 v[32:47], v[56:59], v[168:171], v[32:47]
	v_and_b32_e32 v21, 0xffff0000, v20
	v_lshlrev_b32_e32 v20, 16, v20
	v_and_b32_e32 v31, 0xffff0000, v19
	v_lshlrev_b32_e32 v30, 16, v19
	v_and_b32_e32 v19, 0xffff0000, v18
	v_lshlrev_b32_e32 v18, 16, v18
	v_pk_mul_f32 v[18:19], v[112:113], v[18:19] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[52:55], v[168:171], v[2:17]
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_mul_f32_e64 v20, v112, v20
	v_mul_f32_e64 v21, v112, v21
	v_mul_f32_e64 v48, v112, v118
	v_mul_f32_e64 v49, v112, v119
	v_cvt_pk_bf16_f32 v175, v48, v49
	v_cvt_pk_bf16_f32 v174, v20, v21
	v_cvt_pk_bf16_f32 v173, v30, v31
	v_cvt_pk_bf16_f32 v172, v18, v19
	s_lshl_b32 s13, s30, 5
	s_add_i32 s13, s13, s18
	v_mfma_f32_32x32x16_bf16 v[32:47], v[22:25], v[172:175], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[26:29], v[172:175], v[2:17]

; Prologue: causal mask for the first KV tile (kv_tile_idx = 0).
; HPP:
;   if constexpr (T::CAUSAL) {
;       const int kv_end_pos = T::KV_TILE_SIZE;
;       if (q_start_pos < kv_end_pos) {
;           attn_mask_causal_tile<T>(v_s[0], q_start_pos, 0, neg_inf_v, lane_id);
;       }
;   }
; NOTE:
;   - `s_cmp_gt_i32 s13, 63` tests `q_start_pos > 63` (= `q_start_pos >=
;     KV_TILE_SIZE = 64`); `s_cbranch_scc1 .LBB0_4` skips the mask body when
;     this wave's Q rows are past the first KV tile boundary.
;   - s13 was set to `q_block_start + warp_id * T::Q_TILE_SIZE` (HPP line
;     394) in the prologue preamble; `v_or_b32_e32 v213, s13, v113` adds
;     `lane_id % T::W_M` to derive the per-lane `q_pos`.
;   - `v_lshlrev_b32_e32 v1, 2, v209` + `v_sub_u32_e32 v1, v213, v1` computes
;     the per-lane `rel = q_pos - k_pos` (k_pos = lane_group * 4 for the first
;     wave-tile column, since k_start_pos = 0 for tile 0). v209 = lane_group.
;   - `v_mov_b32_e32 v18, 0xff800000` holds `neg_inf_v` (-inf as fp32 bits).
;   - The 16 `;;#ASMSTART`/`;;#ASMEND` blocks below each emit a pair of
;     `v_cmp_lt_i32_e64 / v_cndmask_b32_e64` for two adjacent elements:
;     `attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf, x_ref, y_ref)` from HPP
;     lines 234-249. Thresholds (THR_X, THR_Y) take values
;     {0,1},{2,3},{8,9},{10,11},{16,17},{18,19},{24,25},{26,27} for the first
;     wave-tile column and the same for the second column with rel shifted by
;     32 (`v_subrev_u32_e32 v1, 32, v1`).
;   - Acccumulator registers masked: v[32:47] (first 16-elem half of v_s[0])
;     by the first 8 ASM blocks, then v[2:17] (second half) by the next 8.
	s_cmp_gt_i32 s13, 63
	v_or_b32_e32 v213, s13, v113
	s_cbranch_scc1 .LBB0_4
	v_lshlrev_b32_e32 v1, 2, v209
	v_sub_u32_e32 v1, v213, v1
	v_mov_b32_e32 v18, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 0
	v_cmp_lt_i32_e64 s[10:11], v1, 1
	v_cndmask_b32_e64 v32, v32, v18, s[6:7]
	v_cndmask_b32_e64 v33, v33, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 2
	v_cmp_lt_i32_e64 s[10:11], v1, 3
	v_cndmask_b32_e64 v34, v34, v18, s[6:7]
	v_cndmask_b32_e64 v35, v35, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 8
	v_cmp_lt_i32_e64 s[10:11], v1, 9
	v_cndmask_b32_e64 v36, v36, v18, s[6:7]
	v_cndmask_b32_e64 v37, v37, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 10
	v_cmp_lt_i32_e64 s[10:11], v1, 11
	v_cndmask_b32_e64 v38, v38, v18, s[6:7]
	v_cndmask_b32_e64 v39, v39, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 16
	v_cmp_lt_i32_e64 s[10:11], v1, 17
	v_cndmask_b32_e64 v40, v40, v18, s[6:7]
	v_cndmask_b32_e64 v41, v41, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 18
	v_cmp_lt_i32_e64 s[10:11], v1, 19
	v_cndmask_b32_e64 v42, v42, v18, s[6:7]
	v_cndmask_b32_e64 v43, v43, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 24
	v_cmp_lt_i32_e64 s[10:11], v1, 25
	v_cndmask_b32_e64 v44, v44, v18, s[6:7]
	v_cndmask_b32_e64 v45, v45, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 26
	v_cmp_lt_i32_e64 s[10:11], v1, 27
	v_cndmask_b32_e64 v46, v46, v18, s[6:7]
	v_cndmask_b32_e64 v47, v47, v18, s[10:11]
	
	;;#ASMEND
	v_subrev_u32_e32 v1, 32, v1
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 0
	v_cmp_lt_i32_e64 s[10:11], v1, 1
	v_cndmask_b32_e64 v2, v2, v18, s[6:7]
	v_cndmask_b32_e64 v3, v3, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 2
	v_cmp_lt_i32_e64 s[10:11], v1, 3
	v_cndmask_b32_e64 v4, v4, v18, s[6:7]
	v_cndmask_b32_e64 v5, v5, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 8
	v_cmp_lt_i32_e64 s[10:11], v1, 9
	v_cndmask_b32_e64 v6, v6, v18, s[6:7]
	v_cndmask_b32_e64 v7, v7, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 10
	v_cmp_lt_i32_e64 s[10:11], v1, 11
	v_cndmask_b32_e64 v8, v8, v18, s[6:7]
	v_cndmask_b32_e64 v9, v9, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 16
	v_cmp_lt_i32_e64 s[10:11], v1, 17
	v_cndmask_b32_e64 v10, v10, v18, s[6:7]
	v_cndmask_b32_e64 v11, v11, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 18
	v_cmp_lt_i32_e64 s[10:11], v1, 19
	v_cndmask_b32_e64 v12, v12, v18, s[6:7]
	v_cndmask_b32_e64 v13, v13, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 24
	v_cmp_lt_i32_e64 s[10:11], v1, 25
	v_cndmask_b32_e64 v14, v14, v18, s[6:7]
	v_cndmask_b32_e64 v15, v15, v18, s[10:11]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v1, 26
	v_cmp_lt_i32_e64 s[10:11], v1, 27
	v_cndmask_b32_e64 v16, v16, v18, s[6:7]
	v_cndmask_b32_e64 v17, v17, v18, s[10:11]
	
	;;#ASMEND
.LBB0_4:

; Prologue: m_row = attn_row_max + sub_row + exp2 first half + barrier.
; HPP:
;   m_row = attn_row_max<T>(v_s[0]);
;   attn_sub_row<T>(v_s[0], m_row);
;   asm volatile("" : "+v"(v_s[0]) ::);
;   attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - `s_mov_b32 s11, 0xf149f2ca` = -1e30f (lowest fp32 used as the row_max
;     initial value in `attn_row_max`).
;   - `v_bfe_u32 v48, v0, 2, 2` (bits[3:2]), `v_bfe_u32 v49, v0, 4, 1`
;     (bit 4), `v_and_b32_e32 v50, 3, v0` (bits[1:0]) extract lane bits
;     used by the subsequent `make_layout_rv` (V LDS read layout) coords:
;       lane_lo = lane_id % 4 (v50)
;       lane_hi = (lane_id / 4) % 4 (v48)
;       grp_id  = lane_id / 16 (v49)
;   - 16 `v_max3_f32 v0, vA, vB, vC` instructions reduce the 32 fp32 elements
;     of v_s[0] to a single max in v0 (`attn_row_max`'s static_for loop).
;   - `v_permlane32_swap_b32_e64 v0, v1 bound_ctrl:1` performs the cross-lane
;     swap from HPP line 195 (`__builtin_amdgcn_permlane32_swap(...)`), and
;     `v_max_f32_e32 v219, v0, v1` finalises `m_row` into v219.
;   - 32 `v_sub_f32_e32 vDst, vSrc, v219` instructions implement
;     `attn_sub_row` (HPP lines 200-205). vDst occupies v[0:15] and v[16:31].
;   - The `;;#ASMSTART/ASMEND` pair (empty inline asm) is the
;     `asm volatile("" : "+v"(v_s[0]) ::)` anchor that prevents LLVM CSE
;     from merging the sub_row results with later softmax computations.
;   - 16 `v_exp_f32_e32 vX, vX` implement `attn_exp2_slice<0, s_half_len>` —
;     first 16 elements of v_s[0] (HPP line 431). The remaining 16 (second
;     half) are computed inside the main-loop Cluster 1 to interleave with
;     MFMA / cast latency.
;   - The trailing `s_barrier` is the source-level barrier from HPP line 433,
;     synchronising the wave-group after the first MMA0+softmax tile.
;   - Interleaved scalar ops `s_add_i32 s6, s16, 63` ... `s_min_i32 s16, s7,
;     s6` compute `max_num_tiles` = min(ceil_div(N, KV_TILE_SIZE), causal_num_tiles).
;     LLVM hoists this here from HPP lines 384-390 to fill VALU gaps.
	s_mov_b32 s11, 0xf149f2ca
	v_bfe_u32 v48, v0, 2, 2
	v_bfe_u32 v49, v0, 4, 1
	v_and_b32_e32 v50, 3, v0
	s_nop 3
	v_max3_f32 v0, v32, s11, v33
	v_max3_f32 v0, v0, v34, v35
	v_max3_f32 v0, v0, v36, v37
	v_max3_f32 v0, v0, v38, v39
	v_max3_f32 v0, v0, v40, v41
	v_max3_f32 v0, v0, v42, v43
	v_max3_f32 v0, v0, v44, v45
	v_max3_f32 v0, v0, v46, v47
	v_max3_f32 v0, v0, v2, v3
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_max3_f32 v0, v0, v12, v13
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	s_add_i32 s6, s16, 63
	v_mov_b32_e32 v1, v0
	s_ashr_i32 s7, s6, 31
	s_nop 0
	v_permlane32_swap_b32_e64 v0, v1 bound_ctrl:1
	s_lshr_b32 s7, s7, 26
	v_max_f32_e32 v219, v0, v1
	s_add_i32 s6, s6, s7
	s_add_i32 s7, s18, 0x13f
	v_sub_f32_e32 v31, v17, v219
	v_sub_f32_e32 v30, v16, v219
	v_sub_f32_e32 v29, v15, v219
	v_sub_f32_e32 v28, v14, v219
	v_sub_f32_e32 v27, v13, v219
	v_sub_f32_e32 v26, v12, v219
	v_sub_f32_e32 v25, v11, v219
	v_sub_f32_e32 v24, v10, v219
	v_sub_f32_e32 v23, v9, v219
	v_sub_f32_e32 v22, v8, v219
	v_sub_f32_e32 v21, v7, v219
	v_sub_f32_e32 v20, v6, v219
	v_sub_f32_e32 v19, v5, v219
	v_sub_f32_e32 v18, v4, v219
	v_sub_f32_e32 v17, v3, v219
	v_sub_f32_e32 v16, v2, v219
	v_sub_f32_e32 v15, v47, v219
	v_sub_f32_e32 v14, v46, v219
	v_sub_f32_e32 v13, v45, v219
	v_sub_f32_e32 v12, v44, v219
	v_sub_f32_e32 v11, v43, v219
	v_sub_f32_e32 v10, v42, v219
	v_sub_f32_e32 v9, v41, v219
	v_sub_f32_e32 v8, v40, v219
	v_sub_f32_e32 v7, v39, v219
	v_sub_f32_e32 v6, v38, v219
	v_sub_f32_e32 v5, v37, v219
	v_sub_f32_e32 v4, v36, v219
	v_sub_f32_e32 v3, v35, v219
	v_sub_f32_e32 v2, v34, v219
	v_sub_f32_e32 v1, v33, v219
	v_sub_f32_e32 v0, v32, v219
	s_ashr_i32 s10, s7, 31
	;;#ASMSTART
	;;#ASMEND
	s_lshr_b32 s10, s10, 26
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v2, v2
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	v_exp_f32_e32 v11, v11
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_exp_f32_e32 v14, v14
	v_exp_f32_e32 v15, v15
	s_add_i32 s7, s7, s10
	s_ashr_i32 s6, s6, 6
	s_ashr_i32 s7, s7, 6
	s_min_i32 s16, s7, s6
	s_barrier

; Prologue: prefetch K[2] async, then loop preheader (zero-trip-count guard).
; HPP:
;   async_load<T::VEC_KV>(g_k, s_k[0].ptr, u_gk, u_sk, kv_tile(2));
;   for (int j = 3; j < max_num_tiles - 1; j += 2) { ... }
; NOTE:
;   - `s_mov_b32 m0, s27` (K[0] LDS base) plus `s_lshl_b32 s6, s24, 8` (=
;     2*KV_TILE_SIZE * stride_kv_n = `kv_tile(2)` gmem byte offset). The
;     async K[2] prefetch goes into s_k[0] (overwriting the K[0] data we
;     just consumed) — this is the loop's ping-pong scheme priming the next
;     K buffer two tiles ahead.
;   - `s_cmp_gt_i32 s16, 4` tests `max_num_tiles > 4`. If false (zero-trip
;     main loop because the loop bound `j < max_num_tiles - 1` starts at 3),
;     `s_cbranch_scc1 .LBB0_6` jumps to the empty preheader (sets s[6:7]=-1
;     to disable the loop entry). Otherwise fall through to .LBB0_5 (loop
;     preheader proper) which sets s[6:7]=0.
;   - `v_lshlrev_b32_e32 v220, 3, v50` + `v_lshlrev_b32_e32 v221, 5, v49`
;     pre-compute lane components for the V LDS read coordinates used inside
;     the loop body.
	s_mov_b32 m0, s27
	s_lshl_b32 s6, s24, 8
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	buffer_load_dwordx4 v211, s[8:11], s6 offen lds
	s_mov_b32 m0, s28
	s_add_i32 s17, s17, 0xc600
	buffer_load_dwordx4 v212, s[8:11], s6 offen lds
	s_cmp_gt_i32 s16, 4
	v_lshlrev_b32_e32 v220, 3, v50
	v_lshlrev_b32_e32 v221, 5, v49
	s_cbranch_scc1 .LBB0_6
	v_mul_u32_u24_e32 v32, 0x880, v209
	v_mul_u32_u24_e32 v33, 0x220, v48
	v_lshlrev_b32_e32 v176, 3, v50
	v_lshlrev_b32_e32 v177, 5, v49
	v_add_lshl_u32 v216, v32, v33, 1
	s_mov_b64 s[6:7], 0
	s_branch .LBB0_7
.LBB0_6:
	s_mov_b64 s[6:7], -1
.LBB0_7:
	v_mul_u32_u24_e32 v217, 0x410, v114
	s_add_i32 s19, s16, -1
	s_add_i32 s18, s17, 0x2200
	s_andn2_b64 vcc, exec, s[6:7]
	s_mov_b32 s38, 0
	s_cbranch_vccnz .LBB0_17

; Prologue: main-loop preheader (set up loop-invariant addresses, zero v_o, load constants).
; HPP:
;   D_ACC l_row = 0.0f;
;   D_ACC rescale_m = 1.0f;
;   constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
;   typename decltype(mma1)::vtype_c v_o;
;   clear(v_o);
;   const int kv_tile_stride = T::KV_TILE_SIZE * kargs.stride_kv_n;
;   ...
;   for (int j = 3; j < max_num_tiles - 1; j += 2) { ... }
; NOTE:
;   - `v_mul_u32_u24_e32 v33, 0x880, v209` (0x880 = 2176 = `make_layout_rv`
;     stride for `grp_id / grp_n`), `v_mul_u32_u24_e32 v34, 0x220, v48`
;     (0x220 = 544 = `make_layout_rv` stride for `lane_in_grp / lane_lo`),
;     plus `v_add_lshl_u32 v216, v33, v34, 1` (combine + bf16-byte) build the
;     V LDS read base for `tr_load` used inside the loop.
;   - `s_mov_b32 s6, 0x8500` and `v_add_u32_e32 v222, 0x4100, v33` /
;     `v_add_u32_e32 v223, 0xc600, v33` derive the V[0] and V[1] LDS read
;     bases (s_v[0] is at offset 0x4100, s_v[1] at 0xc600 within the smem
;     buffer pair). Stored in v222 (V[0]) and v223 (V[1]) for the loop body.
;   - `v_add3_u32 v32, v218, v208, s6` derives the K LDS read base for
;     mask-positioned ds_reads inside the loop.
;   - `v_mad_i32_i24 v33, v209, -4, s13` + `s_movk_i32 s6, 0xff60` (= -160)
;     + `v_add3_u32 v224, v33, v113, s6` builds the per-lane causal-mask
;     base `q_pos - k_pos` used by `attn_mask_causal_tile` inside the loop.
;   - `v_mov_b32_e32 v214, 0` provides the literal zero used for `clear(v_o)`.
;   - `s_mov_b32 s30, 3` initialises the loop induction `j = 3`.
;   - `s_lshl_b32 s31, s29, 1` (= 2 * stride_kv_n * KV_TILE_SIZE = `2 *
;     kv_tile_stride`) is the per-iteration j-step in gmem bytes.
;   - `s_lshl_b32 s33, s24, 9` (= 512 * stride_kv_n = `kv_tile(8)` offset
;     stride), `s_mul_i32 s34, s24, 0x180` (= 384*stride_kv_n = `kv_tile(6)`
;     offset stride), and `s_movk_i32 s35, 0xc0` (= 192 = 3*KV_TILE_SIZE)
;     are address-base preloads for the staged K[j-1]/V[j-2] gmem loads.
;   - `s_mov_b32 s36, 0xf149f2ca` keeps -1e30f for in-loop `attn_row_max`
;     reductions (`row_max` initialisation per HPP line 191).
;   - `s_mov_b32 s37, 0x41000000` keeps 8.0f = RESCALE_THRESHOLD for the
;     `(row_max - m_row) <= 8.0f` lazy-rescale check (HPP line 475).
;   - `v_mov_b32_e32 v226, 0xff800000` keeps `neg_inf_v` for in-loop causal
;     mask cndmask operands (HPP line 395).
;   - The next 64 `v_mov_b32_e32 v[X], v214` (and 4 `v_mov_b32_e32 v[X], 0`)
;     are LLVM's PHI-elimination zero-init copies for `v_o = clear(...)`.
;     v_o = vtype_c<mma1> = 4 wave-tile accumulator banks * 16 fp32 = 64
;     VGPRs, split into v[80:95], v[64:79], v[48:63], v[32:47] (the GEMM1
;     accumulators).
	v_mul_u32_u24_e32 v33, 0x880, v209
	v_mul_u32_u24_e32 v34, 0x220, v48
	v_add_lshl_u32 v216, v33, v34, 1
	s_mov_b32 s6, 0x8500
	v_or3_b32 v33, v220, v221, v216
	v_add3_u32 v32, v218, v208, s6
	v_add_u32_e32 v222, 0x4100, v33
	v_add_u32_e32 v223, 0xc600, v33
	v_mad_i32_i24 v33, v209, -4, s13
	s_movk_i32 s6, 0xff60
	v_mov_b32_e32 v214, 0
	s_mov_b32 s30, 3
	s_lshl_b32 s31, s29, 1
	v_add3_u32 v224, v33, v113, s6
	s_lshl_b32 s33, s24, 9
	s_mul_i32 s34, s24, 0x180
	s_movk_i32 s35, 0xc0
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	v_add_u32_e32 v225, v32, v217
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_mov_b32 s36, 0xf149f2ca
	s_mov_b32 s37, 0x41000000
	v_mov_b32_e32 v226, 0xff800000
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, v214
	v_mov_b32_e32 v82, v214
	v_mov_b32_e32 v83, v214
	v_mov_b32_e32 v84, v214
	v_mov_b32_e32 v85, v214
	v_mov_b32_e32 v86, v214
	v_mov_b32_e32 v87, v214
	v_mov_b32_e32 v88, v214
	v_mov_b32_e32 v89, v214
	v_mov_b32_e32 v90, v214
	v_mov_b32_e32 v91, v214
	v_mov_b32_e32 v92, v214
	v_mov_b32_e32 v93, v214
	v_mov_b32_e32 v94, v214
	v_mov_b32_e32 v95, v214
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, v214
	v_mov_b32_e32 v66, v214
	v_mov_b32_e32 v67, v214
	v_mov_b32_e32 v68, v214
	v_mov_b32_e32 v69, v214
	v_mov_b32_e32 v70, v214
	v_mov_b32_e32 v71, v214
	v_mov_b32_e32 v72, v214
	v_mov_b32_e32 v73, v214
	v_mov_b32_e32 v74, v214
	v_mov_b32_e32 v75, v214
	v_mov_b32_e32 v76, v214
	v_mov_b32_e32 v77, v214
	v_mov_b32_e32 v78, v214
	v_mov_b32_e32 v79, v214
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, v214
	v_mov_b32_e32 v50, v214
	v_mov_b32_e32 v51, v214
	v_mov_b32_e32 v52, v214
	v_mov_b32_e32 v53, v214
	v_mov_b32_e32 v54, v214
	v_mov_b32_e32 v55, v214
	v_mov_b32_e32 v56, v214
	v_mov_b32_e32 v57, v214
	v_mov_b32_e32 v58, v214
	v_mov_b32_e32 v59, v214
	v_mov_b32_e32 v60, v214
	v_mov_b32_e32 v61, v214
	v_mov_b32_e32 v62, v214
	v_mov_b32_e32 v63, v214
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, v214
	v_mov_b32_e32 v34, v214
	v_mov_b32_e32 v35, v214
	v_mov_b32_e32 v36, v214
	v_mov_b32_e32 v37, v214
	v_mov_b32_e32 v38, v214
	v_mov_b32_e32 v39, v214
	v_mov_b32_e32 v40, v214
	v_mov_b32_e32 v41, v214
	v_mov_b32_e32 v42, v214
	v_mov_b32_e32 v43, v214
	v_mov_b32_e32 v44, v214
	v_mov_b32_e32 v45, v214
	v_mov_b32_e32 v46, v214
	v_mov_b32_e32 v47, v214
.LBB0_9:
; Main loop header (`for (int j = 3; j < max_num_tiles - 1; j += 2)`).
;
; Cluster 0:
; HPP:
;   async_load<T::VEC_KV>(g_v, s_v[1].ptr, u_gv, u_sv, kv_tile(j - 2));
;   v_k = load<T::VEC_KV>(s_k[1], u_rk);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - The two `buffer_load_dwordx4 v211/v212, s[4:7], s39 offen lds` issue the
;     V[j-2] async copy into s_v[1]. m0 is alternated between s17 (V[j-2] LDS
;     first-half base) and s18 (= s17 + 0x2200 = second-half offset) for the
;     two halves of v_buffer_load_insts = 2.
;   - `s_add_i32 s39, s29, s38` builds the gmem offset for kv_tile(j-2):
;     s38 = kv_tile(j-3) base (advanced each iteration by s_lshl_b32 s31, s29, 1
;     = 2 * kv_tile_stride at the loop tail), s29 = stride_kv_n * KV_TILE_SIZE
;     = kv_tile_stride; together they form kv_tile(j-2)_offset.
;   - 16 `ds_read_b128 v[X:X+3], v225 offset:Y` form `v_k = load<VEC_KV>(s_k[1],
;     u_rk)`. The offsets match the K[0] LDS read in the prologue but read from
;     s_k[1] instead of s_k[0] (v225 base = u_rk for s_k[1]).
;   - `s_waitcnt vmcnt(4) lgkmcnt(0)` waits for all LDS reads and leaves 4
;     outstanding VMEM ops (the 2 K[2] async + 2 V[j-2] async we just issued).
;   - `s_barrier` is the source-level `__builtin_amdgcn_s_barrier()` synchronising
;     the wave-group before Cluster 1's MMA0 starts.
	s_mov_b32 m0, s17
	s_add_i32 s39, s29, s38
	buffer_load_dwordx4 v211, s[4:7], s39 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v212, s[4:7], s39 offen lds
	ds_read_b128 v[96:99], v225
	ds_read_b128 v[116:119], v225 offset:64
	ds_read_b128 v[120:123], v225 offset:96
	ds_read_b128 v[124:127], v225 offset:8320
	ds_read_b128 v[176:179], v225 offset:8352
	ds_read_b128 v[180:183], v225 offset:8384
	ds_read_b128 v[184:187], v225 offset:8416
	ds_read_b128 v[128:131], v225 offset:512
	ds_read_b128 v[188:191], v225 offset:544
	ds_read_b128 v[192:195], v225 offset:576
	ds_read_b128 v[196:199], v225 offset:608
	ds_read_b128 v[200:203], v225 offset:8832
	ds_read_b128 v[112:115], v225 offset:32
	ds_read_b128 v[204:207], v225 offset:8864
	ds_read_b128 v[228:231], v225 offset:8896
	ds_read_b128 v[232:235], v225 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 1:
; HPP:
;   v_s[1] = mma0(v_q, v_k);
;   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
;   l_row += attn_sum<T>(v_s[0]);
;   v_p = opus::cast<D_ATTN>(v_s[0]);
;   asm volatile("" : "+v"(v_p) ::);
;   sched_barrier_exp_pairs<6, 3, 1>();
;   sched_barrier_pairs<10, 5, 1>();
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - 16 `v_mfma_f32_32x32x16_bf16 v[96:111]/v[128:143], v[X:X+3], v[Y:Y+3],
;     v[ACC:ACC+15]` instructions form `v_s[1] = mma0(v_q, v_k)`. The two
;     accumulator banks v[96:111] (GEMM0_E_N=0) and v[128:143] (GEMM0_E_N=1)
;     hold the 32-element v_s[1]. v_q packs in v[144:175] (8 bf16 pairs); v_k
;     in the recently-loaded v[96:99], v[112:115], ... v[228:231], v[232:235].
;     Note v_q lives in v[144:175] across the entire loop body.
;   - Interleaved with the MFMA chain, 16 `v_exp_f32_e32 vX, vX` execute the
;     second half of `attn_exp2_slice` on v_s[0] (the score tile from the
;     previous iteration's GEMM0). These fold into IGroupLP's
;     `sched_barrier_exp_pairs<6, 3, 1>` rule (6 pairs of MFMA + 3 EXP).
;   - 32 `v_add_f32_e32 v112, v112, vX` accumulate `attn_sum(v_s[0])` into
;     v112. `v_permlane32_swap_b32_e64 v112, v113 bound_ctrl:1` performs the
;     cross-lane reduction. `v_add_f32_e32 v113, v214, v113` + `v_add_f32_e32
;     v214, v113, v112` folds the new tile_sum into the running `l_row` (v214).
;   - 16 `v_cvt_pk_bf16_f32 v[112:127], vL, vH` cast v_s[0] back to bf16 for
;     v_p, ready to be consumed by the next Cluster 3 MMA1. The
;     `;;#ASMSTART/ASMEND` empty pair right after is the
;     `asm volatile("" : "+v"(v_p) ::)` anchor preventing LLVM from CSE-merging
;     these casts with later iterations' casts.
;   - The trailing `s_barrier` is `__builtin_amdgcn_s_barrier()` synchronising
;     the wave-group before Cluster 2's V LDS reads.
	v_mfma_f32_32x32x16_bf16 v[96:111], v[96:99], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	v_mfma_f32_32x32x16_bf16 v[128:143], v[128:131], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[96:111], v[112:115], v[148:151], v[96:111]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_mfma_f32_32x32x16_bf16 v[128:143], v[188:191], v[148:151], v[128:143]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_mfma_f32_32x32x16_bf16 v[96:111], v[116:119], v[152:155], v[96:111]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_mfma_f32_32x32x16_bf16 v[128:143], v[192:195], v[152:155], v[128:143]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[96:111], v[120:123], v[156:159], v[96:111]
	v_add_f32_e32 v112, v1, v0
	v_add_f32_e32 v112, v112, v2
	v_add_f32_e32 v112, v112, v3
	v_add_f32_e32 v112, v112, v4
	v_add_f32_e32 v112, v112, v5
	v_cvt_pk_bf16_f32 v120, v16, v17
	v_mfma_f32_32x32x16_bf16 v[128:143], v[196:199], v[156:159], v[128:143]
	v_add_f32_e32 v112, v112, v6
	v_add_f32_e32 v112, v112, v7
	v_add_f32_e32 v112, v112, v8
	v_add_f32_e32 v112, v112, v9
	v_add_f32_e32 v112, v112, v10
	v_mfma_f32_32x32x16_bf16 v[96:111], v[124:127], v[160:163], v[96:111]
	v_add_f32_e32 v112, v112, v11
	v_add_f32_e32 v112, v112, v12
	v_add_f32_e32 v112, v112, v13
	v_add_f32_e32 v112, v112, v14
	v_add_f32_e32 v112, v112, v15
	v_mfma_f32_32x32x16_bf16 v[128:143], v[200:203], v[160:163], v[128:143]
	v_add_f32_e32 v112, v112, v16
	v_add_f32_e32 v112, v112, v17
	v_add_f32_e32 v112, v112, v18
	v_add_f32_e32 v112, v112, v19
	v_add_f32_e32 v112, v112, v20
	v_mfma_f32_32x32x16_bf16 v[96:111], v[176:179], v[164:167], v[96:111]
	v_add_f32_e32 v112, v112, v21
	v_add_f32_e32 v112, v112, v22
	v_add_f32_e32 v112, v112, v23
	v_add_f32_e32 v112, v112, v24
	v_add_f32_e32 v112, v112, v25
	v_mfma_f32_32x32x16_bf16 v[128:143], v[204:207], v[164:167], v[128:143]
	v_add_f32_e32 v112, v112, v26
	v_add_f32_e32 v112, v112, v27
	v_add_f32_e32 v112, v112, v28
	v_add_f32_e32 v112, v112, v29
	v_add_f32_e32 v112, v112, v30
	v_mfma_f32_32x32x16_bf16 v[96:111], v[180:183], v[168:171], v[96:111]
	v_add_f32_e32 v112, v112, v31
	v_mov_b32_e32 v113, v112
	s_nop 1
	v_permlane32_swap_b32_e64 v112, v113 bound_ctrl:1
	v_add_f32_e32 v113, v214, v113
	v_add_f32_e32 v214, v113, v112
	v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[168:171], v[128:143]
	v_cvt_pk_bf16_f32 v119, v14, v15
	v_cvt_pk_bf16_f32 v118, v12, v13
	v_cvt_pk_bf16_f32 v117, v10, v11
	v_cvt_pk_bf16_f32 v116, v8, v9
	v_cvt_pk_bf16_f32 v115, v6, v7
	v_mfma_f32_32x32x16_bf16 v[96:111], v[184:187], v[172:175], v[96:111]
	v_cvt_pk_bf16_f32 v114, v4, v5
	v_cvt_pk_bf16_f32 v113, v2, v3
	v_cvt_pk_bf16_f32 v112, v0, v1
	v_cvt_pk_bf16_f32 v127, v30, v31
	v_cvt_pk_bf16_f32 v126, v28, v29
	v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[172:175], v[128:143]
	v_cvt_pk_bf16_f32 v125, v26, v27
	v_cvt_pk_bf16_f32 v124, v24, v25
	v_cvt_pk_bf16_f32 v123, v22, v23
	v_cvt_pk_bf16_f32 v122, v20, v21
	v_cvt_pk_bf16_f32 v121, v18, v19
	;;#ASMSTART
	;;#ASMEND
	s_barrier

; Cluster 2:
; HPP:
;   async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(j));
;   v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - `s_add_i32 s39, s34, s38` builds `kv_tile(j)` gmem byte offset using
;     s34 = (3 - 0)*KV_TILE_SIZE*stride_kv_n (a constant kv_tile(3)-relative
;     stride from preheader) added to s38 = current iteration's `kv_tile(j-3)`
;     base. (j starts at 3 → s34 + s38 = kv_tile(j).)
;   - Two `buffer_load_dwordx4 v211/v212, s[8:11], s39 offen lds` issue the
;     K[j] async copy into s_k[1]. m0 alternates s26 (K[1] first-half base)
;     and s25 (= s26 + 0x2080 second-half offset).
;   - 32 `ds_read_b64_tr_b16 v[X:X+1], v222 offset:Y` form
;     `v_v = tr_load<VEC_TR_V>(s_v[0], u_rv)`. v_ds_read_insts = 32 (see
;     header). The `b64_tr_b16` opcode performs a transpose-load: each
;     instruction reads 64 bits from LDS but permutes the lanes so the
;     loaded bf16 pair lands at the correct register position for the next
;     MMA1 (V is consumed with `mfma_adaptor_swap_ab`). The offsets cover
;     both V wave-tiles for GEMM1_E_N=4 chunks (16 offsets for the first half
;     of GEMM1_E_K + 16 for the second half, base + 0x2200 stride between).
;   - `s_waitcnt vmcnt(4) lgkmcnt(0)` waits for the 32 LDS tr_loads and
;     leaves the 4 outstanding VMEM ops (V[j-2] + K[j] async = 4) pending.
;   - The trailing `s_barrier` separates Cluster 2 from Cluster 3 below.
	s_add_i32 s39, s34, s38
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v211, s[8:11], s39 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v212, s[8:11], s39 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v222 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v222 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v222 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v222 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0:1], v222 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[2:3], v222 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v222 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v222 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v222 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v222 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[4:5], v222 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[6:7], v222 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v222 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v222 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v222 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v222 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v222 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v222 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[8:9], v222 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[10:11], v222 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v222 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v222 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v222 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v222 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v222 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v222 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[12:13], v222 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v222 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v222 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v222 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v222 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v222 offset:0x25c0

	;;#ASMEND
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 3: (first half: setprio + 1st MMA1 step + attn_row_max + rescale test).
; HPP:
;   __builtin_amdgcn_s_setprio(1);
;   v_o = mma1.step_k(0_I, v_p, v_v, v_o);
;   D_ACC row_max = attn_row_max<T>(v_s[1]);
;   sched_barrier_pairs<4, 5, 2>();
;   bool below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
;   bool all_below = (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
;   if (__builtin_expect(all_below, 1)) {
;       row_max = m_row;
;   } else {
;       rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;       scale_output_tile<T>(v_o, rescale_m);
;       l_row *= rescale_m;
;       m_row = row_max;
;   }
; NOTE:
;   - `s_setprio 1` raises this wave-group's MFMA-dispatch priority so the 4
;     MMA1 step_k chain quickly grabs the MFMA unit (the paired
;     `s_setprio 0` at the end of Cluster 3 yields back).
;   - 4 `v_mfma_f32_32x32x16_bf16 v[80:95]/v[64:79]/v[48:63]/v[32:47],
;     v[20:23]/v[24:27]/v[28:31]/v[204:207], v[112:115], v[ACC:ACC+15]` =
;     `mma1.step_k(0_I, v_p, v_v, v_o)`: one step over GEMM1_E_M=1 * GEMM1_E_N=4
;     wave-tiles for K=0 micro-step (out of GEMM1_E_K=4). v_p packs in
;     v[112:127]; the 4 v_v packs `v[20..]/v[24..]/v[28..]/v[204..]` came from
;     the tr_load above; the 4 accumulator banks `v[80:95]/v[64:79]/v[48:63]/
;     v[32:47]` are the four GEMM1_E_N outputs of v_o.
;   - Interleaved: 16 `v_max3_f32 v20, v20, vA, vB` chain reduce v_s[1] (the
;     score tile from this iteration's GEMM0, in v[96:143]) to a per-lane max
;     in v20. Then `v_permlane32_swap_b32_e64 v20, v21 bound_ctrl:1` does the
;     32-lane swap and `v_max_f32_e32 v20, v20, v21` yields `row_max`.
;   - `v_sub_f32_e32 v21, v20, v219` computes `row_max - m_row` (v219 carries
;     the running m_row across iterations).
;   - `v_cmp_ge_f32_e32 vcc, s37, v21` tests `RESCALE_THRESHOLD (8.0) >=
;     (row_max - m_row)` and `s_cmp_eq_u64 vcc, exec` ballots the result.
;   - `s_cbranch_scc0 .LBB0_15` takes the rescale path when ANY lane reports
;     `(row_max - m_row) > 8.0` (the `__builtin_expect(all_below, 1)` hints
;     LLVM to make the all-below path the fall-through hot path).
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[112:115], v[80:95]
	v_max3_f32 v20, v96, s36, v97
	v_max3_f32 v20, v20, v98, v99
	v_max3_f32 v20, v20, v100, v101
	v_max3_f32 v20, v20, v102, v103
	v_max3_f32 v20, v20, v104, v105
	v_mfma_f32_32x32x16_bf16 v[64:79], v[24:27], v[112:115], v[64:79]
	v_max3_f32 v20, v20, v106, v107
	v_max3_f32 v20, v20, v108, v109
	v_max3_f32 v20, v20, v110, v111
	v_max3_f32 v20, v20, v128, v129
	v_max3_f32 v20, v20, v130, v131
	v_mfma_f32_32x32x16_bf16 v[48:63], v[28:31], v[112:115], v[48:63]
	v_max3_f32 v20, v20, v132, v133
	v_max3_f32 v20, v20, v134, v135
	v_max3_f32 v20, v20, v136, v137
	v_max3_f32 v20, v20, v138, v139
	v_max3_f32 v20, v20, v140, v141
	v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[112:115], v[32:47]
	v_max3_f32 v20, v20, v142, v143
	v_mov_b32_e32 v21, v20
	s_nop 1
	v_permlane32_swap_b32_e64 v20, v21 bound_ctrl:1
	v_max_f32_e32 v20, v20, v21
	v_sub_f32_e32 v21, v20, v219
	v_cmp_ge_f32_e32 vcc, s37, v21
	s_cmp_eq_u64 vcc, exec
	s_cbranch_scc0 .LBB0_15
.LBB0_10:
; Cluster 3 (second half: 3 remaining MMA1 steps + sub_row + exp2 first half + setprio 0 + barrier).
; HPP:
;   v_o = mma1.step_k(1_I, v_p, v_v, v_o);
;   attn_sub_row<T>(v_s[1], row_max);
;   v_o = mma1.step_k(2_I, v_p, v_v, v_o);
;   asm volatile("" : "+v"(v_s[1]) ::);
;   v_o = mma1.step_k(3_I, v_p, v_v, v_o);
;   attn_exp2_slice<T, 0, s_half_len>(v_s[1]);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_setprio(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
;   sched_barrier_pairs<3, 5, 3>();
; NOTE:
;   - 12 `v_mfma_f32_32x32x16_bf16` instructions implement the 3 remaining
;     GEMM1_E_K micro-steps (4 wave-tiles each = 12 MFMA): step_k(1,2,3) into
;     v[80:95]/v[64:79]/v[48:63]/v[32:47] (the 4 wave-tile accumulator banks
;     of v_o). v_p remains v[112:127] across all 4 steps; v_v's 4 packs cycle
;     through groups of 4 GEMM1_E_K offsets each.
;   - Interleaved between the MFMA chain, 32 `v_sub_f32_e32 vX, vY, v219`
;     implement `attn_sub_row(v_s[1], row_max)`. The order is reverse-element:
;     LLVM scheduled them tail-first to align with the MFMA dispatch window.
;     v219 was updated either by the fall-through path (kept as old m_row,
;     since `if (all_below) row_max = m_row` and we always reuse v219) OR by
;     .LBB0_15 (rescale path stores the new row_max into v219).
;   - Empty `;;#ASMSTART/ASMEND` is the `asm volatile("" : "+v"(v_s[1]) ::)`
;     anchor preventing LLVM from CSE-merging the sub_row results with the
;     subsequent exp2.
;   - 16 `v_exp_f32_e32 v200..v207, v229..v236, vX` implement the first half
;     of `attn_exp2_slice<0, s_half_len>(v_s[1])`. The destinations are
;     write-allocated VGPRs (v200-v207, v229-v236) to maintain liveness across
;     the next clusters (Cluster 5 reads these for the per-iteration sum).
;   - `s_setprio 0` yields priority back to the other wave-group. `s_barrier`
;     synchronises both groups before Cluster 4's V/K LDS reads.
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[116:119], v[80:95]
	v_sub_f32_e32 v31, v143, v219
	v_sub_f32_e32 v30, v142, v219
	v_sub_f32_e32 v29, v141, v219
	v_sub_f32_e32 v28, v140, v219
	v_sub_f32_e32 v27, v139, v219
	v_mfma_f32_32x32x16_bf16 v[64:79], v[4:7], v[116:119], v[64:79]
	v_sub_f32_e32 v26, v138, v219
	v_sub_f32_e32 v25, v137, v219
	v_sub_f32_e32 v24, v136, v219
	v_sub_f32_e32 v23, v135, v219
	v_sub_f32_e32 v22, v134, v219
	v_mfma_f32_32x32x16_bf16 v[48:63], v[8:11], v[116:119], v[48:63]
	v_sub_f32_e32 v21, v133, v219
	v_sub_f32_e32 v20, v132, v219
	v_sub_f32_e32 v19, v131, v219
	v_sub_f32_e32 v18, v130, v219
	v_sub_f32_e32 v17, v129, v219
	v_mfma_f32_32x32x16_bf16 v[32:47], v[12:15], v[116:119], v[32:47]
	v_sub_f32_e32 v16, v128, v219
	v_sub_f32_e32 v15, v111, v219
	v_sub_f32_e32 v14, v110, v219
	v_sub_f32_e32 v13, v109, v219
	v_sub_f32_e32 v12, v108, v219
	v_mfma_f32_32x32x16_bf16 v[80:95], v[0:3], v[120:123], v[80:95]
	v_sub_f32_e32 v11, v107, v219
	v_sub_f32_e32 v10, v106, v219
	v_sub_f32_e32 v9, v105, v219
	v_sub_f32_e32 v8, v104, v219
	v_sub_f32_e32 v7, v103, v219
	v_sub_f32_e32 v1, v97, v219
	v_sub_f32_e32 v0, v96, v219
	v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[120:123], v[64:79]
	v_sub_f32_e32 v6, v102, v219
	v_sub_f32_e32 v5, v101, v219
	v_sub_f32_e32 v4, v100, v219
	v_sub_f32_e32 v3, v99, v219
	v_sub_f32_e32 v2, v98, v219
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[120:123], v[48:63]
	v_exp_f32_e32 v200, v0
	v_exp_f32_e32 v201, v1
	v_exp_f32_e32 v202, v2
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[120:123], v[32:47]
	v_exp_f32_e32 v203, v3
	v_exp_f32_e32 v204, v4
	v_exp_f32_e32 v205, v5
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[124:127], v[80:95]
	v_exp_f32_e32 v206, v6
	v_exp_f32_e32 v207, v7
	v_exp_f32_e32 v229, v8
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[124:127], v[64:79]
	v_exp_f32_e32 v230, v9
	v_exp_f32_e32 v231, v10
	v_exp_f32_e32 v232, v11
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[124:127], v[48:63]
	v_exp_f32_e32 v233, v12
	v_exp_f32_e32 v234, v13
	v_exp_f32_e32 v235, v14
	v_mfma_f32_32x32x16_bf16 v[32:47], v[188:191], v[124:127], v[32:47]
	v_exp_f32_e32 v236, v15
	s_setprio 0
	s_barrier

; Cluster 4:
; HPP:
;   async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(j - 1));
;   v_k = load<T::VEC_KV>(s_k[0], u_rk);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - This is the symmetric mirror of Cluster 0 with V/K buffer indices flipped:
;     V[j-1] async into s_v[0], K[0] LDS reads from s_k[0]. The two
;     `buffer_load_dwordx4 v211/v212, s[4:7], s39 offen lds` issue V[j-1]
;     async; `s_add_i32 s39, s31, s38` builds the offset
;     (`kv_tile(j-1) = j*kv_tile_stride + kv_tile_stride = kv_tile_stride + s38`,
;     where s31 = 2*kv_tile_stride was preheader-init; remainder rolls into s38
;     at the loop tail).
;   - `v_add_u32_e32 v4, v215, v217` derives the per-lane K[0] LDS read base
;     (different from v225 used in Cluster 0 because s_k[0] base differs by
;     0x8500 = smem_buffer_elems*2 bytes).
;   - 16 `ds_read_b128 v[X:X+3], v4 offset:Y` form `v_k = load<VEC_KV>(s_k[0],
;     u_rk)`. The offsets mirror Cluster 0 (16/32/64/96 + 8320/8352/.../8928
;     and the +512-base variants).
;   - `s_waitcnt vmcnt(4) lgkmcnt(0)` + `s_barrier` close the cluster the
;     same way as Cluster 0.
	s_mov_b32 m0, s21
	s_add_i32 s39, s31, s38
	buffer_load_dwordx4 v211, s[4:7], s39 offen lds
	s_mov_b32 m0, s20
	v_add_u32_e32 v4, v215, v217
	buffer_load_dwordx4 v212, s[4:7], s39 offen lds
	ds_read_b128 v[0:3], v4
	ds_read_b128 v[96:99], v4 offset:32
	ds_read_b128 v[100:103], v4 offset:64
	ds_read_b128 v[104:107], v4 offset:96
	ds_read_b128 v[108:111], v4 offset:8320
	ds_read_b128 v[128:131], v4 offset:8352
	ds_read_b128 v[132:135], v4 offset:8384
	ds_read_b128 v[136:139], v4 offset:8416
	ds_read_b128 v[112:115], v4 offset:512
	ds_read_b128 v[140:143], v4 offset:544
	ds_read_b128 v[176:179], v4 offset:576
	ds_read_b128 v[180:183], v4 offset:608
	ds_read_b128 v[184:187], v4 offset:8832
	ds_read_b128 v[188:191], v4 offset:8864
	ds_read_b128 v[192:195], v4 offset:8896
	ds_read_b128 v[196:199], v4 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 5:
; HPP:
;   v_s[0] = mma0(v_q, v_k);
;   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1]);
;   l_row += attn_sum<T>(v_s[1]);
;   v_p = opus::cast<D_ATTN>(v_s[1]);
;   asm volatile("" : "+v"(v_p) ::);
;   sched_barrier_exp_pairs<6, 3, 4>();
;   sched_barrier_pairs<10, 5, 4>();
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Mirror of Cluster 1 with score-tile swap: now `mma0(v_q, v_k)` writes
;     v_s[0] into v[0:15] / v[112:127] (two GEMM0_E_N banks), and the exp2
;     second-half + sum + cast operate on v_s[1] (the previous-iteration
;     score tile in v[96:143]).
;   - 16 `v_mfma_f32_32x32x16_bf16 v[0:15]/v[112:127], v[X:X+3], v[Y:Y+3]`
;     are the GEMM0 chain. v_p builds in v[96:111] this time.
;   - Interleaved 16 `v_exp_f32_e32 vX, vX` (on v_s[1]'s second half),
;     32 `v_add_f32_e32 v96, v96, vX` for `attn_sum(v_s[1])` into v227,
;     `v_permlane32_swap_b32_e64 v227, v228 bound_ctrl:1` for cross-lane
;     reduction. The actual `l_row += tile_sum` fold (`v_add_f32_e32 v214,
;     v214, v228/v227`) is hoisted by LLVM into Cluster 6 below to keep this
;     cluster's MFMA chain dense.
;   - 16 `v_cvt_pk_bf16_f32 v[96:111], vL, vH` cast v_s[1] to bf16 for v_p
;     consumption in Cluster 7's MMA1.
;   - `;;#ASMSTART/ASMEND` is the v_p asm-volatile anchor; trailing
;     `s_barrier` separates from Cluster 6.
	v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	v_mfma_f32_32x32x16_bf16 v[112:127], v[112:115], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[148:151], v[0:15]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_mfma_f32_32x32x16_bf16 v[112:127], v[140:143], v[148:151], v[112:127]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_mfma_f32_32x32x16_bf16 v[0:15], v[100:103], v[152:155], v[0:15]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_mfma_f32_32x32x16_bf16 v[112:127], v[176:179], v[152:155], v[112:127]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[0:15], v[104:107], v[156:159], v[0:15]
	v_add_f32_e32 v96, v201, v200
	v_add_f32_e32 v96, v96, v202
	v_add_f32_e32 v96, v96, v203
	v_add_f32_e32 v96, v96, v204
	v_add_f32_e32 v96, v96, v205
	v_mfma_f32_32x32x16_bf16 v[112:127], v[180:183], v[156:159], v[112:127]
	v_add_f32_e32 v96, v96, v206
	v_add_f32_e32 v96, v96, v207
	v_add_f32_e32 v96, v96, v229
	v_add_f32_e32 v96, v96, v230
	v_add_f32_e32 v96, v96, v231
	v_mfma_f32_32x32x16_bf16 v[0:15], v[108:111], v[160:163], v[0:15]
	v_add_f32_e32 v96, v96, v232
	v_add_f32_e32 v96, v96, v233
	v_add_f32_e32 v96, v96, v234
	v_add_f32_e32 v96, v96, v235
	v_add_f32_e32 v96, v96, v236
	v_mfma_f32_32x32x16_bf16 v[112:127], v[184:187], v[160:163], v[112:127]
	v_add_f32_e32 v96, v96, v16
	v_add_f32_e32 v96, v96, v17
	v_add_f32_e32 v96, v96, v18
	v_add_f32_e32 v96, v96, v19
	v_add_f32_e32 v96, v96, v20
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[164:167], v[0:15]
	v_add_f32_e32 v96, v96, v21
	v_add_f32_e32 v96, v96, v22
	v_add_f32_e32 v96, v96, v23
	v_add_f32_e32 v96, v96, v24
	v_add_f32_e32 v96, v96, v25
	v_mfma_f32_32x32x16_bf16 v[112:127], v[188:191], v[164:167], v[112:127]
	v_add_f32_e32 v96, v96, v26
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[168:171], v[0:15]
	v_add_f32_e32 v227, v96, v31
	v_mov_b32_e32 v228, v227
	s_nop 1
	v_permlane32_swap_b32_e64 v227, v228 bound_ctrl:1
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	v_mfma_f32_32x32x16_bf16 v[112:127], v[192:195], v[168:171], v[112:127]
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[172:175], v[0:15]
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v235, v236
	v_cvt_pk_bf16_f32 v102, v233, v234
	v_cvt_pk_bf16_f32 v101, v231, v232
	v_cvt_pk_bf16_f32 v100, v229, v230
	v_mfma_f32_32x32x16_bf16 v[112:127], v[196:199], v[172:175], v[112:127]
	v_cvt_pk_bf16_f32 v99, v206, v207
	v_cvt_pk_bf16_f32 v98, v204, v205
	v_cvt_pk_bf16_f32 v97, v202, v203
	v_cvt_pk_bf16_f32 v96, v200, v201
	;;#ASMSTART
	;;#ASMEND
	s_barrier

; Cluster 6:
; HPP:
;   async_load<T::VEC_KV>(g_k, s_k[0].ptr, u_gk, u_sk, kv_tile(j + 1));
;   v_v = tr_load<T::VEC_TR_V>(s_v[1], u_rv);
;   if constexpr (T::CAUSAL) {
;       const int kv_end_pos = (j + 1) * T::KV_TILE_SIZE;
;       if (q_start_pos < kv_end_pos) {
;           attn_mask_causal_tile<T>(v_s[0], q_start_pos, (j - 1) * T::KV_TILE_SIZE, neg_inf_v, lane_id);
;       }
;   }
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Two `buffer_load_dwordx4 v211/v212, s[8:11], s38 offen lds` issue
;     K[j+1] async into s_k[0]. `s_add_i32 s38, s33, s38` advances the gmem
;     base by 8*kv_tile_stride (s33 from preheader) — this is the j+=2 stride
;     baked into s38 for the next iteration.
;   - 32 `ds_read_b64_tr_b16 v[X:X+1], v223 offset:Y` form
;     `v_v = tr_load<VEC_TR_V>(s_v[1], u_rv)`. v223 base = u_rv for s_v[1].
;     Offsets mirror Cluster 2 with the same b64_tr_b16 transpose layout for
;     the V wave-tiles.
;   - `s_cmp_ge_i32 s13, s35` tests `q_start_pos >= (j+1) * KV_TILE_SIZE`
;     (s35 was preheader-init to 0xc0 = 192 = 3*KV_TILE_SIZE and is bumped
;     by `s_addk_i32 s35, 0x80` each iteration so it tracks (j+1)*KV_TILE_SIZE
;     ahead of the j+=2 step at the loop tail).
;   - `s_cbranch_scc1 .LBB0_12` skips the causal-mask body when this wave's
;     Q rows are past the right edge of K tile (j+1).
;   - The causal-mask body (16 ASM blocks of v_cmp_lt_i32 + v_cndmask_b32
;     pairs for the first wave-tile, then 16 more for the second) implements
;     `attn_mask_causal_tile<T>(v_s[0], q_start_pos, (j-1)*KV_TILE_SIZE,
;     neg_inf_v, lane_id)`. v229 = v224 + 32 (rel for second wave-tile);
;     v224 = q_pos - 4*lane_group - 160 (kept loop-invariant from preheader
;     and updated at loop tail with `v_add_u32_e32 v224, 0xffffff80, v224`
;     = decrement by 128 to track (j+1)*KV_TILE_SIZE advancing).
;   - The 32 cndmask groups target v[0:15] (first wave-tile half of v_s[0])
;     and v[112:127] (second wave-tile half), masking with v226 = neg_inf_v.
;   - After the masked path rejoin at `.LBB0_12`, two `v_add_f32_e32 v214,
;     v214, v228/v227` fold the Cluster 5 tile_sum into l_row (LLVM hoisted
;     them down here to avoid pressuring the MFMA chain above).
;   - `s_waitcnt vmcnt(4) lgkmcnt(0)` + `s_barrier` close the cluster.
	s_mov_b32 m0, s27
	s_add_i32 s38, s33, s38
	buffer_load_dwordx4 v211, s[8:11], s38 offen lds
	s_mov_b32 m0, s28
	s_cmp_ge_i32 s13, s35
	buffer_load_dwordx4 v212, s[8:11], s38 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v223 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v223 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v223 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v223 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v223 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v223 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v223 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v223 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v223 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v223 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v223 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v223 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v223 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v223 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v223 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v223 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v223 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v223 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v223 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v223 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v223 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v223 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v223 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v223 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v223 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v223 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v223 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v223 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v223 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v223 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v223 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v223 offset:0x25c0

	;;#ASMEND
	s_cbranch_scc1 .LBB0_12
	v_add_u32_e32 v229, 32, v224
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 0
	v_cmp_lt_i32_e64 s[42:43], v229, 1
	v_cndmask_b32_e64 v0, v0, v226, s[40:41]
	v_cndmask_b32_e64 v1, v1, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 2
	v_cmp_lt_i32_e64 s[42:43], v229, 3
	v_cndmask_b32_e64 v2, v2, v226, s[40:41]
	v_cndmask_b32_e64 v3, v3, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 8
	v_cmp_lt_i32_e64 s[42:43], v229, 9
	v_cndmask_b32_e64 v4, v4, v226, s[40:41]
	v_cndmask_b32_e64 v5, v5, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 10
	v_cmp_lt_i32_e64 s[42:43], v229, 11
	v_cndmask_b32_e64 v6, v6, v226, s[40:41]
	v_cndmask_b32_e64 v7, v7, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 16
	v_cmp_lt_i32_e64 s[42:43], v229, 17
	v_cndmask_b32_e64 v8, v8, v226, s[40:41]
	v_cndmask_b32_e64 v9, v9, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 18
	v_cmp_lt_i32_e64 s[42:43], v229, 19
	v_cndmask_b32_e64 v10, v10, v226, s[40:41]
	v_cndmask_b32_e64 v11, v11, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 24
	v_cmp_lt_i32_e64 s[42:43], v229, 25
	v_cndmask_b32_e64 v12, v12, v226, s[40:41]
	v_cndmask_b32_e64 v13, v13, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v229, 26
	v_cmp_lt_i32_e64 s[42:43], v229, 27
	v_cndmask_b32_e64 v14, v14, v226, s[40:41]
	v_cndmask_b32_e64 v15, v15, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 0
	v_cmp_lt_i32_e64 s[42:43], v224, 1
	v_cndmask_b32_e64 v112, v112, v226, s[40:41]
	v_cndmask_b32_e64 v113, v113, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 2
	v_cmp_lt_i32_e64 s[42:43], v224, 3
	v_cndmask_b32_e64 v114, v114, v226, s[40:41]
	v_cndmask_b32_e64 v115, v115, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 8
	v_cmp_lt_i32_e64 s[42:43], v224, 9
	v_cndmask_b32_e64 v116, v116, v226, s[40:41]
	v_cndmask_b32_e64 v117, v117, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 10
	v_cmp_lt_i32_e64 s[42:43], v224, 11
	v_cndmask_b32_e64 v118, v118, v226, s[40:41]
	v_cndmask_b32_e64 v119, v119, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 16
	v_cmp_lt_i32_e64 s[42:43], v224, 17
	v_cndmask_b32_e64 v120, v120, v226, s[40:41]
	v_cndmask_b32_e64 v121, v121, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 18
	v_cmp_lt_i32_e64 s[42:43], v224, 19
	v_cndmask_b32_e64 v122, v122, v226, s[40:41]
	v_cndmask_b32_e64 v123, v123, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 24
	v_cmp_lt_i32_e64 s[42:43], v224, 25
	v_cndmask_b32_e64 v124, v124, v226, s[40:41]
	v_cndmask_b32_e64 v125, v125, v226, s[42:43]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v224, 26
	v_cmp_lt_i32_e64 s[42:43], v224, 27
	v_cndmask_b32_e64 v126, v126, v226, s[40:41]
	v_cndmask_b32_e64 v127, v127, v226, s[42:43]
	
	;;#ASMEND
.LBB0_12:
	v_add_f32_e32 v214, v214, v228
	v_add_f32_e32 v214, v214, v227
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 7: (first half: setprio + 1st MMA1 step + attn_row_max + rescale test).
; HPP:
;   __builtin_amdgcn_s_setprio(1);
;   v_o = mma1.step_k(0_I, v_p, v_v, v_o);
;   D_ACC row_max = attn_row_max<T>(v_s[0]);
;   sched_barrier_pairs<4, 5, 5>();
;   bool below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
;   bool all_below = (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
;   if (__builtin_expect(all_below, 1)) {
;       row_max = m_row;
;   } else {
;       rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;       scale_output_tile<T>(v_o, rescale_m);
;       l_row *= rescale_m;
;       m_row = row_max;
;   }
; NOTE:
;   - Same shape as Cluster 3 with v_p swapped: now v_p is in v[96:111]
;     (built by Cluster 5) and v_v is in v[204:207]/v[24:27]/v[28:31]/v[200:203]
;     (the second-cluster tr_load destinations).
;   - 4 `v_mfma_f32_32x32x16_bf16` are step_k(0) over the 4 GEMM1_E_N banks
;     of v_o (v[80:95]/v[64:79]/v[48:63]/v[32:47]).
;   - 16 `v_max3_f32 v204/v24, ...` reduce v_s[0] (v[0:127] split = v[0:15]
;     + v[112:127] from Cluster 5) to per-lane `row_max`. v0/v1 hold the
;     first two elements of v_s[0]; the chain reads v[2:15] and v[112:127]
;     via the v_max3 ternary form (3 operands max in one instruction).
;   - `v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1` + `v_max_f32_e32 v24,
;     v24, v25` finishes the cross-lane row_max.
;   - `v_sub_f32_e32 v25, v24, v219` + `v_cmp_ge_f32_e32 vcc, s37, v25` +
;     `s_cmp_eq_u64 vcc, exec` + `s_cbranch_scc0 .LBB0_16` is the
;     symmetric `below_thresh` ballot + branch to the rescale path.
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[204:207], v[96:99], v[80:95]
	v_max3_f32 v204, v0, s36, v1
	v_max3_f32 v204, v204, v2, v3
	v_max3_f32 v204, v204, v4, v5
	v_max3_f32 v204, v204, v6, v7
	v_max3_f32 v204, v204, v8, v9
	v_mfma_f32_32x32x16_bf16 v[64:79], v[24:27], v[96:99], v[64:79]
	v_max3_f32 v24, v204, v10, v11
	v_max3_f32 v24, v24, v12, v13
	v_max3_f32 v24, v24, v14, v15
	v_max3_f32 v24, v24, v112, v113
	v_max3_f32 v24, v24, v114, v115
	v_mfma_f32_32x32x16_bf16 v[48:63], v[28:31], v[96:99], v[48:63]
	v_max3_f32 v24, v24, v116, v117
	v_max3_f32 v24, v24, v118, v119
	v_max3_f32 v24, v24, v120, v121
	v_max3_f32 v24, v24, v122, v123
	v_max3_f32 v24, v24, v124, v125
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[96:99], v[32:47]
	v_max3_f32 v24, v24, v126, v127
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1
	v_max_f32_e32 v24, v24, v25
	v_sub_f32_e32 v25, v24, v219
	v_cmp_ge_f32_e32 vcc, s37, v25
	s_cmp_eq_u64 vcc, exec
	s_cbranch_scc0 .LBB0_16
.LBB0_13:
; Cluster 7: (second half: 3 remaining MMA1 steps + sub_row + exp2 first half + setprio 0 + barrier).
; HPP:
;   v_o = mma1.step_k(1_I, v_p, v_v, v_o);
;   attn_sub_row<T>(v_s[0], row_max);
;   v_o = mma1.step_k(2_I, v_p, v_v, v_o);
;   asm volatile("" : "+v"(v_s[0]) ::);
;   v_o = mma1.step_k(3_I, v_p, v_v, v_o);
;   attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_setprio(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
;   sched_barrier_pairs<3, 5, 6>();
;
; Loop tail:
;   j += 2;  /* via s_add_i32 s30, s30, 2 below */
;   /* update s35 = (j+1)*KV_TILE_SIZE for causal mask */
;   /* update v224 -= 128 = -2*KV_TILE_SIZE for mask rel */
;   if (j >= max_num_tiles - 1) goto .LBB0_18 (exit);
;   else { s38 = next iteration's kv_tile(j-3) base; goto .LBB0_9; }
; NOTE:
;   - Same shape as Cluster 3 with operand index swap. 12
;     `v_mfma_f32_32x32x16_bf16` form step_k(1,2,3); v_p remains v[96:111];
;     v_v's 4 packs cycle through (v[20:23]/v[16:19]/v[192:195]/v[196:199],
;     v[180:183]/v[184:187]/v[188:191]/v[176:179],
;     v[140:143]/v[136:139]/v[132:135]/v[128:131]) for the 3 remaining K
;     micro-steps.
;   - 32 `v_sub_f32_e32 vX, vY, v219` implement `attn_sub_row(v_s[0],
;     row_max)`. Source registers cover v_s[0]'s 32 elements (v[127],
;     v[126], ..., v[112], v[15], ..., v[0]).
;   - Empty `;;#ASMSTART/ASMEND` is the v_s[0] anchor.
;   - 16 `v_exp_f32_e32 vX, vX` execute the first half of `attn_exp2_slice<0,
;     s_half_len>(v_s[0])`.
;   - `s_setprio 0` + `s_barrier` close Cluster 7. The four tail instructions
;     update loop state:
;       * `s_add_i32 s30, s30, 2`           — `j += 2`
;       * `s_addk_i32 s35, 0x80`            — `kv_end_pos += 2*KV_TILE_SIZE`
;       * `s_cmp_ge_i32 s30, s19`           — `j >= max_num_tiles - 1`
;       * `v_add_u32_e32 v224, 0xffffff80,  — `mask_rel_base -= 128` (= shift
;          v224`                              by 2 KV_TILE_SIZE for next iter)
;     `s_cbranch_scc1 .LBB0_18` exits the loop; `s_mov_b32 s38, s39` +
;     `s_branch .LBB0_9` carries the s38 base forward and re-enters the loop.
	v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[100:103], v[80:95]
	v_sub_f32_e32 v31, v127, v219
	v_sub_f32_e32 v30, v126, v219
	v_sub_f32_e32 v29, v125, v219
	v_sub_f32_e32 v28, v124, v219
	v_sub_f32_e32 v27, v123, v219
	v_sub_f32_e32 v1, v1, v219
	v_sub_f32_e32 v0, v0, v219
	v_mfma_f32_32x32x16_bf16 v[64:79], v[16:19], v[100:103], v[64:79]
	v_sub_f32_e32 v26, v122, v219
	v_sub_f32_e32 v25, v121, v219
	v_sub_f32_e32 v24, v120, v219
	v_sub_f32_e32 v23, v119, v219
	v_sub_f32_e32 v22, v118, v219
	v_mfma_f32_32x32x16_bf16 v[48:63], v[192:195], v[100:103], v[48:63]
	v_sub_f32_e32 v21, v117, v219
	v_sub_f32_e32 v20, v116, v219
	v_sub_f32_e32 v19, v115, v219
	v_sub_f32_e32 v18, v114, v219
	v_sub_f32_e32 v17, v113, v219
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[100:103], v[32:47]
	v_sub_f32_e32 v16, v112, v219
	v_sub_f32_e32 v15, v15, v219
	v_sub_f32_e32 v14, v14, v219
	v_sub_f32_e32 v13, v13, v219
	v_sub_f32_e32 v12, v12, v219
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[104:107], v[80:95]
	v_sub_f32_e32 v11, v11, v219
	v_sub_f32_e32 v10, v10, v219
	v_sub_f32_e32 v9, v9, v219
	v_sub_f32_e32 v8, v8, v219
	v_sub_f32_e32 v7, v7, v219
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[104:107], v[64:79]
	v_sub_f32_e32 v6, v6, v219
	v_sub_f32_e32 v5, v5, v219
	v_sub_f32_e32 v4, v4, v219
	v_sub_f32_e32 v3, v3, v219
	v_sub_f32_e32 v2, v2, v219
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[104:107], v[48:63]
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v2, v2
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[104:107], v[32:47]
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	v_mfma_f32_32x32x16_bf16 v[80:95], v[140:143], v[108:111], v[80:95]
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v8, v8
	v_mfma_f32_32x32x16_bf16 v[64:79], v[136:139], v[108:111], v[64:79]
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	v_exp_f32_e32 v11, v11
	v_mfma_f32_32x32x16_bf16 v[48:63], v[132:135], v[108:111], v[48:63]
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_exp_f32_e32 v14, v14
	v_mfma_f32_32x32x16_bf16 v[32:47], v[128:131], v[108:111], v[32:47]
	v_exp_f32_e32 v15, v15
	s_setprio 0
	s_barrier
	s_add_i32 s30, s30, 2
	s_addk_i32 s35, 0x80
	s_cmp_ge_i32 s30, s19
	v_add_u32_e32 v224, 0xffffff80, v224
	s_cbranch_scc1 .LBB0_18
	s_mov_b32 s38, s39
	s_branch .LBB0_9
.LBB0_15:
; Cluster 3: rescale branch (taken when some lane has row_max - m_row > RESCALE_THRESHOLD).
; HPP:
;   rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;   scale_output_tile<T>(v_o, rescale_m);
;   l_row *= rescale_m;
;   m_row = row_max;
; NOTE:
;   - `v_sub_f32_e32 v21, v219, v20` computes `m_row - row_max` (v219 holds
;     old m_row, v20 holds new row_max from Cluster 3 reduction).
;   - `v_exp_f32_e32 v22, v21` computes `rescale_m = exp2(m_row - row_max)`
;     (using exp2 directly since LLVM lowers `__builtin_amdgcn_exp2f` to
;     `v_exp_f32`).
;   - `v_mov_b32_e32 v219, v20` updates `m_row = row_max` for the next
;     iteration's sub_row baseline.
;   - 32 `v_pk_mul_f32 v[X:X+1], v[22:23], v[X:X+1] op_sel_hi:[0,1]` scale
;     all 64 fp32 elements of v_o by `rescale_m`. The four v_o accumulator
;     banks v[80:95]/v[64:79]/v[48:63]/v[32:47] are each scaled in 8 pk_mul
;     pairs (16 elts/bank).
;   - `v_mul_f32_e32 v214, v22, v214` updates `l_row *= rescale_m`.
;   - `s_branch .LBB0_10` returns to the loop body to execute the rest of
;     Cluster 3's MMA1 chain (the rescale path skips the row_max reduction
;     since v219 is now the new m_row).
	v_sub_f32_e32 v21, v219, v20
	v_exp_f32_e32 v22, v21
	v_mov_b32_e32 v219, v20
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[22:23], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[22:23], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[22:23], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[22:23], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81] op_sel_hi:[0,1]
	v_mul_f32_e32 v214, v22, v214
	s_branch .LBB0_10
.LBB0_16:
; Cluster 7: rescale branch (symmetric to Cluster 3 rescale).
; HPP:
;   rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;   scale_output_tile<T>(v_o, rescale_m);
;   l_row *= rescale_m;
;   m_row = row_max;
; NOTE:
;   - Same shape as .LBB0_15 with the Cluster 7 row_max in v24 (vs. v20 for
;     Cluster 3) and rescale_m landing in v26 (vs. v22). 32 `v_pk_mul_f32
;     v[X:X+1], v[26:27], v[X:X+1] op_sel_hi:[0,1]` scale v_o, and
;     `v_mul_f32_e32 v214, v26, v214` updates l_row.
;   - `s_branch .LBB0_13` returns to the Cluster 7 second half.
	v_sub_f32_e32 v25, v219, v24
	v_exp_f32_e32 v26, v25
	v_mov_b32_e32 v219, v24
	v_pk_mul_f32 v[46:47], v[26:27], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[26:27], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[26:27], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[26:27], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[26:27], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[26:27], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[26:27], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[26:27], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[26:27], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[26:27], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[26:27], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[26:27], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[26:27], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[26:27], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[26:27], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[26:27], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[26:27], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[26:27], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[26:27], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[26:27], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[26:27], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95], v[26:27], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[26:27], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[26:27], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[26:27], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[26:27], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[26:27], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[26:27], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[26:27], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[26:27], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[26:27], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[26:27], v[80:81] op_sel_hi:[0,1]
	v_mul_f32_e32 v214, v26, v214
	s_branch .LBB0_13
.LBB0_17:
; Zero-trip-count loop exit (taken when max_num_tiles <= 4 means no main-loop iteration ran).
; HPP:
;   /* If the for-loop iteration count is zero, v_o, l_row are still in
;      their pre-loop state (l_row from prologue first tile sum, v_o = 0).
;      LLVM emits a separate fixup block to re-clear v_o and l_row to
;      defined zero values since the main-loop preheader's clear has been
;      sunk past the loop entry guard. */
; NOTE:
;   - 64 `v_mov_b32_e32 vX, v47` instructions zero all 4 v_o accumulator
;     banks (v[32:47]/v[48:63]/v[64:79]/v[80:95]) using v47=0 as the source
;     register.
;   - `v_mov_b32_e32 v214, v47` zeroes l_row.
;   - `s_branch .LBB0_19` jumps to the epilogue entry shared with the normal
;     exit at .LBB0_18.
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, v47
	v_mov_b32_e32 v45, v47
	v_mov_b32_e32 v44, v47
	v_mov_b32_e32 v43, v47
	v_mov_b32_e32 v42, v47
	v_mov_b32_e32 v41, v47
	v_mov_b32_e32 v40, v47
	v_mov_b32_e32 v39, v47
	v_mov_b32_e32 v38, v47
	v_mov_b32_e32 v37, v47
	v_mov_b32_e32 v36, v47
	v_mov_b32_e32 v35, v47
	v_mov_b32_e32 v34, v47
	v_mov_b32_e32 v33, v47
	v_mov_b32_e32 v32, v47
	v_mov_b32_e32 v63, v47
	v_mov_b32_e32 v62, v47
	v_mov_b32_e32 v61, v47
	v_mov_b32_e32 v60, v47
	v_mov_b32_e32 v59, v47
	v_mov_b32_e32 v58, v47
	v_mov_b32_e32 v57, v47
	v_mov_b32_e32 v56, v47
	v_mov_b32_e32 v55, v47
	v_mov_b32_e32 v54, v47
	v_mov_b32_e32 v53, v47
	v_mov_b32_e32 v52, v47
	v_mov_b32_e32 v51, v47
	v_mov_b32_e32 v50, v47
	v_mov_b32_e32 v49, v47
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v79, v47
	v_mov_b32_e32 v78, v47
	v_mov_b32_e32 v77, v47
	v_mov_b32_e32 v76, v47
	v_mov_b32_e32 v75, v47
	v_mov_b32_e32 v74, v47
	v_mov_b32_e32 v73, v47
	v_mov_b32_e32 v72, v47
	v_mov_b32_e32 v71, v47
	v_mov_b32_e32 v70, v47
	v_mov_b32_e32 v69, v47
	v_mov_b32_e32 v68, v47
	v_mov_b32_e32 v67, v47
	v_mov_b32_e32 v66, v47
	v_mov_b32_e32 v65, v47
	v_mov_b32_e32 v64, v47
	v_mov_b32_e32 v95, v47
	v_mov_b32_e32 v94, v47
	v_mov_b32_e32 v93, v47
	v_mov_b32_e32 v92, v47
	v_mov_b32_e32 v91, v47
	v_mov_b32_e32 v90, v47
	v_mov_b32_e32 v89, v47
	v_mov_b32_e32 v88, v47
	v_mov_b32_e32 v87, v47
	v_mov_b32_e32 v86, v47
	v_mov_b32_e32 v85, v47
	v_mov_b32_e32 v84, v47
	v_mov_b32_e32 v83, v47
	v_mov_b32_e32 v82, v47
	v_mov_b32_e32 v81, v47
	v_mov_b32_e32 v80, v47
	v_mov_b32_e32 v214, v47
	s_branch .LBB0_19
.LBB0_18:
; Normal loop exit path (taken when j reaches max_num_tiles - 1).
; HPP:
;   /* PHI fix-up: copy the loop-carry registers v220/v221 into v176/v177
;      so the epilogue can read them via its single liveness convention.
;      These two VGPRs carry coordinate components that the epilogue
;      reuses for its first tile-row LDS read. */
; NOTE:
;   - `v_mov_b32_e32 v176, v220` + `v_mov_b32_e32 v177, v221` are LLVM
;     PHI-elimination copies inserted by the register allocator to unify the
;     two loop-exit paths (.LBB0_17 zero-trip vs. normal exit) at .LBB0_19.
	v_mov_b32_e32 v176, v220
	v_mov_b32_e32 v177, v221
.LBB0_19:
; Epilogue Cluster 0: (Ep C0): V async + K LDS read + barrier.
; HPP (lines 565-571):
;   async_load<T::VEC_KV>(g_v, s_v[1].ptr, u_gv, u_sv, kv_tile(max_num_tiles - 3));
;   v_k = load<T::VEC_KV>(s_k[1], u_rk);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - The epilogue drains the final 3 KV tiles (max_num_tiles-3, max_num_tiles-2,
;     max_num_tiles-1) using 14 clusters (Ep C0..C13) that pipeline the same
;     loads/MMA/softmax/rescale shape as the main loop but with `rescale_m`
;     always applied (no `all_below` ballot shortcut) and the final V tr_load
;     stripped of any post-K loads.
;   - `s_lshl_b32 s27, s24, 6` = `KV_TILE_SIZE * stride_kv_n` (6 = log2(64)).
;   - `s_add_i32 s28, s16, -3` = `max_num_tiles - 3` (= the first epilogue
;     tile index when max_num_tiles is even; for odd values the kernel's
;     even/odd parity branching at HPP line 481-482 selects this same path).
;   - `s_mul_i32 s6, s28, s27` + `s_lshl_b32 s10, s6, 1` = byte offset for
;     `kv_tile(max_num_tiles - 3)` gmem load.
;   - Two `buffer_load_dwordx4 v211/v212, s[4:7], s10 offen lds` issue
;     V[max_num_tiles-3] async into s_v[1]; m0 alternates s17/s18.
;   - `v_add3_u32 v96, v218, v208, s6` (s6 = 0x8500 = smem_buffer_elems*2)
;     then `v_add_u32_e32 v223, v96, v217` derives the K[max_num_tiles-3]
;     LDS read base for s_k[1].
;   - 16 `ds_read_b128 v[X:X+3], v223 offset:Y` form `v_k = load<VEC_KV>(s_k[1],
;     u_rk)` with the same offset pattern as the main loop's Cluster 0.
;   - `s_waitcnt vmcnt(4) lgkmcnt(0)` + `s_barrier` close Ep C0.
	s_lshl_b32 s27, s24, 6
	s_add_i32 s28, s16, -3
	s_mul_i32 s6, s28, s27
	s_mov_b32 m0, s17
	s_lshl_b32 s10, s6, 1
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v211, s[4:7], s10 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v212, s[4:7], s10 offen lds
	s_mov_b32 s6, 0x8500
	v_add3_u32 v96, v218, v208, s6
	v_add_u32_e32 v223, v96, v217
	ds_read_b128 v[96:99], v223
	ds_read_b128 v[112:115], v223 offset:32
	ds_read_b128 v[116:119], v223 offset:64
	ds_read_b128 v[120:123], v223 offset:96
	ds_read_b128 v[124:127], v223 offset:8320
	ds_read_b128 v[178:181], v223 offset:8352
	ds_read_b128 v[182:185], v223 offset:8384
	ds_read_b128 v[186:189], v223 offset:8416
	ds_read_b128 v[128:131], v223 offset:512
	ds_read_b128 v[190:193], v223 offset:544
	ds_read_b128 v[194:197], v223 offset:576
	ds_read_b128 v[198:201], v223 offset:608
	ds_read_b128 v[202:205], v223 offset:8832
	ds_read_b128 v[224:227], v223 offset:8864
	ds_read_b128 v[228:231], v223 offset:8896
	ds_read_b128 v[232:235], v223 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Epilogue Cluster 1: (Ep C1): MMA0 + exp2 half-2 + sum + cast P + barrier.
; HPP (lines 574-583):
;   v_s[1] = mma0(v_q, v_k);
;   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
;   l_row += attn_sum<T>(v_s[0]);
;   v_p = opus::cast<D_ATTN>(v_s[0]);
;   asm volatile("" : "+v"(v_p) ::);
;   sched_barrier_exp_pairs<6, 3, 5>();
;   sched_barrier_pairs<10, 5, 5>();
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Same shape as the main loop's Cluster 1: 16 MFMA (v_s[1] into
;     v[96:143]) interleaved with 16 v_exp (second half of v_s[0]), 32
;     v_add_f32 for attn_sum into v218 + permlane swap, and 16
;     v_cvt_pk_bf16_f32 for v_p in v[112:127].
;   - The trailing `s_barrier` is the source-level barrier on HPP line 582.
	v_mfma_f32_32x32x16_bf16 v[96:111], v[96:99], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	v_mfma_f32_32x32x16_bf16 v[128:143], v[128:131], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[96:111], v[112:115], v[148:151], v[96:111]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_mfma_f32_32x32x16_bf16 v[128:143], v[190:193], v[148:151], v[128:143]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_mfma_f32_32x32x16_bf16 v[96:111], v[116:119], v[152:155], v[96:111]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_mfma_f32_32x32x16_bf16 v[128:143], v[194:197], v[152:155], v[128:143]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[96:111], v[120:123], v[156:159], v[96:111]
	v_add_f32_e32 v112, v1, v0
	v_add_f32_e32 v112, v112, v2
	v_add_f32_e32 v112, v112, v3
	v_add_f32_e32 v112, v112, v4
	v_add_f32_e32 v112, v112, v5
	v_mfma_f32_32x32x16_bf16 v[128:143], v[198:201], v[156:159], v[128:143]
	v_add_f32_e32 v112, v112, v6
	v_add_f32_e32 v112, v112, v7
	v_add_f32_e32 v112, v112, v8
	v_add_f32_e32 v112, v112, v9
	v_add_f32_e32 v112, v112, v10
	v_mfma_f32_32x32x16_bf16 v[96:111], v[124:127], v[160:163], v[96:111]
	v_add_f32_e32 v112, v112, v11
	v_add_f32_e32 v112, v112, v12
	v_add_f32_e32 v112, v112, v13
	v_add_f32_e32 v112, v112, v14
	v_add_f32_e32 v112, v112, v15
	v_mfma_f32_32x32x16_bf16 v[128:143], v[202:205], v[160:163], v[128:143]
	v_add_f32_e32 v112, v112, v16
	v_add_f32_e32 v112, v112, v17
	v_add_f32_e32 v112, v112, v18
	v_add_f32_e32 v112, v112, v19
	v_add_f32_e32 v112, v112, v20
	v_mfma_f32_32x32x16_bf16 v[96:111], v[178:181], v[164:167], v[96:111]
	v_add_f32_e32 v112, v112, v21
	v_add_f32_e32 v112, v112, v22
	v_add_f32_e32 v112, v112, v23
	v_add_f32_e32 v112, v112, v24
	v_add_f32_e32 v112, v112, v25
	v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[164:167], v[128:143]
	v_add_f32_e32 v112, v112, v26
	v_add_f32_e32 v112, v112, v27
	v_add_f32_e32 v112, v112, v28
	v_add_f32_e32 v112, v112, v29
	v_add_f32_e32 v112, v112, v30
	v_mfma_f32_32x32x16_bf16 v[96:111], v[182:185], v[168:171], v[96:111]
	v_add_f32_e32 v218, v112, v31
	v_mov_b32_e32 v220, v218
	s_nop 1
	v_permlane32_swap_b32_e64 v218, v220 bound_ctrl:1
	v_cvt_pk_bf16_f32 v119, v14, v15
	v_cvt_pk_bf16_f32 v118, v12, v13
	v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[168:171], v[128:143]
	v_cvt_pk_bf16_f32 v117, v10, v11
	v_cvt_pk_bf16_f32 v116, v8, v9
	v_cvt_pk_bf16_f32 v115, v6, v7
	v_cvt_pk_bf16_f32 v114, v4, v5
	v_cvt_pk_bf16_f32 v113, v2, v3
	v_mfma_f32_32x32x16_bf16 v[96:111], v[186:189], v[172:175], v[96:111]
	v_cvt_pk_bf16_f32 v112, v0, v1
	v_cvt_pk_bf16_f32 v127, v30, v31
	v_cvt_pk_bf16_f32 v126, v28, v29
	v_cvt_pk_bf16_f32 v125, v26, v27
	v_cvt_pk_bf16_f32 v124, v24, v25
	v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[172:175], v[128:143]
	v_cvt_pk_bf16_f32 v123, v22, v23
	v_cvt_pk_bf16_f32 v122, v20, v21
	v_cvt_pk_bf16_f32 v121, v18, v19
	v_cvt_pk_bf16_f32 v120, v16, v17
	;;#ASMSTART
	;;#ASMEND
	s_barrier

; Epilogue Cluster 2: (Ep C2): K async + V tr_load + causal mask + barrier.
; HPP (lines 586-598):
;   async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(max_num_tiles - 1));
;   v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);
;   if constexpr (T::CAUSAL) {
;       const int kv_end_pos = (max_num_tiles - 2) * T::KV_TILE_SIZE;
;       if (q_start_pos < kv_end_pos) {
;           attn_mask_causal_tile<T>(v_s[1], q_start_pos, max_num_tiles - 3, neg_inf_v, lane_id);
;       }
;   }
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - `s_mul_i32 s6, s19, s27` (s19 = max_num_tiles-1, s27 =
;     KV_TILE_SIZE*stride_kv_n) + `s_lshl_b32 s24, s6, 1` builds the
;     gmem byte offset for `kv_tile(max_num_tiles - 1)` K async load.
;     The two `buffer_load_dwordx4 v211/v212, s[8:11], s24 offen lds` issue
;     K[max_num_tiles-1] async into s_k[1]; m0 alternates s26/s25.
;   - `v_add_u32_e32 v224, v176, v177` + `v_add3_u32 v221, v224, v216, s6`
;     (s6 = 0x4100 = s_v[0] base) builds the V[max_num_tiles-3] LDS read
;     base in v221 (uses s_v[0]).
;   - 32 `ds_read_b64_tr_b16 v[X:X+1], v221 offset:Y` form
;     `v_v = tr_load<VEC_TR_V>(s_v[0], u_rv)`.
;   - `s_cbranch_scc0 .LBB0_21` (forward branch later) skips the causal-mask
;     body when `q_start_pos >= (max_num_tiles-2) * KV_TILE_SIZE`. The mask
;     body has the same 32-cndmask shape as the main loop's Cluster 6 mask
;     (targeting v_s[1] = v[96:143] this time).
;   - .LBB0_21 (causal-mask rejoin) is followed by `s_waitcnt vmcnt(4)
;     lgkmcnt(0)` + `s_barrier` closing Ep C2.
	s_mul_i32 s6, s19, s27
	s_mov_b32 m0, s26
	s_lshl_b32 s24, s6, 1
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	buffer_load_dwordx4 v211, s[8:11], s24 offen lds
	s_mov_b32 m0, s25
	v_add_u32_e32 v224, v176, v177
	buffer_load_dwordx4 v212, s[8:11], s24 offen lds
	s_movk_i32 s6, 0x4100
	v_add3_u32 v221, v224, v216, s6
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v221 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[4:5], v221 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[6:7], v221 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0:1], v221 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[2:3], v221 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v221 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v221 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v221 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[8:9], v221 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[10:11], v221 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v221 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v221 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v221 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v221 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v221 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v221 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[12:13], v221 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v221 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v221 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v221 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v221 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v221 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v221 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v221 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v221 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v221 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v221 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v221 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v221 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v221 offset:0x25c0

	;;#ASMEND
	s_add_i32 s6, s16, -2
	s_lshl_b32 s8, s6, 6
	s_cmp_lt_i32 s13, s8
	v_mul_i32_i24_e32 v222, -4, v209
	s_cbranch_scc0 .LBB0_21
	s_lshl_b32 s7, s28, 6
	v_subrev_u32_e32 v208, s7, v222
	v_add_u32_e32 v208, v213, v208
	v_mov_b32_e32 v225, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 0
	v_cmp_lt_i32_e64 s[28:29], v208, 1
	v_cndmask_b32_e64 v96, v96, v225, s[10:11]
	v_cndmask_b32_e64 v97, v97, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 2
	v_cmp_lt_i32_e64 s[28:29], v208, 3
	v_cndmask_b32_e64 v98, v98, v225, s[10:11]
	v_cndmask_b32_e64 v99, v99, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 8
	v_cmp_lt_i32_e64 s[28:29], v208, 9
	v_cndmask_b32_e64 v100, v100, v225, s[10:11]
	v_cndmask_b32_e64 v101, v101, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 10
	v_cmp_lt_i32_e64 s[28:29], v208, 11
	v_cndmask_b32_e64 v102, v102, v225, s[10:11]
	v_cndmask_b32_e64 v103, v103, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 16
	v_cmp_lt_i32_e64 s[28:29], v208, 17
	v_cndmask_b32_e64 v104, v104, v225, s[10:11]
	v_cndmask_b32_e64 v105, v105, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 18
	v_cmp_lt_i32_e64 s[28:29], v208, 19
	v_cndmask_b32_e64 v106, v106, v225, s[10:11]
	v_cndmask_b32_e64 v107, v107, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 24
	v_cmp_lt_i32_e64 s[28:29], v208, 25
	v_cndmask_b32_e64 v108, v108, v225, s[10:11]
	v_cndmask_b32_e64 v109, v109, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 26
	v_cmp_lt_i32_e64 s[28:29], v208, 27
	v_cndmask_b32_e64 v110, v110, v225, s[10:11]
	v_cndmask_b32_e64 v111, v111, v225, s[28:29]
	
	;;#ASMEND
	v_subrev_u32_e32 v208, 32, v208
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 0
	v_cmp_lt_i32_e64 s[28:29], v208, 1
	v_cndmask_b32_e64 v128, v128, v225, s[10:11]
	v_cndmask_b32_e64 v129, v129, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 2
	v_cmp_lt_i32_e64 s[28:29], v208, 3
	v_cndmask_b32_e64 v130, v130, v225, s[10:11]
	v_cndmask_b32_e64 v131, v131, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 8
	v_cmp_lt_i32_e64 s[28:29], v208, 9
	v_cndmask_b32_e64 v132, v132, v225, s[10:11]
	v_cndmask_b32_e64 v133, v133, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 10
	v_cmp_lt_i32_e64 s[28:29], v208, 11
	v_cndmask_b32_e64 v134, v134, v225, s[10:11]
	v_cndmask_b32_e64 v135, v135, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 16
	v_cmp_lt_i32_e64 s[28:29], v208, 17
	v_cndmask_b32_e64 v136, v136, v225, s[10:11]
	v_cndmask_b32_e64 v137, v137, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 18
	v_cmp_lt_i32_e64 s[28:29], v208, 19
	v_cndmask_b32_e64 v138, v138, v225, s[10:11]
	v_cndmask_b32_e64 v139, v139, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 24
	v_cmp_lt_i32_e64 s[28:29], v208, 25
	v_cndmask_b32_e64 v140, v140, v225, s[10:11]
	v_cndmask_b32_e64 v141, v141, v225, s[28:29]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[10:11], v208, 26
	v_cmp_lt_i32_e64 s[28:29], v208, 27
	v_cndmask_b32_e64 v142, v142, v225, s[10:11]
	v_cndmask_b32_e64 v143, v143, v225, s[28:29]
	
	;;#ASMEND
.LBB0_21:
; Epilogue Cluster 3: (Ep C3) start: setprio 1 + MMA1 (full 4 step_k) + always-rescale + sub_row + exp2 + scale_output_tile + setprio 0 + barrier.
; HPP (lines 600-618):
;   __builtin_amdgcn_s_setprio(1);
;   v_o = mma1(v_p, v_v, v_o);
;   D_ACC row_max = max(m_row, attn_row_max<T>(v_s[1]));
;   rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;   m_row = row_max;
;   attn_sub_row<T>(v_s[1], row_max);
;   asm volatile("" : "+v"(v_s[1]) ::);
;   attn_exp2_slice<T, 0, s_half_len>(v_s[1]);
;   sched_barrier_pairs<10, 5, 6>();
;   sched_barrier_exp_pairs<6, 3, 6>();
;   __builtin_amdgcn_sched_barrier(0);
;   scale_output_tile<T>(v_o, rescale_m);
;   auto* v_o_pin = reinterpret_cast<vector_t<fp32_t, 16>*>(&v_o);
;   asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]), "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
;   __builtin_amdgcn_s_setprio(0);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Unlike the main loop, the epilogue's `row_max` reduction folds the
;     running `m_row` (v219) directly via `v_max3_f32 v225, v219, v16, v17`
;     so `row_max = max(m_row, attn_row_max(v_s[1]))` in a single 3-operand
;     v_max3. The `if (all_below)` shortcut from the loop is removed because
;     the epilogue always rescales for correctness on the final partial sums.
;   - `v_sub_f32_e32 v112, v219, v225` = `m_row - row_max`; the first v_exp
;     (`v_exp_f32_e32 v208, v112`) computes `rescale_m`.
;   - 16 MFMA `v_mfma_f32_32x32x16_bf16` execute the full
;     `v_o = mma1(v_p, v_v, v_o)` (4 step_k * 4 wave-tiles = 16 MFMA).
;   - 32 `v_sub_f32_e32 vX, vY, v225` implement `attn_sub_row(v_s[1],
;     row_max)`; `;;#ASMSTART/ASMEND` is the v_s[1] anchor.
;   - 15 additional `v_exp_f32_e32 v200..v207, v219, v226..v232, vX` form
;     `attn_exp2_slice<0, s_half_len>(v_s[1])` (v208 already burned for
;     rescale_m, so 15 of the 16 exp ops land in v200-v207, v219, v226-v232).
;   - 32 `v_pk_mul_f32 v[X:X+1], v[208:209], v[X:X+1] op_sel_hi:[0,1]`
;     scale all 64 fp32 elements of v_o by rescale_m (v208) — the
;     `scale_output_tile` step inside the cluster (vs. between clusters
;     in the loop). v209 is don't-care due to op_sel_hi:[0,1].
;   - The trailing `;;#ASMSTART/ASMEND` is the v_o_pin asm-volatile anchor
;     that LLVM uses to keep the v_o registers live (otherwise the four
;     16-element banks could be CSE'd or split across other VGPRs).
;   - `s_setprio 0` + `s_barrier` close Ep C3.
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_setprio 1
	s_mov_b32 s10, 0xf149f2ca
	v_mfma_f32_32x32x16_bf16 v[80:95], v[28:31], v[112:115], v[80:95]
	v_max3_f32 v28, v96, s10, v97
	v_max3_f32 v28, v28, v98, v99
	v_max3_f32 v28, v28, v100, v101
	v_max3_f32 v28, v28, v102, v103
	v_max3_f32 v28, v28, v104, v105
	v_mfma_f32_32x32x16_bf16 v[64:79], v[16:19], v[112:115], v[64:79]
	v_max3_f32 v16, v28, v106, v107
	v_max3_f32 v16, v16, v108, v109
	v_max3_f32 v16, v16, v110, v111
	v_max3_f32 v16, v16, v128, v129
	v_max3_f32 v16, v16, v130, v131
	v_mfma_f32_32x32x16_bf16 v[48:63], v[20:23], v[112:115], v[48:63]
	v_max3_f32 v16, v16, v132, v133
	v_max3_f32 v16, v16, v134, v135
	v_max3_f32 v16, v16, v136, v137
	v_max3_f32 v16, v16, v138, v139
	v_max3_f32 v16, v16, v140, v141
	v_mfma_f32_32x32x16_bf16 v[32:47], v[24:27], v[112:115], v[32:47]
	v_max3_f32 v16, v16, v142, v143
	v_mov_b32_e32 v17, v16
	s_nop 1
	v_permlane32_swap_b32_e64 v16, v17 bound_ctrl:1
	v_max3_f32 v225, v219, v16, v17
	v_sub_f32_e32 v112, v219, v225
	v_mfma_f32_32x32x16_bf16 v[80:95], v[4:7], v[116:119], v[80:95]
	v_sub_f32_e32 v31, v143, v225
	v_sub_f32_e32 v30, v142, v225
	v_sub_f32_e32 v29, v141, v225
	v_sub_f32_e32 v28, v140, v225
	v_sub_f32_e32 v27, v139, v225
	v_mfma_f32_32x32x16_bf16 v[64:79], v[8:11], v[116:119], v[64:79]
	v_sub_f32_e32 v26, v138, v225
	v_sub_f32_e32 v25, v137, v225
	v_sub_f32_e32 v24, v136, v225
	v_sub_f32_e32 v23, v135, v225
	v_sub_f32_e32 v22, v134, v225
	v_mfma_f32_32x32x16_bf16 v[48:63], v[12:15], v[116:119], v[48:63]
	v_sub_f32_e32 v21, v133, v225
	v_sub_f32_e32 v20, v132, v225
	v_sub_f32_e32 v19, v131, v225
	v_sub_f32_e32 v18, v130, v225
	v_sub_f32_e32 v17, v129, v225
	v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[116:119], v[32:47]
	v_sub_f32_e32 v16, v128, v225
	v_sub_f32_e32 v15, v111, v225
	v_sub_f32_e32 v14, v110, v225
	v_sub_f32_e32 v13, v109, v225
	v_sub_f32_e32 v12, v108, v225
	v_mfma_f32_32x32x16_bf16 v[80:95], v[0:3], v[120:123], v[80:95]
	v_sub_f32_e32 v11, v107, v225
	v_sub_f32_e32 v10, v106, v225
	v_sub_f32_e32 v9, v105, v225
	v_sub_f32_e32 v8, v104, v225
	v_sub_f32_e32 v7, v103, v225
	v_sub_f32_e32 v1, v97, v225
	v_sub_f32_e32 v0, v96, v225
	v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[120:123], v[64:79]
	v_sub_f32_e32 v6, v102, v225
	v_sub_f32_e32 v5, v101, v225
	v_sub_f32_e32 v4, v100, v225
	v_sub_f32_e32 v3, v99, v225
	v_sub_f32_e32 v2, v98, v225
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[120:123], v[48:63]
	v_exp_f32_e32 v208, v112
	v_exp_f32_e32 v200, v0
	v_exp_f32_e32 v201, v1
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[120:123], v[32:47]
	v_exp_f32_e32 v202, v2
	v_exp_f32_e32 v203, v3
	v_exp_f32_e32 v204, v4
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[124:127], v[80:95]
	v_exp_f32_e32 v205, v5
	v_exp_f32_e32 v206, v6
	v_exp_f32_e32 v207, v7
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[124:127], v[64:79]
	v_exp_f32_e32 v219, v8
	v_exp_f32_e32 v226, v9
	v_exp_f32_e32 v227, v10
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[124:127], v[48:63]
	v_exp_f32_e32 v228, v11
	v_exp_f32_e32 v229, v12
	v_exp_f32_e32 v230, v13
	v_mfma_f32_32x32x16_bf16 v[32:47], v[188:191], v[124:127], v[32:47]
	v_exp_f32_e32 v231, v14
	v_exp_f32_e32 v232, v15
	v_pk_mul_f32 v[94:95], v[208:209], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[208:209], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[208:209], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[208:209], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[208:209], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[208:209], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[208:209], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[208:209], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[208:209], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[208:209], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[208:209], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[208:209], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[208:209], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[208:209], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[208:209], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[208:209], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[208:209], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[208:209], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[208:209], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[208:209], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[208:209], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[208:209], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[208:209], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[208:209], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[208:209], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[208:209], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[208:209], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[208:209], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[208:209], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[208:209], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[208:209], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[208:209], v[32:33] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
	s_barrier

; Epilogue Cluster 4: (Ep C4): V async + K LDS read + barrier.
; HPP (lines 621-627):
;   async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(max_num_tiles - 2));
;   v_k = load<T::VEC_KV>(s_k[0], u_rk);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Mirror of the loop's Cluster 4 with V[max_num_tiles-2] async into s_v[0]
;     and K[max_num_tiles-2] LDS read from s_k[0] (the K[max_num_tiles-2]
;     async was issued in Ep C0 + Ep C2; this cluster reads it from LDS).
;   - `s_mul_i32 s6, s6, s27` + `s_lshl_b32 s9, s6, 1` builds the gmem byte
;     offset for `kv_tile(max_num_tiles - 2)`; m0 alternates s21/s20.
;   - `v_add_u32_e32 v4, v215, v217` builds the K[0] LDS read base (uses
;     v215 = u_rk_base from preheader).
;   - 16 `ds_read_b128 v[X:X+3], v4 offset:Y` form v_k load with the same
;     offsets as the loop's Cluster 4.
	s_mul_i32 s6, s6, s27
	s_mov_b32 m0, s21
	s_lshl_b32 s9, s6, 1
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v211, s[4:7], s9 offen lds
	s_mov_b32 m0, s20
	v_add_u32_e32 v4, v215, v217
	buffer_load_dwordx4 v212, s[4:7], s9 offen lds
	ds_read_b128 v[0:3], v4
	ds_read_b128 v[96:99], v4 offset:32
	ds_read_b128 v[100:103], v4 offset:64
	ds_read_b128 v[104:107], v4 offset:96
	ds_read_b128 v[108:111], v4 offset:8320
	ds_read_b128 v[128:131], v4 offset:8352
	ds_read_b128 v[132:135], v4 offset:8384
	ds_read_b128 v[136:139], v4 offset:8416
	ds_read_b128 v[112:115], v4 offset:512
	ds_read_b128 v[140:143], v4 offset:544
	ds_read_b128 v[176:179], v4 offset:576
	ds_read_b128 v[180:183], v4 offset:608
	ds_read_b128 v[184:187], v4 offset:8832
	ds_read_b128 v[188:191], v4 offset:8864
	ds_read_b128 v[192:195], v4 offset:8896
	ds_read_b128 v[196:199], v4 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Epilogue Cluster 5: (Ep C5): MMA0 + l_row *= rescale_m + exp2 half-2 + sum + cast P + barrier.
; HPP (lines 630-640):
;   v_s[0] = mma0(v_q, v_k);
;   l_row *= rescale_m;
;   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1]);
;   l_row += attn_sum<T>(v_s[1]);
;   v_p = opus::cast<D_ATTN>(v_s[1]);
;   asm volatile("" : "+v"(v_p) ::);
;   sched_barrier_exp_pairs<6, 3, 7>();
;   sched_barrier_pairs<10, 5, 7>();
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Mirror of the loop's Cluster 5 but additionally folds the Ep C3
;     `rescale_m` (held in v208) into l_row before adding the new tile sum:
;     this is the `l_row *= rescale_m` step that the loop fused into the
;     rescale branch.
;   - 16 MFMA `v_mfma_f32_32x32x16_bf16 v[0:15]/v[112:127], v[X:X+3],
;     v[Y:Y+3]` form `v_s[0] = mma0(v_q, v_k)`.
;   - 16 `v_exp_f32_e32 vX, vX` execute the second half of exp2 on v_s[1].
;   - 32 `v_add_f32_e32 v96, v96, vX` + `v_permlane32_swap_b32_e64 v215,
;     v217 bound_ctrl:1` form `attn_sum(v_s[1])` into v215. The `l_row *=
;     rescale_m + l_row += tile_sum` fold happens in Ep C6's preamble
;     (LLVM hoisted it down for scheduling).
;   - 16 `v_cvt_pk_bf16_f32 v[96:111], vL, vH` cast v_s[1] to bf16 for v_p.
	v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	v_mfma_f32_32x32x16_bf16 v[112:127], v[112:115], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[148:151], v[0:15]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_mfma_f32_32x32x16_bf16 v[112:127], v[140:143], v[148:151], v[112:127]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_mfma_f32_32x32x16_bf16 v[0:15], v[100:103], v[152:155], v[0:15]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_mfma_f32_32x32x16_bf16 v[112:127], v[176:179], v[152:155], v[112:127]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[0:15], v[104:107], v[156:159], v[0:15]
	v_add_f32_e32 v96, v201, v200
	v_add_f32_e32 v96, v96, v202
	v_add_f32_e32 v96, v96, v203
	v_add_f32_e32 v96, v96, v204
	v_add_f32_e32 v96, v96, v205
	v_mfma_f32_32x32x16_bf16 v[112:127], v[180:183], v[156:159], v[112:127]
	v_add_f32_e32 v96, v96, v206
	v_add_f32_e32 v96, v96, v207
	v_add_f32_e32 v96, v96, v219
	v_add_f32_e32 v96, v96, v226
	v_add_f32_e32 v96, v96, v227
	v_mfma_f32_32x32x16_bf16 v[0:15], v[108:111], v[160:163], v[0:15]
	v_add_f32_e32 v96, v96, v228
	v_add_f32_e32 v96, v96, v229
	v_add_f32_e32 v96, v96, v230
	v_add_f32_e32 v96, v96, v231
	v_add_f32_e32 v96, v96, v232
	v_mfma_f32_32x32x16_bf16 v[112:127], v[184:187], v[160:163], v[112:127]
	v_add_f32_e32 v96, v96, v16
	v_add_f32_e32 v96, v96, v17
	v_add_f32_e32 v96, v96, v18
	v_add_f32_e32 v96, v96, v19
	v_add_f32_e32 v96, v96, v20
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[164:167], v[0:15]
	v_add_f32_e32 v96, v96, v21
	v_add_f32_e32 v96, v96, v22
	v_add_f32_e32 v96, v96, v23
	v_add_f32_e32 v96, v96, v24
	v_add_f32_e32 v96, v96, v25
	v_mfma_f32_32x32x16_bf16 v[112:127], v[188:191], v[164:167], v[112:127]
	v_add_f32_e32 v96, v96, v26
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[168:171], v[0:15]
	v_add_f32_e32 v215, v96, v31
	v_mov_b32_e32 v217, v215
	s_nop 1
	v_permlane32_swap_b32_e64 v215, v217 bound_ctrl:1
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	v_mfma_f32_32x32x16_bf16 v[112:127], v[192:195], v[168:171], v[112:127]
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[172:175], v[0:15]
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v231, v232
	v_cvt_pk_bf16_f32 v102, v229, v230
	v_cvt_pk_bf16_f32 v101, v227, v228
	v_cvt_pk_bf16_f32 v100, v219, v226
	v_mfma_f32_32x32x16_bf16 v[112:127], v[196:199], v[172:175], v[112:127]
	v_cvt_pk_bf16_f32 v99, v206, v207
	v_cvt_pk_bf16_f32 v98, v204, v205
	v_cvt_pk_bf16_f32 v97, v202, v203
	v_cvt_pk_bf16_f32 v96, v200, v201
	;;#ASMSTART
	;;#ASMEND
	s_barrier

; Epilogue Cluster 6: (Ep C6): V tr_load + causal mask + barrier (no K async — already issued in Ep C4).
; HPP (lines 643-654):
;   v_v = tr_load<T::VEC_TR_V>(s_v[1], u_rv);
;   if constexpr (T::CAUSAL) {
;       const int kv_end_pos = (max_num_tiles - 1) * T::KV_TILE_SIZE;
;       if (q_start_pos < kv_end_pos) {
;           attn_mask_causal_tile<T>(v_s[0], q_start_pos, max_num_tiles - 2, neg_inf_v, lane_id);
;       }
;   }
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - `s_mov_b32 s6, 0xc600` + `v_add3_u32 v216, v224, v216, s6` builds the
;     V[max_num_tiles-2] LDS read base via s_v[1] (0xc600 = s_v[1] base
;     offset).
;   - 32 `ds_read_b64_tr_b16 v[X:X+1], v216 offset:Y` form `v_v = tr_load
;     <VEC_TR_V>(s_v[1], u_rv)`.
;   - Causal-mask body has the same 32 cndmask groups as the loop's
;     Cluster 6, targeting v_s[0] (in v[0:15] + v[112:127]).
;   - `s_cbranch_scc1 .LBB0_23` skips the causal mask when this wave's Q
;     rows are past the right edge of the (max_num_tiles-2) tile.
;   - .LBB0_23 (causal-mask rejoin) is followed by `s_waitcnt vmcnt(2)
;     lgkmcnt(0)` (only 2 outstanding VMEM = V[max_num_tiles-2] async; no K
;     async this cluster) + `s_barrier` closing Ep C6.
	s_mov_b32 s6, 0xc600
	v_add3_u32 v216, v224, v216, s6
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v216 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v216 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v216 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v216 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v216 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v216 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v216 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v216 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v216 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v216 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v216 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v216 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v216 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v216 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v216 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v216 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v216 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v216 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v216 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v216 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v216 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v216 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v216 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v216 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v216 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v216 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v216 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v216 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v216 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v216 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v216 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v216 offset:0x25c0

	;;#ASMEND
	s_lshl_b32 s9, s19, 6
	s_cmp_ge_i32 s13, s9
	s_cbranch_scc1 .LBB0_23
	v_subrev_u32_e32 v219, s8, v222
	v_add_u32_e32 v219, v213, v219
	v_mov_b32_e32 v224, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 0
	v_cmp_lt_i32_e64 s[20:21], v219, 1
	v_cndmask_b32_e64 v0, v0, v224, s[6:7]
	v_cndmask_b32_e64 v1, v1, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 2
	v_cmp_lt_i32_e64 s[20:21], v219, 3
	v_cndmask_b32_e64 v2, v2, v224, s[6:7]
	v_cndmask_b32_e64 v3, v3, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 8
	v_cmp_lt_i32_e64 s[20:21], v219, 9
	v_cndmask_b32_e64 v4, v4, v224, s[6:7]
	v_cndmask_b32_e64 v5, v5, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 10
	v_cmp_lt_i32_e64 s[20:21], v219, 11
	v_cndmask_b32_e64 v6, v6, v224, s[6:7]
	v_cndmask_b32_e64 v7, v7, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 16
	v_cmp_lt_i32_e64 s[20:21], v219, 17
	v_cndmask_b32_e64 v8, v8, v224, s[6:7]
	v_cndmask_b32_e64 v9, v9, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 18
	v_cmp_lt_i32_e64 s[20:21], v219, 19
	v_cndmask_b32_e64 v10, v10, v224, s[6:7]
	v_cndmask_b32_e64 v11, v11, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 24
	v_cmp_lt_i32_e64 s[20:21], v219, 25
	v_cndmask_b32_e64 v12, v12, v224, s[6:7]
	v_cndmask_b32_e64 v13, v13, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 26
	v_cmp_lt_i32_e64 s[20:21], v219, 27
	v_cndmask_b32_e64 v14, v14, v224, s[6:7]
	v_cndmask_b32_e64 v15, v15, v224, s[20:21]
	
	;;#ASMEND
	v_subrev_u32_e32 v219, 32, v219
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 0
	v_cmp_lt_i32_e64 s[20:21], v219, 1
	v_cndmask_b32_e64 v112, v112, v224, s[6:7]
	v_cndmask_b32_e64 v113, v113, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 2
	v_cmp_lt_i32_e64 s[20:21], v219, 3
	v_cndmask_b32_e64 v114, v114, v224, s[6:7]
	v_cndmask_b32_e64 v115, v115, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 8
	v_cmp_lt_i32_e64 s[20:21], v219, 9
	v_cndmask_b32_e64 v116, v116, v224, s[6:7]
	v_cndmask_b32_e64 v117, v117, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 10
	v_cmp_lt_i32_e64 s[20:21], v219, 11
	v_cndmask_b32_e64 v118, v118, v224, s[6:7]
	v_cndmask_b32_e64 v119, v119, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 16
	v_cmp_lt_i32_e64 s[20:21], v219, 17
	v_cndmask_b32_e64 v120, v120, v224, s[6:7]
	v_cndmask_b32_e64 v121, v121, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 18
	v_cmp_lt_i32_e64 s[20:21], v219, 19
	v_cndmask_b32_e64 v122, v122, v224, s[6:7]
	v_cndmask_b32_e64 v123, v123, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 24
	v_cmp_lt_i32_e64 s[20:21], v219, 25
	v_cndmask_b32_e64 v124, v124, v224, s[6:7]
	v_cndmask_b32_e64 v125, v125, v224, s[20:21]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v219, 26
	v_cmp_lt_i32_e64 s[20:21], v219, 27
	v_cndmask_b32_e64 v126, v126, v224, s[6:7]
	v_cndmask_b32_e64 v127, v127, v224, s[20:21]
	
	;;#ASMEND
.LBB0_23:
; Epilogue Cluster 7: (Ep C7): setprio 1 + MMA1 + always-rescale + sub_row + exp2 + scale_output_tile + setprio 0 + barrier.
; HPP (lines 657-673):
;   __builtin_amdgcn_s_setprio(1);
;   v_o = mma1(v_p, v_v, v_o);
;   row_max = max(m_row, attn_row_max<T>(v_s[0]));
;   rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;   m_row = row_max;
;   attn_sub_row<T>(v_s[0], row_max);
;   asm volatile("" : "+v"(v_s[0]) ::);
;   attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
;   sched_barrier_pairs<10, 5, 8>();
;   sched_barrier_exp_pairs<6, 3, 8>();
;   __builtin_amdgcn_sched_barrier(0);
;   scale_output_tile<T>(v_o, rescale_m);
;   asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]), "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
;   __builtin_amdgcn_s_setprio(0);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Same structure as Ep C3 with score-tile swap (v_s[0] this time).
;   - 16 MFMA `v_mfma_f32_32x32x16_bf16` form `v_o = mma1(v_p, v_v, v_o)`.
;   - 16 `v_max3_f32` reduce v_s[0] (v[0:15] + v[112:127]) to per-lane max;
;     `v_max3_f32 v200, v225, v24, v25` folds the running m_row (v225) so
;     `row_max = max(m_row, attn_row_max(v_s[0]))`.
;   - `v_sub_f32_e32 v96, v225, v200` = `m_row - row_max`; the first v_exp
;     produces `rescale_m`.
;   - 32 `v_sub_f32_e32 vX, vY, v200` execute `attn_sub_row(v_s[0],
;     row_max)`; `;;#ASMSTART/ASMEND` is the v_s[0] anchor.
;   - 15 more `v_exp_f32_e32 vX, vX` form the first half of exp2(v_s[0])
;     (combined with the rescale_m exp = 16 total).
;   - 32 `v_pk_mul_f32 v[X:X+1], v[208:209], v[X:X+1] op_sel_hi:[0,1]`
;     scale v_o by rescale_m.
;   - `s_setprio 0` + `s_barrier` close Ep C7.
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[204:207], v[96:99], v[80:95]
	v_max3_f32 v204, v0, s10, v1
	v_max3_f32 v204, v204, v2, v3
	v_max3_f32 v204, v204, v4, v5
	v_max3_f32 v204, v204, v6, v7
	v_max3_f32 v204, v204, v8, v9
	v_mfma_f32_32x32x16_bf16 v[64:79], v[200:203], v[96:99], v[64:79]
	v_max3_f32 v200, v204, v10, v11
	v_max3_f32 v200, v200, v12, v13
	v_max3_f32 v200, v200, v14, v15
	v_max3_f32 v200, v200, v112, v113
	v_max3_f32 v200, v200, v114, v115
	v_mfma_f32_32x32x16_bf16 v[48:63], v[28:31], v[96:99], v[48:63]
	v_max3_f32 v28, v200, v116, v117
	v_max3_f32 v28, v28, v118, v119
	v_max3_f32 v28, v28, v120, v121
	v_max3_f32 v28, v28, v122, v123
	v_max3_f32 v28, v28, v124, v125
	v_mfma_f32_32x32x16_bf16 v[32:47], v[24:27], v[96:99], v[32:47]
	v_max3_f32 v24, v28, v126, v127
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1
	v_max3_f32 v200, v225, v24, v25
	v_sub_f32_e32 v96, v225, v200
	v_sub_f32_e32 v1, v1, v200
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[100:103], v[80:95]
	v_sub_f32_e32 v31, v127, v200
	v_sub_f32_e32 v30, v126, v200
	v_sub_f32_e32 v29, v125, v200
	v_sub_f32_e32 v28, v124, v200
	v_sub_f32_e32 v27, v123, v200
	v_sub_f32_e32 v0, v0, v200
	v_mfma_f32_32x32x16_bf16 v[64:79], v[20:23], v[100:103], v[64:79]
	v_sub_f32_e32 v26, v122, v200
	v_sub_f32_e32 v25, v121, v200
	v_sub_f32_e32 v24, v120, v200
	v_sub_f32_e32 v23, v119, v200
	v_sub_f32_e32 v22, v118, v200
	v_mfma_f32_32x32x16_bf16 v[48:63], v[192:195], v[100:103], v[48:63]
	v_sub_f32_e32 v21, v117, v200
	v_sub_f32_e32 v20, v116, v200
	v_sub_f32_e32 v19, v115, v200
	v_sub_f32_e32 v18, v114, v200
	v_sub_f32_e32 v17, v113, v200
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[100:103], v[32:47]
	v_sub_f32_e32 v16, v112, v200
	v_sub_f32_e32 v15, v15, v200
	v_sub_f32_e32 v14, v14, v200
	v_sub_f32_e32 v13, v13, v200
	v_sub_f32_e32 v12, v12, v200
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[104:107], v[80:95]
	v_sub_f32_e32 v11, v11, v200
	v_sub_f32_e32 v10, v10, v200
	v_sub_f32_e32 v9, v9, v200
	v_sub_f32_e32 v8, v8, v200
	v_sub_f32_e32 v7, v7, v200
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[104:107], v[64:79]
	v_sub_f32_e32 v6, v6, v200
	v_sub_f32_e32 v5, v5, v200
	v_sub_f32_e32 v4, v4, v200
	v_sub_f32_e32 v3, v3, v200
	v_sub_f32_e32 v2, v2, v200
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[104:107], v[48:63]
	v_exp_f32_e32 v180, v96
	v_exp_f32_e32 v181, v0
	v_exp_f32_e32 v198, v1
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[104:107], v[32:47]
	v_exp_f32_e32 v199, v2
	v_exp_f32_e32 v201, v3
	v_exp_f32_e32 v206, v4
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[108:111], v[80:95]
	v_exp_f32_e32 v207, v5
	v_exp_f32_e32 v219, v6
	v_exp_f32_e32 v224, v7
	v_mfma_f32_32x32x16_bf16 v[64:79], v[132:135], v[108:111], v[64:79]
	v_exp_f32_e32 v225, v8
	v_exp_f32_e32 v226, v9
	v_exp_f32_e32 v227, v10
	v_mfma_f32_32x32x16_bf16 v[48:63], v[136:139], v[108:111], v[48:63]
	v_exp_f32_e32 v228, v11
	v_exp_f32_e32 v229, v12
	v_exp_f32_e32 v230, v13
	v_mfma_f32_32x32x16_bf16 v[32:47], v[140:143], v[108:111], v[32:47]
	v_exp_f32_e32 v231, v14
	v_exp_f32_e32 v232, v15
	v_pk_mul_f32 v[94:95], v[180:181], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[180:181], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[180:181], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[180:181], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[180:181], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[180:181], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[180:181], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[180:181], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[180:181], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[180:181], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[180:181], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[180:181], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[180:181], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[180:181], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[180:181], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[180:181], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[180:181], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[180:181], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[180:181], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[180:181], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[180:181], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[180:181], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[180:181], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[180:181], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[180:181], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[180:181], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[180:181], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[180:181], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[180:181], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[180:181], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[180:181], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[180:181], v[32:33] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
	s_barrier

; Epilogue Cluster 8: (Ep C8): V async + K LDS read + barrier (final V async).
; HPP (lines 676-682):
;   async_load<T::VEC_KV>(g_v, s_v[1].ptr, u_gv, u_sv, kv_tile(max_num_tiles - 1));
;   v_k = load<T::VEC_KV>(s_k[1], u_rk);
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - The third (and final) V async load goes into s_v[1] with byte offset
;     s24 (s24 was overwritten in Ep C2 to hold `kv_tile(max_num_tiles - 1)`
;     gmem byte offset, which conveniently matches the V load here too).
;   - 16 `ds_read_b128 v[X:X+3], v223 offset:Y` form `v_k = load<VEC_KV>(s_k[1],
;     u_rk)` reading the K[max_num_tiles-1] tile that was prefetched in Ep C2.
;   - `s_waitcnt vmcnt(2) lgkmcnt(0)` waits for K LDS reads and leaves 2
;     outstanding V VMEM ops.
	s_mov_b32 m0, s17
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v211, s[4:7], s24 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v212, s[4:7], s24 offen lds
	ds_read_b128 v[0:3], v223
	ds_read_b128 v[96:99], v223 offset:32
	ds_read_b128 v[100:103], v223 offset:64
	ds_read_b128 v[104:107], v223 offset:96
	ds_read_b128 v[108:111], v223 offset:8320
	ds_read_b128 v[128:131], v223 offset:8352
	ds_read_b128 v[132:135], v223 offset:8384
	ds_read_b128 v[136:139], v223 offset:8416
	ds_read_b128 v[112:115], v223 offset:512
	ds_read_b128 v[140:143], v223 offset:544
	ds_read_b128 v[176:179], v223 offset:576
	ds_read_b128 v[182:185], v223 offset:608
	ds_read_b128 v[186:189], v223 offset:8832
	ds_read_b128 v[190:193], v223 offset:8864
	ds_read_b128 v[194:197], v223 offset:8896
	ds_read_b128 v[202:205], v223 offset:8928
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier

; Epilogue Cluster 9: (Ep C9): MMA0 + l_row *= rescale_m + exp2 half-2 + sum + cast P + barrier.
; HPP (lines 685-695):
;   v_s[1] = mma0(v_q, v_k);
;   l_row *= rescale_m;
;   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
;   l_row += attn_sum<T>(v_s[0]);
;   v_p = opus::cast<D_ATTN>(v_s[0]);
;   asm volatile("" : "+v"(v_p) ::);
;   sched_barrier_exp_pairs<6, 3, 9>();
;   sched_barrier_pairs<10, 5, 9>();
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - Symmetric mirror of Ep C5 with score-tile swap. 16 MFMA form
;     `v_s[1] = mma0(v_q, v_k)` into v[0:15] / v[112:127].
;   - 16 v_exp execute the second half of exp2 on v_s[0]; 32 v_add for
;     attn_sum into v176; permlane swap; 16 v_cvt_pk_bf16_f32 cast to bf16
;     for v_p in v[96:111].
;   - The `l_row *= rescale_m` from Ep C7 is folded later (LLVM hoisted
;     the multiply across the cluster boundary).
	v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	v_mfma_f32_32x32x16_bf16 v[112:127], v[112:115], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[148:151], v[0:15]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_mfma_f32_32x32x16_bf16 v[112:127], v[140:143], v[148:151], v[112:127]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_mfma_f32_32x32x16_bf16 v[0:15], v[100:103], v[152:155], v[0:15]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_mfma_f32_32x32x16_bf16 v[112:127], v[176:179], v[152:155], v[112:127]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[0:15], v[104:107], v[156:159], v[0:15]
	v_add_f32_e32 v96, v198, v181
	v_add_f32_e32 v96, v96, v199
	v_add_f32_e32 v96, v96, v201
	v_add_f32_e32 v96, v96, v206
	v_add_f32_e32 v96, v96, v207
	v_mfma_f32_32x32x16_bf16 v[112:127], v[182:185], v[156:159], v[112:127]
	v_add_f32_e32 v96, v96, v219
	v_add_f32_e32 v96, v96, v224
	v_add_f32_e32 v96, v96, v225
	v_add_f32_e32 v96, v96, v226
	v_add_f32_e32 v96, v96, v227
	v_mfma_f32_32x32x16_bf16 v[0:15], v[108:111], v[160:163], v[0:15]
	v_add_f32_e32 v96, v96, v228
	v_add_f32_e32 v96, v96, v229
	v_add_f32_e32 v96, v96, v230
	v_add_f32_e32 v96, v96, v231
	v_add_f32_e32 v96, v96, v232
	v_mfma_f32_32x32x16_bf16 v[112:127], v[186:189], v[160:163], v[112:127]
	v_add_f32_e32 v96, v96, v16
	v_add_f32_e32 v96, v96, v17
	v_add_f32_e32 v96, v96, v18
	v_add_f32_e32 v96, v96, v19
	v_add_f32_e32 v96, v96, v20
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[164:167], v[0:15]
	v_add_f32_e32 v96, v96, v21
	v_add_f32_e32 v96, v96, v22
	v_add_f32_e32 v96, v96, v23
	v_add_f32_e32 v96, v96, v24
	v_add_f32_e32 v96, v96, v25
	v_mfma_f32_32x32x16_bf16 v[112:127], v[190:193], v[164:167], v[112:127]
	v_add_f32_e32 v96, v96, v26
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[168:171], v[0:15]
	v_add_f32_e32 v176, v96, v31
	v_mov_b32_e32 v177, v176
	s_nop 1
	v_permlane32_swap_b32_e64 v176, v177 bound_ctrl:1
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	v_mfma_f32_32x32x16_bf16 v[112:127], v[194:197], v[168:171], v[112:127]
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[172:175], v[0:15]
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v231, v232
	v_cvt_pk_bf16_f32 v102, v229, v230
	v_cvt_pk_bf16_f32 v101, v227, v228
	v_cvt_pk_bf16_f32 v100, v225, v226
	v_mfma_f32_32x32x16_bf16 v[112:127], v[202:205], v[172:175], v[112:127]
	v_cvt_pk_bf16_f32 v99, v219, v224
	v_cvt_pk_bf16_f32 v98, v206, v207
	v_cvt_pk_bf16_f32 v97, v199, v201
	v_cvt_pk_bf16_f32 v96, v181, v198
	;;#ASMSTART
	;;#ASMEND
	s_barrier

; Epilogue Cluster 10: (Ep C10): V tr_load + causal mask + barrier (drain all VMEM).
; HPP (lines 698-709):
;   v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);
;   if constexpr (T::CAUSAL) {
;       const int kv_end_pos = max_num_tiles * T::KV_TILE_SIZE;
;       if (q_start_pos < kv_end_pos) {
;           attn_mask_causal_tile<T>(v_s[1], q_start_pos, max_num_tiles - 1, neg_inf_v, lane_id);
;       }
;   }
;   s_waitcnt_lgkmcnt(0_I);
;   s_waitcnt_vmcnt(0_I);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - The final V tr_load uses v221 base = s_v[0] u_rv (s_v[0] was just
;     refilled in Ep C4 with V[max_num_tiles-2]).
;   - 32 `ds_read_b64_tr_b16 v[X:X+1], v221 offset:Y` form the v_v load.
;   - The causal-mask body targets v_s[1] (the score tile from Ep C9 in
;     v[0:15] + v[112:127]) and uses the right-edge of the entire
;     sequence: `kv_end_pos = max_num_tiles * KV_TILE_SIZE`. After the
;     mask rejoin at .LBB0_25, `s_waitcnt vmcnt(0) lgkmcnt(0)` drains
;     ALL outstanding VMEM ops (this was Ep C8's V[max_num_tiles-1] async)
;     before the final MMA1 chain.
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v221 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v221 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v221 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v221 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v221 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v221 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v221 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v221 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v221 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v221 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v221 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v221 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v221 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v221 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v221 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v221 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v221 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v221 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v221 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v221 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v221 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v221 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v221 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v221 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v221 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v221 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v221 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v221 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v221 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v221 offset:0x25c0

	;;#ASMEND
	s_lshl_b32 s2, s16, 6
	s_cmp_ge_i32 s13, s2
	s_cbranch_scc1 .LBB0_25
	v_subrev_u32_e32 v178, s9, v222
	v_add_u32_e32 v178, v213, v178
	v_mov_b32_e32 v179, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 0
	v_cmp_lt_i32_e64 s[4:5], v178, 1
	v_cndmask_b32_e64 v0, v0, v179, s[2:3]
	v_cndmask_b32_e64 v1, v1, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 2
	v_cmp_lt_i32_e64 s[4:5], v178, 3
	v_cndmask_b32_e64 v2, v2, v179, s[2:3]
	v_cndmask_b32_e64 v3, v3, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 8
	v_cmp_lt_i32_e64 s[4:5], v178, 9
	v_cndmask_b32_e64 v4, v4, v179, s[2:3]
	v_cndmask_b32_e64 v5, v5, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 10
	v_cmp_lt_i32_e64 s[4:5], v178, 11
	v_cndmask_b32_e64 v6, v6, v179, s[2:3]
	v_cndmask_b32_e64 v7, v7, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 16
	v_cmp_lt_i32_e64 s[4:5], v178, 17
	v_cndmask_b32_e64 v8, v8, v179, s[2:3]
	v_cndmask_b32_e64 v9, v9, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 18
	v_cmp_lt_i32_e64 s[4:5], v178, 19
	v_cndmask_b32_e64 v10, v10, v179, s[2:3]
	v_cndmask_b32_e64 v11, v11, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 24
	v_cmp_lt_i32_e64 s[4:5], v178, 25
	v_cndmask_b32_e64 v12, v12, v179, s[2:3]
	v_cndmask_b32_e64 v13, v13, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 26
	v_cmp_lt_i32_e64 s[4:5], v178, 27
	v_cndmask_b32_e64 v14, v14, v179, s[2:3]
	v_cndmask_b32_e64 v15, v15, v179, s[4:5]
	
	;;#ASMEND
	v_subrev_u32_e32 v178, 32, v178
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 0
	v_cmp_lt_i32_e64 s[4:5], v178, 1
	v_cndmask_b32_e64 v112, v112, v179, s[2:3]
	v_cndmask_b32_e64 v113, v113, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 2
	v_cmp_lt_i32_e64 s[4:5], v178, 3
	v_cndmask_b32_e64 v114, v114, v179, s[2:3]
	v_cndmask_b32_e64 v115, v115, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 8
	v_cmp_lt_i32_e64 s[4:5], v178, 9
	v_cndmask_b32_e64 v116, v116, v179, s[2:3]
	v_cndmask_b32_e64 v117, v117, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 10
	v_cmp_lt_i32_e64 s[4:5], v178, 11
	v_cndmask_b32_e64 v118, v118, v179, s[2:3]
	v_cndmask_b32_e64 v119, v119, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 16
	v_cmp_lt_i32_e64 s[4:5], v178, 17
	v_cndmask_b32_e64 v120, v120, v179, s[2:3]
	v_cndmask_b32_e64 v121, v121, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 18
	v_cmp_lt_i32_e64 s[4:5], v178, 19
	v_cndmask_b32_e64 v122, v122, v179, s[2:3]
	v_cndmask_b32_e64 v123, v123, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 24
	v_cmp_lt_i32_e64 s[4:5], v178, 25
	v_cndmask_b32_e64 v124, v124, v179, s[2:3]
	v_cndmask_b32_e64 v125, v125, v179, s[4:5]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[2:3], v178, 26
	v_cmp_lt_i32_e64 s[4:5], v178, 27
	v_cndmask_b32_e64 v126, v126, v179, s[2:3]
	v_cndmask_b32_e64 v127, v127, v179, s[4:5]
	
	;;#ASMEND
.LBB0_25:
; Epilogue Cluster 11: (Ep C11): full final MMA1 + always-rescale + sub_row + full exp2 + l_row update + cast P + scale_output_tile + barrier.
; HPP (lines 712-731):
;   v_o = mma1(v_p, v_v, v_o);
;   row_max = max(m_row, attn_row_max<T>(v_s[1]));
;   rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
;   m_row = row_max;
;   attn_sub_row<T>(v_s[1], row_max);
;   asm volatile("" : "+v"(v_s[1]) ::);
;   attn_exp2_slice<T, 0, s_half_len>(v_s[1]);
;   sched_barrier_pairs<10, 5, 10>();
;   sched_barrier_exp_pairs<6, 3, 10>();
;   __builtin_amdgcn_sched_barrier(0);
;
;   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1]);
;   l_row *= rescale_m;
;   l_row += attn_sum<T>(v_s[1]);
;   v_p = opus::cast<D_ATTN>(v_s[1]);
;   asm volatile("" : "+v"(v_p) ::);
;   __builtin_amdgcn_sched_barrier(0);
;   scale_output_tile<T>(v_o, rescale_m);
;   asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]), "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - No `s_setprio` here — the kernel's final cluster does not need the
;     priority dance because both wave-groups have entered their epilogues
;     and won't compete for the MFMA unit any longer.
;   - 16 MFMA form `v_o = mma1(v_p, v_v, v_o)` (4 step_k * 4 wave-tiles).
;   - 16 v_max3 reduce v_s[1] (v[0:15] + v[112:127]); `v_max3_f32 v96,
;     v200, v24, v25` folds running m_row (v200) so
;     `row_max = max(m_row, attn_row_max(v_s[1]))`.
;   - `v_sub_f32_e32 v97, v200, v96` = `m_row - row_max`; `v_exp_f32_e32
;     v112, v97` = rescale_m in v112.
;   - 32 v_sub_f32 implement `attn_sub_row(v_s[1], row_max)`; the
;     `;;#ASMSTART/ASMEND` is the v_s[1] anchor.
;   - 32 v_exp_f32 implement the full `attn_exp2_slice<0, s_full>(v_s[1])`
;     (LLVM fused the two-half exp2 into a single 32-wide chain because
;     no inter-half barrier separates them in the IR).
;   - 32 v_add_f32 + permlane_swap implement `attn_sum(v_s[1])` into v113.
;     Note `l_row *= rescale_m` + `l_row += tile_sum` is later folded into
;     `v_add_f32_e32 v214, v214, v114` and `v_mul_f32_e32 v214, v112, v214`
;     by LLVM in Ep C13's tail.
;   - 16 v_cvt_pk_bf16_f32 cast v_s[1] to bf16 for v_p (v[96:111]).
;   - 32 `v_pk_mul_f32 v[X:X+1], v[112:113], v[X:X+1] op_sel_hi:[0,1]` scale
;     v_o by rescale_m (v112). Note destination v[14:15], v[12:13], ...,
;     v[0:1], v[30:31], ..., v[16:17] are overwriting non-v_o registers —
;     LLVM uses temporary destinations for the scale step then folds back
;     into v_o via implicit operand renaming (these scaled values then feed
;     Ep C13's final MMA1).
;   - `s_barrier` synchronizes wave-groups before Ep C12's V tr_load.
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_mov_b32 s2, 0xf149f2ca
	v_mfma_f32_32x32x16_bf16 v[80:95], v[172:175], v[96:99], v[80:95]
	v_max3_f32 v172, v0, s2, v1
	v_max3_f32 v172, v172, v2, v3
	v_max3_f32 v172, v172, v4, v5
	v_max3_f32 v172, v172, v6, v7
	v_max3_f32 v172, v172, v8, v9
	v_mfma_f32_32x32x16_bf16 v[64:79], v[24:27], v[96:99], v[64:79]
	v_max3_f32 v24, v172, v10, v11
	v_max3_f32 v24, v24, v12, v13
	v_max3_f32 v24, v24, v14, v15
	v_max3_f32 v24, v24, v112, v113
	v_max3_f32 v24, v24, v114, v115
	v_mfma_f32_32x32x16_bf16 v[48:63], v[28:31], v[96:99], v[48:63]
	v_max3_f32 v24, v24, v116, v117
	v_max3_f32 v24, v24, v118, v119
	v_max3_f32 v24, v24, v120, v121
	v_max3_f32 v24, v24, v122, v123
	v_max3_f32 v24, v24, v124, v125
	v_mfma_f32_32x32x16_bf16 v[32:47], v[168:171], v[96:99], v[32:47]
	v_max3_f32 v24, v24, v126, v127
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1
	v_max3_f32 v96, v200, v24, v25
	v_sub_f32_e32 v97, v200, v96
	v_sub_f32_e32 v1, v1, v96
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[100:103], v[80:95]
	v_sub_f32_e32 v31, v127, v96
	v_sub_f32_e32 v30, v126, v96
	v_sub_f32_e32 v29, v125, v96
	v_sub_f32_e32 v28, v124, v96
	v_sub_f32_e32 v27, v123, v96
	v_sub_f32_e32 v0, v0, v96
	v_mfma_f32_32x32x16_bf16 v[64:79], v[20:23], v[100:103], v[64:79]
	v_sub_f32_e32 v26, v122, v96
	v_sub_f32_e32 v25, v121, v96
	v_sub_f32_e32 v24, v120, v96
	v_sub_f32_e32 v23, v119, v96
	v_sub_f32_e32 v22, v118, v96
	v_mfma_f32_32x32x16_bf16 v[48:63], v[160:163], v[100:103], v[48:63]
	v_sub_f32_e32 v21, v117, v96
	v_sub_f32_e32 v20, v116, v96
	v_sub_f32_e32 v19, v115, v96
	v_sub_f32_e32 v18, v114, v96
	v_sub_f32_e32 v17, v113, v96
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[100:103], v[32:47]
	v_sub_f32_e32 v16, v112, v96
	v_sub_f32_e32 v15, v15, v96
	v_sub_f32_e32 v14, v14, v96
	v_sub_f32_e32 v13, v13, v96
	v_sub_f32_e32 v12, v12, v96
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[104:107], v[80:95]
	v_sub_f32_e32 v11, v11, v96
	v_sub_f32_e32 v10, v10, v96
	v_sub_f32_e32 v9, v9, v96
	v_sub_f32_e32 v8, v8, v96
	v_sub_f32_e32 v7, v7, v96
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[104:107], v[64:79]
	v_sub_f32_e32 v6, v6, v96
	v_sub_f32_e32 v5, v5, v96
	v_sub_f32_e32 v4, v4, v96
	v_sub_f32_e32 v3, v3, v96
	v_sub_f32_e32 v2, v2, v96
	;;#ASMSTART
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[104:107], v[48:63]
	v_exp_f32_e32 v112, v97
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_mfma_f32_32x32x16_bf16 v[32:47], v[144:147], v[104:107], v[32:47]
	v_exp_f32_e32 v2, v2
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[108:111], v[80:95]
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_mfma_f32_32x32x16_bf16 v[64:79], v[132:135], v[108:111], v[64:79]
	v_exp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	v_mfma_f32_32x32x16_bf16 v[48:63], v[136:139], v[108:111], v[48:63]
	v_exp_f32_e32 v11, v11
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_mfma_f32_32x32x16_bf16 v[32:47], v[140:143], v[108:111], v[32:47]
	v_exp_f32_e32 v14, v14
	v_exp_f32_e32 v15, v15
	v_add_f32_e32 v96, v1, v0
	v_add_f32_e32 v96, v96, v2
	v_add_f32_e32 v96, v96, v3
	v_add_f32_e32 v96, v96, v4
	v_add_f32_e32 v96, v96, v5
	v_add_f32_e32 v96, v96, v6
	v_add_f32_e32 v96, v96, v7
	v_add_f32_e32 v96, v96, v8
	v_add_f32_e32 v96, v96, v9
	v_add_f32_e32 v96, v96, v10
	v_add_f32_e32 v96, v96, v11
	v_exp_f32_e32 v16, v16
	v_add_f32_e32 v96, v96, v12
	v_exp_f32_e32 v17, v17
	v_add_f32_e32 v96, v96, v13
	v_exp_f32_e32 v18, v18
	v_add_f32_e32 v96, v96, v14
	v_exp_f32_e32 v19, v19
	v_add_f32_e32 v96, v96, v15
	v_exp_f32_e32 v20, v20
	v_add_f32_e32 v96, v96, v16
	v_exp_f32_e32 v21, v21
	v_add_f32_e32 v96, v96, v17
	v_exp_f32_e32 v22, v22
	v_add_f32_e32 v96, v96, v18
	v_exp_f32_e32 v23, v23
	v_add_f32_e32 v96, v96, v19
	v_exp_f32_e32 v24, v24
	v_add_f32_e32 v96, v96, v20
	v_exp_f32_e32 v25, v25
	v_add_f32_e32 v96, v96, v21
	v_exp_f32_e32 v26, v26
	v_add_f32_e32 v96, v96, v22
	v_exp_f32_e32 v27, v27
	v_add_f32_e32 v96, v96, v23
	v_exp_f32_e32 v28, v28
	v_add_f32_e32 v96, v96, v24
	v_exp_f32_e32 v29, v29
	v_add_f32_e32 v96, v96, v25
	v_exp_f32_e32 v30, v30
	v_add_f32_e32 v96, v96, v26
	v_exp_f32_e32 v31, v31
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	v_add_f32_e32 v113, v96, v31
	v_mov_b32_e32 v114, v113
	s_nop 1
	v_permlane32_swap_b32_e64 v113, v114 bound_ctrl:1
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v14, v15
	v_cvt_pk_bf16_f32 v102, v12, v13
	v_cvt_pk_bf16_f32 v101, v10, v11
	v_cvt_pk_bf16_f32 v100, v8, v9
	v_cvt_pk_bf16_f32 v99, v6, v7
	v_cvt_pk_bf16_f32 v98, v4, v5
	v_cvt_pk_bf16_f32 v97, v2, v3
	v_cvt_pk_bf16_f32 v96, v0, v1
	;;#ASMSTART
	;;#ASMEND
	v_pk_mul_f32 v[14:15], v[112:113], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[112:113], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[112:113], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[112:113], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[112:113], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[112:113], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[112:113], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[112:113], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[112:113], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[112:113], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[112:113], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[112:113], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[112:113], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[112:113], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[112:113], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[112:113], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[112:113], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[112:113], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[112:113], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[112:113], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[112:113], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[112:113], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[112:113], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[112:113], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[112:113], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[112:113], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[112:113], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[112:113], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[112:113], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[112:113], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[112:113], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[112:113], v[32:33] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_barrier

; Epilogue Cluster 12: (Ep C12): final V tr_load + barrier.
; HPP (lines 735-739):
;   v_v = tr_load<T::VEC_TR_V>(s_v[1], u_rv);
;   s_waitcnt_lgkmcnt(0_I);
;   __builtin_amdgcn_sched_barrier(0);
;   __builtin_amdgcn_s_barrier();
;   __builtin_amdgcn_sched_barrier(0);
; NOTE:
;   - The final V load reads from s_v[1] (V[max_num_tiles-1], prefetched in
;     Ep C8). v216 was rebuilt in Ep C5/C6 to point at the s_v[1] base
;     (offset 0xc600 in the smem buffer).
;   - 32 `ds_read_b64_tr_b16 v[X:X+1], v216 offset:Y` form `v_v = tr_load
;     <VEC_TR_V>(s_v[1], u_rv)`. Destinations land in v[64:147] (the
;     interim VGPRs that LLVM allocated since v_p occupies v[96:111] and
;     v_o occupies the four scaled banks).
;   - `s_waitcnt lgkmcnt(0)` + `s_barrier` close Ep C12.
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:65], v216 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[66:67], v216 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[68:69], v216 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[70:71], v216 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[72:73], v216 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[74:75], v216 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[76:77], v216 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[78:79], v216 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[80:81], v216 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[82:83], v216 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[84:85], v216 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[86:87], v216 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[88:89], v216 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[90:91], v216 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[92:93], v216 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v216 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v216 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v216 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v216 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v216 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v216 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v216 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v216 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v216 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v216 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v216 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v216 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v216 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v216 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v216 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v216 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v216 offset:0x25c0

	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	s_barrier

; Epilogue Cluster 13: (Ep C13): final MMA1 + (!stagger) barrier.
; HPP (lines 742-750):
;   v_o = mma1(v_p, v_v, v_o);
;   ...
;   if (!stagger) {
;       __builtin_amdgcn_s_barrier();
;   }
; NOTE:
;   - 16 `v_mfma_f32_32x32x16_bf16 v[0:15]/v[16:31]/v[48:63]/v[32:47],
;     v[X:X+3], v[Y:Y+3], v[ACC:ACC+15]` form the final
;     `v_o = mma1(v_p, v_v, v_o)`. The 4 wave-tile output banks land in
;     v[0:15], v[16:31], v[48:63], v[32:47] (different layout from earlier
;     since v_o has been swapped through multiple temporary banks during
;     scale_output_tile).
;   - `s_andn2_b64 vcc, exec, s[0:1]` reconstructs vcc from
;     `!stagger` (s[0:1] = warp_id/4 bit lifted earlier in the kernel).
;   - `s_cbranch_vccnz .LBB0_27` skips the conditional `s_barrier` when
;     `stagger == 1` (warps 4-7); otherwise falls through to `s_barrier`
;     and then to .LBB0_27. The stagger barrier ensures warps 0-3 and 4-7
;     leave Ep C13 in lockstep for the final store.
;   - .LBB0_27 is the rejoin where normalize + store starts.
	v_mfma_f32_32x32x16_bf16 v[0:15], v[64:67], v[96:99], v[0:15]
	s_andn2_b64 vcc, exec, s[0:1]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[80:83], v[96:99], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[116:119], v[96:99], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[132:135], v[96:99], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[68:71], v[100:103], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[84:87], v[100:103], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[120:123], v[100:103], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[136:139], v[100:103], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[72:75], v[104:107], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[88:91], v[104:107], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[124:127], v[104:107], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[140:143], v[104:107], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[76:79], v[108:111], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[92:95], v[108:111], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[128:131], v[108:111], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[144:147], v[108:111], v[32:47]
	s_cbranch_vccnz .LBB0_27
	s_barrier
.LBB0_27:
; Normalize O by 1/l_row + cast to bf16 + store to gmem.
; HPP (lines 745-754):
;   D_ACC l_inv = (l_row > D_ACC(0.0f)) ? (D_ACC(1.0f) / l_row) : D_ACC(0.0f);
;   static_for<o_len>([&](auto i) { v_o[i.value] *= l_inv; });
;   if (!stagger) {
;       __builtin_amdgcn_s_barrier();
;   }
;   auto u_o = make_layout_o<T>(warp_id, lane_id, kargs.stride_q_n);
;   auto v_o_bf16 = opus::cast<D_ATTN>(v_o);
;   store<T::VEC_O>(g_o, v_o_bf16, u_o);
; NOTE:
;   - The 7-instruction chain `v_add_f32 + v_fmac_f32 + ...` realises the
;     deferred l_row updates that the epilogue clusters left pending. Each
;     Cluster pair (Ep C1/C3, C5/C7, C9/C11) contributed one tile_sum and one
;     rescale_m that must be folded together:
;       l_row = (((l_row + ts_ep1 + ts_ep3) * rs_ep3 + ts_ep5) * rs_ep7 + ts_ep9) * rs_ep11 + ts_ep11
;     The `v_fmac_f32_e32 v217/v177/v114, vRS, vSUM` instructions are the
;     three `* rescale + tile_sum` FMA folds; the `v_add_f32_e32 v64, ...`
;     instructions are the in-between additions.
;   - `v_rcp_f32_e32 v65, v64` computes `1.0f / l_row`.
;   - `v_cmp_lt_f32_e32 vcc, 0, v64` tests `0 < l_row` (= l_row > 0).
;   - `v_cndmask_b32_e32 v64, 0, v65, vcc` selects between 0.0 and the
;     reciprocal: `l_inv = (l_row > 0) ? (1.0f / l_row) : 0.0f`.
;   - `s_add_u32 s0, s14, s22` + `s_addc_u32 s1, s15, s23` builds
;     `ptr_o_base = ptr_o + qo_gmem_offset*2`. s[14:15] = ptr_o from the
;     dwordx8 kargs load, s[22:23] = qo_gmem_offset*2 (preserved from setup).
;   - 32 `v_pk_mul_f32 v[X:X+1], v[X:X+1], v[64:65] op_sel_hi:[1,0]` scale
;     all 64 fp32 elements of v_o by l_inv (v64 holds the scalar l_inv;
;     op_sel_hi:[1,0] broadcasts it to both pk lanes). Destinations are the
;     four scaled v_o banks v[0:31] (mma_acc#0+1) and v[32:63] (mma_acc#2+3).
;   - `v_lshl_add_u32 v64, v209, 2, v210` + `v_add_lshl_u32 v64, v64, s12, 1`
;     forms the per-lane O gmem byte address using the `make_layout_o`
;     descriptor (mirror of the Q load layout from the prologue).
;   - 16 `v_cvt_pk_bf16_f32 vX, vL, vH` cast pairs of fp32 to bf16 packs.
;   - 16 `buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:Y` store
;     8 bf16 (16 bytes) per lane per instruction → 16 * 16 = 256 bytes/lane,
;     totalling 64 bf16/lane covering the lane's slice of the 32-row, 128-col
;     output tile. Offsets 0/16/32/.../240 cover the 16 micro-store packs.
;   - `s_endpgm` terminates the kernel.
	v_add_f32_e32 v64, v214, v220
	v_add_f32_e32 v64, v64, v218
	v_fmac_f32_e32 v217, v208, v64
	v_add_f32_e32 v64, v217, v215
	v_fmac_f32_e32 v177, v180, v64
	v_add_f32_e32 v64, v177, v176
	v_fmac_f32_e32 v114, v112, v64
	v_add_f32_e32 v64, v114, v113
	v_rcp_f32_e32 v65, v64
	v_cmp_lt_f32_e32 vcc, 0, v64
	s_add_u32 s0, s14, s22
	s_addc_u32 s1, s15, s23
	v_cndmask_b32_e32 v64, 0, v65, vcc
	v_pk_mul_f32 v[2:3], v[2:3], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[64:65] op_sel_hi:[1,0]
	v_lshl_add_u32 v64, v209, 2, v210
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	v_cvt_pk_bf16_f32 v3, v2, v3
	v_cvt_pk_bf16_f32 v2, v0, v1
	v_add_lshl_u32 v64, v64, s12, 1
	v_cvt_pk_bf16_f32 v1, v6, v7
	v_cvt_pk_bf16_f32 v0, v4, v5
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:16
	v_cvt_pk_bf16_f32 v1, v10, v11
	v_cvt_pk_bf16_f32 v0, v8, v9
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:32
	v_cvt_pk_bf16_f32 v1, v14, v15
	v_cvt_pk_bf16_f32 v0, v12, v13
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:48
	v_cvt_pk_bf16_f32 v1, v18, v19
	v_cvt_pk_bf16_f32 v0, v16, v17
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v22, v23
	v_cvt_pk_bf16_f32 v0, v20, v21
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:80
	v_cvt_pk_bf16_f32 v1, v26, v27
	v_cvt_pk_bf16_f32 v0, v24, v25
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:96
	v_cvt_pk_bf16_f32 v1, v30, v31
	v_cvt_pk_bf16_f32 v0, v28, v29
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:112
	v_cvt_pk_bf16_f32 v1, v50, v51
	v_cvt_pk_bf16_f32 v0, v48, v49
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v54, v55
	v_cvt_pk_bf16_f32 v0, v52, v53
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:144
	v_cvt_pk_bf16_f32 v1, v58, v59
	v_cvt_pk_bf16_f32 v0, v56, v57
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:160
	v_cvt_pk_bf16_f32 v1, v62, v63
	v_cvt_pk_bf16_f32 v0, v60, v61
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:176
	v_cvt_pk_bf16_f32 v1, v34, v35
	v_cvt_pk_bf16_f32 v0, v32, v33
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v38, v39
	v_cvt_pk_bf16_f32 v0, v36, v37
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:208
	v_cvt_pk_bf16_f32 v1, v42, v43
	v_cvt_pk_bf16_f32 v0, v40, v41
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:224
	v_cvt_pk_bf16_f32 v1, v46, v47
	v_cvt_pk_bf16_f32 v0, v44, v45
	buffer_store_dwordx2 v[2:3], v64, s[0:3], 0 offen
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen offset:240
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs
		.amdhsa_group_segment_fixed_size 68096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 237
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 240
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs,"axG",@progbits,_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs,comdat
.Lfunc_end0:
	.size	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs, .Lfunc_end0-_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.num_vgpr, 237
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.num_agpr, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.numbered_sgpr, 44
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.private_seg_size, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.uses_vcc, 1
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.uses_flat_scratch, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.has_dyn_sized_stack, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.has_recursion, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_cf23ca35d902b312,@object
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_cf23ca35d902b312
__hip_cuid_cf23ca35d902b312:
	.byte	0
	.size	__hip_cuid_cf23ca35d902b312, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_cf23ca35d902b312
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           80
        .value_kind:     by_value
    .group_segment_fixed_size: 68096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb1EEEv14opus_gqa_kargs.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     237
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
