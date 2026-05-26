	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	flash_attn_opus_kernel_0
	.p2align	8
	.type	flash_attn_opus_kernel_0,@function
flash_attn_opus_kernel_0:
; OPUS ISA ANNOTATION
; Source map: FlyDSL/kernels/flash_attn_opus.py, selected kernel body around Python lines 1241-1987.
; Notation:
;   ; PY   = nearby source-level FlyDSL/Python operation lowered into this ISA region.
;   ; NOTE = lowering/scheduling note; LLVM may reorder, combine, or split Python operations.
; Only comments have been added to this dump. Executable ISA instructions are intentionally unchanged.
;
; OPUS high-level flow:
;   1. Kernel prologue derives q/head/batch coordinates, buffer resource offsets, lane/warp ids, and LDS addresses.
;   2. Prologue prefetches the first K/V tiles into LDS, loads Q, scales Q by 1/sqrt(d), and computes the first QK scores.
;   3. The steady-state loop pipelines K/V global loads, LDS reads, QK MMA, causal masking, softmax exp/sum,
;      P*V MMA, lazy rescale of O, and loop-carried online-softmax state.
;   4. The epilogue drains the final tiles, normalizes O by 1/l_row, converts fp32 accumulators to bf16, and stores.
;
; LLVM SCHED HINT LEGEND
;   Source `rocdl.sched.group.barrier(mask, size, groupId)` lowers to the
;   `llvm.amdgcn.sched.group.barrier` intrinsic and then to the AMDGPU
;   `SCHED_GROUP_BARRIER` pseudo. The pseudo is consumed by IGroupLP scheduling
;   and is not emitted as real ISA.
;   Masks used by this kernel:
;     0x008 = MFMA/WMMA, 0x002 = VALU, 0x400 = TRANS/EXP (`v_exp_f32`).
;   Directionality:
;     `SCHED_GROUP_BARRIER` scans upward from the marker to earlier SUnits
;     (`initSchedGroupBarrierPipelineStage` calls `findCandidateSUnits(RIter,
;     SUnits.rend(), ...)`). It groups already-emitted candidate instructions,
;     not future instructions below the marker. Same `groupId` barriers are
;     solved together as one pipeline by `PipelineSolver`.
;   Source `rocdl.sched_barrier(0)` lowers to `SCHED_BARRIER 0`. In this LLVM
;   tree, `SIInstrInfo::isSchedulingBoundary()` returns true for
;   `SCHED_BARRIER` with immediate 0, so no real instruction, including
;   `S_BARRIER`, may be moved across that scheduling boundary. The pseudo is
;   removed before final ISA emission; only the resulting ordering remains.
	s_mov_b32 s6, s3
	s_load_dwordx4 s[16:19], s[0:1], 0x28
	s_load_dwordx8 s[8:15], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s23, s16, 31
	s_ashr_i32 s5, s17, 31
	s_ashr_i32 s1, s18, 31
	s_mov_b32 s0, s18
	s_lshl_b64 s[24:25], s[0:1], 1
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s7, s6, 31
	v_lshrrev_b32_e32 v1, 6, v0
	v_readfirstlane_b32 s0, v0
	s_lshr_b32 s35, s0, 6
	s_lshr_b32 s25, s0, 8
	s_lshl_b64 s[26:27], s[6:7], 8
	s_and_b32 s0, s2, 63
	s_lshr_b64 s[2:3], s[2:3], 6
	s_add_u32 s2, s0, s2
	s_addc_u32 s3, 0, s3
	s_mul_hi_i32 s6, s16, s4
	s_mul_i32 s4, s16, s4
	s_add_u32 s7, s4, s26
	s_addc_u32 s20, s6, s27
	s_mul_i32 s5, s7, s5
	s_mul_hi_u32 s21, s7, s17
	s_add_i32 s5, s21, s5
	s_mul_i32 s20, s20, s17
	s_add_i32 s5, s5, s20
	s_mul_i32 s7, s7, s17
	s_lshl_b64 s[2:3], s[2:3], 7
	s_add_u32 s20, s7, s2
	s_addc_u32 s21, s5, s3
	s_lshl_b64 s[2:3], s[20:21], 1
	s_lshl_b32 s5, s0, 8
	s_mul_i32 s0, s4, s1
	s_mul_hi_u32 s1, s4, s18
	s_add_i32 s0, s1, s0
	s_mul_i32 s6, s6, s18
	s_add_i32 s1, s0, s6
	s_mul_i32 s0, s4, s18
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s0, s5
	s_addc_u32 s1, s1, 0
	s_add_u32 s8, s8, s2
	s_addc_u32 s2, s9, s3
	s_and_b32 s9, s2, 0xffff
	s_add_u32 s4, s10, s0
	s_addc_u32 s2, s11, s1
	s_and_b32 s5, s2, 0xffff
	s_add_u32 s0, s12, s0
	s_addc_u32 s1, s13, s1
	v_and_b32_e32 v2, 7, v0
	s_mul_i32 s31, s35, 0x410
	v_and_or_b32 v3, v0, 56, v1
	v_mul_lo_u32 v3, s24, v3
	s_add_i32 s33, s31, 0x2080
	v_lshl_add_u32 v215, v2, 4, v3
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, -1
	s_mov_b32 s6, s10
	s_mov_b32 s7, s11
	s_mov_b32 m0, s31
	s_nop 0
; Prologue: prime LDS with the first K tile, then synchronize before Q load.
;   _async_load_k(fx.Index(0), 0)
;   rocdl.s_waitcnt(0)
;   rocdl.sched_barrier(0)
;   rocdl.s_barrier()
;   buffer_load_dwordx4 ... lds writes directly into LDS through m0-selected offsets.
	buffer_load_dwordx4 v215, s[4:7], 0 offen lds
	v_add_u32_e32 v216, 0x80, v215
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v216, s[4:7], 0 offen lds

; Prologue: Q row and Q global address setup.
;   q_row_in_block = wave_q_offset + lane_mod_32
;   q_start_pos_i32 = fx.Int32(q_start + wave_id_uni * fx.Index(ROWS_PER_WAVE))
;   q_row = q_start + q_row_in_block
;   q_row_i32 = fx.Int32(q_row)
; NOTE:
;   The preceding buffer_loads still belong to the initial K LDS prime. The
;   address math below prepares the Q row used by `_load_q_all`.
;   - `v_lshlrev_b32 v1, 5, v1` builds the wave-row component
;     (`wave_id_uni * ROWS_PER_WAVE`, folded with q_start in the lowered value).
;   - `v_and_or_b32 v58, v0, 31, v1` combines `lane_mod_32` with that row
;     component, producing the per-lane Q row index used as `q_row`.
;   - `v_mad_i64_i32 v[210:211], ... v58 ...` forms the 64-bit global Q row
;     base address.
;   - `v_add_lshl_u32 v1, v210, v11, 1` adds the per-lane Q column offset for
;     the first vectorized Q load.
	s_mov_b32 s22, s16
	v_bfe_u32 v214, v0, 5, 1
	s_and_b32 s1, s1, 0xffff
	s_add_i32 s30, s31, 0x8500
	s_lshl_b32 s34, s18, 7
	s_add_i32 s29, s31, 0xa580
	s_mul_i32 s21, s35, 0x440
	s_add_i32 s28, s21, 0x4100
	s_addk_i32 s21, 0x6300
	s_cmp_eq_u32 s25, 0
	v_lshlrev_b32_e32 v1, 5, v1
	s_mov_b32 s2, s10
	s_mov_b32 s3, s11
	s_mov_b32 s36, s0
	s_mov_b32 s37, s1
	s_mov_b32 s38, s10
	s_mov_b32 s39, s11
	v_mul_u32_u24_e32 v2, 0x208, v2
	v_lshlrev_b32_e32 v3, 3, v0
	v_and_b32_e32 v3, 0xc0, v3
	v_add_u32_e32 v10, v2, v3
	v_lshlrev_b32_e32 v11, 3, v214
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 m0, s30
	s_barrier
	v_and_or_b32 v58, v0, 31, v1
	v_mad_i64_i32 v[210:211], s[12:13], v58, s17, 0
	v_add_lshl_u32 v1, v210, v11, 1

; Prologue: load the 128-wide Q row tile from global memory.
;   q_all_bf16 = _load_q_all(q_row_in_block)
; NOTE:
;   The eight buffer_load_dwordx4 instructions are the `range_constexpr(K_STEPS_QK)`
;   body inside `_load_q_all`: each load reads one v4i32 pack, bitcast as bf16,
;   and the helper concatenates the 8 packs into the 64-element Q vector.
;   `_scale_q_all` starts later at .LBB0_2 and is also interleaved with the
;   first QK MFMA chain.
	buffer_load_dwordx4 v[6:9], v1, s[8:11], 0 offen
	buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen offset:32
	buffer_load_dwordx4 v[54:57], v1, s[8:11], 0 offen offset:64
	buffer_load_dwordx4 v[50:53], v1, s[8:11], 0 offen offset:96
	buffer_load_dwordx4 v[46:49], v1, s[8:11], 0 offen offset:128
	buffer_load_dwordx4 v[42:45], v1, s[8:11], 0 offen offset:160
	buffer_load_dwordx4 v[38:41], v1, s[8:11], 0 offen offset:192
	buffer_load_dwordx4 v[34:37], v1, s[8:11], 0 offen offset:224

; Prologue: prefetch initial K/V tiles to LDS.
;   _async_load_k(fx.Index(BLOCK_N), 1)
;   _async_load_v(fx.Index(0), 0)
; NOTE:
;   buffer_load_dwordx4 ... lds performs the async global-to-LDS copies for
;   the next K tile in s_k[1] and the current V tile in s_v[0].
	s_nop 0
	buffer_load_dwordx4 v215, s[4:7], s34 offen lds
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v216, s[4:7], s34 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v215, s[36:39], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v216, s[36:39], 0 offen lds

; Prologue: read K tile from LDS into VGPRs.
;   v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
; NOTE:
;   The ds_read_b128 sequence materializes the K fragments used by the first
;   QK MFMA chain below.
	v_add_lshl_u32 v211, v10, v11, 1
	ds_read_b128 v[10:13], v211
	ds_read_b128 v[138:141], v211 offset:32
	ds_read_b128 v[134:137], v211 offset:512
	ds_read_b128 v[126:129], v211 offset:544
	ds_read_b128 v[122:125], v211 offset:64
	ds_read_b128 v[114:117], v211 offset:96
	ds_read_b128 v[118:121], v211 offset:576
	ds_read_b128 v[110:113], v211 offset:608
	ds_read_b128 v[106:109], v211 offset:8320
	ds_read_b128 v[98:101], v211 offset:8352
	ds_read_b128 v[102:105], v211 offset:8832
	ds_read_b128 v[94:97], v211 offset:8864
	ds_read_b128 v[90:93], v211 offset:8384
	ds_read_b128 v[82:85], v211 offset:8416
	ds_read_b128 v[86:89], v211 offset:8896
	ds_read_b128 v[78:81], v211 offset:8928

; Prologue: scheduling fence and waits before first QK use.
; rocdl.sched_barrier(0)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_V)
	s_waitcnt vmcnt(2) lgkmcnt(0)
; Prologue: optional stagger barrier for wave-group phase shift.
;   if const_expr(OPUS_ENABLE_STAGGER):
;       _stagger_extra_barrier_if_one()
	s_cbranch_scc1 .LBB0_2
; The body runs on warps 4-7 only, advancing their s_barrier ordinal
; by one relative to warps 0-3 → starts the dual-group phase shift.
; if_op = scf.IfOp(stagger_is_one_i1, [], has_else=False, loc=ir.Location.unknown())
; with ir.InsertionPoint(if_op.regions[0].blocks[0]):
; 	rocdl.sched_barrier(0)
; 	rocdl.s_barrier()
; 	scf.YieldOp([])
	s_barrier

.LBB0_2:
; Prologue: scale first Q packs for MFMA operands.
;   q_all_scaled_bf16 = _scale_q_all(q_all_bf16)
; NOTE:
;   Computes `rsqrt(head_dim_runtime) * log2(e)`, expands packed bf16 Q lanes
;   to f32, multiplies by the scale, and packs the first scaled Q fragments
;   back to bf16 registers v130-v133.
	v_cvt_f32_i32_e32 v1, s19
	v_and_b32_e32 v61, 0xffff0000, v37
	v_and_b32_e32 v63, 0xffff0000, v36
	v_and_b32_e32 v67, 0xffff0000, v41
	v_rsq_f32_e32 v1, v1
	v_and_b32_e32 v69, 0xffff0000, v40
	v_and_b32_e32 v71, 0xffff0000, v45
	v_and_b32_e32 v73, 0xffff0000, v44
	v_mul_f32_e32 v64, 0x3fb8aa3b, v1
	v_and_b32_e32 v75, 0xffff0000, v49
	v_and_b32_e32 v77, 0xffff0000, v48
	v_and_b32_e32 v143, 0xffff0000, v53
	v_and_b32_e32 v145, 0xffff0000, v52
	v_and_b32_e32 v147, 0xffff0000, v57
	v_and_b32_e32 v149, 0xffff0000, v56
	v_and_b32_e32 v151, 0xffff0000, v5
	v_and_b32_e32 v153, 0xffff0000, v4
	v_and_b32_e32 v15, 0xffff0000, v9
	v_and_b32_e32 v17, 0xffff0000, v8
	v_and_b32_e32 v19, 0xffff0000, v7
	v_and_b32_e32 v21, 0xffff0000, v6
	v_lshlrev_b32_e32 v14, 16, v9
	v_lshlrev_b32_e32 v16, 16, v8
	v_lshlrev_b32_e32 v18, 16, v7
	v_lshlrev_b32_e32 v20, 16, v6
	v_pk_mul_f32 v[6:7], v[64:65], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[64:65], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[64:65], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[64:65], v[14:15] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v133, v14, v15
	v_cvt_pk_bf16_f32 v132, v16, v17
	v_cvt_pk_bf16_f32 v131, v8, v9
	v_cvt_pk_bf16_f32 v130, v6, v7
	s_nop 1

; Prologue: first QK score tile plus interleaved Q scaling.
;   v_s_0 = _mma0(v_k)
;   rocdl.sched_barrier(0)
;   q_all_scaled_bf16 = _scale_q_all(q_all_bf16)
; NOTE:
;   v_mfma_f32_32x32x16_bf16 computes Q*K^T into fp32 score accumulators.
;   Interleaved VALU (`v_and`, `v_lshl`, `v_mul`, `v_pk_mul`,
;   `v_cvt_pk_bf16_f32`) is the continuation of `_scale_q_all`, preparing the
;   next scaled bf16 Q packs for later MFMA operands while the current MFMA
;   chain advances.
	v_mfma_f32_32x32x16_bf16 v[18:33], v[10:13], v[130:133], 0
	v_and_b32_e32 v155, 0xffff0000, v3
	v_and_b32_e32 v7, 0xffff0000, v2
	v_lshlrev_b32_e32 v150, 16, v5
	v_lshlrev_b32_e32 v152, 16, v4
	v_lshlrev_b32_e32 v154, 16, v3
	v_lshlrev_b32_e32 v6, 16, v2
	v_pk_mul_f32 v[156:157], v[64:65], v[6:7] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[134:137], v[130:133], 0
	v_mul_f32_e64 v134, v64, v154
	v_mul_f32_e64 v135, v64, v155
	v_mul_f32_e64 v152, v64, v152
	v_mul_f32_e64 v153, v64, v153
	v_mul_f32_e64 v136, v64, v150
	v_mul_f32_e64 v137, v64, v151
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_cvt_pk_bf16_f32 v136, v152, v153
	v_cvt_pk_bf16_f32 v135, v134, v135
	v_cvt_pk_bf16_f32 v134, v156, v157
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[138:141], v[134:137], v[18:33]
	v_and_b32_e32 v139, 0xffff0000, v55
	v_and_b32_e32 v141, 0xffff0000, v54
	v_lshlrev_b32_e32 v146, 16, v57
	v_lshlrev_b32_e32 v148, 16, v56
	v_lshlrev_b32_e32 v138, 16, v55
	v_lshlrev_b32_e32 v140, 16, v54
	v_pk_mul_f32 v[54:55], v[64:65], v[140:141] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[126:129], v[134:137], v[2:17]
	v_mul_f32_e64 v56, v64, v138
	v_mul_f32_e64 v57, v64, v139
	v_mul_f32_e64 v126, v64, v148
	v_mul_f32_e64 v127, v64, v149
	v_mul_f32_e64 v128, v64, v146
	v_mul_f32_e64 v129, v64, v147
	v_cvt_pk_bf16_f32 v141, v128, v129
	v_cvt_pk_bf16_f32 v140, v126, v127
	v_cvt_pk_bf16_f32 v139, v56, v57
	v_cvt_pk_bf16_f32 v138, v54, v55
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[122:125], v[138:141], v[18:33]
	v_and_b32_e32 v55, 0xffff0000, v51
	v_and_b32_e32 v57, 0xffff0000, v50
	v_lshlrev_b32_e32 v142, 16, v53
	v_lshlrev_b32_e32 v144, 16, v52
	v_lshlrev_b32_e32 v54, 16, v51
	v_lshlrev_b32_e32 v56, 16, v50
	v_pk_mul_f32 v[50:51], v[64:65], v[56:57] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[118:121], v[138:141], v[2:17]
	v_mul_f32_e64 v52, v64, v54
	v_mul_f32_e64 v53, v64, v55
	v_mul_f32_e64 v54, v64, v144
	v_mul_f32_e64 v55, v64, v145
	v_mul_f32_e64 v56, v64, v142
	v_mul_f32_e64 v57, v64, v143
	v_cvt_pk_bf16_f32 v145, v56, v57
	v_cvt_pk_bf16_f32 v144, v54, v55
	v_cvt_pk_bf16_f32 v143, v52, v53
	v_cvt_pk_bf16_f32 v142, v50, v51
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[114:117], v[142:145], v[18:33]
	v_and_b32_e32 v51, 0xffff0000, v47
	v_and_b32_e32 v53, 0xffff0000, v46
	v_lshlrev_b32_e32 v74, 16, v49
	v_lshlrev_b32_e32 v76, 16, v48
	v_lshlrev_b32_e32 v50, 16, v47
	v_lshlrev_b32_e32 v52, 16, v46
	v_pk_mul_f32 v[46:47], v[64:65], v[52:53] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[110:113], v[142:145], v[2:17]
	v_mul_f32_e64 v48, v64, v50
	v_mul_f32_e64 v49, v64, v51
	v_mul_f32_e64 v50, v64, v76
	v_mul_f32_e64 v51, v64, v77
	v_mul_f32_e64 v52, v64, v74
	v_mul_f32_e64 v53, v64, v75
	v_cvt_pk_bf16_f32 v149, v52, v53
	v_cvt_pk_bf16_f32 v148, v50, v51
	v_cvt_pk_bf16_f32 v147, v48, v49
	v_cvt_pk_bf16_f32 v146, v46, v47
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[106:109], v[146:149], v[18:33]
	v_and_b32_e32 v47, 0xffff0000, v43
	v_and_b32_e32 v49, 0xffff0000, v42
	v_lshlrev_b32_e32 v70, 16, v45
	v_lshlrev_b32_e32 v72, 16, v44
	v_lshlrev_b32_e32 v46, 16, v43
	v_lshlrev_b32_e32 v48, 16, v42
	v_pk_mul_f32 v[42:43], v[64:65], v[48:49] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[102:105], v[146:149], v[2:17]
	v_mul_f32_e64 v44, v64, v46
	v_mul_f32_e64 v45, v64, v47
	v_mul_f32_e64 v46, v64, v72
	v_mul_f32_e64 v47, v64, v73
	v_mul_f32_e64 v48, v64, v70
	v_mul_f32_e64 v49, v64, v71
	v_cvt_pk_bf16_f32 v153, v48, v49
	v_cvt_pk_bf16_f32 v152, v46, v47
	v_cvt_pk_bf16_f32 v151, v44, v45
	v_cvt_pk_bf16_f32 v150, v42, v43
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[98:101], v[150:153], v[18:33]
	v_and_b32_e32 v43, 0xffff0000, v39
	v_and_b32_e32 v45, 0xffff0000, v38
	v_lshlrev_b32_e32 v66, 16, v41
	v_lshlrev_b32_e32 v68, 16, v40
	v_lshlrev_b32_e32 v42, 16, v39
	v_lshlrev_b32_e32 v44, 16, v38
	v_pk_mul_f32 v[38:39], v[64:65], v[44:45] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[94:97], v[150:153], v[2:17]
	v_mul_f32_e64 v40, v64, v42
	v_mul_f32_e64 v41, v64, v43
	v_mul_f32_e64 v42, v64, v68
	v_mul_f32_e64 v43, v64, v69
	v_mul_f32_e64 v44, v64, v66
	v_mul_f32_e64 v45, v64, v67
	v_cvt_pk_bf16_f32 v157, v44, v45
	v_cvt_pk_bf16_f32 v156, v42, v43
	v_cvt_pk_bf16_f32 v155, v40, v41
	v_cvt_pk_bf16_f32 v154, v38, v39
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[90:93], v[154:157], v[18:33]
	v_and_b32_e32 v39, 0xffff0000, v35
	v_and_b32_e32 v41, 0xffff0000, v34
	v_lshlrev_b32_e32 v60, 16, v37
	v_lshlrev_b32_e32 v62, 16, v36
	v_lshlrev_b32_e32 v38, 16, v35
	v_lshlrev_b32_e32 v40, 16, v34
	v_pk_mul_f32 v[34:35], v[64:65], v[40:41] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[86:89], v[154:157], v[2:17]
	v_mul_f32_e64 v36, v64, v38
	v_mul_f32_e64 v37, v64, v39
	v_mul_f32_e64 v38, v64, v62
	v_mul_f32_e64 v39, v64, v63
	v_mul_f32_e64 v40, v64, v60
	v_mul_f32_e64 v41, v64, v61
	v_cvt_pk_bf16_f32 v161, v40, v41
	v_cvt_pk_bf16_f32 v160, v38, v39
	v_cvt_pk_bf16_f32 v159, v36, v37
	v_cvt_pk_bf16_f32 v158, v34, v35
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[82:85], v[158:161], v[18:33]
	s_lshl_b32 s16, s35, 5
	s_add_i32 s16, s16, s26
	v_mov_b32_e32 v213, s27
	v_or_b32_e32 v212, s26, v58
	v_mfma_f32_32x32x16_bf16 v[2:17], v[78:81], v[158:161], v[2:17]

; Prologue causal mask for the first score tile.
; v_s_0 = _causal_mask_prologue_if_needed(v_s_0)
; NOTE:
;   The v_cmp/v_cndmask pairs replace invalid score lanes with -inf before row max and exp.
;   The branch skips this mask when the whole tile is known to be before the causal boundary.
; s_lo, s_hi = _v_s_vec_to_lists(v_s)
; acc_values = [_raw(v) for v in (s_lo + s_hi)]
; result_types = [v.type for v in acc_values]
; mask_needed = arith.cmpi(
; 	arith.CmpIPredicate.slt,
; 	q_start_pos_i32,
; 	_raw(fx.Int32(kv_end_pos)),
; )
; if_op = scf.IfOp(mask_needed, result_types, has_else=True, loc=ir.Location.unknown())
	s_cmp_gt_i32 s16, 63
	v_lshlrev_b32_e32 v1, 2, v214
	s_cbranch_scc1 .LBB0_4
; with ir.InsertionPoint(if_op.regions[0].blocks[0]):
; 	then_lo = list(s_lo)
; 	then_hi = list(s_hi)
; 	_causal_mask_inplace((then_lo, then_hi), tile_idx)
; 	scf.YieldOp([_raw(v) for v in (then_lo + then_hi)])
	v_sub_u32_e32 v34, v212, v1
	v_subrev_u32_e32 v35, 32, v34
	v_mov_b32_e32 v36, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 0
	v_cmp_lt_i32_e64 s[8:9], v34, 1
	v_cndmask_b32_e64 v18, v18, v36, s[6:7]
	v_cndmask_b32_e64 v19, v19, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 2
	v_cmp_lt_i32_e64 s[8:9], v34, 3
	v_cndmask_b32_e64 v20, v20, v36, s[6:7]
	v_cndmask_b32_e64 v21, v21, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 8
	v_cmp_lt_i32_e64 s[8:9], v34, 9
	v_cndmask_b32_e64 v22, v22, v36, s[6:7]
	v_cndmask_b32_e64 v23, v23, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 10
	v_cmp_lt_i32_e64 s[8:9], v34, 11
	v_cndmask_b32_e64 v24, v24, v36, s[6:7]
	v_cndmask_b32_e64 v25, v25, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 16
	v_cmp_lt_i32_e64 s[8:9], v34, 17
	v_cndmask_b32_e64 v26, v26, v36, s[6:7]
	v_cndmask_b32_e64 v27, v27, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 18
	v_cmp_lt_i32_e64 s[8:9], v34, 19
	v_cndmask_b32_e64 v28, v28, v36, s[6:7]
	v_cndmask_b32_e64 v29, v29, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 24
	v_cmp_lt_i32_e64 s[8:9], v34, 25
	v_cndmask_b32_e64 v30, v30, v36, s[6:7]
	v_cndmask_b32_e64 v31, v31, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v34, 26
	v_cmp_lt_i32_e64 s[8:9], v34, 27
	v_cndmask_b32_e64 v32, v32, v36, s[6:7]
	v_cndmask_b32_e64 v33, v33, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 0
	v_cmp_lt_i32_e64 s[8:9], v35, 1
	v_cndmask_b32_e64 v2, v2, v36, s[6:7]
	v_cndmask_b32_e64 v3, v3, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 2
	v_cmp_lt_i32_e64 s[8:9], v35, 3
	v_cndmask_b32_e64 v4, v4, v36, s[6:7]
	v_cndmask_b32_e64 v5, v5, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 8
	v_cmp_lt_i32_e64 s[8:9], v35, 9
	v_cndmask_b32_e64 v6, v6, v36, s[6:7]
	v_cndmask_b32_e64 v7, v7, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 10
	v_cmp_lt_i32_e64 s[8:9], v35, 11
	v_cndmask_b32_e64 v8, v8, v36, s[6:7]
	v_cndmask_b32_e64 v9, v9, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 16
	v_cmp_lt_i32_e64 s[8:9], v35, 17
	v_cndmask_b32_e64 v10, v10, v36, s[6:7]
	v_cndmask_b32_e64 v11, v11, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 18
	v_cmp_lt_i32_e64 s[8:9], v35, 19
	v_cndmask_b32_e64 v12, v12, v36, s[6:7]
	v_cndmask_b32_e64 v13, v13, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 24
	v_cmp_lt_i32_e64 s[8:9], v35, 25
	v_cndmask_b32_e64 v14, v14, v36, s[6:7]
	v_cndmask_b32_e64 v15, v15, v36, s[8:9]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v35, 26
	v_cmp_lt_i32_e64 s[8:9], v35, 27
	v_cndmask_b32_e64 v16, v16, v36, s[6:7]
	v_cndmask_b32_e64 v17, v17, v36, s[8:9]
	;;#ASMEND

.LBB0_4:
;   m_row_pro = _attn_row_max(v_s_0)
	s_add_u32 s6, s22, 63
	s_addc_u32 s7, s23, 0
	s_lshr_b64 s[6:7], s[6:7], 6
	s_add_u32 s8, s26, 0x100
	s_addc_u32 s9, s27, 0
	s_lshr_b64 s[8:9], s[8:9], 6
	v_mov_b64_e32 v[34:35], s[6:7]
	v_cmp_lt_u64_e32 vcc, s[8:9], v[34:35]
	s_and_b64 s[10:11], vcc, exec
	s_cselect_b32 s9, s9, s7
	s_cselect_b32 s8, s8, s6
	v_mul_u32_u24_e32 v34, 0x880, v214
	v_bfe_u32 v35, v0, 2, 2
	v_mul_u32_u24_e32 v35, 0x220, v35
	v_add_u32_e32 v34, v34, v35
	v_and_b32_e32 v35, 16, v0
	v_max_f32_e32 v36, v18, v19
	v_max3_f32 v36, v36, v20, v21
	v_max3_f32 v36, v36, v22, v23
	v_max3_f32 v36, v36, v24, v25
	v_max3_f32 v36, v36, v26, v27
	v_max3_f32 v36, v36, v28, v29
	v_max3_f32 v36, v36, v30, v31
	v_max3_f32 v36, v36, v32, v33
	v_max3_f32 v36, v36, v2, v3
	v_max3_f32 v36, v36, v4, v5
	v_max3_f32 v36, v36, v6, v7
	v_max3_f32 v36, v36, v8, v9
	v_max3_f32 v36, v36, v10, v11
	v_max3_f32 v36, v36, v12, v13
	v_max3_f32 v36, v36, v14, v15
	v_max3_f32 v36, v36, v16, v17
	v_mov_b32_e32 v37, v36
	s_nop 1
	v_permlane32_swap_b32_e64 v36, v37 bound_ctrl:1
	v_max_f32_e32 v220, v36, v37
;   v_s_0 = _attn_sub_row(v_s_0, m_row_pro)
	v_sub_f32_e32 v33, v33, v220
	v_sub_f32_e32 v32, v32, v220
	v_sub_f32_e32 v31, v31, v220
	v_sub_f32_e32 v30, v30, v220
	v_sub_f32_e32 v29, v29, v220
	v_sub_f32_e32 v28, v28, v220
	v_sub_f32_e32 v27, v27, v220
	v_sub_f32_e32 v26, v26, v220
	v_sub_f32_e32 v25, v25, v220
	v_sub_f32_e32 v24, v24, v220
	v_sub_f32_e32 v23, v23, v220
	v_sub_f32_e32 v22, v22, v220
	v_sub_f32_e32 v21, v21, v220
	v_sub_f32_e32 v20, v20, v220
	v_sub_f32_e32 v19, v19, v220
	v_sub_f32_e32 v18, v18, v220
	v_sub_f32_e32 v81, v17, v220
	v_sub_f32_e32 v80, v16, v220
	v_sub_f32_e32 v79, v15, v220
	v_sub_f32_e32 v78, v14, v220
	v_sub_f32_e32 v77, v13, v220
	v_sub_f32_e32 v76, v12, v220
	v_sub_f32_e32 v75, v11, v220
	v_sub_f32_e32 v74, v10, v220
	v_sub_f32_e32 v73, v9, v220
	v_sub_f32_e32 v72, v8, v220
	v_sub_f32_e32 v71, v7, v220
	v_sub_f32_e32 v70, v6, v220
	v_sub_f32_e32 v69, v5, v220
	v_sub_f32_e32 v68, v4, v220
	v_sub_f32_e32 v67, v3, v220
	v_sub_f32_e32 v66, v2, v220
;   v_s_0 = _anchor_v_s(v_s_0)
	;;#ASMSTART
	;;#ASMEND
	s_nop 0

; Prologue softmax first half.
;   v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
; NOTE:
;   v_max/v_permlane reductions produce the per-row maximum.
; SCHED HINT:
;   The first softmax half is bounded by source `sched_barrier(0); s_barrier();
;   sched_barrier(0)` before the next K/V prefetch. This keeps these initial
;   EXP operations and online-softmax state setup from crossing into the
;   steady-state handoff.
	v_exp_f32_e32 v183, v18
	v_exp_f32_e32 v184, v19
	v_exp_f32_e32 v182, v20
	v_exp_f32_e32 v180, v21
	v_exp_f32_e32 v178, v22
	v_exp_f32_e32 v179, v23
	v_exp_f32_e32 v162, v24
	v_exp_f32_e32 v163, v25
	v_exp_f32_e32 v119, v26
	v_exp_f32_e32 v121, v27
	v_exp_f32_e32 v117, v28
	v_exp_f32_e32 v120, v29
	v_exp_f32_e32 v116, v30
	v_exp_f32_e32 v118, v31
	v_exp_f32_e32 v114, v32
	v_exp_f32_e32 v115, v33
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v2, 12, v2
	v_or3_b32 v2, v35, v34, v2
;   rocdl.sched_barrier(0)
;   rocdl.s_barrier()
;   rocdl.sched_barrier(0)
	s_barrier

;   _async_load_k(fx.Index(2 * BLOCK_N), 0)
	s_mov_b32 m0, s31
	s_lshl_b32 s27, s18, 8
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v215, s[4:7], s27 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v216, s[4:7], s27 offen lds

; Prologue → main loop preheader, PART 1: zero-trip-count guard.
; PY (flash_attn_opus.py L1326-L1333):
;   for j, loop_args in range(
;       fx.Index(3),                              # start = 3
;       max_num_tiles - fx.Index(1),              # bound (computed here)
;       fx.Index(2),                              # step = 2
;       init=init_args,
;   ):
; NOTE:
;   This is LLVM-generated guard code for the FlyDSL `range(...)` loop.
;   It tests whether the loop body executes at least once (i.e., 3 < bound).
;   No single Python source line corresponds to this; it is the structural
;   lowering of the for-loop's pre-entry zero-iteration test.
;
;   s8/s9 hold `max_num_tiles` (loaded from kernel arguments earlier).
;     s_mov_b64 s[6:7], -1               ; flag := -1 (default: loop WILL iter)
;     s_add_u32/addc_u32  s[10:11], s[8:9], -1   ; bound = max_num_tiles - 1
;                                                ; (used by loop back-edge at ASM 1073
;                                                ; and by epilogue addressing)
;     v_cmp_gt_u64_e64 s[12:13], s[8:9], 4       ; cond = (max_num_tiles > 4)
;                                                ;   ⟺ (3 < max_num_tiles - 1)
;                                                ;   ⟺ "loop body runs at least once"
;     s_and_b64 vcc, exec, s[12:13]              ; gate cond by exec
;     v_lshlrev_b32_e32 v219, 1, v2              ; v219 = v2 << 1
;                                                ;   v2 = wave/lane-derived base;
;                                                ;   v219 will be reused below to
;                                                ;   build per-lane LDS read addrs
;                                                ;   (v217, v221) for both paths.
;     s_cbranch_vccnz .LBB0_6            ; if loop will iter → take fast fall-through
;     ; --- else: zero-iter path, fall through to here ---
;     v_add_u32_e32 v217, 0x4100, v219   ; v217 := v219 + 0x4100
;                                        ; pre-init per-lane LDS Q-buffer addr
;                                        ; (epilogue needs v217 for its ds_read; the
;                                        ; loop path will overwrite this at ASM 870)
;     s_mov_b64 s[6:7], 0                ; flag := 0 (loop will NOT iter; later
;                                        ; tested at ASM 832 to branch to epilogue)
;     s_branch .LBB0_7                   ; jump to merge block
	s_mov_b64 s[6:7], -1
	s_add_u32 s10, s8, -1
	s_addc_u32 s11, s9, -1
	v_cmp_gt_u64_e64 s[12:13], s[8:9], 4
	s_and_b64 vcc, exec, s[12:13]
	v_lshlrev_b32_e32 v219, 1, v2
	s_cbranch_vccnz .LBB0_6
	v_add_u32_e32 v217, 0x4100, v219
	s_mov_b64 s[6:7], 0
	s_branch .LBB0_7
.LBB0_6:
.LBB0_7:
; Prologue → main loop preheader, PART 2: merge of zero-iter guard.
; NOTE:
;   .LBB0_6 is the "loop will iter" branch target — it has no body and
;   immediately falls through to .LBB0_7. .LBB0_7 is the merge block of the
;   if/else above. Both paths converge here.
;   From here, code is COMMON to both the zero-iter and the will-iter paths;
;   the flag s[6:7] is used at ASM 832 to dispatch to either the epilogue
;   (zero-iter) or the loop body (will-iter).

; Prologue → main loop preheader, PART 3: zero-init of loop-carry regs.
; PY (flash_attn_opus.py L1319-L1323):
;   l_row_init = c_zero_f                        # = 0.0
;   init_args = [m_row_pro, l_row_init]
;   for _ in range_constexpr(D_CHUNKS):          # D_CHUNKS = 4
;       init_args.append(c_zero_v16f32)          # 16 zeros each → 64 acc regs total
;   init_args.append(_v_pair_to_vec32(v_p_0))
; NOTE:
;   s28 is a per-XCC LDS base (computed in prologue); s19/s17 are the LDS
;   K-tile base offsets for buf-0 (s19 = s28 + 0x8500) and buf-1
;   (s17 = s28 + 0xa700). These will be loaded into m0 register before
;   `buffer_load_dwordx4 ... lds` in Cluster 0 staging (ASM 1080-1087 area).
;   v17 is set to 0 first, then broadcast to v2..v65 (64 regs = 4 D_CHUNKS
;   × 16 vec elements each) — this materializes `init_args[2:2+D_CHUNKS] =
;   [c_zero_v16f32] * 4`. v218 := 0 corresponds to `l_row_init = c_zero_f`.
;   The `s_andn2_b64 vcc, exec, s[6:7]` computes vcc = exec & ~s[6:7]:
;     • s[6:7]=-1 (loop iter)  → vcc = 0       → don't branch at ASM 832
;     • s[6:7]= 0 (loop skip)  → vcc = exec≠0  → branch to .LBB0_16 at ASM 832
	s_add_i32 s19, s28, 0x8500
	s_add_i32 s17, s28, 0xa700
	v_mov_b32_e32 v17, 0
	s_andn2_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v15, v17
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v13, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v9, v17
	v_mov_b32_e32 v8, v17
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v4, v17
	v_mov_b32_e32 v3, v17
	v_mov_b32_e32 v2, v17
	v_mov_b32_e32 v65, v17
	v_mov_b32_e32 v64, v17
	v_mov_b32_e32 v63, v17
	v_mov_b32_e32 v62, v17
	v_mov_b32_e32 v61, v17
	v_mov_b32_e32 v60, v17
	v_mov_b32_e32 v59, v17
	v_mov_b32_e32 v58, v17
	v_mov_b32_e32 v57, v17
	v_mov_b32_e32 v56, v17
	v_mov_b32_e32 v55, v17
	v_mov_b32_e32 v54, v17
	v_mov_b32_e32 v53, v17
	v_mov_b32_e32 v52, v17
	v_mov_b32_e32 v51, v17
	v_mov_b32_e32 v50, v17
	v_mov_b32_e32 v49, v17
	v_mov_b32_e32 v48, v17
	v_mov_b32_e32 v47, v17
	v_mov_b32_e32 v46, v17
	v_mov_b32_e32 v45, v17
	v_mov_b32_e32 v44, v17
	v_mov_b32_e32 v43, v17
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v41, v17
	v_mov_b32_e32 v40, v17
	v_mov_b32_e32 v39, v17
	v_mov_b32_e32 v38, v17
	v_mov_b32_e32 v37, v17
	v_mov_b32_e32 v36, v17
	v_mov_b32_e32 v35, v17
	v_mov_b32_e32 v34, v17
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v32, v17
	v_mov_b32_e32 v31, v17
	v_mov_b32_e32 v30, v17
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v28, v17
	v_mov_b32_e32 v27, v17
	v_mov_b32_e32 v26, v17
	v_mov_b32_e32 v25, v17
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v23, v17
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v21, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v18, v17

; Prologue → main loop preheader, PART 4: dispatch on zero-iter flag.
; NOTE:
;   If s[6:7]=0 (zero-trip-count path) → branch to .LBB0_16 (epilogue).
;   In that case v_o accumulators (v2..v65) are all zero from PART 3, l_row
;   (v218) is zero, m_row (v220) is `m_row_pro` from prologue softmax max,
;   v_p_0 was computed in prologue `_attn_exp2_slice(v_s_0, 0, 16)`.
;   The epilogue then uses these `init_args` directly without ever running
;   the main loop body — this is the FlyDSL `range(...)`-with-init semantic
;   when the iteration range is empty (PY L1326-L1333).
	s_cbranch_vccnz .LBB0_16

; Prologue → main loop preheader, PART 5: loop-only preheader setup.
; PY (flash_attn_opus.py L1326-L1333, L1338, L1190, L1294):
;   for j, loop_args in range(fx.Index(3), max_num_tiles - fx.Index(1),
;                             fx.Index(2), init=init_args):
;       ...
;       j_idx = j                                # alias for clarity in body
; NOTE:
;   These instructions only run when the loop will iterate at least once
;   (fall-through from ASM 832 `s_cbranch_vccnz`). They prepare:
;   • Per-lane LDS read addresses for K/V buffers:
;       v217 = v219 + 0x4100   (K buf-0 LDS base, used by ASM 1274+ ds_read)
;       v221 = v219 + 0xc600   (K buf-1 LDS base, used by ASM 1820+ ds_read)
;   • Scale factors:
;       s35 = s18 << 9         (V LDS row stride: 9 = log2(BLOCK_N*4 bytes))
;       s18 = s18 * 0x180      (V VMEM/global stride: 0x180 = 3*128 bytes)
;   • Per-lane offset chain (v0):
;       v0 = ((v0 >> 1) & 0xe0) + (v0 & 31)     # lane → (row, col) decode
;       v0 = s26 + v2 + v0 - v1 + 0xffffff60    # subtract base and align
;     This v0 will be the per-lane VMEM offset for `buffer_load_dwordx4` in
;     each iteration's Cluster 0 (`_async_load_v((j_idx - 2) * BLOCK_N, 1)`).
;   • Loop induction state:
;       s[12:13] = 3           # j = 3 (loop start, PY L1329 `fx.Index(3)`)
;       s37 = 0                # rolling LDS K-tile DMA base (snapshot used by
;                              # ASM 1081 `s_add_i32 s38, s34, s37` in Cluster 0)
;       s26 = 0xc0 = 192       # j_idx-derived KV byte offset initial value
;       v218 = 0               # l_row (= c_zero_f, re-init for loop-carry SSA)
;       v2 = 0                 # zero source for v_o re-init below
;   • Scalar constants used INSIDE the loop body (loop-invariant):
;       s36 = 0x41000000 = 8.0f         # PY L447 `c_eight_f = OPUS_RESCALE_THRESHOLD`
;                                       # used by `_lazy_rescale_o` at PY L1190
;                                       # `below = (m_diff <= c_eight_f)`.
;                                       # Compare at ASM 2190: `v_cmp_ge_f32 vcc, s36, v99`.
;       v222 = 0xff800000 = -inf (f32)  # PY L435 `c_neg_inf` / `neg_inf_v`
;                                       # used by `_causal_mask_*` (PY L1294)
;                                       # to write NEG_INF into masked-out lanes.
;       s6/s7 = s2/s3                   # global buffer descriptor copy
	v_add_u32_e32 v217, 0x4100, v219
	v_add_u32_e32 v221, 0xc600, v219
	s_lshl_b32 s35, s18, 9
	s_mulk_i32 s18, 0x180
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v2, 0xe0, v2
	v_and_b32_e32 v0, 31, v0
	v_add3_u32 v0, s26, v2, v0
	v_sub_u32_e32 v0, v0, v1
	v_add_u32_e32 v0, 0xffffff60, v0
	s_mov_b64 s[12:13], 3
	v_mov_b32_e32 v2, 0
	s_movk_i32 s26, 0xc0
	s_mov_b32 s37, 0
	v_mov_b32_e32 v218, 0
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	s_mov_b32 s36, 0x41000000
	v_mov_b32_e32 v222, 0xff800000

; OPUS prologue → main loop preheader, PART 6: PHI-elimination zero-init.
; PY (flash_attn_opus.py L1321-L1322): the same `init_args.append(c_zero_v16f32)`.
; NOTE:
;   This block is structurally REQUIRED by LLVM's PHI-elimination, not by data
;   flow. Two distinct phi nodes (epilogue's v_o phi and loop-body's v_o phi)
;   both consume the same SSA constant zero from PART 3, but PHI-elimination
;   processes each phi independently and inserts copies in every predecessor:
;     • PART 3 = copies for epilogue phi's "skip_loop" predecessor.
;     • PART 6 = copies for loop-body phi's "preheader_loop" predecessor.
;   Machine Copy Propagation is intra-block and cannot see across the
;   s_cbranch_vccnz at ASM 832, so it cannot prove the PART 3 zeros are still
;   live here and remove these copies. Only v2's re-init at PART 5 line 881 is
;   data-flow necessary (PART 5 clobbered v2 as an address-arithmetic scratch);
;   the 63 copies below (v3..v17, v50..v65, v34..v49, v18..v33) are LLVM
;   codegen redundancy with negligible runtime cost (one-time, ~63 cycles).
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2

; Prologue → main loop preheader, PART 7: enter loop body.
; PY (flash_attn_opus.py L1326-L1333, L1338):
;   for j, loop_args in range(...): j_idx = j; ...  # body starts here
; NOTE:
;   Unconditional jump SKIPS over .LBB0_9 (the Cluster 7 lazy-rescale merge
;   block at ASM 973, which is positioned BEFORE .LBB0_10 for fall-through
;   layout — see prior annotation on .LBB0_9). The first iteration enters
;   directly at .LBB0_10 (loop body, ASM 1074, Cluster 0/1 staging for tile A).
;   Subsequent iterations reach .LBB0_10 by falling through from the back-edge
;   at ASM 1073 (`s_cbranch_vccz .LBB0_16; fall through`).
	s_branch .LBB0_10

.LBB0_9:
; Cluster 7:
; v_o = _mma1_step_k(1, v_p_1, v_v, v_o)
; v_o = _mma1_step_k(2, v_p_1, v_v, v_o)
; v_o = _mma1_step_k(3, v_p_1, v_v, v_o)
; v_s_0 = _attn_sub_row(v_s_0, m_row)
; v_s_0 = _anchor_v_s(v_s_0)
; v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
; _sched_barrier_pairs(6, 5, 4)
; _sched_barrier_exp_pairs(6, 3, 4)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[118:121], v[102:105], v[2:17]
	v_cndmask_b32_e32 v220, v98, v220, vcc
	v_sub_f32_e32 v129, v81, v220
	v_sub_f32_e32 v128, v80, v220
	v_sub_f32_e32 v127, v79, v220
	v_sub_f32_e32 v126, v78, v220
	v_mfma_f32_32x32x16_bf16 v[50:65], v[114:117], v[102:105], v[50:65]
	v_sub_f32_e32 v125, v77, v220
	v_sub_f32_e32 v124, v76, v220
	v_sub_f32_e32 v123, v75, v220
	v_sub_f32_e32 v122, v74, v220
	v_sub_f32_e32 v121, v73, v220
	v_mfma_f32_32x32x16_bf16 v[34:49], v[194:197], v[102:105], v[34:49]
	v_sub_f32_e32 v120, v72, v220
	v_sub_f32_e32 v119, v71, v220
	v_sub_f32_e32 v118, v70, v220
	v_sub_f32_e32 v117, v69, v220
	v_sub_f32_e32 v116, v68, v220
	v_mfma_f32_32x32x16_bf16 v[18:33], v[198:201], v[102:105], v[18:33]
	v_sub_f32_e32 v115, v67, v220
	v_sub_f32_e32 v114, v66, v220
	v_sub_f32_e32 v81, v97, v220
	v_sub_f32_e32 v80, v96, v220
	v_sub_f32_e32 v79, v95, v220
	v_mfma_f32_32x32x16_bf16 v[2:17], v[182:185], v[106:109], v[2:17]
	v_sub_f32_e32 v78, v94, v220
	v_sub_f32_e32 v77, v93, v220
	v_sub_f32_e32 v76, v92, v220
	v_sub_f32_e32 v75, v91, v220
	v_sub_f32_e32 v74, v90, v220
	v_mfma_f32_32x32x16_bf16 v[50:65], v[186:189], v[106:109], v[50:65]
	v_sub_f32_e32 v73, v89, v220
	v_sub_f32_e32 v72, v88, v220
	v_sub_f32_e32 v71, v87, v220
	v_sub_f32_e32 v70, v86, v220
	v_sub_f32_e32 v69, v85, v220
	v_mfma_f32_32x32x16_bf16 v[34:49], v[190:193], v[106:109], v[34:49]
	v_sub_f32_e32 v68, v84, v220
	v_sub_f32_e32 v67, v83, v220
	v_sub_f32_e32 v66, v82, v220
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_exp_f32_e32 v183, v114
	v_exp_f32_e32 v184, v115
	v_exp_f32_e32 v182, v116
	v_mfma_f32_32x32x16_bf16 v[18:33], v[178:181], v[106:109], v[18:33]
	v_exp_f32_e32 v180, v117
	v_exp_f32_e32 v178, v118
	v_exp_f32_e32 v179, v119
	v_mfma_f32_32x32x16_bf16 v[2:17], v[162:165], v[110:113], v[2:17]
	v_exp_f32_e32 v162, v120
	v_exp_f32_e32 v163, v121
	v_exp_f32_e32 v119, v122
	v_mfma_f32_32x32x16_bf16 v[50:65], v[166:169], v[110:113], v[50:65]
	v_exp_f32_e32 v121, v123
	v_exp_f32_e32 v117, v124
	v_exp_f32_e32 v120, v125
	v_mfma_f32_32x32x16_bf16 v[34:49], v[170:173], v[110:113], v[34:49]
	v_exp_f32_e32 v116, v126
	v_exp_f32_e32 v118, v127
	v_exp_f32_e32 v114, v128
	v_mfma_f32_32x32x16_bf16 v[18:33], v[174:177], v[110:113], v[18:33]
	v_exp_f32_e32 v115, v129
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(0)
	s_setprio 0
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	s_barrier

; OPUS main loop back-edge / loop control (NOT inside any loop-body cluster).
; PY (flash_attn_opus.py L1326-L1333, L1338, L1668-L1669):
;   for j, loop_args in range(
;       fx.Index(3),                          # start = 3
;       max_num_tiles - fx.Index(1),          # bound = max_num_tiles - 1
;       fx.Index(2),                          # step = 2
;       init=init_args,
;   ):
;       ...                                   # body lowered to .LBB0_10 ... s_barrier above
;       j_idx = j                             # PY L1338 (alias)
;       ...
;       yield_args = [m_row, l_row] + v_o + [_v_pair_to_vec32(v_p_0)]
;       loop_results = yield yield_args       # PY L1669 — implicit back-edge fires here
; NOTE:
;   These 8 instructions are the LLVM-generated loop control / back-edge,
;   NOT a translation of any Python statement inside the loop body. They run
;   exactly ONCE per iteration, between the body's terminating `s_barrier`
;   (above) and either the next iteration's .LBB0_10 (fall-through) or the
;   epilogue .LBB0_16 (taken branch). The back-edge converges control from
;   both lazy-rescale merge points: it is reached only via .LBB0_9 fall-through
;   (Cluster 7's second lazy rescale → MMA1/softmax → s_barrier → here), so
;   j += 2 is incremented exactly once per Python-level loop iteration.
;
;     s_add_u32  s12, s12, 2     ; PY L1331 step: s[12:13] (= j) += 2
;     s_addc_u32 s13, s13, 0     ;   64-bit carry for j
;     s_addk_i32 s26, 0x80       ; j_idx-derived KV byte offset += 128 (= 2 * BLOCK_N
;                                ;   in some packed unit; tracks the loop-rolling K/V
;                                ;   tile base pointer used by Cluster 0 staging)
;     v_add_u32_e32 v0, 0xffffff80, v0   ; per-lane mirror of s26: v0 -= 128
;                                ;   (compensates for the absolute base advancing by
;                                ;   the same amount so per-lane VMEM offsets stay
;                                ;   referenced to the new tile base)
;     v_mov_b64_e32 v[82:83], s[10:11]   ; load loop bound to a VGPR pair so the
;                                ;   v_cmp_lt_i64 can use it as a vector source
;                                ;   (s[10:11] = max_num_tiles - 1, set in PART 1)
;     v_cmp_lt_i64_e32 vcc, s[12:13], v[82:83]  ; vcc = (j < max_num_tiles - 1)
;     s_mov_b32 s37, s38         ; snapshot the LDS K-tile DMA base used by this
;                                ;   iteration's Cluster 0 (`s_add_i32 s38, s34, s37`
;                                ;   at .LBB0_10 below). The "rolling base pointer"
;                                ;   for double-buffered LDS K loads.
;     s_cbranch_vccz .LBB0_16    ; if vccz (= !vcc → j NOT less than bound → loop
;                                ;   done), jump to epilogue .LBB0_16. Otherwise
;                                ;   fall through to .LBB0_10 (next iteration's
;                                ;   Cluster 0/1 staging for tile A).
	s_add_u32 s12, s12, 2
	s_addc_u32 s13, s13, 0
	s_addk_i32 s26, 0x80
	v_add_u32_e32 v0, 0xffffff80, v0
	v_mov_b64_e32 v[82:83], s[10:11]
	v_cmp_lt_i64_e32 vcc, s[12:13], v[82:83]
	s_mov_b32 s37, s38
	s_cbranch_vccz .LBB0_16

.LBB0_10:
; Main loop
; Cluster 0:
; _async_load_v((j_idx - fx.Index(2)) * fx.Index(BLOCK_N), 1)
; v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; _async_load_v((j_idx - fx.Index(2)) * fx.Index(BLOCK_N), 1)
	s_mov_b32 m0, s19
	s_add_i32 s38, s34, s37
	buffer_load_dwordx4 v215, s[0:3], s38 offen lds
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v216, s[0:3], s38 offen lds
;   v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
	ds_read_b128 v[98:101], v211 offset:34048
	ds_read_b128 v[126:129], v211 offset:34080
	ds_read_b128 v[164:167], v211 offset:34560
	ds_read_b128 v[168:171], v211 offset:34592
	ds_read_b128 v[172:175], v211 offset:34112
	ds_read_b128 v[186:189], v211 offset:34144
	ds_read_b128 v[190:193], v211 offset:34624
	ds_read_b128 v[194:197], v211 offset:34656
	ds_read_b128 v[198:201], v211 offset:42368
	ds_read_b128 v[202:205], v211 offset:42400
	ds_read_b128 v[206:209], v211 offset:42880
	ds_read_b128 v[224:227], v211 offset:42912
	ds_read_b128 v[228:231], v211 offset:42432
	ds_read_b128 v[232:235], v211 offset:42464
	ds_read_b128 v[236:239], v211 offset:42944
	ds_read_b128 v[240:243], v211 offset:42976
;   rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
;   _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
;   rocdl.sched_barrier(0)
;   rocdl.s_barrier()
;   rocdl.sched_barrier(0)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Main loop
; Cluster 1:
; v_s_1 = _mma0(v_k)
; v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
; tile_sum_a = _attn_sum(v_p_0)
; l_row = _fadd(l_row, tile_sum_a)
; v_p_0 = _cast_p(v_p_0)
; v_p_0 = _anchor_v_p(v_p_0)
; _sched_barrier_exp_pairs(6, 3, 1)
; _sched_barrier_pairs(10, 5, 1)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	v_mfma_f32_32x32x16_bf16 v[82:97], v[98:101], v[130:133], 0
; v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
	v_exp_f32_e32 v122, v66
	v_exp_f32_e32 v123, v67
	v_exp_f32_e32 v124, v68
	v_mfma_f32_32x32x16_bf16 v[98:113], v[164:167], v[130:133], 0
	v_exp_f32_e32 v125, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mfma_f32_32x32x16_bf16 v[82:97], v[126:129], v[134:137], v[82:97]
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v126, v74
	v_mfma_f32_32x32x16_bf16 v[98:113], v[168:171], v[134:137], v[98:113]
	v_exp_f32_e32 v127, v75
	v_exp_f32_e32 v128, v76
	v_exp_f32_e32 v129, v77
	v_mfma_f32_32x32x16_bf16 v[82:97], v[172:175], v[138:141], v[82:97]
	v_exp_f32_e32 v164, v78
	v_exp_f32_e32 v165, v79
	v_exp_f32_e32 v166, v80
	v_mfma_f32_32x32x16_bf16 v[98:113], v[190:193], v[138:141], v[98:113]
	v_exp_f32_e32 v81, v81
	v_mfma_f32_32x32x16_bf16 v[82:97], v[186:189], v[142:145], v[82:97]
; tile_sum_a = _attn_sum(v_p_0)
	v_add_f32_e32 v66, v183, v184
	v_add_f32_e32 v66, v66, v182
	v_add_f32_e32 v66, v66, v180
	v_add_f32_e32 v66, v66, v178
	v_add_f32_e32 v66, v66, v179
	v_mfma_f32_32x32x16_bf16 v[98:113], v[194:197], v[142:145], v[98:113]
	v_add_f32_e32 v66, v66, v162
	v_add_f32_e32 v66, v66, v163
	v_add_f32_e32 v66, v66, v119
	v_add_f32_e32 v66, v66, v121
	v_add_f32_e32 v66, v66, v117
	v_mfma_f32_32x32x16_bf16 v[82:97], v[198:201], v[146:149], v[82:97]
	v_add_f32_e32 v66, v66, v120
	v_add_f32_e32 v66, v66, v116
	v_add_f32_e32 v66, v66, v118
	v_add_f32_e32 v66, v66, v114
	v_add_f32_e32 v66, v66, v115
	v_mfma_f32_32x32x16_bf16 v[98:113], v[206:209], v[146:149], v[98:113]
	v_add_f32_e32 v66, v66, v122
	v_add_f32_e32 v66, v66, v123
	v_add_f32_e32 v66, v66, v124
	v_add_f32_e32 v66, v66, v125
	v_add_f32_e32 v66, v66, v70
	v_mfma_f32_32x32x16_bf16 v[82:97], v[202:205], v[150:153], v[82:97]
	v_add_f32_e32 v66, v66, v71
	v_add_f32_e32 v66, v66, v72
	v_add_f32_e32 v66, v66, v73
	v_add_f32_e32 v66, v66, v126
	v_add_f32_e32 v66, v66, v127
	v_mfma_f32_32x32x16_bf16 v[98:113], v[224:227], v[150:153], v[98:113]
	v_add_f32_e32 v66, v66, v128
	v_add_f32_e32 v66, v66, v129
	v_add_f32_e32 v66, v66, v164
	v_add_f32_e32 v66, v66, v165
	v_add_f32_e32 v66, v66, v166
	v_mfma_f32_32x32x16_bf16 v[82:97], v[228:231], v[154:157], v[82:97]
	v_add_f32_e32 v66, v66, v81
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e64 v66, v67 bound_ctrl:1
	v_add_f32_e32 v67, v218, v67
	v_add_f32_e32 v218, v67, v66
; v_p_0 = _cast_p(v_p_0)
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v183, v184
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[98:113], v[236:239], v[154:157], v[98:113]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v182, v180
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v178, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v122, v123
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v124, v125
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[232:235], v[158:161], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v72, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v119, v121
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v117, v120
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v116, v118
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[98:113], v[240:243], v[158:161], v[98:113]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v114, v115
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v126, v127
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v128, v129
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v80, v164, v165
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v81, v166, v81
	;;#ASMEND
	s_nop 0
; v_p_0 = _anchor_v_p(v_p_0)
	;;#ASMSTART
	;;#ASMEND
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	s_barrier

; Main loop
; Cluster 2:
; _async_load_k(j_idx * fx.Index(BLOCK_N), 1)
; v_v = _read_v_packs_for_buf(0, urv_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; _async_load_k(j_idx * fx.Index(BLOCK_N), 1)
	s_add_i32 s38, s18, s37
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v215, s[4:7], s38 offen lds
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v216, s[4:7], s38 offen lds
;   v_v = _read_v_packs_for_buf(0, urv_base_per_lane)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v217 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v217 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v217 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v217 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v217 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v217 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v217 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v217 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v217 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v217 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v217 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v217 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v217 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v217 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v217 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v217 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v217 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v217 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v217 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v217 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v217 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v217 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v217 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v217 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v217 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v217 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v217 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v217 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v217 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v217 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v217 offset:9664

	;;#ASMEND
;   rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
;   _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
;   rocdl.sched_barrier(0)
;   rocdl.s_barrier()
;   rocdl.sched_barrier(0)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 3:
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(1)
	s_setprio 1
; v_o = _mma1_step_k(0, v_p_0, v_v, v_o)
; v_s_1 = _v_s_vec_to_lists(v_s_1)
; m_tile_max_a = _attn_row_max(v_s_1)
; _sched_barrier_pairs(4, 5, 2)
; if const_expr(OPUS_LAZY_RESCALE):
; 	v_o, m_row, l_row, v_p_0 = _lazy_rescale_o(
; 		v_o, m_row, l_row, m_tile_max_a, v_p_0
; 	)
; v_o = _mma1_step_k(1, v_p_0, v_v, v_o)
; v_o = _mma1_step_k(2, v_p_0, v_v, v_o)
; v_o = _mma1_step_k(3, v_p_0, v_v, v_o)
; v_s_1 = _attn_sub_row(v_s_1, m_row)
; v_s_1 = _anchor_v_s(v_s_1)
; v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
; _sched_barrier_pairs(6, 5, 2)
; _sched_barrier_exp_pairs(6, 3, 2)
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(0)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[194:197], v[66:69], v[2:17]
; m_tile_max_a = _attn_row_max(v_s_1)
	v_max_f32_e32 v194, v82, v83
	v_max3_f32 v194, v194, v84, v85
	v_max3_f32 v194, v194, v86, v87
	v_max3_f32 v194, v194, v88, v89
	v_max3_f32 v194, v194, v90, v91
	v_mfma_f32_32x32x16_bf16 v[50:65], v[198:201], v[66:69], v[50:65]
	v_max3_f32 v194, v194, v92, v93
	v_max3_f32 v194, v194, v94, v95
	v_max3_f32 v194, v194, v96, v97
	v_max3_f32 v194, v194, v98, v99
	v_max3_f32 v194, v194, v100, v101
	v_mfma_f32_32x32x16_bf16 v[34:49], v[202:205], v[66:69], v[34:49]
	v_max3_f32 v194, v194, v102, v103
	v_max3_f32 v194, v194, v104, v105
	v_max3_f32 v194, v194, v106, v107
	v_max3_f32 v194, v194, v108, v109
	v_max3_f32 v194, v194, v110, v111
	v_mfma_f32_32x32x16_bf16 v[18:33], v[206:209], v[66:69], v[18:33]
	v_max3_f32 v66, v194, v112, v113
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e64 v66, v67 bound_ctrl:1
	v_max_f32_e32 v66, v66, v67
; v_o, m_row, l_row, v_p_0 = _lazy_rescale_o(
; 	v_o, m_row, l_row, m_tile_max_a, v_p_0
; )
; m_diff = _fsub(m_tile_max, m_row)
	v_sub_f32_e32 v67, v66, v220
; below = ArithValue(fx.Float32(m_diff) <= c_eight_f)
; ballot = rocdl.ballot(T.i64, _raw(below))
; all_below = arith.cmpi(
; 	arith.CmpIPredicate.eq,
; 	_raw(ballot),
; 	_read_exec_i64(),
; )
	v_cmp_ge_f32_e32 vcc, s36, v67
	s_cmp_eq_u64 vcc, exec
	s_cselect_b64 vcc, -1, 0
	s_cbranch_vccnz .LBB0_12
; corr = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_tile_max)))
	v_sub_f32_e32 v67, v220, v66
	v_exp_f32_e32 v68, v67
	s_nop 0
; _lazy_rescale_o else-branch: scale old O accumulators by corr.
;   scaled_accs = list(v_o)
;   _scale_o(scaled_accs, corr)
	v_pk_mul_f32 v[16:17], v[68:69], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[68:69], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[68:69], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[68:69], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[68:69], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[68:69], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[68:69], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[68:69], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[68:69], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[68:69], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[68:69], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[68:69], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[68:69], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[68:69], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[68:69], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[68:69], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[68:69], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[68:69], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[68:69], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[68:69], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[68:69], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[68:69], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[68:69], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[68:69], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[68:69], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[68:69], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[68:69], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[68:69], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[68:69], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[68:69], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[68:69], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[68:69], v[18:19] op_sel_hi:[0,1]
;   scaled_v_p = _scale_v_p(v_p, corr)
;   p_all = _v_p_to_vec32(v_p)
;   p_all_f32 = FPExt(p_all)          # bf16 -> f32
;   p_scaled_f32 = corr * p_all_f32
;   p_scaled_bf16 = FPTrunc(p_scaled_f32)
;   return _v_vec32_to_p(p_scaled_bf16)
	v_and_b32_e32 v195, 0xffff0000, v81
	v_lshlrev_b32_e32 v194, 16, v81
	v_and_b32_e32 v81, 0xffff0000, v80
	v_lshlrev_b32_e32 v80, 16, v80
	v_and_b32_e32 v197, 0xffff0000, v79
	v_lshlrev_b32_e32 v196, 16, v79
	v_and_b32_e32 v79, 0xffff0000, v78
	v_lshlrev_b32_e32 v78, 16, v78
	v_and_b32_e32 v199, 0xffff0000, v77
	v_lshlrev_b32_e32 v198, 16, v77
	v_and_b32_e32 v77, 0xffff0000, v76
	v_lshlrev_b32_e32 v76, 16, v76
	v_and_b32_e32 v201, 0xffff0000, v75
	v_lshlrev_b32_e32 v200, 16, v75
	v_and_b32_e32 v75, 0xffff0000, v74
	v_lshlrev_b32_e32 v74, 16, v74
	v_and_b32_e32 v203, 0xffff0000, v73
	v_lshlrev_b32_e32 v202, 16, v73
	v_and_b32_e32 v73, 0xffff0000, v72
	v_lshlrev_b32_e32 v72, 16, v72
	v_and_b32_e32 v205, 0xffff0000, v71
	v_lshlrev_b32_e32 v204, 16, v71
	v_and_b32_e32 v71, 0xffff0000, v70
	v_lshlrev_b32_e32 v70, 16, v70
	v_pk_mul_f32 v[206:207], v[68:69], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[68:69], v[204:205] op_sel_hi:[0,1]
	v_pk_mul_f32 v[204:205], v[68:69], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[68:69], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203], v[68:69], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[68:69], v[200:201] op_sel_hi:[0,1]
	v_pk_mul_f32 v[200:201], v[68:69], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[68:69], v[198:199] op_sel_hi:[0,1]
	v_pk_mul_f32 v[198:199], v[68:69], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[68:69], v[196:197] op_sel_hi:[0,1]
	v_pk_mul_f32 v[196:197], v[68:69], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[68:69], v[194:195] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v81, v80, v81
	v_cvt_pk_bf16_f32 v80, v196, v197
	v_cvt_pk_bf16_f32 v79, v78, v79
	v_cvt_pk_bf16_f32 v78, v198, v199
	v_cvt_pk_bf16_f32 v77, v76, v77
	v_cvt_pk_bf16_f32 v76, v200, v201
	v_cvt_pk_bf16_f32 v75, v74, v75
	v_cvt_pk_bf16_f32 v74, v202, v203
	v_cvt_pk_bf16_f32 v73, v72, v73
	v_cvt_pk_bf16_f32 v72, v204, v205
	v_cvt_pk_bf16_f32 v71, v70, v71
	v_cvt_pk_bf16_f32 v70, v206, v207
; scaled_l_row = _fmul(l_row, corr)
	v_mul_f32_e32 v218, v68, v218

.LBB0_12:
; v_o = _mma1_step_k(1, v_p_0, v_v, v_o)
; v_o = _mma1_step_k(2, v_p_0, v_v, v_o)
; v_o = _mma1_step_k(3, v_p_0, v_v, v_o)
; v_s_1 = _attn_sub_row(v_s_1, m_row)
; v_s_1 = _anchor_v_s(v_s_1)
; v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
; _sched_barrier_pairs(6, 5, 2)
; _sched_barrier_exp_pairs(6, 3, 2)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[2:17], v[190:193], v[70:73], v[2:17]
	v_cndmask_b32_e32 v220, v66, v220, vcc
	v_sub_f32_e32 v97, v97, v220
	v_sub_f32_e32 v96, v96, v220
	v_sub_f32_e32 v95, v95, v220
	v_sub_f32_e32 v94, v94, v220
	v_mfma_f32_32x32x16_bf16 v[50:65], v[178:181], v[70:73], v[50:65]
	v_sub_f32_e32 v93, v93, v220
	v_sub_f32_e32 v92, v92, v220
	v_sub_f32_e32 v91, v91, v220
	v_sub_f32_e32 v90, v90, v220
	v_sub_f32_e32 v89, v89, v220
	v_mfma_f32_32x32x16_bf16 v[34:49], v[182:185], v[70:73], v[34:49]
	v_sub_f32_e32 v88, v88, v220
	v_sub_f32_e32 v87, v87, v220
	v_sub_f32_e32 v86, v86, v220
	v_sub_f32_e32 v85, v85, v220
	v_sub_f32_e32 v84, v84, v220
	v_mfma_f32_32x32x16_bf16 v[18:33], v[186:189], v[70:73], v[18:33]
	v_sub_f32_e32 v83, v83, v220
	v_sub_f32_e32 v82, v82, v220
	v_sub_f32_e32 v113, v113, v220
	v_sub_f32_e32 v112, v112, v220
	v_sub_f32_e32 v111, v111, v220
	v_mfma_f32_32x32x16_bf16 v[2:17], v[166:169], v[74:77], v[2:17]
	v_sub_f32_e32 v110, v110, v220
	v_sub_f32_e32 v109, v109, v220
	v_sub_f32_e32 v108, v108, v220
	v_sub_f32_e32 v107, v107, v220
	v_sub_f32_e32 v106, v106, v220
	v_mfma_f32_32x32x16_bf16 v[50:65], v[170:173], v[74:77], v[50:65]
	v_sub_f32_e32 v105, v105, v220
	v_sub_f32_e32 v104, v104, v220
	v_sub_f32_e32 v103, v103, v220
	v_sub_f32_e32 v102, v102, v220
	v_sub_f32_e32 v101, v101, v220
	v_mfma_f32_32x32x16_bf16 v[34:49], v[174:177], v[74:77], v[34:49]
	v_sub_f32_e32 v100, v100, v220
	v_sub_f32_e32 v99, v99, v220
	v_sub_f32_e32 v98, v98, v220
;   v_s_1 = _anchor_v_s(v_s_1)
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
;   v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
	v_exp_f32_e32 v166, v82
	v_exp_f32_e32 v167, v83
	v_exp_f32_e32 v168, v84
	v_mfma_f32_32x32x16_bf16 v[18:33], v[162:165], v[74:77], v[18:33]
	v_exp_f32_e32 v162, v85
	v_exp_f32_e32 v163, v86
	v_exp_f32_e32 v164, v87
	v_mfma_f32_32x32x16_bf16 v[2:17], v[114:117], v[78:81], v[2:17]
	v_exp_f32_e32 v114, v88
	v_exp_f32_e32 v115, v89
	v_exp_f32_e32 v116, v90
	v_mfma_f32_32x32x16_bf16 v[50:65], v[118:121], v[78:81], v[50:65]
	v_exp_f32_e32 v117, v91
	v_exp_f32_e32 v118, v92
	v_exp_f32_e32 v119, v93
	v_mfma_f32_32x32x16_bf16 v[34:49], v[122:125], v[78:81], v[34:49]
	v_exp_f32_e32 v120, v94
	v_exp_f32_e32 v121, v95
	v_exp_f32_e32 v122, v96
	v_mfma_f32_32x32x16_bf16 v[18:33], v[126:129], v[78:81], v[18:33]
	v_exp_f32_e32 v123, v97
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(0)
	s_setprio 0
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	s_barrier

; Cluster 4:
; _async_load_v((j_idx - fx.Index(1)) * fx.Index(BLOCK_N), 0)
; v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; _async_load_v((j_idx - fx.Index(1)) * fx.Index(BLOCK_N), 0)
	s_mov_b32 m0, s28
	s_add_i32 s38, s27, s37
	buffer_load_dwordx4 v215, s[0:3], s38 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v216, s[0:3], s38 offen lds
; v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
	ds_read_b128 v[82:85], v211
	ds_read_b128 v[170:173], v211 offset:32
	ds_read_b128 v[174:177], v211 offset:512
	ds_read_b128 v[178:181], v211 offset:544
	ds_read_b128 v[182:185], v211 offset:64
	ds_read_b128 v[186:189], v211 offset:96
	ds_read_b128 v[190:193], v211 offset:576
	ds_read_b128 v[194:197], v211 offset:608
	ds_read_b128 v[198:201], v211 offset:8320
	ds_read_b128 v[202:205], v211 offset:8352
	ds_read_b128 v[206:209], v211 offset:8832
	ds_read_b128 v[224:227], v211 offset:8864
	ds_read_b128 v[228:231], v211 offset:8384
	ds_read_b128 v[232:235], v211 offset:8416
	ds_read_b128 v[236:239], v211 offset:8896
	ds_read_b128 v[240:243], v211 offset:8928
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 5:
; v_s_0 = _mma0(v_k)
; v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
; tile_sum_b = _attn_sum(v_p_1)
; l_row = _fadd(l_row, tile_sum_b)
; v_p_1 = _cast_p(v_p_1)
; v_p_1 = _anchor_v_p(v_p_1)
; _sched_barrier_exp_pairs(6, 3, 3)
; _sched_barrier_pairs(10, 5, 3)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	v_mfma_f32_32x32x16_bf16 v[66:81], v[82:85], v[130:133], 0
	v_exp_f32_e32 v124, v98
	v_exp_f32_e32 v125, v99
	v_exp_f32_e32 v126, v100
	v_mfma_f32_32x32x16_bf16 v[82:97], v[174:177], v[130:133], 0
	v_exp_f32_e32 v127, v101
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_mfma_f32_32x32x16_bf16 v[66:81], v[170:173], v[134:137], v[66:81]
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v128, v106
	v_mfma_f32_32x32x16_bf16 v[82:97], v[178:181], v[134:137], v[82:97]
	v_exp_f32_e32 v129, v107
	v_exp_f32_e32 v165, v108
	v_exp_f32_e32 v169, v109
	v_mfma_f32_32x32x16_bf16 v[66:81], v[182:185], v[138:141], v[66:81]
	v_exp_f32_e32 v170, v110
	v_exp_f32_e32 v171, v111
	v_exp_f32_e32 v172, v112
	v_mfma_f32_32x32x16_bf16 v[82:97], v[190:193], v[138:141], v[82:97]
	v_exp_f32_e32 v113, v113
	v_mfma_f32_32x32x16_bf16 v[66:81], v[186:189], v[142:145], v[66:81]
	v_add_f32_e32 v98, v166, v167
	v_add_f32_e32 v98, v98, v168
	v_add_f32_e32 v98, v98, v162
	v_add_f32_e32 v98, v98, v163
	v_add_f32_e32 v98, v98, v164
	v_mfma_f32_32x32x16_bf16 v[82:97], v[194:197], v[142:145], v[82:97]
	v_add_f32_e32 v98, v98, v114
	v_add_f32_e32 v98, v98, v115
	v_add_f32_e32 v98, v98, v116
	v_add_f32_e32 v98, v98, v117
	v_add_f32_e32 v98, v98, v118
	v_mfma_f32_32x32x16_bf16 v[66:81], v[198:201], v[146:149], v[66:81]
	v_add_f32_e32 v98, v98, v119
	v_add_f32_e32 v98, v98, v120
	v_add_f32_e32 v98, v98, v121
	v_add_f32_e32 v98, v98, v122
	v_add_f32_e32 v98, v98, v123
	v_mfma_f32_32x32x16_bf16 v[82:97], v[206:209], v[146:149], v[82:97]
	v_add_f32_e32 v98, v98, v124
	v_add_f32_e32 v98, v98, v125
	v_add_f32_e32 v98, v98, v126
	v_add_f32_e32 v98, v98, v127
	v_add_f32_e32 v98, v98, v102
	v_mfma_f32_32x32x16_bf16 v[66:81], v[202:205], v[150:153], v[66:81]
	v_add_f32_e32 v98, v98, v103
	v_add_f32_e32 v98, v98, v104
	v_add_f32_e32 v98, v98, v105
	v_add_f32_e32 v98, v98, v128
	v_add_f32_e32 v98, v98, v129
	v_mfma_f32_32x32x16_bf16 v[82:97], v[224:227], v[150:153], v[82:97]
	v_add_f32_e32 v98, v98, v165
	v_add_f32_e32 v98, v98, v169
	v_add_f32_e32 v98, v98, v170
	v_add_f32_e32 v98, v98, v171
	v_add_f32_e32 v98, v98, v172
	v_mfma_f32_32x32x16_bf16 v[66:81], v[228:231], v[154:157], v[66:81]
	v_add_f32_e32 v223, v98, v113
	v_mov_b32_e32 v224, v223
	s_nop 1
	v_permlane32_swap_b32_e64 v223, v224 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v166, v167
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v168, v162
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v163, v164
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[236:239], v[154:157], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v114, v115
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v124, v125
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v126, v127
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v102, v103
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v104, v105
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[66:81], v[232:235], v[158:161], v[66:81]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v116, v117
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v118, v119
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v120, v121
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v122, v123
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v128, v129
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[240:243], v[158:161], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v165, v169
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v170, v171
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v172, v113
	;;#ASMEND
	s_nop 0
;   v_p_1 = _anchor_v_p(v_p_1)
	;;#ASMSTART
	;;#ASMEND
;   rocdl.sched_barrier(0)
;   rocdl.s_barrier()
;   rocdl.sched_barrier(0)
	s_barrier

; Cluster 6:
; Main loop
; _async_load_k((j_idx + fx.Index(1)) * fx.Index(BLOCK_N), 0)
; v_packs_b = _read_v_packs_for_buf(1, urv_base_per_lane)
; if const_expr(CAUSAL):
; 	v_s_0 = _causal_mask_prologue_if_needed(
; 		v_s_0,
; 		j_idx - fx.Index(1),
; 		j_idx * fx.Index(BLOCK_N),
; 	)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; _async_load_k((j_idx + fx.Index(1)) * fx.Index(BLOCK_N), 0)
	s_mov_b32 m0, s31
	s_add_i32 s37, s35, s37
	buffer_load_dwordx4 v215, s[4:7], s37 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v216, s[4:7], s37 offen lds
;   v_packs_b = _read_v_packs_for_buf(1, urv_base_per_lane)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v221 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v221 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v221 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v221 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v221 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v221 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v221 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v221 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v221 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v221 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v221 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v221 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v221 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v221 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v221 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v221 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v221 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v221 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v221 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v221 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v221 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v221 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v221 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v221 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v221 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v221 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v221 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v221 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v221 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s16, s26
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v221 offset:9664

	;;#ASMEND
; PY:
;   if const_expr(CAUSAL):
;       v_s_0 = _causal_mask_prologue_if_needed(
;           v_s_0,
;           j_idx - fx.Index(1),         # tile_idx = j - 1
;           j_idx * fx.Index(BLOCK_N),   # kv_end_pos = j * 64
;       )
; CAUSAL DESIGN INVARIANT (why loop body has only ONE mask call for 2 tiles/iter):
;   Each loop iter processes 2 KV tiles: v_s_1 (= tile j_idx - 2, Cluster 1)
;   and v_s_0 (= tile j_idx - 1, Cluster 5). Only v_s_0 has a mask call here;
;   v_s_1 has NO mask call anywhere in the loop body. This is correct because:
;
;   1. Under causal, max_num_tiles is clipped (PY L460-L479):
;        max_num_tiles = ceil(q_block_end / BLOCK_N) = 4(q_block_idx + 1)
;      so each workgroup has EXACTLY 4 boundary tiles {4k, 4k+1, 4k+2, 4k+3}
;      where k = q_block_idx. Tiles T < 4k are fully unmasked; tiles T >= 4(k+1)
;      are never reached (clipped away).
;
;   2. The 4 boundary tiles are partitioned across the kernel:
;        • Epilogue masks tiles 4k+1, 4k+2, 4k+3 (PY L1732, L1841, L1948).
;        • LAST loop iter (j = max_num_tiles - 3 = 4k+1) masks tile 4k as
;          its v_s_0 (tile j_idx - 1 = 4k) → this code path.
;        • Tile 0 (only a boundary for workgroup 0) is handled by prologue
;          (PY L1298); for workgroup 0 the loop body is empty (PART 1 guard).
;
;   3. v_s_1 = tile (j_idx - 2) is ALWAYS structurally < 4k:
;        • For iter j < max_num_tiles - 3: j_idx - 2 < 4k - 1 < 4k.
;        • For the LAST iter j = 4k+1: j_idx - 2 = 4k - 1 < 4k.
;      So v_s_1 is never a boundary tile — no mask call is needed for it.
;
;   4. v_s_0 = tile (j_idx - 1) only triggers a REAL mask in the LAST iter
;      (j = 4k+1, tile j-1 = 4k). For earlier iters, the SCF if-op's runtime
;      check `q_start_pos < kv_end_pos` (PY L1038-L1042) fails for all waves
;      in the workgroup (q_start_pos >= k*256 >= 4k*64 > (j-1)*64), and the
;      s_cbranch_scc1 below takes the no-op .LBB0_14 ELSE branch.
;
;   So `_causal_mask_prologue_if_needed` here is DEFENSIVE: structurally a
;   no-op for all but the last iter of the loop.
;
;   ASM that follows is the THEN branch of `_causal_mask_prologue_if_needed`'s
;   SCF if-op (the actual mask application via `_causal_mask_inplace`,
;   PY L949-L1018). The 16 ;;#ASMSTART/;;#ASMEND inline-asm blocks below are
;   8 vec2_imm pairs for s_hi (each pair masks 2 of 16 hi-half elements with
;   (thr_x, thr_y) ∈ {(0,1),(2,3),(8,9),(10,11),(16,17),(18,19),(24,25),
;   (26,27)}). The 8 pairs for s_lo are emitted earlier in the same THEN
;   branch (above this point). v222 holds 0xff800000 = -inf, set in PART 5
;   of the preheader.
	s_cbranch_scc1 .LBB0_14
	v_add_u32_e32 v225, 32, v0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 0
	v_cmp_lt_i32_e64 s[42:43], v225, 1
	v_cndmask_b32_e64 v66, v66, v222, s[40:41]
	v_cndmask_b32_e64 v67, v67, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 2
	v_cmp_lt_i32_e64 s[42:43], v225, 3
	v_cndmask_b32_e64 v68, v68, v222, s[40:41]
	v_cndmask_b32_e64 v69, v69, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 8
	v_cmp_lt_i32_e64 s[42:43], v225, 9
	v_cndmask_b32_e64 v70, v70, v222, s[40:41]
	v_cndmask_b32_e64 v71, v71, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 10
	v_cmp_lt_i32_e64 s[42:43], v225, 11
	v_cndmask_b32_e64 v72, v72, v222, s[40:41]
	v_cndmask_b32_e64 v73, v73, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 16
	v_cmp_lt_i32_e64 s[42:43], v225, 17
	v_cndmask_b32_e64 v74, v74, v222, s[40:41]
	v_cndmask_b32_e64 v75, v75, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 18
	v_cmp_lt_i32_e64 s[42:43], v225, 19
	v_cndmask_b32_e64 v76, v76, v222, s[40:41]
	v_cndmask_b32_e64 v77, v77, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 24
	v_cmp_lt_i32_e64 s[42:43], v225, 25
	v_cndmask_b32_e64 v78, v78, v222, s[40:41]
	v_cndmask_b32_e64 v79, v79, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v225, 26
	v_cmp_lt_i32_e64 s[42:43], v225, 27
	v_cndmask_b32_e64 v80, v80, v222, s[40:41]
	v_cndmask_b32_e64 v81, v81, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 0
	v_cmp_lt_i32_e64 s[42:43], v0, 1
	v_cndmask_b32_e64 v82, v82, v222, s[40:41]
	v_cndmask_b32_e64 v83, v83, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 2
	v_cmp_lt_i32_e64 s[42:43], v0, 3
	v_cndmask_b32_e64 v84, v84, v222, s[40:41]
	v_cndmask_b32_e64 v85, v85, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 8
	v_cmp_lt_i32_e64 s[42:43], v0, 9
	v_cndmask_b32_e64 v86, v86, v222, s[40:41]
	v_cndmask_b32_e64 v87, v87, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 10
	v_cmp_lt_i32_e64 s[42:43], v0, 11
	v_cndmask_b32_e64 v88, v88, v222, s[40:41]
	v_cndmask_b32_e64 v89, v89, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 16
	v_cmp_lt_i32_e64 s[42:43], v0, 17
	v_cndmask_b32_e64 v90, v90, v222, s[40:41]
	v_cndmask_b32_e64 v91, v91, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 18
	v_cmp_lt_i32_e64 s[42:43], v0, 19
	v_cndmask_b32_e64 v92, v92, v222, s[40:41]
	v_cndmask_b32_e64 v93, v93, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 24
	v_cmp_lt_i32_e64 s[42:43], v0, 25
	v_cndmask_b32_e64 v94, v94, v222, s[40:41]
	v_cndmask_b32_e64 v95, v95, v222, s[42:43]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[40:41], v0, 26
	v_cmp_lt_i32_e64 s[42:43], v0, 27
	v_cndmask_b32_e64 v96, v96, v222, s[40:41]
	v_cndmask_b32_e64 v97, v97, v222, s[42:43]
	;;#ASMEND
.LBB0_14:
;   1. 源码层级展开
;   _attn_sum 的内联展开后，L1561-L1562 这两行代码在 IR 层（未优化前）实际包含 33 次浮点加法：

;   # L1561: tile_sum_b = _attn_sum(v_p_1)  →  内联展开为：
;   local_sum = 0.0                                # init
;   local_sum += v_p_1.lo_partial_list[0]          # 32 个累加（lo 16 + hi 16）
;   local_sum += v_p_1.lo_partial_list[1]
;   ...
;   local_sum += v_p_1.hi_full[15]                 # 第 32 个加法（_attn_sum 内）
;   lhs_sum, rhs_sum = _reduction_pair(local_sum)  # permlane32_swap，无加法
;   tile_sum_b = lhs_sum + rhs_sum                 # 第 33 个加法（_attn_sum 的 return）
;   # L1562:
;   l_row = l_row + tile_sum_b                     # 第 34 个加法（在 _attn_sum 外）

;   整条 IR 链路总共有 32 (在 _attn_sum 中累加) + 1 (在 _attn_sum 中合并 lhs/rhs) + 1 (在 L1562 中累加进 l_row) = 34 次 v_add_f32。

;   2. ASM 层级实际产物
;   LLVM 的代数重关联（reassociation）把最后两次加法（即 _attn_sum 的 return add 和 L1562 的 add）合并消除掉了中间值 tile_sum_b：

;    FlyDSL/flash_attn_opus.v1.s lines 32-40

;   v_add_f32_e32 v223, v98, v113        ; ASM 1697: local_sum (= _attn_sum 32 次累加链的最后一条)
;   v_mov_b32_e32 v224, v223             ; ASM 1698: 复制 local_sum 到 v224，准备喂给 permlane
;   s_nop 1                              ; ASM 1699: hazard 间隔
;   v_permlane32_swap_b32_e64 v223, v224 bound_ctrl:1   ; ASM 1700: 跨 lane 交换
;                                        ;   交换后：v224 = lhs_sum，v223 = rhs_sum
;   ... (中间 320+ 条 MFMA/EXP/causal_mask/cvt 指令，调度器插在 1700 与 2022 之间) ...
;   v_add_f32_e32 v218, v218, v224       ; ASM 2022: l_row += lhs_sum
;   v_add_f32_e32 v218, v218, v223       ; ASM 2023: l_row += rhs_sum
;   s_waitcnt vmcnt(4) lgkmcnt(0)        ; ASM 2024: 不属于这一组

;   寄存器语义对照表：

;   ┌───────────────┬─────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────┐
;   │ VGPR          │ 角色                                                                │ 写入时机                                      │
;   ├───────────────┼─────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
;   │ v98           │ _attn_sum 内部 32 元素链式累加的 running sum                        │ ASM 1664-1696（连续 31 条 v_add_f32）         │
;   │ v113          │ 第 32 个被累加的元素（hi_full[15] 的 exp2 结果）                    │ 上游 v_exp_f32 产生                           │
;   │ v223 (初值)   │ local_sum，即 v98 + v113                                            │ ASM 1697                                      │
;   │ v224 (初值)   │ local_sum 的拷贝（permlane 的双源约束）                             │ ASM 1698                                      │
;   │ v223 (交换后) │ rhs_sum（cross-lane pair 中的另一 lane）                            │ ASM 1700                                      │
;   │ v224 (交换后) │ lhs_sum（本 lane 留存的 local_sum）                                 │ ASM 1700                                      │
;   │ v218          │ l_row（在线 softmax 的 denominator 累加器，全 kernel 范围循环载体） │ 多处更新；ASM 2022-2023 是 Cluster 5 这次更新 │
;   └───────────────┴─────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────┘

;   3. LLVM 做的代数重关联
;   重写前（IR 直翻为 ASM 应该是 2 次 add）：

;   tile_sum_b = lhs_sum + rhs_sum     ; add #1，写入新寄存器（比如 v_tmp）
;   l_row      = l_row + tile_sum_b    ; add #2，写入 v218

;   重写后（实际 ASM 里看到的）：

;   l_row = l_row + lhs_sum            ; add #1，直接写回 v218（ASM 2022）
;   l_row = l_row + rhs_sum            ; add #2，直接写回 v218（ASM 2023）

;   代数等价性：l_row + (lhs + rhs) = (l_row + lhs) + rhs（结合律重排，启用了 reassoc 后允许）。

;   ┌────────────────┬──────────────────────────────────┬─────────────────────────┐
;   │ 维度           │ 重写前                           │ 重写后                  │
;   ├────────────────┼──────────────────────────────────┼─────────────────────────┤
;   │ 加法次数       │ 2                                │ 2                       │
;   │ 关键路径深度   │ 2（lhs+rhs → 加 l_row）          │ 2（l_row+lhs → 加 rhs） │
;   │ 中间寄存器     │ 需要 1 个临时 VGPR 存 tile_sum_b │ 0，复用 v218            │
;   │ 寄存器压力影响 │ +1 VGPR 活跃区间                 │ 不增加                  │
;   └────────────────┴──────────────────────────────────┴─────────────────────────┘

;   LLVM 选择重写的动机：消除 tile_sum_b 的临时活跃区间，节省 1 个 VGPR。在 256 VGPR 紧张的内核里这是常见的 reassoc + 合并优化。
	v_add_f32_e32 v218, v218, v224
	v_add_f32_e32 v218, v218, v223

; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier

; Cluster 7:
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(1)
	s_setprio 1
; v_o = _mma1_step_k(0, v_p_1, v_v, v_o)
; m_tile_max_b = _attn_row_max(v_s_0)
; _sched_barrier_pairs(4, 5, 4)
; if const_expr(OPUS_LAZY_RESCALE):
; 	v_o, m_row, l_row, v_p_1 = _lazy_rescale_o(
; 		v_o, m_row, l_row, m_tile_max_b, v_p_1
; 	)
; v_o = _mma1_step_k(1, v_p_1, v_v, v_o)
; v_o = _mma1_step_k(2, v_p_1, v_v, v_o)
; v_o = _mma1_step_k(3, v_p_1, v_v, v_o)
; v_s_0 = _attn_sub_row(v_s_0, m_row)
; v_s_0 = _anchor_v_s(v_s_0)
; v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
; _sched_barrier_pairs(6, 5, 4)
; _sched_barrier_exp_pairs(6, 3, 4)
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(0)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[206:209], v[98:101], v[2:17]
; m_tile_max_b = _attn_row_max(v_s_0)
	v_max_f32_e32 v206, v66, v67
	v_max3_f32 v206, v206, v68, v69
	v_max3_f32 v206, v206, v70, v71
	v_max3_f32 v206, v206, v72, v73
	v_max3_f32 v206, v206, v74, v75
	v_mfma_f32_32x32x16_bf16 v[50:65], v[122:125], v[98:101], v[50:65]
	v_max3_f32 v122, v206, v76, v77
	v_max3_f32 v122, v122, v78, v79
	v_max3_f32 v122, v122, v80, v81
	v_max3_f32 v122, v122, v82, v83
	v_max3_f32 v122, v122, v84, v85
	v_mfma_f32_32x32x16_bf16 v[34:49], v[126:129], v[98:101], v[34:49]
	v_max3_f32 v122, v122, v86, v87
	v_max3_f32 v122, v122, v88, v89
	v_max3_f32 v122, v122, v90, v91
	v_max3_f32 v122, v122, v92, v93
	v_max3_f32 v122, v122, v94, v95
	v_mfma_f32_32x32x16_bf16 v[18:33], v[202:205], v[98:101], v[18:33]
	v_max3_f32 v98, v122, v96, v97
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e64 v98, v99 bound_ctrl:1
	v_max_f32_e32 v98, v98, v99
; v_o, m_row, l_row, v_p_1 = _lazy_rescale_o(
; 	v_o, m_row, l_row, m_tile_max_b, v_p_1
; )
; m_diff = _fsub(m_tile_max, m_row)
	v_sub_f32_e32 v99, v98, v220
; below = ArithValue(fx.Float32(m_diff) <= c_eight_f)
; ballot = rocdl.ballot(T.i64, _raw(below))
; all_below = arith.cmpi(
; 	arith.CmpIPredicate.eq,
; 	_raw(ballot),
; 	_read_exec_i64(),
; )
	v_cmp_ge_f32_e32 vcc, s36, v99
	s_cmp_eq_u64 vcc, exec
	s_cselect_b64 vcc, -1, 0
	s_cbranch_vccnz .LBB0_9
; _lazy_rescale_o else-branch: scale old O accumulators by corr.
; corr = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_tile_max)))
	v_sub_f32_e32 v99, v220, v98
	v_exp_f32_e32 v100, v99
	s_nop 0
;   scaled_accs = list(v_o)
;   _scale_o(scaled_accs, corr)
	v_pk_mul_f32 v[16:17], v[100:101], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[100:101], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[100:101], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[100:101], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[100:101], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[100:101], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[100:101], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[100:101], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[100:101], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[100:101], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[100:101], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[100:101], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[100:101], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[100:101], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[100:101], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[100:101], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[100:101], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[100:101], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[100:101], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[100:101], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[100:101], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[100:101], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[100:101], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[100:101], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[100:101], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[100:101], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[100:101], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[100:101], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[100:101], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[100:101], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[100:101], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[100:101], v[18:19] op_sel_hi:[0,1]
;   scaled_v_p = _scale_v_p(v_p, corr)
;   p_all = _v_p_to_vec32(v_p)
;   p_all_f32 = FPExt(p_all)          # bf16 -> f32
;   p_scaled_f32 = corr * p_all_f32
;   p_scaled_bf16 = FPTrunc(p_scaled_f32)
;   return _v_vec32_to_p(p_scaled_bf16)
	v_and_b32_e32 v123, 0xffff0000, v113
	v_lshlrev_b32_e32 v122, 16, v113
	v_and_b32_e32 v113, 0xffff0000, v112
	v_lshlrev_b32_e32 v112, 16, v112
	v_and_b32_e32 v125, 0xffff0000, v111
	v_lshlrev_b32_e32 v124, 16, v111
	v_and_b32_e32 v111, 0xffff0000, v110
	v_lshlrev_b32_e32 v110, 16, v110
	v_and_b32_e32 v127, 0xffff0000, v109
	v_lshlrev_b32_e32 v126, 16, v109
	v_and_b32_e32 v109, 0xffff0000, v108
	v_lshlrev_b32_e32 v108, 16, v108
	v_and_b32_e32 v129, 0xffff0000, v107
	v_lshlrev_b32_e32 v128, 16, v107
	v_and_b32_e32 v107, 0xffff0000, v106
	v_lshlrev_b32_e32 v106, 16, v106
	v_and_b32_e32 v203, 0xffff0000, v105
	v_lshlrev_b32_e32 v202, 16, v105
	v_and_b32_e32 v105, 0xffff0000, v104
	v_lshlrev_b32_e32 v104, 16, v104
	v_and_b32_e32 v205, 0xffff0000, v103
	v_lshlrev_b32_e32 v204, 16, v103
	v_and_b32_e32 v103, 0xffff0000, v102
	v_lshlrev_b32_e32 v102, 16, v102
	v_pk_mul_f32 v[206:207], v[100:101], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[100:101], v[204:205] op_sel_hi:[0,1]
	v_pk_mul_f32 v[204:205], v[100:101], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[100:101], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203], v[100:101], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[100:101], v[128:129] op_sel_hi:[0,1]
	v_pk_mul_f32 v[128:129], v[100:101], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[100:101], v[126:127] op_sel_hi:[0,1]
	v_pk_mul_f32 v[126:127], v[100:101], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[100:101], v[124:125] op_sel_hi:[0,1]
	v_pk_mul_f32 v[124:125], v[100:101], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[100:101], v[122:123] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v113, v112, v113
	v_cvt_pk_bf16_f32 v112, v124, v125
	v_cvt_pk_bf16_f32 v111, v110, v111
	v_cvt_pk_bf16_f32 v110, v126, v127
	v_cvt_pk_bf16_f32 v109, v108, v109
	v_cvt_pk_bf16_f32 v108, v128, v129
	v_cvt_pk_bf16_f32 v107, v106, v107
	v_cvt_pk_bf16_f32 v106, v202, v203
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v204, v205
	v_cvt_pk_bf16_f32 v103, v102, v103
	v_cvt_pk_bf16_f32 v102, v206, v207
; scaled_l_row = _fmul(l_row, corr)
	v_mul_f32_e32 v218, v100, v218
	s_branch .LBB0_9

.LBB0_16:
; Epilogue
; Cluster 0:
; m_row = loop_results[0]
; l_row = loop_results[1]
; v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]
; v_p_0 = _v_vec32_to_pair(loop_results[2 + D_CHUNKS])
; max_m3 = max_num_tiles - fx.Index(3)
; max_m2 = max_num_tiles - fx.Index(2)
; max_m1 = max_num_tiles - fx.Index(1)
; _async_load_v(max_m3 * fx.Index(BLOCK_N), 1)
; v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   Control reaches `.LBB0_16` either from the loop back-edge at ASM 1130
;   (`s_cbranch_vccz .LBB0_16` when j >= max_num_tiles - 1) or from the
;   zero-trip-count guard at ASM 846 (when max_num_tiles <= 4). In the
;   zero-trip case loop_results[*] are the initial values from PART 3 of
;   the preheader (v218=0, v220=m_row_pro, v2..v65=0); in the normal case
;   they are the SSA yields from the last loop iteration.
;   `s_mov_b32 m0, s19` selects the LDS write-base for the V[max_m3]
;   async prefetch (V buf-1 LDS base set in prologue at ASM 768).
;   Two `buffer_load_dwordx4 v215/v216, s[0:3], s6 offen lds` issue the
;   async global-to-LDS V[max_m3] load. The 16 `ds_read_b128 v[X:X+3],
;   v211 offset:Y` read K from LDS buf-1 (offsets 34048..34656) and V from
;   LDS buf-0 (offsets 42368..42976) into the v_k registers used by
;   Cluster 1's MMA0.
	s_mov_b32 m0, s19
	s_lshl_b64 s[12:13], s[8:9], 6
	s_add_u32 s13, s12, 0xffffff40
	s_mul_i32 s6, s13, s24
	buffer_load_dwordx4 v215, s[0:3], s6 offen lds
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v216, s[0:3], s6 offen lds
	ds_read_b128 v[98:101], v211 offset:34048
	ds_read_b128 v[126:129], v211 offset:34080
	ds_read_b128 v[164:167], v211 offset:34560
	ds_read_b128 v[168:171], v211 offset:34592
	ds_read_b128 v[172:175], v211 offset:34112
	ds_read_b128 v[186:189], v211 offset:34144
	ds_read_b128 v[190:193], v211 offset:34624
	ds_read_b128 v[194:197], v211 offset:34656
	ds_read_b128 v[198:201], v211 offset:42368
	ds_read_b128 v[202:205], v211 offset:42400
	ds_read_b128 v[206:209], v211 offset:42880
	ds_read_b128 v[222:225], v211 offset:42912
	ds_read_b128 v[226:229], v211 offset:42432
	ds_read_b128 v[230:233], v211 offset:42464
	ds_read_b128 v[234:237], v211 offset:42944
	ds_read_b128 v[238:241], v211 offset:42976
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
; Cluster 1:
; v_s_1 = _mma0(v_k)
; v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
; tile_sum_e1 = _attn_sum(v_p_0)
; l_row = _fadd(l_row, tile_sum_e1)
; v_p_0 = _cast_p(v_p_0)
; v_p_0 = _anchor_v_p(v_p_0)
; _sched_barrier_exp_pairs(6, 3, 5)
; _sched_barrier_pairs(10, 5, 5)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   First epilogue cluster after the steady-state loop exits. Reuses the
;   v_p_0 yielded from the loop's last iteration to (a) finish the second
;   half of its `_attn_exp2_slice(v_p_0, 16, 16)` via 16 `v_exp_f32_e32`
;   instructions interleaved with the MMA0 chain, (b) reduce the
;   32-element local sum via the `v_add_f32_e32 v66, v183, v184` chain
;   ending at `v_add_f32_e32 v221, v66, v81`, (c) cross-lane swap via
;   `v_permlane32_swap_b32_e64 v221, v222 bound_ctrl:1` so v221/v222 hold
;   the two halves of the warp-level tile_sum_e1, and (d) cast P back to
;   bf16 via the 16 `v_cvt_pk_bf16_f32 v[66..81], ...` block (wrapped in
;   `;;#ASMSTART/;;#ASMEND` to keep MachineCSE from merging them).
;   The l_row update is deferred: v221/v222 are kept live in their own
;   VGPRs and folded into v218 only by the final reduction chain at
;   ASM ~4274-4275 (`v_add_f32_e32 v66, v218, v222`,
;   `v_add_f32_e32 v66, v66, v221`).
	v_mfma_f32_32x32x16_bf16 v[82:97], v[98:101], v[130:133], 0
	v_exp_f32_e32 v0, v66
	v_exp_f32_e32 v122, v67
	v_exp_f32_e32 v123, v68
	v_mfma_f32_32x32x16_bf16 v[98:113], v[164:167], v[130:133], 0
	v_exp_f32_e32 v124, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mfma_f32_32x32x16_bf16 v[82:97], v[126:129], v[134:137], v[82:97]
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v125, v74
	v_mfma_f32_32x32x16_bf16 v[98:113], v[168:171], v[134:137], v[98:113]
	v_exp_f32_e32 v126, v75
	v_exp_f32_e32 v127, v76
	v_exp_f32_e32 v128, v77
	v_mfma_f32_32x32x16_bf16 v[82:97], v[172:175], v[138:141], v[82:97]
	v_exp_f32_e32 v129, v78
	v_exp_f32_e32 v164, v79
	v_exp_f32_e32 v165, v80
	v_mfma_f32_32x32x16_bf16 v[98:113], v[190:193], v[138:141], v[98:113]
	v_exp_f32_e32 v81, v81
	v_mfma_f32_32x32x16_bf16 v[82:97], v[186:189], v[142:145], v[82:97]
	v_add_f32_e32 v66, v183, v184
	v_add_f32_e32 v66, v66, v182
	v_add_f32_e32 v66, v66, v180
	v_add_f32_e32 v66, v66, v178
	v_add_f32_e32 v66, v66, v179
	v_mfma_f32_32x32x16_bf16 v[98:113], v[194:197], v[142:145], v[98:113]
	v_add_f32_e32 v66, v66, v162
	v_add_f32_e32 v66, v66, v163
	v_add_f32_e32 v66, v66, v119
	v_add_f32_e32 v66, v66, v121
	v_add_f32_e32 v66, v66, v117
	v_mfma_f32_32x32x16_bf16 v[82:97], v[198:201], v[146:149], v[82:97]
	v_add_f32_e32 v66, v66, v120
	v_add_f32_e32 v66, v66, v116
	v_add_f32_e32 v66, v66, v118
	v_add_f32_e32 v66, v66, v114
	v_add_f32_e32 v66, v66, v115
	v_mfma_f32_32x32x16_bf16 v[98:113], v[206:209], v[146:149], v[98:113]
	v_add_f32_e32 v66, v66, v0
	v_add_f32_e32 v66, v66, v122
	v_add_f32_e32 v66, v66, v123
	v_add_f32_e32 v66, v66, v124
	v_add_f32_e32 v66, v66, v70
	v_mfma_f32_32x32x16_bf16 v[82:97], v[202:205], v[150:153], v[82:97]
	v_add_f32_e32 v66, v66, v71
	v_add_f32_e32 v66, v66, v72
	v_add_f32_e32 v66, v66, v73
	v_add_f32_e32 v66, v66, v125
	v_add_f32_e32 v66, v66, v126
	v_mfma_f32_32x32x16_bf16 v[98:113], v[222:225], v[150:153], v[98:113]
	v_add_f32_e32 v66, v66, v127
	v_add_f32_e32 v66, v66, v128
	v_add_f32_e32 v66, v66, v129
	v_add_f32_e32 v66, v66, v164
	v_add_f32_e32 v66, v66, v165
	v_mfma_f32_32x32x16_bf16 v[82:97], v[226:229], v[154:157], v[82:97]
	v_add_f32_e32 v221, v66, v81
	v_mov_b32_e32 v222, v221
	s_nop 1
	v_permlane32_swap_b32_e64 v221, v222 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v183, v184
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v182, v180
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v178, v179
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[98:113], v[234:237], v[154:157], v[98:113]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v0, v122
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v123, v124
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v72, v73
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[230:233], v[158:161], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v119, v121
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v117, v120
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v116, v118
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v114, v115
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v125, v126
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[98:113], v[238:241], v[158:161], v[98:113]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v127, v128
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v80, v129, v164
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v81, v165, v81
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_barrier
; Cluster 2:
; _async_load_k(max_m1 * fx.Index(BLOCK_N), 1)
; v_packs_e3 = _read_v_packs_for_buf(0, urv_base_per_lane)
; if const_expr(CAUSAL):
; 	v_s_1 = _causal_mask_prologue_if_needed(
; 		v_s_1,
; 		max_m3,
; 		max_m2 * fx.Index(BLOCK_N),
; 	)
; else:
; 	v_s_1 = _v_s_vec_to_lists(v_s_1)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   `s_mov_b32 m0, s30` selects the LDS write-base for the K[max_m1] async
;   prefetch (K buf-1 LDS base, set in prologue at ASM 137). Two
;   `buffer_load_dwordx4 v215/v216, s[4:7], s9 offen lds` issue the async
;   global-to-LDS K[max_num_tiles-1] load.
;   The 32 `ds_read_b64_tr_b16 v[X:X+1], v217 offset:Y` block (offsets
;   0..960 + 8704..9664) reads the transposed V[max_m3] tile from LDS
;   buf-0 into v_packs_e3 (used by Cluster 3's MMA1).
;   The trailing `s_add_u32 s4, s12, 0xffffff80` + `s_cmp_ge_i32 s16, s4`
;   compute `q_start_pos >= max_m2 * BLOCK_N`; the next
;   `;;#ASMSTART/.../;;#ASMEND` inline-asm block performs
;   `attn_mask_causal_tile(v_s_1)` (a chain of `v_cmp_lt_i32 + v_cndmask`
;   writes NEG_INF=`0xff800000` into v224 then onto v82..v97/v98..v113);
;   `s_cbranch_scc1 .LBB0_18` skips the mask body when out of range.
	s_mov_b32 m0, s30
	s_lshl_b64 s[10:11], s[10:11], 6
	s_mul_i32 s9, s10, s24
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v215, s[4:7], s9 offen lds
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v216, s[4:7], s9 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v217 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v217 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v217 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v217 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v217 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v217 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v217 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v217 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v217 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v217 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v217 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v217 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v217 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v217 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v217 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v217 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v217 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v217 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v217 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v217 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v217 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v217 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v217 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v217 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v217 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v217 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v217 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v217 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v217 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v217 offset:9536

	;;#ASMEND
	s_add_u32 s4, s12, 0xffffff80
	s_cmp_ge_i32 s16, s4
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v217 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_18
	v_lshl_or_b32 v0, v214, 2, s13
	v_sub_u32_e32 v0, v212, v0
	v_subrev_u32_e32 v223, 32, v0
	v_mov_b32_e32 v224, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 0
	v_cmp_lt_i32_e64 s[12:13], v0, 1
	v_cndmask_b32_e64 v82, v82, v224, s[6:7]
	v_cndmask_b32_e64 v83, v83, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 2
	v_cmp_lt_i32_e64 s[12:13], v0, 3
	v_cndmask_b32_e64 v84, v84, v224, s[6:7]
	v_cndmask_b32_e64 v85, v85, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 8
	v_cmp_lt_i32_e64 s[12:13], v0, 9
	v_cndmask_b32_e64 v86, v86, v224, s[6:7]
	v_cndmask_b32_e64 v87, v87, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 10
	v_cmp_lt_i32_e64 s[12:13], v0, 11
	v_cndmask_b32_e64 v88, v88, v224, s[6:7]
	v_cndmask_b32_e64 v89, v89, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 16
	v_cmp_lt_i32_e64 s[12:13], v0, 17
	v_cndmask_b32_e64 v90, v90, v224, s[6:7]
	v_cndmask_b32_e64 v91, v91, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 18
	v_cmp_lt_i32_e64 s[12:13], v0, 19
	v_cndmask_b32_e64 v92, v92, v224, s[6:7]
	v_cndmask_b32_e64 v93, v93, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 24
	v_cmp_lt_i32_e64 s[12:13], v0, 25
	v_cndmask_b32_e64 v94, v94, v224, s[6:7]
	v_cndmask_b32_e64 v95, v95, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v0, 26
	v_cmp_lt_i32_e64 s[12:13], v0, 27
	v_cndmask_b32_e64 v96, v96, v224, s[6:7]
	v_cndmask_b32_e64 v97, v97, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v0, v99
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 0
	v_cmp_lt_i32_e64 s[12:13], v223, 1
	v_cndmask_b32_e64 v98, v98, v224, s[6:7]
	v_cndmask_b32_e64 v0, v0, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v99, v101
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 2
	v_cmp_lt_i32_e64 s[12:13], v223, 3
	v_cndmask_b32_e64 v100, v100, v224, s[6:7]
	v_cndmask_b32_e64 v99, v99, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v101, v103
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 8
	v_cmp_lt_i32_e64 s[12:13], v223, 9
	v_cndmask_b32_e64 v102, v102, v224, s[6:7]
	v_cndmask_b32_e64 v101, v101, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v103, v105
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 10
	v_cmp_lt_i32_e64 s[12:13], v223, 11
	v_cndmask_b32_e64 v104, v104, v224, s[6:7]
	v_cndmask_b32_e64 v103, v103, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v105, v107
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 16
	v_cmp_lt_i32_e64 s[12:13], v223, 17
	v_cndmask_b32_e64 v106, v106, v224, s[6:7]
	v_cndmask_b32_e64 v105, v105, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v107, v109
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 18
	v_cmp_lt_i32_e64 s[12:13], v223, 19
	v_cndmask_b32_e64 v108, v108, v224, s[6:7]
	v_cndmask_b32_e64 v107, v107, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v109, v111
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 24
	v_cmp_lt_i32_e64 s[12:13], v223, 25
	v_cndmask_b32_e64 v110, v110, v224, s[6:7]
	v_cndmask_b32_e64 v109, v109, v224, s[12:13]
	;;#ASMEND
	v_mov_b32_e32 v111, v113
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[6:7], v223, 26
	v_cmp_lt_i32_e64 s[12:13], v223, 27
	v_cndmask_b32_e64 v112, v112, v224, s[6:7]
	v_cndmask_b32_e64 v111, v111, v224, s[12:13]
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v113, v111
	v_mov_b32_e32 v111, v109
	v_mov_b32_e32 v109, v107
	v_mov_b32_e32 v107, v105
	v_mov_b32_e32 v105, v103
	v_mov_b32_e32 v103, v101
	v_mov_b32_e32 v101, v99
	v_mov_b32_e32 v99, v0
.LBB0_18:
; Cluster 2 causal-mask skip target.
; if const_expr(CAUSAL):
; 	v_s_1 = _causal_mask_prologue_if_needed(
; 		v_s_1,
; 		max_m3,
; 		max_m2 * fx.Index(BLOCK_N),
; 	)
; NOTE:
;   `.LBB0_18` is reached when `q_start_pos >= max_m2 * BLOCK_N` (i.e. this
;   wave is past the causal boundary for the second-to-last KV tile in the
;   epilogue), so the mask body is skipped. Both paths converge here.
;   The next `s_waitcnt vmcnt(4) lgkmcnt(0)` + `s_barrier` is Cluster 2's
;   closing synchronization fence (PY L1746 `rocdl.s_barrier()`) — it
;   waits for the K[max_m1] async load and the LDS V reads to finish
;   before Cluster 3 begins.
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
; Cluster 3:
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(1)
; v_o = _mma1(v_p_0, v_packs_e3, v_o)
; m_tile_max_e3 = _attn_row_max(v_s_1)
; row_max_e3 = _fmax(m_row, m_tile_max_e3)
; rescale_e3 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e3)))
; m_row = row_max_e3
; v_s_1 = _attn_sub_row(v_s_1, row_max_e3)
; v_s_1 = _anchor_v_s(v_s_1)
; v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
; _sched_barrier_pairs(10, 5, 6)
; _sched_barrier_exp_pairs(6, 3, 6)
; rocdl.sched_barrier(0)
; _scale_o(v_o, rescale_e3)
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(0)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   `s_setprio 1` raises priority of the dual-group time-multiplexing
;   handoff so the slowest wave reaches the barrier first.
;   MMA1 chain (16 `v_mfma_f32_32x32x16_bf16 v[X:Y], v[A:B], v[66:69]/...`)
;   accumulates `v_p_0 * v_packs_e3` into v_o (banks v[2:17], v[50:65],
;   v[34:49], v[18:33]). Interleaved with: (a) `_attn_row_max(v_s_1)` —
;   the `v_max_f32_e32 v0, v82, v83` + 15× `v_max3_f32 v0, v0, ...`
;   reduction + `v_permlane32_swap_b32_e64 v0, v66 bound_ctrl:1` +
;   `v_max3_f32 v225, v220, v0, v66` (final fmax with m_row); (b)
;   `v_sub_f32_e32 v97, v97, v225` and following — the 32 sub_row
;   subtractions; (c) `v_exp_f32_e32 v0, v82` and following — the first
;   half of `_attn_exp2_slice(v_s_1, 0, 16)` (16 v_exp).
;   `s_setprio 0` + `s_barrier` close the cluster. The
;   `_scale_o(v_o, rescale_e3)` step is HOISTED out of this cluster by
;   LLVM and emitted at ASM ~3343-3377 (right before Cluster 7's MMA1),
;   so the v_o accumulators are scaled exactly at the point they would
;   otherwise be re-read.
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[2:17], v[194:197], v[66:69], v[2:17]
	v_max_f32_e32 v0, v82, v83
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_mfma_f32_32x32x16_bf16 v[50:65], v[198:201], v[66:69], v[50:65]
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	v_mfma_f32_32x32x16_bf16 v[34:49], v[202:205], v[66:69], v[34:49]
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_mfma_f32_32x32x16_bf16 v[18:33], v[206:209], v[66:69], v[18:33]
	v_max3_f32 v0, v0, v112, v113
	v_mov_b32_e32 v66, v0
	s_nop 1
	v_permlane32_swap_b32_e64 v0, v66 bound_ctrl:1
	v_max3_f32 v225, v220, v0, v66
	v_sub_f32_e32 v97, v97, v225
	v_sub_f32_e32 v96, v96, v225
	v_mfma_f32_32x32x16_bf16 v[2:17], v[178:181], v[70:73], v[2:17]
	v_sub_f32_e32 v95, v95, v225
	v_sub_f32_e32 v94, v94, v225
	v_sub_f32_e32 v93, v93, v225
	v_sub_f32_e32 v92, v92, v225
	v_sub_f32_e32 v91, v91, v225
	v_mfma_f32_32x32x16_bf16 v[50:65], v[182:185], v[70:73], v[50:65]
	v_sub_f32_e32 v90, v90, v225
	v_sub_f32_e32 v89, v89, v225
	v_sub_f32_e32 v88, v88, v225
	v_sub_f32_e32 v87, v87, v225
	v_sub_f32_e32 v86, v86, v225
	v_mfma_f32_32x32x16_bf16 v[34:49], v[186:189], v[70:73], v[34:49]
	v_sub_f32_e32 v85, v85, v225
	v_sub_f32_e32 v84, v84, v225
	v_sub_f32_e32 v83, v83, v225
	v_sub_f32_e32 v82, v82, v225
	v_sub_f32_e32 v113, v113, v225
	v_mfma_f32_32x32x16_bf16 v[18:33], v[190:193], v[70:73], v[18:33]
	v_sub_f32_e32 v112, v112, v225
	v_sub_f32_e32 v111, v111, v225
	v_sub_f32_e32 v110, v110, v225
	v_sub_f32_e32 v109, v109, v225
	v_sub_f32_e32 v108, v108, v225
	v_mfma_f32_32x32x16_bf16 v[2:17], v[166:169], v[74:77], v[2:17]
	v_sub_f32_e32 v107, v107, v225
	v_sub_f32_e32 v106, v106, v225
	v_sub_f32_e32 v105, v105, v225
	v_sub_f32_e32 v104, v104, v225
	v_sub_f32_e32 v103, v103, v225
	v_mfma_f32_32x32x16_bf16 v[50:65], v[170:173], v[74:77], v[50:65]
	v_sub_f32_e32 v102, v102, v225
	v_sub_f32_e32 v101, v101, v225
	v_sub_f32_e32 v100, v100, v225
	v_sub_f32_e32 v99, v99, v225
	v_sub_f32_e32 v98, v98, v225
	v_mfma_f32_32x32x16_bf16 v[34:49], v[174:177], v[74:77], v[34:49]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_exp_f32_e32 v0, v82
	v_exp_f32_e32 v166, v83
	v_exp_f32_e32 v167, v84
	v_mfma_f32_32x32x16_bf16 v[18:33], v[162:165], v[74:77], v[18:33]
	v_exp_f32_e32 v162, v85
	v_exp_f32_e32 v163, v86
	v_exp_f32_e32 v164, v87
	v_mfma_f32_32x32x16_bf16 v[2:17], v[114:117], v[78:81], v[2:17]
	v_exp_f32_e32 v114, v88
	v_exp_f32_e32 v115, v89
	v_exp_f32_e32 v116, v90
	v_mfma_f32_32x32x16_bf16 v[50:65], v[118:121], v[78:81], v[50:65]
	v_exp_f32_e32 v117, v91
	v_exp_f32_e32 v118, v92
	v_exp_f32_e32 v119, v93
	v_mfma_f32_32x32x16_bf16 v[34:49], v[122:125], v[78:81], v[34:49]
	v_exp_f32_e32 v120, v94
	v_exp_f32_e32 v121, v95
	v_exp_f32_e32 v122, v96
	v_mfma_f32_32x32x16_bf16 v[18:33], v[126:129], v[78:81], v[18:33]
	v_exp_f32_e32 v123, v97
	s_setprio 0
	s_barrier
; Cluster 4:
; _async_load_v(max_m2 * fx.Index(BLOCK_N), 0)
; v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   `s_mov_b32 m0, s28` selects the LDS write-base for the V[max_m2] async
;   prefetch (V buf-0 LDS base set in prologue at ASM 139). The two
;   `buffer_load_dwordx4 v215/v216, s[0:3], s5 offen lds` issue the async
;   global-to-LDS V load. The 16 `ds_read_b128 v[82:85]/.../v[242:245],
;   v211 offset:Y` read K from LDS buf-0 (offsets 0..608) and the
;   accompanying V slices (offsets 8320..8928) into v_k for Cluster 5's
;   MMA0.
	s_mov_b32 m0, s28
	s_mul_i32 s5, s4, s24
	buffer_load_dwordx4 v215, s[0:3], s5 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v216, s[0:3], s5 offen lds
	ds_read_b128 v[82:85], v211
	ds_read_b128 v[168:171], v211 offset:32
	ds_read_b128 v[172:175], v211 offset:512
	ds_read_b128 v[176:179], v211 offset:544
	ds_read_b128 v[180:183], v211 offset:64
	ds_read_b128 v[184:187], v211 offset:96
	ds_read_b128 v[188:191], v211 offset:576
	ds_read_b128 v[192:195], v211 offset:608
	ds_read_b128 v[196:199], v211 offset:8320
	ds_read_b128 v[200:203], v211 offset:8352
	ds_read_b128 v[204:207], v211 offset:8832
	ds_read_b128 v[226:229], v211 offset:8864
	ds_read_b128 v[230:233], v211 offset:8384
	ds_read_b128 v[234:237], v211 offset:8416
	ds_read_b128 v[238:241], v211 offset:8896
	ds_read_b128 v[242:245], v211 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
; Cluster 5:
; v_s_0 = _mma0(v_k)
; l_row = _fmul(l_row, rescale_e3)
; v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
; tile_sum_e5 = _attn_sum(v_p_1)
; l_row = _fadd(l_row, tile_sum_e5)
; v_p_1 = _cast_p(v_p_1)
; v_p_1 = _anchor_v_p(v_p_1)
; _sched_barrier_exp_pairs(6, 3, 7)
; _sched_barrier_pairs(10, 5, 7)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   16 MMA0 instructions on alternating bank pairs `v[66:81]` /
;   `v[82:97]`, interleaved with 16 `v_exp_f32_e32` (second half of v_p_1's
;   exp2_slice covering elements 16..31) and 32 `v_add_f32_e32 v98, ...`
;   accumulations (`_attn_sum(v_p_1)` running sum). The final
;   `v_add_f32_e32 v223, v98, v113` + `v_mov_b32_e32 v224, v223` +
;   `v_permlane32_swap_b32_e64 v223, v224 bound_ctrl:1` produces the two
;   warp-level tile_sum_e5 halves kept in v223/v224.
;   The `l_row = _fmul(l_row, rescale_e3)` and the `l_row += tile_sum_e5`
;   ops are NOT emitted here — LLVM defers them, expressing them at the
;   normalize-O reduction (ASM ~4276 `v_fmac_f32_e32 v223, v0, v66`).
;   The 16 `v_cvt_pk_bf16_f32 v[98..113], ...` block then casts v_p_1
;   back to bf16 in place.
	v_mfma_f32_32x32x16_bf16 v[66:81], v[82:85], v[130:133], 0
	v_exp_f32_e32 v124, v98
	v_exp_f32_e32 v125, v99
	v_exp_f32_e32 v126, v100
	v_mfma_f32_32x32x16_bf16 v[82:97], v[172:175], v[130:133], 0
	v_exp_f32_e32 v127, v101
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_mfma_f32_32x32x16_bf16 v[66:81], v[168:171], v[134:137], v[66:81]
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v128, v106
	v_mfma_f32_32x32x16_bf16 v[82:97], v[176:179], v[134:137], v[82:97]
	v_exp_f32_e32 v129, v107
	v_exp_f32_e32 v165, v108
	v_exp_f32_e32 v168, v109
	v_mfma_f32_32x32x16_bf16 v[66:81], v[180:183], v[138:141], v[66:81]
	v_exp_f32_e32 v169, v110
	v_exp_f32_e32 v170, v111
	v_exp_f32_e32 v171, v112
	v_mfma_f32_32x32x16_bf16 v[82:97], v[188:191], v[138:141], v[82:97]
	v_exp_f32_e32 v113, v113
	v_mfma_f32_32x32x16_bf16 v[66:81], v[184:187], v[142:145], v[66:81]
	v_add_f32_e32 v98, v0, v166
	v_add_f32_e32 v98, v98, v167
	v_add_f32_e32 v98, v98, v162
	v_add_f32_e32 v98, v98, v163
	v_add_f32_e32 v98, v98, v164
	v_mfma_f32_32x32x16_bf16 v[82:97], v[192:195], v[142:145], v[82:97]
	v_add_f32_e32 v98, v98, v114
	v_add_f32_e32 v98, v98, v115
	v_add_f32_e32 v98, v98, v116
	v_add_f32_e32 v98, v98, v117
	v_add_f32_e32 v98, v98, v118
	v_mfma_f32_32x32x16_bf16 v[66:81], v[196:199], v[146:149], v[66:81]
	v_add_f32_e32 v98, v98, v119
	v_add_f32_e32 v98, v98, v120
	v_add_f32_e32 v98, v98, v121
	v_add_f32_e32 v98, v98, v122
	v_add_f32_e32 v98, v98, v123
	v_mfma_f32_32x32x16_bf16 v[82:97], v[204:207], v[146:149], v[82:97]
	v_add_f32_e32 v98, v98, v124
	v_add_f32_e32 v98, v98, v125
	v_add_f32_e32 v98, v98, v126
	v_add_f32_e32 v98, v98, v127
	v_add_f32_e32 v98, v98, v102
	v_mfma_f32_32x32x16_bf16 v[66:81], v[200:203], v[150:153], v[66:81]
	v_add_f32_e32 v98, v98, v103
	v_add_f32_e32 v98, v98, v104
	v_add_f32_e32 v98, v98, v105
	v_add_f32_e32 v98, v98, v128
	v_add_f32_e32 v98, v98, v129
	v_mfma_f32_32x32x16_bf16 v[82:97], v[226:229], v[150:153], v[82:97]
	v_add_f32_e32 v98, v98, v165
	v_add_f32_e32 v98, v98, v168
	v_add_f32_e32 v98, v98, v169
	v_add_f32_e32 v98, v98, v170
	v_add_f32_e32 v98, v98, v171
	v_mfma_f32_32x32x16_bf16 v[66:81], v[230:233], v[154:157], v[66:81]
	v_add_f32_e32 v223, v98, v113
	v_mov_b32_e32 v224, v223
	s_nop 1
	v_permlane32_swap_b32_e64 v224, v223 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v0, v166
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v167, v162
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v163, v164
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[238:241], v[154:157], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v114, v115
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v124, v125
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v126, v127
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v102, v103
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v104, v105
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[66:81], v[234:237], v[158:161], v[66:81]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v116, v117
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v118, v119
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v120, v121
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v122, v123
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v128, v129
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[242:245], v[158:161], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v165, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v169, v170
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v171, v113
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_barrier
; Cluster 6:
; v_packs_e7 = _read_v_packs_for_buf(1, urv_base_per_lane)
; if const_expr(CAUSAL):
; 	v_s_0 = _causal_mask_prologue_if_needed(
; 		v_s_0,
; 		max_m2,
; 		max_m1 * fx.Index(BLOCK_N),
; 	)
; else:
; 	v_s_0 = _v_s_vec_to_lists(v_s_0)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   `v_add_u32_e32 v219, 0xc600, v219` rebases the v219 lane offset by
;   0xc600 to point at LDS buf-1 V slots (v219 was set up in the prologue
;   preheader at ASM 734, then advanced per-iteration via
;   `v_add_u32_e32 v0, 0xffffff80, v0` at ASM 1126; here we reuse it to
;   index V buf-1 instead of buf-0).
;   The 32 `ds_read_b64_tr_b16 v[X:X+1], v219 offset:Y` block reads the
;   transposed V[max_m1] tile from LDS buf-1 into v_packs_e7 (used by
;   Cluster 7's MMA1).
;   The trailing `s_add_u32 ...` + `s_cmp_ge_i32 s16, ...` compute
;   `q_start_pos >= max_m1 * BLOCK_N`; the `;;#ASMSTART/.../;;#ASMEND`
;   inline-asm block writes NEG_INF into out-of-range lanes via
;   `v_cmp_lt_i32 + v_cndmask` chains; `s_cbranch_scc1 .LBB0_20` skips
;   the mask body when this wave is past the causal boundary. Both paths
;   converge at .LBB0_20 below.
	v_add_u32_e32 v219, 0xc600, v219
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v219 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v219 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v219 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v219 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v219 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v219 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v219 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v219 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v219 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v219 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v219 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v219 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v219 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v219 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v219 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v219 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v219 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v219 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v219 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v219 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v219 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v219 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v219 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v219 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v219 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v219 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v219 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v219 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v219 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v219 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v219 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s16, s10
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v219 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_20
	v_lshl_or_b32 v0, v214, 2, s4
	v_sub_u32_e32 v0, v212, v0
	v_subrev_u32_e32 v226, 32, v0
	v_mov_b32_e32 v227, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 0
	v_cmp_lt_i32_e64 s[6:7], v0, 1
	v_cndmask_b32_e64 v66, v66, v227, s[4:5]
	v_cndmask_b32_e64 v67, v67, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 2
	v_cmp_lt_i32_e64 s[6:7], v0, 3
	v_cndmask_b32_e64 v68, v68, v227, s[4:5]
	v_cndmask_b32_e64 v69, v69, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 8
	v_cmp_lt_i32_e64 s[6:7], v0, 9
	v_cndmask_b32_e64 v70, v70, v227, s[4:5]
	v_cndmask_b32_e64 v71, v71, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 10
	v_cmp_lt_i32_e64 s[6:7], v0, 11
	v_cndmask_b32_e64 v72, v72, v227, s[4:5]
	v_cndmask_b32_e64 v73, v73, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 16
	v_cmp_lt_i32_e64 s[6:7], v0, 17
	v_cndmask_b32_e64 v74, v74, v227, s[4:5]
	v_cndmask_b32_e64 v75, v75, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 18
	v_cmp_lt_i32_e64 s[6:7], v0, 19
	v_cndmask_b32_e64 v76, v76, v227, s[4:5]
	v_cndmask_b32_e64 v77, v77, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 24
	v_cmp_lt_i32_e64 s[6:7], v0, 25
	v_cndmask_b32_e64 v78, v78, v227, s[4:5]
	v_cndmask_b32_e64 v79, v79, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v0, 26
	v_cmp_lt_i32_e64 s[6:7], v0, 27
	v_cndmask_b32_e64 v80, v80, v227, s[4:5]
	v_cndmask_b32_e64 v81, v81, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v0, v83
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 0
	v_cmp_lt_i32_e64 s[6:7], v226, 1
	v_cndmask_b32_e64 v82, v82, v227, s[4:5]
	v_cndmask_b32_e64 v0, v0, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v83, v85
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 2
	v_cmp_lt_i32_e64 s[6:7], v226, 3
	v_cndmask_b32_e64 v84, v84, v227, s[4:5]
	v_cndmask_b32_e64 v83, v83, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v85, v87
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 8
	v_cmp_lt_i32_e64 s[6:7], v226, 9
	v_cndmask_b32_e64 v86, v86, v227, s[4:5]
	v_cndmask_b32_e64 v85, v85, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v87, v89
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 10
	v_cmp_lt_i32_e64 s[6:7], v226, 11
	v_cndmask_b32_e64 v88, v88, v227, s[4:5]
	v_cndmask_b32_e64 v87, v87, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v89, v91
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 16
	v_cmp_lt_i32_e64 s[6:7], v226, 17
	v_cndmask_b32_e64 v90, v90, v227, s[4:5]
	v_cndmask_b32_e64 v89, v89, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v91, v93
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 18
	v_cmp_lt_i32_e64 s[6:7], v226, 19
	v_cndmask_b32_e64 v92, v92, v227, s[4:5]
	v_cndmask_b32_e64 v91, v91, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v93, v95
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 24
	v_cmp_lt_i32_e64 s[6:7], v226, 25
	v_cndmask_b32_e64 v94, v94, v227, s[4:5]
	v_cndmask_b32_e64 v93, v93, v227, s[6:7]
	;;#ASMEND
	v_mov_b32_e32 v95, v97
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v226, 26
	v_cmp_lt_i32_e64 s[6:7], v226, 27
	v_cndmask_b32_e64 v96, v96, v227, s[4:5]
	v_cndmask_b32_e64 v95, v95, v227, s[6:7]
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v97, v95
	v_mov_b32_e32 v95, v93
	v_mov_b32_e32 v93, v91
	v_mov_b32_e32 v91, v89
	v_mov_b32_e32 v89, v87
	v_mov_b32_e32 v87, v85
	v_mov_b32_e32 v85, v83
	v_mov_b32_e32 v83, v0
.LBB0_20:
	v_sub_f32_e32 v0, v220, v225
	v_exp_f32_e32 v0, v0
	s_nop 0
	v_pk_mul_f32 v[16:17], v[0:1], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[0:1], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[0:1], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[0:1], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[0:1], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[0:1], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[0:1], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[0:1], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[0:1], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[0:1], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[0:1], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[0:1], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[0:1], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[0:1], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[0:1], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[0:1], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[0:1], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[0:1], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[0:1], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[0:1], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
; Cluster 7:
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(1)
; v_o = _mma1(v_p_1, v_packs_e7, v_o)
; m_tile_max_e7 = _attn_row_max(v_s_0)
; row_max_e7 = _fmax(m_row, m_tile_max_e7)
; rescale_e7 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e7)))
; m_row = row_max_e7
; v_s_0 = _attn_sub_row(v_s_0, row_max_e7)
; v_s_0 = _anchor_v_s(v_s_0)
; v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
; _sched_barrier_pairs(10, 5, 8)
; _sched_barrier_exp_pairs(6, 3, 8)
; rocdl.sched_barrier(0)
; _scale_o(v_o, rescale_e7)
; if const_expr(OPUS_SETPRIO):
; 	rocdl.s_setprio(0)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   Mirrors Cluster 3 with v_p_1/v_packs_e7/v_s_0 sources. MMA1 chain
;   (16 `v_mfma_f32_32x32x16_bf16 v[X:Y], v[A:B], v[98:101]/...`)
;   accumulates `v_p_1 * v_packs_e7` into v_o, interleaved with: (a)
;   `_attn_row_max(v_s_0)` reduction (`v_max_f32_e32 v194` + 15
;   `v_max3_f32` + `v_permlane32_swap` + `v_max3_f32 v194, v225, ...`
;   giving row_max_e7 = fmax(m_row=v225, m_tile_max_e7)); (b)
;   `v_sub_f32_e32 v81, v81, v194` and following — 32 sub_row
;   subtractions; (c) `v_exp_f32_e32 v98, v66` and following — first half
;   of v_p_0's exp2_slice (16 v_exp).
;   `s_setprio 0` + `s_barrier` close. As with Cluster 3,
;   `_scale_o(v_o, rescale_e7)` is HOISTED out and emitted at ASM
;   ~3859-3893 (right before Cluster 11's MMA1).
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[2:17], v[194:197], v[98:101], v[2:17]
	v_max_f32_e32 v194, v66, v67
	v_max3_f32 v194, v194, v68, v69
	v_max3_f32 v194, v194, v70, v71
	v_max3_f32 v194, v194, v72, v73
	v_max3_f32 v194, v194, v74, v75
	v_mfma_f32_32x32x16_bf16 v[50:65], v[198:201], v[98:101], v[50:65]
	v_max3_f32 v194, v194, v76, v77
	v_max3_f32 v194, v194, v78, v79
	v_max3_f32 v194, v194, v80, v81
	v_max3_f32 v194, v194, v82, v83
	v_max3_f32 v194, v194, v84, v85
	v_mfma_f32_32x32x16_bf16 v[34:49], v[202:205], v[98:101], v[34:49]
	v_max3_f32 v194, v194, v86, v87
	v_max3_f32 v194, v194, v88, v89
	v_max3_f32 v194, v194, v90, v91
	v_max3_f32 v194, v194, v92, v93
	v_max3_f32 v194, v194, v94, v95
	v_mfma_f32_32x32x16_bf16 v[18:33], v[206:209], v[98:101], v[18:33]
	v_max3_f32 v98, v194, v96, v97
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e64 v98, v99 bound_ctrl:1
	v_max3_f32 v194, v225, v98, v99
	v_sub_f32_e32 v81, v81, v194
	v_sub_f32_e32 v80, v80, v194
	v_mfma_f32_32x32x16_bf16 v[2:17], v[118:121], v[102:105], v[2:17]
	v_sub_f32_e32 v79, v79, v194
	v_sub_f32_e32 v78, v78, v194
	v_sub_f32_e32 v77, v77, v194
	v_sub_f32_e32 v76, v76, v194
	v_sub_f32_e32 v75, v75, v194
	v_mfma_f32_32x32x16_bf16 v[50:65], v[122:125], v[102:105], v[50:65]
	v_sub_f32_e32 v74, v74, v194
	v_sub_f32_e32 v73, v73, v194
	v_sub_f32_e32 v72, v72, v194
	v_sub_f32_e32 v71, v71, v194
	v_sub_f32_e32 v70, v70, v194
	v_mfma_f32_32x32x16_bf16 v[34:49], v[126:129], v[102:105], v[34:49]
	v_sub_f32_e32 v69, v69, v194
	v_sub_f32_e32 v68, v68, v194
	v_sub_f32_e32 v67, v67, v194
	v_sub_f32_e32 v66, v66, v194
	v_sub_f32_e32 v129, v97, v194
	v_mfma_f32_32x32x16_bf16 v[18:33], v[190:193], v[102:105], v[18:33]
	v_sub_f32_e32 v128, v96, v194
	v_sub_f32_e32 v127, v95, v194
	v_sub_f32_e32 v126, v94, v194
	v_sub_f32_e32 v125, v93, v194
	v_sub_f32_e32 v124, v92, v194
	v_mfma_f32_32x32x16_bf16 v[2:17], v[114:117], v[106:109], v[2:17]
	v_sub_f32_e32 v123, v91, v194
	v_sub_f32_e32 v122, v90, v194
	v_sub_f32_e32 v121, v89, v194
	v_sub_f32_e32 v120, v88, v194
	v_sub_f32_e32 v119, v87, v194
	v_mfma_f32_32x32x16_bf16 v[50:65], v[182:185], v[106:109], v[50:65]
	v_sub_f32_e32 v118, v86, v194
	v_sub_f32_e32 v117, v85, v194
	v_sub_f32_e32 v116, v84, v194
	v_sub_f32_e32 v115, v83, v194
	v_sub_f32_e32 v114, v82, v194
	v_mfma_f32_32x32x16_bf16 v[34:49], v[186:189], v[106:109], v[34:49]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_exp_f32_e32 v98, v66
	v_exp_f32_e32 v99, v67
	v_exp_f32_e32 v100, v68
	v_mfma_f32_32x32x16_bf16 v[18:33], v[178:181], v[106:109], v[18:33]
	v_exp_f32_e32 v101, v69
	v_exp_f32_e32 v102, v70
	v_exp_f32_e32 v103, v71
	v_mfma_f32_32x32x16_bf16 v[2:17], v[162:165], v[110:113], v[2:17]
	v_exp_f32_e32 v104, v72
	v_exp_f32_e32 v105, v73
	v_exp_f32_e32 v162, v74
	v_mfma_f32_32x32x16_bf16 v[50:65], v[166:169], v[110:113], v[50:65]
	v_exp_f32_e32 v163, v75
	v_exp_f32_e32 v164, v76
	v_exp_f32_e32 v165, v77
	v_mfma_f32_32x32x16_bf16 v[34:49], v[170:173], v[110:113], v[34:49]
	v_exp_f32_e32 v166, v78
	v_exp_f32_e32 v167, v79
	v_exp_f32_e32 v168, v80
	v_mfma_f32_32x32x16_bf16 v[18:33], v[174:177], v[110:113], v[18:33]
	v_exp_f32_e32 v110, v81
	s_setprio 0
	s_barrier
; Cluster 8:
; _async_load_v(max_m1 * fx.Index(BLOCK_N), 1)
; v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(NUM_DMA_V)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   `s_mov_b32 m0, s19` selects LDS write-base for the V[max_m1] async
;   prefetch (V buf-1 LDS base). Two `buffer_load_dwordx4 v215/v216,
;   s[0:3], s9 offen lds` issue the async load (s9 reuses the K[max_m1]
;   global offset already computed for Cluster 2). The 16 `ds_read_b128
;   v[82:85]/.../v[246:249], v211 offset:Y` read K from LDS buf-1
;   (offsets 34048..34656) and V slices (42368..42976) into v_k for
;   Cluster 9's MMA0.
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v215, s[0:3], s9 offen lds
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v216, s[0:3], s9 offen lds
	ds_read_b128 v[82:85], v211 offset:34048
	ds_read_b128 v[170:173], v211 offset:34080
	ds_read_b128 v[174:177], v211 offset:34560
	ds_read_b128 v[178:181], v211 offset:34592
	ds_read_b128 v[182:185], v211 offset:34112
	ds_read_b128 v[186:189], v211 offset:34144
	ds_read_b128 v[190:193], v211 offset:34624
	ds_read_b128 v[196:199], v211 offset:34656
	ds_read_b128 v[200:203], v211 offset:42368
	ds_read_b128 v[204:207], v211 offset:42400
	ds_read_b128 v[226:229], v211 offset:42880
	ds_read_b128 v[230:233], v211 offset:42912
	ds_read_b128 v[234:237], v211 offset:42432
	ds_read_b128 v[238:241], v211 offset:42464
	ds_read_b128 v[242:245], v211 offset:42944
	ds_read_b128 v[246:249], v211 offset:42976
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
; Cluster 9:
; v_s_1 = _mma0(v_k)
; l_row = _fmul(l_row, rescale_e7)
; v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
; tile_sum_e9 = _attn_sum(v_p_0)
; l_row = _fadd(l_row, tile_sum_e9)
; v_p_0 = _cast_p(v_p_0)
; v_p_0 = _anchor_v_p(v_p_0)
; _sched_barrier_exp_pairs(6, 3, 9)
; _sched_barrier_pairs(10, 5, 9)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   Same shape as Cluster 5 but for the second-to-last tile. 16 MMA0 on
;   `v[66:81]` / `v[82:97]` ping-pong, interleaved with 16 v_exp (second
;   half of v_p_0 exp2_slice) and 32 v_add (tile_sum_e9 chain). The final
;   `v_add_f32_e32 v179, v98, v113` + `v_mov_b32_e32 v180, v179` +
;   `v_permlane32_swap_b32_e64 v179, v180 bound_ctrl:1` produces the two
;   warp-level tile_sum_e9 halves kept in v179/v180.
;   As before, `l_row = _fmul(l_row, rescale_e7)` and
;   `l_row += tile_sum_e9` are deferred to the normalize-O reduction
;   (ASM ~4278 `v_fmac_f32_e32 v179, v178, v0`).
;   The 16 `v_cvt_pk_bf16_f32 v[98..113], ...` block then casts v_p_0
;   back to bf16 in place.
	v_mfma_f32_32x32x16_bf16 v[66:81], v[82:85], v[130:133], 0
	v_exp_f32_e32 v106, v114
	v_exp_f32_e32 v107, v115
	v_exp_f32_e32 v108, v116
	v_mfma_f32_32x32x16_bf16 v[82:97], v[174:177], v[130:133], 0
	v_exp_f32_e32 v109, v117
	v_exp_f32_e32 v111, v118
	v_exp_f32_e32 v112, v119
	v_mfma_f32_32x32x16_bf16 v[66:81], v[170:173], v[134:137], v[66:81]
	v_exp_f32_e32 v113, v120
	v_exp_f32_e32 v114, v121
	v_exp_f32_e32 v115, v122
	v_mfma_f32_32x32x16_bf16 v[82:97], v[178:181], v[134:137], v[82:97]
	v_exp_f32_e32 v116, v123
	v_exp_f32_e32 v117, v124
	v_exp_f32_e32 v118, v125
	v_mfma_f32_32x32x16_bf16 v[66:81], v[182:185], v[138:141], v[66:81]
	v_exp_f32_e32 v119, v126
	v_exp_f32_e32 v120, v127
	v_exp_f32_e32 v121, v128
	v_mfma_f32_32x32x16_bf16 v[82:97], v[190:193], v[138:141], v[82:97]
	v_exp_f32_e32 v122, v129
	v_mfma_f32_32x32x16_bf16 v[66:81], v[186:189], v[142:145], v[66:81]
	v_add_f32_e32 v123, v98, v99
	v_add_f32_e32 v123, v123, v100
	v_add_f32_e32 v123, v123, v101
	v_add_f32_e32 v123, v123, v102
	v_add_f32_e32 v123, v123, v103
	v_mfma_f32_32x32x16_bf16 v[82:97], v[196:199], v[142:145], v[82:97]
	v_add_f32_e32 v123, v123, v104
	v_add_f32_e32 v123, v123, v105
	v_add_f32_e32 v123, v123, v162
	v_add_f32_e32 v123, v123, v163
	v_add_f32_e32 v123, v123, v164
	v_mfma_f32_32x32x16_bf16 v[66:81], v[200:203], v[146:149], v[66:81]
	v_add_f32_e32 v123, v123, v165
	v_add_f32_e32 v123, v123, v166
	v_add_f32_e32 v123, v123, v167
	v_add_f32_e32 v123, v123, v168
	v_add_f32_e32 v123, v123, v110
	v_mfma_f32_32x32x16_bf16 v[82:97], v[226:229], v[146:149], v[82:97]
	v_add_f32_e32 v123, v123, v106
	v_add_f32_e32 v123, v123, v107
	v_add_f32_e32 v123, v123, v108
	v_add_f32_e32 v123, v123, v109
	v_add_f32_e32 v123, v123, v111
	v_mfma_f32_32x32x16_bf16 v[66:81], v[204:207], v[150:153], v[66:81]
	v_add_f32_e32 v123, v123, v112
	v_add_f32_e32 v123, v123, v113
	v_add_f32_e32 v123, v123, v114
	v_add_f32_e32 v123, v123, v115
	v_add_f32_e32 v123, v123, v116
	v_mfma_f32_32x32x16_bf16 v[82:97], v[230:233], v[150:153], v[82:97]
	v_add_f32_e32 v123, v123, v117
	v_add_f32_e32 v123, v123, v118
	v_add_f32_e32 v123, v123, v119
	v_add_f32_e32 v123, v123, v120
	v_add_f32_e32 v123, v123, v121
	v_mfma_f32_32x32x16_bf16 v[66:81], v[234:237], v[154:157], v[66:81]
	v_add_f32_e32 v179, v123, v122
	v_mov_b32_e32 v180, v179
	s_nop 1
	v_permlane32_swap_b32_e64 v180, v179 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v98, v99
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v100, v101
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v102, v103
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[242:245], v[154:157], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v106, v107
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v108, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v111, v112
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v113, v114
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[66:81], v[238:241], v[158:161], v[66:81]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v164, v165
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v166, v167
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v168, v110
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v115, v116
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[82:97], v[246:249], v[158:161], v[82:97]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v117, v118
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v119, v120
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v121, v122
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	s_barrier
; Cluster 10:
; v_packs_e11 = _read_v_packs_for_buf(0, urv_base_per_lane)
; if const_expr(CAUSAL):
; 	v_s_1 = _causal_mask_prologue_if_needed(
; 		v_s_1,
; 		max_m1,
; 		max_num_tiles * fx.Index(BLOCK_N),
; 	)
; else:
; 	v_s_1 = _v_s_vec_to_lists(v_s_1)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; _waitcnt_vm_n(0)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   The 32 `ds_read_b64_tr_b16 v[X:X+1], v217 offset:Y` block reads the
;   transposed V[max_num_tiles-1] tile from LDS buf-0 into v_packs_e11
;   (used by Cluster 11's MMA1).
;   The trailing `s_cmp_ge_i32 s16, s4` + `;;#ASMSTART/.../;;#ASMEND`
;   inline-asm block computes the causal-boundary test
;   `q_start_pos >= max_num_tiles * BLOCK_N` and conditionally writes
;   NEG_INF=`0xff800000` into out-of-range lanes via `v_cmp_lt_i32` +
;   `v_cndmask` chains; `s_cbranch_scc1 .LBB0_22` skips the mask body
;   when out of range. Both paths converge at .LBB0_22 below.
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v217 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v217 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v217 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v217 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v217 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v217 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v217 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v217 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v217 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v217 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v217 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v217 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v217 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v217 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v217 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v217 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v217 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v217 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v217 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v217 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v217 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v217 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v217 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v217 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v217 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v217 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v217 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v217 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v217 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v217 offset:9536

	;;#ASMEND
	s_lshl_b32 s0, s8, 6
	s_cmp_le_i32 s0, s16
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v217 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_22
	v_lshl_or_b32 v178, v214, 2, s10
	v_sub_u32_e32 v178, v212, v178
	v_subrev_u32_e32 v181, 32, v178
	v_mov_b32_e32 v182, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 0
	v_cmp_lt_i32_e64 s[2:3], v178, 1
	v_cndmask_b32_e64 v66, v66, v182, s[0:1]
	v_cndmask_b32_e64 v67, v67, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 2
	v_cmp_lt_i32_e64 s[2:3], v178, 3
	v_cndmask_b32_e64 v68, v68, v182, s[0:1]
	v_cndmask_b32_e64 v69, v69, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 8
	v_cmp_lt_i32_e64 s[2:3], v178, 9
	v_cndmask_b32_e64 v70, v70, v182, s[0:1]
	v_cndmask_b32_e64 v71, v71, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 10
	v_cmp_lt_i32_e64 s[2:3], v178, 11
	v_cndmask_b32_e64 v72, v72, v182, s[0:1]
	v_cndmask_b32_e64 v73, v73, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 16
	v_cmp_lt_i32_e64 s[2:3], v178, 17
	v_cndmask_b32_e64 v74, v74, v182, s[0:1]
	v_cndmask_b32_e64 v75, v75, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 18
	v_cmp_lt_i32_e64 s[2:3], v178, 19
	v_cndmask_b32_e64 v76, v76, v182, s[0:1]
	v_cndmask_b32_e64 v77, v77, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 24
	v_cmp_lt_i32_e64 s[2:3], v178, 25
	v_cndmask_b32_e64 v78, v78, v182, s[0:1]
	v_cndmask_b32_e64 v79, v79, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v178, 26
	v_cmp_lt_i32_e64 s[2:3], v178, 27
	v_cndmask_b32_e64 v80, v80, v182, s[0:1]
	v_cndmask_b32_e64 v81, v81, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 0
	v_cmp_lt_i32_e64 s[2:3], v181, 1
	v_cndmask_b32_e64 v82, v82, v182, s[0:1]
	v_cndmask_b32_e64 v83, v83, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 2
	v_cmp_lt_i32_e64 s[2:3], v181, 3
	v_cndmask_b32_e64 v84, v84, v182, s[0:1]
	v_cndmask_b32_e64 v85, v85, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 8
	v_cmp_lt_i32_e64 s[2:3], v181, 9
	v_cndmask_b32_e64 v86, v86, v182, s[0:1]
	v_cndmask_b32_e64 v87, v87, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 10
	v_cmp_lt_i32_e64 s[2:3], v181, 11
	v_cndmask_b32_e64 v88, v88, v182, s[0:1]
	v_cndmask_b32_e64 v89, v89, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 16
	v_cmp_lt_i32_e64 s[2:3], v181, 17
	v_cndmask_b32_e64 v90, v90, v182, s[0:1]
	v_cndmask_b32_e64 v91, v91, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 18
	v_cmp_lt_i32_e64 s[2:3], v181, 19
	v_cndmask_b32_e64 v92, v92, v182, s[0:1]
	v_cndmask_b32_e64 v93, v93, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 24
	v_cmp_lt_i32_e64 s[2:3], v181, 25
	v_cndmask_b32_e64 v94, v94, v182, s[0:1]
	v_cndmask_b32_e64 v95, v95, v182, s[2:3]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[0:1], v181, 26
	v_cmp_lt_i32_e64 s[2:3], v181, 27
	v_cndmask_b32_e64 v96, v96, v182, s[0:1]
	v_cndmask_b32_e64 v97, v97, v182, s[2:3]
	;;#ASMEND
.LBB0_22:
; Cluster 10 causal-mask skip target.
; if const_expr(CAUSAL):
; 	v_s_1 = _causal_mask_prologue_if_needed(
; 		v_s_1,
; 		max_m1,
; 		max_num_tiles * fx.Index(BLOCK_N),
; 	)
; NOTE:
;   `.LBB0_22` is reached when `q_start_pos >= max_num_tiles * BLOCK_N`
;   (this wave is past the causal boundary for the final KV tile). Both
;   the masked and skipped paths converge here.
; SCHED HINT (Cluster 7 hoisted rescale + scale_o):
;   The next ~37 instructions (from `v_sub_f32_e32 v178, v225, v194` up
;   to the closing `s_waitcnt vmcnt(0) lgkmcnt(0)` + `s_barrier`) are NOT
;   Cluster 10 code — they are Cluster 7's deferred
;   `rescale_e7 = exp2(m_row_prev - row_max_e7)` (the `v_sub_f32_e32 v178,
;   v225, v194` → `v_exp_f32_e32 v178, v178` pair) followed by
;   `_scale_o(v_o, rescale_e7)` (32 `v_pk_mul_f32 v[X:Y], v[178:179],
;   v[X:Y] op_sel_hi:[0,1]`). LLVM hoisted them out of Cluster 7 (PY
;   L1886-L1893) and emitted them here so v_o is freshly scaled exactly
;   when Cluster 11's MMA1 reads it next.
;   The closing `s_barrier` is Cluster 6's source-level `rocdl.s_barrier()`
;   (PY L1855), retained as the synchronization fence that separates
;   Cluster 6/10 (read v_packs + mask + Cluster 7 hoisted scale_o) from
;   Cluster 11 (mma1 + softmax + cast P + scale_o).
	v_sub_f32_e32 v178, v225, v194
	v_exp_f32_e32 v178, v178
	s_nop 0
	v_pk_mul_f32 v[16:17], v[178:179], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[178:179], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[178:179], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[178:179], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[178:179], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[178:179], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[178:179], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[178:179], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[178:179], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[178:179], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[178:179], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[178:179], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[178:179], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[178:179], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[178:179], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[178:179], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[178:179], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[178:179], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[178:179], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[178:179], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[178:179], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[178:179], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[178:179], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[178:179], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[178:179], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[178:179], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[178:179], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[178:179], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[178:179], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[178:179], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[178:179], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[178:179], v[18:19] op_sel_hi:[0,1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
; Cluster 11:
; v_o = _mma1(v_p_0, v_packs_e11, v_o)
; m_tile_max_e11 = _attn_row_max(v_s_1)
; row_max_e11 = _fmax(m_row, m_tile_max_e11)
; rescale_e11 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e11)))
; m_row = row_max_e11
; v_s_1 = _attn_sub_row(v_s_1, row_max_e11)
; v_s_1 = _anchor_v_s(v_s_1)
; v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
; _sched_barrier_pairs(10, 5, 10)
; _sched_barrier_exp_pairs(6, 3, 10)
; rocdl.sched_barrier(0)
; v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
; l_row = _fmul(l_row, rescale_e11)
; tile_sum_e11 = _attn_sum(v_p_1)
; l_row = _fadd(l_row, tile_sum_e11)
; v_p_1 = _cast_p(v_p_1)
; v_p_1 = _anchor_v_p(v_p_1)
; rocdl.sched_barrier(0)
; _scale_o(v_o, rescale_e11)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   The last full per-tile online-softmax update. Unlike Clusters 3 and 7,
;   Cluster 11 does NOT use `s_setprio` — the dual-group phase-shift only
;   matters when the next-iteration K/V prefetch must overlap with VALU;
;   here both wave-groups will fall through to the final V read + drain.
;   The cluster combines both halves of the exp2_slice and a full
;   `_attn_sum` in one stretch:
;     • `v_mfma_f32_32x32x16_bf16 v[2:17], v[122:125], v[98:101], v[2:17]`
;       and 15 more MMA1 — `v_o = _mma1(v_p_0, v_packs_e11, v_o)`.
;     • `v_max_f32_e32 v122, v66, v67` + 15 `v_max3_f32` +
;       `v_permlane32_swap` + `v_max3_f32 v98, v194, v98, v99` —
;       `_attn_row_max(v_s_1)` then `_fmax(m_row, ...)`.
;     • `v_sub_f32_e32 v99, v194, v98` — `m_row_prev - row_max_e11` for
;       rescale_e11; the subsequent `v_exp_f32_e32 v84, v99` computes
;       rescale_e11 itself (held in v84).
;     • `v_sub_f32_e32 v129..v66, vX, v98` — 32 sub_row.
;     • `v_exp_f32_e32 v85..v100, vX` + `v_exp_f32_e32 v101..v81, vX` —
;       the full v_p_1 exp2 (both halves merged: 16 from
;       `_attn_exp2_slice(v_s_1, 0, 16)` and 16 from
;       `_attn_exp2_slice(v_p_1, 16, 16)`).
;     • `v_add_f32_e32 v66, v85, v86` + 31 more — `_attn_sum(v_p_1)`.
;     • `v_add_f32_e32 v82, v66, v81` + `v_permlane32_swap_b32_e64 v83,
;       v82 bound_ctrl:1` — warp-level tile_sum_e11 reduction; v82/v83
;       are kept live and folded into l_row at ASM ~4282
;       (`v_fmac_f32_e32 v82, v84, v0`).
;     • 16 `v_cvt_pk_bf16_f32 v[66..81], ...` — `_cast_p(v_p_1)`.
;     • 32 `v_pk_mul_f32 v[X:Y], v[84:85], v[X:Y] op_sel_hi:[0,1]` —
;       `_scale_o(v_o, rescale_e11)` (rescale_e11 in v84/v85).
;     • Final `s_barrier` (PY L2005) — synchronizes both wave-groups
;       before the last V read in Cluster 12.
	v_mfma_f32_32x32x16_bf16 v[2:17], v[122:125], v[98:101], v[2:17]
	v_max_f32_e32 v122, v66, v67
	v_max3_f32 v122, v122, v68, v69
	v_max3_f32 v122, v122, v70, v71
	v_max3_f32 v122, v122, v72, v73
	v_max3_f32 v122, v122, v74, v75
	v_mfma_f32_32x32x16_bf16 v[50:65], v[126:129], v[98:101], v[50:65]
	v_max3_f32 v122, v122, v76, v77
	v_max3_f32 v122, v122, v78, v79
	v_max3_f32 v122, v122, v80, v81
	v_max3_f32 v122, v122, v82, v83
	v_max3_f32 v122, v122, v84, v85
	v_mfma_f32_32x32x16_bf16 v[34:49], v[170:173], v[98:101], v[34:49]
	v_max3_f32 v122, v122, v86, v87
	v_max3_f32 v122, v122, v88, v89
	v_max3_f32 v122, v122, v90, v91
	v_max3_f32 v122, v122, v92, v93
	v_max3_f32 v122, v122, v94, v95
	v_mfma_f32_32x32x16_bf16 v[18:33], v[174:177], v[98:101], v[18:33]
	v_max3_f32 v98, v122, v96, v97
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e64 v98, v99 bound_ctrl:1
	v_max3_f32 v98, v194, v98, v99
	v_sub_f32_e32 v99, v194, v98
	v_sub_f32_e32 v129, v81, v98
	v_mfma_f32_32x32x16_bf16 v[2:17], v[114:117], v[102:105], v[2:17]
	v_sub_f32_e32 v128, v80, v98
	v_sub_f32_e32 v127, v79, v98
	v_sub_f32_e32 v126, v78, v98
	v_sub_f32_e32 v125, v77, v98
	v_sub_f32_e32 v124, v76, v98
	v_mfma_f32_32x32x16_bf16 v[50:65], v[118:121], v[102:105], v[50:65]
	v_sub_f32_e32 v123, v75, v98
	v_sub_f32_e32 v122, v74, v98
	v_sub_f32_e32 v121, v73, v98
	v_sub_f32_e32 v120, v72, v98
	v_sub_f32_e32 v119, v71, v98
	v_mfma_f32_32x32x16_bf16 v[34:49], v[162:165], v[102:105], v[34:49]
	v_sub_f32_e32 v118, v70, v98
	v_sub_f32_e32 v117, v69, v98
	v_sub_f32_e32 v116, v68, v98
	v_sub_f32_e32 v115, v67, v98
	v_sub_f32_e32 v114, v66, v98
	v_mfma_f32_32x32x16_bf16 v[18:33], v[166:169], v[102:105], v[18:33]
	v_sub_f32_e32 v81, v97, v98
	v_sub_f32_e32 v80, v96, v98
	v_sub_f32_e32 v79, v95, v98
	v_sub_f32_e32 v78, v94, v98
	v_sub_f32_e32 v77, v93, v98
	v_mfma_f32_32x32x16_bf16 v[2:17], v[150:153], v[106:109], v[2:17]
	v_sub_f32_e32 v76, v92, v98
	v_sub_f32_e32 v75, v91, v98
	v_sub_f32_e32 v74, v90, v98
	v_sub_f32_e32 v73, v89, v98
	v_sub_f32_e32 v72, v88, v98
	v_mfma_f32_32x32x16_bf16 v[50:65], v[154:157], v[106:109], v[50:65]
	v_sub_f32_e32 v71, v87, v98
	v_sub_f32_e32 v70, v86, v98
	v_sub_f32_e32 v69, v85, v98
	v_sub_f32_e32 v68, v84, v98
	v_sub_f32_e32 v67, v83, v98
	v_mfma_f32_32x32x16_bf16 v[34:49], v[158:161], v[106:109], v[34:49]
	v_exp_f32_e32 v84, v99
	v_sub_f32_e32 v66, v82, v98
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_exp_f32_e32 v85, v114
	v_exp_f32_e32 v86, v115
	v_mfma_f32_32x32x16_bf16 v[18:33], v[146:149], v[106:109], v[18:33]
	v_exp_f32_e32 v87, v116
	v_exp_f32_e32 v88, v117
	v_exp_f32_e32 v89, v118
	v_mfma_f32_32x32x16_bf16 v[2:17], v[130:133], v[110:113], v[2:17]
	v_exp_f32_e32 v90, v119
	v_exp_f32_e32 v91, v120
	v_exp_f32_e32 v92, v121
	v_mfma_f32_32x32x16_bf16 v[50:65], v[134:137], v[110:113], v[50:65]
	v_exp_f32_e32 v93, v122
	v_exp_f32_e32 v94, v123
	v_exp_f32_e32 v95, v124
	v_mfma_f32_32x32x16_bf16 v[34:49], v[138:141], v[110:113], v[34:49]
	v_exp_f32_e32 v96, v125
	v_exp_f32_e32 v97, v126
	v_exp_f32_e32 v98, v127
	v_mfma_f32_32x32x16_bf16 v[18:33], v[142:145], v[110:113], v[18:33]
	v_exp_f32_e32 v99, v128
	v_exp_f32_e32 v100, v129
	v_exp_f32_e32 v101, v66
	v_exp_f32_e32 v102, v67
	v_exp_f32_e32 v103, v68
	v_exp_f32_e32 v104, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v105, v74
	v_exp_f32_e32 v106, v75
	v_exp_f32_e32 v107, v76
	v_exp_f32_e32 v108, v77
	v_exp_f32_e32 v109, v78
	v_exp_f32_e32 v110, v79
	v_exp_f32_e32 v111, v80
	v_exp_f32_e32 v81, v81
	v_add_f32_e32 v66, v85, v86
	v_add_f32_e32 v66, v66, v87
	v_add_f32_e32 v66, v66, v88
	v_add_f32_e32 v66, v66, v89
	v_add_f32_e32 v66, v66, v90
	v_add_f32_e32 v66, v66, v91
	v_add_f32_e32 v66, v66, v92
	v_add_f32_e32 v66, v66, v93
	v_add_f32_e32 v66, v66, v94
	v_add_f32_e32 v66, v66, v95
	v_add_f32_e32 v66, v66, v96
	v_add_f32_e32 v66, v66, v97
	v_add_f32_e32 v66, v66, v98
	v_add_f32_e32 v66, v66, v99
	v_add_f32_e32 v66, v66, v100
	v_add_f32_e32 v66, v66, v101
	v_add_f32_e32 v66, v66, v102
	v_add_f32_e32 v66, v66, v103
	v_add_f32_e32 v66, v66, v104
	v_add_f32_e32 v66, v66, v70
	v_add_f32_e32 v66, v66, v71
	v_add_f32_e32 v66, v66, v72
	v_add_f32_e32 v66, v66, v73
	v_add_f32_e32 v66, v66, v105
	v_add_f32_e32 v66, v66, v106
	v_add_f32_e32 v66, v66, v107
	v_add_f32_e32 v66, v66, v108
	v_add_f32_e32 v66, v66, v109
	v_add_f32_e32 v66, v66, v110
	v_add_f32_e32 v66, v66, v111
	v_add_f32_e32 v82, v66, v81
	v_mov_b32_e32 v83, v82
	s_nop 1
	v_permlane32_swap_b32_e64 v83, v82 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v85, v86
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v87, v88
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v89, v90
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v91, v92
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v101, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v103, v104
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v72, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v93, v94
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v95, v96
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v97, v98
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v99, v100
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v105, v106
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v107, v108
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v80, v109, v110
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v81, v111, v81
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	v_pk_mul_f32 v[16:17], v[84:85], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[84:85], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[84:85], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[84:85], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[84:85], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[84:85], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[84:85], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[84:85], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[84:85], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[84:85], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[84:85], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[84:85], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[84:85], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[84:85], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[84:85], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[84:85], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[84:85], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[84:85], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[84:85], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[84:85], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[84:85], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[84:85], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[84:85], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[84:85], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[84:85], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[84:85], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[84:85], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[84:85], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[84:85], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[84:85], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[84:85], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[84:85], v[18:19] op_sel_hi:[0,1]
	s_barrier
; Cluster 12:
; v_packs_e13 = _read_v_packs_for_buf(1, urv_base_per_lane)
; rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
; rocdl.sched_barrier(0)
; rocdl.s_barrier()
; rocdl.sched_barrier(0)
; NOTE:
;   Final V tile read. Thirty-two `ds_read_b64_tr_b16 v[X:X+1], v219
;   offset:Y` (offsets 0..960 + 8704..9664) materialize the transposed
;   V[max_num_tiles-1] tile into v_packs_e13 (used by Cluster 13's
;   final MMA1). No async_load_k / async_load_v — the kernel has now
;   exhausted all KV tiles. The closing `s_waitcnt lgkmcnt(0)` +
;   `s_barrier` (PY L2017) waits for the V LDS reads to finish and
;   synchronizes wave-groups before the final MMA1 chain.
	;;#ASMSTART
	ds_read_b64_tr_b16 v[86:87], v219 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[88:89], v219 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[90:91], v219 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[92:93], v219 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v219 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v219 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v219 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[100:101], v219 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[102:103], v219 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[104:105], v219 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[106:107], v219 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[108:109], v219 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[110:111], v219 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v219 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v219 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v219 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v219 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v219 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v219 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v219 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v219 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v219 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v219 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v219 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v219 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v219 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v219 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v219 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v219 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v219 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v219 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v219 offset:9664

	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	s_barrier
; Cluster 13:
; v_o = _mma1(v_p_1, v_packs_e13, v_o)
; NOTE:
;   Final P*V MMA chain: sixteen `v_mfma_f32_32x32x16_bf16` instructions
;   drain v_packs_e13 (V tile read in Cluster 12) into the four D_CHUNKS
;   fp32 accumulator banks `v[2:17]`, `v[50:65]`, `v[34:49]`, `v[18:33]`.
;   No softmax / no rescale — the normalize phase (l_row reduction +
;   rcp + scale_o(v_o, inv_l) + stores) follows immediately after the
;   stagger barrier below.
	v_mfma_f32_32x32x16_bf16 v[2:17], v[86:89], v[66:69], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[50:65], v[102:105], v[66:69], v[50:65]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[118:121], v[66:69], v[34:49]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[134:137], v[66:69], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[90:93], v[70:73], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[50:65], v[106:109], v[70:73], v[50:65]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[122:125], v[70:73], v[34:49]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[138:141], v[70:73], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[94:97], v[74:77], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[50:65], v[110:113], v[74:77], v[50:65]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[126:129], v[74:77], v[34:49]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[142:145], v[74:77], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[98:101], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[50:65], v[114:117], v[78:81], v[50:65]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[130:133], v[78:81], v[34:49]
; Stagger barrier (end of Cluster 13 → entry to normalize/store).
; if const_expr(OPUS_ENABLE_STAGGER):
; 	_stagger_extra_barrier_if_zero()
; else:
; 	rocdl.s_barrier()
; NOTE:
;   Inline-asm block: `s_cmp_eq_u32 s25, 0; s_cbranch_scc0 1f; s_barrier;
;   1:`. Warps 0-3 (`stagger_id == 0`) execute the inline `s_barrier` and
;   thereby consume the extra barrier that mirror-paired with warps 4-7's
;   prologue `_stagger_extra_barrier_if_one` (.LBB0_2 path at ASM 235).
;   Together with that prologue partner, this keeps both wave-groups
;   exactly one s_barrier ordinal apart through the kernel and re-aligns
;   them here before the final O normalization and store phase.
	;;#ASMSTART
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc0 1f
	s_barrier
	1:
	;;#ASMEND
	v_cmp_gt_u64_e32 vcc, s[22:23], v[212:213]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[146:149], v[78:81], v[18:33]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_24
; Epilogue normalize O.
; inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
; inv_l = ArithValue(fx.Float32(l_row) > c_zero_f).select(inv_l_rcp, c_zero_f)
; _scale_o(v_o, inv_l)
; if const_expr(OPUS_ENABLE_STAGGER):
; 	_stagger_extra_barrier_if_zero()    # already emitted above
; else:
; 	rocdl.s_barrier()                   # already emitted above
; NOTE:
;   The `v_add_f32_e32` / `v_fmac_f32_e32` chain folds the per-cluster
;   partial l_row contributions into the final denominator in v0:
;     • v218         = main-loop l_row (set up at ASM 898; updated by the
;                      loop body's tile_sum_a chain).
;     • v222, v221   = Cluster 1 tile_sum_e1 halves (lhs / rhs of the
;                      `v_permlane32_swap` at ASM 2532).
;     • v223, v224   = Cluster 5 tile_sum_e5 halves; v0 = `rescale_e?`
;                      threading them into v223 via `v_fmac_f32_e32
;                      v223, v0, v66`.
;     • v179, v180   = Cluster 9 tile_sum_e9 halves; same pattern
;                      with `v_fmac_f32_e32 v179, v178, v0` (v178 is
;                      Cluster 11's rescale_e11 carrier).
;     • v82, v83/v84 = Cluster 11 tile_sum_e11 halves; finished by
;                      `v_fmac_f32_e32 v82, v84, v0`.
;   The carried fmacs implement the deferred
;   `l_row = (l_row * rescale_eX) + tile_sum_eX` updates that were
;   omitted from Clusters 5/9/11.
;   `v_rcp_f32_e32 v66, v0` computes `1.0 / l_row`.
;   `v_cmp_lt_f32_e32 vcc, 0, v0` + `v_cndmask_b32_e32 v0, 0, v66, vcc`
;   implements the `(l_row > 0) ? 1/l_row : 0` guard from the
;   `ArithValue(...).select(inv_l_rcp, c_zero_f)` source expression.
;   The thirty-two `v_pk_mul_f32 v[X:Y], v[X:Y], v[0:1] op_sel_hi:[1,0]`
;   below are `_scale_o(v_o, inv_l)` — D_CHUNKS=4 chunks × 8
;   packed-pair multiplies each.
	v_add_f32_e32 v66, v218, v222
	v_add_f32_e32 v66, v66, v221
	v_fmac_f32_e32 v223, v0, v66
	v_add_f32_e32 v0, v223, v224
	v_fmac_f32_e32 v179, v178, v0
	v_add_f32_e32 v0, v179, v180
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	v_fmac_f32_e32 v82, v84, v0
	v_add_f32_e32 v0, v82, v83
	v_rcp_f32_e32 v66, v0
	s_mov_b32 s0, s14
	s_mov_b32 s1, s15
	v_cmp_lt_f32_e32 vcc, 0, v0
	s_nop 1
	v_cndmask_b32_e32 v0, 0, v66, vcc
	v_pk_mul_f32 v[32:33], v[32:33], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[0:1] op_sel_hi:[1,0]
	s_nop 0
; Epilogue final output stores.
; q_in_bounds = q_row < seq_len_v
; if q_in_bounds:
; 	for dc in range_constexpr(D_CHUNKS):
; 		for store_group in range_constexpr(4):
; 			r_base = store_group * 4
; 			lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base],
; 			                            Vec(v_o[dc])[r_base + 1])
; 			hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2],
; 			                            Vec(v_o[dc])[r_base + 3])
; 			o_pack = Vec.from_elements([lo, hi], fx.Int32).ir_value()
; 			d_row_rel = lane_div_32 * 4 + store_group * 8
; 			d_col = fx.Index(dc * D_CHUNK) + d_row_rel
; 			o_global = _global_idx_q(q_row, d_col)
; 			o_byte_offset = fx.Int32(o_global * fx.Index(BF16_BYTES))
; 			rocdl.raw_buffer_store(o_pack, o_rsrc,
; 				_llvm_value(o_byte_offset), _llvm_value(fx.Int32(0)),
; 				_llvm_value(fx.Int32(0)))
; NOTE:
;   The `q_in_bounds` guard was actually emitted earlier as
;   `v_cmp_gt_u64_e32 vcc, s[22:23], v[212:213]` + `s_and_saveexec_b64
;   s[0:1], vcc` + `s_cbranch_execz .LBB0_24` (ASM ~4270-4273); lanes
;   beyond seq_len skip the stores by exec-masking.
;   For each of the four D_CHUNKS, four `store_group`s emit:
;     • `v_cvt_pk_bf16_f32 v0, vX, vY` — pack two fp32 lanes into bf16.
;     • `v_cvt_pk_bf16_f32 v1, vZ, vW` — pack the next two.
;     • `buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:Y` —
;       write the 8-byte bf16 pack to O at offset Y = 0, 16, 32, ..., 240.
;   Address `v4 = v210 + v1 + s20` was constructed once at
;   `v_add_lshl_u32 v4, v0, v1, 1` (just below) and reused for all 32
;   stores covering this lane's 128-wide head dimension.
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v2, v3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v4, v5
	;;#ASMEND
	v_add_u32_e32 v0, s20, v210
	v_add_lshl_u32 v4, v0, v1, 1
	buffer_store_dwordx2 v[2:3], v4, s[0:3], 0 offen
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v8, v9
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:16
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v10, v11
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v12, v13
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:32
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v14, v15
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v16, v17
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:48
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v50, v51
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v52, v53
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:64
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v54, v55
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v56, v57
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:80
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v58, v59
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v60, v61
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:96
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v62, v63
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v64, v65
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:112
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v34, v35
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v36, v37
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:128
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v38, v39
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v40, v41
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:144
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v42, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v44, v45
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:160
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v46, v47
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v48, v49
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:176
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v20, v21
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:192
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v22, v23
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v24, v25
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:208
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v28, v29
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:224
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v30, v31
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v32, v33
	;;#ASMEND
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen offset:240
.LBB0_24:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flash_attn_opus_kernel_0
		.amdhsa_group_segment_fixed_size 68096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
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
		.amdhsa_next_free_vgpr 250
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 252
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
	.text
.Lfunc_end0:
	.size	flash_attn_opus_kernel_0, .Lfunc_end0-flash_attn_opus_kernel_0

	.set flash_attn_opus_kernel_0.num_vgpr, 250
	.set flash_attn_opus_kernel_0.num_agpr, 0
	.set flash_attn_opus_kernel_0.numbered_sgpr, 44
	.set flash_attn_opus_kernel_0.num_named_barrier, 0
	.set flash_attn_opus_kernel_0.private_seg_size, 0
	.set flash_attn_opus_kernel_0.uses_vcc, 1
	.set flash_attn_opus_kernel_0.uses_flat_scratch, 0
	.set flash_attn_opus_kernel_0.has_dyn_sized_stack, 0
	.set flash_attn_opus_kernel_0.has_recursion, 0
	.set flash_attn_opus_kernel_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 68096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 512
    .name:           flash_attn_opus_kernel_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         flash_attn_opus_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     250
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
