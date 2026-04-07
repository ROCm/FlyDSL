	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_gemm1_0
	.p2align	8
	.type	moe_gemm1_0,@function
moe_gemm1_0:
	s_load_dwordx2 s[24:25], s[0:1], 0x40
	s_mov_b32 s26, 4
	s_mov_b32 s27, 0x27000
	s_mov_b32 s8, s3
	s_ashr_i32 s9, s3, 31
	s_waitcnt lgkmcnt(0)
	s_and_b32 s25, s25, 0xffff
	buffer_load_dword v1, off, s[24:27], 0
	s_lshl_b64 s[10:11], s[8:9], 4
	s_mov_b32 s11, 0
	s_waitcnt vmcnt(0)
	v_cmp_ge_u32_e32 vcc, s10, v1
	s_cbranch_vccnz .LBB0_11
	s_load_dwordx4 s[20:23], s[0:1], 0x20
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx4 s[4:7], s[0:1], 0x48
	s_lshl_b32 s3, s8, 2
	s_mov_b32 s15, s27
	v_mov_b32_e32 v1, s3
	s_waitcnt lgkmcnt(0)
	s_and_b32 s13, s13, 0xffff
	s_lshl_b32 s14, s7, 2
	buffer_load_dword v18, v1, s[12:15], 0 offen
	v_lshrrev_b32_e32 v23, 5, v0
	s_mov_b32 s30, -1
	s_and_b32 s9, s23, 0xffff
	v_or_b32_e32 v1, s10, v23
	v_or_b32_e32 v27, 8, v23
	s_mov_b32 s16, s22
	s_mov_b32 s17, s9
	s_mov_b32 s18, s30
	s_mov_b32 s19, s27
	v_lshlrev_b32_e32 v1, 2, v1
	v_or_b32_e32 v2, s10, v27
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_load_dword v3, v1, s[16:19], 0 offen
	buffer_load_dword v4, v2, s[16:19], 0 offen
	s_ashr_i32 s3, s2, 31
	v_lshlrev_b32_e32 v1, 2, v0
	v_bfe_u32 v26, v0, 4, 2
	v_and_b32_e32 v41, 15, v0
	v_lshrrev_b32_e32 v0, 2, v0
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_lshl_b64 s[0:1], s[2:3], 6
	v_and_b32_e32 v0, 48, v0
	v_and_b32_e32 v53, 0x7c, v1
	v_or3_b32 v24, s0, v0, v41
	v_mov_b32_e32 v25, s1
	s_ashr_i32 s1, s6, 31
	s_mov_b32 s0, s6
	s_ashr_i32 s33, s6, 3
	s_mul_i32 s6, s4, s6
	s_lshl_b32 s42, s6, 1
	s_lshr_b64 s[6:7], s[0:1], 1
	s_mov_b32 s23, 0x2aaaaaab
	s_waitcnt lgkmcnt(0)
	s_and_b32 s41, s15, 0xffff
	s_mov_b32 s40, s14
	s_mov_b32 s24, s14
	v_mov_b32_e32 v17, 0
	s_lshr_b64 s[2:3], s[0:1], 3
	s_and_b32 s17, s17, 0xffff
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s8, s22
	s_mov_b32 s25, s41
	s_lshl_b32 s3, s2, 6
	s_mov_b32 s26, s42
	s_mov_b32 s43, s27
	s_mov_b32 s31, s27
	s_mov_b32 s39, s27
	s_mov_b32 s38, s30
	s_mov_b32 s28, s16
	s_mov_b32 s36, s20
	s_mov_b32 s29, s17
	s_mov_b32 s37, s21
	s_waitcnt vmcnt(2)
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshlrev_b64 v[0:1], 9, v[18:19]
	v_lshl_add_u64 v[20:21], v[0:1], 0, v[24:25]
	v_ashrrev_i32_e32 v0, 31, v20
	v_add_u32_e32 v19, 0x100, v20
	v_lshrrev_b32_e32 v0, 28, v0
	v_ashrrev_i32_e32 v1, 31, v19
	v_add_u32_e32 v0, v20, v0
	v_lshrrev_b32_e32 v1, 28, v1
	v_ashrrev_i32_e32 v21, 4, v0
	v_add_u32_e32 v1, v19, v1
	v_and_b32_e32 v0, -16, v0
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, 0xffffff, v3
	v_cmp_gt_u32_e32 vcc, s4, v2
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v4
	v_sub_u32_e32 v16, v20, v0
	v_cndmask_b32_e32 v2, 0, v2, vcc
	v_cmp_gt_u32_e32 vcc, s4, v3
	v_mad_u64_u32 v[34:35], s[14:15], s6, v2, 0
	s_nop 0
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_mul_hi_i32 v2, v21, s23
	v_ashrrev_i32_e32 v35, 4, v1
	v_mad_u64_u32 v[32:33], s[6:7], s6, v3, 0
	v_lshrrev_b32_e32 v0, 31, v2
	v_ashrrev_i32_e32 v2, 11, v2
	v_mul_hi_i32 v3, v35, s23
	v_add_u32_e32 v0, v2, v0
	v_lshrrev_b32_e32 v2, 31, v3
	v_ashrrev_i32_e32 v3, 11, v3
	v_mul_i32_i24_e32 v0, 0x3000, v0
	v_add_u32_e32 v2, v3, v2
	v_sub_u32_e32 v36, v21, v0
	v_mul_i32_i24_e32 v0, 0x3000, v2
	v_sub_u32_e32 v38, v35, v0
	v_and_b32_e32 v0, -16, v1
	v_ashrrev_i32_e32 v37, 31, v36
	v_sub_u32_e32 v0, v19, v0
	v_mov_b32_e32 v1, v17
	v_ashrrev_i32_e32 v39, 31, v38
	v_mul_lo_u32 v2, s2, v26
	s_mov_b32 s14, 0x1be00
	v_mul_lo_u32 v40, v0, s33
	v_lshlrev_b32_e32 v6, 4, v2
	v_mad_i64_i32 v[2:3], s[6:7], v18, s14, v[16:17]
	v_mad_i64_i32 v[0:1], s[6:7], v18, s14, v[0:1]
	v_lshl_add_u64 v[30:31], v[36:37], 4, v[2:3]
	v_lshl_add_u64 v[28:29], v[38:39], 4, v[0:1]
	v_mul_lo_u32 v4, v36, s3
	v_mul_lo_u32 v22, v16, s33
	v_mul_lo_u32 v5, v38, s3
	v_lshlrev_b32_e32 v0, 2, v30
	v_lshlrev_b32_e32 v1, 2, v28
	v_add3_u32 v55, v22, v6, v4
	v_add3_u32 v45, v40, v6, v5
	v_add_u32_e32 v2, 0x1000, v0
	v_add_u32_e32 v3, 0x2000, v0
	v_add_u32_e32 v4, 0x3000, v0
	v_add_u32_e32 v5, 0x1000, v1
	v_add_u32_e32 v6, 0x2000, v1
	v_and_b32_e32 v16, -4, v55
	v_and_b32_e32 v25, -4, v45
	v_add_u32_e32 v29, 0x3000, v1
	buffer_load_dword v76, v0, s[36:39], 0 offen
	buffer_load_dword v68, v0, s[36:39], 0 offen offset:2048
	buffer_load_dword v64, v2, s[36:39], 0 offen
	buffer_load_dword v62, v2, s[36:39], 0 offen offset:2048
	buffer_load_dword v58, v3, s[36:39], 0 offen
	buffer_load_dword v54, v3, s[36:39], 0 offen offset:2048
	buffer_load_dword v50, v4, s[36:39], 0 offen
	buffer_load_dword v46, v4, s[36:39], 0 offen offset:2048
	buffer_load_dword v72, v1, s[36:39], 0 offen
	buffer_load_dword v70, v1, s[36:39], 0 offen offset:2048
	buffer_load_dword v66, v5, s[36:39], 0 offen
	buffer_load_dword v60, v5, s[36:39], 0 offen offset:2048
	buffer_load_dword v56, v6, s[36:39], 0 offen
	buffer_load_dword v52, v6, s[36:39], 0 offen offset:2048
	buffer_load_dword v48, v29, s[36:39], 0 offen
	buffer_load_dword v44, v29, s[36:39], 0 offen offset:2048
	buffer_load_dwordx4 v[12:15], v16, s[28:31], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[4:7], v16, s[28:31], 0 offen offset:16
	buffer_load_dwordx4 v[8:11], v25, s[28:31], 0 offen
	buffer_load_dwordx4 v[0:3], v25, s[28:31], 0 offen offset:16
	v_add_lshl_u32 v43, v34, v53, 2
	v_add_lshl_u32 v57, v32, v53, 2
	buffer_load_dwordx4 v[78:81], v43, s[40:43], 0 offen
	buffer_load_dwordx4 v[82:85], v57, s[40:43], 0 offen
	v_lshlrev_b32_e32 v16, 9, v23
	v_lshlrev_b32_e32 v23, 4, v23
	v_lshlrev_b32_e32 v25, 2, v53
	v_lshlrev_b32_e32 v37, 4, v41
	v_lshlrev_b32_e32 v39, 9, v41
	v_lshlrev_b32_e32 v31, 4, v26
	v_lshlrev_b32_e32 v29, 9, v27
	v_lshlrev_b32_e32 v27, 4, v27
	s_mov_b32 s3, 0x6f800
	v_bitop3_b32 v51, v16, v25, v23 bitop3:0xf6
	v_bitop3_b32 v47, v39, v31, v37 bitop3:0xf6
	v_lshlrev_b32_e32 v38, 6, v38
	v_lshlrev_b32_e32 v36, 6, v36
	v_bitop3_b32 v49, v29, v27, v25 bitop3:0xf6
	v_or_b32_e32 v16, 64, v31
	v_mul_lo_u32 v18, v18, s3
	v_or_b32_e32 v23, 0x80, v31
	v_or_b32_e32 v25, 0xc0, v31
	v_or_b32_e32 v27, 0x100, v31
	v_or_b32_e32 v29, 0x140, v31
	v_or_b32_e32 v41, 0x180, v31
	v_or_b32_e32 v42, 0x1c0, v31
	v_or_b32_e32 v59, v38, v31
	v_or_b32_e32 v61, v36, v31
	v_bitop3_b32 v133, v39, v16, v37 bitop3:0xf6
	v_add_u32_e32 v16, v18, v38
	v_bitop3_b32 v132, v39, v23, v37 bitop3:0xf6
	v_bitop3_b32 v33, v39, v25, v37 bitop3:0xf6
	v_bitop3_b32 v31, v39, v27, v37 bitop3:0xf6
	v_bitop3_b32 v29, v39, v29, v37 bitop3:0xf6
	v_bitop3_b32 v27, v39, v41, v37 bitop3:0xf6
	v_bitop3_b32 v25, v39, v42, v37 bitop3:0xf6
	v_add_u32_e32 v18, v18, v36
	v_mad_u64_u32 v[36:37], s[6:7], v59, s2, v[40:41]
	v_mad_u64_u32 v[38:39], s[2:3], v61, s2, v[22:23]
	v_lshl_add_u32 v16, v19, 2, v16
	v_lshlrev_b32_e32 v19, 6, v35
	v_lshl_add_u32 v18, v20, 2, v18
	v_sub_u32_e32 v40, v16, v19
	v_lshlrev_b32_e32 v16, 6, v21
	s_mov_b32 s2, 0xfff98000
	v_sub_u32_e32 v42, v18, v16
	v_add_u32_e32 v35, 0x400, v57
	v_add_u32_e32 v37, 0x400, v43
	s_mov_b32 s3, -1
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v21, v17
	s_waitcnt vmcnt(1)
	ds_write_b128 v51, v[78:81]
	s_waitcnt vmcnt(0)
	ds_write_b128 v49, v[82:85]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[84:87], v47
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	.p2align	5, , 4
.LBB0_2:
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v12, 4, v12
	v_pk_mul_f32 v[78:79], v[76:77], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[76:77], v[74:75] op_sel_hi:[0,1]
	v_and_b32_e32 v43, 0xffff0000, v79
	v_and_b32_e32 v57, 0xffff0000, v78
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v75, v43, v75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v57, v74 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[78:79], v[76:77], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[76:77], v[80:81] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v77
	v_and_b32_e32 v43, 0xffff0000, v76
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	ds_read_b128 v[96:99], v133
	ds_read_b128 v[100:103], v132
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[84:85], v[74:75], v[16:19]
	v_or_b32_sdwa v77, v12, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v76, v43, v78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[80:81], v[72:73], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[72:73], v[78:79] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v81
	v_and_b32_e32 v43, 0xffff0000, v80
	v_or_b32_sdwa v79, v12, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v78, v43, v78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[16:19], v[86:87], v[76:77], v[16:19]
	ds_read_b128 v[88:91], v33
	v_lshrrev_b32_e32 v8, 4, v8
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[20:23], v[84:85], v[78:79], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v39, s11, v38
	v_pk_mul_f32 v[74:75], v[72:73], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[72:73], v[82:83] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v73
	v_and_b32_e32 v12, 0xffff0000, v72
	v_or_b32_sdwa v73, v8, v75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v12, v74 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	ds_read_b128 v[74:77], v31
	v_pk_mul_f32 v[78:79], v[68:69], v[78:79] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[86:87], v[72:73], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_add_u32_e32 v41, 32, v39
	v_pk_mul_f32 v[72:73], v[68:69], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v79
	v_and_b32_e32 v12, 0xffff0000, v78
	v_and_b32_e32 v41, -4, v41
	v_or_b32_sdwa v73, v8, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v12, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v8, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dwordx4 v[20:23], v41, s[16:19], 0 offen
	v_pk_mul_f32 v[12:13], v[68:69], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[68:69], v[78:79] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[96:97], v[72:73], v[16:19]
	v_and_b32_e32 v8, 0xffff0000, v69
	v_and_b32_e32 v43, 0xffff0000, v68
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[78:79], v[70:71], v[78:79] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v8, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v43, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[68:69], v[70:71], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v79
	v_and_b32_e32 v43, 0xffff0000, v78
	v_or_b32_sdwa v69, v8, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v43, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v43, 4, v9
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[92:95], v[98:99], v[12:13], v[16:19]
	v_mul_f32_e64 v12, v70, v78
	v_mul_f32_e64 v13, v70, v79
	ds_read_b128 v[78:81], v29
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[16:19], v[96:97], v[68:69], v[82:85]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v13, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[70:71], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_mov_b32 s22, s18
	v_pk_mul_f32 v[12:13], v[64:65], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[98:99], v[8:9], v[16:19]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v13, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[64:65], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dwordx4 v[16:19], v41, s[16:19], 0 offen offset:16
	v_pk_mul_f32 v[12:13], v[64:65], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[64:65], v[72:73] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[82:85], v[100:101], v[8:9], v[92:95]
	v_and_b32_e32 v14, 0xffff0000, v65
	v_and_b32_e32 v57, 0xffff0000, v64
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[66:67], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v14, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v57, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[64:65], v[66:67], v[64:65] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v73
	v_and_b32_e32 v57, 0xffff0000, v72
	v_or_b32_sdwa v73, v14, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v57, v64 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v10, 4, v10
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[82:85], v[102:103], v[12:13], v[82:85]
	v_mul_f32_e64 v8, v66, v64
	v_mul_f32_e64 v9, v66, v65
	v_pk_mul_f32 v[12:13], v[66:67], v[86:87] op_sel_hi:[0,1]
	ds_read_b128 v[64:67], v27
	v_mfma_f32_16x16x16_bf16 v[68:71], v[100:101], v[72:73], v[68:71]
	v_and_b32_e32 v10, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v41, s2, v42
	v_pk_mul_f32 v[12:13], v[62:63], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[102:103], v[8:9], v[68:71]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[62:63], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v10, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[14:15], v[62:63], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[62:63], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v15
	v_or_b32_sdwa v12, v14, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[62:63], v[60:61], v[62:63] op_sel_hi:[0,1]
	v_add_u32_e32 v43, 0x6c000, v41
	s_mov_b32 s23, s19
	v_or_b32_sdwa v13, v10, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[14:15], v[60:61], v[14:15] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v63
	v_and_b32_e32 v57, 0xffff0000, v62
	buffer_load_dword v70, v43, s[20:23], 0 offen
	v_or_b32_sdwa v63, v10, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v57, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v57, 4, v11
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[8:11], v[88:89], v[8:9], v[82:85]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_mul_f32_e64 v72, v60, v14
	v_mul_f32_e64 v73, v60, v15
	v_mfma_f32_16x16x16_bf16 v[12:15], v[90:91], v[12:13], v[8:11]
	v_mul_f32_e64 v68, v60, v68
	v_mul_f32_e64 v69, v60, v69
	v_and_b32_e32 v57, 0xffff0000, v69
	v_and_b32_e32 v59, 0xffff0000, v68
	ds_read_b128 v[8:11], v25
	v_mfma_f32_16x16x16_bf16 v[60:63], v[88:89], v[62:63], v[92:95]
	v_or_b32_sdwa v69, v57, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v59, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[82:83], v[58:59], v[82:83] op_sel_hi:[0,1]
	v_and_b32_e32 v57, 0xffff0000, v83
	v_mfma_f32_16x16x16_bf16 v[60:63], v[90:91], v[68:69], v[60:63]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 4, v4
	v_pk_mul_f32 v[68:69], v[58:59], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v59, 0xffff0000, v82
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v69, v57, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v59, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[82:83], v[58:59], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[58:59], v[84:85] op_sel_hi:[0,1]
	buffer_load_dword v72, v43, s[20:23], 0 offen offset:2048
	v_and_b32_e32 v4, 0xffff0000, v59
	v_and_b32_e32 v57, 0xffff0000, v58
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[74:75], v[68:69], v[12:15]
	v_or_b32_sdwa v59, v4, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v58, v57, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[84:85], v[56:57], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[56:57], v[82:83] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v85
	v_and_b32_e32 v57, 0xffff0000, v84
	v_or_b32_sdwa v83, v4, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v82, v57, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[12:15], v[76:77], v[58:59], v[12:15]
	v_mul_f32_e64 v84, v56, v84
	v_mul_f32_e64 v85, v56, v85
	v_pk_mul_f32 v[68:69], v[56:57], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v69
	v_mfma_f32_16x16x16_bf16 v[56:59], v[74:75], v[82:83], v[60:63]
	v_and_b32_e32 v4, 0xffff0000, v68
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v43, 0x6d000, v41
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	s_nop 0
	v_or_b32_sdwa v61, v0, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v60, v4, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[62:63], v[54:55], v[62:63] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v63
	v_mfma_f32_16x16x16_bf16 v[56:59], v[76:77], v[60:61], v[56:59]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v4, 0xffff0000, v62
	v_pk_mul_f32 v[60:61], v[54:55], v[60:61] op_sel_hi:[0,1]
	v_or_b32_sdwa v61, v0, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v60, v4, v60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v0, 4, v5
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dword v76, v43, s[20:23], 0 offen
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[62:63], v[54:55], v[62:63] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[78:79], v[60:61], v[12:15]
	v_mul_f32_e64 v4, v54, v4
	v_mul_f32_e64 v5, v54, v5
	v_and_b32_e32 v0, 0xffff0000, v63
	v_and_b32_e32 v54, 0xffff0000, v62
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[52:53], v[68:69] op_sel_hi:[0,1]
	v_or_b32_sdwa v5, v0, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v4, v54, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[62:63], v[52:53], v[62:63] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v69
	v_and_b32_e32 v54, 0xffff0000, v68
	v_or_b32_sdwa v63, v0, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v54, v62 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[12:15], v[80:81], v[4:5], v[12:15]
	v_lshrrev_b32_e32 v54, 4, v1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[56:59], v[78:79], v[62:63], v[56:59]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v4, v52, v60
	v_mul_f32_e64 v5, v52, v61
	v_pk_mul_f32 v[0:1], v[52:53], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v5, 0xffff0000, v5
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[48:49], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[50:51], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[80:81], v[0:1], v[56:59]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v5, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[50:51], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v6, 4, v6
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dword v60, v43, s[20:23], 0 offen offset:2048
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[62:63], v[50:51], v[62:63] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[64:65], v[0:1], v[12:15]
	v_mul_f32_e64 v4, v50, v4
	v_mul_f32_e64 v5, v50, v5
	v_and_b32_e32 v6, 0xffff0000, v63
	v_and_b32_e32 v50, 0xffff0000, v62
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v5, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v4, v50, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[62:63], v[48:49], v[62:63] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v69
	v_and_b32_e32 v50, 0xffff0000, v68
	v_or_b32_sdwa v63, v6, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v50, v62 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v2, 4, v2
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[12:15], v[66:67], v[4:5], v[12:15]
	v_mul_f32_e64 v0, v48, v0
	v_mul_f32_e64 v1, v48, v1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[56:59], v[64:65], v[62:63], v[56:59]
	v_mul_f32_e64 v4, v48, v68
	v_mul_f32_e64 v5, v48, v69
	v_and_b32_e32 v1, 0xffff0000, v1
	v_and_b32_e32 v0, 0xffff0000, v0
	v_or_b32_sdwa v1, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v43, 0x6e000, v41
	v_pk_mul_f32 v[4:5], v[46:47], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[66:67], v[0:1], v[56:59]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v2, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[46:47], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v2, 4, v7
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v6, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v7, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[6:7], v[46:47], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[46:47], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v6
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v2, 0xffff0000, v7
	v_or_b32_sdwa v4, v6, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v6, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v7, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[44:45], v[58:59] op_sel_hi:[0,1]
	v_or_b32_sdwa v5, v2, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[6:7], v[44:45], v[6:7] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v59
	v_and_b32_e32 v46, 0xffff0000, v58
	buffer_load_dword v56, v43, s[20:23], 0 offen
	v_or_b32_sdwa v59, v2, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v58, v46, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v6, 4, v3
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[0:1], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v66, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v67, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[4:5], v[0:3]
	s_nop 1
	v_mul_f32_e64 v12, v44, v12
	v_mul_f32_e64 v13, v44, v13
	v_pk_mul_f32 v[14:15], v[44:45], v[66:67] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[58:59], v[62:65]
	v_and_b32_e32 v8, 0xffff0000, v12
	v_or_b32_sdwa v9, v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v57, s11, v36
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[8:9], v[0:3]
	v_add_u32_e32 v8, 0x6f000, v41
	buffer_load_dword v58, v8, s[20:23], 0 offen
	buffer_load_dword v44, v8, s[20:23], 0 offen offset:2048
	v_add_u32_e32 v8, 32, v57
	buffer_load_dword v62, v43, s[20:23], 0 offen offset:2048
	v_and_b32_e32 v43, -4, v8
	buffer_load_dwordx4 v[12:15], v43, s[16:19], 0 offen
	buffer_load_dwordx4 v[8:11], v43, s[16:19], 0 offen offset:16
	v_add_u32_e32 v43, s2, v40
	v_add_u32_e32 v46, 0x6c000, v43
	buffer_load_dword v68, v46, s[20:23], 0 offen
	buffer_load_dword v66, v46, s[20:23], 0 offen offset:2048
	v_add_u32_e32 v46, 0x6d000, v43
	v_add_u32_e32 v48, 0x6e000, v43
	v_add_u32_e32 v59, 0x6f000, v43
	buffer_load_dword v64, v46, s[20:23], 0 offen
	buffer_load_dword v54, v46, s[20:23], 0 offen offset:2048
	buffer_load_dword v50, v48, s[20:23], 0 offen
	buffer_load_dword v52, v59, s[20:23], 0 offen
	s_nop 0
	buffer_load_dword v46, v48, s[20:23], 0 offen offset:2048
	s_nop 0
	buffer_load_dword v48, v59, s[20:23], 0 offen offset:2048
	s_waitcnt vmcnt(19)
	v_lshrrev_b32_e32 v71, 4, v21
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v108, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v109, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v61, 4, v16
	v_lshrrev_b32_e32 v65, 4, v17
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v122, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v123, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v124, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v125, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v126, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v127, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v141, 4, v12
	v_lshrrev_b32_e32 v140, 4, v13
	v_lshrrev_b32_e32 v139, 4, v14
	v_lshrrev_b32_e32 v134, 4, v15
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v128, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v129, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v130, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v131, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v138, 4, v8
	v_lshrrev_b32_e32 v137, 4, v9
	v_pk_mul_f32 v[8:9], v[70:71], v[74:75] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v73, 4, v20
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v90, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v91, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v92, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v93, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v98, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v99, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v100, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v101, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v136, 4, v10
	v_lshrrev_b32_e32 v135, 4, v11
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[70:71], v[84:85] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v88, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v89, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v111, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v110, v11, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[72:73], v[88:89] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[72:73], v[86:87] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v85, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v84, v11, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[76:77], v[96:97] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v94, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v95, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[76:77], v[94:95] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v102, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v103, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v67, 4, v22
	v_lshrrev_b32_e32 v69, 4, v23
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v63, 4, v18
	v_lshrrev_b32_e32 v59, 4, v19
	v_or_b32_sdwa v19, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v18, v11, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[60:61], v[104:105] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v74, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[60:61], v[102:103] op_sel_hi:[0,1]
	v_or_b32_sdwa v11, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v74, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[56:57], v[108:109] op_sel_hi:[0,1]
	v_and_b32_e32 v74, 0xffff0000, v9
	v_and_b32_e32 v75, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[56:57], v[106:107] op_sel_hi:[0,1]
	v_or_b32_sdwa v9, v74, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v75, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[74:75], v[62:63], v[114:115] op_sel_hi:[0,1]
	v_and_b32_e32 v77, 0xffff0000, v75
	v_and_b32_e32 v86, 0xffff0000, v74
	v_pk_mul_f32 v[74:75], v[62:63], v[112:113] op_sel_hi:[0,1]
	s_waitcnt vmcnt(7)
	v_pk_mul_f32 v[82:83], v[68:69], v[82:83] op_sel_hi:[0,1]
	v_or_b32_sdwa v75, v77, v75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v77, 0xffff0000, v83
	v_and_b32_e32 v82, 0xffff0000, v82
	v_pk_mul_f32 v[16:17], v[68:69], v[16:17] op_sel_hi:[0,1]
	v_or_b32_sdwa v95, v77, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v94, v82, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(6)
	v_pk_mul_f32 v[16:17], v[66:67], v[118:119] op_sel_hi:[0,1]
	v_and_b32_e32 v77, 0xffff0000, v17
	v_and_b32_e32 v82, 0xffff0000, v16
	v_pk_mul_f32 v[16:17], v[66:67], v[116:117] op_sel_hi:[0,1]
	v_or_b32_sdwa v89, v77, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v88, v82, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(5)
	v_pk_mul_f32 v[16:17], v[64:65], v[122:123] op_sel_hi:[0,1]
	v_and_b32_e32 v77, 0xffff0000, v17
	v_and_b32_e32 v82, 0xffff0000, v16
	v_pk_mul_f32 v[16:17], v[64:65], v[120:121] op_sel_hi:[0,1]
	v_or_b32_sdwa v74, v86, v74 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v87, v77, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v86, v82, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(4)
	v_pk_mul_f32 v[16:17], v[54:55], v[126:127] op_sel_hi:[0,1]
	v_and_b32_e32 v77, 0xffff0000, v17
	v_and_b32_e32 v82, 0xffff0000, v16
	v_pk_mul_f32 v[16:17], v[54:55], v[124:125] op_sel_hi:[0,1]
	v_or_b32_sdwa v83, v77, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v82, v82, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[16:17], v[50:51], v[130:131] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[50:51], v[128:129] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v17, v17, v97 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v16, v96 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v102, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v103, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[96:97], v[70:71], v[96:97] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[70:71], v[102:103] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[104:105], v[72:73], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[72:73], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v72, 0xffff0000, v103
	v_and_b32_e32 v73, 0xffff0000, v102
	v_and_b32_e32 v71, 0xffff0000, v71
	v_and_b32_e32 v70, 0xffff0000, v70
	v_or_b32_sdwa v103, v72, v97 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v102, v73, v96 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v73, v71, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v70, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[70:71], v[76:77], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[76:77], v[106:107] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v67, 0xffff0000, v71
	v_and_b32_e32 v69, 0xffff0000, v70
	s_barrier
	ds_read_b128 v[156:159], v47 offset:8192
	ds_read_b128 v[144:147], v33 offset:8192
	v_or_b32_sdwa v71, v67, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v69, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v141 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v141 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v141 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v141 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	ds_read_b128 v[152:155], v133 offset:8192
	ds_read_b128 v[148:151], v132 offset:8192
	v_pk_mul_f32 v[104:105], v[68:69], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[68:69], v[106:107] op_sel_hi:[0,1]
	v_and_b32_e32 v67, 0xffff0000, v69
	v_and_b32_e32 v68, 0xffff0000, v68
	v_or_b32_sdwa v69, v67, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v68, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v140 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v140 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v140 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v140 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[156:157], v[94:95], v[0:3]
	v_mul_f32_e64 v104, v66, v104
	v_mul_f32_e64 v105, v66, v105
	v_pk_mul_f32 v[66:67], v[66:67], v[106:107] op_sel_hi:[0,1]
	v_and_b32_e32 v67, 0xffff0000, v67
	v_and_b32_e32 v66, 0xffff0000, v66
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v139 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v139 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v67, v67, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v66, v66, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v139 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v139 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[106:107], v[64:65], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[64:65], v[104:105] op_sel_hi:[0,1]
	v_and_b32_e32 v64, 0xffff0000, v107
	v_and_b32_e32 v106, 0xffff0000, v106
	v_or_b32_sdwa v117, v64, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v116, v106, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[104:107], v[156:157], v[110:111], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v108, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v109, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v120, v60, v96
	v_mul_f32_e64 v121, v60, v97
	v_pk_mul_f32 v[60:61], v[60:61], v[76:77] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v163, 64, v57
	v_pk_mul_f32 v[4:5], v[56:57], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[56:57], v[112:113] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v56, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v57, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[94:95], v[62:63], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[62:63], v[64:65] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[158:159], v[102:103], v[104:107]
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[112:113], v[46:47], v[14:15] op_sel_hi:[0,1]
	ds_read_b128 v[140:143], v31 offset:8192
	v_pk_mul_f32 v[102:103], v[44:45], v[78:79] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[158:159], v[68:69], v[0:3]
	v_mul_f32_e64 v106, v46, v12
	v_mul_f32_e64 v107, v46, v13
	v_pk_mul_f32 v[104:105], v[58:59], v[20:21] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v138 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[152:153], v[84:85], v[62:65]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v138 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v114, v52, v92
	v_mul_f32_e64 v115, v52, v93
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[78:79], v[48:49], v[98:99] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[152:153], v[88:89], v[0:3]
	v_mul_f32_e64 v92, v58, v118
	v_mul_f32_e64 v93, v58, v119
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v122, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v156, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[12:15], v[154:155], v[72:73], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v123, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v157, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v108, v58, v22
	v_mul_f32_e64 v109, v58, v23
	v_mfma_f32_16x16x16_bf16 v[0:3], v[154:155], v[66:67], v[0:3]
	v_mul_f32_e64 v154, v50, v20
	v_mul_f32_e64 v155, v50, v21
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v137 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v137 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[148:149], v[18:19], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v137 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v137 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v98, v46, v20
	v_mul_f32_e64 v99, v46, v21
	v_mfma_f32_16x16x16_bf16 v[0:3], v[148:149], v[86:87], v[0:3]
	v_mul_f32_e64 v118, v46, v18
	v_mul_f32_e64 v119, v46, v19
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[12:15], v[150:151], v[70:71], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v136 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v138 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[150:151], v[116:117], v[0:3]
	v_mul_f32_e64 v84, v52, v18
	v_mul_f32_e64 v85, v52, v19
	v_pk_mul_f32 v[88:89], v[52:53], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v18, 0xffff0000, v61
	v_mfma_f32_16x16x16_bf16 v[10:13], v[144:145], v[10:11], v[12:15]
	v_and_b32_e32 v20, 0xffff0000, v60
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v138 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[138:139], v[54:55], v[156:157] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[144:145], v[82:83], v[0:3]
	v_or_b32_sdwa v19, v18, v121 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v18, v20, v120 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[122:123], v[54:55], v[122:123] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121], v[48:49], v[20:21] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[146:147], v[18:19], v[10:13]
	buffer_load_dwordx4 v[124:127], v37, s[24:27], 0 offen
	buffer_load_dwordx4 v[128:131], v35, s[24:27], 0 offen
	v_add_u32_e32 v39, 64, v39
	v_and_b32_e32 v10, 0xffff0000, v139
	v_and_b32_e32 v12, 0xffff0000, v138
	v_or_b32_sdwa v11, v10, v123 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v12, v122 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v160, 0x70000, v41
	v_add_u32_e32 v161, 0x71000, v41
	v_mfma_f32_16x16x16_bf16 v[0:3], v[146:147], v[10:11], v[0:3]
	v_add_u32_e32 v162, 0x72000, v41
	v_add_u32_e32 v41, 0x73000, v41
	v_add_u32_e32 v164, 0x70000, v43
	v_add_u32_e32 v165, 0x71000, v43
	v_add_u32_e32 v77, 0x72000, v43
	v_add_u32_e32 v43, 0x73000, v43
	v_pk_mul_f32 v[22:23], v[50:51], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[44:45], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[52:53], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[48:49], v[100:101] op_sel_hi:[0,1]
	v_and_b32_e32 v39, -4, v39
	buffer_load_dword v76, v160, s[20:23], 0 offen
	buffer_load_dword v68, v160, s[20:23], 0 offen offset:2048
	buffer_load_dword v50, v41, s[20:23], 0 offen
	buffer_load_dword v46, v41, s[20:23], 0 offen offset:2048
	buffer_load_dword v52, v77, s[20:23], 0 offen offset:2048
	v_pk_mul_f32 v[80:81], v[58:59], v[56:57] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v152, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v156, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v153, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v157, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dword v56, v77, s[20:23], 0 offen
	v_pk_mul_f32 v[86:87], v[44:45], v[152:153] op_sel_hi:[0,1]
	v_pk_mul_f32 v[116:117], v[44:45], v[156:157] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	buffer_load_dword v44, v43, s[20:23], 0 offen offset:2048
	v_pk_mul_f32 v[82:83], v[48:49], v[14:15] op_sel_hi:[0,1]
	buffer_load_dword v48, v43, s[20:23], 0 offen
	v_and_b32_e32 v41, -4, v163
	v_and_b32_e32 v7, 0xffff0000, v7
	v_and_b32_e32 v6, 0xffff0000, v6
	ds_read_b128 v[144:147], v29 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[134:137], v[140:141], v[8:9], v[18:21]
	v_and_b32_e32 v8, 0xffff0000, v23
	v_and_b32_e32 v9, 0xffff0000, v22
	buffer_load_dword v64, v161, s[20:23], 0 offen
	buffer_load_dword v62, v161, s[20:23], 0 offen offset:2048
	buffer_load_dword v58, v162, s[20:23], 0 offen
	buffer_load_dword v54, v162, s[20:23], 0 offen offset:2048
	buffer_load_dword v72, v164, s[20:23], 0 offen
	buffer_load_dword v70, v164, s[20:23], 0 offen offset:2048
	buffer_load_dword v66, v165, s[20:23], 0 offen
	buffer_load_dword v60, v165, s[20:23], 0 offen offset:2048
	buffer_load_dwordx4 v[12:15], v39, s[16:19], 0 offen
	v_or_b32_sdwa v139, v7, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v138, v6, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[4:7], v39, s[16:19], 0 offen offset:16
	v_mfma_f32_16x16x16_bf16 v[20:23], v[140:141], v[16:17], v[0:3]
	v_or_b32_sdwa v123, v8, v155 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v122, v9, v154 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[8:11], v41, s[16:19], 0 offen
	buffer_load_dwordx4 v[0:3], v41, s[16:19], 0 offen offset:16
	v_mfma_f32_16x16x16_bf16 v[16:19], v[142:143], v[138:139], v[134:137]
	ds_read_b128 v[138:141], v27 offset:8192
	v_and_b32_e32 v39, 0xffff0000, v109
	v_and_b32_e32 v41, 0xffff0000, v108
	v_mfma_f32_16x16x16_bf16 v[20:23], v[142:143], v[122:123], v[20:23]
	v_and_b32_e32 v59, 0xffff0000, v113
	v_and_b32_e32 v61, 0xffff0000, v112
	v_and_b32_e32 v67, 0xffff0000, v101
	v_and_b32_e32 v69, 0xffff0000, v100
	v_or_b32_sdwa v101, v39, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v100, v41, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v105, v59, v107 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[144:145], v[74:75], v[16:19]
	v_or_b32_sdwa v104, v61, v106 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v43, 0xffff0000, v111
	v_and_b32_e32 v39, 0xffff0000, v97
	v_mfma_f32_16x16x16_bf16 v[20:23], v[144:145], v[104:105], v[20:23]
	v_and_b32_e32 v41, 0xffff0000, v96
	v_or_b32_sdwa v103, v43, v103 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v43, 0xffff0000, v119
	v_or_b32_sdwa v75, v39, v95 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v41, v94 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v39, 0xffff0000, v118
	v_and_b32_e32 v63, 0xffff0000, v115
	v_mfma_f32_16x16x16_bf16 v[16:19], v[146:147], v[74:75], v[16:19]
	v_or_b32_sdwa v75, v43, v99 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v39, v98 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v65, 0xffff0000, v114
	ds_read_b128 v[134:137], v25 offset:8192
	v_mfma_f32_16x16x16_bf16 v[20:23], v[146:147], v[74:75], v[20:23]
	v_or_b32_sdwa v75, v63, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v65, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v39, 0xffff0000, v93
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[138:139], v[100:101], v[16:19]
	v_and_b32_e32 v41, 0xffff0000, v92
	v_and_b32_e32 v43, 0xffff0000, v89
	v_and_b32_e32 v57, 0xffff0000, v110
	v_mfma_f32_16x16x16_bf16 v[20:23], v[138:139], v[74:75], v[20:23]
	v_or_b32_sdwa v75, v39, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v41, v80 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v39, 0xffff0000, v88
	v_or_b32_sdwa v102, v57, v102 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[16:19], v[140:141], v[74:75], v[16:19]
	v_or_b32_sdwa v75, v43, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v39, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v79, v67, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v78, v69, v78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[140:141], v[74:75], v[20:23]
	v_and_b32_e32 v39, 0xffff0000, v117
	v_and_b32_e32 v41, 0xffff0000, v116
	v_and_b32_e32 v43, 0xffff0000, v121
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[134:135], v[102:103], v[16:19]
	v_and_b32_e32 v57, 0xffff0000, v120
	v_or_b32_sdwa v75, v39, v87 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v41, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[134:135], v[78:79], v[20:23]
	v_or_b32_sdwa v79, v43, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v78, v57, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(21)
	ds_write_b128 v51, v[124:127]
	s_waitcnt vmcnt(20)
	ds_write_b128 v49, v[128:131]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[84:87], v47
	v_mfma_f32_16x16x16_bf16 v[16:19], v[136:137], v[74:75], v[16:19]
	s_add_u32 s2, s2, 0x8000
	s_addc_u32 s3, s3, 0
	s_add_i32 s11, s11, 64
	v_mfma_f32_16x16x16_bf16 v[20:23], v[136:137], v[78:79], v[20:23]
	v_add_u32_e32 v35, 0x400, v35
	s_cmp_lg_u64 s[2:3], 0
	v_add_u32_e32 v37, 0x400, v37
	s_cbranch_scc1 .LBB0_2
	s_mul_i32 s2, s4, s5
	s_and_b32 s13, s13, 0xffff
	s_lshl_b32 s14, s2, 4
	s_add_u32 s2, s0, 0xffffff00
	s_addc_u32 s3, s1, -1
	s_lshr_b64 s[6:7], s[2:3], 1
	v_add_u32_e32 v34, s6, v34
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v57, v53, v34, 2
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[36:37], v[76:77], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[76:77], v[34:35] op_sel_hi:[0,1]
	v_and_b32_e32 v37, 0xffff0000, v37
	v_and_b32_e32 v36, 0xffff0000, v36
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v35, v37, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v34, v36, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[38:39], v[76:77], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[76:77], v[36:37] op_sel_hi:[0,1]
	v_and_b32_e32 v38, 0xffff0000, v38
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	ds_read_b128 v[88:91], v133
	ds_read_b128 v[92:95], v132
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[84:85], v[34:35], v[16:19]
	v_and_b32_e32 v12, 0xffff0000, v39
	v_or_b32_sdwa v36, v38, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[40:41], v[72:73], v[40:41] op_sel_hi:[0,1]
	v_or_b32_sdwa v37, v12, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[38:39], v[72:73], v[38:39] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v41
	v_and_b32_e32 v40, 0xffff0000, v40
	v_or_b32_sdwa v39, v12, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v38, v40, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v8, 4, v8
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[34:37], v[86:87], v[36:37], v[16:19]
	v_mul_f32_e64 v40, v72, v40
	v_mul_f32_e64 v41, v72, v41
	v_pk_mul_f32 v[42:43], v[72:73], v[42:43] op_sel_hi:[0,1]
	ds_read_b128 v[72:75], v33
	v_mfma_f32_16x16x16_bf16 v[16:19], v[84:85], v[38:39], v[20:23]
	v_and_b32_e32 v8, 0xffff0000, v43
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v12, 0xffff0000, v42
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_nop 0
	v_or_b32_sdwa v21, v8, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v8, s6, v32
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[76:77], v[68:69], v[76:77] op_sel_hi:[0,1]
	v_or_b32_sdwa v20, v12, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_lshl_u32 v32, v53, v8, 2
	v_pk_mul_f32 v[42:43], v[68:69], v[42:43] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v77
	v_and_b32_e32 v12, 0xffff0000, v76
	ds_read_b128 v[38:41], v31
	v_mfma_f32_16x16x16_bf16 v[20:23], v[86:87], v[20:21], v[16:19]
	v_or_b32_sdwa v43, v8, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v42, v12, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v8, 4, v13
	buffer_load_dwordx4 v[16:19], v57, s[24:27], 0 offen
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[34:37], v[88:89], v[42:43], v[34:37]
	v_mul_f32_e64 v12, v68, v12
	v_mul_f32_e64 v13, v68, v13
	v_pk_mul_f32 v[68:69], v[68:69], v[76:77] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v8, 0xffff0000, v69
	v_and_b32_e32 v53, 0xffff0000, v68
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[76:77], v[70:71], v[76:77] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v8, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v53, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[68:69], v[70:71], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v77
	v_and_b32_e32 v53, 0xffff0000, v76
	v_or_b32_sdwa v81, v8, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v80, v53, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v53, 4, v9
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[76:79], v[90:91], v[12:13], v[34:37]
	v_mul_f32_e64 v8, v70, v8
	v_mul_f32_e64 v9, v70, v9
	v_pk_mul_f32 v[12:13], v[70:71], v[68:69] op_sel_hi:[0,1]
	ds_read_b128 v[68:71], v29
	v_mfma_f32_16x16x16_bf16 v[20:23], v[88:89], v[80:81], v[20:23]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_lshr_b64 s[6:7], s[2:3], 3
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_lshl_b32 s1, s2, 4
	v_mfma_f32_16x16x16_bf16 v[20:23], v[90:91], v[8:9], v[20:23]
	v_add_u32_e32 v8, s6, v55
	buffer_load_dwordx4 v[34:37], v32, s[24:27], 0 offen
	v_and_b32_e32 v32, -4, v8
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[12:13], v[64:65], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[64:65], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[42:43], v[64:65], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[64:65], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v42, 0xffff0000, v42
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[76:79], v[92:93], v[8:9], v[76:79]
	v_and_b32_e32 v14, 0xffff0000, v43
	v_or_b32_sdwa v12, v42, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[64:65], v[66:67], v[64:65] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v14, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[42:43], v[66:67], v[42:43] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v65
	v_and_b32_e32 v53, 0xffff0000, v64
	v_or_b32_sdwa v43, v14, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v42, v53, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v10, 4, v10
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[76:79], v[94:95], v[12:13], v[76:79]
	v_mul_f32_e64 v8, v66, v64
	v_mul_f32_e64 v9, v66, v65
	v_pk_mul_f32 v[12:13], v[66:67], v[80:81] op_sel_hi:[0,1]
	ds_read_b128 v[64:67], v27
	v_mfma_f32_16x16x16_bf16 v[20:23], v[92:93], v[42:43], v[20:23]
	v_and_b32_e32 v10, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v53, 4, v11
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[80:83], v[94:95], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v12, v62, v12
	v_mul_f32_e64 v13, v62, v13
	v_pk_mul_f32 v[8:9], v[62:63], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v10, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[14:15], v[62:63], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[62:63], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v14
	v_and_b32_e32 v10, 0xffff0000, v15
	v_or_b32_sdwa v12, v14, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[42:43], v[60:61], v[42:43] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v10, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[14:15], v[60:61], v[14:15] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v43
	buffer_load_dwordx4 v[20:23], v32, s[16:19], 0 offen
	v_or_b32_sdwa v15, v10, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[8:11], v[72:73], v[8:9], v[76:79]
	v_and_b32_e32 v42, 0xffff0000, v42
	v_or_b32_sdwa v14, v42, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[76:79], v[74:75], v[12:13], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v42, v60, v42
	v_mul_f32_e64 v43, v60, v43
	v_pk_mul_f32 v[60:61], v[60:61], v[62:63] op_sel_hi:[0,1]
	v_and_b32_e32 v53, 0xffff0000, v61
	s_nop 0
	ds_read_b128 v[8:11], v25
	v_mfma_f32_16x16x16_bf16 v[12:15], v[72:73], v[14:15], v[80:83]
	v_and_b32_e32 v55, 0xffff0000, v60
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v43, v53, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v42, v55, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[72:73], v[58:59], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v53, 0xffff0000, v73
	v_mfma_f32_16x16x16_bf16 v[60:63], v[74:75], v[42:43], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v55, 0xffff0000, v72
	v_lshrrev_b32_e32 v4, 4, v4
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[42:43], v[58:59], v[42:43] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[58:59], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[58:59], v[74:75] op_sel_hi:[0,1]
	v_or_b32_sdwa v43, v53, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v4, 0xffff0000, v59
	v_and_b32_e32 v53, 0xffff0000, v58
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v59, v4, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v58, v53, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[74:75], v[56:57], v[74:75] op_sel_hi:[0,1]
	v_or_b32_sdwa v42, v55, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[72:73], v[56:57], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v75
	v_and_b32_e32 v53, 0xffff0000, v74
	buffer_load_dwordx4 v[12:15], v32, s[16:19], 0 offen offset:16
	v_or_b32_sdwa v81, v4, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v80, v53, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[72:75], v[38:39], v[42:43], v[76:79]
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[72:75], v[40:41], v[58:59], v[72:75]
	s_nop 0
	v_mul_f32_e64 v76, v56, v82
	v_mul_f32_e64 v77, v56, v83
	v_pk_mul_f32 v[42:43], v[56:57], v[42:43] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v43
	v_mfma_f32_16x16x16_bf16 v[56:59], v[38:39], v[80:81], v[60:63]
	v_and_b32_e32 v4, 0xffff0000, v42
	v_or_b32_sdwa v39, v0, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v38, v4, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_and_b32 s1, s1, 0xfffffe00
	v_add_lshl_u32 v32, v30, s1, 2
	v_mfma_f32_16x16x16_bf16 v[40:43], v[40:41], v[38:39], v[56:59]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v56, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v57, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	buffer_load_dword v38, v32, s[20:23], 0 offen
	s_nop 1
	v_pk_mul_f32 v[58:59], v[54:55], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[54:55], v[56:57] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v59
	v_and_b32_e32 v4, 0xffff0000, v58
	v_or_b32_sdwa v57, v0, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v56, v4, v56 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v0, 4, v5
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_lshl_b32 s0, s0, 4
	v_pk_mul_f32 v[4:5], v[54:55], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[54:55], v[58:59] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v0, 0xffff0000, v55
	v_and_b32_e32 v39, 0xffff0000, v54
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v54, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v55, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[52:53], v[58:59] op_sel_hi:[0,1]
	v_or_b32_sdwa v5, v0, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v4, v39, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[54:55], v[52:53], v[54:55] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v59
	v_and_b32_e32 v39, 0xffff0000, v58
	v_or_b32_sdwa v59, v0, v55 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v58, v39, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[54:57], v[68:69], v[56:57], v[72:75]
	v_lshrrev_b32_e32 v39, 4, v1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[54:57], v[70:71], v[4:5], v[54:57]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_and_b32 s0, s0, 0xfffffe00
	v_mfma_f32_16x16x16_bf16 v[40:43], v[68:69], v[58:59], v[40:43]
	v_mul_f32_e64 v4, v52, v60
	v_mul_f32_e64 v5, v52, v61
	v_pk_mul_f32 v[0:1], v[52:53], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v5, 0xffff0000, v5
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s2, s0, 0xfffff200
	v_pk_mul_f32 v[4:5], v[50:51], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[70:71], v[0:1], v[40:43]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v5, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[50:51], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_lshrrev_b32_e32 v6, 4, v6
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[40:41], v[50:51], v[40:41] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v32, v30, s2, 2
	v_pk_mul_f32 v[4:5], v[50:51], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v41
	v_and_b32_e32 v39, 0xffff0000, v40
	v_pk_mul_f32 v[52:53], v[48:49], v[52:53] op_sel_hi:[0,1]
	buffer_load_dword v42, v32, s[20:23], 0 offen
	v_or_b32_sdwa v5, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v4, v39, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v6, 0xffff0000, v53
	v_and_b32_e32 v39, 0xffff0000, v52
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[52:55], v[64:65], v[0:1], v[54:57]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v2, 4, v2
	v_pk_mul_f32 v[40:41], v[48:49], v[40:41] op_sel_hi:[0,1]
	v_or_b32_sdwa v41, v6, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v40, v39, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[52:55], v[66:67], v[4:5], v[52:55]
	v_mul_f32_e64 v0, v48, v0
	v_mul_f32_e64 v1, v48, v1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[56:59], v[64:65], v[40:41], v[58:61]
	v_mul_f32_e64 v4, v48, v62
	v_mul_f32_e64 v5, v48, v63
	v_and_b32_e32 v1, 0xffff0000, v1
	v_and_b32_e32 v0, 0xffff0000, v0
	v_or_b32_sdwa v1, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s3, s0, 0xfffff400
	v_pk_mul_f32 v[4:5], v[46:47], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[66:67], v[0:1], v[56:59]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v2, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[46:47], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v2, 4, v7
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v6, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v7, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[6:7], v[46:47], v[6:7] op_sel_hi:[0,1]
	s_waitcnt vmcnt(5)
	ds_write_b128 v51, v[16:19] offset:8192
	v_pk_mul_f32 v[4:5], v[46:47], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v6
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v2, 0xffff0000, v7
	v_or_b32_sdwa v4, v6, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v6, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v7, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[16:17], v[44:45], v[16:17] op_sel_hi:[0,1]
	v_add_lshl_u32 v32, v30, s3, 2
	v_or_b32_sdwa v5, v2, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[6:7], v[44:45], v[6:7] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	buffer_load_dword v50, v32, s[20:23], 0 offen
	v_or_b32_sdwa v17, v2, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v16, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v6, 4, v3
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[0:1], v[52:55]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[4:5], v[0:3]
	v_mul_f32_e64 v40, v44, v40
	v_mul_f32_e64 v41, v44, v41
	v_pk_mul_f32 v[18:19], v[44:45], v[18:19] op_sel_hi:[0,1]
	v_and_b32_e32 v39, 0xffff0000, v41
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[16:17], v[56:59]
	v_and_b32_e32 v8, 0xffff0000, v40
	v_or_b32_sdwa v9, v39, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_add_i32 s7, s0, 0xfffff800
	s_add_i32 s11, s0, 0xfffffa00
	v_mfma_f32_16x16x16_bf16 v[8:11], v[10:11], v[8:9], v[0:3]
	s_add_i32 s15, s0, 0xfffffc00
	s_add_i32 s5, s0, 0xfffff600
	s_addk_i32 s0, 0xfe00
	v_add_lshl_u32 v0, v30, s7, 2
	buffer_load_dword v48, v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v30, s11, 2
	buffer_load_dword v54, v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v30, s15, 2
	buffer_load_dword v56, v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, v30, s0, 2
	buffer_load_dword v68, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, s6, v45
	v_add_lshl_u32 v32, v30, s5, 2
	v_and_b32_e32 v30, -4, v0
	buffer_load_dwordx4 v[16:19], v30, s[16:19], 0 offen
	buffer_load_dwordx4 v[0:3], v30, s[16:19], 0 offen offset:16
	v_add_lshl_u32 v30, v28, s1, 2
	buffer_load_dword v66, v30, s[20:23], 0 offen
	v_add_lshl_u32 v30, v28, s2, 2
	buffer_load_dword v64, v30, s[20:23], 0 offen
	v_add_lshl_u32 v30, v28, s3, 2
	buffer_load_dword v62, v30, s[20:23], 0 offen
	v_add_lshl_u32 v30, v28, s5, 2
	buffer_load_dword v60, v30, s[20:23], 0 offen
	v_add_lshl_u32 v30, v28, s7, 2
	s_waitcnt vmcnt(15)
	ds_write_b128 v49, v[34:37] offset:8192
	buffer_load_dword v34, v30, s[20:23], 0 offen
	v_add_lshl_u32 v30, v28, s11, 2
	buffer_load_dword v52, v32, s[20:23], 0 offen
	s_nop 0
	buffer_load_dword v32, v30, s[20:23], 0 offen
	v_add_lshl_u32 v30, v28, s15, 2
	v_add_lshl_u32 v28, v28, s0, 2
	buffer_load_dword v30, v30, s[20:23], 0 offen
	s_mov_b32 s15, 0x27000
	buffer_load_dword v28, v28, s[20:23], 0 offen
	s_waitcnt vmcnt(19)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v20, 4, v20
	s_waitcnt vmcnt(17)
	v_pk_mul_f32 v[40:41], v[38:39], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[38:39], v[36:37] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v41
	v_and_b32_e32 v39, 0xffff0000, v40
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v36, v39, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[40:41], v[38:39], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[38:39], v[44:45] op_sel_hi:[0,1]
	v_or_b32_sdwa v37, v35, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v20, 0xffff0000, v39
	v_and_b32_e32 v35, 0xffff0000, v38
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v39, v20, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v38, v35, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(16)
	v_pk_mul_f32 v[44:45], v[42:43], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[42:43], v[40:41] op_sel_hi:[0,1]
	v_and_b32_e32 v20, 0xffff0000, v45
	v_and_b32_e32 v35, 0xffff0000, v44
	v_or_b32_sdwa v41, v20, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v40, v35, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v35, 4, v21
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[20:21], v[42:43], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[42:43], v[44:45] op_sel_hi:[0,1]
	v_and_b32_e32 v42, 0xffff0000, v42
	v_and_b32_e32 v35, 0xffff0000, v43
	v_or_b32_sdwa v46, v42, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_barrier
	ds_read_b128 v[74:77], v47 offset:8192
	v_or_b32_sdwa v47, v35, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(15)
	v_pk_mul_f32 v[42:43], v[50:51], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[50:51], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v42, 0xffff0000, v42
	v_and_b32_e32 v35, 0xffff0000, v43
	v_or_b32_sdwa v44, v42, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v22, 4, v22
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v45, v35, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[42:43], v[50:51], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[50:51], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v22, 0xffff0000, v43
	v_and_b32_e32 v35, 0xffff0000, v42
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v51, v22, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v50, v35, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[42:43], v[52:53], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[52:53], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v42
	v_and_b32_e32 v22, 0xffff0000, v43
	v_or_b32_sdwa v20, v35, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v35, 4, v23
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v21, v22, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[42:43], v[52:53], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[52:53], v[22:23] op_sel_hi:[0,1]
	v_and_b32_e32 v42, 0xffff0000, v42
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v35, 0xffff0000, v43
	v_or_b32_sdwa v22, v42, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[52:53], v[48:49], v[52:53] op_sel_hi:[0,1]
	v_or_b32_sdwa v23, v35, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[42:43], v[48:49], v[42:43] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v53
	v_and_b32_e32 v49, 0xffff0000, v52
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v42, v49, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[52:53], v[48:49], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[48:49], v[58:59] op_sel_hi:[0,1]
	v_or_b32_sdwa v43, v35, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v12, 0xffff0000, v49
	v_and_b32_e32 v35, 0xffff0000, v48
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v49, v12, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v48, v35, v52 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[54:55], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[54:55], v[52:53] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v59
	v_and_b32_e32 v35, 0xffff0000, v58
	v_or_b32_sdwa v53, v12, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v52, v35, v52 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v35, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[74:75], v[36:37], v[4:7]
	v_mul_f32_e64 v12, v54, v12
	v_mul_f32_e64 v13, v54, v13
	v_pk_mul_f32 v[54:55], v[54:55], v[58:59] op_sel_hi:[0,1]
	v_and_b32_e32 v54, 0xffff0000, v54
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v35, 0xffff0000, v55
	v_or_b32_sdwa v12, v54, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v54, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v55, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[56:57], v[58:59] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v35, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[54:55], v[56:57], v[54:55] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v59
	v_and_b32_e32 v57, 0xffff0000, v58
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v54, v57, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[56:57], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[56:57], v[70:71] op_sel_hi:[0,1]
	v_or_b32_sdwa v55, v35, v55 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v14, 0xffff0000, v57
	v_and_b32_e32 v35, 0xffff0000, v56
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v57, v14, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v56, v35, v58 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[70:71], v[68:69], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[68:69], v[58:59] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v71
	v_and_b32_e32 v35, 0xffff0000, v70
	v_or_b32_sdwa v59, v14, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v58, v35, v58 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v35, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[76:77], v[38:39], v[4:7]
	v_mul_f32_e64 v14, v68, v14
	v_mul_f32_e64 v15, v68, v15
	v_pk_mul_f32 v[68:69], v[68:69], v[70:71] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v35, 0xffff0000, v69
	v_and_b32_e32 v61, 0xffff0000, v68
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71] op_sel_hi:[0,1]
	v_or_b32_sdwa v15, v35, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v14, v61, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v35, 0xffff0000, v71
	v_and_b32_e32 v61, 0xffff0000, v70
	v_lshrrev_b32_e32 v16, 4, v16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[66:67], v[68:69] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[66:67], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v69, v35, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v16, 0xffff0000, v67
	v_and_b32_e32 v35, 0xffff0000, v66
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v67, v16, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v66, v35, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[64:65], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[64:65], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v16, 0xffff0000, v73
	v_and_b32_e32 v35, 0xffff0000, v72
	v_or_b32_sdwa v71, v16, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v35, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v35, 4, v17
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v68, v61, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[16:17], v[64:65], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[64:65], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v65
	v_and_b32_e32 v61, 0xffff0000, v64
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v65, v35, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v64, v61, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[62:63], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[62:63], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v73
	v_and_b32_e32 v61, 0xffff0000, v72
	v_or_b32_sdwa v73, v35, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v61, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v18, 4, v18
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[74:75], v[68:69], v[8:11]
	v_mul_f32_e64 v16, v62, v16
	v_mul_f32_e64 v17, v62, v17
	v_pk_mul_f32 v[62:63], v[62:63], v[78:79] op_sel_hi:[0,1]
	v_and_b32_e32 v18, 0xffff0000, v63
	v_and_b32_e32 v35, 0xffff0000, v62
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v63, v18, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v35, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[78:79], v[60:61], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[60:61], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v18, 0xffff0000, v79
	v_and_b32_e32 v35, 0xffff0000, v78
	v_or_b32_sdwa v17, v18, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v35, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v35, 4, v19
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[76:77], v[66:67], v[8:11]
	v_mul_f32_e64 v18, v60, v18
	v_mul_f32_e64 v19, v60, v19
	v_pk_mul_f32 v[60:61], v[60:61], v[78:79] op_sel_hi:[0,1]
	ds_read_b128 v[78:81], v133 offset:8192
	ds_read_b128 v[66:69], v132 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[8:11], v[78:79], v[70:71], v[8:11]
	v_and_b32_e32 v35, 0xffff0000, v61
	v_and_b32_e32 v60, 0xffff0000, v60
	v_or_b32_sdwa v18, v60, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[78:79], v[40:41], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[80:81], v[64:65], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v19, v35, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[38:39], v[34:35], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[34:35], v[36:37] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v39
	v_and_b32_e32 v38, 0xffff0000, v38
	v_or_b32_sdwa v39, v35, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v38, v38, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[80:81], v[46:47], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v36, v34, v36
	v_mul_f32_e64 v37, v34, v37
	v_pk_mul_f32 v[34:35], v[34:35], v[40:41] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[8:11], v[66:67], v[72:73], v[8:11]
	v_and_b32_e32 v34, 0xffff0000, v34
	v_and_b32_e32 v0, 0xffff0000, v35
	v_or_b32_sdwa v40, v34, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v41, v0, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[66:67], v[44:45], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt vmcnt(2)
	v_pk_mul_f32 v[44:45], v[32:33], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[32:33], v[36:37] op_sel_hi:[0,1]
	s_lshl_b32 s0, s10, 2
	ds_read_b128 v[74:77], v33 offset:8192
	v_and_b32_e32 v0, 0xffff0000, v35
	v_and_b32_e32 v33, 0xffff0000, v34
	v_mfma_f32_16x16x16_bf16 v[34:37], v[68:69], v[62:63], v[8:11]
	s_mov_b32 s10, s18
	s_mov_b32 s11, s19
	ds_read_b128 v[60:63], v31 offset:8192
	v_lshl_or_b32 v8, v26, 4, s0
	buffer_load_dword v9, v8, s[8:11], 0 offen
	v_mfma_f32_16x16x16_bf16 v[4:7], v[68:69], v[50:51], v[4:7]
	v_lshrrev_b32_e32 v11, 4, v1
	v_or_b32_sdwa v45, v0, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[74:75], v[20:21], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[76:77], v[22:23], v[4:7]
	v_mul_f32_e64 v10, v32, v10
	v_mul_f32_e64 v11, v32, v11
	v_pk_mul_f32 v[0:1], v[32:33], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v11, 0xffff0000, v11
	v_mfma_f32_16x16x16_bf16 v[20:23], v[74:75], v[16:17], v[34:37]
	v_and_b32_e32 v10, 0xffff0000, v10
	v_or_b32_sdwa v1, v11, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v10, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	ds_read_b128 v[34:37], v29 offset:8192
	v_mfma_f32_16x16x16_bf16 v[16:19], v[76:77], v[18:19], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[60:61], v[42:43], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(2)
	v_pk_mul_f32 v[20:21], v[30:31], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[30:31], v[10:11] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[62:63], v[48:49], v[4:7]
	v_and_b32_e32 v21, 0xffff0000, v21
	v_and_b32_e32 v20, 0xffff0000, v20
	v_or_b32_sdwa v44, v33, v44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[16:19], v[60:61], v[38:39], v[16:19]
	v_or_b32_sdwa v21, v21, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v20, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v2, 4, v2
	v_mfma_f32_16x16x16_bf16 v[16:19], v[62:63], v[40:41], v[16:19]
	ds_read_b128 v[38:41], v27 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[34:35], v[52:53], v[4:7]
	v_mul_f32_e64 v32, v30, v10
	v_mul_f32_e64 v33, v30, v11
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[36:37], v[12:13], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[34:35], v[44:45], v[16:19]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[10:13], v[36:37], v[0:1], v[10:13]
	s_nop 1
	v_mul_f32_e64 v16, v30, v22
	v_mul_f32_e64 v17, v30, v23
	v_and_b32_e32 v2, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v1, v2, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v16, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	ds_read_b128 v[30:33], v25 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[38:39], v[54:55], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[18:19], v[28:29], v[18:19] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[40:41], v[56:57], v[4:7]
	v_mul_f32_e64 v16, v28, v16
	v_mul_f32_e64 v17, v28, v17
	v_and_b32_e32 v2, 0xffff0000, v19
	v_or_b32_sdwa v17, v2, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[10:13], v[38:39], v[20:21], v[10:13]
	v_lshrrev_b32_e32 v20, 4, v3
	v_and_b32_e32 v18, 0xffff0000, v18
	v_or_b32_sdwa v16, v18, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[10:13], v[40:41], v[0:1], v[10:13]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[30:31], v[58:59], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[32:33], v[14:15], v[0:3]
	v_mul_f32_e64 v14, v28, v18
	v_mul_f32_e64 v15, v28, v19
	v_pk_mul_f32 v[18:19], v[28:29], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v19, 0xffff0000, v19
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[16:17], v[10:13]
	s_nop 2
	v_and_b32_e32 v10, 0xffff0000, v18
	v_or_b32_sdwa v11, v19, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v10, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[32:33], v[10:11], v[4:7]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v10, 0xffffff, v9
	v_cmp_gt_u32_e32 vcc, s4, v10
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
	s_nop 2
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v0
	s_mov_b32 s2, 0x41800000
	v_pk_mul_f32 v[10:11], v[10:11], s[2:3] op_sel_hi:[1,0]
	v_mov_b32_e32 v4, 8
	v_mul_f32_e32 v0, 0xbfb8aa3b, v11
	v_exp_f32_e32 v0, v0
	v_lshl_add_u32 v12, v9, 11, v24
	v_lshlrev_b32_sdwa v4, v4, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_add_lshl_u32 v4, v12, v4, 1
	v_add_f32_e32 v0, 1.0, v0
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, v11, v0
	v_fma_mixlo_f16 v0, v10, v0, 0
	buffer_store_short v0, v4, s[12:15], 0 offen
.LBB0_5:
	s_or_b64 exec, exec, s[0:1]
	s_nop 1
	buffer_load_dword v4, v8, s[8:11], 0 offen offset:4
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v0, 0xffffff, v4
	v_cmp_gt_u32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_7
	v_mov_b32_e32 v0, v5
	s_mov_b32 s2, 0x41800000
	v_pk_mul_f32 v[0:1], v[0:1], s[2:3] op_sel_hi:[1,0]
	v_mov_b32_e32 v9, 8
	v_mul_f32_e32 v5, 0xbfb8aa3b, v1
	v_exp_f32_e32 v5, v5
	v_lshl_add_u32 v10, v4, 11, v24
	v_lshlrev_b32_sdwa v4, v9, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_add_f32_e32 v5, 1.0, v5
	v_rcp_f32_e32 v5, v5
	s_nop 0
	v_mul_f32_e32 v1, v1, v5
	v_fma_mixlo_f16 v0, v0, v1, 0
	v_add_lshl_u32 v1, v10, v4, 1
	buffer_store_short v0, v1, s[12:15], 0 offen
.LBB0_7:
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s10, s18
	s_mov_b32 s11, s19
	buffer_load_dword v0, v8, s[8:11], 0 offen offset:8
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v0
	v_cmp_gt_u32_e32 vcc, s4, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_9
	v_mov_b32_e32 v4, v6
	v_mov_b32_e32 v5, v2
	s_mov_b32 s2, 0x41800000
	v_pk_mul_f32 v[4:5], v[4:5], s[2:3] op_sel_hi:[1,0]
	v_mov_b32_e32 v2, 8
	v_mul_f32_e32 v1, 0xbfb8aa3b, v5
	v_exp_f32_e32 v1, v1
	v_lshl_add_u32 v6, v0, 11, v24
	v_lshlrev_b32_sdwa v0, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_add_lshl_u32 v0, v6, v0, 1
	v_add_f32_e32 v1, 1.0, v1
	v_rcp_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, v5, v1
	v_fma_mixlo_f16 v1, v4, v1, 0
	buffer_store_short v1, v0, s[12:15], 0 offen
.LBB0_9:
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v0, v8, s[8:11], 0 offen offset:12
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v0
	v_cmp_gt_u32_e32 vcc, s4, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_11
	v_mov_b32_e32 v2, v7
	s_mov_b32 s0, 0x41800000
	v_pk_mul_f32 v[2:3], v[2:3], s[0:1] op_sel_hi:[1,0]
	v_mov_b32_e32 v4, 8
	v_mul_f32_e32 v1, 0xbfb8aa3b, v3
	v_exp_f32_e32 v1, v1
	v_lshl_add_u32 v5, v0, 11, v24
	v_lshlrev_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_add_lshl_u32 v0, v5, v0, 1
	v_add_f32_e32 v1, 1.0, v1
	v_rcp_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, v3, v1
	v_fma_mixlo_f16 v1, v2, v1, 0
	buffer_store_short v1, v0, s[12:15], 0 offen
.LBB0_11:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_gemm1_0
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 88
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
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 166
		.amdhsa_next_free_sgpr 44
		.amdhsa_accum_offset 168
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
	.size	moe_gemm1_0, .Lfunc_end0-moe_gemm1_0

	.set moe_gemm1_0.num_vgpr, 166
	.set moe_gemm1_0.num_agpr, 0
	.set moe_gemm1_0.numbered_sgpr, 44
	.set moe_gemm1_0.num_named_barrier, 0
	.set moe_gemm1_0.private_seg_size, 0
	.set moe_gemm1_0.uses_vcc, 1
	.set moe_gemm1_0.uses_flat_scratch, 0
	.set moe_gemm1_0.has_dyn_sized_stack, 0
	.set moe_gemm1_0.has_recursion, 0
	.set moe_gemm1_0.has_indirect_call, 0
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
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 256
    .name:           moe_gemm1_0
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         moe_gemm1_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     166
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
