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
	v_lshrrev_b32_e32 v27, 5, v0
	s_mov_b32 s30, -1
	s_and_b32 s9, s23, 0xffff
	v_or_b32_e32 v1, s10, v27
	v_or_b32_e32 v29, 8, v27
	s_mov_b32 s16, s22
	s_mov_b32 s17, s9
	s_mov_b32 s18, s30
	s_mov_b32 s19, s27
	v_lshlrev_b32_e32 v1, 2, v1
	v_or_b32_e32 v2, s10, v29
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_load_dword v3, v1, s[16:19], 0 offen
	buffer_load_dword v4, v2, s[16:19], 0 offen
	s_ashr_i32 s3, s2, 31
	v_lshlrev_b32_e32 v1, 2, v0
	v_bfe_u32 v34, v0, 4, 2
	v_and_b32_e32 v30, 15, v0
	v_lshrrev_b32_e32 v0, 2, v0
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_lshl_b64 s[0:1], s[2:3], 6
	v_and_b32_e32 v0, 48, v0
	v_and_b32_e32 v49, 0x7c, v1
	v_or3_b32 v32, s0, v0, v30
	v_mov_b32_e32 v33, s1
	s_ashr_i32 s1, s6, 31
	s_mov_b32 s0, s6
	s_ashr_i32 s33, s6, 3
	s_mul_i32 s6, s4, s6
	s_mov_b32 s23, 0x2aaaaaab
	s_lshl_b32 s42, s6, 1
	s_lshr_b64 s[6:7], s[0:1], 1
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
	v_lshl_add_u64 v[20:21], v[0:1], 0, v[32:33]
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
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v4
	v_cmp_gt_u32_e32 vcc, s4, v2
	v_ashrrev_i32_e32 v31, 4, v1
	v_sub_u32_e32 v16, v20, v0
	v_cndmask_b32_e32 v2, 0, v2, vcc
	v_cmp_gt_u32_e32 vcc, s4, v3
	v_mad_u64_u32 v[42:43], s[14:15], s6, v2, 0
	s_nop 0
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_mul_hi_i32 v2, v21, s23
	v_mad_u64_u32 v[40:41], s[6:7], s6, v3, 0
	v_lshrrev_b32_e32 v0, 31, v2
	v_ashrrev_i32_e32 v2, 11, v2
	v_mul_hi_i32 v3, v31, s23
	v_add_u32_e32 v0, v2, v0
	v_lshrrev_b32_e32 v2, 31, v3
	v_ashrrev_i32_e32 v3, 11, v3
	v_mul_i32_i24_e32 v0, 0x3000, v0
	v_add_u32_e32 v2, v3, v2
	v_sub_u32_e32 v22, v21, v0
	v_mul_i32_i24_e32 v0, 0x3000, v2
	v_sub_u32_e32 v24, v31, v0
	v_and_b32_e32 v0, -16, v1
	v_ashrrev_i32_e32 v23, 31, v22
	v_sub_u32_e32 v0, v19, v0
	v_mov_b32_e32 v1, v17
	v_ashrrev_i32_e32 v25, 31, v24
	v_mul_lo_u32 v2, s2, v34
	s_mov_b32 s14, 0x1be00
	v_mul_lo_u32 v28, v0, s33
	v_lshlrev_b32_e32 v6, 4, v2
	v_mad_i64_i32 v[2:3], s[6:7], v18, s14, v[16:17]
	v_mad_i64_i32 v[0:1], s[6:7], v18, s14, v[0:1]
	v_lshl_add_u64 v[38:39], v[22:23], 4, v[2:3]
	v_lshl_add_u64 v[36:37], v[24:25], 4, v[0:1]
	v_mul_lo_u32 v4, v22, s3
	v_mul_lo_u32 v26, v16, s33
	v_mul_lo_u32 v5, v24, s3
	v_lshlrev_b32_e32 v0, 2, v38
	v_lshlrev_b32_e32 v1, 2, v36
	v_add3_u32 v51, v26, v6, v4
	v_add3_u32 v43, v28, v6, v5
	v_add_u32_e32 v2, 0x1000, v0
	v_add_u32_e32 v3, 0x2000, v0
	v_add_u32_e32 v4, 0x3000, v0
	v_add_u32_e32 v5, 0x1000, v1
	v_add_u32_e32 v6, 0x2000, v1
	v_and_b32_e32 v16, -4, v51
	v_and_b32_e32 v23, -4, v43
	v_add_u32_e32 v25, 0x3000, v1
	buffer_load_dword v80, v0, s[36:39], 0 offen
	buffer_load_dword v78, v0, s[36:39], 0 offen offset:2048
	buffer_load_dword v74, v2, s[36:39], 0 offen
	buffer_load_dword v70, v2, s[36:39], 0 offen offset:2048
	buffer_load_dword v66, v3, s[36:39], 0 offen
	buffer_load_dword v62, v3, s[36:39], 0 offen offset:2048
	buffer_load_dword v60, v4, s[36:39], 0 offen
	buffer_load_dword v54, v4, s[36:39], 0 offen offset:2048
	buffer_load_dword v82, v1, s[36:39], 0 offen
	buffer_load_dword v76, v1, s[36:39], 0 offen offset:2048
	buffer_load_dword v72, v5, s[36:39], 0 offen
	buffer_load_dword v68, v5, s[36:39], 0 offen offset:2048
	buffer_load_dword v64, v6, s[36:39], 0 offen
	buffer_load_dword v58, v6, s[36:39], 0 offen offset:2048
	buffer_load_dword v56, v25, s[36:39], 0 offen
	buffer_load_dword v52, v25, s[36:39], 0 offen offset:2048
	buffer_load_dwordx4 v[12:15], v16, s[28:31], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[4:7], v16, s[28:31], 0 offen offset:16
	buffer_load_dwordx4 v[8:11], v23, s[28:31], 0 offen
	buffer_load_dwordx4 v[0:3], v23, s[28:31], 0 offen offset:16
	v_add_lshl_u32 v53, v42, v49, 2
	v_add_lshl_u32 v55, v40, v49, 2
	buffer_load_dwordx4 v[84:87], v53, s[40:43], 0 offen
	buffer_load_dwordx4 v[88:91], v55, s[40:43], 0 offen
	v_lshlrev_b32_e32 v16, 9, v27
	v_lshlrev_b32_e32 v23, 4, v27
	v_lshlrev_b32_e32 v25, 2, v49
	v_lshlrev_b32_e32 v33, 4, v30
	v_lshlrev_b32_e32 v30, 9, v30
	v_lshlrev_b32_e32 v35, 4, v34
	v_lshlrev_b32_e32 v27, 9, v29
	v_lshlrev_b32_e32 v29, 4, v29
	s_mov_b32 s3, 0x6f800
	v_bitop3_b32 v41, v16, v25, v23 bitop3:0xf6
	v_bitop3_b32 v37, v30, v35, v33 bitop3:0xf6
	v_lshlrev_b32_e32 v24, 6, v24
	v_lshlrev_b32_e32 v22, 6, v22
	v_bitop3_b32 v39, v27, v29, v25 bitop3:0xf6
	v_or_b32_e32 v16, 64, v35
	v_mul_lo_u32 v18, v18, s3
	v_or_b32_e32 v23, 0x80, v35
	v_or_b32_e32 v25, 0xc0, v35
	v_or_b32_e32 v27, 0x100, v35
	v_or_b32_e32 v29, 0x140, v35
	v_or_b32_e32 v44, 0x180, v35
	v_or_b32_e32 v45, 0x1c0, v35
	v_or_b32_e32 v46, v24, v35
	v_or_b32_e32 v47, v22, v35
	v_bitop3_b32 v126, v30, v16, v33 bitop3:0xf6
	v_add_u32_e32 v16, v18, v24
	v_bitop3_b32 v125, v30, v23, v33 bitop3:0xf6
	v_bitop3_b32 v124, v30, v25, v33 bitop3:0xf6
	v_bitop3_b32 v77, v30, v27, v33 bitop3:0xf6
	v_bitop3_b32 v59, v30, v29, v33 bitop3:0xf6
	v_bitop3_b32 v35, v30, v44, v33 bitop3:0xf6
	v_bitop3_b32 v33, v30, v45, v33 bitop3:0xf6
	v_add_u32_e32 v18, v18, v22
	v_mad_u64_u32 v[44:45], s[6:7], v46, s2, v[28:29]
	v_mad_u64_u32 v[46:47], s[2:3], v47, s2, v[26:27]
	v_lshl_add_u32 v16, v19, 2, v16
	v_lshlrev_b32_e32 v19, 6, v31
	v_lshl_add_u32 v18, v20, 2, v18
	v_sub_u32_e32 v48, v16, v19
	v_lshlrev_b32_e32 v16, 6, v21
	s_mov_b32 s2, 0xfff98000
	v_sub_u32_e32 v50, v18, v16
	v_add_u32_e32 v45, 0x400, v55
	v_add_u32_e32 v47, 0x400, v53
	s_mov_b32 s3, -1
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v21, v17
	s_waitcnt vmcnt(1)
	ds_write_b128 v41, v[84:87]
	s_waitcnt vmcnt(0)
	ds_write_b128 v39, v[88:91]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[84:87], v37
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	.p2align	5, , 4
.LBB0_2:
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v90, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v91, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v53, s11, v46
	v_pk_mul_f32 v[90:91], v[80:81], v[90:91] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v88, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v89, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v61, 0xffff0000, v91
	v_and_b32_e32 v63, 0xffff0000, v90
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v90, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v91, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_add_u32_e32 v24, 32, v53
	v_pk_mul_f32 v[88:89], v[80:81], v[88:89] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v92, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v93, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[90:91], v[80:81], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[80:81], v[92:93] op_sel_hi:[0,1]
	v_and_b32_e32 v55, -4, v24
	v_or_b32_sdwa v89, v61, v89 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v88, v63, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v12, 0xffff0000, v81
	v_and_b32_e32 v61, 0xffff0000, v80
	ds_read_b128 v[94:97], v126
	ds_read_b128 v[98:101], v125
	buffer_load_dwordx4 v[28:31], v55, s[16:19], 0 offen
	buffer_load_dwordx4 v[24:27], v55, s[16:19], 0 offen offset:16
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[84:85], v[88:89], v[16:19]
	v_or_b32_sdwa v81, v12, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v80, v61, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v90, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v91, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v88, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v89, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_add_u32_e32 v55, s2, v50
	v_pk_mul_f32 v[90:91], v[82:83], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[82:83], v[88:89] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v91
	v_and_b32_e32 v61, 0xffff0000, v90
	v_add_u32_e32 v57, 0x6c000, v55
	s_mov_b32 s22, s18
	s_mov_b32 s23, s19
	v_or_b32_sdwa v89, v12, v89 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v88, v61, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[16:19], v[86:87], v[80:81], v[16:19]
	buffer_load_dword v80, v57, s[20:23], 0 offen
	v_lshrrev_b32_e32 v8, 4, v8
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v90, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v91, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[20:23], v[84:85], v[88:89], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v92, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v93, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v90, v82, v90
	v_mul_f32_e64 v91, v82, v91
	v_pk_mul_f32 v[82:83], v[82:83], v[92:93] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v83
	v_and_b32_e32 v12, 0xffff0000, v82
	v_or_b32_sdwa v83, v8, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v82, v12, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[84:85], v[78:79], v[84:85] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[86:87], v[82:83], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	ds_read_b128 v[88:91], v124
	v_pk_mul_f32 v[86:87], v[78:79], v[86:87] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v87
	v_and_b32_e32 v12, 0xffff0000, v86
	v_or_b32_sdwa v85, v8, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v84, v12, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v82, v57, s[20:23], 0 offen offset:2048
	v_lshrrev_b32_e32 v8, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[94:95], v[84:85], v[16:19]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v12, v78, v12
	v_mul_f32_e64 v13, v78, v13
	v_pk_mul_f32 v[78:79], v[78:79], v[86:87] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v79
	v_and_b32_e32 v61, 0xffff0000, v78
	v_or_b32_sdwa v13, v8, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v61, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v57, 0x6d000, v55
	v_pk_mul_f32 v[84:85], v[76:77], v[84:85] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[96:97], v[12:13], v[16:19]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v8, 0xffff0000, v85
	v_pk_mul_f32 v[12:13], v[76:77], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v61, 0xffff0000, v84
	v_or_b32_sdwa v13, v8, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v61, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v78, v57, s[20:23], 0 offen
	v_lshrrev_b32_e32 v61, 4, v9
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[20:23], v[94:95], v[12:13], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v84, v76, v84
	v_mul_f32_e64 v85, v76, v85
	v_pk_mul_f32 v[8:9], v[76:77], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v61, 0xffff0000, v85
	v_and_b32_e32 v12, 0xffff0000, v84
	v_or_b32_sdwa v9, v61, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[12:13], v[74:75], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[96:97], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v13, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[74:75], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	ds_read_b128 v[92:95], v77
	buffer_load_dword v76, v57, s[20:23], 0 offen offset:2048
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[98:99], v[8:9], v[16:19]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v12, v74, v12
	v_mul_f32_e64 v13, v74, v13
	v_pk_mul_f32 v[74:75], v[74:75], v[84:85] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v75
	v_and_b32_e32 v8, 0xffff0000, v74
	v_or_b32_sdwa v9, v14, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v57, 0x6e000, v55
	v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[100:101], v[8:9], v[16:19]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v13, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[72:73], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v18, v57, s[20:23], 0 offen
	v_lshrrev_b32_e32 v10, 4, v10
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[20:23], v[98:99], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v16, v72, v16
	v_mul_f32_e64 v17, v72, v17
	v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v17
	v_and_b32_e32 v8, 0xffff0000, v16
	v_or_b32_sdwa v9, v10, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v19, 0x6f000, v55
	v_pk_mul_f32 v[12:13], v[70:71], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[100:101], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[70:71], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v10, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	ds_read_b128 v[96:99], v59
	buffer_load_dword v22, v57, s[20:23], 0 offen offset:2048
	v_pk_mul_f32 v[16:17], v[70:71], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[70:71], v[14:15] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[88:89], v[8:9], v[84:87]
	v_and_b32_e32 v10, 0xffff0000, v21
	v_and_b32_e32 v8, 0xffff0000, v20
	v_or_b32_sdwa v9, v10, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v21, 4, v7
	v_pk_mul_f32 v[16:17], v[68:69], v[16:17] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[90:91], v[8:9], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v17
	v_pk_mul_f32 v[8:9], v[68:69], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v16, 0xffff0000, v16
	v_lshrrev_b32_e32 v17, 4, v11
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v16, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dword v20, v19, s[20:23], 0 offen
	v_pk_mul_f32 v[70:71], v[68:69], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[68:69], v[16:17] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[88:89], v[8:9], v[72:75]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v17, v17, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v16, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 4, v4
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[90:91], v[16:17], v[8:11]
	ds_read_b128 v[88:91], v35
	buffer_load_dword v16, v19, s[20:23], 0 offen offset:2048
	v_and_b32_e32 v17, 0xffff0000, v71
	v_and_b32_e32 v19, 0xffff0000, v70
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[66:67], v[68:69] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[66:67], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v69, v17, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v4, 0xffff0000, v67
	v_and_b32_e32 v17, 0xffff0000, v66
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v67, v4, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v66, v17, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[62:63], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[62:63], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v73
	v_and_b32_e32 v17, 0xffff0000, v72
	v_or_b32_sdwa v71, v4, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v17, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v17, 4, v5
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v68, v19, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[4:5], v[62:63], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[62:63], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v63
	v_and_b32_e32 v19, 0xffff0000, v62
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v63, v17, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v19, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[60:61], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[60:61], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v73
	v_and_b32_e32 v19, 0xffff0000, v72
	v_or_b32_sdwa v73, v17, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v19, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v6, 4, v6
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v19, s11, v44
	v_pk_mul_f32 v[4:5], v[60:61], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[60:61], v[74:75] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v61
	v_and_b32_e32 v17, 0xffff0000, v60
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v61, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v60, v17, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[74:75], v[54:55], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[54:55], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v75
	v_and_b32_e32 v17, 0xffff0000, v74
	v_or_b32_sdwa v75, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v17, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[92:93], v[68:69], v[12:15]
	s_waitcnt vmcnt(10)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[94:95], v[66:67], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v66, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v67, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v68, v64, v68
	v_mul_f32_e64 v69, v64, v69
	v_add_u32_e32 v12, 32, v19
	v_pk_mul_f32 v[66:67], v[64:65], v[66:67] op_sel_hi:[0,1]
	v_and_b32_e32 v23, 0xffff0000, v69
	v_and_b32_e32 v57, 0xffff0000, v68
	v_and_b32_e32 v17, -4, v12
	v_or_b32_sdwa v67, v23, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v66, v57, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[12:15], v17, s[16:19], 0 offen
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[92:93], v[66:67], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v68, v64, v68
	v_mul_f32_e64 v69, v64, v69
	v_pk_mul_f32 v[64:65], v[64:65], v[84:85] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v65
	v_and_b32_e32 v23, 0xffff0000, v64
	v_or_b32_sdwa v65, v0, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v64, v23, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[64:67], v[94:95], v[64:65], v[8:11]
	ds_read_b128 v[92:95], v33
	v_pk_mul_f32 v[68:69], v[58:59], v[68:69] op_sel_hi:[0,1]
	s_nop 0
	buffer_load_dwordx4 v[8:11], v17, s[16:19], 0 offen offset:16
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[96:97], v[70:71], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v17, s2, v48
	v_pk_mul_f32 v[70:71], v[58:59], v[70:71] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[98:99], v[62:63], v[4:7]
	v_and_b32_e32 v57, 0xffff0000, v71
	v_and_b32_e32 v63, 0xffff0000, v70
	v_add_u32_e32 v23, 0x6c000, v17
	v_or_b32_sdwa v69, v57, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v63, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v62, v23, s[20:23], 0 offen
	v_lshrrev_b32_e32 v1, 4, v1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[64:67], v[96:97], v[68:69], v[64:67]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v86, v58, v86
	v_mul_f32_e64 v87, v58, v87
	v_pk_mul_f32 v[70:71], v[58:59], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v1, 0xffff0000, v87
	v_and_b32_e32 v57, 0xffff0000, v86
	v_or_b32_sdwa v69, v1, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v57, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_add_u32_e32 v21, 0x6d000, v17
	v_pk_mul_f32 v[0:1], v[54:55], v[0:1] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[98:99], v[68:69], v[64:67]
	s_nop 2
	buffer_load_dword v66, v23, s[20:23], 0 offen offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[88:89], v[72:73], v[4:7]
	v_mul_f32_e64 v64, v54, v84
	v_mul_f32_e64 v65, v54, v85
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[90:91], v[60:61], v[4:7]
	v_mul_f32_e64 v84, v56, v84
	v_mul_f32_e64 v85, v56, v85
	v_pk_mul_f32 v[72:73], v[56:57], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v23, 0xffff0000, v85
	v_and_b32_e32 v54, 0xffff0000, v84
	v_or_b32_sdwa v73, v23, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v54, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v60, v21, s[20:23], 0 offen
	v_lshrrev_b32_e32 v2, 4, v2
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[68:71], v[88:89], v[72:73], v[68:71]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v84, v56, v84
	v_mul_f32_e64 v85, v56, v85
	v_pk_mul_f32 v[56:57], v[56:57], v[86:87] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v57
	v_and_b32_e32 v23, 0xffff0000, v56
	v_or_b32_sdwa v57, v2, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v56, v23, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v2, 0xffff0000, v65
	v_and_b32_e32 v23, 0xffff0000, v64
	v_mfma_f32_16x16x16_bf16 v[68:71], v[90:91], v[56:57], v[68:71]
	buffer_load_dword v54, v21, s[20:23], 0 offen offset:2048
	v_or_b32_sdwa v1, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v23, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[92:93], v[74:75], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v21, 0x6e000, v17
	v_mfma_f32_16x16x16_bf16 v[4:7], v[94:95], v[0:1], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v64, v52, v64
	v_mul_f32_e64 v65, v52, v65
	v_pk_mul_f32 v[0:1], v[52:53], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v65
	v_and_b32_e32 v23, 0xffff0000, v64
	v_or_b32_sdwa v1, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v23, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v23, 4, v3
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	buffer_load_dword v56, v21, s[20:23], 0 offen
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[52:53], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[92:93], v[0:1], v[68:71]
	v_mul_f32_e64 v64, v52, v64
	v_mul_f32_e64 v65, v52, v65
	v_and_b32_e32 v23, 0xffff0000, v65
	v_and_b32_e32 v52, 0xffff0000, v64
	v_or_b32_sdwa v65, v23, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v64, v52, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[94:95], v[64:65], v[0:3]
	buffer_load_dword v64, v21, s[20:23], 0 offen offset:2048
	v_add_u32_e32 v21, 0x6f000, v17
	buffer_load_dword v58, v21, s[20:23], 0 offen
	buffer_load_dword v52, v21, s[20:23], 0 offen offset:2048
	s_waitcnt vmcnt(19)
	v_lshrrev_b32_e32 v81, 4, v29
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v88, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v89, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v94, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v95, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v102, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v103, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v129, 4, v28
	v_lshrrev_b32_e32 v79, 4, v30
	v_lshrrev_b32_e32 v83, 4, v31
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v30, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v31, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v23, 4, v26
	v_lshrrev_b32_e32 v21, 4, v27
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v26, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v27, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
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
	v_cvt_off_f32_i4_sdwa v132, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v133, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v140, 4, v12
	v_lshrrev_b32_e32 v131, 4, v13
	v_lshrrev_b32_e32 v130, 4, v14
	v_lshrrev_b32_e32 v63, 4, v15
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v136, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v137, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v138, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v139, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
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
	v_lshrrev_b32_e32 v128, 4, v8
	v_lshrrev_b32_e32 v127, 4, v9
	v_pk_mul_f32 v[8:9], v[80:81], v[74:75] op_sel_hi:[0,1]
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
	v_lshrrev_b32_e32 v67, 4, v10
	v_lshrrev_b32_e32 v65, 4, v11
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[80:81], v[84:85] op_sel_hi:[0,1]
	v_or_b32_sdwa v111, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v110, v11, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[82:83], v[88:89] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[82:83], v[86:87] op_sel_hi:[0,1]
	v_or_b32_sdwa v75, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v11, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[78:79], v[96:97] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v11, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[78:79], v[94:95] op_sel_hi:[0,1]
	v_or_b32_sdwa v85, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v84, v11, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[76:77], v[104:105] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v108, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v109, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v61, 4, v24
	v_and_b32_e32 v10, 0xffff0000, v9
	v_and_b32_e32 v24, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[76:77], v[102:103] op_sel_hi:[0,1]
	s_waitcnt vmcnt(7)
	v_pk_mul_f32 v[72:73], v[62:63], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v11, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v24, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[8:9], v[18:19], v[108:109] op_sel_hi:[0,1]
	v_and_b32_e32 v73, 0xffff0000, v73
	v_and_b32_e32 v72, 0xffff0000, v72
	v_pk_mul_f32 v[26:27], v[62:63], v[26:27] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v57, 4, v25
	v_and_b32_e32 v24, 0xffff0000, v9
	v_and_b32_e32 v25, 0xffff0000, v8
	v_pk_mul_f32 v[8:9], v[18:19], v[106:107] op_sel_hi:[0,1]
	v_or_b32_sdwa v95, v73, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v94, v72, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(6)
	v_pk_mul_f32 v[26:27], v[66:67], v[118:119] op_sel_hi:[0,1]
	v_or_b32_sdwa v9, v24, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v25, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[24:25], v[22:23], v[114:115] op_sel_hi:[0,1]
	v_and_b32_e32 v72, 0xffff0000, v27
	v_and_b32_e32 v73, 0xffff0000, v26
	v_pk_mul_f32 v[26:27], v[66:67], v[116:117] op_sel_hi:[0,1]
	v_and_b32_e32 v86, 0xffff0000, v25
	v_and_b32_e32 v87, 0xffff0000, v24
	v_pk_mul_f32 v[24:25], v[22:23], v[112:113] op_sel_hi:[0,1]
	v_or_b32_sdwa v89, v72, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v88, v73, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(5)
	v_pk_mul_f32 v[26:27], v[60:61], v[122:123] op_sel_hi:[0,1]
	v_or_b32_sdwa v25, v86, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v72, 0xffff0000, v27
	v_and_b32_e32 v86, 0xffff0000, v26
	v_pk_mul_f32 v[26:27], v[60:61], v[120:121] op_sel_hi:[0,1]
	v_or_b32_sdwa v73, v72, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v86, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(4)
	v_pk_mul_f32 v[26:27], v[54:55], v[134:135] op_sel_hi:[0,1]
	v_and_b32_e32 v86, 0xffff0000, v27
	v_and_b32_e32 v96, 0xffff0000, v26
	v_pk_mul_f32 v[26:27], v[54:55], v[132:133] op_sel_hi:[0,1]
	v_or_b32_sdwa v24, v87, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v87, v86, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v86, v96, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[26:27], v[56:57], v[138:139] op_sel_hi:[0,1]
	s_barrier
	ds_read_b128 v[148:151], v37 offset:8192
	v_pk_mul_f32 v[96:97], v[56:57], v[136:137] op_sel_hi:[0,1]
	v_and_b32_e32 v27, 0xffff0000, v27
	v_and_b32_e32 v26, 0xffff0000, v26
	v_or_b32_sdwa v27, v27, v97 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v26, v26, v96 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v129 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v129 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v102, v129 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v103, v129 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v108, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[96:97], v[80:81], v[96:97] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[80:81], v[102:103] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[104:105], v[82:83], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[82:83], v[80:81] op_sel_hi:[0,1]
	v_and_b32_e32 v80, 0xffff0000, v103
	v_and_b32_e32 v82, 0xffff0000, v102
	v_or_b32_sdwa v81, v80, v97 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v80, v82, v96 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v82, 0xffff0000, v107
	v_and_b32_e32 v96, 0xffff0000, v106
	v_or_b32_sdwa v103, v82, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v102, v96, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v109, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	ds_read_b128 v[156:159], v126 offset:8192
	v_pk_mul_f32 v[104:105], v[78:79], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[78:79], v[108:109] op_sel_hi:[0,1]
	v_and_b32_e32 v82, 0xffff0000, v105
	v_and_b32_e32 v104, 0xffff0000, v104
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v83, v82, v107 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v82, v104, v106 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
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
	ds_read_b128 v[140:143], v77 offset:8192
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[148:149], v[110:111], v[4:7]
	v_mul_f32_e64 v106, v62, v106
	v_mul_f32_e64 v107, v62, v107
	v_pk_mul_f32 v[104:105], v[62:63], v[104:105] op_sel_hi:[0,1]
	v_and_b32_e32 v62, 0xffff0000, v107
	v_mfma_f32_16x16x16_bf16 v[0:3], v[148:149], v[94:95], v[0:3]
	v_and_b32_e32 v106, 0xffff0000, v106
	v_or_b32_sdwa v119, v62, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v118, v106, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[150:151], v[80:81], v[4:7]
	ds_read_b128 v[160:163], v125 offset:8192
	ds_read_b128 v[164:167], v124 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v131 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[150:151], v[118:119], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v131 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v131 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v131 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[156:157], v[74:75], v[4:7]
	v_mul_f32_e64 v106, v66, v106
	v_mul_f32_e64 v107, v66, v107
	v_pk_mul_f32 v[104:105], v[66:67], v[104:105] op_sel_hi:[0,1]
	v_and_b32_e32 v62, 0xffff0000, v107
	v_mfma_f32_16x16x16_bf16 v[0:3], v[156:157], v[88:89], v[0:3]
	v_and_b32_e32 v66, 0xffff0000, v106
	v_or_b32_sdwa v121, v62, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v120, v66, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[158:159], v[102:103], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v106, v130 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v107, v130 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v130 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[160:161], v[84:85], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v130 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v106, v60, v106
	v_mul_f32_e64 v107, v60, v107
	v_pk_mul_f32 v[104:105], v[60:61], v[104:105] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[158:159], v[120:121], v[0:3]
	v_and_b32_e32 v60, 0xffff0000, v107
	v_and_b32_e32 v62, 0xffff0000, v106
	v_or_b32_sdwa v123, v60, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[160:161], v[72:73], v[0:3]
	v_or_b32_sdwa v122, v62, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[130:131], v[76:77], v[96:97] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[76:77], v[78:79] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[162:163], v[82:83], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v96, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[162:163], v[122:123], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v97, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v57, 0x70000, v17
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v148, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v149, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt vmcnt(2)
	v_pk_mul_f32 v[114:115], v[64:65], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[64:65], v[12:13] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v128 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v128 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[116:117], v[58:59], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[20:21], v[148:149] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v128 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v128 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[164:165], v[10:11], v[4:7]
	v_mul_f32_e64 v148, v56, v12
	v_mul_f32_e64 v149, v56, v13
	v_pk_mul_f32 v[150:151], v[56:57], v[14:15] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v127 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v127 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_add_u32_e32 v53, 64, v53
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v127 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v127 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v129, 0x70000, v55
	v_pk_mul_f32 v[102:103], v[64:65], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121], v[64:65], v[14:15] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_add_u32_e32 v66, 0x71000, v55
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v104, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v105, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v152, 0x72000, v55
	v_add_u32_e32 v55, 0x73000, v55
	v_add_u32_e32 v154, 0x71000, v17
	v_pk_mul_f32 v[94:95], v[22:23], v[78:79] op_sel_hi:[0,1]
	v_add_u32_e32 v79, 0x72000, v17
	v_pk_mul_f32 v[106:107], v[20:21], v[28:29] op_sel_hi:[0,1]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[28:29], v[52:53], v[98:99] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[164:165], v[86:87], v[0:3]
	v_mul_f32_e64 v84, v58, v12
	v_mul_f32_e64 v85, v58, v13
	v_pk_mul_f32 v[98:99], v[58:59], v[14:15] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v145
	v_and_b32_e32 v14, 0xffff0000, v144
	v_add_u32_e32 v153, 64, v19
	v_pk_mul_f32 v[146:147], v[18:19], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[18:19], v[60:61] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v62, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v63, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[90:91], v[58:59], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[52:53], v[100:101] op_sel_hi:[0,1]
	v_and_b32_e32 v53, -4, v53
	buffer_load_dword v80, v129, s[20:23], 0 offen
	buffer_load_dword v78, v129, s[20:23], 0 offen offset:2048
	buffer_load_dword v72, v154, s[20:23], 0 offen
	buffer_load_dword v64, v79, s[20:23], 0 offen
	buffer_load_dword v58, v79, s[20:23], 0 offen offset:2048
	v_pk_mul_f32 v[128:129], v[54:55], v[62:63] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v12, v131 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v14, v130 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_pk_mul_f32 v[22:23], v[54:55], v[22:23] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	buffer_load_dwordx4 v[132:135], v47, s[24:27], 0 offen
	buffer_load_dwordx4 v[136:139], v45, s[24:27], 0 offen
	v_pk_mul_f32 v[86:87], v[52:53], v[10:11] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[166:167], v[12:13], v[4:7]
	v_and_b32_e32 v10, 0xffff0000, v129
	v_and_b32_e32 v12, 0xffff0000, v128
	v_or_b32_sdwa v11, v10, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v12, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v17, 0x73000, v17
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v88, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v89, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[110:111], v[20:21], v[30:31] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[166:167], v[10:11], v[0:3]
	v_mul_f32_e64 v112, v16, v70
	v_mul_f32_e64 v113, v16, v71
	v_pk_mul_f32 v[104:105], v[16:17], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[20:21], v[60:61] op_sel_hi:[0,1]
	buffer_load_dword v74, v66, s[20:23], 0 offen
	buffer_load_dword v70, v66, s[20:23], 0 offen offset:2048
	buffer_load_dword v60, v55, s[20:23], 0 offen
	buffer_load_dword v54, v55, s[20:23], 0 offen offset:2048
	buffer_load_dword v82, v57, s[20:23], 0 offen
	buffer_load_dword v76, v57, s[20:23], 0 offen offset:2048
	buffer_load_dword v56, v17, s[20:23], 0 offen
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[88:89], v[16:17], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[118:119], v[16:17], v[20:21] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v19
	v_pk_mul_f32 v[122:123], v[52:53], v[14:15] op_sel_hi:[0,1]
	buffer_load_dword v52, v17, s[20:23], 0 offen offset:2048
	v_and_b32_e32 v55, -4, v153
	v_and_b32_e32 v11, 0xffff0000, v18
	v_mfma_f32_16x16x16_bf16 v[16:19], v[140:141], v[8:9], v[4:7]
	v_and_b32_e32 v8, 0xffff0000, v151
	v_and_b32_e32 v9, 0xffff0000, v150
	buffer_load_dword v66, v152, s[20:23], 0 offen
	buffer_load_dword v62, v152, s[20:23], 0 offen offset:2048
	buffer_load_dword v68, v154, s[20:23], 0 offen offset:2048
	buffer_load_dwordx4 v[12:15], v53, s[16:19], 0 offen
	buffer_load_dwordx4 v[4:7], v53, s[16:19], 0 offen offset:16
	v_or_b32_sdwa v129, v10, v147 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v128, v11, v146 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[140:141], v[26:27], v[0:3]
	v_or_b32_sdwa v27, v8, v149 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v26, v9, v148 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[8:11], v55, s[16:19], 0 offen
	buffer_load_dwordx4 v[0:3], v55, s[16:19], 0 offen offset:16
	v_mfma_f32_16x16x16_bf16 v[16:19], v[142:143], v[128:129], v[16:19]
	ds_read_b128 v[128:131], v59 offset:8192
	ds_read_b128 v[144:147], v35 offset:8192
	ds_read_b128 v[148:151], v33 offset:8192
	v_mfma_f32_16x16x16_bf16 v[20:23], v[142:143], v[26:27], v[20:23]
	v_and_b32_e32 v53, 0xffff0000, v111
	v_and_b32_e32 v55, 0xffff0000, v110
	v_and_b32_e32 v63, 0xffff0000, v115
	v_and_b32_e32 v65, 0xffff0000, v114
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[128:129], v[24:25], v[16:19]
	v_and_b32_e32 v71, 0xffff0000, v101
	v_and_b32_e32 v73, 0xffff0000, v100
	v_or_b32_sdwa v101, v53, v107 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v100, v55, v106 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v107, v63, v109 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v106, v65, v108 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v24, 0xffff0000, v97
	v_and_b32_e32 v26, 0xffff0000, v96
	v_mfma_f32_16x16x16_bf16 v[20:23], v[128:129], v[106:107], v[20:23]
	v_or_b32_sdwa v25, v24, v95 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v26, v94 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v53, 0xffff0000, v121
	v_and_b32_e32 v67, 0xffff0000, v117
	v_mfma_f32_16x16x16_bf16 v[16:19], v[130:131], v[24:25], v[16:19]
	v_and_b32_e32 v24, 0xffff0000, v120
	v_or_b32_sdwa v25, v53, v103 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v24, v102 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v69, 0xffff0000, v116
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[144:145], v[100:101], v[16:19]
	v_and_b32_e32 v53, 0xffff0000, v93
	v_and_b32_e32 v55, 0xffff0000, v92
	v_and_b32_e32 v57, 0xffff0000, v113
	v_mfma_f32_16x16x16_bf16 v[20:23], v[130:131], v[24:25], v[20:23]
	v_or_b32_sdwa v25, v67, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v69, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v27, v57, v105 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v57, 0xffff0000, v99
	v_mfma_f32_16x16x16_bf16 v[20:23], v[144:145], v[24:25], v[20:23]
	v_or_b32_sdwa v25, v53, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v55, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v61, 0xffff0000, v112
	v_or_b32_sdwa v26, v61, v104 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[16:19], v[146:147], v[24:25], v[16:19]
	v_and_b32_e32 v24, 0xffff0000, v98
	v_or_b32_sdwa v25, v57, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v24, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v29, v71, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v28, v73, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[146:147], v[24:25], v[20:23]
	v_and_b32_e32 v24, 0xffff0000, v119
	v_and_b32_e32 v30, 0xffff0000, v118
	v_and_b32_e32 v31, 0xffff0000, v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[148:149], v[26:27], v[16:19]
	v_and_b32_e32 v26, 0xffff0000, v123
	v_or_b32_sdwa v25, v24, v89 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v30, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[148:149], v[28:29], v[20:23]
	v_or_b32_sdwa v27, v26, v87 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v26, v31, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(16)
	ds_write_b128 v41, v[132:135]
	s_waitcnt vmcnt(15)
	ds_write_b128 v39, v[136:139]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[84:87], v37
	v_mfma_f32_16x16x16_bf16 v[16:19], v[150:151], v[24:25], v[16:19]
	s_add_u32 s2, s2, 0x8000
	s_addc_u32 s3, s3, 0
	s_add_i32 s11, s11, 64
	v_mfma_f32_16x16x16_bf16 v[20:23], v[150:151], v[26:27], v[20:23]
	v_add_u32_e32 v45, 0x400, v45
	s_cmp_lg_u64 s[2:3], 0
	v_add_u32_e32 v47, 0x400, v47
	s_cbranch_scc1 .LBB0_2
	s_mul_i32 s2, s4, s5
	s_and_b32 s13, s13, 0xffff
	s_lshl_b32 s14, s2, 4
	s_add_u32 s2, s0, 0xffffff00
	s_addc_u32 s3, s1, -1
	s_lshr_b64 s[6:7], s[2:3], 1
	v_add_u32_e32 v24, s6, v42
	v_add_lshl_u32 v24, v49, v24, 2
	ds_read_b128 v[88:91], v126
	ds_read_b128 v[92:95], v125
	buffer_load_dwordx4 v[96:99], v24, s[24:27], 0 offen
	v_add_u32_e32 v24, s6, v40
	v_add_lshl_u32 v24, v49, v24, 2
	s_lshr_b64 s[6:7], s[2:3], 3
	buffer_load_dwordx4 v[100:103], v24, s[24:27], 0 offen
	v_add_u32_e32 v24, s6, v51
	s_waitcnt vmcnt(5)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v26, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v27, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v40, -4, v24
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[26:27], v[80:81], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[80:81], v[24:25] op_sel_hi:[0,1]
	v_and_b32_e32 v27, 0xffff0000, v27
	v_and_b32_e32 v26, 0xffff0000, v26
	v_or_b32_sdwa v25, v27, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v26, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v26, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v27, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[16:19], v[84:85], v[24:25], v[16:19]
	v_mul_f32_e64 v28, v80, v28
	v_mul_f32_e64 v29, v80, v29
	v_pk_mul_f32 v[26:27], v[80:81], v[26:27] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v29
	v_and_b32_e32 v24, 0xffff0000, v28
	v_or_b32_sdwa v25, v12, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v24, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_lshl_b32 s1, s2, 4
	s_and_b32 s1, s1, 0xfffffe00
	v_mfma_f32_16x16x16_bf16 v[28:31], v[86:87], v[24:25], v[16:19]
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 4, v8
	s_nop 0
	v_pk_mul_f32 v[18:19], v[82:83], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[82:83], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v19
	v_and_b32_e32 v18, 0xffff0000, v18
	v_or_b32_sdwa v17, v12, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v18, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dwordx4 v[24:27], v40, s[16:19], 0 offen
	v_pk_mul_f32 v[46:47], v[82:83], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[82:83], v[44:45] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[84:85], v[16:17], v[20:23]
	v_and_b32_e32 v8, 0xffff0000, v45
	v_and_b32_e32 v12, 0xffff0000, v44
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_lshl_b32 s0, s0, 4
	v_or_b32_sdwa v21, v8, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v12, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[44:45], v[78:79], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[78:79], v[46:47] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[86:87], v[20:21], v[16:19]
	ds_read_b128 v[80:83], v124
	v_and_b32_e32 v12, 0xffff0000, v47
	v_or_b32_sdwa v45, v12, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[16:19], v40, s[16:19], 0 offen offset:16
	v_and_b32_e32 v40, 0xffff0000, v46
	v_or_b32_sdwa v44, v40, v44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v40, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[28:31], v[88:89], v[44:45], v[28:31]
	v_mul_f32_e64 v46, v78, v46
	v_mul_f32_e64 v47, v78, v47
	v_pk_mul_f32 v[12:13], v[78:79], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v40, 0xffff0000, v47
	v_and_b32_e32 v42, 0xffff0000, v46
	v_or_b32_sdwa v13, v40, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v42, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v8, v38, s1, 2
	v_pk_mul_f32 v[44:45], v[76:77], v[44:45] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[90:91], v[12:13], v[28:31]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	buffer_load_dword v40, v8, s[20:23], 0 offen
	v_pk_mul_f32 v[12:13], v[76:77], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v45
	v_and_b32_e32 v44, 0xffff0000, v44
	v_or_b32_sdwa v13, v8, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v44, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v9
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[20:23], v[88:89], v[12:13], v[20:23]
	v_mul_f32_e64 v44, v76, v44
	v_mul_f32_e64 v45, v76, v45
	v_pk_mul_f32 v[8:9], v[76:77], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v45
	v_and_b32_e32 v12, 0xffff0000, v44
	v_or_b32_sdwa v9, v45, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_and_b32 s15, s0, 0xfffffe00
	v_pk_mul_f32 v[12:13], v[74:75], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[90:91], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_add_i32 s0, s15, 0xfffff200
	v_pk_mul_f32 v[8:9], v[74:75], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_add_lshl_u32 v42, v38, s0, 2
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	ds_read_b128 v[84:87], v77
	buffer_load_dword v44, v42, s[20:23], 0 offen
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[28:31], v[92:93], v[8:9], v[28:31]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v46, v74, v46
	v_mul_f32_e64 v47, v74, v47
	v_pk_mul_f32 v[12:13], v[74:75], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v47
	v_and_b32_e32 v8, 0xffff0000, v46
	v_or_b32_sdwa v9, v14, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s2, s15, 0xfffff400
	v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[94:95], v[8:9], v[28:31]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v13, 0xffff0000, v13
	v_pk_mul_f32 v[8:9], v[72:73], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_add_lshl_u32 v42, v38, s2, 2
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v50, v42, s[20:23], 0 offen
	v_lshrrev_b32_e32 v10, 4, v10
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[20:23], v[92:93], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v46, v72, v46
	v_mul_f32_e64 v47, v72, v47
	v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v47
	v_and_b32_e32 v8, 0xffff0000, v46
	v_or_b32_sdwa v9, v10, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s3, s15, 0xfffff600
	v_pk_mul_f32 v[12:13], v[70:71], v[12:13] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[94:95], v[8:9], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_add_lshl_u32 v14, v38, s3, 2
	v_pk_mul_f32 v[8:9], v[70:71], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_lshrrev_b32_e32 v15, 4, v15
	ds_read_b128 v[72:75], v59
	buffer_load_dword v42, v14, s[20:23], 0 offen
	v_or_b32_sdwa v9, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s5, s15, 0xfffff800
	v_pk_mul_f32 v[46:47], v[70:71], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[70:71], v[14:15] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[80:81], v[8:9], v[28:31]
	v_and_b32_e32 v45, 0xffff0000, v49
	v_and_b32_e32 v8, 0xffff0000, v48
	v_or_b32_sdwa v9, v45, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v10, v38, s5, 2
	v_pk_mul_f32 v[28:29], v[68:69], v[28:29] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[82:83], v[8:9], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	buffer_load_dword v48, v10, s[20:23], 0 offen
	v_pk_mul_f32 v[8:9], v[68:69], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v10, 0xffff0000, v29
	v_and_b32_e32 v28, 0xffff0000, v28
	v_or_b32_sdwa v9, v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v28, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v29, 4, v11
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s7, s15, 0xfffffa00
	v_pk_mul_f32 v[30:31], v[68:69], v[10:11] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[80:81], v[8:9], v[20:23]
	v_mul_f32_e64 v28, v68, v28
	v_mul_f32_e64 v29, v68, v29
	v_and_b32_e32 v29, 0xffff0000, v29
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v45, v38, s7, 2
	v_and_b32_e32 v20, 0xffff0000, v28
	v_or_b32_sdwa v21, v29, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v20, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[22:23], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_and_b32_e32 v23, 0xffff0000, v23
	v_mfma_f32_16x16x16_bf16 v[8:11], v[82:83], v[20:21], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v22, 0xffff0000, v22
	v_pk_mul_f32 v[20:21], v[66:67], v[20:21] op_sel_hi:[0,1]
	v_or_b32_sdwa v21, v23, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v22, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	ds_read_b128 v[78:81], v35
	buffer_load_dword v68, v45, s[20:23], 0 offen
	v_lshrrev_b32_e32 v4, 4, v4
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[84:85], v[20:21], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v28, v66, v28
	v_mul_f32_e64 v29, v66, v29
	v_pk_mul_f32 v[22:23], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v29
	v_and_b32_e32 v20, 0xffff0000, v28
	v_or_b32_sdwa v21, v4, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_i32 s11, s15, 0xfffffc00
	v_pk_mul_f32 v[22:23], v[64:65], v[22:23] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[86:87], v[20:21], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v23, 0xffff0000, v23
	v_pk_mul_f32 v[20:21], v[64:65], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v22, 0xffff0000, v22
	v_add_lshl_u32 v30, v38, s11, 2
	v_or_b32_sdwa v21, v23, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v22, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v66, v30, s[20:23], 0 offen
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[84:85], v[20:21], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v28, v64, v28
	v_mul_f32_e64 v29, v64, v29
	v_pk_mul_f32 v[22:23], v[64:65], v[22:23] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v29
	v_and_b32_e32 v20, 0xffff0000, v28
	s_addk_i32 s15, 0xfe00
	v_or_b32_sdwa v21, v0, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v4, v38, s15, 2
	v_add_u32_e32 v0, s6, v43
	v_mfma_f32_16x16x16_bf16 v[8:11], v[86:87], v[20:21], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v22, v62, v22
	v_mul_f32_e64 v23, v62, v23
	ds_read_b128 v[82:85], v33
	buffer_load_dword v64, v4, s[20:23], 0 offen
	v_and_b32_e32 v38, -4, v0
	v_pk_mul_f32 v[20:21], v[62:63], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v23
	v_and_b32_e32 v4, 0xffff0000, v22
	v_or_b32_sdwa v21, v0, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v4, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v0, 4, v5
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[12:15], v[72:73], v[20:21], v[12:15]
	v_mul_f32_e64 v22, v62, v22
	v_mul_f32_e64 v23, v62, v23
	v_pk_mul_f32 v[4:5], v[62:63], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v23
	v_and_b32_e32 v20, 0xffff0000, v22
	v_or_b32_sdwa v5, v0, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v4, v20, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[28:29], v[58:59], v[28:29] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[74:75], v[4:5], v[12:15]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v0, 0xffff0000, v29
	v_pk_mul_f32 v[4:5], v[58:59], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v28, 0xffff0000, v28
	v_or_b32_sdwa v5, v0, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v4, v28, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v29, 4, v1
	buffer_load_dwordx4 v[12:15], v38, s[16:19], 0 offen
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[72:73], v[4:5], v[8:11]
	v_mul_f32_e64 v28, v58, v28
	v_mul_f32_e64 v29, v58, v29
	v_pk_mul_f32 v[0:1], v[58:59], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v29, 0xffff0000, v29
	v_and_b32_e32 v4, 0xffff0000, v28
	v_or_b32_sdwa v1, v29, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[4:5], v[60:61], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[74:75], v[0:1], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v5, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[60:61], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[8:11], v38, s[16:19], 0 offen offset:16
	v_lshrrev_b32_e32 v6, 4, v6
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[20:23], v[78:79], v[0:1], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v46, v60, v46
	v_mul_f32_e64 v47, v60, v47
	v_pk_mul_f32 v[4:5], v[60:61], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v6, 0xffff0000, v47
	v_and_b32_e32 v0, 0xffff0000, v46
	v_or_b32_sdwa v1, v6, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v38, v36, s1, 2
	v_pk_mul_f32 v[4:5], v[56:57], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[80:81], v[0:1], v[20:23]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v5, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[56:57], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(13)
	ds_write_b128 v41, v[96:99] offset:8192
	buffer_load_dword v72, v38, s[20:23], 0 offen
	v_lshrrev_b32_e32 v2, 4, v2
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[28:31], v[78:79], v[0:1], v[28:31]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v46, v56, v46
	v_mul_f32_e64 v47, v56, v47
	v_pk_mul_f32 v[4:5], v[56:57], v[4:5] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v47
	v_and_b32_e32 v0, 0xffff0000, v46
	v_or_b32_sdwa v1, v2, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v6, v36, s0, 2
	v_pk_mul_f32 v[4:5], v[54:55], v[4:5] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[80:81], v[0:1], v[28:31]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v5, 0xffff0000, v5
	v_pk_mul_f32 v[0:1], v[54:55], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v4, 0xffff0000, v4
	v_lshrrev_b32_e32 v7, 4, v7
	buffer_load_dword v70, v6, s[20:23], 0 offen
	v_or_b32_sdwa v1, v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v6, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v7, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v2, v36, s2, 2
	v_pk_mul_f32 v[46:47], v[54:55], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[54:55], v[6:7] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[82:83], v[0:1], v[20:23]
	v_and_b32_e32 v38, 0xffff0000, v55
	v_and_b32_e32 v0, 0xffff0000, v54
	v_or_b32_sdwa v1, v38, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v0, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_lshl_u32 v38, v36, s3, 2
	v_pk_mul_f32 v[20:21], v[52:53], v[20:21] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[84:85], v[0:1], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(14)
	ds_write_b128 v39, v[100:103] offset:8192
	buffer_load_dword v62, v2, s[20:23], 0 offen
	v_pk_mul_f32 v[0:1], v[52:53], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v21
	v_and_b32_e32 v20, 0xffff0000, v20
	v_or_b32_sdwa v1, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v0, v20, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v3
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[22:23], v[52:53], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[82:83], v[0:1], v[28:31]
	v_mul_f32_e64 v20, v52, v20
	v_mul_f32_e64 v21, v52, v21
	v_and_b32_e32 v21, 0xffff0000, v21
	v_and_b32_e32 v20, 0xffff0000, v20
	v_or_b32_sdwa v21, v21, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[84:85], v[20:21], v[0:3]
	v_add_lshl_u32 v20, v36, s5, 2
	buffer_load_dword v30, v20, s[20:23], 0 offen
	v_add_lshl_u32 v20, v36, s7, 2
	buffer_load_dword v28, v20, s[20:23], 0 offen
	v_add_lshl_u32 v20, v36, s11, 2
	buffer_load_dword v22, v20, s[20:23], 0 offen
	v_add_lshl_u32 v20, v36, s15, 2
	buffer_load_dword v58, v38, s[20:23], 0 offen
	s_mov_b32 s15, 0x27000
	buffer_load_dword v20, v20, s[20:23], 0 offen
	s_waitcnt vmcnt(19)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[78:81], v37 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(17)
	v_pk_mul_f32 v[38:39], v[40:41], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[40:41], v[36:37] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v39
	v_and_b32_e32 v23, 0xffff0000, v38
	v_or_b32_sdwa v37, v21, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v24
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v36, v23, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[38:39], v[40:41], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[40:41], v[46:47] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v41
	v_and_b32_e32 v23, 0xffff0000, v40
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(16)
	v_pk_mul_f32 v[46:47], v[44:45], v[46:47] op_sel_hi:[0,1]
	v_or_b32_sdwa v39, v21, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[40:41], v[44:45], v[40:41] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v47
	v_or_b32_sdwa v41, v21, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v25
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v38, v23, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v23, 0xffff0000, v46
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[24:25], v[44:45], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[44:45], v[46:47] op_sel_hi:[0,1]
	v_or_b32_sdwa v40, v23, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v21, 0xffff0000, v45
	v_and_b32_e32 v23, 0xffff0000, v44
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v44, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v45, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v47, v21, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v46, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(15)
	v_pk_mul_f32 v[44:45], v[50:51], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[50:51], v[24:25] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v45
	v_and_b32_e32 v23, 0xffff0000, v44
	v_or_b32_sdwa v45, v21, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v44, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v26
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v56, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v57, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	ds_read_b128 v[82:85], v126 offset:8192
	v_pk_mul_f32 v[24:25], v[50:51], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[50:51], v[52:53] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v51
	v_and_b32_e32 v23, 0xffff0000, v50
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v51, v21, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v50, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(14)
	v_pk_mul_f32 v[52:53], v[42:43], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[42:43], v[24:25] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v53
	v_and_b32_e32 v23, 0xffff0000, v52
	v_or_b32_sdwa v25, v21, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v27
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v26, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v27, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v24, v23, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[26:27], v[42:43], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[42:43], v[52:53] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v43
	v_and_b32_e32 v23, 0xffff0000, v42
	s_waitcnt vmcnt(13)
	v_pk_mul_f32 v[52:53], v[48:49], v[52:53] op_sel_hi:[0,1]
	v_or_b32_sdwa v27, v21, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v26, v23, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v42, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v43, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v53
	v_and_b32_e32 v23, 0xffff0000, v52
	v_lshrrev_b32_e32 v16, 4, v16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[42:43], v[48:49], v[42:43] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v54, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v55, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[52:53], v[48:49], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[48:49], v[54:55] op_sel_hi:[0,1]
	v_or_b32_sdwa v43, v21, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v16, 0xffff0000, v49
	v_and_b32_e32 v21, 0xffff0000, v48
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v54, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v55, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v49, v16, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v48, v21, v52 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(12)
	v_pk_mul_f32 v[54:55], v[68:69], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[68:69], v[52:53] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v54
	v_and_b32_e32 v16, 0xffff0000, v55
	v_or_b32_sdwa v52, v21, v52 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v17
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v54, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v55, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v53, v16, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[54:55], v[68:69], v[54:55] op_sel_hi:[0,1]
	v_or_b32_sdwa v42, v23, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[16:17], v[68:69], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v55
	v_and_b32_e32 v23, 0xffff0000, v54
	s_waitcnt vmcnt(11)
	v_pk_mul_f32 v[56:57], v[66:67], v[56:57] op_sel_hi:[0,1]
	v_or_b32_sdwa v17, v21, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v23, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v54, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v55, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v57
	v_and_b32_e32 v23, 0xffff0000, v56
	v_lshrrev_b32_e32 v18, 4, v18
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v56, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v57, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[54:55], v[66:67], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[66:67], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[66:67], v[60:61] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v66, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v67, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v55, v21, v55 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v18, 0xffff0000, v61
	v_and_b32_e32 v21, 0xffff0000, v60
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(10)
	v_pk_mul_f32 v[66:67], v[64:65], v[66:67] op_sel_hi:[0,1]
	v_or_b32_sdwa v57, v18, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v56, v21, v56 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[60:61], v[64:65], v[60:61] op_sel_hi:[0,1]
	v_and_b32_e32 v18, 0xffff0000, v67
	v_and_b32_e32 v21, 0xffff0000, v66
	v_or_b32_sdwa v61, v18, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v60, v21, v60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v19
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v66, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v67, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v54, v23, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[18:19], v[64:65], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[0,1]
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v66, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v67, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v65
	v_and_b32_e32 v23, 0xffff0000, v64
	s_waitcnt vmcnt(7)
	v_pk_mul_f32 v[66:67], v[72:73], v[66:67] op_sel_hi:[0,1]
	v_or_b32_sdwa v19, v21, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v18, v23, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v64, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v65, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v67
	v_and_b32_e32 v23, 0xffff0000, v66
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v66, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v67, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[64:65], v[72:73], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[72:73], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[72:73], v[68:69] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v65, v21, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v12, 0xffff0000, v69
	v_and_b32_e32 v21, 0xffff0000, v68
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(6)
	v_pk_mul_f32 v[72:73], v[70:71], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v67, v12, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v66, v21, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[68:69], v[70:71], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v73
	v_and_b32_e32 v21, 0xffff0000, v72
	v_or_b32_sdwa v69, v12, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v21, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v64, v23, v64 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[12:13], v[70:71], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[70:71], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v71
	v_and_b32_e32 v23, 0xffff0000, v70
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v71, v21, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v23, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(5)
	v_pk_mul_f32 v[72:73], v[62:63], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[62:63], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v73
	v_and_b32_e32 v23, 0xffff0000, v72
	v_or_b32_sdwa v73, v21, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v23, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[78:79], v[36:37], v[4:7]
	v_mul_f32_e64 v12, v62, v12
	v_mul_f32_e64 v13, v62, v13
	v_pk_mul_f32 v[62:63], v[62:63], v[74:75] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v63
	v_and_b32_e32 v21, 0xffff0000, v62
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v63, v14, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v21, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[74:75], v[58:59], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[58:59], v[12:13] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v74
	v_and_b32_e32 v14, 0xffff0000, v75
	v_or_b32_sdwa v12, v21, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v21, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[80:81], v[38:39], v[4:7]
	v_or_b32_sdwa v13, v14, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[74:75], v[58:59], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[58:59], v[14:15] op_sel_hi:[0,1]
	v_and_b32_e32 v21, 0xffff0000, v75
	v_and_b32_e32 v23, 0xffff0000, v74
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v15, v21, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[38:39], v[30:31], v[74:75] op_sel_hi:[0,1]
	v_or_b32_sdwa v14, v23, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v21, 0xffff0000, v39
	v_and_b32_e32 v23, 0xffff0000, v38
	v_lshrrev_b32_e32 v8, 4, v8
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[36:37], v[30:31], v[36:37] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[82:83], v[40:41], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v38, v30, v38
	v_mul_f32_e64 v39, v30, v39
	v_pk_mul_f32 v[30:31], v[30:31], v[40:41] op_sel_hi:[0,1]
	v_or_b32_sdwa v37, v21, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v8, 0xffff0000, v31
	v_and_b32_e32 v21, 0xffff0000, v30
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v31, v8, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v30, v21, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[40:41], v[28:29], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[28:29], v[38:39] op_sel_hi:[0,1]
	v_and_b32_e32 v8, 0xffff0000, v41
	s_lshl_b32 s0, s10, 2
	v_or_b32_sdwa v39, v8, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshl_or_b32 v8, v34, 4, s0
	s_mov_b32 s10, s18
	s_mov_b32 s11, s19
	buffer_load_dword v21, v8, s[8:11], 0 offen
	v_mfma_f32_16x16x16_bf16 v[0:3], v[78:79], v[64:65], v[0:3]
	v_or_b32_sdwa v36, v23, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v9, 4, v9
	v_and_b32_e32 v23, 0xffff0000, v40
	v_mfma_f32_16x16x16_bf16 v[0:3], v[80:81], v[66:67], v[0:3]
	ds_read_b128 v[64:67], v125 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v40, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v41, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[82:83], v[68:69], v[0:3]
	v_or_b32_sdwa v38, v23, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[84:85], v[46:47], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[84:85], v[70:71], v[0:3]
	ds_read_b128 v[68:71], v124 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[64:65], v[44:45], v[4:7]
	ds_read_b128 v[44:47], v77 offset:8192
	v_mfma_f32_16x16x16_bf16 v[0:3], v[64:65], v[72:73], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[66:67], v[50:51], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[66:67], v[62:63], v[0:3]
	ds_read_b128 v[62:65], v59 offset:8192
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[68:69], v[24:25], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[68:69], v[12:13], v[0:3]
	v_mul_f32_e64 v12, v28, v24
	v_mul_f32_e64 v13, v28, v25
	v_pk_mul_f32 v[24:25], v[28:29], v[40:41] op_sel_hi:[0,1]
	v_and_b32_e32 v9, 0xffff0000, v25
	v_mfma_f32_16x16x16_bf16 v[4:7], v[70:71], v[26:27], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v13, v9, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[70:71], v[14:15], v[0:3]
	v_and_b32_e32 v14, 0xffff0000, v24
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_or_b32_sdwa v12, v14, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[44:45], v[42:43], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v24, v22, v24
	v_mul_f32_e64 v25, v22, v25
	v_mfma_f32_16x16x16_bf16 v[0:3], v[44:45], v[36:37], v[0:3]
	v_mul_f32_e64 v14, v22, v14
	v_mul_f32_e64 v15, v22, v15
	v_and_b32_e32 v9, 0xffff0000, v25
	v_and_b32_e32 v23, 0xffff0000, v24
	v_mfma_f32_16x16x16_bf16 v[4:7], v[46:47], v[48:49], v[4:7]
	v_or_b32_sdwa v15, v9, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v9, 4, v10
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v24, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[46:47], v[30:31], v[0:3]
	ds_read_b128 v[28:31], v35 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v26, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v25, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[62:63], v[52:53], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v27, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v14, v23, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[64:65], v[16:17], v[4:7]
	v_mul_f32_e64 v16, v22, v24
	v_mul_f32_e64 v17, v22, v25
	v_pk_mul_f32 v[22:23], v[22:23], v[26:27] op_sel_hi:[0,1]
	ds_read_b128 v[24:27], v33 offset:8192
	v_mfma_f32_16x16x16_bf16 v[0:3], v[62:63], v[38:39], v[0:3]
	v_and_b32_e32 v9, 0xffff0000, v23
	v_and_b32_e32 v10, 0xffff0000, v22
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[64:65], v[12:13], v[0:3]
	v_or_b32_sdwa v13, v9, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v10, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[28:29], v[54:55], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[22:23], v[20:21], v[22:23] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[56:57], v[4:7]
	v_mul_f32_e64 v16, v20, v16
	v_mul_f32_e64 v17, v20, v17
	v_and_b32_e32 v9, 0xffff0000, v23
	v_and_b32_e32 v10, 0xffff0000, v22
	v_mfma_f32_16x16x16_bf16 v[0:3], v[28:29], v[14:15], v[0:3]
	v_or_b32_sdwa v17, v9, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v10, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v9, 4, v11
	v_mfma_f32_16x16x16_bf16 v[12:15], v[30:31], v[12:13], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[24:25], v[60:61], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v10, v20, v10
	v_mul_f32_e64 v11, v20, v11
	v_mfma_f32_16x16x16_bf16 v[0:3], v[26:27], v[18:19], v[0:3]
	v_mul_f32_e64 v18, v20, v4
	v_mul_f32_e64 v19, v20, v5
	v_and_b32_e32 v9, 0xffff0000, v19
	v_or_b32_sdwa v11, v9, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[24:25], v[16:17], v[12:15]
	v_and_b32_e32 v9, 0xffffff, v21
	v_cmp_gt_u32_e32 vcc, s4, v9
	s_nop 0
	v_and_b32_e32 v12, 0xffff0000, v18
	v_or_b32_sdwa v10, v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[26:27], v[10:11], v[4:7]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
	s_nop 5
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v0
	s_mov_b32 s2, 0x41800000
	v_pk_mul_f32 v[10:11], v[10:11], s[2:3] op_sel_hi:[1,0]
	v_mov_b32_e32 v4, 8
	v_mul_f32_e32 v0, 0xbfb8aa3b, v11
	v_exp_f32_e32 v0, v0
	v_lshl_add_u32 v9, v21, 11, v32
	v_lshlrev_b32_sdwa v4, v4, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_add_lshl_u32 v4, v9, v4, 1
	v_add_f32_e32 v0, 1.0, v0
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, v11, v0
	v_fma_mixlo_f16 v0, v10, v0, 0
	buffer_store_short v0, v4, s[12:15], 0 offen
.LBB0_5:
	s_or_b64 exec, exec, s[0:1]
	s_nop 4
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
	v_lshl_add_u32 v10, v4, 11, v32
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
	v_lshl_add_u32 v6, v0, 11, v32
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
	v_lshl_add_u32 v5, v0, 11, v32
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
		.amdhsa_next_free_vgpr 168
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

	.set moe_gemm1_0.num_vgpr, 168
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
    .vgpr_count:     168
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
