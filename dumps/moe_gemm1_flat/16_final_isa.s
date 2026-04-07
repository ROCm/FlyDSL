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
	s_load_dwordx4 s[36:39], s[0:1], 0x20
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx4 s[4:7], s[0:1], 0x48
	s_lshl_b32 s3, s8, 2
	s_mov_b32 s15, s27
	v_mov_b32_e32 v1, s3
	s_waitcnt lgkmcnt(0)
	s_and_b32 s13, s13, 0xffff
	s_lshl_b32 s14, s7, 2
	buffer_load_dword v6, v1, s[12:15], 0 offen
	v_lshrrev_b32_e32 v27, 5, v0
	s_mov_b32 s22, -1
	s_and_b32 s9, s39, 0xffff
	v_or_b32_e32 v1, s10, v27
	v_or_b32_e32 v29, 8, v27
	s_mov_b32 s16, s38
	s_mov_b32 s17, s9
	s_mov_b32 s18, s22
	s_mov_b32 s19, s27
	v_lshlrev_b32_e32 v1, 2, v1
	v_or_b32_e32 v2, s10, v29
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_load_dword v3, v1, s[16:19], 0 offen
	buffer_load_dword v4, v2, s[16:19], 0 offen
	s_ashr_i32 s3, s2, 31
	v_lshlrev_b32_e32 v1, 2, v0
	v_bfe_u32 v42, v0, 4, 2
	v_and_b32_e32 v30, 15, v0
	v_lshrrev_b32_e32 v0, 2, v0
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_load_dwordx2 s[20:21], s[0:1], 0x10
	s_lshl_b64 s[0:1], s[2:3], 6
	v_and_b32_e32 v0, 48, v0
	v_and_b32_e32 v97, 0x7c, v1
	v_or3_b32 v40, s0, v0, v30
	v_mov_b32_e32 v41, s1
	s_ashr_i32 s1, s6, 31
	s_mov_b32 s0, s6
	s_ashr_i32 s34, s6, 3
	s_mul_i32 s6, s4, s6
	s_mov_b32 s33, 0x2aaaaaab
	s_lshl_b32 s42, s6, 1
	s_lshr_b64 s[6:7], s[0:1], 1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s41, s15, 0xffff
	s_mov_b32 s40, s14
	s_mov_b32 s24, s14
	v_mov_b32_e32 v5, 0
	s_lshr_b64 s[2:3], s[0:1], 3
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s17, s37, 0xffff
	s_mov_b32 s16, s36
	s_mov_b32 s8, s38
	s_mov_b32 s25, s41
	s_lshl_b32 s3, s2, 6
	s_mov_b32 s26, s42
	s_mov_b32 s43, s27
	s_mov_b32 s23, s27
	s_mov_b32 s31, s27
	s_mov_b32 s30, s22
	s_mov_b32 s28, s36
	s_mov_b32 s29, s17
	s_waitcnt vmcnt(2)
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshlrev_b64 v[0:1], 9, v[6:7]
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[40:41]
	v_ashrrev_i32_e32 v0, 31, v8
	v_add_u32_e32 v7, 0x100, v8
	v_lshrrev_b32_e32 v0, 28, v0
	v_ashrrev_i32_e32 v1, 31, v7
	v_add_u32_e32 v0, v8, v0
	v_lshrrev_b32_e32 v1, 28, v1
	v_ashrrev_i32_e32 v9, 4, v0
	v_add_u32_e32 v1, v7, v1
	v_and_b32_e32 v0, -16, v0
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, 0xffffff, v3
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v4
	v_cmp_gt_u32_e32 vcc, s4, v2
	v_ashrrev_i32_e32 v31, 4, v1
	v_sub_u32_e32 v4, v8, v0
	v_cndmask_b32_e32 v2, 0, v2, vcc
	v_cmp_gt_u32_e32 vcc, s4, v3
	v_mad_u64_u32 v[50:51], s[14:15], s6, v2, 0
	s_nop 0
	v_cndmask_b32_e32 v3, 0, v3, vcc
	v_mul_hi_i32 v2, v9, s33
	v_mad_u64_u32 v[48:49], s[6:7], s6, v3, 0
	v_lshrrev_b32_e32 v0, 31, v2
	v_ashrrev_i32_e32 v2, 11, v2
	v_mul_hi_i32 v3, v31, s33
	v_add_u32_e32 v0, v2, v0
	v_lshrrev_b32_e32 v2, 31, v3
	v_ashrrev_i32_e32 v3, 11, v3
	v_mul_i32_i24_e32 v0, 0x3000, v0
	v_add_u32_e32 v2, v3, v2
	v_sub_u32_e32 v10, v9, v0
	v_mul_i32_i24_e32 v0, 0x3000, v2
	v_sub_u32_e32 v24, v31, v0
	v_and_b32_e32 v0, -16, v1
	v_ashrrev_i32_e32 v11, 31, v10
	v_sub_u32_e32 v0, v7, v0
	v_mov_b32_e32 v1, v5
	v_ashrrev_i32_e32 v25, 31, v24
	s_mov_b32 s14, 0x1be00
	v_mul_lo_u32 v28, v0, s34
	v_mad_i64_i32 v[2:3], s[6:7], v6, s14, v[4:5]
	v_mad_i64_i32 v[0:1], s[6:7], v6, s14, v[0:1]
	v_mul_lo_u32 v13, s2, v42
	v_lshl_add_u64 v[56:57], v[10:11], 4, v[2:3]
	v_lshl_add_u64 v[54:55], v[24:25], 4, v[0:1]
	v_mul_lo_u32 v12, v10, s3
	v_mul_lo_u32 v26, v4, s34
	v_mul_lo_u32 v14, v24, s3
	v_lshlrev_b32_e32 v4, 4, v13
	v_lshlrev_b32_e32 v0, 2, v56
	v_lshlrev_b32_e32 v1, 2, v54
	v_add3_u32 v111, v26, v4, v12
	v_add3_u32 v109, v28, v4, v14
	v_add_u32_e32 v2, 0x1000, v0
	v_add_u32_e32 v3, 0x2000, v0
	v_add_u32_e32 v11, 0x3000, v0
	v_add_u32_e32 v12, 0x1000, v1
	v_add_u32_e32 v13, 0x2000, v1
	v_add_u32_e32 v14, 0x3000, v1
	buffer_load_dword v90, v0, s[28:31], 0 offen
	buffer_load_dword v88, v0, s[28:31], 0 offen offset:2048
	buffer_load_dword v86, v2, s[28:31], 0 offen
	buffer_load_dword v84, v2, s[28:31], 0 offen offset:2048
	buffer_load_dword v82, v3, s[28:31], 0 offen
	buffer_load_dword v80, v3, s[28:31], 0 offen offset:2048
	buffer_load_dword v78, v11, s[28:31], 0 offen
	buffer_load_dword v76, v11, s[28:31], 0 offen offset:2048
	buffer_load_dword v74, v1, s[28:31], 0 offen
	buffer_load_dword v72, v1, s[28:31], 0 offen offset:2048
	buffer_load_dword v70, v12, s[28:31], 0 offen
	buffer_load_dword v68, v12, s[28:31], 0 offen offset:2048
	buffer_load_dword v58, v13, s[28:31], 0 offen
	buffer_load_dword v52, v13, s[28:31], 0 offen offset:2048
	buffer_load_dword v46, v14, s[28:31], 0 offen
	buffer_load_dword v44, v14, s[28:31], 0 offen offset:2048
	v_add_lshl_u32 v32, v50, v97, 2
	v_add_lshl_u32 v33, v48, v97, 2
	buffer_load_dwordx4 v[98:101], v32, s[40:43], 0 offen
	buffer_load_dwordx4 v[102:105], v33, s[40:43], 0 offen
	v_and_b32_e32 v4, -4, v111
	v_and_b32_e32 v11, -4, v109
	buffer_load_dwordx4 v[20:23], v4, s[20:23], 0 offen
	buffer_load_dwordx4 v[16:19], v4, s[20:23], 0 offen offset:16
	buffer_load_dwordx4 v[12:15], v11, s[20:23], 0 offen
	buffer_load_dwordx4 v[0:3], v11, s[20:23], 0 offen offset:16
	v_lshlrev_b32_e32 v4, 9, v27
	v_lshlrev_b32_e32 v11, 4, v27
	v_lshlrev_b32_e32 v25, 2, v97
	v_lshlrev_b32_e32 v34, 4, v30
	v_lshlrev_b32_e32 v30, 9, v30
	v_lshlrev_b32_e32 v35, 4, v42
	v_lshlrev_b32_e32 v27, 9, v29
	v_lshlrev_b32_e32 v29, 4, v29
	s_mov_b32 s3, 0x6f800
	v_bitop3_b32 v67, v4, v25, v11 bitop3:0xf6
	v_bitop3_b32 v65, v30, v35, v34 bitop3:0xf6
	v_lshlrev_b32_e32 v24, 6, v24
	v_lshlrev_b32_e32 v10, 6, v10
	v_bitop3_b32 v95, v27, v29, v25 bitop3:0xf6
	v_or_b32_e32 v4, 64, v35
	v_mul_lo_u32 v6, v6, s3
	v_or_b32_e32 v11, 0x80, v35
	v_or_b32_e32 v25, 0xc0, v35
	v_or_b32_e32 v27, 0x100, v35
	v_or_b32_e32 v29, 0x140, v35
	v_or_b32_e32 v36, 0x180, v35
	v_or_b32_e32 v37, 0x1c0, v35
	v_or_b32_e32 v38, v24, v35
	v_or_b32_e32 v35, v10, v35
	v_bitop3_b32 v93, v30, v4, v34 bitop3:0xf6
	v_add_u32_e32 v4, v6, v24
	v_mad_u64_u32 v[60:61], s[6:7], v38, s2, v[28:29]
	v_mad_u64_u32 v[62:63], s[2:3], v35, s2, v[26:27]
	v_add_u32_e32 v6, v6, v10
	v_lshl_add_u32 v4, v7, 2, v4
	v_lshlrev_b32_e32 v7, 6, v31
	v_lshl_add_u32 v6, v8, 2, v6
	v_sub_u32_e32 v64, v4, v7
	v_lshlrev_b32_e32 v4, 6, v9
	s_mov_b32 s2, 0xfff98000
	v_bitop3_b32 v57, v30, v11, v34 bitop3:0xf6
	v_bitop3_b32 v55, v30, v25, v34 bitop3:0xf6
	v_bitop3_b32 v51, v30, v27, v34 bitop3:0xf6
	v_bitop3_b32 v49, v30, v29, v34 bitop3:0xf6
	v_bitop3_b32 v43, v30, v36, v34 bitop3:0xf6
	v_bitop3_b32 v41, v30, v37, v34 bitop3:0xf6
	v_sub_u32_e32 v66, v6, v4
	v_add_u32_e32 v61, 0x400, v33
	v_add_u32_e32 v63, 0x400, v32
	s_mov_b32 s3, -1
	v_mov_b32_e32 v4, v5
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v8, v5
	s_waitcnt vmcnt(5)
	ds_write_b128 v67, v[98:101]
	s_waitcnt vmcnt(4)
	ds_write_b128 v95, v[102:105]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[144:147], v65
	v_mov_b32_e32 v9, v5
	v_mov_b32_e32 v10, v5
	v_mov_b32_e32 v11, v5
.LBB0_2:
	v_add_u32_e32 v99, s2, v66
	v_add_u32_e32 v24, 0x6c000, v99
	buffer_load_dword v106, v24, s[16:19], 0 offen
	buffer_load_dword v104, v24, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v24, 0x6d000, v99
	buffer_load_dword v102, v24, s[16:19], 0 offen
	buffer_load_dword v100, v24, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v24, 0x6e000, v99
	buffer_load_dword v98, v24, s[16:19], 0 offen
	buffer_load_dword v96, v24, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v24, 0x6f000, v99
	v_add_u32_e32 v101, s11, v62
	buffer_load_dword v94, v24, s[16:19], 0 offen
	buffer_load_dword v92, v24, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v24, 32, v101
	v_and_b32_e32 v32, -4, v24
	s_mov_b32 s22, s18
	s_mov_b32 s23, s19
	s_waitcnt vmcnt(11)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	buffer_load_dwordx4 v[28:31], v32, s[20:23], 0 offen
	buffer_load_dwordx4 v[24:27], v32, s[20:23], 0 offen offset:16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[90:91], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[90:91], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v131, v35, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v130, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v20, 4, v20
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_add_u32_e32 v113, s2, v64
	v_pk_mul_f32 v[34:35], v[90:91], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[90:91], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v34, 0xffff0000, v34
	v_and_b32_e32 v20, 0xffff0000, v35
	v_or_b32_sdwa v132, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v133, v20, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[88:89], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[88:89], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v20, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v91, v20, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v33, 4, v21
	v_or_b32_sdwa v90, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_add_u32_e32 v115, s11, v60
	v_pk_mul_f32 v[32:33], v[88:89], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[88:89], v[20:21] op_sel_hi:[0,1]
	v_and_b32_e32 v33, 0xffff0000, v33
	v_and_b32_e32 v32, 0xffff0000, v32
	v_or_b32_sdwa v21, v33, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v32, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[86:87], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[86:87], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v89, v35, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v88, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v22, 4, v22
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt vmcnt(11)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_add_u32 s2, s2, 0x8000
	v_pk_mul_f32 v[34:35], v[86:87], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[86:87], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v34, 0xffff0000, v34
	v_and_b32_e32 v22, 0xffff0000, v35
	v_or_b32_sdwa v86, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v87, v22, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[84:85], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[84:85], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v22, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v125, v22, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v33, 4, v23
	v_or_b32_sdwa v124, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[134:135], v[74:75], v[134:135] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[84:85], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[84:85], v[22:23] op_sel_hi:[0,1]
	v_and_b32_e32 v33, 0xffff0000, v33
	v_and_b32_e32 v32, 0xffff0000, v32
	v_or_b32_sdwa v23, v33, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v22, v32, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[82:83], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[82:83], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v85, v35, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v84, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v16, 4, v16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v47, 0xffff0000, v134
	v_pk_mul_f32 v[34:35], v[82:83], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[82:83], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v34, 0xffff0000, v34
	v_and_b32_e32 v16, 0xffff0000, v35
	v_or_b32_sdwa v82, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v83, v16, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[80:81], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[80:81], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v16, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v127, v16, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v33, 4, v17
	v_or_b32_sdwa v126, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_addc_u32 s3, s3, 0
	v_pk_mul_f32 v[32:33], v[80:81], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[80:81], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v33, 0xffff0000, v33
	v_and_b32_e32 v32, 0xffff0000, v32
	v_or_b32_sdwa v17, v33, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v32, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[78:79], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[78:79], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v35, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v81, v35, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v80, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v18, 4, v18
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_add_i32 s11, s11, 64
	v_pk_mul_f32 v[34:35], v[78:79], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[78:79], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v34, 0xffff0000, v34
	v_and_b32_e32 v18, 0xffff0000, v35
	v_or_b32_sdwa v78, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v79, v18, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[76:77], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[76:77], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v18, 0xffff0000, v35
	v_and_b32_e32 v34, 0xffff0000, v34
	v_or_b32_sdwa v129, v18, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v33, 4, v19
	v_or_b32_sdwa v128, v34, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_cmp_lg_u64 s[2:3], 0
	v_pk_mul_f32 v[32:33], v[76:77], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[76:77], v[18:19] op_sel_hi:[0,1]
	v_and_b32_e32 v32, 0xffff0000, v32
	v_or_b32_sdwa v18, v32, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v32, 0x6c000, v113
	buffer_load_dword v122, v32, s[16:19], 0 offen
	buffer_load_dword v120, v32, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v32, 0x6d000, v113
	buffer_load_dword v118, v32, s[16:19], 0 offen
	buffer_load_dword v116, v32, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v32, 0x6e000, v113
	buffer_load_dword v114, v32, s[16:19], 0 offen
	buffer_load_dword v112, v32, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v32, 0x6f000, v113
	buffer_load_dword v110, v32, s[16:19], 0 offen
	buffer_load_dword v108, v32, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v32, 32, v115
	v_and_b32_e32 v33, 0xffff0000, v33
	v_and_b32_e32 v45, -4, v32
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_or_b32_sdwa v19, v33, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[36:39], v45, s[20:23], 0 offen
	buffer_load_dwordx4 v[32:35], v45, s[20:23], 0 offen offset:16
	v_pk_mul_f32 v[76:77], v[74:75], v[76:77] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v135
	v_or_b32_sdwa v137, v45, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v136, v47, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v150, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_pk_mul_f32 v[76:77], v[74:75], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[74:75], v[134:135] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v75
	v_and_b32_e32 v45, 0xffff0000, v74
	v_or_b32_sdwa v139, v12, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v138, v45, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v151, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[76:77], v[72:73], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[72:73], v[74:75] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v77
	v_and_b32_e32 v45, 0xffff0000, v76
	v_or_b32_sdwa v75, v12, v75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v45, v74 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[150:151], v[108:109], v[150:151] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[72:73], v[76:77] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v73
	v_and_b32_e32 v47, 0xffff0000, v72
	v_pk_mul_f32 v[76:77], v[70:71], v[76:77] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v45, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v47, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v77
	v_and_b32_e32 v47, 0xffff0000, v76
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[70:71], v[72:73] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[76:77], v[70:71], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[70:71], v[134:135] op_sel_hi:[0,1]
	v_or_b32_sdwa v73, v45, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v14, 0xffff0000, v71
	v_and_b32_e32 v45, 0xffff0000, v70
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v71, v14, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v45, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[134:135], v[68:69], v[134:135] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[68:69], v[76:77] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v135
	v_and_b32_e32 v45, 0xffff0000, v134
	v_or_b32_sdwa v77, v14, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v76, v45, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v72, v47, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[14:15], v[68:69], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[68:69], v[134:135] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v69
	v_and_b32_e32 v47, 0xffff0000, v68
	v_pk_mul_f32 v[134:135], v[58:59], v[134:135] op_sel_hi:[0,1]
	v_or_b32_sdwa v15, v45, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v14, v47, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v135
	v_and_b32_e32 v47, 0xffff0000, v134
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[58:59], v[68:69] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[134:135], v[58:59], v[134:135] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[58:59], v[140:141] op_sel_hi:[0,1]
	v_or_b32_sdwa v69, v45, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v0, 0xffff0000, v59
	v_and_b32_e32 v45, 0xffff0000, v58
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v59, v0, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v58, v45, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v134, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v135, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[140:141], v[52:53], v[140:141] op_sel_hi:[0,1]
	v_pk_mul_f32 v[134:135], v[52:53], v[134:135] op_sel_hi:[0,1]
	v_and_b32_e32 v0, 0xffff0000, v141
	v_and_b32_e32 v45, 0xffff0000, v140
	v_or_b32_sdwa v135, v0, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v134, v45, v134 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v68, v47, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[0:1], v[52:53], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[52:53], v[140:141] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v53
	v_and_b32_e32 v47, 0xffff0000, v52
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v53, v45, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v52, v47, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[140:141], v[46:47], v[140:141] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[46:47], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v141
	v_and_b32_e32 v47, 0xffff0000, v140
	v_or_b32_sdwa v141, v45, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v140, v47, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v2, 4, v2
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[0:1], v[46:47], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[46:47], v[142:143] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v47
	v_and_b32_e32 v45, 0xffff0000, v46
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v47, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v46, v45, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[142:143], v[44:45], v[142:143] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[44:45], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v143
	v_and_b32_e32 v45, 0xffff0000, v142
	v_lshrrev_b32_e32 v3, 4, v3
	v_or_b32_sdwa v143, v2, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v142, v45, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[44:45], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[44:45], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v45, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v44, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[144:145], v[130:131], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[144:145], v[136:137], v[8:11]
	s_nop 2
	ds_read_b128 v[8:11], v93
	v_mfma_f32_16x16x16_bf16 v[0:3], v[146:147], v[132:133], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[146:147], v[138:139], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v138, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v139, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[90:91], v[0:3]
	v_mul_f32_e64 v138, v122, v138
	v_mul_f32_e64 v139, v122, v139
	v_mfma_f32_16x16x16_bf16 v[4:7], v[8:9], v[74:75], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[20:21], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[12:13], v[4:7]
	ds_read_b128 v[8:11], v57
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[88:89], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[8:9], v[72:73], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[86:87], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[70:71], v[4:7]
	ds_read_b128 v[8:11], v55
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[124:125], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[8:9], v[76:77], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[22:23], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[14:15], v[4:7]
	ds_read_b128 v[8:11], v51
	ds_read_b128 v[12:15], v41
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[84:85], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[8:9], v[68:69], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[82:83], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[58:59], v[4:7]
	ds_read_b128 v[8:11], v49
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[126:127], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[8:9], v[134:135], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[16:17], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[10:11], v[52:53], v[4:7]
	ds_read_b128 v[8:11], v43
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[0:3], v[8:9], v[80:81], v[0:3]
	ds_read_b128 v[152:155], v65 offset:8192
	buffer_load_dwordx4 v[156:159], v63, s[24:27], 0 offen
	buffer_load_dwordx4 v[160:163], v61, s[24:27], 0 offen
	v_mfma_f32_16x16x16_bf16 v[0:3], v[10:11], v[78:79], v[0:3]
	v_add_u32_e32 v61, 0x400, v61
	v_add_u32_e32 v63, 0x400, v63
	v_mfma_f32_16x16x16_bf16 v[4:7], v[8:9], v[140:141], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v36, 4, v36
	v_mfma_f32_16x16x16_bf16 v[8:11], v[10:11], v[46:47], v[4:7]
	v_mul_f32_e64 v140, v122, v140
	v_mul_f32_e64 v141, v122, v141
	v_and_b32_e32 v47, 0xffff0000, v140
	v_or_b32_sdwa v138, v47, v138 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[12:13], v[128:129], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[14:15], v[18:19], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[12:13], v[142:143], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[14:15], v[44:45], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 6
	v_add_u32_e32 v0, 0x70000, v99
	buffer_load_dword v90, v0, s[16:19], 0 offen
	buffer_load_dword v88, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 0x71000, v99
	buffer_load_dword v86, v0, s[16:19], 0 offen
	buffer_load_dword v84, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 0x72000, v99
	buffer_load_dword v82, v0, s[16:19], 0 offen
	buffer_load_dword v80, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 0x73000, v99
	buffer_load_dword v78, v0, s[16:19], 0 offen
	buffer_load_dword v76, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 64, v101
	v_and_b32_e32 v0, -4, v0
	buffer_load_dwordx4 v[20:23], v0, s[20:23], 0 offen
	buffer_load_dwordx4 v[16:19], v0, s[20:23], 0 offen offset:16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_pk_mul_f32 v[2:3], v[106:107], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[106:107], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v125, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v28
	v_or_b32_sdwa v124, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[106:107], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[106:107], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v107, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v106, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[152:153], v[124:125], v[4:7]
	v_mul_f32_e64 v2, v104, v2
	v_mul_f32_e64 v3, v104, v3
	v_pk_mul_f32 v[0:1], v[104:105], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v127, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v29
	v_or_b32_sdwa v126, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[154:155], v[106:107], v[4:7]
	v_mul_f32_e64 v2, v104, v2
	v_mul_f32_e64 v3, v104, v3
	v_pk_mul_f32 v[0:1], v[104:105], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v29, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v28, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[102:103], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[102:103], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v105, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v30
	v_or_b32_sdwa v104, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[102:103], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[102:103], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v103, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v102, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[100:101], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[100:101], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v129, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v31
	v_or_b32_sdwa v128, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[100:101], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[100:101], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v31, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v30, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[98:99], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[98:99], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v101, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v24
	v_or_b32_sdwa v100, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[98:99], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[98:99], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v99, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v98, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v131, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v25
	v_or_b32_sdwa v130, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v25, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v24, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[94:95], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[94:95], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v133, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v26
	v_or_b32_sdwa v132, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[94:95], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[94:95], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v135, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v134, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[92:93], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[92:93], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v137, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v3, 4, v27
	v_or_b32_sdwa v136, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[2:3], v[92:93], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[92:93], v[0:1] op_sel_hi:[0,1]
	v_and_b32_e32 v2, 0xffff0000, v2
	v_or_b32_sdwa v26, v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v0, 0x70000, v113
	buffer_load_dword v74, v0, s[16:19], 0 offen
	buffer_load_dword v72, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 0x71000, v113
	buffer_load_dword v70, v0, s[16:19], 0 offen
	buffer_load_dword v68, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 0x72000, v113
	buffer_load_dword v58, v0, s[16:19], 0 offen
	buffer_load_dword v52, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 0x73000, v113
	buffer_load_dword v46, v0, s[16:19], 0 offen
	buffer_load_dword v44, v0, s[16:19], 0 offen offset:2048
	v_add_u32_e32 v0, 64, v115
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v45, -4, v0
	v_or_b32_sdwa v27, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dwordx4 v[12:15], v45, s[20:23], 0 offen
	buffer_load_dwordx4 v[0:3], v45, s[20:23], 0 offen offset:16
	v_and_b32_e32 v45, 0xffff0000, v141
	v_or_b32_sdwa v139, v45, v139 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[140:141], v[122:123], v[140:141] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[152:153], v[138:139], v[8:11]
	v_pk_mul_f32 v[122:123], v[122:123], v[142:143] op_sel_hi:[0,1]
	v_and_b32_e32 v36, 0xffff0000, v123
	v_and_b32_e32 v45, 0xffff0000, v122
	v_or_b32_sdwa v123, v36, v141 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v122, v45, v140 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v140, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v141, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[142:143], v[120:121], v[142:143] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[154:155], v[122:123], v[8:11]
	ds_read_b128 v[122:125], v93 offset:8192
	v_pk_mul_f32 v[140:141], v[120:121], v[140:141] op_sel_hi:[0,1]
	v_and_b32_e32 v36, 0xffff0000, v143
	v_and_b32_e32 v45, 0xffff0000, v142
	v_or_b32_sdwa v141, v36, v141 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v140, v45, v140 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v37
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[122:123], v[126:127], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v36, v120, v36
	v_mul_f32_e64 v37, v120, v37
	v_mfma_f32_16x16x16_bf16 v[8:11], v[122:123], v[140:141], v[8:11]
	v_pk_mul_f32 v[120:121], v[120:121], v[142:143] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v121
	v_and_b32_e32 v47, 0xffff0000, v120
	v_or_b32_sdwa v37, v45, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v36, v47, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[124:125], v[28:29], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[8:11], v[124:125], v[36:37], v[8:11]
	ds_read_b128 v[122:125], v57 offset:8192
	v_pk_mul_f32 v[142:143], v[118:119], v[142:143] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121], v[118:119], v[120:121] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[122:123], v[104:105], v[4:7]
	v_and_b32_e32 v45, 0xffff0000, v143
	v_and_b32_e32 v47, 0xffff0000, v142
	v_or_b32_sdwa v121, v45, v121 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[124:125], v[102:103], v[4:7]
	ds_read_b128 v[102:105], v55 offset:8192
	v_or_b32_sdwa v120, v47, v120 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v38, 4, v38
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v144, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v145, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[8:11], v[122:123], v[120:121], v[8:11]
	v_mul_f32_e64 v142, v118, v142
	v_mul_f32_e64 v143, v118, v143
	v_pk_mul_f32 v[118:119], v[118:119], v[144:145] op_sel_hi:[0,1]
	v_and_b32_e32 v38, 0xffff0000, v119
	v_and_b32_e32 v45, 0xffff0000, v118
	v_or_b32_sdwa v119, v38, v143 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v118, v45, v142 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[102:103], v[128:129], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v144, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v145, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v142, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v143, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[124:125], v[118:119], v[8:11]
	v_mul_f32_e64 v144, v116, v144
	v_mul_f32_e64 v145, v116, v145
	v_pk_mul_f32 v[142:143], v[116:117], v[142:143] op_sel_hi:[0,1]
	v_and_b32_e32 v38, 0xffff0000, v145
	v_and_b32_e32 v45, 0xffff0000, v144
	v_or_b32_sdwa v143, v38, v143 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v142, v45, v142 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[104:105], v[30:31], v[4:7]
	ds_read_b128 v[28:31], v51 offset:8192
	v_lshrrev_b32_e32 v45, 4, v39
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[102:103], v[142:143], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v144, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v145, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v38, v116, v38
	v_mul_f32_e64 v39, v116, v39
	v_pk_mul_f32 v[116:117], v[116:117], v[144:145] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v117
	v_and_b32_e32 v47, 0xffff0000, v116
	v_or_b32_sdwa v39, v45, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v38, v47, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v144, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v145, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v32, 4, v32
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[8:11], v[104:105], v[38:39], v[8:11]
	v_mul_f32_e64 v144, v114, v144
	v_mul_f32_e64 v145, v114, v145
	v_pk_mul_f32 v[116:117], v[114:115], v[116:117] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v145
	v_and_b32_e32 v47, 0xffff0000, v144
	v_or_b32_sdwa v117, v45, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v116, v47, v116 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v144, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v145, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[28:29], v[100:101], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v146, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v147, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v144, v114, v144
	v_mul_f32_e64 v145, v114, v145
	v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], v[116:117], v[8:11]
	v_pk_mul_f32 v[114:115], v[114:115], v[146:147] op_sel_hi:[0,1]
	v_and_b32_e32 v32, 0xffff0000, v115
	v_and_b32_e32 v45, 0xffff0000, v114
	v_or_b32_sdwa v115, v32, v145 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v114, v45, v144 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[98:99], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v146, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v147, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v144, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v145, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], v[114:115], v[8:11]
	ds_read_b128 v[28:31], v49 offset:8192
	v_pk_mul_f32 v[146:147], v[112:113], v[146:147] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[112:113], v[144:145] op_sel_hi:[0,1]
	v_and_b32_e32 v32, 0xffff0000, v147
	v_and_b32_e32 v45, 0xffff0000, v146
	v_or_b32_sdwa v145, v32, v145 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v144, v45, v144 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v33
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[28:29], v[130:131], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v146, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v147, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v32, v112, v32
	v_mul_f32_e64 v33, v112, v33
	v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], v[144:145], v[8:11]
	v_pk_mul_f32 v[112:113], v[112:113], v[146:147] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v113
	v_and_b32_e32 v47, 0xffff0000, v112
	v_or_b32_sdwa v33, v45, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v32, v47, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[24:25], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v146, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v147, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], v[32:33], v[8:11]
	ds_read_b128 v[28:31], v43 offset:8192
	v_pk_mul_f32 v[146:147], v[110:111], v[146:147] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[110:111], v[112:113] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v147
	v_and_b32_e32 v47, 0xffff0000, v146
	v_or_b32_sdwa v113, v45, v113 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v112, v47, v112 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v34, 4, v34
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v148, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v149, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[28:29], v[132:133], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v146, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v147, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v148, v110, v148
	v_mul_f32_e64 v149, v110, v149
	v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], v[112:113], v[8:11]
	v_mul_f32_e64 v146, v110, v146
	v_mul_f32_e64 v147, v110, v147
	v_and_b32_e32 v34, 0xffff0000, v149
	v_and_b32_e32 v45, 0xffff0000, v148
	v_or_b32_sdwa v147, v34, v147 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v146, v45, v146 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[134:135], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v148, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v149, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v34, 0xffff0000, v151
	v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], v[146:147], v[8:11]
	ds_read_b128 v[28:31], v41 offset:8192
	v_pk_mul_f32 v[148:149], v[108:109], v[148:149] op_sel_hi:[0,1]
	v_and_b32_e32 v45, 0xffff0000, v150
	v_or_b32_sdwa v149, v34, v149 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v148, v45, v148 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v35
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v150, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v151, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[28:29], v[136:137], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v150, v108, v150
	v_mul_f32_e64 v151, v108, v151
	v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], v[148:149], v[8:11]
	v_mul_f32_e64 v34, v108, v34
	v_mul_f32_e64 v35, v108, v35
	v_and_b32_e32 v45, 0xffff0000, v151
	v_and_b32_e32 v47, 0xffff0000, v150
	v_or_b32_sdwa v35, v45, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v34, v47, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(21)
	ds_write_b128 v67, v[156:159]
	s_waitcnt vmcnt(20)
	ds_write_b128 v95, v[160:163]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[144:147], v65
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[26:27], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], v[34:35], v[8:11]
	s_cbranch_scc1 .LBB0_2
	s_mul_i32 s2, s4, s5
	s_and_b32 s13, s13, 0xffff
	s_lshl_b32 s14, s2, 4
	s_add_u32 s2, s0, 0xffffff00
	s_addc_u32 s3, s1, -1
	s_lshr_b64 s[6:7], s[2:3], 1
	v_add_u32_e32 v24, s6, v50
	s_lshl_b32 s1, s2, 4
	s_lshl_b32 s0, s0, 4
	v_add_lshl_u32 v24, v97, v24, 2
	v_add_u32_e32 v25, s6, v48
	s_and_b32 s5, s1, 0xfffffe00
	s_and_b32 s0, s0, 0xfffffe00
	v_add_lshl_u32 v25, v97, v25, 2
	buffer_load_dwordx4 v[122:125], v24, s[24:27], 0 offen
	buffer_load_dwordx4 v[126:129], v25, s[24:27], 0 offen
	v_add_lshl_u32 v24, v56, s5, 2
	s_add_i32 s6, s0, 0xfffff200
	s_add_i32 s7, s0, 0xfffff400
	s_add_i32 s11, s0, 0xfffff600
	s_add_i32 s24, s0, 0xfffff800
	s_add_i32 s25, s0, 0xfffffa00
	s_add_i32 s26, s0, 0xfffffc00
	s_add_i32 s27, s0, 0xfffffe00
	s_lshr_b64 s[0:1], s[2:3], 3
	v_add_lshl_u32 v25, v56, s6, 2
	v_add_lshl_u32 v26, v56, s7, 2
	v_add_lshl_u32 v27, v56, s11, 2
	v_add_lshl_u32 v28, v56, s24, 2
	v_add_lshl_u32 v29, v56, s25, 2
	v_add_lshl_u32 v30, v56, s26, 2
	v_add_lshl_u32 v31, v56, s27, 2
	buffer_load_dword v106, v24, s[16:19], 0 offen
	buffer_load_dword v104, v25, s[16:19], 0 offen
	buffer_load_dword v102, v26, s[16:19], 0 offen
	buffer_load_dword v100, v27, s[16:19], 0 offen
	buffer_load_dword v98, v28, s[16:19], 0 offen
	buffer_load_dword v96, v29, s[16:19], 0 offen
	buffer_load_dword v94, v30, s[16:19], 0 offen
	buffer_load_dword v92, v31, s[16:19], 0 offen
	v_add_u32_e32 v24, s0, v111
	v_and_b32_e32 v24, -4, v24
	buffer_load_dwordx4 v[36:39], v24, s[20:23], 0 offen
	buffer_load_dwordx4 v[32:35], v24, s[20:23], 0 offen offset:16
	v_add_lshl_u32 v24, v54, s5, 2
	v_add_lshl_u32 v25, v54, s6, 2
	v_add_lshl_u32 v26, v54, s7, 2
	v_add_lshl_u32 v27, v54, s11, 2
	v_add_lshl_u32 v28, v54, s24, 2
	v_add_lshl_u32 v29, v54, s25, 2
	v_add_lshl_u32 v30, v54, s26, 2
	v_add_lshl_u32 v31, v54, s27, 2
	buffer_load_dword v66, v24, s[16:19], 0 offen
	buffer_load_dword v64, v25, s[16:19], 0 offen
	buffer_load_dword v62, v26, s[16:19], 0 offen
	buffer_load_dword v60, v27, s[16:19], 0 offen
	buffer_load_dword v56, v28, s[16:19], 0 offen
	buffer_load_dword v54, v29, s[16:19], 0 offen
	buffer_load_dword v50, v30, s[16:19], 0 offen
	buffer_load_dword v48, v31, s[16:19], 0 offen
	v_add_u32_e32 v24, s0, v109
	s_waitcnt vmcnt(31)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v110, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v111, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, -4, v24
	v_pk_mul_f32 v[110:111], v[90:91], v[110:111] op_sel_hi:[0,1]
	buffer_load_dwordx4 v[28:31], v45, s[20:23], 0 offen
	buffer_load_dwordx4 v[24:27], v45, s[20:23], 0 offen offset:16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v108, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v109, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v111
	v_and_b32_e32 v47, 0xffff0000, v110
	v_lshrrev_b32_e32 v20, 4, v20
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v110, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v111, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[108:109], v[90:91], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[90:91], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[90:91], v[112:113] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v109, v45, v109 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v20, 0xffff0000, v91
	v_and_b32_e32 v45, 0xffff0000, v90
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v90, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v91, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[112:113], v[88:89], v[112:113] op_sel_hi:[0,1]
	v_or_b32_sdwa v111, v20, v111 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v110, v45, v110 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[90:91], v[88:89], v[90:91] op_sel_hi:[0,1]
	v_and_b32_e32 v20, 0xffff0000, v113
	v_and_b32_e32 v45, 0xffff0000, v112
	v_or_b32_sdwa v91, v20, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v90, v45, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v21
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v108, v47, v108 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[20:21], v[88:89], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[88:89], v[112:113] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v89
	v_and_b32_e32 v47, 0xffff0000, v88
	v_pk_mul_f32 v[112:113], v[86:87], v[112:113] op_sel_hi:[0,1]
	v_or_b32_sdwa v21, v45, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v47, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v88, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v89, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v113
	v_and_b32_e32 v47, 0xffff0000, v112
	v_lshrrev_b32_e32 v22, 4, v22
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v112, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v113, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[88:89], v[86:87], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[86:87], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[86:87], v[114:115] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v89, v45, v89 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v22, 0xffff0000, v87
	v_and_b32_e32 v45, 0xffff0000, v86
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v86, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v87, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[114:115], v[84:85], v[114:115] op_sel_hi:[0,1]
	v_or_b32_sdwa v113, v22, v113 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v112, v45, v112 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[86:87], v[84:85], v[86:87] op_sel_hi:[0,1]
	v_and_b32_e32 v22, 0xffff0000, v115
	v_and_b32_e32 v45, 0xffff0000, v114
	v_or_b32_sdwa v87, v22, v87 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v86, v45, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v23
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v88, v47, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[22:23], v[84:85], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[84:85], v[114:115] op_sel_hi:[0,1]
	s_waitcnt vmcnt(32)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v85
	v_and_b32_e32 v47, 0xffff0000, v84
	v_pk_mul_f32 v[114:115], v[82:83], v[114:115] op_sel_hi:[0,1]
	v_or_b32_sdwa v23, v45, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v22, v47, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v84, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v85, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v115
	v_and_b32_e32 v47, 0xffff0000, v114
	v_lshrrev_b32_e32 v16, 4, v16
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v114, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v115, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[84:85], v[82:83], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115], v[82:83], v[114:115] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[82:83], v[116:117] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v85, v45, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v16, 0xffff0000, v83
	v_and_b32_e32 v45, 0xffff0000, v82
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v82, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v83, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[116:117], v[80:81], v[116:117] op_sel_hi:[0,1]
	v_or_b32_sdwa v115, v16, v115 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v114, v45, v114 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[82:83], v[80:81], v[82:83] op_sel_hi:[0,1]
	v_and_b32_e32 v16, 0xffff0000, v117
	v_and_b32_e32 v45, 0xffff0000, v116
	v_or_b32_sdwa v83, v16, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v82, v45, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v17
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v84, v47, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[16:17], v[80:81], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[80:81], v[116:117] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v81
	v_and_b32_e32 v47, 0xffff0000, v80
	v_pk_mul_f32 v[116:117], v[78:79], v[116:117] op_sel_hi:[0,1]
	v_or_b32_sdwa v17, v45, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v47, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v80, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v81, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v117
	v_and_b32_e32 v47, 0xffff0000, v116
	v_lshrrev_b32_e32 v18, 4, v18
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[80:81], v[78:79], v[80:81] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[116:117], v[78:79], v[116:117] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[78:79], v[118:119] op_sel_hi:[0,1]
	v_or_b32_sdwa v81, v45, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v18, 0xffff0000, v79
	v_and_b32_e32 v45, 0xffff0000, v78
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v79, v18, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v78, v45, v116 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v116, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v117, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[118:119], v[76:77], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[116:117], v[76:77], v[116:117] op_sel_hi:[0,1]
	v_and_b32_e32 v18, 0xffff0000, v119
	v_and_b32_e32 v45, 0xffff0000, v118
	v_or_b32_sdwa v117, v18, v117 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v116, v45, v116 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v19
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v80, v47, v80 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[18:19], v[76:77], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[76:77], v[118:119] op_sel_hi:[0,1]
	s_waitcnt vmcnt(23)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v77
	v_and_b32_e32 v47, 0xffff0000, v76
	v_pk_mul_f32 v[118:119], v[74:75], v[118:119] op_sel_hi:[0,1]
	v_or_b32_sdwa v19, v45, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v18, v47, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v76, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v77, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v119
	v_and_b32_e32 v47, 0xffff0000, v118
	v_lshrrev_b32_e32 v12, 4, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[76:77], v[74:75], v[76:77] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[118:119], v[74:75], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[74:75], v[120:121] op_sel_hi:[0,1]
	v_or_b32_sdwa v77, v45, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v12, 0xffff0000, v75
	v_and_b32_e32 v45, 0xffff0000, v74
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v75, v12, v119 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v74, v45, v118 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v118, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v119, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[120:121], v[72:73], v[120:121] op_sel_hi:[0,1]
	v_pk_mul_f32 v[118:119], v[72:73], v[118:119] op_sel_hi:[0,1]
	v_and_b32_e32 v12, 0xffff0000, v121
	v_and_b32_e32 v45, 0xffff0000, v120
	v_or_b32_sdwa v119, v12, v119 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v118, v45, v118 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v13
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v76, v47, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[72:73], v[120:121] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v73
	v_and_b32_e32 v47, 0xffff0000, v72
	v_pk_mul_f32 v[120:121], v[70:71], v[120:121] op_sel_hi:[0,1]
	v_or_b32_sdwa v13, v45, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v47, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v121
	v_and_b32_e32 v47, 0xffff0000, v120
	v_lshrrev_b32_e32 v14, 4, v14
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[70:71], v[72:73] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v130, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v131, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[120:121], v[70:71], v[120:121] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[70:71], v[130:131] op_sel_hi:[0,1]
	v_or_b32_sdwa v73, v45, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v14, 0xffff0000, v71
	v_and_b32_e32 v45, 0xffff0000, v70
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v130, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v131, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v71, v14, v121 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v45, v120 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v120, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v121, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[130:131], v[68:69], v[130:131] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121], v[68:69], v[120:121] op_sel_hi:[0,1]
	v_and_b32_e32 v14, 0xffff0000, v131
	v_and_b32_e32 v45, 0xffff0000, v130
	v_or_b32_sdwa v121, v14, v121 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v120, v45, v120 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v15
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v130, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v131, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[144:145], v[108:109], v[4:7]
	v_mul_f32_e64 v14, v68, v14
	v_mul_f32_e64 v15, v68, v15
	v_pk_mul_f32 v[68:69], v[68:69], v[130:131] op_sel_hi:[0,1]
	ds_read_b128 v[130:133], v93
	v_mfma_f32_16x16x16_bf16 v[8:11], v[144:145], v[76:77], v[8:11]
	s_waitcnt vmcnt(22)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v108, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v109, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v72, v47, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[146:147], v[110:111], v[4:7]
	v_mul_f32_e64 v76, v58, v108
	v_mul_f32_e64 v77, v58, v109
	ds_read_b128 v[108:111], v57
	v_and_b32_e32 v45, 0xffff0000, v69
	v_mfma_f32_16x16x16_bf16 v[8:11], v[146:147], v[74:75], v[8:11]
	v_and_b32_e32 v47, 0xffff0000, v68
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[130:131], v[90:91], v[4:7]
	v_lshrrev_b32_e32 v0, 4, v0
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v74, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v75, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[130:131], v[118:119], v[8:11]
	v_or_b32_sdwa v15, v45, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v14, v47, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[68:69], v[58:59], v[68:69] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[132:133], v[20:21], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v45, 0xffff0000, v77
	v_mfma_f32_16x16x16_bf16 v[8:11], v[132:133], v[12:13], v[8:11]
	v_and_b32_e32 v47, 0xffff0000, v76
	v_pk_mul_f32 v[20:21], v[58:59], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[58:59], v[74:75] op_sel_hi:[0,1]
	ds_read_b128 v[74:77], v55
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[108:109], v[88:89], v[4:7]
	v_or_b32_sdwa v69, v45, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v68, v47, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v0, 0xffff0000, v59
	v_mfma_f32_16x16x16_bf16 v[8:11], v[108:109], v[72:73], v[8:11]
	v_and_b32_e32 v12, 0xffff0000, v58
	v_or_b32_sdwa v13, v0, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v12, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[110:111], v[112:113], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[110:111], v[70:71], v[8:11]
	ds_read_b128 v[70:73], v51
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[52:53], v[58:59] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[74:75], v[86:87], v[4:7]
	v_mul_f32_e64 v20, v52, v20
	v_mul_f32_e64 v21, v52, v21
	v_and_b32_e32 v0, 0xffff0000, v59
	v_and_b32_e32 v45, 0xffff0000, v58
	v_mfma_f32_16x16x16_bf16 v[8:11], v[74:75], v[120:121], v[8:11]
	v_or_b32_sdwa v21, v0, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v20, v45, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v45, 4, v1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[76:77], v[22:23], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v0, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[76:77], v[14:15], v[8:11]
	ds_read_b128 v[74:77], v49
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v1, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[22:23], v[52:53], v[22:23] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[70:71], v[84:85], v[4:7]
	v_mul_f32_e64 v0, v52, v0
	v_mul_f32_e64 v1, v52, v1
	v_and_b32_e32 v23, 0xffff0000, v23
	v_and_b32_e32 v14, 0xffff0000, v22
	v_mfma_f32_16x16x16_bf16 v[8:11], v[70:71], v[68:69], v[8:11]
	ds_read_b128 v[68:71], v43
	v_or_b32_sdwa v1, v23, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[72:73], v[114:115], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v0, v14, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[72:73], v[12:13], v[8:11]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v22, v46, v22
	v_mul_f32_e64 v23, v46, v23
	v_pk_mul_f32 v[14:15], v[46:47], v[14:15] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[74:75], v[82:83], v[4:7]
	v_and_b32_e32 v23, 0xffff0000, v23
	v_and_b32_e32 v22, 0xffff0000, v22
	v_or_b32_sdwa v13, v23, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[8:11], v[74:75], v[20:21], v[8:11]
	v_or_b32_sdwa v12, v22, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v2, 4, v2
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[76:77], v[16:17], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[8:11], v[76:77], v[0:1], v[8:11]
	v_mul_f32_e64 v16, v46, v22
	v_mul_f32_e64 v17, v46, v23
	ds_read_b128 v[20:23], v41
	v_pk_mul_f32 v[14:15], v[46:47], v[14:15] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[4:7], v[68:69], v[80:81], v[4:7]
	v_and_b32_e32 v2, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v1, v2, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[8:11], v[68:69], v[12:13], v[8:11]
	v_or_b32_sdwa v0, v16, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[70:71], v[78:79], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v16, v44, v16
	v_mul_f32_e64 v17, v44, v17
	v_mfma_f32_16x16x16_bf16 v[8:11], v[70:71], v[0:1], v[8:11]
	v_mul_f32_e64 v14, v44, v14
	v_mul_f32_e64 v15, v44, v15
	v_and_b32_e32 v2, 0xffff0000, v17
	v_and_b32_e32 v12, 0xffff0000, v16
	v_or_b32_sdwa v13, v2, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v16, 4, v3
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[20:21], v[116:117], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v4, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v5, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[22:23], v[18:19], v[0:3]
	v_mul_f32_e64 v16, v44, v4
	v_mul_f32_e64 v17, v44, v5
	v_pk_mul_f32 v[14:15], v[44:45], v[14:15] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_mfma_f32_16x16x16_bf16 v[4:7], v[20:21], v[12:13], v[8:11]
	s_waitcnt vmcnt(11)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v13, 4, v36
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v12, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_lshrrev_b32_e32 v19, 4, v39
	v_and_b32_e32 v8, 0xffff0000, v16
	v_or_b32_sdwa v9, v17, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v8, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[10:11], v[106:107], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v11, 0xffff0000, v11
	v_mfma_f32_16x16x16_bf16 v[4:7], v[22:23], v[8:9], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v10, 0xffff0000, v10
	v_pk_mul_f32 v[8:9], v[106:107], v[8:9] op_sel_hi:[0,1]
	v_or_b32_sdwa v9, v11, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v8, v10, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v13, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v17, 4, v37
	v_pk_mul_f32 v[12:13], v[106:107], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[106:107], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v13, v13, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[14:15], v[104:105], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[104:105], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v14, 0xffff0000, v14
	v_or_b32_sdwa v15, v15, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v14, v14, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
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
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	s_waitcnt vmcnt(10)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_pk_mul_f32 v[10:11], v[104:105], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[104:105], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v23, v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v22, v16, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[36:37], v[98:99], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[102:103], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[102:103], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v21, v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v17, 4, v38
	v_or_b32_sdwa v20, v16, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
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
	v_and_b32_e32 v37, 0xffff0000, v37
	v_pk_mul_f32 v[16:17], v[102:103], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[102:103], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v45, v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v44, v16, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v36, 0xffff0000, v36
	v_pk_mul_f32 v[16:17], v[100:101], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[100:101], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v11, v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v16, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[16:17], v[100:101], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[100:101], v[18:19] op_sel_hi:[0,1]
	v_and_b32_e32 v19, 0xffff0000, v19
	v_and_b32_e32 v18, 0xffff0000, v18
	v_or_b32_sdwa v17, v19, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v16, v18, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_lshrrev_b32_e32 v32, 4, v32
	v_pk_mul_f32 v[18:19], v[98:99], v[18:19] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v19, v37, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v18, v36, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v36, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v37, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[38:39], v[98:99], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[98:99], v[36:37] op_sel_hi:[0,1]
	v_and_b32_e32 v38, 0xffff0000, v38
	v_and_b32_e32 v32, 0xffff0000, v39
	v_or_b32_sdwa v36, v38, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v38, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v39, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[46:47], v[96:97], v[46:47] op_sel_hi:[0,1]
	v_or_b32_sdwa v37, v32, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_pk_mul_f32 v[38:39], v[96:97], v[38:39] op_sel_hi:[0,1]
	v_and_b32_e32 v32, 0xffff0000, v47
	v_and_b32_e32 v46, 0xffff0000, v46
	v_lshrrev_b32_e32 v47, 4, v33
	v_or_b32_sdwa v39, v32, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v38, v46, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v32, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v33, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[52:53], v[94:95], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[96:97], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[96:97], v[32:33] op_sel_hi:[0,1]
	v_and_b32_e32 v47, 0xffff0000, v47
	v_and_b32_e32 v46, 0xffff0000, v46
	v_or_b32_sdwa v33, v47, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v32, v46, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v46, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v47, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v53, 0xffff0000, v53
	v_pk_mul_f32 v[46:47], v[94:95], v[46:47] op_sel_hi:[0,1]
	v_and_b32_e32 v52, 0xffff0000, v52
	v_lshrrev_b32_e32 v34, 4, v34
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v47, v53, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v46, v52, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v52, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v53, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[58:59], v[94:95], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[94:95], v[52:53] op_sel_hi:[0,1]
	v_and_b32_e32 v58, 0xffff0000, v58
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_and_b32_e32 v34, 0xffff0000, v59
	v_or_b32_sdwa v52, v58, v52 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v58, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v59, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[92:93], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[92:93], v[58:59] op_sel_hi:[0,1]
	v_and_b32_e32 v61, 0xffff0000, v68
	v_or_b32_sdwa v53, v34, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v34, 0xffff0000, v69
	v_or_b32_sdwa v58, v61, v58 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v61, 4, v35
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v59, v34, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v34, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v35, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[68:69], v[92:93], v[68:69] op_sel_hi:[0,1]
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[34:35], v[92:93], v[34:35] op_sel_hi:[0,1]
	v_and_b32_e32 v61, 0xffff0000, v69
	v_and_b32_e32 v63, 0xffff0000, v68
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71] op_sel_hi:[0,1]
	v_or_b32_sdwa v35, v61, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v34, v63, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v68, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v69, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_and_b32_e32 v61, 0xffff0000, v71
	v_and_b32_e32 v63, 0xffff0000, v70
	v_lshrrev_b32_e32 v28, 4, v28
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	ds_write_b128 v67, v[122:125] offset:8192
	ds_write_b128 v95, v[126:129] offset:8192
	v_pk_mul_f32 v[68:69], v[66:67], v[68:69] op_sel_hi:[0,1]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[66:67], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v69, v61, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v28, 0xffff0000, v67
	v_and_b32_e32 v61, 0xffff0000, v66
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v67, v28, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v66, v61, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v70, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v71, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[64:65], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[64:65], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v28, 0xffff0000, v73
	v_and_b32_e32 v61, 0xffff0000, v72
	v_or_b32_sdwa v71, v28, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v70, v61, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v61, 4, v29
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[74:77], v65 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_pk_mul_f32 v[28:29], v[64:65], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[64:65], v[72:73] op_sel_hi:[0,1]
	v_or_b32_sdwa v68, v63, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_b32_e32 v61, 0xffff0000, v65
	v_and_b32_e32 v63, 0xffff0000, v64
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v72, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v73, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v65, v61, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v64, v63, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[72:73], v[62:63], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[62:63], v[28:29] op_sel_hi:[0,1]
	v_and_b32_e32 v61, 0xffff0000, v73
	v_and_b32_e32 v63, 0xffff0000, v72
	v_or_b32_sdwa v73, v61, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v72, v63, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v30, 4, v30
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[74:75], v[8:9], v[0:3]
	v_mul_f32_e64 v28, v62, v28
	v_mul_f32_e64 v29, v62, v29
	v_pk_mul_f32 v[62:63], v[62:63], v[78:79] op_sel_hi:[0,1]
	v_and_b32_e32 v30, 0xffff0000, v63
	v_and_b32_e32 v61, 0xffff0000, v62
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v63, v30, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v62, v61, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v28, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v29, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[78:79], v[60:61], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[60:61], v[28:29] op_sel_hi:[0,1]
	v_and_b32_e32 v30, 0xffff0000, v79
	v_and_b32_e32 v61, 0xffff0000, v78
	v_or_b32_sdwa v29, v30, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v28, v61, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v61, 4, v31
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v30, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v78, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v31, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v79, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[74:75], v[68:69], v[4:7]
	v_mul_f32_e64 v30, v60, v30
	v_mul_f32_e64 v31, v60, v31
	v_pk_mul_f32 v[60:61], v[60:61], v[78:79] op_sel_hi:[0,1]
	ds_read_b128 v[78:81], v93 offset:8192
	v_mfma_f32_16x16x16_bf16 v[0:3], v[76:77], v[12:13], v[0:3]
	v_and_b32_e32 v61, 0xffff0000, v61
	v_and_b32_e32 v60, 0xffff0000, v60
	v_or_b32_sdwa v31, v61, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[76:77], v[66:67], v[4:7]
	ds_read_b128 v[66:69], v57 offset:8192
	ds_read_b128 v[74:77], v55 offset:8192
	v_or_b32_sdwa v30, v60, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[78:79], v[14:15], v[0:3]
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v60, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v61, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_lshrrev_b32_e32 v24, 4, v24
	v_pk_mul_f32 v[12:13], v[56:57], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[56:57], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v12, 0xffff0000, v12
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[80:81], v[22:23], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_or_b32_sdwa v13, v13, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v12, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_pk_mul_f32 v[14:15], v[56:57], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[56:57], v[8:9] op_sel_hi:[0,1]
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v14, 0xffff0000, v14
	v_or_b32_sdwa v15, v15, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v14, v14, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v8, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v9, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[66:67], v[20:21], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v20, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v21, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mul_f32_e64 v22, v54, v8
	v_mul_f32_e64 v23, v54, v9
	v_pk_mul_f32 v[8:9], v[54:55], v[20:21] op_sel_hi:[0,1]
	s_lshl_b32 s0, s10, 2
	v_and_b32_e32 v9, 0xffff0000, v9
	v_and_b32_e32 v20, 0xffff0000, v8
	v_lshl_or_b32 v8, v42, 4, s0
	s_mov_b32 s10, s18
	s_mov_b32 s11, s19
	v_or_b32_sdwa v21, v9, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	buffer_load_dword v9, v8, s[8:11], 0 offen
	v_mfma_f32_16x16x16_bf16 v[4:7], v[78:79], v[70:71], v[4:7]
	v_lshrrev_b32_e32 v23, 4, v25
	v_or_b32_sdwa v20, v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v22, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[80:81], v[64:65], v[4:7]
	s_mov_b32 s15, 0x27000
	v_mfma_f32_16x16x16_bf16 v[4:7], v[66:67], v[72:73], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[68:69], v[44:45], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[68:69], v[62:63], v[4:7]
	ds_read_b128 v[60:63], v51 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[74:75], v[10:11], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v23, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[74:75], v[28:29], v[4:7]
	v_mul_f32_e64 v10, v54, v10
	v_mul_f32_e64 v11, v54, v11
	v_mfma_f32_16x16x16_bf16 v[0:3], v[76:77], v[16:17], v[0:3]
	v_mul_f32_e64 v16, v54, v22
	v_mul_f32_e64 v17, v54, v23
	ds_read_b128 v[22:25], v49 offset:8192
	v_and_b32_e32 v17, 0xffff0000, v17
	v_mfma_f32_16x16x16_bf16 v[4:7], v[76:77], v[30:31], v[4:7]
	ds_read_b128 v[28:31], v43 offset:8192
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v11, v17, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[60:61], v[18:19], v[0:3]
	v_or_b32_sdwa v10, v16, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v18, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[60:61], v[12:13], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v19, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[62:63], v[36:37], v[0:3]
	v_mul_f32_e64 v12, v50, v18
	v_mul_f32_e64 v13, v50, v19
	v_pk_mul_f32 v[16:17], v[50:51], v[16:17] op_sel_hi:[0,1]
	v_and_b32_e32 v13, 0xffff0000, v13
	v_mfma_f32_16x16x16_bf16 v[4:7], v[62:63], v[14:15], v[4:7]
	v_and_b32_e32 v12, 0xffff0000, v12
	v_or_b32_sdwa v13, v13, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v12, v12, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[22:23], v[38:39], v[0:3]
	v_lshrrev_b32_e32 v17, 4, v26
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[22:23], v[20:21], v[4:7]
	ds_read_b128 v[18:21], v41 offset:8192
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[0:3], v[24:25], v[32:33], v[0:3]
	v_mul_f32_e64 v16, v50, v16
	v_mul_f32_e64 v17, v50, v17
	v_pk_mul_f32 v[14:15], v[50:51], v[14:15] op_sel_hi:[0,1]
	v_and_b32_e32 v17, 0xffff0000, v17
	v_mfma_f32_16x16x16_bf16 v[4:7], v[24:25], v[10:11], v[4:7]
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v11, v17, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v16, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[28:29], v[46:47], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v16, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v17, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[28:29], v[12:13], v[4:7]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	v_mul_f32_e64 v16, v48, v16
	v_mul_f32_e64 v17, v48, v17
	v_pk_mul_f32 v[14:15], v[48:49], v[14:15] op_sel_hi:[0,1]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[30:31], v[52:53], v[0:3]
	v_and_b32_e32 v12, 0xffff0000, v17
	v_and_b32_e32 v16, 0xffff0000, v16
	v_or_b32_sdwa v13, v12, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[4:7], v[30:31], v[10:11], v[4:7]
	v_or_b32_sdwa v12, v16, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_lshrrev_b32_e32 v15, 4, v27
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v10, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[0:3], v[18:19], v[58:59], v[0:3]
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v14, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v11, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2
	;;#ASMEND
	;;#ASMSTART
	v_cvt_off_f32_i4_sdwa v15, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3
	;;#ASMEND
	v_mfma_f32_16x16x16_bf16 v[4:7], v[18:19], v[12:13], v[4:7]
	v_mul_f32_e64 v14, v48, v14
	v_mul_f32_e64 v15, v48, v15
	v_pk_mul_f32 v[10:11], v[48:49], v[10:11] op_sel_hi:[0,1]
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v12, 0xffff0000, v14
	v_or_b32_sdwa v11, v15, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_or_b32_sdwa v10, v12, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mfma_f32_16x16x16_bf16 v[0:3], v[20:21], v[34:35], v[0:3]
	s_nop 0
	v_mfma_f32_16x16x16_bf16 v[4:7], v[20:21], v[10:11], v[4:7]
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
	v_lshl_add_u32 v12, v9, 11, v40
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
	v_lshl_add_u32 v10, v4, 11, v40
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
	v_lshl_add_u32 v6, v0, 11, v40
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
	v_lshl_add_u32 v5, v0, 11, v40
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
		.amdhsa_next_free_vgpr 164
		.amdhsa_next_free_sgpr 44
		.amdhsa_accum_offset 164
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

	.set moe_gemm1_0.num_vgpr, 164
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
    .vgpr_count:     164
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
