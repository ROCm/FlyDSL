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
