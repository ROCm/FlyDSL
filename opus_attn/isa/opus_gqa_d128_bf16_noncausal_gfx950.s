	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs,"axG",@progbits,_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs,comdat
	.protected	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs ; -- Begin function _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs
	.globl	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs
	.p2align	8
	.type	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs,@function
_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs: ; @_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs
; %bb.0:
	s_load_dwordx8 s[16:23], s[0:1], 0x24
	s_load_dwordx8 s[8:15], s[0:1], 0x0
	s_load_dwordx2 s[24:25], s[0:1], 0x44
	v_readfirstlane_b32 s28, v0
	s_lshr_b32 s29, s28, 6
	v_and_b32_e32 v113, 7, v0
	s_waitcnt lgkmcnt(0)
	s_abs_i32 s5, s18
	v_cvt_f32_u32_e32 v1, s5
	s_sub_i32 s6, 0, s5
	s_abs_i32 s1, s17
	s_xor_b32 s0, s17, s18
	v_rcp_iflag_f32_e32 v1, v1
	s_ashr_i32 s0, s0, 31
	s_mul_i32 s3, s3, s21
	v_bfe_u32 v208, v0, 5, 1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_movk_i32 s30, 0x410
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
	s_sub_i32 s2, s2, s0
	s_mul_i32 s0, s20, s4
	s_lshl_b32 s3, s3, 8
	s_add_i32 s0, s3, s0
	s_mul_i32 s1, s1, s22
	s_add_i32 s0, s0, s1
	s_mul_i32 s1, s23, s4
	s_mul_i32 s2, s2, s25
	s_add_i32 s4, s2, s1
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
	v_add_u32_e32 v1, s29, v1
	s_and_b32 s9, s6, 0xffff
	v_mul_lo_u32 v1, v1, s24
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	s_add_u32 s4, s12, s4
	s_mul_i32 s26, s29, 0x410
	v_lshlrev_b32_e32 v1, 1, v1
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_addc_u32 s5, s13, s5
	v_lshl_add_u32 v210, v113, 4, v1
	s_mov_b32 m0, s26
	s_add_i32 s27, s26, 0x2080
	buffer_load_dwordx4 v210, s[8:11], 0 offen lds
	v_add_u32_e32 v211, 0x80, v210
	s_mov_b32 m0, s27
	s_and_b32 s5, s5, 0xffff
	buffer_load_dwordx4 v211, s[8:11], 0 offen lds
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	v_and_b32_e32 v1, 31, v0
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_mul_i32 s12, s21, s29
	v_mul_lo_u32 v209, s21, v1
	s_lshl_b32 s12, s12, 5
	v_lshl_add_u32 v1, v208, 3, v209
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
	s_add_i32 s25, s26, 0x8500
	s_lshl_b32 s17, s24, 7
	s_mov_b32 m0, s25
	s_add_i32 s21, s25, 0x2080
	s_mul_i32 s13, s29, 0x440
	buffer_load_dwordx4 v210, s[8:11], s17 offen lds
	s_mov_b32 m0, s21
	s_add_i32 s20, s13, 0x4100
	buffer_load_dwordx4 v211, s[8:11], s17 offen lds
	s_mov_b32 m0, s20
	s_add_i32 s18, s20, 0x2200
	buffer_load_dwordx4 v210, s[4:7], 0 offen lds
	s_mov_b32 m0, s18
	v_lshlrev_b32_e32 v1, 4, v0
	buffer_load_dwordx4 v211, s[4:7], 0 offen lds
	v_lshlrev_b32_e32 v204, 4, v208
	v_and_b32_e32 v205, 0x180, v1
	v_or_b32_e32 v213, v204, v205
	v_mad_u32_u24 v1, v113, s30, v213
	ds_read_b128 v[10:13], v1
	ds_read_b128 v[2:5], v1 offset:32
	ds_read_b128 v[104:107], v1 offset:64
	ds_read_b128 v[92:95], v1 offset:96
	ds_read_b128 v[80:83], v1 offset:8320
	ds_read_b128 v[68:71], v1 offset:8352
	ds_read_b128 v[56:59], v1 offset:8384
	ds_read_b128 v[26:29], v1 offset:8416
	ds_read_b128 v[6:9], v1 offset:512
	ds_read_b128 v[108:111], v1 offset:544
	ds_read_b128 v[100:103], v1 offset:576
	ds_read_b128 v[88:91], v1 offset:608
	ds_read_b128 v[76:79], v1 offset:8832
	ds_read_b128 v[64:67], v1 offset:8864
	ds_read_b128 v[52:55], v1 offset:8896
	ds_read_b128 v[22:25], v1 offset:8928
	; sched_barrier mask(0x00000000)
	s_cmpk_lt_u32 s28, 0x100
	s_cselect_b64 s[0:1], -1, 0
	s_and_b64 s[0:1], exec, s[0:1]
	s_mov_b64 vcc, s[0:1]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	; sched_barrier mask(0x00000000)
	s_barrier
.LBB0_2:
	v_cvt_f32_i32_e32 v1, s19
	v_and_b32_e32 v115, 0xffff0000, v33
	v_lshlrev_b32_e32 v114, 16, v33
	v_and_b32_e32 v117, 0xffff0000, v32
	v_rsq_f32_e32 v1, v1
	v_lshlrev_b32_e32 v116, 16, v32
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
	v_and_b32_e32 v119, 0xffff0000, v31
	v_lshlrev_b32_e32 v118, 16, v31
	v_mfma_f32_32x32x16_bf16 v[32:47], v[10:13], v[144:147], 0
	v_and_b32_e32 v31, 0xffff0000, v30
	v_lshlrev_b32_e32 v30, 16, v30
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_mul_f32_e64 v10, v112, v118
	v_mul_f32_e64 v11, v112, v119
	v_pk_mul_f32 v[12:13], v[112:113], v[116:117] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[112:113], v[114:115] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v151, v14, v15
	v_cvt_pk_bf16_f32 v150, v12, v13
	v_cvt_pk_bf16_f32 v149, v10, v11
	v_cvt_pk_bf16_f32 v148, v30, v31
	v_and_b32_e32 v31, 0xffff0000, v99
	v_lshlrev_b32_e32 v30, 16, v99
	v_mfma_f32_32x32x16_bf16 v[32:47], v[2:5], v[148:151], v[32:47]
	v_and_b32_e32 v99, 0xffff0000, v98
	v_lshlrev_b32_e32 v98, 16, v98
	v_and_b32_e32 v115, 0xffff0000, v97
	v_lshlrev_b32_e32 v114, 16, v97
	v_and_b32_e32 v117, 0xffff0000, v96
	v_lshlrev_b32_e32 v116, 16, v96
	v_pk_mul_f32 v[98:99], v[112:113], v[98:99] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[6:9], v[144:147], 0
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_cvt_pk_bf16_f32 v154, v98, v99
	v_mul_f32_e64 v98, v112, v114
	v_mul_f32_e64 v99, v112, v115
	v_cvt_pk_bf16_f32 v155, v30, v31
	v_cvt_pk_bf16_f32 v153, v98, v99
	v_and_b32_e32 v99, 0xffff0000, v87
	v_lshlrev_b32_e32 v98, 16, v87
	v_mfma_f32_32x32x16_bf16 v[2:17], v[108:111], v[148:151], v[2:17]
	v_mul_f32_e64 v110, v112, v116
	v_mul_f32_e64 v111, v112, v117
	v_cvt_pk_bf16_f32 v152, v110, v111
	v_and_b32_e32 v87, 0xffff0000, v86
	v_lshlrev_b32_e32 v86, 16, v86
	v_pk_mul_f32 v[86:87], v[112:113], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[112:113], v[98:99] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v159, v98, v99
	v_mfma_f32_32x32x16_bf16 v[2:17], v[100:103], v[152:155], v[2:17]
	v_cvt_pk_bf16_f32 v158, v86, v87
	v_and_b32_e32 v111, 0xffff0000, v75
	v_lshlrev_b32_e32 v110, 16, v75
	v_and_b32_e32 v75, 0xffff0000, v74
	v_lshlrev_b32_e32 v74, 16, v74
	v_pk_mul_f32 v[74:75], v[112:113], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[112:113], v[110:111] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[104:107], v[152:155], v[32:47]
	v_and_b32_e32 v105, 0xffff0000, v85
	v_lshlrev_b32_e32 v104, 16, v85
	v_and_b32_e32 v85, 0xffff0000, v84
	v_lshlrev_b32_e32 v84, 16, v84
	v_mul_f32_e64 v84, v112, v84
	v_mul_f32_e64 v85, v112, v85
	v_pk_mul_f32 v[100:101], v[112:113], v[104:105] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v157, v100, v101
	v_cvt_pk_bf16_f32 v156, v84, v85
	v_and_b32_e32 v85, 0xffff0000, v73
	v_lshlrev_b32_e32 v84, 16, v73
	v_mfma_f32_32x32x16_bf16 v[2:17], v[88:91], v[156:159], v[2:17]
	v_and_b32_e32 v73, 0xffff0000, v72
	v_lshlrev_b32_e32 v72, 16, v72
	v_mul_f32_e64 v72, v112, v72
	v_mul_f32_e64 v73, v112, v73
	v_mul_f32_e64 v84, v112, v84
	v_mul_f32_e64 v85, v112, v85
	v_cvt_pk_bf16_f32 v163, v86, v87
	v_cvt_pk_bf16_f32 v162, v74, v75
	v_cvt_pk_bf16_f32 v161, v84, v85
	v_cvt_pk_bf16_f32 v160, v72, v73
	v_mfma_f32_32x32x16_bf16 v[32:47], v[92:95], v[156:159], v[32:47]
	v_and_b32_e32 v107, 0xffff0000, v63
	v_lshlrev_b32_e32 v106, 16, v63
	v_and_b32_e32 v63, 0xffff0000, v62
	v_lshlrev_b32_e32 v62, 16, v62
	v_and_b32_e32 v73, 0xffff0000, v61
	v_lshlrev_b32_e32 v72, 16, v61
	v_and_b32_e32 v61, 0xffff0000, v60
	v_mfma_f32_32x32x16_bf16 v[2:17], v[76:79], v[160:163], v[2:17]
	v_lshlrev_b32_e32 v60, 16, v60
	v_mul_f32_e64 v60, v112, v60
	v_mul_f32_e64 v61, v112, v61
	v_mul_f32_e64 v72, v112, v72
	v_mul_f32_e64 v73, v112, v73
	v_pk_mul_f32 v[62:63], v[112:113], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[112:113], v[106:107] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v167, v74, v75
	v_cvt_pk_bf16_f32 v166, v62, v63
	v_cvt_pk_bf16_f32 v165, v72, v73
	v_cvt_pk_bf16_f32 v164, v60, v61
	v_mfma_f32_32x32x16_bf16 v[32:47], v[80:83], v[160:163], v[32:47]
	v_and_b32_e32 v31, 0xffff0000, v51
	v_lshlrev_b32_e32 v30, 16, v51
	v_and_b32_e32 v51, 0xffff0000, v50
	v_lshlrev_b32_e32 v50, 16, v50
	v_and_b32_e32 v61, 0xffff0000, v49
	v_lshlrev_b32_e32 v60, 16, v49
	v_and_b32_e32 v49, 0xffff0000, v48
	v_mfma_f32_32x32x16_bf16 v[2:17], v[64:67], v[164:167], v[2:17]
	v_lshlrev_b32_e32 v48, 16, v48
	v_mul_f32_e64 v48, v112, v48
	v_mul_f32_e64 v49, v112, v49
	v_mul_f32_e64 v60, v112, v60
	v_mul_f32_e64 v61, v112, v61
	v_pk_mul_f32 v[50:51], v[112:113], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[112:113], v[30:31] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v171, v30, v31
	v_cvt_pk_bf16_f32 v170, v50, v51
	v_cvt_pk_bf16_f32 v169, v60, v61
	v_cvt_pk_bf16_f32 v168, v48, v49
	v_mfma_f32_32x32x16_bf16 v[32:47], v[68:71], v[164:167], v[32:47]
	v_and_b32_e32 v109, 0xffff0000, v21
	v_lshlrev_b32_e32 v108, 16, v21
	v_and_b32_e32 v21, 0xffff0000, v20
	v_lshlrev_b32_e32 v20, 16, v20
	v_and_b32_e32 v31, 0xffff0000, v19
	v_lshlrev_b32_e32 v30, 16, v19
	v_and_b32_e32 v19, 0xffff0000, v18
	v_mfma_f32_32x32x16_bf16 v[2:17], v[52:55], v[168:171], v[2:17]
	v_lshlrev_b32_e32 v18, 16, v18
	v_mul_f32_e64 v18, v112, v18
	v_mul_f32_e64 v19, v112, v19
	v_mul_f32_e64 v30, v112, v30
	v_mul_f32_e64 v31, v112, v31
	v_pk_mul_f32 v[20:21], v[112:113], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[112:113], v[108:109] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v175, v48, v49
	v_cvt_pk_bf16_f32 v174, v20, v21
	v_cvt_pk_bf16_f32 v173, v30, v31
	v_cvt_pk_bf16_f32 v172, v18, v19
	v_mfma_f32_32x32x16_bf16 v[32:47], v[56:59], v[168:171], v[32:47]
	s_add_i32 s6, s16, 63
	s_ashr_i32 s7, s6, 31
	s_lshr_b32 s7, s7, 26
	s_add_i32 s6, s6, s7
	v_bfe_u32 v96, v0, 2, 2
	v_bfe_u32 v48, v0, 4, 1
	v_and_b32_e32 v49, 3, v0
	v_mfma_f32_32x32x16_bf16 v[2:17], v[22:25], v[172:175], v[2:17]
	s_ashr_i32 s28, s6, 6
	v_mfma_f32_32x32x16_bf16 v[32:47], v[26:29], v[172:175], v[32:47]
	; sched_barrier mask(0x00000000)
	s_mov_b32 s6, 0xf149f2ca
	s_nop 7
	s_nop 2
	v_max3_f32 v0, v32, s6, v33
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
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_permlane32_swap_b32_e64 v0, v1 bound_ctrl:1
	v_max_f32_e32 v215, v0, v1
	v_sub_f32_e32 v31, v17, v215
	v_sub_f32_e32 v30, v16, v215
	v_sub_f32_e32 v29, v15, v215
	v_sub_f32_e32 v28, v14, v215
	v_sub_f32_e32 v27, v13, v215
	v_sub_f32_e32 v26, v12, v215
	v_sub_f32_e32 v25, v11, v215
	v_sub_f32_e32 v24, v10, v215
	v_sub_f32_e32 v23, v9, v215
	v_sub_f32_e32 v22, v8, v215
	v_sub_f32_e32 v21, v7, v215
	v_sub_f32_e32 v20, v6, v215
	v_sub_f32_e32 v19, v5, v215
	v_sub_f32_e32 v18, v4, v215
	v_sub_f32_e32 v17, v3, v215
	v_sub_f32_e32 v16, v2, v215
	v_sub_f32_e32 v15, v47, v215
	v_sub_f32_e32 v14, v46, v215
	v_sub_f32_e32 v13, v45, v215
	v_sub_f32_e32 v12, v44, v215
	v_sub_f32_e32 v11, v43, v215
	v_sub_f32_e32 v10, v42, v215
	v_sub_f32_e32 v9, v41, v215
	v_sub_f32_e32 v8, v40, v215
	v_sub_f32_e32 v7, v39, v215
	v_sub_f32_e32 v6, v38, v215
	v_sub_f32_e32 v5, v37, v215
	v_sub_f32_e32 v4, v36, v215
	v_sub_f32_e32 v3, v35, v215
	v_sub_f32_e32 v2, v34, v215
	v_sub_f32_e32 v1, v33, v215
	v_sub_f32_e32 v0, v32, v215
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
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
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s26
	s_lshl_b32 s6, s24, 8
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	buffer_load_dwordx4 v210, s[8:11], s6 offen lds
	s_mov_b32 m0, s27
	s_add_i32 s13, s13, 0xc600
	buffer_load_dwordx4 v211, s[8:11], s6 offen lds
	s_cmpk_gt_i32 s16, 0x100
	v_lshlrev_b32_e32 v207, 3, v49
	v_lshlrev_b32_e32 v216, 5, v48
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	v_mul_u32_u24_e32 v32, 0x880, v208
	v_mul_u32_u24_e32 v33, 0x220, v96
	v_lshlrev_b32_e32 v176, 3, v49
	v_lshlrev_b32_e32 v177, 5, v48
	v_add_lshl_u32 v206, v32, v33, 1
	s_mov_b64 s[6:7], 0
	s_branch .LBB0_5
.LBB0_4:
	s_mov_b64 s[6:7], -1
                                        ; implicit-def: $vgpr176
                                        ; implicit-def: $vgpr177
                                        ; implicit-def: $vgpr206
.LBB0_5:
	v_mul_u32_u24_e32 v214, 0x410, v113
	s_add_i32 s19, s28, -1
	s_add_i32 s16, s13, 0x2200
	s_andn2_b64 vcc, exec, s[6:7]
	s_mov_b32 s36, 0
	s_cbranch_vccnz .LBB0_13
; %bb.6:
	v_mul_u32_u24_e32 v33, 0x880, v208
	v_mul_u32_u24_e32 v34, 0x220, v96
	s_mov_b32 s6, 0x8500
	v_add_lshl_u32 v206, v33, v34, 1
	v_add3_u32 v32, v205, v204, s6
	v_or3_b32 v33, v207, v216, v206
	v_mov_b32_e32 v212, 0
	s_mov_b32 s29, 3
	v_add_u32_e32 v217, 0x4100, v33
	v_add_u32_e32 v218, 0xc600, v33
	s_mul_i32 s30, s17, 0x80000002
	s_lshl_b32 s31, s24, 9
	s_mul_i32 s33, s24, 0x180
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	v_add_u32_e32 v219, v32, v214
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_mov_b32 s34, 0xf149f2ca
	s_mov_b32 s35, 0x41000000
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, v212
	v_mov_b32_e32 v82, v212
	v_mov_b32_e32 v83, v212
	v_mov_b32_e32 v84, v212
	v_mov_b32_e32 v85, v212
	v_mov_b32_e32 v86, v212
	v_mov_b32_e32 v87, v212
	v_mov_b32_e32 v88, v212
	v_mov_b32_e32 v89, v212
	v_mov_b32_e32 v90, v212
	v_mov_b32_e32 v91, v212
	v_mov_b32_e32 v92, v212
	v_mov_b32_e32 v93, v212
	v_mov_b32_e32 v94, v212
	v_mov_b32_e32 v95, v212
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, v212
	v_mov_b32_e32 v66, v212
	v_mov_b32_e32 v67, v212
	v_mov_b32_e32 v68, v212
	v_mov_b32_e32 v69, v212
	v_mov_b32_e32 v70, v212
	v_mov_b32_e32 v71, v212
	v_mov_b32_e32 v72, v212
	v_mov_b32_e32 v73, v212
	v_mov_b32_e32 v74, v212
	v_mov_b32_e32 v75, v212
	v_mov_b32_e32 v76, v212
	v_mov_b32_e32 v77, v212
	v_mov_b32_e32 v78, v212
	v_mov_b32_e32 v79, v212
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, v212
	v_mov_b32_e32 v50, v212
	v_mov_b32_e32 v51, v212
	v_mov_b32_e32 v52, v212
	v_mov_b32_e32 v53, v212
	v_mov_b32_e32 v54, v212
	v_mov_b32_e32 v55, v212
	v_mov_b32_e32 v56, v212
	v_mov_b32_e32 v57, v212
	v_mov_b32_e32 v58, v212
	v_mov_b32_e32 v59, v212
	v_mov_b32_e32 v60, v212
	v_mov_b32_e32 v61, v212
	v_mov_b32_e32 v62, v212
	v_mov_b32_e32 v63, v212
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, v212
	v_mov_b32_e32 v34, v212
	v_mov_b32_e32 v35, v212
	v_mov_b32_e32 v36, v212
	v_mov_b32_e32 v37, v212
	v_mov_b32_e32 v38, v212
	v_mov_b32_e32 v39, v212
	v_mov_b32_e32 v40, v212
	v_mov_b32_e32 v41, v212
	v_mov_b32_e32 v42, v212
	v_mov_b32_e32 v43, v212
	v_mov_b32_e32 v44, v212
	v_mov_b32_e32 v45, v212
	v_mov_b32_e32 v46, v212
	v_mov_b32_e32 v47, v212
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 m0, s13
	s_add_i32 s37, s17, s36
	buffer_load_dwordx4 v210, s[4:7], s37 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v211, s[4:7], s37 offen lds
	ds_read_b128 v[96:99], v219
	ds_read_b128 v[116:119], v219 offset:64
	ds_read_b128 v[120:123], v219 offset:96
	ds_read_b128 v[124:127], v219 offset:8320
	ds_read_b128 v[176:179], v219 offset:8352
	ds_read_b128 v[180:183], v219 offset:8384
	ds_read_b128 v[184:187], v219 offset:8416
	ds_read_b128 v[128:131], v219 offset:512
	ds_read_b128 v[188:191], v219 offset:544
	ds_read_b128 v[192:195], v219 offset:576
	ds_read_b128 v[196:199], v219 offset:608
	ds_read_b128 v[200:203], v219 offset:8832
	ds_read_b128 v[112:115], v219 offset:32
	ds_read_b128 v[220:223], v219 offset:8864
	ds_read_b128 v[224:227], v219 offset:8896
	ds_read_b128 v[228:231], v219 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[96:99], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[128:131], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[112:115], v[148:151], v[96:111]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[188:191], v[148:151], v[128:143]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[116:119], v[152:155], v[96:111]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[192:195], v[152:155], v[128:143]
	v_exp_f32_e32 v31, v31
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[120:123], v[156:159], v[96:111]
	v_add_f32_e32 v112, v1, v0
	v_add_f32_e32 v112, v112, v2
	v_add_f32_e32 v112, v112, v3
	v_add_f32_e32 v112, v112, v4
	v_add_f32_e32 v112, v112, v5
	v_cvt_pk_bf16_f32 v120, v16, v17
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[196:199], v[156:159], v[128:143]
	v_add_f32_e32 v112, v112, v6
	v_add_f32_e32 v112, v112, v7
	v_add_f32_e32 v112, v112, v8
	v_add_f32_e32 v112, v112, v9
	v_add_f32_e32 v112, v112, v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[124:127], v[160:163], v[96:111]
	v_add_f32_e32 v112, v112, v11
	v_add_f32_e32 v112, v112, v12
	v_add_f32_e32 v112, v112, v13
	v_add_f32_e32 v112, v112, v14
	v_add_f32_e32 v112, v112, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[200:203], v[160:163], v[128:143]
	v_add_f32_e32 v112, v112, v16
	v_add_f32_e32 v112, v112, v17
	v_add_f32_e32 v112, v112, v18
	v_add_f32_e32 v112, v112, v19
	v_add_f32_e32 v112, v112, v20
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[176:179], v[164:167], v[96:111]
	v_add_f32_e32 v112, v112, v21
	v_add_f32_e32 v112, v112, v22
	v_add_f32_e32 v112, v112, v23
	v_add_f32_e32 v112, v112, v24
	v_add_f32_e32 v112, v112, v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[220:223], v[164:167], v[128:143]
	v_add_f32_e32 v112, v112, v26
	v_add_f32_e32 v112, v112, v27
	v_add_f32_e32 v112, v112, v28
	v_add_f32_e32 v112, v112, v29
	v_add_f32_e32 v112, v112, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[180:183], v[168:171], v[96:111]
	v_add_f32_e32 v112, v112, v31
	v_mov_b32_e32 v113, v112
	s_nop 1
	v_permlane32_swap_b32_e64 v112, v113 bound_ctrl:1
	v_add_f32_e32 v113, v212, v113
	v_add_f32_e32 v212, v113, v112
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[168:171], v[128:143]
	v_cvt_pk_bf16_f32 v119, v14, v15
	v_cvt_pk_bf16_f32 v118, v12, v13
	v_cvt_pk_bf16_f32 v117, v10, v11
	v_cvt_pk_bf16_f32 v116, v8, v9
	v_cvt_pk_bf16_f32 v115, v6, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[184:187], v[172:175], v[96:111]
	v_cvt_pk_bf16_f32 v114, v4, v5
	v_cvt_pk_bf16_f32 v113, v2, v3
	v_cvt_pk_bf16_f32 v112, v0, v1
	v_cvt_pk_bf16_f32 v127, v30, v31
	v_cvt_pk_bf16_f32 v126, v28, v29
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[172:175], v[128:143]
	v_cvt_pk_bf16_f32 v125, v26, v27
	v_cvt_pk_bf16_f32 v124, v24, v25
	v_cvt_pk_bf16_f32 v123, v22, v23
	v_cvt_pk_bf16_f32 v122, v20, v21
	v_cvt_pk_bf16_f32 v121, v18, v19
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s37, s33, s36
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v210, s[8:11], s37 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v211, s[8:11], s37 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v217 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v217 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v217 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0:1], v217 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[2:3], v217 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v217 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v217 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v217 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v217 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[4:5], v217 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[6:7], v217 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v217 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v217 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v217 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v217 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v217 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v217 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[8:9], v217 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[10:11], v217 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v217 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v217 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v217 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v217 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v217 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v217 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[12:13], v217 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v217 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v217 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v217 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v217 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v217 offset:0x25c0

	;;#ASMEND
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[112:115], v[80:95]
	v_max3_f32 v20, v96, s34, v97
	v_max3_f32 v20, v20, v98, v99
	v_max3_f32 v20, v20, v100, v101
	v_max3_f32 v20, v20, v102, v103
	v_max3_f32 v20, v20, v104, v105
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[24:27], v[112:115], v[64:79]
	v_max3_f32 v20, v20, v106, v107
	v_max3_f32 v20, v20, v108, v109
	v_max3_f32 v20, v20, v110, v111
	v_max3_f32 v20, v20, v128, v129
	v_max3_f32 v20, v20, v130, v131
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[28:31], v[112:115], v[48:63]
	v_max3_f32 v20, v20, v132, v133
	v_max3_f32 v20, v20, v134, v135
	v_max3_f32 v20, v20, v136, v137
	v_max3_f32 v20, v20, v138, v139
	v_max3_f32 v20, v20, v140, v141
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[220:223], v[112:115], v[32:47]
	v_max3_f32 v20, v20, v142, v143
	v_mov_b32_e32 v21, v20
	s_nop 1
	v_permlane32_swap_b32_e64 v20, v21 bound_ctrl:1
	v_max_f32_e32 v20, v20, v21
	v_sub_f32_e32 v21, v20, v215
	v_cmp_ge_f32_e32 vcc, s35, v21
	s_cmp_eq_u64 vcc, exec
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	s_cbranch_scc0 .LBB0_11
.LBB0_8:                                ;   in Loop: Header=BB0_7 Depth=1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[116:119], v[80:95]
	v_sub_f32_e32 v31, v143, v215
	v_sub_f32_e32 v30, v142, v215
	v_sub_f32_e32 v29, v141, v215
	v_sub_f32_e32 v28, v140, v215
	v_sub_f32_e32 v27, v139, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[4:7], v[116:119], v[64:79]
	v_sub_f32_e32 v26, v138, v215
	v_sub_f32_e32 v25, v137, v215
	v_sub_f32_e32 v24, v136, v215
	v_sub_f32_e32 v23, v135, v215
	v_sub_f32_e32 v22, v134, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[8:11], v[116:119], v[48:63]
	v_sub_f32_e32 v21, v133, v215
	v_sub_f32_e32 v20, v132, v215
	v_sub_f32_e32 v19, v131, v215
	v_sub_f32_e32 v18, v130, v215
	v_sub_f32_e32 v17, v129, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[12:15], v[116:119], v[32:47]
	v_sub_f32_e32 v16, v128, v215
	v_sub_f32_e32 v15, v111, v215
	v_sub_f32_e32 v14, v110, v215
	v_sub_f32_e32 v13, v109, v215
	v_sub_f32_e32 v12, v108, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[0:3], v[120:123], v[80:95]
	v_sub_f32_e32 v11, v107, v215
	v_sub_f32_e32 v10, v106, v215
	v_sub_f32_e32 v9, v105, v215
	v_sub_f32_e32 v8, v104, v215
	v_sub_f32_e32 v7, v103, v215
	v_sub_f32_e32 v1, v97, v215
	v_sub_f32_e32 v0, v96, v215
	v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[120:123], v[64:79]
	v_sub_f32_e32 v6, v102, v215
	v_sub_f32_e32 v5, v101, v215
	v_sub_f32_e32 v4, v100, v215
	v_sub_f32_e32 v3, v99, v215
	v_sub_f32_e32 v2, v98, v215
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[120:123], v[48:63]
	v_exp_f32_e32 v200, v0
	v_exp_f32_e32 v201, v1
	v_exp_f32_e32 v202, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[120:123], v[32:47]
	v_exp_f32_e32 v203, v3
	v_exp_f32_e32 v220, v4
	v_exp_f32_e32 v221, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[124:127], v[80:95]
	v_exp_f32_e32 v222, v6
	v_exp_f32_e32 v223, v7
	v_exp_f32_e32 v224, v8
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[124:127], v[64:79]
	v_exp_f32_e32 v225, v9
	v_exp_f32_e32 v226, v10
	v_exp_f32_e32 v227, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[124:127], v[48:63]
	v_exp_f32_e32 v228, v12
	v_exp_f32_e32 v229, v13
	v_exp_f32_e32 v230, v14
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[188:191], v[124:127], v[32:47]
	v_exp_f32_e32 v231, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s20
	s_add_i32 s37, s30, s36
	buffer_load_dwordx4 v210, s[4:7], s37 offen lds
	s_mov_b32 m0, s18
	v_add_u32_e32 v4, v213, v214
	buffer_load_dwordx4 v211, s[4:7], s37 offen lds
	ds_read_b128 v[0:3], v4
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
	ds_read_b128 v[96:99], v4 offset:32
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[112:115], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[148:151], v[0:15]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[140:143], v[148:151], v[112:127]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[100:103], v[152:155], v[0:15]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[176:179], v[152:155], v[112:127]
	v_exp_f32_e32 v31, v31
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[104:107], v[156:159], v[0:15]
	v_add_f32_e32 v96, v201, v200
	v_add_f32_e32 v96, v96, v202
	v_add_f32_e32 v96, v96, v203
	v_add_f32_e32 v96, v96, v220
	v_add_f32_e32 v96, v96, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[180:183], v[156:159], v[112:127]
	v_add_f32_e32 v96, v96, v222
	v_add_f32_e32 v96, v96, v223
	v_add_f32_e32 v96, v96, v224
	v_add_f32_e32 v96, v96, v225
	v_add_f32_e32 v96, v96, v226
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[108:111], v[160:163], v[0:15]
	v_add_f32_e32 v96, v96, v227
	v_add_f32_e32 v96, v96, v228
	v_add_f32_e32 v96, v96, v229
	v_add_f32_e32 v96, v96, v230
	v_add_f32_e32 v96, v96, v231
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[184:187], v[160:163], v[112:127]
	v_add_f32_e32 v96, v96, v16
	v_add_f32_e32 v96, v96, v17
	v_add_f32_e32 v96, v96, v18
	v_add_f32_e32 v96, v96, v19
	v_add_f32_e32 v96, v96, v20
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[164:167], v[0:15]
	v_add_f32_e32 v96, v96, v21
	v_add_f32_e32 v96, v96, v22
	v_add_f32_e32 v96, v96, v23
	v_add_f32_e32 v96, v96, v24
	v_add_f32_e32 v96, v96, v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[188:191], v[164:167], v[112:127]
	v_add_f32_e32 v96, v96, v26
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[168:171], v[0:15]
	v_add_f32_e32 v96, v96, v31
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_add_f32_e32 v97, v212, v97
	v_add_f32_e32 v212, v97, v96
	v_cvt_pk_bf16_f32 v96, v200, v201
	v_mfma_f32_32x32x16_bf16 v[112:127], v[192:195], v[168:171], v[112:127]
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[172:175], v[0:15]
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v230, v231
	v_cvt_pk_bf16_f32 v102, v228, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[196:199], v[172:175], v[112:127]
	v_cvt_pk_bf16_f32 v101, v226, v227
	v_cvt_pk_bf16_f32 v100, v224, v225
	v_cvt_pk_bf16_f32 v99, v222, v223
	v_cvt_pk_bf16_f32 v98, v220, v221
	v_cvt_pk_bf16_f32 v97, v202, v203
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s36, s31, s36
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v210, s[8:11], s36 offen lds
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v211, s[8:11], s36 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v218 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v218 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v218 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v218 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v218 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v218 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v218 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v218 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v218 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v218 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v218 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v218 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v218 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v218 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v218 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v218 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v218 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v218 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v218 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v218 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v218 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v218 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v218 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v218 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v218 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v218 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v218 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v218 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v218 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v218 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v218 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v218 offset:0x25c0

	;;#ASMEND
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[24:27], v[96:99], v[80:95]
	v_max3_f32 v24, v0, s34, v1
	v_max3_f32 v24, v24, v2, v3
	v_max3_f32 v24, v24, v4, v5
	v_max3_f32 v24, v24, v6, v7
	v_max3_f32 v24, v24, v8, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[28:31], v[96:99], v[64:79]
	v_max3_f32 v24, v24, v10, v11
	v_max3_f32 v24, v24, v12, v13
	v_max3_f32 v24, v24, v14, v15
	v_max3_f32 v24, v24, v112, v113
	v_max3_f32 v24, v24, v114, v115
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[96:99], v[48:63]
	v_max3_f32 v24, v24, v116, v117
	v_max3_f32 v24, v24, v118, v119
	v_max3_f32 v24, v24, v120, v121
	v_max3_f32 v24, v24, v122, v123
	v_max3_f32 v24, v24, v124, v125
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[220:223], v[96:99], v[32:47]
	v_max3_f32 v24, v24, v126, v127
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1
	v_max_f32_e32 v24, v24, v25
	v_sub_f32_e32 v25, v24, v215
	v_cmp_ge_f32_e32 vcc, s35, v25
	s_cmp_eq_u64 vcc, exec
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	s_cbranch_scc0 .LBB0_12
.LBB0_9:                                ;   in Loop: Header=BB0_7 Depth=1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[100:103], v[80:95]
	v_sub_f32_e32 v31, v127, v215
	v_sub_f32_e32 v30, v126, v215
	v_sub_f32_e32 v29, v125, v215
	v_sub_f32_e32 v28, v124, v215
	v_sub_f32_e32 v27, v123, v215
	v_sub_f32_e32 v1, v1, v215
	v_sub_f32_e32 v0, v0, v215
	v_mfma_f32_32x32x16_bf16 v[64:79], v[16:19], v[100:103], v[64:79]
	v_sub_f32_e32 v26, v122, v215
	v_sub_f32_e32 v25, v121, v215
	v_sub_f32_e32 v24, v120, v215
	v_sub_f32_e32 v23, v119, v215
	v_sub_f32_e32 v22, v118, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[192:195], v[100:103], v[48:63]
	v_sub_f32_e32 v21, v117, v215
	v_sub_f32_e32 v20, v116, v215
	v_sub_f32_e32 v19, v115, v215
	v_sub_f32_e32 v18, v114, v215
	v_sub_f32_e32 v17, v113, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[100:103], v[32:47]
	v_sub_f32_e32 v16, v112, v215
	v_sub_f32_e32 v15, v15, v215
	v_sub_f32_e32 v14, v14, v215
	v_sub_f32_e32 v13, v13, v215
	v_sub_f32_e32 v12, v12, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[104:107], v[80:95]
	v_sub_f32_e32 v11, v11, v215
	v_sub_f32_e32 v10, v10, v215
	v_sub_f32_e32 v9, v9, v215
	v_sub_f32_e32 v8, v8, v215
	v_sub_f32_e32 v7, v7, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[104:107], v[64:79]
	v_sub_f32_e32 v6, v6, v215
	v_sub_f32_e32 v5, v5, v215
	v_sub_f32_e32 v4, v4, v215
	v_sub_f32_e32 v3, v3, v215
	v_sub_f32_e32 v2, v2, v215
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[104:107], v[48:63]
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v2, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[104:107], v[32:47]
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[136:139], v[108:111], v[80:95]
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v8, v8
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[140:143], v[108:111], v[64:79]
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	v_exp_f32_e32 v11, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[132:135], v[108:111], v[48:63]
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_exp_f32_e32 v14, v14
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[128:131], v[108:111], v[32:47]
	v_exp_f32_e32 v15, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s29, s29, 2
	s_cmp_ge_i32 s29, s19
	s_cbranch_scc1 .LBB0_14
; %bb.10:                               ;   in Loop: Header=BB0_7 Depth=1
	s_mov_b32 s36, s37
	s_branch .LBB0_7
.LBB0_11:                               ;   in Loop: Header=BB0_7 Depth=1
	v_sub_f32_e32 v21, v215, v20
	v_exp_f32_e32 v22, v21
	v_mov_b32_e32 v215, v20
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
	v_mul_f32_e32 v212, v22, v212
	s_branch .LBB0_8
.LBB0_12:                               ;   in Loop: Header=BB0_7 Depth=1
	v_sub_f32_e32 v25, v215, v24
	v_exp_f32_e32 v26, v25
	v_mov_b32_e32 v215, v24
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
	v_mul_f32_e32 v212, v26, v212
	s_branch .LBB0_9
.LBB0_13:
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
	v_mov_b32_e32 v212, v47
	s_branch .LBB0_15
.LBB0_14:
	v_mov_b32_e32 v176, v207
	v_mov_b32_e32 v177, v216
.LBB0_15:
	s_lshl_b32 s6, s28, 1
	s_lshl_b32 s10, s24, 6
	s_add_i32 s24, s6, -6
	s_mov_b32 m0, s13
	s_mul_i32 s24, s24, s10
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v210, s[4:7], s24 offen lds
	s_mov_b32 m0, s16
	v_add3_u32 v219, v205, v204, v214
	buffer_load_dwordx4 v211, s[4:7], s24 offen lds
	ds_read_b128 v[112:115], v219 offset:34080
	ds_read_b128 v[178:181], v219 offset:42400
	ds_read_b128 v[198:201], v219 offset:34656
	ds_read_b128 v[202:205], v219 offset:42880
	ds_read_b128 v[96:99], v219 offset:34048
	ds_read_b128 v[116:119], v219 offset:34112
	ds_read_b128 v[120:123], v219 offset:34144
	ds_read_b128 v[124:127], v219 offset:42368
	ds_read_b128 v[182:185], v219 offset:42432
	ds_read_b128 v[186:189], v219 offset:42464
	ds_read_b128 v[128:131], v219 offset:34560
	ds_read_b128 v[190:193], v219 offset:34592
	ds_read_b128 v[194:197], v219 offset:34624
	ds_read_b128 v[220:223], v219 offset:42912
	ds_read_b128 v[224:227], v219 offset:42944
	ds_read_b128 v[228:231], v219 offset:42976
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[96:99], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[128:131], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[112:115], v[148:151], v[96:111]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[190:193], v[148:151], v[128:143]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[116:119], v[152:155], v[96:111]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[194:197], v[152:155], v[128:143]
	v_exp_f32_e32 v31, v31
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[120:123], v[156:159], v[96:111]
	v_add_f32_e32 v112, v1, v0
	v_add_f32_e32 v112, v112, v2
	v_add_f32_e32 v112, v112, v3
	v_add_f32_e32 v112, v112, v4
	v_add_f32_e32 v112, v112, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[198:201], v[156:159], v[128:143]
	v_add_f32_e32 v112, v112, v6
	v_add_f32_e32 v112, v112, v7
	v_add_f32_e32 v112, v112, v8
	v_add_f32_e32 v112, v112, v9
	v_add_f32_e32 v112, v112, v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[124:127], v[160:163], v[96:111]
	v_add_f32_e32 v112, v112, v11
	v_add_f32_e32 v112, v112, v12
	v_add_f32_e32 v112, v112, v13
	v_add_f32_e32 v112, v112, v14
	v_add_f32_e32 v112, v112, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[202:205], v[160:163], v[128:143]
	v_add_f32_e32 v112, v112, v16
	v_add_f32_e32 v112, v112, v17
	v_add_f32_e32 v112, v112, v18
	v_add_f32_e32 v112, v112, v19
	v_add_f32_e32 v112, v112, v20
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[178:181], v[164:167], v[96:111]
	v_add_f32_e32 v112, v112, v21
	v_add_f32_e32 v112, v112, v22
	v_add_f32_e32 v112, v112, v23
	v_add_f32_e32 v112, v112, v24
	v_add_f32_e32 v112, v112, v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[220:223], v[164:167], v[128:143]
	v_add_f32_e32 v112, v112, v26
	v_add_f32_e32 v112, v112, v27
	v_add_f32_e32 v112, v112, v28
	v_add_f32_e32 v112, v112, v29
	v_add_f32_e32 v112, v112, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[182:185], v[168:171], v[96:111]
	v_add_f32_e32 v216, v112, v31
	v_mov_b32_e32 v217, v216
	s_nop 1
	v_permlane32_swap_b32_e64 v216, v217 bound_ctrl:1
	v_cvt_pk_bf16_f32 v119, v14, v15
	v_cvt_pk_bf16_f32 v118, v12, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[168:171], v[128:143]
	v_cvt_pk_bf16_f32 v117, v10, v11
	v_cvt_pk_bf16_f32 v116, v8, v9
	v_cvt_pk_bf16_f32 v115, v6, v7
	v_cvt_pk_bf16_f32 v114, v4, v5
	v_cvt_pk_bf16_f32 v113, v2, v3
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[96:111], v[186:189], v[172:175], v[96:111]
	v_cvt_pk_bf16_f32 v112, v0, v1
	v_cvt_pk_bf16_f32 v127, v30, v31
	v_cvt_pk_bf16_f32 v126, v28, v29
	v_cvt_pk_bf16_f32 v125, v26, v27
	v_cvt_pk_bf16_f32 v124, v24, v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[172:175], v[128:143]
	v_cvt_pk_bf16_f32 v123, v22, v23
	v_cvt_pk_bf16_f32 v122, v20, v21
	v_cvt_pk_bf16_f32 v121, v18, v19
	v_cvt_pk_bf16_f32 v120, v16, v17
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mul_i32 s19, s19, s10
	s_mov_b32 m0, s25
	s_lshl_b32 s19, s19, 1
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	buffer_load_dwordx4 v210, s[8:11], s19 offen lds
	s_mov_b32 m0, s21
	v_add3_u32 v220, v176, v177, v206
	buffer_load_dwordx4 v211, s[8:11], s19 offen lds
	v_add_u32_e32 v218, 0x4100, v220
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v218 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v218 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[4:5], v218 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[6:7], v218 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0:1], v218 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[2:3], v218 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v218 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v218 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v218 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v218 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[8:9], v218 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[10:11], v218 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v218 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v218 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v218 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v218 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v218 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v218 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[12:13], v218 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v218 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v218 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v218 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v218 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v218 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v218 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v218 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v218 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v218 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v218 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v218 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v218 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v218 offset:0x25c0

	;;#ASMEND
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_mov_b32 s2, 0xf149f2ca
	v_mfma_f32_32x32x16_bf16 v[80:95], v[28:31], v[112:115], v[80:95]
	v_max3_f32 v28, v96, s2, v97
	v_max3_f32 v28, v28, v98, v99
	v_max3_f32 v28, v28, v100, v101
	v_max3_f32 v28, v28, v102, v103
	v_max3_f32 v28, v28, v104, v105
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[16:19], v[112:115], v[64:79]
	v_max3_f32 v16, v28, v106, v107
	v_max3_f32 v16, v16, v108, v109
	v_max3_f32 v16, v16, v110, v111
	v_max3_f32 v16, v16, v128, v129
	v_max3_f32 v16, v16, v130, v131
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[20:23], v[112:115], v[48:63]
	v_max3_f32 v16, v16, v132, v133
	v_max3_f32 v16, v16, v134, v135
	v_max3_f32 v16, v16, v136, v137
	v_max3_f32 v16, v16, v138, v139
	v_max3_f32 v16, v16, v140, v141
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[24:27], v[112:115], v[32:47]
	v_max3_f32 v16, v16, v142, v143
	v_mov_b32_e32 v17, v16
	s_nop 1
	v_permlane32_swap_b32_e64 v16, v17 bound_ctrl:1
	v_max3_f32 v221, v215, v16, v17
	v_sub_f32_e32 v112, v215, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[4:7], v[116:119], v[80:95]
	v_sub_f32_e32 v31, v143, v221
	v_sub_f32_e32 v30, v142, v221
	v_sub_f32_e32 v29, v141, v221
	v_sub_f32_e32 v28, v140, v221
	v_sub_f32_e32 v27, v139, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[8:11], v[116:119], v[64:79]
	v_sub_f32_e32 v26, v138, v221
	v_sub_f32_e32 v25, v137, v221
	v_sub_f32_e32 v24, v136, v221
	v_sub_f32_e32 v23, v135, v221
	v_sub_f32_e32 v22, v134, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[12:15], v[116:119], v[48:63]
	v_sub_f32_e32 v21, v133, v221
	v_sub_f32_e32 v20, v132, v221
	v_sub_f32_e32 v19, v131, v221
	v_sub_f32_e32 v18, v130, v221
	v_sub_f32_e32 v17, v129, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[116:119], v[32:47]
	v_sub_f32_e32 v16, v128, v221
	v_sub_f32_e32 v15, v111, v221
	v_sub_f32_e32 v14, v110, v221
	v_sub_f32_e32 v13, v109, v221
	v_sub_f32_e32 v12, v108, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[0:3], v[120:123], v[80:95]
	v_sub_f32_e32 v11, v107, v221
	v_sub_f32_e32 v10, v106, v221
	v_sub_f32_e32 v9, v105, v221
	v_sub_f32_e32 v8, v104, v221
	v_sub_f32_e32 v7, v103, v221
	v_sub_f32_e32 v1, v97, v221
	v_sub_f32_e32 v0, v96, v221
	v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[120:123], v[64:79]
	v_sub_f32_e32 v6, v102, v221
	v_sub_f32_e32 v5, v101, v221
	v_sub_f32_e32 v4, v100, v221
	v_sub_f32_e32 v3, v99, v221
	v_sub_f32_e32 v2, v98, v221
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[120:123], v[48:63]
	v_exp_f32_e32 v200, v112
	v_exp_f32_e32 v203, v0
	v_exp_f32_e32 v204, v1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[120:123], v[32:47]
	v_exp_f32_e32 v205, v2
	v_exp_f32_e32 v206, v3
	v_exp_f32_e32 v207, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[124:127], v[80:95]
	v_exp_f32_e32 v215, v5
	v_exp_f32_e32 v222, v6
	v_exp_f32_e32 v223, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[124:127], v[64:79]
	v_exp_f32_e32 v224, v8
	v_exp_f32_e32 v225, v9
	v_exp_f32_e32 v226, v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[124:127], v[48:63]
	v_exp_f32_e32 v227, v11
	v_exp_f32_e32 v228, v12
	v_exp_f32_e32 v229, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[188:191], v[124:127], v[32:47]
	v_exp_f32_e32 v230, v14
	v_exp_f32_e32 v231, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[94:95], v[200:201], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[200:201], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[200:201], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[200:201], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[200:201], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[200:201], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[200:201], v[82:83] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[200:201], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[200:201], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[200:201], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[200:201], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[200:201], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[200:201], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[200:201], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[200:201], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[200:201], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[200:201], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[200:201], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[200:201], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[200:201], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[200:201], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[200:201], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[200:201], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[200:201], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[200:201], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[200:201], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[200:201], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[200:201], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[200:201], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[200:201], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[200:201], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[200:201], v[32:33] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s20
	s_add_i32 s24, s24, s17
	buffer_load_dwordx4 v210, s[4:7], s24 offen lds
	s_mov_b32 m0, s18
	v_add_u32_e32 v4, v213, v214
	buffer_load_dwordx4 v211, s[4:7], s24 offen lds
	ds_read_b128 v[0:3], v4
	ds_read_b128 v[112:115], v4 offset:512
	ds_read_b128 v[176:179], v4 offset:576
	ds_read_b128 v[180:183], v4 offset:608
	ds_read_b128 v[96:99], v4 offset:32
	ds_read_b128 v[100:103], v4 offset:64
	ds_read_b128 v[104:107], v4 offset:96
	ds_read_b128 v[108:111], v4 offset:8320
	ds_read_b128 v[128:131], v4 offset:8352
	ds_read_b128 v[132:135], v4 offset:8384
	ds_read_b128 v[136:139], v4 offset:8416
	ds_read_b128 v[140:143], v4 offset:544
	ds_read_b128 v[184:187], v4 offset:8832
	ds_read_b128 v[188:191], v4 offset:8864
	ds_read_b128 v[192:195], v4 offset:8896
	ds_read_b128 v[196:199], v4 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[112:115], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[148:151], v[0:15]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[140:143], v[148:151], v[112:127]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[100:103], v[152:155], v[0:15]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[176:179], v[152:155], v[112:127]
	v_exp_f32_e32 v31, v31
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[104:107], v[156:159], v[0:15]
	v_add_f32_e32 v96, v204, v203
	v_add_f32_e32 v96, v96, v205
	v_add_f32_e32 v96, v96, v206
	v_add_f32_e32 v96, v96, v207
	v_add_f32_e32 v96, v96, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[180:183], v[156:159], v[112:127]
	v_add_f32_e32 v96, v96, v222
	v_add_f32_e32 v96, v96, v223
	v_add_f32_e32 v96, v96, v224
	v_add_f32_e32 v96, v96, v225
	v_add_f32_e32 v96, v96, v226
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[108:111], v[160:163], v[0:15]
	v_add_f32_e32 v96, v96, v227
	v_add_f32_e32 v96, v96, v228
	v_add_f32_e32 v96, v96, v229
	v_add_f32_e32 v96, v96, v230
	v_add_f32_e32 v96, v96, v231
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[184:187], v[160:163], v[112:127]
	v_add_f32_e32 v96, v96, v16
	v_add_f32_e32 v96, v96, v17
	v_add_f32_e32 v96, v96, v18
	v_add_f32_e32 v96, v96, v19
	v_add_f32_e32 v96, v96, v20
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[164:167], v[0:15]
	v_add_f32_e32 v96, v96, v21
	v_add_f32_e32 v96, v96, v22
	v_add_f32_e32 v96, v96, v23
	v_add_f32_e32 v96, v96, v24
	v_add_f32_e32 v96, v96, v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[188:191], v[164:167], v[112:127]
	v_add_f32_e32 v96, v96, v26
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[168:171], v[0:15]
	v_add_f32_e32 v201, v96, v31
	v_mov_b32_e32 v202, v201
	s_nop 1
	v_permlane32_swap_b32_e64 v201, v202 bound_ctrl:1
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[192:195], v[168:171], v[112:127]
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[172:175], v[0:15]
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v230, v231
	v_cvt_pk_bf16_f32 v102, v228, v229
	v_cvt_pk_bf16_f32 v101, v226, v227
	v_cvt_pk_bf16_f32 v100, v224, v225
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[196:199], v[172:175], v[112:127]
	v_cvt_pk_bf16_f32 v99, v222, v223
	v_cvt_pk_bf16_f32 v98, v207, v215
	v_cvt_pk_bf16_f32 v97, v205, v206
	v_cvt_pk_bf16_f32 v96, v203, v204
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v203, 0xc600, v220
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v203 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v203 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v203 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v203 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v203 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v203 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v203 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v203 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v203 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v203 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v203 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v203 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v203 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v203 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v203 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v203 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v203 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v203 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v203 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v203 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v203 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v203 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v203 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v203 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[224:225], v203 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v203 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v203 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v203 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v203 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v203 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v203 offset:0x25c0

	;;#ASMEND
	s_waitcnt vmcnt(2) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[80:95], v[24:27], v[96:99], v[80:95]
	v_max3_f32 v24, v0, s2, v1
	v_max3_f32 v24, v24, v2, v3
	v_max3_f32 v24, v24, v4, v5
	v_max3_f32 v24, v24, v6, v7
	v_max3_f32 v24, v24, v8, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[28:31], v[96:99], v[64:79]
	v_max3_f32 v24, v24, v10, v11
	v_max3_f32 v24, v24, v12, v13
	v_max3_f32 v24, v24, v14, v15
	v_max3_f32 v24, v24, v112, v113
	v_max3_f32 v24, v24, v114, v115
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[204:207], v[96:99], v[48:63]
	v_max3_f32 v24, v24, v116, v117
	v_max3_f32 v24, v24, v118, v119
	v_max3_f32 v24, v24, v120, v121
	v_max3_f32 v24, v24, v122, v123
	v_max3_f32 v24, v24, v124, v125
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[222:225], v[96:99], v[32:47]
	v_max3_f32 v24, v24, v126, v127
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1
	v_max3_f32 v204, v221, v24, v25
	v_sub_f32_e32 v96, v221, v204
	v_sub_f32_e32 v1, v1, v204
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[100:103], v[80:95]
	v_sub_f32_e32 v31, v127, v204
	v_sub_f32_e32 v30, v126, v204
	v_sub_f32_e32 v29, v125, v204
	v_sub_f32_e32 v28, v124, v204
	v_sub_f32_e32 v27, v123, v204
	v_sub_f32_e32 v0, v0, v204
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[20:23], v[100:103], v[64:79]
	v_sub_f32_e32 v26, v122, v204
	v_sub_f32_e32 v25, v121, v204
	v_sub_f32_e32 v24, v120, v204
	v_sub_f32_e32 v23, v119, v204
	v_sub_f32_e32 v22, v118, v204
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[192:195], v[100:103], v[48:63]
	v_sub_f32_e32 v21, v117, v204
	v_sub_f32_e32 v20, v116, v204
	v_sub_f32_e32 v19, v115, v204
	v_sub_f32_e32 v18, v114, v204
	v_sub_f32_e32 v17, v113, v204
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[100:103], v[32:47]
	v_sub_f32_e32 v16, v112, v204
	v_sub_f32_e32 v15, v15, v204
	v_sub_f32_e32 v14, v14, v204
	v_sub_f32_e32 v13, v13, v204
	v_sub_f32_e32 v12, v12, v204
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[104:107], v[80:95]
	v_sub_f32_e32 v11, v11, v204
	v_sub_f32_e32 v10, v10, v204
	v_sub_f32_e32 v9, v9, v204
	v_sub_f32_e32 v8, v8, v204
	v_sub_f32_e32 v7, v7, v204
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[104:107], v[64:79]
	v_sub_f32_e32 v6, v6, v204
	v_sub_f32_e32 v5, v5, v204
	v_sub_f32_e32 v4, v4, v204
	v_sub_f32_e32 v3, v3, v204
	v_sub_f32_e32 v2, v2, v204
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[104:107], v[48:63]
	v_exp_f32_e32 v180, v96
	v_exp_f32_e32 v181, v0
	v_exp_f32_e32 v198, v1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[104:107], v[32:47]
	v_exp_f32_e32 v199, v2
	v_exp_f32_e32 v205, v3
	v_exp_f32_e32 v206, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[108:111], v[80:95]
	v_exp_f32_e32 v207, v5
	v_exp_f32_e32 v213, v6
	v_exp_f32_e32 v214, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[132:135], v[108:111], v[64:79]
	v_exp_f32_e32 v215, v8
	v_exp_f32_e32 v224, v9
	v_exp_f32_e32 v225, v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[136:139], v[108:111], v[48:63]
	v_exp_f32_e32 v226, v11
	v_exp_f32_e32 v227, v12
	v_exp_f32_e32 v228, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[140:143], v[108:111], v[32:47]
	v_exp_f32_e32 v229, v14
	v_exp_f32_e32 v230, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	; sched_barrier mask(0x00000000)
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
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s13
	s_nop 0
	buffer_load_dwordx4 v210, s[4:7], s19 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v211, s[4:7], s19 offen lds
	ds_read_b128 v[0:3], v219 offset:34048
	ds_read_b128 v[112:115], v219 offset:34560
	ds_read_b128 v[176:179], v219 offset:34624
	ds_read_b128 v[96:99], v219 offset:34080
	ds_read_b128 v[100:103], v219 offset:34112
	ds_read_b128 v[104:107], v219 offset:34144
	ds_read_b128 v[108:111], v219 offset:42368
	ds_read_b128 v[128:131], v219 offset:42400
	ds_read_b128 v[132:135], v219 offset:42432
	ds_read_b128 v[136:139], v219 offset:42464
	ds_read_b128 v[140:143], v219 offset:34592
	ds_read_b128 v[182:185], v219 offset:34656
	ds_read_b128 v[186:189], v219 offset:42880
	ds_read_b128 v[190:193], v219 offset:42912
	ds_read_b128 v[194:197], v219 offset:42944
	ds_read_b128 v[220:223], v219 offset:42976
	s_waitcnt vmcnt(2) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[144:147], 0
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[112:115], v[144:147], 0
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[148:151], v[0:15]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[140:143], v[148:151], v[112:127]
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[100:103], v[152:155], v[0:15]
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[176:179], v[152:155], v[112:127]
	v_exp_f32_e32 v31, v31
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[104:107], v[156:159], v[0:15]
	v_add_f32_e32 v96, v198, v181
	v_add_f32_e32 v96, v96, v199
	v_add_f32_e32 v96, v96, v205
	v_add_f32_e32 v96, v96, v206
	v_add_f32_e32 v96, v96, v207
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[182:185], v[156:159], v[112:127]
	v_add_f32_e32 v96, v96, v213
	v_add_f32_e32 v96, v96, v214
	v_add_f32_e32 v96, v96, v215
	v_add_f32_e32 v96, v96, v224
	v_add_f32_e32 v96, v96, v225
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[108:111], v[160:163], v[0:15]
	v_add_f32_e32 v96, v96, v226
	v_add_f32_e32 v96, v96, v227
	v_add_f32_e32 v96, v96, v228
	v_add_f32_e32 v96, v96, v229
	v_add_f32_e32 v96, v96, v230
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[186:189], v[160:163], v[112:127]
	v_add_f32_e32 v96, v96, v16
	v_add_f32_e32 v96, v96, v17
	v_add_f32_e32 v96, v96, v18
	v_add_f32_e32 v96, v96, v19
	v_add_f32_e32 v96, v96, v20
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[164:167], v[0:15]
	v_add_f32_e32 v96, v96, v21
	v_add_f32_e32 v96, v96, v22
	v_add_f32_e32 v96, v96, v23
	v_add_f32_e32 v96, v96, v24
	v_add_f32_e32 v96, v96, v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[190:193], v[164:167], v[112:127]
	v_add_f32_e32 v96, v96, v26
	v_add_f32_e32 v96, v96, v27
	v_add_f32_e32 v96, v96, v28
	v_add_f32_e32 v96, v96, v29
	v_add_f32_e32 v96, v96, v30
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[168:171], v[0:15]
	v_add_f32_e32 v176, v96, v31
	v_mov_b32_e32 v177, v176
	s_nop 1
	v_permlane32_swap_b32_e64 v176, v177 bound_ctrl:1
	v_cvt_pk_bf16_f32 v111, v30, v31
	v_cvt_pk_bf16_f32 v110, v28, v29
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[194:197], v[168:171], v[112:127]
	v_cvt_pk_bf16_f32 v109, v26, v27
	v_cvt_pk_bf16_f32 v108, v24, v25
	v_cvt_pk_bf16_f32 v107, v22, v23
	v_cvt_pk_bf16_f32 v106, v20, v21
	v_cvt_pk_bf16_f32 v105, v18, v19
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[172:175], v[0:15]
	v_cvt_pk_bf16_f32 v104, v16, v17
	v_cvt_pk_bf16_f32 v103, v229, v230
	v_cvt_pk_bf16_f32 v102, v227, v228
	v_cvt_pk_bf16_f32 v101, v225, v226
	v_cvt_pk_bf16_f32 v100, v215, v224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[172:175], v[112:127]
	v_cvt_pk_bf16_f32 v99, v213, v214
	v_cvt_pk_bf16_f32 v98, v206, v207
	v_cvt_pk_bf16_f32 v97, v199, v205
	v_cvt_pk_bf16_f32 v96, v181, v198
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[24:25], v218 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[26:27], v218 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[16:17], v218 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[18:19], v218 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v218 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v218 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v218 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v218 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[28:29], v218 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v218 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[20:21], v218 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[22:23], v218 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v218 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v218 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v218 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v218 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v218 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v218 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v218 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v218 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v218 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v218 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v218 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v218 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v218 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v218 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v218 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v218 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v218 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v218 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v218 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v218 offset:0x25c0

	;;#ASMEND
	s_waitcnt vmcnt(0) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[24:27], v[96:99], v[80:95]
	v_max3_f32 v24, v0, s2, v1
	v_max3_f32 v24, v24, v2, v3
	v_max3_f32 v24, v24, v4, v5
	v_max3_f32 v24, v24, v6, v7
	v_max3_f32 v24, v24, v8, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[28:31], v[96:99], v[64:79]
	v_max3_f32 v24, v24, v10, v11
	v_max3_f32 v24, v24, v12, v13
	v_max3_f32 v24, v24, v14, v15
	v_max3_f32 v24, v24, v112, v113
	v_max3_f32 v24, v24, v114, v115
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[96:99], v[48:63]
	v_max3_f32 v24, v24, v116, v117
	v_max3_f32 v24, v24, v118, v119
	v_max3_f32 v24, v24, v120, v121
	v_max3_f32 v24, v24, v122, v123
	v_max3_f32 v24, v24, v124, v125
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[172:175], v[96:99], v[32:47]
	v_max3_f32 v24, v24, v126, v127
	v_mov_b32_e32 v25, v24
	s_nop 1
	v_permlane32_swap_b32_e64 v24, v25 bound_ctrl:1
	v_max3_f32 v96, v204, v24, v25
	v_sub_f32_e32 v97, v204, v96
	v_sub_f32_e32 v1, v1, v96
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[100:103], v[80:95]
	v_sub_f32_e32 v31, v127, v96
	v_sub_f32_e32 v30, v126, v96
	v_sub_f32_e32 v29, v125, v96
	v_sub_f32_e32 v28, v124, v96
	v_sub_f32_e32 v27, v123, v96
	v_sub_f32_e32 v0, v0, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[20:23], v[100:103], v[64:79]
	v_sub_f32_e32 v26, v122, v96
	v_sub_f32_e32 v25, v121, v96
	v_sub_f32_e32 v24, v120, v96
	v_sub_f32_e32 v23, v119, v96
	v_sub_f32_e32 v22, v118, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[160:163], v[100:103], v[48:63]
	v_sub_f32_e32 v21, v117, v96
	v_sub_f32_e32 v20, v116, v96
	v_sub_f32_e32 v19, v115, v96
	v_sub_f32_e32 v18, v114, v96
	v_sub_f32_e32 v17, v113, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[100:103], v[32:47]
	v_sub_f32_e32 v16, v112, v96
	v_sub_f32_e32 v15, v15, v96
	v_sub_f32_e32 v14, v14, v96
	v_sub_f32_e32 v13, v13, v96
	v_sub_f32_e32 v12, v12, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[104:107], v[80:95]
	v_sub_f32_e32 v11, v11, v96
	v_sub_f32_e32 v10, v10, v96
	v_sub_f32_e32 v9, v9, v96
	v_sub_f32_e32 v8, v8, v96
	v_sub_f32_e32 v7, v7, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[104:107], v[64:79]
	v_sub_f32_e32 v6, v6, v96
	v_sub_f32_e32 v5, v5, v96
	v_sub_f32_e32 v4, v4, v96
	v_sub_f32_e32 v3, v3, v96
	v_sub_f32_e32 v2, v2, v96
	;;#ASMSTART
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[104:107], v[48:63]
	v_exp_f32_e32 v112, v97
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[144:147], v[104:107], v[32:47]
	v_exp_f32_e32 v2, v2
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[108:111], v[80:95]
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[132:135], v[108:111], v[64:79]
	v_exp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[136:139], v[108:111], v[48:63]
	v_exp_f32_e32 v11, v11
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[140:143], v[108:111], v[32:47]
	v_exp_f32_e32 v14, v14
	v_exp_f32_e32 v15, v15
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	; sched_barrier mask(0x00000000)
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
	; sched_barrier mask(0x00000000)
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
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:65], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[66:67], v203 offset:0x80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[68:69], v203 offset:0x100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[70:71], v203 offset:0x180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[72:73], v203 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[74:75], v203 offset:0x280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[76:77], v203 offset:0x300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[78:79], v203 offset:0x380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[80:81], v203 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[82:83], v203 offset:0xc0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[84:85], v203 offset:0x140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[86:87], v203 offset:0x1c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[88:89], v203 offset:0x240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[90:91], v203 offset:0x2c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[92:93], v203 offset:0x340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v203 offset:0x3c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v203 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v203 offset:0x2280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v203 offset:0x2300

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v203 offset:0x2380

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v203 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v203 offset:0x2480

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v203 offset:0x2500

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v203 offset:0x2580

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v203 offset:0x2240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v203 offset:0x22c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v203 offset:0x2340

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v203 offset:0x23c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v203 offset:0x2440

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v203 offset:0x24c0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v203 offset:0x2540

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v203 offset:0x25c0

	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[64:67], v[96:99], v[0:15]
	s_mov_b64 vcc, s[0:1]
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
	s_cbranch_vccz .LBB0_17
; %bb.16:
	s_barrier
.LBB0_17:
	v_add_f32_e32 v64, v212, v217
	v_add_f32_e32 v64, v64, v216
	v_fmac_f32_e32 v202, v200, v64
	v_add_f32_e32 v64, v202, v201
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
	v_lshl_add_u32 v64, v208, 2, v209
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
	.amdhsa_kernel _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs
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
		.amdhsa_next_free_vgpr 232
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 232
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
	.section	.text._Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs,"axG",@progbits,_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs,comdat
.Lfunc_end0:
	.size	_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs, .Lfunc_end0-_Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs
                                        ; -- End function
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.num_vgpr, 232
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.num_agpr, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.numbered_sgpr, 38
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.private_seg_size, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.uses_vcc, 1
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.uses_flat_scratch, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.has_dyn_sized_stack, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.has_recursion, 0
	.set _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14752
; TotalNumSgprs: 44
; NumVgprs: 232
; NumAgprs: 0
; TotalNumVgprs: 232
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 68096 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 232
; AccumOffset: 232
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 57
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_42d2ad477525d2a0,@object ; @__hip_cuid_42d2ad477525d2a0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_42d2ad477525d2a0
__hip_cuid_42d2ad477525d2a0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_42d2ad477525d2a0, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_42d2ad477525d2a0
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
    .name:           _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         _Z15gqa_d128_kernelI15opus_gqa_traitsILi32ELi64ELi128ELi8ELb0EEEv14opus_gqa_kargs.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     232
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
