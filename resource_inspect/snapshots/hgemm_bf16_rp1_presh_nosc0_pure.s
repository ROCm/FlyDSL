	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
	.p2align	8
	.type	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0,@function
hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0:
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
	.file	1 "/root/FlyDSL/kernels" "tensor_shim.py"
	s_load_dword s12, s[0:1], 0x60	;.loc	1 289 24 prologue_end
	s_load_dwordx2 s[4:5], s[0:1], 0x18
	s_load_dwordx2 s[8:9], s[0:1], 0x30
	.file	2 "/root/FlyDSL/kernels" "hgemm_splitk.py"
	s_ashr_i32 s14, s2, 31	;.loc	2 399 0
	s_lshr_b32 s14, s14, 27
	s_add_i32 s14, s2, s14
	s_waitcnt lgkmcnt(0)	;.loc	1 289 24
	s_ashr_i32 s13, s12, 31
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s9, s9, 0xffff
	s_ashr_i32 s18, s14, 5	;.loc	2 399 0
	s_and_b32 s19, s14, 0xffffffe0
	s_cmp_lg_u32 s2, s19
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b64 s[16:17], -1, 0
	s_and_b64 s[14:15], s[16:17], s[14:15]
	v_lshrrev_b32_e32 v002, 6, v000	;.loc	2 392 0
	s_subb_u32 s14, s18, 0	;.loc	2 399 0
	s_sub_i32 s15, s2, s19
	s_lshl_b32 s20, s14, 8	;.loc	2 442 0
	v_readfirstlane_b32 s14, v002	;.loc	2 606 0
	s_lshl_b32 s18, s15, 8	;.loc	2 443 0
	s_lshl_b32 s21, s14, 10	;.loc	2 606 0
	s_lshl_b32 s2, s3, 13	;.loc	2 440 0
	.file	3 "/root/FlyDSL/python/flydsl/expr" "numeric.py"
	s_ashr_i32 s19, s20, 31	;.loc	3 875 16
	s_ashr_i32 s16, s18, 31
	s_add_i32 s25, s21, 0x10000	;.loc	2 641 0
	v_lshrrev_b32_e32 v109, 2, v000	;.loc	2 685 0
	s_cmpk_lt_u32 s18, 0x2000	;.loc	2 704 12
	v_xor_b32_e32 v002, v109, v000	;.loc	2 689 0
	s_cselect_b64 vcc, -1, 0	;.loc	2 704 12
	v_lshlrev_b32_e32 v002, 3, v002	;.loc	2 689 0
	s_and_b64 s[14:15], vcc, exec	;.loc	2 703 23
	v_and_b32_e32 v010, 24, v002	;.loc	2 689 0
	v_or_b32_e32 v002, s18, v109	;.loc	2 702 0
	s_cselect_b32 s14, s16, 0	;.loc	2 703 23
	v_cndmask_b32_e32 v110, 0, v002, vcc
	v_mov_b32_e32 v111, s14
	v_lshlrev_b64 v002 v003, 7, v110 v111	;.loc	2 371 0
	v_and_b32_e32 v107, 63, v110	;.loc	2 1225 0
	v_and_b32_e32 v002, 0xffff8000, v002	;.loc	2 371 0
	v_and_b32_e32 v003, 0x3ffffff, v003
	v_or_b32_e32 v002, v002, v107
	v_lshlrev_b64 v002 v003, 6, v002 v003
	v_or_b32_e32 v094, v002, v010
	s_lshl_b32 s26, s3, 21	;.loc	2 372 0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_add_lshl_u32 v003, v094, s26, 1	;.loc	2 712 24
	v_or_b32_e32 v011, 64, v109	;.loc	2 687 0
	s_mov_b32 s10, s6	;.loc	1 289 24
	s_mov_b32 s11, s7
	s_mov_b32 m0, s25	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen lds
	v_or_b32_e32 v003, s18, v011	;.loc	2 702 0
	v_cndmask_b32_e32 v112, 0, v003, vcc	;.loc	2 703 23
	v_mov_b32_e32 v113, s14
	v_lshlrev_b64 v004 v005, 7, v112 v113	;.loc	2 371 0
	v_and_b32_e32 v003, 0x7f, v112	;.loc	2 369 0
	v_and_b32_e32 v004, 0xffff8000, v004	;.loc	2 371 0
	v_and_b32_e32 v005, 0x3ffffff, v005
	v_or_b32_e32 v004, v004, v003
	v_lshlrev_b64 v004 v005, 6, v004 v005
	v_or_b32_e32 v096, v004, v010
	v_add_lshl_u32 v003, v096, s26, 1	;.loc	2 712 24
	v_or_b32_e32 v012, 0x80, v109	;.loc	2 687 0
	s_add_i32 s24, s21, 0x11000	;.loc	2 641 0
	s_mov_b32 m0, s24	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen lds
	v_or_b32_e32 v003, s18, v012	;.loc	2 702 0
	v_cndmask_b32_e32 v114, 0, v003, vcc	;.loc	2 703 23
	v_mov_b32_e32 v115, s14
	v_lshlrev_b64 v006 v007, 7, v114 v115	;.loc	2 371 0
	v_and_b32_e32 v003, 0xbf, v114	;.loc	2 369 0
	v_and_b32_e32 v005, 0xffff8000, v006	;.loc	2 371 0
	v_and_b32_e32 v007, 0x3ffffff, v007
	v_or_b32_e32 v006, v005, v003
	v_lshlrev_b64 v006 v007, 6, v006 v007
	v_or_b32_e32 v098, v006, v010
	v_add_lshl_u32 v003, v098, s26, 1	;.loc	2 712 24
	v_or_b32_e32 v013, 0xc0, v109	;.loc	2 687 0
	s_add_i32 s23, s21, 0x12000	;.loc	2 641 0
	s_mov_b32 m0, s23	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen lds
	v_or_b32_e32 v003, s18, v013	;.loc	2 702 0
	v_cndmask_b32_e32 v116, 0, v003, vcc	;.loc	2 703 23
	v_mov_b32_e32 v117, s14
	v_lshlrev_b64 v008 v009, 7, v116 v117	;.loc	2 371 0
	v_and_b32_e32 v111, 0xff, v116	;.loc	2 1225 0
	v_and_b32_e32 v003, 0xffff8000, v008	;.loc	2 371 0
	v_and_b32_e32 v009, 0x3ffffff, v009
	v_or_b32_e32 v008, v003, v111
	v_or_b32_e32 v014, 32, v010	;.loc	2 690 0
	v_lshlrev_b64 v008 v009, 6, v008 v009	;.loc	2 371 0
	v_or_b32_e32 v046, v002, v014
	v_or_b32_e32 v100, v008, v010
	v_add_lshl_u32 v002, v046, s26, 1	;.loc	2 712 24
	v_or_b32_e32 v047, v004, v014	;.loc	2 371 0
	s_add_i32 s17, s21, 0x13000	;.loc	2 641 0
	v_add_lshl_u32 v003, v100, s26, 1	;.loc	2 712 24
	s_mov_b32 m0, s17	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen lds
	s_add_i32 s14, s21, 0x14000	;.loc	2 641 0
	s_mov_b32 m0, s14	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v047, s26, 1	;.loc	2 712 24
	v_or_b32_e32 v058, v006, v014	;.loc	2 371 0
	s_add_i32 s14, s21, 0x15000	;.loc	2 641 0
	s_mov_b32 m0, s14	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v058, s26, 1	;.loc	2 712 24
	v_or_b32_e32 v062, v008, v014	;.loc	2 371 0
	s_add_i32 s14, s21, 0x16000	;.loc	2 641 0
	s_mov_b32 m0, s14	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v062, s26, 1	;.loc	2 712 24
	s_add_i32 s14, s21, 0x17000	;.loc	2 641 0
	s_mov_b32 m0, s14	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_or_b32_e32 v002, s20, v109	;.loc	2 664 0
	v_mov_b32_e32 v003, s19
	v_mov_b32_e32 v015, s19	;.loc	2 665 23
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 666 12
	v_or_b32_e32 v004, s20, v011	;.loc	2 664 0
	v_mov_b32_e32 v005, s19
	v_cndmask_b32_e32 v119, 0, v015, vcc	;.loc	2 665 23
	v_cndmask_b32_e32 v118, 0, v002, vcc
	v_lshlrev_b64 v002 v003, 7, v118 v119	;.loc	2 371 0
	v_and_b32_e32 v113, 63, v118	;.loc	2 1225 0
	v_and_b32_e32 v002, 0xffff8000, v002	;.loc	2 371 0
	v_and_b32_e32 v003, 0x3ffffff, v003
	v_or_b32_e32 v002, v002, v113
	v_lshlrev_b64 v002 v003, 6, v002 v003
	v_cmp_gt_u64_e32 vcc, s[12:13], v004 v005	;.loc	2 666 12
	v_or_b32_e32 v102, v002, v010	;.loc	2 371 0
	v_add_lshl_u32 v003, v102, s26, 1	;.loc	2 674 24
	v_cndmask_b32_e32 v121, 0, v015, vcc	;.loc	2 665 23
	v_cndmask_b32_e32 v120, 0, v004, vcc
	v_lshlrev_b64 v004 v005, 7, v120 v121	;.loc	2 371 0
	s_mov_b32 m0, s21	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen lds
	v_and_b32_e32 v003, 0x7f, v120	;.loc	2 369 0
	v_and_b32_e32 v004, 0xffff8000, v004	;.loc	2 371 0
	v_and_b32_e32 v005, 0x3ffffff, v005
	v_or_b32_e32 v004, v004, v003
	v_or_b32_e32 v006, s20, v012	;.loc	2 664 0
	v_mov_b32_e32 v007, s19
	v_lshlrev_b64 v004 v005, 6, v004 v005	;.loc	2 371 0
	v_cmp_gt_u64_e32 vcc, s[12:13], v006 v007	;.loc	2 666 12
	v_or_b32_e32 v104, v004, v010	;.loc	2 371 0
	v_add_lshl_u32 v003, v104, s26, 1	;.loc	2 674 24
	v_cndmask_b32_e32 v123, 0, v015, vcc	;.loc	2 665 23
	v_cndmask_b32_e32 v122, 0, v006, vcc
	v_lshlrev_b64 v006 v007, 7, v122 v123	;.loc	2 371 0
	s_add_i32 s16, s21, 0x1000	;.loc	2 641 0
	s_mov_b32 m0, s16	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen lds
	v_and_b32_e32 v003, 0xbf, v122	;.loc	2 369 0
	v_and_b32_e32 v005, 0xffff8000, v006	;.loc	2 371 0
	v_and_b32_e32 v007, 0x3ffffff, v007
	v_or_b32_e32 v006, v005, v003
	v_or_b32_e32 v008, s20, v013	;.loc	2 664 0
	v_mov_b32_e32 v009, s19
	v_lshlrev_b64 v006 v007, 6, v006 v007	;.loc	2 371 0
	v_cmp_gt_u64_e32 vcc, s[12:13], v008 v009	;.loc	2 666 12
	v_or_b32_e32 v106, v006, v010	;.loc	2 371 0
	v_add_lshl_u32 v003, v106, s26, 1	;.loc	2 674 24
	v_cndmask_b32_e32 v125, 0, v015, vcc	;.loc	2 665 23
	v_cndmask_b32_e32 v124, 0, v008, vcc
	v_lshlrev_b64 v008 v009, 7, v124 v125	;.loc	2 371 0
	s_add_i32 s15, s21, 0x2000	;.loc	2 641 0
	s_mov_b32 m0, s15	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen lds
	v_and_b32_e32 v117, 0xff, v124	;.loc	2 1225 0
	v_and_b32_e32 v003, 0xffff8000, v008	;.loc	2 371 0
	v_and_b32_e32 v009, 0x3ffffff, v009
	v_or_b32_e32 v008, v003, v117
	v_lshlrev_b64 v008 v009, 6, v008 v009
	v_or_b32_e32 v070, v002, v014
	v_or_b32_e32 v108, v008, v010
	v_add_lshl_u32 v002, v070, s26, 1	;.loc	2 674 24
	v_or_b32_e32 v078, v004, v014	;.loc	2 371 0
	s_add_i32 s14, s21, 0x3000	;.loc	2 641 0
	v_add_lshl_u32 v003, v108, s26, 1	;.loc	2 674 24
	s_mov_b32 m0, s14	;.loc	2 625 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen lds
	s_add_i32 s28, s21, 0x4000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v078, s26, 1	;.loc	2 674 24
	v_or_b32_e32 v086, v006, v014	;.loc	2 371 0
	s_add_i32 s28, s21, 0x5000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v086, s26, 1	;.loc	2 674 24
	v_or_b32_e32 v119, v008, v014	;.loc	2 371 0
	s_add_i32 s28, s21, 0x6000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v119, s26, 1	;.loc	2 674 24
	s_or_b32 s29, s26, 0x4000	;.loc	2 372 0
	s_add_i32 s28, s21, 0x7000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v094, s29, 1	;.loc	2 712 24
	s_add_i32 s28, s21, 0x18000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v096, s29, 1	;.loc	2 712 24
	s_add_i32 s28, s21, 0x19000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v098, s29, 1	;.loc	2 712 24
	s_add_i32 s28, s21, 0x1a000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v100, s29, 1	;.loc	2 712 24
	s_add_i32 s28, s21, 0x1b000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	v_add_lshl_u32 v002, v102, s29, 1	;.loc	2 674 24
	s_add_i32 s28, s21, 0x8000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v104, s29, 1	;.loc	2 674 24
	s_add_i32 s28, s21, 0x9000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v106, s29, 1	;.loc	2 674 24
	v_and_b32_e32 v001, 0x80, v000	;.loc	2 446 0
	v_lshlrev_b32_e32 v034, 1, v000	;.loc	2 447 0
	v_and_b32_e32 v035, 15, v000	;.loc	2 448 0
	v_lshrrev_b32_e32 v095, 4, v000	;.loc	2 449 0
	s_mov_b32 s22, 0x10000
	s_add_i32 s28, s21, 0xa000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_add_lshl_u32 v002, v108, s29, 1	;.loc	2 674 24
	s_movk_i32 s27, 0x80
	s_add_i32 s28, s21, 0xb000	;.loc	2 641 0
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen lds
	v_xor_b32_e32 v002, v095, v000	;.loc	2 761 0
	v_lshlrev_b32_e32 v002, 4, v002
	v_and_b32_e32 v099, 48, v002
	v_lshlrev_b32_e32 v002, 5, v000	;.loc	2 762 0
	v_and_b32_e32 v002, 0x11e0, v002
	v_lshlrev_b32_e32 v105, 1, v002	;.loc	1 317 12
	v_or3_b32 v006, v001, v035, 16	;.loc	2 756 0
	v_and_or_b32 v097, v034, s27, v035	;.loc	2 773 0
	v_mov_b32_e32 v034, 0x10000	;.loc	1 317 12
	v_or_b32_e32 v121, v105, v099
	v_lshlrev_b32_e32 v103, 6, v006
	v_lshl_or_b32 v101, v097, 6, v034
	s_waitcnt vmcnt(0)	;.loc	2 462 0
	s_barrier
	ds_read_b128 v002 v003 v004 v005, v121	;.loc	1 317 12
	v_or_b32_e32 v123, v103, v099
	v_or_b32_e32 v125, v101, v099
	v_lshlrev_b32_e32 v115, 5, v006	;.loc	2 762 0
	ds_read_b128 v006 v007 v008 v009, v123	;.loc	1 317 12
	ds_read_b128 v010 v011 v012 v013, v123 offset:1024
	ds_read_b128 v014 v015 v016 v017, v123 offset:2048
	ds_read_b128 v018 v019 v020 v021, v123 offset:3072
	ds_read_b128 v022 v023 v024 v025, v123 offset:4096
	ds_read_b128 v026 v027 v028 v029, v123 offset:5120
	ds_read_b128 v030 v031 v032 v033, v123 offset:6144
	ds_read_b128 v034 v035 v036 v037, v125
	ds_read_b128 v038 v039 v040 v041, v125 offset:1024
	ds_read_b128 v042 v043 v044 v045, v125 offset:2048
	ds_read_b128 v050 v051 v052 v053, v125 offset:3072
	ds_read_b128 v054 v055 v056 v057, v125 offset:4096
	ds_read_b128 v126 v127 v128 v129, v125 offset:5120
	ds_read_b128 v130 v131 v132 v133, v125 offset:6144
	ds_read_b128 v134 v135 v136 v137, v125 offset:7168
	s_waitcnt lgkmcnt(7)	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v002 v003 v004 v005, v034 v035 v036 v037, 0
	ds_read_b128 v138 v139 v140 v141, v121 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v002 v003 v004 v005, v038 v039 v040 v041, 0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v002 v003 v004 v005, v042 v043 v044 v045, 0
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v002 v003 v004 v005, v050 v051 v052 v053, 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v002 v003 v004 v005, v054 v055 v056 v057, 0
	ds_read_b128 v142 v143 v144 v145, v123 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v002 v003 v004 v005, v126 v127 v128 v129, 0
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v002 v003 v004 v005, v130 v131 v132 v133, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v002 v003 v004 v005, v134 v135 v136 v137, 0
	v_add_lshl_u32 v002, v046, s29, 1	;.loc	2 712 24
	v_mfma_f32_16x16x32_bf16 a[220:223], v006 v007 v008 v009, v034 v035 v036 v037, 0	;.loc	2 1054 0
	s_add_i32 s27, s21, 0x1c000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	ds_read_b128 v002 v003 v004 v005, v123 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v006 v007 v008 v009, v038 v039 v040 v041, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v006 v007 v008 v009, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[208:211], v006 v007 v008 v009, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[204:207], v006 v007 v008 v009, v054 v055 v056 v057, 0
	ds_read_b128 v146 v147 v148 v149, v123 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v006 v007 v008 v009, v126 v127 v128 v129, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v006 v007 v008 v009, v130 v131 v132 v133, 0
	v_mfma_f32_16x16x32_bf16 a[192:195], v006 v007 v008 v009, v134 v135 v136 v137, 0
	v_add_lshl_u32 v006, v047, s29, 1	;.loc	2 712 24
	s_add_i32 s27, s21, 0x1d000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen lds
	v_add_lshl_u32 v006, v058, s29, 1	;.loc	2 712 24
	v_mfma_f32_16x16x32_bf16 a[188:191], v010 v011 v012 v013, v034 v035 v036 v037, 0	;.loc	2 1054 0
	ds_read_b128 v150 v151 v152 v153, v123 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v010 v011 v012 v013, v038 v039 v040 v041, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v010 v011 v012 v013, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[176:179], v010 v011 v012 v013, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[172:175], v010 v011 v012 v013, v054 v055 v056 v057, 0
	ds_read_b128 v046 v047 v048 v049, v123 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v010 v011 v012 v013, v126 v127 v128 v129, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v010 v011 v012 v013, v130 v131 v132 v133, 0
	s_add_i32 s27, s21, 0x1e000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen lds
	v_add_lshl_u32 v006, v062, s29, 1	;.loc	2 712 24
	v_mfma_f32_16x16x32_bf16 a[160:163], v010 v011 v012 v013, v134 v135 v136 v137, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[156:159], v014 v015 v016 v017, v034 v035 v036 v037, 0
	ds_read_b128 v058 v059 v060 v061, v123 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v014 v015 v016 v017, v038 v039 v040 v041, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v014 v015 v016 v017, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[144:147], v014 v015 v016 v017, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[140:143], v014 v015 v016 v017, v054 v055 v056 v057, 0
	ds_read_b128 v066 v067 v068 v069, v123 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v014 v015 v016 v017, v126 v127 v128 v129, 0	;.loc	2 1054 0
	s_add_i32 s27, s21, 0x1f000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen lds
	v_add_lshl_u32 v006, v070, s29, 1	;.loc	2 674 24
	v_mfma_f32_16x16x32_bf16 a[132:135], v014 v015 v016 v017, v130 v131 v132 v133, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[128:131], v014 v015 v016 v017, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[124:127], v018 v019 v020 v021, v034 v035 v036 v037, 0
	ds_read_b128 v062 v063 v064 v065, v125 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v018 v019 v020 v021, v038 v039 v040 v041, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v018 v019 v020 v021, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[112:115], v018 v019 v020 v021, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[108:111], v018 v019 v020 v021, v054 v055 v056 v057, 0
	s_add_i32 s27, s21, 0xc000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen lds
	v_add_lshl_u32 v006, v078, s29, 1	;.loc	2 674 24
	ds_read_b128 v070 v071 v072 v073, v125 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v018 v019 v020 v021, v126 v127 v128 v129, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v018 v019 v020 v021, v130 v131 v132 v133, 0
	v_mfma_f32_16x16x32_bf16 a[96:99], v018 v019 v020 v021, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[92:95], v022 v023 v024 v025, v034 v035 v036 v037, 0
	ds_read_b128 v074 v075 v076 v077, v125 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v022 v023 v024 v025, v038 v039 v040 v041, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v022 v023 v024 v025, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[80:83], v022 v023 v024 v025, v050 v051 v052 v053, 0
	s_add_i32 s27, s21, 0xd000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen lds
	v_add_lshl_u32 v006, v086, s29, 1	;.loc	2 674 24
	v_mfma_f32_16x16x32_bf16 a[76:79], v022 v023 v024 v025, v054 v055 v056 v057, 0	;.loc	2 1054 0
	ds_read_b128 v078 v079 v080 v081, v125 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v022 v023 v024 v025, v126 v127 v128 v129, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v022 v023 v024 v025, v130 v131 v132 v133, 0
	v_mfma_f32_16x16x32_bf16 a[64:67], v022 v023 v024 v025, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[60:63], v026 v027 v028 v029, v034 v035 v036 v037, 0
	ds_read_b128 v082 v083 v084 v085, v125 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v026 v027 v028 v029, v038 v039 v040 v041, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v026 v027 v028 v029, v042 v043 v044 v045, 0
	s_add_i32 s27, s21, 0xe000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen lds
	v_add_lshl_u32 v006, v119, s29, 1	;.loc	2 674 24
	s_bitset1_b32 s26, 15	;.loc	2 372 0
	v_mfma_f32_16x16x32_bf16 a[48:51], v026 v027 v028 v029, v050 v051 v052 v053, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[44:47], v026 v027 v028 v029, v054 v055 v056 v057, 0
	ds_read_b128 v086 v087 v088 v089, v125 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v026 v027 v028 v029, v126 v127 v128 v129, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v026 v027 v028 v029, v130 v131 v132 v133, 0
	v_mfma_f32_16x16x32_bf16 a[32:35], v026 v027 v028 v029, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[28:31], v030 v031 v032 v033, v034 v035 v036 v037, 0
	ds_read_b128 v090 v091 v092 v093, v125 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v030 v031 v032 v033, v038 v039 v040 v041, 0	;.loc	2 1054 0
	s_add_i32 s27, s21, 0xf000	;.loc	2 641 0
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen lds
	v_add_lshl_u32 v006, v094, s26, 1	;.loc	2 712 24
	v_mfma_f32_16x16x32_bf16 a[20:23], v030 v031 v032 v033, v042 v043 v044 v045, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[16:19], v030 v031 v032 v033, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[12:15], v030 v031 v032 v033, v054 v055 v056 v057, 0
	ds_read_b128 v154 v155 v156 v157, v125 offset:23552	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v030 v031 v032 v033, v126 v127 v128 v129, 0	;.loc	2 1054 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v030 v031 v032 v033, v130 v131 v132 v133, 0
	v_mfma_f32_16x16x32_bf16 a[0:3], v030 v031 v032 v033, v134 v135 v136 v137, 0
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 475 0
	s_barrier
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v138 v139 v140 v141, v062 v063 v064 v065, a[252:255]
	s_mov_b32 m0, s25	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen lds
	v_add_lshl_u32 v006, v096, s26, 1	;.loc	2 712 24
	ds_read_b128 v054 v055 v056 v057, v121 offset:32768	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v138 v139 v140 v141, v070 v071 v072 v073, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v138 v139 v140 v141, v074 v075 v076 v077, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v138 v139 v140 v141, v078 v079 v080 v081, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v138 v139 v140 v141, v082 v083 v084 v085, a[236:239]
	ds_read_b128 v042 v043 v044 v045, v123 offset:32768	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v138 v139 v140 v141, v086 v087 v088 v089, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v138 v139 v140 v141, v090 v091 v092 v093, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v138 v139 v140 v141, v154 v155 v156 v157, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v142 v143 v144 v145, v062 v063 v064 v065, a[220:223]
	s_mov_b32 m0, s24	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen lds
	ds_read_b128 v034 v035 v036 v037, v123 offset:33792	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v142 v143 v144 v145, v070 v071 v072 v073, a[216:219]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v142 v143 v144 v145, v074 v075 v076 v077, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v142 v143 v144 v145, v078 v079 v080 v081, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v142 v143 v144 v145, v082 v083 v084 v085, a[204:207]
	ds_read_b128 v026 v027 v028 v029, v123 offset:34816	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v142 v143 v144 v145, v086 v087 v088 v089, a[200:203]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v142 v143 v144 v145, v090 v091 v092 v093, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v142 v143 v144 v145, v154 v155 v156 v157, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v002 v003 v004 v005, v062 v063 v064 v065, a[188:191]
	v_add_lshl_u32 v006, v098, s26, 1	;.loc	2 712 24
	s_mov_b32 m0, s23	;.loc	2 625 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen lds
	ds_read_b128 v014 v015 v016 v017, v123 offset:35840	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v002 v003 v004 v005, v070 v071 v072 v073, a[184:187]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v002 v003 v004 v005, v074 v075 v076 v077, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v002 v003 v004 v005, v078 v079 v080 v081, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v002 v003 v004 v005, v082 v083 v084 v085, a[172:175]
	ds_read_b128 v010 v011 v012 v013, v123 offset:36864	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v002 v003 v004 v005, v086 v087 v088 v089, a[168:171]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v002 v003 v004 v005, v090 v091 v092 v093, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v002 v003 v004 v005, v154 v155 v156 v157, a[160:163]
	v_add_lshl_u32 v002, v100, s26, 1	;.loc	2 712 24
	v_add_lshl_u32 v018, v102, s26, 1	;.loc	2 674 24
	v_add_lshl_u32 v030, v104, s26, 1
	v_mfma_f32_16x16x32_bf16 a[156:159], v146 v147 v148 v149, v062 v063 v064 v065, a[156:159]	;.loc	2 999 0
	s_mov_b32 m0, s17	;.loc	2 625 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen lds
	ds_read_b128 v006 v007 v008 v009, v123 offset:37888	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v146 v147 v148 v149, v070 v071 v072 v073, a[152:155]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v146 v147 v148 v149, v074 v075 v076 v077, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v146 v147 v148 v149, v078 v079 v080 v081, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v146 v147 v148 v149, v082 v083 v084 v085, a[140:143]
	ds_read_b128 v002 v003 v004 v005, v123 offset:38912	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v146 v147 v148 v149, v086 v087 v088 v089, a[136:139]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v146 v147 v148 v149, v090 v091 v092 v093, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v146 v147 v148 v149, v154 v155 v156 v157, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v150 v151 v152 v153, v062 v063 v064 v065, a[124:127]
	s_mov_b32 m0, s21	;.loc	2 625 0
	buffer_load_dwordx4 v018, s[4:7], 0 offen lds
	ds_read_b128 v018 v019 v020 v021, v125 offset:32768	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v150 v151 v152 v153, v070 v071 v072 v073, a[120:123]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v150 v151 v152 v153, v074 v075 v076 v077, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v150 v151 v152 v153, v078 v079 v080 v081, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v150 v151 v152 v153, v082 v083 v084 v085, a[108:111]
	ds_read_b128 v022 v023 v024 v025, v125 offset:33792	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v150 v151 v152 v153, v086 v087 v088 v089, a[104:107]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v150 v151 v152 v153, v090 v091 v092 v093, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v150 v151 v152 v153, v154 v155 v156 v157, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v046 v047 v048 v049, v062 v063 v064 v065, a[92:95]
	s_mov_b32 m0, s16	;.loc	2 625 0
	buffer_load_dwordx4 v030, s[4:7], 0 offen lds
	ds_read_b128 v030 v031 v032 v033, v125 offset:34816	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v046 v047 v048 v049, v070 v071 v072 v073, a[88:91]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v046 v047 v048 v049, v074 v075 v076 v077, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v046 v047 v048 v049, v078 v079 v080 v081, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v046 v047 v048 v049, v082 v083 v084 v085, a[76:79]
	ds_read_b128 v038 v039 v040 v041, v125 offset:35840	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v046 v047 v048 v049, v086 v087 v088 v089, a[72:75]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v046 v047 v048 v049, v090 v091 v092 v093, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v046 v047 v048 v049, v154 v155 v156 v157, a[64:67]
	v_add_lshl_u32 v046, v106, s26, 1	;.loc	2 674 24
	v_mfma_f32_16x16x32_bf16 a[60:63], v058 v059 v060 v061, v062 v063 v064 v065, a[60:63]	;.loc	2 999 0
	s_mov_b32 m0, s15	;.loc	2 625 0
	buffer_load_dwordx4 v046, s[4:7], 0 offen lds
	ds_read_b128 v046 v047 v048 v049, v125 offset:36864	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v058 v059 v060 v061, v070 v071 v072 v073, a[56:59]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v058 v059 v060 v061, v074 v075 v076 v077, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v058 v059 v060 v061, v078 v079 v080 v081, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v058 v059 v060 v061, v082 v083 v084 v085, a[44:47]
	ds_read_b128 v050 v051 v052 v053, v125 offset:37888	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v058 v059 v060 v061, v086 v087 v088 v089, a[40:43]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v058 v059 v060 v061, v090 v091 v092 v093, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v058 v059 v060 v061, v154 v155 v156 v157, a[32:35]
	v_add_lshl_u32 v058, v108, s26, 1	;.loc	2 674 24
	v_mfma_f32_16x16x32_bf16 a[28:31], v066 v067 v068 v069, v062 v063 v064 v065, a[28:31]	;.loc	2 999 0
	s_mov_b32 m0, s14	;.loc	2 625 0
	buffer_load_dwordx4 v058, s[4:7], 0 offen lds
	ds_read_b128 v058 v059 v060 v061, v125 offset:38912	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v066 v067 v068 v069, v070 v071 v072 v073, a[24:27]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v066 v067 v068 v069, v074 v075 v076 v077, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v066 v067 v068 v069, v078 v079 v080 v081, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v066 v067 v068 v069, v082 v083 v084 v085, a[12:15]
	ds_read_b128 v062 v063 v064 v065, v125 offset:39936	;.loc	1 317 12
	v_lshlrev_b32_e32 v071, 5, v097	;.loc	2 779 0
	v_mfma_f32_16x16x32_bf16 a[8:11], v066 v067 v068 v069, v086 v087 v088 v089, a[8:11]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v066 v067 v068 v069, v090 v091 v092 v093, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v066 v067 v068 v069, v154 v155 v156 v157, a[0:3]
	v_lshlrev_b32_e32 v066, 14, v124	;.loc	2 1225 0
	s_lshl_b32 s3, s3, 22
	v_and_b32_e32 v066, 0xffc00000, v066
	v_bitop3_b32 v068, v109, 3, v000 bitop3:0x48
	v_add_u32_e32 v066, s3, v066
	v_lshlrev_b32_e32 v067, 7, v117
	v_lshlrev_b32_e32 v069, 4, v068
	v_or3_b32 v066, v066, v067, v069
	v_lshlrev_b32_e32 v067, 14, v122
	v_and_b32_e32 v067, 0xffc00000, v067
	v_mov_b32_e32 v073, 7
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_sdwa v068, v073, v122 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_or3_b32 v068, v067, v068, v069
	v_lshlrev_b32_e32 v067, 14, v120
	v_and_b32_e32 v067, 0xffc00000, v067
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_sdwa v070, v073, v120 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_or3_b32 v070, v067, v070, v069
	v_lshlrev_b32_e32 v067, 14, v118
	v_and_b32_e32 v067, 0xffc00000, v067
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_e32 v072, 7, v113
	v_or3_b32 v072, v067, v072, v069
	v_lshlrev_b32_e32 v067, 14, v116
	v_and_b32_e32 v067, 0xffc00000, v067
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_e32 v074, 7, v111
	v_or3_b32 v074, v067, v074, v069
	v_lshlrev_b32_e32 v067, 14, v114
	v_and_b32_e32 v067, 0xffc00000, v067
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_sdwa v075, v073, v114 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_or3_b32 v076, v067, v075, v069
	v_lshlrev_b32_e32 v067, 14, v112
	v_and_b32_e32 v067, 0xffc00000, v067
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_sdwa v073, v073, v112 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_or3_b32 v078, v067, v073, v069
	v_lshlrev_b32_e32 v067, 14, v110
	v_and_b32_e32 v067, 0xffc00000, v067
	s_or_b32 s23, s2, 0x1fc0
	v_add_u32_e32 v067, s3, v067
	v_lshlrev_b32_e32 v073, 7, v107
	s_or_b32 s24, s2, 0xc0
	s_mov_b32 s2, 0xffc10000
	v_or3_b32 v080, v067, v073, v069
	s_mov_b64 s[14:15], 1
	s_mov_b32 s3, -1
	v_lshlrev_b32_e32 v067, 1, v115
	v_lshlrev_b32_e32 v069, 1, v071
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 475 0
	s_barrier
.LBB0_1:
	s_mov_b64 s[16:17], s[14:15]	;.loc	2 0 0 is_stmt 0
	v_add_u32_e32 v082, s2, v068	;.loc	2 625 0 is_stmt 1
	s_xor_b32 s14, s16, 1	;.loc	2 1231 0
	s_lshl_b32 s16, s16, 15	;.loc	1 317 12
	s_min_i32 s15, s24, s23	;.loc	2 1237 31
	v_add_u32_e32 v083, s2, v066	;.loc	2 625 0
	v_add_u32_e32 v107, 0x400040, v082
	v_or_b32_e32 v082, s16, v099	;.loc	1 317 12
	s_lshl_b32 s25, s14, 15	;.loc	2 640 0
	v_add_u32_e32 v071, s2, v080	;.loc	2 625 0
	v_add_u32_e32 v073, s2, v078
	v_add_u32_e32 v075, s2, v076
	v_add_u32_e32 v077, s2, v074
	v_add_u32_e32 v079, s2, v072
	v_add_u32_e32 v081, s2, v070
	v_add_u32_e32 v109, 0x400040, v083
	s_lshl_b32 s17, s15, 8	;.loc	2 372 0
	s_add_i32 s15, s16, s21	;.loc	2 640 0
	v_add_u32_e32 v083, v082, v105	;.loc	1 317 12
	v_add_u32_e32 v126, v082, v067
	s_add_i32 s16, s21, s25	;.loc	2 640 0
	v_add3_u32 v158, v082, v069, s22	;.loc	1 317 12
	v_or_b32_e32 v090, s25, v099
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v054 v055 v056 v057, v018 v019 v020 v021, a[252:255]
	v_add_u32_e32 v071, 0x400040, v071	;.loc	2 625 0
	v_add_u32_e32 v073, 0x400040, v073
	v_add_u32_e32 v075, 0x400040, v075
	v_add_u32_e32 v077, 0x400040, v077
	v_add_u32_e32 v079, 0x400040, v079
	v_add_u32_e32 v081, 0x400040, v081
	ds_read_b128 v082 v083 v084 v085, v083 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v054 v055 v056 v057, v022 v023 v024 v025, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v054 v055 v056 v057, v030 v031 v032 v033, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v054 v055 v056 v057, v038 v039 v040 v041, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v054 v055 v056 v057, v046 v047 v048 v049, a[236:239]
	ds_read_b128 v086 v087 v088 v089, v126 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v054 v055 v056 v057, v050 v051 v052 v053, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v054 v055 v056 v057, v058 v059 v060 v061, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v054 v055 v056 v057, v062 v063 v064 v065, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v042 v043 v044 v045, v018 v019 v020 v021, a[220:223]
	s_add_i32 s31, s16, 0x4000	;.loc	2 640 0
	s_add_i32 s33, s16, 0x14000	;.loc	2 641 0
	s_add_i32 s34, s16, 0x5000	;.loc	2 640 0
	s_add_i32 s35, s16, 0x15000	;.loc	2 641 0
	s_add_i32 s36, s16, 0x6000	;.loc	2 640 0
	s_add_i32 s37, s16, 0x16000	;.loc	2 641 0
	s_add_i32 s38, s16, 0x7000	;.loc	2 640 0
	s_add_i32 s16, s16, 0x17000	;.loc	2 641 0
	v_add_u32_e32 v054, v090, v105	;.loc	1 317 12
	v_add_u32_e32 v170, v090, v067
	v_add3_u32 v171, v090, v069, s22
	s_mov_b32 m0, s33	;.loc	2 625 0
	buffer_load_dwordx4 v071, s[8:11], 0 offen lds
	ds_read_b128 v090 v091 v092 v093, v126 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v042 v043 v044 v045, v022 v023 v024 v025, a[216:219]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v042 v043 v044 v045, v030 v031 v032 v033, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v042 v043 v044 v045, v038 v039 v040 v041, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v042 v043 v044 v045, v046 v047 v048 v049, a[204:207]
	ds_read_b128 v110 v111 v112 v113, v126 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v042 v043 v044 v045, v050 v051 v052 v053, a[200:203]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v042 v043 v044 v045, v058 v059 v060 v061, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v042 v043 v044 v045, v062 v063 v064 v065, a[192:195]
	s_mov_b32 m0, s35	;.loc	2 625 0
	buffer_load_dwordx4 v073, s[8:11], 0 offen lds
	v_mfma_f32_16x16x32_bf16 a[188:191], v034 v035 v036 v037, v018 v019 v020 v021, a[188:191]	;.loc	2 999 0
	ds_read_b128 v114 v115 v116 v117, v126 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v034 v035 v036 v037, v022 v023 v024 v025, a[184:187]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v034 v035 v036 v037, v030 v031 v032 v033, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v034 v035 v036 v037, v038 v039 v040 v041, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v034 v035 v036 v037, v046 v047 v048 v049, a[172:175]
	ds_read_b128 v118 v119 v120 v121, v126 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v034 v035 v036 v037, v050 v051 v052 v053, a[168:171]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v034 v035 v036 v037, v058 v059 v060 v061, a[164:167]
	s_mov_b32 m0, s37	;.loc	2 625 0
	buffer_load_dwordx4 v075, s[8:11], 0 offen lds
	v_mfma_f32_16x16x32_bf16 a[160:163], v034 v035 v036 v037, v062 v063 v064 v065, a[160:163]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[156:159], v026 v027 v028 v029, v018 v019 v020 v021, a[156:159]
	ds_read_b128 v122 v123 v124 v125, v126 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v026 v027 v028 v029, v022 v023 v024 v025, a[152:155]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v026 v027 v028 v029, v030 v031 v032 v033, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v026 v027 v028 v029, v038 v039 v040 v041, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v026 v027 v028 v029, v046 v047 v048 v049, a[140:143]
	ds_read_b128 v126 v127 v128 v129, v126 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v026 v027 v028 v029, v050 v051 v052 v053, a[136:139]	;.loc	2 999 0
	s_mov_b32 m0, s16	;.loc	2 625 0
	buffer_load_dwordx4 v077, s[8:11], 0 offen lds
	v_mfma_f32_16x16x32_bf16 a[132:135], v026 v027 v028 v029, v058 v059 v060 v061, a[132:135]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[128:131], v026 v027 v028 v029, v062 v063 v064 v065, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v014 v015 v016 v017, v018 v019 v020 v021, a[124:127]
	ds_read_b128 v130 v131 v132 v133, v158 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v014 v015 v016 v017, v022 v023 v024 v025, a[120:123]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v014 v015 v016 v017, v030 v031 v032 v033, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v014 v015 v016 v017, v038 v039 v040 v041, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v014 v015 v016 v017, v046 v047 v048 v049, a[108:111]
	s_mov_b32 m0, s31	;.loc	2 625 0
	buffer_load_dwordx4 v079, s[4:7], 0 offen lds
	ds_read_b128 v134 v135 v136 v137, v158 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v014 v015 v016 v017, v050 v051 v052 v053, a[104:107]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v014 v015 v016 v017, v058 v059 v060 v061, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v014 v015 v016 v017, v062 v063 v064 v065, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v010 v011 v012 v013, v018 v019 v020 v021, a[92:95]
	ds_read_b128 v138 v139 v140 v141, v158 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v010 v011 v012 v013, v022 v023 v024 v025, a[88:91]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v010 v011 v012 v013, v030 v031 v032 v033, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v010 v011 v012 v013, v038 v039 v040 v041, a[80:83]
	s_mov_b32 m0, s34	;.loc	2 625 0
	buffer_load_dwordx4 v081, s[4:7], 0 offen lds
	v_mfma_f32_16x16x32_bf16 a[76:79], v010 v011 v012 v013, v046 v047 v048 v049, a[76:79]	;.loc	2 999 0
	ds_read_b128 v142 v143 v144 v145, v158 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v010 v011 v012 v013, v050 v051 v052 v053, a[72:75]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v010 v011 v012 v013, v058 v059 v060 v061, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v010 v011 v012 v013, v062 v063 v064 v065, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v006 v007 v008 v009, v018 v019 v020 v021, a[60:63]
	ds_read_b128 v146 v147 v148 v149, v158 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v006 v007 v008 v009, v022 v023 v024 v025, a[56:59]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v006 v007 v008 v009, v030 v031 v032 v033, a[52:55]
	s_mov_b32 m0, s36	;.loc	2 625 0
	buffer_load_dwordx4 v107, s[4:7], 0 offen lds
	v_mfma_f32_16x16x32_bf16 a[48:51], v006 v007 v008 v009, v038 v039 v040 v041, a[48:51]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[44:47], v006 v007 v008 v009, v046 v047 v048 v049, a[44:47]
	ds_read_b128 v150 v151 v152 v153, v158 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v006 v007 v008 v009, v050 v051 v052 v053, a[40:43]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v006 v007 v008 v009, v058 v059 v060 v061, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v006 v007 v008 v009, v062 v063 v064 v065, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v002 v003 v004 v005, v018 v019 v020 v021, a[28:31]
	ds_read_b128 v154 v155 v156 v157, v158 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v002 v003 v004 v005, v022 v023 v024 v025, a[24:27]	;.loc	2 999 0
	s_mov_b32 m0, s38	;.loc	2 625 0
	buffer_load_dwordx4 v109, s[4:7], 0 offen lds
	v_mfma_f32_16x16x32_bf16 a[20:23], v002 v003 v004 v005, v030 v031 v032 v033, a[20:23]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[16:19], v002 v003 v004 v005, v038 v039 v040 v041, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v002 v003 v004 v005, v046 v047 v048 v049, a[12:15]
	v_add_lshl_u32 v162, s17, v094, 1	;.loc	2 712 24
	v_add_lshl_u32 v163, s17, v096, 1
	v_add_lshl_u32 v164, s17, v098, 1
	v_add_lshl_u32 v165, s17, v100, 1
	v_add_lshl_u32 v166, s17, v102, 1	;.loc	2 674 24
	v_add_lshl_u32 v167, s17, v104, 1
	v_add_lshl_u32 v168, s17, v106, 1
	v_add_lshl_u32 v169, s17, v108, 1
	s_add_i32 s17, s15, 0x10000	;.loc	2 641 0
	s_add_i32 s25, s15, 0x11000
	s_add_i32 s26, s15, 0x12000
	s_add_i32 s27, s15, 0x13000
	s_add_i32 s28, s15, 0x1000
	s_add_i32 s29, s15, 0x2000
	s_add_i32 s30, s15, 0x3000
	ds_read_b128 v158 v159 v160 v161, v158 offset:23552	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v002 v003 v004 v005, v050 v051 v052 v053, a[8:11]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v002 v003 v004 v005, v058 v059 v060 v061, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v002 v003 v004 v005, v062 v063 v064 v065, a[0:3]
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 475 0
	s_barrier
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v082 v083 v084 v085, v130 v131 v132 v133, a[252:255]
	s_mov_b32 m0, s17	;.loc	2 625 0
	buffer_load_dwordx4 v162, s[8:11], 0 offen lds
	ds_read_b128 v054 v055 v056 v057, v054	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v082 v083 v084 v085, v134 v135 v136 v137, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v082 v083 v084 v085, v138 v139 v140 v141, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v082 v083 v084 v085, v142 v143 v144 v145, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v082 v083 v084 v085, v146 v147 v148 v149, a[236:239]
	ds_read_b128 v042 v043 v044 v045, v170	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v082 v083 v084 v085, v150 v151 v152 v153, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v082 v083 v084 v085, v154 v155 v156 v157, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v082 v083 v084 v085, v158 v159 v160 v161, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v086 v087 v088 v089, v130 v131 v132 v133, a[220:223]
	s_mov_b32 m0, s25	;.loc	2 625 0
	buffer_load_dwordx4 v163, s[8:11], 0 offen lds
	ds_read_b128 v034 v035 v036 v037, v170 offset:1024	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v086 v087 v088 v089, v134 v135 v136 v137, a[216:219]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v086 v087 v088 v089, v138 v139 v140 v141, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v086 v087 v088 v089, v142 v143 v144 v145, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v086 v087 v088 v089, v146 v147 v148 v149, a[204:207]
	ds_read_b128 v026 v027 v028 v029, v170 offset:2048	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v086 v087 v088 v089, v150 v151 v152 v153, a[200:203]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v086 v087 v088 v089, v154 v155 v156 v157, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v086 v087 v088 v089, v158 v159 v160 v161, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v090 v091 v092 v093, v130 v131 v132 v133, a[188:191]
	s_mov_b32 m0, s26	;.loc	2 625 0
	buffer_load_dwordx4 v164, s[8:11], 0 offen lds
	ds_read_b128 v014 v015 v016 v017, v170 offset:3072	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v090 v091 v092 v093, v134 v135 v136 v137, a[184:187]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v090 v091 v092 v093, v138 v139 v140 v141, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v090 v091 v092 v093, v142 v143 v144 v145, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v090 v091 v092 v093, v146 v147 v148 v149, a[172:175]
	ds_read_b128 v010 v011 v012 v013, v170 offset:4096	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v090 v091 v092 v093, v150 v151 v152 v153, a[168:171]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v090 v091 v092 v093, v154 v155 v156 v157, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v090 v091 v092 v093, v158 v159 v160 v161, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v110 v111 v112 v113, v130 v131 v132 v133, a[156:159]
	s_mov_b32 m0, s27	;.loc	2 625 0
	buffer_load_dwordx4 v165, s[8:11], 0 offen lds
	ds_read_b128 v006 v007 v008 v009, v170 offset:5120	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v110 v111 v112 v113, v134 v135 v136 v137, a[152:155]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v110 v111 v112 v113, v138 v139 v140 v141, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v110 v111 v112 v113, v142 v143 v144 v145, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v110 v111 v112 v113, v146 v147 v148 v149, a[140:143]
	ds_read_b128 v002 v003 v004 v005, v170 offset:6144	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v110 v111 v112 v113, v150 v151 v152 v153, a[136:139]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v110 v111 v112 v113, v154 v155 v156 v157, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v110 v111 v112 v113, v158 v159 v160 v161, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v114 v115 v116 v117, v130 v131 v132 v133, a[124:127]
	s_mov_b32 m0, s15	;.loc	2 625 0
	buffer_load_dwordx4 v166, s[4:7], 0 offen lds
	ds_read_b128 v018 v019 v020 v021, v171	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v114 v115 v116 v117, v134 v135 v136 v137, a[120:123]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v114 v115 v116 v117, v138 v139 v140 v141, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v114 v115 v116 v117, v142 v143 v144 v145, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v114 v115 v116 v117, v146 v147 v148 v149, a[108:111]
	ds_read_b128 v022 v023 v024 v025, v171 offset:1024	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v114 v115 v116 v117, v150 v151 v152 v153, a[104:107]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v114 v115 v116 v117, v154 v155 v156 v157, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v114 v115 v116 v117, v158 v159 v160 v161, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v118 v119 v120 v121, v130 v131 v132 v133, a[92:95]
	s_mov_b32 m0, s28	;.loc	2 625 0
	buffer_load_dwordx4 v167, s[4:7], 0 offen lds
	ds_read_b128 v030 v031 v032 v033, v171 offset:2048	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v118 v119 v120 v121, v134 v135 v136 v137, a[88:91]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v118 v119 v120 v121, v138 v139 v140 v141, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v118 v119 v120 v121, v142 v143 v144 v145, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v118 v119 v120 v121, v146 v147 v148 v149, a[76:79]
	ds_read_b128 v038 v039 v040 v041, v171 offset:3072	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v118 v119 v120 v121, v150 v151 v152 v153, a[72:75]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v118 v119 v120 v121, v154 v155 v156 v157, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v118 v119 v120 v121, v158 v159 v160 v161, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v122 v123 v124 v125, v130 v131 v132 v133, a[60:63]
	s_mov_b32 m0, s29	;.loc	2 625 0
	buffer_load_dwordx4 v168, s[4:7], 0 offen lds
	ds_read_b128 v046 v047 v048 v049, v171 offset:4096	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v122 v123 v124 v125, v134 v135 v136 v137, a[56:59]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v122 v123 v124 v125, v138 v139 v140 v141, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v122 v123 v124 v125, v142 v143 v144 v145, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v122 v123 v124 v125, v146 v147 v148 v149, a[44:47]
	ds_read_b128 v050 v051 v052 v053, v171 offset:5120	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v122 v123 v124 v125, v150 v151 v152 v153, a[40:43]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v122 v123 v124 v125, v154 v155 v156 v157, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v122 v123 v124 v125, v158 v159 v160 v161, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v126 v127 v128 v129, v130 v131 v132 v133, a[28:31]
	s_mov_b32 m0, s30	;.loc	2 625 0
	buffer_load_dwordx4 v169, s[4:7], 0 offen lds
	ds_read_b128 v058 v059 v060 v061, v171 offset:6144	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v126 v127 v128 v129, v134 v135 v136 v137, a[24:27]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v126 v127 v128 v129, v138 v139 v140 v141, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v126 v127 v128 v129, v142 v143 v144 v145, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v126 v127 v128 v129, v146 v147 v148 v149, a[12:15]
	ds_read_b128 v062 v063 v064 v065, v171 offset:7168	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v126 v127 v128 v129, v150 v151 v152 v153, a[8:11]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v126 v127 v128 v129, v154 v155 v156 v157, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v126 v127 v128 v129, v158 v159 v160 v161, a[0:3]
	s_add_u32 s2, s2, 0x8000	;.loc	2 1225 0
	s_addc_u32 s3, s3, 0
	s_add_i32 s24, s24, 64
	s_cmp_lg_u64 s[2:3], 0
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 475 0
	s_barrier
.JUMP.LBB0_1:
	s_cbranch_scc1 .LBB0_1	;.loc	2 1225 0
	s_load_dwordx2 s[0:1], s[0:1], 0x0	;.loc	1 289 24
	s_lshl_b32 s4, s14, 14	;.loc	1 0 0 is_stmt 0
	s_lshl_b32 s4, s4, 1	;.loc	1 317 12 is_stmt 1
	v_add3_u32 v067, v105, s4, v099
	s_waitcnt lgkmcnt(0)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v054 v055 v056 v057, v018 v019 v020 v021, a[252:255]
	ds_read_b128 v068 v069 v070 v071, v067 offset:16384	;.loc	1 317 12
	v_add3_u32 v067, v103, s4, v099
	v_mfma_f32_16x16x32_bf16 a[248:251], v054 v055 v056 v057, v022 v023 v024 v025, a[248:251]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[244:247], v054 v055 v056 v057, v030 v031 v032 v033, a[244:247]
	v_mfma_f32_16x16x32_bf16 a[240:243], v054 v055 v056 v057, v038 v039 v040 v041, a[240:243]
	v_mfma_f32_16x16x32_bf16 a[236:239], v054 v055 v056 v057, v046 v047 v048 v049, a[236:239]
	ds_read_b128 v072 v073 v074 v075, v067 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[232:235], v054 v055 v056 v057, v050 v051 v052 v053, a[232:235]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[228:231], v054 v055 v056 v057, v058 v059 v060 v061, a[228:231]
	v_mfma_f32_16x16x32_bf16 a[224:227], v054 v055 v056 v057, v062 v063 v064 v065, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v042 v043 v044 v045, v018 v019 v020 v021, a[220:223]
	ds_read_b128 v076 v077 v078 v079, v067 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v042 v043 v044 v045, v022 v023 v024 v025, a[216:219]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v042 v043 v044 v045, v030 v031 v032 v033, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v042 v043 v044 v045, v038 v039 v040 v041, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v042 v043 v044 v045, v046 v047 v048 v049, a[204:207]
	ds_read_b128 v080 v081 v082 v083, v067 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v042 v043 v044 v045, v050 v051 v052 v053, a[200:203]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v042 v043 v044 v045, v058 v059 v060 v061, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v042 v043 v044 v045, v062 v063 v064 v065, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v034 v035 v036 v037, v018 v019 v020 v021, a[188:191]
	ds_read_b128 v084 v085 v086 v087, v067 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v034 v035 v036 v037, v022 v023 v024 v025, a[184:187]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v034 v035 v036 v037, v030 v031 v032 v033, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v034 v035 v036 v037, v038 v039 v040 v041, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v034 v035 v036 v037, v046 v047 v048 v049, a[172:175]
	ds_read_b128 v054 v055 v056 v057, v067 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v034 v035 v036 v037, v050 v051 v052 v053, a[168:171]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v034 v035 v036 v037, v058 v059 v060 v061, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v034 v035 v036 v037, v062 v063 v064 v065, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v026 v027 v028 v029, v018 v019 v020 v021, a[156:159]
	ds_read_b128 v042 v043 v044 v045, v067 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v026 v027 v028 v029, v022 v023 v024 v025, a[152:155]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v026 v027 v028 v029, v030 v031 v032 v033, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v026 v027 v028 v029, v038 v039 v040 v041, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v026 v027 v028 v029, v046 v047 v048 v049, a[140:143]
	ds_read_b128 v034 v035 v036 v037, v067 offset:22528	;.loc	1 317 12
	v_add3_u32 v067, v101, s4, v099
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_and_b32 s1, s1, 0xffff	;.loc	1 289 24
	v_mfma_f32_16x16x32_bf16 a[136:139], v026 v027 v028 v029, v050 v051 v052 v053, a[136:139]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v026 v027 v028 v029, v058 v059 v060 v061, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v026 v027 v028 v029, v062 v063 v064 v065, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v014 v015 v016 v017, v018 v019 v020 v021, a[124:127]
	ds_read_b128 v026 v027 v028 v029, v067 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v014 v015 v016 v017, v022 v023 v024 v025, a[120:123]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v014 v015 v016 v017, v030 v031 v032 v033, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v014 v015 v016 v017, v038 v039 v040 v041, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v014 v015 v016 v017, v046 v047 v048 v049, a[108:111]
	ds_read_b128 v088 v089 v090 v091, v067 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v014 v015 v016 v017, v050 v051 v052 v053, a[104:107]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v014 v015 v016 v017, v058 v059 v060 v061, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v014 v015 v016 v017, v062 v063 v064 v065, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v010 v011 v012 v013, v018 v019 v020 v021, a[92:95]
	ds_read_b128 v014 v015 v016 v017, v067 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v010 v011 v012 v013, v022 v023 v024 v025, a[88:91]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v010 v011 v012 v013, v030 v031 v032 v033, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v010 v011 v012 v013, v038 v039 v040 v041, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v010 v011 v012 v013, v046 v047 v048 v049, a[76:79]
	ds_read_b128 v098 v099 v100 v101, v067 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v010 v011 v012 v013, v050 v051 v052 v053, a[72:75]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v010 v011 v012 v013, v058 v059 v060 v061, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v010 v011 v012 v013, v062 v063 v064 v065, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v006 v007 v008 v009, v018 v019 v020 v021, a[60:63]
	ds_read_b128 v010 v011 v012 v013, v067 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v006 v007 v008 v009, v022 v023 v024 v025, a[56:59]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v006 v007 v008 v009, v030 v031 v032 v033, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v006 v007 v008 v009, v038 v039 v040 v041, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v006 v007 v008 v009, v046 v047 v048 v049, a[44:47]
	ds_read_b128 v102 v103 v104 v105, v067 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v006 v007 v008 v009, v050 v051 v052 v053, a[40:43]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v006 v007 v008 v009, v058 v059 v060 v061, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v006 v007 v008 v009, v062 v063 v064 v065, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v002 v003 v004 v005, v018 v019 v020 v021, a[28:31]
	ds_read_b128 v006 v007 v008 v009, v067 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v002 v003 v004 v005, v022 v023 v024 v025, a[24:27]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v002 v003 v004 v005, v030 v031 v032 v033, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v002 v003 v004 v005, v038 v039 v040 v041, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v002 v003 v004 v005, v046 v047 v048 v049, a[12:15]
	ds_read_b128 v018 v019 v020 v021, v067 offset:23552	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v002 v003 v004 v005, v050 v051 v052 v053, a[8:11]	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v002 v003 v004 v005, v058 v059 v060 v061, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v002 v003 v004 v005, v062 v063 v064 v065, a[0:3]
	v_or_b32_e32 v066, 0x70, v001	;.loc	2 755 0
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 475 0
	s_barrier
	s_waitcnt lgkmcnt(7)	;.loc	2 999 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v068 v069 v070 v071, v026 v027 v028 v029, a[252:255]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[248:251], v068 v069 v070 v071, v088 v089 v090 v091, a[248:251]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[244:247], v068 v069 v070 v071, v014 v015 v016 v017, a[244:247]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[240:243], v068 v069 v070 v071, v098 v099 v100 v101, a[240:243]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[236:239], v068 v069 v070 v071, v010 v011 v012 v013, a[236:239]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[232:235], v068 v069 v070 v071, v102 v103 v104 v105, a[232:235]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 a[228:231], v068 v069 v070 v071, v006 v007 v008 v009, a[228:231]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 a[224:227], v068 v069 v070 v071, v018 v019 v020 v021, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v072 v073 v074 v075, v026 v027 v028 v029, a[220:223]
	v_mfma_f32_16x16x32_bf16 a[216:219], v072 v073 v074 v075, v088 v089 v090 v091, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v072 v073 v074 v075, v014 v015 v016 v017, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v072 v073 v074 v075, v098 v099 v100 v101, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v072 v073 v074 v075, v010 v011 v012 v013, a[204:207]
	v_mfma_f32_16x16x32_bf16 a[200:203], v072 v073 v074 v075, v102 v103 v104 v105, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v072 v073 v074 v075, v006 v007 v008 v009, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v072 v073 v074 v075, v018 v019 v020 v021, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v076 v077 v078 v079, v026 v027 v028 v029, a[188:191]
	v_mfma_f32_16x16x32_bf16 a[184:187], v076 v077 v078 v079, v088 v089 v090 v091, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v076 v077 v078 v079, v014 v015 v016 v017, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v076 v077 v078 v079, v098 v099 v100 v101, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v076 v077 v078 v079, v010 v011 v012 v013, a[172:175]
	v_mfma_f32_16x16x32_bf16 a[168:171], v076 v077 v078 v079, v102 v103 v104 v105, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v076 v077 v078 v079, v006 v007 v008 v009, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v076 v077 v078 v079, v018 v019 v020 v021, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v080 v081 v082 v083, v026 v027 v028 v029, a[156:159]
	v_mfma_f32_16x16x32_bf16 a[152:155], v080 v081 v082 v083, v088 v089 v090 v091, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v080 v081 v082 v083, v014 v015 v016 v017, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v080 v081 v082 v083, v098 v099 v100 v101, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v080 v081 v082 v083, v010 v011 v012 v013, a[140:143]
	v_mfma_f32_16x16x32_bf16 a[136:139], v080 v081 v082 v083, v102 v103 v104 v105, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v080 v081 v082 v083, v006 v007 v008 v009, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v080 v081 v082 v083, v018 v019 v020 v021, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v084 v085 v086 v087, v026 v027 v028 v029, a[124:127]
	v_mfma_f32_16x16x32_bf16 a[120:123], v084 v085 v086 v087, v088 v089 v090 v091, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v084 v085 v086 v087, v014 v015 v016 v017, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v084 v085 v086 v087, v098 v099 v100 v101, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v084 v085 v086 v087, v010 v011 v012 v013, a[108:111]
	v_mfma_f32_16x16x32_bf16 a[104:107], v084 v085 v086 v087, v102 v103 v104 v105, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v084 v085 v086 v087, v006 v007 v008 v009, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v084 v085 v086 v087, v018 v019 v020 v021, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v054 v055 v056 v057, v026 v027 v028 v029, a[92:95]
	v_mfma_f32_16x16x32_bf16 a[88:91], v054 v055 v056 v057, v088 v089 v090 v091, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v054 v055 v056 v057, v014 v015 v016 v017, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v054 v055 v056 v057, v098 v099 v100 v101, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v054 v055 v056 v057, v010 v011 v012 v013, a[76:79]
	v_mfma_f32_16x16x32_bf16 a[72:75], v054 v055 v056 v057, v102 v103 v104 v105, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v054 v055 v056 v057, v006 v007 v008 v009, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v054 v055 v056 v057, v018 v019 v020 v021, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v042 v043 v044 v045, v026 v027 v028 v029, a[60:63]
	v_mfma_f32_16x16x32_bf16 a[56:59], v042 v043 v044 v045, v088 v089 v090 v091, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v042 v043 v044 v045, v014 v015 v016 v017, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v042 v043 v044 v045, v098 v099 v100 v101, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v042 v043 v044 v045, v010 v011 v012 v013, a[44:47]
	v_mfma_f32_16x16x32_bf16 a[40:43], v042 v043 v044 v045, v102 v103 v104 v105, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v042 v043 v044 v045, v006 v007 v008 v009, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v042 v043 v044 v045, v018 v019 v020 v021, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v034 v035 v036 v037, v026 v027 v028 v029, a[28:31]
	v_mfma_f32_16x16x32_bf16 a[24:27], v034 v035 v036 v037, v088 v089 v090 v091, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v034 v035 v036 v037, v014 v015 v016 v017, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v034 v035 v036 v037, v098 v099 v100 v101, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v034 v035 v036 v037, v010 v011 v012 v013, a[12:15]
	v_mfma_f32_16x16x32_bf16 a[8:11], v034 v035 v036 v037, v102 v103 v104 v105, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v034 v035 v036 v037, v006 v007 v008 v009, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v034 v035 v036 v037, v018 v019 v020 v021, a[0:3]
	v_lshlrev_b32_e32 v002, 2, v095	;.loc	2 1379 0
	v_accvgpr_read_b32 v003, a252	;.loc	2 1389 22
	v_and_or_b32 v004, v002, 12, v001	;.loc	2 1387 0
	v_cvt_pk_bf16_f32 v005, v003, s0	;.loc	2 1390 0
	v_lshlrev_b32_e32 v003, 1, v097	;.loc	1 330 12
	v_lshl_or_b32 v004, v004, 9, v003
	s_barrier	;.loc	2 1381 0
	ds_write_b16 v004, v005	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a253	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:512	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a254	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1024	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a255	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1536	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a248	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:32	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a249	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:544	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a250	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1056	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a251	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1568	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a244	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:64	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a245	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:576	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a246	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1088	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a247	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1600	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a240	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:96	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a241	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:608	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a242	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1120	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a243	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1632	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a236	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:128	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a237	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:640	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a238	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1152	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a239	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1664	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a232	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:160	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a233	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:672	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a234	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1184	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a235	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1696	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a228	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:192	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a229	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:704	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a230	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1216	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a231	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1728	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a224	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:224	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a225	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:736	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a226	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1248	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a227	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:1760	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a220	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8192	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a221	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8704	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a222	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9216	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a223	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9728	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a216	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8224	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a217	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8736	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a218	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9248	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a219	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9760	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a212	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8256	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a213	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8768	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a214	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9280	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a215	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9792	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a208	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8288	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a209	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8800	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a210	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9312	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a211	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9824	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a204	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8320	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a205	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8832	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a206	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9344	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a207	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9856	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a200	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8352	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a201	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8864	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a202	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9376	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a203	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9888	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a196	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8384	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a197	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8896	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a198	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9408	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a199	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9920	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a192	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8416	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a193	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:8928	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a194	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9440	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a195	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:9952	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a188	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16384	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a189	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16896	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a190	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17408	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a191	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17920	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a184	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16416	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a185	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16928	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a186	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17440	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a187	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17952	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a180	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16448	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a181	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16960	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a182	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17472	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a183	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17984	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a176	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16480	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a177	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16992	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a178	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17504	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a179	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:18016	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a172	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16512	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a173	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17024	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a174	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17536	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a175	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:18048	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a168	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16544	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a169	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17056	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a170	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17568	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a171	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:18080	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a164	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16576	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a165	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17088	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a166	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17600	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a167	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:18112	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a160	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:16608	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a161	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17120	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a162	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:17632	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a163	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v004, v005 offset:18144	;.loc	1 330 12
	v_or3_b32 v001, v001, 48, v002	;.loc	2 1387 0
	v_accvgpr_read_b32 v005, a156	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	v_lshl_or_b32 v001, v001, 9, v003	;.loc	1 330 12
	ds_write_b16 v001, v005
	v_accvgpr_read_b32 v005, a157	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:512	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a158	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1024	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a159	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1536	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a152	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:32	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a153	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:544	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a154	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1056	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a155	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1568	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a148	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:64	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a149	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:576	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a150	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1088	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a151	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1600	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a144	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:96	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a145	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:608	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a146	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1120	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a147	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1632	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a140	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:128	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a141	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:640	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a142	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1152	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a143	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1664	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a136	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:160	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a137	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:672	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a138	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1184	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a139	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1696	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a132	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:192	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a133	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:704	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a134	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1216	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a135	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1728	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a128	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:224	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a129	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:736	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a130	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1248	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a131	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1390 0
	ds_write_b16 v001, v005 offset:1760	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a124	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32768	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a125	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33280	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a126	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33792	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a127	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34304	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a120	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32800	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a121	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33312	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a122	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33824	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a123	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34336	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a116	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32832	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a117	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33344	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a118	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33856	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a119	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34368	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a112	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32864	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a113	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33376	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a114	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33888	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a115	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34400	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a108	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32896	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a109	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33408	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a110	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33920	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a111	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34432	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a104	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32928	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a105	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33440	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a106	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33952	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a107	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34464	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a100	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32960	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a101	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33472	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a102	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33984	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a103	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34496	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a96	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:32992	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a97	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:33504	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a98	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34016	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a99	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:34528	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a92	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:40960	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a93	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41472	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a94	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41984	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a95	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42496	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a88	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:40992	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a89	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41504	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a90	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42016	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a91	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42528	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a84	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41024	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a85	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41536	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a86	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42048	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a87	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42560	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a80	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41056	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a81	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41568	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a82	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42080	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a83	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42592	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a76	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41088	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a77	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41600	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a78	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42112	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a79	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42624	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a72	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41120	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a73	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41632	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a74	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42144	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a75	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42656	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a68	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41152	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a69	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41664	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a70	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42176	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a71	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42688	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a64	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41184	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a65	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:41696	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a66	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42208	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a67	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:42720	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a60	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49152	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a61	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49664	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a62	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50176	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a63	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50688	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a56	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49184	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a57	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49696	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a58	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50208	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a59	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50720	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a52	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49216	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a53	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49728	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a54	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50240	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a55	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50752	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a48	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49248	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a49	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49760	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a50	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50272	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a51	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50784	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a44	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49280	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a45	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49792	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a46	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50304	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a47	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50816	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a40	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49312	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a41	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49824	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a42	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50336	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a43	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50848	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a36	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49344	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a37	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49856	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a38	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50368	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a39	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50880	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a32	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49376	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a33	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:49888	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a34	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50400	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a35	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1390 0
	ds_write_b16 v004, v001 offset:50912	;.loc	1 330 12
	v_or_b32_e32 v001, v002, v066	;.loc	2 1387 0
	v_accvgpr_read_b32 v002, a28	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	v_lshl_or_b32 v001, v001, 9, v003	;.loc	1 330 12
	ds_write_b16 v001, v002
	v_accvgpr_read_b32 v002, a29	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:512	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a30	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1024	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a31	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1536	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a24	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:32	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a25	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:544	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a26	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1056	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a27	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1568	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a20	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:64	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a21	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:576	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a22	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1088	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a23	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1600	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a16	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:96	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a17	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:608	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a18	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1120	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a19	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1632	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a12	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:128	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a13	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:640	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a14	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1152	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a15	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1664	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a8	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:160	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a9	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:672	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a10	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1184	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a11	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1696	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a4	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:192	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a5	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:704	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a6	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1216	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a7	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1728	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a0	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:224	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a1	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:736	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a2	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	ds_write_b16 v001, v002 offset:1248	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a3	;.loc	2 1389 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1390 0
	v_lshrrev_b32_e32 v004, 5, v000	;.loc	2 1433 0
	ds_write_b16 v001, v002 offset:1760	;.loc	1 330 12
	v_or_b32_e32 v002, s20, v004	;.loc	2 1435 0
	v_mov_b32_e32 v003, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 1436 28
	v_lshlrev_b32_e32 v003, 3, v000
	s_waitcnt lgkmcnt(0)	;.loc	2 1430 0
	s_barrier
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_4:
	s_cbranch_execz .LBB0_4
	v_and_b32_e32 v000, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v001, 1, v000	;.loc	1 317 12
	v_lshl_or_b32 v001, v004, 9, v001
	ds_read_b128 v006 v007 v008 v009, v001
	v_or_b32_e32 v000, s18, v000	;.loc	2 1445 0
	v_lshlrev_b32_e32 v001, 13, v002
	v_add_lshl_u32 v000, v000, v001, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_4:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_6:
	s_cbranch_execz .LBB0_6
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 16, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_8:
	s_cbranch_execz .LBB0_8
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_8:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 24, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_10:
	s_cbranch_execz .LBB0_10
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_10:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 32, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_12:
	s_cbranch_execz .LBB0_12
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_12:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 40, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_14:
	s_cbranch_execz .LBB0_14
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_14:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 48, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_16:
	s_cbranch_execz .LBB0_16
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_16:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 56, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_18:
	s_cbranch_execz .LBB0_18
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_18:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 64, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_20:
	s_cbranch_execz .LBB0_20
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x48, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_22:
	s_cbranch_execz .LBB0_22
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_22:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x50, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_24:
	s_cbranch_execz .LBB0_24
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_24:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x58, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_26:
	s_cbranch_execz .LBB0_26
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_26:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x60, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_28:
	s_cbranch_execz .LBB0_28
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x68, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_30:
	s_cbranch_execz .LBB0_30
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x70, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_32:
	s_cbranch_execz .LBB0_32
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x78, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_34:
	s_cbranch_execz .LBB0_34
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x80, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_36:
	s_cbranch_execz .LBB0_36
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x88, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_38:
	s_cbranch_execz .LBB0_38
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x90, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_40:
	s_cbranch_execz .LBB0_40
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x98, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_42:
	s_cbranch_execz .LBB0_42
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xa0, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_44:
	s_cbranch_execz .LBB0_44
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xa8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_46:
	s_cbranch_execz .LBB0_46
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xb0, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_48:
	s_cbranch_execz .LBB0_48
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xb8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_50:
	s_cbranch_execz .LBB0_50
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_50:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xc0, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_52:
	s_cbranch_execz .LBB0_52
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xc8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_54:
	s_cbranch_execz .LBB0_54
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xd0, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_56:
	s_cbranch_execz .LBB0_56
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xd8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_58:
	s_cbranch_execz .LBB0_58
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_58:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xe0, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_60:
	s_cbranch_execz .LBB0_60
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_60:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xe8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_62:
	s_cbranch_execz .LBB0_62
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_62:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xf0, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_64:
	s_cbranch_execz .LBB0_64
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_64:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xf8, v004	;.loc	2 1433 0 is_stmt 1
	v_or_b32_e32 v000, s20, v002	;.loc	2 1435 0
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1436 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1437 0
.JUMP.LBB0_66:
	s_cbranch_execz .LBB0_66
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1434 0
	v_lshlrev_b32_e32 v003, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v003
	ds_read_b128 v002 v003 v004 v005, v002
	v_or_b32_e32 v001, s18, v001	;.loc	2 1445 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v002 v003 v004 v005, v000, s[0:3], 0 offen
.LBB0_66:
	s_endpgm	;.loc	2 327 0
.Lfunc_end0:
	.size	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0, .Lfunc_end0-hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
	.cfi_endproc
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.num_vgpr, 172
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.num_agpr, 256
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.numbered_sgpr, 39
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.num_named_barrier, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.private_seg_size, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.uses_vcc, 1
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.uses_flat_scratch, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.has_dyn_sized_stack, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.has_recursion, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	0
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
...
	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
