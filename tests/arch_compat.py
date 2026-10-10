"""Architecture compatibility configuration for GPU tests and examples.

Single source of truth for GPU architecture restrictions.
Referenced by:
  - tests/kernels/conftest.py  (pytest collection filter)
  - scripts/run_tests.sh       (example script filter)
"""

# Test files that ONLY work on CDNA (gfx9xx) GPUs.
# Reasons: MFMA instructions, hardcoded wave64, or imports from CDNA-only kernels.
CDNA_ONLY_TESTS = frozenset(
    {
        "test_flash_attn_fwd.py",  # MFMA + hardcoded wave64 FMHA kernels
        "test_gemm_a16w16_gfx950.py",  # gfx950-only A16W16 GEMM kernel
        "test_preshuffle_gemm.py",
        "test_moe_gemm.py",
        "test_moe_reduce.py",
        "test_pa.py",
        "test_swa_gfx950.py",
        "test_quant.py",
        "test_allreduce.py",  # custom_all_reduce requires CDNA (gfx9xx)
        "test_mega_moe_v2.py",  # MegaMoEV2 A8W4/A4W4 requires CDNA4 (gfx95x)
    }
)

# Paths relative to examples/ -> supported architectures (shell glob patterns).
# "*" allows any backend
# Register each standalone example here so CI selects it before starting Python.
EXAMPLE_ARCHITECTURES = {
    "01-vectorAdd.py": ("*",),
    "02-tiledCopy.py": ("gfx*",),
    "03-tiledMma.py": ("gfx9*",),
    "04-preshuffle_gemm.py": ("gfx9*",),
    "05-gather_scatter.py": ("*",),
    "06-cdna5_tensor_copy.py": ("gfx1250",),
    "07-tiledMma_gfx120x.py": ("gfx120*",),  # RDNA4 WMMA sibling of 03-tiledMma (MFMA)
    "08-flash_attn_gfx120x.py": ("gfx120*",),
    "09-int8_linear_gfx120x.py": ("gfx120*",),
    "10-scaled_mm_fp8_gfx120x.py": ("gfx120*",),
    "11-convrot_w4a4_gfx120x.py": ("gfx120*",),
    "12-norm_rope_gfx120x.py": ("gfx120*",),
    "13-quant_int8_gfx120x.py": ("gfx120*",),
    "14-iu4_gemm_gfx120x.py": ("gfx120*",),
    "15-fused_mlp_gfx120x.py": ("gfx120*",),
    "16-awq_w4a16_gfx120x.py": ("gfx120*",),
    "17-svdquant_w4a4_gfx120x.py": ("gfx120*",),
    "18-asym_w4a8_gfx120x.py": ("gfx120*",),
    "19-mxfp8_block_gemm_gfx120x.py": ("gfx120*",),
    "20-swiglu_gfx120x.py": ("gfx120*",),
    "21-flash_attn_quant_gfx120x.py": ("gfx120*",),
    "22-adaln_gfx120x.py": ("gfx120*",),
    "extension/coop/01-warp_collectives.py": ("*",),
    "extension/coop/02-block_scan.py": ("*",),
}
