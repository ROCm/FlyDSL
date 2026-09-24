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
    "extension/coop/01-warp_collectives.py": ("*",),
    "extension/coop/02-block_scan.py": ("*",),
}
