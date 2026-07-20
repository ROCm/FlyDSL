"""Architecture compatibility configuration for GPU tests and examples.

Single source of truth for what runs on CDNA vs RDNA GPUs.
Referenced by:
  - tests/kernels/conftest.py  (pytest collection filter)
  - scripts/run_tests.sh       (example script filter)
"""

# Test files that ONLY work on CDNA (gfx9xx) GPUs.
# Reasons: MFMA instructions, hardcoded wave64, or imports from CDNA-only kernels.
CDNA_ONLY_TESTS = frozenset({
    "test_preshuffle_gemm.py",
    "test_blockscale_preshuffle_gemm.py",
    "test_moe_gemm.py",
    "test_moe_blockscale.py",
    "test_moe_reduce.py",
    "test_pa.py",
    "test_quant.py",
})

# Example scripts verified to work on RDNA (non-CDNA) GPUs.
# On CDNA all examples run; on RDNA only whitelisted ones run.
RDNA_COMPATIBLE_EXAMPLES = frozenset({
    "01-vectorAdd.py",
    "02-tiledCopy.py",
})
