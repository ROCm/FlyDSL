# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Experimental megamoe stage-1 workspace (copied from the retired ``tmp_test``).

Self-contained package for developing the mega stage-1 dispatch/GEMM overlap
(``docs/moe_stage1_overlap_design.md``) WITHOUT touching the production megamoe
kernels under ``kernels/``. Production modules (layout_utils, mfma_*,
moe_common, dispatch_combine_intranode_*) are imported absolutely as read-only
dependencies; mega-exp modules import each other relatively.
"""
