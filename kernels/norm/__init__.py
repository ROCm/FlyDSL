# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x norm, RoPE, and AdaLN kernels.

FlyDSL-native RMS / RoPE / AdaLN for RDNA4. Prefer fused ``rms_rope_gfx120x``
when RMSNorm and RoPE run in the same layer.
"""
