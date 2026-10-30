# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Assemble shared compile-time tools for every GLM stage and quant format."""

import flydsl.compiler as flyc
from kernels.monokernel.glm.context import Operations
from kernels.monokernel.glm.tools.communication import build_communication
from kernels.monokernel.glm.tools.gemm import build_gemm
from kernels.monokernel.glm.tools.memory import build_memory
from kernels.monokernel.glm.tools.normalization import build_normalization
from kernels.monokernel.glm.tools.quantization import build_quantization
from kernels.monokernel.glm.tools.reductions import build_reductions


@flyc.jit
def build_tools(plan, execution, index_weights):
    memory = build_memory(plan, execution, index_weights)
    reductions = build_reductions(plan, execution, memory)
    gemm = build_gemm(plan, execution, memory)
    normalization = build_normalization(plan, execution, memory, reductions)
    quantization = build_quantization(plan, execution, memory, reductions)
    communication = build_communication(plan, execution, memory, reductions)
    return Operations(memory, reductions, gemm, normalization, quantization, communication)
