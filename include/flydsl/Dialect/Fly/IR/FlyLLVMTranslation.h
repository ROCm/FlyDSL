// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_IR_FLYLLVMTRANSLATION_H
#define FLYDSL_DIALECT_FLY_IR_FLYLLVMTRANSLATION_H

namespace mlir {
class MLIRContext;

namespace fly {
void registerExplicitModuleOffloadingLLVMTranslation(MLIRContext &context);
} // namespace fly
} // namespace mlir

#endif // FLYDSL_DIALECT_FLY_IR_FLYLLVMTRANSLATION_H
