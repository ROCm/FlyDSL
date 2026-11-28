// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_INITALLDIALECTS_H
#define FLYDSL_INITALLDIALECTS_H

namespace mlir {
class DialectRegistry;

namespace fly {
/// Add the FlyDSL dialects, the dialects of every enabled backend, and their
/// external models to `registry`.
void registerAllDialects(DialectRegistry &registry);
} // namespace fly
} // namespace mlir

#endif // FLYDSL_INITALLDIALECTS_H
