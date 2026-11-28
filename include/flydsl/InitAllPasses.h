// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_INITALLPASSES_H
#define FLYDSL_INITALLPASSES_H

namespace mlir {
namespace fly {
/// Register the FlyDSL passes and the passes of every enabled backend.
void registerAllPasses();
} // namespace fly
} // namespace mlir

#endif // FLYDSL_INITALLPASSES_H
