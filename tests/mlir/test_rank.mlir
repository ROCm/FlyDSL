// SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Test flir.rank operation
func.func @test_rank() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = flir.make_shape %c8, %c16, %c32 : (index, index, index) -> !flir.shape<(?,?,?)>
  %rank = flir.rank %shape : !flir.shape<(?,?,?)> -> index
  
  return %rank : index
}
