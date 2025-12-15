// Test rocir.composition operation
func.func @test_composition() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // Create first layout: shape(8,16), stride(1,8) - row-major 8x16
  %shape_a = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride_a = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout_a = rocir.make_layout %shape_a, %stride_a : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Create second layout: shape(8,16), stride(2,16) - strided
  %shape_b = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride_b = rocir.make_stride %c2, %c16 : (index, index) -> !rocir.stride<(?,?)>
  %layout_b = rocir.make_layout %shape_b, %stride_b : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Compose layouts: layoutA ∘ layoutB
  // Result should have shape(8,16) and stride(2*1, 16*8) = stride(2, 128)
  %composed = rocir.composition %layout_a, %layout_b : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Test: get size of composed layout
  %size = rocir.size %composed : !rocir.layout<(?,?)> -> index
  
  return %size : index
}

// Test composition followed by coordinate mapping
func.func @test_composition_crd2idx() -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  
  // Layout A: identity-like
  %shape_a = rocir.make_shape %c4, %c8 : (index, index) -> !rocir.shape<(?,?)>
  %stride_a = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<(?,?)>
  %layout_a = rocir.make_layout %shape_a, %stride_a : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Layout B: with stride multipliers
  %shape_b = rocir.make_shape %c4, %c8 : (index, index) -> !rocir.shape<(?,?)>
  %stride_b = rocir.make_stride %c2, %c1 : (index, index) -> !rocir.stride<(?,?)>
  %layout_b = rocir.make_layout %shape_b, %stride_b : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Compose: stride should be (2*1, 1*4) = (2, 4)
  %composed = rocir.composition %layout_a, %layout_b : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Map coordinate (1, 2) through composed layout
  %coord = rocir.make_coord %c1, %c2 : (index, index) -> !rocir.coord<(?,?)>
  %idx = rocir.crd2idx %coord, %composed : (!rocir.coord<(?,?)>, !rocir.layout<(?,?)>) -> index
  // Expected: 1*2 + 2*4 = 2 + 8 = 10
  
  return %idx : index
}

// Test composition preserves shape
func.func @test_composition_shape() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_a = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride_a = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout_a = rocir.make_layout %shape_a, %stride_a : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %shape_b = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride_b = rocir.make_stride %c2, %c16 : (index, index) -> !rocir.stride<(?,?)>
  %layout_b = rocir.make_layout %shape_b, %stride_b : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %composed = rocir.composition %layout_a, %layout_b : (!rocir.layout<(?,?)>, !rocir.layout<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Extract and verify shape is preserved (should be shape_b)
  %result_shape = rocir.get_shape %composed : !rocir.layout<(?,?)> -> !rocir.shape<(?,?)>
  %rank = rocir.rank %result_shape : !rocir.shape<(?,?)> -> index
  
  return
}
