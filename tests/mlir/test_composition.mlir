// Test flir.composition operation
func.func @test_composition() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // Create first layout: shape(8,16), stride(1,8) - row-major 8x16
  %shape_a = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_a = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout_a = flir.make_layout %shape_a, %stride_a : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Create second layout: shape(8,16), stride(2,16) - strided
  %shape_b = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_b = flir.make_stride %c2, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout_b = flir.make_layout %shape_b, %stride_b : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Compose layouts: layoutA ∘ layoutB
  // Result should have shape(8,16) and stride(2*1, 16*8) = stride(2, 128)
  %composed = flir.composition %layout_a, %layout_b : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  // Test: get size of composed layout
  %size = flir.size %composed : !flir.layout<(?,?)> -> index
  
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
  %shape_a = flir.make_shape %c4, %c8 : (index, index) -> !flir.shape<(?,?)>
  %stride_a = flir.make_stride %c1, %c4 : (index, index) -> !flir.stride<(?,?)>
  %layout_a = flir.make_layout %shape_a, %stride_a : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Layout B: with stride multipliers
  %shape_b = flir.make_shape %c4, %c8 : (index, index) -> !flir.shape<(?,?)>
  %stride_b = flir.make_stride %c2, %c1 : (index, index) -> !flir.stride<(?,?)>
  %layout_b = flir.make_layout %shape_b, %stride_b : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Compose: stride should be (2*1, 1*4) = (2, 4)
  %composed = flir.composition %layout_a, %layout_b : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  // Map coordinate (1, 2) through composed layout
  %coord = flir.make_coord %c1, %c2 : (index, index) -> !flir.coord<(?,?)>
  %idx = flir.crd2idx %coord, %composed : (!flir.coord<(?,?)>, !flir.layout<(?,?)>) -> index
  // Expected: 1*2 + 2*4 = 2 + 8 = 10
  
  return %idx : index
}

// Test composition preserves shape
func.func @test_composition_shape() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_a = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_a = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout_a = flir.make_layout %shape_a, %stride_a : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %shape_b = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride_b = flir.make_stride %c2, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout_b = flir.make_layout %shape_b, %stride_b : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %composed = flir.composition %layout_a, %layout_b : (!flir.layout<(?,?)>, !flir.layout<(?,?)>) -> !flir.layout<(?,?)>
  
  // Extract and verify shape is preserved (should be shape_b)
  %result_shape = flir.get_shape %composed : !flir.layout<(?,?)> -> !flir.shape<(?,?)>
  %rank = flir.rank %result_shape : !flir.shape<(?,?)> -> index
  
  return
}
