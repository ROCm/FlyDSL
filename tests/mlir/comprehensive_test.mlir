// Comprehensive test of all working Flir operations

// Test 1: size - Product of shape dimensions
func.func @test_size() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = flir.make_shape %c8, %c16, %c32 : (index, index, index) -> !flir.shape<(?,?,?)>
  %size = flir.size %shape : !flir.shape<(?,?,?)> -> index
  // Expected lowering: %size = 8 * 16 * 32 = 4096
  
  return %size : index
}

// Test 2: rank - Number of dimensions
func.func @test_rank() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = flir.make_shape %c8, %c16, %c32 : (index, index, index) -> !flir.shape<(?,?,?)>
  %rank = flir.rank %shape : !flir.shape<(?,?,?)> -> index
  // Expected lowering: %rank = constant 3
  
  return %rank : index
}

// Test 3: cosize - Codomain size (span of mapped indices)
func.func @test_cosize() -> index {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = flir.make_shape %c8, %c128 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %cosize = flir.cosize %layout : !flir.layout<(?,?)> -> index
  // Expected lowering: max((8-1)*1, (128-1)*16) + 1 = max(7, 2032) + 1 = 2033
  
  return %cosize : index
}

// Test 4: crd2idx - Convert coordinate to linear index
func.func @test_crd2idx() -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  
  %coord = flir.make_coord %c2, %c3 : (index, index) -> !flir.coord<(?,?)>
  %shape = flir.make_shape %c8, %c128 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %idx = flir.crd2idx %coord, %layout : (!flir.coord<(?,?)>, !flir.layout<(?,?)>) -> index
  // Expected lowering: idx = 2*1 + 3*16 = 2 + 48 = 50
  
  return %idx : index
}

// Test 5: Layout size (using size on layout instead of shape)
func.func @test_layout_size() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c8_1 = arith.constant 8 : index
  
  %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c8_1 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %size = flir.size %layout : !flir.layout<(?,?)> -> index
  // Expected lowering: size = 8 * 16 = 128
  
  return %size : index
}
