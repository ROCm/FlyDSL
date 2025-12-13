// Test coordinate lowering with dynamic values (no constant folding)

module {
  func.func @test_crd2idx_dynamic(%arg0: index, %arg1: index) -> index {
    // Use function arguments to prevent constant folding
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    
    %shape = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<(?,?)>
    %stride = rocir.make_stride %c64, %c1 : (index, index) -> !rocir.stride<(?,?)>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // Create coordinate from dynamic arguments
    %coord = rocir.make_coord %arg0, %arg1 : (index, index) -> !rocir.coord<(?,?)>
    
    // Convert to linear index
    %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<(?,?)>, !rocir.layout<(?,?)>) -> index
    
    return %idx : index
  }
}
