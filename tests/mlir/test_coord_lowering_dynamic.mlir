// Test coordinate lowering with dynamic values (no constant folding)

module {
  func.func @test_crd2idx_dynamic(%arg0: index, %arg1: index) -> index {
    // Use function arguments to prevent constant folding
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    
    %shape = flir.make_shape %c32, %c64 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %c64, %c1 : (index, index) -> !flir.stride<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
    
    // Create coordinate from dynamic arguments
    %coord = flir.make_coord %arg0, %arg1 : (index, index) -> !flir.coord<(?,?)>
    
    // Convert to linear index
    %idx = flir.crd2idx %coord, %layout : (!flir.coord<(?,?)>, !flir.layout<(?,?)>) -> index
    
    return %idx : index
  }
}
