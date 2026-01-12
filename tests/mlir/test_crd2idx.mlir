// Test crd2idx lowering
module {
  func.func @test_crd2idx() -> index {
    // Create coord values: coord = (2, 3)  
    %c0 = arith.constant 2 : index
    %c1 = arith.constant 3 : index
    
    // Create stride values: stride = (1, 16)
    %s0 = arith.constant 1 : index
    %s1 = arith.constant 16 : index
    
    // Make coord and layout
    %coord = flir.make_coord %c0, %c1 : (index, index) -> !flir.coord<(?,?)>
    %stride = flir.make_stride %s0, %s1 : (index, index) -> !flir.stride<(?,?)>
    %shape = flir.make_shape %c0, %c1 : (index, index) -> !flir.shape<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    
    // Compute linear index: 2*1 + 3*16 = 2 + 48 = 50
    %idx = flir.crd2idx %coord, %layout : (!flir.coord<(?,?)>, !flir.layout<(?,?):(?,?)>) -> index
    
    return %idx : index
  }
}
