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
    %coord = rocir.make_coord %c0, %c1 : (index, index) -> !rocir.coord<(?,?)>
    %stride = rocir.make_stride %s0, %s1 : (index, index) -> !rocir.stride<(?,?)>
    %shape = rocir.make_shape %c0, %c1 : (index, index) -> !rocir.shape<(?,?)>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    // Compute linear index: 2*1 + 3*16 = 2 + 48 = 50
    %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<(?,?)>, !rocir.layout<(?,?)>) -> index
    
    return %idx : index
  }
}
