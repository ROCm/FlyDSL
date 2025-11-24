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
    %coord = cute.make_coord %c0, %c1 : (index, index) -> !cute.coord<2>
    %stride = cute.make_stride %s0, %s1 : (index, index) -> !cute.stride<2>
    %shape = cute.make_shape %c0, %c1 : (index, index) -> !cute.shape<2>
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    // Compute linear index: 2*1 + 3*16 = 2 + 48 = 50
    %idx = cute.crd2idx %coord, %layout : (!cute.coord<2>, !cute.layout<2>) -> index
    
    return %idx : index
  }
}
