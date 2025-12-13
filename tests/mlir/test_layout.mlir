// Test basic layout creation and manipulation

module {
  func.func @test_make_layout(%i8: index, %i16: index, %i1: index) -> !rocir.layout<(?,?)> {
    // Create shape and stride
    %shape = rocir.make_shape %i8, %i16 : (index, index) -> !rocir.shape<(?,?)>
    %stride = rocir.make_stride %i1, %i8 : (index, index) -> !rocir.stride<(?,?)>
    
    // Create layout from shape and stride
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    
    return %layout : !rocir.layout<(?,?)>
  }
  
  func.func @test_layout_size(%arg0: !rocir.shape<(?,?)>) -> index {
    // Query layout size
    %size = rocir.size %arg0 : !rocir.shape<(?,?)> -> index
    
    return %size : index
  }
  
  func.func @test_coord_to_index(%layout: !rocir.layout<(?,?)>, %i3: index, %i5: index) -> index {
    %coord = rocir.make_coord %i3, %i5 : (index, index) -> !rocir.coord<(?,?)>
    
    // Convert coordinate to linear index
    %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<(?,?)>, !rocir.layout<(?,?)>) -> index
    
    return %idx : index
  }
}
