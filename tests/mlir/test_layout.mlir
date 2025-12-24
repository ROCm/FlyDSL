// Test basic layout creation and manipulation

module {
  func.func @test_make_layout(%i8: index, %i16: index, %i1: index) -> !flir.layout<(?,?)> {
    // Create shape and stride
    %shape = flir.make_shape %i8, %i16 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %i1, %i8 : (index, index) -> !flir.stride<(?,?)>
    
    // Create layout from shape and stride
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
    
    return %layout : !flir.layout<(?,?)>
  }
  
  func.func @test_layout_size(%arg0: !flir.shape<(?,?)>) -> index {
    // Query layout size
    %size = flir.size %arg0 : !flir.shape<(?,?)> -> index
    
    return %size : index
  }
  
  func.func @test_coord_to_index(%layout: !flir.layout<(?,?)>, %i3: index, %i5: index) -> index {
    %coord = flir.make_coord %i3, %i5 : (index, index) -> !flir.coord<(?,?)>
    
    // Convert coordinate to linear index
    %idx = flir.crd2idx %coord, %layout : (!flir.coord<(?,?)>, !flir.layout<(?,?)>) -> index
    
    return %idx : index
  }
}
