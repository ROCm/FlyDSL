// Test basic type parsing and printing

module {
  func.func @test_types(%i1: index, %i2: index) -> !flir.layout<(?,?):(?,?)> {
    %shape = flir.make_shape %i1, %i2 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %i1, %i2 : (index, index) -> !flir.stride<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    return %layout : !flir.layout<(?,?):(?,?)>
  }
  
  func.func @test_index_type(%i: index) -> index {
    return %i : index
  }
  
  func.func @test_all_types(%s: !flir.shape<(?,?,?)>, %st: !flir.stride<(?,?,?)>, 
                            %l: !flir.layout<(?,?):(?,?)>, %c: !flir.coord<(?,?)>) {
    return
  }
}
