func.func @test1() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %extracted_shape = flir.get_shape %layout : !flir.layout<(?,?):(?,?)> -> !flir.shape<(?,?)>
  %size = flir.size %extracted_shape : !flir.shape<(?,?)> -> index
  
  return
}
