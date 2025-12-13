func.func @test1() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %extracted_shape = rocir.get_shape %layout : !rocir.layout<(?,?)> -> !rocir.shape<(?,?)>
  %size = rocir.size %extracted_shape : !rocir.shape<(?,?)> -> index
  
  return
}
