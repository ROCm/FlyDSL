func.func @test1() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %extracted_shape = cute.get_shape %layout : !cute.layout<2> -> !cute.shape<2>
  %size = cute.size %extracted_shape : !cute.shape<2> -> index
  
  return
}
