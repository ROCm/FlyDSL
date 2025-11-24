// Test chained operations: get_shape -> size, get_stride -> get
// This tests the value tracking fix

func.func @test_get_shape_then_size() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Extract shape from layout
  %extracted_shape = cute.get_shape %layout : !cute.layout<2> -> !cute.shape<2>
  
  // CRITICAL TEST: Can we compute size from extracted shape?
  %size = cute.size %extracted_shape : !cute.shape<2> -> index
  
  return
}

func.func @test_get_stride_then_get() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  
  %shape = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Extract stride from layout
  %extracted_stride = cute.get_stride %layout : !cute.layout<2> -> !cute.stride<2>
  
  // CRITICAL TEST: Can we get element from extracted stride?
  %stride0 = cute.get %extracted_stride, %c0 : (!cute.stride<2>, index) -> index
  
  return
}
