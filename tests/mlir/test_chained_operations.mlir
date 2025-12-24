// Test chained operations: get_shape -> size, get_stride -> get
// This tests the value tracking fix

func.func @test_get_shape_then_size() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Extract shape from layout
  %extracted_shape = flir.get_shape %layout : !flir.layout<(?,?)> -> !flir.shape<(?,?)>
  
  // CRITICAL TEST: Can we compute size from extracted shape?
  %size = flir.size %extracted_shape : !flir.shape<(?,?)> -> index
  
  return
}

func.func @test_get_stride_then_get() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  
  %shape = flir.make_shape %c8, %c16 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c8 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  // Extract stride from layout
  %extracted_stride = flir.get_stride %layout : !flir.layout<(?,?)> -> !flir.stride<(?,?)>
  
  // CRITICAL TEST: Can we get element from extracted stride?
  %stride0 = flir.get %extracted_stride, %c0 : !flir.stride<(?,?)>, index -> index
  
  return
}
