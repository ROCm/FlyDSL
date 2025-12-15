// Test chained operations: get_shape -> size, get_stride -> get
// This tests the value tracking fix

func.func @test_get_shape_then_size() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Extract shape from layout
  %extracted_shape = rocir.get_shape %layout : !rocir.layout<(?,?)> -> !rocir.shape<(?,?)>
  
  // CRITICAL TEST: Can we compute size from extracted shape?
  %size = rocir.size %extracted_shape : !rocir.shape<(?,?)> -> index
  
  return
}

func.func @test_get_stride_then_get() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  
  %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  // Extract stride from layout
  %extracted_stride = rocir.get_stride %layout : !rocir.layout<(?,?)> -> !rocir.stride<(?,?)>
  
  // CRITICAL TEST: Can we get element from extracted stride?
  %stride0 = rocir.get %extracted_stride, %c0 : !rocir.stride<(?,?)>, index -> index
  
  return
}
