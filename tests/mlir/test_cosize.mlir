// Test rocir.cosize operation
func.func @test_cosize() -> index {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = rocir.make_shape %c8, %c128 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %cosize = rocir.cosize %layout : !rocir.layout<(?,?)> -> index
  
  return %cosize : index
}
