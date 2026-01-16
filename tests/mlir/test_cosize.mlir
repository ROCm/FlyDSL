// Test flir.cosize operation
func.func @test_cosize() -> index {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = flir.make_shape %c8, %c128 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
  
  %cosize = flir.cosize %layout : !flir.layout<(?,?):(?,?)> -> index
  
  return %cosize : index
}
