// Test flir.size operation
func.func @test_size() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = flir.make_shape %c8, %c16, %c32 : (index, index, index) -> !flir.shape<(?,?,?)>
  %size = flir.size %shape : !flir.shape<(?,?,?)> -> index
  
  return %size : index
}
