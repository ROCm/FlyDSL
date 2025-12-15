// Test rocir.rank operation
func.func @test_rank() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = rocir.make_shape %c8, %c16, %c32 : (index, index, index) -> !rocir.shape<(?,?,?)>
  %rank = rocir.rank %shape : !rocir.shape<(?,?,?)> -> index
  
  return %rank : index
}
