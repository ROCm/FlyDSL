// Test flir.get operation
func.func @test_get() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  
  // Create shape (8, 16, 32)
  %shape = flir.make_shape %c8, %c16, %c32 : (index, index, index) -> !flir.shape<(?,?,?)>
  
  // Get element at index 1 (should return 16)
  %elem = flir.get %shape, %c1 : !flir.shape<(?,?,?)>, index -> index
  
  return %elem : index
}
