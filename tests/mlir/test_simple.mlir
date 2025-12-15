func.func @test_idx2crd() {
  %c50 = arith.constant 50 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = rocir.make_shape %c8, %c128 : (index, index) -> !rocir.shape<(?,?)>
  %stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<(?,?)>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
  
  %coord = rocir.idx2crd %c50, %layout : (index, !rocir.layout<(?,?)>) -> !rocir.coord<(?,?)>
  
  return
}
