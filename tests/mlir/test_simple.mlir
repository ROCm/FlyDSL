func.func @test_idx2crd() {
  %c50 = arith.constant 50 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = flir.make_shape %c8, %c128 : (index, index) -> !flir.shape<(?,?)>
  %stride = flir.make_stride %c1, %c16 : (index, index) -> !flir.stride<(?,?)>
  %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?)>
  
  %coord = flir.idx2crd %c50, %layout : (index, !flir.layout<(?,?)>) -> !flir.coord<(?,?)>
  
  return
}
