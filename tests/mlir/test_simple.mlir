func.func @test_idx2crd() {
  %c50 = arith.constant 50 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = rocir.make_shape %c8, %c128 : (index, index) -> !rocir.shape<-1>
  %stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<-1>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<-1>, !rocir.stride<-1>) -> !rocir.layout<-1>
  
  %coord = rocir.idx2crd %c50, %layout : (index, !rocir.layout<-1>) -> !rocir.coord<-1>
  
  return
}
