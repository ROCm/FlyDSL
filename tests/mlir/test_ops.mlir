module {
  func.func @test_rocir_ops(%i1: index, %i2: index, %i3: index) {
    %s = rocir.make_shape %i1, %i2, %i3 : (index, index, index) -> !rocir.shape<(?,?,?)>
    %st = rocir.make_stride %i1, %i2, %i3 : (index, index, index) -> !rocir.stride<(?,?,?)>
    %l = rocir.make_layout %s, %st : (!rocir.shape<(?,?,?)>, !rocir.stride<(?,?,?)>) -> !rocir.layout<(?,?,?)>
    %c = rocir.make_coord %i1, %i2 : (index, index) -> !rocir.coord<(?,?)>
    %size = rocir.size %s : !rocir.shape<(?,?,?)> -> index
    %idx = rocir.crd2idx %c, %l : (!rocir.coord<(?,?)>, !rocir.layout<(?,?,?)>) -> index
    return
  }
}
