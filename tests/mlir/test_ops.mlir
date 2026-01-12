module {
  func.func @test_flir_ops(%i1: index, %i2: index, %i3: index) {
    %s = flir.make_shape %i1, %i2, %i3 : (index, index, index) -> !flir.shape<(?,?,?)>
    %st = flir.make_stride %i1, %i2, %i3 : (index, index, index) -> !flir.stride<(?,?,?)>
    %l = flir.make_layout %s, %st : (!flir.shape<(?,?,?)>, !flir.stride<(?,?,?)>) -> !flir.layout<(?,?,?):(?,?,?)>
    %c = flir.make_coord %i1, %i2 : (index, index) -> !flir.coord<(?,?)>
    %size = flir.size %s : !flir.shape<(?,?,?)> -> index
    %idx = flir.crd2idx %c, %l : (!flir.coord<(?,?)>, !flir.layout<(?,?,?):(?,?,?)>) -> index
    return
  }
}
