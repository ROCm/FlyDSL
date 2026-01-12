// Test partial pass lowering (crd2idx only)
module {
  func.func @test_crd2idx_simple(%c: !flir.coord<(?,?)>, %l: !flir.layout<(?,?):(?,?)>) {
    %idx = flir.crd2idx %c, %l : (!flir.coord<(?,?)>, !flir.layout<(?,?):(?,?)>) -> index
    // Note: %idx is lowered to index type but not used, so no type conversion error
    return
  }
}
