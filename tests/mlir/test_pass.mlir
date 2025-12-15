// Test partial pass lowering (crd2idx only)
module {
  func.func @test_crd2idx_simple(%c: !rocir.coord<(?,?)>, %l: !rocir.layout<(?,?)>) {
    %idx = rocir.crd2idx %c, %l : (!rocir.coord<(?,?)>, !rocir.layout<(?,?)>) -> index
    // Note: %idx is lowered to index type but not used, so no type conversion error
    return
  }
}
