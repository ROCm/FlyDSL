// Test layout operations for AMD GFX942

module attributes {
  rocir.target_arch = "gfx942",
  rocir.target_vendor = "amd"
} {
  
  func.func @test_make_layout_amd() -> !rocir.layout<(?,?)> {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %shape = rocir.make_shape %c32, %c32 : (index, index) -> !rocir.shape<(?,?)>
    %stride = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<(?,?)>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<(?,?)>
    return %layout : !rocir.layout<(?,?)>
  }
}
