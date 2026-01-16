// Test layout operations for AMD GFX942

module attributes {
  flir.target_arch = "gfx942",
  flir.target_vendor = "amd"
} {
  
  func.func @test_make_layout_amd() -> !flir.layout<(?,?):(?,?)> {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %shape = flir.make_shape %c32, %c32 : (index, index) -> !flir.shape<(?,?)>
    %stride = flir.make_stride %c1, %c32 : (index, index) -> !flir.stride<(?,?)>
    %layout = flir.make_layout %shape, %stride : (!flir.shape<(?,?)>, !flir.stride<(?,?)>) -> !flir.layout<(?,?):(?,?)>
    return %layout : !flir.layout<(?,?):(?,?)>
  }
}
