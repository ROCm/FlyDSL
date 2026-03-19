module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @vectorAddKernel_0(%arg0: !fly.memref<f32, global, ?:1,  align<4>>, %arg1: !fly.memref<f32, global, 128:1,  align<4>>, %arg2: !fly.memref<f32, global, 128:1,  align<4>>) kernel {
      %block_id_x = gpu.block_id  x
      %0 = arith.index_cast %block_id_x : index to i32
      %thread_id_x = gpu.thread_id  x
      %1 = arith.index_cast %thread_id_x : index to i32
      %2 = fly.make_shape() : () -> !fly.int_tuple<64>
      %3 = fly.make_stride() : () -> !fly.int_tuple<1>
      %4 = fly.make_layout(%2, %3) : (!fly.int_tuple<64>, !fly.int_tuple<1>) -> !fly.layout<64:1>
      %5 = fly.logical_divide(%arg0, %4) : (!fly.memref<f32, global, ?:1,  align<4>>, !fly.layout<64:1>) -> !fly.memref<f32, global, (64,?):(1,64),  align<4>>
      %6 = fly.make_shape() : () -> !fly.int_tuple<64>
      %7 = fly.make_stride() : () -> !fly.int_tuple<1>
      %8 = fly.make_layout(%6, %7) : (!fly.int_tuple<64>, !fly.int_tuple<1>) -> !fly.layout<64:1>
      %9 = fly.logical_divide(%arg1, %8) : (!fly.memref<f32, global, 128:1,  align<4>>, !fly.layout<64:1>) -> !fly.memref<f32, global, (64,2):(1,64),  align<4>>
      %10 = fly.make_shape() : () -> !fly.int_tuple<64>
      %11 = fly.make_stride() : () -> !fly.int_tuple<1>
      %12 = fly.make_layout(%10, %11) : (!fly.int_tuple<64>, !fly.int_tuple<1>) -> !fly.layout<64:1>
      %13 = fly.logical_divide(%arg2, %12) : (!fly.memref<f32, global, 128:1,  align<4>>, !fly.layout<64:1>) -> !fly.memref<f32, global, (64,2):(1,64),  align<4>>
      %14 = fly.make_coord(%0) : (i32) -> !fly.int_tuple<(*,?)>
      %15 = fly.slice(%5, %14) : (!fly.memref<f32, global, (64,?):(1,64),  align<4>>, !fly.int_tuple<(*,?)>) -> !fly.memref<f32, global, 64:1,  align<4>>
      %16 = fly.make_coord(%0) : (i32) -> !fly.int_tuple<(*,?)>
      %17 = fly.slice(%9, %16) : (!fly.memref<f32, global, (64,2):(1,64),  align<4>>, !fly.int_tuple<(*,?)>) -> !fly.memref<f32, global, 64:1,  align<4>>
      %18 = fly.make_coord(%0) : (i32) -> !fly.int_tuple<(*,?)>
      %19 = fly.slice(%13, %18) : (!fly.memref<f32, global, (64,2):(1,64),  align<4>>, !fly.int_tuple<(*,?)>) -> !fly.memref<f32, global, 64:1,  align<4>>
      %20 = fly.make_shape() : () -> !fly.int_tuple<1>
      %21 = fly.make_stride() : () -> !fly.int_tuple<1>
      %22 = fly.make_layout(%20, %21) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %23 = fly.logical_divide(%15, %22) : (!fly.memref<f32, global, 64:1,  align<4>>, !fly.layout<1:1>) -> !fly.memref<f32, global, (1,64):(0,1),  align<4>>
      %24 = fly.make_shape() : () -> !fly.int_tuple<1>
      %25 = fly.make_stride() : () -> !fly.int_tuple<1>
      %26 = fly.make_layout(%24, %25) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %27 = fly.logical_divide(%17, %26) : (!fly.memref<f32, global, 64:1,  align<4>>, !fly.layout<1:1>) -> !fly.memref<f32, global, (1,64):(0,1),  align<4>>
      %28 = fly.make_shape() : () -> !fly.int_tuple<1>
      %29 = fly.make_stride() : () -> !fly.int_tuple<1>
      %30 = fly.make_layout(%28, %29) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %31 = fly.logical_divide(%19, %30) : (!fly.memref<f32, global, 64:1,  align<4>>, !fly.layout<1:1>) -> !fly.memref<f32, global, (1,64):(0,1),  align<4>>
      %32 = fly.make_atom : () -> !fly.atom.universal_copy<32>
      %33 = fly.make_shape() : () -> !fly.int_tuple<1>
      %34 = fly.make_stride() : () -> !fly.int_tuple<1>
      %35 = fly.make_layout(%33, %34) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %36 = fly.memref.alloca(%35) : (!fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
      %37 = fly.make_shape() : () -> !fly.int_tuple<1>
      %38 = fly.make_stride() : () -> !fly.int_tuple<1>
      %39 = fly.make_layout(%37, %38) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %40 = fly.memref.alloca(%39) : (!fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
      %41 = fly.make_shape() : () -> !fly.int_tuple<1>
      %42 = fly.make_stride() : () -> !fly.int_tuple<1>
      %43 = fly.make_layout(%41, %42) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %44 = fly.memref.alloca(%43) : (!fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
      %45 = fly.make_coord(%1) : (i32) -> !fly.int_tuple<(*,?)>
      %46 = fly.slice(%23, %45) : (!fly.memref<f32, global, (1,64):(0,1),  align<4>>, !fly.int_tuple<(*,?)>) -> !fly.memref<f32, global, 1:0,  align<4>>
      fly.copy_atom_call(%32, %46, %36) : (!fly.atom.universal_copy<32>, !fly.memref<f32, global, 1:0,  align<4>>, !fly.memref<f32, register, 1:1>) -> ()
      %47 = fly.make_coord(%1) : (i32) -> !fly.int_tuple<(*,?)>
      %48 = fly.slice(%27, %47) : (!fly.memref<f32, global, (1,64):(0,1),  align<4>>, !fly.int_tuple<(*,?)>) -> !fly.memref<f32, global, 1:0,  align<4>>
      fly.copy_atom_call(%32, %48, %40) : (!fly.atom.universal_copy<32>, !fly.memref<f32, global, 1:0,  align<4>>, !fly.memref<f32, register, 1:1>) -> ()
      %49 = fly.memref.load_vec(%36) : (!fly.memref<f32, register, 1:1>) -> vector<1xf32>
      %50 = fly.memref.load_vec(%40) : (!fly.memref<f32, register, 1:1>) -> vector<1xf32>
      %51 = arith.addf %49, %50 : vector<1xf32>
      fly.memref.store_vec(%51, %44) : (vector<1xf32>, !fly.memref<f32, register, 1:1>) -> ()
      %52 = fly.make_coord(%1) : (i32) -> !fly.int_tuple<(*,?)>
      %53 = fly.slice(%31, %52) : (!fly.memref<f32, global, (1,64):(0,1),  align<4>>, !fly.int_tuple<(*,?)>) -> !fly.memref<f32, global, 1:0,  align<4>>
      fly.copy_atom_call(%32, %44, %53) : (!fly.atom.universal_copy<32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, global, 1:0,  align<4>>) -> ()
      gpu.return
    }
  }
  func.func @vectorAdd(%arg0: !fly.memref<f32, global, ?:1,  align<4>>, %arg1: !fly.memref<f32, global, 128:1,  align<4>>, %arg2: !fly.memref<f32, global, 128:1,  align<4>>, %arg3: i32, %arg4: !gpu.async.token) attributes {llvm.emit_c_interface} {
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.addi %arg3, %c64_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %2 = arith.floordivsi %1, %c64_i32_0 : i32
    %3 = arith.index_cast %2 : i32 to index
    %c1 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch_func <%arg4 : !gpu.async.token> @kernels::@vectorAddKernel_0 blocks in (%3, %c1, %c1_1) threads in (%c64, %c1_2, %c1_3)  args(%arg0 : !fly.memref<f32, global, ?:1,  align<4>>, %arg1 : !fly.memref<f32, global, 128:1,  align<4>>, %arg2 : !fly.memref<f32, global, 128:1,  align<4>>)
    return
  }
}
