module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @vectorAddKernel_0(%arg0: !fly.ptr<f32, global,  align<4>>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !fly.ptr<f32, global,  align<4>>, %arg3: !fly.ptr<f32, global,  align<4>>) kernel {
      %c64_i32 = arith.constant 64 : i32
      %block_id_x = gpu.block_id  x
      %0 = arith.index_cast %block_id_x : index to i32
      %thread_id_x = gpu.thread_id  x
      %1 = arith.index_cast %thread_id_x : index to i32
      %2 = arith.muli %0, %c64_i32 : i32
      %3 = fly.make_int_tuple(%2) : (i32) -> !fly.int_tuple<?{div=64}>
      %4 = fly.add_offset(%arg0, %3) : (!fly.ptr<f32, global,  align<4>>, !fly.int_tuple<?{div=64}>) -> !fly.ptr<f32, global,  align<4>>
      %5 = arith.muli %0, %c64_i32 : i32
      %6 = fly.make_int_tuple(%5) : (i32) -> !fly.int_tuple<?{div=64}>
      %7 = fly.add_offset(%arg2, %6) : (!fly.ptr<f32, global,  align<4>>, !fly.int_tuple<?{div=64}>) -> !fly.ptr<f32, global,  align<4>>
      %8 = arith.muli %0, %c64_i32 : i32
      %9 = fly.make_int_tuple(%8) : (i32) -> !fly.int_tuple<?{div=64}>
      %10 = fly.add_offset(%arg3, %9) : (!fly.ptr<f32, global,  align<4>>, !fly.int_tuple<?{div=64}>) -> !fly.ptr<f32, global,  align<4>>
      %11 = fly.make_atom : () -> !fly.atom.universal_copy<32>
      %12 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %13 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %14 = fly.make_layout(%12, %13) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %15 = fly.memref.alloca(%14) : (!fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
      %16 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %17 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %18 = fly.make_layout(%16, %17) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %19 = fly.memref.alloca(%18) : (!fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
      %20 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %21 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %22 = fly.make_layout(%20, %21) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
      %23 = fly.memref.alloca(%22) : (!fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
      %24 = fly.make_int_tuple(%1) : (i32) -> !fly.int_tuple<?>
      %25 = fly.add_offset(%4, %24) : (!fly.ptr<f32, global,  align<4>>, !fly.int_tuple<?>) -> !fly.ptr<f32, global,  align<4>>
      %26 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %27 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
      %28 = fly.make_layout(%26, %27) : (!fly.int_tuple<1>, !fly.int_tuple<0>) -> !fly.layout<1:0>
      %29 = fly.make_view(%25, %28) : (!fly.ptr<f32, global,  align<4>>, !fly.layout<1:0>) -> !fly.memref<f32, global, 1:0,  align<4>>
      fly.copy_atom_call(%11, %29, %15) : (!fly.atom.universal_copy<32>, !fly.memref<f32, global, 1:0,  align<4>>, !fly.memref<f32, register, 1:1>) -> ()
      %30 = fly.make_int_tuple(%1) : (i32) -> !fly.int_tuple<?>
      %31 = fly.add_offset(%7, %30) : (!fly.ptr<f32, global,  align<4>>, !fly.int_tuple<?>) -> !fly.ptr<f32, global,  align<4>>
      %32 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %33 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
      %34 = fly.make_layout(%32, %33) : (!fly.int_tuple<1>, !fly.int_tuple<0>) -> !fly.layout<1:0>
      %35 = fly.make_view(%31, %34) : (!fly.ptr<f32, global,  align<4>>, !fly.layout<1:0>) -> !fly.memref<f32, global, 1:0,  align<4>>
      fly.copy_atom_call(%11, %35, %19) : (!fly.atom.universal_copy<32>, !fly.memref<f32, global, 1:0,  align<4>>, !fly.memref<f32, register, 1:1>) -> ()
      %36 = fly.memref.load_vec(%15) : (!fly.memref<f32, register, 1:1>) -> vector<1xf32>
      %37 = fly.memref.load_vec(%19) : (!fly.memref<f32, register, 1:1>) -> vector<1xf32>
      %38 = arith.addf %36, %37 : vector<1xf32>
      fly.memref.store_vec(%38, %23) : (vector<1xf32>, !fly.memref<f32, register, 1:1>) -> ()
      %39 = fly.make_int_tuple(%1) : (i32) -> !fly.int_tuple<?>
      %40 = fly.add_offset(%10, %39) : (!fly.ptr<f32, global,  align<4>>, !fly.int_tuple<?>) -> !fly.ptr<f32, global,  align<4>>
      %41 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
      %42 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
      %43 = fly.make_layout(%41, %42) : (!fly.int_tuple<1>, !fly.int_tuple<0>) -> !fly.layout<1:0>
      %44 = fly.make_view(%40, %43) : (!fly.ptr<f32, global,  align<4>>, !fly.layout<1:0>) -> !fly.memref<f32, global, 1:0,  align<4>>
      fly.copy_atom_call(%11, %23, %44) : (!fly.atom.universal_copy<32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, global, 1:0,  align<4>>) -> ()
      gpu.return
    }
  }
  func.func @vectorAdd(%arg0: !fly.ptr<f32, global,  align<4>>, %arg1: !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2: !fly.ptr<f32, global,  align<4>>, %arg3: !fly.ptr<f32, global,  align<4>>, %arg4: i32, %arg5: !gpu.async.token) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>
    %1 = llvm.mlir.undef : !llvm.struct<packed ()>
    %2 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %3 = llvm.extractvalue %arg1[0, 0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %4 = arith.addi %arg4, %c64_i32 : i32
    %5 = arith.subi %4, %c1_i32 : i32
    %6 = arith.floordivsi %5, %c64_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = llvm.insertvalue %3, %2[0] : !llvm.struct<packed (i32)> 
    %9 = llvm.insertvalue %8, %0[0] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    %10 = llvm.insertvalue %1, %9[1] : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)> 
    gpu.launch_func <%arg5 : !gpu.async.token> @kernels::@vectorAddKernel_0 blocks in (%7, %c1, %c1) threads in (%c64, %c1, %c1)  args(%arg0 : !fly.ptr<f32, global,  align<4>>, %10 : !llvm.struct<packed (struct<packed (i32)>, struct<packed ()>)>, %arg2 : !fly.ptr<f32, global,  align<4>>, %arg3 : !fly.ptr<f32, global,  align<4>>)
    return
  }
}
