; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@gemm2port_smem = external addrspace(3) global [32768 x i8], align 1024

define amdgpu_kernel void @gemm2_a4w4_port_bm32_atomic(ptr addrspace(1) %0, <{ <{ i32, i32 }>, <{ i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32 }>, <{ i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32 }>, <{ i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32 }> }> %9, ptr addrspace(1) %10, <{ <{ i32 }> }> %11, ptr addrspace(1) %12, <{ <{ i32 }> }> %13, ptr addrspace(1) %14, <{ <{ i32 }> }> %15, i32 %16, ptr addrspace(1) %17, <{ <{ i32, i32 }>, <{ i64 }> }> %18) #0 !reqd_work_group_size !1 {
  %20 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  %21 = sext i32 %20 to i64
  %22 = call i32 @llvm.amdgcn.workgroup.id.x()
  %23 = sext i32 %22 to i64
  %24 = trunc i64 %21 to i32
  %25 = trunc i64 %23 to i32
  %26 = srem i32 %24, 64
  %27 = sdiv i32 %24, 64
  %28 = mul i32 %27, 64
  %29 = icmp ne i32 %24, %28
  %30 = icmp slt i32 %24, 0
  %31 = icmp ne i1 %30, false
  %32 = and i1 %29, %31
  %33 = add i32 %27, -1
  %34 = select i1 %32, i32 %33, i32 %27
  %35 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %34)
  %36 = ptrtoint ptr addrspace(1) %10 to i64
  %37 = inttoptr i64 %36 to ptr addrspace(1)
  %38 = load i32, ptr addrspace(1) %37, align 4
  %39 = sdiv i32 %38, 32
  %40 = mul i32 %39, 32
  %41 = icmp ne i32 %38, %40
  %42 = icmp slt i32 %38, 0
  %43 = icmp ne i1 %42, false
  %44 = and i1 %41, %43
  %45 = add i32 %39, -1
  %46 = select i1 %44, i32 %45, i32 %39
  %47 = mul i32 %46, 28
  %48 = icmp slt i32 %25, %47
  br i1 %48, label %49, label %844

49:                                               ; preds = %19
  %50 = srem i32 %25, 28
  %51 = sdiv i32 %25, 28
  %52 = mul i32 %51, 28
  %53 = icmp ne i32 %25, %52
  %54 = icmp slt i32 %25, 0
  %55 = icmp ne i1 %54, false
  %56 = and i1 %53, %55
  %57 = add i32 %51, -1
  %58 = select i1 %56, i32 %57, i32 %51
  %59 = mul i32 %58, 4
  %60 = ptrtoint ptr addrspace(1) %8 to i64
  %61 = sext i32 %59 to i64
  %62 = add i64 %60, %61
  %63 = inttoptr i64 %62 to ptr addrspace(1)
  %64 = load i32, ptr addrspace(1) %63, align 4
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64)
  %66 = mul i32 %58, 32
  %67 = addrspacecast ptr addrspace(1) %0 to ptr
  %68 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %67, i16 0, i64 167772160, i32 159744)
  %69 = addrspacecast ptr addrspace(1) %2 to ptr
  %70 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %69, i16 0, i64 10485760, i32 159744)
  %71 = addrspacecast ptr addrspace(1) %4 to ptr
  %72 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %71, i16 0, i64 706478080, i32 159744)
  %73 = addrspacecast ptr addrspace(1) %6 to ptr
  %74 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %73, i16 0, i64 44154880, i32 159744)
  %75 = sdiv i32 %26, 8
  %76 = mul i32 %75, 8
  %77 = icmp ne i32 %26, %76
  %78 = icmp slt i32 %26, 0
  %79 = icmp ne i1 %78, false
  %80 = and i1 %77, %79
  %81 = add i32 %75, -1
  %82 = select i1 %80, i32 %81, i32 %75
  %83 = srem i32 %26, 8
  %84 = sdiv i32 %26, 16
  %85 = mul i32 %84, 16
  %86 = icmp ne i32 %26, %85
  %87 = icmp slt i32 %26, 0
  %88 = icmp ne i1 %87, false
  %89 = and i1 %86, %88
  %90 = add i32 %84, -1
  %91 = select i1 %89, i32 %90, i32 %84
  %92 = srem i32 %26, 16
  %93 = mul i32 %65, 7168
  %94 = mul i32 %50, 256
  %95 = add i32 %93, %94
  %96 = mul i32 %35, 64
  %97 = add i32 %95, %96
  %98 = mul i32 %97, 256
  %99 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98)
  %100 = add i32 %97, 16
  %101 = mul i32 %100, 256
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101)
  %103 = add i32 %97, 32
  %104 = mul i32 %103, 256
  %105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104)
  %106 = add i32 %97, 48
  %107 = mul i32 %106, 256
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107)
  %109 = mul i32 %50, 8
  %110 = mul i32 %35, 2
  %111 = add i32 %109, %110
  %112 = mul i32 %65, 28672
  %113 = mul i32 %111, 128
  %114 = add i32 %112, %113
  %115 = mul i32 %114, 4
  %116 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115)
  %117 = add i32 %111, 1
  %118 = mul i32 %117, 128
  %119 = add i32 %112, %118
  %120 = mul i32 %119, 4
  %121 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120)
  %122 = sdiv i32 %66, 32
  %123 = mul i32 %122, 32
  %124 = icmp ne i32 %66, %123
  %125 = icmp slt i32 %66, 0
  %126 = icmp ne i1 %125, false
  %127 = and i1 %124, %126
  %128 = add i32 %122, -1
  %129 = select i1 %127, i32 %128, i32 %122
  %130 = mul i32 %129, 512
  %131 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %130)
  %132 = mul i32 %35, 8
  %133 = add i32 %66, %132
  %134 = add i32 %133, %82
  %135 = add i32 %132, %82
  %136 = and i32 %135, 14
  %137 = shl i32 %136, 3
  %138 = mul i32 %83, 16
  %139 = xor i32 %138, %137
  %140 = mul i32 %134, 256
  %141 = add i32 %139, %140
  %142 = mul i32 %35, 1024
  %143 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %142
  %144 = sext i32 %143 to i64
  %145 = inttoptr i64 %144 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %68, ptr addrspace(3) %145, i32 16, i32 %141, i32 0, i32 0, i32 0)
  %146 = add i32 %142, 4096
  %147 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %146
  %148 = sext i32 %147 to i64
  %149 = inttoptr i64 %148 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %68, ptr addrspace(3) %149, i32 16, i32 %141, i32 128, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %150 = mul i32 %91, 16
  %151 = add i32 %150, %92
  %152 = mul i32 %151, 4
  %153 = sdiv i32 %152, 4
  %154 = mul i32 %153, 4
  %155 = icmp ne i32 %152, %154
  %156 = icmp slt i32 %152, 0
  %157 = icmp ne i1 %156, false
  %158 = and i1 %155, %157
  %159 = add i32 %153, -1
  %160 = select i1 %158, i32 %159, i32 %153
  %161 = mul i32 %160, 4
  %162 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %70, i32 %161, i32 %131, i32 0)
  %163 = add i32 %152, 256
  %164 = sdiv i32 %163, 4
  %165 = mul i32 %164, 4
  %166 = icmp ne i32 %163, %165
  %167 = icmp slt i32 %163, 0
  %168 = icmp ne i1 %167, false
  %169 = and i1 %166, %168
  %170 = add i32 %164, -1
  %171 = select i1 %169, i32 %170, i32 %164
  %172 = mul i32 %171, 4
  %173 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %70, i32 %172, i32 %131, i32 0)
  %174 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %161, i32 %116, i32 0)
  %175 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %161, i32 %121, i32 0)
  %176 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %172, i32 %116, i32 0)
  %177 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %74, i32 %172, i32 %121, i32 0)
  %178 = mul i32 %91, 256
  %179 = mul i32 %92, 16
  %180 = add i32 %178, %179
  %181 = sdiv i32 %180, 4
  %182 = mul i32 %181, 4
  %183 = icmp ne i32 %180, %182
  %184 = icmp slt i32 %180, 0
  %185 = icmp ne i1 %184, false
  %186 = and i1 %183, %185
  %187 = add i32 %181, -1
  %188 = select i1 %186, i32 %187, i32 %181
  %189 = mul i32 %188, 4
  %190 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %99, i32 0)
  %191 = add i32 %180, 1024
  %192 = sdiv i32 %191, 4
  %193 = mul i32 %192, 4
  %194 = icmp ne i32 %191, %193
  %195 = icmp slt i32 %191, 0
  %196 = icmp ne i1 %195, false
  %197 = and i1 %194, %196
  %198 = add i32 %192, -1
  %199 = select i1 %197, i32 %198, i32 %192
  %200 = mul i32 %199, 4
  %201 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %99, i32 0)
  %202 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %102, i32 0)
  %203 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %102, i32 0)
  %204 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %105, i32 0)
  %205 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %105, i32 0)
  %206 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %189, i32 %108, i32 0)
  %207 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %200, i32 %108, i32 0)
  %208 = add i32 %180, 2048
  %209 = sdiv i32 %208, 4
  %210 = mul i32 %209, 4
  %211 = icmp ne i32 %208, %210
  %212 = icmp slt i32 %208, 0
  %213 = icmp ne i1 %212, false
  %214 = and i1 %211, %213
  %215 = add i32 %209, -1
  %216 = select i1 %214, i32 %215, i32 %209
  %217 = mul i32 %216, 4
  %218 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %99, i32 0)
  %219 = add i32 %180, 3072
  %220 = sdiv i32 %219, 4
  %221 = mul i32 %220, 4
  %222 = icmp ne i32 %219, %221
  %223 = icmp slt i32 %219, 0
  %224 = icmp ne i1 %223, false
  %225 = and i1 %222, %224
  %226 = add i32 %220, -1
  %227 = select i1 %225, i32 %226, i32 %220
  %228 = mul i32 %227, 4
  %229 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %99, i32 0)
  %230 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %102, i32 0)
  %231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %102, i32 0)
  %232 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %105, i32 0)
  %233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %105, i32 0)
  %234 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %217, i32 %108, i32 0)
  %235 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %72, i32 %228, i32 %108, i32 0)
  call void asm sideeffect "s_waitcnt vmcnt(23)", ""()
  call void asm sideeffect "s_barrier", ""()
  %236 = and i32 %92, 14
  %237 = shl i32 %236, 3
  %238 = sext i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32) to i64
  %239 = inttoptr i64 %238 to ptr addrspace(3)
  %240 = xor i32 %150, %237
  %241 = mul i32 %92, 128
  %242 = add i32 %241, %240
  %243 = getelementptr i8, ptr addrspace(3) %239, i32 %242
  %244 = load <4 x i32>, ptr addrspace(3) %243, align 16
  %245 = add i32 %92, 16
  %246 = mul i32 %245, 128
  %247 = add i32 %246, %240
  %248 = getelementptr i8, ptr addrspace(3) %239, i32 %247
  %249 = load <4 x i32>, ptr addrspace(3) %248, align 16
  %250 = add i32 %150, 64
  %251 = xor i32 %250, %237
  %252 = add i32 %241, %251
  %253 = getelementptr i8, ptr addrspace(3) %239, i32 %252
  %254 = load <4 x i32>, ptr addrspace(3) %253, align 16
  %255 = add i32 %246, %251
  %256 = getelementptr i8, ptr addrspace(3) %239, i32 %255
  %257 = load <4 x i32>, ptr addrspace(3) %256, align 16
  %258 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %190, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 0, i32 %174)
  %259 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %190, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 0, i32 %174)
  %260 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %254, <4 x i32> %201, <4 x float> %258, i32 4, i32 4, i32 2, i32 %162, i32 2, i32 %174)
  %261 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %257, <4 x i32> %201, <4 x float> %259, i32 4, i32 4, i32 3, i32 %162, i32 2, i32 %174)
  %262 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %202, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 1, i32 %174)
  %263 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %202, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 1, i32 %174)
  %264 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %254, <4 x i32> %203, <4 x float> %262, i32 4, i32 4, i32 2, i32 %162, i32 3, i32 %174)
  %265 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %257, <4 x i32> %203, <4 x float> %263, i32 4, i32 4, i32 3, i32 %162, i32 3, i32 %174)
  %266 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %204, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 0, i32 %175)
  %267 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %204, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 0, i32 %175)
  %268 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %254, <4 x i32> %205, <4 x float> %266, i32 4, i32 4, i32 2, i32 %162, i32 2, i32 %175)
  %269 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %257, <4 x i32> %205, <4 x float> %267, i32 4, i32 4, i32 3, i32 %162, i32 2, i32 %175)
  %270 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %244, <4 x i32> %206, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %162, i32 1, i32 %175)
  %271 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %249, <4 x i32> %206, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %162, i32 1, i32 %175)
  %272 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %254, <4 x i32> %207, <4 x float> %270, i32 4, i32 4, i32 2, i32 %162, i32 3, i32 %175)
  %273 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %257, <4 x i32> %207, <4 x float> %271, i32 4, i32 4, i32 3, i32 %162, i32 3, i32 %175)
  call void asm sideeffect "s_waitcnt vmcnt(22)", ""()
  call void asm sideeffect "s_barrier", ""()
  %274 = add i32 %241, 4096
  %275 = add i32 %274, %240
  %276 = getelementptr i8, ptr addrspace(3) %239, i32 %275
  %277 = load <4 x i32>, ptr addrspace(3) %276, align 16
  %278 = add i32 %246, 4096
  %279 = add i32 %278, %240
  %280 = getelementptr i8, ptr addrspace(3) %239, i32 %279
  %281 = load <4 x i32>, ptr addrspace(3) %280, align 16
  %282 = add i32 %274, %251
  %283 = getelementptr i8, ptr addrspace(3) %239, i32 %282
  %284 = load <4 x i32>, ptr addrspace(3) %283, align 16
  %285 = add i32 %278, %251
  %286 = getelementptr i8, ptr addrspace(3) %239, i32 %285
  %287 = load <4 x i32>, ptr addrspace(3) %286, align 16
  %288 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %277, <4 x i32> %218, <4 x float> %260, i32 4, i32 4, i32 0, i32 %173, i32 0, i32 %176)
  %289 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %281, <4 x i32> %218, <4 x float> %261, i32 4, i32 4, i32 1, i32 %173, i32 0, i32 %176)
  %290 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %284, <4 x i32> %229, <4 x float> %288, i32 4, i32 4, i32 2, i32 %173, i32 2, i32 %176)
  %291 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %287, <4 x i32> %229, <4 x float> %289, i32 4, i32 4, i32 3, i32 %173, i32 2, i32 %176)
  %292 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %277, <4 x i32> %230, <4 x float> %264, i32 4, i32 4, i32 0, i32 %173, i32 1, i32 %176)
  %293 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %281, <4 x i32> %230, <4 x float> %265, i32 4, i32 4, i32 1, i32 %173, i32 1, i32 %176)
  %294 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %284, <4 x i32> %231, <4 x float> %292, i32 4, i32 4, i32 2, i32 %173, i32 3, i32 %176)
  %295 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %287, <4 x i32> %231, <4 x float> %293, i32 4, i32 4, i32 3, i32 %173, i32 3, i32 %176)
  %296 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %277, <4 x i32> %232, <4 x float> %268, i32 4, i32 4, i32 0, i32 %173, i32 0, i32 %177)
  %297 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %281, <4 x i32> %232, <4 x float> %269, i32 4, i32 4, i32 1, i32 %173, i32 0, i32 %177)
  %298 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %284, <4 x i32> %233, <4 x float> %296, i32 4, i32 4, i32 2, i32 %173, i32 2, i32 %177)
  %299 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %287, <4 x i32> %233, <4 x float> %297, i32 4, i32 4, i32 3, i32 %173, i32 2, i32 %177)
  %300 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %277, <4 x i32> %234, <4 x float> %272, i32 4, i32 4, i32 0, i32 %173, i32 1, i32 %177)
  %301 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %281, <4 x i32> %234, <4 x float> %273, i32 4, i32 4, i32 1, i32 %173, i32 1, i32 %177)
  %302 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %284, <4 x i32> %235, <4 x float> %300, i32 4, i32 4, i32 2, i32 %173, i32 3, i32 %177)
  %303 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %287, <4 x i32> %235, <4 x float> %301, i32 4, i32 4, i32 3, i32 %173, i32 3, i32 %177)
  %304 = sext i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32) to i64
  %305 = inttoptr i64 %304 to ptr addrspace(3)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %306 = mul i32 %91, 4
  %307 = add i32 %96, %92
  %308 = mul i32 %91, 1024
  %309 = add i32 %308, %307
  %310 = extractelement <4 x float> %290, i64 0
  %311 = mul i32 %309, 4
  %312 = getelementptr i8, ptr addrspace(3) %305, i32 %311
  store float %310, ptr addrspace(3) %312, align 4
  %313 = add i32 %306, 1
  %314 = mul i32 %313, 256
  %315 = add i32 %314, %307
  %316 = extractelement <4 x float> %290, i64 1
  %317 = mul i32 %315, 4
  %318 = getelementptr i8, ptr addrspace(3) %305, i32 %317
  store float %316, ptr addrspace(3) %318, align 4
  %319 = add i32 %306, 2
  %320 = mul i32 %319, 256
  %321 = add i32 %320, %307
  %322 = extractelement <4 x float> %290, i64 2
  %323 = mul i32 %321, 4
  %324 = getelementptr i8, ptr addrspace(3) %305, i32 %323
  store float %322, ptr addrspace(3) %324, align 4
  %325 = add i32 %306, 3
  %326 = mul i32 %325, 256
  %327 = add i32 %326, %307
  %328 = extractelement <4 x float> %290, i64 3
  %329 = mul i32 %327, 4
  %330 = getelementptr i8, ptr addrspace(3) %305, i32 %329
  store float %328, ptr addrspace(3) %330, align 4
  %331 = add i32 %96, 16
  %332 = add i32 %331, %92
  %333 = add i32 %308, %332
  %334 = extractelement <4 x float> %294, i64 0
  %335 = mul i32 %333, 4
  %336 = getelementptr i8, ptr addrspace(3) %305, i32 %335
  store float %334, ptr addrspace(3) %336, align 4
  %337 = add i32 %314, %332
  %338 = extractelement <4 x float> %294, i64 1
  %339 = mul i32 %337, 4
  %340 = getelementptr i8, ptr addrspace(3) %305, i32 %339
  store float %338, ptr addrspace(3) %340, align 4
  %341 = add i32 %320, %332
  %342 = extractelement <4 x float> %294, i64 2
  %343 = mul i32 %341, 4
  %344 = getelementptr i8, ptr addrspace(3) %305, i32 %343
  store float %342, ptr addrspace(3) %344, align 4
  %345 = add i32 %326, %332
  %346 = extractelement <4 x float> %294, i64 3
  %347 = mul i32 %345, 4
  %348 = getelementptr i8, ptr addrspace(3) %305, i32 %347
  store float %346, ptr addrspace(3) %348, align 4
  %349 = add i32 %96, 32
  %350 = add i32 %349, %92
  %351 = add i32 %308, %350
  %352 = extractelement <4 x float> %298, i64 0
  %353 = mul i32 %351, 4
  %354 = getelementptr i8, ptr addrspace(3) %305, i32 %353
  store float %352, ptr addrspace(3) %354, align 4
  %355 = add i32 %314, %350
  %356 = extractelement <4 x float> %298, i64 1
  %357 = mul i32 %355, 4
  %358 = getelementptr i8, ptr addrspace(3) %305, i32 %357
  store float %356, ptr addrspace(3) %358, align 4
  %359 = add i32 %320, %350
  %360 = extractelement <4 x float> %298, i64 2
  %361 = mul i32 %359, 4
  %362 = getelementptr i8, ptr addrspace(3) %305, i32 %361
  store float %360, ptr addrspace(3) %362, align 4
  %363 = add i32 %326, %350
  %364 = extractelement <4 x float> %298, i64 3
  %365 = mul i32 %363, 4
  %366 = getelementptr i8, ptr addrspace(3) %305, i32 %365
  store float %364, ptr addrspace(3) %366, align 4
  %367 = add i32 %96, 48
  %368 = add i32 %367, %92
  %369 = add i32 %308, %368
  %370 = extractelement <4 x float> %302, i64 0
  %371 = mul i32 %369, 4
  %372 = getelementptr i8, ptr addrspace(3) %305, i32 %371
  store float %370, ptr addrspace(3) %372, align 4
  %373 = add i32 %314, %368
  %374 = extractelement <4 x float> %302, i64 1
  %375 = mul i32 %373, 4
  %376 = getelementptr i8, ptr addrspace(3) %305, i32 %375
  store float %374, ptr addrspace(3) %376, align 4
  %377 = add i32 %320, %368
  %378 = extractelement <4 x float> %302, i64 2
  %379 = mul i32 %377, 4
  %380 = getelementptr i8, ptr addrspace(3) %305, i32 %379
  store float %378, ptr addrspace(3) %380, align 4
  %381 = add i32 %326, %368
  %382 = extractelement <4 x float> %302, i64 3
  %383 = mul i32 %381, 4
  %384 = getelementptr i8, ptr addrspace(3) %305, i32 %383
  store float %382, ptr addrspace(3) %384, align 4
  %385 = add i32 %306, 16
  %386 = mul i32 %385, 256
  %387 = add i32 %386, %307
  %388 = extractelement <4 x float> %291, i64 0
  %389 = mul i32 %387, 4
  %390 = getelementptr i8, ptr addrspace(3) %305, i32 %389
  store float %388, ptr addrspace(3) %390, align 4
  %391 = add i32 %306, 17
  %392 = mul i32 %391, 256
  %393 = add i32 %392, %307
  %394 = extractelement <4 x float> %291, i64 1
  %395 = mul i32 %393, 4
  %396 = getelementptr i8, ptr addrspace(3) %305, i32 %395
  store float %394, ptr addrspace(3) %396, align 4
  %397 = add i32 %306, 18
  %398 = mul i32 %397, 256
  %399 = add i32 %398, %307
  %400 = extractelement <4 x float> %291, i64 2
  %401 = mul i32 %399, 4
  %402 = getelementptr i8, ptr addrspace(3) %305, i32 %401
  store float %400, ptr addrspace(3) %402, align 4
  %403 = add i32 %306, 19
  %404 = mul i32 %403, 256
  %405 = add i32 %404, %307
  %406 = extractelement <4 x float> %291, i64 3
  %407 = mul i32 %405, 4
  %408 = getelementptr i8, ptr addrspace(3) %305, i32 %407
  store float %406, ptr addrspace(3) %408, align 4
  %409 = add i32 %386, %332
  %410 = extractelement <4 x float> %295, i64 0
  %411 = mul i32 %409, 4
  %412 = getelementptr i8, ptr addrspace(3) %305, i32 %411
  store float %410, ptr addrspace(3) %412, align 4
  %413 = add i32 %392, %332
  %414 = extractelement <4 x float> %295, i64 1
  %415 = mul i32 %413, 4
  %416 = getelementptr i8, ptr addrspace(3) %305, i32 %415
  store float %414, ptr addrspace(3) %416, align 4
  %417 = add i32 %398, %332
  %418 = extractelement <4 x float> %295, i64 2
  %419 = mul i32 %417, 4
  %420 = getelementptr i8, ptr addrspace(3) %305, i32 %419
  store float %418, ptr addrspace(3) %420, align 4
  %421 = add i32 %404, %332
  %422 = extractelement <4 x float> %295, i64 3
  %423 = mul i32 %421, 4
  %424 = getelementptr i8, ptr addrspace(3) %305, i32 %423
  store float %422, ptr addrspace(3) %424, align 4
  %425 = add i32 %386, %350
  %426 = extractelement <4 x float> %299, i64 0
  %427 = mul i32 %425, 4
  %428 = getelementptr i8, ptr addrspace(3) %305, i32 %427
  store float %426, ptr addrspace(3) %428, align 4
  %429 = add i32 %392, %350
  %430 = extractelement <4 x float> %299, i64 1
  %431 = mul i32 %429, 4
  %432 = getelementptr i8, ptr addrspace(3) %305, i32 %431
  store float %430, ptr addrspace(3) %432, align 4
  %433 = add i32 %398, %350
  %434 = extractelement <4 x float> %299, i64 2
  %435 = mul i32 %433, 4
  %436 = getelementptr i8, ptr addrspace(3) %305, i32 %435
  store float %434, ptr addrspace(3) %436, align 4
  %437 = add i32 %404, %350
  %438 = extractelement <4 x float> %299, i64 3
  %439 = mul i32 %437, 4
  %440 = getelementptr i8, ptr addrspace(3) %305, i32 %439
  store float %438, ptr addrspace(3) %440, align 4
  %441 = add i32 %386, %368
  %442 = extractelement <4 x float> %303, i64 0
  %443 = mul i32 %441, 4
  %444 = getelementptr i8, ptr addrspace(3) %305, i32 %443
  store float %442, ptr addrspace(3) %444, align 4
  %445 = add i32 %392, %368
  %446 = extractelement <4 x float> %303, i64 1
  %447 = mul i32 %445, 4
  %448 = getelementptr i8, ptr addrspace(3) %305, i32 %447
  store float %446, ptr addrspace(3) %448, align 4
  %449 = add i32 %398, %368
  %450 = extractelement <4 x float> %303, i64 2
  %451 = mul i32 %449, 4
  %452 = getelementptr i8, ptr addrspace(3) %305, i32 %451
  store float %450, ptr addrspace(3) %452, align 4
  %453 = add i32 %404, %368
  %454 = extractelement <4 x float> %303, i64 3
  %455 = mul i32 %453, 4
  %456 = getelementptr i8, ptr addrspace(3) %305, i32 %455
  store float %454, ptr addrspace(3) %456, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %457 = sdiv i32 %24, 32
  %458 = mul i32 %457, 32
  %459 = icmp ne i32 %24, %458
  %460 = icmp slt i32 %24, 0
  %461 = icmp ne i1 %460, false
  %462 = and i1 %459, %461
  %463 = add i32 %457, -1
  %464 = select i1 %462, i32 %463, i32 %457
  %465 = srem i32 %24, 32
  %466 = mul i32 %465, 2
  %467 = add i32 %66, %464
  %468 = mul i32 %467, 4
  %469 = ptrtoint ptr addrspace(1) %12 to i64
  %470 = sext i32 %468 to i64
  %471 = add i64 %469, %470
  %472 = inttoptr i64 %471 to ptr addrspace(1)
  %473 = load i32, ptr addrspace(1) %472, align 4
  %474 = and i32 %473, 16777215
  %475 = icmp slt i32 %474, %16
  %476 = ptrtoint ptr addrspace(1) %14 to i64
  %477 = add i64 %476, %470
  %478 = inttoptr i64 %477 to ptr addrspace(1)
  %479 = load float, ptr addrspace(1) %478, align 4
  br i1 %475, label %480, label %561

480:                                              ; preds = %49
  %481 = mul i32 %474, 7168
  %482 = add i32 %481, %94
  %483 = add i32 %482, %466
  %484 = mul i32 %464, 256
  %485 = add i32 %484, %466
  %486 = mul i32 %485, 4
  %487 = getelementptr i8, ptr addrspace(3) %305, i32 %486
  %488 = load float, ptr addrspace(3) %487, align 4
  %489 = fmul float %488, %479
  %490 = add i32 %485, 1
  %491 = mul i32 %490, 4
  %492 = getelementptr i8, ptr addrspace(3) %305, i32 %491
  %493 = load float, ptr addrspace(3) %492, align 4
  %494 = fmul float %493, %479
  %495 = insertelement <2 x float> poison, float %489, i64 0
  %496 = insertelement <2 x float> %495, float %494, i64 1
  %497 = fptrunc <2 x float> %496 to <2 x bfloat>
  %498 = mul i32 %483, 2
  %499 = ptrtoint ptr addrspace(1) %17 to i64
  %500 = sext i32 %498 to i64
  %501 = add i64 %499, %500
  %502 = inttoptr i64 %501 to ptr addrspace(1)
  %503 = atomicrmw fadd ptr addrspace(1) %502, <2 x bfloat> %497 syncscope("agent") monotonic, align 4
  %504 = add i32 %485, 64
  %505 = mul i32 %504, 4
  %506 = getelementptr i8, ptr addrspace(3) %305, i32 %505
  %507 = load float, ptr addrspace(3) %506, align 4
  %508 = fmul float %507, %479
  %509 = add i32 %485, 65
  %510 = mul i32 %509, 4
  %511 = getelementptr i8, ptr addrspace(3) %305, i32 %510
  %512 = load float, ptr addrspace(3) %511, align 4
  %513 = fmul float %512, %479
  %514 = insertelement <2 x float> poison, float %508, i64 0
  %515 = insertelement <2 x float> %514, float %513, i64 1
  %516 = fptrunc <2 x float> %515 to <2 x bfloat>
  %517 = add i32 %483, 64
  %518 = mul i32 %517, 2
  %519 = sext i32 %518 to i64
  %520 = add i64 %499, %519
  %521 = inttoptr i64 %520 to ptr addrspace(1)
  %522 = atomicrmw fadd ptr addrspace(1) %521, <2 x bfloat> %516 syncscope("agent") monotonic, align 4
  %523 = add i32 %485, 128
  %524 = mul i32 %523, 4
  %525 = getelementptr i8, ptr addrspace(3) %305, i32 %524
  %526 = load float, ptr addrspace(3) %525, align 4
  %527 = fmul float %526, %479
  %528 = add i32 %485, 129
  %529 = mul i32 %528, 4
  %530 = getelementptr i8, ptr addrspace(3) %305, i32 %529
  %531 = load float, ptr addrspace(3) %530, align 4
  %532 = fmul float %531, %479
  %533 = insertelement <2 x float> poison, float %527, i64 0
  %534 = insertelement <2 x float> %533, float %532, i64 1
  %535 = fptrunc <2 x float> %534 to <2 x bfloat>
  %536 = add i32 %483, 128
  %537 = mul i32 %536, 2
  %538 = sext i32 %537 to i64
  %539 = add i64 %499, %538
  %540 = inttoptr i64 %539 to ptr addrspace(1)
  %541 = atomicrmw fadd ptr addrspace(1) %540, <2 x bfloat> %535 syncscope("agent") monotonic, align 4
  %542 = add i32 %485, 192
  %543 = mul i32 %542, 4
  %544 = getelementptr i8, ptr addrspace(3) %305, i32 %543
  %545 = load float, ptr addrspace(3) %544, align 4
  %546 = fmul float %545, %479
  %547 = add i32 %485, 193
  %548 = mul i32 %547, 4
  %549 = getelementptr i8, ptr addrspace(3) %305, i32 %548
  %550 = load float, ptr addrspace(3) %549, align 4
  %551 = fmul float %550, %479
  %552 = insertelement <2 x float> poison, float %546, i64 0
  %553 = insertelement <2 x float> %552, float %551, i64 1
  %554 = fptrunc <2 x float> %553 to <2 x bfloat>
  %555 = add i32 %483, 192
  %556 = mul i32 %555, 2
  %557 = sext i32 %556 to i64
  %558 = add i64 %499, %557
  %559 = inttoptr i64 %558 to ptr addrspace(1)
  %560 = atomicrmw fadd ptr addrspace(1) %559, <2 x bfloat> %554 syncscope("agent") monotonic, align 4
  br label %561

561:                                              ; preds = %480, %49
  %562 = add i32 %464, 8
  %563 = add i32 %66, %562
  %564 = mul i32 %563, 4
  %565 = sext i32 %564 to i64
  %566 = add i64 %469, %565
  %567 = inttoptr i64 %566 to ptr addrspace(1)
  %568 = load i32, ptr addrspace(1) %567, align 4
  %569 = and i32 %568, 16777215
  %570 = icmp slt i32 %569, %16
  %571 = add i64 %476, %565
  %572 = inttoptr i64 %571 to ptr addrspace(1)
  %573 = load float, ptr addrspace(1) %572, align 4
  br i1 %570, label %574, label %655

574:                                              ; preds = %561
  %575 = mul i32 %569, 7168
  %576 = add i32 %575, %94
  %577 = add i32 %576, %466
  %578 = mul i32 %562, 256
  %579 = add i32 %578, %466
  %580 = mul i32 %579, 4
  %581 = getelementptr i8, ptr addrspace(3) %305, i32 %580
  %582 = load float, ptr addrspace(3) %581, align 4
  %583 = fmul float %582, %573
  %584 = add i32 %579, 1
  %585 = mul i32 %584, 4
  %586 = getelementptr i8, ptr addrspace(3) %305, i32 %585
  %587 = load float, ptr addrspace(3) %586, align 4
  %588 = fmul float %587, %573
  %589 = insertelement <2 x float> poison, float %583, i64 0
  %590 = insertelement <2 x float> %589, float %588, i64 1
  %591 = fptrunc <2 x float> %590 to <2 x bfloat>
  %592 = mul i32 %577, 2
  %593 = ptrtoint ptr addrspace(1) %17 to i64
  %594 = sext i32 %592 to i64
  %595 = add i64 %593, %594
  %596 = inttoptr i64 %595 to ptr addrspace(1)
  %597 = atomicrmw fadd ptr addrspace(1) %596, <2 x bfloat> %591 syncscope("agent") monotonic, align 4
  %598 = add i32 %579, 64
  %599 = mul i32 %598, 4
  %600 = getelementptr i8, ptr addrspace(3) %305, i32 %599
  %601 = load float, ptr addrspace(3) %600, align 4
  %602 = fmul float %601, %573
  %603 = add i32 %579, 65
  %604 = mul i32 %603, 4
  %605 = getelementptr i8, ptr addrspace(3) %305, i32 %604
  %606 = load float, ptr addrspace(3) %605, align 4
  %607 = fmul float %606, %573
  %608 = insertelement <2 x float> poison, float %602, i64 0
  %609 = insertelement <2 x float> %608, float %607, i64 1
  %610 = fptrunc <2 x float> %609 to <2 x bfloat>
  %611 = add i32 %577, 64
  %612 = mul i32 %611, 2
  %613 = sext i32 %612 to i64
  %614 = add i64 %593, %613
  %615 = inttoptr i64 %614 to ptr addrspace(1)
  %616 = atomicrmw fadd ptr addrspace(1) %615, <2 x bfloat> %610 syncscope("agent") monotonic, align 4
  %617 = add i32 %579, 128
  %618 = mul i32 %617, 4
  %619 = getelementptr i8, ptr addrspace(3) %305, i32 %618
  %620 = load float, ptr addrspace(3) %619, align 4
  %621 = fmul float %620, %573
  %622 = add i32 %579, 129
  %623 = mul i32 %622, 4
  %624 = getelementptr i8, ptr addrspace(3) %305, i32 %623
  %625 = load float, ptr addrspace(3) %624, align 4
  %626 = fmul float %625, %573
  %627 = insertelement <2 x float> poison, float %621, i64 0
  %628 = insertelement <2 x float> %627, float %626, i64 1
  %629 = fptrunc <2 x float> %628 to <2 x bfloat>
  %630 = add i32 %577, 128
  %631 = mul i32 %630, 2
  %632 = sext i32 %631 to i64
  %633 = add i64 %593, %632
  %634 = inttoptr i64 %633 to ptr addrspace(1)
  %635 = atomicrmw fadd ptr addrspace(1) %634, <2 x bfloat> %629 syncscope("agent") monotonic, align 4
  %636 = add i32 %579, 192
  %637 = mul i32 %636, 4
  %638 = getelementptr i8, ptr addrspace(3) %305, i32 %637
  %639 = load float, ptr addrspace(3) %638, align 4
  %640 = fmul float %639, %573
  %641 = add i32 %579, 193
  %642 = mul i32 %641, 4
  %643 = getelementptr i8, ptr addrspace(3) %305, i32 %642
  %644 = load float, ptr addrspace(3) %643, align 4
  %645 = fmul float %644, %573
  %646 = insertelement <2 x float> poison, float %640, i64 0
  %647 = insertelement <2 x float> %646, float %645, i64 1
  %648 = fptrunc <2 x float> %647 to <2 x bfloat>
  %649 = add i32 %577, 192
  %650 = mul i32 %649, 2
  %651 = sext i32 %650 to i64
  %652 = add i64 %593, %651
  %653 = inttoptr i64 %652 to ptr addrspace(1)
  %654 = atomicrmw fadd ptr addrspace(1) %653, <2 x bfloat> %648 syncscope("agent") monotonic, align 4
  br label %655

655:                                              ; preds = %574, %561
  %656 = add i32 %464, 16
  %657 = add i32 %66, %656
  %658 = mul i32 %657, 4
  %659 = sext i32 %658 to i64
  %660 = add i64 %469, %659
  %661 = inttoptr i64 %660 to ptr addrspace(1)
  %662 = load i32, ptr addrspace(1) %661, align 4
  %663 = and i32 %662, 16777215
  %664 = icmp slt i32 %663, %16
  %665 = add i64 %476, %659
  %666 = inttoptr i64 %665 to ptr addrspace(1)
  %667 = load float, ptr addrspace(1) %666, align 4
  br i1 %664, label %668, label %749

668:                                              ; preds = %655
  %669 = mul i32 %663, 7168
  %670 = add i32 %669, %94
  %671 = add i32 %670, %466
  %672 = mul i32 %656, 256
  %673 = add i32 %672, %466
  %674 = mul i32 %673, 4
  %675 = getelementptr i8, ptr addrspace(3) %305, i32 %674
  %676 = load float, ptr addrspace(3) %675, align 4
  %677 = fmul float %676, %667
  %678 = add i32 %673, 1
  %679 = mul i32 %678, 4
  %680 = getelementptr i8, ptr addrspace(3) %305, i32 %679
  %681 = load float, ptr addrspace(3) %680, align 4
  %682 = fmul float %681, %667
  %683 = insertelement <2 x float> poison, float %677, i64 0
  %684 = insertelement <2 x float> %683, float %682, i64 1
  %685 = fptrunc <2 x float> %684 to <2 x bfloat>
  %686 = mul i32 %671, 2
  %687 = ptrtoint ptr addrspace(1) %17 to i64
  %688 = sext i32 %686 to i64
  %689 = add i64 %687, %688
  %690 = inttoptr i64 %689 to ptr addrspace(1)
  %691 = atomicrmw fadd ptr addrspace(1) %690, <2 x bfloat> %685 syncscope("agent") monotonic, align 4
  %692 = add i32 %673, 64
  %693 = mul i32 %692, 4
  %694 = getelementptr i8, ptr addrspace(3) %305, i32 %693
  %695 = load float, ptr addrspace(3) %694, align 4
  %696 = fmul float %695, %667
  %697 = add i32 %673, 65
  %698 = mul i32 %697, 4
  %699 = getelementptr i8, ptr addrspace(3) %305, i32 %698
  %700 = load float, ptr addrspace(3) %699, align 4
  %701 = fmul float %700, %667
  %702 = insertelement <2 x float> poison, float %696, i64 0
  %703 = insertelement <2 x float> %702, float %701, i64 1
  %704 = fptrunc <2 x float> %703 to <2 x bfloat>
  %705 = add i32 %671, 64
  %706 = mul i32 %705, 2
  %707 = sext i32 %706 to i64
  %708 = add i64 %687, %707
  %709 = inttoptr i64 %708 to ptr addrspace(1)
  %710 = atomicrmw fadd ptr addrspace(1) %709, <2 x bfloat> %704 syncscope("agent") monotonic, align 4
  %711 = add i32 %673, 128
  %712 = mul i32 %711, 4
  %713 = getelementptr i8, ptr addrspace(3) %305, i32 %712
  %714 = load float, ptr addrspace(3) %713, align 4
  %715 = fmul float %714, %667
  %716 = add i32 %673, 129
  %717 = mul i32 %716, 4
  %718 = getelementptr i8, ptr addrspace(3) %305, i32 %717
  %719 = load float, ptr addrspace(3) %718, align 4
  %720 = fmul float %719, %667
  %721 = insertelement <2 x float> poison, float %715, i64 0
  %722 = insertelement <2 x float> %721, float %720, i64 1
  %723 = fptrunc <2 x float> %722 to <2 x bfloat>
  %724 = add i32 %671, 128
  %725 = mul i32 %724, 2
  %726 = sext i32 %725 to i64
  %727 = add i64 %687, %726
  %728 = inttoptr i64 %727 to ptr addrspace(1)
  %729 = atomicrmw fadd ptr addrspace(1) %728, <2 x bfloat> %723 syncscope("agent") monotonic, align 4
  %730 = add i32 %673, 192
  %731 = mul i32 %730, 4
  %732 = getelementptr i8, ptr addrspace(3) %305, i32 %731
  %733 = load float, ptr addrspace(3) %732, align 4
  %734 = fmul float %733, %667
  %735 = add i32 %673, 193
  %736 = mul i32 %735, 4
  %737 = getelementptr i8, ptr addrspace(3) %305, i32 %736
  %738 = load float, ptr addrspace(3) %737, align 4
  %739 = fmul float %738, %667
  %740 = insertelement <2 x float> poison, float %734, i64 0
  %741 = insertelement <2 x float> %740, float %739, i64 1
  %742 = fptrunc <2 x float> %741 to <2 x bfloat>
  %743 = add i32 %671, 192
  %744 = mul i32 %743, 2
  %745 = sext i32 %744 to i64
  %746 = add i64 %687, %745
  %747 = inttoptr i64 %746 to ptr addrspace(1)
  %748 = atomicrmw fadd ptr addrspace(1) %747, <2 x bfloat> %742 syncscope("agent") monotonic, align 4
  br label %749

749:                                              ; preds = %668, %655
  %750 = add i32 %464, 24
  %751 = add i32 %66, %750
  %752 = mul i32 %751, 4
  %753 = sext i32 %752 to i64
  %754 = add i64 %469, %753
  %755 = inttoptr i64 %754 to ptr addrspace(1)
  %756 = load i32, ptr addrspace(1) %755, align 4
  %757 = and i32 %756, 16777215
  %758 = icmp slt i32 %757, %16
  %759 = add i64 %476, %753
  %760 = inttoptr i64 %759 to ptr addrspace(1)
  %761 = load float, ptr addrspace(1) %760, align 4
  br i1 %758, label %762, label %843

762:                                              ; preds = %749
  %763 = mul i32 %757, 7168
  %764 = add i32 %763, %94
  %765 = add i32 %764, %466
  %766 = mul i32 %750, 256
  %767 = add i32 %766, %466
  %768 = mul i32 %767, 4
  %769 = getelementptr i8, ptr addrspace(3) %305, i32 %768
  %770 = load float, ptr addrspace(3) %769, align 4
  %771 = fmul float %770, %761
  %772 = add i32 %767, 1
  %773 = mul i32 %772, 4
  %774 = getelementptr i8, ptr addrspace(3) %305, i32 %773
  %775 = load float, ptr addrspace(3) %774, align 4
  %776 = fmul float %775, %761
  %777 = insertelement <2 x float> poison, float %771, i64 0
  %778 = insertelement <2 x float> %777, float %776, i64 1
  %779 = fptrunc <2 x float> %778 to <2 x bfloat>
  %780 = mul i32 %765, 2
  %781 = ptrtoint ptr addrspace(1) %17 to i64
  %782 = sext i32 %780 to i64
  %783 = add i64 %781, %782
  %784 = inttoptr i64 %783 to ptr addrspace(1)
  %785 = atomicrmw fadd ptr addrspace(1) %784, <2 x bfloat> %779 syncscope("agent") monotonic, align 4
  %786 = add i32 %767, 64
  %787 = mul i32 %786, 4
  %788 = getelementptr i8, ptr addrspace(3) %305, i32 %787
  %789 = load float, ptr addrspace(3) %788, align 4
  %790 = fmul float %789, %761
  %791 = add i32 %767, 65
  %792 = mul i32 %791, 4
  %793 = getelementptr i8, ptr addrspace(3) %305, i32 %792
  %794 = load float, ptr addrspace(3) %793, align 4
  %795 = fmul float %794, %761
  %796 = insertelement <2 x float> poison, float %790, i64 0
  %797 = insertelement <2 x float> %796, float %795, i64 1
  %798 = fptrunc <2 x float> %797 to <2 x bfloat>
  %799 = add i32 %765, 64
  %800 = mul i32 %799, 2
  %801 = sext i32 %800 to i64
  %802 = add i64 %781, %801
  %803 = inttoptr i64 %802 to ptr addrspace(1)
  %804 = atomicrmw fadd ptr addrspace(1) %803, <2 x bfloat> %798 syncscope("agent") monotonic, align 4
  %805 = add i32 %767, 128
  %806 = mul i32 %805, 4
  %807 = getelementptr i8, ptr addrspace(3) %305, i32 %806
  %808 = load float, ptr addrspace(3) %807, align 4
  %809 = fmul float %808, %761
  %810 = add i32 %767, 129
  %811 = mul i32 %810, 4
  %812 = getelementptr i8, ptr addrspace(3) %305, i32 %811
  %813 = load float, ptr addrspace(3) %812, align 4
  %814 = fmul float %813, %761
  %815 = insertelement <2 x float> poison, float %809, i64 0
  %816 = insertelement <2 x float> %815, float %814, i64 1
  %817 = fptrunc <2 x float> %816 to <2 x bfloat>
  %818 = add i32 %765, 128
  %819 = mul i32 %818, 2
  %820 = sext i32 %819 to i64
  %821 = add i64 %781, %820
  %822 = inttoptr i64 %821 to ptr addrspace(1)
  %823 = atomicrmw fadd ptr addrspace(1) %822, <2 x bfloat> %817 syncscope("agent") monotonic, align 4
  %824 = add i32 %767, 192
  %825 = mul i32 %824, 4
  %826 = getelementptr i8, ptr addrspace(3) %305, i32 %825
  %827 = load float, ptr addrspace(3) %826, align 4
  %828 = fmul float %827, %761
  %829 = add i32 %767, 193
  %830 = mul i32 %829, 4
  %831 = getelementptr i8, ptr addrspace(3) %305, i32 %830
  %832 = load float, ptr addrspace(3) %831, align 4
  %833 = fmul float %832, %761
  %834 = insertelement <2 x float> poison, float %828, i64 0
  %835 = insertelement <2 x float> %834, float %833, i64 1
  %836 = fptrunc <2 x float> %835 to <2 x bfloat>
  %837 = add i32 %765, 192
  %838 = mul i32 %837, 2
  %839 = sext i32 %838 to i64
  %840 = add i64 %781, %839
  %841 = inttoptr i64 %840 to ptr addrspace(1)
  %842 = atomicrmw fadd ptr addrspace(1) %841, <2 x bfloat> %836 syncscope("agent") monotonic, align 4
  br label %843

843:                                              ; preds = %762, %749
  br label %844

844:                                              ; preds = %843, %19
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #7 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
