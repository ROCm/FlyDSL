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
  br i1 %48, label %49, label %799

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
  %61 = inttoptr i64 %60 to ptr addrspace(1)
  %62 = getelementptr i8, ptr addrspace(1) %61, i32 %59
  %63 = load i32, ptr addrspace(1) %62, align 4
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %63)
  %65 = mul i32 %58, 32
  %66 = addrspacecast ptr addrspace(1) %0 to ptr
  %67 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %66, i16 0, i64 167772160, i32 159744)
  %68 = addrspacecast ptr addrspace(1) %2 to ptr
  %69 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %68, i16 0, i64 10485760, i32 159744)
  %70 = addrspacecast ptr addrspace(1) %4 to ptr
  %71 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %70, i16 0, i64 706478080, i32 159744)
  %72 = addrspacecast ptr addrspace(1) %6 to ptr
  %73 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %72, i16 0, i64 44154880, i32 159744)
  %74 = sdiv i32 %26, 8
  %75 = mul i32 %74, 8
  %76 = icmp ne i32 %26, %75
  %77 = icmp slt i32 %26, 0
  %78 = icmp ne i1 %77, false
  %79 = and i1 %76, %78
  %80 = add i32 %74, -1
  %81 = select i1 %79, i32 %80, i32 %74
  %82 = srem i32 %26, 8
  %83 = sdiv i32 %26, 16
  %84 = mul i32 %83, 16
  %85 = icmp ne i32 %26, %84
  %86 = icmp slt i32 %26, 0
  %87 = icmp ne i1 %86, false
  %88 = and i1 %85, %87
  %89 = add i32 %83, -1
  %90 = select i1 %88, i32 %89, i32 %83
  %91 = srem i32 %26, 16
  %92 = mul i32 %64, 7168
  %93 = mul i32 %50, 256
  %94 = add i32 %92, %93
  %95 = mul i32 %35, 64
  %96 = add i32 %94, %95
  %97 = mul i32 %96, 256
  %98 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97)
  %99 = add i32 %96, 16
  %100 = mul i32 %99, 256
  %101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %100)
  %102 = add i32 %96, 32
  %103 = mul i32 %102, 256
  %104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103)
  %105 = add i32 %96, 48
  %106 = mul i32 %105, 256
  %107 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106)
  %108 = mul i32 %50, 8
  %109 = mul i32 %35, 2
  %110 = add i32 %108, %109
  %111 = mul i32 %64, 28672
  %112 = mul i32 %110, 128
  %113 = add i32 %111, %112
  %114 = mul i32 %113, 4
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %114)
  %116 = add i32 %110, 1
  %117 = mul i32 %116, 128
  %118 = add i32 %111, %117
  %119 = mul i32 %118, 4
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %119)
  %121 = sdiv i32 %65, 32
  %122 = mul i32 %121, 32
  %123 = icmp ne i32 %65, %122
  %124 = icmp slt i32 %65, 0
  %125 = icmp ne i1 %124, false
  %126 = and i1 %123, %125
  %127 = add i32 %121, -1
  %128 = select i1 %126, i32 %127, i32 %121
  %129 = mul i32 %128, 512
  %130 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129)
  %131 = mul i32 %35, 8
  %132 = add i32 %65, %131
  %133 = add i32 %132, %81
  %134 = add i32 %131, %81
  %135 = and i32 %134, 14
  %136 = shl i32 %135, 3
  %137 = mul i32 %82, 16
  %138 = xor i32 %137, %136
  %139 = mul i32 %133, 256
  %140 = add i32 %138, %139
  %141 = mul i32 %35, 1024
  %142 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %141
  %143 = sext i32 %142 to i64
  %144 = inttoptr i64 %143 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %67, ptr addrspace(3) %144, i32 16, i32 %140, i32 0, i32 0, i32 0)
  %145 = add i32 %141, 4096
  %146 = add i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32), %145
  %147 = sext i32 %146 to i64
  %148 = inttoptr i64 %147 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %67, ptr addrspace(3) %148, i32 16, i32 %140, i32 128, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %149 = mul i32 %90, 16
  %150 = add i32 %149, %91
  %151 = mul i32 %150, 4
  %152 = sdiv i32 %151, 4
  %153 = mul i32 %152, 4
  %154 = icmp ne i32 %151, %153
  %155 = icmp slt i32 %151, 0
  %156 = icmp ne i1 %155, false
  %157 = and i1 %154, %156
  %158 = add i32 %152, -1
  %159 = select i1 %157, i32 %158, i32 %152
  %160 = mul i32 %159, 4
  %161 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %69, i32 %160, i32 %130, i32 0)
  %162 = add i32 %151, 256
  %163 = sdiv i32 %162, 4
  %164 = mul i32 %163, 4
  %165 = icmp ne i32 %162, %164
  %166 = icmp slt i32 %162, 0
  %167 = icmp ne i1 %166, false
  %168 = and i1 %165, %167
  %169 = add i32 %163, -1
  %170 = select i1 %168, i32 %169, i32 %163
  %171 = mul i32 %170, 4
  %172 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %69, i32 %171, i32 %130, i32 0)
  %173 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %160, i32 %115, i32 0)
  %174 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %160, i32 %120, i32 0)
  %175 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %171, i32 %115, i32 0)
  %176 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %73, i32 %171, i32 %120, i32 0)
  %177 = mul i32 %90, 256
  %178 = mul i32 %91, 16
  %179 = add i32 %177, %178
  %180 = sdiv i32 %179, 4
  %181 = mul i32 %180, 4
  %182 = icmp ne i32 %179, %181
  %183 = icmp slt i32 %179, 0
  %184 = icmp ne i1 %183, false
  %185 = and i1 %182, %184
  %186 = add i32 %180, -1
  %187 = select i1 %185, i32 %186, i32 %180
  %188 = mul i32 %187, 4
  %189 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %98, i32 0)
  %190 = add i32 %179, 1024
  %191 = sdiv i32 %190, 4
  %192 = mul i32 %191, 4
  %193 = icmp ne i32 %190, %192
  %194 = icmp slt i32 %190, 0
  %195 = icmp ne i1 %194, false
  %196 = and i1 %193, %195
  %197 = add i32 %191, -1
  %198 = select i1 %196, i32 %197, i32 %191
  %199 = mul i32 %198, 4
  %200 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %98, i32 0)
  %201 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %101, i32 0)
  %202 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %101, i32 0)
  %203 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %104, i32 0)
  %204 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %104, i32 0)
  %205 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %188, i32 %107, i32 0)
  %206 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %199, i32 %107, i32 0)
  %207 = add i32 %179, 2048
  %208 = sdiv i32 %207, 4
  %209 = mul i32 %208, 4
  %210 = icmp ne i32 %207, %209
  %211 = icmp slt i32 %207, 0
  %212 = icmp ne i1 %211, false
  %213 = and i1 %210, %212
  %214 = add i32 %208, -1
  %215 = select i1 %213, i32 %214, i32 %208
  %216 = mul i32 %215, 4
  %217 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %98, i32 0)
  %218 = add i32 %179, 3072
  %219 = sdiv i32 %218, 4
  %220 = mul i32 %219, 4
  %221 = icmp ne i32 %218, %220
  %222 = icmp slt i32 %218, 0
  %223 = icmp ne i1 %222, false
  %224 = and i1 %221, %223
  %225 = add i32 %219, -1
  %226 = select i1 %224, i32 %225, i32 %219
  %227 = mul i32 %226, 4
  %228 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %98, i32 0)
  %229 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %101, i32 0)
  %230 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %101, i32 0)
  %231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %104, i32 0)
  %232 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %104, i32 0)
  %233 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %216, i32 %107, i32 0)
  %234 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %71, i32 %227, i32 %107, i32 0)
  call void asm sideeffect "s_waitcnt vmcnt(23)", ""()
  call void asm sideeffect "s_barrier", ""()
  %235 = and i32 %91, 14
  %236 = shl i32 %235, 3
  %237 = sext i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32) to i64
  %238 = inttoptr i64 %237 to ptr addrspace(3)
  %239 = xor i32 %149, %236
  %240 = mul i32 %91, 128
  %241 = add i32 %240, %239
  %242 = getelementptr i8, ptr addrspace(3) %238, i32 %241
  %243 = load <4 x i32>, ptr addrspace(3) %242, align 16
  %244 = add i32 %91, 16
  %245 = mul i32 %244, 128
  %246 = add i32 %245, %239
  %247 = getelementptr i8, ptr addrspace(3) %238, i32 %246
  %248 = load <4 x i32>, ptr addrspace(3) %247, align 16
  %249 = add i32 %149, 64
  %250 = xor i32 %249, %236
  %251 = add i32 %240, %250
  %252 = getelementptr i8, ptr addrspace(3) %238, i32 %251
  %253 = load <4 x i32>, ptr addrspace(3) %252, align 16
  %254 = add i32 %245, %250
  %255 = getelementptr i8, ptr addrspace(3) %238, i32 %254
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16
  %257 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %189, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 0, i32 %173)
  %258 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %189, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 0, i32 %173)
  %259 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %200, <4 x float> %257, i32 4, i32 4, i32 2, i32 %161, i32 2, i32 %173)
  %260 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %200, <4 x float> %258, i32 4, i32 4, i32 3, i32 %161, i32 2, i32 %173)
  %261 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %201, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 1, i32 %173)
  %262 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %201, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 1, i32 %173)
  %263 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %202, <4 x float> %261, i32 4, i32 4, i32 2, i32 %161, i32 3, i32 %173)
  %264 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %202, <4 x float> %262, i32 4, i32 4, i32 3, i32 %161, i32 3, i32 %173)
  %265 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %203, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 0, i32 %174)
  %266 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %203, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 0, i32 %174)
  %267 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %204, <4 x float> %265, i32 4, i32 4, i32 2, i32 %161, i32 2, i32 %174)
  %268 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %204, <4 x float> %266, i32 4, i32 4, i32 3, i32 %161, i32 2, i32 %174)
  %269 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %243, <4 x i32> %205, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %161, i32 1, i32 %174)
  %270 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %248, <4 x i32> %205, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %161, i32 1, i32 %174)
  %271 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %253, <4 x i32> %206, <4 x float> %269, i32 4, i32 4, i32 2, i32 %161, i32 3, i32 %174)
  %272 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %256, <4 x i32> %206, <4 x float> %270, i32 4, i32 4, i32 3, i32 %161, i32 3, i32 %174)
  call void asm sideeffect "s_waitcnt vmcnt(22)", ""()
  call void asm sideeffect "s_barrier", ""()
  %273 = add i32 %240, 4096
  %274 = add i32 %273, %239
  %275 = getelementptr i8, ptr addrspace(3) %238, i32 %274
  %276 = load <4 x i32>, ptr addrspace(3) %275, align 16
  %277 = add i32 %245, 4096
  %278 = add i32 %277, %239
  %279 = getelementptr i8, ptr addrspace(3) %238, i32 %278
  %280 = load <4 x i32>, ptr addrspace(3) %279, align 16
  %281 = add i32 %273, %250
  %282 = getelementptr i8, ptr addrspace(3) %238, i32 %281
  %283 = load <4 x i32>, ptr addrspace(3) %282, align 16
  %284 = add i32 %277, %250
  %285 = getelementptr i8, ptr addrspace(3) %238, i32 %284
  %286 = load <4 x i32>, ptr addrspace(3) %285, align 16
  %287 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %217, <4 x float> %259, i32 4, i32 4, i32 0, i32 %172, i32 0, i32 %175)
  %288 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %217, <4 x float> %260, i32 4, i32 4, i32 1, i32 %172, i32 0, i32 %175)
  %289 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %228, <4 x float> %287, i32 4, i32 4, i32 2, i32 %172, i32 2, i32 %175)
  %290 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %228, <4 x float> %288, i32 4, i32 4, i32 3, i32 %172, i32 2, i32 %175)
  %291 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %229, <4 x float> %263, i32 4, i32 4, i32 0, i32 %172, i32 1, i32 %175)
  %292 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %229, <4 x float> %264, i32 4, i32 4, i32 1, i32 %172, i32 1, i32 %175)
  %293 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %230, <4 x float> %291, i32 4, i32 4, i32 2, i32 %172, i32 3, i32 %175)
  %294 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %230, <4 x float> %292, i32 4, i32 4, i32 3, i32 %172, i32 3, i32 %175)
  %295 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %231, <4 x float> %267, i32 4, i32 4, i32 0, i32 %172, i32 0, i32 %176)
  %296 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %231, <4 x float> %268, i32 4, i32 4, i32 1, i32 %172, i32 0, i32 %176)
  %297 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %232, <4 x float> %295, i32 4, i32 4, i32 2, i32 %172, i32 2, i32 %176)
  %298 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %232, <4 x float> %296, i32 4, i32 4, i32 3, i32 %172, i32 2, i32 %176)
  %299 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %276, <4 x i32> %233, <4 x float> %271, i32 4, i32 4, i32 0, i32 %172, i32 1, i32 %176)
  %300 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %280, <4 x i32> %233, <4 x float> %272, i32 4, i32 4, i32 1, i32 %172, i32 1, i32 %176)
  %301 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %283, <4 x i32> %234, <4 x float> %299, i32 4, i32 4, i32 2, i32 %172, i32 3, i32 %176)
  %302 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %286, <4 x i32> %234, <4 x float> %300, i32 4, i32 4, i32 3, i32 %172, i32 3, i32 %176)
  %303 = sext i32 ptrtoint (ptr addrspace(3) @gemm2port_smem to i32) to i64
  %304 = inttoptr i64 %303 to ptr addrspace(3)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %305 = mul i32 %90, 4
  %306 = add i32 %95, %91
  %307 = mul i32 %90, 1024
  %308 = add i32 %307, %306
  %309 = extractelement <4 x float> %289, i64 0
  %310 = mul i32 %308, 4
  %311 = getelementptr i8, ptr addrspace(3) %304, i32 %310
  store float %309, ptr addrspace(3) %311, align 4
  %312 = add i32 %305, 1
  %313 = mul i32 %312, 256
  %314 = add i32 %313, %306
  %315 = extractelement <4 x float> %289, i64 1
  %316 = mul i32 %314, 4
  %317 = getelementptr i8, ptr addrspace(3) %304, i32 %316
  store float %315, ptr addrspace(3) %317, align 4
  %318 = add i32 %305, 2
  %319 = mul i32 %318, 256
  %320 = add i32 %319, %306
  %321 = extractelement <4 x float> %289, i64 2
  %322 = mul i32 %320, 4
  %323 = getelementptr i8, ptr addrspace(3) %304, i32 %322
  store float %321, ptr addrspace(3) %323, align 4
  %324 = add i32 %305, 3
  %325 = mul i32 %324, 256
  %326 = add i32 %325, %306
  %327 = extractelement <4 x float> %289, i64 3
  %328 = mul i32 %326, 4
  %329 = getelementptr i8, ptr addrspace(3) %304, i32 %328
  store float %327, ptr addrspace(3) %329, align 4
  %330 = add i32 %95, 16
  %331 = add i32 %330, %91
  %332 = add i32 %307, %331
  %333 = extractelement <4 x float> %293, i64 0
  %334 = mul i32 %332, 4
  %335 = getelementptr i8, ptr addrspace(3) %304, i32 %334
  store float %333, ptr addrspace(3) %335, align 4
  %336 = add i32 %313, %331
  %337 = extractelement <4 x float> %293, i64 1
  %338 = mul i32 %336, 4
  %339 = getelementptr i8, ptr addrspace(3) %304, i32 %338
  store float %337, ptr addrspace(3) %339, align 4
  %340 = add i32 %319, %331
  %341 = extractelement <4 x float> %293, i64 2
  %342 = mul i32 %340, 4
  %343 = getelementptr i8, ptr addrspace(3) %304, i32 %342
  store float %341, ptr addrspace(3) %343, align 4
  %344 = add i32 %325, %331
  %345 = extractelement <4 x float> %293, i64 3
  %346 = mul i32 %344, 4
  %347 = getelementptr i8, ptr addrspace(3) %304, i32 %346
  store float %345, ptr addrspace(3) %347, align 4
  %348 = add i32 %95, 32
  %349 = add i32 %348, %91
  %350 = add i32 %307, %349
  %351 = extractelement <4 x float> %297, i64 0
  %352 = mul i32 %350, 4
  %353 = getelementptr i8, ptr addrspace(3) %304, i32 %352
  store float %351, ptr addrspace(3) %353, align 4
  %354 = add i32 %313, %349
  %355 = extractelement <4 x float> %297, i64 1
  %356 = mul i32 %354, 4
  %357 = getelementptr i8, ptr addrspace(3) %304, i32 %356
  store float %355, ptr addrspace(3) %357, align 4
  %358 = add i32 %319, %349
  %359 = extractelement <4 x float> %297, i64 2
  %360 = mul i32 %358, 4
  %361 = getelementptr i8, ptr addrspace(3) %304, i32 %360
  store float %359, ptr addrspace(3) %361, align 4
  %362 = add i32 %325, %349
  %363 = extractelement <4 x float> %297, i64 3
  %364 = mul i32 %362, 4
  %365 = getelementptr i8, ptr addrspace(3) %304, i32 %364
  store float %363, ptr addrspace(3) %365, align 4
  %366 = add i32 %95, 48
  %367 = add i32 %366, %91
  %368 = add i32 %307, %367
  %369 = extractelement <4 x float> %301, i64 0
  %370 = mul i32 %368, 4
  %371 = getelementptr i8, ptr addrspace(3) %304, i32 %370
  store float %369, ptr addrspace(3) %371, align 4
  %372 = add i32 %313, %367
  %373 = extractelement <4 x float> %301, i64 1
  %374 = mul i32 %372, 4
  %375 = getelementptr i8, ptr addrspace(3) %304, i32 %374
  store float %373, ptr addrspace(3) %375, align 4
  %376 = add i32 %319, %367
  %377 = extractelement <4 x float> %301, i64 2
  %378 = mul i32 %376, 4
  %379 = getelementptr i8, ptr addrspace(3) %304, i32 %378
  store float %377, ptr addrspace(3) %379, align 4
  %380 = add i32 %325, %367
  %381 = extractelement <4 x float> %301, i64 3
  %382 = mul i32 %380, 4
  %383 = getelementptr i8, ptr addrspace(3) %304, i32 %382
  store float %381, ptr addrspace(3) %383, align 4
  %384 = add i32 %305, 16
  %385 = mul i32 %384, 256
  %386 = add i32 %385, %306
  %387 = extractelement <4 x float> %290, i64 0
  %388 = mul i32 %386, 4
  %389 = getelementptr i8, ptr addrspace(3) %304, i32 %388
  store float %387, ptr addrspace(3) %389, align 4
  %390 = add i32 %305, 17
  %391 = mul i32 %390, 256
  %392 = add i32 %391, %306
  %393 = extractelement <4 x float> %290, i64 1
  %394 = mul i32 %392, 4
  %395 = getelementptr i8, ptr addrspace(3) %304, i32 %394
  store float %393, ptr addrspace(3) %395, align 4
  %396 = add i32 %305, 18
  %397 = mul i32 %396, 256
  %398 = add i32 %397, %306
  %399 = extractelement <4 x float> %290, i64 2
  %400 = mul i32 %398, 4
  %401 = getelementptr i8, ptr addrspace(3) %304, i32 %400
  store float %399, ptr addrspace(3) %401, align 4
  %402 = add i32 %305, 19
  %403 = mul i32 %402, 256
  %404 = add i32 %403, %306
  %405 = extractelement <4 x float> %290, i64 3
  %406 = mul i32 %404, 4
  %407 = getelementptr i8, ptr addrspace(3) %304, i32 %406
  store float %405, ptr addrspace(3) %407, align 4
  %408 = add i32 %385, %331
  %409 = extractelement <4 x float> %294, i64 0
  %410 = mul i32 %408, 4
  %411 = getelementptr i8, ptr addrspace(3) %304, i32 %410
  store float %409, ptr addrspace(3) %411, align 4
  %412 = add i32 %391, %331
  %413 = extractelement <4 x float> %294, i64 1
  %414 = mul i32 %412, 4
  %415 = getelementptr i8, ptr addrspace(3) %304, i32 %414
  store float %413, ptr addrspace(3) %415, align 4
  %416 = add i32 %397, %331
  %417 = extractelement <4 x float> %294, i64 2
  %418 = mul i32 %416, 4
  %419 = getelementptr i8, ptr addrspace(3) %304, i32 %418
  store float %417, ptr addrspace(3) %419, align 4
  %420 = add i32 %403, %331
  %421 = extractelement <4 x float> %294, i64 3
  %422 = mul i32 %420, 4
  %423 = getelementptr i8, ptr addrspace(3) %304, i32 %422
  store float %421, ptr addrspace(3) %423, align 4
  %424 = add i32 %385, %349
  %425 = extractelement <4 x float> %298, i64 0
  %426 = mul i32 %424, 4
  %427 = getelementptr i8, ptr addrspace(3) %304, i32 %426
  store float %425, ptr addrspace(3) %427, align 4
  %428 = add i32 %391, %349
  %429 = extractelement <4 x float> %298, i64 1
  %430 = mul i32 %428, 4
  %431 = getelementptr i8, ptr addrspace(3) %304, i32 %430
  store float %429, ptr addrspace(3) %431, align 4
  %432 = add i32 %397, %349
  %433 = extractelement <4 x float> %298, i64 2
  %434 = mul i32 %432, 4
  %435 = getelementptr i8, ptr addrspace(3) %304, i32 %434
  store float %433, ptr addrspace(3) %435, align 4
  %436 = add i32 %403, %349
  %437 = extractelement <4 x float> %298, i64 3
  %438 = mul i32 %436, 4
  %439 = getelementptr i8, ptr addrspace(3) %304, i32 %438
  store float %437, ptr addrspace(3) %439, align 4
  %440 = add i32 %385, %367
  %441 = extractelement <4 x float> %302, i64 0
  %442 = mul i32 %440, 4
  %443 = getelementptr i8, ptr addrspace(3) %304, i32 %442
  store float %441, ptr addrspace(3) %443, align 4
  %444 = add i32 %391, %367
  %445 = extractelement <4 x float> %302, i64 1
  %446 = mul i32 %444, 4
  %447 = getelementptr i8, ptr addrspace(3) %304, i32 %446
  store float %445, ptr addrspace(3) %447, align 4
  %448 = add i32 %397, %367
  %449 = extractelement <4 x float> %302, i64 2
  %450 = mul i32 %448, 4
  %451 = getelementptr i8, ptr addrspace(3) %304, i32 %450
  store float %449, ptr addrspace(3) %451, align 4
  %452 = add i32 %403, %367
  %453 = extractelement <4 x float> %302, i64 3
  %454 = mul i32 %452, 4
  %455 = getelementptr i8, ptr addrspace(3) %304, i32 %454
  store float %453, ptr addrspace(3) %455, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %456 = sdiv i32 %24, 32
  %457 = mul i32 %456, 32
  %458 = icmp ne i32 %24, %457
  %459 = icmp slt i32 %24, 0
  %460 = icmp ne i1 %459, false
  %461 = and i1 %458, %460
  %462 = add i32 %456, -1
  %463 = select i1 %461, i32 %462, i32 %456
  %464 = srem i32 %24, 32
  %465 = mul i32 %464, 2
  %466 = ptrtoint ptr addrspace(1) %12 to i64
  %467 = inttoptr i64 %466 to ptr addrspace(1)
  %468 = ptrtoint ptr addrspace(1) %14 to i64
  %469 = inttoptr i64 %468 to ptr addrspace(1)
  %470 = ptrtoint ptr addrspace(1) %17 to i64
  %471 = inttoptr i64 %470 to ptr addrspace(1)
  %472 = add i32 %65, %463
  %473 = mul i32 %472, 4
  %474 = getelementptr i8, ptr addrspace(1) %467, i32 %473
  %475 = load i32, ptr addrspace(1) %474, align 4
  %476 = and i32 %475, 16777215
  %477 = icmp slt i32 %476, %16
  %478 = getelementptr i8, ptr addrspace(1) %469, i32 %473
  %479 = load float, ptr addrspace(1) %478, align 4
  br i1 %477, label %480, label %552

480:                                              ; preds = %49
  %481 = mul i32 %476, 7168
  %482 = add i32 %481, %93
  %483 = add i32 %482, %465
  %484 = mul i32 %463, 256
  %485 = add i32 %484, %465
  %486 = mul i32 %485, 4
  %487 = getelementptr i8, ptr addrspace(3) %304, i32 %486
  %488 = load float, ptr addrspace(3) %487, align 4
  %489 = fmul float %488, %479
  %490 = add i32 %485, 1
  %491 = mul i32 %490, 4
  %492 = getelementptr i8, ptr addrspace(3) %304, i32 %491
  %493 = load float, ptr addrspace(3) %492, align 4
  %494 = fmul float %493, %479
  %495 = insertelement <2 x float> poison, float %489, i64 0
  %496 = insertelement <2 x float> %495, float %494, i64 1
  %497 = fptrunc <2 x float> %496 to <2 x bfloat>
  %498 = mul i32 %483, 2
  %499 = getelementptr i8, ptr addrspace(1) %471, i32 %498
  %500 = atomicrmw fadd ptr addrspace(1) %499, <2 x bfloat> %497 syncscope("agent") monotonic, align 4
  %501 = add i32 %485, 64
  %502 = mul i32 %501, 4
  %503 = getelementptr i8, ptr addrspace(3) %304, i32 %502
  %504 = load float, ptr addrspace(3) %503, align 4
  %505 = fmul float %504, %479
  %506 = add i32 %485, 65
  %507 = mul i32 %506, 4
  %508 = getelementptr i8, ptr addrspace(3) %304, i32 %507
  %509 = load float, ptr addrspace(3) %508, align 4
  %510 = fmul float %509, %479
  %511 = insertelement <2 x float> poison, float %505, i64 0
  %512 = insertelement <2 x float> %511, float %510, i64 1
  %513 = fptrunc <2 x float> %512 to <2 x bfloat>
  %514 = add i32 %483, 64
  %515 = mul i32 %514, 2
  %516 = getelementptr i8, ptr addrspace(1) %471, i32 %515
  %517 = atomicrmw fadd ptr addrspace(1) %516, <2 x bfloat> %513 syncscope("agent") monotonic, align 4
  %518 = add i32 %485, 128
  %519 = mul i32 %518, 4
  %520 = getelementptr i8, ptr addrspace(3) %304, i32 %519
  %521 = load float, ptr addrspace(3) %520, align 4
  %522 = fmul float %521, %479
  %523 = add i32 %485, 129
  %524 = mul i32 %523, 4
  %525 = getelementptr i8, ptr addrspace(3) %304, i32 %524
  %526 = load float, ptr addrspace(3) %525, align 4
  %527 = fmul float %526, %479
  %528 = insertelement <2 x float> poison, float %522, i64 0
  %529 = insertelement <2 x float> %528, float %527, i64 1
  %530 = fptrunc <2 x float> %529 to <2 x bfloat>
  %531 = add i32 %483, 128
  %532 = mul i32 %531, 2
  %533 = getelementptr i8, ptr addrspace(1) %471, i32 %532
  %534 = atomicrmw fadd ptr addrspace(1) %533, <2 x bfloat> %530 syncscope("agent") monotonic, align 4
  %535 = add i32 %485, 192
  %536 = mul i32 %535, 4
  %537 = getelementptr i8, ptr addrspace(3) %304, i32 %536
  %538 = load float, ptr addrspace(3) %537, align 4
  %539 = fmul float %538, %479
  %540 = add i32 %485, 193
  %541 = mul i32 %540, 4
  %542 = getelementptr i8, ptr addrspace(3) %304, i32 %541
  %543 = load float, ptr addrspace(3) %542, align 4
  %544 = fmul float %543, %479
  %545 = insertelement <2 x float> poison, float %539, i64 0
  %546 = insertelement <2 x float> %545, float %544, i64 1
  %547 = fptrunc <2 x float> %546 to <2 x bfloat>
  %548 = add i32 %483, 192
  %549 = mul i32 %548, 2
  %550 = getelementptr i8, ptr addrspace(1) %471, i32 %549
  %551 = atomicrmw fadd ptr addrspace(1) %550, <2 x bfloat> %547 syncscope("agent") monotonic, align 4
  br label %552

552:                                              ; preds = %480, %49
  %553 = add i32 %463, 8
  %554 = add i32 %65, %553
  %555 = mul i32 %554, 4
  %556 = getelementptr i8, ptr addrspace(1) %467, i32 %555
  %557 = load i32, ptr addrspace(1) %556, align 4
  %558 = and i32 %557, 16777215
  %559 = icmp slt i32 %558, %16
  %560 = getelementptr i8, ptr addrspace(1) %469, i32 %555
  %561 = load float, ptr addrspace(1) %560, align 4
  br i1 %559, label %562, label %634

562:                                              ; preds = %552
  %563 = mul i32 %558, 7168
  %564 = add i32 %563, %93
  %565 = add i32 %564, %465
  %566 = mul i32 %553, 256
  %567 = add i32 %566, %465
  %568 = mul i32 %567, 4
  %569 = getelementptr i8, ptr addrspace(3) %304, i32 %568
  %570 = load float, ptr addrspace(3) %569, align 4
  %571 = fmul float %570, %561
  %572 = add i32 %567, 1
  %573 = mul i32 %572, 4
  %574 = getelementptr i8, ptr addrspace(3) %304, i32 %573
  %575 = load float, ptr addrspace(3) %574, align 4
  %576 = fmul float %575, %561
  %577 = insertelement <2 x float> poison, float %571, i64 0
  %578 = insertelement <2 x float> %577, float %576, i64 1
  %579 = fptrunc <2 x float> %578 to <2 x bfloat>
  %580 = mul i32 %565, 2
  %581 = getelementptr i8, ptr addrspace(1) %471, i32 %580
  %582 = atomicrmw fadd ptr addrspace(1) %581, <2 x bfloat> %579 syncscope("agent") monotonic, align 4
  %583 = add i32 %567, 64
  %584 = mul i32 %583, 4
  %585 = getelementptr i8, ptr addrspace(3) %304, i32 %584
  %586 = load float, ptr addrspace(3) %585, align 4
  %587 = fmul float %586, %561
  %588 = add i32 %567, 65
  %589 = mul i32 %588, 4
  %590 = getelementptr i8, ptr addrspace(3) %304, i32 %589
  %591 = load float, ptr addrspace(3) %590, align 4
  %592 = fmul float %591, %561
  %593 = insertelement <2 x float> poison, float %587, i64 0
  %594 = insertelement <2 x float> %593, float %592, i64 1
  %595 = fptrunc <2 x float> %594 to <2 x bfloat>
  %596 = add i32 %565, 64
  %597 = mul i32 %596, 2
  %598 = getelementptr i8, ptr addrspace(1) %471, i32 %597
  %599 = atomicrmw fadd ptr addrspace(1) %598, <2 x bfloat> %595 syncscope("agent") monotonic, align 4
  %600 = add i32 %567, 128
  %601 = mul i32 %600, 4
  %602 = getelementptr i8, ptr addrspace(3) %304, i32 %601
  %603 = load float, ptr addrspace(3) %602, align 4
  %604 = fmul float %603, %561
  %605 = add i32 %567, 129
  %606 = mul i32 %605, 4
  %607 = getelementptr i8, ptr addrspace(3) %304, i32 %606
  %608 = load float, ptr addrspace(3) %607, align 4
  %609 = fmul float %608, %561
  %610 = insertelement <2 x float> poison, float %604, i64 0
  %611 = insertelement <2 x float> %610, float %609, i64 1
  %612 = fptrunc <2 x float> %611 to <2 x bfloat>
  %613 = add i32 %565, 128
  %614 = mul i32 %613, 2
  %615 = getelementptr i8, ptr addrspace(1) %471, i32 %614
  %616 = atomicrmw fadd ptr addrspace(1) %615, <2 x bfloat> %612 syncscope("agent") monotonic, align 4
  %617 = add i32 %567, 192
  %618 = mul i32 %617, 4
  %619 = getelementptr i8, ptr addrspace(3) %304, i32 %618
  %620 = load float, ptr addrspace(3) %619, align 4
  %621 = fmul float %620, %561
  %622 = add i32 %567, 193
  %623 = mul i32 %622, 4
  %624 = getelementptr i8, ptr addrspace(3) %304, i32 %623
  %625 = load float, ptr addrspace(3) %624, align 4
  %626 = fmul float %625, %561
  %627 = insertelement <2 x float> poison, float %621, i64 0
  %628 = insertelement <2 x float> %627, float %626, i64 1
  %629 = fptrunc <2 x float> %628 to <2 x bfloat>
  %630 = add i32 %565, 192
  %631 = mul i32 %630, 2
  %632 = getelementptr i8, ptr addrspace(1) %471, i32 %631
  %633 = atomicrmw fadd ptr addrspace(1) %632, <2 x bfloat> %629 syncscope("agent") monotonic, align 4
  br label %634

634:                                              ; preds = %562, %552
  %635 = add i32 %463, 16
  %636 = add i32 %65, %635
  %637 = mul i32 %636, 4
  %638 = getelementptr i8, ptr addrspace(1) %467, i32 %637
  %639 = load i32, ptr addrspace(1) %638, align 4
  %640 = and i32 %639, 16777215
  %641 = icmp slt i32 %640, %16
  %642 = getelementptr i8, ptr addrspace(1) %469, i32 %637
  %643 = load float, ptr addrspace(1) %642, align 4
  br i1 %641, label %644, label %716

644:                                              ; preds = %634
  %645 = mul i32 %640, 7168
  %646 = add i32 %645, %93
  %647 = add i32 %646, %465
  %648 = mul i32 %635, 256
  %649 = add i32 %648, %465
  %650 = mul i32 %649, 4
  %651 = getelementptr i8, ptr addrspace(3) %304, i32 %650
  %652 = load float, ptr addrspace(3) %651, align 4
  %653 = fmul float %652, %643
  %654 = add i32 %649, 1
  %655 = mul i32 %654, 4
  %656 = getelementptr i8, ptr addrspace(3) %304, i32 %655
  %657 = load float, ptr addrspace(3) %656, align 4
  %658 = fmul float %657, %643
  %659 = insertelement <2 x float> poison, float %653, i64 0
  %660 = insertelement <2 x float> %659, float %658, i64 1
  %661 = fptrunc <2 x float> %660 to <2 x bfloat>
  %662 = mul i32 %647, 2
  %663 = getelementptr i8, ptr addrspace(1) %471, i32 %662
  %664 = atomicrmw fadd ptr addrspace(1) %663, <2 x bfloat> %661 syncscope("agent") monotonic, align 4
  %665 = add i32 %649, 64
  %666 = mul i32 %665, 4
  %667 = getelementptr i8, ptr addrspace(3) %304, i32 %666
  %668 = load float, ptr addrspace(3) %667, align 4
  %669 = fmul float %668, %643
  %670 = add i32 %649, 65
  %671 = mul i32 %670, 4
  %672 = getelementptr i8, ptr addrspace(3) %304, i32 %671
  %673 = load float, ptr addrspace(3) %672, align 4
  %674 = fmul float %673, %643
  %675 = insertelement <2 x float> poison, float %669, i64 0
  %676 = insertelement <2 x float> %675, float %674, i64 1
  %677 = fptrunc <2 x float> %676 to <2 x bfloat>
  %678 = add i32 %647, 64
  %679 = mul i32 %678, 2
  %680 = getelementptr i8, ptr addrspace(1) %471, i32 %679
  %681 = atomicrmw fadd ptr addrspace(1) %680, <2 x bfloat> %677 syncscope("agent") monotonic, align 4
  %682 = add i32 %649, 128
  %683 = mul i32 %682, 4
  %684 = getelementptr i8, ptr addrspace(3) %304, i32 %683
  %685 = load float, ptr addrspace(3) %684, align 4
  %686 = fmul float %685, %643
  %687 = add i32 %649, 129
  %688 = mul i32 %687, 4
  %689 = getelementptr i8, ptr addrspace(3) %304, i32 %688
  %690 = load float, ptr addrspace(3) %689, align 4
  %691 = fmul float %690, %643
  %692 = insertelement <2 x float> poison, float %686, i64 0
  %693 = insertelement <2 x float> %692, float %691, i64 1
  %694 = fptrunc <2 x float> %693 to <2 x bfloat>
  %695 = add i32 %647, 128
  %696 = mul i32 %695, 2
  %697 = getelementptr i8, ptr addrspace(1) %471, i32 %696
  %698 = atomicrmw fadd ptr addrspace(1) %697, <2 x bfloat> %694 syncscope("agent") monotonic, align 4
  %699 = add i32 %649, 192
  %700 = mul i32 %699, 4
  %701 = getelementptr i8, ptr addrspace(3) %304, i32 %700
  %702 = load float, ptr addrspace(3) %701, align 4
  %703 = fmul float %702, %643
  %704 = add i32 %649, 193
  %705 = mul i32 %704, 4
  %706 = getelementptr i8, ptr addrspace(3) %304, i32 %705
  %707 = load float, ptr addrspace(3) %706, align 4
  %708 = fmul float %707, %643
  %709 = insertelement <2 x float> poison, float %703, i64 0
  %710 = insertelement <2 x float> %709, float %708, i64 1
  %711 = fptrunc <2 x float> %710 to <2 x bfloat>
  %712 = add i32 %647, 192
  %713 = mul i32 %712, 2
  %714 = getelementptr i8, ptr addrspace(1) %471, i32 %713
  %715 = atomicrmw fadd ptr addrspace(1) %714, <2 x bfloat> %711 syncscope("agent") monotonic, align 4
  br label %716

716:                                              ; preds = %644, %634
  %717 = add i32 %463, 24
  %718 = add i32 %65, %717
  %719 = mul i32 %718, 4
  %720 = getelementptr i8, ptr addrspace(1) %467, i32 %719
  %721 = load i32, ptr addrspace(1) %720, align 4
  %722 = and i32 %721, 16777215
  %723 = icmp slt i32 %722, %16
  %724 = getelementptr i8, ptr addrspace(1) %469, i32 %719
  %725 = load float, ptr addrspace(1) %724, align 4
  br i1 %723, label %726, label %798

726:                                              ; preds = %716
  %727 = mul i32 %722, 7168
  %728 = add i32 %727, %93
  %729 = add i32 %728, %465
  %730 = mul i32 %717, 256
  %731 = add i32 %730, %465
  %732 = mul i32 %731, 4
  %733 = getelementptr i8, ptr addrspace(3) %304, i32 %732
  %734 = load float, ptr addrspace(3) %733, align 4
  %735 = fmul float %734, %725
  %736 = add i32 %731, 1
  %737 = mul i32 %736, 4
  %738 = getelementptr i8, ptr addrspace(3) %304, i32 %737
  %739 = load float, ptr addrspace(3) %738, align 4
  %740 = fmul float %739, %725
  %741 = insertelement <2 x float> poison, float %735, i64 0
  %742 = insertelement <2 x float> %741, float %740, i64 1
  %743 = fptrunc <2 x float> %742 to <2 x bfloat>
  %744 = mul i32 %729, 2
  %745 = getelementptr i8, ptr addrspace(1) %471, i32 %744
  %746 = atomicrmw fadd ptr addrspace(1) %745, <2 x bfloat> %743 syncscope("agent") monotonic, align 4
  %747 = add i32 %731, 64
  %748 = mul i32 %747, 4
  %749 = getelementptr i8, ptr addrspace(3) %304, i32 %748
  %750 = load float, ptr addrspace(3) %749, align 4
  %751 = fmul float %750, %725
  %752 = add i32 %731, 65
  %753 = mul i32 %752, 4
  %754 = getelementptr i8, ptr addrspace(3) %304, i32 %753
  %755 = load float, ptr addrspace(3) %754, align 4
  %756 = fmul float %755, %725
  %757 = insertelement <2 x float> poison, float %751, i64 0
  %758 = insertelement <2 x float> %757, float %756, i64 1
  %759 = fptrunc <2 x float> %758 to <2 x bfloat>
  %760 = add i32 %729, 64
  %761 = mul i32 %760, 2
  %762 = getelementptr i8, ptr addrspace(1) %471, i32 %761
  %763 = atomicrmw fadd ptr addrspace(1) %762, <2 x bfloat> %759 syncscope("agent") monotonic, align 4
  %764 = add i32 %731, 128
  %765 = mul i32 %764, 4
  %766 = getelementptr i8, ptr addrspace(3) %304, i32 %765
  %767 = load float, ptr addrspace(3) %766, align 4
  %768 = fmul float %767, %725
  %769 = add i32 %731, 129
  %770 = mul i32 %769, 4
  %771 = getelementptr i8, ptr addrspace(3) %304, i32 %770
  %772 = load float, ptr addrspace(3) %771, align 4
  %773 = fmul float %772, %725
  %774 = insertelement <2 x float> poison, float %768, i64 0
  %775 = insertelement <2 x float> %774, float %773, i64 1
  %776 = fptrunc <2 x float> %775 to <2 x bfloat>
  %777 = add i32 %729, 128
  %778 = mul i32 %777, 2
  %779 = getelementptr i8, ptr addrspace(1) %471, i32 %778
  %780 = atomicrmw fadd ptr addrspace(1) %779, <2 x bfloat> %776 syncscope("agent") monotonic, align 4
  %781 = add i32 %731, 192
  %782 = mul i32 %781, 4
  %783 = getelementptr i8, ptr addrspace(3) %304, i32 %782
  %784 = load float, ptr addrspace(3) %783, align 4
  %785 = fmul float %784, %725
  %786 = add i32 %731, 193
  %787 = mul i32 %786, 4
  %788 = getelementptr i8, ptr addrspace(3) %304, i32 %787
  %789 = load float, ptr addrspace(3) %788, align 4
  %790 = fmul float %789, %725
  %791 = insertelement <2 x float> poison, float %785, i64 0
  %792 = insertelement <2 x float> %791, float %790, i64 1
  %793 = fptrunc <2 x float> %792 to <2 x bfloat>
  %794 = add i32 %729, 192
  %795 = mul i32 %794, 2
  %796 = getelementptr i8, ptr addrspace(1) %471, i32 %795
  %797 = atomicrmw fadd ptr addrspace(1) %796, <2 x bfloat> %793 syncscope("agent") monotonic, align 4
  br label %798

798:                                              ; preds = %726, %716
  br label %799

799:                                              ; preds = %798, %19
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
