; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@smem0 = external addrspace(3) global [32896 x i8], align 1024
@smem1 = external addrspace(3) global [4096 x i8], align 1024

define amdgpu_kernel void @mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_persist_cu256(ptr addrspace(1) noalias %0, ptr addrspace(1) noalias %1, ptr addrspace(1) noalias %2, ptr addrspace(1) noalias %3, ptr addrspace(1) noalias %4, ptr addrspace(1) noalias %5, ptr addrspace(1) noalias %6, ptr addrspace(1) noalias %7, ptr addrspace(1) noalias %8, ptr addrspace(1) noalias %9, i32 %10, i32 %11, i32 %12, i32 %13) #0 {
  %15 = sext i32 %10 to i64
  %16 = sext i32 %12 to i64
  %17 = sext i32 %13 to i64
  %18 = call i32 @llvm.amdgcn.workitem.id.x()
  %19 = sext i32 %18 to i64
  %20 = call i32 @llvm.amdgcn.workgroup.id.x()
  %21 = sext i32 %20 to i64
  %22 = call i32 @llvm.amdgcn.workgroup.id.y()
  %23 = sext i32 %22 to i64
  %24 = mul i64 %15, 9
  %25 = mul i64 %24, %16
  %26 = lshr i64 %25, 1
  %27 = trunc i64 %26 to i32
  %28 = ptrtoint ptr addrspace(1) %1 to i64
  %29 = inttoptr i64 %28 to ptr
  %30 = sext i32 %27 to i64
  %31 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %29, i16 0, i64 %30, i32 159744)
  %32 = ptrtoint ptr addrspace(1) %2 to i64
  %33 = inttoptr i64 %32 to ptr
  %34 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %33, i16 0, i64 706478080, i32 159744)
  %35 = ptrtoint ptr addrspace(1) %8 to i64
  %36 = inttoptr i64 %35 to ptr
  %37 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %36, i16 0, i64 4, i32 159744)
  %38 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %37, i32 0, i32 0, i32 0)
  %39 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38)
  %40 = sext i32 %39 to i64
  %41 = lshr i64 %16, 5
  %42 = mul i64 %40, %41
  %43 = trunc i64 %42 to i32
  %44 = ptrtoint ptr addrspace(1) %3 to i64
  %45 = inttoptr i64 %44 to ptr
  %46 = sext i32 %43 to i64
  %47 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %45, i16 0, i64 %46, i32 159744)
  %48 = mul i64 %41, 2759680
  %49 = trunc i64 %48 to i32
  %50 = ptrtoint ptr addrspace(1) %4 to i64
  %51 = inttoptr i64 %50 to ptr
  %52 = sext i32 %49 to i64
  %53 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %51, i16 0, i64 %52, i32 159744)
  %54 = mul i64 %17, 32
  %55 = mul i64 %17, 128
  %56 = trunc i64 %55 to i32
  %57 = ptrtoint ptr addrspace(1) %5 to i64
  %58 = inttoptr i64 %57 to ptr
  %59 = sext i32 %56 to i64
  %60 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %58, i16 0, i64 %59, i32 159744)
  %61 = ptrtoint ptr addrspace(1) %7 to i64
  %62 = inttoptr i64 %61 to ptr
  %63 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %62, i16 0, i64 %59, i32 159744)
  %64 = add i64 %54, 31
  %65 = lshr i64 %64, 5
  %66 = mul i64 %65, 4
  %67 = trunc i64 %66 to i32
  %68 = ptrtoint ptr addrspace(1) %6 to i64
  %69 = inttoptr i64 %68 to ptr
  %70 = sext i32 %67 to i64
  %71 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %69, i16 0, i64 %70, i32 159744)
  %72 = add i64 %40, 31
  %73 = udiv i64 %72, 32
  %74 = add i64 %73, 255
  %75 = udiv i64 %74, 256
  br label %76

76:                                               ; preds = %1118, %14
  %77 = phi i64 [ %1119, %1118 ], [ 0, %14 ]
  %78 = phi i1 [ %97, %1118 ], [ true, %14 ]
  %79 = icmp slt i64 %77, %75
  br i1 %79, label %80, label %1120

80:                                               ; preds = %76
  %81 = mul i64 %23, %75
  %82 = add i64 %81, %77
  %83 = mul i64 %82, 32
  %84 = trunc i64 %83 to i32
  %85 = icmp ult i32 %84, %39
  %86 = lshr i64 %83, 5
  %87 = trunc i64 %86 to i32
  %88 = mul i32 %87, 4
  %89 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %71, i32 %88, i32 0, i32 0)
  %90 = sext i32 %89 to i64
  %91 = icmp ult i32 %89, 385
  %92 = mul i64 %90, 1835008
  %93 = mul i32 %84, 4
  %94 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %60, i32 %93, i32 0, i32 0)
  %95 = and i32 %94, 16777215
  %96 = icmp ult i32 %95, %10
  %97 = and i1 %78, %85
  %98 = and i1 %91, %96
  %99 = and i1 %97, %98
  br i1 %99, label %100, label %1118

100:                                              ; preds = %80
  %101 = mul i64 %90, 7168
  %102 = lshr i64 %16, 1
  %103 = lshr i64 %102, 2
  %104 = mul i64 %19, 4
  %105 = trunc i64 %104 to i32
  %106 = sdiv i32 %105, 32
  %107 = srem i32 %106, 32
  %108 = srem i32 %105, 32
  %109 = sext i32 %107 to i64
  %110 = sext i32 %108 to i64
  %111 = add i64 %83, %109
  %112 = trunc i64 %111 to i32
  %113 = mul i32 %112, 4
  %114 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %60, i32 %113, i32 0, i32 0)
  %115 = and i32 %114, 16777215
  %116 = lshr i32 %114, 24
  %117 = mul i32 %115, 9
  %118 = add i32 %117, %116
  %119 = sext i32 %118 to i64
  %120 = mul i64 %119, %103
  %121 = lshr i64 %19, 6
  %122 = and i64 %121, 3
  %123 = and i64 %19, 63
  %124 = lshr i64 %123, 4
  %125 = and i64 %124, 3
  %126 = and i64 %123, 15
  %127 = mul i64 %125, 16
  %128 = mul i64 %122, 64
  %129 = mul i64 %21, 256
  %130 = add i64 %129, %128
  %131 = add i64 %130, %126
  %132 = lshr i64 %131, 4
  %133 = and i64 %131, 15
  %134 = add i64 %130, 16
  %135 = add i64 %134, %126
  %136 = lshr i64 %135, 4
  %137 = and i64 %135, 15
  %138 = add i64 %130, 32
  %139 = add i64 %138, %126
  %140 = lshr i64 %139, 4
  %141 = and i64 %139, 15
  %142 = add i64 %130, 48
  %143 = add i64 %142, %126
  %144 = lshr i64 %143, 4
  %145 = and i64 %143, 15
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %146 = icmp ult i64 %19, 32
  br i1 %146, label %147, label %154

147:                                              ; preds = %100
  %148 = add i64 %83, %19
  %149 = trunc i64 %148 to i32
  %150 = mul i32 %149, 4
  %151 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %60, i32 %150, i32 0, i32 0)
  %152 = insertelement <1 x i32> poison, i32 %151, i32 0
  %153 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem0, i32 32768), i64 %19
  store <1 x i32> %152, ptr addrspace(3) %153, align 4
  br label %154

154:                                              ; preds = %147, %100
  %155 = mul i64 %132, 4096
  %156 = add i64 %92, %155
  %157 = mul i64 %125, 256
  %158 = add i64 %156, %157
  %159 = mul i64 %133, 16
  %160 = add i64 %158, %159
  %161 = lshr i64 %160, 2
  %162 = trunc i64 %161 to i32
  %163 = mul i32 %162, 4
  %164 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %163, i32 0, i32 2)
  %165 = bitcast <4 x i32> %164 to <2 x i64>
  %166 = extractelement <2 x i64> %165, i64 0
  %167 = extractelement <2 x i64> %165, i64 1
  %168 = mul i64 %136, 4096
  %169 = add i64 %92, %168
  %170 = add i64 %169, %157
  %171 = mul i64 %137, 16
  %172 = add i64 %170, %171
  %173 = lshr i64 %172, 2
  %174 = trunc i64 %173 to i32
  %175 = mul i32 %174, 4
  %176 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %175, i32 0, i32 2)
  %177 = bitcast <4 x i32> %176 to <2 x i64>
  %178 = extractelement <2 x i64> %177, i64 0
  %179 = extractelement <2 x i64> %177, i64 1
  %180 = mul i64 %140, 4096
  %181 = add i64 %92, %180
  %182 = add i64 %181, %157
  %183 = mul i64 %141, 16
  %184 = add i64 %182, %183
  %185 = lshr i64 %184, 2
  %186 = trunc i64 %185 to i32
  %187 = mul i32 %186, 4
  %188 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %187, i32 0, i32 2)
  %189 = bitcast <4 x i32> %188 to <2 x i64>
  %190 = extractelement <2 x i64> %189, i64 0
  %191 = extractelement <2 x i64> %189, i64 1
  %192 = mul i64 %144, 4096
  %193 = add i64 %92, %192
  %194 = add i64 %193, %157
  %195 = mul i64 %145, 16
  %196 = add i64 %194, %195
  %197 = lshr i64 %196, 2
  %198 = trunc i64 %197 to i32
  %199 = mul i32 %198, 4
  %200 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %199, i32 0, i32 2)
  %201 = bitcast <4 x i32> %200 to <2 x i64>
  %202 = extractelement <2 x i64> %201, i64 0
  %203 = extractelement <2 x i64> %201, i64 1
  %204 = lshr i64 %83, 1
  %205 = lshr i64 %204, 4
  %206 = mul i64 %205, 128
  %207 = add i64 %206, %127
  %208 = add i64 %207, %126
  %209 = trunc i64 %208 to i32
  %210 = mul i32 %209, 4
  %211 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %47, i32 %210, i32 0, i32 0)
  %212 = add i64 %101, %129
  %213 = add i64 %212, %128
  %214 = lshr i64 %213, 1
  %215 = lshr i64 %214, 4
  %216 = mul i64 %215, 128
  %217 = add i64 %216, %127
  %218 = add i64 %217, %126
  %219 = trunc i64 %218 to i32
  %220 = mul i32 %219, 4
  %221 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %53, i32 %220, i32 0, i32 0)
  %222 = add i64 %215, 1
  %223 = mul i64 %222, 128
  %224 = add i64 %223, %127
  %225 = add i64 %224, %126
  %226 = trunc i64 %225 to i32
  %227 = mul i32 %226, 4
  %228 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %53, i32 %227, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %229 = mul i64 %110, 4
  %230 = and i64 %109, 7
  %231 = mul i64 %230, 16
  %232 = xor i64 %229, %231
  %233 = mul i64 %120, 4
  %234 = add i64 %233, %232
  %235 = trunc i64 %234 to i32
  %236 = mul i64 %122, 1024
  %237 = add i64 ptrtoint (ptr addrspace(3) @smem0 to i64), %236
  %238 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %237)
  %239 = inttoptr i64 %238 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %31, ptr addrspace(3) %239, i32 16, i32 %235, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %240 = and i64 %126, 7
  %241 = mul i64 %240, 16
  %242 = xor i64 %127, %241
  %243 = mul i64 %126, 128
  %244 = add i64 %243, %242
  %245 = getelementptr i8, ptr addrspace(3) @smem0, i64 %244
  %246 = load <16 x i8>, ptr addrspace(3) %245, align 1
  %247 = bitcast <16 x i8> %246 to <2 x i64>
  %248 = extractelement <2 x i64> %247, i64 0
  %249 = extractelement <2 x i64> %247, i64 1
  %250 = add i64 %127, 64
  %251 = xor i64 %250, %241
  %252 = add i64 %243, %251
  %253 = getelementptr i8, ptr addrspace(3) @smem0, i64 %252
  %254 = load <16 x i8>, ptr addrspace(3) %253, align 1
  %255 = bitcast <16 x i8> %254 to <2 x i64>
  %256 = extractelement <2 x i64> %255, i64 0
  %257 = extractelement <2 x i64> %255, i64 1
  %258 = add i64 %16, 255
  %259 = udiv i64 %258, 256
  %260 = mul i64 %259, 256
  %261 = sub i64 %260, 256
  %262 = udiv i64 %261, 2
  %263 = lshr i64 %261, 1
  %264 = lshr i64 %263, 2
  %265 = add i64 %120, %264
  %266 = mul i64 %265, 4
  %267 = add i64 %266, %232
  %268 = trunc i64 %267 to i32
  %269 = add i64 ptrtoint (ptr addrspace(3) @smem1 to i64), %236
  %270 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %269)
  %271 = inttoptr i64 %270 to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %31, ptr addrspace(3) %271, i32 16, i32 %268, i32 0, i32 0, i32 0)
  %272 = lshr i64 %262, 6
  %273 = mul i64 %272, 1024
  %274 = add i64 %156, %273
  %275 = add i64 %274, %157
  %276 = add i64 %275, %159
  %277 = lshr i64 %276, 2
  %278 = trunc i64 %277 to i32
  %279 = mul i32 %278, 4
  %280 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %279, i32 0, i32 2)
  %281 = bitcast <4 x i32> %280 to <2 x i64>
  %282 = extractelement <2 x i64> %281, i64 0
  %283 = extractelement <2 x i64> %281, i64 1
  %284 = add i64 %169, %273
  %285 = add i64 %284, %157
  %286 = add i64 %285, %171
  %287 = lshr i64 %286, 2
  %288 = trunc i64 %287 to i32
  %289 = mul i32 %288, 4
  %290 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %289, i32 0, i32 2)
  %291 = bitcast <4 x i32> %290 to <2 x i64>
  %292 = extractelement <2 x i64> %291, i64 0
  %293 = extractelement <2 x i64> %291, i64 1
  %294 = add i64 %181, %273
  %295 = add i64 %294, %157
  %296 = add i64 %295, %183
  %297 = lshr i64 %296, 2
  %298 = trunc i64 %297 to i32
  %299 = mul i32 %298, 4
  %300 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %299, i32 0, i32 2)
  %301 = bitcast <4 x i32> %300 to <2 x i64>
  %302 = extractelement <2 x i64> %301, i64 0
  %303 = extractelement <2 x i64> %301, i64 1
  %304 = add i64 %193, %273
  %305 = add i64 %304, %157
  %306 = add i64 %305, %195
  %307 = lshr i64 %306, 2
  %308 = trunc i64 %307 to i32
  %309 = mul i32 %308, 4
  %310 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %309, i32 0, i32 2)
  %311 = bitcast <4 x i32> %310 to <2 x i64>
  %312 = extractelement <2 x i64> %311, i64 0
  %313 = extractelement <2 x i64> %311, i64 1
  %314 = add i64 %206, 64
  %315 = add i64 %314, %127
  %316 = add i64 %315, %126
  %317 = trunc i64 %316 to i32
  %318 = mul i32 %317, 4
  %319 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %47, i32 %318, i32 0, i32 0)
  %320 = add i64 %216, 64
  %321 = add i64 %320, %127
  %322 = add i64 %321, %126
  %323 = trunc i64 %322 to i32
  %324 = mul i32 %323, 4
  %325 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %53, i32 %324, i32 0, i32 0)
  %326 = add i64 %223, 64
  %327 = add i64 %326, %127
  %328 = add i64 %327, %126
  %329 = trunc i64 %328 to i32
  %330 = mul i32 %329, 4
  %331 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %53, i32 %330, i32 0, i32 0)
  %332 = add i64 %156, 1024
  %333 = add i64 %332, %157
  %334 = add i64 %333, %159
  %335 = lshr i64 %334, 2
  %336 = trunc i64 %335 to i32
  %337 = mul i32 %336, 4
  %338 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %337, i32 0, i32 2)
  %339 = bitcast <4 x i32> %338 to <2 x i64>
  %340 = extractelement <2 x i64> %339, i64 0
  %341 = extractelement <2 x i64> %339, i64 1
  %342 = add i64 %169, 1024
  %343 = add i64 %342, %157
  %344 = add i64 %343, %171
  %345 = lshr i64 %344, 2
  %346 = trunc i64 %345 to i32
  %347 = mul i32 %346, 4
  %348 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %347, i32 0, i32 2)
  %349 = bitcast <4 x i32> %348 to <2 x i64>
  %350 = extractelement <2 x i64> %349, i64 0
  %351 = extractelement <2 x i64> %349, i64 1
  %352 = add i64 %181, 1024
  %353 = add i64 %352, %157
  %354 = add i64 %353, %183
  %355 = lshr i64 %354, 2
  %356 = trunc i64 %355 to i32
  %357 = mul i32 %356, 4
  %358 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %357, i32 0, i32 2)
  %359 = bitcast <4 x i32> %358 to <2 x i64>
  %360 = extractelement <2 x i64> %359, i64 0
  %361 = extractelement <2 x i64> %359, i64 1
  %362 = add i64 %193, 1024
  %363 = add i64 %362, %157
  %364 = add i64 %363, %195
  %365 = lshr i64 %364, 2
  %366 = trunc i64 %365 to i32
  %367 = mul i32 %366, 4
  %368 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %367, i32 0, i32 2)
  %369 = bitcast <4 x i32> %368 to <2 x i64>
  %370 = extractelement <2 x i64> %369, i64 0
  %371 = extractelement <2 x i64> %369, i64 1
  %372 = insertelement <4 x i64> poison, i64 %248, i64 0
  %373 = insertelement <4 x i64> %372, i64 %249, i64 1
  %374 = insertelement <4 x i64> %373, i64 0, i64 2
  %375 = insertelement <4 x i64> %374, i64 0, i64 3
  %376 = bitcast <4 x i64> %375 to <8 x i32>
  %377 = insertelement <4 x i64> poison, i64 %166, i64 0
  %378 = insertelement <4 x i64> %377, i64 %167, i64 1
  %379 = insertelement <4 x i64> %378, i64 0, i64 2
  %380 = insertelement <4 x i64> %379, i64 0, i64 3
  %381 = bitcast <4 x i64> %380 to <8 x i32>
  %382 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %376, <8 x i32> %381, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %211, i32 0, i32 %221)
  %383 = insertelement <4 x i64> poison, i64 %178, i64 0
  %384 = insertelement <4 x i64> %383, i64 %179, i64 1
  %385 = insertelement <4 x i64> %384, i64 0, i64 2
  %386 = insertelement <4 x i64> %385, i64 0, i64 3
  %387 = bitcast <4 x i64> %386 to <8 x i32>
  %388 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %376, <8 x i32> %387, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %211, i32 1, i32 %221)
  %389 = add i64 %126, 16
  %390 = and i64 %389, 7
  %391 = mul i64 %390, 16
  %392 = xor i64 %127, %391
  %393 = mul i64 %389, 128
  %394 = add i64 %393, %392
  %395 = getelementptr i8, ptr addrspace(3) @smem0, i64 %394
  %396 = load <16 x i8>, ptr addrspace(3) %395, align 1
  %397 = bitcast <16 x i8> %396 to <2 x i64>
  %398 = extractelement <2 x i64> %397, i64 0
  %399 = extractelement <2 x i64> %397, i64 1
  %400 = insertelement <4 x i64> poison, i64 %398, i64 0
  %401 = insertelement <4 x i64> %400, i64 %399, i64 1
  %402 = insertelement <4 x i64> %401, i64 0, i64 2
  %403 = insertelement <4 x i64> %402, i64 0, i64 3
  %404 = bitcast <4 x i64> %403 to <8 x i32>
  %405 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %404, <8 x i32> %381, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %211, i32 0, i32 %221)
  %406 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %404, <8 x i32> %387, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %211, i32 1, i32 %221)
  %407 = insertelement <4 x i64> poison, i64 %190, i64 0
  %408 = insertelement <4 x i64> %407, i64 %191, i64 1
  %409 = insertelement <4 x i64> %408, i64 0, i64 2
  %410 = insertelement <4 x i64> %409, i64 0, i64 3
  %411 = bitcast <4 x i64> %410 to <8 x i32>
  %412 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %376, <8 x i32> %411, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %211, i32 0, i32 %228)
  %413 = insertelement <4 x i64> poison, i64 %202, i64 0
  %414 = insertelement <4 x i64> %413, i64 %203, i64 1
  %415 = insertelement <4 x i64> %414, i64 0, i64 2
  %416 = insertelement <4 x i64> %415, i64 0, i64 3
  %417 = bitcast <4 x i64> %416 to <8 x i32>
  %418 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %376, <8 x i32> %417, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 %211, i32 1, i32 %228)
  %419 = getelementptr i8, ptr addrspace(3) @smem0, i64 %394
  %420 = load <16 x i8>, ptr addrspace(3) %419, align 1
  %421 = bitcast <16 x i8> %420 to <2 x i64>
  %422 = extractelement <2 x i64> %421, i64 0
  %423 = extractelement <2 x i64> %421, i64 1
  %424 = insertelement <4 x i64> poison, i64 %422, i64 0
  %425 = insertelement <4 x i64> %424, i64 %423, i64 1
  %426 = insertelement <4 x i64> %425, i64 0, i64 2
  %427 = insertelement <4 x i64> %426, i64 0, i64 3
  %428 = bitcast <4 x i64> %427 to <8 x i32>
  %429 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %428, <8 x i32> %411, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %211, i32 0, i32 %228)
  %430 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %428, <8 x i32> %417, <4 x float> zeroinitializer, i32 4, i32 4, i32 1, i32 %211, i32 1, i32 %228)
  %431 = insertelement <4 x i64> poison, i64 %256, i64 0
  %432 = insertelement <4 x i64> %431, i64 %257, i64 1
  %433 = insertelement <4 x i64> %432, i64 0, i64 2
  %434 = insertelement <4 x i64> %433, i64 0, i64 3
  %435 = bitcast <4 x i64> %434 to <8 x i32>
  %436 = insertelement <4 x i64> poison, i64 %340, i64 0
  %437 = insertelement <4 x i64> %436, i64 %341, i64 1
  %438 = insertelement <4 x i64> %437, i64 0, i64 2
  %439 = insertelement <4 x i64> %438, i64 0, i64 3
  %440 = bitcast <4 x i64> %439 to <8 x i32>
  %441 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %435, <8 x i32> %440, <4 x float> %382, i32 4, i32 4, i32 2, i32 %211, i32 2, i32 %221)
  %442 = insertelement <4 x i64> poison, i64 %350, i64 0
  %443 = insertelement <4 x i64> %442, i64 %351, i64 1
  %444 = insertelement <4 x i64> %443, i64 0, i64 2
  %445 = insertelement <4 x i64> %444, i64 0, i64 3
  %446 = bitcast <4 x i64> %445 to <8 x i32>
  %447 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %435, <8 x i32> %446, <4 x float> %388, i32 4, i32 4, i32 2, i32 %211, i32 3, i32 %221)
  %448 = xor i64 %250, %391
  %449 = add i64 %393, %448
  %450 = getelementptr i8, ptr addrspace(3) @smem0, i64 %449
  %451 = load <16 x i8>, ptr addrspace(3) %450, align 1
  %452 = bitcast <16 x i8> %451 to <2 x i64>
  %453 = extractelement <2 x i64> %452, i64 0
  %454 = extractelement <2 x i64> %452, i64 1
  %455 = insertelement <4 x i64> poison, i64 %453, i64 0
  %456 = insertelement <4 x i64> %455, i64 %454, i64 1
  %457 = insertelement <4 x i64> %456, i64 0, i64 2
  %458 = insertelement <4 x i64> %457, i64 0, i64 3
  %459 = bitcast <4 x i64> %458 to <8 x i32>
  %460 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %459, <8 x i32> %440, <4 x float> %405, i32 4, i32 4, i32 3, i32 %211, i32 2, i32 %221)
  %461 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %459, <8 x i32> %446, <4 x float> %406, i32 4, i32 4, i32 3, i32 %211, i32 3, i32 %221)
  %462 = insertelement <4 x i64> poison, i64 %360, i64 0
  %463 = insertelement <4 x i64> %462, i64 %361, i64 1
  %464 = insertelement <4 x i64> %463, i64 0, i64 2
  %465 = insertelement <4 x i64> %464, i64 0, i64 3
  %466 = bitcast <4 x i64> %465 to <8 x i32>
  %467 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %435, <8 x i32> %466, <4 x float> %412, i32 4, i32 4, i32 2, i32 %211, i32 2, i32 %228)
  %468 = insertelement <4 x i64> poison, i64 %370, i64 0
  %469 = insertelement <4 x i64> %468, i64 %371, i64 1
  %470 = insertelement <4 x i64> %469, i64 0, i64 2
  %471 = insertelement <4 x i64> %470, i64 0, i64 3
  %472 = bitcast <4 x i64> %471 to <8 x i32>
  %473 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %435, <8 x i32> %472, <4 x float> %418, i32 4, i32 4, i32 2, i32 %211, i32 3, i32 %228)
  %474 = getelementptr i8, ptr addrspace(3) @smem0, i64 %449
  %475 = load <16 x i8>, ptr addrspace(3) %474, align 1
  %476 = bitcast <16 x i8> %475 to <2 x i64>
  %477 = extractelement <2 x i64> %476, i64 0
  %478 = extractelement <2 x i64> %476, i64 1
  %479 = insertelement <4 x i64> poison, i64 %477, i64 0
  %480 = insertelement <4 x i64> %479, i64 %478, i64 1
  %481 = insertelement <4 x i64> %480, i64 0, i64 2
  %482 = insertelement <4 x i64> %481, i64 0, i64 3
  %483 = bitcast <4 x i64> %482 to <8 x i32>
  %484 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %483, <8 x i32> %466, <4 x float> %429, i32 4, i32 4, i32 3, i32 %211, i32 2, i32 %228)
  %485 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %483, <8 x i32> %472, <4 x float> %430, i32 4, i32 4, i32 3, i32 %211, i32 3, i32 %228)
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %486 = getelementptr i8, ptr addrspace(3) @smem1, i64 %244
  %487 = load <16 x i8>, ptr addrspace(3) %486, align 1
  %488 = bitcast <16 x i8> %487 to <2 x i64>
  %489 = extractelement <2 x i64> %488, i64 0
  %490 = extractelement <2 x i64> %488, i64 1
  %491 = getelementptr i8, ptr addrspace(3) @smem1, i64 %252
  %492 = load <16 x i8>, ptr addrspace(3) %491, align 1
  %493 = bitcast <16 x i8> %492 to <2 x i64>
  %494 = extractelement <2 x i64> %493, i64 0
  %495 = extractelement <2 x i64> %493, i64 1
  %496 = mul i64 %125, 4
  %497 = add i64 %83, %496
  %498 = trunc i64 %497 to i32
  %499 = mul i32 %498, 4
  %500 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %499, i32 0, i32 0)
  %501 = add i64 %496, 1
  %502 = add i64 %83, %501
  %503 = trunc i64 %502 to i32
  %504 = mul i32 %503, 4
  %505 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %504, i32 0, i32 0)
  %506 = add i64 %496, 2
  %507 = add i64 %83, %506
  %508 = trunc i64 %507 to i32
  %509 = mul i32 %508, 4
  %510 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %509, i32 0, i32 0)
  %511 = add i64 %496, 3
  %512 = add i64 %83, %511
  %513 = trunc i64 %512 to i32
  %514 = mul i32 %513, 4
  %515 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %514, i32 0, i32 0)
  %516 = add i64 %496, 16
  %517 = add i64 %83, %516
  %518 = trunc i64 %517 to i32
  %519 = mul i32 %518, 4
  %520 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %519, i32 0, i32 0)
  %521 = add i64 %496, 17
  %522 = add i64 %83, %521
  %523 = trunc i64 %522 to i32
  %524 = mul i32 %523, 4
  %525 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %524, i32 0, i32 0)
  %526 = add i64 %496, 18
  %527 = add i64 %83, %526
  %528 = trunc i64 %527 to i32
  %529 = mul i32 %528, 4
  %530 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %529, i32 0, i32 0)
  %531 = add i64 %496, 19
  %532 = add i64 %83, %531
  %533 = trunc i64 %532 to i32
  %534 = mul i32 %533, 4
  %535 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %63, i32 %534, i32 0, i32 0)
  %536 = add i64 %272, 1
  %537 = mul i64 %536, 1024
  %538 = add i64 %156, %537
  %539 = add i64 %538, %157
  %540 = add i64 %539, %159
  %541 = lshr i64 %540, 2
  %542 = trunc i64 %541 to i32
  %543 = mul i32 %542, 4
  %544 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %543, i32 0, i32 2)
  %545 = bitcast <4 x i32> %544 to <2 x i64>
  %546 = extractelement <2 x i64> %545, i64 0
  %547 = extractelement <2 x i64> %545, i64 1
  %548 = add i64 %169, %537
  %549 = add i64 %548, %157
  %550 = add i64 %549, %171
  %551 = lshr i64 %550, 2
  %552 = trunc i64 %551 to i32
  %553 = mul i32 %552, 4
  %554 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %553, i32 0, i32 2)
  %555 = bitcast <4 x i32> %554 to <2 x i64>
  %556 = extractelement <2 x i64> %555, i64 0
  %557 = extractelement <2 x i64> %555, i64 1
  %558 = add i64 %181, %537
  %559 = add i64 %558, %157
  %560 = add i64 %559, %183
  %561 = lshr i64 %560, 2
  %562 = trunc i64 %561 to i32
  %563 = mul i32 %562, 4
  %564 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %563, i32 0, i32 2)
  %565 = bitcast <4 x i32> %564 to <2 x i64>
  %566 = extractelement <2 x i64> %565, i64 0
  %567 = extractelement <2 x i64> %565, i64 1
  %568 = add i64 %193, %537
  %569 = add i64 %568, %157
  %570 = add i64 %569, %195
  %571 = lshr i64 %570, 2
  %572 = trunc i64 %571 to i32
  %573 = mul i32 %572, 4
  %574 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %34, i32 %573, i32 0, i32 2)
  %575 = bitcast <4 x i32> %574 to <2 x i64>
  %576 = extractelement <2 x i64> %575, i64 0
  %577 = extractelement <2 x i64> %575, i64 1
  %578 = insertelement <4 x i64> poison, i64 %489, i64 0
  %579 = insertelement <4 x i64> %578, i64 %490, i64 1
  %580 = insertelement <4 x i64> %579, i64 0, i64 2
  %581 = insertelement <4 x i64> %580, i64 0, i64 3
  %582 = bitcast <4 x i64> %581 to <8 x i32>
  %583 = insertelement <4 x i64> poison, i64 %282, i64 0
  %584 = insertelement <4 x i64> %583, i64 %283, i64 1
  %585 = insertelement <4 x i64> %584, i64 0, i64 2
  %586 = insertelement <4 x i64> %585, i64 0, i64 3
  %587 = bitcast <4 x i64> %586 to <8 x i32>
  %588 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %582, <8 x i32> %587, <4 x float> %441, i32 4, i32 4, i32 0, i32 %319, i32 0, i32 %325)
  %589 = insertelement <4 x i64> poison, i64 %292, i64 0
  %590 = insertelement <4 x i64> %589, i64 %293, i64 1
  %591 = insertelement <4 x i64> %590, i64 0, i64 2
  %592 = insertelement <4 x i64> %591, i64 0, i64 3
  %593 = bitcast <4 x i64> %592 to <8 x i32>
  %594 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %582, <8 x i32> %593, <4 x float> %447, i32 4, i32 4, i32 0, i32 %319, i32 1, i32 %325)
  %595 = getelementptr i8, ptr addrspace(3) @smem1, i64 %394
  %596 = load <16 x i8>, ptr addrspace(3) %595, align 1
  %597 = bitcast <16 x i8> %596 to <2 x i64>
  %598 = extractelement <2 x i64> %597, i64 0
  %599 = extractelement <2 x i64> %597, i64 1
  %600 = insertelement <4 x i64> poison, i64 %598, i64 0
  %601 = insertelement <4 x i64> %600, i64 %599, i64 1
  %602 = insertelement <4 x i64> %601, i64 0, i64 2
  %603 = insertelement <4 x i64> %602, i64 0, i64 3
  %604 = bitcast <4 x i64> %603 to <8 x i32>
  %605 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %604, <8 x i32> %587, <4 x float> %460, i32 4, i32 4, i32 1, i32 %319, i32 0, i32 %325)
  %606 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %604, <8 x i32> %593, <4 x float> %461, i32 4, i32 4, i32 1, i32 %319, i32 1, i32 %325)
  %607 = insertelement <4 x i64> poison, i64 %302, i64 0
  %608 = insertelement <4 x i64> %607, i64 %303, i64 1
  %609 = insertelement <4 x i64> %608, i64 0, i64 2
  %610 = insertelement <4 x i64> %609, i64 0, i64 3
  %611 = bitcast <4 x i64> %610 to <8 x i32>
  %612 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %582, <8 x i32> %611, <4 x float> %467, i32 4, i32 4, i32 0, i32 %319, i32 0, i32 %331)
  %613 = insertelement <4 x i64> poison, i64 %312, i64 0
  %614 = insertelement <4 x i64> %613, i64 %313, i64 1
  %615 = insertelement <4 x i64> %614, i64 0, i64 2
  %616 = insertelement <4 x i64> %615, i64 0, i64 3
  %617 = bitcast <4 x i64> %616 to <8 x i32>
  %618 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %582, <8 x i32> %617, <4 x float> %473, i32 4, i32 4, i32 0, i32 %319, i32 1, i32 %331)
  %619 = getelementptr i8, ptr addrspace(3) @smem1, i64 %394
  %620 = load <16 x i8>, ptr addrspace(3) %619, align 1
  %621 = bitcast <16 x i8> %620 to <2 x i64>
  %622 = extractelement <2 x i64> %621, i64 0
  %623 = extractelement <2 x i64> %621, i64 1
  %624 = insertelement <4 x i64> poison, i64 %622, i64 0
  %625 = insertelement <4 x i64> %624, i64 %623, i64 1
  %626 = insertelement <4 x i64> %625, i64 0, i64 2
  %627 = insertelement <4 x i64> %626, i64 0, i64 3
  %628 = bitcast <4 x i64> %627 to <8 x i32>
  %629 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %628, <8 x i32> %611, <4 x float> %484, i32 4, i32 4, i32 1, i32 %319, i32 0, i32 %331)
  %630 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %628, <8 x i32> %617, <4 x float> %485, i32 4, i32 4, i32 1, i32 %319, i32 1, i32 %331)
  %631 = insertelement <4 x i64> poison, i64 %494, i64 0
  %632 = insertelement <4 x i64> %631, i64 %495, i64 1
  %633 = insertelement <4 x i64> %632, i64 0, i64 2
  %634 = insertelement <4 x i64> %633, i64 0, i64 3
  %635 = bitcast <4 x i64> %634 to <8 x i32>
  %636 = insertelement <4 x i64> poison, i64 %546, i64 0
  %637 = insertelement <4 x i64> %636, i64 %547, i64 1
  %638 = insertelement <4 x i64> %637, i64 0, i64 2
  %639 = insertelement <4 x i64> %638, i64 0, i64 3
  %640 = bitcast <4 x i64> %639 to <8 x i32>
  %641 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %635, <8 x i32> %640, <4 x float> %588, i32 4, i32 4, i32 2, i32 %319, i32 2, i32 %325)
  %642 = insertelement <4 x i64> poison, i64 %556, i64 0
  %643 = insertelement <4 x i64> %642, i64 %557, i64 1
  %644 = insertelement <4 x i64> %643, i64 0, i64 2
  %645 = insertelement <4 x i64> %644, i64 0, i64 3
  %646 = bitcast <4 x i64> %645 to <8 x i32>
  %647 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %635, <8 x i32> %646, <4 x float> %594, i32 4, i32 4, i32 2, i32 %319, i32 3, i32 %325)
  %648 = getelementptr i8, ptr addrspace(3) @smem1, i64 %449
  %649 = load <16 x i8>, ptr addrspace(3) %648, align 1
  %650 = bitcast <16 x i8> %649 to <2 x i64>
  %651 = extractelement <2 x i64> %650, i64 0
  %652 = extractelement <2 x i64> %650, i64 1
  %653 = insertelement <4 x i64> poison, i64 %651, i64 0
  %654 = insertelement <4 x i64> %653, i64 %652, i64 1
  %655 = insertelement <4 x i64> %654, i64 0, i64 2
  %656 = insertelement <4 x i64> %655, i64 0, i64 3
  %657 = bitcast <4 x i64> %656 to <8 x i32>
  %658 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %657, <8 x i32> %640, <4 x float> %605, i32 4, i32 4, i32 3, i32 %319, i32 2, i32 %325)
  %659 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %657, <8 x i32> %646, <4 x float> %606, i32 4, i32 4, i32 3, i32 %319, i32 3, i32 %325)
  %660 = insertelement <4 x i64> poison, i64 %566, i64 0
  %661 = insertelement <4 x i64> %660, i64 %567, i64 1
  %662 = insertelement <4 x i64> %661, i64 0, i64 2
  %663 = insertelement <4 x i64> %662, i64 0, i64 3
  %664 = bitcast <4 x i64> %663 to <8 x i32>
  %665 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %635, <8 x i32> %664, <4 x float> %612, i32 4, i32 4, i32 2, i32 %319, i32 2, i32 %331)
  %666 = insertelement <4 x i64> poison, i64 %576, i64 0
  %667 = insertelement <4 x i64> %666, i64 %577, i64 1
  %668 = insertelement <4 x i64> %667, i64 0, i64 2
  %669 = insertelement <4 x i64> %668, i64 0, i64 3
  %670 = bitcast <4 x i64> %669 to <8 x i32>
  %671 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %635, <8 x i32> %670, <4 x float> %618, i32 4, i32 4, i32 2, i32 %319, i32 3, i32 %331)
  %672 = getelementptr i8, ptr addrspace(3) @smem1, i64 %449
  %673 = load <16 x i8>, ptr addrspace(3) %672, align 1
  %674 = bitcast <16 x i8> %673 to <2 x i64>
  %675 = extractelement <2 x i64> %674, i64 0
  %676 = extractelement <2 x i64> %674, i64 1
  %677 = insertelement <4 x i64> poison, i64 %675, i64 0
  %678 = insertelement <4 x i64> %677, i64 %676, i64 1
  %679 = insertelement <4 x i64> %678, i64 0, i64 2
  %680 = insertelement <4 x i64> %679, i64 0, i64 3
  %681 = bitcast <4 x i64> %680 to <8 x i32>
  %682 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %681, <8 x i32> %664, <4 x float> %629, i32 4, i32 4, i32 3, i32 %319, i32 2, i32 %331)
  %683 = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %681, <8 x i32> %670, <4 x float> %630, i32 4, i32 4, i32 3, i32 %319, i32 3, i32 %331)
  %684 = ptrtoint ptr addrspace(1) %0 to i64
  %685 = add i64 %128, %126
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %686 = mul i64 %125, 1024
  %687 = extractelement <4 x float> %641, i64 0
  %688 = extractelement <4 x float> %647, i64 0
  %689 = extractelement <4 x float> %665, i64 0
  %690 = extractelement <4 x float> %671, i64 0
  %691 = insertelement <4 x float> poison, float %687, i64 0
  %692 = insertelement <4 x float> %691, float %688, i64 1
  %693 = insertelement <4 x float> %692, float %689, i64 2
  %694 = insertelement <4 x float> %693, float %690, i64 3
  %695 = insertelement <4 x float> poison, float %500, i32 0
  %696 = shufflevector <4 x float> %695, <4 x float> poison, <4 x i32> zeroinitializer
  %697 = fmul <4 x float> %694, %696
  %698 = extractelement <4 x float> %697, i64 0
  %699 = add i64 %686, %685
  %700 = insertelement <1 x float> poison, float %698, i32 0
  %701 = getelementptr float, ptr addrspace(3) @smem0, i64 %699
  store <1 x float> %700, ptr addrspace(3) %701, align 4
  %702 = add i64 %685, 16
  %703 = extractelement <4 x float> %697, i64 1
  %704 = add i64 %686, %702
  %705 = insertelement <1 x float> poison, float %703, i32 0
  %706 = getelementptr float, ptr addrspace(3) @smem0, i64 %704
  store <1 x float> %705, ptr addrspace(3) %706, align 4
  %707 = add i64 %685, 32
  %708 = extractelement <4 x float> %697, i64 2
  %709 = add i64 %686, %707
  %710 = insertelement <1 x float> poison, float %708, i32 0
  %711 = getelementptr float, ptr addrspace(3) @smem0, i64 %709
  store <1 x float> %710, ptr addrspace(3) %711, align 4
  %712 = add i64 %685, 48
  %713 = extractelement <4 x float> %697, i64 3
  %714 = add i64 %686, %712
  %715 = insertelement <1 x float> poison, float %713, i32 0
  %716 = getelementptr float, ptr addrspace(3) @smem0, i64 %714
  store <1 x float> %715, ptr addrspace(3) %716, align 4
  %717 = mul i64 %501, 256
  %718 = extractelement <4 x float> %641, i64 1
  %719 = extractelement <4 x float> %647, i64 1
  %720 = extractelement <4 x float> %665, i64 1
  %721 = extractelement <4 x float> %671, i64 1
  %722 = insertelement <4 x float> poison, float %718, i64 0
  %723 = insertelement <4 x float> %722, float %719, i64 1
  %724 = insertelement <4 x float> %723, float %720, i64 2
  %725 = insertelement <4 x float> %724, float %721, i64 3
  %726 = insertelement <4 x float> poison, float %505, i32 0
  %727 = shufflevector <4 x float> %726, <4 x float> poison, <4 x i32> zeroinitializer
  %728 = fmul <4 x float> %725, %727
  %729 = extractelement <4 x float> %728, i64 0
  %730 = add i64 %717, %685
  %731 = insertelement <1 x float> poison, float %729, i32 0
  %732 = getelementptr float, ptr addrspace(3) @smem0, i64 %730
  store <1 x float> %731, ptr addrspace(3) %732, align 4
  %733 = extractelement <4 x float> %728, i64 1
  %734 = add i64 %717, %702
  %735 = insertelement <1 x float> poison, float %733, i32 0
  %736 = getelementptr float, ptr addrspace(3) @smem0, i64 %734
  store <1 x float> %735, ptr addrspace(3) %736, align 4
  %737 = extractelement <4 x float> %728, i64 2
  %738 = add i64 %717, %707
  %739 = insertelement <1 x float> poison, float %737, i32 0
  %740 = getelementptr float, ptr addrspace(3) @smem0, i64 %738
  store <1 x float> %739, ptr addrspace(3) %740, align 4
  %741 = extractelement <4 x float> %728, i64 3
  %742 = add i64 %717, %712
  %743 = insertelement <1 x float> poison, float %741, i32 0
  %744 = getelementptr float, ptr addrspace(3) @smem0, i64 %742
  store <1 x float> %743, ptr addrspace(3) %744, align 4
  %745 = mul i64 %506, 256
  %746 = extractelement <4 x float> %641, i64 2
  %747 = extractelement <4 x float> %647, i64 2
  %748 = extractelement <4 x float> %665, i64 2
  %749 = extractelement <4 x float> %671, i64 2
  %750 = insertelement <4 x float> poison, float %746, i64 0
  %751 = insertelement <4 x float> %750, float %747, i64 1
  %752 = insertelement <4 x float> %751, float %748, i64 2
  %753 = insertelement <4 x float> %752, float %749, i64 3
  %754 = insertelement <4 x float> poison, float %510, i32 0
  %755 = shufflevector <4 x float> %754, <4 x float> poison, <4 x i32> zeroinitializer
  %756 = fmul <4 x float> %753, %755
  %757 = extractelement <4 x float> %756, i64 0
  %758 = add i64 %745, %685
  %759 = insertelement <1 x float> poison, float %757, i32 0
  %760 = getelementptr float, ptr addrspace(3) @smem0, i64 %758
  store <1 x float> %759, ptr addrspace(3) %760, align 4
  %761 = extractelement <4 x float> %756, i64 1
  %762 = add i64 %745, %702
  %763 = insertelement <1 x float> poison, float %761, i32 0
  %764 = getelementptr float, ptr addrspace(3) @smem0, i64 %762
  store <1 x float> %763, ptr addrspace(3) %764, align 4
  %765 = extractelement <4 x float> %756, i64 2
  %766 = add i64 %745, %707
  %767 = insertelement <1 x float> poison, float %765, i32 0
  %768 = getelementptr float, ptr addrspace(3) @smem0, i64 %766
  store <1 x float> %767, ptr addrspace(3) %768, align 4
  %769 = extractelement <4 x float> %756, i64 3
  %770 = add i64 %745, %712
  %771 = insertelement <1 x float> poison, float %769, i32 0
  %772 = getelementptr float, ptr addrspace(3) @smem0, i64 %770
  store <1 x float> %771, ptr addrspace(3) %772, align 4
  %773 = mul i64 %511, 256
  %774 = extractelement <4 x float> %641, i64 3
  %775 = extractelement <4 x float> %647, i64 3
  %776 = extractelement <4 x float> %665, i64 3
  %777 = extractelement <4 x float> %671, i64 3
  %778 = insertelement <4 x float> poison, float %774, i64 0
  %779 = insertelement <4 x float> %778, float %775, i64 1
  %780 = insertelement <4 x float> %779, float %776, i64 2
  %781 = insertelement <4 x float> %780, float %777, i64 3
  %782 = insertelement <4 x float> poison, float %515, i32 0
  %783 = shufflevector <4 x float> %782, <4 x float> poison, <4 x i32> zeroinitializer
  %784 = fmul <4 x float> %781, %783
  %785 = extractelement <4 x float> %784, i64 0
  %786 = add i64 %773, %685
  %787 = insertelement <1 x float> poison, float %785, i32 0
  %788 = getelementptr float, ptr addrspace(3) @smem0, i64 %786
  store <1 x float> %787, ptr addrspace(3) %788, align 4
  %789 = extractelement <4 x float> %784, i64 1
  %790 = add i64 %773, %702
  %791 = insertelement <1 x float> poison, float %789, i32 0
  %792 = getelementptr float, ptr addrspace(3) @smem0, i64 %790
  store <1 x float> %791, ptr addrspace(3) %792, align 4
  %793 = extractelement <4 x float> %784, i64 2
  %794 = add i64 %773, %707
  %795 = insertelement <1 x float> poison, float %793, i32 0
  %796 = getelementptr float, ptr addrspace(3) @smem0, i64 %794
  store <1 x float> %795, ptr addrspace(3) %796, align 4
  %797 = extractelement <4 x float> %784, i64 3
  %798 = add i64 %773, %712
  %799 = insertelement <1 x float> poison, float %797, i32 0
  %800 = getelementptr float, ptr addrspace(3) @smem0, i64 %798
  store <1 x float> %799, ptr addrspace(3) %800, align 4
  %801 = mul i64 %516, 256
  %802 = extractelement <4 x float> %658, i64 0
  %803 = extractelement <4 x float> %659, i64 0
  %804 = extractelement <4 x float> %682, i64 0
  %805 = extractelement <4 x float> %683, i64 0
  %806 = insertelement <4 x float> poison, float %802, i64 0
  %807 = insertelement <4 x float> %806, float %803, i64 1
  %808 = insertelement <4 x float> %807, float %804, i64 2
  %809 = insertelement <4 x float> %808, float %805, i64 3
  %810 = insertelement <4 x float> poison, float %520, i32 0
  %811 = shufflevector <4 x float> %810, <4 x float> poison, <4 x i32> zeroinitializer
  %812 = fmul <4 x float> %809, %811
  %813 = extractelement <4 x float> %812, i64 0
  %814 = add i64 %801, %685
  %815 = insertelement <1 x float> poison, float %813, i32 0
  %816 = getelementptr float, ptr addrspace(3) @smem0, i64 %814
  store <1 x float> %815, ptr addrspace(3) %816, align 4
  %817 = extractelement <4 x float> %812, i64 1
  %818 = add i64 %801, %702
  %819 = insertelement <1 x float> poison, float %817, i32 0
  %820 = getelementptr float, ptr addrspace(3) @smem0, i64 %818
  store <1 x float> %819, ptr addrspace(3) %820, align 4
  %821 = extractelement <4 x float> %812, i64 2
  %822 = add i64 %801, %707
  %823 = insertelement <1 x float> poison, float %821, i32 0
  %824 = getelementptr float, ptr addrspace(3) @smem0, i64 %822
  store <1 x float> %823, ptr addrspace(3) %824, align 4
  %825 = extractelement <4 x float> %812, i64 3
  %826 = add i64 %801, %712
  %827 = insertelement <1 x float> poison, float %825, i32 0
  %828 = getelementptr float, ptr addrspace(3) @smem0, i64 %826
  store <1 x float> %827, ptr addrspace(3) %828, align 4
  %829 = mul i64 %521, 256
  %830 = extractelement <4 x float> %658, i64 1
  %831 = extractelement <4 x float> %659, i64 1
  %832 = extractelement <4 x float> %682, i64 1
  %833 = extractelement <4 x float> %683, i64 1
  %834 = insertelement <4 x float> poison, float %830, i64 0
  %835 = insertelement <4 x float> %834, float %831, i64 1
  %836 = insertelement <4 x float> %835, float %832, i64 2
  %837 = insertelement <4 x float> %836, float %833, i64 3
  %838 = insertelement <4 x float> poison, float %525, i32 0
  %839 = shufflevector <4 x float> %838, <4 x float> poison, <4 x i32> zeroinitializer
  %840 = fmul <4 x float> %837, %839
  %841 = extractelement <4 x float> %840, i64 0
  %842 = add i64 %829, %685
  %843 = insertelement <1 x float> poison, float %841, i32 0
  %844 = getelementptr float, ptr addrspace(3) @smem0, i64 %842
  store <1 x float> %843, ptr addrspace(3) %844, align 4
  %845 = extractelement <4 x float> %840, i64 1
  %846 = add i64 %829, %702
  %847 = insertelement <1 x float> poison, float %845, i32 0
  %848 = getelementptr float, ptr addrspace(3) @smem0, i64 %846
  store <1 x float> %847, ptr addrspace(3) %848, align 4
  %849 = extractelement <4 x float> %840, i64 2
  %850 = add i64 %829, %707
  %851 = insertelement <1 x float> poison, float %849, i32 0
  %852 = getelementptr float, ptr addrspace(3) @smem0, i64 %850
  store <1 x float> %851, ptr addrspace(3) %852, align 4
  %853 = extractelement <4 x float> %840, i64 3
  %854 = add i64 %829, %712
  %855 = insertelement <1 x float> poison, float %853, i32 0
  %856 = getelementptr float, ptr addrspace(3) @smem0, i64 %854
  store <1 x float> %855, ptr addrspace(3) %856, align 4
  %857 = mul i64 %526, 256
  %858 = extractelement <4 x float> %658, i64 2
  %859 = extractelement <4 x float> %659, i64 2
  %860 = extractelement <4 x float> %682, i64 2
  %861 = extractelement <4 x float> %683, i64 2
  %862 = insertelement <4 x float> poison, float %858, i64 0
  %863 = insertelement <4 x float> %862, float %859, i64 1
  %864 = insertelement <4 x float> %863, float %860, i64 2
  %865 = insertelement <4 x float> %864, float %861, i64 3
  %866 = insertelement <4 x float> poison, float %530, i32 0
  %867 = shufflevector <4 x float> %866, <4 x float> poison, <4 x i32> zeroinitializer
  %868 = fmul <4 x float> %865, %867
  %869 = extractelement <4 x float> %868, i64 0
  %870 = add i64 %857, %685
  %871 = insertelement <1 x float> poison, float %869, i32 0
  %872 = getelementptr float, ptr addrspace(3) @smem0, i64 %870
  store <1 x float> %871, ptr addrspace(3) %872, align 4
  %873 = extractelement <4 x float> %868, i64 1
  %874 = add i64 %857, %702
  %875 = insertelement <1 x float> poison, float %873, i32 0
  %876 = getelementptr float, ptr addrspace(3) @smem0, i64 %874
  store <1 x float> %875, ptr addrspace(3) %876, align 4
  %877 = extractelement <4 x float> %868, i64 2
  %878 = add i64 %857, %707
  %879 = insertelement <1 x float> poison, float %877, i32 0
  %880 = getelementptr float, ptr addrspace(3) @smem0, i64 %878
  store <1 x float> %879, ptr addrspace(3) %880, align 4
  %881 = extractelement <4 x float> %868, i64 3
  %882 = add i64 %857, %712
  %883 = insertelement <1 x float> poison, float %881, i32 0
  %884 = getelementptr float, ptr addrspace(3) @smem0, i64 %882
  store <1 x float> %883, ptr addrspace(3) %884, align 4
  %885 = mul i64 %531, 256
  %886 = extractelement <4 x float> %658, i64 3
  %887 = extractelement <4 x float> %659, i64 3
  %888 = extractelement <4 x float> %682, i64 3
  %889 = extractelement <4 x float> %683, i64 3
  %890 = insertelement <4 x float> poison, float %886, i64 0
  %891 = insertelement <4 x float> %890, float %887, i64 1
  %892 = insertelement <4 x float> %891, float %888, i64 2
  %893 = insertelement <4 x float> %892, float %889, i64 3
  %894 = insertelement <4 x float> poison, float %535, i32 0
  %895 = shufflevector <4 x float> %894, <4 x float> poison, <4 x i32> zeroinitializer
  %896 = fmul <4 x float> %893, %895
  %897 = extractelement <4 x float> %896, i64 0
  %898 = add i64 %885, %685
  %899 = insertelement <1 x float> poison, float %897, i32 0
  %900 = getelementptr float, ptr addrspace(3) @smem0, i64 %898
  store <1 x float> %899, ptr addrspace(3) %900, align 4
  %901 = extractelement <4 x float> %896, i64 1
  %902 = add i64 %885, %702
  %903 = insertelement <1 x float> poison, float %901, i32 0
  %904 = getelementptr float, ptr addrspace(3) @smem0, i64 %902
  store <1 x float> %903, ptr addrspace(3) %904, align 4
  %905 = extractelement <4 x float> %896, i64 2
  %906 = add i64 %885, %707
  %907 = insertelement <1 x float> poison, float %905, i32 0
  %908 = getelementptr float, ptr addrspace(3) @smem0, i64 %906
  store <1 x float> %907, ptr addrspace(3) %908, align 4
  %909 = extractelement <4 x float> %896, i64 3
  %910 = add i64 %885, %712
  %911 = insertelement <1 x float> poison, float %909, i32 0
  %912 = getelementptr float, ptr addrspace(3) @smem0, i64 %910
  store <1 x float> %911, ptr addrspace(3) %912, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %913 = udiv i64 %19, 32
  %914 = urem i64 %19, 32
  %915 = getelementptr inbounds nuw i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem0, i32 32768), i64 %913
  %916 = load i32, ptr addrspace(3) %915, align 4
  %917 = and i32 %916, 16777215
  %918 = icmp ult i32 %917, %10
  %919 = sext i32 %917 to i64
  %920 = mul i64 %919, 14336
  %921 = add i64 %684, %920
  %922 = add i64 %913, 8
  %923 = getelementptr inbounds nuw i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem0, i32 32768), i64 %922
  %924 = load i32, ptr addrspace(3) %923, align 4
  %925 = and i32 %924, 16777215
  %926 = icmp ult i32 %925, %10
  %927 = sext i32 %925 to i64
  %928 = mul i64 %927, 14336
  %929 = add i64 %684, %928
  %930 = add i64 %913, 16
  %931 = getelementptr inbounds nuw i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem0, i32 32768), i64 %930
  %932 = load i32, ptr addrspace(3) %931, align 4
  %933 = and i32 %932, 16777215
  %934 = icmp ult i32 %933, %10
  %935 = sext i32 %933 to i64
  %936 = mul i64 %935, 14336
  %937 = add i64 %684, %936
  %938 = add i64 %913, 24
  %939 = getelementptr inbounds nuw i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @smem0, i32 32768), i64 %938
  %940 = load i32, ptr addrspace(3) %939, align 4
  %941 = and i32 %940, 16777215
  %942 = icmp ult i32 %941, %10
  %943 = sext i32 %941 to i64
  %944 = mul i64 %943, 14336
  %945 = add i64 %684, %944
  br i1 %918, label %946, label %988

946:                                              ; preds = %154
  %947 = mul i64 %913, 256
  %948 = mul i64 %914, 2
  %949 = add i64 %947, %948
  %950 = getelementptr float, ptr addrspace(3) @smem0, i64 %949
  %951 = load <2 x float>, ptr addrspace(3) %950, align 4
  %952 = add i64 %129, %948
  %953 = fptrunc <2 x float> %951 to <2 x bfloat>
  %954 = mul i64 %952, 2
  %955 = add i64 %921, %954
  %956 = inttoptr i64 %955 to ptr addrspace(1)
  %957 = atomicrmw fadd ptr addrspace(1) %956, <2 x bfloat> %953 syncscope("agent") monotonic, align 4
  %958 = add i64 %948, 64
  %959 = add i64 %947, %958
  %960 = getelementptr float, ptr addrspace(3) @smem0, i64 %959
  %961 = load <2 x float>, ptr addrspace(3) %960, align 4
  %962 = add i64 %129, %958
  %963 = fptrunc <2 x float> %961 to <2 x bfloat>
  %964 = mul i64 %962, 2
  %965 = add i64 %921, %964
  %966 = inttoptr i64 %965 to ptr addrspace(1)
  %967 = atomicrmw fadd ptr addrspace(1) %966, <2 x bfloat> %963 syncscope("agent") monotonic, align 4
  %968 = add i64 %948, 128
  %969 = add i64 %947, %968
  %970 = getelementptr float, ptr addrspace(3) @smem0, i64 %969
  %971 = load <2 x float>, ptr addrspace(3) %970, align 4
  %972 = add i64 %129, %968
  %973 = fptrunc <2 x float> %971 to <2 x bfloat>
  %974 = mul i64 %972, 2
  %975 = add i64 %921, %974
  %976 = inttoptr i64 %975 to ptr addrspace(1)
  %977 = atomicrmw fadd ptr addrspace(1) %976, <2 x bfloat> %973 syncscope("agent") monotonic, align 4
  %978 = add i64 %948, 192
  %979 = add i64 %947, %978
  %980 = getelementptr float, ptr addrspace(3) @smem0, i64 %979
  %981 = load <2 x float>, ptr addrspace(3) %980, align 4
  %982 = add i64 %129, %978
  %983 = fptrunc <2 x float> %981 to <2 x bfloat>
  %984 = mul i64 %982, 2
  %985 = add i64 %921, %984
  %986 = inttoptr i64 %985 to ptr addrspace(1)
  %987 = atomicrmw fadd ptr addrspace(1) %986, <2 x bfloat> %983 syncscope("agent") monotonic, align 4
  br label %988

988:                                              ; preds = %946, %154
  br i1 %926, label %989, label %1031

989:                                              ; preds = %988
  %990 = mul i64 %922, 256
  %991 = mul i64 %914, 2
  %992 = add i64 %990, %991
  %993 = getelementptr float, ptr addrspace(3) @smem0, i64 %992
  %994 = load <2 x float>, ptr addrspace(3) %993, align 4
  %995 = add i64 %129, %991
  %996 = fptrunc <2 x float> %994 to <2 x bfloat>
  %997 = mul i64 %995, 2
  %998 = add i64 %929, %997
  %999 = inttoptr i64 %998 to ptr addrspace(1)
  %1000 = atomicrmw fadd ptr addrspace(1) %999, <2 x bfloat> %996 syncscope("agent") monotonic, align 4
  %1001 = add i64 %991, 64
  %1002 = add i64 %990, %1001
  %1003 = getelementptr float, ptr addrspace(3) @smem0, i64 %1002
  %1004 = load <2 x float>, ptr addrspace(3) %1003, align 4
  %1005 = add i64 %129, %1001
  %1006 = fptrunc <2 x float> %1004 to <2 x bfloat>
  %1007 = mul i64 %1005, 2
  %1008 = add i64 %929, %1007
  %1009 = inttoptr i64 %1008 to ptr addrspace(1)
  %1010 = atomicrmw fadd ptr addrspace(1) %1009, <2 x bfloat> %1006 syncscope("agent") monotonic, align 4
  %1011 = add i64 %991, 128
  %1012 = add i64 %990, %1011
  %1013 = getelementptr float, ptr addrspace(3) @smem0, i64 %1012
  %1014 = load <2 x float>, ptr addrspace(3) %1013, align 4
  %1015 = add i64 %129, %1011
  %1016 = fptrunc <2 x float> %1014 to <2 x bfloat>
  %1017 = mul i64 %1015, 2
  %1018 = add i64 %929, %1017
  %1019 = inttoptr i64 %1018 to ptr addrspace(1)
  %1020 = atomicrmw fadd ptr addrspace(1) %1019, <2 x bfloat> %1016 syncscope("agent") monotonic, align 4
  %1021 = add i64 %991, 192
  %1022 = add i64 %990, %1021
  %1023 = getelementptr float, ptr addrspace(3) @smem0, i64 %1022
  %1024 = load <2 x float>, ptr addrspace(3) %1023, align 4
  %1025 = add i64 %129, %1021
  %1026 = fptrunc <2 x float> %1024 to <2 x bfloat>
  %1027 = mul i64 %1025, 2
  %1028 = add i64 %929, %1027
  %1029 = inttoptr i64 %1028 to ptr addrspace(1)
  %1030 = atomicrmw fadd ptr addrspace(1) %1029, <2 x bfloat> %1026 syncscope("agent") monotonic, align 4
  br label %1031

1031:                                             ; preds = %989, %988
  br i1 %934, label %1032, label %1074

1032:                                             ; preds = %1031
  %1033 = mul i64 %930, 256
  %1034 = mul i64 %914, 2
  %1035 = add i64 %1033, %1034
  %1036 = getelementptr float, ptr addrspace(3) @smem0, i64 %1035
  %1037 = load <2 x float>, ptr addrspace(3) %1036, align 4
  %1038 = add i64 %129, %1034
  %1039 = fptrunc <2 x float> %1037 to <2 x bfloat>
  %1040 = mul i64 %1038, 2
  %1041 = add i64 %937, %1040
  %1042 = inttoptr i64 %1041 to ptr addrspace(1)
  %1043 = atomicrmw fadd ptr addrspace(1) %1042, <2 x bfloat> %1039 syncscope("agent") monotonic, align 4
  %1044 = add i64 %1034, 64
  %1045 = add i64 %1033, %1044
  %1046 = getelementptr float, ptr addrspace(3) @smem0, i64 %1045
  %1047 = load <2 x float>, ptr addrspace(3) %1046, align 4
  %1048 = add i64 %129, %1044
  %1049 = fptrunc <2 x float> %1047 to <2 x bfloat>
  %1050 = mul i64 %1048, 2
  %1051 = add i64 %937, %1050
  %1052 = inttoptr i64 %1051 to ptr addrspace(1)
  %1053 = atomicrmw fadd ptr addrspace(1) %1052, <2 x bfloat> %1049 syncscope("agent") monotonic, align 4
  %1054 = add i64 %1034, 128
  %1055 = add i64 %1033, %1054
  %1056 = getelementptr float, ptr addrspace(3) @smem0, i64 %1055
  %1057 = load <2 x float>, ptr addrspace(3) %1056, align 4
  %1058 = add i64 %129, %1054
  %1059 = fptrunc <2 x float> %1057 to <2 x bfloat>
  %1060 = mul i64 %1058, 2
  %1061 = add i64 %937, %1060
  %1062 = inttoptr i64 %1061 to ptr addrspace(1)
  %1063 = atomicrmw fadd ptr addrspace(1) %1062, <2 x bfloat> %1059 syncscope("agent") monotonic, align 4
  %1064 = add i64 %1034, 192
  %1065 = add i64 %1033, %1064
  %1066 = getelementptr float, ptr addrspace(3) @smem0, i64 %1065
  %1067 = load <2 x float>, ptr addrspace(3) %1066, align 4
  %1068 = add i64 %129, %1064
  %1069 = fptrunc <2 x float> %1067 to <2 x bfloat>
  %1070 = mul i64 %1068, 2
  %1071 = add i64 %937, %1070
  %1072 = inttoptr i64 %1071 to ptr addrspace(1)
  %1073 = atomicrmw fadd ptr addrspace(1) %1072, <2 x bfloat> %1069 syncscope("agent") monotonic, align 4
  br label %1074

1074:                                             ; preds = %1032, %1031
  br i1 %942, label %1075, label %1117

1075:                                             ; preds = %1074
  %1076 = mul i64 %938, 256
  %1077 = mul i64 %914, 2
  %1078 = add i64 %1076, %1077
  %1079 = getelementptr float, ptr addrspace(3) @smem0, i64 %1078
  %1080 = load <2 x float>, ptr addrspace(3) %1079, align 4
  %1081 = add i64 %129, %1077
  %1082 = fptrunc <2 x float> %1080 to <2 x bfloat>
  %1083 = mul i64 %1081, 2
  %1084 = add i64 %945, %1083
  %1085 = inttoptr i64 %1084 to ptr addrspace(1)
  %1086 = atomicrmw fadd ptr addrspace(1) %1085, <2 x bfloat> %1082 syncscope("agent") monotonic, align 4
  %1087 = add i64 %1077, 64
  %1088 = add i64 %1076, %1087
  %1089 = getelementptr float, ptr addrspace(3) @smem0, i64 %1088
  %1090 = load <2 x float>, ptr addrspace(3) %1089, align 4
  %1091 = add i64 %129, %1087
  %1092 = fptrunc <2 x float> %1090 to <2 x bfloat>
  %1093 = mul i64 %1091, 2
  %1094 = add i64 %945, %1093
  %1095 = inttoptr i64 %1094 to ptr addrspace(1)
  %1096 = atomicrmw fadd ptr addrspace(1) %1095, <2 x bfloat> %1092 syncscope("agent") monotonic, align 4
  %1097 = add i64 %1077, 128
  %1098 = add i64 %1076, %1097
  %1099 = getelementptr float, ptr addrspace(3) @smem0, i64 %1098
  %1100 = load <2 x float>, ptr addrspace(3) %1099, align 4
  %1101 = add i64 %129, %1097
  %1102 = fptrunc <2 x float> %1100 to <2 x bfloat>
  %1103 = mul i64 %1101, 2
  %1104 = add i64 %945, %1103
  %1105 = inttoptr i64 %1104 to ptr addrspace(1)
  %1106 = atomicrmw fadd ptr addrspace(1) %1105, <2 x bfloat> %1102 syncscope("agent") monotonic, align 4
  %1107 = add i64 %1077, 192
  %1108 = add i64 %1076, %1107
  %1109 = getelementptr float, ptr addrspace(3) @smem0, i64 %1108
  %1110 = load <2 x float>, ptr addrspace(3) %1109, align 4
  %1111 = add i64 %129, %1107
  %1112 = fptrunc <2 x float> %1110 to <2 x bfloat>
  %1113 = mul i64 %1111, 2
  %1114 = add i64 %945, %1113
  %1115 = inttoptr i64 %1114 to ptr addrspace(1)
  %1116 = atomicrmw fadd ptr addrspace(1) %1115, <2 x bfloat> %1112 syncscope("agent") monotonic, align 4
  br label %1117

1117:                                             ; preds = %1075, %1074
  br label %1118

1118:                                             ; preds = %1117, %80
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1119 = add i64 %77, 1
  br label %76

1120:                                             ; preds = %76
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.readfirstlane.i64(i64) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="4" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
