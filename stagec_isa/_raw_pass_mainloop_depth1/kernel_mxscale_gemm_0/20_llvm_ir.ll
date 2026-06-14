; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@mxscale_a8w4_32x32x256_2x2_2buf_arena = external addrspace(3) global [28672 x i8], align 1024

define amdgpu_kernel void @kernel_mxscale_gemm_0(ptr addrspace(1) %0, <{ <{ i32, i32 }>, <{ i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32 }>, <{ i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32 }>, <{ i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32 }>, <{ i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32, i32 }>, <{ i64 }> }> %9, i32 %10, i32 %11, i32 %12, i32 %13) #0 !reqd_work_group_size !1 {
  call void @llvm.amdgcn.s.setreg(i32 282, i32 1)
  %15 = call range(i32 0, 128) i32 @llvm.amdgcn.workitem.id.x()
  %16 = sext i32 %15 to i64
  %17 = call i32 @llvm.amdgcn.workgroup.id.x()
  %18 = sext i32 %17 to i64
  %19 = call i32 @llvm.amdgcn.workgroup.id.y()
  %20 = sext i32 %19 to i64
  %21 = mul i64 %18, 32
  %22 = mul i64 %20, 32
  %23 = trunc i64 %16 to i32
  %24 = sdiv i32 %23, 64
  %25 = srem i32 %24, 2
  %26 = sdiv i32 %23, 32
  %27 = srem i32 %26, 2
  %28 = sdiv i32 %23, 16
  %29 = srem i32 %28, 2
  %30 = srem i32 %23, 16
  %31 = sext i32 %25 to i64
  %32 = sext i32 %27 to i64
  %33 = sext i32 %29 to i64
  %34 = sext i32 %30 to i64
  %35 = mul i64 %31, 16
  %36 = mul i64 %32, 16
  %37 = sext i32 %10 to i64
  %38 = sext i32 %12 to i64
  %39 = sext i32 %13 to i64
  %40 = mul i64 %37, %39
  %41 = mul i64 %40, 2
  %42 = addrspacecast ptr addrspace(1) %0 to ptr
  %43 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %42, i16 0, i64 %41, i32 159744)
  %44 = call i32 @llvm.amdgcn.wave.id()
  %45 = icmp eq i32 %44, 0
  %46 = icmp eq i32 %44, 1
  %47 = icmp eq i32 %44, 2
  %48 = mul i64 %31, 2
  %49 = add i64 %48, %32
  %50 = mul i64 %49, 384
  %51 = mul i64 %34, 24
  %52 = add i64 %50, %51
  %53 = mul i64 %33, 8
  %54 = add i64 %52, %53
  %55 = sext i32 %44 to i64
  %56 = udiv i64 %55, 2
  %57 = urem i64 %55, 2
  %58 = mul i64 %56, 2
  %59 = add i64 %58, %57
  %60 = mul i64 %59, 768
  %61 = mul i64 %56, 16
  %62 = mul i64 %57, 16
  %63 = add i64 %21, %61
  %64 = add i64 %22, %62
  %65 = ptrtoint ptr addrspace(1) %0 to i64
  %66 = mul i64 %63, %39
  %67 = add i64 %66, %64
  %68 = mul i64 %67, 2
  %69 = add i64 %65, %68
  %70 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena to i64), %60
  %71 = trunc i64 %70 to i32
  %72 = trunc i64 %69 to i32
  %73 = lshr i64 %69, 32
  %74 = trunc i64 %73 to i32
  %75 = or i32 %74, -2147483648
  %76 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %71, i64 1
  %77 = insertelement <4 x i32> %76, i32 %72, i64 2
  %78 = insertelement <4 x i32> %77, i32 %75, i64 3
  %79 = trunc i64 %63 to i32
  %80 = sub i32 %10, %79
  %81 = call i32 @llvm.smax.i32(i32 %80, i32 0)
  %82 = and i32 %81, 65535
  %83 = shl i32 %82, 16
  %84 = lshr i32 %81, 16
  %85 = and i32 %84, 65535
  %86 = or i32 %85, 1572864
  %87 = insertelement <8 x i32> <i32 65536, i32 1048576, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %83, i64 2
  %88 = insertelement <8 x i32> %87, i32 %86, i64 3
  %89 = insertelement <8 x i32> %88, i32 16, i64 4
  %90 = insertelement <8 x i32> %89, i32 %13, i64 5
  %91 = insertelement <8 x i32> %90, i32 0, i64 6
  %92 = insertelement <8 x i32> %91, i32 0, i64 7
  %93 = ptrtoint ptr addrspace(1) %2 to i64
  %94 = mul i64 %21, %38
  %95 = add i64 %93, %94
  %96 = trunc i64 %95 to i32
  %97 = lshr i64 %95, 32
  %98 = trunc i64 %97 to i32
  %99 = or i32 %98, -2147483648
  %100 = trunc i64 %21 to i32
  %101 = sub i32 %10, %100
  %102 = call i32 @llvm.smax.i32(i32 %101, i32 0)
  %103 = and i32 %102, 65535
  %104 = shl i32 %103, 16
  %105 = lshr i32 %102, 16
  %106 = and i32 %105, 65535
  %107 = or i32 %106, 16777216
  %108 = insertelement <8 x i32> <i32 124780544, i32 16777216, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %104, i64 2
  %109 = insertelement <8 x i32> %108, i32 %107, i64 3
  %110 = insertelement <8 x i32> %109, i32 32, i64 4
  %111 = insertelement <8 x i32> %110, i32 %12, i64 5
  %112 = insertelement <8 x i32> %111, i32 0, i64 6
  %113 = insertelement <8 x i32> %112, i32 0, i64 7
  %114 = udiv i64 %22, 32
  %115 = ptrtoint ptr addrspace(1) %4 to i64
  %116 = mul i64 %114, 16384
  %117 = add i64 %115, %116
  %118 = trunc i64 %117 to i32
  %119 = lshr i64 %117, 32
  %120 = trunc i64 %119 to i32
  %121 = or i32 %120, -2147483648
  %122 = ptrtoint ptr addrspace(1) %6 to i64
  %123 = mul i64 %18, 1024
  %124 = add i64 %122, %123
  %125 = trunc i64 %124 to i32
  %126 = lshr i64 %124, 32
  %127 = trunc i64 %126 to i32
  %128 = or i32 %127, -2147483648
  %129 = ptrtoint ptr addrspace(1) %8 to i64
  %130 = mul i64 %20, 1024
  %131 = add i64 %129, %130
  %132 = trunc i64 %131 to i32
  %133 = lshr i64 %131, 32
  %134 = trunc i64 %133 to i32
  %135 = or i32 %134, -2147483648
  %136 = icmp slt i32 %44, 4
  %137 = zext i1 %136 to i32
  %138 = select i1 %47, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13056) to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13328) to i32)
  %139 = select i1 %46, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 8704) to i32), i32 %138
  %140 = select i1 %45, i32 ptrtoint (ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena to i32), i32 %139
  %141 = select i1 %47, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27392) to i32), i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27664) to i32)
  %142 = select i1 %46, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 23040) to i32), i32 %141
  %143 = select i1 %45, i32 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 14336) to i32), i32 %142
  %144 = select i1 %47, i32 %125, i32 %132
  %145 = select i1 %46, i32 %118, i32 %144
  %146 = select i1 %45, i32 %96, i32 %145
  %147 = select i1 %47, i32 %128, i32 %135
  %148 = select i1 %46, i32 %121, i32 %147
  %149 = select i1 %45, i32 %99, i32 %148
  %150 = select i1 %46, <8 x i32> <i32 124780544, i32 16777216, i32 1048576, i32 16777216, i32 16, i32 256, i32 0, i32 0>, <8 x i32> <i32 2097152, i32 524288, i32 2097152, i32 524288, i32 32, i32 32, i32 0, i32 0>
  %151 = select i1 %45, <8 x i32> %113, <8 x i32> %150
  %152 = select i1 %46, i32 4096, i32 8
  %153 = select i1 %45, i32 256, i32 %152
  %154 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %137)
  %155 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149)
  %156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %140)
  %157 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %143)
  %158 = insertelement <2 x i32> poison, i32 %154, i64 0
  %159 = insertelement <2 x i32> %158, i32 %156, i64 1
  %160 = insertelement <2 x i32> poison, i32 %146, i64 0
  %161 = insertelement <2 x i32> %160, i32 %155, i64 1
  %162 = shufflevector <2 x i32> %159, <2 x i32> %161, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %162, <8 x i32> %151, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %163 = add i32 %146, %153
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  %164 = add i64 %35, %34
  %165 = mul i64 %164, 272
  %166 = mul i64 %33, 16
  %167 = add i64 %165, %166
  %168 = mul i64 %34, 16
  %169 = mul i64 %33, 544
  %170 = mul i64 %32, 272
  %171 = add i64 %170, %168
  %172 = add i64 %171, %169
  %173 = mul i64 %164, 8
  %174 = mul i64 %33, 4
  %175 = add i64 %173, %174
  %176 = add i64 %36, %34
  %177 = mul i64 %176, 8
  %178 = add i64 %177, %174
  %179 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 8704) to i64), %172
  %180 = trunc i64 %179 to i32
  %181 = inttoptr i32 %180 to ptr addrspace(3)
  %182 = load <4 x i32>, ptr addrspace(3) %181, align 16
  %183 = getelementptr i8, ptr addrspace(3) %181, i32 1088
  %184 = load <4 x i32>, ptr addrspace(3) %183, align 16
  %185 = shufflevector <4 x i32> %182, <4 x i32> %184, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %186 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13056) to i64), %175
  %187 = trunc i64 %186 to i32
  %188 = inttoptr i32 %187 to ptr addrspace(3)
  %189 = load <4 x i32>, ptr addrspace(3) %188, align 16
  %190 = extractelement <4 x i32> %189, i64 0
  %191 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 13328) to i64), %178
  %192 = trunc i64 %191 to i32
  %193 = inttoptr i32 %192 to ptr addrspace(3)
  %194 = load <4 x i32>, ptr addrspace(3) %193, align 16
  %195 = extractelement <4 x i32> %194, i64 0
  %196 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena to i64), %167
  %197 = trunc i64 %196 to i32
  %198 = inttoptr i32 %197 to ptr addrspace(3)
  %199 = load <4 x i32>, ptr addrspace(3) %198, align 16
  %200 = getelementptr i8, ptr addrspace(3) %198, i32 32
  %201 = load <4 x i32>, ptr addrspace(3) %200, align 16
  %202 = getelementptr i8, ptr addrspace(3) %198, i32 64
  %203 = load <4 x i32>, ptr addrspace(3) %202, align 16
  %204 = getelementptr i8, ptr addrspace(3) %198, i32 96
  %205 = load <4 x i32>, ptr addrspace(3) %204, align 16
  %206 = shufflevector <4 x i32> %199, <4 x i32> %201, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %207 = shufflevector <4 x i32> %203, <4 x i32> %205, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %208 = shufflevector <8 x i32> %206, <8 x i32> %207, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %209 = getelementptr i8, ptr addrspace(3) %181, i32 2176
  %210 = load <4 x i32>, ptr addrspace(3) %209, align 16
  %211 = getelementptr i8, ptr addrspace(3) %181, i32 3264
  %212 = load <4 x i32>, ptr addrspace(3) %211, align 16
  %213 = shufflevector <4 x i32> %210, <4 x i32> %212, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %214 = getelementptr i8, ptr addrspace(3) %188, i32 4
  %215 = load <4 x i32>, ptr addrspace(3) %214, align 16
  %216 = extractelement <4 x i32> %215, i64 0
  %217 = getelementptr i8, ptr addrspace(3) %193, i32 4
  %218 = load <4 x i32>, ptr addrspace(3) %217, align 16
  %219 = extractelement <4 x i32> %218, i64 0
  %220 = getelementptr i8, ptr addrspace(3) %198, i32 128
  %221 = load <4 x i32>, ptr addrspace(3) %220, align 16
  %222 = getelementptr i8, ptr addrspace(3) %198, i32 160
  %223 = load <4 x i32>, ptr addrspace(3) %222, align 16
  %224 = getelementptr i8, ptr addrspace(3) %198, i32 192
  %225 = load <4 x i32>, ptr addrspace(3) %224, align 16
  %226 = getelementptr i8, ptr addrspace(3) %198, i32 224
  %227 = load <4 x i32>, ptr addrspace(3) %226, align 16
  %228 = shufflevector <4 x i32> %221, <4 x i32> %223, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %229 = shufflevector <4 x i32> %225, <4 x i32> %227, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %230 = shufflevector <8 x i32> %228, <8 x i32> %229, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %231 = insertelement <2 x i32> poison, i32 %154, i64 0
  %232 = insertelement <2 x i32> %231, i32 %157, i64 1
  %233 = insertelement <2 x i32> poison, i32 %163, i64 0
  %234 = insertelement <2 x i32> %233, i32 %155, i64 1
  %235 = shufflevector <2 x i32> %232, <2 x i32> %234, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %235, <8 x i32> %151, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %236 = add i32 %163, %153
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %237 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27392) to i64), %175
  %238 = trunc i64 %237 to i32
  %239 = inttoptr i32 %238 to ptr addrspace(3)
  %240 = load <4 x i32>, ptr addrspace(3) %239, align 16
  %241 = extractelement <4 x i32> %240, i64 0
  %242 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 27664) to i64), %178
  %243 = trunc i64 %242 to i32
  %244 = inttoptr i32 %243 to ptr addrspace(3)
  %245 = load <4 x i32>, ptr addrspace(3) %244, align 16
  %246 = extractelement <4 x i32> %245, i64 0
  %247 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 14336) to i64), %167
  %248 = trunc i64 %247 to i32
  %249 = inttoptr i32 %248 to ptr addrspace(3)
  %250 = load <4 x i32>, ptr addrspace(3) %249, align 16
  %251 = getelementptr i8, ptr addrspace(3) %249, i32 32
  %252 = load <4 x i32>, ptr addrspace(3) %251, align 16
  %253 = getelementptr i8, ptr addrspace(3) %249, i32 64
  %254 = load <4 x i32>, ptr addrspace(3) %253, align 16
  %255 = getelementptr i8, ptr addrspace(3) %249, i32 96
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16
  %257 = shufflevector <4 x i32> %250, <4 x i32> %252, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %258 = shufflevector <4 x i32> %254, <4 x i32> %256, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %259 = shufflevector <8 x i32> %257, <8 x i32> %258, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %260 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i32 23040) to i64), %172
  %261 = trunc i64 %260 to i32
  %262 = inttoptr i32 %261 to ptr addrspace(3)
  %263 = load <4 x i32>, ptr addrspace(3) %262, align 16
  %264 = getelementptr i8, ptr addrspace(3) %262, i32 1088
  %265 = load <4 x i32>, ptr addrspace(3) %264, align 16
  %266 = shufflevector <4 x i32> %263, <4 x i32> %265, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %267 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %185, i32 0, <16 x i32> %208, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 %195, i32 0, i32 0, i32 %190, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %268 = getelementptr i8, ptr addrspace(3) %239, i32 4
  %269 = load <4 x i32>, ptr addrspace(3) %268, align 16
  %270 = extractelement <4 x i32> %269, i64 0
  %271 = getelementptr i8, ptr addrspace(3) %244, i32 4
  %272 = load <4 x i32>, ptr addrspace(3) %271, align 16
  %273 = extractelement <4 x i32> %272, i64 0
  %274 = getelementptr i8, ptr addrspace(3) %249, i32 128
  %275 = load <4 x i32>, ptr addrspace(3) %274, align 16
  %276 = getelementptr i8, ptr addrspace(3) %249, i32 160
  %277 = load <4 x i32>, ptr addrspace(3) %276, align 16
  %278 = getelementptr i8, ptr addrspace(3) %249, i32 192
  %279 = load <4 x i32>, ptr addrspace(3) %278, align 16
  %280 = getelementptr i8, ptr addrspace(3) %249, i32 224
  %281 = load <4 x i32>, ptr addrspace(3) %280, align 16
  %282 = shufflevector <4 x i32> %275, <4 x i32> %277, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %283 = shufflevector <4 x i32> %279, <4 x i32> %281, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %284 = shufflevector <8 x i32> %282, <8 x i32> %283, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %285 = getelementptr i8, ptr addrspace(3) %262, i32 2176
  %286 = load <4 x i32>, ptr addrspace(3) %285, align 16
  %287 = getelementptr i8, ptr addrspace(3) %262, i32 3264
  %288 = load <4 x i32>, ptr addrspace(3) %287, align 16
  %289 = shufflevector <4 x i32> %286, <4 x i32> %288, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %290 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %213, i32 0, <16 x i32> %230, i16 0, <8 x float> %267, i32 0, i32 0, i32 %219, i32 0, i32 0, i32 %216, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %291 = insertelement <2 x i32> poison, i32 %236, i64 0
  %292 = insertelement <2 x i32> %291, i32 %155, i64 1
  %293 = shufflevector <2 x i32> %159, <2 x i32> %292, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %293, <8 x i32> %151, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %294 = add i32 %236, %153
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %295 = load <4 x i32>, ptr addrspace(3) %188, align 16
  %296 = extractelement <4 x i32> %295, i64 0
  %297 = load <4 x i32>, ptr addrspace(3) %193, align 16
  %298 = extractelement <4 x i32> %297, i64 0
  %299 = load <4 x i32>, ptr addrspace(3) %198, align 16
  %300 = load <4 x i32>, ptr addrspace(3) %200, align 16
  %301 = load <4 x i32>, ptr addrspace(3) %202, align 16
  %302 = load <4 x i32>, ptr addrspace(3) %204, align 16
  %303 = shufflevector <4 x i32> %299, <4 x i32> %300, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %304 = shufflevector <4 x i32> %301, <4 x i32> %302, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %305 = shufflevector <8 x i32> %303, <8 x i32> %304, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %306 = load <4 x i32>, ptr addrspace(3) %181, align 16
  %307 = load <4 x i32>, ptr addrspace(3) %183, align 16
  %308 = shufflevector <4 x i32> %306, <4 x i32> %307, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %309 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %266, i32 0, <16 x i32> %259, i16 0, <8 x float> %290, i32 0, i32 0, i32 %246, i32 0, i32 0, i32 %241, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %310 = load <4 x i32>, ptr addrspace(3) %214, align 16
  %311 = extractelement <4 x i32> %310, i64 0
  %312 = load <4 x i32>, ptr addrspace(3) %217, align 16
  %313 = extractelement <4 x i32> %312, i64 0
  %314 = load <4 x i32>, ptr addrspace(3) %220, align 16
  %315 = load <4 x i32>, ptr addrspace(3) %222, align 16
  %316 = load <4 x i32>, ptr addrspace(3) %224, align 16
  %317 = load <4 x i32>, ptr addrspace(3) %226, align 16
  %318 = shufflevector <4 x i32> %314, <4 x i32> %315, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %319 = shufflevector <4 x i32> %316, <4 x i32> %317, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %320 = shufflevector <8 x i32> %318, <8 x i32> %319, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %321 = load <4 x i32>, ptr addrspace(3) %209, align 16
  %322 = load <4 x i32>, ptr addrspace(3) %211, align 16
  %323 = shufflevector <4 x i32> %321, <4 x i32> %322, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %324 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %289, i32 0, <16 x i32> %284, i16 0, <8 x float> %309, i32 0, i32 0, i32 %273, i32 0, i32 0, i32 %270, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %325 = insertelement <2 x i32> poison, i32 %294, i64 0
  %326 = insertelement <2 x i32> %325, i32 %155, i64 1
  %327 = shufflevector <2 x i32> %232, <2 x i32> %326, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %327, <8 x i32> %151, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 1)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %328 = load <4 x i32>, ptr addrspace(3) %239, align 16
  %329 = extractelement <4 x i32> %328, i64 0
  %330 = load <4 x i32>, ptr addrspace(3) %244, align 16
  %331 = extractelement <4 x i32> %330, i64 0
  %332 = load <4 x i32>, ptr addrspace(3) %249, align 16
  %333 = load <4 x i32>, ptr addrspace(3) %251, align 16
  %334 = load <4 x i32>, ptr addrspace(3) %253, align 16
  %335 = load <4 x i32>, ptr addrspace(3) %255, align 16
  %336 = shufflevector <4 x i32> %332, <4 x i32> %333, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %337 = shufflevector <4 x i32> %334, <4 x i32> %335, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %338 = shufflevector <8 x i32> %336, <8 x i32> %337, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %339 = load <4 x i32>, ptr addrspace(3) %262, align 16
  %340 = load <4 x i32>, ptr addrspace(3) %264, align 16
  %341 = shufflevector <4 x i32> %339, <4 x i32> %340, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %342 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %308, i32 0, <16 x i32> %305, i16 0, <8 x float> %324, i32 0, i32 0, i32 %298, i32 0, i32 0, i32 %296, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %343 = load <4 x i32>, ptr addrspace(3) %268, align 16
  %344 = extractelement <4 x i32> %343, i64 0
  %345 = load <4 x i32>, ptr addrspace(3) %271, align 16
  %346 = extractelement <4 x i32> %345, i64 0
  %347 = load <4 x i32>, ptr addrspace(3) %274, align 16
  %348 = load <4 x i32>, ptr addrspace(3) %276, align 16
  %349 = load <4 x i32>, ptr addrspace(3) %278, align 16
  %350 = load <4 x i32>, ptr addrspace(3) %280, align 16
  %351 = shufflevector <4 x i32> %347, <4 x i32> %348, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %352 = shufflevector <4 x i32> %349, <4 x i32> %350, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %353 = shufflevector <8 x i32> %351, <8 x i32> %352, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %354 = load <4 x i32>, ptr addrspace(3) %285, align 16
  %355 = load <4 x i32>, ptr addrspace(3) %287, align 16
  %356 = shufflevector <4 x i32> %354, <4 x i32> %355, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %357 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %323, i32 0, <16 x i32> %320, i16 0, <8 x float> %342, i32 0, i32 0, i32 %313, i32 0, i32 0, i32 %311, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %358 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %341, i32 0, <16 x i32> %338, i16 0, <8 x float> %357, i32 0, i32 0, i32 %331, i32 0, i32 0, i32 %329, i1 false, i1 false)
  %359 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %356, i32 0, <16 x i32> %353, i16 0, <8 x float> %358, i32 0, i32 0, i32 %346, i32 0, i32 0, i32 %344, i1 false, i1 false)
  %360 = add i64 %21, 32
  %361 = icmp sle i64 %360, %37
  br i1 %361, label %362, label %366

362:                                              ; preds = %14
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %363 = fptrunc <8 x float> %359 to <8 x bfloat>
  %364 = bitcast <8 x bfloat> %363 to <8 x half>
  %365 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_32x32x256_2x2_2buf_arena, i64 %54
  store <8 x half> %364, ptr addrspace(3) %365, align 2
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %78, <8 x i32> %92, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  br label %377

366:                                              ; preds = %14
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %367 = add i64 %21, %35
  %368 = add i64 %367, %34
  %369 = add i64 %22, %36
  %370 = add i64 %369, %53
  %371 = mul i64 %368, %39
  %372 = add i64 %371, %370
  %373 = mul i64 %372, 2
  %374 = fptrunc <8 x float> %359 to <8 x bfloat>
  %375 = bitcast <8 x bfloat> %374 to <4 x i32>
  %376 = trunc i64 %373 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %375, ptr addrspace(8) %43, i32 %376, i32 0, i32 0)
  br label %377

377:                                              ; preds = %362, %366
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setreg(i32 immarg, i32) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.wave.id() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.wait.dscnt(i16 immarg) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 immarg, <8 x i32>, i32 immarg, <16 x i32>, i16 immarg, <8 x float>, i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg, i32, i1 immarg, i1 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #9

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #6 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nofree nounwind willreturn }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 128, i32 1, i32 1}
