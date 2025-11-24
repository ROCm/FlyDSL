//===- cute_runtime.cpp - CuTe Runtime Library Implementation ---*- C++ -*-===//

#include "../include/cute_runtime.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <array>

#if defined(HAVE_HIP) || defined(ENABLE_ROCM_SUPPORT)
#include <hip/hip_runtime.h>
#endif

namespace cute {
namespace runtime {

//===----------------------------------------------------------------------===//
// TMA Descriptor Implementation
//===----------------------------------------------------------------------===//

void TMADescriptor::initialize_2d(
    void* global_ptr,
#if defined(HAVE_HIP) || defined(ENABLE_ROCM_SUPPORT)
    int dtype,
#else
    cudaDataType dtype,
#endif
    uint32_t global_dim_x,
    uint32_t global_dim_y,
    uint32_t tile_dim_x,
    uint32_t tile_dim_y,
    SwizzleMode swizzle
) {
#if !defined(HAVE_HIP) && !defined(ENABLE_ROCM_SUPPORT)
    // Element size mapping
    uint32_t elem_size;
    switch (dtype) {
        case CUDA_R_16F: elem_size = 2; break;
        case CUDA_R_16BF: elem_size = 2; break;
        case CUDA_R_32F: elem_size = 4; break;
        case CUDA_R_64F: elem_size = 8; break;
        default: 
            throw CuteRuntimeError("Unsupported data type for TMA");
    }
    
    // Swizzle size mapping
    cuTensorMapSwizzle swizzle_cu;
    switch (swizzle) {
        case SwizzleMode::None: 
            swizzle_cu = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case SwizzleMode::Swizzle32B: 
            swizzle_cu = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case SwizzleMode::Swizzle64B: 
            swizzle_cu = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case SwizzleMode::Swizzle128B: 
            swizzle_cu = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }
    
    // Create TMA descriptor
    uint64_t global_dims[2] = {global_dim_x, global_dim_y};
    uint64_t global_strides[1] = {global_dim_x * elem_size};
    uint32_t box_dims[2] = {tile_dim_x, tile_dim_y};
    uint32_t elem_strides[2] = {1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(
        desc_,
        dtype,
        2,  // rank
        global_ptr,
        global_dims,
        global_strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_cu,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
#else
    // TMA not supported on HIP/AMD yet
    (void)global_ptr; (void)dtype; (void)global_dim_x; (void)global_dim_y;
    (void)tile_dim_x; (void)tile_dim_y; (void)swizzle;
    throw CuteRuntimeError("TMA not supported on AMD GPUs");
#endif
}

//===----------------------------------------------------------------------===//
// Kernel Executor Implementation
//===----------------------------------------------------------------------===//

KernelExecutor::KernelExecutor() 
    : module_(nullptr), kernel_(nullptr),
      module_loaded_(false), kernel_set_(false) {
    CU_CHECK(cuInit(0));
}

KernelExecutor::~KernelExecutor() {
    if (module_) cuModuleUnload(module_);
}

void KernelExecutor::load_cubin(const std::string& cubin_path) {
    std::ifstream file(cubin_path, std::ios::binary);
    if (!file) {
        throw CuteRuntimeError("Failed to open cubin file: " + cubin_path);
    }
    
    std::vector<char> cubin_data(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    
    CU_CHECK(cuModuleLoadData(&module_, cubin_data.data()));
    module_loaded_ = true;
}

void KernelExecutor::load_ptx(const std::string& ptx_path) {
    std::ifstream file(ptx_path);
    if (!file) {
        throw CuteRuntimeError("Failed to open PTX file: " + ptx_path);
    }
    
    std::string ptx_code(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    
    CU_CHECK(cuModuleLoadData(&module_, ptx_code.c_str()));
    module_loaded_ = true;
}

void KernelExecutor::set_kernel(const std::string& kernel_name) {
    if (!module_loaded_) {
        throw CuteRuntimeError("Module not loaded");
    }
    
    CU_CHECK(cuModuleGetFunction(&kernel_, module_, kernel_name.c_str()));
    kernel_set_ = true;
}

void KernelExecutor::launch(
    const std::vector<void*>& args, 
    const LaunchConfig& config
) {
    if (!kernel_set_) {
        throw CuteRuntimeError("Kernel not set");
    }
    
    CU_CHECK(cuLaunchKernel(
        kernel_,
        config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
        config.block_dim.x, config.block_dim.y, config.block_dim.z,
        config.shared_mem_bytes,
        config.stream,
        const_cast<void**>(args.data()),
        nullptr
    ));
}

void KernelExecutor::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

cudaDeviceProp KernelExecutor::get_device_properties(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop;
}

//===----------------------------------------------------------------------===//
// Compiler Implementation
//===----------------------------------------------------------------------===//

CuteCompiler::CuteCompiler() : mlir_bin_path_("/usr/local/bin") {
    // Try to auto-detect MLIR installation
    const char* llvm_install = std::getenv("LLVM_INSTALL_DIR");
    if (llvm_install) {
        mlir_bin_path_ = std::string(llvm_install) + "/bin";
    }
}

void CuteCompiler::set_mlir_bin_path(const std::string& path) {
    mlir_bin_path_ = path;
}

std::string CuteCompiler::run_command(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw CuteRuntimeError("Failed to execute command: " + cmd);
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    
    int ret = pclose(pipe);
    if (ret != 0) {
        throw CuteRuntimeError("Command failed with code " + 
                               std::to_string(ret) + ": " + cmd);
    }
    
    return result;
}

std::string CuteCompiler::compile_to_ptx(
    const std::string& mlir_code,
    Arch arch,
    int opt_level
) {
#if defined(HAVE_HIP) || defined(ENABLE_ROCM_SUPPORT)
    throw CuteRuntimeError("Compilation not supported on AMD yet");
#else
    // Write MLIR to temp file
    std::string temp_mlir = "/tmp/cute_kernel.mlir";
    std::ofstream mlir_file(temp_mlir);
    mlir_file << mlir_code;
    mlir_file.close();
    
    // Output PTX file
    std::string temp_ptx = "/tmp/cute_kernel.ptx";
    
    // Build mlir-opt pass pipeline
    std::string pass_pipeline = 
        "--pass-pipeline="
        "'builtin.module("
        "  cute-canonicalize,"
        "  cute-layout-analysis,"
        "  cute-to-standard,"
        "  cute-nvgpu-to-nvgpu,"
        "  convert-nvgpu-to-nvvm,"
        "  gpu-kernel-outlining,"
        "  convert-gpu-to-nvvm,"
        "  gpu-to-llvm,"
        "  reconcile-unrealized-casts"
        ")'";
    
    // MLIR → LLVM IR
    std::string mlir_opt_cmd = 
        mlir_bin_path_ + "/mlir-opt " + temp_mlir + " " +
        pass_pipeline + " -o /tmp/cute_kernel.llvm.mlir";
    run_command(mlir_opt_cmd);
    
    // LLVM IR → PTX
    std::string sm_version = "sm_" + std::to_string(static_cast<int>(arch));
    std::string translate_cmd = 
        mlir_bin_path_ + "/mlir-translate "
        "--mlir-to-nvvmir /tmp/cute_kernel.llvm.mlir | "
        "llc -march=nvptx64 -mcpu=" + sm_version + 
        " -O" + std::to_string(opt_level) + " -o " + temp_ptx;
    run_command(translate_cmd);
    
    // Read PTX
    std::ifstream ptx_file(temp_ptx);
    std::string ptx_code(
        (std::istreambuf_iterator<char>(ptx_file)),
        std::istreambuf_iterator<char>()
    );
    
    return ptx_code;
#endif
}

std::string CuteCompiler::compile_to_cubin(
    const std::string& ptx_code,
    Arch arch
) {
#if defined(HAVE_HIP) || defined(ENABLE_ROCM_SUPPORT)
    throw CuteRuntimeError("Compilation not supported on AMD yet");
#else
    // Write PTX to temp file
    std::string temp_ptx = "/tmp/cute_kernel.ptx";
    std::ofstream ptx_file(temp_ptx);
    ptx_file << ptx_code;
    ptx_file.close();
    
    // Output CUBIN file
    std::string temp_cubin = "/tmp/cute_kernel.cubin";
    
    // PTX → CUBIN using ptxas
    std::string sm_version = std::to_string(static_cast<int>(arch));
    std::string ptxas_cmd = 
        "ptxas -arch=sm_" + sm_version + " " +
        temp_ptx + " -o " + temp_cubin;
    run_command(ptxas_cmd);
    
    return temp_cubin;
#endif
}

std::string CuteCompiler::compile(
    const std::string& mlir_code,
    Arch arch,
    int opt_level
) {
    std::string ptx = compile_to_ptx(mlir_code, arch, opt_level);
    return compile_to_cubin(ptx, arch);
}

//===----------------------------------------------------------------------===//
// GEMM Executor Implementation (Template Specialization Required)
//===----------------------------------------------------------------------===//

template<typename TA, typename TB, typename TC>
GemmExecutor<TA, TB, TC>::GemmExecutor(
    size_t M, size_t N, size_t K,
    Arch arch,
    bool use_tma
) : M_(M), N_(N), K_(K), arch_(arch), use_tma_(use_tma) {
    
    executor_ = std::make_unique<KernelExecutor>();
    
    // Allocate device buffers
    d_A_ = std::make_unique<DeviceBuffer<TA>>(M * K);
    d_B_ = std::make_unique<DeviceBuffer<TB>>(K * N);
    d_C_ = std::make_unique<DeviceBuffer<TC>>(M * N);
    
    // Initialize TMA descriptors for SM90+
    if (use_tma && static_cast<int>(arch) >= 90 && static_cast<int>(arch) < 900) {
        tma_desc_A_ = std::make_unique<TMADescriptor>();
        tma_desc_B_ = std::make_unique<TMADescriptor>();
        
#if !defined(HAVE_HIP) && !defined(ENABLE_ROCM_SUPPORT)
        cudaDataType dtype_a = sizeof(TA) == 2 ? CUDA_R_16F : CUDA_R_32F;
        cudaDataType dtype_b = sizeof(TB) == 2 ? CUDA_R_16F : CUDA_R_32F;
        
        // Example tile size (should be configurable)
        uint32_t tile_m = 128, tile_k = 64, tile_n = 128;
        
        tma_desc_A_->initialize_2d(
            d_A_->ptr(), dtype_a,
            static_cast<uint32_t>(K), static_cast<uint32_t>(M),
            tile_k, tile_m
        );
        
        tma_desc_B_->initialize_2d(
            d_B_->ptr(), dtype_b,
            static_cast<uint32_t>(N), static_cast<uint32_t>(K),
            tile_n, tile_k
        );
#endif
    }
}

template<typename TA, typename TB, typename TC>
void GemmExecutor<TA, TB, TC>::execute(
    const TA* A, const TB* B, TC* C,
    bool is_device_ptr
) {
    // Copy data to device if needed
    if (!is_device_ptr) {
        d_A_->copy_from_host(A, M_ * K_);
        d_B_->copy_from_host(B, K_ * N_);
    }
    
    // Prepare kernel arguments
    std::vector<void*> args;
    void* ptr_a = d_A_->ptr();
    void* ptr_b = d_B_->ptr();
    void* ptr_c = d_C_->ptr();
    args.push_back(&ptr_a);
    args.push_back(&ptr_b);
    args.push_back(&ptr_c);
    
    if (use_tma_) {
        void* ptr_tma_a = tma_desc_A_->get();
        void* ptr_tma_b = tma_desc_B_->get();
        args.push_back(&ptr_tma_a);
        args.push_back(&ptr_tma_b);
    }
    
    // Compute grid/block dimensions
    auto [tile_m, tile_n, tile_k] = get_optimal_tile_size(M_, N_, K_, arch_);
    
    dim3 grid(
        (N_ + tile_n - 1) / tile_n,
        (M_ + tile_m - 1) / tile_m
    );
    dim3 block(128, 1, 1);  // 4 warps
    
    size_t smem_bytes = static_cast<int>(arch_) >= 90 ? 
        (tile_m * tile_k + tile_k * tile_n) * sizeof(TA) : 0;
    
    LaunchConfig config(grid, block, smem_bytes);
    
    // Launch kernel
    executor_->launch(args, config);
    executor_->synchronize();
    
    // Copy result back
    if (!is_device_ptr) {
        d_C_->copy_to_host(C, M_ * N_);
    }
}

template<typename TA, typename TB, typename TC>
std::tuple<size_t, size_t, size_t> 
GemmExecutor<TA, TB, TC>::get_optimal_tile_size(
    size_t M, size_t N, size_t K, Arch arch
) {
    switch (arch) {
        case Arch::SM80:  // Ampere
            return {128, 128, 32};
        case Arch::SM90:  // Hopper
            return {128, 128, 64};
        case Arch::SM100: // Blackwell
            return {256, 128, 64};
        case Arch::GFX942: // MI300
            return {256, 256, 32};
        default:
            return {64, 64, 16};
    }
}

// Explicit instantiations for common types
// Use float for half if half is not available or use hip_fp16.h
#if defined(HAVE_HIP) || defined(ENABLE_ROCM_SUPPORT)
// For now just instantiate float
template class GemmExecutor<float, float, float>;
#else
template class GemmExecutor<half, half, float>;
template class GemmExecutor<half, half, half>;
template class GemmExecutor<float, float, float>;
#endif

} // namespace runtime
} // namespace cute
