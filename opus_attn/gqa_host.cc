// Host-only: benchmark harness, CPU reference, main()
#include <opus/hip_minimal.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <numeric>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>

#include "gqa_defs.h"

// Declared in gqa_d128_kernel_*.cc / gqa_d512_kernel_*.cc (device TUs)
template<class Traits>
__global__ void gqa_d128_kernel(opus_gqa_kargs kargs);
template<class Traits>
__global__ void gqa_d512_kernel(opus_gqa_kargs kargs);

// Common launch wrapper that dispatches by D_TILE_SIZE at compile time
template<class Traits>
inline void gqa_launch(const opus_gqa_kargs& kargs, dim3 grid, dim3 block) {
    if constexpr (Traits::D_TILE_SIZE == 128) {
        gqa_d128_kernel<Traits><<<grid, block>>>(kargs);
    } else {
        gqa_d512_kernel<Traits><<<grid, block>>>(kargs);
    }
}

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

// Fill a contiguous vector with random values
template<typename T>
void rand_vector(T* ptr, size_t size, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

// Deterministic bf16-friendly inputs for cross-checking the exe and Python wrapper.
template<typename T>
void deterministic_vector(T* ptr, size_t size, int stream_id) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        const int bucket = static_cast<int>((i * 13 + stream_id * 17) % 33);
        ptr[i] = static_cast<T>((bucket - 16) / 16.0f);
    }
}

std::string join_path(const std::string& dir, const char* file_name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + file_name;
    }
    return dir + "/" + file_name;
}

bool read_binary_file(const std::string& path, void* dst, size_t byte_count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open input file: " << path << "\n";
        return false;
    }

    const std::streamoff file_size = file.tellg();
    if (file_size != static_cast<std::streamoff>(byte_count)) {
        std::cerr << "Input file has unexpected size: " << path
                  << " expected " << byte_count << " bytes, got " << file_size << "\n";
        return false;
    }

    file.seekg(0, std::ios::beg);
    file.read(static_cast<char*>(dst), static_cast<std::streamsize>(byte_count));
    if (!file) {
        std::cerr << "Failed to read input file: " << path << "\n";
        return false;
    }
    return true;
}

bool write_binary_file(const std::string& path, const void* src, size_t byte_count) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output file: " << path << "\n";
        return false;
    }

    file.write(static_cast<const char*>(src), static_cast<std::streamsize>(byte_count));
    if (!file) {
        std::cerr << "Failed to write output file: " << path << "\n";
        return false;
    }
    return true;
}

// Benchmark GQA kernel performance with warm-up and timing
template<class Traits>
void benchmark_gqa_kernel(const opus_gqa_kargs& kargs, dim3 grid, dim3 block,
                          int warmup = 100, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        gqa_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gqa_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;
    //   full attention  -> 4 * B * H * N^2 * D
    //   causal attention -> half of the full-attention work
    const double flops = (4.0 * kargs.B * kargs.H * kargs.N * kargs.N * kargs.D)
                       / (Traits::CAUSAL ? 2.0 : 1.0);
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("GQA %s Kernel Performance: avg_time=%.3f ms, %.2f TFlops\n",
           Traits::CAUSAL ? "Causal" : "Non-causal", avg_time, tflops);
}

// Validate GQA GPU results against CPU reference
bool validate_gqa_results(const bf16_t* ref, const bf16_t* gpu,
                          int B, int N, int H, int D, float threshold = 5e-2f) {
    bool all_valid = true;
    size_t total_errors = 0;
    const size_t total_elements = (size_t)B * N * H * D;

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            for (int h = 0; h < H; h++) {
                const size_t offset = ((size_t)b * N * H + (size_t)i * H + h) * D;
                for (int d = 0; d < D; d++) {
                    const float ref_val = static_cast<float>(ref[offset + d]);
                    const float gpu_val = static_cast<float>(gpu[offset + d]);
                    const float diff = std::abs(gpu_val - ref_val);
                    if (diff > threshold) {
                        total_errors++;
                        all_valid = false;
                        printf("  mismatch [b=%d,n=%d,h=%d,d=%d] ref=%.6f gpu=%.6f diff=%.6f\n",
                               b, i, h, d, ref_val, gpu_val, diff);
                    }
                }
            }
        }
    }
    
    if (all_valid) {
        printf("✓ Full validation passed (checked %zu elements)\n", total_elements);
    } else {
        printf("✗ Validation failed with %zu/%zu total errors\n",
               total_errors, total_elements);
    }
    
    return all_valid;
}

// ─── CPU reference: Grouped-Query Attention (GQA) ──────────────────────────
//
// Q  layout: [B, N, H,    D]   (row-major, contiguous in D)
// K  layout: [B, N, H_KV, D]
// V  layout: [B, N, H_KV, D]
// O  layout: [B, N, H,    D]
//
// Standard scaled-dot-product attention with online softmax:
//   S[i,j]  = sum_d Q[b,i,h,d] * K[b,j,h_kv,d]   (h_kv = h / group_size)
//   P[i,:]  = softmax( S[i,:] / sqrt(D) )
//   O[i,d]  = sum_j P[i,j] * V[b,j,h_kv,d]
//
void gqa_attention_ref(
    const bf16_t* Q,  // [B, N, H, D]
    const bf16_t* K,  // [B, N, H_KV, D]
    const bf16_t* V,  // [B, N, H_KV, D]
    bf16_t*       O,  // [B, N, H, D]
    int B, int N, int H, int H_KV, int D, bool causal = false)
{
    const int GROUP_SIZE = H / H_KV;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Strides (row-major, last dim = D is contiguous)
    const int stride_q_b = N * H * D;
    const int stride_q_n = H * D;
    const int stride_q_h = D;

    const int stride_kv_b = N * H_KV * D;
    const int stride_kv_n = H_KV * D;
    const int stride_kv_h = D;

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {
                const int h_kv = h / GROUP_SIZE;
                const bf16_t* q_row = Q + b * stride_q_b + i * stride_q_n + h * stride_q_h;

                // ---- Compute attention scores S[j] = Q[b,i,h,:] . K[b,j,h_kv,:] ----
                const int max_j = causal ? (i + 1) : N;
                std::vector<float> scores(max_j);
                for (int j = 0; j < max_j; j++) {
                    const bf16_t* k_row = K + b * stride_kv_b + j * stride_kv_n + h_kv * stride_kv_h;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        dot += static_cast<float>(q_row[d] * k_row[d]);
                    }
                    scores[j] = dot * scale;
                }

                // ---- Softmax ----
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int j = 0; j < max_j; j++) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < max_j; j++) {
                    scores[j] /= sum_exp;
                }
                std::vector<bf16_t> p_row(max_j);
                for (int j = 0; j < max_j; j++) {
                    p_row[j] = static_cast<bf16_t>(scores[j]);
                }

                // ---- Output: O[b,i,h,d] = sum_j P[j] * V[b,j,h_kv,d] ----
                bf16_t* o_row = O + b * stride_q_b + i * stride_q_n + h * stride_q_h;
                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < max_j; j++) {
                        const bf16_t* v_row = V + b * stride_kv_b + j * stride_kv_n + h_kv * stride_kv_h;
                        acc += static_cast<float>(p_row[j] * v_row[d]);
                    }
                    o_row[d] = static_cast<bf16_t>(acc);
                }
            }
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int B    = 16;    // batch size
    int H    = 64;    // query heads
    int H_KV = 8;     // key/value heads
    int N    = 1024;  // sequence length
    int D    = 128;   // head dimension

    // Parse command line arguments. Supports: -n 16384 and -n=16384.
    bool causal = true;
    bool verify = false;
    bool deterministic_inputs = false;
    bool skip_benchmark = false;
    std::string input_dir;
    std::string output_file;
    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') return arg + len + 1;       // -flag=value
            if (arg[len] == '\0') return reinterpret_cast<const char*>(1); // -flag value (next arg)
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val;
        if (std::strcmp(arg, "--causal") == 0) { causal = true; continue; }
        if (std::strcmp(arg, "--no-causal") == 0) { causal = false; continue; }
        if (std::strcmp(arg, "--verify") == 0) { verify = true; continue; }
        if (std::strcmp(arg, "--deterministic-inputs") == 0) { deterministic_inputs = true; continue; }
        if (std::strcmp(arg, "--skip-benchmark") == 0) { skip_benchmark = true; continue; }
        if ((val = parse_val(arg, "--input-dir"))) {
            if (val == reinterpret_cast<const char*>(1)) {
                if (i + 1 < argc) input_dir = argv[++i];
            } else {
                input_dir = val;
            }
            continue;
        }
        if ((val = parse_val(arg, "--output-file"))) {
            if (val == reinterpret_cast<const char*>(1)) {
                if (i + 1 < argc) output_file = argv[++i];
            } else {
                output_file = val;
            }
            continue;
        }
        auto try_parse = [&](int& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        if (try_parse(B, "-b")) continue;
        if (try_parse(H, "-h_q")) continue;
        if (try_parse(H_KV, "-h_kv")) continue;
        if (try_parse(N, "-n")) continue;
        if (try_parse(D, "-d")) continue;
    }

    if (B <= 0 || H <= 0 || H_KV <= 0 || N <= 0 || D <= 0 || H % H_KV != 0) {
        std::cerr << "Invalid parameters. B,H_Q,H_KV,N,D must be positive and H_Q must be divisible by H_KV.\n";
        return 1;
    }

    const int GROUP_SIZE = H / H_KV;
    printf("GQA Attention: B=%d, H_Q=%d, H_KV=%d, GROUP_SIZE=%d, N=%d, D=%d, CAUSAL=%d\n",
           B, H, H_KV, GROUP_SIZE, N, D, causal ? 1 : 0);

    // Allocate host memory
    const size_t q_size = (size_t)B * N * H * D;
    const size_t kv_size = (size_t)B * N * H_KV * D;
    auto host_q = std::make_unique<bf16_t[]>(q_size);
    auto host_k = std::make_unique<bf16_t[]>(kv_size);
    auto host_v = std::make_unique<bf16_t[]>(kv_size);
    auto host_o_ref = std::make_unique<bf16_t[]>(q_size);
    auto host_o_gpu = std::make_unique<bf16_t[]>(q_size);

    // Initialize with random data
    if (!input_dir.empty()) {
        const bool loaded = read_binary_file(join_path(input_dir, "q.bin"), host_q.get(), q_size * sizeof(bf16_t)) &&
                            read_binary_file(join_path(input_dir, "k.bin"), host_k.get(), kv_size * sizeof(bf16_t)) &&
                            read_binary_file(join_path(input_dir, "v.bin"), host_v.get(), kv_size * sizeof(bf16_t));
        if (!loaded) return 1;
    } else if (deterministic_inputs) {
        deterministic_vector(host_q.get(), q_size, 0);
        deterministic_vector(host_k.get(), kv_size, 1);
        deterministic_vector(host_v.get(), kv_size, 2);
    } else {
        rand_vector(host_q.get(), q_size, -2.f, 2.f);
        rand_vector(host_k.get(), kv_size, -2.f, 2.f);
        rand_vector(host_v.get(), kv_size, -2.f, 2.f);
    }

    // Allocate device memory
    bf16_t *dev_q, *dev_k, *dev_v, *dev_o;
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_k, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_v, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_o, q_size * sizeof(bf16_t)));

    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_k, host_k.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_v, host_v.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Setup kernel arguments
    opus_gqa_kargs kargs{};
    kargs.ptr_q = dev_q;
    kargs.ptr_k = dev_k;
    kargs.ptr_v = dev_v;
    kargs.ptr_o = dev_o;
    kargs.B = B;
    kargs.N = N;
    kargs.H = H;
    kargs.H_KV = H_KV;
    kargs.D = D;
    kargs.stride_q_b = N * H * D;
    kargs.stride_q_n = H * D;
    kargs.stride_q_h = D;
    kargs.stride_kv_b = N * H_KV * D;
    kargs.stride_kv_n = H_KV * D;
    kargs.stride_kv_h = D;

    // Dispatch to causal or non-causal kernel
    auto run = [&]<typename GqaTraits>(GqaTraits) {
        if (D != GqaTraits::D_TILE_SIZE) {
            std::cerr << "This kernel only supports head dimension D=" << GqaTraits::D_TILE_SIZE << ", got D=" << D << "\n";
            return 1;
        }
        if ((N % GqaTraits::KV_TILE_SIZE) != 0 || (N / GqaTraits::KV_TILE_SIZE) < 6) {
            std::cerr << "This attend-style pipeline requires N to be a multiple of "
                      << GqaTraits::KV_TILE_SIZE << " and span at least 6 KV tiles, got N=" << N << "\n";
            return 1;
        }
        if ((N % (GqaTraits::Q_TILE_SIZE * GqaTraits::NUM_WARPS)) != 0) {
            std::cerr << "This kernel requires N to be a multiple of "
                      << (GqaTraits::Q_TILE_SIZE * GqaTraits::NUM_WARPS)
                      << " so every warp maps to a valid Q tile, got N=" << N << "\n";
            return 1;
        }
        const int num_q_tiles = ceil_div(N, GqaTraits::Q_TILE_SIZE);
        const int num_q_blocks = ceil_div(num_q_tiles, GqaTraits::NUM_WARPS);
        dim3 grid(H, num_q_blocks, B);
        dim3 block(GqaTraits::BLOCK_SIZE);

        printf("GQA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
               grid.x, grid.y, grid.z, (int)block.x, GqaTraits::NUM_WARPS, GqaTraits::smem_size_bytes());

        CHECK_HIP(hipMemset(dev_o, 0, q_size * sizeof(bf16_t)));
        gqa_launch<GqaTraits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();

        CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, q_size * sizeof(bf16_t), hipMemcpyDeviceToHost));
        if (!output_file.empty()) {
            if (!write_binary_file(output_file, host_o_gpu.get(), q_size * sizeof(bf16_t))) {
                return 1;
            }
            printf("  wrote_result = %s\n", output_file.c_str());
        }

        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            gqa_attention_ref(host_q.get(), host_k.get(), host_v.get(), host_o_ref.get(),
                              B, N, H, H_KV, D, GqaTraits::CAUSAL);

            bool all_valid = validate_gqa_results(host_o_ref.get(), host_o_gpu.get(), B, N, H, D);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) return 1;
        }

        if (!skip_benchmark) {
            printf("\n");
            benchmark_gqa_kernel<GqaTraits>(kargs, grid, block);
            printf("\n");
        }
        return 0;
    };

    int rc;
    if (D == 128) {
        rc = causal ? run(opus_gqa_traits<32, 64, 128, 8, true>{})
                    : run(opus_gqa_traits<32, 64, 128, 8, false>{});
    } else if (D == 512) {
        rc = causal ? run(opus_gqa_traits<16, 32, 512, 8, true>{})
                    : run(opus_gqa_traits<16, 32, 512, 8, false>{});
    } else {
        std::cerr << "-d must be 128 or 512, got " << D << "\n";
        return 1;
    }
    if (rc) return rc;

    // Cleanup
    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_k));
    CHECK_HIP(hipFree(dev_v));
    CHECK_HIP(hipFree(dev_o));

    return 0;
}
