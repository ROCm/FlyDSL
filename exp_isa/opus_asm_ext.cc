#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstring>
#include <string>

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t err = (call);                                                  \
    TORCH_CHECK(err == hipSuccess, #call " failed: ", hipGetErrorString(err)); \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Must match exp_isa/flash_attn_opus.v1.s .amdgpu_metadata exactly.
struct __attribute__((packed)) KernelArgs {
  void *q;
  void *k;
  void *v;
  void *o;
  void *debug_counts;
  int seq_len;
  int stride_q_n;
  int stride_kv_n;
  int head_dim_runtime;
};

static_assert(sizeof(KernelArgs) == 56, "KernelArgs must match kernarg_segment_size");

namespace {

hipModule_t g_module = nullptr;
hipFunction_t g_kernel = nullptr;
std::string g_loaded_path;

void load_kernel_once(const std::string &code_object_path) {
  if (g_module != nullptr) {
    TORCH_CHECK(
        g_loaded_path == code_object_path,
        "flash_attn_opus assembly module already loaded from ",
        g_loaded_path,
        ", cannot reload from ",
        code_object_path);
    return;
  }

  HIP_CHECK(hipModuleLoad(&g_module, code_object_path.c_str()));
  g_loaded_path = code_object_path;
  HIP_CHECK(hipModuleGetFunction(&g_kernel, g_module, "flash_attn_opus_kernel_hand_asm"));
}

}  // namespace

void flash_attn_opus_forward_out(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    const std::string &code_object_path) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(out);

  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be torch.bfloat16");
  TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be torch.bfloat16");
  TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be torch.bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be torch.bfloat16");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, S, H, D]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [B, S, H_KV, D]");
  TORCH_CHECK(v.dim() == 4, "v must have shape [B, S, H_KV, D]");
  TORCH_CHECK(out.sizes() == q.sizes(), "out shape must match q shape");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == out.device(),
              "q, k, v, and out must be on the same device");

  const int64_t batch = q.size(0);
  const int64_t seq_len = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t num_kv_heads = k.size(2);
  const int64_t head_dim = q.size(3);

  TORCH_CHECK(k.size(0) == batch && v.size(0) == batch, "batch mismatch");
  TORCH_CHECK(k.size(1) == seq_len && v.size(1) == seq_len, "sequence length mismatch");
  TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim, "head_dim mismatch");
  TORCH_CHECK(head_dim == 128, "flash_attn_opus.v1.s supports head_dim=128 only");
  TORCH_CHECK(num_heads == 64, "flash_attn_opus.v1.s was compiled for num_heads=64");
  TORCH_CHECK(num_kv_heads == 64, "flash_attn_opus.v1.s was compiled for num_kv_heads=64");
  TORCH_CHECK(seq_len % 256 == 0, "seq_len must be divisible by 256");
  TORCH_CHECK(batch > 0, "batch must be positive");

  const int device_index = q.get_device();
  const c10::cuda::OptionalCUDAGuard device_guard(device_index);
  HIP_CHECK(hipSetDevice(device_index));
  load_kernel_once(code_object_path);

  KernelArgs args;
  std::memset(&args, 0, sizeof(args));
  args.q = q.data_ptr();
  args.k = k.data_ptr();
  args.v = v.data_ptr();
  args.o = out.data_ptr();
  args.debug_counts = out.data_ptr();
  args.seq_len = static_cast<int>(seq_len);
  args.stride_q_n = static_cast<int>(num_heads * head_dim);
  args.stride_kv_n = static_cast<int>(num_kv_heads * head_dim);
  args.head_dim_runtime = static_cast<int>(head_dim);

  size_t arg_size = sizeof(args);
  void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &arg_size,
      HIP_LAUNCH_PARAM_END};

  const int grid_x = static_cast<int>(num_heads);
  const int grid_y = static_cast<int>((seq_len + 255) / 256);
  const int grid_z = static_cast<int>(batch);
  hipStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();

  HIP_CHECK(hipModuleLaunchKernel(
      g_kernel,
      grid_x,
      grid_y,
      grid_z,
      512,
      1,
      1,
      0,
      stream,
      nullptr,
      reinterpret_cast<void **>(&config)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward_out",
      &flash_attn_opus_forward_out,
      "Launch exp_isa/flash_attn_opus.v1.s into an existing output tensor");
}
