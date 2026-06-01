#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>

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

struct p3 {
  unsigned int _p0;
  unsigned int _p1;
  unsigned int _p2;
};

struct p2 {
  unsigned int _p0;
  unsigned int _p1;
};

// Must match fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk{0,1}_gm0.s metadata.
struct __attribute__((packed)) KernelArgs {
  void *ptr_R;
  p2 _p0;
  void *ptr_Q;
  p2 _p1;
  void *ptr_K;
  p2 _p2;
  void *ptr_V;
  p2 _p3;
  void *ptr_LSE;
  p2 _p4;
  float scalar;
  p3 _p5;
  unsigned int s_seq_len;
  p3 _p6;
  unsigned int s_Seqs;
  p3 _p7;
  unsigned int s_Ts;
  p3 _p8;
  unsigned int s_Hs;
  p3 _p9;
  unsigned int s_Bs;
  p3 _p10;
  unsigned int s_gqa;
  p3 _p11;
  unsigned int s_k_Seqs;
  p3 _p12;
  unsigned int s_k_Hs;
  p3 _p13;
  unsigned int s_k_Bs;
  p3 _p14;
  unsigned int s_opt;
  p3 _p15;
  unsigned int s_lse;
  p3 _p16;
  unsigned int s_kv_seq_len;
  p3 _p17;
  unsigned int s_qk_head_dim;
  p3 _p18;
  unsigned int s_v_head_dim;
  p3 _p19;
  unsigned int s_q_head_num;
  p3 _p20;
  unsigned int s_v_Seqs;
  p3 _p21;
  unsigned int s_v_Hs;
  p3 _p22;
  unsigned int s_v_Bs;
  p3 _p23;
  unsigned int s_o_Seqs;
  p3 _p24;
  unsigned int s_o_Hs;
  p3 _p25;
  unsigned int s_o_Bs;
  p3 _p26;
  void *ptr_qseq;
  p2 _p27;
  void *ptr_kseq;
  p2 _p28;
  unsigned int s_LSE_Hs;
  p3 _p29;
  void *ptr_qseq_padding;
  p2 _p30;
  void *ptr_kseq_padding;
  p2 _p31;
};

static_assert(sizeof(KernelArgs) == 512, "KernelArgs must match kernarg_segment_size");

namespace {

struct ModuleState {
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
};

std::unordered_map<std::string, ModuleState> g_modules;

hipFunction_t load_kernel(const std::string &code_object_path) {
  auto &state = g_modules[code_object_path];
  if (state.module == nullptr) {
    HIP_CHECK(hipModuleLoad(&state.module, code_object_path.c_str()));
    HIP_CHECK(hipModuleGetFunction(&state.kernel, state.module, "fwd_kernel_func"));
  }
  return state.kernel;
}

}  // namespace

void fmha_asm_forward_out(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor lse,
    bool causal,
    const std::string &code_object_path) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(out);
  CHECK_INPUT(lse);

  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be torch.bfloat16");
  TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be torch.bfloat16");
  TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be torch.bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be torch.bfloat16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be torch.float32");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, S, H, D]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [B, S, H_KV, D]");
  TORCH_CHECK(v.dim() == 4, "v must have shape [B, S, H_KV, D]");
  TORCH_CHECK(out.sizes() == q.sizes(), "out shape must match q shape");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == out.device() && q.device() == lse.device(),
              "q, k, v, out, and lse must be on the same device");

  const int64_t batch = q.size(0);
  const int64_t seq_len = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t num_kv_heads = k.size(2);
  const int64_t head_dim = q.size(3);

  TORCH_CHECK(k.size(0) == batch && v.size(0) == batch, "batch mismatch");
  TORCH_CHECK(k.size(1) == seq_len && v.size(1) == seq_len, "sequence length mismatch");
  TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim, "head_dim mismatch");
  TORCH_CHECK(lse.dim() == 3, "lse must have shape [B, H, S]");
  TORCH_CHECK(lse.size(0) == batch && lse.size(1) == num_heads && lse.size(2) == seq_len,
              "lse shape must be [B, H, S]");

  TORCH_CHECK(head_dim == 128, "MI350 fmha asm wrapper supports head_dim=128 only");
  TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");
  TORCH_CHECK(!causal || num_heads % 8 == 0,
              "causal MI350 fmha asm wrapper requires num_heads to be a multiple of 8");
  TORCH_CHECK(seq_len % 256 == 0, "seq_len must be divisible by 256");
  TORCH_CHECK(batch > 0, "batch must be positive");

  const int device_index = q.get_device();
  const c10::cuda::OptionalCUDAGuard device_guard(device_index);
  HIP_CHECK(hipSetDevice(device_index));
  hipFunction_t kernel = load_kernel(code_object_path);

  constexpr int kElemBytes = 2;
  constexpr int kSubQ = 256;
  constexpr int kWavesPerThreadgroup = 8;
  const int b = static_cast<int>(batch);
  const int s = static_cast<int>(seq_len);
  const int h = static_cast<int>(num_heads);
  const int h_kv = static_cast<int>(num_kv_heads);
  const int d = static_cast<int>(head_dim);
  const int gqa = h / h_kv;

  const int stride_q_head = d * kElemBytes;
  const int stride_q_seq = h * stride_q_head;
  const int stride_q_tg = kSubQ * stride_q_seq;
  const int stride_q_batch = s * stride_q_seq;

  const int stride_k_head = d * kElemBytes;
  const int stride_k_seq = h_kv * stride_k_head;
  const int stride_k_batch = s * stride_k_seq;

  const int stride_v_head = d * kElemBytes;
  const int stride_v_seq = h_kv * stride_v_head;
  const int stride_v_batch = s * stride_v_seq;

  const int stride_o_head = d * kElemBytes;
  const int stride_o_seq = h * stride_o_head;
  const int stride_o_batch = s * stride_o_seq;
  const int stride_lse_head = s * 4;

  KernelArgs args;
  std::memset(&args, 0, sizeof(args));
  args.ptr_R = out.data_ptr();
  args.ptr_Q = q.data_ptr();
  args.ptr_K = k.data_ptr();
  args.ptr_V = v.data_ptr();
  args.ptr_LSE = lse.data_ptr();
  args.scalar = 1.0f / std::sqrt(static_cast<float>(d));
  args.s_seq_len = static_cast<unsigned int>(s);
  args.s_Seqs = static_cast<unsigned int>(stride_q_seq);
  args.s_Ts = static_cast<unsigned int>(stride_q_tg);
  args.s_Hs = static_cast<unsigned int>(stride_q_head);
  args.s_Bs = static_cast<unsigned int>(stride_q_batch);
  args.s_gqa = static_cast<unsigned int>(gqa);
  args.s_k_Seqs = static_cast<unsigned int>(stride_k_seq);
  args.s_k_Hs = static_cast<unsigned int>(stride_k_head);
  args.s_k_Bs = static_cast<unsigned int>(stride_k_batch);
  args.s_opt = 5;
  args.s_lse = 1;
  args.s_kv_seq_len = static_cast<unsigned int>(s);
  args.s_qk_head_dim = static_cast<unsigned int>(d);
  args.s_v_head_dim = static_cast<unsigned int>(d);
  args.s_q_head_num = static_cast<unsigned int>(h);
  args.s_v_Seqs = static_cast<unsigned int>(stride_v_seq);
  args.s_v_Hs = static_cast<unsigned int>(stride_v_head);
  args.s_v_Bs = static_cast<unsigned int>(stride_v_batch);
  args.s_o_Seqs = static_cast<unsigned int>(stride_o_seq);
  args.s_o_Hs = static_cast<unsigned int>(stride_o_head);
  args.s_o_Bs = static_cast<unsigned int>(stride_o_batch);
  args.ptr_qseq = nullptr;
  args.ptr_kseq = nullptr;
  args.s_LSE_Hs = static_cast<unsigned int>(stride_lse_head);
  args.ptr_qseq_padding = nullptr;
  args.ptr_kseq_padding = nullptr;

  size_t arg_size = sizeof(args);
  void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &arg_size,
      HIP_LAUNCH_PARAM_END};

  const int tg_div = causal ? 2 : 1;
  const int q_blocks = (s + kSubQ - 1) / kSubQ;
  const int grid_x = (q_blocks + tg_div - 1) / tg_div;
  const int grid_y = h;
  const int grid_z = b;
  hipStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();

  HIP_CHECK(hipModuleLaunchKernel(
      kernel,
      grid_x,
      grid_y,
      grid_z,
      kWavesPerThreadgroup * 64,
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
      &fmha_asm_forward_out,
      "Launch MI350 256x64 fmha asm kernel into an existing output tensor");
}
