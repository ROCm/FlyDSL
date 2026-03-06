"""PA Decode FP8 — FlyDSL (single launch, grid z=partitions) vs Gluon."""
import sys, os, torch, math, logging
sys.path.insert(0, 'build-fly/python_packages'); sys.path.insert(1, '.')
os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '1'

logging.basicConfig(level=logging.INFO, format='%(message)s')

from tests.test_common import run_perftest, verify_output

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
HEAD_SIZE = 128; NKV = 4; QG = 16; NH = NKV * QG; BS = 16
CTX = 4096; BPS = CTX // BS; CPSZ = 256
NUM_PARTS = CTX // CPSZ  # 16
fp8 = torch.float8_e4m3fnuz; bf16 = torch.bfloat16; dev = 'cuda'
scale = 1.0 / math.sqrt(HEAD_SIZE)

# ── random data (shared) ────────────────────────────────────────
torch.manual_seed(42)
total = BPS * batch
q  = (torch.randn(batch, NH, HEAD_SIZE, device=dev) * 0.1).to(fp8)
kc = (torch.randn(total, NKV, HEAD_SIZE // 16, BS, 16, device=dev) * 0.1).to(fp8)
vc = (torch.randn(total, NKV, HEAD_SIZE, BS, device=dev) * 0.1).to(fp8)
bt = torch.zeros(batch, BPS, dtype=torch.int32, device=dev)
for i in range(batch):
    bt[i] = torch.arange(i * BPS, (i + 1) * BPS, dtype=torch.int32, device=dev)

# ── Gluon: run first (mp=1 full context) ────────────────────────
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
gl_inter = (batch, NKV, 1, QG)
cl_t   = torch.full((batch,), CTX, dtype=torch.int32, device=dev)
gl_es  = torch.empty(gl_inter, dtype=torch.float32, device=dev)
gl_ml  = torch.empty(gl_inter, dtype=torch.float32, device=dev)
gl_tmp = torch.empty(*gl_inter, HEAD_SIZE, dtype=bf16, device=dev)
gl_out = torch.empty(batch, NH, HEAD_SIZE, dtype=bf16, device=dev)

print("running Gluon (reference)...", flush=True)
pa_decode_gluon(gl_out, q, kc, vc, cl_t, bt, scale, 1, 1, CPSZ, fp8,
                None, None, None, gl_es, gl_ml, gl_tmp, None)
torch.cuda.synchronize()
print("Gluon done", flush=True)

# ── FlyDSL: single launch with grid z = NUM_PARTS ──────────────
from kernels.pa_decode_fp8 import build_pa_decode_module, BLOCK_THREADS
import kernels.pa_decode_fp8 as _pa
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir as _ir
import flydsl.compiler as flyc, flydsl.expr as fx

fd_kfn = build_pa_decode_module(batch, NKV, NUM_PARTS, BPS)
fd_al = _pa.allocator

fd_out = torch.zeros(batch, NKV, NUM_PARTS, QG, HEAD_SIZE, dtype=bf16, device=dev)
fd_es  = torch.zeros(batch, NKV, NUM_PARTS, QG, dtype=torch.float32, device=dev)
fd_ml  = torch.full((batch, NKV, NUM_PARTS, QG), float('-inf'), dtype=torch.float32, device=dev)

@flyc.jit
def fd_launch(out, es, ml, q, kc, vc, bt, cl: fx.Int32,
              stream: fx.Stream = fx.Stream(None)):
    fd_al.finalized = False
    ctx = CompilationContext.get_current()
    with _ir.InsertionPoint(ctx.gpu_module_body):
        fd_al.finalize()
    fd_kfn(out, es, ml, q, kc, vc, bt, cl).launch(
        grid=(batch, NKV, NUM_PARTS), block=(BLOCK_THREADS, 1, 1), stream=stream)

print(f"FlyDSL: single launch grid=({batch},{NKV},{NUM_PARTS})...", flush=True)
fd_launch(fd_out, fd_es, fd_ml, q, kc, vc, bt, CTX)
torch.cuda.synchronize()
print("FlyDSL done", flush=True)

# ── Reduce FlyDSL partitions ────────────────────────────────────
ml = fd_ml                                          # [batch, NKV, P, QG]
es = fd_es                                          # [batch, NKV, P, QG]
out_s = fd_out                                      # [batch, NKV, P, QG, HEAD_SIZE]
G  = ml.max(dim=2).values                           # [batch, NKV, QG]
w  = es * torch.exp(ml - G.unsqueeze(2))            # [batch, NKV, P, QG]
w_sum = w.sum(dim=2, keepdim=True)                  # [batch, NKV, 1, QG]
norm_w = (w / w_sum).unsqueeze(-1)                  # [batch, NKV, P, QG, 1]
fd_final = (out_s.float() * norm_w).sum(dim=2).to(bf16)  # [batch, NKV, QG, HEAD_SIZE]

# ── Correctness ─────────────────────────────────────────────────
print(f"\n=== Correctness: FlyDSL (grid z={NUM_PARTS}) vs Gluon bs={batch} ===")
verify_output(fd_final.reshape(batch, NH, HEAD_SIZE).float(),
              gl_out.float(), msg='FlyDSL vs Gluon',
              atol=0.1, rtol=0.1)

# ── Performance ─────────────────────────────────────────────────
print(f"\n=== Performance: bs={batch}, CTX={CTX} ===")

def fd_run():
    fd_launch(fd_out, fd_es, fd_ml, q, kc, vc, bt, CTX)

def gl_run():
    pa_decode_gluon(gl_out, q, kc, vc, cl_t, bt, scale, 1, 1, CPSZ, fp8,
                    None, None, None, gl_es, gl_ml, gl_tmp, None)

_, fd_us = run_perftest(fd_run, num_iters=100, num_warmup=5, num_rotate_args=1)
_, gl_us = run_perftest(gl_run, num_iters=100, num_warmup=5, num_rotate_args=1)

print(f"\n[Summary] bs={batch} CTX={CTX}")
print(f"  FlyDSL (1 launch, {NUM_PARTS} partitions): {fd_us:.2f} us/iter")
print(f"  Gluon  (full ctx):                {gl_us:.2f} us/iter")
print(f"  Speedup: {gl_us / fd_us:.2f}x")
