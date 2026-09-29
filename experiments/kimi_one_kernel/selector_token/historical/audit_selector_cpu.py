"""Execute the two actual selector ASTs on CPU and check an independent stable sort.

No HIP, torch, or GPU imports. BF16 bias arrives already rounded. The CPU rcp is
only a common emulation: the test verifies equal inputs to the same GPU rcp,
not the accuracy of AMD's approximate reciprocal or compiled GPU instructions.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = ['full_tail4_proj4_ug32_events', 'selector_carry_bias_events', 'selector_local_gt_events', 'selector_reduce2_events', 'selector_native_reduce_events']
N = 896
TOP = 16

class V(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)

    def select(self, yes, no):
        return V(np.where(self, yes, no))

    def bitcast(self, target):
        return V(np.asarray(self).view(target.dtype))

def caster(dtype):
    def cast(value):
        return V(value, dtype)
    cast.dtype = dtype
    return cast

def bf(value):
    words = np.asarray(value, np.float32).copy().view(np.uint32)
    words = (words + np.uint32(0x7FFF) + ((words >> 16) & 1)) & np.uint32(0xFFFF0000)
    return words.view(np.float32)

class LiftLaneZero(ast.NodeTransformer):
    """Run lane-zero stores for all lanes, then assert wave agreement."""
    def visit_If(self, node):
        assert ast.unparse(node.test) == 'lane == 0', ast.unparse(node.test)
        assert not node.orelse
        return node.body

def compile_selector(name):
    source = (ROOT / name / 'kernels/kimi_k3_monokernel/kernel.py').read_text()
    tree = ast.parse(source)
    body = next(n.body for n in ast.walk(tree) if isinstance(n, ast.If)
                and n.body and isinstance(n.body[0], ast.Assign)
                and ast.unparse(n.body[0]) == 'scores = []')
    module = LiftLaneZero().visit(ast.Module(body=body, type_ignores=[]))
    ast.fix_missing_locations(module)
    return compile(module, name + ':selector', 'exec'), hashlib.sha256(source.encode()).hexdigest()

def execute(code, scores, bias):
    batch = len(scores)
    ids, raw, weights = {}, {}, {}
    lane = V(np.arange(64, dtype=np.int32))
    def gather(values, index):
        index = np.broadcast_to(index, (batch, 64))
        return V(np.take_along_axis(values, index, axis=1))
    def store(dst, index, value):
        v = np.broadcast_to(value, (batch, 64))
        assert np.array_equal(v.view(np.uint32), np.broadcast_to(v[:, :1], v.shape).view(np.uint32))
        dst[int(index)] = v[:, 0].copy()
    ns = dict(
        lane=lane, wave=0, sample=0, _N_EXPERTS=N, _WAVE_SIZE=64, _TOP_K=TOP,
        range_constexpr=range, ArithValue=lambda x: x,
        fx=SimpleNamespace(Float32=caster(np.float32),
                           Int32=caster(np.int32), BFloat16=caster(np.float32),
                           max=lambda a,b: V(np.maximum(a,b)), min=lambda a,b: V(np.minimum(a,b)),
                           ReductionOp=SimpleNamespace(MAX='max', MIN='min'),
                           coop=SimpleNamespace(warp_reduce=lambda value,op,width: V(np.broadcast_to(
                               (np.max if op=='max' else np.min)(value, axis=-1, keepdims=True), value.shape)))),
        bo=SimpleNamespace(buffer_load=lambda resource, expert, **kw: gather(bias, expert)),
        T=SimpleNamespace(bf16='bf16', i32='i32'), rsrc=lambda x: x,
        uniform=lambda x: V(x),
        rocdl=SimpleNamespace(readlane=lambda dtype,value,index: gather(value,index)),
        correction_bias=None, router_mailbox_rsrc=None,
        load_raw_f32=lambda resource, index: gather(scores, index),
        xshfl=lambda value, offset: V(value[..., np.arange(64) ^ offset]),
        rcp=lambda value: V(np.float32(1.0) / value, np.float32),
        selection_id_rsrc=ids, selection_weight_rsrc=weights, output_values=raw,
        store_i32=store, store_f32=store, lds_store=store,
        lds_load=lambda resource, index: V(resource[int(index)][:, None]))
    with np.errstate(invalid='ignore', divide='ignore'):
        exec(code, ns)
    return dict(ids=np.stack([ids[i] for i in range(TOP)], axis=1),
                raw=np.stack([raw[i] for i in range(TOP)], axis=1),
                weights=np.stack([weights[i] for i in range(TOP)], axis=1),
                selected_sum=np.asarray(ns['selected_sum'])[:, 0].copy())

def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))

def batches():
    rng = np.random.default_rng(20260929)
    # Random finite logits/biases plus saturation and very small sigmoid scores.
    for i in range(16):
        logits = bf(rng.normal(0, [0.25, 1, 4, 12][i % 4], (256, N)))
        scores = np.asarray(1 / (1 + np.exp(-logits.astype(np.float64))), np.float32)
        bias = bf(rng.normal(0, [0.01, 0.25, 2, 32][i % 4], (256, N)))
        yield 'random_sigmoid', scores, bias
    for family in ['all_ties', 'corrected_ties', 'ulp_neighbors', 'signed_zero',
                   'cross_lane_ties', 'saturated', 'cancellation']:
        scores = np.full((64, N), np.float32(0.5))
        bias = np.zeros_like(scores)
        for i in range(64):
            order = rng.permutation(N)
            if family == 'corrected_ties':
                bias[i] = bf(rng.choice([0.0, 0.125, 0.25], N))
                scores[i] = np.float32(0.75) - bias[i]
            elif family == 'ulp_neighbors':
                words = scores[i].view(np.uint32)
                words[:] += rng.integers(0, 4, N, dtype=np.uint32)
            elif family == 'signed_zero':
                scores[i] = rng.choice(np.array([0.0, -0.0], np.float32), N)
                scores[i, order[:8]] = np.float32(0.5)
            elif family == 'cross_lane_ties':
                scores[i, order[:32]] = np.float32(0.75)
            elif family == 'saturated':
                scores[i] = rng.choice(np.array([0.0, 1.0, 0.5], np.float32), N)
            elif family == 'cancellation':
                scores[i] = rng.random(N, dtype=np.float32)
                bias[i] = bf(rng.choice([0, 0.5, 4, 32, 256], N))
        yield family, scores, bias

def main():
    programs = [compile_selector(s) for s in SOURCES]
    counts = {}
    different_reconstructed_raw = 0
    for family, scores, bias in batches():
        assert np.array_equal(bias, bf(bias))
        results = [execute(p, scores, bias) for p, _ in programs]
        old = results[0]
        for name, new in zip(SOURCES[1:], results[1:]):
            for key in old:
                assert bits_equal(old[key], new[key]), (family, name, key)
        corrected = scores + bias
        ids = np.argsort(-corrected, axis=1, kind='stable')[:, :TOP].astype(np.int32)
        assert np.array_equal(old['ids'], ids), family
        raw = np.take_along_axis(corrected, ids, axis=1) - np.take_along_axis(bias, ids, axis=1)
        assert bits_equal(old['raw'], raw), family
        total = np.zeros(len(scores), np.float32)
        for i in range(TOP):
            total = total + raw[:, i]
        assert bits_equal(old['selected_sum'], total), family
        different_reconstructed_raw += int(np.count_nonzero(
            raw.view(np.uint32) != np.take_along_axis(scores, ids, axis=1).view(np.uint32)))
        counts[family] = counts.get(family, 0) + len(scores)
    result = dict(cpu_only=True, cases=sum(counts.values()), selected_routes=sum(counts.values()) * TOP,
        cases_by_family=counts, actual_selector_asts_executed=True,
        all_lanes_agree=True, ids_raw_sum_and_emulated_weights_bitwise_equal=True,
        independent_stable_sort_ids_equal=True, independent_reconstructed_raw_and_ordered_sum_equal=True,
        reconstructed_raw_differs_from_original_score_count=different_reconstructed_raw,
        sources=dict(zip(SOURCES, [h for _, h in programs])),
        limitations=['CPU reciprocal emulates both equally; not a hardware reciprocal accuracy test.',
                    'Finite score/bias inputs only; not a GPU compiler, memory-order or performance test.'])
    (ROOT / 'selector_cpu_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
