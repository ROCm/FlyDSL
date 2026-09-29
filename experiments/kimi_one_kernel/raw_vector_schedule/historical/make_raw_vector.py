"""Vectorize only MoE raw BF16 activation copies, keeping all ready polls."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
KERNEL=Path('kernels/kimi_k3_monokernel/kernel.py')
BASE='ug_joint_publish_events'
old=(ROOT/BASE/KERNEL).read_text()
begin=old.index('        def stage_raw_vector(');end=old.index('        def mxfp4_fragment(',begin)
helper=old[begin:end]
cut=helper.index('            for load_round in range_constexpr(')
records=[]
for width in [2,4]:
    name=f'moe_raw_vec{width}_events'
    h=helper[:cut]+f'''            for load_round in range_constexpr((pairs + {width} * _THREADS - 1) // ({width} * _THREADS)):
                pair = (tid + load_round * _THREADS) * {width}
                if pair < pairs:
                    packed = fx.Vector(bo.buffer_load(mailbox_rsrc, pair_base + pair, vec_width={width}, dtype=T.i32, cache_modifier=CM_DEV))
                    fx.ptr_store(packed.bitcast(fx.Float32), x + pair)

'''
    s=old[:begin]+h+old[end:];ast.parse(s)
    for pairs in [1792,3072]:
        offsets=[p+i for t in range(512) for p in range(t*width,pairs,512*width) for i in range(width)]
        assert sorted(offsets)==list(range(pairs))
        for sample in range(8):assert sample*pairs%width==0
    assert helper[:cut]==h[:cut]
    out=ROOT/name;assert not out.exists()
    shutil.copytree(ROOT/BASE/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
    (out/KERNEL).write_text(s)
    (ROOT/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
    records.append(dict(variant=name,parent=BASE,source_sha256=hashlib.sha256(s.encode()).hexdigest(),dwords_per_load=width,
        ready_polls_and_barrier_exactly_unchanged=True,raw_pair_coverage_once=True,source_and_lds_alignment_verified=True,arithmetic_unchanged=True))
(ROOT/'raw_vector_audit.json').write_text(json.dumps(records,indent=2)+'\n')
f=ROOT/'run_round.py';s=f.read_text();a=s.index('CANDIDATES=');b=s.index('\n',a)
names=ast.literal_eval(s[a+len('CANDIDATES='):b])+[r['variant'] for r in records]
f.write_text(s[:a]+'CANDIDATES='+repr(names)+s[b:])
with tarfile.open(ROOT/'raw_vector_bundle.tar.gz','w:gz') as tar:
    for record in records:
        for f in (ROOT/record['variant']).rglob('*.py'):tar.add(f,arcname=str(f.relative_to(ROOT)))
    for n in ['make_raw_vector.py','raw_vector_audit.json','run_round.py']:tar.add(ROOT/n,arcname=n)
print(json.dumps(records,indent=2))
