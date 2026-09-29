"""Prefetch complete selected UG tiles before tagged latent activation waits."""
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
BASE=sys.argv[1] if len(sys.argv)>1 else 'tail_rotate192_downsplit_events'
old=(ROOT/BASE/KERNEL).read_text()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
begin=old.index('            # Stage 6:');end=old.index('            stamp(6)',begin)
up=old[begin:end]
records=[]
for tiles in [1,2]:
    name=f'ug_pf{tiles}_events'
    code=f'''                up_prefetched = []
                for pre_tile in range_constexpr({tiles}):
                    pre_row = (wave < 4).select(first_row_group + pre_tile, first_row_group + pre_tile + _INTER // 16)
                    for pre_chunk in range_constexpr((_ROUTED_HIDDEN // 128) // 4):
                        up_prefetched.append(mxfp4_fragment(
                            up_weight_rsrc, up_scale_rsrc, pre_row,
                            (wave % 4) * ((_ROUTED_HIDDEN // 128) // 4) + pre_chunk,
                            _ROUTED_HIDDEN,
                        ))
'''
    s=once(up,'                stage_raw_vector(',code+'                stage_raw_vector(')
    a=s.index('                        fragment = mxfp4_fragment(');b=s.index('                        accumulator = mxfp4_apply(',a)
    fragment=s[a:b]
    replacement=f'                        if const_expr(paired_tile < {tiles}):\n                            fragment = up_prefetched[paired_tile * chunks_per_wave + local_chunk]\n                        else:\n'+''.join('    '+line if line.strip() else line for line in fragment.splitlines(True))
    s=once(s,fragment,replacement)
    s=old[:begin]+s+old[end:]
    ast.parse(s)
    out=ROOT/name;assert not out.exists()
    shutil.copytree(ROOT/BASE/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
    (out/KERNEL).write_text(s)
    (ROOT/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
    records.append(dict(variant=name,parent=BASE,source_sha256=hashlib.sha256(s.encode()).hexdigest(),prefetch_tiles=tiles,
        arithmetic_and_bf16_order_unchanged=True,selected_expert_id_wait_retained=True,activation_waits_unchanged=True,dag_unchanged=True))
(ROOT/'up_prefetch_audit.json').write_text(json.dumps(records,indent=2)+'\n')
f=ROOT/'run_round.py';s=f.read_text();a=s.index('CANDIDATES=');b=s.index('\n',a)
names=ast.literal_eval(s[a+len('CANDIDATES='):b])+[r['variant'] for r in records]
f.write_text(s[:a]+'CANDIDATES='+repr(names)+s[b:])
with tarfile.open(ROOT/'up_prefetch_bundle.tar.gz','w:gz') as tar:
    for record in records:
        for f in (ROOT/record['variant']).rglob('*.py'):tar.add(f,arcname=str(f.relative_to(ROOT)))
    for n in ['make_up_prefetch.py','up_prefetch_audit.json','run_round.py']:tar.add(ROOT/n,arcname=n)
print(json.dumps(records,indent=2))
