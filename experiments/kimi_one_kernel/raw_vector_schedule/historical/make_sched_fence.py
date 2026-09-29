"""Compile-only candidates: bound UG tile scheduling without CTA sync."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
KERNEL=Path('kernels/kimi_k3_monokernel/kernel.py')
records=[]
for parent in ['ug_joint_publish_events','moe_raw_vec2_events','moe_raw_vec4_events']:
    name=parent.removesuffix('_events')+'_sched_events'
    old=(ROOT/parent/KERNEL).read_text()
    begin=old.index('            # Stage 6:');end=old.index('            stamp(6)',begin)
    u=old[begin:end]
    a='                    if lane % 16 == 0:'
    assert u.count(a)==1
    u=u.replace(a,'                    rocdl.sched_barrier(0)\n'+a)
    a='                gpu.barrier()\n                if tid < 2 * (16 // 2):'
    assert u.count(a)==1
    u=u.replace(a,'                    rocdl.sched_barrier(0)\n'+a)
    s=old[:begin]+u+old[end:]
    assert s.replace('                    rocdl.sched_barrier(0)\n','')==old
    ast.parse(s)
    out=ROOT/name;assert not out.exists()
    shutil.copytree(ROOT/parent/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
    (out/KERNEL).write_text(s)
    (ROOT/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
    records.append(dict(variant=name,parent=parent,source_sha256=hashlib.sha256(s.encode()).hexdigest(),
        all_non_sched_source_unchanged=True,added_cta_barriers=0,mode='compile-only while node46 occupied'))
(ROOT/'sched_fence_audit.json').write_text(json.dumps(records,indent=2)+'\n')
f=ROOT/'run_round.py';s=f.read_text();a=s.index('CANDIDATES=');b=s.index('\n',a)
names=ast.literal_eval(s[a+len('CANDIDATES='):b])+[r['variant'] for r in records]
f.write_text(s[:a]+'CANDIDATES='+repr(names)+s[b:])
with tarfile.open(ROOT/'sched_fence_bundle.tar.gz','w:gz') as tar:
    for record in records:
        for f in (ROOT/record['variant']).rglob('*.py'):tar.add(f,arcname=str(f.relative_to(ROOT)))
    for n in ['make_sched_fence.py','sched_fence_audit.json','run_round.py']:tar.add(ROOT/n,arcname=n)
print(json.dumps(records,indent=2))
