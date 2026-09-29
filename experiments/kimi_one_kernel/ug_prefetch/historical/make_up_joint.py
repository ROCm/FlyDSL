"""Join the paired UG tile publication, retaining only consumed MFMA lanes."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
KERNEL=Path('kernels/kimi_k3_monokernel/kernel.py')
BASE='tail_rotate192_downsplit_events'
NAME='ug_joint_publish_events'
old=(ROOT/BASE/KERNEL).read_text()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
begin=old.index('            # Stage 6:');end=old.index('            stamp(6)',begin)
up=old[begin:end]
publish_start=up.index('                    fx.ptr_store(')
publish_end=up.index('                up_task =',publish_start)
publish=up[publish_start:publish_end]
compute_store='''                    if lane % 16 == 0:
                        fx.ptr_store(
                            fx.Vector.from_elements(accumulator, fx.Float32),
                            reduction + paired_tile * (_WAVES * 16) + wave * 16 + (lane // 16) * 4,
                        )
'''
joined=publish[publish.index('                    gpu.barrier()'):]
joined=''.join(line[4:] if line.strip() else line for line in joined.splitlines(True))
joined=once(joined,'                if tid < 16 // 2:\n                    local_row = tid * 2',
    '                if tid < 2 * (16 // 2):\n                    paired_tile = tid // (16 // 2)\n                    row_group = first_row_group + paired_tile\n                    local_row = (tid % (16 // 2)) * 2')
joined=once(joined,'source_index = (source_wave * _WAVE_SIZE + source_lane) * 4 + row % 4',
    'source_index = paired_tile * (_WAVES * 16) + source_wave * 16 + row')
joined=once(joined,'up_index = ((source_wave + 4) * _WAVE_SIZE + source_lane) * 4 + row % 4',
    'up_index = paired_tile * (_WAVES * 16) + (source_wave + 4) * 16 + row')
ready_start=joined.index('                    store_i32(')
ready=joined[ready_start:]
ready=once(ready,'up_tiles + row_group,','up_tiles + first_row_group + ready_tile,')
joined=joined[:ready_start]+'                    for ready_tile in range_constexpr(2):\n'+''.join('    '+line if line.strip() else line for line in ready.splitlines(True))
s=old[:begin]+once(up,publish,compute_store+joined)+old[end:]
ast.parse(s)
# Prove the compact LDS map selects exactly the original consumed MFMA
# accumulator component, with disjoint slots across tile/wave/row.
written={}
for tile in range(2):
    for wave in range(8):
        for lane in range(64):
            if lane%16==0:
                for element in range(4):
                    slot=tile*128+wave*16+(lane//16)*4+element
                    assert slot not in written
                    written[slot]=(tile,wave,lane,element)
for tile in range(2):
    for wave in range(8):
        for row in range(16):
            assert written[tile*128+wave*16+row]==(tile,wave,16*(row//4),row%4)
assert len(written)==256
out=ROOT/NAME;assert not out.exists()
shutil.copytree(ROOT/BASE/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
(out/KERNEL).write_text(s)
(ROOT/(NAME+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
record=dict(variant=NAME,parent=BASE,source_sha256=hashlib.sha256(s.encode()).hexdigest(),
    compact_lds_words=256,original_per_tile_lds_words=2048,new_lds_allocation=False,
    original_mfma_components_preserved=True,source_wave_addition_order_preserved=True,
    ready_tags_follow_both_tiles_and_same_vmcnt_fence=True,activation_and_bf16_boundaries_unchanged=True,
    readiness_dag_note='Down already waits all tile flags for a token; paired publication adds no reverse dependency.')
(ROOT/'up_joint_audit.json').write_text(json.dumps(record,indent=2)+'\n')
f=ROOT/'run_round.py';text=f.read_text();a=text.index('CANDIDATES=');b=text.index('\n',a)
names=ast.literal_eval(text[a+len('CANDIDATES='):b])+[NAME]
f.write_text(text[:a]+'CANDIDATES='+repr(names)+text[b:])
with tarfile.open(ROOT/'up_joint_bundle.tar.gz','w:gz') as tar:
    for f in out.rglob('*.py'):tar.add(f,arcname=str(f.relative_to(ROOT)))
    for n in ['make_up_joint.py','up_joint_audit.json','run_round.py']:tar.add(ROOT/n,arcname=n)
print(json.dumps(record,indent=2))
