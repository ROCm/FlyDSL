"""Isolated tail send/collect scheduling; all arithmetic and mailbox ABI retained."""
import ast
from collections import defaultdict, deque
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'kimi_frontend_prefetch_20260929'
KERNEL = Path('kernels/kimi_k3_monokernel/kernel.py')
BASE = 'front_wave_pf7_recur_events'
STAGED = 'staged_relocate_events'

def once(s, a, b):
    assert s.count(a) == 1, (a, s.count(a))
    return s.replace(a, b)

def copy(name, source):
    assert not (ROOT/name).exists()
    shutil.copytree(source/'kernels', ROOT/name/'kernels', ignore=shutil.ignore_patterns('__pycache__'))

for name in [BASE, STAGED]:
    copy(name, OLD/name)
for name in ['compile_full.py','check_occupancy.py','full_replay.py','run_jobs.py','run_round.py','summarize_validation.py']:
    shutil.copyfile(OLD/name, ROOT/name)
old = (ROOT/BASE/KERNEL).read_text()

start = old.index('        def moe_peer_reduce_samples(')
end = old.index('        def publish_mfma_pairs(', start)
helper = old[start:end]
split = helper.index('            if tid < local_pairs:')
push = helper[:split]
push = once(push, 'def moe_peer_reduce_samples(local_pairs, pair_base, local_values, region, emit):',
            'def moe_peer_push_samples(local_pairs, pair_base, local_values, region):')
collect = ('        def moe_peer_collect_samples(local_pairs, pair_base, region, emit):\n' +
           helper[helper.index('            moe_max_pairs'):helper.index('            peer_rounds')] + helper[split:])
assert push.endswith('            gpu.barrier()\n')
# The exact original sys-scope packed value/tag sends, CTA barrier, polling,
# ascending peer reduction and final barrier are preserved in the two halves.
factored = once(old, helper, push+'\n'+collect)
begin = factored.index('                def emit_final(')
finish = factored.index('                tail_task = tail_task + _BLOCKS', begin)
emitter = factored[begin:finish]
emitter = once(emitter, 'moe_peer_reduce_samples(staged_samples * (16 // 2), pair_base, output_values, 1, emit_final)',
               'moe_peer_collect_samples(staged_samples * (16 // 2), pair_base, 1, emit_final)')
body = factored[begin:finish]
factored = once(factored, body, '                moe_peer_push_samples(staged_samples * (16 // 2), pair_base, output_values, 1)\n')
endloop = '                tail_task = tail_task + _BLOCKS\n            stamp(9)'
collect_loop = '''                tail_task = tail_task + _BLOCKS

            # The tagged symmetric mailboxes retain every partial, so LDS can
            # be reused for the next compute task before any peer is polled.
            tail_task = bid
            while tail_task < tail_tasks:
                sample_base = (tail_task // hidden_tiles) * staged_samples
                row_group = tail_task % hidden_tiles
                pair_base = sample_base * (_HIDDEN // 2) + row_group * (16 // 2)

'''+emitter+'                tail_task = tail_task + _BLOCKS\n            stamp(9)'
factored = once(factored, endloop, collect_loop)

def audit_dag(rotation):
    graph = defaultdict(set)
    def edge(a,b):
        graph[a].add(b); graph.setdefault(b,set())
    for rank in range(8):
        for bid in range(256):
            previous = (rank,'prefix',bid)
            for task in range(bid,768,256):
                node=(rank,'up',task)
                edge(previous,node); edge(node,(rank,'mid',task//192)); previous=node
            for task in range(bid,896,256):
                node=(rank,'down_compute',task)
                edge(previous,node); edge((rank,'mid',task//224),node)
                edge(node,('down_join',task))
                done=(rank,'down_done',task)
                edge(('down_join',task),done); edge(done,(rank,'norm',task//224)); previous=done
            if 245 <= bid < 249:
                node=(rank,'norm',bid-245); edge(previous,node); previous=node
            tasks=[(t+(rank*56-rotation if rotation is not None else 0))%448 for t in range(bid,448,256)]
            for task in tasks:
                node=(rank,'tail_push',task); edge(previous,node)
                if rank*56 <= task < (rank+1)*56:
                    for sample in range(4):edge((rank,'norm',sample),node)
                edge(node,('tail_join',task)); previous=node
            for task in tasks:
                node=(rank,'tail_collect',task)
                edge(previous,node); edge(('tail_join',task),node); previous=node
    indegree=dict.fromkeys(graph,0)
    for targets in graph.values():
        for target in targets:indegree[target]+=1
    q=deque(n for n,d in indegree.items() if not d); visited=0
    while q:
        n=q.popleft(); visited+=1
        for d in graph[n]:
            indegree[d]-=1
            if not indegree[d]:q.append(d)
    assert visited==len(graph),[n for n,d in indegree.items() if d][:20]
    return dict(nodes=visited,edges=sum(map(len,graph.values())),acyclic=True)

records=[]
for name,rotation in [('tail_split_events',None),('tail_rotate0_events',0),('tail_rotate192_events',192)]:
    s=factored
    replacements=[]
    if rotation is not None:
        a='                row_group = tail_task % hidden_tiles'
        b=f'                row_group = (tail_task % hidden_tiles + rank * (_HIDDEN_SHARD // 16) + hidden_tiles - {rotation}) % hidden_tiles'
        assert s.count(a)==2
        s=s.replace(a,b)
    ast.parse(s)
    for samples in [1,2,4,8]:
        groups=(samples+min(samples,4)-1)//min(samples,4)
        for rank in range(8):
            tasks=[]
            for bid in range(256):
                for task in range(bid,groups*448,256):
                    group=task//448
                    row=(task%448+(rank*56-rotation if rotation is not None else 0))%448
                    tasks.append((group,row))
            assert sorted(tasks)==[(g,r) for g in range(groups) for r in range(448)]
    copy(name, ROOT/BASE); (ROOT/name/KERNEL).write_text(s)
    (ROOT/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
    records.append(dict(variant=name,parent=BASE,source_sha256=hashlib.sha256(s.encode()).hexdigest(),
        rotation=rotation,coverage_samples=[1,2,4,8],all_tasks_unique=True,tp8_s4_dag=audit_dag(rotation),
        mailbox_layout_unchanged=True,no_new_scratch=True,packed_value_tag_sends_and_barriers_unchanged=True,
        arithmetic_and_bf16_boundaries_unchanged=True,peer_addition_order_unchanged=True,
        caveat='DAG starts after unchanged prefix and assumes resident fair execution; actual HIP occupancy and replay required.'))

for f in [ROOT/'run_round.py',ROOT/'summarize_validation.py']:
    s=f.read_text().replace("BASE='selector_native_reduce_events'",f"BASE='{BASE}'").replace("BASE = 'selector_native_reduce_events'",f"BASE = '{BASE}'")
    if f.name=='run_round.py':
        start=s.index('CANDIDATES=');end=s.index('\n',start)
        s=s[:start]+'CANDIDATES='+repr([r['variant'] for r in records])+s[end:]
        s=s.replace("FINE='front_fine_timeline'","FINE='tail_base_fine_timeline'")
    f.write_text(s)
(ROOT/'source_audit.json').write_text(json.dumps(records,indent=2)+'\n')
manifest={}
for name in [BASE,STAGED,*[r['variant'] for r in records]]:
    for f in (ROOT/name).rglob('*.py'):
        ast.parse(f.read_text());manifest[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(ROOT/'round_bundle.tar.gz','w:gz') as tar:
    for name in manifest:tar.add(ROOT/name,arcname=name)
    for f in ROOT.glob('*.py'):tar.add(f,arcname=f.name)
    for f in ROOT.glob('*.json'):tar.add(f,arcname=f.name)
print(json.dumps(records,indent=2))
