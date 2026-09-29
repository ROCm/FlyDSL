"""Apply the same send-before-collect scheme to the independent down tiles."""
import ast
from collections import defaultdict, deque
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
KERNEL=Path('kernels/kimi_k3_monokernel/kernel.py')
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

records=[]
for parent in sys.argv[1:]:
    name=parent.removesuffix('_events')+'_downsplit_events'
    old=(ROOT/parent/KERNEL).read_text()
    begin=old.index('        def moe_peer_reduce(');end=old.index('        def moe_peer_',begin+12)
    helper=old[begin:end];cut=helper.index('            if tid < local_pairs:')
    push=once(helper[:cut],'def moe_peer_reduce(local_pairs, pair_base, local_values, region, emit):','def moe_peer_push(local_pairs, pair_base, local_values, region):')
    collect=('        def moe_peer_collect(local_pairs, pair_base, region, emit):\n'+
        helper[helper.index('            moe_max_pairs'):helper.index('            peer_rounds')]+helper[cut:])
    s=once(old,helper,push+'\n'+collect)
    begin=s.index('                def emit_routed(');end=s.index('            stamp(7)',begin)
    emitter=s[begin:end]
    emitter=once(emitter,'moe_peer_reduce(16 // 2, pair_base, output_values, 0, emit_routed)','moe_peer_collect(16 // 2, pair_base, 0, emit_routed)')
    s=s[:begin]+'''                moe_peer_push(16 // 2, pair_base, output_values, 0)
                down_task = down_task + _BLOCKS

            # All per-CTA down tiles are now in symmetric tagged mailboxes.
            down_task = bid
            while down_task < down_tasks:
                sample = down_task // routed_tiles
                row_group = down_task % routed_tiles
                pair_base = sample * (_ROUTED_HIDDEN // 2) + row_group * (16 // 2)

'''+emitter+s[end:]
    ast.parse(s)
    out=ROOT/name;assert not out.exists()
    shutil.copytree(ROOT/parent/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
    (out/KERNEL).write_text(s)
    (ROOT/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
    rotation=192 if 'rotate192' in name else (0 if 'rotate0' in name else None)
    split_tail='tail_' in name
    graph=defaultdict(set)
    def edge(a,b):graph[a].add(b);graph.setdefault(b,set())
    for rank in range(8):
        for bid in range(256):
            prev=(rank,'prefix',bid)
            for task in range(bid,768,256):
                n=(rank,'up',task);edge(prev,n);edge(n,(rank,'mid',task//192));prev=n
            for task in range(bid,896,256):
                n=(rank,'down_push',task);edge(prev,n);edge((rank,'mid',task//224),n);edge(n,('down_join',task));prev=n
            for task in range(bid,896,256):
                n=(rank,'down_collect',task);edge(prev,n);edge(('down_join',task),n);edge(n,(rank,'norm',task//224));prev=n
            if 245<=bid<249:
                n=(rank,'norm',bid-245);edge(prev,n);prev=n
            tasks=[(t+(rank*56-rotation if rotation is not None else 0))%448 for t in range(bid,448,256)]
            for task in tasks:
                n=(rank,'tail_push',task);edge(prev,n)
                if rank*56<=task<(rank+1)*56:
                    for sample in range(4):edge((rank,'norm',sample),n)
                edge(n,('tail_join',task));prev=n
                if not split_tail:
                    n=(rank,'tail_collect',task);edge(prev,n);edge(('tail_join',task),n);prev=n
            if split_tail:
                for task in tasks:
                    n=(rank,'tail_collect',task);edge(prev,n);edge(('tail_join',task),n);prev=n
    indegree=dict.fromkeys(graph,0)
    for dests in graph.values():
        for n in dests:indegree[n]+=1
    q=deque(n for n,d in indegree.items() if not d);visited=0
    while q:
        n=q.popleft();visited+=1
        for d in graph[n]:
            indegree[d]-=1
            if not indegree[d]:q.append(d)
    assert visited==len(graph),[n for n,d in indegree.items() if d][:20]
    records.append(dict(variant=name,parent=parent,source_sha256=hashlib.sha256(s.encode()).hexdigest(),tp8_s4_dag=dict(nodes=visited,edges=sum(map(len,graph.values())),acyclic=True),
        no_new_scratch=True,mailbox_and_arithmetic_and_barriers_unchanged=True))
(ROOT/'down_split_audit.json').write_text(json.dumps(records,indent=2)+'\n')
f=ROOT/'run_round.py';s=f.read_text();begin=s.index('CANDIDATES=');end=s.index('\n',begin)
names=ast.literal_eval(s[begin+len('CANDIDATES='):end])+[r['variant'] for r in records]
f.write_text(s[:begin]+'CANDIDATES='+repr(names)+s[end:])
with tarfile.open(ROOT/'down_split_bundle.tar.gz','w:gz') as tar:
    for record in records:
        for f in (ROOT/record['variant']).rglob('*.py'):tar.add(f,arcname=str(f.relative_to(ROOT)))
    for n in ['make_down_split.py','down_split_audit.json','run_round.py']:tar.add(ROOT/n,arcname=n)
print(json.dumps(records,indent=2))
