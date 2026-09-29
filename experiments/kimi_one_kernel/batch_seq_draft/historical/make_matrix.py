"""Generalize isolated benchmark copies to independent batch/sequence chains."""
import ast, difflib, hashlib, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'kimi_tail_async_20260929'
K=Path('kernels/kimi_k3_monokernel')
SOURCES={'current':'tail_rotate192_downsplit_events','staged':'staged_relocate_events'}
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def grow_loop(s,test,bound,var):
    nodes=[n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.If) and ast.unparse(n.test)==test]
    assert len(nodes)==1,(test,len(nodes))
    n=nodes[0];lines=s.splitlines(True);old=''.join(lines[n.lineno-1:n.end_lineno])
    ind=' '*n.col_offset;body=''.join(lines[n.lineno:n.end_lineno])
    new=ind+f'if const_expr({bound} <= _BLOCKS):\n'+''.join('    '+x for x in old.splitlines(True))
    new+=ind+'else:\n'+ind+f'    while {var} < {bound}:\n'+''.join('    '+x for x in body.splitlines(True))
    new+=ind+f'        {var} = {var} + _BLOCKS\n'
    return once(s,old,new)

for name,parent in SOURCES.items():
    out=ROOT/name;assert not out.exists()
    shutil.copytree(OLD/parent/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
    f=out/K/'kernel.py';s=f.read_text()
    s=s.replace('if samples not in {1, 2, 4, 8}:','if not 1 <= samples <= 32:')
    s=s.replace('samples must be one of {{1, 2, 4, 8}}','samples must be in [1, 32]')
    s=once(s,'    mtp: bool = False,\n):','    mtp: bool = False,\n    mtp_seq_len: int | None = None,\n):')
    point='    fuse_attn_res = attn_res_blocks >= 0\n'
    s=once(s,point,'    mtp_seq_len = samples if mtp_seq_len is None else mtp_seq_len\n    if mtp and (mtp_seq_len < 1 or samples % mtp_seq_len):\n        raise ValueError("samples must be divisible by mtp_seq_len")\n'+point)
    if name=='current':
        s=once(s,'    if fuse_moe and (samples != 4 or not mtp):\n        raise ValueError("Experimental projection split-K4 requires S4 true MTP with fused MoE")\n','')
    s=once(s,'    staged_samples = min(samples, 4)','    staged_samples = max(n for n in range(1, min(samples, 4) + 1) if samples % n == 0)')
    s=s.replace('if sample > 0:', 'if sample % mtp_seq_len > 0:')
    s=s.replace('input_slot = uniform(bo.buffer_load(indices_rsrc, sample, vec_width=1, dtype=T.i32))',
                'input_slot = uniform(bo.buffer_load(indices_rsrc, sample + sample // mtp_seq_len, vec_width=1, dtype=T.i32))')
    s=s.replace('output_slot = uniform(bo.buffer_load(indices_rsrc, sample + 1, vec_width=1, dtype=T.i32))',
                'output_slot = uniform(bo.buffer_load(indices_rsrc, sample + sample // mtp_seq_len + 1, vec_width=1, dtype=T.i32))')
    s=grow_loop(s,'conv_active','conv_tasks','conv_task')
    s=grow_loop(s,'mtp_recurrence_task < mtp_recurrence_tasks','mtp_recurrence_tasks','mtp_recurrence_task')
    if name=='current':s=grow_loop(s,'projection_task < projection_tasks','projection_tasks','projection_task')
    ast.parse(s);f.write_text(s)
    f=out/K/'kda.py';s=f.read_text();s=s.replace('import torch\n','import os\nimport torch\n',1)
    s=once(s,'        self.mtp = mtp\n','        self.mtp = mtp\n        self.mtp_seq_len = int(os.environ.get("KIMI_MTP_SEQ_LEN", samples))\n        if mtp and (self.mtp_seq_len < 1 or samples % self.mtp_seq_len):\n            raise ValueError("invalid independent MTP chain dimensions")\n')
    s=once(s,'                mtp=mtp,\n            )\n        elif mtp:', '                mtp=mtp,\n                mtp_seq_len=self.mtp_seq_len,\n            )\n        elif mtp:')
    s=once(s,'            self.mtp,\n        )','            self.mtp,\n            mtp_seq_len=self.mtp_seq_len,\n        )')
    s=once(s,'expected_indices = (self.S + 1,) if self.mtp else (self.S,)',
           'expected_indices = (self.S + self.S // self.mtp_seq_len,) if self.mtp else (self.S,)')
    f.write_text(s)
    f=out/K/'op.py';s=f.read_text()
    s=s.replace('if samples != 4 or npes != 8 or not mtp:', 'if not 1 <= samples <= 32 or npes != 8 or not mtp:').replace('requires TP8, S4, true MTP','requires TP8, 1..32 tokens, true MTP')
    f.write_text(s)
    for fn in ['attn_res.py','router.py','tail.py','mxfp8_linear.py']:
        f=out/K/fn;s=f.read_text()
        s=s.replace('if samples not in {1, 2, 4, 8}:','if not 1 <= samples <= 32:').replace('if rows not in {1, 2, 4, 8}:','if not 1 <= rows <= 32:')
        s=s.replace('must be one of {{1, 2, 4, 8}}','must be in [1, 32]')
        if fn=='tail.py':
            start=s.index('def build_kimi_k3_tail(');end=s.index('\nclass ',start)
            body=s[start:end]
            body=once(body,'    sample_group = min(samples, _SAMPLES_PER_CTA)',
                '    sample_group = max(n for n in range(1, min(samples, _SAMPLES_PER_CTA) + 1) if samples % n == 0)')
            body=body.replace('_BLOCKS','tail_blocks')
            i=body.index('    task_rounds =')
            body=body[:i]+'    tail_blocks = min(_BLOCKS, 256 - samples * (((routed_hidden // 2) + _THREADS - 1) // _THREADS))\n'+body[i:]
            s=s[:start]+body+s[end:]
        f.write_text(s)
    f=out/'kernels/monokernel/reference.py';s=f.read_text();s=s.replace('import torch\n','import os\nimport torch\n',1)
    s=once(s,'    expected_indices = hidden_states.shape[0] + int(mtp)',
        '    seq_len = int(os.environ.get("KIMI_MTP_SEQ_LEN", hidden_states.shape[0]))\n    expected_indices = hidden_states.shape[0] + (hidden_states.shape[0] // seq_len if mtp else 0)')
    s=once(s,'        input_slot = int(state_indices[sample])\n        output_slot = int(state_indices[sample + 1]) if mtp else input_slot',
        '        chain_index = sample + sample // seq_len if mtp else sample\n        input_slot = int(state_indices[chain_index])\n        output_slot = int(state_indices[chain_index + 1]) if mtp else input_slot')
    f.write_text(s)
    f=out/K/'tools/monokernel.py';s=f.read_text()
    s=s.replace('choices=(1, 2, 4, 8), default=1','choices=range(1, 33), default=1',1)
    s=once(s,'    args = parser.parse_args()','    parser.add_argument("--batch", type=int, choices=(1,2,4,8))\n    parser.add_argument("--seq", type=int, choices=(1,2,3,4))\n    args = parser.parse_args()\n    args.batch = args.batch or 1\n    args.seq = args.seq or args.samples\n    if args.samples != args.batch * args.seq:\n        parser.error("samples must equal batch * seq")\n    os.environ["KIMI_MTP_SEQ_LEN"] = str(args.seq)')
    s=s.replace('args.mtp and args.samples == 4 and not args.staged','args.mtp and not args.staged')
    s=once(s,'    slots = args.samples + 3\n    state_count = args.samples + 1 if args.mtp else args.samples',
        '    slot_stride = 7 if args.full_replay_check else args.seq + 3\n    slots = args.batch * slot_stride\n    state_count = args.batch * (args.seq + 1) if args.mtp else args.samples')
    s=once(s,'    state_indices = torch.arange(state_count, device=device, dtype=torch.int32)',
        '    state_indices = torch.tensor([b * slot_stride + t for b in range(args.batch) for t in range(args.seq + 1)], device=device, dtype=torch.int32) if args.mtp else torch.arange(state_count, device=device,dtype=torch.int32)')
    s=once(s,'        "mtp": args.mtp,','        "mtp": args.mtp,\n        "batch": args.batch,\n        "seq": args.seq,')
    s=once(s,'            "samples": args.samples,','            "samples": args.samples,\n            "batch": args.batch,\n            "seq": args.seq,')
    f.write_text(s)

for n in ['run_jobs.py','check_occupancy.py']:
    shutil.copyfile(OLD/n,ROOT/n)
s=(OLD/'full_replay.py').read_text()
s=s.replace(' and args.samples == 4','')
s=once(s,'    layout = monokernel_layout(4,','    n = args.samples\n    layout = monokernel_layout(n,')
start=s.index('    chains = (');end=s.index('    previous_blocks',start)
s=s[:start]+'''    patterns = [list(range(args.seq + 1)), [5, 2, 6, 1, 4][:args.seq + 1],
                [0 if t == 0 else (-1 if t == min(2,args.seq) else t) for t in range(args.seq + 1)],
                [-1] * (args.seq + 1), [0] * (args.seq + 1), [t % 2 for t in range(args.seq + 1)]]
    chains = [[b * 7 + slot if slot >= 0 else -1 for b in range(args.batch) for slot in pattern] for pattern in patterns]
'''+s[end:]
s=once(s,'        destinations = {chain[i + 1] for i in range(4) if chain[i] >= 0 and chain[i + 1] >= 0}',
    '        token_indices = [b * (args.seq + 1) + t for b in range(args.batch) for t in range(args.seq)]\n        destinations = {chain[i + 1] for i in token_indices if chain[i] >= 0 and chain[i + 1] >= 0}')
s=s.replace("4 * config.","n * config.").replace('.view(4,','.view(n,').replace("tagged('routed_inv', 4)","tagged('routed_inv', n)")
ast.parse(s);(ROOT/'full_replay.py').write_text(s)
manifest={};changes=[]
for name,parent in SOURCES.items():
    for f in (ROOT/name/'kernels').rglob('*.py'):
        ast.parse(f.read_text());rel=f.relative_to(ROOT/name);manifest[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        p=OLD/parent/rel
        if p.read_bytes()!=f.read_bytes():
            changes.append(dict(source=name,file=str(rel)))
            with (ROOT/(name+'.patch')).open('a') as patch:
                patch.write(''.join(difflib.unified_diff(p.read_text().splitlines(True),f.read_text().splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel))))
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'source_audit.json').write_text(json.dumps(dict(changes=changes,batch_semantics='B disjoint L-token chains; B*(L+1) slot indices',flat_tokens='B*L',original_sources_preserved=True),indent=2)+'\n')
print(json.dumps(dict(sources=SOURCES,files=len(manifest),changes=changes),indent=2))
