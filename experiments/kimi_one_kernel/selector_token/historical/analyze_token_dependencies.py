"""Track which UG tokens precede each down task in the audited TP8 program DAG.

This is a conservative source dependency analysis; it assigns no durations and
does not predict GPU scheduling, measured overlap, or speedup.
"""
from collections import Counter, deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'kimi_full_token_pipeline_20260929'
sys.path.insert(0, str(OLD))
from audit_dependencies import DAG

class CaptureDAG(DAG):
    latest = None
    def __init__(self):
        super().__init__()
        CaptureDAG.latest = self

def load_audit(grouped):
    if grouped:
        outer = (ROOT / 'audit_round.py').read_text().split('\nrecords=[]')[0]
        setup = {'__file__': str(ROOT / 'audit_round.py')}
        exec(compile(outer, 'grouped_audit_setup', 'exec'), setup)
        ns = setup['ns']
    else:
        source = (OLD / 'audit_goal_schedule.py').read_text().split('\nvariants=')[0]
        ns = {'__file__': str(ROOT / 'analyze_token_dependencies.py')}
        exec(compile(source, 'baseline_audit_setup', 'exec'), ns)
    ns['DAG'] = CaptureDAG
    return ns['audit']

def analyze(name, grouped, pair):
    audit = load_audit(grouped)
    result = audit(name, service_shift=True, ug_pair=pair, projection_split4=True)
    d = CaptureDAG.latest
    degree = [len(v) for v in d.inc]
    ready = deque(i for i, n in enumerate(degree) if n == 0)
    masks = [0] * len(d.names)
    while ready:
        i = ready.popleft()
        n = d.names[i]
        if n[0] == 'up':
            masks[i] |= 1 << n[2]
        for j in d.out[i]:
            masks[j] |= masks[i]
            degree[j] -= 1
            if degree[j] == 0:
                ready.append(j)
    assert not any(degree)
    def tokens(mask):
        return ','.join(str(s) for s in range(4) if mask & (1 << s))
    result['down_ug_token_ancestors_per_rank'] = [dict(Counter(
        tokens(masks[i]) for i, n in enumerate(d.names)
        if n[0] == 'down_push' and n[1:3] == (0, sample))) for sample in range(4)]
    result['rms_ug_token_ancestors_rank0'] = [tokens(masks[d.ids['routed_norm', 0, s]]) for s in range(4)]
    result['tail_ug_token_ancestors_per_rank'] = dict(Counter(
        tokens(masks[i]) for i, n in enumerate(d.names) if n[0] == 'tail_push' and n[1] == 0))
    return result

rows = [analyze('full_tail4_proj4_ug32_events', False, True),
        analyze('moe_g2_ug16_events', True, False),
        analyze('moe_g2_ug32_events', True, True)]
out = dict(cpu_only=True, variants=rows,
    limitation='A token in an ancestor mask means some UG work for that token precedes the task; it does not mean all its UG tasks must finish. No timing estimates.')
(ROOT / 'token_dependency_analysis.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
