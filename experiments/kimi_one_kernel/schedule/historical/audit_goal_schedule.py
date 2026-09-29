"""CPU-only dependency and ownership audit for the retained full kernel.

Readiness fan-ins deliberately overapproximate dependencies. Acyclicity here
does not prove compiled memory ordering or predict GPU latency.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from audit_dependencies import DAG

ROOT = Path(__file__).resolve().parent


def audit(name, service_shift=False, ug_pair=False, projection_split4=False):
    path = ROOT / name / 'kernels/kimi_k3_monokernel/kernel.py'
    code = path.read_text()
    assert 'mtp_recurrence_task = (bid + 128) % _BLOCKS' in code
    assert 'tail_tasks = sample_groups * hidden_tiles' in code
    if ug_pair:
        assert 'up_task % paired_up_tiles) * 2' in code
    if projection_split4:
        assert 'latent_projection_tiles = 2' in code and 'shared_projection_tiles = 2' in code
        assert 'mxfp8_scaled_mfma_split4' in code
    if service_shift:
        for var, shift in [('post_owner_task',32),('selector_owner_task',16),
                           ('shared_owner_task',15),('norm_owner_task',11)]:
            assert f'{var} = (bid + {shift}) % _BLOCKS' in code
    d = DAG()
    coverage = Counter()
    up_per_route = 12 if ug_pair else 24
    projection_tasks = 216 if projection_split4 else 163
    latent_end = 168 if projection_split4 else 131
    for rank in range(8):
        for cta in range(256):
            def event(kind, *indices, waits=(), publishes=()):
                return d.event(rank, cta, (kind, rank, *indices), waits, publishes)
            if cta < 16:
                sample, chunk = divmod(cta, 4)
                event('pre_stats', sample, chunk, publishes=[('pre_stats_ready',rank,sample)])
                event('pre', sample, chunk, waits=[('pre_stats_ready',rank,sample)],
                      publishes=[('pre_ready',rank)])
            if cta < 200:
                event('input',cta,waits=[('pre_ready',rank)],publishes=[('input_ready',rank)])
                coverage['input',rank,cta] += 1
            if cta < 48:
                sample, head = divmod(cta,12)
                waits=[('input_ready',rank)]
                if sample: waits.append(('conv_ready',rank,sample-1,head))
                event('conv',sample,head,waits=waits,publishes=[('conv_ready',rank,sample,head)])
                coverage['conv',rank,sample,head] += 1
            recurrence=(cta+128)%256
            if recurrence < 96:
                sample, head_split=divmod(recurrence,24)
                head, split=divmod(head_split,2)
                waits=[('conv_ready',rank,sample,head)]
                if sample: waits.append(('state_ready',rank,sample-1,head))
                event('state',sample,head,split,waits=waits,
                      publishes=[('state_ready',rank,sample,head)])
                event('kda_norm',sample,head,split,waits=[('state_ready',rank,sample,head)],
                      publishes=[('norm_ready',rank)])
                coverage['recurrence',rank,sample,head,split] += 1
            if cta < 112:
                event('output_push',cta,waits=[('norm_ready',rank)],publishes=[('output_tp',cta)])
                event('output',cta,waits=[('output_tp',cta)],publishes=[('attention_ready',rank)])
                coverage['output',rank,cta] += 1
            post=(cta+32)%256 if service_shift else cta
            if post < 16:
                sample, chunk=divmod(post,4)
                event('post_stats',sample,chunk,waits=[('attention_ready',rank)],
                      publishes=[('post_stats_ready',rank,sample)])
                event('post',sample,chunk,waits=[('post_stats_ready',rank,sample)],
                      publishes=[('moe_ready',rank)])
                coverage['post',rank,sample,chunk] += 1
            if cta < projection_tasks:
                kind='router' if cta<56 else 'latent' if cta<latent_end else 'shared_gu'
                event(kind,cta,waits=[('moe_ready',rank)],publishes=[(kind+'_ready',rank)])
                coverage['projection',rank,cta] += 1
            selector=(cta+16)%256 if service_shift else cta
            if selector==0:
                event('select',waits=[('router_ready',rank)],publishes=[('selection_ready',rank)])
                coverage['select',rank] += 1
            shared=(cta+15)%256 if service_shift else cta
            if shared<4:
                event('shared_mid',shared,waits=[('shared_gu_ready',rank)],
                      publishes=[('shared_mid_ready',rank)])
                coverage['shared_mid',rank,shared] += 1
            for task in range(cta,4*16*up_per_route,256):
                sample=task//(16*up_per_route)
                route=(task//up_per_route)%16
                tile=task%up_per_route
                event('up',sample,route,tile,waits=[('selection_ready',rank),('latent_ready',rank)],
                      publishes=[('up_ready',rank,sample)])
                for i in range(2 if ug_pair else 1):
                    coverage['up',rank,sample,route,tile*(2 if ug_pair else 1)+i] += 1
            for task in range(cta,4*224,256):
                sample,tile=divmod(task,224)
                event('down_push',sample,tile,waits=[('up_ready',rank,sample)],
                      publishes=[('down_tp',sample,tile)])
                event('down',sample,tile,waits=[('down_tp',sample,tile)],
                      publishes=[('down_ready',rank,sample)])
                coverage['down',rank,sample,tile] += 1
            norm=(cta+11)%256 if service_shift else cta
            if norm<4:
                event('routed_norm',norm,waits=[('down_ready',rank,norm)],
                      publishes=[('routed_inv',rank,norm)])
                coverage['routed_norm',rank,norm] += 1
            for tile in range(cta,448,256):
                waits=[('shared_mid_ready',rank)]
                if rank*56 <= tile < (rank+1)*56:
                    waits += [('routed_inv',rank,sample) for sample in range(4)]
                event('tail_push',tile,waits=waits,publishes=[('tail_tp',tile)])
                event('tail',tile,waits=[('tail_tp',tile)],publishes=[('final_ready',rank)])
                coverage['tail',rank,tile] += 1
    expected=dict(input=200*8,conv=48*8,recurrence=96*8,output=112*8,
                  post=16*8,projection=projection_tasks*8,select=8,shared_mid=4*8,
                  up=1536*8,down=896*8,routed_norm=4*8,tail=448*8)
    counts=dict(Counter(k[0] for k in coverage))
    assert set(coverage.values())=={1} and counts==expected,(counts,expected)
    for i,n in enumerate(d.names):
        if 'ready' in n[0] or n[0] in ['output_tp','down_tp','tail_tp','routed_inv']:
            assert d.inc[i],('unproduced',n)
    d.check()
    return dict(variant=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                acyclic=True,task_coverage_once=True,covered_tasks=counts,
                nodes=len(d.names),edges=sum(map(len,d.out)))


variants=[audit('full_relocate_wave_tail4_events'),
          audit('full_tail4_services_events',service_shift=True),
          audit('full_tail4_ug32_events',ug_pair=True),
          audit('full_tail4_ug32seq_events',ug_pair=True),
          audit('full_tail4_services_proj4_events',service_shift=True,projection_split4=True)]
projection=dict(current=dict(latent_tiles=224,shared_tiles=96,latent_ctas=75,shared_ctas=32,
                            allocated_waves=107*8,valid_active_waves=320,executing_waves=321,k256_per_wave=28,
                            native_mfma_per_wave=56,projection_ctas_including_router=163),
                proposed_split4=dict(row_groups_per_cta=2,k_split_waves=4,
                            latent_ctas=112,shared_ctas=48,allocated_waves=160*8,
                            active_waves=320*4,k256_per_wave=7,native_mfma_per_wave=14,
                            projection_ctas_including_router=216,
                            changes_fp32_accumulation_grouping=True,
                            implemented=True,compiled=False,gpu_validated=False))
metrics=dict(ug_tasks={'current':1536,'paired':768},
             ug_activation_payload_logical_mib={'current':1536*3584*2/2**20,'paired':768*3584*2/2**20},
             norm_owner_prior_down_tasks={'current':[len(range(b,896,256)) for b in range(4)],
                                          'shifted':[len(range(b,896,256)) for b in range(245,249)]})
out=dict(scope='TP8/S4/layer1 CPU-only conservative schedule audit',variants=variants,
         projection_analysis=projection,metrics=metrics,
         limitations=['No GPU execution or performance inferred',
                      'Readiness fan-ins overapproximate dependencies; not a memory-ordering proof',
                      'Logical payload counts are not measured HBM traffic'])
(ROOT/'goal_schedule_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
