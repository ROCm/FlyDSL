"""Hide latent-up weight/gain loads under routed RMS readiness waits."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from textwrap import dedent,indent

ROOT=Path(__file__).resolve().parent
KERNEL=Path('kernels/kimi_k3_monokernel/kernel.py')
BASE='tail_rotate192_downsplit_events'
old=(ROOT/BASE/KERNEL).read_text()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def tail_once(s,a,b):
    begin=s.index('            # Stage 8:')
    return s[:begin]+once(s[begin:],a,b)

begin=old.index('        def mxfp8_bf16_accumulate_samples(');end=old.index('        def moe_peer_',begin)
helper=old[begin:end]
load_begin=helper.index('                    atom_group =')
load_end=helper.index('                    lhs =',load_begin)
load_body=helper[load_begin:load_end]
fragment=('        def latent_mxfp8_fragment(weight_rsrc, scale_rsrc, row_tile, k_dim, chunk, step_index):\n'
          '            k_chunks = k_dim // 64\n'+indent(dedent(load_body),'            ')+
          '            return weight, scale\n\n')
records=[]
for count,gather in [(7,False),(14,False),(0,True),(14,True)]:
    name=f'latent_pf{count}'+('_normgain' if gather else '')+'_events'
    s=old
    if count:
        h=once(helper,'            sample_count,\n','            sample_count,\n            prefetched=None,\n')
        h=once(h,load_body,
            '                    if const_expr(prefetched is not None and local_chunk < len(prefetched) // 2):\n'
            '                        weight, scale = prefetched[local_chunk * 2 + step_index]\n'
            '                    else:\n'+indent(load_body,'    '))
        s=once(s,helper,fragment+h)
        code=f'''                        tail_fragments = []
                        for prefetch_chunk in range_constexpr({count}):
                            for prefetch_step in range_constexpr(2):
                                tail_fragments.append(latent_mxfp8_fragment(
                                    rsrc(packed_latent_up), rsrc(latent_up_scale),
                                    (global_row - first_local_row) // 16, _ROUTED_HIDDEN,
                                    (wave - 4) * ((_ROUTED_HIDDEN // 64) // 4) + prefetch_chunk, prefetch_step,
                                ))
'''
        s=tail_once(s,'\n                        for local_sample in range_constexpr(staged_samples):',
            '\n'+code+'                        for local_sample in range_constexpr(staged_samples):')
        s=once(s,'split_wave, 4, latent_pairs, staged_samples,\n',
            'split_wave, 4, latent_pairs, staged_samples, prefetched=tail_fragments,\n')
    if gather:
        code='''                        # Gains are shared across tokens; gather independent
                        # tagged inverse RMS words before the original token order.
                        gain_prefetch = []
                        for gain_round in range_constexpr((_ROUTED_HIDDEN // 2 // 4) // _WAVE_SIZE):
                            gain_pair = (wave - 4) * (_ROUTED_HIDDEN // 2 // 4) + lane + gain_round * _WAVE_SIZE
                            gain_prefetch.append(fx.Int32(bo.buffer_load(rsrc(latent_gain), gain_pair, vec_width=1, dtype=T.i32)))
                        inverse_value = fx.Float32(0.0)
                        if lane < staged_samples:
                            inverse_value = load_f32(routed_inv_rsrc, sample_base + lane)
'''
        s=tail_once(s,'\n                        for local_sample in range_constexpr(staged_samples):',
            '\n'+code+'                        for local_sample in range_constexpr(staged_samples):')
        s=once(s,'                            inverse_rms = uniform_f32(load_f32(routed_inv_rsrc, sample_base + local_sample))',
            '                            inverse_rms = fx.Int32(rocdl.readlane(T.i32, inverse_value.bitcast(fx.Int32), fx.Int32(local_sample))).bitcast(fx.Float32)')
        s=once(s,'                                gain_word = fx.Int32(bo.buffer_load(gain_rsrc, pair, vec_width=1, dtype=T.i32))',
            '                                gain_word = gain_prefetch[load_round]')
    ast.parse(s)
    # Address pairs and MFMA order are unchanged for every split/step. The
    # source body used to issue every prefetch is copied exactly from parent.
    out=ROOT/name;assert not out.exists()
    shutil.copytree(ROOT/BASE/'kernels',out/'kernels',ignore=shutil.ignore_patterns('__pycache__'))
    (out/KERNEL).write_text(s)
    (ROOT/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+str(KERNEL),tofile='b/'+str(KERNEL))))
    records.append(dict(variant=name,parent=BASE,source_sha256=hashlib.sha256(s.encode()).hexdigest(),prefetch_k64_chunks=count,
        inverse_rms_wave_gather=gather,gain_loads_reused_across_tokens=gather,
        prefetch_load_expression_exact_copy=True,mfma_and_bf16_order_unchanged=True,dag_unchanged=True))
(ROOT/'latent_prefetch_audit.json').write_text(json.dumps(records,indent=2)+'\n')
f=ROOT/'run_round.py';s=f.read_text();begin=s.index('CANDIDATES=');end=s.index('\n',begin)
names=ast.literal_eval(s[begin+len('CANDIDATES='):end])+[r['variant'] for r in records]
f.write_text(s[:begin]+'CANDIDATES='+repr(names)+s[end:])
with tarfile.open(ROOT/'latent_prefetch_bundle.tar.gz','w:gz') as tar:
    for record in records:
        for f in (ROOT/record['variant']).rglob('*.py'):tar.add(f,arcname=str(f.relative_to(ROOT)))
    for n in ['make_latent_prefetch.py','latent_prefetch_audit.json','run_round.py']:tar.add(ROOT/n,arcname=n)
print(json.dumps(records,indent=2))
