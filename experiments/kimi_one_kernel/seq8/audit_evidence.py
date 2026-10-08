"""Independently audit frozen seq8 sources, binaries, replay and event records."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

LIMITS = dict(pre_attn_rel_l2=2e-3, attention_rel_l2=2e-3, moe_input_rel_l2=2e-3,
              updated_prefix_rel_l2=2e-3, conv_snapshot_max_rel_l2=5e-4,
              recurrent_snapshot_max_rel_l2=5e-4, selection_weight_rel_l2=2e-3,
              expert_mid_rel_l2=2e-3, routed_rel_l2=2e-2, routed_inv_rel_l2=1e-3,
              output_rel_l2=1e-2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--source', default='seq8_r6')
    parser.add_argument('--opt254-shapes', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root, source = args.evidence.resolve(), args.source
    manifest = json.loads((root / (source + '_manifest.json')).read_text())
    for rel, digest in manifest.items():
        assert sha(root / source / rel) == digest, rel
    sys.path.insert(0, str(root / source))
    from kernels.kimi_k3_monokernel.compile_config import KimiK3CompileConfig
    old = {(r['batch'], r['seq']): r for r in json.loads(args.opt254_shapes.read_text())}
    compiled, replay_rows, benchmarks = [], [], []
    maxima, diagnostics = {key: 0.0 for key in LIMITS}, {}
    records = ranks = 0
    evidence_hashes = {}

    def read(name):
        path = root / name
        evidence_hashes[name] = sha(path)
        return json.loads(path.read_text())

    def successful(name):
        path = root / (name + '.rc')
        evidence_hashes[path.name] = sha(path)
        assert path.read_text().strip() == '0', name
        cmd = read(name + '.command.json')
        assert cmd['source'] == source, name
        for relative, digest in cmd['harness_manifest'].items():
            assert sha(root / relative) == digest, (name, relative)
        return cmd

    for seq in range(1, 9):
        for batch in range(1, 9):
            key = f'{source}_b{batch}_s{seq}_auto'
            successful('compile_' + key)
            d = read('resources_' + key + '/compile.json')
            assert d['source_manifest'].items() <= manifest.items()
            assert len(d['kernels']) == 1
            k = d['kernels'][0]
            binary = root / ('resources_' + key) / Path(k['hsaco']).name
            assert sha(binary) == k['sha256']
            evidence_hashes[str(binary.relative_to(root))] = sha(binary)
            assert k['private_segment_fixed_size'] == k['vgpr_spill_count'] == 0
            resolved = KimiK3CompileConfig().resolve(batch * seq, seq > 1, seq)
            assert asdict(resolved) == d['specialization']
            gate = read('resources_' + key + '/occupancy.json')
            assert gate['passed'] and gate['hip_occupancy_checked']
            assert gate['source_manifest'] == d['source_manifest']
            assert gate['kernels'] == d['kernels'] and len(gate['devices']) == 8
            for device in gate['devices']:
                assert device['pass_residency']
                assert device['compute_units'] * device['resident_blocks_per_cu'] >= d['grid_blocks']
            row = dict(batch=batch, seq=seq, specialization=d['specialization'],
                       grid_blocks=d['grid_blocks'], sha256=k['sha256'],
                       vgpr_count=k['vgpr_count'], group_segment_fixed_size=k['group_segment_fixed_size'],
                       private_segment_fixed_size=k['private_segment_fixed_size'], vgpr_spill_count=k['vgpr_spill_count'],
                       devices=gate['devices'])
            if seq <= 4:
                row['unchanged_opt254_binary'] = k['sha256'] == old[batch, seq]['kernels'][0]['sha256']
                assert row['unchanged_opt254_binary'], (batch, seq)
            compiled.append(row)

    for seed in [1234, 2025, 3141]:
        shapes = [(b, s) for s in [8, 5, 6, 7] for b in range(1, 9)]
        if seed == 1234:
            shapes += [(1, 1), (1, 4), (2, 4), (8, 4)]
        for batch, seq in shapes:
            name = f'check_{source}_b{batch}_s{seq}_auto_seed{seed}'
            command = successful(name)
            assert '--full-replay-check' in command['cmd'] and '--check' in command['cmd']
            assert command['gpu']
            assert command['cmd'][command['cmd'].index('--seed') + 1] == str(seed)
            d = read(name + '.json')
            assert d['batch'] == batch and d['seq'] == seq and d['npes'] == 8
            assert d['launch_mode'] == 'single' and not d['instrumented'] and d['layer_idx'] == 1
            assert len(d['all_ranks']) == 8
            count = 0
            for rank in d['all_ranks']:
                assert rank['full_replay_correct'] and rank['finite'] and rank['rank_equal']
                assert rank['full_replay_limits'] == LIMITS
                assert len(rank['full_replay_records']) == (18 if seq > 1 else 15)
                for record in rank['full_replay_records']:
                    assert len(record['chain']) == batch * (seq + 1 if seq > 1 else 1)
                    for key, limit in LIMITS.items():
                        assert math.isfinite(record[key]) and record[key] < limit, (name, key, record[key])
                        maxima[key] = max(maxima[key], record[key])
                    assert all(value for value in record.values() if isinstance(value, bool))
                    for key in ['e2e_attention_rel_l2', 'e2e_recurrent_snapshot_max_rel_l2']:
                        assert math.isfinite(record[key])
                        diagnostics[key] = max(diagnostics.get(key, 0), record[key])
                count += len(rank['full_replay_records'])
            records += count
            ranks += 8
            replay_rows.append(dict(batch=batch, seq=seq, seed=seed, job=name, rank_checks=8, records=count, passed=True))

    for seed in [1234, 2025, 3141]:
        name = f'bench_{source}_b1_s8_auto_seed{seed}'
        command = successful(name)
        assert '--bench' in command['cmd'] and '--check' in command['cmd']
        assert command['gpu']
        assert command['cmd'][command['cmd'].index('--seed') + 1] == str(seed)
        d = read(name + '.json')
        assert d['batch'] == 1 and d['seq'] == 8 and d['npes'] == 8
        assert d['benchmark_scope'] == 'layer' and d['launch_mode'] == 'single' and not d['instrumented']
        assert d['layers'] == 16 and d['repeats'] == 50 and len(d['all_ranks']) == 8
        values = d['critical_times_us']
        assert len(values) == 50 and all(math.isfinite(v) and v > 0 for v in values)
        assert statistics.median(values) == d['median_us']
        assert all(r['critical_times_us'] == values and r['finite'] and r['rank_equal'] for r in d['all_ranks'])
        benchmarks.append(dict(seed=seed, job=name, median_us=d['median_us'], min_us=min(values), max_us=max(values)))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'compiled_shapes.json').write_text(json.dumps(compiled, indent=2) + '\n')
    new_rows = [r for r in replay_rows if r['seq'] > 4]
    summary = dict(passed=True, source=source, source_files=len(manifest), compiled_shapes=len(compiled),
                   unchanged_opt254_binaries=32, all_eight_device_residency=True, zero_private_and_vgpr_spill=True,
                   new_shape_cases=len(new_rows), new_rank_checks=sum(r['rank_checks'] for r in new_rows),
                   new_records=sum(r['records'] for r in new_rows), old_shape_smoke_cases=4,
                   total_rank_checks=ranks, total_records=records, replay_limits=LIMITS, replay_maxima=maxima,
                   ungated_e2e_diagnostic_maxima=diagnostics, replay_jobs=replay_rows,
                   b1_s8_benchmarks=benchmarks, b1_s8_median_us=statistics.median(r['median_us'] for r in benchmarks),
                   event_protocol='TP8 layer 1; 16-layer graph, 50 repeats; slowest rank per repeat; median of 3 seed medians',
                   evidence_hashes=evidence_hashes)
    (args.out / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['evidence_hashes', 'replay_jobs']}, indent=2))


if __name__ == '__main__':
    main()
