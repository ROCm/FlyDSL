"""Check source identities and restore archived patches without a GPU or FlyDSL."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
KERNEL = 'kernels/kimi_k3_monokernel/kernel.py'
HISTORICAL_REVISION = '8137256d25ccea61eeca4f43d26fc3aad280cb1d'
OPT254_REVISION = '9cb1d40e0c3c344007852be1d9aa92b5b69a4ce3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes(files):
    return {name: sha(data) for name, data in sorted(files.items())}


def restore(base, patch, changes):
    result = dict(base)
    with tempfile.TemporaryDirectory(prefix='kimi-archive-check-') as directory:
        root = Path(directory)
        for change in changes:
            name = change['path']
            if change['base_sha256'] is not None:
                assert sha(base[name]) == change['base_sha256'], name
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base[name])
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for change in changes:
            name = change['path']
            if change['target_sha256'] is None:
                assert not (root / name).exists(), name
                result.pop(name)
            else:
                data = (root / name).read_bytes()
                assert sha(data) == change['target_sha256'], (patch.name, name)
                ast.parse(data, filename=name)
                result[name] = data
    return result


def main():
    # The September patches retain their original base after promoting Opt254.
    opt254 = json.loads((ROOT / 'opt254/source_manifest.json').read_text())
    for name, digest in opt254['files'].items():
        data = subprocess.check_output(['git', 'show', f'{OPT254_REVISION}:{name}'], cwd=REPO)
        assert sha(data) == digest, name
        ast.parse(data, filename=name)
    for name, digest in opt254['supporting_files'].items():
        assert sha((ROOT / 'opt254' / name).read_bytes()) == digest, name
    current = json.loads((ROOT / 'seq8/source_manifest.json').read_text())
    for name, digest in current['files'].items():
        data = (REPO / name).read_bytes()
        assert sha(data) == digest, name
        ast.parse(data, filename=name)
    for name, digest in current['supporting_files'].items():
        assert sha((ROOT / 'seq8' / name).read_bytes()) == digest, name
    sys.path.insert(0, str(REPO))
    from dataclasses import asdict
    from kernels.kimi_k3_monokernel.compile_config import KimiK3CompileConfig

    shapes = json.loads((ROOT / 'opt254/compiled_shapes.json').read_text())
    assert len(shapes) == 32
    assert {(row['batch'], row['seq']) for row in shapes} == {
        (batch, seq) for batch in range(1, 9) for seq in range(1, 5)
    }
    for row in shapes:
        batch, seq = row['batch'], row['seq']
        resolved = KimiK3CompileConfig().resolve(batch * seq, seq > 1, seq)
        assert asdict(resolved) == row['specialization'], (batch, seq)
        assert resolved.grid_blocks == row['grid_blocks']

    extended_shapes = json.loads((ROOT / 'seq8/compiled_shapes.json').read_text())
    assert len(extended_shapes) == 64
    assert {(row['batch'], row['seq']) for row in extended_shapes} == {
        (batch, seq) for batch in range(1, 9) for seq in range(1, 9)
    }
    for row in extended_shapes:
        batch, seq = row['batch'], row['seq']
        resolved = KimiK3CompileConfig().resolve(batch * seq, seq > 1, seq)
        assert asdict(resolved) == row['specialization'], (batch, seq)
        assert row['private_segment_fixed_size'] == row['vgpr_spill_count'] == 0
        if seq <= 4:
            assert row['unchanged_opt254_binary'], (batch, seq)

    manifest = json.loads((ROOT / 'selected_source_manifest.json').read_text())
    selected = {
        name: subprocess.check_output(['git', 'show', f'{HISTORICAL_REVISION}:{name}'], cwd=REPO)
        for name in manifest['files']
    }
    assert hashes(selected) == manifest['files']
    for name, data in selected.items():
        ast.parse(data, filename=name)
    patch_count = 0
    staged = None
    for path in sorted(ROOT.glob('*/manifest.json')):
        archive = json.loads(path.read_text())
        assert archive['patch_base_kernel_sha256'] == sha(selected[KERNEL])
        for source in archive['sources']:
            restored = restore(selected, path.parent / (source['name'] + '.patch'), source['changed_files'])
            assert sha(restored[KERNEL]) == source['kernel_sha256']
            digest = sha(json.dumps(hashes(restored), sort_keys=True).encode())
            assert digest == source['source_tree_sha256'], source['name']
            if path.parent.name == 'baselines' and source['name'] == 'staged_relocate_events':
                staged = restored
            patch_count += 1
    assert staged is not None
    matrix = json.loads((ROOT / 'batch_seq_draft/status.json').read_text())
    assert matrix['gpu_jobs_started'] == matrix['compile_jobs_started'] == 0
    for source in matrix['patches']:
        base = selected if source['source'] == 'current' else staged
        assert sha(base[KERNEL]) == source['base_kernel_sha256']
        restored = restore(base, ROOT / 'batch_seq_draft' / (source['source'] + '.patch'), source['changed_files'])
        assert hashes(restored) == source['target_files']
        patch_count += 1
    print(json.dumps(dict(selected_source_files=len(current['files']), opt254_source_files=len(opt254['files']),
                          historical_source_files=len(selected), compiled_shape_configs_matched=len(extended_shapes),
                          unchanged_opt254_configs=len(shapes), independent_patches_restored=patch_count,
                          exact_source_hashes=True, python_ast_parse=True, gpu_execution=False), indent=2))


if __name__ == '__main__':
    main()
