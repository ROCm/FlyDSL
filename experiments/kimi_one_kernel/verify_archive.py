"""Check source identities and restore archived patches without a GPU or FlyDSL."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
KERNEL = 'kernels/kimi_k3_monokernel/kernel.py'


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
    manifest = json.loads((ROOT / 'selected_source_manifest.json').read_text())
    selected = {name: (REPO / name).read_bytes() for name in manifest['files']}
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
    print(json.dumps(dict(selected_source_files=len(selected), independent_patches_restored=patch_count,
                          exact_source_hashes=True, python_ast_parse=True, gpu_execution=False), indent=2))


if __name__ == '__main__':
    main()
