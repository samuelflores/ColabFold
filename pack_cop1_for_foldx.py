#!/usr/bin/env python3
"""Bundle the 400 COP1/PIRH2 AF3 models without rerunning inference.

Run with system python3 on Arrhenius; no third-party packages required.
Preserves original mmCIF structures and phosphorylation input metadata.
This prepares inputs only: no FoldX energies are calculated here.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

RUNS = [
    ('test-4463024', 'dog9A', 49),
    ('canine11a-4463521', 'dog11A', 51),
    ('canine13a-4463952', 'dog13A', 53),
    ('human-4463953', 'human', 49),
]


def collect(root):
    manifest, copies = [], []
    for run, construct, distal in RUNS:
        results = root / run / 'results'
        wanted = {'none', 'S28', 'S39', f'S{distal}', f'S39+S{distal}'}
        completions = list(results.glob('*/BATCH_INFERENCE_COMPLETE.json'))
        found = {}
        comparisons = list(results.glob('comparison_*/comparison_confidence.csv'))
        if len(comparisons) != 1:
            raise ValueError(f'{run}: expected one confidence CSV, found {len(comparisons)}')
        with comparisons[0].open(newline='') as f:
            confidence = list(csv.DictReader(f))
        by_key = {(r['condition'], int(r['seed']), int(r['sample'])): r for r in confidence}
        expected_keys = {(c, s, 0) for c in wanted for s in range(1, 21)}
        if len(confidence) != 100 or set(by_key) != expected_keys:
            raise ValueError(f'{run}: confidence CSV does not describe exactly 100 expected models')
        for marker in completions:
            data = json.loads(marker.read_text())
            condition = data['condition']
            if condition not in wanted or condition in found:
                raise ValueError(f'{run}: unexpected or duplicate condition {condition!r}')
            found[condition] = marker
            inputs = data['input']
            proteins = {p['protein']['id']: p['protein'] for p in inputs['sequences'] if 'protein' in p}
            if set(proteins) != {'A', 'B'}:
                raise ValueError(f'{marker}: expected protein chains A and B')
            # These predictions have no other specified ligand/nucleic-acid chains.
            if len(inputs['sequences']) != 2:
                raise ValueError(f'{marker}: unexpected extra molecular entity')
            expected_sep = [] if condition == 'none' else sorted(int(s[1:]) for s in condition.split('+'))
            modifications = proteins['B'].get('modifications', [])
            actual_sep = sorted(m['ptmPosition'] for m in modifications if m['ptmType'] == 'SEP')
            if actual_sep != expected_sep or len(modifications) != len(expected_sep) or proteins['A'].get('modifications', []):
                raise ValueError(f'{marker}: modification metadata disagrees with condition')
            models = sorted(marker.parent.glob('seed-*_sample-*/*_model.cif'))
            identities = set()
            for model in models:
                match = re.fullmatch(r'seed-(\d+)_sample-(\d+)', model.parent.name)
                seed, sample = map(int, match.groups())
                if (seed, sample) in identities:
                    raise ValueError(f'{marker}: duplicate seed/sample')
                identities.add((seed, sample))
                key = (condition, seed, sample)
                if key not in by_key:
                    raise ValueError(f'{model}: model is absent from confidence CSV')
                label = condition.replace('+', '_')
                model_id = f'{construct}_{label}_seed{seed:02d}_sample{sample}'
                target = f'models/{model_id}.cif'
                row = dict(model_id=model_id, construct=construct, condition=condition,
                           seed=seed, sample=sample, cif=target,
                           expected_SEP_B=';'.join(map(str, expected_sep)),
                           chain_A_length=len(proteins['A']['sequence']),
                           chain_B_length=len(proteins['B']['sequence']),
                           cif_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                           ipTM=by_key[key]['ipTM'], pTM=by_key[key]['pTM'],
                           source_run=run, source_cif=str(model))
                manifest.append(row)
                copies.append((model, target))
            if len(models) != 20 or identities != {(s, 0) for s in range(1, 21)}:
                raise ValueError(f'{marker}: expected seeds 1–20 and sample 0 exactly once each')
            copies.append((marker, f'provenance/{construct}/{label}/BATCH_INFERENCE_COMPLETE.json'))
        if set(found) != wanted:
            raise ValueError(f'{run}: missing conditions {wanted-set(found)}')
        for filename in ('comparison_confidence.csv', 'comparison_condition_summary.csv',
                         'comparison_contact_summary.csv'):
            p = comparisons[0].parent / filename
            if not p.is_file():
                raise FileNotFoundError(p)
            copies.append((p, f'provenance/{construct}/{filename}'))
    if len(manifest) != 400 or len({r['model_id'] for r in manifest}) != 400:
        raise ValueError('Expected 400 uniquely labelled structures')
    return sorted(manifest, key=lambda r: r['model_id']), copies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/nobackup/proj/disk/naiss2026-4-1221/personal/samuelf/af3_batch'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or args.root / 'cop1_foldx_400.tar.gz').resolve()
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose another --output')
    manifest, copies = collect(args.root.resolve())
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
    notes = ('400 original AlphaFold3 mmCIF structures: four constructs, five conditions, 20 seeds.\n'
             'Manifest identifies each model, original source, checksum, and expected chain-B SEP positions.\n'
             'No structures have been repaired, mutated, stripped, or energetically scored.\n'
             'For FoldX: convert to PDB preserving SEP and atoms; verify chain/residue/atom identity.\n'
             'Test RepairPDB and AnalyseComplex for one none and one phosphorylated model first.\n'
             'Check SEP survives repair, then apply identical settings to all 400 models.\n'
             'Score PIRH2 chain A versus COP1 chain B using analyseComplexChains=A,B.\n'
             'Retain energy components, especially electrostatics, polar/hydrophobic solvation and clashes.\n'
             'These scores are model-based interaction energies, not measured binding affinities.\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.cop1-foldx-', suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
    try:
        with tarfile.open(temporary, 'w:gz') as tar:
            for source, destination in copies:
                tar.add(source, arcname='cop1_foldx_400/' + destination, recursive=False)
            for name, text in [('manifest.csv', buffer.getvalue()), ('README.txt', notes)]:
                payload = text.encode()
                entry = tarfile.TarInfo('cop1_foldx_400/' + name)
                entry.size = len(payload)
                entry.mode = 0o644
                tar.addfile(entry, io.BytesIO(payload))
        with tarfile.open(temporary, 'r:gz') as tar:
            members = tar.getnames()
            assert sum(n.endswith('.cif') for n in members) == 400
            for row in manifest:
                member = tar.extractfile('cop1_foldx_400/' + row['cif'])
                assert hashlib.sha256(member.read()).hexdigest() == row['cif_sha256']
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    for construct, count in Counter(r['construct'] for r in manifest).items():
        print(f'{construct}: {count} models')
    print(f'Archive: {output}\nSize: {output.stat().st_size / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
