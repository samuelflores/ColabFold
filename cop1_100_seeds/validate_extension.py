#!/usr/bin/env python3
"""Validate frozen inputs and exact seed identities; optionally check the GPU."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import sys


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--construct', required=True, choices=['dog9A', 'dog11A', 'dog13A'])
    parser.add_argument('--gpu-check', action='store_true')
    parser.add_argument('--results', type=Path)
    parser.add_argument('--first', type=int)
    parser.add_argument('--last', type=int)
    args = parser.parse_args()
    spec = json.loads((args.bundle / 'extension_spec.json').read_text())[args.construct]
    source = (args.bundle / spec['driver']).read_bytes()
    check(hashlib.sha256(source).hexdigest() == spec['driver_sha256'], 'Driver changed after validation')
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('PROTEIN_INPUT', 'CONDITIONS'):
                    constants[target.id] = ast.literal_eval(node.value)
    check(constants['PROTEIN_INPUT'].split(':') == spec['sequences'], 'Protein sequence mismatch')
    check(constants['CONDITIONS'][spec['condition']] == spec['phosphorylation_positions'], 'Phosphorylation mismatch')
    report = dict(construct=args.construct, condition=spec['condition'],
                  chain_lengths=[len(s) for s in spec['sequences']],
                  source_validation='matches original 20-model ensemble sequences')
    if args.gpu_check:
        import importlib.metadata
        versions = {}
        for name in ('alphafold3', 'gemmi', 'matplotlib', 'numpy', 'jax'):
            module = importlib.import_module(name)
            versions[name] = dict(version=getattr(module, '__version__', 'unknown'),
                                  path=getattr(module, '__file__', 'unknown'))
        import jax
        check(platform.machine() == 'aarch64', 'Expected an Arrhenius ARM GPU node')
        check(any(d.platform == 'gpu' for d in jax.devices()), 'JAX has no GPU device')
        report.update(python=sys.executable, python_version=sys.version,
                      architecture=platform.machine(), modules=versions,
                      devices=[str(d) for d in jax.devices()])
        report['installed_distributions'] = sorted(
            '{}=={}'.format(d.metadata['Name'], d.version)
            for d in importlib.metadata.distributions() if d.metadata.get('Name'))
    if args.results:
        check(args.first is not None and args.last is not None, 'Specify first and last seed')
        wanted = list(range(args.first, args.last + 1))
        check(min(wanted) >= 21 and max(wanted) <= 100, 'Only new seeds 21..100 are allowed')
        plans = list(args.results.glob('plan_*.json'))
        check(len(plans) == 1, 'Expected exactly one inference plan')
        plan = json.loads(plans[0].read_text())
        check(len(plan) == 1, 'Expected only one double-phosphorylation condition')
        entry = plan[0]
        check(entry['condition'] == spec['condition'], 'Wrong condition in plan')
        proteins = [s['protein'] for s in entry['data']['sequences']]
        check([p['id'] for p in proteins] == ['A', 'B'], 'Unexpected chain IDs')
        check([p['sequence'] for p in proteins] == spec['sequences'], 'Wrong sequences in plan')
        check(not proteins[0].get('modifications'), 'Unexpected PIRH2 modification')
        check(proteins[1].get('modifications') == [dict(ptmType='SEP', ptmPosition=p)
              for p in spec['phosphorylation_positions']], 'Wrong COP1 modifications')
        check(entry['data']['modelSeeds'] == wanted, 'Wrong requested seed set')
        settings = entry['settings']
        for key, value in dict(num_diffusion_samples=1, num_recycles=10,
                               msa_mode='mmseqs2_server', weights='af3').items():
            check(settings[key] == value, 'Wrong inference setting: ' + key)
        folder = Path(entry['job_dir'])
        files = list(folder.glob('seed-*_sample-*/*_model.cif'))
        observed = []
        for path in files:
            match = re.fullmatch(r'seed-(\d+)_sample-(\d+)', path.parent.name)
            check(match is not None, 'Unrecognised seed/sample directory')
            observed.append(tuple(map(int, match.groups())))
        check(len(files) == len(wanted) and sorted(observed) == [(s, 0) for s in wanted],
              'Incomplete, duplicate or unexpected models')
        check((folder / 'BATCH_INFERENCE_COMPLETE.json').is_file(), 'Missing inference completion marker')
        check((folder / 'BATCH_ANALYSIS_COMPLETE.json').is_file(), 'Missing analysis completion marker')
        report.update(first_seed=args.first, last_seed=args.last, model_count=len(files),
                      results=str(folder), verified=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
