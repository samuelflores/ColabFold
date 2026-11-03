#!/usr/bin/env python3
"""Canine PIRH2 / canine 11A COP1: matched-seed phosphorylation comparisons.

Defaults: S51, S39, S39+S51, none, S28; each uses seeds 1..20, one
sample per seed, ten recycles, the MMseqs2 server, and official AF3 weights.
S labels mean phosphorylation (SEP) at that 1-based COP1/chain-B position;
none means no phosphorylation. Protein sequences do not vary.
Requires the existing prepared AlphaFold environment, runner, and weights.
No package installation, weight downloads, or installed-package patches occur.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

PROTEIN_INPUT = 'MAAAAREDGARGREQGLRGCEHYDRGCLLKAPCCDKLYTCRLCHDNNEDHQLDRFKVKEVQCINCEKIQHAQQSCEECSTLFGEYYCSICHLFDKDKKQYHCENCGICRIGPKEDFFHCLKCNLCLAMNLQGKHKCIENVSRQNCPICLEDIHTSRVVAHVLPCGHLLHRTCYEEMLKEGYRCPLCMHSALDMTRYWRQLDDEVAQTPMPSEYQNMTVDILCNDCNRRSTVQFHILGMKCNICESYNTAQAKRISLDQQ:MSGSRQAGSGSAGTSPGSSAASSVTSASSSLSSSPSPPSAAAAAAAAAAASGGAAQAAGSGGLGGPAARPVLVAAAVSGSGGGAVSAGQSRLSCAARPSAGVGGSSSGLGSGSRKRPLLA'
CONDITIONS = {"S51": [51], "S39": [39], "S39+S51": [39, 51], "none": [], "S28": [28]}
TARGETS = (28, 39, 51)


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def seed_values(text):
    if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", text.strip()):
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative integers")
    return list(dict.fromkeys(map(int, text.split(','))))


def make_input(condition, seeds, samples, recycles, msa_mode, weights, model_dir, basejob):
    pirh2, cop1 = PROTEIN_INPUT.split(':')
    if cop1[39:51] != 'AAAAAAAAAAAS' or cop1[18:27] != 'SAASSVTSA':
        raise ValueError('Unexpected canine 11A COP1 sequence')
    if any(cop1[pos - 1] != 'S' for pos in TARGETS):
        raise ValueError('Target positions must be serines in the unmodified sequence')
    proteins = [dict(id='A', sequence=pirh2, templates=[]),
                dict(id='B', sequence=cop1, templates=[])]
    positions = CONDITIONS[condition]
    if positions:
        proteins[1]['modifications'] = [dict(ptmType='SEP', ptmPosition=p) for p in positions]
    if msa_mode == 'single_sequence':
        for p in proteins:
            p.update(unpairedMsa=f">query\n{p['sequence']}\n", pairedMsa='')
    sequences = [{'protein': p} for p in proteins]
    settings = dict(sequences=sequences, modelSeeds=seeds, msa_mode=msa_mode,
                    num_recycles=recycles, num_diffusion_samples=samples,
                    weights=weights, model_dir=str(model_dir))
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    base = re.sub(r'\W+', '', basejob).lower() or 'canine11a'
    label = condition.lower().replace('+', '_')
    name = f'{base}_{label}_{digest}'
    data = dict(name=name, sequences=sequences, modelSeeds=seeds, dialect='alphafold3', version=1)
    return data, settings


def device_settings():
    flags = []
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15, check=True)
        cap = min(float(x) for x in result.stdout.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        print('No GPU detected; using CPU/XLA with --nojit.', flush=True)
        return 'xla', True, flags
    if cap < 8.0:
        flags = ['--xla_disable_hlo_passes=custom-kernel-fusion-rewriter']
        impl = 'xla'
    else:
        flags = ['--xla_gpu_enable_triton_gemm=false']
        impl = 'xla' if 8.0 < cap < 9.0 else 'triton'
    return impl, False, flags


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def model_files(folder):
    return sorted(folder.glob('seed-*_sample-*/*_model.cif'))


def run_condition(args, data, settings, condition, input_path, job_dir, runner, model_dir, device):
    expected = len(args.seeds) * args.num_diffusion_samples
    completion = job_dir / 'BATCH_INFERENCE_COMPLETE.json'
    ready = False
    if job_dir.exists() and not args.dry_run:
        if args.on_existing == 'skip' and completion.is_file():
            saved = json.loads(completion.read_text())
            ready = saved.get('settings') == settings and len(model_files(job_dir)) == expected
        if not ready and args.on_existing != 'overwrite':
            raise FileExistsError(f'{job_dir}: use a new --jobname, or explicitly choose --on-existing overwrite')
    impl, nojit, flags = device
    env = os.environ.copy()
    current = env.get('XLA_FLAGS', '')
    for flag in flags:
        if flag not in current:
            current = (current + ' ' + flag).strip()
    if current:
        env['XLA_FLAGS'] = current
    cmd = [sys.executable, str(runner), f'--json_path={input_path}',
           f'--model_dir={model_dir}', '--norun_data_pipeline',
           f'--output_dir={args.output_dir}', f'--cache_dir={args.cache_dir}',
           '--force_output_dir', f'--flash_attention_implementation={impl}',
           f'--num_recycles={args.num_recycles}',
           f'--num_diffusion_samples={args.num_diffusion_samples}']
    if args.msa_mode == 'mmseqs2_server':
        cmd.append('--use_msa_server')
    if args.weights == 'of3':
        cmd.append('--of3_weights')
    if nojit:
        cmd.append('--nojit')
    print(f'\n{condition}: {expected} models -> {job_dir}', flush=True)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not ready:
        if job_dir.exists():
            shutil.rmtree(job_dir)  # Only reached on explicit --on-existing overwrite.
        subprocess.run(cmd, env=env, check=True)
        # Validate seed/sample identities as well as total count.
        files = model_files(job_dir)
        observed = set()
        for f in files:
            match = re.fullmatch(r'seed-(\d+)_sample-(\d+)', f.parent.name)
            observed.add(tuple(map(int, match.groups())))
        wanted = {(s, sample) for s in args.seeds for sample in range(args.num_diffusion_samples)}
        if len(files) != expected or observed != wanted:
            raise RuntimeError(f'{condition}: model seed/sample set differs from request; not marked complete')
        write_json(completion, dict(condition=condition, input=data, settings=settings,
                                    command=cmd, model_count=len(files)))
    else:
        print(f'Reusing completed inference for {condition}', flush=True)
    if not args.skip_analysis:
        analyse_condition(job_dir, data, condition, args.contact_cutoff)
        write_json(job_dir / 'BATCH_ANALYSIS_COMPLETE.json', dict(condition=condition, cutoff_A=args.contact_cutoff))


def write_csv(path, rows):
    import csv
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, cutoff):
    import numpy as np
    groups = {}
    for row in rows:
        key = (row['target_B'], row['target_name'], row['probe'], row['residue_A'])
        groups.setdefault(key, []).append(row)
    result = []
    for (position, name, probe, residue), values in groups.items():
        distances = [r['distance_A'] for r in values]
        result.append(dict(target_B=position, target_name=name, probe=probe,
                           residue_A=residue, n_models=len(values),
                           within_cutoff=sum(d <= cutoff for d in distances),
                           fraction_within_cutoff=np.mean(np.array(distances) <= cutoff),
                           cutoff_A=cutoff, median_distance_A=np.median(distances),
                           min_distance_A=min(distances), max_distance_A=max(distances),
                           median_PAE_A_B=np.median([r['PAE_A_B'] for r in values]),
                           median_PAE_B_A=np.median([r['PAE_B_A'] for r in values])))
    return sorted(result, key=lambda r: (r['probe'], -r['fraction_within_cutoff'], r['median_distance_A']))


def analyse_condition(folder, data, condition, cutoff):
    import gemmi
    import numpy as np
    files = sorted(folder.glob('seed-*_sample-*/*_model.cif'))
    if not files:
        raise ValueError(f'No individual models found in {folder}')
    proteins = {}
    for entity in data['sequences']:
        if 'protein' in entity:
            p = entity['protein']
            for cid in ([p['id']] if isinstance(p['id'], str) else p['id']):
                proteins[cid] = p
    b = proteins['B']
    modified = {m['ptmPosition'] for m in b.get('modifications', []) if m['ptmType'] == 'SEP'}
    targets = set(TARGETS) | modified
    confidence, contacts = [], []
    for cif in files:
        name = cif.parent.name
        seed, sample = map(int, re.fullmatch(r'seed-(\d+)_sample-(\d+)', name).groups())
        conf_file, = [p for p in cif.parent.glob('*_confidences.json') if 'summary' not in p.name]
        summary_file, = cif.parent.glob('*_summary_confidences.json')
        conf, summary = json.loads(conf_file.read_text()), json.loads(summary_file.read_text())
        pae = np.asarray(conf['pae'])
        ids = list(zip(conf['token_chain_ids'], map(int, conf['token_res_ids'])))
        def indices(chain, number=None):
            return [i for i, (c, r) in enumerate(ids) if c == chain and (number is None or r == number)]
        aidx, bidx = indices('A'), indices('B')
        if not aidx or not bidx:
            raise ValueError(f'{name}: missing chain A/B PAE tokens')
        confidence.append(dict(condition=condition, model=name, seed=seed, sample=sample,
                               ipTM=summary.get('iptm'), pTM=summary.get('ptm'),
                               ranking_score=summary.get('ranking_score'),
                               mean_pLDDT=np.mean(conf.get('atom_plddts', conf.get('token_plddts', []))),
                               mean_interchain_PAE_A_B=pae[np.ix_(aidx, bidx)].mean(),
                               mean_interchain_PAE_B_A=pae[np.ix_(bidx, aidx)].mean()))
        model = gemmi.read_structure(str(cif))[0]
        ar = [r for c in model if c.name == 'A' for r in c]
        br = {r.seqid.num: r for c in model if c.name == 'B' for r in c}
        for pos in sorted(targets):
            ridx = indices('B', pos)
            if not ridx:
                raise ValueError(f'{name}: B{pos} has no PAE tokens')
            confidence[-1][f'B{pos}_mean_pLDDT'] = np.mean([a.b_iso for a in br[pos]])
            confidence[-1][f'B{pos}_PAE_A_B'] = pae[np.ix_(aidx, ridx)].mean()
            confidence[-1][f'B{pos}_PAE_B_A'] = pae[np.ix_(ridx, aidx)].mean()
        for pos in sorted(targets):
            residue_b = br[pos]
            expected = 'SEP' if pos in modified else 'SER'
            if residue_b.name != expected:
                raise ValueError(f'{name}: B{pos} is {residue_b.name}, expected {expected}')
            bi = indices('B', pos)
            if not bi:
                raise ValueError(f'{name}: B{pos} has no PAE tokens')
            probes = {'sidechain_O_to_polar': ([a for a in residue_b if a.name == 'OG'], False),
                      'backbone_to_polar': ([a for a in residue_b if a.name in {'N', 'O'}], False)}
            if expected == 'SEP':
                probes['phosphate_to_ARG'] = ([a for a in residue_b if a.name in {'O1P','O2P','O3P','OP1','OP2','OP3'}], True)
            for probe, (target_atoms, arg_only) in probes.items():
                if not target_atoms:
                    raise ValueError(f'{name}: B{pos} missing atoms for {probe}')
                for ra in ar:
                    if arg_only and ra.name != 'ARG':
                        continue
                    aa = [a for a in ra if (a.name in {'NE','NH1','NH2'} if arg_only else a.element.name in {'N','O'})]
                    ai = indices('A', ra.seqid.num)
                    if not aa or not ai:
                        raise ValueError(f'{name}: missing atoms/tokens for A{ra.seqid.num}')
                    distance, atom_a, atom_b = min((a.pos.dist(t.pos), a.name, t.name) for a in aa for t in target_atoms)
                    contacts.append(dict(condition=condition, model=name, seed=seed, sample=sample,
                                         target_B=pos, target_name=expected, probe=probe,
                                         residue_A=f'{ra.name}{ra.seqid.num}', atoms=f'{atom_a}-{atom_b}',
                                         distance_A=distance,
                                         PAE_A_B=pae[np.ix_(ai, bi)].mean(),
                                         PAE_B_A=pae[np.ix_(bi, ai)].mean()))
    write_csv(folder / 'ensemble_confidence.csv', confidence)
    write_csv(folder / 'ensemble_contacts.csv', contacts)
    summary = aggregate(contacts, cutoff)
    for row in summary:
        row['condition'] = condition
    write_csv(folder / 'ensemble_contact_summary.csv', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in sorted({r['target_B'] for r in contacts if r['probe'] == 'phosphate_to_ARG'}):
        group = [r for r in contacts if r['probe'] == 'phosphate_to_ARG' and r['target_B'] == target]
        ax.scatter([r['distance_A'] for r in group],
                   [max(r['PAE_A_B'], r['PAE_B_A']) for r in group], alpha=0.5, label=f'B{target} phosphate–ARG')
    ax.axvline(cutoff, color='grey', linestyle='--')
    ax.set(xlabel='Minimum ARG nitrogen–phosphate oxygen distance (Å)',
           ylabel='Larger of the two directional residue-pair PAEs (Å)',
           title=f'{condition}: {len(files)} models; proximity and positioning confidence')
    if modified:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No phosphorylated serines in this condition',
                transform=ax.transAxes, ha='center')
    fig.tight_layout()
    fig.savefig(folder / 'ensemble_distance_pae.png', dpi=180)
    plt.close(fig)
    print(f'{condition}: analysed {len(files)} models from {len({r["seed"] for r in confidence})} seeds: {folder}')
    print(f'Candidate close approaches use {cutoff} Å; these are not experimental occupancies or confirmed bonds.')
    for r in [r for r in summary if r['probe'] == 'phosphate_to_ARG'][:10]:
        print(f'B{r["target_B"]}–{r["residue_A"]}: {r["within_cutoff"]}/{r["n_models"]} models; median distance {r["median_distance_A"]:.2f} Å; median PAE A/B {r["median_PAE_A_B"]:.1f}, B/A {r["median_PAE_B_A"]:.1f} Å')


def combine_analyses(output_dir, plan):
    import csv
    confidence = []
    contacts = []
    summaries = []
    for entry in plan:
        job = Path(entry['job_dir'])
        for name, rows in [('ensemble_confidence.csv', confidence),
                           ('ensemble_contacts.csv', contacts),
                           ('ensemble_contact_summary.csv', summaries)]:
            with (job / name).open() as handle:
                rows.extend(csv.DictReader(handle))
    write_csv(output_dir / 'comparison_confidence.csv', confidence)
    write_csv(output_dir / 'comparison_contacts.csv', contacts)
    write_csv(output_dir / 'comparison_contact_summary.csv', summaries)
    # Summarise distributions, without treating models as experimental replicates.
    import numpy as np
    rows = []
    for condition in dict.fromkeys(e['condition'] for e in plan):
        group = [r for r in confidence if r['condition'] == condition]
        row = dict(condition=condition, n_models=len(group), n_seeds=len({r['seed'] for r in group}))
        for metric in ('ipTM', 'pTM', 'mean_pLDDT', 'mean_interchain_PAE_A_B', 'mean_interchain_PAE_B_A'):
            values = [float(r[metric]) for r in group if r[metric] not in ('', 'None')]
            for stat, value in [('median', np.median(values) if values else ''),
                                ('min', min(values) if values else ''),
                                ('max', max(values) if values else '')]:
                row[f'{metric}_{stat}'] = value
        rows.append(row)
    write_csv(output_dir / 'comparison_condition_summary.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [e['condition'] for e in plan]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric, ylabel in zip(axes, ('ipTM', 'mean_interchain_PAE_A_B'),
                                     ('ipTM', 'Mean interchain PAE A/B (Å)')):
        for x, condition in enumerate(labels, 1):
            group = [r for r in confidence if r['condition'] == condition and r[metric] not in ('', 'None')]
            values = [float(r[metric]) for r in group]
            if values:
                axis.scatter(np.full(len(values), x) + np.linspace(-0.12, 0.12, len(values)), values, alpha=0.65)
                axis.plot([x - .2, x + .2], [np.median(values)] * 2, color='black')
        axis.set_xticks(range(1, len(labels) + 1), labels)
        axis.set_ylabel(ylabel)
    fig.suptitle('Canine 11A COP1 phosphorylation conditions; dots are predictions, lines are medians')
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison_confidence.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--seeds', type=seed_values, default=list(range(1, 21)))
    parser.add_argument('--num-diffusion-samples', type=positive, default=1)
    parser.add_argument('--num-recycles', type=positive, default=10)
    parser.add_argument('--jobname', default='canine11a')
    parser.add_argument('--msa-mode', choices=['mmseqs2_server', 'single_sequence'], default='mmseqs2_server')
    parser.add_argument('--input-dir', type=Path, default=Path('af3_inputs'))
    parser.add_argument('--output-dir', type=Path, default=Path('af3_output'))
    parser.add_argument('--cache-dir', type=Path, default=Path('af3_cache'))
    parser.add_argument('--runner', type=Path, default=Path('run_alphafold.py'))
    parser.add_argument('--weights', choices=['af3', 'of3'], default='af3')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--on-existing', choices=['error', 'skip', 'overwrite'], default='error')
    parser.add_argument('--skip-analysis', action='store_true')
    parser.add_argument('--contact-cutoff', type=float, default=3.5)
    parser.add_argument('--dry-run', action='store_true', help='Write inputs and print commands, without running inference or analysis')
    args = parser.parse_args()
    if args.contact_cutoff <= 0:
        parser.error('--contact-cutoff must be positive')
    args.conditions = list(dict.fromkeys(args.conditions))
    for key in ('input_dir', 'output_dir', 'cache_dir'):
        setattr(args, key, getattr(args, key).resolve())
        getattr(args, key).mkdir(parents=True, exist_ok=True)
    runner = args.runner.resolve()
    model_dir = (args.model_dir or Path('af3_native_weights' if args.weights == 'af3' else 'af3_converted_weights')).resolve()
    if not args.dry_run:
        if not runner.is_file():
            raise FileNotFoundError(f'Missing runner: {runner}')
        if not model_dir.is_dir() or not list(model_dir.glob('*.bin*')):
            raise FileNotFoundError(f'No prepared weights in {model_dir}')
        if not args.skip_analysis:
            missing = [name for name in ('numpy', 'gemmi', 'matplotlib') if importlib.util.find_spec(name) is None]
            if missing:
                raise ImportError(f'Analysis dependencies missing: {missing}; install them in the working GPU environment first')
    plan = []
    for condition in args.conditions:
        data, settings = make_input(condition, args.seeds, args.num_diffusion_samples, args.num_recycles,
                                    args.msa_mode, args.weights, model_dir, args.jobname)
        path = args.input_dir / f'{data["name"]}.json'
        write_json(path, data)
        plan.append(dict(condition=condition, phosphorylation_positions=CONDITIONS[condition],
                         job_dir=str(args.output_dir / data['name']), input_path=str(path),
                         data=data, settings=settings))
    # Refuse all conflicting output directories before spending GPU time.
    if not args.dry_run and args.on_existing == 'error':
        existing = [e['job_dir'] for e in plan if Path(e['job_dir']).exists()]
        if existing:
            raise FileExistsError(f'Result directories already exist: {existing}')
    print(f'Python: {sys.executable}', flush=True)
    print(f'{len(plan)} conditions x {len(args.seeds)} seeds x {args.num_diffusion_samples} samples = '
          f'{len(plan) * len(args.seeds) * args.num_diffusion_samples} models', flush=True)
    # Keep a separate manifest per requested plan so subset reruns cannot hide an earlier plan.
    plan_id = hashlib.sha256(json.dumps([e['data']['name'] for e in plan]).encode()).hexdigest()[:12]
    write_json(args.output_dir / f'plan_{plan_id}.json', plan)
    device = device_settings()
    for entry in plan:
        run_condition(args, entry['data'], entry['settings'], entry['condition'],
                      Path(entry['input_path']), Path(entry['job_dir']), runner, model_dir, device)
    if not args.dry_run and not args.skip_analysis:
        comparison_dir = args.output_dir / f'comparison_{plan_id}'
        comparison_dir.mkdir(exist_ok=True)
        combine_analyses(comparison_dir, plan)
        print(f'Comparison tables and figure: {comparison_dir}', flush=True)
    print('DRY RUN COMPLETE' if args.dry_run else 'ALL REQUESTED CONDITIONS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
