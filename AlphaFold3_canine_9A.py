#!/usr/bin/env python3
"""Batch adaptation of AlphaFold3_canine_9A.ipynb.

Run in the existing AlphaFold environment from the Arrhenius working directory.
Requires an existing run_alphafold.py (notebook version: sokrypton v3.1.5),
prepared model weights, and built AF3 data. This script does not install packages,
download weights, patch installed packages, or open interactive viewers.
Default: canine PIRH2 / 9A COP1, B49 SEP, seed 1, five samples, ten recycles.
Use --help for options; --dry-run prepares input and prints the inference command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main():
    import re, sys, os, json, hashlib
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobname', default='canine_9a')
    parser.add_argument('--seeds', default='1', help='Comma-separated nonnegative integer seeds')
    parser.add_argument('--num-diffusion-samples', type=positive, default=5)
    parser.add_argument('--num-recycles', type=positive, default=10)
    parser.add_argument('--msa-mode', choices=['mmseqs2_server', 'single_sequence'], default='mmseqs2_server')
    parser.add_argument('--output-dir', default='af3_output')
    parser.add_argument('--input-dir', default='af3_inputs')
    parser.add_argument('--cache-dir', default='af3_cache')
    parser.add_argument('--runner', default='run_alphafold.py')
    parser.add_argument('--weights', choices=['af3', 'of3'], default='af3')
    parser.add_argument('--model-dir', help='Default: af3_native_weights or af3_converted_weights')
    parser.add_argument('--on-existing', choices=['error', 'skip', 'overwrite'], default='error')
    parser.add_argument('--skip-analysis', action='store_true', help='Run inference without contact analysis or plots')
    parser.add_argument('--dry-run', action='store_true', help='Write input JSON and show command without inference or analysis')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+(?:\s*,\s*\d+)*', args.seeds.strip()):
        parser.error('--seeds must contain comma-separated nonnegative integers')
    print('Python:', sys.executable, flush=True)
    import re, os, json, hashlib


    protein = 'MAAAAREDGARGREQGLRGCEHYDRGCLLKAPCCDKLYTCRLCHDNNEDHQLDRFKVKEVQCINCEKIQHAQQSCEECSTLFGEYYCSICHLFDKDKKQYHCENCGICRIGPKEDFFHCLKCNLCLAMNLQGKHKCIENVSRQNCPICLEDIHTSRVVAHVLPCGHLLHRTCYEEMLKEGYRCPLCMHSALDMTRYWRQLDDEVAQTPMPSEYQNMTVDILCNDCNRRSTVQFHILGMKCNICESYNTAQAKRISLDQQ:MSGSRQAGSGSAGTSPGSSAASSVTSASSSLSSSPSPPSAAAAAAAAASGGAAQAAGSGGLGGPAARPVLVAAAVSGSGGGAVSAGQSRLSCAARPSAGVGGSSSGLGSGSRKRPLLA'


    dna = ''
    rna = ''
    ligand_ccd = ''
    ligand_smiles = ''

    jobname = args.jobname
    msa_mode = args.msa_mode
    seeds = args.seeds
    on_existing = args.on_existing

    def split_entries(s):
      s = re.sub(r':+', ':', s).strip(':')
      return [e for e in (''.join(tok.split()) for tok in s.split(':')) if e]

    prot_seqs   = [e.upper() for e in split_entries(protein)]
    dna_seqs    = [e.upper() for e in split_entries(dna)]
    rna_seqs    = [e.upper() for e in split_entries(rna)]
    ccd_codes   = [e.upper() for e in split_entries(ligand_ccd)]
    smiles_strs = split_entries(ligand_smiles)         # case-sensitive: leave as typed

    CHAIN_IDS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    chains, prot_groups, idx = [], {}, 0

    for seq in prot_seqs:
      cid = CHAIN_IDS[idx]; idx += 1
      if seq in prot_groups:                           # merge identical seqs -> homo-oligomer
        ent = prot_groups[seq]
        ids = ent['id'] if isinstance(ent['id'], list) else [ent['id']]
        ent['id'] = ids + [cid]
      else:
        ent = {'id': cid, 'sequence': seq, 'templates': []}
        if msa_mode == 'single_sequence':
          ent.update({'unpairedMsa': f'>query\n{seq}\n', 'pairedMsa': ''})
        prot_groups[seq] = ent
        chains.append({'protein': ent})

    for seq in rna_seqs:
      c = {'id': CHAIN_IDS[idx], 'sequence': seq}
      if msa_mode == 'single_sequence':
        c['unpairedMsa'] = f'>query\n{seq}\n'
      chains.append({'rna': c}); idx += 1

    for seq in dna_seqs:
      chains.append({'dna': {'id': CHAIN_IDS[idx], 'sequence': seq}}); idx += 1

    for code in ccd_codes:
      chains.append({'ligand': {'id': CHAIN_IDS[idx], 'ccdCodes': [code]}}); idx += 1

    for smiles in smiles_strs:
      chains.append({'ligand': {'id': CHAIN_IDS[idx], 'smiles': smiles}}); idx += 1

    if not chains:
      raise ValueError('No valid input found - fill in at least one box.')


    chain_b = next(
        c["protein"] for c in chains
        if "protein" in c and c["protein"]["id"] == "B"
    )




    assert chain_b["sequence"][39:49] == "AAAAAAAAAS"
    chain_b["modifications"] = [
        {"ptmType": "SEP", "ptmPosition": 49},
    ]

    seed_list = []
    for tok in re.findall(r'\d+', seeds):
      v = int(tok)
      if v not in seed_list:
        seed_list.append(v)
    if not seed_list:
      seed_list = [1]

    def _flat(mol):
      if 'sequence' in mol: return mol['sequence']
      if 'ccdCodes' in mol: return ','.join(mol['ccdCodes'])
      return mol.get('smiles', '?')
    # Include modifications and inference settings in the job identifier.
    identity = {'sequences': chains, 'modelSeeds': seed_list, 'msa_mode': msa_mode,
                'num_recycles': args.num_recycles,
                'num_diffusion_samples': args.num_diffusion_samples,
                'weights': args.weights,
                'model_dir': str(Path(args.model_dir or ('af3_native_weights' if args.weights == 'af3' else 'af3_converted_weights')).resolve())}
    basejob = (re.sub(r'\W+', '', ''.join(jobname.split())) or 'job').lower()
    jobname = basejob + '_' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    INPUT_DIR = str(Path(args.input_dir).resolve())
    OUTPUT_DIR = str(Path(args.output_dir).resolve())
    job_dir = str(Path(OUTPUT_DIR) / jobname)
    fold_input = {'name': jobname, 'sequences': chains, 'modelSeeds': seed_list,
                  'dialect': 'alphafold3', 'version': 1}
    os.makedirs(INPUT_DIR, exist_ok=True)
    json_path = f'{INPUT_DIR}/{jobname}.json'
    with open(json_path, 'w') as f:
      json.dump(fold_input, f, indent=2)

    print(f'Job "{jobname}"  ->  results will be written to {job_dir}/')
    fold_input

    num_recycles = args.num_recycles
    num_diffusion_samples = args.num_diffusion_samples
    #@markdown - `num_recycles`: refinement passes through the network (default 10). More can help large/hard targets, but is slower.
    #@markdown - `num_diffusion_samples`: candidate structures generated per seed (default 5). Total models = seeds x samples.

    num_recycles = max(1, int(num_recycles))
    num_diffusion_samples = max(1, int(num_diffusion_samples))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Re-run policy (one folder per job, no timestamped duplicates):
    #   overwrite -> wipe this job's folder and recompute
    #   skip      -> if a finished result (.cif) is already there, don't recompute
    completion = Path(job_dir) / 'BATCH_INFERENCE_COMPLETE.json'
    if Path(job_dir).exists() and not args.dry_run:
      if on_existing == 'skip' and completion.is_file():
        print(f'Skipping completed job: {job_dir}', flush=True)
        return
      if on_existing != 'overwrite':
        raise FileExistsError(f'{job_dir} exists. Use a new name or explicitly choose --on-existing overwrite. Incomplete jobs are not skipped.')

    # Pick attention impl + XLA flags from the actual device.
    # Triton/cuDNN flash attention need Ampere (compute capability >= 8.0);
    # 7.x GPUs (T4=7.5, V100=7.0) and CPU use the portable XLA path, and 7.x
    # additionally needs the XLA flag that disables the custom-kernel fusion pass.
    def detect_device():
      try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15)
        caps = [float(x) for x in out.stdout.split() if x.strip()]
        if caps:
          return 'gpu', min(caps)
      except Exception:
        pass
      return 'cpu', None

    device, cap = detect_device()
    nojit = False
    xla_flags = []   # extra XLA flags to export for this device (per AlphaFold 3's guidance)

    if device == 'cpu':
      flash_impl = 'xla'
      nojit = True
      print('No GPU detected - running on CPU with XLA attention + --nojit (slow, but avoids the compile).')
    elif cap < 8.0:
      # T4 / V100 (cc 7.x): XLA attention; disable the custom-kernel fusion pass.
      # (Triton GEMM is not supported on these cards, so it is not disabled here.)
      flash_impl = 'xla'
      xla_flags = ['--xla_disable_hlo_passes=custom-kernel-fusion-rewriter']
      print(f'Pre-Ampere GPU (compute capability {cap}) - XLA attention + custom-kernel fusion disabled.')
    elif 8.0 < cap < 9.0:
      # L4 / Ada / consumer Ampere (cc 8.6 / 8.9): limited shared memory. XLA's Triton GEMM
      # kernels exceed it ('Shared memory size limit exceeded'), so disable Triton GEMM
      # (falls back to cuBLAS) and use XLA attention to also avoid the Triton attention kernel.
      flash_impl = 'xla'
      xla_flags = ['--xla_gpu_enable_triton_gemm=false']
      print(f'Ada/consumer GPU (compute capability {cap}) - XLA attention + Triton GEMM disabled (shared-memory limit).')
    else:
      # A100 (cc 8.0) and H100 (cc 9.0+): ample shared memory. Triton flash attention,
      # with Triton GEMM disabled per AlphaFold 3's recommended XLA_FLAGS.
      flash_impl = 'triton'
      xla_flags = ['--xla_gpu_enable_triton_gemm=false']
      print(f'Datacenter GPU (compute capability {cap}) - Triton flash attention + Triton GEMM disabled.')

    # Export XLA flags so the child shell (and JAX inside it) inherit them.
    cur = os.environ.get('XLA_FLAGS', '')
    for f in xla_flags:
      if f not in cur:
        cur = (cur + ' ' + f).strip()
    if cur:
      os.environ['XLA_FLAGS'] = cur

    print('XLA_FLAGS =', os.environ.get('XLA_FLAGS', '(unset)'))

    # Weights: use the official AlphaFold 3 weights if they were fetched, else OpenFold3.
    use_of3 = args.weights == 'of3'
    model_dir = str(Path(args.model_dir or ('af3_converted_weights' if use_of3 else 'af3_native_weights')).resolve())
    runner = Path(args.runner).resolve()
    if not args.dry_run:
      if not runner.is_file():
        raise FileNotFoundError(f'Missing runner: {runner}; use --runner PATH')
      if not Path(model_dir).is_dir() or not list(Path(model_dir).glob('*.bin*')):
        raise FileNotFoundError(f'No prepared weights in {model_dir}; use --model-dir PATH')

    cmd = [
        sys.executable, str(runner),
        f'--json_path={json_path}',
        f'--model_dir={model_dir}',
        '--norun_data_pipeline',
        f'--output_dir={OUTPUT_DIR}',
        f'--cache_dir={Path(args.cache_dir).resolve()}',
        '--force_output_dir',          # reuse af3_output/<jobname>/ instead of a timestamped copy
        f'--flash_attention_implementation={flash_impl}',
        f'--num_recycles={num_recycles}',
        f'--num_diffusion_samples={num_diffusion_samples}',
    ]
    if msa_mode == 'mmseqs2_server':
      cmd.append('--use_msa_server')
    if nojit:
      cmd.append('--nojit')
    if use_of3:
      cmd.append('--of3_weights')

    import shlex
    print(shlex.join(cmd), flush=True)
    print(f'{len(seed_list)} seeds x {num_diffusion_samples} samples = {len(seed_list) * num_diffusion_samples} models', flush=True)
    if args.dry_run:
      return
    # Remove only this specific job directory, and only on explicit request.
    if Path(job_dir).exists() and on_existing == 'overwrite':
      shutil.rmtree(job_dir)
    subprocess.run(cmd, check=True)
    expected = len(seed_list) * num_diffusion_samples
    models = list(Path(job_dir).glob('seed-*_sample-*/*_model.cif'))
    if len(models) != expected:
      raise RuntimeError(f'Expected {expected} models but found {len(models)}; job not marked complete')
    completion.write_text(json.dumps({'input': fold_input, 'settings': identity,
                                      'command': cmd, 'model_count': len(models)}, indent=2))
    print(f'Inference complete: {job_dir}', flush=True)
    if args.skip_analysis:
      return
    import matplotlib
    matplotlib.use('Agg')


    from pathlib import Path
    import csv, json
    import numpy as np


    folder = Path(job_dir)
    ranking = list(folder.glob("*ranking_scores.csv"))

    if ranking:
        with ranking[0].open() as f:
            rows = sorted(csv.DictReader(f),
                          key=lambda r: float(r["ranking_score"]), reverse=True)
        cifs = [
            next((folder / f"seed-{r['seed']}_sample-{r['sample']}").glob("*_model.cif"))
            for r in rows
        ]
    else:
        cifs = sorted(folder.glob("seed-*_sample-*/*_model.cif"))

    assert cifs, f"No models found in {folder}"


    #Get contacts between chains
    import csv, json
    from pathlib import Path
    import gemmi
    import numpy as np

    rows = []
    for cif in cifs:
        path = Path(cif)
        conf_path = next(
            p for p in path.parent.glob("*_confidences.json")
            if "summary" not in p.name
        )
        conf = json.loads(conf_path.read_text())
        pae = np.asarray(conf["pae"])
        ids = list(zip(conf["token_chain_ids"],
                       map(int, conf["token_res_ids"])))

        def indices(chain, residue):
            return [i for i, pair in enumerate(ids)
                    if pair == (chain, residue)]

        model = gemmi.read_structure(str(path))[0]
        arginines = [r for c in model if c.name == "A"
                     for r in c if r.name == "ARG"]
        phosphoserines = [
            r for c in model if c.name == "B"
            for r in c
            if r.name == "SEP" and 38 <= r.seqid.num <= 51
        ]
        for ser in phosphoserines:
            oxygens = [a for a in ser
                       if a.name in {"O1P", "O2P", "O3P",
                                     "OP1", "OP2", "OP3"}]
            if not oxygens:
                raise ValueError("Phosphate oxygen names not recognised.")

            for arg in arginines:
                nitrogens = [a for a in arg
                             if a.name in {"NE", "NH1", "NH2"}]
                distance, nitrogen, oxygen = min(
                    (n.pos.dist(o.pos), n.name, o.name)
                    for n in nitrogens for o in oxygens
                )
                ai = indices("A", arg.seqid.num)
                bi = indices("B", ser.seqid.num)
                if not ai or not bi:
                    raise ValueError("Cannot map residue numbers to PAE tokens.")

                rows.append({
                    "model": path.parent.name,
                    "Arg_A": arg.seqid.num,
                    "pSer_B": ser.seqid.num,
                    "atoms": f"{nitrogen}-{oxygen}",
                    "distance_A": round(distance, 2),
                    "PAE_A_B": round(float(pae[np.ix_(ai, bi)].mean()), 2),
                    "PAE_B_A": round(float(pae[np.ix_(bi, ai)].mean()), 2),
                })

    if not rows:
        raise ValueError("No B39/B49 phosphoserines found; check the loaded job.")

    output = Path(job_dir) / "arginine_phosphoserine_contacts.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Closest three arginines per phosphoserine in each model:")
    for model, ser in sorted({(r["model"], r["pSer_B"]) for r in rows}):
        group = [r for r in rows
                 if r["model"] == model and r["pSer_B"] == ser]
        for r in sorted(group, key=lambda r: r["distance_A"])[:3]:
            print(r)
    print("Full table:", output)
    #@title Quality metrics and plots
    import json, os
    import numpy as np
    import matplotlib.pyplot as plt

    conf_path = f'{OUTPUT_DIR}/{jobname}/{jobname}_confidences.json'
    summ_path = f'{OUTPUT_DIR}/{jobname}/{jobname}_summary_confidences.json'

    with open(conf_path) as f:
      conf = json.load(f)
    with open(summ_path) as f:
      summ = json.load(f)

    plddts = np.array(conf.get('atom_plddts', conf.get('token_plddts', [])), dtype=float)
    plddt_chain_ids = conf.get('atom_chain_ids', conf.get('token_chain_ids', []))   # pLDDT is per-ATOM
    token_chain_ids = conf.get('token_chain_ids', [])                                # PAE is per-TOKEN
    pae = np.array(conf.get('pae', []), dtype=float)

    # ── Summary (ipTM is None for single-chain jobs — guard before formatting) ─
    def fmt(v):
      return f'{v:.3f}' if isinstance(v, (int, float)) else 'n/a'

    mean_plddt = summ.get('mean_plddt')
    if mean_plddt is None and plddts.size:
      mean_plddt = float(np.mean(plddts))
    iptm = summ.get('iptm')

    print('=' * 38)
    print(f'Mean pLDDT     : {fmt(mean_plddt)}')
    print(f'pTM            : {fmt(summ.get("ptm"))}')
    print(f'ipTM           : {fmt(iptm)}' + ('   (single chain — no interface)' if iptm is None else ''))
    print(f'Ranking score  : {fmt(summ.get("ranking_score"))}')
    print('=' * 38)

    # ── Plots ───────────────────────────────────────────────────
    has_pae = pae.ndim == 2 and pae.size > 0
    ncols = 2 if has_pae else 1
    fig, axes = plt.subplots(1, ncols, figsize=(13 if has_pae else 6.5, 4))
    axes = np.atleast_1d(axes)

    # pLDDT per residue — a line (coloured per chain when there is more than one)
    ax = axes[0]
    x = np.arange(len(plddts))
    xmax = max(len(plddts) - 1, 1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)

    unique_chains = list(dict.fromkeys(plddt_chain_ids))
    if len(unique_chains) > 1:
      colors = plt.cm.tab10(np.linspace(0, 1, len(unique_chains)))
      tcid = np.array(plddt_chain_ids)
      for ch, col in zip(unique_chains, colors):
        y = np.where(tcid == ch, plddts, np.nan)   # NaN gaps keep chains as separate lines
        ax.plot(x, y, lw=1.5, color=col, label=f'Chain {ch}')
      for b in [i for i in range(1, len(plddt_chain_ids)) if plddt_chain_ids[i] != plddt_chain_ids[i-1]]:
        ax.axvline(b - 0.5, color='grey', lw=0.6, alpha=0.5)
      ax.legend(loc='lower right', fontsize=8)
    else:
      ax.plot(x, plddts, lw=1.5, color='#1f77b4')

    for y in (50, 70, 90):
      ax.axhline(y, ls='--', lw=0.7, color='grey', alpha=0.5)
      ax.text(xmax, y, f' {y}', va='center', ha='left', fontsize=7, color='grey')
    ax.set_xlabel('Atom')
    ax.set_ylabel('pLDDT')
    ax.set_title('Predicted pLDDT per atom')

    # PAE matrix
    if has_pae:
      ax = axes[1]
      im = ax.imshow(pae, cmap='bwr', vmin=0, vmax=30, interpolation='nearest')
      plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='PAE (Å)')
      if token_chain_ids:
        for b in [i for i in range(1, len(token_chain_ids)) if token_chain_ids[i] != token_chain_ids[i-1]]:
          ax.axhline(b - 0.5, c='black', lw=0.8)
          ax.axvline(b - 0.5, c='black', lw=0.8)
      ax.set_xlabel('Scored residue')
      ax.set_ylabel('Aligned residue')
      ax.set_title('Predicted Aligned Error (PAE)')

    plt.tight_layout()
    plt.savefig(Path(job_dir) / 'quality_metrics.png', dpi=180)
    plt.close(fig)

    import json
    from pathlib import Path
    import gemmi
    import numpy as np
    from collections import defaultdict

    results = defaultdict(list)

    for cif in cifs:
        path = Path(cif)
        model = gemmi.read_structure(str(path))[0]
        ser = next(r for c in model if c.name == "B"
                   for r in c if r.seqid.num == 39)
        assert ser.name == "SER", f"B39 is {ser.name}, not unmodified SER"
        og = next(a for a in ser if a.name == "OG")

        conf_path = next(p for p in path.parent.glob("*_confidences.json")
                         if "summary" not in p.name)
        conf = json.loads(conf_path.read_text())
        pae = np.asarray(conf["pae"])
        ids = list(zip(conf["token_chain_ids"],
                       map(int, conf["token_res_ids"])))

        def idx(chain, number):
            return [i for i, pair in enumerate(ids)
                    if pair == (chain, number)]

        bi = idx("B", 39)
        nearby = []

        for chain in model:
            if chain.name != "A":
                continue
            for residue in chain:
                atoms = [a for a in residue
                         if a.element.name in ("N", "O")]
                ai = idx("A", residue.seqid.num)
                if not atoms or not ai or not bi:
                    continue

                distance, atom = min(
                    (og.pos.dist(a.pos), a.name) for a in atoms
                )
                ab = float(pae[np.ix_(ai, bi)].mean())
                ba = float(pae[np.ix_(bi, ai)].mean())
                label = f"{residue.name}{residue.seqid.num}"
                results[label].append((distance, ab, ba))
                nearby.append((distance, label, atom, ab, ba))

        print(f"\n{path.parent.name}: closest to B39 OG")
        for d, label, atom, ab, ba in sorted(nearby)[:3]:
            print(f"  {label}:{atom}  {d:.2f} Å"
                  f"  PAE A/B={ab:.1f}, B/A={ba:.1f} Å")

    print("\nRecurring close approaches (OG–N/O ≤3.5 Å):")
    for label, values in sorted(
        results.items(),
        key=lambda item: (-sum(v[0] <= 3.5 for v in item[1]),
                          np.mean([v[0] for v in item[1]]))
    ):
        count = sum(v[0] <= 3.5 for v in values)
        if count:
            print(f"  {label}: {count}/{len(values)} models;"
                  f" distances {[round(v[0], 2) for v in values]} Å")

if __name__ == '__main__':
    main()
