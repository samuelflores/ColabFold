#!/bin/bash
#SBATCH --job-name=canine11a_phospho
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/samuelf/github/ColabFold
#SBATCH --output=canine11a_phospho-%j.log

# Submit with the absolute path to the working AlphaFold Python as argument 1.
# Optional argument 2: shell file loading the modules used by the working notebook.
set -euo pipefail
# Avoid inheriting CPU masks when submitted from another allocation.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
AF3_PYTHON="${1:?Pass the working AlphaFold Python interpreter path}"
if [ "$#" -ge 2 ]; then source "$2"; fi
[[ "$AF3_PYTHON" == /* ]] && test -x "$AF3_PYTHON"
export PATH="$(dirname "$AF3_PYTHON"):$PATH"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

storage=/nobackup/proj/disk/naiss2026-4-1221/personal/samuelf
run_root="$storage/af3_batch/canine11a-${SLURM_JOB_ID}"
test -w "$storage"
mkdir -p "$run_root" "$storage/af3_batch/cache/canine11a"
export TMPDIR="$run_root/tmp" MPLCONFIGDIR="$run_root/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cp AlphaFold3_canine_11A.py "$run_root/AlphaFold3_canine_11A.py"
cp run_alphafold.py "$run_root/run_alphafold.py"
git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
echo "Host: $(hostname); job: $SLURM_JOB_ID; run directory: $run_root"
nvidia-smi

# Check analysis dependencies before spending time on inference.
srun "$AF3_PYTHON" -u - <<'PY'
import platform, sys
import alphafold3, gemmi, matplotlib, numpy, jax
print('Architecture:', platform.machine(), 'Python:', sys.executable)
print('JAX devices:', jax.devices())
if not any(d.platform == 'gpu' for d in jax.devices()):
    raise RuntimeError('JAX cannot access a GPU; check the ARM/GH200 environment')
PY

# Five conditions, identical canine 11A sequences, matched seeds 1..20: 100 models.
# Analysis is built into this species/sequence-specific Python file.
srun "$AF3_PYTHON" -u "$run_root/AlphaFold3_canine_11A.py" \
    --runner "$run_root/run_alphafold.py" \
    --conditions S51 S39 S39+S51 none S28 \
    --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
    --num-diffusion-samples 1 --num-recycles 10 \
    --weights af3 \
    --model-dir /home/samuelf/github/ColabFold/af3_native_weights \
    --input-dir "$run_root/inputs" --output-dir "$run_root/results" \
    --cache-dir "$storage/af3_batch/cache/canine11a"

echo "ALL FIVE CONDITIONS AND ANALYSIS SUCCEEDED: $run_root"
