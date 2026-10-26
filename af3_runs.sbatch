#!/bin/bash
#SBATCH --job-name=af3_9a_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/samuelf/github/ColabFold
#SBATCH --output=af3_9a_test-%j.log

# Submit with: sbatch -A YOUR_GPU_COMPUTE_ACCOUNT af3_canine_9A_test.sbatch /absolute/path/to/af3/python
# Optional second argument: a shell file loading the same modules/environment as the working notebook.
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo 'Pass the absolute path to the Python interpreter used by the working AlphaFold notebook.' >&2
    exit 2
fi
AF3_PYTHON="$1"
if [ "$#" -ge 2 ]; then
    source "$2"
fi
if [[ "$AF3_PYTHON" != /* ]] || [ ! -x "$AF3_PYTHON" ]; then
    echo "Not an executable absolute Python path: $AF3_PYTHON" >&2
    exit 2
fi
export PATH="$(dirname "$AF3_PYTHON"):$PATH"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

storage=/nobackup/proj/disk/naiss2026-4-1221/personal/samuelf
run_root="$storage/af3_batch/test-${SLURM_JOB_ID}"
test -w "$storage"
mkdir -p "$run_root" "$storage/af3_batch/cache"
export TMPDIR="$run_root/tmp"
mkdir -p "$TMPDIR"
# Save the exact submitted driver alongside the results.
cp AlphaFold3_canine_9A.py "$run_root/AlphaFold3_canine_9A.py"
cp run_alphafold.py "$run_root/run_alphafold.py"
git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
echo "Host: $(hostname); job: $SLURM_JOB_ID; results: $run_root"
nvidia-smi

# Fail before inference if the interpreter is incompatible or JAX cannot use the GPU.
srun "$AF3_PYTHON" -u - <<'PY'
import platform, sys
import alphafold3
import jax
print('Architecture:', platform.machine())
print('Python:', sys.executable)
devices = jax.devices()
print('JAX devices:', devices)
if not any(d.platform == 'gpu' for d in devices):
    raise RuntimeError('JAX has no GPU device; check the ARM/GH200 environment.')
PY

# One seed and one sample: an execution test, not a biological comparison.
# Preserve the notebook's ten recycles, MSA mode, sequences, and B49 SEP.
srun "$AF3_PYTHON" -u "$run_root/AlphaFold3_canine_9A.py" \
    --runner "$run_root/run_alphafold.py" \
    --jobname canine_9a_test \
    --seeds 1 \
    --num-diffusion-samples 1 \
    --num-recycles 10 \
    --weights af3 \
    --model-dir /home/samuelf/github/ColabFold/af3_native_weights \
    --input-dir "$run_root/inputs" \
    --output-dir "$run_root/results" \
    --cache-dir "$storage/af3_batch/cache" \
    --skip-analysis

echo "TEST SUCCEEDED: $run_root"
