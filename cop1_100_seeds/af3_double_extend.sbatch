#!/bin/bash
#SBATCH --job-name=cop1_double_extend
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=cop1_double-%A_%a.log

# Submit FROM this package directory; each task adds 80 seeds in four blocks of 20.
# Arguments: absolute AF3 Python path; optional shell file loading GPU modules.
set -euo pipefail
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
AF3_PYTHON="${1:?Supply the absolute path to the working AF3 Python}"
if [ "$#" -ge 2 ]; then source "$2"; fi
[[ "$AF3_PYTHON" == /* ]] && test -x "$AF3_PYTHON"
bundle="$SLURM_SUBMIT_DIR"
repo=/home/samuelf/github/ColabFold
storage=/nobackup/proj/disk/naiss2026-4-1221/personal/samuelf
constructs=(dog9A dog11A dog13A)
drivers=(AlphaFold3_canine_9A.py AlphaFold3_canine_11A.py AlphaFold3_canine_13A.py)
conditions=(S39+S49 S39+S51 S39+S53)
caches=(cache cache/canine11a cache/canine13a)
task="$SLURM_ARRAY_TASK_ID"
(( task >= 0 && task < 3 ))
group="$task"
construct="${constructs[$group]}"
driver="${drivers[$group]}"
condition="${conditions[$group]}"
root="$storage/af3_batch/double100-${SLURM_ARRAY_JOB_ID}"
test -w "$storage"
test -f "$repo/run_alphafold.py"
# Optionally resume only later blocks in a NEW array: COP1_FIRST_SEED=41,61,81.
block_start="${COP1_FIRST_SEED:-21}"
case "$block_start" in 21|41|61|81) ;; *) echo "Invalid COP1_FIRST_SEED" >&2; exit 2 ;; esac
for first in 21 41 61 81; do
    (( first >= block_start )) || continue
    last=$((first + 19))
    run="$root/$construct/seeds${first}-${last}"
    # Never reuse a previous task directory or overwrite earlier results.
    mkdir -p "$(dirname "$run")"
    mkdir "$run"
    export PATH="$(dirname "$AF3_PYTHON"):$PATH"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export TMPDIR="$run/tmp" MPLCONFIGDIR="$run/matplotlib"
    mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$storage/af3_batch/${caches[$group]}"
    cp "$bundle/$driver" "$run/$driver"
    cp "$bundle/validate_extension.py" "$run/validate_extension.py"
    cp "$bundle/extension_spec.json" "$run/extension_spec.json"
    cp "$repo/run_alphafold.py" "$run/run_alphafold.py"
    git -C "$repo" rev-parse HEAD > "$run/git_commit.txt"
    git -C "$repo" status --short > "$run/git_status.txt"
    sha256sum "$run/$driver" "$run/run_alphafold.py" > "$run/source_sha256.txt"
    echo "Construct: $construct; condition: $condition; NEW seeds: $first-$last"
    echo "Host: $(hostname); array: $SLURM_ARRAY_JOB_ID; task: $task; run: $run"
    nvidia-smi
    srun --cpu-bind=none "$AF3_PYTHON" -u "$run/validate_extension.py" \
        --bundle "$run" --construct "$construct" --gpu-check > "$run/environment_check.txt"
    cat "$run/environment_check.txt"
    seeds=$(seq -s, "$first" "$last")
    cd "$repo"
    srun --cpu-bind=none "$AF3_PYTHON" -u "$run/$driver" \
        --runner "$run/run_alphafold.py" \
        --jobname "${construct}_double_extension" \
        --conditions "$condition" --seeds "$seeds" \
        --num-diffusion-samples 1 --num-recycles 10 \
        --msa-mode mmseqs2_server --weights af3 \
        --model-dir "$repo/af3_native_weights" \
        --input-dir "$run/inputs" --output-dir "$run/results" \
        --cache-dir "$storage/af3_batch/${caches[$group]}"
    srun --cpu-bind=none "$AF3_PYTHON" -u "$run/validate_extension.py" \
        --bundle "$run" --construct "$construct" \
        --results "$run/results" --first "$first" --last "$last" \
        > "$run/extension_verified.json"
    echo "EXTENSION BATCH COMPLETE: $construct seeds $first-$last; $run"
done
