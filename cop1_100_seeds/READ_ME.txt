COP1 double-phosphorylation extension: 20 to 100 seeds per construct

Scope: canine 9A (pS39+pS49), canine 11A (pS39+pS51), canine 13A (pS39+pS53).
Human and single-phosphorylation conditions are not rerun.
Generate ONLY seeds 21..100: 80 new models per construct, 240 new models total.
Keep the original seeds 1..20. The combined total will be 300 double-phosphorylated models.

Submit this package on Arrhenius, from its own directory:
cd /home/samuelf/github/ColabFold/cop1_100_seeds
sbatch -A naiss2026-4-1221-gpu af3_double_extend.sbatch /home/samuelf/.conda-hpc/el9_gh200/envs/af3-jupyter/bin/python

Optional final argument: absolute path to a shell file loading the same GPU modules as the working environment.
Submit from the login node. The ARM interpreter runs inside srun on allocated GPU nodes.
run_alphafold.py and prepared af3_native_weights must exist in /home/samuelf/github/ColabFold.

The array contains 3 tasks, at most three running simultaneously:
0: dog9A; 1: dog11A; 2: dog13A. Each task runs four sequential 20-seed blocks, avoiding simultaneous writes to the same construct cache.
Within each construct the blocks are 21-40, 41-60, 61-80, 81-100.
Each task requests 1 GPU, 8 CPU cores, 64 GB host RAM and a four-hour limit.
The four-hour limit is a ceiling, not a runtime estimate.

Predictions retain the existing settings: native AF3 weights, 10 recycles, 1 diffusion sample per seed, MMseqs2 server, original untrimmed sequences, and SEP modifications only at the two stated sites.
Frozen drivers were checked against the sequences in the original structure archive; all chains matched.
Original cache locations are reused. This does not itself guarantee identical MSAs if an MSA must be fetched afresh.
Keep existing caches, software and weights unchanged. Environment versions, runner and driver hashes are recorded in each batch; compare them with original provenance where available.
Each new batch has a distinct output directory; existing directories cause an error instead of being overwritten.

Results:
 /nobackup/proj/disk/naiss2026-4-1221/personal/samuelf/af3_batch/double100-ARRAY_JOB_ID/CONSTRUCT/seedsFIRST-LAST/
Logs in this package directory: cop1_double-ARRAY_JOB_ID_TASK_ID.log
Progress:
 squeue -u samuelf
 sacct -j ARRAY_JOB_ID --format=JobID,State,Elapsed,MaxRSS
 python3 check_extension.py /nobackup/proj/disk/naiss2026-4-1221/personal/samuelf/af3_batch/double100-ARRAY_JOB_ID

Do not resubmit all three tasks if only one fails. A failed-task-only array override (e.g. --array=1 for dog11A) generates a separate array root; keep all roots for analysis. If earlier blocks completed, --export=ALL,COP1_FIRST_SEED=61 starts from seeds 61-80 and then 81-100; supported starts are 21,41,61,81.

Analysis plan fixed before additional predictions:
Segment: chain B, S39 to the downstream phosphoserine.
Primary endpoint: backbone compactness = terminal CA distance / intervening CA contour length.
Back-folding definition: compactness <=0.5 AND terminal-arm angle >=120 degrees.
Arm vectors: CA39->CA42 and CA(end-3)->CA(end).
Do not change these thresholds after seeing new models.
Main confirmation comparison: dog11A versus dog9A in the NEW 80 seeds, using a two-sided Fisher test for the back-folding counts; report effect size and uncertainty as well as the p value.
Compactness, 13A comparisons, contact geometry and confidence measures are supporting analyses; correct multiple pairwise tests where performed.
Report new 80 separately from original 20; the combined 100 is a pooled descriptive analysis.
Do not select or discard seeds based on FoldX score or visual appearance.
Model frequencies are not biological replicate frequencies or equilibrium populations.
The inference driver's built-in analysis covers confidence and phosphate contacts. The frozen backbone definitions above will be evaluated after collecting the new structures.
No FoldX scoring is launched by this package.

Local checks are dry-runs and simulated output validation only. GPU inference has not been run or submitted from this laptop.
