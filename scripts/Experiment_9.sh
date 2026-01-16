#!/bin/bash
#SBATCH --job-name=thesis_experiment1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=28GB
#SBATCH --cpus-per-task=16           # Use 16 cores
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-300%10             # 125 jobs, max 10 running at once
#SBATCH --partition=rome

# Load environment
module purge
module load 2024
module load SciPy-bundle/2024.05-gfbf-2024a

# Prevent numpy, OpenBLAS, etc. from using multiple threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "Holistic/" || { echo "ERROR: Directory 'Experiment_1/' not found."; exit 1; }
mkdir -p results_snellius

PARAMS_FILE="params_holistic.txt"
BATCH_SIZE=16

# Launch each simulation in parallel (1 core per sim)
for i in $(seq 0 $((BATCH_SIZE - 1))); do
  idx=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + i + 1 ))
  line=$(sed -n "${idx}p" "$PARAMS_FILE")

  [[ -z "$line" ]] && continue  # skip if line is empty

  read -r group_rounds turnover_rounds pop_size network mode run_id dynamic replace init <<< "$(echo "$line" | tr -d '\r' | xargs)"

  echo "[$(date '+%F %T')] Launching sim $idx: $line"
  

  # Run the simulation using 1 core
  python sim.py \
    --mode "$mode" \
    --group_round "$group_rounds" \
    --turnover_round "$turnover_rounds" \
    --pop_size "$pop_size" \
    --run_id "$run_id" \
    --network_type "$network" \
    --dynamic "$dynamic"\
    --replace "$replace"\
    --init_rounds "$init"\
    > "results_snellius/out_${idx}.txt" 2>&1 &
done

# Wait for all background tasks
wait
echo "[$(date '+%F %T')] Task $SLURM_ARRAY_TASK_ID complete."
