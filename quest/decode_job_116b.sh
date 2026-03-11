#!/bin/bash
#SBATCH --account=p33146
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=decode_states
#SBATCH --array=0-3
#SBATCH --output=logs/decode_%A_%a.out
#SBATCH --error=logs/decode_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tuguldur.gerelmaa@northwestern.edu

DATA_PATH='/projects/p33146/Tuguldur/nrem_sc/processed'
SAVE_PATH='/scratch/iii9781/nrem_sc'

echo "=== Job array task $SLURM_ARRAY_TASK_ID  |  Unit: 116b ==="
date

# Load environment
module purge all
module load mamba/24.3.0
source activate ~/.conda/envs/decoding

python ~/nrem_sc/decode_hpc.py "${DATA_PATH}/116b" "${SAVE_PATH}/116b" "$SLURM_ARRAY_TASK_ID"

echo "=== Finished unit 116b ==="
date