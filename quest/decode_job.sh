#!/bin/bash
#SBATCH --account=p33146
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=decode_states
#SBATCH --array=0-3
#SBATCH --output=logs/decode_%A_%a.out
#SBATCH --error=logs/decode_%A_%a.err

UNIT_IDS=(83b 85b 116b 119b)
UNIT=${UNIT_IDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Job array task $SLURM_ARRAY_TASK_ID  |  Unit: $UNIT ==="
date

# Load environment
module load mamba/24.3.0
mamba activate nrem_sc

cd ~/nrem_sc
python quest/decode_hpc.py "$UNIT"

echo "=== Finished unit $UNIT ==="
date