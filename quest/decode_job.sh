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
#SBATCH --mail-type=END
#SBATCH --mail-user=tuguldur.gerelmaa@northwestern.edu

UNIT_IDS=(83b 85b 116b 119b)
UNIT=${UNIT_IDS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH='/projects/p33146/Tuguldur/nrem_sc/processed'
SAVE_PATH='/scratch/iii9781/nrem_sc'

echo "=== Job array task $SLURM_ARRAY_TASK_ID  |  Unit: $UNIT ==="
echo $SLURM_PRIO_PROCESS
echo $SLURM_CPUS_PER_TASK
date

# Load environment
module purge all
module load mamba/24.3.0
source activate /home/iii9781/.conda/envs/decoding

# cd ~/nrem_sc

# mamba create -n decoding -f env.txt -c conda-forge
python decode_hpc.py "${DATA_PATH}/${UNIT}" "${SAVE_PATH}/${UNIT}"

echo "=== Finished unit $UNIT ==="
date