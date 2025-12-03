#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=01:00:00
#SBATCH --qos=normal
#SBATCH --array=0-99               # Create a job array with indices from 1 to 10
#SBATCH --output=output_log_training/output_log_%A_%a.out
#SBATCH --error=output_log_training/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log_training

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/ODE

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/PINN_ODE_parm
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID))

#python ./Experiments/FN_SA.py --seed $seed --true_sigma 2e-1
python ./Experiments/FN2.py --seed $seed --true_sigma 2e-1
