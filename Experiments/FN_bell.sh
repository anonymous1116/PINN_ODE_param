#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=04:00:00
#SBATCH --qos=standby
#SBATCH --array=0-99               # Create a job array with indices from 1 to 10
#SBATCH --output=output_log_training/output_log_%A_%a.out
#SBATCH --error=output_log_training/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log_training

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/ODE

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/PINN_ODE_param
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID))

penalty=3e-02
#python ./Experiments/FN_CV.py --seed $seed --true_sigma 2e-1
python ./Experiments/FN_CV_individual.py --seed $seed --true_sigma 2e-2 --penalty $penalty
python ./Experiments/FN_penalty.py --seed $seed --true_sigma 2e-2 --penalty $penalty
#python ./Experiments/FN_CV_individual.py --seed 1 --true_sigma 2e-1 --penalty 1.0e+00

#python ./Experiments/PTrans_penalty.py --seed 1 --true_sigma 1e-1 --penalty 1e+00
#python ./Experiments/PTrans_CV.py --seed $seed --true_sigma 1e-1
#python ./Experiments/PTrans_CV.py --seed $seed --true_sigma 1e-2
#python ./Experiments/FN_SA2.py --seed 1 --true_sigma 2e-1 --penalty 1e+00
