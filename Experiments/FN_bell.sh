#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=00:29:00
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

# --- Logic for Seeds and Penalties ---
# Define the 8 penalty values
#penalty_list=(0.0001 0.00005)
penalty_list=(0.0001 0.001 0.1 10)
#penalty_list=(3 5 10 30 50 100 300 500)

# Calculate which penalty to use (0 to 7)
penalty_idx=$((SLURM_ARRAY_TASK_ID / 100))
penalty=${penalty_list[$penalty_idx]}

# Calculate which seed to use (0 to 99)
seed=$((SLURM_ARRAY_TASK_ID % 100))


#python ./Experiments/FN_CV_individual.py --seed $seed --true_sigma 0.05 --penalty $penalty
#python ./Experiments/FN_CV_optimal.py --seed $seed --true_sigma 0.05 
#python ./Experiments/FN_penalty.py --seed $seed --true_sigma 0.1 --penalty 1
#python ./Experiments/PTrans2_penalty.py --seed $seed --true_sigma 0.01 --penalty $penalty
python ./Experiments/PTrans2_CV_optimal.py --seed $seed --true_sigma 0.01
#python ./Experiments/PTrans_penalty.py --seed $seed --true_sigma 0.1 --penalty $penalty
#python ./Experiments/PTrans_penalty2.py --seed $seed --true_sigma 0.01 --penalty 0.01

#python ./Experiments/SIR_penalty.py --seed $seed --true_sigma 1 --penalty $penalty
#python ./Experiments/SIR_CV_individual.py --seed $seed --true_sigma 5 --penalty $penalty
#python ./Experiments/SIR_CV_optimal.py --seed $seed --true_sigma 1
#python ./Experiments/PTrans_CV.py --seed $seed --true_sigma 1e-1
#python ./Experiments/PTrans_CV.py --seed $seed --true_sigma 1e-2
#python ./Experiments/FN_SA2.py --seed 1 --true_sigma 2e-1 --penalty 1e+00
