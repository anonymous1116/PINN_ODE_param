#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --array=0-99
#SBATCH --partition=v100
#SBATCH --output=output_log_training/output_log_%A_%a.out
#SBATCH --error=output_log_training/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
# ##SBATCH -w gilbreth-h[000-015]

mkdir -p output_log_training

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/ODE

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/PINN_ODE_parm
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID + 1))
#dim_out=$((SLURM_ARRAY_TASK_ID % 10 + 1))

#echo "Running with seed=$dim_out, dim_out=10, task=$TASK, N_EPOCHS=$N_EPOCHS, layer_len=$layer_len, num_training=$num_training"
#seed=$((SLURM_ARRAY_TASK_ID % 10 + 1)) # ones digit 1, 2, 3
#python ./Experiments/FN.py 
#python ./benchmark/benchmark_training.py --num_training 100000 --seed 1 --task "cont_table" --N_EPOCHS 1 --layer_len 256 

python ./Experiments/FN2.py --seed $seed
