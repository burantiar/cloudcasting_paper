#!/bin/bash -l
# Train the full-resolution ConvLSTM using the existing Apocrita environment.
#SBATCH --job-name=convlstm
#SBATCH --partition=gpushort
#SBATCH --constraint=hopper
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=convlstm-%j.out
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita

set -eo pipefail

# Use the existing environment; no package installation inside the job.
module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate
DATA_PATH=/gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr

# One Python process uses the GPU; DataLoader workers use the allocated CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

# With --array=0-4, each task trains one of the five planned seeds.
SEED=${SLURM_ARRAY_TASK_ID:-0}
exec python train.py --data "$DATA_PATH" --seed "$SEED" "$@"
