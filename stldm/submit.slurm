#!/bin/bash -l

#SBATCH --job-name=stldm
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00

# Request graceful shutdown three minutes before the time limit.
#SBATCH --signal=B:USR1@180

#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/stldm
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/stldm/stldm-%j.out

# Stop if environment setup or training fails.
set -eo pipefail

module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Avoid competing CPU thread pools in the data-loading workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Support the trainer's deterministic CUDA settings.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Replace the shell so Python receives the advance shutdown signal.
# Keep this output directory unchanged to resume from its last.pt.
exec python -u train.py \
    --data /gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr \
    --output /data/home/bsw887/projects/cloudcasting/apocrita/stldm/runs/full_seed0 \
    --seed 0 \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5