#!/bin/bash -l
# Run one resumable, full-resolution SimVP training session.

#SBATCH --job-name=simvp
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/simvp
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/simvp/simvp-%j.out

# Stop if environment setup fails.
set -eo pipefail

module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Limit CPU-library threads alongside the data-loading workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Required for deterministic CUDA execution.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

SIMVP_SEED=0
SIMVP_DATA="/gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr"

# Keep this directory unchanged between continuation submissions.
SIMVP_RUN_DIR="/data/home/bsw887/projects/cloudcasting/apocrita/simvp/runs/full_seed${SIMVP_SEED}"

# Launch one Python trainer.
# exec replaces the batch shell, allowing Python to receive its USR1 signal.
exec python -u train.py \
    --data "$SIMVP_DATA" \
    --output "$SIMVP_RUN_DIR" \
    --seed "$SIMVP_SEED" \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5