#!/bin/bash -l
# Run one resumable CNN–LSTM session on a GPU.

#SBATCH --job-name=cnn-lstm
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/cnn_lstm
#SBATCH --output=cnn-lstm-%j.out

# Stop if environment setup fails.
set -eo pipefail

module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Limit CPU-library threads alongside the four data-loading workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Required by the deterministic CUDA configuration in train.py.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

CNN_SEED=0
CNN_DATA="/gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr"

# Keep this directory unchanged between continuation jobs.
CNN_RUN_DIR="/data/home/bsw887/projects/cloudcasting/apocrita/cnn_lstm/runs/full_seed${CNN_SEED}"

# Launch one Python trainer.
# exec replaces the batch shell so Python receives the B:USR1 warning directly.
exec python -u train.py \
    --data "$CNN_DATA" \
    --output "$CNN_RUN_DIR" \
    --seed "$CNN_SEED" \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5