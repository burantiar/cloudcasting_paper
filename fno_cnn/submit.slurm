#!/bin/bash -l
# Run one resumable, full-resolution FNO–CNN hybrid training session.

#SBATCH --job-name=fno-cnn
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/fno_cnn
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/fno_cnn/fno-cnn-%j.out

# Stop if environment setup or a command fails.
set -eo pipefail

# Load the established Cloudcasting environment.
module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Limit library threads alongside the data-loading workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Support deterministic CUDA matrix operations.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Request shutdown after 55 minutes, finishing the current batch before saving.
# Slurm also sends USR1 three minutes before the one-hour limit.
# exec replaces the shell so Python receives the signal directly.
#
# Keep this hybrid's output directory fixed to resume last.pt automatically.
exec python -u train.py \
    --data /gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr \
    --output /data/home/bsw887/projects/cloudcasting/apocrita/fno_cnn/runs/full_seed0 \
    --seed 0 \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5