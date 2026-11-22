#!/bin/bash -l
# Run one resumable, full-resolution FNO training session.

#SBATCH --job-name=fno
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/fno
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/fno/fno-%j.out

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

# The trainer requests shutdown after 55 minutes and saves after its batch.
# Slurm also sends USR1 three minutes before the one-hour limit.
# exec replaces the shell so Python receives the signal directly.
#
# Keep the output directory fixed: subsequent submissions resume last.pt.
exec python -u train.py \
    --data /gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr \
    --output /data/home/bsw887/projects/cloudcasting/apocrita/fno/runs/full_seed0 \
    --seed 0 \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5