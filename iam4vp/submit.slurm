#!/bin/bash -l
#SBATCH --job-name=iam4vp
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/iam4vp
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/iam4vp/iam4vp-%j.out

# Train one IAM4VP session using the existing Cloudcasting environment.
set -eo pipefail

module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Avoid CPU thread oversubscription and preserve deterministic CUDA settings.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Replace the batch shell so the advance shutdown signal reaches Python.
# The trainer stops after approximately 55 minutes and resumes from last.pt.
exec python -u train.py \
    --data /gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr \
    --output /data/home/bsw887/projects/cloudcasting/apocrita/iam4vp/runs/full_seed0 \
    --seed 0 \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5