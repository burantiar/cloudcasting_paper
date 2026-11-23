#!/bin/bash -l
# Run one resumable, full-resolution diffusion training session.

#SBATCH --job-name=diffusion
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/diffusion
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/diffusion/diffusion-%j.out

# Stop if environment setup fails.
set -eo pipefail

# Load the established Python environment.
# It must contain diffusers==0.30.3 and huggingface-hub==0.25.1.
module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Limit numerical-library threads; the dataloader uses four workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Support deterministic CUDA matrix operations.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Replace the shell so Python receives the advance shutdown signal.
# The trainer finishes its current batch, saves progress and exits.
# Reusing this output directory automatically resumes last.pt.
# Twenty epochs is the total training budget across all submissions.
exec python -u train.py \
    --data /gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr \
    --output /data/home/bsw887/projects/cloudcasting/apocrita/diffusion/runs/full_seed0 \
    --seed 0 \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5