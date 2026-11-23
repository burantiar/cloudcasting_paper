#!/bin/bash -l
# Run one resumable residual-diffusion session with IAM4VP frozen.

#SBATCH --job-name=residual-diffusion
#SBATCH --partition=gpushort
#SBATCH --constraint="ampere|hopper"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/data/home/bsw887/projects/cloudcasting/apocrita/residual_diffusion
#SBATCH --output=/data/home/bsw887/projects/cloudcasting/apocrita/residual_diffusion/residual-diffusion-%j.out

# Stop if environment setup fails.
set -eo pipefail

# Use the established environment containing the diffusion dependencies.
# Expected: diffusers==0.30.3 and huggingface-hub==0.25.1.
module load python/3.11.7-gcc-12.2.0
source /data/home/bsw887/projects/cloudcasting/.venv/bin/activate

# Limit numerical-library threads; data loading uses four worker processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Replace the shell so Python receives the advance shutdown signal.
# The trainer requests shutdown after 55 minutes, finishes the current batch
# and saves progress. Slurm also sends USR1 three minutes before its limit.
#
# The first session pins the supplied full-resolution IAM4VP checkpoint.
# Later sessions restore that same frozen predictor from this run's last.pt.
# Reuse this output directory to continue toward 20 total denoiser epochs.
exec python -u train.py \
    --data /gpfs/scratch/bsw887/cloudcasting/jun_aug_2020/2020_training_nonhrv.zarr \
    --output /data/home/bsw887/projects/cloudcasting/apocrita/residual_diffusion/runs/full_seed0 \
    --predictor /data/home/bsw887/projects/cloudcasting/apocrita/iam4vp/runs/full_seed0/best.pt \
    --predictor-source /data/home/bsw887/projects/cloudcasting/apocrita/iam4vp/model.py \
    --seed 0 \
    --epochs 20 \
    --batch-size 1 \
    --workers 4 \
    --session-minutes 55 \
    --checkpoint-minutes 5
    