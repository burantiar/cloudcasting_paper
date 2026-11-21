ConvLSTM on Apocrita
===================

Files
  model.py           Notebook ConvLSTM: separate encoder/decoder, 16 hidden channels.
  train.py           Native-resolution training and validation; optional resume.
  config.json        Dates and training settings. Edit before a new run.
  submit.slurm       One Hopper GPU on gpushort, using Natalia's existing environment.

Protocol
  Full 372 x 614 images, all 11 channels, no resizing or cropping of model inputs.
  Five input frames (60-minute history), twelve forecasts (+15 to +180 minutes).
  Cloudcasting SatelliteDataModule selects complete sequences within each split.
  Training: June-July 2020. Validation: 1-15 August. Test: 16-31 August, unused here.
  Missing inputs/targets are filled with -1 after recording target validity.
  Training uses masked full-image MSE. Validation uses the native GB scoring mask.
  Validation averages spatial MSE separately for each sample/channel/lead, then
  gives equal weight to samples, channels, and leads. Lowest validation MSE wins.
  Batch size 1, Adam 0.001, clipping 1.0, 20 epochs, float32, no TF32.
  No scheduler or early stopping. Seed 0 by default; seeds 0-4 supported.
  These scripts start fresh full-resolution runs; they do not reuse exploratory
  low-resolution weights. Final test MAE/MSE/SSIM evaluation is a separate step.

Upload this directory from your Mac
  # Copy the local scripts into the repository on Apocrita.
  scp -i ~/.ssh/apocrita_ed25519 -r \
    /Users/natalia/Documents/Code/cloudcasting/apocrita \
    bsw887@login.hpc.qmul.ac.uk:/data/home/bsw887/projects/cloudcasting/

Launch on Apocrita
  # Submit the job yourself from the uploaded directory.
  cd /data/home/bsw887/projects/cloudcasting/apocrita
  sbatch submit.slurm

  Natalia confirmed the June-August download completed on 20 November 2026.
  No download-job dependency is needed now.
  The script uses the final dataset directory, never the unfinished .partial store.

Code style
  Keep future Python code commented and give modules, classes and functions concise
  docstrings. Use explanatory comments for shell commands and job scripts.

GPU access
  gpushort is limited to one hour. The current launcher requests one Hopper GPU
  because that partition lists H100s and the installed PyTorch includes CUDA 13.
  GPU execution and driver compatibility have not yet been demonstrated.
  This is a real ConvLSTM training job. It saves every completed epoch; an unfinished
  epoch must be repeated after a timeout. A complete 20-epoch run may need more than
  one short job. Longer GPU access requires approval from QMUL Research-IT.
  Once approved, override the partition and runtime without editing the script:
  sbatch --partition=YOUR_APPROVED_PARTITION --time=24:00:00 submit.slurm
  Override --constraint too if that allocation does not contain Hopper GPUs.

Resume after interruption or the wall-time limit
  sbatch submit.slurm --resume /absolute/path/to/run/last.pt
  Resume uses the saved configuration, seed and output directory, and starts at
  the next epoch. The --data option can point to the same dataset in a new location.
  Load only your own checkpoints. Earlier notebook checkpoints use a different
  format and are not inputs to this script's resume command.

Five seeds, at most one running at a time (after longer GPU access is approved)
  sbatch --partition=YOUR_APPROVED_PARTITION --time=24:00:00 \
    --array=0-4%1 submit.slurm

Outputs
  runs/convlstm_seed<seed>_<UTC timestamp>/
    config.json   Settings, sample counts, software versions and GPU.
    metrics.csv   One row per epoch: training MSE, GB validation MSE, seconds.
    best.pt       Checkpoint with the lowest GB validation MSE.
    last.pt       Latest completed epoch, including optimizer and RNG states.
  convlstm-<jobid>.out contains brief Slurm progress output.

The job loads python/3.11.7-gcc-12.2.0 and activates
/data/home/bsw887/projects/cloudcasting/.venv. Keep the working Xarray/Zarr
environment; no dependency changes are needed just to transfer these scripts.

If downloading different dates later, edit the date splits in config.json before
a new run and apply those same splits across all models.

Verification performed locally
  Python and shell syntax, command-line interface, exact notebook-model equivalence,
  masked validation aggregation with uneven batches, CPU backward/optimizer steps,
  and checkpoint/shuffle-state restoration passed using tiny synthetic inputs.
  No Apocrita connection or GPU training was performed during preparation.

References
  https://docs.hpc.qmul.ac.uk/using/usingGPU/
  https://docs.hpc.qmul.ac.uk/apps/languages/python/
