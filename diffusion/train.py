"""Train diffusion in resumable 55-minute Apocrita GPU sessions.

Training: masked full-image L1 noise-prediction loss.
Validation: GB image MSE with equal sample/channel/lead weighting.
Resume automatically from last.pt in the selected output directory.
"""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import signal
import time

# Include imports and dataset preparation in the session budget.
SESSION_START = time.monotonic()
STOP_REQUESTED = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from cloudcasting.constants import CUTOUT_MASK
from cloudcasting.dataset import SatelliteDataModule
from model import DiffusionForecaster, MODEL_CONFIG


CHANNELS = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
    "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
]

SPATIAL_SIZE = (372, 614)
TRAIN_PERIOD = ["2020-06-01 00:00", "2020-07-31 23:45"]
VAL_PERIOD = ["2020-08-01 00:00", "2020-08-15 23:45"]
TEST_PERIOD = ["2020-08-16 00:00", "2020-08-31 23:45"]

OPTIMIZER_CONFIG = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "foreach": False,
}

MISSING_VALUE = -1.0
GRADIENT_CLIP = 1.0


def request_stop(signum, frame):
    """Request an orderly exit after the current batch."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Replace a checkpoint only after its new file is fully written."""
    torch.cuda.synchronize()
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def capture_rng(shuffle_generator, validation_generator):
    """Capture training, shuffle and validation-noise random states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "shuffle": shuffle_generator.get_state(),
        "validation": validation_generator.get_state(),
    }


def restore_rng(states, shuffle_generator, validation_generator):
    """Restore random states after constructing the model and optimizer."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])
    shuffle_generator.set_state(states["shuffle"])
    validation_generator.set_state(states["validation"])


def make_loader(dataset, indices, args):
    """Load remaining windows in their saved order."""
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Loader initialization must not consume model or sampling randomness.
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )

    return DataLoader(Subset(dataset, indices), **options)


def process_batch(
    model,
    optimizer,
    batch,
    gb_mask,
    training,
    validation_generator,
):
    """Train on noise L1 or evaluate generated forecasts using GB MSE."""
    inputs, targets = batch
    inputs = inputs.to("cuda", dtype=torch.float32, non_blocking=True)
    targets = targets.to("cuda", dtype=torch.float32, non_blocking=True)

    # Preserve target validity before replacing missing values.
    target_mask = torch.isfinite(targets)
    inputs = torch.where(torch.isfinite(inputs), inputs, MISSING_VALUE)
    targets = torch.where(target_mask, targets, MISSING_VALUE)

    with torch.set_grad_enabled(training):
        if training:
            pixel_count = target_mask.sum().item()
            if pixel_count == 0:
                raise ValueError("Training batch has no valid target pixels.")

            optimizer.zero_grad(set_to_none=True)

            # The model applies its internal scaling and predicts target noise.
            # All twelve forecast frames contribute to one optimizer update.
            loss = model.denoising_loss(inputs, targets, target_mask)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Training noise loss became non-finite.")

            loss_value = loss.detach().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
                error_if_nonfinite=True,
                foreach=False,
            )
            optimizer.step()

            # Accumulate a valid-pixel-weighted epoch training loss.
            return loss_value * pixel_count, pixel_count

        # Sampling uses history only and returns the original data scale.
        predictions = model.sample(
            inputs, generator=validation_generator
        )
        mask = target_mask & gb_mask

        errors = torch.where(
            mask, predictions - targets, 0.0
        ).square()

        # Spatial averaging first: one MSE for every sample/channel/lead.
        pixel_counts = mask.sum(dim=(-2, -1))
        field_mse = errors.sum(dim=(-2, -1)) / pixel_counts

        if not torch.isfinite(field_mse).all().item():
            raise FloatingPointError(
                "Validation contains an empty or non-finite GB field."
            )

        # Equal weighting across all sample/channel/lead fields.
        return field_mse.sum().item(), field_mse.numel()


def write_history(history, path):
    """Write completed-epoch metrics without duplicating resumed epochs."""
    temporary = path.with_suffix(".tmp")
    columns = [
        "epoch",
        "train_noise_l1",
        "validation_gb_mse",
        "optimizer_updates",
    ]

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    temporary.replace(path)


def main():
    """Initialize or resume diffusion and run one time-limited session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-minutes", type=float, default=55)
    parser.add_argument("--checkpoint-minutes", type=float, default=5)
    args = parser.parse_args()

    # Both Slurm signals request a checkpointed exit at a batch boundary.
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside an Apocrita GPU allocation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)

    # Match the numerical settings of the other full-resolution experiments.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.data = args.data.expanduser().resolve()

    # Use the repository's complete, contiguous sequence selection.
    # No additional loader normalization, resizing or missing-value filling.
    data_module = SatelliteDataModule(
        zarr_path=str(args.data),
        history_mins=60,
        forecast_mins=180,
        sample_freq_mins=15,
        batch_size=args.batch_size,
        num_workers=0,
        variables=CHANNELS,
        train_period=TRAIN_PERIOD,
        val_period=VAL_PERIOD,
        test_period=TEST_PERIOD,
        nan_to_num=False,
    )
    datasets = {
        "train": data_module.train_dataloader().dataset,
        "val": data_module.val_dataloader().dataset,
    }

    # Restore chronological validation order after the repository's preshuffle.
    datasets["val"].t0_times = datasets["val"].t0_times.sort_values()

    # Prevent accidentally training on the reduced-resolution Mac cache.
    for dataset in datasets.values():
        spatial_size = (
            dataset.ds.sizes["y_geostationary"],
            dataset.ds.sizes["x_geostationary"],
        )
        if spatial_size != SPATIAL_SIZE:
            raise ValueError("This experiment requires native 372 x 614 data.")

    model = DiffusionForecaster(**MODEL_CONFIG).to(
        device="cuda", dtype=torch.float32
    )
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_CONFIG)

    # Native GB evaluation mask; no spatial resampling.
    gb_mask = torch.from_numpy(np.isfinite(CUTOUT_MASK)).to("cuda")

    # Freeze scientific settings across submissions.
    # Worker counts and session duration may change without changing the run.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "model_name": "ocf-diffusion compact joint-forecast adaptation",
        "model_class": "DiffusionForecaster",
        "model_kwargs": MODEL_CONFIG,
        "model_source_sha256": hashlib.sha256(
            Path(__file__).with_name("model.py").read_bytes()
        ).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "reference_repository": (
            "https://github.com/alan-turing-institute/ocf-diffusion"
        ),
        "reference_commit": "3ea38c28f8d43f9c45a2bda9cba950199f149f83",
        "source_is_exact_reproduction": False,
        "data": str(args.data),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels": CHANNELS,
        "spatial_size": list(SPATIAL_SIZE),
        "train_period": TRAIN_PERIOD,
        "validation_period": VAL_PERIOD,
        "test_period": TEST_PERIOD,
        "test_evaluation": False,
        "input_frames": 5,
        "forecast_frames": 12,
        "interval_minutes": 15,
        "forecast_leads_minutes": list(range(15, 181, 15)),
        "preprocessing": "Existing native-grid values; no loader rescaling",
        "missing_pixel_marker": MISSING_VALUE,
        "model_value_transform": "2*x - 1; inverse (x + 1)/2",
        "conditioning_missing_value_after_transform": -3.0,
        "missing_target_fill_in_model_space": 0.0,
        "model_padding": (
            "Zero-pad U-Net inputs to multiples of four; "
            "crop predicted noise back to native dimensions"
        ),
        "training_metric": "Full-image valid-pixel L1 noise prediction",
        "validation_metric": (
            "Spatial MSE over valid GB pixels per sample/channel/lead; "
            "then equal weighting across samples, channels and leads"
        ),
        "diffusion": {
            "training_noise_levels": MODEL_CONFIG["train_steps"],
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "sampler": "DDIM",
            "sampling_steps": MODEL_CONFIG["sample_steps"],
            "timestep_spacing": "leading",
            "eta": 0.0,
            "clip_sample": False,
            "ema": False,
        },
        "validation_samples_per_origin": 1,
        "validation_sampling_seed": args.seed,
        "validation_noise_policy": (
            "Reset CPU generator at each new validation pass; "
            "restore its saved state when resuming a partial pass"
        ),
        "forecast_generation": "Joint twelve-frame conditional diffusion",
        "teacher_forcing": False,
        "prediction_clipping": False,
        "load_pretrained_weights": False,
        "optimizer": OPTIMIZER_CONFIG,
        "optimizer_updates_per_batch": 1,
        "gradient_clip_norm": GRADIENT_CLIP,
        "scheduler": None,
        "early_stopping": False,
        "precision": "float32",
        "deterministic_algorithms": True,
        "tf32": False,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "diffusers_version": version("diffusers"),
        "huggingface_hub_version": version("huggingface-hub"),
        # Saved indices must identify the same windows on every submission.
        "forecast_origin_hashes": {
            name: hashlib.sha256(
                dataset.t0_times.asi8.tobytes()
            ).hexdigest()
            for name, dataset in datasets.items()
        },
    }

    run_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path(__file__).resolve().parent / "runs" / f"full_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    # Keep sample shuffling and validation noise independent.
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Position counts completed samples, never prefetched samples.
    progress = {
        "epoch": 1,
        "phase": "train",
        "position": 0,
        "order": None,
        "train_error": 0.0,
        "train_count": 0,
        "val_error": 0.0,
        "val_count": 0,
        "updates": 0,
        "best_mse": float("inf"),
        "best_epoch": 0,
        "history": [],
    }

    if last_path.exists():
        # This is our own checkpoint, including Python and NumPy RNG states.
        checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        if checkpoint["config"] != config:
            raise ValueError(
                "Saved experiment settings differ. Restore the original "
                "settings or choose a new --output directory."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = checkpoint["progress"]
        restore_rng(
            checkpoint["rng"],
            shuffle_generator,
            validation_generator,
        )
        del checkpoint

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_history(progress["history"], metrics_path)
    last_saved = time.monotonic()

    def stopping():
        """Check whether the session timer or a signal requests shutdown."""
        return (
            STOP_REQUESTED
            or time.monotonic() - SESSION_START >= args.session_minutes * 60
        )

    def save_last():
        """Save weights, optimizer, phase progress and all random states."""
        nonlocal last_saved
        atomic_save(
            {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "progress": progress,
                "rng": capture_rng(
                    shuffle_generator, validation_generator
                ),
            },
            last_path,
        )
        last_saved = time.monotonic()

    print(f"Run: {run_dir}", flush=True)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | seed: {args.seed}"
        f" | parameters: {config['parameters']:,}",
        flush=True,
    )
    print(
        f"Train: {len(datasets['train']):,}"
        f" | validation: {len(datasets['val']):,} sequences",
        flush=True,
    )
    print(
        f"Sampling: {MODEL_CONFIG['sample_steps']} DDIM steps"
        " for twelve joint forecast frames",
        flush=True,
    )
    print(
        f"Starting at epoch {progress['epoch']}, phase {progress['phase']}, "
        f"sample position {progress['position']}.",
        flush=True,
    )

    while progress["epoch"] <= args.epochs:
        if stopping():
            break

        phase = progress["phase"]
        training = phase == "train"
        dataset = datasets[phase]

        if training:
            if progress["order"] is None:
                # Shuffle complete windows once per epoch.
                progress["order"] = list(
                    RandomSampler(dataset, generator=shuffle_generator)
                )
            order = progress["order"]
        else:
            order = list(range(len(dataset)))

        remaining = order[progress["position"]:]
        loader = make_loader(dataset, remaining, args) if remaining else []
        model.train(training)

        print(
            f"\nEpoch {progress['epoch']}/{args.epochs} — {phase}",
            flush=True,
        )

        for batch in loader:
            if stopping():
                break

            error_sum, count = process_batch(
                model,
                optimizer,
                batch,
                gb_mask,
                training,
                validation_generator,
            )

            # Advance only after a complete update or validation forecast.
            progress[f"{phase}_error"] += error_sum
            progress[f"{phase}_count"] += count
            progress["position"] += batch[0].shape[0]
            if training:
                progress["updates"] += 1

            report_every = (250 if training else 100) * args.batch_size
            if (
                progress["position"] % report_every == 0
                or progress["position"] == len(order)
            ):
                running_metric = (
                    progress[f"{phase}_error"] / progress[f"{phase}_count"]
                )
                label = "noise L1" if training else "GB MSE"
                print(
                    f"  {phase}: {progress['position']:,}/{len(order):,}"
                    f" sequences | {label} {running_metric:.6f}",
                    flush=True,
                )

            # Periodic checkpoints supplement the orderly shutdown checkpoint.
            if time.monotonic() - last_saved >= args.checkpoint_minutes * 60:
                save_last()

            if stopping():
                break

        # Keep partial metrics, the saved order and generator states intact.
        if progress["position"] < len(order):
            break

        if training:
            progress["phase"] = "val"
            progress["position"] = 0

            # Reset only when starting a NEW validation pass.
            # A resumed validation pass retains its restored generator state.
            validation_generator.manual_seed(args.seed)
        else:
            epoch = progress["epoch"]
            train_l1 = progress["train_error"] / progress["train_count"]
            val_mse = progress["val_error"] / progress["val_count"]

            progress["history"].append({
                "epoch": epoch,
                "train_noise_l1": train_l1,
                "validation_gb_mse": val_mse,
                "optimizer_updates": progress["updates"],
            })

            # Select the best model only after full validation is complete.
            if val_mse < progress["best_mse"]:
                progress["best_mse"] = val_mse
                progress["best_epoch"] = epoch
                atomic_save(
                    {
                        "config": config,
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation_gb_mse": val_mse,
                    },
                    best_path,
                )

            print(
                f"Epoch {epoch} complete | train noise L1 {train_l1:.6f}"
                f" | validation GB MSE {val_mse:.6f}"
                f" | best epoch {progress['best_epoch']}",
                flush=True,
            )

            # Start the next epoch with fresh totals and a new training order.
            progress.update({
                "epoch": epoch + 1,
                "phase": "train",
                "position": 0,
                "order": None,
                "train_error": 0.0,
                "train_count": 0,
                "val_error": 0.0,
                "val_count": 0,
            })

        # Save at each training/validation boundary.
        save_last()
        write_history(progress["history"], metrics_path)

    # Preserve the last completed batch before ending the session.
    save_last()
    write_history(progress["history"], metrics_path)

    if progress["epoch"] > args.epochs:
        print(
            f"\nTraining complete. Best epoch: {progress['best_epoch']}"
            f" | validation GB MSE: {progress['best_mse']:.6f}",
            flush=True,
        )
    else:
        print(
            f"\nSession saved: epoch {progress['epoch']}, "
            f"phase {progress['phase']}, "
            f"sample position {progress['position']}.",
            flush=True,
        )
        print("The next submission using this output directory resumes here.")

    print("Resume checkpoint:", last_path)
    print("Completed-epoch metrics:", metrics_path)
    if progress["best_epoch"] > 0:
        print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()