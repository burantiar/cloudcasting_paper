"""Train the FNO–CNN hybrid in resumable 55-minute GPU sessions.

Use native-resolution data, masked full-image training MSE and GB validation
MSE with equal sample/channel/lead weighting. Resume automatically from last.pt.
"""

import argparse
import csv
import hashlib
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
from model import FNOCNNForecaster, MODEL_CONFIG


CHANNELS = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
    "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
]

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
    """Request a checkpointed exit after the current batch."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Replace a checkpoint only after the new file is fully written."""
    torch.cuda.synchronize()
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def capture_rng(shuffle_generator):
    """Save random states alongside the training position."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "shuffle": shuffle_generator.get_state(),
    }


def restore_rng(states, shuffle_generator):
    """Restore random states after model and optimizer initialization."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])
    shuffle_generator.set_state(states["shuffle"])


def make_loader(dataset, indices, args):
    """Load the remaining windows in their saved order."""
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Keep loader initialization separate from model and shuffle RNGs.
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )

    return DataLoader(Subset(dataset, indices), **options)


def process_batch(model, optimizer, batch, gb_mask, training):
    """Train on valid image pixels or evaluate equally weighted GB fields."""
    inputs, targets = batch
    inputs = inputs.to("cuda", dtype=torch.float32, non_blocking=True)
    targets = targets.to("cuda", dtype=torch.float32, non_blocking=True)

    # Capture target validity before replacing missing values.
    target_mask = torch.isfinite(targets)
    inputs = torch.where(torch.isfinite(inputs), inputs, MISSING_VALUE)
    targets = torch.where(target_mask, targets, MISSING_VALUE)
    mask = target_mask if training else target_mask & gb_mask

    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad(set_to_none=True)

        # Generate twelve absolute forecasts with connected rollout gradients.
        predictions = model(inputs)
        errors = torch.where(mask, predictions - targets, 0.0).square()

        if training:
            # Full-image MSE across all valid pixels, channels and leads.
            error_sum = errors.sum()
            pixel_count = mask.sum()
            loss = error_sum / pixel_count

            if not torch.isfinite(loss).item():
                raise FloatingPointError("Training loss became non-finite.")

            # Update Fourier and CNN branches through the complete rollout.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
                error_if_nonfinite=True,
            )
            optimizer.step()

            return error_sum.detach().item(), pixel_count.item()

        # Average spatially within each sample/channel/lead first.
        pixel_counts = mask.sum(dim=(-2, -1))
        field_mse = errors.sum(dim=(-2, -1)) / pixel_counts

        if not torch.isfinite(field_mse).all().item():
            raise FloatingPointError(
                "Validation contains an empty or non-finite GB field."
            )

        # Then give every sample, channel and lead equal weight.
        return field_mse.sum().item(), field_mse.numel()


def write_history(history, path):
    """Write completed-epoch metrics without duplicating resumed epochs."""
    temporary = path.with_suffix(".tmp")
    columns = [
        "epoch", "train_mse", "validation_gb_mse", "optimizer_updates",
    ]

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    temporary.replace(path)


def main():
    """Initialize or resume the hybrid and run one time-limited session."""
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

    # Slurm's warning and the session timer request the same orderly exit.
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

    # Match the numerical settings of the other full-resolution FNO models.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.data = args.data.expanduser().resolve()

    # Use the repository's contiguous-window selection and existing values.
    # Preserve missing values; apply no additional scaling or resizing.
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

    # Prevent accidentally using the reduced-resolution Mac cache.
    for dataset in datasets.values():
        spatial_size = (
            dataset.ds.sizes["y_geostationary"],
            dataset.ds.sizes["x_geostationary"],
        )
        if spatial_size != (372, 614):
            raise ValueError("This experiment requires native 372 x 614 data.")

    model = FNOCNNForecaster(**MODEL_CONFIG).to(
        device="cuda", dtype=torch.float32
    )
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_CONFIG)

    # Finite entries define the native-grid GB evaluation region.
    gb_mask = torch.from_numpy(np.isfinite(CUTOUT_MASK)).to("cuda")

    # These scientific settings must match when continuing an existing run.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "model_name": "FNO-CNN hybrid Cloudcasting adaptation",
        "model_class": "FNOCNNForecaster",
        "model_kwargs": MODEL_CONFIG,
        "model_source_sha256": hashlib.sha256(
            Path(__file__).with_name("model.py").read_bytes()
        ).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "additional_cnn_parameters": sum(
            p.numel() for p in model.local_layers.parameters()
        ),
        "parameter_count_convention": "Real scalar parameters",
        "branch_combination": "Spectral + pointwise 1x1 + local 3x3",
        "local_branch_count": MODEL_CONFIG["depth"],
        "initialization": (
            "Standard FNO backbone in its original construction order; "
            "additional CNN branches initialized afterwards"
        ),
        "data": str(args.data),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels": CHANNELS,
        "spatial_size": [372, 614],
        "train_period": TRAIN_PERIOD,
        "validation_period": VAL_PERIOD,
        "test_period": TEST_PERIOD,
        "test_evaluation": False,
        "input_frames": 5,
        "forecast_frames": 12,
        "interval_minutes": 15,
        "forecast_leads_minutes": list(range(15, 181, 15)),
        "preprocessing": "Existing native-grid values; no additional scaling",
        "missing_pixel_marker": MISSING_VALUE,
        "coordinate_channels": "Normalized row and column positions in [0,1]",
        "model_padding": "Zero padding of lifted features, bottom and right",
        "fft_normalization": "ortho",
        "output_formulation": "Absolute next-frame prediction",
        "persistence_residual_output": False,
        "training_metric": "Full-image valid-pixel MSE",
        "validation_metric": (
            "Spatial MSE over valid GB pixels per sample/channel/lead; "
            "then equal weighting across samples, channels and leads"
        ),
        "optimizer": OPTIMIZER_CONFIG,
        "gradient_clip_norm": GRADIENT_CLIP,
        "scheduler": None,
        "early_stopping": False,
        "precision": "float32; complex64 Fourier coefficients",
        "deterministic_algorithms": True,
        "tf32": False,
        "prediction_clipping": False,
        "teacher_forcing": False,
        "forecast_generation": "Twelve one-step predictions with sliding history",
        "backpropagate_through_rollout": True,
        "optimizer_updates_per_batch": 1,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        # Saved indices must continue to identify the same forecast windows.
        "forecast_origin_hashes": {
            name: hashlib.sha256(
                dataset.t0_times.asi8.tobytes()
            ).hexdigest()
            for name, dataset in datasets.items()
        },
    }

    # The default directory belongs to this hybrid model and seed.
    run_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path(__file__).resolve().parent / "runs" / f"full_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    # Position counts completed samples, not samples prefetched by workers.
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
        # Load our own checkpoint, including Python and NumPy random states.
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
        restore_rng(checkpoint["rng"], shuffle_generator)
        del checkpoint

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_history(progress["history"], metrics_path)
    last_saved = time.monotonic()

    def stopping():
        """Check whether the timer or a shutdown signal requests exit."""
        return (
            STOP_REQUESTED
            or time.monotonic() - SESSION_START >= args.session_minutes * 60
        )

    def save_last():
        """Save everything needed to resume after the last completed batch."""
        nonlocal last_saved
        atomic_save(
            {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "progress": progress,
                "rng": capture_rng(shuffle_generator),
            },
            last_path,
        )
        last_saved = time.monotonic()

    print(f"Run: {run_dir}", flush=True)
    print(
        f"Model: FNO-CNN hybrid | GPU: {torch.cuda.get_device_name(0)}"
        f" | seed: {args.seed} | parameters: {config['parameters']:,}",
        flush=True,
    )
    print(
        f"Train: {len(datasets['train']):,}"
        f" | validation: {len(datasets['val']):,} sequences",
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
                # Shuffle whole windows once per epoch; frames remain ordered.
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
                model, optimizer, batch, gb_mask, training
            )

            # Advance only after the complete batch has finished successfully.
            progress[f"{phase}_error"] += error_sum
            progress[f"{phase}_count"] += count
            progress["position"] += batch[0].shape[0]
            if training:
                progress["updates"] += 1

            if (
                progress["position"] % (250 * args.batch_size) == 0
                or progress["position"] == len(order)
            ):
                running_mse = (
                    progress[f"{phase}_error"] / progress[f"{phase}_count"]
                )
                print(
                    f"  {phase}: {progress['position']:,}/{len(order):,}"
                    f" sequences | MSE {running_mse:.6f}",
                    flush=True,
                )

            # Save periodically, in addition to orderly session shutdown.
            if time.monotonic() - last_saved >= args.checkpoint_minutes * 60:
                save_last()

            if stopping():
                break

        # Retain partial phase totals and the sample order for the next session.
        if progress["position"] < len(order):
            break

        if training:
            progress["phase"] = "val"
            progress["position"] = 0
        else:
            epoch = progress["epoch"]
            train_mse = progress["train_error"] / progress["train_count"]
            val_mse = progress["val_error"] / progress["val_count"]

            progress["history"].append({
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_gb_mse": val_mse,
                "optimizer_updates": progress["updates"],
            })

            # Choose best.pt only after completing the entire validation split.
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
                f"Epoch {epoch} complete | train MSE {train_mse:.6f}"
                f" | validation GB MSE {val_mse:.6f}"
                f" | best epoch {progress['best_epoch']}",
                flush=True,
            )

            # Begin the next epoch with fresh totals and a new sample order.
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

        # Save at every training/validation boundary.
        save_last()
        write_history(progress["history"], metrics_path)

    # Save the latest completed batch before the session exits.
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