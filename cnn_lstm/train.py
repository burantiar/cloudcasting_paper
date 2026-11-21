"""Train CNN–LSTM in resumable GPU sessions.

Each invocation runs for approximately 55 minutes.
Later invocations resume automatically from last.pt.
This script does not submit any jobs.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

# Include Python imports and setup in the session's time budget.
SESSION_START = time.monotonic()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from cloudcasting.constants import CUTOUT_MASK
from cloudcasting.dataset import SatelliteDataModule
from model import CNNLSTMForecaster


CHANNELS = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
    "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
]

TRAIN_PERIOD = ["2020-06-01 00:00", "2020-07-31 23:45"]
VAL_PERIOD = ["2020-08-01 00:00", "2020-08-15 23:45"]
TEST_PERIOD = ["2020-08-16 00:00", "2020-08-31 23:45"]

MODEL_CONFIG = {
    "channels": 11,
    "spatial_size": (372, 614),
    "embedding_dim": 64,
    "hidden_dim": 128,
    "forecast_steps": 12,
}

OPTIMIZER_CONFIG = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "foreach": False,
}

MISSING_VALUE = -1.0
CLIP_NORM = 1.0
STOP_REQUESTED = False


def request_stop(signum, frame):
    """Handle a scheduler warning by stopping after the current batch."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Keep the previous checkpoint until its replacement is fully written."""
    torch.cuda.synchronize()
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def capture_rng(shuffle_generator):
    """Capture initialization, model and sequence-shuffling random states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "shuffle": shuffle_generator.get_state(),
    }


def restore_rng(states, shuffle_generator):
    """Restore random states after loading the model and optimizer."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])
    shuffle_generator.set_state(states["shuffle"])


def make_loader(dataset, remaining_indices, args):
    """Load the remaining windows in their saved order, without reshuffling."""
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Worker startup must not consume the model's random generator.
        "generator": torch.Generator().manual_seed(args.seed),
    }

    if args.workers > 0:
        # Spawn avoids forking a process that has already initialized CUDA.
        kwargs.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )

    return DataLoader(
        Subset(dataset, remaining_indices),
        **kwargs,
    )


def process_batch(model, optimizer, batch, gb_mask, training):
    """Process one batch and return the error sum and its aggregation count."""
    inputs, targets = [
        tensor.to("cuda", dtype=torch.float32, non_blocking=True)
        for tensor in batch
    ]

    # Capture missingness before replacing missing values.
    valid = torch.isfinite(targets)
    inputs = torch.where(torch.isfinite(inputs), inputs, MISSING_VALUE)
    targets = torch.where(valid, targets, MISSING_VALUE)
    mask = valid if training else valid & gb_mask

    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(inputs)
        errors = (
            torch.where(mask, predictions, 0.0)
            - torch.where(mask, targets, 0.0)
        ).square()

        if training:
            # Full-image MSE, averaged over valid target pixels.
            error_sum = errors.sum()
            count = mask.sum()
            loss = error_sum / count

            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=CLIP_NORM,
                error_if_nonfinite=True,
            )
            optimizer.step()

            return error_sum.detach().item(), count.item()

        # GB spatial MSE per sample/channel/lead, followed by an equal mean.
        counts = mask.sum(dim=(-2, -1))
        if (counts == 0).any():
            raise ValueError(
                "A validation sample/channel/lead has no valid GB pixels."
            )

        field_mse = errors.sum(dim=(-2, -1)) / counts
        if not torch.isfinite(field_mse).all():
            raise FloatingPointError("Validation MSE became non-finite.")

        return field_mse.sum().item(), field_mse.numel()


def write_history(history, path):
    """Rebuild the metrics file so resumed jobs do not duplicate epoch rows."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "train_mse",
                "validation_gb_mse",
                "optimizer_updates",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def main():
    """Initialize or resume one training run, then execute a timed session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-minutes", type=float, default=55)
    parser.add_argument("--checkpoint-minutes", type=float, default=5)
    args = parser.parse_args()

    # The submission script will enable the advance SIGUSR1 warning.
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run inside a GPU allocation with a "
            "compatible PyTorch/CUDA environment."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    # Keep float32 execution and deterministic settings aligned across runs.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Cloudcasting selects complete windows within each split.
    # We construct our own loaders afterward to support a saved sample position.
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
    train_dataset = data_module.train_dataloader().dataset
    val_dataset = data_module.val_dataloader().dataset
    val_dataset.t0_times = val_dataset.t0_times.sort_values()

    model = CNNLSTMForecaster(**MODEL_CONFIG).to("cuda")
    optimizer = torch.optim.Adam(
        model.parameters(), **OPTIMIZER_CONFIG
    )
    gb_mask = torch.from_numpy(
        np.isfinite(CUTOUT_MASK)
    ).to("cuda")

    # Scientific settings must match when resuming.
    # Runtime settings such as session duration and worker count may change.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "data": str(args.data.resolve()),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels": CHANNELS,
        "train_period": TRAIN_PERIOD,
        "validation_period": VAL_PERIOD,
        "test_period": TEST_PERIOD,
        "test_evaluation": False,
        "history_minutes": 60,
        "forecast_minutes": 180,
        "interval_minutes": 15,
        "model": MODEL_CONFIG,
        "model_architecture": str(model),
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": OPTIMIZER_CONFIG,
        "gradient_clip_norm": CLIP_NORM,
        "missing_value": MISSING_VALUE,
        "preprocessing": "Native grid; existing scaling; no resampling",
        "training_metric": "Full-image valid-pixel MSE",
        "validation_metric": "GB spatial MSE, then equal sample/channel/lead mean",
        "precision": "float32",
        "deterministic_algorithms": True,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        # Detect changes to the windows or their indexing before a resume.
        "forecast_origin_hashes": {
            name: hashlib.sha256(dataset.t0_times.asi8.tobytes()).hexdigest()
            for name, dataset in [
                ("train", train_dataset),
                ("validation", val_dataset),
            ]
        },
    }

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent
            / "runs"
            / f"full_seed{args.seed}"
        )
    output.mkdir(parents=True, exist_ok=True)

    last_path = output / "last.pt"
    best_path = output / "best.pt"
    metrics_path = output / "metrics.csv"
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    # Position is the NEXT sample in the current phase's order.
    # Training counts valid pixels; validation counts sample/channel/lead fields.
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
        # Load our own checkpoint, including Python and NumPy RNG states.
        checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        if checkpoint["config"] != config:
            raise ValueError(
                "Checkpoint configuration differs. Restore the original "
                "settings or choose a new --output directory."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = checkpoint["progress"]
        restore_rng(checkpoint["rng"], shuffle_generator)
        del checkpoint

    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    write_history(progress["history"], metrics_path)

    last_saved = time.monotonic()

    def save_last():
        """Save all state required to continue at the next unprocessed window."""
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

    def stopping():
        """Stop on a scheduler warning or when the soft time budget expires."""
        elapsed = time.monotonic() - SESSION_START
        return STOP_REQUESTED or elapsed >= args.session_minutes * 60

    print(
        f"Job {os.environ.get('SLURM_JOB_ID', 'interactive')} | "
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}",
        flush=True,
    )
    print(
        f"Training: {len(train_dataset):,} | "
        f"Validation: {len(val_dataset):,} | "
        f"Parameters: {config['parameters']:,}",
        flush=True,
    )
    print(
        f"Starting at epoch {progress['epoch']}, "
        f"phase {progress['phase']}, "
        f"sample position {progress['position']}.",
        flush=True,
    )

    while progress["epoch"] <= args.epochs:
        if stopping():
            break

        phase = progress["phase"]
        training = phase == "train"
        dataset = train_dataset if training else val_dataset

        if training and progress["order"] is None:
            # Shuffle whole windows once per epoch, then retain that order.
            progress["order"] = list(
                RandomSampler(
                    train_dataset,
                    generator=shuffle_generator,
                )
            )

        order = (
            progress["order"]
            if training
            else list(range(len(val_dataset)))
        )
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

            # Advance only after the whole batch has completed successfully.
            progress[f"{phase}_error"] += error_sum
            progress[f"{phase}_count"] += count
            progress["position"] += len(batch[0])
            if training:
                progress["updates"] += 1

            position = progress["position"]
            if (
                position % (250 * args.batch_size) == 0
                or position == len(order)
            ):
                running_mse = (
                    progress[f"{phase}_error"]
                    / progress[f"{phase}_count"]
                )
                print(
                    f"  {phase}: {position:,}/{len(order):,} sequences"
                    f" | MSE {running_mse:.6f}",
                    flush=True,
                )

            if (
                time.monotonic() - last_saved
                >= args.checkpoint_minutes * 60
            ):
                save_last()

            if stopping():
                break

        # A partial phase is preserved and continued by the next job.
        if progress["position"] < len(order):
            break

        if training:
            # Training is complete for this epoch; validation starts next.
            progress["phase"] = "val"
            progress["position"] = 0

        else:
            # Only a COMPLETE validation split can select the best checkpoint.
            epoch = progress["epoch"]
            train_mse = progress["train_error"] / progress["train_count"]
            val_mse = progress["val_error"] / progress["val_count"]

            progress["history"].append({
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_gb_mse": val_mse,
                "optimizer_updates": progress["updates"],
            })

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

            # Reset only the counters belonging to the completed epoch.
            progress.update(
                epoch=epoch + 1,
                phase="train",
                position=0,
                order=None,
                train_error=0.0,
                train_count=0,
                val_error=0.0,
                val_count=0,
            )

        save_last()
        write_history(progress["history"], metrics_path)

    # Normal session endings save the final completed batch, not just an epoch.
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
            f"\nSession saved at epoch {progress['epoch']}, "
            f"phase {progress['phase']}, "
            f"sample position {progress['position']}.",
            flush=True,
        )
        print(
            "The next job using this output directory will resume here.",
            flush=True,
        )

    print("Resume checkpoint:", last_path, flush=True)
    print("Epoch metrics:", metrics_path, flush=True)


if __name__ == "__main__":
    main()