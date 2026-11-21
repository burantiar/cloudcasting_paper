"""Train either residual ConvLSTM using resumable, time-limited GPU sessions."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

# Include Python imports and dataset preparation in the session budget.
SESSION_START = time.monotonic()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from cloudcasting.dataset import SatelliteDataModule
from cloudcasting.constants import CUTOUT_MASK
from model import MODEL_CONFIGS, build_model


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
STOP_REQUESTED = False


def request_stop(signum, frame):
    """Request a checkpoint and exit after the current batch finishes."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Preserve the previous checkpoint until its replacement is fully written."""
    torch.cuda.synchronize()
    temporary_path = path.with_suffix(".tmp")
    torch.save(contents, temporary_path)
    temporary_path.replace(path)


def capture_rng(shuffle_generator):
    """Collect random states needed to continue the same experiment."""
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


def make_loader(dataset, indices, args):
    """Load only the remaining windows, preserving their saved order."""
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Keep worker initialization separate from the window-shuffling RNG.
        "generator": torch.Generator().manual_seed(args.seed),
    }

    if args.workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )

    return DataLoader(Subset(dataset, indices), **options)


def process_batch(model, optimizer, batch, gb_mask, training):
    """Perform one training update or accumulate equally weighted GB errors."""
    inputs, targets = batch
    inputs = inputs.to("cuda", dtype=torch.float32, non_blocking=True)
    targets = targets.to("cuda", dtype=torch.float32, non_blocking=True)

    # Preserve target validity before replacing missing values.
    target_mask = torch.isfinite(targets)
    inputs = torch.where(torch.isfinite(inputs), inputs, MISSING_VALUE)
    targets = torch.where(target_mask, targets, MISSING_VALUE)

    mask = target_mask if training else target_mask & gb_mask

    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad(set_to_none=True)

        # Residual forecasts remain linear and unclipped.
        predictions = model(inputs)
        errors = torch.where(mask, predictions - targets, 0.0).square()

        if training:
            error_sum = errors.sum()
            pixel_count = mask.sum()
            loss = error_sum / pixel_count

            if not torch.isfinite(loss).item():
                raise FloatingPointError("Training loss became non-finite.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
                error_if_nonfinite=True,
            )
            optimizer.step()

            return error_sum.detach().item(), pixel_count.item()

        # Average over valid GB pixels separately for each sample/channel/lead.
        pixel_counts = mask.sum(dim=(-2, -1))
        if (pixel_counts == 0).any().item():
            raise ValueError("A validation field has no valid GB pixels.")

        field_mse = errors.sum(dim=(-2, -1)) / pixel_counts
        if not torch.isfinite(field_mse).all().item():
            raise FloatingPointError("Validation MSE became non-finite.")

        # These totals give every sample/channel/lead equal weight.
        return field_mse.sum().item(), field_mse.numel()


def write_history(history, path):
    """Write completed epochs without duplicating rows after resumption."""
    temporary_path = path.with_suffix(".tmp")
    columns = [
        "epoch", "train_mse", "validation_gb_mse", "optimizer_updates",
    ]

    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    temporary_path.replace(path)


def main():
    """Train or resume the selected model within this job's session budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-minutes", type=float, default=55)
    parser.add_argument("--checkpoint-minutes", type=float, default=5)
    args = parser.parse_args()

    # Slurm's advance warning requests the same orderly stop as the timer.
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside an Apocrita GPU allocation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    # Match the established float32 training protocol.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.data = args.data.expanduser().resolve()

    # Obtain the core Cloudcasting datasets. Custom loaders below handle resume.
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
    datasets["val"].t0_times = datasets["val"].t0_times.sort_values()

    model = build_model(args.model).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_CONFIG)
    gb_mask = torch.from_numpy(np.isfinite(CUTOUT_MASK)).to("cuda")

    # Record experiment settings separately from adjustable session settings.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "model_name": args.model,
        "model_kwargs": MODEL_CONFIGS[args.model],
        "architecture": str(model),
        "parameters": sum(p.numel() for p in model.parameters()),
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
        "preprocessing": "Native grid; existing scaling; no resampling",
        "missing_pixel_marker": MISSING_VALUE,
        "training_metric": "Full-image valid-pixel MSE",
        "validation_metric": (
            "Spatial MSE over valid GB pixels per sample/channel/lead; "
            "then equal weighting across samples, channels and leads"
        ),
        "optimizer": OPTIMIZER_CONFIG,
        "gradient_clip_norm": GRADIENT_CLIP,
        "scheduler": None,
        "early_stopping": False,
        "precision": "float32",
        "deterministic_algorithms": True,
        "tf32": False,
        "output_head_initialization": "Zero weights and bias",
        "prediction_clipping": False,
        "teacher_forcing": False,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        # Saved indices must still refer to exactly the same forecast windows.
        "forecast_origin_hashes": {
            name: hashlib.sha256(
                dataset.t0_times.asi8.tobytes()
            ).hexdigest()
            for name, dataset in datasets.items()
        },
    }

    # Each model and seed has a stable, separate directory across submissions.
    run_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path(__file__).resolve().parent
        / "runs"
        / f"full_{args.model}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    shuffle_generator = torch.Generator().manual_seed(args.seed)

    # Position counts completed samples within the current phase's window order.
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
        # Our own checkpoint includes Python/NumPy RNG states as well as weights.
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
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    # The checkpoint is authoritative if a previous job stopped during CSV output.
    write_history(progress["history"], metrics_path)
    last_saved = time.monotonic()

    def stopping():
        """Return whether the timer or Slurm has requested an orderly stop."""
        return (
            STOP_REQUESTED
            or time.monotonic() - SESSION_START >= args.session_minutes * 60
        )

    def save_last():
        """Save the complete state needed to resume training or validation."""
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
        f"Model: {args.model} | GPU: {torch.cuda.get_device_name(0)}"
        f" | seed: {args.seed}",
        flush=True,
    )
    print(
        f"Train: {len(datasets['train'])} | "
        f"validation: {len(datasets['val'])} sequences",
        flush=True,
    )
    print(f"Parameters: {config['parameters']:,}", flush=True)
    print(
        f"Starting at epoch {progress['epoch']}, {progress['phase']}, "
        f"{progress['position']} sequences completed",
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
                # Shuffle complete windows once per epoch and save that order.
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

            # Advance only after a whole batch has completed successfully.
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
                    progress[f"{phase}_error"]
                    / progress[f"{phase}_count"]
                )
                print(
                    f"  {phase}: {progress['position']:,}/{len(order):,}"
                    f" sequences | MSE {running_mse:.6f}",
                    flush=True,
                )

            if time.monotonic() - last_saved >= args.checkpoint_minutes * 60:
                save_last()

            if stopping():
                break

        # A partial phase retains its position and accumulated metric totals.
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

            # Strict improvement selects the checkpoint; ties keep earlier epochs.
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

            # Start the next epoch with a new shuffle and empty metric totals.
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

        # Save at every training/validation boundary, including completed epochs.
        save_last()
        write_history(progress["history"], metrics_path)

    # Save the current position before ending this session.
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
            f"{progress['phase']}, "
            f"{progress['position']} sequences completed.",
            flush=True,
        )
        print("The next submission with the same settings resumes here.", flush=True)

    print(f"Resume checkpoint: {last_path}", flush=True)
    print(f"Completed-epoch metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()