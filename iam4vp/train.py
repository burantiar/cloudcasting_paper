"""Train IAM4VP in resumable, approximately 55-minute GPU sessions."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

# Include imports and data preparation in the session budget.
SESSION_START = time.monotonic()
STOP_REQUESTED = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from cloudcasting.constants import CUTOUT_MASK
from cloudcasting.dataset import SatelliteDataModule
from model import IAM4VPForecaster, MODEL_CONFIG, SOURCE_REPOSITORY


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
    """Request checkpointing and exit after the current complete batch."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Replace a checkpoint only after its new version is fully written."""
    torch.cuda.synchronize()
    temporary_path = path.with_suffix(".tmp")
    torch.save(contents, temporary_path)
    temporary_path.replace(path)


def capture_rng(shuffle_generator):
    """Capture random states for continuation."""
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
    """Load remaining windows in their saved order."""
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Worker initialization must not change the training shuffle.
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )

    return DataLoader(Subset(dataset, indices), **options)


def process_batch(model, optimizer, batch, gb_mask, training):
    """Process twelve leads, with one optimizer update for a training batch."""
    inputs, targets = batch
    inputs = inputs.to("cuda", dtype=torch.float32, non_blocking=True)
    targets = targets.to("cuda", dtype=torch.float32, non_blocking=True)

    # Preserve validity before replacing missing pixels with the agreed marker.
    target_mask = torch.isfinite(targets)
    inputs = torch.where(torch.isfinite(inputs), inputs, MISSING_VALUE)
    targets = torch.where(target_mask, targets, MISSING_VALUE)
    mask = target_mask if training else target_mask & gb_mask

    # Counts have shape (batch, channel, lead).
    field_counts = mask.sum(dim=(-2, -1))

    if training:
        # Every lead shares the denominator for the entire forecast batch.
        # Summed gradients therefore represent full-image masked MSE.
        pixel_count = field_counts.sum()
        count = pixel_count.item()
        if count == 0:
            raise ValueError("Training batch has no valid target pixels.")
    else:
        if (field_counts == 0).any().item():
            raise ValueError("Validation contains an empty GB field.")
        count = field_counts.numel()

    total_error = torch.zeros((), device="cuda")
    previous_predictions = []

    with torch.set_grad_enabled(training):
        if training:
            optimizer.zero_grad(set_to_none=True)

        for lead in range(model.forecast_frames):
            prediction = model.forward_step(inputs, previous_predictions)
            errors = torch.where(
                mask[:, :, lead],
                prediction - targets[:, :, lead],
                0.0,
            ).square()

            if training:
                error_sum = errors.sum()

                # Free this lead's graph while accumulating parameter gradients.
                (error_sum / pixel_count).backward()
                total_error += error_sum.detach()
            else:
                # Spatial mean first, then equal sample/channel/lead weighting.
                field_mse = (
                    errors.sum(dim=(-2, -1)) / field_counts[:, :, lead]
                )
                total_error += field_mse.sum()

            # Forecast feedback is detached; no future observations are used.
            previous_predictions.append(prediction.detach())

    error_value = total_error.item()
    if not np.isfinite(error_value):
        raise FloatingPointError("A non-finite training or validation loss occurred.")

    if training:
        # Clip and update once, after accumulating all twelve lead gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
            error_if_nonfinite=True,
        )
        optimizer.step()

    return error_value, count


def write_history(history, path):
    """Write completed-epoch metrics without duplicate rows after resuming."""
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
    """Initialize or resume IAM4VP and run one time-limited session."""
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

    # Register handlers only in the main process, not in DataLoader workers.
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside an Apocrita GPU allocation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    # Match the other full-resolution experiments.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.data = args.data.expanduser().resolve()

    # Keep Cloudcasting's scaling and continuous-window selection.
    # The test split is defined for the protocol but never loaded or evaluated.
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

    model = IAM4VPForecaster(**MODEL_CONFIG).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_CONFIG)
    gb_mask = torch.from_numpy(np.isfinite(CUTOUT_MASK)).to("cuda")

    # Record scientific settings separately from adjustable session settings.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "model_name": "IAM4VP standalone compact adaptation",
        "model_class": "IAM4VPForecaster",
        "model_kwargs": MODEL_CONFIG,
        "source_repository": SOURCE_REPOSITORY,
        "model_source_sha256": hashlib.sha256(
            Path(__file__).with_name("model.py").read_bytes()
        ).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "learned_mask_shape": list(model.mask_token.shape),
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
        "initialization": "Same as notebook 06_iam4vp_exploratory",
        "prediction_clipping": False,
        "teacher_forcing": False,
        "backpropagate_through_rollout": False,
        "backward_strategy": "Per lead, using the full-batch valid-pixel denominator",
        "optimizer_updates_per_batch": 1,
        "feedback": "Detached forecasts, re-encoded by the prediction encoder",
        "time_embedding": "100 times the zero-based forecast lead index",
        "learned_prior_auxiliary_finetuning": False,
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

    # Position counts completed samples, never merely prefetched samples.
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
        restore_rng(checkpoint["rng"], shuffle_generator)
        del checkpoint

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_history(progress["history"], metrics_path)
    last_saved = time.monotonic()

    def stopping():
        """Stop between complete batches when requested or time is exhausted."""
        elapsed = time.monotonic() - SESSION_START
        return STOP_REQUESTED or elapsed >= 60 * args.session_minutes

    def save_last():
        """Save everything needed to resume the current training phase."""
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
        f"GPU: {torch.cuda.get_device_name(0)} | seed: {args.seed} | "
        f"Parameters: {config['parameters']:,}",
        flush=True,
    )
    print(
        f"Train: {len(datasets['train']):,} | "
        f"Validation: {len(datasets['val']):,} sequences",
        flush=True,
    )
    print(
        f"Starting at epoch {progress['epoch']}, "
        f"phase {progress['phase']}, sample position {progress['position']}.",
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
                # Shuffle whole temporal windows, preserving frame order within.
                progress["order"] = list(
                    RandomSampler(dataset, generator=shuffle_generator)
                )
            order = progress["order"]
        else:
            order = list(range(len(dataset)))

        remaining = order[progress["position"]:]
        loader = make_loader(dataset, remaining, args) if remaining else ()
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

            # Advance only after all leads and the optimizer update finish.
            previous_position = progress["position"]
            progress[f"{phase}_error"] += error_sum
            progress[f"{phase}_count"] += count
            progress["position"] += batch[0].shape[0]
            if training:
                progress["updates"] += 1

            interval = 250 * args.batch_size
            if (
                progress["position"] // interval > previous_position // interval
                or progress["position"] == len(order)
            ):
                mean_error = (
                    progress[f"{phase}_error"] / progress[f"{phase}_count"]
                )
                print(
                    f"  {phase}: {progress['position']:,}/{len(order):,} "
                    f"sequences | MSE {mean_error:.6f}",
                    flush=True,
                )

            if time.monotonic() - last_saved >= 60 * args.checkpoint_minutes:
                save_last()

            if stopping():
                break

        # Preserve the current phase and its accumulated metrics if unfinished.
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

            # Select the best model only after evaluating the full val split.
            improved = val_mse < progress["best_mse"]
            if improved:
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
                f"Epoch {epoch} complete | train MSE {train_mse:.6f} | "
                f"validation GB MSE {val_mse:.6f}"
                + (" | saved best checkpoint" if improved else ""),
                flush=True,
            )

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

        # Save at phase boundaries as well as periodically within phases.
        save_last()
        write_history(progress["history"], metrics_path)

    save_last()
    write_history(progress["history"], metrics_path)

    if progress["epoch"] > args.epochs:
        print(
            f"\nTraining complete. Best epoch: {progress['best_epoch']} | "
            f"validation GB MSE: {progress['best_mse']:.6f}",
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
            "The next submission using this output directory resumes here.",
            flush=True,
        )

    print(f"Resume checkpoint: {last_path}", flush=True)
    print(f"Completed-epoch metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()