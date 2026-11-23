"""Train the STLDM adaptation in resumable 55-minute GPU sessions.

Joint loss:
    reconstruction MSE + forecast MSE + latent noise MSE + 0.0001 * KL.

Validation:
    Generated-image GB MSE with equal sample/channel/lead weighting.

All components are trained jointly from scratch.
Resume automatically from last.pt in the selected output directory.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

# Include library imports and dataset preparation in the session budget.
SESSION_START = time.monotonic()
STOP_REQUESTED = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from cloudcasting.constants import CUTOUT_MASK
from cloudcasting.dataset import SatelliteDataModule
from model import MODEL_CONFIG, STLDMForecaster


HERE = Path(__file__).resolve().parent

CHANNELS = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
    "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
]

SPATIAL_SIZE = (372, 614)
TRAIN_PERIOD = ["2020-06-01 00:00", "2020-07-31 23:45"]
VAL_PERIOD = ["2020-08-01 00:00", "2020-08-15 23:45"]
TEST_PERIOD = ["2020-08-16 00:00", "2020-08-31 23:45"]

MISSING_VALUE = -1.0
GRADIENT_CLIP = 1.0

OPTIMIZER_CONFIG = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "foreach": False,
}


def request_stop(signum, frame):
    """Request an orderly exit after the current complete batch."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Replace a checkpoint only after its new file is fully written."""
    torch.cuda.synchronize()
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def write_history(history, path):
    """Write completed-epoch metrics without duplicating resumed epochs."""
    columns = [
        "epoch",
        "train_joint_loss",
        *[f"train_{name}" for name in MODEL_CONFIG["loss_weights"]],
        "validation_gb_mse",
        "optimizer_updates",
    ]
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

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
    """Restore randomness after constructing the model and optimizer."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])
    shuffle_generator.set_state(states["shuffle"])
    validation_generator.set_state(states["validation"])


def make_loader(dataset, indices, args):
    """Load remaining complete windows in their saved order."""
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Worker initialization must not consume training or sampling RNG.
        "generator": torch.Generator().manual_seed(args.seed),
    }

    if args.workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )

    return DataLoader(Subset(dataset, indices), **options)


def prepare_batch(batch):
    """Preserve validity masks before replacing missing pixels with -1."""
    inputs, targets = [
        tensor.to("cuda", dtype=torch.float32, non_blocking=True)
        for tensor in batch
    ]
    input_mask = torch.isfinite(inputs)
    target_mask = torch.isfinite(targets)

    inputs = torch.where(input_mask, inputs, MISSING_VALUE)
    targets = torch.where(target_mask, targets, MISSING_VALUE)

    return inputs, targets, input_mask, target_mask


def train_batch(model, optimizer, batch):
    """Make one joint optimizer update and return component totals."""
    inputs, targets, input_mask, target_mask = prepare_batch(batch)

    input_pixels = input_mask.sum().item()
    target_pixels = target_mask.sum().item()
    if target_pixels == 0:
        raise ValueError("Training batch has no valid target pixels.")

    optimizer.zero_grad(set_to_none=True)

    # The model computes all four losses, including target latent encoding.
    # No components are frozen and no latent tensors are detached here.
    loss, components = model.training_loss(
        inputs, targets, input_mask, target_mask
    )

    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite STLDM joint training loss.")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRADIENT_CLIP,
        error_if_nonfinite=True,
        foreach=False,
    )
    optimizer.step()

    # Match notebook 16's aggregation using each component's denominator.
    # Latent coverage averages channel validity within 4x4 spatial cells.
    # Padded pixels contribute zero coverage.
    all_pixels = input_pixels + target_pixels
    latent_factor = model.latent_channels / (model.channels * 4**2)

    counts = {
        "reconstruction": all_pixels,
        "forecast": target_pixels,
        "denoising": target_pixels * latent_factor,
        "kl": all_pixels * latent_factor,
    }
    sums = {
        name: value.item() * counts[name]
        for name, value in components.items()
    }
    return sums, counts


@torch.inference_mode()
def validate_batch(model, batch, gb_mask, generator):
    """Return summed GB field MSE and the number of evaluated fields."""
    inputs, targets, _, target_mask = prepare_batch(batch)

    # Only the five observations condition the generated forecasts.
    # Outputs are already in the original data scale; do not clip them.
    predictions = model.sample(inputs, generator=generator)
    mask = target_mask & gb_mask

    squared_errors = torch.where(
        mask, predictions - targets, 0.0
    ).square()

    # Spatial averaging first, separately for every sample/channel/lead.
    pixel_counts = mask.sum(dim=(-2, -1))
    field_mse = squared_errors.sum(dim=(-2, -1)) / pixel_counts

    if not torch.isfinite(field_mse).all().item():
        raise FloatingPointError(
            "Validation contains an empty or non-finite GB field."
        )

    # Pool fields equally, including across differently sized final batches.
    return field_mse.sum().item(), field_mse.numel()


def training_means(progress, weights):
    """Return pooled component means and their weighted joint loss."""
    means = {
        name: progress["train_sums"][name]
        / max(progress["train_counts"][name], 1e-8)
        for name in weights
    }
    joint = sum(weights[name] * means[name] for name in weights)
    return means, joint


def run_session(args, run_dir):
    """Initialize or resume STLDM and execute one GPU training session."""
    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside a GPU allocation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)

    # Keep the numerical settings aligned with the other cluster models.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # The repository selects complete, contiguous windows within each split.
    # Preserve the downloaded values and missing pixels; no loader resizing.
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

    # Undo the repository's validation preshuffle.
    datasets["val"].t0_times = datasets["val"].t0_times.sort_values()

    for dataset in datasets.values():
        size = (
            dataset.ds.sizes["y_geostationary"],
            dataset.ds.sizes["x_geostationary"],
        )
        if size != SPATIAL_SIZE:
            raise ValueError("Use the native 372 x 614 dataset.")

    model = STLDMForecaster(**MODEL_CONFIG).to(
        device="cuda", dtype=torch.float32
    )
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_CONFIG)

    # Native GB mask: no interpolation or spatial resampling.
    gb_mask = torch.from_numpy(np.isfinite(CUTOUT_MASK)).to("cuda")
    weights = dict(model.loss_weights)

    # Save the scientific settings and require them to match on resume.
    # Session duration and worker count may change between submissions.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "model_name": "STLDM-inspired compact latent diffusion",
        "model_class": "STLDMForecaster",
        "source_is_exact_reproduction": False,
        "model_kwargs": MODEL_CONFIG,
        "model_source_sha256": hashlib.sha256(
            (HERE / "model.py").read_bytes()
        ).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
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
        "model_value_transform": "Encoder: 2*x - 1; decoder: (x + 1)/2",
        "spatial_padding": (
            "Model pads transformed images to multiples of four and "
            "crops decoded outputs back to the original extent"
        ),
        "latent_spatial_size": [93, 154],
        "training_strategy": "Joint training of all components from scratch",
        "joint_loss_weights": weights,
        "loss_components": {
            "reconstruction": "Masked image MSE over history and future",
            "forecast": "Masked deterministic forecast MSE over future",
            "denoising": "Coverage-weighted latent epsilon MSE over future",
            "kl": "Coverage-weighted posterior KL over history and future",
        },
        "training_aggregation": (
            "Pool each component using its own pixel or coverage count; "
            "report the weighted sum of these component means"
        ),
        "latent_masking": (
            "Average validity across channels and 4x4 spatial cells; "
            "broadcast across latent channels. Coverage approximation."
        ),
        "gradient_flow": "Joint, including target latent encoding",
        "validation_metric": (
            "Spatial MSE over finite GB target pixels per sample/channel/lead; "
            "then equal weighting across all fields"
        ),
        "validation_samples_per_origin": 1,
        "validation_sampling_seed": args.seed,
        "validation_noise_policy": (
            "Reset CPU generator for each new chronological validation pass; "
            "restore its state when resuming a partial pass"
        ),
        "diffusion": {
            "training_noise_levels": model.train_steps,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "sampler": "DDIM",
            "sampling_steps": model.sample_steps,
            "timestep_spacing": "leading",
            "eta": 0.0,
            "classifier_free_guidance": False,
            "ema": False,
        },
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
        "numpy_version": np.__version__,
        "cuda_build": torch.version.cuda,
        # Saved indices must identify the same forecast windows on resume.
        "forecast_origin_hashes": {
            name: hashlib.sha256(
                dataset.t0_times.asi8.tobytes()
            ).hexdigest()
            for name, dataset in datasets.items()
        },
    }

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    validation_generator = torch.Generator(
        device="cpu"
    ).manual_seed(args.seed)

    # Position counts completed samples, never prefetched samples.
    progress = {
        "epoch": 1,
        "phase": "train",
        "position": 0,
        "order": None,
        "train_sums": dict.fromkeys(weights, 0.0),
        "train_counts": dict.fromkeys(weights, 0.0),
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
            checkpoint["rng"], shuffle_generator, validation_generator
        )
        del checkpoint

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_history(progress["history"], metrics_path)
    last_saved = time.monotonic()

    def stopping():
        """Check the session timer and Slurm shutdown request."""
        return (
            STOP_REQUESTED
            or time.monotonic() - SESSION_START >= args.session_minutes * 60
        )

    def save_last():
        """Save the complete state needed to continue this experiment."""
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
    print(f"Joint loss weights: {weights}", flush=True)
    print(
        f"Starting at epoch {progress['epoch']}, phase {progress['phase']}, "
        f"sample position {progress['position']}.",
        flush=True,
    )

    # Give a new run an initial checkpoint before its first batch.
    if not last_path.exists():
        save_last()

    while progress["epoch"] <= args.epochs:
        if stopping():
            break

        phase = progress["phase"]
        training = phase == "train"
        dataset = datasets[phase]

        if training:
            if progress["order"] is None:
                # Shuffle complete sequences once per epoch.
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

            if training:
                sums, counts = train_batch(model, optimizer, batch)
                for name in weights:
                    progress["train_sums"][name] += sums[name]
                    progress["train_counts"][name] += counts[name]
                progress["updates"] += 1
            else:
                error, count = validate_batch(
                    model, batch, gb_mask, validation_generator
                )
                progress["val_error"] += error
                progress["val_count"] += count

            # Advance only after a complete update or generated forecast batch.
            progress["position"] += batch[0].shape[0]

            report_every = (250 if training else 100) * args.batch_size
            if (
                progress["position"] % report_every == 0
                or progress["position"] == len(order)
            ):
                if training:
                    _, metric = training_means(progress, weights)
                    label = "joint loss"
                else:
                    metric = progress["val_error"] / progress["val_count"]
                    label = "GB MSE"

                print(
                    f"  {phase}: {progress['position']:,}/{len(order):,}"
                    f" sequences | {label} {metric:.6f}",
                    flush=True,
                )

            # Save periodically, as well as at orderly session shutdown.
            if time.monotonic() - last_saved >= args.checkpoint_minutes * 60:
                save_last()

            if stopping():
                break

        # Preserve partial totals and the exact remaining sample order.
        if progress["position"] < len(order):
            break

        if training:
            progress["phase"] = "val"
            progress["position"] = 0

            # Reset only for a NEW validation pass, never a resumed partial pass.
            validation_generator.manual_seed(args.seed)
        else:
            epoch = progress["epoch"]
            components, joint_loss = training_means(progress, weights)
            val_mse = progress["val_error"] / progress["val_count"]

            progress["history"].append({
                "epoch": epoch,
                "train_joint_loss": joint_loss,
                **{
                    f"train_{name}": value
                    for name, value in components.items()
                },
                "validation_gb_mse": val_mse,
                "optimizer_updates": progress["updates"],
            })

            # Select best weights only after evaluating the full validation split.
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
                f"Epoch {epoch} complete | train joint loss {joint_loss:.6f}"
                f" | validation GB MSE {val_mse:.6f}"
                f" | best epoch {progress['best_epoch']}"
                + (" [best]" if improved else ""),
                flush=True,
            )

            # Clear epoch totals only after validation has finished.
            progress.update({
                "epoch": epoch + 1,
                "phase": "train",
                "position": 0,
                "order": None,
                "train_sums": dict.fromkeys(weights, 0.0),
                "train_counts": dict.fromkeys(weights, 0.0),
                "val_error": 0.0,
                "val_count": 0,
            })

        # Also checkpoint at each training/validation boundary.
        save_last()
        write_history(progress["history"], metrics_path)

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


def main():
    """Parse session settings and prevent simultaneous checkpoint writers."""
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

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args.data = args.data.expanduser().resolve()
    run_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else HERE / "runs" / f"full_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Release automatically on exit; a leftover lock file is harmless.
    with (run_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Another process is already training in {run_dir}."
            ) from None

        run_session(args, run_dir)


if __name__ == "__main__":
    main()