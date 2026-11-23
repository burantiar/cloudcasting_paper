"""Train residual diffusion in resumable 55-minute GPU sessions.

IAM4VP remains frozen.
Training: full-image masked L1 noise-prediction loss.
Validation: GB MSE with equal sample/channel/lead weighting.
"""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
from importlib.metadata import version
import io
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
from model import MODEL_CONFIG, ResidualDiffusionForecaster


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
    """Request a checkpointed exit after the current complete batch."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_save(contents, path):
    """Replace a checkpoint only after its new file is fully written."""
    torch.cuda.synchronize()
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def capture_rng(shuffle_generator, validation_generator):
    """Capture training randomness, sample shuffling and validation noise."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "shuffle": shuffle_generator.get_state(),
        "validation": validation_generator.get_state(),
    }


def restore_rng(states, shuffle_generator, validation_generator):
    """Restore random states after model and optimizer construction."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])
    shuffle_generator.set_state(states["shuffle"])
    validation_generator.set_state(states["validation"])


def load_predictor(args, run_dir, resume):
    """Construct IAM4VP using a fixed checkpoint and a saved source definition."""
    source_copy = run_dir / "iam4vp_model.py"

    if resume is None:
        # Read the checkpoint once so its hash identifies exactly these weights.
        checkpoint_path = args.predictor.expanduser().resolve()
        checkpoint_bytes = checkpoint_path.read_bytes()
        pretrained = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
        predictor_config = pretrained["config"]

        # Require a predictor trained with the same data and forecasting task.
        expected = {
            "model_class": "IAM4VPForecaster",
            "data": str(args.data),
            "channels": CHANNELS,
            "spatial_size": list(SPATIAL_SIZE),
            "input_frames": 5,
            "forecast_frames": 12,
            "interval_minutes": 15,
            "train_period": TRAIN_PERIOD,
            "validation_period": VAL_PERIOD,
            "missing_pixel_marker": MISSING_VALUE,
        }
        differences = [
            key for key, value in expected.items()
            if predictor_config.get(key) != value
        ]
        if differences:
            raise ValueError(
                "IAM4VP checkpoint differs from this experiment: "
                + ", ".join(differences)
            )

        # Keep its architecture definition fixed even if the sibling file changes.
        source_bytes = args.predictor_source.expanduser().read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if source_hash != predictor_config["model_source_sha256"]:
            raise ValueError(
                "The IAM4VP source does not match the selected checkpoint."
            )
        source_copy.write_bytes(source_bytes)

        metadata = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_epoch": pretrained["epoch"],
            "validation_gb_mse": pretrained["validation_gb_mse"],
            "source_sha256": source_hash,
            "config": predictor_config,
        }
    else:
        # Resume uses its own predictor provenance, not a newer external best.pt.
        metadata = resume["config"]["predictor"]

    if hashlib.sha256(source_copy.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("The saved IAM4VP definition has changed.")

    spec = importlib.util.spec_from_file_location("frozen_iam4vp", source_copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    predictor = module.IAM4VPForecaster(
        **metadata["config"]["model_kwargs"]
    )

    if resume is None:
        predictor.load_state_dict(pretrained["model"])
    # On resume, the composite model's state restores these weights below.
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor, metadata


def make_loader(dataset, indices, args):
    """Load the remaining complete windows in their saved order."""
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": False,
        # Loader initialization must not consume the model's random state.
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers > 0:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )
    return DataLoader(Subset(dataset, indices), **options)


def process_batch(
    model, optimizer, batch, gb_mask, training, validation_generator
):
    """Perform one denoiser update or evaluate one generated forecast batch."""
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

            # IAM4VP runs without gradients inside denoising_loss().
            # One update covers the complete twelve-frame residual sequence.
            loss = model.denoising_loss(inputs, targets, target_mask)
            loss_value = loss.detach().item()
            if not np.isfinite(loss_value):
                raise FloatingPointError("Non-finite training noise loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.unet.parameters(),
                GRADIENT_CLIP,
                error_if_nonfinite=True,
                foreach=False,
            )
            optimizer.step()

            # Training loss is weighted by valid target-pixel count.
            return loss_value * pixel_count, pixel_count

        # Forecast = frozen IAM4VP prediction + sampled diffusion correction.
        predictions = model.sample(inputs, generator=validation_generator)
        mask = target_mask & gb_mask

        squared_errors = torch.where(
            mask, predictions - targets, 0.0
        ).square()

        # Average spatially within each sample/channel/lead, then equally
        # across all fields. Predictions remain in the original data scale.
        pixel_counts = mask.sum(dim=(-2, -1))
        field_mse = squared_errors.sum(dim=(-2, -1)) / pixel_counts

        if not torch.isfinite(field_mse).all().item():
            raise FloatingPointError(
                "Validation contains an empty or non-finite GB field."
            )

        return field_mse.sum().item(), field_mse.numel()


def write_history(history, path):
    """Write completed-epoch metrics without duplicating resumed epochs."""
    temporary = path.with_suffix(".tmp")
    columns = [
        "epoch", "train_noise_l1", "validation_gb_mse", "optimizer_updates",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def main():
    """Initialize or resume one time-limited residual-diffusion session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--predictor",
        type=Path,
        default=HERE.parent / "iam4vp/runs/full_seed0/best.pt",
    )
    parser.add_argument(
        "--predictor-source",
        type=Path,
        default=HERE.parent / "iam4vp/model.py",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session-minutes", type=float, default=55)
    parser.add_argument("--checkpoint-minutes", type=float, default=5)
    args = parser.parse_args()

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside an Apocrita GPU allocation.")

    args.data = args.data.expanduser().resolve()
    run_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else HERE / "runs" / f"full_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prevent two jobs from updating the same run simultaneously.
    run_lock = (run_dir / ".run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    # Our own resume checkpoint includes Python and NumPy random states.
    resume = (
        torch.load(last_path, map_location="cpu", weights_only=False)
        if last_path.exists()
        else None
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)

    # Match the other full-resolution experiments.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Use the repository's contiguous-window selection and existing scaling.
    # Missing pixels remain NaN until process_batch captures target validity.
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

    for dataset in datasets.values():
        size = (
            dataset.ds.sizes["y_geostationary"],
            dataset.ds.sizes["x_geostationary"],
        )
        if size != SPATIAL_SIZE:
            raise ValueError("This experiment requires native 372 x 614 data.")

    predictor, predictor_metadata = load_predictor(args, run_dir, resume)

    # Match notebook 15: seed denoiser initialization after constructing IAM4VP.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = ResidualDiffusionForecaster(
        predictor=predictor, **MODEL_CONFIG
    ).to(device="cuda", dtype=torch.float32)

    # IAM4VP is excluded from both optimizer updates and gradient clipping.
    optimizer = torch.optim.Adam(
        model.unet.parameters(), **OPTIMIZER_CONFIG
    )
    gb_mask = torch.from_numpy(np.isfinite(CUTOUT_MASK)).to("cuda")

    # Scientific settings are fixed across submissions. Session duration and
    # worker count can change without changing this experiment's definition.
    config = {
        "stage": "full_resolution_three_month_pilot",
        "model_class": "ResidualDiffusionForecaster",
        "model_kwargs": MODEL_CONFIG,
        "model_source_sha256": hashlib.sha256(
            HERE.joinpath("model.py").read_bytes()
        ).hexdigest(),
        "reference_notebook": "15_residual_diffusion_exploratory",
        "source_is_exact_reproduction": False,
        "predictor": predictor_metadata,
        "predictor_frozen": True,
        "predictor_pretraining_counted_separately": True,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
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
        "preprocessing": "Native grid; existing scaling; no loader resampling",
        "missing_pixel_marker": MISSING_VALUE,
        "conditioning_transform": "2*x - 1",
        "residual_model_space": "2 * (target - frozen IAM4VP forecast)",
        "forecast_reconstruction": "IAM4VP forecast + sampled residual / 2",
        "missing_target_residual_fill": 0.0,
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
            "Reset CPU generator for each new validation pass; "
            "restore its state during partial-pass resume"
        ),
        "prediction_clipping": False,
        "teacher_forcing": False,
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
        "forecast_origin_hashes": {
            name: hashlib.sha256(dataset.t0_times.asi8.tobytes()).hexdigest()
            for name, dataset in datasets.items()
        },
    }

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(args.seed)

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

    if resume is not None:
        if resume["config"] != config:
            raise ValueError(
                "Saved experiment settings differ. Restore the original "
                "settings or choose a new --output directory."
            )
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        progress = resume["progress"]
        restore_rng(
            resume["rng"], shuffle_generator, validation_generator
        )
    del resume

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_history(progress["history"], metrics_path)
    last_saved = time.monotonic()

    def stopping():
        """Check the session budget and advance-shutdown signals."""
        return (
            STOP_REQUESTED
            or time.monotonic() - SESSION_START >= args.session_minutes * 60
        )

    def save_last():
        """Save both models, optimizer, partial metrics and random states."""
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
    print(f"GPU: {torch.cuda.get_device_name(0)} | seed: {args.seed}", flush=True)
    print(
        f"Frozen IAM4VP checkpoint epoch: "
        f"{predictor_metadata['checkpoint_epoch']} | "
        f"GB MSE: {predictor_metadata['validation_gb_mse']:.6f}",
        flush=True,
    )
    print(
        f"Train: {len(datasets['train']):,} | "
        f"validation: {len(datasets['val']):,} sequences",
        flush=True,
    )
    print(
        f"Starting at epoch {progress['epoch']}, "
        f"phase {progress['phase']}, sample position {progress['position']}.",
        flush=True,
    )

    # Preserve the pinned predictor before performing the first update.
    save_last()

    while progress["epoch"] <= args.epochs:
        if stopping():
            break

        phase = progress["phase"]
        training = phase == "train"
        dataset = datasets[phase]

        if training:
            if progress["order"] is None:
                # Shuffle whole windows once per epoch; never shuffle frames.
                progress["order"] = list(
                    RandomSampler(dataset, generator=shuffle_generator)
                )
            order = progress["order"]
        else:
            order = list(range(len(dataset)))

        remaining = order[progress["position"]:]
        loader = make_loader(dataset, remaining, args) if remaining else []

        # The model's train() override keeps IAM4VP in evaluation mode.
        model.train(training)
        print(
            f"\nEpoch {progress['epoch']}/{args.epochs} — {phase}",
            flush=True,
        )

        for batch in loader:
            if stopping():
                break

            error_sum, count = process_batch(
                model, optimizer, batch, gb_mask,
                training, validation_generator,
            )

            # Advance only after a complete optimizer update or forecast batch.
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
                value = (
                    progress[f"{phase}_error"]
                    / progress[f"{phase}_count"]
                )
                label = "noise L1" if training else "GB MSE"
                print(
                    f"  {phase}: {progress['position']:,}/{len(order):,}"
                    f" sequences | {label} {value:.6f}",
                    flush=True,
                )

            if time.monotonic() - last_saved >= args.checkpoint_minutes * 60:
                save_last()
            if stopping():
                break

        # Preserve partial-phase totals and sample order for the next session.
        if progress["position"] < len(order):
            break

        if training:
            progress["phase"] = "val"
            progress["position"] = 0

            # Reset noise only for a NEW validation pass.
            # A resumed partial pass retains its saved generator state.
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

            # Best-checkpoint selection requires the complete validation split.
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
                f"Epoch {epoch} complete | train noise L1 {train_l1:.6f}"
                f" | validation GB MSE {val_mse:.6f}"
                + (" | saved best checkpoint" if improved else ""),
                flush=True,
            )

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

    # Finish by preserving the last completed batch.
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