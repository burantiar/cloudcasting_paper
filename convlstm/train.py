"""Train the native-resolution ConvLSTM using Cloudcasting's data loaders."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from cloudcasting.constants import CUTOUT_MASK
from cloudcasting.dataset import SatelliteDataModule

from model import ConvLSTMForecaster


def seed_everything(seed):
    """Seed the random generators and select deterministic float32 operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_loaders(config, train_generator):
    """Build complete native-resolution training and validation sequences."""
    workers = config["num_workers"]
    data = SatelliteDataModule(
        zarr_path=config["data_path"],
        variables=config["channels"],
        history_mins=config["history_mins"],
        forecast_mins=config["forecast_mins"],
        sample_freq_mins=config["sample_freq_mins"],
        train_period=config["train_period"],
        val_period=config["val_period"],
        test_period=config["test_period"],
        batch_size=config["batch_size"],
        num_workers=workers,
        prefetch_factor=1 if workers else None,
        persistent_workers=workers > 0,
        pin_memory=True,
        nan_to_num=False,  # Capture validity before replacing missing pixels.
    )
    train, val = data.train_dataloader(), data.val_dataloader()
    for loader in (train, val):
        loader.dataset.t0_times = loader.dataset.t0_times.sort_values()
        size = tuple(loader.dataset.ds.sizes[d] for d in ("y_geostationary", "x_geostationary"))
        if size != CUTOUT_MASK.shape:
            raise ValueError(f"Expected native {CUTOUT_MASK.shape} images; received {size}.")
        loader.generator = torch.Generator().manual_seed(config["seed"])
        if workers:
            loader.multiprocessing_context = "spawn"
    # Separate sample-order RNG from worker RNG so epoch-boundary resume is stable.
    train.sampler.generator = train_generator
    return train, val


def prepare_batch(X, target, device):
    """Move a batch to the GPU, retain validity masks, and fill missing pixels."""
    X = X.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    if torch.isinf(X).any() or torch.isinf(target).any():
        raise ValueError("The input or target contains an infinite value.")
    valid = torch.isfinite(target)
    X = torch.where(torch.isfinite(X), X, -1.0)
    target = torch.where(valid, target, -1.0)
    return X, target, valid


def squared_errors(prediction, target, mask):
    """Return squared errors with zero contribution from masked-out targets."""
    # Mask before subtraction: ignored NaNs must not enter loss or gradients.
    prediction = torch.where(mask, prediction, 0.0)
    target = torch.where(mask, target, 0.0)
    return (prediction - target).square()


def run_epoch(model, loader, device, region, clip_norm, optimizer=None):
    """Train on full images or evaluate equally weighted GB validation MSE."""
    training = optimizer is not None
    model.train(training)
    total = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.int64, device=device)
    samples = 0
    val_total = None

    with torch.set_grad_enabled(training):
        for X, target in loader:
            X, target, valid = prepare_batch(X, target, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(X)
            if not torch.isfinite(prediction).all():
                raise ValueError("The model produced non-finite predictions.")

            if training:
                n_valid = valid.sum()
                if n_valid == 0:
                    raise ValueError("A training batch has no valid target pixels.")
                sse = squared_errors(prediction, target, valid).sum()
                loss = sse / n_valid
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm,
                    error_if_nonfinite=True, foreach=False,
                )
                optimizer.step()
                total += sse.detach().double()
                count += n_valid
            else:
                # Spatial MSE per sample/channel/lead, then equal sample weighting.
                mask = valid & region
                n_valid = mask.sum(dim=(-2, -1))
                if (n_valid == 0).any():
                    raise ValueError("A validation sample/channel/lead has no valid GB pixels.")
                scores = squared_errors(prediction, target, mask).sum(dim=(-2, -1)) / n_valid
                batch_total = scores.double().sum(dim=0)
                val_total = batch_total if val_total is None else val_total + batch_total
                samples += X.shape[0]

    if training:
        optimizer.zero_grad(set_to_none=True)
        return (total / count).item()
    return (val_total / samples).mean().item()


def save_checkpoint(path, state):
    """Replace a checkpoint only after its temporary file is fully written."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_metrics(path, history):
    """Write one CSV row per completed epoch from the saved training history."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main():
    """Read the run settings, initialize or resume training, and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Downloaded native-resolution Zarr store")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="New run directory; defaults to runs/<timestamp>")
    parser.add_argument("--resume", type=Path, help="This script's last.pt; restart at the next epoch")
    args = parser.parse_args()

    # Resume our own checkpoints, including optimizer and random-generator states.
    saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = saved["config"] if saved else json.loads(args.config.read_text())
    config["data_path"] = str(args.data.expanduser().resolve())
    if not saved:
        config["seed"] = args.seed
    seed_everything(config["seed"])
    device = torch.device("cuda")

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.resume.resolve().parent if saved else (
        args.output or Path("runs") / f"convlstm_seed{config['seed']}_{tag}"
    )
    output.mkdir(parents=True, exist_ok=bool(saved))
    train_generator = torch.Generator().manual_seed(config["seed"])
    train_loader, val_loader = make_loaders(config, train_generator)
    model = ConvLSTMForecaster(
        input_channels=len(config["channels"]),
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        history_steps=config["history_mins"] // config["sample_freq_mins"] + 1,
        forecast_steps=config["forecast_mins"] // config["sample_freq_mins"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
    region = torch.as_tensor(np.isfinite(CUTOUT_MASK), device=device)[None, None, None]

    start_epoch, best_mse, history = 1, float("inf"), []
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        train_generator.set_state(saved["train_generator"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state(saved["rng"]["cuda"])
        start_epoch, best_mse = saved["epoch"] + 1, saved["best_mse"]
        history = saved["history"]
        write_metrics(output / "metrics.csv", history)

    config["protocol"] = {
        "spatial_size": list(CUTOUT_MASK.shape), "spatial_resizing": False,
        "missing_value": -1.0, "precision": "float32", "tf32": False,
        "training_loss": "Full-image masked MSE; valid-pixel-weighted epoch mean",
        "validation_metric": "Native GB MSE: equal samples, channels and forecast leads",
        "checkpoint_selection": "Lowest validation MSE; ties keep earlier epoch",
        "scheduler": None, "early_stopping": False, "test_evaluation": False,
        "training_sequences": len(train_loader.dataset),
        "validation_sequences": len(val_loader.dataset),
    }
    config["runtime"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "packages": {name: version(name) for name in ("cloudcasting", "numpy", "xarray", "zarr")},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Run: {output}\nGPU: {config['runtime']['gpu']} | seed: {config['seed']}", flush=True)
    print(f"Train: {len(train_loader.dataset)} | validation: {len(val_loader.dataset)} sequences", flush=True)

    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch}/{config['epochs']} starting", flush=True)
        train_mse = run_epoch(model, train_loader, device, region, config["gradient_clip_norm"], optimizer)
        val_mse = run_epoch(model, val_loader, device, region, config["gradient_clip_norm"])
        improved = val_mse < best_mse
        best_mse = min(best_mse, val_mse)
        history.append({
            "epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
            "seconds": time.perf_counter() - started,
        })

        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_mse": best_mse, "history": history, "config": config,
            "train_generator": train_generator.get_state(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
        }
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        write_metrics(output / "metrics.csv", history)
        print(f"Epoch {epoch}: train MSE={train_mse:.6f}, GB val MSE={val_mse:.6f}, "
              f"{history[-1]['seconds'] / 60:.1f} min" + (" [best]" if improved else ""), flush=True)


if __name__ == "__main__":
    main()
