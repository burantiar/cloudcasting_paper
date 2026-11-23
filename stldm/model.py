"""STLDM-inspired forecaster matching notebook 16.

Input:  (batch, 11, 5, 372, 614).
Output: (batch, 11, 12, 372, 614).

All components train jointly from scratch.
Preprocessing, device selection and checkpoint management belong in train.py.
Requires PyTorch only.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


SOURCE_REPOSITORY = "https://github.com/sqfoo/stldm_official"

# Preserve the notebook's fixed exploratory loss weights.
LOSS_WEIGHTS = {
    "reconstruction": 1.0,
    "forecast": 1.0,
    "denoising": 1.0,
    "kl": 1e-4,
}

MODEL_CONFIG = {
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "latent_channels": 8,
    "width": 32,
    "translator_width": 64,
    "denoiser_width": 32,
    "train_steps": 1000,
    "sample_steps": 20,
    "loss_weights": LOSS_WEIGHTS.copy(),
}


def conv_block(in_channels, out_channels, stride=1):
    """Create a spatial convolution with per-image normalization."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


def masked_mse(prediction, target, mask):
    """Average squared errors over valid image pixels."""
    errors = torch.where(mask, prediction - target, 0.0).square()
    return errors.sum() / mask.sum().clamp_min(1)


def latent_mean(values, coverage):
    """Average latent losses using coverage shared across latent channels."""
    weights = coverage.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class GatedSpatialBlock(nn.Module):
    """Mix spatial features after history frames have been channel-stacked."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.spatial = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels
        )
        self.gates = nn.Conv2d(channels, 2 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Apply a gated residual transformation."""
        value, gate = self.gates(
            self.spatial(self.norm(x))
        ).chunk(2, dim=1)
        return x + self.output(F.gelu(value) * gate.sigmoid())


class TimeResidualBlock(nn.Module):
    """Spatial residual block conditioned on the diffusion timestep."""

    def __init__(self, in_channels, out_channels, time_dim=128):
        super().__init__()
        self.first = conv_block(in_channels, out_channels)
        self.time = nn.Linear(time_dim, out_channels)
        self.second = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, time_embedding):
        """Add the diffusion-time embedding to spatial features."""
        hidden = self.first(x)
        hidden = hidden + self.time(
            F.silu(time_embedding)
        )[:, :, None, None]
        return self.skip(x) + self.second(F.silu(hidden))


class TemporalAttention(nn.Module):
    """Attend across forecast leads independently at each spatial location."""

    def __init__(self, channels, frames, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.LayerNorm(channels)
        self.position = nn.Parameter(torch.randn(1, frames, channels) * 0.02)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, x):
        """Transform a B,T,C,H,W sequence without mixing separate samples."""
        batch, frames, channels, height, width = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, frames, channels
        )

        hidden = self.norm(tokens) + self.position
        qkv = self.qkv(hidden).reshape(
            -1, frames, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention = (
            query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        ).softmax(dim=-1)
        mixed = (attention @ value).transpose(1, 2).reshape_as(tokens)
        tokens = tokens + self.output(mixed)

        return tokens.reshape(
            batch, height, width, frames, channels
        ).permute(0, 3, 4, 1, 2).contiguous()


class LatentDenoiser(nn.Module):
    """Compact spatial U-Net with temporal attention at its bottleneck."""

    def __init__(self, latent_channels, input_frames, forecast_frames, width=32):
        super().__init__()
        self.latent_channels = latent_channels

        # Each lead receives its noisy latent, predicted latent and all history.
        input_channels = latent_channels * (input_frames + 2)

        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(32, dtype=torch.float32) / 31
        )
        self.register_buffer("frequencies", frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        self.enc1 = TimeResidualBlock(input_channels, width)
        self.down1 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.enc2 = TimeResidualBlock(2 * width, 2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.middle = TimeResidualBlock(4 * width, 4 * width)
        self.temporal = TemporalAttention(4 * width, forecast_frames)

        self.dec2 = TimeResidualBlock(6 * width, 2 * width)
        self.dec1 = TimeResidualBlock(3 * width, width)
        self.output = nn.Conv2d(width, latent_channels, 3, padding=1)

    def forward(self, noisy, history, condition, timesteps):
        """Predict latent noise jointly for all forecast leads."""
        batch, latent_channels, frames, height, width = noisy.shape

        observed = history.flatten(1, 2)
        observed = observed[:, None].expand(-1, frames, -1, -1, -1)

        features = torch.cat(
            [
                noisy.permute(0, 2, 1, 3, 4),
                condition.permute(0, 2, 1, 3, 4),
                observed,
            ],
            dim=2,
        ).reshape(batch * frames, -1, height, width)

        angles = timesteps.float()[:, None] * self.frequencies[None]
        time_embedding = self.time_mlp(
            torch.cat([angles.sin(), angles.cos()], dim=1)
        )
        time_embedding = time_embedding[:, None].expand(
            -1, frames, -1
        ).reshape(batch * frames, -1)

        skip1 = self.enc1(features, time_embedding)
        skip2 = self.enc2(self.down1(skip1), time_embedding)
        hidden = self.middle(self.down2(skip2), time_embedding)

        # Joint temporal attention across the twelve forecast leads.
        channels, small_h, small_w = hidden.shape[1:]
        hidden = self.temporal(
            hidden.reshape(batch, frames, channels, small_h, small_w)
        ).reshape(batch * frames, channels, small_h, small_w)

        # Explicit sizes support the native 93x154 latent grid.
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
        hidden = self.dec2(torch.cat([hidden, skip2], dim=1), time_embedding)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.dec1(torch.cat([hidden, skip1], dim=1), time_embedding)

        noise = self.output(hidden)
        return noise.reshape(
            batch, frames, latent_channels, height, width
        ).permute(0, 2, 1, 3, 4)


class STLDMForecaster(nn.Module):
    """Jointly train a frame VAE, latent forecaster and conditional diffusion."""

    def __init__(
        self,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        latent_channels=8,
        width=32,
        translator_width=64,
        denoiser_width=32,
        train_steps=1000,
        sample_steps=20,
        loss_weights=None,
    ):
        """Construct the notebook architecture and cosine diffusion schedule."""
        super().__init__()
        self.channels = channels
        self.input_frames = input_frames
        self.forecast_frames = forecast_frames
        self.latent_channels = latent_channels
        self.train_steps = train_steps
        self.sample_steps = sample_steps
        self.loss_weights = dict(
            LOSS_WEIGHTS if loss_weights is None else loss_weights
        )

        # Framewise encoding prevents normalization from mixing history/targets.
        self.encoder = nn.Sequential(
            conv_block(channels, width),
            conv_block(width, width, stride=2),
            conv_block(width, 2 * width, stride=2),
            nn.Conv2d(2 * width, 2 * latent_channels, 1),
        )

        # Linear output preserves the shared no-clipping protocol.
        self.decoder = nn.Sequential(
            conv_block(latent_channels, 2 * width),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_block(2 * width, width),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_block(width, width),
            nn.Conv2d(width, channels, 3, padding=1),
        )

        self.translator = nn.Sequential(
            nn.Conv2d(input_frames * latent_channels, translator_width, 1),
            GatedSpatialBlock(translator_width),
            GatedSpatialBlock(translator_width),
            nn.Conv2d(
                translator_width, forecast_frames * latent_channels, 1
            ),
        )
        self.denoiser = LatentDenoiser(
            latent_channels, input_frames, forecast_frames, denoiser_width
        )

        # Calculate on CPU, then register float32 buffers for device transfer.
        grid = torch.linspace(
            0, 1, train_steps + 1, dtype=torch.float64, device="cpu"
        )
        cumulative = torch.cos(
            (grid + 0.008) / 1.008 * math.pi / 2
        ).square()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        self.register_buffer(
            "alphas_cumprod", torch.cumprod(1 - betas, dim=0).float()
        )

    def encode(self, images):
        """Return per-frame latent means and bounded log variances."""
        batch, channels, frames, height, width = images.shape
        flat = images.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )

        # Preserve the notebook's internal scaling; missing -1 becomes -3.
        flat = 2.0 * flat - 1.0

        # Native 372x614 becomes 372x616 before fourfold compression.
        flat = F.pad(flat, (0, (-width) % 4, 0, (-height) % 4), value=0.0)
        mean, logvar = self.encoder(flat).chunk(2, dim=1)
        latent_h, latent_w = mean.shape[-2:]

        mean = mean.reshape(
            batch, frames, self.latent_channels, latent_h, latent_w
        ).permute(0, 2, 1, 3, 4)
        logvar = logvar.reshape(
            batch, frames, self.latent_channels, latent_h, latent_w
        ).permute(0, 2, 1, 3, 4)

        # This bounds posterior variance, not predicted image values.
        return mean, logvar.clamp(-10.0, 4.0)

    def decode(self, latents, image_size):
        """Decode into the original image scale and remove padding."""
        batch, channels, frames, height, width = latents.shape
        flat = latents.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        images = self.decoder(flat)
        image_h, image_w = image_size
        images = images[..., :image_h, :image_w]
        images = images.reshape(
            batch, frames, self.channels, image_h, image_w
        ).permute(0, 2, 1, 3, 4)
        return (images + 1.0) / 2.0

    def translate(self, history):
        """Predict twelve latent frames from five observed latent means."""
        batch, _, _, height, width = history.shape
        prediction = self.translator(history.flatten(1, 2))
        return prediction.reshape(
            batch, self.latent_channels, self.forecast_frames, height, width
        )

    def latent_coverage(self, mask):
        """Pool validity over channels and 4x4 spatial cells.

        These are coverage weights, not exact encoder receptive-field masks.
        Computational padding contributes zero valid coverage.
        """
        batch, _, frames, height, width = mask.shape
        coverage = mask.float().mean(dim=1).reshape(
            batch * frames, 1, height, width
        )
        coverage = F.pad(
            coverage, (0, (-width) % 4, 0, (-height) % 4), value=0.0
        )
        coverage = F.avg_pool2d(coverage, kernel_size=4, stride=4)
        return coverage.reshape(
            batch, frames, 1, *coverage.shape[-2:]
        ).permute(0, 2, 1, 3, 4)

    def training_loss(self, inputs, targets, input_mask, target_mask):
        """Return joint loss and detached component means.

        The trainer must capture both validity masks before filling missing
        pixels with -1. All four components receive gradients jointly.
        """
        images = torch.cat([inputs, targets], dim=2)
        image_mask = torch.cat([input_mask, target_mask], dim=2)

        # Reconstruct observed and future training images through the shared VAE.
        mean, logvar = self.encode(images)
        latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        reconstruction = self.decode(latents, images.shape[-2:])

        # Forecast conditioning uses only the five observed frames.
        history = mean[:, :, :self.input_frames]
        condition = self.translate(history)
        preliminary_forecast = self.decode(condition, targets.shape[-2:])

        coverage = self.latent_coverage(image_mask)
        future_coverage = coverage[:, :, self.input_frames:]
        clean_future = latents[:, :, self.input_frames:]
        clean_future = torch.where(future_coverage > 0, clean_future, 0.0)

        # Sample one diffusion timestep per sequence.
        timesteps = torch.randint(
            self.train_steps, (inputs.shape[0],), device=inputs.device
        )
        alpha = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1, 1)
        noise = torch.randn_like(clean_future)
        noisy = alpha.sqrt() * clean_future + (1 - alpha).sqrt() * noise
        estimated_noise = self.denoiser(noisy, history, condition, timesteps)

        kl_values = 0.5 * (
            mean.square() + logvar.exp() - 1.0 - logvar
        )
        components = {
            "reconstruction": masked_mse(
                reconstruction, images, image_mask
            ),
            "forecast": masked_mse(
                preliminary_forecast, targets, target_mask
            ),
            "denoising": latent_mean(
                (estimated_noise - noise).square(), future_coverage
            ),
            "kl": latent_mean(kl_values, coverage),
        }

        # Keep gradients through the encoder, including encoded targets.
        total = sum(
            self.loss_weights[name] * value
            for name, value in components.items()
        )
        return total, {
            name: value.detach() for name, value in components.items()
        }

    @torch.inference_mode()
    def deterministic_forecast(self, inputs):
        """Decode the translator's forecast without diffusion sampling."""
        history, _ = self.encode(inputs)
        return self.decode(self.translate(history), inputs.shape[-2:])

    @torch.inference_mode()
    def sample(self, inputs, generator=None):
        """Generate one forecast using the shared DDIM sampling protocol.

        The trainer supplies a CPU generator and saves its state when
        checkpointing a partially completed validation pass.
        """
        history, _ = self.encode(inputs)
        condition = self.translate(history)

        latents = torch.randn(
            condition.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(inputs.device)

        # Leading timestep spacing: 950, 900, ..., 0 for the default settings.
        stride = self.train_steps // self.sample_steps
        for index in reversed(range(self.sample_steps)):
            timestep = index * stride
            previous = timestep - stride
            times = torch.full(
                (inputs.shape[0],),
                timestep,
                device=inputs.device,
                dtype=torch.long,
            )
            noise = self.denoiser(latents, history, condition, times)

            alpha = self.alphas_cumprod[timestep]
            alpha_previous = (
                self.alphas_cumprod[previous]
                if previous >= 0
                else latents.new_tensor(1.0)
            )

            # DDIM with eta=0; retain unclipped latent and image predictions.
            clean = (latents - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            latents = (
                alpha_previous.sqrt() * clean
                + (1 - alpha_previous).sqrt() * noise
            )

        return self.decode(latents, inputs.shape[-2:])

    def forward(self, inputs, generator=None):
        """Return forecasts with shape B,C,12,H,W."""
        return self.sample(inputs, generator=generator)