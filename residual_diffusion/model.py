"""Residual-correction diffusion matching notebook 15.

Full-resolution inputs:  (batch, 11, 5, 372, 614).
Full-resolution outputs: (batch, 11, 12, 372, 614).

The trainer supplies IAM4VP loaded from a fixed full-resolution checkpoint.
Only the residual denoiser is trained.

Dependencies match notebook 14:
    diffusers==0.30.3
    huggingface-hub==0.25.1
"""

import torch
from torch import nn
import torch.nn.functional as F

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel


# Preserve notebook 15's architecture and diffusion settings.
MODEL_CONFIG = {
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "block_out_channels": (32, 64, 128),
    "train_steps": 1000,
    "sample_steps": 20,
    "residual_scale": 2.0,
}


class ResidualDiffusionForecaster(nn.Module):
    """Generate joint forecast corrections around a frozen IAM4VP predictor."""

    def __init__(
        self,
        predictor,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        block_out_channels=(32, 64, 128),
        train_steps=1000,
        sample_steps=20,
        residual_scale=2.0,
    ):
        """Create the denoiser around an already loaded deterministic predictor."""
        super().__init__()

        # Register the predictor so its weights are included in our checkpoints.
        self.predictor = predictor.requires_grad_(False).eval()

        self.channels = channels
        self.input_frames = input_frames
        self.forecast_frames = forecast_frames
        self.train_steps = train_steps
        self.sample_steps = sample_steps
        self.residual_scale = residual_scale

        # Three U-Net resolution levels require dimensions divisible by four.
        self.spatial_multiple = 2 ** (len(block_out_channels) - 1)

        # Input: 12 noisy residual frames + 5 observations + 12 IAM4VP forecasts.
        # Stack time within each spectral channel: 11 * 29 = 319 input channels.
        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=channels * (input_frames + 2 * forecast_frames),
            out_channels=channels * forecast_frames,
            layers_per_block=2,
            block_out_channels=tuple(block_out_channels),
            down_block_types=("DownBlock2D",) * len(block_out_channels),
            up_block_types=("UpBlock2D",) * len(block_out_channels),
            norm_num_groups=32,
            dropout=0.0,
            add_attention=False,
        )

        scheduler_settings = {
            "num_train_timesteps": train_steps,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "clip_sample": False,
        }

        self.noise_scheduler = DDPMScheduler(**scheduler_settings)
        self.sampler = DDIMScheduler(
            **scheduler_settings,
            timestep_spacing="leading",
            set_alpha_to_one=True,
        )

    def train(self, mode=True):
        """Change denoiser mode while keeping IAM4VP in evaluation mode."""
        super().train(mode)
        self.predictor.eval()
        return self

    @torch.no_grad()
    def make_conditioning(self, inputs):
        """Compute the frozen forecast and prepare image-valued conditioning."""
        forecast = self.predictor(inputs)

        # Match the notebook's internal transformation.
        # The shared missing-input marker -1 becomes -3 after transformation.
        conditioning = torch.cat(
            [2.0 * inputs - 1.0, 2.0 * forecast - 1.0],
            dim=2,
        )
        return forecast, conditioning

    def predict_noise(self, noisy_residual, conditioning, timesteps):
        """Predict Gaussian noise for all twelve residual frames jointly."""
        height, width = noisy_residual.shape[-2:]

        # B,C,(12+5+12),H,W -> B,319,H,W.
        net_input = torch.cat(
            [noisy_residual, conditioning], dim=2
        ).flatten(1, 2)

        # Native 372x614 becomes 372x616 inside the U-Net.
        # Padding preserves all original pixels without resizing the input.
        pad_h = (-height) % self.spatial_multiple
        pad_w = (-width) % self.spatial_multiple
        net_input = F.pad(
            net_input, (0, pad_w, 0, pad_h), value=0.0
        )

        predicted_noise = self.unet(net_input, timesteps).sample

        # Remove computational padding and restore B,C,12,H,W.
        return predicted_noise[..., :height, :width].reshape_as(
            noisy_residual
        )

    def denoising_loss(self, inputs, targets, target_mask):
        """Return full-image masked L1 noise-prediction loss.

        Inputs and targets retain the shared dataset scale.
        target_mask must capture validity before missing values are filled.
        """
        forecast, conditioning = self.make_conditioning(inputs)

        # Residuals remain zero-centred: multiply by two without subtracting one.
        # This equals the difference between images transformed using 2*x - 1.
        clean_residual = torch.where(
            target_mask,
            self.residual_scale * (targets - forecast),
            0.0,
        )

        # Draw one diffusion timestep per sequence and independent pixel noise.
        timesteps = torch.randint(
            self.train_steps,
            (inputs.shape[0],),
            device=inputs.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_residual)

        noisy_residual = self.noise_scheduler.add_noise(
            clean_residual, noise, timesteps
        )
        predicted_noise = self.predict_noise(
            noisy_residual, conditioning, timesteps
        )

        # Missing target pixels contribute neither loss nor gradient.
        errors = torch.where(
            target_mask,
            predicted_noise - noise,
            0.0,
        ).abs()

        return errors.sum() / target_mask.sum().clamp_min(1)

    @torch.inference_mode()
    def sample(self, inputs, generator=None):
        """Generate one joint forecast in the original dataset scale.

        An optional CPU generator controls the initial diffusion noise.
        The trainer manages its state for validation and checkpoint resume.
        """
        forecast, conditioning = self.make_conditioning(inputs)

        # Retain CPU noise generation for consistency with the Mac notebook.
        residual = torch.randn(
            forecast.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(inputs.device)

        self.sampler.set_timesteps(self.sample_steps)

        # Every iteration denoises all twelve correction frames.
        # With eta=0, sampling is deterministic for a given initial noise draw.
        for timestep in self.sampler.timesteps.tolist():
            predicted_noise = self.predict_noise(
                residual, conditioning, timestep
            )
            residual = self.sampler.step(
                predicted_noise,
                timestep,
                residual,
                eta=0.0,
            ).prev_sample

        # Reconstruct forecasts in the shared dataset scale without clipping.
        return forecast + residual / self.residual_scale

    def forward(self, inputs, generator=None):
        """Return forecasts with shape B,C,12,H,W."""
        return self.sample(inputs, generator=generator)