"""Conditional diffusion matching notebook 14 at native image resolution.

Inputs:  (batch, 11, 5, height, width).
Outputs: (batch, 11, 12, height, width).

Based on the Alan Turing ocf-diffusion repository, with the compact
joint-forecast adaptation recorded in notebook 14.

Dependencies match the notebook:
    diffusers==0.30.3
    huggingface-hub==0.25.1
"""

import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# Keep the notebook architecture and diffusion settings unchanged.
MODEL_CONFIG = {
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "block_out_channels": (32, 64, 128),
    "train_steps": 1000,
    "sample_steps": 20,
}


class DiffusionForecaster(nn.Module):
    """Generate twelve future frames jointly from five observed frames."""

    def __init__(
        self,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        block_out_channels=(32, 64, 128),
        train_steps=1000,
        sample_steps=20,
    ):
        """Create the conditional U-Net and matching diffusion schedules."""
        super().__init__()

        self.channels = channels
        self.input_frames = input_frames
        self.forecast_frames = forecast_frames
        self.train_steps = train_steps
        self.sample_steps = sample_steps

        # Three resolution levels require dimensions divisible by four.
        self.spatial_multiple = 2 ** (len(block_out_channels) - 1)

        # Combine five observed frames with twelve noisy future frames.
        # Predict Gaussian noise for all twelve future frames simultaneously.
        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=channels * (input_frames + forecast_frames),
            out_channels=channels * forecast_frames,
            layers_per_block=2,
            block_out_channels=block_out_channels,
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

    def predict_noise(self, noisy_future, conditioning, timesteps):
        """Predict future noise using observed history and the noise level."""
        height, width = noisy_future.shape[-2:]

        # Stack time within each spectral channel, matching notebook 14.
        # B,C,(12+5),H,W becomes B,187,H,W.
        net_input = torch.cat(
            [noisy_future, conditioning], dim=2
        ).flatten(1, 2)

        # Native 372x614 inputs become 372x616 inside the U-Net.
        # Padding preserves every original image pixel.
        pad_h = (-height) % self.spatial_multiple
        pad_w = (-width) % self.spatial_multiple
        net_input = F.pad(
            net_input, (0, pad_w, 0, pad_h), value=0.0
        )

        predicted_noise = self.unet(net_input, timesteps).sample

        # Remove computational padding and restore B,C,12,H,W.
        return predicted_noise[..., :height, :width].reshape_as(
            noisy_future
        )

    def denoising_loss(self, inputs, targets, target_mask):
        """Return masked L1 noise-prediction loss over all target frames.

        Inputs and targets use the existing dataset scale. The trainer
        must capture target_mask before filling missing values with -1.
        """
        # Apply the same internal transformation as in the notebook.
        # Missing conditioning values (-1) become -3.
        conditioning = inputs * 2.0 - 1.0

        # Missing targets receive a neutral internal fill and no loss weight.
        clean_future = torch.where(
            target_mask,
            targets * 2.0 - 1.0,
            0.0,
        )

        # Sample one noise level per sequence and independent pixel noise.
        timesteps = torch.randint(
            self.train_steps,
            (inputs.shape[0],),
            device=inputs.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_future)

        noisy_future = self.noise_scheduler.add_noise(
            clean_future, noise, timesteps
        )
        predicted_noise = self.predict_noise(
            noisy_future, conditioning, timesteps
        )

        # Reduce over valid pixels across samples, channels and forecast leads.
        errors = torch.where(
            target_mask,
            predicted_noise - noise,
            0.0,
        ).abs()

        return errors.sum() / target_mask.sum().clamp_min(1)

    @torch.inference_mode()
    def sample(self, inputs, generator=None):
        """Generate one joint forecast in the original dataset scale.

        An optional CPU torch.Generator controls the initial noise.
        The trainer manages its state for reproducible validation and resume.
        """
        conditioning = inputs * 2.0 - 1.0

        # Retain CPU noise generation for consistency with notebook 14.
        shape = (
            inputs.shape[0],
            self.channels,
            self.forecast_frames,
            *inputs.shape[-2:],
        )
        forecast = torch.randn(
            shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(inputs.device)

        self.sampler.set_timesteps(self.sample_steps)

        # Each DDIM iteration denoises the complete twelve-frame forecast.
        # eta=0 introduces no additional randomness after the initial noise.
        for timestep in self.sampler.timesteps.tolist():
            predicted_noise = self.predict_noise(
                forecast, conditioning, timestep
            )
            forecast = self.sampler.step(
                predicted_noise,
                timestep,
                forecast,
                eta=0.0,
            ).prev_sample

        # Convert back before image-space validation; do not clip predictions.
        return (forecast + 1.0) / 2.0

    def forward(self, inputs, generator=None):
        """Return forecasts with shape B,C,12,H,W."""
        return self.sample(inputs, generator=generator)