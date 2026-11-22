"""Standalone IAM4VP matching notebook 06_iam4vp_exploratory.

Full-resolution configuration:
    Input:  (batch, 11, 5, 372, 614)
    Output: (batch, 11, 12, 372, 614)

Preprocessing, device selection and training belong in train.py.
"""

import math

import torch
from torch import nn


SOURCE_REPOSITORY = (
    "https://github.com/seominseok0429/"
    "Masked-Autoregressive-Model-for-Weather-Forecasting"
)

MODEL_CONFIG = {
    "image_size": (372, 614),
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "hidden_channels": 32,
    "temporal_depth": 4,
}


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        """Apply channel normalization to a BCHW tensor."""
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpatialBlock(nn.Module):
    """Convolution, optional PixelShuffle, channel normalization and SiLU."""

    def __init__(self, in_channels, out_channels, stride=1, upsample=False):
        super().__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels * 4 if upsample else out_channels,
            kernel_size=3,
            stride=1 if upsample else stride,
            padding=1,
        )

        # Preserve the notebook's spatial-convolution initialization.
        nn.init.trunc_normal_(conv.weight, std=0.02)
        nn.init.zeros_(conv.bias)

        self.layers = nn.Sequential(
            conv,
            nn.PixelShuffle(2) if upsample else nn.Identity(),
            ChannelLayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        """Transform spatial features."""
        return self.layers(x)


class SpatialEncoder(nn.Module):
    """Four spatial stages with strides 1, 2, 1, 2."""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.first = SpatialBlock(channels, hidden_channels)
        self.remaining = nn.Sequential(
            SpatialBlock(hidden_channels, hidden_channels, stride=2),
            SpatialBlock(hidden_channels, hidden_channels),
            SpatialBlock(hidden_channels, hidden_channels, stride=2),
        )

    def forward(self, x):
        """Return compressed features and a full-resolution skip."""
        skip = self.first(x)
        return self.remaining(skip), skip


class LargeKernelAttention(nn.Module):
    """Spatial gating with depthwise and dilated depthwise convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(
                channels, channels, 7,
                padding=9, dilation=3, groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        """Multiply features by their learned spatial attention."""
        return x * self.attention(x)


class TemporalBlock(nn.Module):
    """Time-conditioned ConvNeXt block with optional channel reduction."""

    def __init__(self, in_channels, out_channels, time_dim=64):
        super().__init__()

        if in_channels != out_channels:
            # Preserve the notebook's generalized grouped bottleneck.
            self.spatial = nn.Conv2d(
                in_channels, out_channels, 7, padding=3,
                groups=math.gcd(in_channels, out_channels),
            )
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.spatial = LargeKernelAttention(out_channels)
            self.shortcut = nn.Identity()

        self.time_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_dim, out_channels),
        )
        self.channel_mlp = nn.Sequential(
            ChannelLayerNorm(out_channels),
            nn.Conv2d(out_channels, 4 * out_channels, 1),
            nn.GELU(),
            nn.Conv2d(4 * out_channels, out_channels, 1),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), 1e-6)
        )

    def forward(self, x, time_embedding):
        """Mix spatial features with the requested forecast lead."""
        time_features = self.time_projection(time_embedding)[:, :, None, None]
        features = self.spatial(x) + time_features
        return self.shortcut(x) + self.layer_scale * self.channel_mlp(features)


class IAM4VPForecaster(nn.Module):
    """Forecast with fixed observations and a separate future-frame queue.

    Inputs use BCTHW layout and the common missing-pixel marker of -1.
    Outputs are linear predictions without clipping.
    """

    def __init__(
        self,
        image_size,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        hidden_channels=32,
        temporal_depth=4,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.forecast_frames = forecast_frames
        self.hidden_channels = hidden_channels

        # Observations and previous predictions have separate encoder weights.
        self.observed_encoder = SpatialEncoder(channels, hidden_channels)
        self.prediction_encoder = SpatialEncoder(channels, hidden_channels)

        height, width = image_size
        latent_height = (height + 3) // 4
        latent_width = (width + 3) // 4

        # At native resolution, the learned mask has shape (12, 32, 93, 154).
        self.mask_token = nn.Parameter(torch.zeros(
            forecast_frames, hidden_channels, latent_height, latent_width
        ))

        # Preserve the notebook's 64-dimensional sinusoidal time embedding.
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(32, dtype=torch.float32) / 31
        )
        self.register_buffer("time_frequencies", frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

        # Seventeen queued latent frames become five internal feature maps.
        queue_channels = (input_frames + forecast_frames) * hidden_channels
        predictor_channels = input_frames * hidden_channels
        self.predictor = nn.ModuleList(
            [TemporalBlock(queue_channels, predictor_channels)]
            + [
                TemporalBlock(predictor_channels, predictor_channels)
                for _ in range(temporal_depth - 1)
            ]
        )

        self.decoder = nn.Sequential(
            SpatialBlock(hidden_channels, hidden_channels, upsample=True),
            SpatialBlock(hidden_channels, hidden_channels),
            SpatialBlock(hidden_channels, hidden_channels, upsample=True),
        )
        self.skip_fusion = SpatialBlock(2 * hidden_channels, hidden_channels)

        # Combine decoded features and refine one forecast frame.
        self.temporal_fusion = nn.Conv2d(
            input_frames * hidden_channels, hidden_channels, 1
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            LargeKernelAttention(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
        )
        self.output_projection = nn.Conv2d(hidden_channels, channels, 1)

    def forward_step(self, x, previous_predictions):
        """Predict one lead using fixed observations and detached forecasts.

        Training can backpropagate each lead separately, then perform one
        optimizer update after all twelve leads.
        """
        batch, channels, history, height, width = x.shape
        hidden = self.hidden_channels
        lead_index = len(previous_predictions)

        # Keep all five observed frames available at every forecast lead.
        observed_images = x.permute(0, 2, 1, 3, 4).reshape(
            batch * history, channels, height, width
        )
        observed, skip = self.observed_encoder(observed_images)
        latent_height, latent_width = observed.shape[-2:]
        observed = observed.reshape(
            batch, history, hidden, latent_height, latent_width
        )

        # Unfilled future slots retain their learned placeholders.
        future = self.mask_token[lead_index:].unsqueeze(0).expand(
            batch, -1, -1, -1, -1
        )

        if previous_predictions:
            # Detach forecast feedback, while training its separate encoder.
            previous = torch.stack(previous_predictions, dim=1).detach()
            encoded_previous, _ = self.prediction_encoder(
                previous.reshape(
                    batch * lead_index, channels, height, width
                )
            )
            encoded_previous = encoded_previous.reshape(
                batch, lead_index, hidden, latent_height, latent_width
            )
            future = torch.cat((encoded_previous, future), dim=1)

        features = torch.cat((observed, future), dim=1).reshape(
            batch, -1, latent_height, latent_width
        )

        # Encoding indices 0, 100, ..., 1100 represent leads 15–180 minutes.
        angles = (100.0 * lead_index) * self.time_frequencies
        time_embedding = torch.cat((angles.sin(), angles.cos()))
        time_embedding = self.time_mlp(
            time_embedding.unsqueeze(0).expand(batch, -1)
        )

        for block in self.predictor:
            features = block(features, time_embedding)

        decoded = self.decoder(features.reshape(
            batch * history, hidden, latent_height, latent_width
        ))

        # Crop rounding-related border pixels: native decoder width 616 -> 614.
        decoded = decoded[..., :height, :width]
        decoded = self.skip_fusion(torch.cat((decoded, skip), dim=1))
        decoded = decoded.reshape(
            batch, history * hidden, height, width
        )

        fused = self.temporal_fusion(decoded)
        return self.output_projection(fused + self.refinement(fused))

    def forward(self, x):
        """Produce all twelve forecasts in BCTHW layout."""
        predictions = []
        for _ in range(self.forecast_frames):
            predictions.append(self.forward_step(x, predictions))
        return torch.stack(predictions, dim=2)