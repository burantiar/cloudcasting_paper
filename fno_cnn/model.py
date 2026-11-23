"""FNO–CNN hybrid for full-resolution Cloudcasting.

Input:  (batch, 11 channels, 5 observations, height, width)
Output: (batch, 11 channels, 12 forecasts, height, width)

Matches notebook 12. Each operator layer combines spectral, pointwise
and local convolution branches. Spatial dimensions are handled dynamically.
"""

import torch
from torch import nn
import torch.nn.functional as F


# Shared settings for model construction and training metadata.
MODEL_CONFIG = {
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "width": 32,
    "depth": 4,
    "modes": (12, 12),
    "projection_width": 128,
    "padding": 8,
    "local_kernel_size": 3,
    "local_stride": 1,
    "local_padding": 1,
    "local_bias": True,
}


class SpectralConv2d(nn.Module):
    """Mix spatial Fourier coefficients using learned complex weights."""

    def __init__(self, width, modes):
        """Initialize spectral weights exactly as in standard FNO."""
        super().__init__()
        self.modes_y, self.modes_x = modes

        # Two vertical frequency bands, with real/imaginary components.
        self.weights = nn.Parameter(
            torch.rand(
                2, width, width, self.modes_y, self.modes_x, 2
            ) / width**2
        )

    @staticmethod
    def mix(coefficients, weights):
        """Multiply complex coefficients using real-valued contractions."""
        real_weight, imag_weight = weights.unbind(dim=-1)
        real_input = coefficients.real
        imag_input = coefficients.imag

        real_output = (
            torch.einsum("bihw,iohw->bohw", real_input, real_weight)
            - torch.einsum("bihw,iohw->bohw", imag_input, imag_weight)
        )
        imag_output = (
            torch.einsum("bihw,iohw->bohw", real_input, imag_weight)
            + torch.einsum("bihw,iohw->bohw", imag_input, real_weight)
        )
        return torch.complex(real_output, imag_output)

    def forward(self, features):
        """Apply spatial Fourier mixing and return to image space."""
        height, width = features.shape[-2:]
        spectrum = torch.fft.rfft2(features, norm="ortho")
        filtered = torch.zeros_like(spectrum)

        my, mx = self.modes_y, self.modes_x

        # Retain positive/zero and negative vertical frequency bands.
        filtered[:, :, :my, :mx] = self.mix(
            spectrum[:, :, :my, :mx], self.weights[0]
        )
        filtered[:, :, -my:, :mx] = self.mix(
            spectrum[:, :, -my:, :mx], self.weights[1]
        )

        return torch.fft.irfft2(
            filtered, s=(height, width), norm="ortho"
        )


class FNOCNNForecaster(nn.Module):
    """Generate absolute forecasts using Fourier and local CNN branches."""

    def __init__(
        self,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        width=32,
        depth=4,
        modes=(12, 12),
        projection_width=128,
        padding=8,
        local_kernel_size=3,
        local_stride=1,
        local_padding=1,
        local_bias=True,
    ):
        """Initialize the standard FNO backbone first, then CNN branches."""
        super().__init__()
        self.forecast_frames = forecast_frames
        self.padding = padding

        # Preserve the standard FNO's module construction order.
        self.lift = nn.Conv2d(
            channels * input_frames + 2, width, kernel_size=1
        )

        self.spectral_layers = nn.ModuleList(
            SpectralConv2d(width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(depth)
        )

        self.project = nn.Sequential(
            nn.Conv2d(width, projection_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(projection_width, channels, kernel_size=1),
        )

        # Construct these last to preserve the backbone's seeded initialization.
        # The 3x3 branches retain the incoming feature dimensions.
        self.local_layers = nn.ModuleList(
            nn.Conv2d(
                width,
                width,
                kernel_size=local_kernel_size,
                stride=local_stride,
                padding=local_padding,
                bias=local_bias,
            )
            for _ in range(depth)
        )

    def predict_next(self, history, coordinates):
        """Predict one absolute frame from the five-frame history."""
        height, width = history.shape[-2:]

        # B,C,T,H,W -> B,(C*T),H,W, preserving each channel's time order.
        features = torch.cat(
            (history.flatten(1, 2), coordinates), dim=1
        )
        features = self.lift(features)

        # Temporarily zero-pad hidden features at the bottom and right.
        if self.padding:
            features = F.pad(
                features, (0, self.padding, 0, self.padding)
            )

        for index, (spectral, pointwise, local) in enumerate(
            zip(
                self.spectral_layers,
                self.pointwise_layers,
                self.local_layers,
            )
        ):
            # Every branch receives the same incoming features.
            features = (
                spectral(features)
                + pointwise(features)
                + local(features)
            )

            # Match standard FNO's activation pattern.
            if index < len(self.spectral_layers) - 1:
                features = F.gelu(features)

        # Remove padding before projecting to the eleven output channels.
        features = features[..., :height, :width]
        return self.project(features)

    def forward(self, inputs):
        """Return twelve absolute forecasts with connected rollout gradients."""
        batch, _, _, height, width = inputs.shape

        # Fixed normalized coordinates for the current spatial grid.
        rows = torch.linspace(
            0, 1, height, device=inputs.device, dtype=inputs.dtype
        )
        columns = torch.linspace(
            0, 1, width, device=inputs.device, dtype=inputs.dtype
        )
        yy, xx = torch.meshgrid(rows, columns, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=0)
        coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1, -1)

        history = inputs
        predictions = []

        for _ in range(self.forecast_frames):
            prediction = self.predict_next(history, coordinates)
            predictions.append(prediction)

            # Advance the history using the absolute prediction.
            # Preserve gradients throughout the twelve-step rollout.
            history = torch.cat(
                (history[:, :, 1:], prediction.unsqueeze(2)), dim=2
            )

        return torch.stack(predictions, dim=2)