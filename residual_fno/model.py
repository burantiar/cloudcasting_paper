"""Persistence-residual FNO-2D for Cloudcasting.

Input:  (batch, 11 channels, 5 observations, height, width)
Output: (batch, 11 channels, 12 forecasts, height, width)

Matches notebook 11. Each prediction adds an FNO correction to the latest
frame reference. Spatial dimensions are handled dynamically, without resizing.
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
    "missing_pixel_marker": -1.0,
    "initial_reference_missing_fill": 0.0,
}


class SpectralConv2d(nn.Module):
    """Mix spatial Fourier coefficients using learned complex weights."""

    def __init__(self, width, modes):
        """Initialize spectral weights exactly as in the standard FNO."""
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


class ResidualFNO2DForecaster(nn.Module):
    """Generate forecasts by adding FNO corrections to the latest frame."""

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
        missing_pixel_marker=-1.0,
        initial_reference_missing_fill=0.0,
    ):
        """Build the same backbone and initialization as notebook 11."""
        super().__init__()
        self.forecast_frames = forecast_frames
        self.padding = padding
        self.missing_pixel_marker = missing_pixel_marker
        self.initial_reference_missing_fill = initial_reference_missing_fill

        # Preserve the standard FNO's layer construction and initialization order.
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

        # Keep random initialization, including the final correction layer.
        self.project = nn.Sequential(
            nn.Conv2d(width, projection_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(projection_width, channels, kernel_size=1),
        )

    def predict_correction(self, history, coordinates):
        """Predict an additive correction from the five-frame history."""
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

        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers)
        ):
            features = spectral(features) + pointwise(features)

            if index < len(self.spectral_layers) - 1:
                features = F.gelu(features)

        # Remove padding before producing the eleven-channel correction.
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

        # Fill missing pixels in the initial output reference with zero.
        # The input history keeps the missing-value markers supplied by training.
        reference = inputs[:, :, -1]
        valid_reference = (
            torch.isfinite(reference)
            & (reference != self.missing_pixel_marker)
        )
        reference = torch.where(
            valid_reference,
            reference,
            self.initial_reference_missing_fill,
        )

        history = inputs
        predictions = []

        for _ in range(self.forecast_frames):
            correction = self.predict_correction(history, coordinates)
            prediction = reference + correction
            predictions.append(prediction)

            # Advance the five-frame history using the new prediction.
            history = torch.cat(
                (history[:, :, 1:], prediction.unsqueeze(2)), dim=2
            )

            # Use later predictions unchanged, with gradients preserved.
            # Missing-reference replacement applies only to the last observation.
            reference = prediction

        return torch.stack(predictions, dim=2)