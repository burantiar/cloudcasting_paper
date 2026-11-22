"""Standard FNO-2D for Cloudcasting.

Input:  (batch, 11 channels, 5 observations, height, width)
Output: (batch, 11 channels, 12 forecasts, height, width)

Matches notebook 10. Spatial dimensions are handled dynamically, including
the native 372 x 614 grid. Each forecast advances time by fifteen minutes.
"""

import torch
from torch import nn
import torch.nn.functional as F


# Shared architecture settings for training and checkpoint metadata.
MODEL_CONFIG = {
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "width": 32,
    "depth": 4,
    "modes": (12, 12),
    "projection_width": 128,
    "padding": 8,
}


class SpectralConv2d(nn.Module):
    """Mix spatial Fourier coefficients using learned complex weights."""

    def __init__(self, width, modes):
        """Set the channel width and retained Fourier modes."""
        super().__init__()
        self.modes_y, self.modes_x = modes

        # Separate weights for positive/zero and negative vertical frequencies.
        # Store real and imaginary parts as ordinary real-valued parameters.
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

        # Complex multiplication: (a + ib)(c + id).
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
        """Transform spatially, mix retained modes and return to image space."""
        height, width = features.shape[-2:]
        spectrum = torch.fft.rfft2(features, norm="ortho")
        filtered = torch.zeros_like(spectrum)

        my, mx = self.modes_y, self.modes_x

        # The horizontal FFT dimension contains nonnegative frequencies only.
        filtered[:, :, :my, :mx] = self.mix(
            spectrum[:, :, :my, :mx], self.weights[0]
        )
        filtered[:, :, -my:, :mx] = self.mix(
            spectrum[:, :, -my:, :mx], self.weights[1]
        )

        return torch.fft.irfft2(
            filtered, s=(height, width), norm="ortho"
        )


class FNO2DForecaster(nn.Module):
    """Generate absolute forecasts using a shared autoregressive FNO."""

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
    ):
        """Build the lifting, Fourier and output projection layers."""
        super().__init__()
        self.forecast_frames = forecast_frames
        self.padding = padding

        # Lift 55 observed values plus two spatial coordinates at each pixel.
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

        # Predict absolute channel values without sigmoid or clipping.
        self.project = nn.Sequential(
            nn.Conv2d(width, projection_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(projection_width, channels, kernel_size=1),
        )

    def predict_next(self, history, coordinates):
        """Predict the next frame from the current five-frame history."""
        height, width = history.shape[-2:]

        # B,C,T,H,W -> B,(C*T),H,W, preserving each channel's time order.
        features = torch.cat(
            (history.flatten(1, 2), coordinates), dim=1
        )
        features = self.lift(features)

        # Temporary padding of hidden features; observations are not resized.
        if self.padding:
            features = F.pad(
                features, (0, self.padding, 0, self.padding)
            )

        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers)
        ):
            features = spectral(features) + pointwise(features)

            # Match the notebook: GELU between the four Fourier layers.
            if index < len(self.spectral_layers) - 1:
                features = F.gelu(features)

        # Remove padding before projecting back to the eleven channels.
        features = features[..., :height, :width]
        return self.project(features)

    def forward(self, inputs):
        """Return twelve forecasts with gradients through the full rollout."""
        batch, _, _, height, width = inputs.shape

        # Fixed normalized row/column coordinates for the current grid.
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

            # Advance the history using the prediction, without detaching.
            # There is no teacher forcing or persistence output addition.
            history = torch.cat(
                (history[:, :, 1:], prediction.unsqueeze(2)), dim=2
            )

        return torch.stack(predictions, dim=2)