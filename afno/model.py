"""Compact standalone AFNO for full-resolution Cloudcasting.

Input:  (batch, 11 channels, 5 observations, height, width)
Output: (batch, 11 channels, 12 forecasts, height, width)

Matches notebook 13. Predictions are absolute values, generated through
a shared one-step model with gradients retained across the full rollout.
"""

import torch
from torch import nn
import torch.nn.functional as F


# Shared settings for model construction and training metadata.
MODEL_CONFIG = {
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "patch_size": 4,
    "embed_dim": 128,
    "depth": 4,
    "num_blocks": 8,
    "hidden_size_factor": 1,
    "mlp_ratio": 2,
    "sparsity_threshold": 0.01,
    "hard_thresholding_fraction": 1.0,
    "dropout": 0.0,
    "norm_eps": 1e-6,
    "spectral_init_std": 0.02,
    "padding_value": -1.0,
}


class AFNOMixer2d(nn.Module):
    """Apply grouped nonlinear channel mixing in spatial Fourier space."""

    def __init__(
        self,
        embed_dim,
        num_blocks,
        hidden_size_factor,
        sparsity_threshold,
        spectral_init_std,
    ):
        """Initialize complex weights shared across all Fourier modes."""
        super().__init__()

        if embed_dim % num_blocks != 0:
            raise ValueError("embed_dim must be divisible by num_blocks.")

        self.num_blocks = num_blocks
        self.block_size = embed_dim // num_blocks
        self.sparsity_threshold = sparsity_threshold
        hidden_size = self.block_size * hidden_size_factor

        # The leading dimension stores real and imaginary components.
        # Match the notebook's Gaussian initialization for weights and biases.
        self.weight1 = nn.Parameter(
            spectral_init_std
            * torch.randn(2, num_blocks, self.block_size, hidden_size)
        )
        self.bias1 = nn.Parameter(
            spectral_init_std * torch.randn(2, num_blocks, hidden_size)
        )
        self.weight2 = nn.Parameter(
            spectral_init_std
            * torch.randn(2, num_blocks, hidden_size, self.block_size)
        )
        self.bias2 = nn.Parameter(
            spectral_init_std * torch.randn(2, num_blocks, self.block_size)
        )

    @staticmethod
    def complex_linear(real, imag, weight, bias):
        """Apply a complex linear map independently within each channel group."""
        real_output = (
            torch.einsum("bhwgi,gio->bhwgo", real, weight[0])
            - torch.einsum("bhwgi,gio->bhwgo", imag, weight[1])
            + bias[0]
        )
        imag_output = (
            torch.einsum("bhwgi,gio->bhwgo", real, weight[1])
            + torch.einsum("bhwgi,gio->bhwgo", imag, weight[0])
            + bias[1]
        )
        return real_output, imag_output

    def forward(self, features):
        """Return a learned Fourier update for B,H,W,C feature tensors."""
        batch, height, width, channels = features.shape
        original_dtype = features.dtype

        # Transform only the spatial dimensions, using float32 FFT inputs.
        spectrum = torch.fft.rfft2(
            features.float(), dim=(1, 2), norm="ortho"
        )
        frequency_width = spectrum.shape[2]
        spectrum = spectrum.reshape(
            batch, height, frequency_width,
            self.num_blocks, self.block_size,
        )

        # First complex transformation and componentwise nonlinearity.
        real, imag = self.complex_linear(
            spectrum.real, spectrum.imag, self.weight1, self.bias1
        )
        real, imag = F.relu(real), F.relu(imag)

        # Second complex transformation.
        real, imag = self.complex_linear(
            real, imag, self.weight2, self.bias2
        )

        # All modes participate; soft-thresholding supplies spectral sparsity.
        real = F.softshrink(real, lambd=self.sparsity_threshold)
        imag = F.softshrink(imag, lambd=self.sparsity_threshold)
        mixed = torch.complex(real, imag).reshape(
            batch, height, frequency_width, channels
        )

        update = torch.fft.irfft2(
            mixed,
            s=(height, width),
            dim=(1, 2),
            norm="ortho",
        )

        # The surrounding block adds the feature shortcut.
        return update.to(original_dtype)


class AFNOBlock(nn.Module):
    """Combine pre-normalized Fourier mixing and a channel MLP."""

    def __init__(
        self,
        embed_dim,
        num_blocks,
        hidden_size_factor,
        mlp_ratio,
        sparsity_threshold,
        dropout,
        norm_eps,
        spectral_init_std,
    ):
        """Build two feature sublayers, each with its own shortcut."""
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mixer = AFNOMixer2d(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            hidden_size_factor=hidden_size_factor,
            sparsity_threshold=sparsity_threshold,
            spectral_init_std=spectral_init_std,
        )

        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        """Apply one shortcut around each learned feature update."""
        features = features + self.mixer(self.norm1(features))
        return features + self.mlp(self.norm2(features))


class AFNOForecaster(nn.Module):
    """Generate absolute forecasts using a shared one-step AFNO."""

    def __init__(
        self,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        patch_size=4,
        embed_dim=128,
        depth=4,
        num_blocks=8,
        hidden_size_factor=1,
        mlp_ratio=2,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        dropout=0.0,
        norm_eps=1e-6,
        spectral_init_std=0.02,
        padding_value=-1.0,
    ):
        """Build the patch embedding, AFNO blocks and image reconstruction head."""
        super().__init__()

        # Match the notebook's all-mode configuration.
        if hard_thresholding_fraction != 1.0:
            raise ValueError("This experiment uses all Fourier modes: fraction=1.0.")

        self.patch_size = patch_size
        self.forecast_frames = forecast_frames
        self.padding_value = padding_value

        # Embed 55 history features plus two spatial coordinate channels.
        self.patch_embed = nn.Conv2d(
            channels * input_frames + 2,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.blocks = nn.ModuleList(
            AFNOBlock(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                hidden_size_factor=hidden_size_factor,
                mlp_ratio=mlp_ratio,
                sparsity_threshold=sparsity_threshold,
                dropout=dropout,
                norm_eps=norm_eps,
                spectral_init_std=spectral_init_std,
            )
            for _ in range(depth)
        )

        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # Predict every pixel and channel within each output patch.
        self.head = nn.Linear(embed_dim, channels * patch_size**2)

    def predict_next(self, history, coordinates):
        """Predict one absolute frame from the current five-frame history."""
        height, width = history.shape[-2:]
        features = torch.cat(
            (history.flatten(1, 2), coordinates), dim=1
        )

        # Pad to complete patches without resizing the observations.
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h or pad_w:
            features = F.pad(
                features,
                (0, pad_w, 0, pad_h),
                value=self.padding_value,
            )

        # AFNO blocks operate on B,patch_height,patch_width,features.
        features = self.patch_embed(features).permute(0, 2, 3, 1)
        for block in self.blocks:
            features = block(features)

        patch_pixels = self.head(self.norm(features))

        # Reconstruct the image and remove any bottom/right padding.
        patch_pixels = patch_pixels.permute(0, 3, 1, 2).contiguous()
        prediction = F.pixel_shuffle(patch_pixels, self.patch_size)
        return prediction[..., :height, :width]

    def forward(self, inputs):
        """Return twelve forecasts with gradients through every rollout step."""
        batch, _, _, height, width = inputs.shape

        # Fixed coordinates work with both cached and native spatial grids.
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

            # Advance the five-frame history with the new absolute prediction.
            # Keep the complete twelve-step computation graph connected.
            history = torch.cat(
                (history[:, :, 1:], prediction.unsqueeze(2)), dim=2
            )

        return torch.stack(predictions, dim=2)