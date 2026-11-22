"""Standalone PredFormer Fac-T-S for full-resolution cloudcasting.

Input:  (batch, 11, 5, 372, 614)
Output: (batch, 11, 12, 372, 614)

Matches notebook 08: two temporal blocks followed by two spatial blocks,
with a shared five-frame predictor rolled forward to produce twelve forecasts.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


REFERENCE_REPOSITORY = "https://github.com/yyyujintang/PredFormer"
REFERENCE_COMMIT = "a1493c4ade8eed740d7aed11da819e9aa7914c2d"

MODEL_CONFIG = {
    "image_size": (372, 614),
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "patch_size": 8,
    "dim": 128,
    "heads": 4,
    "depth": 2,
    "mlp_ratio": 2,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "drop_path": 0.25,
    "padding_value": -1.0,
}


def sinusoidal_positions(length, dim):
    """Encode flattened time-and-patch positions using fixed sinusoids."""
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dim, 2, dtype=torch.float32)
        / dim
    )
    angles = positions * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


def drop_residual(x, probability, training):
    """Randomly drop a residual branch per attention sequence during training."""
    if not training or probability == 0:
        return x

    keep = 1.0 - probability
    mask = torch.rand((x.shape[0], 1, 1), device=x.device) < keep
    return x * mask.to(x.dtype) / keep


def initialize_transformer(module):
    """Initialize transformer linear layers as in the notebook."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GatedTransformerBlock(nn.Module):
    """Apply pre-normalized self-attention followed by a SwiGLU residual branch."""

    def __init__(
        self, dim, heads, mlp_ratio, dropout, attention_dropout, drop_path
    ):
        super().__init__()
        self.heads = heads
        self.drop_path = drop_path

        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attention_output = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        hidden_dim = dim * mlp_ratio
        self.ffn_norm = nn.LayerNorm(dim)
        self.gate_and_value = nn.Linear(dim, 2 * hidden_dim)
        self.ffn_output = nn.Linear(hidden_dim, dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Process a batch of temporal or spatial token sequences."""
        groups, length, dim = x.shape

        qkv = self.qkv(self.attention_norm(x))
        qkv = qkv.reshape(
            groups, length, 3, self.heads, dim // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # Match the notebook: dropout follows the attention output projection.
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).reshape(groups, length, dim)
        attended = self.attention_dropout(self.attention_output(attended))
        x = x + drop_residual(attended, self.drop_path, self.training)

        # SwiGLU combines a SiLU-activated gate with a value projection.
        gate, value = self.gate_and_value(self.ffn_norm(x)).chunk(2, dim=-1)
        update = self.ffn_dropout(F.silu(gate) * value)
        update = self.ffn_dropout(self.ffn_output(update))

        return x + drop_residual(update, self.drop_path, self.training)


def make_transformer_stack(
    dim, depth, heads, mlp_ratio, dropout, attention_dropout, drop_path
):
    """Build one attention stage, including its final layer normalization."""
    blocks = [
        GatedTransformerBlock(
            dim, heads, mlp_ratio, dropout, attention_dropout, drop_path
        )
        for _ in range(depth)
    ]
    stack = nn.Sequential(*blocks, nn.LayerNorm(dim))
    stack.apply(initialize_transformer)
    return stack


class PredFormerForecaster(nn.Module):
    """Forecast future frames using a shared temporal-first transformer."""

    def __init__(
        self,
        image_size=(372, 614),
        channels=11,
        input_frames=5,
        forecast_frames=12,
        patch_size=8,
        dim=128,
        heads=4,
        depth=2,
        mlp_ratio=2,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path=0.25,
        padding_value=-1.0,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.forecast_frames = forecast_frames
        self.patch_size = patch_size
        self.dim = dim
        self.padding_value = padding_value

        height, width = image_size
        self.grid_size = (
            (height + patch_size - 1) // patch_size,
            (width + patch_size - 1) // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        patch_dim = channels * patch_size * patch_size

        # Embed patches directly, preserving the native image pixels.
        self.patch_embedding = nn.Linear(patch_dim, dim)

        # Fixed positional encodings move with the model and are checkpointed.
        positions = sinusoidal_positions(
            input_frames * self.num_patches, dim
        ).reshape(1, input_frames, self.num_patches, dim)
        self.register_buffer("positions", positions)

        # Complete all temporal blocks before processing spatial relationships.
        self.temporal = make_transformer_stack(
            dim, depth, heads, mlp_ratio, dropout, attention_dropout, drop_path
        )
        self.spatial = make_transformer_stack(
            dim, depth, heads, mlp_ratio, dropout, attention_dropout, drop_path
        )

        # Recover absolute pixel values from each output token.
        self.readout = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

    def predict_block(self, x):
        """Map five frames to the next five frames in B,C,T,H,W layout."""
        batch, channels, frames, height, width = x.shape
        rows, columns = self.grid_size
        patch = self.patch_size

        # At native resolution, add four bottom rows and two right columns.
        # These padding values are removed before returning predictions.
        images = F.pad(
            x,
            (0, columns * patch - width, 0, rows * patch - height),
            value=self.padding_value,
        )

        # Flatten each patch in row, column, channel order.
        patches = images.reshape(
            batch, channels, frames, rows, patch, columns, patch
        )
        patches = patches.permute(0, 2, 3, 5, 4, 6, 1).reshape(
            batch, frames, self.num_patches, patch * patch * channels
        )
        tokens = self.patch_embedding(patches) + self.positions

        # Temporal attention processes each patch's five-frame history.
        tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch * self.num_patches, frames, self.dim
        )
        tokens = self.temporal(tokens)

        # Spatial attention exchanges information across all patches per frame.
        tokens = tokens.reshape(
            batch, self.num_patches, frames, self.dim
        ).permute(0, 2, 1, 3).reshape(
            batch * frames, self.num_patches, self.dim
        )
        tokens = self.spatial(tokens)

        # Restore images and remove only the padding added above.
        pixels = self.readout(tokens).reshape(
            batch, frames, rows, columns, channels, patch, patch
        )
        images = pixels.permute(0, 4, 1, 2, 5, 3, 6).reshape(
            batch, channels, frames, rows * patch, columns * patch
        )
        return images[..., :height, :width].contiguous()

    def forward(self, x):
        """Generate twelve forecasts by retaining five, five and two frames."""
        predictions = []
        current = x

        for start in range(0, self.forecast_frames, self.input_frames):
            # Reuse the same weights, preserving gradients through the rollout.
            current = self.predict_block(current)
            keep = min(self.input_frames, self.forecast_frames - start)
            predictions.append(current[:, :, :keep])

        return torch.cat(predictions, dim=2)