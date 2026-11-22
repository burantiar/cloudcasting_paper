"""Standalone Earthformer-style adaptation for full-resolution cloudcasting.

Input:  (batch, 11, 5, 372, 614)
Output: (batch, 11, 12, 372, 614)

Matches notebook 07: two feature scales, axial attention, global context,
and simultaneous prediction of all twelve forecast leads.
"""

import torch
from torch import nn
import torch.nn.functional as F


MODEL_CONFIG = {
    "image_size": (372, 614),
    "channels": 11,
    "input_frames": 5,
    "forecast_frames": 12,
    "base_channels": 32,
    "heads": 4,
    "global_vectors": 8,
    "dropout": 0.1,
}


def make_ffn(dim, dropout):
    """Build the feed-forward network following each attention operation."""
    return nn.Sequential(
        nn.Linear(dim, 4 * dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * dim, dim),
        nn.Dropout(dropout),
    )


class LearnedPosition(nn.Module):
    """Add separate learned embeddings for time, rows and columns."""

    def __init__(self, frames, height, width, dim):
        super().__init__()
        self.time = nn.Parameter(torch.empty(1, frames, 1, 1, dim))
        self.row = nn.Parameter(torch.empty(1, 1, height, 1, dim))
        self.column = nn.Parameter(torch.empty(1, 1, 1, width, dim))

        for parameter in self.parameters():
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, x):
        """Add positions to features arranged as B,T,H,W,C."""
        return x + self.time + self.row + self.column


class AxialAttention(nn.Module):
    """Attend along one axis, optionally exchanging information globally."""

    def __init__(self, dim, heads, axis, dropout, use_global):
        super().__init__()
        self.axis = axis
        self.use_global = use_global

        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = make_ffn(dim, dropout)

        if use_global:
            self.global_norm = nn.LayerNorm(dim)
            self.global_attention = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )

    def forward(self, x, global_vectors=None):
        """Update attention lines and, when enabled, the global vectors."""
        batch, _, _, _, dim = x.shape
        normalized = self.norm(x)

        # Form independent attention lines along time, height or width.
        arranged = normalized.movedim(self.axis, -2)
        lines = arranged.reshape(batch, -1, x.shape[self.axis], dim)
        context = lines

        if self.use_global:
            normalized_global = self.global_norm(global_vectors)

            # Each line can also read the shared global context.
            expanded_global = normalized_global[:, None].expand(
                batch, lines.shape[1], -1, -1
            )
            context = torch.cat((lines, expanded_global), dim=2)

            # Global vectors gather information from the whole feature grid.
            global_context = torch.cat(
                (normalized_global, normalized.reshape(batch, -1, dim)),
                dim=1,
            )
            global_update, _ = self.global_attention(
                normalized_global,
                global_context,
                global_context,
                need_weights=False,
            )

        update, _ = self.attention(
            lines.flatten(0, 1),
            context.flatten(0, 1),
            context.flatten(0, 1),
            need_weights=False,
        )

        # Restore B,T,H,W,C before the residual updates.
        update = update.reshape(arranged.shape).movedim(-2, self.axis)
        x = x + self.output_dropout(update)
        x = x + self.ffn(self.ffn_norm(x))

        if self.use_global:
            global_vectors = (
                global_vectors + self.output_dropout(global_update)
            )

        return x, global_vectors


class AxialBlock(nn.Module):
    """Apply temporal, height and width attention in sequence."""

    def __init__(self, dim, heads, dropout, use_global=False):
        super().__init__()
        self.layers = nn.ModuleList([
            AxialAttention(dim, heads, axis, dropout, use_global)
            for axis in (1, 2, 3)
        ])

    def forward(self, x, global_vectors=None):
        """Process all three axes while preserving the feature-grid shape."""
        for layer in self.layers:
            x, global_vectors = layer(x, global_vectors)
        return x, global_vectors


class TemporalCrossAttention(nn.Module):
    """Connect forecast queries to observed history and global context."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = make_ffn(dim, dropout)

    def forward(self, x, memory, global_vectors):
        """Attend to encoded observations at each spatial location."""
        batch, frames, height, width, dim = x.shape
        history = memory.shape[1]

        queries = self.query_norm(x).permute(0, 2, 3, 1, 4)
        queries = queries.reshape(batch * height * width, frames, dim)

        observed = self.memory_norm(memory).permute(0, 2, 3, 1, 4)
        observed = observed.reshape(batch, height * width, history, dim)

        # Every spatial location receives the same global context vectors.
        global_context = self.global_norm(global_vectors)[:, None].expand(
            batch, height * width, -1, -1
        )
        context = torch.cat((observed, global_context), dim=2)
        context = context.reshape(batch * height * width, -1, dim)

        update, _ = self.attention(
            queries, context, context, need_weights=False
        )
        update = update.reshape(batch, height, width, frames, dim)
        update = update.permute(0, 3, 1, 2, 4)

        x = x + self.output_dropout(update)
        return x + self.ffn(self.ffn_norm(x))


class PatchMerge(nn.Module):
    """Merge neighbouring 2×2 features into a coarser spatial grid."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.projection = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x):
        """Reduce spatial dimensions while preserving all time steps."""
        height, width = x.shape[2:4]

        # Pad the bottom/right edges when a latent dimension is odd.
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        patches = torch.cat(
            [
                x[:, :, 0::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 1::2, 1::2],
            ],
            dim=-1,
        )
        return self.projection(self.norm(patches))


class SpatialUpsample(nn.Module):
    """Upsample each frame and refine its features with a convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, size):
        """Restore an explicit spatial size without mixing forecast leads."""
        batch, frames, height, width, channels = x.shape
        images = x.permute(0, 1, 4, 2, 3).reshape(
            batch * frames, channels, height, width
        )
        images = F.interpolate(images, size=size, mode="nearest")
        images = self.refine(images)

        return images.reshape(
            batch, frames, self.out_channels, *size
        ).permute(0, 1, 3, 4, 2)


class EarthformerForecaster(nn.Module):
    """Forecast twelve frames from five observations at two feature scales."""

    def __init__(
        self,
        image_size=(372, 614),
        channels=11,
        input_frames=5,
        forecast_frames=12,
        base_channels=32,
        heads=4,
        global_vectors=8,
        dropout=0.1,
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.forecast_frames = forecast_frames

        height, width = self.image_size
        self.half_size = ((height + 1) // 2, (width + 1) // 2)
        self.fine_size = ((height + 3) // 4, (width + 3) // 4)
        self.coarse_size = tuple((size + 1) // 2 for size in self.fine_size)

        fine_dim = base_channels
        coarse_dim = 2 * base_channels

        # Extract spatial features independently from each observed frame.
        self.stem = nn.Sequential(
            nn.Conv2d(channels, fine_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(fine_dim, fine_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.encoder_position = LearnedPosition(
            input_frames, *self.fine_size, fine_dim
        )

        self.global_vectors = nn.Parameter(
            torch.empty(1, global_vectors, fine_dim)
        )
        nn.init.trunc_normal_(self.global_vectors, std=0.02)

        self.encoder_fine = AxialBlock(
            fine_dim, heads, dropout, use_global=True
        )
        self.patch_merge = PatchMerge(fine_dim)
        self.global_projection = nn.Linear(fine_dim, coarse_dim)
        self.encoder_coarse = AxialBlock(
            coarse_dim, heads, dropout, use_global=True
        )

        # Forecast queries begin with learned positions and zero features.
        self.decoder_position = LearnedPosition(
            forecast_frames, *self.coarse_size, coarse_dim
        )
        self.cross_coarse = TemporalCrossAttention(
            coarse_dim, heads, dropout
        )

        # Decode coarse-to-fine, accessing observations at both scales.
        self.up_to_fine = SpatialUpsample(coarse_dim, fine_dim)
        self.decoder_fine = AxialBlock(fine_dim, heads, dropout)
        self.cross_fine = TemporalCrossAttention(fine_dim, heads, dropout)

        self.up_to_half = SpatialUpsample(fine_dim, fine_dim)
        self.up_to_full = SpatialUpsample(fine_dim, fine_dim)
        self.readout = nn.Linear(fine_dim, channels)

    def forward(self, x):
        """Map B,C,5,H,W observations to B,C,12,H,W predictions."""
        batch, channels, history, height, width = x.shape

        images = x.permute(0, 2, 1, 3, 4).reshape(
            batch * history, channels, height, width
        )
        features = self.stem(images)
        features = features.reshape(
            batch, history, features.shape[1], *self.fine_size
        ).permute(0, 1, 3, 4, 2)

        features = self.encoder_position(features)
        global_initial = self.global_vectors.expand(batch, -1, -1)

        memory_fine, global_fine = self.encoder_fine(
            features, global_initial
        )
        memory_coarse, global_coarse = self.encoder_coarse(
            self.patch_merge(memory_fine),
            self.global_projection(global_fine),
        )

        # Generate all twelve leads together using observed frames only.
        queries = memory_coarse.new_zeros(
            batch,
            self.forecast_frames,
            *self.coarse_size,
            memory_coarse.shape[-1],
        )
        queries = self.decoder_position(queries)

        # Match the notebook: first coarse stage uses cross-attention directly.
        queries = self.cross_coarse(
            queries, memory_coarse, global_coarse
        )
        queries = self.up_to_fine(queries, self.fine_size)
        queries, _ = self.decoder_fine(queries)
        queries = self.cross_fine(queries, memory_fine, global_fine)

        # Explicit target sizes restore the original 372×614 grid exactly.
        queries = self.up_to_half(queries, self.half_size)
        queries = self.up_to_full(queries, self.image_size)

        # Linear output matches the notebook's absolute-frame predictions.
        return self.readout(queries).permute(0, 4, 1, 2, 3).contiguous()