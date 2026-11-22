"""Compact SimVP-v1 for Cloudcasting: five observations to twelve forecasts.

Preserves the notebook's architecture and autoregressive rollout.
Native image dimensions are handled by cropping decoder rounding.
"""

import torch
from torch import nn

SOURCE_REPOSITORY = "https://github.com/A4Bio/SimVP"
SOURCE_COMMIT = "4dcd4451282e014573ca4346ced8223e342f7893"


def spatial_block(in_channels, out_channels, stride=1, upsample=False):
    """Build the original convolution–GroupNorm–LeakyReLU spatial block."""
    if upsample:
        convolution = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
    else:
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

    return nn.Sequential(
        convolution,
        nn.GroupNorm(2, out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class InceptionBlock(nn.Module):
    """Combine the four grouped-convolution branches used by SimVP-v1."""

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, middle_channels, 1)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    middle_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=8,
                ),
                nn.GroupNorm(8, out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for kernel in (3, 5, 7, 11)
        ])

    def forward(self, x):
        """Project the input and sum the multi-scale branch outputs."""
        x = self.projection(x)
        return sum(branch(x) for branch in self.branches)


class TemporalMixer(nn.Module):
    """Mix time and channels using an Inception encoder–decoder."""

    def __init__(self, input_channels, hidden_channels=64, depth=4):
        super().__init__()
        middle_channels = hidden_channels // 2

        self.encoder = nn.ModuleList([
            InceptionBlock(
                input_channels if index == 0 else hidden_channels,
                middle_channels,
                hidden_channels,
            )
            for index in range(depth)
        ])

        self.decoder = nn.ModuleList([
            InceptionBlock(
                hidden_channels if index == 0 else 2 * hidden_channels,
                middle_channels,
                input_channels if index == depth - 1 else hidden_channels,
            )
            for index in range(depth)
        ])

    def forward(self, x):
        """Process flattened temporal features with encoder skip connections."""
        skips = []

        for index, block in enumerate(self.encoder):
            x = block(x)
            if index < len(self.encoder) - 1:
                skips.append(x)

        x = self.decoder[0](x)
        for index in range(1, len(self.decoder)):
            x = self.decoder[index](
                torch.cat((x, skips[-index]), dim=1)
            )

        return x


class SimVPForecaster(nn.Module):
    """Forecast satellite images in Cloudcasting's BCTHW tensor layout."""

    def __init__(
        self,
        channels=11,
        input_frames=5,
        forecast_frames=12,
        spatial_channels=16,
        temporal_channels=64,
        temporal_depth=4,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.forecast_frames = forecast_frames
        self.spatial_channels = spatial_channels

        # Four spatial stages, matching the notebook's N_S=4.
        self.encoder = nn.ModuleList([
            spatial_block(channels, spatial_channels),
            spatial_block(spatial_channels, spatial_channels, stride=2),
            spatial_block(spatial_channels, spatial_channels),
            spatial_block(spatial_channels, spatial_channels, stride=2),
        ])

        self.temporal_mixer = TemporalMixer(
            input_channels=input_frames * spatial_channels,
            hidden_channels=temporal_channels,
            depth=temporal_depth,
        )

        self.decoder = nn.Sequential(
            spatial_block(
                spatial_channels, spatial_channels, upsample=True
            ),
            spatial_block(spatial_channels, spatial_channels),
            spatial_block(
                spatial_channels, spatial_channels, upsample=True
            ),
        )
        self.skip_fusion = spatial_block(
            2 * spatial_channels, spatial_channels
        )
        self.readout = nn.Conv2d(spatial_channels, channels, 1)

    def predict_block(self, frames):
        """Map five BTCHW frames to the next five frames."""
        batch, steps, channels, height, width = frames.shape
        hidden_channels = self.spatial_channels

        # Apply the same spatial encoder independently to each input frame.
        images = frames.reshape(
            batch * steps, channels, height, width
        )
        skip = self.encoder[0](images)
        features = skip

        for block in self.encoder[1:]:
            features = block(features)

        latent_height, latent_width = features.shape[-2:]

        # Temporal mixing treats the five latent frames as channel groups.
        features = features.reshape(
            batch,
            steps * hidden_channels,
            latent_height,
            latent_width,
        )
        features = self.temporal_mixer(features)
        features = features.reshape(
            batch * steps,
            hidden_channels,
            latent_height,
            latent_width,
        )

        decoded = self.decoder(features)

        # For native images, decoder width is 616 rather than 614.
        # Crop the rounding-related border before joining the spatial skip.
        decoded = decoded[..., :height, :width]
        decoded = self.skip_fusion(
            torch.cat((decoded, skip), dim=1)
        )
        predictions = self.readout(decoded)

        return predictions.reshape(
            batch, steps, channels, height, width
        )

    def forward(self, inputs):
        """Return twelve forecasts, retaining gradients through all blocks."""
        frames = inputs.permute(0, 2, 1, 3, 4).contiguous()
        predicted_blocks = []

        block_count = (
            self.forecast_frames + self.input_frames - 1
        ) // self.input_frames

        # Three five-frame calls produce fifteen frames; retain the first twelve.
        # Feedback remains connected for backpropagation through the rollout.
        for _ in range(block_count):
            frames = self.predict_block(frames)
            predicted_blocks.append(frames)

        predictions = torch.cat(predicted_blocks, dim=1)
        predictions = predictions[:, :self.forecast_frames]

        # Linear outputs, without clipping or additional normalization.
        return predictions.permute(0, 2, 1, 3, 4).contiguous()