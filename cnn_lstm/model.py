"""CNN–LSTM for native-resolution Cloudcasting forecasts."""

import torch
from torch import nn


class CNNLSTMForecaster(nn.Module):
    """Encode observed satellite frames and decode twelve future frames."""

    def __init__(
        self,
        channels=11,
        spatial_size=(372, 614),
        embedding_dim=64,
        hidden_dim=128,
        forecast_steps=12,
    ):
        """Construct the same CNN–LSTM architecture used in the Mac experiment."""
        super().__init__()
        self.forecast_steps = forecast_steps

        # Three stride-two convolutions round spatial dimensions upward.
        # Native 372×614 inputs therefore produce 47×77 feature maps.
        self.latent_shape = (
            32,
            (spatial_size[0] + 7) // 8,
            (spatial_size[1] + 7) // 8,
        )
        flattened_size = (
            self.latent_shape[0]
            * self.latent_shape[1]
            * self.latent_shape[2]
        )

        # Shared spatial encoder for every observed frame.
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(channels, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.embedding = nn.Linear(flattened_size, embedding_dim)

        self.temporal_encoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.temporal_decoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )

        # Decoder inputs identify forecast leads; they contain no target data.
        self.lead_embeddings = nn.Parameter(
            torch.randn(forecast_steps, embedding_dim) * 0.02
        )

        self.expand = nn.Linear(hidden_dim, flattened_size)
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        """Map (B,C,5,H,W) to (B,C,12,H,W), with fresh recurrent state."""
        batch, channels, steps, height, width = inputs.shape

        # Encode each frame, preserving chronological order.
        frames = inputs.permute(0, 2, 1, 3, 4).reshape(
            batch * steps, channels, height, width
        )
        features = self.spatial_encoder(frames).flatten(1)
        embeddings = self.embedding(features).reshape(batch, steps, -1)

        _, history_state = self.temporal_encoder(embeddings)

        decoder_inputs = self.lead_embeddings[None].expand(
            batch, -1, -1
        ).contiguous()
        future_features, _ = self.temporal_decoder(
            decoder_inputs, history_state
        )

        spatial_features = self.expand(future_features).reshape(
            batch * self.forecast_steps, *self.latent_shape
        )
        predictions = self.spatial_decoder(spatial_features)

        # The decoder produces 376×616 at native resolution.
        # Trim extra bottom/right outputs to match the original target grid.
        predictions = predictions[..., :height, :width]

        return predictions.reshape(
            batch, self.forecast_steps, channels, height, width
        ).permute(0, 2, 1, 3, 4)