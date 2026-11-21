"""The notebook's encoder-decoder ConvLSTM, without spatial resizing."""

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """Update hidden and cell states with spatial convolutional gates."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        """Create one convolution that produces the four recurrent gates."""
        super().__init__()
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, state):
        """Process one frame and return the updated (hidden, cell) states."""
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Encode observed frames and predict future frames autoregressively."""

    def __init__(
        self, input_channels=11, hidden_channels=16, kernel_size=3,
        history_steps=5, forecast_steps=12,
    ):
        """Create separate encoder/decoder cells and a channel projection."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.encoder = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.decoder = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.output_projection = nn.Conv2d(hidden_channels, input_channels, 1)

    def forward(self, X):
        """Map (B, C, history, H, W) inputs to (B, C, forecast, H, W)."""
        batch, _, _, height, width = X.shape
        # Each sequence starts with fresh recurrent states.
        h = X.new_zeros(batch, self.hidden_channels, height, width)
        c = torch.zeros_like(h)
        for step in range(self.history_steps):
            h, c = self.encoder(X[:, :, step], (h, c))

        # Forecast autoregressively; future targets never enter the decoder.
        previous = X[:, :, -1]
        predictions = []
        for _ in range(self.forecast_steps):
            h, c = self.decoder(previous, (h, c))
            previous = torch.sigmoid(self.output_projection(h))
            predictions.append(previous)
        return torch.stack(predictions, dim=2)
