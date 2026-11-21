"""Residual ConvLSTM architectures for full-resolution Cloudcasting training."""

import torch
from torch import nn


# Preserve the architectures used in the exploratory notebooks.
MODEL_CONFIGS = {
    "residual": {
        "input_channels": 11,
        "hidden_channels": [16],
        "kernel_size": 3,
        "forecast_steps": 12,
    },
    "stacked_residual": {
        "input_channels": 11,
        "hidden_channels": [32, 16],
        "kernel_size": 3,
        "forecast_steps": 12,
    },
}


class ConvLSTMCell(nn.Module):
    """Update spatial hidden and cell states using convolutional LSTM gates."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        """Construct the four gates while preserving spatial dimensions."""
        super().__init__()

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, inputs, state):
        """Process one frame and return the updated recurrent states."""
        hidden, cell = state
        combined = torch.cat([inputs, hidden], dim=1)

        input_gate, forget_gate, candidate, output_gate = (
            self.gates(combined).chunk(4, dim=1)
        )

        cell = (
            torch.sigmoid(forget_gate) * cell
            + torch.sigmoid(input_gate) * torch.tanh(candidate)
        )
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)

        return hidden, cell


class ResidualConvLSTMForecaster(nn.Module):
    """Forecast learned corrections using one or more recurrent layers."""

    def __init__(
        self,
        input_channels=11,
        hidden_channels=(16,),
        kernel_size=3,
        forecast_steps=12,
    ):
        """Build separate encoder and decoder stacks with a linear output head."""
        super().__init__()

        self.hidden_channels = tuple(hidden_channels)
        self.forecast_steps = forecast_steps
        layer_inputs = (input_channels,) + self.hidden_channels[:-1]

        # The encoder processes observations; the decoder generates forecasts.
        self.encoder = nn.ModuleList([
            ConvLSTMCell(in_channels, out_channels, kernel_size)
            for in_channels, out_channels in zip(
                layer_inputs, self.hidden_channels
            )
        ])

        self.decoder = nn.ModuleList([
            ConvLSTMCell(in_channels, out_channels, kernel_size)
            for in_channels, out_channels in zip(
                layer_inputs, self.hidden_channels
            )
        ])

        self.output_projection = nn.Conv2d(
            self.hidden_channels[-1],
            input_channels,
            kernel_size=1,
        )

        # Initially, zero corrections reproduce the last preprocessed input.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs):
        """Map (B,C,5,H,W) to (B,C,12,H,W) at the input spatial resolution."""
        batch, _, history_steps, height, width = inputs.shape

        # Reset recurrent states for each independent forecast window.
        states = [
            (
                inputs.new_zeros((batch, channels, height, width)),
                inputs.new_zeros((batch, channels, height, width)),
            )
            for channels in self.hidden_channels
        ]

        # Encode the five observed frames in chronological order.
        for step in range(history_steps):
            layer_input = inputs[:, :, step]

            for layer, recurrent_cell in enumerate(self.encoder):
                states[layer] = recurrent_cell(layer_input, states[layer])
                layer_input = states[layer][0]

        # Transfer each encoder layer's states to its corresponding decoder.
        previous_frame = inputs[:, :, -1]
        predictions = []

        for _ in range(self.forecast_steps):
            layer_input = previous_frame

            for layer, recurrent_cell in enumerate(self.decoder):
                states[layer] = recurrent_cell(layer_input, states[layer])
                layer_input = states[layer][0]

            # Feed back the forecast, retaining gradients across all leads.
            correction = self.output_projection(layer_input)
            previous_frame = previous_frame + correction
            predictions.append(previous_frame)

        return torch.stack(predictions, dim=2)


def build_model(model_name):
    """Construct the selected architecture with freshly initialized weights."""
    return ResidualConvLSTMForecaster(**MODEL_CONFIGS[model_name])