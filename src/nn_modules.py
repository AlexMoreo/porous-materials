import torch
from torch import nn
from torch.nn import functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2, bidirectional=True):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size   , output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class FFModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation=nn.ReLU, smooth_length=5):
        """
        Parameters:
        - input_size (int): Number of input features.
        - hidden_sizes (list of int): List containing the number of units in each hidden layer.
        - output_size (int): Number of output units.
        - activation (nn.Module): Activation function to use (default: ReLU).
        """
        super(FFModel, self).__init__()

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

        self.smooth_length=smooth_length

        if smooth_length>0:
            self.smooth_layer = nn.Conv1d(1, 1, kernel_size=smooth_length, padding='same', bias=False)
            self.smooth_layer.weight.data.fill_(1 / smooth_length)

    def forward(self, x):
        output = self.network(x)
        # smoothing
        if self.smooth_length > 0:
            output = output.unsqueeze(1)
            output = self.smooth_layer(output)
            output = output.squeeze(1)
        return output


class MonotonicNN(FFModel):
    def __init__(self, input_size, output_size, hidden_sizes, activation=nn.ReLU, smooth_length=5):
        super(MonotonicNN, self).__init__(input_size, output_size, hidden_sizes, activation, smooth_length)

    def forward(self, x):
        increments = self.network(x)  # Predicts increments only
        increments = F.relu(increments)
        cumulative = torch.cumsum(increments, dim=1)  # guarantees monotonicity
        return cumulative


import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create standard sinusoidal positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len,
                 d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Project scalar input at each feature position to model dimension
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding to provide order information
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool the sequence and map to the desired output dimension
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, n_features)
        # Treat features as a sequence of scalars
        x = x.unsqueeze(-1)                 # (batch, seq_len, 1)
        x = self.input_proj(x)              # (batch, seq_len, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)     # (batch, seq_len, d_model)

        # Aggregate information across feature positions
        x = x.mean(dim=1)                   # (batch, d_model)
        y = self.output_proj(x)             # (batch, output_dim)
        return y

