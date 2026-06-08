import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, lstm_output):

        weights = self.attention(
            lstm_output
        )

        weights = F.softmax(
            weights,
            dim=1
        )

        context = torch.sum(
            weights * lstm_output,
            dim=1
        )

        return context


class BiLSTMAttention(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = Attention(
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        lstm_output, _ = self.bilstm(x)

        context = self.attention(
            lstm_output
        )

        out = self.fc(context)

        return out