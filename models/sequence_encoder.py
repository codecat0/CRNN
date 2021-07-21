"""
@File : sequence_encoder.py
@Author : CodeCat
@Time : 2021/7/15 下午9:28
"""
from torch import nn


class Im2Seq(nn.Module):
    def __init__(self, in_channels):
        super(Im2Seq, self).__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(dim=2)
        # [B, C, 1, W] -> [B, W, C]
        x = x.permute(0, 2, 1)
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size=48):
        super(SequenceEncoder, self).__init__()

        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels

        self.encoder = EncoderWithRNN(self.out_channels, hidden_size)
        self.out_channels = self.encoder.out_channels


    def forward(self, x):
        x = self.encoder_reshape(x)
        x = self.encoder(x)
        return x