"""
@File : crnn.py
@Author : CodeCat
@Time : 2021/7/15 下午9:32
"""
from torch import nn

from .rec_mobilenet_v3 import MobileNetV3
from .sequence_encoder import SequenceEncoder
from .rec_ctc_head import CTCHead


class CRNN(nn.Module):
    def __init__(self, out_channels, type='large',
                 hidden_size=96,
                 ):
        super(CRNN, self).__init__()

        # CNN
        self.cnn = MobileNetV3(type=type)

        # RNN
        in_channels = self.cnn.out_channels
        self.rnn = SequenceEncoder(in_channels, hidden_size)

        # CTC Head
        in_channels = self.rnn.out_channels
        self.head = CTCHead(in_channels, out_channels)

    def forward(self, x, data=None):
        # x : [batch_size, 3, 32, 100] -> [batch_size, 960, 1, 25]
        x = self.cnn(x)
        # x : [batch_size, 960, 1, 25] -> [batch_size, 25, 96 * 2]
        x = self.rnn(x)
        # x : [batch_size, 25, 96 * 2] -> [batch_size, 25, 96]
        if data is None:
            x = self.head(x)
        else:
            x = self.head(x, data)
        return x