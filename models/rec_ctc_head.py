"""
@File : rec_ctc_head.py
@Author : CodeCat
@Time : 2021/7/15 下午9:31
"""

import math

import torch
from torch import nn
from torch.nn import functional as F



class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, 2)
        return predicts