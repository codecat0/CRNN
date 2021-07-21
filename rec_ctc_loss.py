"""
@File : rec_ctc_loss.py
@Author : CodeCat
@Time : 2021/7/15 下午9:49
"""
import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self,):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def __call__(self, predicts, batch):
        # [B, W, C] -> [W, B, C]
        predicts = predicts.permute(1, 0, 2).log_softmax(2)
        N, B, _ = predicts.shape
        preds_lengths = torch.Tensor([N] * B).to(torch.int32)
        labels = batch[1].to(torch.int32)
        label_lengths = batch[2].to(torch.int32)
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        loss = loss.mean()
        return {'loss': loss}