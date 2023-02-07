import torch
import torch.nn as nn


class RestoreNetLoss():
    def __init__(self):
        self.l1 = nn.L1Loss()

    def forward(self, x, target):
        l = self.l1(x, target) +
            (x.max().log2() - target.max().log2()).abs()
        return l


class EnhanceNetLoss():
    def __init__(self):
        self.l1 = nn.L1Loss()

    def forward(self, x, target):
        return self.l1(x, target)
