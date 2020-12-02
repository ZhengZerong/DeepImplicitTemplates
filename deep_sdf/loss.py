#!/usr/bin/env python3
# Copyright 2020-present Zerong Zheng. All Rights Reserved.

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LipschitzLoss(nn.Module):
    def __init__(self, k, reduction=None):
        super(LipschitzLoss, self).__init__()
        self.relu = nn.ReLU()
        self.k = k
        self.reduction = reduction

    def forward(self, x1, x2, y1, y2):
        l = self.relu(torch.norm(y1-y2, dim=-1) / (torch.norm(x1-x2, dim=-1)+1e-3) - self.k)
        l = torch.clamp(l, 0.0, 5.0)    # avoid
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class HuberFunc(nn.Module):
    def __init__(self, reduction=None):
        super(HuberFunc, self).__init__()
        self.reduction = reduction

    def forward(self, x, delta):
        n = torch.abs(x)
        cond = n < delta
        l = torch.where(cond, 0.5 * n ** 2, n*delta - 0.5 * delta**2)
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class SoftL1Loss(nn.Module):
    def __init__(self, reduction=None):
        super(SoftL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, eps=0.0, lamb=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        ret = ret * (1 + lamb * torch.sign(target) * torch.sign(target-input))
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(ret)
        else:
            return torch.sum(ret)
