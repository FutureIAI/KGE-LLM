# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""
import torch
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.05, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):

        target = target.to(pred.device)

        with torch.no_grad():
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing

        loss = self.bce(pred, target)  # [B, C]

        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)  # [C]
            loss = loss * weights

        return loss.mean()





class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma_base=2.0, gamma_scale=1.0, reduction='mean', debug=False):

        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_base = gamma_base
        self.gamma_scale = gamma_scale
        self.reduction = reduction
        self.debug = debug

    def forward(self, inputs, targets):

        inputs = torch.clamp(inputs, -10, 10)
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = torch.nan_to_num(BCE_loss, nan=0.0, posinf=1.0, neginf=0.0)

        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = torch.clamp(pt, 1e-8, 1 - 1e-8)


        with torch.no_grad():
            dynamic_gamma = self.gamma_base + self.gamma_scale * (1 - pt)

        focal_factor = (1 - pt) ** dynamic_gamma


        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(inputs.device)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0).expand_as(inputs)
        else:
            alpha = self.alpha

        loss = alpha * focal_factor * BCE_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

