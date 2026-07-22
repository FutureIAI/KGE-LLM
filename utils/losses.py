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
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 用于后面加权平均

    def forward(self, pred, target):
        # 保证 target 和 weights 都在 pred 的设备上
        target = target.to(pred.device)

        # 平滑标签
        with torch.no_grad():
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing

        loss = self.bce(pred, target)  # [B, C]

        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)  # [C]
            loss = loss * weights  # 广播到 [B, C]

        return loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', debug=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.debug = debug

    def forward(self, inputs, targets):

        # 数值裁剪防止极端输入
        inputs = torch.clamp(inputs, -10, 10)   # sig(20)=1-2e-9 已足够稳定

        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = torch.nan_to_num(BCE_loss, nan=0.0, posinf=1.0, neginf=0.0)

        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = torch.clamp(pt, 1e-8, 1 - 1e-8)
        focal_factor = (1 - pt) ** self.gamma

        # 修复设备与广播问题
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(inputs.device)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0).expand_as(inputs)
        else:
            alpha = self.alpha

        loss = alpha * focal_factor * BCE_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

        if self.debug:
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("⚠️ [DEBUG] FocalLoss detected NaN or Inf:")
                print(f"  inputs range: {inputs.min().item():.4f} ~ {inputs.max().item():.4f}")
                print(f"  probs range: {probs.min().item():.4f} ~ {probs.max().item():.4f}")
                print(f"  BCE_loss mean/std: {BCE_loss.mean().item():.6f} / {BCE_loss.std().item():.6f}")
                print(f"  focal_factor mean/std: {focal_factor.mean().item():.6f} / {focal_factor.std().item():.6f}")
                print(f"  final loss mean/std: {loss.mean().item():.6f} / {loss.std().item():.6f}")
        if self.debug and loss.mean().item() < 1e-7:
            print("⚠️ [DEBUG] Loss extremely small!")
            print("  inputs mean/std:", inputs.mean().item(), inputs.std().item())
            print("  probs mean/std:", probs.mean().item(), probs.std().item())
            print("  BCE mean/std:", BCE_loss.mean().item(), BCE_loss.std().item())
            print("  focal_factor mean/std:", focal_factor.mean().item(), focal_factor.std().item())

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma_base=1.0, gamma_scale=2.0, reduction='mean', debug=False, detach_gamma=True):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha  # Expecting Tensor [21]
        self.gamma_base = gamma_base
        self.gamma_scale = gamma_scale
        self.reduction = reduction
        self.debug = debug
        self.detach_gamma = detach_gamma

    def forward(self, inputs, targets):
        # --- 1. 基础处理 (保持不变) ---
        inputs = torch.clamp(inputs, -10, 10)
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = torch.nan_to_num(BCE_loss, nan=0.0, posinf=1.0, neginf=0.0)

        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = torch.clamp(pt, 1e-8, 1 - 1e-8)

        if self.detach_gamma:
            with torch.no_grad():
                dynamic_gamma = self.gamma_base + self.gamma_scale * (1 - pt)
        else:
            dynamic_gamma = self.gamma_base + self.gamma_scale * (1 - pt)

        focal_factor = (1 - pt) ** dynamic_gamma

        # --- 2. 关键修改：智能 Alpha 匹配 ---
        loss_tmp = focal_factor * BCE_loss  # 先计算不带 Alpha 的 Loss

        if self.alpha is not None:
            # 确保 alpha 在正确的设备上
            if hasattr(self.alpha, 'device') and self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # 🚨 形状检查 (Shape Check) 🚨
            # inputs.shape[1] 是当前输入的类别数 (例如 21 或 5)
            # self.alpha.shape[0] 是预设的类别权重数 (21)
            if isinstance(self.alpha, torch.Tensor) and inputs.shape[1] == self.alpha.shape[0]:
                # 【情况A：训练或全量验证】维度匹配 (21 vs 21)，应用类别权重
                alpha = self.alpha
                if alpha.ndim == 1:
                    alpha = alpha.unsqueeze(0).expand_as(inputs)
                loss_tmp = alpha * loss_tmp
            else:
                # 【情况B：验证集子任务】维度不匹配 (5 vs 21)，跳过权重
                # 这种情况下我们不加权，仅计算 Loss 以便程序继续运行
                # 这不会影响训练效果，因为训练总是走情况 A
                pass

        loss = torch.nan_to_num(loss_tmp, nan=0.0, posinf=1.0, neginf=0.0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PlainBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight.float() if isinstance(pos_weight, torch.Tensor) else None)

    def forward(self, inputs, targets):
        pos_weight = self.pos_weight.to(inputs.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(
            torch.clamp(inputs, -10, 10),
            targets,
            pos_weight=pos_weight,
            reduction='none',
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, -10, 10)
        probs = torch.sigmoid(inputs).clamp(1e-8, 1 - 1e-8)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device) if isinstance(self.alpha, torch.Tensor) else self.alpha
            if isinstance(alpha, torch.Tensor) and inputs.shape[1] == alpha.shape[0]:
                loss = loss * alpha.unsqueeze(0).expand_as(inputs)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, -10, 10)
        xs_pos = torch.sigmoid(inputs)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * ((1 - pt).clamp(min=self.eps) ** gamma)
        loss = -loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
