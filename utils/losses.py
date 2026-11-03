"""Classification losses for industrial multilabel diagnosis."""

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from utils.numerics import bound_probabilities, clip_logits, validate_logit_clip


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.05, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):

        target = target.to(pred.device)

        with torch.no_grad():
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing

        loss = self.bce(pred, target)  # [B, C]

        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)  # [C]
            loss = loss * weights

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(
        self, alpha=1.0, gamma=2.0, reduction="mean", debug=False, logit_clip=0.0
    ):
        super(FocalLoss, self).__init__()
        self.logit_clip = validate_logit_clip(logit_clip)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.debug = debug

    def forward(self, inputs, targets):

        inputs = clip_logits(inputs, self.logit_clip)

        probs = torch.sigmoid(inputs)
        probs = bound_probabilities(probs)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        BCE_loss = torch.nan_to_num(BCE_loss, nan=0.0, posinf=1.0, neginf=0.0)

        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = bound_probabilities(pt)
        focal_factor = (1 - pt) ** self.gamma

        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(inputs.device)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0).expand_as(inputs)
        else:
            alpha = self.alpha

        loss = alpha * focal_factor * BCE_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

        if self.debug:
            logging.getLogger(__name__).debug(
                "Focal loss mean: %.6g", loss.detach().mean().item()
            )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DynamicFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma_base=1.0,
        gamma_scale=2.0,
        reduction="mean",
        debug=False,
        detach_gamma=True,
        logit_clip=0.0,
    ):
        super(DynamicFocalLoss, self).__init__()
        self.logit_clip = validate_logit_clip(logit_clip)
        self.alpha = alpha  # Expecting Tensor [21]
        self.gamma_base = gamma_base
        self.gamma_scale = gamma_scale
        self.reduction = reduction
        self.debug = debug
        self.detach_gamma = detach_gamma

    def forward(self, inputs, targets):

        inputs = clip_logits(inputs, self.logit_clip)
        probs = torch.sigmoid(inputs)
        probs = bound_probabilities(probs)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        BCE_loss = torch.nan_to_num(BCE_loss, nan=0.0, posinf=1.0, neginf=0.0)

        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = bound_probabilities(pt)

        if self.detach_gamma:
            with torch.no_grad():
                dynamic_gamma = self.gamma_base + self.gamma_scale * (1 - pt)
        else:
            dynamic_gamma = self.gamma_base + self.gamma_scale * (1 - pt)

        focal_factor = (1 - pt) ** dynamic_gamma

        loss_tmp = focal_factor * BCE_loss

        if self.alpha is not None:

            if hasattr(self.alpha, "device") and self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            if (
                isinstance(self.alpha, torch.Tensor)
                and inputs.shape[1] == self.alpha.shape[0]
            ):

                alpha = self.alpha
                if alpha.ndim == 1:
                    alpha = alpha.unsqueeze(0).expand_as(inputs)
                loss_tmp = alpha * loss_tmp
            else:

                pass

        loss = torch.nan_to_num(loss_tmp, nan=0.0, posinf=1.0, neginf=0.0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PlainBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction="mean", logit_clip=0.0):
        super().__init__()
        self.logit_clip = validate_logit_clip(logit_clip)
        self.reduction = reduction
        self.register_buffer(
            "pos_weight",
            pos_weight.float() if isinstance(pos_weight, torch.Tensor) else None,
        )

    def forward(self, inputs, targets):
        pos_weight = (
            self.pos_weight.to(inputs.device) if self.pos_weight is not None else None
        )
        loss = F.binary_cross_entropy_with_logits(
            clip_logits(inputs, self.logit_clip),
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", logit_clip=0.0):
        super().__init__()
        self.logit_clip = validate_logit_clip(logit_clip)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = clip_logits(inputs, self.logit_clip)
        probs = bound_probabilities(torch.sigmoid(inputs))
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            alpha = (
                self.alpha.to(inputs.device)
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha
            )
            if isinstance(alpha, torch.Tensor) and inputs.shape[1] == alpha.shape[0]:
                loss = loss * alpha.unsqueeze(0).expand_as(inputs)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        eps=1e-8,
        reduction="mean",
        logit_clip=0.0,
    ):
        super().__init__()
        self.logit_clip = validate_logit_clip(logit_clip)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = clip_logits(inputs, self.logit_clip)
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
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
