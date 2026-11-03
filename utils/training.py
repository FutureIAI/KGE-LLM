"""Shared classification objectives and optimizer updates."""

import math

import torch


COMPONENT_SLICES = {"motor": (0, 5), "gear": (5, 14), "LA": (14, 19), "RA": (19, 21)}
COMPONENT_WEIGHTS = {"motor": 1.0, "gear": 1.0, "LA": 1.0, "RA": 0.5}


def validate_max_grad_norm(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("max_grad_norm must be finite and nonnegative")
    return value


def clip_gradients(parameters, max_grad_norm=0.0):
    """A zero limit leaves gradients unchanged."""
    limit = validate_max_grad_norm(max_grad_norm)
    if limit == 0:
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=limit)


def optimizer_step(optimizer, parameters, scaler=None, max_grad_norm=0.0):
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = clip_gradients(parameters, max_grad_norm)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return grad_norm


def classification_losses(
    outputs, global_logits, local_logits, targets, criterion, local_only=False
):
    def loss_for(name, logits, labels):
        selected = (
            criterion.get(name, criterion.get("full"))
            if isinstance(criterion, dict)
            else criterion
        )
        return selected(logits, labels)

    if local_only:
        local = loss_for("full", local_logits, targets)
        zero = local.new_zeros(())
        return {"total": local, "fused": local, "global": zero, "local": local}

    component = {
        name: loss_for(name, outputs[:, start:end], targets[:, start:end])
        for name, (start, end) in COMPONENT_SLICES.items()
    }
    fused = sum(
        COMPONENT_WEIGHTS[name] * value for name, value in component.items()
    ) / (sum(COMPONENT_WEIGHTS.values()) + 1e-12)
    global_loss = loss_for("full", global_logits, targets)
    local_loss = loss_for("full", local_logits, targets)
    return {
        "total": fused + 0.8 * global_loss + 0.2 * local_loss,
        "fused": fused,
        "global": global_loss,
        "local": local_loss,
    }
