"""Optional logit clipping, separate from probability-domain safeguards."""

import math


def validate_logit_clip(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Logit clipping limits must be finite and non-negative")
    return value


def clip_logits(logits, limit=0.0):
    return logits.clamp(-limit, limit) if limit > 0 else logits


def bound_probabilities(probabilities):
    # The upper bound must remain below one in the tensor's actual dtype.
    import torch

    epsilon = max(1e-8, torch.finfo(probabilities.dtype).eps)
    return probabilities.clamp(epsilon, 1.0 - epsilon)
