"""Component-routed local experts with analytic envelopes and RMS filter pooling."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from utils.numerics import clip_logits


def analytic_envelope_descriptor(x, topk):
    n = x.shape[-1]
    frequency_mask = x.new_zeros(n)
    frequency_mask[0] = 1
    frequency_mask[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        frequency_mask[n // 2] = 1
    envelope = torch.fft.ifft(
        torch.fft.fft(x, dim=-1) * frequency_mask, dim=-1
    ).abs()
    spectrum = torch.fft.rfft(envelope, dim=-1).abs().mean(dim=1)
    peaks = spectrum.topk(topk, dim=-1).values
    return torch.cat((envelope.mean(dim=-1), peaks), dim=-1).clamp(-1e3, 1e3)


def rms_filter_response(x, filter_bank):
    batch, channels, _ = x.shape
    kernels = filter_bank.scales.shape[0]
    kernel_width = filter_bank.scales.shape[-1]
    weights = filter_bank.scales.repeat(channels, 1, 1)
    response = F.conv1d(
        F.pad(x, (kernel_width // 2, kernel_width // 2)),
        weights,
        groups=channels,
    ).reshape(batch, channels, kernels, -1)
    return (response.square().mean(dim=-1) + 1e-12).sqrt()


class RevisedRmsMoE(nn.Module):
    """Retain the original expert parameters and add fixed descriptor statistics."""

    def __init__(self, routed_moe):
        super().__init__()
        if routed_moe.routing_variant != "correct":
            raise ValueError("Revised local mode requires fixed component routing")
        self.experts = copy.deepcopy(routed_moe.experts)
        self.expert_channels = [list(group) for group in routed_moe.expert_channels]
        self.label_dims = list(routed_moe.label_dims)
        self.routing_variant = "correct"
        self.logit_clip = routed_moe.logit_clip
        self.routing_manifest_hash = ""
        for index, expert in enumerate(self.experts):
            dimension = 2 * expert.c_in + expert.hilbert.topk
            self.register_buffer(f"mean_{index}", torch.zeros(dimension))
            self.register_buffer(f"std_{index}", torch.ones(dimension))

    def local_parameter_report(self):
        count = sum(parameter.numel() for parameter in self.experts.parameters())
        return {
            "routing_variant": self.routing_variant,
            "reference_local_params": count,
            "local_params": count,
            "relative_error": 0.0,
            "unified_hidden": None,
        }

    def descriptors(self, x):
        output = []
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            for index, expert in enumerate(self.experts):
                part = x[:, self.expert_channels[index], :].float()
                envelope = analytic_envelope_descriptor(part, expert.hilbert.topk)
                pooled = rms_filter_response(part, expert.wavelet)
                temporal = expert.wavelet.proj(pooled).mean(dim=-1).clamp(-1e3, 1e3)
                output.append(torch.cat((envelope, temporal), dim=-1))
        return output

    @torch.no_grad()
    def fit_standardization(self, normalized_batches):
        """Fit once from normalized training windows; never pass validation/test data."""
        self.eval()
        parts = [[] for _ in self.experts]
        for batch in normalized_batches:
            for index, descriptor in enumerate(self.descriptors(batch)):
                parts[index].append(descriptor.detach().cpu())
        if not parts or any(not chunk for chunk in parts):
            raise ValueError("Cannot fit descriptors on an empty training split")
        for index, chunks in enumerate(parts):
            values = torch.cat(chunks, dim=0)
            getattr(self, f"mean_{index}").copy_(values.mean(dim=0))
            getattr(self, f"std_{index}").copy_(
                values.std(dim=0, unbiased=False).clamp_min(1e-6)
            )

    def forward(self, x, return_details=False, intervention=None):
        if intervention is not None:
            raise ValueError("Intervention controls use the legacy local mode")
        logits_parts = []
        features = []
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            for index, (expert, descriptor) in enumerate(
                zip(self.experts, self.descriptors(x))
            ):
                standardized = (
                    descriptor - getattr(self, f"mean_{index}")
                ) / getattr(self, f"std_{index}")
                hidden = expert.pre_proj(standardized)
                representation = expert.mlp[1](expert.mlp[0](hidden))
                features.append(representation)
                logits_parts.append(clip_logits(expert.mlp[2](representation), self.logit_clip))
        logits = torch.cat(logits_parts, dim=-1)
        if return_details:
            return x, logits, {
                "component_representations": features,
                "component_logits": logits_parts,
                "routing_variant": self.routing_variant,
            }
        return x, logits
