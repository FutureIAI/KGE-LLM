from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrozenDecoderBackboneSpec:
    name: str
    repo_id: str
    revision: str
    model_path: str
    active_layers: int
    attention_implementation: str
    apply_final_norm: bool
    official_model_forward: bool


class FrozenDecoderBackboneAdapter:
    """Execution adapter for frozen decoder backbones.

    The LLaMA branch intentionally retains the historical manual-layer execution
    path for checkpoint-level backward compatibility. Mistral uses the official
    model forward so its causal/sliding-window mask and final RMSNorm are applied.
    """

    def __init__(self, spec: FrozenDecoderBackboneSpec):
        if spec.name not in {"llama", "mistral"}:
            raise ValueError(f"Unsupported frozen decoder backbone: {spec.name}")
        if spec.active_layers <= 0:
            raise ValueError("active_layers must be positive")
        self.spec = spec
        self.last_forward_report = None

    @staticmethod
    def layer_list(model):
        layers = getattr(model, "layers", None)
        if layers is None:
            raise RuntimeError(f"Decoder model {type(model).__name__} does not expose .layers")
        return layers

    @staticmethod
    def final_norm(model):
        return getattr(model, "norm", None)

    def run(self, model, hidden_states):
        if model is None:
            raise RuntimeError("Frozen decoder model is not loaded")
        if len(self.layer_list(model)) != self.spec.active_layers:
            raise RuntimeError(
                f"Active decoder layer mismatch: model={len(self.layer_list(model))} "
                f"expected={self.spec.active_layers}"
            )

        input_shape = list(hidden_states.shape)
        if self.spec.official_model_forward:
            output = model(
                inputs_embeds=hidden_states,
                attention_mask=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            final_norm_executed = True
        else:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            output = hidden_states
            for layer in self.layer_list(model):
                output = layer(
                    output,
                    attention_mask=None,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False,
                )[0]
            final_norm_executed = False
            if self.spec.apply_final_norm:
                norm = self.final_norm(model)
                if norm is None:
                    raise RuntimeError("Final decoder norm requested but not present")
                output = norm(output)
                final_norm_executed = True

        self.last_forward_report = {
            "backbone": self.spec.name,
            "input_shape": input_shape,
            "output_shape": list(output.shape),
            "active_decoder_layers": self.spec.active_layers,
            "active_layer_indices": list(range(self.spec.active_layers)),
            "attention_implementation": self.spec.attention_implementation,
            "official_model_forward": self.spec.official_model_forward,
            "final_norm_requested": self.spec.apply_final_norm,
            "final_norm_executed": final_norm_executed,
            "finite": bool(torch.isfinite(output).all().item()),
        }
        return output
