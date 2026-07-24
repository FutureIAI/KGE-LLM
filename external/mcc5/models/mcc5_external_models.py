"""KG-LLMoE and compact-Transformer models for MCC5-THU evaluation."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.moe_ffn import PhysicsExpert


class MeasurementLocalBranch(nn.Module):
    """Two independent, architecture-matched measurement-location experts."""

    def __init__(self, mode: str = "measurement_correct", hidden: int = 128):
        super().__init__()
        self.mode = mode
        if mode == "measurement_correct":
            self.motor = PhysicsExpert(3, None, hidden=hidden)
            self.gearbox = PhysicsExpert(3, None, hidden=hidden)
            self.project = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        elif mode == "unified":
            self.unified = PhysicsExpert(6, None, hidden=2 * hidden)
            self.project = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        else:
            raise ValueError(f"Unsupported local mode: {mode}")

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = signal.transpose(1, 2)
        if self.mode == "measurement_correct":
            _, _, motor = self.motor(x[:, :3], return_features=True)
            _, _, gearbox = self.gearbox(x[:, 3:], return_features=True)
            local = torch.cat([motor, gearbox], dim=-1)
        else:
            _, _, local = self.unified(x, return_features=True)
        return self.project(local)


class ContextEncoder(nn.Module):
    def __init__(self, enabled: bool, output_dim: int = 32):
        super().__init__()
        self.enabled = bool(enabled)
        self.output_dim = output_dim
        self.net = nn.Sequential(nn.Linear(4, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, output_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            if context.ndim != 2 or context.shape[-1] != 4:
                raise ValueError(f"Expected context [B,4], got {tuple(context.shape)}")
            return self.net(context)
        return torch.zeros(context.shape[0], self.output_dim, device=context.device, dtype=context.dtype)


class CompactTransformerGlobal(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.patch = nn.Conv1d(6, hidden, kernel_size=128, stride=128)
        block = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=512, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.context_token = nn.Linear(32, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, signal: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(signal.transpose(1, 2)).transpose(1, 2)
        tokens = torch.cat([self.context_token(context_feature).unsqueeze(1), tokens], dim=1)
        return self.norm(self.encoder(tokens).mean(dim=1))


class FrozenLlamaGlobal(nn.Module):
    """Frozen pretrained LLaMA with the KG-LLMoE reprogramming interface."""

    def __init__(self, llama_path: str, active_layers: int = 8, dropout: float = 0.1):
        super().__init__()
        from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
        from models.KGLLMoE import ReprogrammingLayer

        path = Path(llama_path)
        if not path.exists():
            raise FileNotFoundError(f"LLaMA path does not exist: {path}")
        config = LlamaConfig.from_pretrained(path, local_files_only=True)
        config.num_hidden_layers = int(active_layers)
        config.output_attentions = False
        config.output_hidden_states = False
        self.frozen_llama = LlamaModel.from_pretrained(
            path, config=config, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16,
        )
        for parameter in self.frozen_llama.parameters():
            parameter.requires_grad = False
        self.tokenizer = LlamaTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.patch = nn.Conv1d(6, 32, kernel_size=128, stride=128)
        vocab_size = self.frozen_llama.get_input_embeddings().weight.shape[0]
        self.mapping_layer = nn.Linear(vocab_size, 1000)
        self.reprogramming = ReprogrammingLayer(32, 8, 128, config.hidden_size)
        self.context_token = nn.Linear(32, config.hidden_size)
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, 128), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.prompt_text = (
            "Six-channel vibration monitoring from motor-side and gearbox-side measurement locations. "
            "Summarize global temporal and cross-location operating patterns without using class labels."
        )

    @property
    def pretrained_parameter_checksum(self) -> tuple[float, float]:
        parameter = next(self.frozen_llama.parameters()).detach().float()
        return float(parameter.sum().cpu()), float(parameter.square().sum().cpu())

    def forward(self, signal: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        batch = signal.shape[0]
        patch_tokens = self.patch(signal.transpose(1, 2)).transpose(1, 2)
        word_embeddings = self.frozen_llama.get_input_embeddings().weight
        source = self.mapping_layer(word_embeddings.transpose(0, 1)).transpose(0, 1)
        patch_tokens = self.reprogramming(patch_tokens, source, source)
        prompt_ids = self.tokenizer(
            [self.prompt_text] * batch, return_tensors="pt", padding=True, truncation=True, max_length=64,
        ).input_ids.to(signal.device)
        prompt = self.frozen_llama.get_input_embeddings()(prompt_ids)
        context_token = self.context_token(context_feature).unsqueeze(1)
        hidden = torch.cat([prompt, context_token, patch_tokens], dim=1)
        position_ids = torch.arange(hidden.shape[1], dtype=torch.long, device=hidden.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        for layer in self.frozen_llama.layers:
            hidden = layer(
                hidden, attention_mask=None, position_ids=position_ids,
                output_attentions=False, use_cache=False,
            )[0]
        pooled = hidden[:, -patch_tokens.shape[1]:].mean(dim=1)
        return self.output(self.dropout(pooled.float()))


class LocalGlobalMCC5(nn.Module):
    def __init__(
        self,
        backbone: str,
        local_mode: str,
        context_on: bool,
        llama_path: str,
        active_layers: int = 8,
        global_fusion_weight: float = 0.8,
    ):
        super().__init__()
        if not 0.0 <= global_fusion_weight <= 1.0:
            raise ValueError("global_fusion_weight must be in [0, 1]")
        self.model_family = backbone
        self.global_fusion_weight = float(global_fusion_weight)
        self.local = MeasurementLocalBranch(local_mode, hidden=128)
        self.context = ContextEncoder(context_on, output_dim=32)
        if backbone == "m2":
            self.global_encoder = CompactTransformerGlobal(hidden=128, layers=4)
        elif backbone == "m0":
            self.global_encoder = FrozenLlamaGlobal(llama_path, active_layers=active_layers)
        else:
            raise ValueError(backbone)
        self.local_head = nn.Linear(128, 10)
        self.global_head = nn.Linear(128, 10)
        self.system_head = nn.Sequential(nn.Linear(128 + 128 + 32, 128), nn.GELU(), nn.Linear(128, 20))

    def forward(self, signal: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        local = self.local(signal)
        context_feature = self.context(context)
        global_feature = self.global_encoder(signal, context_feature)
        local_logits = self.local_head(local)
        global_logits = self.global_head(global_feature)
        fine_logits = (
            (1.0 - self.global_fusion_weight) * local_logits
            + self.global_fusion_weight * global_logits
        )
        system_logits = self.system_head(torch.cat([local, global_feature, context_feature], dim=-1))
        return {
            "fine_multilabel_logits": fine_logits,
            "system20_logits": system_logits,
            "local_aux_logits": local_logits,
            "global_aux_logits": global_logits,
            "use_auxiliary_losses": True,
        }


def build_mcc5_model(
    model_name: str,
    local_mode: str,
    context_on: bool,
    llama_path: str,
    active_layers: int = 8,
    global_fusion_weight: float = 0.8,
):
    if model_name not in {"m0", "m2"}:
        raise ValueError("The public core release supports MCC5 model_name='m0' or 'm2'.")
    return LocalGlobalMCC5(
        model_name,
        local_mode,
        context_on,
        llama_path,
        active_layers,
        global_fusion_weight,
    )
