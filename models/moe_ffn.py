import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


COMPONENT_NAMES = ["motor", "gear", "LA", "RA"]
CORRECT_EXPERT_CHANNELS = [
    list(range(0, 9)),
    list(range(9, 15)),
    list(range(15, 18)),
    list(range(18, 21)),
]
COMPONENT_LABEL_DIMS = [5, 9, 5, 2]
PERMUTED_SOURCE_FOR_HEAD = [3, 0, 1, 2]


# ============================================================
#                  Hilbert Envelope Feature
# ============================================================
class HilbertEnvelope(nn.Module):
    def __init__(self, topk=5):
        super().__init__()
        self.topk = topk

    def forward(self, x):
        """
        x: [B, C, T]
        returns: [B, C + topk]
        """
        # FFT -> analytic -> envelope
        fft = torch.fft.rfft(x, dim=-1)                        # [B,C,F]
        # 防止 nan propagation
        fft = torch.nan_to_num(fft, nan=0.0, posinf=1e6, neginf=-1e6)

        analytic = torch.zeros_like(fft)
        if fft.shape[-1] >= 2:
            analytic[..., 0] = fft[..., 0]
            analytic[..., 1:-1] = 2 * fft[..., 1:-1]
            analytic[..., -1] = fft[..., -1]  # keep Nyquist term if present
        else:
            analytic = fft

        env = torch.fft.irfft(analytic, n=x.shape[-1], dim=-1).abs()  # envelope [B,C,T]
        env = torch.nan_to_num(env, nan=0.0, posinf=1e6, neginf=0.0)

        env_energy = env.mean(dim=-1)                          # [B,C]

        spec = torch.fft.rfft(env, dim=-1).abs()               # [B,C,F]
        spec = torch.nan_to_num(spec, nan=0.0, posinf=1e6, neginf=0.0)

        # topk on averaged across channels? keep current logic but safe
        topk_vals, _ = torch.topk(spec.mean(dim=1), self.topk) # [B, topk]
        topk_vals = torch.nan_to_num(topk_vals, nan=0.0, posinf=1e6, neginf=0.0)

        out = torch.cat([env_energy, topk_vals], dim=-1)      # [B, C + topk]
        out = torch.clamp(out, -1e3, 1e3)
        return out


# ============================================================
#                  Wavelet Time–Frequency Feature
# ============================================================
class WaveletTF(nn.Module):
    """可学习 CWT 风格：捕捉冲击 + 共振（数值稳定改进）"""
    def __init__(self, c_in, scales=6, wavelet_len=32):
        super().__init__()
        # 注意改动：初始化幅度很小，避免前向放大
        self.scales = nn.Parameter(torch.randn(scales, 1, wavelet_len) * 1e-2)
        self.proj = nn.Linear(scales, c_in)

    def forward(self, x):
        """
        x: [B, C, T]
        returns: tf_feat [B, C]
        """
        B, C, T = x.shape
        pad = self.scales.shape[-1] // 2
        x_pad = F.pad(x, (pad, pad))
        convs = []
        for k in range(len(self.scales)):
            w = self.scales[k:k+1]                 # [1,1,L]
            # repeat for groups
            w = w.repeat(C, 1, 1)
            conv = F.conv1d(x_pad, w, groups=C)    # [B,C,T]
            # 数值保护
            conv = torch.nan_to_num(conv, nan=0.0, posinf=1e6, neginf=-1e6)
            convs.append(conv.mean(dim=-1))        # [B,C]
        tf = torch.stack(convs, dim=-1)            # [B,C,scales]
        tf = torch.nan_to_num(tf, nan=0.0, posinf=1e6, neginf=-1e6)
        out = self.proj(tf)                        # [B, C, c_in]
        # proj yields shape [B,C,c_in]; 平均尺度并防止爆炸
        out = out.mean(dim=-1)
        # 最后 clamp 防止极值
        out = torch.clamp(out, -1e3, 1e3)
        return out


# ============================================================
#                    Expert Module
# ============================================================
class PhysicsExpert(nn.Module):
    def __init__(self, c_in, label_dim, hidden=128):
        super().__init__()
        self.c_in = c_in
        self.hilbert = HilbertEnvelope(topk=5)
        self.wavelet = WaveletTF(c_in)
        # 加一层投影与归一化，避免原特征尺度冲突
        feat_dim = c_in * 2 + 5
        self.pre_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU()
        )
        layers = [nn.Linear(feat_dim, hidden), nn.SiLU()]
        if label_dim is not None:
            layers.append(nn.Linear(hidden, label_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        """
        x: [B, C_i, T]
        returns: recon (same as x for now), logits
        """
        B, C, T = x.shape

        # Safety: If input contains non-finite -> print & sanitize
        if not torch.isfinite(x).all():
            # debug print once (可改为 logging)
            print("⚠️ [BUG] PhysicsExpert received non-finite input; sanitizing.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        feat_env = self.hilbert(x)         # [B, C + topk]
        feat_wav = self.wavelet(x)         # [B, C]
        feat = torch.cat([feat_env, feat_wav], dim=-1)  # [B, 2C + topk]

        feat = self.pre_proj(feat)         # normalize + non-linear
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
        hidden = self.mlp[1](self.mlp[0](feat))
        logits = self.mlp[2](hidden) if len(self.mlp) > 2 else None

        # 防止 logits 爆炸
        if logits is not None:
            logits = torch.clamp(logits, -20.0, 20.0)

        # recon 仍返回原始信号（或可替换为增强信号）
        recon = x
        if return_features:
            return recon, logits, hidden
        return recon, logits


# ============================================================
#                    MoE (Fixed Routing)
# ============================================================
class LlamaMoEFeedForward(nn.Module):
    def __init__(self, routing_variant="correct", routing_manifest="",
                 expert_head_permutation="none", unified_hidden=None):
        super().__init__()
        self.routing_variant = routing_variant
        self.routing_manifest_path = str(routing_manifest or "")
        self.expert_head_permutation = expert_head_permutation
        self.label_dims = list(COMPONENT_LABEL_DIMS)
        self.routing_manifest_hash = ""

        if routing_variant not in {"correct", "matched_random", "permuted_alignment", "unified"}:
            raise ValueError(f"Unknown routing_variant={routing_variant}")

        self.expert_channels = [list(group) for group in CORRECT_EXPERT_CHANNELS]
        if routing_variant == "matched_random":
            self.expert_channels = self._load_routing_manifest(self.routing_manifest_path)

        if routing_variant == "unified":
            if unified_hidden is None:
                unified_hidden = self.find_parameter_matched_hidden()["unified_hidden"]
            self.unified_hidden = int(unified_hidden)
            self.unified_expert = PhysicsExpert(21, None, hidden=self.unified_hidden)
            self.unified_heads = nn.ModuleList([
                nn.Linear(self.unified_hidden, dim) for dim in self.label_dims
            ])
            self.experts = nn.ModuleList()
        else:
            self.unified_hidden = None
            self.unified_expert = None
            self.unified_heads = nn.ModuleList()
            self.experts = nn.ModuleList([
                PhysicsExpert(len(self.expert_channels[i]), self.label_dims[i])
                for i in range(len(self.expert_channels))
            ])

        self._assert_routing()

    @staticmethod
    def _module_params(module):
        return sum(p.numel() for p in module.parameters())

    @classmethod
    def reference_local_param_count(cls):
        experts = nn.ModuleList([
            PhysicsExpert(len(CORRECT_EXPERT_CHANNELS[i]), COMPONENT_LABEL_DIMS[i])
            for i in range(4)
        ])
        return cls._module_params(experts)

    @classmethod
    def unified_local_param_count(cls, hidden):
        expert = PhysicsExpert(21, None, hidden=hidden)
        heads = nn.ModuleList([nn.Linear(hidden, dim) for dim in COMPONENT_LABEL_DIMS])
        return cls._module_params(expert) + cls._module_params(heads)

    @classmethod
    def find_parameter_matched_hidden(cls):
        reference = cls.reference_local_param_count()
        candidates = []
        for hidden in range(1, 2049):
            count = cls.unified_local_param_count(hidden)
            error = abs(count - reference) / reference
            candidates.append((error, hidden, count))
        error, hidden, count = min(candidates)
        return {
            "reference_local_params": reference,
            "unified_local_params": count,
            "unified_hidden": hidden,
            "relative_error": error,
        }

    def local_parameter_report(self):
        current = self._module_params(self.unified_expert) + self._module_params(self.unified_heads) \
            if self.routing_variant == "unified" else self._module_params(self.experts)
        reference = self.reference_local_param_count()
        return {
            "routing_variant": self.routing_variant,
            "reference_local_params": reference,
            "local_params": current,
            "relative_error": abs(current - reference) / reference,
            "unified_hidden": self.unified_hidden,
        }

    def _load_routing_manifest(self, path):
        if not path:
            raise ValueError("matched_random requires --routing_manifest")
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text())
        entries = payload.get("assignments", payload.get("channels", []))
        groups = {name: [] for name in COMPONENT_NAMES}
        for entry in entries:
            groups[str(entry["assigned_expert"])].append(int(entry["channel_id"]))
        raw = manifest_path.read_bytes()
        self.routing_manifest_hash = hashlib.sha256(raw).hexdigest()
        expected_hash = payload.get("manifest_hash")
        if expected_hash and expected_hash != self.routing_manifest_hash:
            canonical = dict(payload)
            canonical.pop("manifest_hash", None)
            canonical_hash = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if expected_hash != canonical_hash:
                raise RuntimeError(f"Routing manifest hash mismatch: {path}")
        return [sorted(groups[name]) for name in COMPONENT_NAMES]

    def _assert_routing(self):
        if self.routing_variant == "unified":
            return
        flattened = [channel for group in self.expert_channels for channel in group]
        if sorted(flattened) != list(range(21)):
            raise RuntimeError(f"Routing must contain each channel exactly once: {flattened}")
        expected_sizes = [len(group) for group in CORRECT_EXPERT_CHANNELS]
        if [len(group) for group in self.expert_channels] != expected_sizes:
            raise RuntimeError("Routing expert group sizes do not match R0")

    def forward(self, x, return_details=False, intervention=None):
        """
        x : [B,21,T]
        return recon : [B,21,T]
               logits: [B,21]
        """
        # 1) 基础输入检查
        if not torch.isfinite(x).all():
            # 打印出异常位置与统计，便于排查
            print("⚠️ [BUG] MoE received non-finite x in forward()")
            print("x stats:", {
                "mean": float(torch.nanmean(x)),
                "std": float(torch.nanstd(x)),
                "min": float(torch.nanmin(torch.where(torch.isfinite(x), x, torch.tensor(float('inf'), device=x.device)))),
            })
            # 尝试 sanitize（生产可改为抛错）
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.routing_variant == "unified":
            _, _, shared = self.unified_expert(x, return_features=True)
            features = [shared for _ in COMPONENT_NAMES]
            logits_parts = [head(shared) for head in self.unified_heads]
        else:
            features = []
            direct_logits = []
            for i, ch_list in enumerate(self.expert_channels):
                x_sub = x[:, ch_list, :]
                _, logits_i, feature_i = self.experts[i](x_sub, return_features=True)
                features.append(feature_i)
                direct_logits.append(logits_i)

            if intervention is not None:
                expert_index = int(intervention["expert_index"])
                permutation = intervention["permutation"].to(features[expert_index].device)
                features[expert_index] = features[expert_index][permutation]

            if self.routing_variant == "permuted_alignment" or self.expert_head_permutation == "cyclic":
                logits_parts = [
                    self.experts[target_idx].mlp[2](features[source_idx])
                    for target_idx, source_idx in enumerate(PERMUTED_SOURCE_FOR_HEAD)
                ]
            elif intervention is not None:
                logits_parts = [
                    self.experts[i].mlp[2](features[i]) for i in range(4)
                ]
            else:
                logits_parts = direct_logits

        logits_parts = [torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6) for v in logits_parts]
        recon = x
        logits = torch.cat(logits_parts, dim=1)
        # clamp final logits as additional safety
        logits = torch.clamp(logits, -20.0, 20.0)
        if return_details:
            return recon, logits, {
                "component_representations": features,
                "component_logits": logits_parts,
                "routing_variant": self.routing_variant,
            }
        return recon, logits
