import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, c_in, scales=6, wavelet_len=32):
        super().__init__()

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
            conv = torch.nan_to_num(conv, nan=0.0, posinf=1e6, neginf=-1e6)
            convs.append(conv.mean(dim=-1))        # [B,C]
        tf = torch.stack(convs, dim=-1)            # [B,C,scales]
        tf = torch.nan_to_num(tf, nan=0.0, posinf=1e6, neginf=-1e6)
        out = self.proj(tf)                        # [B, C, c_in]

        out = out.mean(dim=-1)

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
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, label_dim)
        )

    def forward(self, x):
        """
        x: [B, C_i, T]
        returns: recon (same as x for now), logits
        """
        B, C, T = x.shape


        if not torch.isfinite(x).all():

            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        feat_env = self.hilbert(x)         # [B, C + topk]
        feat_wav = self.wavelet(x)         # [B, C]
        feat = torch.cat([feat_env, feat_wav], dim=-1)  # [B, 2C + topk]

        feat = self.pre_proj(feat)         # normalize + non-linear
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
        logits = self.mlp(feat)


        logits = torch.clamp(logits, -20.0, 20.0)


        recon = x
        return recon, logits


# ============================================================
#                    MoE (Fixed Routing)
# ============================================================
class LlamaMoEFeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        # fixed routing by channel groups
        self.expert_channels = [
            list(range(0, 9)),  # Motor: CH 0–8
            list(range(9, 15)),  # Gearbox: CH 9–14
            list(range(15, 18)),  # LA: CH 15–17
            list(range(18, 21)),  # RA: CH 18–20
        ]

        # each expert output label dims = 5 + 9 + 5 + 2 = 21
        self.label_dims = [5, 9, 5, 2]

        self.experts = nn.ModuleList([
            PhysicsExpert(len(self.expert_channels[i]), self.label_dims[i])
            for i in range(len(self.expert_channels))
        ])
    def forward(self, x):
        """
        x : [B,21,T]
        return recon : [B,21,T]
               logits: [B,21]
        """

        if not torch.isfinite(x).all():

            print("⚠️ [BUG] MoE received non-finite x in forward()")
            print("x stats:", {
                "mean": float(torch.nanmean(x)),
                "std": float(torch.nanstd(x)),
                "min": float(torch.nanmin(torch.where(torch.isfinite(x), x, torch.tensor(float('inf'), device=x.device)))),
            })

            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        recon_parts = []
        logits_parts = []

        for i, ch_list in enumerate(self.expert_channels):
            x_sub = x[:, ch_list, :]
            recon_i, logits_i = self.experts[i](x_sub)
            recon_parts.append(recon_i)

            logits_i = torch.nan_to_num(logits_i, nan=0.0, posinf=1e6, neginf=-1e6)
            logits_parts.append(logits_i)

        recon = torch.cat(recon_parts, dim=1)
        logits = torch.cat(logits_parts, dim=1)

        logits = torch.clamp(logits, -20.0, 20.0)
        return recon, logits