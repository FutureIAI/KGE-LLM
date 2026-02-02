# recursive_autoencoder_integration.py
"""
Recursive Autoencoder + Dynamic Termination + Integration wrapper for TimeLLM.

Usage:
    - As demo: python recursive_autoencoder_integration.py
    - To integrate: import AutoencoderTrainer, recursive_inference_and_expand, integrate_with_time_llm
Notes:
    - Assumptions and possible adaptions are written in comments below.
"""

import os
import math
import copy
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# Utility / config
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Autoencoder architectures
# Two small variants:
#   - MLP_AE : for non-time-series/tabular N x C
#   - Conv1D_AE: for time-series B x T x C  (channels as features)
# ---------------------------

class MLP_AE(nn.Module):
    def __init__(self, C: int, hidden_dims: List[int] = [128, 64]):
        """
        Simple MLP Autoencoder for X: [N, C]
        """
        super().__init__()
        enc_layers = []
        in_dim = C
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        # latent -> decoder
        dec_layers = []
        for h in reversed(hidden_dims[:-1] + [C]):
            dec_layers.append(nn.Linear(in_dim, h))
            if h != C:
                dec_layers.append(nn.ReLU())
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_r = self.decoder(z)
        return x_r


class Conv1D_AE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int] = [64, 128], kernel=3):
        """
        Conv1D autoencoder for time-series: x shape [B, T, C]
        We'll treat features per time step as channels by transposing to [B, C, T].
        """
        super().__init__()
        layers_enc = []
        ch_in = in_channels
        for ch in hidden_channels:
            layers_enc.append(nn.Conv1d(ch_in, ch, kernel_size=kernel, padding=kernel//2, stride=1))
            layers_enc.append(nn.ReLU())
            ch_in = ch
        self.encoder = nn.Sequential(*layers_enc)

        layers_dec = []
        hidden_rev = list(reversed(hidden_channels))
        for i, ch in enumerate(hidden_rev):
            next_ch = hidden_rev[i+1] if i+1 < len(hidden_rev) else in_channels
            layers_dec.append(nn.Conv1d(ch, next_ch, kernel_size=kernel, padding=kernel//2, stride=1))
            if next_ch != in_channels:
                layers_dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers_dec)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x_t = x.permute(0, 2, 1).contiguous()
        z = self.encoder(x_t)
        xr_t = self.decoder(z)
        xr = xr_t.permute(0, 2, 1).contiguous()
        return xr

# ---------------------------
# Loss: RMSE as described
# ---------------------------
def rmse_loss(x, x_hat, eps=1e-8):
    return torch.sqrt(torch.mean((x - x_hat) ** 2) + eps)

# ---------------------------
# Recursive inference implementation (Algorithm 1)
# ---------------------------

def recursive_inference(
    ae: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    x_init: Optional[torch.Tensor] = None,
    K_max: int = 50,
    P: int = 5,
    W: int = 3,
    alpha: float = 0.01,
    eps: float = 1e-8,
    verbose: bool = False,
) -> torch.Tensor:
    """
    ae: trained autoencoder
    x: original data with missing entries (same shape as mask)
    mask: binary mask (1 known, 0 missing)
    x_init: initial guess for missing entries (if None uses mean imputation per feature/time-channel)
    returns: reconstructed tensor x_hat (same shape)
    Notes:
        - Supports x shape [N, C] or [B, T, C]
        - All tensors should be torch.float, on same device
    """
    ae.eval()
    with torch.no_grad():
        shape = x.shape
        # create initial estimation x_hat^(0)
        if x_init is None:
            # mean imputation along batch/time axis but preserving dims
            if x.dim() == 2:
                # [N, C] : mean of known values per column
                col_sum = torch.where(mask.bool(), x, torch.tensor(0., device=x.device)).sum(dim=0)
                col_cnt = mask.sum(dim=0).clamp(min=1.)
                col_mean = (col_sum / col_cnt).unsqueeze(0)  # [1, C]
                x_init = col_mean.expand_as(x)
            elif x.dim() == 3:
                # [B, T, C] : mean per channel over batch*time known values
                known = mask.bool()
                val = torch.where(known, x, torch.tensor(0., device=x.device))
                denom = known.sum(dim=(0,1)).clamp(min=1.)
                channel_mean = (val.sum(dim=(0,1)) / denom).view(1, 1, -1)
                x_init = channel_mean.expand_as(x)
            else:
                raise ValueError("Unsupported tensor dimension for x")
        # initialize
        x_hat_k = x.clone()
        x_hat_k = mask * x + (1 - mask) * x_init  # X ⊙ M + X_init ⊙ (1 - M)
        # compute initial error E(0) on missing positions
        def compute_E(cur, prev=None):
            # E(k) = RMSE((1-M) ⊙ (x - x_hat_k)) + RMSE(x_hat_k - x_hat_{k-1})  (if prev provided)
            term1 = rmse_loss((1 - mask) * (x - cur), torch.zeros_like(cur))
            if prev is None:
                return term1
            term2 = rmse_loss(cur, prev)
            return term1 + term2

        E_prev = compute_E(x_hat_k, None).item()
        eps_dyn = alpha * E_prev

        p_deg = 0
        p_rel = 0
        r_list = []

        k = 0
        while k < K_max:
            k += 1
            # forward
            # ensure correct input shape for ae: our AEs accept the same shape as x (MLP expects [N,C]; Conv1D expects [B,T,C])
            x_in = x_hat_k
            x_out = ae(x_in)
            # After forwarding, we MUST keep original known values fixed
            x_hat_new = mask * x + (1 - mask) * x_out

            # compute E(k)
            E_k = compute_E(x_hat_new, x_hat_k).item()

            if k >= 2:
                deltaE = E_prev - E_k
                # increase p_deg if deltaE < 0 (error increased)
                if deltaE < 0:
                    p_deg += 1
                else:
                    p_deg = 0

                r_k = abs(deltaE) / (E_prev + eps)
                r_list.append(r_k)
                # compute moving average over last W elements if enough
                if k >= W + 2:
                    r_bar = sum(r_list[-W:]) / float(W)
                    if r_bar < eps_dyn:
                        p_rel += 1
                    else:
                        p_rel = 0

            # update for next iter
            if verbose:
                print(f"[iter {k}] E_prev={E_prev:.6f} E_k={E_k:.6f} p_deg={p_deg} p_rel={p_rel}")
            E_prev = E_k
            x_hat_k = x_hat_new

            # termination
            if p_deg >= P or p_rel >= P:
                if verbose:
                    print("Terminated: counters exceeded P")
                break

        # final ensure known entries replaced
        x_hat_final = mask * x + (1 - mask) * x_hat_k
        return x_hat_final

# ---------------------------
# Trainer utility for autoencoder
# ---------------------------
class AutoencoderTrainer:
    def __init__(self, ae: nn.Module, lr=1e-3, wd=0.0, batch_size=64, device=DEVICE):
        self.ae = ae.to(device)
        self.optim = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=wd)
        self.criterion = lambda a, b: rmse_loss(a, b)
        self.batch_size = batch_size
        self.device = device

    def fit(self, X_train: torch.Tensor, mask_train: Optional[torch.Tensor] = None, epochs=30, val_data=None):
        """
        Train AE on only UNCORRUPTED data as per paper.
        For training, we expect X_train to be "clean" (no missing).
        If mask_train is provided (to simulate some masking augmentation), we can mask inputs during training.
        """
        # ensure tensors
        dataset = TensorDataset(X_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.ae.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in loader:
                x = batch[0].to(self.device)
                # optional input masking augmentation: here we don't corrupt training data unless mask_train provided
                x_in = x.clone()
                self.optim.zero_grad()
                x_out = self.ae(x_in)
                loss = self.criterion(x, x_out)
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * x.size(0)
            avg = total_loss / len(loader.dataset)
            if epoch % max(1, epochs // 5) == 0:
                print(f"[AE Train] Epoch {epoch}/{epochs} loss={avg:.6f}")
        return self.ae

    def save(self, path):
        torch.save(self.ae.state_dict(), path)

    def load(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        self.ae.load_state_dict(torch.load(path, map_location=map_location))
        self.ae.to(self.device)

# ---------------------------
# Integration wrapper to expand specific channels
# ---------------------------
def recursive_inference_and_expand(
    ae: nn.Module,
    X: torch.Tensor,
    missing_mask: torch.Tensor,
    target_channels: List[int],
    recursive_params: dict,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    X: input tensor (either [N,C] or [B,T,C]) on CPU or device
    missing_mask: same shape, 1 known, 0 missing
    target_channels: list of channel indices to reconstruct/expand (for multi-channel expansion)
    returns: X_aug where target channels are replaced/filled by reconstructed values (others kept)
    """
    ae = ae.to(device)
    X = X.to(device)
    mask = missing_mask.to(device)
    # call recursive inference on the whole tensor
    x_hat = recursive_inference(ae, X, mask, verbose=recursive_params.get("verbose", False),
                                K_max=recursive_params.get("K_max", 50),
                                P=recursive_params.get("P", 5),
                                W=recursive_params.get("W", 3),
                                alpha=recursive_params.get("alpha", 0.01),
                                eps=recursive_params.get("eps", 1e-8))
    # build augmented X
    # only replace target channels (if specified). If target_channels is None -> replace all missing channels.
    X_aug = X.clone()
    if X.dim() == 2:
        # [N, C]
        if target_channels is None:
            X_aug = mask * X + (1 - mask) * x_hat
        else:
            for c in target_channels:
                X_aug[:, c] = mask[:, c] * X[:, c] + (1 - mask[:, c]) * x_hat[:, c]
    elif X.dim() == 3:
        # [B, T, C]
        if target_channels is None:
            X_aug = mask * X + (1 - mask) * x_hat
        else:
            for c in target_channels:
                X_aug[:, :, c] = mask[:, :, c] * X[:, :, c] + (1 - mask[:, :, c]) * x_hat[:, :, c]
    else:
        raise ValueError("Unsupported dims")
    return X_aug

# ---------------------------
# Integration with TimeLLM
# ---------------------------
# We try to import your TimeLLM. If not present, create a stub for demo.
try:
    from models.PM_LLMoE import Model as TimeLLM  # your actual model
    HAS_TIMELLM = True
except Exception as e:
    HAS_TIMELLM = False

    class TimeLLM(nn.Module):
        """Stub TimeLLM - replace with your actual model import"""
        def __init__(self, in_channels: int, hidden_dim=64):
            super().__init__()
            self.lin = nn.Linear(in_channels, hidden_dim)
            self.out = nn.Linear(hidden_dim, in_channels)
        def forward(self, x):
            # expects [B, T, C] -> flatten last dim for demo
            if x.dim() == 3:
                B, T, C = x.shape
                y = self.lin(x.reshape(B*T, C))
                return self.out(y).reshape(B, T, C)
            else:
                return self.out(self.lin(x))

def integrate_with_time_llm(
    ae: nn.Module,
    time_llm: nn.Module,
    X_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    target_channels: List[int],
    recursive_params: dict,
    device: torch.device = DEVICE,
):
    """
    1) Use recursive AE to reconstruct/expand the specified channels in the batch
    2) Pass the augmented batch into time_llm and return outputs
    """
    X_aug = recursive_inference_and_expand(ae, X_batch, mask_batch, target_channels, recursive_params, device=device)
    # ensure time_llm on device
    time_llm = time_llm.to(device)
    time_llm.eval()
    with torch.no_grad():
        out = time_llm(X_aug.to(device))
    return out, X_aug

# ---------------------------
# Demo: synthetic dataset, train AE, integrate
# ---------------------------
def demo_run():
    print("=== Demo run: synthetic data (time-series) ===")
    # create synthetic time series data [B, T, C]
    B, T, C = 128, 32, 6
    # simulate normal data from random sinusoids + noise (shared across channels to create correlations)
    t = np.linspace(0, 2*math.pi, T)
    base = np.sin(t)[None, :, None]  # [1, T, 1]
    channel_mix = np.random.randn(C, ) * 0.5 + np.linspace(0, 1, C)
    X = (base * channel_mix.reshape(1, 1, C)).repeat(B, axis=0)
    X = X + 0.05 * np.random.randn(B, T, C)
    X = torch.tensor(X, dtype=torch.float32)

    # create missing mask: randomly zero ~20% of entries in some channels
    mask = torch.ones_like(X)
    rng = np.random.RandomState(SEED)
    missing_prob = 0.2
    rand = torch.tensor(rng.rand(*X.shape), dtype=torch.float32)
    mask[rand < missing_prob] = 0.0
    X_corrupt = mask * X  # corrupted inputs (known entries kept, missing zeroed)

    # split train (clean) and test: train only on uncorrupted samples
    # For demo, create X_train from subset without masking
    X_train = X[:int(B*0.5)].clone()  # clean data
    # Build AE (Conv1D)
    ae = Conv1D_AE(in_channels=C, hidden_channels=[32, 64], kernel=3).to(DEVICE)
    trainer = AutoencoderTrainer(ae, lr=1e-3, batch_size=16, device=DEVICE)
    trainer.fit(X_train.to(DEVICE), epochs=10)

    # recursive inference on corrupted batch
    recursive_params = {"K_max": 50, "P": 5, "W": 3, "alpha": 0.01, "eps": 1e-8, "verbose": True}
    X_aug = recursive_inference_and_expand(ae, X_corrupt.to(DEVICE), mask.to(DEVICE), target_channels=None, recursive_params=recursive_params)

    # integrate with stub TimeLLM
    tl = TimeLLM(in_channels=C, hidden_dim=32).to(DEVICE)
    out, X_aug_batched = integrate_with_time_llm(ae, tl, X_corrupt.to(DEVICE), mask.to(DEVICE), target_channels=None, recursive_params=recursive_params)
    print("Output shape from TimeLLM:", out.shape)
    print("Demo finished.")

if __name__ == "__main__":
    demo_run()
