"""Dense VAE used as detector in case study 1.

Input (features) -> 24 -> 12 -> latent 8 -> 12 -> 24 -> input. ELBO with summed squared error
plus KL, Adam (lr 1e-3), batch 16, up to 100 epochs, early stopping on the training loss
(patience 15), latent log-variance clipped to [-4, 4], gradient norm clipped to 1.

Two scores from the same model:
  VAE-Rec   mean squared reconstruction error, decoding the latent mean (deterministic)
  VAE-Maha  Mahalanobis distance of the latent mean to the training latent means
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HIDDEN, LATENT = 24, 8
EPOCHS, LR, BATCH, PATIENCE = 100, 1e-3, 16, 15
LOGVAR_CLIP, GRAD_CLIP, SEED = 4.0, 1.0, 42
torch.set_num_threads(1)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN // 2, LATENT)
        self.fc_lv = nn.Linear(HIDDEN // 2, LATENT)
        self.dec = nn.Sequential(nn.Linear(LATENT, HIDDEN // 2), nn.ReLU(),
                                 nn.Linear(HIDDEN // 2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, d))

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), torch.clamp(self.fc_lv(h), -LOGVAR_CLIP, LOGVAR_CLIP)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv


def _tensor(x):
    return torch.tensor(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)


def fit(Xtr):
    set_seeds()
    m = VAE(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(_tensor(Xtr)), batch_size=BATCH, shuffle=True)
    best, wait, state = float('inf'), 0, None
    m.train()
    for _ in range(EPOCHS):
        total = 0.0
        for (b,) in loader:
            opt.zero_grad()
            r, mu, lv = m(b)
            loss = nn.functional.mse_loss(r, b, reduction='sum') - 0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item()
        if not np.isfinite(total):
            break
        if total < best - 1e-4:
            best, wait, state = total, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if state:
        m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def scores(m, X):
    """Reconstruction error (from the latent mean) and the latent means."""
    t = _tensor(X)
    mu, _ = m.encode(t)
    return ((t - m.dec(mu)) ** 2).mean(1).numpy(), mu.numpy()


def mahalanobis(mu_ref, mu):
    cov = np.cov(mu_ref.T) + np.eye(mu_ref.shape[1]) * 1e-6
    d = mu - mu_ref.mean(0)
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', d, np.linalg.pinv(cov), d), 0))
