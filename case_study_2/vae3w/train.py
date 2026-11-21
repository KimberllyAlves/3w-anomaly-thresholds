"""Train the convolutional VAE on the windows written by preprocessing.py.

The model is trained on every training window (normal, transient and steady-state faulty). Training
only on normal windows would make high uncertainty on transients trivial, since the model would never
have seen them.

Loss: reconstruction + beta * KL, both summed per sample and averaged over the batch. With
--predict-variance the reconstruction term is the heteroscedastic Gaussian negative log-likelihood.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import Conv1DVAE, ConvVAEConfig

CHANNEL_PRESETS = {
    'small': (8, 16, 32),
    'medium': (16, 32, 64, 128),
    'large': (32, 64, 128, 256),
}
LOGVAR_CLAMP = 10.0
GRAD_CLIP = 5.0


class WindowDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.windows = data['windows'].astype(np.float32)
        self.labels = data['labels']
        self.class_dirs = data['class_dirs']
        self.sensors = [str(s) for s in data['sensors']]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i])

    @property
    def window_size(self):
        return self.windows.shape[1]

    @property
    def n_sensors(self):
        return self.windows.shape[2]


def vae_loss(recon, x, mu, logvar, beta=1.0, recon_logvar=None):
    if recon_logvar is None:
        rec = ((recon - x) ** 2).flatten(start_dim=1).sum(dim=1)
    else:
        lv = recon_logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
        rec = (0.5 * (lv + (x - recon) ** 2 / lv.exp())).flatten(start_dim=1).sum(dim=1)
    # the encoder log-variance is clamped too, otherwise exp() can blow up and the run diverges
    lv = logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1)
    rec, kl = rec.mean(), kl.mean()
    return {'total': rec + beta * kl, 'recon': rec, 'kl': kl}


def run_epoch(model, loader, optimizer, beta, device, train):
    model.train(train)
    sums, n = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}, 0
    with torch.set_grad_enabled(train):
        for x in loader:
            x = x.to(device)
            out = model(x)
            loss = vae_loss(out['reconstruction'], x, out['mu'], out['logvar'], beta, out.get('recon_logvar'))
            if train:
                optimizer.zero_grad()
                loss['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
            for k in sums:
                sums[k] += loss[k].item() * len(x)
            n += len(x)
    return {k: v / n for k, v in sums.items()}


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    config = ConvVAEConfig(**(cfg if isinstance(cfg, dict) else asdict(cfg)))
    model = Conv1DVAE(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    return model


def train(train_npz, val_npz, output_dir, latent_dim=32, channels=CHANNEL_PRESETS['medium'],
          kernel_size=5, dropout=0.2, lr=1e-3, batch_size=256, epochs=20, beta=1.0, seed=42,
          predict_variance=False, device='cpu'):
    torch.manual_seed(seed)
    tr, va = WindowDataset(train_npz), WindowDataset(val_npz)
    assert (tr.window_size, tr.n_sensors) == (va.window_size, va.n_sensors)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

    config = ConvVAEConfig(in_channels=tr.n_sensors, window_size=tr.window_size, latent_dim=latent_dim,
                           channels=tuple(channels), kernel_size=kernel_size, dropout=dropout,
                           predict_variance=predict_variance)
    model = Conv1DVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out = Path(output_dir)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    print(f'{len(tr)} train / {len(va)} val windows, sensors {tr.sensors}, '
          f'{model.count_parameters():,} parameters')

    best, history = float('inf'), []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        m_tr = run_epoch(model, tr_loader, optimizer, beta, device, train=True)
        m_va = run_epoch(model, va_loader, optimizer, beta, device, train=False)
        history.append({'epoch': epoch, **{f'train_{k}': v for k, v in m_tr.items()},
                        **{f'val_{k}': v for k, v in m_va.items()}, 'seconds': time.time() - t0})
        print(f"{epoch:3d}  train {m_tr['total']:.4f}  val {m_va['total']:.4f} "
              f"(recon {m_va['recon']:.4f}, kl {m_va['kl']:.4f})")

        state = {'model_state': model.state_dict(), 'config': asdict(config), 'epoch': epoch}
        torch.save(state, out / 'checkpoints' / 'last_model.pt')
        if m_va['total'] < best:
            best = m_va['total']
            torch.save({**state, 'val_loss': best}, out / 'checkpoints' / 'best_model.pt')

    with open(out / 'loss_history.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_history(history, out / 'loss_curves.png')
    return model, best


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, key, title in zip(axes, ['total', 'recon', 'kl'], ['ELBO loss', 'Reconstruction', 'KL divergence']):
        ax.plot(ep, [h[f'train_{key}'] for h in history], label='train')
        ax.plot(ep, [h[f'val_{key}'] for h in history], label='validation')
        ax.set(title=title, xlabel='epoch')
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='outputs/processed', help='folder with train.npz and val.npz')
    p.add_argument('--output-dir', default='outputs/training')
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--channels', default='medium', choices=list(CHANNEL_PRESETS))
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--predict-variance', action='store_true')
    p.add_argument('--hyperparams-json', help='best_hyperparams.json from hpsearch; overrides the model options above')
    a = p.parse_args()

    hp = dict(latent_dim=a.latent_dim, channels=a.channels, dropout=a.dropout, lr=a.lr,
              batch_size=a.batch_size, beta=a.beta)
    if a.hyperparams_json:
        hp.update(json.loads(Path(a.hyperparams_json).read_text())['best_params'])
    hp['channels'] = CHANNEL_PRESETS[hp['channels']]
    train(f'{a.processed}/train.npz', f'{a.processed}/val.npz', a.output_dir, epochs=a.epochs,
          seed=a.seed, predict_variance=a.predict_variance, **hp)
