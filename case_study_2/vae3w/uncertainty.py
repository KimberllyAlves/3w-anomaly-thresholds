"""Uncertainty decomposition with MC dropout (Kendall and Gal, 2017).

For each window, with the model trained with --predict-variance:
    epistemic           variance of the predicted mean over M dropout passes, z fixed at the posterior mean
    aleatoric           mean of the predicted variance over the same passes
    latent_variability  variance of the predicted mean over M samples of z, dropout off
    total_uncertainty   epistemic + aleatoric
    recon_error         squared error of the deterministic reconstruction (dropout off, z = mu)

Dropout and z sampling are kept apart because resampling z in every dropout pass mixes the
stochasticity of the VAE bottleneck into the epistemic term. recon_error and epistemic are also
written per sensor (mean over time only).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .train import WindowDataset, load_model


@torch.no_grad()
def estimate_uncertainty(model, x, n_z_samples=30, n_mc_dropout=30):
    """x: (batch, window, sensors)."""
    if not model.config.predict_variance:
        raise ValueError('the model must be trained with --predict-variance')
    model.eval()
    x = x.permute(0, 2, 1)
    mu, logvar = model.encode(x)
    ref, _ = model.decode(mu)
    sq = (ref - x) ** 2

    model.enable_mc_dropout()
    means, variances = [], []
    for _ in range(n_mc_dropout):
        m, lv = model.decode(mu)
        means.append(m)
        variances.append(lv.clamp(-10, 10).exp())
    model.eval()
    var_dropout = torch.stack(means).var(dim=0)
    aleatoric = torch.stack(variances).mean(dim=0).flatten(start_dim=1).mean(dim=1)

    latent = torch.stack([model.decode(model.reparameterize(mu, logvar))[0] for _ in range(n_z_samples)])

    epistemic = var_dropout.flatten(start_dim=1).mean(dim=1)
    return {
        'recon_error': sq.flatten(start_dim=1).mean(dim=1),
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'latent_variability': latent.var(dim=0).flatten(start_dim=1).mean(dim=1),
        'total_uncertainty': aleatoric + epistemic,
        'recon_error_per_sensor': sq.mean(dim=2),
        'epistemic_per_sensor': var_dropout.mean(dim=2),
    }


def run(checkpoint, npz, output_csv, n_z_samples=30, n_mc_dropout=30, batch_size=32, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model = load_model(checkpoint)
    ds = WindowDataset(npz)
    scalar = ['recon_error', 'aleatoric', 'epistemic', 'latent_variability', 'total_uncertainty']

    rows = []
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        out = estimate_uncertainty(model, torch.stack([ds[i] for i in idx]), n_z_samples, n_mc_dropout)
        for j, i in enumerate(idx):
            row = {'window_index': i, 'label': int(ds.labels[i]), 'class_dir': int(ds.class_dirs[i])}
            row.update({k: out[k][j].item() for k in scalar})
            for s, name in enumerate(ds.sensors):
                row[f'recon_error_{name}'] = out['recon_error_per_sensor'][j, s].item()
                row[f'epistemic_{name}'] = out['epistemic_per_sensor'][j, s].item()
            rows.append(row)

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} windows -> {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='outputs/training/checkpoints/best_model.pt')
    p.add_argument('--npz', default='outputs/processed/val.npz')
    p.add_argument('--output-csv', default='outputs/uncertainty/val_uncertainty.csv')
    p.add_argument('--n-z-samples', type=int, default=30)
    p.add_argument('--n-mc-dropout', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    run(a.checkpoint, a.npz, a.output_csv, a.n_z_samples, a.n_mc_dropout, a.batch_size, a.seed)
