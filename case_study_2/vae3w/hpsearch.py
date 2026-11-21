"""Hyperparameter search with Optuna (TPE sampler and median pruning), scored on the validation split.

Each trial trains for a few epochs only. With small budgets the default warm-up settings of Optuna
would spend the whole budget on random trials, so the number of startup trials of the sampler and
of the pruner scales with --n-trials. The study is stored in SQLite, so running the same command
again resumes it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from .model import Conv1DVAE, ConvVAEConfig
from .train import CHANNEL_PRESETS, WindowDataset, run_epoch

PARAMS = ['latent_dim', 'channels', 'dropout', 'lr', 'batch_size', 'beta']


def objective(tr, va, epochs, channel_choices, predict_variance=True, device='cpu'):
    def run(trial):
        latent_dim = trial.suggest_categorical('latent_dim', [8, 16, 32, 64])
        channels = trial.suggest_categorical('channels', channel_choices)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        beta = trial.suggest_float('beta', 0.1, 2.0, log=True)

        torch.manual_seed(42)
        config = ConvVAEConfig(in_channels=tr.n_sensors, window_size=tr.window_size, latent_dim=latent_dim,
                               channels=CHANNEL_PRESETS[channels], dropout=dropout,
                               predict_variance=predict_variance)
        try:
            model = Conv1DVAE(config).to(device)
        except ValueError:          # depth not compatible with the window size
            raise optuna.TrialPruned()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

        best = float('inf')
        for epoch in range(1, epochs + 1):
            run_epoch(model, tr_loader, optimizer, beta, device, train=True)
            loss = run_epoch(model, va_loader, optimizer, beta, device, train=False)['total']
            best = min(best, loss)
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best
    return run


def save(study, out, info):
    with open(out / 'trials.csv', 'w') as f:
        f.write('trial,state,value,' + ','.join(PARAMS) + '\n')
        for t in study.trials:
            value = '' if t.value is None else f'{t.value:.6f}'
            f.write(f'{t.number},{t.state.name},{value},' + ','.join(str(t.params.get(k, '')) for k in PARAMS) + '\n')
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return
    best = study.best_trial
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    result = {'best_value': best.value, 'best_params': best.params, **info,
              'n_completed': len(done), 'n_pruned': pruned}
    (out / 'best_hyperparams.json').write_text(json.dumps(result, indent=2))


def run_search(train_npz, val_npz, output_dir, n_trials=20, epochs_per_trial=15, seed=42,
               predict_variance=True, exclude_channels=(), study_name='hpsearch'):
    tr, va = WindowDataset(train_npz), WindowDataset(val_npz)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    choices = [k for k in CHANNEL_PRESETS if k not in exclude_channels]

    study = optuna.create_study(
        study_name=study_name, storage=f'sqlite:///{out / "study.db"}', load_if_exists=True,
        direction='minimize',
        sampler=TPESampler(seed=seed, n_startup_trials=max(2, n_trials // 2)),
        pruner=MedianPruner(n_startup_trials=max(2, n_trials // 4), n_warmup_steps=max(2, epochs_per_trial // 2)))

    t0 = time.time()
    info = lambda: {'predict_variance': predict_variance, 'epochs_per_trial': epochs_per_trial,
                    'search_time_seconds': time.time() - t0}
    study.optimize(objective(tr, va, epochs_per_trial, choices, predict_variance), n_trials=n_trials,
                   callbacks=[lambda s, t: save(s, out, info())])
    save(study, out, info())
    print(json.dumps(study.best_trial.params, indent=2))
    return study


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='outputs/processed')
    p.add_argument('--output-dir', default='outputs/hyperparameter_search')
    p.add_argument('--n-trials', type=int, default=20)
    p.add_argument('--epochs-per-trial', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-predict-variance', dest='predict_variance', action='store_false')
    p.add_argument('--exclude-channels', default='', help='comma separated, e.g. large')
    p.add_argument('--study-name', default='hpsearch')
    a = p.parse_args()
    run_search(f'{a.processed}/train.npz', f'{a.processed}/val.npz', a.output_dir, a.n_trials,
               a.epochs_per_trial, a.seed, a.predict_variance,
               [s for s in a.exclude_channels.split(',') if s], a.study_name)
