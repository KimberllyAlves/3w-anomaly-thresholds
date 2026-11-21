"""The four detectors and the static threshold rules. Higher score = more anomalous."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import vae
from protocol import RANDOM_STATE

DETECTORS = ['IF', 'LOF', 'VAE-Rec', 'VAE-Maha']

RULES = {
    'max': lambda s: s.max(),                     # Vargas et al. (2019)
    'p95': lambda s: np.percentile(s, 95),
    'm2sd': lambda s: s.mean() + 2 * s.std(),     # Schmöckel et al. (2026)
    'p50': lambda s: np.median(s),
}


def score(Xtr, Xte, detectors=DETECTORS):
    """{detector: (training scores, scores of Xte)}. Training scores are in-sample."""
    out = {}
    if 'IF' in detectors:
        m = IsolationForest(random_state=RANDOM_STATE).fit(Xtr)
        out['IF'] = (-m.score_samples(Xtr), -m.score_samples(Xte))
    if 'LOF' in detectors:
        m = LocalOutlierFactor(n_neighbors=min(20, len(Xtr) - 1), novelty=True).fit(Xtr)
        out['LOF'] = (-m.negative_outlier_factor_, -m.score_samples(Xte))
    if 'VAE-Rec' in detectors or 'VAE-Maha' in detectors:
        m = vae.fit(Xtr)
        e_tr, mu_tr = vae.scores(m, Xtr)
        e_te, mu_te = vae.scores(m, Xte)
        if 'VAE-Rec' in detectors:
            out['VAE-Rec'] = (e_tr, e_te)
        if 'VAE-Maha' in detectors:
            out['VAE-Maha'] = (vae.mahalanobis(mu_tr, mu_tr), vae.mahalanobis(mu_tr, mu_te))
    return out
