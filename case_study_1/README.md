# Case study 1: alarm thresholds

Four detectors trained only on normal operation (Isolation Forest, LOF, dense VAE with the
reconstruction error and with the Mahalanobis distance in the latent space) and several ways of
turning their scores into alarms.

The data protocol (`protocol.py`) follows the code of Benchmark 2 of Vargas et al. (2019):
classes 1, 2, 5, 6, 7 and 8, 180 s windows with 6 statistics per variable, 12 test windows per
instance (6 normal before the transient, 6 of the failure). Every change to that protocol is
marked in the code with `Extension:`. The main ones are an extended training set, with every normal
window before the test, and class 9 as an extra evaluation set.

Run the scripts from this folder. Results go to `RESULTS` (default `results/`).

| Step | Script | Output |
|---|---|---|
| Check against the original benchmark code | `check_original_protocol.py` | F1 micro and AUC-ROC of the IF (printed) |
| Static rules (max, p95, mean + 2 sd, p50), Benchmark 2 and class 9 | `static_thresholds.py --learning-curve` | `static_failures.csv`, `static_scores.pkl`, `learning_curve.csv` |
| Friedman and Wilcoxon-Holm tests per well | `stats_tests.py` | printed |
| Static rules over long normal operation (class 0) | `normal_operation.py` | `static_class0.csv` |
| Streaming strategies (conformal, moving, ACI, persistence, CUSUM, DSPOT) | `streaming.py` | `stream_class0.csv`, `stream_failures.csv`, `stream_summary.csv` |
| Acknowledged recalibration and alarm grouping | `recalibration.py` | `recalibration.csv` |
| DSPOT sensitivity | `dspot_sensitivity.py` | `dspot_sensitivity.csv` |

`streaming.py` caches the window scores (`stream_scores_*.pkl`), which `recalibration.py` and
`dspot_sensitivity.py` reuse.

Figures (after the steps above):

| Figure | Script |
|---|---|
| ROC curves with the operating point of each rule | `fig_roc.py` |
| Confusion matrices, transient and steady state apart | `fig_confusion_matrix.py` |

Running everything takes a few hours on a CPU, mostly the VAE in `static_thresholds.py` and
`streaming.py`.
