# Anomaly detection in offshore oil wells: alarm thresholds and VAE uncertainty

Code for two case studies on the [3W dataset](https://github.com/petrobras/3W) (version 2.0.0).

- `case_study_1/`: alarm threshold rules, static and in streaming mode, applied to the scores of
  four unsupervised detectors (Isolation Forest, LOF and a dense VAE scored by reconstruction
  error and by Mahalanobis distance), under the anomaly detection protocol of Benchmark 2 of
  Vargas et al. (2019).
- `case_study_2/`: a 1-D convolutional VAE with MC dropout and a heteroscedastic decoder, used to
  test whether the epistemic uncertainty is higher in the transient that precedes a fault than in
  normal operation.

Each folder has its own README with the steps and the script behind each table and figure.

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # .venv\Scripts\pip on Windows
```

Python 3.10 or later. The versions in `requirements.txt` are the ones used for the reported
results. The Isolation Forest changes slightly between scikit-learn versions.

Download the 3W dataset and point `DATA_3W` to its `dataset` folder (the one with the class
subfolders `0` to `9`):

```
export DATA_3W=/path/to/3W/dataset
```

## References

Vargas, R. E. V. et al. A realistic and public dataset with rare undesirable real events in oil
wells. Journal of Petroleum Science and Engineering, v. 181, 2019.

Kendall, A.; Gal, Y. What uncertainties do we need in Bayesian deep learning for computer vision?
NeurIPS, 2017.
