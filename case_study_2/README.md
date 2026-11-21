# Case study 2: uncertainty of a convolutional VAE

Question: is the epistemic uncertainty of the model higher in the transient that precedes a fault
than in the normal operation before it?

Install the package from this folder (`pip install -e .`) and run the commands from here.
Each config sets the window size, the excluded sensors and the output folder:

- `configs/w32.yaml`: windows of 32 samples, sensors P-MON-CKP, P-TPT and T-JUS-CKP
- `configs/w128.yaml`: same sensors, windows of 128 samples
- `configs/w128_two_sensors.yaml`: P-JUS-CKGL and P-TPT only

## Pipeline

```
python -m vae3w.window_size                                     # window size, all sensors
python scripts/select_sensors.py --config configs/w32.yaml       # greedy sensor selection
python scripts/window_size_final_sensors.py --config configs/w32.yaml

python -m vae3w.preprocessing --config configs/w32.yaml          # outputs_w32/processed/*.npz
python -m vae3w.hpsearch --processed outputs_w32/processed --output-dir outputs_w32/hyperparameter_search \
    --n-trials 10 --epochs-per-trial 8
python -m vae3w.train --processed outputs_w32/processed --output-dir outputs_w32/training \
    --predict-variance --epochs 50 --hyperparams-json outputs_w32/hyperparameter_search/best_hyperparams.json
python -m vae3w.uncertainty --checkpoint outputs_w32/training/checkpoints/best_model.pt \
    --npz outputs_w32/processed/val.npz --output-csv outputs_w32/uncertainty/val_uncertainty.csv
python -m vae3w.uncertainty --checkpoint outputs_w32/training/checkpoints/best_model.pt \
    --npz outputs_w32/processed/test.npz --output-csv outputs_w32/uncertainty/test_uncertainty.csv
```

The search used 10 trials of 8 epochs for w32 and 20 trials of 15 epochs for w128 (the defaults).
Training and the uncertainty estimate run on CPU. The split is by instance, stratified by class,
with a fixed seed, so it only depends on the file names of the dataset.

## Analyses

| Result | Script |
|---|---|
| Ratios, Mann-Whitney, Cliff's delta, residual relative to the reconstruction error, Spearman, results by origin | `scripts/hypothesis_tests.py --config ... --val ... --test ...` |
| Epistemic ratio per sensor (figure) | `scripts/fig_uncertainty_by_sensor.py VAL_CSV` |
| Threshold detector (Youden calibration on classes 6 and 7, F1 and false alarm per class) | `scripts/detector.py VAL_CSV TEST_CSV` |
| Real wells shared between splits | `scripts/well_overlap.py --config ...` |

## Uncertainty terms

With z fixed at the posterior mean and M dropout passes, `epistemic` is the variance of the
predicted mean and `aleatoric` the mean of the predicted variance (Kendall and Gal, 2017).
`latent_variability` is the variance of the predicted mean over M samples of z with dropout off.
Resampling z inside the dropout passes would mix both sources in the epistemic term, so they are
kept apart. The CSVs also have `recon_error` and `epistemic` per sensor.
