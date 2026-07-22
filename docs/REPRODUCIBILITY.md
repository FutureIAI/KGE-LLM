# Reproducibility Notes

- Use source-disjoint or condition-aware split metadata and preserve its hash.
- Fit normalization and class-imbalance statistics on training data only.
- Calibrate decision thresholds on validation data only.
- Never use held-out conditions for early stopping or hyperparameter selection.
- Record the random seed, local decoder identity/revision, active layer count,
  readout mode, routing variant, and reprogramming mode.
- Use `--save_trainable_only`; the public entry point enables it by default.
- Preserve `config.json`, `metrics.json`, `predictions.npz`,
  `trainable_param_report.json`, and `checkpoint_key_report.json` for each run.

The repository contains code, not pretrained or trained weights. Exact numeric
reproduction also requires the dataset release, decoder revision, split
metadata, and compute environment used by the researcher.
