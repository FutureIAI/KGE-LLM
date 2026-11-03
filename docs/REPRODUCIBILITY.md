# Reproducibility Notes

- Use source-disjoint or condition-aware split metadata and preserve its hash.
- Fit normalization and class-imbalance statistics on training data only.
- Calibrate decision thresholds on validation data only.
- Never use held-out conditions for early stopping or hyperparameter selection.
- Record the random seed, local decoder identity/revision, active layer count,
  readout mode, routing variant, and reprogramming mode.
- Use the pinned Transformers version and install the listed tokenizer dependencies.
- Record `max_grad_norm`, `logit_clip`, and `loss_logit_clip`. All default to zero
  (disabled); optional limits apply during both warm-up and joint optimization.
- Use `--save_trainable_only`; the public entry point enables it by default.
- In revised-local mode, fit descriptor means and population standard deviations
  once on the complete training partition. Keep them fixed during local warm-up,
  joint training, and evaluation; they are part of the saved checkpoint state.
- The unified BJTU entry runs five local-only warm-up epochs before creating a
  fresh joint optimizer. Select the warm-up state on seen validation.
- The standalone local-only entry trains the local model alone for five epochs.
- Validate with the same total objective used for training. Validation and test
  loaders preserve sample order and retain the final partial batch.
- Use `scripts/evaluate_heldout.py` for validation-calibrated held-out metrics.
  The training entry's seen-test diagnostics use fixed 0.5 thresholds; do not
  mix the two reports. Rare labels are defined by training-split frequencies.
- Preserve `config.json`, `metrics.json`, `predictions.npz`,
  `trainable_param_report.json`, and `checkpoint_key_report.json` for each run.

The repository contains code, not pretrained or trained weights. Exact numeric
reproduction also requires the dataset release, decoder revision, split
metadata, and compute environment used by the researcher.
