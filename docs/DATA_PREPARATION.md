# Data Preparation

## BJTU-RAO / PHM Layout

`Dataset_PHM` expects CSV files below label directories. Each CSV contains the
multichannel signal columns; the label directory encodes the component labels.
The public repository does not include the dataset or generated cache files.

Create condition-temporal metadata with a dry run first:

```bash
python scripts/split/build_condition_temporal_split.py \
  --root_path ./dataset/Training_new \
  --seq_len 512 \
  --window_stride 1024 \
  --seen_conditions 1,2,3,4,5,6 \
  --heldout_conditions 7,8,9 \
  --train_ratio 0.60 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --embargo 1024 \
  --output_dir ./cache_data/splits_condition_temporal \
  --dry_run
```

After reviewing the generated report, replace `--dry_run` with `--write`.
The three seen-condition ratios cover 90% of each source trajectory; the
remaining 10% tail is intentionally unused. Held-out conditions are written
only to the held-out split and must not contribute to normalization, loss
weights, threshold calibration, early stopping, or model selection.

Audit the written metadata before training:

```bash
python scripts/audit/audit_condition_temporal_split.py \
  --metadata ./cache_data/splits_condition_temporal/metadata.json
```

## MCC5-THU

The MCC5 adapter consumes a recording/window manifest plus task and local-group
schema JSON files. Split membership is taken from the manifest, and the adapter
rejects normalization metadata unless it explicitly asserts train-only sources.
Dataset files and manifests are not redistributed in this repository.
