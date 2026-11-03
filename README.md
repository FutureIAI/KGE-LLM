# KGE-LLM

**Knowledge-Guided Local Experts with Large Language Models for Industrial Time-Series Fault Diagnosis**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)

KGE-LLM is a multichannel, multilabel fault-diagnosis framework designed to
preserve component-attributed evidence while modeling temporal and
cross-component context. It combines a deterministic sensor-component routing
prior and component-specific local experts with a configurable global sequence
encoder. The public implementation supports frozen pretrained LLaMA and Mistral
backbones, compact trainable or frozen Transformers, prompt controls,
reprogramming controls, and routing ablations through one entry point.
Sensor assignments are predefined, and all local experts operate without a
learned expert-selection gate. Component-aligned local and global logits are
combined using a fixed scalar weight.

This repository contains the public research core required to inspect, train,
evaluate, and audit the disclosed method. It intentionally excludes datasets,
pretrained decoder files, trained checkpoints, cached arrays, predictions,
logs, manuscript material, private experiment orchestration, and third-party
baseline source trees.

## Why KGE-LLM?

- **Component-attributed local evidence.** Known sensor-component
  correspondence routes each channel to a dedicated local expert instead of
  learning an unconstrained expert assignment from labels alone.
- **Local-global modeling.** The local branch combines analytic-envelope
  descriptors with RMS-pooled responses from learnable temporal filters. The
  global branch encodes the complete multichannel sequence as temporal patches.
- **Configurable global encoder.** The same interface supports frozen
  pretrained decoders, compact trainable Transformers, and frozen-random
  controls, making mechanism comparisons reproducible without changing the
  local branch or label space.
- **Explicit multi-label supervision.** Component-aligned local and global
  logits are fused before prediction, with Dynamic Focal Loss and alternative
  imbalance-aware objectives available from the training entry point.
- **Leakage-aware evaluation.** Source-aware, condition-temporal metadata keeps
  held-out operating conditions outside optimization, normalization fitting,
  class weighting, early stopping, and threshold calibration.
- **Checkpoint safety.** Task-trained parameters and fixed local-descriptor
  statistics are saved by default; registered frozen decoder weights are excluded.

## Architecture

### Local-global framework

The same normalized multichannel input feeds two parallel paths. The local path
uses the predefined topology to build component-specific evidence. The global
path preserves the complete sequence and produces a shared contextual
representation. Component-aligned logits from both paths are fused in the
common multilabel space.

![KGE-LLM local-global architecture](docs/assets/fig1_kg_llmoe_architecture.png)

**Figure 1.** KGE-LLM overview: component-topology-guided local experts, a
parallel contextual sequence branch, and component-wise logit fusion.

### Patch, prompt, and backbone interface

The global branch unfolds each channel into temporal patches, embeds the
patches, and maps them into the backbone hidden space. The canonical interface
uses prototype cross-attention; a direct linear projection is available as a
controlled alternative. Optional prompt tokens are prepended before the frozen
backbone, but only patch positions contribute to mean-patch readout.

![KGE-LLM patch and reprogramming pipeline](docs/assets/fig2_reprogramming_pipeline.png)

**Figure 2.** Generic patch-prompt-interface-backbone pipeline. Dataset- and
experiment-specific dimensions are configured at runtime.

## Repository Contents

The paper and repository use the name KGE-LLM. Existing source filenames,
script names, and command-line identifiers are retained for compatibility;
use the reproduction commands below as written.

| Path | Purpose |
|---|---|
| `models/KGLLMoE.py` | Canonical KGE-LLM model and global-encoder controls |
| `models/moe_ffn.py` | Component-specific experts and routing variants |
| `models/revised_local.py` | Analytic-envelope and RMS local descriptors |
| `models/frozen_decoder_backbone.py` | Frozen decoder adapter |
| `run_main.py` | Training, validation, testing, artifact export, and safe checkpoint loading |
| `data_provider/` | PHM dataset loading and split-metadata integration |
| `scripts/split/` | Condition-temporal split construction |
| `scripts/audit/` | Split-leakage audit |
| `scripts/run_bjtu_kg_llmoe.sh` | Canonical BJTU training entry |
| `scripts/run_bjtu_ablation.sh` | Mechanism-control entry |
| `scripts/evaluate_heldout.py` | Validation-calibrated held-out evaluation |
| `scripts/verify_checkpoint_keys.py` | Trainable-only checkpoint audit |
| `external/mcc5/` | Legacy MCC5-THU adapter, separate from BJTU reproduction |

## Reproduction Guide

### 1. Create an environment

Python 3.10 or later is recommended. Install PyTorch for the CUDA version on
your system, then install the remaining dependencies:

```bash
conda create -n kge_llm python=3.10 -y
conda activate kge_llm

# Install the appropriate PyTorch build first when required by your CUDA setup.
python -m pip install -r requirements.txt
```

No dependency installation script downloads a dataset or pretrained model.
The decoder adapter is tested with Transformers 4.41.2; keep the pinned version
in `requirements.txt`. SentencePiece and Protobuf are included for LLaMA tokenizers.

### 2. Prepare external assets

Obtain BJTU-RAO and any pretrained decoder from their official distributors and
accept the corresponding licenses. Keep these assets outside Git. A typical
local layout is:

```text
KGE-LLM/
├── dataset/
│   └── Training_new/        # BJTU-RAO CSV hierarchy; not tracked
├── LLMs/
│   └── llama-7b/            # local decoder/tokenizer files; not tracked
└── cache_data/              # generated metadata and caches; not tracked
```

The expected CSV hierarchy and the MCC5 manifest contract are documented in
[`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md). Do not commit dataset
files, split instances containing machine-specific absolute paths, or model
weights.

### 3. Build and audit the BJTU condition-temporal split

Create the split with a dry run first. Conditions 1-6 form the development
pool; conditions 7-9 remain held out. The 60%/15%/15% temporal allocation uses
90% of each seen-condition source trajectory, and the terminal 10% remains
unused. A 1,024-sample embargo removes boundary-adjacent windows.
The signal cache is created automatically if absent; the dry run writes reports
but does not save split indices. The final metadata entry is `metadata.json`.

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

After reviewing the dry-run report, create the metadata and run the leakage
audit:

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
  --write

python scripts/audit/audit_condition_temporal_split.py \
  --metadata ./cache_data/splits_condition_temporal/metadata.json
```

Do not continue to training unless the audit completes successfully.

### 4. Train the canonical BJTU model

The BJTU script fits local-descriptor statistics on the complete training
partition, trains the local experts for five epochs, selects their warm-up state
on seen validation, and then starts joint training with a fresh optimizer.
The local mode is `revised_rms`; the older signed-mean mode remains available
for historical controls. Checkpoints include the fixed descriptor mean and
standard-deviation buffers but exclude frozen decoder weights.

```bash
export DATA_ROOT=./dataset/Training_new
export SPLIT_METADATA=./cache_data/splits_condition_temporal/metadata.json
export LLM_MODEL_PATH=./LLMs/llama-7b
export CHECKPOINT_ROOT=./checkpoints/kg_llmoe
export SEED=2026

bash scripts/run_bjtu_kg_llmoe.sh
```

For repeated runs, keep the split metadata and configuration fixed and provide
the intended seed list explicitly:

```bash
: "${SEEDS:?Set SEEDS to the space-separated run seeds}"
for seed in $SEEDS; do
  SEED="$seed" \
  CHECKPOINT_ROOT="./checkpoints/kg_llmoe_seed${seed}" \
  bash scripts/run_bjtu_kg_llmoe.sh
done
```

This is a unified implementation of the revised method. Historical archived
runs used different local warm-up sample budgets, so this script is not an
exact replay of every previously reported checkpoint.

Gradient norm clipping and logit clipping are **disabled by default**, in both
local warm-up and joint training. To enable them explicitly:

```bash
MAX_GRAD_NORM=1 LOGIT_CLIP=20 LOSS_LOGIT_CLIP=10 \
bash scripts/run_bjtu_kg_llmoe.sh
```

The corresponding CLI options are `--max_grad_norm`, `--logit_clip` (local
output logits), and `--loss_logit_clip` (inputs to classification losses).
Each defaults to `0`; positive values set symmetric logit limits or a gradient
norm limit. The former local/loss logit limits can be recovered with `20`/`10`.
Probability-domain epsilon bounds, descriptor bounds, and the ASL probability
shift are separate operations and remain enabled. Probability bounds use a
representable epsilon for the tensor dtype. All three clipping settings
are recorded in the run configuration.

### 5. Run mechanism controls

Run one variant at a time with the same split and seed:

```bash
VARIANT=M2_lite_trainable \
DATA_ROOT=./dataset/Training_new \
SPLIT_METADATA=./cache_data/splits_condition_temporal/metadata.json \
LLM_MODEL_PATH=./LLMs/llama-7b \
SEED=2026 \
bash scripts/run_bjtu_ablation.sh
```

Supported variants:

| Variant | Global path | Prompt | Interface |
|---|---|---|---|
| `M0_full` | Frozen pretrained LLaMA | Full | Reprogramming |
| `M1_kg_moe_only` | None | None | Not used |
| `M2_lite_trainable` | Compact trainable Transformer | None | Linear-compatible |
| `M3_lite_frozen` | Compact frozen-random Transformer | None | Linear-compatible |
| `M4_random_llama_shape` | Frozen random LLaMA-shape encoder | Full | Reprogramming |
| `M5_no_prompt` | Frozen pretrained LLaMA | None | Reprogramming |
| `M9_linear_projection` | Frozen pretrained LLaMA | Full | Linear projection |

All variants use the analytic-envelope/RMS local descriptors and training-only
standardization. `M1_kg_moe_only` runs five standalone local epochs, with no
global pathway or auxiliary global loss. Other variants use five local warm-up
epochs followed by up to fifteen joint epochs with patience three. Set
`TRAIN_EPOCHS` and `PATIENCE` explicitly when using another training budget.

### 6. Evaluate a held-out checkpoint

The evaluator loads a trainable-only checkpoint, calibrates per-label thresholds
on the seen-condition validation split, and applies the fixed thresholds to the
held-out conditions without adaptation:

```bash
python scripts/evaluate_heldout.py \
  --checkpoint /path/to/checkpoint.pt \
  --config /path/to/config.json \
  --split-metadata ./cache_data/splits_condition_temporal/metadata.json \
  --output-dir ./results/heldout/seed2026 \
  --batch-size 16 \
  --seed 2026 \
  --variant KGE-LLM
```

Training selects the checkpoint by the complete validation objective, including
the auxiliary branch losses. Training-time seen-test diagnostics use a fixed
0.5 threshold; use the evaluator above for validation-calibrated held-out results.
Both reports derive rare-label membership from the training partition only.

### 7. Verify checkpoint safety

The public entry point enables `--save_trainable_only` by default. It rejects a
save attempt if a registered frozen decoder would be serialized. Audit an
existing checkpoint with:

```bash
python scripts/verify_checkpoint_keys.py /path/to/checkpoint.pt
```

The report must show:

```text
llm_model_keys: 0
frozen_backbone_keys: 0
```

### 8. Preserve run artifacts

Each experiment directory should retain the following files when the
corresponding export options are enabled:

- `config.json`
- `metrics.json`
- `predictions.npz`
- `trainable_param_report.json`
- `checkpoint_key_report.json`
- `latency.json` when inference profiling is enabled

Record the split-metadata hash, random seed, decoder identity and revision,
active block count, routing mode, prompt mode, interface mode, loss, and readout
mode. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete
checklist.

## Legacy MCC5-THU Adapter

The public MCC5 adapter uses recording/window manifests, train-only
normalization metadata, measurement-location groups, and recording-level
aggregation. This retained adapter predates the revised BJTU local pathway and
is not an entry point for reproducing the latest MCC5 paper results. The current
reproduction guide covers BJTU only. Dataset files and private manifests are not redistributed. See
[`external/mcc5/README.md`](external/mcc5/README.md) and
[`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) for the public adapter
contract.

## Third-Party Baselines

TimeMIL, TimesNet, and other external baseline implementations are not vendored
here. Obtain them from their official repositories and review their licenses
before use. The shared split, normalization, threshold-calibration, seed, and
reporting requirements are summarized in
[`baselines/README.md`](baselines/README.md).

## Public Release Boundary

Before creating a public commit, run:

```bash
python scripts/check_release_safety.py --root .
```

The command must exit with `status: PASS`. Then manually inspect the Git index
to confirm that no dataset, pretrained model, checkpoint, cache, prediction,
log, credential, machine-specific path, third-party source tree, or private
research module is staged. The complete boundary is documented in
[`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).

## License

The KGE-LLM core is distributed under the Apache License 2.0, except for the
third-party material identified in NOTICE and individual file headers. Datasets,
pretrained models, and third-party implementations remain subject to their own
licenses and are not redistributed by this repository.
See [NOTICE](NOTICE) for source attribution and the scope of the classification code.
