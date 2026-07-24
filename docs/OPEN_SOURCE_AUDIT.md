# Open-Source Release Audit

## Decision

This directory is a whitelist-built release candidate, not a copy of the
research workspace. It contains the minimum author-maintained implementation
needed to inspect and execute the disclosed KG-LLMoE architecture and its main
mechanism controls.

## Source Selection

- The canonical BJTU model and training entry point were selected from the
  verified LLaMA/Mistral implementation.
- Component routing and local experts are represented by one canonical module;
  older duplicate/prototype model files are excluded.
- The MCC5 package contains only the author-maintained dataset/schema adapters,
  measurement-location routing, KG-LLMoE model, and compact Transformer control.
- Condition-temporal split construction, leakage audit, and held-out evaluation
  are included without machine-specific paths or experiment-stage names.

## Exclusion Audit

No files were copied from dataset, pretrained-model, checkpoint, cache, log,
result, manuscript, backup, or downloaded third-party source directories.
Historical run queues, monitors, one-off repair scripts, raw split instances,
and unpublished modules outside the disclosed method are also excluded.

## Checkpoint Boundary

`run_main.py` enables trainable-only checkpointing by default. If a model has
registered `llm_model.*` parameters, requesting a full state-dictionary save
raises an error. Trainable-only saves emit `checkpoint_key_report.json` and fail
if an `llm_model.*` key is present. `scripts/verify_checkpoint_keys.py` provides
an independent checkpoint audit before publication.

## Validation Record

Validation was performed without training or loading pretrained weights:

- release safety scan: PASS;
- Python AST parsing: 30 files parsed;
- `run_main.py --help`: PASS;
- held-out evaluator and split/audit CLI imports: PASS;
- MCC5 model tests: 2 passed;
- ignore-rule probes for LLMs, datasets, checkpoints, caches, logs, results,
  and `.env`: all matched the intended rules.

The release safety scanner enforces a 5 MiB per-file ceiling and rejects model
weights, arrays, archives, machine-specific paths, network addresses, likely
credentials, symlinks, and blocked asset directories.

## Manual Gate Before Push

1. Review `git status --short` and `MANIFEST.sha256`.
2. Run `python scripts/check_release_safety.py --root .`.
3. Confirm that no data, pretrained weight, trained checkpoint, result artifact,
   internal document, or third-party code has been added.
4. Set the intended public remote only after this review.
