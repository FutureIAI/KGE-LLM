# Public Package Checks

The repository contains the author-maintained KGE-LLM classification core,
condition-temporal split utilities, validation-calibrated evaluation, and a
legacy MCC5 adapter. Datasets, decoder files, trained weights and private
experiment artifacts are excluded.

## Automated Checks

Run these commands from the repository root:

```bash
python -m pip install pytest
python -m pytest -q
python scripts/check_release_safety.py --root .
shasum -a 256 -c MANIFEST.sha256
bash -n scripts/run_bjtu_kg_llmoe.sh
bash -n scripts/run_bjtu_ablation.sh
```

Tests use synthetic inputs and tiny random decoder configurations. They do not
download pretrained weights or run dataset experiments. They cover analytic
envelopes, RMS pooling, fixed descriptor statistics, checkpoint round trips,
local-only separation, random-frozen decoder forward/backward, optional clipping,
training/validation objective agreement, and empty-cache split preparation.

## Checkpoint Boundary

`run_main.py` saves task-trained parameters and fixed local-descriptor buffers.
Registered frozen decoder weights are excluded. Audit a saved checkpoint with
`scripts/verify_checkpoint_keys.py` before sharing it.

## File Boundary

The release scanner checks for model weights, arrays, archives, likely
credentials, machine-specific paths, network addresses, symlinks, and blocked
asset directories. Inspect the Git index as well: generated data, logs,
predictions and private documents must not be staged.

`MANIFEST.sha256` covers the public files except the manifest itself. Regenerate
it whenever tracked source, documentation or images change. Third-party source
attribution is recorded in `NOTICE`.
