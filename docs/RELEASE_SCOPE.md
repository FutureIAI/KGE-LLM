# Public Release Scope

## Public Core

The release candidate exposes the code needed to inspect and run the paper's
principal architecture: component routing, local experts, global encoder
selection, prompt/reprogramming controls, local-global prediction fusion,
classification losses, metrics, split metadata, and safe checkpoint loading.

The canonical implementation is `models/CTLLMoE.py`. Historical duplicate
models are intentionally omitted so that the public API has one source of truth.

## Excluded Assets

The following categories must remain outside the public repository:

- raw or processed datasets;
- pretrained LLaMA/Mistral files and tokenizer assets;
- trained model checkpoints and optimizer states;
- caches, split instances containing machine-specific absolute paths, logs,
  predictions, result workbooks, and manuscript material;
- backup files, remote-machine scripts, status monitors, and historical
  experiment directories;
- third-party baseline source trees and downloaded repositories;
- unpublished research modules that are not required to execute the disclosed
  CT-LLMoE method.

## Third-Party Code

TimeMIL, TimesNet, and other comparison methods are not vendored in this core
release. Reproduce those comparisons from the official implementations cited
in the paper after independently reviewing their licenses.

## Release Gate

Before any push, run:

```bash
python scripts/check_release_safety.py --root .
```

A public push is allowed only when the command exits with status 0 and a manual
review confirms that the Git index contains only the intended release files.
