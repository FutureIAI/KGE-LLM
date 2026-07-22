#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PM-LLMoE checkpoint without loading a model")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")

    keys = [str(key) for key in state]
    llm_keys = [key for key in keys if key.startswith("llm_model.")]
    frozen_backbone_keys = [
        key for key in keys
        if "frozen_llama." in key or "frozen_decoder." in key or "frozen_backbone." in key
    ]
    report = {
        "checkpoint": str(path.resolve()),
        "checkpoint_size_mb": path.stat().st_size / 1024 ** 2,
        "format": checkpoint.get("format", "raw_state_dict") if isinstance(checkpoint, dict) else "raw_state_dict",
        "save_trainable_only": bool(checkpoint.get("save_trainable_only", False)) if isinstance(checkpoint, dict) else False,
        "total_keys": len(keys),
        "llm_model_keys": len(llm_keys),
        "frozen_backbone_keys": len(frozen_backbone_keys),
        "safe": not llm_keys and not frozen_backbone_keys,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not report["safe"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
