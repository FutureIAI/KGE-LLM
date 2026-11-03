"""Build the PHM signal cache without requiring split metadata."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def component_labels(dirname):
    labels = [0] * 21
    ranges = {"M": (0, 5), "G": (5, 9), "LA": (14, 5), "RA": (19, 2)}
    for component in dirname.split("_"):
        for state in component.split("+"):
            prefix = next(
                (key for key in ("LA", "RA", "M", "G") if state.startswith(key)), None
            )
            if prefix is None or not state[len(prefix) :].isdigit():
                raise ValueError(f"Invalid component label: {state}")
            index = int(state[len(prefix) :])
            offset, count = ranges[prefix]
            if not 0 <= index < count:
                raise ValueError(f"Component label out of range: {state}")
            labels[offset + index] = 1
    return labels


def prepare_phm_cache(root_path, seq_len=512, stride=1024, cache_dir="./cache_data"):
    root = Path(root_path).resolve()
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")
    cache = Path(cache_dir) / f"phm_cache_optimized_v2_seq{seq_len}_stride{stride}.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("root_path") and Path(payload["root_path"]).resolve() != root:
            raise ValueError(
                "Cache belongs to a different dataset root; choose another cache_dir"
            )
        return cache
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No PHM CSV files found below {root}")
    arrays, pointers, labels = [], [], {}
    for file_index, path in enumerate(files):
        values = pd.read_csv(path).to_numpy(dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 21:
            raise ValueError(f"Expected 21 signal columns in {path}")
        label_dir = path.parent.name
        labels.setdefault(label_dir, component_labels(label_dir))
        arrays.append(values)
        pointers.extend(
            (file_index, start, label_dir)
            for start in range(0, len(values) - seq_len + 1, stride)
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "all_file_data": arrays,
            "window_pointers": pointers,
            "labels": labels,
            "root_path": str(root),
            "seq_len": seq_len,
            "window_stride": stride,
        },
        cache,
    )
    return cache
