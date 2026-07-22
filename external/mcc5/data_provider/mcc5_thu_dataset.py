"""Manifest-driven, recording-safe MCC5-THU PyTorch dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mcc5_thu_schema import MCC5Schema, load_json, sha256_file


SPLIT_BY_FLAG = {"train": "train", "val": "validation", "test": "heldout_test", "heldout": "heldout_test"}


def _read_manifest(path):
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return pd.read_csv(fallback)
            raise
    return pd.read_csv(path)


class Dataset_MCC5THU(Dataset):
    def __init__(
        self,
        window_manifest,
        task_schema,
        local_group_schema,
        normalization_stats="",
        flag="train",
        context_mode="context_on",
        local_group_mode="measurement_correct",
        processed_cache_root="",
        allow_preview_normalization=False,
        seed=2026,
        **_,
    ):
        if flag not in SPLIT_BY_FLAG:
            raise ValueError(f"Unsupported MCC5 split flag: {flag}")
        if context_mode not in {"context_on", "context_off"}:
            raise ValueError(f"Unsupported context_mode={context_mode}")
        if local_group_mode not in {"measurement_correct", "unified"}:
            raise ValueError(f"Unsupported local_group_mode={local_group_mode}")
        self.flag = flag
        self.split = SPLIT_BY_FLAG[flag]
        self.context_mode = context_mode
        self.local_group_mode = local_group_mode
        self.seed = int(seed)
        self.schema = MCC5Schema(task_schema, local_group_schema)
        self.window_manifest_path = str(Path(window_manifest).resolve())
        manifest = _read_manifest(self.window_manifest_path)
        required = {
            "window_id", "recording_id", "source_path", "start_index", "end_index", "seq_len",
            "condition_id", "split", "system_class_id", "multilabel_targets",
            "gear_fault_present", "bearing_fault_present",
        }
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Window manifest missing columns: {sorted(missing)}")
        self.rows = manifest.loc[manifest["split"] == self.split].sort_values("window_id").reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No MCC5 windows for split={self.split}")
        if set(self.rows["split"]) != {self.split}:
            raise AssertionError("Window manifest crossed split boundary")
        self.processed_cache_root = Path(processed_cache_root) if processed_cache_root else None
        self.normalization = None
        if normalization_stats:
            stats = load_json(normalization_stats)
            if stats.get("mode") == "sampled_preview" and not allow_preview_normalization:
                raise ValueError("Preview normalization is forbidden for pilot/formal use")
            if not stats.get("assert_train_only"):
                raise ValueError("Normalization metadata does not assert train-only sources")
            if set(stats.get("source_splits", [])) != {"train"}:
                raise ValueError("Normalization includes non-train sources")
            self.normalization = stats

    def __len__(self):
        return len(self.rows)

    def _load_raw(self, row):
        path = Path(row.source_path)
        start = int(row.start_index)
        length = int(row.end_index) - start
        try:
            frame = pd.read_csv(
                path,
                usecols=list(MCC5Schema.raw_columns),
                skiprows=range(1, start + 1),
                nrows=length,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed reading {path} rows [{start}, {start + length}): {exc}") from exc
        if len(frame) != length:
            raise RuntimeError(f"Short MCC5 read {path} rows [{start}, {start + length}): got {len(frame)}")
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite MCC5 values in {path} rows [{start}, {start + length})")
        return values

    def _load_cache(self, row):
        if self.processed_cache_root is None:
            return None
        folder = self.processed_cache_root / str(row.recording_id)
        signal_path = folder / "signal.npy"
        context_path = folder / "context.npy"
        if not signal_path.exists() or not context_path.exists():
            return None
        start, end = int(row.start_index), int(row.end_index)
        signal = np.load(signal_path, mmap_mode="r")[start:end].astype(np.float32, copy=True)
        context = np.load(context_path, mmap_mode="r")[start:end].astype(np.float32, copy=True)
        return np.concatenate([context, signal], axis=1)

    def _normalize_signal(self, signal):
        if self.normalization is None:
            return signal
        mean = np.asarray(self.normalization["mean"], dtype=np.float32)[2:]
        std = np.asarray(self.normalization["std"], dtype=np.float32)[2:]
        return (signal - mean) / np.maximum(std, 1e-8)

    def __getitem__(self, index):
        row = self.rows.iloc[int(index)]
        values = self._load_cache(row)
        if values is None:
            values = self._load_raw(row)
        context_series = values[:, :2]
        signal = self._normalize_signal(values[:, 2:])
        if self.context_mode == "context_on":
            context = np.asarray([
                context_series[:, 0].mean(), context_series[:, 0].std(),
                context_series[:, 1].mean(), context_series[:, 1].std(),
            ], dtype=np.float32)
        else:
            context = np.empty((0,), dtype=np.float32)
        signal_tensor = torch.from_numpy(np.ascontiguousarray(signal))
        if self.local_group_mode == "measurement_correct":
            local_groups = {
                name: signal_tensor[:, list(indices)] for name, indices in self.schema.group_indices.items()
            }
        else:
            local_groups = {"unified": signal_tensor}
        target = np.asarray(json.loads(row.multilabel_targets), dtype=np.float32)
        return {
            "signal": signal_tensor,
            "local_groups": local_groups,
            "context": torch.from_numpy(context),
            "system_label": torch.tensor(int(row.system_class_id), dtype=torch.long),
            "multilabel_target": torch.from_numpy(target),
            "coarse_component_target": torch.tensor(
                [float(row.gear_fault_present), float(row.bearing_fault_present)], dtype=torch.float32
            ),
            "recording_id": str(row.recording_id),
            "window_id": str(row.window_id),
            "condition_id": str(row.condition_id),
            "split": str(row.split),
        }

    @property
    def manifest_identity(self):
        return {"window_manifest_sha256": sha256_file(self.window_manifest_path), "split": self.split}
