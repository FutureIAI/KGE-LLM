#!/usr/bin/env python3
"""Audit condition-aware temporal PHM split files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence, Tuple

import torch


LABEL_NAMES = [
    "M0", "M1", "M2", "M3", "M4",
    "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
    "LA0", "LA1", "LA2", "LA3", "LA4",
    "RA0", "RA1",
]

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def segment_bounds(num_rows: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict:
    train_end = int(math.floor(num_rows * train_ratio))
    val_end = int(math.floor(num_rows * (train_ratio + val_ratio)))
    test_end = int(math.floor(num_rows * (train_ratio + val_ratio + test_ratio)))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def usable_bounds(base_bounds: dict, split: str, embargo: int) -> Tuple[int, int]:
    start, end = base_bounds[split]
    if split == "train":
        return start, end - embargo
    if split == "val":
        return start + embargo, end - embargo
    if split == "test":
        return start + embargo, end
    raise ValueError(split)


def in_region(start: int, seq_len: int, region: Tuple[int, int]) -> bool:
    return start >= region[0] and start + seq_len <= region[1]


def empty_stats() -> dict:
    return {
        split: {
            "window_count": 0,
            "source_files": set(),
            "label_dirs": set(),
            "conditions": set(),
            "condition_counts": Counter(),
            "label_dir_counts": Counter(),
            "label_positive_counts": Counter({name: 0 for name in LABEL_NAMES}),
            "normal_windows": 0,
            "fault_windows": 0,
        }
        for split in ["train", "val", "test", "heldout"]
    }


def finalize_stats(stats: dict) -> dict:
    out = {}
    for split, item in stats.items():
        total = item["window_count"]
        out[split] = {
            "window_count": total,
            "source_count": len(item["source_files"]),
            "label_dir_count": len(item["label_dirs"]),
            "condition_coverage": sorted(item["conditions"]),
            "condition_counts": dict(sorted(item["condition_counts"].items(), key=lambda kv: int(kv[0]))),
            "label_dir_counts": dict(sorted(item["label_dir_counts"].items())),
            "label_positive_counts": dict(item["label_positive_counts"]),
            "normal_windows": item["normal_windows"],
            "fault_windows": item["fault_windows"],
            "normal_ratio": (item["normal_windows"] / total) if total else 0.0,
        }
    return out


def source_lookup(metadata: dict) -> dict:
    return {int(row["file_index"]): row for row in metadata["source_records"]}


def audit(metadata_path: Path, docs_dir: Path, results_dir: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_path = Path(metadata["cache_path"])
    cache_exists = cache_path.exists()
    cache_size_match = cache_exists and cache_path.stat().st_size == metadata.get("cache_file_size")

    print(f"Loading cache for audit: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    window_pointers = cache["window_pointers"]
    cache_len = len(window_pointers)
    cache_file_count = len(cache["all_file_data"])

    split_indices = {}
    for split in ["train", "val", "test", "heldout"]:
        path = Path(metadata["split_files"][split])
        split_indices[split] = list(torch.load(path, map_location="cpu", weights_only=False))

    errors = []
    if cache_len != metadata["num_windows"]:
        errors.append(f"cache length mismatch: {cache_len} != {metadata['num_windows']}")
    if cache_file_count != metadata["num_source_files"]:
        errors.append(f"cache source file count mismatch: {cache_file_count} != {metadata['num_source_files']}")
    if not cache_size_match:
        errors.append("cache file size mismatch or cache missing")

    split_sets = {}
    for split, indices in split_indices.items():
        if not indices:
            errors.append(f"{split} split is empty")
        if min(indices) < 0 or max(indices) >= cache_len:
            errors.append(f"{split} split has out-of-range index")
        split_sets[split] = set(indices)
        if len(split_sets[split]) != len(indices):
            errors.append(f"{split} split has duplicate indices")
    for left in ["train", "val", "test", "heldout"]:
        for right in ["train", "val", "test", "heldout"]:
            if left >= right:
                continue
            overlap = len(split_sets[left] & split_sets[right])
            if overlap:
                errors.append(f"{left}/{right} overlap: {overlap}")

    records = source_lookup(metadata)
    seen = set(metadata["seen_conditions"])
    heldout = set(metadata["heldout_conditions"])
    stats = empty_stats()
    audit_rows = []
    boundary_errors = []
    condition_errors = []
    embargo_errors = []

    for split, indices in split_indices.items():
        for real_idx in indices:
            file_idx, start, label_dir = window_pointers[real_idx]
            rec = records[int(file_idx)]
            condition_id = int(rec["condition_id"])
            if rec["label_dir"] != label_dir:
                condition_errors.append(f"label mismatch idx={real_idx}: {label_dir} != {rec['label_dir']}")
                continue
            if split == "heldout":
                if condition_id not in heldout:
                    condition_errors.append(f"heldout has non-heldout condition idx={real_idx} cond={condition_id}")
            else:
                if condition_id not in seen:
                    condition_errors.append(f"{split} has non-seen condition idx={real_idx} cond={condition_id}")
                base = segment_bounds(
                    int(rec["num_rows"]),
                    float(metadata["train_ratio"]),
                    float(metadata["val_ratio"]),
                    float(metadata["test_ratio"]),
                )
                raw = base[split]
                usable = usable_bounds(base, split, int(metadata["embargo"]))
                if not in_region(start, int(metadata["seq_len"]), raw):
                    boundary_errors.append(f"{split} crosses raw boundary idx={real_idx} start={start}")
                if not in_region(start, int(metadata["seq_len"]), usable):
                    embargo_errors.append(f"{split} violates embargo idx={real_idx} start={start}")

            item = stats[split]
            item["window_count"] += 1
            item["source_files"].add(rec["csv_path"])
            item["label_dirs"].add(rec["label_dir"])
            item["conditions"].add(condition_id)
            item["condition_counts"][str(condition_id)] += 1
            item["label_dir_counts"][rec["label_dir"]] += 1
            for name, value in zip(LABEL_NAMES, rec["label_vector"]):
                if value:
                    item["label_positive_counts"][name] += 1
            if rec["label_dir"] == "M0_G0_LA0_RA0":
                item["normal_windows"] += 1
            else:
                item["fault_windows"] += 1

    if condition_errors:
        errors.extend(condition_errors[:20])
    if boundary_errors:
        errors.extend(boundary_errors[:20])
    if embargo_errors:
        errors.extend(embargo_errors[:20])

    summary = finalize_stats(stats)
    for split in ["train", "val", "test"]:
        if summary[split]["condition_coverage"] != sorted(seen):
            errors.append(f"{split} condition coverage mismatch: {summary[split]['condition_coverage']}")
        if summary[split]["label_dir_count"] != metadata["num_label_dirs"]:
            errors.append(f"{split} label_dir coverage mismatch: {summary[split]['label_dir_count']}")
    if summary["heldout"]["condition_coverage"] != sorted(heldout):
        errors.append(f"heldout condition coverage mismatch: {summary['heldout']['condition_coverage']}")
    if summary["heldout"]["label_dir_count"] != metadata["num_label_dirs"]:
        errors.append(f"heldout label_dir coverage mismatch: {summary['heldout']['label_dir_count']}")

    for split, item in summary.items():
        audit_rows.append({
            "split": split,
            "window_count": item["window_count"],
            "source_count": item["source_count"],
            "label_dir_count": item["label_dir_count"],
            "condition_coverage": ",".join(str(x) for x in item["condition_coverage"]),
            "normal_windows": item["normal_windows"],
            "fault_windows": item["fault_windows"],
            "normal_ratio": item["normal_ratio"],
        })

    result = {
        "pass": not errors,
        "errors": errors,
        "metadata_path": str(metadata_path.resolve()),
        "cache_path": str(cache_path),
        "cache_exists": cache_exists,
        "cache_size_match": cache_size_match,
        "cache_len": cache_len,
        "metadata_num_windows": metadata["num_windows"],
        "split_lengths": {split: len(indices) for split, indices in split_indices.items()},
        "disjoint": len(errors) == 0 or not any("overlap" in err for err in errors),
        "boundary_errors": len(boundary_errors),
        "embargo_errors": len(embargo_errors),
        "condition_errors": len(condition_errors),
        "summary": summary,
        "normalization_leakage_risk": (
            "Fit scaler, class weights, pos_weight, thresholds and priors on train split only. "
            "Heldout must not influence model selection."
        ),
        "condition_order_assumption": metadata.get("condition_order_assumption"),
    }

    write_json(results_dir / "condition_temporal_split_audit.json", result)
    write_csv(results_dir / "condition_temporal_split_audit.csv", audit_rows)

    lines = [
        "# Condition-Temporal Split Audit",
        "",
        f"- PASS: `{result['pass']}`",
        f"- cache length: `{cache_len}`",
        f"- metadata num_windows: `{metadata['num_windows']}`",
        f"- cache size match: `{cache_size_match}`",
        f"- split lengths: `{result['split_lengths']}`",
        f"- duplicate/overlap errors: `{[e for e in errors if 'overlap' in e or 'duplicate' in e]}`",
        f"- condition errors: `{len(condition_errors)}`",
        f"- boundary errors: `{len(boundary_errors)}`",
        f"- embargo errors: `{len(embargo_errors)}`",
        "",
        "## Split Summary",
        "",
        "| split | windows | sources | label_dirs | conditions | normal_ratio |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['split']} | {row['window_count']} | {row['source_count']} | "
            f"{row['label_dir_count']} | {row['condition_coverage']} | {float(row['normal_ratio']):.6f} |"
        )
    lines.extend([
        "",
        "## Leakage Risk Note",
        "",
        "The split audit validates index, condition and temporal boundary isolation. "
        "Training code must still ensure scaler/class-weight/threshold statistics are fit from train only.",
    ])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {err}" for err in errors[:50])
    write_text(docs_dir / "condition_temporal_split_audit.md", "\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--docs_dir", default="docs/split_audit")
    parser.add_argument("--results_dir", default="results/split_audit")
    args = parser.parse_args()
    result = audit(Path(args.metadata), Path(args.docs_dir), Path(args.results_dir))
    print(json.dumps({
        "pass": result["pass"],
        "split_lengths": result["split_lengths"],
        "boundary_errors": result["boundary_errors"],
        "embargo_errors": result["embargo_errors"],
        "condition_errors": result["condition_errors"],
    }, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
