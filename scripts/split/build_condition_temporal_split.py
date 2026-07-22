#!/usr/bin/env python3
"""Build a condition-aware temporal PHM split.

This script is intentionally independent from Dataset_PHM. It reads the
existing PHM cache/window pointers and writes only a new split protocol under
cache_data/splits_condition_temporal_v1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


LABEL_NAMES = [
    "M0", "M1", "M2", "M3", "M4",
    "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
    "LA0", "LA1", "LA2", "LA3", "LA4",
    "RA0", "RA1",
]

LEGACY_METADATA = Path(
    "cache_data/splits_v3/"
    "phm_split_v3_root8f1f61ff4a_seq512_wstride8_mode_file_seed2026_metadata.json"
)


def short_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def parse_label(label_dir: str) -> List[int]:
    label = [0] * 21
    for comp in label_dir.split("_"):
        for sub in comp.split("+"):
            if re.fullmatch(r"M\d+", sub):
                label[int(sub[1:])] = 1
            elif re.fullmatch(r"G\d+", sub):
                label[5 + int(sub[1:])] = 1
            elif re.fullmatch(r"LA\d+", sub):
                label[14 + int(sub[2:])] = 1
            elif re.fullmatch(r"RA\d+", sub):
                label[19 + int(sub[2:])] = 1
    return label


def parse_condition_id(csv_name: str, method: str) -> Tuple[int | None, str, str]:
    if method != "filename_number":
        raise ValueError(f"Unsupported condition_parse={method}")
    stem = Path(csv_name).stem
    match = re.search(r"(?:sample|condition|cond|case|run)[_-]?(\d+)", stem, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)$", stem)
    if not match:
        return None, "unparsed", "none"
    return int(match.group(1)), "filename_number", "high"


def count_rows_and_columns(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.readline()
        cols = len(header.decode("utf-8", errors="replace").rstrip("\r\n").split(","))
        rows = 0
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            rows += chunk.count(b"\n")
    return rows, cols


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


def in_window_region(start: int, seq_len: int, region: Tuple[int, int]) -> bool:
    left, right = region
    return start >= left and start + seq_len <= right


def build_source_inventory(root: Path, condition_parse: str, seen: set[int], heldout: set[int]) -> Tuple[List[dict], List[dict]]:
    all_csv = sorted(root.rglob("*.csv"))
    by_label: Dict[str, List[Path]] = defaultdict(list)
    for path in all_csv:
        by_label[path.parent.name].append(path)

    confirmation_rows: List[dict] = []
    source_records: List[dict] = []
    failures: List[str] = []
    file_index = 0
    for label_dir in sorted(by_label):
        paths = sorted(by_label[label_dir])
        if len(paths) != 9:
            failures.append(f"{label_dir}: expected 9 CSV files, found {len(paths)}")
        for path in paths:
            condition_id, method, confidence = parse_condition_id(path.name, condition_parse)
            if condition_id is None:
                failures.append(f"{path}: cannot parse condition id")
            rows, cols = count_rows_and_columns(path)
            row = {
                "label_dir": label_dir,
                "condition_id": condition_id if condition_id is not None else "",
                "csv_name": path.name,
                "csv_path": str(path.resolve()),
                "parse_method": method,
                "parse_confidence": confidence,
                "is_seen_condition": bool(condition_id in seen) if condition_id is not None else False,
                "is_heldout_condition": bool(condition_id in heldout) if condition_id is not None else False,
                "num_rows": rows,
                "num_columns": cols,
            }
            confirmation_rows.append(row)
            source_records.append({
                **row,
                "file_index": file_index,
                "label_vector": parse_label(label_dir),
                "num_windows": max(0, (rows - args_global.seq_len) // args_global.window_stride + 1)
                if rows >= args_global.seq_len else 0,
            })
            file_index += 1

    if failures:
        raise RuntimeError("; ".join(failures))
    return confirmation_rows, source_records


def update_stats(stats: dict, split: str, record: dict, index_count: int) -> None:
    if index_count <= 0:
        return
    stats[split]["window_counts"] += index_count
    stats[split]["source_files"].add(record["csv_path"])
    stats[split]["label_dirs"].add(record["label_dir"])
    stats[split]["conditions"].add(record["condition_id"])
    stats[split]["condition_counts"][str(record["condition_id"])] += index_count
    stats[split]["label_dir_counts"][record["label_dir"]] += index_count
    for name, value in zip(LABEL_NAMES, record["label_vector"]):
        if value:
            stats[split]["label_positive_counts"][name] += index_count
    if record["label_dir"] == "M0_G0_LA0_RA0":
        stats[split]["normal_windows"] += index_count
    else:
        stats[split]["fault_windows"] += index_count


def finalize_stats(stats: dict) -> dict:
    final = {}
    for split, data in stats.items():
        total = data["window_counts"]
        final[split] = {
            "window_count": total,
            "source_count": len(data["source_files"]),
            "label_dir_count": len(data["label_dirs"]),
            "condition_coverage": sorted(data["conditions"]),
            "condition_counts": dict(sorted(data["condition_counts"].items(), key=lambda kv: int(kv[0]))),
            "label_dir_counts": dict(sorted(data["label_dir_counts"].items())),
            "label_positive_counts": dict(data["label_positive_counts"]),
            "normal_windows": data["normal_windows"],
            "fault_windows": data["fault_windows"],
            "normal_ratio": (data["normal_windows"] / total) if total else 0.0,
            "fault_ratio": (data["fault_windows"] / total) if total else 0.0,
        }
    return final


def make_stats() -> dict:
    return {
        split: {
            "window_counts": 0,
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


def write_condition_confirmation(docs_dir: Path, results_dir: Path, rows: List[dict]) -> None:
    fields = [
        "label_dir", "condition_id", "csv_name", "parse_method", "parse_confidence",
        "is_seen_condition", "is_heldout_condition", "num_rows", "num_columns", "csv_path",
    ]
    write_csv(results_dir / "condition_order_confirmation.csv", rows, fields)
    label_dirs = sorted({row["label_dir"] for row in rows})
    bad_dirs = [label for label in label_dirs if sum(1 for row in rows if row["label_dir"] == label) != 9]
    unparsed = [row for row in rows if row["parse_method"] == "unparsed"]
    lines = [
        "# Condition Order Confirmation",
        "",
        "This confirmation assumes `Sample_1.csv` to `Sample_9.csv` correspond to true condition 1 to 9.",
        "This condition mapping must be confirmed before split generation.",
        "If this assumption is later found false, the generated split must not be used.",
        "",
        f"- label directories: `{len(label_dirs)}`",
        f"- CSV files: `{len(rows)}`",
        f"- directories not containing 9 CSV files: `{len(bad_dirs)}`",
        f"- unparsed condition files: `{len(unparsed)}`",
        f"- parse method: `filename_number`",
        "",
        "## First 30 Rows",
        "",
        "| label_dir | condition_id | csv_name | seen | heldout |",
        "|---|---:|---|---:|---:|",
    ]
    for row in rows[:30]:
        lines.append(
            f"| {row['label_dir']} | {row['condition_id']} | {row['csv_name']} | "
            f"{row['is_seen_condition']} | {row['is_heldout_condition']} |"
        )
    write_text(docs_dir / "condition_order_confirmation.md", "\n".join(lines) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_file_prefix(args: argparse.Namespace, root_hash: str) -> str:
    seen = f"{min(args.seen_conditions)}-{max(args.seen_conditions)}"
    heldout = f"{min(args.heldout_conditions)}-{max(args.heldout_conditions)}"
    return (
        f"phm_split_{args.version}_root{root_hash}_seq{args.seq_len}_wstride{args.window_stride}_"
        f"seen{seen}_holdout{heldout}_embargo{args.embargo}"
    )


def save_split_files(output_dir: Path, prefix: str, split_indices: dict) -> dict:
    split_files = {}
    for split in ["train", "val", "test", "heldout"]:
        path = (output_dir / f"{prefix}_{split}.pt").resolve()
        torch.save(split_indices[split], path)
        split_files[split] = str(path)
    return split_files


def write_reports(docs_dir: Path, metadata: dict, dry_run: bool, legacy_exists: bool) -> None:
    audit = metadata["audit"]
    lines = [
        "# Condition-Temporal Split Build Report",
        "",
        "## 1. Purpose And Scope",
        "",
        "This utility builds a condition-aware temporal split for PHM. It does not start training, change the model structure, or overwrite the reference file-level split.",
        "",
        "## 2. Condition Confirmation",
        "",
        f"- label directories: `{metadata['num_label_dirs']}`",
        "- CSV files per label directory: `9`",
        "- condition mapping: `filename_number`",
        "- manual assumption: `Sample_1` to `Sample_9` correspond to condition 1 to 9",
        "",
        "## 3. Protocol",
        "",
        f"- seen conditions: `{metadata['seen_conditions']}`",
        f"- heldout conditions: `{metadata['heldout_conditions']}`",
        f"- train/val/test ratios: `{metadata['train_ratio']}/{metadata['val_ratio']}/{metadata['test_ratio']}`",
        f"- embargo: `{metadata['embargo']}`",
        f"- seq_len/window_stride: `{metadata['seq_len']}/{metadata['window_stride']}`",
        "",
        "## 4. Split Files",
        "",
    ]
    for split, path in metadata["split_files"].items():
        lines.append(f"- {split}: `{path}`")
    lines.extend([
        "",
        "## 5. Counts",
        "",
        "| split | windows | sources | label_dirs | conditions | normal_ratio |",
        "|---|---:|---:|---:|---|---:|",
    ])
    for split in ["train", "val", "test", "heldout"]:
        item = audit[split]
        lines.append(
            f"| {split} | {item['window_count']} | {item['source_count']} | "
            f"{item['label_dir_count']} | {item['condition_coverage']} | {item['normal_ratio']:.6f} |"
        )
    lines.extend([
        "",
        "## 6. Leakage Controls",
        "",
        "- Train/val/test windows are fully contained in their assigned temporal regions.",
        "- Embargo gaps are excluded around train/val and val/test boundaries.",
        "- Conditions 7-9 are excluded from train/val/test and saved only in heldout.",
        "- Heldout must not be used for training, early stopping, threshold calibration, class weights, pos_weight, or normalization fitting.",
        "",
        "## 7. Relation To The File-Level Split",
        "",
        f"- Existing file-level metadata exists: `{legacy_exists}`",
        "- The file-level split remains the baseline protocol and was not overwritten.",
        "- The condition-temporal split serves a different experimental goal: seen-condition temporal evaluation plus later held-out-condition evaluation.",
        "",
        "## 8. Recommendation",
        "",
        "- This split can be used in the next M0 full training only if the manual condition order assumption remains valid.",
        "- Use explicit `--phm_split_metadata` and do not enable debug, balanced subset, or legacy split flags.",
        "",
        f"Dry run only: `{dry_run}`",
    ])
    write_text(docs_dir / "condition_temporal_split_build_report.md", "\n".join(lines) + "\n")

    decision = [
        "# Condition-Temporal Split Decision Note",
        "",
        f"- Formal split build dry_run: `{dry_run}`.",
        f"- Split files written: `{not dry_run}`.",
        f"- Audit expected train/val/test/heldout windows: `{ {k: audit[k]['window_count'] for k in ['train', 'val', 'test', 'heldout']} }`.",
        "- Condition 1-6 are assigned only to train/val/test.",
        "- Condition 7-9 are assigned only to heldout.",
        "- No train/val/test temporal boundary crossing is allowed by construction.",
        f"- Existing file-level split metadata still exists: `{legacy_exists}`.",
        "- Start training only with explicit condition-temporal metadata and after validating the condition mapping.",
    ]
    write_text(docs_dir / "condition_temporal_split_decision_note.md", "\n".join(decision) + "\n")

    comparison = [
        "# File-Level vs Condition-Temporal Protocol Comparison",
        "",
        "## File-Level Split",
        "",
        "- Kept as the baseline protocol.",
        "- Advantage: source-disjoint train/val/test CSVs.",
        "- Limitation: all condition IDs can be mixed across train/val/test, so it does not reserve condition shift for a separate evaluation.",
        "",
        "## Condition-Temporal Split",
        "",
        "- Main protocol: seen conditions 1-6 split by time inside each CSV.",
        "- Cross-condition protocol: conditions 7-9 are held out for later evaluation.",
        "- Advantage: separates seen-condition learning from held-out-condition validation.",
        "- Limitation: train/val/test come from different temporal segments of the same seen CSVs, so it depends on strict boundary containment, embargo, and train-only statistics.",
        "",
        "These protocols serve different experimental goals. This report does not claim one is universally better than the other.",
    ]
    write_text(docs_dir / "file_level_vs_condition_temporal_protocol_comparison.md", "\n".join(comparison) + "\n")

    howto = [
        "# How To Use The Condition-Temporal Split",
        "",
        "Use the generated metadata explicitly in later training:",
        "",
        "```bash",
        "python run_main.py \\",
        f"  --phm_split_metadata {metadata['metadata_path']} \\",
        "  --root_path ./dataset/Training_new \\",
        "  --save_trainable_only \\",
        "  --use_amp",
        "```",
        "",
        "Formal training must not include:",
        "",
        "- `--debug`",
        "- `--debug_iters`",
        "- `--phm_balanced_subset_metadata`",
        "- `--allow_legacy_phm_split`",
        "",
        "Heldout condition split is for later cross-condition evaluation only. It must not be used for training, early stopping, threshold calibration, class weight or pos_weight calculation, or normalization fitting.",
    ]
    write_text(docs_dir / "how_to_use_condition_temporal_split.md", "\n".join(howto) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./dataset/Training_new")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--seen_conditions", default="1,2,3,4,5,6")
    parser.add_argument("--heldout_conditions", default="7,8,9")
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--embargo", type=int, default=1024)
    parser.add_argument("--condition_parse", default="filename_number")
    parser.add_argument("--version", default="condition_temporal_v1")
    parser.add_argument("--cache_dir", default="./cache_data")
    parser.add_argument("--output_dir", default="cache_data/splits_condition_temporal_v1")
    parser.add_argument("--docs_dir", default="docs/condition_temporal_split")
    parser.add_argument("--results_dir", default="results/condition_temporal_split")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.dry_run and args.write:
        raise SystemExit("Use only one of --dry_run or --write.")
    if not args.dry_run and not args.write:
        args.dry_run = True
    args.seen_conditions = parse_int_list(args.seen_conditions)
    args.heldout_conditions = parse_int_list(args.heldout_conditions)
    seen = set(args.seen_conditions)
    heldout = set(args.heldout_conditions)
    if seen & heldout:
        raise SystemExit("seen_conditions and heldout_conditions overlap.")

    global args_global
    args_global = args

    repo = Path.cwd()
    root = Path(args.root_path).resolve()
    cache_path = (Path(args.cache_dir) / f"phm_cache_optimized_v2_seq{args.seq_len}_stride{args.window_stride}.pt").resolve()
    output_dir = Path(args.output_dir).resolve()
    docs_dir = Path(args.docs_dir)
    results_dir = Path(args.results_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.write:
        output_dir.mkdir(parents=True, exist_ok=True)

    confirmation_rows, source_records = build_source_inventory(root, args.condition_parse, seen, heldout)
    write_condition_confirmation(docs_dir, results_dir, confirmation_rows)

    by_file_index = {row["file_index"]: row for row in source_records}
    if sorted(by_file_index) != list(range(len(source_records))):
        raise RuntimeError("Source records are not contiguous by file_index.")
    bad_cols = [row for row in source_records if row["num_columns"] != 21]
    if bad_cols:
        raise RuntimeError(f"Found non-21-column CSV files: {bad_cols[:3]}")
    condition_values = sorted({row["condition_id"] for row in source_records})
    expected_conditions = sorted(seen | heldout)
    if condition_values != expected_conditions:
        raise RuntimeError(f"Unexpected condition coverage: {condition_values} != {expected_conditions}")

    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    window_pointers = cache["window_pointers"]
    if len(cache["all_file_data"]) != len(source_records):
        raise RuntimeError(f"Cache file count mismatch: {len(cache['all_file_data'])} != {len(source_records)}")

    split_indices = {split: [] for split in ["train", "val", "test", "heldout"]}
    stats = make_stats()
    boundary_rows = []
    for real_idx, pointer in enumerate(window_pointers):
        file_idx, start, label_dir = pointer
        record = by_file_index[file_idx]
        if label_dir != record["label_dir"]:
            raise RuntimeError(f"Cache label mismatch at index {real_idx}: {label_dir} != {record['label_dir']}")
        condition_id = record["condition_id"]
        if condition_id in heldout:
            split_indices["heldout"].append(real_idx)
            update_stats(stats, "heldout", record, 1)
            continue
        if condition_id not in seen:
            continue
        base = segment_bounds(record["num_rows"], args.train_ratio, args.val_ratio, args.test_ratio)
        assigned = None
        for split in ["train", "val", "test"]:
            if in_window_region(start, args.seq_len, usable_bounds(base, split, args.embargo)):
                assigned = split
                break
        if assigned:
            split_indices[assigned].append(real_idx)
            update_stats(stats, assigned, record, 1)

    for record in source_records:
        base = segment_bounds(record["num_rows"], args.train_ratio, args.val_ratio, args.test_ratio)
        row = {
            "file_index": record["file_index"],
            "label_dir": record["label_dir"],
            "condition_id": record["condition_id"],
            "csv_name": record["csv_name"],
            "num_rows": record["num_rows"],
        }
        for split in ["train", "val", "test"]:
            raw_start, raw_end = base[split]
            usable_start, usable_end = usable_bounds(base, split, args.embargo)
            row[f"{split}_raw_start"] = raw_start
            row[f"{split}_raw_end"] = raw_end
            row[f"{split}_usable_start"] = usable_start
            row[f"{split}_usable_end"] = usable_end
        boundary_rows.append(row)
    write_csv(results_dir / "condition_temporal_source_boundaries.csv", boundary_rows)

    root_hash = short_hash(str(root))
    prefix = split_file_prefix(args, root_hash)
    split_files = {
        split: str((output_dir / f"{prefix}_{split}.pt").resolve())
        for split in ["train", "val", "test", "heldout"]
    }
    if args.write:
        split_files = save_split_files(output_dir, prefix, split_indices)

    metadata_path = str((output_dir / f"{prefix}_metadata.json").resolve())
    audit = finalize_stats(stats)
    legacy_exists = LEGACY_METADATA.exists()
    metadata = {
        "version": args.version,
        "split_mode": "condition_temporal",
        "official_root_path": args.root_path,
        "resolved_root_path": str(root),
        "root_path": str(root),
        "root_hash": root_hash,
        "cache_path": str(cache_path),
        "cache_file_size": cache_path.stat().st_size,
        "seq_len": args.seq_len,
        "window_stride": args.window_stride,
        "dataset_window_stride": args.window_stride,
        "model_patch_stride_recommended": 8,
        "num_windows": len(window_pointers),
        "num_source_files": len(source_records),
        "num_label_dirs": len({row["label_dir"] for row in source_records}),
        "condition_parse_method": args.condition_parse,
        "condition_order_assumption": "User confirmed Sample_1..Sample_9 correspond to condition_1..condition_9.",
        "seen_conditions": args.seen_conditions,
        "heldout_conditions": args.heldout_conditions,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "embargo": args.embargo,
        "split_files": split_files,
        "metadata_path": metadata_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str((repo / "scripts/split/build_condition_temporal_split.py").resolve()),
        "git_hash": git_hash(repo),
        "source_records": source_records,
        "audit": audit,
        "audit_summary": {
            "condition_order_confirmed": True,
            "csv_files_per_label_dir": 9,
            "all_conditions_parsed": True,
            "legacy_file_level_split_metadata_exists": legacy_exists,
            "heldout_usage_rule": "Heldout split must not be used for training, early stopping, threshold calibration, class weights, pos_weight, or normalization fitting.",
        },
        "dry_run": bool(args.dry_run),
    }
    write_json(results_dir / "condition_temporal_split_metadata_preview.json", metadata)
    if args.write:
        write_json(Path(metadata_path), metadata)
    write_reports(docs_dir, metadata, dry_run=bool(args.dry_run), legacy_exists=legacy_exists)

    print(json.dumps({
        "dry_run": bool(args.dry_run),
        "metadata_path": metadata_path if args.write else None,
        "window_counts": {k: audit[k]["window_count"] for k in ["train", "val", "test", "heldout"]},
        "source_counts": {k: audit[k]["source_count"] for k in ["train", "val", "test", "heldout"]},
        "label_dir_counts": {k: audit[k]["label_dir_count"] for k in ["train", "val", "test", "heldout"]},
        "condition_coverage": {k: audit[k]["condition_coverage"] for k in ["train", "val", "test", "heldout"]},
        "legacy_metadata_exists": legacy_exists,
    }, indent=2))


if __name__ == "__main__":
    main()
