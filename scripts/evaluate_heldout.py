#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_PHM
from models.KGLLMoE import Model


COMPONENTS = {"motor": (0, 5), "gear": (5, 14), "LA": (14, 19), "RA": (19, 21)}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def parse_conditions(value):
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_model(checkpoint_path, config_path, device):
    config = json.loads(Path(config_path).read_text())
    model_args = dict(config.get("args", config))
    model_args.setdefault("routing_variant", "correct")
    model_args.setdefault("routing_manifest", "")
    model_args.setdefault("expert_head_permutation", "none")
    model_args.setdefault("routing_prompt_mode", "default")
    model_args.setdefault("unified_expert_hidden", None)
    model = Model(SimpleNamespace(**model_args))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    incompatible = model.load_state_dict(state, strict=False)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    local_statistics = {
        name for name, _ in model.named_buffers()
        if name.startswith("moe_ffn.mean_") or name.startswith("moe_ffn.std_")
    }
    missing_trainable = sorted((trainable | local_statistics) - set(state))
    llm_keys = sorted(key for key in state if str(key).startswith("llm_model."))
    if missing_trainable:
        raise RuntimeError(f"Missing task-state checkpoint keys: {missing_trainable[:20]}")
    if llm_keys:
        raise RuntimeError(f"Unsafe checkpoint contains llm_model keys: {llm_keys[:5]}")
    model.to(device).eval()
    audit = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_size_mb": Path(checkpoint_path).stat().st_size / 1024 ** 2,
        "save_trainable_only": bool(isinstance(checkpoint, dict) and checkpoint.get("save_trainable_only")),
        "total_keys": len(state),
        "trainable_keys": len(set(state) & trainable),
        "llm_model_keys": 0,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }
    return model, model_args, audit


def build_loader(flag, model_args, split_metadata, batch_size):
    dataset = Dataset_PHM(
        root_path=model_args["root_path"], flag=flag, seq_len=int(model_args["seq_len"]),
        stride=int(model_args["dataset_window_stride"]),
        cache_dir=model_args.get("phm_cache_dir", "./cache_data"),
        split_metadata_path=split_metadata, allow_legacy_split=False, verbose=False,
    )
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


def infer(model, loader, device, use_amp, save_representations):
    branches = {"final": [], "global": [], "local": []}
    labels = []
    representations = {name: [] for name in COMPONENTS}
    meta = {"sample_ids": [], "source_ids": [], "source_files": [], "condition_ids": [],
            "window_starts": [], "window_indices": [], "label_dirs": []}
    field_map = {
        "sample_ids": "sample_id", "source_ids": "source_id", "source_files": "source_file",
        "condition_ids": "condition_id", "window_starts": "window_start",
        "window_indices": "window_index", "label_dirs": "label_dir",
    }
    with torch.no_grad():
        for batch in loader:
            x = batch["data"].float().to(device)
            with torch.amp.autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
                out = model(x, return_branch_outputs=True)
            branches["final"].append(out["final_logits"].float().cpu())
            branches["global"].append(out["global_logits"].float().cpu())
            branches["local"].append(out["local_logits"].float().cpu())
            labels.append(batch["label"].float().cpu())
            if save_representations:
                for name, rep in zip(COMPONENTS, out["component_representations"]):
                    representations[name].append(rep.float().cpu())
            for output_name, batch_name in field_map.items():
                values = batch[batch_name]
                if torch.is_tensor(values):
                    values = values.cpu().tolist()
                elif not isinstance(values, (list, tuple)):
                    values = [values]
                meta[output_name].extend(values)
    return (
        {key: torch.cat(value).numpy() for key, value in branches.items()},
        torch.cat(labels).numpy().astype(np.int64),
        {key: torch.cat(value).numpy() for key, value in representations.items() if value},
        {key: np.asarray(value) for key, value in meta.items()},
    )


def calibrate_thresholds(y_true, logits):
    probabilities = expit(logits)
    grid = np.arange(0.05, 0.951, 0.01)
    thresholds = []
    for label_index in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, label_index], probabilities[:, label_index] >= threshold,
                           zero_division=0) for threshold in grid]
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        choice = best[np.argmin(np.abs(grid[best] - 0.5))]
        thresholds.append(float(grid[choice]))
    return np.asarray(thresholds, dtype=np.float32)


def safe_auprc(y_true, y_prob):
    values = []
    for label_index in range(y_true.shape[1]):
        if len(np.unique(y_true[:, label_index])) > 1:
            values.append(average_precision_score(y_true[:, label_index], y_prob[:, label_index]))
    return float(np.mean(values)) if values else float("nan")


def metrics(y_true, logits, thresholds, rare_indices):
    probability = expit(logits)
    prediction = (probability >= thresholds.reshape(1, -1)).astype(np.int64)
    per_label = precision_recall_fscore_support(y_true, prediction, average=None, zero_division=0)
    component = {}
    for name, (start, end) in COMPONENTS.items():
        component[name] = float(f1_score(y_true[:, start:end], prediction[:, start:end],
                                         average="macro", zero_division=0))
    result = {
        "num_samples": int(len(y_true)),
        "EMR": float(np.mean(np.all(prediction == y_true, axis=1))),
        "label_wise_acc": float(np.mean(prediction == y_true)),
        "Macro_F1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "Micro_F1": float(f1_score(y_true, prediction, average="micro", zero_division=0)),
        "AUPRC": safe_auprc(y_true, probability),
        "rare_label_F1": float(np.mean(per_label[2][rare_indices])) if rare_indices else float("nan"),
        "per_component_Macro_F1": component,
        "component_balanced_F1": float(np.mean(list(component.values()))),
        "per_label_precision": per_label[0].tolist(),
        "per_label_recall": per_label[1].tolist(),
        "per_label_F1": per_label[2].tolist(),
        "per_label_support": per_label[3].astype(int).tolist(),
    }
    return result, probability, prediction


def load_rare_indices(checkpoint_path):
    checkpoint_dir = Path(checkpoint_path).parent
    report_path = checkpoint_dir / "focal_alpha_report.json"
    if not report_path.exists():
        return [], str(report_path), "missing"
    report = json.loads(report_path.read_text())
    indices = [int(value) for value in report.get("rare_label_indices_bottom25", [])]
    return indices, str(report_path), "training_split_bottom25"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-metadata", required=True)
    parser.add_argument("--eval-split-metadata", default="",
                        help="Optional few-shot metadata used only for target-test loading")
    parser.add_argument("--eval-flag", default="heldout", choices=["heldout", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiment-tag", default="heldout_evaluation")
    parser.add_argument("--variant", default="KG-LLMoE")
    parser.add_argument("--seen-conditions", default="1,2,3,4,5,6")
    parser.add_argument("--heldout-conditions", default="7,8,9")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seen_conditions = parse_conditions(args.seen_conditions)
    heldout_conditions = parse_conditions(args.heldout_conditions)
    if seen_conditions & heldout_conditions:
        raise ValueError("Seen and held-out condition sets must be disjoint")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_args, checkpoint_audit = load_model(args.checkpoint, args.config, device)
    eval_metadata = args.eval_split_metadata or args.split_metadata
    val_dataset, val_loader = build_loader("val", model_args, args.split_metadata, args.batch_size)
    heldout_dataset, heldout_loader = build_loader(
        args.eval_flag, model_args, eval_metadata, args.batch_size
    )
    val_branches, y_val, _, val_meta = infer(model, val_loader, device, model_args.get("use_amp", True), False)
    heldout_branches, y_heldout, representations, heldout_meta = infer(
        model, heldout_loader, device, model_args.get("use_amp", True), True
    )
    if set(np.asarray(val_meta["condition_ids"], dtype=int)) - seen_conditions:
        raise RuntimeError("Threshold calibration data contains held-out conditions")
    if set(np.asarray(heldout_meta["condition_ids"], dtype=int)) != heldout_conditions:
        raise RuntimeError("Held-out loader condition coverage mismatch")

    rare_indices, rare_source, rare_policy = load_rare_indices(args.checkpoint)
    all_metrics = {}
    thresholds_by_branch = {}
    probability_by_branch = {}
    prediction_by_branch = {}
    evaluated_branches = ["local", "final"] if model.local_only else ["local", "global", "final"]
    for branch in evaluated_branches:
        thresholds = calibrate_thresholds(y_val, val_branches[branch])
        branch_metrics, probability, prediction = metrics(
            y_heldout, heldout_branches[branch], thresholds, rare_indices
        )
        condition_metrics = {}
        condition_ids = np.asarray(heldout_meta["condition_ids"], dtype=int)
        for condition in sorted(heldout_conditions):
            mask = condition_ids == condition
            condition_metrics[str(condition)] = metrics(
                y_heldout[mask], heldout_branches[branch][mask], thresholds, rare_indices
            )[0]
        branch_metrics["per_condition"] = condition_metrics
        all_metrics[branch] = branch_metrics
        thresholds_by_branch[branch] = thresholds
        probability_by_branch[branch] = probability
        prediction_by_branch[branch] = prediction

    prediction_payload = {"y_true": y_heldout}
    for branch in evaluated_branches:
        prediction_payload[f"{branch}_logits"] = heldout_branches[branch]
        prediction_payload[f"{branch}_probability"] = probability_by_branch[branch]
        prediction_payload[f"{branch}_prediction"] = prediction_by_branch[branch]
        prediction_payload[f"{branch}_thresholds"] = thresholds_by_branch[branch]
    prediction_payload.update(heldout_meta)
    for component, values in representations.items():
        prediction_payload[f"{component}_representation"] = values
    np.savez_compressed(output_dir / "predictions.npz", **prediction_payload)

    split_hash = __import__("hashlib").sha256(Path(args.split_metadata).read_bytes()).hexdigest()
    report = {
        "status": "PASS", "experiment_tag": args.experiment_tag, "variant": args.variant, "seed": args.seed,
        "routing_variant": model_args.get("routing_variant", "correct"),
        "routing_manifest": model_args.get("routing_manifest", ""),
        "split_manifest": str(Path(args.split_metadata).resolve()), "split_manifest_sha256": split_hash,
        "evaluation_split_manifest": str(Path(eval_metadata).resolve()),
        "evaluation_split_manifest_sha256": __import__("hashlib").sha256(Path(eval_metadata).read_bytes()).hexdigest(),
        "calibration": {
            "method": "per_label_max_F1_grid_0.05_to_0.95_step_0.01_tie_nearest_0.5",
            "split": "seen_validation_conditions", "num_samples": len(y_val),
            "seen_conditions": sorted(seen_conditions),
            "heldout_used": False,
            "thresholds": {key: value.tolist() for key, value in thresholds_by_branch.items()},
        },
        "rare_label": {"indices": rare_indices, "source": rare_source, "policy": rare_policy},
        "checkpoint_audit": checkpoint_audit,
        "sample_audit": {
            "val": len(y_val),
            "heldout": len(y_heldout),
            "heldout_conditions": sorted(heldout_conditions),
        },
        "metrics": all_metrics,
        "global_branch_present": not model.local_only,
    }
    write_json(output_dir / "metrics.json", report)
    write_json(output_dir / "checkpoint_key_report.json", checkpoint_audit)
    write_json(output_dir / "threshold_calibration.json", report["calibration"])
    (output_dir / "EVAL_COMPLETE").write_text("PASS\n")
    print(json.dumps({
        "status": "PASS", "variant": args.variant, "seed": args.seed,
        "final_Macro_F1": all_metrics["final"]["Macro_F1"],
        "local_component_balanced_F1": all_metrics["local"]["component_balanced_F1"],
    }, indent=2))


if __name__ == "__main__":
    main()
