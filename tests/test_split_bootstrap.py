import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import SequentialSampler

from data_provider.data_factory import data_provider
from data_provider.phm_cache import component_labels, prepare_phm_cache


ROOT = Path(__file__).resolve().parents[1]


def test_split_bootstraps_cache_and_keeps_every_evaluation_sample(tmp_path):
    data = tmp_path / "data" / "M0_G0_LA0_RA0"
    data.mkdir(parents=True)
    for condition in range(1, 10):
        values = np.random.default_rng(condition).standard_normal((512, 21))
        pd.DataFrame(values).to_csv(data / f"Sample_{condition}.csv", index=False)
    cache = tmp_path / "cache"
    split = tmp_path / "split"
    command = [
        sys.executable,
        str(ROOT / "scripts/split/build_condition_temporal_split.py"),
        "--root_path",
        str(data.parent),
        "--seq_len",
        "8",
        "--window_stride",
        "16",
        "--embargo",
        "8",
        "--cache_dir",
        str(cache),
        "--output_dir",
        str(split),
        "--docs_dir",
        str(tmp_path / "docs"),
        "--results_dir",
        str(tmp_path / "reports"),
        "--write",
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    metadata_path = split / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert Path(metadata["cache_path"]).exists()
    result_dir = tmp_path / "audit_results"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit/audit_condition_temporal_split.py"),
            "--metadata",
            str(metadata_path),
            "--docs_dir",
            str(tmp_path / "audit_docs"),
            "--results_dir",
            str(result_dir),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((result_dir / "condition_temporal_split_audit.json").read_text())[
        "pass"
    ]
    args = SimpleNamespace(
        data="phm",
        embed="timeF",
        percent=100,
        batch_size=16,
        eval_batch_size=5,
        freq="h",
        seq_len=8,
        root_path=str(data.parent),
        dataset_window_stride=16,
        phm_cache_dir=str(cache),
        phm_split_metadata=str(metadata_path),
        num_workers=0,
    )
    for flag in ["val", "test", "heldout"]:
        dataset, loader = data_provider(args, flag)
        assert not loader.drop_last
        assert isinstance(loader.sampler, SequentialSampler)
        assert sum(len(batch["data"]) for batch in loader) == len(dataset)


@pytest.mark.parametrize("backbone", ["none", "transformer_trainable_lite"])
def test_cpu_training_and_heldout_cli_without_amp(tmp_path, backbone):
    data = tmp_path / "data" / "M0_G0_LA0_RA0"
    data.mkdir(parents=True)
    for condition in range(1, 10):
        values = np.random.default_rng(condition).standard_normal((4096, 21))
        pd.DataFrame(values).to_csv(data / f"Sample_{condition}.csv", index=False)
    cache, split, checkpoints = (
        tmp_path / name for name in ["cache", "split", "checkpoints"]
    )
    env = dict(
        __import__("os").environ,
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    def run(arguments):
        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            completed.stdout[-2000:] + completed.stderr[-4000:]
        )

    run(
        [
            str(ROOT / "scripts/split/build_condition_temporal_split.py"),
            "--root_path",
            str(data.parent),
            "--seq_len",
            "64",
            "--window_stride",
            "128",
            "--embargo",
            "8",
            "--cache_dir",
            str(cache),
            "--output_dir",
            str(split),
            "--docs_dir",
            str(tmp_path / "docs"),
            "--results_dir",
            str(tmp_path / "reports"),
            "--write",
        ]
    )
    run(
        [
            str(ROOT / "run_main.py"),
            "--task_name",
            "classification",
            "--is_training",
            "1",
            "--model_id",
            "synthetic",
            "--model_comment",
            "synthetic",
            "--model",
            "KGLLMoE",
            "--data",
            "phm",
            "--root_path",
            str(data.parent),
            "--phm_cache_dir",
            str(cache),
            "--phm_split_metadata",
            str(split / "metadata.json"),
            "--seq_len",
            "64",
            "--patch_len",
            "16",
            "--dataset_window_stride",
            "128",
            "--global_backbone_mode",
            backbone,
            "--d_model",
            "8",
            "--llm_dim",
            "16",
            "--d_ff",
            "8",
            "--n_heads",
            "2",
            "--transformer_d_model",
            "16",
            "--transformer_nhead",
            "2",
            "--transformer_num_layers",
            "1",
            "--transformer_dim_feedforward",
            "32",
            "--transformer_dropout",
            "0",
            "--prompt_mode",
            "none",
            "--reprogramming_mode",
            "linear",
            "--local_descriptor_mode",
            "revised_rms",
            "--local_warmup_epochs",
            "0" if backbone == "none" else "5",
            "--train_epochs",
            "1",
            "--batch_size",
            "1",
            "--eval_batch_size",
            "5",
            "--num_workers",
            "0",
            "--loss_type",
            "dfl_detach",
            "--lradj",
            "COS",
            "--checkpoints",
            str(checkpoints),
            "--eval_metrics_extended",
            "--save_predictions",
            "--save_branch_logits",
            "--save_sample_metadata",
        ]
    )
    run_dir = next(path.parent for path in checkpoints.rglob("config.json"))
    config = json.loads((run_dir / "config.json").read_text())
    args = config.get("args", config)
    assert args["max_grad_norm"] == args["logit_clip"] == args["loss_logit_clip"] == 0
    run(
        [
            str(ROOT / "scripts/evaluate_heldout.py"),
            "--checkpoint",
            str(run_dir / "checkpoint.pt"),
            "--config",
            str(run_dir / "config.json"),
            "--split-metadata",
            str(split / "metadata.json"),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--batch-size",
            "5",
            "--variant",
            "local",
            "--seed",
            str(args["seed"]),
        ]
    )
    result = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert result["status"] == "PASS"
    assert result["global_branch_present"] == (backbone != "none")
    assert set(result["metrics"]) == (
        {"final", "local"} if backbone == "none" else {"final", "global", "local"}
    )
    if backbone != "none":
        assert (
            len(json.loads((run_dir / "local_warmup.json").read_text())["history"]) == 5
        )


@pytest.mark.parametrize("name", ["M5_G0_LA0_RA0", "M0_G0_LA0_RA2", "unknown"])
def test_invalid_component_labels(name):
    with pytest.raises(ValueError):
        component_labels(name)
