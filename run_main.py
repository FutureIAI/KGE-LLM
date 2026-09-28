import argparse
import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import KGLLMoE
from data_provider.data_factory import data_provider
from utils.local_warmup import fit_local_descriptors, warm_up_local

import time
import random
import numpy as np
import os
import logging
import json
import csv
import hashlib
import subprocess


from utils.losses import (
    FocalLoss,
    DynamicFocalLoss,
    PlainBCEWithLogitsLoss,
    ClassBalancedFocalLoss,
    AsymmetricLoss,
)
from utils.tools import (
    del_files,
    EarlyStopping,
    adjust_learning_rate,
    vali_classify,
    load_content,
    test_classify,
    evaluate_metrics,
    evaluate_metrics_extended,
    plot_co_occurrence_matrix,
    plot_convergence_curves,
)

# 初始化日志系统
log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='KG-LLMoE industrial multilabel classification')
# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, choices=['KGLLMoE'],
                    help='Released model entry point')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--phm_cache_dir', type=str, default='./cache_data',
                    help='Dataset_PHM cache directory')
parser.add_argument('--phm_split_metadata', type=str, default='',
                    help='Root/stride-aware Dataset_PHM split metadata json')
parser.add_argument('--phm_balanced_subset_metadata', type=str, default='',
                    help='Optional balanced debug subset metadata json')
parser.add_argument('--allow_legacy_phm_split', action='store_true', default=False,
                    help='Allow legacy phm_split_*.pt loading. Disabled by default to prevent split/cache mismatch.')

# # forecasting task
# parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
# parser.add_argument('--label_len', type=int, default=48, help='start token length')
# parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
# parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# classification task
parser.add_argument('--num_classes', type=int, default=21)  # 类别数量，建议根据你的文件夹来设定
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len', type=int, default=0)  # 分类不需要预测长度
parser.add_argument('--debug', action='store_true', help='Enable debug mode to limit number of iters per epoch')
parser.add_argument('--debug_iters', type=int, default=1000, help='Maximum iters per epoch in debug mode')


# model define
parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
parser.add_argument('--c_out', type=int, default=21, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=256, help='stride')
parser.add_argument('--dataset_window_stride', type=int, default=None,
                    help='Dataset_PHM window stride. If unset, defaults to --stride for backward compatibility.')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--llm_backbone', type=str, default='llama', choices=['llama', 'mistral'],
                    help='Frozen pretrained decoder family. Default preserves the formal LLaMA path.')
parser.add_argument('--llm_model_path', type=str, default='', help='Pinned local pretrained model directory')
parser.add_argument('--llm_repo_id', type=str, default='', help='Audited upstream model repository id')
parser.add_argument('--llm_revision', type=str, default='', help='Audited immutable upstream revision')
parser.add_argument('--llm_active_layers', type=int, default=0,
                    help='Number of active decoder layers; 0 inherits --llm_layers')
parser.add_argument('--llm_readout', type=str, default='',
                    choices=['', 'first_token_legacy', 'first_patch', 'mean_patch', 'mean_all'],
                    help='Optional alias overriding --readout_mode')
parser.add_argument('--llm_final_norm', action=argparse.BooleanOptionalAction, default=None,
                    help='Apply final decoder RMSNorm. Default is unchanged LLaMA behavior and enabled for Mistral.')
parser.add_argument('--llm_attention_implementation', type=str, default='sdpa', choices=['eager', 'sdpa'],
                    help='Transformers attention implementation for the frozen decoder')
parser.add_argument('--content', type=str, default='KG-LLMoE run', help='description for the model run')
parser.add_argument('--global_backbone_mode', type=str, default='llama',
                    choices=[
                        'llama', 'none',
                        'transformer_trainable', 'transformer_frozen',
                        'transformer_trainable_lite', 'transformer_frozen_lite',
                        'llama_random_frozen',
                    ],
                    help='LLM/global backbone ablation mode')
parser.add_argument('--prompt_mode', type=str, default='full',
                    choices=['full', 'none', 'static', 'dynamic', 'random', 'shuffled'],
                    help='prompt ablation mode')
parser.add_argument('--reprogramming_mode', type=str, default='full',
                    choices=['full', 'linear', 'mlp', 'frozen_random'],
                    help='time-series-to-backbone reprogramming ablation mode')
parser.add_argument('--readout_mode', type=str, default='first_token_legacy',
                    choices=['first_token_legacy', 'first_patch', 'mean_patch', 'mean_all'],
                    help='classification readout token/pooling mode')
parser.add_argument('--transformer_d_model', type=int, default=512,
                    help='Hidden size for capacity-controlled lite Transformer backbone modes')
parser.add_argument('--transformer_nhead', type=int, default=8,
                    help='Attention heads for lite Transformer backbone modes')
parser.add_argument('--transformer_num_layers', type=int, default=4,
                    help='Number of layers for lite Transformer backbone modes')
parser.add_argument('--transformer_dim_feedforward', type=int, default=2048,
                    help='Feedforward width for lite Transformer backbone modes')
parser.add_argument('--transformer_dropout', type=float, default=0.1,
                    help='Dropout for lite Transformer backbone modes')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Micro-batch accumulation used only to preserve effective batch size after OOM fallback')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--loss_type', type=str, default='dfl_detach',
                    choices=[
                        'bce', 'weighted_bce', 'focal', 'cb_focal', 'asl',
                        'dfl_detach', 'dfl_full', 'dfl_static_low', 'dfl_static_high',
                    ],
                    help='classification loss variant for PHM controlled screening')
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='gamma for fixed focal and class-balanced focal losses')
parser.add_argument('--cb_beta', type=float, default=0.999,
                    help='effective-number beta for class-balanced focal loss')
parser.add_argument('--asl_gamma_pos', type=float, default=0.0,
                    help='ASL positive focusing gamma')
parser.add_argument('--asl_gamma_neg', type=float, default=4.0,
                    help='ASL negative focusing gamma')
parser.add_argument('--asl_clip', type=float, default=0.05,
                    help='ASL negative probability clipping')
parser.add_argument('--dfl_gamma_base', type=float, default=1.0,
                    help='DynamicFocalLoss base gamma')
parser.add_argument('--dfl_gamma_scale', type=float, default=2.0,
                    help='DynamicFocalLoss dynamic gamma scale')
parser.add_argument('--dfl_detach_gamma', action='store_true', default=False,
                    help='Force DynamicFocalLoss gamma(p_t) detach/no_grad behavior')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save_trainable_only', action=argparse.BooleanOptionalAction, default=True,
                    help='Save only requires_grad parameters. Enabled by default; full frozen-LLM saves are rejected.')
parser.add_argument('--load_trainable_checkpoint', type=str, default='',
                    help='Optional trainable-only checkpoint to load before training/evaluation')
parser.add_argument('--finetune_from_checkpoint', type=str, default='',
                    help='Alias for --load_trainable_checkpoint used by few-shot adaptation scripts')
parser.add_argument('--save_predictions', action='store_true', default=False,
                    help='save y_true/y_prob/y_pred/logits/sample_ids to predictions.npz')
parser.add_argument('--eval_metrics_extended', action='store_true', default=False,
                    help='compute extended multilabel metrics and write metrics.json')
parser.add_argument('--experiment_tag', type=str, default='', help='free-form experiment tag written to config.json')
parser.add_argument('--profile_inference', action='store_true', default=False,
                    help='profile final test inference latency and write latency.json')
parser.add_argument('--resource_profile_dir', type=str, default='',
                    help='Optional directory for epoch-level resource profiling json/csv outputs')
parser.add_argument('--routing_variant', type=str, default='correct',
                    choices=['correct', 'matched_random', 'permuted_alignment', 'unified'],
                    help='Component-topology routing variant')
parser.add_argument('--routing_manifest', type=str, default='',
                    help='Fixed routing manifest used by matched_random')
parser.add_argument('--expert_head_permutation', type=str, default='none',
                    choices=['none', 'cyclic'],
                    help='Fixed expert-representation to component-head alignment')
parser.add_argument('--unified_expert', action='store_true', default=False,
                    help='Alias that selects routing_variant=unified')
parser.add_argument('--unified_expert_hidden', type=int, default=None,
                    help='Parameter-count-derived unified expert hidden width')
parser.add_argument('--local_descriptor_mode', choices=['legacy', 'revised_rms'], default='legacy',
                    help='Use analytic envelopes, RMS filter pooling, and fixed descriptor statistics')
parser.add_argument('--local_warmup_epochs', type=int, default=0,
                    help='Full-training-split local warm-up epochs before joint optimization')
parser.add_argument('--routing_prompt_mode', type=str, default='default',
                    choices=['default', 'raw_stats_only'],
                    help='Prompt policy for routing comparisons')
parser.add_argument('--save_branch_logits', action='store_true', default=False,
                    help='Save local/global/final logits and component representations')
parser.add_argument('--save_sample_metadata', action='store_true', default=False,
                    help='Save source file, condition, and window metadata with predictions')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# Moe-Settings
parser.add_argument('--moe_loss_weight', type=float, default=0.01)
parser.add_argument('--moe_alpha', type=float, default=0.01)
args = parser.parse_args()
if args.llm_readout:
    args.readout_mode = args.llm_readout
if args.unified_expert:
    args.routing_variant = 'unified'
if args.routing_variant == 'permuted_alignment' and args.expert_head_permutation == 'none':
    args.expert_head_permutation = 'cyclic'


def get_checkpoint_model(model, accelerator=None):
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    return model


def safe_args_dict(args):
    safe = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
    return safe


def _sha256_file(path):
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(values):
    raw = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_experiment_metadata(args):
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = "unknown"
    safe = safe_args_dict(args)
    training_keys = [
        "seed", "root_path", "seq_len", "dataset_window_stride", "stride", "patch_len",
        "batch_size", "learning_rate", "train_epochs", "patience", "lradj", "loss_type",
        "focal_gamma", "cb_beta", "moe_alpha", "global_backbone_mode", "prompt_mode",
        "reprogramming_mode", "readout_mode", "routing_variant", "routing_manifest",
        "expert_head_permutation", "local_descriptor_mode", "local_warmup_epochs",
    ]
    evaluation_keys = [
        "phm_split_metadata", "eval_batch_size", "eval_metrics_extended",
        "llm_backbone", "llm_model_path", "llm_repo_id", "llm_revision", "llm_active_layers",
        "llm_final_norm", "llm_attention_implementation",
        "gradient_accumulation_steps",
        "save_predictions", "save_branch_logits", "save_sample_metadata",
    ]
    return {
        "git_commit": git_commit,
        "model_name": safe.get("model"),
        "routing_variant": safe.get("routing_variant", "correct"),
        "routing_manifest": safe.get("routing_manifest", ""),
        "routing_manifest_sha256": _sha256_file(safe.get("routing_manifest", "")),
        "split_manifest": safe.get("phm_split_metadata", ""),
        "split_manifest_sha256": _sha256_file(safe.get("phm_split_metadata", "")),
        "seed": safe.get("seed"),
        "loss_type": safe.get("loss_type"),
        "global_encoder_type": safe.get("global_backbone_mode"),
        "prompt_mode": safe.get("prompt_mode"),
        "training_config_fingerprint": _fingerprint({key: safe.get(key) for key in training_keys}),
        "evaluation_config_fingerprint": _fingerprint({key: safe.get(key) for key in evaluation_keys}),
    }


def get_trainable_param_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def get_required_state_names(model):
    names = get_trainable_param_names(model)
    names.update(
        name for name, _ in model.named_buffers()
        if name.startswith('moe_ffn.mean_') or name.startswith('moe_ffn.std_')
    )
    return names


def trainable_state_dict(model):
    required = get_required_state_names(model)
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in required
    }


def build_checkpoint_key_report(model, state_dict=None):
    trainable_names = get_required_state_names(model)
    if state_dict is None:
        state_keys = set(model.state_dict().keys())
    else:
        state_keys = set(state_dict.keys())
    llm_model_keys = sorted(k for k in state_keys if k.startswith("llm_model."))
    report = {
        "total_keys": len(state_keys),
        "trainable_keys": len(state_keys & trainable_names),
        "trainable_expected_keys": len(trainable_names),
        "llm_model_keys": len(llm_model_keys),
        "missing_trainable_keys": sorted(trainable_names - state_keys),
        "unexpected_non_trainable_keys": sorted(state_keys - trainable_names),
    }
    return report


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def profile_enabled(args):
    return bool(getattr(args, "resource_profile_dir", ""))


def profile_paths(args):
    out_dir = getattr(args, "resource_profile_dir", "")
    if not out_dir:
        return {}
    os.makedirs(out_dir, exist_ok=True)
    return {
        "dir": out_dir,
        "partial": os.path.join(out_dir, "resource_profile_partial.jsonl"),
        "json": os.path.join(out_dir, "resource_profile.json"),
        "csv": os.path.join(out_dir, "resource_profile.csv"),
    }


def sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_memory_gb():
    if not torch.cuda.is_available():
        return {"max_allocated_gb": 0.0, "max_reserved_gb": 0.0}
    return {
        "max_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
        "max_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024 ** 3)),
    }


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def model_param_counts(model):
    model_to_count = get_checkpoint_model(model)
    total = 0
    trainable = 0
    for param in model_to_count.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }


def effective_loader_samples(loader, args):
    max_iters = args.debug_iters if getattr(args, "debug", False) else len(loader)
    batches = min(len(loader), max_iters)
    return int(batches * args.batch_size), int(batches)


def checkpoint_size_mb(path):
    if path and os.path.exists(path):
        return float(os.path.getsize(path) / (1024 ** 2))
    return 0.0


def append_resource_profile(args, record):
    if not profile_enabled(args):
        return
    paths = profile_paths(args)
    with open(paths["partial"], "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def finalize_resource_profile(args):
    if not profile_enabled(args):
        return
    paths = profile_paths(args)
    records = []
    if os.path.exists(paths["partial"]):
        with open(paths["partial"], "r") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    write_json(paths["json"], {
        "num_records": len(records),
        "records": records,
    })
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(paths["csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def append_final_test_profile(args, model, test_loader, checkpoint_dir, test_seconds, test_loss, test_label_wise_acc):
    if not profile_enabled(args):
        return
    test_windows, test_batches = effective_loader_samples(test_loader, args)
    counts = model_param_counts(model)
    record = {
        "record_type": "final_test",
        "epoch": None,
        "train_start_time": None,
        "train_end_time": None,
        "train_seconds": None,
        "val_seconds": None,
        "test_seconds": float(test_seconds),
        "train_windows": None,
        "val_windows": None,
        "test_windows": test_windows,
        "train_batches": None,
        "val_batches": None,
        "test_batches": test_batches,
        "train_samples_per_sec": None,
        "val_samples_per_sec": None,
        "test_samples_per_sec": float(test_windows / test_seconds) if test_seconds > 0 else 0.0,
        "average_batch_time": float(test_seconds / test_batches) if test_batches > 0 else 0.0,
        "max_batch_time": None,
        "gpu_name": gpu_name(),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "llm_layers": args.llm_layers,
        "readout_mode": getattr(args, "readout_mode", ""),
        "checkpoint_size_mb": checkpoint_size_mb(os.path.join(checkpoint_dir, "checkpoint.pt")),
        "best_checkpoint_size_mb": checkpoint_size_mb(os.path.join(checkpoint_dir, "checkpoint.pt")),
        "total_params": counts["total_params"],
        "trainable_params": counts["trainable_params"],
        "frozen_params": counts["frozen_params"],
        "active_llm_layers": args.llm_layers,
        "amp_enabled": bool(args.use_amp),
        "learning_rate": None,
        "train_loss": None,
        "val_loss": None,
        "val_label_wise_acc": None,
        "test_loss": float(test_loss),
        "test_label_wise_acc": float(test_label_wise_acc),
    }
    record.update(gpu_memory_gb())
    append_resource_profile(args, record)


def write_run_config(path, args, setting):
    write_json(os.path.join(path, "config.json"), {
        "setting": setting,
        "args": safe_args_dict(args),
        "ablation": {
            "global_backbone_mode": args.global_backbone_mode,
            "prompt_mode": args.prompt_mode,
            "reprogramming_mode": args.reprogramming_mode,
            "readout_mode": args.readout_mode,
            "transformer_d_model": getattr(args, "transformer_d_model", None),
            "transformer_nhead": getattr(args, "transformer_nhead", None),
            "transformer_num_layers": getattr(args, "transformer_num_layers", None),
            "transformer_dim_feedforward": getattr(args, "transformer_dim_feedforward", None),
            "transformer_dropout": getattr(args, "transformer_dropout", None),
        },
    })


def write_trainable_param_report(path, model):
    rows = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        total_params += count
        if param.requires_grad:
            trainable_params += count
        rows.append({
            "name": name,
            "shape": list(param.shape),
            "numel": count,
            "requires_grad": bool(param.requires_grad),
        })
    write_json(os.path.join(path, "trainable_param_report.json"), {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "parameters": rows,
    })


def write_readout_report(path, model):
    model_to_report = get_checkpoint_model(model)
    report = getattr(model_to_report, "last_readout_report", None)
    if report:
        write_json(os.path.join(path, "readout_report.json"), report)


def write_backbone_report(path, model):
    model_to_report = get_checkpoint_model(model)
    report = getattr(model_to_report, "last_backbone_report", None)
    if report:
        report = dict(report)
        report["prompt"] = getattr(model_to_report, "last_prompt_report", None)
        write_json(os.path.join(path, "backbone_report.json"), report)


PHM_COMPONENT_NAMES = ["motor", "gear", "LA", "RA"]
PHM_COMPONENT_DIMS = [5, 9, 5, 2]
PHM_COMPONENT_SLICES = [(0, 5), (5, 14), (14, 19), (19, 21)]


def _select_train_criterion(criterion, name):
    if isinstance(criterion, dict):
        return criterion.get(name, criterion.get("full"))
    return criterion


def _unwrap_phm_dataset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "window_pointers"):
        return dataset.dataset
    return dataset


def _iter_phm_real_indices(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "window_pointers"):
        base_dataset = dataset.dataset
        base_positions = getattr(dataset, "indices", range(len(dataset)))
        for position in base_positions:
            yield int(base_dataset.indices[int(position)])
        return

    for index in getattr(dataset, "indices", range(len(dataset))):
        yield int(index)


def _phm_train_label_statistics(train_data, args):
    base_dataset = _unwrap_phm_dataset(train_data)
    counts = np.zeros(args.num_classes, dtype=np.float64)
    total_windows = 0

    for real_idx in _iter_phm_real_indices(train_data):
        pointer = base_dataset.window_pointers[real_idx]
        folder_name = pointer[2]
        label = np.asarray(base_dataset.labels[folder_name], dtype=np.float64)
        if label.shape[0] != args.num_classes:
            raise RuntimeError(f"PHM label dim mismatch for {folder_name}: {label.shape[0]} != {args.num_classes}")
        counts += label
        total_windows += 1

    if total_windows <= 0:
        raise RuntimeError("Cannot build PHM loss statistics from an empty train split")

    positive_counts = np.maximum(counts, 1.0)
    negative_counts = np.maximum(total_windows - counts, 1.0)
    positive_ratios = counts / max(total_windows, 1)
    rare_cutoff = float(np.percentile(positive_ratios, 25))
    rare_labels = [int(i) for i, ratio in enumerate(positive_ratios) if ratio <= rare_cutoff]
    rare_labels_le_5pct = [int(i) for i, ratio in enumerate(positive_ratios) if ratio <= 0.05]
    return {
        "counts": counts,
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "positive_ratios": positive_ratios,
        "total_windows": int(total_windows),
        "rare_cutoff_bottom25_positive_ratio": rare_cutoff,
        "rare_label_indices_bottom25": rare_labels,
        "rare_label_indices_positive_ratio_le_5pct": rare_labels_le_5pct,
    }


def _component_tensor_dict(full_tensor):
    criteria = {"full": full_tensor}
    for name, (start, end) in zip(PHM_COMPONENT_NAMES, PHM_COMPONENT_SLICES):
        criteria[name] = full_tensor[start:end].clone()
    return criteria


def _criteria_from_factory(factory, tensor_by_scope=None):
    criteria = {}
    scopes = ["full"] + PHM_COMPONENT_NAMES
    for scope in scopes:
        tensor = tensor_by_scope.get(scope) if tensor_by_scope else None
        criteria[scope] = factory(tensor)
    return criteria


def build_dynamic_focal_criteria(train_data, args, path):
    stats = _phm_train_label_statistics(train_data, args)
    counts = stats["counts"]
    positive_counts = stats["positive_counts"]
    negative_counts = stats["negative_counts"]
    positive_ratios = stats["positive_ratios"]
    total_windows = stats["total_windows"]

    raw_alpha = total_windows / (args.num_classes * positive_counts)
    raw_alpha = raw_alpha / (raw_alpha.mean() + 1e-12)
    alpha = np.clip(raw_alpha, 0.25, 4.0).astype(np.float32)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    pos_weight = np.clip(negative_counts / positive_counts, 0.25, 20.0).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    beta = float(getattr(args, "cb_beta", 0.999))
    effective_num = 1.0 - np.power(beta, positive_counts)
    cb_weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    cb_weights = cb_weights / (cb_weights.mean() + 1e-12)
    cb_weights = np.clip(cb_weights, 0.25, 4.0).astype(np.float32)
    cb_weight_tensor = torch.tensor(cb_weights, dtype=torch.float32)

    loss_type = getattr(args, "loss_type", "dfl_detach")
    dfl_detach = bool(getattr(args, "dfl_detach_gamma", False) or loss_type == "dfl_detach")
    if loss_type == "dfl_full":
        dfl_detach = False
    gamma_base = float(getattr(args, "dfl_gamma_base", 1.0))
    gamma_scale = float(getattr(args, "dfl_gamma_scale", 2.0))
    if loss_type == "dfl_static_low":
        gamma_scale = 0.0
    elif loss_type == "dfl_static_high":
        gamma_base = gamma_base + gamma_scale
        gamma_scale = 0.0

    report = {
        "enabled": True,
        "source": "train_split_label_frequency_only",
        "loss_type": loss_type,
        "num_windows": int(total_windows),
        "num_classes": int(args.num_classes),
        "positive_counts": counts.astype(int).tolist(),
        "positive_ratios": positive_ratios.tolist(),
        "alpha": alpha.tolist(),
        "pos_weight": pos_weight.tolist(),
        "class_balanced_weights": cb_weights.tolist(),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean()),
        "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
        "cb_beta": beta,
        "asl_gamma_pos": float(getattr(args, "asl_gamma_pos", 0.0)),
        "asl_gamma_neg": float(getattr(args, "asl_gamma_neg", 4.0)),
        "asl_clip": float(getattr(args, "asl_clip", 0.05)),
        "gamma_base": gamma_base,
        "gamma_scale": gamma_scale,
        "dfl_detach_gamma": dfl_detach,
        "rare_label_definition": "bottom 25% labels by positive ratio computed from training split only",
        "rare_label_indices_bottom25": stats["rare_label_indices_bottom25"],
        "rare_cutoff_bottom25_positive_ratio": stats["rare_cutoff_bottom25_positive_ratio"],
        "rare_label_indices_positive_ratio_le_5pct": stats["rare_label_indices_positive_ratio_le_5pct"],
        "alternative_threshold_note": "positive_ratio <= 5%",
        "component_slices": {
            name: [int(start), int(end)]
            for name, (start, end) in zip(PHM_COMPONENT_NAMES, PHM_COMPONENT_SLICES)
        },
        "component_alpha": {
            name: alpha[start:end].tolist()
            for name, (start, end) in zip(PHM_COMPONENT_NAMES, PHM_COMPONENT_SLICES)
        },
    }
    write_json(os.path.join(path, "focal_alpha_report.json"), report)
    write_json(os.path.join(path, "loss_config_report.json"), report)
    print(
        "PHM_LOSS_CONFIG "
        f"loss_type={loss_type} num_windows={total_windows} "
        f"alpha_min={report['alpha_min']:.4f} alpha_max={report['alpha_max']:.4f} "
        f"gamma_base={gamma_base:.4f} gamma_scale={gamma_scale:.4f} detach_gamma={dfl_detach}"
    )

    alpha_by_scope = _component_tensor_dict(alpha_tensor)
    pos_weight_by_scope = _component_tensor_dict(pos_weight_tensor)
    cb_by_scope = _component_tensor_dict(cb_weight_tensor)

    if loss_type == "bce":
        criteria = _criteria_from_factory(lambda _: PlainBCEWithLogitsLoss())
    elif loss_type == "weighted_bce":
        criteria = _criteria_from_factory(lambda t: PlainBCEWithLogitsLoss(pos_weight=t), pos_weight_by_scope)
    elif loss_type == "focal":
        criteria = _criteria_from_factory(lambda _: FocalLoss(alpha=1.0, gamma=float(getattr(args, "focal_gamma", 2.0))))
    elif loss_type == "cb_focal":
        criteria = _criteria_from_factory(
            lambda t: ClassBalancedFocalLoss(alpha=t, gamma=float(getattr(args, "focal_gamma", 2.0))),
            cb_by_scope,
        )
    elif loss_type == "asl":
        criteria = _criteria_from_factory(
            lambda _: AsymmetricLoss(
                gamma_pos=float(getattr(args, "asl_gamma_pos", 0.0)),
                gamma_neg=float(getattr(args, "asl_gamma_neg", 4.0)),
                clip=float(getattr(args, "asl_clip", 0.05)),
            )
        )
    elif loss_type in {"dfl_detach", "dfl_full", "dfl_static_low", "dfl_static_high"}:
        criteria = _criteria_from_factory(
            lambda t: DynamicFocalLoss(
                alpha=t,
                gamma_base=gamma_base,
                gamma_scale=gamma_scale,
                detach_gamma=dfl_detach,
            ),
            alpha_by_scope,
        )
    else:
        raise ValueError(f"Unsupported loss_type={loss_type}")
    return criteria


def run_head_alignment_check(path, model, train_loader, args, device):
    report = {
        "component_names": PHM_COMPONENT_NAMES,
        "component_dims": PHM_COMPONENT_DIMS,
        "component_slices": PHM_COMPONENT_SLICES,
        "num_classes": args.num_classes,
        "status": "FAIL",
    }
    if sum(PHM_COMPONENT_DIMS) != args.num_classes or args.num_classes != 21:
        report["error"] = "component_dims must sum to num_classes=21"
        write_json(os.path.join(path, "head_alignment_report.json"), report)
        raise RuntimeError(report["error"])

    model_to_check = get_checkpoint_model(model)
    for attr, expected in [
        ("component_names", PHM_COMPONENT_NAMES),
        ("component_dims", PHM_COMPONENT_DIMS),
        ("component_slices", PHM_COMPONENT_SLICES),
    ]:
        actual = getattr(model_to_check, attr, None)
        if actual != expected:
            report["error"] = f"Model {attr} mismatch: {actual} != {expected}"
            write_json(os.path.join(path, "head_alignment_report.json"), report)
            raise RuntimeError(report["error"])

    head_dims = [
        getattr(model_to_check, "motor_head").out_features,
        getattr(model_to_check, "gear_head").out_features,
        getattr(model_to_check, "LA_head").out_features,
        getattr(model_to_check, "RA_head").out_features,
    ]
    report["model_head_dims"] = head_dims
    if head_dims != PHM_COMPONENT_DIMS:
        report["error"] = f"Model head dims mismatch: {head_dims} != {PHM_COMPONENT_DIMS}"
        write_json(os.path.join(path, "head_alignment_report.json"), report)
        raise RuntimeError(report["error"])

    batch = next(iter(train_loader))
    batch_x = batch["data"][:1].float().to(device)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        if args.use_amp:
            with torch.cuda.amp.autocast():
                final_logits, llm_logits, moe_logits = model(batch_x)
        else:
            final_logits, llm_logits, moe_logits = model(batch_x)
    if model_was_training:
        model.train()
    shapes = {
        "input": list(batch_x.shape),
        "final_logits": list(final_logits.shape),
        "llm_logits": list(llm_logits.shape),
        "moe_logits": list(moe_logits.shape),
    }
    report["shapes"] = shapes
    for name, tensor in [("final_logits", final_logits), ("llm_logits", llm_logits), ("moe_logits", moe_logits)]:
        if tensor.shape[-1] != args.num_classes:
            report["error"] = f"{name} last dim {tensor.shape[-1]} != num_classes {args.num_classes}"
            write_json(os.path.join(path, "head_alignment_report.json"), report)
            raise RuntimeError(report["error"])
    report["status"] = "HEAD_ALIGNMENT_OK"
    write_json(os.path.join(path, "head_alignment_report.json"), report)
    print("HEAD_ALIGNMENT_OK")
    print(json.dumps(report, indent=2))
    return report


def save_model_checkpoint(model, checkpoint_path, args, accelerator=None, extra=None):
    model_to_save = get_checkpoint_model(model, accelerator)
    has_registered_llm = any(name.startswith("llm_model.") for name, _ in model_to_save.named_parameters())
    if has_registered_llm and not getattr(args, "save_trainable_only", True):
        raise RuntimeError(
            "Refusing to serialize a full frozen LLM. Use --save_trainable_only (the default)."
        )
    if getattr(args, "save_trainable_only", False):
        state = trainable_state_dict(model_to_save)
        key_report = build_checkpoint_key_report(model_to_save, state)
        if key_report["llm_model_keys"] > 0:
            raise RuntimeError(f"Trainable-only checkpoint contains llm_model.* keys: {key_report['llm_model_keys']}")
        experiment_metadata = build_experiment_metadata(args)
        payload = {
            "format": "trainable_only",
            "save_trainable_only": True,
            "num_saved_tensors": len(state),
            "args": safe_args_dict(args),
            "model_state_dict": state,
            "experiment_metadata": experiment_metadata,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, checkpoint_path)
        report_path = os.path.join(os.path.dirname(checkpoint_path), "checkpoint_key_report.json")
        write_json(report_path, {
            **key_report,
            "checkpoint_path": checkpoint_path,
            "experiment_metadata": experiment_metadata,
        })
        print(f"💾 Saved trainable-only checkpoint: {checkpoint_path} | tensors={len(state)} | llm_model_keys=0")
    else:
        torch.save(model_to_save.state_dict(), checkpoint_path)


def load_model_checkpoint(model, checkpoint_path, args, accelerator=None, map_location="cpu"):
    model_to_load = get_checkpoint_model(model, accelerator)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    is_trainable_only = isinstance(checkpoint, dict) and checkpoint.get("save_trainable_only", False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    incompatible = model_to_load.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    print(f"🔁 Loaded checkpoint: {checkpoint_path}")
    print(f"   ┣━ save_trainable_only: {is_trainable_only}")
    print(f"   ┣━ loaded_keys: {len(state_dict)}")
    print(f"   ┣━ missing_keys: {len(missing_keys)}")
    print(f"   ┗━ unexpected_keys: {len(unexpected_keys)}")
    if is_trainable_only:
        key_report = build_checkpoint_key_report(model_to_load, state_dict)
        if key_report["llm_model_keys"] > 0:
            raise RuntimeError("Trainable-only checkpoint contains llm_model.* keys")
        if key_report["missing_trainable_keys"]:
            raise RuntimeError(f"Trainable parameters missing from checkpoint: {key_report['missing_trainable_keys'][:20]}")
    return checkpoint


def collect_classification_predictions(args, accelerator, model, loader, path):
    model.eval()
    logits_list, global_logits_list, local_logits_list, labels_list, names = [], [], [], [], []
    representation_lists = [[], [], [], []]
    metadata = {
        "source_ids": [], "source_files": [], "condition_ids": [],
        "window_starts": [], "window_indices": [], "label_dirs": [],
    }
    max_iters = args.debug_iters if getattr(args, "debug", False) else float("inf")
    start_time = time.time()
    num_batches = 0
    num_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_iters:
                break
            batch_x = batch["data"].float().to(accelerator.device)
            batch_y = batch["label"].float().to(accelerator.device)
            collect_branches = bool(getattr(args, "save_branch_logits", False))
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    model_outputs = model(batch_x, return_branch_outputs=collect_branches)
            else:
                model_outputs = model(batch_x, return_branch_outputs=collect_branches)
            if collect_branches:
                outputs = model_outputs["final_logits"]
                global_outputs = model_outputs["global_logits"]
                local_outputs = model_outputs["local_logits"]
                component_representations = model_outputs["component_representations"]
            else:
                outputs, global_outputs, local_outputs = model_outputs
                component_representations = None
            write_readout_report(path, model)
            write_backbone_report(path, model)
            outputs, global_outputs, local_outputs, batch_y = accelerator.gather_for_metrics(
                (outputs, global_outputs, local_outputs, batch_y)
            )
            logits_list.append(outputs.detach().cpu())
            global_logits_list.append(global_outputs.detach().cpu())
            local_logits_list.append(local_outputs.detach().cpu())
            labels_list.append(batch_y.detach().cpu())
            if component_representations is not None:
                for component_index, values in enumerate(component_representations):
                    gathered = accelerator.gather_for_metrics(values)
                    representation_lists[component_index].append(gathered.detach().cpu())
            batch_names = batch.get("sample_id", batch.get("name", None))
            if batch_names is None:
                names.extend([f"sample_{num_samples + j}" for j in range(batch_y.shape[0])])
            elif isinstance(batch_names, (list, tuple)):
                names.extend([str(x) for x in batch_names])
            else:
                names.extend([str(batch_names)])
            if getattr(args, "save_sample_metadata", False):
                field_map = {
                    "source_ids": "source_id", "source_files": "source_file",
                    "condition_ids": "condition_id", "window_starts": "window_start",
                    "window_indices": "window_index", "label_dirs": "label_dir",
                }
                for output_name, batch_name in field_map.items():
                    values = batch.get(batch_name, [])
                    if torch.is_tensor(values):
                        values = values.detach().cpu().tolist()
                    elif not isinstance(values, (list, tuple)):
                        values = [values]
                    metadata[output_name].extend(values)
            num_batches += 1
            num_samples += int(batch_y.shape[0])
    elapsed = time.time() - start_time
    logits = torch.cat(logits_list, dim=0)
    global_logits = torch.cat(global_logits_list, dim=0)
    local_logits = torch.cat(local_logits_list, dim=0)
    y_true = torch.cat(labels_list, dim=0)
    y_prob = torch.sigmoid(logits)
    y_pred = (y_prob > 0.5).float()
    if getattr(args, "save_predictions", False):
        prediction_payload = {
            "y_true": y_true.numpy(),
            "y_prob": y_prob.numpy(),
            "y_pred": y_pred.numpy(),
            "logits": logits.numpy(),
            "final_logits": logits.numpy(),
            "global_logits": global_logits.numpy(),
            "local_logits": local_logits.numpy(),
            "sample_ids": np.array(names, dtype=object),
            "routing_variant": np.array([getattr(args, "routing_variant", "correct")], dtype=object),
            "seed": np.array([int(args.seed)], dtype=np.int64),
        }
        if getattr(args, "save_branch_logits", False):
            for component_index, component_name in enumerate(PHM_COMPONENT_NAMES):
                if representation_lists[component_index]:
                    prediction_payload[f"{component_name}_representation"] = torch.cat(
                        representation_lists[component_index], dim=0
                    ).numpy()
        if getattr(args, "save_sample_metadata", False):
            for key, values in metadata.items():
                prediction_payload[key] = np.asarray(values)
        np.savez_compressed(os.path.join(path, "predictions.npz"), **prediction_payload)
    metrics = evaluate_metrics_extended(logits, y_true, verbose=True) if getattr(args, "eval_metrics_extended", False) else {}
    if metrics:
        write_json(os.path.join(path, "metrics.json"), metrics)
        if getattr(args, "save_branch_logits", False):
            write_json(os.path.join(path, "branch_metrics.json"), {
                "final": metrics,
                "global": evaluate_metrics_extended(global_logits, y_true, verbose=False),
                "local": evaluate_metrics_extended(local_logits, y_true, verbose=False),
            })
    if getattr(args, "profile_inference", False):
        write_json(os.path.join(path, "latency.json"), {
            "num_batches": num_batches,
            "num_samples": num_samples,
            "elapsed_sec": elapsed,
            "samples_per_sec": num_samples / elapsed if elapsed > 0 else 0.0,
            "avg_batch_latency_sec": elapsed / num_batches if num_batches > 0 else 0.0,
        })
    write_readout_report(path, model)
    write_backbone_report(path, model)
    model.train()
    return metrics





SHORT_CLASS_NAMES = [
    "M0", "M1", "M2", "M3", "M4",
    "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
    "LA0", "LA1", "LA2", "LA3", "LA4",
    "RA0", "RA1"
]
# 设置随机种子
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
log.info(f"Random seed set to {fix_seed}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")
accelerator=Accelerator()

MAX_ITERS_PER_EPOCH = args.debug_iters if getattr(args, "debug", False) else float('inf')



for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)
    log.info(f"Experiment setting: {setting}")
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    os.makedirs(path, exist_ok=True)  # 提前创建文件夹
    write_run_config(path, args, setting)
    log_file_path = os.path.join(path, "training_log.csv")
    with open(log_file_path, "w") as f:
        f.write("epoch,train_loss,vali_loss,vali_label_wise_acc\n")
    if profile_enabled(args):
        os.makedirs(args.resource_profile_dir, exist_ok=True)
        with open(profile_paths(args)["partial"], "w") as f:
            f.write("")

    # 2. 创建列表来收集每个 step 的训练损失
    all_train_losses = []
    log.info("Loading datasets...")

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    log.info("Building model...")
    model = KGLLMoE.Model(args).float()

    model.to(device)
    log.info(f"Model {args.model} initialized and moved to device.")
    if args.task_name == "classification" and args.data == "phm" and hasattr(model, "moe_ffn"):
        local_report = model.moe_ffn.local_parameter_report()
        local_names = {f"moe_ffn.{name}" for name, _ in model.moe_ffn.named_parameters()}
        total_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        local_trainable = sum(
            param.numel() for name, param in model.named_parameters()
            if param.requires_grad and name in local_names
        )
        global_trainable = total_trainable - local_trainable
        local_branch_param_report = {
            **local_report,
            "total_trainable_params": total_trainable,
            "local_trainable_params": local_trainable,
            "nonlocal_trainable_params": global_trainable,
            "routing_manifest_hash": getattr(model.moe_ffn, "routing_manifest_hash", ""),
            "routing_variant": getattr(args, "routing_variant", "correct"),
        }
        if getattr(args, "routing_variant", "correct") == "unified" and local_report["relative_error"] > 0.02:
            raise RuntimeError(f"Unified local parameter mismatch: {local_report}")
        write_json(os.path.join(path, "local_branch_parameter_report.json"), local_branch_param_report)

    finetune_checkpoint = getattr(args, 'finetune_from_checkpoint', '') or getattr(args, 'load_trainable_checkpoint', '')
    if finetune_checkpoint:
        if not os.path.exists(finetune_checkpoint):
            raise FileNotFoundError(f"Fine-tune source checkpoint not found: {finetune_checkpoint}")
        loaded_checkpoint = load_model_checkpoint(
            model,
            finetune_checkpoint,
            args,
            accelerator=accelerator,
            map_location='cpu',
        )
        write_json(os.path.join(path, 'finetune_source_checkpoint.json'), {
            'source_checkpoint': finetune_checkpoint,
            'checkpoint_format': loaded_checkpoint.get('format') if isinstance(loaded_checkpoint, dict) else 'raw_state_dict',
            'save_trainable_only': bool(loaded_checkpoint.get('save_trainable_only', False)) if isinstance(loaded_checkpoint, dict) else False,
            'num_saved_tensors': int(loaded_checkpoint.get('num_saved_tensors', 0)) if isinstance(loaded_checkpoint, dict) else None,
        })
        print(f"FINETUNE_SOURCE_LOADED {finetune_checkpoint}")

    if args.task_name == 'classification' and args.data == 'phm':
        criterion = build_dynamic_focal_criteria(train_data, args, path)
    else:
        criterion = DynamicFocalLoss(alpha=1.0, gamma_base=1.0, gamma_scale=2.0)

    if args.local_descriptor_mode == 'revised_rms' and args.is_training:
        if args.local_warmup_epochs != 5 and not finetune_checkpoint:
            raise ValueError('Unified revised-local training requires five warm-up epochs')
        if args.debug or args.phm_balanced_subset_metadata:
            raise ValueError('Revised-local warm-up requires the full training partition')
        if not args.phm_split_metadata:
            raise ValueError('Revised-local warm-up requires explicit split metadata')
        if not finetune_checkpoint:
            local_train_loader = DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, drop_last=False,
            )
            local_val_loader = DataLoader(
                vali_data, batch_size=args.eval_batch_size, shuffle=False,
                num_workers=args.num_workers, drop_last=False,
            )
            fit_local_descriptors(model, local_train_loader, device)
            report = warm_up_local(
                model, local_train_loader, local_val_loader,
                _select_train_criterion(criterion, 'full'), device,
                args.local_warmup_epochs, args.learning_rate,
                os.path.join(path, 'local_warmup.pt'),
            )
            write_json(os.path.join(path, 'local_warmup.json'), report)
    elif args.local_warmup_epochs:
        raise ValueError('--local_warmup_epochs requires revised_rms training')

    if args.task_name == "classification" and args.data == "phm":
        run_head_alignment_check(path, model, train_loader, args, device)
        write_readout_report(path, model)
        write_backbone_report(path, model)
    write_trainable_param_report(path, get_checkpoint_model(model, accelerator))

    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    args.content = load_content(args)
    if not os.path.exists(path):
        os.makedirs(path)
    log.info(f"Checkpoints path: {path}")

    time_now = time.time()
    train_steps = len(train_loader)
    early_save_fn = None
    if getattr(args, "save_trainable_only", False):
        early_save_fn = lambda m, p: save_model_checkpoint(
            m,
            p,
            args,
            accelerator=accelerator,
            extra={"setting": setting, "checkpoint_type": "early_stopping"},
        )
    early_stopping = EarlyStopping(
        patience=args.patience,
        save_mode=not getattr(args, "debug", False),
        save_fn=early_save_fn,
    )

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    #选择两种学习率调整策略 1.余弦退火 2.单周期学习率
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim,
            T_max=max(1, args.train_epochs),
            eta_min=1e-8,
        )
        print(f"Using CosineAnnealingLR: T_max={max(1, args.train_epochs)}, eta_min=1e-8")
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)


    #criterion = FocalLoss(alpha=1.0,gamma=1.7)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    log.info("Start training...")
    train_loss_history = []
    vali_loss_history = []
    param_counts = model_param_counts(model)
    for epoch in range(args.train_epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        epoch_start_wall = time.time()
        train_start_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        time_now = time.time()
        epoch_train_losses = []  # 用于计算当前epoch的平均loss
        batch_times = []

        total_samples = 0
        total_correct = 0
        total_exact_matches = 0
        total_windows_seen = 0

        progress = epoch / args.train_epochs
        alpha = 0.8
        beta = 0.2  # MoE 辅助但不过度主导

        accumulation_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        max_train_batches = min(len(train_loader), MAX_ITERS_PER_EPOCH)
        model_optim.zero_grad(set_to_none=True)


        for i, batch in enumerate(tqdm(train_loader, desc=f"[Train Epoch {epoch + 1}]")):
            if i >= MAX_ITERS_PER_EPOCH:
                break
            sync_cuda_if_needed()
            batch_time_start = time.time()
            iter_count += 1
            if i % accumulation_steps == 0:
                model_optim.zero_grad(set_to_none=True)

            batch_x = batch['data'].float().to(device)
            batch_y = batch['label'].float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    # ===== forward =====
                    outputs, llm_logits, moe_logits = model(batch_x)  # outputs == final_logits

                    # ===== multi-head split =====
                    motor_out = outputs[:, 0:5]
                    gear_out = outputs[:, 5:14]
                    LA_out = outputs[:, 14:19]
                    RA_out = outputs[:, 19:21]

                    motor_y = batch_y[:, 0:5]
                    gear_y = batch_y[:, 5:14]
                    LA_y = batch_y[:, 14:19]
                    RA_y = batch_y[:, 19:21]

                    # ===== multi-head loss (using your criterion only) =====
                    motor_loss = _select_train_criterion(criterion, "motor")(motor_out, motor_y)
                    gear_loss = _select_train_criterion(criterion, "gear")(gear_out, gear_y)
                    LA_loss = _select_train_criterion(criterion, "LA")(LA_out, LA_y)
                    RA_loss = _select_train_criterion(criterion, "RA")(RA_out, RA_y)

                    weights = {"motor": 1.0, "gear": 1.0, "LA": 1.0, "RA": 0.5}

                    weighted_loss = (
                                            weights["motor"] * motor_loss +
                                            weights["gear"] * gear_loss +
                                            weights["LA"] * LA_loss +
                                            weights["RA"] * RA_loss
                                    ) / (sum(weights.values()) + 1e-12)

                    # ===== LLM loss (new, using criterion) =====
                    # llm_logits shape == [B, 21]
                    llm_loss = _select_train_criterion(criterion, "full")(llm_logits, batch_y)

                    # ===== MoE loss (new, using criterion) =====
                    # moe_logits shape == [B, 21]
                    moe_loss = _select_train_criterion(criterion, "full")(moe_logits, batch_y)

                    # ===== Print info =====
                    if (i + 1) % 100 == 0:
                        print(f"🔵 LLM loss = {llm_loss.item():.6f}")
                        print(f"🟢 MoE loss = {moe_loss.item():.6f}")
                        print(f"⚫ Weighted loss = {weighted_loss.item():.6f}")

                    # ===== Final loss =====
                    # beta 从 0.3 开始
                    loss = weighted_loss + alpha * llm_loss + beta * moe_loss
                    train_loss_history.append(loss.item())
                    epoch_train_losses.append(loss.item())

                train_loss.append(loss.item())
                scaler.scale(loss / accumulation_steps).backward()
                should_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == max_train_batches)
                if should_step:
                    scaler.unscale_(model_optim)
                    #梯度剪裁
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()

            # === 计算多标签预测 ===
            preds = (torch.sigmoid(outputs) > 0.5).float()  # 转为 0/1
            correct = (preds == batch_y).float().sum()  # 全部标签中预测对的个数
            total_correct += correct.item()
            total_samples += batch_y.numel()  # 所有标签个数（B × 20）
            total_exact_matches += (preds == batch_y).all(dim=1).float().sum().item()
            total_windows_seen += batch_y.shape[0]


            if args.lradj == 'TST':
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

            if (i + 1) % 100 == 0:
                label_wise_acc = total_correct / total_samples
                current_emr = total_exact_matches / total_windows_seen if total_windows_seen > 0 else 0.0

                print(f"\n[Batch {i + 1} @ Epoch {epoch + 1}]")
                print(f"  ┣━ Total loss: {loss.item():.6f} | label_wise_acc: {label_wise_acc:.4f} | current_EMR: {current_emr:.4f}")
                print(f"  ┣━ Head losses:")
                print(f"  ┃   Motor: {motor_loss.item():.6f}")
                print(f"  ┃   Gear : {gear_loss.item():.6f}")
                print(f"  ┃   LA   : {LA_loss.item():.6f}")
                print(f"  ┃   RA   : {RA_loss.item():.6f}")
                total_head_loss = motor_loss + gear_loss + LA_loss + RA_loss
                eps=1e-8
                print(f"  ┣━ Contribution ratio:")
                print(f"  ┃   Motor: {motor_loss.item() / (total_head_loss.item()+eps):.3f}")
                print(f"  ┃   Gear : {gear_loss.item() / (total_head_loss.item()+eps):.3f}")
                print(f"  ┃   LA   : {LA_loss.item() / (total_head_loss.item()+eps):.3f}")
                print(f"  ┃   RA   : {RA_loss.item() / (total_head_loss.item()+eps):.3f}")

                # 打印 MoE 激活率
                if hasattr(model, "last_moe_scores") and model.last_moe_scores is not None:
                    with torch.no_grad():
                        probs = model.last_moe_scores.mean(dim=(0, 1))  # [E]
                        probs_cpu = probs.cpu().numpy().tolist()
                        print(f"  ┣━ MoE mean probs: {['%.3f' % p for p in probs_cpu]}")
            sync_cuda_if_needed()
            batch_times.append(time.time() - batch_time_start)


        train_end_wall = time.time()
        train_end_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        train_seconds = train_end_wall - epoch_start_wall
        print(f"🕓 Epoch {epoch + 1} finished in {time.time() - epoch_time:.2f} seconds")
        train_loss = np.average(train_loss)


        # === 验证与测试 ===（注意这里的 vali/test 逻辑也要适配多标签）
        sync_cuda_if_needed()
        val_start_wall = time.time()
        vali_loss, vali_label_wise_acc = vali_classify(
            args=args,
            model=model,
            vali_data=vali_data,
            vali_loader=vali_loader,
            criterion=criterion,
            accelerator=accelerator
        )
        sync_cuda_if_needed()
        val_seconds = time.time() - val_start_wall

        vali_loss_history.append(vali_loss)


        print(
            f"📉 Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Vali Loss: {vali_loss:.6f} | Vali label_wise_acc: {vali_label_wise_acc:.6f}")

        with open(log_file_path, "a") as f:
            f.write(f"{epoch + 1},{train_loss},{vali_loss},{vali_label_wise_acc}\n")
        os.makedirs(path, exist_ok=True)
        # === 保存每个 epoch 的 checkpoint ===
        epoch_ckpt = os.path.join(path, f"epoch_{epoch + 1}.pt")
        if getattr(args, "debug", False) and not getattr(args, "save_trainable_only", False):
            torch.save({
                "debug": True,
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss),
                "vali_label_wise_acc": float(vali_label_wise_acc),
                "model": args.model,
                "model_id": args.model_id,
            }, epoch_ckpt)
            print(f"💾 Saved debug checkpoint marker: {epoch_ckpt}")
        else:
            save_model_checkpoint(
                model,
                epoch_ckpt,
                args,
                accelerator=accelerator,
                extra={
                    "setting": setting,
                    "checkpoint_type": "epoch",
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "vali_loss": float(vali_loss),
                    "vali_label_wise_acc": float(vali_label_wise_acc),
                },
            )
        print(f"💾 Saved checkpoint: {epoch_ckpt}")
        early_stopping(vali_loss, model, path)

        train_windows, train_batches_profile = effective_loader_samples(train_loader, args)
        val_windows, val_batches_profile = effective_loader_samples(vali_loader, args)
        epoch_profile = {
            "record_type": "epoch",
            "epoch": epoch + 1,
            "train_start_time": train_start_time,
            "train_end_time": train_end_time,
            "train_seconds": float(train_seconds),
            "val_seconds": float(val_seconds),
            "test_seconds": None,
            "train_windows": train_windows,
            "val_windows": val_windows,
            "test_windows": None,
            "train_batches": train_batches_profile,
            "val_batches": val_batches_profile,
            "test_batches": None,
            "train_samples_per_sec": float(train_windows / train_seconds) if train_seconds > 0 else 0.0,
            "val_samples_per_sec": float(val_windows / val_seconds) if val_seconds > 0 else 0.0,
            "test_samples_per_sec": None,
            "average_batch_time": float(np.mean(batch_times)) if batch_times else 0.0,
            "max_batch_time": float(np.max(batch_times)) if batch_times else 0.0,
            "gpu_name": gpu_name(),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "patch_len": args.patch_len,
            "llm_layers": args.llm_layers,
            "readout_mode": getattr(args, "readout_mode", ""),
            "checkpoint_size_mb": checkpoint_size_mb(epoch_ckpt),
            "best_checkpoint_size_mb": checkpoint_size_mb(os.path.join(path, "checkpoint.pt")),
            "total_params": param_counts["total_params"],
            "trainable_params": param_counts["trainable_params"],
            "frozen_params": param_counts["frozen_params"],
            "active_llm_layers": args.llm_layers,
            "amp_enabled": bool(args.use_amp),
            "learning_rate": float(model_optim.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(vali_loss),
            "val_label_wise_acc": float(vali_label_wise_acc),
        }
        epoch_profile.update(gpu_memory_gb())
        append_resource_profile(args, epoch_profile)

        if early_stopping.early_stop:
            print("⛔ Early stopping triggered")
            plot_convergence_curves(
                train_losses_steps=train_loss_history,
                save_path=os.path.join(path, "training_convergence_curve.png")
            )

            # ✅ 加载最佳模型进行最终 test 评估
            print("🎯 Loading best model from checkpoint for final test evaluation...")

            checkpoint_path = os.path.join(path, 'checkpoint.pt')

            load_model_checkpoint(model, checkpoint_path, args, accelerator=accelerator, map_location="cpu")

            model.eval()
            sync_cuda_if_needed()
            test_start_wall = time.time()
            test_loss, test_label_wise_acc, all_preds, all_labels = test_classify(
                args=args,
                accelerator=accelerator,
                model=model,
                test_loader=test_loader,
                criterion=criterion
            )
            sync_cuda_if_needed()
            append_final_test_profile(
                args, model, test_loader, path, time.time() - test_start_wall, test_loss, test_label_wise_acc
            )

            print(f"🧪 Final Test Loss: {test_loss:.6f} | Test label_wise_acc: {test_label_wise_acc:.6f}")
            # ✅ 调用你写的 evaluate_metrics 输出详细指标
            evaluate_metrics(
                preds_list=[torch.tensor(all_preds)],
                labels_list=[torch.tensor(all_labels)],
                threshold=0.5,
                verbose=True
            )
            if getattr(args, "save_predictions", False) or getattr(args, "eval_metrics_extended", False) or getattr(args, "profile_inference", False):
                collect_classification_predictions(args, accelerator, model, test_loader, path)
            preds_binary = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).int().numpy()

            # ============= 关键改动：调用新的绘图函数 =============
            try:
                plot_co_occurrence_matrix(
                    train_losses_steps=train_loss_history,
                    vali_losses_epochs=vali_loss_history,
                    save_path=os.path.join(path, "convergence_curves_smooth.png")
                )
            except Exception as exc:
                print(f"⚠️ Plotting skipped after successful evaluation: {exc}")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


    if not early_stopping.early_stop:
        print("⚠️ EarlyStopping 未触发，加载最终模型进行测试评估...")

        if getattr(args, "debug", False):
            print("🧪 Debug mode: evaluating current in-memory model without loading checkpoint.pt")
            model.eval()
            sync_cuda_if_needed()
            test_start_wall = time.time()
            test_loss, test_label_wise_acc, all_preds, all_labels = test_classify(
                args=args,
                accelerator=accelerator,
                model=model,
                test_loader=test_loader,
                criterion=criterion
            )
            sync_cuda_if_needed()
            append_final_test_profile(
                args, model, test_loader, path, time.time() - test_start_wall, test_loss, test_label_wise_acc
            )
            print(f"🧪 Final Test Loss: {test_loss:.6f} | Test label_wise_acc: {test_label_wise_acc:.6f}")
            evaluate_metrics(
                preds_list=[torch.tensor(all_preds)],
                labels_list=[torch.tensor(all_labels)],
                threshold=0.5,
                verbose=True
            )
            if getattr(args, "save_predictions", False) or getattr(args, "eval_metrics_extended", False) or getattr(args, "profile_inference", False):
                collect_classification_predictions(args, accelerator, model, test_loader, path)
            continue

        checkpoint_path = os.path.join(path, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            load_model_checkpoint(model, checkpoint_path, args, accelerator=accelerator, map_location="cpu")

            plot_convergence_curves(
                train_losses_steps=train_loss_history,
                save_path=os.path.join(path, "training_convergence_curve.png")
            )

            model.eval()
            sync_cuda_if_needed()
            test_start_wall = time.time()
            test_loss, test_label_wise_acc, all_preds, all_labels = test_classify(
                args=args,
                accelerator=accelerator,
                model=model,
                test_loader=test_loader,
                criterion=criterion
            )
            sync_cuda_if_needed()
            append_final_test_profile(
                args, model, test_loader, path, time.time() - test_start_wall, test_loss, test_label_wise_acc
            )

            print(f"🧪 Final Test Loss: {test_loss:.6f} | Test label_wise_acc: {test_label_wise_acc:.6f}")
            evaluate_metrics(
                preds_list=[torch.tensor(all_preds)],
                labels_list=[torch.tensor(all_labels)],
                threshold=0.5,
                verbose=True
            )
            if getattr(args, "save_predictions", False) or getattr(args, "eval_metrics_extended", False) or getattr(args, "profile_inference", False):
                collect_classification_predictions(args, accelerator, model, test_loader, path)
            preds_binary = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).int().numpy()

            # ============= 关键改动：调用新的绘图函数 =============
            try:
                plot_co_occurrence_matrix(
                    y_true=all_labels,
                    y_pred_binary=preds_binary,
                    class_names=SHORT_CLASS_NAMES,  # 确保提供了21个类别名
                    save_path=os.path.join(path, "co_occurrence_matrix.png"),
                    normalize=True  # 推荐使用归一化，结果更具可比性
                )
            except Exception as exc:
                print(f"⚠️ Plotting skipped after successful evaluation: {exc}")
        else:
            print("🚫 未找到模型 checkpoint，跳过测试集评估。")

finalize_resource_profile(args)
# 清理模型缓存
torch.cuda.empty_cache()
#del_files('./checkpoints')
#print("Checkpoints cleared.")
