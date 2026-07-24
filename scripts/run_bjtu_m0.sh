#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./dataset/Training_new}"
SPLIT_METADATA="${SPLIT_METADATA:?Set SPLIT_METADATA to audited condition-temporal metadata}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to a local LLaMA directory}"
CACHE_DIR="${CACHE_DIR:-./cache_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/ct_llmoe_m0}"
SEED="${SEED:-2026}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-15}"

python -u run_main.py \
  --task_name classification --is_training 1 \
  --model_id "CT_LLMoE_M0_seed${SEED}" --model TimeLLM \
  --model_comment "M0_seed${SEED}" --experiment_tag "M0_full" \
  --data phm --root_path "$DATA_ROOT" --phm_cache_dir "$CACHE_DIR" \
  --phm_split_metadata "$SPLIT_METADATA" --features M \
  --seq_len 512 --label_len 0 --pred_len 0 \
  --enc_in 21 --dec_in 21 --c_out 21 --num_classes 21 \
  --d_model 32 --d_ff 128 --n_heads 8 \
  --llm_model LLAMA --llm_backbone llama --llm_model_path "$LLM_MODEL_PATH" \
  --llm_dim 4096 --llm_layers 8 --llm_active_layers 8 \
  --patch_len 128 --stride 8 --dataset_window_stride 1024 \
  --batch_size 16 --eval_batch_size 16 --num_workers 0 \
  --learning_rate 0.0001 --train_epochs "$TRAIN_EPOCHS" --patience 3 --itr 1 \
  --loss_type dfl_detach --dfl_gamma_base 1 --dfl_gamma_scale 2 \
  --global_backbone_mode llama --prompt_mode full --reprogramming_mode full \
  --routing_variant correct --readout_mode mean_patch --moe_alpha 0.2 --lradj COS \
  --use_amp --save_trainable_only --eval_metrics_extended --save_predictions \
  --save_branch_logits --save_sample_metadata \
  --checkpoints "$CHECKPOINT_ROOT" --seed "$SEED"
