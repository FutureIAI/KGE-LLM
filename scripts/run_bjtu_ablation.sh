#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VARIANT="${VARIANT:-${1:-M0_full}}"
DATA_ROOT="${DATA_ROOT:-./dataset/Training_new}"
SPLIT_METADATA="${SPLIT_METADATA:?Set SPLIT_METADATA to audited condition-temporal metadata}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-./LLMs/llama-7b}"
CACHE_DIR="${CACHE_DIR:-./cache_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/ablation}"
SEED="${SEED:-2026}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"

case "$VARIANT" in
  M0_full)                BACKBONE=llama; PROMPT=full; REPROGRAMMING=full ;;
  M1_pm_moe_only)         BACKBONE=none; PROMPT=none; REPROGRAMMING=linear ;;
  M2_lite_trainable)      BACKBONE=transformer_trainable_lite; PROMPT=none; REPROGRAMMING=linear ;;
  M3_lite_frozen)         BACKBONE=transformer_frozen_lite; PROMPT=none; REPROGRAMMING=linear ;;
  M4_random_llama_shape)  BACKBONE=llama_random_frozen; PROMPT=full; REPROGRAMMING=full ;;
  M5_no_prompt)           BACKBONE=llama; PROMPT=none; REPROGRAMMING=full ;;
  M9_linear_projection)   BACKBONE=llama; PROMPT=full; REPROGRAMMING=linear ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

python -u run_main.py \
  --task_name classification --is_training 1 \
  --model_id "CT_LLMoE_${VARIANT}_seed${SEED}" --model CTLLMoE \
  --model_comment "${VARIANT}_seed${SEED}" --experiment_tag "$VARIANT" \
  --data phm --root_path "$DATA_ROOT" --phm_cache_dir "$CACHE_DIR" \
  --phm_split_metadata "$SPLIT_METADATA" --features M \
  --seq_len 512 --label_len 0 --pred_len 0 \
  --enc_in 21 --dec_in 21 --c_out 21 --num_classes 21 \
  --d_model 32 --d_ff 128 --n_heads 8 \
  --llm_model LLAMA --llm_backbone llama --llm_model_path "$LLM_MODEL_PATH" \
  --llm_dim 4096 --llm_layers 8 --llm_active_layers 8 \
  --patch_len 128 --stride 8 --dataset_window_stride 1024 \
  --transformer_d_model 512 --transformer_nhead 8 --transformer_num_layers 4 \
  --transformer_dim_feedforward 2048 --transformer_dropout 0.1 \
  --batch_size 16 --eval_batch_size 16 --num_workers 0 \
  --learning_rate 0.0001 --train_epochs "$TRAIN_EPOCHS" --patience 2 --itr 1 \
  --loss_type dfl_detach --dfl_gamma_base 1 --dfl_gamma_scale 2 \
  --global_backbone_mode "$BACKBONE" --prompt_mode "$PROMPT" \
  --reprogramming_mode "$REPROGRAMMING" --routing_variant correct \
  --readout_mode mean_patch --moe_alpha 0.2 --lradj COS \
  --use_amp --save_trainable_only --eval_metrics_extended --save_predictions \
  --checkpoints "$CHECKPOINT_ROOT/$VARIANT" --seed "$SEED"
