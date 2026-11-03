import numpy as np
from math import sqrt

import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer, \
    GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
import os

from models.moe_ffn import LlamaMoEFeedForward
from models.revised_local import RevisedRmsMoE
from models.frozen_decoder_backbone import FrozenDecoderBackboneAdapter, FrozenDecoderBackboneSpec

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.configs=configs
        self.llm_backbone = getattr(configs, "llm_backbone", "llama")
        self.llm_model_path = getattr(configs, "llm_model_path", "")
        self.llm_repo_id = getattr(configs, "llm_repo_id", "")
        self.llm_revision = getattr(configs, "llm_revision", "")
        self.llm_active_layers = int(getattr(configs, "llm_active_layers", 0) or configs.llm_layers)
        self.llm_attention_implementation = getattr(configs, "llm_attention_implementation", "sdpa")
        self.global_backbone_mode = getattr(configs, "global_backbone_mode", "llama")
        self.prompt_mode = getattr(configs, "prompt_mode", "full")
        self.reprogramming_mode = getattr(configs, "reprogramming_mode", "full")
        self.readout_mode = getattr(configs, "readout_mode", "first_token_legacy")
        self.routing_variant = getattr(configs, "routing_variant", "correct")
        self.routing_manifest = getattr(configs, "routing_manifest", "")
        self.expert_head_permutation = getattr(configs, "expert_head_permutation", "none")
        self.routing_prompt_mode = getattr(configs, "routing_prompt_mode", "default")
        valid_backbones = {
            "llama", "none",
            "transformer_trainable", "transformer_frozen",
            "transformer_trainable_lite", "transformer_frozen_lite",
            "llama_random_frozen",
        }
        valid_prompt_modes = {"full", "none", "static", "dynamic", "random", "shuffled"}
        valid_reprogramming_modes = {"full", "linear", "mlp", "frozen_random"}
        valid_readout_modes = {"first_token_legacy", "first_patch", "mean_patch", "mean_all"}
        if self.global_backbone_mode not in valid_backbones:
            raise ValueError(f"Unknown global_backbone_mode={self.global_backbone_mode}")
        if self.prompt_mode not in valid_prompt_modes:
            raise ValueError(f"Unknown prompt_mode={self.prompt_mode}")
        if self.reprogramming_mode not in valid_reprogramming_modes:
            raise ValueError(f"Unknown reprogramming_mode={self.reprogramming_mode}")
        if self.readout_mode not in valid_readout_modes:
            raise ValueError(f"Unknown readout_mode={self.readout_mode}")
        if self.llm_backbone not in {"llama", "mistral"}:
            raise ValueError(f"Unknown llm_backbone={self.llm_backbone}")

        self.local_only = self.global_backbone_mode == "none"
        if self.local_only:
            self.moe_ffn = self._build_local_branch(configs)
            self.moe_alpha = 1.0
            self.normalize_layers = Normalize(configs.enc_in, affine=False)
            self.component_names = ["motor", "gear", "LA", "RA"]
            self.component_dims = [5, 9, 5, 2]
            self.component_slices = [(0, 5), (5, 14), (14, 19), (19, 21)]
            self.llm_model = None
            self.tokenizer = None
            self.decoder_adapter = None
            self.last_readout_report = {"readout_mode": "local_only", "readout_source": "local"}
            self.last_branch_outputs = None
            self.last_prompt_report = None
            self.last_backbone_report = None
            self.last_global_representation = None
            return

        self.channel_projection = nn.Linear(self.configs.d_model, 21)


        llama_path = self.llm_model_path or os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'llama-7b')

        self.tokenizer = None
        self.llm_model = None
        self.llm_config = None
        needs_llama_shape = self.global_backbone_mode in {"llama", "llama_random_frozen"}
        needs_llama_tokens = self.prompt_mode in {"full", "static", "dynamic", "shuffled"} or self.reprogramming_mode == "full"

        if self.llm_backbone == "mistral" and (needs_llama_shape or needs_llama_tokens):
            mistral_path = self.llm_model_path
            if not mistral_path or not os.path.isdir(mistral_path):
                raise FileNotFoundError(f"Mistral local model path does not exist: {mistral_path!r}")
            self.llm_config = AutoConfig.from_pretrained(
                mistral_path, local_files_only=True, trust_remote_code=False
            )
            if self.llm_config.model_type != "mistral":
                raise RuntimeError(f"Expected Mistral config, got {self.llm_config.model_type}")
            self.llm_config.num_hidden_layers = self.llm_active_layers
            self.llm_config.output_attentions = False
            self.llm_config.output_hidden_states = False
            self.llm_config.use_cache = False
            causal_model = AutoModelForCausalLM.from_pretrained(
                mistral_path,
                config=self.llm_config,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=self.llm_attention_implementation,
            )
            self.llm_model = causal_model.model
            del causal_model
            self.tokenizer = AutoTokenizer.from_pretrained(
                mistral_path, local_files_only=True, trust_remote_code=False
            )
        elif configs.llm_model == 'LLAMA' and (needs_llama_shape or needs_llama_tokens):
            if not os.path.isdir(llama_path):
                raise FileNotFoundError(f"LLaMA local model path does not exist: {llama_path!r}")
            self.llama_config = LlamaConfig.from_pretrained(llama_path)
            self.llama_config.num_hidden_layers = self.llm_active_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_config = self.llama_config

            if self.global_backbone_mode == "llama":
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        llama_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                    )
                except EnvironmentError:
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        llama_path,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                    )
            elif self.global_backbone_mode == "llama_random_frozen":
                # Same active LLaMA-shaped layers as the baseline, but randomly initialized and frozen.
                self.llm_model = LlamaModel(self.llama_config)

            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    llama_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    llama_path,
                    trust_remote_code=True,
                    local_files_only=False
                )


        elif configs.llm_model == 'GPT2' and needs_llama_shape:
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT' and needs_llama_shape:
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif needs_llama_shape or needs_llama_tokens:
            raise Exception('LLM model is not defined')

        lite_transformer_modes = {"transformer_trainable_lite", "transformer_frozen_lite"}
        current_transformer_modes = {"transformer_trainable", "transformer_frozen"}
        transformer_modes = current_transformer_modes | lite_transformer_modes

        self.backbone_hidden_size = self.d_llm
        if self.global_backbone_mode in lite_transformer_modes:
            self.backbone_hidden_size = int(getattr(configs, "transformer_d_model", 512))
        if self.llm_model is not None:
            self.backbone_hidden_size = self.llm_model.config.hidden_size

        if self.tokenizer is not None and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer is not None:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        if self.llm_model is not None:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        requested_final_norm = getattr(configs, "llm_final_norm", None)
        if requested_final_norm is None:
            requested_final_norm = self.llm_backbone == "mistral"
        self.decoder_adapter = None
        if self.llm_model is not None and self.global_backbone_mode in {"llama", "llama_random_frozen"}:
            self.decoder_adapter = FrozenDecoderBackboneAdapter(FrozenDecoderBackboneSpec(
                name=self.llm_backbone,
                repo_id=self.llm_repo_id,
                revision=self.llm_revision,
                model_path=self.llm_model_path or llama_path,
                active_layers=self.llm_active_layers,
                attention_implementation=str(getattr(self.llm_model.config, "_attn_implementation", "unknown")),
                apply_final_norm=bool(requested_final_norm),
                official_model_forward=self.llm_backbone == "mistral",
            ))

        if self.global_backbone_mode in transformer_modes:
            if self.global_backbone_mode in lite_transformer_modes:
                transformer_nhead = int(getattr(configs, "transformer_nhead", 8))
                transformer_num_layers = int(getattr(configs, "transformer_num_layers", 4))
                transformer_dim_feedforward = int(getattr(configs, "transformer_dim_feedforward", 2048))
                transformer_dropout = float(getattr(configs, "transformer_dropout", configs.dropout))
            else:
                transformer_nhead = configs.n_heads
                transformer_num_layers = configs.llm_layers
                transformer_dim_feedforward = max(configs.d_ff * 4, 128)
                transformer_dropout = configs.dropout
            if self.backbone_hidden_size % transformer_nhead != 0:
                raise ValueError(
                    f"Transformer hidden size {self.backbone_hidden_size} must be divisible by nhead={transformer_nhead}"
                )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.backbone_hidden_size,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                activation=configs.activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer_backbone = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_num_layers,
            )
            self.transformer_backbone_config = {
                "d_model": self.backbone_hidden_size,
                "nhead": transformer_nhead,
                "num_layers": transformer_num_layers,
                "dim_feedforward": transformer_dim_feedforward,
                "dropout": transformer_dropout,
                "frozen": self.global_backbone_mode in {"transformer_frozen", "transformer_frozen_lite"},
            }
            if self.transformer_backbone_config["frozen"]:
                for param in self.transformer_backbone.parameters():
                    param.requires_grad = False
        else:
            self.transformer_backbone = None
            self.transformer_backbone_config = None


        self.pre_llm_proj = nn.Linear(self.backbone_hidden_size, self.backbone_hidden_size)

        if configs.prompt_domain:
            self.description = """Dataset Description:
This dataset consists of multi-channel time series data collected from a traction motor drive system under various operating and fault conditions. Each sample contains 21 sensor channels, sampled at 64 kHz, covering two main signal types: tri-axial vibration and three-phase current.
"""

        else:
            self.description = getattr(
                configs,
                "content",
                "Industrial multichannel time-series fault diagnosis.",
            )

        self.moe_ffn = self._build_local_branch(configs)
        self.moe_alpha = configs.moe_alpha
        # ---- Component-specific heads ----
        self.component_names = ["motor", "gear", "LA", "RA"]
        self.component_dims = [5, 9, 5, 2]
        self.component_slices = [(0, 5), (5, 14), (14, 19), (19, 21)]
        self.motor_head = nn.Linear(self.backbone_hidden_size, 5)
        self.gear_head = nn.Linear(self.backbone_hidden_size, 9)
        self.LA_head = nn.Linear(self.backbone_hidden_size, 5)
        self.RA_head = nn.Linear(self.backbone_hidden_size, 2)

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        if self.llm_model is not None:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        else:
            self.word_embeddings = None
            self.vocab_size = 0
            self.num_tokens = 0
            self.mapping_layer = None

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.linear_reprogramming = nn.Linear(configs.d_model, self.backbone_hidden_size)
        self.mlp_reprogramming = nn.Sequential(
            nn.Linear(configs.d_model, max(configs.d_ff, configs.d_model)),
            nn.GELU(),
            nn.Linear(max(configs.d_ff, configs.d_model), self.backbone_hidden_size),
        )
        if self.reprogramming_mode == "frozen_random":
            for param in self.reprogramming_layer.parameters():
                param.requires_grad = False

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.output_projection = nn.Linear(self.d_ff, configs.num_classes)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.last_readout_report = None
        self.last_branch_outputs = None
        self.last_prompt_report = None
        self.last_backbone_report = None
        self.last_global_representation = None

    def _build_local_branch(self, configs):
        branch = LlamaMoEFeedForward(
            routing_variant=self.routing_variant,
            routing_manifest=self.routing_manifest,
            expert_head_permutation=self.expert_head_permutation,
            unified_hidden=getattr(configs, "unified_expert_hidden", None),
            logit_clip=getattr(configs, "logit_clip", 0.0),
        )
        local_mode = getattr(configs, "local_descriptor_mode", "legacy")
        if local_mode == "revised_rms":
            return RevisedRmsMoE(branch)
        if local_mode != "legacy":
            raise ValueError(f"Unknown local_descriptor_mode={local_mode}")
        return branch

    def _build_prompt_embeddings(self, x_moe, x_enc, device):
        B = x_moe.shape[0]
        mode = self.prompt_mode
        if mode == "none":
            return None
        if mode == "random":
            # Match a short prompt sequence while removing sample semantics.
            return torch.randn(B, 16, self.backbone_hidden_size, device=device, dtype=x_moe.dtype)
        if self.tokenizer is None or self.llm_model is None:
            # Text prompts are defined only for decoder-backed configurations.
            return None

        x_for_prompt = x_moe.mean(dim=2)  # [B,21]
        prompt_texts = []
        for b in range(B):
            energy = float((x_for_prompt[b].cpu().detach().numpy() ** 2).sum())
            if mode == "static":
                prompt_text = "<|start_prompt|>PHM multi-component sensor task. Identify motor/gear/axle faults.<|end_prompt|>"
            elif mode == "dynamic":
                prompt_text = f"<|start_prompt|>Energy={energy:.2f}.<|end_prompt|>"
            else:
                prompt_text = (
                    f"<|start_prompt|>PHM multi-component sensor task. "
                    f"Energy={energy:.2f}. Identify motor/gear/axle faults.<|end_prompt|>"
                )
            prompt_texts.append(prompt_text)

        untruncated = self.tokenizer(prompt_texts, padding=False, truncation=False).input_ids
        encoded_prompt = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        prompt = encoded_prompt.input_ids
        attention_mask = getattr(encoded_prompt, "attention_mask", None)
        token_counts = (
            attention_mask.sum(dim=1).tolist()
            if attention_mask is not None
            else [int(prompt.shape[1])] * B
        )
        self.last_prompt_report = {
            "mode": mode,
            "raw_texts": prompt_texts,
            "token_counts": [int(value) for value in token_counts],
            "untruncated_token_counts": [len(value) for value in untruncated],
            "sequence_length": int(prompt.shape[1]),
            "max_length": 256,
            "truncation_enabled": True,
            "truncation_occurred": any(len(value) > 256 for value in untruncated),
            "padding_enabled": True,
            "padding_occurred": any(int(value) < int(prompt.shape[1]) for value in token_counts),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "backbone": self.llm_backbone,
        }
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(device))
        if mode == "shuffled" and B > 1:
            prompt_embeddings = prompt_embeddings[torch.randperm(B, device=device)]
        return prompt_embeddings

    def _apply_reprogramming(self, enc_out):
        if self.reprogramming_mode in {"linear", "frozen_random"} and self.reprogramming_mode != "frozen_random":
            return self.linear_reprogramming(enc_out)
        if self.reprogramming_mode == "mlp":
            return self.mlp_reprogramming(enc_out)
        if self.reprogramming_mode == "frozen_random":
            if self.word_embeddings is None or self.mapping_layer is None:
                return self.linear_reprogramming(enc_out)
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            return self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        if self.word_embeddings is None or self.mapping_layer is None:
            return self.linear_reprogramming(enc_out)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        return self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

    def _run_global_backbone(self, llm_in):
        if self.global_backbone_mode == "none":
            return llm_in
        if self.global_backbone_mode in {
            "transformer_trainable", "transformer_frozen",
            "transformer_trainable_lite", "transformer_frozen_lite",
        }:
            return self.transformer_backbone(llm_in)

        if self.decoder_adapter is None:
            raise RuntimeError("Decoder adapter is not initialized")
        hidden_states = self.decoder_adapter.run(self.llm_model, llm_in)
        self.last_backbone_report = dict(self.decoder_adapter.last_forward_report)
        self.last_backbone_report.update({
            "repo_id": self.llm_repo_id,
            "revision": self.llm_revision,
            "model_path": self.llm_model_path,
            "sliding_window": getattr(self.llm_model.config, "sliding_window", None),
        })
        return hidden_states

    def _select_readout(self, llm_out, prompt_token_count, patch_token_count):
        if patch_token_count <= 0:
            raise RuntimeError("patch_token_count must be positive for classification readout")
        total_tokens = llm_out.shape[1]
        patch_start = int(prompt_token_count)
        patch_end = int(prompt_token_count + patch_token_count)
        if patch_start < 0 or patch_end > total_tokens:
            raise RuntimeError(
                f"Invalid readout range: prompt={prompt_token_count}, patch={patch_token_count}, total={total_tokens}"
            )

        mode = self.readout_mode
        if mode == "first_token_legacy":
            z = llm_out[:, 0, :]
            report = {
                "readout_mode": mode,
                "readout_type": "index",
                "readout_index": 0,
                "readout_source": "prompt" if prompt_token_count > 0 else "patch",
            }
        elif mode == "first_patch":
            z = llm_out[:, patch_start, :]
            report = {
                "readout_mode": mode,
                "readout_type": "index",
                "readout_index": patch_start,
                "readout_source": "patch",
            }
        elif mode == "mean_patch":
            z = llm_out[:, patch_start:patch_end, :].mean(dim=1)
            report = {
                "readout_mode": mode,
                "readout_type": "mean",
                "pooling_range": [patch_start, patch_end],
                "readout_source": "patch",
            }
        elif mode == "mean_all":
            z = llm_out.mean(dim=1)
            report = {
                "readout_mode": mode,
                "readout_type": "mean",
                "pooling_range": [0, total_tokens],
                "readout_source": "all",
            }
        else:
            raise ValueError(f"Unknown readout_mode={mode}")

        report.update({
            "prompt_token_count": int(prompt_token_count),
            "patch_token_count": int(patch_token_count),
            "llm_output_shape": list(llm_out.shape),
            "z_shape": list(z.shape),
        })
        self.last_readout_report = report
        return z

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,
                return_branch_outputs=False, intervention=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            return self.classify(
                x_enc,
                x_mark_enc,
                return_branch_outputs=return_branch_outputs,
                intervention=intervention,
            )
        else:
            raise NotImplementedError

    def classify(self, x_enc, x_mark_enc=None, return_attention=None,
                 return_branch_outputs=False, intervention=None):
        # ===== Preprocess =====
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, C = x_enc.shape

        # ---- convert to [B, C, T] ----
        x_enc = x_enc.permute(0, 2, 1).contiguous()

        need_details = bool(return_branch_outputs or intervention is not None)
        if need_details:
            recon, moe_logits, moe_details = self.moe_ffn(
                x_enc, return_details=True, intervention=intervention
            )
        else:
            recon, moe_logits = self.moe_ffn(x_enc)
            moe_details = None

        x_moe = recon

        if self.local_only:
            global_logits = torch.zeros_like(moe_logits)
            self.last_branch_outputs = {
                "final_logits": moe_logits,
                "global_logits": global_logits,
                "local_logits": moe_logits,
                "component_representations": (
                    moe_details["component_representations"] if moe_details is not None else None
                ),
                "routing_variant": self.routing_variant,
            }
            if return_branch_outputs:
                return self.last_branch_outputs
            return moe_logits, global_logits, moe_logits


        # ===== prompts =====
        prompt_embeddings = self._build_prompt_embeddings(x_moe, x_enc, x_enc.device)

        # Embed each channel independently before assembling global tokens.
        enc_out, n_vars = self.patch_embedding(x_moe)

        enc_out = self._apply_reprogramming(enc_out)

        C = 21
        Lp = enc_out.shape[1]
        D = enc_out.shape[2]


        enc_out = enc_out.view(B, C, Lp, D)

        tokens = enc_out.reshape(B, C * Lp, D)


        tokens = self.pre_llm_proj(tokens)  # [B, C*Lp, backbone_hidden_size]


        prompt_token_count = int(prompt_embeddings.shape[1]) if prompt_embeddings is not None else 0
        patch_token_count = int(tokens.shape[1])
        if prompt_embeddings is not None:
            llm_in = torch.cat([prompt_embeddings, tokens], dim=1)
        else:
            llm_in = tokens


        llm_out = self._run_global_backbone(llm_in)

        z = self._select_readout(llm_out, prompt_token_count, patch_token_count)
        self.last_global_representation = z.detach()

        # ===== Heads =====
        logit_motor = self.motor_head(z)
        logit_gear = self.gear_head(z)
        logit_LA = self.LA_head(z)
        logit_RA = self.RA_head(z)

        llm_logits = torch.cat([logit_motor, logit_gear, logit_LA, logit_RA], dim=-1)


        final_logits = self.moe_alpha * moe_logits + (1.0 - self.moe_alpha) * llm_logits

        self.last_branch_outputs = {
            "final_logits": final_logits,
            "global_logits": llm_logits,
            "local_logits": moe_logits,
            "component_representations": (
                moe_details["component_representations"] if moe_details is not None else None
            ),
            "routing_variant": self.routing_variant,
        }
        if return_branch_outputs:
            return self.last_branch_outputs
        return final_logits, llm_logits, moe_logits

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding, return_attention=False):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        out = self.out_projection(out)
        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
