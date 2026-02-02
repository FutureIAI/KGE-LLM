from math import sqrt
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
import os

from models.moe_ffn import LlamaMoEFeedForward

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
        self.channel_projection = nn.Linear(self.configs.d_model, 21)



        llama_path = os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'llama-7b')

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained(llama_path)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

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


        elif configs.llm_model == 'GPT2':
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
        elif configs.llm_model == 'BERT':
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
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.pre_llm_proj = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)

        if configs.prompt_domain:
            self.description = """Dataset Description:
This dataset consists of multi-channel time series data collected from a traction motor drive system under various operating and fault conditions. Each sample contains 21 sensor channels, sampled at 64 kHz, covering two main signal types: tri-axial vibration and three-phase current.
"""

        self.moe_ffn = LlamaMoEFeedForward()
        self.moe_alpha = configs.moe_alpha
        # ---- Component-specific heads ----
        self.motor_head = nn.Linear(self.llm_model.config.hidden_size, 9)  # motor = 9 classes
        self.gear_head = nn.Linear(self.llm_model.config.hidden_size, 6)  # gearbox = 6 classes
        self.LA_head = nn.Linear(self.llm_model.config.hidden_size, 3)  # left axle = 3 classes
        self.RA_head = nn.Linear(self.llm_model.config.hidden_size, 3)  # right axle = 3 classes

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

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

    def forward(self, x_enc, x_mark_enc=None,  x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            return self.classify(x_enc, x_mark_enc)
        else:
            raise NotImplementedError

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )


            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        enc_out, moe_scores = self.moe_ffn(enc_out)
        self.last_moe_scores = moe_scores  # for loss
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    import torch
    import torch.nn as nn

    def classify(self, x_enc, x_mark_enc=None, return_attention=None):
        # ===== Preprocess =====
        # Normalize the input data
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, C = x_enc.shape

        # ---- convert to [B, C, T] ----
        # Change data layout for convolutional or channel-wise operations
        x_enc_permuted = x_enc.permute(0, 2, 1).contiguous()

        recon, moe_logits = self.moe_ffn(x_enc_permuted)

        x_moe = recon

        # ===== Generate Dynamic Prompts =====
        # Based on the input data x_enc [B, T, C], calculate statistical metrics for the prompt.
        # Metrics are calculated along the time dimension (T).

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-6

        # Calculate metrics along the time dimension (dim=1)
        mu = torch.mean(x_enc, dim=1)  # Mean
        sigma = torch.std(x_enc, dim=1)  # Standard Deviation

        # Skewness: E[((X - mu) / sigma)^3]
        skew = torch.mean(((x_enc - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + epsilon)) ** 3, dim=1)

        # Kurtosis: E[((X - mu) / sigma)^4] - 3 (Fisher's definition)
        kurt = torch.mean(((x_enc - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + epsilon)) ** 4, dim=1) - 3

        x_rms = torch.sqrt(torch.mean(x_enc ** 2, dim=1))  # Root Mean Square

        # Peak-to-Peak: max(X) - min(X)
        x_ptp = torch.max(x_enc, dim=1).values - torch.min(x_enc, dim=1).values

        abs_x = torch.abs(x_enc)
        max_abs_x = torch.max(abs_x, dim=1).values
        mean_abs_x = torch.mean(abs_x, dim=1)

        # Crest Factor: max(|X|) / X_rms
        crest_factor = max_abs_x / (x_rms + epsilon)

        # Shape Factor: X_rms / mean(|X|)
        shape_factor = x_rms / (mean_abs_x + epsilon)

        # Impulse Factor: max(|X|) / mean(|X|)
        impulse_factor = max_abs_x / (mean_abs_x + epsilon)

        # Average the metrics across all channels (C) for a concise prompt
        mu_agg = mu.mean(dim=-1)
        sigma_agg = sigma.mean(dim=-1)
        skew_agg = skew.mean(dim=-1)
        kurt_agg = kurt.mean(dim=-1)
        x_rms_agg = x_rms.mean(dim=-1)
        x_ptp_agg = x_ptp.mean(dim=-1)
        crest_factor_agg = crest_factor.mean(dim=-1)
        shape_factor_agg = shape_factor.mean(dim=-1)
        impulse_factor_agg = impulse_factor.mean(dim=-1)

        prompt_texts = []
        for b in range(B):
            # Static part of the prompt
            p_static = self.description

            # Dynamic part of the prompt with calculated metrics
            p_dynamic = (
                f"Mean={mu_agg[b]:.2f}, Std={sigma_agg[b]:.2f}, Skewness={skew_agg[b]:.2f}, "
                f"Kurtosis={kurt_agg[b]:.2f}, RMS={x_rms_agg[b]:.2f}, PTP={x_ptp_agg[b]:.2f}, "
                f"CrestFactor={crest_factor_agg[b]:.2f}, ShapeFactor={shape_factor_agg[b]:.2f}, "
                f"ImpulseFactor={impulse_factor_agg[b]:.2f}. "
            )

            # Final combined prompt
            p_final = f"<|start_prompt|>{p_static}{p_dynamic}Identify motor/gear/axle faults.<|end_prompt|>"
            prompt_texts.append(p_final)

        prompt = self.tokenizer(prompt_texts, return_tensors="pt",
                                padding=True, truncation=True, max_length=256).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # ===== Patch embedding AFTER MoE =====
        enc_out, n_vars = self.patch_embedding(x_moe)

        # ===== Reprogramming (from Patch Dim to LLM Dim) =====
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # ===== LLM Fusion =====
        B = prompt_embeddings.shape[0]
        # C is already defined as x_enc.shape[2]
        Lp = enc_out.shape[1]  # Number of patches
        D = enc_out.shape[2]  # Patch embedding dim (from reprogram layer output)

        # Original enc_out: [B, Lp, D], but it should be processed per channel.
        # Reshape it back to: [B, C, Lp, D]
        enc_out = enc_out.view(B, C, Lp, D)

        # ---------------------------------------
        # Plan A: Expand (C * Lp) tokens.
        # No pooling, to preserve temporal and channel structure.
        # ---------------------------------------

        # Reshape dimensions to a token sequence: [B, C*Lp, D]
        tokens = enc_out.reshape(B, C * Lp, D)

        # Project to LLM hidden size (e.g., 4096)
        # You need to define self.pre_llm_proj = nn.Linear(D, llm_hidden_size) in __init__
        tokens = self.pre_llm_proj(tokens)  # Shape: [B, C*Lp, 4096]

        # Concatenate prompt tokens to form the final LLM input
        llm_in = torch.cat([prompt_embeddings, tokens], dim=1)
        # llm_in shape: [B, prompt_len + C*Lp, 4096]

        # ===== Pass to LLM =====
        hidden_states = llm_in
        for i, layer in enumerate(self.llm_model.layers):
            hidden_states = layer(hidden_states)[0]
        llm_out = hidden_states  # Shape: (B, Total_Tokens, 4096)

        # Still take the first token (usually the CLS token of the <prompt>) for classification
        z = llm_out[:, 0, :]

        # ===== Heads =====
        # Classification heads for different components
        logit_motor = self.motor_head(z)
        logit_gear = self.gear_head(z)
        logit_LA = self.LA_head(z)
        logit_RA = self.RA_head(z)

        llm_logits = torch.cat([logit_motor, logit_gear, logit_LA, logit_RA], dim=-1)

        # Combine MoE and LLM logits with a weighting factor alpha
        final_logits = self.moe_alpha * moe_logits + (1.0 - self.moe_alpha) * llm_logits

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