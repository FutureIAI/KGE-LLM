import math
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from models.moe_ffn import LlamaMoEFeedForward
from models.revised_local import (
    RevisedRmsMoE,
    analytic_envelope_descriptor,
    rms_filter_response,
)
from utils.local_warmup import fit_local_descriptors, warm_up_local


class IdentityNormalization(nn.Module):
    def forward(self, x, mode):
        assert mode == "norm"
        return x


class RevisedLocalTests(unittest.TestCase):
    def test_analytic_envelope_of_sinusoid(self):
        time = torch.arange(64, dtype=torch.float32)
        signal = torch.sin(2 * math.pi * 4 * time / 64).reshape(1, 1, 64)
        descriptor = analytic_envelope_descriptor(signal, topk=3)
        self.assertAlmostEqual(float(descriptor[0, 0]), 1.0, places=5)

    def test_rms_response_is_nonnegative(self):
        expert = LlamaMoEFeedForward().experts[0]
        response = rms_filter_response(torch.randn(2, 9, 64), expert.wavelet)
        self.assertEqual(tuple(response.shape), (2, 9, 6))
        self.assertTrue(bool((response >= 0).all()))

    def test_full_partition_fit_warmup_and_checkpoint(self):
        torch.manual_seed(7)
        model = SimpleNamespace(
            moe_ffn=RevisedRmsMoE(LlamaMoEFeedForward()),
            normalize_layers=IdentityNormalization(),
        )
        train = [
            {"data": torch.randn(2, 64, 21), "label": torch.randint(0, 2, (2, 21)).float()}
            for _ in range(3)
        ]
        validation = [
            {"data": torch.randn(2, 64, 21), "label": torch.randint(0, 2, (2, 21)).float()}
        ]
        fit_local_descriptors(model, train, torch.device("cpu"))
        for index in range(4):
            self.assertTrue(bool(torch.isfinite(getattr(model.moe_ffn, f"std_{index}")).all()))
            self.assertTrue(bool((getattr(model.moe_ffn, f"std_{index}") >= 1e-6).all()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "local.pt"
            report = warm_up_local(
                model, train, validation, nn.BCEWithLogitsLoss(),
                torch.device("cpu"), 2, 1e-4, checkpoint_path,
            )
            self.assertEqual(len(report["history"]), 2)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"]
            restored = RevisedRmsMoE(LlamaMoEFeedForward())
            restored.load_state_dict(state, strict=True)
            _, logits = restored(train[0]["data"].transpose(1, 2))
            self.assertEqual(tuple(logits.shape), (2, 21))
            self.assertTrue(bool(torch.isfinite(logits).all()))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
    def test_full_model_interface_without_decoder_download(self):
        from models.KGLLMoE import Model

        config = SimpleNamespace(
            task_name="classification", pred_len=0, seq_len=64, d_ff=8,
            llm_dim=16, patch_len=16, stride=8, d_model=8, n_heads=2,
            llm_layers=1, llm_model="LLAMA", enc_in=21, num_classes=21,
            prompt_domain=0, dropout=0.0, activation="gelu", moe_alpha=0.2,
            global_backbone_mode="none", prompt_mode="none",
            reprogramming_mode="linear", readout_mode="mean_patch",
            local_descriptor_mode="revised_rms",
        )
        model = Model(config)
        input_batch = torch.randn(2, 64, 21)
        normalized = model.normalize_layers(input_batch, "norm").transpose(1, 2)
        model.moe_ffn.fit_standardization([normalized])
        final, global_branch, local_branch = model(input_batch)
        self.assertEqual(tuple(final.shape), (2, 21))
        self.assertTrue(torch.allclose(final, 0.8 * global_branch + 0.2 * local_branch))
        for index in range(4):
            self.assertIn(f"moe_ffn.mean_{index}", model.state_dict())
            self.assertIn(f"moe_ffn.std_{index}", model.state_dict())
        required = {name for name, param in model.named_parameters() if param.requires_grad}
        required.update(
            name for name, _ in model.named_buffers()
            if name.startswith("moe_ffn.mean_") or name.startswith("moe_ffn.std_")
        )
        saved = {name: value for name, value in model.state_dict().items() if name in required}
        restored = Model(config)
        incompatible = restored.load_state_dict(saved, strict=False)
        self.assertFalse(set(required) & set(incompatible.missing_keys))
        restored.eval()
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(input_batch)[0], restored(input_batch)[0]))
        global_before = model.linear_reprogramming.weight.detach().clone()
        small_train = [{
            "data": input_batch,
            "label": torch.randint(0, 2, (2, 21)).float(),
        }]
        with tempfile.TemporaryDirectory() as directory:
            report = warm_up_local(
                model, small_train, small_train, nn.BCEWithLogitsLoss(),
                torch.device("cpu"), 5, 1e-4, Path(directory) / "warmup.pt",
            )
        self.assertEqual(len(report["history"]), 5)
        self.assertTrue(torch.equal(global_before, model.linear_reprogramming.weight))


if __name__ == "__main__":
    unittest.main()
