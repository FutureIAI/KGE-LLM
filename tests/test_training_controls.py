from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from models.moe_ffn import LlamaMoEFeedForward
from models.revised_local import RevisedRmsMoE
from utils.losses import (
    AsymmetricLoss,
    ClassBalancedFocalLoss,
    DynamicFocalLoss,
    FocalLoss,
    PlainBCEWithLogitsLoss,
)
from utils.numerics import clip_logits, validate_logit_clip
from utils.training import classification_losses, optimizer_step, validate_max_grad_norm


@pytest.mark.parametrize("validator", [validate_logit_clip, validate_max_grad_norm])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_clipping_limits(validator, value):
    with pytest.raises(ValueError):
        validator(value)


def test_gradient_clipping_is_opt_in():
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.tensor([10.0])
    optimizer_step(optimizer, [parameter])
    assert parameter.item() == -10.0
    parameter.data.zero_()
    parameter.grad = torch.tensor([10.0])
    optimizer_step(optimizer, [parameter], max_grad_norm=1.0)
    assert parameter.item() == pytest.approx(-1.0, abs=1e-6)


def test_amp_unscales_before_optional_clipping():
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([20.0])
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    events = []

    class Scaler:
        def unscale_(self, optim):
            events.append("unscale")
            parameter.grad.div_(2)

        def step(self, optim):
            events.append("step")
            assert parameter.grad.item() == pytest.approx(1.0, abs=1e-6)
            optim.step()

        def update(self):
            events.append("update")

    optimizer_step(optimizer, [parameter], Scaler(), 1.0)
    assert events == ["unscale", "step", "update"]


@pytest.mark.parametrize(
    "loss_class",
    [PlainBCEWithLogitsLoss, FocalLoss, DynamicFocalLoss, ClassBalancedFocalLoss],
)
def test_unclipped_losses_keep_large_error_gradients(loss_class):
    logits = torch.tensor([[100.0, -100.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0]])
    loss = loss_class()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs() > 0).all()
    logits.grad.zero_()
    loss_class(logit_clip=10.0)(logits, targets).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "loss_class",
    [
        PlainBCEWithLogitsLoss,
        FocalLoss,
        DynamicFocalLoss,
        ClassBalancedFocalLoss,
        AsymmetricLoss,
    ],
)
def test_loss_clip_recovers_explicit_input_clamp(loss_class):
    logits = torch.tensor([[100.0, -100.0, 1.0, -1.0]])
    targets = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    assert torch.equal(
        loss_class(logit_clip=10.0)(logits, targets),
        loss_class()(logits.clamp(-10, 10), targets),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("detach_gamma", [True, False])
def test_dfl_extreme_correct_and_incorrect_predictions_are_finite(dtype, detach_gamma):
    logits = torch.tensor(
        [[1000.0, -1000.0, 1000.0, -1000.0]], dtype=dtype, requires_grad=True
    )
    targets = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=dtype)
    loss = DynamicFocalLoss(detach_gamma=detach_gamma)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("revised", [False, True])
def test_local_output_clipping_is_opt_in_and_state_compatible(revised):
    branch = LlamaMoEFeedForward()
    if revised:
        branch = RevisedRmsMoE(branch)
    with torch.no_grad():
        for expert in branch.experts:
            expert.mlp[2].weight.zero_()
            expert.mlp[2].bias.fill_(50.0)
    x = torch.randn(2, 21, 64)
    assert torch.equal(branch(x)[1], torch.full((2, 21), 50.0))
    limited = LlamaMoEFeedForward(logit_clip=20.0)
    if revised:
        limited = RevisedRmsMoE(limited)
    limited.load_state_dict(branch.state_dict(), strict=True)
    assert torch.equal(limited(x)[1], torch.full((2, 21), 20.0))
    assert clip_logits(x) is x


def model_config(**overrides):
    config = dict(
        task_name="classification",
        pred_len=0,
        seq_len=64,
        d_ff=8,
        llm_dim=16,
        patch_len=16,
        stride=8,
        d_model=8,
        n_heads=2,
        llm_layers=1,
        llm_model="LLAMA",
        enc_in=21,
        num_classes=21,
        prompt_domain=0,
        dropout=0.0,
        activation="gelu",
        moe_alpha=0.2,
        global_backbone_mode="none",
        prompt_mode="none",
        reprogramming_mode="linear",
        readout_mode="mean_patch",
        local_descriptor_mode="revised_rms",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_local_only_has_no_global_trainable_parameters():
    from models.KGLLMoE import Model

    model = Model(model_config())
    assert all(name.startswith("moe_ffn.") for name, _ in model.named_parameters())
    final, global_logits, local = model(torch.randn(2, 64, 21))
    assert torch.equal(final, local)
    assert not global_logits.count_nonzero()
    targets = torch.randint(0, 2, final.shape).float()
    criterion = PlainBCEWithLogitsLoss()
    losses = classification_losses(
        final, global_logits, local, targets, criterion, local_only=True
    )
    assert torch.equal(losses["total"], criterion(local, targets))


def test_tiny_random_frozen_llama_forward_and_interface_gradient(tmp_path):
    from models.KGLLMoE import Model
    from transformers import LlamaConfig

    class Tokenizer:
        eos_token = "</s>"

        def __call__(self, texts, **kwargs):
            ids = [[1, 2] for _ in texts]
            if kwargs.get("return_tensors") == "pt":
                ids = torch.tensor(ids)
            return SimpleNamespace(input_ids=ids)

    tiny = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=20,
    )
    with patch("models.KGLLMoE.LlamaConfig.from_pretrained", return_value=tiny), patch(
        "models.KGLLMoE.LlamaTokenizer",
        new=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
    ):
        model = Model(
            model_config(
                global_backbone_mode="llama_random_frozen",
                prompt_mode="full",
                reprogramming_mode="full",
                llm_model_path=str(tmp_path),
            )
        ).eval()
    logits = model(torch.randn(2, 64, 21))[0]
    assert logits.shape == (2, 21)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is None for p in model.llm_model.parameters())
    assert model.reprogramming_layer.query_projection.weight.grad is not None


def test_tiny_mistral_uses_official_forward():
    from models.frozen_decoder_backbone import (
        FrozenDecoderBackboneAdapter,
        FrozenDecoderBackboneSpec,
    )
    from transformers import MistralConfig, MistralModel

    config = MistralConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=20,
    )
    model = MistralModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter = FrozenDecoderBackboneAdapter(
        FrozenDecoderBackboneSpec(
            name="mistral",
            repo_id="",
            revision="",
            model_path="",
            active_layers=1,
            attention_implementation="eager",
            apply_final_norm=True,
            official_model_forward=True,
        )
    )
    inputs = torch.randn(2, 8, 16, requires_grad=True)
    output = adapter.run(model, inputs)
    expected = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
    assert torch.equal(output, expected)
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()


def test_validation_uses_the_training_total_objective():
    from utils.tools import vali_classify

    class FixedModel(nn.Module):
        local_only = False

        def forward(self, inputs):
            global_logits = inputs[:, 0, :]
            local_logits = inputs[:, 1, :]
            return 0.8 * global_logits + 0.2 * local_logits, global_logits, local_logits

    model = FixedModel()
    batch = {
        "data": torch.randn(3, 2, 21),
        "label": torch.randint(0, 2, (3, 21)).float(),
    }
    criterion = PlainBCEWithLogitsLoss()
    expected = classification_losses(*model(batch["data"]), batch["label"], criterion)[
        "total"
    ]
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), gather_for_metrics=lambda x: x
    )
    args = SimpleNamespace(use_amp=False)
    loss, _ = vali_classify(args, accelerator, model, None, [batch], criterion)
    assert loss == pytest.approx(expected.item())
