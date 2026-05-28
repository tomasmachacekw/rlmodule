"""Unit tests for CNN Milestone A (flat-vector CNN, no SKRL changes).

These cover the :class:`CnnBackbone` presets and the
:class:`CnnMlp` composite module, which wires a CNN backbone into the same
MLP / ResidualMlp machinery used elsewhere in rlmodule.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rlmodule.source.network import CnnBackbone, CnnMlp, CnnProprioMlp
from rlmodule.source.network_cfg import (
    CnnBackboneCfg,
    CnnMlpCfg,
    CnnProprioMlpCfg,
    MlpCfg,
    ResidualMlpCfg,
)


@pytest.mark.parametrize("preset,input_states", [
    ("nature", (84, 84, 4)),
    ("impala", (64, 64, 3)),
])
def test_cnn_backbone_preset_forward(preset, input_states):
    cfg = CnnBackboneCfg(input_states=input_states, preset=preset)
    net = CnnBackbone(cfg)
    h, w, c = input_states
    flat = torch.randn(3, h * w * c)
    out = net(flat)
    assert out.dim() == 2
    assert out.shape[0] == 3
    assert out.shape[1] > 0


def test_cnn_backbone_custom_preset_requires_layers():
    with pytest.raises(ValueError):
        CnnBackbone(CnnBackboneCfg(input_states=(8, 8, 1), preset="custom"))


def test_cnn_backbone_invalid_preset_raises():
    with pytest.raises(ValueError):
        CnnBackbone(CnnBackboneCfg(input_states=(8, 8, 1), preset="does-not-exist"))


def test_cnn_backbone_custom_preset_uses_user_layers():
    cfg = CnnBackboneCfg(
        input_states=(13, 13, 1),
        preset="custom",
        layers=[nn.Conv2d(1, 16, kernel_size=3, stride=2), nn.ReLU(), nn.Flatten()],
    )
    net = CnnBackbone(cfg)
    flat = torch.randn(2, 13 * 13 * 1)
    out = net(flat)
    assert out.shape[0] == 2 and out.dim() == 2


def test_cnn_mlp_with_layer_norm_trunk():
    cfg = CnnMlpCfg(
        input_states=(84, 84, 4),
        cnn=CnnBackboneCfg(preset="nature"),
        mlp=MlpCfg(hidden_units=[128, 64], activation=nn.ELU, layer_norm=True),
    )
    net = CnnMlp(cfg)
    flat = torch.randn(2, 84 * 84 * 4)
    out = net(flat)
    assert out.shape == (2, 64)
    assert net.hidden_size == 64


def test_cnn_mlp_with_residual_trunk_backward():
    torch.manual_seed(0)
    cfg = CnnMlpCfg(
        input_states=(64, 64, 3),
        cnn=CnnBackboneCfg(preset="impala"),
        mlp=ResidualMlpCfg(
            hidden_size=128, num_blocks=2, activation=nn.ELU, layer_norm=True,
        ),
    )
    net = CnnMlp(cfg)
    flat = torch.randn(2, 64 * 64 * 3, requires_grad=True)
    out = net(flat)
    assert out.shape == (2, 128)

    out.sum().backward()
    # Every leaf parameter should receive a finite gradient.
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()


def test_cnn_mlp_input_size_override_is_respected():
    """If the cfg's flat input_size is given explicitly, the shape probe uses it."""
    cfg = CnnMlpCfg(
        input_states=(84, 84, 4),
        input_size=84 * 84 * 4,
        cnn=CnnBackboneCfg(preset="nature"),
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU),
    )
    net = CnnMlp(cfg)
    flat = torch.randn(1, 84 * 84 * 4)
    assert net(flat).shape == (1, 32)


def test_cnn_proprio_mlp_fuses_depth_and_vector_paths():
    cfg = CnnProprioMlpCfg(
        input_size=12 + 32 * 32,
        image_shape=(32, 32, 1),
        proprio_dim=12,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=32,
        mlp=MlpCfg(hidden_units=[64, 32], activation=nn.ELU, layer_norm=True),
    )
    net = CnnProprioMlp(cfg)
    x = torch.randn(4, 12 + 32 * 32)
    out = net(x)

    assert out.shape == (4, 32)
    assert net.hidden_size == 32
    assert any(isinstance(module, nn.LayerNorm) for module in net.visual_projection.modules())


def test_cnn_proprio_mlp_accepts_appended_critic_actions():
    action_dim = 9
    cfg = CnnProprioMlpCfg(
        input_size=12 + 32 * 32 + action_dim,
        image_shape=(32, 32, 1),
        proprio_dim=12,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=32,
        mlp=MlpCfg(hidden_units=[64, 32], activation=nn.ELU, layer_norm=True),
    )
    net = CnnProprioMlp(cfg)
    x = torch.randn(4, 12 + 32 * 32 + action_dim)
    out = net(x)

    assert out.shape == (4, 32)
    assert net.extra_input_dim == action_dim


def test_cnn_proprio_mlp_rejects_bad_split():
    cfg = CnnProprioMlpCfg(
        input_size=10,
        image_shape=(8, 8, 1),
        proprio_dim=3,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 8, 8)),
        mlp=MlpCfg(hidden_units=[16], activation=nn.ELU),
    )

    with pytest.raises(ValueError, match="input_size does not match"):
        CnnProprioMlp(cfg)


def test_cnn_proprio_mlp_backward_reaches_both_paths():
    torch.manual_seed(0)
    cfg = CnnProprioMlpCfg(
        input_size=6 + 32 * 32,
        image_shape=(32, 32, 1),
        proprio_dim=6,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=16,
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
    )
    net = CnnProprioMlp(cfg)
    x = torch.randn(3, 6 + 32 * 32, requires_grad=True)

    net(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad[:, :6]).all()
    assert torch.isfinite(x.grad[:, 6:]).all()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Visual encoder autocast / fp16 overflow regression tests
# ---------------------------------------------------------------------------
#
# History: the very first version of ``CnnProprioMlp`` always autocasted the
# visual encoder to ``torch.float16`` whenever the image lived on CUDA. That is
# unsafe for deep residual CNNs (Impala): late in training, intermediate
# activations in the visual projection's ``Linear`` can exceed the fp16
# representable range (~65504) and produce ``inf`` -> ``nan``, which then
# propagates through the rest of the policy and yields NaN actions for every
# env in lockstep.
#
# The cfg now exposes ``visual_amp_dtype`` and the default is
# ``torch.bfloat16``: 2-byte activation storage like ``float16`` (so we keep
# the original GPU-memory budget) but with ``float32``'s dynamic range (so we
# cannot overflow). ``None`` and ``torch.float16`` remain available; the
# tests below lock in the contract for all three.


def _make_cfg(image_shape=(32, 32, 1), proprio_dim=6, **overrides):
    h, w, c = image_shape
    cfg = CnnProprioMlpCfg(
        input_size=proprio_dim + h * w * c,
        image_shape=image_shape,
        proprio_dim=proprio_dim,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=16,
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_visual_amp_dtype_default_is_bfloat16():
    """Default config uses bf16 autocast: fp16-equivalent memory with fp32 range.

    Regression for two distinct issues:
    - ``ValueError: Action is nan for envs`` when the legacy default forced
      ``float16`` autocast on CUDA and Impala activations overflowed.
    - ``Warp CUDA error 2: out of memory`` when an earlier fix flipped the
      default to ``None`` (fp32) and doubled visual-encoder activation memory.
    """
    cfg = _make_cfg()
    assert cfg.visual_amp_dtype is torch.bfloat16
    net = CnnProprioMlp(cfg)
    assert net._visual_amp_dtype is torch.bfloat16


def test_visual_path_finite_with_default():
    """Forward pass on CPU produces finite outputs with the default cfg."""
    cfg = _make_cfg()
    net = CnnProprioMlp(cfg)
    x = torch.randn(2, cfg.input_size)
    out = net(x)
    assert torch.isfinite(out).all()


def test_visual_path_finite_with_explicit_none():
    """``visual_amp_dtype=None`` runs at the parameter dtype (fp32) and stays finite."""
    cfg = _make_cfg(visual_amp_dtype=None)
    net = CnnProprioMlp(cfg)
    x = torch.randn(2, cfg.input_size)
    out = net(x)
    assert torch.isfinite(out).all()
    assert net._visual_amp_dtype is None


def test_visual_path_handles_large_inputs_without_overflow_in_fp32():
    """``visual_amp_dtype=None`` must not overflow on large activations.

    Locks in numerical stability for the safest fallback path so a user
    switching to ``None`` for debugging never sees overflow.
    """
    torch.manual_seed(0)
    cfg = _make_cfg(visual_amp_dtype=None)
    net = CnnProprioMlp(cfg)
    x = torch.full((1, cfg.input_size), 50.0)
    out = net(x)
    assert torch.isfinite(out).all()


def test_visual_path_handles_large_inputs_without_overflow_in_bf16_on_cuda():
    """The default bf16 path also handles inputs that would overflow fp16.

    This is the primary positive guard: bf16's ``float32``-equivalent dynamic
    range means realistic mid-training activations never produce ``inf``.
    """
    if not torch.cuda.is_available():
        pytest.skip("bf16 autocast only triggers on CUDA tensors.")
    torch.manual_seed(0)
    cfg = _make_cfg()  # default visual_amp_dtype=torch.bfloat16
    net = CnnProprioMlp(cfg).cuda()
    # Scale weights up so a fp16 path would overflow; bf16 must stay finite.
    with torch.no_grad():
        for p in net.cnn.parameters():
            p.mul_(5.0)
        for p in net.visual_projection.parameters():
            p.mul_(5.0)
    x = torch.randn(2, cfg.input_size, device="cuda")
    out = net(x)
    assert torch.isfinite(out).all()


def test_visual_amp_dtype_fp16_logs_warning_on_nonfinite_output():
    """Opt-in fp16 autocast must surface a one-shot warning when the encoder
    produces a non-finite output, so the user can diagnose ``Action is nan``
    failures fast instead of staring at silent NaN actions for hours.
    """
    if not torch.cuda.is_available():
        pytest.skip("fp16 autocast warning only fires on CUDA tensors.")
    torch.manual_seed(0)
    cfg = _make_cfg(visual_amp_dtype=torch.float16)
    net = CnnProprioMlp(cfg).cuda()
    original = net.visual_projection

    class _InfProjection(nn.Module):
        def __init__(self, real):
            super().__init__()
            self.real = real

        def forward(self, x):
            out = self.real(x)
            return torch.full_like(out, float("inf"))

    net.visual_projection = _InfProjection(original)
    x = torch.randn(2, cfg.input_size, device="cuda")
    _ = net(x)
    assert net._nonfinite_visual_warned is True
    _ = net(x)
    assert net._nonfinite_visual_warned is True
