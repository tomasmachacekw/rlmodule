"""Unit tests for CNN Milestone A (flat-vector CNN, no SKRL changes).

These cover the :class:`CnnBackbone` presets and the
:class:`CnnMlp` composite module, which wires a CNN backbone into the same
MLP / ResidualMlp machinery used elsewhere in rlmodule.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rlmodule.source.network import CnnBackbone, CnnMlp
from rlmodule.source.network_cfg import (
    CnnBackboneCfg,
    CnnMlpCfg,
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
