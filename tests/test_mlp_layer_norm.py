"""Unit tests for per-layer LayerNorm on :class:`rlmodule.source.network.MLP`.

References (cited in module docstrings):
    * Ball et al., 2023 - https://arxiv.org/abs/2302.02948
    * Lyle et al., 2024 - https://arxiv.org/abs/2407.01800
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rlmodule.source.network import MLP, _resolve_per_layer_flags
from rlmodule.source.network_cfg import MlpCfg


def _count_layer_norms(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, nn.LayerNorm))


def test_mlp_default_has_no_layer_norm():
    """With ``layer_norm=False`` the module is exactly Linear -> Activation (back-compat)."""
    net = MLP(MlpCfg(input_size=10, hidden_units=[32, 16], activation=nn.ELU))
    # no LayerNorm at all
    assert _count_layer_norms(net) == 0
    # forward works
    y = net(torch.randn(4, 10))
    assert y.shape == (4, 16)
    # layer sequence is exactly [Linear, ELU, Linear, ELU]
    flat = [type(m).__name__ for m in net.mlp.modules() if not isinstance(m, (nn.Sequential, MLP))]
    assert flat == ["Linear", "ELU", "Linear", "ELU"]


def test_mlp_layer_norm_true_adds_ln_every_layer_pre_activation():
    net = MLP(MlpCfg(input_size=8, hidden_units=[16, 32, 8], activation=nn.ReLU, layer_norm=True))
    # one LayerNorm per hidden layer
    assert _count_layer_norms(net) == 3

    # pre-activation: LN appears *before* the ReLU inside each block
    for block in net.mlp:
        children = list(block.children())
        assert isinstance(children[0], nn.Linear)
        assert isinstance(children[1], nn.LayerNorm)
        assert isinstance(children[2], nn.ReLU)


def test_mlp_layer_norm_post_activation():
    net = MLP(MlpCfg(
        input_size=4, hidden_units=[8], activation=nn.Tanh,
        layer_norm=True, layer_norm_position="post",
    ))
    block = net.mlp[0]
    children = list(block.children())
    assert isinstance(children[0], nn.Linear)
    assert isinstance(children[1], nn.Tanh)
    assert isinstance(children[2], nn.LayerNorm)


def test_mlp_per_layer_layer_norm_flags():
    flags = [True, False, True, False]
    net = MLP(MlpCfg(
        input_size=4, hidden_units=[8, 8, 8, 8], activation=nn.ELU,
        layer_norm=flags,
    ))
    for i, (block, want) in enumerate(zip(net.mlp, flags)):
        has_ln = any(isinstance(m, nn.LayerNorm) for m in block.children())
        assert has_ln is want, f"block {i}: layer_norm should be {want}"


@pytest.mark.parametrize("bad_flag", [[True], [True, False, True], []])
def test_mlp_per_layer_length_mismatch_raises(bad_flag):
    with pytest.raises(ValueError):
        MLP(MlpCfg(
            input_size=4, hidden_units=[8, 8], activation=nn.ELU,
            layer_norm=bad_flag,
        ))


def test_mlp_invalid_layer_norm_position_raises():
    with pytest.raises(ValueError):
        MLP(MlpCfg(
            input_size=4, hidden_units=[8], activation=nn.ELU,
            layer_norm=True, layer_norm_position="bogus",
        ))


def test_mlp_empty_hidden_units_raises():
    with pytest.raises(ValueError):
        MLP(MlpCfg(input_size=4, hidden_units=[], activation=nn.ELU))


def test_resolve_per_layer_flags_helper():
    assert _resolve_per_layer_flags(True, 3) == [True, True, True]
    assert _resolve_per_layer_flags(False, 2) == [False, False]
    assert _resolve_per_layer_flags([True, False], 2) == [True, False]

    with pytest.raises(ValueError):
        _resolve_per_layer_flags([True], 2)

    with pytest.raises(TypeError):
        _resolve_per_layer_flags([1, 0], 2)  # type: ignore[list-item]
