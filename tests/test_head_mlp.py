"""Unit tests for the optional per-head MLP in output layers.

The ``OutputLayerCfg.head`` field lets the user plug a dedicated MLP (or
ResidualMlp) between the shared trunk and the final projection. This is what
enables independent per-layer LayerNorm for the actor and critic heads in
a shared architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlmodule.source.network import MLP, ResidualMlp
from rlmodule.source.network_cfg import MlpCfg, ResidualMlpCfg
from rlmodule.source.output_layer import DeterministicLayer, GaussianLayer
from rlmodule.source.output_layer_cfg import DeterministicLayerCfg, GaussianLayerCfg
from rlmodule.source.log_std_cfg import ParameterLogStdCfg


def test_deterministic_head_identity_is_back_compat():
    layer = DeterministicLayer(
        device="cpu", input_size=16,
        cfg=DeterministicLayerCfg(output_size=1),
    )
    assert isinstance(layer._head, nn.Identity)
    # Final Linear should consume the trunk output size directly.
    last_linear = layer._net[0]
    assert isinstance(last_linear, nn.Linear)
    assert last_linear.in_features == 16


def test_deterministic_with_mlp_head_rewires_final_projection():
    layer = DeterministicLayer(
        device="cpu", input_size=64,
        cfg=DeterministicLayerCfg(
            output_size=1,
            head=MlpCfg(hidden_units=[32, 16], activation=nn.ELU, layer_norm=[True, False]),
        ),
    )
    assert isinstance(layer._head, MLP)
    # Final projection now takes the head output (16), not the trunk (64).
    last_linear = layer._net[0]
    assert isinstance(last_linear, nn.Linear)
    assert last_linear.in_features == 16

    # Forward produces correct output.
    x = torch.randn(5, 64)
    out, _, extras = layer(x, None, {})
    assert out.shape == (5, 1)
    assert extras == {}


def test_gaussian_with_residual_head():
    log_std_cfg = ParameterLogStdCfg(min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0)
    layer = GaussianLayer(
        device="cpu", input_size=128,
        cfg=GaussianLayerCfg(
            output_size=6,
            head=ResidualMlpCfg(
                hidden_size=64, num_blocks=2, activation=nn.ELU,
                layer_norm=True, final_layer_norm=True,
            ),
            log_std=log_std_cfg,
        ),
    )
    assert isinstance(layer._head, ResidualMlp)
    last_linear = layer._net[0]
    assert isinstance(last_linear, nn.Linear)
    assert last_linear.in_features == 64

    x = torch.randn(3, 128)
    actions, log_prob, extras = layer(x, None, {})
    assert actions.shape == (3, 6)
    assert log_prob.shape == (3, 1)


def test_gaussian_log_std_is_projected_from_head_when_head_is_set():
    """ParameterLogStd ignores input, but when NNLogStd is used the log_std
    network should see the *head* output dim, not the trunk input dim."""
    # This test exercises the path that feeds log_std with `self._projection_in`.
    log_std_cfg = ParameterLogStdCfg(min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0)
    layer = GaussianLayer(
        device="cpu", input_size=128,
        cfg=GaussianLayerCfg(
            output_size=4,
            head=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
            log_std=log_std_cfg,
        ),
    )
    # The internal log_std module was wired with input_size = head output (32).
    assert layer._log_std._input_size == 32
