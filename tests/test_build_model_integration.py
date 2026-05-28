"""Integration tests exercising ``build_model`` with LN + residual cfgs.

These tests cover both :class:`SharedRLModelCfg` and :class:`SeparatedRLModelCfg`,
including the new per-head MLP wiring.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rlmodule.skrl.torch import (
    RLModelCfg,
    SeparatedRLModelCfg,
    SharedRLModelCfg,
    build_model,
)
from rlmodule.skrl.torch.network import MlpCfg, ResidualMlpCfg
from rlmodule.skrl.torch.output_layer import (
    DeterministicLayerCfg,
    GaussianLayerCfg,
    ParameterLogStdCfg,
)


@pytest.fixture(autouse=True)
def _reproducible():
    torch.manual_seed(0)


def _gaussian_head(output_size: int, head_cfg=None):
    return GaussianLayerCfg(
        output_size=output_size,
        head=head_cfg,
        log_std=ParameterLogStdCfg(min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0),
    )


def test_build_shared_plain_mlp_backcompat():
    """Existing MlpCfg without layer_norm still builds identical behaviour."""
    cfg = SharedRLModelCfg(
        network=MlpCfg(input_states=12, hidden_units=[32, 16], activation=nn.ELU),
        device="cpu",
        policy_output_layer=_gaussian_head(4),
        value_output_layer=DeterministicLayerCfg(output_size=1),
    )
    model = build_model(cfg)
    x = torch.randn(3, 12)
    actions, _ = model({"states": x}, role="policy")
    value, _ = model({"states": x}, role="value")
    assert actions.shape == (3, 4)
    assert value.shape == (3, 1)
    # no LayerNorms anywhere
    assert sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm)) == 0


def test_build_shared_residual_trunk_with_layer_norm_heads():
    cfg = SharedRLModelCfg(
        network=ResidualMlpCfg(
            input_states=12, hidden_size=64, num_blocks=3,
            activation=nn.ELU, layer_norm=True, final_layer_norm=True,
        ),
        device="cpu",
        policy_output_layer=_gaussian_head(
            4, head_cfg=MlpCfg(hidden_units=[32, 16], activation=nn.ELU, layer_norm=[True, False]),
        ),
        value_output_layer=DeterministicLayerCfg(
            output_size=1,
            head=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
        ),
    )
    model = build_model(cfg)

    # Trunk: input_proj + 3 blocks * 2 LN + final_ln = 7
    # Actor head: [True, False] -> +1
    # Critic head: [True] -> +1
    # Total: 9
    assert sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm)) == 9

    x = torch.randn(2, 12)
    actions, outputs = model({"states": x}, role="policy")
    log_prob = outputs["log_prob"]
    value, _ = model({"states": x}, role="value")
    assert actions.shape == (2, 4)
    assert log_prob.shape == (2, 1)
    assert value.shape == (2, 1)


def test_build_separated_mlp_ln_actor_residual_critic():
    cfg = SeparatedRLModelCfg(
        policy=RLModelCfg(
            network=MlpCfg(
                input_states=8, hidden_units=[32, 32, 32], activation=nn.ELU,
                layer_norm=[False, True, True],
            ),
            device="cpu",
            output_layer=_gaussian_head(2),
        ),
        value=RLModelCfg(
            network=ResidualMlpCfg(
                input_states=8, hidden_size=32, num_blocks=2, activation=nn.ELU,
                layer_norm=True, final_layer_norm=True,
            ),
            device="cpu",
            output_layer=DeterministicLayerCfg(output_size=1),
        ),
    )
    policy = build_model(cfg.policy)
    value = build_model(cfg.value)

    x = torch.randn(4, 8)
    a, outputs = policy({"states": x})
    lp = outputs["log_prob"]
    v, _ = value({"states": x})
    assert a.shape == (4, 2)
    assert lp.shape == (4, 1)
    assert v.shape == (4, 1)

    # Actor: 2 LNs (layers 1 and 2)
    actor_lns = sum(1 for m in policy.modules() if isinstance(m, nn.LayerNorm))
    assert actor_lns == 2

    # Critic: 2 blocks * 2 LNs + final_ln = 5
    critic_lns = sum(1 for m in value.modules() if isinstance(m, nn.LayerNorm))
    assert critic_lns == 5


def test_build_end_to_end_backward_pass():
    """A full forward+backward on a Shared model with LN+residuals converges gradients."""
    cfg = SharedRLModelCfg(
        network=ResidualMlpCfg(
            input_states=6, hidden_size=16, num_blocks=2,
            activation=nn.ELU, layer_norm=True,
        ),
        device="cpu",
        policy_output_layer=_gaussian_head(
            2, head_cfg=MlpCfg(hidden_units=[16], activation=nn.ELU, layer_norm=True),
        ),
        value_output_layer=DeterministicLayerCfg(
            output_size=1,
            head=MlpCfg(hidden_units=[16], activation=nn.ELU, layer_norm=True),
        ),
    )
    model = build_model(cfg)
    x = torch.randn(5, 6)
    actions, outputs = model({"states": x}, role="policy")
    log_prob = outputs["log_prob"]
    value, _ = model({"states": x}, role="value")
    # Fake PPO-ish loss
    loss = -(log_prob.mean()) + (value ** 2).mean()
    loss.backward()

    # Every leaf parameter should have a grad that's finite.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert not torch.isnan(p.grad).any(), f"{name} grad is NaN"
        assert not torch.isinf(p.grad).any(), f"{name} grad is Inf"
