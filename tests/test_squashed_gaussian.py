"""Tests for ``GaussianLayerCfg.squash`` (tanh-squashed Gaussian) and
``GaussianLayerCfg.entropy_reduction``.

An unbounded Gaussian head trained with SAC can grow ``std`` without limit
while the env clamps every sample to its bound: the entropy term keeps
rewarding a spread the env never executes. ``squash=True`` bounds the action
with ``tanh`` and makes ``log_prob`` the density of that bounded action.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.distributions import AffineTransform, Normal, TanhTransform, TransformedDistribution

from rlmodule.source.log_std_cfg import ParameterLogStdCfg
from rlmodule.source.output_layer import GaussianLayer
from rlmodule.source.output_layer_cfg import GaussianLayerCfg

N_IN, N_ACT = 6, 4


def _layer(**kw) -> GaussianLayer:
    torch.manual_seed(0)
    kw.setdefault("log_std", ParameterLogStdCfg(initial_log_std=1.5))
    cfg = GaussianLayerCfg(output_size=N_ACT, **kw)
    return GaussianLayer("cpu", N_IN, cfg)


def _squashed(**kw) -> GaussianLayer:
    return _layer(squash=True, output_activation=nn.Identity, **kw)


def test_squashed_actions_stay_within_output_scale() -> None:
    layer = _squashed(output_scale=2.0)
    actions, _ = layer(torch.randn(4096, N_IN) * 10, None, {})
    assert actions.abs().max() <= 2.0


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_squashed_log_prob_matches_torch_transformed_distribution(scale: float) -> None:
    layer = _squashed(output_scale=scale)
    x = torch.randn(256, N_IN)
    actions, out = layer(x, None, {})
    base = Normal(layer._net(x), layer._cached_log_std.exp())
    ref = TransformedDistribution(base, [TanhTransform(), AffineTransform(0.0, scale)])
    # Keep away from the bound, where the reference's atanh loses precision.
    ok = (actions.abs() < 0.99 * scale).all(dim=-1)
    expected = ref.log_prob(actions).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(out["log_prob"][ok], expected[ok], atol=1e-4, rtol=1e-4)


def test_squashed_log_prob_of_taken_actions_matches_the_sampled_one() -> None:
    # PPO-style re-evaluation: feeding the sampled action back must give the same density.
    layer = _squashed()
    x = torch.randn(512, N_IN)
    actions, out = layer(x, None, {})
    _, again = layer(x, actions, {})
    ok = (actions.abs() < 0.999).all(dim=-1)
    torch.testing.assert_close(again["log_prob"][ok], out["log_prob"][ok], atol=1e-3, rtol=1e-4)


def test_squashed_log_prob_is_finite_at_the_bound() -> None:
    layer = _squashed()
    x = torch.randn(3, N_IN)
    _, out = layer(x, torch.ones(3, N_ACT), {})
    assert torch.isfinite(out["log_prob"]).all()


def test_squashed_mean_action_is_tanh_of_the_mean() -> None:
    layer = _squashed(output_scale=2.0)
    x = torch.randn(8, N_IN)
    _, out = layer(x, None, {})
    torch.testing.assert_close(out["mean_actions"], 2.0 * torch.tanh(layer._net(x)))


def test_saturating_the_bound_lowers_the_entropy_estimate() -> None:
    # The SAC failure: without squashing, a huge std reads as huge entropy even though
    # the env executes +-1. With squashing, a very wide std piles samples at the bound,
    # where the density is high, so -log_prob falls instead of rising.
    x = torch.zeros(20000, N_IN)
    ent = {}
    for log_std in (0.0, 3.0):
        layer = _squashed(log_std=ParameterLogStdCfg(initial_log_std=log_std, max_log_std=4.0))
        _, out = layer(x, None, {})
        ent[log_std] = -out["log_prob"].mean().item()
    assert ent[3.0] < ent[0.0]


def test_squash_requires_identity_output_activation() -> None:
    with pytest.raises(ValueError, match="output_activation=nn.Identity"):
        _layer(squash=True)


def test_unsquashed_layer_is_unchanged() -> None:
    layer = _layer()
    x = torch.randn(64, N_IN)
    torch.manual_seed(1)
    actions, out = layer(x, None, {})
    mean = layer._net(x)
    ref = Normal(mean, layer._cached_log_std.exp()).log_prob(actions).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(out["log_prob"], ref)
    torch.testing.assert_close(out["mean_actions"], mean)


def test_entropy_reduction_defaults_to_per_dimension() -> None:
    layer = _layer()
    layer(torch.randn(5, N_IN), None, {})
    assert layer.get_entropy().shape == (5, N_ACT)


def test_entropy_reduction_sum_is_the_joint_entropy() -> None:
    layer = _layer(entropy_reduction="sum", log_std=ParameterLogStdCfg(initial_log_std=0.0))
    layer(torch.randn(5, N_IN), None, {})
    entropy = layer.get_entropy()
    assert entropy.shape == (5, 1)
    torch.testing.assert_close(entropy, torch.full((5, 1), N_ACT * 0.5 * math.log(2 * math.pi * math.e)))
