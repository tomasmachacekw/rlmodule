"""Unit tests for :class:`rlmodule.source.network.ResidualMlp` (SimBa-style).

References:
    * Ball et al., 2023 - https://arxiv.org/abs/2302.02948
    * Lyle et al., 2024 - https://arxiv.org/abs/2407.01800
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rlmodule.source.network import ResidualMlp, _ResidualBlock
from rlmodule.source.network_cfg import ResidualMlpCfg


def _block_count(net: ResidualMlp) -> int:
    return sum(1 for m in net.blocks if isinstance(m, _ResidualBlock))


def test_residual_mlp_forward_shape():
    cfg = ResidualMlpCfg(input_size=12, hidden_size=32, num_blocks=3, activation=nn.ELU)
    net = ResidualMlp(cfg)
    y = net(torch.randn(7, 12))
    assert y.shape == (7, 32)
    assert net.hidden_size == 32
    assert _block_count(net) == 3


def test_residual_mlp_default_enables_layer_norm():
    """Default recipe (``layer_norm=True``) applies LN inside every block
    and also a trunk-final LayerNorm (``final_layer_norm=True``)."""
    cfg = ResidualMlpCfg(input_size=4, hidden_size=16, num_blocks=2, activation=nn.ELU)
    net = ResidualMlp(cfg)
    # 2 LNs per block + final_ln = 2*2 + 1 = 5
    assert sum(1 for m in net.modules() if isinstance(m, nn.LayerNorm)) == 5


def test_residual_mlp_per_block_layer_norm_flags():
    cfg = ResidualMlpCfg(
        input_size=4, hidden_size=16, num_blocks=3, activation=nn.ELU,
        layer_norm=[True, False, True], final_layer_norm=False,
    )
    net = ResidualMlp(cfg)
    for i, (block, want) in enumerate(zip(net.blocks, [True, False, True])):
        has_ln = isinstance(block.ln1, nn.LayerNorm)
        assert has_ln is want, f"block {i}: layer_norm should be {want}"
    assert isinstance(net.final_ln, nn.Identity)


def test_residual_mlp_gradient_flows_through_skip():
    torch.manual_seed(0)
    cfg = ResidualMlpCfg(input_size=8, hidden_size=16, num_blocks=4, activation=nn.ELU)
    net = ResidualMlp(cfg)
    x = torch.randn(3, 8, requires_grad=True)
    y = net(x)
    y.sum().backward()

    # input_proj receives gradient (coming through the residual path)
    assert net.input_proj.weight.grad is not None
    assert not torch.isnan(net.input_proj.weight.grad).any()
    assert not torch.isinf(net.input_proj.weight.grad).any()

    # every residual block's internal linears also receive gradient
    for block in net.blocks:
        assert not torch.isnan(block.fc1.weight.grad).any()
        assert not torch.isnan(block.fc2.weight.grad).any()


@pytest.mark.parametrize("n_blocks", [0, -1])
def test_residual_mlp_bad_num_blocks_raises(n_blocks):
    with pytest.raises(ValueError):
        ResidualMlp(ResidualMlpCfg(
            input_size=4, hidden_size=8, num_blocks=n_blocks, activation=nn.ELU,
        ))


def test_residual_mlp_bad_hidden_size_raises():
    with pytest.raises(ValueError):
        ResidualMlp(ResidualMlpCfg(
            input_size=4, hidden_size=0, num_blocks=1, activation=nn.ELU,
        ))


def test_residual_mlp_identity_when_lns_off():
    """When LN is fully off the block is still a valid residual (no NaNs)."""
    cfg = ResidualMlpCfg(
        input_size=4, hidden_size=8, num_blocks=2, activation=nn.ELU,
        layer_norm=False, final_layer_norm=False,
    )
    net = ResidualMlp(cfg)
    # no LayerNorms anywhere
    assert sum(1 for m in net.modules() if isinstance(m, nn.LayerNorm)) == 0
    y = net(torch.randn(2, 4))
    assert y.shape == (2, 8)
    assert not torch.isnan(y).any()
