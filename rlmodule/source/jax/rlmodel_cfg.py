from __future__ import annotations

from dataclasses import MISSING

import flax.linen as nn
import jax

from rlmodule.source.torch.output_layer_cfg import GaussianLayerCfg, OutputLayerCfg
from rlmodule.source.nvidia_utils import configclass



@configclass
class BaseRLCfg:
    class_type: type = MISSING
    network: nn.Module = MISSING  # Flax/Linen module (definition configured by `network` cfg)
    device: str | jax.Device | None = MISSING


@configclass
class RLModelCfg(BaseRLCfg):
    class_type: type = MISSING  # will be assigned to RLModel in __init__.py
    output_layer: OutputLayerCfg = GaussianLayerCfg()


@configclass
class SharedRLModelCfg(BaseRLCfg):
    """Policy and value share the function approximator; separate output heads."""

    class_type: type = MISSING  # will be assigned to SharedRLModel in __init__.py
    policy_output_layer: OutputLayerCfg = MISSING
    value_output_layer: OutputLayerCfg = MISSING


@configclass
class SeparatedRLModelCfg:
    """Convenience container for two independent models."""

    policy: RLModelCfg = MISSING
    value: RLModelCfg = MISSING

