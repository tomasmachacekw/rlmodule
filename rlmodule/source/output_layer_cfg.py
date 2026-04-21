from typing import Optional, Sequence, Union

from dataclasses import MISSING
import gym
import gymnasium

import torch.nn as nn

from rlmodule.source.log_std_cfg import LogStdCfg, ParameterLogStdCfg
from rlmodule.source.network_cfg import NetworkCfg
from rlmodule.source.output_layer import DeterministicLayer, GaussianLayer


# use isaac-lab native configclass if available to avoid double declaration
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    from rlmodule.source.nvidia_utils import configclass


@configclass
class OutputLayerCfg:
    class_type: type[nn.Module] = MISSING
    """Descendant class of the OutputLayer class implementing corresponding layer."""

    output_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = None
    """Output size of the output layer. Usually an action (sub)space.
    Can be kept as None if the value is set during execution time.
    """

    output_activation: nn.Module = MISSING
    """Activation function for the output layer"""

    output_scale: float = 1.0
    """Scales output of layer by output scale"""

    clip_actions: bool = False
    """Flag to indicate whether the actions should be clipped. Only supported for gym/gymnasium space"""

    head: Optional[NetworkCfg] = None
    """Optional per-head network inserted between the trunk and the final projection.

    When this is ``None`` (default), the output layer is exactly a single ``Linear``
    followed by ``output_activation`` - i.e. the original behaviour, preserved for
    backwards compatibility.

    When this is set (typically to an :class:`rlmodule.source.network_cfg.MlpCfg`
    or :class:`rlmodule.source.network_cfg.ResidualMlpCfg`), it becomes a dedicated
    "head" MLP that runs between the shared trunk and the final output projection.
    This is the mechanism that lets a ``SharedRLModelCfg`` express:

        shared_trunk -> actor_head  -> Linear -> Gaussian
                     -> critic_head -> Linear -> Deterministic

    with independently configured per-layer LayerNorm and/or residual structure
    on each head (see the references on ``ResidualMlp`` for motivation).
    """


@configclass
class GaussianLayerCfg(OutputLayerCfg):
    """Keep configuration parameters for GaussianLayer class."""

    class_type: type[GaussianLayer] = GaussianLayer
    output_activation: type[nn.Module] = nn.Tanh

    reduction: str = "sum"
    """Reduction method for returning the log probability density function: (default: ``"sum"``).
    Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``,
    the log probability density function is returned as a tensor of shape ``(num_samples, num_actions)``
    instead of ``(num_samples, 1).
    """

    log_std: LogStdCfg = ParameterLogStdCfg()
    """Module to specify how standard deviation is computed."""


@configclass
class DeterministicLayerCfg(OutputLayerCfg):
    """Keep configuration parameters for DeterministicLayer class.
    Default values (output_size = 1, output_activation = Identity) are set for a use as value function predictor.
    """

    class_type: type[DeterministicLayer] = DeterministicLayer
    output_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = 1
    output_activation: nn.Module = nn.Identity
