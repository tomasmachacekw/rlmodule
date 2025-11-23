
from collections.abc import Callable, Sequence
from dataclasses import MISSING
import gym
import gymnasium

import torch.nn as nn

from rlmodule import logger
from rlmodule.source.torch.network import CNN, GRU, LSTM, MLP, RNN, RnnBase, RnnMlp
from rlmodule.source.nvidia_utils import configclass


@configclass
class NetworkCfg:
    module: nn.Module | Callable[..., nn.Module] = MISSING
    input_size: int = None
    """Size of all network inputs. This field should be automatically derived from input_states/input_actions."""
    input_states: int | Sequence[int] | gym.Space | gymnasium.Space = None
    """Observations passed to the network as an input."""
    input_actions: int | Sequence[int] | gym.Space | gymnasium.Space = None
    """Action passed to the network as an input. If it is None, actions will not be passed."""


@configclass
class MlpCfg(NetworkCfg):
    module: type[MLP] = MLP

    hidden_units: Sequence[int] = MISSING
    activation: type[nn.Module] = MISSING


@configclass
class RnnBaseCfg(NetworkCfg):
    num_envs: int = MISSING
    num_layers: int = MISSING
    hidden_size: int = MISSING
    sequence_length: int = MISSING


@configclass
class RnnCfg(RnnBaseCfg):
    module: type[RNN] = RNN


@configclass
class GruCfg(RnnBaseCfg):
    module: type[GRU] = GRU


@configclass
class LstmCfg(RnnBaseCfg):
    module: type[LSTM] = LSTM


@configclass
class RnnMlpCfg(NetworkCfg):
    module: type[RnnBase] = RnnMlp

    rnn: RnnBaseCfg = MISSING
    mlp: MlpCfg = MISSING


@configclass
class CnnCfg(NetworkCfg):
    """Config for convolutional neural network.
    For network to work properly the input states need to be specified as a triple in order (x,y,channels)
    Warning: CNN is in an experimental phase. Some combination of inputs may not be yet supported.
    """

    module: type[CNN] = CNN

    layers: Sequence[nn.Module] = MISSING
    """Layers of convolutional neural network."""
