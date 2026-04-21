from typing import Sequence, Union

from collections.abc import Callable
from dataclasses import MISSING
import gym
import gymnasium

import torch.nn as nn

from rlmodule import logger
from rlmodule.source.network import (
    CNN,
    GRU,
    LSTM,
    MLP,
    RNN,
    CnnBackbone,
    CnnMlp,
    ResidualMlp,
    RnnBase,
    RnnMlp,
)


# use isaac-lab native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    logger.info("Importing local configclass.")
    from rlmodule.source.nvidia_utils import configclass


@configclass
class NetworkCfg:
    module: Union[nn.Module, Callable[..., nn.Module]] = MISSING
    input_size: int = None
    """Size of all network inputs. This field should be automatically derived from input_states/input_actions."""
    input_states: Union[int, Sequence[int], gym.Space, gymnasium.Space] = None
    """Observations passed to the network as an input."""
    input_actions: Union[int, Sequence[int], gym.Space, gymnasium.Space] = None
    """Action passed to the network as an input. If it is None, actions will not be passed."""


@configclass
class MlpCfg(NetworkCfg):
    """Configuration for :class:`rlmodule.source.network.MLP`.

    ``layer_norm`` enables LayerNorm inside each hidden block. It accepts either a
    single bool (applied uniformly) or a per-layer sequence. ``layer_norm_position``
    selects where the LayerNorm sits relative to the activation; ``"pre"`` is the
    default and matches the recommendations in

      * Ball et al., 2023 (https://arxiv.org/abs/2302.02948)
      * Lyle et al., 2024 (https://arxiv.org/abs/2407.01800)
    """

    module: type[MLP] = MLP

    hidden_units: Sequence[int] = MISSING
    activation: type[nn.Module] = MISSING

    layer_norm: Union[bool, Sequence[bool]] = False
    """Per-layer LayerNorm toggle.

    Accepts either:
      * a single ``bool``: same setting for every hidden layer (default: ``False``,
        preserves the pre-LN behaviour of older cfgs);
      * a sequence of ``bool`` with length == ``len(hidden_units)``: one toggle per
        hidden layer.
    """

    layer_norm_position: str = "pre"
    """Where to place the LayerNorm within each hidden block:

      * ``"pre"``  -> ``Linear -> LayerNorm -> Activation`` (recommended default)
      * ``"post"`` -> ``Linear -> Activation -> LayerNorm``
    """


@configclass
class ResidualMlpCfg(NetworkCfg):
    """Configuration for :class:`rlmodule.source.network.ResidualMlp` (SimBa-style).

    The residual trunk is a stack of ``num_blocks`` pre-LN residual blocks all
    operating at ``hidden_size``. Motivated by

      * Ball et al., 2023 (https://arxiv.org/abs/2302.02948)
      * Lyle et al., 2024 (https://arxiv.org/abs/2407.01800)
    """

    module: type[ResidualMlp] = ResidualMlp

    hidden_size: int = MISSING
    """Single hidden width used inside every residual block and at the trunk output."""

    num_blocks: int = MISSING
    """Number of stacked residual blocks."""

    activation: type[nn.Module] = MISSING
    """Activation class used inside each residual block (e.g. ``nn.ELU``)."""

    layer_norm: Union[bool, Sequence[bool]] = True
    """Per-block LayerNorm toggle, same semantics as ``MlpCfg.layer_norm``.

    Defaults to ``True`` because LayerNorm is considered core to the residual
    policy recipe referenced above.
    """

    final_layer_norm: bool = True
    """If True, apply a final LayerNorm on the trunk output before returning it."""


@configclass
class RnnBaseCfg(NetworkCfg):
    """Base configuration for RNN / GRU / LSTM trunks.

    ``num_envs`` MUST match the trainer's vector-env count. It is only
    consumed by :meth:`rlmodule.source.model.Model.get_specification` to
    size the SKRL-side hidden-state buffer
    (shape ``(num_layers, num_envs, hidden_size)``); the RNN ``forward``
    itself does not read it. A mismatched value silently produces a
    wrong-sized buffer and surfaces as an opaque shape error inside the
    LSTM/RNN forward during training.
    """

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


@configclass
class CnnBackboneCfg(NetworkCfg):
    """Configuration for :class:`rlmodule.source.network.CnnBackbone`.

    Exposes two named presets in addition to the existing "custom" path:

      * ``preset="nature"`` - NatureCNN (Mnih et al. 2015).
      * ``preset="impala"`` - IMPALA residual CNN (Espeholt et al. 2018).
      * ``preset="custom"`` - build from an explicit ``layers`` sequence
        (same semantics as :class:`CnnCfg`).

    ``input_states`` must be a HWC-ordered shape (``(H, W, C)``), matching
    the convention of :class:`CnnCfg`. Internally the module reshapes the
    flat observation to ``(B, C, H, W)``, runs the conv stack, and returns
    a flat feature vector suitable for an MLP head.

    This config corresponds to "CNN Milestone A" - it keeps the current
    SKRL flattened-obs convention so no SKRL-side changes are required.
    """

    module: type[CnnBackbone] = CnnBackbone

    preset: str = "nature"
    """Backbone preset: one of ``"nature"``, ``"impala"``, ``"custom"``."""

    layers: Sequence[nn.Module] = None
    """Explicit layers; only consulted when ``preset="custom"``."""

    impala_widths: Sequence[int] = (16, 32, 32)
    """Per-stage channel widths for the IMPALA preset."""


@configclass
class CnnMlpCfg(NetworkCfg):
    """Configuration for :class:`rlmodule.source.network.CnnMlp` (CNN trunk + MLP head).

    This is the recommended way to plug a CNN into a Shared / Separated RL
    model today (Milestone A): the CNN consumes a flat observation, and
    everything downstream is the same MLP / ResidualMlp machinery used
    elsewhere, so per-layer LayerNorm and residual structure on the
    post-conv trunk work out of the box.

    Example::

        CnnMlpCfg(
            input_states=(84, 84, 4),
            cnn=CnnBackboneCfg(preset="impala"),
            mlp=MlpCfg(hidden_units=[512, 256], activation=nn.ELU, layer_norm=True),
        )
    """

    module: type[CnnMlp] = CnnMlp

    cnn: CnnBackboneCfg = MISSING
    """Convolutional backbone (flat-vector input)."""

    mlp: NetworkCfg = MISSING
    """Trunk that runs on the CNN's flat feature vector. Typically an
    :class:`MlpCfg` or :class:`ResidualMlpCfg`."""
