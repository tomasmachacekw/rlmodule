from typing import Any, Sequence, Union

import logging

import torch
import torch.nn as nn

from rlmodule.source.utils import get_cnn_shape, get_output_size, get_space_size


logger = logging.getLogger("rlmodule")


LN_POSITION_PRE = "pre"
LN_POSITION_POST = "post"
_VALID_LN_POSITIONS = (LN_POSITION_PRE, LN_POSITION_POST)


def _resolve_per_layer_flags(flag: Union[bool, Sequence[bool]], n_layers: int) -> list[bool]:
    """Normalise a layer-wise toggle into a per-layer list of booleans.

    Accepts either a single boolean (applied uniformly to every layer) or a
    sequence of booleans whose length must match ``n_layers``. This lets callers
    express either "whole model" or "per layer" granularity with the same cfg
    field.
    """
    if isinstance(flag, bool):
        return [flag] * n_layers
    flags = list(flag)
    if len(flags) != n_layers:
        raise ValueError(
            f"Per-layer flag sequence length {len(flags)} does not match number of layers {n_layers}"
        )
    if not all(isinstance(v, bool) for v in flags):
        raise TypeError("Per-layer flag sequence must contain only booleans.")
    return flags


def _make_linear_block(
    in_features: int,
    out_features: int,
    activation_cls: type[nn.Module],
    layer_norm: bool,
    ln_position: str,
) -> nn.Sequential:
    """Build a single ``Linear -> (LayerNorm?) -> Activation`` block.

    Layer-norm placement defaults to pre-activation (``"pre"``) which matches
    the recommendations in:

    * Ball, Smith, Kostrikov, Levine. *Efficient online reinforcement learning
      with offline data.* 2023. https://arxiv.org/abs/2302.02948
    * Lyle, Zheng, Khetarpal, Martens, van Hasselt, Pascanu, Dabney.
      *Normalization and effective learning rates in reinforcement learning.*
      2024. https://arxiv.org/abs/2407.01800
    """
    if ln_position not in _VALID_LN_POSITIONS:
        raise ValueError(
            f"Invalid layer_norm_position={ln_position!r}; expected one of {_VALID_LN_POSITIONS}."
        )

    parts: list[nn.Module] = [nn.Linear(in_features, out_features)]
    if layer_norm and ln_position == LN_POSITION_PRE:
        parts.append(nn.LayerNorm(out_features))
    parts.append(activation_cls())
    if layer_norm and ln_position == LN_POSITION_POST:
        parts.append(nn.LayerNorm(out_features))
    return nn.Sequential(*parts)


class MLP(nn.Module):
    """Configurable multilayer perceptron module.

    Architecture is defined by providing modules_cfg.MlpCfg. Optionally inserts
    LayerNorm into each hidden block, which is useful for stabilising effective
    learning rates in deep RL policies (see references in ``_make_linear_block``).

    Examples::

        # Plain MLP (original behaviour, default).
        cfg = MlpCfg(
            input_size=517,
            hidden_units=[2048, 1024, 1024, 512],
            activation=nn.ELU,
        )

        # MLP with pre-activation LayerNorm on every hidden layer.
        cfg = MlpCfg(
            input_size=517,
            hidden_units=[2048, 1024, 1024, 512],
            activation=nn.ELU,
            layer_norm=True,
        )

        # MLP with per-layer LayerNorm toggles (e.g. last two layers only).
        cfg = MlpCfg(
            input_size=517,
            hidden_units=[2048, 1024, 1024, 512],
            activation=nn.ELU,
            layer_norm=[False, False, True, True],
        )
    """

    def __init__(self, cfg):
        super().__init__()

        hidden_units = list(cfg.hidden_units)
        n_layers = len(hidden_units)
        if n_layers == 0:
            raise ValueError("MlpCfg.hidden_units must contain at least one layer.")

        # Optional fields - fall back to plain-MLP defaults for older cfgs.
        layer_norm_flag = getattr(cfg, "layer_norm", False)
        ln_position = getattr(cfg, "layer_norm_position", LN_POSITION_PRE)
        ln_flags = _resolve_per_layer_flags(layer_norm_flag, n_layers)

        sizes = [cfg.input_size] + hidden_units
        blocks = [
            _make_linear_block(
                sizes[i],
                sizes[i + 1],
                cfg.activation,
                layer_norm=ln_flags[i],
                ln_position=ln_position,
            )
            for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*blocks)

    def forward(self, input):
        return self.mlp(input)


class _ResidualBlock(nn.Module):
    """Pre-LN residual block used by :class:`ResidualMlp` (SimBa-style).

    Block structure (fixed width ``W`` throughout)::

        x ──► LayerNorm ──► activation ──► Linear(W, W)
                                                │
              LayerNorm ──► activation ──► Linear(W, W) ──► + ──► out
                                                ▲           │
                                                └───────────┘   (skip connection)

    This is the canonical "pre-activation" residual MLP block used in recent
    RL architecture work (Simba, RLPD's critic) and strongly motivated by:

    * Ball, Smith, Kostrikov, Levine. *Efficient online reinforcement learning
      with offline data.* 2023. https://arxiv.org/abs/2302.02948
    * Lyle, Zheng, Khetarpal, Martens, van Hasselt, Pascanu, Dabney.
      *Normalization and effective learning rates in reinforcement learning.*
      2024. https://arxiv.org/abs/2407.01800

    The skip connection makes effective depth trainable; pre-activation
    LayerNorm stabilises the effective learning rate per block.
    """

    def __init__(self, width: int, activation_cls: type[nn.Module], layer_norm: bool):
        super().__init__()
        self.ln1: nn.Module = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.act1: nn.Module = activation_cls()
        self.fc1 = nn.Linear(width, width)
        self.ln2: nn.Module = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.act2: nn.Module = activation_cls()
        self.fc2 = nn.Linear(width, width)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.act1(self.ln1(input)))
        h = self.fc2(self.act2(self.ln2(h)))
        return input + h


class ResidualMlp(nn.Module):
    """Residual MLP with pre-activation LayerNorm blocks.

    A single hidden width ``hidden_size`` is used throughout; inputs are
    projected up to that width by ``input_proj`` and an optional ``final_ln``
    is applied to the trunk output. This lets you stack arbitrarily many
    residual blocks without suffering from vanishing-gradient or
    effective-learning-rate collapse, which is especially helpful for deep
    policy / value networks.

    Example::

        cfg = ResidualMlpCfg(
            input_size=517,
            hidden_size=512,
            num_blocks=6,
            activation=nn.ELU,
            layer_norm=True,
            final_layer_norm=True,
        )
        net = ResidualMlp(cfg)

    References: see :class:`_ResidualBlock`.
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg.num_blocks < 1:
            raise ValueError("ResidualMlpCfg.num_blocks must be >= 1.")
        if cfg.hidden_size < 1:
            raise ValueError("ResidualMlpCfg.hidden_size must be >= 1.")

        layer_norm_flag = getattr(cfg, "layer_norm", True)
        ln_flags = _resolve_per_layer_flags(layer_norm_flag, cfg.num_blocks)

        self.hidden_size = cfg.hidden_size
        self.input_proj = nn.Linear(cfg.input_size, cfg.hidden_size)
        self.blocks = nn.Sequential(
            *[
                _ResidualBlock(cfg.hidden_size, cfg.activation, layer_norm=ln_flags[i])
                for i in range(cfg.num_blocks)
            ]
        )
        self.final_ln: nn.Module = (
            nn.LayerNorm(cfg.hidden_size) if getattr(cfg, "final_layer_norm", True) else nn.Identity()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.final_ln(self.blocks(self.input_proj(input)))


# TODO annotation types to config
class RnnBase(nn.Module):
    """Base class for all modules containing RNN (including GRU, LSTM) substructure.

    All modules that have RNN substructure should inherit from this module.
    It makes sure all the necessary variables used by RL algorithm are defined.
    """

    def __init__(self, cfg):
        super().__init__()

        # parameters necessary for RNN, GRU and LSTM networks
        self.num_envs = cfg.num_envs
        self.num_layers = cfg.num_layers
        self.hidden_size = cfg.hidden_size
        self.sequence_length = cfg.sequence_length


class LSTM(RnnBase):
    """Configurable LSTM module.

    Architecture is defined by providing modules_cfg.LstmCfg.

    Contains implementation of forward pass handling hidden and cell states
    necessary for run with PPO_RNN algorithm.

    Example of use:

        cfg = LstmCfg(
            input_size = 517,
            num_envs = 2048,
            num_layers = 1,
            hidden_size = 512 + 256,
            sequence_length = 128,
        )
        net = LSTM(cfg)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )

    def forward(self, states, terminated, rnn_inputs):

        # Process rnn inputs
        hidden_states, cell_states = rnn_inputs[0], rnn_inputs[1]

        # Dimensions Explained
        # ---------------------
        # 'N'       - Batch size
        # 'L'       - Sequence length
        # 'Hin'     - Input size (number of features per input at each time step)
        # 'Hout'    - Hidden size (number of features in the hidden state of the LSTM)
        # 'HCell'   - Cell state size (same as hidden size in LSTM context)
        # 'D'       - Number of directions (1 for unidirectional, 2 for bidirectional)
        # 'num_layers' - Number of stacked LSTM layers

        # training
        if self.training:
            # reshape (N * L, Hin) -> (N, L, Hin)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

            # reshape (D * num_layers, N * L, Hout) -> (D * num_layers, N, L, Hout)
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])

            # reshape (D * num_layers, N * L, Hcell) -> (D * num_layers, N, L, Hcell)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])

            # get the hidden/cell states corresponding to the first time step of the sequence for each batch.
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    # Compute the RNN output for the segment of the sequence
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    # Reset hidden and cell states where sequences are terminated.
                    # Out-of-place masking (rather than `h[:, mask, :] = 0`) avoids
                    # mutating a tensor that will be saved by the next LSTM call's
                    # autograd save-for-backward; the gradient for the zeroed
                    # positions is still zero because (1 - mask) has zero grad.
                    mask = terminated[:, i1 - 1]
                    if torch.any(mask):
                        keep = (~mask).to(hidden_states.dtype).view(1, -1, 1)
                        hidden_states = hidden_states * keep
                        cell_states = cell_states * keep
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # if no termination reset is needed, simply compute the RNN output
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            # reshape (N, Hin) -> (N, 1, Hin)
            # This is done to add a sequence length dimension of 1 for processing one time step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # reshape (N, L, D * Hout) -> (N * L, D * Hout)
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": [rnn_states[0], rnn_states[1]]}


class RnnModule(RnnBase):
    """Common base class for RNN and GRU module.

    Contains implementation of forward pass handling hidden states
    necessary for run with PPO_RNN algorithm.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, states, terminated, rnn_inputs):

        # Process rnn inputs
        hidden_states = rnn_inputs[0]

        # Dimensions Explained
        # ---------------------
        # 'N'       - Batch size
        # 'L'       - Sequence length
        # 'Hin'     - Input size (number of features per input at each time step)
        # 'Hout'    - Hidden size (number of features in the hidden state of the RNN)
        # 'D'       - Number of directions (1 for unidirectional, 2 for bidirectional)
        # 'num_layers' - Number of stacked RNN layers

        # training
        if self.training:
            # reshape (N * L, Hin) -> (N, L, Hin)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

            # reshape (D * num_layers, N * L, Hout) -> (D * num_layers, N, L, Hout)
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])

            # get the hidden states corresponding to the first time step of the sequence for each batch.
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    # Compute the RNN output for the segment of the sequence
                    rnn_output, hidden_states = self.rnn(rnn_input[:, i0:i1, :], hidden_states)
                    # Reset hidden states where sequences are terminated.
                    # Out-of-place masking (rather than `h[:, mask, :] = 0`)
                    # avoids mutating a tensor that will be saved by the next
                    # RNN/GRU call's autograd save-for-backward.
                    mask = terminated[:, i1 - 1]
                    if torch.any(mask):
                        keep = (~mask).to(hidden_states.dtype).view(1, -1, 1)
                        hidden_states = hidden_states * keep
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # if no termination reset is needed, simply compute the RNN output
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            # reshape (N, Hin) -> (N, 1, Hin)
            # This is done to add a sequence length dimension of 1 for processing one time step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # reshape (N, L, D * Hout) -> (N * L, D * Hout)
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {"rnn": [hidden_states]}


class RNN(RnnModule):
    """Configurable RNN module.

    Architecture is defined by providing modules_cfg.RnnCfg.

    Contains implementation of forward pass handling hidden states
    necessary for run with PPO_RNN algorithm.

    Example of use:

        cfg = RnnCfg(
            input_size = 517,
            num_envs = 2048,
            num_layers = 1,
            hidden_size = 512 + 256,
            sequence_length = 128,
        )
        net = Rnn(cfg)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.rnn = nn.RNN(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )


class GRU(RnnModule):
    """Configurable GRU module.

    Architecture is defined by providing modules_cfg.GruCfg.

    Contains implementation of forward pass handling hidden states
    necessary for run with PPO_RNN algorithm.

    Example of use:

        cfg = GruCfg(
            input_size = 517,
            num_envs = 2048,
            num_layers = 1,
            hidden_size = 512 + 256,
            sequence_length = 128,
        )
        net = Gru(cfg)
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.rnn = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )


class RnnMlp(RnnBase):
    """Configurable module for Rnn-based module followed by MLP.

    Architecture is defined by providing modules_cfg.RnnMlpCfg,
    based on which the RNN, GRU or LSTM module will be used.

    Inherits implementation of forward pass handling hidden states
    necessary for run with PPO_RNN algorithm.

    1) Example of use (with RNN module):

        cfg = RnnMlpCfg(
                input_size = 517,
                rnn = RnnCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activation = nn.ELU,
                )
            )
        net = RnnMlp(cfg)

    2) Example of use (with GRU module):

        cfg = RnnMlpCfg(
                input_size = 517,
                rnn = GruCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activation = nn.ELU,
                )
            )
        net = RnnMlp(cfg)

    3) Example of use (with LSTM module):

        cfg = RnnMlpCfg(
                input_size = 517,
                rnn = LstmCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activation = nn.ELU,
                )
            )
        net = RnnMlp(cfg)
    """

    def __init__(self, cfg):
        super().__init__(cfg.rnn)
        # Copy nested cfgs before mutating ``input_size`` so we don't leak a
        # side-effect back into the user's configuration tree. ``configclass``
        # ``.copy()`` is a shallow copy (``dataclasses.replace``), so without
        # this the ``cfg.rnn`` / ``cfg.mlp`` objects reused across separate
        # policy/value builds would race mutations through shared references.
        rnn_cfg = cfg.rnn.copy()
        rnn_cfg.input_size = cfg.input_size
        self.rnn = rnn_cfg.module(rnn_cfg)
        mlp_cfg = cfg.mlp.copy()
        mlp_cfg.input_size = self.hidden_size
        self.mlp = MLP(mlp_cfg)

    def forward(self, states, terminated, rnn_inputs):
        rnn_output, output_dict = self.rnn(states, terminated, rnn_inputs)
        return self.mlp(rnn_output), output_dict


class RnnMlpWithForwardedInput(RnnBase):
    """Configurable module for Rnn-based module followed by MLP with
    fast-forward input into MLP.

    Architecture is defined by providing modules_cfg.RnnMlpCfg,
    based on which the RNN, GRU or LSTM module will be used.

    Inherits implementation of forward pass handling hidden states
    necessary for run with PPO_RNN algorithm.

    1) Example of use (with RNN module):

        cfg = RnnMlpCfg(
                input_size = 517,
                module = RnnMlpWithForwardedInput,
                rnn = RnnCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activation = nn.ELU,
                )
            )
        net = RnnMlp(cfg)

    2) Example of use (with GRU module):

        cfg = RnnMlpCfg(
                input_size = 517,
                module = RnnMlpWithForwardedInput,
                rnn = GruCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activation = nn.ELU,
                )
            )
        net = RnnMlp(cfg)

    3) Example of use (with LSTM module):

        cfg = RnnMlpCfg(
                input_size = 517,
                module = RnnMlpWithForwardedInput,
                rnn = LstmCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activation = nn.ELU,
                )
            )
        net = RnnMlp(cfg)

    Note: `module = RnnMlpWithForwardedInput` is not necessary for standalone example
        to work. However it is included here because it is necessary for use in rlmodule.build_model to
        differentiate it from RnnMlp class.
    """

    def __init__(self, cfg):
        super().__init__(cfg.rnn)
        # Copy nested cfgs before mutating ``input_size`` (see ``RnnMlp``
        # for the rationale).
        rnn_cfg = cfg.rnn.copy()
        rnn_cfg.input_size = cfg.input_size
        self.rnn = rnn_cfg.module(rnn_cfg)
        mlp_cfg = cfg.mlp.copy()
        mlp_cfg.input_size = cfg.input_size + self.hidden_size
        self.mlp = MLP(mlp_cfg)

    def forward(self, states, terminated, rnn_inputs):
        rnn_output, output_dict = self.rnn(states, terminated, rnn_inputs)
        mlp_input = torch.cat((states, rnn_output), dim=1)

        return self.mlp(mlp_input), output_dict


class _ImpalaResidualBlock(nn.Module):
    """ReLU-inside conv residual block used by the IMPALA CNN backbone.

    Block structure::

        x ──► ReLU ──► Conv3x3 ──► ReLU ──► Conv3x3 ──► (+ x)

    Reference: Espeholt et al., IMPALA (2018), https://arxiv.org/abs/1802.01561
    (Fig. 3, "large network").
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.relu(x)
        h = self.conv1(h)
        h = nn.functional.relu(h)
        h = self.conv2(h)
        return x + h


class _ImpalaStage(nn.Sequential):
    """A single IMPALA stage: Conv -> MaxPool -> 2 residual blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _ImpalaResidualBlock(out_channels),
            _ImpalaResidualBlock(out_channels),
        )


def _build_nature_cnn(in_channels: int) -> nn.Sequential:
    """Classic NatureCNN backbone (Mnih et al. 2015)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )


def _build_impala_cnn(in_channels: int, widths: Sequence[int] = (16, 32, 32)) -> nn.Sequential:
    """IMPALA residual CNN backbone (Espeholt et al. 2018)."""
    stages: list[nn.Module] = []
    c = in_channels
    for w in widths:
        stages.append(_ImpalaStage(c, w))
        c = w
    stages.append(nn.ReLU())
    stages.append(nn.Flatten())
    return nn.Sequential(*stages)


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax keypoints (Levine et al. 2016, "End-to-End Training of
    Deep Visuomotor Policies").

    For a ``(B, C, H, W)`` feature map, each channel is turned into a softmax
    distribution over its ``H * W`` positions and reduced to that distribution's
    expected image coordinates. The output is ``(B, 2 * C)``, flat:
    ``[x_0, y_0, x_1, y_1, ...]``, with ``x`` (columns) and ``y`` (rows) in
    ``[-1, 1]`` (pixel centres, -1 = left / top). It gives *where* a feature is,
    which average pooling blurs away for features only a few pixels wide.

    ``temperature`` divides the logits (lower = sharper, closer to a hard
    argmax). With ``learnable_temperature`` it is one trained scalar, stored as
    its log so it stays positive. The reduction runs in float32 whatever the
    input dtype, since a bf16 softmax over hundreds of positions loses the
    sub-pixel precision the keypoints are for.
    """

    def __init__(self, temperature: float = 1.0, learnable_temperature: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"SpatialSoftmax temperature must be > 0; got {temperature!r}.")
        log_t = torch.tensor(float(temperature)).log()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temperature", log_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"SpatialSoftmax expects (B, C, H, W); got shape {tuple(x.shape)}.")
        b, c, h, w = x.shape
        logits = x.float().reshape(b, c, h * w) / self.log_temperature.float().exp()
        attention = torch.softmax(logits, dim=-1)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device) if h > 1 else torch.zeros(1, device=x.device)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device) if w > 1 else torch.zeros(1, device=x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ex = (attention * grid_x.reshape(1, 1, -1)).sum(-1)
        ey = (attention * grid_y.reshape(1, 1, -1)).sum(-1)
        return torch.stack([ex, ey], dim=-1).reshape(b, 2 * c).to(x.dtype)


_NN_LAYER_REGISTRY: dict[str, type] = {
    "Conv2d": nn.Conv2d,
    "ELU": nn.ELU,
    "ReLU": nn.ReLU,
    "MaxPool2d": nn.MaxPool2d,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    # Keeps a small, strong response (a box a few pixels wide) that average
    # pooling would dilute with its neighbourhood.
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    # Per-channel expected (x, y) keypoints; its output is already flat.
    "SpatialSoftmax": SpatialSoftmax,
    "Flatten": nn.Flatten,
    "BatchNorm2d": nn.BatchNorm2d,
    # 3D variants for volumetric (e.g. occupancy-grid) backbones.
    "Conv3d": nn.Conv3d,
    "MaxPool3d": nn.MaxPool3d,
    "AdaptiveAvgPool3d": nn.AdaptiveAvgPool3d,
    "BatchNorm3d": nn.BatchNorm3d,
}


def _build_nn_layers_from_spec(layers_spec: list[dict]) -> list[nn.Module]:
    """Construct ``nn.Module`` instances from a JSON-serialisable layer list.

    Each dict must contain a ``"type"`` key matching one of the registered
    layer types. The remaining keys are forwarded as keyword arguments to
    the constructor. Values that are lists are converted to tuples where
    the constructor expects it (e.g. ``kernel_size``, ``output_size``).
    """
    modules: list[nn.Module] = []
    for spec in layers_spec:
        spec = dict(spec)
        layer_type = spec.pop("type")
        cls = _NN_LAYER_REGISTRY.get(layer_type)
        if cls is None:
            raise ValueError(
                f"Unknown layer type {layer_type!r} in custom CNN spec. "
                f"Supported: {sorted(_NN_LAYER_REGISTRY)}"
            )
        cleaned: dict[str, object] = {}
        for k, v in spec.items():
            if isinstance(v, list):
                v = tuple(v)
            cleaned[k] = v
        modules.append(cls(**cleaned))
    return modules


class CnnBackbone(nn.Module):
    """Flat-vector CNN backbone with named presets.

    Accepts a flat ``(B, H*W*C)`` observation, reshapes it internally to
    ``(B, C, H, W)`` (using the HWC-ordered ``input_states`` shape, same
    convention as the experimental :class:`CNN`), runs the chosen conv stack,
    and returns a flat feature vector suitable for an MLP head.

    Presets:

    * ``"nature"`` - NatureCNN from Mnih et al. 2015 (8/4 - 4/2 - 3/1 convs).
    * ``"impala"`` - IMPALA residual CNN from Espeholt et al. 2018
      (3 stages of conv+maxpool+2 residual blocks, widths [16, 32, 32]).
    * ``"custom"`` - build from ``cfg.layers`` exactly like the legacy
      :class:`CNN`.

    This is CNN "Milestone A" of the rlmodule roadmap: it still receives a
    flat-vector observation (as produced by skrl's current obs flattening)
    so it drops straight into ``SharedRLModel`` / ``SeparatedRLModel`` with
    **no** SKRL-side changes. First-class image obs is Milestone B.
    """

    def __init__(self, cfg):
        super().__init__()
        self._input_shape = get_cnn_shape(cfg.input_states)  # (H, W, C)
        h, w, c = self._input_shape

        preset = getattr(cfg, "preset", "custom")
        if preset == "nature":
            self.cnn = _build_nature_cnn(in_channels=c)
        elif preset == "impala":
            widths = tuple(getattr(cfg, "impala_widths", (16, 32, 32)))
            self.cnn = _build_impala_cnn(in_channels=c, widths=widths)
        elif preset == "custom":
            if cfg.layers is None or len(cfg.layers) == 0:
                raise ValueError(
                    "CnnBackboneCfg.preset='custom' requires an explicit 'layers' sequence."
                )
            self.cnn = nn.Sequential(*list(cfg.layers))
        else:
            raise ValueError(
                f"Unknown CNN preset {preset!r}; expected one of 'nature', 'impala', 'custom'."
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reshape a flat (B, H*W*C) input to (B, C, H, W). HWC->CHW permute mirrors
        # the experimental CNN's convention.
        x = input.view(-1, *self._input_shape).permute(0, 3, 1, 2).contiguous()
        return self.cnn(x)


class Cnn3dBackbone(nn.Module):
    """Flat-vector **3D** CNN backbone (the volumetric analogue of :class:`CnnBackbone`).

    Accepts a flat ``(B, D*H*W*C)`` observation, reshapes it internally to
    ``(B, C, D, H, W)`` (using the DHWC-ordered ``input_states`` shape — same
    channels-last convention as :class:`CnnBackbone`, extended by a leading
    depth axis), runs the chosen ``Conv3d`` stack, and returns a flat feature
    vector suitable for an MLP head.

    Unlike the 2D backbone there are no canned ``nature`` / ``impala`` presets
    (those reference 2D image architectures), so only ``preset="custom"`` is
    supported: build the conv stack from an explicit ``cfg.layers`` sequence
    (e.g. ``Conv3d``/``MaxPool3d``/``AdaptiveAvgPool3d``/``Flatten``).

    This mirrors the flat-vector contract of :class:`CnnBackbone` exactly, so it
    drops straight into ``SharedRLModel`` / ``SeparatedRLModel`` with no
    SKRL-side changes. It is the right primitive for occupancy grids and other
    small volumetric observations, where treating the grid as a stack of 2D
    channels would discard 3D locality.
    """

    def __init__(self, cfg):
        super().__init__()
        self._input_shape = get_cnn_shape(cfg.input_states)  # (D, H, W, C)
        if len(self._input_shape) != 4:
            raise ValueError(
                "Cnn3dBackboneCfg.input_states must be a 4-tuple in DHWC order "
                f"(D, H, W, C); got {self._input_shape!r}."
            )
        d, h, w, c = self._input_shape
        self._flat_size = int(d) * int(h) * int(w) * int(c)

        preset = getattr(cfg, "preset", "custom")
        if preset == "custom":
            if cfg.layers is None or len(cfg.layers) == 0:
                raise ValueError(
                    "Cnn3dBackboneCfg.preset='custom' requires an explicit 'layers' sequence."
                )
            self.cnn = nn.Sequential(*list(cfg.layers))
        else:
            raise ValueError(
                f"Unknown 3D CNN preset {preset!r}; only 'custom' is supported "
                "(2D 'nature'/'impala' presets do not apply to volumetric input)."
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reshape a flat (B, D*H*W*C) input to (B, C, D, H, W). DHWC->CDHW permute
        # mirrors the 2D backbone's HWC->CHW convention. Validate the flat size up
        # front so a layout/offset mismatch fails with a clear message instead of
        # an opaque ``.view()`` "shape is invalid for input of size" error.
        if input.shape[-1] != self._flat_size:
            raise ValueError(
                f"Cnn3dBackbone expected a flat input of size {self._flat_size} "
                f"(D*H*W*C for input_states={self._input_shape}), got "
                f"{input.shape[-1]}."
            )
        x = input.view(-1, *self._input_shape).permute(0, 4, 1, 2, 3).contiguous()
        return self.cnn(x)


class CnnMlp(nn.Module):
    """Convolutional backbone followed by an MLP (or ResidualMlp) trunk.

    The CNN runs first on the flat-vector observation and produces a flat
    feature vector whose size is auto-detected. That feature vector is then
    fed into the user-provided MLP / ResidualMlp trunk, so per-layer
    LayerNorm and residual structure work the same way as for the
    vanilla MLP trunks.

    Example::

        cfg = CnnMlpCfg(
            input_states=(84, 84, 4),
            cnn=CnnBackboneCfg(preset="impala"),
            mlp=MlpCfg(hidden_units=[512, 256], activation=nn.ELU, layer_norm=True),
        )
        net = CnnMlp(cfg)
    """

    def __init__(self, cfg):
        super().__init__()

        # Build the CNN first.
        cnn_cfg = cfg.cnn.copy()
        # Forward the observation shape so the CNN can reshape correctly.
        cnn_cfg.input_states = cfg.input_states
        # The cnn backbone does not consume a flat input_size, but set it for
        # consistency with the rest of the framework.
        cnn_cfg.input_size = getattr(cfg, "input_size", None)
        self.cnn = cnn_cfg.module(cnn_cfg)

        # Probe the CNN's output size against a flat dummy tensor so we know
        # how wide to make the MLP's input.
        flat_input = int(getattr(cfg, "input_size", 0) or 0)
        if flat_input <= 0:
            # Derive from the HWC shape.
            h, w, c = get_cnn_shape(cfg.input_states)
            flat_input = int(h) * int(w) * int(c)
        cnn_out_size = get_output_size(self.cnn, flat_input)

        mlp_cfg = cfg.mlp.copy()
        mlp_cfg.input_size = cnn_out_size
        self.mlp = mlp_cfg.module(mlp_cfg)
        # Expose the MLP output width in the same way ResidualMlp does, so
        # downstream shape probes (e.g. in `get_output_size`) keep working.
        self.hidden_size = get_output_size(self.mlp, cnn_out_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(input))


class CnnProprioMlp(nn.Module):
    """CNN encoder fused with leading vector proprioception.

    Input layout is ``[proprio, flattened_image]`` for actors and
    ``[proprio, flattened_image, action]`` for SAC-style critics. This keeps the
    environment and SKRL-facing state as one flat tensor while avoiding the
    common mistake of treating proprioception or actions as extra image pixels.
    """

    def __init__(self, cfg):
        super().__init__()
        self.image_shape = get_cnn_shape(cfg.image_shape)
        self.proprio_dim = int(cfg.proprio_dim)
        if self.proprio_dim <= 0:
            raise ValueError("CnnProprioMlpCfg.proprio_dim must be > 0.")

        h, w, c = self.image_shape
        self.image_flat_dim = int(h) * int(w) * int(c)
        expected_input = self.proprio_dim + self.image_flat_dim
        actual_input = int(getattr(cfg, "input_size", 0) or expected_input)
        if actual_input < expected_input:
            raise ValueError(
                "CnnProprioMlp input_size does not match proprio/image split: "
                f"input_size={actual_input}, proprio_dim={self.proprio_dim}, "
                f"image_flat_dim={self.image_flat_dim}."
            )
        self.extra_input_dim = actual_input - expected_input

        cnn_cfg = cfg.cnn.copy()
        cnn_cfg.input_states = self.image_shape
        cnn_cfg.input_size = self.image_flat_dim
        self.cnn = cnn_cfg.module(cnn_cfg)
        cnn_out_size = get_output_size(self.cnn, self.image_flat_dim)

        embedding_size = int(getattr(cfg, "cnn_embedding_size", 256))
        if embedding_size <= 0:
            raise ValueError("CnnProprioMlpCfg.cnn_embedding_size must be > 0.")
        activation_cls = getattr(cfg, "cnn_embedding_activation", nn.ELU)
        self.visual_projection = nn.Sequential(
            nn.Linear(cnn_out_size, embedding_size),
            nn.LayerNorm(embedding_size),
            activation_cls(),
        )

        mlp_cfg = cfg.mlp.copy()
        mlp_cfg.input_size = self.proprio_dim + self.extra_input_dim + embedding_size
        self.mlp = mlp_cfg.module(mlp_cfg)
        self.hidden_size = get_output_size(self.mlp, mlp_cfg.input_size)

        # Optional autocast dtype for the visual encoder forward pass. The
        # default (``torch.bfloat16``) keeps fp16-equivalent activation memory
        # while matching fp32's dynamic range, so deep residual CNNs do not
        # overflow mid-training. ``None`` runs at the parameter dtype (fp32),
        # which roughly doubles activation memory; ``torch.float16`` is the
        # legacy unsafe path that previously caused NaN actions on Impala
        # backbones once policy weights drifted enough for one batch's
        # activations to exceed fp16 range.
        self._visual_amp_dtype: torch.dtype | None = getattr(
            cfg, "visual_amp_dtype", torch.bfloat16
        )
        # We log a single warning if the visual encoder ever produces a
        # non-finite output (typically only seen with an explicit fp16 opt-in)
        # so the user gets a clear diagnostic instead of silent NaN actions.
        self._nonfinite_visual_warned = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        expected_input = self.proprio_dim + self.image_flat_dim + self.extra_input_dim
        if input.shape[-1] != expected_input:
            raise ValueError(
                f"Expected last input dim {expected_input}, "
                f"got {input.shape[-1]}."
            )
        proprio = input[..., : self.proprio_dim]
        image_start = self.proprio_dim
        image_end = image_start + self.image_flat_dim
        image = input[..., image_start:image_end]
        extra = input[..., image_end:]

        cnn_dtype = next(self.cnn.parameters()).dtype
        mlp_dtype = next(self.mlp.parameters()).dtype

        amp_dtype = self._visual_amp_dtype
        # autocast is only safe / useful on CUDA tensors.
        use_autocast = amp_dtype is not None and image.is_cuda

        if use_autocast and amp_dtype == torch.float16:
            # Mirror the legacy behavior: pre-cast image to fp16 so storage
            # matches the autocasted ops. For bf16 we leave the input at its
            # original dtype because PyTorch autocast handles the promotion
            # without overflow.
            image = image.to(dtype=torch.float16)
        else:
            image = image.to(dtype=cnn_dtype)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                visual = self.visual_projection(self.cnn(image))
        else:
            visual = self.visual_projection(self.cnn(image))

        if not self._nonfinite_visual_warned and not torch.isfinite(visual).all():
            logger.error(
                "[CnnProprioMlp] Non-finite values detected in visual encoder output "
                "(amp_dtype=%s). This is typically a float16 overflow inside the CNN "
                "or visual projection; switch CnnProprioMlpCfg.visual_amp_dtype to "
                "None or torch.bfloat16 to fix.",
                amp_dtype,
            )
            self._nonfinite_visual_warned = True

        fused = torch.cat(
            (
                proprio.to(dtype=mlp_dtype),
                extra.to(dtype=mlp_dtype),
                visual.to(dtype=mlp_dtype),
            ),
            dim=-1,
        )
        return self.mlp(fused)


def _require_graph_param(params: dict, key: str, node_id: str, node_type: str):
    """Return a required ``GraphNetwork`` node parameter or raise.

    Required parameters mirror the ``MISSING`` fields on the corresponding
    cfg dataclasses (e.g. ``MlpCfg.hidden_units``, ``RnnBaseCfg.num_envs``).
    Optional parameters are intentionally *not* funneled through here: they
    are simply omitted when absent so the cfg dataclass default applies,
    keeping each default defined in exactly one place (the cfg) instead of
    being re-hardcoded inside the executor.
    """
    if key not in params or params[key] is None:
        raise ValueError(
            f"GraphNetwork {node_type} node {node_id!r} requires params[{key!r}]."
        )
    return params[key]


class GraphNetwork(nn.Module):
    """Declarative graph executor for rlmodule primitives.

    ``GraphNetwork`` runs a DAG of pre-existing rlmodule primitives
    (MLP, ResidualMlp, LayerNorm, CNN backbone, RNN/GRU/LSTM, …) plus
    a small set of utility nodes (observation/action slicing,
    projection, concat, residual add). The cfg is JSON-friendly so it
    can be authored from the Cyberwave RL Config graph builder UI and
    persisted to the task's ``rl_config_spec``.

    Forward signature mirrors ``RnnBase`` modules so the graph can be
    dropped into ``Shared`` / ``Separated`` RL models that expect the
    ``(states, terminated, rnn_inputs)`` tuple. When the graph carries
    no recurrent node the forward returns ``(features, {})`` so the
    SKRL hidden-state plumbing stays a no-op.

    See :class:`rlmodule.source.network_cfg.GraphNetworkCfg` for the
    accepted node types and parameter shapes.
    """

    _SUPPORTED_NODE_TYPES = (
        "observation_input",
        "action_input",
        "mlp",
        "residual_mlp",
        "layer_norm",
        "activation",
        "projection",
        "concat",
        "residual_add",
        "cnn_backbone",
        "cnn3d_backbone",
        "rnn",
        "gru",
        "lstm",
    )

    def __init__(self, cfg):
        super().__init__()

        nodes_raw = list(getattr(cfg, "nodes", None) or [])
        edges_raw = list(getattr(cfg, "edges", None) or [])
        output_node_id = getattr(cfg, "output_node_id", None)
        if not nodes_raw:
            raise ValueError("GraphNetworkCfg.nodes must contain at least one node.")
        if not output_node_id:
            raise ValueError(
                "GraphNetworkCfg.output_node_id must point at a non-input node."
            )

        # Resolve flat-input / action-tail sizes.
        #
        # ``flat_input_size`` is the size of the observation-only prefix of
        # the flat state tensor; ``action_input_size`` is the size of the
        # appended action tail (used by SAC-style critics where
        # ``RLModel.forward`` concatenates ``[states, taken_actions]``).
        # We deliberately do not fall back to ``cfg.input_size`` for
        # ``flat_input_size`` here, because ``build_model`` overwrites
        # ``cfg.input_size`` with ``obs_dim + action_dim`` whenever
        # ``input_actions`` is configured. Using that total as the
        # observation prefix would cause every ``action_input`` node to
        # slice past the end of the tensor.
        flat_input_size = getattr(cfg, "flat_input_size", None)
        if not flat_input_size:
            input_states = getattr(cfg, "input_states", None)
            if input_states is not None:
                derived = get_space_size(input_states)
                if derived > 0:
                    flat_input_size = derived
        flat_input_size = int(flat_input_size or 0)

        action_input_size = getattr(cfg, "action_input_size", None)
        if not action_input_size:
            input_actions = getattr(cfg, "input_actions", None)
            if input_actions is not None:
                derived = get_space_size(input_actions)
                if derived > 0:
                    action_input_size = derived
        action_input_size = int(action_input_size or 0)

        # Pre-computed per-input slice metadata (kind/mode/start/length/buffer
        # name). Populated by ``_register_input_indices`` during build so
        # ``_slice_input`` can avoid recomputing whether each input's index
        # range is contiguous on every forward.
        self._input_specs: dict[str, dict[str, Any]] = {}

        # Normalise node configs.
        self._node_specs: dict[str, dict] = {}
        self._node_order: list[str] = []
        for node in nodes_raw:
            node_id = node.get("id")
            node_type = node.get("type")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("Every GraphNetwork node must have a string 'id'.")
            if node_type not in self._SUPPORTED_NODE_TYPES:
                raise ValueError(
                    f"GraphNetwork node {node_id!r} has unsupported type "
                    f"{node_type!r}. Supported: {self._SUPPORTED_NODE_TYPES}."
                )
            if node_id in self._node_specs:
                raise ValueError(f"GraphNetwork node id {node_id!r} is duplicated.")
            self._node_specs[node_id] = dict(node)
            self._node_order.append(node_id)

        # Compute adjacency / fan-in for shape inference and execution.
        self._fanin: dict[str, list[str]] = {nid: [] for nid in self._node_specs}
        adjacency: dict[str, list[str]] = {nid: [] for nid in self._node_specs}
        for edge in edges_raw:
            src = edge.get("from")
            dst = edge.get("to")
            if src not in self._node_specs or dst not in self._node_specs:
                raise ValueError(
                    f"GraphNetwork edge {edge!r} references unknown node ids."
                )
            adjacency[src].append(dst)
            self._fanin[dst].append(src)

        # Kahn's algorithm for topological order. We run it on the whole
        # graph (including any nodes that may end up being unreachable
        # from ``output_node_id``) so cycle detection still catches the
        # full structural problem before we prune.
        indegree = {nid: len(self._fanin[nid]) for nid in self._node_specs}
        queue = [nid for nid, deg in indegree.items() if deg == 0]
        topo: list[str] = []
        while queue:
            current = queue.pop(0)
            topo.append(current)
            for next_id in adjacency[current]:
                indegree[next_id] -= 1
                if indegree[next_id] == 0:
                    queue.append(next_id)
        if len(topo) != len(self._node_specs):
            raise ValueError("GraphNetwork contains a cycle.")

        if output_node_id not in self._node_specs:
            raise ValueError(
                f"GraphNetwork.output_node_id={output_node_id!r} is not a graph node."
            )
        self._output_node_id: str = output_node_id

        # Prune to the set of nodes that actually contribute to the
        # output. Nodes left dangling by the UI (alternate heads, debug
        # branches, etc.) would otherwise run on every forward and waste
        # both compute and activation memory; restricting execution to
        # ancestors of the output keeps the runtime predictable. Cycle
        # detection above has already validated the full graph, so this
        # is purely a hot-path optimisation.
        ancestors: set[str] = set()
        stack = [output_node_id]
        while stack:
            cur = stack.pop()
            if cur in ancestors:
                continue
            ancestors.add(cur)
            stack.extend(self._fanin[cur])
        self._active_nodes: set[str] = ancestors

        inactive = [nid for nid in topo if nid not in ancestors]
        if inactive:
            logger.info(
                f"GraphNetwork: {len(inactive)} node(s) not connected to "
                f"output {output_node_id!r}; they will be skipped: {inactive}"
            )

        active_topo: list[str] = [nid for nid in topo if nid in ancestors]

        # Build submodules in topological order for active nodes only. We
        # resolve each node's input dim from its inputs' inferred output
        # dims so MLP / residual / projection nodes can be authored
        # without manual input_size bookkeeping. Because the ancestor set
        # is closed under ``_fanin``, every fan-in node we touch here is
        # also active and has already been built earlier in ``active_topo``.
        self._modules_by_id: dict[str, nn.Module] = {}
        self._dim_by_id: dict[str, int] = {}
        self._is_recurrent: dict[str, bool] = {nid: False for nid in self._node_specs}
        self._recurrent_node_id: str | None = None
        for node_id in active_topo:
            spec = self._node_specs[node_id]
            module, out_dim = self._build_node(
                spec,
                flat_input_size=flat_input_size,
                action_input_size=action_input_size,
            )
            if module is not None:
                self.add_module(_module_attr_name(node_id), module)
                self._modules_by_id[node_id] = module
            self._dim_by_id[node_id] = int(out_dim)
            if spec["type"] in ("rnn", "gru", "lstm"):
                if self._recurrent_node_id is not None:
                    raise ValueError(
                        "GraphNetwork supports at most one recurrent node per "
                        "role (v1); got both "
                        f"{self._recurrent_node_id!r} and {node_id!r}."
                    )
                self._recurrent_node_id = node_id
                self._is_recurrent[node_id] = True

        self._exec_order: list[str] = active_topo

        # SKRL hidden-state spec; only meaningful when a recurrent
        # node is present. We mirror :class:`RnnBase` so the rlmodule
        # model adapter can read the same attributes for graph and
        # block trunks.
        if self._recurrent_node_id is not None:
            rec_module = self._modules_by_id[self._recurrent_node_id]
            self.num_envs = rec_module.num_envs
            self.num_layers = rec_module.num_layers
            self.hidden_size = rec_module.hidden_size
            self.sequence_length = rec_module.sequence_length
        else:
            self.num_envs = 0
            self.num_layers = 0
            self.hidden_size = int(self._dim_by_id[self._output_node_id])
            self.sequence_length = 0

    @property
    def output_dim(self) -> int:
        return int(self._dim_by_id[self._output_node_id])

    @property
    def has_recurrent(self) -> bool:
        return self._recurrent_node_id is not None

    # -----------------------------------------------------------------
    # build helpers
    # -----------------------------------------------------------------

    def _input_dim_for(self, node_id: str) -> int:
        """Return the concat-equivalent input dim for ``node_id``."""

        in_ids = self._fanin[node_id]
        if not in_ids:
            return 0
        return int(sum(self._dim_by_id[i] for i in in_ids))

    def _single_input_dim(self, node_id: str) -> int:
        in_ids = self._fanin[node_id]
        if len(in_ids) != 1:
            raise ValueError(
                f"GraphNetwork node {node_id!r} expects exactly 1 input; "
                f"got {len(in_ids)}."
            )
        return int(self._dim_by_id[in_ids[0]])

    def _build_node(
        self,
        spec: dict,
        *,
        flat_input_size: int,
        action_input_size: int,
    ):
        from rlmodule.source.network_cfg import (  # local import to avoid cycle
            Cnn3dBackboneCfg,
            CnnBackboneCfg,
            GruCfg,
            LstmCfg,
            MlpCfg,
            ResidualMlpCfg,
            RnnCfg,
        )

        node_id = spec["id"]
        params = spec.get("params") or {}
        node_type = spec["type"]

        if node_type == "observation_input":
            indices = list(params.get("indices") or [])
            dim = params.get("dim", len(indices))
            if dim <= 0:
                raise ValueError(
                    f"GraphNetwork observation_input {node_id!r} must have dim > 0."
                )
            if not indices:
                # Allow a contiguous default: take first ``dim`` features.
                indices = list(range(int(dim)))
            self._register_input_indices(node_id, indices, kind="observation")
            return None, int(dim)

        if node_type == "action_input":
            dim = int(params.get("dim", action_input_size))
            if dim <= 0:
                raise ValueError(
                    f"GraphNetwork action_input {node_id!r} must have dim > 0."
                )
            if flat_input_size <= 0:
                # SAC-style critics receive a concatenated ``[states,
                # taken_actions]`` tensor (see ``RLModel.forward``), so
                # the action tail starts at ``flat_input_size``. Without
                # that offset we would silently slice from the start of
                # the observation tensor.
                raise ValueError(
                    f"GraphNetwork action_input {node_id!r}: cannot determine "
                    "the action-tail offset because flat_input_size is "
                    "unknown. Set GraphNetworkCfg.flat_input_size or "
                    "GraphNetworkCfg.input_states."
                )
            indices = list(range(flat_input_size, flat_input_size + dim))
            self._register_input_indices(node_id, indices, kind="action")
            return None, dim

        if node_type == "mlp":
            in_dim = self._single_input_dim(node_id)
            hidden_units = list(_require_graph_param(params, "hidden_units", node_id, "mlp"))
            if not hidden_units:
                raise ValueError(
                    f"GraphNetwork mlp node {node_id!r} requires a non-empty "
                    "params['hidden_units']."
                )
            # Optional fields fall back to the MlpCfg dataclass defaults.
            optional = {k: params[k] for k in ("layer_norm", "layer_norm_position") if k in params}
            sub_cfg = MlpCfg(
                input_size=in_dim,
                hidden_units=hidden_units,
                activation=_require_graph_param(params, "activation", node_id, "mlp"),
                **optional,
            )
            module = MLP(sub_cfg)
            return module, int(sub_cfg.hidden_units[-1])

        if node_type == "residual_mlp":
            in_dim = self._single_input_dim(node_id)
            hidden_size = int(_require_graph_param(params, "hidden_size", node_id, "residual_mlp"))
            num_blocks = int(_require_graph_param(params, "num_blocks", node_id, "residual_mlp"))
            # Optional fields fall back to the ResidualMlpCfg dataclass defaults.
            optional = {k: params[k] for k in ("layer_norm", "final_layer_norm") if k in params}
            sub_cfg = ResidualMlpCfg(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_blocks=num_blocks,
                activation=_require_graph_param(params, "activation", node_id, "residual_mlp"),
                **optional,
            )
            module = ResidualMlp(sub_cfg)
            return module, hidden_size

        if node_type == "layer_norm":
            in_dim = self._single_input_dim(node_id)
            return nn.LayerNorm(in_dim), in_dim

        if node_type == "activation":
            in_dim = self._single_input_dim(node_id)
            activation_cls = _require_graph_param(params, "activation", node_id, "activation")
            return activation_cls(), in_dim

        if node_type == "projection":
            in_dim = self._single_input_dim(node_id)
            out_dim = int(_require_graph_param(params, "units", node_id, "projection"))
            return nn.Linear(in_dim, out_dim), out_dim

        if node_type == "concat":
            return None, self._input_dim_for(node_id)

        if node_type == "residual_add":
            in_ids = self._fanin[node_id]
            if len(in_ids) != 2:
                raise ValueError(
                    f"GraphNetwork residual_add {node_id!r} requires exactly 2 inputs."
                )
            d0 = self._dim_by_id[in_ids[0]]
            d1 = self._dim_by_id[in_ids[1]]
            if d0 != d1:
                raise ValueError(
                    f"GraphNetwork residual_add {node_id!r} inputs must have "
                    f"matching dimensions; got {d0} and {d1}."
                )
            return None, int(d0)

        if node_type == "cnn_backbone":
            in_dim = self._single_input_dim(node_id)
            image_shape = params.get("image_shape")
            if image_shape is None:
                raise ValueError(
                    f"GraphNetwork cnn_backbone {node_id!r} requires "
                    "params['image_shape'] = (H, W, C)."
                )
            # ``preset`` / ``impala_widths`` are optional: when absent we let
            # the CnnBackboneCfg dataclass defaults apply rather than
            # re-hardcoding them here.
            preset = params.get("preset")
            custom_layers = None
            if preset == "custom":
                layers_spec = params.get("layers")
                if not layers_spec:
                    raise ValueError(
                        f"GraphNetwork cnn_backbone {node_id!r} with "
                        "preset='custom' requires a 'layers' list."
                    )
                custom_layers = _build_nn_layers_from_spec(layers_spec)
            cfg_kwargs: dict[str, Any] = {
                "input_size": in_dim,
                "input_states": tuple(image_shape),
                "layers": custom_layers,
            }
            if preset is not None:
                cfg_kwargs["preset"] = preset
            if "impala_widths" in params:
                cfg_kwargs["impala_widths"] = tuple(params["impala_widths"])
            sub_cfg = CnnBackboneCfg(**cfg_kwargs)
            module = CnnBackbone(sub_cfg)
            # Probe the conv stack's flat output size.
            cnn_out = get_output_size(module, in_dim)
            return module, int(cnn_out)

        if node_type == "cnn3d_backbone":
            in_dim = self._single_input_dim(node_id)
            # Accept either ``volume_shape`` or ``image_shape`` for symmetry with
            # the 2D node; both must be a 4-dim DHWC tuple ``(D, H, W, C)``.
            volume_shape = params.get("volume_shape")
            if volume_shape is None:
                volume_shape = params.get("image_shape")
            if volume_shape is None:
                raise ValueError(
                    f"GraphNetwork cnn3d_backbone {node_id!r} requires "
                    "params['volume_shape'] = (D, H, W, C)."
                )
            volume_shape = tuple(volume_shape)
            if len(volume_shape) != 4:
                raise ValueError(
                    f"GraphNetwork cnn3d_backbone {node_id!r} volume_shape must "
                    f"have 4 dims (D, H, W, C); got {volume_shape!r}."
                )
            # 3D backbone only supports the 'custom' preset; require explicit layers.
            layers_spec = params.get("layers")
            if not layers_spec:
                raise ValueError(
                    f"GraphNetwork cnn3d_backbone {node_id!r} requires a 'layers' list "
                    "(only the 'custom' preset is supported for 3D)."
                )
            custom_layers = _build_nn_layers_from_spec(layers_spec)
            sub_cfg = Cnn3dBackboneCfg(
                input_size=in_dim,
                input_states=volume_shape,
                preset="custom",
                layers=custom_layers,
            )
            module = Cnn3dBackbone(sub_cfg)
            # Probe the conv stack's flat output size.
            cnn_out = get_output_size(module, in_dim)
            return module, int(cnn_out)

        if node_type in ("rnn", "gru", "lstm"):
            in_dim = self._single_input_dim(node_id)
            # RnnBaseCfg marks all four of these as required (MISSING), so the
            # graph node requires them too rather than inventing silent values.
            hidden_size = int(_require_graph_param(params, "hidden_size", node_id, node_type))
            num_layers = int(_require_graph_param(params, "num_layers", node_id, node_type))
            sequence_length = int(_require_graph_param(params, "sequence_length", node_id, node_type))
            num_envs = int(_require_graph_param(params, "num_envs", node_id, node_type))
            if node_type == "rnn":
                sub_cfg = RnnCfg(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    sequence_length=sequence_length,
                    num_envs=num_envs,
                )
                module = RNN(sub_cfg)
            elif node_type == "gru":
                sub_cfg = GruCfg(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    sequence_length=sequence_length,
                    num_envs=num_envs,
                )
                module = GRU(sub_cfg)
            else:
                sub_cfg = LstmCfg(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    sequence_length=sequence_length,
                    num_envs=num_envs,
                )
                module = LSTM(sub_cfg)
            return module, hidden_size

        raise ValueError(f"Unsupported GraphNetwork node type: {node_type!r}")

    def _register_input_indices(
        self, node_id: str, indices: list[int], *, kind: str
    ) -> None:
        """Pre-compute slice metadata for an ``observation_input`` /
        ``action_input`` node.

        Contiguous arithmetic ranges (step 1) take the ``"view"`` fast
        path in ``_slice_input``: ``Tensor.narrow`` returns a view on the
        original observation tensor with no copy. Arbitrary index lists
        fall back to ``index_select``, which copies into a fresh
        contiguous output.

        The view path is the common case in practice — flat
        ``[proprio, image, action]`` layouts produce contiguous slices —
        and avoids a per-forward copy of the (often image-sized) slice
        on the rollout / training hot path. The non-contiguous path
        keeps full expressivity for graphs that pick arbitrary feature
        positions.
        """
        if not indices:
            raise ValueError(
                f"GraphNetwork input node {node_id!r} requires a non-empty "
                "index list."
            )
        indices = [int(i) for i in indices]
        contiguous = all(
            indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)
        )
        if contiguous:
            self._input_specs[node_id] = {
                "kind": kind,
                "mode": "view",
                "start": indices[0],
                "length": len(indices),
            }
            return

        index_tensor = torch.tensor(indices, dtype=torch.long)
        buf_name = f"_idx_{_module_attr_name(node_id)}"
        # Register as buffer so ``.to(device)`` propagation works without
        # manual moves on every forward.
        self.register_buffer(buf_name, index_tensor, persistent=False)
        self._input_specs[node_id] = {
            "kind": kind,
            "mode": "gather",
            "start": indices[0],
            "length": len(indices),
            "buffer_name": buf_name,
        }

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def _slice_input(self, node_id: str, observation: torch.Tensor) -> torch.Tensor:
        spec = self._input_specs[node_id]
        if spec["mode"] == "view":
            return observation.narrow(-1, spec["start"], spec["length"])
        return observation.index_select(
            dim=-1, index=getattr(self, spec["buffer_name"])
        )

    def _gather_inputs(
        self, node_id: str, values: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        return [values[i] for i in self._fanin[node_id]]

    def forward(self, states, terminated=None, rnn_inputs=None):
        values: dict[str, torch.Tensor] = {}
        rnn_outputs: dict[str, Any] | None = None
        for node_id in self._exec_order:
            spec = self._node_specs[node_id]
            node_type = spec["type"]

            if node_type in ("observation_input", "action_input"):
                values[node_id] = self._slice_input(node_id, states)
                continue

            if node_type == "concat":
                inputs = self._gather_inputs(node_id, values)
                values[node_id] = torch.cat(inputs, dim=-1)
                continue

            if node_type == "residual_add":
                inputs = self._gather_inputs(node_id, values)
                values[node_id] = inputs[0] + inputs[1]
                continue

            module = self._modules_by_id[node_id]
            if node_type in ("rnn", "gru", "lstm"):
                input_tensor = self._gather_inputs(node_id, values)[0]
                output, info = module(input_tensor, terminated, rnn_inputs)
                values[node_id] = output
                rnn_outputs = info
                continue

            input_tensor = self._gather_inputs(node_id, values)[0]
            values[node_id] = module(input_tensor)

        out = values[self._output_node_id]
        # Only emit the RNN-style ``(features, rnn_outputs)`` tuple when
        # this graph actually carries a recurrent node — Model.forward
        # decides whether to unpack via ``self._rnn`` (structural
        # ``contains_rnn_module`` check), and a tuple from a
        # non-recurrent graph would otherwise be passed straight into
        # the output layer as if it were a feature tensor.
        if self._recurrent_node_id is not None:
            return out, rnn_outputs if rnn_outputs is not None else {}
        return out


def _module_attr_name(node_id: str) -> str:
    """Map a graph node id into a PyTorch-safe submodule attribute name."""

    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in node_id)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"node_{safe}"
    return f"node_{safe}"


class CNN(nn.Module):
    """Experimental implementation of Convolutional neural network.
    Architecture is defined by providing modules_cfg.MlpCfg.

    Warning: CNN is in an experimental phase. Some combination of inputs may not be yet supported.

    Example of use:

        cfg = CnnCfg(
            input_size = (13, 13, 1)  # input states in order (x,y,channels)
            layers=[
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten()
            ]
        )
        net = CNN(cfg)
    """

    def __init__(self, cfg):
        super().__init__()
        self._input_size = cfg.input_size

        self._input_shape = get_cnn_shape(cfg.input_states)

        modules = []
        for layer in cfg.layers:
            modules.append(layer)

        self.cnn = nn.Sequential(*modules)

    def forward(self, input):
        # Cnn expects inputs in shape (batch, channels, x, y)
        return self.cnn(input.view(-1, *self._input_shape).permute(0, 3, 1, 2))


# class TripleCnnAndMlp(nn.Module):
#     """
#     Split the observation space into parts and pass some parts through different CNNs.

#     ----------------------------------------------------------------------
#     |                       Network architecture  (cnn shape 1x13x13)    |
#     |--------------------------------------------------------------------|
#     |                                                                    |
#     |       10  1x13x13  1x13x13  1x13x13                                |
#     |       |    |         |         |                                   |
#     |       |   CNN0      CNN1      CNN2                                 |
#     |       |    |         |         |                                   |
#     |       | Flatten   Flatten   Flatten                                |
#     |       |    |         |         |                                   |
#     |       |____|_________|_________|                                   |
#     |       |                                                            |
#     |       Join to shape 10 + OutUnits0 + OutUnits1 + OutUnits2         |
#     |       |                                                            |
#     |       MLP: UNITS[0]                                                |
#     |       |                                                            |
#     |       ...                                                          |
#     |       |                                                            |
#     |       MLP: UNITS[-1]                                               |
#     |       |                                                            |
#     |       Output                                                       |
#     |                                                                    |
#     ----------------------------------------------------------------------

#     """

#     def __init__(self, cnn_params, mlp_params):
#         super().__init__()
#         self.input_shape = cnn_params['input_shape']

#         self.prefix_length = 10
#         self.cnn_number = 3

#         self.cnns = nn.ModuleList([CNN(cnn_params) for _ in range(self.cnn_number)])

#         mlp_params['input_size'] = self.prefix_length + self.cnn_number * get_output_size(
#             self.cnns[0], self.input_shape
#         )

#         self.mlp = MLP(mlp_params)

#     def forward(self, input):
#         # Split the input
#         prefix = input[:, : self.prefix_length]

#         c, h, w = self.input_shape
#         cnn_input = input[:, self.prefix_length :].view(-1, self.cnn_number * c, h, w)

#         # Forward pass through CNNs
#         cnn_outputs = []
#         for i in range(self.cnn_number):
#             cnn_output = self.cnns[i](cnn_input[:, i : i + 1, :, :]).view(input.size(0), -1)
#             cnn_outputs.append(cnn_output)

#         # Concatenate the outputs
#         combined = torch.cat([prefix] + cnn_outputs, dim=1)

#         # Forward pass through MLP
#         output = self.mlp(combined)
#         return output


# def triple_cnn_and_mlp_example():
#     cnn_params = {
#         'input_shape': [1, 13, 13],
#         'layers': [
#             {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1,
#              'out_channels': 32, 'activation': 'relu'},
#             {
#                 'type': 'conv',
#                 'kernel_size': 3,
#                 'stride': 1,
#                 'in_channels': 32,
#                 'out_channels': 64,
#                 'activation': 'relu',
#             },
#         ],
#     }

#     mlp_params = {'hidden_units': [1024, 1024, 512], 'activations': ['elu', 'elu', 'elu']}

#     return TripleCnnAndMlp(cnn_params, mlp_params)


# class CrazyNet(RnnBase):
#     def __init__(self, rnn_class, rnn_params, cnn_params, mlp_params):
#         super().__init__(rnn_params)

#         self.input_shape = cnn_params['input_shape']

#         self.prefix_length = 10
#         self.cnn_number = 3

#         self.cnns = nn.ModuleList([CNN(cnn_params) for _ in range(self.cnn_number)])

#         rnn_params['input_size'] = self.prefix_length + self.cnn_number * get_output_size(
#             self.cnns[0], self.input_shape
#         )

#         self.rnn = rnn_class(rnn_params)

#         mlp_params['input_size'] = self.rnn.hidden_size + self.rnn.input_size

#         self.mlp = MLP(mlp_params)

#     def forward(self, states, terminated, rnn_inputs):
#         # Split the input
#         prefix = states[:, : self.prefix_length]

#         c, h, w = self.input_shape
#         cnn_input = states[:, self.prefix_length :].view(-1, self.cnn_number * c, h, w)

#         # Forward pass through CNNs
#         cnn_outputs = []
#         for i in range(self.cnn_number):
#             cnn_output = self.cnns[i](cnn_input[:, i : i + 1, :, :]).view(states.size(0), -1)
#             cnn_outputs.append(cnn_output)

#         # Concatenate the outputs of prefix and CNNs
#         cnn_combined = torch.cat([prefix] + cnn_outputs, dim=1)

#         # Forward pass through RNN
#         rnn_output, output_dict = self.rnn(cnn_combined, terminated, rnn_inputs)

#         # Concatenate the outputs of RNN, CNNs and prefix
#         combined = torch.cat([cnn_combined, rnn_output], dim=1)

#         # Forward pass through MLP
#         output = self.mlp(combined)

#         return output, output_dict


# def crazy_net_example():
#     rnn_params = {
#         'num_envs': 2048,
#         'num_layers': 1,
#         'hidden_size': 512 + 256,
#         'sequence_length': 128,
#     }

#     cnn_params = {
#         'input_shape': [1, 13, 13],
#         'layers': [
#             {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1,
#              'out_channels': 32, 'activation': 'relu'},
#             {
#                 'type': 'conv',
#                 'kernel_size': 3,
#                 'stride': 1,
#                 'in_channels': 32,
#                 'out_channels': 64,
#                 'activation': 'relu',
#             },
#         ],
#     }

#     mlp_params = {'hidden_units': [1024, 1024, 512], 'activations': ['elu', 'elu', 'elu']}

#     return CrazyNet(LSTM, rnn_params, cnn_params, mlp_params)
