from typing import Sequence, Union

import torch
import torch.nn as nn

from rlmodule.source.utils import get_cnn_shape, get_output_size


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
