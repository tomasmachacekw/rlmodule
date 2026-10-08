from typing import Any, Optional, Sequence, Union

from collections.abc import Callable
from dataclasses import MISSING, field
import gym
import gymnasium
import logging

import torch
import torch.nn as nn

from rlmodule.source.network import (
    CNN,
    GRU,
    LSTM,
    MLP,
    RNN,
    Cnn3dBackbone,
    CnnBackbone,
    CnnMlp,
    CnnProprioMlp,
    GraphNetwork,
    ResidualMlp,
    RnnBase,
    RnnMlp,
)


logger = logging.getLogger("rlmodule")


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
class Cnn3dBackboneCfg(NetworkCfg):
    """Configuration for :class:`rlmodule.source.network.Cnn3dBackbone`.

    The volumetric analogue of :class:`CnnBackboneCfg`. ``input_states`` must be
    a **DHWC**-ordered shape (``(D, H, W, C)``): a leading depth axis prepended
    to the 2D backbone's HWC convention. Internally the module reshapes the flat
    observation to ``(B, C, D, H, W)``, runs the ``Conv3d`` stack, and returns a
    flat feature vector suitable for an MLP head.

    Only ``preset="custom"`` is supported (the 2D ``"nature"`` / ``"impala"``
    presets reference image architectures that have no 3D counterpart here), so
    ``layers`` must be provided — e.g. a stack of ``Conv3d`` / ``MaxPool3d`` /
    ``AdaptiveAvgPool3d`` / ``Flatten`` modules.
    """

    module: type[Cnn3dBackbone] = Cnn3dBackbone

    preset: str = "custom"
    """Backbone preset. Only ``"custom"`` is supported for 3D."""

    layers: Sequence[nn.Module] = None
    """Explicit ``Conv3d`` layer stack; required for ``preset="custom"``."""


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


@configclass
class CnnProprioMlpCfg(NetworkCfg):
    """CNN image encoder fused with vector proprioception.

    The input is a single flat vector laid out as::

        [proprio_dim values, H * W * C image values]

    SAC-style critics may append actions after the image slice::

        [proprio_dim values, H * W * C image values, action values]

    This preserves the flat SKRL observation contract while giving the network a
    first-class CNN path for pixels/depth and a direct vector path for robot
    state.
    """

    module: type[CnnProprioMlp] = CnnProprioMlp

    image_shape: Sequence[int] = MISSING
    """Image shape in HWC order."""

    proprio_dim: int = MISSING
    """Number of leading vector/proprioception features before the image slice."""

    cnn: CnnBackboneCfg = MISSING
    """Convolutional backbone for the flat image slice."""

    cnn_embedding_size: int = 256
    """Size of the learned visual embedding used for fusion."""

    cnn_embedding_activation: type[nn.Module] = nn.ELU
    """Activation after the visual projection and LayerNorm."""

    mlp: NetworkCfg = MISSING
    """Post-fusion trunk, typically :class:`MlpCfg` with LayerNorm."""

    visual_amp_dtype: Optional[torch.dtype] = torch.bfloat16
    """Optional autocast dtype for the visual encoder forward pass.

    - ``torch.bfloat16`` (default): the recommended fast path. ``bfloat16``
      uses 2 bytes per activation - the same memory footprint as ``float16`` -
      while keeping ``float32``'s dynamic range, so it does not overflow on
      realistic activations the way ``float16`` does on deep residual CNNs
      (Impala and similar). This is what every modern CUDA GPU (Ampere+) is
      tuned for; pre-Ampere GPUs still execute it correctly, just without
      Tensor Core acceleration.
    - ``None``: run the visual encoder at the model's parameter dtype
      (typically ``torch.float32``). Safest numerically, but doubles activation
      memory vs ``bfloat16`` and is noticeably slower; reach for this only if
      ``bfloat16`` is somehow misbehaving.
    - ``torch.float16``: legacy behavior. **Not recommended** without an
      ``amp.GradScaler`` and gradient/activation clipping; leaving this on
      while training Impala-style residual CNNs is the most common cause of
      mid-training NaN actions on this stack (intermediate activations exceed
      ~65504, become ``inf``, then ``nan`` propagates to the policy output).
    """


# ---------------------------------------------------------------------------
# Graph network (declarative DAG of vector / recurrent / conv primitives)
# ---------------------------------------------------------------------------


@configclass
class GraphNetworkCfg(NetworkCfg):
    """Configuration for :class:`rlmodule.source.network.GraphNetwork`.

    ``GraphNetwork`` is a thin executor that runs a DAG of pre-existing
    rlmodule primitives (MLP, ResidualMlp, LayerNorm, CNN backbone,
    RNN/GRU/LSTM, …) plus a small set of utility nodes (observation
    slice, projection, concat, residual add). It is the runtime
    counterpart to the Cyberwave RL Config graph builder UI.

    Every node entry is a small JSON-friendly dict:

    .. code-block:: python

        {
            "id": "mlp_trunk",
            "type": "mlp",
            "params": {
                "hidden_units": [256, 256, 128],
                "activation": nn.ELU,
                "layer_norm": False,
                "layer_norm_position": "pre",
            },
        }

    Node connectivity is defined separately via ``edges`` (a list of
    ``{"from": ..., "to": ...}`` dicts), not by an ``inputs`` field on
    the node.

    Supported ``type`` values:

    - ``"observation_input"`` — declarative input node. Takes
      ``"indices"`` (a list of int positions in the flat observation
      tensor) and ``"dim"``. The graph slices the global flat input on
      forward.
    - ``"action_input"`` — same idea but slicing the appended action
      tail for SAC critics. Requires ``"dim"``.
    - ``"mlp"`` — wraps :class:`MlpCfg`.
    - ``"residual_mlp"`` — wraps :class:`ResidualMlpCfg`.
    - ``"layer_norm"`` — :class:`nn.LayerNorm` over the inferred width.
    - ``"activation"`` — single activation; ``params["activation"]``
      is the ``nn.Module`` subclass.
    - ``"projection"`` — :class:`nn.Linear` to ``params["units"]``.
    - ``"concat"`` — concatenates inputs along the last dim.
    - ``"residual_add"`` — adds inputs of matching dim.
    - ``"cnn_backbone"`` — wraps :class:`CnnBackboneCfg`. The matching
      ``observation_input`` must declare the image shape via
      ``params["image_shape"]``.
    - ``"rnn"`` / ``"gru"`` / ``"lstm"`` — wraps :class:`RnnCfg` /
      :class:`GruCfg` / :class:`LstmCfg`. At most one recurrent node
      per graph; the model adapter picks up the embedded
      :class:`RnnBaseCfg` for SKRL hidden-state sizing.

    The graph **does not** materialise output layers; the runtime
    adapter wires Gaussian / Deterministic output layers based on the
    user's actor/value/Q head intent. ``output_node_id`` therefore
    points at the trunk-feature node, not at a UI head node.
    """

    module: Any = GraphNetwork

    nodes: Sequence[dict[str, Any]] = field(default_factory=list)
    """Declarative node list. Order does not matter; execution order is
    derived from edges via topological sort."""

    edges: Sequence[dict[str, str]] = field(default_factory=list)
    """List of ``{"from": <node_id>, "to": <node_id>}`` edges."""

    output_node_id: str = MISSING
    """Trunk feature node id; the graph returns this node's output."""

    flat_input_size: Optional[int] = None
    """Observation-only prefix of the flat input tensor.

    When ``None`` this is derived from :attr:`input_states` via
    :func:`get_space_size`. Note we do **not** fall back to
    :attr:`input_size` because ``build_model`` overwrites that with
    ``obs_dim + action_dim`` whenever :attr:`input_actions` is set —
    using that total here would cause ``action_input`` nodes to slice
    past the action tail.
    """

    action_input_size: Optional[int] = None
    """Optional size of the action tail appended to the flat input.

    Only meaningful when at least one ``action_input`` node is present.
    When ``None`` this is derived from :attr:`input_actions` via
    :func:`get_space_size`.
    """
