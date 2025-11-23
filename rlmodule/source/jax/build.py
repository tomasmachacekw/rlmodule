from __future__ import annotations

from copy import deepcopy

from rlmodule.source.jax.rlmodel_cfg import BaseRLCfg, RLModelCfg, SharedRLModelCfg
from .utils import get_output_size, get_space_size
from .model import RLModel, SharedRLModel


def build_model(cfg: BaseRLCfg):
    """Construct a JAX/Flax RL model matching the Torch builder semantics (MLP only)."""

    # create a copy; we'll mutate shapes
    network_cfg = deepcopy(cfg.network)

    actions_as_input = hasattr(network_cfg, "input_actions") and network_cfg.input_actions is not None

    # process input size
    network_cfg.input_size = get_space_size(network_cfg.input_states)
    if actions_as_input:
        network_cfg.input_size += get_space_size(network_cfg.input_actions)

    # build network (Flax module definition)
    net_def = network_cfg.module(network_cfg)

    # infer flat output size for the output layer
    network_output_size = get_output_size(net_def, network_cfg.input_size)

    # output layer factories (Flax modules). Create configured module instances.
    def build_output_layer(layer_cfg):
        # Mirror Torch: pass device, layer cfg, and input_size into the layer constructor
        return layer_cfg.class_type(device=cfg.device, input_size=network_output_size, cfg=layer_cfg)

    # Instantiate model class
    if isinstance(cfg, RLModelCfg):
        # observation space: from network inputs; action space: provided by the output layer cfg (set by caller)
        return RLModel(
            observation_space=network_cfg.input_states,
            action_space=cfg.output_layer.output_size,
            device=cfg.device,
            network=net_def,
            output_layer=build_output_layer(cfg.output_layer),
            input_actions=actions_as_input,
        )
    elif isinstance(cfg, SharedRLModelCfg):
        return SharedRLModel(
            observation_space=network_cfg.input_states,
            action_space=cfg.policy_output_layer.output_size,  # policy drives the action space
            device=cfg.device,
            network=net_def,
            policy_output_layer=build_output_layer(cfg.policy_output_layer),
            value_output_layer=build_output_layer(cfg.value_output_layer),
        )
    else:
        raise TypeError(f"Received unsupported class: '{type(cfg)}'.")

