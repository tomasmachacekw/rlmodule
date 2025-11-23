from rlmodule.source.torch.rlmodel_cfg import BaseRLCfg, RLModelCfg, SharedRLModelCfg
from rlmodule.source.utils import get_output_size, get_space_size


def build_model(cfg: BaseRLCfg):

    # create a copy, because we are transforming the inputs
    network_cfg = cfg.network.copy()

    actions_as_input = hasattr(network_cfg, "input_actions") and network_cfg.input_actions is not None

    # process input size
    network_cfg.input_size = get_space_size(network_cfg.input_states)
    if actions_as_input:
        network_cfg.input_size += get_space_size(network_cfg.input_actions)

    # build base network of function approximator
    net = network_cfg.module(network_cfg)

    if actions_as_input:
        net.input_actions = True

    # get output size to be used as output layer input
    network_output_size = get_output_size(net, network_cfg.input_size)

    def build_output_layer(layer_cfg):
        return layer_cfg.class_type(device=cfg.device, input_size=network_output_size, cfg=layer_cfg)

    # build function approximator
    if isinstance(cfg, RLModelCfg):
        rl_model = cfg.class_type(device=cfg.device, network=net, output_layer=build_output_layer(cfg.output_layer))
    elif isinstance(cfg, SharedRLModelCfg):
        rl_model = cfg.class_type(
            device=cfg.device,
            network=net,
            policy_output_layer=build_output_layer(cfg.policy_output_layer),
            value_output_layer=build_output_layer(cfg.value_output_layer),
        )
    else:
        raise TypeError(f" Received unsupported class: '{type(cfg)}'.")

    return rl_model.to(cfg.device)
