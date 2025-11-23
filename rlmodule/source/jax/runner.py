from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from collections.abc import Mapping

from . import build_model
from rlmodule.source.jax.rlmodel_cfg import SeparatedRLModelCfg, SharedRLModelCfg
from skrl.agents.jax.a2c import A2C, A2C_DEFAULT_CONFIG
from skrl.agents.jax.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.jax import Wrapper, MultiAgentEnvWrapper
from skrl.memories.jax import RandomMemory
from skrl.models.jax import Model
from skrl.multi_agents.jax.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.multi_agents.jax.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.utils.runner.jax import Runner as SkrlRunner


def _override_field(cfg: dataclass, field_name: str, new_value):
    """Recursively override all fields named field_name in given config to the new_value."""
    if hasattr(cfg, field_name):
        setattr(cfg, field_name, new_value)
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if is_dataclass(value):
            _override_field(value, field_name, new_value)


def _set_common_model_fields(model_cfg, env: Wrapper, input_actions: bool = False):
    model_cfg.network.input_states = env.observation_space
    if input_actions:
        model_cfg.network.input_actions = env.action_space
        model_cfg.network.input_states = env.observation_space
    model_cfg.device = env.unwrapped.device


def _set_implicitly_derived_fields(model_cfg, env: Wrapper | MultiAgentEnvWrapper, algorithm: str):
    """Set fields that can be implicitly derived from env at runtime."""
    if isinstance(model_cfg, SharedRLModelCfg):
        _set_common_model_fields(model_cfg, env)
        model_cfg.policy_output_layer.output_size = env.action_space
    elif isinstance(model_cfg, SeparatedRLModelCfg):
        _set_common_model_fields(model_cfg.value, env, input_actions=(algorithm in ["SAC", "DDPG"]))
        _set_common_model_fields(model_cfg.policy, env)
        model_cfg.policy.output_layer.output_size = env.action_space
    _override_field(model_cfg, "num_envs", env.num_envs)


class SkrlRunner(SkrlRunner):
    """Runner that uses rlmodule's JAX build_model instead of skrl model instantiators."""

    def __init__(self, env: Wrapper | MultiAgentEnvWrapper, cfg: Mapping[str, Any], rl_cfg) -> None:
        self._rl_cfg = rl_cfg
        super().__init__(env, cfg)  # base Runner wires .agent and .trainer

    @property
    def env(self) -> Wrapper | MultiAgentEnvWrapper:
        return self._env

    # ---------- models ----------
    def _generate_models(self, env: Wrapper | MultiAgentEnvWrapper, cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:
        model_cfg = self._rl_cfg.model
        _set_implicitly_derived_fields(model_cfg, env, cfg["agent"]["class"])

        if isinstance(model_cfg, SharedRLModelCfg):
            # JAX typically uses 'separate=True' in skrl's runner; keep parity by wiring shared instance for PPO/A2C only.
            if cfg["agent"]["class"] in ["PPO", "A2C"]:
                model = build_model(model_cfg)
                return {"agent": {"policy": model, "value": model}}
            raise ValueError(f"Shared model is not supported for {cfg['agent']['class']} algorithm in this JAX backend.")

        elif isinstance(model_cfg, SeparatedRLModelCfg):
            policy_model = build_model(model_cfg.policy)
            value_model = build_model(model_cfg.value)

            if cfg["agent"]["class"] in ["PPO", "A2C"]:
                return {"agent": {"policy": policy_model, "value": value_model}}
            elif cfg["agent"]["class"] == "DDPG":
                return {
                    "agent": {
                        "policy": policy_model,
                        "target_policy": build_model(model_cfg.policy),
                        "critic": value_model,
                        "target_critic": build_model(model_cfg.value),
                    }
                }
            elif cfg["agent"]["class"] == "SAC":
                return {
                    "agent": {
                        "policy": policy_model,
                        "critic_1": value_model,
                        "critic_2": build_model(model_cfg.value),
                        "target_critic_1": build_model(model_cfg.value),
                        "target_critic_2": build_model(model_cfg.value),
                    }
                }
            else:
                raise ValueError(f"Algorithm {cfg['agent']['class']} is not supported.")
        else:
            raise TypeError(
                f"Invalid type for model config: {type(model_cfg)}. Expected SharedRLModelCfg or SeparatedRLModelCfg."
            )

    # ---------- agent class resolution (no RNN variants in this JAX backend) ----------
    def _component(self, name: str):
        name = name.lower()
        if name == "ppo":
            return PPO
        if name == "a2c":
            return A2C
        if name == "ddpg":
            return DDPG
        if name == "sac":
            return SAC
        if name == "ippo":
            return IPPO
        if name == "mappo":
            return MAPPO
        return super()._component(name)

    # ---------- agent instantiation ----------
    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ):
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.unwrapped.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # memory
        if "memory" not in cfg:
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        memory_class = RandomMemory
        memories = {}
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # agent class & defaults
        try:
            agent_class = self._component(cfg["agent"]["class"])
        except KeyError:
            agent_class = self._component("PPO")

        if agent_class in [PPO, A2C, DDPG, SAC]:
            agent_id = possible_agents[0]

            if cfg["agent"]["class"] == "PPO":
                agent_cfg = PPO_DEFAULT_CONFIG.copy()
            elif cfg["agent"]["class"] == "A2C":
                agent_cfg = A2C_DEFAULT_CONFIG.copy()
            elif cfg["agent"]["class"] == "DDPG":
                agent_cfg = DDPG_DEFAULT_CONFIG.copy()
            elif cfg["agent"]["class"] == "SAC":
                agent_cfg = SAC_DEFAULT_CONFIG.copy()

            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({"size": observation_spaces[agent_id], "device": device})
            if cfg["agent"]["class"] in ["PPO", "A2C"]:
                agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
            return agent_class(cfg=agent_cfg, device=device, **agent_kwargs)

        elif agent_class in [IPPO]:
            agent_cfg = IPPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {aid: {"size": observation_spaces[aid], "device": device} for aid in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
            return agent_class(cfg=agent_cfg, device=device, **agent_kwargs)

        elif agent_class in [MAPPO]:
            agent_cfg = MAPPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {aid: {"size": observation_spaces[aid], "device": device} for aid in possible_agents}
            )
            agent_cfg["shared_state_preprocessor_kwargs"].update(
                {aid: {"size": state_spaces[aid], "device": device} for aid in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "shared_observation_spaces": state_spaces,
                "possible_agents": possible_agents,
            }
            return agent_class(cfg=agent_cfg, device=device, **agent_kwargs)

        raise ValueError(f"Unsupported agent class: {agent_class}")

