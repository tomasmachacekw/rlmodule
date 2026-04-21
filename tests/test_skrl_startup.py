"""SKRL startup tests for every rlmodule network.

For each implemented network family we:

1. Build a vectorized gymnasium env with the appropriate obs/action spaces
   and wrap it through skrl's ``GymnasiumWrapper`` so ``device`` and
   ``num_envs`` are available (matching the contract of ``SkrlRunner``).
2. Build models with :func:`rlmodule.skrl.torch.build_model`.
3. Instantiate the corresponding skrl agent (PPO / PPO_RNN / SAC / A2C /
   DDPG), call ``.init()``, and run a single ``.act(...)`` call.

These tests are fast and do NOT invoke ``SequentialTrainer.train()`` -
see ``test_skrl_train_smoke.py`` for that.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.memories.torch import RandomMemory

from rlmodule.skrl.torch import RLModelCfg, SharedRLModelCfg, build_model
from rlmodule.skrl.torch.network import (
    CnnBackboneCfg,
    CnnMlpCfg,
    GruCfg,
    LstmCfg,
    MlpCfg,
    ResidualMlpCfg,
    RnnCfg,
    RnnMlpCfg,
)
from rlmodule.skrl.torch.output_layer import (
    DeterministicLayerCfg,
    GaussianLayerCfg,
    ParameterLogStdCfg,
)


NUM_ENVS = 2


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _wrap(env: gym.Env) -> GymnasiumWrapper:
    env.reset(seed=0)
    return GymnasiumWrapper(env)


def _make_pendulum_env() -> GymnasiumWrapper:
    return _wrap(
        gym.make_vec("Pendulum-v1", num_envs=NUM_ENVS, vectorization_mode="sync")
    )


class _DummyImageEnv(gym.Env):
    """Tiny synthetic image env used for CNN-family startup tests.

    Produces ``uint8`` observations of shape ``(H, W, C)`` and a
    continuous ``Box`` action space. The episode terminates every
    few steps so trainers can still step the env cheaply.
    """

    def __init__(self, shape: Tuple[int, int, int] = (32, 32, 3), action_dim: int = 2):
        super().__init__()
        self._shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self._t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        reward = float(-np.linalg.norm(action))
        terminated = self._t >= 8
        return self._obs(), reward, terminated, False, {}

    def _obs(self) -> np.ndarray:
        return np.random.randint(0, 256, self._shape, dtype=np.uint8)


def _make_image_env(shape: Tuple[int, int, int] = (32, 32, 3), action_dim: int = 2) -> GymnasiumWrapper:
    return _wrap(SyncVectorEnv([lambda s=shape, a=action_dim: _DummyImageEnv(s, a) for _ in range(NUM_ENVS)]))


# ---------------------------------------------------------------------------
# Common cfg helpers
# ---------------------------------------------------------------------------


def _silent_experiment(cfg: dict) -> dict:
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    return cfg


def _ppo_cfg() -> dict:
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 16
    cfg["learning_epochs"] = 1
    cfg["mini_batches"] = 1
    return _silent_experiment(cfg)


def _a2c_cfg() -> dict:
    cfg = A2C_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 16
    return _silent_experiment(cfg)


def _sac_cfg() -> dict:
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 0
    cfg["random_timesteps"] = 0
    return _silent_experiment(cfg)


def _ddpg_cfg() -> dict:
    cfg = DDPG_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 0
    cfg["random_timesteps"] = 0
    return _silent_experiment(cfg)


def _assert_act(agent, env, *, expected_action_dim: int) -> None:
    """Run ``.init()`` + one ``.act()`` call and sanity-check the output."""
    agent.init()
    states, _ = env.reset()
    actions, _, _ = agent.act(states, timestep=0, timesteps=10)
    assert isinstance(actions, torch.Tensor)
    assert actions.shape[0] == env.num_envs
    assert actions.shape[-1] == expected_action_dim
    assert torch.isfinite(actions).all()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reproducible():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def pendulum_env():
    env = _make_pendulum_env()
    yield env
    env.close()


@pytest.fixture
def image_env():
    env = _make_image_env()
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Small model-building helpers
# ---------------------------------------------------------------------------


def _gaussian_head(action_space, head=None):
    return GaussianLayerCfg(
        output_size=action_space,
        head=head,
        log_std=ParameterLogStdCfg(min_log_std=-5.0, max_log_std=2.0, initial_log_std=0.0),
    )


def _value_head(head=None):
    return DeterministicLayerCfg(output_size=1, head=head)


def _build_shared_ppo(env, network_cfg, *, policy_head=None, value_head=None):
    return build_model(
        SharedRLModelCfg(
            network=network_cfg,
            device=env.device,
            policy_output_layer=_gaussian_head(env.action_space, head=policy_head),
            value_output_layer=_value_head(head=value_head),
        )
    )


def _build_separated_ppo(env, policy_network_cfg, value_network_cfg):
    policy = build_model(
        RLModelCfg(
            network=policy_network_cfg,
            device=env.device,
            output_layer=_gaussian_head(env.action_space),
        )
    )
    value = build_model(
        RLModelCfg(
            network=value_network_cfg,
            device=env.device,
            output_layer=_value_head(),
        )
    )
    return {"policy": policy, "value": value}


def _make_ppo_agent(env, models, *, rnn: bool = False):
    memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=env.device)
    agent_cls = PPO_RNN if rnn else PPO
    return agent_cls(
        models=models,
        memory=memory,
        cfg=_ppo_cfg(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


# ---------------------------------------------------------------------------
# Basic MLP variants
# ---------------------------------------------------------------------------


def test_ppo_shared_mlp(pendulum_env):
    net = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=True,
    )
    model = _build_shared_ppo(pendulum_env, net)
    assert isinstance(model, nn.Module)
    assert next(model.parameters()).device == torch.device(pendulum_env.device)

    agent = _make_ppo_agent(pendulum_env, {"policy": model, "value": model})
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


def test_ppo_shared_mlp_with_head(pendulum_env):
    """Cover the ``OutputLayerCfg.head`` path on both policy and value sides."""
    net = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=False,
    )
    policy_head = MlpCfg(hidden_units=[16], activation=nn.ELU, layer_norm=True)
    value_head = MlpCfg(hidden_units=[16, 16], activation=nn.ELU, layer_norm=[True, False])
    model = _build_shared_ppo(pendulum_env, net, policy_head=policy_head, value_head=value_head)

    # Shared trunk has no LNs (layer_norm=False on net). Each head contributes exactly
    # one LN: policy_head has 1 layer with LN=True, value_head has 2 layers but only
    # the first has LN (layer_norm=[True, False]). So the whole model should contain
    # exactly 2 LayerNorm instances, entirely inside the output layers' _head subtrees.
    lns = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
    assert len(lns) == 2, f"expected exactly 2 LayerNorm modules, got {[n for n, _ in lns]}"
    assert all("_head" in n for n, _ in lns), f"LayerNorms are not inside head subtrees: {[n for n, _ in lns]}"

    agent = _make_ppo_agent(pendulum_env, {"policy": model, "value": model})
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


def test_ppo_separated_mlp(pendulum_env):
    policy_net = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=[True, False],
    )
    value_net = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=True,
    )
    models = _build_separated_ppo(pendulum_env, policy_net, value_net)
    agent = _make_ppo_agent(pendulum_env, models)
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


def test_ppo_separated_residual_mlp(pendulum_env):
    net = ResidualMlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_size=32,
        num_blocks=2,
        activation=nn.ELU,
        layer_norm=True,
        final_layer_norm=True,
    )
    models = _build_separated_ppo(pendulum_env, net, net.copy())
    # 2 blocks * 2 LN + final_ln + input_proj = 7 per network
    for m in models.values():
        ln_count = sum(1 for mod in m.modules() if isinstance(mod, nn.LayerNorm))
        assert ln_count >= 5, f"expected LN modules in residual trunk, got {ln_count}"

    agent = _make_ppo_agent(pendulum_env, models)
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


# ---------------------------------------------------------------------------
# RNN variants
# ---------------------------------------------------------------------------


def test_ppo_rnn_mlp_lstm(pendulum_env):
    net = RnnMlpCfg(
        input_states=pendulum_env.observation_space,
        rnn=LstmCfg(num_envs=pendulum_env.num_envs, num_layers=1, hidden_size=16, sequence_length=4),
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
    )
    model = _build_shared_ppo(pendulum_env, net)
    assert model.is_rnn
    assert model.is_lstm

    agent = _make_ppo_agent(pendulum_env, {"policy": model, "value": model}, rnn=True)
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


def test_ppo_rnn_mlp_gru(pendulum_env):
    net = RnnMlpCfg(
        input_states=pendulum_env.observation_space,
        rnn=GruCfg(num_envs=pendulum_env.num_envs, num_layers=1, hidden_size=16, sequence_length=4),
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU),
    )
    model = _build_shared_ppo(pendulum_env, net)
    assert model.is_rnn
    assert not model.is_lstm
    agent = _make_ppo_agent(pendulum_env, {"policy": model, "value": model}, rnn=True)
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


def test_ppo_bare_rnn(pendulum_env):
    """Bare :class:`RnnCfg` - no MLP head after the RNN trunk."""
    net = RnnCfg(
        input_states=pendulum_env.observation_space,
        num_envs=pendulum_env.num_envs,
        num_layers=1,
        hidden_size=16,
        sequence_length=4,
    )
    model = _build_shared_ppo(pendulum_env, net)
    assert model.is_rnn
    agent = _make_ppo_agent(pendulum_env, {"policy": model, "value": model}, rnn=True)
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


# ---------------------------------------------------------------------------
# CNN variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset,shape",
    [
        ("nature", (84, 84, 4)),
        ("impala", (32, 32, 3)),
    ],
)
def test_ppo_cnn_backbone(preset, shape):
    env = _make_image_env(shape=shape)
    try:
        net_cfg = CnnMlpCfg(
            input_states=shape,
            cnn=CnnBackboneCfg(preset=preset),
            mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
        )
        model = _build_shared_ppo(env, net_cfg)

        # Verify the first Conv2d sees the expected channel count.
        first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
        assert first_conv.in_channels == shape[-1]

        agent = _make_ppo_agent(env, {"policy": model, "value": model})
        _assert_act(agent, env, expected_action_dim=env.action_space.shape[0])
    finally:
        env.close()


def test_ppo_cnn_mlp_with_residual_trunk(image_env):
    shape = image_env.observation_space.shape
    net_cfg = CnnMlpCfg(
        input_states=shape,
        cnn=CnnBackboneCfg(preset="impala"),
        mlp=ResidualMlpCfg(
            hidden_size=32,
            num_blocks=2,
            activation=nn.ELU,
            layer_norm=True,
            final_layer_norm=True,
        ),
    )
    model = _build_shared_ppo(image_env, net_cfg)
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == shape[-1]
    agent = _make_ppo_agent(image_env, {"policy": model, "value": model})
    _assert_act(agent, image_env, expected_action_dim=image_env.action_space.shape[0])


# ---------------------------------------------------------------------------
# SAC with separated policy + twin critics (input_actions path)
# ---------------------------------------------------------------------------


def test_sac_separated_mlp(pendulum_env):
    net_cfg = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=True,
    )
    net_cfg_with_actions = net_cfg.copy()
    net_cfg_with_actions.input_actions = pendulum_env.action_space

    policy = build_model(
        RLModelCfg(
            network=net_cfg,
            device=pendulum_env.device,
            output_layer=_gaussian_head(pendulum_env.action_space),
        )
    )
    critic_cfg = RLModelCfg(
        network=net_cfg_with_actions,
        device=pendulum_env.device,
        output_layer=DeterministicLayerCfg(),
    )
    models = {
        "policy": policy,
        "critic_1": build_model(critic_cfg),
        "critic_2": build_model(critic_cfg),
        "target_critic_1": build_model(critic_cfg),
        "target_critic_2": build_model(critic_cfg),
    }

    memory = RandomMemory(memory_size=256, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = SAC(
        models=models,
        memory=memory,
        cfg=_sac_cfg(),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


# ---------------------------------------------------------------------------
# A2C + DDPG (cover the other algorithms SkrlRunner wires up)
# ---------------------------------------------------------------------------


def test_a2c_shared_mlp(pendulum_env):
    net = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=True,
    )
    model = _build_shared_ppo(pendulum_env, net)
    memory = RandomMemory(memory_size=32, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = A2C(
        models={"policy": model, "value": model},
        memory=memory,
        cfg=_a2c_cfg(),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    _assert_act(agent, pendulum_env, expected_action_dim=pendulum_env.action_space.shape[0])


def test_ddpg_separated_mlp(pendulum_env):
    net_cfg = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=True,
    )
    action_dim = pendulum_env.action_space.shape[0]
    policy_cfg = RLModelCfg(
        network=net_cfg,
        device=pendulum_env.device,
        output_layer=DeterministicLayerCfg(output_size=action_dim, output_activation=nn.Tanh),
    )
    policy = build_model(policy_cfg)
    target_policy = build_model(policy_cfg)

    net_cfg_with_actions = net_cfg.copy()
    net_cfg_with_actions.input_actions = pendulum_env.action_space
    critic_cfg = RLModelCfg(
        network=net_cfg_with_actions,
        device=pendulum_env.device,
        output_layer=DeterministicLayerCfg(),
    )
    critic = build_model(critic_cfg)
    target_critic = build_model(critic_cfg)

    memory = RandomMemory(memory_size=256, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = DDPG(
        models={
            "policy": policy,
            "target_policy": target_policy,
            "critic": critic,
            "target_critic": target_critic,
        },
        memory=memory,
        cfg=_ddpg_cfg(),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    _assert_act(agent, pendulum_env, expected_action_dim=action_dim)
