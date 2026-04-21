"""End-to-end smoke tests: run ``SequentialTrainer.train()`` for a handful
of rollouts over each important network family and verify that gradients
actually flowed (i.e. the policy weights changed and the agent tracked
at least one loss step).

These are the "can we train at all?" tests. They are deliberately small
(~4 PPO rollouts per run) so they complete on CPU in a few seconds. They
are tagged ``@pytest.mark.slow`` so the fast unit tests in
``test_skrl_startup.py`` remain usable in a quick iteration loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from rlmodule.skrl.torch import RLModelCfg, SharedRLModelCfg, build_model
from rlmodule.skrl.torch.network import (
    CnnBackboneCfg,
    CnnMlpCfg,
    GruCfg,
    MlpCfg,
    ResidualMlpCfg,
    RnnMlpCfg,
)
from rlmodule.skrl.torch.output_layer import (
    DeterministicLayerCfg,
    GaussianLayerCfg,
    ParameterLogStdCfg,
)


pytestmark = pytest.mark.slow


NUM_ENVS = 2
ROLLOUTS = 16
N_ROLLOUT_CYCLES = 4
TIMESTEPS = ROLLOUTS * N_ROLLOUT_CYCLES  # 4 PPO rollout cycles


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _wrap(env) -> GymnasiumWrapper:
    env.reset(seed=0)
    return GymnasiumWrapper(env)


def _make_pendulum_env() -> GymnasiumWrapper:
    return _wrap(gym.make_vec("Pendulum-v1", num_envs=NUM_ENVS, vectorization_mode="sync"))


class _DummyImageEnv(gym.Env):
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


def _silent_experiment(cfg: dict, tmp_path) -> dict:
    """Configure the agent to skip disk I/O and (almost) skip tensorboard writes.

    We cannot use ``write_interval=0`` because SAC only populates
    ``tracking_data`` when ``write_interval > 0`` (see ``SAC._update``). Instead
    we set a very large ``write_interval`` so the condition
    ``timestep % write_interval == 0`` never triggers, and redirect the
    ``experiment.directory`` into a pytest ``tmp_path`` so nothing is written
    to the user's CWD.
    """
    cfg["experiment"]["write_interval"] = 10**9
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["experiment"]["directory"] = str(tmp_path)
    return cfg


def _ppo_cfg(tmp_path, *, rollouts: int = ROLLOUTS) -> dict:
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts
    cfg["learning_epochs"] = 2
    cfg["mini_batches"] = 2
    cfg["learning_rate"] = 1e-3
    return _silent_experiment(cfg, tmp_path)


def _sac_cfg(tmp_path) -> dict:
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 16
    cfg["random_timesteps"] = 0
    cfg["batch_size"] = 16
    cfg["gradient_steps"] = 1
    return _silent_experiment(cfg, tmp_path)


def _trainer_cfg(timesteps: int = TIMESTEPS) -> dict:
    return {
        "timesteps": timesteps,
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }


def _gaussian_head(action_space):
    return GaussianLayerCfg(
        output_size=action_space,
        log_std=ParameterLogStdCfg(min_log_std=-5.0, max_log_std=2.0, initial_log_std=0.0),
    )


def _first_trainable_weight(model: nn.Module) -> torch.Tensor:
    """Snapshot the first Linear/Conv2d weight in the policy for comparison."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            return m.weight.detach().clone()
    raise AssertionError("No Linear/Conv2d found in policy model")


def _assert_trained(agent, before: torch.Tensor, model: nn.Module) -> None:
    """Assert the model changed and that at least one loss was tracked."""
    after = _first_trainable_weight(model)
    assert after.shape == before.shape
    assert not torch.allclose(before, after), "policy weights did not change after training"

    # Loss tracking (populated by agent._track_data during .post_interaction()).
    tracking = getattr(agent, "tracking_data", {})
    loss_keys = [k for k in tracking.keys() if "loss" in k.lower()]
    assert loss_keys, f"no loss entries in tracking_data (got {list(tracking.keys())})"
    any_values = any(len(tracking[k]) > 0 for k in loss_keys)
    assert any_values, "loss tracking_data entries are empty"


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
# Smoke tests
# ---------------------------------------------------------------------------


def test_train_ppo_shared_mlp(pendulum_env, tmp_path):
    """PPO + shared MLP trunk with per-layer LayerNorm."""
    net_cfg = MlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_units=[32, 32],
        activation=nn.ELU,
        layer_norm=True,
    )
    model = build_model(
        SharedRLModelCfg(
            network=net_cfg,
            device=pendulum_env.device,
            policy_output_layer=_gaussian_head(pendulum_env.action_space),
            value_output_layer=DeterministicLayerCfg(),
        )
    )
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = PPO(
        models={"policy": model, "value": model},
        memory=memory,
        cfg=_ppo_cfg(tmp_path),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    before = _first_trainable_weight(model)

    trainer = SequentialTrainer(cfg=_trainer_cfg(), env=pendulum_env, agents=[agent])
    trainer.train()

    _assert_trained(agent, before, model)


def test_train_ppo_separated_residual_mlp(pendulum_env, tmp_path):
    """PPO + two independent ResidualMlp networks (separated actor/critic)."""
    net_cfg = ResidualMlpCfg(
        input_states=pendulum_env.observation_space,
        hidden_size=32,
        num_blocks=2,
        activation=nn.ELU,
        layer_norm=True,
        final_layer_norm=True,
    )
    policy = build_model(
        RLModelCfg(
            network=net_cfg,
            device=pendulum_env.device,
            output_layer=_gaussian_head(pendulum_env.action_space),
        )
    )
    value = build_model(
        RLModelCfg(
            network=net_cfg.copy(),
            device=pendulum_env.device,
            output_layer=DeterministicLayerCfg(),
        )
    )
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=_ppo_cfg(tmp_path),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    before = _first_trainable_weight(policy)

    trainer = SequentialTrainer(cfg=_trainer_cfg(), env=pendulum_env, agents=[agent])
    trainer.train()

    _assert_trained(agent, before, policy)


def test_train_ppo_rnn_mlp(pendulum_env, tmp_path):
    """PPO_RNN + RnnMlp (GRU backbone + MLP head)."""
    net_cfg = RnnMlpCfg(
        input_states=pendulum_env.observation_space,
        rnn=GruCfg(num_envs=pendulum_env.num_envs, num_layers=1, hidden_size=16, sequence_length=4),
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
    )
    model = build_model(
        SharedRLModelCfg(
            network=net_cfg,
            device=pendulum_env.device,
            policy_output_layer=_gaussian_head(pendulum_env.action_space),
            value_output_layer=DeterministicLayerCfg(),
        )
    )
    assert model.is_rnn

    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = PPO_RNN(
        models={"policy": model, "value": model},
        memory=memory,
        cfg=_ppo_cfg(tmp_path),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    before = _first_trainable_weight(model)

    trainer = SequentialTrainer(cfg=_trainer_cfg(), env=pendulum_env, agents=[agent])
    trainer.train()

    _assert_trained(agent, before, model)


def test_train_sac_separated_mlp(pendulum_env, tmp_path):
    """SAC + separated policy / twin critics / twin target critics (MLP)."""
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

    memory = RandomMemory(memory_size=512, num_envs=pendulum_env.num_envs, device=pendulum_env.device)
    agent = SAC(
        models=models,
        memory=memory,
        cfg=_sac_cfg(tmp_path),
        observation_space=pendulum_env.observation_space,
        action_space=pendulum_env.action_space,
        device=pendulum_env.device,
    )
    before = _first_trainable_weight(policy)

    trainer = SequentialTrainer(cfg=_trainer_cfg(timesteps=TIMESTEPS), env=pendulum_env, agents=[agent])
    trainer.train()

    _assert_trained(agent, before, policy)


def test_train_ppo_cnn_mlp(image_env, tmp_path):
    """PPO + CnnMlp (IMPALA backbone + MLP head) on a synthetic image env."""
    shape = image_env.observation_space.shape
    net_cfg = CnnMlpCfg(
        input_states=shape,
        cnn=CnnBackboneCfg(preset="impala"),
        mlp=MlpCfg(hidden_units=[64, 64], activation=nn.ELU, layer_norm=True),
    )
    model = build_model(
        SharedRLModelCfg(
            network=net_cfg,
            device=image_env.device,
            policy_output_layer=_gaussian_head(image_env.action_space),
            value_output_layer=DeterministicLayerCfg(),
        )
    )
    # Verify the first conv sees the expected channel count before training.
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == shape[-1]

    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=image_env.num_envs, device=image_env.device)
    agent = PPO(
        models={"policy": model, "value": model},
        memory=memory,
        cfg=_ppo_cfg(tmp_path),
        observation_space=image_env.observation_space,
        action_space=image_env.action_space,
        device=image_env.device,
    )
    before = _first_trainable_weight(model)

    trainer = SequentialTrainer(cfg=_trainer_cfg(), env=image_env, agents=[agent])
    trainer.train()

    _assert_trained(agent, before, model)
