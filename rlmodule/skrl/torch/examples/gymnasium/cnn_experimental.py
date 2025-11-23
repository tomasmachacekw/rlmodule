
import os
from datetime import datetime
import gymnasium as gym

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import numpy as np
import torch.nn as nn

from rlmodule.skrl.torch import SharedRLModelCfg, build_model
from rlmodule.skrl.torch.network import CnnCfg
from rlmodule.skrl.torch.output_layer import DeterministicLayerCfg, GaussianLayerCfg, ParameterLogStdCfg


# CNN is in an experimental phase. Some combination of inputs may not be yet supported


class DummyEnv(gym.Env):
    """Dummy env producing random observations of given shape."""

    def __init__(self):
        super(DummyEnv, self).__init__()
        # Define observation space:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(169,), dtype=np.uint8)
        # Define action space: Continuous values between -1.0 and 1.0, with 6 action dimensions
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self) -> tuple[np.ndarray, dict]:
        # Reset environment and return initial random observation
        observation = self._get_random_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Create a new random observation
        observation = self._get_random_observation()
        reward = 0.0  # No meaningful reward in this dummy environment
        terminated = False  # This dummy environment never ends
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def _get_random_observation(self) -> np.ndarray:
        # Generate a random observation with values between 0 and 255
        return np.random.randint(0, 256, 169, dtype=np.uint8)


def get_model(env):
    """Instantiate the agent's models (function approximators)."""

    net_cfg = CnnCfg(
        input_states=(13, 13, 1),
        layers=[
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        ],
    )

    model = build_model(
        SharedRLModelCfg(
            network=net_cfg,
            device=device,
            policy_output_layer=GaussianLayerCfg(
                output_size=env.action_space,
                log_std=ParameterLogStdCfg(
                    min_log_std=-1.2,
                    max_log_std=2,
                    initial_log_std=0.0,
                ),
            ),
            value_output_layer=DeterministicLayerCfg(),
        )
    )

    print(model)
    return {"policy": model, "value": model}


# set seed for reproducibility
seed = 42
set_seed(seed)

env = DummyEnv()
# env = gym.make_vec("ALE/Pong-v5", num_envs=4, vectorization_mode="sync")
env = GymnasiumWrapper(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models = get_model(env)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/skrl/torch/"
cfg["experiment"]["experiment_name"] = (
    os.path.splitext(os.path.basename(__file__))[0] + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)

params = {
    "models": models,
    "memory": memory,
    "cfg": cfg,
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    "device": device,
}

if models["policy"].is_rnn:
    agent = PPO_RNN(**params)
else:
    agent = PPO(**params)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
