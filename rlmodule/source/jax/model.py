from __future__ import annotations

from typing import Any

from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

# skrl's JAX model base (Flax Module). Child models define __call__, not act.
# skrl initializes params and calls .act(), which delegates to __call__.
from skrl.models.jax import Model as SkrlModel


class Model(SkrlModel):
    """Base class representing a function approximator (MLP-only in JAX).

    Flax requirement: if overriding __init__, call flax.linen.Module.__post_init__ via parent init.
    The skrl base already subclasses flax.linen.Module and expects __call__(inputs, role).
    For JAX/Flax, module attributes must have default values to avoid init errors.
    """

    def __init__(
        self,
        observation_space: int | tuple[int, ...] | Any,
        action_space: int | tuple[int, ...] | Any,
        device: str | jax.Device | None,
        *,
        network: nn.Module,
        input_actions: bool = False,
        parent: Any | None = None,
        name: str | None = None,
    ) -> None:
        # skrl base init sets num_observations / num_actions and device
        super().__init__(observation_space, action_space, device, parent, name)
        self._net = network
        self._input_actions = bool(input_actions)
        nn.Module.__post_init__(self)  # ensure Flax dataclass invariants

    # JAX backend: no RNN support in this implementation
    @property
    def is_rnn(self) -> bool:
        return False


class RLModel(Model):
    """Single-head RL model: network -> output_layer."""

    def __init__(
        self,
        observation_space: int | tuple[int, ...] | Any,
        action_space: int | tuple[int, ...] | Any,
        device: str | jax.Device | None,
        *,
        network: nn.Module,
        output_layer: nn.Module,
        input_actions: bool = False,
        parent: Any | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            network=network,
            input_actions=input_actions,
            parent=parent,
            name=name,
        )
        self._output_layer = output_layer
        nn.Module.__post_init__(self)  # Flax post-init as we override __init__

    def __call__(
        self,
        inputs: Mapping[str, jnp.ndarray | Any],
        role: str = "",
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, Mapping[str, jnp.ndarray | Any]]:
        states = inputs["states"]

        if self._input_actions and ("taken_actions" in inputs) and (inputs["taken_actions"] is not None):
            states = jnp.concatenate([states, inputs["taken_actions"]], axis=-1)

        features = self._net(states)  # submodule call; parameters are tracked by Flax
        # Output layer must return (actions, log_prob | None, extras) to match skrl model contract
        return self._output_layer(features, inputs.get("taken_actions", None), {})


class SharedRLModel(Model):
    """Shared network with separate policy/value heads (no caching).

    Note: skrl's JAX runner expects separate models in most setups; this class
    keeps parity with your Torch API but performs two forward passes through the
    output heads over a single shared feature tensor.
    """

    def __init__(
        self,
        observation_space: int | tuple[int, ...] | Any,
        action_space: int | tuple[int, ...] | Any,
        device: str | jax.Device | None,
        *,
        network: nn.Module,
        policy_output_layer: nn.Module,
        value_output_layer: nn.Module,
        parent: Any | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            network=network,
            input_actions=False,  # aligned with Torch: actions-as-input not propagated for shared models
            parent=parent,
            name=name,
        )
        self._policy_output_layer = policy_output_layer
        self._value_output_layer = value_output_layer
        nn.Module.__post_init__(self)

    def __call__(
        self,
        inputs: Mapping[str, jnp.ndarray | Any],
        role: str = "",
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, Mapping[str, jnp.ndarray | Any]]:
        features = self._net(inputs["states"])
        if role == "policy":
            return self._policy_output_layer(features, inputs.get("taken_actions", None), {})
        elif role == "value":
            return self._value_output_layer(features, inputs.get("taken_actions", None), {})
        raise ValueError("role must be 'policy' or 'value'")


