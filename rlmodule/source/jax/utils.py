from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np


def get_space_size(
    space: int | Sequence[int] | gym.Space, number_of_elements: bool = True
) -> int:
    """Get the size (number of elements) of a Gymnasium space (or shape)."""
    size = None
    if isinstance(space, (int, float)):
        size = int(space)
    elif isinstance(space, (tuple, list)):
        size = int(np.prod(space))
    elif isinstance(space, gym.Space):
        if isinstance(space, gym.spaces.Discrete):
            size = int(space.n if number_of_elements else 1)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            size = int(np.sum(space.nvec) if number_of_elements else space.nvec.shape[0])
        elif isinstance(space, gym.spaces.Box):
            size = int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.Dict):
            size = sum(get_space_size(s, number_of_elements) for s in space.spaces.values())
    if size is None:
        raise ValueError(f"Space type {type(space)} not supported")
    return size


def get_output_size(module_def, input_shape: int | Sequence[int]) -> int:
    """Infer flattened output size of a Flax module with a single (MLP-like) input.

    Assumes `__call__(x)` signature with no RNN state (MLP only).
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    dummy = jnp.zeros((1, *input_shape))
    params = module_def.init(jax.random.PRNGKey(0), dummy)
    out = module_def.apply(params, dummy)
    # flatten everything except batch dimension
    return int(np.prod(out.shape[1:]))


