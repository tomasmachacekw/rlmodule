from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from rlmodule import logger

from .model import Model


class RLModel(Model):
    def __init__(
        self,
        device: Union[str, torch.device],
        network: nn.Module,
        output_layer: nn.Module,
    ):
        # TODO(ll) description
        """Reinforcement learning model"""

        super().__init__(device, network)
        self._output_layer = output_layer

    def forward(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:

        if self._input_actions:
            states = torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)
        else:
            states = inputs["states"]
        if self._rnn:
            output, output_dict = self._net(states, inputs.get("terminated", None), inputs["rnn"])
        else:
            output = self._net(states)
            output_dict = {}

        return self._output_layer(output, inputs.get("taken_actions", None), output_dict)


class SharedRLModel(Model):
    def __init__(
        self,
        device: Union[str, torch.device],
        network: nn.Module,
        policy_output_layer: Optional[nn.Module],
        value_output_layer: Optional[nn.Module],
    ):
        # TODO(ll) description
        """Shared Reinforcement learning model."""

        super().__init__(device, network)

        if self._input_actions:
            logger.critical("Action inputs will not be propagated for the shared model.")

        self._policy_output_layer = policy_output_layer
        self._value_output_layer = value_output_layer

        # caching for shared forward pass for policy and value roles
        self._cached_states = None
        self._cached_output = None

    def _have_cached_states(self, states):
        # Identity check is enough: SKRL passes the same `sampled_states`
        # reference to the policy and value role calls within a minibatch.
        # Using ``torch.equal`` here would force an O(N) value compare and a
        # host sync on CUDA, and would also risk hitting a stale autograd
        # graph if two different minibatches happened to share values.
        return self._cached_states is not None and self._cached_states is states

    def _reset_cache(self) -> None:
        """Drop references to the last forward's outputs.

        This is important for memory: the cached tuple holds the trunk output
        and (for RNN backbones) the hidden-state tensors ``h_n`` / ``c_n``.
        After ``loss.backward()`` frees the graph these tensors are still
        allocated until we drop the reference. We release them on cache miss
        and on any train/eval mode switch.
        """
        self._cached_states = None
        self._cached_output = None

    def train(self, mode: bool = True):
        self._reset_cache()
        return super().train(mode)

    def forward(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:

        states = inputs["states"]

        if not self._have_cached_states(states):
            # Drop the old cache BEFORE computing the new forward, so the
            # previous minibatch's activations (and any RNN h_n/c_n dangling
            # in ``output_dict["rnn"]``) are released before the new forward
            # allocates.
            self._reset_cache()

            if self._rnn:
                output, output_dict = self._net(states, inputs.get("terminated", None), inputs["rnn"])
            else:
                output = self._net(states)
                output_dict = {}

            self._cached_states = states
            self._cached_output = output, output_dict

        output, output_dict = self._cached_output

        if role == "policy":
            return self._policy_output_layer(output, inputs.get("taken_actions", None), output_dict)
        elif role == "value":
            return self._value_output_layer(output, inputs.get("taken_actions", None), output_dict)
