from typing import Any

from collections.abc import Mapping

import torch
import torch.nn as nn

from rlmodule import logger

from rlmodule.source.torch.model import Model


class RLModel(Model):
    def __init__(
        self,
        device: str | torch.device,
        network: nn.Module,
        output_layer: nn.Module,
    ):
        # TODO(ll) description
        """Reinforcement learning model"""

        super().__init__(device, network)
        self._output_layer = output_layer

    def forward(
        self, inputs: Mapping[str, torch.Tensor | Any], role: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, torch.Tensor | Any]]:

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
        device: str | torch.device,
        network: nn.Module,
        policy_output_layer: nn.Module | None,
        value_output_layer: nn.Module | None,
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
        return self._cached_states is not None and torch.equal(self._cached_states, states)

    def forward(
        self, inputs: Mapping[str, torch.Tensor | Any], role: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, torch.Tensor | Any]]:

        states = inputs["states"]

        if not self._have_cached_states(states):
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
