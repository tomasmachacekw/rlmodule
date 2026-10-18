from typing import Any, Mapping, Optional, Tuple, Union

import logging

import torch
import torch.nn as nn

from .model import Model


logger = logging.getLogger("rlmodule")


# SKRL 2.x roles whose canonical trunk input is the actor's observation. Every
# other role (``value``, ``critic_*``, ``target_critic_*``) is a critic and
# reads ``states``.
_ACTOR_ROLES = frozenset({"policy", "target_policy"})


def _select_trunk_input(inputs: Mapping[str, Union[torch.Tensor, Any]], role: str) -> torch.Tensor:
    """Pick the trunk input tensor for ``role`` per the SKRL 2.x contract.

    SKRL 2.x agents always pass both ``inputs["observations"]`` (the actor's
    view) and ``inputs["states"]`` (the critic's view). In a symmetric setup
    the agent sets ``state_space = observation_space`` so the two are equal;
    they differ only for an asymmetric / central critic. Actors read
    ``observations``; value / critic roles read ``states``.

    The other key is used only as a fallback when the preferred one is absent
    (e.g. a model invoked directly with a single key); if neither is present we
    raise instead of silently masking the intended source.
    """
    if role in _ACTOR_ROLES:
        prefer, other = "observations", "states"
    else:
        prefer, other = "states", "observations"
    trunk_input = inputs.get(prefer)
    if trunk_input is None:
        trunk_input = inputs.get(other)
    if trunk_input is None:
        raise KeyError(
            f"Model.forward expected '{prefer}' (role={role!r}) in inputs, "
            "but neither 'observations' nor 'states' was provided."
        )
    return trunk_input


def _episode_ends(inputs: Mapping[str, Union[torch.Tensor, Any]]) -> Optional[torch.Tensor]:
    """Where a sampled sequence's episodes end, for the recurrent trunk's resets.

    skrl's RNN agents zero the hidden state after a ``terminated`` *or* a
    ``truncated`` step while collecting, and pass both with every sampled
    sequence; resetting on ``terminated`` alone would carry a timed-out
    episode's hidden state into the next episode of the same sequence.
    """
    terminated = inputs.get("terminated", None)
    truncated = inputs.get("truncated", None)
    if terminated is None or truncated is None:
        return terminated if terminated is not None else truncated
    return terminated | truncated


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
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Forward the trunk + output layer.

        Input selection follows the SKRL 2.x convention (see
        :func:`_select_trunk_input`): actor roles read ``observations`` and
        value / critic roles read ``states``. Q-functions
        (``self._input_actions``) are critics, so they read ``states`` and
        additionally concatenate ``taken_actions`` onto the trunk input. The
        result follows the 2.x 2-tuple ``(actions, outputs_dict)`` contract.
        """

        trunk_input = _select_trunk_input(inputs, role)
        if self._input_actions:
            trunk_input = torch.cat((trunk_input, inputs["taken_actions"]), dim=1)
        if self._rnn:
            output, output_dict = self._net(trunk_input, _episode_ends(inputs), inputs["rnn"])
        else:
            output = self._net(trunk_input)
            output_dict = {}

        if "taken_pre_squash" in inputs:
            output_dict = {**output_dict, "taken_pre_squash": inputs["taken_pre_squash"]}
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
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Forward the shared trunk and dispatch to a per-role head.

        The trunk is shared by the policy and value heads, so it must consume a
        single, role-independent input. It reads ``observations`` (the actor's
        view), which equals ``states`` in the symmetric setups a shared
        actor-critic assumes (SKRL 2.x defaults ``state_space`` to
        ``observation_space``). ``states`` is used only as a fallback for
        direct invocation. The trunk runs once per unique input tensor and the
        cached output is dispatched to the policy or value head by ``role``.
        """

        states = inputs.get("observations")
        if states is None:
            states = inputs["states"]

        if not self._have_cached_states(states):
            # Drop the old cache BEFORE computing the new forward, so the
            # previous minibatch's activations (and any RNN h_n/c_n dangling
            # in ``output_dict["rnn"]``) are released before the new forward
            # allocates.
            self._reset_cache()

            if self._rnn:
                output, output_dict = self._net(states, _episode_ends(inputs), inputs["rnn"])
            else:
                output = self._net(states)
                output_dict = {}

            self._cached_states = states
            self._cached_output = output, output_dict

        output, output_dict = self._cached_output

        if role == "policy":
            if "taken_pre_squash" in inputs:
                output_dict = {**output_dict, "taken_pre_squash": inputs["taken_pre_squash"]}
            return self._policy_output_layer(output, inputs.get("taken_actions", None), output_dict)
        elif role == "value":
            return self._value_output_layer(output, inputs.get("taken_actions", None), output_dict)
        raise ValueError(f"SharedRLModel.forward got unsupported role {role!r}; expected 'policy' or 'value'.")
