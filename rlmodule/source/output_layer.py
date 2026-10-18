import math
from typing import Union

import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rlmodule.source.utils import get_output_size, get_space_size


def _build_head(cfg, input_size: int) -> tuple[nn.Module, int]:
    """Build the optional pre-projection head MLP.

    Returns a tuple of ``(head_module, projection_in_features)``. When
    ``cfg.head`` is ``None`` this returns ``(nn.Identity(), input_size)`` so
    the caller can feed the projection straight from the trunk output.
    """
    head_cfg = getattr(cfg, "head", None)
    if head_cfg is None:
        return nn.Identity(), input_size

    head_cfg = head_cfg.copy()
    head_cfg.input_size = input_size
    head_module = head_cfg.module(head_cfg)
    projection_in = get_output_size(head_module, input_size)
    return head_module, projection_in


class OutputLayer(nn.Module):
    def __init__(self, device: Union[str, torch.device], input_size: int, cfg):
        """Base class for OutputLayer class hierarchy."""
        super().__init__()
        self.device = device

        # clip action is only supported for gym/gymnasium space
        self._clip_actions = False

        if issubclass(type(cfg.output_size), gym.Space) or issubclass(type(cfg.output_size), gymnasium.Space):
            self._clip_actions = cfg.clip_actions

        if self._clip_actions:
            self._clip_actions_min = torch.tensor(cfg.output_size.low, device=self.device, dtype=torch.float32)
            self._clip_actions_max = torch.tensor(cfg.output_size.high, device=self.device, dtype=torch.float32)

        self._input_size = input_size
        self._output_size = get_space_size(cfg.output_size)

        self._output_scale = cfg.output_scale

        # Optional pre-projection head MLP (see OutputLayerCfg.head).
        self._head, self._projection_in = _build_head(cfg, input_size)


class GaussianLayer(OutputLayer):
    def __init__(self, device: Union[str, torch.device], input_size: int, cfg):
        """Gaussian output layer

        :param device: Device on which a tensor/array is or will be allocated
        :type device: str or torch.device
        :param input_size
        :type int
        :param cfg: Configuration of Gaussian output layer
        :type OutputLayerCfg

        :raises ValueError: If the reduction method is not valid
        """
        super().__init__(device, input_size, cfg)

        self._cached_log_std = None
        self._num_samples = None
        self._distribution = None

        if cfg.reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = (
            torch.mean
            if cfg.reduction == "mean"
            else torch.sum if cfg.reduction == "sum" else torch.prod if cfg.reduction == "prod" else None
        )

        self._squash = cfg.squash
        if self._squash and cfg.output_activation is not nn.Identity:
            raise ValueError(
                "squash=True bounds the sampled action with tanh, so the mean must be the pre-squash value: "
                f"set output_activation=nn.Identity (got {cfg.output_activation})"
            )
        if cfg.entropy_reduction not in ["mean", "sum"]:
            raise ValueError("entropy_reduction must be one of 'mean' or 'sum'")
        self._entropy_reduction = cfg.entropy_reduction

        self._net = nn.Sequential(nn.Linear(self._projection_in, self._output_size), cfg.output_activation())

        self._log_std = cfg.log_std.class_type(device, self._projection_in, self._output_size, cfg.log_std)

    def forward(self, input, taken_actions, outputs_dict):
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # ``RLModel.forward`` passes ``inputs["taken_pre_squash"]`` here (``_squashed_forward``).
        taken_pre_squash = outputs_dict.pop("taken_pre_squash", None)
        if taken_pre_squash is not None and not self._squash:
            raise ValueError("taken_pre_squash is the pre-tanh sample of a squashed head, but squash=False")

        # Optional per-head pre-projection (see OutputLayerCfg.head).
        head_out = self._head(input)

        mean_actions = self._output_scale * self._net(
            head_out
        )  # TODO in skrl example the self._cfg.output_scale * is done here. -> why understand this (is it correct).

        self._cached_log_std = self._log_std(head_out)
        self._num_samples = mean_actions.shape[0]

        # distribution
        self._distribution = Normal(mean_actions, self._cached_log_std.exp())

        if self._squash:
            return self._squashed_forward(mean_actions, taken_actions, outputs_dict, taken_pre_squash)

        # sample using the reparametrization trick
        actions = self._distribution.rsample()

        # clip actions
        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        if taken_actions is None:
            taken_actions = actions

        # log of the probability density function
        log_prob = self._distribution.log_prob(taken_actions)
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # SKRL 2.x model.act contract: return ``(actions, outputs_dict)``
        # with auxiliary tensors (``log_prob``, ``mean_actions``) nested
        # inside the dict, matching the upstream PPO ``outputs["log_prob"]``
        # call site.
        outputs_dict["mean_actions"] = mean_actions
        outputs_dict["log_prob"] = log_prob
        return actions, outputs_dict

        ####

        # TODO(ll)
        # CNN
        # 1) changing shape if it comes in linear fashion of input, check how I was doing this.
        # def compute(self, inputs, role):
        # # permute (samples, width * height * channels) -> (samples, channels, width, height)
        # return self._net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)),
        # self._log_std, {}
        # 2) what with that weird Shapes?  search for taken_actions, who called it with this input.
        # How should CNN be applied to such things..just in states?

        # TODO(ll) output scale removed .. check that tanh where is
        # return output * self.instantiator_output_scale, self._log_std, {}
        # return output, self._log_std, output_dict

    # Keeps atanh finite for actions stored at the bound (float32 tanh saturates to 1).
    _SQUASH_EPS = 1e-6

    def _squashed_forward(self, mean_actions, taken_actions, outputs_dict, taken_pre_squash=None):
        """Tanh-squashed Gaussian: ``u ~ N(mean, std)``, ``a = output_scale * tanh(u)``.

        ``self._distribution`` stays the pre-squash Normal over ``u``. ``mean_actions``
        here is ``output_scale * net(x)`` (``output_activation`` is Identity), so
        ``mu = mean_actions / output_scale`` in ``u`` space.

        ``log_prob`` applies the change of variables per action dimension before the
        reduction: ``log p(a) = log N(u) - log(output_scale) - log(1 - tanh(u)^2)``,
        with ``log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))`` for stability.

        A sample's ``u`` is returned as ``outputs_dict["pre_squash"]``. Re-evaluating a
        taken action recovers ``u = atanh(a / output_scale)``, which float32 loses once
        ``tanh(u)`` rounds to 1 (``|u|`` above about 8): at a std of 0.08 a ``u`` of 12
        comes back 4.9 off, thousands of nats in ``log_prob``. ``taken_pre_squash``
        (the stored ``u``) skips the inversion; PPO stores it (``PPOSquash``).
        """
        scale = self._output_scale
        self._distribution = Normal(mean_actions / scale, self._cached_log_std.exp())

        if taken_pre_squash is not None:
            pre_squash = taken_pre_squash
            actions = scale * torch.tanh(pre_squash)
        elif taken_actions is None:
            pre_squash = self._distribution.rsample()
            actions = scale * torch.tanh(pre_squash)
        else:
            actions = taken_actions
            bound = 1.0 - self._SQUASH_EPS
            pre_squash = torch.atanh(torch.clamp(taken_actions / scale, -bound, bound))

        log_det = 2.0 * (math.log(2.0) - pre_squash - F.softplus(-2.0 * pre_squash)) + math.log(scale)
        log_prob = self._distribution.log_prob(pre_squash) - log_det
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs_dict["mean_actions"] = scale * torch.tanh(mean_actions / scale)
        outputs_dict["log_prob"] = log_prob
        outputs_dict["pre_squash"] = pre_squash
        return actions, outputs_dict

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])  # entropy_reduction="mean"; [4096, 1] with "sum"

        With ``squash=True`` this is the entropy of the pre-squash Gaussian: the
        squashed distribution has no closed form (SAC uses ``-log_prob`` instead).
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        entropy = self._distribution.entropy().to(self.device)
        if self._entropy_reduction == "sum":
            entropy = entropy.sum(dim=-1, keepdim=True)
        return entropy

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return the log standard deviation of the model

        :return: Log standard deviation of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> log_std = model.get_log_std()
            >>> print(log_std.shape)
            torch.Size([4096, 8])
        """
        return self._cached_log_std.repeat(self._num_samples, 1)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Normal
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Normal(loc: torch.Size([4096, 8]), scale: torch.Size([4096, 8]))
        """
        return self._distribution


class DeterministicLayer(OutputLayer):
    def __init__(self, device: Union[str, torch.device], input_size: int, cfg):
        """Deterministic output layer

        :param device: Device on which a tensor/array is or will be allocated
        :type device: str or torch.device
        :param input_size
        :type int
        :param cfg: Configuration of Gaussian output layer
        :type OutputLayerCfg
        """

        super().__init__(device, input_size, cfg)

        self._net = nn.Sequential(nn.Linear(self._projection_in, cfg.output_size), cfg.output_activation())

    def forward(self, input, taken_actions, outputs_dict):
        """Act deterministically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions.shape, outputs)
            torch.Size([4096, 1]) {}
        """

        # Optional per-head pre-projection (see OutputLayerCfg.head).
        head_out = self._head(input)

        actions = self._output_scale * self._net(head_out)

        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        # SKRL 2.x model.act contract: 2-tuple ``(actions, outputs_dict)``.
        # The deterministic head has no ``log_prob`` so we leave it out
        # of the dict; PPO's value head reads predicted values from the
        # ``actions`` slot directly.
        return actions, outputs_dict
