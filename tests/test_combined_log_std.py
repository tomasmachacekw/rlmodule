"""Regression tests for ``CombinedLogStd`` parameter registration.

``CombinedLogStd`` previously stored its sub-modules in a plain list assigned
to ``self._modules`` — which is ``nn.Module``'s *internal* child-module
registry. That silently prevented the sub-LogStd parameters from being
registered, so they never reached the optimizer and the policy std stayed
frozen. These tests pin the fix: sub-module parameters must be discoverable via
``named_parameters()`` and must receive gradients.
"""

import torch

from rlmodule.source.log_std import CombinedLogStd
from rlmodule.source.log_std_cfg import CombinedLogStdCfg, ParameterLogStdCfg


def _combined_cfg() -> CombinedLogStdCfg:
    return CombinedLogStdCfg(
        combination_method="max",
        combined_modules=[
            ParameterLogStdCfg(initial_log_std=0.5),
            ParameterLogStdCfg(initial_log_std=-0.5),
        ],
    )


def test_combined_log_std_registers_submodule_parameters() -> None:
    module = CombinedLogStd("cpu", input_size=8, output_size=4, cfg=_combined_cfg())
    log_params = [name for name, _ in module.named_parameters() if "_log_parameter" in name]
    # Two ParameterLogStd sub-modules -> two registered learnable tensors.
    assert len(log_params) == 2, f"expected 2 registered log_std params, got {log_params}"


def test_combined_log_std_submodule_params_receive_grad() -> None:
    module = CombinedLogStd("cpu", input_size=8, output_size=4, cfg=_combined_cfg())
    out = module(torch.zeros(3, 8))
    out.sum().backward()
    # The element-wise ``max`` routes gradient to the dominant sub-module; the
    # point is that the registered parameters are on the autograd graph at all
    # (a plain-list assignment detached them entirely).
    graphed = [p for name, p in module.named_parameters() if "_log_parameter" in name and p.grad is not None]
    assert graphed, "no CombinedLogStd sub-parameter received a gradient"
