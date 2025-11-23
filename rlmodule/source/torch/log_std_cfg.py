
from dataclasses import MISSING

from rlmodule.source.torch.log_std import CombinedLogStd, LogStd, NNLogStd, ParameterLogStd
from rlmodule.source.torch.network_cfg import MlpCfg


from rlmodule.source.nvidia_utils import configclass


@configclass
class LogStdCfg:
    """Configuration for base `LogStd` module."""

    class_type: type[LogStd] = LogStd

    clip_log_std: bool = True
    """Flag to indicate whether the log standard deviations should be clipped."""

    min_log_std: float = -20.0
    """Minimum value of the log standard deviation."""

    max_log_std: float = 2.0
    """Maximum value of the log standard deviation."""


@configclass
class ParameterLogStdCfg(LogStdCfg):
    """Configuration for `ParameterLogStd` module."""

    class_type: type[LogStd] = ParameterLogStd

    initial_log_std: float = 0.0
    """Initial value for the log standard deviation."""


@configclass
class NNLogStdCfg(LogStdCfg):
    """Configuration for `NNLogStd` module."""

    class_type: type[LogStd] = NNLogStd

    network_cfg: MlpCfg | None = None
    """
    Config for the hidden part of the network.

    Input and output sizes are computed on runtime.
    Output layer is automatically connected with linear layer with
    identity activation function.

    If network_cfg is None, then input is connected to output with linear
    layer.
    """


@configclass
class CombinedLogStdCfg(LogStdCfg):
    """Configuration for `CombinedLogStd` module."""

    class_type: type[LogStd] = CombinedLogStd

    combination_method: str = MISSING
    """
    Method used to combine the standard deviations from multiple modules.

    Currently supported modes:
    - max
    - min
    """

    combined_modules: list[LogStdCfg] = MISSING
    """List of modules to be combined."""

    combination_constants: list[float] | None = None
    """List of constants used to multiple Std from each module before combination."""
