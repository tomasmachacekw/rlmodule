# LayerNorm and Residual policies in `rlmodule`

This page documents the LayerNorm and residual-MLP knobs exposed by `rlmodule`
and when to reach for each.

## References

Both features are motivated by recent RL architecture work:

- **Ball, Smith, Kostrikov, Levine.** *Efficient online reinforcement learning
  with offline data.* 2023. <https://arxiv.org/abs/2302.02948>
  (RLPD — LayerNorm in the critic is central to the recipe.)
- **Lyle, Zheng, Khetarpal, Martens, van Hasselt, Pascanu, Dabney.*
  *Normalization and effective learning rates in reinforcement learning.*
  2024. <https://arxiv.org/abs/2407.01800>
  (Pre-activation LayerNorm stabilises effective learning rates under
  non-stationary RL targets.)

## When to use each knob

| Knob | Good default | When to turn on |
| --- | --- | --- |
| `MlpCfg.layer_norm: bool \| Sequence[bool]` | `False` | You want to deepen an MLP policy / value network, your updates are unstable, or you're seeing Q/value-target drift. Start with `True` (all layers). Use the per-layer list to taper LN usage (e.g. `[False, True, True, True]`). |
| `MlpCfg.layer_norm_position: "pre" \| "post"` | `"pre"` | Prefer `"pre"` (Lyle et al. 2024). `"post"` is provided for legacy / compatibility experiments. |
| `ResidualMlpCfg` (replaces `MlpCfg` for a network) | — | You want deep (> 4 layers) expressive policies, typically in a **shared** architecture or in a critic (RLPD). The residual skip makes effective depth trainable; LN keeps effective learning rates finite. |
| `OutputLayerCfg.head: NetworkCfg` | `None` | You're using a shared architecture and want different LN / residual treatment on the actor head vs. the critic head. |

## Configurations by architecture shape

### Separated (PPO / RLPD-style)

Each of `policy.network` and `value.network` is independent, so you just plug
the right cfg into each side.

```python
from rlmodule.skrl.torch import RLModelCfg, SeparatedRLModelCfg
from rlmodule.skrl.torch.network import MlpCfg, ResidualMlpCfg
from rlmodule.skrl.torch.output_layer import (
    DeterministicLayerCfg, GaussianLayerCfg, ParameterLogStdCfg,
)
import torch.nn as nn

SeparatedRLModelCfg(
    policy=RLModelCfg(
        network=MlpCfg(
            hidden_units=[512, 512, 512, 512],
            activation=nn.ELU,
            layer_norm=[False, True, True, True],   # per-layer LN
        ),
        output_layer=GaussianLayerCfg(
            log_std=ParameterLogStdCfg(
                min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0
            ),
        ),
    ),
    value=RLModelCfg(
        network=ResidualMlpCfg(                     # RLPD-style deep critic
            hidden_size=512,
            num_blocks=8,
            activation=nn.ELU,
            layer_norm=True,
            final_layer_norm=True,
        ),
        output_layer=DeterministicLayerCfg(),
    ),
)
```

### Shared

For a shared architecture the key addition is the `head` field on
`OutputLayerCfg`: it lets you place a dedicated per-head MLP
(with its own per-layer LayerNorm / residual structure) between the shared
trunk and the final projection. Without it, "heads" collapse to a single
`nn.Linear`.

```python
from rlmodule.skrl.torch import SharedRLModelCfg
from rlmodule.skrl.torch.network import MlpCfg, ResidualMlpCfg
from rlmodule.skrl.torch.output_layer import (
    DeterministicLayerCfg, GaussianLayerCfg, ParameterLogStdCfg,
)
import torch.nn as nn

SharedRLModelCfg(
    # Shared trunk: deep pre-LN residual MLP.
    network=ResidualMlpCfg(
        hidden_size=1024,
        num_blocks=6,
        activation=nn.ELU,
        layer_norm=True,
        final_layer_norm=True,
    ),
    policy_output_layer=GaussianLayerCfg(
        head=MlpCfg(                                # actor head
            hidden_units=[512, 256],
            activation=nn.ELU,
            layer_norm=[True, True],
        ),
        log_std=ParameterLogStdCfg(
            min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0
        ),
    ),
    value_output_layer=DeterministicLayerCfg(
        head=MlpCfg(                                # critic head
            hidden_units=[512, 256],
            activation=nn.ELU,
            layer_norm=[True, True],
        ),
    ),
)
```

The resulting forward pass is:

```
states -> ResidualMlp trunk ┬─► actor head MLP  ─► Linear ─► Gaussian
                            └─► critic head MLP ─► Linear ─► Deterministic
```

## Backward compatibility

- `MlpCfg.layer_norm` defaults to `False`, so **existing cfgs produce exactly
  the same module as before this change**.
- `OutputLayerCfg.head` defaults to `None`, so existing shared configs still
  compile to the pre-change behaviour (single `nn.Linear` as head).

## CNN backbones (Milestone A, in this change)

The same per-layer LayerNorm / residual trunk story also applies to vision
policies via :class:`rlmodule.source.network_cfg.CnnBackboneCfg` +
:class:`rlmodule.source.network_cfg.CnnMlpCfg`. Two named presets are
provided, `"nature"` (NatureCNN) and `"impala"` (IMPALA residual CNN), plus
a `"custom"` escape hatch. The CNN feeds its flat feature vector into any
`MlpCfg` / `ResidualMlpCfg`, so per-layer LN and residual blocks on the
post-conv trunk work exactly as above:

```python
from rlmodule.skrl.torch import SharedRLModelCfg
from rlmodule.skrl.torch.network import CnnBackboneCfg, CnnMlpCfg, MlpCfg
from rlmodule.skrl.torch.output_layer import GaussianLayerCfg, DeterministicLayerCfg, ParameterLogStdCfg
import torch.nn as nn

SharedRLModelCfg(
    network=CnnMlpCfg(
        input_states=(84, 84, 4),            # HWC, e.g. an Atari stack
        cnn=CnnBackboneCfg(preset="impala"),
        mlp=MlpCfg(hidden_units=[512, 256], activation=nn.ELU, layer_norm=True),
    ),
    policy_output_layer=GaussianLayerCfg(
        log_std=ParameterLogStdCfg(min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0),
    ),
    value_output_layer=DeterministicLayerCfg(),
)
```

Milestone A keeps the current SKRL flattened-obs convention (so memory and
env wrappers do not need to change). Milestone B (image-first observation
spaces, Dict obs, image-aware preprocessors) is tracked separately and
includes the `ImageRunningStandardScaler` / `PixelScaler` shipped in
`cyberwave-rl/source/skrl` alongside this change.

## Unit tests

See `tests/test_mlp_layer_norm.py`, `tests/test_residual_mlp.py`,
`tests/test_head_mlp.py`, `tests/test_build_model_integration.py` and
`tests/test_cnn_backbone.py` for examples of every combination covered above.
