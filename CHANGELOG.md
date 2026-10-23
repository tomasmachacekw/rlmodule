# Changelog

## 0.2.1

### Fixed

- A recurrent model resets its hidden state where an episode was truncated, not
  only where it terminated: a sequence that crosses a time-limit reset no longer
  carries the previous episode's hidden state.
- A squashed Gaussian head re-evaluates a stored sample from its pre-tanh value.
  Sampling returns `outputs["pre_squash"]`, and `RLModel` / `SharedRLModel` pass
  `inputs["taken_pre_squash"]` to the layer, which then uses that `u` exactly.
  Recovering `u` with `atanh(a / scale)` (still the path without it) loses float32
  precision for `|u|` above ~8.

## 0.2.0

### Added

- `GaussianLayerCfg.squash` (default `False`): a tanh-squashed Gaussian head,
  `a = output_scale * tanh(u)` with `u ~ N(mean, std)`. `log_prob` includes the
  tanh change-of-variables term, so it is the density of the bounded action (the
  SAC actor parameterisation). Requires `output_activation=nn.Identity`.
- `GaussianLayerCfg.entropy_reduction`: `"mean"` (default, as before) keeps the
  per-dimension entropy; `"sum"` returns the joint entropy, `(N, 1)`.
- Custom-layer spec: `AdaptiveMaxPool2d` and `SpatialSoftmax` (per-channel
  expected (x, y) keypoints, learnable temperature).
- `Cnn3dBackbone` / `Cnn3dBackboneCfg` for volumetric observations (flat DHWC
  input, `preset="custom"` only), the `cnn3d_backbone` `GraphNetwork` node, and
  `Conv3d`, `MaxPool3d`, `AdaptiveAvgPool3d`, `BatchNorm3d` in the layer spec.

### Fixed

- `CombinedLogStd` registers its sub-modules. A plain list assigned to
  `self._modules` (`nn.Module`'s child registry) hid their parameters from the
  optimizer and `state_dict()`, so a combined std stayed at its initial value.

### Upgrading from 0.1.x

- Policies using `CombinedLogStd` now train their std.
- Their checkpoints gain `<...>._sub_log_stds.<i>.*` keys. A 0.1.x checkpoint has
  none, so a strict `load_state_dict` reports them missing; with `strict=False`
  the stds start at their configured initial values, which is what the 0.1.x
  model used throughout.
