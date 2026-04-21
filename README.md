# rlmodule
Flexible reinforcement learning models instantiators library

Function approximators simple, but still strong. RNN - GRU - LSTM / SAC

Now it only supports skrl, but is intended to be library agnostic - in later expansion

try other algos
shared separate model


<!--
# todo
# some envs have by default range on which they work, like pendulum -2, 2, need to propaget that in.
# cnn
# WRITE README tutorial
# Run & fix pre-commit
# annotate cfgs in modules - why doesn't work TYPE_CHECKING
# extensive comments
# Launch new version to pip
# Import new version in Isaac-lab
# lazy linear? what is it ?
# random model run add function back
-->


## How to run

### Install rlmodule from local code

- Make sure you are in base rlmodule dict.

- Start virtual env.
```
python3 -m venv venv
source venv/bin/activate
```
- Install library from local code
```
pip install -e .
```
Note: sometimes installation may fail, if there is a run/ dir present, you may need to remove it (TODO: fix)
```
rm -rf runs
```

### Deep policies: LayerNorm and Residual MLPs

`rlmodule` supports per-layer LayerNorm on `MlpCfg` and a SimBa-style residual
MLP (`ResidualMlpCfg`). These work for both separated and shared
architectures, with independent per-head granularity via the new
`OutputLayerCfg.head` field. See [docs/layer_norm_and_residual.md](docs/layer_norm_and_residual.md)
for motivation, references, and copy-pasteable cfg snippets.

### Run chosen example
```
python3 rlmodule/skrl/torch/examples/gymnasium/pendulum_ppo_mlp_separate_model.py
```

Optional: observe run results in Tensorboard

```
tensorboard --logdir=runs/
```



## Update new version to PIP

Change version name in pyproject.toml

```
pip install build twine
```

```
rm -rf runs
python -m build
```

```
twine upload dist/*
```
