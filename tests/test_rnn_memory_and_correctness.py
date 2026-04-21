"""Regression tests for the RNN / LSTM / GRU audit fixes.

These tests lock in the behaviour of four focused fixes landed in the
``rnn-audit-fixes`` plan:

1. Termination reset in ``LSTM.forward`` / ``RnnModule.forward`` is
   out-of-place (masked multiply) rather than in-place assignment.
2. ``SharedRLModel`` cache: identity hit check, eager release on mode
   switch and on cache miss, and no dangling reference to the last
   minibatch's forward.
3. ``RnnMlp`` / ``RnnMlpWithForwardedInput`` do not mutate the caller's
   ``cfg.rnn`` / ``cfg.mlp``.
4. ``Model.get_specification`` validates ``num_envs`` / ``num_layers`` /
   ``hidden_size`` up front instead of letting SKRL allocate a
   wrong-sized buffer that blows up later inside the LSTM/RNN forward.

Everything here is pure PyTorch (no SKRL), fast (<2s total), and lives
next to the existing ``test_skrl_startup.py`` / ``test_skrl_train_smoke.py``.
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
import torch.nn as nn

from rlmodule.skrl.torch import RLModelCfg, SharedRLModelCfg, build_model
from rlmodule.skrl.torch.network import (
    GruCfg,
    LstmCfg,
    MlpCfg,
    RnnCfg,
    RnnMlpCfg,
)
from rlmodule.skrl.torch.output_layer import (
    DeterministicLayerCfg,
    GaussianLayerCfg,
    ParameterLogStdCfg,
)
from rlmodule.source.network import GRU, LSTM, RNN, RnnMlp


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reproducible():
    torch.manual_seed(0)


N, L = 2, 4          # batch, sequence_length
HIN, HOUT = 6, 12    # input size, hidden size
NUM_LAYERS = 1


def _make_lstm() -> LSTM:
    return LSTM(
        LstmCfg(
            input_size=HIN,
            num_envs=N,
            num_layers=NUM_LAYERS,
            hidden_size=HOUT,
            sequence_length=L,
        )
    )


def _make_rnn() -> RNN:
    return RNN(
        RnnCfg(
            input_size=HIN,
            num_envs=N,
            num_layers=NUM_LAYERS,
            hidden_size=HOUT,
            sequence_length=L,
        )
    )


def _make_gru() -> GRU:
    return GRU(
        GruCfg(
            input_size=HIN,
            num_envs=N,
            num_layers=NUM_LAYERS,
            hidden_size=HOUT,
            sequence_length=L,
        )
    )


def _training_inputs():
    """Inputs matching what SKRL feeds an RnnBase during training."""
    states = torch.randn(N * L, HIN, requires_grad=True)
    h = torch.randn(NUM_LAYERS, N * L, HOUT, requires_grad=True)
    c = torch.randn(NUM_LAYERS, N * L, HOUT, requires_grad=True)
    return states, h, c


def _terminated_at(position: int) -> torch.Tensor:
    """Bool terminated mask of shape (N*L,) with termination at ``position``
    within the sequence for the first batch row."""
    t = torch.zeros(N, L, dtype=torch.bool)
    t[0, position] = True
    return t.view(-1)


# ---------------------------------------------------------------------------
# (1) Termination reset: backward stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, L - 2])
def test_lstm_termination_backward_is_stable(position):
    """LSTM forward + backward with a termination event anywhere but the very
    last step must succeed and produce finite grads for states and initial
    hidden/cell tensors.

    Terminations at ``position == L - 1`` are a no-op for the segmentation
    (they only affect the NEXT sequence's initial state which lives outside
    this call) and intentionally not covered here.
    """
    lstm = _make_lstm()
    lstm.train(True)

    states, h, c = _training_inputs()
    terminated = _terminated_at(position)

    out, _ = lstm(states, terminated, [h, c])
    loss = out.sum()
    loss.backward()

    assert torch.isfinite(out).all()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()


@pytest.mark.parametrize("module_factory", [_make_rnn, _make_gru])
@pytest.mark.parametrize("position", [0, 1, L - 2])
def test_rnn_module_termination_backward_is_stable(module_factory, position):
    """Same stability guarantee for vanilla RNN and GRU trunks."""
    module = module_factory()
    module.train(True)

    states = torch.randn(N * L, HIN, requires_grad=True)
    h = torch.randn(NUM_LAYERS, N * L, HOUT, requires_grad=True)
    terminated = _terminated_at(position)

    out, _ = module(states, terminated, [h])
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_hidden_reset_matches_manual_masked_computation():
    """For a single-layer LSTM the segmented termination-reset path must
    be equivalent to running the LSTM per-segment by hand and concatenating.

    This locks in the semantics of the out-of-place masked reset: a
    terminated position yields a zeroed initial hidden+cell state for the
    next segment, exactly like the original in-place assignment did.
    """
    lstm = _make_lstm()
    lstm.train(True)

    states = torch.randn(N * L, HIN)
    h0 = torch.randn(NUM_LAYERS, N * L, HOUT)
    c0 = torch.randn(NUM_LAYERS, N * L, HOUT)
    # Termination in the middle of the sequence.
    terminated = torch.zeros(N, L, dtype=torch.bool)
    terminated[:, 1] = True   # every env resets after step 1
    terminated_flat = terminated.view(-1)

    with torch.no_grad():
        out, out_dict = lstm(states, terminated_flat, [h0.clone(), c0.clone()])

        # Manual reference: replicate the segmentation by hand.
        rnn_input = states.view(N, L, HIN)
        h = h0.view(NUM_LAYERS, N, L, HOUT)[:, :, 0, :].contiguous()
        c = c0.view(NUM_LAYERS, N, L, HOUT)[:, :, 0, :].contiguous()

        seg_outputs = []
        # Segment 1: steps [0, 2) (inclusive of the step at which terminated==True)
        seg1, (h, c) = lstm.lstm(rnn_input[:, 0:2, :], (h, c))
        seg_outputs.append(seg1)
        # Apply termination mask (all envs reset after step 1)
        keep = (~terminated[:, 1]).to(h.dtype).view(1, -1, 1)
        h = h * keep
        c = c * keep
        # Segment 2: steps [2, L)
        seg2, (h, c) = lstm.lstm(rnn_input[:, 2:L, :], (h, c))
        seg_outputs.append(seg2)

        manual = torch.cat(seg_outputs, dim=1).reshape(N * L, HOUT)

    # Same forward path, same numerics.
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# (1b) RSS non-growth
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - non-POSIX fallback
        return float("nan")


@pytest.mark.parametrize("module_factory", [_make_lstm, _make_rnn, _make_gru])
def test_no_rss_growth_over_300_steps(module_factory):
    """Stress test: 300 training iterations with 20% terminations must not
    leak significant RSS. 20MB is generous (CPython peak fluctuations,
    allocator pooling) but catches any unbounded growth.
    """
    module = module_factory()
    module.train(True)
    opt = torch.optim.SGD(module.parameters(), lr=1e-3)

    torch.manual_seed(0)

    def _step():
        states = torch.randn(N * L, HIN)
        h = torch.randn(NUM_LAYERS, N * L, HOUT)
        c = torch.randn(NUM_LAYERS, N * L, HOUT)
        terminated = torch.rand(N * L) < 0.2
        rnn_inputs = [h, c] if isinstance(module, LSTM) else [h]
        out, _ = module(states, terminated, rnn_inputs)
        loss = out.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Warm-up: allocator settles.
    for _ in range(50):
        _step()
    gc.collect()
    baseline = _rss_mb()

    for _ in range(250):
        _step()
    gc.collect()
    final = _rss_mb()

    # If maxrss is not supported, skip the numeric assertion.
    if baseline == baseline and final == final:  # NaN check
        assert final - baseline < 20.0, (
            f"RSS grew by {final - baseline:.2f} MB over 250 steps "
            f"(baseline={baseline:.2f} MB, final={final:.2f} MB)"
        )


# ---------------------------------------------------------------------------
# (2) SharedRLModel cache: release + identity hit
# ---------------------------------------------------------------------------


def _gaussian_head(action_dim: int):
    return GaussianLayerCfg(
        output_size=action_dim,
        log_std=ParameterLogStdCfg(min_log_std=-5.0, max_log_std=2.0, initial_log_std=0.0),
    )


def _build_shared_lstm_model(action_dim: int = 2, state_dim: int = HIN):
    return build_model(
        SharedRLModelCfg(
            network=RnnMlpCfg(
                input_states=state_dim,
                rnn=LstmCfg(num_envs=N, num_layers=NUM_LAYERS, hidden_size=HOUT, sequence_length=L),
                mlp=MlpCfg(hidden_units=[8], activation=nn.ELU),
            ),
            device="cpu",
            policy_output_layer=_gaussian_head(action_dim),
            value_output_layer=DeterministicLayerCfg(output_size=1),
        )
    )


def _rnn_inputs(model):
    # Mirror the layout the LSTM backbone expects in training mode.
    h = torch.zeros(NUM_LAYERS, N * L, HOUT)
    c = torch.zeros(NUM_LAYERS, N * L, HOUT)
    return [h, c]


def test_shared_rlmodel_cache_released_after_backward():
    """After a forward+backward and a ``train(True)`` refresh, the cache
    must be fully released so the previous minibatch's activations /
    hidden states are not pinned.
    """
    model = _build_shared_lstm_model()
    model.train(True)

    states = torch.randn(N * L, HIN)
    inputs = {"states": states, "terminated": torch.zeros(N * L, dtype=torch.bool), "rnn": _rnn_inputs(model)}

    actions, log_prob, _ = model(inputs, role="policy")
    value, _, _ = model(inputs, role="value")
    loss = actions.sum() + value.sum() + log_prob.sum()
    loss.backward()

    # Cache is populated from the last forward.
    assert model._cached_states is not None
    assert model._cached_output is not None

    # A train-mode refresh drops the references to the previous forward.
    model.train(True)

    assert model._cached_states is None
    assert model._cached_output is None


def test_shared_rlmodel_cache_identity_hit_and_miss():
    """Two consecutive calls with the same ``states`` tensor hit the cache
    (trunk is invoked exactly once); a third call with a freshly allocated
    tensor forces a recompute.
    """
    model = _build_shared_lstm_model()
    model.train(True)

    trunk = model._net
    forward_calls = {"n": 0}
    original_forward = trunk.forward

    def _counting_forward(*args, **kwargs):
        forward_calls["n"] += 1
        return original_forward(*args, **kwargs)

    trunk.forward = _counting_forward  # type: ignore[assignment]

    try:
        states = torch.randn(N * L, HIN)
        inputs = {"states": states, "terminated": torch.zeros(N * L, dtype=torch.bool), "rnn": _rnn_inputs(model)}

        model(inputs, role="policy")
        first_state_ref = model._cached_states
        model(inputs, role="value")
        assert forward_calls["n"] == 1, "Same-identity states should reuse the cached trunk output."
        assert model._cached_states is first_state_ref

        # Value-equal but identity-different states must force a recompute.
        inputs2 = {"states": states.clone(), "terminated": inputs["terminated"], "rnn": _rnn_inputs(model)}
        model(inputs2, role="policy")
        assert forward_calls["n"] == 2
        assert model._cached_states is inputs2["states"]
    finally:
        trunk.forward = original_forward  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# (3) Cfg non-mutation
# ---------------------------------------------------------------------------


def test_rnn_mlp_does_not_mutate_user_cfg():
    """``RnnMlp.__init__`` must not write into the user's nested ``cfg.rnn`` /
    ``cfg.mlp`` objects. This protects ``SeparatedRLModelCfg`` setups that
    share nested configs between policy and value.
    """
    rnn_cfg = RnnCfg(
        input_size=0,
        num_envs=N,
        num_layers=NUM_LAYERS,
        hidden_size=HOUT,
        sequence_length=L,
    )
    mlp_cfg = MlpCfg(input_size=0, hidden_units=[16, 16], activation=nn.ELU)
    cfg = RnnMlpCfg(input_size=HIN, rnn=rnn_cfg, mlp=mlp_cfg)

    _ = RnnMlp(cfg)

    # User's originals are untouched.
    assert rnn_cfg.input_size == 0
    assert mlp_cfg.input_size == 0
    # ``configclass`` may have produced its own copies of the nested cfgs on
    # the outer ``RnnMlpCfg`` (that's a shallow ``dataclasses.replace`` that
    # re-wraps), but either way ``RnnMlp.__init__`` must not write into
    # whatever is sitting at ``cfg.rnn`` / ``cfg.mlp``.
    assert cfg.rnn.input_size == 0
    assert cfg.mlp.input_size == 0


# ---------------------------------------------------------------------------
# (4) num_envs validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_num_envs", [0, -1])
def test_num_envs_misconfig_is_caught(bad_num_envs):
    """``Model.get_specification`` must raise a clear ``ValueError`` for
    obviously-wrong ``num_envs`` rather than silently producing a garbage
    size that blows up later inside SKRL.
    """
    model = build_model(
        RLModelCfg(
            network=RnnMlpCfg(
                input_states=HIN,
                rnn=LstmCfg(
                    num_envs=bad_num_envs,
                    num_layers=NUM_LAYERS,
                    hidden_size=HOUT,
                    sequence_length=L,
                ),
                mlp=MlpCfg(hidden_units=[8], activation=nn.ELU),
            ),
            device="cpu",
            output_layer=_gaussian_head(2),
        )
    )
    with pytest.raises(ValueError, match="num_envs"):
        model.get_specification()


def test_num_envs_happy_path_returns_expected_shape():
    """Sanity check that a correctly configured RNN still reports the
    expected buffer spec after the validation was added."""
    model = build_model(
        RLModelCfg(
            network=RnnMlpCfg(
                input_states=HIN,
                rnn=LstmCfg(num_envs=N, num_layers=NUM_LAYERS, hidden_size=HOUT, sequence_length=L),
                mlp=MlpCfg(hidden_units=[8], activation=nn.ELU),
            ),
            device="cpu",
            output_layer=_gaussian_head(2),
        )
    )
    spec = model.get_specification()
    assert spec["rnn"]["sequence_length"] == L
    assert spec["rnn"]["sizes"] == [(NUM_LAYERS, N, HOUT), (NUM_LAYERS, N, HOUT)]
