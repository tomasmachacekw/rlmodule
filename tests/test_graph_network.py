"""Unit tests for the declarative :class:`GraphNetwork` executor.

The GraphNetwork is the runtime counterpart of the Cyberwave RL Config
graph builder UI. Tests cover:

* Topological execution + cycle detection.
* Observation slicing via ``observation_input`` nodes.
* MLP / Residual / LayerNorm vector graphs (forward pass returns
  finite tensors of the expected width).
* Concat + residual_add fan-in.
* GRU recurrent graphs surface ``has_recurrent`` and accept the
  ``(states, terminated, rnn_inputs)`` tuple.
* CNN backbone graphs accept HWC slices from a flat observation
  tensor.
* Output dim is exposed via ``output_dim``.
"""

from __future__ import annotations

import pytest
import torch

from rlmodule.source.network import GraphNetwork
from rlmodule.source.network_cfg import GraphNetworkCfg


def _run_forward(graph: GraphNetwork, states: torch.Tensor):
    """Run forward and normalise the output to ``(features, info_or_none)``.

    ``GraphNetwork.forward`` returns a bare tensor for non-recurrent
    graphs (so it can plug straight into the skrl output layer) and a
    ``(features, rnn_outputs)`` tuple for recurrent graphs (so the
    SKRL hidden-state plumbing keeps working). The tests below all
    unpack via ``out, _ = _run_forward(...)``, so we project the bare
    tensor into a ``(features, None)`` tuple here.
    """

    if graph.has_recurrent:
        rnn_inputs = [torch.zeros(0)]  # placeholder, GraphNetwork builds its own state
        return graph(states, terminated=None, rnn_inputs=rnn_inputs)
    out = graph(states, terminated=None, rnn_inputs=None)
    return out, None


def test_graph_network_vector_mlp_forward_pass_returns_finite_features():
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(8)), "dim": 8},
            },
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [16, 32],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
        ],
        edges=[{"from": "obs", "to": "trunk"}],
        output_node_id="trunk",
        flat_input_size=8,
    )
    net = GraphNetwork(cfg)
    out, _ = _run_forward(net, torch.randn(4, 8))
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()
    assert net.output_dim == 32
    assert net.has_recurrent is False


def test_graph_network_residual_mlp_plus_layernorm():
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(8)), "dim": 8},
            },
            {
                "id": "trunk",
                "type": "residual_mlp",
                "params": {
                    "hidden_size": 16,
                    "num_blocks": 2,
                    "activation": torch.nn.ELU,
                    "layer_norm": True,
                    "final_layer_norm": True,
                },
            },
            {"id": "ln", "type": "layer_norm", "params": {}},
        ],
        edges=[
            {"from": "obs", "to": "trunk"},
            {"from": "trunk", "to": "ln"},
        ],
        output_node_id="ln",
        flat_input_size=8,
    )
    net = GraphNetwork(cfg)
    out, _ = _run_forward(net, torch.randn(3, 8))
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_graph_network_concat_merges_two_inputs():
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "a",
                "type": "observation_input",
                "params": {"indices": list(range(4)), "dim": 4},
            },
            {
                "id": "b",
                "type": "observation_input",
                "params": {"indices": list(range(4, 8)), "dim": 4},
            },
            {"id": "merge", "type": "concat", "params": {}},
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [16],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
        ],
        edges=[
            {"from": "a", "to": "merge"},
            {"from": "b", "to": "merge"},
            {"from": "merge", "to": "trunk"},
        ],
        output_node_id="trunk",
        flat_input_size=8,
    )
    net = GraphNetwork(cfg)
    out, _ = _run_forward(net, torch.randn(2, 8))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_graph_network_residual_add_requires_matching_dims():
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(8)), "dim": 8},
            },
            {
                "id": "branch_a",
                "type": "projection",
                "params": {"units": 16},
            },
            {
                "id": "branch_b",
                "type": "projection",
                "params": {"units": 16},
            },
            {"id": "merge", "type": "residual_add", "params": {}},
        ],
        edges=[
            {"from": "obs", "to": "branch_a"},
            {"from": "obs", "to": "branch_b"},
            {"from": "branch_a", "to": "merge"},
            {"from": "branch_b", "to": "merge"},
        ],
        output_node_id="merge",
        flat_input_size=8,
    )
    net = GraphNetwork(cfg)
    out, _ = _run_forward(net, torch.randn(2, 8))
    assert out.shape == (2, 16)


def test_graph_network_rejects_cycle():
    with pytest.raises(ValueError):
        GraphNetwork(
            GraphNetworkCfg(
                nodes=[
                    {
                        "id": "obs",
                        "type": "observation_input",
                        "params": {"indices": [0, 1, 2, 3], "dim": 4},
                    },
                    {
                        "id": "trunk",
                        "type": "mlp",
                        "params": {
                            "hidden_units": [4],
                            "activation": torch.nn.ELU,
                            "layer_norm": False,
                            "layer_norm_position": "pre",
                        },
                    },
                ],
                edges=[
                    {"from": "obs", "to": "trunk"},
                    {"from": "trunk", "to": "obs"},
                ],
                output_node_id="trunk",
                flat_input_size=4,
            )
        )


def test_graph_network_rejects_unknown_node_type():
    with pytest.raises(ValueError):
        GraphNetwork(
            GraphNetworkCfg(
                nodes=[
                    {
                        "id": "obs",
                        "type": "made_up_node",
                        "params": {},
                    }
                ],
                edges=[],
                output_node_id="obs",
                flat_input_size=4,
            )
        )


def test_graph_network_gru_marks_recurrent_and_exposes_output_dim():
    """Single GRU node in a graph: ``has_recurrent`` must flip to
    ``True`` and ``output_dim`` must reflect the GRU hidden width so
    SKRL can wire its output head.

    Exercising the full GRU forward path requires SKRL's hidden-state
    plumbing; that flow is covered end-to-end by the cyberwave-rl
    network-builder integration tests instead of from rlmodule.
    """

    seq_len = 4
    num_envs = 2
    hidden_size = 8
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(6)), "dim": 6},
            },
            {
                "id": "rec",
                "type": "gru",
                "params": {
                    "hidden_size": hidden_size,
                    "num_layers": 1,
                    "sequence_length": seq_len,
                    "num_envs": num_envs,
                },
            },
        ],
        edges=[{"from": "obs", "to": "rec"}],
        output_node_id="rec",
        flat_input_size=6,
    )
    net = GraphNetwork(cfg)
    assert net.has_recurrent is True
    assert net.output_dim == hidden_size


def test_graph_network_cnn_backbone_accepts_image_slice():
    # NatureCNN's filter sizes (8, 4, 3) need at least an 84x84 image;
    # we use the canonical Atari shape here to keep the test stable
    # against future backbone tweaks.
    h = w = 84
    c = 1
    obs_dim = h * w * c
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {
                    "indices": list(range(obs_dim)),
                    "dim": obs_dim,
                    "shape": [h, w, c],
                    "modality": "image",
                },
            },
            {
                "id": "cnn",
                "type": "cnn_backbone",
                "params": {
                    "preset": "nature",
                    "image_shape": [h, w, c],
                },
            },
        ],
        edges=[{"from": "obs", "to": "cnn"}],
        output_node_id="cnn",
        flat_input_size=obs_dim,
    )
    net = GraphNetwork(cfg)
    flat = torch.randn(2, obs_dim)
    out, _ = _run_forward(net, flat)
    assert out.dim() == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_graph_network_output_dim_is_exposed_after_construction():
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(4)), "dim": 4},
            },
            {
                "id": "proj",
                "type": "projection",
                "params": {"units": 12},
            },
        ],
        edges=[{"from": "obs", "to": "proj"}],
        output_node_id="proj",
        flat_input_size=4,
    )
    net = GraphNetwork(cfg)
    assert net.output_dim == 12


def test_graph_network_non_recurrent_forward_returns_bare_tensor():
    """Non-recurrent graphs must return a bare tensor, not ``(tensor, dict)``.

    Recurrent graphs need to expose the ``rnn_outputs`` dict so SKRL's
    hidden-state plumbing can fold it back into the rollout, but non-
    recurrent graphs are wired straight into the SKRL output layer
    (Gaussian/Deterministic) which expects a plain feature tensor.
    Returning a 2-tuple from a non-recurrent graph causes ``linear``
    calls downstream to fail with ``TypeError: argument 'input' must
    be Tensor, not tuple``. This test pins the contract so the SKRL
    integration cannot silently break again.
    """

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(4)), "dim": 4},
            },
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [8],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
        ],
        edges=[{"from": "obs", "to": "trunk"}],
        output_node_id="trunk",
        flat_input_size=4,
    )
    net = GraphNetwork(cfg)
    out = net(torch.randn(2, 4), terminated=None, rnn_inputs=None)
    assert isinstance(out, torch.Tensor), (
        f"Non-recurrent GraphNetwork.forward must return a bare tensor "
        f"so it can plug into SKRL output layers; got {type(out).__name__}"
    )
    assert out.shape == (2, 8)


def test_get_output_size_unwraps_graph_network_tuple_returns():
    """``get_output_size`` must work for both bare-tensor and tuple returns.

    ``get_output_size`` is used by SKRL to derive policy / value head
    widths from the trunk network's output. For recurrent
    ``GraphNetwork`` instances ``contains_rnn_module`` finds the RNN
    submodule and the helper feeds it ``(states, terminated, hidden)``
    explicitly. The non-recurrent branch sees a bare tensor; if a
    future refactor reintroduces a tuple return, the defensive
    ``isinstance(out, (tuple, list))`` unwrap in ``get_output_size``
    keeps the helper functional. This test exercises both flows so
    callers can rely on the contract.
    """

    from rlmodule.source.utils import get_output_size

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(6)), "dim": 6},
            },
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [4, 16],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
        ],
        edges=[{"from": "obs", "to": "trunk"}],
        output_node_id="trunk",
        flat_input_size=6,
    )
    net = GraphNetwork(cfg)
    assert get_output_size(net, (6,)) == 16


def test_graph_network_observation_input_uses_view_for_contiguous_indices():
    """Contiguous ``observation_input`` ranges must take the zero-copy
    ``narrow`` path so image-sized slices do not pay an ``index_select``
    copy on every rollout step."""

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(2, 6)), "dim": 4},
            },
            {"id": "ln", "type": "layer_norm", "params": {}},
        ],
        edges=[{"from": "obs", "to": "ln"}],
        output_node_id="ln",
        flat_input_size=8,
    )
    net = GraphNetwork(cfg)
    spec = net._input_specs["obs"]
    assert spec["mode"] == "view"
    assert spec["start"] == 2
    assert spec["length"] == 4

    states = torch.randn(3, 8)
    sliced = net._slice_input("obs", states)
    # ``narrow`` returns a view, so the sliced tensor shares the
    # underlying storage with the input. We compare storage data
    # pointers to verify this without materialising a fresh allocation.
    assert sliced.untyped_storage().data_ptr() == states.untyped_storage().data_ptr()
    assert torch.equal(sliced, states[:, 2:6])


def test_graph_network_observation_input_uses_gather_for_noncontiguous_indices():
    """Non-contiguous index lists must still work via ``index_select``,
    even though they cost a copy."""

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": [0, 2, 4, 7], "dim": 4},
            },
            {"id": "ln", "type": "layer_norm", "params": {}},
        ],
        edges=[{"from": "obs", "to": "ln"}],
        output_node_id="ln",
        flat_input_size=8,
    )
    net = GraphNetwork(cfg)
    spec = net._input_specs["obs"]
    assert spec["mode"] == "gather"

    states = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]])
    sliced = net._slice_input("obs", states)
    assert torch.equal(sliced, torch.tensor([[10.0, 30.0, 50.0, 80.0]]))


def test_graph_network_action_input_slices_action_tail():
    """SAC-style critics receive ``[states, taken_actions]`` concatenated
    along the feature dim (see ``RLModel.forward``). The ``action_input``
    node must therefore slice from offset ``flat_input_size`` — not from
    the head of the tensor, which would silently read observation
    features instead of the action."""

    obs_dim, action_dim = 4, 3
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(obs_dim)), "dim": obs_dim},
            },
            {
                "id": "act",
                "type": "action_input",
                "params": {"dim": action_dim},
            },
            {"id": "merge", "type": "concat", "params": {}},
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [8],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
        ],
        edges=[
            {"from": "obs", "to": "merge"},
            {"from": "act", "to": "merge"},
            {"from": "merge", "to": "trunk"},
        ],
        output_node_id="trunk",
        flat_input_size=obs_dim,
        action_input_size=action_dim,
    )
    net = GraphNetwork(cfg)

    act_spec = net._input_specs["act"]
    assert act_spec["kind"] == "action"
    assert act_spec["mode"] == "view"
    assert act_spec["start"] == obs_dim
    assert act_spec["length"] == action_dim

    states = torch.cat(
        [torch.full((2, obs_dim), 1.0), torch.full((2, action_dim), 5.0)], dim=-1
    )
    assert torch.equal(
        net._slice_input("obs", states), torch.full((2, obs_dim), 1.0)
    )
    assert torch.equal(
        net._slice_input("act", states), torch.full((2, action_dim), 5.0)
    )

    out = net(states, terminated=None, rnn_inputs=None)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_graph_network_action_input_without_flat_size_raises():
    """If ``flat_input_size`` is unknown we cannot compute the
    action-tail offset, so building the graph must fail loudly instead
    of silently slicing into the observation prefix."""

    with pytest.raises(ValueError, match="flat_input_size"):
        GraphNetwork(
            GraphNetworkCfg(
                nodes=[
                    {
                        "id": "act",
                        "type": "action_input",
                        "params": {"dim": 2},
                    },
                ],
                edges=[],
                output_node_id="act",
                # flat_input_size left unset and no input_states available.
                action_input_size=2,
            )
        )


def test_graph_network_skips_inactive_nodes_in_exec_order():
    """Nodes not transitively connected to ``output_node_id`` must not
    appear in the execution order and must not be built as submodules.
    Pruning keeps the rollout-time loop tight when the authoring UI
    leaves alternate heads / debug branches behind."""

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(4)), "dim": 4},
            },
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [8],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
            # Dead branch: reachable as a graph but not from the output.
            {
                "id": "dead_obs",
                "type": "observation_input",
                "params": {"indices": list(range(4)), "dim": 4},
            },
            {
                "id": "dead_proj",
                "type": "projection",
                "params": {"units": 16},
            },
        ],
        edges=[
            {"from": "obs", "to": "trunk"},
            {"from": "dead_obs", "to": "dead_proj"},
        ],
        output_node_id="trunk",
        flat_input_size=4,
    )
    net = GraphNetwork(cfg)
    assert "dead_obs" not in net._exec_order
    assert "dead_proj" not in net._exec_order
    # The dead projection should not be materialised as a submodule, so
    # it cannot contribute parameters or activations.
    assert "dead_proj" not in net._modules_by_id

    out = net(torch.randn(2, 4), terminated=None, rnn_inputs=None)
    assert out.shape == (2, 8)


def test_graph_network_ignores_dead_recurrent_node():
    """A recurrent node that is not on the path to ``output_node_id``
    must not flip ``has_recurrent`` or block construction. Without
    pruning, an authoring UI that left a stray GRU on the canvas would
    silently force SKRL to allocate hidden-state buffers and switch the
    forward signature, even though the GRU never runs."""

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(6)), "dim": 6},
            },
            {
                "id": "trunk",
                "type": "mlp",
                "params": {
                    "hidden_units": [16],
                    "activation": torch.nn.ELU,
                    "layer_norm": False,
                    "layer_norm_position": "pre",
                },
            },
            {
                "id": "dead_obs",
                "type": "observation_input",
                "params": {"indices": list(range(6)), "dim": 6},
            },
            {
                "id": "dead_rnn",
                "type": "gru",
                "params": {
                    "hidden_size": 4,
                    "num_layers": 1,
                    "sequence_length": 8,
                    "num_envs": 2,
                },
            },
        ],
        edges=[
            {"from": "obs", "to": "trunk"},
            {"from": "dead_obs", "to": "dead_rnn"},
        ],
        output_node_id="trunk",
        flat_input_size=6,
    )
    net = GraphNetwork(cfg)
    assert net.has_recurrent is False
    assert "dead_rnn" not in net._exec_order
    assert "dead_rnn" not in net._modules_by_id


def test_graph_network_input_specs_view_avoid_buffer_registration():
    """Contiguous slices should not register an index buffer at all —
    we only need ``(start, length)`` to call ``narrow``. The buffer is
    reserved for the genuine ``index_select`` path."""

    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(0, 4)), "dim": 4},
            },
            {"id": "ln", "type": "layer_norm", "params": {}},
        ],
        edges=[{"from": "obs", "to": "ln"}],
        output_node_id="ln",
        flat_input_size=4,
    )
    net = GraphNetwork(cfg)
    spec = net._input_specs["obs"]
    assert spec["mode"] == "view"
    assert "buffer_name" not in spec
    assert "_idx_node_obs" not in dict(net.named_buffers())


def test_graph_network_gru_forward_executes_in_rollout():
    """Exercise the recurrent branch of the executor end-to-end.

    The metadata-only test above checks ``has_recurrent`` / ``output_dim``;
    this one actually runs the GRU forward in eval/rollout mode, feeding a
    real hidden state and asserting the graph returns finite features plus an
    updated ``{"rnn": [...]}`` dict shaped for SKRL's hidden-state plumbing.
    """

    obs_dim = 6
    hidden_size = 8
    num_layers = 1
    num_envs = 3
    cfg = GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(obs_dim)), "dim": obs_dim},
            },
            {
                "id": "rec",
                "type": "gru",
                "params": {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "sequence_length": 4,
                    "num_envs": num_envs,
                },
            },
        ],
        edges=[{"from": "obs", "to": "rec"}],
        output_node_id="rec",
        flat_input_size=obs_dim,
    )
    net = GraphNetwork(cfg).eval()
    states = torch.randn(num_envs, obs_dim)
    h0 = torch.zeros(num_layers, num_envs, hidden_size)

    out, info = net(states, terminated=None, rnn_inputs=[h0])

    assert out.shape == (num_envs, hidden_size)
    assert torch.isfinite(out).all()
    assert "rnn" in info
    assert info["rnn"][0].shape == (num_layers, num_envs, hidden_size)


def _two_node_graph(trunk_spec: dict, *, in_dim: int = 6):
    """Build an ``observation_input -> trunk`` graph with the given trunk."""

    return GraphNetworkCfg(
        nodes=[
            {
                "id": "obs",
                "type": "observation_input",
                "params": {"indices": list(range(in_dim)), "dim": in_dim},
            },
            {"id": "trunk", **trunk_spec},
        ],
        edges=[{"from": "obs", "to": "trunk"}],
        output_node_id="trunk",
        flat_input_size=in_dim,
    )


def test_graph_network_mlp_requires_activation():
    """A required cfg field (``activation``) must be supplied explicitly;
    the executor refuses to invent a default."""

    cfg = _two_node_graph({"type": "mlp", "params": {"hidden_units": [8]}})
    with pytest.raises(ValueError, match="activation"):
        GraphNetwork(cfg)


def test_graph_network_residual_mlp_requires_hidden_size():
    cfg = _two_node_graph(
        {"type": "residual_mlp", "params": {"num_blocks": 2, "activation": torch.nn.ELU}}
    )
    with pytest.raises(ValueError, match="hidden_size"):
        GraphNetwork(cfg)


def test_graph_network_rnn_requires_num_envs():
    cfg = _two_node_graph(
        {
            "type": "gru",
            "params": {"hidden_size": 8, "num_layers": 1, "sequence_length": 4},
        }
    )
    with pytest.raises(ValueError, match="num_envs"):
        GraphNetwork(cfg)
