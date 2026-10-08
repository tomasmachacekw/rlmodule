"""Unit tests for CNN Milestone A (flat-vector CNN, no SKRL changes).

These cover the :class:`CnnBackbone` presets and the
:class:`CnnMlp` composite module, which wires a CNN backbone into the same
MLP / ResidualMlp machinery used elsewhere in rlmodule.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rlmodule.source.network import Cnn3dBackbone, CnnBackbone, CnnMlp, CnnProprioMlp
from rlmodule.source.network_cfg import (
    Cnn3dBackboneCfg,
    CnnBackboneCfg,
    CnnMlpCfg,
    CnnProprioMlpCfg,
    MlpCfg,
    ResidualMlpCfg,
)


@pytest.mark.parametrize("preset,input_states", [
    ("nature", (84, 84, 4)),
    ("impala", (64, 64, 3)),
])
def test_cnn_backbone_preset_forward(preset, input_states):
    cfg = CnnBackboneCfg(input_states=input_states, preset=preset)
    net = CnnBackbone(cfg)
    h, w, c = input_states
    flat = torch.randn(3, h * w * c)
    out = net(flat)
    assert out.dim() == 2
    assert out.shape[0] == 3
    assert out.shape[1] > 0


def test_cnn_backbone_custom_preset_requires_layers():
    with pytest.raises(ValueError):
        CnnBackbone(CnnBackboneCfg(input_states=(8, 8, 1), preset="custom"))


def test_cnn_backbone_invalid_preset_raises():
    with pytest.raises(ValueError):
        CnnBackbone(CnnBackboneCfg(input_states=(8, 8, 1), preset="does-not-exist"))


def test_cnn_backbone_custom_preset_uses_user_layers():
    cfg = CnnBackboneCfg(
        input_states=(13, 13, 1),
        preset="custom",
        layers=[nn.Conv2d(1, 16, kernel_size=3, stride=2), nn.ReLU(), nn.Flatten()],
    )
    net = CnnBackbone(cfg)
    flat = torch.randn(2, 13 * 13 * 1)
    out = net(flat)
    assert out.shape[0] == 2 and out.dim() == 2


# ---------------------------------------------------------------------------
# Cnn3dBackbone (volumetric / occupancy-grid CNN, flat-vector contract)
# ---------------------------------------------------------------------------


def _occ_3d_layers():
    """A small Conv3d stack that flattens to a fixed-size embedding regardless of
    the (possibly anisotropic) input volume dims."""
    return [
        nn.Conv3d(1, 8, kernel_size=3, padding=1),
        nn.ELU(),
        nn.MaxPool3d(kernel_size=2, ceil_mode=True),
        nn.Conv3d(8, 16, kernel_size=3, padding=1),
        nn.ELU(),
        nn.AdaptiveAvgPool3d((2, 2, 2)),
        nn.Flatten(),
    ]


@pytest.mark.parametrize("volume", [(5, 5, 5, 1), (8, 8, 4, 1), (6, 4, 3, 1)])
def test_cnn3d_backbone_custom_forward(volume):
    cfg = Cnn3dBackboneCfg(input_states=volume, preset="custom", layers=_occ_3d_layers())
    net = Cnn3dBackbone(cfg)
    d, h, w, c = volume
    flat = torch.randn(3, d * h * w * c)
    out = net(flat)
    assert out.dim() == 2
    assert out.shape[0] == 3
    # AdaptiveAvgPool3d((2,2,2)) over 16 channels -> 16*2*2*2 = 128 features.
    assert out.shape[1] == 128


def test_cnn3d_backbone_custom_preset_requires_layers():
    with pytest.raises(ValueError):
        Cnn3dBackbone(Cnn3dBackboneCfg(input_states=(5, 5, 5, 1), preset="custom"))


def test_cnn3d_backbone_rejects_2d_presets():
    for preset in ("nature", "impala", "does-not-exist"):
        with pytest.raises(ValueError):
            Cnn3dBackbone(
                Cnn3dBackboneCfg(
                    input_states=(5, 5, 5, 1), preset=preset, layers=_occ_3d_layers()
                )
            )


def test_cnn3d_backbone_rejects_wrong_flat_size():
    cfg = Cnn3dBackboneCfg(input_states=(5, 5, 5, 1), preset="custom", layers=_occ_3d_layers())
    net = Cnn3dBackbone(cfg)
    bad = torch.randn(2, 5 * 5 * 5 * 1 + 7)  # off-by-7 flat slice
    with pytest.raises(ValueError, match="expected a flat input of size 125"):
        net(bad)


def test_cnn3d_backbone_requires_4d_input_states():
    with pytest.raises(ValueError, match="4-tuple in DHWC"):
        Cnn3dBackbone(
            Cnn3dBackboneCfg(input_states=(5, 5, 1), preset="custom", layers=_occ_3d_layers())
        )


def test_cnn3d_backbone_dhwc_reshape_is_correct():
    """A flat DHWC vector must map to (B, C, D, H, W) so 3D locality is preserved.

    We use an identity-ish Conv3d (Flatten only) and check the permutation maps
    the right elements: element at flat index for (d, h, w) with C=1 must land at
    spatial position (d, h, w) of the volume the conv sees.
    """
    d, h, w, c = 2, 3, 4, 1
    net = Cnn3dBackbone(
        Cnn3dBackboneCfg(
            input_states=(d, h, w, c),
            preset="custom",
            # AvgPool over the H,W plane per depth slice so each output element is
            # the mean of one (h,w) plane -> lets us verify the depth axis ordering.
            layers=[nn.AdaptiveAvgPool3d((d, 1, 1)), nn.Flatten()],
        )
    )
    # Build a volume where every voxel in depth-slice k has value k.
    vol = torch.zeros(1, d, h, w, c)
    for k in range(d):
        vol[:, k] = float(k)
    flat = vol.reshape(1, -1)
    out = net(flat)  # shape (1, d)
    assert out.shape == (1, d)
    assert torch.allclose(out[0], torch.arange(d, dtype=torch.float32))


def test_cnn3d_backbone_backward_reaches_input_and_params():
    torch.manual_seed(0)
    volume = (5, 5, 5, 1)
    cfg = Cnn3dBackboneCfg(input_states=volume, preset="custom", layers=_occ_3d_layers())
    net = Cnn3dBackbone(cfg)
    d, h, w, c = volume
    flat = torch.randn(2, d * h * w * c, requires_grad=True)
    out = net(flat)
    out.sum().backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()
    conv_params = [
        (name, p) for name, p in net.named_parameters() if isinstance_conv3d_param(name)
    ]
    assert conv_params, "expected Conv3d parameters in the backbone"
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all()


def isinstance_conv3d_param(name: str) -> bool:
    # Conv3d submodules live at indices 0 and 3 of the sequential stack here.
    return name.startswith("cnn.0.") or name.startswith("cnn.3.")


def test_cnn_mlp_with_layer_norm_trunk():
    cfg = CnnMlpCfg(
        input_states=(84, 84, 4),
        cnn=CnnBackboneCfg(preset="nature"),
        mlp=MlpCfg(hidden_units=[128, 64], activation=nn.ELU, layer_norm=True),
    )
    net = CnnMlp(cfg)
    flat = torch.randn(2, 84 * 84 * 4)
    out = net(flat)
    assert out.shape == (2, 64)
    assert net.hidden_size == 64


def test_cnn_mlp_with_residual_trunk_backward():
    torch.manual_seed(0)
    cfg = CnnMlpCfg(
        input_states=(64, 64, 3),
        cnn=CnnBackboneCfg(preset="impala"),
        mlp=ResidualMlpCfg(
            hidden_size=128, num_blocks=2, activation=nn.ELU, layer_norm=True,
        ),
    )
    net = CnnMlp(cfg)
    flat = torch.randn(2, 64 * 64 * 3, requires_grad=True)
    out = net(flat)
    assert out.shape == (2, 128)

    out.sum().backward()
    # Every leaf parameter should receive a finite gradient.
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()


def test_cnn_mlp_input_size_override_is_respected():
    """If the cfg's flat input_size is given explicitly, the shape probe uses it."""
    cfg = CnnMlpCfg(
        input_states=(84, 84, 4),
        input_size=84 * 84 * 4,
        cnn=CnnBackboneCfg(preset="nature"),
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU),
    )
    net = CnnMlp(cfg)
    flat = torch.randn(1, 84 * 84 * 4)
    assert net(flat).shape == (1, 32)


def test_cnn_proprio_mlp_fuses_depth_and_vector_paths():
    cfg = CnnProprioMlpCfg(
        input_size=12 + 32 * 32,
        image_shape=(32, 32, 1),
        proprio_dim=12,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=32,
        mlp=MlpCfg(hidden_units=[64, 32], activation=nn.ELU, layer_norm=True),
    )
    net = CnnProprioMlp(cfg)
    x = torch.randn(4, 12 + 32 * 32)
    out = net(x)

    assert out.shape == (4, 32)
    assert net.hidden_size == 32
    assert any(isinstance(module, nn.LayerNorm) for module in net.visual_projection.modules())


def test_cnn_proprio_mlp_accepts_appended_critic_actions():
    action_dim = 9
    cfg = CnnProprioMlpCfg(
        input_size=12 + 32 * 32 + action_dim,
        image_shape=(32, 32, 1),
        proprio_dim=12,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=32,
        mlp=MlpCfg(hidden_units=[64, 32], activation=nn.ELU, layer_norm=True),
    )
    net = CnnProprioMlp(cfg)
    x = torch.randn(4, 12 + 32 * 32 + action_dim)
    out = net(x)

    assert out.shape == (4, 32)
    assert net.extra_input_dim == action_dim


def test_cnn_proprio_mlp_rejects_bad_split():
    cfg = CnnProprioMlpCfg(
        input_size=10,
        image_shape=(8, 8, 1),
        proprio_dim=3,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 8, 8)),
        mlp=MlpCfg(hidden_units=[16], activation=nn.ELU),
    )

    with pytest.raises(ValueError, match="input_size does not match"):
        CnnProprioMlp(cfg)


def test_cnn_proprio_mlp_backward_reaches_both_paths():
    torch.manual_seed(0)
    cfg = CnnProprioMlpCfg(
        input_size=6 + 32 * 32,
        image_shape=(32, 32, 1),
        proprio_dim=6,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=16,
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
    )
    net = CnnProprioMlp(cfg)
    x = torch.randn(3, 6 + 32 * 32, requires_grad=True)

    net(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad[:, :6]).all()
    assert torch.isfinite(x.grad[:, 6:]).all()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Visual encoder autocast / fp16 overflow regression tests
# ---------------------------------------------------------------------------
#
# History: the very first version of ``CnnProprioMlp`` always autocasted the
# visual encoder to ``torch.float16`` whenever the image lived on CUDA. That is
# unsafe for deep residual CNNs (Impala): late in training, intermediate
# activations in the visual projection's ``Linear`` can exceed the fp16
# representable range (~65504) and produce ``inf`` -> ``nan``, which then
# propagates through the rest of the policy and yields NaN actions for every
# env in lockstep.
#
# The cfg now exposes ``visual_amp_dtype`` and the default is
# ``torch.bfloat16``: 2-byte activation storage like ``float16`` (so we keep
# the original GPU-memory budget) but with ``float32``'s dynamic range (so we
# cannot overflow). ``None`` and ``torch.float16`` remain available; the
# tests below lock in the contract for all three.


def _make_cfg(image_shape=(32, 32, 1), proprio_dim=6, **overrides):
    h, w, c = image_shape
    cfg = CnnProprioMlpCfg(
        input_size=proprio_dim + h * w * c,
        image_shape=image_shape,
        proprio_dim=proprio_dim,
        cnn=CnnBackboneCfg(preset="impala", impala_widths=(8, 16, 16)),
        cnn_embedding_size=16,
        mlp=MlpCfg(hidden_units=[32], activation=nn.ELU, layer_norm=True),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_visual_amp_dtype_default_is_bfloat16():
    """Default config uses bf16 autocast: fp16-equivalent memory with fp32 range.

    Regression for two distinct issues:
    - ``ValueError: Action is nan for envs`` when the legacy default forced
      ``float16`` autocast on CUDA and Impala activations overflowed.
    - ``Warp CUDA error 2: out of memory`` when an earlier fix flipped the
      default to ``None`` (fp32) and doubled visual-encoder activation memory.
    """
    cfg = _make_cfg()
    assert cfg.visual_amp_dtype is torch.bfloat16
    net = CnnProprioMlp(cfg)
    assert net._visual_amp_dtype is torch.bfloat16


def test_visual_path_finite_with_default():
    """Forward pass on CPU produces finite outputs with the default cfg."""
    cfg = _make_cfg()
    net = CnnProprioMlp(cfg)
    x = torch.randn(2, cfg.input_size)
    out = net(x)
    assert torch.isfinite(out).all()


def test_visual_path_finite_with_explicit_none():
    """``visual_amp_dtype=None`` runs at the parameter dtype (fp32) and stays finite."""
    cfg = _make_cfg(visual_amp_dtype=None)
    net = CnnProprioMlp(cfg)
    x = torch.randn(2, cfg.input_size)
    out = net(x)
    assert torch.isfinite(out).all()
    assert net._visual_amp_dtype is None


def test_visual_path_handles_large_inputs_without_overflow_in_fp32():
    """``visual_amp_dtype=None`` must not overflow on large activations.

    Locks in numerical stability for the safest fallback path so a user
    switching to ``None`` for debugging never sees overflow.
    """
    torch.manual_seed(0)
    cfg = _make_cfg(visual_amp_dtype=None)
    net = CnnProprioMlp(cfg)
    x = torch.full((1, cfg.input_size), 50.0)
    out = net(x)
    assert torch.isfinite(out).all()


def test_visual_path_handles_large_inputs_without_overflow_in_bf16_on_cuda():
    """The default bf16 path also handles inputs that would overflow fp16.

    This is the primary positive guard: bf16's ``float32``-equivalent dynamic
    range means realistic mid-training activations never produce ``inf``.
    """
    if not torch.cuda.is_available():
        pytest.skip("bf16 autocast only triggers on CUDA tensors.")
    torch.manual_seed(0)
    cfg = _make_cfg()  # default visual_amp_dtype=torch.bfloat16
    net = CnnProprioMlp(cfg).cuda()
    # Scale weights up so a fp16 path would overflow; bf16 must stay finite.
    with torch.no_grad():
        for p in net.cnn.parameters():
            p.mul_(5.0)
        for p in net.visual_projection.parameters():
            p.mul_(5.0)
    x = torch.randn(2, cfg.input_size, device="cuda")
    out = net(x)
    assert torch.isfinite(out).all()


def test_visual_amp_dtype_fp16_logs_warning_on_nonfinite_output():
    """Opt-in fp16 autocast must surface a one-shot warning when the encoder
    produces a non-finite output, so the user can diagnose ``Action is nan``
    failures fast instead of staring at silent NaN actions for hours.
    """
    if not torch.cuda.is_available():
        pytest.skip("fp16 autocast warning only fires on CUDA tensors.")
    torch.manual_seed(0)
    cfg = _make_cfg(visual_amp_dtype=torch.float16)
    net = CnnProprioMlp(cfg).cuda()
    original = net.visual_projection

    class _InfProjection(nn.Module):
        def __init__(self, real):
            super().__init__()
            self.real = real

        def forward(self, x):
            out = self.real(x)
            return torch.full_like(out, float("inf"))

    net.visual_projection = _InfProjection(original)
    x = torch.randn(2, cfg.input_size, device="cuda")
    _ = net(x)
    assert net._nonfinite_visual_warned is True
    _ = net(x)
    assert net._nonfinite_visual_warned is True


# ---------------------------------------------------------------------------
# AdaptiveMaxPool2d / SpatialSoftmax custom layers
# ---------------------------------------------------------------------------


def test_spatial_softmax_finds_a_single_hot_pixel():
    from rlmodule.source.network import SpatialSoftmax

    layer = SpatialSoftmax(temperature=0.01, learnable_temperature=False)
    x = torch.zeros(2, 3, 5, 9)
    x[0, 0, 0, 0] = 10.0  # top-left
    x[0, 1, 4, 8] = 10.0  # bottom-right
    x[0, 2, 2, 4] = 10.0  # centre
    x[1, 0, 1, 6] = 10.0  # row 1 of 5 -> y -0.5; column 6 of 9 -> x 0.5
    out = layer(x)
    assert out.shape == (2, 6)
    torch.testing.assert_close(out[0], torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0]), atol=1e-4, rtol=0)
    torch.testing.assert_close(out[1, :2], torch.tensor([0.5, -0.5]), atol=1e-4, rtol=0)
    # A flat map spreads the attention uniformly: the keypoint is the centre.
    torch.testing.assert_close(out[1, 2:], torch.zeros(4), atol=1e-6, rtol=0)


def test_spatial_softmax_temperature_is_trainable_and_positive():
    from rlmodule.source.network import SpatialSoftmax

    layer = SpatialSoftmax(temperature=0.5)
    assert float(layer.log_temperature.detach().exp()) == pytest.approx(0.5)
    x = torch.randn(4, 2, 6, 6, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.log_temperature.grad is not None
    fixed = SpatialSoftmax(learnable_temperature=False)
    assert not list(fixed.parameters())
    with pytest.raises(ValueError):
        SpatialSoftmax(temperature=0.0)


def test_spatial_softmax_keeps_the_input_dtype():
    from rlmodule.source.network import SpatialSoftmax

    out = SpatialSoftmax()(torch.randn(2, 3, 4, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_adaptive_max_pool_keeps_a_small_peak_that_avg_pool_dilutes():
    from rlmodule.source.network import _build_nn_layers_from_spec

    mx, = _build_nn_layers_from_spec([{"type": "AdaptiveMaxPool2d", "output_size": [2, 2]}])
    av, = _build_nn_layers_from_spec([{"type": "AdaptiveAvgPool2d", "output_size": [2, 2]}])
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 1, 1] = 1.0
    assert float(mx(x)[0, 0, 0, 0]) == 1.0
    assert float(av(x)[0, 0, 0, 0]) == pytest.approx(1 / 16)


def test_cnn_backbone_custom_spec_with_spatial_softmax_and_max_pool_branches():
    """Two branches over one 32x48 depth image, as a GraphNetwork builds them."""
    from rlmodule.source.network import GraphNetwork
    from rlmodule.source.network_cfg import GraphNetworkCfg

    h, w = 32, 48
    conv = {"type": "Conv2d", "in_channels": 1, "out_channels": 8, "kernel_size": 3, "padding": 1}
    cfg = GraphNetworkCfg(
        nodes=[
            {"id": "img", "type": "observation_input",
             "params": {"indices": list(range(h * w)), "dim": h * w, "shape": [h, w, 1], "modality": "image"}},
            {"id": "keypoints", "type": "cnn_backbone",
             "params": {"preset": "custom", "image_shape": [h, w, 1],
                        "layers": [conv, {"type": "ELU"}, {"type": "SpatialSoftmax", "temperature": 1.0}]}},
            {"id": "maxpool", "type": "cnn_backbone",
             "params": {"preset": "custom", "image_shape": [h, w, 1],
                        "layers": [conv, {"type": "ELU"}, {"type": "AdaptiveMaxPool2d", "output_size": [4, 6]},
                                   {"type": "Flatten"}]}},
            {"id": "cat", "type": "concat", "params": {}},
        ],
        edges=[{"from": "img", "to": "keypoints"}, {"from": "img", "to": "maxpool"},
               {"from": "keypoints", "to": "cat"}, {"from": "maxpool", "to": "cat"}],
        output_node_id="cat",
        flat_input_size=h * w,
    )
    net = GraphNetwork(cfg)
    out = net(torch.rand(3, h * w))
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (3, 8 * 2 + 8 * 4 * 6)
    assert torch.isfinite(out).all()
    assert out[:, :16].abs().max() <= 1.0
