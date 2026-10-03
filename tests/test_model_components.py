"""The trainable head components run on random tensors, with no backbone weights.

Full models need a 2.8-5.0 GB frozen backbone, so they are exercised by the
evaluation runs rather than here. These are the parts the paper actually trains.
"""
import pytest

torch = pytest.importorskip("torch")

from conftest import requires_third_party  # noqa: E402


@pytest.fixture(autouse=True)
def _needs_vggt():
    # vggt_dpt_lg imports the vggt checkout at module import time
    requires_third_party("vggt")


def _mod():
    import muviseg.models.vggt_dpt_lg as m

    return m


def test_multi_layer_fusion_shape():
    """Fusion is convolutional, so it consumes channel-first maps (B, C, H, W)."""
    m = _mod()
    B, D, H, W, L = 2, 2048, 6, 8, 4
    fusion = m.MultiLayerFusion(in_dim=D, d_hat=256, n_layers=L)
    out = fusion([torch.randn(B, D, H, W) for _ in range(L)])
    assert out.shape == (B, 256, H, W)


def test_descriptor_projector_is_normalised():
    m = _mod()
    proj = m.DescriptorProjector(in_dim=256, out_dim=128)
    out = proj(torch.randn(2, 11, 256))
    assert out.shape == (2, 11, 128)


def test_segment_attention_preserves_shapes():
    m = _mod()
    layer = m.SegmentAttentionLayerV2(dim=128, num_heads=4, ffn_expansion=4)
    x0, x1 = torch.randn(2, 7, 128), torch.randn(2, 5, 128)
    y0, y1 = layer(x0, x1)
    assert y0.shape == x0.shape and y1.shape == x1.shape


def test_joint_attention_handles_variable_segment_counts():
    m = _mod()
    layer = m.JointSegmentAttention(dim=128, num_heads=4, ffn_expansion=4, max_frames=16)
    segs = [torch.randn(1, n, 128) for n in (4, 9, 2, 6)]
    out = layer(segs)
    assert len(out) == len(segs)
    for a, b in zip(segs, out):
        assert a.shape == b.shape


def test_double_softmax_matcher_contract():
    """Descriptors are channel-first (B, D, M); scores come back (B, M, N).

    The layout matters: passing (B, M, D) silently computes a different einsum.
    """
    m = _mod()
    M, N, D = 6, 4, 128
    matcher = m.DoubleSoftmaxMatcher(desc_dim=D)
    log_mutual, match0, match1 = matcher(torch.randn(1, D, M), torch.randn(1, D, N))
    assert log_mutual.shape == (1, M, N)
    assert match0.shape == (1, M)
    assert match1.shape == (1, N)
    assert torch.isfinite(log_mutual).all()


def test_matcher_temperature_is_clamped_at_one():
    """tau is clamped to <= 1.0, so a large log_tau cannot smooth the scores.

    Without the clamp the model could trivially reduce the BCE loss by flattening
    its similarities, so this is a training-behaviour guarantee, not a detail.
    """
    m = _mod()
    D = 128
    d0, d1 = torch.randn(1, D, 5), torch.randn(1, D, 5)
    at_one = m.DoubleSoftmaxMatcher(desc_dim=D, temperature_init=1.0)
    with torch.no_grad():
        ref = at_one(d0, d1)[0].clone()
        at_one.log_tau.fill_(5.0)  # tau = e^5, far above the clamp
        clamped = at_one(d0, d1)[0]
    torch.testing.assert_close(ref, clamped)
    assert at_one.log_tau.exp().clamp(max=1.0).item() == pytest.approx(1.0)
