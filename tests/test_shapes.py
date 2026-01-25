import torch

from asa.asa import AddressedStateAttention


def test_shapes_basic():
    attn = AddressedStateAttention(
        embed_dim=32,
        num_heads=4,
        num_slots=8,
        use_content_read=True,
        use_slotspace_refine=True,
    )
    x = torch.randn(2, 5, 32)
    out, info = attn(x, return_info=True)
    assert out.shape == x.shape
    assert "read_weights" in info
    assert info["read_weights"].shape[:3] == (2, 5, 4)


def test_shapes_toggle():
    attn = AddressedStateAttention(
        embed_dim=32,
        num_heads=4,
        num_slots=8,
        use_content_read=False,
        use_slotspace_refine=False,
    )
    x = torch.randn(1, 7, 32)
    out, _ = attn(x)
    assert out.shape == x.shape
