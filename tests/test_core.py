import pytest
import torch
import torch.nn as nn

from afnn import AFNN, AFNNConfig
from afnn.fractal import FractalBlock
from afnn.layers import ConvBlock


def small_cfg(**kw):
    base = dict(input_size=16, num_classes=4, base_channels=8,
                channel_multipliers=(1.0, 1.0), stages=2, initial_depth=1,
                initial_branches=2, max_branches_per_stage=5,
                use_mixup=False, use_cutmix=False, use_amp=False, use_ema=False)
    base.update(kw)
    return AFNNConfig(**base)


def test_forward_and_backward():
    model = AFNN(small_cfg())
    x = torch.randn(3, 3, 16, 16)
    out = model(x)
    assert out.shape == (3, 4)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("kwargs", [
    dict(stages=0), dict(initial_branches=1),
    dict(max_branches_per_stage=1, initial_branches=2),
    dict(normalization="layernorm"), dict(activation="tanh"),
    dict(label_smoothing=1.0), dict(use_mixup=True, mixup_alpha=0.0),
    dict(use_cutmix=True, cutmix_alpha=-1.0), dict(rejuvenation_ema_stride=0),
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((AssertionError, ValueError)):
        AFNNConfig(**kwargs)


def test_config_roundtrip():
    cfg = small_cfg()
    assert AFNNConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_growth_adds_branch_and_keeps_output():
    model = AFNN(small_cfg()).eval()
    x = torch.randn(2, 3, 16, 16)
    before = model(x)
    added = model.grow_one_branch_per_stage(5, 10_000_000)
    assert added == 2
    assert model.count_branches_per_stage() == [3, 3]
    # The new branch starts with a tiny merge weight, so the output barely moves.
    assert torch.allclose(before, model(x), atol=1e-3)


def test_new_branch_matches_device_and_dtype():
    block = FractalBlock(4, depth=1, branches=2, max_branches=4).double()
    assert block.add_branch(4)
    assert {p.dtype for p in block.parameters()} == {torch.float64}


def test_growth_never_exceeds_param_cap():
    cfg = small_cfg(max_total_params=60_000)
    model = AFNN(cfg)
    for _ in range(30):
        model.grow_one_branch_per_stage(cfg.max_branches_per_stage, cfg.max_total_params)
        assert model.count_parameters() <= cfg.max_total_params


def test_pruning_keeps_weight_to_branch_mapping():
    """Weights must stay bound to the right branch after a non-contiguous prune."""
    torch.manual_seed(0)
    channels = 4
    block = FractalBlock(channels, depth=1, branches=6, norm="none",
                         act="relu", max_branches=6).eval()

    def leaf(module):
        return module if isinstance(module, ConvBlock) else module.leaf

    tags = [1.0, 2.5, 3.5, 4.5, 5.0, 6.0]
    with torch.no_grad():
        for branch, tag in zip(block._all_branches(), tags):
            conv = leaf(branch)
            nn.init.zeros_(conv.conv1.weight)
            nn.init.zeros_(conv.conv2.weight)
            conv.conv1.bias.zero_()
            conv.conv2.bias.fill_(tag)          # each branch outputs a constant
        block.merge.logits[:6] = torch.tensor([0.0, 5.0, 1.0, 4.0, 2.0, 3.0])

    x = torch.zeros(1, channels, 8, 8)
    assert block.prune_smallest_branches(keep_top=3) == 3
    survivors = [tags[0], tags[1], tags[3]]
    with torch.no_grad():
        weights = block.merge.weights()
        expected = sum(w.item() * t for w, t in zip(weights, survivors))
        assert abs(block(x).mean().item() - expected) < 1e-4


def test_count_leaf_paths():
    # depth 1 with 2 branches: one ConvBlock + one depth-0 block per stage
    assert AFNN(small_cfg()).count_leaf_paths() == 4
    assert AFNN(small_cfg(initial_depth=2)).count_leaf_paths() == 2 * (1 + 2)
