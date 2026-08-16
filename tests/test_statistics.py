import torch

from snn2.stats import StatisticsStore


def test_softmax_site_uses_fixed_max_position_state():
    store = StatisticsStore(max_channels_by_site={5: 16})
    store.update(0, 5, torch.rand(1, 2, 4, 4))
    store.update_saliency(0, 5, torch.rand(1, 2, 4, 4))
    store.update(0, 5, torch.rand(1, 2, 7, 7))
    store.update_saliency(0, 5, torch.rand(1, 2, 7, 7))
    stats = next(iter(store.items.values()))
    assert stats.channels == 16
    assert stats.variable_channels
    assert torch.all(stats.saliency_row_count[:7] > 0)
    assert torch.all(stats.saliency_row_count[7:] == 0)
