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

def test_phase_tau_ema_matches_spikingllm_update_and_forgets_outlier():
    store = StatisticsStore()
    first = torch.tensor([[[100.0, 2.0]]])
    second = torch.tensor([[[1.0, 4.0]]])
    store.update(0, 1, first)
    store.update(0, 1, second)
    stats = next(iter(store.items.values()))
    expected = torch.tensor([0.99 * 100.0 + 0.01 * 1.0, 0.99 * 2.0 + 0.01 * 4.0])
    assert stats.phase_ema_abs_max.dtype == torch.float32
    torch.testing.assert_close(stats.phase_ema_abs_max, expected.to(torch.float32))
    assert stats.phase_ema_abs_max[0] < stats.value_max[0]
    assert stats.phase_ema_updates.tolist() == [2, 2]
