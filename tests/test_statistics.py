import torch
import pytest

from snn2.calibration import build_phase_state
from snn2.phase_statistics import phase_statistical_view
from snn2.stats import StatisticsStore


def test_softmax_site_uses_fixed_max_position_state():
    store = StatisticsStore(max_channels_by_site={5: 16})
    first = torch.rand(1, 2, 4, 4)
    store.update(0, 5, first, phase_activation=phase_statistical_view(5, first))
    store.update_saliency(0, 5, torch.rand(1, 2, 4, 4))
    second = torch.rand(1, 2, 7, 7)
    store.update(0, 5, second, phase_activation=phase_statistical_view(5, second))
    store.update_saliency(0, 5, torch.rand(1, 2, 7, 7))
    stats = next(iter(store.items.values()))
    assert stats.channels == 16
    assert stats.variable_channels
    assert stats.phase_channels == 2
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


@pytest.mark.parametrize(
    ("site", "shape", "expected_shape"),
    [
        (2, (2, 4, 3, 5), (2, 3, 20)),
        (3, (2, 2, 7, 5), (2, 7, 10)),
        (4, (2, 2, 7, 5), (2, 7, 10)),
        (5, (2, 4, 3, 7), (2, 21, 4)),
        (6, (2, 4, 3, 5), (2, 3, 20)),
    ],
)
def test_phase_statistical_view_matches_spikingllm_layout(site, shape, expected_shape):
    x = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
    view = phase_statistical_view(site, x)
    assert tuple(view.shape) == expected_shape
    torch.testing.assert_close(view.flatten().sort().values, x.flatten().sort().values)


def test_phase_channels_are_independent_from_generic_softmax_channels():
    store = StatisticsStore(max_channels_by_site={5: 7})
    activation = torch.rand(1, 4, 3, 7)
    store.update(
        0, 5, activation, phase_activation=phase_statistical_view(5, activation)
    )
    stats = next(iter(store.items.values()))
    assert stats.channels == 7
    assert stats.phase_channels == 4
    assert stats.value_min.numel() == 7
    assert stats.phase_ema_abs_max.numel() == 4


def test_phase_tau_is_global_max_after_per_channel_ema():
    store = StatisticsStore()
    generic = torch.zeros(1, 1)
    store.update(0, 1, generic, phase_activation=torch.tensor([[[100.0, 0.0]]]))
    store.update(0, 1, generic, phase_activation=torch.tensor([[[0.0, 100.0]]]))
    statistics = next(iter(store.items.values())).state_dict()
    state = build_phase_state(
        statistics,
        {
            "calibration": {"group_size": 1},
            "phase": {"T": 4, "base": 2.0, "surrogate_slope": 1.0},
        },
    )
    torch.testing.assert_close(state["tau"], torch.tensor([99.0]))
    assert state["tau"].numel() == 1
    assert state["group_size"] == -1
    assert state["tau_reduction_policy"] == "per_channel_ema_then_global_max"
