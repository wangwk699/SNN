import torch

from snn2.model_integration import _make_mlp_forward
from snn2.sites import SITE_COORDINATES, SITE_COUNT, SITE_IDS, SITE_NAMES, site_key


class _Controller:
    def __init__(self, mode="identity"):
        self.mode = mode
        self.applied = []
        self.saliency = {}

    def apply(self, layer, site, value):
        self.applied.append(site)
        return value

    def record_saliency(self, layer, site, value):
        self.saliency[site] = value


class _MLP:
    def act_fn(self, value):
        return value + 1

    def gate_proj(self, value):
        return value * 2

    def up_proj(self, value):
        return value * 3

    def down_proj(self, value):
        return value * 5


def test_site_topology_is_ten_sites():
    assert SITE_COUNT == 10
    assert SITE_IDS == tuple(range(1, 11))
    assert (SITE_NAMES[8], SITE_COORDINATES[8]) == ("post_spiking_silu", "I")
    assert (SITE_NAMES[9], SITE_COORDINATES[9]) == ("post_mlp_up_proj", "I")
    assert (SITE_NAMES[10], SITE_COORDINATES[10]) == ("post_mlp_product_r4", "R4")
    assert site_key(0, 9).endswith("site_09_post_mlp_up_proj")
    assert site_key(0, 10).endswith("site_10_post_mlp_product_r4")


def test_mlp_places_new_up_site_before_product_and_preserves_identity_parity():
    x = torch.tensor([[2.0]])
    controller = _Controller()
    output = _make_mlp_forward(controller, 0, None)(_MLP(), x)
    reference = ((x * 2 + 1) * (x * 3)) * 5
    torch.testing.assert_close(output, reference)
    assert controller.applied == [8, 9, 10]


def test_mlp_collects_symmetric_product_saliency_for_gate_and_up_sites():
    x = torch.tensor([[2.0]])
    controller = _Controller(mode="collect")
    _make_mlp_forward(controller, 0, None)(_MLP(), x)
    expected = torch.tensor([[5.0]]) ** 2 * torch.tensor([[6.0]]) ** 2
    torch.testing.assert_close(controller.saliency[8], expected)
    torch.testing.assert_close(controller.saliency[9], expected)
