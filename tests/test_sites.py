import torch

import snn2.model_integration as model_integration
from snn2.model_integration import _make_mlp_forward, _linear_score, record_down_proj_saliency
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


def test_mlp_applies_site_nine_before_r4_and_site_ten(monkeypatch):
    events = []

    class Controller(_Controller):
        def apply(self, layer, site, value):
            events.append(f"site_{site}")
            return value

    monkeypatch.setattr(
        model_integration,
        "random_hadamard",
        lambda value, spec: (events.append("r4") or value),
    )
    _make_mlp_forward(Controller(), 0, object())(_MLP(), torch.tensor([[2.0]]))
    assert events == ["site_8", "site_9", "r4", "site_10"]


def test_down_proj_saliency_is_recorded_at_site_ten():
    controller = _Controller(mode="collect")
    inputs = (torch.tensor([[2.0, 3.0]]),)
    output = torch.tensor([[5.0, 7.0]])
    weight = torch.eye(2)
    record_down_proj_saliency(controller, 0, inputs, output, weight)
    torch.testing.assert_close(controller.saliency[10], _linear_score(inputs[0], output, weight))
    assert 9 not in controller.saliency


def test_deploy_mode_does_not_bypass_site_nine():
    class DeployController(_Controller):
        def apply(self, layer, site, value):
            self.applied.append(site)
            return value + 10 if site == 9 else value

    controller = DeployController(mode="deploy_phase")
    output = _make_mlp_forward(controller, 0, None)(_MLP(), torch.tensor([[2.0]]))
    assert controller.applied == [8, 9, 10]
    torch.testing.assert_close(output, torch.tensor([[400.0]]))
