import pytest
import torch
from torch import nn

from snn2.energy_profiler import (
    EnergyInstrumentation, _pair, count_temporal_matmul, energy_j_from_raw_counts,
    fixed_length_input, gif_integer_multiplicity, linear_counts,
    select_validation_positions, temporal_product_counts,
)
from snn2.neurons import IdentityGIF, SoftmaxIdentityGIF, StaticGIF


def test_units_and_basic_counts():
    assert energy_j_from_raw_counts(10**9, 0) == pytest.approx(.0046)
    assert energy_j_from_raw_counts(0, 10**9) == pytest.approx(.0009)
    assert energy_j_from_raw_counts(100 * 10**9, 500 * 10**9) == pytest.approx(.91)
    x = torch.ones(1, 2, 3)
    assert linear_counts(x, 4).mac_raw == 24
    events = torch.tensor([[[1, 0, 2], [0, 2, 0]]])
    assert linear_counts(x, 4, events).synaptic_ac_raw == 20
    assert ((torch.tensor([1., -3., 0.]) != 0).int().sum().item()) == 2


def test_temporal_three_terms_and_dynamic():
    a = torch.ones(2, 1, 1, 1)
    b = torch.ones_like(a)
    m = torch.ones_like(a, dtype=torch.int32)
    assert count_temporal_matmul(a, b, m, m).synaptic_ac_raw == 8
    assert count_temporal_matmul(a, b, None, None).mac_raw == 6
    assert count_temporal_matmul(a, b, None, m).mac_raw == 7
    assert count_temporal_matmul(a, b, m, None).mac_raw == 7
    assert temporal_product_counts(a, b, None, None).mac_raw == 4
    assert temporal_product_counts(a, b, None, m).mac_raw == 6
    assert temporal_product_counts(a, b, m, None).mac_raw == 6
    assert temporal_product_counts(a, b, m, m).synaptic_ac_raw == 8


def test_selection_and_fixed_input():
    assert select_validation_positions(10, 4, 3) == select_validation_positions(10, 4, 3)
    with pytest.raises(ValueError):
        select_validation_positions(2, 4, 3)
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 9
        def encode(self, text, add_special_tokens):
            return ([1] if add_special_tokens else []) + [2] * len(text)
    cfg = {"experiment": {"task": "tldr"}, "data": {"max_seq_length": 1000, "truncation_side": "left"}}
    ids, mask = fixed_length_input({"prompt": "a", "completion": "b"}, Tokenizer(), cfg)
    assert ids.shape == mask.shape == (1, 512)
    assert mask.sum().item() == 4
    ids, mask = fixed_length_input({"prompt": "a" * 700, "completion": "b"}, Tokenizer(), cfg)
    assert mask.sum().item() == 512
    assert ids[0, 0].item() == 2


def test_hook_restores_numeric_forward_on_error_and_exit():
    class Controller:
        mode = "identity"
        def apply(self, layer, site, x, *, gif_role=None):
            return x
        def apply_final_norm_neuron(self, x):
            return x
    model = nn.Sequential(nn.Linear(3, 4))
    controller = Controller()
    x = torch.ones(2, 3)
    before = model(x)
    original = controller.apply
    with EnergyInstrumentation(model, controller, "ann") as profiler:
        assert torch.equal(model(x), before)
        assert profiler.counts.mac_raw == 24
    assert torch.equal(model(x), before)
    assert controller.apply == original


def test_gif_code_expansion_roles_and_identity():
    from snn2.temporal_ops import GIF_INTEGER_DECOMPOSITION, SITE_STATE_FORMAT_VERSION, TEMPORAL_IMPLEMENTATION_VERSION
    state = {
        "state_kind": "gif", "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "parameter_layout": "last_dim_grouped", "configured_group_size": 1,
        "group_size": 1, "num_heads": None, "channels_per_head": 5,
        "groups_per_head": 5, "gif_policy": "ordinary_salient_static_qmax30",
        "base_bits": 4, "add_bits": 1, "low_qmin": 0, "low_qmax": 15,
        "high_qmin": 0, "high_qmax": 30, "temporal_steps": 2,
        "per_step_qmin": 0, "per_step_qmax": 15,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "low_scale": torch.ones(5), "low_zero": torch.zeros(5),
        "high_scale": torch.ones(5), "high_zero": torch.zeros(5),
        "mask_policy": "multi_role", "mask_roles": ["q", "k", "v"],
        "mask_low_by_role": {
            "q": torch.ones(5, dtype=torch.bool),
            "k": torch.zeros(5, dtype=torch.bool),
            "v": torch.tensor([True, False, True, False, True]),
        },
    }
    module = StaticGIF(state)
    incoming = torch.zeros(2, 1, 1, 5)
    incoming[0, 0, 0] = torch.tensor([0., 1., 15., 16., 30.])
    k = gif_integer_multiplicity(module, incoming, "k")
    assert k[:, 0, 0].tolist() == [[0, 1, 15, 15, 15], [0, 0, 0, 1, 15]]
    q = gif_integer_multiplicity(module, incoming, "q")
    assert q[1].count_nonzero().item() == 0
    assert q[0, 0, 0].tolist() == [0, 1, 15, 15, 15]
    assert gif_integer_multiplicity(IdentityGIF({
        "state_kind": "gif", "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "gif_policy": "identity", "quantization_applied": False, "temporal_steps": 2,
    }), incoming) is None


def test_energy_paths_isolate_deployments():
    from snn2.artifacts import ArtifactLayout, prefix_enabled_dirname
    from snn2.config import load_config
    cfg = load_config("configs/generated/exp2_llama3_8b_tulu3__phase_aware.yaml")
    layout = ArtifactLayout(cfg)
    ann128 = layout.energy_dir("ann")
    phase4 = layout.energy_dir("phase")
    cfg["phase"]["T"] = 8
    phase8 = layout.energy_dir("phase")
    cfg["phase"]["base"] = 3.0
    phase_base3 = layout.energy_dir("phase")
    cfg["mtn"]["T"], cfg["mtn"]["K"] = 4, 6
    mtn4 = layout.energy_dir("mtn")
    cfg["mtn"]["T"], cfg["mtn"]["K"] = 8, 12
    mtn8 = layout.energy_dir("mtn")
    cfg["conversion"]["use_post_finetuning_artifacts"] = not cfg["conversion"]["use_post_finetuning_artifacts"]
    selector = layout.energy_dir("mtn")
    cfg["evaluation"]["prefix_enabled"] = not cfg["evaluation"]["prefix_enabled"]
    prefix_off_mtn = layout.energy_dir("mtn")
    ann_prefix_off = layout.energy_dir("ann")
    cfg["calibration"]["num_samples"] = 256
    mtn256 = layout.energy_dir("mtn")
    ann256 = layout.energy_dir("ann")
    paths = {ann128, phase4, phase8, phase_base3, mtn4, mtn8, selector,
             prefix_off_mtn, ann_prefix_off, mtn256, ann256}
    assert len(paths) == 11
    assert ann128.parts[-2:] == (prefix_enabled_dirname(True), "profile_num_samples_128")
    assert ann256.parts[-2:] == (prefix_enabled_dirname(False), "profile_num_samples_256")
    vanilla = load_config("configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml")
    vanilla["evaluation"]["prefix_enabled"] = True
    assert ArtifactLayout(vanilla).energy_dir("ann").parts[-2] == prefix_enabled_dirname(False)


def test_gif_all_low_second_frame_and_softmax_identity():
    from snn2.neurons import AllLowStaticGIF
    from snn2.temporal_ops import SITE_STATE_FORMAT_VERSION, TEMPORAL_IMPLEMENTATION_VERSION, SOFTMAX_SITE5_GIF_POLICY
    header = {"state_kind": "gif", "format_version": SITE_STATE_FORMAT_VERSION,
              "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION}
    all_low = {
        **header, "parameter_layout": "last_dim_grouped", "configured_group_size": 1,
        "group_size": 1, "num_heads": None, "channels_per_head": 1, "groups_per_head": 1,
        "gif_policy": "all_low_static_qmax15", "base_bits": 4, "add_bits": 1,
        "low_qmin": 0, "low_qmax": 15, "temporal_steps": 2,
        "per_step_qmin": 0, "per_step_qmax": 15,
        "quantization_path": "low_only", "quantization_applied": True,
        "saliency_enabled": False, "temporal_policy": "low_at_t0_zero_at_t1",
        "low_scale": torch.ones(1), "low_zero": torch.zeros(1),
    }
    incoming = torch.tensor([[[[7.]]], [[[0.]]]])
    mult = gif_integer_multiplicity(AllLowStaticGIF(all_low), incoming)
    assert mult[:, 0, 0, 0].tolist() == [7, 0]
    softmax = SoftmaxIdentityGIF({
        **header, "parameter_layout": "softmax_identity", "configured_group_size": 1,
        "group_size": -1, "group_size_source": "site5_identity_override",
        "num_heads": 1, "gif_policy": SOFTMAX_SITE5_GIF_POLICY,
        "reference_n_bits": 16, "reference_metric": "fix0to1",
        "quantization_applied": False, "temporal_steps": 2,
        "temporal_policy": "identity",
    })
    assert gif_integer_multiplicity(softmax, torch.ones(2, 1, 1, 1, 1)) is None


def test_temporal_controller_instrumentation_preserves_output():
    class Controller:
        mode = "deploy_phase"
        temporal_steps = 2
        def apply(self, layer, site, x, *, gif_role=None):
            return x
        def apply_final_norm_neuron(self, x):
            return x
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 3, bias=False)
    class Toy(nn.Module):
        def __init__(self, controller):
            super().__init__()
            self.controller = controller
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block()])
        def forward(self, x):
            return self.model.layers[0].q_proj(self.controller.apply(0, 1, x))
    controller = Controller()
    model = Toy(controller)
    x = torch.tensor([[[1., 0.]], [[-1., 2.]]])
    before = model(x)
    with EnergyInstrumentation(model, controller, "phase") as profiler:
        during = model(x)
        assert profiler.counts.synaptic_ac_raw == 9
        assert profiler.counts.neuron_ac_raw == 2 + 3
    after = model(x)
    torch.testing.assert_close(before, during, rtol=0, atol=0)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_temporal_attention_wrapper_counts_prefix_length_and_restores_binding():
    import snn2.temporal_model as temporal_model
    class Controller:
        mode = "deploy_gif"
        temporal_steps = 2
        def apply(self, layer, site, x, *, gif_role=None):
            return x
        def apply_final_norm_neuron(self, x):
            return x
    original = temporal_model.temporal_seq_matmul
    with EnergyInstrumentation(nn.Module(), Controller(), "gif") as profiler:
        # Key/value length 3 represents a two-token sequence and one cached Prefix token.
        profiler.events[(0, 2, None)] = torch.ones(2, 1, 1, 2, 2, dtype=torch.int32)
        profiler.events[(0, 3, None)] = torch.ones(2, 1, 3, 2, dtype=torch.int32)
        profiler.events[(0, 4, None)] = torch.ones(2, 1, 3, 2, dtype=torch.int32)
        q = torch.ones(2, 1, 1, 2, 2)
        k = torch.ones(2, 1, 1, 2, 3)
        temporal_model.temporal_seq_matmul(q, k)
        assert profiler.counts.synaptic_ac_raw == 96
        weights = torch.ones(2, 1, 1, 2, 3)
        value = torch.ones(2, 1, 1, 3, 2)
        temporal_model.temporal_seq_matmul(weights, value)
        assert profiler.counts.mac_raw == 84
    assert temporal_model.temporal_seq_matmul is original


def test_gif_pv_dynamic_attention_times_event_value_uses_mixed_mac_rule():
    # Site 5 SoftmaxIdentityGIF is dense; Site 4 emits GIF unit events.
    attention = torch.ones(2, 1, 1, 1, 2)
    value = torch.ones(2, 1, 1, 2, 1)
    events = torch.tensor([[[[[1], [0]]]], [[[[2], [1]]]]], dtype=torch.int32)
    result = count_temporal_matmul(attention, value, None, events)
    # cumsum(P)*V: 4; P*cumsum(V): 5; P*V: 4.
    assert result.mac_raw == 13
    assert result.synaptic_ac_raw == 0


def test_pair_reduction_matches_tiny_matmul():
    a = torch.tensor([[[[1, 2], [3, 4]]], [[[2, 0], [1, 5]]]], dtype=torch.int32)
    b = torch.tensor([[[[2, 1, 0], [0, 3, 1]]], [[[1, 1, 1], [2, 0, 3]]]], dtype=torch.int32)
    assert _pair(a, b) == int(torch.matmul(a.float(), b.float()).sum().item())


def test_pair_counter_does_not_call_dense_matmul(monkeypatch):
    a = torch.ones(2, 1, 2, 3, dtype=torch.int32)
    b = torch.ones(2, 1, 3, 4, dtype=torch.int32)
    monkeypatch.setattr(torch, "matmul", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dense matmul called")))
    assert _pair(a, b) == 48


def test_gif_nonzero_zero_point_uses_unsigned_code_event_policy():
    from snn2.temporal_ops import GIF_INTEGER_DECOMPOSITION, SITE_STATE_FORMAT_VERSION, TEMPORAL_IMPLEMENTATION_VERSION
    state = {
        "state_kind": "gif", "format_version": SITE_STATE_FORMAT_VERSION,
        "temporal_implementation_version": TEMPORAL_IMPLEMENTATION_VERSION,
        "parameter_layout": "last_dim_grouped", "configured_group_size": 1,
        "group_size": 1, "num_heads": None, "channels_per_head": 2,
        "groups_per_head": 2, "gif_policy": "ordinary_salient_static_qmax30",
        "base_bits": 4, "add_bits": 1, "low_qmin": 0, "low_qmax": 15,
        "high_qmin": 0, "high_qmax": 30, "temporal_steps": 2,
        "per_step_qmin": 0, "per_step_qmax": 15,
        "integer_decomposition": GIF_INTEGER_DECOMPOSITION,
        "low_scale": torch.ones(2), "low_zero": torch.full((2,), 3.),
        "high_scale": torch.ones(2), "high_zero": torch.full((2,), 5.),
        "mask_low": torch.tensor([True, False]),
    }
    module = StaticGIF(state)
    incoming = torch.tensor([[[[0., 16.]]], [[[0., 0.]]]])
    assert gif_integer_multiplicity(module, incoming)[:, 0, 0].tolist() == [[3, 15], [0, 6]]
    # Numerical GIF output still subtracts the fixed zero points.
    torch.testing.assert_close(module.temporal(incoming).sum(0), incoming.sum(0))
