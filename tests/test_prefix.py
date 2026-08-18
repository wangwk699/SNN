import torch

from snn2.prefix import PrefixOutlierCollector


class _Tokenizer:
    bos_token_id = 128000
    eos_token_id = 128001
    name_or_path = "dummy"

    def decode(self, ids):
        return " ".join(map(str, ids))


def _collect(skip_initial_position: bool, activation: torch.Tensor):
    collector = PrefixOutlierCollector(
        64.0,
        skip_initial_position=skip_initial_position,
    )
    collector.set_batch(
        torch.tensor([[11, 12, 13]], dtype=torch.long),
        torch.ones((1, 3), dtype=torch.long),
    )
    collector.hook("model.layers.0.mlp.down_proj")(None, (activation,))
    return collector


def test_qwen_counts_position_zero_and_does_not_append_start_token():
    activation = torch.tensor([[[100.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    collector = _collect(False, activation)
    state = collector.result(_Tokenizer(), append_start_token=False)
    assert state["prefix_token_ids"] == [11]
    assert state["skip_initial_position_in_frequency"] is False
    assert state["appended_start_token_id"] is None


def test_llama_skips_position_zero_and_appends_bos():
    activation = torch.tensor([[[100.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    collector = _collect(True, activation)
    state = collector.result(_Tokenizer(), append_start_token=True)
    assert state["prefix_token_ids"] == [128000]
    assert state["skip_initial_position_in_frequency"] is True
    assert state["appended_start_token_id"] == 128000


def test_qwen_can_have_empty_prefix():
    activation = torch.ones((1, 3, 2))
    collector = _collect(False, activation)
    state = collector.result(_Tokenizer(), append_start_token=False)
    assert state["prefix_token_ids"] == []
