from types import SimpleNamespace

import pytest

from snn2.lm_eval_distributed import (
    DistributedPreinitializedHFLM,
    distributed_max_seconds,
    gather_sum_execution_counter,
    indices_for_rank,
    sum_execution_counters,
)


def test_indices_for_rank_partition_is_disjoint_and_complete():
    indices = list(range(17))
    shards = [indices_for_rank(indices, rank=rank, world_size=4) for rank in range(4)]
    flattened = [item for shard in shards for item in shard]
    assert sorted(flattened) == indices
    assert len(flattened) == len(set(flattened))
    assert indices_for_rank(indices, rank=0, world_size=1) == indices


@pytest.mark.parametrize("rank, world_size", [(-1, 4), (4, 4), (0, 0), (True, 4)])
def test_indices_for_rank_rejects_invalid_partition(rank, world_size):
    with pytest.raises(ValueError):
        indices_for_rank([1, 2], rank=rank, world_size=world_size)


def test_execution_counter_merge_and_single_process_helpers():
    counters = [{"model_forward_calls": 10, "temporal_sample_step_forwards": 20},
                {"model_forward_calls": 7, "temporal_sample_step_forwards": 15, "extra": 2}]
    assert sum_execution_counters(counters) == {"model_forward_calls": 17, "temporal_sample_step_forwards": 35, "extra": 2}
    assert gather_sum_execution_counter(counters[0], world_size=1) == counters[0]
    assert distributed_max_seconds(1.25, device=__import__("torch").device("cpu"), world_size=1) == 1.25


def test_preinitialized_hflm_binds_accelerate_global_rank(monkeypatch):
    import snn2.lm_eval_distributed as module

    monkeypatch.setattr(module.HFLM, "__init__", lambda self, **kwargs: None)
    accelerator = SimpleNamespace(process_index=2, num_processes=4, device=__import__("torch").device("cuda", 2))
    model = DistributedPreinitializedHFLM(accelerator=accelerator, pretrained=object())
    assert model.rank == 2
    assert model.world_size == 4
    assert model.device == accelerator.device
