import json
from types import SimpleNamespace

import torch
import pytest

from snn2.training_audit import TrainingAudit, layer_name, valid_partition_length, summarize_audits


def test_partition_padding():
    assert [valid_partition_length(10, 3, rank) for rank in range(4)] == [3, 3, 3, 1]
    assert valid_partition_length(1, 1, 3) == 0
    assert layer_name('model.layers.3.mlp.up_proj.weight') == 'layer_003'


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_audit_preserves_step_and_records_update(tmp_path, monkeypatch, cpu_offload):
    import deepspeed.utils
    model = torch.nn.Linear(2, 1, bias=False)
    p = model.weight
    p.data.fill_(2.)
    p.ds_numel, p.ds_tensor, p.ds_process_group = 2, p.detach().flatten(), None
    master = p.detach().clone().flatten().float()
    grad = torch.tensor([3., 4.])
    calls = []
    def step():
        calls.append(1)
        master.sub_(torch.tensor([.3, .4]))
        return 'original return'
    optimizer = SimpleNamespace(step=step, loss_scale=1., offload_optimizer=cpu_offload,
        swap_optimizer=False, overflow=False, _global_grad_norm=5.)
    optimizer._get_fp32_grad_state_partition = lambda p, release_swap_buffers: (grad, 0)
    optimizer._get_fp32_opt_state_partition = lambda p, release_swap_buffers: (master, 0)
    engine = SimpleNamespace(module=model, optimizer=optimizer, global_rank=0,
        zero_optimization_stage=lambda: 3)
    def gpu_gradient(p):
        assert not cpu_offload, 'CPU audit must not copy full partitions to GPU'
        return grad
    monkeypatch.setattr(deepspeed.utils, 'safe_get_local_grad', gpu_gradient)
    monkeypatch.setattr(deepspeed.utils, 'safe_get_local_fp32_param', lambda p: master)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda group: 'gloo')
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group: 0)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor, group: None)
    path = tmp_path / 'training_audit.jsonl'
    audit = TrainingAudit(SimpleNamespace(model_wrapped=engine), path)
    audit.on_train_begin(SimpleNamespace(max_grad_norm=1.), SimpleNamespace(global_step=0), None)
    assert optimizer.step() == 'original return'
    assert calls == [1]
    record = json.loads(path.read_text())
    assert record['optimizer_cpu_offload'] == cpu_offload
    assert record['global_preclip_norm'] == 5.
    assert abs(record['layers']['weight']['sampled_update_norm'] - .5) < 1e-6
    plan = {'jobs': [{'id': 'p01', 'kind': 'trial', 'ann_dir': str(tmp_path)}]}
    summarize_audits(plan, tmp_path)
    assert '0.199999' in (tmp_path / 'training_audit_layers.csv').read_text()
