"""Opt-in ZeRO-3 preclip gradients and sampled FP32 master-weight updates."""
import json
import math
import re
from pathlib import Path

import torch
from transformers import TrainerCallback


def layer_name(name):
    match = re.search(r'layers\.(\d+)\.', name)
    return f'layer_{int(match[1]):03d}' if match else name.rsplit('.', 1)[0]


def valid_partition_length(total, partition, rank):
    return max(0, min(partition, total - rank * partition))


class TrainingAudit(TrainerCallback):
    def __init__(self, trainer, path):
        self.trainer, self.path = trainer, Path(path)

    def on_train_begin(self, args, state, control, **kwargs):
        from deepspeed.utils import safe_get_local_grad, safe_get_local_fp32_param
        engine = self.trainer.model_wrapped
        if not hasattr(engine, 'zero_optimization_stage') or engine.zero_optimization_stage() != 3:
            raise RuntimeError('Training audit requires DeepSpeed ZeRO-3')
        optimizer = engine.optimizer
        if optimizer.swap_optimizer:
            raise RuntimeError('Training audit does not support NVMe optimizer swapping')
        cpu_offload = bool(optimizer.offload_optimizer)
        if cpu_offload:
            # Public safe_get_local_* moves whole partitions to CUDA in this DS
            # version. Read synchronized native CPU views to avoid those copies.
            for method in ('_get_fp32_grad_state_partition', '_get_fp32_opt_state_partition'):
                if not callable(getattr(optimizer, method, None)):
                    raise RuntimeError(f'CPU training audit requires DeepSpeed {method}')

        def local_gradient(p):
            if cpu_offload:
                return optimizer._get_fp32_grad_state_partition(p, release_swap_buffers=True)[0]
            return safe_get_local_grad(p)

        def local_weight(p):
            if cpu_offload:
                return optimizer._get_fp32_opt_state_partition(p, release_swap_buffers=True)[0]
            return safe_get_local_fp32_param(p)
        parameters = [(name, p) for name, p in engine.module.named_parameters() if p.requires_grad]
        groups = sorted({layer_name(name) for name, _ in parameters})
        indices = {name: i for i, name in enumerate(groups)}
        original = optimizer.step
        if engine.global_rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text('')

        def audited_step(*a, **kw):
            device = torch.device('cpu') if cpu_offload else parameters[0][1].device
            totals = torch.zeros((len(groups), 4), dtype=torch.float64, device=device)
            snapshots = []
            scale = float(optimizer.loss_scale)
            for name, p in parameters:
                rank = torch.distributed.get_rank(p.ds_process_group)
                n = valid_partition_length(p.ds_numel, p.ds_tensor.numel(), rank)
                g = local_gradient(p).detach().flatten()[:n]
                weight = local_weight(p).detach().flatten()[:n]
                i = indices[layer_name(name)]
                totals[i, 0] += torch.linalg.vector_norm(g, dtype=torch.float32).double().square() / scale**2
                stride = max(1, math.ceil(n / 4096))
                before = weight[::stride].clone()
                totals[i, 1] += before.double().square().sum()
                totals[i, 3] += before.numel()
                snapshots.append((p, i, n, stride, before))
            result = original(*a, **kw)
            for p, i, n, stride, before in snapshots:
                after = local_weight(p).detach().flatten()[:n:stride]
                totals[i, 2] += (after.double() - before.double()).square().sum()
            group = parameters[0][1].ds_process_group
            if torch.distributed.get_backend(group) == 'nccl':
                totals = totals.to(torch.device('cuda', torch.cuda.current_device()))
            torch.distributed.all_reduce(totals, group=group)
            if engine.global_rank == 0:
                values = totals.cpu().tolist()
                norm = math.sqrt(sum(v[0] for v in values))
                ds_norm = float(optimizer._global_grad_norm)
                threshold = float(args.max_grad_norm)
                record = {'step': state.global_step + 1, 'optimizer_cpu_offload': cpu_offload, 'global_preclip_norm': norm,
                    'deepspeed_global_preclip_norm': ds_norm,
                    'clip_coefficient_estimate': min(1., threshold / (norm + 1e-6)) if threshold > 0 else 1.,
                    'overflow': bool(optimizer.overflow),
                    'update_measurement': 'deterministic samples of FP32 master weights; includes AdamW and weight decay',
                    'layers': {name: {'preclip_gradient_norm': math.sqrt(v[0]),
                        'sampled_weight_norm': math.sqrt(v[1]), 'sampled_update_norm': math.sqrt(v[2]),
                        'sampled_relative_update': math.sqrt(v[2] / max(v[1], 1e-30)),
                        'sampled_elements': int(v[3])} for name, v in zip(groups, values)}}
                with self.path.open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
            return result
        optimizer.step = audited_step


def summarize_audits(plan, output):
    """Export every recorded optimizer step; retain missing historical audits explicitly."""
    import csv
    rows, status = [], []
    for job in plan['jobs']:
        if job['kind'] != 'trial':
            continue
        path = Path(job['ann_dir']) / 'training_audit.jsonl'
        records = []
        if path.exists():
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        status.append({'id': job['id'], 'records': len(records), 'path': str(path),
                       'status': 'present' if records else 'unavailable'})
        for record in records:
            for layer, values in record['layers'].items():
                rows.append({'id': job['id'], 'step': record['step'], 'layer': layer,
                    'global_preclip_norm': record['global_preclip_norm'],
                    'deepspeed_global_preclip_norm': record['deepspeed_global_preclip_norm'],
                    'clip_coefficient_estimate': record['clip_coefficient_estimate'], **values})
    output = Path(output)
    (output / 'training_audit_status.json').write_text(json.dumps(status, indent=2))
    if rows:
        with (output / 'training_audit_layers.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
