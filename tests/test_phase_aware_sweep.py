"""Small orchestration regressions: no models, CUDA, or datasets are loaded."""
import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase_sweep', ROOT / 'scripts/phase_aware_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.fixture
def cfg(tmp_path):
    matrix = yaml.safe_load((ROOT / 'configs/experiment_matrix.yaml').read_text())
    c = copy.deepcopy(next(e['config'] for e in matrix['experiments']
        if e['name'] == 'exp1_qwen3_8b_tldr' and e['ann_modes'] == ['vanilla']))
    c['experiment'].update(ann_mode='vanilla', output_root=str(tmp_path))
    c['phase']['T'] = c['mtn']['T'] = 4
    c['mtn']['K'] = 6
    c['evaluation'].update(tldr_test_samples=128, tldr_test_seed=42, prefix_enabled=False)
    return sweep.resolve_config(c)


def test_matrix_validation():
    rows = sweep.matrix_rows('core 1 1 1e-6 0.1 true\nclip 1 1 1e-6 0.1 false')
    assert len(rows) == 2 and rows[1]['clip'] is False
    for text in ('x 0 1 1e-6 0.1 true', 'x 1 nan 1e-6 0.1 true',
                 'x 1 1 1e-6 0.1 true\ny 1 1.0 0.000001 .1 true'):
        with pytest.raises(ValueError):
            sweep.matrix_rows(text)


def test_immutable_config(tmp_path, cfg):
    p = tmp_path / 'config.yaml'
    sweep.save_config(p, copy.deepcopy(cfg))
    sweep.save_config(p, copy.deepcopy(cfg))
    cfg['training']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='Immutable'):
        sweep.save_config(p, cfg)


def test_missing_checkpoint_shard(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    sweep.write(tmp_path / 'model.safetensors.index.json', {'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}})
    (tmp_path / 'one.safetensors').write_bytes(b'test')
    with pytest.raises(FileNotFoundError, match='shards'):
        sweep.validate_checkpoint(tmp_path)
    (tmp_path / 'two.safetensors').write_bytes(b'test')
    sweep.validate_checkpoint(tmp_path)


def reference_job(tmp_path, cfg):
    p = tmp_path / 'input.yaml'
    cfg = sweep.save_config(p, cfg)
    layout = sweep.ArtifactLayout(cfg)
    layout.config_dir.mkdir(parents=True)
    (layout.config_dir / 'resolved_config.yaml').write_text(p.read_text())
    sweep.write(layout.ann_dir / 'training_result.json', {'world_size': 4, 'train_samples': 10000})
    layout.ann_checkpoint_dir.mkdir()
    (layout.ann_checkpoint_dir / 'config.json').write_text('{}')
    (layout.ann_checkpoint_dir / 'model.safetensors').write_bytes(b'fake checkpoint for orchestration test')
    job = dict(id='ref_vanilla', kind='reference', family='reference', config=str(p),
        signature=sweep.digest(cfg), ann_dir=str(layout.ann_dir))
    metric = layout.ann_dir / 'evaluation/tldr/test_samples_128/prefix_enabled_false/metrics.json'
    data = dict(samples=128, ann_mode='vanilla', neuron='ann', tldr_test_seed=42, world_size=4,
        max_new_tokens=32, input_length=512, prefix_enabled=False,
        rouge1=.3, rouge2=.1, rougeL=.2, rougeLsum=.2)
    return job, metric, data


def test_reference_evaluate_then_reuse_and_reject_mismatch(tmp_path, cfg):
    job, metric, data = reference_job(tmp_path, cfg)
    class Fake:
        calls = []
        def command(self, job_id, stage, argv):
            self.calls.append(stage)
            assert stage == 'evaluate'  # Existing checkpoint must never be retrained.
            sweep.write(metric, data)
    executor = Fake()
    assert sweep.run_job(job, executor, 4) == 'completed'
    assert executor.calls == ['evaluate']
    assert sweep.run_job(job, executor, 4) == 'reused'
    assert executor.calls == ['evaluate']
    data['samples'] = 1000
    sweep.write(metric, data)
    with pytest.raises(ValueError, match='protocol mismatch'):
        sweep.run_job(job, executor, 4)


def test_failed_evaluation_retry(tmp_path, cfg):
    job, metric, data = reference_job(tmp_path, cfg)
    class Failing:
        def command(self, job_id, stage, argv):
            raise RuntimeError('fake evaluation failure')
    with pytest.raises(RuntimeError, match='fake evaluation'):
        sweep.run_job(job, Failing(), 4)
    assert not metric.exists()
    assert (Path(job['ann_dir']) / 'training_result.json').exists()


def test_summarize_does_not_rank_failed_metrics(tmp_path, cfg):
    job, metric, data = reference_job(tmp_path, cfg)
    sweep.write(metric, data)
    sweep.write(tmp_path / 'plan.json', {'jobs': [job]})
    sweep.write(tmp_path / 'status/ref_vanilla.json', {'status': 'failed', 'world_size': 4, 'error': 'provenance rejected'})
    sweep.summarize(tmp_path)
    row = sweep.read(tmp_path / 'summary.json')[0]
    assert row['status'] == 'metrics_present_unverified'
    assert 'provenance rejected' in row['error']
    assert '完成/复用 0 项' in (tmp_path / 'summary.md').read_text()


def test_executor_stops_own_process_group(tmp_path):
    executor = sweep.Executor(tmp_path)
    executor.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    executor.stop()
    assert executor.process.poll() is not None


@pytest.mark.parametrize('sample_count', [128, 1000])
def test_paired_bootstrap_requires_matching_samples(tmp_path, sample_count):
    rows = [{'index': i, 'reference': 'a useful summary', 'prediction': 'a useful summary'} for i in range(sample_count)]
    candidate = tmp_path / 'candidate'
    reference = tmp_path / 'reference'
    for directory in (candidate, reference):
        directory.mkdir()
        (directory / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    best = {'id': 'best', 'metrics_path': str(candidate / 'metrics.json'), 'test_samples': sample_count}
    refs = [{'id': 'ref', 'metrics_path': str(reference / 'metrics.json')}]
    sweep.paired_report(tmp_path, best, refs)
    result = sweep.read(tmp_path / 'paired_bootstrap.json')[0]
    assert result['mean_rougeL_delta'] == 0
    assert result['bootstrap_95_interval'] == [0, 0]
    rows[0]['reference'] = 'different reference'
    (reference / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    sweep.paired_report(tmp_path, best, refs)
    assert 'differ' in sweep.read(tmp_path / 'paired_bootstrap.json')[0]['error']


def test_vanilla_ignores_generic_prefix_switch(tmp_path, cfg):
    cfg['evaluation']['prefix_enabled'] = True
    job, metric, data = reference_job(tmp_path, cfg)
    sweep.write(metric, data)
    assert sweep.validate_metrics(job, sweep.load_config(job['config']), 4) == metric


def test_paired_bootstrap_matches_nonstemming_evaluation(tmp_path):
    for name, prediction in [('candidate', 'running'), ('reference', 'run')]:
        directory = tmp_path / name
        directory.mkdir()
        rows = [{'index': i, 'reference': 'run', 'prediction': prediction} for i in range(128)]
        (directory / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    sweep.paired_report(tmp_path, {'id': 'best', 'metrics_path': str(tmp_path / 'candidate/metrics.json')},
        [{'id': 'ref', 'metrics_path': str(tmp_path / 'reference/metrics.json')}])
    assert sweep.read(tmp_path / 'paired_bootstrap.json')[0]['mean_rougeL_delta'] == -1.0


def test_formal_metrics_do_not_reuse_quick_results(tmp_path, cfg):
    cfg['evaluation']['tldr_test_samples'] = 1000
    job, quick_metric, data = reference_job(tmp_path, cfg)
    sweep.write(quick_metric, data)
    loaded = sweep.load_config(job['config'])
    assert sweep.validate_metrics(job, loaded, 4) is None
    formal_metric = Path(str(quick_metric).replace('test_samples_128', 'test_samples_1000'))
    data['samples'] = 1000
    sweep.write(formal_metric, data)
    assert sweep.validate_metrics(job, loaded, 4) == formal_metric


def test_audit_flag_does_not_change_training_compatibility(cfg):
    changed = copy.deepcopy(cfg)
    changed['training']['audit_gradients'] = True
    assert sweep.training_digest(changed) == sweep.training_digest(cfg)
    changed['training']['learning_rate'] = 0.123
    assert sweep.training_digest(changed) != sweep.training_digest(cfg)
