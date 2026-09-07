"""Plan/run a resumable Phase ANN sweep; summarize without loading GPU models."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from snn2.artifacts import ArtifactLayout
from snn2.config import load_config, resolve_config, validate_config, final_ann_evaluation_prefix_enabled


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def semantic_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop('_meta', None)
    # Logging and requested evaluation sample counts do not change training.
    cfg['training'].pop('logging_steps', None)
    cfg['evaluation'].pop('tldr_test_samples', None)
    return cfg


def digest(cfg):
    return hashlib.sha256(json.dumps(semantic_config(cfg), sort_keys=True).encode()).hexdigest()


def matrix_rows(text):
    rows, seen = [], set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        family, g, slope, lr, warmup, clip = line.split()
        g, slope, lr, warmup = int(g), float(slope), float(lr), float(warmup)
        if g != -1 and g <= 0:
            raise ValueError('Invalid group size')
        if not all(math.isfinite(x) for x in (slope, lr, warmup)) or min(slope, lr) <= 0 or not 0 <= warmup < 1:
            raise ValueError('Invalid slope/lr/warmup')
        if clip not in ('true', 'false'):
            raise ValueError('Clip must be true/false')
        values = (g, slope, lr, warmup, clip == 'true')
        if values in seen:
            raise ValueError(f'Duplicate experiment: {line}')
        seen.add(values)
        rows.append(dict(zip(('group', 'slope', 'lr', 'warmup', 'clip'), values), family=family))
    if not rows:
        raise ValueError('Empty experiment matrix')
    return rows


def assert_fixed(cfg):
    if (cfg['phase']['T'], cfg['mtn']['T'], cfg['mtn']['K']) != (4, 4, 6):
        raise ValueError('Required phase.T=4, mtn.T=4, mtn.K=6')
    if cfg['experiment']['model_name'] != 'Qwen/Qwen3-8B-Base' or cfg['experiment']['task'] != 'tldr':
        raise ValueError('This sweep is only for Qwen3-8B-Base / TL;DR')


def save_config(path, cfg):
    cfg.pop('_meta', None)
    cfg = resolve_config(cfg)
    validate_config(cfg)
    assert_fixed(cfg)
    if path.exists():
        previous = load_config(path)
        previous.pop('_meta', None)
        if previous != cfg:
            raise ValueError(f'Immutable sweep config changed: {path}; use a new SWEEP_DIR')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return load_config(path)


def make_plan(args, output):
    base = load_config(args.source_config)
    base.pop('_meta', None)
    base['experiment']['ann_mode'] = 'phase_aware'
    base['experiment']['seed'] = 42
    base['phase'].update(T=4)
    base['mtn'].update(T=4, K=6)
    base['calibration'].update(num_samples=128, batch_size=1, seed=42)
    base['ann_training']['prefix_enabled'] = True
    base['prefix']['enabled'] = True
    base['conversion']['use_post_finetuning_artifacts'] = False
    base['data'].update(max_seq_length=2048, evaluation_split='test')
    base['training'].update(tldr_train_samples=args.train_samples, tldr_train_seed=42,
        num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=16,
        logging_steps=1, warmup_ratio=0.1, save_strategy='no', eval_strategy='no',
        load_best_model_at_end=False, resume_from_checkpoint=None)
    base['evaluation'].update(tldr_test_samples=args.test_samples, tldr_test_seed=42,
        batch_size=8, tldr_input_length=512, max_new_tokens=32, prefix_enabled=True)
    if getattr(args, 'audit_gradients', False):
        base['training']['audit_gradients'] = True
        base['training']['max_grad_norm'] = 1.0
    jobs = []
    for index, row in enumerate(matrix_rows(args.matrix_text), 1):
        if getattr(args, 'require_clip_true', False) and not row['clip']:
            raise ValueError('This sweep requires common_clip_enabled=true for every new trial')
        cfg = copy.deepcopy(base)
        cfg['calibration']['group_size'] = row['group']
        cfg['phase']['surrogate_slope'] = row['slope']
        cfg['training'].update(learning_rate=row['lr'], warmup_ratio=row['warmup'])
        cfg['replacement']['common_clip_enabled'] = row['clip']
        job_id = f"p{index:02d}_g{row['group']}_s{row['slope']}_lr{row['lr']}_w{row['warmup']}_clip{int(row['clip'])}"
        path = output / 'configs' / f'{job_id}.yaml'
        cfg = save_config(path, cfg)
        jobs.append(dict(id=job_id, kind='trial', config=str(path), signature=digest(cfg),
                         ann_dir=str(ArtifactLayout(cfg).ann_dir), **row))
    # Exact user-designated pretrained reference checkpoints; evaluation only.
    model_root = ArtifactLayout(base).model_root
    refs = {
        'vanilla': 'vanilla/lr1e-06_train_samples_10000/prefix_enabled_false/seed42',
        'unaware': 'unaware/lr1e-06_train_samples_10000/prefix_enabled_ture/seed42',
        'gif_aware': 'gif_aware/num_samples_128_lr2e-06_train_samples_10000_calibration_group_size_128/prefix_enabled_ture_common_clip_enabled_true/phase_T_4_mtn_T_4_warmup_ratio_0.0/seed42',
    }
    references = []
    for mode, relative in refs.items():
        source = model_root / relative / 'config/resolved_config.yaml'
        cfg = load_config(source)
        cfg['phase']['T'], cfg['mtn']['T'], cfg['mtn']['K'] = 4, 4, 6
        cfg['evaluation']['tldr_test_samples'] = args.test_samples
        path = output / 'configs' / f'ref_{mode}.yaml'
        cfg = save_config(path, cfg)
        references.append(dict(id=f'ref_{mode}', kind='reference', config=str(path),
            signature=digest(cfg), ann_dir=str(ArtifactLayout(cfg).ann_dir), family='reference'))
    if getattr(args, 'phase_reference_config', None):
        cfg = load_config(args.phase_reference_config)
        if cfg['experiment']['ann_mode'] != 'phase_aware' or cfg['training']['tldr_train_samples'] != args.train_samples:
            raise ValueError('Phase reference must match candidate mode and training sample count')
        cfg['evaluation']['tldr_test_samples'] = args.test_samples
        path = output / 'configs' / 'ref_phase_control.yaml'
        cfg = save_config(path, cfg)
        references.append(dict(id='ref_phase_control', kind='reference', config=str(path),
            signature=digest(cfg), ann_dir=str(ArtifactLayout(cfg).ann_dir), family='reference'))
    plan = dict(schema=1, fixed={'phase.T': 4, 'mtn.T': 4, 'mtn.K': 6},
        train_samples=args.train_samples, test_samples=args.test_samples, jobs=references + jobs)
    plan_path = output / 'plan.json'
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError('Plan changed; use a new SWEEP_DIR to retain the original plan')
    write(plan_path, plan)
    return plan


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another sweep holds {path}') from exc
        yield


class Executor:
    def __init__(self, output):
        self.output = output
        self.process = None

    def command(self, job_id, stage, args):
        log = self.output / 'logs' / job_id / f'{time.time_ns()}_{stage}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'[{job_id}] {stage}: {shlex.join(args)}\n  log: {log}', flush=True)
        with log.open('w') as stream:
            stream.write(shlex.join(args) + '\n'); stream.flush()
            self.process = subprocess.Popen(args, cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    try:
                        code = self.process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        tail = log.read_text(errors='replace').replace('\r', '\n').splitlines()[-1:]
                        print(f'[{job_id}] {stage} running: {" ".join(tail)[-250:]}', flush=True)
            finally:
                if self.process.poll() is None:
                    self.stop()
                self.process = None
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {log}')

    def stop(self):
        if self.process is None:
            return
        pgid = self.process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        # The launcher may exit before its distributed workers.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()


def metrics_for(job):
    count = load_config(job['config'])['evaluation']['tldr_test_samples']
    return sorted(Path(job['ann_dir']).glob(f'evaluation/tldr/test_samples_{count}/**/metrics.json'))


def validate_metrics(job, cfg, ngpu):
    paths = metrics_for(job)
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError('Ambiguous evaluation results')
    d = read(paths[0])
    expected = dict(samples=cfg['evaluation']['tldr_test_samples'], ann_mode=cfg['experiment']['ann_mode'], neuron='ann',
                    tldr_test_seed=42, world_size=ngpu, max_new_tokens=32, input_length=512,
                    prefix_enabled=final_ann_evaluation_prefix_enabled(cfg))
    if any(d.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Existing metrics protocol mismatch: {paths[0]}')
    if cfg['experiment']['ann_mode'] in ('phase_aware', 'gif_aware'):
        if d.get('calibration_group_size') != cfg['calibration']['group_size'] or d.get('evaluation_common_clip_applied') != cfg['replacement']['common_clip_enabled']:
            raise ValueError('Existing metrics grouping/Clip mismatch')
    if any(not isinstance(d.get(k), (float, int)) or not math.isfinite(d[k]) for k in ('rouge1', 'rouge2', 'rougeL', 'rougeLsum')):
        raise ValueError('Missing/nonfinite ROUGE')
    return paths[0]


def training_digest(cfg):
    comparable = copy.deepcopy(cfg)
    comparable.get('training', {}).pop('audit_gradients', None)
    return digest(comparable)


def run_job(job, executor, ngpu):
    cfg = load_config(job['config'])
    assert_fixed(cfg)
    if digest(cfg) != job['signature']:
        raise ValueError('Planned config was modified')
    layout = ArtifactLayout(cfg)
    old = layout.config_dir / 'resolved_config.yaml'
    if old.exists() and training_digest(load_config(old)) != training_digest(cfg):
        raise ValueError(f'Existing run has different training/config settings: {old}; refusing overwrite')
    if (layout.ann_dir / 'training_result.json').exists() and not old.exists():
        raise ValueError('Existing checkpoint has no resolved config; cannot safely reuse')
    py = sys.executable
    command = lambda stage, argv: executor.command(job['id'], stage, [py, *argv])
    if job['kind'] == 'trial':
        # Require shared base/data/Prefix from this already-prepared project.
        required = [layout.rotation_dir / 'fused_base/config.json',
                    layout.ann_training_prefix_dir / 'prefix_state.json',
                    layout.calibration_data_manifest_path]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f'Shared prerequisite missing: {path}; prepare it using project scripts')
        from snn2.state_validation import validate_site_state_bundle, validate_clip_profile
        sites, clips = layout.ann_training_site_dir, layout.ann_training_clip_profile_dir
        if not (sites / 'calibration_state_manifest.json').exists():
            if sites.exists() and any(sites.iterdir()):
                raise ValueError(f'Incomplete Stage A; preserve/inspect before retry: {sites}')
            command('calibration_A', ['scripts/calibrate_sites.py', '--config', job['config'], '--stage', 'ann_training', '--calibration-phase', 'A'])
        manifest = validate_site_state_bundle(sites, clip_policy='forbid_all')['manifest']
        if manifest.get('calibration_group_size') != cfg['calibration']['group_size'] or manifest.get('calibration_num_samples') != 128:
            raise ValueError('Stage A grouping/sample count mismatch; refusing reuse')
        if not (clips / 'clip_profile_manifest.json').exists():
            if clips.exists() and any(p.name not in ('config', 'logs') for p in clips.iterdir()):
                raise ValueError(f'Incomplete Stage B; preserve/inspect before retry: {clips}')
            command('calibration_B', ['scripts/calibrate_sites.py', '--config', job['config'], '--stage', 'ann_training', '--calibration-phase', 'B'])
        validate_clip_profile(sites, clips, phase_T=4, mtn_T=4,
            group_size=cfg['calibration']['group_size'], num_samples=128)
    training = layout.ann_dir / 'training_result.json'
    if not training.exists():
        if job['kind'] == 'reference':
            raise FileNotFoundError(f'Reference training result missing: {training}')
        if layout.ann_checkpoint_dir.exists() and any(layout.ann_checkpoint_dir.iterdir()):
            raise ValueError('Partial final checkpoint without completed training; refusing overwrite')
        if metrics_for(job):
            raise ValueError('Orphan metrics without completed training; refusing overwrite')
        command('train', ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ngpu}',
                         'scripts/train_ann.py', '--config', job['config']])
    result = read(training)
    if result.get('world_size') != ngpu or result.get('train_samples') != cfg['training']['tldr_train_samples']:
        raise ValueError('Training sample count/world size mismatch')
    validate_checkpoint(layout.ann_checkpoint_dir)
    if cfg['experiment']['ann_mode'] in ('phase_aware', 'gif_aware'):
        from snn2.training import validate_recorded_training_artifact_provenance
        validate_recorded_training_artifact_provenance(cfg, layout)
    if validate_metrics(job, cfg, ngpu):
        return 'reused'
    command('evaluate', ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ngpu}',
        'scripts/evaluate_tldr.py', '--config', job['config'], '--neuron', 'ann'])
    if not validate_metrics(job, cfg, ngpu):
        raise RuntimeError('Evaluation returned without metrics')
    return 'completed'


def validate_checkpoint(directory):
    if not (directory / 'config.json').exists():
        raise FileNotFoundError('Completed checkpoint config missing')
    index = directory / 'model.safetensors.index.json'
    if index.exists():
        shards = set(read(index)['weight_map'].values())
    else:
        shards = {'model.safetensors'}
    if not shards or any(not (directory / name).is_file() or (directory / name).stat().st_size == 0 for name in shards):
        raise FileNotFoundError('Completed checkpoint has missing/empty weight shards')


def paired_report(output, best, references):
    """Exploratory paired bootstrap; never equate different sample IDs/references."""
    import numpy as np
    from rouge_score import rouge_scorer
    comparisons = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    def predictions(row):
        path = Path(row['metrics_path']).with_name('predictions.jsonl')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        result = {record['index']: record for record in records}
        if len(result) != len(records):
            raise ValueError('Duplicate prediction indices')
        return result
    for reference in references:
        comparison = dict(candidate=best['id'], reference=reference['id'],
            interpretation='Exploratory test-subset comparison after model selection; not an independent confirmatory interval')
        try:
            a, b = predictions(best), predictions(reference)
            if len(a) != best.get('test_samples', 128) or a.keys() != b.keys() or any(a[i]['reference'] != b[i]['reference'] for i in a):
                raise ValueError('Prediction samples/references differ')
            differences = np.array([scorer.score(a[i]['reference'], a[i]['prediction'])['rougeL'].fmeasure -
                scorer.score(b[i]['reference'], b[i]['prediction'])['rougeL'].fmeasure for i in sorted(a)])
            means = np.random.default_rng(42).choice(differences, (5000, len(a)), replace=True).mean(axis=1)
            comparison.update(samples=len(a), mean_rougeL_delta=float(differences.mean()),
                bootstrap_95_interval=np.quantile(means, [.025, .975]).tolist())
        except (OSError, ValueError, KeyError) as exc:
            comparison['error'] = str(exc)
        comparisons.append(comparison)
    write(output / 'paired_bootstrap.json', comparisons)


def log_statistics(output, job_id):
    points = []
    for log in sorted((output / 'logs' / job_id).glob('*_train.log'))[-1:]:
        for item in re.findall(r"\{'loss':[^\r\n]*?\}", log.read_text(errors='replace')):
            try:
                points.append(ast.literal_eval(item))
            except (ValueError, SyntaxError):
                pass
    norms = [p['grad_norm'] for p in points if isinstance(p.get('grad_norm'), (float, int))]
    return dict(logged_steps=len(points), first_loss=points[0]['loss'] if points else None,
        last_loss=points[-1]['loss'] if points else None, max_grad_norm=max(norms) if norms else None)


def summarize(output):
    plan = read(output / 'plan.json')
    from snn2.training_audit import summarize_audits
    summarize_audits(plan, output)
    rows = []
    for job in plan['jobs']:
        cfg = load_config(job['config'])
        state_path = output / 'status' / f"{job['id']}.json"
        state = read(state_path) if state_path.exists() else {'status': 'pending'}
        row = dict(id=job['id'], kind=job['kind'], family=job['family'], status=state['status'],
            error=state.get('error', ''), group=cfg['calibration']['group_size'],
            slope=cfg['phase']['surrogate_slope'], lr=cfg['training']['learning_rate'],
            warmup=cfg['training']['warmup_ratio'], clip=cfg['replacement']['common_clip_enabled'],
            train_samples=cfg['training']['tldr_train_samples'], test_samples=cfg['evaluation']['tldr_test_samples'],
            config=job['config'], **log_statistics(output, job['id']))
        audit_path = Path(job['ann_dir']) / 'training_audit.jsonl'
        row['training_audit_path'] = str(audit_path) if audit_path.exists() else ''
        row['training_audit_status'] = 'present' if audit_path.exists() else 'unavailable'
        training_path = Path(job['ann_dir']) / 'training_result.json'
        if training_path.exists():
            try:
                t = read(training_path)
                row.update(train_loss=t.get('train_loss'), train_runtime=t.get('train_runtime'),
                           world_size=t.get('world_size'))
            except ValueError as exc:
                row['training_error'] = str(exc)
        try:
            path = validate_metrics(job, cfg, state.get('world_size', 4))
            if path:
                m = read(path)
                row.update(metrics_path=str(path), **{k: m[k] for k in ('rouge1', 'rouge2', 'rougeL', 'rougeLsum')})
                if row['status'] not in ('completed', 'reused'):
                    row['status'] = 'metrics_present_unverified'  # Never silently count a failed trial as success.
        except (ValueError, KeyError) as exc:
            row['metrics_error'] = str(exc)
        rows.append(row)
    write(output / 'summary.json', rows)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r['status'] in ('completed', 'reused') and 'rougeL' in r]
    references = [r for r in valid if r['kind'] == 'reference']
    ranked = sorted((r for r in valid if r['kind'] == 'trial'), key=lambda r: r['rougeL'], reverse=True)
    lines = ['# Phase-aware sweep results', '',
        f"固定 phase.T=4、mtn.T=4、mtn.K=6；候选训练 {plan.get('train_samples', 1024)} 条，基线训练 10000 条，均评估相同 seed42 的 {plan.get('test_samples', 128)} 条测试样本。",
        '这是测试子集上的探索性比较；不同样本数不能直接配对，扩大测试集也不等于新的独立验证集。',
        '指标仅计入完成或已验证复用的实验；失败、未运行和未验证结果保留在 CSV/JSON。', '',
        f'计划 {len(rows)} 项；完成/复用 {len(valid)} 项；其余 {len(rows)-len(valid)} 项。', '',
        '| ID | G | slope | lr | warmup | Clip | R1 | R2 | RL | RLsum |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in references + ranked:
        lines.append(f"| {r['id']} | {r['group']} | {r['slope']} | {r['lr']} | {r['warmup']} | {r['clip']} | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} | {r['rougeLsum']:.4f} |")
    if ranked:
        best = ranked[0]
        lines += ['', f"当前候选最高 ROUGE-L：{best['id']}，{best['rougeL']:.6f}。"]
        lines += [f"相对 {r['id']} 的 ROUGE-L 差值：{best['rougeL']-r['rougeL']:+.6f}。" for r in references]
    lines += ['', '## 未完成项', '']
    lines += [f"- {r['id']}: {r['status']} {r['error']}" for r in rows if r['status'] not in ('completed', 'reused')]
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    # Preserve all historical comparisons with their actual test sizes and paths.
    history = []
    model_root = ArtifactLayout(load_config(plan['jobs'][0]['config'])).model_root
    for p in sorted(model_root.glob('**/ann/evaluation/**/metrics.json')):
        try:
            d = read(p)
        except ValueError as exc:
            history.append(dict(path=str(p), error=str(exc)))
            continue
        history.append(dict(path=str(p), **{k: d.get(k) for k in ('ann_mode', 'samples', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'world_size')}))
    write(output / 'historical_metrics.json', history)
    if ranked:
        paired_report(output, ranked[0], references)
    print(f'Summary: {output / "summary.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', default='configs/generated/exp1_qwen3_8b_tldr__phase_aware.yaml')
    parser.add_argument('--matrix-text', default='')
    parser.add_argument('--output', default='artifacts/phase_aware_sweep_v1')
    parser.add_argument('--train-samples', type=int, choices=(1024, 10000), default=1024)
    parser.add_argument('--test-samples', type=int, choices=(128, 1000), default=128)
    parser.add_argument('--require-clip-true', action='store_true')
    parser.add_argument('--phase-reference-config', default=None)
    parser.add_argument('--audit-gradients', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / 'sweep.lock'):
        if args.summarize_only:
            summarize(output); return 0
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
        if any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
            raise ValueError('CUDA_VISIBLE_DEVICES must contain unique comma-separated GPU indices')
        if len(devices) != 4:
            raise ValueError('Use four GPUs to match the reference and existing trial world size')
        plan = make_plan(args, output)
        trial_count = sum(j['kind'] == 'trial' for j in plan['jobs'])
        print(f"Plan: {trial_count} training trials + {len(plan['jobs'])-trial_count} reference evaluations; GPUs={devices}")
        print('Compatible completed runs are reused; each new 8B checkpoint needs approximately 16 GB.')
        if args.dry_run:
            for job in plan['jobs']:
                print(job['id'], job['config'])
            summarize(output); return 0
        executor = Executor(output)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        failures = 0
        # Serialize this runner across output dirs: Stage A and data are shared.
        with lock(ROOT / 'artifacts/.qwen3_8b_phase_sweep.lock'):
            for index, job in enumerate(plan['jobs'], 1):
                state_path = output / 'status' / f"{job['id']}.json"
                state = dict(status='running', started_at=now(), world_size=4)
                write(state_path, state)
                print(f"\n[{index}/{len(plan['jobs'])}] {job['id']}", flush=True)
                try:
                    state['status'] = run_job(job, executor, 4)
                except KeyboardInterrupt:
                    executor.stop()
                    state.update(status='interrupted', finished_at=now(), error='User interrupted; rerun restarts unfinished training, completed runs are reused')
                    write(state_path, state); summarize(output)
                    return 130
                except Exception as exc:
                    failures += 1
                    state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    print(f"FAILED: {state['error']}", flush=True)
                state['finished_at'] = now()
                write(state_path, state)
                summarize(output)
        return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
