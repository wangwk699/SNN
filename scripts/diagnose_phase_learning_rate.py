#!/usr/bin/env python
"""Paired validation diagnostics for lr2e-6/lr3e-6; never write model weights."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import numpy as np
import torch
import diagnose_phase_aware as core
from phase_aware_sweep import Executor, lock, read, write

ROOT = core.ROOT
BLOCKS = {'early': range(0,12), 'middle': range(12,24), 'late': range(24,36), 'all': range(36)}
TAGS = {'p01': 'lr2e-6', 'p03': 'lr3e-6'}


def targets(controller, block):
    keys = {k+'/phase' for k, v in controller._modules.items()
            if 'phase' in v and '/site_05_' in k and int(k.split('/')[0].split('_')[1]) in BLOCKS[block]}
    if len(keys) != len(BLOCKS[block]):
        raise ValueError('Expected one Site5 per layer in the selected block')
    return keys


def worker(output, tag, limit):
    fingerprint = read(output/'run_fingerprint.json')['sha256']
    pending = []
    for row in read(output/'validation_manifest.json')['samples'][:limit]:
        path = output/'records'/tag/f"sample_{row['index']:05d}.json"
        if path.exists():
            if read(path)['fingerprint'] != fingerprint: raise ValueError('Stale diagnostic record')
        else: pending.append((row,path))
    if not pending: return
    model, controller, probe, cfg = core.build_model(output, tag)
    core.warmup(model, controller, probe, core.batch_from(pending[0][0], model))
    try:
        block_targets = {k: targets(controller,k) for k in BLOCKS}
        for row, path in pending:
            batch = core.batch_from(row,model)
            # Query t predicts label t+1; keep every key including fixed Prefix.
            probe.query_indices = torch.nonzero(batch['labels'][0,1:] != -100).flatten()
            probe.bypass = None; probe.reference_hidden = None
            identity, ref_logits = core.run_pass(model,controller,probe,batch,'identity',4.)
            probe.reference_hidden = probe.hidden.copy()
            baseline, logits = core.run_pass(model,controller,probe,batch,'phase',4.)
            baseline['against_identity'] = core.logit_difference(ref_logits,logits)
            variants = {}
            for block, selected in block_targets.items():
                probe.bypass = selected
                result, block_logits = core.run_pass(model,controller,probe,batch,'phase',4.,collect=False)
                variants[block] = {'loss':result['loss'], 'delta_vs_phase':result['loss']-baseline['loss'],
                    'against_identity':core.logit_difference(ref_logits,block_logits)}
            write(path, {'fingerprint':fingerprint,'index':row['index'],'checkpoint':TAGS[tag],
                'diagnostic_only':True,'identity':identity,'phase':baseline,'bypasses':variants})
            print(f"{TAGS[tag]} index={row['index']} phase NLL={baseline['loss']:.5f}",flush=True)
    finally: probe.close()


def summarize(output):
    data = {tag:[read(p) for p in sorted((output/'records'/tag).glob('sample_*.json'))] for tag in TAGS}
    result = {'diagnostic_only':True,'mapping':TAGS,'checkpoints':{},'paired':{}}
    for tag, rows in data.items():
        if not rows: continue
        w = np.array([r['phase']['completion_tokens'] for r in rows])
        info = {'samples':len(rows),'completion_tokens':int(w.sum())}
        for mode in ['identity','phase']:
            info[mode+'_nll'] = float(np.average([r[mode]['loss'] for r in rows],weights=w))
        for category in ['local_sampled','attention_sampled_rows','cumulative_hidden_sampled']:
            info[category] = core.merge_accumulators([r['phase'][category] for r in rows])
        info['bypasses'] = {}
        ix = np.random.default_rng(42).integers(0,len(rows),(5000,len(rows)))
        for block in BLOCKS:
            delta=np.array([r['bypasses'][block]['delta_vs_phase'] for r in rows])
            distribution=(delta[ix]*w[ix]).sum(1)/w[ix].sum(1)
            info['bypasses'][block]={'weighted_nll_delta':float(np.average(delta,weights=w)),
                'pointwise_95_ci':np.quantile(distribution,[.025,.975]).tolist()}
        result['checkpoints'][TAGS[tag]]=info
    a={r['index']:r for r in data['p01']}; b={r['index']:r for r in data['p03']}
    common=sorted(a.keys() & b.keys())
    if common:
        w=np.array([a[i]['phase']['completion_tokens'] for i in common])
        assert all(a[i]['phase']['completion_tokens']==b[i]['phase']['completion_tokens'] for i in common)
        ix=np.random.default_rng(42).integers(0,len(common),(5000,len(common)))
        for mode in ['identity','phase']:
            d=np.array([b[i][mode]['loss']-a[i][mode]['loss'] for i in common]);boot=(d[ix]*w[ix]).sum(1)/w[ix].sum(1)
            result['paired'][mode]={'samples':len(common),'lr3_minus_lr2_nll':float(np.average(d,weights=w)),
                'pointwise_95_ci':np.quantile(boot,[.025,.975]).tolist()}
    write(output/'comparison.json',result)
    lines=['# Learning-rate diagnostics','', 'Validation teacher forcing; diagnostic-only Site5 bypasses. Negative NLL delta means improvement.','',
        '| checkpoint | samples | identity NLL | Phase NLL |','|---|---:|---:|---:|']
    for tag,v in result['checkpoints'].items():
        lines.append(f"| {tag} | {v['samples']} | {v['identity_nll']:.6f} | {v['phase_nll']:.6f} |")
        for block, values in v['bypasses'].items():lines.append(f"\n{tag} bypass {block}: {values}")
    lines += ['', 'Paired lr3-minus-lr2: '+json.dumps(result['paired']),
        '', 'Intervals: 5000 paired example bootstrap resamples, seed42, no multiplicity correction. Attention rows are completion-predicting queries only.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/phase_lr_diagnostics_v1/calibration_group_size_4'))
    p.add_argument('--lr2-config',default='artifacts/phase_aware_round2_formal_cliptrue_v3/configs/p01_g4_s4.0_lr2e-06_w0.2_clip1.yaml')
    p.add_argument('--lr3-config',default='artifacts/phase_aware_lr3e6_formal_v1/configs/p01_g4_s4.0_lr3e-06_w0.2_clip1.yaml')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--worker',choices=list(TAGS));p.add_argument('--limit',type=int,default=128)
    args=p.parse_args();os.chdir(ROOT);output=args.output.resolve()
    if args.worker:worker(output,args.worker,args.limit);return
    with lock(output/'diagnostics.lock'):
        if args.summarize_only:summarize(output);return
        paths={'p01':args.lr2_config,'p03':args.lr3_config};core.validate_configs(paths)
        for tag,path in paths.items():
            cfg=core.load_config(path)
            expected=2e-6 if tag=='p01' else 3e-6
            if cfg['training']['learning_rate']!=expected or cfg['phase']['surrogate_slope']!=4 or cfg['training']['warmup_ratio']!=.2:
                raise ValueError('Expected slope4, warmup0.2 and the designated learning rates')
        print('Two checkpoints; validation128 (smoke16); identity + Phase + four Site5 block bypasses; no backward or optimizer.',flush=True)
        if args.dry_run:return
        if torch.cuda.device_count()!=4:raise ValueError('Exactly four visible GPUs required')
        options=SimpleNamespace(samples=128,smoke_samples=16,slopes=[4.],top_k=3,formal_dir=Path(args.lr3_config).resolve().parents[1])
        # Freeze this extension too, before any output can be reused.
        extension={'script_sha256':core.sha256_file(__file__),'blocks':{k:list(v) for k,v in BLOCKS.items()},'mapping':TAGS}
        if (output/'extension.json').exists() and read(output/'extension.json')!=extension:raise ValueError('Changed diagnostic code: use new output directory')
        write(output/'extension.json',extension)
        core.prepare(output,paths,options)
        executor=Executor(output)
        def interrupt(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        try:
            for limit in ([16] if args.smoke_only else [16,128]):
                for tag in TAGS:
                    executor.command(tag+f'_{limit}','diagnose',[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--worker',tag,'--limit',str(limit)])
                    summarize(output)
            write(output/'status.json',{'status':'smoke_completed' if args.smoke_only else 'completed'})
        except BaseException as exc:
            executor.stop();write(output/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','error':str(exc)})
            summarize(output);raise

if __name__=='__main__':main()
