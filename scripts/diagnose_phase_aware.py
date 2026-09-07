#!/usr/bin/env python
"""Teacher-forced validation probes, no optimizer, no official evaluation writes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from snn2.artifacts import ArtifactLayout, sha256_file
from snn2.config import load_config
from snn2.phase_diagnostics import Probe, completion_loss, logit_sample, logit_difference
from phase_aware_sweep import Executor, lock, read, write


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [clean(v) for v in value]
    return value


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config_paths(formal):
    jobs = read(formal/'plan.json')['jobs']
    p01 = next(j for j in jobs if j.get('family') == 'formal_p05')
    p03 = next(j for j in jobs if j.get('family') == 'formal_p47')
    return {'untrained_rotated': p03['config'], 'p01': p01['config'], 'p03': p03['config']}


def source_for(tag, cfg):
    layout = ArtifactLayout(cfg)
    return layout.rotation_dir/'fused_base' if tag == 'untrained_rotated' else layout.ann_checkpoint_dir


def validate_configs(paths):
    reference=load_config(paths['p03'])
    for tag, path in paths.items():
        cfg=load_config(path)
        if cfg['data']!=reference['data'] or cfg['experiment']['model_name']!=reference['experiment']['model_name'] or cfg['training']['dtype']!=reference['training']['dtype']:
            raise ValueError('Diagnostic checkpoints must share data/tokenization and model dtype')
        if (cfg['phase']['T'],cfg['mtn']['T'],cfg['mtn']['K'],cfg['calibration']['group_size']) != (4,4,6,4):
            raise ValueError(f'{tag}: requires phase.T=4, mtn.T=4, mtn.K=6, G=4')
        if cfg['experiment']['ann_mode'] != 'phase_aware' or not cfg['replacement']['common_clip_enabled']:
            raise ValueError('p01/p03 must retain Phase-aware + common Clip')
        if cfg['training']['gradient_checkpointing'] or cfg['training']['attn_implementation'] != 'eager':
            raise ValueError('Diagnostic uses original eager, no-gradient-checkpointing semantics')
        if not (source_for(tag,cfg)/'config.json').exists():
            raise FileNotFoundError(source_for(tag,cfg))


def prepare(output, paths, args):
    from snn2.training import validate_recorded_training_artifact_provenance
    from snn2.modeling import load_tokenizer
    from snn2.data import _load_raw, tokenize_row
    cfg=load_config(paths['p03'])
    provenance={}
    for tag,path in paths.items():
        c=load_config(path); layout=ArtifactLayout(c)
        recorded=validate_recorded_training_artifact_provenance(c,layout)
        source=source_for(tag,c)
        files=sorted(source.glob('*.safetensors'))+sorted(source.glob('*.json'))
        provenance[tag]={'config':str(Path(path).resolve()),'config_hash':sha256_file(path),
            'source':str(source.resolve()),'source_hashes':{p.name:sha256_file(p) for p in files},
            'frozen_training_artifacts':recorded}
    for key in ('ann_training_prefix_kv_sha256','ann_training_stage_a_manifest_sha256','ann_training_clip_profile_manifest_sha256'):
        if provenance['p01']['frozen_training_artifacts'][key] != provenance['p03']['frozen_training_artifacts'][key]:
            raise ValueError(f'p01/p03 do not share {key}')
    code_files=['snn2/phase_diagnostics.py','scripts/diagnose_phase_aware.py','snn2/neurons.py',
        'snn2/controller.py','snn2/model_integration.py','snn2/data.py','snn2/modeling.py','snn2/prefix_cache.py']
    specification={'schema':1,'diagnostic_only':True,'provenance':provenance,
        'code_hashes':{p:sha256_file(ROOT/p) for p in code_files},'samples':args.samples,
        'smoke_samples':args.smoke_samples,'slopes':args.slopes,'top_k':args.top_k,
        'torch_version':torch.__version__,'seed':42,'split':'validation','formal_dir':str(args.formal_dir.resolve()),
        'sampling':'seeded permutation; first 128 records with nonempty shifted completion labels',
        'maximum_sequence_length':cfg['data']['max_seq_length'],
        'activation_sample_limit':4096,'head_sample_limit':256,'gradient_sample_limit':512,
        'logit_completion_positions':32,'gpu_count':torch.cuda.device_count()}
    if (output/'specification.json').exists() and read(output/'specification.json') != specification:
        raise ValueError('Inputs/code/options changed. Use a new DIAGNOSTIC_DIR; do not mix runs.')
    write(output/'specification.json',specification)
    tokenizer=load_tokenizer(cfg,str(source_for('p03',cfg)))
    for tag, path in paths.items():
        c=load_config(path); other=load_tokenizer(c,str(source_for(tag,c)))
        if other.get_vocab()!=tokenizer.get_vocab() or other.all_special_ids!=tokenizer.all_special_ids:
            raise ValueError(f'Tokenizer mismatch: {tag}')
    dataset=_load_raw(cfg)['validation']
    indices=list(range(len(dataset))); random.Random(42).shuffle(indices)
    samples=[]; excluded=[]
    for index in indices:
        encoded=tokenize_row(dataset[index],tokenizer,cfg)
        if not any(label != -100 for label in encoded['labels'][1:]):
            excluded.append(index);continue
        samples.append({'index':index,'raw_record_sha256':json_hash(dataset[index]),'encoded':encoded})
        if len(samples)==args.samples:break
    if len(samples)!=args.samples:raise ValueError('Not enough eligible validation records')
    manifest={'dataset':cfg['data']['dataset_name'],'revision':cfg['data']['dataset_revision'],
        'split':'validation','seed':42,'samples':samples,'excluded_no_completion':excluded,
        'tokenizer_hashes':{p.name:sha256_file(p) for p in source_for('p03',cfg).glob('*token*')}}
    if (output/'validation_manifest.json').exists() and read(output/'validation_manifest.json') != manifest:
        raise ValueError('Validation selection changed')
    write(output/'validation_manifest.json',manifest)
    write(output/'run_fingerprint.json',{'sha256':json_hash([specification,manifest])})


def build_model(output,tag):
    from snn2.controller import SiteController
    from snn2.modeling import load_model, rotation_state, prefix_key_values_for_stage
    from snn2.model_integration import install_model_integration
    from snn2.prefix_cache import install_prefix_kv_forward, prefix_length
    spec=read(output/'specification.json'); entry=spec['provenance'][tag]
    cfg=load_config(entry['config']);layout=ArtifactLayout(cfg)
    model=load_model(cfg,entry['source'],training=False,device_map='balanced')
    if any(p.device.type!='cuda' for p in model.parameters()):
        raise RuntimeError('CPU/disk model offload is not supported for this backward diagnostic')
    model.eval()  # deterministic dropout-free diagnosis, not optimizer training
    for p in model.parameters():p.requires_grad_(True)
    controller=SiteController(mode='phase',site_root=layout.ann_training_site_dir,
        clip_root=layout.ann_training_clip_profile_dir,common_clip_enabled=True,
        phase_T=4,mtn_T=4,mtn_K=6,mtn_threshold_factor=cfg['mtn']['threshold_factor'],
        phase_surrogate_slope=cfg['phase']['surrogate_slope'])
    install_model_integration(model,controller,rotation_state(cfg,layout))
    prefix=prefix_key_values_for_stage(cfg,layout,stage='ann_training')
    install_prefix_kv_forward(model,prefix,controller=controller)
    return model,controller,Probe(controller,prefix_length(prefix)),cfg


def batch_from(record, model):
    device=next(model.parameters()).device
    return {k:torch.tensor([v],dtype=torch.long,device=device) for k,v in record['encoded'].items()}


def set_mode(controller,mode,slope):
    controller.mode=mode
    controller.phase_surrogate_slope=slope
    for modules in controller._modules.values():
        if 'phase' in modules:modules['phase'].slope=slope
    if controller._final_norm_phase is not None:controller._final_norm_phase.slope=slope


def forward(model,batch):
    outputs=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
    loss,indices=completion_loss(outputs.logits,batch['labels'])
    selected=logit_sample(outputs.logits,indices)
    return loss,selected,len(indices)


def run_pass(model,controller,probe,batch,mode,slope,backward=False,collect=True):
    model.zero_grad(set_to_none=True)
    probe.reset();probe.enabled=collect;probe.backward=backward
    set_mode(controller,mode,slope)
    with torch.set_grad_enabled(backward):
        loss,logits,tokens=forward(model,batch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite completion loss')
        if backward:loss.backward()
    result={'loss':float(loss.detach()),'completion_tokens':tokens,'mode':mode,'slope':slope,
        'local_sampled':probe.local.export(),'attention_sampled_rows':probe.attention.export(),
        'cumulative_hidden_sampled':probe.cumulative.export(),'gradient_statistics':probe.gradients.export()}
    if backward:
        norm=math.sqrt(probe.global_gradient2)
        result.update(global_gradient_norm=norm,nonfinite_gradient_sample_count=probe.global_nonfinite,
            hypothetical_clip_coefficient=min(1.,1./max(norm,1e-30)),
            hypothetical_global_clip_triggered=norm>1.)
    del loss
    model.zero_grad(set_to_none=True)
    return clean(result),logits


def warmup(model,controller,probe,batch):
    with torch.no_grad():forward(model,batch)
    probe.attach(model)


def worker(output,tag,limit):
    manifest=read(output/'validation_manifest.json');spec=read(output/'specification.json')
    fingerprint=read(output/'run_fingerprint.json')['sha256']
    samples=manifest['samples'][:limit]; destination=output/'checkpoints'/tag
    pending=[]
    for row in samples:
        path=destination/f"sample_{row['index']:05d}.json"
        if path.exists():
            if read(path).get('fingerprint')!=fingerprint:raise ValueError('Stale sample result')
        else:pending.append(row)
    if not pending:return
    model,controller,probe,cfg=build_model(output,tag)
    warmup(model,controller,probe,batch_from(pending[0],model))
    native=float(cfg['phase']['surrogate_slope'])
    slopes=spec['slopes'] if tag=='p03' else [native]
    for counter,row in enumerate(pending,1):
        start=time.monotonic();batch=batch_from(row,model)
        probe.reference_hidden=None;probe.gradient_reference=None;probe.bypass=None
        identity,ref_logits=run_pass(model,controller,probe,batch,'identity',native,backward=True)
        ref_hidden=probe.hidden.copy();ref_grad=probe.parameter_samples.copy()
        record={'fingerprint':fingerprint,'index':row['index'],'checkpoint':tag,
            'diagnostic_only':True,'identity':identity,'phase':{}}
        first_logits=None;first_loss=None
        for slope in slopes:
            probe.reference_hidden=ref_hidden;probe.gradient_reference=ref_grad
            result,logits=run_pass(model,controller,probe,batch,'phase',slope,backward=True)
            result['against_identity']=logit_difference(ref_logits,logits)
            result['loss_delta_vs_identity']=result['loss']-identity['loss']
            if first_logits is None:first_logits=logits;first_loss=result['loss']
            result['slope_forward_invariance']={'max_abs_sampled_logit_difference':float((logits-first_logits).abs().max()),
                'loss_difference':result['loss']-first_loss}
            if not torch.equal(logits,first_logits) or abs(result['loss']-first_loss)>1e-6:
                write(destination/'invariance_failure.json',dict(index=row['index'],slope=slope,details=result['slope_forward_invariance']))
                raise RuntimeError('Changing only slope changed hard forward: inspect invariance_failure.json')
            record['phase'][str(float(slope))]=result
        record['wall_seconds']=time.monotonic()-start
        write(destination/f"sample_{row['index']:05d}.json",record)
        print(f'{tag} {counter}/{len(pending)} validation_index={row["index"]} loss={record["phase"][str(native)]["loss"]:.4f} elapsed={record["wall_seconds"]:.1f}s',flush=True)
    probe.close()


def merge_accumulators(records):
    merged={}
    for record in records:
        for key,row in record.items():
            out=merged.setdefault(key,{})
            for name,value in row.items():
                if name.endswith('_max'):out[name]=max(out.get(name,-math.inf),value)
                else:out[name]=out.get(name,0)+value
    for row in merged.values():
        if 'x2_sum' in row:
            row['pooled_sampled_nmse']=row.get('error2_sum',0)/max(row['x2_sum'],1e-30)
            row['pooled_sampled_relative_l2']=math.sqrt(row['pooled_sampled_nmse'])
            row['pooled_sampled_signed_error_mean']=row.get('signed_error_sum',0)/max(row.get('count_sum',0),1)
        for name,value in list(row.items()):
            if name.endswith('_sum'):
                metric=name[:-4];n=row.get(metric+'_observations',row['calls'])
                row[metric+'_mean_per_call']=value/max(n,1)
    return merged


def summarize(output):
    output.mkdir(parents=True,exist_ok=True)
    records={tag:[read(p) for p in sorted((output/'checkpoints'/tag).glob('sample_*.json'))]
        for tag in ('untrained_rotated','p01','p03')}
    requested=read(output/'specification.json')['samples'] if (output/'specification.json').exists() else 128
    summary={'diagnostic_only':True,'checkpoints':{},'expected_samples_per_checkpoint':requested,'progress':{}}
    for tag,rows in records.items():
        if not rows:continue
        seconds=sum(row.get('wall_seconds',0) for row in rows)
        summary['progress'][tag]={'completed_samples':len(rows),'expected_samples':requested,
            'recorded_sample_seconds':seconds,'estimated_total_sample_seconds':seconds/len(rows)*requested,
            'note':'sample timing excludes model loading, hashing and bypass phase'}
        table={}
        names=['identity']+list(rows[0]['phase'])
        for name in names:
            passes=[r['identity'] if name=='identity' else r['phase'][name] for r in rows]
            tokens=sum(p['completion_tokens'] for p in passes)
            info={'samples':len(rows),'completion_tokens':tokens,
                'completion_nll_token_weighted':sum(p['loss']*p['completion_tokens'] for p in passes)/tokens,
                'mean_global_gradient_norm':sum(p['global_gradient_norm'] for p in passes if p['global_gradient_norm'] is not None)/max(sum(p['global_gradient_norm'] is not None for p in passes),1),
                'nonfinite_global_gradient_norm_passes':sum(p['global_gradient_norm'] is None for p in passes),
                'global_clip_trigger_fraction':sum(p['hypothetical_global_clip_triggered'] for p in passes)/len(passes),
                'nonfinite_gradient_sample_count':sum(p['nonfinite_gradient_sample_count'] for p in passes)}
            if name!='identity':
                info.update(mean_kl_identity_to_phase=sum(p['against_identity']['kl_identity_to_probe'] for p in passes)/len(passes),
                    max_slope_invariance_logit_difference=max(p['slope_forward_invariance']['max_abs_sampled_logit_difference'] for p in passes))
            for kind in ('local_sampled','attention_sampled_rows','cumulative_hidden_sampled','gradient_statistics'):
                info[kind]=merge_accumulators([p[kind] for p in passes])
            gradients=info['gradient_statistics']
            for key,stats in gradients.items():
                n=stats.get('count_sum',0)-stats.get('nonfinite_sum',0)
                stats['sampled_gradient_rms']=math.sqrt(stats.get('sum_sq_sum',0)/max(n,1))
            for key,stats in gradients.items():
                if key.endswith('/input') and key[:-5]+'output' in gradients:
                    other=gradients[key[:-5]+'output']['sampled_gradient_rms']
                    stats['input_over_output_gradient_rms']=stats['sampled_gradient_rms']/max(other,1e-30)
            table[name]=info
        summary['checkpoints'][tag]=table
    write(output/'summary.json',clean(summary))
    for category in ('local_sampled','attention_sampled_rows','cumulative_hidden_sampled','gradient_statistics'):
        table=[]
        for tag, passes in summary['checkpoints'].items():
            for mode, values in passes.items():
                for location, stats in values[category].items():
                    table.append(dict(checkpoint=tag,pass_or_slope=mode,location=location,**stats))
        if table:
            with (output/(category+'.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in table for k in row)))
                writer.writeheader();writer.writerows(table)
    summarize_layer_gradients(output,summary)
    audit_training_clip(output)
    lines=['# Phase diagnostic summary','','Diagnostic only; validation teacher forcing, no optimizer steps. Activation/error quantiles are deterministic samples, not full-tensor distributions.','',
        '| checkpoint | pass/slope | samples | completion NLL | global grad norm mean | hypothetical clip fraction |',
        '|---|---|---:|---:|---:|---:|']
    for tag,table in summary['checkpoints'].items():
        for name,v in table.items():lines.append(f"| {tag} | {name} | {v['samples']} | {v['completion_nll_token_weighted']:.5f} | {v['mean_global_gradient_norm']:.4g} | {v['global_clip_trigger_fraction']:.3f} |")
    lines+=['','Full per-site/head, layer and parameter tables: summary.json; per-sample paired records: checkpoints/.','A high error rank is a locator, not proof of a bug or a valid replacement for formal model scoring.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    return summary


def summarize_layer_gradients(output,summary):
    rows=[]
    for tag,passes in summary['checkpoints'].items():
        for mode,metrics in passes.items():
            groups={}
            for key,values in metrics['gradient_statistics'].items():
                if not key.startswith('parameter/'):continue
                match=re.search(r'layers\.(\d+)\.',key)
                group=f'layer_{int(match[1]):03d}' if match else key.split('/')[1].rsplit('.',1)[0]
                dest=groups.setdefault(group,{'gradient_squared':0.,'weight_squared':0.})
                dest['gradient_squared']+=values.get('exact_norm_squared_sum',0.)/max(metrics['samples'],1)
                dest['weight_squared']+=values.get('weight_norm_squared_sum',0.)/max(metrics['samples'],1)
            for group,value in sorted(groups.items()):
                grad=math.sqrt(value['gradient_squared']); weight=math.sqrt(value['weight_squared'])
                rows.append(dict(checkpoint=tag,pass_or_slope=mode,layer=group,
                    gradient_norm_rms_across_samples=grad,weight_norm=weight,
                    gradient_weight_ratio=grad/max(weight,1e-30)))
    write(output/'layer_gradient_norms.json',clean(rows))
    if rows:
        with (output/'layer_gradient_norms.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_training_clip(output):
    path=output/'specification.json'
    if not path.exists():return
    spec=read(path); rows={}
    for tag in ('p01','p03'):
        config=Path(spec['provenance'][tag]['config']); cfg=load_config(config)
        logs=sorted((Path(spec['formal_dir'])/'logs'/config.stem).glob('*_train.log'))
        if not logs:rows[tag]={'status':'training stdout unavailable'};continue
        text=logs[-1].read_text(errors='replace')
        values=[float(v) for v in re.findall(r"'grad_norm':\s*([^,}]+)",text)]
        threshold=float(cfg['training']['max_grad_norm'])
        rows[tag]={'log':str(logs[-1]),'logged_steps':len(values),'threshold':threshold,
            'estimated_clip_trigger_fraction':sum(v>threshold for v in values)/max(len(values),1),
            'max_logged_preclip_norm':max(values) if values else None,
            'interpretation':'from logged pre-clipping norms; not measured optimizer update magnitude'}
    write(output/'training_clip_audit.json',clean(rows))


def select_ablation_targets(summary,top_k):
    local=summary['checkpoints']['p03']['4.0']['local_sampled']
    phase=[(key,row) for key,row in local.items() if key.endswith('/phase')]
    ranked=sorted(phase,key=lambda item:item[1].get('nmse_mean_per_call',0),reverse=True)
    choices=[key for key,_ in ranked[:top_k]]
    choices.append('_global/final_rmsnorm/phase')
    site5=next(key for key,_ in ranked if '/site_05_' in key)
    choices.append(site5)
    choices=list(dict.fromkeys(choices))
    clips=[key[:-5]+'clip' for key in choices if any(k == key[:-5]+'clip' or k.startswith(key[:-5]+'clip/role_') for k in local)]
    return choices+clips


def ablations(output):
    spec=read(output/'specification.json');summary=summarize(output)
    if summary['checkpoints'].get('p03',{}).get('4.0',{}).get('samples')!=spec['samples']:
        raise ValueError('p03 diagnostics must complete before ranking bypasses')
    targets=select_ablation_targets(summary,spec['top_k'])
    write(output/'ablation_targets.json',{'diagnostic_only':True,'selection':'top local sampled NMSE + final RMSNorm + worst Site5, plus Clip at those positions when present','targets':targets})
    samples=read(output/'validation_manifest.json')['samples']
    model,controller,probe,cfg=build_model(output,'p03')
    warmup(model,controller,probe,batch_from(samples[0],model))
    fingerprint=read(output/'run_fingerprint.json')['sha256']
    for row in samples:
        dest=output/'ablations'/f"sample_{row['index']:05d}.json"
        if dest.exists():
            existing=read(dest)
            if existing['fingerprint']!=fingerprint or existing['targets']!=targets:raise ValueError('Stale ablation record')
            continue
        batch=batch_from(row,model);probe.reference_hidden=None;probe.gradient_reference=None
        probe.bypass=None
        identity,identity_logits=run_pass(model,controller,probe,batch,'identity',4.,collect=False)
        probe.reference_hidden=None
        baseline,base_logits=run_pass(model,controller,probe,batch,'phase',4.,collect=False)
        results={}
        for target in targets:
            probe.bypass=target
            result,logits=run_pass(model,controller,probe,batch,'phase',4.,collect=False)
            results[target]={'completion_tokens':result['completion_tokens'],'loss':result['loss'],
                'loss_delta_vs_phase':result['loss']-baseline['loss'],
                'against_identity':logit_difference(identity_logits,logits)}
        write(dest,dict(fingerprint=fingerprint,index=row['index'],targets=targets,baseline_loss=baseline['loss'],
            baseline_against_identity=logit_difference(identity_logits,base_logits),results=results,diagnostic_only=True))
        print('ablation validation_index=',row['index'],flush=True)
    probe.close()
    summarize_ablations(output)


def summarize_ablations(output):
    rows=[read(p) for p in sorted((output/'ablations').glob('sample_*.json'))]
    result={}
    for row in rows:
        for key,v in row['results'].items():
            dest=result.setdefault(key,{'samples':0,'completion_tokens':0,'weighted_loss_delta':0.,'kl_delta_sum':0.})
            dest['samples']+=1;dest['completion_tokens']+=v['completion_tokens']
            dest['weighted_loss_delta']+=v['loss_delta_vs_phase']*v['completion_tokens']
            dest['kl_delta_sum']+=v['against_identity']['kl_identity_to_probe']-row['baseline_against_identity']['kl_identity_to_probe']
    for v in result.values():
        v['loss_delta_vs_phase']=v['weighted_loss_delta']/v['completion_tokens']
        v['mean_kl_delta_vs_phase']=v['kl_delta_sum']/v['samples']
    write(output/'ablation_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--formal-dir',type=Path,default=Path('artifacts/phase_aware_formal_v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/phase_diagnostics_v1/calibration_group_size_4'))
    p.add_argument('--samples',type=int,default=128);p.add_argument('--smoke-samples',type=int,default=16)
    p.add_argument('--slopes',type=float,nargs='+',default=[.5,1.,2.,4.,8.])
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--worker',choices=['untrained_rotated','p01','p03','ablations'])
    p.add_argument('--limit',type=int)
    args=p.parse_args();os.chdir(ROOT);output=args.output.resolve()
    if args.summarize_only:
        with lock(output/'diagnostics.lock'):summarize(output);summarize_ablations(output)
        return
    if args.worker:
        if args.worker=='ablations':ablations(output)
        else:worker(output,args.worker,args.limit)
        return
    paths=config_paths(args.formal_dir);validate_configs(paths)
    if not 0<args.smoke_samples<=args.samples or args.top_k<=0:raise ValueError('Invalid sample/top-k limits')
    if 4. not in args.slopes or any(not math.isfinite(s) or s<=0 for s in args.slopes) or len(set(args.slopes))!=len(args.slopes):raise ValueError('Unique positive finite slopes including 4 are required')
    print(f'Diagnostic plan: validation {args.smoke_samples} -> {args.samples}; smoke_only={args.smoke_only}; untrained rotated, p01, p03; p03 slopes',args.slopes)
    print('No optimizer, no checkpoint writes; only validation diagnostics and diagnostic-only bypasses.')
    if args.dry_run:
        print('Configs:',paths);print('Output:',output);return
    if torch.cuda.device_count()!=4:raise ValueError('Expose exactly four GPUs for balanced model-parallel diagnostics')
    with lock(output/'diagnostics.lock'):
        prepare(output,paths,args)
        executor=Executor(output)
        def stop(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,stop)
        limits=[args.smoke_samples] if args.smoke_only else list(dict.fromkeys([args.smoke_samples,args.samples]))
        tasks=[(tag,limit) for limit in limits for tag in paths]
        if not args.smoke_only:tasks.append(('ablations',args.samples))
        try:
            for tag,limit in tasks:
                task=f'{tag}_{limit}';state=output/'status'/f'{task}.json'
                write(state,{'status':'running','tag':tag,'samples':limit})
                try:
                    executor.command(task,'diagnostic',[sys.executable,str(Path(__file__).resolve()),
                        '--output',str(output),'--worker',tag,'--limit',str(limit)])
                except BaseException as exc:
                    write(state,{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','error':str(exc)})
                    raise
                write(state,{'status':'completed','tag':tag,'samples':limit})
                summarize(output)
        finally:
            executor.stop();summarize(output);summarize_ablations(output)


if __name__=='__main__':main()
