"""CPU-only tests of diagnostic math and intervention hooks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import pytest
from snn2.phase_diagnostics import sample, aligned_parameter, error_metrics, completion_loss, logit_difference, Probe


def test_sampled_broadcast_parameter_matches_expanded():
    x=torch.randn(2,4,8,16)
    tau=torch.rand(1,4,1,16)
    assert torch.equal(aligned_parameter(x,tau,17),sample(tau.expand_as(x),17))


def test_errors_and_logits_identity():
    x=torch.tensor([1.,2.,0.,-3.])
    result=error_metrics(x,x)
    assert result['relative_l2']==0 and result['cosine']==pytest.approx(1)
    result=error_metrics(x,torch.zeros_like(x))
    assert result['nonzero_to_zero_fraction']==1 and result['relative_l2']==1
    logits=torch.randn(3,7)
    assert logit_difference(logits,logits)['kl_identity_to_probe']==0


def test_completion_loss_ignores_prompt_and_shifts():
    logits=torch.randn(1,7,11,requires_grad=True)
    labels=torch.tensor([[-100,-100,-100,3,4,5,6]])
    loss,indices=completion_loss(logits,labels)
    reference=torch.nn.functional.cross_entropy(logits[0,2:6],labels[0,3:])
    assert torch.allclose(loss,reference)
    assert indices.tolist()==[2,3,4,5]
    loss.backward()
    assert torch.count_nonzero(logits.grad[0,:2])==0
    with pytest.raises(ValueError,match='completion'):
        completion_loss(logits,torch.full_like(labels,-100))


def test_hook_preserves_output_and_gradient_until_explicit_bypass():
    module=torch.nn.Identity()
    probe=Probe(SimpleNamespace(),1)
    probe.attach_operator('layer_000/site_01/clip',module)
    x=torch.randn(3,requires_grad=True)
    probe.enabled=True;probe.backward=True
    y=module(x);y.sum().backward()
    assert torch.equal(y,x) and torch.equal(x.grad,torch.ones_like(x))
    assert probe.local.data and probe.gradients.data
    probe.close()
    class Double(torch.nn.Module):
        def forward(self,x):return x*2
    module=Double();probe=Probe(SimpleNamespace(),0)
    probe.attach_operator('test/phase',module)
    assert torch.equal(module(x),x*2)
    probe.bypass='test/phase'
    assert torch.equal(module(x),x)
    probe.close()


def test_attention_excludes_masked_zeros_and_preserves_prefix_mass():
    probe=Probe(SimpleNamespace(),1)
    probe.values['layer_000']=torch.tensor([[[1.,0.],[0.,1.]]])
    x=torch.tensor([[[[1.,0.],[.5,.5]]]])
    y=torch.tensor([[[[.75,0.],[.5,.5]]]])
    probe.soft_attention('layer_000/site_05/phase',x,y)
    result=probe.attention.data['layer_000/site_05/phase/head_00']
    assert result['row_sum_before_sum']==1
    assert result['row_sum_after_sum']==.875
    assert result['prefix_mass_before_sum']==.75
    assert result['active_entry_zeroed_fraction_sum']==0
    assert result['pv_relative_l2_sum']>0


def test_head_sampling_uses_logical_query_heads():
    from snn2.phase_diagnostics import heads
    x=torch.arange(48.).reshape(1,3,16)
    module=SimpleNamespace(layout={'num_heads':4})
    output=heads(x,module,3)
    assert len(output)==4
    assert torch.equal(output[2][1],x.reshape(1,3,4,4)[...,2,:])


def runner_module():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('diagnostic_runner',root/'scripts/diagnose_phase_aware.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_target_selection_includes_final_site5_and_role_clip():
    runner=runner_module()
    data={'layer_000/site_01_post_input_rmsnorm/phase':{'nmse_mean_per_call':2},
          'layer_000/site_01_post_input_rmsnorm/clip/role_q':{'nmse_mean_per_call':1},
          'layer_000/site_05_post_spiking_softmax/phase':{'nmse_mean_per_call':.8},
          '_global/final_rmsnorm/phase':{'nmse_mean_per_call':.1}}
    summary={'checkpoints':{'p03':{'4.0':{'local_sampled':data}}}}
    selected=runner.select_ablation_targets(summary,1)
    assert '_global/final_rmsnorm/phase' in selected
    assert 'layer_000/site_05_post_spiking_softmax/phase' in selected
    assert 'layer_000/site_01_post_input_rmsnorm/clip' in selected
    assert not any('site_05' in key and key.endswith('/clip') for key in selected)


def test_aggregation_weights_calls_and_preserves_max():
    runner=runner_module()
    row={'a':{'calls':1,'error2_sum':2.,'error2_max':2.,'error2_observations':1}}
    output=runner.merge_accumulators([row,row])['a']
    assert output['calls']==2 and output['error2_sum']==4
    assert output['error2_mean_per_call']==2 and output['error2_max']==2


def test_clip_gradient_records_blocked_values():
    class Clip(torch.nn.Module):
        def forward(self,x):return x.clamp(-1,1)
    module=Clip();probe=Probe(SimpleNamespace(),0)
    probe.attach_operator('test/clip',module)
    probe.enabled=probe.backward=True
    x=torch.tensor([-2.,.5,2.],requires_grad=True)
    module(x).sum().backward()
    row=probe.gradients.data['test/clip/input']
    assert row['clip_outside_count_sum']==2
    assert row['clip_outside_zero_gradient_fraction_sum']==1
    probe.close()


def test_real_phase_hook_does_not_change_forward_or_backward():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('neuron_test_fixture',root/'tests/test_neurons.py')
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    from snn2.neurons import PhaseSurrogate
    module=PhaseSurrogate(fixtures._phase_state(),T=4,surrogate_slope=4.)
    x=torch.tensor([[[.1,.3,.7,1.9]]],requires_grad=True)
    original=module(x);original.sum().backward();gradient=x.grad.clone();x.grad=None
    probe=Probe(SimpleNamespace(),0);probe.enabled=probe.backward=True
    probe.attach_operator('layer_000/site_01_post_input_rmsnorm/phase',module)
    observed=module(x);observed.sum().backward()
    assert torch.equal(original,observed) and torch.equal(gradient,x.grad)
    for slope in [.5,1.,2.,8.]:
        module.slope=slope
        assert torch.equal(module(x),original)
    probe.close()


def test_empty_summary_is_safe(tmp_path):
    runner=runner_module()
    summary=runner.summarize(tmp_path)
    assert summary['checkpoints']=={}
    runner.summarize_ablations(tmp_path)
    assert (tmp_path/'summary.md').exists()


def test_tiny_model_worker_and_sample_resume(tmp_path,monkeypatch):
    runner=runner_module()
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('tiny_phase_fixture',root/'tests/test_neurons.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from snn2.neurons import PhaseSurrogate
    controller=SimpleNamespace(mode='phase',phase_surrogate_slope=4.,
        _modules={'layer_000/site_01_post_input_rmsnorm':{'phase':PhaseSurrogate(fixture._phase_state(),T=4,surrogate_slope=4.)}},
        _final_norm_phase=PhaseSurrogate(fixture._phase_state(),T=4,surrogate_slope=4.))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=torch.nn.Module()
            self.model.embed_tokens=torch.nn.Embedding(11,4)
            self.model.layers=torch.nn.ModuleList([torch.nn.Linear(4,4)])
            self.model.norm=torch.nn.Identity()
            self.lm_head=torch.nn.Linear(4,11)
            self.model.layers[0].register_forward_hook(lambda m,a,y:controller._modules['layer_000/site_01_post_input_rmsnorm']['phase'](y) if controller.mode=='phase' else y)
            self.model.norm.register_forward_hook(lambda m,a,y:controller._final_norm_phase(y) if controller.mode=='phase' else y)
        def forward(self,input_ids,attention_mask,use_cache):
            x=self.model.embed_tokens(input_ids)
            for layer in self.model.layers:x=layer(x)
            return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))
    model=Tiny()
    calls=[]
    def build(*args):
        calls.append(1)
        return model,controller,Probe(controller,0),{'phase':{'surrogate_slope':4.}}
    monkeypatch.setattr(runner,'build_model',build)
    runner.write(tmp_path/'specification.json',{'samples':2,'slopes':[.5,4.]})
    runner.write(tmp_path/'run_fingerprint.json',{'sha256':'test'})
    runner.write(tmp_path/'validation_manifest.json',{'samples':[
        {'index':i,'encoded':{'input_ids':[1,2,3,4,5],'labels':[-100,-100,3,4,5],'attention_mask':[1]*5}}
        for i in [7,19]]})
    runner.worker(tmp_path,'p03',2)
    runner.worker(tmp_path,'p03',2)
    assert len(calls)==1
    result=runner.read(tmp_path/'checkpoints/p03/sample_00007.json')
    assert result['phase']['4.0']['slope_forward_invariance']['max_abs_sampled_logit_difference']==0
    assert result['identity']['gradient_statistics']
    monkeypatch.setattr(runner,'audit_training_clip',lambda output:None)
    summary=runner.summarize(tmp_path)
    assert summary['checkpoints']['p03']['4.0']['samples']==2
    assert (tmp_path/'gradient_statistics.csv').exists()
