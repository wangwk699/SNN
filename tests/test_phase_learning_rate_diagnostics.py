import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_phase_learning_rate as lr


def test_blocks_cover_layers():
    controller=SimpleNamespace(_modules={f'layer_{i:03d}/site_05_post_spiking_softmax':{'phase':object()} for i in range(36)})
    parts=[lr.targets(controller,k) for k in ['early','middle','late']]
    assert all(len(x)==12 for x in parts)
    assert not parts[0]&parts[1]
    assert set.union(*parts)==lr.targets(controller,'all')


def test_paired_summary(tmp_path):
    for tag,loss in [('p01',2.),('p03',3.)]:
        row={'index':4,'phase':{'loss':loss,'completion_tokens':10,'local_sampled':{},'attention_sampled_rows':{},'cumulative_hidden_sampled':{}},
             'identity':{'loss':1.},'bypasses':{k:{'delta_vs_phase':-.1} for k in lr.BLOCKS}}
        lr.write(tmp_path/'records'/tag/'sample_00004.json',row)
    lr.summarize(tmp_path)
    result=lr.read(tmp_path/'comparison.json')
    assert result['paired']['phase']['lr3_minus_lr2_nll']==1.
    assert result['checkpoints']['lr2e-6']['bypasses']['all']['weighted_nll_delta']==pytest.approx(-.1)
