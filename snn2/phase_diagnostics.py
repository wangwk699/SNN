"""Read-only, explicitly diagnostic probes; never installed by training/evaluation."""
from __future__ import annotations
import math
import re
from collections import defaultdict
import torch
import torch.nn.functional as F
from .neurons import PhaseSurrogate, _parameter_values


def sample(x, limit=4096):
    """Evenly spaced flattened coordinates, deterministic for a given shape."""
    flat = x.detach().reshape(-1)
    if not flat.numel():
        return flat.float()
    stride = max(1, math.ceil(flat.numel() / limit))
    return flat[::stride].float()


def aligned_parameter(x, parameter, limit=4096):
    """Sample a broadcast parameter without materializing its expanded tensor."""
    stride = max(1, math.ceil(x.numel() / limit))
    indices = torch.arange(0, x.numel(), stride, device=x.device)
    coords = torch.unravel_index(indices, x.shape)
    p = parameter.reshape((1,) * (x.ndim - parameter.ndim) + parameter.shape)
    return p[tuple(torch.zeros_like(c) if d == 1 else c for c, d in zip(coords, p.shape))].float()


def moments(x):
    x = x.detach().float()
    finite = torch.isfinite(x)
    good = x[finite]
    result = {'count': x.numel(), 'nonfinite': int((~finite).sum())}
    if good.numel():
        q = torch.quantile(good.abs(), torch.tensor([.05, .5, .95], device=good.device))
        result.update(sum=float(good.sum()), sum_sq=float(good.square().sum()),
            zero=int((good == 0).sum()), abs_max=float(good.abs().max()),
            abs_p05=float(q[0]), abs_p50=float(q[1]), abs_p95=float(q[2]))
    return result


def error_metrics(x, y):
    a, b = x.detach().float(), y.detach().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask], b[mask]
    if not a.numel():
        return {'count': 0, 'nonfinite': x.numel()}
    diff = b-a
    xx, yy = float(a.square().sum()), float(b.square().sum())
    ee = float(diff.square().sum())
    return dict(count=a.numel(), nonfinite=x.numel()-a.numel(), x2=xx, y2=yy,
        error2=ee, dot=float((a*b).sum()), signed_error=float(diff.sum()),
        relative_l2=math.sqrt(ee/max(xx, 1e-30)), nmse=ee/max(xx, 1e-30),
        cosine=float((a*b).sum())/max(math.sqrt(xx*yy), 1e-30),
        zero_output_fraction=float((b == 0).float().mean()),
        nonzero_to_zero_fraction=float(((a != 0) & (b == 0)).sum())/max(int((a != 0).sum()), 1),
        changed_fraction=float((a != b).float().mean()))


class Accumulator:
    def __init__(self):
        self.data = {}

    def add(self, key, values):
        row = self.data.setdefault(key, {'calls': 0})
        row['calls'] += 1
        for name, value in values.items():
            if isinstance(value, (float, int)) and math.isfinite(value):
                row[name + '_sum'] = row.get(name + '_sum', 0.) + value
                row[name + '_max'] = max(row.get(name + '_max', -math.inf), value)
                row[name + '_observations'] = row.get(name + '_observations', 0) + 1

    def export(self):
        return self.data


def heads(x, module, site):
    layout = getattr(module, 'layout', {})
    h = layout.get('num_heads')
    if site not in (2, 3, 4, 5) or not h:
        return []
    if x.ndim == 4:
        return [(i, x[:, i]) for i in range(h)]
    if x.ndim == 3:
        shaped = x.reshape(*x.shape[:-1], h, -1)
        return [(i, shaped[..., i, :]) for i in range(h)]
    raise ValueError(f'Unexpected head layout at site {site}: {x.shape}')


class Probe:
    """Module hooks preserve Phase-then-Clip order; bypasses are diagnostic only."""
    def __init__(self, controller, prefix_length):
        self.controller = controller
        self.prefix_length = prefix_length
        self.enabled = False
        self.backward = False
        self.bypass = None
        self.local = Accumulator()
        self.gradients = Accumulator()
        self.attention = Accumulator()
        self.handles = []
        self.hidden = {}
        self.reference_hidden = None
        self.cumulative = Accumulator()
        self.gradient_reference = None
        self.parameter_samples = {}
        self.global_gradient2 = 0.
        self.global_nonfinite = 0
        self.values = {}

    def reset(self):
        self.local, self.gradients, self.attention = Accumulator(), Accumulator(), Accumulator()
        self.cumulative = Accumulator()
        self.hidden, self.parameter_samples = {}, {}
        self.values = {}
        self.global_gradient2, self.global_nonfinite = 0., 0

    def attach(self, model):
        # A warm-up forward has materialized all lazy controller modules.
        for key, modules in self.controller._modules.items():
            for kind in ('phase', 'clip'):
                if kind in modules:
                    self.attach_operator(key + '/' + kind, modules[kind])
        if self.controller._final_norm_phase is None:
            raise RuntimeError('Warmup did not initialize global Final RMSNorm Phase')
        self.attach_operator('_global/final_rmsnorm/phase', self.controller._final_norm_phase)
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_hook(self.layer_hook(f'layer_{index:03d}')))
        self.handles.append(model.model.norm.register_forward_hook(self.layer_hook('final_norm')))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                weight_norm = float(torch.linalg.vector_norm(parameter.detach(), dtype=torch.float32))
                self.handles.append(parameter.register_hook(self.parameter_hook(name, weight_norm)))

    def attach_operator(self, key, module):
        def hook(mod, args, kwargs, output):
            x = args[0]
            role = kwargs.get('role')
            tag = key + (f'/role_{role}' if role else '')
            effective = x if (self.bypass == key or isinstance(self.bypass, (set, frozenset)) and key in self.bypass) else output
            if self.enabled and '/site_04_' in key:
                self.values[key.split('/')[0]] = effective.detach()
            if self.enabled:
                a, b = sample(x), sample(output)
                result = error_metrics(a, b)
                if isinstance(mod, PhaseSurrogate):
                    tau = aligned_parameter(x, _parameter_values(x, mod.tau, mod.layout))
                    ratio = a.abs()/tau
                    cap = tau*(1-2**(-mod.T))
                    result.update(input_over_range_fraction=float((a.abs()>cap).float().mean()),
                        saturation_fraction=float(torch.isclose(b.abs(), cap, rtol=.01, atol=1e-7).float().mean()),
                        abs_over_tau_p50=float(ratio.median()), abs_over_tau_p95=float(torch.quantile(ratio, .95)))
                    match = re.search(r'site_(\d+)', key)
                    site = int(match[1]) if match else None
                    for (h, hx), (_, hy) in zip(heads(x, mod, site), heads(output, mod, site)):
                        self.local.add(f'{tag}/head_{h:02d}', error_metrics(sample(hx, 256), sample(hy, 256)))
                    if site == 5:
                        self.soft_attention(tag, x, output)
                self.local.add(tag, result)
            if self.backward:
                for label, tensor in [('input', x), ('output', effective)]:
                    if tensor.requires_grad:
                        if label == 'input' and not isinstance(mod, PhaseSurrogate):
                            outside = (sample(x) != sample(output)).detach()
                            tensor.register_hook(lambda grad, k=tag+'/input', mask=outside: self.clip_gradient(k, grad, mask))
                        else:
                            tensor.register_hook(lambda grad, k=tag+'/'+label: self.gradient_tensor(k, grad))
            return effective
        self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def soft_attention(self, tag, x, y):
        # Sample query rows only. Keep ALL key columns to preserve row sums.
        stride = max(1, math.ceil(x.shape[-2]/32))
        query_indices = getattr(self, 'query_indices', None)
        if query_indices is None:
            a, b = x.detach()[..., ::stride, :].float(), y.detach()[..., ::stride, :].float()
        else:
            selected = query_indices[::max(1, math.ceil(len(query_indices)/32))].to(x.device)
            a, b = x.detach().index_select(-2, selected).float(), y.detach().index_select(-2, selected).float()
        value = self.values.pop(tag.split('/')[0], None)
        if value is not None:
            value = value.reshape(value.shape[0], value.shape[1], a.shape[1], -1).transpose(1, 2).float()
        for h in range(a.shape[1]):
            u, v = a[:, h], b[:, h]
            active = u > 0  # excludes causal-mask zeros; no runtime mask is changed
            p = self.prefix_length
            values = dict(row_sum_before=float(u.sum(-1).mean()), row_sum_after=float(v.sum(-1).mean()),
                active_entry_zeroed_fraction=float(((v == 0) & active).sum())/max(int(active.sum()), 1),
                zero_row_fraction=float((v.sum(-1) == 0).float().mean()),
                prefix_mass_before=float(u[..., :p].sum(-1).mean()),
                prefix_mass_after=float(v[..., :p].sum(-1).mean()),
                text_mass_before=float(u[..., p:].sum(-1).mean()),
                text_mass_after=float(v[..., p:].sum(-1).mean()))
            if value is not None:
                pv_before = u @ value[:, h]
                pv_after = v @ value[:, h]
                values.update({'pv_'+k: val for k, val in error_metrics(pv_before, pv_after).items()})
            self.attention.add(f'{tag}/head_{h:02d}', values)

    def layer_hook(self, key):
        def hook(module, args, output):
            if not self.enabled:
                return
            y = output[0] if isinstance(output, tuple) else output
            current = sample(y, 8192).cpu()
            self.hidden[key] = current
            if self.reference_hidden is not None:
                self.cumulative.add(key, error_metrics(self.reference_hidden[key], current))
        return hook

    def gradient_tensor(self, key, grad):
        values = moments(sample(grad))
        self.gradients.add(key, values)
        return grad

    def clip_gradient(self, key, grad, outside):
        values = moments(sample(grad))
        values['clip_outside_count'] = int(outside.sum())
        if outside.any():
            values['clip_outside_zero_gradient_fraction'] = float((sample(grad)[outside] == 0).float().mean())
        self.gradients.add(key, values)
        return grad

    def parameter_hook(self, name, weight_norm):
        def hook(grad):
            if not self.backward:
                return grad
            g = sample(grad, 512).cpu()
            norm = float(torch.linalg.vector_norm(grad.detach(), dtype=torch.float32))
            self.global_gradient2 += norm**2
            nonfinite = int((~torch.isfinite(g)).sum())
            self.global_nonfinite += nonfinite
            values = moments(g)
            values.update(exact_norm=norm, exact_norm_squared=norm**2, weight_norm_squared=weight_norm**2, gradient_weight_ratio=norm/max(weight_norm, 1e-30))
            if self.gradient_reference is not None:
                ref = self.gradient_reference.get(name)
                values['identity_gradient_missing'] = int(ref is None)
                if ref is not None:
                    values['identity_gradient_cosine_sampled'] = float(torch.dot(g, ref))/max(float(g.norm()*ref.norm()),1e-30)
            self.gradients.add('parameter/'+name, values)
            self.parameter_samples[name] = g
            return grad
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def completion_loss(logits, labels):
    labels = labels[:, 1:].to(logits.device).reshape(-1)
    indices = (labels != -100).nonzero().flatten()
    if not indices.numel():
        raise ValueError('No completion labels after truncation')
    pred = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    total = None
    for positions in indices.split(32):
        loss = F.cross_entropy(pred[positions].float(), labels[positions], reduction='sum')
        total = loss if total is None else total+loss
    return total/indices.numel(), indices


def logit_sample(logits, indices):
    positions = indices[::max(1, math.ceil(indices.numel()/32))]
    return logits[:, :-1].reshape(-1, logits.shape[-1])[positions].detach().float().cpu()


def logit_difference(reference, actual):
    p, q = reference.float().log_softmax(-1), actual.float().log_softmax(-1)
    return dict(kl_identity_to_probe=float((p.exp()*(p-q)).sum(-1).mean()),
        top1_agreement=float((p.argmax(-1)==q.argmax(-1)).float().mean()),
        sampled_logits_relative_l2=float((actual-reference).norm()/reference.norm().clamp_min(1e-30)))
