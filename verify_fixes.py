"""Run numerical regressions without measured datasets.

python verify_fixes.py --device all --json verification.json
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import exp1
import torch


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'all'], default='auto')
parser.add_argument('--json', type=Path, help='Optional machine-readable results')
args = parser.parse_args()
if args.device in ('cpu', 'cuda'):
    os.environ['PROPBR_DEVICE'] = args.device
torch.set_num_threads(min(4, torch.get_num_threads()))
import func
from prosailt.prospect_dx import _stack_layers, prospect_dt
from prosailt.special import e1
from prospect import prospect_d

if args.device == 'cuda' and not torch.cuda.is_available():
    parser.error('CUDA was requested but is unavailable')
devices = (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']) if args.device == 'all' else [
    func._DEFAULT_DEVICE if args.device == 'auto' else args.device]
records = []
P8 = [1.8, 48., 12., 3., .017, .009, .12, .28]
LOW = np.array([1., 0, 0, 0, 0, 0, 0, .01], dtype=np.float32)
HIGH = np.array([3.5, 120, 30, 40, .06, .05, 1., 1.], dtype=np.float32)


def run(name, test, device):
    try:
        detail = test(device)
        record = dict(name=name, device=device, passed=True, detail=detail)
    except Exception as exc:
        record = dict(name=name, device=device, passed=False,
                      error=type(exc).__name__ + ': ' + str(exc))
    records.append(record)
    print(('PASS' if record['passed'] else 'FAIL') + ' [' + device + '] ' + name +
          ' ' + json.dumps(record.get('detail', record.get('error')), ensure_ascii=True), flush=True)


def integral(device):
    x = torch.logspace(-10, 2, 70, device=device, dtype=torch.float64)
    expected = torch.from_numpy(exp1(x.cpu().numpy())).to(device)
    value_error = ((e1(x) - expected).abs() / expected.abs()).max().item()
    jf = torch.func.jacfwd(e1)(x).diagonal()
    q = x.clone().requires_grad_(True)
    jr = torch.autograd.grad(e1(q).sum(), q)[0]
    derivative = -torch.exp(-x) / x
    derivative_error = max(((jf - derivative).abs() / derivative.abs()).max().item(),
                           ((jr - derivative).abs() / derivative.abs()).max().item())
    assert value_error < 1e-9 and derivative_error < 1e-9
    return dict(value_relative_error=value_error, derivative_relative_error=derivative_error)


def stack_identities(device):
    # Independent two-plate adding equations, including a lossless plate.
    r = torch.tensor([.2, .12, .05], dtype=torch.float64, device=device, requires_grad=True)
    t = torch.tensor([.8, .3, .1], dtype=torch.float64, device=device, requires_grad=True)
    for n in [0., 1., 2.]:
        rr, tt = _stack_layers(r, t, torch.full_like(r, n))
        if n == 0:
            er, et = torch.zeros_like(r), torch.ones_like(t)
        elif n == 1:
            er, et = r, t
        else:
            er = r + t.square() * r / (1 - r.square())
            et = t.square() / (1 - r.square())
        torch.testing.assert_close(rr, er, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(tt, et, atol=1e-12, rtol=1e-12)
        if n:
            actual = torch.autograd.grad((rr + tt).sum(), (r, t), retain_graph=True)
            expected = torch.autograd.grad((er + et).sum(), (r, t), retain_graph=True)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    return 'zero/one/two layers: values and lossless derivatives agree'


def numpy_reference(device):
    rng = np.random.default_rng(42)
    lo = np.array([1.2, 10, 2, .2, .005, .002, .01])
    hi = np.array([2.8, 80, 20, 8, .03, .02, .4])
    rows = lo + rng.random((12, 7)) * (hi - lo)
    parameters = torch.as_tensor(rows, dtype=torch.float64, device=device)
    actual = prospect_dt(*[parameters[:, i:i+1] for i in range(7)], device=device)[0]
    expected = np.stack([np.asarray(prospect_d(*row)[0]) for row in rows])
    error = float(np.abs(actual.detach().cpu().numpy() - expected).max())
    assert np.isfinite(expected).all() and error < 2e-5
    return dict(leaves=len(rows), max_abs_brf_difference=error)


def gradients(device):
    p = torch.tensor([P8 + [.9]], device=device)
    details = {}
    for forward, width in [(func.kan_torch, 8), (func.kan_torch_lF, 9),
                           (func.full_torch, 9), (func.spropbr_torch, 8)]:
        q = p[0, :width].detach()
        def one(v):
            return forward(v.unsqueeze(0))[0, [100, 300, 600, 1000, 1600]]
        jf = torch.func.jacfwd(one)(q)
        jr = torch.autograd.functional.jacobian(one, q)
        rel = ((jf - jr).norm(dim=0) / jr.norm(dim=0).clamp_min(1e-10)).max().item()
        assert torch.isfinite(jf).all() and torch.isfinite(jr).all() and rel < 1e-4
        details[forward.__name__] = rel
    return details


def zero_absorption(device):
    x = torch.tensor([1.8, 0, 0, 0, 0, 0, 0, .28], device=device, dtype=torch.float64)
    direction = x.new_tensor([0, 40, 12, 3, .02, .01, .1, 0])
    details = {}
    for forward in [func.kan_torch, func.spropbr_torch]:
        q = x.clone().requires_grad_(True)
        y = forward(q.unsqueeze(0))
        g = torch.autograd.grad(y.sum(), q)[0]
        jf = torch.func.jacfwd(lambda v: forward(v.unsqueeze(0)).sum())(x)
        assert torch.isfinite(y).all() and torch.isfinite(g).all() and torch.isfinite(jf).all()
        torch.testing.assert_close(jf, g, rtol=1e-5, atol=1e-6)
        h = 1e-7
        fd = (forward((x + h * direction).unsqueeze(0)).sum() - y.detach().sum()) / h
        derivative = (g * direction).sum()
        relative = float(((fd - derivative).abs() / derivative.abs().clamp_min(1e-10)).detach())
        assert relative < 1e-3
        details[forward.__name__] = dict(directional_derivative=float(derivative),
                                       one_sided_relative_error=relative)
    return details


def bounds(device):
    rows = [[1., 0, 0, 0, 0, 0, 0, .01],
            [3.5, 0, 0, 0, 0, 0, 0, 1.],
            [1., 120, 30, 40, .06, .05, 1., .01],
            [3.5, 120, 30, 40, .06, .05, 1., 1.],
            [1.8, 1e-8, 0, 0, 1e-12, 1e-12, 0, .28]]
    for row in rows:
        p = torch.tensor([row], device=device, requires_grad=True)
        y = func.kan_torch(p)
        g = torch.autograd.grad(y.sum(), p)[0]
        assert torch.isfinite(y).all() and torch.isfinite(g).all(), str(row)
    return dict(boundary_cases=len(rows))


def surface(device):
    p = torch.tensor([P8 + [.9]], device=device)
    multi = func.full_torch_l(p).reshape(1, 24, 1201)
    single = func.full_torch(p)[:, 50:1251]
    torch.testing.assert_close(multi[:, 5], single, atol=1e-6, rtol=1e-6)
    diffuse = (func._prospect_terms(p) - func._reference_specular(p[:, 7:8])).detach().cpu().numpy()
    nr = np.load(ROOT / 'lut' / 'DLM_nr.npy')
    f0 = ((1 - nr) / (1 + nr)) ** 2
    sigma, fs = .28, .9
    max_error = 0.
    for i, deg in enumerate([10., 35., 65.]):
        angle = np.deg2rad(deg)
        G = 2 / (1 + np.sqrt(1 + np.tan(angle)**2 * sigma**2))
        D = sigma**2 / (np.pi * (np.cos(angle/2)**2 * (sigma**2 - 1) + 1)**2)
        schlick = f0 + (1 - f0) * (1 - np.cos(angle/2))**5
        expected = diffuse[:, 50:1251] + (np.pi * fs * G * D * schlick / (4 * np.cos(angle)))[50:1251]
        error = np.max(np.abs(multi[:, [0, 5, 11][i]].detach().cpu().numpy() - expected))
        max_error = max(max_error, float(error))
    assert max_error < 2e-6
    assert torch.all(multi - torch.as_tensor(diffuse[:, None, 50:1251], device=device) >= -1e-7)
    return dict(max_analytic_Schlick_difference=max_error, shape=list(multi.shape))


def model_interfaces(device):
    p = torch.tensor([P8 + [.6]], device=device)
    a = func.kan_torch_lF(p)
    q = p.clone(); q[:, 8] = 3.5
    b = func.kan_torch_lF(q)
    assert (a - b).abs().max() > .01
    q[:, 8] = (3.5 + .6) / 2
    torch.testing.assert_close(func.kan_torch_lF(q), (a + b) / 2, atol=1e-6, rtol=1e-6)
    assert func.Kf_torch is func.spropbr_torch
    x = p[:, :8].clone(); x[:, 7] = 1.2
    y = func.spropbr_torch(x)
    z = x.clone(); z[:, 7] += 1
    delta = func.spropbr_torch(z) - y
    expected = func._fresnel0(func._ensure(device)[0])
    torch.testing.assert_close(delta, expected, atol=1e-7, rtol=1e-5)
    return 'free fs is linear; canonical sPROPBR and legacy alias agree'


def best_iterate(device):
    target = torch.tensor([[.25], [.8]], device=device)
    with contextlib.redirect_stdout(io.StringIO()):
        result = func.invert_mse(target, np.array([.251]), [0.], [1.],
            forward_model=lambda p: p, n_iters=1, lr=.05, device=device, optimizer='adam')
        exact = func.invert_mse(target[:1], np.array([.25]), [0.], [1.],
            forward_model=lambda p: p, n_iters=1, lr=.05, device=device)
    initial_loss = (np.array([[.251], [.251]]) - target.cpu().numpy()) ** 2
    actual_loss = (result - target.cpu().numpy()) ** 2
    assert np.all(actual_loss <= initial_loss + 1e-10)
    assert actual_loss[1, 0] < initial_loss[1, 0]  # Includes the last optimizer update.
    assert abs(float(exact[0, 0]) - .25) < 1e-7
    return dict(initial_mse=initial_loss[:, 0].tolist(), returned_mse=actual_loss[:, 0].tolist(),
                returned_parameters=result[:, 0].tolist())


def learning_rate_and_objective(device):
    weights = torch.tensor([[.001, .003, -.002]], device=device)
    target = torch.tensor([[.001, .002, -.001]], device=device)
    trace = []
    results = []
    with contextlib.redirect_stdout(io.StringIO()):
        for mode in ['backward', 'jacfwd']:
            results.append(func.invert_mse(target, np.array([.3]), [0.], [1.],
                forward_model=lambda p: p * weights, n_iters=30, lr=.05, device=device,
                optimizer='adam', grad_mode=mode, lr_trace=trace if mode == 'backward' else None))
    assert len(trace) == 30 and all(b <= a + 1e-12 for a, b in zip(trace, trace[1:]))
    assert abs(trace[-1] - .0025) < 1e-12
    np.testing.assert_allclose(results[0], results[1], atol=2e-6, rtol=2e-6)
    return dict(post_step_first_lr=trace[0], final_lr=trace[-1],
                gradient_mode_parameter_difference=float(np.abs(results[0] - results[1]).max()))


def eight_parameter_propbr(device):
    x = torch.tensor(P8, device=device)
    results = {}
    for forward, bands in [(func.propbr_35_torch, 2101), (func.propbr_24_torch, 24 * 1201)]:
        value = forward(x.unsqueeze(0))
        assert value.shape == (1, bands) and torch.isfinite(value).all()
        indices = [100, 500, 1000, 1600] if bands == 2101 else [100, 4000, 14000, 28000]
        def selected(p):
            return forward(p.unsqueeze(0))[0, indices]
        jf = torch.func.jacfwd(selected)(x)
        jr = torch.autograd.functional.jacobian(selected, x)
        torch.testing.assert_close(jf, jr, atol=1e-5, rtol=1e-4)
        step = torch.zeros_like(x); step[7] = 1e-4
        finite_difference = (selected(x + step) - selected(x - step)) / 2e-4
        relative_error = ((finite_difference - jr[:, 7]).norm() /
                          jr[:, 7].norm().clamp_min(1e-10)).item()
        assert relative_error < .01
        try:
            forward(torch.cat([x, x.new_tensor([1.])]).unsqueeze(0))
        except ValueError:
            pass
        else:
            raise AssertionError('The public PROPBR entry point must take exactly 8 parameters')
        results[forward.__name__] = dict(shape=list(value.shape), sigma_derivative_relative_error=relative_error)
    # The 35-degree direction uses the same sigma-dependent law in both paths.
    q = x.unsqueeze(0)
    analytic = func.propbr_24_torch(q).reshape(1, 24, 1201)[:, 5]
    surrogate = func.propbr_35_torch(q)[:, 50:1251]
    assert (analytic - surrogate).abs().max() < .003
    return results


def synthetic_inversion(device):
    p = torch.tensor([P8], device=device)
    configs = [(func.propbr_35_torch, p, LOW, HIGH),
               (func.spropbr_torch, torch.cat([p[:, :7], p.new_tensor([[1.2]])], dim=1),
                np.r_[LOW[:7], 0.], np.r_[HIGH[:7], 30.]),
               (func.propbr_24_torch, p, LOW, HIGH)]
    out = {}
    for forward, truth, lo, hi in configs:
        initial = ((truth[0].cpu().numpy() - lo) / (hi - lo)) * .9 + .01
        obs = forward(truth).detach()
        initial_p = torch.as_tensor(lo + (hi - lo) * initial, dtype=torch.float32, device=device)[None]
        first = (forward(initial_p) - obs).square().mean().item()
        with contextlib.redirect_stdout(io.StringIO()):
            fit = func.invert_mse(obs, initial, lo, hi, forward_model=forward, n_iters=30,
                                  lr=.05, device=device, optimizer='adam')
        final = (forward(torch.as_tensor(fit, device=device)) - obs).square().mean().item()
        assert np.isfinite(fit).all() and final < first * .5
        out[forward.__name__] = dict(initial_mse=first, returned_mse=final)
    return out


tests = [('E1 values and derivatives', integral), ('Stokes stacking identities', stack_identities),
         ('PROSPECT vs NumPy reference', numpy_reference), ('forward/reverse AD', gradients),
         ('zero-absorption derivative vs one-sided difference', zero_absorption),
         ('parameter boundary gradients', bounds), ('multi-angle Schlick expression', surface),
         ('PROPBR/sPROPBR interfaces', model_interfaces), ('best iterate and final step', best_iterate),
         ('cosine schedule and MSE gradient normalization', learning_rate_and_objective),
         ('8-parameter PROPBR interfaces and sigma derivatives', eight_parameter_propbr),
         ('synthetic model inversions', synthetic_inversion)]
for device in devices:
    func.warmup(device)
    for name, test in tests:
        run(name, test, device)

# Verify resource paths by importing in a separate interpreter outside the repo.
def outside_import(device):
    code = 'import sys;sys.path.insert(0,' + repr(str(ROOT)) + ');import func,metric;print("ok")'
    env = dict(os.environ, PROPBR_DEVICE=device, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', code], cwd=ROOT.parent,
                            env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stderr
    return 'imports succeeded outside the repository'

run('module-relative resource paths', outside_import, devices[0])
report = dict(torch=torch.__version__, numpy=np.__version__, devices=devices, checks=records,
              passed=sum(r['passed'] for r in records), failed=sum(not r['passed'] for r in records))
if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n%d passed, %d failed' % (report['passed'], report['failed']), flush=True)
sys.exit(bool(report['failed']))
