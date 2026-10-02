import numpy as np
import pytest
import torch
from scipy.special import exp1 as scipy_exp1
from propbr import PROPBR, SPROPBR, invert, local_sensitivity, response_correlation
from propbr.prospect import exp1, prospect_d
from propbr.surface import reduced_ks, surface_factor, fresnel0, schlick

WAVES = np.arange(400., 2501., 25.)
PARAMS = [1.8, 48., 12., 3., .017, .009, .12, .28]


def test_exp1_values_and_derivatives():
    x = torch.tensor(np.r_[np.geomspace(1e-10, 100., 150), .999999, 1., 1.000001], dtype=torch.float64, requires_grad=True)
    y = exp1(x)
    np.testing.assert_allclose(y.detach(), scipy_exp1(x.detach().numpy()), rtol=3e-12, atol=2e-14)
    derivative = torch.autograd.grad(y.sum(), x)[0]
    torch.testing.assert_close(derivative, -torch.exp(-x) / x, rtol=3e-11, atol=2e-13)


def test_prospect_against_independent_numpy_reference():
    prosail = pytest.importorskip("prosail")
    model = PROPBR(kernel="analytic")
    cases = [PARAMS, [1., 0., 0., 0., 0., 0., 0., .01],
             [3.5, 120., 30., 40., .06, .05, 1., 1.],
             [2.5, 0., 0., 0., .005, .001, 0., .3]]
    p = model.tensor(cases)
    r, t, _ = prospect_d(p, model.spectra, model.tav40, model.tav90)
    for i, case in enumerate(cases):
        N, cab, car, anth, cw, cm, brown = case[:7]
        _, rn, tn = prosail.run_prospect(N, cab, car, brown, cw, cm, ant=anth, prospect_version="D")
        np.testing.assert_allclose(r[i], rn, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(t[i], tn, rtol=1e-9, atol=1e-10)
    assert torch.isfinite(r).all() and torch.isfinite(t).all()


def test_paper_components_and_no_double_surface_multiplier():
    m = PROPBR(kernel="analytic", wavelengths=WAVES)
    p = m.tensor([PARAMS])
    c = m.components(p)
    phase = m.phase_rad
    rough = p[:, 7:8]
    D = rough.square() / (torch.pi * (torch.cos(phase / 2.).square() * (rough.square() - 1.) + 1.).square())
    G = 2. / (1. + torch.sqrt(1. + rough.square() * torch.tan(phase).square()))
    expected = torch.pi * surface_factor(rough) * D * G / (4. * torch.cos(phase))
    torch.testing.assert_close(c["ks"], expected)
    torch.testing.assert_close(c["specular"], expected[..., None] * schlick(m.spectra.surface_nr, phase[:, None]))
    torch.testing.assert_close(c["brf"], c["dhrf"][:, None, :] - c["reference_specular"][:, None, :] + c["specular"])
    s = SPROPBR(wavelengths=WAVES)
    q = p.clone(); q[:, 7] = 1.2
    cs = s.components(q)
    torch.testing.assert_close(cs["brf"], (cs["dhrf"] - (1. - s.tav40) + 1.2 * fresnel0(s.spectra.surface_nr))[:, None, :])


@pytest.mark.parametrize("kernel", ["analytic", "kan"])
def test_jacobian_against_central_differences(kernel):
    m = PROPBR(phase_angles=[15., 35., 55.], kernel=kernel, wavelengths=WAVES)
    p = m.tensor([PARAMS])
    J = local_sensitivity(m, p)
    assert torch.isfinite(J["physical"]).all()
    lo, hi = m.space.bounds(p)
    delta = (hi - lo) * 1e-5
    for k in range(8):
        plus, minus = p.clone(), p.clone()
        plus[0, k] += delta[k]; minus[0, k] -= delta[k]
        fd = (m(plus) - m(minus)) / (2. * delta[k])
        torch.testing.assert_close(J["physical"][..., k], fd, rtol=5e-5, atol=2e-7)
    z = m.space.encode(p)
    from torch.func import jacfwd, vmap
    Jz = vmap(jacfwd(lambda u: m(m.space.decode(u).unsqueeze(0)).squeeze(0)))(z)
    torch.testing.assert_close(J["latent"], Jz, rtol=1e-9, atol=1e-10)
    corr = response_correlation(J["physical"])["correlation"]
    torch.testing.assert_close(corr, corr.transpose(-1, -2))
    torch.testing.assert_close(corr.diagonal(dim1=-2, dim2=-1), torch.ones((1, 8), dtype=m.dtype))


def test_full_geometry_equals_reduction_with_nadir_illumination():
    reduced = PROPBR(phase_angles=[15., 35., 55.], kernel="analytic", wavelengths=WAVES)
    full = PROPBR(geometry=[[0., 15., 0.], [0., 35., 0.], [0., 55., 0.]], kernel="analytic", wavelengths=WAVES)
    p = reduced.tensor([PARAMS])
    torch.testing.assert_close(full(p), reduced(p), rtol=1e-11, atol=1e-12)


def test_batched_inversion_with_missing_bands_and_shared_geometry():
    model = PROPBR(phase_angles=[15., 35., 55.], kernel="analytic", wavelengths=WAVES)
    truth = model.tensor([PARAMS])
    obs = model(truth).detach()
    obs[:, :, 10:15] = torch.nan
    initial = truth * model.tensor([1.05, .9, 1.1, .9, 1.1, .9, 1.1, 1.1])
    fit = invert(model, obs, initial=initial, n_iters=500, n_restarts=1, lr=.025, tolerance=1e-14)
    assert fit.parameters.shape == truth.shape
    assert float(fit.mse.sqrt().max()) < 3e-4
    model.space.validate(fit.parameters)
    assert torch.isfinite(fit.prediction).all()
    with pytest.raises(ValueError):
        invert(model, torch.full_like(obs, torch.nan), n_iters=1)


def test_kan_domain_and_free_surface_calibration():
    with pytest.raises(ValueError):
        PROPBR(phase_angles=[0.], kernel="kan")
    with pytest.raises(ValueError):
        PROPBR(phase_angles=[90.], kernel="analytic")
    with pytest.raises(ValueError):
        PROPBR(free_fsurf=True, kernel="kan")
    m = PROPBR(free_fsurf=True, kernel="analytic", wavelengths=WAVES)
    p = m.tensor([PARAMS + [1.2]])
    assert m.simulate(p).shape == (1, 1, len(WAVES))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_forward_and_jacobian_agree():
    for kernel in ("analytic", "kan"):
        cpu = PROPBR(kernel=kernel, wavelengths=WAVES)
        gpu = PROPBR(kernel=kernel, wavelengths=WAVES).to("cuda")
        pc, pg = cpu.tensor([PARAMS]), gpu.tensor([PARAMS])
        torch.testing.assert_close(cpu(pc), gpu(pg).cpu(), rtol=1e-9, atol=1e-10)
        jc = local_sensitivity(cpu, pc)["physical"]
        jg = local_sensitivity(gpu, pg)["physical"].cpu()
        torch.testing.assert_close(jc, jg, rtol=1e-8, atol=1e-8)
    # Same bounded inversion updates on CPU/CUDA; no matplotlib / local runtime workaround.
    p = cpu.tensor([PARAMS])
    initial = p * cpu.tensor([1.02, .95, 1.05, 1.02, .98, 1.03, .97, 1.03])
    obs = cpu(p).detach()
    fc = invert(cpu, obs, initial=initial, n_iters=100, n_restarts=1, tolerance=1e-14)
    fg = invert(gpu, obs.to("cuda"), initial=initial.to("cuda"), n_iters=100, n_restarts=1, tolerance=1e-14)
    torch.testing.assert_close(fc.prediction, fg.prediction.cpu(), rtol=1e-7, atol=1e-8)
