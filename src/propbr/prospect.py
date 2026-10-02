"""PROSPECT-D in native PyTorch, supporting autograd, jacfwd and vmap.

Adapted from PROSAIL (J. Gomez-Dans, GPLv3) and the author's Torch port.
The exponential integral uses bounded series / continued fraction branches;
neither a SciPy call nor a custom backward-only autograd Function is used.
"""
import math
import torch


def exp1(x):
    """E1(x), x > 0; safe inactive branches and differentiable tensor operations."""
    small = x.clamp(min=1e-30, max=1.)
    term = -small
    series = term
    for k in range(2, 26):
        term = term * (-small) / k
        series = series + term / k
    low = -0.5772156649015328606 - torch.log(small) - series
    large = x.clamp(min=1.)
    tail = torch.zeros_like(large)
    for k in range(60, 0, -1):
        tail = k / (1. + k / (large + tail))
    high = torch.exp(-large) / (large + tail)
    return torch.where(x <= 1., low, high)


def calctav(alpha_deg, nr):
    """Stern–Allen average interface transmittance; alpha is a fixed scalar angle."""
    if not 0 < alpha_deg <= 90:
        raise ValueError("calctav angle must be in (0,90] degrees")
    n2 = nr.square()
    npx, nm = n2 + 1., n2 - 1.
    a, k = (nr + 1.).square() / 2., -nm.square() / 4.
    sa2 = math.sin(math.radians(alpha_deg)) ** 2
    b1 = 0. if alpha_deg == 90 else torch.sqrt((sa2 - npx / 2.).square() + k)
    b = b1 - (sa2 - npx / 2.)
    ts = (k.square() / (6. * b**3) + k / b - b / 2. -
          k.square() / (6. * a**3) - k / a + a / 2.)
    tp1 = -2. * n2 * (b - a) / npx.square()
    tp2 = -2. * n2 * npx * torch.log(b / a) / nm.square()
    tp3 = n2 * (1. / b - 1. / a) / 2.
    tp4 = (16. * n2.square() * (n2.square() + 1.) *
           torch.log((2. * npx * b - nm.square()) / (2. * npx * a - nm.square())) /
           (npx**3 * nm.square()))
    tp5 = 16. * n2**3 * (1. / (2. * npx * b - nm.square()) -
                          1. / (2. * npx * a - nm.square())) / npx**3
    return (ts + tp1 + tp2 + tp3 + tp4 + tp5) / (2. * sa2)


def prospect_d(params, spectra, tav40=None, tav90=None):
    """Return (DHRF, transmittance, interface reflectance) for (...,7) inputs."""
    N, cab, car, anth, cw, cm, brown = params[..., :7].unbind(-1)
    N = N.unsqueeze(-1)
    k = (cab.unsqueeze(-1) * spectra.kab + car.unsqueeze(-1) * spectra.kcar +
         anth.unsqueeze(-1) * spectra.kant + cw.unsqueeze(-1) * spectra.kw +
         cm.unsqueeze(-1) * spectra.km + brown.unsqueeze(-1) * spectra.kbrown) / N
    positive = k.clamp(min=1e-12)
    tau_abs = (1. - positive) * torch.exp(-positive) + positive.square() * exp1(positive)
    # Limit at zero has derivative -2; avoid log(0) in an inactive branch.
    tau = torch.where(k > 1e-12, tau_abs, 1. - 2. * k)
    talf = calctav(40., spectra.nr) if tav40 is None else tav40
    t12 = calctav(90., spectra.nr) if tav90 is None else tav90
    ralf, r12 = 1. - talf, 1. - t12
    t21 = t12 / spectra.nr.square()
    r21 = 1. - t21
    denom = 1. - r21.square() * tau.square()
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t
    delta2 = (1. + r + t) * (1. + r - t) * (1. - r + t) * (1. - r - t)
    # Lossless stack has its own analytic limit; keep the unused square root finite.
    lossless = r + t >= 1. - 1e-12
    delta = torch.sqrt(delta2.clamp(min=1e-24))
    a = (1. + r.square() - t.square() + delta) / (2. * r)
    q = (2. * t) / (1. - r.square() + t.square() + delta)
    qN = q.clamp(min=1e-30).pow(N - 1.)
    # Inverse b avoids large powers of b for highly absorbing leaves.
    denom = (a.square() - qN.square()).clamp(min=1e-24)
    Rsub = a * (1. - qN.square()) / denom
    Tsub = qN * (a.square() - 1.) / denom
    Tlossless = t / (t + (1. - t) * (N - 1.))
    Rsub = torch.where(lossless, 1. - Tlossless, Rsub)
    Tsub = torch.where(lossless, Tlossless, Tsub)
    denom = 1. - Rsub * r
    return Ra + Ta * Rsub * t / denom, Ta * Tsub / denom, ralf
