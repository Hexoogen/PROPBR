"""Smith–GGX, Schlick and reference hemispherical terms (paper Eqs. 2–14)."""
import math
import numpy as np
import torch


def fresnel0(nr):
    return ((1. - nr) / (1. + nr)).square()


def schlick(nr, phase_rad):
    f0 = fresnel0(nr)
    return f0 + (1. - f0) * (1. - torch.cos(phase_rad / 2.)).pow(5)


def surface_factor(sigma):
    return .66 * sigma.pow(-.63)


def ggx(sigma, cos_half_normal):
    return sigma.square() / (math.pi * (cos_half_normal.square() * (sigma.square() - 1.) + 1.).square())


def reduced_ks(phase_rad, sigma, fsurf=None):
    """Eq.14. Ks already includes fsurf. Stable limit at 90 degrees for LUT training."""
    f = surface_factor(sigma) if fsurf is None else fsurf
    c, s = torch.cos(phase_rad), torch.sin(phase_rad)
    # Algebraic form of G / (4*cos(phase)); avoids tan/cos cancellation at 90°.
    visibility = 1. / (2. * (c + torch.sqrt(c.square() + sigma.square() * s.square())))
    return math.pi * f * ggx(sigma, torch.cos(phase_rad / 2.)) * visibility


def reference_specular(sigma, internal_nr):
    """Paper Sec.4.1.3 fit: use n_in, without the empirical surface multiplier."""
    return fresnel0(internal_nr) * torch.exp(-1.26 * sigma.pow(1.79))


def full_geometry_ks(sigma, incidence, viewing, azimuth, fsurf=None):
    """Eqs.2–10 for explicit radian geometry; returns (Ks, phase angle)."""
    ci, cr = torch.cos(incidence), torch.cos(viewing)
    cp = ci * cr + torch.sin(incidence) * torch.sin(viewing) * torch.cos(azimuth)
    phase = torch.acos(cp.clamp(-1., 1.))
    cos_half = torch.sqrt(((1. + cp) / 2.).clamp(min=1e-15))
    cos_normal = ((ci + cr) / (2. * cos_half)).clamp(max=1.)
    # Correlated Smith visibility, algebraically divided by 4*cos(i)*cos(r).
    si = torch.sqrt(ci.square() + sigma.square() * torch.sin(incidence).square())
    sr = torch.sqrt(cr.square() + sigma.square() * torch.sin(viewing).square())
    visibility = 1. / (2. * (si * cr + sr * ci))
    f = surface_factor(sigma) if fsurf is None else fsurf
    return math.pi * f * ggx(sigma, cos_normal) * visibility, phase


def hemispherical_specular(sigma, nr, incidence_deg=0., order=192):
    """Numerical unscaled GGX hemispherical reflectance for the reference fit.

    Midpoint azimuth + Gauss–Legendre integration of u=tan(theta_r/2).
    Geometric u panels resolve the narrow specular lobe for sigma=0.01.
    This diagnostic is NumPy; it is not used inside differentiable inversion.
    """
    if not .01 <= sigma <= 1. or not 0 <= incidence_deg < 90. or order < 16:
        raise ValueError("Invalid hemispherical integration configuration")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.array([0., .002, .01, .05, .2, .5, 1.])
    u = np.concatenate([lo + (nodes + 1.) * (hi - lo) / 2. for lo, hi in zip(edges[:-1], edges[1:])])
    wu = np.concatenate([weights * (hi - lo) / 2. for lo, hi in zip(edges[:-1], edges[1:])])
    theta = 2. * np.arctan(u)
    phi = (np.arange(order * 2) + .5) * math.pi / order
    ci, si = math.cos(math.radians(incidence_deg)), math.sin(math.radians(incidence_deg))
    cr, sr = np.cos(theta)[:, None], np.sin(theta)[:, None]
    cp = ci * cr + si * sr * np.cos(phi)[None, :]
    ch = np.sqrt(np.maximum((1. + cp) / 2., 1e-15))
    cm = (ci + cr) / (2. * ch)
    D = sigma**2 / (math.pi * (cm**2 * (sigma**2 - 1.) + 1.)**2)
    G = 2. / (np.sqrt(1. + sigma**2 * (si / ci)**2) + np.sqrt(1. + sigma**2 * (sr / cr)**2))
    n = np.atleast_1d(nr)
    f0 = ((1. - n) / (1. + n))**2
    # Integrate specular BRDF*cos(theta)*sin(theta) dtheta dphi.
    base = D * G * sr / (4. * ci) * (2. / (1. + u[:, None]**2))
    b = (1. - ch)**5
    slope = np.sum(wu[:, None] * base * (1. - b)) * math.pi / order
    intercept = np.sum(wu[:, None] * base * b) * math.pi / order
    result = f0 * slope + intercept
    return float(result[0]) if np.ndim(nr) == 0 else result
