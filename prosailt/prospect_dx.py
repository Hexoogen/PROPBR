#!/usr/bin/env python
"""The PROSPECT leaf optical properties model
Versions 5, D and PRO

Thanks for @jajberni for ProspectPRO implementation!

"""
import torch
from .spectral_library import get_spectra
from .special import e1

spectral_lib = get_spectra()


def calctav(alpha, nr):
    # ***********************************************************************
    # calctav
    # ***********************************************************************
    # Stern F. (1964), Transmission of isotropic radiation across an
    # interface between two dielectrics, Appl. Opt., 3(1):111-113.
    # Allen W.A. (1973), Transmission of isotropic light across a
    # dielectric surface in two and three dimensions, J. Opt. Soc. Am.,
    # 63(6):664-666.
    # ***********************************************************************

    # rd  = pi/180 torch.deg2rad
    n2 = nr * nr
    npx = n2 + 1
    nm = n2 - 1
    a = (nr + 1) * (nr + 1) / 2.0
    k = -(n2 - 1) * (n2 - 1) / 4.0
    sa = torch.sin(torch.deg2rad(alpha))

    if alpha != 90:
        b1 = torch.sqrt((sa * sa - npx / 2) * (sa * sa - npx / 2) + k)
    else:
        b1 = 0.0
    b2 = sa * sa - npx / 2
    b = b1 - b2
    b3 = b**3
    a3 = a**3
    ts = (k**2 / (6 * b3) + k / b - b / 2) - (k**2.0 /
                                              (6 * a3) + k / a - a / 2)

    tp1 = -2 * n2 * (b - a) / (npx**2)
    tp2 = -2 * n2 * npx * torch.log(b / a) / (nm**2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = (16 * n2**2 * (n2**2 + 1) * torch.log(
        (2 * npx * b - nm**2) / (2 * npx * a - nm**2)) / (npx**3 * nm**2))
    tp5 = (16 * n2**3 * (1.0 / (2 * npx * b - nm**2) - 1 /
                         (2 * npx * a - nm**2)) / (npx**3))
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts + tp) / (2 * sa**2)

    return tav


def refl_trans_one_layer(alpha, nr, tau):
    # ***********************************************************************
    # reflectance and transmittance of one layer
    # ***********************************************************************
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    # ***********************************************************************
    # reflectivity and transmissivity at the interface
    # -------------------------------------------------
    talf = calctav(alpha, nr)
    ralf = 1.0 - talf
    t12 = calctav(torch.as_tensor(90.0), nr)
    r12 = 1.0 - t12
    t21 = t12 / (nr * nr)
    r21 = 1 - t21

    # top surface side
    denom = 1.0 - r21 * r21 * tau * tau
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta

    # bottom surface side
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    return r, t, Ra, Ta, denom,ralf


def _stack_layers(r, t, layers):
    """Stokes stack with a differentiable limit at zero absorption.

    For c=(1+t**2-r**2)/(2*t), the transfer matrix gives
    T=t/(t*T_layers(c)+(1-c*t)*U_(layers-1)(c)) and
    R=r*U_(layers-1)(c)/(t*T_layers(c)+(1-c*t)*U_(layers-1)(c)).
    Taylor expansions around c=1 retain absorption derivatives without the
    sqrt(0) and 0/0 in the usual Stokes form. Away from that limit, inverse
    powers of b avoid overflow in highly absorbing layers.
    """
    delta = ((1.0 - r - t) * (1.0 + r - t)) / (2.0 * t.clamp_min(1e-30))
    near_lossless = delta.abs() < 1e-4

    # Safe inactive inputs keep both AD modes finite before torch.where.
    rr = torch.where(near_lossless, torch.full_like(r, 0.1), r)
    tt = torch.where(near_lossless, torch.full_like(t, 0.5), t)
    disc = ((1 + rr + tt) * (1 + rr - tt) *
            (1 - rr + tt) * (1 - rr - tt)).clamp_min(1e-24).sqrt()
    a = (1 + rr.square() - tt.square() + disc) / (2 * rr.clamp_min(1e-30))
    q = (2 * tt) / (1 - rr.square() + tt.square() + disc).clamp_min(1e-30)
    qn = q.clamp_min(1e-30).pow(layers)
    denom = (a.square() - qn.square()).clamp_min(1e-24)
    reflected = a * (1 - qn.square()) / denom
    transmitted = qn * (a.square() - 1) / denom

    d = torch.where(near_lossless, delta, torch.zeros_like(delta))
    n2 = layers.square()
    cheb_t = 1 + n2 * d + n2 * (n2 - 1) * d.square() / 6
    cheb_t = cheb_t + n2 * (n2 - 1) * (n2 - 4) * d.pow(3) / 90
    cheb_u = layers * (1 + (n2 - 1) * d / 3 +
                       (n2 - 1) * (n2 - 4) * d.square() / 30 +
                       (n2 - 1) * (n2 - 4) * (n2 - 9) * d.pow(3) / 630)
    rs = torch.where(near_lossless, r, torch.full_like(r, 0.1))
    ts = torch.where(near_lossless, t, torch.full_like(t, 0.5))
    near_denom = ts * cheb_t + (1 + rs.square() - ts.square()) * cheb_u / 2
    return (torch.where(near_lossless, rs * cheb_u / near_denom, reflected),
            torch.where(near_lossless, ts / near_denom, transmitted))


def prospect_dt(N,cab,car,ant,cw,cm,cbrown,alpha=torch.as_tensor(40.0),device=None):
    device = device or (N.device if isinstance(N, torch.Tensor) else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    nr = spectral_lib.prospectd.nr.to(device)
    N = torch.as_tensor(N, dtype=nr.dtype, device=device) if not isinstance(N, torch.Tensor) else N
    kab = spectral_lib.prospectd.kab.to(device)
    kcar = spectral_lib.prospectd.kcar.to(device)
    kbrown = spectral_lib.prospectd.kbrown.to(device)
    kw = spectral_lib.prospectd.kw.to(device)
    km = spectral_lib.prospectd.km.to(device)
    kant = spectral_lib.prospectd.kant.to(device)


    kall = (cab * kab + car * kcar + ant * kant + cbrown * kbrown + cw * kw + cm * km) / N

    positive = kall.clamp_min(1e-12)
    tau_abs = (1 - positive) * torch.exp(-positive) + positive.square() * e1(positive)
    # tau(0)=1 and tau'(0)=-2; a constant branch would lose the latter.
    tau = torch.where(kall > 1e-12, tau_abs, 1 - 2 * kall)

    r, t, Ra, Ta, denom,ralf = refl_trans_one_layer(alpha, nr, tau)

    # ***********************************************************************
    # reflectance and transmittance of N layers
    # Stokes equations to compute properties of next N-1 layers (N real)
    # Normal case
    # ***********************************************************************
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    # ***********************************************************************
    Rsub, Tsub = _stack_layers(r, t, N - 1)

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom = 1 - Rsub * r
    # tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom
    refl_diff = refl-ralf

    return refl, refl_diff
