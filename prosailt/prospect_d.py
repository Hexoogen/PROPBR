#!/usr/bin/env python
"""The PROSPECT leaf optical properties model
Versions 5, D and PRO

Thanks for @jajberni for ProspectPRO implementation!

"""
import torch
from .special import e1
from prosailt import spectral_lib
from scipy.signal import decimate
from scipy.interpolate import interp1d
#
# def pos_expi(x):
#     k = torch.arange(1, 75, dtype=x.dtype, device=x.device)
#     r = torch.cumprod(x.unsqueeze(-1)*k/torch.square(k+1), dim=-1)
#     ga = torch.tensor([0.5772156649015328], dtype=x.dtype, device=x.device)
#     y = ga + torch.log(x) + x * (1+(r).sum(-1))
#     return y
#
# def ein(x):
#     k = torch.arange(1, 75, dtype=x.dtype, device=x.device)
#     r = torch.cumprod(- x.unsqueeze(-1)*k/torch.square(k+1), dim=-1)
#     return x * (1+(r).sum(-1))
#
#
# def torch_e1xg(x):
#     t0 = torch.zeros_like(x)
#     M=20
#     for k in range(M,0,-1):
#         t0 = k/(1+k/(x+t0))
#     e = torch.exp(-x) / (x + t0)
#     return e
#
# def torch_e1xl(x):
#     ga = torch.tensor([0.5772156649015328], dtype=x.dtype, device=x.device)
#     e = - ga - torch.log(x) + ein(x)
#     return e
#
# def e1(x,approx_switch=1):
#     e = torch.zeros_like(x)
#     x_l0 = x<0
#     x_la = x<approx_switch
#     x_ga = x>=approx_switch
#     e[x_l0] = torch.nan
#     e[x_la] = torch_e1xl(x[x_la])
#     e[x_ga] = torch_e1xg(x[x_ga])
#     return e
#
# def neg_expi(x):
#     return - e1(-x)
#
# def expi(x):
#     ei = torch.zeros_like(x)
#     x_l0 = x<0
#     x_g0 = x>0
#     ei[x_l0] = neg_expi(x[x_l0])
#     ei[x_g0] = pos_expi(x[x_g0])
#     return ei
#
# class Expi(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = expi(i)
#         ctx.save_for_backward(i)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#          i, = ctx.saved_tensors
#          return grad_output * (-i.exp()/i)
def expi_torch(x, n_terms=25):
    """
    指数积分 Ei(x)，纯张量操作，完全兼容 vmap
    无任何条件分支、无 .item()、无 .all()、无提前 break
    """
    gamma = 0.5772156649015329

    result = gamma + torch.log(torch.abs(x).clamp(min=1e-30))

    x_pow = x.clone()
    factorial = 1.0

    for k in range(1, n_terms):
        factorial *= k
        term = x_pow / (k * factorial)

        # clamp 代替 if 判断（纯张量操作）
        term = term.clamp(min=-1e10, max=1e10)
        result = result + term
        x_pow = (x_pow * x).clamp(min=-1e30, max=1e30)

    result = torch.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

    return result


def expi_pade(x):
    """
    Padé 逼近，比级数展开在 |x| 较大时更稳定
    同样纯 PyTorch，自动可微，自动兼容 vmap
    """
    gamma = 0.5772156649015329

    # 对 |x| < 1 用级数，|x| >= 1 用渐近展开
    small = x.abs() < 1.0

    # 级数展开部分
    result_series = gamma + torch.log(torch.abs(x))
    x_pow = x.clone()
    factorial = 1.0
    for k in range(1, 25):
        factorial *= k
        result_series = result_series + x_pow / (k * factorial)
        x_pow = x_pow * x

    # 渐近展开部分（|x| >= 1）
    # Ei(x) ≈ e^x / x * Σ(k=0..n) k! / x^k
    result_asymp = torch.zeros_like(x)
    term = 1.0 / x
    result_asymp = result_asymp + term
    for k in range(1, 20):
        term = term * k / x
        result_asymp = result_asymp + term
    result_asymp = result_asymp * torch.exp(x)

    # 选择
    return torch.where(small, result_series, result_asymp)


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

    if alpha == 90:b1 = 0.0
    else:
        b1 = torch.sqrt((sa * sa - npx / 2) * (sa * sa - npx / 2) + k)

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


def refl_trans_one_layer(nr, tau):
    # ***********************************************************************
    # reflectance and transmittance of one layer
    # ***********************************************************************
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    # ***********************************************************************
    # reflectivity and transmissivity at the interface
    # -------------------------------------------------
    talf = calctav(torch.as_tensor(40.0), nr)
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

    return r, t, Ra, Ta, denom, ralf


def prospect_dt(N,cab,car,ant,cw,cm,cbrown,device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    nr = spectral_lib.prospectd.nr.to(device)
    kab = spectral_lib.prospectd.kab.to(device)
    kcar = spectral_lib.prospectd.kcar.to(device)
    kbrown = spectral_lib.prospectd.kbrown.to(device)
    kw = spectral_lib.prospectd.kw.to(device)
    km = spectral_lib.prospectd.km.to(device)
    kant = spectral_lib.prospectd.kant.to(device)

    # params = torch.rand(1000, 7) * torch.tensor([2.5, 120, 30, 40,0.06, 0.05,1])
    # params[:, 0] += 1  # N ∈ [1, 3.5]
    # params = params.to('cuda')
    # N, cab, car, ant, cw, cm, cbrown=params[:,0],params[:,1],params[:,2],params[:,3],params[:,4],params[:,5],params[:,6]
    # N, cab, car, ant, cw, cm, cbrown= N.reshape(-1,1), cab.reshape(-1,1), car.reshape(-1,1), ant.reshape(-1,1), cw.reshape(-1,1), cm.reshape(-1,1), cbrown.reshape(-1,1)

    kall = (cab * kab + car * kcar + ant * kant + cbrown * kbrown + cw * kw + cm * km) / N

    j = kall > 0
    t1 = (1 - kall) * torch.exp(-kall)
    # t2 = kall ** 2 * (-expi_torch(-kall))
    t2 = kall ** 2 * e1(kall)
    # tau = torch.ones_like(t1)
    # tau[j] = t1[j] + t2[j]
    tau = torch.where(kall > 0, t1 + t2, torch.ones_like(t1))

    r, t, Ra, Ta, denom, ralf = refl_trans_one_layer(nr, tau)

    # ***********************************************************************
    # reflectance and transmittance of N layers
    # Stokes equations to compute properties of next N-1 layers (N real)
    # Normal case
    # ***********************************************************************
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    # ***********************************************************************
    D = torch.sqrt((1 + r + t) * (1 + r - t) * (1.0 - r + t) * (1.0 - r - t))
    rq = r * r
    tq = t * t
    a = (1 + rq - tq + D) / (2 * r)
    b = (1 - rq + tq + D) / (2 * t)

    bNm1 = torch.pow(b, N - 1)
    bN2 = bNm1 * bNm1
    a2 = a * a
    denom = a2 * bN2 - 1
    Rsub = a * (bN2 - 1) / denom
    Tsub = bNm1 * (a2 - 1) / denom

    # Case of zero absorption
    # j = r + t >= 1.0
    # if len(torch.where(j)[0])>0:
    #     Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (N[torch.where(j)[0]].squeeze() - 1))
    #     Rsub[j] = 1 - Tsub[j]
    Tsub_zero = t / (t + (1 - t) * (N - 1))  # 零吸收分支
    Rsub_zero = 1 - Tsub_zero

    j = r + t >= 1.0
    # Tsub = torch.where(j, Tsub_zero, Tsub)
    Rsub = torch.where(j, Rsub_zero, Rsub)

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom = 1 - Rsub * r
    # tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom
    refl_diff = refl - ralf.squeeze(0)

    return refl, refl_diff
