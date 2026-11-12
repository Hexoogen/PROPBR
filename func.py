import os
from pathlib import Path
import numpy as np
import prospect

from prospect import prospect_pro,prospect_d,spectral_lib
from prosailt import spectral_lib as _spectra_lib
from prospect import calctav
import pandas as pd
import math
import logging

_LUT_DIR = Path(__file__).resolve().parent / 'lut'

def prospecular(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    cb = para[3]
    cw = para[4]
    cm = para[5]
    ant = para[6]
    prot = para[7]
    cbc = para[8]
    rough = para[9]
    fs= para[10]
    sza = np.deg2rad(para[11])
    vza = np.deg2rad(para[12])
    raa = np.deg2rad(para[13])

    _, DHR_diff= prospect_pro(n, cab, car, cb, cw, cm, ant, prot, cbc)

    ph = np.arccos(np.cos(sza)* np.cos(vza) + np.sin(sza)* np.sin(vza)*np.cos(raa))
    pha = np.arccos((np.cos(sza) + np.cos(vza)) / (2 * np.cos(ph/2)))

    E1 = 2 * np.cos(pha) *np.cos(sza) / np.cos(ph / 2)
    E2 = 2 * np.cos(pha) *np.cos(vza) / np.cos(ph / 2)
    G = np.minimum(1, np.minimum(E1, E2))

    nr = np.load(_LUT_DIR / 'DLM_nr.npy')
    g = np.sqrt(np.abs((fs*nr) ** 2 + np.cos(ph / 2) ** 2 - 1))  # g
    beckman=np.exp(-np.tan(pha) ** 2 / rough ** 2)/(np.pi*(rough ** 2) * np.cos(pha) ** 4)
    fnq = 0.5 * ((g - np.cos(ph / 2)) / (g + np.cos(ph / 2))) ** 2  # The first half of Fresnel factor
    fnh = 1 + ((np.cos(ph / 2) * (g + np.cos(ph / 2)) - 1) /
               (np.cos(ph / 2) * (g - np.cos(ph / 2)) + 1)) ** 2  # The second half of Fresnel factor
    Fn = fnq * fnh  # Fresnel factor
    fmq = 4 * np.cos(sza)* np.cos(vza)
    brdfj = beckman * Fn * G / fmq  # BRDF of specular component
    brf=DHR_diff+brdfj*np.pi
    return brf

def prospeculard(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    ant = para[3]
    cw = para[4]
    cm = para[5]
    cb = para[6]
    rough = para[7]
    fs= para[8]
    sza = np.deg2rad(para[9])
    vza = np.deg2rad(para[10])
    raa = np.deg2rad(para[11])

    _, DHR_diff= prospect_d(n, cab, car, ant, cw, cm, cb)

    ph = np.arccos(np.cos(sza)* np.cos(vza) + np.sin(sza)* np.sin(vza)*np.cos(raa))
    pha = np.arccos((np.cos(sza) + np.cos(vza)) / (2 * np.cos(ph/2)))

    E1 = 2 * np.cos(pha) *np.cos(sza) / np.cos(ph / 2)
    E2 = 2 * np.cos(pha) *np.cos(vza) / np.cos(ph / 2)
    G = np.minimum(1, np.minimum(E1, E2))

    nr = np.load(_LUT_DIR / 'DLM_nr.npy')
    g = np.sqrt(np.abs((fs*nr) ** 2 + np.cos(ph / 2) ** 2 - 1))  # g
    beckman=np.exp(-np.tan(pha) ** 2 / rough ** 2)/(np.pi*(rough ** 2) * np.cos(pha) ** 4)
    fnq = 0.5 * ((g - np.cos(ph / 2)) / (g + np.cos(ph / 2))) ** 2  # The first half of Fresnel factor
    fnh = 1 + ((np.cos(ph / 2) * (g + np.cos(ph / 2)) - 1) /
               (np.cos(ph / 2) * (g - np.cos(ph / 2)) + 1)) ** 2  # The second half of Fresnel factor
    Fn = fnq * fnh  # Fresnel factor
    fmq = 4 * np.cos(sza)* np.cos(vza)
    brdfj = beckman * Fn * G / fmq  # BRDF of specular component
    brf=DHR_diff+brdfj*np.pi
    return brf

def pbrfull(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    ant = para[3]
    cw = para[4]
    cm = para[5]
    cb = para[6]
    rough = para[7]
    fs= para[8]
    sza = np.deg2rad(para[9])
    vza = np.deg2rad(para[10])
    raa = np.deg2rad(para[11])

    DHR, _= prospect_d(n, cab, car, ant, cw, cm, cb)

    rough2=rough**2
    ph = np.arccos(np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(raa))
    vh = np.cos(ph / 2)
    nm = (np.cos(sza) + np.cos(vza)) / (2 * vh)
    G = 2 / (np.sqrt(1 + np.tan(sza) ** 2 * rough2) + np.sqrt(1 + np.tan(vza) ** 2 * rough2))
    ggx = rough2 / (np.pi * (nm ** 2 * (rough2 - 1) + 1) ** 2)

    nr = np.load(_LUT_DIR / 'DLM_nr.npy')
    F0 = ((1 - fs*nr) / (1 + fs*nr)) ** 2  # Fresnel factor
    Fn = F0 + (1 - F0) * (1 - vh) ** 5  # Fresnel factor
    brf_spec = np.pi * G * ggx * Fn / (4 * np.cos(sza) * np.cos(vza))
    brf=DHR- dhr_spec(rough)+brf_spec
    return brf

def pbrfullf(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    ant = para[3]
    cw = para[4]
    cm = para[5]
    cb = para[6]
    rough = para[7]
    fs= para[8]
    sza = np.deg2rad(para[9])
    vza = np.deg2rad(para[10])
    raa = np.deg2rad(para[11])

    DHR, _= prospect_d(n, cab, car, ant, cw, cm, cb)

    rough2=rough**2
    ph = np.arccos(np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(raa))
    vh = np.cos(ph / 2)
    nm = (np.cos(sza) + np.cos(vza)) / (2 * vh)
    G = 2 / (np.sqrt(1 + np.tan(sza) ** 2 * rough2) + np.sqrt(1 + np.tan(vza) ** 2 * rough2))
    ggx = rough2 / (np.pi * (nm ** 2 * (rough2 - 1) + 1) ** 2)

    nr = np.load(_LUT_DIR / 'DLM_nr.npy')
    g = np.sqrt(np.abs((fs * nr) ** 2 + np.cos(ph / 2) ** 2 - 1))  # g
    fnq = 0.5 * ((g - np.cos(ph / 2)) / (g + np.cos(ph / 2))) ** 2  # The first half of Fresnel factor
    fnh = 1 + ((np.cos(ph / 2) * (g + np.cos(ph / 2)) - 1) /
               (np.cos(ph / 2) * (g - np.cos(ph / 2)) + 1)) ** 2  # The second half of Fresnel factor
    Fn = fnq * fnh  # Fresnel factor

    brf_spec = np.pi * G * ggx * Fn / (4 * np.cos(sza) * np.cos(vza))
    brf=DHR- dhr_spec(rough)+brf_spec
    return brf

def dhr_spec(rough):
    nr=spectral_lib.prospectd.nr
    F0= ((1 - nr) / (1 + nr)) ** 2
    return F0*np.exp(-1.26 * rough**1.79)

def Kf_lut(lut,ph,rough):
    pidx0 = int(np.floor(ph) - 1)
    pidx1 = int(np.floor(ph))
    pper = ph - np.floor(ph)

    ridx0=int(np.floor(rough*100)-1)
    ridx1=int(np.floor(rough*100))
    rper=rough*100-np.floor(rough*100)

    f00,f01,f10,f11=lut[pidx0,ridx0],lut[pidx0,ridx1],lut[pidx1,ridx0],lut[pidx1,ridx1]
    Kf=(1 - pper) * ((1 - rper) * f00 + rper * f01) + pper * ((1 - rper) * f10 + rper * f11)
    return Kf

def diff_lut(h,h2,dhr,ph,rough):

    pidx0 = int(np.floor(ph) - 1)
    pidx1 = int(np.floor(ph))
    pper = ph - np.floor(ph)

    ridx0=int(np.floor(rough*100)-1)
    ridx1=int(np.floor(rough*100))
    rper=rough*100-np.floor(rough*100)

    f00,f01,f10,f11=h[pidx0,ridx0],h[pidx0,ridx1],h[pidx1,ridx0],h[pidx1,ridx1]
    k=(1 - pper) * ((1 - rper) * f00 + rper * f01) + pper * ((1 - rper) * f10 + rper * f11)

    f00,f01,f10,f11=h2[pidx0,ridx0],h2[pidx0,ridx1],h2[pidx1,ridx0],h2[pidx1,ridx1]
    b=(1 - pper) * ((1 - rper) * f00 + rper * f01) + pper * ((1 - rper) * f10 + rper * f11)
    return k*(dhr-b)

#%%
def resample_to_10nm(refl):
    wl=np.arange(400,2501)
    bins = np.arange(400, 2501, 10)
    resampled = []
    new_waves = []

    for i in range(len(bins) - 1):
        mask = (wl >= bins[i]) & (wl < bins[i+1])
        if mask.sum() == 0:
            continue
        resampled.append(refl[:, mask].mean(axis=1))
        new_waves.append((bins[i] + bins[i+1]) / 2)

    return np.vstack(resampled).T, np.array(new_waves)
#%% data
def get_global_best():
    D = pd.read_csv('spec/global.csv')
    D = D[D['Foreoptic type'] != 'Integrating sphere']
    X = D.iloc[:, :196].values
    X = X[X[:, 55] > 0.05]
    return X

def get_SVC():
    resampler = np.load(_LUT_DIR / 'Resampler.npy')
    D = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    return np.dot(D[wl], resampler.T)

def get_ga():
    X1=get_global_best()
    X2=get_SVC()
    return np.vstack((X1,X2))

def get_global_no():
    D = pd.read_csv('spec/global.csv')
    D1 = D[D['Foreoptic type'] == 'Integrating sphere']
    D2 = D[D.iloc[:, 55] < 0.05]
    X = pd.concat([D1, D2], axis=0).iloc[:, :196].values
    return X

def get_all():
    D1 = pd.read_csv('spec/global.csv').iloc[:, :196]
    resampler = np.load(_LUT_DIR / 'Resampler.npy')
    D2 = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    D2 = np.dot(D2[wl], resampler.T)
    D2 = pd.DataFrame(D2,columns=D1.columns)
    D=pd.concat([D1,D2], axis=0)
    return D.values

def get_all_sim():
    D1 = pd.read_csv('spec/global.csv').iloc[:, :196].values
    D2 = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    D2 = D2[wl].values
    return D1,D2
def get_all_simr():
    D1 = pd.read_csv('spec/global.csv').iloc[:, :196].values.ravel()
    D2 = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    D2 = D2[wl].values.ravel()
    D=np.hstack((D1,D2))
    return D

def get_hsi():
    D1 = pd.read_csv('spec/ApMV900.csv')
    D2 = pd.read_csv('spec/Grape.csv')
    X1 = D1.values[:, :115]
    X2 = D2.values[:, :115]
    X = np.vstack((X1, X2))
    return X

#%%
import numpy as np
import torch
from prosailt.prospect_dx import prospect_dt
from kan_model import KsKAN

_DEFAULT_DEVICE = os.environ.get('PROPBR_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
_CACHE = {}


def _ensure(device=None):
    """Lazily build the module-level tensors (LUT / KAN weights / resampler) on `device`.

    Defaults to CUDA when available, otherwise CPU.  Results are cached per device,
    so loops that alternate devices only pay the transfer once.
    """
    dev = torch.device(device or _DEFAULT_DEVICE)
    # torch.device('cuda') != torch.device('cuda:0'); normalise so that a cached
    # build is reused (building inside a functorch transform is not allowed).
    if dev.type == 'cuda' and dev.index is None:
        dev = torch.device('cuda', torch.cuda.current_device())
    if dev not in _CACHE:
        _CACHE[dev] = (
            torch.tensor(np.load(_LUT_DIR / 'DLM_nr.npy'), dtype=torch.float32, device=dev).unsqueeze(0),
            torch.hstack((torch.arange(10, 66, 5), torch.arange(10, 66, 5))).to(dev),
            KsKAN(_LUT_DIR / 'ks_kan_weights1.pt').to(dev),
            torch.tensor(np.load(_LUT_DIR / 'Resampler.npy').T, dtype=torch.float32, device=dev),
        )
    return _CACHE[dev]


dlm_nr, szas, ks_kan, resampler = _ensure()
# Also cache the CPU copy on a GPU box: building tensors inside a functorch
# transform (jacfwd/vmap) is illegal, so a later CPU call must not be the first
# one to build them.  The extra copy costs ~2 MB.
if torch.device(_DEFAULT_DEVICE).type != 'cpu' and torch.cuda.is_available():
    _ensure(torch.device('cpu'))


def warmup(device=None):
    """Build (and cache) the device tensors before entering a functorch transform.

    `torch.func.jacfwd` / `vmap` forbid building modules inside the transform
    (nn.Module.to mutates parameters in place), so call this once up-front when
    you differentiate a forward model directly, e.g. ``func.warmup('cpu')``.
    """
    _ensure(device)


# --------------------------------------------------------------------------
# Paper-consistent surface terms (Eqs. 14, 18)
# --------------------------------------------------------------------------
# The KAN surrogate reproduces Ks *including* the surface factor
# f = 0.66 * sigma**(-0.63) (it was trained on Eq. 14, which already contains f).
# Apply the Schlick Fresnel factor once, after the geometry-only KAN output.
# Subtract the roughness-dependent hemispherical reference from DHRF.
# --------------------------------------------------------------------------

def _fresnel0(nr):
    """Normal-incidence Fresnel reflectance for refractive index `nr`."""
    return ((1.0 - nr) / (1.0 + nr)) ** 2


def _schlick(nr, phase_rad):
    """Paper Eq. 9: phase angle is in radians."""
    f0 = _fresnel0(nr)
    return f0 + (1 - f0) * (1 - torch.cos(phase_rad / 2)).pow(5)


def _reference_specular(rough):
    """Paper Eq. 18: rho_ref = F0(n_internal) * exp(-1.26 * sigma**1.79)."""
    nr = _spectra_lib.prospectd.nr.to(rough.device)
    return _fresnel0(nr) * torch.exp(-1.26 * rough ** 1.79)


def _prospect_terms(params):
    """Unpack the seven PROSPECT-D parameters and run the leaf optics."""
    N = params[:,0].reshape(-1,1)
    cab = params[:,1].reshape(-1,1)
    car = params[:,2].reshape(-1,1)
    ant = params[:,3].reshape(-1,1)
    cw = params[:,4].reshape(-1,1)
    cm = params[:,5].reshape(-1,1)
    cb = params[:,6].reshape(-1,1)
    dhr, _ = prospect_dt(N,cab,car,ant,cw,cm,cb,device=params.device)
    return dhr


def _surface_factor(rough):
    return 0.66 * rough ** -0.63


def kan_torch(params, device=None):
    """PROPBR, eight parameters (f_s fixed by sigma).  Phase angle 15 deg."""
    dlm_nr, szas, ks_kan, resampler = _ensure(device if device is not None else params.device)
    rough = params[:,7].reshape(-1,1)
    pha = torch.full_like(rough, np.deg2rad(15))
    dhr = _prospect_terms(params)
    ks = ks_kan(pha, rough)                       # already includes f(sigma)
    brf = (dhr - _reference_specular(rough)) + ks * _schlick(dlm_nr, pha)
    return brf


def kan_torch_lF(params, device=None):
    """PROPBR, nine parameters: f_s free, phase angle 35 deg.

    The surrogate is linear in the surface factor, so a different f_s is applied
    as an exact rescaling of the surrogate output.
    """
    dlm_nr, szas, ks_kan, resampler = _ensure(device if device is not None else params.device)
    rough = params[:,7].reshape(-1,1)
    fs = params[:,8].reshape(-1,1)
    pha = torch.full_like(rough, np.deg2rad(35))
    dhr = _prospect_terms(params)
    ks = ks_kan(pha, rough) * (fs / _surface_factor(rough))
    brf = (dhr - _reference_specular(rough)) + ks * _schlick(dlm_nr, pha)
    return brf


def full_torch(params, device=None):
    """PROPBR analytic kernel, nine parameters, phase angle 35 deg."""
    dlm_nr, szas, ks_kan, resampler = _ensure(device if device is not None else params.device)
    rough = params[:,7].reshape(-1,1)
    fs = params[:,8].reshape(-1,1)
    pha = torch.full_like(rough, np.deg2rad(35))
    dhr = _prospect_terms(params)
    rough2 = rough ** 2
    G = 2 / (1 + torch.sqrt(1 + torch.tan(pha) ** 2 * rough2))
    ggx = rough2 / (torch.pi * (torch.cos(pha / 2) ** 2 * (rough2 - 1) + 1) ** 2)
    ks = torch.pi * G * ggx / (4 * torch.cos(pha))
    brf = (dhr - _reference_specular(rough)) + fs * ks * _schlick(dlm_nr, pha)
    return brf


def full_torch_l(params, device=None):
    """PROPBR analytic kernel at the 24 LOTUS phase angles (10-65 deg)."""
    dlm_nr, szas, ks_kan, resampler = _ensure(device if device is not None else params.device)
    rough = params[:,7].reshape(-1,1)
    fs = params[:,8].reshape(-1,1)
    dhr = _prospect_terms(params)
    rough2 = rough ** 2
    pbrf = torch.zeros((params.shape[0], 0), device=params.device)
    for i in range(24):
        pha = torch.deg2rad(szas[i])          # szas is in degrees
        G = 2 / (1 + torch.sqrt(1 + torch.tan(pha) ** 2 * rough2))
        ggx = rough2 / (torch.pi * (torch.cos(pha / 2) ** 2 * (rough2 - 1) + 1) ** 2)
        ks = torch.pi * G * ggx / (4 * torch.cos(pha))
        brf = (dhr - _reference_specular(rough)) + fs * ks * _schlick(dlm_nr, pha)
        pbrf = torch.hstack((pbrf, brf[:, 50:1251]))
    return pbrf


def propbr_35_torch(params, device=None):
    """Eight-parameter PROPBR at 35 degrees.

    Columns are N, Cab, Car, Anth, Cw, Cm, Cbrown, sigma. The trained
    directional kernel includes the factor 0.66 * sigma**(-0.63).
    Returns (B, 2101) for 400-2500 nm at 1 nm spacing.
    """
    if params.ndim != 2 or params.shape[1] != 8:
        raise ValueError('PROPBR expects shape (B, 8): N, Cab, Car, Anth, Cw, Cm, Cbrown, sigma')
    nr, _, kernel, _ = _ensure(device if device is not None else params.device)
    sigma = params[:, 7:8]
    phase = torch.full_like(sigma, np.deg2rad(35))
    return (_prospect_terms(params) - _reference_specular(sigma) +
            kernel(phase, sigma) * _schlick(nr, phase))


def propbr_24_torch(params, device=None):
    """Eight-parameter PROPBR for 24 directions at 10-65 degrees.

    The directional amplitude is determined by 0.66 * sigma**(-0.63).
    Returns (B, 24 * 1201), with 450-1650 nm for each direction.
    """
    if params.ndim != 2 or params.shape[1] != 8:
        raise ValueError('PROPBR expects shape (B, 8): N, Cab, Car, Anth, Cw, Cm, Cbrown, sigma')
    sigma = params[:, 7:8]
    coupled = torch.cat((params, 0.66 * sigma.pow(-0.63)), dim=1)
    return full_torch_l(coupled, device=device)


def spropbr_torch(params, device=None):
    """sPROPBR with [N, Cab, Car, Anth, Cw, Cm, Cbrown, mu].

    Geometry is absorbed into mu. Remove 1-tav(40 deg) from PROSPECT DHRF,
    then add mu*F0(n_surface), as in paper Eq. 21.
    """
    dlm_nr, szas, ks_kan, resampler = _ensure(device if device is not None else params.device)
    Kf = params[:,7].reshape(-1,1)
    N = params[:,0].reshape(-1,1)
    cab = params[:,1].reshape(-1,1)
    car = params[:,2].reshape(-1,1)
    ant = params[:,3].reshape(-1,1)
    cw = params[:,4].reshape(-1,1)
    cm = params[:,5].reshape(-1,1)
    cb = params[:,6].reshape(-1,1)
    _, dhr_diff = prospect_dt(N,cab,car,ant,cw,cm,cb,device=params.device)
    brf = dhr_diff + Kf * _fresnel0(dlm_nr)
    return brf


Kf_torch = spropbr_torch  # Existing experiment scripts keep their original entry point.


#%%
import torch
from torch.func import jacfwd, vmap


class AdaptiveLM:
    """
    批量自适应 Levenberg-Marquardt 优化器
    - sigmoid reparameterization 实现有界约束
    - torch.func.jacfwd + vmap 计算批量 Jacobian
    - 逐样本自适应阻尼 + 早停
    """

    def __init__(
        self,
        forward_model,
        low,
        high,
        n_params,
        device=None,
    ):
        self.forward_model = forward_model
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.P = n_params

        # 边界
        self.low  = low.to(device).float()
        self.high = high.to(device).float()
        self.scale = self.high - self.low

        # Jacobian 函数
        self._build_jacobian_fn()

    def _build_jacobian_fn(self):
        """构建 vmap(jacfwd) 的 Jacobian 函数"""
        def residual_single(param, single_obs):
            pred = self.forward_model(param.unsqueeze(0)).squeeze(0)
            return pred - single_obs

        self.jacobian_fn = vmap(jacfwd(residual_single), in_dims=(0, 0))

    def _to_params(self, z):
        """z (无约束) → 物理参数 (有界)，纯线性映射"""
        s = torch.sigmoid(z)
        return s * self.scale + self.low

    def _dp_dz(self, z):
        """∂params/∂z，用于链式法则转换 Jacobian"""
        s = torch.sigmoid(z)
        return s * (1 - s) * self.scale

    def _to_norm(self, params):
        """物理参数 → 归一化 (0, 1)，纯线性"""
        return ((params - self.low) / self.scale).clamp(0.02, 0.98)

    def solve(
        self,
        obs,
        init_params,
        n_iters=80,
        lam_init=1e-2,
        lam_up=10.0,
        lam_down=10.0,
        max_lam=1e8,
        min_lam=1e-12,
        tol_loss=1e-12,
        tol_param=1e-10,
        patience=15,
        grad_clip=10.0,
        verbose=False,
    ):
        B, L = obs.shape
        P = self.P
        device = self.device

        # 初值映射到无约束空间
        p_norm = self._to_norm(init_params)
        z = torch.logit(p_norm)

        obs_gpu = obs.to(device).float()

        # 逐样本状态
        lam         = torch.full((B, 1), lam_init, device=device)
        prev_loss   = torch.full((B,), float('inf'), device=device)
        stall_count = torch.zeros(B, dtype=torch.long, device=device)
        converged   = torch.zeros(B, dtype=torch.bool, device=device)

        loss_history = []

        for it in range(n_iters):
            if converged.all():
                break

            with torch.no_grad():
                params = self._to_params(z)
                pred = self.forward_model(params)
                r = pred - obs_gpu
                loss = (r ** 2).sum(dim=1)

            loss_mean = loss[~converged].mean().item()

            if verbose and it % 10 == 0:
                print(f'  iter={it:3d}  loss={loss_mean:.10f}  '
                      f'λ={lam.mean().item():.2e}  '
                      f'conv={converged.sum().item()}/{B}')

            with torch.no_grad():
                J_params = self.jacobian_fn(params, obs_gpu)
                dpdz = self._dp_dz(z)
                J = J_params * dpdz.unsqueeze(1)

                JT = J.transpose(1, 2)
                H = JT @ J
                g = JT @ r.unsqueeze(-1)

                # 梯度裁剪
                g_norm = g.squeeze(-1).norm(dim=1, keepdim=True)
                g_scale = torch.where(
                    g_norm > grad_clip,
                    grad_clip / (g_norm + 1e-15),
                    torch.ones_like(g_norm),
                )
                g = g * g_scale.unsqueeze(-1)

                # 阻尼
                H_diag = H.diagonal(dim1=1, dim2=2)
                H_damped = H + torch.diag_embed(lam * (H_diag + 1e-8))

                # 求解
                try:
                    delta = torch.linalg.solve(
                        H_damped, g
                    ).squeeze(-1).reshape(B, P)
                except Exception:
                    delta = torch.linalg.lstsq(
                        H_damped, g
                    ).solution.squeeze(-1).reshape(B, P)

                # 预测下降
                Jd = (J @ delta.unsqueeze(-1)).squeeze(-1)
                Hd = (H @ delta.unsqueeze(-1)).squeeze(-1)
                predicted = (Jd ** 2).sum(dim=1) + (delta * Hd).sum(dim=1)

                # 试探
                z_new = z - delta
                new_params = self._to_params(z_new)
                new_pred = self.forward_model(new_params)
                new_loss = ((new_pred - obs_gpu) ** 2).sum(dim=1)

                # 增益比
                rho = (loss - new_loss) / (predicted.abs() + 1e-15)

                # 逐样本分类
                active     = ~converged
                very_good  = (rho > 0.75)  & active
                good       = (rho > 0.25)  & (rho <= 0.75) & active
                poor       = (rho > 0.0)   & (rho <= 0.25) & active
                bad        = (rho <= 0.0)   & active

                accept = very_good | good | poor

                # z 更新
                z = torch.where(accept.unsqueeze(1), z_new, z)

                # λ 更新
                lam = torch.where(
                    very_good.unsqueeze(1), lam / lam_down,
                    torch.where(
                        poor.unsqueeze(1), lam * (lam_up ** 0.5),
                        torch.where(
                            bad.unsqueeze(1), lam * lam_up,
                            lam,
                        )
                    )
                )
                lam = lam.clamp(min=min_lam, max=max_lam)

                # loss 更新
                loss_accepted = torch.where(accept, new_loss, loss)

                # 收敛判断
                loss_unchanged = (prev_loss - loss_accepted).abs() < tol_loss
                param_changed  = delta.abs().max(dim=1).values > tol_param
                grad_small     = g.squeeze(-1).norm(dim=1) < 1e-8

                stalled     = (loss_unchanged & ~param_changed) | grad_small
                stall_count = torch.where(stalled, stall_count + 1, stall_count * 0)
                newly_done  = stall_count >= patience
                converged   = converged | newly_done

                prev_loss = loss_accepted

        # 最终参数
        with torch.no_grad():
            final_params = self._to_params(z)
            final_params = torch.clamp(
                final_params, self.low.unsqueeze(0), self.high.unsqueeze(0)
            )

        if verbose:
            print(f'\n  Done: {converged.sum().item()}/{B} converged')

        return final_params, converged, loss_history


def batch_solve_with_retry(obs_all, lm, p0, low, high,
                           target=0.95, max_rounds=10, batch_size=2000):
    N = obs_all.shape[0]
    P = len(p0)
    all_params = np.zeros((N, P))
    all_conv   = np.zeros(N, dtype=bool)

    round_cfgs = [
        {'n_iters': 80,  'patience': 25, 'lam_init': 1e-2},
        {'n_iters': 120, 'patience': 25, 'lam_init': 1e-3},
        {'n_iters': 160, 'patience': 35, 'lam_init': 1e-3},
        {'n_iters': 200, 'patience': 50, 'lam_init': 1e-4},
        {'n_iters': 300, 'patience': 60, 'lam_init': 1e-4},
    ]

    for rnd in range(max_rounds):
        mask = ~all_conv
        rate = all_conv.mean()

        if rate >= target and rnd > 0:
            break
        if mask.sum() == 0:
            break

        cfg = round_cfgs[min(rnd, len(round_cfgs) - 1)]
        n_todo = mask.sum()
        print(f'\n{"="*60}')
        print(f'Round {rnd+1}: {n_todo} remaining, '
              f'convergence={rate*100:.1f}%')
        print(f'{"="*60}')

        lo = low.to(lm.device)
        hi = high.to(lm.device)

        # 生成初值
        if rnd == 0:
            init = p0.unsqueeze(0).expand(n_todo, -1).to(lm.device)

        elif rnd <= 2:
            init = torch.tensor(all_params[mask], dtype=torch.float32, device=lm.device)
            noise = torch.randn_like(init) * 0.05 * (hi - lo)
            init = (init + noise).clamp(lo, hi)

        else:
            init = lo + torch.rand(n_todo, P, device=lm.device) * (hi - lo)

        # 逐批反演
        obs_todo = obs_all[mask]

        for start in range(0, n_todo, batch_size):
            end = min(start + batch_size, n_todo)

            obs_b  = torch.tensor(obs_todo[start:end], dtype=torch.float32)
            init_b = init[start:end]

            params_b, conv_b, _ = lm.solve(
                obs=obs_b, init_params=init_b, verbose=False, **cfg
            )

            with torch.no_grad():
                pred   = lm.forward_model(params_b.to(lm.device))
                obs_t  = obs_b.to(lm.device)
                new_loss = ((pred - obs_t) ** 2).mean(dim=1).cpu().numpy()

            todo_idx = np.where(mask)[0][start:end]

            if rnd > 0:
                old_loss = np.full(end - start, np.inf)
                has_old  = all_conv[todo_idx]
                if has_old.any():
                    old_p    = torch.tensor(all_params[todo_idx],
                                            dtype=torch.float32, device=lm.device)
                    old_pred = lm.forward_model(old_p)
                    old_loss = ((old_pred - obs_t) ** 2).mean(dim=1).cpu().numpy()
                improved = new_loss < old_loss
            else:
                improved = np.ones(end - start, dtype=bool)

            all_params[todo_idx[improved]] = params_b[improved].cpu().numpy()
            all_conv[todo_idx[improved]]   = conv_b[improved].cpu().numpy()

            torch.cuda.empty_cache()

    print(f'\n{"="*60}')
    print(f'Final: {all_conv.sum()}/{N} ({all_conv.mean()*100:.1f}%)')
    print(f'{"="*60}')

    return all_params, all_conv
#%%
import torch
import numpy as np
from torch.func import jacfwd, vmap


class FastL_BFGS:
    """
    精简版批量 L-BFGS
    - 反向模式 AD（一次反向 = 所有参数梯度）
    - 回溯线搜索（比强 Wolfe 快 3-5 倍）
    - 无冗余前向模拟
    """

    def __init__(self, forward_model, low, high, n_params, device=None):
        self.forward_model = forward_model
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.P = n_params
        self.low  = low.to(device).double()
        self.high = high.to(device).double()
        self.scale = self.high - self.low

    def _to_params(self, z):
        return torch.sigmoid(z) * self.scale + self.low

    def _to_norm(self, params):
        return ((params - self.low) / self.scale).clamp(0.02, 0.98)

    def _loss_and_grad(self, z, obs):
        """一次反向得到 loss 和梯度"""
        params = self._to_params(z)
        pred = self.forward_model(params)
        r = pred - obs
        loss_per = (r ** 2).sum(dim=1)           # (B,)
        grad = torch.autograd.grad(loss_per.sum(), z)[0]  # (B, P)
        return loss_per.detach(), grad.detach()

    def _loss_only(self, z, obs):
        """只算 loss（不需要梯度，省一半计算）"""
        with torch.no_grad():
            params = self._to_params(z)
            pred = self.forward_model(params)
            return ((pred - obs) ** 2).sum(dim=1)  # (B,)

    def solve(
        self,
        obs,
        init_params,
        n_iters=500,
        history_size=15,
        c1=1e-4,
        max_ls=15,
        tol_grad=1e-10,
        tol_loss=1e-15,
        patience=50,
        verbose=False,
    ):
        B, L = obs.shape
        P = self.P
        device = self.device

        p_norm = self._to_norm(init_params.double())
        z = torch.logit(p_norm).requires_grad_(True)
        obs_gpu = obs.to(device).double()

        # L-BFGS 历史
        S, Y, SY = [], [], []   # (B,P), (B,P), (B,)

        stall_count = torch.zeros(B, dtype=torch.long, device=device)
        converged   = torch.zeros(B, dtype=torch.bool, device=device)

        # 初始 loss 和梯度
        loss, grad = self._loss_and_grad(z, obs_gpu)

        for it in range(n_iters):
            if converged.all():
                break

            if verbose and it % 50 == 0:
                print(f'  iter={it:4d}  RMSE={torch.sqrt(loss[~converged].mean()/L):.2e}  '
                      f'conv={converged.sum().item()}/{B}')

            # ── 收敛判断 ──
            if it > 0:
                grad_norm = grad.norm(dim=1)
                loss_change = (prev_loss - loss).abs()
                stalled = (loss_change < tol_loss) | (grad_norm < tol_grad)
                stall_count = torch.where(stalled, stall_count + 1, 0)
                converged = converged | (stall_count >= patience)
                if converged.all():
                    break

            # ── L-BFGS 搜索方向 ──
            d = self._lbfgs_direction(grad, S, Y, SY)  # (B, P)

            # ── 回溯线搜索 ──
            alpha = torch.ones(B, device=device, dtype=torch.float64)
            dg = (grad * d).sum(dim=1)                  # 方向导数

            z_new = None
            loss_new = None

            for ls in range(max_ls):
                z_trial = z.detach() + alpha.unsqueeze(1) * d
                loss_trial = self._loss_only(z_trial, obs_gpu)

                # Armijo 条件
                armijo = loss_trial <= loss + c1 * alpha * dg

                # 接受的样本
                accept_ls = armijo & ~converged

                if accept_ls.any():
                    if z_new is None:
                        z_new = z.detach().clone()
                        loss_new = loss.clone()
                    z_new = torch.where(accept_ls.unsqueeze(1), z_trial, z_new)
                    loss_new = torch.where(accept_ls, loss_trial, loss_new)

                # 全部接受则退出线搜索
                if accept_ls.all():
                    break

                # 未接受的缩半步长
                alpha = torch.where(accept_ls, alpha, alpha * 0.5)

            # ── 如果线搜索全部失败，用最速下降 ──
            if z_new is None:
                d = -grad
                dg = (grad * d).sum(dim=1)
                alpha = torch.ones(B, device=device, dtype=torch.float64) * 0.001
                z_new = z.detach() + alpha.unsqueeze(1) * d
                loss_new = self._loss_only(z_new, obs_gpu)

            # ── 更新梯度（只对接受的样本）──
            z_prev = z.detach().clone()
            grad_prev = grad.clone()
            loss_prev = loss.clone()

            # 计算新梯度
            z = z_new.requires_grad_(True)
            loss, grad = self._loss_and_grad(z, obs_gpu)

            # ── L-BFGS 历史更新 ──
            s = z.detach() - z_prev                # (B, P)
            y_new = grad - grad_prev               # (B, P)
            sy = (s * y_new).sum(dim=1)            # (B,)

            valid = sy > 1e-15
            if valid.any():
                S.append(s)
                Y.append(y_new)
                SY.append(sy)
                if len(S) > history_size:
                    S.pop(0)
                    Y.pop(0)
                    SY.pop(0)

            prev_loss = loss_prev

        with torch.no_grad():
            final_params = self._to_params(z.detach()).clamp(self.low, self.high)

        if verbose:
            print(f'  Done: {converged.sum().item()}/{B} converged')

        return final_params, converged

    def _lbfgs_direction(self, grad, S, Y, SY):
        """L-BFGS 两循环递归"""
        q = -grad.clone()
        m = len(S)

        if m == 0:
            return q

        alpha_list = []
        for i in range(m - 1, -1, -1):
            a = SY[i] * (S[i] * q).sum(dim=1)
            alpha_list.append(a)
            q = q - a.unsqueeze(1) * Y[i]

        # 初始缩放
        s, y = S[-1], Y[-1]
        gamma = (s * y).sum(dim=1) / (y * y).sum(dim=1)
        r = gamma.unsqueeze(1) * q

        for i in range(m):
            b = SY[i] * (Y[i] * r).sum(dim=1)
            r = r + (alpha_list[m - 1 - i] - b).unsqueeze(1) * S[i]

        # 确保下降方向
        descent = (r * grad).sum(dim=1) < 0
        return torch.where(descent.unsqueeze(1), r, -grad)


def batch_solve_lbfgs(obs_all, lm, p0, low, high,
                      target=0.95, max_rounds=10, batch_size=2000):
    N = obs_all.shape[0]
    P = len(p0)
    all_params = np.zeros((N, P))
    all_conv   = np.zeros(N, dtype=bool)

    round_cfgs = [
        {'n_iters': 500,  'patience': 50,  'history_size': 15},
        {'n_iters': 800,  'patience': 80,  'history_size': 20},
        {'n_iters': 1200, 'patience': 100, 'history_size': 20},
        {'n_iters': 1500, 'patience': 150, 'history_size': 25},
        {'n_iters': 2000, 'patience': 200, 'history_size': 30},
    ]

    for rnd in range(max_rounds):
        mask = ~all_conv
        rate = all_conv.mean()
        if rate >= target and rnd > 0:
            break
        if mask.sum() == 0:
            break

        cfg = round_cfgs[min(rnd, len(round_cfgs) - 1)]
        n_todo = mask.sum()
        print(f'\nRound {rnd+1}: {n_todo} remaining, {rate*100:.1f}%')

        lo = low.to(lm.device).double()
        hi = high.to(lm.device).double()

        if rnd == 0:
            init = p0.double().unsqueeze(0).expand(n_todo, -1).to(lm.device)
        elif rnd <= 2:
            init = torch.tensor(all_params[mask], dtype=torch.float64, device=lm.device)
            init = (init + torch.randn_like(init) * 0.05 * (hi - lo)).clamp(lo, hi)
        else:
            init = lo + torch.rand(n_todo, P, dtype=torch.float64, device=lm.device)

        obs_todo = obs_all[mask]

        for start in range(0, n_todo, batch_size):
            end = min(start + batch_size, n_todo)
            obs_b  = torch.tensor(obs_todo[start:end], dtype=torch.float64)
            init_b = init[start:end]

            params_b, conv_b = lm.solve(obs=obs_b, init_params=init_b, verbose=True, **cfg)

            with torch.no_grad():
                pred = lm.forward_model(params_b.to(lm.device))
                new_loss = ((pred - obs_b.to(lm.device)) ** 2).mean(dim=1).cpu().numpy()

            todo_idx = np.where(mask)[0][start:end]

            if rnd > 0:
                old_loss = np.full(end - start, np.inf)
                has_old = all_conv[todo_idx]
                if has_old.any():
                    old_p = torch.tensor(all_params[todo_idx], dtype=torch.float64, device=lm.device)
                    old_loss = ((lm.forward_model(old_p) - obs_b.to(lm.device)) ** 2).mean(dim=1).cpu().numpy()
                improved = new_loss < old_loss
            else:
                improved = np.ones(end - start, dtype=bool)

            all_params[todo_idx[improved]] = params_b[improved].cpu().numpy()
            all_conv[todo_idx[improved]]   = conv_b[improved].cpu().numpy()

            del params_b, conv_b, pred, obs_b, init_b
            torch.cuda.empty_cache()

    print(f'\nFinal: {all_conv.sum()}/{N} ({all_conv.mean()*100:.1f}%)')
    return all_params, all_conv


def invert_mse(obs, p0_norm, low, high, forward_model=kan_torch,
               n_iters=400, lr=0.02, batch_size=2000,
               patience=None, device=None, grad_mode='backward',
               optimizer='adamw', scheduler='cosine', lr_min_ratio=0.05,
               lr_trace=None):
    """以 MSE(残差平方和)为目标、Adam/AdamW + 反向 autograd.grad 的批量反演.

    参数
    ----
    obs: (N, L) 观测 BRF 光谱 (float32, GPU 前会转移).
    p0_norm: (P,) 归一化初值, 取值范围 (0,1); plain 值或 logit 逆映射前先 clamp(0.02,0.98).
    low, high: (P,) 参数物理下/上界.
    forward_model: 前向函数, 输入 (B,P) 物理参数 -> (B,L) 输出光谱.
    n_iters: 每个起点的迭代数.
    lr: 优化器学习率.
    optimizer: 'adamw' (默认, torch.optim.AdamW, 解耦权重衰减 wd=0.01)
               或 'adam' (torch.optim.Adam, wd=0). 两者在 wd=0 时更新式相同,
               本函数无显式正则项, 差异仅来自权重衰减项.
    batch_size: 分批处理的样本数.
    patience: 连续 patience 步所有样本 MSE < 1e-8 时早停; None=不早停.
    device: 计算设备; None 表示跟随 `PROPBR_DEVICE` / 自动检测.
    scheduler: 'cosine' (默认, 余弦退火 lr -> lr_min_ratio*lr) 或 'none' (恒定 lr).
    lr_min_ratio: 退火终值相对初值的比例, 默认 0.05 (即 lr=0.05 -> 2.5e-3).
    lr_trace: 可选 list; 若给出, 每步迭代后追加当前学习率, 便于验证退火是否生效.
    grad_mode: 反演时的梯度来源.
        'backward' (默认): 用 torch.autograd 反向求 ∂loss/∂z, 快且解析.
        'jacfwd'    : 用 jacfwd+vmap 求逐波长 Jacobian J:(B,L,P) 再收缩 grad=J^T r.
                      数学上与 backward 等价(其结果可与 backward 对比), 速度较慢.

    返回
    ----
    params: (N, P) 反演物理参数
    """
    from torch.func import jacfwd, vmap as _vmap
    device = device or _DEFAULT_DEVICE
    _ensure(device)          # build caches outside any functorch transform
    if n_iters < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('n_iters, batch_size and lr must be positive')
    if grad_mode not in ('backward', 'jacfwd'):
        raise ValueError("grad_mode must be 'backward' or 'jacfwd'")
    N, L = obs.shape
    P = p0_norm.shape[0]
    low = torch.as_tensor(low, dtype=torch.float32, device=device)
    high = torch.as_tensor(high, dtype=torch.float32, device=device)
    scale = high - low
    obs_all = torch.as_tensor(obs, dtype=torch.float32, device=device).detach()

    # Jacobian 模式: 单样本 y_single( z:(P,) -> y:(L,) ), 供 jacfwd
    if grad_mode == 'jacfwd':
        def params_from_z_b(zb): return torch.sigmoid(zb)*scale + low
        def y_single(zp): return forward_model(params_from_z_b(zp.reshape(1,-1))).squeeze(0)
        per = jacfwd(y_single)
        def compute_grad(z, obs_b):
            J = _vmap(per)(z.detach())               # (B,L,P)
            y = _vmap(y_single)(z.detach())          # (B,L)
            r = y - obs_b
            return (2.0 / L) * torch.einsum('blp,bl->bp', J, r)  # gradient of spectral MSE
    else:
        def compute_grad(z, obs_b):
            z.requires_grad_(True)
            params = torch.sigmoid(z)*scale + low
            pred = forward_model(params)
            loss = ((pred-obs_b)**2).mean(dim=1)
            return torch.autograd.grad(loss.sum(), z)[0]   # (B,P)

    # 每个起点随机归一化初值 (n_restarts) → 多次反演选 loss 最小
    # inits = np.vstack([p0_norm.cpu().numpy() if hasattr(p0_norm,'cpu') else np.asarray(p0_norm,dtype='float32')] +
    #                   [rng.uniform(0.02,0.98,size=P) for _ in range(n_restarts-1)])

    best_params = torch.full((N, P), float('nan'), device=device)
    best_loss = torch.full((N,), float('inf'), device=device)   # 每样本历史最低 loss (用于跨起点选优)

    # for istart in range(len(inits)):
    z0 = torch.logit(torch.as_tensor(p0_norm, dtype=torch.float32, device=device).clamp(0.02,0.98))
    print(f'\n[start] ', flush=True)
    for b_start in range(0, N, batch_size):
        b_end = min(N, b_start+batch_size)
        n_b = b_end-b_start
        z = z0.unsqueeze(0).expand(n_b, -1).clone().requires_grad_(True)
        if optimizer == 'adamw':
            opt = torch.optim.AdamW([z], lr=lr)
        elif optimizer == 'adam':
            opt = torch.optim.Adam([z], lr=lr)
        else:
            raise ValueError("optimizer must be 'adamw' or 'adam', got %r" % (optimizer,))
        if scheduler == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, n_iters), eta_min=lr * lr_min_ratio)
        elif scheduler in (None, 'none'):
            sched = None
        else:
            raise ValueError("scheduler must be 'cosine' or 'none', got %r" % (scheduler,))
        obs_b = obs_all[b_start:b_end]
        stall = 0
        best_b = torch.full((n_b,), float('inf'), device=device)
        best_z = z.detach().clone()
        for it in range(n_iters):
            params = torch.sigmoid(z)*scale + low
            pred = forward_model(params)
            loss = ((pred-obs_b)**2).mean(dim=1)         # (n_b,), 逐样本 MSE
            # The saved state and its loss must come from the same evaluation.
            with torch.no_grad():
                better = loss < best_b
                best_b = torch.where(better, loss.detach(), best_b)
                best_z = torch.where(better.unsqueeze(1), z.detach(), best_z)
            # 按 grad_mode 取梯度
            if grad_mode == 'jacfwd':
                g = compute_grad(z, obs_b).detach()      # (B,P) J^T r
                if z.grad is not None: z.grad.detach_().zero_()
                z.grad = g
            else:
                opt.zero_grad(set_to_none=True)
                loss.sum().backward()
                if not torch.isfinite(z.grad).all():
                    z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=1e6, neginf=-1e6)
            opt.step()
            if sched is not None:
                sched.step()
            if lr_trace is not None:
                lr_trace.append(opt.param_groups[0]['lr'])
            # 早停
            if patience is not None:
                stall = int((loss.detach().abs() < 1e-8).all().item())* (stall+1)
                if stall >= patience: break
        # Include the last optimizer update, which has not yet been evaluated.
        with torch.no_grad():
            last_params = torch.sigmoid(z) * scale + low
            last_loss = ((forward_model(last_params) - obs_b) ** 2).mean(dim=1)
            better = last_loss < best_b
            best_b = torch.where(better, last_loss, best_b)
            best_z = torch.where(better.unsqueeze(1), z.detach(), best_z)
        params_b = torch.sigmoid(best_z)*scale + low
        # 跨起点取最优 (用最低 loss 判据)
        upd = best_b < best_loss[b_start:b_end]
        best_loss[b_start:b_end] = torch.where(upd, best_b, best_loss[b_start:b_end])
        best_params[b_start:b_end] = torch.where(upd.unsqueeze(1), params_b, best_params[b_start:b_end])

        print(f'  batch[{b_start}:{b_end}] Done', flush=True)


    return best_params.detach().cpu().numpy()
