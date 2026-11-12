# PROPBR

**Differentiable radiative transfer with consistent directional and diffuse reflection for leaf BRF simulation and biochemical trait retrieval.**

PROPBR couples the Smith–GGX microfacet formulation from physically based rendering
(PBR) with the leaf radiative transfer model PROSPECT-D to simulate leaf
bidirectional reflectance factors (BRFs) through a differentiable computational graph.

- **PROPBR** explicitly combines surface specular reflection and internal diffuse
  reflection. A Kolmogorov–Arnold network (KAN) approximates the specular kernel
  `K_s` from a precomputed 2D lookup table, providing a differentiable surrogate
  for batched GPU inversion with PyTorch automatic differentiation.
- **sPROPBR** uses an effective surface reflection coefficient μ in place of the
  geometric terms. It supports proximal hyperspectral imaging (HSI), including
  pixel-level chlorophyll mapping, when complete observation geometry is unavailable.

This repository contains the core code for **forward BRF simulation** and
**parameter inversion**.

---

## Repository structure

```text
.
├── README.md
├── environment.yml      # Conda environment
├── func.py              # PROPBR/sPROPBR forward models, reference specular removal,
│                        # and batched Adam/AdamW inversion
├── prospect.py          # NumPy PROSPECT-D/PROSPECT-PRO implementation used by func.py
├── prospect_h.py        # PROSPECT wrapper for HSI
├── kan_model.py         # KsKAN surrogate: 2-4-1 architecture, B-splines, grid=10
├── trf_solver.py        # Optional batched PyTorch GPU trust-region dogleg solver
├── metric.py            # RMSE and R² evaluation
├── verify_fixes.py      # Optical references, AD, bounds, best iterates and inversion checks
├── prosailt/            # PyTorch PROSPECT-5/D/PRO implementation used by kan_torch
└── lut/                 # Model weights and precomputed lookup tables
    ├── ks_kan_weights1.pt   # KsKAN weights
    ├── DLM_nr.npy           # Refractive-index spectrum
    └── Resampler.npy        # Resampling matrix: 2101 to 196 bands
```

Measured datasets are distributed separately; see [Data](#data).

---

## Installation

```bash
conda env create -f environment.yml
conda activate propbr

# For GPU use, select a CUDA build for your system from the PyTorch installation page:
# https://pytorch.org/get-started/locally/
```

Alternatively, install the dependencies in an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

Select a CPU or CUDA build of PyTorch using the
[official installation page](https://pytorch.org/get-started/locally/).
The upstream `prosail` package is optional: `python -m pip install prosail`.

> **Device selection.** The module selects `cuda` when available and otherwise
> uses `cpu`. Set `PROPBR_DEVICE=cpu` to override this choice. Forward and inversion
> functions accept an explicit `device=` argument. On a CPU-only system, import
> `func` normally and use `device='cpu'`.
>
> Before applying `torch.func.jacfwd` or `vmap` directly to a forward function,
> call `func.warmup(device)`. Module construction and in-place parameter changes
> through `nn.Module.to` must occur outside these transforms. `invert_mse` performs
> this warm-up internally.
>
> Run `python verify_fixes.py --device all` to check the available CPU/GPU devices.
> Use `--device cpu` or `--device cuda` to select one device, or set
> `PROPBR_DEVICE=cpu`. Model assets used by `func.py` are located relative to the
> module, so imports and verification do not require the repository root to be
> the current working directory.
>
> The NumPy wrappers use the upstream `prosail` spectral library when installed
> and otherwise use the bundled `prosailt` coefficients, stored as float32 arrays.
> A local validation environment used Windows, Python 3.10, PyTorch 2.1.0 with
> CUDA, and prosail 2.0.5.

---

## Quick start

### 1) Forward BRF simulation

PROPBR uses `[N, Cab, Car, Anth, Cw, Cm, Cbrown, σ]` with the empirical relation
`f_s = 0.66·σ^-0.63`. The 9-parameter variant appends a freely estimated `f_s`.
sPROPBR uses `[N, Cab, Car, Anth, Cw, Cm, Cbrown, μ]`, where μ is the effective
surface reflection coefficient. A single-angle output has shape `(B, 2101)`,
covering 400–2500 nm at 1 nm intervals.

| Forward function | Model | Geometry and output |
|---|---|---|
| `kan_torch` | 8-parameter PROPBR; fs determined by σ | Phase angle 15°; `(B, 2101)` |
| `kan_torch_lF` | 9-parameter PROPBR; free fs | Phase angle 35°; `(B, 2101)` |
| `full_torch` | 9-parameter PROPBR; analytical specular kernel | Phase angle 35°; `(B, 2101)` |
| `full_torch_l` | 9-parameter PROPBR; analytical specular kernel | 24 directions at 12 phase angles from 10–65°; 450–1650 nm; flattened to `(B, 24*1201)` |
| `spropbr_torch` | 8-parameter sPROPBR; μ replaces σ | No geometric input; `(B, 2101)` |

`Kf_torch` remains an alias of `spropbr_torch` for compatibility.

PROPBR evaluates `DHR − ρ_ref(σ) + Ks·F_Schlick`, where
`ρ_ref(σ) = F0(n_internal)·exp(−1.26σ^1.79)`. The internal refractive index comes
from the PROSPECT spectral library, and the surface refractive index is stored in
`lut/DLM_nr.npy`. Following Eq. (9) of the manuscript,
`F_Schlick = F0(n_surface) + (1−F0(n_surface))·(1−cos(phase/2))^5`.
The KAN output Ks already includes the empirical coefficient fs. The free-fs
variant rescales it as `Ks·(fs/f(sigma))`.

sPROPBR evaluates `DHR − [1−tav(40°, n_internal)] + μ·F0(n_surface)`.
Near zero absorption, PROSPECT layer stacking uses a stable expansion about the
lossless limit that preserves derivatives with respect to absorption parameters.

```python
import torch, func

device = 'cuda' if torch.cuda.is_available() else 'cpu'
p8 = torch.tensor([[1.5, 40.0, 10.0, 0.0, 0.01, 0.005, 0.0, 0.3]], device=device)
brf = func.kan_torch(p8)          # (1, 2101), PROPBR at 15°

# CPU example
p8_cpu = p8.cpu()
brf_cpu = func.kan_torch(p8_cpu)  # Or explicitly: func.kan_torch(p8_cpu, device='cpu')
print(brf.shape, float(brf.mean()))

p9 = torch.cat([p8, p8.new_tensor([[1.5]])], dim=1)
brf9 = func.kan_torch_lF(p9)      # (1, 2101), PROPBR with free f_s

q8 = p8.clone()
q8[:, 7] = 1.2                   # The 8th column is μ
sbrf = func.spropbr_torch(q8)     # (1, 2101), sPROPBR
```

Parameter bounds used in the manuscript:

| Parameter | N | Cab | Car | Anth | Cw | Cm | Cbrown | σ | f_s |
|---|---|---|---|---|---|---|---|---|---|
| Lower bound | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.6 |
| Upper bound | 3.5 | 120 | 30 | 40 | 0.06 | 0.05 | 1.0 | 1.0 | 3.5 |

sPROPBR uses the same bounds for its first 7 parameters and an inversion range
of 0–30 for μ.

### 2) Parameter inversion

`func.invert_mse` performs batched inversion using sigmoid reparameterization
for bounded parameters and reverse-mode automatic differentiation:

```python
import numpy as np, torch, func

low  = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0.01], device=device)
high = torch.tensor([3.5, 120, 30, 40, 0.06, 0.05, 1., 1.], device=device)

obs = brf.detach()                # Runnable synthetic example; measured BRFs also use (B, 2101)
p0  = np.array([0.2, 0.33, 0.33, 0.0, 0.17, 0.1, 0.0, 0.3])   # Normalized initial values

paras = func.invert_mse(obs=obs, p0_norm=p0, low=low, high=high,
                        n_iters=500, lr=0.05, batch_size=1000,
                        optimizer='adam', device=device)
# paras: (B, 8), physical parameter values
# For CPU use, create low/high/obs on the CPU and pass device='cpu'.
```

- The example explicitly selects Adam, 500 iterations and an initial learning
  rate of 0.05, with cosine annealing to 0.0025. The function defaults are AdamW,
  400 iterations and a learning rate of 0.02. AdamW uses the PyTorch default weight
  decay of 0.01; Adam uses zero weight decay.
- For sPROPBR inversion, pass `forward_model=func.spropbr_torch` and use 0 and 30
  as the bounds of the last parameter. For joint multi-angle inversion, pass
  `forward_model=func.full_torch_l` with the corresponding 9-parameter bounds.
  Flatten the 24 directions for each leaf into one row, ordered as two groups
  of phase angles from 10–65°. The lowest-MSE solution is retained independently
  for each sample over all iterates, including the final update.
- The optional `trf_solver.batch_trf` implements a batched GPU dogleg trust-region
  solver. It can converge faster for individual samples, but yielded lower trait
  retrieval accuracy than Adam in the manuscript's joint multi-angle experiments.

---

## Runtime comparison: 2,244 orchard spectra × 2,101 bands

The table reports the experiments recorded in the manuscript. The revised
implementation uses the Schlick term and corrected best-iterate tracking;
trait retrieval metrics and timings for this revision require rerunning the
original datasets and experimental configurations.

Protocol: 9-parameter PROPBR at 35°, with the reference specular component
`F0(n)·exp(−1.26σ^1.79)` removed; Adam with 500 iterations and batch size 1000.
The `curve_fit` baseline uses SciPy TRF and the repository's NumPy PROSPECT
implementation. Cab/Car are reported in μg/cm² and EWT/LMA in mg/cm².

| Method | Time | Per spectrum | Pooled RMSE | Median RMSE | Cab | Car | EWT | LMA |
|---|---|---|---|---|---|---|---|---|
| **GPU-Adam** (RTX 4070) | **27.6 s** | **12.3 ms** | 0.00449 | 0.00386 | **13.52** | 5.31 | **1.86** | 3.09 |
| CPU-Adam (20 cores) | 322.3 s | 143.6 ms | 0.00449 | 0.00386 | **13.52** | 5.31 | **1.86** | 3.09 |
| curve_fit, single core | 823.9 s | 367.1 ms | **0.00396** | **0.00327** | 13.89 | **4.60** | 1.90 | **2.74** |
| curve_fit, 20 cores | 191.3 s | 85.2 ms | **0.00396** | **0.00327** | 13.89 | **4.60** | 1.90 | **2.74** |

GPU batched inversion was 11.7× faster than the same algorithm on the CPU and
29.9× faster than single-core SciPy TRF. On the single-angle dataset, TRF achieved
slightly better spectral fits, whereas Adam performed better for Cab/EWT.
Differences of 2–13% between the approaches reflect the different solutions
reached by the solvers.

---

## Data

Measured datasets are not included. The manuscript uses:

- **#LOTUS**: multi-angle BRFs from 255 leaves measured in 24 directions;
  Federated Research Data Repository, Peters and Noble (2022, 2024).
- **#Global / #Orchard**: fixed-angle leaf BRFs compiled from public databases,
  including EcoSIS; see the manuscript references.
- **#Apple / #Grape**: proximal HSI acquired by the authors or obtained from
  the public Data INRAE repository.

For full-spectrum input, store spectra as a float32 array of shape `(N, 2101)`,
covering 400–2500 nm at 1 nm intervals, and pass them to `func.invert_mse`.
Use the trait units specified in the manuscript for ground-truth comparisons.

---

## Citation and license

The code retains this repository's GPL-3.0 license; see [LICENSE](LICENSE).
The efficient-kan MIT license and other source acknowledgments are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Each dataset retains its original
license. Software citation metadata are provided in [CITATION.cff](CITATION.cff).
