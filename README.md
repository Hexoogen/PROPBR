# PROPBR

**Differentiable leaf BRF simulation and biochemical trait retrieval.**

PROPBR couples PROSPECT-D with a Smith–GGX microfacet model to combine internal
diffuse reflection and directional surface reflection. PyTorch automatic
differentiation supports forward simulation, sensitivity analysis and batched
parameter inversion.

PROPBR uses 8 parameters:

```text
[N, Cab, Car, Anth, Cw, Cm, Cbrown, σ]
```

Two observation configurations are provided: a fixed phase angle and
joint simulation of **24 directions**. The simplified model, **sPROPBR**, replaces
σ with an effective surface reflection coefficient μ for observations without
complete geometric information, such as proximal hyperspectral imaging.

## Repository structure

```text
.
├── README.md
├── environment.yml      # Conda environment
├── requirements.txt     # Python dependencies
├── func.py              # Forward models and batched Adam inversion
├── prospect.py          # NumPy PROSPECT implementation
├── kan_model.py         # Differentiable KAN approximation of the directional kernel
├── metric.py            # Retrieval accuracy metrics
├── verify_fixes.py      # Numerical and gradient verification
├── prosailt/            # PyTorch leaf optical model and spectral coefficients
└── lut/
    ├── ks_kan_weights1.pt   # Trained directional-kernel weights
    └── DLM_nr.npy           # Surface refractive-index spectrum
```

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate propbr
```

For an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

Select a CPU or CUDA build of PyTorch using the
[official installation page](https://pytorch.org/get-started/locally/).
The repository includes the spectral coefficients required by the models.

The module selects CUDA when available and otherwise uses the CPU. Set
`PROPBR_DEVICE=cpu` to select the CPU explicitly. Model assets are located
relative to the module, so imports do not depend on the current working directory.

## Forward simulation

| Function | Model parameters | Geometry and output |
|---|---|---|
| `propbr_35_torch` | 8 parameters ending in σ | Phase angle 35°; `(B, 2101)`, covering 400–2500 nm |
| `propbr_24_torch` | The same 8 parameters | 24 directions at 12 phase angles from 10–65°; `(B, 24*1201)`, covering 450–1650 nm per direction |
| `spropbr_torch` | 8 parameters ending in μ | No geometric input; `(B, 2101)`, covering 400–2500 nm |

Wavelength spacing is 1 nm. For the multi-angle output, directions are ordered
as two groups of phase angles from 10° to 65° in 5° steps. All directions for
one leaf share the same parameter vector.

```python
import os
import torch
import func

device = os.environ.get('PROPBR_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
p = torch.tensor([[1.5, 40.0, 10.0, 0.0, 0.01, 0.005, 0.0, 0.3]], device=device)

brf35 = func.propbr_35_torch(p)     # (1, 2101)
brf24 = func.propbr_24_torch(p)     # (1, 24*1201)
angular_brf = brf24.reshape(-1, 24, 1201)

q = p.clone()
q[:, 7] = 1.2                     # Effective surface reflection coefficient μ
sbrf = func.spropbr_torch(q)       # (1, 2101)
```

PROPBR parameter bounds:

| Parameter | N | Cab | Car | Anth | Cw | Cm | Cbrown | σ |
|---|---|---|---|---|---|---|---|---|
| Lower bound | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 |
| Upper bound | 3.5 | 120 | 30 | 40 | 0.06 | 0.05 | 1.0 | 1.0 |

sPROPBR uses the same bounds for the first 7 parameters and a range of 0–30 for μ.

## Parameter inversion

`invert_mse` uses sigmoid reparameterization to keep physical parameters within
their bounds. The example uses Adam with 500 iterations and cosine annealing
from a learning rate of 0.05 to 0.0025.

```python
import numpy as np

low = np.array([1., 0, 0, 0, 0, 0, 0, 0.01])
high = np.array([3.5, 120, 30, 40, 0.06, 0.05, 1., 1.])
initial = np.array([0.2, 0.33, 0.33, 0.0, 0.17, 0.1, 0.0, 0.3])

retrieved35 = func.invert_mse(
    obs=brf35.detach(), p0_norm=initial, low=low, high=high,
    forward_model=func.propbr_35_torch,
    n_iters=500, lr=0.05, batch_size=1000,
    optimizer='adam', scheduler='cosine', device=device,
)

retrieved24 = func.invert_mse(
    obs=brf24.detach(), p0_norm=initial, low=low, high=high,
    forward_model=func.propbr_24_torch,
    n_iters=500, lr=0.05, batch_size=1000,
    optimizer='adam', scheduler='cosine', device=device,
)
```

Both outputs have shape `(B, 8)`. Joint inversion fits all 24 directions of each
leaf using one shared parameter vector. The lowest-MSE solution is retained
independently for every sample over all iterates, including the final update.

For sPROPBR, pass `forward_model=func.spropbr_torch` and the corresponding BRF
array. Set the last parameter bounds to 0 and 30 and provide its normalized
initial value.

## Numerical verification

```bash
python verify_fixes.py --device all
```

Use `--device cpu` or `--device cuda` to select a device. The checks cover optical
reference calculations, forward- and reverse-mode derivatives, zero-absorption
limits, parameter bounds, the 8-parameter interfaces, and synthetic inversions.
Before applying `torch.func.jacfwd` or `vmap` directly to a forward function,
call `func.warmup(device)`; `invert_mse` performs this warm-up internally.

## Citation and license

The code retains this repository's GPL-3.0 license; see [LICENSE](LICENSE).
The efficient-kan MIT license and other source acknowledgments are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Each dataset retains its original
license. Software citation metadata are provided in [CITATION.cff](CITATION.cff).
