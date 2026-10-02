# PROPBR and sPROPBR

Differentiable leaf bidirectional reflectance models in PyTorch. This repository connects forward simulation, bounded inversion and sensitivity analysis through the same model implementation.

**No measured datasets are included.** The demonstration generates synthetic spectra at runtime. The small bundled assets are PROSPECT-D numerical optical coefficients, the author's DLM 2.1 surface refractive-index spectrum, and the frozen Ks KAN weights required by the model.

中文说明见 [README_zh.md](README_zh.md)。Equation mappings and differences from historical experiment scripts are recorded in [docs/model_and_reproducibility.md](docs/model_and_reproducibility.md).

## Install

Use Python 3.10 or newer in a clean virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

This installs EFAST support through SALib. To install only the core model, use `pip install -e .`; add `--skip-global` when running the demo. For NVIDIA GPUs, install a CUDA-enabled PyTorch build following the [official installation instructions](https://pytorch.org/get-started/locally/) before installing this package.

## Run the complete synthetic workflow

```bash
python run_pipeline.py
```

Equivalent installed command:

```bash
python -m propbr demo --output outputs/demo --device cpu
```

The workflow runs:

1. Fixed-angle PROPBR simulation and inversion.
2. Joint PROPBR inversion of three viewing directions sharing one parameter vector per leaf.
3. sPROPBR simulation and inversion without geometric input.
4. Physical, range-scaled and sigmoid-latent Jacobians, plus the normalized Jacobian response correlation matrix.
5. KAN validation against analytical Ks on the training grid and an offset fine grid.
6. Validation of the reference hemispherical approximation against numerical quadrature.
7. EFAST global sensitivity for PROPBR, including phase angle, and for sPROPBR.

The default demo uses two synthetic leaves, 400–2500 nm at 10 nm intervals, float64, AdamW with 1,200 iterations and three starts, and EFAST with N=1,000 at 50 nm intervals. It writes NPZ results, JSON settings/metrics and PNG figures under `outputs/demo/`. These sampling choices make the demonstration convenient; they are not a substitute for the paper's full measured-data experiments.

```bash
# Full 1 nm simulation / inversion grid
python run_pipeline.py --step 1 --global-step 10

# NVIDIA GPU
python run_pipeline.py --device cuda

# Shorter demonstration; this checks execution, not optimization convergence
python run_pipeline.py --iterations 200 --restarts 1 --skip-global

# Analytical surface kernel in place of the frozen KAN
python run_pipeline.py --kernel analytic
```

## Model API

```python
import torch
from propbr import PROPBR, SPROPBR, invert, local_sensitivity, response_correlation

# Parameter order: N, Cab, Car, Anth, Cw, Cm, Cbrown, sigma
p = torch.tensor([[1.8, 48., 12., 3., .017, .009, .12, .28]], dtype=torch.float64)
model = PROPBR(phase_angles=[15., 35., 55.], kernel="kan")
brf = model.simulate(p)                    # (leaves, angles, wavelengths)
fit = invert(model, brf.detach(), n_iters=1200, n_restarts=3)
J = local_sensitivity(model, fit.parameters)
coupling = response_correlation(J["physical"])

# Parameter order: N, Cab, Car, Anth, Cw, Cm, Cbrown, mu
smodel = SPROPBR()
q = p.clone()
q[:, 7] = 1.2
sbrf = smodel.simulate(q)                   # (leaves, 1, wavelengths)
```

`simulate()` checks input bounds; `forward()` is the tensor-only entry point used inside automatic differentiation. Move the model with `.to("cuda")` and construct input tensors on the same device/dtype. Spectral constants and KAN weights follow the model.

### Parameter definitions

| Parameter | Meaning | Unit | Inversion range |
|---|---|---|---|
| N | Leaf structure parameter | dimensionless | 1–3.5 |
| Cab | Chlorophyll a+b content | μg cm⁻² | 0–120 |
| Car | Carotenoid content | μg cm⁻² | 0–30 |
| Anth | Anthocyanin content | μg cm⁻² | 0–40 |
| Cw | Equivalent water thickness (EWT) | cm | 0–0.06 |
| Cm | Leaf mass per area (LMA) | g cm⁻² | 0–0.05 |
| Cbrown | Brown pigment parameter | dimensionless | 0–1 |
| sigma | PROPBR surface roughness | dimensionless | 0.01–1 |
| mu | sPROPBR effective surface reflection coefficient | dimensionless | 0–30 |

Cw in cm equals water mass per area in g cm⁻² when water density is 1 g cm⁻³; multiplying Cw or Cm by 1,000 gives mg cm⁻². The paper's sPROPBR EFAST range is **mu=0–2.5**, distinct from its inversion range of 0–30.

For calibration of the empirical fsurf–sigma relationship, use `PROPBR(kernel="analytic", free_fsurf=True)`: fsurf is appended as the ninth parameter with bounds 0.6–3.5. The default eight-parameter PROPBR instead sets fsurf=0.66 sigma⁻⁰·⁶³ and does not optimize it independently. Calibrating this empirical relationship from real retrievals requires external data.

### Geometries

The default model implements the paper's reduced phase-angle geometry. The phase angle is supplied in degrees. The KAN domain is 5–90° and sigma=0.01–1; model evaluation accepts **5≤phase<90°**. Use `kernel="analytic"` for phase angles below 5°. The paper's evaluated phase angles are mainly 0–65°.

An analytical full-geometry surface term is also available:

```python
model = PROPBR(kernel="analytic", geometry=[[0., 15., 0.], [0., 35., 0.]])
# Each row: incidence zenith, viewing zenith, relative azimuth, all in degrees.
```

The diffuse correction remains h=1, as in Eq.20. The full-geometry option changes only the surface term and does not establish validity under arbitrary oblique illumination. Angle conventions follow the paper's Eq.5; phase angle alone cannot represent arbitrary illumination–viewing geometry.

### Supply your own data

NPZ files are read with `allow_pickle=False`. See [examples/create_input.py](examples/create_input.py) for a synthetic schema example.

| Workflow | Required NPZ arrays | Optional arrays |
|---|---|---|
| simulate | `parameters` (B,P), `wavelengths` (L) | — |
| invert | `brf` (B,L) or (B,A,L), `wavelengths` (L) | `initial` (P) or (B,P), `mask` (L), (B,L) or (B,A,L) |
| sensitivity | `parameters` (B,P), `wavelengths` (L) | — |

Wavelengths must increase within 400–2500 nm. The model interpolates optical coefficients at the requested wavelengths. This is point sampling, not convolution with an instrument spectral response function. NaN/Inf observations and bands where `mask=False` are excluded; every leaf needs at least one valid observation.

```bash
python examples/create_input.py --output outputs/example_input
python -m propbr simulate --input outputs/example_input/parameters.npz --phase 15,35,55
python -m propbr invert --input outputs/example_input/observations.npz --phase 15,35,55
python -m propbr sensitivity --input outputs/example_input/parameters.npz --phase 15,35,55
python -m propbr sensitivity --input outputs/example_input/parameters.npz --kernel analytic --global-analysis --include-phase
```

sPROPBR has one independent effective coefficient per observed spectrum. To invert multi-angle data with sPROPBR as in the paper, reshape `(leaves, angles, bands)` to `(leaves*angles, bands)`; joint biochemical sharing is the PROPBR design.

## Validate and retrain the surrogate

```bash
python -m propbr validate --output outputs/validation
python -m propbr train-kan --epochs 105000 --device cuda --output outputs/training/ks_kan.pt
```

Training regenerates the analytical LUT rather than reading a dataset. It uses the documented 2→4→1 network, cubic B-splines, grid size 10, standardized inputs and log1p(Ks) targets. A short training run verifies the training path but will not reproduce the supplied weight's accuracy. Load a fresh checkpoint through `PROPBR(weights_path="outputs/training/ks_kan.pt")`; the bundled checkpoint is preserved.

## Tests

```bash
python -m pip install -e ".[full,dev]"
python -m pytest -q
```

Tests check PROSPECT-D against independent NumPy PROSAIL, exponential-integral values and derivatives against SciPy, analytical surface equations, physical/latent Jacobians against finite differences and direct AD, missing-band inversion, EFAST sampling counts, and CPU/CUDA agreement when CUDA is available. GitHub Actions runs CPU tests on Windows and Linux.

Actual local verification outcomes are recorded in [docs/verification.md](docs/verification.md).

## Scientific scope and licensing

The public implementation follows the manuscript equations. It corrects repeated fsurf multiplication, restores angular Schlick Fresnel reflectance, and uses the roughness-dependent reference subtraction in PROPBR; these differ from some historical scripts. **Previously reported measured-data metrics and speedups must be evaluated again with the selected implementation and the external datasets.** No measured-data metrics are inferred from the synthetic examples.

Code and the adapted PROSPECT implementation are distributed under GPLv3; the vendored efficient-kan implementation retains its MIT notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Asset hashes and provenance are recorded in `src/propbr/assets/manifest.json`.

The manuscript has no public DOI recorded in this package. Use `CITATION.cff` for this software, and cite the published PROSPECT-D and KAN methods when appropriate. NLSQ/JAXFit are references for the manuscript's GPU least-squares comparator; this package's inversion uses PyTorch AdamW and does not depend on those solvers.
