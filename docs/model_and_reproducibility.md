# Model equations and reproducibility

## Equation map

| Manuscript expression | Implementation |
|---|---|
| PROSPECT-D DHRF | `prospect.prospect_d` |
| Eqs.2–7 Smith–GGX surface and geometry | `surface.full_geometry_ks` |
| Eqs.8–10 surface multiplier and Schlick Fresnel factor | `surface.surface_factor`, `schlick`, `fresnel0` |
| Eqs.11–14 reduced geometry and Ks | `surface.reduced_ks`, `kan.KsKAN` |
| Reference hemispherical approximation, Sec.4.1.3 | `surface.reference_specular` |
| Eq.20 PROPBR | `models.PROPBR.components` |
| Eq.21 sPROPBR | `models.SPROPBR.components` |
| Eqs.27–32 sigmoid parameterization and chain rule | `parameters.ParameterSpace`, `inversion.invert` |
| Eq.33 local response | `sensitivity.local_sensitivity` physical Jacobian |
| Eq.34 normalized response correlation | `sensitivity.response_correlation` |
| EFAST, Sec.3.4.1 | `sensitivity.global_sensitivity` |

For reduced geometry,

`BRF_PROPBR = DHRF_PROSPECT − F0(n_in)*exp(−1.26*sigma**1.79) + Ks(phase,sigma)*F_Schlick(n_surf,phase)`.

`Ks = pi*fsurf*D*G/(4*cos(phase))`, with `fsurf=.66*sigma**(-.63)`.

For unknown geometry,

`BRF_sPROPBR = DHRF_PROSPECT − [1−tav(40,n_in)] + mu*F0(n_surf)`.

Internal and surface refractive-index spectra are separate model constants. The reference hemispherical approximation was calibrated with an **unscaled** microfacet kernel and n_in. The empirical fsurf multiplier belongs to the surface directional term; it is not multiplied into the reference fit.

## Corrections relative to historical code

Some historical experiment functions used `diffuse + fsurf*KAN(Ks)*F0`, although the KAN was trained on a target already containing fsurf. This repository instead uses `Ks*F_Schlick`, applying fsurf exactly once, and restores the angular term. PROPBR also uses its own roughness-dependent reference subtraction rather than the sPROPBR interface approximation.

The supplied trained tensors and standardization values are unchanged. The checkpoint container is sanitized to retain tensors and primitive metadata, and historic metric labels are removed. The original weight SHA-256 and the public checkpoint SHA-256 are both recorded in the asset manifest.

The validation command distinguishes all training-grid values, the filtered training region 0.1≤Ks≤50, and the offset fine grid. A training-grid fit score is not an independent test score.

## Inversion and derivative semantics

The default objective is spectral MSE over all valid bands and angles for each leaf. AdamW acts on unconstrained variables z with the physical map `x=lower+sigmoid(z)*(upper−lower)`. Weight decay defaults to zero to avoid adding an implicit prior to the data objective. The first restart uses the paper's table initial values; the other starts are seeded random physical values. The best iterate is retained for every restart, and the best restart is selected independently for each leaf.

Initial values on a boundary are mapped to normalized values 0.001 or 0.999 to give finite logits. This is only an initialization convention; the forward model accepts the closed physical bounds. Physical Jacobians are evaluated at the supplied physical values. Latent Jacobians are evaluated at `decode(encode(parameters))`, which can differ from a boundary value; `latent_parameters` records the actual evaluation points.

The response correlation is calculated from the full angle/wavelength Jacobian per leaf. It is neither a correlation of retrieved traits nor a Jacobian averaged across wavelengths. `J^T J` can be interpreted as Fisher information only after a noise model is specified; under independent unit-variance noise it is the corresponding information matrix. This code does not claim independent parameter recovery from spectral fit alone.

## Sampling and limits

The frozen KAN is calibrated at 5–90° and sigma=0.01–1. The forward reduced model rejects 90° and any extrapolation below 5° for KAN. The analytical formula is available below 5°. Values at exactly 90° used for surrogate validation/training are calculated through the finite analytical limit of the Smith visibility ratio.

The diffuse correction is h=1. Full-geometry surface evaluation does not implement or validate h under arbitrary oblique incidence. An empirical specular multiplier is not a guarantee of energy conservation throughout all possible geometries and parameter combinations. BRF is a directional reflectance factor and is not clipped to [0,1].

The native PyTorch exponential integral avoids the historical backward-only custom Function and supports both forward-mode and reverse-mode AD. Safe inactive branches and inverse-layer stacking handle zero absorption and large absorption. Numerical equivalence and derivative accuracy are checked against independent NumPy/SciPy implementations.

Default demo wavelength grids and sample counts are declared in the run summary. Use 1 nm simulation and the relevant measured wavelengths for scientific experiments. Instrument spectral response convolution and experimental preprocessing must be supplied as appropriate for those observations.

## What requires external datasets

The empirical fsurf relationship can be used as published, and a nine-parameter calibration model is provided. Re-estimating that relationship, cross-validation across datasets, validating against measured biochemical traits, reproducing the paper's six-model comparisons, and reproducing the reported hardware speedups require the external observations, experiment settings and solver implementations. The synthetic workflow verifies model execution and numerical derivatives; it cannot establish those measured-data claims.
