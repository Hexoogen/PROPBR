# Third-party code and model constants

## PROSPECT-D / PROSAIL

The leaf layer equations and Stern–Allen interface implementation are adapted from the PROSAIL implementation by J. Gomez-Dans, Copyright 2017–2018 J. Gomez-Dans, and the author's local PyTorch implementation. PROSAIL identifies its license as GPLv3. The numerical PROSPECT-D coefficients are taken from the PROSAIL 2.0.5 distribution.

- Upstream package: https://pypi.org/project/prosail/2.0.5/
- Upstream repository: https://github.com/jgomezdans/prosail
- Full GPLv3 text is included in `LICENSE`.

The public wrapper does not vendor SAIL, soil spectra, illumination spectra or any measured leaf dataset. Numerical optical coefficients are included as required model constants, with a hash in the asset manifest.

## efficient-kan

`src/propbr/_efficient_kan.py` is adapted from efficient-kan by Huanqi Cao, Copyright (c) 2024 Huanqi Cao, under the MIT License. The code retains the upstream network structure and adds numerical protection in the entropy regularizer. The full MIT notice is included in `licenses/efficient-kan-MIT.txt`.

- Source: https://github.com/Blealtan/efficient-kan

## Author-supplied model assets

`surface_nr.npy` is the author-supplied DLM 2.1 surface refractive-index spectrum used by the manuscript (400–2500 nm). Its local source did not include a separate redistribution license. Its provenance is recorded as an author-supplied model constant, rather than assigning it the PROSAIL license.

`ks_kan.pt` contains the author's trained network tensors and scalers. It is a sanitized copy of the checkpoint used in the manuscript audit; no training is performed during packaging. The sanitization removes old fit-score metadata and preserves every trained tensor and scaler value.
