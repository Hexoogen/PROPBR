# Third-party code and model constants

## PROSPECT-D / PROSAIL

The leaf layer equations, interface implementation, and bundled spectral library
in `prospect.py`, `prospect_h.py`, and `prosailt/` are adapted from PROSAIL by
J. Gomez-Dans, Copyright 2017-2018 J. Gomez-Dans, and the author's PyTorch port.
PROSAIL identifies its license as GPLv3. The full GPLv3 text is retained in `LICENSE`.
The bundled optical, soil and illumination coefficients retain their upstream origin.

- Upstream package: https://pypi.org/project/prosail/2.0.5/
- Upstream repository: https://github.com/jgomezdans/prosail

## efficient-kan

`kan_model.py` is adapted from efficient-kan by Huanqi Cao, Copyright (c) 2024
Huanqi Cao, under the MIT License. The complete MIT notice is retained in
`licenses/efficient-kan-MIT.txt`.

- Source: https://github.com/Blealtan/efficient-kan

## Author-supplied model assets

`lut/` contains the author's pretrained KAN weights and precomputed model arrays.
`lut/DLM_nr.npy` is the author-supplied DLM 2.1 surface refractive-index spectrum,
covering 400-2500 nm. Its source did not include a separate redistribution license;
it is recorded as an author-supplied model constant.

The weight files are the original author-supplied checkpoints. No retraining or
checkpoint rewriting is performed for this update. Exact filenames, sizes and
SHA-256 hashes are recorded in `lut/manifest.json`.

Measured leaf datasets and generated inversion outputs are not included.
