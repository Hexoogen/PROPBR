"""Load numerical model constants. No measured leaf datasets are bundled."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

ASSETS = Path(__file__).resolve().parent / "assets"


class SpectralConstants(nn.Module):
    def __init__(self, wavelengths=None, dtype=torch.float64):
        super().__init__()
        table = np.loadtxt(ASSETS / "prospect_d_spectra.txt")
        surface_nr = np.load(ASSETS / "surface_nr.npy", allow_pickle=False)
        grid = table[:, 0]
        if surface_nr.shape != grid.shape:
            raise ValueError("Surface and internal spectral grids differ")
        target = grid if wavelengths is None else np.asarray(wavelengths, dtype=float)
        if (target.ndim != 1 or target.size == 0 or not np.isfinite(target).all() or
                np.any(np.diff(target) <= 0) or target[0] < grid[0] or target[-1] > grid[-1]):
            raise ValueError("Wavelengths must increase strictly within 400–2500 nm")
        # Point interpolation, not an instrument spectral response convolution.
        for name, values in zip(("nr", "kab", "kcar", "kant", "kbrown", "kw", "km"), table[:, 1:].T):
            self.register_buffer(name, torch.tensor(np.interp(target, grid, values), dtype=dtype))
        self.register_buffer("surface_nr", torch.tensor(np.interp(target, grid, surface_nr), dtype=dtype))
        self.register_buffer("wavelengths", torch.tensor(target, dtype=dtype))
