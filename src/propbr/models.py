"""Paper-consistent PROPBR (Eq.20) and sPROPBR (Eq.21)."""
import numpy as np
import torch
from torch import nn
from .spectra import SpectralConstants
from .prospect import prospect_d, calctav
from .surface import (reduced_ks, reference_specular, full_geometry_ks,
                      schlick, fresnel0)
from .parameters import ParameterSpace
from .kan import KsKAN


class LeafModel(nn.Module):
    def __init__(self, wavelengths=None, dtype=torch.float64):
        super().__init__()
        self.spectra = SpectralConstants(wavelengths, dtype=dtype)
        self.register_buffer("tav40", calctav(40., self.spectra.nr))
        self.register_buffer("tav90", calctav(90., self.spectra.nr))

    @property
    def wavelengths(self):
        return self.spectra.wavelengths

    @property
    def device(self):
        return self.wavelengths.device

    @property
    def dtype(self):
        return self.wavelengths.dtype

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def simulate(self, parameters):
        """Checked entry point; returns (B,A,L), even for one angle."""
        parameters = self.tensor(parameters)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)
        self.space.validate(parameters)
        return self(parameters)


class PROPBR(LeafModel):
    def __init__(self, phase_angles=(35.,), kernel="kan", free_fsurf=False,
                 geometry=None, wavelengths=None, dtype=torch.float64, weights_path=None):
        """
        phase_angles: degree-valued vector shared by all leaves; reduced geometry.
        geometry: optional (A,3) [incidence, viewing, relative azimuth] in degrees.
        Explicit geometry requires kernel='analytic'. Diffuse correction h=1.
        free_fsurf: nine parameters for empirical calibration; analytic only.
        Otherwise eight parameters, with fsurf=.66*sigma**(-.63).
        """
        super().__init__(wavelengths, dtype)
        if kernel not in ("analytic", "kan"):
            raise ValueError("kernel must be analytic or kan")
        if (free_fsurf or geometry is not None) and kernel != "analytic":
            raise ValueError("Free fsurf / full geometry requires the analytic kernel")
        self.kernel, self.free_fsurf = kernel, free_fsurf
        self.space = ParameterSpace.propbr(free_fsurf)
        phase = np.atleast_1d(np.asarray(phase_angles, dtype=float))
        if geometry is None:
            if (phase.ndim != 1 or phase.size == 0 or not np.isfinite(phase).all() or
                    (phase < 0).any() or (phase >= 90).any()):
                raise ValueError("Reduced phase angles must lie in [0,90) degrees")
            if kernel == "kan" and (phase < 5).any():
                raise ValueError("KAN was trained at 5–90 degrees; use analytic below 5 degrees")
            self.register_buffer("geometry_rad", None)
        else:
            geom = np.asarray(geometry, dtype=float)
            if (geom.ndim != 2 or geom.shape[1] != 3 or geom.shape[0] == 0 or
                    not np.isfinite(geom).all() or (geom[:, :2] < 0).any() or
                    (geom[:, :2] >= 90).any()):
                raise ValueError("Geometry must be (A,3), with zenith angles in [0,90)")
            self.register_buffer("geometry_rad", self.tensor(np.deg2rad(geom)))
            ci, cr, az = np.deg2rad(geom).T
            phase = np.rad2deg(np.arccos(np.clip(np.cos(ci) * np.cos(cr) +
                               np.sin(ci) * np.sin(cr) * np.cos(az), -1., 1.)))
        self.register_buffer("phase_rad", self.tensor(np.deg2rad(phase)))
        self.ks_surrogate = KsKAN(weights_path).to(dtype=dtype) if kernel == "kan" else None

    @property
    def n_angles(self):
        return self.phase_rad.numel()

    def components(self, parameters):
        dhr, trans, _ = prospect_d(parameters, self.spectra, self.tav40, self.tav90)
        sigma = parameters[..., 7:8]
        fsurf = parameters[..., 8:9] if self.free_fsurf else None
        reference = reference_specular(sigma, self.spectra.nr)
        diffuse = (dhr - reference).unsqueeze(-2)
        phase = self.phase_rad
        if self.geometry_rad is not None:
            inc, view, az = self.geometry_rad.unbind(-1)
            ks, phase = full_geometry_ks(sigma, inc, view, az, fsurf)
        elif self.kernel == "kan":
            ks = self.ks_surrogate(phase, sigma)
        else:
            ks = reduced_ks(phase, sigma, fsurf)
        specular = ks.unsqueeze(-1) * schlick(self.spectra.surface_nr, phase.unsqueeze(-1))
        return {"brf": diffuse + specular, "dhrf": dhr, "transmittance": trans,
                "reference_specular": reference, "diffuse": diffuse, "specular": specular,
                "ks": ks}

    def forward(self, parameters):
        return self.components(parameters)["brf"]


class SPROPBR(LeafModel):
    def __init__(self, wavelengths=None, dtype=torch.float64):
        super().__init__(wavelengths, dtype)
        self.space = ParameterSpace.spropbr()
        self.n_angles = 1

    def components(self, parameters):
        dhr, trans, ralf = prospect_d(parameters, self.spectra, self.tav40, self.tav90)
        diffuse = (dhr - ralf).unsqueeze(-2)
        specular = (parameters[..., 7:8] * fresnel0(self.spectra.surface_nr)).unsqueeze(-2)
        return {"brf": diffuse + specular, "dhrf": dhr, "transmittance": trans,
                "reference_specular": ralf.expand_as(dhr), "diffuse": diffuse, "specular": specular}

    def forward(self, parameters):
        return self.components(parameters)["brf"]
