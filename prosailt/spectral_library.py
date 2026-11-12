#!/usr/bin/env python
"""Spectral libraries for PROSPECT + SAIL
"""
import pkgutil
from collections import namedtuple
from io import BytesIO

import torch
import numpy as np
# import warnings

Spectra = namedtuple("Spectra", "prospect5 prospectd prospectpro soil light")
Prospect5Spectra = namedtuple("Prospect5Spectra", "nr kab kcar kbrown kw km")
ProspectDSpectra = namedtuple("ProspectDSpectra",
                              "nr kab kcar kbrown kw km kant")
ProspectPROSpectra = namedtuple("ProspectPROSpectra",
                                "nr kab kcar kbrown kw km kant kprot kcbc")
SoilSpectra = namedtuple("SoilSpectra", "rsoil1 rsoil2")
LightSpectra = namedtuple("LightSpectra", "es ed")


def get_spectra(device="cpu"):
    """Reads the spectral information and stores is for future use."""
    # warnings.warn("WARNING: Spectra data should only be read at intialization.")
    # PROSPECT-D
    prospect_d_spectraf = pkgutil.get_data(__package__, "prospect_d_spectra.txt")
    _, nr, kab, kcar, kant, kbrown, kw, km = torch.tensor(
        np.loadtxt(BytesIO(prospect_d_spectraf), unpack=True),dtype=torch.float32,device=device)
    prospect_d_spectra = ProspectDSpectra(nr, kab, kcar, kbrown, kw, km, kant)
    # PROSPECT-PRO
    prospect_pro_spectraf = pkgutil.get_data(__package__, "prospect_pro_spectra.txt")
    _, nr, kab, kcar, kant, kbrown, kw, km, kprot, kcbc = torch.tensor(
        np.loadtxt(BytesIO(prospect_pro_spectraf), unpack=True),dtype=torch.float32,device=device)
    prospect_pro_spectra = ProspectPROSpectra(nr, kab, kcar, kbrown, kw, km,
                                              kant, kprot, kcbc)
    # PROSPECT 5
    prospect_5_spectraf = pkgutil.get_data(__package__, "prospect5_spectra.txt")
    nr, kab, kcar, kbrown, kw, km = torch.tensor(
        np.loadtxt(BytesIO(prospect_5_spectraf), unpack=True),dtype=torch.float32,device=device)
    prospect_5_spectra = Prospect5Spectra(nr, kab, kcar, kbrown, kw, km)
    # SOIL
    soil_spectraf = pkgutil.get_data(__package__, "soil_reflectance.txt")
    rsoil1, rsoil2 = torch.tensor(
        np.loadtxt(BytesIO(soil_spectraf), unpack=True),dtype=torch.float32,device=device)
    soil_spectra = SoilSpectra(rsoil1, rsoil2)
    # LIGHT
    light_spectraf = pkgutil.get_data(__package__, "light_spectra.txt")
    es, ed = torch.tensor(np.loadtxt(BytesIO(light_spectraf), unpack=True),dtype=torch.float32,device=device)
    light_spectra = LightSpectra(es, ed)
    spectra = Spectra(
        prospect_5_spectra,
        prospect_d_spectra,
        prospect_pro_spectra,
        soil_spectra,
        light_spectra,
    )
    return spectra
