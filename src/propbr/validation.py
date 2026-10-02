"""Numerical checks against analytic targets, not manuscript experimental metrics."""
import hashlib
import json
import numpy as np
import torch
from .spectra import ASSETS
from .kan import KsKAN
from .surface import reduced_ks, hemispherical_specular


def metrics(reference, predicted):
    y, p = np.asarray(reference), np.asarray(predicted)
    residual = p - y
    return {"count": int(y.size), "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1. - np.sum(residual**2) / np.sum((y - y.mean())**2)),
            "mre_percent": float(100. * np.mean(np.abs(residual) / np.maximum(np.abs(y), 1e-8)))}


def asset_integrity():
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    report = {}
    for name, metadata in manifest["files"].items():
        actual = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
        if actual != metadata["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {name}")
        report[name] = actual
    return report


def validate_surrogate(device="cpu", dtype=torch.float64, batch_size=4096):
    surrogate = KsKAN().to(device=device, dtype=dtype)
    report = {}
    for label, degrees, sigma in (
        ("training_grid", np.linspace(5., 90., 86), np.linspace(.01, 1., 100)),
        ("offset_fine_grid", np.linspace(5.1, 89.9, 258), np.linspace(.011, .999, 300)),
    ):
        phase, rough = np.meshgrid(np.deg2rad(degrees), sigma, indexing="ij")
        p = torch.tensor(phase.ravel(), device=device, dtype=dtype)
        r = torch.tensor(rough.ravel(), device=device, dtype=dtype)
        with torch.no_grad():
            y = reduced_ks(p, r).cpu().numpy()
            pred = torch.cat([surrogate(a, b) for a, b in zip(p.split(batch_size), r.split(batch_size))]).cpu().numpy()
        if not np.isfinite(pred).all():
            raise FloatingPointError("Non-finite surrogate output")
        report[label] = metrics(y, pred)
        if label == "training_grid":
            mask = (y >= .1) & (y <= 50.) & (pred > 0.)
            report["filtered_training_grid_Ks_0.1_to_50"] = metrics(y[mask], pred[mask])
    report["note"] = "Filtered training-grid fit and offset-grid evaluation are reported separately."
    return report


def validate_reference(order=96):
    roughness = np.linspace(.01, 1., 20)
    nr = np.linspace(1.2, 1.6, 9)
    reference = np.array([hemispherical_specular(s, nr, order=order) for s in roughness])
    approximation = ((1. - nr) / (1. + nr))**2 * np.exp(-1.26 * roughness[:, None]**1.79)
    refined = np.array([hemispherical_specular(s, nr, order=order * 2) for s in roughness])
    report = metrics(refined, approximation)
    report["quadrature_max_change_on_order_doubling"] = float(np.max(np.abs(reference - refined)))
    report["grid"] = {"sigma": [0.01, 1., 20], "n_in": [1.2, 1.6, 9], "order": order * 2}
    return report
