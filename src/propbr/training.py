"""Regenerate the Ks LUT and optionally train a fresh 2→4→1 cubic B-spline KAN."""
from pathlib import Path
import json
import numpy as np
import torch
from ._efficient_kan import KAN
from .surface import reduced_ks


def train_kan(output, *, epochs=105000, seed=42, device="cpu", verbose=True):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    phase = torch.deg2rad(torch.linspace(5., 90., 86, dtype=torch.float64))
    sigma = torch.linspace(.01, 1., 100, dtype=torch.float64)
    p, r = torch.meshgrid(phase, sigma, indexing="ij")
    X = torch.stack((p.flatten(), r.flatten()), -1)
    Y = torch.log1p(reduced_ks(p, r).flatten()).unsqueeze(-1)
    xm, xs = X.mean(0), X.std(0, unbiased=False)
    ym, ys = Y.mean(), Y.std(unbiased=False)
    x = ((X - xm) / xs).to(device=device, dtype=torch.float32)
    y = ((Y - ym) / ys).to(device=device, dtype=torch.float32)
    weight = (1. / (p.flatten() + .02)).clamp(.5, 15.)
    weight = (weight / (1. / (p.flatten() + .02)).mean()).to(device=device, dtype=torch.float32)
    network = KAN([2, 4, 1], grid_size=10, spline_order=3, grid_range=[-1.5, 1.5]).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=.5)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (weight[:, None] * (network(x) - y).square()).mean()
        total = loss + network.regularization_loss(1e-4, 1e-4)
        total.backward()
        optimizer.step()
        scheduler.step(total.detach())
        if epoch % 150 == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "loss": float(total.detach())})
            if verbose:
                print(f"  KAN epoch {epoch + 1}/{epochs}, loss {history[-1]['loss']:.6g}", flush=True)
    checkpoint = {"model_state": {k: v.cpu() for k, v in network.state_dict().items()},
                  "scaler_X_mean": xm.tolist(), "scaler_X_scale": xs.tolist(),
                  "scaler_Y_mean": float(ym), "scaler_Y_scale": float(ys),
                  "layers_hidden": [2, 4, 1], "grid_size": 10, "spline_order": 3,
                  "target": "Ks includes fsurf; no additional surface multiplier at inference",
                  "training_domain": {"phase_deg": [5., 90.], "sigma": [.01, 1.]},
                  "epochs": epochs, "seed": seed}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output.with_suffix(".training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output
