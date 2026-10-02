"""Frozen B-spline Ks surrogate; fsurf is included in its training target.

KAN implementation: efficient-kan, Copyright (c) 2024 Huanqi Cao (MIT).
See THIRD_PARTY_NOTICES.md and licenses/efficient-kan-MIT.txt.
"""
import torch
from torch import nn
from ._efficient_kan import KAN
from .spectra import ASSETS


class KsKAN(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        checkpoint = torch.load(weights_path or ASSETS / "ks_kan.pt", map_location="cpu", weights_only=True)
        # Construction should not advance the user's random stream.
        with torch.random.fork_rng(devices=[]):
            self.kan = KAN(checkpoint["layers_hidden"], grid_size=checkpoint["grid_size"],
                           spline_order=checkpoint["spline_order"], grid_range=[-1.5, 1.5])
        self.kan.load_state_dict(checkpoint["model_state"])
        for name, key in (("x_mean", "scaler_X_mean"), ("x_scale", "scaler_X_scale"),
                          ("y_mean", "scaler_Y_mean"), ("y_scale", "scaler_Y_scale")):
            self.register_buffer(name, torch.as_tensor(checkpoint[key], dtype=torch.float64).clone())
        self.requires_grad_(False)
        self.eval()

    def forward(self, phase_rad, sigma):
        phase_rad, sigma = torch.broadcast_tensors(phase_rad, sigma)
        shape = phase_rad.shape
        x = torch.stack((phase_rad.flatten(), sigma.flatten()), dim=-1)
        y = self.kan((x - self.x_mean) / self.x_scale).squeeze(-1)
        return torch.expm1(y * self.y_scale + self.y_mean).reshape(shape)
