"""Explicit NPZ conventions for users supplying their own spectra."""
from pathlib import Path
import json
import numpy as np
import torch


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for key, value in arrays.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        clean[key] = np.asarray(value)
    np.savez_compressed(path, **clean)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
