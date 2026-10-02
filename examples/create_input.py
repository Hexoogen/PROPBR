"""Write the public NPZ input schema using generated spectra, not a dataset."""
import argparse
from pathlib import Path
import numpy as np
from propbr import PROPBR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/example_input")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    wavelengths = np.arange(400., 2501., 10.)
    parameters = np.array([[1.8, 48., 12., 3., .017, .009, .12, .28]])
    model = PROPBR(phase_angles=[15., 35., 55.], wavelengths=wavelengths)
    brf = model.simulate(parameters).detach().numpy()
    np.savez_compressed(folder / "parameters.npz", parameters=parameters, wavelengths=wavelengths)
    np.savez_compressed(folder / "observations.npz", brf=brf, wavelengths=wavelengths)
    print(folder.resolve())
