"""Command-line workflows and synthetic end-to-end demonstration."""
import argparse
import importlib.metadata
from pathlib import Path
import time
import numpy as np
import torch
from . import __version__
from .models import PROPBR, SPROPBR
from .inversion import invert
from .sensitivity import local_sensitivity, response_correlation, global_sensitivity
from .io import save_npz, save_json, load_npz
from .validation import asset_integrity, validate_reference, validate_surrogate


def _model(args, wavelengths, model=None, phase=None, kernel=None):
    name = model or args.model
    if name == "spropbr":
        result = SPROPBR(wavelengths=wavelengths)
    else:
        result = PROPBR(phase_angles=phase if phase is not None else [float(x) for x in args.phase.split(",")],
                        kernel=kernel or args.kernel, wavelengths=wavelengths,
                        free_fsurf=getattr(args, "free_fsurf", False))
    return result.to(args.device)


def _inversion_options(args):
    return {"n_iters": args.iterations, "n_restarts": args.restarts, "seed": args.seed,
            "lr": args.lr, "batch_size": args.batch_size, "verbose": True}


def _diagnostics(model, p, folder):
    J = local_sensitivity(model, p)
    coupling = response_correlation(J["physical"])
    save_npz(folder / "local_sensitivity.npz", **J, **coupling,
             names=model.space.names, wavelengths=model.wavelengths)
    return J, coupling


def _plots(model, observed, result, J, coupling, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wl = model.wavelengths.cpu().numpy()
    obs, pred = observed.detach().cpu().numpy(), result.prediction.cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for a in range(model.n_angles):
        line, = axes[0].plot(wl, obs[0, a], lw=1.3, label=f"Synthetic angle {a + 1}")
        axes[0].plot(wl, pred[0, a], "--", color=line.get_color(), lw=1.)
        axes[1].plot(wl, pred[0, a] - obs[0, a], lw=1.)
    axes[0].set_ylabel("BRF")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Solid: synthetic input; dashed: inverted model")
    axes[1].set_ylabel("BRF residual")
    axes[1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    fig.savefig(folder / "spectral_fit.png", dpi=180)
    plt.close(fig)
    local = J["range_scaled"][0, model.n_angles // 2].cpu().numpy().T
    fig, ax = plt.subplots(figsize=(9, 4))
    maximum = max(float(np.max(np.abs(local))), 1e-10)
    image = ax.imshow(local, aspect="auto", origin="lower", extent=[wl[0], wl[-1], -.5, len(model.space.names) - .5],
                      cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    ax.set_yticks(np.arange(len(model.space.names)), model.space.names)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title("Local response at retrieved parameters (one observation angle)")
    fig.colorbar(image, ax=ax, label="dBRF/dparameter × parameter range")
    fig.tight_layout()
    fig.savefig(folder / "local_sensitivity.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(coupling["correlation"][0].cpu().numpy(), cmap="RdBu_r", vmin=-1., vmax=1.)
    ax.set_xticks(np.arange(len(model.space.names)), model.space.names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(model.space.names)), model.space.names)
    ax.set_title("Normalized Jacobian response overlap")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(folder / "response_correlation.png", dpi=180)
    plt.close(fig)


def _global(model, args, folder, include_phase=False):
    result = global_sensitivity(model, n=args.global_n, seed=args.seed,
                                batch_size=args.batch_size, include_phase=include_phase,
                                n_resamples=20, verbose=True)
    save_npz(folder / "efast.npz", **{k: result[k] for k in ("S1", "ST", "S1_conf", "ST_conf")},
             names=result["names"], wavelengths=model.wavelengths)
    save_json(folder / "efast_config.json", {k: v for k, v in result.items() if k not in ("S1", "ST", "S1_conf", "ST_conf")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.imshow(100. * result["S1"][0].T, aspect="auto", origin="lower", cmap="viridis",
                      extent=[float(model.wavelengths[0]), float(model.wavelengths[-1]), -.5, len(result["names"]) - .5])
    ax.set_yticks(np.arange(len(result["names"])), result["names"])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title("EFAST first-order variance contribution")
    fig.colorbar(image, ax=ax, label="S1 (%)")
    fig.tight_layout()
    fig.savefig(folder / "efast.png", dpi=180)
    plt.close(fig)


def demo(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    wavelengths = np.arange(400., 2501., args.step)
    generator = np.random.default_rng(args.seed)
    # Plausible, explicitly synthetic leaves; independent of optimizer initial guesses.
    base = np.array([1.8, 48., 12., 3., .017, .009, .12, .28])
    scale = np.array([.18, 12., 3., 1., .004, .002, .04, .08])
    truth = base + generator.uniform(-1., 1., (args.samples, 8)) * scale
    records = {}
    for name, kind, phases in (("propbr_fixed", "propbr", [35.]),
                                ("propbr_multiangle", "propbr", [15., 35., 55.]),
                                ("spropbr_unknown_geometry", "spropbr", [35.])):
        print(f"\n{name}: simulation → inversion → local sensitivity", flush=True)
        out = folder / name
        out.mkdir(exist_ok=True)
        model = _model(args, wavelengths, model=kind, phase=phases)
        p = truth.copy()
        if kind == "spropbr":
            p[:, 7] = generator.uniform(.5, 2., args.samples)
        p = model.tensor(p)
        with torch.no_grad():
            components = model.components(p)
            clean = components["brf"]
            noise = model.tensor(generator.normal(0., args.noise, clean.shape))
            observed = clean + noise
        save_npz(out / "synthetic_input.npz", wavelengths=wavelengths, parameters=p,
                 parameter_names=model.space.names, phase_angles=phases, brf=observed, clean_brf=clean)
        save_npz(out / "forward_components.npz", **components, wavelengths=wavelengths)
        started = time.perf_counter()
        fit = invert(model, observed, **_inversion_options(args))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        save_npz(out / "inversion.npz", parameters=fit.parameters, brf=fit.prediction, mse=fit.mse,
                 parameter_names=model.space.names, wavelengths=wavelengths)
        J, coupling = _diagnostics(model, fit.parameters, out)
        _plots(model, observed, fit, J, coupling, out)
        lo, hi = model.space.bounds(p)
        records[name] = {"samples": args.samples, "angles": model.n_angles,
                         "spectral_rmse_per_sample": fit.mse.sqrt().cpu().tolist(),
                         "parameter_rmse": (fit.parameters - p).square().mean(0).sqrt().cpu().tolist(),
                         "parameter_rmse_divided_by_range": ((fit.parameters - p) / (hi - lo)).square().mean(0).sqrt().cpu().tolist(),
                         "parameter_names": list(model.space.names), "iterations": fit.iterations,
                         "inversion_seconds": seconds, "loss_history": fit.loss_history}
        print(f"  spectral RMSE: {records[name]['spectral_rmse_per_sample']}", flush=True)
        if kind == "propbr":
            alternate = _model(args, wavelengths, model=kind, phase=phases,
                               kernel="analytic" if args.kernel == "kan" else "kan")
            with torch.no_grad():
                difference = alternate(p) - clean
            records[name]["analytic_vs_kan_brf_rmse"] = float(difference.square().mean().sqrt())
    print("\nValidating model assets, Ks surrogate and hemispherical reference", flush=True)
    records["validation"] = {"assets": asset_integrity(), "surrogate": validate_surrogate(args.device),
                              "reference_hemisphere": validate_reference()}
    if not args.skip_global:
        print("\nEFAST global sensitivity (synthetic parameter sampling)", flush=True)
        global_waves = np.arange(400., 2501., args.global_step)
        _global(_model(args, global_waves, model="propbr", phase=[35.], kernel="analytic"),
                args, folder / "propbr_fixed", include_phase=True)
        _global(_model(args, global_waves, model="spropbr"), args, folder / "spropbr_unknown_geometry")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy", "matplotlib")}
    versions["torch_runtime"] = torch.__version__
    save_json(folder / "run_summary.json", {"propbr_version": __version__, "device": args.device,
              "dtype": "float64", "seed": args.seed, "versions": versions,
              "settings": vars(args), "results": records,
              "note": "Synthetic workflow verification. These are not the paper's measured-data scores."})
    print(f"\nComplete: {folder.resolve() / 'run_summary.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PROPBR and sPROPBR scientific workflows")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "simulate", "invert", "sensitivity", "validate", "train-kan"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0 or auto")
        p.add_argument("--threads", type=int, default=1, help="CPU Torch threads")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output", default="outputs/demo" if name == "demo" else f"outputs/{name}")
        if name in ("demo", "simulate", "invert", "sensitivity"):
            p.add_argument("--kernel", choices=("kan", "analytic"), default="kan")
            p.add_argument("--batch-size", type=int, default=128)
        if name in ("simulate", "invert", "sensitivity"):
            p.add_argument("--input", required=True, help="NPZ; see README for schema")
            p.add_argument("--model", choices=("propbr", "spropbr"), default="propbr")
            p.add_argument("--phase", default="35", help="Degree angles, e.g. 15,35,55")
            p.add_argument("--free-fsurf", action="store_true")
        if name in ("demo", "invert"):
            p.add_argument("--iterations", type=int, default=1200)
            p.add_argument("--restarts", type=int, default=3)
            p.add_argument("--lr", type=float, default=.03)
        if name in ("demo", "sensitivity"):
            p.add_argument("--global-n", type=int, default=1000)
        if name == "demo":
            p.add_argument("--samples", type=int, default=2)
            p.add_argument("--step", type=float, default=10., help="Simulation wavelength step in nm; 1 for full resolution")
            p.add_argument("--noise", type=float, default=0., help="Synthetic Gaussian BRF noise standard deviation")
            p.add_argument("--global-step", type=float, default=50.)
            p.add_argument("--skip-global", action="store_true")
        if name == "sensitivity":
            p.add_argument("--global-analysis", action="store_true")
            p.add_argument("--include-phase", action="store_true")
        if name == "train-kan":
            p.set_defaults(output="outputs/training/ks_kan.pt")
            p.add_argument("--epochs", type=int, default=105000)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation; select --device cpu")
    if args.command == "demo":
        if args.samples < 1 or args.step <= 0 or args.global_step <= 0 or args.noise < 0:
            parser.error("Invalid synthetic sample/wavelength/noise configuration")
        demo(args)
    elif args.command == "validate":
        save_json(Path(args.output) / "validation.json", {"assets": asset_integrity(),
                  "surrogate": validate_surrogate(args.device), "reference": validate_reference()})
    elif args.command == "train-kan":
        from .training import train_kan
        train_kan(args.output, epochs=args.epochs, seed=args.seed, device=args.device)
    else:
        data = load_npz(args.input)
        model = _model(args, data["wavelengths"])
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        if args.command == "simulate":
            p = model.tensor(data["parameters"])
            if p.ndim == 1:
                p = p.unsqueeze(0)
            model.space.validate(p)
            with torch.no_grad():
                save_npz(out / "simulation.npz", **model.components(p), wavelengths=model.wavelengths,
                         parameters=p, parameter_names=model.space.names)
        elif args.command == "invert":
            fit = invert(model, data["brf"], initial=data.get("initial"), mask=data.get("mask"), **_inversion_options(args))
            save_npz(out / "inversion.npz", parameters=fit.parameters, prediction=fit.prediction,
                     mse=fit.mse, wavelengths=model.wavelengths, parameter_names=model.space.names)
        else:
            _diagnostics(model, data["parameters"], out)
            if args.global_analysis:
                _global(model, args, out, include_phase=args.include_phase)
        save_json(out / "settings.json", vars(args))


if __name__ == "__main__":
    main()
