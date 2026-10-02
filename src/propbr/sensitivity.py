"""Physical/latent Jacobians, response correlation and SALib EFAST."""
import numpy as np
import torch
from torch.func import jacfwd, vmap
from .parameters import ParameterSpace


def local_sensitivity(model, parameters, *, space=None, chunk_size=16):
    """Return B×A×L×P Jacobians in physical, range-scaled and latent coordinates."""
    space = space or model.space
    p = model.tensor(parameters).detach()
    if p.ndim == 1:
        p = p.unsqueeze(0)
    space.validate(p)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    def single(x):
        return model(x.unsqueeze(0)).squeeze(0)
    J = torch.cat([vmap(jacfwd(single))(batch) for batch in p.split(chunk_size)])
    lo, hi = space.bounds(p)
    z = space.encode(p)
    u = torch.sigmoid(z)
    # Initial boundary values are moved into the open sigmoid domain.
    # Calculate J_z at decode(encode(p)), which may differ from an exact boundary p.
    latent_p = space.decode(z)
    J_at_latent = torch.cat([vmap(jacfwd(single))(batch) for batch in latent_p.split(chunk_size)])
    dp_dz = (hi - lo) * u * (1. - u)
    return {"physical": J.detach(), "range_scaled": (J * (hi - lo)).detach(),
            "latent": (J_at_latent * dp_dz[:, None, None, :]).detach(),
            "latent_parameters": latent_p.detach()}


def response_correlation(jacobian):
    """Eq.34 per sample: normalized J^T J, flattening angles/wavelengths only.

    Zero-norm parameter columns have undefined correlation and are returned NaN.
    Gram is Fisher information only under a specified independent unit-noise model.
    This is response overlap, not an empirical correlation of retrieved traits.
    """
    J = torch.as_tensor(jacobian)
    flat = J.reshape(J.shape[0], -1, J.shape[-1])
    gram = flat.transpose(-1, -2) @ flat
    norms = gram.diagonal(dim1=-2, dim2=-1).clamp(min=0).sqrt()
    denom = norms.unsqueeze(-1) * norms.unsqueeze(-2)
    correlation = torch.where(denom > 0, gram / denom.clamp(min=1e-30), torch.full_like(gram, float("nan")))
    return {"gram": gram, "correlation": correlation.clamp(-1., 1.)}


def global_sensitivity(model, *, n=1000, harmonics=4, seed=42, batch_size=128,
                       include_phase=False, n_resamples=100, verbose=False):
    """EFAST S1/ST per angle and wavelength. SALib uses N*P total evaluations.

    For PROPBR, include_phase adds theta_phase in [0,65] degrees; analytic only.
    sPROPBR uses the paper's mu sensitivity range [0,2.5], not inversion [0,30].
    """
    try:
        from SALib.sample.fast_sampler import sample
        from SALib.analyze.fast import analyze
    except ImportError as exc:
        raise ImportError("EFAST needs SALib: pip install -e '.[full]'") from exc
    from .models import PROPBR, SPROPBR
    if n <= 4 * harmonics**2 or batch_size < 1 or n_resamples < 2:
        raise ValueError("EFAST requires N>4*M^2, positive batch size and >=2 resamples")
    if include_phase and (not isinstance(model, PROPBR) or model.kernel != "analytic" or model.geometry_rad is not None):
        raise ValueError("Variable phase requires reduced analytic PROPBR")
    space = ParameterSpace.spropbr(sensitivity=True) if isinstance(model, SPROPBR) else model.space
    names = list(space.names) + (["theta_phase_deg"] if include_phase else [])
    bounds = list(zip(space.lower, space.upper)) + ([(0., 65.)] if include_phase else [])
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    samples = sample(problem, N=n, M=harmonics, seed=seed)
    responses = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            x = model.tensor(samples[start:start + batch_size])
            if include_phase:
                from .prospect import prospect_d
                from .surface import reduced_ks, reference_specular, schlick
                p, phase = x[:, :-1], torch.deg2rad(x[:, -1:])
                dhr, _, _ = prospect_d(p, model.spectra, model.tav40, model.tav90)
                sigma = p[:, 7:8]
                f = p[:, 8:9] if model.free_fsurf else None
                spec = reduced_ks(phase, sigma, f) * schlick(model.spectra.surface_nr, phase)
                result = (dhr - reference_specular(sigma, model.spectra.nr) + spec).unsqueeze(1)
            else:
                result = model(x)
            responses.append(result.cpu().numpy())
    Y = np.concatenate(responses)
    if not np.isfinite(Y).all():
        raise FloatingPointError("Non-finite EFAST simulations")
    indices = {k: np.full(Y.shape[1:] + (len(names),), np.nan) for k in ("S1", "ST", "S1_conf", "ST_conf")}
    for a in range(Y.shape[1]):
        for wave in range(Y.shape[2]):
            y = Y[:, a, wave]
            if np.var(y) > 1e-24:
                result = analyze(problem, y, M=harmonics, num_resamples=n_resamples, seed=seed, print_to_console=False)
                for k in indices:
                    indices[k][a, wave] = result[k]
        if verbose:
            print(f"  EFAST angle {a + 1}/{Y.shape[1]}, {len(samples)} evaluations", flush=True)
    return {**indices, "names": names, "bounds": bounds, "n_per_parameter": n,
            "evaluations": len(samples), "harmonics": harmonics}
