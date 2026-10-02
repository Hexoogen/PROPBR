"""Bounded, batched multi-start AdamW inversion through the forward graph."""
from dataclasses import dataclass
import torch


@dataclass
class InversionResult:
    parameters: torch.Tensor
    prediction: torch.Tensor
    mse: torch.Tensor
    iterations: int
    loss_history: list[float]


def invert(model, observations, *, initial=None, space=None, n_iters=1000, lr=.03,
           n_restarts=3, seed=42, batch_size=256, mask=None, weight_decay=0.,
           patience=150, tolerance=1e-10, verbose=False):
    """Return best parameters per sample over all iterates and all restarts.

    Observations: (B,L) or (B,A,L). One shared parameter vector per leaf.
    NaN/Inf bands are excluded; mask=True means an observed band is used.
    Spectral MSE averages over valid wavelengths AND angles per sample.
    The optimized sum of sample MSEs adds no parameter prior by default.
    """
    space = space or model.space
    if tuple(space.names) != tuple(model.space.names):
        raise ValueError("Parameter names/order must match the model")
    if n_iters < 1 or n_restarts < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer configuration")
    if patience < 1 or tolerance < 0:
        raise ValueError("Invalid stopping configuration")
    obs = model.tensor(observations).detach()
    if obs.ndim == 2:
        obs = obs.unsqueeze(1)
    if obs.ndim != 3 or obs.shape[1:] != (model.n_angles, model.wavelengths.numel()) or len(obs) == 0:
        raise ValueError("Observation shape must be (B,A,L), matching model angles and wavelengths")
    valid = torch.isfinite(obs)
    if mask is not None:
        mask = torch.as_tensor(mask, device=model.device, dtype=torch.bool)
        if mask.ndim == 2 and mask.shape == (len(obs), obs.shape[-1]):
            mask = mask.unsqueeze(1)
        valid = valid & torch.broadcast_to(mask, obs.shape)
    if not bool(valid.flatten(1).any(1).all()):
        raise ValueError("Each sample needs at least one finite, unmasked observation")
    target = torch.where(valid, obs, torch.zeros_like(obs))
    B, P = len(obs), len(space.names)
    init = model.tensor(space.initial if initial is None else initial).detach()
    if init.ndim == 1:
        init = init.expand(B, -1).clone()
    if init.shape != (B, P):
        raise ValueError("Initial parameters must have shape (P,) or (B,P)")
    space.validate(init)
    # Generate all restart initializations before chunking: batch size does not change them.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    uniform = torch.rand(n_restarts, B, P, generator=generator, dtype=model.dtype).clamp(.01, .99)
    lo, hi = space.bounds(uniform)
    starts = (lo + (hi - lo) * uniform).to(model.device)
    starts[0] = init
    result_params, result_mse = [], []
    history, max_steps = [], 0
    for offset in range(0, B, batch_size):
        end = min(offset + batch_size, B)
        size = end - offset
        z = torch.nn.Parameter(space.encode(starts[:, offset:end].reshape(-1, P)))
        y = target[offset:end].repeat(n_restarts, 1, 1)
        v = valid[offset:end].repeat(n_restarts, 1, 1)
        count = v.flatten(1).sum(1)
        opt = torch.optim.AdamW([z], lr=lr, weight_decay=weight_decay)
        best_mse = torch.full((len(z),), float("inf"), device=model.device, dtype=model.dtype)
        best_p = space.decode(z).detach().clone()
        stale, last_improvement = 0, float("inf")

        def evaluate():
            p = space.decode(z)
            residual = model(p) - y
            if not bool(torch.isfinite(residual[v]).all()):
                raise FloatingPointError("Non-finite forward result during inversion")
            mse = torch.where(v, residual.square(), torch.zeros_like(residual)).flatten(1).sum(1) / count
            with torch.no_grad():
                better = mse < best_mse
                best_mse.copy_(torch.minimum(best_mse, mse))
                best_p.copy_(torch.where(better[:, None], p, best_p))
            return mse

        for step in range(n_iters):
            opt.zero_grad(set_to_none=True)
            mse = evaluate()
            mse.sum().backward()
            if not bool(torch.isfinite(z.grad).all()):
                raise FloatingPointError("Non-finite inversion gradient")
            opt.step()
            current = float(best_mse.mean())
            if last_improvement - current > tolerance:
                stale, last_improvement = 0, current
            else:
                stale += 1
            if step % 100 == 0 or step == n_iters - 1:
                history.append(current)
                if verbose:
                    print(f"  samples {offset}:{end}, step {step + 1}, best MSE {current:.6g}", flush=True)
            if stale >= patience:
                break
        with torch.no_grad():
            evaluate()  # Include the final optimizer update in best-iterate selection.
            m = best_mse.reshape(n_restarts, size)
            p = best_p.reshape(n_restarts, size, P)
            winner = m.argmin(0)
            cols = torch.arange(size, device=model.device)
            result_params.append(p[winner, cols])
            result_mse.append(m[winner, cols])
        max_steps = max(max_steps, step + 1)
    parameters = torch.cat(result_params).detach()
    with torch.no_grad():
        prediction = torch.cat([model(p) for p in parameters.split(batch_size)])
    return InversionResult(parameters, prediction, torch.cat(result_mse), max_steps, history)
