# -*- coding: utf-8 -*-
"""
Batch trust-region (dogleg) solver with sigmoid-bounded parameterization,
implemented in PyTorch so the whole 5000-sample ablation can run on GPU.

This mirrors scipy.optimize.least_squares(method='trf')'s trust-region idea:
    - bounded variables are mapped to an unconstrained space via sigmoid
      reparameterization (params = sigmoid(z) * scale + low), so every trial
      step stays strictly inside [low, high] (no reflective line search needed).
    - each iteration solves the trust-region subproblem with a dogleg path
      (steepest-descent step + Gauss-Newton step), then accepts/rejects the
      trial step by the standard gain ratio and updates the trust radius.

The residual Jacobian is computed with torch.func.jacfwd + vmap, exactly like
func.AdaptiveLM, so it is compatible with any torch forward model.
"""
import torch
from torch.func import jacfwd, vmap

_EPS = 1e-12


def _safe_solve(H, g, max_retry=6):
    """Solve H d = -g for a batch of (P,P) systems with adaptive damping."""
    B, P, _ = H.shape
    eye = torch.eye(P, dtype=H.dtype, device=H.device).unsqueeze(0)
    diag = H.diagonal(dim1=1, dim2=2)
    lam = (diag.abs().amax(dim=1, keepdim=True) * 1e-8).clamp_min(1e-10)
    for _ in range(max_retry):
        try:
            d = torch.linalg.solve(H + lam.unsqueeze(-1) * eye, -g)
            if torch.isfinite(d).all():
                return d, lam
        except Exception:
            pass
        lam = lam * 100.0
    # final fallback: gradient step
    return -g / (g.norm(dim=1, keepdim=True).clamp_min(_EPS)), lam


def batch_trf(forward_model, obs, init_params, low, high, device='cuda',
              max_iter=150, gtol=1e-8, xtol=1e-12, ftol=1e-12,
              delta0=1.0, delta_max=100.0, eta_low=0.25, eta_high=0.75,
              grad_clip=1e4, verbose=False):
    """
    obs:        (B, L) float32 numpy array
    init_params:(P,) or (B, P) numpy array in physical space
    low, high:  (P,) numpy arrays of physical bounds
    Returns:    (B, P) physical params on CPU
    """
    B, L = obs.shape
    P = init_params.shape[-1]
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    scale = high_t - low_t
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(init_params, dtype=torch.float32, device=device)
    if init_t.dim() == 1:
        init_t = init_t.unsqueeze(0).expand(B, -1)

    # unconstrained latent z
    p_norm = ((init_t - low_t) / scale).clamp(0.02, 0.98)
    z = torch.logit(p_norm)

    def params_from_z(z_):
        return torch.sigmoid(z_) * scale + low_t

    def residual(z_):
        return forward_model(params_from_z(z_)) - obs_t  # (B,L)

    # jacobian of residual w.r.t. z
    def res_single(zp, ob):
        return forward_model((torch.sigmoid(zp) * scale + low_t).unsqueeze(0)).squeeze(0) - ob

    jac_fn = vmap(jacfwd(res_single), in_dims=(0, 0))

    delta = torch.full((B, 1), delta0, dtype=torch.float32, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for it in range(max_iter):
        if finished.all():
            break
        with torch.no_grad():
            r = residual(z)
            loss = (r * r).sum(dim=1)
            J = jac_fn(z, obs_t)                       # (B,L,P)
            g = torch.einsum('blp,bl->bp', J, r)       # (B,P)
            H = torch.einsum('blp,blq->bpq', J, J)     # (B,P,P)

        # gradient clipping for numerical stability
        gnorm = g.norm(dim=1, keepdim=True)
        g = g * torch.where(gnorm > grad_clip, grad_clip / (gnorm + _EPS),
                            torch.ones_like(gnorm))

        with torch.no_grad():
            # dogleg components
            d_gn, lam = _safe_solve(H, g)               # Gauss-Newton step
            gHg = torch.einsum('bp,bpq,bq->b', g, H, g)
            gg = (g * g).sum(dim=1)
            alpha = gg / gHg.clamp_min(_EPS)
            d_sd = -alpha.unsqueeze(1) * g             # steepest-descent step

            dgn_n = d_gn.norm(dim=1)                   # (B,)
            dsd_n = d_sd.norm(dim=1)
            dlt = delta.squeeze(1)

            # dogleg interpolation: find tau such that ||d_sd + tau*(d_gn-d_sd)|| = delta
            diff = d_gn - d_sd
            a = (diff * diff).sum(dim=1).clamp_min(_EPS)
            bcoef = 2.0 * (d_sd * diff).sum(dim=1)
            ccoef = dsd_n * dsd_n - dlt * dlt
            disc = (bcoef * bcoef - 4.0 * a * ccoef).clamp_min(0.0)
            tau = ((-bcoef + torch.sqrt(disc)) / (2.0 * a)).clamp(0.0, 1.0)
            d_interp = d_sd + tau.unsqueeze(1) * diff

            d = torch.where(
                (dgn_n <= dlt).unsqueeze(1), d_gn,
                torch.where((dsd_n >= dlt).unsqueeze(1),
                            (dlt / dsd_n.clamp_min(_EPS)).unsqueeze(1) * d_sd,
                            d_interp))

            # trial step
            z_trial = z + d
            r_trial = residual(z_trial)
            loss_trial = (r_trial * r_trial).sum(dim=1)

            gd = torch.einsum('bp,bp->b', g, d)
            dHd = torch.einsum('bp,bpq,bq->b', d, H, d)
            pred = -(gd + 0.5 * dHd)
            actual = loss - loss_trial
            ratio = actual / pred.clamp_min(_EPS)

            # update trust radius
            delta = torch.where(
                (ratio < eta_low).unsqueeze(1), 0.25 * delta,
                torch.where((ratio > eta_high).unsqueeze(1),
                            (2.0 * delta).clamp_max(delta_max), delta))

            accept = ratio > 1e-4
            z = torch.where(accept.unsqueeze(1), z_trial, z)
            loss = torch.where(accept, loss_trial, loss)

            # convergence flags
            gmax = g.norm(dim=1)
            finished = finished | (gmax < gtol) | (delta.squeeze(1) < xtol) | (pred.abs() < ftol)
            finished = finished | (~torch.isfinite(loss))

        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print('  trf iter=%3d mean_loss=%.6e accept=%.2f delta=%.3f done=%d/%d' % (
                it, loss[torch.isfinite(loss)].mean().item(),
                accept.float().mean().item(), delta.mean().item(),
                finished.sum().item(), B), flush=True)

    with torch.no_grad():
        out = params_from_z(z)
    return out.detach().cpu().numpy()
