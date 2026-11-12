"""Special functions used by the PROSPECT-D port.

``e1`` replaces the former ``Expi`` autograd Function: it is written with plain
torch ops only, so reverse mode (``autograd.grad``) and forward mode
(``torch.func.jacfwd`` / ``jvp``) agree exactly, and it is ``vmap``-safe.
The previous custom ``Function`` only defined ``forward``/``backward``; under
forward-mode AD its ``jvp`` returned zeros, which silently corrupted the
Jacobians used by the LM/TRF solvers and by the sensitivity analysis.
"""

import torch

_EULER_GAMMA = 0.5772156649015329


def e1(x, terms=40, cf_terms=50):
    """Exponential integral E1(x) = ∫_x^∞ exp(-t)/t dt for x > 0.

    Series for x < 1 (fast, no cancellation), Lentz continued fraction for
    x >= 1 (accurate to ~1e-7 relative).  Both branches are plain elementwise
    ops; the branch is selected with ``torch.where`` so ``vmap``/``jacfwd``
    work without special-casing.
    """
    x = torch.clamp(x, min=1e-30)

    # --- series: E1(x) = -gamma - ln x + sum_{n>=1} (-1)^(n+1) x^n / (n n!)
    xs = torch.clamp(x, max=1.0)
    series = torch.zeros_like(xs)
    a = xs.clone()                       # a_n = (-1)^(n+1) x^n / n!
    for n in range(1, terms + 1):
        series = series + a / n
        a = -a * xs / (n + 1)
    e1_small = -_EULER_GAMMA - torch.log(xs) + series

    # --- continued fraction (Lentz) for x >= 1
    xl = torch.clamp(x, min=1.0)
    t0 = torch.zeros_like(xl)
    for k in range(cf_terms, 0, -1):
        t0 = k / (1.0 + k / (xl + t0))
    e1_large = torch.exp(-xl) / (xl + t0)

    return torch.where(x < 1.0, e1_small, e1_large)
