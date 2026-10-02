"""Parameter order, units and the paper's bounded sigmoid map."""
from dataclasses import dataclass
import torch

BIO_NAMES = ("N", "Cab", "Car", "Anth", "Cw", "Cm", "Cbrown")
BIO_UNITS = ("1", "ug cm^-2", "ug cm^-2", "ug cm^-2", "cm", "g cm^-2", "1")
BIO_LOW = (1., 0., 0., 0., 0., 0., 0.)
BIO_HIGH = (3.5, 120., 30., 40., .06, .05, 1.)
BIO_INITIAL = (1.5, 30., 10., 0., .01, .005, 0.)


@dataclass(frozen=True)
class ParameterSpace:
    names: tuple[str, ...]
    units: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    initial: tuple[float, ...]

    def __post_init__(self):
        if not (len(self.names) == len(self.units) == len(self.lower) ==
                len(self.upper) == len(self.initial)):
            raise ValueError("Parameter metadata lengths differ")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Every lower bound must be smaller than its upper bound")
        if any(not lo <= x <= hi for x, lo, hi in zip(self.initial, self.lower, self.upper)):
            raise ValueError("Initial parameters must lie within their bounds")

    @classmethod
    def propbr(cls, free_fsurf=False):
        tail = ("sigma", "fsurf") if free_fsurf else ("sigma",)
        return cls(BIO_NAMES + tail, BIO_UNITS + ("1",) * len(tail),
                   BIO_LOW + ((.01, .6) if free_fsurf else (.01,)),
                   BIO_HIGH + ((1., 3.5) if free_fsurf else (1.,)),
                   BIO_INITIAL + ((.3, 1.5) if free_fsurf else (.3,)))

    @classmethod
    def spropbr(cls, sensitivity=False):
        # Table 1 distinguishes mu inversion [0,30] from EFAST [0,2.5].
        return cls(BIO_NAMES + ("mu",), BIO_UNITS + ("1",), BIO_LOW + (0.,),
                   BIO_HIGH + ((2.5,) if sensitivity else (30.,)), BIO_INITIAL + (1.,))

    def bounds(self, like):
        return (torch.tensor(self.lower, dtype=like.dtype, device=like.device),
                torch.tensor(self.upper, dtype=like.dtype, device=like.device))

    def decode(self, z):
        lo, hi = self.bounds(z)
        return lo + (hi - lo) * torch.sigmoid(z)

    def encode(self, x):
        lo, hi = self.bounds(x)
        # Finite initialization for parameters whose table initial value is zero.
        u = ((x - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)
        return torch.logit(u)

    def validate(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.names) or x.shape[0] == 0:
            raise ValueError(f"Expected nonempty (samples, {len(self.names)}) parameters")
        lo, hi = self.bounds(x)
        if not bool(torch.isfinite(x).all()) or not bool(((x >= lo) & (x <= hi)).all()):
            raise ValueError(f"Parameters must be finite and within bounds for {self.names}")
