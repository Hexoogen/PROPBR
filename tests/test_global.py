import numpy as np
import pytest
from propbr import PROPBR, SPROPBR, global_sensitivity


def test_efast_sample_counts_and_model_specific_bounds():
    pytest.importorskip("SALib")
    p = PROPBR(kernel="analytic", wavelengths=[550., 850., 1450.])
    r = global_sensitivity(p, n=129, include_phase=True, n_resamples=5)
    assert r["evaluations"] == 129 * 9
    assert r["S1"].shape == (1, 3, 9)
    assert np.isfinite(r["S1"]).all()
    s = global_sensitivity(SPROPBR(wavelengths=[550., 850.]), n=129, n_resamples=5)
    assert s["bounds"][-1] == (0., 2.5)
    assert s["evaluations"] == 129 * 8
