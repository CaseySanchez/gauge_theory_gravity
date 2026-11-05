"""Spin(2,4) versors: construction, checks, independent oracle."""
from __future__ import annotations

import numpy as np
import scipy.linalg

from .algebra import mv, ZERO_B
from .fields import biv, even


def exp_bivector(c_blade):
    """V = exp(B) from bivector blade coefficients (numga generic exp)."""
    return biv(c_blade).exp()


def unit_residual(V) -> float:
    """max |V ~V - 1| over even coefficients."""
    one = np.zeros(32)
    one[0] = 1.0
    return float(np.abs(np.asarray((V * ~V).kernel) - one).max())


def left_regular_even(B_blade) -> np.ndarray:
    """32x32 matrix of left multiplication by bivector B on the even subalgebra."""
    basis = even(np.eye(32))
    cols = np.asarray((biv(B_blade) * basis).kernel)      # row j = B * e_j
    return cols.T


def exp_oracle(B_blade) -> np.ndarray:
    """exp(B) via scipy matrix exponential of the left-regular representation, applied to 1.
    Independent of numga's exp."""
    L = left_regular_even(np.asarray(B_blade))
    one = np.zeros(32)
    one[0] = 1.0
    return scipy.linalg.expm(L) @ one
