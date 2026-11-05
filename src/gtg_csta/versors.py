"""Spin(2,4) versors: checks and an independent oracle for exp."""
from __future__ import annotations

import numpy as np
import scipy.linalg

from . import algebra as al
from .fields import bivector, even


def unit_residual(V) -> float:
    """max |V ~V - 1|."""
    return float(np.abs(np.asarray((V * ~V - 1).kernel)).max())


def left_regular_even(B_blade) -> np.ndarray:
    """32x32 matrix of left multiplication by a bivector on the even subalgebra, built as an extensor
    map  Even <- Even  and read out."""
    L = (bivector(B_blade) * al.Even)                  # Even <- Even
    return np.asarray(L.kernel)


def exp_oracle(B_blade) -> np.ndarray:
    """exp(B) via scipy's matrix exponential of the left-regular representation applied to 1; independent of
    numga's exp.  Returns even-subalgebra coefficients."""
    one = np.zeros(32)
    one[0] = 1.0
    return scipy.linalg.expm(left_regular_even(np.asarray(B_blade))) @ one
