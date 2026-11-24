"""Tier B: translation sector as coframe; normal conformal Cartan connection (spec 9.2, 10), on extensors.

    Omega = Omega_Lorentz  +  e ^ n_inf  +  f ^ n_o  +  b (n_o ^ n_inf)

with e the coframe (Direction <- Direction), f another such map (the special-conformal part) and b a form
(the dilation part).  With directions intrinsic, the translation generator along a direction d is the dual of its
join with the point at infinity, `(d & INFINITY).dual() = e_d ^ n_inf`, and the special-conformal one the dual of its
join with the origin, as in numga's conformal examples.  Each term is a map Bivector <- Direction.
"""
from __future__ import annotations

import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from . import cartan
from .cartan import Coframe
from .curvature import curvature
from .fields import Field

Direction = al.Direction
CoframeType = cartan.CoframeType
Dilation = al.n_o ^ al.n_inf                                    # [] Bivector+


def connection(coframe: Coframe, special: Field | None = None, dilation: Field | None = None) -> Field:
    """x -> Omega(x): Bivector <- Direction.  `special(x)`: Direction <- Direction;  `dilation(x)`: Scalar <- Direction."""
    lorentz = cartan.lorentz_connection(coframe)

    def full(x: Extensor) -> Extensor:
        total = lorentz(x) + (coframe(x) & al.INFINITY).dual().cast(al.Bivector)
        if special is not None:
            total = total + (special(x) & al.ORIGIN).dual().cast(al.Bivector)
        if dilation is not None:
            total = total + (dilation(x) * Dilation).cast(al.Bivector)
        return total

    return full


def ricci_vector(strength: Extensor, frame: Extensor) -> Extensor:
    """Ric(nu) = sum_a e^a _| F_M(E_a, nu): the trace of the Lorentz-sector curvature over its first coordinate
    input and one frame index, a map Direction <- Direction (= R^b_nu, frame index up).  On a direction c the
    bivector acts by its commutator, `c _| B` dualized being `-B.commutator(c)`; the frame input a and c are then
    contracted with the metric."""
    lorentz = al.sector(strength, "M")
    return (-lorentz(frame, Direction).commutator(Direction)).contract(1, 3).cast(Direction)


def ricci_scalar(coframe: Coframe, x: Extensor) -> Extensor:
    """R: the trace of the Ricci map over the frame."""
    frame = cartan.frame_at(coframe, x)
    strength = curvature(cartan.lorentz_connection(coframe))(x)
    return ricci_vector(strength, frame)(frame).trace()


def schouten(coframe: Coframe) -> Field:
    """x -> P = 1/2 (Ric - R e / 6): Direction <- Direction (frame index up, coordinate index down), from the
    curvature of the Lorentz connection alone."""
    strength = curvature(cartan.lorentz_connection(coframe))

    def tensor(x: Extensor) -> Extensor:
        e = coframe(x)
        frame = cartan.frame_at(coframe, x)
        ricci = ricci_vector(strength(x), frame)                                 # R^b_nu
        return 0.5 * (ricci - (ricci(frame).trace() / 6.0) * e)

    return tensor


# --- the normal conditions, solved for f ---------------------------------------------------------
def normal_residual(coframe: Coframe, x: Extensor, special: Extensor) -> tuple[Extensor, Extensor]:
    """The normal conditions for a special-conformal part `special` (Direction <- Direction) held constant near x:
    the dilation part of F, a map Bivector <- (a, b), and the trace of its Lorentz part, a map Direction <- Direction.
    Both vanish when the conditions hold, and both are affine in `special`."""
    def held(_: Extensor) -> Extensor:
        return special

    strength = curvature(connection(coframe, held))(x)
    return al.sector(strength, "D"), ricci_vector(strength, cartan.frame_at(coframe, x))


def solve_normal_f(coframe: Coframe, x: Extensor) -> tuple[Extensor, float, int]:
    """Least-squares f making the normal conditions hold, by solving the 16-unknown linear system numerically.
    Returns f (Direction <- Direction), the largest residual left, and the rank of the system."""
    unit = [al.CTX.extensor(CoframeType, row.reshape(4, 4)) for row in np.eye(16)]
    base = [np.asarray(c.kernel).ravel() for c in normal_residual(coframe, x, 0 * coframe(x))]
    columns = []
    for direction in unit:
        parts = [np.asarray(c.kernel).ravel() for c in normal_residual(coframe, x, direction)]
        columns.append(np.concatenate([p - b for p, b in zip(parts, base)]))
    jacobian = np.stack(columns, axis=1)
    offset = np.concatenate(base)
    solution, *_ = np.linalg.lstsq(jacobian, -offset, rcond=None)
    left = np.concatenate([np.asarray(c.kernel).ravel() for c in
                           normal_residual(coframe, x, al.CTX.extensor(CoframeType, solution.reshape(4, 4)))])
    return al.CTX.extensor(CoframeType, solution.reshape(4, 4)), float(np.abs(left).max()), int(np.linalg.matrix_rank(jacobian))
