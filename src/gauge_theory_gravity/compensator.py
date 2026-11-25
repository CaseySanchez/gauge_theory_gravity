"""The scale tractor: a compensator for the conformal connection, the CSTA form of Penrose's infinity twistor.

A choice of scale sigma (a positive function, the metric g/sigma^2 being the one picked out) gives a field of vectors
of R^{2,4},

    I = sigma n_inf + grad sigma + rho n_o,      rho = -1/4 (box sigma + J sigma),

with grad sigma the frame gradient raised to a spacetime vector, box the wave operator of g and J = R/6 the trace of
the Schouten tensor.  The normal conformal connection moves it by the tractor derivative

    D_a I = a . grad I + Omega(a) x I,

the commutator action that `cartan` uses for the covariant derivative of directions.  The standard facts it carries
(Bailey-Eastwood-Gover; Gover-Nurowski), checked in `tests/test_compensator.py`:  I transforms under a Weyl
rescaling as the gauge versor of `tests/test_weyl_rescaling.py` says,  I . I = 2 sigma rho + |grad sigma|^2  is
Weyl-invariant and equals -J/2 = -R/12 at sigma = 1, and  D I = 0  exactly when g/sigma^2 is Einstein, with
I = n_inf (sigma = 1) when the metric is Ricci-flat.  The non-null case I . I != 0 is the cosmological constant.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from . import cartan, conformal
from .cartan import Coframe
from .curvature import curvature
from .fields import Field, derivative

Direction = al.Direction
Scale = Callable[[Extensor], Extensor]             # position -> a positive scalar


def _raised_gradient(coframe: Coframe, scale: Scale) -> Field:
    """x -> g^{mu nu} d_nu sigma, a coordinate direction."""
    gradient = derivative(scale)                    # Scalar <- Direction
    duality = cartan.metric_map(coframe)            # Slice <- Direction

    def raised(x: Extensor) -> Extensor:
        covector = (gradient(x)(al.DIRECTIONS) * al.SLICES).sum(axis=0)          # [] Slice
        return duality(x).inverse()(covector)                                     # [] Direction
    return raised


def wave_operator(coframe: Coframe, scale: Scale) -> Field:
    """x -> box sigma = g^{mu nu} (d_mu d_nu sigma - Gamma^lam_{mu nu} d_lam sigma), a scalar."""
    hessian = derivative(derivative(scale))         # Scalar <- (Direction, step Direction)
    symbols = cartan.christoffel_lower(coframe)     # Gamma_{lam mu nu}
    raised = _raised_gradient(coframe, scale)
    duality = cartan.metric_map(coframe)

    def box(x: Extensor) -> Extensor:
        covariant = hessian(x) - symbols(x)(raised(x), Direction, Direction)      # Scalar <- (mu, nu)
        inverse = duality(x).inverse()                                            # Direction <- Slice
        return covariant(inverse(al.SLICES), al.DIRECTIONS).sum(axis=0)
    return box


def schouten_trace(coframe: Coframe) -> Field:
    """x -> J = P^a_a = R/6."""
    schouten = conformal.schouten(coframe)

    def trace(x: Extensor) -> Extensor:
        return schouten(x)(cartan.frame_at(coframe, x)).trace()
    return trace


def scale_tractor(coframe: Coframe, scale: Scale) -> Field:
    """x -> I(x), a vector of R^{2,4}: sigma n_inf + grad sigma + rho n_o."""
    raised = _raised_gradient(coframe, scale)
    box = wave_operator(coframe, scale)
    trace = schouten_trace(coframe)

    def tractor(x: Extensor) -> Extensor:
        sigma = scale(x)
        across = coframe(x)(raised(x)).dual_inverse()                             # [] Vector, frame gradient
        rho = -0.25 * (box(x) + trace(x) * sigma)
        return sigma * al.n_inf + across + rho * al.n_o
    return tractor


def tractor_derivative(omega: Field, tractor: Field) -> Field:
    """x -> D I : Vector <- Direction, the covariant derivative of a field of R^{2,4} vectors along a direction."""
    gradient = derivative(tractor)

    def covariant(x: Extensor) -> Extensor:
        return gradient(x) + omega(x).commutator(tractor(x))
    return covariant


def unit_scale(x: Extensor) -> Extensor:
    """sigma = 1: the metric itself is the one picked out."""
    return 0 * cartan.coordinate(x, 0) + 1.0


# --- the integrability condition -----------------------------------------------------------------
BASIS = [al.t, al.x, al.y, al.z, al.p, al.n]        # R^{2,4}


def parallel_candidates(omega: Field, x: Extensor, tolerance: float = 1e-9) -> list[Extensor]:
    """The vectors I with F(a, b) x I = 0 for all a, b at x: a parallel tractor must lie among them, since
    [D_a, D_b] I = F(a, b) x I.  Returned as an orthonormal (in coefficients) basis of the null space, so an empty
    list means no parallel tractor through x, and so no Einstein metric in the conformal class near x."""
    strength = curvature(omega)(x)
    columns = [np.asarray(strength.commutator(v).kernel).ravel() for v in BASIS]
    _, singular, rows = np.linalg.svd(np.stack(columns, axis=1))
    scale = max(float(singular[0]), 1.0)
    singular = np.concatenate([singular, np.zeros(len(BASIS) - len(singular))])
    null = rows[singular < tolerance * scale]
    return [sum((float(c) * v for c, v in zip(row, BASIS)), 0 * al.t) for row in null]
