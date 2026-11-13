"""Path-ordered parallel transport (spec section 8), on extensors.

Parallel transport of Psi along gamma solves gamma'.D Psi = 0, i.e. dPsi/ds = -1/2 Omega(gamma') Psi, so
Psi(s) = T(s) Psi(0) with dT/ds = -1/2 Omega(gamma'(s)) T.  Later points multiply on the LEFT:
T = exp(-1/2 Omega_N ds) ... exp(-1/2 Omega_1 ds), the spec's P exp(-1/2 int Omega).
All steps of a path are one batch: the positions, the velocities `derivative(path)`, the connection bound to them,
and the product of the batch of exponentials, `cumprod`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numga.backend.jax import derivative
from numga.extensor import Extensor

from . import algebra as al
from .fields import Field

Path = Callable[[Extensor], Extensor]               # Scalar parameter -> Spacetime position
UNIT = al.mv.scalar([1.0])                          # [] Scalar+
GAUSS = np.sqrt(3.0) / 6.0


@dataclass(frozen=True)
class Line:
    """The straight path from `start` at parameter 0 to `end` at parameter 1. A batch of lines has batch axes in
    `start`, and its velocity is known exactly, so no derivative of the path is needed."""
    start: Extensor
    end: Extensor

    def __call__(self, parameter: Extensor) -> Extensor:
        return self.start + parameter * (self.end - self.start)                         # Spacetime

    def velocity(self, parameter: Extensor) -> Extensor:
        return self.end - self.start                                                     # Spacetime, constant along the line


def straight_line(start: Extensor, end: Extensor) -> Line:
    return Line(start, end)


def velocity_along(path: Path, parameters: Extensor) -> Extensor:
    """The path's velocity at a batch of parameters: exact for a `Line`, else the derivative of the path."""
    if isinstance(path, Line):
        return path.velocity(parameters)
    return derivative(path)(parameters)(UNIT)                                           # [steps] Spacetime


def connection_along(omega: Field, path: Path, parameters: Extensor) -> Extensor:
    """Omega(gamma'(s)) at a batch of parameters: the connection bound to the path's velocity, a bivector."""
    return omega(path(parameters))(velocity_along(path, parameters))                     # [..., steps] Bivector


def step_versors(omega: Field, path: Path, steps: int, start: float, stop: float, scheme: str) -> Extensor:
    """The versor of each step, `exp(-1/2 Omega ds)` at the midpoint, or by the fourth-order Magnus formula."""
    size = (stop - start) / steps
    begin = al.mv.scalar((start + size * np.arange(steps))[:, None])                    # [steps] Scalar+
    if scheme == "midpoint":
        return (-0.5 * size * connection_along(omega, path, begin + 0.5 * size)).exp()
    if scheme == "magnus4":
        first = -0.5 * connection_along(omega, path, begin + (0.5 - GAUSS) * size)
        second = -0.5 * connection_along(omega, path, begin + (0.5 + GAUSS) * size)
        # numga's commutator is [,]/2, so the Magnus term -sqrt(3)/12 h^2 [A1, A2] is sqrt(3)/6 h^2 times it.
        return (0.5 * size * (first + second) - GAUSS * size**2 * first.commutator(second)).exp()
    raise ValueError(scheme)


def transport(omega: Field, path: Path, steps: int = 200, start: float = 0.0, stop: float = 1.0,
              scheme: str = "midpoint") -> Extensor:
    """The versor T along `path`, parameter from `start` to `stop`. scheme: 'midpoint' (2nd order) or
    'magnus4' (4th order, two Gauss points)."""
    versors = step_versors(omega, path, steps, start, stop, scheme)                      # [..., steps] Even
    # numga's cumprod multiplies later elements on the left, `K[k] ... K[0]`: the last of them is the transport.
    return versors.cumprod(axis=-1)[..., -1]                                            # [...] Even


def holonomy_loop(omega: Field, start: Extensor, mu: int, nu: int, size: float,
                  steps_per_side: int = 40, scheme: str = "midpoint") -> Extensor:
    """Transport around the rectangle start -> +mu -> +nu -> -mu -> -nu by composing its four legs.
    Verified: T = exp(-1/2 size^2 F(e_mu, e_nu)) + O(size^3)."""
    along, across = size * al.SPACETIME[mu], size * al.SPACETIME[nu]
    corners = [start, start + along, start + along + across, start + across, start]
    holonomy = None
    for corner, following in zip(corners[:-1], corners[1:]):
        leg = transport(omega, straight_line(corner, following), steps_per_side, scheme=scheme)
        holonomy = leg if holonomy is None else leg * holonomy                           # later legs on the left
    return holonomy                                                                      # [] Even
