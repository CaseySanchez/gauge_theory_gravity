"""Fields as extensors.

A field is a function of a spacetime position `x`, an extensor of type `Spacetime`, and returns an extensor:
  * connection  Omega(x): Bivector <- Spacetime       (the value `Omega(a)` on a direction a is a bivector)
  * versor      V(x):     Even, a Spin(2,4) element
  * spinor      Psi(x):   Even
Its derivative is an extensor too: `derivative(f)(x)` has the output of f and one more input, the step
direction, so `derivative(Omega)(x)(b, a)` is  a . grad Omega(b).
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial, reduce
from operator import add

import numpy as np
from numga import Algebra
from numga.backend.jax import derivative  # noqa: F401  (re-exported: exact, by autodiff)
from numga.extensor import Extensor
from numga.gatype import GAType

from . import algebra as al

Spacetime = al.Spacetime
Field = Callable[[Extensor], Extensor]
# The step direction of a derivative: one more input, in the spacetime directions.
STEP = (Spacetime.subspaces[0], Spacetime.orientations[0])


def position(x4) -> Extensor:
    """A spacetime position (t, x, y, z) as an extensor."""
    return al.point(x4)


def bivector(coefficients) -> Extensor:
    """A bivector from numga blade coefficients (..., 15)."""
    return al.mv.bivector(coefficients)


def even(coefficients) -> Extensor:
    """An even multivector from its coefficients (..., 32)."""
    return al.mv.even(coefficients)


def with_inputs(output: GAType, n: int) -> GAType:
    """The type of `output` with `n` more inputs, each a spacetime direction."""
    subspaces, orientations = output.subspaces, output.orientations
    return al.ALG.gatype.oriented((*subspaces, *[STEP[0]] * n), (*orientations, *[STEP[1]] * n))


def permute_inputs(map_: Extensor, order: tuple[int, ...]) -> Extensor:
    """Reorder the inputs of a multilinear map: new input j is old input order[j], so that for a bilinear map
    `permute_inputs(T, (1, 0))(a, b) == T(b, a)`. numga has no axis permutation for multilinear maps."""
    n = len(order)
    kind = map_.gatype
    subspaces = (kind.subspaces[0], *(kind.subspaces[1 + k] for k in order))
    orientations = (kind.orientations[0], *(kind.orientations[1 + k] for k in order))
    axes = [*range(map_.kernel.ndim - n), *(map_.kernel.ndim - n + k for k in order)]
    return al.CTX.extensor(al.ALG.gatype.oriented(subspaces, orientations), map_.kernel.transpose(axes))


def finite_difference(field: Field, step: float = 1e-4) -> Field:
    """Central-difference counterpart of `derivative` (4th order): the same result type, f's output with one more
    input, the step direction. It is for testing the exact derivative, and for fields available only as samples.
    The partial derivative along a basis vector `e` is a map; it is paired with the coordinate form of the step,
    `eta_ee (e | step)`, and the four are summed."""
    def difference(x: Extensor) -> Extensor:
        pieces = []
        for axis, basis in enumerate(al.SPACETIME):
            around = [field(x + (k * step) * basis) for k in (2, 1, -1, -2)]
            partial = (8 * (around[1] - around[2]) - (around[0] - around[3])) / (12 * step)
            pieces.append(partial * (al.ETA4[axis, axis] * (basis | Spacetime)))
        return reduce(add, pieces)
    return difference


def derivative_of(method: str = "jax", step: float = 1e-4) -> Callable[[Field], Field]:
    """The derivative operator: 'jax' (exact autodiff) or 'fd' (central differences with the given step)."""
    return derivative if method == "jax" else partial(finite_difference, step=step)


# ---------------- smooth test fields: polynomials in the position ----------------

def polynomial(output: GAType, seed: int = 0, scale: float = 0.3, degree: int = 2) -> Field:
    """A smooth, genuinely x-dependent field of type `output`: a random polynomial in the position,
    `c0 + c1(x) + c2(x, x) + ...`, each coefficient a multilinear map with the output's own inputs first
    and then one input per power of x, bound with the position."""
    rng = np.random.default_rng(seed)
    terms = []
    for power in range(degree + 1):
        kind = with_inputs(output, power)
        shape = tuple(len(space) for space in kind.subspaces)
        terms.append(al.CTX.extensor(kind, rng.normal(size=shape) * scale))
    n_free = output.arity

    def evaluate(x: Extensor) -> Extensor:
        total = terms[0]
        for power, term in enumerate(terms[1:], start=1):
            total = total + term(*[al.ALG.gatype(Spacetime)] * n_free, *[x] * power)
        return total

    return evaluate


def random_versor_field(seed: int = 0, scale: float = 0.3) -> Field:
    """V(x) = exp(B(x)), B a smooth x-dependent bivector: a Spin(2,4) field."""
    bivectors = polynomial(al.Bivector, seed, scale)

    def versor(x: Extensor) -> Extensor:
        return bivectors(x).exp()

    return versor


def random_connection_field(seed: int = 0, scale: float = 0.3, degree: int = 2) -> Field:
    """A generic (non-flat) smooth connection Omega(x): Bivector <- Spacetime."""
    return polynomial(al.ConnectionType, seed, scale, degree)


def random_spinor_field(seed: int = 0, scale: float = 0.5) -> Field:
    return polynomial(al.Even, seed, scale)
