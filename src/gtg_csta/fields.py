"""Fields as extensors.

A field is a callable on a spacetime position `x` (an extensor of type `Spacetime`) returning an extensor:
  * connection  Omega(x): Bivector <- Spacetime        (the value Omega(a) on a direction a is a bivector)
  * versor      V(x):     Even (a Spin(2,4) element)
  * spinor      Psi(x):   Even
Derivatives are extensors too: `derivative(f)(x)` has the output of f and one more input, the step
direction, so `derivative(Omega)(x)(b, a)` is  a . grad Omega(b).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numga.backend.jax import derivative  # noqa: F401  (re-exported: exact, by autodiff)

from . import algebra as al

Spacetime = al.Spacetime


def position(x4):
    return al.point(x4)


def fd_derivative(f, h: float = 1e-4):
    """Central-difference counterpart of `derivative` (4th order); same result type: f's output plus one
    input, the step direction.  For testing the accuracy of the exact derivative, and for fields that are
    only available as samples."""
    def d(x):
        cols = []
        out = None
        for m in range(4):
            e = jnp.zeros(4).at[m].set(h)
            vals = [f(al.point(x.kernel + s * e)) for s in (2, 1, -1, -2)]
            out = vals[0]
            cols.append((8 * (vals[1].kernel - vals[2].kernel) - (vals[0].kernel - vals[3].kernel)) / (12 * h))
        gt = out.gatype
        step = al.ALG.gatype.oriented((*gt.subspaces, Spacetime.subspaces[0]),
                                      (*gt.orientations, Spacetime.orientations[0]))
        return al.CTX.extensor(step, jnp.stack(cols, axis=-1))
    return d


def get_derivative(method: str = "jax", h: float = 1e-4):
    """The derivative operator: 'jax' (exact autodiff) or 'fd' (central differences)."""
    return derivative if method == "jax" else (lambda f: fd_derivative(f, h))


def permute_inputs(T, perm):
    """Reorder the inputs of a multilinear map: new input j is old input perm[j].  For a bilinear map,
    permute_inputs(T, (1, 0))(a, b) == T(b, a)."""
    k = len(perm)
    gt = T.gatype
    subs = (gt.subspaces[0], *(gt.subspaces[1 + p] for p in perm))
    ors = (gt.orientations[0], *(gt.orientations[1 + p] for p in perm))
    nd = T.kernel.ndim
    axes = list(range(nd - k)) + [nd - k + p for p in perm]
    return al.CTX.extensor(al.ALG.gatype.oriented(subs, ors), jnp.transpose(T.kernel, axes))


def bivector(coefficients):
    """A bivector extensor from numga blade coefficients (..., 15)."""
    return al.mv.bivector(jnp.asarray(coefficients))


def even(coefficients):
    return al.mv.even(jnp.asarray(coefficients))


# ---------------- test fields (smooth, generic, x-dependent) ----------------

def random_versor_field(seed: int = 0, scale: float = 0.3):
    """V(x) = exp(B(x)), B a smooth, genuinely x-dependent bivector: a Spin(2,4) field."""
    rng = np.random.default_rng(seed)
    B0, B1, B2 = rng.normal(size=15) * scale, rng.normal(size=(15, 4)) * scale, rng.normal(size=(15, 4)) * scale

    def V(x):
        k = x.kernel
        return bivector(B0 + B1 @ k + B2 @ jnp.sin(1.3 * k)).exp()

    return V


def random_connection_field(seed: int = 0, scale: float = 0.3):
    """A generic (non-flat) smooth connection Omega(x): Bivector <- Spacetime."""
    rng = np.random.default_rng(seed)
    # drawn as (mu, blade[, nu]) like the array reference implementation, stored output-first
    A0, A1, A2 = (rng.normal(size=(4, 15)) * scale, rng.normal(size=(4, 15, 4)) * scale,
                  rng.normal(size=(4, 15, 4)) * scale)

    def Om(x):
        k = x.kernel
        return al.connection_from_columns(A0 + A1 @ k + A2 @ jnp.cos(0.9 * k))

    return Om


def random_spinor_field(seed: int = 0, scale: float = 0.5):
    rng = np.random.default_rng(seed)
    P0, P1, P2 = rng.normal(size=32) * scale, rng.normal(size=(32, 4)) * scale, rng.normal(size=(32, 4)) * scale

    def Psi(x):
        k = x.kernel
        return even(P0 + P1 @ k + P2 @ jnp.sin(0.7 * k))

    return Psi
