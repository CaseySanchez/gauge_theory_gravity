"""Phase 2 of docs/PLAN_DIRAC.md: the independent GTG oracle (`gtg_dirac_reference`, Lasenby, Doran and Gull's
formulas in the spacetime algebra) checked against itself and against the CSTA.

  * LDG's closed solution of the wedge equation (4.49, 4.53) is the repo's Levi-Civita Lorentz connection, on the
    Painleve-Gullstrand, Kerr-Schild Kerr, static-lapse and de Sitter coframes;
  * their Ricci scalar (4.2, 4.11, 4.12) has the repo's sign (de Sitter: R = -4 Lambda), so LDG's 1/2 R is -S1/4 at
    f = 0 (`test_dirac_conventions`);
  * the oracle's Euler-Lagrange derivative of the Dirac action (3.56) is 2 |det e| <b_A gamma_0 ~R>, R the residual of
    LDG's coupled Dirac equation (3.62) with its torsion-trace term, with and without torsion;
  * the oracle's Dirac density equals the CSTA density with the Lorentz coupling through `dirac.from_sta`, with and
    without torsion and mass.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, dirac as dr
import gtg_dirac_reference as ref

t, x, y, z = al.SPACETIME
BLADES = [(1 + 0 * t * t).cast(al.Even), *[b.cast(al.Even) for b in (t * x, t * y, t * z, x * y, x * z, y * z, t * x * y * z)]]
RNG = np.random.default_rng(4)
CONSTANT, LINEAR, QUADRATIC = RNG.normal(size=8) * 0.6, RNG.normal(size=(8, 4)) * 0.3, RNG.normal(size=(8, 4, 4)) * 0.1
# A constant Lorentz piece added to the connection: torsion, with a nonzero torsion trace on Kerr.
TWIST = 0.1 * al.GENS[3] * (al.SLICES[1] & al.Direction) + 0.07 * al.GENS[1] * (al.SLICES[3] & al.Direction)
GEOMETRIES = {
    "painleve_gullstrand": (cartan.schwarzschild(1.0), [0.3, 1.7, -0.9, 1.1]),
    "kerr": (cartan.kerr(1.0, 0.9), [-0.4, 2.5, 0.8, -1.3]),
    "static_lapse": (cartan.static_lapse(), [1.3, 0.4, -0.7, 0.9]),
    "de_sitter": (cartan.de_sitter(0.5), [0.7, 0.4, -0.7, 0.9]),
}


def sta_spinor(coordinates):
    """psi(x), quadratic in the coordinates, an even element of the spacetime algebra."""
    terms = []
    for k in range(8):
        weight = CONSTANT[k] + sum(LINEAR[k, m] * coordinates[m] for m in range(4))
        weight = weight + sum(QUADRATIC[k, m, n] * coordinates[m] * coordinates[n] for m in range(4) for n in range(4))
        terms.append(weight * BLADES[k])
    return sum(terms, 0 * BLADES[0]).cast(al.Even)


def oracle_spinor(x4):
    return sta_spinor([x4[m] for m in range(4)]).kernel


def csta_spinor(position):
    return dr.from_sta(sta_spinor([cartan.coordinate(position, m) for m in range(4)]))


def traceable(geometry):
    """x4 -> e[a, mu], traceable by jax: the oracle's only input from a geometry."""
    def coframe(x4):
        e = geometry(al.point(x4))
        return jnp.stack([jnp.stack([(al.SLICES[a] & e(al.DIRECTIONS[mu])).kernel.reshape(())
                                     for mu in range(4)]) for a in range(4)])
    return coframe


def lorentz_kernels(geometry, torsion):
    def kernels(x4):
        connection = cartan.lorentz_connection(geometry)(al.point(x4)) + torsion * TWIST
        return jnp.stack([connection(al.DIRECTIONS[mu]).kernel for mu in range(4)])
    return kernels


def size(e):
    return float(np.abs(np.asarray(e.kernel)).max())


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_closed_form_rotation_gauge_field_is_levi_civita(name):
    geometry, x4 = GEOMETRIES[name]
    closed = ref.rotation_gauge(traceable(geometry), x4)
    levi_civita = cartan.lorentz_connection(geometry)(al.point(x4))
    scale = max(size(levi_civita(al.DIRECTIONS[mu])) for mu in range(4))
    assert scale > 0.1
    assert max(size(closed[mu] - levi_civita(al.DIRECTIONS[mu])) for mu in range(4)) < 1e-14


@pytest.mark.parametrize("name", ["static_lapse", "de_sitter"])
def test_ricci_scalar_has_the_repo_sign(name):
    geometry, x4 = GEOMETRIES[name]
    oracle = ref.ricci_scalar(traceable(geometry), lorentz_kernels(geometry, 0), x4)
    repo = float(np.asarray(cf.ricci_scalar(geometry, al.point(x4)).kernel).reshape(-1)[0])
    assert abs(oracle - repo) < 1e-12
    if name == "de_sitter":
        assert abs(oracle + 4 * 3 * 0.5 ** 2) < 1e-12                                      # R = -4 Lambda


@pytest.mark.parametrize("torsion", [0, 1])
def test_oracle_action_gives_the_coupled_dirac_equation(torsion):
    geometry, x4 = GEOMETRIES["kerr"]
    coframe, kernels = traceable(geometry), lorentz_kernels(geometry, torsion)
    trace = ref.coefficients(ref.torsion_trace(coframe, kernels, x4))
    assert (np.abs(trace).max() < 1e-12) if torsion == 0 else (np.abs(trace).max() > 1e-2)
    equations = ref.euler_lagrange(coframe, kernels, oracle_spinor, x4, mass=0.7)
    residual = ref.dirac_residual(coframe, kernels, oracle_spinor, x4, mass=0.7)
    volume = abs(np.linalg.det(np.asarray(coframe(jnp.asarray(x4)))))
    indices = [int(np.flatnonzero(np.abs(np.asarray(b.kernel)) > 0.5)[0]) for b in BLADES]
    signs = [float(np.asarray(b.kernel)[i]) for b, i in zip(BLADES, indices)]
    by_blade = np.array([equations[i] * s for i, s in zip(indices, signs)])
    expected = np.array([2 * volume * ref.scalar(b * t * ~residual) for b in BLADES])
    assert np.abs(by_blade - expected).max() < 1e-12
    assert np.abs(by_blade).max() > 0.1


@pytest.mark.parametrize("name", ["painleve_gullstrand", "kerr"])
@pytest.mark.parametrize("torsion", [0, 1])
@pytest.mark.parametrize("mass", [0.0, 0.7])
def test_oracle_density_is_the_csta_density_with_lorentz_coupling(name, torsion, mass):
    geometry, x4 = GEOMETRIES[name]
    closed = ref.rotation_gauge(traceable(geometry), x4)

    def rotation(_):
        return [closed[mu] + torsion * TWIST(al.DIRECTIONS[mu]) for mu in range(4)]

    def omega(position):
        return cf.connection(geometry, lambda p: 0.25 * geometry(p))(position) + torsion * TWIST

    oracle = ref.dirac_density(traceable(geometry), rotation, oracle_spinor, x4, mass)
    csta = float(dr.density(omega, csta_spinor, al.point(x4), mass, "lorentz"))
    assert abs(oracle - csta) < 1e-13 * max(1.0, abs(oracle))
