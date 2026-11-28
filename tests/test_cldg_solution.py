"""Phase 0 of docs/PLAN_DIRAC_SOLUTION.md: the self-consistent Dirac cosmology of Challinor, Lasenby, Doran and Gull
(with Lambda, gr-qc/0311016) checked inside the GTG oracle, before the CSTA is involved.

  * The closed forms for rho, H and chi satisfy rho' = -3 rho H, chi' = m cos beta + 3 pi rho and the constraint
    3 H^2 - 12 pi^2 rho^2 - 8 pi m rho cos beta - Lambda = 0, for particle and antiparticle; chi is derived here,
    chi = +- m t - 1/2 ln(alpha coth(alpha t) +- m).
  * The Newtonian-gauge rotation gauge field H b ^ gamma_0 is LDG's closed form (4.53) for that h_bar.
  * With the contorsion (kappa/2) b . S, all of LDG's field equations vanish: the spinor equation, the h-equation
    G - Lambda = kappa T and the Omega-equation (spin-torsion), with or without the constant boost L. Without the
    contorsion, or with the opposite sign, the h- and Omega-equations fail.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al
import gtg_dirac_reference as ref
from cldg_solution import KAPPA, Cosmology

PARTICLE, ANTIPARTICLE = Cosmology(sign=1), Cosmology(sign=-1)
BOOSTED = Cosmology(sign=1, xi=0.4, eta=0.7)


@pytest.mark.parametrize("solution", [PARTICLE, ANTIPARTICLE, Cosmology(sign=1, coupling=4.0),
                                      Cosmology(sign=-1, coupling=2.0), Cosmology(sign=1, coupling=-1.0)])
@pytest.mark.parametrize("t", [0.2, 0.4])
def test_closed_forms_solve_the_reduced_equations(solution, t):
    """The thesis's equations, with m0 and a for the full transport (a = 1, m0 = m without it)."""
    rho, H = float(solution.density(t)), float(solution.hubble(t))
    m, a, s = solution.effective_mass, solution.contact, solution.sign
    assert rho > 0
    assert abs(float(jax.grad(solution.density)(t)) + 3 * rho * H) < 1e-12 * max(1.0, rho * H)
    assert abs(float(jax.grad(solution.phase)(t)) - (s * m + 3 * np.pi * a * rho)) < 1e-12
    constraint = 3 * H ** 2 - 12 * np.pi ** 2 * a * rho ** 2 - 8 * np.pi * m * rho * s
    assert abs(constraint - solution.cosmological) < 1e-12 * max(1.0, 3 * H ** 2)


def test_newtonian_gauge_rotation_field_is_ldg_closed_form():
    x4 = [0.4, 0.3, -0.2, 0.5]
    closed = ref.rotation_gauge(PARTICLE.coframe, x4)
    transcribed = PARTICLE.rotation_kernels(jnp.asarray(x4), torsion=False)
    assert max(float(np.abs(np.asarray((closed[mu] - al.mv.bivector(transcribed[mu])).kernel)).max())
               for mu in range(4)) < 1e-14


def equations(solution, x4, torsion=1.0):
    def rotation(y):
        bare = solution.rotation_kernels(y, torsion=False)
        return bare + torsion * (solution.rotation_kernels(y) - bare)
    return ref.field_equations(solution.coframe, rotation, solution.spinor, x4, solution.mass, KAPPA,
                               solution.cosmological)


@pytest.mark.parametrize("solution,x4", [(PARTICLE, [0.4, 0.3, -0.2, 0.5]), (ANTIPARTICLE, [0.3, -0.6, 0.1, 0.2]),
                                         (BOOSTED, [0.25, 0.1, 0.4, -0.3])])
def test_all_ldg_field_equations_vanish(solution, x4):
    spinor, coframe, connection = equations(solution, x4)
    assert np.abs(spinor).max() < 1e-12
    assert np.abs(coframe).max() < 1e-12
    assert np.abs(connection).max() < 1e-12


@pytest.mark.parametrize("torsion", [0.0, -1.0])
def test_without_ldg_contorsion_it_is_not_a_solution(torsion):
    _, coframe, connection = equations(PARTICLE, [0.4, 0.3, -0.2, 0.5], torsion)
    assert np.abs(coframe).max() > 0.1 and np.abs(connection).max() > 0.1
