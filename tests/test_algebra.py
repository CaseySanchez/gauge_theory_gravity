"""Phase 1: spec sections 2-4."""
import numpy as np
import gtg_csta  # noqa: F401
from gtg_csta import algebra as al


def sc(a):
    return al.scalar(a)


def test_basis_squares():
    sq = [sc(e * e) for e in (al.t, al.x, al.y, al.z, al.p, al.n)]
    assert sq == [1, -1, -1, -1, 1, -1]


def test_basis_orthogonal():
    es = [al.t, al.x, al.y, al.z, al.p, al.n]
    for i in range(6):
        for j in range(i):
            assert np.abs(np.asarray((es[i] * es[j] + es[j] * es[i]).kernel)).max() < 1e-14


def test_dimensions():
    assert al.ALG.subspace.bivector().__len__() == 15 if hasattr(al.ALG.subspace.bivector(), "__len__") else True
    assert al.GEN.shape == (15, 15)


def test_null_basis():
    assert abs(sc(al.n_o * al.n_o)) < 1e-15
    assert abs(sc(al.n_inf * al.n_inf)) < 1e-15
    assert abs(sc(al.n_o.dot(al.n_inf)) - 1) < 1e-14
    # n_o n_inf + n_inf n_o = 2
    assert abs(sc(al.n_o * al.n_inf + al.n_inf * al.n_o) - 2) < 1e-14


def test_generators_span_bivectors():
    assert np.linalg.matrix_rank(al.GEN) == 15
    assert abs(np.linalg.det(al.GEN)) > 1e-3


def test_roundtrip_coordinates():
    c = np.random.default_rng(0).normal(size=(3, 15))
    assert np.abs(al.to_generators(al.to_blades(c)) - c).max() < 1e-13


def test_structure_constants_so24():
    f = al.structure_constants()
    S = al.SECTORS
    # antisymmetry, Jacobi
    assert np.abs(f + f.transpose(1, 0, 2)).max() < 1e-12
    J = (np.einsum("ijm,mkl->ijkl", f, f) + np.einsum("jkm,mil->ijkl", f, f)
         + np.einsum("kim,mjl->ijkl", f, f))
    assert np.abs(J).max() < 1e-12

    def block(a, b, c):
        return f[S[a], S[b]][..., S[c]]

    # [M,M] in M
    assert np.abs(f[S["M"], S["M"]][..., 6:]).max() < 1e-12
    # [M,P] in P, [M,K] in K, [M,D] = 0
    assert np.abs(f[S["M"], S["P"]][..., [*range(6), *range(10, 15)]]).max() < 1e-12
    assert np.abs(f[S["M"], S["K"]][..., [*range(10), 14]]).max() < 1e-12
    assert np.abs(f[S["M"], S["D"]]).max() < 1e-12
    # [P,P] = [K,K] = 0
    assert np.abs(f[S["P"], S["P"]]).max() < 1e-12
    assert np.abs(f[S["K"], S["K"]]).max() < 1e-12
    # [D,P] ~ P, [D,K] ~ K (grading), nonzero, opposite signs
    dp = f[14, S["P"]][:, S["P"]]
    dk = f[14, S["K"]][:, S["K"]]
    assert np.abs(dp).max() > 1 and np.abs(dk).max() > 1
    assert np.abs(f[14, S["P"]][:, [*range(6), *range(10, 15)]]).max() < 1e-12
    assert np.abs(f[14, S["K"]][:, [*range(10), 14]]).max() < 1e-12
    assert np.sign(np.trace(dp)) == -np.sign(np.trace(dk))
    # [P,K] in M + D
    assert np.abs(f[S["P"], S["K"]][..., 6:14]).max() < 1e-12
    assert np.abs(f[S["P"], S["K"]]).max() > 1
