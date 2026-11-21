# Gauge Theory Gravity in Conformal Spacetime Algebra

This repository holds a proposal for formulating Gauge Theory Gravity (GTG) in the real conformal spacetime algebra
$\mathrm{Cl}(2,4;\mathbb R)$, and a numerical test-bed that checks it. The proposal is reproduced in full
[below](#the-proposal-conformal-spacetime-algebra-gauge-theory-of-gravity) (also as
[a PDF](Gauge_Theory_Gravity_in_Conformal_Spacetime_Algebra.pdf)).

The test-bed, `src/gauge_theory_gravity`, is written on [numga](https://github.com/EelcoHoogendoorn/numga) extensors with JAX
autodiff. It implements the 15-component bivector connection $\Omega(a,x)$, versor gauge transformations, the covariant
derivative, the curvature and path-ordered transport, and evaluates them on given spacetimes. It does not solve any
field equations: the proposal leaves the dynamics open, and so does the code.

## What has been checked

- **Every formula in sections 4–8 holds as written**, including each factor of $\tfrac12$: gauge covariance of
  $\mathcal D_a$, $[\mathcal D_a,\mathcal D_b]\Psi=\tfrac12\mathcal F(a,b)\Psi$, the Bianchi identity, flatness of pure
  gauge, transport as a versor, and holonomy $\approx\exp(-\tfrac12\epsilon^2\mathcal F)$ around small loops.
- **Lorentz sector (9.1):** the Levi-Civita connection of a tetrad, embedded in the Lorentz sector, has curvature equal
  to the Riemann tensor computed independently from Christoffel symbols. A general $\mathrm{Spin}(2,4)$ transformation
  takes it out of the sector, so the restriction is a gauge choice.
- **Translation sector as coframe (9.2, 10):** with the coframe in the translation sector the connection is torsion-free,
  the normal conditions fix the special-conformal part to the Schouten tensor, and the quadratic curvature density is
  $\tfrac12 C_{abcd}C^{abcd}$ (Weyl squared), as section 10 anticipates.
- **Kerr:** GTG's Kerr–Schild position gauge field and its rotation gauge field are exactly the translation and Lorentz
  sectors: torsion-free, vacuum, special-conformal part zero, and the curvature density reproduces the Kerr Kretschmann
  scalar. They agree with numga's independent GTG Kerr code to $10^{-16}$.
- **Ray tracing:** light is followed by parallel transport with the connection itself. The Schwarzschild shadow radius
  agrees with the analytic value to 0.2%.

Not established: any action or field equations, recovery of Einstein gravity (11.4), the role of the dilation field
(11.3), and GTG gauges other than Kerr–Schild and Painlevé–Gullstrand. [RESULTS.md](RESULTS.md) records every claim with
its test and tolerance; [DEEP_DIVE.md](DEEP_DIVE.md) explains the construction; [PLAN.md](PLAN.md) is the original plan.

| Schwarzschild | Kerr, $a=0.9$ |
|---|---|
| ![Schwarzschild disk](output/render_lensing.png) | ![Kerr disk](output/render_kerr.png) |

Animated: [Kepler flow](output/render_lensing.gif), [rotating polarizer](output/render_polarizer.gif). Each image pairs the
camera view (blackbody colour of the Page–Thorne disk, shifted by gravitational redshift and Doppler motion) with a map of
the frequency ratio observed/emitted.

## Running it

Python 3.11 or later. numga is installed from GitHub at the commit this code was tested against (`8b22fde9`); its
`examples` package comes with it and is used by the Kerr cross-check test:

```sh
python -m venv .venv && source .venv/bin/activate
pip install "numga[linalg,jax,examples,test] @ git+https://github.com/EelcoHoogendoorn/numga@8b22fde9" h5py vtk pillow
pip install -U "jax[cuda12]"          # optional, for an NVIDIA GPU
pytest                                 # 150 tests, about 19 min on an RTX 3060
pytest -m slow tests/test_export.py    # compiles the full curvature chain on a grid
cd src
python -m gauge_theory_gravity.examples.scenarios                                # figures in ../output
python -m gauge_theory_gravity.examples.render slices holonomy lensing kerr      # lensing and kerr take about an hour each
python -m gauge_theory_gravity.export flyby --n 48 --frames 16                   # ParaView time series in ./export (not in git)
```

The package sets `--xla_gpu_enable_triton_gemm=false` before JAX starts, because XLA's Triton fusion of float64 products
returned wrong values on the GPU (see RESULTS.md).

## Layout

| Path | Contents |
|---|---|
| `src/gauge_theory_gravity/algebra.py` | the algebra, null basis, generators, direction and slice types, sector projectors |
| `fields.py`, `gauge.py`, `covariant.py`, `curvature.py`, `transport.py` | sections 4–8 of the proposal |
| `cartan.py` | geometries as coframes with closed-form frames (Painlevé–Gullstrand, FRW, lapse, boosted, Kerr–Schild), Christoffel symbols, Lorentz connection |
| `conformal.py`, `invariants.py`, `sta_embedding.py` | the full conformal connection, normal conditions, Schouten, Weyl squared, the Lorentz sector |
| `export.py` | VTKHDF export of spacetime slices for ParaView |
| `examples/` | scenarios, renders, disk light |
| `_array/`, `tests/array_reference/` | the first, index-based implementation, kept as an independent oracle |
| `tests/` | the checks listed in RESULTS.md |

## Licence

MIT, see [LICENSE](LICENSE). `src/gauge_theory_gravity/examples/light.py` adapts code from numga's Kerr example
(MIT, Copyright (c) 2022 Eelco Hoogendoorn); its header keeps that notice.

---

## The proposal: Conformal Spacetime Algebra Gauge Theory of Gravity

### 1. Overview

This document proposes a formulation of Gauge Theory Gravity (GTG) within the real Conformal Spacetime Algebra (CSTA),

$$
\mathrm{Cl}(2,4;\mathbb R).
$$

The objective is to extend the geometric framework of conventional Spacetime Algebra (STA),

$$
\mathrm{Cl}(1,3;\mathbb R),
$$

to a six-dimensional conformal algebra while retaining the fundamental gauge-theoretic structure of GTG.

The central proposal is to represent the gravitational gauge connection as a bivector-valued field,

$$
\boxed{
\Omega(a,x)\in\mathrm{Cl}^{2}(2,4;\mathbb R)
}
$$

whose bivector components generate infinitesimal conformal transformations.

Finite gauge transformations, meanwhile, are represented by CSTA versors,

$$
\boxed{
V(x)\in\mathrm{Spin}(2,4).
}
$$

This establishes a distinction between the infinitesimal gauge connection and the finite transformations it generates.

The proposed framework seeks to:

1. Preserve the gauge-theoretic interpretation of conventional GTG.
2. Represent Lorentz transformations, translations, dilations, and special conformal transformations within a unified Clifford algebra.
3. Express gauge transformations and parallel transport using real CSTA versors.
4. Investigate whether conventional GTG can be recovered as a restricted sector of the conformal formulation.

The proposal is a mathematical framework for investigation, not yet a complete gravitational theory. In particular, the gravitational action, field equations, and constraints required to recover Einstein gravity remain to be established.

---

### 2. Algebraic Foundation

#### 2.1 Conformal Spacetime Algebra

The underlying algebra is

$$
\mathcal G=\mathrm{Cl}(2,4;\mathbb R).
$$

It is generated by six orthogonal basis vectors,

$$
\lbrace \mathbf e_t,\mathbf e_x,\mathbf e_y,\mathbf e_z,
\mathbf e_+,\mathbf e_-\rbrace .
$$

The basis vectors satisfy

$$
\mathbf e_A\mathbf e_B+
\mathbf e_B\mathbf e_A=2\eta_{AB},
$$

where the metric is

$$
\eta_{AB} =
\mathrm{diag}(+1,-1,-1,-1,+1,-1).
$$

Consequently,

$$
\begin{aligned}
\mathbf e_t^2 &= +1,\\
\mathbf e_x^2=\mathbf e_y^2=\mathbf e_z^2 &= -1,\\
\mathbf e_+^2 &= +1,\\
\mathbf e_-^2 &= -1.
\end{aligned}
$$

The algebra has dimension

$$
\dim\mathcal G=2^6=64.
$$

Its bivector subspace has dimension

$$
\dim\mathcal G^2 =
\binom{6}{2} = 15.
$$

These 15 bivectors generate the Lie algebra

$$
\mathfrak{spin}(2,4)
\cong
\mathfrak{so}(2,4).
$$

The corresponding spin group provides a double-cover representation of the identity component of the four-dimensional conformal group.

#### 2.2 Index Conventions

Two distinct index sets are required.

Spacetime indices:

$$
a,b,\mu,\nu\in\lbrace t,x,y,z\rbrace .
$$

Conformal-space indices:

$$
A,B\in\lbrace t,x,y,z,+,-\rbrace .
$$

The distinction is important.

The spacetime indices identify directions in the four-dimensional spacetime on which the gauge connection is evaluated.

The conformal-space indices identify the internal bivector generators of the six-dimensional gauge algebra.

For example,

$$
\omega^{AB}(a,x)
$$

contains two conformal-space indices and one spacetime direction argument.

---

### 3. Null Basis and Conformal Generators

Although the six-dimensional basis above is orthogonal, the conventional identification of conformal transformations is most naturally expressed using a null basis.

Define

$$
\mathbf n_o=
\frac{1}{\sqrt{2}}(\mathbf e_++\mathbf e_-)
$$

and

$$
\mathbf n_\infty=
\frac{1}{\sqrt{2}}(\mathbf e_+-\mathbf e_-).
$$

These satisfy

$$
\mathbf n_o^2=\mathbf n_\infty^2=0
$$

and

$$
\mathbf n_o\cdot\mathbf n_\infty=1.
$$

The resulting basis consists of four spacetime basis vectors and two mutually null conformal basis vectors.

The six-dimensional bivector space can now be decomposed into four sectors.

#### 3.1 Lorentz Generators

The Lorentz generators are

$$
M_{ab}=\mathbf e_a\wedge\mathbf e_b.
$$

There are six independent generators.

They generate local rotations and boosts in four-dimensional spacetime.

Their gauge field is the familiar Lorentz connection,

$$
\omega^{ab}(a,x).
$$

#### 3.2 Translation Generators

Define

$$
P_a=\mathbf e_a\wedge\mathbf n_\infty.
$$

These four generators represent conformal-space translations.

Their corresponding gauge field is denoted

$$
e^a(a,x).
$$

The normalization of this field is conventional and can absorb factors associated with the definition of the translation generators.

#### 3.3 Special Conformal Generators

Define

$$
K_a=\mathbf e_a\wedge\mathbf n_o.
$$

These four generators represent special conformal transformations.

Their corresponding gauge field is

$$
f^a(a,x).
$$

#### 3.4 Dilation Generator

The remaining generator is

$$
D=\mathbf n_o\wedge\mathbf n_\infty.
$$

Its gauge field is denoted

$$
b(a,x).
$$

It generates local scale transformations.

#### 3.5 Generator Decomposition

The conformal bivector space therefore decomposes as

$$
\mathfrak{so}(2,4) =
\mathfrak{so}(1,3)
\oplus
\mathfrak p
\oplus
\mathfrak k
\oplus
\mathfrak d.
$$

Its dimensions are

$$
15=6+4+4+1.
$$

This provides the geometric decomposition necessary for interpreting the CSTA gauge connection.

---

### 4. Bivector-Valued Gauge Connection

The fundamental gauge field is

$$
\boxed{
\Omega(a,x) =
\frac12\omega^{AB}(a,x)
\mathbf e_A\wedge\mathbf e_B.
}
$$

Here,

$$
\Omega(a,x)\in\mathcal G^2.
$$

The field is a bivector-valued function of spacetime direction $a$ and position $x$.

Using the null-basis decomposition, it can equivalently be expressed as

$$
\boxed{
\begin{aligned}
\Omega(a,x)
={}&
\frac12\omega^{ab}(a,x)
\mathbf e_a\wedge\mathbf e_b\\
&+e^a(a,x)\mathbf e_a\wedge\mathbf n_\infty\\
&+f^a(a,x)\mathbf e_a\wedge\mathbf n_o\\
&+b(a,x)\mathbf n_o\wedge\mathbf n_\infty.
\end{aligned}
}
$$

The coefficient conventions can be adjusted to accommodate alternative normalizations of the conformal generators.

The connection therefore contains four types of gauge fields:

| Gauge sector | Field | Independent components |
|---|---|---:|
| Lorentz | $\omega^{ab}$ | 6 |
| Translation | $e^a$ | 4 |
| Special conformal | $f^a$ | 4 |
| Dilation | $b$ | 1 |
| **Total** | | **15** |

All four sectors are represented within a single CSTA bivector-valued field.

Importantly, $\Omega(a,x)$ is the infinitesimal gauge connection, not a finite gauge transformation.

---

### 5. Versor-Valued Gauge Transformations

A central feature of the proposed formulation is that finite gauge transformations are represented by CSTA versors.

Let

$$
V(x)\in\mathrm{Spin}(2,4).
$$

For a normalized spin versor,

$$
\boxed{
V\widetilde V=1.
}
$$

Consequently,

$$
V^{-1}=\widetilde V.
$$

A multivector field transforms according to

$$
\Psi'(x)=V(x)\Psi(x).
$$

For geometric objects transforming by conjugation, the corresponding transformation is

$$
X'(x)=V(x)X(x)\widetilde V(x).
$$

The distinction between these two transformation laws depends on the representation of the field.

The versor represents a finite conformal transformation, whereas the bivector connection generates infinitesimal transformations.

#### 5.1 Gauge Transformation of the Connection

The connection transforms according to

$$
\boxed{
\Omega'(a,x) =
V\Omega(a,x)\widetilde V -
2(a\cdot\nabla V)\widetilde V.
}
$$

This is the CSTA generalization of the rotational gauge transformation in conventional GTG.

The expression preserves the bivector-valued nature of the connection.

The first term describes the transformation of the existing connection under the adjoint action of the versor.

The second term accounts for the position dependence of the gauge transformation.

---

### 6. Covariant Derivative

The covariant derivative incorporates the CSTA gauge connection.

Define

$$
\boxed{
\mathcal D_a\Psi =
a\cdot\nabla\Psi +
\frac12\Omega(a,x)\Psi.
}
$$

Here:

- $a\cdot\nabla$ is the directional derivative.
- $\Omega(a,x)$ is the bivector-valued conformal connection.
- $\Psi$ is a field transforming in the appropriate spin representation.

The factor of $1/2$ follows the conventional spin representation of bivector generators.

The derivative is constructed so that

$$
\mathcal D'_a\Psi' =
V\mathcal D_a\Psi.
$$

This is the required gauge covariance condition.

---

### 7. Curvature

The curvature associated with the connection is defined by

$$
\boxed{
\begin{aligned}
\mathcal F(a,b)
={}&
a\cdot\nabla\Omega(b) -
b\cdot\nabla\Omega(a)\\
&+
\frac12[\Omega(a),\Omega(b)].
\end{aligned}
}
$$

Here,

$$
[X,Y]=XY-YX
$$

denotes the commutator product.

This expression assumes constant directional vectors $a,b$ and a flat background directional derivative.

For general vector fields, the corresponding term

$$
-\Omega([a,b])
$$

must also be included.

The curvature is itself bivector-valued:

$$
\mathcal F(a,b)\in\mathcal G^2.
$$

It represents the failure of parallel transport to be path-independent.

#### 7.1 Relation to the Covariant Derivative

The curvature is related to the commutator of covariant derivatives through

$$
\boxed{
[\mathcal D_a,\mathcal D_b]\Psi =
\frac12\mathcal F(a,b)\Psi
}
$$

for constant commuting directional vectors.

#### 7.2 Curvature Decomposition

Because the connection contains four conformal sectors, its curvature can also be decomposed:

$$
\mathcal F =
\mathcal F_{\mathrm{Lorentz}} +
\mathcal F_{\mathrm{translation}} +
\mathcal F_{\mathrm{special}} +
\mathcal F_{\mathrm{dilation}}.
$$

This allows the individual geometric contributions to be investigated independently while retaining their unified algebraic representation.

---

### 8. Versor-Valued Parallel Transport

The bivector-valued connection generates finite gauge transformations through parallel transport.

Along a path $\gamma$, define

$$
\boxed{
V(x_1,x_2) =
\mathcal P
\exp\left(
-\frac12
\int_{\gamma}\Omega
\right).
}
$$

Here, $\mathcal P$ denotes path ordering.

The resulting object is a versor, assuming the connection takes values in the appropriate spin Lie algebra.

It satisfies

$$
V\widetilde V=1.
$$

This gives a natural relationship between the infinitesimal and finite descriptions:

$$
\boxed{
\text{Bivector connection}
\longrightarrow
\text{Versor-valued parallel transport}.
}
$$

An important distinction must be maintained.

A general connection cannot necessarily be reconstructed from a single globally defined versor field.

For example, defining

$$
\Omega_\mu=-2(\partial_\mu V)\widetilde V
$$

produces a pure-gauge connection.

A general connection with nonzero curvature must be allowed to exist independently of a globally defined versor potential, or be described using local bundle sections and transition functions.

This distinction is important if versors are to become fundamental dynamical variables.

---

### 9. Relationship to Conventional Gauge Theory Gravity

Conventional GTG is formulated within

$$
\mathrm{Cl}(1,3;\mathbb R).
$$

Its gravitational gauge structure includes:

1. A displacement gauge field.
2. A rotational gauge field.

The rotational gauge field is bivector-valued:

$$
\Omega_{\mathrm{STA}}(a,x)\in\mathrm{Cl}^{2}(1,3).
$$

Its local gauge transformations are represented by STA rotors.

The proposed CSTA formulation extends the internal gauge algebra:

$$
\mathrm{Cl}^{2}(1,3)
\longrightarrow
\mathrm{Cl}^{2}(2,4).
$$

This increases the number of available bivector generators from six to fifteen.

The additional generators permit the representation of translations, dilations, and special conformal transformations.

#### 9.1 Recovering the Lorentz Sector

A natural restriction is

$$
\Omega(a,x) =
\frac12\omega^{ab}(a,x)
\mathbf e_a\wedge\mathbf e_b.
$$

The remaining conformal sectors are set to zero:

$$
e^a=0,\qquad f^a=0,\qquad b=0.
$$

This recovers a Lorentz-valued connection embedded in CSTA.

However, this restriction alone does not recover the complete gravitational structure of GTG.

#### 9.2 Recovering the Translational and Displacement Structure

The translational sector is particularly important.

In conventional gravity, the displacement gauge field establishes the relationship between the physical spacetime geometry and a fiducial background.

In a conformal Cartan formulation, the translational connection is closely related to the soldering form or coframe.

It is therefore reasonable to investigate whether the GTG displacement gauge field can be represented through the translational sector of the CSTA connection.

However, this identification is not automatic.

A precise equivalence requires specifying:

- The relationship between the translational connection and the GTG displacement map.
- The conformal gauge choice.
- The constraints imposed on the dilation and special conformal sectors.
- The resulting gravitational field equations.

Consequently, recovering GTG should be treated as a separate mathematical objective rather than assumed from the algebraic embedding.

---

### 10. Gravitational Dynamics

The connection and curvature establish the kinematic structure of the theory.

A complete gravitational theory additionally requires an action and field equations.

The conformal curvature is

$$
\mathcal F=d\Omega+\frac12\Omega\wedge_{\mathrm{Lie}}\Omega,
$$

where the second term denotes the Lie-algebra-valued connection product.

A gravitational action could potentially be constructed from appropriate scalar contractions of the curvature.

However, a straightforward quadratic curvature action for the normal conformal Cartan connection generally produces Weyl-squared gravity rather than Einstein gravity.

Recovering Einstein gravity requires additional structure, potentially including:

- A compensator field.
- A symmetry-breaking mechanism.
- Constraints on the conformal connection.
- A suitable identification of the translational sector with the physical coframe.

The specific choice remains an open part of the proposed formulation.

---

### 11. Research Questions

Several questions should be investigated before treating this construction as a complete gravitational theory.

#### 11.1 Fundamental Versor Fields

Can the gravitational degrees of freedom be represented fundamentally using normalized CSTA versor fields?

In particular, can one formulate a theory in which the finite conformal transformations are primary dynamical objects, with the bivector-valued connection emerging from their derivatives and additional independent connection data?

The distinction between general connections and pure-gauge connections must be preserved.

#### 11.2 Displacement Gauge Field

Can the displacement gauge field of conventional GTG be identified with the translational component of the conformal Cartan connection?

If so, can the conventional displacement and rotational gauge fields be incorporated into a single CSTA-valued geometric structure?

#### 11.3 Additional Conformal Degrees of Freedom

What physical interpretation should be assigned to the dilation and special conformal gauge fields?

Should these represent additional gravitational degrees of freedom, or should they be constrained or eliminated through an appropriate gauge choice?

#### 11.4 Recovery of Einstein Gravity

What action and constraints produce Einstein gravity as a restricted sector of the CSTA formulation?

This is necessary to distinguish an alternative representation of conventional gravity from a genuinely different gravitational theory.

#### 11.5 Real Clifford Algebra Formulation

Can the entire construction be developed using only

$$
\mathrm{Cl}(2,4;\mathbb R)
$$

without complexification?

The proposed algebraic framework suggests that this is possible.

The spin group, bivector generators, connection, curvature, and versor transformations can all be represented using the real Clifford algebra.

---

### 12. Proposed Mathematical Structure

The proposed theory can be summarized by the following objects.

| Object | Mathematical representation | Purpose |
|---|---|---|
| Algebra | $\mathrm{Cl}(2,4;\mathbb R)$ | Underlying conformal geometry |
| Gauge group | $\mathrm{Spin}(2,4)$ | Finite conformal transformations |
| Gauge algebra | $\mathfrak{so}(2,4)$ | Infinitesimal conformal transformations |
| Gauge connection | $\Omega(a,x)\in\mathrm{Cl}^{2}(2,4)$ | Infinitesimal gauge field |
| Gauge transformation | $V(x)\in\mathrm{Spin}(2,4)$ | Finite gauge transformation |
| Covariant derivative | $\mathcal D_a$ | Gauge-covariant differentiation |
| Curvature | $\mathcal F(a,b)\in\mathrm{Cl}^{2}(2,4)$ | Gauge-field strength |
| Parallel transport | $V(x_1,x_2)$ | Finite transport between points |

The central relationships are

$$
\boxed{
\Omega(a,x) =
\frac12\omega^{AB}(a,x)
\mathbf e_A\wedge\mathbf e_B
}
$$

$$
\boxed{
V\widetilde V=1
}
$$

$$
\boxed{
\mathcal D_a\Psi =
a\cdot\nabla\Psi +
\frac12\Omega(a,x)\Psi
}
$$

$$
\boxed{
\mathcal F(a,b) =
a\cdot\nabla\Omega(b) -
b\cdot\nabla\Omega(a) +
\frac12[\Omega(a),\Omega(b)].
}
$$

Together, these define the fundamental kinematic structure of a conformal gauge theory formulated in real CSTA.

---

### 13. Conclusion

The proposed formulation extends the algebraic setting of conventional Gauge Theory Gravity from STA to CSTA.

Its principal feature is the representation of the complete conformal gauge connection as a single bivector-valued field:

$$
\boxed{
\Omega(a,x)\in\mathrm{Cl}^{2}(2,4;\mathbb R).
}
$$

The connection contains the generators of Lorentz transformations, translations, dilations, and special conformal transformations.

Finite gauge transformations and parallel transport are represented by normalized CSTA versors:

$$
\boxed{
V(x)\widetilde V(x)=1.
}
$$

This establishes a unified geometric framework in which the gauge connection and its finite transformations are expressed through the geometric product of a real Clifford algebra.

The most significant research objective is to determine whether the displacement and rotational gauge structures of conventional GTG can be recovered as a constrained sector of this conformal formulation, while retaining the possibility of additional conformal gravitational degrees of freedom.
