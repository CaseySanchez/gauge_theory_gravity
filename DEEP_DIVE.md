# Deep dive: the design, and the mathematics it implements

This document explains how the code in `src/gtg_csta/` realises the conformal gauge connection of
`gauge_theory_gravity_conformal_spacetime_algebra.md` (the "spec") on `numga` extensors. Each section gives
the mathematics, then the code that expresses it, then how it is checked. Results and caveats are in
`RESULTS.md`; the original plan is `PLAN.md`.

Notation: spacetime indices $\mu,\nu$ (coordinates) and $a,b$ (frame), both over $\{t,x,y,z\}$; conformal indices
$A,B$ over six directions. $[X,Y]=XY-YX$ is the full commutator. Everything is in $\mathrm{Cl}(2,4;\mathbb R)$ with
`Algebra("t+x-y-z-p+n-")`, i.e. $\eta=\mathrm{diag}(+,-,-,-,+,-)$, exactly the spec's convention.

---

## 1. The algebra and its generators (`algebra.py`)

### 1.1 Null basis

With $e_+=\mathtt p$ ($e_+^2=+1$) and $e_-=\mathtt n$ ($e_-^2=-1$),

$$
n_o=\tfrac1{\sqrt2}(e_++e_-),\qquad n_\infty=\tfrac1{\sqrt2}(e_+-e_-),\qquad
n_o^2=n_\infty^2=0,\quad n_o\!\cdot\! n_\infty=1 .
$$

### 1.2 Fifteen bivectors in four sectors

$$
M_{ab}=e_a\wedge e_b\ (6),\quad P_a=e_a\wedge n_\infty\ (4),\quad K_a=e_a\wedge n_o\ (4),\quad D=n_o\wedge n_\infty\ (1).
$$

Checked (`tests/test_algebra.py`): the 15 generators are linearly independent, and their commutators close
as $\mathfrak{so}(2,4)$:

$$
[M,M]\subset M,\ [M,P]\subset P,\ [M,K]\subset K,\ [M,D]=0,\ [P,P]=[K,K]=0,\
[D,P]\propto P,\ [D,K]\propto K\ (\text{opposite sign}),\ [P,K]\subset M\oplus D .
$$

### 1.3 Sector projectors as extensor expressions

A bivector decomposes uniquely as

$$
B=B_M+u\wedge n_\infty+v\wedge n_o+b\,n_o\wedge n_\infty,\qquad u,v\ \text{spacetime vectors},\ B_M\ \text{Lorentz}.
$$

Contracting with the null vectors on the left (using $n_o\!\cdot\!n_o=0$, $n_o\!\cdot\!n_\infty=1$,
$n_{o,\infty}\!\cdot u=0$):

$$
n_o\lrcorner B=-u-b\,n_o,\qquad n_\infty\lrcorner B=-v+b\,n_\infty,\qquad
b=-\,n_\infty\lrcorner(n_o\lrcorner B).
$$

Hence

$$
\begin{aligned}
\Pi_D(B)&=-\,n_o\!\wedge\! n_\infty\ \bigl[n_\infty\lrcorner(n_o\lrcorner B)\bigr],\\
\Pi_P(B)&=-(n_o\lrcorner B)\wedge n_\infty-\Pi_D(B),\\
\Pi_K(B)&=-(n_\infty\lrcorner B)\wedge n_o-\Pi_D(B),\\
\Pi_M(B)&=B-\Pi_P-\Pi_K-\Pi_D .
\end{aligned}
$$

In code these are literally four extensors `Bivector <- Bivector` (`D_PART`, `P_PART`, `K_PART`, `M_PART`),
written with `|` (inner product), `^` and the bare type `Bivector` (the identity map). They apply to a bivector
or to any bivector-valued map, e.g. `al.sector(F, "K")` on a curvature map. A test checks them against the
coordinate split through the $15\times15$ generator matrix. That matrix `GEN` is kept only to *specify*
test connections and for the structure-constant test.

---

## 2. Fields as typed maps (`fields.py`)

The key design decision: the gauge connection is not a table of components but a linear map. In the spec,
$\Omega(a,x)$ is bivector-valued and linear in the direction $a$. In numga that is the type

$$
\Omega(x):\ \texttt{Bivector}\leftarrow\texttt{Spacetime},
$$

where `Spacetime` is the 4-dimensional subspace $\{t,x,y,z\}$ of the 6 vectors. Binding a direction gives a
bivector: `Omega(a)`. A versor field is an `Even` extensor, a spinor likewise. A *field* is a callable on a
position, itself an extensor of type `Spacetime`.

The directional derivative $a\cdot\nabla$ is numga's `derivative`: for $f(x)$ of type $S\leftarrow(I\dots)$ it
produces $S\leftarrow(I\dots,\ \text{step})$, so for the connection

$$
\mathtt{derivative(Omega)(x)}\,(b,a)\;=\;a\!\cdot\!\nabla\,\Omega(b).
$$

(Argument order: the connection's own input first, the step last.) Everything below is built from this.
A finite-difference `fd_derivative` with the same result type (4th-order stencil) exists to confirm the
autodiff and to test convergence order.

---

## 3. Gauge transformations and the covariant derivative (spec §5, §6; `gauge.py`, `covariant.py`)

**Transformation.** For a versor field $V(x)\in\mathrm{Spin}(2,4)$ (so $V\tilde V=1$),

$$
\Omega'(a)=V\,\Omega(a)\,\tilde V-2\,(a\!\cdot\!\nabla V)\,\tilde V .
$$

In code: `V >> Omega` is the sandwich $V\Omega\tilde V$ of the map's *output* (the input $a$ is a position-space
direction and is not transformed), and the inhomogeneous term is `-2 * (dV(x) * ~V(x))`, with `dV = derivative(V)`
of type `Even <- Spacetime`. It is grade-projected with `.restrict[2]`.

That the result is a pure bivector is a claim of the spec; it holds because
$(\partial V)\tilde V$ is a bivector for $V\tilde V=1$ (differentiate: $(\partial V)\tilde V+V\,\partial\tilde V=0$,
so $(\partial V)\tilde V$ is reverse-antisymmetric, hence of grade 2 or 6 in $\mathrm{Cl}(2,4)$; that the
grade-6 (pseudoscalar) part vanishes is not derived here). The test measures the non-bivector residual
directly for $V=\exp(B(x))$, and finds it zero (1e-11).

**Covariant derivative.** $\mathcal D_a\Psi=a\!\cdot\!\nabla\Psi+\tfrac12\Omega(a)\Psi$, code:
`derivative(Psi)(x) + 0.5 * (Omega(x) * Psi(x))`. Gauge covariance, $\mathcal D'\Psi'=V\mathcal D\Psi$ with
$\Psi'=V\Psi$:

$$
(\partial V)\Psi+V\partial\Psi+\tfrac12\bigl(V\Omega\tilde V-2(\partial V)\tilde V\bigr)V\Psi
=V\partial\Psi+\tfrac12V\Omega\Psi,
$$

using $\tilde VV=1$. Verified numerically for random $V,\Omega,\Psi$ (1e-11 exact derivative, 1e-7 finite
difference).

**Pure gauge.** $\Omega=0\mapsto\Omega'=-2(\partial V)\tilde V$. Then $\mathcal D(V\Phi)=V\,\partial\Phi$, so
$V\Phi_0$ is covariantly constant: such connections are flat.

---

## 4. Curvature (spec §7; `curvature.py`)

For coordinate vectors (so $-\Omega([a,b])=0$):

$$
F(a,b)=a\!\cdot\!\nabla\Omega(b)-b\!\cdot\!\nabla\Omega(a)+\tfrac12[\Omega(a),\Omega(b)] .
$$

Why the $\tfrac12$ is right: $[\mathcal D_a,\mathcal D_b]\Psi=\tfrac12\bigl(\partial_a\Omega_b-\partial_b\Omega_a\bigr)\Psi
+\tfrac14[\Omega_a,\Omega_b]\Psi=\tfrac12F(a,b)\Psi$. Both this identity and the $\tfrac12$ are tested.

**Code.** With `d = derivative(Omega)(x)` of inputs $(b,a)$:

```python
F = permute_inputs(d, (1, 0)) - d + Om.commutator(Om)
```

`permute_inputs(d,(1,0))(a,b)=d(b,a)=a\!\cdot\!\nabla\Omega(b)`; `d(a,b)=b\!\cdot\!\nabla\Omega(a)`; and numga's
`commutator` is $(XY-YX)/2$, which is exactly the spec's $\tfrac12[\Omega(a),\Omega(b)]$. The product
`Om.commutator(Om)` of two single-input maps is a two-input map with inputs in order of appearance, $(a,b)$.
`permute_inputs` is a small local helper because numga has no axis permutation for multilinear maps.

**Properties checked:** $F$ is bivector valued; antisymmetric in $(a,b)$; covariant,
$F'=V F\tilde V$ (`V >> F`); zero for pure gauge, non-zero otherwise; and the Bianchi identity

$$
\oint_{(\lambda,a,b)}\Bigl(\partial_\lambda F(a,b)+\tfrac12[\Omega(\lambda),F(a,b)]\Bigr)=0 ,
$$

implemented as the cyclic sum of a three-input map (the adjoint covariant derivative has the $\tfrac12$
because bivectors act on bivectors through $\tfrac12[\Omega,\cdot]$).

---

## 5. Parallel transport and holonomy (spec §8; `transport.py`)

Parallel transport solves $\gamma'\!\cdot\mathcal D\Psi=0$, i.e. $\dot\Psi=-\tfrac12\Omega(\gamma')\Psi$, so
$\Psi(s)=T(s)\Psi(0)$ with $\dot T=-\tfrac12\Omega(\gamma')\,T$. Later steps multiply on the **left**:

$$
T=\mathcal P\exp\!\Bigl(-\tfrac12\!\int_\gamma\Omega\Bigr)=\prod_{k=N}^{1}\exp\bigl(-\tfrac12\Omega(\gamma'_k)\,\Delta s\bigr).
$$

`omega(x)(v)` is the binding of the velocity $v=\gamma'(s)$ into the connection map. Two schemes:

- midpoint ($O(h^2)$): one exponential per step;
- Magnus-4 ($O(h^4)$), with $A_i=-\tfrac12\Omega(\gamma'(s+c_i h))$, $c_{1,2}=\tfrac12\mp\tfrac{\sqrt3}{6}$:
  $$\Theta=\tfrac h2(A_1+A_2)-\tfrac{\sqrt3}{12}h^2[A_1,A_2],$$ code uses numga's half-commutator, hence $\tfrac{\sqrt3}{6}h^2\,A_1.\mathrm{commutator}(A_2)$.

A versor stays a versor under this scheme because each factor is the exponential of a bivector
($T\tilde T=1$ to 1e-10). Orders of convergence are tested ($\approx4\times$ and $\approx16\times$ error drop
per halving).

**Holonomy.** For the loop $x_0\to+\mu\to+\nu\to-\mu\to-\nu$ of side $\epsilon$:

$$
T_{\text{loop}}=\exp\!\bigl(-\tfrac12\,\epsilon^2\,F(e_\mu,e_\nu)\bigr)+O(\epsilon^3).
$$

The sign was not assumed: the test fits $\pm$ against numerics, finds $-$ in three different planes, and also
checks that the error scales as $\epsilon^3$. This is the concrete content of "curvature is the failure of
parallel transport to be path independent", and the reason pure-gauge connections give $T_{\text{loop}}=1$ while a
generic $\Omega$ does not: a global versor field cannot reproduce a general connection (spec §8).

---

## 6. Geometry from a tetrad (`cartan.py`)

Spacetime geometry enters through a coframe $e^a{}_\nu$, an extensor $\texttt{Spacetime}_{\text{frame}}\leftarrow\texttt{Spacetime}_{\text{coord}}$,
$e(\partial_\nu)=e^a{}_\nu e_a$. The algebra's own metric supplies the rest:

$$
g=e\,|\,e\quad(\text{a form},\ g_{\mu\nu}=\eta_{ab}e^a{}_\mu e^b{}_\nu),\qquad E=e^{-1}\quad(\texttt{e.inverse()}).
$$

### 6.1 Christoffel symbols

From `dg = derivative(g)` (a three-input form, $\partial_\gamma g_{\alpha\beta}$ with inputs $(\alpha,\beta,\gamma)$):

$$
\Gamma_{\lambda\mu\nu}=\tfrac12\bigl(\partial_\mu g_{\lambda\nu}+\partial_\nu g_{\lambda\mu}-\partial_\lambda g_{\mu\nu}\bigr)
=\tfrac12\bigl(\mathrm{perm}_{(0,2,1)}\,dg+dg-\mathrm{perm}_{(2,0,1)}\,dg\bigr).
$$

### 6.2 The Lorentz connection as a bivector map

The Levi-Civita connection written in the frame is
$\omega_\mu{}^a{}_b=e^a{}_\lambda\bigl(\partial_\mu E_b{}^\lambda+\Gamma^\lambda_{\mu\sigma}E_b{}^\sigma\bigr)$,
defined by $\nabla_\mu E_b=\omega_\mu{}^a{}_bE_a$. Lowering the first index with $\eta$, the trilinear form

$$
W(c,\mu,b)=\Gamma(E_c,\partial_\mu,E_b)+e_c\!\cdot\!\bigl(e(\partial_\mu E_b)\bigr)
$$

is $\omega_{\mu\,cb}$ (the first term binds the frame vectors into $\Gamma_{\lambda\mu\nu}$; the second is
$\eta_{cd}e^d{}_\lambda\partial_\mu E_b{}^\lambda$). The bivector connection of the CSTA is

$$
\Omega(\partial_\mu)=\tfrac12\,\omega_\mu{}^{ab}\,e_a\wedge e_b=\tfrac12\,\omega_{\mu\,ab}\,e^a\wedge e^b ,
$$

obtained with two metric contractions:

```python
B = (W * (S ^ S)).contract(1, 4).contract(2, 3)     # sums W(e_k,.,e_l) e^k ^ e^l
Omega = 0.5 * B
```

**The normalisation is not a choice.** For a bivector $B$, $\tfrac12[B,v]=B\lrcorner v$, so
$\tfrac12[\Omega(\partial_\mu),v]=\omega_\mu{}^a{}_c\,v^c e_a$, i.e. the algebra's adjoint action reproduces the
frame-component covariant derivative of a vector, and consequently the curvature $F$ of §4, evaluated on this
connection, *is* the Riemann bivector map $R^{ab}{}_{\mu\nu}$. Both statements are tested against an independent
index implementation that builds $R^a{}_{b\mu\nu}$ from Christoffel symbols directly.

### 6.3 Spacetimes used

| Geometry | Coframe | Nature |
|---|---|---|
| Schwarzschild, Painlevé–Gullstrand | $e^0=dt,\ e^i=dx^i+\beta^i dt$, $\beta=\sqrt{2M/r}\,\hat x$ | vacuum; regular through $r=2M$ |
| spatially flat FRW | $e^0=dt,\ e^i=a(t)dx^i$ | conformally flat, with matter |
| static lapse | $e^0=N(\mathbf x)dt,\ e^i=dx^i$ | generic: Weyl, Ricci and Cotton all non-zero |
| boosted PG | $e'^a{}_\nu(x)=e^a{}_\mu(Lx)L^\mu{}_\nu$ | vacuum, moving hole |
| PG with $M(t)$ | as PG, $M\to M(t)$ | toy with matter |

Painlevé–Gullstrand coordinates are used because the spatial slices are flat Euclidean, the tetrad is simple, and
nothing blows up at the horizon, which makes the interior grids and ray tracing well defined down to $r\to0$.

---

## 7. The conformal connection and the Cartan reading (spec §9, §10; `conformal.py`)

With the translation sector identified with the coframe, the full 15-component connection is

$$
\boxed{\ \Omega=\Omega_{\text{Lorentz}}+e\wedge n_\infty+f\wedge n_o+b\,(n_o\wedge n_\infty)\ }
$$

where $e$ is the coframe, $f:\texttt{Spacetime}\leftarrow\texttt{Spacetime}$ the special-conformal part and $b$ a form
(dilation). Each term is one wedge product of an extensor with a null vector.

### 7.1 Sectors of the curvature

$F$ splits with the projectors of §1.3:

| Sector | Meaning |
|---|---|
| $\Pi_P F$ | torsion: $de+\omega\wedge e$ (vanishes for Levi-Civita) |
| $\Pi_M F$ | Riemann, minus $e\wedge f$ terms: becomes Weyl after normalisation |
| $\Pi_K F$ | Cotton: $df+\omega\wedge f$-type |
| $\Pi_D F$ | $e\cdot f$ antisymmetrised, zero when $f$ is symmetric in this sense |

### 7.2 Normal conditions

The *normal* conformal Cartan connection fixes $f$ by requiring torsion-free ($\Pi_PF=0$), $\Pi_DF=0$ and
$\Pi_MF$ trace-free:

$$
\mathrm{Ric}(\nu)=\sum_a e^a\lrcorner\,\Pi_MF(E_a,\nu)=0
\qquad\bigl(\texttt{(S | FM(E, S)).contract(1, 2)}\bigr).
$$

The conditions are affine in the 16 unknowns $f^a{}_\mu$ and involve no derivatives of $f$ (so they are pointwise).
The code builds the residual as an extensor expression, differentiates it by `jacfwd` and solves the 16-unknown
least squares. Result, in every test spacetime and to 1e-16:

$$
\boxed{\ f^a{}_\mu=P^a{}_\mu=\tfrac12\Bigl(\mathrm{Ric}^a{}_\mu-\tfrac R6\,e^a{}_\mu\Bigr)\ }
$$

the Schouten tensor (rank 16, so the solution is unique). In this normalisation the proportionality constant is
exactly 1. The Schouten tensor is computed separately from the Lorentz curvature alone with the same contraction,
so the equality $f=P$ is a genuine check of the construction.

With $f=P$: $\Pi_PF=0$, $\Pi_DF=0$ always; $\Pi_MF$ is the Weyl tensor; $\Pi_KF$ is the Cotton tensor, which is
zero for FRW (conformally flat) and in vacuum, and non-zero for the lapse metric.

### 7.3 The quadratic action density

Take the scalar part of $F\tilde F$ with both index pairs contracted through the metric:

$$
I=g^{\mu\rho}g^{\nu\sigma}\,\langle F_{\mu\nu}\tilde F_{\rho\sigma}\rangle_0,
$$

in code the two `.contract` calls on $F(E,E)\,\tilde{F(E,E)}$. Results (three spacetimes, 1e-9):

$$
I[\text{normal connection}]=\tfrac12\,C_{abcd}C^{abcd},\qquad
I[\text{Lorentz only}]=\tfrac12\,R_{abcd}R^{abcd}\ \text{(Ricci-flat)},
$$

with $C^2=R_{abcd}R^{abcd}-2R_{ab}R^{ab}+\tfrac13R^2$ in four dimensions. For Schwarzschild $R_{abcd}R^{abcd}=48M^2/r^6$,
so $I=24M^2/r^6$ (verified to 1e-11).

This is the spec's §10 remark made concrete: a quadratic curvature action of the normal conformal connection is
Weyl-squared, not Einstein, gravity. **Only the Lagrangian density was checked**: no field equations (Bach), no
compensator, no symmetry breaking. Einstein recovery (spec §11.4) remains open.

---

## 8. Generator roles through the conformal action of versors (tests)

Points embed as the null vector $X=n_o+x-\tfrac12x^2\,n_\infty$ (null because $n_o\!\cdot n_\infty=+1$). A versor acts by
$X\mapsto VX\tilde V$. Measured, with $V=\exp(cG)$:

| Generator | Effect on $x$ | Why |
|---|---|---|
| $P_a$ | $x\to x+2c\,e_a$, $n_\infty$ fixed | translation; the sandwich doubles the parameter |
| $D$ | $x\to e^{-2\lambda}x$ | dilation, same factor of two |
| $K_a$ | fixes $n_o$; preserves the null cone | special conformal |
| $M_{ab}$ | $x^2$ preserved, no rescaling | Lorentz |

This confirms the spec's labelling: $P_a=e_a\wedge n_\infty$ really generates translations (it fixes the point at
infinity $n_\infty$), $K_a=e_a\wedge n_o$ fixes the origin.

---

## 9. The flyby field data and the `shift` array (`export.py`)

For the boosted geometry $e'^a{}_\nu(x)=e^a{}_\mu(Lx)L^\mu{}_\nu$ with $L$ the boost in the $t$–$x$ plane
($t'=\gamma(t-vx)$, $x'=\gamma(x-vt)$), the translation-sector component $e'^i{}_0$ stored as `shift` is

$$
\texttt{shift}^i=\gamma\bigl(\beta^i(y)-v\,\delta^i_x\bigr),\qquad y=Lx,\quad \beta(y)=\sqrt{2M/r_y}\,\hat y .
$$

Consequences for the picture:

- far away $\to-\gamma v\,\hat x=-0.75\,\hat x$ (a uniform frame artefact, pointing against the hole's motion),
- plus an outward part $\gamma\sqrt{2M/r_y}$ (outward: it is the coframe component, and the physical infall
  velocity of the rain frame is $-\beta$),
- a stagnation point ahead of the hole where the two cancel: $\sqrt{2M/r_y}=v\Rightarrow r_y=2M/v^2\approx5.6M$
  (rest frame), i.e. $\approx4.4M$ ahead in the lab,
- the pattern translates with the hole and is contracted by $1/\gamma$ along $x$.

The `shift` array is a gauge-dependent component of the connection; the gauge-invariant arrays are
`weyl2_half`, `riem2_half`, `ricci_scalar`. Exported fields are evaluated pointwise on a regular grid (`jit(vmap)` in
chunks) and written as VTKHDF ImageData, one file per time step plus a `.series` index. The writer's layout was
checked by reading with VTK's own reader.

---

## 10. Why this architecture

**Extensors for the CSTA objects.** The spec's objects are linear maps with a typed output and typed input:
$\Omega:\texttt{Spacetime}\to\texttt{Bivector}$, $F:(\texttt{Spacetime},\texttt{Spacetime})\to\texttt{Bivector}$,
$e:\texttt{Spacetime}\to\texttt{Spacetime}$, the sector projectors $\texttt{Bivector}\to\texttt{Bivector}$. Binding is
$\Omega(a)$; composition is a function call; the adjoint action is `>>`; contractions are `.contract`.
This removed index bookkeeping from the curvature, Bianchi, spin connection, Ricci, Schouten and the invariants.

**Idiom.** Following `numga/examples`: module-level named types, small named functions, no `lambda` and no
`jax.numpy` in the package. Geometries are *identity plus dyads*, as in gauge theory gravity:
$e_{PG}=\mathrm{Id}+\beta\,(t\,|\,\cdot)$ is `Spacetime + shift * (time | Spacetime)`, FRW and the lapse metric are
`TimePart + a * SpacePart` and `N * TimePart + SpacePart`, and a boost is a rotor sandwich,
$e'(x)=e(R\rhd x)\circ(R\rhd\cdot)$ with $R=\exp(\tfrac12\,\mathrm{artanh}(v)\,t\wedge x)$. Test fields are random
polynomials $c_0+c_1(x)+c_2(x,x)$ of multilinear maps. Positions, steps and loops are batch axes of extensors, and the ordered
product of step versors is `cumprod` (numga multiplies later elements on the left, which is the transport order).

**Where arrays remain (and why).**
- the $15\times15$ generator matrix `GEN`: converts between generator coordinates and numga's blade order, used to
  specify test connections and in the structure-constant test;
- the 16-unknown linear solve for the normal conditions (numpy `lstsq` on columns assembled from extensor residuals);
- the carry flags of the ray tracer's `lax.scan`;
- `permute_inputs`, a small helper for the missing axis permutation of multilinear maps.

**The ray tracer is the connection.** A ray's momentum is a frame vector $k$ and its position an event $x$:
$$\dot x=E(k),\qquad \dot k=-\tfrac12[\Omega(\dot x),k]\quad(\texttt{-Omega(xdot).commutator(k)}),$$
the geodesic equation as parallel transport of $k$ by the Lorentz connection of §6.2 along its own path.

**Two independent implementations.** The first version was written with raw `einsum` and is retained in
`_array/`. It is an oracle: curvature, transport, spin connection, Christoffel symbols, Schouten and Weyl² of
the extensor version are compared against it (agreement 1e-12 or better), and the Riemann tensor is built there
directly from Christoffel symbols, i.e. independently of the CSTA construction.

**Fields are callables on points, not lattices.** Differentiation is by autodiff (exact) and the identities are
pointwise; this avoids discretisation error in every identity test. Grids appear only at the output stage
(slices, rendering, export).

---

## 11. Verification map

| Claim | Where | Level |
|---|---|---|
| $\eta$, null basis, 15 generators, $\mathfrak{so}(2,4)$ closure, Jacobi | `test_algebra.py` | exact |
| $V\tilde V=1$, `exp` vs an independent matrix exponential | `test_gauge_curvature.py` | 1e-9 |
| $\Omega'$ stays a bivector; $\mathcal D'\Psi'=V\mathcal D\Psi$ | same | 1e-11 |
| $[\mathcal D,\mathcal D]\Psi=\tfrac12F\Psi$; Bianchi; flatness of pure gauge | same | 1e-10 |
| transport is a versor; orders 2 and 4; reparametrisation invariance | `test_transport.py` | 1e-6 … 1e-10 |
| holonomy $=\exp(-\tfrac12\epsilon^2F)$ | same | $O(\epsilon^3)$ |
| Lorentz sector closes under Lorentz rotors, leaks under general $\mathrm{Spin}(2,4)$ | `test_lorentz_schwarzschild.py` | 1e-11 / O(0.1) |
| spin connection, Christoffel, Riemann vs index code; Ricci $=0$; $48M^2/r^6$ | same | 1e-12 … 1e-8 |
| torsion $=0$; normal $f=$ Schouten; Cotton sector; $I=\tfrac12C^2$ | `test_conformal_cartan.py`, `test_invariants_and_conformal_action.py` | 1e-8 … 1e-16 |
| generator roles ($2c$ translation, $e^{-2\lambda}$ dilation) | same | 1e-12 |
| VTKHDF layout; fields of a small flyby grid (vacuum identities) | `test_export.py` (`-m slow`) | read back by `vtkHDFReader` |

---

## 12. Open questions and limits

- **No dynamics.** There are no field equations; "simulation" here means evaluating the kinematic fields on given
  geometries. The spec itself says the dynamics are open.
- **Einstein recovery (spec §11.4)** needs more than the Weyl-squared density found here: a compensator, a symmetry
  breaking mechanism, or constraints. Not attempted.
- **GTG displacement field (spec §9.2, §11.2).** The translation sector is identified with an orthonormal coframe of a
  torsion-free metric. That is standard Cartan geometry; whether it is a faithful reading of Doran–Lasenby's
  $h(a)$ (with its rotor/gauge structure and fiducial flat background) is untested.
- **Dilation field $b$** is zero throughout. Its role (spec §11.3) is unexplored.
- **Fundamental versors (spec §11.1).** Only pure-gauge flatness and transport as versors were examined.
- **Numerical issues found:** on the GPU, XLA's Triton fusion of float64 dots miscompiles `jit`ed code in which two
  extensor chains share subexpressions (set `--xla_gpu_enable_triton_gemm=false`, done in `__init__.py`); earlier,
  `jit(vmap)` of the holonomy loop gave wrong values on CPU for batches above about 16 (cause not identified; loops are now one
  extensor batch). Backward ray tracing in the PG chart must stop at $r\lesssim2.2M$, because a backward ray hugging the
  horizon continues onto an unphysical branch.
- **Rendering** (lensed image, GIF) is Schwarzschild only, with no redshift or beaming; the camera shadow radius agrees
  with the analytic value to 0.65%.
