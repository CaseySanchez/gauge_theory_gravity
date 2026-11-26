# Deep dive: how the code realizes the proposal

This document walks through `src/gauge_theory_gravity/` from the algebra up: what each object is mathematically,
how it is written on [numga](https://github.com/EelcoHoogendoorn/numga) extensors, and how it is checked. Section
numbers such as §6 refer to the proposal in [README.md](README.md). Every numerical claim, with its test and
tolerance, is recorded in [RESULTS.md](RESULTS.md).

**Conventions.** The algebra is $\mathrm{Cl}(2,4;\mathbb R)$, `Algebra("t+x-y-z-p+n-")`, with metric
$\eta=\mathrm{diag}(+,-,-,-,+,-)$, where `p` is $e_+$ and `n` is $e_-$. Spacetime indices $\mu,\nu$ are coordinate
indices and $a,b$ frame indices, both over $t,x,y,z$. $[X,Y]=XY-YX$ is the full commutator; numga's `commutator` is
half of it. $\widetilde X$ is the reverse.

---

## 1. The algebra (`algebra.py`)

### 1.1 Null basis and generators

```math
n_o=\tfrac1{\sqrt2}(e_++e_-),\qquad n_\infty=\tfrac1{\sqrt2}(e_+-e_-),\qquad
n_o^2=n_\infty^2=0,\qquad n_o\cdot n_\infty=1 .
```

The fifteen bivectors split into the four sectors of §3:

```math
M_{ab}=e_a\wedge e_b\ (6),\qquad P_a=e_a\wedge n_\infty\ (4),\qquad K_a=e_a\wedge n_o\ (4),\qquad D=n_o\wedge n_\infty\ (1).
```

`GENS` lists them in this order. `GEN`, the $15\times15$ matrix from generator coordinates to numga's blade order, is
used only to specify test connections and to compute structure constants; nothing in the construction goes through
it. `tests/test_algebra.py` checks that the generators are independent, that they close as
$\mathfrak{so}(2,4)$ with the expected pattern ($[M,P]\subset P$, $[P,P]=[K,K]=0$, $[D,P]\propto P$ and
$[D,K]\propto K$ with opposite signs, $[P,K]\subset M\oplus D$), and the Jacobi identity.

### 1.2 Sector projectors

A bivector decomposes uniquely as

```math
B=B_M+u\wedge n_\infty+v\wedge n_o+b\,n_o\wedge n_\infty ,
```

with $u,v$ spacetime vectors and $B_M$ a Lorentz bivector. Contracting with the null vectors,

```math
n_o\cdot B=-u-b\,n_o,\qquad n_\infty\cdot B=-v+b\,n_\infty,\qquad b=-\,n_\infty\cdot(n_o\cdot B),
```

so the projectors are

```math
\begin{aligned}
\Pi_D(B)&=-\,(n_o\wedge n_\infty)\,\bigl[n_\infty\cdot(n_o\cdot B)\bigr],\\
\Pi_P(B)&=-(n_o\cdot B)\wedge n_\infty-\Pi_D(B),\\
\Pi_K(B)&=-(n_\infty\cdot B)\wedge n_o-\Pi_D(B),\\
\Pi_M(B)&=B-\Pi_P(B)-\Pi_K(B)-\Pi_D(B).
\end{aligned}
```

In code these are four extensors of type `Bivector <- Bivector` (`D_PART`, `P_PART`, `K_PART`, `M_PART`), written with
`|`, `^` and the bare type `Bivector`, which acts as the identity map. Because they are maps, they apply to a bivector
or to any bivector-valued map: `al.sector(F, "K")` is the special-conformal part of a whole curvature map.

### 1.3 Translators and conformal points

The translation generators move the origin. For an event $x$ the translator

```math
T_x=\exp\bigl(\tfrac12\,x\wedge n_\infty\bigr)=1+\tfrac12\,x\,n_\infty
```

is exact at first order because $(x\wedge n_\infty)^2=0$, and the event's conformal point is the origin carried there:

```math
X=T_x\,n_o\,\widetilde T_x .
```

`al.translator(event)` and `al.conformal_point(event) = translator(event) >> n_o` implement this; `>>` is numga's
sandwich. $X$ is null, has weight $X\cdot n_\infty=1$, and reads back the event's coordinates.

### 1.4 The pseudoscalar

$I=e_te_xe_ye_ze_+e_-$ squares to $-1$, commutes with every even element and anticommutes with every vector. It is
the imaginary unit of the twistor construction (§7).

---

## 2. Types and fields (`algebra.py`, `fields.py`)

### 2.1 Directions are intrinsic

The proposal's objects are linear maps with typed inputs and outputs, so the code represents them as extensors
rather than component tables. Following numga's conformal idiom, events and spacetime directions are **intrinsic
antivectors**: `Direction` is the dual of the slice algebra `t+x-y-z-`, a grade-5 type in six dimensions. They are
read by **extrinsic slices**, vectors in $t,x,y,z$, through the regressive product `slice & direction`.

- `DIRECTIONS = [t*, x*, y*, z*]` are the coordinate directions $\partial_\mu$.
- `SLICES = [t, -x, -y, -z]` read them: `SLICES[mu] & DIRECTIONS[nu]` is $\delta^\mu_\nu$.
- The metric on directions is still `|`, with $\partial_\mu|\partial_\nu=\eta_{\mu\nu}$.
- `ORIGIN` and `INFINITY` are $n_o$ and $n_\infty$ as intrinsic points.

numga checks orientation statically, so adding a slice to a direction, or feeding a slice where the connection
expects a direction, fails before any arithmetic runs. Generators, versors and spinors are extrinsic.

| Object | Type |
|---|---|
| connection $\Omega(x)$ | `Bivector <- Direction` |
| curvature $F(x)$ | `Bivector <- (Direction, Direction)` |
| coframe $e(x)$ | `Direction <- Direction` (frame from coordinate) |
| versor $V(x)$, spinor $\Psi(x)$, twistor $Z$ | `Even` |

### 2.2 Fields and derivatives

A field is a Python callable from an event (a `Direction` extensor, possibly batched) to an extensor. Its derivative
is numga's `derivative`, exact by autodiff: for a field of type $S\leftarrow(I_1,\dots)$ it returns a field of type
$S\leftarrow(I_1,\dots,\text{step})$. For the connection,

```math
\texttt{derivative(Omega)(x)}(b,a)=a\cdot\nabla\,\Omega(b),
```

with the map's own input first and the step last. `finite_difference` produces the same type with a fourth-order
central stencil; it exists to test the autodiff and the convergence orders. `permute_inputs` reorders the inputs of a
multilinear map, which numga has no operation for.

Test fields are random polynomials $c_0+c_1(x)+c_2(x,x)$ whose coefficients are multilinear maps (`fields.polynomial`),
so they are smooth and genuinely position-dependent; a random versor field is the exponential of such a bivector
field.

---

## 3. Gauge structure, §5–8 (`gauge.py`, `covariant.py`, `curvature.py`, `transport.py`)

### 3.1 Gauge transformations

For a versor field $V(x)$ with $V\widetilde V=1$,

```math
\Omega'(a)=V\,\Omega(a)\,\widetilde V-2\,(a\cdot\nabla V)\,\widetilde V .
```

In `transform_connection`, `rotor >> omega(x)` sandwiches the map's output (its input $a$ is a direction of position
and is not transformed), and the second term is `-2 * (gradient(x) * ~rotor).restrict[2]`. That the full expression is
a bivector is a claim of §5: $(\partial V)\widetilde V$ is reverse-antisymmetric, so it can only have grades 2 and 6,
and `non_bivector_part` measures the remainder directly; it is zero to rounding for $V=\exp B(x)$.

### 3.2 Covariant derivative

```math
\mathcal D_a\Psi=a\cdot\nabla\Psi+\tfrac12\,\Omega(a)\Psi ,
```

written `gradient(x) + 0.5 * (omega(x) * spinor(x))`, a map `Even <- Direction`. Gauge covariance,
$\mathcal D'_a(V\Psi)=V\mathcal D_a\Psi$, follows from $\widetilde VV=1$ and is checked for random fields. From
$\Omega=0$ the transformation gives the pure-gauge connection $-2(\nabla V)\widetilde V$ (`pure_gauge`), under which
$V\Psi_0$ is covariantly constant, so pure-gauge connections are flat.

### 3.3 Curvature

For coordinate directions the $\Omega([a,b])$ term vanishes and

```math
F(a,b)=a\cdot\nabla\Omega(b)-b\cdot\nabla\Omega(a)+\tfrac12[\Omega(a),\Omega(b)] .
```

With `d = derivative(Omega)(x)`, whose inputs are $(b,a)$:

```python
F = permute_inputs(d, (1, 0)) - d + connection.commutator(connection)
```

numga's `commutator` is $(XY-YX)/2$, exactly the $\tfrac12[\,,]$ of §7, and the product of two one-input maps is a
two-input map with inputs in order of appearance. The factor $\tfrac12$ is fixed by

```math
[\mathcal D_a,\mathcal D_b]\Psi=\tfrac12\,F(a,b)\Psi ,
```

which `commutator_of_derivatives` computes from the covariant derivative itself. The tests also check that $F$ is
antisymmetric, covariant ($F'=V\,F\,\widetilde V$), zero for pure gauge, and satisfies the Bianchi identity

```math
\sum_{\text{cyclic }(l,a,b)}\Bigl(\partial_lF(a,b)+\tfrac12\bigl[\Omega(l),F(a,b)\bigr]\Bigr)=0 .
```

### 3.4 Parallel transport and holonomy

Transport along $\gamma$ solves $\dot\Psi=-\tfrac12\Omega(\dot\gamma)\Psi$, so $\Psi(s)=T(s)\Psi(0)$ with later steps on
the left:

```math
T=\mathcal P\exp\Bigl(-\tfrac12\int_\gamma\Omega\Bigr)=\prod_{k=N}^{1}\exp\bigl(-\tfrac12\,\Omega(\dot\gamma_k)\,\Delta s\bigr).
```

All steps of a path are one batch: positions, velocities, the connection bound to them, and their exponentials, whose
ordered product is numga's `cumprod` (it multiplies later elements on the left). Two schemes are implemented:

- midpoint, second order;
- fourth-order Magnus with two Gauss points, $\Theta=\tfrac h2(A_1+A_2)-\tfrac{\sqrt3}{12}h^2[A_1,A_2]$ with
  $A_i=-\tfrac12\Omega(\dot\gamma)$ at $s+(\tfrac12\mp\tfrac{\sqrt3}6)h$; with numga's half-commutator the second term is
  $\tfrac{\sqrt3}{6}h^2\,A_1\text{.commutator}(A_2)$.

Each factor is the exponential of a bivector, so $T$ stays a versor. Around the rectangle
$x_0\to+\mu\to+\nu\to-\mu\to-\nu$ of side $\epsilon$ (`holonomy_loop`),

```math
T_{\text{loop}}=\exp\bigl(-\tfrac12\,\epsilon^2F(\partial_\mu,\partial_\nu)\bigr)+O(\epsilon^3).
```

The sign was measured, not assumed: the test fits both signs in three planes, and checks that the error scales as
$\epsilon^3$.

---

## 4. Geometry from a coframe (`cartan.py`)

### 4.1 Coframe, metric, frame

Spacetime geometry enters as a coframe $e(\partial_\nu)=e^a{}_\nu\,\partial_a$, an extensor
`Direction <- Direction` from coordinate to frame directions. The metric is the algebra's own metric pulled back,
$g=e\,|\,e$, and the frame is $E=e^{-1}$.

Each `Geometry` carries its frame in closed form. Gauge theory gravity writes its fields as the identity plus dyads,
and where the dyad squares to zero the inverse is immediate:

```math
e=\mathrm{Id}+N,\quad N^2=0\quad\Longrightarrow\quad E=\mathrm{Id}-N .
```

- **Painlevé–Gullstrand:** `Direction + shift * (t & Direction)`, with shift $\beta=\sqrt{2M/r}\,\hat x$. The time
  slice does not read the spatial shift, so the dyad squares to zero.
- **Kerr–Schild:** `Direction - profile * null * (null.dual_inverse() & Direction)`, with $l$ null in the flat
  metric (`null_dyad`).
- **FRW and the static lapse:** `TimePart + a * SpacePart` and `N * TimePart + SpacePart`, inverted term by term.
- **Boost:** a rotor sandwich, $e'(x)=e(R\rhd x)\circ(R\rhd\cdot)$ with $R=\exp\bigl(\tfrac12\mathrm{artanh}(v)\,t\wedge x\bigr)$,
  and frame $E'(x)=R^{-1}\rhd E(R\rhd x)$ (`rotor << ...`).

A bare coframe field falls back to `e.inverse()`. `tests/test_types_and_frames.py` checks every closed-form frame
against the inverse.

### 4.2 Christoffel symbols and the Lorentz connection

From the derivative of the metric, a three-input form with inputs $(\alpha,\beta,\gamma)$ for
$\partial_\gamma g_{\alpha\beta}$:

```math
\Gamma_{\lambda\mu\nu}=\tfrac12\bigl(\partial_\mu g_{\lambda\nu}+\partial_\nu g_{\lambda\mu}-\partial_\lambda g_{\mu\nu}\bigr),
```

assembled by `christoffel_lower` with two `permute_inputs`. The spin connection in the frame, with its first index
lowered, is the trilinear form

```math
W(c,\mu,b)=\Gamma(E_c,\partial_\mu,E_b)+e_c\cdot e(\partial_\mu E_b),
```

and the bivector connection is

```math
\Omega(\partial_\mu)=\tfrac12\,W(\partial_k,\mu,\partial_l)\,s^k\wedge s^l ,
```

summed over the slices $s^k$ that read the frame directions:

```python
bivectors = (W * (Slice ^ Slice)).trace(1, 4).trace(2, 3)
Omega = 0.5 * bivectors
```

The normalization is forced, not chosen: for a bivector, $\tfrac12[B,v]=B\cdot v$, so
$\tfrac12[\Omega(\partial_\mu),v]$ is the frame covariant derivative of $v$, and the curvature of §3.3 evaluated on this
connection is the Riemann bivector map. Both are checked against the index implementation in `_array/`, which builds
$R^a{}_{b\mu\nu}$ from Christoffel symbols directly.

### 4.3 Geometries

| Geometry | Coframe | Character |
|---|---|---|
| Schwarzschild (`schwarzschild`) | Painlevé–Gullstrand, $e^0=dt$, $e^i=dx^i+\beta^idt$ | vacuum; regular through $r=2M$ |
| growing mass (`growing_mass`) | as above with $M(t)$ rising by a tanh | torsion-free toy with matter |
| FRW (`friedmann`) | $e^0=dt$, $e^i=a(t)dx^i$, $a=t^{2/3}$ | conformally flat, with matter |
| static lapse (`static_lapse`) | $e^0=N(\mathbf x)dt$, $e^i=dx^i$ | Weyl, Ricci and Cotton all nonzero |
| boosted (`boosted`) | $e'(x)=e(Lx)L$ | a moving hole |
| Kerr (`kerr`) | $e=\mathrm{Id}-f\,l\,(l\cdot)$, $f=Mr^3/(r^4+a^2z^2)$, $l=\partial_t-n$ | vacuum, rotating; GTG's position gauge field |

Both black-hole charts are horizon-penetrating, so grids and ray tracing can run inside the horizon.

---

## 5. The conformal connection, §9–10 (`conformal.py`, `invariants.py`)

### 5.1 The full connection

With the translation sector identified with the coframe,

```math
\Omega=\Omega_{\text{Lorentz}}+e\wedge n_\infty+f\wedge n_o+b\,(n_o\wedge n_\infty),
```

where $f$ is a `Direction <- Direction` map (special conformal part) and $b$ a form (dilation). Because directions are
intrinsic, the wedges are joins with intrinsic points: the translation generator along a direction $d$ is
$(d\,\&\,n_\infty^{*})^{*}=e_d\wedge n_\infty$, written `(coframe(x) & al.INFINITY).dual()`, and the special-conformal
one uses `al.ORIGIN` the same way.

### 5.2 Curvature sectors and the normal conditions

The projectors of §1.2 split $F$:

| Sector | Content |
|---|---|
| $\Pi_PF$ | torsion; zero for the Levi-Civita connection |
| $\Pi_MF$ | Riemann minus terms in $e\wedge f$ |
| $\Pi_KF$ | the Cotton-type derivative of $f$ |
| $\Pi_DF$ | the antisymmetric part of $e\cdot f$ |

The normal conformal connection fixes $f$ by three conditions: $\Pi_PF=0$, $\Pi_DF=0$, and $\Pi_MF$ trace-free, where
the trace is the Ricci map

```math
\mathrm{Ric}(\nu)=\sum_a e^a\cdot\Pi_MF(E_a,\nu),
```

written `(-lorentz(frame, Direction).commutator(Direction)).contract(1, 3)`. The conditions are affine in the sixteen
components of $f$ and involve no derivative of $f$, so they hold pointwise. `solve_normal_f` builds the residual for
sixteen unit perturbations of $f$ (exact, by affinity) and solves the least-squares system. In every test spacetime
the system has rank 16 and the solution is

```math
f=P=\tfrac12\Bigl(\mathrm{Ric}-\tfrac R6\,e\Bigr),
```

the Schouten tensor, with coefficient exactly 1. `schouten` computes $P$ independently, from the Lorentz curvature
alone, so the equality is a real check. With $f=P$, $\Pi_PF=\Pi_DF=0$, $\Pi_MF$ is the Weyl tensor, and $\Pi_KF$ is the
Cotton tensor: zero for FRW and in vacuum, nonzero for the lapse metric.

### 5.3 The quadratic density

```math
I=g^{\mu\rho}g^{\nu\sigma}\,\bigl\langle F_{\mu\nu}\widetilde F_{\rho\sigma}\bigr\rangle_0 ,
```

evaluated in the frame as `(framed * ~framed).restrict[0].contract(1, 3).contract(1, 2)` with `framed = F(E, E)`.
For the normal connection $I=\tfrac12C_{abcd}C^{abcd}$; for the Lorentz connection alone, in vacuum,
$I=\tfrac12R_{abcd}R^{abcd}$, which is $24M^2/r^6$ for Schwarzschild. `weyl_squared` computes $C^2$ independently as
$R_{abcd}R^{abcd}-2R_{ab}R^{ab}+\tfrac13R^2$. This is the remark of §10 made concrete: a curvature-squared action of the
normal connection is Weyl-squared gravity. Only the density is computed; there are no field equations.

### 5.4 Generator roles

Acting on conformal points with $V=\exp(cG)$ confirms the labels of §3
(`tests/test_invariants_and_conformal_action.py`):

| Generator | Effect |
|---|---|
| $P_a$ | $x\to x+2c\,e_a$, $n_\infty$ fixed: translation (the sandwich doubles the parameter) |
| $D$ | $x\to e^{-2\lambda}x$: dilation |
| $K_a$ | fixes $n_o$, preserves the null cone: special conformal |
| $M_{ab}$ | preserves $x^2$: Lorentz |

### 5.5 The Lorentz sector and Kerr

`sta_embedding.py` embeds a Lorentz connection and measures leakage: Lorentz rotors keep the sector closed, while a
general $\mathrm{Spin}(2,4)$ transformation moves it into the other sectors, so restricting to the Lorentz sector is a
gauge choice (§9.1). For Kerr (`tests/test_kerr.py`), the coframe and the Lorentz connection reproduce GTG's position
and rotation gauge fields from numga's independent Kerr code; the connection is torsion-free and Ricci-flat, the
normal $f$ vanishes, and $2I$ equals the Kerr Kretschmann scalar.

---

## 6. Twistors (`twistor.py`)

### 6.1 Twistor space

Since $I$ is central in the even subalgebra and $I^2=-1$, the even subalgebra is $M(4,\mathbb C)$ and
$\mathrm{Spin}(2,4)\cong SU(2,2)$. The idempotent

```math
P=\tfrac14\bigl(1+I\,e_xe_y\bigr)\bigl(1+I\,e_te_+\bigr),\qquad P^2=P=\widetilde P,
```

has rank one, and twistor space is the left ideal $\mathbb T=\mathcal G^+P$, four complex dimensions. `Twistor` is the
type `Even`, since elements of the ideal are general even multivectors. Left multiplication by a versor, a connection
or a transport acts on twistors unchanged, so `covariant_derivative`, `transport` and `holonomy_loop` apply to twistors
as they are.

- **Norm:** `inner` computes $h(Z_1,Z_2)=4\langle\widetilde Z_1Z_2\rangle_{0,6}$, a scalar plus an $I$ part, read as a
  complex number by `as_complex`. Reversion is its adjoint, so versors preserve it. Its signature is $(2,2)$.
- **Coordinates:** the basis `BASIS` $=[\omega^0,\omega^1,\pi_0,\pi_1]$ is built from the seeds $\sqrt2P$ and
  $\sqrt2\,e_xe_zP$; in it $h=2\,\mathrm{Re}(\bar\omega\cdot\pi)$. `coordinates` and `from_coordinates` convert, and
  `matrix` gives the $4\times4$ complex matrix of an even element.

### 6.2 The ω/π split and incidence

```math
\omega=\tfrac12\,n_\infty n_oZ\quad(n_\infty\omega=0),\qquad \pi=\tfrac12\,n_on_\infty Z\quad(n_o\pi=0).
```

The two projectors (`OMEGA_PART`, `PI_PART`) are idempotents summing to one. Lorentz bivectors commute with both, so
they act on each half separately; translations $e_a\wedge n_\infty$ map $\pi$ into $\omega$, special conformal
transformations map $\omega$ into $\pi$, and the dilation scales the halves oppositely.

An event and a twistor are incident when $XZ=0$ with $X=T_xn_o\widetilde T_x$ (`incidence`). Since
$XZ=T_x\,n_o\,(\widetilde T_xZ)$, this holds exactly when the twistor translated back to the origin has no $\omega$
half, that is $Z=T_x\pi_0$, or $\omega=\tfrac12\,x\,n_\infty\pi$. `through(event, momentum)` builds such twistors as
`translator(event) * pi(momentum)`. In coordinates $\omega=-\tfrac{i}{\sqrt2}\,x^{AA'}\pi$ with $x^{AA'}$ Hermitian of
determinant $x^2$, Penrose's $\omega=ix\pi$ up to normalization.

For a null twistor, $h(Z,Z)=0$, the real vectors annihilating $Z$ span a totally null 2-plane, the twistor's light
ray; `annihilator` finds it by an SVD of the map $V\mapsto VZ$, and `ray` reads off an event and the null direction. A
non-null twistor is annihilated by no real vector.

### 6.3 Momentum and flag

The momentum is the moment map of the translations, $p^a=-\mathrm{Im}\,h(Z,(e^a\wedge n_\infty)Z)$. Because the
scalar part is cyclic and $I$ is central in the even subalgebra, and $Z\widetilde Z$ (its own reverse) has only grades
0 and 4, this reduces to one product:

```math
p^a=4\,\bigl\langle (e^a\wedge n_\infty)\,\langle Z\widetilde Z\rangle_4\,I\bigr\rangle_0 .
```

It is null and future-pointing, Penrose's $p=\pi\bar\pi$, independent of $\omega$ and of the phase of $Z$.
`from_momentum` inverts it for a ray through the origin.

The phase is seen by the **null flag**: the real part of the self-dual bivector $\epsilon_{AB}\pi^A\pi^B$, which
`flag` evaluates against each Lorentz generator. It is a null bivector $k\wedge s$, with $s$ a unit spacelike
direction across the ray, fixed up to multiples of $k$. Multiplying $Z$ by $e^{I\theta}$ (`with_phase`) turns $s$ by
$2\theta$ about $k$, and a Lorentz rotor carries the flag direction of $Z$ to that of $RZ$ as it carries vectors.
`flag_direction(Z, u)` picks the representative orthogonal to a timelike direction $u$. The flag is quadratic, not
Hermitian, in $\pi$; since $I$ is central in the even subalgebra, no even element conjugates a twistor, so the flag is
computed in coordinates rather than as a product.

### 6.4 Local twistor transport

With the normal conformal connection of §5.2, $\mathcal D_aZ=a\cdot\nabla Z+\tfrac12\Omega(a)Z$ is Penrose's local twistor
transport. Its curvature on $\mathbb T$ vanishes for FRW and not for the lapse metric, and holonomy loops lie in
$SU(2,2)$ and match $\exp(-\tfrac12\epsilon^2F)$. Along a null geodesic, the translation part $k\wedge n_\infty$ moves
$\pi$ into $\omega$ at the rate the event moves, so a twistor through the current event stays through it; this is
what the ray tracer uses (§7).

### 6.5 The complex Weyl invariant

Lorentz bivectors act on the $\omega$ half by their self-dual part, so the trace of the curvature squared over that
half is complex:

```math
\mathcal W=2\Bigl\langle\tfrac12\,n_\infty n_o\,F(E_a,E_b)\,\widetilde F(E^a,E^b)\Bigr\rangle_{0,6} =
\tfrac12\bigl(C_{abcd}C^{abcd}+i\,C_{abcd}{}^*C^{abcd}\bigr).
```

`weyl_invariant` computes it. Its real part is $I$ of §5.3; its imaginary part is half the Chern–Pontryagin density.
For Kerr it equals $24M^2/(r-ia\cos\theta)^6$, and for Schwarzschild it is real.

---

## 7. The ray tracer (`examples/render.py`, `examples/light.py`)

### 7.1 State and equations

A ray is its event $x$ and the local twistor $Z$ of its light ray, both in the frame:

```math
\dot x=E(k),\qquad \dot Z=-\tfrac12\,\Omega_{\text{conf}}(\dot x)\,Z,\qquad k=\texttt{momentum}(Z),
```

with $\Omega_{\text{conf}}$ the Lorentz connection plus the translation $k\wedge n_\infty$ (the Schouten part is zero in
vacuum). `rates` returns the two derivatives; `runge_kutta` steps them with fourth-order Runge–Kutta. The Lorentz part
turns $\pi$ as parallel transport would, so $k$ obeys the geodesic equation in the frame and stays exactly null; the
translation part keeps $\omega=0$ relative to the moving event.

The step size multiplies from the left, `(size / 2) * rate`. With the scalar on the right, numga contracts it with the
product table first, which for an even multivector builds an $n\times32\times32$ intermediate that XLA's GPU compiler
took 27 minutes over at 16384 rays; from the left the whole step compiles in seconds.

### 7.2 Camera and screen

`camera_view` places a static observer at distance $D$ and inclination $i$ above the disk plane, and gives each pixel
the photon momentum $k=u-n$ (so $k\cdot u=1$) and the camera's right and up made orthogonal to the ray.
`static_camera` builds each twistor with `from_momentum` and turns its phase so that its flag lies along the camera's
right.

The polarization needs the camera screen at the disk. Because Lorentz transport turns flag and screen alike,
`screen_frame` reads it from the twistor: screen $x$ along the flag, screen $y$ along the flag of $e^{I\pi/4}Z$ (a
quarter turn on), each orthogonal to the frame's time direction. This equals the parallel-transported screen up to
multiples of $k$. The emission axis is orthogonal to $k$, so those multiples do not change the polarization.
`readout(state)` returns the event, momentum and screen.

### 7.3 Tracing

Rays are traced backward from the camera with steps $-0.03\,\mathrm{clip}(r,1,10)$ in the affine parameter, in chunks of
16384 through one compiled `lax.scan`. A ray stops when it escapes ($r>120$, counted as escaped beyond $r=100$) or falls
below $1.1\,r_+$; in these horizon-penetrating charts a backward ray hugging the horizon would otherwise continue onto
an unphysical branch. Rays continue through the disk so the shadow is measured on its own. The first crossing of
$z=0$ within the disk is located by cubic Hermite interpolation of the step, using the rates carried in the scan, refined
by two Newton steps on the height.

### 7.4 Disk light and polarization

From numga's Kerr example (`light.py`, MIT): prograde Kepler gas from the innermost stable orbit, the Page–Thorne
temperature, and blackbody color through the Wyman–Sloan–Shirley fit of the CIE matching functions, tone-mapped over
a star field. The frequency ratio observed/emitted is $1/(k\cdot u_{\text{gas}})$ with $k\cdot u_{\text{camera}}=1$, which carries
gravitational redshift and Doppler shift together. Emission is polarized across a toroidal field in the local frame
(gas aberration is left out), stored as a unit-intensity coherency `Screen <- ScreenSlice` on the screen and read by
a rotating analyzer.

### 7.5 Twistor holonomy figure

`twistor_holonomy` transports twistors around $0.4\times0.4$ loops in the meridian plane of Kerr ($a=0.9$) with the normal
connection and plots three panels: the largest $|\lambda-1|$ of each loop's $SU(2,2)$ matrix, how boost-like the loop's
curvature is (the angle of the dominant eigenvalue's logarithm from the imaginary axis, 0° rotation to 90° boost), and
for comparison the phase of $\mathcal W$, $6\arctan(a\cos\theta/r)$. In type D the Weyl curvature acts on self-dual
bivectors as multiplication by $\Psi_2\propto(r-ia\cos\theta)^{-3}$, so the boost fraction follows
$|\arg\Psi_2|=3\arctan(a\cos\theta/r)$ folded into 0–90°: zero on the equator, largest near the poles. For spin zero the
boost fraction is zero outside $r=2.5$, the region checked.

---

### 7.6 Ray paths, extrinsic and intrinsic transport (`examples/rays.py`)

`trace_paths` follows rays as `trace_scene` does but records their state every few steps, so that each ray becomes a
polyline. It offers two transports of the one equation $\dot Z=AZ$, $A=-\tfrac12\Omega_{\text{conf}}(\dot x)$.
The *intrinsic* transport steps the carried twistor itself, $(x,Z)$ by additive Runge–Kutta, as the renderer does;
each stage's generator is built from the twistor it moves, so $\omega=0$, and with it $h(Z,Z)$, hold to round-off.
The *extrinsic* transport, the default, keeps the camera's twistor $Z_0$ fixed and accumulates the carrier, a versor,

```math
Z(\lambda)=U(\lambda)\,Z_0,\qquad \dot U=A\,U,\qquad U\in\mathrm{Spin}(2,4),
```

stepped on the group by the commutator-free method CF4 (Celledoni, Marthinsen and Owren):

```math
U_2=e^{\frac h2A_1}U,\quad U_3=e^{\frac h2A_2}U,\quad U_4=e^{\frac h2(2A_3-A_1)}U_2,\quad
U'=e^{\frac h{12}(-A_1+2A_2+2A_3+3A_4)}\,e^{\frac h{12}(3A_1+2A_2+2A_3-A_4)}\,U,
```

with $A_i$ the generator at stage $i$ (`versor_rates`) and the event moved by the classical Runge–Kutta stages. Each
update is a product of exponentials of bivectors, so $U\widetilde U=1$ and $h(Z,Z)$ hold to round-off. The method is
fourth order and agrees with a fine reference. It keeps the group exactly, while $\omega$ drifts at truncation level,
because a mixed generator such as $2A_3-A_1$ is not built from the twistor it acts on.

`observables` reads, per record: coordinates and affine parameter; the frame momentum, coordinate velocity, frame
frequency and redshift relative to the camera; the screen carried by the flag; the drifts of the Killing energy and
angular momentum; the twistor's coordinates, amplitudes, $h(Z,Z)$, $|\omega|/|\pi|$ and relative phase;
and, for the extrinsic transport, $U$'s coefficients, $|U\widetilde U-1|$, the transported basis $Ue_a\widetilde U$ of $\mathbb R^{2,4}$,
$\mathrm{tr}\,U/4$ on twistor space and $\mathrm{tr}\,(\widetilde U_0U)/4$ against the first ray. `write_paths` writes them as VTKHDF PolyData, drawn in any three of $(t,x,y,z)$.

---

## 8. Export (`hypersurface.py`, `export.py`)

### 8.1 Fields over spacetime, hypersurfaces for display

A field is a function of the coordinates $(t,x,y,z)$, sampled, when sampling is needed, on a `Grid4`. A dataset is a
hypersurface

```math
\Sigma_\mu(c):\ x^\mu=c,
```

for any of the four coordinates, with the other three as its axes: $t=c$ is a spatial snapshot, $x$, $y$ or $z=c$ a
section through time. Which coordinate is fixed decides the hypersurface; the order of the remaining three (which is
VTK's $x$, $y$, $z$) only decides how it is laid out. A vector field is shown by its components along the display
axes, in display order, next to all four (`name_txyz`), so a permutation changes how a vector is drawn but not what it
is. Components of algebraic objects carry labels that name physical axes and are not permuted.

`export.evaluate_section` evaluates the fields at the hypersurface's own events, which is exact since the fields are
callables. `export.evaluate_grid` samples a whole 4D grid once and `hypersurface.extract` cuts it: the grid slice on a
grid line, linear interpolation between grid lines (exact for fields linear along the fixed axis, and assuming nothing
beyond the samples). A sequence of $t=c$ hypersurfaces is written with a `.series` file, ParaView's time series; any
other sequence gets a `.sections.json` index and no time metadata. `stem.json`, and a root attribute of each file,
record the coordinate order, signature $(+,-,-,-)$, units of $M$, the hypersurface, and each field's kind, basis and
labels.

### 8.2 Fields

`field_function` evaluates, per point: $\tfrac12C^2$ (`weyl2_half`), $\tfrac12C\cdot{}^*C$ (`weyl2_dual_half`, the
imaginary part of $\mathcal W$), $\tfrac12R_{abcd}R^{abcd}$ (`riem2_half`), $R$, the magnitudes of the four curvature
sectors of the normal connection, and $e(\partial_t)$ (`shift`), in frame components. With `components`, it also
evaluates the curvature on the six coordinate planes, $F(\partial_\mu,\partial_\nu)$, and `readout` splits each into
the generator coefficients of its sectors: `F_M_components` (36), `F_P_components` (24), `F_K_components` (24),
`F_D_components` (6), labeled `G(μν)`. Fields are evaluated with `jit` in chunks and written as VTKHDF ImageData.

The cases are a boosted Schwarzschild flyby, a boosted Kerr flyby ($a=0.9$, where `weyl2_dual_half` is nonzero) and the
growing mass. For the boosted Schwarzschild coframe the spatial part of the shift is

```math
\texttt{shift}^i=\gamma\bigl(\beta^i(y)-v\,\delta^i_x\bigr),\qquad y=Lx,\quad\beta(y)=\sqrt{2M/r_y}\,\hat y :
```

a uniform $-\gamma v\,\hat x$ far away, an outward part near the hole, and a stagnation point ahead of it where
$\sqrt{2M/r_y}=v$. It is a gauge-dependent component of the connection; the invariants are the gauge-independent output.

---

## 9. Design choices

**Extensors throughout.** Binding is a call, composition is a call, the adjoint action is `>>`, and contractions are
`.contract` or `.trace`. Curvature, Bianchi, the spin connection, Ricci, Schouten, the invariants and the twistor
structure are written without index bookkeeping.

**numga's idiom.** Module-level named types, small named functions, and no `lambda` or `jax.numpy` in the package.
Positions, steps and loops are batch axes of extensors.

**Fields are callables, not lattices.** Derivatives are exact, the identities hold pointwise, and there is no
discretization error in the identity tests. Grids appear only at the output: slices, renders and export.

**Where arrays remain.**

- `GEN`, to specify test connections in generator coordinates and for the structure constants;
- the sixteen-unknown least-squares solve for the normal conditions;
- the twistor flag and annihilator, which need complex coordinates and an SVD;
- the carry flags of the ray tracer's `lax.scan`;
- `permute_inputs`, for the missing axis permutation.

**An independent oracle.** The first implementation, index-based with `einsum`, is kept in `_array/` with its tests in
`tests/array_reference/`. The extensor code's curvature, transport, spin connection, Christoffel symbols, Schouten
tensor and Weyl squared are compared against it, and it builds the Riemann tensor directly from Christoffel symbols.

**Numerical hazards found.**

- On the GPU, XLA's Triton fusion of float64 dot products returned wrong values when two extensor chains shared
  subexpressions inside one `jit`. `__init__.py` sets `--xla_gpu_enable_triton_gemm=false` before JAX starts.
- `jax.jit(jax.vmap(holonomy_loop))` returned wrong values for batches above about 16 on the CPU; the cause was not
  found, and loops are now evaluated as one extensor batch.
- numga's `even * scalar` with both operands batched compiles very slowly on the GPU (§7.1); scalars go on the left.

---

## 10. Verification map

| Claim | Test file | Level |
|---|---|---|
| metric, null basis, 15 generators, $\mathfrak{so}(2,4)$ closure, Jacobi | `test_algebra.py` | exact |
| $V\widetilde V=1$; `exp` against a matrix exponential | `test_gauge_curvature.py` | 1e-9 |
| $\Omega'$ is a bivector; $\mathcal D'\Psi'=V\mathcal D\Psi$ | same | 1e-11 |
| $[\mathcal D_a,\mathcal D_b]\Psi=\tfrac12F\Psi$; Bianchi; pure gauge is flat | same | 1e-10 |
| transport is a versor; orders 2 and 4; reparametrization invariance; holonomy $\exp(-\tfrac12\epsilon^2F)$ | `test_transport.py` | 1e-6 to 1e-10; $O(\epsilon^3)$ |
| Lorentz sector closed under Lorentz rotors, leaks under $\mathrm{Spin}(2,4)$; spin connection and Riemann against the index code; $48M^2/r^6$ | `test_lorentz_schwarzschild.py` | 1e-8 to 1e-12 |
| torsion zero; normal $f$ is Schouten; Cotton sector; $I=\tfrac12C^2$; generator roles | `test_conformal_cartan.py`, `test_invariants_and_conformal_action.py` | 1e-8 to 1e-16 |
| orientation typing; closed-form frames of six geometries | `test_types_and_frames.py` | 1e-13 |
| Kerr: GTG gauge fields, torsion, Ricci, normal $f=0$, Kretschmann | `test_kerr.py` | 1e-12 to 1e-14 |
| twistors: ideal, norm, split, translator and incidence, Klein correspondence, rays, transport curvature and holonomy, momentum, flag, Kerr Weyl invariant | `test_twistor.py` | 1e-9 to 1e-14 |
| ray tracer: conservation, screen, redshift, polarization, incidence, camera screen | `test_render.py` | 5e-3 (integrator), 1e-12 |
| VTKHDF layout; dual Weyl field; sector components rebuild $F(\partial_\mu,\partial_\nu)$; a fixed-$x$ section equals the fields at its events and the cut 4D grid; polylines | `test_export.py` | read back by VTK; 1e-10 |
| 4D grid order; extraction on and between grid lines; display permutations; series only for $t=c$ | `test_hypersurface.py` | exact |
| ray paths: $h(Z,Z)$, $|\omega|/|\pi|$, $U\widetilde U=1$ and $Ue_a\widetilde U$ orthonormal, fourth order against a fine reference | `test_rays.py` | 1e-9 to 1e-12; ratio 16 |

---

## 11. Limits and open questions

- **No dynamics yet.** The code evaluates the kinematic structure on given geometries; the proposal leaves the dynamics
  open. `variation.py` can now vary a coframe density pointwise (Einstein tensor from $\sqrt g R$, Bach tensor from
  $\sqrt g C^2$), and the compensator action below has been varied.
- **Einstein gravity (§11.4)** needs more than the Weyl-squared density: a compensator, a symmetry-breaking mechanism
  or constraints. The infinity twistor, which fixes $n_\infty$ and so the ω/π split, is a candidate for that structure.
  As the scale tractor $I=\sigma n_\infty+\nabla\sigma+\rho n_o$ it is implemented (`compensator.py`): it is
  parallel exactly on Einstein metrics ($I=n_\infty$ when Ricci-flat, $I\cdot I=\Lambda/3$ otherwise) and
  $I\cdot I=-R/12$ at $\sigma=1$. The Weyl-invariant action $\int\sqrt g\,\sigma^{-4}(I\cdot I+c)$, varied in the
  coframe and $\sigma$ (`variation.py`), gives exactly Einstein's equations with $\Lambda=-6c$ in the gauge
  $\sigma=1$, and the $\sigma$ equation is the trace of the coframe equation. This recovers general relativity in
  second-order form. In first-order form (`first_order.py`), with the origin $n_o$ and the infinity twistor $n_\infty$
  as compensators, the action $\langle F\,DX\,DX\,X\,I\rangle+\beta\langle DI\,DI\,DX\,DX\,X\,I\rangle+
  \gamma\langle (DX)^4X\,I\rangle$ varied in the full connection gives Einstein's equations with
  $\Lambda=6(\gamma-1/\beta)$, and at $\gamma=0$ its solutions are the normal connections of Einstein metrics. The
  quadratic action $\langle F\wedge F\wedge X\wedge I\rangle$, by contrast, is $-\sqrt g\,C^2$ at the normal
  connection and gives Weyl gravity (FRW solves it) with a Cotton condition from its Lorentz equation. Adding
  either quadratic term to an Einstein action keeps the Einstein solutions; linearized about flat space it adds a
  massive spin-2 ghost, $m^2=-1/(24\alpha)$ for $\alpha\sqrt g\,C^2$ added to the compensator action.
- **The GTG displacement field (§9.2, §11.2).** The translation sector is identified with an orthonormal coframe of a
  torsion-free metric. Whether that is a faithful reading of Doran and Lasenby's $h(a)$, with its fiducial flat
  background, is not tested in general. For Schwarzschild, GTG's two gauge freedoms (a displacement and a rotation)
  carry the Painlevé–Gullstrand gauge to the Kerr–Schild one and act on the CSTA connection as a spec gauge
  transformation.
- **The dilation field (§11.3)** is zero except as pure gauge: a Weyl rescaling $e\to e^\lambda e$ is the gauge
  transformation $e^{-\lambda\,n_o\wedge n_\infty/2}e^{\Upsilon\wedge n_o/2}$, $\Upsilon=\nabla\lambda$, of the
  normal connection, and its dilation factor alone gives $b=d\lambda$. A dilation field with curl is not examined.
- **Fundamental versors (§11.1).** Only pure-gauge flatness and transport were examined.
- **Twistors.** No twistor form of the dynamics, no global twistor space (it exists only for conformally flat or
  half-flat spacetimes), and no two-twistor massive particles.
- **Rendering.** No light-travel delay, a single disk crossing per ray, and no gas aberration of the polarization.
  Rays are stopped at $r<1.1\,r_+$ and their final steps there are not converged: a step that ends captured may
  cross $z=0$ far from where the cubic through its ends puts it, so such a step is never counted as a disk crossing
  (it once put six captured Kerr rays on the disk, with $k\cdot\partial_t$ changed by up to $3\times10^4$).
