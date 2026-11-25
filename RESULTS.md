# Results log

All checks below are executable: `source .venv/bin/activate && pytest` (190 tests, ~24 min on the GPU, plus three slow tests: `pytest -m slow tests/test_export.py tests/test_rays.py`). Demos:
`cd src && python -m gauge_theory_gravity.examples.scenarios` and `python -m gauge_theory_gravity.examples.render [slices holonomy twistor schwarzschild kerr]` (figures in `output/`; schwarzschild and kerr take about an hour each at 480×360 on an RTX 3060).
Working algebra: `Algebra("t+x-y-z-p+n-")` = the spec's Cl(2,4), literally. Backend: numga on JAX (float64).

## Implementation: extensors, with an independent array reference

The package is written with numga extensors (`src/gauge_theory_gravity/`). The first implementation used raw coefficient
arrays and `einsum`; it is kept as `src/gauge_theory_gravity/_array/` and used as an independent oracle
(`tests/array_reference/`, and cross-checks in the main tests). Statements below were first verified with the
array version and re-verified on the extensor version; the two agree to 1e-12 or better.

What is an extensor now:
* Orientation follows numga's conformal idiom (as in its Kerr and conformal-circle examples): events and directions are
  intrinsic antivectors, `Direction = Slice.derive.dual` (grade 5 here, the duals of t, x, y, z), read by extrinsic
  vectors `Slice` (t, x, y, z) through `slice & direction`; `SLICES = [t, -x, -y, -z]` read `DIRECTIONS` as delta, and
  `d_mu | d_nu = eta`. The conformal origin and infinity are intrinsic points `n_o.dual()`, `n_inf.dual()`. Connection,
  curvature and versors are extrinsic. numga rejects mixing them (e.g. a slice fed to Omega) before any arithmetic.
* Connection Ω : `Bivector <- Direction`; `Omega(a)` is binding.
* Curvature F : `Bivector <- (Direction, Direction)` = `swap(dΩ) - dΩ + Ω.commutator(Ω)`, with `dΩ = derivative(Ω)(x)`
  from numga's JAX `derivative` (a map with one more input, the step) and numga's `commutator` (= [,]/2, exactly the spec's ½[Ω,Ω]).
  Bianchi uses a small `permute_inputs` helper (numga has no axis permutation for multilinear maps).
* Gauge transformation `V >> Omega` (sandwich of a map's output) minus `2 dV ~V`; covariant derivative `dPsi + ½ Omega * Psi`.
* The sector split M/P/K/D is four extensor projectors `Bivector <- Bivector` built from the null basis by
  contractions and wedges (no 15x15 matrix); the connection is `Ω_L + e ^ n_∞ + f ^ n_o + b (n_o ^ n_∞)`, the wedges
  written as joins with the conformal points, `(e & INFINITY).dual()` and `(f & ORIGIN).dual()` (verified equal to the wedges).
* Coframe `Direction <- Direction`, metric `g = e | e`, frame `E` in closed form (each geometry is a
  `Geometry(coframe, frame)`: identity minus the null dyad for PG and Kerr-Schild, whose dyads square to zero; own closed
  forms for FRW, lapse, boost), Christoffel from `derivative(g)`, spin connection
  `Omega = ½ (W * (Slice ^ Slice)).trace(1,4).trace(2,3)` (each frame input of W paired with the slice that reads it),
  Ricci as `(-F(E, D).commutator(D)).contract(1,3)`, Schouten from it, the quadratic density by two `.contract`s.
* Written in numga's own idiom (after reading `numga/examples`): module-level named types, small named functions with
  docstrings, no `lambda` and no `jax.numpy` in the package (JAX only for `derivative`, and `jit`/`scan` inside the
  functions that need them). Geometries are identity-plus-dyad expressions (`Direction + shift * (t & Direction)`),
  a boost is a rotor sandwich (`rotor >> Direction`), test fields are random polynomials in the position built from
  multilinear maps (`c0 + c1(x) + c2(x, x)`), batches of positions/steps/loops are extensor batch axes, and the
  ordered product of step versors is `cumprod`.
* Still arrays: the `GEN` matrix (generator <-> blade coordinates, used to specify test connections and in the
  structure-constant test), the 16-unknown least squares for the normal conditions (numpy `lstsq` on columns built
  from extensor residuals), the carry flags of the ray tracer's `lax.scan`, and `permute_inputs` (numga has no
  axis permutation for multilinear maps; it transposes the kernel).
* The ray tracer now uses the CSTA's own connection: the momentum is a frame direction, `dk/dλ = -Ω(dx/dλ).commutator(k)`,
  the geodesic equation as parallel transport; it reproduces the shadow radius (0.1670 vs 0.1681 rad at 160×120)
  exactly as the earlier autodiff-of-the-Hamiltonian version did.

## JAX on the GPU (RTX 3060, `jax[cuda12]`)

`jit` of two extensor chains that share subexpressions returned wrong values for the second one on the GPU
(~1e-19 instead of 4e-3; correct on CPU and for each chain alone). It is XLA's Triton fusion of float64 dots:
`--xla_gpu_enable_triton_gemm=false` (or disabling the `cse` pass) fixes it. The package sets the flag before JAX starts
(`gauge_theory_gravity/__init__.py`). Without it, exports and the vacuum identities in `tests/test_export.py -m slow` fail.
The earlier `jit(vmap(holonomy_loop))` failure was on CPU and its cause was not identified; the holonomy map now
evaluates its loops as one extensor batch instead.

Compile time: numga lowers `even * scalar` (the scalar on the right, both batched) by contracting the scalar with the
32x32x1 product table first, an `n x 32 x 32` intermediate; XLA's GPU compiler takes minutes over it once n is a few
thousand (133 s at n = 4096, 27 min 44 s inside the ray tracer with the twistor at 16384 rays), while `scalar * even`
compiles in under a second, and the CPU is unaffected. The ray tracer and `twistor.from_momentum` put the scalar on the
left.

## Verified (Tier A, spec sections 2-8)

| Spec | Claim | Status | Evidence |
|---|---|---|---|
| 2.1 | e_t²=+1, e_{xyz}²=-1, e_+²=+1, e_-²=-1; 15 bivectors | verified | `test_algebra.py` |
| 3 | n_o²=n_∞²=0, n_o·n_∞=1; 15 = 6+4+4+1 generators span the bivectors | verified | `test_algebra.py` (GEN invertible) |
| 3 | so(2,4) sector structure: [M,M]⊂M, [M,P]⊂P, [M,K]⊂K, [M,D]=0, [P,P]=[K,K]=0, [D,P]∝P, [D,K]∝-K-type grading, [P,K]⊂M⊕D; Jacobi | verified | `test_structure_constants_so24` |
| 5 | V Ṽ = 1 for V = exp(B), random 15-component B | verified, 1e-9 | numga generic exp agrees with an independent scipy `expm` of the 32×32 left-regular representation |
| 5 | V and -V give the same adjoint action; adjoint action preserves grade-2 and the inner product | verified | `test_gauge_curvature.py` |
| 5.1 | Ω' = VΩṼ − 2(a·∇V)Ṽ stays bivector-valued | verified, 1e-11 | `test_spec_5_1...` |
| 6 | D'Ψ' = V DΨ | verified: JAX 1e-11; finite-difference 1e-7 | `test_spec_6...` |
| 7 | F = ∂Ω − ∂Ω + ½[Ω,Ω] is bivector valued, antisymmetric, covariant (F' = VFṼ) | verified | `test_gauge_curvature.py` |
| 7.1 | [D_μ,D_ν]Ψ = ½F_μνΨ (including the ½ in front of the commutator) | verified, 1e-10 | `test_spec_7_1...` |
| 7 | Bianchi: cyclic D_λF_μν = 0 with D X = ∂X + ½[Ω,X] | verified, 1e-9 | `test_bianchi_identity` |
| 7 | pure gauge Ω = −2(∂V)Ṽ is flat; generic Ω is not | verified | `test_pure_gauge_is_flat...` |
| 8 | transport T = P exp(−½∫Ω) is a versor (TT̃=1) | verified, 1e-10 | `test_transport.py` |
| 8 | pure-gauge transport = V(x2)V(x1)⁻¹ ; non-flat Ω has non-trivial holonomy | verified | `test_transport.py` |
| 8 | spec caveat: a general connection is not pure gauge | verified: holonomy ≠ 1 for the same class of fields that fail flatness | `test_pure_gauge_holonomy_trivial_but_generic_not` |

Derived (not stated in spec) and now pinned numerically:
* Parallel transport convention: Ψ(s) = T(s)Ψ(0), dT/ds = −½Ω(γ')T; later steps multiply on the left.
* Holonomy around the rectangle x0 → +μ → +ν → −μ → −ν of side ε: **T = exp(−½ ε² F_μν) + O(ε³)**.
  Same sign in three planes tested (μν = 01, 12, 03).
* Magnus-4 transport is 4th order; midpoint is 2nd order; 4th-order finite-difference stencil gives ~16× error drop per halving.
* Conformal action of versors on embedded points X = T_x n_o T̃_x, T_x = exp(½ x∧n_∞) (null cone preserved):
  * V = exp(c P_a): x → x + 2c e_a (translation, n_∞ fixed). So P_a really are translations.
  * V = exp(λ D): x → e^{−2λ} x (dilation).
  * V = exp(c K_a): fixes n_o, maps null cone to itself (special conformal).
  * V = exp(c M_ab): preserves x² (Lorentz).
  This confirms the spec's generator labeling (P ↔ e_a∧n_∞, K ↔ e_a∧n_o).

## Lorentz sector (spec 9.1)

* Restricting Ω to M_ab keeps F in the M sector (P=K=D = 0 to 1e-12), and Lorentz-rotor gauge transformations keep Ω in the sector.
* A generic Spin(2,4) transformation takes a Lorentz connection out of the sector (P, K, D components O(0.1)); so the Lorentz restriction is gauge-fixing, not a gauge-invariant statement. This is the spec's caveat, made concrete.
* Sign map to numga's own STA (`x+y+z+t-`, opposite sign): φ(M_ab) = −M_ab is a Lie-algebra homomorphism (structure constants differ by exactly −1).
* Schwarzschild (Painlevé–Gullstrand tetrad, Levi-Civita spin connection embedded as Ω in the Lorentz sector):
  F^{ab}_{μν} from the CSTA formula equals the Riemann tensor computed independently from Christoffel symbols (1e-9);
  ½[Ω,v] reproduces the Levi-Civita derivative of frame vectors; Ricci = 0 (1e-16); Kretschmann = 48M²/r⁶ (relative error 3e-15, r from 2.2 to 10).

## Tier B: translation sector, normal conformal connection (spec 9.2, 10)

* **Translation sector = coframe.** Put e^a_μ in the P coordinates, Levi-Civita ω in the M sector: F_P = 0 (torsion-free) for Schwarzschild, FRW and a generic static-lapse metric.
* **Normal conditions** (F_P = 0, F_D = 0, F_M trace-free) are always solvable pointwise for the K coefficients f (16 unknowns, rank 16), and the solution is
  **f^a_μ = P^a_μ, the Schouten tensor** ½(Ric − R g/6), exactly (ratio 1.0000000000000002 in both non-vacuum test spacetimes) in this normalization. For Ricci-flat spacetimes f = 0.
* With the normal f: F_M = trace-free (Weyl-type) part, F_K = Cotton-type: zero for FRW (conformally flat), nonzero for the lapse metric, zero in vacuum.
* **Quadratic action density.** g^{μρ}g^{νσ}⟨F_μν F̃_ρσ⟩₀ of the full 15-component F equals **½ C_{abcd}C^{abcd}** (Weyl-squared) to 1e-9 for the three spacetimes; for the Lorentz-only connection it does not (it also sees Ricci). This confirms spec section 10's remark that a quadratic curvature action gives Weyl-squared gravity, at the level of the Lagrangian density.

## Kerr in the CSTA (`cartan.kerr`, `tests/test_kerr.py`)

* Geometry: GTG's position gauge field in Cartesian Kerr-Schild form, identity plus a null dyad,
  `e = Direction - f l (l . )`, `f = M r^3/(r^4 + a^2 z^2)`, `l = d_t - n` ingoing and flat-null, so `E = Direction + f l (l . )`
  exactly. r, f, l and `g = eta - 2 f l l` agree with the coordinate formulas to 1e-14.
* As a conformal connection (coframe in the translation sector, its Levi-Civita connection in the Lorentz sector),
  at a = 0.6 and a = 0.9: F_P, F_K, F_D <= 1e-16 (torsion-free, Cotton and dilation zero), R ~ 1e-16, the normal
  conditions give f = 0 (rank 16, f ~ 1e-16), and `2 <F F~>` of the normal connection equals the analytic Kerr
  Kretschmann scalar 48 M^2 (r^6 - 15 r^4 u + 15 r^2 u^2 - u^3)/(r^2 + u)^6, u = a^2 z^2/r^2, to 5e-15 relative.
  With a = 0 it gives 48 M^2/r^6 (Eddington-Finkelstein form of Schwarzschild).
* Cross-check against numga's own Kerr example (`numga/examples/relativity/kerr/core.py`, STA `x+y+z+t-`, the
  opposite overall sign): its `gauge_inverse` equals our coframe exactly (difference 0.0), and its closed-form rotation
  gauge bivector (Francis-Kosowsky) along `e(d_mu)` turns directions exactly as our autodiff Levi-Civita connection
  `Omega(d_mu)` does (1e-16; bivectors compared by their action on directions, which does not depend on the overall
  sign of the metric). The result is the same for `rotation(d_mu)` and `rotation(E(d_mu))`, since the Kerr-Schild
  rotation along `l` itself vanishes.
* This settles, for Kerr-Schild Kerr, the question left open below: GTG's h-function and rotation gauge field are
  exactly the coframe and Lorentz connection that the CSTA's translation and Lorentz sectors hold.

## Twistors (`twistor.py`, `tests/test_twistor.py`)

The pseudoscalar `I = txyzpn` squares to -1 and is central in the even subalgebra, which is therefore M(4,C), and
Spin(2,4) = SU(2,2). Twistor space is the minimal left ideal `T = Even * P`, `P = 1/4 (1 + I xy)(1 + I tp)`: P is
idempotent and its own reverse, and T has real dimension 8. Every even spinor of the gauge theory is four twistors,
and `Psi P` picks one. All of the following hold to 1e-11 or better:
* Norm `h(Z1, Z2) = 4 <~Z1 Z2>_{0,6}` (scalar plus I part, read as a complex number): Hermitian, signature (2,2). In the
  adapted basis `h = 2 Re(conj(omega) . pi)`.
* Split `omega = 1/2 n_inf n_o Z`, `pi = 1/2 n_o n_inf Z`: the null kernels of infinity and the origin, each h-null.
* Versors act as SU(2,2) matrices (unit determinant, h preserved); bivectors as traceless h-antihermitian matrices.
* Incidence: the conformal point X = T_x n_o T̃_x of an event (null, unit weight, reads back x; T_x = exp(½ x∧n_∞) is
  a versor whose series stops at the linear term) annihilates a complex 2-plane of T, the twistors `T_x pi_0`, i.e.
  `omega = 1/2 x n_inf pi`; in coordinates `omega = -(i/sqrt 2) x^{AA'} pi` with `x^{AA'}` Hermitian and `det = x^2`.
  The alpha-planes of two events meet in a complex line when the events are null separated and only in 0 otherwise.
* Light rays: the real six-vectors annihilating a null twistor form a 2-plane; none annihilate a non-null one. `ray(Z)`
  reads off an event and a null direction, and every event along it is incident with Z.
* Local twistor transport: with the normal conformal connection (f = Schouten), the curvature acting on T is zero for
  FRW (< 1e-9) and of size 0.2 for the static-lapse metric (t,x plane); a holonomy loop of side 0.05 is in SU(2,2) and
  equals `expm(-1/2 size^2 F)` to O(size^3).
* Momentum: the moment map of the translations, `p^mu = -Im h(Z, (s^mu ^ n_inf) Z)`, is the null flag pole of pi;
  `from_momentum` builds a twistor through the origin with a given momentum (round trip to 1e-13, phase-invariant,
  quadratic in Z, unchanged by translating Z). The complex trace on T is `4 <A>_{0,6}` (matches the 4x4 matrix trace).
* Complex Weyl invariant `weyl_invariant`: 2 tr_omega(F F~) contracted over the frame = (C^2 + i C.*C)/2. For Kerr
  (a = 0.6, three events) it equals 24 M^2 / (r - i a cos theta)^6 to 1e-10, its real part is the quadratic density to
  1e-12, and for Schwarzschild it is real. Exported as `weyl2_dual_half` (checked eagerly through the export's field
  function and readout on Kerr a = 0.9 and Schwarzschild); new export case `kerr_flyby` (Kerr boosted to v = 0.6).
  The kerr_flyby export itself has not been run.
* Ray tracer (`render.py`): each ray carries its local twistor, started at the camera from the photon's momentum
  (omega = 0) and moved by `dZ/dlambda = -1/2 Omega(velocity) Z` with the normal connection (Lorentz plus `k ^ n_inf`;
  Schouten zero in vacuum). At the disk (12x9 test scenes, 1800 steps), max |omega|/|pi| is 3.0e-7 (Schwarzschild) and
  3.2e-6 (Kerr a = 0.9), and the frequency ratio read from the twistor's momentum matches the transported momentum's to
  3.9e-8 and 8.1e-6 (`test_render.py`, tolerance 5e-3). The full-size renders were not re-run; the twistor does not
  change the shading.
* Twistor holonomy of Kerr a = 0.9 on the meridian plane (`render twistor`, `output/render_twistor.png`; 40x40 loops of
  side 0.4 in (x, z), 27 s): eigenvalue deviation up to 0.345, SU(2,2) to 1.1e-14, boost fraction of the loop curvature
  (angle of the dominant eigenvalue's logarithm from the imaginary axis) 0.8 to 88 degrees. Control, 64 loops outside
  r = 2.5: 0.00 degrees for spin 0 and 1.3 to 56 degrees for a = 0.9, so the boost part comes from the spin. The panel
  beside it shows the phase of the complex Weyl invariant for comparison only; the two are not compared quantitatively.
  With spin 0, loops next to r = 0 blow up (deviation 4e8), so the figure is for Kerr only.
* Not done: the Petrov type of Kerr from the eigenvectors of the twistor holonomy, and the infinity twistor's role in
  breaking conformal to Poincare invariance.

## Closed-form frame fields (`cartan.Geometry`, `tests/test_types_and_frames.py`)

Each geometry carries its frame field E with the coframe. PG (constant or growing mass) and Kerr-Schild are identity
plus a dyad that squares to zero (for PG the time slice does not read the spatial shift), so E is identity minus the
dyad; FRW and de Sitter `TimePart + SpacePart/a`, lapse `TimePart/N + SpacePart`, boost `rotor << E(rotor >> x)`. For all seven
geometries `e E = 1` to 1e-13 and E equals `e.inverse()` to 1e-12; every earlier result is unchanged.

## GTG gauge freedom and Weyl rescaling (spec 11.2, 11.3; `tests/test_gtg_gauges.py`, `tests/test_weyl_rescaling.py`)

* **Displacement and rotation gauges (11.2).** Schwarzschild in the Newtonian (Painleve-Gullstrand) gauge and in the
  Kerr-Schild gauge (`kerr(M, 0)`) are related by a GTG displacement t_KS = t_PG + F(r),
  F = 4M ln((sqrt r + sqrt 2M)/sqrt 2M) - 2 sqrt(2Mr), and a radial boost of rapidity ln(1 + sqrt(2M/r)) (derived by
  hand; its sign fixed numerically). At r = 2.23 M and r = 1.12 M (inside the horizon): the pulled-back Kerr-Schild
  metric equals PG to 1e-14, the pulled-back coframe is the boosted PG coframe to 1e-14, and the pulled-back full CSTA
  connection (Lorentz and translation sectors) equals the PG connection transformed by the boost with the spec's law
  Omega' = R Omega ~R - 2 dR ~R, to 1e-13. D phi from autodiff matches its closed form, F' = -s/(1+s), to 1e-14.
* **Doran-Lasenby PG rotation gauge field.** In the PG frame, Omega(E_0) = 0 and Omega(E_k) is a pure boost turning
  d_0 into -B_ik d_i, B = sqrt(2M/r^3)(1 - 3/2 r^ r^) (Cartan's first structure equation), to 1e-14.
* **Weyl rescaling (11.3).** For e -> e^lambda e with a lambda depending on all four coordinates, on Schwarzschild and
  the static lapse: the normal connection of the rescaled coframe is the normal connection of e transformed by
  V = exp(-lambda/2 n_o ^ n_inf) exp(1/2 Upsilon ^ n_o), Upsilon = grad lambda in the frame of e, to 1e-13 (the factor
  order, signs and halves were found by a search over the obvious candidates; only this one matched). The dilation
  factor alone leaves the Lorentz sector unchanged, scales the translation sector by e^lambda and the special-conformal
  sector by e^-lambda, and switches on the dilation sector with b = d lambda exactly (1e-14); the curvature's dilation
  sector stays zero. Weyl squared scales by e^{-4 lambda} to 1e-12.
* So in these checks the dilation and special-conformal sectors carry a choice of scale (gauge), not extra degrees of
  freedom. Not checked: a dilation field that is not d lambda (b with nonzero curl, which changes the curvature's
  dilation sector), Doran's spinning PG gauge of Kerr, and any dynamics.

## The scale tractor as compensator (spec 11.4, first step; `compensator.py`, `tests/test_compensator.py`)

* `scale_tractor(coframe, sigma)` builds I = sigma n_inf + grad sigma + rho n_o, rho = -1/4 (box sigma + J sigma), a
  field of R^{2,4} vectors (the standard tractor formula; its signs and factors in this algebra were read off V >> n_inf,
  and rho matched at the first try). `tractor_derivative` is D_a I = a . grad I + Omega(a) x I with the normal connection.
* **Weyl covariance.** On e^lambda times Schwarzschild, I(sigma = e^lambda) = V >> n_inf with the 11.3 versor V, to
  1e-14, and it is parallel. On the static lapse (not Einstein), I(e^lambda e, e^lambda) = V >> I(e, 1) to 1e-13, so
  I . I is Weyl-invariant.
* **Einstein-Hilbert density.** At sigma = 1, I . I = -J/2 = -R/12 to 1e-14 on the static lapse, growing mass and de
  Sitter. The conformally invariant density sqrt(g) sigma^-4 I . I is therefore -sqrt(g) R/12 in the gauge sigma = 1.
* **Parallel exactly on Einstein metrics.** D I = 0 at sigma = 1 for Schwarzschild, Kerr a = 0.9 (I = n_inf, null) and
  de Sitter (`cartan.de_sitter`, flat slicing, I . I = H^2 = Lambda/3), to 1e-13. Not parallel (> 1e-2) on the static
  lapse and the growing mass. `wave_operator` matches the closed form on de Sitter to 1e-13.
* **Integrability.** A parallel I satisfies F(a, b) x I = 0. `parallel_candidates` returns the null space at a point.
  It always contains n_o when the torsion sector vanishes (Weyl and Cotton curvature commute with n_o), but n_o has
  scale I . n_o = 0. For the static lapse and the growing mass n_o is the only candidate, so neither is conformally
  Einstein near the point; for Schwarzschild the null space is span(n_o, n_inf).
* So the infinity twistor of proposal section 10 can be read as this compensator: fixing it to n_inf (Ricci-flat) or
  to a constant non-null vector (Lambda) selects Einstein metrics, and its norm gives the Einstein-Hilbert density.
  Not done: varying an action built from it (needs the Euler-Lagrange machinery of plan phase 1), and the
  MacDowell-Mansouri form.

## Field equations by the pointwise Euler-Lagrange operator (plan for 11.4, phase 1; `variation.py`, `tests/test_variation.py`)

* `euler_lagrange(density, coframe, x4)` returns E[a, mu] = delta S / delta e^a_mu for a density of order <= 2 in the
  coframe. The coefficients dL/d(d_alpha e) are read at x by perturbing the coframe with a polynomial centred there
  and differentiating in its amplitude (`jax.jacfwd`); E = A - d_mu A^mu + sum_{mu<=nu} d_mu d_nu A^{mu nu}. No grid,
  no integration. About 45-125 s per point on the CPU (the GPU was occupied by another process).
* **Einstein-Hilbert** sqrt(g) R: E = -2 sqrt(g) e_{a nu} G^{mu nu}, i.e. delta S/delta g_{mu nu} = -sqrt(g) G^{mu nu},
  against G from the array reference's Christoffel symbols, to 1e-13 (static lapse) and 1e-12 (FRW, slow); zero on
  Schwarzschild to 1e-13.
* **Weyl squared** sqrt(g) C^2: E = -8 sqrt(g) e_{a nu} B^{mu nu}, i.e. delta S/delta g_{mu nu} = -4 sqrt(g) B^{mu nu},
  against the Bach tensor B_ab = nabla^c nabla_c P_ab - nabla^c nabla_a P_bc + P^cd C_acbd written out on the array
  reference, to 1e-13 on the static lapse (|B| ~ 0.7). Bach-flat to 1e-13 on Schwarzschild, and (slow) on Kerr a = 0.9,
  FRW and de Sitter to 1e-12.
* **Noether identities.** Local Lorentz invariance (eta E e^T symmetric) holds to 1e-13 for both densities; the Weyl
  trace E[a, mu] e^a_mu vanishes for Weyl squared to 1e-13 (B is trace-free).
* The factors -2 and -8 were not assumed: they came out of a fit at one point and then hold to rounding.

## The compensator action: Einstein's equations (plan for 11.4, phase 3A; `tests/test_compensator_action.py`)

* `variation.compensator_density(c)` is sqrt|g| sigma^-4 (I . I + c) with the scale tractor I of sigma; it is
  Weyl-invariant, and with g' = g/sigma^2 it is -sqrt|g'| (R' + 12 c)/12. `euler_lagrange_with_scale` varies it in the
  coframe and in sigma.
* **Einstein's equations in the gauge sigma = 1.** The coframe equation is sqrt(g) e_{a nu} (G^{mu nu}/6 + c g^{mu nu}):
  on the static lapse it equals -1/12 of the array reference's Einstein-Hilbert equation to 1e-14. So the field
  equations are G + 6 c g = 0, Einstein's with Lambda = -6c (de Sitter at Hubble rate H has G = 3 H^2 g here), and on
  shell I . I = -R/12 = -2c.
* **Solutions.** de Sitter (H = 0.5) with c = -H^2/2 solves both equations to 1e-14; with c = 0 it does not (0.36,
  slow). Kerr a = 0.9 with c = 0 (slow). A Weyl-rescaled Schwarzschild (e^lambda e, not Einstein) with its scale
  sigma = e^lambda solves both to 1e-14, and with sigma = 1 it does not (slow): the solution is a class under Weyl
  rescaling together with its scale, as the gauge picture of 11.3 says.
* **The scale equation is the trace.** E_sigma sigma + E[a, mu] e^a_mu = 0 to 1e-14 off shell, at sigma = 1 and (slow)
  with a nonconstant sigma and c = 0.3: Noether's identity of the Weyl invariance, so sigma carries no equation of its
  own and the gauge sigma = 1 loses nothing.
* What this establishes for 11.4: in the CSTA, with the scale tractor (infinity twistor) as compensator, this action's
  equations are exactly Einstein's with a cosmological constant. It is a reformulation of general relativity, not a
  new theory. Not done: adding alpha Weyl^2 (Einstein metrics still solve it, being Bach-flat) and whether the
  proposal wants it; the MacDowell-Mansouri form (phase 3B); writing the action in terms of the full connection
  Omega and varying Omega independently (first-order form); matter.

## Not done / not established (be careful when citing)

* **Einstein recovery only in second-order form.** The compensator action sqrt(g) sigma^-4 (I . I + c) gives Einstein's equations with Lambda (section above), but as an action of the coframe and the scale, with the connection the normal one; a first-order form varying the full CSTA connection, the MacDowell-Mansouri form, matter couplings and any Weyl^2 admixture were not attempted. Spec 11.4 is answered for this action only.
* **Displacement gauge field of GTG.** For Kerr-Schild Kerr (and so Schwarzschild at a = 0) the CSTA's translation and
  Lorentz sectors were checked to hold exactly GTG's position gauge field and rotation gauge field, against an
  independent GTG implementation (numga's Kerr example; section "Kerr in the CSTA"). GTG's gauge freedom in h was
  then checked for Schwarzschild between the PG and Kerr-Schild gauges (section above). Not checked: Doran's spinning
  PG gauge of Kerr, and GTG's field equations themselves.
* **Dilation gauge field b** is zero in Tier B. The only nonzero b examined is the pure gauge b = d lambda of a Weyl
  rescaling (section above); a b with nonzero curl and any dynamics or gauge fixing were not explored (spec 11.3).
* **Fields are callables on points.** Identities are checked pointwise; 4D lattices appear only for export (`hypersurface.Grid4`), sampled from the callables.
* **Fundamental versor fields (spec 11.1)** not explored beyond pure-gauge flatness.
* Phase 6's Lorentz-sector equivalence is kinematic: it verifies that F in the sector is the Riemann bivector map; no GTG dynamical equations were involved.

## Findings about the spec

* All formulas stated in sections 4-8 are mutually consistent as written (checked numerically, including all factors of ½).
* The only convention that needed pinning down was the holonomy orientation sign above, and the normalization of P (translation by 2c per unit coefficient) and D (scale e^{−2λ}), both due to the ½ in the spin-representation sandwich.

## Renderings (`src/gauge_theory_gravity/examples/render.py`, figures `output/render_*`)

* **Field-strength slice** (`render_slices.png`): the gauge-invariant `<F F~>` on a plane through the hole, agrees with 24M²/r⁶ to 1e-11 down to r≈0.4; it diverges toward the singularity, with the horizon (r=2M) unremarkable in PG coordinates. The translation-sector shift √(2M/r) is shown as a gauge-dependent comparison.
* **Holonomy map** (`render_holonomy.png`): max |T_loop − 1| for 0.4×0.4 loops of the full conformal connection (coframe + Lorentz), ≈ ε²·curvature, from 1e-4 at r=8 to O(1) at the center.
* **Ray tracer** (`trace_scene`): a ray's frame momentum k and the camera screen (a map `Direction <- Screen`) are
  parallel transported by the Lorentz connection, `-Omega(E(k)).commutator(.)`, RK4 with steps 0.03·clip(r, 1, 10)
  backward in the affine parameter, pixels in chunks of 16384 through one compiled `lax.scan`. Rays go on past the disk
  (so the shadow is measured by itself); the first disk crossing is located by cubic Hermite interpolation of the step,
  with the end rates carried in the scan, and two Newton steps on its height. Rays reaching r < 1.1 r_+ are captured
  (in these horizon-penetrating charts a backward ray hugging the horizon continues onto an unphysical branch).
* **Disk physics** (`examples/light.py`, adapted from numga's Kerr example, MIT): prograde Kepler gas
  `d_t + Omega d_phi`, from the innermost stable orbit (Bardeen-Press-Teukolsky) to 14 M; frequency ratio
  observed/emitted `1/(k | u_gas)` with `k | u_camera = 1`, which carries gravitational redshift and Doppler shift;
  Page-Thorne temperature (matches numga's to all digits at a = 0.8; peaks at 9.55 M for a = 0) times that ratio,
  shown as tone-mapped blackbody color over a star field. Polarization: numga's model, emission polarized across a
  toroidal field in the local frame (gas aberration left out), as a unit-intensity coherency `Screen <- ScreenSlice` on
  the transported screen, read by a rotating analyzer slice.
* **Schwarzschild** (`render_schwarzschild.png`, `.gif`; 480×360, camera at r = 50 M, 20° above the disk, field 1.0 rad):
  shadow angular radius 0.1018 rad against the analytic 0.1020 (static observer, sin α = (3√3 M/D)√(1−2M/D)), 0.2% low
  (it was 0.1670 vs 0.1681 at r = 30, 160×120). Frequency ratio on the disk 0.49–1.35, observed temperature up to
  4580 K. Photon energy k·∂t and angular momentum k·∂φ change by at most 1.1e-6 from camera to disk; at the disk
  |ω|/|π| < 3e-15. The GIF shades the same trace
  with a two-armed pattern riding the Kepler flow (no light-travel delay).
* **Kerr a = 0.9** (`render_kerr.png`, `.gif`; 480×360, 15° above the disk): D-shaped shadow flattened
  on the approaching side, disk down to 2.32 M, frequency ratio up to 1.41, observed temperature up to 11137 K,
  |ω|/|π| < 2.4e-15 at the disk, k·∂t and k·∂φ conserved camera to disk to 2.6e-4, frequency ratio 0.185–1.409.
  **Fixed defect:** six captured rays on the receding edge of the shadow were shaded as disk, with k·∂t and k·∂φ
  changed by up to 2.8e4 and frequency ratio 0.000 (a red speckle). Each reached just above the capture radius
  (r 1.58–1.60 against 1.1 r₊ = 1.579), and its last step landed inside the horizon (r 0.63–1.01) across z = 0;
  the cubic Hermite crossing of that step put the crossing at r 4–12, on the disk. A crossing now counts only if the
  step ends outside the capture radius. Re-traced with `rays.py` under both transports, at the render's step and
  half of it, none of the six crosses the disk: the transport played no part. Polarizer: crossed analyzers pass 1 to 1e-12; one analyzer 0.10–0.90; per disk pixel the brightest over
  darkest frame is 1.6 (median) to 3.7 (90th percentile), a band sweeping round the disk. Its exposure is dimmer
  (factor 0.6 instead of 2) to keep that swing out of the tone curve's roll-off.
* **Tests** (`tests/test_render.py`, 12×9 scenes): conservation (< 5e-3), screen orthonormal and across the ray
  (< 5e-3), frequency ratio against the conserved quantities (< 5e-3), unit-intensity coherency (1e-12), Page-Thorne.
* The first versions of the holonomy map and lensing image were wrong (as reported when they were made). Causes: (i) `jax.jit(jax.vmap(holonomy_loop))` returns wrong values for batches of more than about 16 points while `jax.vmap` without `jit`, `jax.lax.map` and single-point calls agree with each other (reproduced with the array and the extensor transport alike; the map now uses `lax.map`); I did not establish whether the fault is in numga, JAX or XLA. (ii) The camera lay in the disk plane, and the horizon speckle above. Later, at r = 30 with a 0.5 rad field the disk overflowed the frame; the camera moved to r = 50 with 0.7 rad.
* Not done: light-travel delay in the animation; multiple disk crossings (the disk is opaque, nearest crossing only);
  gas aberration in the polarization; the Schwarzschild image re-traced with the Hermite crossing.

## ParaView export: hypersurfaces of spacetime (`hypersurface.py`, `export.py`)

* **Model.** Fields are functions of (t, x, y, z) (order, signature (+,−,−,−) and units M recorded in every file). A
  dataset is a hypersurface x^μ = c for any one of the four coordinates, laid out along any order of the other three:
  t = c is a spatial snapshot, x, y or z = c a section over time and two spatial axes. Fields are evaluated at the
  hypersurface's own events, or sampled once on a 4D grid (`evaluate_grid`) and cut from it (`hypersurface.extract`:
  the grid slice on a grid line, linear interpolation between grid lines).
* **Files.** VTKHDF ImageData per hypersurface, the display axes as VTK's x, y, z; a `.vtkhdf.series` only when every
  hypersurface is t = c (a time series), else a `.sections.json` index; `stem.json` and a root attribute of each file
  record the coordinate conventions, the hypersurface and, per field, its kind, basis and labels.
* **Fields.** ½C² (normal connection), ½C·*C, ½Riem² (Lorentz-only), R, the sector magnitudes M/P/K/D, the
  translation-sector vector e(d_t) (`shift`: its frame components along the display axes, so on t = c the old spatial
  shift; `shift_txyz`: all four), log10 versions of the invariants, a validity mask; with `--components`, each sector's
  generator coefficients of F(d_μ, d_ν) on the six coordinate planes (36 + 24 + 24 + 6 components), labeled `G(μν)`.
* **Verified** (`tests/test_hypersurface.py`, `tests/test_export.py`): grid ordering, spacing and metadata;
  extraction on grid lines equals the grid slice for each of the four axes; between grid lines it reproduces linear
  fields exactly; display permutations move points and the shown vector components together while the four-component
  array is unchanged; VTK's reader places every point at the coordinates stored for it, also for a permuted (z, t, y)
  section; the sector components rebuild each sector of F(d_μ, d_ν) and, summed, F itself (Kerr, 1e-10); a section at
  fixed x of the Schwarzschild flyby equals the fields evaluated at its events and the same section cut from a sampled
  4D grid (slow test). ParaView itself was not opened.
* Cost, from a timing of 4000 points on CPU with `jit(vmap)`: about 0.65 ms/point, i.e. about 70 s per section at
  n = 48 (110k points); a whole 48⁴ grid (`--via-grid`) is 48 times that. Run:
  `PYTHONPATH=src python -m gauge_theory_gravity.export flyby --n 48 --frames 16 [--fixed x --order tyz]`.
  The full-size exports have not been run.

## Ray paths, twistor states, extrinsic and intrinsic transport (`examples/rays.py`, `tests/test_rays.py`)

* **Export.** Rays from the static camera, recorded every few steps, as VTKHDF PolyData polylines drawn in any three
  of (t, x, y, z); point data: coordinates, affine parameter, frame momentum, coordinate velocity, frame frequency
  and redshift relative to the camera, the screen carried by the twistor's flag, drifts of k·∂_t and k·∂_φ, k·k, the
  twistor's coordinates, amplitudes, h(Z, Z), h(Z, Z)/|Z|², |ω|/|π|, arg(π₁ π̄₀), and the
  versor's coefficients, |UŨ − 1|, the transported basis U e_a Ũ of R^{2,4}, tr U / 4 and tr(Ũ₀U)/4 against the first
  ray (extrinsic transport only). Files are named `rays_<hole>_<transport>.vtkhdf`.
* **Extrinsic (versor) transport**, the default (`--transport extrinsic`). The state is (x, U) with Z = U Z₀ and U in Spin(2,4), stepped by the commutator-free Lie
  group method CF4 (exponentials of so(2,4) bivectors, numga's `exp`, which matches a matrix exponential to 4e-16), the
  event by the classical Runge–Kutta stages.
* **Intrinsic (twistor) transport** (`--transport intrinsic`): the state is (x, Z), stepped additively by the
  renderer's Runge–Kutta step, which is the renderer's transport. No versor is formed. Same rays: |ω|/|π| < 1e-15 and
  h(Z, Z)/|Z|² < 1e-15 (ω = 0 forces h = 2 Re(ω̄·π) = 0), Killing drifts 3e-11, as for the extrinsic transport; the
  two agree on the events and twistors within 1e-8 after 60 steps of 0.2.
* **Verified** (Schwarzschild, five rays from r = 30, fixed steps 0.2): h(Z, Z)/|Z|² < 1e-12 and k null; the versor
  stays in the group (|UŨ − 1| < 1e-10, U e_a Ũ orthonormal to 1e-10), starts at the identity, carries the camera
  twistor exactly at the first record, and agrees with a run at half the step to 7e-11. Against steps of 0.025 it
  converges at fourth order (error ratio 16.0 per halving, slow test). ω drifts at truncation level (8e-11 at step 0.2,
  falling as h⁴), because CF4's mixed generators, such as 2A₃ − A₁, are not built from the twistor they act on; over
  200 fixed steps of 0.2 from r = 30 it reached 4e-4.
* The trace compiles in about 50 s on the GPU (five exponentials per step).
