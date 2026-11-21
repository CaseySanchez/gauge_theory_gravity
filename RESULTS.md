# Results log

All checks below are executable: `source .venv/bin/activate && pytest` (150 tests, ~19 min on the GPU, plus one slow test: `pytest -m slow tests/test_export.py`). Demos:
`cd src && python -m gauge_theory_gravity.examples.scenarios` and `python -m gauge_theory_gravity.examples.render [slices holonomy lensing kerr]` (figures in `output/`; lensing and kerr take about an hour each at 480×360 on an RTX 3060).
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
* Conformal action of versors on embedded points X = n_o + x − ½x² n_∞ (null cone preserved):
  * V = exp(c P_a): x → x + 2c e_a (translation, n_∞ fixed). So P_a really are translations.
  * V = exp(λ D): x → e^{−2λ} x (dilation).
  * V = exp(c K_a): fixes n_o, maps null cone to itself (special conformal).
  * V = exp(c M_ab): preserves x² (Lorentz).
  This confirms the spec's generator labelling (P ↔ e_a∧n_∞, K ↔ e_a∧n_o).

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
  **f^a_μ = P^a_μ, the Schouten tensor** ½(Ric − R g/6), exactly (ratio 1.0000000000000002 in both non-vacuum test spacetimes) in this normalisation. For Ricci-flat spacetimes f = 0.
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

## Closed-form frame fields (`cartan.Geometry`, `tests/test_types_and_frames.py`)

Each geometry carries its frame field E with the coframe. PG (constant or growing mass) and Kerr-Schild are identity
plus a dyad that squares to zero (for PG the time slice does not read the spatial shift), so E is identity minus the
dyad; FRW `TimePart + SpacePart/a`, lapse `TimePart/N + SpacePart`, boost `rotor << E(rotor >> x)`. For all six
geometries `e E = 1` to 1e-13 and E equals `e.inverse()` to 1e-12; every earlier result is unchanged.

## Not done / not established (be careful when citing)

* **No field equations, no compensator, no Einstein recovery.** Only the density of the quadratic action was checked. Varying it (Bach equations), adding a compensator field, and a symmetry-breaking mechanism were not attempted. Spec 11.4 remains open.
* **Displacement gauge field of GTG.** For Kerr-Schild Kerr (and so Schwarzschild at a = 0) the CSTA's translation and
  Lorentz sectors were checked to hold exactly GTG's position gauge field and rotation gauge field, against an
  independent GTG implementation (numga's Kerr example; section "Kerr in the CSTA"). Not checked: GTG's gauge freedom in
  h (other gauges, e.g. the Doran PG gauge with its own rotation field), and GTG's field equations themselves.
* **Dilation gauge field b** was set to zero everywhere in Tier B; its dynamics/gauge-fixing was not explored (spec 11.3).
* **Fields are callables on points, not 4D lattices** (deviation from Phase 3 option 1). This was enough for every identity and costs nothing in accuracy; lattice demos can be added later.
* **Fundamental versor fields (spec 11.1)** not explored beyond pure-gauge flatness.
* Phase 6's Lorentz-sector equivalence is kinematic: it verifies that F in the sector is the Riemann bivector map; no GTG dynamical equations were involved.

## Findings about the spec

* All formulas stated in sections 4-8 are mutually consistent as written (checked numerically, including all factors of ½).
* The only convention that needed pinning down was the holonomy orientation sign above, and the normalisation of P (translation by 2c per unit coefficient) and D (scale e^{−2λ}), both due to the ½ in the spin-representation sandwich.

## Renderings (`src/gauge_theory_gravity/examples/render.py`, figures `output/render_*`)

* **Field-strength slice** (`render_slices.png`): the gauge-invariant `<F F~>` on a plane through the hole, agrees with 24M²/r⁶ to 1e-11 down to r≈0.4; it diverges toward the singularity, with the horizon (r=2M) unremarkable in PG coordinates. The translation-sector shift √(2M/r) is shown as a gauge-dependent comparison.
* **Holonomy map** (`render_holonomy.png`): max |T_loop − 1| for 0.4×0.4 loops of the full conformal connection (coframe + Lorentz), ≈ ε²·curvature, from 1e-4 at r=8 to O(1) at the centre.
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
  shown as tone-mapped blackbody colour over a star field. Polarization: numga's model, emission polarized across a
  toroidal field in the local frame (gas aberration left out), as a unit-intensity coherency `Screen <- ScreenSlice` on
  the transported screen, read by a rotating analyzer slice.
* **Schwarzschild** (`render_lensing.png`, `.gif`; 480×360, camera at r = 50 M, 20° above the disk, field 0.7 rad):
  shadow angular radius 0.1018 rad against the analytic 0.1020 (static observer, sin α = (3√3 M/D)√(1−2M/D)), 0.2% low
  (it was 0.1670 vs 0.1681 at r = 30, 160×120). Frequency ratio on the disk 0.49–1.35, observed temperature up to
  4580 K. The image was traced before the Hermite crossing: photon energy k·∂t and angular momentum k·∂φ change by at most
  2.1e-3 from camera to disk there (the straight-line crossing), invisible in the image. The GIF shades the same trace
  with a two-armed pattern riding the Kepler flow (no light-travel delay).
* **Kerr a = 0.9** (`render_kerr.png`, `render_polarizer.gif`; 480×360, 15° above the disk): D-shaped shadow flattened
  on the approaching side, disk down to 2.32 M, frequency ratio 0.18–1.41, observed temperature up to 11141 K;
  k·∂t and k·∂φ conserved from camera to disk to 4.6e-4 (Hermite crossing; 2.4e-2 with the straight line at
  320×240). Polarizer: crossed analyzers pass 1 to 1e-12; one analyzer 0.10–0.90; per disk pixel the brightest over
  darkest frame is 1.6 (median) to 3.7 (90th percentile), a band sweeping round the disk. Its exposure is dimmer
  (factor 0.6 instead of 2) to keep that swing out of the tone curve's roll-off.
* **Tests** (`tests/test_render.py`, 12×9 scenes): conservation (< 5e-3), screen orthonormal and across the ray
  (< 5e-3), frequency ratio against the conserved quantities (< 5e-3), unit-intensity coherency (1e-12), Page-Thorne.
* The first versions of the holonomy map and lensing image were wrong (as reported when they were made). Causes: (i) `jax.jit(jax.vmap(holonomy_loop))` returns wrong values for batches of more than about 16 points while `jax.vmap` without `jit`, `jax.lax.map` and single-point calls agree with each other (reproduced with the array and the extensor transport alike; the map now uses `lax.map`); I did not establish whether the fault is in numga, JAX or XLA. (ii) The camera lay in the disk plane, and the horizon speckle above. Later, at r = 30 with a 0.5 rad field the disk overflowed the frame; the camera moved to r = 50 with 0.7 rad.
* Not done: light-travel delay in the animation; multiple disk crossings (the disk is opaque, nearest crossing only);
  gas aberration in the polarization; the Schwarzschild image re-traced with the Hermite crossing.

## ParaView export of spacetime slices (`src/gauge_theory_gravity/export.py`)

Implemented, **not run**: the simulation itself (evaluating the fields on a 3D grid for each time step) has not been executed.
* Format: VTKHDF ImageData, one `.vtkhdf` file per time step plus a `<stem>.vtkhdf.series` JSON that ParaView reads as a time series. Needs `h5py` (writer) and, for the tests, `vtk`.
* Verified: the writer's layout, by reading files back with VTK's own `vtkHDFReader` (point ordering for scalars, 3-vectors and `uint8`, dimensions, origin and spacing), `tests/test_export.py`. ParaView itself was not opened.
* Written but not executed: the field evaluator and the slow test `-m slow` on a 3×3×3 grid (`pytest -m slow tests/test_export.py`).
* Geometries: `flyby` (Schwarzschild boosted to v=0.6, a vacuum solution; its Weyl²/2 is the boosted r⁻⁶ profile, Cotton and Ricci vanish) and `growing_mass` (PG form with M(t) from 1 to 2, a toy with matter, so the Cotton sector switches on while the mass changes).
* Fields per step: ½C² (normal connection), ½Riem² (Lorentz-only), R, magnitudes of the sectors M/P/K/D of the normal connection's curvature, the translation-sector shift vector, log10 versions of the invariants, and a validity mask.
* Cost, from a timing of 4000 points on CPU with `jit(vmap)`: about 0.65 ms/point, i.e. about 70 s per frame at n = 48 (110k points), about 20 min for 16 frames. Run: `PYTHONPATH=src python -m gauge_theory_gravity.export flyby --n 48 --frames 16`.
