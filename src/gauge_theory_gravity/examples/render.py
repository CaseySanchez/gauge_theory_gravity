"""Renderings of black holes in the CSTA formulation: Schwarzschild (Painleve-Gullstrand tetrad) and Kerr (Kerr-Schild
position gauge field).

From ./src:  python -m gauge_theory_gravity.examples.render [slices holonomy twistor schwarzschild kerr]   (figures -> ../output)
Painleve-Gullstrand coordinates are regular across the horizon, so grids may run down to the center.

The rays are followed with the gauge theory of gravity's own objects: a ray is its event `x` and the local twistor
`Z` of its light ray, carried by the normal conformal connection of the CSTA (Lorentz plus the coframe's translations;
the Schouten part is zero in these vacuum holes),

    dx/dlambda = E(k),        dZ/dlambda = -1/2 Omega(dx/dlambda) Z,        k = momentum(Z) = pi conj(pi),

Penrose's local twistor transport, which keeps the twistor incident with the ray (omega = 0); its Lorentz part turns pi
as parallel transport would, so the frame momentum `k` follows the geodesic equation in the frame, and stays exactly
null.  The phase of pi turns its null flag about `k`: the camera sets each flag along the screen's x, and the screen, a
map from screen directions to frame directions, is read back from the flag wherever it is needed, as the parallel
transported screen of numga's Kerr example (up to multiples of `k`).  At the disk the gas orbits on prograde Kepler
orbits and glows with the Page-Thorne temperature; the ratio of
the photon's frequency at the camera to that in the gas, `(k | u_camera) / (k | u_gas)`, shifts the blackbody color,
which carries gravitational redshift and Doppler beaming at once.  The polarization model is numga's Kerr example's:
synchrotron-like emission polarized across a toroidal magnetic field, set up in the local frame (aberration by the gas
is left out), read on the screen as a coherency map and passed through a rotating analyzer.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from dataclasses import dataclass
from functools import partial

from numga import Algebra, concatenate, where
from numga.extensor import Extensor

from .. import algebra as al, cartan, conformal as cf, export, transport as tp, twistor as tw
from . import light

OUT = Path(__file__).resolve().parents[3] / "output"
OUT.mkdir(exist_ok=True)
M = 1.0
Direction = al.Direction
COFRAME = cartan.schwarzschild(M)                                 # position -> (Direction <- Direction)
# The camera's slices x and y, which read screen directions, and the screen directions themselves.
ScreenSlice = al.Slice.derive.within(Algebra("x-y-"))            # Vector+
Screen = ScreenSlice.derive.dual                                  # Antivector-
ScreenFrame = al.ALG.gatype((Direction, Screen))                  # Direction- <- Screen-
# The light's polarization covariance, from the slices that read the screen to screen directions.
Coherency = al.ALG.gatype((Screen, ScreenSlice))                  # Screen- <- ScreenSlice+
State = tuple[Extensor, Extensor]                                 # (event, local twistor)


# --- field-strength slice and holonomy map ---------------------------------------------------------
def grid(half: float = 6.0, n: int = 121) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-half, half, n)
    X, Y = np.meshgrid(axis, axis)
    return axis, X, Y


def plane_points(X: np.ndarray, Y: np.ndarray, height: float = 0.37, time: float = 0.0) -> np.ndarray:
    """Spacetime points (N, 4) on a horizontal plane, offset from z = 0 so that r never vanishes."""
    return np.stack([np.full(X.size, time), X.ravel(), Y.ravel(), np.full(X.size, height)], axis=-1)


def horizon(ax) -> None:
    ax.add_patch(plt.Circle((0, 0), 2 * M, fill=False, color="w", lw=1, ls="--"))


def slices(half: float = 6.0, n: int = 101) -> None:
    """The gauge-invariant density, the Lorentz-sector curvature size and the translation-sector shift on a plane."""
    _, X, Y = grid(half, n)
    height = 0.37
    data = export.evaluate_points(COFRAME, plane_points(X, Y, height))
    density = data["riem2_half"].reshape(X.shape)
    curvature_size = data["F_M"].reshape(X.shape)
    shift = np.linalg.norm(data["shift"][:, 1:], axis=-1).reshape(X.shape)                 # e^i_0, the spatial part
    r = np.sqrt(X ** 2 + Y ** 2 + height ** 2)
    print("slices: max |I - 24 M^2/r^6| =", np.abs(density - 24 * M ** 2 / r ** 6).max())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    extent = [-half, half, -half, half]
    image = ax[0].imshow(np.abs(density) + 1e-12, extent=extent, origin="lower", norm=LogNorm(1e-4, 1e3), cmap="inferno")
    ax[0].set_title("<F F~> = C^2/2  (gauge-invariant)"); fig.colorbar(image, ax=ax[0])
    image = ax[1].imshow(curvature_size + 1e-12, extent=extent, origin="lower", norm=LogNorm(1e-3, 1e1), cmap="magma")
    ax[1].set_title("|F_M| (frame-dependent component size)"); fig.colorbar(image, ax=ax[1])
    image = ax[2].imshow(shift, extent=extent, origin="lower", vmin=0, vmax=2, cmap="viridis")
    ax[2].set_title("translation sector: shift |beta| = sqrt(2M/r)\n(gauge dependent; =1 at horizon)")
    fig.colorbar(image, ax=ax[2])
    for a in ax:
        horizon(a); a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle("Schwarzschild in CSTA, PG tetrad; slice z=0.37, horizon dashed (r=2M)")
    fig.tight_layout(); fig.savefig(OUT / "render_slices.png", dpi=130)
    print("slices: max |F_P|, |F_K|, |F_D| =", [float(data[k].max()) for k in ("F_P", "F_K", "F_D")])


def holonomy_map(half: float = 6.0, n: int = 41, size: float = 0.4, steps: int = 10, chunk: int = 256) -> None:
    """max |T_loop - 1| for a size x size loop in the (x,y) plane centered on each grid point: the full conformal
    connection's holonomy, a batch of loops through one compiled evaluation."""
    import jax

    _, X, Y = grid(half, n)
    centers = plane_points(X, Y)
    connection = cf.connection(COFRAME)
    identity = np.eye(1, 32)[0]

    @jax.jit
    def around(starts: Extensor) -> Extensor:
        return tp.holonomy_loop(connection, starts, 1, 2, size, steps, "magnus4")

    corners = centers - 0.5 * size * np.array([0.0, 1.0, 1.0, 0.0])               # loops centered on the points
    deviation = []
    for start in range(0, len(corners), chunk):
        block = corners[start:start + chunk]
        padding = chunk - len(block)
        block = np.concatenate([block, np.repeat(block[-1:], padding, axis=0)]) if padding else block
        kernel = np.asarray(around(al.point(block[:, None, :])).kernel)
        deviation.append(np.abs(kernel - identity).max(axis=-1)[:chunk - padding])
    holonomy = np.concatenate(deviation).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(holonomy + 1e-14, extent=[-half, half, -half, half], origin="lower",
                      norm=LogNorm(1e-4, 1e1), cmap="inferno")
    fig.colorbar(image, ax=ax, label="max |T_loop - 1|"); horizon(ax)
    ax.set_title(f"holonomy of the full conformal connection\n{size}x{size} loops in (x,y)")
    fig.tight_layout(); fig.savefig(OUT / "render_holonomy.png", dpi=130)
    print("holonomy_map: min/max =", holonomy.min(), holonomy.max())


def twistor_holonomy(half: float = 5.0, n: int = 40, size: float = 0.4, steps: int = 10, chunk: int = 256,
                     spin: float = 0.9) -> None:
    """Twistor holonomy of a Kerr hole on its meridian plane (y = 0, x and z), `render_twistor.png`. Each size x size
    loop in the (x, z) plane, centered on a grid point, transports twistors by an SU(2,2) matrix; its eigenvalues are
    gauge invariant (a gauge transformation conjugates the matrix). Panels: how far they are from 1; the boost fraction
    of the loop's curvature, the angle of the dominant eigenvalue's logarithm from the imaginary axis (0 for a pure
    rotation, 90 degrees for a pure boost); and, for comparison, the phase of the local complex Weyl invariant
    `twistor.weyl_invariant`, 6 arctan(a cos theta / r), from which a nonzero boost fraction in the meridian plane
    comes."""
    import jax

    hole = kerr_hole(M, spin)
    axis = np.linspace(-half, half, n)                                          # n even: the grid avoids z = 0
    X, Z = np.meshgrid(axis, axis)
    centers = np.stack([np.zeros(X.size), X.ravel(), np.zeros(X.size), Z.ravel()], axis=-1)
    connection = cf.connection(hole.geometry)                                   # the normal connection in vacuum

    @jax.jit
    def around(starts: Extensor) -> Extensor:
        return tp.holonomy_loop(connection, starts, 1, 3, size, steps, "magnus4")

    corners = centers - 0.5 * size * np.array([0.0, 1.0, 0.0, 1.0])
    matrices = []
    for start in range(0, len(corners), chunk):
        block = corners[start:start + chunk]
        padding = chunk - len(block)
        block = np.concatenate([block, np.repeat(block[-1:], padding, axis=0)]) if padding else block
        matrices.append(tw.matrix(around(al.point(block[:, None, :])))[:chunk - padding])
    holonomy = np.concatenate(matrices)                                         # [points, 4, 4] SU(2,2)
    finite = np.isfinite(holonomy).all(axis=(1, 2))
    eigenvalues = np.linalg.eigvals(np.where(finite[:, None, None], holonomy, np.eye(4)))
    logarithms = np.log(eigenvalues)
    dominant = logarithms[np.arange(len(logarithms)), np.abs(logarithms).argmax(axis=-1)]
    deviation = np.where(finite, np.abs(eigenvalues - 1).max(axis=-1), np.nan).reshape(X.shape)
    boost = np.where(finite, np.degrees(np.arctan2(np.abs(dominant.real), np.abs(dominant.imag))), np.nan).reshape(X.shape)
    radius = np.asarray(kerr_radius(al.point(centers), M, spin).kernel)[..., 0]
    phase = np.degrees(6 * np.arctan2(spin * centers[:, 3] / radius, radius)).reshape(X.shape)
    phase = (phase + 180) % 360 - 180
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    extent = [-half, half, -half, half]
    image = ax[0].imshow(deviation + 1e-14, extent=extent, origin="lower", norm=LogNorm(1e-4, 1e0), cmap="inferno")
    fig.colorbar(image, ax=ax[0], label="max |eigenvalue - 1|")
    ax[0].set_title(f"twistor holonomy, {size}x{size} loops in (x, z)")
    image = ax[1].imshow(boost, extent=extent, origin="lower", vmin=0, vmax=90, cmap="viridis")
    fig.colorbar(image, ax=ax[1], label="degrees")
    ax[1].set_title("boost fraction of the loop's curvature\n(0 rotation, 90 boost)")
    image = ax[2].imshow(phase, extent=extent, origin="lower", vmin=-180, vmax=180, cmap="twilight_shifted")
    fig.colorbar(image, ax=ax[2], label="degrees")
    ax[2].set_title("phase of the complex Weyl invariant\n6 arctan(a cos theta / r)")
    for a in ax:
        a.add_patch(plt.Circle((0, 0), hole.horizon, fill=False, color="w", lw=1, ls="--"))
        a.plot([-spin, spin], [0, 0], "wo", ms=3)                               # the ring singularity
        a.set_xlabel("x"); a.set_ylabel("z")
    fig.suptitle(f"{hole.name}: twistor holonomy on the meridian plane y = 0 (horizon dashed, ring dots)")
    fig.tight_layout(); fig.savefig(OUT / "render_twistor.png", dpi=130); plt.close(fig)
    unitary = np.abs(np.einsum("pji,jk,pkl->pil", holonomy[finite].conj(), HERMITIAN, holonomy[finite]) - HERMITIAN).max()
    print(f"twistor_holonomy: max |eigenvalue - 1| {np.nanmax(deviation):.3g}, boost fraction "
          f"{np.nanmin(boost):.1f}..{np.nanmax(boost):.1f} deg, max |H^+ h H - h| {unitary:.1e}, "
          f"{(~finite).sum()} loops not finite")


HERMITIAN = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])   # the twistor norm in coordinates


# --- the holes -----------------------------------------------------------------------------------
def kerr_radius(position: Extensor, mass: float, spin: float) -> Extensor:
    """The oblate radius of an event in the Kerr-Schild geometry."""
    return cartan.kerr_schild_fields(position, mass, spin)[0]


@dataclass(frozen=True)
class Hole:
    """A black hole to render: its geometry (coframe and frame), Lorentz connection, mass, spin, and the radius the
    disk and the stopping rules are measured in."""
    name: str
    geometry: cartan.Geometry
    mass: float
    spin: float
    radius: Callable[[Extensor], Extensor]

    @property
    def lorentz(self) -> Callable[[Extensor], Extensor]:
        return cartan.lorentz_connection(self.geometry)

    @property
    def horizon(self) -> float:
        return self.mass + np.sqrt(self.mass ** 2 - self.spin ** 2)


def schwarzschild_hole(mass: float = M) -> Hole:
    return Hole("Schwarzschild, Painleve-Gullstrand tetrad", cartan.schwarzschild(mass), mass, 0.0, cartan.spatial_radius)


def kerr_hole(mass: float = M, spin: float = 0.9) -> Hole:
    return Hole(f"Kerr a={spin:g}, Kerr-Schild gauge field", cartan.kerr(mass, spin), mass, spin,
                partial(kerr_radius, mass=mass, spin=spin))


# --- ray tracing ---------------------------------------------------------------------------------
def rates(hole: Hole, position: Extensor, twistor: Extensor) -> State:
    """The rates of a ray: the coordinate velocity is the twistor's momentum carried out of the frame, `E(k)`, and the
    twistor moves by the normal conformal connection, `-1/2 Omega(velocity) Z`: the Lorentz connection along the
    velocity, which turns pi (its momentum and flag) as parallel transport would, plus the translation part, the
    coframe's image of the velocity, `e(E(k)) = k`, as the bivector `k ^ n_inf`, which keeps omega = 0."""
    velocity, conformal = generator(hole, position, twistor)
    return velocity, (-0.5 * conformal * twistor).cast(tw.Twistor)


def generator(hole: Hole, position: Extensor, twistor: Extensor) -> tuple[Extensor, Extensor]:
    """The coordinate velocity `E(k)` of a ray and the normal conformal connection along it, `Omega(velocity)`, the
    bivector of so(2,4) whose -1/2 multiple moves the twistor (`rates`) or the accumulated versor (`rays`)."""
    momentum = tw.momentum(twistor)                                                     # [...] Direction-
    velocity = hole.geometry.frame(position)(momentum).cast(Direction)                  # [...] Direction-
    turning = hole.lorentz(position)(velocity)                                          # [...] Bivector+
    conformal = turning + (momentum & al.INFINITY).dual().cast(al.Bivector)             # [...] Bivector+
    return velocity, conformal


def runge_kutta(rates: Callable[..., State], state: State, size: Extensor, first: State | None = None) -> State:
    """One fourth-order step of a state, a tuple of extensors, along its rates; each keeps its own type. `first`, the
    rates at the start, may be passed in when already known (the previous step's end)."""
    # The step size multiplies from the left: numga contracts a right-hand scalar with the product table first, which
    # for an even multivector (the twistor) builds a batch of 32 x 32 matrices that XLA's GPU compiler takes tens of
    # minutes over; from the left the contraction is with the multivector first and compiles at once.
    first = rates(*state) if first is None else first
    second = rates(*(value + (size / 2) * rate for value, rate in zip(state, first)))
    third = rates(*(value + (size / 2) * rate for value, rate in zip(state, second)))
    fourth = rates(*(value + size * rate for value, rate in zip(state, third)))
    return tuple((value + (size / 6) * (a + 2 * b + 2 * c + d)).cast(value.gatype)
                 for value, a, b, c, d in zip(state, first, second, third, fourth))


def hermite(start: State, end: State, start_rate: State, end_rate: State, size: Extensor,
            fraction: Extensor) -> State:
    """The cubic Hermite interpolant of a step at `fraction` of the way (0..1), from the states and rates at both ends:
    fourth-order accurate, where a straight line between the ends is second-order."""
    f2, f3 = fraction * fraction, fraction * fraction * fraction
    weights = (2 * f3 - 3 * f2 + 1, (f3 - 2 * f2 + fraction) * size, 3 * f2 - 2 * f3, (f3 - f2) * size)
    return tuple((weights[0] * a + weights[1] * da + weights[2] * b + weights[3] * db).cast(a.gatype)
                 for a, b, da, db in zip(start, end, start_rate, end_rate))


def hermite_slope(start: Extensor, end: Extensor, start_rate: Extensor, end_rate: Extensor, size: Extensor,
                  fraction: Extensor) -> Extensor:
    """The derivative with respect to `fraction` of the Hermite interpolant of one component."""
    f2 = fraction * fraction
    return ((6 * f2 - 6 * fraction) * start + (3 * f2 - 4 * fraction + 1) * size * start_rate
            + (6 * fraction - 6 * f2) * end + (3 * f2 - 2 * fraction) * size * end_rate)


def plane_crossing(start: State, end: State, start_rate: State, end_rate: State, size: Extensor) -> State:
    """The state where the step crosses the plane z = 0: the straight-line guess for the fraction, refined by two
    Newton steps on the Hermite interpolant's height, then the interpolated state there."""
    height = cartan.coordinate(start[0], 3)
    fraction = height / (height - cartan.coordinate(end[0], 3))
    for _ in range(2):
        position = hermite(start[:1], end[:1], start_rate[:1], end_rate[:1], size, fraction)[0]
        slope = hermite_slope(start[0], end[0], start_rate[0], end_rate[0], size, fraction)
        fraction = fraction - cartan.coordinate(position, 3) / cartan.coordinate(slope.cast(Direction), 3)
    return hermite(start, end, start_rate, end_rate, size, fraction)


def orthonormal(vector: Extensor, previous: list[Extensor]) -> Extensor:
    """The unit spacelike direction along `vector` once its parts along `previous` are removed (frame metric)."""
    for other in previous:
        vector = vector - ((vector | other) / (other | other)) * other
    return vector / (-(vector | vector)).square_root()


def static_camera(hole: Hole, distance: float, field_of_view: float, nx: int, ny: int,
                  inclination: float = np.radians(20.0)) -> tuple[State, float]:
    """Initial state for every pixel: a static observer at `distance`, `inclination` above the disk plane (z = 0),
    looking at the hole. The ray's local twistor passes through the camera's event (omega = 0) with the arriving
    photon's frame momentum, `k = u - n` for the unit direction n the pixel looks along, so that `k | u = 1`, traced
    backward with a negative step; its phase sets its flag along the camera's right, made orthogonal to the pixel's ray,
    so that `screen_frame` reads the screen's x and y as the camera's right and up. Also returns the image plane's
    half-width."""
    events, momentum, right_axis, up_axis, half = camera_view(hole, distance, field_of_view, nx, ny, inclination)
    twistor = tw.from_momentum(momentum)                                        # [pixels] the ray's local twistor
    # A phase turns the flag by twice itself about k; turn each one from where it lies onto the camera's right.
    across = tw.flag_direction(twistor, al.time)
    angle = np.arctan2(-np.asarray((across | up_axis).kernel)[..., 0], -np.asarray((across | right_axis).kernel)[..., 0])
    return (events, tw.with_phase(twistor, -angle / 2)), half


def camera_view(hole: Hole, distance: float, field_of_view: float, nx: int, ny: int,
                inclination: float = np.radians(20.0)) -> tuple[Extensor, Extensor, Extensor, Extensor, float]:
    """Each pixel's camera event, frame momentum `k = u - n` and the camera's right and up made orthogonal to the
    pixel's ray (unit Directions, screen x and y), with the image plane's half-width; see `static_camera`."""
    position = al.point([0.0, distance * np.cos(inclination), 0.0, distance * np.sin(inclination)])
    frame = hole.geometry(position)
    observer = frame(al.time)
    observer = observer / (observer | observer).square_root()                  # the static observer's frame velocity
    # Spatial triad in coordinates (outward, right, up), carried into the frame and orthonormalized there.
    outward = np.array([np.cos(inclination), 0.0, np.sin(inclination)])
    right = np.cross([0.0, 0.0, 1.0], outward)
    right /= np.linalg.norm(right)
    up = np.cross(outward, right)
    triad = []
    for direction in (outward, right, up):
        triad.append(orthonormal(frame(al.point([0.0, *direction])), [observer, *triad]))
    towards, across, upward = triad
    half = np.tan(field_of_view / 2)
    sx = np.linspace(-half, half, nx)
    sy = np.linspace(-half * ny / nx, half * ny / nx, ny)
    SX, SY = np.meshgrid(sx, sy)
    looking = -towards + al.mv.scalar(SX.reshape(-1, 1)) * across + al.mv.scalar(SY.reshape(-1, 1)) * upward
    looking = looking / (-(looking | looking)).square_root()                    # [pixels] unit spatial direction
    momentum = (observer - looking).cast(Direction)                             # [pixels] the photon's frame momentum
    right_axis = orthonormal(across, [observer, looking])                       # [pixels] Direction-
    up_axis = orthonormal(upward, [observer, looking, right_axis])              # [pixels] Direction-
    events = (position + 0 * looking).cast(Direction)                           # [pixels] the camera's event, per pixel
    return events, momentum, right_axis, up_axis, half


def screen_frame(twistor: Extensor) -> Extensor:
    """The camera screen carried along with a ray's twistor: screen x along its flag, screen y along the flag of
    e^{I pi/4} Z (a quarter turn on), each the unit direction across the ray orthogonal to the frame's time direction.
    Lorentz transport turns flag and screen alike and the camera set the flag on screen x, so this is the parallel
    transported screen up to multiples of the momentum; the choice orthogonal to the frame's time direction makes it
    read a frame direction by its part across the ray in the frame's rest space.  Direction- <- Screen-."""
    right = tw.flag_direction(twistor, al.time)                                 # [...] Direction-
    up = tw.flag_direction(tw.with_phase(twistor, np.pi / 4), al.time)          # [...] Direction-
    # Screen x and y are read by the slices -x and -y (x & x.dual() = -1 in this signature).
    return (right * (-al.x & Screen) + up * (-al.y & Screen)).cast(ScreenFrame)


def readout(state: State) -> tuple[Extensor, Extensor, Extensor]:
    """A ray's event, frame momentum and screen, the last two read from its twistor."""
    position, twistor = state
    return position, tw.momentum(twistor), screen_frame(twistor)


def trace_scene(hole: Hole, nx: int, ny: int, distance: float, field_of_view: float, steps: int, r_in: float,
                r_out: float, inclination: float, chunk: int = 16384, base_step: float = 0.03) -> dict:
    """Backward-trace every pixel until it escapes (r > 100 M) or is captured (r < 1.1 r_+: in these
    horizon-penetrating charts a backward ray hugging the horizon continues onto an unphysical branch), recording its
    first crossing of the disk (z = 0, r_in < r < r_out), the one nearest the camera; the disk is opaque for shading,
    but the ray goes on, so that `escaped` and captured describe the shadow by themselves. Returns the camera state, the
    final state, the state at the crossing (cubic Hermite interpolation to z = 0) and per-pixel flags `escaped` and `hit`, with
    `a` the image plane's half-width. Pixels go through one compiled trace in chunks of `chunk` rays (the last one
    padded), which bounds the memory of the scan."""
    import jax

    start, half = static_camera(hole, distance, field_of_view, nx, ny, inclination)
    pixels = nx * ny
    capture = 1.1 * hole.horizon
    move = partial(rates, hole)

    def step(carry, _):
        state, rate, alive, hit, crossing = carry
        size = -base_step * hole.radius(state[0]).clip(1.0, 10.0)               # backward in the affine parameter
        moved = runge_kutta(move, state, size, rate)
        moved_rate = move(*moved)                                               # the next step's first stage
        height, new_height = cartan.coordinate(state[0], 3), cartan.coordinate(moved[0], 3)
        at_plane = plane_crossing(state, moved, rate, moved_rate, size)
        radius_there = hole.radius(at_plane[0])
        on_disk = (height * new_height < 0) & alive & ~hit & (radius_there > r_in) & (radius_there < r_out)
        crossing = tuple(where(on_disk, new, old) for new, old in zip(at_plane, crossing))
        new_radius = hole.radius(moved[0])
        alive = alive & (new_radius > capture) & (new_radius < 120.0)
        state = tuple(where(alive, new, old) for new, old in zip(moved, state))
        rate = tuple(where(alive, new, old) for new, old in zip(moved_rate, rate))
        return (state, rate, alive, hit | on_disk, crossing), None

    def run(carry):
        return jax.lax.scan(step, carry, None, length=steps)[0]

    compiled = jax.jit(run)
    parts = []
    for begin in range(0, pixels, chunk):
        count = min(chunk, pixels - begin)
        rays = np.minimum(np.arange(begin, begin + chunk), pixels - 1)               # the last chunk repeats its last ray
        part = tuple(value[rays] for value in start)
        final, _, _, hit, crossing = compiled((part, move(*part), np.ones(chunk, dtype=bool),
                                               np.zeros(chunk, dtype=bool), part))
        parts.append((tuple(value[:count] for value in final), np.asarray(hit)[:count],
                      tuple(value[:count] for value in crossing)))
        print(f"trace: {begin + count}/{pixels} rays", flush=True)
    final = tuple(concatenate([part[0][k] for part in parts]) for k in range(len(start)))
    hit = np.concatenate([part[1] for part in parts])
    crossing = tuple(concatenate([part[2][k] for part in parts]) for k in range(len(start)))
    radius_final = np.asarray(hole.radius(final[0]).kernel)[..., 0]
    return dict(start=start, final=final, crossing=crossing, hit=np.asarray(hit), escaped=radius_final > 100.0,
                a=half, shape=(ny, nx))


def sky_directions(hole: Hole, final: State) -> np.ndarray:
    """The unit direction each escaped ray came from: minus the spatial part of its coordinate velocity at the end."""
    velocity = np.asarray(al.coordinates(hole.geometry.frame(final[0])(tw.momentum(final[1]))))[..., 1:]
    return -velocity / np.linalg.norm(velocity, axis=-1, keepdims=True)


# --- the disk --------------------------------------------------------------------------------------
def azimuth(position: Extensor) -> Extensor:
    """d_phi at an event, the counter-clockwise turning direction about z: the position's commutator with the xy plane."""
    return (-cartan.SpacePart(position).commutator(al.x ^ al.y)).cast(Direction)        # [...] Direction-


def gas_velocity(hole: Hole, position: Extensor) -> Extensor:
    """The frame velocity of disk gas on a prograde circular orbit: d_t + Omega d_phi, carried into the frame and made
    unit. (Prograde is counter-clockwise: the Kerr gauge field drags frames that way for positive spin.)"""
    radius = hole.radius(position)                                                       # [...] Scalar+
    speed = np.sqrt(hole.mass) / (radius * radius.square_root() + hole.spin * np.sqrt(hole.mass))
    local = hole.geometry(position)(al.time + speed * azimuth(position))                 # [...] Direction-
    return local / (local | local).square_root()                                         # [...] Direction-


def frequency_ratio(hole: Hole, position: Extensor, momentum: Extensor) -> Extensor:
    """Observed over emitted frequency: the camera's `k | u = 1` over the gas's `k | u_gas`."""
    return 1 / (momentum | gas_velocity(hole, position))                                 # [...] Scalar+


def killing_momenta(hole: Hole, position: Extensor, momentum: Extensor) -> tuple[np.ndarray, np.ndarray]:
    """The photon's energy k . d_t and angular momentum k . d_phi, both conserved along a ray in these stationary,
    axisymmetric geometries: the frame momentum paired with the Killing directions carried into the frame."""
    frame = hole.geometry(position)
    energy = momentum | frame(al.time)
    angular = momentum | frame(azimuth(position))
    return np.asarray(energy.kernel)[..., 0], np.asarray(angular.kernel)[..., 0]


def disk_coherency(position: Extensor, momentum: Extensor, screen: Extensor, degree: float) -> Extensor:
    """Disk emission as a unit-intensity polarization covariance on the screen (numga's Kerr model): the
    toroidal field circles z; emission is polarized across it and across the ray, in the meridian plane, by `degree`
    times how far the ray is from looking along the field.  Screen- <- ScreenSlice+."""
    direction = cartan.SpacePart(momentum) / (al.t & momentum)                           # [...] unit spatial heading
    meridian = cartan.SpacePart(position).dual_inverse() ^ al.z                          # [...] Bivector+, radial and normal
    axis = (meridian.commutator(direction) / meridian.norm()).cast(Direction)            # [...] Direction-
    polarization = screen.adjoint()(axis)                                                # [...] Screen-
    fraction = -degree * polarization.squared()                                          # screen directions square to -1
    # Unpolarized light is round, -ScreenSlice.dual() (the screen's metric is negative); the polarized part weights the
    # polarization by each slice's reading of it.
    return (1 - fraction) / 2 * -ScreenSlice.dual() + degree * polarization * (ScreenSlice & polarization)


def transmitted(coherency: Extensor, analyzers: Extensor) -> Extensor:
    """The power each unit analyzer slice passes, read by pairing across the coherency."""
    return analyzers & coherency(analyzers)                                              # [...] Scalar+


def analyzer_slices(angles: np.ndarray) -> Extensor:
    """Unit analyzer slices across the screen, turned from the screen's x by `angles`."""
    turns = ((al.x ^ al.y) * (-angles / 2)).exp()                                         # [n] Rotor+
    return (turns >> -al.x).cast(ScreenSlice)                                            # [n] ScreenSlice+


@dataclass
class Disk:
    """What the camera sees of the disk, per pixel that hits it: radius, azimuth, frequency ratio, observed temperature."""
    pixels: np.ndarray
    radius: np.ndarray
    angle: np.ndarray
    ratio: np.ndarray
    temperature: np.ndarray


def disk_view(hole: Hole, scene: dict, temperature_scale: float) -> Disk:
    """Evaluate the disk physics at every crossing: redshift from the twistor's momentum, Page-Thorne temperature."""
    hit = scene["hit"]
    position, twistor = scene["crossing"]
    momentum = tw.momentum(twistor)
    radius = np.asarray(hole.radius(position).kernel)[..., 0]
    coordinates = np.asarray(al.coordinates(position))
    ratio = np.asarray(frequency_ratio(hole, position, momentum).kernel)[..., 0]
    emitted = light.disk_temperature(radius, hole.spin, temperature_scale, hole.mass)
    return Disk(np.flatnonzero(hit), radius[hit], np.arctan2(coordinates[hit, 2], coordinates[hit, 1]),
                ratio[hit], (ratio * emitted)[hit])


# --- shading -------------------------------------------------------------------------------------
TEMPERATURE_SCALE = 58500.0                     # K, the disk's temperature scale (numga's Kerr example)
# Exposure: the disk's 90th-percentile luminance is shown at twice mid-gray exposure, whatever the hole's temperatures.
EXPOSURE_PERCENTILE, EXPOSURE_FACTOR = 90.0, 2.0
# Behind the analyzer the light is doubled and swings with its turn; a dimmer exposure keeps the swing out of the
# tone curve's roll-off, where brightness differences flatten: the 90th percentile passed at 0.9 lands near the knee.
POLARIZER_EXPOSURE_FACTOR = 0.6


def shade(scene: dict, disk: Disk, sky: np.ndarray, emissivity: np.ndarray | None = None,
          gain: np.ndarray | float = 1.0, exposure: float = EXPOSURE_FACTOR) -> np.ndarray:
    """Display colors [ny, nx, 3]: the disk's observed blackbody light (times an emissivity pattern and a gain, such as
    an analyzer's transmission) over the sky seen by the escaped rays."""
    ny, nx = scene["shape"]
    light_ = np.zeros((nx * ny, 3))
    glow = light.balanced_blackbody(disk.temperature)
    if emissivity is not None:
        glow = glow * emissivity[:, None]
    reference = np.percentile(glow @ light.luminance_weights, EXPOSURE_PERCENTILE) / exposure
    light_[disk.pixels] = glow * np.reshape(gain, (-1, 1))
    image = light.tone_mapped(light_, reference)
    # The sky shows where a ray escaped without meeting the disk (rays go on past the disk, for the shadow's sake).
    clear = scene["escaped"] & ~scene["hit"]
    image[clear] += sky[clear]
    return np.clip(image, 0, 1).reshape(ny, nx, 3)


def rings(radius: np.ndarray) -> np.ndarray:
    """A faint ring texture on the disk, so its lensed images can be followed."""
    return 0.90 + 0.045 * np.sin(11 * radius) + 0.025 * np.sin(19 * radius)


def sky_light(hole: Hole, scene: dict) -> np.ndarray:
    return light.sky_color(sky_directions(hole, scene["final"]), light.star_texture())


def twistor_incidence(scene: dict) -> float:
    """How well local twistor transport kept each disk ray's twistor on its ray: the largest |omega| / |pi| at the
    disk, zero while the twistor passes through the ray's current event."""
    coordinates = tw.coordinates(scene["crossing"][1][scene["hit"]])
    return float((np.linalg.norm(coordinates[:, :2], axis=-1) / np.linalg.norm(coordinates[:, 2:], axis=-1)).max())


def conservation_error(hole: Hole, scene: dict) -> float:
    """Largest change of the photon's energy and angular momentum between camera and disk, relative to the energy."""
    camera = killing_momenta(hole, scene["start"][0], tw.momentum(scene["start"][1]))
    disk = killing_momenta(hole, scene["crossing"][0], tw.momentum(scene["crossing"][1]))
    hit = scene["hit"]
    return float(max(np.abs(disk[k][hit] - camera[k][hit]).max() for k in (0, 1)) / np.abs(camera[0]).max())


def camera_figure(hole: Hole, scene: dict, disk: Disk, sky: np.ndarray, title: str, path: Path) -> None:
    """The camera image beside the map of the frequency ratio observed/emitted on the disk."""
    ny, nx = scene["shape"]
    ratio = np.full(nx * ny, np.nan)
    ratio[disk.pixels] = disk.ratio
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="black")
    ax[0].imshow(shade(scene, disk, sky, rings(disk.radius)), origin="lower", interpolation="none")
    spread = max(abs(np.nanmax(ratio) - 1), abs(1 - np.nanmin(ratio)))
    image = ax[1].imshow(ratio.reshape(ny, nx), origin="lower", cmap="coolwarm_r", vmin=1 - spread, vmax=1 + spread,
                         interpolation="none")
    bar = fig.colorbar(image, ax=ax[1], fraction=0.04)
    bar.set_label("frequency observed / emitted", color="w"); bar.ax.tick_params(colors="w")
    for a in ax:
        a.axis("off")
    ax[0].set_title(title, color="w", fontsize=10)
    ax[1].set_title("gravitational redshift and Doppler shift of the disk light", color="w", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor="black"); plt.close(fig)


def disk_scene(hole: Hole, nx: int, ny: int, distance: float, fov: float, steps: int, r_out: float,
               inclination: float, r_in: float | None = None) -> tuple[dict, Disk, np.ndarray, float]:
    """Trace and evaluate a disk from the innermost stable orbit (unless `r_in` is given) out to `r_out`."""
    inner = light.innermost_stable_orbit(hole.spin, hole.mass) if r_in is None else r_in
    scene = trace_scene(hole, nx, ny, distance, fov, steps, inner, r_out, inclination)
    return scene, disk_view(hole, scene, TEMPERATURE_SCALE), sky_light(hole, scene), inner


def schwarzschild(nx=480, ny=360, D=50.0, fov=1.0, steps=1800, r_out=14.0, incl=np.radians(20.0), frames=60, turns=1.0):
    """Backward ray tracing of a thin Page-Thorne accretion disk around Schwarzschild, from the innermost stable orbit
    (6M) to r_out, colored by its observed blackbody temperature, and a starry sky: `render_schwarzschild.png`, and from the
    same trace `render_schwarzschild.gif`, with a two-armed emissivity pattern riding the Kepler flow so that it shears in the
    direct and the lensed images alike (`turns` inner-edge orbits per loop; Schwarzschild is static, one trace serves)."""
    hole = schwarzschild_hole()
    scene, disk, sky, inner = disk_scene(hole, nx, ny, D, fov, steps, r_out, incl)
    camera_figure(hole, scene, disk, sky,
                  f"{hole.name}: camera at r={D:g}, {np.degrees(incl):.0f} deg above the disk ({inner:g}-{r_out:g} M)",
                  OUT / "render_schwarzschild.png")
    # validation: the shadow radius of a static observer, sin(alpha) = (3 sqrt3 M / D) sqrt(1 - 2M/D)
    rho = np.sqrt((~scene["escaped"]).mean() * 4 * scene["a"] ** 2 * ny / nx / np.pi)   # tan(alpha) from the area
    alpha = np.arcsin(3 * np.sqrt(3) * M / D * np.sqrt(1 - 2 * M / D))
    print(f"schwarzschild: shadow angular radius {np.arctan(rho):.4f} rad, analytic {alpha:.4f} rad")
    report(hole, scene, disk)
    period = 2 * np.pi / light.keplerian_speed(inner, hole.spin, hole.mass)
    speed = light.keplerian_speed(disk.radius, hole.spin, hole.mass)
    frames_ = []
    for k in range(frames):
        time = turns * period * k / frames
        frames_.append(shade(scene, disk, sky, rings(disk.radius)
                             * (0.6 + 0.4 * np.cos(2 * (disk.angle - speed * time) + 0.6 * disk.radius))))
    save_gif(frames_, OUT / "render_schwarzschild.gif", 50)


def report(hole: Hole, scene: dict, disk: Disk) -> None:
    esc, hit = scene["escaped"], scene["hit"]
    print(f"{hole.name}: escaped {esc.mean():.2f}, captured {(~esc).mean():.2f}, disk {hit.mean():.2f}")
    print(f"{hole.name}: frequency ratio on the disk {disk.ratio.min():.3f} .. {disk.ratio.max():.3f};"
          f" observed temperature up to {disk.temperature.max():.0f} K")
    print(f"{hole.name}: conservation of k.d_t and k.d_phi from camera to disk, max relative change "
          f"{conservation_error(hole, scene):.1e}")
    print(f"{hole.name}: local twistor at the disk, max |omega|/|pi| {twistor_incidence(scene):.1e}")


def save_gif(frames: list[np.ndarray], path: Path, duration: int, width: int = 960) -> None:
    """Frames as a looping GIF, enlarged by whole pixels to about `width`."""
    from PIL import Image
    ny, nx = frames[0].shape[:2]
    zoom = max(1, width // nx)
    images = [Image.fromarray((np.flipud(frame) * 255).astype(np.uint8)).resize((nx * zoom, ny * zoom), Image.NEAREST)
              for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=True)
    print("wrote", path, f"({len(frames)} frames)")


def kerr(nx=480, ny=360, D=50.0, fov=1.0, steps=1800, r_out=14.0, incl=np.radians(15.0), spin=0.9,
         degree=0.8, frames=48):
    """The same camera and disk around a Kerr hole, `render_kerr.png`: frame dragging flattens the approaching side of
    the shadow, and the disk reaches in to the prograde innermost stable orbit.  From the same trace,
    `render_kerr.gif`: the disk through a linear analyzer turning half a revolution; the polarization, carried to the
    camera on the screen carried by each ray's twistor, sweeps a dark band across the image. Disk light is doubled behind the analyzer, so
    unpolarized parts keep their brightness."""
    hole = kerr_hole(M, spin)
    scene, disk, sky, inner = disk_scene(hole, nx, ny, D, fov, steps, r_out, incl)
    camera_figure(hole, scene, disk, sky,
                  f"{hole.name}: camera at r={D:g}, {np.degrees(incl):.0f} deg above the disk ({inner:.2f}-{r_out:g} M)",
                  OUT / "render_kerr.png")
    report(hole, scene, disk)
    hit = scene["hit"]
    coherency = disk_coherency(*readout(scene["crossing"]), degree)                             # [pixels] Screen- <- ScreenSlice+
    angles = np.linspace(0, np.pi, frames, endpoint=False)
    passed = np.asarray(transmitted(coherency, analyzer_slices(angles)[:, None]).kernel)[..., 0]   # [frames, pixels]
    # Unit intensity: two crossed analyzers pass all the light between them (frames is even, so frame k + frames/2 is
    # turned a quarter turn from frame k).
    total = passed[: frames // 2] + passed[frames // 2:]
    print(f"polarizer: crossed analyzers pass {total[:, hit].min():.12f} .. {total[:, hit].max():.12f} of the light;"
          f" one analyzer {passed[:, hit].min():.3f} .. {passed[:, hit].max():.3f}")
    save_gif([shade(scene, disk, sky, rings(disk.radius), 2 * share[disk.pixels], POLARIZER_EXPOSURE_FACTOR)
              for share in passed],
             OUT / "render_kerr.gif", 60)


ALL = dict(slices=slices, holonomy=holonomy_map, twistor=twistor_holonomy, schwarzschild=schwarzschild, kerr=kerr)

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
