"""Renderings of the Schwarzschild gauge field in the CSTA formulation.

From ./src:  python -m gtg_csta.examples.render [slices holonomy lensing lensing_gif]   (figures -> ../output)
Painleve-Gullstrand coordinates are regular across the horizon, so grids may run down to the centre.

The rays are followed with the gauge theory of gravity's own objects: a ray's momentum `k` is a vector of the
local frame, its position `x` an event, and the frame is carried by the Lorentz connection of the CSTA,

    dx/dlambda = E(k),        dk/dlambda = -1/2 [Omega(dx/dlambda), k],

i.e. the momentum is parallel transported along its own path (the geodesic equation in the frame).
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
from numga import where
from numga.extensor import Extensor

from .. import algebra as al, cartan, conformal as cf, export, transport as tp

OUT = Path(__file__).resolve().parents[3] / "output"
OUT.mkdir(exist_ok=True)
M = 1.0
Spacetime = al.Spacetime
COFRAME = cartan.schwarzschild(M)                                 # position -> (Spacetime <- Spacetime)
LORENTZ = cartan.lorentz_connection(COFRAME)                      # position -> (Bivector <- Spacetime)
State = tuple[Extensor, Extensor]                                 # (event, frame momentum)


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
    shift = np.linalg.norm(data["shift"], axis=-1).reshape(X.shape)
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
    """max |T_loop - 1| for a size x size loop in the (x,y) plane centred on each grid point: the full conformal
    connection's holonomy, a batch of loops through one compiled evaluation."""
    import jax

    _, X, Y = grid(half, n)
    centres = plane_points(X, Y)
    connection = cf.connection(COFRAME)
    identity = np.eye(1, 32)[0]

    @jax.jit
    def around(starts: Extensor) -> Extensor:
        return tp.holonomy_loop(connection, starts, 1, 2, size, steps, "magnus4")

    corners = centres - 0.5 * size * np.array([0.0, 1.0, 1.0, 0.0])               # loops centred on the points
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


# --- ray tracing ---------------------------------------------------------------------------------
def radius_of(position: Extensor) -> Extensor:
    """The Cartesian radius of a position, a scalar."""
    return (-cartan.SpacePart(position).squared()).square_root()


def dynamics(position: Extensor, momentum: Extensor) -> State:
    """The rates of a ray: the coordinate velocity is the momentum carried out of the frame, `E(k)`, and the frame
    momentum is parallel transported by the Lorentz connection along it, `-1/2 [Omega(velocity), k]`."""
    velocity = COFRAME(position).inverse()(momentum).cast(Spacetime)
    transport_rate = -LORENTZ(position)(velocity).commutator(momentum)
    return velocity, transport_rate.cast(Spacetime)


def runge_kutta(rates: Callable[..., State], state: State, size: Extensor) -> State:
    """One fourth-order step of a state, a tuple of extensors, along its rates."""
    first = rates(*state)
    second = rates(*(value + rate * (size / 2) for value, rate in zip(state, first)))
    third = rates(*(value + rate * (size / 2) for value, rate in zip(state, second)))
    fourth = rates(*(value + rate * size for value, rate in zip(state, third)))
    return tuple((value + (a + 2 * b + 2 * c + d) * (size / 6)).cast(Spacetime)
                 for value, a, b, c, d in zip(state, first, second, third, fourth))


def orthonormal(vector: Extensor, previous: list[Extensor]) -> Extensor:
    """The unit vector along `vector` once its parts along `previous` are removed (frame metric)."""
    for other in previous:
        vector = vector - ((vector | other) / (other | other)) * other
    return vector / (-(vector | vector)).square_root()


def static_camera(distance: float, field_of_view: float, nx: int, ny: int,
                  inclination: float = np.radians(20.0)) -> tuple[State, float]:
    """Initial state for every pixel: a static observer at `distance`, `inclination` above the disk plane (z = 0),
    looking at the hole. The ray's frame momentum is the arriving photon's, `k = u - n` for the unit direction n
    the pixel looks along, traced backward with a negative step. Also returns the image plane's half-width."""
    position = al.point([0.0, distance * np.cos(inclination), 0.0, distance * np.sin(inclination)])
    frame = COFRAME(position)
    observer = frame(al.t)
    observer = observer / (observer | observer).square_root()                  # the static observer's frame velocity
    # Spatial triad in coordinates (outward, right, up), carried into the frame and orthonormalised there.
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
    momentum = (observer - looking).cast(Spacetime)                             # [pixels] the photon's frame momentum
    events = (position + 0 * looking).cast(Spacetime)                           # [pixels] the camera's event, per pixel
    return (events, momentum), half


def trace_scene(nx: int, ny: int, distance: float, field_of_view: float, steps: int, r_in: float, r_out: float,
                inclination: float) -> dict:
    """Backward-trace every pixel. Returns per-pixel arrays: `escaped`, `sky` (the unit direction the ray reaches at
    infinity), `hit` (it crossed the disk), `hr` and `hphi` (radius and azimuth of the nearest crossing) and `a`, the
    image plane's half-width. A ray traced back to r < 2.2 M is classed black: in the PG chart a backward ray
    hugging the horizon continues onto an unphysical branch."""
    import jax

    start, half = static_camera(distance, field_of_view, nx, ny, inclination)
    pixels = nx * ny
    base_step = 0.03

    def step(carry, _):
        state, alive, hit, crossing = carry
        radius = radius_of(state[0])
        size = -base_step * radius.clip(1.0, 10.0)                              # backward in the affine parameter
        moved = runge_kutta(dynamics, state, size)
        crosses = (cartan.coordinate(state[0], 3) * cartan.coordinate(moved[0], 3) < 0) & alive & ~hit
        midpoint = (0.5 * (state[0] + moved[0])).cast(Spacetime)
        radius_there = radius_of(midpoint)
        on_disk = crosses & (radius_there > r_in) & (radius_there < r_out)
        crossing = where(on_disk, midpoint, crossing)
        new_radius = radius_of(moved[0])
        alive = alive & (new_radius > 2.2 * M) & (new_radius < 120.0)
        state = tuple(where(alive, new, old) for new, old in zip(moved, state))
        return (state, alive, hit | on_disk, crossing), None

    def run(carry):
        return jax.lax.scan(step, carry, None, length=steps)[0]

    carry = (start, np.ones(pixels, dtype=bool), np.zeros(pixels, dtype=bool), start[0])
    final, _, hit, crossing = jax.jit(run)(carry)
    final_position = np.asarray(final[0].kernel)
    crossed = np.asarray(crossing.kernel)
    radius_final = np.linalg.norm(final_position[:, 1:], axis=1)
    return dict(escaped=radius_final > 100.0, sky=final_position[:, 1:] / np.maximum(radius_final, 1e-9)[:, None],
                hit=np.asarray(hit), hr=np.linalg.norm(crossed[:, 1:], axis=1),
                hphi=np.arctan2(crossed[:, 2], crossed[:, 1]), a=half)


def _shade(sc, nx, ny, r_in, r_out, t=0.0, sky_turn=0.0):
    """Colour the traced scene at time t.  The disk turns differentially (Kepler, angular velocity r^-3/2) and
    carries a two-armed pattern; the sky may be turned about the z axis by `sky_turn`."""
    d = sc["sky"]
    theta = np.arccos(np.clip(d[:, 2], -1, 1)); phi = np.arctan2(d[:, 1], d[:, 0]) - sky_turn
    check = ((np.floor(theta / (np.pi / 12)) + np.floor(phi / (np.pi / 12))) % 2).astype(float)
    sky = np.stack([0.10 + 0.25 * check, 0.12 + 0.2 * check, 0.25 + 0.3 * check], -1)
    hr = sc["hr"]
    ramp = np.clip((r_out - hr) / (r_out - r_in), 0, 1)
    pattern = 0.65 + 0.35 * np.cos(2 * (sc["hphi"] - t * np.maximum(hr, 1.0) ** -1.5) + 0.6 * hr)
    disk = plt.cm.inferno(np.clip(ramp * pattern * 1.25, 0, 1))[:, :3]
    img = np.zeros((nx * ny, 3))
    img[sc["escaped"]] = sky[sc["escaped"]]
    img[sc["hit"]] = disk[sc["hit"]]                    # nearest disk crossing wins
    return img.reshape(ny, nx, 3)


def lensing(nx=160, ny=120, D=30.0, fov=0.5, steps=1800, r_in=6.0, r_out=14.0, incl=np.radians(20.0)):
    """Backward ray tracing of a thin accretion disk (z=0, r_in..r_out) and a checkered sky."""
    sc = trace_scene(nx, ny, D, fov, steps, r_in, r_out, incl)
    img = _shade(sc, nx, ny, r_in, r_out)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.imshow(img, origin="lower"); ax.axis("off")
    ax.set_title(f"Schwarzschild lensing from the PG tetrad (M=1, camera at r={D:g}, disk {r_in:g}-{r_out:g})")
    fig.tight_layout(); fig.savefig(OUT / "render_lensing.png", dpi=140)
    # validation: the shadow radius of a static observer, sin(alpha) = (3 sqrt3 M / D) sqrt(1 - 2M/D)
    esc, hit = sc["escaped"], sc["hit"]
    rho = np.sqrt((~esc).mean() * 4 * sc["a"] ** 2 * ny / nx / np.pi)                 # tan(alpha) from the area
    alpha = np.arcsin(3 * np.sqrt(3) * M / D * np.sqrt(1 - 2 * M / D))
    print(f"lensing: shadow angular radius {np.arctan(rho):.4f} rad, analytic {alpha:.4f} rad")
    print(f"lensing: escaped {esc.mean():.2f}, captured {(~esc & ~hit).mean():.2f}, disk {hit.mean():.2f}")


def lensing_gif(nx=160, ny=120, D=30.0, fov=0.5, steps=1800, r_in=6.0, r_out=14.0, incl=np.radians(20.0),
                frames=60, turns=1.0):
    """Animated GIF: the same traced image (one ray trace; Schwarzschild is static) with the disk turning
    differentially (Kepler), so that its two-armed pattern shears, in the direct and the lensed images alike.  `turns` is the number of inner-edge orbits in the loop."""
    from PIL import Image
    sc = trace_scene(nx, ny, D, fov, steps, r_in, r_out, incl)
    period = 2 * np.pi * r_in ** 1.5                      # orbital period at the inner edge
    imgs = []
    for k in range(frames):
        f = k / frames
        img = _shade(sc, nx, ny, r_in, r_out, t=turns * period * f)
        im = Image.fromarray((np.flipud(img) * 255).astype(np.uint8)).resize((nx * 4, ny * 4), Image.NEAREST)
        imgs.append(im)
    path = OUT / "render_lensing.gif"
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=50, loop=0, optimize=True)
    print("lensing_gif: wrote", path, f"({frames} frames)")


ALL = dict(slices=slices, holonomy=holonomy_map, lensing=lensing, lensing_gif=lensing_gif)

if __name__ == "__main__":
    for name in (sys.argv[1:] or [k for k in ALL if k != 'lensing_gif']):
        ALL[name]()
