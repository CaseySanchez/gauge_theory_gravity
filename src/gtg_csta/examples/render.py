"""Renderings of the Schwarzschild gauge field in the CSTA formulation.

From ./src:  python -m gtg_csta.examples.render [slices holonomy lensing ...]   (figures -> ../output)
Painleve-Gullstrand coordinates are regular across the horizon, so grids may run down to the centre.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import jax
import jax.numpy as jnp

from .. import algebra as al, cartan, conformal as cf, curvature as cv
from .. import invariants as iv, transport as tp

OUT = Path(__file__).resolve().parents[3] / "output"
OUT.mkdir(exist_ok=True)
M = 1.0
COF = cartan.painleve_gullstrand(M)
OMEGA = cartan.lorentz_connection(COF)


def _grid(half=6.0, n=121):
    u = np.linspace(-half, half, n)
    X, Y = np.meshgrid(u, u)
    return u, X, Y


def _eval_on_plane(fn, X, Y, z=0.37, t=0.0):
    """vmap fn(x4) over a slice at height z (offset so r never hits exactly 0)."""
    pts = jnp.stack([jnp.full(X.size, t), X.ravel(), Y.ravel(), jnp.full(X.size, z)], axis=-1)
    return np.asarray(jax.jit(jax.vmap(lambda k: fn(al.point(k))))(pts))


def _horizon(ax):
    ax.add_patch(plt.Circle((0, 0), 2 * M, fill=False, color="w", lw=1, ls="--"))


def slices(half=6.0, n=101):
    u, X, Y = _grid(half, n)
    I = lambda x: iv.quadratic_invariant(COF, OMEGA)(x).kernel.reshape(())
    sec_fn = lambda x: jnp.array([jnp.abs(al.sector(cv.curvature(OMEGA)(x), k).kernel).max() for k in "MPKD"])
    inv = _eval_on_plane(I, X, Y).reshape(X.shape)
    sec = _eval_on_plane(sec_fn, X, Y).reshape(*X.shape, 4)
    r = np.sqrt(X ** 2 + Y ** 2 + 0.37 ** 2)
    print("slices: max |I - 24 M^2/r^6| =", np.abs(inv - 24 * M ** 2 / r ** 6).max())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    im = ax[0].imshow(np.abs(inv) + 1e-12, extent=[-half, half, -half, half], origin="lower",
                      norm=LogNorm(1e-4, 1e3), cmap="inferno")
    ax[0].set_title("<F F~> = C^2/2  (gauge-invariant)"); fig.colorbar(im, ax=ax[0])
    shift = np.sqrt(2 * M / r)
    im = ax[1].imshow(sec[..., 0] + 1e-12, extent=[-half, half, -half, half], origin="lower",
                      norm=LogNorm(1e-3, 1e1), cmap="magma")
    ax[1].set_title("max |F_M| (frame-dependent component size)"); fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(shift, extent=[-half, half, -half, half], origin="lower", vmin=0, vmax=2, cmap="viridis")
    ax[2].set_title("translation sector: shift |beta| = sqrt(2M/r)\n(gauge dependent; =1 at horizon)")
    fig.colorbar(im, ax=ax[2])
    for a in ax:
        _horizon(a); a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle("Schwarzschild in CSTA, PG tetrad; slice z=0.37, horizon dashed (r=2M)")
    fig.tight_layout(); fig.savefig(OUT / "render_slices.png", dpi=130)
    # sectors P,K,D identically zero for the vacuum normal connection
    print("slices: max |F_P|,|F_K|,|F_D| =", [float(sec[..., i].max()) for i in (1, 2, 3)])


def holonomy_map(half=6.0, n=41, eps=0.4, steps=10):
    """||T_loop - 1|| for an eps-square loop in the (x,y) plane centred on each grid point."""
    u, X, Y = _grid(half, n)
    one = jnp.eye(1, 32)[0]
    Om = cf.connection(COF)                       # full 15-sector connection: Lorentz + translation (f=b=0)

    def dev(x):
        T = tp.holonomy_loop(Om, x - 0.5 * eps * jnp.array([0, 1.0, 1.0, 0]), 1, 2, eps, steps, "magnus4")
        return jnp.abs(T - one).max()

    pts = jnp.stack([jnp.zeros(X.size), X.ravel(), Y.ravel(), jnp.full(X.size, 0.37)], axis=-1)
    # lax.map, not jit(vmap): jit(vmap(holonomy)) returns wrong values for batches >~ 16 (see RESULTS.md)
    h = np.asarray(jax.jit(lambda P: jax.lax.map(dev, P))(pts)).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(h + 1e-14, extent=[-half, half, -half, half], origin="lower", norm=LogNorm(1e-4, 1e1), cmap="inferno")
    fig.colorbar(im, ax=ax, label="max |T_loop - 1|"); _horizon(ax)
    ax.set_title(f"holonomy of the full conformal connection\n{eps}x{eps} loops in (x,y)")
    fig.tight_layout(); fig.savefig(OUT / "render_holonomy.png", dpi=130)
    print("holonomy_map: min/max =", h.min(), h.max())


# ---------------- ray tracing ----------------

def _hamiltonian(x, p):
    g = cartan.metric(COF)(al.point(x)).kernel[0]
    return 0.5 * p @ jnp.linalg.solve(g, p)


_dH = jax.grad(_hamiltonian, argnums=(0, 1))


def _rhs(s):
    x, p = s
    dx, dp = _dH(x, p)
    return (dp, -dx)                                   # dx/dlam = dH/dp,  dp/dlam = -dH/dx


def _rk4(s, h):
    add = lambda a, b, c: jax.tree_util.tree_map(lambda u, v: u + c * v, a, b)
    k1 = _rhs(s); k2 = _rhs(add(s, k1, h / 2)); k3 = _rhs(add(s, k2, h / 2)); k4 = _rhs(add(s, k3, h))
    return jax.tree_util.tree_map(lambda u, a, b, c, d: u + h / 6 * (a + 2 * b + 2 * c + d), s, k1, k2, k3, k4)


def _static_camera(D, fov, nx, ny, incl=np.radians(20.0)):
    """Initial (x, p) for every pixel: a static observer at distance D, elevated by `incl` above the disk plane
    (z = 0), looking at the hole.  The ray tangent is the arriving photon's momentum k = u - n (traced
    backward with a negative step).  Returns also the image-plane half-width tan(fov/2)."""
    x0 = jnp.array([0.0, D * np.cos(incl), 0.0, D * np.sin(incl)])
    g = cartan.metric(COF)(al.point(x0)).kernel[0]
    u = jnp.array([1.0, 0, 0, 0]); u = u / jnp.sqrt(u @ g @ u)
    # spatial triad in coordinates: outward r, right = z x r, up = r x right; then orthonormalise wrt g, orthogonal to u
    rh = np.array(x0[1:]) / D
    right = np.cross([0, 0, 1.0], rh); right /= np.linalg.norm(right)
    up = np.cross(rh, right)
    def proj(v, basis):
        for b in basis:
            v = v - (v @ g @ b) / (b @ g @ b) * b
        return v
    es = []
    for d in (rh, right, up):
        v = proj(jnp.concatenate([jnp.zeros(1), jnp.asarray(d)]), [u] + es)
        es.append(v / jnp.sqrt(-(v @ g @ v)))
    ex, ey, ez = es                                    # outward, right, up
    a = np.tan(fov / 2)
    sx = np.linspace(-a, a, nx); sy = np.linspace(-a * ny / nx, a * ny / nx, ny)
    SX, SY = np.meshgrid(sx, sy)
    n = (-ex)[None] + SX.ravel()[:, None] * ey[None] + SY.ravel()[:, None] * ez[None]   # unnormalised direction
    nn = jnp.sqrt(-jnp.einsum("pa,ab,pb->p", n, g, n))
    n = n / nn[:, None]                                # unit spatial vector, looking direction
    k = u[None] - n                                    # arriving photon's tangent (future-directed, null)
    p = jnp.einsum("ab,pb->pa", g, k)
    return jnp.broadcast_to(x0, (p.shape[0], 4)), p, a


def _trace_scene(nx, ny, D, fov, steps, r_in, r_out, incl):
    """Backward-trace every pixel.  Returns a dict of per-pixel arrays: `escaped`, `sky` (unit direction the
    ray reaches at infinity), `hit` (crossed the disk), `hr`, `hphi` (radius and azimuth of the nearest disk
    crossing), and the image-plane half-width `a`."""
    x0, p0, a = _static_camera(D, fov, nx, ny, incl)
    h0 = 0.03

    def trace(x, p):
        def body(c, _):
            (xx, pp), hit, hr, hp, alive = c
            r = jnp.linalg.norm(xx[1:])
            h = -h0 * jnp.clip(r, 1.0, 10.0)            # backward in affine parameter
            xn, pn = _rk4((xx, pp), h)
            crossed = (xx[3] * xn[3] < 0) & alive & (~hit)
            mid = 0.5 * (xx[1:] + xn[1:])
            rc = jnp.linalg.norm(mid)
            on = crossed & (rc > r_in) & (rc < r_out)
            hit = hit | on
            hr = jnp.where(on, rc, hr)
            hp = jnp.where(on, jnp.arctan2(mid[1], mid[0]), hp)
            rn = jnp.linalg.norm(xn[1:])
            alive = alive & (rn > 2.2 * M) & (rn < 120.0)      # traced back to the horizon: emitted by nothing (black)
            xn = jnp.where(alive, xn, xx); pn = jnp.where(alive, pn, pp)
            return ((xn, pn), hit, hr, hp, alive), None
        c0 = ((x, p), False, 0.0, 0.0, True)
        ((xf, pf), hit, hr, hp, alive), _ = jax.lax.scan(body, c0, None, length=steps)
        return xf, hit, hr, hp

    xs, hit, hr, hp = map(np.asarray, jax.jit(jax.vmap(trace))(x0, p0))
    rf = np.linalg.norm(xs[:, 1:], axis=1)
    return dict(escaped=rf > 100.0, sky=xs[:, 1:] / np.maximum(rf, 1e-9)[:, None], hit=hit, hr=hr, hphi=hp, a=a)


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
    sc = _trace_scene(nx, ny, D, fov, steps, r_in, r_out, incl)
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
    sc = _trace_scene(nx, ny, D, fov, steps, r_in, r_out, incl)
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
