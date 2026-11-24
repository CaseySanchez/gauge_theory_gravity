"""Ray paths, their twistors and the accumulated conformal transport, exported to ParaView as VTKHDF PolyData.

    PYTHONPATH=src python -m gauge_theory_gravity.examples.rays --hole schwarzschild --nx 15 --ny 3

A ray is followed as in `render` (backward from a static camera), but its state is recorded every `every` steps, so
that each ray becomes a polyline through spacetime. The twistor is carried by a versor: the state is (x, U) with
Z = U Z0, the camera's twistor Z0 fixed and U in Spin(2,4) accumulating the transport, dU/dlambda = A U with
A = -1/2 Omega(dx/dlambda), stepped on the group by the fourth-order commutator-free Lie group method CF4 (Celledoni,
Marthinsen and Owren 2003): every update is a product of exponentials of so(2,4) bivectors, so U stays a versor
(U ~U = 1) and h(Z, Z) is kept by construction, up to round-off. The event x moves by the classical Runge-Kutta stages.

Each recorded point carries (point data):
  coordinates     x^mu, (t, x, y, z); the polyline's own points are three of them, the display axes
  lambda          the affine parameter (negative: traced backward from the camera)
  momentum        k, frame components (t, x, y, z), read from the twistor;  velocity  dx/dlambda = E(k), coordinates
  frequency       k . e_t in the frame, the frequency a frame observer measures;  redshift  frequency over its value
                  at the camera minus one, the redshift of light sent from there by that observer to the camera
  screen_x/_y     the camera screen carried by the twistor's flag (`render.screen_frame`): the polarization basis
  energy_drift, angular_momentum_drift   change of k . d_t and k . d_phi since the camera, over the camera energy
  null_residual   k . k (zero: the momentum of a twistor is null)
  omega_re/_im, pi_re/_im   the twistor's coordinates in `twistor.BASIS`;  amplitude  |omega^0|, |omega^1|, |pi_0|, |pi_1|
  twistor_norm    h(Z, Z);  nullity  h(Z, Z) / |Z|^2;  incidence  |omega| / |pi| (zero while Z passes through x)
  relative_phase  arg(pi_1 conj pi_0), with phase_defined 0 where either amplitude is below 1e-9 of |pi|
  versor          the 32 coefficients of U;  versor_residual  |U ~U - 1|
  frame_t ... frame_n   the transported basis vectors U e_a ~U of R^{2,4}, six components each
  transport_trace       tr(U) / 4 on twistor space (real, imaginary): 1 for the identity, a conjugation invariant
  relative_trace        tr(~U_0 U) / 4 against the first ray, U_0, at the same record: the relative transport
Cell data per ray: pixel, escaped, captured.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from numga import concatenate, where
from numga.extensor import Extensor

from .. import algebra as al, cartan, export, hypersurface as hs, twistor as tw
from . import render as R

Direction = al.Direction
State = tuple[Extensor, Extensor]
OUT = Path(__file__).resolve().parents[3] / "export" / "rays"


# --- the versor transport ------------------------------------------------------------------------
def versor_rates(hole: R.Hole, initial: Extensor, position: Extensor, versor: Extensor) -> State:
    """The coordinate velocity and the generator A = -1/2 Omega(velocity) of the versor, for Z = U Z0."""
    twistor = (versor * initial).cast(tw.Twistor)
    velocity, conformal = R.generator(hole, position, twistor)
    return velocity, (-0.5 * conformal).cast(al.Bivector)


def commutator_free(rates: Callable[..., State], state: State, size: Extensor, first: State | None = None) -> State:
    """One step of CF4 for (x, U): the event by the classical Runge-Kutta stages, the versor by products of
    exponentials of the stage generators, so that it never leaves the group.
        U2 = exp(h/2 A1) U,  U3 = exp(h/2 A2) U,  U4 = exp(h/2 (2 A3 - A1)) U2,
        U' = exp(h/12 (-A1 + 2 A2 + 2 A3 + 3 A4)) exp(h/12 (3 A1 + 2 A2 + 2 A3 - A4)) U."""
    position, versor = state
    v1, a1 = rates(*state) if first is None else first
    second = ((size / 2) * a1).exp() * versor
    v2, a2 = rates(position + (size / 2) * v1, second)
    v3, a3 = rates(position + (size / 2) * v2, ((size / 2) * a2).exp() * versor)
    v4, a4 = rates(position + size * v3, ((size / 2) * (2 * a3 - a1)).exp() * second)
    moved = position + (size / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
    early = ((size / 12) * (3 * a1 + 2 * a2 + 2 * a3 - a4)).exp()
    late = ((size / 12) * (-a1 + 2 * a2 + 2 * a3 + 3 * a4)).exp()
    return moved.cast(Direction), (late * early * versor).cast(tw.Twistor)


# --- recording paths -----------------------------------------------------------------------------
def trace_paths(hole: R.Hole, start: State, records: int, every: int, base_step: float = 0.03,
                adaptive: bool = True) -> dict:
    """Backward-trace rays from `start` (events and twistors), recording the state `records` times, every `every`
    steps, after the starting record. The step is -base_step times the radius clipped to [1, 10] (as in `render`) or,
    with `adaptive=False`, -base_step throughout. A ray stops (and repeats its last state) once it is captured
    (r < 1.1 r_+) or escapes (r > 120 M). Returns the camera twistors and, per record (leading axis, records + 1),
    the events, the versors, the affine parameter and whether the ray was still moving."""
    import jax

    capture = 1.1 * hole.horizon
    events, initial = start
    count = len(np.asarray(events.kernel))
    move = partial(versor_rates, hole, initial)

    def size_of(position: Extensor) -> Extensor:
        if adaptive:
            return -base_step * hole.radius(position).clip(1.0, 10.0)
        return -base_step + 0 * hole.radius(position)

    def step(carry, _):
        state, rate, alive, parameter = carry
        size = size_of(state[0])
        moved = commutator_free(move, state, size, rate)
        moved_rate = move(*moved)
        radius = hole.radius(moved[0])
        still = alive & (radius > capture) & (radius < 120.0)
        state = tuple(where(still, new, old) for new, old in zip(moved, state))
        rate = tuple(where(still, new, old) for new, old in zip(moved_rate, rate))
        parameter = parameter + where(still, size, 0 * size)
        return (state, rate, still, parameter), None

    def record(carry, _):
        carry = jax.lax.scan(step, carry, None, length=every)[0]
        return carry, carry

    @jax.jit
    def run(carry):
        return jax.lax.scan(record, carry, None, length=records)[1]

    state = (events, (0 * initial + 1).cast(tw.Twistor))
    carry = (state, move(*state), np.ones(count, dtype=bool), 0 * hole.radius(events))
    states, _, alive, parameter = run(carry)
    return dict(count=count, initial=initial,
                events=concatenate([state[0][None], states[0]]),
                versors=concatenate([state[1][None], states[1]]),
                alive=np.concatenate([np.ones((1, count), dtype=bool), np.asarray(alive)]),
                parameter=np.concatenate([np.asarray(carry[3].kernel)[None, :, 0],
                                          np.asarray(parameter.kernel)[..., 0]]))


def twistors_of(paths: dict) -> Extensor:
    """The twistor at every record, Z = U Z0, [records, rays]."""
    return (paths["versors"] * paths["initial"][None]).cast(tw.Twistor)


# --- observables ---------------------------------------------------------------------------------
def observables(hole: R.Hole, paths: dict) -> dict[str, np.ndarray]:
    """Every exported quantity, [records, rays, components...]."""
    records, rays = paths["alive"].shape
    events = paths["events"]
    twistors = twistors_of(paths)
    flat_events, flat_twistors = events.reshape(-1), twistors.reshape(-1)
    momentum = tw.momentum(flat_twistors)
    out = {"coordinates": np.asarray(al.coordinates(flat_events)),
           "lambda": paths["parameter"].reshape(-1),
           "momentum": np.asarray(al.coordinates(momentum)),
           "velocity": np.asarray(al.coordinates(hole.geometry.frame(flat_events)(momentum)))}
    frequency = out["momentum"][:, 0]
    out["frequency"] = frequency
    out["redshift"] = frequency / np.tile(frequency[:rays], records) - 1
    screen = R.screen_frame(flat_twistors)
    out["screen_x"] = np.asarray(al.coordinates(screen(al.mv.x.dual())))
    out["screen_y"] = np.asarray(al.coordinates(screen(al.mv.y.dual())))
    energy, angular = R.killing_momenta(hole, flat_events, momentum)
    reference = np.abs(np.tile(energy[:rays], records))
    out["energy_drift"] = (energy - np.tile(energy[:rays], records)) / reference
    out["angular_momentum_drift"] = (angular - np.tile(angular[:rays], records)) / reference
    out["null_residual"] = np.einsum("na,ab,nb->n", out["momentum"], al.ETA4, out["momentum"])
    out |= twistor_observables(flat_twistors)
    out |= versor_observables(paths["versors"])
    return {name: value.reshape(records, rays, *value.shape[1:]) for name, value in out.items()}


def twistor_observables(twistors: Extensor) -> dict[str, np.ndarray]:
    """The twistor's coordinates and the diagnostics of its state, per twistor."""
    values = tw.coordinates(twistors)                                                    # [n, 4] complex
    norm = tw.norm(twistors)
    size = np.sqrt((np.abs(values) ** 2).sum(-1))
    pi_size = np.sqrt((np.abs(values[:, 2:]) ** 2).sum(-1))
    amplitude = np.abs(values)
    defined = (amplitude[:, 2:] > 1e-9 * pi_size[:, None]).all(-1)
    phase = np.where(defined, np.angle(values[:, 3] * np.conj(values[:, 2])), 0.0)
    return {"omega_re": values[:, :2].real, "omega_im": values[:, :2].imag,
            "pi_re": values[:, 2:].real, "pi_im": values[:, 2:].imag,
            "amplitude": amplitude, "twistor_norm": norm, "nullity": norm / size ** 2,
            "incidence": np.sqrt((np.abs(values[:, :2]) ** 2).sum(-1)) / pi_size,
            "relative_phase": phase, "phase_defined": defined.astype(np.uint8)}


FRAME = (("t", al.t), ("x", al.x), ("y", al.y), ("z", al.z), ("p", al.p), ("n", al.n))


def versor_observables(versors: Extensor) -> dict[str, np.ndarray]:
    """The accumulated versor, [records, rays]: its coefficients, how far it is from the group, the transported
    basis of R^{2,4}, its twistor trace and its trace relative to the first ray's versor at the same record."""
    records, rays = versors.shape
    flat = versors.reshape(-1)
    out = {"versor": np.asarray(flat.kernel),
           "versor_residual": np.abs(np.asarray((flat * ~flat - 1).kernel)).max(axis=-1)}
    for name, basis in FRAME:
        out["frame_" + name] = np.asarray((flat >> basis).cast(al.Vector).kernel)
    trace = tw.as_complex(tw.trace(flat))
    out["transport_trace"] = np.stack([trace.real, trace.imag], axis=-1) / 4
    relative = (~versors[:, :1] * versors).reshape(-1)
    trace = tw.as_complex(tw.trace(relative))
    out["relative_trace"] = np.stack([trace.real, trace.imag], axis=-1) / 4
    return out


# --- export --------------------------------------------------------------------------------------
def write_paths(path: Path, hole: R.Hole, paths: dict, display: tuple[str, str, str] = ("x", "y", "z")) -> None:
    """The rays as polylines in the three display coordinates, every observable as point data. A ray's polyline
    stops at its last moving record."""
    data = observables(hole, paths)
    records, rays = paths["alive"].shape
    columns = [hs.axis_index(name) for name in display]
    # point index of (record, ray) when points are listed ray by ray
    order = np.arange(records * rays).reshape(records, rays).T.reshape(-1)
    lengths = [int(min(paths["alive"][:, ray].sum() + 1, records)) for ray in range(rays)]
    lines = [np.arange(ray * records, ray * records + lengths[ray]) for ray in range(rays)]
    point_data = {name: value.reshape(records * rays, *value.shape[2:])[order] for name, value in data.items()}
    points = point_data["coordinates"][:, columns]
    final = data["coordinates"][-1]
    radius = np.asarray(hole.radius(al.point(final)).kernel)[..., 0]
    escaped = radius > 100.0
    cell_data = {"pixel": np.arange(rays), "escaped": escaped.astype(np.uint8),
                 "captured": (~paths["alive"][-1] & ~escaped).astype(np.uint8)}
    metadata = hs.coordinate_metadata() | dict(
        hole=hole.name, mass=hole.mass, spin=hole.spin, transport="versor", display_axes=list(display),
        momentum_basis="frame", velocity_basis="coordinate", screen_basis="frame",
        twistor_basis="twistor.BASIS: omega^0, omega^1, pi_0, pi_1")
    export.write_vtkhdf_polydata(path, points, lines, point_data, cell_data, metadata)


def camera_rays(hole: R.Hole, nx: int, ny: int, distance: float, field_of_view: float, inclination: float) -> State:
    return R.static_camera(hole, distance, field_of_view, nx, ny, inclination)[0]


HOLES = {"schwarzschild": R.schwarzschild_hole, "kerr": R.kerr_hole}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hole", choices=list(HOLES), default="schwarzschild")
    ap.add_argument("--nx", type=int, default=15)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--distance", type=float, default=50.0)
    ap.add_argument("--fov", type=float, default=1.0, help="horizontal field of view, radians")
    ap.add_argument("--inclination", type=float, default=20.0, help="degrees above the disk plane")
    ap.add_argument("--records", type=int, default=300)
    ap.add_argument("--every", type=int, default=6, help="steps between records")
    ap.add_argument("--display", default="xyz", help="the three coordinates the polylines are drawn in, e.g. txz")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    hole = HOLES[a.hole]()
    start = camera_rays(hole, a.nx, a.ny, a.distance, a.fov, np.radians(a.inclination))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = trace_paths(hole, start, a.records, a.every)
    path = out / f"rays_{a.hole}.vtkhdf"
    write_paths(path, hole, paths, tuple(a.display))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
