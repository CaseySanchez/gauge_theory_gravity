"""Light for the renders: thin-disk temperature, blackbody colour, tone mapping and a starry sky.

Adapted from numga's Kerr example (`numga/examples/relativity/kerr/render.py` and `core.py`, MIT licence,
Copyright (c) 2022 Eelco Hoogendoorn): the Page-Thorne temperature, the Wyman-Sloan-Shirley colour-matching fit, the
blackbody integration and the tone mapping follow that code; plain numpy here, since none of it involves the geometry.
"""
from __future__ import annotations

import numpy as np

# Wavelengths across the visible band, nm, and the second radiation constant, nm K.
visible_wavelengths = np.linspace(380.0, 780.0, 81)
radiation_constant = 1.4388e7
# Linear sRGB from CIE XYZ, and the luminance of linear sRGB.
xyz_to_rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
luminance_weights = np.array([0.2126, 0.7152, 0.0722])
mid_grey = 0.5
# A milder lift than a display's gamma of 2.2, so brightness differences stay visible.
display_gamma = 1.6
# Blackbody colours look pale on a display; their saturation is exaggerated to show the shift.
colour_saturation = 2.5
# The white balance: a blackbody at this temperature shows neutral, hotter ones blue, cooler ones red.
white_temperature = 6500.0                                                     # K


# --- the thin disk ---------------------------------------------------------------------------------
def innermost_stable_orbit(spin: float, mass: float = 1.0) -> float:
    """The prograde innermost stable circular equatorial orbit, Bardeen, Press and Teukolsky (1972); 6M without spin."""
    ratio = spin / mass
    first = 1 + np.cbrt(1 - ratio ** 2) * (np.cbrt(1 + ratio) + np.cbrt(1 - ratio))
    second = np.sqrt(3 * ratio ** 2 + first ** 2)
    return mass * (3 + second - np.sqrt((3 - first) * (3 + first + 2 * second)))


def keplerian_speed(radius: np.ndarray, spin: float, mass: float = 1.0) -> np.ndarray:
    """The coordinate angular speed of a prograde circular equatorial orbit, sqrt(M) / (r^3/2 + a sqrt(M)).  It holds in
    Boyer-Lindquist, Kerr-Schild and Painleve-Gullstrand coordinates alike: on a circular orbit their t and phi differ only
    by functions of r."""
    return np.sqrt(mass) / (radius ** 1.5 + spin * np.sqrt(mass))


def disk_temperature(radius: np.ndarray, spin: float, scale: float, mass: float = 1.0) -> np.ndarray:
    """The Page and Thorne (1974) thin-disk temperature, zero at the innermost stable orbit, peaking just outside it:
    T = scale F^(1/4), with F the flux of the relativistic thin disk (prograde, units of the mass)."""
    ratio = spin / mass
    root_radius = np.sqrt(np.asarray(radius) / mass)
    edge = np.sqrt(innermost_stable_orbit(spin, mass) / mass)
    # The roots of x^3 - 3x + 2 ratio, and the weight of each in the torque (the cubic's derivative there is 3(x^2 - 1)).
    roots = 2 * np.cos((np.arccos(ratio) + np.array([-np.pi, np.pi, 3 * np.pi])) / 3)
    weights = (roots - ratio) ** 2 / (roots * (roots ** 2 - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        torque = (root_radius - edge - 1.5 * ratio * np.log(root_radius / edge)
                  - (weights * np.log((root_radius[..., None] - roots) / (edge - roots))).sum(axis=-1))
        flux = 3 / (8 * np.pi) * np.clip(torque, 0, None) / (root_radius ** 4 * (root_radius ** 3 - 3 * root_radius + 2 * ratio))
    inside = root_radius < edge                                               # no disk inside the innermost stable orbit
    return np.where(inside, 0.0, scale * np.nan_to_num(np.clip(flux, 0, None)) ** 0.25)


# --- colour ----------------------------------------------------------------------------------------
def colour_matching(wavelengths: np.ndarray) -> np.ndarray:
    """The Wyman, Sloan and Shirley (2013) multi-lobe fit of the CIE 1931 XYZ colour-matching functions ([..., 3])."""
    def lobe(centre: float, below: float, above: float) -> np.ndarray:
        width = np.where(wavelengths < centre, below, above)
        return np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)
    x = 1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2)
    y = 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
    z = 1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def blackbody(temperature: np.ndarray) -> np.ndarray:
    """Planck spectra integrated across the visible band into linear sRGB ([..., 3]); zero for zero temperature."""
    temperature = np.asarray(temperature, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        spectrum = 1 / (visible_wavelengths ** 5 * np.expm1(radiation_constant / (visible_wavelengths * temperature[..., None])))
    return np.clip(np.nan_to_num(spectrum) @ colour_matching(visible_wavelengths) @ xyz_to_rgb.T, 0, None)


def balanced_blackbody(temperature: np.ndarray) -> np.ndarray:
    """Blackbody light white-balanced so that `white_temperature` shows neutral, with its luminance kept."""
    white = blackbody(np.array(white_temperature))
    return blackbody(temperature) * (white @ luminance_weights) / white


def tone_mapped(light: np.ndarray, reference: float) -> np.ndarray:
    """Linear sRGB light mapped for display: light of luminance `reference` sits at mid-grey, brighter light rolls off
    exponentially, with a gamma lift and a saturation boost."""
    luminance = light @ luminance_weights
    exposure = luminance / reference
    mapped = np.where(exposure < 1, mid_grey * exposure, 1 - (1 - mid_grey) * np.exp(-(exposure - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(luminance > 0, mapped / luminance, 0)
    shown = (light * scale[..., None]) ** (1 / display_gamma)
    grey = shown @ luminance_weights
    vivid = np.clip(grey[..., None] + colour_saturation * (shown - grey[..., None]), 0, None)
    return vivid / np.maximum(vivid.max(axis=-1, keepdims=True), 1)


# --- the sky ---------------------------------------------------------------------------------------
def star_texture(n_width: int = 1024, n_height: int = 512, seed: int = 8) -> np.ndarray:
    """An equirectangular sky ([n_height, n_width, 3]): unresolved stars on a dark background with a faint band."""
    random = np.random.default_rng(seed)
    longitude = (np.arange(n_width) + 0.5) * (2 * np.pi / n_width) - np.pi
    latitude = (0.5 - (np.arange(n_height) + 0.5) / n_height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    direction = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    band_height = direction @ np.array([0.2, -0.6, -np.sqrt(0.6)])
    band = np.exp(-(band_height / 0.16) ** 2) * (0.8 + 0.2 * np.sin(13 * lon + 7 * lat))
    texture = np.array([0.006, 0.009, 0.019]) + band[..., None] * np.array([0.065, 0.058, 0.09])
    n_stars = n_width * n_height // 140
    star_x = random.uniform(0, n_width, n_stars)
    star_y = (0.5 - np.arcsin(random.uniform(-1, 1, n_stars)) / np.pi) * n_height
    brightness = 0.35 + 2.5 * random.random(n_stars) ** 6
    warmth = random.uniform(0, 1, (n_stars, 1))
    colours = np.array([0.57, 0.72, 1.0]) * (1 - warmth) + np.array([1.0, 0.79, 0.51]) * warmth
    offsets = np.arange(-2, 3)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    columns = np.floor(star_x).astype(int)[:, None] + offset_x.ravel()
    rows = np.floor(star_y).astype(int)[:, None] + offset_y.ravel()
    distance_squared = (columns + 0.5 - star_x[:, None]) ** 2 + (rows + 0.5 - star_y[:, None]) ** 2
    weights = np.exp(-distance_squared / (2 * 0.6 ** 2)) * brightness[:, None]
    valid = (rows >= 0) & (rows < n_height)
    np.add.at(texture, (rows[valid], columns[valid] % n_width), (weights[..., None] * colours[:, None])[valid])
    return np.clip(texture, 0, 1)


def sky_colour(direction: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """The texture at the longitude and latitude of each unit direction ([..., 3] Cartesian), nearest pixel."""
    n_height, n_width = texture.shape[:2]
    longitude = np.arctan2(direction[..., 1], direction[..., 0])
    latitude = np.arcsin(np.clip(direction[..., 2], -1, 1))
    column = np.floor((longitude / (2 * np.pi) + 0.5) * n_width).astype(int) % n_width
    row = np.clip(np.floor((0.5 - latitude / np.pi) * n_height).astype(int), 0, n_height - 1)
    return texture[row, column]
