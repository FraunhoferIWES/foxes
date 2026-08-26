from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from scipy.special import i0e
from scipy.interpolate import RegularGridInterpolator

from foxes.utils.xarray_utils import write_nc


AXIS_R_OVER_SIGMA = "r_over_sigma"
AXIS_SIGMA_OVER_D = "sigma_over_d"
DATA_WEIGHT = "weight"
LOOKUP_VERSION = "v1"
_MAX_EXPANSION_STEPS = 256


def create_lookup_axes(
    r_over_sigma_max: float = 28.0,
    n_r: int = 201,
    sigma_over_d_min: float = 0.02,
    sigma_over_d_max: float = 20.0,
    n_sigma: int = 161,
    sigma_spacing: Literal["linear", "log"] = "log",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create normalized lookup axes for Gaussian rotor-disc weights.

    The returned axes are dimensionless and use:

    - ``r_over_sigma = R / sigma`` for wake-centre offset over Gaussian width,
    - ``sigma_over_d = sigma / D`` for Gaussian width over rotor diameter.

    Parameters
    ----------
    r_over_sigma_max
        Upper bound of the radial-offset axis. Lower bound is always 0.
    n_r
        Number of points along the ``r_over_sigma`` axis.
    sigma_over_d_min
        Lower bound of the ``sigma_over_d`` axis.
    sigma_over_d_max
        Upper bound of the ``sigma_over_d`` axis.
    n_sigma
        Number of points along the ``sigma_over_d`` axis.
    sigma_spacing
        Spacing strategy for ``sigma_over_d``. Use ``"log"`` to resolve
        narrow wakes with higher density near the lower bound.

    Returns
    -------
    r_over_sigma
        Monotonic non-negative axis values.
    sigma_over_d
        Monotonic positive axis values.
    """
    if r_over_sigma_max <= 0.0:
        raise ValueError("r_over_sigma_max must be > 0")
    if n_r < 2:
        raise ValueError("n_r must be >= 2")
    if sigma_over_d_min <= 0.0:
        raise ValueError("sigma_over_d_min must be > 0")
    if sigma_over_d_max <= sigma_over_d_min:
        raise ValueError("sigma_over_d_max must be > sigma_over_d_min")
    if n_sigma < 2:
        raise ValueError("n_sigma must be >= 2")

    r_axis = np.linspace(0.0, r_over_sigma_max, n_r)
    if sigma_spacing == "linear":
        s_axis = np.linspace(sigma_over_d_min, sigma_over_d_max, n_sigma)
    elif sigma_spacing == "log":
        s_axis = np.geomspace(sigma_over_d_min, sigma_over_d_max, n_sigma)
    else:
        raise ValueError("sigma_spacing must be 'linear' or 'log'")

    return r_axis, s_axis


def gaussian_disc_weight(
    r_over_sigma: np.ndarray,
    sigma_over_d: np.ndarray,
    n_rho: int = 512,
) -> np.ndarray:
    r"""
    Compute rotor-disc averaged Gaussian weights on normalized axes.

    The wake profile is modeled as:

    .. math::

       g(\mathbf{u}) = \exp\left(-\frac{\|\mathbf{u} - \mathbf{u}_c\|^2}{2\sigma^2}\right)

    where normalized coordinates are used in units of rotor diameter.

    Parameters
    ----------
    r_over_sigma
        1D axis of normalized wake-centre offsets ``R / sigma``.
    sigma_over_d
        1D axis of normalized Gaussian widths ``sigma / D``.
    n_rho
        Number of midpoint samples for radial quadrature in the closed-form
        angularly integrated expression.

    Returns
    -------
    weights
        2D array of shape ``(len(r_over_sigma), len(sigma_over_d))`` containing
        disc-averaged Gaussian factors.
    """
    r_axis = np.asarray(r_over_sigma, dtype=float)
    s_axis = np.asarray(sigma_over_d, dtype=float)

    if r_axis.ndim != 1:
        raise ValueError("r_over_sigma must be a 1D array")
    if s_axis.ndim != 1:
        raise ValueError("sigma_over_d must be a 1D array")
    if np.any(r_axis < 0.0):
        raise ValueError("r_over_sigma values must be >= 0")
    if np.any(s_axis <= 0.0):
        raise ValueError("sigma_over_d values must be > 0")
    if n_rho < 16:
        raise ValueError("n_rho must be >= 16")

    a = 0.5  # rotor radius in D-normalized coordinates
    dr = a / n_rho
    rho = (np.arange(n_rho, dtype=float) + 0.5) * dr

    rho2 = rho[None, :] ** 2
    weights = np.empty((r_axis.size, s_axis.size), dtype=float)

    for j, s_val in enumerate(s_axis):
        rr = (r_axis * s_val)[:, None]
        s2 = s_val * s_val
        inv2s2 = 0.5 / s2
        x = (rr * rho[None, :]) / s2

        # Stable form: exp(-(rho^2+r^2)/(2*s^2)) * i0(x)
        # = exp(-((rho-r)^2)/(2*s^2)) * i0e(x), where x >= 0 here.
        integrand = (
            np.exp(-((rho[None, :] - rr) ** 2) * inv2s2)
            * i0e(x)
            * rho[None, :]
        )
        integral = np.sum(integrand, axis=1) * dr
        weights[:, j] = (2.0 / (a * a)) * integral

    return weights


def build_lookup_dataset(
    r_over_sigma: np.ndarray,
    sigma_over_d: np.ndarray,
    n_rho: int = 512,
    *,
    version_tag: str = LOOKUP_VERSION,
    sigma_spacing: Literal["linear", "log"] | None = None,
) -> xr.Dataset:
    """
    Build an in-memory lookup dataset for Gaussian partial-wake geometry.

    Parameters
    ----------
    r_over_sigma
        1D axis of normalized wake-centre offsets ``R / sigma``.
    sigma_over_d
        1D axis of normalized Gaussian widths ``sigma / D``.
    n_rho
        Radial quadrature resolution for weight generation.
    version_tag
        Version tag of the lookup artifact schema.
    sigma_spacing
        Optional spacing descriptor used for ``sigma_over_d``.

    Returns
    -------
    ds
        Dataset with coordinates ``r_over_sigma`` and ``sigma_over_d`` and data
        variable ``weight``.
    """
    w = gaussian_disc_weight(
        r_over_sigma=r_over_sigma,
        sigma_over_d=sigma_over_d,
        n_rho=n_rho,
    )

    ds = xr.Dataset(
        data_vars={
            DATA_WEIGHT: ((AXIS_R_OVER_SIGMA, AXIS_SIGMA_OVER_D), w),
        },
        coords={
            AXIS_R_OVER_SIGMA: np.asarray(r_over_sigma, dtype=float),
            AXIS_SIGMA_OVER_D: np.asarray(sigma_over_d, dtype=float),
        },
        attrs={
            "description": "Gaussian rotor-disc lookup weights on normalized geometry axes",
            "normalization": "r_over_sigma=R/sigma, sigma_over_d=sigma/D",
            "version_tag": version_tag,
            "generator": "foxes.utils.gaussian_lookup.build_lookup_dataset",
            "n_rho": int(n_rho),
        },
    )
    if sigma_spacing is not None:
        ds.attrs["sigma_spacing"] = sigma_spacing

    return ds


def generate_lookup_dataset(
    min_weight: float = 1.0e-8,
    r_over_sigma_max: float | None = None,
    sigma_over_d_min: float = 0.02,
    sigma_over_d_max: float = 20.0,
    radial_resolution: float = 0.1,
    sigma_resolution: float = 0.05,
    sigma_spacing: Literal["linear", "log"] = "log",
    n_rho: int = 512,
    version_tag: str = LOOKUP_VERSION,
    radial_expand_factor: float = 1.2,
) -> xr.Dataset:
    """
    Deterministically generate a Gaussian lookup dataset from axis settings.

    Parameters
    ----------
    min_weight
        Lower weight threshold for the lookup table. It determines the lower
        retained weights and the radial extent where lookup contributions
        become negligible when ``r_over_sigma_max`` is omitted.
    r_over_sigma_max
        Upper bound of the ``r_over_sigma`` axis. If omitted, a conservative
        bound is derived from ``min_weight`` and ``sigma_over_d_min``.
    sigma_over_d_min
        Lower bound of the ``sigma_over_d`` axis.
    sigma_over_d_max
        Upper bound of the ``sigma_over_d`` axis. Sigma values above that
        bound use the large-sigma asymptote when evaluated with
        ``bounds_policy="clip"``.
    radial_resolution
        Approximate spacing between interpolation points along the
        ``r_over_sigma`` axis.
    sigma_resolution
        Approximate spacing between interpolation points along the
        ``sigma_over_d`` axis.
    sigma_spacing
        Spacing strategy for ``sigma_over_d``.
    n_rho
        Radial quadrature resolution for weight generation.
    version_tag
        Version tag of the lookup artifact schema.
    radial_expand_factor
        Multiplicative growth factor for ``r_over_sigma_max`` during
        auto-expansion.

    Returns
    -------
    ds
        Lookup dataset with generation metadata.
    """
    if min_weight <= 0.0:
        raise ValueError("min_weight must be > 0")
    if sigma_over_d_min <= 0.0:
        raise ValueError("sigma_over_d_min must be > 0")
    if sigma_over_d_max <= sigma_over_d_min:
        raise ValueError("sigma_over_d_max must be > sigma_over_d_min")
    if r_over_sigma_max is not None and r_over_sigma_max <= 0.0:
        raise ValueError("r_over_sigma_max must be > 0")
    if radial_resolution <= 0.0:
        raise ValueError("radial_resolution must be > 0")
    if sigma_resolution <= 0.0:
        raise ValueError("sigma_resolution must be > 0")
    if radial_expand_factor <= 1.0:
        raise ValueError("radial_expand_factor must be > 1")

    auto_expand_radial = r_over_sigma_max is None
    if auto_expand_radial:
        r_curr = 0.5 / float(sigma_over_d_min) + float(
            np.sqrt(-2.0 * np.log(min_weight))
        )
    else:
        r_curr = float(r_over_sigma_max)
    n_steps = 0
    while True:
        r_axis, s_axis = create_lookup_axes(
            r_over_sigma_max=r_curr,
            n_r=max(2, int(np.ceil(r_curr / radial_resolution)) + 1),
            sigma_over_d_min=sigma_over_d_min,
            sigma_over_d_max=sigma_over_d_max,
            n_sigma=max(
                2,
                int(np.ceil((sigma_over_d_max - sigma_over_d_min) / sigma_resolution))
                + 1,
            ),
            sigma_spacing=sigma_spacing,
        )
        ds = build_lookup_dataset(
            r_over_sigma=r_axis,
            sigma_over_d=s_axis,
            n_rho=n_rho,
            version_tag=version_tag,
            sigma_spacing=sigma_spacing,
        )

        wtab = ds[DATA_WEIGHT].to_numpy()
        edge_r_max = float(np.max(wtab[-1, :]))
        edge_max = edge_r_max
        radial_ready = edge_r_max <= min_weight
        if radial_ready or not auto_expand_radial:
            ds.attrs["auto_min_weight"] = float(min_weight)
            ds.attrs["auto_edge_weight_max"] = edge_max
            ds.attrs["auto_edge_weight_r_max"] = edge_r_max
            ds.attrs["auto_expand_steps"] = int(n_steps)
            break

        if n_steps >= _MAX_EXPANSION_STEPS:
            raise ValueError(
                "Failed to auto-expand lookup ranges to min_weight target: "
            f"edge_weight_max={edge_max} > min_weight={min_weight} after "
            f"{_MAX_EXPANSION_STEPS} steps"
            )

        if auto_expand_radial and not radial_ready:
            r_curr *= float(radial_expand_factor)
        n_steps += 1

    ds.attrs["axis_r_over_sigma_max"] = float(r_curr)
    ds.attrs["axis_sigma_over_d_min"] = float(sigma_over_d_min)
    ds.attrs["axis_sigma_over_d_max"] = float(sigma_over_d_max)
    ds.attrs["axis_n_r"] = int(r_axis.size)
    ds.attrs["axis_n_sigma"] = int(s_axis.size)
    ds.attrs["radial_resolution"] = float(radial_resolution)
    ds.attrs["sigma_resolution"] = float(sigma_resolution)
    return ds


def save_lookup_dataset(
    ds: xr.Dataset,
    fpath: str | Path,
    *,
    complevel: int = 5,
    nc_engine: str | None = None,
) -> None:
    """
    Persist a Gaussian lookup dataset to NetCDF.

    Parameters
    ----------
    ds
        Lookup dataset.
    fpath
        Output path for the NetCDF artifact.
    complevel
        Compression level passed to NetCDF encoding.
    nc_engine
        NetCDF backend engine.
    """
    validate_lookup_dataset(ds)
    write_nc(
        ds,
        fpath,
        complevel=complevel,
        nc_engine=nc_engine,
        pack=True,
        verbosity=0,
    )


def load_lookup_dataset(fpath: str | Path) -> xr.Dataset:
    """
    Load a Gaussian lookup dataset from NetCDF and validate schema.

    Parameters
    ----------
    fpath
        Input path to a lookup NetCDF artifact.

    Returns
    -------
    ds
        Validated in-memory lookup dataset.
    """
    with xr.open_dataset(fpath) as ds_in:
        ds = ds_in.load()
    validate_lookup_dataset(ds)
    return ds


def validate_lookup_dataset(ds: xr.Dataset) -> None:
    """
    Validate that a dataset matches the Gaussian lookup schema.

    Parameters
    ----------
    ds
        Candidate lookup dataset.
    """
    if AXIS_R_OVER_SIGMA not in ds.coords:
        raise ValueError(f"Dataset missing coordinate '{AXIS_R_OVER_SIGMA}'")
    if AXIS_SIGMA_OVER_D not in ds.coords:
        raise ValueError(f"Dataset missing coordinate '{AXIS_SIGMA_OVER_D}'")
    if DATA_WEIGHT not in ds.data_vars:
        raise ValueError(f"Dataset missing variable '{DATA_WEIGHT}'")

    w = np.asarray(ds[DATA_WEIGHT].to_numpy(), dtype=float)
    if w.ndim != 2:
        raise ValueError(f"'{DATA_WEIGHT}' must be 2D")

    r_axis = np.asarray(ds.coords[AXIS_R_OVER_SIGMA].to_numpy(), dtype=float)
    s_axis = np.asarray(ds.coords[AXIS_SIGMA_OVER_D].to_numpy(), dtype=float)
    if r_axis.ndim != 1:
        raise ValueError(f"'{AXIS_R_OVER_SIGMA}' must be 1D")
    if s_axis.ndim != 1:
        raise ValueError(f"'{AXIS_SIGMA_OVER_D}' must be 1D")
    if np.any(np.diff(r_axis) <= 0.0):
        raise ValueError(f"'{AXIS_R_OVER_SIGMA}' must be strictly increasing")
    if np.any(np.diff(s_axis) <= 0.0):
        raise ValueError(f"'{AXIS_SIGMA_OVER_D}' must be strictly increasing")
    if np.any(r_axis < 0.0):
        raise ValueError(f"'{AXIS_R_OVER_SIGMA}' values must be >= 0")
    if np.any(s_axis <= 0.0):
        raise ValueError(f"'{AXIS_SIGMA_OVER_D}' values must be > 0")
    if w.shape != (r_axis.size, s_axis.size):
        raise ValueError(
            f"'{DATA_WEIGHT}' shape must match coordinates: "
            f"expected {(r_axis.size, s_axis.size)}, got {w.shape}"
        )


def evaluate_lookup_dataset(
    ds: xr.Dataset,
    r_over_sigma: np.ndarray,
    sigma_over_d: np.ndarray,
    *,
    bounds_policy: Literal["clip", "nan", "raise"] = "clip",
    fill_value: float | None = np.nan,
) -> np.ndarray:
    """
    Interpolate lookup weights at query points.

    Parameters
    ----------
    ds
        Lookup dataset produced by :func:`build_lookup_dataset`.
    r_over_sigma
        Query values for normalized wake-centre offsets ``R / sigma``.
    sigma_over_d
        Query values for normalized Gaussian widths ``sigma / D``.
    bounds_policy
        Out-of-bounds policy for lookup queries:

        - ``"clip"``: clip to axis limits before interpolation,
        - ``"nan"``: allow extrapolation requests to resolve to ``fill_value``,
        - ``"raise"``: raise ``ValueError`` when a query is out of bounds.
    fill_value
        Forwarded to ``RegularGridInterpolator``.

    Returns
    -------
    weights
        Interpolated weights with broadcasted query shape.
    """
    validate_lookup_dataset(ds)

    r_q, s_q = np.broadcast_arrays(
        np.asarray(r_over_sigma, dtype=float),
        np.asarray(sigma_over_d, dtype=float),
    )

    if np.any(~np.isfinite(r_q)) or np.any(~np.isfinite(s_q)):
        raise ValueError("Lookup queries must be finite")

    if bounds_policy not in {"clip", "nan", "raise"}:
        raise ValueError("bounds_policy must be one of {'clip', 'nan', 'raise'}")

    r_axis = np.asarray(ds.coords[AXIS_R_OVER_SIGMA].to_numpy(), dtype=float)
    s_axis = np.asarray(ds.coords[AXIS_SIGMA_OVER_D].to_numpy(), dtype=float)
    r_min, r_max = float(r_axis[0]), float(r_axis[-1])
    s_min, s_max = float(s_axis[0]), float(s_axis[-1])

    r_orig = r_q.copy()
    sigma_high = None
    if bounds_policy == "clip":
        sigma_high = s_q > s_max

    if bounds_policy == "raise":
        r_bad = (r_q < r_min) | (r_q > r_max)
        s_bad = (s_q < s_min) | (s_q > s_max)
        if np.any(r_bad) or np.any(s_bad):
            bad = r_bad | s_bad
            bad_i = np.flatnonzero(bad.ravel())[0]
            r_bad_v = float(r_q.ravel()[bad_i])
            s_bad_v = float(s_q.ravel()[bad_i])

            r_near_i = int(np.argmin(np.abs(r_axis - r_bad_v)))
            s_near_i = int(np.argmin(np.abs(s_axis - s_bad_v)))
            r_near_v = float(r_axis[r_near_i])
            s_near_v = float(s_axis[s_near_i])
            wtab = np.asarray(ds[DATA_WEIGHT].to_numpy(), dtype=float)
            w_near = float(wtab[r_near_i, s_near_i])

            raise ValueError(
                "Lookup query outside bounds: "
                f"r_over_sigma in [{r_min}, {r_max}], "
                f"sigma_over_d in [{s_min}, {s_max}]; "
                f"offending point=(r_over_sigma={r_bad_v}, sigma_over_d={s_bad_v}); "
                f"nearest table point=(r_over_sigma={r_near_v}, sigma_over_d={s_near_v}), "
                f"nearest_weight={w_near}"
            )
    if bounds_policy == "clip":
        r_q = np.clip(r_q, r_min, r_max)
        s_q = np.clip(s_q, s_min, s_max)

    interp = RegularGridInterpolator(
        (r_axis, s_axis),
        np.asarray(ds[DATA_WEIGHT].to_numpy(), dtype=float),
        method="linear",
        bounds_error=(bounds_policy == "raise"),
        fill_value=fill_value,
    )

    pts = np.column_stack([r_q.ravel(), s_q.ravel()])
    out = interp(pts).reshape(r_q.shape)
    if sigma_high is not None:
        out[sigma_high] = np.exp(-0.5 * r_orig[sigma_high] ** 2)
    return out


def evaluate_lookup_geometry(
    ds: xr.Dataset,
    r: np.ndarray,
    d: np.ndarray,
    sigma: np.ndarray,
    *,
    is_waked: np.ndarray | None = None,
    bounds_policy: Literal["clip", "nan", "raise"] = "clip",
    fill_value: float | None = np.nan,
    masked_value: float = 0.0,
    min_weight: float = 0.0,
    clip_check_min_weight: float | None = None,
) -> np.ndarray:
    """
    Evaluate lookup weights from geometric inputs with numeric guards.

    Parameters
    ----------
    ds
        Lookup dataset produced by :func:`build_lookup_dataset`.
    r
        Wake-centre radial offsets ``R`` in meters.
    d
        Rotor diameters ``D`` in meters.
    sigma
        Gaussian widths ``sigma`` in meters.
    is_waked
        Optional boolean mask. ``False`` entries are masked to ``masked_value``
        and are not validated as active waked points.
    bounds_policy
        Bounds behavior used by :func:`evaluate_lookup_dataset`.
    fill_value
        Fill value used by interpolation for ``bounds_policy='nan'``.
    masked_value
        Output value for masked (non-waked) entries.
    min_weight
        Weights strictly below this threshold are set to ``masked_value``.
    clip_check_min_weight
        Optional threshold for ``bounds_policy='clip'``: if any out-of-bounds
        query yields a clipped lookup weight greater than this threshold,
        ``ValueError`` is raised.

    Returns
    -------
    weights
        Interpolated rotor weights with broadcasted input shape.
    """
    r_arr, d_arr, s_arr = np.broadcast_arrays(
        np.asarray(r, dtype=float),
        np.asarray(d, dtype=float),
        np.asarray(sigma, dtype=float),
    )

    if is_waked is None:
        active = np.ones(r_arr.shape, dtype=bool)
    else:
        active = np.broadcast_to(np.asarray(is_waked, dtype=bool), r_arr.shape)

    finite = np.isfinite(r_arr) & np.isfinite(d_arr) & np.isfinite(s_arr)
    valid_geo = finite & (d_arr > 0.0) & (s_arr > 0.0)

    bad_active = active & (~valid_geo)
    if np.any(bad_active):
        raise ValueError(
            "Invalid geometry for waked points: require finite R, D > 0, sigma > 0"
        )

    out = np.full(r_arr.shape, masked_value, dtype=float)
    if not np.any(active):
        return out

    valid = active & valid_geo
    if not np.any(valid):
        return out

    r_over_sigma = np.abs(r_arr[valid]) / s_arr[valid]
    sigma_over_d = s_arr[valid] / d_arr[valid]

    oob = None
    if bounds_policy == "clip" and clip_check_min_weight is not None:
        r_axis = np.asarray(ds.coords[AXIS_R_OVER_SIGMA].to_numpy(), dtype=float)
        s_axis = np.asarray(ds.coords[AXIS_SIGMA_OVER_D].to_numpy(), dtype=float)
        r_oob = (r_over_sigma < float(r_axis[0])) | (r_over_sigma > float(r_axis[-1]))
        s_oob = sigma_over_d < float(s_axis[0])
        s_high = sigma_over_d > float(s_axis[-1])
        oob = (r_oob & ~s_high) | s_oob

    out[valid] = evaluate_lookup_dataset(
        ds,
        r_over_sigma=r_over_sigma,
        sigma_over_d=sigma_over_d,
        bounds_policy=bounds_policy,
        fill_value=fill_value,
    )
    if (
        bounds_policy == "clip"
        and clip_check_min_weight is not None
        and oob is not None
        and np.any(oob)
    ):
        oob_w = out[valid][oob]
        high = oob_w > clip_check_min_weight
        if np.any(high):
            idx = np.flatnonzero(oob)[np.flatnonzero(high)[0]]
            raise ValueError(
                "Clipped out-of-bounds lookup query has significant weight: "
                f"r_over_sigma={float(r_over_sigma[idx])}, "
                f"sigma_over_d={float(sigma_over_d[idx])}, "
                f"clipped_weight={float(out[valid][idx])}, "
                f"min_weight={float(clip_check_min_weight)}"
            )

    if min_weight > 0.0:
        low = out < min_weight
        if np.any(low):
            out[low] = masked_value
    return out
