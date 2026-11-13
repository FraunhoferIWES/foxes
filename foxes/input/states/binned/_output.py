"""Convert native binned-state weights into support-point frequencies.

Regular-grid and scattered-support states share this numerical conversion.
Rendering belongs to lifecycle-managed output models.
"""

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV


def wind_rose_dataset(
    support: np.ndarray,
    weights: np.ndarray,
    bin_vars: dict[str, np.ndarray],
    bin_shape: tuple[int, ...],
    bin_indices: np.ndarray,
) -> xr.Dataset:
    """
    Create support-point wind-rose frequencies from sparse bin weights.

    Parameters
    ----------
    support
        Support coordinates with shape ``(n_support, 3)`` and columns
        ``x``, ``y``, and ``height``.
    weights
        Weights for retained bins at every support point. The first dimension
        corresponds to ``bin_indices``; remaining dimensions flatten to
        ``n_support``. Values are interpreted as fractions and converted to
        percent without additional normalization.
    bin_vars
        Wind-speed and wind-direction bin edges, keyed by ``WS`` and ``WD``.
        Their insertion order defines the axes in ``bin_shape``.
    bin_shape
        Number of bins along each axis in ``bin_vars`` order.
    bin_indices
        Flat Cartesian-bin index for every retained weight row.

    Returns
    -------
    data
        Frequencies in percent with dimensions ``(point, WD, WS)``, explicit
        support coordinates, bin centers, and bin-edge attributes.

    Raises
    ------
    KeyError
        If bins contain variables other than exactly ``WS`` and ``WD``.
    """
    if set(bin_vars) != {FV.WS, FV.WD}:
        raise KeyError(
            f"Wind-rose output requires exactly '{FV.WS}' and '{FV.WD}' bins"
        )
    ws_axis = list(bin_vars).index(FV.WS)
    wd_axis = list(bin_vars).index(FV.WD)
    point_weights = np.zeros(
        (int(np.prod(bin_shape)), support.shape[0]),
        dtype=weights.dtype,
    )
    point_weights[bin_indices] = weights.reshape(len(bin_indices), -1)
    point_weights = point_weights.reshape(bin_shape + (support.shape[0],))
    point_weights = np.moveaxis(point_weights, (ws_axis, wd_axis), (0, 1))
    point_weights = np.transpose(point_weights, (2, 1, 0))
    wd_edges = bin_vars[FV.WD]
    ws_edges = bin_vars[FV.WS]
    return xr.Dataset(
        data_vars={
            "frequency": ((FC.POINT, FV.WD, FV.WS), 100.0 * point_weights),
        },
        coords={
            FC.POINT: np.arange(support.shape[0]),
            FC.XYH: (FC.XYH, np.asarray([FV.X, FV.Y, FV.H])),
            FV.WD: 0.5 * (wd_edges[:-1] + wd_edges[1:]),
            FV.WS: 0.5 * (ws_edges[:-1] + ws_edges[1:]),
            "support": ((FC.POINT, FC.XYH), support),
        },
        attrs={
            f"{FV.WD}_bounds": wd_edges,
            f"{FV.WS}_bounds": ws_edges,
        },
    )
