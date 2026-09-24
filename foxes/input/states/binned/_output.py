"""Create and render support-point wind roses for binned states.

The functions in this module convert sparse Cartesian ``WD``/``WS`` bin
weights into a dense wind-rose dataset and render one polar plot per support
location. They are shared by regular-grid and scattered-support states.
"""

from typing import Any

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import plot_wind_rose_bars


def get_support_wind_roses_figure(
    data: xr.Dataset,
    *,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Any:
    """
    Create one wind rose per support point on a single canvas.

    Parameters
    ----------
    data
        Wind-rose dataset returned by :func:`wind_rose_dataset`.
    ncols
        Maximum number of polar axes per row.
    figsize
        Figure size in inches. If ``None``, derive it from the number of
        support points and columns.
    title
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one polar wind rose per support point.
    """
    import matplotlib.pyplot as plt

    support = np.asarray(data["support"])
    frequencies = data["frequency"].to_numpy()
    wd_edges = np.asarray(data.attrs[f"{FV.WD}_bounds"])
    n_points = support.shape[0]
    ncols = max(1, min(ncols, n_points))
    nrows = int(np.ceil(n_points / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        subplot_kw={"projection": "polar"},
    )
    for point_i, point in enumerate(support):
        ax = axes.flat[point_i]
        plot_wind_rose_bars(
            ax,
            frequencies[point_i],
            wd_edges,
            cmap="viridis",
        )
        ax.set_title(f"x={point[0]:.0f}, y={point[1]:.0f}, h={point[2]:.0f}")

    for ax in axes.flat[n_points:]:
        ax.set_visible(False)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def write_support_wind_roses(
    file_name: str,
    data: xr.Dataset,
    *,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> None:
    """
    Write a support-point wind-rose canvas to a file.

    Parameters
    ----------
    file_name
        Output image path accepted by :meth:`matplotlib.figure.Figure.savefig`.
    data
        Wind-rose dataset returned by :func:`wind_rose_dataset`.
    ncols
        Maximum number of polar axes per row.
    figsize
        Figure size in inches. If ``None``, derive it from the number of
        support points and columns.
    title
        Optional figure title.
    """
    import matplotlib.pyplot as plt

    fig = get_support_wind_roses_figure(
        data,
        ncols=ncols,
        figsize=figsize,
        title=title,
    )
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)


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
    xarray.Dataset
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
