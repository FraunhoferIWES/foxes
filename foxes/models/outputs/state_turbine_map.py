"""Constructor-defined state/turbine heat maps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV

from .state_turbine_table import StateTurbineTable
from ._variable_info import get_data_info

if TYPE_CHECKING:
    from foxes.core import Algorithm


class StateTurbineMap(StateTurbineTable):
    """Render a selected field after its state/turbine chunks are assembled."""

    def __init__(
        self,
        variable: str = FV.P,
        plot: bool = True,
        plot_pars: dict[str, Any] | None = None,
        to_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure one heat map without capturing scientific results.

        Parameters
        ----------
        variable
            Farm field to collect, including output-selected evaluation fields.
        plot
            Render axes; otherwise return the selected numerical dataset.
        plot_pars
            ``ax``, ``figsize``, ``title``, ``cbar_label``, ``rotate_xlabels``
            and remaining ``Axes.pcolormesh`` options.
        to_file
            Optional figure filename. ``nofig`` and numerical mode skip rendering.
        kwargs
            Table selections, ``write_pars`` for ``Figure.savefig``, result naming,
            paths and display options. Dimension renaming is not supported.
        """
        super().__init__(variables=[variable], **kwargs)
        if self.name_map or self.transpose:
            raise ValueError(
                "StateTurbineMap: name_map and transpose are not supported"
            )
        self.variable = variable
        self.plot = plot
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)
        self.to_file = to_file

    def process(self, algo: Algorithm, data: Dataset) -> Axes | Dataset:
        """Render a heat map, or return its numerical table without figures.

        Parameters
        ----------
        algo
            Executing algorithm; no additional science is performed.
        data
            Selected farm fields assembled on the parent.

        Returns
        -------
        result
            Matplotlib axes, or the dataset when ``plot=False`` or ``nofig=True``.
            State labels, including singletons, datetimes and strings, are shown
            at cell centres without assuming arithmetic state coordinates.

        Raises
        ------
        ValueError
            If the selection is empty or the field is not state/turbine-shaped.
        """
        table = self._table(data)
        if not self.plot or self.nofig:
            return table
        field = table[self.variable]
        if set(field.dims) != {FC.STATE, FC.TURBINE} or not all(field.shape):
            raise ValueError(
                f"{self.name}: Map requires a nonempty state/turbine field"
            )
        options = dict(self.plot_pars)
        ax = options.pop("ax", None)
        figsize = options.pop("figsize", None)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        title = options.pop("title", self.variable)
        label = options.pop("cbar_label", get_data_info(self.variable)[1])
        rotation = options.pop("rotate_xlabels", 0)
        options.setdefault("shading", "flat")
        values = field.transpose(FC.STATE, FC.TURBINE).to_numpy()
        image = ax.pcolormesh(
            np.arange(values.shape[0] + 1),
            np.arange(values.shape[1] + 1),
            values.T,
            **options,
        )
        for axis, dimension, count in (
            ("x", FC.STATE, values.shape[0]),
            ("y", FC.TURBINE, values.shape[1]),
        ):
            indices = np.unique(np.linspace(0, count - 1, min(count, 10), dtype=int))
            getattr(ax, f"set_{axis}ticks")(
                indices + 0.5,
                table[dimension].to_numpy()[indices].astype(str),
                rotation=rotation if axis == "x" else 0,
            )
        ax.set(title=title, xlabel="State", ylabel="Turbine")
        ax.figure.colorbar(image, ax=ax, label=label)
        if self.to_file is not None:
            ax.figure.savefig(self.get_fpath(self.to_file), **self.write_pars)
        self._show_figures()
        return ax
