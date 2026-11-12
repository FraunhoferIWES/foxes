"""Turbine curves collected from an explicitly configured wind-speed scan."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV

from .state_turbine_table import StateTurbineTable
from ._variable_info import get_data_info

if TYPE_CHECKING:
    from foxes.core import Algorithm


class TurbineTypeCurves(StateTurbineTable):
    """Plot selected turbine curves without launching hidden simulations.

    Configure wind-speed scan states, turbine types and optional power-mask
    comparison turbines on the executing no-wake algorithm.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        ws_var: str = FV.AMB_REWS,
        plot: bool = True,
        plot_pars: dict[str, Any] | None = None,
        to_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure collected fields and final curve rendering.

        Parameters
        ----------
        variables
            Curve fields; ``None`` selects power and thrust coefficient.
        ws_var
            Wind-speed abscissa in metres per second.
        plot
            Render curves instead of returning the selected dataset.
        plot_pars
            ``axs``, ``figsize``, ``titles``, ``x_label``, ``y_labels`` and
            remaining ``Axes.plot`` options. Each selected turbine is one curve.
        to_file
            Optional figure filename, skipped in numerical/``nofig`` mode.
        kwargs
            Table selections, figure ``write_pars``, paths and display options.
            Dimension renaming and transposition are not supported.
        """
        fields = [FV.P, FV.CT] if variables is None else list(variables)
        if not fields:
            raise ValueError("TurbineTypeCurves: variables must contain field names")
        super().__init__(variables=list(dict.fromkeys([ws_var, *fields])), **kwargs)
        if self.name_map or self.transpose:
            raise ValueError(
                "TurbineTypeCurves: name_map and transpose are not supported"
            )
        self.curve_variables = tuple(fields)
        self.ws_var = ws_var
        self.plot = plot
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)
        self.to_file = to_file

    def process(self, algo: Algorithm, data: Dataset) -> Any:
        """Render selected wind-speed curves or return their numerical dataset.

        Parameters
        ----------
        algo
            Algorithm supplying turbine names; never rerun or modified here.
        data
            Selected, chunk-collected curve fields.

        Returns
        -------
        result
            One-dimensional axes array, or the dataset in numerical/``nofig`` mode.

        Raises
        ------
        ValueError
            If the number of supplied axes does not match the curve fields.
        """
        table = self._table(data)
        if not self.plot or self.nofig:
            return table
        options = dict(self.plot_pars)
        axes = options.pop("axs", None)
        size = options.pop("figsize", (5 * len(self.curve_variables), 4))
        if axes is None:
            _, axes = plt.subplots(1, len(self.curve_variables), figsize=size)
        axes = np.atleast_1d(axes).reshape(-1)
        if len(axes) != len(self.curve_variables):
            raise ValueError(f"{self.name}: One axis is required for each curve field")
        titles = options.pop("titles", {})
        labels = options.pop("y_labels", {})
        xlabel = options.pop("x_label", "Wind speed [m/s]")
        for ax, variable in zip(axes, self.curve_variables):
            for index, turbine in enumerate(table[FC.TURBINE].to_numpy()):
                curve = table.sel({FC.TURBINE: turbine})
                order = np.argsort(curve[self.ws_var].to_numpy(), kind="stable")
                ax.plot(
                    curve[self.ws_var].to_numpy()[order],
                    curve[variable].to_numpy()[order],
                    label=algo.farm.turbine_names[int(turbine)],
                    linestyle=options.get(
                        "linestyle", ("-", "--", "-.", ":")[index % 4]
                    ),
                    **{
                        key: value
                        for key, value in options.items()
                        if key != "linestyle"
                    },
                )
            ax.set(
                title=titles.get(variable, variable),
                xlabel=xlabel,
                ylabel=labels.get(
                    variable,
                    get_data_info(variable, table[variable].attrs.get("units"))[1],
                ),
            )
            ax.grid(axis="y")
            ax.legend()
        if self.to_file is not None:
            axes[0].figure.savefig(self.get_fpath(self.to_file), **self.write_pars)
        self._show_figures()
        return axes
